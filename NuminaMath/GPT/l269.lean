import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Geometry.Circle
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Pos
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Probability.ProbabilityMassFunction

namespace find_non_divisible_and_product_l269_269079

-- Define the set of numbers
def numbers : List Nat := [3543, 3552, 3567, 3579, 3581]

-- Function to get the digits of a number
def digits (n : Nat) : List Nat := n.digits 10

-- Function to sum the digits
def sum_of_digits (n : Nat) : Nat := (digits n).sum

-- Function to check divisibility by 3
def divisible_by_3 (n : Nat) : Bool := sum_of_digits n % 3 = 0

-- Find the units digit of a number
def units_digit (n : Nat) : Nat := n % 10

-- Find the tens digit of a number
def tens_digit (n : Nat) : Nat := (n / 10) % 10

-- The problem statement
theorem find_non_divisible_and_product :
  ∃ n ∈ numbers, ¬ divisible_by_3 n ∧ units_digit n * tens_digit n = 8 :=
by
  sorry

end find_non_divisible_and_product_l269_269079


namespace math_problem_correct_l269_269807

noncomputable def total_ways : ℕ := 
  Nat.choose 15 4 * Nat.choose 11 5 * Nat.choose 6 6

noncomputable def favorable_ways : ℕ := 
  1 + Nat.choose 11 1 + Nat.choose 11 2

noncomputable def probability_fraction : rat := 
  favorable_ways / total_ways

noncomputable def reduced_probability_fraction : rat := 
  rat.mk_pnat favorable_ways (nat.gcd favorable_ways total_ways)

def math_problem : Prop := 
  (reduced_probability_fraction.num + reduced_probability_fraction.denom = 630697)
-- The probability that all four mathematics textbooks end up in the same box is given
-- in terms of reduced fractions, to ensure the correct answer aligns with the result 630697.

theorem math_problem_correct : math_problem := by sorry

end math_problem_correct_l269_269807


namespace least_positive_difference_l269_269232

-- Definitions of sequences A and B
def seqA : List ℕ := [3, 9, 27, 81, 243]
def seqB : List ℕ := List.range' 10 10 |>.map (λ n => 10 * n)

-- Theorem statement
theorem least_positive_difference : (∃ a ∈ seqA, ∃ b ∈ seqB, abs (a - b) = 1) :=
by
  -- the proof would go here, but we'll skip it for now
  sorry

end least_positive_difference_l269_269232


namespace probability_of_7_successes_in_7_trials_l269_269900

open Probability

/-- Define the given conditions for the problem -/
def n : ℕ := 7
def k : ℕ := 7
def p : ℚ := 2 / 7

/-- The binomial coefficient and the probability of success in n trials -/
theorem probability_of_7_successes_in_7_trials :
  P(X = k) = (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) :=
by
  have bep_0 : nat.choose 7 7 = 1, from sorry,
  have p_power_k : p ^ k = (2 / 7) ^ 7, from sorry,
  have q_power_rem : (1 - p) ^ (n - k) = 1, from sorry,
  have p_eq_frac : (2 / 7) ^ 7 * 1 = 128 / 823543, from sorry,
  show 1 * (2 / 7) ^ 7 * 1 = 128 / 823543, by sorry

end probability_of_7_successes_in_7_trials_l269_269900


namespace domain_of_f_l269_269107

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.logBase 2 (2 * x - 1))

theorem domain_of_f :
  {x : ℝ | 0 ≤ Real.logBase 2 (2 * x - 1) ∧ 2 * x - 1 > 0} = {x : ℝ | 1 ≤ x} := 
sorry

end domain_of_f_l269_269107


namespace max_complete_bowling_games_l269_269972

theorem max_complete_bowling_games
  (shoes_cost locker_cost hotdog_cost drink_cost game_cost total_money : ℝ)
  (h_shoes : shoes_cost = 0.50)
  (h_locker : locker_cost = 3.00)
  (h_hotdog : hotdog_cost = 2.25)
  (h_drink : drink_cost = 1.50)
  (h_game : game_cost = 1.75)
  (h_total_money : total_money = 12.80) :
  let mandatory_expenses := shoes_cost + locker_cost + hotdog_cost + drink_cost in
  let remaining_money := total_money - mandatory_expenses in
  let max_games := int.ofNat (nat.floor (remaining_money / game_cost)) in
  max_games = 3 :=
by
  -- Proof would go here
  sorry

end max_complete_bowling_games_l269_269972


namespace abs_eq_two_l269_269672

theorem abs_eq_two (m : ℤ) (h : |m| = 2) : m = 2 ∨ m = -2 :=
sorry

end abs_eq_two_l269_269672


namespace total_time_is_three_hours_l269_269076

-- Define the conditions of the problem in Lean
def time_uber_house := 10
def time_uber_airport := 5 * time_uber_house
def time_check_bag := 15
def time_security := 3 * time_check_bag
def time_boarding := 20
def time_takeoff := 2 * time_boarding

-- Total time in minutes
def total_time_minutes := time_uber_house + time_uber_airport + time_check_bag + time_security + time_boarding + time_takeoff

-- Conversion from minutes to hours
def total_time_hours := total_time_minutes / 60

-- The theorem to prove
theorem total_time_is_three_hours : total_time_hours = 3 := by
  sorry

end total_time_is_three_hours_l269_269076


namespace telephone_charge_l269_269918

theorem telephone_charge (x : ℝ) (h1 : ∀ t : ℝ, t = 18.70 → x + 39 * 0.40 = t) : x = 3.10 :=
by
  sorry

end telephone_charge_l269_269918


namespace S_7_l269_269632

section

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Define the initial condition for the sequence
def a_1 : ℕ := 2

-- Define the recursive formula
def a_n1 (n : ℕ) : ℕ := 2 * S n

-- Define the sum of the first n terms
def sum_S (n : ℕ) : ℕ :=
if n = 1 then a 1 else sum (fun i => a i) (finset.range n)

-- Statement to prove
theorem S_7 : S 7 = 2186 :=
sorry

end

end S_7_l269_269632


namespace min_F_eq_T_l269_269743

variable (n : ℕ)

def F (σ : Finₓ n → ℕ) : ℕ :=
  ∑ i in Finₓ.range n, σ i * σ ((i + 1) % n)

def T (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    (1 / 6 * n ^ 3 + 1 / 2 * n ^ 2 + 5 / 6 * n - 1).toInt
  else
    (1 / 6 * n ^ 3 + 1 / 2 * n ^ 2 + 5 / 6 * n - 1 / 2).toInt

theorem min_F_eq_T (σ : Finₓ n → ℕ) (h : ∀ i, σ i ∈ Finₓ.range n) :
  (∃ σ, F σ = T n) :=
by
  sorry

end min_F_eq_T_l269_269743


namespace sweet_numbers_count_l269_269149

-- Define the conditions of the sequence
def tripleOrSubtract (n : ℕ) : ℕ :=
if n ≤ 30 then 3 * n else n - 15

-- Define the sequence function
def sequence (G : ℕ) : ℕ → ℕ
| 0     => G
| (n+1) => tripleOrSubtract (sequence n)

-- Define "sweet number" condition
def is_sweet_number (G : ℕ) : Prop :=
∀ n, sequence G n ≠ 18

-- Define the count of sweet numbers in the range 1 to 60
def count_sweet_numbers : ℕ :=
Nat.card {G : ℕ // 1 ≤ G ∧ G ≤ 60 ∧ is_sweet_number G}

-- The statement to prove
theorem sweet_numbers_count : count_sweet_numbers = 40 := by
  sorry

end sweet_numbers_count_l269_269149


namespace complex_conjugate_of_z_l269_269751

-- Define z based on the given expression
noncomputable def z : ℂ := (2 + complex.i) / (1 + complex.i^2 + complex.i^5)

-- State the theorem to prove the complex conjugate of z is 1 + 2i
theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * complex.i :=
sorry

end complex_conjugate_of_z_l269_269751


namespace number_of_digits_in_product_l269_269983

theorem number_of_digits_in_product :
  let x := 6 ^ 3
  let y := 7 ^ 6
  let product := x * y
  let num_digits := Int.floor (Real.log10 product) + 1
  num_digits = 8 :=
by
  sorry

end number_of_digits_in_product_l269_269983


namespace smallest_possible_sum_l269_269605

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l269_269605


namespace pencils_to_make_profit_l269_269951

theorem pencils_to_make_profit (total_pencils : ℕ) (cost_per_pencil : ℝ) (sell_price_per_pencil : ℝ) (desired_profit : ℝ) (total_cost : ℝ) (required_revenue : ℝ) (num_pencils_to_sell : ℕ) :
  total_pencils = 2000 →
  cost_per_pencil = 0.15 →
  sell_price_per_pencil = 0.30 →
  desired_profit = 180 →
  total_cost = total_pencils * cost_per_pencil →
  required_revenue = total_cost + desired_profit →
  num_pencils_to_sell = required_revenue / sell_price_per_pencil →
  num_pencils_to_sell = 1600 :=
by
  intros h_total_pencils h_cost_per_pencil h_sell_price_per_pencil h_desired_profit h_total_cost h_required_revenue h_num_pencils_to_sell
  rw [h_total_pencils, h_cost_per_pencil, h_sell_price_per_pencil, h_desired_profit] at h_total_cost h_required_revenue h_num_pencils_to_sell
  norm_num at h_total_cost h_required_revenue h_num_pencils_to_sell
  exact h_num_pencils_to_sell

end pencils_to_make_profit_l269_269951


namespace train_crossing_time_l269_269195

-- Definitions for the given conditions
def length_of_train : ℝ := 155
def speed_of_train_kmh : ℝ := 45
def length_of_bridge : ℝ := 220

-- Conversion factor from km/hr to m/s
def kmh_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

-- Total distance the train needs to travel
def total_distance : ℝ := length_of_train + length_of_bridge

-- Speed of the train in m/s
def speed_of_train_mps := kmh_to_mps speed_of_train_kmh

-- Time it takes for the train to cross the bridge
def crossing_time : ℝ := total_distance / speed_of_train_mps

-- The theorem to be proved
theorem train_crossing_time : crossing_time = 30 := 
by
  -- Proof to be added
  sorry

end train_crossing_time_l269_269195


namespace total_vehicles_is_120_l269_269688

def num_trucks : ℕ := 20
def num_tanks : ℕ := 5 * num_trucks
def total_vehicles : ℕ := num_tanks + num_trucks

theorem total_vehicles_is_120 : total_vehicles = 120 :=
by
  sorry

end total_vehicles_is_120_l269_269688


namespace smallest_sum_l269_269596

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l269_269596


namespace company_makes_profit_at_least_126_workers_l269_269925

noncomputable def least_number_of_workers (maintenance_fees worker_wage hours_per_day production_rate sale_price : ℝ) :=
  let daily_worker_wage := worker_wage * hours_per_day
  let revenue_per_worker_per_hour := production_rate * sale_price
  let revenue_per_worker_per_day := revenue_per_worker_per_hour * hours_per_day
  let profit_per_worker_per_day := revenue_per_worker_per_day - daily_worker_wage
  let workers_needed := maintenance_fees / profit_per_worker_per_day
  floor (workers_needed + 1)

theorem company_makes_profit_at_least_126_workers :
  least_number_of_workers 500 15 8 5 3.10 = 126 :=
by
  -- Proof skipped.
  sorry

end company_makes_profit_at_least_126_workers_l269_269925


namespace fraction_of_B_l269_269917

theorem fraction_of_B (A B C : ℝ) 
  (h1 : A = (1/3) * (B + C)) 
  (h2 : A = B + 20) 
  (h3 : A + B + C = 720) : 
  B / (A + C) = 2 / 7 :=
  by 
  sorry

end fraction_of_B_l269_269917


namespace tangent_slope_l269_269472

-- Define the points given in the problem
def center : ℝ × ℝ := (4, 6)
def point_of_tangency : ℝ × ℝ := (7, 3)

-- Define the problem statement using Lean 4
theorem tangent_slope (center point_of_tangency : ℝ × ℝ) (h₁ : center = (4, 6)) (h₂ : point_of_tangency = (7, 3)) :
  let slope_of_radius : ℝ := (point_of_tangency.2 - center.2) / (point_of_tangency.1 - center.1)
  let slope_of_tangent : ℝ := -1 / slope_of_radius
  slope_of_tangent = 1 := by {
  rw [h₁, h₂],
  sorry
}

end tangent_slope_l269_269472


namespace product_mod_division_l269_269470

theorem product_mod_division (a b c : ℕ) (h₁ : a = 98) (h₂ : b = 102) (h₃ : c = 8) :
  ((a * b) % c) = 4 :=
by
  rw [h₁, h₂, h₃]
  sorry

end product_mod_division_l269_269470


namespace circle_line_disjoint_l269_269297

-- Define the necessary variables and conditions
def radius_O : ℝ := 4
def distance_O_l : ℝ := 3 * Real.sqrt 2

-- Define the disjoint relationship between the line and the circle
theorem circle_line_disjoint {O : Type} (r : ℝ) (d : ℝ) 
  (h_r : r = radius_O)
  (h_d : d = distance_O_l) :
  r > d → disjoint (set_of (λ p : ℝ × ℝ, (p.1 - 0) ^ 2 + (p.2 - 0) ^ 2 = r^2))
                    (set_of (λ p : ℝ × ℝ, p.2 = - p.1 + d)) := 
sorry

end circle_line_disjoint_l269_269297


namespace dance_music_ratio_is_one_to_one_l269_269479

def ziggy_song_requests (total_requests electropop_requests rock_requests oldies_requests dj_choice_requests rap_requests dance_requests: ℕ) : Prop :=
  total_requests = 30 ∧
  electropop_requests = total_requests / 2 ∧
  rock_requests = 5 ∧
  oldies_requests = rock_requests - 3 ∧
  dj_choice_requests = oldies_requests / 2 ∧
  rap_requests = 2 ∧
  total_requests = 10 + dance_requests ∧
  electropop_requests = dance_requests

theorem dance_music_ratio_is_one_to_one :
  ∃ (total_requests electropop_requests rock_requests oldies_requests dj_choice_requests rap_requests dance_requests: ℕ),
  ziggy_song_requests total_requests electropop_requests rock_requests oldies_requests dj_choice_requests rap_requests dance_requests → 
  dance_requests / electropop_requests = 1 :=
begin
  simp,
  intros,
  sorry,
end

end dance_music_ratio_is_one_to_one_l269_269479


namespace prove_XY_squared_l269_269337

-- Define the rectangle with given conditions
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (AB BC AD : ℝ)
  (angleB : ℝ) -- angle at B

-- Specific rectangle ABCD as per problem conditions
def ABCDrct : Rectangle :=
{ 
  A := (0,0), B := (0,8), C := (15,8), D := (15,0), 
  AB := 15, BC := 8, AD := 8,
  angleB := 90
}

-- Define midpoints X and Y
def X : ℝ × ℝ := ((ABCD.ABCDrct.B.1 + ABCDrct.C.1) / 2, ABCDrct.B.2)
def Y : ℝ × ℝ := ((ABCD.ABCDrct.A.1 + ABCDrct.D.1) / 2, ABCDrct.A.2)

-- Prove XY² = 64
theorem prove_XY_squared :
  let XY_squared := (X.1 - Y.1)^2 + (X.2 - Y.2)^2 in
  XY_squared = 64 :=
  by
  sorry

end prove_XY_squared_l269_269337


namespace bianca_points_l269_269218

theorem bianca_points :
  ∀ (points_per_bag total_bags not_recycled_bags: ℕ),
    points_per_bag = 5 →
    total_bags = 17 →
    not_recycled_bags = 8 →
    (total_bags - not_recycled_bags) * points_per_bag = 45 :=
by
  intros points_per_bag total_bags not_recycled_bags 
  intros h_ppb h_tb h_nrb
  rw [h_ppb, h_tb, h_nrb]
  norm_num
  sorry

end bianca_points_l269_269218


namespace second_grade_girls_l269_269447

theorem second_grade_girls (G : ℕ) 
  (h1 : ∃ boys_2nd : ℕ, boys_2nd = 20)
  (h2 : ∃ students_3rd : ℕ, students_3rd = 2 * (20 + G))
  (h3 : 20 + G + (2 * (20 + G)) = 93) :
  G = 11 :=
by
  sorry

end second_grade_girls_l269_269447


namespace original_price_of_coat_l269_269441

theorem original_price_of_coat (P : ℝ) (h : P * 0.50 = 250) : P = 500 := 
begin
  -- exact proof omitted
  sorry
end

end original_price_of_coat_l269_269441


namespace interval_solution_l269_269740

-- Definition of set T of lattice points
def T : set (ℤ × ℤ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 40 ∧ 1 ≤ p.2 ∧ p.2 ≤ 40}

-- Given condition that exactly 400 points lie on or below the line y = mx
def points_below_line (m : ℚ) : ℕ :=
  {p ∈ T | p.2 ≤ m * p.1}.to_finset.card

-- Condition for the interval length of m
noncomputable def interval_length (c d : ℕ) : ℚ := c / d

-- Main theorem statement
theorem interval_solution :
  ∃ (c d : ℕ), rel_prime c d ∧ interval_length c d = 1 / 120 ∧ c + d = 121 :=
by
  sorry

end interval_solution_l269_269740


namespace rationalized_factor_of_3plusSqrt11_simplify_fraction_sqrt_a2_plus_b2_plus_2_l269_269087

-- First problem: Rationalized factor of (3 + √11) is (3 - √11)
theorem rationalized_factor_of_3plusSqrt11 : 
  let x := 3 + Real.sqrt 11 in
  ∃ y, y = 3 - Real.sqrt 11 ∧ ∀ z, (x * y = z) → ¬∃ w, w * w = z := 
sorry

-- Second problem: Simplify (1 - b) / (1 - √b) for b ≥ 0 ∧ b ≠ 1
theorem simplify_fraction (b : ℝ) (hb0 : b ≥ 0) (hb1 : b ≠ 1) :
  (1 - b) / (1 - Real.sqrt b) = 1 + Real.sqrt b := 
sorry

-- Third problem: Given a = 1 / (√3 - 2) and b = 1 / (√3 + 2), √(a^2 + b^2 + 2) = 4
theorem sqrt_a2_plus_b2_plus_2 (a b : ℝ)
  (ha : a = 1 / (Real.sqrt 3 - 2))
  (hb : b = 1 / (Real.sqrt 3 + 2)) :
  Real.sqrt (a ^ 2 + b ^ 2 + 2) = 4 := 
sorry

end rationalized_factor_of_3plusSqrt11_simplify_fraction_sqrt_a2_plus_b2_plus_2_l269_269087


namespace marble_groups_count_l269_269462

noncomputable def count_groups (red green blue yellow black : ℕ) : ℕ :=
  have h1 : red = 1 := rfl,
  have h2 : green = 1 := rfl,
  have h3 : blue = 1 := rfl,
  have h4 : yellow = 2 := rfl,
  have h5 : black = 2 := rfl,
  4 + 3 + 3

theorem marble_groups_count :
  count_groups 1 1 1 2 2 = 10 :=
  by sorry

end marble_groups_count_l269_269462


namespace child_ticket_cost_l269_269811

theorem child_ticket_cost :
  ∃ x : ℤ, (9 * 11 = 7 * x + 50) ∧ x = 7 :=
by
  sorry

end child_ticket_cost_l269_269811


namespace Ivan_cannot_win_l269_269718

theorem Ivan_cannot_win :
  ∀ (f : ℝ → ℝ) (n : ℕ) (a : ℝ), 
    f = (λ x, x - 1) → 
    (∀ f : ℝ → ℝ, ∃ a : ℝ, a ∈ (set_of (λ x, f x = 0)) → 
      (λ x, a * x^(n+1) - f (-x) - 2) → 
      (∀ x, (λ x, a * x^(n+1) - f (-x) - 2) x = 0) →
      ∃ x : ℝ, (λ x, a * x^(n+1) - f (-x) - 2) x = 0) → 
    (∀ (f : ℝ → ℝ) (n : ℕ) (a : ℝ), Ivan_cannot_win) := sorry

end Ivan_cannot_win_l269_269718


namespace eccentricity_of_hyperbola_is_e_l269_269304

-- Definitions and given conditions
variable (a b c : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_hyperbola : ∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1)
variable (h_left_focus : ∀ F : ℝ × ℝ, F = (-c, 0))
variable (h_circle : ∀ E : ℝ × ℝ, E.1^2 + E.2^2 = a^2)
variable (h_parabola : ∀ P : ℝ × ℝ, P.2^2 = 4*c*P.1)
variable (h_midpoint : ∀ E P F : ℝ × ℝ, E = (F.1 + P.1) / 2 ∧ E.2 = (F.2 + P.2) / 2)

-- The statement to be proved
theorem eccentricity_of_hyperbola_is_e :
    ∃ e : ℝ, e = (Real.sqrt 5 + 1) / 2 :=
sorry

end eccentricity_of_hyperbola_is_e_l269_269304


namespace sum_of_possible_amounts_l269_269069

-- Definitions based on conditions:
def possible_quarters_amounts : Finset ℕ := {5, 30, 55, 80}
def possible_dimes_amounts : Finset ℕ := {15, 20, 30, 35, 40, 50, 60, 70, 80, 90}
def both_possible_amounts : Finset ℕ := possible_quarters_amounts ∩ possible_dimes_amounts

-- Statement of the problem:
theorem sum_of_possible_amounts : (both_possible_amounts.sum id) = 110 :=
by
  sorry

end sum_of_possible_amounts_l269_269069


namespace Jackie_exercise_hours_l269_269028

variable (work_hours : ℕ) (sleep_hours : ℕ) (free_time_hours : ℕ) (total_hours_in_day : ℕ)
variable (time_for_exercise : ℕ)

noncomputable def prove_hours_exercising (work_hours sleep_hours free_time_hours total_hours_in_day : ℕ) : Prop :=
  work_hours = 8 ∧
  sleep_hours = 8 ∧
  free_time_hours = 5 ∧
  total_hours_in_day = 24 → 
  time_for_exercise = total_hours_in_day - (work_hours + sleep_hours + free_time_hours)

theorem Jackie_exercise_hours :
  prove_hours_exercising 8 8 5 24 3 :=
by
  -- Proof is omitted as per instruction
  sorry

end Jackie_exercise_hours_l269_269028


namespace length_B1C1_l269_269713

variable (AC BC : ℝ) (A1B1 : ℝ) (T : ℝ)

/-- Given a right triangle ABC with legs AC = 3 and BC = 4, and transformations
  of points to A1, B1, and C1 where A1B1 = 1 and angle B1 = 90 degrees,
  prove that the length of B1C1 is 12. -/
theorem length_B1C1 (h1 : AC = 3) (h2 : BC = 4) (h3 : A1B1 = 1) 
  (TABC : T = 6) (right_triangle_ABC : true) (right_triangle_A1B1C1 : true) : 
  B1C1 = 12 := 
sorry

end length_B1C1_l269_269713


namespace center_of_symmetry_l269_269928

-- Define the symmetry conditions
def is_symmetric_about_x_axis (figure : set (ℝ × ℝ)) : Prop :=
  ∀ {x y : ℝ}, (x, y) ∈ figure → (x, -y) ∈ figure

def is_symmetric_about_y_axis (figure : set (ℝ × ℝ)) : Prop :=
  ∀ {x y : ℝ}, (x, y) ∈ figure → (-x, y) ∈ figure

-- The main theorem stating that a figure with two perpendicular axes of symmetry has a center of symmetry
theorem center_of_symmetry
  (figure : set (ℝ × ℝ))
  (h_x : is_symmetric_about_x_axis figure)
  (h_y : is_symmetric_about_y_axis figure) :
  ∀ {x y : ℝ}, (x, y) ∈ figure → (-x, -y) ∈ figure :=
by
  sorry

end center_of_symmetry_l269_269928


namespace wire_cut_problem_l269_269205

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end wire_cut_problem_l269_269205


namespace complex_conjugate_of_z_l269_269778

theorem complex_conjugate_of_z :
  let z := (2 + complex.i) / (1 + complex.i ^ 2 + complex.i ^ 5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269778


namespace convert_square_decimeters_to_square_meters_convert_cubic_meters_to_cubic_decimeters_l269_269913

theorem convert_square_decimeters_to_square_meters :
  ∀ (d: ℝ), d = 30 → d / 100 = 0.3 :=
by
  intro d hd
  rw hd
  norm_num

theorem convert_cubic_meters_to_cubic_decimeters :
  ∀ (c: ℝ), c = 3.05 → c * 1000 = 3050 :=
by
  intro c hc
  rw hc
  norm_num

end convert_square_decimeters_to_square_meters_convert_cubic_meters_to_cubic_decimeters_l269_269913


namespace frank_spent_per_week_l269_269269

theorem frank_spent_per_week (mowing_dollars : ℕ) (weed_eating_dollars : ℕ) (weeks : ℕ) 
    (total_dollars := mowing_dollars + weed_eating_dollars) 
    (spending_rate := total_dollars / weeks) :
    mowing_dollars = 5 → weed_eating_dollars = 58 → weeks = 9 → spending_rate = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end frank_spent_per_week_l269_269269


namespace sandra_beignets_16_weeks_l269_269404

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end sandra_beignets_16_weeks_l269_269404


namespace rice_pounds_l269_269090

noncomputable def pounds_of_rice (r p : ℝ) : Prop :=
  r + p = 30 ∧ 1.10 * r + 0.55 * p = 23.50

theorem rice_pounds (r p : ℝ) (h : pounds_of_rice r p) : r = 12.7 :=
sorry

end rice_pounds_l269_269090


namespace inequality_am_gm_l269_269370

theorem inequality_am_gm (a b : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : 0 < b) (h₃ : b < 1) :
  1 + a + b > 3 * Real.sqrt (a * b) :=
by
  sorry

end inequality_am_gm_l269_269370


namespace unique_solution_condition_l269_269101

theorem unique_solution_condition {a b : ℝ} : (∃ x : ℝ, 4 * x - 7 + a = b * x + 4) ↔ b ≠ 4 :=
by
  sorry

end unique_solution_condition_l269_269101


namespace find_c_l269_269856

theorem find_c (c : ℝ) :
  (∀ x y : ℝ, 2*x^2 - 4*c*x*y + (2*c^2 + 1)*y^2 - 2*x - 6*y + 9 ≥ 0) ↔ c = 1/6 :=
by
  sorry

end find_c_l269_269856


namespace find_f_2017_l269_269677

def f : ℕ → ℕ := sorry

axiom h1 : ∀ n, f(f(n)) + f(n) = 2 * n + 3
axiom h2 : f(0) = 1

theorem find_f_2017 : f(2017) = 2018 :=
by
  sorry

end find_f_2017_l269_269677


namespace zeros_of_f_l269_269128

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then x^2 - 2*x - 3 else -2 + Real.log x

theorem zeros_of_f : ∃ x : ℝ, f x = 0 :=
begin
  use -1,
  use Real.exp 2,
  sorry
end

end zeros_of_f_l269_269128


namespace marbles_total_l269_269007

theorem marbles_total (r b g y : ℝ) 
  (h1 : r = 1.30 * b)
  (h2 : g = 1.50 * r)
  (h3 : y = 0.80 * g) :
  r + b + g + y = 4.4692 * r :=
by
  sorry

end marbles_total_l269_269007


namespace inverse_equilateral_l269_269112

-- Definitions
def is_equilateral (T : Triangle) : Prop :=
  T.side1 = T.side2 ∧ T.side2 = T.side3

def has_equal_angles (T : Triangle) : Prop :=
  T.angle1 = T.angle2 ∧ T.angle2 = T.angle3

-- Theorem statement
theorem inverse_equilateral (T : Triangle) :
  has_equal_angles T → is_equilateral T :=
sorry

end inverse_equilateral_l269_269112


namespace wire_cut_problem_l269_269203

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end wire_cut_problem_l269_269203


namespace problem_solution_l269_269591

def SequenceGeometric (a_n : ℕ → ℤ) : Prop :=
  ∃ (a : ℤ) (r : ℤ) (a ≠ 0), ∀ n, (a_n (n + 1) = r * a_n n)

def b_n_value (a_n : ℕ → ℤ) (n : ℕ) : ℚ :=
  Real.log (List.prod (List.map a_n (List.range n)))

def sum_reciprocals (b_n : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, 1 / b_n i)

theorem problem_solution (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (h1 : a_n 1 = 2)
  (h2 : S_n = λ n, List.sum (List.map a_n (List.range n)))
  (h3 : ∀ n : ℕ, a_n (n + 1) = 2 * a_n n)
  (h4 : a_n 5 = 32)
  (bn := λ n, Real.log (List.prod (List.map a_n (List.range n))))
  (sum_reciprocals := (Finset.range 6).sum (λ i, 1 / bn i)) :
  (2 * 5 / (5 + 1)) := sorry

end problem_solution_l269_269591


namespace distinct_digit_addition_values_l269_269013

theorem distinct_digit_addition_values (A B C D E : ℕ) :
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    D = 0 ∧ 
    A + C = 10 ∧ 
    2 * B = E ∧ 
    (A > 0 ∧ A < 10) ∧ (B > 0 ∧ B < 10) ∧ (C > 0 ∧ C < 10) ∧ (E > 0 ∧ E < 10)
    → ∃! E, E = 2 ∨ E = 4 ∨ E = 6 ∨ E = 8 :=
begin
  sorry,
end

end distinct_digit_addition_values_l269_269013


namespace derivative_at_pi_over_4_l269_269796

def f (x : ℝ) : ℝ := Real.cos x + 2 * Real.sin x

theorem derivative_at_pi_over_4 : (Real.deriv f) (Real.pi / 4) = Real.sqrt 2 / 2 := sorry

end derivative_at_pi_over_4_l269_269796


namespace effect_on_revenue_l269_269885

variable (P N : ℝ)

def P_new := 0.80 * P
def N_new := 1.60 * N
def R := P * N
def R_new := P_new * N_new

theorem effect_on_revenue :
  R_new = 1.28 * R :=
by
  unfold R R_new P_new N_new
  sorry

end effect_on_revenue_l269_269885


namespace sum_of_first_89_natural_numbers_l269_269221

theorem sum_of_first_89_natural_numbers : (∑ k in Finset.range 90, k) = 4005 :=
by
  sorry

end sum_of_first_89_natural_numbers_l269_269221


namespace impossible_to_form_square_l269_269135

-- Define the lengths of the sticks and their counts
def stick_lengths : List (ℕ × ℕ) := [(1, 6), (2, 3), (3, 6), (4, 5)]

-- Function to compute the total length of the sticks
def total_length (sticks : List (ℕ × ℕ)) : ℕ :=
  sticks.foldr (λ (p : ℕ × ℕ) acc, p.1 * p.2 + acc) 0

-- Total length of the sticks given
def L := total_length stick_lengths

-- Side length of the square, assumed from L being divisible by 4
def side_length (L : ℕ) (h : L % 4 = 0) : ℕ := L / 4

-- Proof statement: we need to show it's impossible to form a square given the stick lengths.
theorem impossible_to_form_square : L % 4 ≠ 0 :=
by
  have h1 : total_length stick_lengths = 50 := by decide
  rw [←h1] -- Show that L is indeed what we calculated
  exact dec_trivial -- Prove that 50 % 4 ≠ 0

end impossible_to_form_square_l269_269135


namespace tom_remaining_trip_speed_l269_269860

theorem tom_remaining_trip_speed :
  ∀ (total_distance first_distance first_speed average_speed : ℝ),
    total_distance = 80 ∧
    first_distance = 30 ∧
    first_speed = 30 ∧
    average_speed = 40 →
    let remaining_distance := total_distance - first_distance in
    let total_time := total_distance / average_speed in
    let first_time := first_distance / first_speed in
    let remaining_time := total_time - first_time in
    let remaining_speed := remaining_distance / remaining_time in
    remaining_speed = 50 :=
by
  sorry

end tom_remaining_trip_speed_l269_269860


namespace relationship_among_a_b_c_l269_269623

-- Define the given conditions as constants in Lean 4
def a : ℝ := (1 / 2) * real.cos (2 * real.pi / 180) - (real.sqrt 3 / 2) * real.sin (2 * real.pi / 180)
def b : ℝ := (2 * real.tan (14 * real.pi / 180)) / (1 - real.tan (14 * real.pi / 180) ^ 2)
def c : ℝ := real.sqrt ((1 - real.cos (50 * real.pi / 180)) / 2)

-- State the theorem to be proved
theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l269_269623


namespace heracles_age_is_10_l269_269978

variable (H : ℕ)

-- Conditions
def audrey_age_now : ℕ := H + 7
def audrey_age_in_3_years : ℕ := audrey_age_now + 3
def heracles_twice_age : ℕ := 2 * H

-- Proof Statement
theorem heracles_age_is_10 (h1 : audrey_age_in_3_years = heracles_twice_age) : H = 10 :=
by 
  sorry

end heracles_age_is_10_l269_269978


namespace compute_expression_modulo_l269_269747

/--
  Let \( m \) be the integer such that \( 0 \le m < 41 \) and \( 5m \equiv 1 \pmod{41} \).
  Then \( (3^m)^2 - 3 \equiv 6 \pmod{41} \).
-/
theorem compute_expression_modulo (m : ℤ) (hm1 : 0 ≤ m) (hm2 : m < 41)
  (hmod : 5 * m ≡ 1 [ZMOD 41]) : (3 ^ m) ^ 2 - 3 ≡ 6 [ZMOD 41] :=
by
  sorry

end compute_expression_modulo_l269_269747


namespace part_b_part_c_l269_269483

variables {x : ℕ → ℝ}

-- Condition: a recursive definition of the sequence
def sequence_condition (x : ℕ → ℝ) := ∀ n ∈ {1, 2, ..., 24}, x n * x (n + 2) = x (n + 1)

theorem part_b (h: ∀ n ∈ {1, 2, ..., 24}, x n * x (n + 2) = x (n + 1)) (a b : ℝ) (h₀: a ≠ 0) (h₁: b ≠ 0) :
  x 7 = a := 
sorry

theorem part_c (h: ∀ n ∈ {1, 2, ..., 24}, x n * x (n + 2) = x (n + 1)) (a b : ℝ) (h₀: a ≠ 0) (h₁: b ≠ 0) (h₂: a * b = 2010) :
  (Finset.range 26).prod x = 2010 := 
sorry

end part_b_part_c_l269_269483


namespace television_dimensions_l269_269551

noncomputable def L := 4 * real.sqrt (1296 / 25)
noncomputable def H := 3 * real.sqrt (1296 / 25)
noncomputable def A := L * H

theorem television_dimensions (L = 28.8 ∧ A = 622.08) : Prop := 
begin
    sorry
end

end television_dimensions_l269_269551


namespace maclaurin_series_1_plus_x_pow_m_maclaurin_series_exp_maclaurin_series_sin_maclaurin_series_cos_maclaurin_series_ln_one_plus_x_l269_269558

-- Statement for Maclaurin series of (1+x)^m
theorem maclaurin_series_1_plus_x_pow_m (m : ℝ) :
  ∀ (x : ℝ), ((1 + x)^m : ℝ) = ∑' (n : ℕ), (m.descFactorial n / n.factorial) * x^n := sorry

-- Statement for Maclaurin series of e^x
theorem maclaurin_series_exp :
  ∀ (x : ℝ), (exp x : ℝ) = ∑' (n : ℕ), (x^n / n.factorial) := sorry

-- Statement for Maclaurin series of sin x
theorem maclaurin_series_sin :
  ∀ (x : ℝ), (Real.sin x : ℝ) = ∑' (n : ℕ), ((-1)^n * x^(2 * n + 1) / (2 * n + 1).factorial) := sorry

-- Statement for Maclaurin series of cos x
theorem maclaurin_series_cos :
  ∀ (x : ℝ), (Real.cos x : ℝ) = ∑' (n : ℕ), ((-1)^n * x^(2 * n) / (2 * n).factorial) := sorry

-- Statement for Maclaurin series of ln(1 + x)
theorem maclaurin_series_ln_one_plus_x :
  ∀ (x : ℝ), abs x < 1 → (Real.log (1 + x) : ℝ) = ∑' (n : ℕ), ((-1)^(n + 1) * x^(n + 1) / (n + 1)) := sorry

end maclaurin_series_1_plus_x_pow_m_maclaurin_series_exp_maclaurin_series_sin_maclaurin_series_cos_maclaurin_series_ln_one_plus_x_l269_269558


namespace total_selling_price_l269_269514

theorem total_selling_price (cost1 cost2 cost3 : ℕ) (profit1 profit2 profit3 : ℚ) 
  (h1 : cost1 = 280) (h2 : cost2 = 350) (h3 : cost3 = 500) 
  (h4 : profit1 = 30) (h5 : profit2 = 45) (h6 : profit3 = 25) : 
  (cost1 + (profit1 / 100) * cost1) + (cost2 + (profit2 / 100) * cost2) + (cost3 + (profit3 / 100) * cost3) = 1496.5 := by
  sorry

end total_selling_price_l269_269514


namespace profit_percent_is_35_l269_269143

variables (P C : ℝ)
-- Condition
def condition (P C : ℝ) : Prop := (2 / 3) * P = 0.9 * C

-- Question (What profit percent is made by selling at price P?)
def profit_percent (P C : ℝ) : ℝ := ((P - C) / C) * 100

-- Theorem statement
theorem profit_percent_is_35 (h: condition P C) : profit_percent P C = 35 :=
by
  sorry

end profit_percent_is_35_l269_269143


namespace initial_balls_count_l269_269915

variables (y w : ℕ)

theorem initial_balls_count (h1 : y = 2 * (w - 10)) (h2 : w - 10 = 5 * (y - 9)) :
  y = 10 ∧ w = 15 :=
sorry

end initial_balls_count_l269_269915


namespace triangle_to_rectangle_l269_269336

noncomputable def rectangle_ratio (AD AB x : ℝ) (P Q R S E : Point) (AP PQ QB DE EC PQS_area rectangle_area : ℝ): Prop :=
  ∃ (x : ℝ) (AP PQ QB DE EC : ℝ) (P Q E R S : Point),
  AP = x ∧
  PQ = x ∧
  QB = x ∧
  DE = x ∧
  EC = x ∧
  AB = 3 * AD ∧
  PQS_area = x^2 / 2 ∧
  rectangle_area = 3 * x^2 ∧
  PQS_area / rectangle_area = 1 / 6

theorem triangle_to_rectangle (AD AB x : ℝ) (P Q R S E : Point) (AP PQ QB DE EC PQS_area rectangle_area : ℝ)
  (h_AP : AP = x) (h_PQ : PQ = x) (h_QB : QB = x) (h_DE : DE = x) (h_EC : EC = x) 
  (h_AB : AB = 3 * AD) (h_PQS_area : PQS_area = x^2 / 2) (h_rectangle_area : rectangle_area = 3 * x^2) : 
  rectangle_ratio AD AB x P Q R S E AP PQ QB DE EC PQS_area rectangle_area :=
begin
  unfold rectangle_ratio,
  use x,
  exact ⟨h_AP, h_PQ, h_QB, h_DE, h_EC, h_AB, h_PQS_area, h_rectangle_area⟩,
  sorry
end

end triangle_to_rectangle_l269_269336


namespace kelly_grade_correct_l269_269033

variable (Jenny Jason Bob Kelly : ℕ)

def jenny_grade : ℕ := 95
def jason_grade := jenny_grade - 25
def bob_grade := jason_grade / 2
def kelly_grade := bob_grade + (bob_grade / 5)  -- 20% of Bob's grade is (Bob's grade * 0.20), which is the same as (Bob's grade / 5)

theorem kelly_grade_correct : kelly_grade = 42 :=
by
  sorry

end kelly_grade_correct_l269_269033


namespace percentage_return_on_investment_l269_269924

theorem percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ)
  (dividend_per_share : ℝ := (dividend_rate / 100) * face_value)
  (percentage_return : ℝ := (dividend_per_share / purchase_price) * 100)
  (h1 : dividend_rate = 15.5)
  (h2 : face_value = 50)
  (h3 : purchase_price = 31) :
  percentage_return = 25 := by
    sorry

end percentage_return_on_investment_l269_269924


namespace ratio_of_areas_l269_269010

theorem ratio_of_areas (ABCD : Type) [Parallelogram ABCD] (A B C D : Points ABCD)
  (α : ℝ) (h_angle : ∠BAD = α)
  (O1 O2 O3 O4 : ℝ) 
  (H1 : CircleCenteredAt O1 (Triangle DAB))
  (H2 : CircleCenteredAt O2 (Triangle DAC))
  (H3 : CircleCenteredAt O3 (Triangle DBC))
  (H4 : CircleCenteredAt O4 (Triangle ABC)) :
  RatioOfAreas (Quadrilateral O1 O2 O3 O4) (Parallelogram ABCD) = cot α ^ 2 := 
sorry

end ratio_of_areas_l269_269010


namespace solution_set_inequality_l269_269797

variable {f : ℝ → ℝ}

theorem solution_set_inequality (h_diff : ∀ x < 0, differentiable_at ℝ f x)
  (h_inequality : ∀ x < 0, 3 * f x + x * (deriv f x) > 0)
  (h_neg3: f (-3) > 0) :
  { x : ℝ | (x + 2015)^3 * f (x + 2015) + 27 * f (-3) > 0 } = set.Ioo (-2018) (-2015) := 
by
  sorry

end solution_set_inequality_l269_269797


namespace find_x_plus_y_l269_269291

theorem find_x_plus_y
  (x y : ℝ)
  (h1 : x + Real.cos y = 2010)
  (h2 : x + 2010 * Real.sin y = 2009)
  (h3 : (π / 2) ≤ y ∧ y ≤ π) :
  x + y = 2011 + π :=
sorry

end find_x_plus_y_l269_269291


namespace complex_conjugate_l269_269773

theorem complex_conjugate (z : ℂ) : 
  (i : ℂ)² = -1 → (i : ℂ)⁵ = i → 
  z = (2 + i) / (1 + (-1) + i) → 
  conj z = 1 + 2 * i :=
by
  intros
  sorry

end complex_conjugate_l269_269773


namespace range_of_a_l269_269309

open Set Real

noncomputable theory

def A := { x : ℝ | abs (x - 2) < 1 }
def B (a : ℝ) := { y : ℝ | ∃ x : ℝ, y = -x^2 + a }

theorem range_of_a (a : ℝ) : A ⊆ B a ↔ a ≥ 3 :=
by
  sorry

end range_of_a_l269_269309


namespace conjugate_z_is_1_add_2i_l269_269759

open Complex

def z_def : ℂ := ((2 : ℂ) + I) / (1 + I^2 + I^5)

theorem conjugate_z_is_1_add_2i : conj z_def = (1 : ℂ) + 2 * I := 
sorry

end conjugate_z_is_1_add_2i_l269_269759


namespace circle_equation_passing_through_P_and_centered_at_intersection_l269_269262

theorem circle_equation_passing_through_P_and_centered_at_intersection :
  ∃ (h k r : ℝ), (h = 0) ∧ (k = 1) ∧ (r = sqrt 2) ∧ (1 - h)^2 + (0 - k)^2 = r^2 ∧
  ∀ (x y : ℝ), (x + y - 1 = 0) ∧ (x - y + 1 = 0) → (x - h)^2 + (y - k)^2 = r^2 → x^2 + (y - 1)^2 = 2 :=
by
  sorry

end circle_equation_passing_through_P_and_centered_at_intersection_l269_269262


namespace integral_sqrt_circle_l269_269984

def f (x : ℝ) : ℝ := real.sqrt (9 - x^2)

theorem integral_sqrt_circle :
  ∫ x in -3..0, f x = (9 * real.pi) / 4 :=
by
  sorry

end integral_sqrt_circle_l269_269984


namespace sum_values_sqrt_eq_nine_l269_269141

theorem sum_values_sqrt_eq_nine (x : ℝ) :
  (∃ x : ℝ, (√((x - 5) ^ 2) = 9)) → (∑ x in ({x | √((x - 5) ^ 2) = 9} : set ℝ), x) = 10 :=
sorry

end sum_values_sqrt_eq_nine_l269_269141


namespace base_conversion_unique_b_l269_269422

theorem base_conversion_unique_b (b : ℕ) (h_b_pos : 0 < b) :
  (1 * 5^2 + 3 * 5^1 + 2 * 5^0) = (2 * b^2 + b) → b = 4 :=
by
  sorry

end base_conversion_unique_b_l269_269422


namespace dot_product_uv_l269_269570

def u : ℝ × ℝ × ℝ := (4, -3, -5)
def v : ℝ × ℝ × ℝ := (-6, 3, 2)

theorem dot_product_uv : (4 * -6 + -3 * 3 + -5 * 2) = -43 :=
by
  dsimp [u, v]
  sorry

end dot_product_uv_l269_269570


namespace slope_and_inclination_of_line_l269_269471

noncomputable section

open Real

theorem slope_and_inclination_of_line :
  (∀ x y : ℝ, x + y - 2 = 0 → slope = -1) ∧
  (∀ θ : ℝ, tan θ = -1 → θ = 3 * π / 4) :=
by
  sorry

end slope_and_inclination_of_line_l269_269471


namespace prob_blue_section_damaged_all_days_l269_269906

noncomputable def prob_of_7_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_blue_section_damaged_all_days :
  prob_of_7_successes 7 7 (2 / 7) = 128 / 823543 :=
by sorry

end prob_blue_section_damaged_all_days_l269_269906


namespace more_red_balls_l269_269725

theorem more_red_balls (red_packs yellow_packs pack_size : ℕ) (h1 : red_packs = 5) (h2 : yellow_packs = 4) (h3 : pack_size = 18) :
  (red_packs * pack_size) - (yellow_packs * pack_size) = 18 :=
by
  sorry

end more_red_balls_l269_269725


namespace min_value_expression_l269_269271

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a + 2) * (1 / b + 2) ≥ 16 :=
sorry

end min_value_expression_l269_269271


namespace num_roots_of_f_l269_269658

def f (x : ℝ) : ℝ := |x| - 1

noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x := x
| (n+1), x := f (f_n n x)

theorem num_roots_of_f :
  ∃ num_roots : ℕ, num_roots = 20 ∧ ∀ x : ℝ, f_n 10 x + 0.5 = 0 → True :=
sorry

end num_roots_of_f_l269_269658


namespace solve_linear_system_l269_269414

theorem solve_linear_system :
  ∃ (x1 x2 x3 : ℚ), 
  (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧ 
  (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧ 
  (5 * x1 + 5 * x2 - 7 * x3 = 27) ∧
  (x1 = 19 / 3 + x3) ∧ 
  (x2 = -14 / 15 + 2 / 5 * x3) := 
by 
  sorry

end solve_linear_system_l269_269414


namespace min_new_hires_needed_l269_269965

theorem min_new_hires_needed
  (W A L : Set α)
  (hW : Fintype.card W = 95)
  (hA : Fintype.card A = 80)
  (hL : Fintype.card L = 50)
  (hWA : Fintype.card (W ∩ A) = 30)
  (hAL : Fintype.card (A ∩ L) = 20)
  (hWL : Fintype.card (W ∩ L) = 15)
  (hWAL : Fintype.card (W ∩ A ∩ L) = 10) :
  Fintype.card (W ∪ A ∪ L) = 170 := 
by
  sorry

end min_new_hires_needed_l269_269965


namespace smallest_number_is_16_l269_269857

theorem smallest_number_is_16 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c) / 3 = 24 ∧ 
  (b = 25) ∧ (c = b + 6) ∧ min a (min b c) = 16 :=
by
  sorry

end smallest_number_is_16_l269_269857


namespace problem1_problem2_problem3_l269_269088

theorem problem1 (a : ℝ) : |a + 2| = 4 → (a = 2 ∨ a = -6) :=
sorry

theorem problem2 (a : ℝ) (h₀ : -4 < a) (h₁ : a < 2) : |a + 4| + |a - 2| = 6 :=
sorry

theorem problem3 (a : ℝ) : ∃ x ∈ Set.Icc (-2 : ℝ) 1, |x-1| + |x+2| = 3 :=
sorry

end problem1_problem2_problem3_l269_269088


namespace compute_d1e1_d2e2_d3e3_l269_269053

-- Given polynomials and conditions
variables {R : Type*} [CommRing R]

noncomputable def P (x : R) : R :=
  x^7 - x^6 + x^4 - x^3 + x^2 - x + 1

noncomputable def Q (x : R) (d1 d2 d3 e1 e2 e3 : R) : R :=
  (x^2 + d1 * x + e1) * (x^2 + d2 * x + e2) * (x^2 + d3 * x + e3)

-- Given conditions
theorem compute_d1e1_d2e2_d3e3 
  (d1 d2 d3 e1 e2 e3 : R)
  (h : ∀ x : R, P x = Q x d1 d2 d3 e1 e2 e3) : 
  d1 * e1 + d2 * e2 + d3 * e3 = -1 :=
by
  sorry

end compute_d1e1_d2e2_d3e3_l269_269053


namespace problem1_correct_problem2_correct_problem3_correct_problem4_correct_l269_269268

-- Define the conditions and the problems
def problem1 (students : List String) (A_positions : Finset ℕ) : ℕ :=
  -- Assuming position is indexed from 0 to 6
  if 3 ∈ A_positions then (students.erase "A").perms.length else 0

def problem2 (students : List String) (female_group_positions : Finset (List ℕ)) : ℕ :=
  -- Assuming female students are {"F1", "F2", "F3"}
  let female_perm_count := ["F1", "F2", "F3"].perms.length
  if  female_group_positions.nonempty then 
    female_perm_count * (students.length - 2).perms.length
  else 0

def problem3 (students : List String) (female_positions : Finset (List ℕ)) : ℕ :=
  if female_positions.card = 3 then 
    ["M1", "M2", "M3", "M4"].perms.length * (Finset.range 5).permutations.filter 
      (λ s, s.toList.nodup).length 
  else 0

def problem4 (students : List String) (positions_apart3 : Finset (ℕ × ℕ)) : ℕ :=
  if (∃ s, (s.fst - s.snd).abs = 3 ∨ (s.snd - s.fst).abs = 3) then 
    [("M1", "M2"), ("M2", "M3")].perms.length *
    (students.length - 2).perms.length * (students.length - 4).perms.length 
  else 0

-- The main theorem statements
theorem problem1_correct : problem1 ["A", "M1", "M2", "M3", "F1", "F2", "F3"] {3} = 720 :=
by sorry

theorem problem2_correct : problem2 ["A", "M1", "M2", "M3", "F1", "F2", "F3"] {List.range 3, [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]} = 720 := 
by sorry

theorem problem3_correct : problem3 ["A", "M1", "M2", "M3", "F1", "F2", "F3"] {List.range 3, [0, 2, 4]} = 1440 :=
by sorry

theorem problem4_correct : problem4 ["A", "M1", "M2", "M3", "F1", "F2", "F3"] {(0, 3), (1, 4), (2, 5), (3, 6)} = 720 := 
by sorry

end problem1_correct_problem2_correct_problem3_correct_problem4_correct_l269_269268


namespace distance_between_points_l269_269541

theorem distance_between_points :
  let p1 := (3, -5)
  let p2 := (-4, 4)
  dist p1 p2 = Real.sqrt 130 := by
  sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

end distance_between_points_l269_269541


namespace max_sum_cubes_l269_269052

theorem max_sum_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  a^3 + b^3 + c^3 + d^3 ≤ 8 :=
sorry

end max_sum_cubes_l269_269052


namespace calorie_intake_in_week_l269_269332

-/
If you wish to live to be 100 years old (hypothetically), you must consume 500 calories less than your average daily allowance for your age.
If you are in your 60's, and your average daily allowance is 2000 calories per day, show that you are allowed 10500 calories in a week.
-/

theorem calorie_intake_in_week
  (avg_daily_allowance : ℕ)
  (reduction : ℕ)
  (days_in_week : ℕ)
  (calories_per_day : ℕ)
  (calories_per_week : ℕ) : 
  avg_daily_allowance = 2000 → 
  reduction = 500 →
  days_in_week = 7 →
  calories_per_day = avg_daily_allowance - reduction →
  calories_per_week = calories_per_day * days_in_week →
  calories_per_week = 10500 := 
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry


end calorie_intake_in_week_l269_269332


namespace john_total_points_l269_269039

theorem john_total_points (shots_2pts : ℕ) (shots_3pts : ℕ) (interval_mins : ℕ) (periods : ℕ) (period_duration : ℕ) : ℕ :=
  let points_per_interval := (2 * shots_2pts + 3 * shots_3pts)
  let total_mins := periods * period_duration
  let intervals := total_mins / interval_mins
  intervals * points_per_interval

example : john_total_points 2 1 4 2 12 = 42 := by 
  simp [john_total_points]
  sorry

end john_total_points_l269_269039


namespace calculation_l269_269544

def mixedToFraction (n : Int) (num : Int) (den : Int) : Rational :=
  Rational.ofInt(n * den + num) / Rational.ofInt den

noncomputable def expression : Rational :=
  let a := mixedToFraction 7 4480 8333
  let b := Rational.ofInt 21934 / Rational.ofInt 25909
  let c := mixedToFraction 1 18556 35255
  a / b / c

theorem calculation: expression = Rational.ofInt 35 / Rational.ofInt 6 := 
  by
    sorry

end calculation_l269_269544


namespace triangle_inequality_sqrt_l269_269792

theorem triangle_inequality_sqrt (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  sqrt (a + b - c) + sqrt (b + c - a) + sqrt (c + a - b) ≤ sqrt a + sqrt b + sqrt c :=
  sorry

end triangle_inequality_sqrt_l269_269792


namespace minimum_value_of_f_l269_269118

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then -x else x^2

theorem minimum_value_of_f : ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y ≥ 0 :=
by {
  use 0,
  split,
  { simp [f] },
  { intro y,
    simp [f],
    split_ifs ;
    linarith }
}

end minimum_value_of_f_l269_269118


namespace min_value_a_sq_plus_b_sq_l269_269620

theorem min_value_a_sq_plus_b_sq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a - 1)^3 + (b - 1)^3 ≥ 3 * (2 - a - b)) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x > 0 → y > 0 → (x - 1)^3 + (y - 1)^3 ≥ 3 * (2 - x - y) → x^2 + y^2 ≥ m) :=
by
  sorry

end min_value_a_sq_plus_b_sq_l269_269620


namespace problem_statement_l269_269049

def S (b : ℕ) : ℕ :=
  (finset.finset.filter
    (λ (a : fin (9) → ℕ),
     3 * b - 1 = finset.sum (finset.fin_range 9) (λ i, a i) ∧
     b^2 + 1 = finset.sum (finset.fin_range 9) (λ i, (a i)^2))
    (finset.finset.univ : finset (fin (9) → ℕ))).card

theorem problem_statement (ε : ℝ) (hε : ε > 0) : ∃ (Cε : ℝ), ∀ (b : ℕ), S b ≤ (Cε * (b^(3 + ε))) :=
by
  sorry

end problem_statement_l269_269049


namespace triangle_angle_equality_l269_269959

noncomputable def incenter (A B C : Type) [triangle A B C] : Type := sorry
noncomputable def point (A B : Type) : Type := sorry
noncomputable def parallel (A B C : Type) : Prop := sorry
noncomputable def angle (A B C : Type) : Type := sorry

theorem triangle_angle_equality (A B C P F I: Type)
  [triangle A B C] [incenter I A B C]
  (hA : angle A B C = 60) 
  (hP : point P (segment B C)) (hP_ratio : 3 * segment_length P B = segment_length B C) 
  (hF : point F (segment A B)) (hParallel : parallel (segment I F) (segment A C)) :
  angle B F P = angle F B I :=
sorry

end triangle_angle_equality_l269_269959


namespace simplify_expression_l269_269831

variable (x : ℝ)

theorem simplify_expression :
  (2 * x + 25) + (150 * x + 35) + (50 * x + 10) = 202 * x + 70 :=
sorry

end simplify_expression_l269_269831


namespace product_of_distinct_elements_of_T_l269_269739

def T := {n : ℕ | n > 0 ∧ n ∣ 72000}

theorem product_of_distinct_elements_of_T :
  (∃! n, ∃ a b : T, a ≠ b ∧ n = a * b) ∧ T.count = 378 :=
by 
  sorry

end product_of_distinct_elements_of_T_l269_269739


namespace number_of_people_in_each_van_l269_269501

theorem number_of_people_in_each_van (x : ℕ) 
  (h1 : 6 * x + 8 * 18 = 180) : x = 6 :=
by sorry

end number_of_people_in_each_van_l269_269501


namespace Jessica_victory_l269_269313

def bullseye_points : ℕ := 10
def other_possible_scores : Set ℕ := {0, 2, 5, 8, 10}
def minimum_score_per_shot : ℕ := 2
def shots_taken : ℕ := 40
def remaining_shots : ℕ := 40
def jessica_advantage : ℕ := 30

def victory_condition (n : ℕ) : Prop :=
  8 * n + 80 > 370

theorem Jessica_victory :
  ∃ n, victory_condition n ∧ n = 37 :=
by
  use 37
  sorry

end Jessica_victory_l269_269313


namespace volume_difference_l269_269190

noncomputable def sphere_radius : ℝ := 7
noncomputable def cylinder_radius : ℝ := 4

theorem volume_difference (V : ℝ) : (V = (1372/3) - 32 * Real.sqrt 33) →
  let r := cylinder_radius in
  let R := sphere_radius in
  let h := 2 * Real.sqrt (R^2 - r^2) in
  let V_sphere := (4/3) * Real.pi * R^3 in
  let V_cylinder := Real.pi * r^2 * h in
  V * Real.pi = V_sphere - V_cylinder :=
sorry

end volume_difference_l269_269190


namespace solution_max_value_l269_269123

noncomputable def sequence (α β λ : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  if n = 1 then λ else
  (α + β)^n * (Finset.sum (Finset.range (n + 1)) (λ k, (α^(n - k) * β^k * λ^(n - k) / Real.factorial (n - k) * λ^k / Real.factorial k))) / (α + β)^n

theorem solution (α β λ : ℝ) (n : ℕ) (hn : 0 < n):
  sequence α β λ n = λ^n / n! :=
by sorry

theorem max_value (λ : ℝ) (n : ℕ) :
  sequence 1 1 λ n ≤ sequence 1 1 λ ⌊λ⌋ :=
by sorry

end solution_max_value_l269_269123


namespace smallest_sum_l269_269960

theorem smallest_sum (nums : Finset ℕ) (h : nums = (Finset.range 1998).map (λ x, x + 1)) :
 ∃ a, a = 1 ∧ (∀ b, b ∈ (Finset.attach (nums.image (λ n, (if (even n) then 10 else 1) + n))) -> a ≤ b) :=
begin
  sorry
end

end smallest_sum_l269_269960


namespace largest_percentage_difference_l269_269418

-- Define the sales data for each month
def trumpet_sales : List ℕ := [5, 6, 6, 4, 3]
def clarinet_sales : List ℕ := [4, 4, 6, 5, 5]

-- Define a function to calculate the percentage difference
def percentage_difference (T C : ℕ) : ℚ :=
  (Int.toRat (Nat.max T C - Nat.min T C) / Int.toRat (Nat.min T C)) * 100

-- Define the conditions for the sales in each month
def January_T := 5
def January_C := 4
def February_T := 6
def February_C := 4
def March_T := 6
def March_C := 6
def April_T := 4
def April_C := 5
def May_T := 3
def May_C := 5

-- Define the calculated percentage differences
def percent_diff_Jan := percentage_difference January_T January_C
def percent_diff_Feb := percentage_difference February_T February_C
def percent_diff_Mar := percentage_difference March_T March_C
def percent_diff_Apr := percentage_difference April_T April_C
def percent_diff_May := percentage_difference May_T May_C

-- Prove the month with the largest percentage difference is May
theorem largest_percentage_difference :
  percent_diff_May > percent_diff_Jan ∧
  percent_diff_May > percent_diff_Feb ∧
  percent_diff_May > percent_diff_Mar ∧
  percent_diff_May > percent_diff_Apr :=
by 
  sorry


end largest_percentage_difference_l269_269418


namespace planes_parallel_l269_269329

variables (α β : Type)
variables (n : ℝ → ℝ → ℝ → Prop) (u v : ℝ × ℝ × ℝ)

-- Conditions: 
def normal_vector_plane_alpha (u : ℝ × ℝ × ℝ) := u = (1, 2, -1)
def normal_vector_plane_beta (v : ℝ × ℝ × ℝ) := v = (-3, -6, 3)

-- Proof Problem: Prove that alpha is parallel to beta
theorem planes_parallel (h1 : normal_vector_plane_alpha u)
                        (h2 : normal_vector_plane_beta v) :
  v = -3 • u :=
by sorry

end planes_parallel_l269_269329


namespace problem_solution_l269_269621

variable (α β : ℝ)

-- Conditions
variable (h1 : 3 * Real.sin α - Real.cos α = 0)
variable (h2 : 7 * Real.sin β + Real.cos β = 0)
variable (h3 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)

theorem problem_solution : 2 * α - β = - (3 * π / 4) := by
  sorry

end problem_solution_l269_269621


namespace complex_conjugate_l269_269774

theorem complex_conjugate (z : ℂ) : 
  (i : ℂ)² = -1 → (i : ℂ)⁵ = i → 
  z = (2 + i) / (1 + (-1) + i) → 
  conj z = 1 + 2 * i :=
by
  intros
  sorry

end complex_conjugate_l269_269774


namespace total_cans_l269_269179

theorem total_cans (c o : ℕ) (h1 : c = 8) (h2 : o = 2 * c) : c + o = 24 := by
  sorry

end total_cans_l269_269179


namespace percentage_of_delivery_fee_l269_269357

theorem percentage_of_delivery_fee :
  ∀ (cost_per_toy_set cost_per_chair total_paid : ℕ) 
    (num_toy_sets num_chairs : ℕ)
    (pre_delivery_cost delivery_fee : ℕ),
  cost_per_toy_set = 78 →
  num_toy_sets = 3 →
  cost_per_chair = 83 →
  num_chairs = 2 →
  total_paid = 420 →
  pre_delivery_cost = (cost_per_toy_set * num_toy_sets) + (cost_per_chair * num_chairs) →
  delivery_fee = total_paid - pre_delivery_cost →
  (delivery_fee * 100 / pre_delivery_cost = 5) := 
begin
  intros cost_per_toy_set cost_per_chair total_paid num_toy_sets num_chairs pre_delivery_cost delivery_fee,
  intros h1 h2 h3 h4 h5 h6 h7,
  rw [h1, h2, h3, h4, h5, h6, h7],
  norm_num,
end

end percentage_of_delivery_fee_l269_269357


namespace problem_f_2016_l269_269637

def f : ℕ → ℝ
| x := if x ≤ 1 then real.log 2 (5 - x) else f (x - 1) + 1

theorem problem_f_2016 : f 2016 = 2017 := 
by sorry

end problem_f_2016_l269_269637


namespace infinite_inscribing_process_l269_269193

noncomputable def ratio_of_areas (C S : ℝ) : ℝ := C / S

theorem infinite_inscribing_process :
  let C := ∑' n : ℕ, π * (r n) ^ 2,
    S := ∑' n : ℕ, 2 * (r n) ^ 2,
    r : ℕ → ℝ := λn, (sqrt 2) ^ (-n) / (sqrt π) 
  in ratio_of_areas C S = π / 2 :=
by
  let C := ∑' n : ℕ, π * (r n) ^ 2
  let S := ∑' n : ℕ, 2 * (r n) ^ 2
  let r : ℕ → ℝ := λn, (sqrt 2) ^ (-n) / (sqrt π)
  have h : C = ∑' n : ℕ, π * (r n) ^ 2 := rfl
  have h2 : S = ∑' n : ℕ, 2 * (r n) ^ 2 := rfl
  sorry

end infinite_inscribing_process_l269_269193


namespace recurring_decimal_to_fraction_l269_269249

theorem recurring_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 1.\overline{06} = 35 / 33 :=
sorry

end recurring_decimal_to_fraction_l269_269249


namespace problem1_problem2_l269_269263

section
  -- Definitions
  def log5 (x : ℝ) := Real.log x / Real.log 5

  -- Expressions
  def E1 := 2 * log5 10 + log5 0.25
  def E2 := (8 / 125) ^ (-1 / 3) - (-3 / 5) ^ 0 + 16 ^ 0.75

  -- Proof statements
  theorem problem1 : E1 = 2 := by
    sorry

  theorem problem2 : E2 = 19 / 2 := by
    sorry
end

end problem1_problem2_l269_269263


namespace blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l269_269250

variables (length magnitude : ℕ)
variable (price : ℝ)
variable (area : ℕ)

-- Definitions based on the conditions
def length_is_about_4 (length : ℕ) : Prop := length = 4
def price_is_about_9_50 (price : ℝ) : Prop := price = 9.50
def large_area_is_about_3 (area : ℕ) : Prop := area = 3
def small_area_is_about_1 (area : ℕ) : Prop := area = 1

-- Proof problem statements
theorem blackboard_length_is_meters : length_is_about_4 length → length = 4 := by sorry
theorem pencil_case_price_is_yuan : price_is_about_9_50 price → price = 9.50 := by sorry
theorem campus_area_is_hectares : large_area_is_about_3 area → area = 3 := by sorry
theorem fingernail_area_is_square_centimeters : small_area_is_about_1 area → area = 1 := by sorry

end blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l269_269250


namespace gcd_18_30_l269_269140

-- Given two integers x and y
def a : ℤ := 18
def b : ℤ := 30

-- Define the greatest common divisor function
def gcd (x y : ℤ) : ℤ := Nat.gcd x.natAbs y.natAbs

-- Problem statement: Prove that the GCD of a and b is 6
theorem gcd_18_30 : gcd a b = 6 := 
by
  sorry -- the proof is omitted

end gcd_18_30_l269_269140


namespace sum_of_three_consecutive_even_nums_l269_269125

theorem sum_of_three_consecutive_even_nums : 80 + 82 + 84 = 246 := by
  sorry

end sum_of_three_consecutive_even_nums_l269_269125


namespace conjugate_z_is_1_add_2i_l269_269756

open Complex

def z_def : ℂ := ((2 : ℂ) + I) / (1 + I^2 + I^5)

theorem conjugate_z_is_1_add_2i : conj z_def = (1 : ℂ) + 2 * I := 
sorry

end conjugate_z_is_1_add_2i_l269_269756


namespace sequence_2000th_term_mod_4_l269_269561

noncomputable def sequence : ℕ → ℕ
| 0       => 1 -- For convenience in indexing
| (n + 1) => sum_upto_k_inverse (n + 1) -- Cumulative sum function

def sum_upto_k_inverse (k : ℕ) : ℕ :=
-- Find the largest integer n such that n(n+1)/2 >= k due to the nature of sequence
let sum := (λ n, n * (n + 1) / 2) in
if h : ∃ (n : ℕ), sum n >= k then classical.some h else 0

def find_2000th_term : ℕ := sum_upto_k_inverse 2000

def remainder_when_divided_by_4 (n : ℕ) : ℕ :=
n % 4

theorem sequence_2000th_term_mod_4 : remainder_when_divided_by_4 find_2000th_term = 3 := by
  sorry

end sequence_2000th_term_mod_4_l269_269561


namespace min_value_S_max_value_S_l269_269066

theorem min_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≥ -512 := 
sorry

theorem max_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288 := 
sorry

end min_value_S_max_value_S_l269_269066


namespace probability_bulb_lit_l269_269912

theorem probability_bulb_lit (bulb_count : ℕ := 100) : 
  let perfect_squares := 10 in
  (perfect_squares : ℚ) / bulb_count = 0.1 := by
  sorry

end probability_bulb_lit_l269_269912


namespace second_parallel_line_length_l269_269423

theorem second_parallel_line_length (base : ℝ) (line_count : ℕ) (triangle_area : ℝ) (second_parallel_line_length_correct: line_count = 3) (base_length : base = 18) (area_division : triangle_area = base * 9) : (second_parallel_line_length) = 9 * Real.sqrt 2 :=
by
  sorry

end second_parallel_line_length_l269_269423


namespace perp_AH_BP_l269_269624

open EuclideanGeometry

variables (A B C M H P : Point)
variable [metric_space Point]
variable [normed_add_torsor Point]

/-- Let A, B, C be points in a Euclidean geometry where AB = BC, M is the midpoint of AC,
    H is on BC such that MH is perpendicular to BC, and P is the midpoint of MH.
    Prove AH is perpendicular to BP. -/
theorem perp_AH_BP
  (h1 : dist A B = dist B C)
  (h2 : midpoint ℝ A C = M)
  (h3 : M ⬝ H ⊥ B ⬝ C)
  (h4 : midpoint ℝ M H = P) :
  A ⬝ H ⊥ B ⬝ P :=
sorry

end perp_AH_BP_l269_269624


namespace tables_left_l269_269521

theorem tables_left (original_tables number_of_customers_per_table current_customers : ℝ) 
(h1 : original_tables = 44.0)
(h2 : number_of_customers_per_table = 8.0)
(h3 : current_customers = 256) : 
(original_tables - current_customers / number_of_customers_per_table) = 12.0 :=
by
  sorry

end tables_left_l269_269521


namespace area_ratio_of_circles_l269_269327

theorem area_ratio_of_circles (R_A R_B : ℝ) 
  (h1 : (60 / 360) * (2 * Real.pi * R_A) = (40 / 360) * (2 * Real.pi * R_B)) :
  (Real.pi * R_A ^ 2) / (Real.pi * R_B ^ 2) = 9 / 4 := 
sorry

end area_ratio_of_circles_l269_269327


namespace tom_age_ratio_l269_269463

-- Definitions of given conditions
variables (T N : ℕ) -- Tom's age (T) and number of years ago (N)

-- Tom's age is T years
-- The sum of the ages of Tom's three children is also T
-- N years ago, Tom's age was twice the sum of his children's ages then

theorem tom_age_ratio (h1 : T - N = 2 * (T - 3 * N)) : T / N = 5 :=
sorry

end tom_age_ratio_l269_269463


namespace fruits_in_box_l269_269092

theorem fruits_in_box (initial_persimmons : ℕ) (added_apples : ℕ) (total_fruits : ℕ) :
  initial_persimmons = 2 → added_apples = 7 → total_fruits = initial_persimmons + added_apples → total_fruits = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end fruits_in_box_l269_269092


namespace value_of_t_l269_269322

theorem value_of_t :
  let t := 2 / (2 - real.cbrt 3)
  t = (2 * (2 + real.cbrt 3) * (4 + real.cbrt 9)) / 10 :=
sorry

end value_of_t_l269_269322


namespace compute_100p_plus_q_l269_269058

-- Conditions stated as definitions
def equation1_has_3_distinct_roots (p q : ℝ) : Prop :=
  let f1 := (x : ℝ) => ((x + p) * (x + q) * (x - 15)) / ((x - 5)^2)
  finset.card (finset.filter (fun r => f1 r = 0) finset.univ) = 3

def equation2_has_2_distinct_roots (p q : ℝ) : Prop :=
  let f2 := (x : ℝ) => ((x - 2 * p) * (x - 5) * (x + 10)) / ((x + q) * (x - 15))
  finset.card (finset.filter (fun r => f2 r = 0) finset.univ) = 2

-- The main theorem to prove
theorem compute_100p_plus_q (p q : ℝ) (h1 : equation1_has_3_distinct_roots p q) 
  (h2 : equation2_has_2_distinct_roots p q) : 100 * p + q = 240 := 
  sorry

end compute_100p_plus_q_l269_269058


namespace complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l269_269363

-- Definitions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1)}

-- Part (Ⅰ)
theorem complement_A_union_B_m_eq_4 :
  (m = 4) → compl (A ∪ B 4) = {x | x < -2} ∪ {x | x > 7} := 
by
  sorry

-- Part (Ⅱ)
theorem B_nonempty_and_subset_A_range_m :
  (∃ x, x ∈ B m) ∧ (B m ⊆ A) → 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l269_269363


namespace number_of_apples_l269_269130

theorem number_of_apples (n : ℕ) 
  (g1 g2 g3 : ℕ) 
  (h1 : g1 = 19) 
  (h2 : g2 = 22) 
  (h3 : g3 = 23) 
  (h4 : abs (g1 - n) = 1 ∨ abs (g2 - n) = 1 ∨ abs (g3 - n) = 1) 
  (h5 : abs (g1 - n) = 2 ∨ abs (g2 - n) = 2 ∨ abs (g3 - n) = 2) 
  (h6 : abs (g1 - n) = 3 ∨ abs (g2 - n) = 3 ∨ abs (g3 - n) = 3) :
  n = 20 :=
by
  sorry

end number_of_apples_l269_269130


namespace avg_age_increase_l269_269334

-- Define the initial conditions
variables (num_students : ℕ) (avg_age_students teacher_age : ℕ)
variables (h_num_students : num_students = 20) (h_avg_age_students : avg_age_students = 21) (h_teacher_age : teacher_age = 42)

-- Define the mathematical assertion we want to prove
theorem avg_age_increase :
  let total_age_students := num_students * avg_age_students in
  let total_age_with_teacher := total_age_students + teacher_age in
  let new_avg_age := total_age_with_teacher / (num_students + 1) in
  new_avg_age - avg_age_students = 1 := 
by
  rw [h_num_students, h_avg_age_students, h_teacher_age]
  sorry

end avg_age_increase_l269_269334


namespace complex_conjugate_l269_269770

theorem complex_conjugate (z : ℂ) : 
  (i : ℂ)² = -1 → (i : ℂ)⁵ = i → 
  z = (2 + i) / (1 + (-1) + i) → 
  conj z = 1 + 2 * i :=
by
  intros
  sorry

end complex_conjugate_l269_269770


namespace player_A_wins_if_n_equals_9_l269_269171

-- Define the conditions
def drawing_game (n : ℕ) : Prop :=
  ∃ strategy : ℕ → ℕ,
    strategy 0 = 1 ∧ -- Player A always starts by drawing 1 ball
    (∀ k, 1 ≤ strategy k ∧ strategy k ≤ 3) ∧ -- Players draw between 1 and 3 balls
    ∀ b, 1 ≤ b → b ≤ 3 → (n - 1 - strategy (b - 1)) ≤ 3 → (strategy (n - 1 - (b - 1)) = n - (b - 1) - 1)

-- State the problem to prove Player A has a winning strategy if n = 9
theorem player_A_wins_if_n_equals_9 : drawing_game 9 :=
sorry

end player_A_wins_if_n_equals_9_l269_269171


namespace probability_of_7_successes_in_7_trials_l269_269899

open Probability

/-- Define the given conditions for the problem -/
def n : ℕ := 7
def k : ℕ := 7
def p : ℚ := 2 / 7

/-- The binomial coefficient and the probability of success in n trials -/
theorem probability_of_7_successes_in_7_trials :
  P(X = k) = (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) :=
by
  have bep_0 : nat.choose 7 7 = 1, from sorry,
  have p_power_k : p ^ k = (2 / 7) ^ 7, from sorry,
  have q_power_rem : (1 - p) ^ (n - k) = 1, from sorry,
  have p_eq_frac : (2 / 7) ^ 7 * 1 = 128 / 823543, from sorry,
  show 1 * (2 / 7) ^ 7 * 1 = 128 / 823543, by sorry

end probability_of_7_successes_in_7_trials_l269_269899


namespace triangle_vertices_l269_269157

theorem triangle_vertices : 
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ x - y = -4 ∧ x = 2 / 3 ∧ y = 14 / 3) ∧ 
  (∃ (x y : ℚ), x - y = -4 ∧ y = -1 ∧ x = -5) ∧
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ y = -1 ∧ x = 7 / 2) :=
by
  sorry

end triangle_vertices_l269_269157


namespace problem1_problem2_problem3_l269_269587

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^2 + 0.5 * x

theorem problem1 (h : ∀ x : ℝ, f (x + 1) = f x + x + 1) (h0 : f 0 = 0) : 
  ∀ x : ℝ, f x = 0.5 * x^2 + 0.5 * x := by 
  sorry

noncomputable def g (t : ℝ) : ℝ :=
  if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
  else if -1.5 < t ∧ t < -0.5 then -1 / 8
  else 0.5 * t^2 + 0.5 * t

theorem problem2 (h : ∀ t : ℝ, g t = min (f (t)) (f (t + 1))) : 
  ∀ t : ℝ, g t = 
    if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
    else if -1.5 < t ∧ t < -0.5 then -1 / 8
    else 0.5 * t^2 + 0.5 * t := by 
  sorry

theorem problem3 (m : ℝ) : (∀ t : ℝ, g t + m ≥ 0) → m ≥ 1 / 8 := by 
  sorry

end problem1_problem2_problem3_l269_269587


namespace function_machine_output_l269_269704

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 25 then step1 - 7 else step1 + 10
  step2

theorem function_machine_output : function_machine 15 = 38 :=
by
  sorry

end function_machine_output_l269_269704


namespace exists_natural_numbers_x_y_l269_269266

theorem exists_natural_numbers_x_y (a : ℕ) : a = 4 → ∃ (x y : ℕ), (x + y)^2 + 3*x + y = 2*a :=
begin
  intro ha,
  rw ha,
  use 1,
  use 1,
  sorry
end

end exists_natural_numbers_x_y_l269_269266


namespace smallest_n_for_win_probability_gt_half_l269_269574

def probability_townspeople_win (n : ℕ) : ℚ :=
if n = 1
then 1 / 3
else 1 / (2 * n + 1) + (2 * n) / (2 * n + 1) * probability_townspeople_win (n - 1)

theorem smallest_n_for_win_probability_gt_half :
  ∃ n : ℕ, (n > 0) ∧ (probability_townspeople_win n > 1 / 2) ∧ (∀ m : ℕ, (m > 0) ∧ (m < n) → probability_townspeople_win m ≤ 1 / 2) :=
by
  sorry

end smallest_n_for_win_probability_gt_half_l269_269574


namespace max_pieces_four_cuts_l269_269950

theorem max_pieces_four_cuts (n : ℕ) (h : n = 4) : (by sorry : ℕ) = 14 := 
by sorry

end max_pieces_four_cuts_l269_269950


namespace magnitude_a_plus_b_angle_between_m_and_n_l269_269653

open Real

variables (α : ℝ)

def a : ℝ × ℝ := (cos α, sin α)
def b : ℝ × ℝ := (-sin α, cos α)

def m : ℝ × ℝ := (sqrt 3 * cos α - sin α, sqrt 3 * sin α + cos α)
def n : ℝ × ℝ := (cos α + sqrt 3 * (-sin α), sin α + sqrt 3 * cos α)

theorem magnitude_a_plus_b :
  ‖(cos α + (-sin α), sin α + cos α)‖ = sqrt 2 :=
sorry

theorem angle_between_m_and_n :
  arccos ((sqrt 3 * cos α - sin α, sqrt 3 * sin α + cos α) • (cos α + sqrt 3 * (-sin α), sin α + sqrt 3 * cos α) / (‖(sqrt 3 * cos α - sin α, sqrt 3 * sin α + cos α)‖ * ‖(cos α + sqrt 3 * (-sin α), sin α + sqrt 3 * cos α)‖)) = π / 6 :=
sorry

end magnitude_a_plus_b_angle_between_m_and_n_l269_269653


namespace solution_exists_solution_unique_l269_269097

noncomputable def abc_solutions : Finset (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (2, 2, 4), (2, 4, 8), (3, 5, 15), 
   (2, 4, 2), (4, 2, 2), (4, 2, 8), (8, 4, 2), 
   (2, 8, 4), (8, 2, 4), (5, 3, 15), (15, 3, 5), (3, 15, 5),
   (2, 2, 4), (4, 2, 2), (4, 8, 2)}

theorem solution_exists (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a * b * c - 1 = (a - 1) * (b - 1) * (c - 1)) ↔ (a, b, c) ∈ abc_solutions := 
by
  sorry

theorem solution_unique (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a, b, c) ∈ abc_solutions → a * b * c - 1 = (a - 1) * (b - 1) * (c - 1) :=
by
  sorry

end solution_exists_solution_unique_l269_269097


namespace simplified_expression_l269_269794

noncomputable def H (x : ℝ) : ℝ := log 2 ((1 - x) / (1 + x))

noncomputable def I (x : ℝ) : ℝ :=
  log 2 ((1 - ((2 * x - x^2) / (1 + 2 * x^2))) / (1 + ((2 * x - x^2) / (1 + 2 * x^2))))

theorem simplified_expression (x : ℝ) : I x = 2 * H x :=
  sorry

end simplified_expression_l269_269794


namespace find_angle_ABC_l269_269486

-- Define the right triangle and the circle conditions
variable {A B C E D: Type}
variable [EuclideanGeometry A B C]

-- A right triangle with the right angle at C
axiom right_triangle : ∀ (A B C: Type) [EuclideanGeometry A B C], (∠ABC = 90°)
-- The circle touches AC at E and BC at D
axiom circle_touches : ∀ (A B C E D: Type) [EuclideanGeometry A B C], E ∈ AC ∧ D ∈ BC ∧ (AE = 1) ∧ (BD = 3)

-- Prove the angle ∠ABC
theorem find_angle_ABC : ∀ (A B C E D: Type) [EuclideanGeometry A B C], 
  right_triangle A B C → circle_touches A B C E D → ∠ABC = 30° := 
by 
  -- skipping the proof with sorry
  sorry

end find_angle_ABC_l269_269486


namespace area_of_triangle_bounded_by_line_and_axes_l269_269520

theorem area_of_triangle_bounded_by_line_and_axes
    (x y : ℝ)
    (h_line_eq : 3 * x + 2 * y = 12)
    (x_intercept : x = 4)
    (y_intercept : y = 6) :
    (1 / 2) * x_intercept * y_intercept = 12 := by
  sorry

end area_of_triangle_bounded_by_line_and_axes_l269_269520


namespace max_value_seq_l269_269308

noncomputable def a_n (n : ℕ) : ℝ := n / (n^2 + 90)

theorem max_value_seq : ∃ n : ℕ, a_n n = 1 / 19 :=
by
  sorry

end max_value_seq_l269_269308


namespace smallest_perimeter_of_triangle_with_consecutive_sides_l269_269867

-- Definitions based on the conditions:
def triangle_valid (n : ℕ) : Prop :=
  n ≥ 4 ∧
  n + (n + 1) > n + 2 ∧
  n + (n + 2) > n + 1 ∧
  (n + 1) + (n + 2) > n

-- The theorem statement for the proof problem:
theorem smallest_perimeter_of_triangle_with_consecutive_sides (n : ℕ) (h : triangle_valid n) : n = 4 → n + (n + 1) + (n + 2) = 15 := 
by
  intro hn
  rw [hn]
  simp
  sorry


end smallest_perimeter_of_triangle_with_consecutive_sides_l269_269867


namespace conjugate_z_is_1_add_2i_l269_269761

open Complex

def z_def : ℂ := ((2 : ℂ) + I) / (1 + I^2 + I^5)

theorem conjugate_z_is_1_add_2i : conj z_def = (1 : ℂ) + 2 * I := 
sorry

end conjugate_z_is_1_add_2i_l269_269761


namespace find_perimeter_l269_269339

-- Given the conditions for the circles and triangle
variable (P Q R S : Type) [CommRing P] [CommRing Q] [CommRing R] [CommRing S]
variable (radius : ℝ) (r2 : radius = 2)
variable (P_center Q_center R_center S_center : ℝ × ℝ)
variable (tangent_PQ : dist P_center Q_center = 4)
variable (tangent_QR : dist Q_center R_center = 4)
variable (tangent_RS : dist R_center S_center = 4)
variable (tangent_PS : dist P_center S_center = 4)
variable (triangle_angle : ∀ (A B C : ℝ), is_right_triangle A B C)
variable (triangle_ABC : is_right_triangle 8 8 (8 * real.sqrt 2))

-- The main statement to prove the perimeter
theorem find_perimeter (A B C : ℝ)
    (triangle : is_right_triangle A B C)
    (side_A : A = 8)
    (side_B : B = 8)
    (hypotenuse_C : C = 8 * real.sqrt 2) :
    perimeter triangle = 16 + 8 * real.sqrt 2 := 
sorry

end find_perimeter_l269_269339


namespace num_subsets_set_3_l269_269660

theorem num_subsets_set_3 : (set.powerset {0, 1, 2}).card = 8 := 
sorry

end num_subsets_set_3_l269_269660


namespace minimum_canvas_dimensions_l269_269964

theorem minimum_canvas_dimensions {
  x y : ℝ 
  (h1 : x * y = 72)
  (h2 : x = 6)
  (h3 : y = 12)
} : (x + 4 = 10) ∧ (y + 8 = 20) :=
by {
  sorry,
}

end minimum_canvas_dimensions_l269_269964


namespace advertising_department_employees_l269_269923

theorem advertising_department_employees (N S A_s x : ℕ) (hN : N = 1000) (hS : S = 80) (hA_s : A_s = 4) 
(h_stratified : x / N = A_s / S) : x = 50 :=
sorry

end advertising_department_employees_l269_269923


namespace monica_usd_start_amount_l269_269381

theorem monica_usd_start_amount (x : ℕ) (H : ∃ (y : ℕ), y = 40 ∧ (8 : ℚ) / 5 * x - y = x) :
  (x / 100) + (x % 100 / 10) + (x % 10) = 2 := 
by
  sorry

end monica_usd_start_amount_l269_269381


namespace proof_x_exists_l269_269233

noncomputable def find_x : ℝ := 33.33

theorem proof_x_exists (A B C : ℝ) (h1 : A = (1 + find_x / 100) * B) (h2 : C = 0.75 * A) (h3 : A > C) (h4 : C > B) :
  find_x = 33.33 := 
by
  -- Proof steps
  sorry

end proof_x_exists_l269_269233


namespace sample_size_stratified_sampling_l269_269947

theorem sample_size_stratified_sampling 
  (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (n : ℕ) (females_drawn : ℕ) 
  (total_people : ℕ := teachers + male_students + female_students) 
  (females_total : ℕ := female_students) 
  (proportion_drawn : ℚ := (females_drawn : ℚ) / females_total) :
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  females_drawn = 80 → 
  proportion_drawn = ((n : ℚ) / total_people) → 
  n = 192 :=
by
  sorry

end sample_size_stratified_sampling_l269_269947


namespace find_number_l269_269460

theorem find_number (x : ℕ) (h : 3 * (x + 2) = 24 + x) : x = 9 :=
by 
  sorry

end find_number_l269_269460


namespace exists_number_starts_with_10_l269_269820

-- Define the structures and conditions of the problem
variables {a d : ℕ}
def is_in_arithmetic_progression (n : ℕ) : Prop := 
  ∃ k : ℕ, n = a + k * d

def starts_with_digits_10 (n : ℕ) : Prop :=
  n / 10^(nat.logBase 10 n) = 10

-- Main theorem statement
theorem exists_number_starts_with_10 (a d : ℕ) (h_infinite : ∀ n, ∃ m > n, is_in_arithmetic_progression m) (h_increasing : ∀ k1 k2, k1 < k2 → (a + k1 * d) < (a + k2 * d)) :
  ∃ n, is_in_arithmetic_progression n ∧ starts_with_digits_10 n :=
begin
  sorry
end

end exists_number_starts_with_10_l269_269820


namespace monotonicity_of_f_extreme_values_of_f_l269_269642

noncomputable def f (x : ℝ) := x^4 - 3 * x^2 + 6

theorem monotonicity_of_f :
  (∀ x, -real.sqrt 6 / 2 < x → x < 0 → 0 < deriv f x) ∧
  (∀ x, x > real.sqrt 6 / 2 → 0 < deriv f x) ∧
  (∀ x, x < -real.sqrt 6 / 2 → deriv f x < 0) ∧
  (∀ x, 0 < x → x < real.sqrt 6 / 2 → deriv f x < 0) :=
by
  sorry

theorem extreme_values_of_f :
  f (-real.sqrt 6 / 2) = 15 / 4 ∧ f 0 = 6 ∧ f (real.sqrt 6 / 2) = 15 / 4 :=
by
  sorry

#eval f (-real.sqrt 6 / 2)
#eval f 0
#eval f (real.sqrt 6 / 2)

end monotonicity_of_f_extreme_values_of_f_l269_269642


namespace twelfth_prime_is_37_l269_269456

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sequence : ℕ → ℕ
| 0     := 2
| 1     := 3
| 2     := 5
| 3     := 7
| 4     := 11
| 5     := 13
| 6     := 17
| 7     := 19
| 8     := 23
| 9     := 29
| 10    := 31
| 11    := 37
| (n+1) := sorry  -- This should continue the prime sequence, but for our proof problem, it's good enough.

theorem twelfth_prime_is_37 : prime_sequence 11 = 37 := by
  sorry

end twelfth_prime_is_37_l269_269456


namespace trig_identity_solution_l269_269580

theorem trig_identity_solution (α : ℝ) (h : Real.tan α = -1 / 2) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 :=
by
  sorry

end trig_identity_solution_l269_269580


namespace obtain_2001_from_22_l269_269185

theorem obtain_2001_from_22 :
  ∃ (f : ℕ → ℕ), (∀ n, f (n + 1) = n ∨ f (n) = n + 1) ∧ (f 22 = 2001) := 
sorry

end obtain_2001_from_22_l269_269185


namespace factorable_polynomials_l269_269878

-- Definitions of the polynomials
def poly1 := 3 * x^2 + 3 * y^2
def poly2 := -x^2 + y^2
def poly3 := -x^2 - y^2
def poly4 := x^2 + x * y + y^2
def poly5 := x^2 + 2 * x * y - y^2
def poly6 := -x^2 + 4 * x * y - 4 * y^2

-- Lean 4 statement to prove which polynomials can be factored using the formula method
theorem factorable_polynomials:
  (∃ a b, poly2 = (a + b) * (a - b)) ∧ (∃ a b, poly6 = (a + b) * (a - b)) :=
by
  sorry

end factorable_polynomials_l269_269878


namespace percentage_increase_in_ducks_l269_269081

def number_of_cows : ℕ := 20
def total_animals : ℕ := 60
def number_of_ducks : ℕ := 30
def number_of_pigs : ℕ := (1 / 5 : ℚ) * (number_of_ducks + number_of_cows)
def percentage_increase (initial final : ℕ) : ℚ := ((final - initial) / initial : ℚ) * 100

theorem percentage_increase_in_ducks (H : number_of_cows + number_of_ducks + (1 / 5 : ℚ) * (number_of_ducks + number_of_cows) = total_animals) :
  percentage_increase number_of_cows number_of_ducks = 50 :=
by sorry

end percentage_increase_in_ducks_l269_269081


namespace packs_needed_proof_l269_269044

def sundaes_on_monday : ℕ := 40
def mms_per_sundae_monday : ℕ := 6
def gummy_bears_per_sundae_monday : ℕ := 4
def mini_marshmallows_per_sundae_monday : ℕ := 8

def sundaes_on_tuesday : ℕ := 20
def mms_per_sundae_tuesday : ℕ := 10
def gummy_bears_per_sundae_tuesday : ℕ := 5
def mini_marshmallows_per_sundae_tuesday : ℕ := 12

def mms_per_pack : ℕ := 40
def gummy_bears_per_pack : ℕ := 30
def mini_marshmallows_per_pack : ℕ := 50

def total_mms_packs_needed : ℕ := 11
def total_gummy_bears_packs_needed : ℕ := 9
def total_mini_marshmallows_packs_needed : ℕ := 12

theorem packs_needed_proof :
  (⌈(sundaes_on_monday * mms_per_sundae_monday + sundaes_on_tuesday * mms_per_sundae_tuesday) / mms_per_pack⌉ = total_mms_packs_needed) ∧
  (⌈(sundaes_on_monday * gummy_bears_per_sundae_monday + sundaes_on_tuesday * gummy_bears_per_sundae_tuesday) / gummy_bears_per_pack⌉ = total_gummy_bears_packs_needed) ∧
  (⌈(sundaes_on_monday * mini_marshmallows_per_sundae_monday + sundaes_on_tuesday * mini_marshmallows_per_sundae_tuesday) / mini_marshmallows_per_pack⌉ = total_mini_marshmallows_packs_needed) := 
by {
  sorry
}

end packs_needed_proof_l269_269044


namespace problem_1_problem_2_problem_3_l269_269445

section Problem

variable {A B C : ℝ}
variable {a_n S_n : ℕ → ℝ}
Variable {b_n T_n : ℕ → ℝ}
variable {c_n : ℕ → ℝ}
variable {λ : ℝ}

/-- Initial conditions: The sum of the first n terms of the sequence {a_n} is S_n and constants A, B, and C, such that for any positive integer n,
    a_n + S_n = A n^2 + B n + C holds true. -/
def initial_conditions : Prop :=
  ∀ n : ℕ, n > 0 → (a_n n + S_n n = A * n^2 + B * n + C)

/-- If A = -1/2, B = -3/2, and C = 1, let b_n = a_n + n. Prove that the sequence {b_n} is a geometric progression. -/
theorem problem_1 (H : initial_conditions)
  (HA : A = -1/2) (HB : B = -3/2) (HC : C = 1):
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → b_n n = r * b_n (n - 1) :=
sorry

/-- Under the condition of problem 1, let c_n = (2n+1)b_n. The sum of the first n terms of the sequence {c_n} is T_n. Prove that T_n < 5. -/
theorem problem_2 (H : initial_conditions)
  (HA : A = -1/2) (HB : B = -3/2) (HC : C = 1):
  ∀ n : ℕ, n > 0 → T_n n = (Finset.range (n+1)).sum (λ i, c_n (i + 1)) →
  T_n n < 5 :=
sorry

/-- If C = 0 and {a_n} is an arithmetic sequence with the first term equal to 1, and for any positive integer n,
    the inequality λ + n ≤ ∑_{i=1}^{n} √[1 + 2/a_i^2 + 1/a_{i+1}^2] holds, find the range of the real number λ. -/
theorem problem_3 (H : initial_conditions)
  (HC : C = 0) (H_arith : ∀ n : ℕ, a_n n = 1 + n) (H_ineq : ∀ n : ℕ, λ + n ≤ ∑ i in Finset.range (n+1), Real.sqrt (1 + 2 / (a_n (i+1))^2 + 1 / (a_n (i+2))^2)):
  λ ≤ 1/2 :=
sorry

end Problem

end problem_1_problem_2_problem_3_l269_269445


namespace range_of_reciprocal_sum_l269_269668

theorem range_of_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : a + b = 1) :
  ∃ c > 4, ∀ x, x = (1 / a + 1 / b) → c < x :=
sorry

end range_of_reciprocal_sum_l269_269668


namespace rational_root_of_factors_l269_269793

theorem rational_root_of_factors (p : ℕ) (a : ℚ) (hprime : Nat.Prime p) 
  (f : Polynomial ℚ) (hf : f = Polynomial.X ^ p - Polynomial.C a)
  (hfactors : ∃ g h : Polynomial ℚ, f = g * h ∧ 1 ≤ g.degree ∧ 1 ≤ h.degree) : 
  ∃ r : ℚ, Polynomial.eval r f = 0 :=
sorry

end rational_root_of_factors_l269_269793


namespace complex_conjugate_of_z_l269_269748

-- Define z based on the given expression
noncomputable def z : ℂ := (2 + complex.i) / (1 + complex.i^2 + complex.i^5)

-- State the theorem to prove the complex conjugate of z is 1 + 2i
theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * complex.i :=
sorry

end complex_conjugate_of_z_l269_269748


namespace evaluate_expression_l269_269566

theorem evaluate_expression : 
  (3^4 + 3^4 + 3^4) / (3^(-4) + 3^(-4)) = 9841.5 :=
by
  sorry

end evaluate_expression_l269_269566


namespace remove_all_stones_l269_269161

theorem remove_all_stones (a b c : ℕ) : 
  ∃ m : ℕ, (∀ n, n ≤ m → (exists p q r, p + q + r = n * 3)) ∧ (∃ k : ℕ → ℕ, ∀ i <= m, k i = 0) :=
sorry

end remove_all_stones_l269_269161


namespace prime_of_1_add_2_pow_n_add_4_pow_n_l269_269083

theorem prime_of_1_add_2_pow_n_add_4_pow_n (n : ℕ) (h : Nat.prime (1 + 2^n + 4^n)) : ∃ k : ℕ, n = 3^k := 
sorry

end prime_of_1_add_2_pow_n_add_4_pow_n_l269_269083


namespace length_of_AB_l269_269648

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 8 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

-- Define the property of being an intersection point of both circles
def is_intersection (A : ℝ × ℝ) : Prop := circle1 A.1 A.2 ∧ circle2 A.1 A.2

-- Define the distance function between two points
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The main theorem statement
theorem length_of_AB : ∃ A B : ℝ × ℝ, is_intersection A ∧ is_intersection B ∧ A ≠ B ∧ distance A B = sorry :=
begin
  sorry
end

end length_of_AB_l269_269648


namespace gcd_lcm_sum_l269_269051

theorem gcd_lcm_sum (a b : ℕ) (h : a = 1999 * b) : Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end gcd_lcm_sum_l269_269051


namespace complex_conjugate_of_z_l269_269777

theorem complex_conjugate_of_z :
  let z := (2 + complex.i) / (1 + complex.i ^ 2 + complex.i ^ 5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269777


namespace smallest_x_plus_y_l269_269613

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269613


namespace planar_graph_edge_vertex_inequality_l269_269821

def planar_graph (G : Type _) : Prop := -- Placeholder for planar graph property
  sorry

variables {V E : ℕ}

theorem planar_graph_edge_vertex_inequality (G : Type _) (h : planar_graph G) :
  E ≤ 3 * V - 6 :=
sorry

end planar_graph_edge_vertex_inequality_l269_269821


namespace fraction_pizza_covered_by_pepperoni_l269_269565

theorem fraction_pizza_covered_by_pepperoni :
  (∀ (r_pizz : ℝ) (n_pepp : ℕ) (d_pepp : ℝ),
      r_pizz = 8 ∧ n_pepp = 32 ∧ d_pepp = 2 →
      (n_pepp * π * (d_pepp / 2)^2) / (π * r_pizz^2) = 1 / 2) :=
sorry

end fraction_pizza_covered_by_pepperoni_l269_269565


namespace max_area_triangle_ABC_l269_269627

noncomputable def circle_radius := 1

def points_on_circle (O A B C D : Type) [MetricSpace O] :=
  dist O A = circle_radius ∧
  dist O B = circle_radius ∧
  dist O C = circle_radius ∧
  dist O D = circle_radius

def vector_eq (O A B C D : Type) [Add O] [HasSmul ℝ O] [AddCommGroup O] : Prop :=
  (B - A) + (C - A) = (D - A)

def max_area_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : ℝ :=
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  1 / 2 * abs (AB * AC) * sin (angle A B C)

theorem max_area_triangle_ABC
  (O A B C D : Type)
  [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  [Add O] [HasSmul ℝ O] [AddCommGroup O]
  (h1 : points_on_circle O A B C D)
  (h2 : vector_eq O A B C D) :
  max_area_triangle A B C = 1 :=
sorry

end max_area_triangle_ABC_l269_269627


namespace coefficient_of_inverse_x_l269_269424

theorem coefficient_of_inverse_x :
  let f := (1 - x^2)^4 * (x + 1) / x in
  coefficient (1/x : ℚ) (expansion f) = -29 :=
by
  sorry

end coefficient_of_inverse_x_l269_269424


namespace prob_blue_section_damaged_all_days_l269_269902

noncomputable def prob_of_7_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_blue_section_damaged_all_days :
  prob_of_7_successes 7 7 (2 / 7) = 128 / 823543 :=
by sorry

end prob_blue_section_damaged_all_days_l269_269902


namespace heracles_age_l269_269974

theorem heracles_age
  (H : ℕ)
  (audrey_current_age : ℕ)
  (audrey_in_3_years : ℕ)
  (h1 : audrey_current_age = H + 7)
  (h2 : audrey_in_3_years = audrey_current_age + 3)
  (h3 : audrey_in_3_years = 2 * H)
  : H = 10 :=
by
  sorry

end heracles_age_l269_269974


namespace total_monsters_l269_269228

theorem total_monsters (a1 a2 a3 a4 a5 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 2 * a1) 
  (h3 : a3 = 2 * a2) 
  (h4 : a4 = 2 * a3) 
  (h5 : a5 = 2 * a4) : 
  a1 + a2 + a3 + a4 + a5 = 62 :=
by
  sorry

end total_monsters_l269_269228


namespace john_money_left_l269_269043

variable (q : ℝ) 

def cost_soda := q
def cost_medium_pizza := 3 * q
def cost_small_pizza := 2 * q

def total_cost := 4 * cost_soda q + 2 * cost_medium_pizza q + 3 * cost_small_pizza q

theorem john_money_left (h : total_cost q = 16 * q) : 50 - total_cost q = 50 - 16 * q := by
  simp [total_cost, cost_soda, cost_medium_pizza, cost_small_pizza]
  sorry

end john_money_left_l269_269043


namespace k_eq_one_f_monotonic_increasing_solve_inequality_m_eq_two_l269_269372

-- Function definition
def f (k a x : ℝ) : ℝ := k * a^x - a^(-x)

-- Problem (1): Prove that k = 1 for f(x) to be an odd function
theorem k_eq_one (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : ∀ x : ℝ, f k a x = -f k a (-x)) : k = 1 :=
sorry

-- Problem (2): Prove that f(x) is monotonically increasing for a > 1
theorem f_monotonic_increasing (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : a > 1) : ∀ x y : ℝ, x < y → f 1 a x < f 1 a y :=
sorry

-- Problem (3): Solve the inequality f(x^2 + 2x) + f(x - 4) > 0 for a > 1
theorem solve_inequality (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : a > 1) : ∀ x : ℝ, (f 1 a (x^2 + 2 * x) + f 1 a (x - 4) > 0) ↔ (x > 1 ∨ x < -4) :=
sorry

-- Problem (4): Prove m = 2 given f(1) = 0 and the minimum value of g on [1, +∞) is -2
theorem m_eq_two (a : ℝ) (m : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : f 1 a 1 = 0) (h₄ : ∀ x ∈ set.Ici 1, g a m x ≥ -2) (h₅ : ∃ x ∈ set.Ici 1, g a m x = -2) : m = 2 :=
sorry

-- Definition of g(x)
def g (a m x : ℝ) : ℝ := a^(2 * x) + a^(-2 * x) - 2 * m * (f 1 a x)

end k_eq_one_f_monotonic_increasing_solve_inequality_m_eq_two_l269_269372


namespace complex_conjugate_of_z_l269_269767

theorem complex_conjugate_of_z : 
  let z := (2 + complex.i) / (1 + complex.i^2 + complex.i^5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269767


namespace unique_prime_digit_product_l269_269121

-- Definition of prime digits allowed
def is_prime_digit (n : ℕ) : Prop := n ∈ {2, 3, 5, 7}

-- Definition of a prime digit number (all digits must be prime digits)
def is_prime_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ n.digits 10, is_prime_digit digit

-- Definition of our specific problem layout
def specific_layout (n m product : ℕ) : Prop :=
  is_prime_digit_number n ∧ n / 100 > 0 ∧ is_prime_digit_number m ∧ m / 10 > 0 ∧ n * m = product ∧ is_prime_digit_number product

theorem unique_prime_digit_product:
  ∃! (n m : ℕ), specific_layout n m 2325 ∧ n = 775 ∧ m = 3 :=
by
  sorry

end unique_prime_digit_product_l269_269121


namespace sum_integers_a_l269_269646

theorem sum_integers_a (a : ℤ) :
  (forall x : ℝ, (3 * x + 3) / 2 < 2 * x + 1 ∧ (3 * (a - x)) / 2 ≤ 2 * x - 1 / 2 → x > 1) → 
  (y : ℝ, (8 - 5 * y) / (1 - y) = a / (y - 1) + 2 → y >= 0) → 
  ∑ i in {2, 1, 0, -1, -2, -4, -5, -6} \ {-3}, i = -15 :=
by
  sorry

end sum_integers_a_l269_269646


namespace distance_from_start_after_7km_l269_269506

-- Define the coordinates of the equilateral triangle vertices.
def equilateral_triangle_coords : list (ℝ × ℝ) :=
  [(0,0), (3,0), (1.5, 3 * Real.sqrt 3 / 2)]

-- Function to find the distance between two points.
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Function to calculate Bob's position after walking a certain distance along the triangle's perimeter.
def bob_position (d : ℝ) : ℝ × ℝ :=
  if d ≤ 3 then
    (d, 0)  -- Along the first side
  else if d ≤ 6 then
    let x := 3 - (d - 3) * 0.5
    let y := (d - 3) * (3 * Real.sqrt 3 / 2 / 3)
    (x, y)  -- Along the second side
  else
    let x := 1.5 - (d - 6) * 0.5
    let y := (3 * Real.sqrt 3 / 2) - (d - 6) * (3 * Real.sqrt 3 / 2 / 3)
    (x, y)  -- Along the third side

-- Prove that after walking 7 km, Bob is sqrt(7) km away from the starting point.
theorem distance_from_start_after_7km :
  distance (bob_position 7) (0, 0) = Real.sqrt 7 :=
by
  sorry

end distance_from_start_after_7km_l269_269506


namespace candy_in_each_box_l269_269817

theorem candy_in_each_box (C K : ℕ) (h1 : 6 * C + 4 * K = 90) (h2 : C = K) : C = 9 :=
by
  -- Proof will go here
  sorry

end candy_in_each_box_l269_269817


namespace sandra_total_beignets_l269_269400

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end sandra_total_beignets_l269_269400


namespace circumcenter_fixed_circle_l269_269229

open EuclideanGeometry

variables (S1 S2 : Circle) (P Q : Point) 
          (A1 B1 A2 B2 C : Point)
          (O1 O2 O : Point) -- Note: O1, O2 are centers of S1, S2. O is a specific point mentioned.

def conditions :=
  S1 ∩ S2 = {P, Q} ∧
  A1 ∈ S1 ∧ B1 ∈ S1 ∧ 
  A1 ≠ P ∧ A1 ≠ Q ∧ B1 ≠ P ∧ B1 ≠ Q ∧
  (Line_through A1 P).Meets S2 = A2 ∧
  (Line_through B1 P).Meets S2 = B2 ∧
  (Line_through A1 B1).Meets (Line_through A2 B2) = C

theorem circumcenter_fixed_circle (h : conditions S1 S2 P Q A1 B1 A2 B2 C O1 O2 O) :
  ∃ fixed_circle,
  ∀ (A1_new B1_new : Point),
  A1_new ∈ S1 ∧ B1_new ∈ S1 ∧ 
  A1_new ≠ P ∧ A1_new ≠ Q ∧ B1_new ≠ P ∧ B1_new ≠ Q ∧
  let A2_new := (Line_through A1_new P).Meets S2 in
  let B2_new := (Line_through B1_new P).Meets S2 in
  let C_new  := (Line_through A1_new B1_new).Meets (Line_through A2_new B2_new) in
  Circumcenter A1_new A2_new C_new ∈ fixed_circle := sorry

end circumcenter_fixed_circle_l269_269229


namespace volume_of_sphere_is_correct_l269_269692

-- We begin by defining the conditions stated in the problem.

-- 1. Defining the distance OG.
def OG : ℝ := sqrt 3 / 3

-- 2. The area of the circumcircle is given.
def area_circumcircle : ℝ := (2 * π) / 3 

-- 3. The radius of the circumcircle can be derived from the given area.
def r : ℝ := sqrt (area_circumcircle / π)

-- 4. The radius of the sphere is the sum of OG and the radius r.
def R : ℝ := OG + r

-- 5. Therefore, the volume of the sphere is:
def volume_sphere : ℝ := (4 / 3) * π * R^3

-- Statement to prove
theorem volume_of_sphere_is_correct : volume_sphere = (4 * π) / 3 :=
by
  -- Proof omitted
  sorry

end volume_of_sphere_is_correct_l269_269692


namespace complex_conjugate_of_z_l269_269752

-- Define z based on the given expression
noncomputable def z : ℂ := (2 + complex.i) / (1 + complex.i^2 + complex.i^5)

-- State the theorem to prove the complex conjugate of z is 1 + 2i
theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * complex.i :=
sorry

end complex_conjugate_of_z_l269_269752


namespace largest_whole_number_l269_269113

theorem largest_whole_number (x : ℕ) : 9 * x < 150 → x ≤ 16 :=
by sorry

end largest_whole_number_l269_269113


namespace find_general_formula_for_an_find_sum_sn_l269_269586

noncomputable theory

open Classical

def sequence (a : ℕ → ℝ) := ∀ n : ℕ, 0 < a n

def property (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ n : ℕ, (a n + a (n+1)) * (a (n+1) - a n) + 2 * a n ^ 2 * a (n+1) ^ 2 = 0

theorem find_general_formula_for_an {a : ℕ → ℝ} 
  (hseq : sequence a) 
  (hprop : property a) : 
  ∀ n : ℕ, a n = 1 / Real.sqrt (2 * ↑n - 1) :=
sorry

theorem find_sum_sn {a : ℕ → ℝ} 
  {b : ℕ → ℝ} 
  {S : ℕ → ℝ}
  (hseq : sequence a) 
  (hprop : property a) 
  (hb : ∀ n : ℕ, b n = a n * a (n+1)) 
  (hbn2 : ∀ n : ℕ, b n ^ 2 = 0.5 * (1 / (2 * ↑n - 1) - 1 / (2 * ↑n + 1)))
  (hS : ∀ n : ℕ, S n = (1 / 2) * (1 - (1 / (2 * ↑n + 1)))) :
  ∀ n : ℕ, S n = ↑n / (2 * ↑n + 1) :=
sorry

end find_general_formula_for_an_find_sum_sn_l269_269586


namespace sqrt_of_9_l269_269096

theorem sqrt_of_9 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_l269_269096


namespace polar_to_parametric_l269_269296

theorem polar_to_parametric (θ : ℝ) : 
  (ρ = 2 * Real.cos θ) →
  ∃ x y, x = 1 + Real.cos θ ∧ y = Real.sin θ :=
by
  intro h
  use (1 + Real.cos θ), (Real.sin θ)
  sorry

end polar_to_parametric_l269_269296


namespace problem1_problem2_problem3_problem4_l269_269098

-- Problem 1: (3x-1)^2 = (x+1)^2
theorem problem1 (x : ℝ) : (3 * x - 1)^2 = (x + 1)^2 ↔ x = 0 ∨ x = 1 := 
by sorry

-- Problem 2: (x-1)^2 + 2x(x-1) = 0
theorem problem2 (x : ℝ) : (x - 1)^2 + 2 * x * (x - 1) = 0 ↔ x = 1 ∨ x = 1/3 := 
by sorry

-- Problem 3: x^2 - 4x + 1 = 0
theorem problem3 (x : ℝ) : x^2 - 4 * x + 1 = 0 ↔ x = 2 + real.sqrt 3 ∨ x = 2 - real.sqrt 3 := 
by sorry

-- Problem 4: 2x^2 + 7x - 4 = 0
theorem problem4 (x : ℝ) : 2 * x^2 + 7 * x - 4 = 0 ↔ x = 1/2 ∨ x = -4 := 
by sorry

end problem1_problem2_problem3_problem4_l269_269098


namespace train_and_car_number_exists_l269_269803

theorem train_and_car_number_exists 
  (SECRET OPEN ANSWER YOUR OPENED : ℕ)
  (h1 : SECRET - OPEN = ANSWER - YOUR)
  (h2 : SECRET - OPENED = 20010)
  (distinct_digits : ∀ (a b : ℕ), a ≠ b → ∀ (c : ℕ), (digits a) ≠ (digits b))
  (same_digits : ∀ (a b : ℕ), a = b → ∀ (c : ℕ), (digits a) = (digits b)) :
  ∃ (train car : ℕ), train = 392 ∧ car = 2 :=
sorry

end train_and_car_number_exists_l269_269803


namespace find_second_sum_l269_269957

theorem find_second_sum (x : ℝ) (total_sum : ℝ) (h : total_sum = 2691) 
  (h1 : (24 * x) / 100 = 15 * (total_sum - x) / 100) : total_sum - x = 1656 :=
by
  sorry

end find_second_sum_l269_269957


namespace range_m_l269_269065

def f (a x : ℝ) := (1 - a) / 2 * x^2 + a * x - Real.log x

def g (a : ℝ) := (a - 3) / (a^2 - 1)

theorem range_m (a m x1 x2 : ℝ) (h1 : 3 < a) (h2 : a < 4)
  (hx1 : 1 ≤ x1) (hx2 : x1 ≤ 2) (hx3 : 1 ≤ x2) (hx4 : x2 ≤ 2)
  (h5 : (a^2 - 1) / 2 * m + Real.log 2 > |f a x1 - f a x2|):
  m ≥ 1 / 15 :=
  sorry

end range_m_l269_269065


namespace mary_biking_time_l269_269377

-- Define the conditions and the task
def total_time_away := 570 -- in minutes
def time_in_classes := 7 * 45 -- in minutes
def lunch_time := 40 -- in minutes
def additional_activities := 105 -- in minutes
def time_in_school_activities := time_in_classes + lunch_time + additional_activities

-- Define the total biking time based on given conditions
theorem mary_biking_time : 
  total_time_away - time_in_school_activities = 110 :=
by 
-- sorry is used to skip the proof step.
  sorry

end mary_biking_time_l269_269377


namespace net_annual_revenue_2020_l269_269002

-- Define conditions for 2018
def initial_stores_2018 := 23
def avg_revenue_2018 := 500000

-- Define conditions for 2019
def new_stores_2019 := 5
def revenue_per_new_store_2019 := 450000
def closed_stores_2019 := 2
def revenue_per_closed_store_2019 := 300000
def expense_per_closed_store_2019 := 350000

-- Define conditions for 2020
def new_stores_2020 := 10
def revenue_per_new_store_2020 := 600000
def closed_stores_2020 := 6
def avg_revenue_per_closed_store_2020 := 350000
def avg_expense_per_closed_store_2020 := 380000

-- Average annual expense for each remaining store in 2020
def avg_expense_per_store_2020 := 400000

-- Prove that net annual revenue at the end of 2020 is $5,130,000
theorem net_annual_revenue_2020 (initial_stores_2018 : ℕ)
    (avg_revenue_2018 : ℕ)
    (new_stores_2019 : ℕ)
    (revenue_per_new_store_2019 : ℕ)
    (closed_stores_2019 : ℕ)
    (revenue_per_closed_store_2019 : ℕ)
    (expense_per_closed_store_2019 : ℕ)
    (new_stores_2020 : ℕ)
    (revenue_per_new_store_2020 : ℕ)
    (closed_stores_2020 : ℕ)
    (avg_revenue_per_closed_store_2020 : ℕ)
    (avg_expense_per_closed_store_2020 : ℕ)
    (avg_expense_per_store_2020 : ℕ):
    let total_revenue_start_2019 := initial_stores_2018 * avg_revenue_2018,
        revenue_new_stores_2019 := new_stores_2019 * revenue_per_new_store_2019,
        revenue_closed_stores_2019 := closed_stores_2019 * revenue_per_closed_store_2019,
        expense_closed_stores_2019 := closed_stores_2019 * expense_per_closed_store_2019,
        net_loss_2019 := revenue_closed_stores_2019 - expense_closed_stores_2019,
        total_revenue_end_2019 := total_revenue_start_2019 + revenue_new_stores_2019 - revenue_closed_stores_2019,
        net_revenue_2019 := total_revenue_end_2019 + net_loss_2019,
        
        revenue_new_stores_2020 := new_stores_2020 * revenue_per_new_store_2020,
        revenue_closed_stores_2020 := closed_stores_2020 * avg_revenue_per_closed_store_2020,
        expense_closed_stores_2020 := closed_stores_2020 * avg_expense_per_closed_store_2020,
        net_gain_2020 := expense_closed_stores_2020 - revenue_closed_stores_2020,
        total_revenue_end_2020 := net_revenue_2019 + revenue_new_stores_2020 - revenue_closed_stores_2020,
        net_revenue_2020_before_expenses := total_revenue_end_2020 + net_gain_2020,
        remaining_stores := initial_stores_2018 + new_stores_2019 - closed_stores_2019 + new_stores_2020 - closed_stores_2020,
        total_expenses_remaining := remaining_stores * avg_expense_per_store_2020,
        net_annual_revenue_2020 := net_revenue_2020_before_expenses - total_expenses_remaining
    in
    net_annual_revenue_2020 = 5130000 :=
  by sorry

end net_annual_revenue_2020_l269_269002


namespace find_P_l269_269837

theorem find_P (P Q R S T U : ℕ) : 
  {P, Q, R, S, T, U} = {1, 2, 3, 4, 5, 6} →
  (10 * P + 10^2 * Q + 10^1 * R + 10 * 0 + 10^1 * 5).mod 4 = 0 →
  (100 * Q + 10 * R + S).mod 5 = 0 →
  (10 * R + 5 * 0).mod 2 = 0 →
  (P + Q + R + S + 5 + U).mod 11 = 0 →
  P = 1 :=
by
  sorry

end find_P_l269_269837


namespace probability_of_7_successes_l269_269891

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l269_269891


namespace hypotenuse_length_l269_269374

theorem hypotenuse_length (x : ℝ) (c : ℝ) (h1 : 0 < x) (h2 : x < π / 2)
  (h3 : x ∈ Ioo 0 (π / 2)) : 
  let AD := Real.tan x,
      AE := Real.cot x in
  c = (2 * Real.sqrt 5) / 3 :=
sorry

end hypotenuse_length_l269_269374


namespace angles_in_triangle_OHC_l269_269022

theorem angles_in_triangle_OHC (A B C O H: Type) {α β γ : ℝ}
(hA_triangle : ∠ A B C = α)
(hB_angle : β = 50)
(hC_angle : γ = 70)
(hO_circumcenter : is_circumcenter O A B C)
(hH_orthocenter: is_orthocenter H A B C)
(hA_angle: α = 180 - β - γ):
∃ δ φ ψ : ℝ, δ = 20 ∧ φ = 10 ∧ ψ = 150 := by
  sorry

end angles_in_triangle_OHC_l269_269022


namespace teacher_budget_shortage_l269_269517

theorem teacher_budget_shortage :
  let last_year_budget := 6
  let this_year_allocation := 50
  let grant := 20
  let notebooks_cost := 18
  let notebooks_discount := 0.10 * notebooks_cost
  let notebooks_final_cost := notebooks_cost - notebooks_discount
  let pens_cost := 27
  let pens_discount := 0.05 * pens_cost
  let pens_final_cost := pens_cost - pens_discount
  let art_supplies_cost := 35
  let folders_cost := 15
  let folders_voucher := 5
  let folders_final_cost := folders_cost - folders_voucher
  let total_expense := notebooks_final_cost + pens_final_cost + art_supplies_cost + folders_final_cost
  let total_budget := last_year_budget + this_year_allocation + grant
  total_budget - total_expense = -11.85 := 
by
  let last_year_budget := 6
  let this_year_allocation := 50
  let grant := 20
  let notebooks_cost := 18
  let notebooks_discount := 0.10 * notebooks_cost
  let notebooks_final_cost := notebooks_cost - notebooks_discount
  let pens_cost := 27
  let pens_discount := 0.05 * pens_cost
  let pens_final_cost := pens_cost - pens_discount
  let art_supplies_cost := 35
  let folders_cost := 15
  let folders_voucher := 5
  let folders_final_cost := folders_cost - folders_voucher
  let total_expense := notebooks_final_cost + pens_final_cost + art_supplies_cost + folders_final_cost
  let total_budget := last_year_budget + this_year_allocation + grant
  have h : total_budget - total_expense = -11.85 := sorry
  exact h

end teacher_budget_shortage_l269_269517


namespace sum_of_coefficients_l269_269293

theorem sum_of_coefficients (n : ℕ) (h₀ : n > 0) (h₁ : Nat.choose n 2 = Nat.choose n 7) : 
  (1 - 2 : ℝ) ^ n = -1 := by
  -- Proof
  sorry

end sum_of_coefficients_l269_269293


namespace complex_conjugate_l269_269775

theorem complex_conjugate (z : ℂ) : 
  (i : ℂ)² = -1 → (i : ℂ)⁵ = i → 
  z = (2 + i) / (1 + (-1) + i) → 
  conj z = 1 + 2 * i :=
by
  intros
  sorry

end complex_conjugate_l269_269775


namespace polynomial_coefficients_l269_269824

theorem polynomial_coefficients (a b : ℝ) 
  (h1 : (let p := x^2 + a * x + b in ∀ x, polynomial.degree p = 2 ∧ polynomial.coeff (polynomial.mul (x^2 + a * x + b) (2 * x^2 - 3 * x - 1)) 3 = -5) 
  (h2 : (let p := x^2 + a * x + b in ∀ x, polynomial.degree p = 2 ∧ polynomial.coeff (polynomial.mul (x^2 + a * x + b) (2 * x^2 - 3 * x - 1)) 2 = -6)) :
  a = -1 ∧ b = -4 :=
begin
  sorry
end

end polynomial_coefficients_l269_269824


namespace arithmetic_sequence_a12_l269_269283

theorem arithmetic_sequence_a12
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arithmetic : ∀ n, a(n + 1) = a n + d)
  (h_a4 : a 4 = 1)
  (h_a7_a9 : a 7 + a 9 = 16) :
  a 12 = 15 := sorry

end arithmetic_sequence_a12_l269_269283


namespace possible_values_PR_l269_269694

theorem possible_values_PR (PQ QR RS SP PR : ℕ) (hPQ : PQ = 7) (hQR : QR = 15) 
  (hRS : RS = 7) (hSP : SP = 8) (hPR_int : ∃ n : ℕ, n = PR) : 
  PR ∈ {9, 10, 11, 12, 13} :=
by
  sorry

end possible_values_PR_l269_269694


namespace Namjoon_has_greater_sum_l269_269036

theorem Namjoon_has_greater_sum : 
  let jimin_sum := 1 + 7
  let namjoon_sum := 6 + 3
  namjoon_sum > jimin_sum :=
by
  let jimin_sum := 1 + 7
  let namjoon_sum := 6 + 3
  have sum_jimin : jimin_sum = 8 := by rfl
  have sum_namjoon : namjoon_sum = 9 := by rfl
  show namjoon_sum > jimin_sum from sorry

end Namjoon_has_greater_sum_l269_269036


namespace problem1_problem2_l269_269310

-- First Problem: General term of the sequence c_n
theorem problem1 
  (a b : ℕ → ℕ) (h : ∀ n, b (n + 1) = a (n + 1) * b n / (a n + 3 * b n))
  (c : ℕ → ℕ) (hc : ∀ n, c n = a n / b n)
  (c1 : c 1 = 1):
  ∀ n, c n = 3 * n - 2 :=
sorry

-- Second Problem: Sum of the first n terms of the sequence a_n
theorem problem2 
  (a b : ℕ → ℕ) (h : ∀ n, b (n + 1) = a (n + 1) * b n / (a n + 3 * b n))
  (c : ℕ → ℕ) 
  (h_geom : ∀ n, b n = (1 / 2)^(n - 1))
  (h_c : ∀ n, c n = 3 * n - 2)
  (a_def : ∀ n, a n = b n * c n):
  ∀ n, (finset.range n).sum a = 8 - (3 * n + 4) / 2^(n - 1) :=
sorry

end problem1_problem2_l269_269310


namespace terminating_fraction_count_l269_269578

open Int

theorem terminating_fraction_count :
  (Finset.card (Finset.filter (λ n, 525 ∣ n) (Finset.range 525))) = 24 :=
by
  sorry

end terminating_fraction_count_l269_269578


namespace centroid_square_distance_l269_269791

theorem centroid_square_distance (A B C O M : Point) (hM : M = centroid A B C) :
  dist_squared O M = 
    (1 / 3) * (dist_squared O A + dist_squared O B + dist_squared O C) -
    (1 / 9) * (dist_squared A B + dist_squared B C + dist_squared A C) :=
by
  sorry

end centroid_square_distance_l269_269791


namespace triangle_similarity_iff_l269_269085

theorem triangle_similarity_iff (a b c a1 b1 c1 : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (a1_pos : 0 < a1) (b1_pos : 0 < b1) (c1_pos : 0 < c1) :
  (Real.sqrt (a * a1) + Real.sqrt (b * b1) + Real.sqrt (c * c1) = Real.sqrt ((a + b + c) * (a1 + b1 + c1))) ↔ 
  (a / a1 = b / b1 ∧ b / b1 = c / c1) :=
sorry

end triangle_similarity_iff_l269_269085


namespace greatest_q_minus_r_l269_269435

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1013 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 39) := 
by
  sorry

end greatest_q_minus_r_l269_269435


namespace hunting_season_fraction_l269_269350

noncomputable def fraction_of_year_hunting_season (hunting_times_per_month : ℕ) 
    (deers_per_hunt : ℕ) (weight_per_deer : ℕ) (fraction_kept : ℚ) 
    (total_weight_kept : ℕ) : ℚ :=
  let total_yearly_weight := total_weight_kept * 2
  let weight_per_hunt := deers_per_hunt * weight_per_deer
  let total_hunts_per_year := total_yearly_weight / weight_per_hunt
  let total_months_hunting := total_hunts_per_year / hunting_times_per_month
  let fraction_of_year := total_months_hunting / 12
  fraction_of_year

theorem hunting_season_fraction : 
  fraction_of_year_hunting_season 6 2 600 (1 / 2 : ℚ) 10800 = 1 / 4 := 
by
  simp [fraction_of_year_hunting_season]
  sorry

end hunting_season_fraction_l269_269350


namespace calculate_expression_l269_269550

theorem calculate_expression : 6^3 - 5 * 7 + 2^4 = 197 := 
by
  -- Generally, we would provide the proof here, but it's not required.
  sorry

end calculate_expression_l269_269550


namespace problem1_problem2_l269_269540

variable (a b : ℝ)

-- Proof problem for Question 1
theorem problem1 : 2 * a * (a^2 - 3 * a - 1) = 2 * a^3 - 6 * a^2 - 2 * a :=
by sorry

-- Proof problem for Question 2
theorem problem2 : (a^2 * b - 2 * a * b^2 + b^3) / b - (a + b)^2 = -4 * a * b :=
by sorry

end problem1_problem2_l269_269540


namespace limit_is_zero_l269_269886

noncomputable def limit_function (x : ℝ) : ℝ :=
  (real.cbrt (8 + 3 * x - x^2) - 2) / real.cbrt (x^2 + x^3)

theorem limit_is_zero : filter.tendsto limit_function (nhds 0) (nhds 0) :=
by
  sorry

end limit_is_zero_l269_269886


namespace midterm_exam_2022_option_probabilities_l269_269710

theorem midterm_exam_2022_option_probabilities :
  let no_option := 4
  let prob_distribution := (1 : ℚ) / 3
  let combs_with_4_correct := 1
  let combs_with_3_correct := 4
  let combs_with_2_correct := 6
  let prob_4_correct := prob_distribution
  let prob_3_correct := prob_distribution / combs_with_3_correct
  let prob_2_correct := prob_distribution / combs_with_2_correct
  
  let prob_B_correct := combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct
  let prob_C_given_event_A := combs_with_3_correct * prob_3_correct / (combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct)
  
  (prob_B_correct > 1 / 2) ∧ (prob_C_given_event_A = 1 / 3) :=
by 
  sorry

end midterm_exam_2022_option_probabilities_l269_269710


namespace crayons_left_l269_269818

theorem crayons_left (initial : ℕ) (given_away : ℕ) (lost : ℕ) 
  (initial_eq : initial = 440) (given_away_eq : given_away = 111) (lost_eq : lost = 106) :
  initial - given_away - lost = 223 :=
by {
  rw [initial_eq, given_away_eq, lost_eq],
  norm_num,
  sorry -- proof will go here
}

end crayons_left_l269_269818


namespace complex_conjugate_of_z_l269_269788

def i : ℂ := complex.I

def z : ℂ := (2 + i) / (1 + i^2 + i^5)

theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * i :=
by {
    unfold z,
    -- The next few statements can formalize simplifying the denominator, but we skip the detailed proof by putting sorry.
    sorry
}

end complex_conjugate_of_z_l269_269788


namespace chocolates_vs_gums_l269_269732

theorem chocolates_vs_gums 
    (c g : ℝ) 
    (Kolya_claim : 2 * c > 5 * g) 
    (Sasha_claim : ¬ ( 3 * c > 8 * g )) : 
    7 * c ≤ 19 * g := 
sorry

end chocolates_vs_gums_l269_269732


namespace number_of_integer_length_chords_l269_269279

theorem number_of_integer_length_chords {P O : Point} (r : ℝ) (d : ℝ)
(hP : dist P O = d) (hC : r = 15) (hD : d = 9) :
  ∃ chord_lengths : set ℝ, (∀ l ∈ chord_lengths, l ∈ ℤ) ∧ (chord_lengths = {24, 25, 26, 27, 28, 29, 30}) ∧
  (cardinality chord_lengths = 12) :=
sorry

end number_of_integer_length_chords_l269_269279


namespace decreasing_interval_of_function_l269_269105

noncomputable def y (x : ℝ) : ℝ := (3 / Real.pi) ^ (x ^ 2 + 2 * x - 3)

theorem decreasing_interval_of_function :
  ∀ x ∈ Set.Ioi (-1 : ℝ), ∃ ε > 0, ∀ δ > 0, δ ≤ ε → y (x - δ) > y x :=
by
  sorry

end decreasing_interval_of_function_l269_269105


namespace loraine_wax_usage_l269_269801

/-
Loraine makes wax sculptures of animals. Large animals take eight sticks of wax, medium animals take five sticks, and small animals take three sticks.
She made twice as many small animals as large animals, and four times as many medium animals as large animals. She used 36 sticks of wax for small animals.
Prove that Loraine used 204 sticks of wax to make all the animals.
-/

theorem loraine_wax_usage :
  ∃ (L M S : ℕ), (S = 2 * L) ∧ (M = 4 * L) ∧ (3 * S = 36) ∧ (8 * L + 5 * M + 3 * S = 204) :=
by {
  sorry
}

end loraine_wax_usage_l269_269801


namespace evaluate_fractional_exponent_l269_269246

theorem evaluate_fractional_exponent : 64^(2/3 : ℝ) = 16 := by
  have h1 : (64 : ℝ) = 2^6 := by
    norm_num
  rw [h1]
  have h2 : (2^6 : ℝ)^(2/3) = 2^(6 * (2/3)) := by
    rw [← Real.rpow_mul (by norm_num : 0 ≤ 2)] -- Using exponent properties
  rw [h2]
  calc 2^(6 * (2/3)) = 2^4 : by congr; ring
                ...  = 16  : by norm_num

end evaluate_fractional_exponent_l269_269246


namespace complex_conjugate_of_z_l269_269786

def i : ℂ := complex.I

def z : ℂ := (2 + i) / (1 + i^2 + i^5)

theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * i :=
by {
    unfold z,
    -- The next few statements can formalize simplifying the denominator, but we skip the detailed proof by putting sorry.
    sorry
}

end complex_conjugate_of_z_l269_269786


namespace oil_amount_in_liters_l269_269955

theorem oil_amount_in_liters (b v : ℕ) (hb : b = 20) (hv : v = 200) :
  (b * v) / 1000 = 4 :=
by
  have h1 : b * v = 4000 := by
    rw [hb, hv]
    exact rfl
  rw [h1]
  norm_num

end oil_amount_in_liters_l269_269955


namespace product_of_rows_is_minus_one_l269_269367

theorem product_of_rows_is_minus_one {a b : Fin 100 → ℝ}
  (h_distinct_a : Function.Injective a)
  (h_distinct_b : Function.Injective b)
  (h_columns_product : ∀ j : Fin 100, (∏ i : Fin 100, a i + b j) = 1) :
  ∀ i : Fin 100, (∏ j : Fin 100, a i + b j) = -1 :=
sorry

end product_of_rows_is_minus_one_l269_269367


namespace bridget_heavier_than_martha_l269_269219

def bridget_weight := 39
def martha_weight := 2

theorem bridget_heavier_than_martha :
  bridget_weight - martha_weight = 37 :=
by
  sorry

end bridget_heavier_than_martha_l269_269219


namespace correct_integer_with_7_divisors_l269_269210

theorem correct_integer_with_7_divisors (n : ℕ) (p : ℕ) (h_prime : Prime p) 
  (h_3_divisors : ∃ (d : ℕ), d = 3 ∧ n = p^2) : n = 4 :=
by
-- Proof omitted
sorry

end correct_integer_with_7_divisors_l269_269210


namespace salesperson_commission_l269_269191

noncomputable def commission (sale_price : ℕ) (rate : ℚ) : ℚ :=
  rate * sale_price

noncomputable def total_commission (machines_sold : ℕ) (first_rate : ℚ) (second_rate : ℚ) (sale_price : ℕ) : ℚ :=
  let first_commission := commission sale_price first_rate * 100
  let second_commission := commission sale_price second_rate * (machines_sold - 100)
  first_commission + second_commission

theorem salesperson_commission :
  total_commission 130 0.03 0.04 10000 = 42000 := by
  sorry

end salesperson_commission_l269_269191


namespace P_desert_but_not_Coffee_is_0_15_l269_269843

-- Define the relevant probabilities as constants
def P_desert_and_coffee := 0.60
def P_not_desert := 0.2500000000000001
def P_desert := 1 - P_not_desert
def P_desert_but_not_coffee := P_desert - P_desert_and_coffee

-- The theorem to prove that the probability of ordering dessert but not coffee is 0.15
theorem P_desert_but_not_Coffee_is_0_15 :
  P_desert_but_not_coffee = 0.15 :=
by 
  -- calculation steps can be filled in here eventually
  sorry

end P_desert_but_not_Coffee_is_0_15_l269_269843


namespace trapezoid_diagonals_l269_269822

theorem trapezoid_diagonals {BC AD AB CD AC BD : ℝ} (h b1 b2 : ℝ) 
  (hBC : BC = b1) (hAD : AD = b2) (hAB : AB = h) (hCD : CD = h) 
  (hAC : AC^2 = AB^2 + BC^2) (hBD : BD^2 = CD^2 + AD^2) :
  BD^2 - AC^2 = b2^2 - b1^2 := 
by 
  -- proof is omitted
  sorry

end trapezoid_diagonals_l269_269822


namespace speed_of_journey_l269_269938

-- Define the conditions
def journey_time : ℕ := 10
def journey_distance : ℕ := 200
def half_journey_distance : ℕ := journey_distance / 2

-- Define the hypothesis that the journey is split into two equal parts, each traveled at the same speed
def equal_speed (v : ℕ) : Prop :=
  (half_journey_distance / v) + (half_journey_distance / v) = journey_time

-- Prove the speed v is 20 km/hr given the conditions
theorem speed_of_journey : ∃ v : ℕ, equal_speed v ∧ v = 20 :=
by
  have h : equal_speed 20 := sorry
  exact ⟨20, h, rfl⟩

end speed_of_journey_l269_269938


namespace y_time_to_complete_work_l269_269481

-- Definitions of the conditions
def work_rate_x := 1 / 40
def work_done_by_x_in_8_days := 8 * work_rate_x
def remaining_work := 1 - work_done_by_x_in_8_days
def y_completion_time := 32
def work_rate_y := remaining_work / y_completion_time

-- Lean theorem
theorem y_time_to_complete_work :
  y_completion_time * work_rate_y = 1 →
  (1 / work_rate_y = 40) :=
by
  sorry

end y_time_to_complete_work_l269_269481


namespace present_age_of_son_l269_269150

theorem present_age_of_son:
  ∀ S F : ℕ, F = S + 22 ∧ F + 2 = 2 * (S + 2) → S = 20 := 
by {
  intros S F,
  intro h,
  sorry
}

end present_age_of_son_l269_269150


namespace parabola_line_intersect_l269_269278

theorem parabola_line_intersect (a : ℝ) (b : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, (y = a * x^2) ↔ (y = 2 * x - 3) → (x, y) = (1, -1)) :
  a = -1 ∧ b = -1 ∧ ((x, y) = (-3, -9) ∨ (x, y) = (1, -1)) := by
  sorry

end parabola_line_intersect_l269_269278


namespace incorrect_options_l269_269320

variable (a b : ℚ) (h : a / b = 5 / 6)

theorem incorrect_options :
  (2 * a - b ≠ b * 6 / 4) ∧
  (a + 3 * b ≠ 2 * a * 19 / 10) :=
by
  sorry

end incorrect_options_l269_269320


namespace square_perimeter_equals_66_88_l269_269155

noncomputable def circle_perimeter : ℝ := 52.5

noncomputable def circle_radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def circle_diameter (r : ℝ) : ℝ := 2 * r

noncomputable def square_side_length (d : ℝ) : ℝ := d

noncomputable def square_perimeter (s : ℝ) : ℝ := 4 * s

theorem square_perimeter_equals_66_88 :
  square_perimeter (square_side_length (circle_diameter (circle_radius circle_perimeter))) = 66.88 := 
by
  -- Placeholder for the proof
  sorry

end square_perimeter_equals_66_88_l269_269155


namespace min_re_z4_re_z4_l269_269675

theorem min_re_z4_re_z4 (z : ℂ) (h : z.re ≠ 0) : 
  ∃ t : ℝ, (t = (z.im / z.re)) ∧ ((1 - 6 * (t^2) + (t^4)) = -8) := sorry

end min_re_z4_re_z4_l269_269675


namespace sequence_properties_l269_269294

variable (a_n b_n c_n : ℕ → ℕ) (S_n : ℕ → ℕ)

theorem sequence_properties :
  (∀ n, a_n n = n + 1) ∧ (∀ n, b_n n = 2^(n-1)) ∧
  (∀ n, c_n n = (a_n n) / (b_n n)) ∧ 
  (∀ n, S_n n = 6 - (n + 3) / 2^(n-1)) :=
by
  -- Definitions based on given conditions
  let a_n : ℕ → ℕ := λ n, n + 1
  let b_n : ℕ → ℕ := λ n, 2^(n-1)
  let c_n : ℕ → ℕ := λ n, (a_n n) / (b_n n)
  let S_n : ℕ → ℕ := λ n, 6 - (n + 3) / 2^(n-1)

  -- prove the properties
  sorry

end sequence_properties_l269_269294


namespace largest_possible_percent_error_l269_269376

theorem largest_possible_percent_error 
  (r : ℝ) (delta : ℝ) (h_r : r = 15) (h_delta : delta = 0.1) : 
  ∃(error : ℝ), error = 0.21 :=
by
  -- The proof would go here
  sorry

end largest_possible_percent_error_l269_269376


namespace regular_2008_gon_min_value_zero_l269_269188

noncomputable def regular_2008_gon_min_value (p : ℕ) (hp : Nat.Prime p) : ℂ :=
  let vertices := {n // n < 2008}.map (λ k, (p + 1) * exp (2 * real.pi * complex.I * k / 2008))
  let S := ∏ k in finset.range 1004, vertices (2 * k)
  let T := ∏ k in finset.range 1004, vertices (2 * k + 1)
  S - T

theorem regular_2008_gon_min_value_zero (p : ℕ) (hp : Nat.Prime p) :
  |regular_2008_gon_min_value p hp| = 0 :=
sorry

end regular_2008_gon_min_value_zero_l269_269188


namespace complex_conjugate_l269_269769

theorem complex_conjugate (z : ℂ) : 
  (i : ℂ)² = -1 → (i : ℂ)⁵ = i → 
  z = (2 + i) / (1 + (-1) + i) → 
  conj z = 1 + 2 * i :=
by
  intros
  sorry

end complex_conjugate_l269_269769


namespace bernoulli_trial_probability_7_successes_l269_269896

theorem bernoulli_trial_probability_7_successes :
  let n := 7
  let k := 7
  let p := (2 : ℝ) / 7
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (128 / 823543) :=
by
  sorry

end bernoulli_trial_probability_7_successes_l269_269896


namespace perpendicular_lines_l269_269649

theorem perpendicular_lines 
  (α β : set Point)
  (l m n : Line)
  (h_planes_perpendicular : mutually_perpendicular_planes α β)
  (h_intersect_line : intersection_line α β l)
  (h_m_parallel_alpha : parallel_to_plane m α)
  (h_n_perpendicular_beta : perpendicular_to_plane n β) : 
  perpendicular n l :=
sorry

end perpendicular_lines_l269_269649


namespace math_problem_l269_269319

theorem math_problem (c d : ℝ) (h1 : 120^c = 2) (h2 : 120^d = 3) :
  24^((1 - c - d) / (2 * (1 - d))) = 2 * Real.sqrt 5 :=
sorry

end math_problem_l269_269319


namespace partition_quadrilateral_into_trapezoids_l269_269094

noncomputable def sum_of_angles (A B C D : ℝ) : ℝ :=
  A + B + C + D

theorem partition_quadrilateral_into_trapezoids (A B C D : ℝ)
  (h : sum_of_angles A B C D = 360) : 
  ∃ t1 t2 t3 : set ℝ, 
  (t1 ⊆ {a | a ∈ (A:ℝ)}) ∧
  (t2 ⊆ {b | b ∈ (B:ℝ)}) ∧
  (t3 ⊆ {c | c ∈ (C:ℝ)})
  sorry

end partition_quadrilateral_into_trapezoids_l269_269094


namespace half_angle_quadrant_second_quadrant_l269_269290

theorem half_angle_quadrant_second_quadrant
  (θ : Real)
  (h1 : π < θ ∧ θ < 3 * π / 2) -- θ is in the third quadrant
  (h2 : Real.cos (θ / 2) < 0) : -- cos (θ / 2) < 0
  π / 2 < θ / 2 ∧ θ / 2 < π := -- θ / 2 is in the second quadrant
sorry

end half_angle_quadrant_second_quadrant_l269_269290


namespace sum_numerator_denominator_probability_l269_269576

theorem sum_numerator_denominator_probability :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  let (a1, a2, a3, a4, a5) := (0, 0, 0, 0, 0) in
  let remaining_set := S \ {a1, a2, a3, a4, a5};
  let (b1, b2) := (0, 0) in
  let p := 1 / 2 in
  sum of the numerator and denominator of p in lowest terms = 3 :=
  sorry

end sum_numerator_denominator_probability_l269_269576


namespace bernoulli_trial_probability_7_successes_l269_269892

theorem bernoulli_trial_probability_7_successes :
  let n := 7
  let k := 7
  let p := (2 : ℝ) / 7
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (128 / 823543) :=
by
  sorry

end bernoulli_trial_probability_7_successes_l269_269892


namespace expected_interval_is_correct_l269_269485

-- Define the travel times via northern and southern routes
def travel_time_north : ℝ := 17
def travel_time_south : ℝ := 11

-- Define the average time difference between train arrivals
noncomputable def avg_time_diff : ℝ := 1.25

-- The average time difference for traveling from home to work versus work to home
noncomputable def time_diff_home_to_work : ℝ := 1

-- Define the expected interval between trains
noncomputable def expected_interval_between_trains := 3

-- Proof problem statement
theorem expected_interval_is_correct :
  ∃ (T : ℝ), (T = expected_interval_between_trains)
  → (travel_time_north - travel_time_south + 2 * avg_time_diff = time_diff_home_to_work)
  → (T = 3) := 
by
  use 3 
  intro h1 h2
  sorry

end expected_interval_is_correct_l269_269485


namespace trig_identity_A_l269_269790

theorem trig_identity_A : 
  let A := (Real.cos (10 * Real.pi / 180))^2  + 
           (Real.cos (50 * Real.pi / 180))^2 -
           Real.sin (40 * Real.pi / 180) * Real.sin (80 * Real.pi / 180)
  in 100 * A = 75 := 
by
  sorry

end trig_identity_A_l269_269790


namespace period_fraction_sum_nines_l269_269439

theorem period_fraction_sum_nines (q : ℕ) (p : ℕ) (N N1 N2 : ℕ) (n : ℕ) (t : ℕ) 
  (hq_prime : Nat.Prime q) (hq_gt_5 : q > 5) (hp_lt_q : p < q)
  (ht_eq_2n : t = 2 * n) (h_period : 10^t ≡ 1 [MOD q])
  (hN_eq_concat : (N = N1 * 10^n + N2) ∧ (N % 10^n = N2))
  : N1 + N2 = (10^n - 1) := 
sorry

end period_fraction_sum_nines_l269_269439


namespace function_machine_output_l269_269705

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 25 then step1 - 7 else step1 + 10
  step2

theorem function_machine_output : function_machine 15 = 38 :=
by
  sorry

end function_machine_output_l269_269705


namespace total_rubber_bands_l269_269656

theorem total_rubber_bands (harper_bands : ℕ) (brother_bands: ℕ):
  harper_bands = 15 →
  brother_bands = harper_bands - 6 →
  harper_bands + brother_bands = 24 :=
by
  intros h1 h2
  sorry

end total_rubber_bands_l269_269656


namespace james_meditation_sessions_l269_269029

theorem james_meditation_sessions (minutes_per_session : ℕ) (hours_per_week : ℕ) (days_per_week : ℕ) (h1 : minutes_per_session = 30) (h2 : hours_per_week = 7) (h3 : days_per_week = 7) : 
  (hours_per_week * 60 / days_per_week / minutes_per_session) = 2 := 
by 
  sorry

end james_meditation_sessions_l269_269029


namespace diameter_is_twice_radius_l269_269663

theorem diameter_is_twice_radius {r d : ℝ} (h : d = 2 * r) : d = 2 * r :=
by {
  sorry
}

end diameter_is_twice_radius_l269_269663


namespace find_a_plus_b_l269_269323

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 1 = a - b) 
  (h2 : 5 = a - b / 5) : a + b = 11 :=
by
  sorry

end find_a_plus_b_l269_269323


namespace complex_conjugate_of_z_l269_269763

theorem complex_conjugate_of_z : 
  let z := (2 + complex.i) / (1 + complex.i^2 + complex.i^5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269763


namespace max_value_of_f_l269_269573

noncomputable def f (x : ℝ) : ℝ := 5 * x - 2 * x^3

theorem max_value_of_f : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f x = f (Real.sqrt (5 / 6)) :=
begin
  sorry
end

end max_value_of_f_l269_269573


namespace algorithm_characteristics_l269_269879

theorem algorithm_characteristics (finiteness : Prop) (definiteness : Prop) (output_capability : Prop) (unique : Prop) 
  (h1 : finiteness = true) 
  (h2 : definiteness = true) 
  (h3 : output_capability = true) 
  (h4 : unique = false) : 
  incorrect_statement = unique := 
by
  sorry

end algorithm_characteristics_l269_269879


namespace radius_of_circle_proof_l269_269494

noncomputable def radius_of_circle (x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : ℝ :=
  r

theorem radius_of_circle_proof (r x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : r = 10 :=
by
  sorry

end radius_of_circle_proof_l269_269494


namespace new_median_of_collection_l269_269495

theorem new_median_of_collection (s : List ℕ)
  (h_length : s.length = 6)
  (h_positive : ∀ x ∈ s, x > 0)
  (h_mean : (s.sum : ℚ) / 6 = 5.5)
  (h_mode : ∀ a : ℕ, s.count a ≤ s.count 4)
  (h_unique_mode : s.count 4 > s.count 6)
  (h_median : ∀ sorted_s, sorted_s = s.sorted → sorted_s.nthLe 2 (by linarith) = 6 ∧ sorted_s.nthLe 3 (by linarith) = 6) :
  let new_s := (10 :: s).sorted in
  new_s.nthLe 3 (by { rw [List.length_cons, h_length, Nat.succ_eq_add_one, Nat.add_comm, Nat.add_one], norm_num }) = 6 :=
sorry

end new_median_of_collection_l269_269495


namespace function_equivalence_l269_269877

theorem function_equivalence (x : ℝ) (h : x ≤ 0) :
  sqrt (-2 * x ^ 3) = -x * sqrt (-2 * x) :=
sorry

end function_equivalence_l269_269877


namespace solve_fractional_eq_l269_269413

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : 
  (2 / (x - 1) = 1 / x) ↔ (x = -1) :=
by 
  sorry

end solve_fractional_eq_l269_269413


namespace probability_third_attempt_success_l269_269507

noncomputable def P_xi_eq_3 : ℚ :=
  (4 / 5) * (3 / 4) * (1 / 3)

theorem probability_third_attempt_success :
  P_xi_eq_3 = 1 / 5 := by
  sorry

end probability_third_attempt_success_l269_269507


namespace smallest_x_plus_y_l269_269608

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269608


namespace fraction_of_new_releases_l269_269151

-- Definitions based on conditions
def total_books := 100
def hist_fiction_books := 40 -- 40% of total_books
def other_books := total_books - hist_fiction_books

def hist_fiction_new_releases := 0.40 * hist_fiction_books
def other_genres_new_releases := 0.20 * other_books

def total_new_releases := hist_fiction_new_releases + other_genres_new_releases

-- Lean 4 statement for the proof
theorem fraction_of_new_releases :
  (hist_fiction_new_releases / total_new_releases) = (4 / 7) :=
by
  have h1 : hist_fiction_new_releases = 16 := by sorry
  have h2 : other_genres_new_releases = 12 := by sorry
  have h3 : total_new_releases = 28 := by sorry
  suffices : 16 / 28 = 4 / 7, from this
  sorry

end fraction_of_new_releases_l269_269151


namespace find_ordered_pair_correct_l269_269100

noncomputable def find_ordered_pair (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0)
  (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∨ x = 2 * d)) : (ℝ × ℝ) :=
  -- We need to prove this statement
  (c, d) = (1 / 2, -1 / 2)

-- Lean statement to formalize our proof goal
theorem find_ordered_pair_correct (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0)
  (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∨ x = 2 * d)) : (c, d) = (1 / 2, -1 / 2) :=
by
  -- Proof steps would go here
  sorry

end find_ordered_pair_correct_l269_269100


namespace round_3_8963_l269_269396

def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  (Float.toIEEE754 x * 100).round * 0.01

theorem round_3_8963 : round_to_nearest_hundredth 3.8963 = 3.90 :=
by sorry

end round_3_8963_l269_269396


namespace complex_conjugate_of_z_l269_269787

def i : ℂ := complex.I

def z : ℂ := (2 + i) / (1 + i^2 + i^5)

theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * i :=
by {
    unfold z,
    -- The next few statements can formalize simplifying the denominator, but we skip the detailed proof by putting sorry.
    sorry
}

end complex_conjugate_of_z_l269_269787


namespace converse_not_universal_but_true_for_circles_l269_269160

-- Definitions based on conditions
def symmetrical (a b : Type) := a = b
def equal_figures (a b : Type) := a = b

-- Lean 4 statement of the problem
theorem converse_not_universal_but_true_for_circles (fig : Type) :
  ¬ (∀ (a b : fig), equal_figures a b → ∃ (axis : Type), symmetrical a b) ∧
  (∀ (c1 c2 : circle), equal_figures c1 c2 → ∃ (axis : Type), symmetrical c1 c2) :=
sorry

end converse_not_universal_but_true_for_circles_l269_269160


namespace dividend_percentage_l269_269504

/-- 
A man invested Rs. 14,400 in Rs. 100 shares of a company at a 20% premium. 
The company declares a certain percentage as dividend at the end of the year, 
and he gets Rs. 720. Prove that the dividend percentage is 6%. 
--/
theorem dividend_percentage
  (investment : ℝ)
  (face_value : ℝ)
  (premium : ℝ)
  (total_dividend : ℝ)
  (h_investment : investment = 14400)
  (h_face_value : face_value = 100)
  (h_premium : premium = 0.20)
  (h_total_dividend : total_dividend = 720) :
  let cost_per_share := face_value * (1 + premium) in
  let number_of_shares := investment / cost_per_share in
  let dividend_per_share := total_dividend / number_of_shares in
  let dividend_percentage := (dividend_per_share / face_value) * 100 in
  dividend_percentage = 6 :=
by
  sorry

end dividend_percentage_l269_269504


namespace sequence_geometric_S_eq_4a_f_at_pi_over_6_l269_269592

-- Define the sequence and sum
def a (n : ℕ) : ℕ := 
  if h : n = 0 then 0 else
  ℕ.recOn (n.pred) 1 (λ n' ih, (n + 2) * ih / n)

def S (n : ℕ) : ℕ := 
  if n = 0 then 0 else ∑ i in Finset.range n, a (i + 1)

-- Prove the two parts for the sequence {a_n}
theorem sequence_geometric (n : ℕ) (h : n > 0) :
  ∃ r : ℝ, (S (n + 1) / (n + 1) = 2 * (S n / n)) ∧ (S 1 = 1) ∧ (r ∈ [2, 2]) :=
by sorry

theorem S_eq_4a (n : ℕ) (h : n > 0) : S (n + 1) = 4 * a n :=
by sorry

-- Define the vectors and function
def a_vec (x : ℝ) : ℝ × ℝ := 
  (Real.cos (3 * x / 2), Real.sin (3 * x / 2))

def b_vec (x : ℝ) : ℝ × ℝ := 
  (Real.cos (x / 2), -Real.sin (x / 2))

def f (x m : ℝ) : ℝ := 
  let a := a_vec x
  let b := b_vec x
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a_plus_b := Real.sqrt (a.1 + b.1)^2 + (a.2 + b.2)^2
  dot_product - m * magnitude_a_plus_b + 1

theorem f_at_pi_over_6 (m : ℝ) (h : m = 0) : f (Real.pi / 6) m = 3 / 2 :=
by sorry

end sequence_geometric_S_eq_4a_f_at_pi_over_6_l269_269592


namespace shift_graph_to_even_l269_269093

noncomputable def shifted_function_even (x : ℝ) : Prop :=
  even_function := -cos (2 * x)

theorem shift_graph_to_even (x : ℝ) :
  even_function (sin (2 * (x + (5/12) * π) + (2/3) * π)) :=
sorry

end shift_graph_to_even_l269_269093


namespace tan_sum_of_angles_problem_l269_269299

theorem tan_sum_of_angles_problem 
(O : Point)
(α : ℝ)
(P : Point)
(hP : P = (2, 1))
(hO : O = (0, 0))
(initial_side : α.initial_side = positive_half_axis_of x_axis)
(terminal_side : terminal_side_passing_through P) :
  Real.tan (2 * α + Real.pi / 4) = -7 := 
sorry

end tan_sum_of_angles_problem_l269_269299


namespace ellipse_properties_l269_269593

theorem ellipse_properties :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a^2 = b^2 + 2 ∧ 3/2^2 / (a ^ 2) + (-1/2)^2 / (b ^ 2) = 1 ∧
  (∃ (x y : ℝ), (x^2 / 3 + y^2 = 1) ∧ (x + y + 4 = 0) ∧
    (∃ (d : ℝ), d = real.sqrt 2 ∧ (x, y) = (-3/2, -1/2))) :=
sorry

end ellipse_properties_l269_269593


namespace pyarelal_loss_l269_269531

variable (P : ℝ)
variable (total_loss : ℝ)
variable (ashok_ratio : ℝ)
variable (pyarelal_share : ℝ)

-- Conditions
def ashok_capital := (1/9) * P
def total_capital := P + ashok_capital
def loss_ratio := P / total_capital
def ashok_share := total_loss - pyarelal_share

-- Statement
theorem pyarelal_loss (h1 : ashok_ratio = 1/9)
                      (h2 : total_loss = 900)
                      (h3 : ashok_capital = (1/9) * P)
                      (h4 : total_capital = (10/9) * P)
                      (h5 : loss_ratio * total_loss = pyarelal_share)
                      : pyarelal_share = 810 := by
  sorry

end pyarelal_loss_l269_269531


namespace find_w_l269_269911

open Real

def vector_in_yz_plane (w : ℝ × ℝ × ℝ) : Prop := w.1 = 0
def unit_vector (w : ℝ × ℝ × ℝ) : Prop := w.2^2 + w.3^2 = 1
def angle_with_vector (w : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) (θ : ℝ) : Prop :=
  (w.1 * v.1 + w.2 * v.2 + w.3 * v.3) / ((sqrt (w.1^2 + w.2^2 + w.3^2)) * (sqrt (v.1^2 + v.2^2 + v.3^2))) = cos θ

def w := (0, (6 - sqrt 5) / 3, sqrt 5 / 3)

theorem find_w : 
  vector_in_yz_plane w ∧ unit_vector w ∧ 
  angle_with_vector w (1, 2, 1) (π / 6) ∧ 
  angle_with_vector w (1, 0, 3) (π / 4) :=
by {
  -- the proof should be written here, but we skip it as the task asks only for the problem statement
  sorry
}

end find_w_l269_269911


namespace number_of_people_in_group_l269_269104

-- Define the conditions as given in the problem
variables (N : ℕ) (old_weight new_weight weight_increase : ℚ)
variables (h_old : old_weight = 40)
variables (h_new : new_weight = 88)
variables (h_increase : weight_increase = 6)

-- Define the equation derived from the conditions
def group_size (N : ℕ) : Prop :=
  weight_increase * N = new_weight - old_weight

-- Prove that the number of people in the group is 8
theorem number_of_people_in_group : group_size N → N = 8 :=
by
  intro h
  rw [weight_increase, h_increase, new_weight, h_new, old_weight, h_old] at h
  sorry

end number_of_people_in_group_l269_269104


namespace shipping_cost_correct_l269_269926

noncomputable def shipping_cost (W : ℝ) : ℕ := 7 + 5 * (⌈W⌉₊ - 1)

theorem shipping_cost_correct (W : ℝ) : shipping_cost W = 5 * ⌈W⌉₊ + 2 :=
by
  sorry

end shipping_cost_correct_l269_269926


namespace sum_of_all_possible_values_of_g_51_l269_269054

def f (x : ℝ) : ℝ := 4 * x ^ 2 - 5
noncomputable def g (x : ℝ) : ℝ := 2 * x ^ 2 - Real.cos x + 2

theorem sum_of_all_possible_values_of_g_51 :
  f x = 51 → ∑ x in (finset.singleton (sqrt 14) ∪ finset.singleton (-sqrt 14)).1, g (f x) = 30 - Real.cos (sqrt 14) :=
by
  sorry

end sum_of_all_possible_values_of_g_51_l269_269054


namespace slope_of_PA_and_PB_is_constant_l269_269697

open Classical

variable {R : Type _} [LinearOrderedField R]

theorem slope_of_PA_and_PB_is_constant :
  ∀ (P A B D : R × R),
  P = (2, 1) →
  (∃ m l : R, (A.1^2 / 6 + A.2^2 / 3 = 1) ∧ (B.1^2 / 6 + B.2^2 / 3 = 1) ∧
    D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ D.2 / D.1 = 1) →
  ∀ k1 k2 : R, (k1 = (A.2 - P.2) / (A.1 - P.1)) → (k2 = (B.2 - P.2) / (B.1 - P.1)) →
  k1 * k2 = 1 / 2 :=
by {
  intros P A B D hP hex k1 k2 hk1 hk2,
  sorry
}

end slope_of_PA_and_PB_is_constant_l269_269697


namespace orthocenters_collinear_l269_269446

theorem orthocenters_collinear 
(O A B C D P: Point)
(H1 H2 H3 H4: Point)
(OB_eq_OA_eq_OD: dist O B = dist O A ∧ dist O D = dist O A)
(AP_perp_BD: ∃ (lineBD: Line), ∃ (lineAP: Line), perpendicular lineAP lineBD ∧ lineContains lineAP A ∧ lineContains lineAP P ∧ lineContains lineBD B ∧ lineContains lineBD D ∧ pointOnLine P lineBD)
(AOD_plus_BOC_180: ∠ A O D + ∠ B O C = 180)
(AOB_plus_COD_180: ∠ A O B + ∠ C O D = 180)
(ABC_collinear: collinear O B C)
(BCD_collinear: collinear O C D)
(CDA_collinear: collinear O D A):
collinear H1 H2 H3 H4 := 
sorry

end orthocenters_collinear_l269_269446


namespace complex_conjugate_of_z_l269_269782

theorem complex_conjugate_of_z :
  let z := (2 + complex.i) / (1 + complex.i ^ 2 + complex.i ^ 5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269782


namespace evaluate_product_l269_269567

theorem evaluate_product (n : ℕ) (h : n = 3) : (n-1) * n * (n+1) * (n+2) * (n+3) * (n+4) = 5040 := by
  rw [h]
  norm_num
  -- The "sorry" is used as a placeholder for the proof
  sorry

end evaluate_product_l269_269567


namespace monotonicity_and_zero_range_l269_269302

noncomputable def f (a x : ℝ) : ℝ := a * exp (2 * x) + (a - 2) * exp x - x

theorem monotonicity_and_zero_range (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → deriv (f a) x ≤ 0) ∧
  (a > 0 → 
    (∀ x : ℝ, x < real.log (1 / a) → deriv (f a) x < 0) ∧
    (∀ x : ℝ, x > real.log (1 / a) → deriv (f a) x > 0)) ∧
  (∀ b : ℝ, 
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) → 
    a ∈ set.Ioo 0 1) := by sorry

end monotonicity_and_zero_range_l269_269302


namespace perimeter_AFC_l269_269023

-- Definitions based on conditions
variables {A B C F D : Type}
variables {BC AC : Real}
variables {D_midpoint_AB : Prop}
variables {DF_perpendicular_AB : Prop}
variables {BC_def : BC = 19}
variables {AC_def : AC = 10}

-- Theorem statement
theorem perimeter_AFC :
  D_midpoint_AB ∧ DF_perpendicular_AB ∧ BC_def ∧ AC_def →
  ∃ P : Real, P = 29 :=
by
  intros h
  sorry

end perimeter_AFC_l269_269023


namespace total_earnings_first_two_weeks_l269_269148

-- Conditions
variable (x : ℝ)  -- Xenia's hourly wage
variable (earnings_first_week : ℝ := 12 * x)  -- Earnings in the first week
variable (earnings_second_week : ℝ := 20 * x)  -- Earnings in the second week

-- Xenia earned $36 more in the second week than in the first
axiom h1 : earnings_second_week = earnings_first_week + 36

-- Proof statement
theorem total_earnings_first_two_weeks : earnings_first_week + earnings_second_week = 144 := by
  -- Proof is omitted
  sorry

end total_earnings_first_two_weeks_l269_269148


namespace equal_lengths_l269_269026

theorem equal_lengths (A B X Y C F : Point)
  (parallel : ∃ (l1 l2 : Line), l1.contains A ∧ l1.contains B ∧ l2.contains F ∧ l2.contains C ∧ l1 ∥ l2) :
  dist A X = dist B Y :=
sorry

end equal_lengths_l269_269026


namespace number_of_appointments_l269_269970

-- Define the conditions
variables {hours_in_workday : ℕ} {appointments_duration : ℕ} {permit_rate : ℕ} {total_permits : ℕ}
variables (H1 : hours_in_workday = 8) (H2 : appointments_duration = 3) (H3 : permit_rate = 50) (H4: total_permits = 100)

-- Define the question as a theorem with the correct answer
theorem number_of_appointments : 
  (hours_in_workday - (total_permits / permit_rate)) / appointments_duration = 2 :=
by
  -- Proof is not required
  sorry

end number_of_appointments_l269_269970


namespace max_value_of_function_l269_269303

/-- Let y(x) = a^(2*x) + 2 * a^x - 1 for a positive real number a and x in [-1, 1].
    Prove that the maximum value of y on the interval [-1, 1] is 14 when a = 1/3 or a = 3. -/
theorem max_value_of_function (a : ℝ) (a_pos : 0 < a) (h : a = 1 / 3 ∨ a = 3) : 
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 14 := 
sorry

end max_value_of_function_l269_269303


namespace find_angle_y_l269_269709

theorem find_angle_y (m n : Line) (p₁ p₂ : Point) (α β : ℝ)
  (H_parallel : parallel m n)
  (H_angles : α = 40 ∧ β = 90) :
  ∃ (y : ℝ), y = 80 := by
  sorry

end find_angle_y_l269_269709


namespace general_term_sequence_l269_269280

noncomputable theory

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (∑ i in finset.range (n+1), a i) = 3 - 2 * a n

theorem general_term_sequence (a : ℕ → ℝ) (h : sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = (2 / 3) ^ (n - 1) :=
begin
  sorry
end

end general_term_sequence_l269_269280


namespace snowflake_weight_scientific_notation_l269_269907

-- Define the weight of a single snowflake
def snowflake_weight : ℝ := 3e-5

-- The main statement proving that snowflake_weight is correctly expressed in scientific notation
theorem snowflake_weight_scientific_notation : snowflake_weight = 3 * 10 ^ (-4) := sorry

end snowflake_weight_scientific_notation_l269_269907


namespace apples_per_box_l269_269045

theorem apples_per_box (x : ℕ) (h1 : 10 * x > 0) (h2 : 3 * (10 * x) / 4 > 0) (h3 : (10 * x) / 4 = 750) : x = 300 :=
by
  sorry

end apples_per_box_l269_269045


namespace caffeine_over_goal_l269_269451

theorem caffeine_over_goal (cups_per_day : ℕ) (mg_per_cup : ℕ) (caffeine_goal : ℕ) (total_cups : ℕ) :
  total_cups = 3 ->
  cups_per_day = 3 ->
  mg_per_cup = 80 ->
  caffeine_goal = 200 ->
  (cups_per_day * mg_per_cup) - caffeine_goal = 40 := by
  sorry

end caffeine_over_goal_l269_269451


namespace ramu_spent_on_repairs_l269_269823

theorem ramu_spent_on_repairs :
  ∃ (R : ℝ), R = 13000 ∧ 
    let initial_cost := 42000 in
    let selling_price := 66900 in
    let profit_percent := 21.636363636363637 in
    profit_percent / 100 * (initial_cost + R) = selling_price - (initial_cost + R) :=
sorry

end ramu_spent_on_repairs_l269_269823


namespace initial_people_count_l269_269182

theorem initial_people_count (left remaining total : ℕ) (h1 : left = 6) (h2 : remaining = 5) : total = 11 :=
  by
  sorry

end initial_people_count_l269_269182


namespace find_x_l269_269863

-- Define the digits used
def digits : List ℕ := [1, 4, 5]

-- Define the sum of all four-digit numbers formed
def sum_of_digits (x : ℕ) : ℕ :=
  24 * (1 + 4 + 5 + x)

-- State the theorem
theorem find_x (x : ℕ) (h : sum_of_digits x = 288) : x = 2 :=
  by
    sorry

end find_x_l269_269863


namespace bar_graph_representation_l269_269177

noncomputable theory

-- Definitions for the angles of each section
def gray_angle (x : ℝ) : ℝ := x
def black_angle (x : ℝ) : ℝ := 2 * x
def white_angle (x : ℝ) : ℝ := 6 * x

-- Total degree of a circle
def total_degrees : ℝ := 360

-- Correct choice properties
def valid_bar_graph (gray black white : ℝ) : Prop :=
  gray = 1 ∧ black = 2 ∧ white = 6

theorem bar_graph_representation :
  ∃ (x : ℝ), gray_angle x + black_angle x + white_angle x = total_degrees ∧
             valid_bar_graph (gray_angle x / x) (black_angle x / x) (white_angle x / x) :=
by
  use (40 : ℝ)
  split
  · sorry  -- Proof for angle sum = total degrees
  · sorry  -- Proof for valid bar graph proportion

end bar_graph_representation_l269_269177


namespace find_radius_of_cone_l269_269124

def slant_height : ℝ := 10
def curved_surface_area : ℝ := 157.07963267948966

theorem find_radius_of_cone
    (l : ℝ) (CSA : ℝ) (h1 : l = slant_height) (h2 : CSA = curved_surface_area) :
    ∃ r : ℝ, r = 5 := 
by
  sorry

end find_radius_of_cone_l269_269124


namespace dragons_total_games_l269_269217

noncomputable def numberOfGames (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) : ℕ :=
y + 12

theorem dragons_total_games (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) :
  numberOfGames y x h1 h2 = 90 := 
sorry

end dragons_total_games_l269_269217


namespace geometric_sequences_common_ratios_l269_269056

theorem geometric_sequences_common_ratios 
  (k m n o : ℝ)
  (a_2 a_3 b_2 b_3 c_2 c_3 : ℝ)
  (h1 : a_2 = k * m)
  (h2 : a_3 = k * m^2)
  (h3 : b_2 = k * n)
  (h4 : b_3 = k * n^2)
  (h5 : c_2 = k * o)
  (h6 : c_3 = k * o^2)
  (h7 : a_3 - b_3 + c_3 = 2 * (a_2 - b_2 + c_2))
  (h8 : m ≠ n)
  (h9 : m ≠ o)
  (h10 : n ≠ o) : 
  m + n + o = 1 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequences_common_ratios_l269_269056


namespace geometric_series_proof_l269_269996

theorem geometric_series_proof (y : ℝ) :
  ((1 + (1/3) + (1/9) + (1/27) + ∑' n : ℕ, (1 / 3^(n+1))) * 
   (1 - (1/3) + (1/9) - (1/27) + ∑' n : ℕ, ((-1)^n * (1 / 3^(n+1)))) = 
   1 + (1/y) + (1/y^2) + (∑' n : ℕ, (1 / y^(n+1)))) → y = 9 := by
  sorry

end geometric_series_proof_l269_269996


namespace smallest_m_l269_269061

-- Define the set X
def X := { n : ℕ | 1 ≤ n ∧ n ≤ 2001 }

-- Define the predicate to check if a number is a power of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Define the condition for the subset W
def condition (m : ℕ) (W : finset ℕ) : Prop := 
  (∀ u v ∈ W, u ≠ v → is_power_of_two (u + v))

-- Prove that the smallest m is 72
theorem smallest_m : ∃ m, (∀ W ⊆ X.to_finset, W.card = m → condition m W) ∧ m = 72 := 
sorry

end smallest_m_l269_269061


namespace c1_not_collinear_c2_l269_269529

def vec3 := (ℝ × ℝ × ℝ)

def a : vec3 := (1, 2, -3)
def b : vec3 := (2, -1, -1)

def scale (c : ℝ) (v : vec3) : vec3 := (c * v.1, c * v.2, c * v.3)
def add (v1 v2 : vec3) : vec3 := (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)
def sub (v1 v2 : vec3) : vec3 := (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def c1 : vec3 := add (scale 4 a) (scale 3 b)
def c2 : vec3 := sub (scale 8 a) b

theorem c1_not_collinear_c2 : ¬∃ (γ : ℝ), c1 = scale γ c2 := by
  sorry

end c1_not_collinear_c2_l269_269529


namespace proof_passenger_arrangement_l269_269487

noncomputable def combinatorial_problem_statement : ℕ :=
  let fiveChooseTwo := Nat.choose 5 2
  let threeChooseTwo := Nat.choose 3 2
  let threeFactorial := Nat.factorial 3
  let twoFactorial := Nat.factorial 2
  let fiveChooseThree := Nat.choose 5 3
in (fiveChooseTwo * threeChooseTwo * threeFactorial / twoFactorial) + (fiveChooseThree * threeFactorial)

theorem proof_passenger_arrangement : combinatorial_problem_statement = 150 := by
  sorry

end proof_passenger_arrangement_l269_269487


namespace proof_theorem_l269_269384

noncomputable def proof_problem (r AB BC : ℝ) (A B C : Type) [AddGroup A] [AddGroup B] [AddGroup C] : Prop :=
  AB = BC ∧ AB > r ∧ (BC / r) = (π / 2) -> (AB / BC = 1)

theorem proof_theorem (r AB BC : ℝ) (A B C : Type) [AddGroup A] [AddGroup B] [AddGroup C] : proof_problem r AB BC A B C :=
by
  sorry

end proof_theorem_l269_269384


namespace surface_area_ratio_is_sqrt3_l269_269497

def cube_side_length : ℝ := 2

def tetrahedron_vertices := [(0,0,0), (2,2,0), (2,0,2), (0,2,2)]

-- Distance formula function
def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

-- Calculate side length of tetrahedron using any pair of vertices
def tetrahedron_side_length : ℝ :=
  distance (0,0,0) (2,2,0)

-- Surface area of a cube
def cube_surface_area (s : ℝ) : ℝ := 6 * s ^ 2

-- Surface area of a regular tetrahedron
def tetrahedron_surface_area (a : ℝ) : ℝ := Real.sqrt 3 * a ^ 2

-- The cube we consider
def cube := (cube_side_length, cube_surface_area cube_side_length)

-- The tetrahedron we consider
def tetrahedron := (tetrahedron_side_length, tetrahedron_surface_area tetrahedron_side_length)

-- The ratio of surface areas
def surface_area_ratio : ℝ :=
  cube.2 / tetrahedron.2

theorem surface_area_ratio_is_sqrt3 :
  surface_area_ratio = Real.sqrt 3 := by
  sorry

end surface_area_ratio_is_sqrt3_l269_269497


namespace sum_of_other_endpoint_coords_l269_269815

theorem sum_of_other_endpoint_coords (x y : ℝ) (hx : (6 + x) / 2 = 5) (hy : (2 + y) / 2 = 7) : x + y = 16 := 
  sorry

end sum_of_other_endpoint_coords_l269_269815


namespace find_x_l269_269873

theorem find_x (x : ℕ) : (4 + x) / (7 + x) = 3 / 4 → x = 5 :=
by
  sorry

end find_x_l269_269873


namespace zeros_between_decimal_point_and_first_nonzero_digit_l269_269476

theorem zeros_between_decimal_point_and_first_nonzero_digit 
  : ∀ (n d : ℕ), n = 7 → d = 5000 → number_of_zeros (n / d) = 2 :=
by sorry

end zeros_between_decimal_point_and_first_nonzero_digit_l269_269476


namespace balls_in_boxes_l269_269317

-- We define a proof problem that states:
-- The number of ways to distribute 7 distinguishable balls in 3 indistinguishable boxes is 365.
theorem balls_in_boxes :
  let balls := 7
  let boxes := 3
  number_of_ways balls boxes = 365 :=
sorry

end balls_in_boxes_l269_269317


namespace min_possible_rank_students_l269_269331

theorem min_possible_rank_students (students : ℕ) (scores : set ℕ) (x : ℕ) 
  (h_students : students = 148) 
  (h_scores_range : scores = {i | 100 ≤ i ∧ i ≤ 120})
  (h_scores_nonempty : scores.nonempty) :
  ∃ x, ∀ i ∈ scores, scores.filter (λ s, s = i).card ≤ x ∧ x = 8 :=
by sorry

end min_possible_rank_students_l269_269331


namespace fraction_left_handed_l269_269533

theorem fraction_left_handed (red blue : ℕ) (h_ratio : red = 2 * blue)
  (h_red_left_handed : left_red = (1/3) * red)
  (h_blue_left_handed : left_blue = (2/3) * blue) :
  (left_red + left_blue) / (red + blue) = 4 / 9 :=
by
  let left_red := (1 / 3) * red
  let left_blue := (2 / 3) * blue
  let total_left_handed := left_red + left_blue
  let total_participants := red + blue
  have h1 : left_red = (1 / 3) * red := sorry
  have h2 : left_blue = (2 / 3) * blue := sorry
  have h3 : total_left_handed = (4 / 3) * blue := sorry -- since red = 2 * blue
  have h4 : total_participants = 3 * blue := sorry -- since red = 2 * blue
  calc
    total_left_handed / total_participants = ((4 / 3) * blue) / (3 * blue) : by rw [h3, h4]
                                         ... = 4 / 9 : by sorry

end fraction_left_handed_l269_269533


namespace regular_polygon_assignment_exists_l269_269552

theorem regular_polygon_assignment_exists :
  ∃ (a : Fin 2007 → ℕ) (m : Fin 2007 → ℕ), 
  (∀ n, a n ∈ Finset.range 1 4015) ∧ 
  (∀ n, m n ∈ Finset.range 2 4015 ∧  Even (m n)) ∧ 
  (∃ S, ∀ n, a n + a (n + 1) % 2007 + m n = S) :=
sorry

end regular_polygon_assignment_exists_l269_269552


namespace divide_perimeter_l269_269467

/--
Given a triangle ABC with an excircle touching side BC at point M,
the line segment AM divides the perimeter of the triangle into two equal parts.
-/
theorem divide_perimeter (A B C I_A M : Point) [triangle ABC] [excircle_touch M B C] :
  divides_perimeter (line_through A M) ABC :=
sorry

end divide_perimeter_l269_269467


namespace geometric_series_product_l269_269998

theorem geometric_series_product (y : ℝ) :
  (∑'n : ℕ, (1 / 3 : ℝ) ^ n) * (∑'n : ℕ, (- 1 / 3 : ℝ) ^ n)
  = ∑'n : ℕ, (y⁻¹ : ℝ) ^ n ↔ y = 9 :=
by
  sorry

end geometric_series_product_l269_269998


namespace determine_train_and_car_number_l269_269805

def SECRET (s e c r e t : ℕ) : ℕ := s*10^5 + e*10^4 + c*10^3 + r*10^2 + e*10^1 + t
def OPEN (o p e n : ℕ) : ℕ := o*10^3 + p*10^2 + e*10^1 + n
def ANSWER (a n s w e r : ℕ) : ℕ := a*10^5 + n*10^4 + s*10^3 + w*10^2 + e*10^1 + r
def YOUR (y o u r : ℕ) : ℕ := y*10^3 + o*10^2 + u*10^1 + r
def OPENED (o p e n e d : ℕ) : ℕ := o*10^5 + p*10^4 + e*10^3 + n*10^2 + e*10^1 + d

theorem determine_train_and_car_number 
  (s e c r t o p n a w y u d : ℕ)
  (h1 : SECRET s e c r e t - OPEN o p e n = ANSWER a n s w e r - YOUR y o u r)
  (h2 : SECRET s e c r e t - OPENED o p e n e d = 20010)
  (unique_digits : ∀ x y, x ≠ y → x ≠ y) :
  ∃ (train car : ℕ), train = 392 ∧ car = 0 :=
sorry

end determine_train_and_car_number_l269_269805


namespace parallel_lines_angle_sum_l269_269375

theorem parallel_lines_angle_sum
  (l m : Line)
  (parallel_l_m : Parallel l m)
  (angleA : ℝ)
  (angleA_eq : angleA = 100)
  (angleB : ℝ)
  (angleB_eq : angleB = 140) :
  ∃ (angleC : ℝ), angleC = 120 :=
by
  -- Definition of angles using the given conditions
  let angleC := angleA - 80 + angleB - 140
  use angleC
  rw [angleA_eq, angleB_eq]
  norm_num
  sorry

end parallel_lines_angle_sum_l269_269375


namespace solve_for_f_sqrt_2_l269_269582

theorem solve_for_f_sqrt_2 (f : ℝ → ℝ) (h : ∀ x, f x = 2 / (2 - x)) : f (Real.sqrt 2) = 2 + Real.sqrt 2 :=
by
  sorry

end solve_for_f_sqrt_2_l269_269582


namespace smallest_possible_sum_l269_269602

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l269_269602


namespace function_machine_output_15_l269_269707

-- Defining the function machine operation
def function_machine (input : ℕ) : ℕ :=
  let after_multiplication := input * 3 in
  if after_multiplication > 25 then 
    after_multiplication - 7
  else 
    after_multiplication + 10

-- Statement of the problem to be proved
theorem function_machine_output_15 : function_machine 15 = 38 :=
by
  sorry

end function_machine_output_15_l269_269707


namespace range_of_m_l269_269330

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = 3) (h3 : x + y > 0) : m > -4 := by
  sorry

end range_of_m_l269_269330


namespace Bret_dinner_time_l269_269539

theorem Bret_dinner_time :
  ∀ (train_ride_time reading_time movies_time nap_time : ℕ)
  (total_planned_time : train_ride_time = 9 ∧ reading_time = 2 ∧ movies_time = 3 ∧ nap_time = 3),
  train_ride_time - (reading_time + movies_time + nap_time) = 1 :=
by
  intros train_ride_time reading_time movies_time nap_time total_planned_time
  cases total_planned_time with h1 h
  cases h with h2 h
  cases h with h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end Bret_dinner_time_l269_269539


namespace inradii_relationship_l269_269206

/-- Definitions of the parameters for the triangles -/
variables {A B C H : Type*}
variables {angleB_is_90 : ∠B = 90} {altitude_BH : BH ⟂ AC}
variables {r r₁ r₂ : ℝ}

/-- Main statement to be proved -/
theorem inradii_relationship
  (hABC : is_right_triangle ABC ∠B)
  (h_inradii : inradius ABC = r ∧ inradius ABH = r₁ ∧ inradius CBH = r₂)
  (h_altitude : is_altitude BH AC) :
  r^2 = r₁^2 + r₂^2 := 
sorry

end inradii_relationship_l269_269206


namespace cannot_obtain_1000000_l269_269119

def q (n : ℕ) : ℕ :=
  if n = 0 then 0
  else n.factorization 5

def initial_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def square_op (a : ℕ) : ℕ := a * a
def lcm_op (a b : ℕ) : ℕ := Nat.lcm a b

theorem cannot_obtain_1000000 : ¬ (∃ n ∈ closure (initial_set ∪ {square_op a | a ∈ initial_set} ∪ {lcm_op a b | a ∈ initial_set ∧ b ∈ initial_set}), n = 1000000) :=
sorry

end cannot_obtain_1000000_l269_269119


namespace store_owner_oil_l269_269953

noncomputable def liters_of_oil (volume_per_bottle : ℕ) (number_of_bottles : ℕ) : ℕ :=
  (volume_per_bottle * number_of_bottles) / 1000

theorem store_owner_oil : liters_of_oil 200 20 = 4 := by
  sorry

end store_owner_oil_l269_269953


namespace good_rectangle_partition_l269_269800

theorem good_rectangle_partition (a b : ℕ) (h_a : a > 100) (h_b : b > 100) :
  ∃ (t : list (ℕ × ℕ)), (∀ (x : ℕ × ℕ), x ∈ t → (x = (2, 2) ∨ x = (1, 11))) ∧
    (∑ (p : ℕ × ℕ) in t, p.fst * p.snd) = a * b :=
by 
  -- proof needed
  sorry

end good_rectangle_partition_l269_269800


namespace problem_a_problem_b_problem_c_l269_269909

theorem problem_a (x : ℝ) : x ^ 2 + 2 * x - 8 = 0 → x = -4 ∨ x = 2 :=
sorry

theorem problem_b (b c : ℝ) 
  (h1 : 1 ^ 2 + b * 1 + c = 2)
  (h2 : 2 ^ 2 + b * 2 + c = 0) 
  : b = -5 ∧ c = 6 :=
sorry

theorem problem_c (d a : ℝ) 
  (h1 : a * (0 - 1) ^ 2 + 8 / 3 = 2) 
  (h2 : a = -2 / 3) 
  : y = a * (d - 1) ^ 2 + 8 / 3 = 0 → d = 3 :=
sorry

end problem_a_problem_b_problem_c_l269_269909


namespace division_remainder_l269_269866

theorem division_remainder : 4053 % 23 = 5 :=
by
  sorry

end division_remainder_l269_269866


namespace chord_length_intercepted_by_line_on_curve_l269_269571

-- Define the curve and line from the problem
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 1 = 0
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Prove the length of the chord intercepted by the line on the curve is 4
theorem chord_length_intercepted_by_line_on_curve : 
  ∀ (x y : ℝ), curve x y → line x y → False := sorry

end chord_length_intercepted_by_line_on_curve_l269_269571


namespace john_total_points_l269_269040

theorem john_total_points (shots_2pts : ℕ) (shots_3pts : ℕ) (interval_mins : ℕ) (periods : ℕ) (period_duration : ℕ) : ℕ :=
  let points_per_interval := (2 * shots_2pts + 3 * shots_3pts)
  let total_mins := periods * period_duration
  let intervals := total_mins / interval_mins
  intervals * points_per_interval

example : john_total_points 2 1 4 2 12 = 42 := by 
  simp [john_total_points]
  sorry

end john_total_points_l269_269040


namespace nuts_initially_l269_269484

-- Define the conditions
def nuts_in_bag_day_1 (n : ℕ) : ℕ :=
  n

def nuts_in_bag_day_2 (n : ℕ) : ℕ :=
  2 * nuts_in_bag_day_1 n - 8

def nuts_in_bag_day_3 (n : ℕ) : ℕ :=
  2 * nuts_in_bag_day_2 n - 8

def nuts_in_bag_day_4 (n : ℕ) : ℕ :=
  2 * nuts_in_bag_day_3 n - 8

-- The problem
theorem nuts_initially (n : ℕ) : nuts_in_bag_day_4 n = 0 → n = 7 :=
by
  intro h,
  sorry

end nuts_initially_l269_269484


namespace range_of_even_quadratic_function_l269_269055

variable {a b : ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def f (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem range_of_even_quadratic_function :
  is_even_function f ∧ (1 + a ≤ 2 ∧ 2 ≤ 2) ∧ (1 + a ≥ -2)
  → (a = -3 ∧ b = 0)
  → ∀ x (hx : -2 ≤ x ∧ x ≤ 2), -10 ≤ f x ∧ f x ≤ 2 :=
by
  sorry

end range_of_even_quadratic_function_l269_269055


namespace find_intersection_and_area_l269_269936

theorem find_intersection_and_area (
  A : ℝ × ℝ,
  B : ℝ × ℝ,
  C : ℝ × ℝ := (6, 0),
  D : ℝ × ℝ,
  E : ℝ × ℝ := (4, 2)
) (hA : A.2 = 0) (hB : B.1 = 0) (hD : D.1 = 0) :
  ∃ A B D, 
    (∀ x, A.2 = 0 ∧ B.1 = 0 ∧ D.1 = 0 ∧
    ∀ y, (y = -2 * A.1 + B.2) ∧ (y = -1 * (x - 6)) ∧
    (E = (4, 2)) ∧
    let OE := (real.sqrt (E.1^2 + E.2^2))
    let OB := (real.sqrt (B.1^2 + B.2^2))
    let Area_OBE := (1/2) * OB * E.1
	let Area_OEC := (1/2) * C.1 * E.2
    let Area_OBEC := Area_OBE - Area_OEC
    in Area_OBEC = 14) :=
    sorry

end find_intersection_and_area_l269_269936


namespace probability_of_grid_conditions_l269_269438

noncomputable def probability_row_odd_column_even (grid : list (list ℕ)) : ℚ :=
  if h : (grid.length = 2 ∧ grid.all (λ row, row.length = 3)) then
    let nums := [1, 2, 3, 4, 4, 5, 5] in
    let occurrences := ∀ n : ℕ, n ∈ nums ↔ n ∈ grid.join in
    let row_sums_odd := ∀ row : list ℕ, row ∈ grid → row.sum % 2 = 1 in
    let col_sums_even := 
      let cols := [grid.head.map (λ row, row.get 0).sum,
                   grid.head.map (λ row, row.get 1).sum,
                   grid.head.map (λ row, row.get 2).sum] in
      ∀ col, col ∈ cols → col % 2 = 0
    in if occurrences ∧ row_sums_odd ∧ col_sums_even then 1 / 60 else 0
  else 0

theorem probability_of_grid_conditions : 
  probability_row_odd_column_even [
    [1, 3, 5],
    [2, 4, 4]
  ] = 1 / 60 :=
sorry

end probability_of_grid_conditions_l269_269438


namespace minimum_value_of_M_l269_269273

def f (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x - 1

theorem minimum_value_of_M
  (n : ℕ) (x : Fin n → ℝ)
  (hx : ∀ i : Fin n, 0 ≤ x i ∧ x i ≤ 3)
  (hx_sorted : ∀ i j : Fin n, i < j → x i < x j) :
  (|f(x 0) - f(x 1)| + |f(x 1) - f(x 2)| + ... + |f(x (n-2)) - f(x (n-1))| ≤ 10) :=
sorry

end minimum_value_of_M_l269_269273


namespace sum_of_tangent_angles_l269_269945

-- Given conditions: a regular pentagon inscribed in a circle and a second circle passing through each exterior angle vertex of the pentagon.
theorem sum_of_tangent_angles (P : Set ℝ) (C : Set ℝ) (C' : Set ℝ)
  (hP : IsRegularPentagon P)
  (hC : IsCircle C ∧ Inscribes P C)
  (hC' : IsCircle C' ∧ PassesThroughExteriorVertices P C') :
  SumTangentAngles P C' = 360 :=
sorry

end sum_of_tangent_angles_l269_269945


namespace williams_probability_l269_269881

noncomputable def prob_correct (n k : ℕ) : ℝ := 
  (Nat.choose n k) * (1/5)^k * (4/5)^(n-k)

noncomputable def prob_at_least_two_correct (n : ℕ) : ℝ := 
  1 - prob_correct n 0 - prob_correct n 1 

theorem williams_probability :
  prob_at_least_two_correct 6 = 5385 / 15625 := by
  sorry

end williams_probability_l269_269881


namespace probability_even_sum_of_sequence_l269_269513

-- Definitions based on conditions
variable (a b c d e f : ℕ)
def is_sequence := {1, 2, 3, 4, 5, 6} = {a, b, c, d, e, f}
def is_even_sum (x y z w u v : ℕ) : Prop := x * y * z + w * u * v % 2 = 0

-- Problem statement
theorem probability_even_sum_of_sequence :
  is_sequence a b c d e f →
  (∑' arrangement : finset.univ^6, ite (is_sequence arrangement.1 arrangement.2 arrangement.3 arrangement.4 arrangement.5 arrangement.6 ∧ is_even_sum arrangement.1 arrangement.2 arrangement.3 arrangement.4 arrangement.5 arrangement.6) 1 0) /
  (∑' arrangement : finset.univ^6, 1) = 9 / 10 :=
sorry

end probability_even_sum_of_sequence_l269_269513


namespace equilateral_triangle_of_equal_inradii_l269_269024
open Triangle

theorem equilateral_triangle_of_equal_inradii 
  {ABC : Triangle} 
  (h_acute : ABC.acute)
  (M : Point)
  (hM_medians : M = ABC.centroid ∨ M = ABC.incenter ∨ M = ABC.orthocenter)
  (h_equal_radii : (inradius (Triangle.mk ABC.A M ABC.B)) = (inradius (Triangle.mk ABC.B M ABC.C)) 
                   ∧ (inradius (Triangle.mk ABC.B M ABC.C)) = (inradius (Triangle.mk ABC.C M ABC.A)))
  : ABC.equilateral := 
sorry

end equilateral_triangle_of_equal_inradii_l269_269024


namespace total_time_is_three_hours_l269_269075

-- Define the conditions of the problem in Lean
def time_uber_house := 10
def time_uber_airport := 5 * time_uber_house
def time_check_bag := 15
def time_security := 3 * time_check_bag
def time_boarding := 20
def time_takeoff := 2 * time_boarding

-- Total time in minutes
def total_time_minutes := time_uber_house + time_uber_airport + time_check_bag + time_security + time_boarding + time_takeoff

-- Conversion from minutes to hours
def total_time_hours := total_time_minutes / 60

-- The theorem to prove
theorem total_time_is_three_hours : total_time_hours = 3 := by
  sorry

end total_time_is_three_hours_l269_269075


namespace angle_AHB_eq_120_l269_269526

theorem angle_AHB_eq_120 (A B C D E H : Type) 
  [triangle ABC]
  [altitude AD of triangle ABC]
  [altitude BE of triangle ABC]
  [orthocenter H of triangle ABC]
  (angle_BAC_eq_40 : ∠BAC = 40)
  (angle_ABC_eq_80 : ∠ABC = 80) : 
  ∠AHB = 120 :=
by sorry

end angle_AHB_eq_120_l269_269526


namespace joe_avg_speed_l269_269729

noncomputable def total_distance : ℝ :=
  420 + 250 + 120 + 65

noncomputable def total_time : ℝ :=
  (420 / 60) + (250 / 50) + (120 / 40) + (65 / 70)

noncomputable def avg_speed : ℝ :=
  total_distance / total_time

theorem joe_avg_speed : avg_speed = 53.67 := by
  sorry

end joe_avg_speed_l269_269729


namespace arithmetic_sequence_S2008_l269_269700

theorem arithmetic_sequence_S2008 (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (h1 : a1 = -2008)
  (h2 : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d)
  (h3 : (S 12 / 12) - (S 10 / 10) = 2) :
  S 2008 = -2008 := 
sorry

end arithmetic_sequence_S2008_l269_269700


namespace S9_value_l269_269014

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Define the arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a_n (n + 1) - a_n n) = (a_n 1 - a_n 0)

-- Sum of the first n terms of arithmetic sequence
def sum_first_n_terms (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n n = n * (a_n 0 + a_n (n - 1)) / 2

-- Given conditions: 
axiom a4_plus_a6 : a_n 4 + a_n 6 = 12
axiom S_definition : sum_first_n_terms S_n a_n

theorem S9_value : S_n 9 = 54 :=
by
  -- assuming the given conditions and definitions, we aim to prove the desired theorem.
  sorry

end S9_value_l269_269014


namespace thread_length_l269_269091

theorem thread_length (initial_length : ℝ) (fraction : ℝ) (additional_length : ℝ) (total_length : ℝ) 
  (h1 : initial_length = 12) 
  (h2 : fraction = 3 / 4) 
  (h3 : additional_length = initial_length * fraction)
  (h4 : total_length = initial_length + additional_length) : 
  total_length = 21 := 
by
  -- proof steps would go here
  sorry

end thread_length_l269_269091


namespace prob_blue_section_damaged_all_days_l269_269905

noncomputable def prob_of_7_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_blue_section_damaged_all_days :
  prob_of_7_successes 7 7 (2 / 7) = 128 / 823543 :=
by sorry

end prob_blue_section_damaged_all_days_l269_269905


namespace find_x0_of_f_x0_eq_1_l269_269636

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then log x else x⁻²

theorem find_x0_of_f_x0_eq_1 :
  ∀ x0 : ℝ, (f x0 = 1) → x0 = 10 :=
by
  intro x0 hx
  unfold f at hx
  split_ifs at hx with h1 h2
  { -- Case x > 0
    have h3: log x0 = 1, from hx
    exact (le_of_eq (exp_inj.1 h3)).symm
  }
  { -- Case x < 0
    have h4: x0⁻² = 1, from hx
    exfalso
    rw [neg_square x0] at h4
    linarith }
  { -- Case x = 0
    simp at h1 h2,
    contradiction }
  { -- Case otherwise (x = 0 should never be reached)
    contradiction }

end find_x0_of_f_x0_eq_1_l269_269636


namespace number_of_factors_l269_269239

theorem number_of_factors :
  let n := (2^3) * (3^5) * (5^4) * (7^2) * (11^6)
  let exponents := [3, 5, 4, 2, 6]
  let calc_number_of_factors (exps : List ℕ) : ℕ := exps.foldl (\prod (exp : ℕ) => prod * (exp + 1)) 1
  calc_number_of_factors exponents = 2520 :=
by
  sorry

end number_of_factors_l269_269239


namespace red_second_given_first_red_l269_269209

theorem red_second_given_first_red (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
(h_total : total_balls = 10) (h_red : red_balls = 6) (h_white : white_balls = 4) :
  let P1 := (red_balls : ℚ) / total_balls in
  let P := (red_balls * (red_balls - 1) : ℚ) / (total_balls * (total_balls - 1)) in
  let P2 := P / P1 in
  P2 = 5 / 9 :=
by
  sorry

end red_second_given_first_red_l269_269209


namespace bankers_gain_correct_l269_269421

-- Conditions
def BankersDiscount : ℝ := 1360
def Time : ℝ := 3
def RateOfInterest : ℝ := 12 / 100

-- Definitions based on conditions
def FaceValue := (BankersDiscount * 100) / (RateOfInterest * Time)
def TrueDiscount := (FaceValue * RateOfInterest * Time) / (100 + (RateOfInterest * Time))
def BankersGain := BankersDiscount - TrueDiscount

-- Goal
theorem bankers_gain_correct : BankersGain = 360 := by
  sorry

end bankers_gain_correct_l269_269421


namespace ab_value_l269_269826

theorem ab_value (a b : ℝ) (h1 : 2^a = 16^(b + 3)) (h2 : 64^b = 8^(a - 2)) : a * b = 40 :=
by
  -- The proof will be filled in later.
  sorry

end ab_value_l269_269826


namespace num_squares_less_than_1000_with_ones_digit_2_3_or_4_l269_269314

-- Define a function that checks if the one's digit of a number is one of 2, 3, or 4.
def ends_in (n : ℕ) (d : ℕ) : Prop := n % 10 = d

-- Define the main theorem to prove
theorem num_squares_less_than_1000_with_ones_digit_2_3_or_4 : 
  ∃ n, n = 6 ∧ ∀ m < 1000, ∃ k, m = k^2 → ends_in m 2 ∨ ends_in m 3 ∨ ends_in m 4 :=
sorry

end num_squares_less_than_1000_with_ones_digit_2_3_or_4_l269_269314


namespace triangle_rectangle_ratio_l269_269966

/--
An equilateral triangle and a rectangle both have perimeters of 60 inches.
The rectangle has a length to width ratio of 2:1.
We need to prove that the ratio of the length of the side of the triangle to
the length of the rectangle is 1.
-/
theorem triangle_rectangle_ratio
  (triangle_perimeter rectangle_perimeter : ℕ)
  (triangle_side rectangle_length rectangle_width : ℕ)
  (h1 : triangle_perimeter = 60)
  (h2 : rectangle_perimeter = 60)
  (h3 : rectangle_length = 2 * rectangle_width)
  (h4 : triangle_side = triangle_perimeter / 3)
  (h5 : rectangle_perimeter = 2 * rectangle_length + 2 * rectangle_width)
  (h6 : rectangle_width = 10)
  (h7 : rectangle_length = 20)
  : triangle_side / rectangle_length = 1 := 
sorry

end triangle_rectangle_ratio_l269_269966


namespace smallest_x_plus_y_l269_269610

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269610


namespace ball_probability_l269_269491

theorem ball_probability:
  let total_balls := 120
  let red_balls := 12
  let purple_balls := 18
  let yellow_balls := 15
  let desired_probability := 33 / 1190
  let probability_red := red_balls / total_balls
  let probability_purple_or_yellow := (purple_balls + yellow_balls) / (total_balls - 1)
  (probability_red * probability_purple_or_yellow = desired_probability) :=
sorry

end ball_probability_l269_269491


namespace correct_statements_l269_269089

def f (x : ℝ) : ℝ := 4 * sin (2 * x + π / 3)

theorem correct_statements :
  (∀ x, f (x + π) = f x) ∧ -- Period
  (2 * 0 + π / 3 = π / 3) ∧ -- Initial phase 
  (∀ x, abs (f x) ≤ 4) -- Amplitude
:= by
  -- Proof not required
  sorry

end correct_statements_l269_269089


namespace triangle_AC_length_l269_269682

noncomputable def AC_length : ℝ :=
  let AE : ℝ := 2
  let BD : ℝ := 2
  let DC : ℝ := 2
  let EC : ℝ := 1
  let x : ℝ := 2 + Real.sqrt 2
  x

theorem triangle_AC_length 
  (D E : ℝ) 
  (AE BD DC EC : ℝ)
  (hAE : AE = 2)
  (hBD : BD = 2)
  (hDC : DC = 2)
  (hEC : EC = 1)
  (h_perp_AB_AC : ∀ (AB AC : ℝ), ⊥)
  (h_perp_AE_BC : ∀ (AE BC : ℝ), ⊥) :
  let AC := 2 + Real.sqrt 2 in
  AC = 2 + Real.sqrt 2 :=
by
  sorry

end triangle_AC_length_l269_269682


namespace complex_conjugate_of_z_l269_269776

theorem complex_conjugate_of_z :
  let z := (2 + complex.i) / (1 + complex.i ^ 2 + complex.i ^ 5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269776


namespace factorial_base_a6_l269_269436

theorem factorial_base_a6 (a : ℕ → ℕ) :
  1735 = a 1 + a 2 * 2! + a 3 * 3! + a 4 * 4! + a 5 * 5! + a 6 * 6! →
  (∀ k : ℕ, k > 0 → k ≤ 6 → 0 ≤ a k ∧ a k ≤ k) →
  a 6 = 2 :=
sorry

end factorial_base_a6_l269_269436


namespace find_certain_number_l269_269492

-- Definition of the condition
def certain_number_divided_by_0.08_is_12.5 (x : ℝ) : Prop :=
  x / 0.08 = 12.5

-- Definition of the proof statement
theorem find_certain_number (x : ℝ) :
  certain_number_divided_by_0.08_is_12.5 x → x = 1 :=
by
  -- Proof omitted
  sorry

end find_certain_number_l269_269492


namespace find_constants_l269_269047

def B : Matrix (Fin 3) (Fin 3) ℤ := 
  ![\[1, 2, 3], 
    \[2, 1, 2], 
    \[3, 2, 1]]

def I : Matrix (Fin 3) (Fin 3) ℤ := Matrix.one

theorem find_constants 
  (a b c : ℤ)
  (h1 : a = -8)
  (h2 : b = -2)
  (h3 : c = -3) :
  (B ^ 3) + a • (B ^ 2) + b • B + c • I = Matrix.zero := 
by
  sorry

end find_constants_l269_269047


namespace possible_amount_in_jar_l269_269183

theorem possible_amount_in_jar :
  ∃ (p : ℕ), let pennies := p,
             let dimes := 3 * p,
             let quarters := 6 * p,
             let total_value := (0.01 * pennies + 0.10 * dimes + 0.25 * quarters : ℤ),
             total_value = 432 :=
begin
  sorry
end

end possible_amount_in_jar_l269_269183


namespace limit_fraction_simplified_l269_269986

theorem limit_fraction_simplified:
  (Real.limit (fun x => (Real.sqrt (9 + 2*x) - 5) / (Real.cbrt x - 2)) 8 = 12 / 5) := by
  sorry

end limit_fraction_simplified_l269_269986


namespace conforms_to_standard_algebraic_notation_l269_269144

-- Conditions provided in the problem statement
def option_A := (b a : ℝ) -> b / a
def option_B := (a : ℝ) -> a * 7
def option_C := (m : ℝ) -> 2 * m - 1 -- ignoring the unit 元 for this formal translation
def option_D := (x : ℝ) -> 3 + 1/2 * x 

-- The statement of the theorem
theorem conforms_to_standard_algebraic_notation : 
  ∀ (b a m x : ℝ), 
  option_A b a ∧ ¬ option_B a ∧ ¬ option_C m ∧ ¬ option_D x := 
by sorry

end conforms_to_standard_algebraic_notation_l269_269144


namespace remainder_when_divided_by_6_l269_269681

theorem remainder_when_divided_by_6 (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 :=
by sorry

end remainder_when_divided_by_6_l269_269681


namespace complex_conjugate_of_z_l269_269780

theorem complex_conjugate_of_z :
  let z := (2 + complex.i) / (1 + complex.i ^ 2 + complex.i ^ 5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269780


namespace sum_g_equals_1000_l269_269365

def g (x : ℝ) : ℝ := 4 / (16^x + 4)

theorem sum_g_equals_1000 :
  (∑ k in finset.range 2000, g ((k + 1) / 2001)) = 1000 :=
by
  -- proof would go here, but we include sorry to indicate it is skipped
  sorry

end sum_g_equals_1000_l269_269365


namespace linear_correlation_coefficient_l269_269265

theorem linear_correlation_coefficient (r : ℝ): (|r| ≤ 1) ∧ ((∀ ε > 0, (|r| > 1 - ε) → (strong_correlation r)) ∧ (∀ ε > 0, (|r| < ε) → (weak_correlation r))) :=
by
  sorry

-- Definitions for strong_correlation and weak_correlation can be added as needed
def strong_correlation (r : ℝ) : Prop := -- definition goes here
  sorry

def weak_correlation (r : ℝ) : Prop := -- definition goes here
  sorry

end linear_correlation_coefficient_l269_269265


namespace sandra_beignets_l269_269402

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end sandra_beignets_l269_269402


namespace impossible_cube_vertex_assignment_l269_269348

-- Define the problem and required conditions
def vertex_number_condition (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 220

def adjacent_vertices_condition (n_i n_j : ℕ) : Prop :=
  nat.gcd n_i n_j > 1

def nonadjacent_vertices_condition (n_i n_j : ℕ) : Prop :=
  nat.gcd n_i n_j = 1

def cube_vertices_conditions(numbers : Fin 8 → ℕ) : Prop :=
  (∀ i, vertex_number_condition (numbers i)) ∧
  (∀ i j, i ≠ j → numbers i ≠ numbers j) ∧
  (∀ i j, adjacent_edges i j → adjacent_vertices_condition (numbers i) (numbers j)) ∧
  (∀ i j, non_adjacent_edges i j → nonadjacent_vertices_condition (numbers i) (numbers j))

-- Define adjacent and non-adjacent vertex relations (to be defined based on cube vertex adjacency)
def adjacent_edges(i j : Fin 8) : Prop :=
  -- Example adjacency relations; to be replaced with actual adjacency relations for cube vertices
  sorry

def non_adjacent_edges(i j : Fin 8) : Prop :=
  -- Example non-adjacency relations; to be replaced with actual non-adjacency relations for cube vertices
  sorry

-- Main theorem statement
theorem impossible_cube_vertex_assignment :
  ¬(∃ numbers : Fin 8 → ℕ, cube_vertices_conditions numbers) :=
by
  -- Proof omitted
  sorry


end impossible_cube_vertex_assignment_l269_269348


namespace isosceles_triangle_perimeter_l269_269120

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4 ∨ a = 7) (h2 : b = 4 ∨ b = 7) (h3 : a ≠ b) :
  (∃ c, (c = a ∨ c = b) ∧ (a + c > b ∧ b + c > a ∧ a + b > c) ∧ (a + b + c = 15 ∨ a + b + c = 18)) := 
by 
  cases h1 with ha1 ha2;
  cases h2 with hb1 hb2;
  cases h3;
  repeat { sorry }

end isosceles_triangle_perimeter_l269_269120


namespace smallest_x_plus_y_l269_269611

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269611


namespace convex_quadrilateral_AD_lt_2_l269_269106

theorem convex_quadrilateral_AD_lt_2 (A B C D E : Point)
  (hAB : dist A B = 1)
  (hBC : dist B C = 1)
  (hCD : dist C D = 1)
  (hDE : dist D E = 1)
  (h_convex : convex_quad A B C D)
  (h_intersect : intersects_diag A C B D at E) :
  dist A D < 2 := sorry

end convex_quadrilateral_AD_lt_2_l269_269106


namespace least_value_of_sum_l269_269678

theorem least_value_of_sum (a b : ℝ) (h : log 3 a + log 3 b ≥ 5) : a + b ≥ 18 * Real.sqrt 3 :=
by
  sorry

end least_value_of_sum_l269_269678


namespace line_BC_eq_line_AM_eq_l269_269633

def Point := (ℤ × ℤ)

def line_eqn (p1 p2 : Point) (a b c : ℤ) : Prop :=
  ∀ x y, (x, y) = p1 ∨ (x, y) = p2 → a * x + b * y + c = 0

def is_midpoint (m p1 p2 : Point) : Prop :=
  ∃ xm ym : ℤ, m = (xm, ym) ∧ xm = (fst p1 + fst p2) / 2 ∧ ym = (snd p1 + snd p2) / 2

theorem line_BC_eq : 
  line_eqn (-2, -1) (2, 3) 1 (-1) 1 :=
by
  sorry

theorem line_AM_eq (M : Point) (hM : is_midpoint M (-2, -1) (2, 3)) :
  line_eqn (-1, 4) M 3 1 (-1) :=
by
  sorry

end line_BC_eq_line_AM_eq_l269_269633


namespace sin_240_eq_neg_sqrt3_over_2_l269_269167

theorem sin_240_eq_neg_sqrt3_over_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_over_2_l269_269167


namespace imaginary_part_eq_one_of_condition_l269_269168

theorem imaginary_part_eq_one_of_condition (z : ℂ) (h : (complex.abs z : ℂ) - complex.I = complex.conj z + 2 + 3 * complex.I) : z / (2 + complex.I) = 2 + complex.I :=
sorry

end imaginary_part_eq_one_of_condition_l269_269168


namespace total_acorns_l269_269409

theorem total_acorns (x y : ℝ) :
  let sheila_acorns := 5.3 * x
  let danny_acorns := sheila_acorns + y
  x + sheila_acorns + danny_acorns = 11.6 * x + y :=
by
  sorry

end total_acorns_l269_269409


namespace tan_difference_identity_l269_269667

theorem tan_difference_identity (a b : ℝ) (h1 : Real.tan a = 2) (h2 : Real.tan b = 3 / 4) :
  Real.tan (a - b) = 1 / 2 :=
sorry

end tan_difference_identity_l269_269667


namespace necessary_but_not_sufficient_condition_l269_269164

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (∀ x : ℝ, x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) ∧ 
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) :=
by
  split
  · intro hx
    cases' le_or_gt x 0 with hneg hpos
    · left
      linarith
    · right
      linarith
  · intros hx
    linarith
  sorry

end necessary_but_not_sufficient_condition_l269_269164


namespace sample_size_l269_269933

theorem sample_size (f_c f_o N: ℕ) (h1: f_c = 8) (h2: f_c = 1 / 4 * f_o) (h3: f_c + f_o = N) : N = 40 :=
  sorry

end sample_size_l269_269933


namespace complex_conjugate_of_z_l269_269768

theorem complex_conjugate_of_z : 
  let z := (2 + complex.i) / (1 + complex.i^2 + complex.i^5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269768


namespace find_lambda_l269_269647

-- Definitions of points A, B, C and vector OA, OB, OC
variables (A B C O : Type) [non_collinear_points : NonCollinear A B C] (OA OB OC : Vector A B C O)

-- The vector definition for OP
def OP (λ : ℝ) := (1/5) • OA + (2/3) • OB + λ • OC

-- Condition that P is coplanar with A, B, and C
def coplanar : Prop := ∃ P : Type, ∃ λ : ℝ, OP λ = P ∧ P ∈ plane A B C

-- The proof problem
theorem find_lambda (h : coplanar) : λ = 2/15 :=
sorry

end find_lambda_l269_269647


namespace angle_between_vectors_is_pi_over_4_l269_269594

open Real

variables (a b : EuclideanSpace ℝ 3)  -- Assume 3-dimensional Euclidean space for the vectors

-- Define the given conditions
def condition1 (a b : EuclideanSpace ℝ 3) : Prop := ∥a∥ = (2 * sqrt 2 / 3) * ∥b∥
def condition2 (a b : EuclideanSpace ℝ 3) : Prop := inner (a - b) (3 • a + 2 • b) = 0

-- Prove the angle between a and b is π/4
theorem angle_between_vectors_is_pi_over_4 
  (a b : EuclideanSpace ℝ 3) 
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  angle a b = π / 4 := 
sorry

end angle_between_vectors_is_pi_over_4_l269_269594


namespace palmer_first_week_photos_l269_269388

theorem palmer_first_week_photos :
  ∀ (X : ℕ), 
    100 + X + 2 * X + 80 = 380 →
    X = 67 :=
by
  intros X h
  -- h represents the condition 100 + X + 2 * X + 80 = 380
  sorry

end palmer_first_week_photos_l269_269388


namespace probability_of_7_successes_l269_269890

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l269_269890


namespace evaluate_fractional_exponent_l269_269245

theorem evaluate_fractional_exponent : 64^(2/3 : ℝ) = 16 := by
  have h1 : (64 : ℝ) = 2^6 := by
    norm_num
  rw [h1]
  have h2 : (2^6 : ℝ)^(2/3) = 2^(6 * (2/3)) := by
    rw [← Real.rpow_mul (by norm_num : 0 ≤ 2)] -- Using exponent properties
  rw [h2]
  calc 2^(6 * (2/3)) = 2^4 : by congr; ring
                ...  = 16  : by norm_num

end evaluate_fractional_exponent_l269_269245


namespace TriangleCircumRadiusExradiusEquality_l269_269275

-- Define the triangle ABC with appropriate points and properties
universe u

variables {α : Type u} [metric_space α] [normed_add_group α] [normed_space ℝ α]

structure Triangle (α : Type u) := 
(A B C : α)

-- Definition of circumcenter
def Circumcenter (T : Triangle α) := 
∃ O : α, C.dist O = B.dist O ∧ B.dist O = A.dist O

-- Definition of incenter
def Incenter (T : Triangle α) := 
∃ I : α, ∀ x ∈ line_segment ℝ A C, dist I x = dist I (line_segment ℝ B C)

-- Definition of altitude from point A to line BC
def Altitude (A B C : α) (D : α) := 
  D ∈ line_segment ℝ B C ∧ ∀ x ∈ span ℝ (set.singleton (B - A)), ⟪x, C - B⟫ = 0

-- Definition of the exradius opposite to side BC
def Exradius_opposite_to_BC (A B C : α) (r_a : ℝ) := 
  ∃ r_a : ℝ, ∀ S, 2 * S / (abs B + abs C - abs A) = r_a

-- The theorem statement
theorem TriangleCircumRadiusExradiusEquality 
  {T : Triangle α}
  (O : α) (I : α) (D : α)
  (circumcenter : Circumcenter T)
  (incenter : Incenter T)
  (altitude : Altitude T.A T.B T.C D)
  (I_on_OD : I ∈ line_segment ℝ O D)
  (r_a : ℝ) :
  ∃ R : ℝ, R = r_a :=
sorry

end TriangleCircumRadiusExradiusEquality_l269_269275


namespace sum_external_angles_of_regular_pentagon_inscribed_in_circle_l269_269189

theorem sum_external_angles_of_regular_pentagon_inscribed_in_circle :
  (pentagon : Type) (circle : Type) 
  (is_regular_pentagon : is_regular_pentagon_inscribed_in_circle pentagon circle) :
  sum_inscribed_external_angles pentagon circle = 540 := 
sorry

end sum_external_angles_of_regular_pentagon_inscribed_in_circle_l269_269189


namespace twelfth_prime_is_thirty_seven_l269_269455

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_prime (n : ℕ) : ℕ :=
  (list.filter is_prime (list.range (2 * n))) !! (n - 1)

theorem twelfth_prime_is_thirty_seven :
  nth_prime 6 = 13 → nth_prime 12 = 37 :=
by
  intros h,
  have h₁ : nth_prime 6 = 13 := h,
  -- Continue the proof steps
  sorry

end twelfth_prime_is_thirty_seven_l269_269455


namespace particular_solution_satisfies_initial_conditions_l269_269099

noncomputable def x_solution : ℝ → ℝ := λ t => (-4/3) * Real.exp t + (7/3) * Real.exp (-2 * t)
noncomputable def y_solution : ℝ → ℝ := λ t => (-1/3) * Real.exp t + (7/3) * Real.exp (-2 * t)

def x_prime (x y : ℝ) := 2 * x - 4 * y
def y_prime (x y : ℝ) := x - 3 * y

theorem particular_solution_satisfies_initial_conditions :
  (∀ t, deriv x_solution t = x_prime (x_solution t) (y_solution t)) ∧
  (∀ t, deriv y_solution t = y_prime (x_solution t) (y_solution t)) ∧
  (x_solution 0 = 1) ∧
  (y_solution 0 = 2) := by
  sorry

end particular_solution_satisfies_initial_conditions_l269_269099


namespace problem_1_l269_269910

theorem problem_1 (f : ℝ → ℝ) (hf_mul : ∀ x y : ℝ, f (x * y) = f x + f y) (hf_4 : f 4 = 2) : f (Real.sqrt 2) = 1 / 2 :=
sorry

end problem_1_l269_269910


namespace sandra_beignets_16_weeks_l269_269405

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end sandra_beignets_16_weeks_l269_269405


namespace correct_statement_is_D_l269_269146

theorem correct_statement_is_D :
  (¬ (∃ (a : ℤ), ∀ (x : ℤ), a^2 + 2 * a + 27 = x^3))
  ∧ (¬ (∀ (x y : ℤ), 4 * xy = 4))
  ∧ (¬ (∀ (x : ℤ), (x - 2) / 2 = -2))
  ∧ (is_monomial 0) :=
by
  -- Definitions and setup
  sorry

end correct_statement_is_D_l269_269146


namespace find_h_k_a_b_l269_269005

noncomputable def hyperbola_center : ℝ × ℝ := (3, 1)
noncomputable def hyperbola_vertex : ℝ × ℝ := (3, -2) 
noncomputable def hyperbola_focus : ℝ × ℝ := (3, 9)

theorem find_h_k_a_b : 
  let h := 3 in
  let k := 1 in
  let a := 3 in
  let c := 8 in
  let b := Real.sqrt (c^2 - a^2) in
  h + k + a + b = 7 + Real.sqrt 55 :=
by
  -- Set the values
  let h := 3
  let k := 1
  let a := 3
  let c := 8
  let b := Real.sqrt (c^2 - a^2)

  -- Perform the calculations
  have b_val : b = Real.sqrt 55 := by { sorry }
  have sum_val : h + k + a + b = 7 + Real.sqrt 55 := by { sorry }

  exact sum_val

end find_h_k_a_b_l269_269005


namespace P_eq_Q_l269_269060

noncomputable def P (z : ℂ) : ℂ := sorry
noncomputable def Q (z : ℂ) : ℂ := sorry
def P0 : Set ℂ := {z | P z = 0}
def Q0 : Set ℂ := {z | Q z = 0}
def P1 : Set ℂ := {z | P z = 1}
def Q1 : Set ℂ := {z | Q z = 1}

-- Given conditions:
axiom P_degree_ge1 : 1 ≤ P.degree
axiom Q_degree_ge1 : 1 ≤ Q.degree
axiom P0_eq_Q0 : P0 = Q0
axiom P1_eq_Q1 : P1 = Q1

theorem P_eq_Q : ∀ z : ℂ, P z = Q z := by
  sorry

end P_eq_Q_l269_269060


namespace rhombus_longer_diagonal_length_l269_269946

theorem rhombus_longer_diagonal_length
  (side_length : ℕ) (shorter_diagonal : ℕ) 
  (side_length_eq : side_length = 53) 
  (shorter_diagonal_eq : shorter_diagonal = 50) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 94 := by
  sorry

end rhombus_longer_diagonal_length_l269_269946


namespace num_solutions_gcd_lcm_l269_269158

noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

theorem num_solutions_gcd_lcm (x y : ℕ) :
  (Nat.gcd x y = factorial 20) ∧ (Nat.lcm x y = factorial 30) →
  2^10 = 1024 :=
  by
  intro h
  sorry

end num_solutions_gcd_lcm_l269_269158


namespace angle_between_p_and_q_is_acute_l269_269654

variables (A B C : ℝ)

-- defining the vectors p and q based on angles A and B
def p : ℝ × ℝ := (Real.cos A, Real.sin A)
def q : ℝ × ℝ := (-Real.cos B, Real.sin B)

-- indicating that A, B, and C are the acute angles of a triangle
def acute_angle (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def triangle_angles (A B C : ℝ) : Prop := acute_angle A ∧ acute_angle B ∧ acute_angle C ∧ A + B + C = π

-- calculating the dot product of vectors p and q
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- theorem statement: the angle between p and q is acute
theorem angle_between_p_and_q_is_acute :
  triangle_angles A B C →
  acute_angle (Real.arccos ((dot_product (p A) (q B)) / ((Real.sqrt (p A).1 ^ 2 + (p A).2 ^ 2) * (Real.sqrt (q B).1 ^ 2 + (q B).2 ^ 2)))) :=
begin
  sorry
end

end angle_between_p_and_q_is_acute_l269_269654


namespace trig_expression_correct_l269_269988

noncomputable def trig_expression_value : ℝ := 
  Real.cos (42 * Real.pi / 180) * Real.cos (78 * Real.pi / 180) + 
  Real.sin (42 * Real.pi / 180) * Real.cos (168 * Real.pi / 180)

theorem trig_expression_correct : trig_expression_value = -1 / 2 :=
by 
  sorry

end trig_expression_correct_l269_269988


namespace frog_weight_difference_l269_269430

theorem frog_weight_difference
  (large_frog_weight : ℕ)
  (small_frog_weight : ℕ)
  (h1 : large_frog_weight = 10 * small_frog_weight)
  (h2 : large_frog_weight = 120) :
  large_frog_weight - small_frog_weight = 108 :=
by
  sorry

end frog_weight_difference_l269_269430


namespace smallest_x_plus_y_l269_269616

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269616


namespace KL_bisects_OA_l269_269115

open EuclideanGeometry

variables (O A B C K L : Point)
variables (circle : Circle)
variables (M : Point)
variables (OA BC OB OC : Line)
variables [Tangent OA circle O A]
variables [Parallel BC OA]
variables [Intersect OB circle K L]
variables [Intersect OC circle K L]

-- Each definition used in Lean 4 statement should directly appear in the conditions from step a)
def line_kl_bisects_segment_oa :=
  ∃ (M : Point), Intersection KL OA M ∧ OM = MA

theorem KL_bisects_OA :
  line_kl_bisects_segment_oa :=
begin
  sorry
end

end KL_bisects_OA_l269_269115


namespace maximize_farmer_profit_l269_269500

theorem maximize_farmer_profit :
  ∃ x y : ℝ, x + y ≤ 2 ∧ 3 * x + y ≤ 5 ∧ x ≥ 0 ∧ y ≥ 0 ∧ x = 1.5 ∧ y = 0.5 ∧ 
  (∀ x' y' : ℝ, x' + y' ≤ 2 ∧ 3 * x' + y' ≤ 5 ∧ x' ≥ 0 ∧ y' ≥ 0 → 14400 * x + 6300 * y ≥ 14400 * x' + 6300 * y') :=
by
  sorry

end maximize_farmer_profit_l269_269500


namespace function_symmetric_about_x_1_l269_269276

variable {f : ℝ → ℝ}

theorem function_symmetric_about_x_1 (h : ∀ x : ℝ, f(x) = f(2 - x)) : 
  ∀ x : ℝ, f(x) = f(1 + (1 - x)) :=
by 
  sorry

end function_symmetric_about_x_1_l269_269276


namespace ratio_x_y_l269_269197

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end ratio_x_y_l269_269197


namespace other_train_speed_l269_269137

noncomputable def speed_of_other_train (l1 l2 v1 : ℕ) (t : ℝ) : ℝ := 
  let relative_speed := (l1 + l2) / 1000 / (t / 3600)
  relative_speed - v1

theorem other_train_speed :
  speed_of_other_train 210 260 40 16.918646508279338 = 60 := 
by
  sorry

end other_train_speed_l269_269137


namespace train_length_l269_269884

-- Define the conditions
def equal_length_trains (L : ℝ) : Prop :=
  ∃ (length : ℝ), length = L

def train_speeds : Prop :=
  ∃ v_fast v_slow : ℝ, v_fast = 46 ∧ v_slow = 36

def pass_time (t : ℝ) : Prop :=
  t = 36

-- The proof problem
theorem train_length (L : ℝ) 
  (h_equal_length : equal_length_trains L) 
  (h_speeds : train_speeds)
  (h_time : pass_time 36) : 
  L = 50 :=
sorry

end train_length_l269_269884


namespace number_of_pictures_in_first_coloring_book_l269_269086

-- Define the conditions
variable (X : ℕ)
variable (total_pictures_colored : ℕ := 44)
variable (pictures_left : ℕ := 11)
variable (pictures_in_second_coloring_book : ℕ := 32)
variable (total_pictures : ℕ := total_pictures_colored + pictures_left)

-- The theorem statement
theorem number_of_pictures_in_first_coloring_book :
  X + pictures_in_second_coloring_book = total_pictures → X = 23 :=
by
  intro h
  sorry

end number_of_pictures_in_first_coloring_book_l269_269086


namespace sequence_sum_l269_269987

theorem sequence_sum :
  let S := list.zipWith (-)
    (list.range' 760 39).reverse
    (list.range' 760 38)
  (list.sum S = 760) :=
begin
  sorry
end

end sequence_sum_l269_269987


namespace quadratic_two_distinct_real_roots_l269_269300

theorem quadratic_two_distinct_real_roots (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + 2 * x1 - 3 = 0) ∧ (a * x2^2 + 2 * x2 - 3 = 0)) ↔ a > -1 / 3 := by
  sorry

end quadratic_two_distinct_real_roots_l269_269300


namespace largest_frog_weight_difference_l269_269432

def frog_weight_difference (largest_frog_weight : ℕ) (weight_ratio : ℕ) : ℕ :=
  let smallest_frog_weight := largest_frog_weight / weight_ratio
  largest_frog_weight - smallest_frog_weight

theorem largest_frog_weight_difference :
  frog_weight_difference 120 10 = 108 :=
begin
  -- Definitions and conditions have been set.
  -- Proof is not required as per the instructions.
  sorry
end

end largest_frog_weight_difference_l269_269432


namespace smallest_possible_sum_l269_269603

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l269_269603


namespace total_people_at_gathering_l269_269214

theorem total_people_at_gathering (total_wine : ℕ) (total_soda : ℕ) (both_wine_soda : ℕ) 
    (H1 : total_wine = 26) (H2 : total_soda = 22) (H3 : both_wine_soda = 17) : 
    total_wine - both_wine_soda + total_soda - both_wine_soda + both_wine_soda = 31 := 
by
  rw [H1, H2, H3]
  exact Nat.correct_answer = 31 -- combining results
  rw [Nat.sub_add_cancel (Nat.le_of_lt (sorry))] -- just using properties
  exact nat.add_comm 17 9 -- final proof step
  sorry -- ending suggestion

end total_people_at_gathering_l269_269214


namespace min_days_is_9_l269_269564

theorem min_days_is_9 (n : ℕ) (rain_morning rain_afternoon sunny_morning sunny_afternoon : ℕ)
  (h1 : rain_morning + rain_afternoon = 7)
  (h2 : rain_afternoon ≤ sunny_morning)
  (h3 : sunny_afternoon = 5)
  (h4 : sunny_morning = 6) :
  n ≥ 9 :=
sorry

end min_days_is_9_l269_269564


namespace sequence_form_l269_269798

theorem sequence_form (y : ℕ → ℝ) (n : ℕ) (h : ∀ (f : ℕ → ℝ), (degree f < n) → (∑ k in finset.range (n + 1), f k * y k = 0)) :
  ∃ λ : ℝ, ∀ k, y k = λ * (-1 : ℝ)^k * (nat.choose n k) := 
sorry

end sequence_form_l269_269798


namespace symmetric_point_coordinates_l269_269698

theorem symmetric_point_coordinates (M : ℝ × ℝ) (N : ℝ × ℝ) (hM : M = (1, -2)) (h_sym : N = (-M.1, -M.2)) :
  N = (-1, 2) :=
by sorry

end symmetric_point_coordinates_l269_269698


namespace max_angle_x_coordinate_P_l269_269696

open Real

-- Definitions of points M, N, and the movement of point P along x-axis
def M : (ℝ × ℝ) := (-1, 2)
def N : (ℝ × ℝ) := (1, 4)

-- Conditions and proofs to be skipped
theorem max_angle_x_coordinate_P : 
  (∃ x : ℝ, (P : ℝ × ℝ) → P = (x, 0) → |angle M P N| = pi) → 
  x = 1 := by 
  sorry

end max_angle_x_coordinate_P_l269_269696


namespace minimum_points_to_determine_polynomial_l269_269080

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

def different_at (f g : ℝ → ℝ) (x : ℝ) : Prop :=
  f x ≠ g x

theorem minimum_points_to_determine_polynomial :
  ∀ (f g : ℝ → ℝ), is_quadratic f → is_quadratic g → 
  (∀ t, t < 8 → (different_at f g t → ∃ t₁ t₂ t₃, different_at f g t₁ ∧ different_at f g t₂ ∧ different_at f g t₃)) → False :=
by {
  sorry
}

end minimum_points_to_determine_polynomial_l269_269080


namespace probability_distinct_odd_digits_l269_269968

theorem probability_distinct_odd_digits :
  let S := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ i j, i ≠ j → (n.digits 10)[i] ≠ (n.digits 10)[j]) ∧ (∀ k, (n.digits 10)[k] ∈ {1, 3, 5, 7, 9}) } in
  |S| / 9000 = 1 / 75 := 
sorry

end probability_distinct_odd_digits_l269_269968


namespace function_machine_output_15_l269_269706

-- Defining the function machine operation
def function_machine (input : ℕ) : ℕ :=
  let after_multiplication := input * 3 in
  if after_multiplication > 25 then 
    after_multiplication - 7
  else 
    after_multiplication + 10

-- Statement of the problem to be proved
theorem function_machine_output_15 : function_machine 15 = 38 :=
by
  sorry

end function_machine_output_15_l269_269706


namespace num_four_digit_numbers_div_by_5_l269_269547

theorem num_four_digit_numbers_div_by_5 
    (digits : Finset ℕ)
    (condition_set : digits = {0, 1, 2, 3, 4, 5, 6}) : 
    ∃ n : ℕ, n = 220 ∧ 
    (∀ d1 d2 d3 d4 ∈ digits, 
       d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
       d2 ≠ d3 ∧ d2 ≠ d4 ∧ 
       d3 ≠ d4 ∧ 
       (10 * (10 * (10 * d1 + d2) + d3) + d4) % 5 = 0 ∧ 
       (10 * (10 * (10 * d1 + d2) + d3) + d4) ≥ 1000) → 
        (Finset.filter (λ x, (∀ d ∈ ({ x.mod 10, (x/10).mod 10, (x/100).mod 10, (x/1000).mod 10} : Finset ℕ), 
                              d ∈ digits ∧ (∀ y ∈ ({ x.mod 10, (x/10).mod 10, (x/100).mod 10, (x/1000).mod 10} : Finset ℕ), d ≠ y → x ≠ y))
                             ∧ x ≥ 1000 ∧ x % 5 = 0) 
                             (Finset.Ico 1000 10000)).card = 220

end num_four_digit_numbers_div_by_5_l269_269547


namespace remainder_of_a_mod_10_l269_269581

theorem remainder_of_a_mod_10 (a : ℕ) (h : a = 1 + ∑ i in finset.range 21, nat.choose 20 i * 2^i) :
  a % 10 = 1 :=
sorry

end remainder_of_a_mod_10_l269_269581


namespace sqrt_and_cbrt_eq_self_l269_269343

theorem sqrt_and_cbrt_eq_self (x : ℝ) (h1 : x = Real.sqrt x) (h2 : x = x^(1/3)) : x = 0 := by
  sorry

end sqrt_and_cbrt_eq_self_l269_269343


namespace line_relation_l269_269277

-- Definitions of the conditions
variable (Line : Type) (Plane : Type) [HasPerp Plane Plane] [HasPerp Line Plane]
variable (m : Line) (α β : Plane)

-- Statement of the theorem
theorem line_relation (h1 : α ⟂ β) (h2 : m ⟂ α) : m ⟂ β ∨ m ⊆ β := sorry

end line_relation_l269_269277


namespace boat_length_is_seven_l269_269172

-- Declaration of the problem's conditions as constants
constant breadth : ℝ := 3
constant sink_depth : ℝ := 0.01
constant man_mass : ℝ := 210
constant water_density : ℝ := 1000

-- Theorem statement to prove the length of the boat is 7 meters
theorem boat_length_is_seven :
  ∃ L : ℝ, man_mass = water_density * (L * breadth * sink_depth) ∧ L = 7 :=
by
  sorry

end boat_length_is_seven_l269_269172


namespace problem_solution_l269_269025

variable (x : ℝ)

-- Given condition
def condition1 : Prop := (7 / 8) * x = 28

-- The main statement to prove
theorem problem_solution (h : condition1 x) : (x + 16) * (5 / 16) = 15 := by
  sorry

end problem_solution_l269_269025


namespace three_valid_pairs_exist_l269_269196

-- Define the conditions given in the problem
namespace TrapezoidalGarden

def area : ℕ := 1800
def altitude : ℕ := 60
def targetSum : ℕ := 60

-- Define a predicate to find the pairs that satisfy the conditions
def valid_pairs (b1 b2 : ℕ) : Prop :=
  b1 % 9 = 0 ∧ b2 % 9 = 0 ∧ b1 + b2 = targetSum

-- Define what we need to prove
theorem three_valid_pairs_exist : 
  ∃ (b1 b2 : ℕ), valid_pairs b1 b2 ∧
  ∃ (b3 b4 : ℕ), valid_pairs b3 b4 ∧ (b1, b2) ≠ (b3, b4) ∧
  ∃ (b5 b6 : ℕ), valid_pairs b5 b6 ∧ (b1, b2) ≠ (b5, b6) ∧ (b3, b4) ≠ (b5, b6) :=
begin
  sorry
end

end TrapezoidalGarden

end three_valid_pairs_exist_l269_269196


namespace convert_decimal_to_base_five_l269_269556

theorem convert_decimal_to_base_five (n : ℕ) : 
  (n = 250) → (5^3 = 125) → (250 \div 125 = 2) → (250 - 2 * 125 = 0) → 
  (let coef := [(2, 3), (0, 2), (0, 1), (0, 0)] -- coefficients and their respective powers of 5
   in n = 2 * 5^3 + 0 * 5^2 + 0 * 5^1 + 0 * 5^0) → 
  true := 
by
  intros
  sorry

end convert_decimal_to_base_five_l269_269556


namespace sequence_2008_term_l269_269109

def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d^3).sum

def sequence (a₀ : ℕ) (n : ℕ) : ℕ :=
  Nat.rec a₀ (λ _ prev, sum_of_cubes_of_digits prev) n

theorem sequence_2008_term (a₀ := 2008) : sequence a₀ 2007 = 133 := by
  sorry

end sequence_2008_term_l269_269109


namespace functional_equation_solution_l269_269236

def f : ℕ → ℕ := sorry
def g : ℕ → ℕ := sorry

theorem functional_equation_solution :
  (∀ n : ℕ, f n = n) ∧ (∀ n : ℕ, g n = 1) :=
  begin
    sorry
  end

end functional_equation_solution_l269_269236


namespace perimeter_ratio_of_divided_square_l269_269701

theorem perimeter_ratio_of_divided_square
  (S_ΔADE : ℝ) (S_EDCB : ℝ)
  (S_ratio : S_ΔADE / S_EDCB = 5 / 19)
  : ∃ (perim_ΔADE perim_EDCB : ℝ),
  perim_ΔADE / perim_EDCB = 15 / 22 :=
by
  -- Let S_ΔADE = 5x and S_EDCB = 19x
  -- x can be calculated based on the given S_ratio = 5/19
  -- Apply geometric properties and simplifications analogous to the described solution.
  sorry

end perimeter_ratio_of_divided_square_l269_269701


namespace possible_digits_for_A_l269_269703

def distinct_digit_sum (A B X Y : ℕ) (hO : 0 = 0) (hAB_distinct : A ≠ B) :=
  A + B = 12 ∧ A, B, X, Y ∈ finset.range 10 ∧ X ≠ Y ∧ X ≠ A ∧ X ≠ B ∧ Y ≠ A ∧ Y ≠ B

theorem possible_digits_for_A :
  (O = 0) → (∀ A B X Y, distinct_digit_sum A B X Y 0 A B X Y → 6) :=
  sorry

end possible_digits_for_A_l269_269703


namespace highest_of_seven_consecutive_with_average_33_l269_269420

theorem highest_of_seven_consecutive_with_average_33 (x : ℤ) 
    (h : (x - 3 + x - 2 + x - 1 + x + x + 1 + x + 2 + x + 3) / 7 = 33) : 
    x + 3 = 36 := 
sorry

end highest_of_seven_consecutive_with_average_33_l269_269420


namespace log_graph_fixed_point_l269_269428

theorem log_graph_fixed_point (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  ∃ x y, y = Real.log a (x + 2) + 1 ∧ (x, y) = (-1, 1) :=
by
  use (-1, 1)
  split
  · calc 1 = Real.log a (-1 + 2) + 1 : by sorry
  · simp

end log_graph_fixed_point_l269_269428


namespace unique_intersecting_lines_with_parabola_l269_269842

noncomputable def number_of_lines (p : ℝ × ℝ) (f : ℝ → ℝ) : ℕ :=
  let tangent_slopes := { x | 2 * x = -2 } -- solutions to the slope equation for tangents
  let tangent_lines := tangent_slopes.map (λ x, (λ y, y = -2 * (x + 1)))
  let parallel_lines := { (λ y, false) }   -- placeholder for lines parallel to symmetry
  if p ∈ { (x, f x) | x : ℝ } then 2 else 0

theorem unique_intersecting_lines_with_parabola :
  number_of_lines (-1, 0) (λ x, x^2 - 1) = 2 :=
sorry

end unique_intersecting_lines_with_parabola_l269_269842


namespace leah_coins_value_l269_269356

theorem leah_coins_value
  (p n : ℕ)
  (h₁ : n + p = 15)
  (h₂ : n + 2 = p) : p + 5 * n = 38 :=
by
  -- definitions used in converting conditions
  sorry

end leah_coins_value_l269_269356


namespace abigail_probability_l269_269207

theorem abigail_probability :
  let p_correct := 1 - (5/6 : ℚ)^5 in
  p_correct = (4651 / 7776 : ℚ) :=
by
  let p_correct := 1 - (5/6 : ℚ)^5
  show p_correct = (4651 / 7776 : ℚ)
  sorry

end abigail_probability_l269_269207


namespace sum_of_squares_l269_269122

theorem sum_of_squares (x : ℚ) (hx : 7 * x = 15) : 
  (x^2 + (2 * x)^2 + (4 * x)^2 = 4725 / 49) := by
  sorry

end sum_of_squares_l269_269122


namespace problem_statement_l269_269638

theorem problem_statement (a b : ℝ) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) (h_b_gt_1 : b > 1)
  (h1 : a = 1) (h2 : 1/2 * (b - 1)^2 + 1 = b) : a + b = 4 :=
sorry

end problem_statement_l269_269638


namespace probability_of_7_successes_in_7_trials_l269_269898

open Probability

/-- Define the given conditions for the problem -/
def n : ℕ := 7
def k : ℕ := 7
def p : ℚ := 2 / 7

/-- The binomial coefficient and the probability of success in n trials -/
theorem probability_of_7_successes_in_7_trials :
  P(X = k) = (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) :=
by
  have bep_0 : nat.choose 7 7 = 1, from sorry,
  have p_power_k : p ^ k = (2 / 7) ^ 7, from sorry,
  have q_power_rem : (1 - p) ^ (n - k) = 1, from sorry,
  have p_eq_frac : (2 / 7) ^ 7 * 1 = 128 / 823543, from sorry,
  show 1 * (2 / 7) ^ 7 * 1 = 128 / 823543, by sorry

end probability_of_7_successes_in_7_trials_l269_269898


namespace volume_calculation_correct_l269_269553

-- Define the dimensions of the parallelepiped
def length : ℝ := 5
def width : ℝ := 6
def height : ℝ := 7

-- Define the extension units
def extension_units : ℝ := 2

-- Define the volume of the core parallelepiped
def volume_parallelepiped : ℝ := length * width * height

-- Define the volume of the prisms formed by extending the face
def volume_prisms : ℝ :=
  2 * (height * width * extension_units +
       length * height * extension_units +
       length * width * extension_units)

-- Define the volume of the spherical segments (corners) and cylindrical segments (edges)
def volume_spherical_segments : ℝ := 8 * (1 / 8) * (4 / 3) * Real.pi * (extension_units ^ 3)
def volume_cylindrical_segments : ℝ :=
  4 * ((1 / 4) * Real.pi * (extension_units ^ 2) * length) +
  4 * ((1 / 4) * Real.pi * (extension_units ^ 2) * width) +
  4 * ((1 / 4) * Real.pi * (extension_units ^ 2) * height)

def volume_spheres_and_cylinders : ℝ :=
  volume_spherical_segments + volume_cylindrical_segments

-- Total volume calculation
def total_volume : ℝ :=
  volume_parallelepiped + (2 * volume_prisms) + volume_spheres_and_cylinders

-- Volume expressed as m + nπ / p
def volume_expression : ℝ :=
  (638 + 248 * Real.pi) / 3

theorem volume_calculation_correct :
  total_volume = volume_expression := by
  sorry

end volume_calculation_correct_l269_269553


namespace units_digit_p2_plus_3p_l269_269366

-- Define p
def p : ℕ := 2017^3 + 3^2017

-- Define the theorem to be proved
theorem units_digit_p2_plus_3p : (p^2 + 3^p) % 10 = 5 :=
by
  sorry -- Proof goes here

end units_digit_p2_plus_3p_l269_269366


namespace MaryBusinessTripTime_l269_269073

theorem MaryBusinessTripTime
  (t_uber_house : Nat := 10) -- Time for Uber to get to her house in minutes
  (t_airport_factor : Nat := 5) -- Factor for time to get to the airport
  (t_check_bag : Nat := 15) -- Time to check her bag in minutes
  (t_security_factor : Nat := 3) -- Factor for time to get through security
  (t_wait_boarding : Nat := 20) -- Time waiting for flight to start boarding in minutes
  (t_take_off_factor : Nat := 2) -- Factor for time waiting for plane to be ready take off
: (t_uber_house + t_uber_house * t_airport_factor + t_check_bag + t_check_bag * t_security_factor + t_wait_boarding + t_wait_boarding * t_take_off_factor) / 60 = 3 := 
begin
  sorry
end

end MaryBusinessTripTime_l269_269073


namespace capacity_of_each_type_l269_269522

def total_capacity_barrels : ℕ := 7000

def increased_by_first_type : ℕ := 8000

def decreased_by_second_type : ℕ := 3000

theorem capacity_of_each_type 
  (x y : ℕ) 
  (n k : ℕ)
  (h1 : x + y = total_capacity_barrels)
  (h2 : x * (n + k) / n = increased_by_first_type)
  (h3 : y * (n + k) / k = decreased_by_second_type) :
  x = 6400 ∧ y = 600 := sorry

end capacity_of_each_type_l269_269522


namespace find_divisor_l269_269813

theorem find_divisor (x : ℤ) : 83 = 9 * x + 2 → x = 9 :=
by
  sorry

end find_divisor_l269_269813


namespace kelly_grade_is_42_l269_269031

noncomputable def jenny_grade := 95

noncomputable def jason_grade := jenny_grade - 25

noncomputable def bob_grade := jason_grade / 2

noncomputable def kelly_grade := bob_grade * 1.2

theorem kelly_grade_is_42 : kelly_grade = 42 := by
  sorry

end kelly_grade_is_42_l269_269031


namespace find_valid_m_l269_269733

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem find_valid_m (m : ℝ) : (∀ x, ∃ y, g m x = y ∧ g m y = x) ↔ (m ∈ Set.Iio (-9 / 4) ∪ Set.Ioi (-9 / 4)) :=
by
  sorry

end find_valid_m_l269_269733


namespace zero_exists_in_interval_l269_269640

noncomputable def f (x : ℝ) : ℝ := (6 / x) - Real.log 2 x

theorem zero_exists_in_interval :
  ∃ x, (2 < x ∧ x < 4) ∧ f x = 0 :=
by
  sorry

end zero_exists_in_interval_l269_269640


namespace prob_blue_section_damaged_all_days_l269_269903

noncomputable def prob_of_7_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_blue_section_damaged_all_days :
  prob_of_7_successes 7 7 (2 / 7) = 128 / 823543 :=
by sorry

end prob_blue_section_damaged_all_days_l269_269903


namespace min_value_of_interval_diff_in_range_l269_269840

noncomputable def f (x : ℝ) : ℝ := abs (log x / log 3)

theorem min_value_of_interval_diff_in_range :
  ∃ (a b : ℝ), ({y : ℝ | y = f x ∧ a ≤ x ∧ x ≤ b} = set.Icc 0 1) ∧ (b - a = 2 / 3) :=
begin
  sorry
end

end min_value_of_interval_diff_in_range_l269_269840


namespace units_digit_powers_difference_l269_269628

theorem units_digit_powers_difference (p : ℕ) 
  (h1: p > 0) 
  (h2: p % 2 = 0) 
  (h3: (p % 10 + 2) % 10 = 8) : 
  ((p ^ 3) % 10 - (p ^ 2) % 10) % 10 = 0 :=
by
  sorry

end units_digit_powers_difference_l269_269628


namespace retailer_initial_thought_profit_percentage_l269_269969

/-
  An uneducated retailer marks all his goods at 60% above the cost price and thinking that he will still make some profit, 
  offers a discount of 25% on the marked price. 
  His actual profit on the sales is 20.000000000000018%. 
  Prove that the profit percentage the retailer initially thought he would make is 60%.
-/

theorem retailer_initial_thought_profit_percentage
  (cost_price marked_price selling_price : ℝ)
  (h1 : marked_price = cost_price + 0.6 * cost_price)
  (h2 : selling_price = marked_price - 0.25 * marked_price)
  (h3 : selling_price - cost_price = 0.20000000000000018 * cost_price) :
  0.6 * 100 = 60 := by
  sorry

end retailer_initial_thought_profit_percentage_l269_269969


namespace work_ratio_l269_269324

theorem work_ratio (m b : ℝ) (h1 : 12 * m + 16 * b = 1 / 5) (h2 : 13 * m + 24 * b = 1 / 4) : m = 2 * b :=
by sorry

end work_ratio_l269_269324


namespace water_in_sport_formulation_l269_269152

variable (C_sport : ℝ)
variable (C_standard : ℝ)
variable (W_standard : ℝ)
variable (C_ratio : ℝ)
variable (W_ratio : ℝ)
variable (corn_syrup_sport : ℝ)

-- Given conditions
axiom standard_ratio : C_standard = 12 ∧ W_standard = 30
axiom sport_flavor_ratio : C_ratio = 3
axiom sport_water_ratio : W_ratio = 2
axiom corn_syrup_sport_amount : corn_syrup_sport = 2

-- Statement to prove
theorem water_in_sport_formulation (H1 : standard_ratio ∧ sport_flavor_ratio ∧ sport_water_ratio ∧ corn_syrup_sport_amount) :
  let F_standard := 1
  F_sport = F_standard / 4
  W_sport = F_sport * 15  :=
  W_sport = 7.5 := 
sorry

end water_in_sport_formulation_l269_269152


namespace sandra_total_beignets_l269_269399

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end sandra_total_beignets_l269_269399


namespace more_red_balls_l269_269723

theorem more_red_balls (red_packs yellow_packs pack_size : ℕ) (h1 : red_packs = 5) (h2 : yellow_packs = 4) (h3 : pack_size = 18) :
  (red_packs * pack_size) - (yellow_packs * pack_size) = 18 :=
by
  sorry

end more_red_balls_l269_269723


namespace magnitude_of_AC1_l269_269018

variable (V : Type) [InnerProductSpace ℝ V]

/-- Given the conditions about vectors in the parallelepiped, determine the magnitude of AC₁. -/
theorem magnitude_of_AC1
  (A B D A1 C1 : V)
  (h1 : dist A B = 1)
  (h2 : dist A D = 1)
  (h3 : dist A A1 = 1)
  (h4 : ⟪B - A, D - A⟫ = 0)
  (h5 : angle (A1 - A) (B - A) = real.pi / 3)
  (h6 : angle (A1 - A) (D - A) = real.pi / 3) :
  dist A C1 = real.sqrt 5 := by
  sorry

end magnitude_of_AC1_l269_269018


namespace remainder_when_divided_by_multiple_of_10_l269_269874

theorem remainder_when_divided_by_multiple_of_10 (N : ℕ) (hN : ∃ k : ℕ, N = 10 * k) (hrem : (19 ^ 19 + 19) % N = 18) : N = 10 := by
  sorry

end remainder_when_divided_by_multiple_of_10_l269_269874


namespace shelby_rain_minutes_l269_269410

-- Definitions based on conditions
def non_rainy_speed : ℝ := 40
def rainy_speed : ℝ := 25
def total_distance : ℝ := 20
def total_time : ℝ := 40 / 60 -- 40 minutes in hours

-- Convert speeds to miles per minute
def non_rainy_speed_per_minute : ℝ := non_rainy_speed / 60
def rainy_speed_per_minute : ℝ := rainy_speed / 60

-- Definition of the problem's parameters and solution
def x (minutes_in_rain: ℝ) (total_minutes: ℝ) (travel_distance: ℝ) : Prop :=
  non_rainy_speed_per_minute * (total_minutes - minutes_in_rain) + 
    rainy_speed_per_minute * minutes_in_rain = travel_distance

-- Statement to prove
theorem shelby_rain_minutes : x 27 (40 : ℝ) total_distance :=
by
  unfold x
  sorry

end shelby_rain_minutes_l269_269410


namespace cost_of_adult_ticket_is_8_l269_269525

variables (A : ℕ) (num_people : ℕ := 22) (total_money : ℕ := 50) (num_children : ℕ := 18) (child_ticket_cost : ℕ := 1)

-- Definitions based on the given conditions
def child_tickets_cost := num_children * child_ticket_cost
def num_adults := num_people - num_children
def adult_tickets_cost := total_money - child_tickets_cost
def cost_per_adult_ticket := adult_tickets_cost / num_adults

-- The theorem stating that the cost of an adult ticket is 8 dollars
theorem cost_of_adult_ticket_is_8 : cost_per_adult_ticket = 8 :=
by sorry

end cost_of_adult_ticket_is_8_l269_269525


namespace sandy_age_l269_269407

theorem sandy_age (S M : ℕ) 
  (h1 : M = S + 16) 
  (h2 : (↑S : ℚ) / ↑M = 7 / 9) : 
  S = 56 :=
by sorry

end sandy_age_l269_269407


namespace quadrilateral_cyclic_l269_269159

variable {A B C D M O1 O2 : Type}
variable [ordered_ring A]

def circumscribed (AMCD BMDC : A) : Prop :=
sorry -- Definition of circumscribed quadrilateral

def isosceles_triangle (M : A) : Prop := 
sorry -- Definition of isosceles_triangle with vertex M

theorem quadrilateral_cyclic
  (h1 : M ∈ segment A B)
  (h2 : circumscribed AMCD O1)
  (h3 : circumscribed BMDC O2)
  (h4 : isosceles_triangle M) :
  cyclic ABCD :=
sorry

end quadrilateral_cyclic_l269_269159


namespace probability_of_drawing_white_ball_l269_269528

theorem probability_of_drawing_white_ball 
  (total_balls : ℕ) (white_balls : ℕ) 
  (h_total : total_balls = 9) (h_white : white_balls = 4) : 
  (white_balls : ℚ) / total_balls = 4 / 9 := 
by 
  sorry

end probability_of_drawing_white_ball_l269_269528


namespace cost_difference_proof_l269_269717

noncomputable def sailboat_daily_rent : ℕ := 60
noncomputable def ski_boat_hourly_rent : ℕ := 80
noncomputable def sailboat_hourly_fuel_cost : ℕ := 10
noncomputable def ski_boat_hourly_fuel_cost : ℕ := 20
noncomputable def discount : ℕ := 10

noncomputable def rent_time : ℕ := 3
noncomputable def rent_days : ℕ := 2

noncomputable def ken_sailboat_rent_cost :=
  sailboat_daily_rent * rent_days - sailboat_daily_rent * discount / 100

noncomputable def ken_sailboat_fuel_cost :=
  sailboat_hourly_fuel_cost * rent_time * rent_days

noncomputable def ken_total_cost :=
  ken_sailboat_rent_cost + ken_sailboat_fuel_cost

noncomputable def aldrich_ski_boat_rent_cost :=
  ski_boat_hourly_rent * rent_time * rent_days - (ski_boat_hourly_rent * rent_time * discount / 100)

noncomputable def aldrich_ski_boat_fuel_cost :=
  ski_boat_hourly_fuel_cost * rent_time * rent_days

noncomputable def aldrich_total_cost :=
  aldrich_ski_boat_rent_cost + aldrich_ski_boat_fuel_cost

noncomputable def cost_difference :=
  aldrich_total_cost - ken_total_cost

theorem cost_difference_proof : cost_difference = 402 := by
  sorry

end cost_difference_proof_l269_269717


namespace segments_intersect_at_point_l269_269117

variable {Point : Type} [AddGroup Point] [Module ℝ Point]
variable (A B C D E : Point)
variable (midpoint : Point → Point → Point)
variable (centroid : Point → Point → Point → Point)
variable (convex_pentagon : Prop)

noncomputable def intersection_point 
  (A B C D E : Point) 
  (midpoint : Point → Point → Point) 
  (centroid : Point → Point → Point → Point) 
  (convex_pentagon : Prop) 
  : Point :=
  (1/5 : ℝ) • (A + B + C + D + E)

theorem segments_intersect_at_point
  (A B C D E : Point)
  (midpoint : Point → Point → Point)
  (centroid : Point → Point → Point → Point)
  (h : convex_pentagon)
  : ∃ P : Point, P = intersection_point A B C D E midpoint centroid ∧ ∀ M₁ M₂ M₃ M₄ M₅ : Point,
      (M₁ = midpoint A B ∧ M₂ = midpoint B C ∧ M₃ = midpoint C D ∧ M₄ = midpoint D E ∧ M₅ = midpoint E A) →
      ∃ O : Point, 
        O = (1/3 : ℝ) • (M₁ + M₂ + M₃ + M₄ + M₅) :=
sorry

end segments_intersect_at_point_l269_269117


namespace values_of_x_l269_269563

theorem values_of_x (x : ℝ) (h1 : x^2 - 3 * x - 10 < 0) (h2 : 1 < x) : 1 < x ∧ x < 5 := 
sorry

end values_of_x_l269_269563


namespace stock_percent_change_l269_269537

theorem stock_percent_change {x : ℕ} (h1 : x > 0) :
    let value_after_day_1 := 0.75 * x
    let value_after_day_2 := 1.35 * value_after_day_1
    (value_after_day_2 / x - 1) * 100 = 1.25 := by 
    sorry

end stock_percent_change_l269_269537


namespace min_cost_to_package_fine_arts_collection_l269_269156

theorem min_cost_to_package_fine_arts_collection :
  let box_length := 20
  let box_width := 20
  let box_height := 12
  let cost_per_box := 0.50
  let required_volume := 1920000
  let volume_of_one_box := box_length * box_width * box_height
  let number_of_boxes := required_volume / volume_of_one_box
  let total_cost := number_of_boxes * cost_per_box
  total_cost = 200 := 
by
  sorry

end min_cost_to_package_fine_arts_collection_l269_269156


namespace smallest_ell_correct_l269_269063

noncomputable def smallest_ell (n k : ℕ) : ℕ :=
if k = 1 then 3
else if k > n / 2 then n
else 2 * k

theorem smallest_ell_correct (n k : ℕ) (hn : n ≥ 3) (hk : k ≥ 1)
  (P : Type) [convex_polygon P n] (A : finset P) (hA : A.card = k) :
  ∃ (ell : ℕ), ell = smallest_ell n k ∧ 
  ∀ x ∈ A, x ∈ convex_hull (vertices P (finset.range ell)) :=
sorry

end smallest_ell_correct_l269_269063


namespace find_f_lg_lg3_l269_269274

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  a * Real.sin x + b * x.cbrt + 4

theorem find_f_lg_lg3 (a b : ℝ) (h : f (Real.log (Real.log 10 / Real.log 3)) a b = 5) :
  f (Real.log (Real.log 3)) a b = 3 := 
sorry

end find_f_lg_lg3_l269_269274


namespace combined_circumference_correct_l269_269027

-- Define the given conditions as Lean definitions
def Jack_head : ℝ := 12
def Charlie_head : ℝ := (1/2 * Jack_head) + 9
def Bill_head : ℝ := (2/3 * Charlie_head)
def Maya_head : ℝ := (Jack_head + Charlie_head) / 2
def Thomas_head : ℝ := 2 * Bill_head - 3

-- Define the combined head circumference in Lean
def combined_head_circumference : ℝ := Jack_head + Charlie_head + Bill_head + Maya_head + Thomas_head

-- State the theorem to be proved
theorem combined_circumference_correct : combined_head_circumference = 67.5 :=
by sorry

end combined_circumference_correct_l269_269027


namespace math_problem_l269_269328

noncomputable def problem_statement (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9

theorem math_problem
  (a b c d : ℕ)
  (h1 : a ≠ b)
  (h2 : a ≠ c)
  (h3 : a ≠ d)
  (h4 : b ≠ c)
  (h5 : b ≠ d)
  (h6 : c ≠ d)
  (h7 : (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9) :
  a + b + c + d = 24 :=
sorry

end math_problem_l269_269328


namespace acute_angle_at_3_27_l269_269985

noncomputable def angle_between_clock_hands (hours minutes : ℕ) : ℝ := 
  let minute_angle := minutes * 6
  let hour_angle := (hours % 12) * 30 + (minutes / 60.0) * 30
  abs (minute_angle - hour_angle)

theorem acute_angle_at_3_27 : angle_between_clock_hands 3 27 = 58.5 := by
  sorry

end acute_angle_at_3_27_l269_269985


namespace class_total_students_l269_269535

def initial_boys : ℕ := 15
def initial_girls : ℕ := (120 * initial_boys) / 100 -- 1.2 * initial_boys

def final_boys : ℕ := initial_boys
def final_girls : ℕ := 2 * initial_girls

def total_students : ℕ := final_boys + final_girls

theorem class_total_students : total_students = 51 := 
by 
  -- the actual proof will go here
  sorry

end class_total_students_l269_269535


namespace jim_pages_per_week_l269_269034

theorem jim_pages_per_week (
  initial_speed_regular: ℝ := 40
  initial_speed_technical: ℝ := 30
  initial_time_regular: ℝ := 10
  initial_time_technical: ℝ := 5
  speed_increase_regular: ℝ := 1.5
  speed_increase_technical: ℝ := 1.3
  time_reduction_regular: ℝ := 4
  time_reduction_technical: ℝ := 2
) : 
  let new_speed_regular := speed_increase_regular * initial_speed_regular
  let new_speed_technical := speed_increase_technical * initial_speed_technical
  let new_time_regular := initial_time_regular - time_reduction_regular
  let new_time_technical := initial_time_technical - time_reduction_technical
  let pages_regular := new_speed_regular * new_time_regular
  let pages_technical := new_speed_technical * new_time_technical
  pages_regular + pages_technical = 477 := 
by sorry

end jim_pages_per_week_l269_269034


namespace find_x_l269_269989

theorem find_x (x : ℝ) (h : 6 * x⁻¹ - 3 * x⁻¹ * 2 * x⁻² = 1.25) : x = 1.698 := 
sorry

end find_x_l269_269989


namespace sum_of_modified_numbers_l269_269465

theorem sum_of_modified_numbers (x y R : ℝ) (h : x + y = R) : 
  2 * (x + 4) + 2 * (y + 5) = 2 * R + 18 :=
by
  sorry

end sum_of_modified_numbers_l269_269465


namespace prove_a_21022_le_1_l269_269741

-- Define the sequence a_n
variable (a : ℕ → ℝ)

-- Conditions for the sequence
axiom seq_condition {n : ℕ} (hn : n ≥ 1) :
  (a (n + 1))^2 + a n * a (n + 2) ≤ a n + a (n + 2)

-- Positive real numbers condition
axiom seq_positive {n : ℕ} (hn : n ≥ 1) :
  a n > 0

-- The main theorem to prove
theorem prove_a_21022_le_1 :
  a 21022 ≤ 1 :=
sorry

end prove_a_21022_le_1_l269_269741


namespace probability_of_7_successes_l269_269889

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l269_269889


namespace complex_conjugate_of_z_l269_269765

theorem complex_conjugate_of_z : 
  let z := (2 + complex.i) / (1 + complex.i^2 + complex.i^5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269765


namespace zeros_at_end_of_product1_value_of_product2_l269_269845

-- Definitions and conditions
def product1 := 360 * 5
def product2 := 250 * 4

-- Statements of the proof problems
theorem zeros_at_end_of_product1 : Nat.digits 10 product1 = [0, 0, 8, 1] := by
  sorry

theorem value_of_product2 : product2 = 1000 := by
  sorry

end zeros_at_end_of_product1_value_of_product2_l269_269845


namespace rectangle_ratio_l269_269490

open Real

-- Definition of the terms
variables {x y : ℝ}

-- Conditions as per the problem statement
def diagonalSavingsRect (x y : ℝ) := x + y - sqrt (x^2 + y^2) = (2 / 3) * y

-- The ratio of the shorter side to the longer side of the rectangle
theorem rectangle_ratio
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (h : diagonalSavingsRect x y) : x / y = 8 / 9 :=
by
sorry

end rectangle_ratio_l269_269490


namespace reflect_point_across_x_axis_l269_269699

theorem reflect_point_across_x_axis : (∀ x y : ℝ, (x, y) = (5, 2) → (x, -y) = (5, -2) ) :=
by
  intros x y h
  rw [h]
  -- Reflection across the x-axis
  apply congr (congr_arg Prod.mk rfl) (neg_eq_neg_one_mul 2).symm
  sorry

end reflect_point_across_x_axis_l269_269699


namespace min_tablets_to_get_two_each_l269_269173

def least_tablets_to_ensure_two_each (A B : ℕ) (A_eq : A = 10) (B_eq : B = 10) : ℕ :=
  if A ≥ 2 ∧ B ≥ 2 then 4 else 12

theorem min_tablets_to_get_two_each :
  least_tablets_to_ensure_two_each 10 10 rfl rfl = 12 :=
by
  sorry

end min_tablets_to_get_two_each_l269_269173


namespace MiaShots_l269_269808

theorem MiaShots (shots_game1_to_5 : ℕ) (total_shots_game1_to_5 : ℕ) (initial_avg : ℕ → ℕ → Prop)
  (shots_game6 : ℕ) (new_avg_shots : ℕ → ℕ → Prop) (total_shots : ℕ) (new_avg : ℕ): 
  shots_game1_to_5 = 20 →
  total_shots_game1_to_5 = 50 →
  initial_avg shots_game1_to_5 total_shots_game1_to_5 →
  shots_game6 = 15 →
  new_avg_shots 29 65 →
  total_shots = total_shots_game1_to_5 + shots_game6 →
  new_avg = 45 →
  (∃ shots_made_game6 : ℕ, shots_made_game6 = 29 - shots_game1_to_5 ∧ shots_made_game6 = 9) :=
by
  sorry

end MiaShots_l269_269808


namespace probability_three_one_l269_269914

-- Definitions based on the conditions
def total_balls : ℕ := 18
def black_balls : ℕ := 10
def white_balls : ℕ := 8
def drawn_balls : ℕ := 4

-- Defining the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the total number of ways to draw 4 balls from 18
def total_ways_to_draw : ℕ := binom total_balls drawn_balls

-- Definition of the number of favorable ways to draw 3 black and 1 white ball
def favorable_black_white : ℕ := binom black_balls 3 * binom white_balls 1

-- Definition of the number of favorable ways to draw 1 black and 3 white balls
def favorable_white_black : ℕ := binom black_balls 1 * binom white_balls 3

-- Total favorable outcomes
def total_favorable_ways : ℕ := favorable_black_white + favorable_white_black

-- The probability of drawing 3 one color and 1 other color
def probability : ℚ := total_favorable_ways / total_ways_to_draw

-- Prove that the probability is 19/38
theorem probability_three_one :
  probability = 19 / 38 :=
sorry

end probability_three_one_l269_269914


namespace winning_percentage_l269_269691

/-- In an election with two candidates, wherein the winner received 490 votes and won by 280 votes,
we aim to prove that the winner received 70% of the total votes. -/

theorem winning_percentage (votes_winner : ℕ) (votes_margin : ℕ) (total_votes : ℕ)
  (h1 : votes_winner = 490) (h2 : votes_margin = 280)
  (h3 : total_votes = votes_winner + (votes_winner - votes_margin)) :
  (votes_winner * 100 / total_votes) = 70 :=
by
  -- Skipping the proof for now
  sorry

end winning_percentage_l269_269691


namespace ratio_of_distances_on_parabola_l269_269305

theorem ratio_of_distances_on_parabola
  (p : ℝ) (h₀ : 0 < p)
  (A B M : ℝ × ℝ)
  (h₁ : M = (p, 0))
  (h₂ : ∃ m : ℝ, ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = m * y + p) ∧ (A = (x, y) ∧ B = (x, -y))) 
  (h₃ : prod.fst A = -2 * prod.fst B + p ∧ prod.snd A = -2 * prod.snd B) :
  |(dist (prod.fst A, 0) (p / 2, 0))/(dist (p / 2, 0) (p, 0))| = 5 / 2 :=
  sorry

end ratio_of_distances_on_parabola_l269_269305


namespace complex_conjugate_of_z_l269_269781

theorem complex_conjugate_of_z :
  let z := (2 + complex.i) / (1 + complex.i ^ 2 + complex.i ^ 5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269781


namespace f_10_eq_420_l269_269559

def f : ℕ → ℕ
| 1 := 1
| 2 := 2
| (n + 3) := f (n + 2) + f (n + 1) + (n + 3)

theorem f_10_eq_420 : f 10 = 420 := 
by
sorry

end f_10_eq_420_l269_269559


namespace find_parallel_line_l269_269254

theorem find_parallel_line (k : ℝ) :
  (∃ k, ∀ x y : ℝ, (3 * x + 4 * y + k = 0) ∧ (1 / 2) * | - k / 3 | * | - k / 4 | = 24) → 
  (k = 24 ∨ k = -24) :=
by
  sorry

end find_parallel_line_l269_269254


namespace a_2_eval_a_3_eval_a_k_general_limit_bn_ln2_l269_269360

-- First, we define the sequence a_k
noncomputable def a_seq (n : ℕ) (a : ℕ → ℚ) : ℕ → ℚ
| 1     => 1 / (n * (n + 1))
| (k+1) => - (1 / (k + n + 1)) + (n / k) * (List.sum (List.range k).map a)

-- Define a_k, taking the limit definition
-- Using an additional hint from the solution instead of the recursive version above
noncomputable def a_k (n k : ℕ) : ℚ := 1 / ((n + k) * (n + k - 1))

-- Define b_n as the sum of the square roots of a_k
noncomputable def b_n (n : ℕ) : ℚ :=
  List.sum $ (List.range n).map (λ k => real.sqrt (a_k n (k + 1)))

-- Task 1: Prove the given values of a_2 and a_3 match
theorem a_2_eval (n : ℕ) : a_k n 2 = 1 / ((n + 2) * (n + 1)) :=
sorry

theorem a_3_eval (n : ℕ) : a_k n 3 = 1 / ((n + 3) * (n + 2)) :=
sorry

-- Task 3: Prove the general term a_k holds
theorem a_k_general (n k : ℕ) : k ≥ 1 → a_seq n (a_k n) k = a_k n k :=
sorry

-- Task 4: Prove the limit of b_n
theorem limit_bn_ln2 : filter.tendsto (λ n, b_n n) filter.at_top (𝓝 (real.log 2)) :=
sorry

end a_2_eval_a_3_eval_a_k_general_limit_bn_ln2_l269_269360


namespace no_arithmetic_progression_of_1999_primes_l269_269830

theorem no_arithmetic_progression_of_1999_primes :
  ¬ ∃ (p : ℕ → ℕ) (a d : ℕ),
    (∀ n : ℕ, n < 1999 → nat.prime (p n)) ∧
    (∀ n : ℕ, p n = a + n * d) ∧
    (∀ n : ℕ, n < 1999 → p n < 12345) := 
by {
  sorry
}

end no_arithmetic_progression_of_1999_primes_l269_269830


namespace minimum_points_in_11th_game_l269_269015

-- Let P(n) represent the points scored in the nth game
-- Assume we have scores for games 8th, 9th, and 10th
def P (n : ℕ) : ℕ := 
  if n = 8 then 15 
  else if n = 9 then 22 
  else if n = 10 then 18 
  else 0

def points_first_seven_games (P : ℕ → ℕ) : ℕ :=
  (List.range 7).sum (fun n => P (n+1))
  
def average_higher_after_tenth_game (points_first_seven : ℕ) : Prop :=
  (points_first_seven + 55) / 10 > points_first_seven / 7

def average_greater_than_20_after_eleven_games (points_first_seven : ℕ) (P11 : ℕ) : Prop :=
  (points_first_seven + 55 + P11) / 11 > 20

noncomputable def minimum_points_11th_game (P : ℕ → ℕ) : ℕ :=
  let points_first_seven := points_first_seven_games P
  if average_higher_after_tenth_game points_first_seven ∧ average_greater_than_20_after_eleven_games points_first_seven 33 then 33 else 0

theorem minimum_points_in_11th_game (P : ℕ → ℕ) : minimum_points_11th_game P = 33 :=
  sorry

end minimum_points_in_11th_game_l269_269015


namespace angle_ACG_right_l269_269814

variable {α : Type*} [metric_space α]

/-- Given an isosceles triangle ABC with points E and F on sides AB and BC respectively,
such that AE = 2BF and point G on the ray EF such that GF = EF, the angle ACG is a right angle.
-/
theorem angle_ACG_right (A B C E F G : α)
  (is_isosceles : ∀ {a b c}, triangle.is_isosceles a b c)
  (on_AB : E ∈ segment A B)
  (on_BC : F ∈ segment B C)
  (AE_2BF : dist A E = 2 * dist B F)
  (G_on_EF : ∃ t : ℝ, t > 0 ∧ G = E + (t + 1) • (F - E))
  : angle A C G = π / 2 := sorry

end angle_ACG_right_l269_269814


namespace pair_cannot_appear_l269_269816

theorem pair_cannot_appear :
  ¬ ∃ (sequence_of_pairs : List (ℤ × ℤ)), 
    (1, 2) ∈ sequence_of_pairs ∧ 
    (2022, 2023) ∈ sequence_of_pairs ∧ 
    ∀ (a b : ℤ) (seq : List (ℤ × ℤ)), 
      (a, b) ∈ seq → 
      ((-a, -b) ∈ seq ∨ (-b, a+b) ∈ seq ∨ 
      ∃ (c d : ℤ), ((a+c, b+d) ∈ seq ∧ (c, d) ∈ seq)) := 
sorry

end pair_cannot_appear_l269_269816


namespace solve_inequality_solution_set_l269_269847

noncomputable def solution_set_of_inequality : Set ℝ := {x : ℝ | 6^(x^2 + x - 2) < 1}

theorem solve_inequality_solution_set :
  ∀ x : ℝ, x ∈ solution_set_of_inequality ↔ x > -2 ∧ x < 1 :=
by
  sorry

end solve_inequality_solution_set_l269_269847


namespace part1_part2_l269_269652

-- Definitions for the given vectors and parameters
def vector_a (α : ℝ) := (2 * Real.cos α, Real.sin α ^ 2)
def vector_b (α t: ℝ) := (2 * Real.sin α, t)

-- Conditions
variable (α : ℝ) (hα : 0 < α ∧ α < Real.pi / 2)

-- First Part: Prove t == 16/25 given vector subtraction condition
theorem part1 (h : vector_a α - vector_b α t = (2 / 5, 0)) 
: t = 16 / 25 :=
sorry

-- Second Part: Prove tan(2α + π/4) == 23/7 given dot product condition
theorem part2 (h : t = 1) (h_dot : vector_a α • vector_b α t = 1) 
: Real.tan (2 * α + Real.pi / 4) = 23 / 7 :=
sorry

end part1_part2_l269_269652


namespace smallest_sum_l269_269601

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l269_269601


namespace reflection_eq_l269_269260

open Matrix

def vector1 : Vector (Fin 2) ℚ := ![0, 5]
def vector2 : Vector (Fin 2) ℚ := ![2, 3]

def reflection_of_vector (v u : Vector (Fin 2) ℚ) : Vector (Fin 2) ℚ :=
  let proj := (dot_product v u / dot_product u u) • u
  2 • proj - v

theorem reflection_eq : reflection_of_vector ![0, 5] ![2, 3] = ![60 / 13, 35 / 13] :=
  sorry

end reflection_eq_l269_269260


namespace stripe_length_l269_269916

/--
A can is shaped like a right circular cylinder. The circumference of the base of the
can is 18 inches, and the height of the can is 8 inches. A spiral strip is painted
on the can such that it winds around the can exactly twice before it reaches from
the bottom of the can to the top, aligning directly above its starting point.
-/
theorem stripe_length (h : ℝ := 8) (circumference : ℝ := 18) : 
  let w := 2 * circumference in
  Float.sqrt (h^2 + w^2) = Float.sqrt 1360 :=
by {
  have w := 2 * circumference,
  suffices : (h ^ 2 + w ^ 2 = 1360), by
  {
    rw [this],
    exact eq.rfl,
  },
  calc
  h ^ 2 + w ^ 2 = 8 ^ 2 + (36) ^ 2 : by {rw [←circumference, w]}
  ... = 64 + 1296 : by norm_num
  ... = 1360 : by norm_num
}

end stripe_length_l269_269916


namespace find_m_l269_269270

open Real

def vec := (ℝ × ℝ)

def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def a : vec := (-1, 2)
def b (m : ℝ) : vec := (3, m)
def sum (m : ℝ) : vec := (a.1 + (b m).1, a.2 + (b m).2)

theorem find_m (m : ℝ) (h : dot_product a (sum m) = 0) : m = -1 :=
by {
  sorry
}

end find_m_l269_269270


namespace positive_integers_satisfy_condition_l269_269426

noncomputable def x := Nat

def condition := ∀ (x : Nat), (25 - 5 * x > 15) → (x < 2)

theorem positive_integers_satisfy_condition : ∃! (x : Nat), (0 < x) ∧ condition x :=
by
  sorry

end positive_integers_satisfy_condition_l269_269426


namespace M_equals_Nat_l269_269050

noncomputable def M (n : ℕ) : Prop :=
n = 2018 ∨ (∃ (m ∈ M), ∀ d | m, d ∈ M) ∨ (∃ k m ∈ M, 1 < k < m ∧ km + 1 ∈ M)

theorem M_equals_Nat :
  (∀ n : ℕ, M n) → M = { n : ℕ | n > 0 } :=
by
  sorry

end M_equals_Nat_l269_269050


namespace solve_quadratic_l269_269412

theorem solve_quadratic (x : ℝ) : (x^2 + x)^2 + (x^2 + x) - 6 = 0 ↔ x = -2 ∨ x = 1 :=
by
  sorry

end solve_quadratic_l269_269412


namespace no_heptagon_cross_section_l269_269876

-- Define what it means for a plane to intersect a cube and form a shape.
noncomputable def possible_cross_section_shapes (P : Plane) (C : Cube) : Set Polygon :=
  sorry -- Placeholder for the actual definition which involves geometric computations.

-- Prove that a heptagon cannot be one of the possible cross-sectional shapes of a cube.
theorem no_heptagon_cross_section (P : Plane) (C : Cube) : 
  Heptagon ∉ possible_cross_section_shapes P C :=
sorry -- Placeholder for the proof.

end no_heptagon_cross_section_l269_269876


namespace percentage_decrease_in_selling_price_l269_269934

theorem percentage_decrease_in_selling_price (S M : ℝ) 
  (purchase_price : S = 240 + M)
  (markup_percentage : M = 0.25 * S)
  (gross_profit : S - 16 = 304) : 
  (320 - 304) / 320 * 100 = 5 := 
by
  sorry

end percentage_decrease_in_selling_price_l269_269934


namespace geometric_series_proof_l269_269997

theorem geometric_series_proof (y : ℝ) :
  ((1 + (1/3) + (1/9) + (1/27) + ∑' n : ℕ, (1 / 3^(n+1))) * 
   (1 - (1/3) + (1/9) - (1/27) + ∑' n : ℕ, ((-1)^n * (1 / 3^(n+1)))) = 
   1 + (1/y) + (1/y^2) + (∑' n : ℕ, (1 / y^(n+1)))) → y = 9 := by
  sorry

end geometric_series_proof_l269_269997


namespace tan_double_angle_parallel_vectors_l269_269311

theorem tan_double_angle_parallel_vectors (θ : ℝ)
  (h : ∃ k : ℝ, (sin θ, 1) = k * (cos θ, 2)) :
  tan (2 * θ) = 4 / 3 :=
by sorry

end tan_double_angle_parallel_vectors_l269_269311


namespace problem_statement_l269_269134

-- Definitions for the isosceles triangle and its properties
variables {A B C P Q M N : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace M] [MetricSpace N]

-- Conditions from the problem statement
hypothesis isosceles_triangle : dist A B = dist A C
hypothesis line_parallel_BC : ∀ x : A, parallel x B C
hypothesis P_on_perp_bisector_AB : is_perpendicular_bisector P A B
hypothesis Q_on_perp_bisector_AC : is_perpendicular_bisector Q A C
hypothesis PQ_perpendicular_BC : ∀ p : P, ∀ q : Q, perpendicular p q B C
hypothesis M_on_line_l : ∀ m : M, parallel m B C
hypothesis N_on_line_l : ∀ n : N, parallel n B C
hypothesis angle_APM_right : ∠ A P M = (π / 2)
hypothesis angle_AQN_right : ∠ A Q N = (π / 2)

theorem problem_statement :
  (1 / dist A M) + (1 / dist A N) ≤ (2 / dist A B) := sorry

end problem_statement_l269_269134


namespace log_probability_l269_269392

noncomputable def geometric_probability (a b : ℝ) (P : set ℝ) : ℝ :=
  (b - a) / 4

theorem log_probability :
  let a := 0
  let b := 4
  let P := {x : ℝ | -1 ≤ Real.log (x + 1/2) / Real.log (1/2) ∧ Real.log (x + 1/2) / Real.log (1/2) ≤ 1}
  geometric_probability a b P = 3 / 8 := sorry

end log_probability_l269_269392


namespace invariant_lines_l269_269461

theorem invariant_lines (transformation : ℝ × ℝ → ℝ × ℝ) :
  transformation = (λ (P : ℝ × ℝ), (P.1 - P.2, -P.2)) →
  ∀ (P : ℝ × ℝ), (P.2 = 0 → transformation P = P) ∨
  (∃ b : ℝ, P.2 = 2 * P.1 + b → transformation P = (P.1 - P.2, -P.2)) :=
by
  intros h_trans P
  split
  case inl =>
    intro h_y_eq_0
    rw h_trans
    rw h_y_eq_0
    exact ⟨P.1, 0⟩
  case inr =>
    intro h_line
    cases h_line with b h_eq
    rw h_trans
    rw h_eq
    exists b
    rw [sub_eq_add_neg, neg_neg]
    exact ⟨P.1 - 2 * P.1 + b, b⟩

end invariant_lines_l269_269461


namespace not_basic_logical_calculation_l269_269145

-- Define the basic logical structures
inductive BasicLogicalStructure
| Sequential
| Conditional
| Loop

-- Define the function that checks if a structure is a basic logical structure
def isBasicLogicalStructure : BasicLogicalStructure → Prop
| BasicLogicalStructure.Sequential => true
| BasicLogicalStructure.Conditional => true
| BasicLogicalStructure.Loop => true

-- Define the calculation structure which is not basic
inductive OtherStructure
| Calculation

-- Theorem: Calculation structure is not a basic logical structure
theorem not_basic_logical_calculation : ¬ isBasicLogicalStructure OtherStructure.Calculation := by
  sorry

end not_basic_logical_calculation_l269_269145


namespace smallest_x_plus_y_l269_269614

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269614


namespace round_nearest_hundredth_l269_269828

theorem round_nearest_hundredth (x : ℚ) (hx : x = 36 + 36 / 99) : round_nearest_hundredth x = 36.37 := 
by 
  sorry

end round_nearest_hundredth_l269_269828


namespace percentage_increase_direct_proportionality_l269_269833

variable (x y k q : ℝ)
variable (h1 : x = k * y)
variable (h2 : x' = x * (1 + q / 100))

theorem percentage_increase_direct_proportionality :
  ∃ q_percent : ℝ, y' = y * (1 + q_percent / 100) ∧ q_percent = q := sorry

end percentage_increase_direct_proportionality_l269_269833


namespace range_of_fraction_l269_269825
open Real

theorem range_of_fraction (x y : ℝ) (h : x^2 + y^2 + 2 * x = 0) :
    ∃ t, t = y / (x - 1) ∧ -sqrt(3) / 3 ≤ t ∧ t ≤ sqrt(3) / 3 := by
    sorry

end range_of_fraction_l269_269825


namespace fraction_of_pizza_eaten_by_TreShawn_l269_269861

theorem fraction_of_pizza_eaten_by_TreShawn 
  (T : ℚ) (Michael : ℚ := 1/3) (LaMar : ℚ := 1/6) :
  T + Michael + LaMar = 1 → T = 1/2 :=
by
  intro h
  have h₁ : Michael + LaMar = 1/2 := by sorry -- proof needed
  have h₂ : T = 1 - (Michael + LaMar) := by sorry -- proof needed
  rw [h₁] at h₂
  rw [h₂]
  norm_num
  sorry

end fraction_of_pizza_eaten_by_TreShawn_l269_269861


namespace calculate_K_3_15_5_l269_269234

def K (x y z : ℝ) : ℝ := x / y + y / z + z / x

theorem calculate_K_3_15_5 : K 3 15 5 = 73 / 15 := by
  sorry

end calculate_K_3_15_5_l269_269234


namespace cos_105_sub_alpha_l269_269288

variable (α : ℝ)

-- Condition
def condition : Prop := Real.cos (75 * Real.pi / 180 + α) = 1 / 2

-- Statement
theorem cos_105_sub_alpha (h : condition α) : Real.cos (105 * Real.pi / 180 - α) = -1 / 2 :=
by
  sorry

end cos_105_sub_alpha_l269_269288


namespace unknown_rate_correct_l269_269523

noncomputable def find_unknown_rate 
  (cost_3_towels : ℕ := 3 * 100) 
  (cost_5_towels : ℕ := 5 * 150) 
  (num_towels : ℕ := 3 + 5 + 2) 
  (total_average_cost : ℕ := 1650) 
  (average_price : ℕ := 165) : ℕ :=
  let total_known_cost := cost_3_towels + cost_5_towels
  let total_cost := average_price * num_towels
  in (total_cost - total_known_cost) / 2

theorem unknown_rate_correct : find_unknown_rate = 300 :=
by
  sorry

end unknown_rate_correct_l269_269523


namespace sum_first_odd_numbers_not_prime_l269_269345

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_first_odd_numbers_not_prime :
  ¬ (is_prime (1 + 3)) ∧
  ¬ (is_prime (1 + 3 + 5)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7 + 9)) :=
by
  sorry

end sum_first_odd_numbers_not_prime_l269_269345


namespace unique_paintings_count_l269_269341

-- Given the conditions of the problem:
-- - N = 6 disks
-- - 3 disks are blue
-- - 2 disks are red
-- - 1 disk is green
-- - Two paintings that can be obtained from one another by a rotation or a reflection are considered the same

-- Define a theorem to calculate the number of unique paintings.
theorem unique_paintings_count : 
    ∃ n : ℕ, n = 13 :=
sorry

end unique_paintings_count_l269_269341


namespace min_value_of_expression_l269_269795

noncomputable def smallest_value (a b c : ℕ) : ℤ :=
  3 * a - 2 * a * b + a * c

theorem min_value_of_expression : ∃ (a b c : ℕ), 0 < a ∧ a < 7 ∧ 0 < b ∧ b ≤ 3 ∧ 0 < c ∧ c ≤ 4 ∧ smallest_value a b c = -12 := by
  sorry

end min_value_of_expression_l269_269795


namespace measure_α_l269_269515

noncomputable def measure_α_proof (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : ℝ :=
  let α := 120
  α

theorem measure_α (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : measure_α_proof AB BC h1 h2 = 120 :=
  sorry

end measure_α_l269_269515


namespace Riku_stickers_more_times_l269_269394

theorem Riku_stickers_more_times (Kristoff_stickers Riku_stickers : ℕ) 
  (h1 : Kristoff_stickers = 85) (h2 : Riku_stickers = 2210) : 
  Riku_stickers / Kristoff_stickers = 26 := 
by
  sorry

end Riku_stickers_more_times_l269_269394


namespace savings_fraction_l269_269524

variable (P : ℝ) -- worker's monthly take-home pay, assumed to be a real number
variable (f : ℝ) -- fraction of the take-home pay that she saves each month, assumed to be a real number

-- Condition: 12 times the fraction saved monthly should equal 8 times the amount not saved monthly.
axiom condition : 12 * f * P = 8 * (1 - f) * P

-- Prove: the fraction saved each month is 2/5
theorem savings_fraction : f = 2 / 5 := 
by
  sorry

end savings_fraction_l269_269524


namespace complex_conjugate_of_z_l269_269750

-- Define z based on the given expression
noncomputable def z : ℂ := (2 + complex.i) / (1 + complex.i^2 + complex.i^5)

-- State the theorem to prove the complex conjugate of z is 1 + 2i
theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * complex.i :=
sorry

end complex_conjugate_of_z_l269_269750


namespace find_positive_integers_solution_l269_269252

theorem find_positive_integers_solution :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ (∑ i in Finset.range (m + 1), (n + i)) = 1000 ∧
               (m = 4 ∧ n = 198 ∨ m = 24 ∧ n = 28) :=
by
  sorry

end find_positive_integers_solution_l269_269252


namespace still_need_more_volunteers_l269_269809

def total_volunteers_needed : ℕ := 80
def students_volunteering_per_class : ℕ := 4
def number_of_classes : ℕ := 5
def teacher_volunteers : ℕ := 10
def total_student_volunteers : ℕ := students_volunteering_per_class * number_of_classes
def total_volunteers_so_far : ℕ := total_student_volunteers + teacher_volunteers

theorem still_need_more_volunteers : total_volunteers_needed - total_volunteers_so_far = 50 := by
  sorry

end still_need_more_volunteers_l269_269809


namespace solve_for_y_l269_269562

theorem solve_for_y (y : ℚ) : 
  y + 1 / 3 = 3 / 8 - 1 / 4 → y = -5 / 24 := 
by
  sorry

end solve_for_y_l269_269562


namespace speed_in_still_water_l269_269503

/--
A man can row upstream at 55 kmph and downstream at 65 kmph.
Prove that his speed in still water is 60 kmph.
-/
theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_upstream : upstream_speed = 55) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 60 := by
  sorry

end speed_in_still_water_l269_269503


namespace find_r1_s1_sum_l269_269359

noncomputable def r (x : ℤ) : ℤ := sorry -- define r(x) with the properties described
noncomputable def s (x : ℤ) : ℤ := sorry -- define s(x) with the properties described

theorem find_r1_s1_sum :
  (∃(r s : polynomial ℤ), (r.monic ∧ s.monic ∧ ¬r.degree.is_zero ∧ ¬s.degree.is_zero ∧
    r * s = polynomial.C (1 : ℤ) * X^6 + polynomial.C (-50 : ℤ) * X^3 + polynomial.C (1 : ℤ)) ∧
    r.eval 1 + s.eval 1 = 4) := 
sorry

end find_r1_s1_sum_l269_269359


namespace piece_distribution_l269_269853

def initialConditions := 
  (boxes: ι) (1 ≤ i ≤ 7) / \(XiaoMing and XiaoQing follows rule as described)

theorem piece_distribution 
  (whiteCount : Fin 7 -> Nat)
  (redCount : Fin 7 -> Nat) :
  whiteCount 1 = 57 ∧ whiteCount 2 = 0 ∧ whiteCount 3 = 58 ∧ whiteCount 4 = 0 ∧ whiteCount 5 = 0 ∧ whiteCount 6 = 29 ∧ whiteCount 7 = 56 ∧
  redCount 1 = 86 ∧ redCount 2 = 85 ∧ redCount 3 = 43 ∧ redCount 4 = 0 ∧ redCount 5 = 0 ∧ redCount 6 = 86 ∧ redCount 7 = 0 ∧
  (∀ i, 1 ≤ i ≤ 7, totalCount i = whiteCount i + redCount i) :=
  sorry

end piece_distribution_l269_269853


namespace castings_exist_l269_269017

noncomputable def casting_possible (n : ℕ) (a : ℕ → ℕ) (p k : ℕ) (h_p_prime : Nat.Prime p) (h_bounds : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≤ p) (h_pk_bound : p ≥ n) (h_k_n : k ≤ n) : Prop :=
∀ (roles : Fin n → Fin p), ∃ (castings : Fin (p^k) → Fin n → Fin p), ∀ (k_roles : Fin k → Fin n), ∃ day, ∀ i j, i ≠ j → roles i ≠ roles j → ∃ (pi pj : Fin p), castings day (k_roles i) = pi ∧ castings day (k_roles j) = pj

theorem castings_exist (n : ℕ) (a : ℕ → ℕ) (p k : ℕ) (h_p_prime : Nat.Prime p) (h_bounds : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≤ p) (h_pk_bound : p ≥ n) (h_k_n : k ≤ n) : casting_possible n a p k h_p_prime h_bounds h_pk_bound h_k_n :=
sorry

end castings_exist_l269_269017


namespace find_speed_of_first_train_l269_269488

-- Define the conditions as constants
constant length_train_one : ℝ := 260
constant length_train_two : ℝ := 240.04
constant speed_train_two : ℝ := 80
constant crossing_time : ℝ := 9

-- Define the speed of the first train to be determined
constant speed_train_one : ℝ

-- Statement to prove
theorem find_speed_of_first_train :
  (length_train_one + length_train_two) / (crossing_time / 3600) = speed_train_one + speed_train_two →
  speed_train_one = 120.016 := by
  sorry

end find_speed_of_first_train_l269_269488


namespace choose_math_class_representative_l269_269006

def number_of_boys : Nat := 26
def number_of_girls : Nat := 24

theorem choose_math_class_representative : number_of_boys + number_of_girls = 50 := 
by
  sorry

end choose_math_class_representative_l269_269006


namespace decreasing_function_on_pos_real_l269_269527

noncomputable def f1 (x : ℝ) : ℝ := Real.log x
noncomputable def f2 (x : ℝ) : ℝ := Real.exp (-x)
noncomputable def f3 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f4 (x : ℝ) : ℝ := - 1 / x

theorem decreasing_function_on_pos_real :
  ∀ (x : ℝ), 0 < x → 
  (f1' x > 0) ∧ (f2' x < 0) ∧ (f3' x > 0) ∧ (f4' x > 0) :=
by sorry

end decreasing_function_on_pos_real_l269_269527


namespace distance_to_origin_l269_269012

theorem distance_to_origin (x y : ℤ) (hx : x = -5) (hy : y = 12) :
  Real.sqrt (x^2 + y^2) = 13 := by
  rw [hx, hy]
  norm_num
  sorry

end distance_to_origin_l269_269012


namespace lisa_caffeine_l269_269453

theorem lisa_caffeine (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_drank : ℕ) : caffeine_per_cup = 80 → daily_goal = 200 → cups_drank = 3 → (caffeine_per_cup * cups_drank - daily_goal) = 40 :=
by
  -- This is a theorem statement, thus no proof is provided here.
  sorry

end lisa_caffeine_l269_269453


namespace adam_initial_books_l269_269208

theorem adam_initial_books (B : ℕ) (h1 : B - 11 + 23 = 45) : B = 33 := 
by
  sorry

end adam_initial_books_l269_269208


namespace photo_arrangements_l269_269908

theorem photo_arrangements (P : Fin 7 → Type) (A B C D : Fin 7) :
  let AB := (A, B) in
  let CD := {x // x ≠ C ∧ x ≠ D} in -- Remaining persons excluding specific ones
  let arrangements := {x // x ∉ {A, B, C, D}}.card! * 4! *
                      {AB_combination : Fin 2 // true}.card! *
                      {CD_combination : Fin 5 // true}.card! in
  arrangements = 960 :=
sorry

end photo_arrangements_l269_269908


namespace find_P_coordinates_l269_269569

-- Define the given points and the conditions
def P1 := (4 : ℝ, 1 : ℝ, 2 : ℝ)
def dist := (√30 : ℝ)

-- Problem: Prove that there exist points on the x-axis at distance sqrt(30) from P1.
theorem find_P_coordinates (x : ℝ) :
  (√((x - 4)^2 + (0 - 1)^2 + (0 - 2)^2) = √30) → (x = 9 ∨ x = -1) :=
by
  sorry

end find_P_coordinates_l269_269569


namespace smallest_moves_to_blackout_chessboard_l269_269170

theorem smallest_moves_to_blackout_chessboard : 
  ∀ (n : ℕ), (n = 98) → 
  let chessboard := (fin n) × (fin n)
  ∀ (color : chessboard → bool) 
    (∀ (i j : fin n), color ⟨i.val, j.val⟩ = (i.val + j.val) % 2 = 0) -- The condition of board being alternately colored
  , let moves := {(i1 j1 i2 j2 : fin n) | i1 ≤ i2 ∧ j1 ≤ j2} -- Define a move as a rectangular subset
  ∃ m, (m = 98) ∧ (∀ (move : moves), 
  ∀ (state : chessboard → bool) 
    ((∀ (x y : fin n), state ⟨x.val, y.val⟩ = color ⟨x.val, y.val⟩) → 
    (∀ (x : fin (n * m)), state (x / m, x % m) = ff))) -- ff represents black

end smallest_moves_to_blackout_chessboard_l269_269170


namespace horizontal_asymptote_rational_func_l269_269671

noncomputable def rational_func (x : ℝ) : ℝ :=
  (8 * x^3 + 3 * x^2 + 6 * x + 4) / (2 * x^3 + x^2 + 5 * x + 2)

theorem horizontal_asymptote_rational_func :
  ∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x : ℝ, x > M → |rational_func(x) - L| < ε :=
  ∃ L : ℝ, y = 4 :=
  sorry

end horizontal_asymptote_rational_func_l269_269671


namespace volume_of_pyramid_is_correct_l269_269186

-- Conditions
def square_area (ABCD : real) := ABCD = 256
def triangle_area_ABE (ABE : real) := ABE = 120
def triangle_area_CDE (CDE : real) := CDE = 108

-- Volume Calculation
def pyramid_volume (volume : real) : Prop :=
  ∀ (ABCD ABE CDE : real), 
    square_area ABCD → 
    triangle_area_ABE ABE → 
    triangle_area_CDE CDE → 
    volume = 1156.61

-- Theorem Statement
theorem volume_of_pyramid_is_correct : pyramid_volume 1156.61 :=
by
  intros ABCD ABE CDE hABCD hABE hCDE
  -- Proof goes here
  sorry

end volume_of_pyramid_is_correct_l269_269186


namespace second_player_wins_l269_269448

noncomputable def optimal_play : Prop :=
  let plays := list (fin 5 → fin 2 → ℕ)
  in ∀ plays (player_first_turn : plays 0 = 1 ∨ plays 0 = 2), 
       ∃ final_digits : fin 5 → fin 2,
       (∀ pos : fin 5, plays pos final_digits.pos < 3) →
       (player_first_turn play sum of all plays from 0 to 4)
       (digits :| 3 ≠0)

theorem second_player_wins : optimal_play :=
sorry

end second_player_wins_l269_269448


namespace Heracles_age_l269_269980

variable (A H : ℕ)

theorem Heracles_age :
  (A = H + 7) →
  (A + 3 = 2 * H) →
  H = 10 :=
by
  sorry

end Heracles_age_l269_269980


namespace kelly_grade_is_42_l269_269030

noncomputable def jenny_grade := 95

noncomputable def jason_grade := jenny_grade - 25

noncomputable def bob_grade := jason_grade / 2

noncomputable def kelly_grade := bob_grade * 1.2

theorem kelly_grade_is_42 : kelly_grade = 42 := by
  sorry

end kelly_grade_is_42_l269_269030


namespace sandra_beignets_l269_269403

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end sandra_beignets_l269_269403


namespace simplify_and_evaluate_l269_269832

theorem simplify_and_evaluate (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ((a - 2) / (a^2 + 2 * a) - (a - 1) / (a^2 + 4 * a + 4)) / ((a - 4) / (a + 2)) = 1 / 3 :=
by sorry

end simplify_and_evaluate_l269_269832


namespace midpoint_MN_on_circumcircle_l269_269361

theorem midpoint_MN_on_circumcircle (M N : Point) (ABC : Triangle) :
  let B := ABC.B
  let C := ABC.C
  -- M is the intersection of the internal angle bisector of ∠B and the external angle bisector of ∠C
  ∃ (M : Point), is_interior_angle_bisector B ABC.A M ∧ is_exterior_angle_bisector C ABC.A M →
  -- N is the intersection of the external angle bisector of ∠B and the internal angle bisector of ∠C
  ∃ (N : Point), is_exterior_angle_bisector B ABC.A N ∧ is_interior_angle_bisector C ABC.A N →
  -- Midpoint P of segment MN
  let P := midpoint M N in
  lies_on_circumcircle P ABC :=
sorry

end midpoint_MN_on_circumcircle_l269_269361


namespace remainder_7_pow_150_mod_4_l269_269865

theorem remainder_7_pow_150_mod_4 : (7 ^ 150) % 4 = 1 :=
by
  sorry

end remainder_7_pow_150_mod_4_l269_269865


namespace find_distance_PQ_of_polar_coords_l269_269362

theorem find_distance_PQ_of_polar_coords (α β : ℝ) (h : β - α = 2 * Real.pi / 3) :
  let P := (5, α)
  let Q := (12, β)
  dist P Q = Real.sqrt 229 :=
by
  sorry

end find_distance_PQ_of_polar_coords_l269_269362


namespace factorial_25_trailing_zeros_l269_269673

theorem factorial_25_trailing_zeros : nat.trailing_zeros (nat.factorial 25) = 6 :=
sorry

end factorial_25_trailing_zeros_l269_269673


namespace store_owner_oil_l269_269952

noncomputable def liters_of_oil (volume_per_bottle : ℕ) (number_of_bottles : ℕ) : ℕ :=
  (volume_per_bottle * number_of_bottles) / 1000

theorem store_owner_oil : liters_of_oil 200 20 = 4 := by
  sorry

end store_owner_oil_l269_269952


namespace preferred_pets_combination_l269_269942

-- Define the number of puppies, kittens, and hamsters
def num_puppies : ℕ := 20
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 12

-- State the main theorem to prove, that the number of ways Alice, Bob, and Charlie 
-- can buy their preferred pets is 2400
theorem preferred_pets_combination : num_puppies * num_kittens * num_hamsters = 2400 :=
by
  sorry

end preferred_pets_combination_l269_269942


namespace distance_of_route_l269_269730

-- Define the conditions
def round_trip_time : ℝ := 1 -- in hours
def avg_speed : ℝ := 3 -- in miles per hour
def return_speed : ℝ := 6.000000000000002 -- in miles per hour

-- Problem statement to prove
theorem distance_of_route : 
  ∃ (D : ℝ), 
  2 * D = avg_speed * round_trip_time ∧ 
  D = 1.5 := 
by
  sorry

end distance_of_route_l269_269730


namespace train_and_car_number_exists_l269_269804

theorem train_and_car_number_exists 
  (SECRET OPEN ANSWER YOUR OPENED : ℕ)
  (h1 : SECRET - OPEN = ANSWER - YOUR)
  (h2 : SECRET - OPENED = 20010)
  (distinct_digits : ∀ (a b : ℕ), a ≠ b → ∀ (c : ℕ), (digits a) ≠ (digits b))
  (same_digits : ∀ (a b : ℕ), a = b → ∀ (c : ℕ), (digits a) = (digits b)) :
  ∃ (train car : ℕ), train = 392 ∧ car = 2 :=
sorry

end train_and_car_number_exists_l269_269804


namespace conditional_probability_l269_269685

-- Define the given probabilities
def P_math : ℝ := 0.16
def P_chinese : ℝ := 0.07
def P_math_and_chinese : ℝ := 0.04

-- Define the conditional probability
def P_math_given_chinese : ℝ := P_math_and_chinese / P_chinese

-- The main statement to prove the probability that a student failed the math test given that they failed the Chinese test
theorem conditional_probability : P_math_given_chinese = 4 / 7 := sorry

end conditional_probability_l269_269685


namespace mixed_number_sum_l269_269165

theorem mixed_number_sum : 
  (4/5 + 9 * 4/5 + 99 * 4/5 + 999 * 4/5 + 9999 * 4/5 + 1 = 11111) := by
  sorry

end mixed_number_sum_l269_269165


namespace total_points_proof_l269_269041

variable (shots2 : Nat) (shots3 : Nat) (shots_period : Nat) (periods : Nat)

def points_in_4_minutes (shots2 : Nat) (shots3 : Nat) : Nat :=
  2 * shots2 + 3 * shots3

def periods_in_a_period (total_minutes : Nat) (interval_minutes : Nat) : Nat :=
  total_minutes / interval_minutes

def points_in_one_period (shots2 : Nat) (shots3 : Nat) (total_minutes : Nat) (interval_minutes : Nat) : Nat :=
  points_in_4_minutes shots2 shots3 * periods_in_a_period total_minutes interval_minutes

def total_points_scored (shots2 : Nat) (shots3 : Nat) (total_minutes : Nat) (interval_minutes : Nat) (periods : Nat) : Nat :=
  points_in_one_period shots2 shots3 total_minutes interval_minutes * periods

theorem total_points_proof (shots2 shots3 : Nat) (total_minutes interval_minutes periods : Nat)
    (h1 : shots2 = 2) (h2 : shots3 = 1) (h3 : total_minutes = 12) (h4 : interval_minutes = 4) (h5 : periods = 2) :
    total_points_scored shots2 shots3 total_minutes interval_minutes periods = 42 :=
  by
  rw [h1, h2, h3, h4, h5]
  unfold total_points_scored points_in_one_period points_in_4_minutes periods_in_a_period
  simp
  -- Calculation steps that will be proven here
  sorry

end total_points_proof_l269_269041


namespace smallest_x_plus_y_l269_269619

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269619


namespace domain_range_eq_l269_269961

variable (x : ℝ)

def y_ex_ln : ℝ := exp (log x)

def y_abs : ℝ := abs x

def y_recip_sqrt : ℝ := 1 / sqrt x

def y_exp_base2 : ℝ := 2 ^ x

def y_log_abs : ℝ := log (abs x)

theorem domain_range_eq :
  (set.Ioi 0) = {x : ℝ | ∃ y : ℝ, y = y_recip_sqrt x} ↔ 
  (set.Ioi 0) = {x : ℝ | ∃ y : ℝ, y = y_ex_ln x} :=
sorry

end domain_range_eq_l269_269961


namespace length_of_median_l269_269138

noncomputable theory

variable {D E F M : Type}

def isosceles_triangle (D E F : Type) (a b : ℕ) : Prop :=
  a = b

def median_bisects (E F M : Type) (c : ℕ) : Prop :=
  2 * c = 12

theorem length_of_median 
  (h₁ : isosceles_triangle D E F 10 10)
  (h₂ : median_bisects E F M 6) : 
  ∃ DM : ℕ, DM = 8 :=
by
  sorry

end length_of_median_l269_269138


namespace find_x_l269_269872

theorem find_x (x : ℕ) : (4 + x) / (7 + x) = 3 / 4 → x = 5 :=
by
  sorry

end find_x_l269_269872


namespace thirty_third_digit_after_decimal_of_sqrt10_plus_3_pow_2001_l269_269139

noncomputable def sqrt10_plus_3_pow_2001 := (Real.sqrt 10 + 3) ^ 2001

theorem thirty_third_digit_after_decimal_of_sqrt10_plus_3_pow_2001 :
  let d := 10 ^ 33 in
  ((sqrt10_plus_3_pow_2001 * d) - Real.floor (sqrt10_plus_3_pow_2001 * d)) * 10 ≥ 0 ∧
  ((sqrt10_plus_3_pow_2001 * d) - Real.floor (sqrt10_plus_3_pow_2001 * d)) * 10 < 10 :=
sorry

end thirty_third_digit_after_decimal_of_sqrt10_plus_3_pow_2001_l269_269139


namespace constant_sequence_l269_269162

theorem constant_sequence {a : ℕ → ℕ} 
  (h : ∀ n : ℕ, (∑ i in Finset.range (n+1), a i) * (∑ i in Finset.range (n+1), (1 / (a i : ℝ))) ≤ (n + 1 : ℝ)^2 + 2019) : 
  ∃ c : ℕ, ∀ n : ℕ, a n = c :=
begin
  sorry -- Proof to be provided
end

end constant_sequence_l269_269162


namespace xiaolong_correct_answers_l269_269684

/-- There are 50 questions in the exam. Correct answers earn 3 points each,
incorrect answers deduct 1 point each, and unanswered questions score 0 points.
Xiaolong scored 120 points. Prove that the maximum number of questions 
Xiaolong answered correctly is 42. -/
theorem xiaolong_correct_answers :
  ∃ (x y : ℕ), 3 * x - y = 120 ∧ x + y = 48 ∧ x ≤ 50 ∧ y ≤ 50 ∧ x = 42 :=
by
  sorry

end xiaolong_correct_answers_l269_269684


namespace order_of_logs_l269_269745

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem order_of_logs (a_def : a = Real.log 6 / Real.log 3)
                      (b_def : b = Real.log 10 / Real.log 5)
                      (c_def : c = Real.log 14 / Real.log 7) : a > b ∧ b > c := 
by
  sorry

end order_of_logs_l269_269745


namespace number_of_expansion_terms_is_seven_l269_269257

noncomputable def num_terms_in_expansion : ℕ :=
  let expr := ((a + 2 * b) ^ 3 * (a - 2 * b) ^ 3) ^ 2
  let simplified_expr := (a ^ 2 - 4 * b ^ 2) ^ 6
  (simplified_expr.expand_into_terms).length

theorem number_of_expansion_terms_is_seven (a b : ℝ) : num_terms_in_expansion = 7 :=
  sorry

end number_of_expansion_terms_is_seven_l269_269257


namespace nested_evaluation_l269_269737

def M (x : ℝ) : ℝ := 3 * Real.sqrt x
def P (x : ℝ) : ℝ := x ^ 3

theorem nested_evaluation : M (P (M (P (M (P 4))))) = 3 * Real.sqrt (372984 * 24 * Real.sqrt 24) := by
  sorry

end nested_evaluation_l269_269737


namespace double_cube_volume_l269_269480

theorem double_cube_volume (v : ℝ) (h₁ : v = 64) (h₂: ∀ x : ℝ, x^3 = v) :
  let new_v := (2 * (∛64)) ^ 3 in 
  new_v = 512 := 
by
  sorry

end double_cube_volume_l269_269480


namespace measure_angle_BCA_l269_269534

theorem measure_angle_BCA 
  (BCD_angle : ℝ)
  (CBA_angle : ℝ)
  (sum_angles : BCD_angle + CBA_angle + BCA_angle = 190)
  (BCD_right : BCD_angle = 90)
  (CBA_given : CBA_angle = 70) :
  BCA_angle = 30 :=
by
  sorry

end measure_angle_BCA_l269_269534


namespace sandra_beignets_l269_269401

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end sandra_beignets_l269_269401


namespace find_angle_2_l269_269577

theorem find_angle_2
  (ABCD : Type)
  (is_rectangle : ∀ (A B C D : ABCD), 
    ∀ (angles : ABCD → ℝ), 
    angles A = 90 ∧ angles B = 90 ∧ angles C = 90 ∧ angles D = 90)
  (fold_condition : ∀ (DCF DEF : Triangle), folds_onto DCF DEF)
  (vertex_lands_on_AB : ∀ (E : Point) (AB : Line), lands_on E AB)
  (angle_1 : ∀ (DCF : Triangle), angle_1 DCF = 20) :
  (∀ (DEF : Triangle), angle_2 DEF = 40) :=
  by
  sorry

end find_angle_2_l269_269577


namespace Jill_ball_difference_l269_269720

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end Jill_ball_difference_l269_269720


namespace ratio_of_girls_to_boys_l269_269683

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) 
  (h1 : total_students = 26) 
  (h2 : girls = boys + 6) 
  (h3 : girls + boys = total_students) : 
  (girls : ℚ) / boys = 8 / 5 :=
by
  sorry

end ratio_of_girls_to_boys_l269_269683


namespace find_number_l269_269169

theorem find_number (x : ℝ) : (0.5 * x = 0.2 * 650 + 190) -> x = 640 :=
by
  intro h
  calc
    0.5 * x = 0.2 * 650 + 190 := h
    ... = 320 by norm_num
    640 = (320 : ℝ / 0.5 : ℝ : by norm_num :mul_div_cancel: by  sorry):


end find_number_l269_269169


namespace total_vehicles_correct_l269_269690

def num_trucks : ℕ := 20
def num_tanks (num_trucks : ℕ) : ℕ := 5 * num_trucks
def total_vehicles (num_trucks : ℕ) (num_tanks : ℕ) : ℕ := num_trucks + num_tanks

theorem total_vehicles_correct : total_vehicles num_trucks (num_tanks num_trucks) = 120 := by
  sorry

end total_vehicles_correct_l269_269690


namespace scientific_notation_equivalence_l269_269244

/-- The scientific notation for 20.26 thousand hectares in square meters is equal to 2.026 × 10^9. -/
theorem scientific_notation_equivalence :
  (20.26 * 10^3 * 10^4) = 2.026 * 10^9 := 
sorry

end scientific_notation_equivalence_l269_269244


namespace projection_correct_l269_269259

noncomputable def vec_projection 
  (u v : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 + u.4 * v.4 in
let dot_vv := v.1 * v.1 + v.2 * v.2 + v.3 * v.3 + v.4 * v.4 in
let scalar := dot_uv / dot_vv in
(scalar * v.1, scalar * v.2, scalar * v.3, scalar * v.4)

def vector_4 := (4, -1, 5, 2) : ℝ × ℝ × ℝ × ℝ
def direction_vector := (3, 1, -2, 3) : ℝ × ℝ × ℝ × ℝ
def expected_projection := (21/23, 7/23, -14/23, 21/23) : ℝ × ℝ × ℝ × ℝ

theorem projection_correct :
  vec_projection vector_4 direction_vector = expected_projection :=
sorry

end projection_correct_l269_269259


namespace bernoulli_trial_probability_7_successes_l269_269894

theorem bernoulli_trial_probability_7_successes :
  let n := 7
  let k := 7
  let p := (2 : ℝ) / 7
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (128 / 823543) :=
by
  sorry

end bernoulli_trial_probability_7_successes_l269_269894


namespace eccentricity_is_sqrt_5_l269_269287

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
  let c := Real.sqrt (5) * a in
  Real.sqrt ((c^2) / (a^2))

theorem eccentricity_is_sqrt_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  eccentricity_of_hyperbola a b h1 h2 = Real.sqrt 5 := by
  sorry

end eccentricity_is_sqrt_5_l269_269287


namespace domain_f_when_m_8_range_of_m_when_fx_ge_1_l269_269644

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := Real.log2 (abs (x + 2) + abs (x - 2) - m)

-- (1) Prove the domain of f(x) when m = 8
theorem domain_f_when_m_8 :
  {x : ℝ | x > 4 ∨ x < -4} = {x : ℝ | abs (x + 2) + abs (x - 2) - 8 > 0} :=
sorry

-- (2) Prove the range of m given f(x) ≥ 1 ∀ x ∈ ℝ
theorem range_of_m_when_fx_ge_1 :
  (∀ x : ℝ, f x m ≥ 1) → m ≤ 2 :=
sorry

end domain_f_when_m_8_range_of_m_when_fx_ge_1_l269_269644


namespace remainder_is_210_l269_269554

-- Define necessary constants and theorems
def x : ℕ := 2^35
def dividend : ℕ := 2^210 + 210
def divisor : ℕ := 2^105 + 2^63 + 1

theorem remainder_is_210 : (dividend % divisor) = 210 :=
by 
  -- Assume the calculation steps in the preceding solution are correct.
  -- No need to manually re-calculate as we've directly taken from the solution.
  sorry

end remainder_is_210_l269_269554


namespace square_side_length_l269_269949

theorem square_side_length 
  (P : ℝ) (hP : P = 48) 
  (A : ℝ) (hA : A = 144) 
  (s : ℝ) : s = 12 :=
by
  -- Conditions provided
  have hp : P = 4 * s := by sorry  -- Perimeter formula
  have ha : A = s * s := by sorry  -- Area formula
  -- Given conditions
  rw [hP, hA] at *,
  sorry

end square_side_length_l269_269949


namespace divide_good_points_into_three_sets_l269_269417

-- A rational point in the plane is called good.
def is_rational (x : ℚ × ℚ) : Prop := true

-- Given a good point, prove we can divide all good points into three sets satisfying the two conditions
theorem divide_good_points_into_three_sets :
  ∃ (A B C : set (ℚ × ℚ)),
    (∀ (x ∈ A ∪ B ∪ C), is_rational x) ∧
    ( ∀ (center : ℚ × ℚ), is_rational center →
      ∀ (r : ℚ), r > 0 →
      ∃ (a ∈ A, b ∈ B, c ∈ C), (a ∈ ball center r) ∧ (b ∈ ball center r) ∧ (c ∈ ball center r)
    ) ∧
    ( ∀ (a ∈ A) (b ∈ B) (c ∈ C),
      ¬(∃ k : ℚ, k * (fst b - fst a) = (snd b - snd a) ∧ k * (fst c - fst a) = (snd c - snd a))
    ) :=
sorry

end divide_good_points_into_three_sets_l269_269417


namespace vertices_edges_sum_impossible_l269_269536

-- Define the vertices and edges
def vertices : List ℕ := [a1, a2, a3, a4, a5, a6, a7, a8]
def edges : List (ℕ → ℕ → ℕ) := 
  [ λ x y, Nat.gcd a1 a2,
    λ x y, Nat.gcd a1 a3,
    λ x y, Nat.gcd a1 a4,
    λ x y, Nat.gcd a2 a3,
    λ x y, Nat.gcd a2 a5,
    λ x y, Nat.gcd a3 a6,
    λ x y, Nat.gcd a4 a5,
    λ x y, Nat.gcd a4 a7,
    λ x y, Nat.gcd a5 a6,
    λ x y, Nat.gcd a6 a8,
    λ x y, Nat.gcd a7 a8,
    λ x y, Nat.gcd a3 a8]

-- Define the sums
def vertices_sum : ℕ := vertices.sum
def edges_sum : ℕ := edges.sum (λ f, f 0 0)

-- The theorem stating the impossibility
theorem vertices_edges_sum_impossible : 
  ∀ (a1 a2 a3 a4 a5 a6 a7 a8 : ℕ),
  list.pairwise Nat.coprime [a1, a2, a3, a4, a5, a6, a7, a8] →
  vertices_sum ≠ edges_sum :=
by
  sorry

end vertices_edges_sum_impossible_l269_269536


namespace function_equivalence_l269_269111

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

theorem function_equivalence (f : ℝ → ℝ) :
  (∀ y, f (f⁻¹ (-y)) = f⁻¹ (f (-y))) → is_odd_function f :=
by
  sorry

end function_equivalence_l269_269111


namespace seashells_count_l269_269859

theorem seashells_count (total_seashells broken_seashells : ℕ) (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) : total_seashells - broken_seashells = 3 := by
  sorry

end seashells_count_l269_269859


namespace percentage_of_students_choose_harvard_l269_269340

theorem percentage_of_students_choose_harvard
  (total_applicants : ℕ)
  (acceptance_rate : ℝ)
  (students_attend_harvard : ℕ)
  (students_attend_other : ℝ)
  (percentage_attended_harvard : ℝ) :
  total_applicants = 20000 →
  acceptance_rate = 0.05 →
  students_attend_harvard = 900 →
  students_attend_other = 0.10 →
  percentage_attended_harvard = ((students_attend_harvard / (total_applicants * acceptance_rate)) * 100) →
  percentage_attended_harvard = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_of_students_choose_harvard_l269_269340


namespace power_function_value_l269_269645

theorem power_function_value :
  ∀ (f : ℝ → ℝ) (α : ℝ),
    (∀ x, f(x) = x^α) →
    f(2) = real.sqrt 2 →
    f(1 / 3) = real.sqrt 3 / 3 :=
by
  sorry

end power_function_value_l269_269645


namespace combined_work_time_l269_269937

theorem combined_work_time (man_rate : ℚ := 1/5) (wife_rate : ℚ := 1/7) (son_rate : ℚ := 1/15) :
  (man_rate + wife_rate + son_rate)⁻¹ = 105 / 43 :=
by
  sorry

end combined_work_time_l269_269937


namespace germination_percentage_second_plot_l269_269264

theorem germination_percentage_second_plot :
  ∀ (seeds_first_plot seeds_second_plot : ℕ)
    (germ_first_plot_percent germ_total_percent : ℕ),
    seeds_first_plot = 300 →
    seeds_second_plot = 200 →
    germ_first_plot_percent = 15 →
    germ_total_percent = 23 →
    (70 * 100 / seeds_second_plot) = 35 := by
  intros seeds_first_plot seeds_second_plot germ_first_plot_percent germ_total_percent
  intros h_seed1 h_seed2 h_germ1 h_germ_tot
  have h_total_seeds : seeds_first_plot + seeds_second_plot = 500 := by
    rw [h_seed1, h_seed2]
  have h_germ_first : seeds_first_plot * germ_first_plot_percent / 100 = 45 := by
    rw [h_seed1, h_germ1]
  have h_germ_total : 500 * germ_total_percent / 100 = 115 := by
    rw [h_total_seeds, h_germ_tot]
  have h_germ_second : 115 - seeds_first_plot * germ_first_plot_percent / 100 = 70 := by
    rw [h_germ_first, h_germ_total]
  show (70 * 100 / seeds_second_plot) = 35
  rwa h_seed2
  sorry

end germination_percentage_second_plot_l269_269264


namespace distance_to_focus_l269_269584

open Real

namespace ParabolaDistanceProof

-- Define the parabola
def parabola (x y : ℝ) := y^2 = -12 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (-3, 0)

-- The point P on the parabola
variable {P : ℝ × ℝ}

-- The condition that P is on the parabola
def on_parabola := parabola P.1 P.2

-- The condition that distance from P to y-axis is 1
def distance_to_y_axis := abs P.1 = 1

-- The goal is to prove the distance from P to the focus is 4
theorem distance_to_focus
  (h1 : on_parabola)
  (h2 : distance_to_y_axis) :
  dist P focus = 4 := sorry

end ParabolaDistanceProof

end distance_to_focus_l269_269584


namespace smallest_possible_sum_l269_269606

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l269_269606


namespace binary_arithmetic_l269_269994

def a : ℕ := 0b10110  -- 10110_2
def b : ℕ := 0b1101   -- 1101_2
def c : ℕ := 0b11100  -- 11100_2
def d : ℕ := 0b11101  -- 11101_2
def e : ℕ := 0b101    -- 101_2

theorem binary_arithmetic :
  (a + b - c + d + e) = 0b101101 := by
  sorry

end binary_arithmetic_l269_269994


namespace first_divisor_l269_269875

theorem first_divisor (d x : ℕ) (h1 : ∃ k : ℕ, x = k * d + 11) (h2 : ∃ m : ℕ, x = 9 * m + 2) : d = 3 :=
sorry

end first_divisor_l269_269875


namespace total_vehicles_correct_l269_269689

def num_trucks : ℕ := 20
def num_tanks (num_trucks : ℕ) : ℕ := 5 * num_trucks
def total_vehicles (num_trucks : ℕ) (num_tanks : ℕ) : ℕ := num_trucks + num_tanks

theorem total_vehicles_correct : total_vehicles num_trucks (num_tanks num_trucks) = 120 := by
  sorry

end total_vehicles_correct_l269_269689


namespace angle_B_eq_60deg_l269_269001

-- Assume a, b, c are the sides of a triangle ΔABC with sides given in the ratios
variables {a b c : ℝ}
variables (h : a / b = 1 / (Real.sqrt 3) ∧ b / c = (Real.sqrt 3) / 2)

theorem angle_B_eq_60deg (h : a / b = 1 / (Real.sqrt 3) ∧ b / c = (Real.sqrt 3) / 2) :
  ∃ B : ℝ, B = 60 ∧ cos B = 1 / 2 :=
by
  sorry

end angle_B_eq_60deg_l269_269001


namespace volume_intersection_l269_269240

-- Define the first octahedron as a condition
def octahedron1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 2

-- Define the second octahedron as a condition, centered at (1,1,1)
def octahedron2 (x y z : ℝ) : Prop := abs (x - 1) + abs (y - 1) + abs (z - 1) ≤ 2

-- Define the final theorem stating the volume of the intersection
theorem volume_intersection : 
  ∀ (x y z : ℝ), octahedron1 x y z ∧ octahedron2 x y z → 
  (volume_of_octants x y z = 4/3) := 
sorry -- Proof goes here

end volume_intersection_l269_269240


namespace total_points_proof_l269_269042

variable (shots2 : Nat) (shots3 : Nat) (shots_period : Nat) (periods : Nat)

def points_in_4_minutes (shots2 : Nat) (shots3 : Nat) : Nat :=
  2 * shots2 + 3 * shots3

def periods_in_a_period (total_minutes : Nat) (interval_minutes : Nat) : Nat :=
  total_minutes / interval_minutes

def points_in_one_period (shots2 : Nat) (shots3 : Nat) (total_minutes : Nat) (interval_minutes : Nat) : Nat :=
  points_in_4_minutes shots2 shots3 * periods_in_a_period total_minutes interval_minutes

def total_points_scored (shots2 : Nat) (shots3 : Nat) (total_minutes : Nat) (interval_minutes : Nat) (periods : Nat) : Nat :=
  points_in_one_period shots2 shots3 total_minutes interval_minutes * periods

theorem total_points_proof (shots2 shots3 : Nat) (total_minutes interval_minutes periods : Nat)
    (h1 : shots2 = 2) (h2 : shots3 = 1) (h3 : total_minutes = 12) (h4 : interval_minutes = 4) (h5 : periods = 2) :
    total_points_scored shots2 shots3 total_minutes interval_minutes periods = 42 :=
  by
  rw [h1, h2, h3, h4, h5]
  unfold total_points_scored points_in_one_period points_in_4_minutes periods_in_a_period
  simp
  -- Calculation steps that will be proven here
  sorry

end total_points_proof_l269_269042


namespace complex_conjugate_of_z_l269_269779

theorem complex_conjugate_of_z :
  let z := (2 + complex.i) / (1 + complex.i ^ 2 + complex.i ^ 5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269779


namespace a12_equals_66_l269_269714

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0     => 2
  | n + 1 => if (n % 2 = 0) then sequence (n-1) + 2 else 2 * sequence (n-2)

theorem a12_equals_66 : sequence 11 = 66 :=
  sorry

end a12_equals_66_l269_269714


namespace expected_value_of_win_is_162_l269_269499

noncomputable def expected_value_of_win : ℝ :=
  (1/8) * (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3)

theorem expected_value_of_win_is_162 : expected_value_of_win = 162 := 
by 
  sorry

end expected_value_of_win_is_162_l269_269499


namespace total_vehicles_is_120_l269_269687

def num_trucks : ℕ := 20
def num_tanks : ℕ := 5 * num_trucks
def total_vehicles : ℕ := num_tanks + num_trucks

theorem total_vehicles_is_120 : total_vehicles = 120 :=
by
  sorry

end total_vehicles_is_120_l269_269687


namespace find_cos_value_l269_269289

noncomputable def given_condition (α : ℝ) : Prop :=
  sin (π / 3 - α) = 1 / 4

theorem find_cos_value (α : ℝ) (h : given_condition α) : 
  cos (π / 3 + 2 * α) = -7 / 8 :=
sorry

end find_cos_value_l269_269289


namespace largest_k_divides_factorial_l269_269549

open Int

theorem largest_k_divides_factorial (n : ℕ) (p : ℕ) (k : ℕ) :
  n = 2023 → p = 17 →
  k = (∑ i in finset.range (nat.floor (log (n : ℝ) / log (p : ℝ))).succ,
    n / p ^ i) →
  k = 126 :=
by {
  intros h₁ h₂ h₃,
  subst h₁,
  subst h₂,
  have : (nat.floor (log (2023 : ℝ) / log (17 : ℝ))).succ = 3 :=
    by norm_num [log, Real.log, Real.log_div],
  rw this,
  norm_num [finset.range, sum],
  sorry
}

end largest_k_divides_factorial_l269_269549


namespace root_sum_abs_gt_6_l269_269321

variables (r1 r2 p : ℝ)

theorem root_sum_abs_gt_6 
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 9)
  (h3 : p^2 > 36) :
  |r1 + r2| > 6 :=
by sorry

end root_sum_abs_gt_6_l269_269321


namespace radius_of_tangent_circle_l269_269921

theorem radius_of_tangent_circle 
    (side_length : ℝ) 
    (tangent_angle : ℝ) 
    (sin_15 : ℝ)
    (circle_radius : ℝ) :
    side_length = 2 * Real.sqrt 3 →
    tangent_angle = 30 →
    sin_15 = (Real.sqrt 3 - 1) / (2 * Real.sqrt 2) →
    circle_radius = 2 :=
by sorry

end radius_of_tangent_circle_l269_269921


namespace center_of_symmetry_l269_269930

-- Define the given conditions
def has_axis_symmetry_x (F : Set (ℝ × ℝ)) : Prop := 
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, y) ∈ F

def has_axis_symmetry_y (F : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ F → (x, -y) ∈ F
  
-- Define the central proof goal
theorem center_of_symmetry (F : Set (ℝ × ℝ)) (H1: has_axis_symmetry_x F) (H2: has_axis_symmetry_y F) :
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, -y) ∈ F :=
sorry

end center_of_symmetry_l269_269930


namespace part1_part2_l269_269518

-- Step 1: Define the problem for a triangle with specific side length conditions and perimeter
theorem part1 (x : ℝ) (h1 : 2 * x + 2 * (2 * x) = 18) : 
  x = 18 / 5 ∧ 2 * x = 36 / 5 :=
by
  sorry

-- Step 2: Verify if an isosceles triangle with a side length of 4 cm can be formed
theorem part2 (a b c : ℝ) (h2 : a = 4 ∨ b = 4 ∨ c = 4) (h3 : a + b + c = 18) : 
  (a = 4 ∧ b = 7 ∧ c = 7 ∨ b = 4 ∧ a = 7 ∧ c = 7 ∨ c = 4 ∧ a = 7 ∧ b = 7) ∨
  (¬(a = 4 ∧ b + c <= a ∨ b = 4 ∧ a + c <= b ∨ c = 4 ∧ a + b <= c)) :=
by
  sorry

end part1_part2_l269_269518


namespace polygon_diagonals_l269_269496

theorem polygon_diagonals (n : ℕ) (h_n : n = 12) (h_angle : ∀ i : fin n, 150 = 150) : 
  (n * (n - 3)) / 2 = 54 :=
by
  have h1 : n = 12 := h_n
  sorry

end polygon_diagonals_l269_269496


namespace inequality_always_true_l269_269962

theorem inequality_always_true (x : ℝ) : (4 * x) / (x ^ 2 + 4) ≤ 1 := by
  sorry

end inequality_always_true_l269_269962


namespace divides_power_diff_l269_269734

theorem divides_power_diff (x : ℤ) (y z w : ℕ) (hy : y % 2 = 1) (hz : z % 2 = 1) (hw : w % 2 = 1) : 17 ∣ x^(y^(z^w)) - x^(y^z) := 
by
  sorry

end divides_power_diff_l269_269734


namespace parallel_lines_sufficient_not_necessary_l269_269163

theorem parallel_lines_sufficient_not_necessary (a : ℝ) (l1 l2 : ℝ → ℝ) 
(Hl1 : l1 = λ x y, a * x - y + 3 = 0) 
(Hl2 : l2 = λ x y, 2 * x - (a + 1) * y + 4 = 0) :
(∀ (x y : ℝ), (a = -2 → (l1 x y = 0 ∧ l2 x y = 0) ↔ (l1 x y = 0 ∧ l2 x y ≠ 0)))
∧ (¬ (∀ (x y : ℝ), (l1 x y = 0 ∧ l2 x y = 0) → a = -2)) :=
by
  -- Skipping the proof, adding a placeholder
  sorry

end parallel_lines_sufficient_not_necessary_l269_269163


namespace nth_equation_l269_269390

theorem nth_equation (n : ℕ) (h : 0 < n) : (- (n : ℤ)) * (n : ℝ) / (n + 1) = - (n : ℤ) + (n : ℝ) / (n + 1) :=
sorry

end nth_equation_l269_269390


namespace find_a_and_subsets_l269_269799

theorem find_a_and_subsets (a : ℝ)
  (U : Set ℝ := {2, 3, a^2 + 2 * a - 1})
  (A : Set ℝ := {abs (1 - 2 * a), 2})
  (complement_A : Set ℝ := {7}) :
  (U = {2, 3, 7} ∧ a = 2) ∧ (∃ (subsets : List (Set ℝ)), 
    subsets = [∅, {2}, {3}, {7}, {2, 3}, {3, 7}, {2, 7}, {2, 3, 7}]) :=
begin
  sorry
end

end find_a_and_subsets_l269_269799


namespace hyperbola_and_line_equation_l269_269629

-- Define the given points and hyperbola conditions
def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
def A : ℝ × ℝ := (2, -2)

-- Define the standard equation of the hyperbola passing through P and Q
def hyperbola_standard_eq (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1

-- Define the condition that the line passes through a point and forms a circle passing through A
def line_through_point (l : ℝ → ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (l t).fst = t * (l t).snd + 2

-- Define the circle condition
def circle_through_A (M N : ℝ × ℝ) : Prop :=
  let AM := (M.fst - 2, M.snd + 2)
  let AN := (N.fst - 2, N.snd + 2)
  AM.fst * AN.fst + AM.snd * AN.snd = 0

-- Prove the standard equation and line equation given conditions
theorem hyperbola_and_line_equation :
  (hyperbola_standard_eq P.fst P.snd ∧ hyperbola_standard_eq Q.fst Q.snd) →
  ∃ l : ℝ → ℝ × ℝ, line_through_point l ∧ circle_through_A (l 1) (l (-1)) ∧
  ((∀ t : ℝ, l t = (t + 2, t)) ∨ (∀ t : ℝ, l t = (7 * t + 2, t))) :=
by
  intro h
  sorry

end hyperbola_and_line_equation_l269_269629


namespace coeff_x_term_expansion_l269_269016

open BigOperators

noncomputable def find_coeff (n : ℕ) : ℤ :=
  let expr := (x : ℝ) in
  let poly := (x + 1) * (x^3 + 1 / real.sqrt x)^n in
  -- Extract the coefficient of x
  sorry

theorem coeff_x_term_expansion (n : ℕ) (h_sum_coeff : (2:ℝ)^n = 256) :
  find_coeff n = 7 :=
begin
  sorry
end

end coeff_x_term_expansion_l269_269016


namespace add_to_fraction_eq_l269_269871

theorem add_to_fraction_eq (n : ℤ) (h : (4 + n) / (7 + n) = 3 / 4) : n = 5 :=
by sorry

end add_to_fraction_eq_l269_269871


namespace number_of_valid_two_digit_numbers_l269_269057

def f (n : ℕ) : ℕ :=
  let a := n / 10 in
  let b := n % 10 in
  a + b + a * b

theorem number_of_valid_two_digit_numbers : (Finset.filter (λ n, f n = n) (Finset.range 100 \ Finset.range 10)).card = 9 := by
  sorry

end number_of_valid_two_digit_numbers_l269_269057


namespace ordered_pair_satisfies_equations_l269_269469

theorem ordered_pair_satisfies_equations :
  ∃ (x y : ℤ), 12 * x + 21 * y = 15 ∧ 21 * x + 12 * y = 51 ∧ x = 3 ∧ y = -1 :=
by {
  use 3, -1,
  simp,
}

end ordered_pair_satisfies_equations_l269_269469


namespace chord_length_of_line_on_curve_l269_269712

noncomputable def line_l_parametric (t : ℝ) : ℝ × ℝ := 
  (sqrt 2 / 2 + 1, t / 2)

noncomputable def curve_C_polar (θ : ℝ) : ℝ := 
  2 * sqrt 2 * sin (θ + π / 4)

noncomputable def chord_length_cut_by_line_on_curve : ℝ :=
  4 * sqrt 3 / 3

theorem chord_length_of_line_on_curve :
  let l := ∀ t : ℝ, line_l_parametric t
  let C := ∀ θ : ℝ, curve_C_polar θ
  true
    := chord_length_cut_by_line_on_curve = 4 * sqrt 3 / 3 := 
  sorry

end chord_length_of_line_on_curve_l269_269712


namespace wire_cut_problem_l269_269204

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end wire_cut_problem_l269_269204


namespace binomial_133_133_l269_269548

theorem binomial_133_133 : @Nat.choose 133 133 = 1 := by   
sorry

end binomial_133_133_l269_269548


namespace domain_of_f_eq_l269_269237

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 9) / (x^2 - 4)

-- Define the domain of the function f(x)
def domain_f : Set ℝ := { x | x ≠ 2 ∧ x ≠ -2 }

-- Express the correct domain in terms of intervals
def correct_domain : Set ℝ := { x | (x < -2) ∨ (x > -2 ∧ x < 2) ∨ (x > 2) }

-- Theorem to prove the equivalence of the domain
theorem domain_of_f_eq : domain_f = correct_domain :=
by sorry

end domain_of_f_eq_l269_269237


namespace fruit_basket_ratio_l269_269003

theorem fruit_basket_ratio (total_fruits : ℕ) (oranges : ℕ) (apples : ℕ) (h1 : total_fruits = 40) (h2 : oranges = 10) (h3 : apples = total_fruits - oranges) :
  (apples / oranges) = 3 := by
  sorry

end fruit_basket_ratio_l269_269003


namespace find_angle_B_sin_sum_range_l269_269021

-- Define the necessary variables and conditions
variables (A B C : ℝ) (a b c : ℝ)
variable (h1 : B ∈ (0 : ℝ), real.pi)
variable (h2 : 2 * real.sin A - real.sqrt 3 * real.cos C = real.sqrt 3 * real.sin C / real.tan B)

-- Prove that angle B is either π/3 or 2π/3
theorem find_angle_B (h1 : B ∈ (0, real.pi)) (h2 : 2 * real.sin A - real.sqrt 3 * real.cos C = real.sqrt 3 * real.sin C / real.tan B) :
  B = real.pi / 3 ∨ B = 2 * real.pi / 3 :=
sorry

-- Prove the range of sinA + sinB + sinC assuming B is acute
theorem sin_sum_range (h3 : B = real.pi / 3) :
  real.sqrt 3 < real.sin A + real.sin B + real.sin C ∧ real.sin A + real.sin B + real.sin C ≤ (3 * real.sqrt 3) / 2 :=
sorry

end find_angle_B_sin_sum_range_l269_269021


namespace obtuse_triangle_iff_l269_269670

theorem obtuse_triangle_iff (x : ℝ) :
    (x > 1 ∧ x < 3) ↔ (x + (x + 1) > (x + 2) ∧
                        (x + 1) + (x + 2) > x ∧
                        (x + 2) + x > (x + 1) ∧
                        (x + 2)^2 > x^2 + (x + 1)^2) :=
by
  sorry

end obtuse_triangle_iff_l269_269670


namespace largest_8_11_triple_l269_269991

def is_base8_rep (m n : ℕ) : Prop :=
  let digits := (nat.digits 8 m)
  let formed_base10 := digits.foldl (λ acc d, acc * 10 + d) 0
  formed_base10 = n

def is_8_11_triple (M : ℕ) : Prop :=
  let formed_base10 := (nat.digits 8 M).foldl (λ acc d, acc * 10 + d) 0
  formed_base10 = 3 * M

theorem largest_8_11_triple : ∃ (n : ℕ), is_8_11_triple n ∧ ∀ (m : ℕ), is_8_11_triple m → m ≤ n :=
  ⟨705, sorry, sorry⟩

end largest_8_11_triple_l269_269991


namespace no_integer_n_exists_l269_269095

theorem no_integer_n_exists (n : ℤ) : ¬(∃ n : ℤ, ∃ k : ℤ, ∃ m : ℤ, (n - 6) = 15 * k ∧ (n - 5) = 24 * m) :=
by
  sorry

end no_integer_n_exists_l269_269095


namespace inscribed_circles_radii_sum_le_radius_l269_269590

theorem inscribed_circles_radii_sum_le_radius (R : ℝ)
  (A B P C D : Point)
  (semicircle : semicircle_structure R A B)
  (on_diameter : P ∈ diameter A B)
  (angle_condition : ∠APD = 60 ∧ ∠DPC = 60 ∧ ∠CPB = 60): 
  ∃ (e f g : ℝ) (E F G : Point), 
  inscribed_circle P B C E e ∧ inscribed_circle P C D F f ∧ inscribed_circle P D A G g →
  e + f + g ≤ R :=
sorry

end inscribed_circles_radii_sum_le_radius_l269_269590


namespace solve_quadratic_l269_269676

theorem solve_quadratic (x : ℝ) (h1 : 2 * x ^ 2 = 9 * x - 4) (h2 : x ≠ 4) : 2 * x = 1 :=
by
  -- The proof will go here
  sorry

end solve_quadratic_l269_269676


namespace total_monsters_l269_269227

theorem total_monsters (a1 a2 a3 a4 a5 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 2 * a1) 
  (h3 : a3 = 2 * a2) 
  (h4 : a4 = 2 * a3) 
  (h5 : a5 = 2 * a4) : 
  a1 + a2 + a3 + a4 + a5 = 62 :=
by
  sorry

end total_monsters_l269_269227


namespace probability_of_7_successes_in_7_trials_l269_269897

open Probability

/-- Define the given conditions for the problem -/
def n : ℕ := 7
def k : ℕ := 7
def p : ℚ := 2 / 7

/-- The binomial coefficient and the probability of success in n trials -/
theorem probability_of_7_successes_in_7_trials :
  P(X = k) = (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) :=
by
  have bep_0 : nat.choose 7 7 = 1, from sorry,
  have p_power_k : p ^ k = (2 / 7) ^ 7, from sorry,
  have q_power_rem : (1 - p) ^ (n - k) = 1, from sorry,
  have p_eq_frac : (2 / 7) ^ 7 * 1 = 128 / 823543, from sorry,
  show 1 * (2 / 7) ^ 7 * 1 = 128 / 823543, by sorry

end probability_of_7_successes_in_7_trials_l269_269897


namespace probability_of_7_successes_l269_269888

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l269_269888


namespace sum_of_prime_factors_is_correct_l269_269868

-- Define what it means for a set of primes to be the prime factors of a number.
def is_prime_factor (n p : ℕ) : Prop :=
  p ≠ 1 ∧ p ≠ n ∧ p ∣ n

-- Define the number in question
def n : ℕ := 360

-- Define the three smallest prime factors of 360
def smallest_prime_factors (n : ℕ) : list ℕ :=
  if n = 360 then [2, 3, 5] else []

-- Define the sum of these three smallest prime factors
def sum_of_smallest_prime_factors (n : ℕ) : ℕ :=
  (smallest_prime_factors n).sum

-- State the theorem
theorem sum_of_prime_factors_is_correct :
  sum_of_smallest_prime_factors 360 = 10 :=
by sorry

end sum_of_prime_factors_is_correct_l269_269868


namespace min_value_expr_l269_269064

open Real

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 :=
by
  sorry

end min_value_expr_l269_269064


namespace circle_radius_in_unit_square_l269_269575

theorem circle_radius_in_unit_square :
  ∃ r : ℝ, (0 < r) ∧ (∀ (c : ℝ×ℝ), 
  (c = (0.5, 0.5) ∨ 
  ( (0 ≤ c.1 ∧ c.1 ≤ 1) ∧ (0 ≤ c.2 ∧ c.2 ≤ 1) ∧ 
    (c.1 - 0.5)^2 + (c.2 - 0.5)^2 = (2 * r)^2 ∧ 
    (c.1 = r ∨ c.1 = 1 - r) ∧ 
    (c.2 = r ∨ c.2 = 1 - r) ∧ 
    ( (c.1 = r ∧ c.2 ≠ r) ∨ (c.1 ≠ r ∧ c.2 = r) )
  )) 
  → r = √2 / (4 + 2 * √2) :=
begin
  sorry
end

end circle_radius_in_unit_square_l269_269575


namespace trapezoid_segment_parallel_l269_269133

variables {A B C D M P Q H K : Type}

/-- Given a trapezoid ABCD with AB parallel to CD, P and Q are the midpoints of BC and AD respectively,
    and M is a point on the extension of diagonal AC. H is the intersection of the line through M and P with BC,
    and K is the intersection of the line through M and Q with AD. Prove that segment HK is parallel to bases AB and CD. -/
theorem trapezoid_segment_parallel (trapezoid : Trapezoid A B C D) 
  (midpoint_P : Midpoint P B C) (midpoint_Q : Midpoint Q A D) 
  (extension_M : PointOnExtension M A C) 
  (intersection_H : IntersectionLinePoint H (LineThrough M P) B C)
  (intersection_K : IntersectionLinePoint K (LineThrough M Q) A D) : 
  ParallelSegment HK AD ∧ ParallelSegment HK BC := 
by
  sorry

end trapezoid_segment_parallel_l269_269133


namespace difference_of_squares_example_l269_269869

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 305) (h2 : b = 295) :
  (a^2 - b^2) / 10 = 600 :=
by
  sorry

end difference_of_squares_example_l269_269869


namespace work_done_approx_l269_269241

noncomputable def gravitational_work_done (m : ℝ) (H : ℝ) (R3 : ℝ) (g : ℝ) : ℝ :=
  let F (x : ℝ) := (m * g * R3^2) / (R3 + x)^2
  ∫ x in 0..H, F x

def m := 5.0 * 1000  -- 5 tons to kg
def H := 450000  -- 450 km to meters
def R3 := 6380000  -- 6380 km to meters
def g := 10  -- m/s^2

-- This would be the statement to show the work A is approximately 2.1017569546 × 10^{10} J
theorem work_done_approx : abs (gravitational_work_done m H R3 g - 2.1017569546 * 10^10) < 0.1 * 10^10 := 
sorry

end work_done_approx_l269_269241


namespace problem_1_problem_2_problem_3_l269_269854

-- Problem 1: Arrangement of students without A in the first position
theorem problem_1 :
  let students := ["A", "B", "C", "D", "E"] in
  let first_pos := "A" in
  ∀ (arr: List String), arr.permutations.length * 4 = 96 :=
sorry

-- Problem 2: Arrangement with A and B next to each other and C and D not next to each other
theorem problem_2 :
  let students := ["A", "B", "C", "D", "E"] in
  ∀ (arr: List String), 
    (arr.are_ab_next_to_each_other "A" "B" ∧ ¬arr.are_cd_next_to_each_other "C" "D") -> 
    arr.permutations.length = 24 :=
sorry

-- Problem 3: Competition participation arrangements
theorem problem_3 :
  let students := ["A", "B", "C", "D", "E"] in
  let competitions := ["Singing", "Dancing", "Chess", "Drawing"] in
  ∀ (assignment: String -> String), 
    (assignment "A" ≠ "Dancing" ∧ 
     all_students_participating students competitions assignment) ->
    arrangement_possibilities students competitions assignment.length = 180 :=
sorry

end problem_1_problem_2_problem_3_l269_269854


namespace smallest_sum_l269_269597

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l269_269597


namespace final_box_contents_l269_269174

theorem final_box_contents :
  ∀ (initial_black initial_white : ℕ),
  initial_black = 100 →
  initial_white = 100 →
  (∀ (black white : ℕ), 
    initial_black + initial_white - 3 * (50 - (white:ℕ)/2) = final_black + final_white →
    2 ∣ white →
    final_white = 2) :=
by 
  intro initial_black initial_white h_black h_white black white,
  sorry

end final_box_contents_l269_269174


namespace right_angled_triangle_from_perspective_l269_269846

theorem right_angled_triangle_from_perspective
  (ABC A'B'C': Triangle)
  (h1 : perspective_drawing ABC A'B'C')
  (h2 : parallel A'B' y'axis)
  (h3 : on_axis B'C' x'axis)
  (h4 : angle x'o'y' = 45°) :
  is_right_angled_triangle ABC :=
sorry

end right_angled_triangle_from_perspective_l269_269846


namespace maxwell_meets_brad_l269_269378

variable (t : ℝ) -- time in hours
variable (distance_between_homes : ℝ) -- total distance
variable (maxwell_speed : ℝ) -- Maxwell's walking speed
variable (brad_speed : ℝ) -- Brad's running speed
variable (brad_delay : ℝ) -- Brad's start time delay

theorem maxwell_meets_brad 
  (hb: brad_delay = 1)
  (d: distance_between_homes = 34)
  (v_m: maxwell_speed = 4)
  (v_b: brad_speed = 6)
  (h : 4 * t + 6 * (t - 1) = distance_between_homes) :
  t = 4 := 
  sorry

end maxwell_meets_brad_l269_269378


namespace fixed_points_1_and_3_range_of_a_for_two_fixed_points_minimum_value_of_b_l269_269235

-- Define fixed point property
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- 1. Prove fixed points of f(x) for given a and b
theorem fixed_points_1_and_3 (x : ℝ) : 
    (let f := λ x, x^2 + 4 * x + 2 in
     is_fixed_point f (-1) ∧ is_fixed_point f (-2)) :=
by
  let f := λ x, x^2 + 4 * x + 2
  simp [is_fixed_point, f] -- Assertion that needs to be proved
  sorry

-- 2. Prove range of a for the function to always have two fixed points for any b
theorem range_of_a_for_two_fixed_points (a : ℝ) : 
    ∀ b : ℝ, 
    (0 < a ∧ a < 1) → 
    (let f := λ x, a * x^2 + (b + 1) * x + b - 1 in
     ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point f x1 ∧ is_fixed_point f x2) :=
by
  intro b
  assume h
  let f := λ x, a * x^2 + (b + 1) * x + b - 1
  simp [is_fixed_point, f] -- Assertion that needs to be proved
  sorry

-- 3. Prove the minimum value of b under the constraints
theorem minimum_value_of_b (a : ℝ)
    (h1 : 0 < a ∧ a < 1) 
    (x1 x2 : ℝ) (h2 : x1 ≠ x2) 
    (h3 : let f := λ x, a * x^2 + (b + 1) * x + b - 1 in 
          is_fixed_point f x1 ∧ is_fixed_point f x2)
    (h4 : let g := λ x, -x + (2 * a) / (5 * a^2 - 4 * a + 1) in 
          g ((x1 + x2) / 2) = (x1 + x2) / 2) : 
    b = -2 :=
by
  have h_b_min : b = -2 := 
    by calc b = (2 * a^2) / (- (4 * a) + 4 * a) : by sorry -- Need full proof
  exact h_b_min

end fixed_points_1_and_3_range_of_a_for_two_fixed_points_minimum_value_of_b_l269_269235


namespace add_to_fraction_eq_l269_269870

theorem add_to_fraction_eq (n : ℤ) (h : (4 + n) / (7 + n) = 3 / 4) : n = 5 :=
by sorry

end add_to_fraction_eq_l269_269870


namespace smallest_sum_l269_269599

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l269_269599


namespace largest_frog_weight_difference_l269_269431

def frog_weight_difference (largest_frog_weight : ℕ) (weight_ratio : ℕ) : ℕ :=
  let smallest_frog_weight := largest_frog_weight / weight_ratio
  largest_frog_weight - smallest_frog_weight

theorem largest_frog_weight_difference :
  frog_weight_difference 120 10 = 108 :=
begin
  -- Definitions and conditions have been set.
  -- Proof is not required as per the instructions.
  sorry
end

end largest_frog_weight_difference_l269_269431


namespace quadratic_has_two_distinct_real_roots_l269_269444

-- Define the quadratic equation and its coefficients
def a := 1
def b := -4
def c := -3

-- Define the discriminant function for a quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- State the problem in Lean: Prove that the quadratic equation x^2 - 4x - 3 = 0 has a positive discriminant.
theorem quadratic_has_two_distinct_real_roots : discriminant a b c > 0 :=
by
  sorry -- This is where the proof would go

end quadratic_has_two_distinct_real_roots_l269_269444


namespace not_possible_placement_l269_269346
open Nat

def adjacent (E : List (ℕ × ℕ)) (u v : ℕ) : Prop := 
  (u, v) ∈ E ∨ (v, u) ∈ E

def non_adjacent (E : List (ℕ × ℕ)) (u v : ℕ) : Prop := 
  ¬ adjacent E u v

theorem not_possible_placement (V : Fin₈ → ℕ) (E : List (ℕ × ℕ)) 
  (hV : ∀ i, 1 ≤ V i ∧ V i ≤ 220) 
  (h_adjacent : ∀ u v, adjacent E u v → gcd (V u) (V v) > 1) 
  (h_non_adjacent : ∀ u v, non_adjacent E u v → gcd (V u) (V v) = 1) : 
  False :=
sorry

end not_possible_placement_l269_269346


namespace total_drawing_sheets_l269_269351

-- Definitions based on the conditions given
def brown_sheets := 28
def yellow_sheets := 27

-- The statement we need to prove
theorem total_drawing_sheets : brown_sheets + yellow_sheets = 55 := by
  sorry

end total_drawing_sheets_l269_269351


namespace value_of_a_plus_b_l269_269364

variables (a b : ℝ)
def M := {-1, b / a, 1}
def N := {a, b, b - a}
def f (x : ℝ) := x
theorem value_of_a_plus_b : a + b = 1 ∨ a + b = -1 :=
sorry

end value_of_a_plus_b_l269_269364


namespace calculation_A_B_l269_269844

theorem calculation_A_B :
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  A - B = 4397 :=
by
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  sorry

end calculation_A_B_l269_269844


namespace diagonal_of_rectangle_l269_269386

theorem diagonal_of_rectangle (a b d : ℝ)
  (h_side : a = 15)
  (h_area : a * b = 120)
  (h_diag : a^2 + b^2 = d^2) :
  d = 17 :=
by
  sorry

end diagonal_of_rectangle_l269_269386


namespace cost_per_handle_is_60_cents_l269_269077

def fixed_cost : ℝ := 7640
def selling_price_per_handle : ℝ := 4.60
def number_of_handles : ℝ := 1910
def break_even_condition (C : ℝ) : Prop :=
  (number_of_handles * selling_price_per_handle) = (fixed_cost + number_of_handles * C)

theorem cost_per_handle_is_60_cents : ∃ C : ℝ, break_even_condition C ∧ C = 0.60 :=
by
  use 0.60
  split
  · unfold break_even_condition
    sorry
  · sorry

end cost_per_handle_is_60_cents_l269_269077


namespace find_third_discount_percentage_l269_269927

noncomputable def third_discount_percentage (x : ℝ) : Prop :=
  let item_price := 68
  let num_items := 3
  let first_discount := 0.15
  let second_discount := 0.10
  let total_initial_price := num_items * item_price
  let price_after_first_discount := total_initial_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount * (1 - x / 100) = 105.32

theorem find_third_discount_percentage : ∃ x : ℝ, third_discount_percentage x ∧ x = 32.5 :=
by
  sorry

end find_third_discount_percentage_l269_269927


namespace p_at_5_l269_269181

noncomputable def p (x : ℝ) : ℝ :=
  sorry

def p_cond (n : ℝ) : Prop :=
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → p n = 1 / n^3

theorem p_at_5 : (∀ n, p_cond n) → p 5 = -149 / 1500 :=
by
  intros
  sorry

end p_at_5_l269_269181


namespace angle_XYZ_of_excircle_circumcircle_incircle_l269_269464

theorem angle_XYZ_of_excircle_circumcircle_incircle 
  (a b c x y z : ℝ) 
  (hA : a = 50)
  (hB : b = 70)
  (hC : c = 60) 
  (triangleABC : a + b + c = 180) 
  (excircle_Omega : Prop) 
  (incircle_Gamma : Prop) 
  (circumcircle_Omega_triangleXYZ : Prop) 
  (X_on_BC : Prop)
  (Y_on_AB : Prop) 
  (Z_on_CA : Prop): 
  x = 115 := 
by 
  sorry

end angle_XYZ_of_excircle_circumcircle_incircle_l269_269464


namespace verify_trip_cost_and_remaining_money_l269_269037

/-- Joe's conditions for the trip and expenses --/
def initial_savings_usd := 6000
def exchange_rate := 1.35
def flight_usd := 1200
def hotel_usd := 800
def food_usd := 3000
def local_transportation_usd := 500
def entertainment_usd := 850
def miscellaneous_usd := 350

/-- Conversion of savings to AUD --/
def savings_aud := initial_savings_usd * exchange_rate

/-- Conversion of each expense to AUD --/
def flight_aud := flight_usd * exchange_rate
def hotel_aud := hotel_usd * exchange_rate
def food_aud := food_usd * exchange_rate
def local_transportation_aud := local_transportation_usd * exchange_rate
def entertainment_aud := entertainment_usd * exchange_rate
def miscellaneous_aud := miscellaneous_usd * exchange_rate

/-- Total cost of the trip in AUD --/
def total_trip_cost_aud := 
  flight_aud + 
  hotel_aud + 
  food_aud + 
  local_transportation_aud + 
  entertainment_aud + 
  miscellaneous_aud

/-- Remaining money in AUD after the trip --/
def remaining_money_aud := savings_aud - total_trip_cost_aud

/-- Proof statement: Total trip cost and remaining money after trip --/
theorem verify_trip_cost_and_remaining_money :
  total_trip_cost_aud = 9045 ∧ remaining_money_aud = -945 := by
  sorry


end verify_trip_cost_and_remaining_money_l269_269037


namespace quadrilateral_area_l269_269011

theorem quadrilateral_area {ABCQ : ℝ} 
  (side_length : ℝ) 
  (D P E N : ℝ → Prop) 
  (midpoints : ℝ) 
  (W X Y Z : ℝ → Prop) :
  side_length = 4 → 
  (∀ a b : ℝ, D a ∧ P b → a = 1 ∧ b = 1) → 
  (∀ c d : ℝ, E c ∧ N d → c = 1 ∧ d = 1) →
  (∀ w x y z : ℝ, W w ∧ X x ∧ Y y ∧ Z z → w = 0.5 ∧ x = 0.5 ∧ y = 0.5 ∧ z = 0.5) →
  ∃ (area : ℝ), area = 0.25 :=
by
  sorry

end quadrilateral_area_l269_269011


namespace taxi_trip_miles_l269_269035

theorem taxi_trip_miles 
  (initial_fee : ℝ := 2.35)
  (additional_charge : ℝ := 0.35)
  (segment_length : ℝ := 2/5)
  (total_charge : ℝ := 5.50) :
  ∃ (miles : ℝ), total_charge = initial_fee + additional_charge * (miles / segment_length) ∧ miles = 3.6 :=
by
  sorry

end taxi_trip_miles_l269_269035


namespace ten_digit_number_with_repeated_operations_contains_duplicate_digits_l269_269437

theorem ten_digit_number_with_repeated_operations_contains_duplicate_digits :
  ∃ x y, x ≠ y ∧ x ∈ digits (transform_to_ten_digits (2 ^ 1970)) ∧ y ∈ digits (transform_to_ten_digits (2 ^ 1970)) :=
  sorry

/--
Variable explaining the number transform steps to ten-digit.
--/
def transform_to_ten_digits (n : ℕ) : ℕ :=
  sorry

/--
Function to get the list of digits of a number.
--/
def digits (n : ℕ) : list ℕ :=
  sorry

end ten_digit_number_with_repeated_operations_contains_duplicate_digits_l269_269437


namespace quadrilateral_inequality_l269_269735

theorem quadrilateral_inequality (A B C D : Point) :
  (dist A C) * (dist B D) ≤ (dist A B) * (dist C D) + (dist A D) * (dist B C) ∧
  ((dist A C * dist B D = dist A B * dist C D + dist A D * dist B C)
  ↔ convex_quadrilateral A B C D ∧ cyclic_quadrilateral A B C D) :=
sorry

end quadrilateral_inequality_l269_269735


namespace volume_of_second_cube_l269_269475

noncomputable def cube_volume (s : ℝ) := s^3
noncomputable def cube_surface_area (s : ℝ) := 6 * s^2

theorem volume_of_second_cube :
  let v₁ := 8,
      s₁ := real.cbrt v₁,
      a₁ := cube_surface_area s₁,
      a₂ := 3 * a₁,
      s₂ := real.sqrt (a₂ / 6),
      v₂ := cube_volume s₂
  in v₂ = 24 * real.sqrt 3 := 
by {
  let v₁ := 8,
  let s₁ := real.cbrt v₁,
  let a₁ := cube_surface_area s₁,
  let a₂ := 3 * a₁,
  let s₂ := real.sqrt (a₂ / 6),
  let v₂ := cube_volume s₂,
  have h1 : v₂ = 24 * real.sqrt 3 := sorry,
  exact h1
}

end volume_of_second_cube_l269_269475


namespace find_angle_B_l269_269000

variables (a b c : ℝ) (A B C : ℝ)
variables [fact (0 < B)] [fact (B < real.pi)]

def cos_ratio_condition := (real.cos B) / (real.cos C) = -b / (2 * a + c)

theorem find_angle_B (h : cos_ratio_condition a b c A B C) : B = 2 * real.pi / 3 :=
sorry

end find_angle_B_l269_269000


namespace proof_1_proof_2_l269_269082

-- Definitions of propositions p, q, and r

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (x^2 + (a - 1) * x + a^2 ≤ 0)

def q (a : ℝ) : Prop :=
  2 * a^2 - a > 1

def r (a : ℝ) : Prop :=
  (2 * a - 1) / (a - 2) ≤ 1

-- The given proof problem statement 1
theorem proof_1 (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → (a ∈ Set.Icc (-1) (-1/2) ∪ Set.Ioo (1/3) 1) :=
sorry

-- The given proof problem statement 2
theorem proof_2 (a : ℝ) : ¬ p a → r a :=
sorry

end proof_1_proof_2_l269_269082


namespace perpendicular_vectors_parallel_vectors_l269_269650

-- Let a and b be vectors defined as follows
def vec_a : ℝ × ℝ × ℝ := (2, -1, 2)
def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

-- Prove that if a is perpendicular to b, then x = 5
theorem perpendicular_vectors (x : ℝ) :
  (let (ax, ay, az) := vec_a in let (bx, by, bz) := vec_b(x) in 
   ax * bx + ay * by + az * bz = 0) → x = 5 := by
  sorry

-- Prove that if a is parallel to b, then x = -4
theorem parallel_vectors (x : ℝ) :
  (let (ax, ay, az) := vec_a in let (bx, by, bz) := vec_b(x) in 
   (ax = 0 ∨ bx = 0 ∨ ax * by = ay * bx) ∧ (ax = 0 ∨ az = 0 ∨ ax * bz = az * bx)) → x = -4 := by
  sorry

end perpendicular_vectors_parallel_vectors_l269_269650


namespace phi_abs_val_correct_l269_269643

noncomputable def abs_phi_proof : Prop :=
∀ (φ : ℝ), (-π < φ ∧ φ < 0) ∧
   (∃ k : ℤ, 3*sin(2*(x + π/6) + φ) = 3*sin(-2*x + k*π)) →
   |φ| = 5*π/6

theorem phi_abs_val_correct : abs_phi_proof := by
  sorry

end phi_abs_val_correct_l269_269643


namespace heracles_age_l269_269973

theorem heracles_age
  (H : ℕ)
  (audrey_current_age : ℕ)
  (audrey_in_3_years : ℕ)
  (h1 : audrey_current_age = H + 7)
  (h2 : audrey_in_3_years = audrey_current_age + 3)
  (h3 : audrey_in_3_years = 2 * H)
  : H = 10 :=
by
  sorry

end heracles_age_l269_269973


namespace second_trial_point_l269_269298

theorem second_trial_point (a b: ℕ) (h_range: a = 10) (h_range_end: b = 90) (h_rounds: ℕ = 4) :
  let interval_length := b - a
  let parts := 8
  let part_length := interval_length / parts
  let trial_points := [a, a + part_length, a + 2 * part_length, a + 3 * part_length,
                      b - 3 * part_length, b - 2 * part_length, b - part_length, b]
  (40 ∈ trial_points ∨ 60 ∈ trial_points) :=
by
  sorry

end second_trial_point_l269_269298


namespace first_place_friend_distance_friend_running_distance_l269_269380

theorem first_place_friend_distance (distance_mina_finish : ℕ) (halfway_condition : ∀ x, x = distance_mina_finish / 2) :
  (∃ y, y = distance_mina_finish / 2) :=
by
  sorry

-- Given conditions
def distance_mina_finish : ℕ := 200
noncomputable def first_place_friend_position := distance_mina_finish / 2

-- The theorem we need to prove
theorem friend_running_distance : first_place_friend_position = 100 :=
by
  sorry

end first_place_friend_distance_friend_running_distance_l269_269380


namespace smallest_k_no_real_roots_l269_269473

theorem smallest_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 13 ≠ 0) ∧
  (∀ n : ℤ, n < k → ∃ x : ℝ, 3 * x * (n * x - 5) - 2 * x^2 + 13 = 0) :=
by sorry

end smallest_k_no_real_roots_l269_269473


namespace correlation_identification_l269_269839

noncomputable def relationship (a b : Type) : Prop := 
  ∃ (f : a → b), true

def correlation (a b : Type) : Prop :=
  relationship a b ∧ relationship b a

def deterministic (a b : Type) : Prop :=
  ∀ x y : a, ∃! z : b, true

def age_wealth : Prop := correlation ℕ ℝ
def point_curve_coordinates : Prop := deterministic (ℝ × ℝ) (ℝ × ℝ)
def apple_production_climate : Prop := correlation ℝ ℝ
def tree_diameter_height : Prop := correlation ℝ ℝ

theorem correlation_identification :
  age_wealth ∧ apple_production_climate ∧ tree_diameter_height ∧ ¬point_curve_coordinates := 
by
  -- proof of these properties
  sorry

end correlation_identification_l269_269839


namespace max_ab_l269_269625

theorem max_ab (a b : ℝ) (h : a + b = 1) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_l269_269625


namespace more_red_balls_l269_269724

theorem more_red_balls (red_packs yellow_packs pack_size : ℕ) (h1 : red_packs = 5) (h2 : yellow_packs = 4) (h3 : pack_size = 18) :
  (red_packs * pack_size) - (yellow_packs * pack_size) = 18 :=
by
  sorry

end more_red_balls_l269_269724


namespace pyramid_cube_volume_l269_269944

/-- A pyramid has a square base with side length 2 units, the lateral faces are equilateral triangles,
    and a cube is positioned inside such that its bottom face covers the base of the pyramid entirely 
    and the top face touches the summit of the pyramid. Prove that the volume of the cube is 8. -/
theorem pyramid_cube_volume
  (s_a : ℝ) (s_b : ℝ) (side_length : s_a = 2) (eq_tri : s_b = 2 * real.sqrt 2)
  (height_eq : (s_b * (real.sqrt 3 / 2)) = (real.sqrt 6)) 
  (cube_side : s_a = 2) : (s_a ^ 3 = 8) :=
begin
  sorry
end

end pyramid_cube_volume_l269_269944


namespace complex_conjugate_l269_269771

theorem complex_conjugate (z : ℂ) : 
  (i : ℂ)² = -1 → (i : ℂ)⁵ = i → 
  z = (2 + i) / (1 + (-1) + i) → 
  conj z = 1 + 2 * i :=
by
  intros
  sorry

end complex_conjugate_l269_269771


namespace rhombus_side_length_l269_269511

-- Define the conditions including the diagonals and area of the rhombus
def diagonal_ratio (d1 d2 : ℝ) : Prop := d1 = 3 * d2
def area_rhombus (b : ℝ) (K : ℝ) : Prop := K = (1 / 2) * b * (3 * b)

-- Define the side length of the rhombus in terms of K
noncomputable def side_length (K : ℝ) : ℝ := Real.sqrt (5 * K / 3)

-- The main theorem statement
theorem rhombus_side_length (K : ℝ) (b : ℝ) (h1 : diagonal_ratio (3 * b) b) (h2 : area_rhombus b K) : 
  side_length K = Real.sqrt (5 * K / 3) := 
sorry

end rhombus_side_length_l269_269511


namespace find_interest_rate_l269_269829

theorem find_interest_rate
  (P : ℝ) (A : ℝ) (n t : ℕ) (hP : P = 3000) (hA : A = 3307.5) (hn : n = 2) (ht : t = 1) :
  ∃ r : ℝ, r = 10 :=
by
  sorry

end find_interest_rate_l269_269829


namespace john_income_increase_percent_l269_269038

theorem john_income_increase_percent :
  ∀ (initial_job_income : ℕ) (initial_freelance_income : ℕ)
    (new_job_income : ℕ) (new_freelance_income : ℕ) (weeks_in_month : ℕ),
    initial_job_income = 60 →
    initial_freelance_income = 40 →
    new_job_income = 120 →
    new_freelance_income = 60 →
    weeks_in_month = 4 →
    let initial_monthly_income := (initial_job_income + initial_freelance_income) * weeks_in_month,
        new_monthly_income := (new_job_income + new_freelance_income) * weeks_in_month,
        percentage_increase := ((new_monthly_income - initial_monthly_income) * 100) / initial_monthly_income
    in
      percentage_increase = 80 := by
  intros initial_job_income initial_freelance_income new_job_income new_freelance_income weeks_in_month
  intros h1 h2 h3 h4 h5
  let initial_monthly_income := (initial_job_income + initial_freelance_income) * weeks_in_month
  let new_monthly_income := (new_job_income + new_freelance_income) * weeks_in_month
  let percentage_increase := ((new_monthly_income - initial_monthly_income) * 100) / initial_monthly_income
  have : percentage_increase = 80 := by sorry
  exact this

end john_income_increase_percent_l269_269038


namespace trigonometric_values_of_angle_B_l269_269695

noncomputable def AC (AB CB : ℝ) : ℝ := Real.sqrt (AB^2 - CB^2)

theorem trigonometric_values_of_angle_B
  (AB : ℝ) (CB : ℝ)
  (h_AB_pos : AB > 0)
  (h_CB_pos : CB > 0)
  (h_right_triangle : (CB^2 + AC AB CB^2) = AB^2) :
  let B := Real.arcsin (AC AB CB / AB) in
  Real.sin B = 12 / 13 ∧ Real.cos B = 5 / 13 ∧ Real.tan B = 12 / 5 :=
by
  sorry

end trigonometric_values_of_angle_B_l269_269695


namespace number_of_valid_pairs_proof_l269_269639

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x) / (abs x + 1)

-- Define M and N
def M (a b : ℝ) : set ℝ := {x | a <= x ∧ x <= b}
def N (a b : ℝ) : set ℝ := {y | ∃ x, (a <= x ∧ x <= b ∧ y = f x)}

-- Definition of the question
def number_of_valid_pairs : ℝ :=
  if M = N then 3 else 0

theorem number_of_valid_pairs_proof :
  ∃ a b : ℝ, a < b ∧ M a b = N a b → number_of_valid_pairs = 3 :=
sorry

end number_of_valid_pairs_proof_l269_269639


namespace smallest_sum_l269_269600

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l269_269600


namespace necessary_and_sufficient_condition_l269_269669

theorem necessary_and_sufficient_condition (p q : Prop) 
  (hpq : p → q) (hqp : q → p) : 
  (p ↔ q) :=
by 
  sorry

end necessary_and_sufficient_condition_l269_269669


namespace ratio_of_S_to_R_l269_269963

noncomputable def find_ratio (total_amount : ℕ) (diff_SP : ℕ) (n : ℕ) (k : ℕ) (P : ℕ) (Q : ℕ) (R : ℕ) (S : ℕ) (ratio_SR : ℕ) :=
  Q = n ∧ R = n ∧ P = k * n ∧ S = ratio_SR * n ∧ P + Q + R + S = total_amount ∧ S - P = diff_SP

theorem ratio_of_S_to_R :
  ∃ n k ratio_SR, k = 2 ∧ ratio_SR = 4 ∧ 
  find_ratio 1000 250 n k 250 125 125 500 ratio_SR :=
by
  sorry

end ratio_of_S_to_R_l269_269963


namespace Jill_ball_difference_l269_269722

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end Jill_ball_difference_l269_269722


namespace twelfth_prime_is_thirty_seven_l269_269454

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_prime (n : ℕ) : ℕ :=
  (list.filter is_prime (list.range (2 * n))) !! (n - 1)

theorem twelfth_prime_is_thirty_seven :
  nth_prime 6 = 13 → nth_prime 12 = 37 :=
by
  intros h,
  have h₁ : nth_prime 6 = 13 := h,
  -- Continue the proof steps
  sorry

end twelfth_prime_is_thirty_seven_l269_269454


namespace smallest_x_plus_y_l269_269617

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269617


namespace nitrogen_mass_percentage_in_ammonium_iodide_l269_269572

def atomic_mass_N : ℝ := 14.01
def atomic_mass_H : ℝ := 1.01
def atomic_mass_I : ℝ := 126.90

def molar_mass_NH4I : ℝ := 1 * atomic_mass_N + 4 * atomic_mass_H + 1 * atomic_mass_I

def mass_percentage_N (molar_mass_NH4I atomic_mass_N : ℝ) : ℝ := 
  (atomic_mass_N / molar_mass_NH4I) * 100

theorem nitrogen_mass_percentage_in_ammonium_iodide :
  mass_percentage_N molar_mass_NH4I atomic_mass_N ≈ 9.67 := sorry

end nitrogen_mass_percentage_in_ammonium_iodide_l269_269572


namespace smallest_k_for_ten_ruble_heads_up_l269_269078

-- Conditions
def num_total_coins : ℕ := 30
def num_ten_ruble_coins : ℕ := 23
def num_five_ruble_coins : ℕ := 7
def num_heads_up : ℕ := 20
def num_tails_up : ℕ := 10

-- Prove the smallest k such that any k coins chosen include at least one ten-ruble coin heads-up.
theorem smallest_k_for_ten_ruble_heads_up (k : ℕ) :
  (∀ (coins : Finset ℕ), coins.card = k → (∃ (coin : ℕ) (h : coin ∈ coins), coin < num_ten_ruble_coins ∧ coin < num_heads_up)) →
  k = 18 :=
sorry

end smallest_k_for_ten_ruble_heads_up_l269_269078


namespace necklaces_count_l269_269353

theorem necklaces_count (spools : ℕ) (length_per_spool : ℕ) (length_per_necklace : ℕ) :
  spools = 3 → length_per_spool = 20 → length_per_necklace = 4 → spools * length_per_spool / length_per_necklace = 15 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end necklaces_count_l269_269353


namespace intersection_lines_of_planes_l269_269679

-- Define what it means for planes to intersect.
def intersects (plane1 plane2 : Type) : Prop :=
  ∃ L : Type, intersecting_line plane1 plane2 L

-- Define a theorem for the given problem.
theorem intersection_lines_of_planes 
  (α β γ : Type) 
  (H1 : intersects α β)
  (H2 : intersects α γ) : 
  exists (n : ℕ), n = 1 ∨ n = 2 ∨ n = 3 :=
sorry

end intersection_lines_of_planes_l269_269679


namespace problem_statement_l269_269834

variables {R : Type*} [LinearOrderedField R]

theorem problem_statement (a b c : R) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : (b - a) ^ 2 - 4 * (b - c) * (c - a) = 0) : (b - c) / (c - a) = -1 :=
sorry

end problem_statement_l269_269834


namespace smallest_possible_sum_l269_269604

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l269_269604


namespace pump_X_time_l269_269882

-- Definitions for the problem conditions.
variables (W : ℝ) (T_x : ℝ) (R_x R_y : ℝ)

-- Condition 1: Rate of pump X
def pump_X_rate := R_x = (W / 2) / T_x

-- Condition 2: Rate of pump Y
def pump_Y_rate := R_y = W / 18

-- Condition 3: Combined rate when both pumps work together for 3 hours to pump the remaining water
def combined_rate := (R_x + R_y) = (W / 2) / 3

-- The statement to prove
theorem pump_X_time : 
  pump_X_rate W T_x R_x →
  pump_Y_rate W R_y →
  combined_rate W R_x R_y →
  T_x = 9 :=
sorry

end pump_X_time_l269_269882


namespace conjugate_z_is_1_add_2i_l269_269755

open Complex

def z_def : ℂ := ((2 : ℂ) + I) / (1 + I^2 + I^5)

theorem conjugate_z_is_1_add_2i : conj z_def = (1 : ℂ) + 2 * I := 
sorry

end conjugate_z_is_1_add_2i_l269_269755


namespace average_X_Y_Z_l269_269166

open Set

-- Definitions
def X : Set Person := sorry
def Y : Set Person := sorry
def Z : Set Person := sorry

-- Conditions
axiom disjoint_X_Y : Disjoint X Y
axiom disjoint_X_Z : Disjoint X Z
axiom disjoint_Y_Z : Disjoint Y Z

axiom average_X_Y (x y : ℕ) : (37 * x + 23 * y) / (x + y) = 29
axiom average_X_Z (x z : ℕ) : (37 * x + 41 * z) / (x + z) = 39.5
axiom average_Y_Z (y z : ℕ) : (23 * y + 41 * z) / (y + z) = 33

-- Theorem stating the average age of the union of X, Y, Z
theorem average_X_Y_Z (x y z : ℕ) : 
  (37 * x + 23 * y + 41 * z) / (x + y + z) = 34 := by
  sorry

end average_X_Y_Z_l269_269166


namespace number_of_distributions_l269_269008

/--
In a room, 4 people each write a greeting card. They then collect them together, and
each person picks a greeting card written by someone else. Prove that the number of
different ways the 4 greeting cards can be distributed is 9.
-/
theorem number_of_distributions : 
  let n := 4 in
  ∃ (count_ways : ℕ), 
    (∀ (people : Fin n → ℕ), 
      (∀ i, people i ≠ i) → 
      ∃ ways, ∀ (i : Fin n), ways i ≠ people i) ∧ count_ways = 9 :=
begin
  let n := 4,
  sorry
end

end number_of_distributions_l269_269008


namespace convert_18_36_l269_269555

-- Define conversion factors
def degrees_to_minutes (deg: ℝ) : ℝ := deg * 60
def minutes_to_seconds (min: ℝ) : ℝ := min * 60

-- Define the main conversion function
def convert_decimal_degrees (deg: ℝ) : (ℕ × ℕ × ℕ) := 
  let d := deg.to_int
  let frac_deg := deg - d
  let min := degrees_to_minutes frac_deg
  let m := min.to_int
  let sec := minutes_to_seconds (min - m)
  (d, m, sec.to_int)

-- Define the given decimal degree value
def input_decimal_degree : ℝ := 18.36

-- Define the expected result
def expected_result : (ℕ × ℕ × ℕ) := (18, 21, 36)

-- State the theorem
theorem convert_18_36 : convert_decimal_degrees input_decimal_degree = expected_result :=
by
  sorry

end convert_18_36_l269_269555


namespace billy_trays_l269_269538

def trays_needed (total_ice_cubes : ℕ) (ice_cubes_per_tray : ℕ) : ℕ :=
  total_ice_cubes / ice_cubes_per_tray

theorem billy_trays (total_ice_cubes ice_cubes_per_tray : ℕ) (h1 : total_ice_cubes = 72) (h2 : ice_cubes_per_tray = 9) :
  trays_needed total_ice_cubes ice_cubes_per_tray = 8 :=
by
  sorry

end billy_trays_l269_269538


namespace coffee_shop_lattes_l269_269425

theorem coffee_shop_lattes (x : ℕ) (number_of_teas number_of_lattes : ℕ)
  (h1 : number_of_teas = 6)
  (h2 : number_of_lattes = 32)
  (h3 : number_of_lattes = x * number_of_teas + 8) :
  x = 4 :=
by
  sorry

end coffee_shop_lattes_l269_269425


namespace article_cost_price_l269_269940

theorem article_cost_price (SP : ℝ) (CP : ℝ) (h1 : SP = 455) (h2 : SP = CP + 0.3 * CP) : CP = 350 :=
by sorry

end article_cost_price_l269_269940


namespace number_of_digits_of_Q_l269_269738

theorem number_of_digits_of_Q :
  let n1 := 12345678987654321
  let n2 := 23456789123
  let n3 := 11
  Q = nat.digits 10 (n1 * n2 * n3) in
  Q.length = 30 :=
by
  sorry

end number_of_digits_of_Q_l269_269738


namespace Heracles_age_l269_269979

variable (A H : ℕ)

theorem Heracles_age :
  (A = H + 7) →
  (A + 3 = 2 * H) →
  H = 10 :=
by
  sorry

end Heracles_age_l269_269979


namespace code_decryption_probability_l269_269132

theorem code_decryption_probability :
  let P_A := (1:ℝ) / 2
  let P_B := (1:ℝ) / 3
  let P_C := (1:ℝ) / 3
  let P_not_A := 1 - P_A
  let P_not_B := 1 - P_B
  let P_not_C := 1 - P_C
  let P_not_all := P_not_A * P_not_B * P_not_C
  let P_code_decrypted := 1 - P_not_all
  P_code_decrypted = (7 / 9: ℝ) :=
begin
  sorry
end

end code_decryption_probability_l269_269132


namespace more_red_than_yellow_l269_269728

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end more_red_than_yellow_l269_269728


namespace circle_intersection_slope_trajectory_l269_269595

/-- Given a point P(1, 1), a moving line l passes through the point P and intersects a circle C at points A and B.
    (1) If |AB| = sqrt(17), find the slope angle of line l.
    (2) Find the equation of the trajectory of the midpoint M of segment AB. -/
theorem circle_intersection_slope_trajectory :
  let C := { p : ℝ × ℝ | (p.1)^2 + (p.2 - 1)^2 = 5 } in
  let P := (1, 1) in
  ∀ A B : ℝ × ℝ,
  A ∈ C → B ∈ C →
  A ≠ P → B ≠ P →
  (∃ l : ℝ → ℝ, ∀ x, l x = (A.2 - B.2) / (A.1 - B.1) * (x - P.1) + P.2) →
  dist A B = sqrt 17 →
  ∃ θ, θ = π / 3 ∨ θ = 2 * π / 3 ∧
  (∀ M, M = ((A.1 + B.1)/2, (A.2 + B.2)/2) → ∃ Q, Q = (1 / 2 : ℝ, 1) ∧ (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 1 / 4) :=
by sorry

end circle_intersection_slope_trajectory_l269_269595


namespace gathering_people_total_l269_269215

theorem gathering_people_total (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 :=
by
  sorry

end gathering_people_total_l269_269215


namespace smallest_x_plus_y_l269_269612

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269612


namespace equal_angles_of_midpoints_and_extension_l269_269711

theorem equal_angles_of_midpoints_and_extension (ABCD : Type) [rectangle ABCD] 
  (A B C D M N P Q : ABCD) 
  (hM : midpoint M A D) 
  (hN : midpoint N B C) 
  (hP : extension P D C)
  (hQ : intersection Q P M A C) : 
  ∠ Q N M = ∠ M N P := 
sorry

end equal_angles_of_midpoints_and_extension_l269_269711


namespace smallest_x_plus_y_l269_269609

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269609


namespace trapezoid_area_ratio_sum_l269_269131

/-- Given triangles DAO, AOB, and OBC congruent with respective sides:
  AD = AO = OB = BC = 10, and AB = DO = OC = 12,
  Point P on AB such that OP ⊥ AB, 
  Point X is the midpoint of AD, 
  Point Y is the midpoint of BC, 
  Prove that the sum of ratio terms of areas of trapezoids ABYX to XYCD, p + q, equals 12. -/
theorem trapezoid_area_ratio_sum :
  (∃ (AD AO OB BC AB DO OC : ℝ), AD = 10 ∧ AO = 10 ∧ OB = 10 ∧ BC = 10 ∧ AB = 12 ∧ DO = 12 ∧ OC = 12) →
  ∃ (OP XY : ℝ), OP = 8 ∧ XY = 18) →
  let height_ABYX : ℝ := 4 in
  let area_ABYX : ℝ := 1/2 * height_ABYX * (12 + 18) in
  let height_XYCD : ℝ := 4 in
  let area_XYCD : ℝ := 1/2 * height_XYCD * (18 + 24) in
  let ratio := area_ABYX / area_XYCD in
  let (p, q) := num_denom ratio in -- Ensure p and q are coprime
  p + q = 12 :=
by
  sorry

end trapezoid_area_ratio_sum_l269_269131


namespace problem_l269_269588

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2 * x + 3

theorem problem (f : ℝ → ℝ)
  (h₀ : ∀ x, f(x + 1) - f(x) = 2 * x - 1)
  (h₁ : f 0 = 3) :
  ( ∀ x, f x = x^2 - 2 * x + 3 ) ∧
  ( let y := λ x m, f (real.log 3 x + m) 
    ∃ m, ∀ x ∈ Icc (1/3) 3, y x m = 3 ↔ m = -1 ∨ m = 3 ) :=
sorry

end problem_l269_269588


namespace max_f_l269_269589

-- Defining the vertices of the n-sided polygon on the unit circle as complex numbers
def vertices (n : ℕ) : ℕ → ℂ :=
  λ i, Complex.exp (2 * Real.pi * Complex.I * (i : ℂ) / n)

-- Defining the function f(P), where P is a point on the unit circle represented as a complex number z
def f (n : ℕ) (z : ℂ) : ℂ :=
  ∏ i in Finset.range n, Complex.abs (z - vertices n i)

-- The statement to prove
theorem max_f (n : ℕ) (hz : Complex.abs z = 1) : 
  ∃ P, f n P = 2 :=
  sorry

end max_f_l269_269589


namespace calculate_exponentiation_l269_269542

theorem calculate_exponentiation : (64^(0.375) * 64^(0.125) = 8) :=
by sorry

end calculate_exponentiation_l269_269542


namespace cone_rotation_ratio_l269_269512

theorem cone_rotation_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rotation_eq : (20 : ℝ) * (2 * Real.pi * r) = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  let p := 1
  let q := 399
  1 + 399 = 400 := by
{
  sorry
}

end cone_rotation_ratio_l269_269512


namespace geometric_series_proof_l269_269995

theorem geometric_series_proof (y : ℝ) :
  ((1 + (1/3) + (1/9) + (1/27) + ∑' n : ℕ, (1 / 3^(n+1))) * 
   (1 - (1/3) + (1/9) - (1/27) + ∑' n : ℕ, ((-1)^n * (1 / 3^(n+1)))) = 
   1 + (1/y) + (1/y^2) + (∑' n : ℕ, (1 / y^(n+1)))) → y = 9 := by
  sorry

end geometric_series_proof_l269_269995


namespace MaryBusinessTripTime_l269_269074

theorem MaryBusinessTripTime
  (t_uber_house : Nat := 10) -- Time for Uber to get to her house in minutes
  (t_airport_factor : Nat := 5) -- Factor for time to get to the airport
  (t_check_bag : Nat := 15) -- Time to check her bag in minutes
  (t_security_factor : Nat := 3) -- Factor for time to get through security
  (t_wait_boarding : Nat := 20) -- Time waiting for flight to start boarding in minutes
  (t_take_off_factor : Nat := 2) -- Factor for time waiting for plane to be ready take off
: (t_uber_house + t_uber_house * t_airport_factor + t_check_bag + t_check_bag * t_security_factor + t_wait_boarding + t_wait_boarding * t_take_off_factor) / 60 = 3 := 
begin
  sorry
end

end MaryBusinessTripTime_l269_269074


namespace xy_difference_l269_269326

theorem xy_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end xy_difference_l269_269326


namespace max_possible_value_of_gcd_l269_269746

theorem max_possible_value_of_gcd (n : ℕ) : gcd ((8^n - 1) / 7) ((8^(n+1) - 1) / 7) = 1 := by
  sorry

end max_possible_value_of_gcd_l269_269746


namespace machine_worked_minutes_l269_269967

theorem machine_worked_minutes
  (shirts_today : ℕ)
  (rate : ℕ)
  (h1 : shirts_today = 8)
  (h2 : rate = 2) :
  (shirts_today / rate) = 4 :=
by
  sorry

end machine_worked_minutes_l269_269967


namespace sum_S_value_l269_269325

noncomputable def sum_S : ℂ :=
  ∑ n in (Finset.range 41), (complex.I ^ n) * real.cos (45 + 90 * n) * (float.pi / 180)

theorem sum_S_value :
  sum_S = (real.sqrt 2 / 2 : ℂ) * (21 - 20 * complex.I) :=
by
  sorry

end sum_S_value_l269_269325


namespace evaluate_fractional_exponent_l269_269247

theorem evaluate_fractional_exponent : 64^(2/3 : ℝ) = 16 := by
  have h1 : (64 : ℝ) = 2^6 := by
    norm_num
  rw [h1]
  have h2 : (2^6 : ℝ)^(2/3) = 2^(6 * (2/3)) := by
    rw [← Real.rpow_mul (by norm_num : 0 ≤ 2)] -- Using exponent properties
  rw [h2]
  calc 2^(6 * (2/3)) = 2^4 : by congr; ring
                ...  = 16  : by norm_num

end evaluate_fractional_exponent_l269_269247


namespace CE_eq_CD_l269_269344

theorem CE_eq_CD (A B C D E : Point) (h1 : Trapezoid A B C D) 
    (h2 : Parallel BC AD) (h3 : Length BC = 0.5 * Length AD) 
    (h4 : Perpendicular DE AB) : 
    Length CE = Length CD := 
by
  sorry

end CE_eq_CD_l269_269344


namespace kelly_grade_correct_l269_269032

variable (Jenny Jason Bob Kelly : ℕ)

def jenny_grade : ℕ := 95
def jason_grade := jenny_grade - 25
def bob_grade := jason_grade / 2
def kelly_grade := bob_grade + (bob_grade / 5)  -- 20% of Bob's grade is (Bob's grade * 0.20), which is the same as (Bob's grade / 5)

theorem kelly_grade_correct : kelly_grade = 42 :=
by
  sorry

end kelly_grade_correct_l269_269032


namespace complex_conjugate_of_z_l269_269789

def i : ℂ := complex.I

def z : ℂ := (2 + i) / (1 + i^2 + i^5)

theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * i :=
by {
    unfold z,
    -- The next few statements can formalize simplifying the denominator, but we skip the detailed proof by putting sorry.
    sorry
}

end complex_conjugate_of_z_l269_269789


namespace lucy_crayons_l269_269147

theorem lucy_crayons (W L : ℕ) (h1 : W = 1400) (h2 : W = L + 1110) : L = 290 :=
by {
  sorry
}

end lucy_crayons_l269_269147


namespace Hatter_is_older_l269_269387

theorem Hatter_is_older (H : nat)
  (M : nat)
  (synchronized_hour : ℕ)
  (next_synchronization : ℕ)
  (birth_year_H : nat)
  (birth_year_M : nat)
  (march_hare_birthday_month : nat)
  (h_gain : ℕ)
  (h_lose : ℕ)
  (next_birthday : ℕ) 
  (h_synchronized : birth_year_H = 1842)
  (m_birthday : march_hare_birthday_month = 3)
  (gain_per_hour : h_gain = 10)
  (lose_per_hour : h_lose = 10)
  (years_diff : next_birthday = 21) : H > M :=
by
  sorry

end Hatter_is_older_l269_269387


namespace marcia_project_hours_l269_269071

theorem marcia_project_hours (minutes_spent : ℕ) (minutes_per_hour : ℕ) 
  (h1 : minutes_spent = 300) 
  (h2 : minutes_per_hour = 60) : 
  (minutes_spent / minutes_per_hour) = 5 :=
by
  sorry

end marcia_project_hours_l269_269071


namespace value_of_x_l269_269142

theorem value_of_x :
  let x := (2010^2 - 2010 + 1) / (2010 + 1)
  in x = 4040091 / 2011 :=
by
  sorry

end value_of_x_l269_269142


namespace manu_wins_probability_l269_269355

theorem manu_wins_probability :
  let prob_turn (n : ℕ) := (1 / 2)^(5 * n) in
  let prob_sum := ∑' n, prob_turn n in
  prob_sum = 1 / 31 :=
begin
  let prob_turn := λ n : ℕ, (1 / 2)^(5 * n),
  let prob_sum := ∑' n, prob_turn n,
  have h_series_sum : prob_sum = (1 / 2^5) / (1 - 1 / 2^5),
  { sorry },
  rw h_series_sum,
  norm_num,
end

end manu_wins_probability_l269_269355


namespace hiring_probability_l269_269136

noncomputable def combinatorics (n k : ℕ) : ℕ := Nat.choose n k

theorem hiring_probability (n : ℕ) (h1 : combinatorics 2 2 = 1)
                          (h2 : combinatorics (n - 2) 1 = n - 2)
                          (h3 : combinatorics n 3 = n * (n - 1) * (n - 2) / 6)
                          (h4 : (6 : ℕ) / (n * (n - 1) : ℚ) = 1 / 15) :
  n = 10 :=
by
  sorry

end hiring_probability_l269_269136


namespace q_transformation_l269_269680

theorem q_transformation (w m z : ℝ) (q : ℝ) (h_q : q = 5 * w / (4 * m * z^2)) :
  let w' := 4 * w
  let m' := 2 * m
  let z' := 3 * z
  q = 5 * w / (4 * m * z^2) → (5 * w') / (4 * m' * (z'^2)) = (5 / 18) * q := by
  sorry

end q_transformation_l269_269680


namespace complex_conjugate_of_z_l269_269783

def i : ℂ := complex.I

def z : ℂ := (2 + i) / (1 + i^2 + i^5)

theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * i :=
by {
    unfold z,
    -- The next few statements can formalize simplifying the denominator, but we skip the detailed proof by putting sorry.
    sorry
}

end complex_conjugate_of_z_l269_269783


namespace place_2_5_in_F_l269_269352

/--
Jenny has fifteen slips of paper with numbers 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 3.5, 4, 4, 4.5, 4.5, 5. 
The slips are to be placed into six cups labeled F, G, H, I, J, K.
The sums of slips in each cup should be consecutive integers, increasing from F to K.
A slip with 3 goes into cup H.
A slip with 4 goes into cup J.
Prove that the slip with 2.5 must go into cup F.
-/
theorem place_2_5_in_F :
  let slips := [1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 3.5, 4, 4, 4.5, 4.5, 5],
      cups := ["F", "G", "H", "I", "J", "K"],
      sums := [5, 6, 7, 8, 9, 10] in
  (3 ∈ {H} ∧ 4 ∈ {J}) →
  (2.5 ∈ {F}) :=
begin
  sorry
end

end place_2_5_in_F_l269_269352


namespace seventh_test_score_107_l269_269046

noncomputable def juliet_score_on_seventh_test (scores : Fin 8 → ℕ) : Prop :=
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℕ),
  (∀ (i j : Fin 8), i ≠ j → scores i ≠ scores j) ∧
  (∀ (i : Fin 8), 88 ≤ scores i ∧ scores i ≤ 97) ∧
  (scores 7 = 92) ∧
  (∀ (n : ℕ), n ∈ Fin.range 8 → (∑ i in Fin.range (n + 1), (scores i) / (n + 1) = (∑ i in Fin.range (n+1), scores i) / (n + 1))) ∧
  (scores 6 = 107)

theorem seventh_test_score_107 :
  ∀ (scores : Fin 8 → ℕ),
    juliet_score_on_seventh_test scores →
    scores 6 = 107 := by
  sorry

end seventh_test_score_107_l269_269046


namespace arrange_balls_in_descending_order_l269_269851

-- Define the masses of the four different balls
variables {m_A m_B m_C m_D : ℝ}

/--
Given 4 balls of different masses, at most 5 weighings on 
a balance scale without weights are sufficient to arrange
these balls in descending order of mass.
-/
theorem arrange_balls_in_descending_order
  (h_diff : m_A ≠ m_B ∧ m_B ≠ m_C ∧ m_C ≠ m_D ∧ m_A ≠ m_C ∧ m_A ≠ m_D ∧ m_B ≠ m_D) :
  ∃ (n : ℕ), n ≤ 5 ∧ (∀ (order : list ℝ),
    order ~ [m_A, m_B, m_C, m_D] → -- ensures that order is a permutation of [m_A, m_B, m_C, m_D]
    sorted (>) order → -- ensures order is sorted in descending order
    ∃ (weighings : list (ℝ × ℝ)),  -- list of pairs of comparisons
    length weighings = n) :=
sorry

end arrange_balls_in_descending_order_l269_269851


namespace cos_B_find_b_l269_269715

theorem cos_B (A B C : ℝ) (a b c : ℝ) (h : sin (A + C) = 8 * sin (B / 2) ^ 2) :
  cos B = 15 / 17 :=
by 
sorry

theorem find_b (A B C : ℝ) (a b c : ℝ) (h1 : a + c = 6) 
  (h2 : 1 / 2 * a * c * sin B = 2) (h3 : cos B = 15 / 17) :
  b = 2 :=
by 
sorry

end cos_B_find_b_l269_269715


namespace ratio_x_y_l269_269199

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end ratio_x_y_l269_269199


namespace min_lines_to_cover_points_l269_269850

theorem min_lines_to_cover_points (n : ℕ)
  (points : Finset Point) (lines : Finset (Set Point))
  (a : lines → ℕ) :
  (∀ l ∈ lines, a l = (points ∩ l).card) →
  (∑ l in lines, a l = 250) →
  (points.card = 100) →
  (∃ l ∈ lines, true) →
  n = lines.card →
  n ≥ 21 :=
by
  sorry

end min_lines_to_cover_points_l269_269850


namespace parabola_equation_proof_chord_length_proof_l269_269583

-- Define the parabola equation and point M conditions
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def point_on_parabola (p : ℝ) (x y : ℝ) := parabola p x y

-- Define point M and focus F conditions
def point_M := (2, m : ℝ)
def focus_F (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define conditions: The distance from M to F is 3
def distance (x1 y1 x2 y2 : ℝ) := (√((x2 - x1)^2 + (y2 - y1)^2))
def distance_condition (p : ℝ) (m : ℝ) := distance 2 m (focus_F p).1 (focus_F p).2 = 3

-- Define line l passing through the focus with inclination 60 degrees
def line_l (p : ℝ) (x y : ℝ) := y = (√3) * (x - p/2)

-- Prove the parabola equation is y^2 = 4x
theorem parabola_equation_proof : ∃ p : ℝ, point_on_parabola p 2 m ∧ distance_condition p m → p = 2 :=
by
  sorry

-- Prove the length of the chord intercepted by line l on the parabola is 16/3
theorem chord_length_proof : ∀ m : ℝ, point_on_parabola 2 2 m ∧ distance_condition 2 m → 
  let (A, B) := ((1 - (√3), -√3)), ((1 + √3), √3) in distance A.1 A.2 B.1 B.2 = 16/3 :=
by
  sorry

end parabola_equation_proof_chord_length_proof_l269_269583


namespace a_2n_is_square_l269_269368

def a_seq (n : ℕ) : ℕ :=
/- Definition of the sequence a_n according to the problem -/

theorem a_2n_is_square (n : ℕ) (h : 2 ≤ n) :
  a_seq (2 * n) = (a_seq n + a_seq (n - 2)) ^ 2 :=
sorry

end a_2n_is_square_l269_269368


namespace find_n_of_arithmetic_sequence_l269_269836

theorem find_n_of_arithmetic_sequence :
  ∃ n : ℕ, (∀ (a : ℕ → ℤ), a 2 = 12 ∧ a n = -20 ∧ (∀ m : ℕ, a (m + 1) = a m - 2) → n = 18) :=
by
  sorry

end find_n_of_arithmetic_sequence_l269_269836


namespace smallest_x_plus_y_l269_269615

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269615


namespace Jeans_average_speed_l269_269993

/-- Definitions based on conditions in the problem --/
def Chantal_ascent_speed := 5
def Chantal_rest_time := 0.25
def Chantal_descent_speed := 4
def Jean_meeting_point_to_trailhead_distance_fraction := 0.75

/-- Prove the calculation of Jean's average speed until they met. --/
theorem Jeans_average_speed :
  ∀ (d : ℝ), d > 0 → 
  let t1 := d / Chantal_ascent_speed in
  let trest := Chantal_rest_time in
  let t2 := (Jean_meeting_point_to_trailhead_distance_fraction * d) / Chantal_descent_speed in
  let T := t1 + trest + t2 in
  let Jean_speed := d / T in
  Jean_speed = 80 / 31 :=
by {
  intros d hd,
  let t1 := d / Chantal_ascent_speed,
  let trest := Chantal_rest_time,
  let t2 := Jean_meeting_point_to_trailhead_distance_fraction * d / Chantal_descent_speed,
  let T := t1 + trest + t2,
  let Jean_speed := d / T,
  show Jean_speed = 80 / 31,
  have hdenom : T = (31 * d + 20) / 80, 
  { calc T 
      = d / 5 + 0.25 + 3 * d / 16 : by refl
  ... = 16 * d / 80 + 20 / 80 + 15 * d / 80 : by norm_cast
  ... = (31 * d + 20) / 80 : by simp }, 
  have : 80 / 31 + 20 / 31 / d ≤ 80 / 31 := begin 
                  sorry
                end,
  exact this,
}

end Jeans_average_speed_l269_269993


namespace M_eq_N_l269_269736

def M : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k + 1) * Real.pi}
def N : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k - 1) * Real.pi}

theorem M_eq_N : M = N := by
  sorry

end M_eq_N_l269_269736


namespace count_primes_between_60_and_85_l269_269316

-- Define the range of interest
def range : Finset ℕ := Finset.range (85 + 1) \ Finset.range 60

-- Define the prime subset of the range
def primes_in_range : Finset ℕ := range.filter Nat.Prime

-- The theorem we aim to prove
theorem count_primes_between_60_and_85 : primes_in_range.card = 6 := by {
  sorry
}

end count_primes_between_60_and_85_l269_269316


namespace problem_l269_269369

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem problem : f (g 2) + g (f 2) = 38 / 5 :=
by
  sorry

end problem_l269_269369


namespace frog_weight_difference_l269_269429

theorem frog_weight_difference
  (large_frog_weight : ℕ)
  (small_frog_weight : ℕ)
  (h1 : large_frog_weight = 10 * small_frog_weight)
  (h2 : large_frog_weight = 120) :
  large_frog_weight - small_frog_weight = 108 :=
by
  sorry

end frog_weight_difference_l269_269429


namespace simplify_expression1_simplify_expression2_l269_269990

section
variables (a b : ℝ)

theorem simplify_expression1 : -b*(2*a - b) + (a + b)^2 = a^2 + 2*b^2 :=
sorry
end

section
variables (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2)

theorem simplify_expression2 : (1 - (x/(2 + x))) / ((x^2 - 4)/(x^2 + 4*x + 4)) = 2/(x - 2) :=
sorry
end

end simplify_expression1_simplify_expression2_l269_269990


namespace complex_conjugate_of_z_l269_269754

-- Define z based on the given expression
noncomputable def z : ℂ := (2 + complex.i) / (1 + complex.i^2 + complex.i^5)

-- State the theorem to prove the complex conjugate of z is 1 + 2i
theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * complex.i :=
sorry

end complex_conjugate_of_z_l269_269754


namespace combined_average_speed_l269_269459

-- Definitions based on conditions
def distance_A : ℕ := 250
def time_A : ℕ := 4

def distance_B : ℕ := 480
def time_B : ℕ := 6

def distance_C : ℕ := 390
def time_C : ℕ := 5

def total_distance : ℕ := distance_A + distance_B + distance_C
def total_time : ℕ := time_A + time_B + time_C

-- Prove combined average speed
theorem combined_average_speed : (total_distance : ℚ) / (total_time : ℚ) = 74.67 :=
  by
    sorry

end combined_average_speed_l269_269459


namespace harmony_numbers_with_first_digit_2_l269_269468

def is_harmony_number (n : ℕ) : Prop :=
  let d4 := n % 10
  let d3 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d1 := (n / 1000) % 10
  (d1 + d2 + d3 + d4 = 6)

def starts_with_2 (n : ℕ) : Prop :=
  (n / 1000) = 2

theorem harmony_numbers_with_first_digit_2 : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ is_harmony_number(n) ∧ starts_with_2(n)}.to_finset.card = 15 := 
sorry

end harmony_numbers_with_first_digit_2_l269_269468


namespace last_student_score_is_61_l269_269810

noncomputable def average_score_19_students := 82
noncomputable def average_score_20_students := 84
noncomputable def total_students := 20
noncomputable def oliver_multiplier := 2

theorem last_student_score_is_61 
  (total_score_19_students : ℝ := total_students - 1 * average_score_19_students)
  (total_score_20_students : ℝ := total_students * average_score_20_students)
  (oliver_score : ℝ := total_score_20_students - total_score_19_students)
  (last_student_score : ℝ := oliver_score / oliver_multiplier) :
  last_student_score = 61 :=
sorry

end last_student_score_is_61_l269_269810


namespace complex_conjugate_of_z_l269_269785

def i : ℂ := complex.I

def z : ℂ := (2 + i) / (1 + i^2 + i^5)

theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * i :=
by {
    unfold z,
    -- The next few statements can formalize simplifying the denominator, but we skip the detailed proof by putting sorry.
    sorry
}

end complex_conjugate_of_z_l269_269785


namespace not_possible_to_form_closed_broken_line_l269_269212

-- Define the conditions as variables and parameters
variable (lateral_edges base_edges total_edges : ℕ)
variable (H_lateral_edges : lateral_edges = 373)
variable (H_total_edges : total_edges = 1119)
variable (H_base_edges : 2 * base_edges + lateral_edges = total_edges)

-- Define the proposition in Lean
theorem not_possible_to_form_closed_broken_line :
  ∀ (base_edges lateral_edges total_edges : ℕ), 
  lateral_edges = 373 → 
  total_edges = 1119 → 
  2 * base_edges + lateral_edges = total_edges → 
  ¬ (∃ edges_translation : ℕ → ℕ × ℕ × ℕ, 
     (∀ i < 1119, (edges_translation (i + 1) - edges_translation i).z ≠ 1) ∧
     (edges_translation 1119 = edges_translation 0)) :=
by
  intros base_edges lateral_edges total_edges H1 H2 H3
  sorry  -- Proof omitted here

end not_possible_to_form_closed_broken_line_l269_269212


namespace tan_increasing_interval_l269_269110

noncomputable def increasing_interval (k : ℤ) : Set ℝ := 
  {x | (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12)}

theorem tan_increasing_interval (k : ℤ) : 
  ∀ x : ℝ, (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12) ↔ 
    (∃ y, y = (2 * x + Real.pi / 3) ∧ Real.tan y > Real.tan (2 * x + Real.pi / 3 - 1e-6)) :=
sorry

end tan_increasing_interval_l269_269110


namespace perp_lines_from_planes_l269_269686

-- Define the entities and relationships
variables {Point Line Plane : Type} [Geometry3D Point Line Plane]
variables (α β : Plane) (a b : Line)

-- Define the conditions
variable (⟦α ∥ β⟧ : Parallel α β)
variable (⟦a ⦿ α⟧ : Perpendicular a α)
variable (⟦b ⊂ β⟧ : Subset b β)

-- The theorem to be proven
theorem perp_lines_from_planes : 
  Parallel α β → Perpendicular a α → Subset b β → Perpendicular a b := 
by 
  intros ⟦α ∥ β⟧ ⟦a ⦿ α⟧ ⟦b ⊂ β⟧ 
  sorry

end perp_lines_from_planes_l269_269686


namespace perfect_squares_with_specific_ones_digit_count_l269_269315

theorem perfect_squares_with_specific_ones_digit_count : 
  ∃ n : ℕ, (∀ k : ℕ, k < 2500 → (k % 10 = 4 ∨ k % 10 = 5 ∨ k % 10 = 6) ↔ ∃ m : ℕ, m < n ∧ (m % 10 = 2 ∨ m % 10 = 8 ∨ m % 10 = 5 ∨ m % 10 = 4 ∨ m % 10 = 6) ∧ k = m * m) 
  ∧ n = 25 := 
by 
  sorry

end perfect_squares_with_specific_ones_digit_count_l269_269315


namespace find_m_n_l269_269371

def a : ℕ → ℚ
def b : ℕ → ℚ

axiom a0_def : a 0 = 3
axiom b0_def : b 0 = 4

axiom a_rec : ∀ n : ℕ, a (n + 1) = (a n ^ 2) / (b n)
axiom b_rec : ∀ n : ℕ, b (n + 1) = (b n ^ 2) / (a n)

noncomputable def b8_expr : ℚ := b 8

theorem find_m_n (m n : ℕ) :
  b8_expr = (4^m : ℚ) / (3^n : ℚ) → (m, n) = (2188, 3280) := by
  sorry

end find_m_n_l269_269371


namespace number_of_subsets_of_s_l269_269662

-- Define the set s
def s : Set ℕ := {0, 1, 2}

-- State the theorem
theorem number_of_subsets_of_s : ∃ n, n = 8 ∧ (Finset.powerset (Finset.ofSet s)).card = n := by
  sorry

end number_of_subsets_of_s_l269_269662


namespace find_x_square_value_l269_269129

noncomputable def positive_real_solution (x : ℝ) : Prop :=
  x > 0 ∧ sin (arctan (2 * x)) = x

theorem find_x_square_value (x : ℝ) (h : positive_real_solution x) : x^2 = 3 / 4 :=
  sorry

end find_x_square_value_l269_269129


namespace total_people_at_gathering_l269_269213

theorem total_people_at_gathering (total_wine : ℕ) (total_soda : ℕ) (both_wine_soda : ℕ) 
    (H1 : total_wine = 26) (H2 : total_soda = 22) (H3 : both_wine_soda = 17) : 
    total_wine - both_wine_soda + total_soda - both_wine_soda + both_wine_soda = 31 := 
by
  rw [H1, H2, H3]
  exact Nat.correct_answer = 31 -- combining results
  rw [Nat.sub_add_cancel (Nat.le_of_lt (sorry))] -- just using properties
  exact nat.add_comm 17 9 -- final proof step
  sorry -- ending suggestion

end total_people_at_gathering_l269_269213


namespace total_rubber_bands_l269_269655

theorem total_rubber_bands (harper_bands : ℕ) (brother_bands: ℕ):
  harper_bands = 15 →
  brother_bands = harper_bands - 6 →
  harper_bands + brother_bands = 24 :=
by
  intros h1 h2
  sorry

end total_rubber_bands_l269_269655


namespace largest_int_not_exceeding_700_pi_l269_269841

theorem largest_int_not_exceeding_700_pi : ⌊700 * Real.pi⌋ = 2199 := sorry

end largest_int_not_exceeding_700_pi_l269_269841


namespace center_of_symmetry_l269_269931

-- Define the given conditions
def has_axis_symmetry_x (F : Set (ℝ × ℝ)) : Prop := 
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, y) ∈ F

def has_axis_symmetry_y (F : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ F → (x, -y) ∈ F
  
-- Define the central proof goal
theorem center_of_symmetry (F : Set (ℝ × ℝ)) (H1: has_axis_symmetry_x F) (H2: has_axis_symmetry_y F) :
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, -y) ∈ F :=
sorry

end center_of_symmetry_l269_269931


namespace find_e1_l269_269585

noncomputable def e_1 := real

theorem find_e1
  (P : Type) (M : Type) (Γ : Type)
  (F1 F2 : Type)
  (cos_angle : real)
  (e1 e2 : real)
  (h1 : P ∈ M ∧ P ∈ Γ)
  (h2 : cos_angle = 4 / 5)
  (h3 : e2 = 2 * e1) :
  e1 = real.sqrt 130 / 20 :=
begin
  sorry,
end

end find_e1_l269_269585


namespace average_weight_of_whole_class_l269_269153

theorem average_weight_of_whole_class :
  ∀ (n_a n_b : ℕ) (w_avg_a w_avg_b : ℝ),
    n_a = 60 →
    n_b = 70 →
    w_avg_a = 60 →
    w_avg_b = 80 →
    (n_a * w_avg_a + n_b * w_avg_b) / (n_a + n_b) = 70.77 :=
by
  intros n_a n_b w_avg_a w_avg_b h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_weight_of_whole_class_l269_269153


namespace probability_condition_l269_269108

theorem probability_condition
  (A B : Prop)
  (P : Prop → ℝ)
  (h_independent : P (A ∧ B) = P A * P B)
  (h_PA_eq_2PB : P A = 2 * P B)
  (h_PUnion_eq_3Intersection : P (A ∨ B) = 3 * P (A ∧ B))
  : P A = 0.75 :=
begin
  -- proof goes here
  sorry
end

end probability_condition_l269_269108


namespace equivalent_annual_rate_8_percent_quarterly_is_8_24_l269_269242

noncomputable def quarterly_interest_rate (annual_rate : ℚ) := annual_rate / 4

noncomputable def growth_factor (interest_rate : ℚ) := 1 + interest_rate / 100

noncomputable def annual_growth_factor_from_quarterly (quarterly_factor : ℚ) := quarterly_factor ^ 4

noncomputable def equivalent_annual_interest_rate (annual_growth_factor : ℚ) := 
  ((annual_growth_factor - 1) * 100)

theorem equivalent_annual_rate_8_percent_quarterly_is_8_24 :
  let quarter_rate := quarterly_interest_rate 8
  let quarterly_factor := growth_factor quarter_rate
  let annual_factor := annual_growth_factor_from_quarterly quarterly_factor
  equivalent_annual_interest_rate annual_factor = 8.24 := by
  sorry

end equivalent_annual_rate_8_percent_quarterly_is_8_24_l269_269242


namespace function_identity_l269_269251

-- Define the conditions as given in the problem
variable (f : ℚ+ → ℚ+)
axiom condition1 : ∀ x : ℚ+, f (x + 1) = f x + 1
axiom condition2 : ∀ x : ℚ+, f (x^2) = f x ^ 2

-- State the theorem that f(x) = x for all x in ℚ+
theorem function_identity : ∀ x : ℚ+, f x = x := by
  sorry

end function_identity_l269_269251


namespace oil_amount_in_liters_l269_269954

theorem oil_amount_in_liters (b v : ℕ) (hb : b = 20) (hv : v = 200) :
  (b * v) / 1000 = 4 :=
by
  have h1 : b * v = 4000 := by
    rw [hb, hv]
    exact rfl
  rw [h1]
  norm_num

end oil_amount_in_liters_l269_269954


namespace shaded_region_area_l269_269256

-- Define the side of the square
def side_length_square : ℝ := 40

-- Define the legs of the triangles
def leg_length_triangle : ℝ := 25

-- Prove that the area of the shaded region is 975 square units
theorem shaded_region_area :
  let square_area := side_length_square^2,
      triangle_area := 0.5 * leg_length_triangle * leg_length_triangle,
      total_triangle_area := 2 * triangle_area,
      shaded_area := square_area - total_triangle_area
  in shaded_area = 975 := 
by
  sorry

end shaded_region_area_l269_269256


namespace tod_drive_west_95_miles_l269_269858

theorem tod_drive_west_95_miles (distance_to_north speed time : ℝ)
  (h_distance_to_north : distance_to_north = 55)
  (h_speed : speed = 25)
  (h_time : time = 6) :
  let total_distance := speed * time in
  let distance_to_west := total_distance - distance_to_north in
  distance_to_west = 95 :=
by
  sorry

end tod_drive_west_95_miles_l269_269858


namespace middle_card_is_4_l269_269458

-- Definitions related to the problem
def cards : Type := ℕ
def valid_set (a b c : cards) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 15 ∧
  a < b ∧ b < c

-- Statements made by Casey, Tracy, and Stacy
def casey_statement (a b c : cards) : Prop :=
  ¬∀ x y : cards, (valid_set a x y) → (x = b ∧ y = c)

def tracy_statement (a b c : cards) : Prop :=
  ¬∀ x y : cards, (valid_set x y c) → (x = a ∧ y = b)

def stacy_statement (a b c : cards) : Prop :=
  ¬∀ x y : cards, (valid_set x b y) → (x = a ∧ y = c)

-- Main theorem: The number on the middle card is 4
theorem middle_card_is_4 :
  ∃ a b c : cards,
    valid_set a b c ∧
    casey_statement a b c ∧
    tracy_statement a b c ∧
    stacy_statement a b c ∧
    b = 4 :=
by
  sorry

end middle_card_is_4_l269_269458


namespace number_of_valid_ns_l269_269716

theorem number_of_valid_ns :
  ∃ (n : ℝ), (n = 8 ∨ n = 1/2) ∧ ∀ n₁ n₂, (n₁ = 8 ∨ n₁ = 1/2) ∧ (n₂ = 8 ∨ n₂ = 1/2) → n₁ = n₂ :=
sorry

end number_of_valid_ns_l269_269716


namespace number_of_noncongruent_dominoes_l269_269812

/--
On an infinite checkerboard, the union of any two distinct unit squares is called a (disconnected) domino. 
A domino is said to be of type \((a, b)\), with \(a \leq b\) integers not both zero, if the centers of the 
two squares are separated by a distance of \(a\) in one orthogonal direction and \(b\) in the other. 

Two dominoes are said to be congruent if they are of the same type. A rectangle is said to be \((a, b)\)-tileable 
if it can be partitioned into dominoes of type \((a, b)\).

Let \(0 < m \leq n\) be integers. Prove that the number of different (i.e., noncongruent) dominoes that can be 
formed by choosing two squares of an \(m \times n\) array is equal to \(mn - \frac{m^2}{2} + \frac{m}{2} - 1\).
-/
theorem number_of_noncongruent_dominoes (m n : ℕ) (h : 0 < m ∧ m ≤ n) : 
  nat.mul m n - nat.div (nat.mul m m - m) 2 + nat.div m 2 - 1 = 
   mn - (m.succ - 1) * (m.succ - 1) / 2 + (m.succ - 1) / 2 - 1 := 
sorry

end number_of_noncongruent_dominoes_l269_269812


namespace ratio_of_money_l269_269354

-- Conditions
def amount_given := 14
def cost_of_gift := 28

-- Theorem statement to prove
theorem ratio_of_money (h1 : amount_given = 14) (h2 : cost_of_gift = 28) :
  amount_given / cost_of_gift = 1 / 2 := by
  sorry

end ratio_of_money_l269_269354


namespace origin_outside_circle_l269_269635

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
    let circle := λ x y : ℝ, x^2 + y^2 + 2 * a * x + 2 * y + (a - 1)^2
    in ∀ (x y : ℝ), circle x y = 0 → (0, 0) ≠ (x, y) :=
by
    let circle := λ x y : ℝ, x^2 + y^2 + 2 * a * x + 2 * y + (a - 1)^2
    have h1 : circle 0 0 = (a-1)^2,
    {
      unfold circle,
      ring,
    }
    have h2 : (a-1)^2 > 0,
    {
      sorry, -- to be proven using 0 < a < 1
    }
    intros x y eq_circ,
    intros h_not_eq,
    sorry -- proof that origin is outside the circle using the conditions and the earlier steps.

end origin_outside_circle_l269_269635


namespace number_of_adult_males_l269_269948

def population := 480
def ratio_children := 1
def ratio_adult_males := 2
def ratio_adult_females := 2
def total_ratio_parts := ratio_children + ratio_adult_males + ratio_adult_females

theorem number_of_adult_males : 
  (population / total_ratio_parts) * ratio_adult_males = 192 :=
by
  sorry

end number_of_adult_males_l269_269948


namespace complex_conjugate_of_z_l269_269753

-- Define z based on the given expression
noncomputable def z : ℂ := (2 + complex.i) / (1 + complex.i^2 + complex.i^5)

-- State the theorem to prove the complex conjugate of z is 1 + 2i
theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * complex.i :=
sorry

end complex_conjugate_of_z_l269_269753


namespace union_of_A_and_B_l269_269408

variable (a b : ℕ)

def A : Set ℕ := {3, 2^a}
def B : Set ℕ := {a, b}
def intersection_condition : A a ∩ B a b = {2} := by sorry

theorem union_of_A_and_B (h : A a ∩ B a b = {2}) : 
  A a ∪ B a b = {1, 2, 3} := by sorry

end union_of_A_and_B_l269_269408


namespace solve_for_x_l269_269666

theorem solve_for_x (x : ℝ) (h : 3 * x + 20 = (1 / 3) * (7 * x + 45)) : x = -7.5 :=
sorry

end solve_for_x_l269_269666


namespace volume_ratio_of_tetrahedrons_l269_269019

-- Define the regular tetrahedron and centroid properties
variables {P A B C D G : Type} [regular_tetrahedron P A B C D] [centroid_of G P B C]

-- Theorem statement
theorem volume_ratio_of_tetrahedrons :
  let V_GPAB := volume (tetrahedron G P A B)
  let V_GPAD := volume (tetrahedron G P A D)
  V_GPAD / V_GPAB = 2 :=
begin
  sorry
end

end volume_ratio_of_tetrahedrons_l269_269019


namespace participants_neither_coffee_nor_tea_l269_269532

-- Define the total number of participants
def total_participants : ℕ := 30

-- Define the number of participants who drank coffee
def coffee_drinkers : ℕ := 15

-- Define the number of participants who drank tea
def tea_drinkers : ℕ := 18

-- Define the number of participants who drank both coffee and tea
def both_drinkers : ℕ := 8

-- The proof statement for the number of participants who drank neither coffee nor tea
theorem participants_neither_coffee_nor_tea :
  total_participants - (coffee_drinkers + tea_drinkers - both_drinkers) = 5 := by
  sorry

end participants_neither_coffee_nor_tea_l269_269532


namespace complex_conjugate_l269_269772

theorem complex_conjugate (z : ℂ) : 
  (i : ℂ)² = -1 → (i : ℂ)⁵ = i → 
  z = (2 + i) / (1 + (-1) + i) → 
  conj z = 1 + 2 * i :=
by
  intros
  sorry

end complex_conjugate_l269_269772


namespace find_angle_PCA_l269_269919

noncomputable def circumscribed_triangle (A B C P Q K L : Point) (ω : Circle) : Prop :=
∃ (circum_circle_triangle : Circle),
circum_circle_triangle = ω ∧ 
(tangent_at_C : Line) (tangent_at_C_tangent : TangentToCircle tangent_at_C ω C) ∧ 
(line_intersecting_rayBA_at_P : P ∈ ray A B) ∧ 
(point_Q_on_ray_PC_beyond_C : Q ∈ ray P C \ {C} ∧ PC = QC) ∧ 
(point_K_on_BQ_intersecting_ω : K ∈ ω ∧ K ≠ B ∧ K ≠ Q) ∧ 
(arc_BK_contains_L_with_angle_LAK_equal_angle_CQB : ∃ (arc_BK : Arc), arc_BK = smaller_arc B K ω ∧ L ∈ arc_BK ∧ ∠LAK = ∠CQB) ∧ 
(given_angle_ALQ_equal_60 : ∠ALQ = 60)

theorem find_angle_PCA (A B C P Q K L : Point) (ω : Circle):
circumscribed_triangle A B C P Q K L ω → ∠PCA = 30 := by
sorry

end find_angle_PCA_l269_269919


namespace rectangular_plot_length_l269_269508

-- Define the conditions
def pole_distance := 5
def num_poles := 32
def width := 30

-- Define what we need to prove
theorem rectangular_plot_length :
  let perimeter := (num_poles - 1) * pole_distance in
  let length := (perimeter / 2) - width in
  length = 47.5 :=
by
  sorry

end rectangular_plot_length_l269_269508


namespace greatest_possible_perimeter_l269_269335

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), 
    (x > 3) ∧ 
    (x < 5) ∧ 
    (let sides := [x, 5*x, 20] in 
     6*x > 20 ∧ 
     x + 20 > 5*x ∧ 
     (5*x + x + 20 = 44)) :=
by
  sorry

end greatest_possible_perimeter_l269_269335


namespace pieces_in_each_package_l269_269395

-- Definitions from conditions
def num_packages : ℕ := 5
def extra_pieces : ℕ := 6
def total_pieces : ℕ := 41

-- Statement to prove
theorem pieces_in_each_package : ∃ x : ℕ, num_packages * x + extra_pieces = total_pieces ∧ x = 7 :=
by
  -- Begin the proof with the given setup
  sorry

end pieces_in_each_package_l269_269395


namespace product_as_difference_of_squares_l269_269827

theorem product_as_difference_of_squares (a b : ℝ) : 
  a * b = ( (a + b) / 2 )^2 - ( (a - b) / 2 )^2 :=
by
  sorry

end product_as_difference_of_squares_l269_269827


namespace equation_of_line_AB_l269_269338

theorem equation_of_line_AB :
  ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → x ≥ 0 → 
  let H := (1, 0) in
  let B := (sqrt (3) / 2 + 1 / 2, sqrt (3) / 2 - 1 / 2) in 
  x * sqrt 3 + y - sqrt 3 - 1 = 0) :=
begin
  use [sqrt 3, 1, -sqrt 3 - 1],
  intros x y hx hy,
  sorry -- Filling the proof is unnecessary as per the requirement
end

end equation_of_line_AB_l269_269338


namespace tan_β_value_l269_269292

-- Definitions based on the conditions
def α_is_acute (α : ℝ) : Prop := 0 < α ∧ α < (π / 2)
def β_is_acute (β : ℝ) : Prop := 0 < β ∧ β < (π / 2)
def cos_α (α : ℝ) : Prop := cos α = 3 / 5
def tan_diff (α β : ℝ) : Prop := tan (α - β) = -1 / 3

-- The theorem we need to prove
theorem tan_β_value (α β : ℝ) (hα : α_is_acute α) (hβ : β_is_acute β) (hcos : cos_α α) (htan_diff : tan_diff α β) : 
  tan β = 3 :=
by
  sorry

end tan_β_value_l269_269292


namespace largest_fraction_l269_269478

theorem largest_fraction :
  ∀ (a b c d : ℚ),
    a = 5 / 11 →
    b = 6 / 13 →
    c = 19 / 39 →
    d = 159 / 319 →
    101 / 199 > a ∧ 101 / 199 > b ∧ 101 / 199 > c ∧ 101 / 199 > d :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  split
  all_goals { sorry }

end largest_fraction_l269_269478


namespace sum_of_roots_quadratic_l269_269474

theorem sum_of_roots_quadratic (z : ℂ) (h : z^2 = 12 * z - 7) : z * (1 - 1) + z * (1 - 1) = 12 :=
begin
  sorry
end

end sum_of_roots_quadratic_l269_269474


namespace simplify_polynomial_problem_l269_269411

theorem simplify_polynomial_problem (p : ℝ) :
  (5 * p^4 - 4 * p^3 + 3 * p + 2) + (-3 * p^4 + 2 * p^3 - 7 * p^2 + 8) = 2 * p^4 - 2 * p^3 - 7 * p^2 + 3 * p + 10 := 
by
  sorry

end simplify_polynomial_problem_l269_269411


namespace max_S_subset_D_n_l269_269062

noncomputable def max_size_S (n : ℕ) : ℕ :=
  (3 * (n + 1)^2 + 1) / 4

theorem max_S_subset_D_n (n : ℕ) :
  ∀ S ⊆ (λ a b c : ℕ, 2^a * 3^b * 5^c) '' (set.Icc 0 n ×ˢ set.Icc 0 n ×ˢ set.Icc 0 n),
  (∀ x y ∈ S, x ≠ y → ¬(x ∣ y))
  → ∃ S' ⊆ S, S'.card = max_size_S n := sorry

end max_S_subset_D_n_l269_269062


namespace num_subsets_set_3_l269_269659

theorem num_subsets_set_3 : (set.powerset {0, 1, 2}).card = 8 := 
sorry

end num_subsets_set_3_l269_269659


namespace complex_conjugate_of_z_l269_269766

theorem complex_conjugate_of_z : 
  let z := (2 + complex.i) / (1 + complex.i^2 + complex.i^5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269766


namespace number_of_subsets_of_s_l269_269661

-- Define the set s
def s : Set ℕ := {0, 1, 2}

-- State the theorem
theorem number_of_subsets_of_s : ∃ n, n = 8 ∧ (Finset.powerset (Finset.ofSet s)).card = n := by
  sorry

end number_of_subsets_of_s_l269_269661


namespace rate_per_sq_meter_l269_269114

theorem rate_per_sq_meter (L W A R C : ℝ) (hL : L = 5.5) (hW : W = 3.75) (hC : C = 12375) (hA : A = L * W) (hR : R = C / A) : R = 600 :=
by
  rw [hL, hW, hC, hA, hR]
  sorry

end rate_per_sq_meter_l269_269114


namespace impurities_mass_is_6_l269_269482

-- Define the mass of the sample and the mass of pure sulfur as constants
constant mass_sample : ℕ
constant mass_pure_sulfur : ℕ

-- State the assumptions
axiom h1 : mass_sample = 38
axiom h2 : mass_pure_sulfur = 32

-- Define the mass of impurities as a function of the mass of the sample and the mass of pure sulfur
def mass_impurities (m_sample m_pure_sulfur : ℕ) : ℕ := m_sample - m_pure_sulfur

-- State the theorem that the mass of impurities is 6 g
theorem impurities_mass_is_6 : mass_impurities mass_sample mass_pure_sulfur = 6 :=
by
  rw [mass_impurities, h1, h2]
  decide

end impurities_mass_is_6_l269_269482


namespace ariana_average_speed_l269_269397

theorem ariana_average_speed
  (sadie_speed : ℝ)
  (sadie_time : ℝ)
  (ariana_time : ℝ)
  (sarah_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (sadie_speed_eq : sadie_speed = 3)
  (sadie_time_eq : sadie_time = 2)
  (ariana_time_eq : ariana_time = 0.5)
  (sarah_speed_eq : sarah_speed = 4)
  (total_time_eq : total_time = 4.5)
  (total_distance_eq : total_distance = 17) :
  ∃ ariana_speed : ℝ, ariana_speed = 6 :=
by {
  sorry
}

end ariana_average_speed_l269_269397


namespace license_plates_count_correct_l269_269665

def is_valid_license_plate (license_plate : String) : Prop :=
  license_plate.length = 4 ∧
  license_plate[0].isAlpha ∧
  (license_plate[1].isAlpha ∨ license_plate[1].isDigit) ∧
  (license_plate[2].isAlpha ∨ license_plate[2].isDigit) ∧
  license_plate[3].isDigit ∧
  license_plate[0] = license_plate[2]

def count_valid_license_plates : Nat :=
  26 * 36 * 1 * 10

theorem license_plates_count_correct :
  count_valid_license_plates = 9360 :=
  by
    simp [count_valid_license_plates]
    exact rfl

end license_plates_count_correct_l269_269665


namespace conjugate_z_is_1_add_2i_l269_269758

open Complex

def z_def : ℂ := ((2 : ℂ) + I) / (1 + I^2 + I^5)

theorem conjugate_z_is_1_add_2i : conj z_def = (1 : ℂ) + 2 * I := 
sorry

end conjugate_z_is_1_add_2i_l269_269758


namespace conjugate_z_is_1_add_2i_l269_269760

open Complex

def z_def : ℂ := ((2 : ℂ) + I) / (1 + I^2 + I^5)

theorem conjugate_z_is_1_add_2i : conj z_def = (1 : ℂ) + 2 * I := 
sorry

end conjugate_z_is_1_add_2i_l269_269760


namespace least_integer_square_eq_triple_plus_52_l269_269864

theorem least_integer_square_eq_triple_plus_52 :
  ∃ x : ℤ, x^2 = 3 * x + 52 ∧ ∀ y : ℤ, y^2 = 3 * y + 52 → y ≥ x :=
begin
  sorry
end

end least_integer_square_eq_triple_plus_52_l269_269864


namespace B_pow_2024_l269_269358

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), 0, -Real.sin (Real.pi / 4)],
    ![0, 1, 0],
    ![Real.sin (Real.pi / 4), 0, Real.cos (Real.pi / 4)]
  ]

theorem B_pow_2024 :
  B ^ 2024 = ![
    ![-1, 0, 0],
    ![0, 1, 0],
    ![0, 0, -1]
  ] :=
by
  sorry

end B_pow_2024_l269_269358


namespace robot_cost_l269_269982

theorem robot_cost (num_friends : ℕ) (total_tax change start_money : ℝ) (h_friends : num_friends = 7) (h_tax : total_tax = 7.22) (h_change : change = 11.53) (h_start : start_money = 80) :
  let spent_money := start_money - change
  let cost_robots := spent_money - total_tax
  let cost_per_robot := cost_robots / num_friends
  cost_per_robot = 8.75 :=
by
  sorry

end robot_cost_l269_269982


namespace value_of_v_1_at_10_l269_269862

-- Define the polynomial
def polynomial (x : ℕ) := 3 * x^4 + 2 * x^2 + x + 4

-- Define the nested form of the polynomial
noncomputable def nested_polynomial (x : ℕ) := ((3 * x + 0) * x + 2) * x + 1) * x + 4

-- Define the initial intermediate value
def v_0 := 3

-- Define v_1 in terms of v_0 and x following the algorithm
def v_1 (x : ℕ) := v_0 * x + 0

-- State the theorem to be proved
theorem value_of_v_1_at_10 : v_1 10 = 30 := by
  sorry

end value_of_v_1_at_10_l269_269862


namespace at_least_one_l269_269848

axiom P : Prop  -- person A is an outstanding student
axiom Q : Prop  -- person B is an outstanding student

theorem at_least_one (H : ¬(¬P ∧ ¬Q)) : P ∨ Q :=
sorry

end at_least_one_l269_269848


namespace geo_seq_square_sum_l269_269708

variable {a : ℕ → ℕ}

-- Define the hypothesis for the geometric sequence sum condition
def geo_seq_sum_condition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (finset.sum (finset.range n) a) = 2^n - 1

-- The theorem to be proven
theorem geo_seq_square_sum (h : geo_seq_sum_condition a) : 
  ∀ n, n > 0 → (finset.sum (finset.range n) (λ i, (a i)^2)) = (4^n - 1) / 3 :=
by
  sorry

end geo_seq_square_sum_l269_269708


namespace equal_ratios_l269_269201

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end equal_ratios_l269_269201


namespace point_on_circle_l269_269920

-- Define the center of the circle
def center : ℝ × ℝ := (0, 0)

-- Define the radius of the circle
def radius : ℝ := 5

-- Define the point P
def P : ℝ × ℝ := (4, 3)

-- Define the distance function between two points in ℝ²
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the theorem that point P is on the circle
theorem point_on_circle : distance center P = radius :=
  sorry

end point_on_circle_l269_269920


namespace complex_conjugate_of_z_l269_269784

def i : ℂ := complex.I

def z : ℂ := (2 + i) / (1 + i^2 + i^5)

theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * i :=
by {
    unfold z,
    -- The next few statements can formalize simplifying the denominator, but we skip the detailed proof by putting sorry.
    sorry
}

end complex_conjugate_of_z_l269_269784


namespace variance_of_data_l269_269127

open Real

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem variance_of_data {data : List ℝ} (h_data : data = [4, 6, 3, 7, 5]) :
  variance data = 2 := by
  sorry

end variance_of_data_l269_269127


namespace num_even_perfect_square_factors_l269_269657

def isEven (n : ℕ) : Prop := n % 2 = 0

def isPerfectSquare (n : ℕ) : Prop := ∃ (m : ℕ), m^2 = n

theorem num_even_perfect_square_factors (n : ℕ) (h : n = 2^6 * 3^2 * 7^{10}) :
  ∃ (count : ℕ), count = 36 ∧
  count = (finset.filter (λ x, isEven x ∧ isPerfectSquare x) ((finset.Icc 2 6).product (finset.Icc 0 10)).product (finset.Icc 0 2)).card :=
sorry

end num_even_perfect_square_factors_l269_269657


namespace angle_between_north_and_south_southeast_l269_269178

-- Given a circular floor pattern with 12 equally spaced rays
def num_rays : ℕ := 12
def total_degrees : ℕ := 360

-- Proving each central angle measure
def central_angle_measure : ℕ := total_degrees / num_rays

-- Define rays of interest
def segments_between_rays : ℕ := 5

-- Prove the angle between the rays pointing due North and South-Southeast
theorem angle_between_north_and_south_southeast :
  (segments_between_rays * central_angle_measure) = 150 := by
  sorry

end angle_between_north_and_south_southeast_l269_269178


namespace find_m_l269_269622

theorem find_m (m : ℕ) : (11 - m + 1 = 5) → m = 7 :=
by
  sorry

end find_m_l269_269622


namespace heracles_age_is_10_l269_269977

variable (H : ℕ)

-- Conditions
def audrey_age_now : ℕ := H + 7
def audrey_age_in_3_years : ℕ := audrey_age_now + 3
def heracles_twice_age : ℕ := 2 * H

-- Proof Statement
theorem heracles_age_is_10 (h1 : audrey_age_in_3_years = heracles_twice_age) : H = 10 :=
by 
  sorry

end heracles_age_is_10_l269_269977


namespace total_candidates_l269_269009

variables (C : ℕ) (G : ℕ := 900)
noncomputable def B := C - G
noncomputable def boys_failed := 0.7 * B
noncomputable def girls_failed := 0.68 * G
noncomputable def total_failed := 0.691 * C

theorem total_candidates (h : boys_failed C G + girls_failed G = total_failed C) : C = 2000 :=
  sorry

end total_candidates_l269_269009


namespace total_blue_marbles_l269_269719

def jason_blue_marbles : Nat := 44
def tom_blue_marbles : Nat := 24

theorem total_blue_marbles : jason_blue_marbles + tom_blue_marbles = 68 := by
  sorry

end total_blue_marbles_l269_269719


namespace find_payment_y_l269_269154

variable (X Y : Real)

axiom h1 : X + Y = 570
axiom h2 : X = 1.2 * Y

theorem find_payment_y : Y = 570 / 2.2 := by
  sorry

end find_payment_y_l269_269154


namespace gross_profit_percentage_is_12_l269_269579

-- Define the conditions
def selling_price : ℝ := 28
def wholesale_cost : ℝ := 25

-- Define the gross profit per sleeping bag
def gross_profit_per_bag (selling_price wholesale_cost : ℝ) : ℝ :=
  selling_price - wholesale_cost

-- Define the gross profit percentage
def gross_profit_percentage (gross_profit wholesale_cost : ℝ) : ℝ :=
  (gross_profit / wholesale_cost) * 100

-- Example: Proof statement for the gross profit percentage
theorem gross_profit_percentage_is_12 :
  gross_profit_percentage (gross_profit_per_bag selling_price wholesale_cost) wholesale_cost = 12 :=
by
  sorry -- Proof goes here

end gross_profit_percentage_is_12_l269_269579


namespace shaded_quadrilateral_area_l269_269267

noncomputable def area_of_shaded_quadrilateral : ℝ :=
  let side_lens : List ℝ := [3, 5, 7, 9]
  let total_base: ℝ := side_lens.sum
  let largest_square_height: ℝ := 9
  let height_base_ratio := largest_square_height / total_base
  let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
  let a := heights.get! 0
  let b := heights.get! heights.length - 1
  (largest_square_height * (a + b)) / 2

theorem shaded_quadrilateral_area :
    let side_lens := [3, 5, 7, 9]
    let total_base := side_lens.sum
    let largest_square_height := 9
    let height_base_ratio := largest_square_height / total_base
    let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
    let a := heights.get! 0
    let b := heights.get! heights.length - 1
    (largest_square_height * (a + b)) / 2 = 30.375 :=
by 
  sorry

end shaded_quadrilateral_area_l269_269267


namespace domain_y_eq_l269_269295

variable (f : ℝ → ℝ)
variable (domain_f : Set.Icc (-1 : ℝ) (2 : ℝ))

theorem domain_y_eq : (∀ x, x ∈ Set.Icc (-1) 1 → x ∈ domain_f ∧ -x ∈ domain_f) ↔ (-1 : ℝ) ≤ x ∧ x ≤ 1 := by
  sorry

end domain_y_eq_l269_269295


namespace total_new_games_l269_269731

variable (Katie_new_games friends_new_games : ℕ)
variable (Katie_total_games friends_total_games : ℕ)
variable (Katie_percentage friends_percentage : ℝ)

axiom Katie_condition :
  Katie_new_games = 84 ∧ Katie_percentage = 0.75 ∧ Katie_new_games = Katie_percentage * Katie_total_games

axiom friends_condition :
  friends_new_games = 8 ∧ friends_percentage = 0.10 ∧ friends_new_games = friends_percentage * friends_total_games

theorem total_new_games (h1 : Katie_condition) (h2 : friends_condition) :
  Katie_new_games + friends_new_games = 92 :=
by
  sorry

end total_new_games_l269_269731


namespace choose_13_3_equals_286_l269_269664

theorem choose_13_3_equals_286 : (nat.choose 13 3) = 286 :=
by
  sorry

end choose_13_3_equals_286_l269_269664


namespace chandra_monster_hunt_l269_269226

theorem chandra_monster_hunt :
    let d0 := 2   -- monsters on the first day
    let d1 := 2 * d0   -- monsters on the second day
    let d2 := 2 * d1   -- monsters on the third day
    let d3 := 2 * d2   -- monsters on the fourth day
    let d4 := 2 * d3   -- monsters on the fifth day
in d0 + d1 + d2 + d3 + d4 = 62 := by
  sorry

end chandra_monster_hunt_l269_269226


namespace number_of_discrete_colorings_l269_269048

noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

def discrete_colorings (n : ℕ) : ℕ :=
  1 + n + binom n 2 + binom n 3

theorem number_of_discrete_colorings (S : set (ℝ × ℝ)) (n : ℕ)
  (h1 : S.finite) 
  (h2 : S.card = n)
  (h3 : ∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → p1 ∈ S → p2 ∈ S → p3 ∈ S → ¬ collinear ℝ {p1, p2, p3})
  (h4 : ∀ (p1 p2 p3 p4 : (ℝ × ℝ)), p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → p3 ≠ p4 → p1 ∈ S → p2 ∈ S → p3 ∈ S → p4 ∈ S → ¬ concyclic {p1, p2, p3, p4}) :
  discrete_colorings n = 1 + n + nat.choose n 2 + nat.choose n 3 :=
sorry

end number_of_discrete_colorings_l269_269048


namespace speed_of_B_l269_269883

theorem speed_of_B (h1 : ∃ (distance_A : ℝ), distance_A = 5 * 0.5)
                   (h2 : ∃ (time_B : ℝ), time_B = 1 + (48 / 60))
                   (h3 : ∃ (distance_A_1_8 : ℝ), distance_A_1_8 = 5 * 1.8) :
                   ∃ (speed_B : ℝ), speed_B ≈ 6.39 :=
by
  have distance_A := 5 * 0.5
  have time_B := 1 + (48 / 60)
  have distance_A_1_8 := 5 * 1.8
  have distance_B := distance_A_1_8 + distance_A
  have speed_B := distance_B / time_B
  exact ⟨speed_B, sorry⟩

end speed_of_B_l269_269883


namespace probability_abs_x_leq_1_l269_269626

theorem probability_abs_x_leq_1 (x : ℝ) (hx : x ∈ Icc (-1 : ℝ) 2) :
  (measure_theory.measure_of_interval (Icc (-1 : ℝ) 1)) / (measure_theory.measure_of_interval (Icc (-1 : ℝ) 2)) = 2 / 3 :=
by
  sorry

end probability_abs_x_leq_1_l269_269626


namespace carla_cream_volume_l269_269545

-- Definitions of the given conditions and problem
def watermelon_puree_volume : ℕ := 500
def servings_count : ℕ := 4
def volume_per_serving : ℕ := 150
def total_smoothies_volume := servings_count * volume_per_serving
def cream_volume := total_smoothies_volume - watermelon_puree_volume

-- Statement of the proposition we want to prove
theorem carla_cream_volume : cream_volume = 100 := by
  sorry

end carla_cream_volume_l269_269545


namespace find_four_digit_number_l269_269932

noncomputable def problem : Nat :=
  let a : Nat := 1
  let b : Nat := 9
  let c : Nat := 7
  let d : Nat := 9
  let number : Nat := 1000 * a + 100 * b + 10 * c + d
  number

theorem find_four_digit_number :
  ∃ (a b c d : ℕ),
  a + b + c + d = 26 ∧
  ((b * d) / 10 % 10) = a + c ∧
  ∃ (m : ℕ), 2 ^ m = b * d - c ^ 2 ∧
  1000 * a + 100 * b + 10 * c + d = 1979 :=
by
  use 1, 9, 7, 9
  split
  . exact rfl
  split
  . exact rfl
  split
  . use 5
    exact rfl
  exact rfl
  sorry

end find_four_digit_number_l269_269932


namespace concyclic_centers_of_circumcircles_of_triangles_l269_269084

open EuclideanGeometry

noncomputable def center_of_circumcircle (t : Triangle) : Point := sorry -- Assumed existing definition

theorem concyclic_centers_of_circumcircles_of_triangles (L1 L2 L3 L4 : Line) :
  let t1 := triangle_from_lines L1 L2 L3,
      t2 := triangle_from_lines L1 L2 L4,
      t3 := triangle_from_lines L1 L3 L4,
      t4 := triangle_from_lines L2 L3 L4 in
  are_concyclic (center_of_circumcircle t1) (center_of_circumcircle t2) (center_of_circumcircle t3) (center_of_circumcircle t4) :=
sorry

end concyclic_centers_of_circumcircles_of_triangles_l269_269084


namespace find_distance_of_post_office_from_village_l269_269175

-- Conditions
def rate_to_post_office : ℝ := 12.5
def rate_back_village : ℝ := 2
def total_time : ℝ := 5.8

-- Statement of the theorem
theorem find_distance_of_post_office_from_village (D : ℝ) 
  (travel_time_to : D / rate_to_post_office = D / 12.5) 
  (travel_time_back : D / rate_back_village = D / 2)
  (journey_time_total : D / 12.5 + D / 2 = total_time) : 
  D = 10 := 
sorry

end find_distance_of_post_office_from_village_l269_269175


namespace geometric_series_product_l269_269999

theorem geometric_series_product (y : ℝ) :
  (∑'n : ℕ, (1 / 3 : ℝ) ^ n) * (∑'n : ℕ, (- 1 / 3 : ℝ) ^ n)
  = ∑'n : ℕ, (y⁻¹ : ℝ) ^ n ↔ y = 9 :=
by
  sorry

end geometric_series_product_l269_269999


namespace negation_proposition_l269_269433

theorem negation_proposition : 
  ¬ (∃ x_0 : ℝ, x_0^2 + x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by {
  sorry
}

end negation_proposition_l269_269433


namespace chord_length_of_intersection_l269_269342

noncomputable def line_polar_equation := (ρ θ : ℝ) → ρ * Real.sin (θ + Real.pi / 4) = 2
noncomputable def circle_polar_equation := (ρ : ℝ) → ρ = 4

def line_rect_equation (x y : ℝ) := x + y = 2 * Real.sqrt 2
def circle_rect_equation (x y : ℝ) := x^2 + y^2 = 16

theorem chord_length_of_intersection
  (line : ∀ (ρ θ : ℝ), line_polar_equation ρ θ)
  (circle : ∀ (ρ : ℝ), circle_polar_equation ρ)
  (x y : ℝ) :
  (line_rect_equation x y) ∧ (circle_rect_equation x y) →
  sqrt (4^2 - 2^2) * 2 = 4 * Real.sqrt 3 :=
sorry

end chord_length_of_intersection_l269_269342


namespace radius_of_smaller_cylinder_l269_269498

theorem radius_of_smaller_cylinder :
  let
    diameter_large := 6, 
    height_large := 8,
    height_small := 5,
    volume_large := π * (diameter_large / 2)^2 * height_large,
    volume_small : ℝ → ℝ := λ r, π * r^2 * height_small
  in
    3 * volume_small (Real.sqrt (24 / 5)) = volume_large :=
by
  sorry

end radius_of_smaller_cylinder_l269_269498


namespace T_12_eq_1696_l269_269742

def validString (str: List Bool) : Prop :=
  (∀ (i : Nat), (i + 2 < str.length → (str[i] + str[i+1] + str[i+2] ≥ 1))) ∧
  ((str.length >= 2 → (str[0] ≠ false ∨ str[1] ≠ false)))

def T (n : Nat) : Nat :=
  (finset (ofList (have_le: List Bool ↥ λ str,validString str ∧ str.length = n)) (λ str, 1)).sum

theorem T_12_eq_1696 : T 12 = 1696 := by
  sorry

end T_12_eq_1696_l269_269742


namespace probability_different_colors_l269_269333

theorem probability_different_colors :
  let total_chips := 18
  let blue_chips := 7
  let red_chips := 6
  let yellow_chips := 5
  let prob_first_blue := blue_chips / total_chips
  let prob_first_red := red_chips / total_chips
  let prob_first_yellow := yellow_chips / total_chips
  let prob_second_not_blue := (red_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_red := (blue_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_yellow := (blue_chips + red_chips) / (total_chips - 1)
  (
    prob_first_blue * prob_second_not_blue +
    prob_first_red * prob_second_not_red +
    prob_first_yellow * prob_second_not_yellow
  ) = 122 / 153 :=
by sorry

end probability_different_colors_l269_269333


namespace equal_ratios_l269_269200

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end equal_ratios_l269_269200


namespace common_difference_arithmetic_sequence_l269_269631

noncomputable def sum_of_first_n_terms (n : ℕ) : ℤ := -n^2 + 4 * n

theorem common_difference_arithmetic_sequence :
  ∃ d : ℤ, (∀ n : ℕ, sum_of_first_n_terms n = -n^2 + 4 * n) ∧ 
  d = (sum_of_first_n_terms 2 - sum_of_first_n_terms 1) - 3 :=
begin
  use -2,
  sorry
end

end common_difference_arithmetic_sequence_l269_269631


namespace well_digging_rate_l269_269255

def well_depth : ℝ := 14
def well_diameter : ℝ := 3
def total_cost : ℝ := 1484.40
noncomputable def π_val : ℝ := Real.pi
def well_radius : ℝ := well_diameter / 2
def well_volume : ℝ := π_val * well_radius^2 * well_depth
def rate_per_cubic_meter : ℝ := total_cost / well_volume

theorem well_digging_rate : rate_per_cubic_meter ≈ 15 :=
by
  sorry

end well_digging_rate_l269_269255


namespace sandra_beignets_16_weeks_l269_269406

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end sandra_beignets_16_weeks_l269_269406


namespace sequence_integral_terms_l269_269443

theorem sequence_integral_terms (x : ℕ → ℝ) (h1 : ∀ n, x n ≠ 0)
  (h2 : ∀ n > 2, x n = (x (n - 2) * x (n - 1)) / (2 * x (n - 2) - x (n - 1))) :
  (∀ n, ∃ k : ℤ, x n = k) → x 1 = x 2 :=
by
  sorry

end sequence_integral_terms_l269_269443


namespace mr_lee_broke_even_l269_269382

theorem mr_lee_broke_even (sp1 sp2 : ℝ) (p1_loss2 : ℝ) (c1 c2 : ℝ) (h1 : sp1 = 1.50) (h2 : sp2 = 1.50) 
    (h3 : c1 = sp1 / 1.25) (h4 : c2 = sp2 / 0.8333) (h5 : p1_loss2 = (sp1 - c1) + (sp2 - c2)) : 
  p1_loss2 = 0 :=
by 
  sorry

end mr_lee_broke_even_l269_269382


namespace eccentricity_is_sqrt2_div_2_equation_of_ellipse_l269_269634

-- Definition of the problem conditions
variable (a b c : ℝ) 
variable (a_gt_b : a > b) (b_gt_0 : b > 0) 
variable (x y : ℝ)
variable (eq1 : x^2 / a^2 + y^2 / b^2 = 1)
variable (c_eq : c = real.sqrt(a^2 - b^2))

-- Definitions for the conditions of Part 1
variable (angle_F1AB_90 : ∀ x' y' : ℝ, ∠((0, b), (-c, 0), (x', y')) = 90)

-- Definitions for the conditions of Part 2
variable (F2_eq_2F1 : ∀ x' y': ℝ, (c, -b) = 2 * (x' - c, y'))
variable (F1_dot_AB_eq_32 : ∀ x' y': ℝ, (-c, -b) • (x' - (0, b)) = 3 / 2)

-- Lean statement for Part 1
theorem eccentricity_is_sqrt2_div_2 : 
  ∀ e : ℝ, e = c / a → e = real.sqrt 2 / 2 := by
  sorry

-- Lean statement for Part 2
theorem equation_of_ellipse :
  ∀ (a b : ℝ),
    b^2 = 2 → a^2 = 3 → (∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1) := by 
  sorry

end eccentricity_is_sqrt2_div_2_equation_of_ellipse_l269_269634


namespace problem_solution_l269_269238

noncomputable def valid_tuple (a b c d : ℕ) : Prop :=
  a * b * c * d - 1 ∣ (a + b + c + d)

theorem problem_solution (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  valid_tuple a b c d ↔
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d ∈ {2, 3, 5}) ∨
  (a = 1 ∧ b = 1 ∧ c = 2 ∧ d ∈ {2, 5}) ∨
  (a = 1 ∧ b = 1 ∧ c = 3 ∧ d = 3) ∨
  (a = 1 ∧ b = 2 ∧ c = 2 ∧ d = 2) :=
sorry

end problem_solution_l269_269238


namespace sandra_total_beignets_l269_269398

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end sandra_total_beignets_l269_269398


namespace determine_a_l269_269281

def in_set_M (f : ℝ → ℝ) : Prop :=
  (∃ x, f x = x) ∧ ∀ x, f' x > 1

def g (x a : ℝ) : ℝ := log x + a * x

theorem determine_a (a : ℝ) (hM : in_set_M (λ x, g x a)) : a ≥ 1 :=
sorry

end determine_a_l269_269281


namespace evaluate_expression_l269_269059

theorem evaluate_expression :
  let x := -2016 in
  (|(| x | - x)| - | x |) - x = 4032 :=
by
  let x := -2016
  sorry

end evaluate_expression_l269_269059


namespace probability_of_7_successes_in_7_trials_l269_269901

open Probability

/-- Define the given conditions for the problem -/
def n : ℕ := 7
def k : ℕ := 7
def p : ℚ := 2 / 7

/-- The binomial coefficient and the probability of success in n trials -/
theorem probability_of_7_successes_in_7_trials :
  P(X = k) = (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) :=
by
  have bep_0 : nat.choose 7 7 = 1, from sorry,
  have p_power_k : p ^ k = (2 / 7) ^ 7, from sorry,
  have q_power_rem : (1 - p) ^ (n - k) = 1, from sorry,
  have p_eq_frac : (2 / 7) ^ 7 * 1 = 128 / 823543, from sorry,
  show 1 * (2 / 7) ^ 7 * 1 = 128 / 823543, by sorry

end probability_of_7_successes_in_7_trials_l269_269901


namespace labor_cost_per_minute_l269_269802

theorem labor_cost_per_minute (total_cost parts_count parts_cost_per_unit total_hours: ℤ) (H1: total_cost = 220)
  (H2: parts_count = 2) (H3: parts_cost_per_unit = 20) (H4: total_hours = 6) :
  (total_cost - (parts_count * parts_cost_per_unit)) / (total_hours * 60) = 0.5 :=
by
  sorry

end labor_cost_per_minute_l269_269802


namespace sum_of_digits_l269_269427

def original_sum := 943587 + 329430
def provided_sum := 1412017
def correct_sum_after_change (d e : ℕ) : ℕ := 
  let new_first := if d = 3 then 944587 else 943587
  let new_second := if d = 3 then 429430 else 329430
  new_first + new_second

theorem sum_of_digits (d e : ℕ) : d = 3 ∧ e = 4 → d + e = 7 :=
by
  intros
  exact sorry

end sum_of_digits_l269_269427


namespace bernoulli_trial_probability_7_successes_l269_269895

theorem bernoulli_trial_probability_7_successes :
  let n := 7
  let k := 7
  let p := (2 : ℝ) / 7
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (128 / 823543) :=
by
  sorry

end bernoulli_trial_probability_7_successes_l269_269895


namespace total_wait_time_difference_l269_269852

theorem total_wait_time_difference :
  let kids_swings := 6
  let kids_slide := 4 * kids_swings
  let wait_time_swings := [210, 420, 840] -- in seconds
  let total_wait_time_swings := wait_time_swings.sum
  let wait_time_slide := [45, 90, 180] -- in seconds
  let total_wait_time_slide := wait_time_slide.sum
  let total_wait_time_all_kids_swings := kids_swings * total_wait_time_swings
  let total_wait_time_all_kids_slide := kids_slide * total_wait_time_slide
  let difference := total_wait_time_all_kids_swings - total_wait_time_all_kids_slide
  difference = 1260 := sorry

end total_wait_time_difference_l269_269852


namespace num_2021_tuples_with_3_l269_269560

theorem num_2021_tuples_with_3 :
  (∃ (tuple : Fin 2021 → ℕ+), 3 ∈ (tuple 'range) ∧ ∀ (i : Fin 2020), abs (tuple i.succ - tuple i) ≤ 1) ↔
  (3 ^ 2021 - 2 ^ 2021) :=
sorry

end num_2021_tuples_with_3_l269_269560


namespace coupon_savings_inequalities_l269_269192

variable {P : ℝ} (p : ℝ) (hP : P = 150 + p) (hp_pos : p > 0)
variable (ha : 0.15 * P > 30) (hb : 0.15 * P > 0.20 * p)
variable (cA_saving : ℝ := 0.15 * P)
variable (cB_saving : ℝ := 30)
variable (cC_saving : ℝ := 0.20 * p)

theorem coupon_savings_inequalities (h1 : 0.15 * P - 30 > 0) (h2 : 0.15 * P - 0.20 * (P - 150) > 0) :
  let x := 200
  let y := 600
  y - x = 400 :=
by
  sorry

end coupon_savings_inequalities_l269_269192


namespace complex_conjugate_of_z_l269_269749

-- Define z based on the given expression
noncomputable def z : ℂ := (2 + complex.i) / (1 + complex.i^2 + complex.i^5)

-- State the theorem to prove the complex conjugate of z is 1 + 2i
theorem complex_conjugate_of_z : complex.conj z = 1 + 2 * complex.i :=
sorry

end complex_conjugate_of_z_l269_269749


namespace find_alpha_after_five_operations_l269_269505

def returns_to_starting_point_after_operations (α : Real) (n : Nat) : Prop :=
  (n * α) % 360 = 0

theorem find_alpha_after_five_operations (α : Real) 
  (hα1 : 0 < α)
  (hα2 : α < 180)
  (h_return : returns_to_starting_point_after_operations α 5) :
  α = 72 ∨ α = 144 :=
sorry

end find_alpha_after_five_operations_l269_269505


namespace polar_to_rectangular_l269_269557

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 10) (h_θ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (5, -5 * Real.sqrt 3) :=
by
  rw [h_r, h_θ]
  rw [Real.cos_sub, Real.sin_sub]
  have hπ := Real.pi_pos
  rw [div_eq_inv_mul, Real.sin_nat_mul_pi hπ, Real.cos_nat_mul_pi hπ, add_k_pi_two]

  have h1 : Real.cos (Real.pi - Real.pi / 3) = Real.cos (Real.pi / 3) :=
    by rw [Real.cos_pi_sub]; apply Real.cos_pos_of_mem_Icc

  have h2 : Real.sin (Real.pi - Real.pi / 3) = Real.sin (Real.pi / 3) :=
    by rw [Real.sin_pi_sub]; apply Real.sin_pos_of_mem_Icc

  calc Real.cos (5 * Real.pi / 3) = Real.cos (2 * Real.pi - Real.pi / 3) : by rw [mul_comm 5, Real.mul_div_cancel]; linarith
                          ... = Real.cos (Real.pi / 3) : by rw [Real.cos_sub2pi]; apply h1
                          ... = 1/2 : by simp only [Real.cos_pi_div_three]
  
  have h3 : Real.cos (2 * Real.pi - Real.pi / 3) = Real.cos (Real.pi / 3) :=
    by rw [Real.cos_sub2pi]; apply Real.cos_pos_of_mem_Icc

  have h4 : Real.sin (2 * Real.pi - Real.pi / 3) = - Real.sin (Real.pi / 3) :=
    by rw [Real.sin_sub2pi]; apply Real.sin_neg_of_mem_Icc

  calc Real.sin (5 * Real.pi / 3) = Real.sin (2 * Real.pi - Real.pi / 3) : by rw [mul_comm 5, Real.mul_div_cancel]; linarith
                          ... = - Real.sin (Real.pi / 3) : by rw [Real.sin_sub2pi]; apply h2
                          ... = -Real.sqrt 3/2 : by simp only [Real.sin_pi_div_three]
  
  sorry

end polar_to_rectangular_l269_269557


namespace relationship_between_P_QA_and_P_DI_l269_269956

-- Definitions of the percentage marks in different sections
variables (P_QA P_DI : ℝ)
def P_VA : ℝ := 66

-- Condition: percentage marks in VA equals the average of all three sections
def cond : Prop := P_VA = (P_QA + P_DI + P_VA) / 3

-- Proof problem statement: prove the relationship between percentage marks of QA and DI
theorem relationship_between_P_QA_and_P_DI
  (h : cond P_QA P_DI) :
  P_DI = 132 - P_QA :=
sorry

end relationship_between_P_QA_and_P_DI_l269_269956


namespace find_b_l269_269415

theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20)
  (h3 : (5 + 4 * 83 + 6 * 83^2 + 3 * 83^3 + 7 * 83^4 + 5 * 83^5 + 2 * 83^6 - b) % 17 = 0) :
  b = 8 :=
sorry

end find_b_l269_269415


namespace condition_for_parallel_MN_B1BDD1_l269_269020

structure Point3D (α : Type _) := (x y z : α)

-- Define the vertices of the right prism
variables {α : Type _} [LinearOrderedField α]
def A := (0 : Point3D α) -- replace with actual coordinates
def B := (0 : Point3D α) -- replace with actual coordinates
def C := (0 : Point3D α) -- replace with actual coordinates
def D := (0 : Point3D α) -- replace with actual coordinates
def A1 := (0 : Point3D α) -- replace with actual coordinates
def B1 := (0 : Point3D α) -- replace with actual coordinates
def C1 := (0 : Point3D α) -- replace with actual coordinates
def D1 := (0 : Point3D α) -- replace with actual coordinates

-- Define the midpoints E, F, G, H, N
def midpoint (P Q : Point3D α) : Point3D α :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2, (P.z + Q.z) / 2⟩

def E := midpoint C C1
def F := midpoint C1 D1
def G := midpoint D1 D
def H := midpoint D C
def N := midpoint B C

-- Define the quadrilateral EFGH and point M within or on EFGH
def isInOrOn (P : Point3D α) : Prop := sorry -- geometric condition for being in area EFGH

-- The proof problem
theorem condition_for_parallel_MN_B1BDD1 (M : Point3D α) (hM_inEFGH : isInOrOn M) :
  (∃ λ, M = F + λ • (H - F)) → (subtype (∃ λ, M = F + λ • (H - F))) := sorry

end condition_for_parallel_MN_B1BDD1_l269_269020


namespace red_higher_than_green_l269_269510

-- Define the conditions of the problem
def redBallDistribution (k : ℕ) : ℚ := 3^(-k)
def greenBallDistribution (p : ℕ) (h : Nat.Prime p) : ℚ := 3^(-p)

-- Define the probability that the red ball lands in a higher-numbered bin than a given prime number bin p
def higherNumberedThan (p : ℕ) (h : Nat.Prime p) : ℚ :=
  ∑' (k : ℕ) in ({k : ℕ | k > p}), 3^(-k)

-- Define the main proof problem statement
theorem red_higher_than_green : 
  (∑' (p : ℕ) in ({p : ℕ | Nat.Prime p}), (1 / 2) * 3^(-p)) = (1 / 10) :=
by sorry

end red_higher_than_green_l269_269510


namespace train_length_correct_l269_269519

noncomputable def speed_km_per_hour : ℝ := 56
noncomputable def time_seconds : ℝ := 32.142857142857146
noncomputable def bridge_length_m : ℝ := 140
noncomputable def train_length_m : ℝ := 360

noncomputable def speed_m_per_s : ℝ := speed_km_per_hour * (1000 / 3600)
noncomputable def total_distance_m : ℝ := speed_m_per_s * time_seconds

theorem train_length_correct :
  (total_distance_m - bridge_length_m) = train_length_m :=
  by
    sorry

end train_length_correct_l269_269519


namespace prop_a_range_l269_269286

variables {a : ℝ} {x : ℝ}

def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * a * x + 7 * a - 6 > 0

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a * x + 4 < 0

theorem prop_a_range (a : ℝ) :
  (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a) ↔ (a ∈ set.Ioo (-∞) (-4) ∪ set.Ico 1 4 ∪ set.Ico 6 ∞) := by
  sorry

end prop_a_range_l269_269286


namespace rectangular_solid_length_l269_269509

theorem rectangular_solid_length (w h : ℕ) (surface_area : ℕ) (l : ℕ) 
  (hw : w = 4) (hh : h = 1) (hsa : surface_area = 58) 
  (h_surface_area_formula : surface_area = 2 * l * w + 2 * l * h + 2 * w * h) : 
  l = 5 :=
by
  rw [hw, hh, hsa] at h_surface_area_formula
  sorry

end rectangular_solid_length_l269_269509


namespace smallest_sum_l269_269598

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l269_269598


namespace eq_from_conditions_l269_269442

theorem eq_from_conditions (a b : ℂ) :
  (1 / (a + b)) ^ 2003 = 1 ∧ (-a + b) ^ 2005 = 1 → a ^ 2003 + b ^ 2004 = 1 := 
by
  sorry

end eq_from_conditions_l269_269442


namespace PQDE_concyclic_l269_269971

open Classical

variables {A B C D E P Q : Point} (circle : Circle)

-- Conditions
def is_inscribed_tri (triangle : Triangle) := 
  triangle.inscribed_in circle

def is_isosceles (triangle : Triangle) := 
  triangle.AB = triangle.AC

def AE_is_chord := ∃ (E : Point), E ∈ circle ∧ A ≠ E
def AQ_is_chord := ∃ (Q : Point), Q ∈ circle ∧ A ≠ Q

def AE_intersects_BC_at_D :=
  AE_is_chord ∧ ∃ (D : Point), D ∈ BC ∨ D ∈ extension BC

def AQ_intersects_CB_at_P :=
  AQ_is_chord ∧ ∃ (P : Point), P ∈ extension CB

-- Question (Prove that P, Q, D, E are concyclic)
theorem PQDE_concyclic 
  (triangle : Triangle)
  [is_inscribed_tri triangle]
  [is_isosceles triangle]
  [AE_is_chord]
  [AQ_is_chord]
  [AE_intersects_BC_at_D]
  [AQ_intersects_CB_at_P] :
  CyclicOrder.circle4 A B C D E P Q :=
sorry

end PQDE_concyclic_l269_269971


namespace Nina_homework_total_l269_269383

theorem Nina_homework_total :
  let math_ruby := 40
      read_ruby := 20
      sci_ruby := 10
      math_nina := math_ruby + (50 / 100) * math_ruby
      read_nina := read_ruby + (25 / 100) * read_ruby
      sci_nina := sci_ruby + (150 / 100) * sci_ruby in
  (math_nina + read_nina + sci_nina) = 110 :=
by
  let math_ruby := 40
  let read_ruby := 20
  let sci_ruby := 10
  let math_nina := math_ruby + (50 / 100 : ℚ) * math_ruby
  let read_nina := read_ruby + (25 / 100 : ℚ) * read_ruby
  let sci_nina := sci_ruby + (150 / 100 : ℚ) * sci_ruby
  have h_math : math_nina = 60 := by sorry
  have h_read : read_nina = 25 := by sorry
  have h_sci : sci_nina = 25 := by sorry
  show (math_nina + read_nina + sci_nina) = 110 from by
    rw [h_math, h_read, h_sci]
    exact rfl

end Nina_homework_total_l269_269383


namespace count_total_shells_l269_269102

theorem count_total_shells 
  (purple_shells : ℕ := 13)
  (pink_shells : ℕ := 8)
  (yellow_shells : ℕ := 18)
  (blue_shells : ℕ := 12)
  (orange_shells : ℕ := 14) :
  purple_shells + pink_shells + yellow_shells + blue_shells + orange_shells = 65 :=
by
  -- Calculation
  sorry

end count_total_shells_l269_269102


namespace all_or_none_triangular_lines_l269_269416

noncomputable def curve (p q r s : ℝ) : ℝ → ℝ := 
  λ x, x^4 + p * x^3 + q * x^2 + r * x + s

def is_horizontal_line (y₀ : ℝ) := ∀ x : ℝ, x ∈ {x : ℝ | curve p q r s x = y₀}

def is_triangular_line (x₁ x₂ x₃ x₄ : ℝ) :=
  x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
  ((x₂ - x₁) + (x₃ - x₁) > (x₄ - x₁) ∧ (x₂ - x₁) > (x₄ - x₃))

theorem all_or_none_triangular_lines (p q r s : ℝ) :
  (∀ y₀ : ℝ, (∃ x₁ x₂ x₃ x₄ : ℝ, is_horizontal_line y₀ ∧ is_triangular_line x₁ x₂ x₃ x₄)) ∨ 
  (∀ y₀ : ℝ, ¬(∃ x₁ x₂ x₃ x₄ : ℝ, is_horizontal_line y₀ ∧ is_triangular_line x₁ x₂ x₃ x₄)) := 
by
  sorry

end all_or_none_triangular_lines_l269_269416


namespace v3_at_2_is_15_l269_269466

-- Define the polynomial f(x)
def f (x : ℝ) := x^4 + 2*x^3 + x^2 - 3*x - 1

-- Define v3 using Horner's Rule at x
def v3 (x : ℝ) := ((x + 2) * x + 1) * x - 3

-- Prove that v3 at x = 2 equals 15
theorem v3_at_2_is_15 : v3 2 = 15 :=
by
  -- Skipping the proof with sorry
  sorry

end v3_at_2_is_15_l269_269466


namespace intersections_of_lines_l269_269070

theorem intersections_of_lines : 
  let lines := (λ m b, λ x y, y = m * x + b) in
  let m_values := [1, -2] in
  let b_values := [0, 1, 2] in
  let distinct_points_count (points : List (ℝ × ℝ)) :=
    points.toFinset.card
  let points := 
    [for m1 in m_values, b1 in b_values, m2 in m_values, b2 in b_values, h : m1 ≠ m2 
      yield let ⟨x, y⟩ := ((b2 - b1) / (m1 - m2), m1 * (b2 - b1) / (m1 - m2) + b1) in (x, y)] 
  in distinct_points_count points = 9 :=
by
  sorry

end intersections_of_lines_l269_269070


namespace total_money_is_correct_l269_269072

variable (mark : ℚ) (carolyn : ℚ) (jim : ℚ)

theorem total_money_is_correct 
  (h_mark : mark = 5 / 6) 
  (h_carolyn : carolyn = 3 / 10) 
  (h_jim : jim = 1 / 2) :
  (mark + carolyn + jim).toReal = 1.63 :=
by
  sorry

end total_money_is_correct_l269_269072


namespace direction_vector_prop_l269_269318

theorem direction_vector_prop (a : ℚ) :
  let p₁ := (-4, 3 : ℚ)
  let p₂ := (2, -2 : ℚ)
  let d₁ := (2 - (-4), -2 - 3 : ℚ)
  let d₂ := (a, -1 : ℚ)
  (∃ k : ℚ, k • d₁ = d₂) → a = 6 / 5 :=
by
  intros
  let p₁ := (-4, 3 : ℚ)
  let p₂ := (2, -2 : ℚ)
  let d₁ := (2 - (-4), -2 - 3 : ℚ)
  let d₂ := (a, -1 : ℚ)
  have k_exists : ∃ k : ℚ, k • d₁ = d₂ := sorry
  let k := 1 / 5
  have : k • d₁ = (6 / 5, -1 : ℚ) := sorry
  exact sorry

end direction_vector_prop_l269_269318


namespace caffeine_over_goal_l269_269450

theorem caffeine_over_goal (cups_per_day : ℕ) (mg_per_cup : ℕ) (caffeine_goal : ℕ) (total_cups : ℕ) :
  total_cups = 3 ->
  cups_per_day = 3 ->
  mg_per_cup = 80 ->
  caffeine_goal = 200 ->
  (cups_per_day * mg_per_cup) - caffeine_goal = 40 := by
  sorry

end caffeine_over_goal_l269_269450


namespace centroid_of_triangle_l269_269651

-- Define the vectors as pairs of real numbers
def OA : ℝ × ℝ := (-1, 3)
def OB : ℝ × ℝ := (1, 2)
def OC : ℝ × ℝ := (2, -5)

-- The expected coordinates of centroid G
def OG_expected : ℝ × ℝ := (2 / 3, 0)

-- Define the centroid formula for the given vectors
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem centroid_of_triangle :
  centroid OA OB OC = OG_expected :=
by
  -- Proof is not required as per the instruction
  sorry

end centroid_of_triangle_l269_269651


namespace more_red_than_yellow_l269_269726

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end more_red_than_yellow_l269_269726


namespace apartments_with_one_resident_l269_269004

theorem apartments_with_one_resident (total_apartments : ℕ) (pct_at_least_one : ℝ) (pct_at_least_two : ℝ)
  (h1 : total_apartments = 120)
  (h2 : pct_at_least_one = 0.85)
  (h3 : pct_at_least_two = 0.60) :
  let at_least_one := pct_at_least_one * total_apartments,
      at_least_two := pct_at_least_two * total_apartments in
  (at_least_one - at_least_two) = 30 :=
by
  let at_least_one := 0.85 * 120
  let at_least_two := 0.60 * 120
  have h4 : at_least_one = 102 := by norm_num
  have h5 : at_least_two = 72 := by norm_num
  have h6 : 102 - 72 = 30 := by norm_num
  exact h6

end apartments_with_one_resident_l269_269004


namespace arithmetic_example_l269_269223

theorem arithmetic_example : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end arithmetic_example_l269_269223


namespace sum_of_roots_Q_l269_269440

noncomputable def Q (φ : ℝ) : Polynomial ℂ := 
  Polynomial.monic (Polynomial.C ℂ) 4 -- Placeholder for the monic quartic polynomial

def cos_phi (φ : ℝ) : ℂ := complex.cos φ
def sin_phi (φ : ℝ) : ℂ := complex.sin φ
def roots (φ : ℝ) : set ℂ := { cos_phi φ + sin_phi φ * complex.I, cos_phi φ - sin_phi φ * complex.I, 
                                -cos_phi φ + sin_phi φ * complex.I, -cos_phi φ - sin_phi φ * complex.I }

def area (φ : ℝ) : ℂ := 4 * abs (complex.cos φ * complex.sin φ)

lemma correct_area (φ : ℝ) (hφ : 0 < φ ∧ φ < π / 6) : area φ = 2 * abs (complex.sin (2 * φ)) :=
by
  sorry -- Detailed proof omitted

theorem sum_of_roots_Q (φ : ℝ) (hφ: 0 < φ ∧ φ < π / 6) (harea: area φ = (3 / 4) * Q φ 0) : 
  (polynomial.roots (Q φ)).sum = 4 * cos_phi φ :=
by 
  sorry -- Detailed proof omitted

end sum_of_roots_Q_l269_269440


namespace carla_smoothies_serving_l269_269992

theorem carla_smoothies_serving :
  ∀ (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ),
  watermelon_puree = 500 → cream = 100 → serving_size = 150 →
  (watermelon_puree + cream) / serving_size = 4 :=
by
  intros watermelon_puree cream serving_size
  intro h1 -- watermelon_puree = 500
  intro h2 -- cream = 100
  intro h3 -- serving_size = 150
  sorry

end carla_smoothies_serving_l269_269992


namespace total_money_spent_l269_269935

def total_cost (blades_cost : Nat) (string_cost : Nat) : Nat :=
  blades_cost + string_cost

theorem total_money_spent 
  (num_blades : Nat)
  (cost_per_blade : Nat)
  (string_cost : Nat)
  (h1 : num_blades = 4)
  (h2 : cost_per_blade = 8)
  (h3 : string_cost = 7) :
  total_cost (num_blades * cost_per_blade) string_cost = 39 :=
by
  sorry

end total_money_spent_l269_269935


namespace heracles_age_l269_269975

theorem heracles_age
  (H : ℕ)
  (audrey_current_age : ℕ)
  (audrey_in_3_years : ℕ)
  (h1 : audrey_current_age = H + 7)
  (h2 : audrey_in_3_years = audrey_current_age + 3)
  (h3 : audrey_in_3_years = 2 * H)
  : H = 10 :=
by
  sorry

end heracles_age_l269_269975


namespace smallest_possible_sum_l269_269607

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l269_269607


namespace line_intersects_circle_l269_269306

open Real

def parametric_line (t : ℝ) : ℝ × ℝ :=
  (2 * t, 1 + 4 * t)

def polar_circle (theta : ℝ) : ℝ :=
  2 * sqrt 2 * sin theta

def cartesian_line_equation (x y : ℝ) : Prop :=
  2 * x - y + 1 = 0

def cartesian_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * sqrt 2 * y

theorem line_intersects_circle :
  (∀ t, ∃ x y, parametric_line t = (x, y) ∧ cartesian_line_equation x y) ∧
  (∀ theta, ∃ x y, polar_circle theta = dist (0, 0) (x, y) ∧ cartesian_circle_equation x y) →
  (∃ x y, cartesian_line_equation x y ∧ cartesian_circle_equation x y) :=
by sorry

end line_intersects_circle_l269_269306


namespace Jill_ball_difference_l269_269721

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end Jill_ball_difference_l269_269721


namespace sum_of_valid_x_values_l269_269211

def is_valid_x_y (x y : ℕ) : Prop := (x * y = 360) ∧ (x ≥ 12) ∧ (y ≥ 15)

def valid_x_values (s : ℕ → ℕ → Prop) : list ℕ :=
  (list.range 361).filter (λ x, ∃ y, s x y)

theorem sum_of_valid_x_values : list.sum (valid_x_values is_valid_x_y) = 89 :=
  sorry

end sum_of_valid_x_values_l269_269211


namespace greater_number_l269_269849

theorem greater_number (x y : ℕ) (h1 : x + y = 22) (h2 : x - y = 4) : x = 13 := 
by sorry

end greater_number_l269_269849


namespace increasing_sequence_k_range_l269_269284

theorem increasing_sequence_k_range (k : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = n^2 + k * n) :
  (∀ n : ℕ, a (n + 1) > a n) → (k ≥ -3) :=
  sorry

end increasing_sequence_k_range_l269_269284


namespace plan_A_charge_for_first_6_minutes_l269_269176

theorem plan_A_charge_for_first_6_minutes 
    (x : ℝ)
    (charge_plan_A_first_6_minutes : x)
    (charge_plan_A_after_6_minutes : ∀ m, m > 6 → ∀ minutes,  charge_plan_A_first_6_minutes + (minutes - 6) * 0.06)
    (charge_plan_B : ∀ minutes, 0.08 * minutes)
    (same_charge_condition : ∀ minutes, minutes = 12 → charge_plan_A_first_6_minutes + (minutes - 6) * 0.06 = 0.08 * minutes):
  charge_plan_A_first_6_minutes = 0.60 :=
by
  sorry

end plan_A_charge_for_first_6_minutes_l269_269176


namespace sum_of_ten_distinct_numbers_lt_75_l269_269389

theorem sum_of_ten_distinct_numbers_lt_75 :
  ∃ (S : Finset ℕ), S.card = 10 ∧
  (∃ (S_div_5 : Finset ℕ), S_div_5 ⊆ S ∧ S_div_5.card = 3 ∧ ∀ x ∈ S_div_5, 5 ∣ x) ∧
  (∃ (S_div_4 : Finset ℕ), S_div_4 ⊆ S ∧ S_div_4.card = 4 ∧ ∀ x ∈ S_div_4, 4 ∣ x) ∧
  S.sum id < 75 :=
by { 
  sorry 
}

end sum_of_ten_distinct_numbers_lt_75_l269_269389


namespace smallest_a_exists_l269_269744

noncomputable theory
open polynomial

variables (P : polynomial ℤ) (a : ℤ)

def satisfies_conditions (P : polynomial ℤ) (a : ℤ) :=
  a > 0 ∧ P.eval 1 = a ∧ P.eval 3 = a ∧ P.eval 5 = a ∧ P.eval 7 = a ∧
  P.eval 2 = -a ∧ P.eval 4 = -a ∧ P.eval 6 = -a ∧ P.eval 8 = -a ∧
  P.eval 0 = 0 ∧ P.eval 9 = a

theorem smallest_a_exists :
  ∃ (a : ℤ) (P : polynomial ℤ),
  satisfies_conditions P a ∧ a = 945 :=
sorry

end smallest_a_exists_l269_269744


namespace ratio_x_y_l269_269198

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end ratio_x_y_l269_269198


namespace negation_of_exists_l269_269434

theorem negation_of_exists (x : ℝ) : 
  ¬ (∃ x : ℝ, 2 * x^2 + 2 * x - 1 ≤ 0) ↔ ∀ x : ℝ, 2 * x^2 + 2 * x - 1 > 0 :=
by
  sorry

end negation_of_exists_l269_269434


namespace CorrectCircleStatementD_l269_269880

theorem CorrectCircleStatementD : 
  (∀ (c : Circle), ∀ (d : Diameter c), IsSymmetryAxis (line_through_diameter d)) :=
by
  sorry

end CorrectCircleStatementD_l269_269880


namespace movie_theater_revenue_l269_269184

theorem movie_theater_revenue 
  (total_tickets : ℕ) 
  (adult_tickets : ℕ) 
  (child_tickets : ℕ) 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_sold : total_tickets = 900)
  (adult_sold : adult_tickets = 500) 
  (child_sold : child_tickets = total_tickets - adult_tickets)
  (adult_price_set : adult_price = 7) 
  (child_price_set : child_price = 4) :
  (adult_tickets * adult_price + child_tickets * child_price = 5100) :=
by
  rw [adult_sold, total_sold, adult_price_set, child_price_set, child_sold]
  norm_num
  -- simplifies the goal to 500 * 7 + 400 * 4 = 5100 and proves it
  sorry

end movie_theater_revenue_l269_269184


namespace combined_area_of_sectors_l269_269253

noncomputable def area_of_sector (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * π * r ^ 2

theorem combined_area_of_sectors :
  area_of_sector 10 42 + area_of_sector 15 60 = 154.46 * π := by
    sorry

end combined_area_of_sectors_l269_269253


namespace number_of_valid_sequences_l269_269230

-- Define the dihedral group D4 transformations
constant L : ℕ
constant R : ℕ
constant H : ℕ
constant V : ℕ
constant I : ℕ

-- Assume transformations form dihedral group D4
axiom L_property1 : L ^ 4 = I
axiom R_property1 : R ^ 4 = I
axiom H_property1 : H ^ 2 = I
axiom V_property1 : V ^ 2 = I
axiom LR_inverse : L * R = I
axiom RL_inverse : R * L = I
axiom HH_inverse : H * H = I
axiom VV_inverse : V * V = I

-- Definitions of sequence and identity condition
def valid_sequence (seq : ℕ → ℕ) : Prop := (∀ n, seq n ∈ {L, R, H, V}) ∧ (∏ i in finset.range 22, seq i) = I

-- Define the main theorem statement
theorem number_of_valid_sequences : {seq : ℕ → ℕ // valid_sequence seq}.card = 4 ^ 21 := 
by 
  sorry

end number_of_valid_sequences_l269_269230


namespace triangle_equilateral_l269_269391

noncomputable def is_equilateral {R p : ℝ} (A B C : ℝ) : Prop :=
  R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p  →
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a

theorem triangle_equilateral
  {A B C : ℝ}
  {R p : ℝ}
  (h : R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p) :
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a :=
sorry

end triangle_equilateral_l269_269391


namespace curve_equation_lambda_mu_squared_is_constant_l269_269285

variables (b x y λ μ : ℝ) (x1 y1 x2 y2 : ℝ)
variables (O P A B : ℝ×ℝ)
variables (M : ℝ×ℝ := (-sqrt 3 * b, 0))
variables (N : ℝ×ℝ := (sqrt 3 * b, 0))

noncomputable def curve_C (P : ℝ × ℝ) := (P.1 ^ 2 + 3 * P.2 ^ 2 = 3 * b ^ 2)

--- Problem 1: Equation of curve C
theorem curve_equation (P : ℝ × ℝ) (hP : curve_C b P) :
  (P.1 ^ 2 + 3 * P.2 ^ 2 = 3 * b ^ 2) :=
begin
  sorry
end

variables (l : ℝ → ℝ) (h_l : l x = x - sqrt 2 * b)

--- Problem 2: Lambda squared + Mu squared is constant
theorem lambda_mu_squared_is_constant (P A B : ℝ × ℝ)
  (hA_curve : curve_C b A) (hB_curve : curve_C b B)
  (h_intersect_A : l A.1 = A.2) (h_intersect_B : l B.1 = B.2) 
  (hP : O.1 * λ + A.1 * μ = x) (h2P : O.2 * λ + A.2 * μ = y)
  ( h_x : (x = λ * A.1 + μ * B.1)) ( h_y : ( y = λ * A.2 + μ * B.2)) :
  λ^2 + μ^2 = 1 :=
begin
  sorry
end

end curve_equation_lambda_mu_squared_is_constant_l269_269285


namespace max_a_for_inequality_l269_269272

open Real

theorem max_a_for_inequality (a : ℝ) : (∀ x : ℝ, 0 ≤ x → exp x + sin x - 2 * x ≥ a * x^2 + 1) → a ≤ 1 / 2 :=
by
  sorry

end max_a_for_inequality_l269_269272


namespace tangent_at_P_bisects_BC_l269_269493

noncomputable def inscribe_circle_in_triangle (A B C P M K : Point) (circ : Circle) : Prop :=
  inscribed_in_triangle circ A B C ∧
  point_of_tangency circ B C M ∧
  diameter circ M K ∧
  line_intersects_circle A K circ P

theorem tangent_at_P_bisects_BC
  (A B C P M K L : Point)
  (circ : Circle)
  (h1 : inscribe_circle_in_triangle A B C P M K circ)
  (tangent_at_P : is_tangent_at circ P L) :
  bisects L B C :=
sorry

end tangent_at_P_bisects_BC_l269_269493


namespace cycle_length_at_least_M_plus_N_div_2_l269_269385

-- Definitions corresponding to the conditions of the problem
variables {M N : ℕ}
variables {Countries : Type} [Fintype Countries]
variables {Cities : Type} [Fintype Cities]
variables {roads : Cities → Cities → Prop} [Symmetric roads]

-- Condition (1): In each country, there are at least three cities.
variable (atLeastThreeCities : ∀ (c : Countries), Fintype.card (Subtype (λ x, x ∈ c)) ≥ 3)

-- Condition (2): Every city in a country is connected to at least half of the other cities in that country.
variable (halfConnected : ∀ (c : Countries) (x : Cities) (hx : x ∈ c),
  Fintype.card (Subtype (λ y, roads x y ∧ y ∈ c)) ≥ Fintype.card (Subtype (λ y, y ∈ c)) / 2)

-- Condition (3): Each city is connected with exactly one other city that is not in its country.
variable (externalConnection : ∀ (x : Cities), ∃! y : Cities, ¬∃ (c : Countries), x ∈ c ∧ y ∈ c ∧ roads x y)

-- Condition (4): There are at most two roads between cities in different countries.
variable (atMostTwoRoadsBetweenCountries : ∀ (x y : Cities), ∃ (c₁ c₂ : Countries), ¬(c₁ = c₂) → ¬(roads x y))

-- Condition (5): If two countries together have less than 2M cities, there is at least one road between them.
variable (roadBetweenSmallCountries : ∀ (c₁ c₂ : Countries), Fintype.card (Subtype (λ x, x ∈ c₁)) + Fintype.card (Subtype (λ x, x ∈ c₂)) < 2 * M → 
  ∃ (x ∈ c₁) (y ∈ c₂), roads x y)

-- The main statement to be proven
theorem cycle_length_at_least_M_plus_N_div_2 :
  ∃ (cycle : List Cities), cycle.Nodup ∧  ∃ subcycle, subcycle.length ≥ M + N / 2 :=
sorry

end cycle_length_at_least_M_plus_N_div_2_l269_269385


namespace not_possible_placement_l269_269347
open Nat

def adjacent (E : List (ℕ × ℕ)) (u v : ℕ) : Prop := 
  (u, v) ∈ E ∨ (v, u) ∈ E

def non_adjacent (E : List (ℕ × ℕ)) (u v : ℕ) : Prop := 
  ¬ adjacent E u v

theorem not_possible_placement (V : Fin₈ → ℕ) (E : List (ℕ × ℕ)) 
  (hV : ∀ i, 1 ≤ V i ∧ V i ≤ 220) 
  (h_adjacent : ∀ u v, adjacent E u v → gcd (V u) (V v) > 1) 
  (h_non_adjacent : ∀ u v, non_adjacent E u v → gcd (V u) (V v) = 1) : 
  False :=
sorry

end not_possible_placement_l269_269347


namespace lisa_caffeine_l269_269452

theorem lisa_caffeine (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_drank : ℕ) : caffeine_per_cup = 80 → daily_goal = 200 → cups_drank = 3 → (caffeine_per_cup * cups_drank - daily_goal) = 40 :=
by
  -- This is a theorem statement, thus no proof is provided here.
  sorry

end lisa_caffeine_l269_269452


namespace rationalize_denominator_l269_269393

theorem rationalize_denominator :
  ∃ (A B C D : ℤ), D > 0 ∧ (∀ p : ℕ, prime p → p^2 ∣ B → false) ∧
  A + B + C + D = 9 ∧ 
  (4*Real.sqrt 3) / (3*Real.sqrt 2 - 2*Real.sqrt 2) = (A*Real.sqrt B + C) / D :=
sorry

end rationalize_denominator_l269_269393


namespace probability_of_red_and_blue_or_blue_and_green_is_four_ninths_l269_269449

-- Define the conditions of the problem
def total_chips := 12
def red_chips := 6
def blue_chips := 4
def green_chips := 2

-- Define the probability of drawing each specific pair of chips
def P_red_blue : ℚ := (red_chips / total_chips) * (blue_chips / total_chips)
def P_blue_red : ℚ := (blue_chips / total_chips) * (red_chips / total_chips)
def P_blue_green : ℚ := (blue_chips / total_chips) * (green_chips / total_chips)
def P_green_blue : ℚ := (green_chips / total_chips) * (blue_chips / total_chips)

-- Add these probabilities together
def P_total : ℚ := P_red_blue + P_blue_red + P_blue_green + P_green_blue

-- Statement of the problem to prove
theorem probability_of_red_and_blue_or_blue_and_green_is_four_ninths : P_total = 4 / 9 :=
by sorry

end probability_of_red_and_blue_or_blue_and_green_is_four_ninths_l269_269449


namespace heracles_age_is_10_l269_269976

variable (H : ℕ)

-- Conditions
def audrey_age_now : ℕ := H + 7
def audrey_age_in_3_years : ℕ := audrey_age_now + 3
def heracles_twice_age : ℕ := 2 * H

-- Proof Statement
theorem heracles_age_is_10 (h1 : audrey_age_in_3_years = heracles_twice_age) : H = 10 :=
by 
  sorry

end heracles_age_is_10_l269_269976


namespace impossible_cube_vertex_assignment_l269_269349

-- Define the problem and required conditions
def vertex_number_condition (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 220

def adjacent_vertices_condition (n_i n_j : ℕ) : Prop :=
  nat.gcd n_i n_j > 1

def nonadjacent_vertices_condition (n_i n_j : ℕ) : Prop :=
  nat.gcd n_i n_j = 1

def cube_vertices_conditions(numbers : Fin 8 → ℕ) : Prop :=
  (∀ i, vertex_number_condition (numbers i)) ∧
  (∀ i j, i ≠ j → numbers i ≠ numbers j) ∧
  (∀ i j, adjacent_edges i j → adjacent_vertices_condition (numbers i) (numbers j)) ∧
  (∀ i j, non_adjacent_edges i j → nonadjacent_vertices_condition (numbers i) (numbers j))

-- Define adjacent and non-adjacent vertex relations (to be defined based on cube vertex adjacency)
def adjacent_edges(i j : Fin 8) : Prop :=
  -- Example adjacency relations; to be replaced with actual adjacency relations for cube vertices
  sorry

def non_adjacent_edges(i j : Fin 8) : Prop :=
  -- Example non-adjacency relations; to be replaced with actual non-adjacency relations for cube vertices
  sorry

-- Main theorem statement
theorem impossible_cube_vertex_assignment :
  ¬(∃ numbers : Fin 8 → ℕ, cube_vertices_conditions numbers) :=
by
  -- Proof omitted
  sorry


end impossible_cube_vertex_assignment_l269_269349


namespace conjugate_z_is_1_add_2i_l269_269757

open Complex

def z_def : ℂ := ((2 : ℂ) + I) / (1 + I^2 + I^5)

theorem conjugate_z_is_1_add_2i : conj z_def = (1 : ℂ) + 2 * I := 
sorry

end conjugate_z_is_1_add_2i_l269_269757


namespace equal_ratios_l269_269202

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end equal_ratios_l269_269202


namespace child_current_height_l269_269943

variable (h_last_visit : ℝ) (h_grown : ℝ)

-- Conditions
def last_height (h_last_visit : ℝ) := h_last_visit = 38.5
def height_grown (h_grown : ℝ) := h_grown = 3

-- Theorem statement
theorem child_current_height (h_last_visit h_grown : ℝ) 
    (h_last : last_height h_last_visit) 
    (h_grow : height_grown h_grown) : 
    h_last_visit + h_grown = 41.5 :=
by
  sorry

end child_current_height_l269_269943


namespace bernoulli_trial_probability_7_successes_l269_269893

theorem bernoulli_trial_probability_7_successes :
  let n := 7
  let k := 7
  let p := (2 : ℝ) / 7
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (128 / 823543) :=
by
  sorry

end bernoulli_trial_probability_7_successes_l269_269893


namespace celine_erasers_collected_l269_269224

theorem celine_erasers_collected (G C J E : ℕ) 
    (hC : C = 2 * G)
    (hJ : J = 4 * G)
    (hE : E = 12 * G)
    (h_total : G + C + J + E = 151) : 
    C = 16 := 
by 
  -- Proof steps skipped, proof body not required as per instructions
  sorry

end celine_erasers_collected_l269_269224


namespace purple_flowers_killed_by_fungus_l269_269180

def initial_flower_count (color: String) : Nat :=
  if color = "red" then 125 
  else if color = "yellow" then 125 
  else if color = "orange" then 125 
  else if color = "purple" then 125 
  else 0

def flowers_killed_by_fungus (color: String) : Nat :=
  if color = "red" then 45 
  else if color = "yellow" then 61 
  else if color = "orange" then 30 
  else if color = "purple" then sorry -- This is the unknown we need to find.
  else 0

def flowers_left (color: String) : Nat :=
  initial_flower_count color - flowers_killed_by_fungus color

def total_flowers_needed : Nat := 36 * 9  -- 36 bouquets each with 9 flowers

def total_non_purple_flowers_left : Nat :=
  flowers_left "red" + flowers_left "yellow" + flowers_left "orange"

def purple_killed_by_fungus : Nat :=
  initial_flower_count "purple" - (total_flowers_needed - total_non_purple_flowers_left)

theorem purple_flowers_killed_by_fungus : purple_killed_by_fungus = 40 :=
by
  unfold initial_flower_count flowers_killed_by_fungus flowers_left total_flowers_needed total_non_purple_flowers_left purple_killed_by_fungus
  sorry

end purple_flowers_killed_by_fungus_l269_269180


namespace required_run_rate_correct_l269_269702

def cricket_run_rate (run_rate_initial : ℝ) (overs_initial : ℕ) (target_runs : ℝ) (remaining_overs : ℕ) : ℝ :=
  let scored_initial := run_rate_initial * overs_initial
  let needed_runs := target_runs - scored_initial
  needed_runs / remaining_overs

theorem required_run_rate_correct (target_runs : ℝ) (run_rate_initial : ℝ) (overs_initial remaining_overs : ℕ) :
  run_rate_initial = 3.2 → overs_initial = 10 → remaining_overs = 40 → target_runs = 350 →
  cricket_run_rate run_rate_initial overs_initial target_runs remaining_overs = 7.95 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold cricket_run_rate
  norm_num
  sorry

end required_run_rate_correct_l269_269702


namespace intersection_proof_l269_269068

open Set

variable (U M N : Set ℝ)

def U : Set ℝ := univ
def M : Set ℝ := { x | x ≤ 1 }
def N : Set ℝ := { x | x^2 - 4 < 0 }
def CUM : Set ℝ := { x | x > 1 }

theorem intersection_proof : (CUM ∩ N) = { x | 1 < x ∧ x < 2 } :=
by {
  dsimp [CUM, N],
  ext,
  simp,
  tauto,
}

end intersection_proof_l269_269068


namespace angle_bisector_theorem_l269_269282

open EuclideanGeometry

variables {A B C I D E F X : Point}

-- Assume the given conditions as hypotheses
variables (h1 : IsIncenter I A B C)
          (h2 : IncircleTouchesAt D E F A B C I)
          (h3 : CircumcircleIntersectsAt X A E F A B C)

-- Stating the theorem
theorem angle_bisector_theorem : ∠BXA = ∠CXD :=
by
  -- skip the proof using sorry for now
  sorry

end angle_bisector_theorem_l269_269282


namespace GoldenRatioExpression_l269_269419

noncomputable def cos_72 := real.cos (real.pi * 4 / 5)
noncomputable def cos_18 := real.cos (real.pi / 10)

theorem GoldenRatioExpression :
  let a := 2 * cos_72 in
  a = 2 * cos_72 → 
  (a * cos_18) / (real.sqrt (2 - a)) = 1 / 2 := 
by
  intro a
  intro h
  rw [h, ← real.cos_eq_iff_eq (real.pi * 2 / 5) ⟨real.pi * 3 / 5, λ h, false.elim (by linarith)⟩]
  sorry

end GoldenRatioExpression_l269_269419


namespace smallest_x_plus_y_l269_269618

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l269_269618


namespace fewer_parking_spaces_on_fourth_level_l269_269939

theorem fewer_parking_spaces_on_fourth_level 
  (spaces_first_level : ℕ) (spaces_second_level : ℕ) (spaces_third_level : ℕ) (spaces_fourth_level : ℕ) 
  (total_spaces_garage : ℕ) (cars_parked : ℕ) 
  (h1 : spaces_first_level = 90)
  (h2 : spaces_second_level = spaces_first_level + 8)
  (h3 : spaces_third_level = spaces_second_level + 12)
  (h4 : total_spaces_garage = 299)
  (h5 : cars_parked = 100)
  (h6 : spaces_first_level + spaces_second_level + spaces_third_level + spaces_fourth_level = total_spaces_garage) :
  spaces_third_level - spaces_fourth_level = 109 := 
by
  sorry

end fewer_parking_spaces_on_fourth_level_l269_269939


namespace min_crates_with_same_apple_count_l269_269958

theorem min_crates_with_same_apple_count : 
  ∀ (crates : ℕ) (min_apples max_apples : ℕ), 
  crates = 150 → 
  min_apples = 110 → 
  max_apples = 145 → 
  (∃ n, n = 5 ∧ ∀ (distribution : ℕ → ℕ), 
    (∀ x, min_apples ≤ x ∧ x ≤ max_apples → distribution x ≤ crates) → 
    ∃ count, min_apples ≤ count ∧ count ≤ max_apples ∧ distribution count ≥ n) := 
by
  intros crates min_apples max_apples h1 h2 h3 
  use 5
  split
  · rfl
  · intros distribution h
    sorry

end min_crates_with_same_apple_count_l269_269958


namespace range_of_a_l269_269301

noncomputable def f (a x : ℝ) : ℝ := log a (x + a / x - 1)

theorem range_of_a (a : ℝ) (R : set ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ x : ℝ, f a x ∈ R) :
  0 < a ∧ a ≤ 1 / 4 :=
by
  sorry

end range_of_a_l269_269301


namespace num_possible_measures_of_C_l269_269116

-- Definitions for conditions
def supplementary (C D : ℕ) : Prop := C + D = 180
def multiple (C D : ℕ) : Prop := ∃ m : ℕ, m ≥ 1 ∧ C = m * D

-- Statement of the problem
theorem num_possible_measures_of_C : 
  ∃ (n : ℕ), n = 17 ∧ ∀ (C D : ℕ), 
    supplementary C D ∧ multiple C D → (∃ m, m ≥ 1 ∧ C = (m * D)) := 
begin
  sorry
end

end num_possible_measures_of_C_l269_269116


namespace percentage_difference_l269_269258

open scoped Classical

theorem percentage_difference (original_number new_number : ℕ) (h₀ : original_number = 60) (h₁ : new_number = 30) :
  (original_number - new_number) / original_number * 100 = 50 :=
by
      sorry

end percentage_difference_l269_269258


namespace move_within_three_trips_l269_269855

def settlements := list ℕ

def directly_connected (A X : ℕ) (connections : list (ℕ × ℕ)) : Prop :=
  (A, X) ∈ connections

def next_settlement (A : ℕ) (n : ℕ) : ℕ :=
  if A = n then 1 else A + 1

theorem move_within_three_trips
  (n : ℕ)
  (settlements : list ℕ)
  (connections : list (ℕ × ℕ))
  (h1 : ∀ A ∈ settlements, ∃ X, directly_connected A X connections)
  (h2 : ∀ A B ∈ settlements, A ≠ B →
    (directly_connected A (next_settlement B n) connections ↔ ¬ directly_connected B (next_settlement (next_settlement A n) n) connections))
  (i j : ℕ)
  (hi : i ∈ settlements)
  (hj : j ∈ settlements)
  (hne : i ≠ j) :
  ∃ (t : list ℕ), length t ≤ 3 ∧ list.chain' (directly_connected connections) i (t ++ [j]) :=
sorry

end move_within_three_trips_l269_269855


namespace probability_of_selecting_male_is_three_fifths_l269_269477

-- Define the number of male and female students
def num_male_students : ℕ := 6
def num_female_students : ℕ := 4

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability of selecting a male student's ID
def probability_male_student : ℚ := num_male_students / total_students

-- Theorem: The probability of selecting a male student's ID is 3/5
theorem probability_of_selecting_male_is_three_fifths : probability_male_student = 3 / 5 :=
by
  -- Proof to be filled in
  sorry

end probability_of_selecting_male_is_three_fifths_l269_269477


namespace nine_digit_no_zero_ending_5_not_perfect_square_l269_269819

theorem nine_digit_no_zero_ending_5_not_perfect_square
  (D : ℕ)
  (hnine : D.digits.length = 9)
  (hnozero : 0 ∉ D.digits)
  (hexdigits : ∀ d, d ∈ D.digits → d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hend5 : D % 10 = 5) :
  ¬ (∃ A : ℕ, D = A^2) :=
sorry

end nine_digit_no_zero_ending_5_not_perfect_square_l269_269819


namespace calculate_difference_l269_269126

theorem calculate_difference (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
by
  sorry

end calculate_difference_l269_269126


namespace _l269_269502

noncomputable theorem rings_on_display
  (total_necklace_capacity : ℕ)
  (current_necklaces : ℕ)
  (total_bracelet_capacity : ℕ)
  (current_bracelets : ℕ)
  (total_cost : ℕ)
  (cost_per_necklace : ℕ)
  (cost_per_ring : ℕ)
  (cost_per_bracelet : ℕ)
  (total_ring_capacity : ℕ)
  (necklaces_to_add := total_necklace_capacity - current_necklaces)
  (bracelets_to_add := total_bracelet_capacity - current_bracelets)
  (cost_for_necklaces := necklaces_to_add * cost_per_necklace)
  (cost_for_bracelets := bracelets_to_add * cost_per_bracelet)
  (remaining_budget := total_cost - (cost_for_necklaces + cost_for_bracelets))
  (rings_to_add := remaining_budget / cost_per_ring)
  (rings_on_display := total_ring_capacity - rings_to_add) :
  rings_on_display = 18 := by
  let current_necklaces := 5
  let total_necklace_capacity := 12
  let current_bracelets := 8
  let total_bracelet_capacity := 15
  let total_cost := 183
  let cost_per_necklace := 4
  let cost_per_ring := 10
  let cost_per_bracelet := 5
  let total_ring_capacity := 30
  have necklaces_to_add := total_necklace_capacity - current_necklaces
  have bracelets_to_add := total_bracelet_capacity - current_bracelets
  have cost_for_necklaces := necklaces_to_add * cost_per_necklace
  have cost_for_bracelets := bracelets_to_add * cost_per_bracelet
  have remaining_budget := total_cost - (cost_for_necklaces + cost_for_bracelets)
  have rings_to_add := remaining_budget / cost_per_ring
  have rings_on_display := total_ring_capacity - rings_to_add
  sorry

end _l269_269502


namespace complex_conjugate_of_z_l269_269762

theorem complex_conjugate_of_z : 
  let z := (2 + complex.i) / (1 + complex.i^2 + complex.i^5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269762


namespace black_to_white_ratio_l269_269231

theorem black_to_white_ratio (initial_black initial_white new_black new_white : ℕ) 
  (h1 : initial_black = 7) (h2 : initial_white = 18)
  (h3 : new_black = 31) (h4 : new_white = 18) :
  (new_black : ℚ) / new_white = 31 / 18 :=
by
  sorry

end black_to_white_ratio_l269_269231


namespace chandra_monster_hunt_l269_269225

theorem chandra_monster_hunt :
    let d0 := 2   -- monsters on the first day
    let d1 := 2 * d0   -- monsters on the second day
    let d2 := 2 * d1   -- monsters on the third day
    let d3 := 2 * d2   -- monsters on the fourth day
    let d4 := 2 * d3   -- monsters on the fifth day
in d0 + d1 + d2 + d3 + d4 = 62 := by
  sorry

end chandra_monster_hunt_l269_269225


namespace correct_statements_count_l269_269103

/-
  Question: How many students have given correct interpretations of the algebraic expression \( 7x \)?
  Conditions:
    - Xiaoming's Statement: \( 7x \) can represent the sum of \( 7 \) and \( x \).
    - Xiaogang's Statement: \( 7x \) can represent the product of \( 7 \) and \( x \).
    - Xiaoliang's Statement: \( 7x \) can represent the total price of buying \( x \) pens at a unit price of \( 7 \) yuan.
  Given these conditions, prove that the number of correct statements is \( 2 \).
-/

theorem correct_statements_count (x : ℕ) :
  (if 7 * x = 7 + x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) = 2 := sorry

end correct_statements_count_l269_269103


namespace more_red_than_yellow_l269_269727

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end more_red_than_yellow_l269_269727


namespace complex_conjugate_of_z_l269_269764

theorem complex_conjugate_of_z : 
  let z := (2 + complex.i) / (1 + complex.i^2 + complex.i^5) in
  complex.conj z = 1 + 2 * complex.i :=
by
  sorry

end complex_conjugate_of_z_l269_269764


namespace syam_investment_l269_269220

def dividend_earned (earnings dividend_rate face_value : ℝ) : Prop :=
  earnings = (dividend_rate / 100) * face_value

def market_value (face_value market_quotient : ℝ) : ℝ :=
  (market_quotient / 100) * face_value

theorem syam_investment (earnings : ℝ) (dividend_rate : ℝ) (market_quotient : ℝ)
  (h_dividend: dividend_earned earnings dividend_rate face_value)
  (face_value : ℝ) : 
  market_quotient = 135 →
  dividend_rate = 9 →
  earnings = 120 →
  face_value = 1333.33 →
  market_value face_value market_quotient = 1800 := by sorry

end syam_investment_l269_269220


namespace carols_rectangle_length_l269_269546

theorem carols_rectangle_length :
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  carol_length = 5 :=
by
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  show carol_length = 5
  sorry

end carols_rectangle_length_l269_269546


namespace arithmetic_example_l269_269222

theorem arithmetic_example : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end arithmetic_example_l269_269222


namespace last_ball_probability_l269_269489

theorem last_ball_probability (w b : ℕ) (H : w > 0 ∨ b > 0) :
  (w % 2 = 1 → ∃ p : ℝ, p = 1 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) ∧ 
  (w % 2 = 0 → ∃ p : ℝ, p = 0 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) :=
by sorry

end last_ball_probability_l269_269489


namespace max_mark_is_600_l269_269516

-- Define the conditions
def forty_percent (M : ℝ) : ℝ := 0.40 * M
def student_score : ℝ := 175
def additional_marks_needed : ℝ := 65

-- The goal is to prove that the maximum mark is 600
theorem max_mark_is_600 (M : ℝ) :
  forty_percent M = student_score + additional_marks_needed → M = 600 := 
by 
  sorry

end max_mark_is_600_l269_269516


namespace magnitude_2a_sub_b_l269_269312

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_2a_sub_b : (‖(2 * a.1 - b.1, 2 * a.2 - b.2)‖ = 5) :=
by {
  sorry
}

end magnitude_2a_sub_b_l269_269312


namespace de_equals_zero_l269_269835

-- Define the roots of the polynomial
def cos_pi_div_5 := Real.cos (Real.pi / 5)
def cos_3pi_div_5 := Real.cos (3 * Real.pi / 5)

-- Define the polynomial
def Q (x : ℝ) (d e : ℝ) := x^2 + d * x + e

-- Roots conditions
lemma roots_condition (d e : ℝ) :
  (∃ x : ℝ, Q x d e = 0 ∧ x = cos_pi_div_5)
  ∧ (∃ x : ℝ, Q x d e = 0 ∧ x = cos_3pi_div_5) :=
sorry

-- Prove that de equals 0 given the conditions
theorem de_equals_zero (d e : ℝ) (h : (∃ x : ℝ, Q x d e = 0 ∧ x = cos_pi_div_5)
  ∧ (∃ x : ℝ, Q x d e = 0 ∧ x = cos_3pi_div_5)) :
  d * e = 0 :=
sorry

end de_equals_zero_l269_269835


namespace nested_fraction_simplification_l269_269543

theorem nested_fraction_simplification :
  (2 / (2 + 2 / (3 + 1 / 4 : ℚ) : ℚ) : ℚ) = 13 / 17 :=
begin
  sorry
end

end nested_fraction_simplification_l269_269543


namespace omega_value_f_value_at_x0_l269_269373

def a (ω x : Real) : Real × Real := (Real.cos (ω * x) - Real.sin (ω * x), -1)
def b (ω x : Real) : Real × Real := (2 * Real.sin (ω * x), -1)
def f (ω x : Real) : Real := let ⟨a1, a2⟩ := a ω x; let ⟨b1, b2⟩ := b ω x; a1 * b1 + a2 * b2

theorem omega_value {ω : Real} (h : (∃ T > 0, ∀ x, f ω (x + T) = f ω x) ∧ (∃ T > 0, T = 4 * π) ∧ ω > 0) :
  ω = 1 / 4 :=
sorry

theorem f_value_at_x0 (x0 : Real) (hx0 : Real.sin x0 = -1/2 ∧ x0 > -π / 2 ∧ x0 < π / 2) :
  f (1 / 4) x0 = sqrt 2 / 2 :=
sorry

end omega_value_f_value_at_x0_l269_269373


namespace range_of_a_l269_269307

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := 
sorry

end range_of_a_l269_269307


namespace num_ways_diagonal_len_l269_269187

theorem num_ways_diagonal_len (PQ QR RS SP : ℕ) (hPQ : PQ = 9) (hQR : QR = 11) (hRS : RS = 15) (hSP : SP = 14) :
  {d : ℕ | 3 ≤ d ∧ d ≤ 19}.finite.to_finset.card = 17 :=
by
  -- Placeholders for the proof steps
  sorry

end num_ways_diagonal_len_l269_269187


namespace gathering_people_total_l269_269216

theorem gathering_people_total (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 :=
by
  sorry

end gathering_people_total_l269_269216


namespace allocation_of_fabric_l269_269922

theorem allocation_of_fabric (x : ℝ) (y : ℝ) 
  (fabric_for_top : 3 * x = 2 * x)
  (fabric_for_pants : 3 * y = 3 * (600 - x))
  (total_fabric : x + y = 600)
  (sets_match : (x / 3) * 2 = (y / 3) * 3) : 
  x = 360 ∧ y = 240 := 
by
  sorry

end allocation_of_fabric_l269_269922


namespace prob_blue_section_damaged_all_days_l269_269904

noncomputable def prob_of_7_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_blue_section_damaged_all_days :
  prob_of_7_successes 7 7 (2 / 7) = 128 / 823543 :=
by sorry

end prob_blue_section_damaged_all_days_l269_269904


namespace find_b_max_value_l269_269630

theorem find_b_max_value (b : ℝ) :
  (∃ θ : ℝ, 7 = 4 * b^2 - 3 * b^2 * sin (2 * θ) - 3 * b * sin θ + 9 / 4) → b = 1 ∨ b = -1 :=
sorry

end find_b_max_value_l269_269630


namespace probability_of_7_successes_l269_269887

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l269_269887


namespace y_pow_x_eq_x_pow_y_l269_269674

open Real

noncomputable def x (n : ℕ) : ℝ := (1 + 1 / n) ^ n
noncomputable def y (n : ℕ) : ℝ := (1 + 1 / n) ^ (n + 1)

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) : (y n) ^ (x n) = (x n) ^ (y n) :=
by
  sorry

end y_pow_x_eq_x_pow_y_l269_269674


namespace find_q_x_l269_269568

noncomputable def q : ℚ[X] :=
  4.5 * (X - 2) * (X + 1)

theorem find_q_x (a : ℚ) (h_asymptotes : ∀ x, q(x) = a * (x - 2) * (x + 1)) (h_no_horizontal : degree (4.5 * X^2 - 4.5 * X - 9) = 2) (h_q_3 : q(3) = 18) :
  q = 4.5 * X^2 - 4.5 * X - 9 :=
by
  sorry

end find_q_x_l269_269568


namespace michael_payment_l269_269379

variables (cats dogs charge_per_animal total_animals total_cost : ℕ)

def michael_animals : ℕ := cats + dogs
def michael_cost (charge_per_animal total_animals : ℕ) : ℕ := total_animals * charge_per_animal

theorem michael_payment : 
  cats = 2 ∧ dogs = 3 ∧ charge_per_animal = 13 →
  michael_animals = 5 ∧ michael_cost charge_per_animal michael_animals = 65 :=
by 
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end michael_payment_l269_269379


namespace line_equation_ab_c_l269_269838

def u : ℂ := -3 + 4 * complex.i
def v : ℂ := 2 + 2 * complex.i

theorem line_equation_ab_c :
  ∃ a b : ℂ, ∃ c : ℝ,
    a = 5 + 2 * complex.i ∧ 
    b = 5 - 2 * complex.i ∧
    c = -2 ∧
    a * b = 21 ∧
    ∀ z : ℂ, ((z - u) / (v - u)).im = 0 ↔ a * z + b * complex.conj z = c
:=
sorry

end line_equation_ab_c_l269_269838


namespace Heracles_age_l269_269981

variable (A H : ℕ)

theorem Heracles_age :
  (A = H + 7) →
  (A + 3 = 2 * H) →
  H = 10 :=
by
  sorry

end Heracles_age_l269_269981


namespace allocation_methods_count_l269_269243

theorem allocation_methods_count
  (V : Fin 5 → Type) -- Five volunteer service groups represented by indices 0 to 4
  (C : Fin 3 → Type) -- Three competition venues represented by indices 0 to 2
  (A_not_assigned_A : ∀ v, (v ≠ 0) → V v) -- Group 0 (A) cannot be assigned to venue 0 (A)
  (each_venue_one : ∀ c, ∃ v, V v) -- At least one volunteer service group must be assigned to each venue
  (disjoint : ∀ {v1 v2 c1 c2}, (V v1) → (V v2) → (v1 = v2 ↔ c1 = c2)) -- Groups can only serve at one venue
  : Fintype.card (Finset.Pi (Fin 3) (fun _ => Finset.Univ : Finset (Fin 5))) - sorry = 100 := sorry


end allocation_methods_count_l269_269243


namespace expand_and_simplify_product_l269_269248

theorem expand_and_simplify_product :
  5 * (x + 6) * (x + 2) * (x + 7) = 5 * x^3 + 75 * x^2 + 340 * x + 420 := 
by
  sorry

end expand_and_simplify_product_l269_269248


namespace sum_of_angles_difference_l269_269067

-- Definition of f(k) as the sum of the interior angles of a convex polygon with k sides
def f (k : ℕ) : ℝ := (k - 2) * real.pi

-- The theorem to be proven
theorem sum_of_angles_difference (k : ℕ) : f (k + 1) - f k = real.pi :=
by sorry

end sum_of_angles_difference_l269_269067


namespace twelfth_prime_is_37_l269_269457

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sequence : ℕ → ℕ
| 0     := 2
| 1     := 3
| 2     := 5
| 3     := 7
| 4     := 11
| 5     := 13
| 6     := 17
| 7     := 19
| 8     := 23
| 9     := 29
| 10    := 31
| 11    := 37
| (n+1) := sorry  -- This should continue the prime sequence, but for our proof problem, it's good enough.

theorem twelfth_prime_is_37 : prime_sequence 11 = 37 := by
  sorry

end twelfth_prime_is_37_l269_269457


namespace isosceles_triangle_perimeter_l269_269693

-- Define the lengths of the sides of the isosceles triangle
def side1 : ℕ := 12
def side2 : ℕ := 12
def base : ℕ := 17

-- Define the perimeter as the sum of all three sides
def perimeter : ℕ := side1 + side2 + base

-- State the theorem that needs to be proved
theorem isosceles_triangle_perimeter : perimeter = 41 := by
  -- Insert the proof here
  sorry

end isosceles_triangle_perimeter_l269_269693


namespace correct_calculation_l269_269194

theorem correct_calculation (N : ℤ) (h : 41 - N = 12) : 41 + N = 70 := 
by 
  sorry

end correct_calculation_l269_269194


namespace determine_train_and_car_number_l269_269806

def SECRET (s e c r e t : ℕ) : ℕ := s*10^5 + e*10^4 + c*10^3 + r*10^2 + e*10^1 + t
def OPEN (o p e n : ℕ) : ℕ := o*10^3 + p*10^2 + e*10^1 + n
def ANSWER (a n s w e r : ℕ) : ℕ := a*10^5 + n*10^4 + s*10^3 + w*10^2 + e*10^1 + r
def YOUR (y o u r : ℕ) : ℕ := y*10^3 + o*10^2 + u*10^1 + r
def OPENED (o p e n e d : ℕ) : ℕ := o*10^5 + p*10^4 + e*10^3 + n*10^2 + e*10^1 + d

theorem determine_train_and_car_number 
  (s e c r t o p n a w y u d : ℕ)
  (h1 : SECRET s e c r e t - OPEN o p e n = ANSWER a n s w e r - YOUR y o u r)
  (h2 : SECRET s e c r e t - OPENED o p e n e d = 20010)
  (unique_digits : ∀ x y, x ≠ y → x ≠ y) :
  ∃ (train car : ℕ), train = 392 ∧ car = 0 :=
sorry

end determine_train_and_car_number_l269_269806


namespace smallest_integer_cube_ends_in_528_l269_269261

theorem smallest_integer_cube_ends_in_528 :
  ∃ (n : ℕ), (n^3 % 1000 = 528 ∧ ∀ m : ℕ, (m^3 % 1000 = 528) → m ≥ n) ∧ n = 428 :=
by
  sorry

end smallest_integer_cube_ends_in_528_l269_269261


namespace puppies_adopted_each_day_l269_269941

variable (initial_puppies additional_puppies days total_puppies puppies_per_day : ℕ)

axiom initial_puppies_ax : initial_puppies = 9
axiom additional_puppies_ax : additional_puppies = 12
axiom days_ax : days = 7
axiom total_puppies_ax : total_puppies = initial_puppies + additional_puppies
axiom adoption_rate_ax : total_puppies / days = puppies_per_day

theorem puppies_adopted_each_day : 
  initial_puppies = 9 → additional_puppies = 12 → days = 7 → total_puppies = initial_puppies + additional_puppies → total_puppies / days = puppies_per_day → puppies_per_day = 3 :=
by
  intro initial_puppies_ax additional_puppies_ax days_ax total_puppies_ax adoption_rate_ax
  sorry

end puppies_adopted_each_day_l269_269941


namespace center_of_symmetry_l269_269929

-- Define the symmetry conditions
def is_symmetric_about_x_axis (figure : set (ℝ × ℝ)) : Prop :=
  ∀ {x y : ℝ}, (x, y) ∈ figure → (x, -y) ∈ figure

def is_symmetric_about_y_axis (figure : set (ℝ × ℝ)) : Prop :=
  ∀ {x y : ℝ}, (x, y) ∈ figure → (-x, y) ∈ figure

-- The main theorem stating that a figure with two perpendicular axes of symmetry has a center of symmetry
theorem center_of_symmetry
  (figure : set (ℝ × ℝ))
  (h_x : is_symmetric_about_x_axis figure)
  (h_y : is_symmetric_about_y_axis figure) :
  ∀ {x y : ℝ}, (x, y) ∈ figure → (-x, -y) ∈ figure :=
by
  sorry

end center_of_symmetry_l269_269929


namespace P_is_B_l269_269530

noncomputable def maximum_area_rectangle (P : Point) (AB : Line) (PNDM : Rectangle) : Prop :=
  -- Define all given geometric conditions
  let side_length_square := 4
  let pentagon_ABCDE := cut_corner_square side_length_square
  let AF := 2
  let FB := 1
  place_point_on_line P AB ∧
  -- Define the criterion for maximum area
  rectangle_with_max_area (PNDM) P

theorem P_is_B 
  (side_length_square : ℝ)
  (pentagon_ABCDE : Pentagon)
  (AF FB : ℝ)
  (AB : Line)
  (P B : Point)
  (PNDM : Rectangle) 
  (h1 : AF = 2) 
  (h2 : FB = 1) 
  (h3 : P ∈ AB)
  (h4 : side_length_square = 4) 
  (h5 : cut_corner_square side_length_square = pentagon_ABCDE)
  (h6 : rectangle_with_max_area (PNDM) P) :
  P = B :=
begin
  sorry
end

end P_is_B_l269_269530


namespace part_I_part_II_l269_269641

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem part_I (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 0) →
  -1 < a ∧ a ≤ 11/5 :=
sorry

noncomputable def g (x a : ℝ) : ℝ := 
  if abs x ≥ 1 then 2 * x^2 - 2 * a * x + a + 1 
  else -2 * a * x + a + 3

theorem part_II (a : ℝ) :
  (∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < 3 ∧ g x1 a = 0 ∧ g x2 a = 0) →
  1 + Real.sqrt 3 < a ∧ a ≤ 19/5 :=
sorry

end part_I_part_II_l269_269641
