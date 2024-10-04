import Mathlib

namespace count_lineup_excluding_youngest_l734_734351

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l734_734351


namespace vasya_fraction_l734_734675

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l734_734675


namespace set_union_complement_l734_734275

def U := {1, 2, 3, 4}
def A := {1, 3}
def B := {1, 3, 4}

theorem set_union_complement :
  A ∪ (U \ B) = {1, 2, 3} := by
  sorry

end set_union_complement_l734_734275


namespace line_intersects_circle_l734_734759

open Real

noncomputable def circle_C : set (ℝ × ℝ) :=
  {p | let ⟨x, y⟩ := p in (x^2 + y^2 - 4 * x = 0)}

noncomputable def line_l (k : ℝ) : set (ℝ × ℝ) :=
  {p | let ⟨x, y⟩ := p in (k * x - 3 * k - y = 0)}

theorem line_intersects_circle (k : ℝ) :
  ∃ p ∈ line_l k, p ∈ circle_C :=
sorry

end line_intersects_circle_l734_734759


namespace moles_of_magnesium_l734_734818

theorem moles_of_magnesium (moles_H2SO4 : ℕ) (h : moles_H2SO4 = 3) : 
  let moles_Mg := moles_H2SO4 in 
  moles_Mg = 3 :=
by 
  rw h 
  refl

end moles_of_magnesium_l734_734818


namespace solve_for_x_l734_734595

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := 
by
  sorry

end solve_for_x_l734_734595


namespace vitya_older_than_masha_probability_l734_734534

-- Definition of Days in June
def days_in_june : ℕ := 30

-- Total number of possible outcomes for birth dates (30 days for Vitya × 30 days for Masha)
def total_outcomes : ℕ := days_in_june * days_in_june

-- Sum of favorable outcomes where Vitya is at least one day older than Masha
def favorable_outcomes : ℕ := (1 to (days_in_june - 1)).sum

-- The probability calculation function
noncomputable def probability (n f : ℕ) : ℚ := ⟨f, n⟩

-- The statement of the theorem
theorem vitya_older_than_masha_probability :
  probability total_outcomes favorable_outcomes = 29 / 60 := sorry

end vitya_older_than_masha_probability_l734_734534


namespace alternating_sum_10002_l734_734559

def alternating_sum (n : ℕ) : ℤ :=
  (List.range (n+1)).map (λ k, if even k then (k : ℤ) else -(k : ℤ)).sum

theorem alternating_sum_10002 : alternating_sum 10002 = -5001 := by
  sorry

end alternating_sum_10002_l734_734559


namespace smallest_integer_for_inequality_l734_734555

theorem smallest_integer_for_inequality :
  ∃ x : ℤ, x^2 < 2 * x + 1 ∧ ∀ y : ℤ, y^2 < 2 * y + 1 → x ≤ y := sorry

end smallest_integer_for_inequality_l734_734555


namespace average_speed_round_trip_l734_734120

theorem average_speed_round_trip :
  ∀ (D : ℝ), 
  D > 0 → 
  let upstream_speed := 6 
  let downstream_speed := 5 
  (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 :=
by
  intro D hD
  let upstream_speed := 6
  let downstream_speed := 5
  have h : (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 := sorry
  exact h

end average_speed_round_trip_l734_734120


namespace polynomial_divisibility_l734_734897

theorem polynomial_divisibility (m : ℕ) (h_pos : 0 < m) : 
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1)^(2 * m) - x^(2 * m) - 2 * x - 1 :=
sorry

end polynomial_divisibility_l734_734897


namespace problem_statement_l734_734226

open Real

noncomputable def log4 (x : ℝ) : ℝ := log x / log 4

noncomputable def a : ℝ := log4 (sqrt 5)
noncomputable def b : ℝ := log 2 / log 5
noncomputable def c : ℝ := log4 5

theorem problem_statement : b < a ∧ a < c :=
by
  sorry

end problem_statement_l734_734226


namespace women_workers_l734_734309

noncomputable def totalWorkers (W : ℤ) : Prop :=
  W / 3 + W / 2 + 120 = W

theorem women_workers (W : ℤ) (h1 : totalWorkers W) : 330 := by
  sorry

end women_workers_l734_734309


namespace stacy_history_paper_length_l734_734033

theorem stacy_history_paper_length
  (days : ℕ)
  (pages_per_day : ℕ)
  (h_days : days = 6)
  (h_pages_per_day : pages_per_day = 11) :
  (days * pages_per_day) = 66 :=
by {
  sorry -- Proof goes here
}

end stacy_history_paper_length_l734_734033


namespace Sachin_younger_than_Rahul_l734_734904

theorem Sachin_younger_than_Rahul :
  ∀ (S R : ℕ), S = 49 ∧ (S : ℚ) / R = 7 / 9 → R - S = 14 :=
by
  intros S R h,
  cases h with hS hRatio,
  rw hS at *,
  have hR : (49 : ℚ) / (R : ℚ) = 7 / 9 := hRatio,
  -- The remaining part of the proof is left as sorry
  sorry

end Sachin_younger_than_Rahul_l734_734904


namespace jessica_total_monthly_payment_l734_734389

-- Definitions for the conditions
def basicCableCost : ℕ := 15
def movieChannelsCost : ℕ := 12
def sportsChannelsCost : ℕ := movieChannelsCost - 3

-- The statement to be proven
theorem jessica_total_monthly_payment :
  basicCableCost + movieChannelsCost + sportsChannelsCost = 36 := 
by
  sorry

end jessica_total_monthly_payment_l734_734389


namespace count_lineup_excluding_youngest_l734_734349

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l734_734349


namespace solve_quadratic_calculate_expression_l734_734996

-- Part 1: Solving the quadratic equation

theorem solve_quadratic (x : ℝ) : x^2 - 4 * x - 3 = 0 ↔ (x = 2 + real.sqrt 7) ∨ (x = 2 - real.sqrt 7) := 
sorry

-- Part 2: Calculating the given expression

theorem calculate_expression : 
  abs (-3) - 4 * real.sin (real.pi / 4) + real.sqrt 8 + (real.pi - 3)^0 = 4 := 
sorry

end solve_quadratic_calculate_expression_l734_734996


namespace pascals_triangle_third_number_l734_734098

theorem pascals_triangle_third_number (n : ℕ) (k : ℕ) (hnk : n = 51) (hk : k = 2) :
  (nat.choose n k) = 1275 :=
by {
  subst hnk,
  subst hk,
  sorry
}

end pascals_triangle_third_number_l734_734098


namespace min_value_of_reciprocal_sum_l734_734932

variables (a m n : ℝ)

-- Conditions translation
def function_def := a > 0 ∧ a ≠ 1 ∧ ∀ x, a^(2-x) + 1 = 2, (2 * m) + (2 * n) = 1, m * n > 0

-- The proof goal
theorem min_value_of_reciprocal_sum
  (h : function_def a m n) :
  (1 / m) + (1 / n) = 8 :=
sorry

end min_value_of_reciprocal_sum_l734_734932


namespace vitya_older_than_masha_l734_734537

theorem vitya_older_than_masha :
  (∃ (days_in_month : ℕ) (total_pairs : ℕ) (favorable_pairs : ℕ)
     (p : ℚ),
    days_in_month = 30 ∧
    total_pairs = days_in_month * days_in_month ∧
    favorable_pairs = ∑ i in Finset.range(30), i ∧
    p = favorable_pairs / total_pairs ∧
    p = 29 / 60) :=
begin
  let days_in_month := 30,
  let total_pairs := days_in_month * days_in_month,
  let favorable_pairs := ∑ i in Finset.range(days_in_month), i,
  let p := favorable_pairs / total_pairs,
  use [days_in_month, total_pairs, favorable_pairs, p],
  split,
  { refl, },
  split,
  { refl, },
  split,
  { sorry, },
  split,
  { sorry, },
end

end vitya_older_than_masha_l734_734537


namespace sum_of_roots_equal_l734_734276

theorem sum_of_roots_equal
  (P Q R : ℚ[X])
  (hP : P.degree = 2) (hQ : Q.degree = 2) (hR : R.degree = 2)
  (hP_pos : P.leading_coeff > 0) (hQ_pos : Q.leading_coeff > 0) (hR_pos : R.leading_coeff > 0)
  (hP_roots : ∃ a1 a2 : ℚ, P = polynomial.C ((a1 * a2) * P.leading_coeff) * (polynomial.X - polynomial.C a1) * (polynomial.X - polynomial.C a2) ∧ a1 ≠ a2)
  (hQ_roots : ∃ b1 b2 : ℚ, Q = polynomial.C ((b1 * b2) * Q.leading_coeff) * (polynomial.X - polynomial.C b1) * (polynomial.X - polynomial.C b2) ∧ b1 ≠ b2)
  (hR_roots : ∃ c1 c2 : ℚ, R = polynomial.C ((c1 * c2) * R.leading_coeff) * (polynomial.X - polynomial.C c1) * (polynomial.X - polynomial.C c2) ∧ c1 ≠ c2)
  (hR_cond : ∀ c1 c2 : ℚ, R.eval c1 = 0 → R.eval c2 = 0 → (P.eval c1 + Q.eval c1) = (P.eval c2 + Q.eval c2))
  (hP_cond : ∀ a1 a2 : ℚ, P.eval a1 = 0 → P.eval a2 = 0 → (Q.eval a1 + R.eval a1) = (Q.eval a2 + R.eval a2))
  (hQ_cond : ∀ b1 b2 : ℚ, Q.eval b1 = 0 → Q.eval b2 = 0 → (P.eval b1 + R.eval b1) = (P.eval b2 + R.eval b2)) :
  let a1, a2 := classical.some hP_roots,
      b1, b2 := classical.some hQ_roots,
      c1, c2 := classical.some hR_roots in
  a1 + a2 = b1 + b2 ∧ b1 + b2 = c1 + c2 :=
sorry

end sum_of_roots_equal_l734_734276


namespace complex_problem_solution_l734_734755

theorem complex_problem_solution (x y : ℝ) (h_eq1 : (x - 2) * complex.I - y = 1 + complex.I) : 
  (1 + complex.I) ^ (x + y) = 2 * complex.I :=
by
  sorry

end complex_problem_solution_l734_734755


namespace line_equation_l734_734971

theorem line_equation (a b c x₀ y₀ : ℝ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : c = 9) (h₄ : x₀ = 2) (h₅ : y₀ = -3) :
  ∃ m t, (∀ x y, (y = m * x + t) ↔ (3 * x + 6 * y = 9) ∧ (x₀, y₀) = (2, -3)) :=
by
  simp [h₁, h₂, h₃, h₄, h₅]
  let m := -1/2
  let t := -2
  use [m, t]
  intros x y
  constructor
  { intro h
    rw [h]
    ring,
    exact ⟨h₁, h₂, h₃, h₄, h₅⟩ }
  { rintro ⟨hx, hy⟩,
    rw [hx, hy],
    sorry }

end line_equation_l734_734971


namespace exponent_proof_l734_734581

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l734_734581


namespace social_event_handshakes_l734_734680

def handshake_count (total_people : ℕ) (group_a : ℕ) (group_b : ℕ) : ℕ :=
  let introductions_handshakes := group_b * (group_b - 1) / 2
  let direct_handshakes := group_b * (group_a - 1)
  introductions_handshakes + direct_handshakes

theorem social_event_handshakes :
  handshake_count 40 25 15 = 465 := by
  sorry

end social_event_handshakes_l734_734680


namespace consecutive_differences_impossible_l734_734447

theorem consecutive_differences_impossible :
  ∀ (n : ℕ), 
    n ≥ 7 
    → ∃ (numbers : Fin n → ℕ),
      (∀ i : Fin n, numbers (i + 1) = numbers i + 1) 
      → ¬(∃ (i j k l m: Fin n), 
           abs (numbers (j + 1) - numbers j) = 2 ∧ 
           abs (numbers (k + 1) - numbers k) = 1 ∧ 
           abs (numbers (l + 1) - numbers l) = 6 ∧ 
           abs (numbers (m + 1) - numbers m) = 1 ∧ 
           abs (numbers (i + 1) - numbers i) = 2) :=
begin
  sorry
end

end consecutive_differences_impossible_l734_734447


namespace A_days_to_complete_work_alone_l734_734598

theorem A_days_to_complete_work_alone (x : ℝ) (h1 : 0 < x) (h2 : 0 < 18) (h3 : 1/x + 1/18 = 1/6) : x = 9 :=
by
  sorry

end A_days_to_complete_work_alone_l734_734598


namespace selection_ways_l734_734068

namespace CulturalPerformance

-- Define basic conditions
def num_students : ℕ := 6
def can_sing : ℕ := 3
def can_dance : ℕ := 2
def both_sing_and_dance : ℕ := 1

-- Define the proof statement
theorem selection_ways :
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end CulturalPerformance

end selection_ways_l734_734068


namespace triangle_side_length_x_l734_734632

theorem triangle_side_length_x (x : ℤ) (hpos : x > 0) (hineq1 : 7 < x^2) (hineq2 : x^2 < 17) :
    x = 3 ∨ x = 4 :=
by {
  apply sorry
}

end triangle_side_length_x_l734_734632


namespace monotonic_on_interval_l734_734828

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * Real.log x + 2 / x

theorem monotonic_on_interval (a : ℝ) :
    (∀ x : ℝ, 1 ≤ x → deriv (λ x, f a x) x ≥ 0) ↔ 0 ≤ a := sorry

end monotonic_on_interval_l734_734828


namespace number_of_whole_numbers_between_roots_l734_734285

theorem number_of_whole_numbers_between_roots :
  let sqrt_18 := Real.sqrt 18
  let sqrt_98 := Real.sqrt 98
  Nat.card { x : ℕ | sqrt_18 < x ∧ x < sqrt_98 } = 5 := 
by
  sorry

end number_of_whole_numbers_between_roots_l734_734285


namespace solutionSet_l734_734051

def passesThroughQuadrants (a b : ℝ) : Prop :=
  a > 0

def intersectsXAxisAt (a b : ℝ) : Prop :=
  b = 2 * a

theorem solutionSet (a b x : ℝ) (hq : passesThroughQuadrants a b) (hi : intersectsXAxisAt a b) :
  (a * x > b) ↔ (x > 2) :=
by
  sorry

end solutionSet_l734_734051


namespace blueberries_per_basket_l734_734006

-- Definitions based on the conditions
def total_blueberries : ℕ := 200
def total_baskets : ℕ := 10

-- Statement to be proven
theorem blueberries_per_basket : total_blueberries / total_baskets = 20 := 
by
  sorry

end blueberries_per_basket_l734_734006


namespace cost_to_fly_C_to_B_is_367_48_l734_734313

-- Definitions based on problem conditions:
def distance_AC := 4000 -- Distance from A to C
def distance_AB := 4500 -- Distance from A to B
def booking_fee := 120 -- Booking fee for the airplane
def cost_per_km := 0.12 -- Cost per kilometer for the airplane

-- Proof statement to show that the cost to fly from C to B is $367.48
theorem cost_to_fly_C_to_B_is_367_48 :
  let distance_BC := Real.sqrt (distance_AB ^ 2 - distance_AC ^ 2) in
  let cost := booking_fee + cost_per_km * distance_BC in
  Float.round (0.01 * cost) = 367.48 :=
begin
  sorry
end

end cost_to_fly_C_to_B_is_367_48_l734_734313


namespace probability_one_six_given_outcomes_different_l734_734965

open MeasureTheory

/-- Definition of the space of outcomes when two fair dice are rolled, ensuring distinct outcomes -/
def two_dice_outcomes_different : Finset (ℕ × ℕ) :=
  { (i, j) | i ∈ Finset.range 1 6 ∧ j ∈ Finset.range 1 6 ∧ i ≠ j }

/-- Definition of the event where at least one die shows a 6 -/
def at_least_one_six (outcome : ℕ × ℕ) : Prop :=
  outcome.1 = 6 ∨ outcome.2 = 6

/-- The probability that at least one outcome is 6, given the outcomes are different -/
theorem probability_one_six_given_outcomes_different :
  (∑ x in two_dice_outcomes_different.filter at_least_one_six, 1) /
  (∑ x in two_dice_outcomes_different, 1) = 1/3 := by
  sorry

end probability_one_six_given_outcomes_different_l734_734965


namespace find_y_l734_734413

theorem find_y (a b : ℝ) (y : ℝ) (h0 : b ≠ 0) (h1 : (3 * a)^(2 * b) = a^b * y^b) : y = 9 * a := by
  sorry

end find_y_l734_734413


namespace vasya_fraction_l734_734674

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l734_734674


namespace prod_97_103_l734_734702

theorem prod_97_103 : (97 * 103) = 9991 := 
by 
  have h1 : 97 = 100 - 3 := by rfl
  have h2 : 103 = 100 + 3 := by rfl
  calc
    97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
         ... = 100^2 - 3^2 : by rw (mul_sub (100:ℤ) 3 3)
         ... = 10000 - 9 : by norm_num
         ... = 9991 : by norm_num
 
end prod_97_103_l734_734702


namespace symmetric_set_cardinality_l734_734622

theorem symmetric_set_cardinality
  (T : Set (ℝ × ℝ))
  (h1 : ∀ p ∈ T, (-p.1, -p.2) ∈ T)  -- Symmetry about the origin
  (h2 : ∀ p ∈ T, (p.1, -p.2) ∈ T)   -- Symmetry about the x-axis
  (h3 : ∀ p ∈ T, (-p.1, p.2) ∈ T)   -- Symmetry about the y-axis
  (h4 : ∀ p ∈ T, (p.2, p.1) ∈ T)    -- Symmetry about the line y = x
  (h5 : ∀ p ∈ T, (-p.2, -p.1) ∈ T)  -- Symmetry about the line y = -x
  (h6 : (3, 4) ∈ T) : 
  Finite (T) ∧ (Set.card T = 8) :=
by
  sorry

end symmetric_set_cardinality_l734_734622


namespace power_equality_l734_734589

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l734_734589


namespace decompose_96_l734_734712

theorem decompose_96 (x y : ℤ) (h1 : x * y = 96) (h2 : x^2 + y^2 = 208) :
  (x = 8 ∧ y = 12) ∨ (x = 12 ∧ y = 8) ∨ (x = -8 ∧ y = -12) ∨ (x = -12 ∧ y = -8) := by
  sorry

end decompose_96_l734_734712


namespace minimum_omega_l734_734476

theorem minimum_omega (ω : ℝ) (h_omega_pos : ω > 0) :
    (∃ y : ℝ → ℝ, (∀ x, y x = sin (ω * x + ω * (π / 2) + (π / 3))) ∧ 
    (∀ x, y x = y (-x))) →
    (ω = 1 / 3) :=
sorry

end minimum_omega_l734_734476


namespace percent_profit_l734_734984

variable (C S : ℝ)

theorem percent_profit (h : 150 * C = 120 * S) : 100 * ((S - C) / C) = 25 := by
  have h1 : S = (5 / 4) * C :=
    calc
      S = (5 / 4) * C : by
        sorry
  calc
    100 * ((S - C) / C) = 100 * (((5 / 4) * C - C) / C) : by rw [h1]
    ... = 100 * ((1 / 4) * C / C) : by
      sorry
    ... = 100 * (1 / 4) : by rw [div_self (ne_of_gt (by linarith))]
    ... = 25 : by norm_num

end percent_profit_l734_734984


namespace mixture_alcohol_quantity_l734_734571

theorem mixture_alcohol_quantity:
  ∀ (A W : ℝ), 
    A / W = 4 / 3 ∧ A / (W + 7) = 4 / 5 → A = 14 :=
by
  intros A W h
  sorry

end mixture_alcohol_quantity_l734_734571


namespace hyperbola_asymptotes_slopes_l734_734516

theorem hyperbola_asymptotes_slopes :
  (∀ (x y : ℝ),
    x^2 / 2 - y^2 / 4 = 1 →
    ∃ (slope : ℝ),
      (slope = sqrt 2 ∨ slope = - sqrt 2)) :=
begin
  intros x y h,
  sorry
end

end hyperbola_asymptotes_slopes_l734_734516


namespace omega_min_value_l734_734482

theorem omega_min_value (ω : ℝ) (hω : ω > 0)
    (hSymmetry : ∀ x : ℝ, sin (ω * x + ω * π / 2 + π / 3) = sin (ω * -x + ω * π / 2 + π / 3)) :
    ω = 1 / 3 :=
begin
  sorry
end

end omega_min_value_l734_734482


namespace largest_C_inequality_l734_734735

theorem largest_C_inequality (C : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + x*y + 1 ≥ C*(x + y)) ↔ (C ≤ (2/√3)) := by
  sorry

end largest_C_inequality_l734_734735


namespace seashells_unbroken_l734_734428

theorem seashells_unbroken (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h1 : total_seashells = 6) 
  (h2 : broken_seashells = 4) 
  (h3 : unbroken_seashells = total_seashells - broken_seashells) :
  unbroken_seashells = 2 :=
by
  sorry

end seashells_unbroken_l734_734428


namespace rhombus_area_l734_734960

noncomputable def area_of_rhombus (s : ℝ) (θ : ℝ) : ℝ :=
  s * s * Math.sin θ

theorem rhombus_area :
  area_of_rhombus 4 (Math.pi / 4) = 8 * Real.sqrt 2 :=
by
  sorry

end rhombus_area_l734_734960


namespace vasya_drove_0_4_of_total_distance_l734_734643

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l734_734643


namespace zoe_total_cost_correct_l734_734565

theorem zoe_total_cost_correct :
  (6 * 0.5) + (6 * (1 + 2 * 0.75)) + (6 * 2 * 3) = 54 :=
by
  sorry

end zoe_total_cost_correct_l734_734565


namespace blocks_differ_in_3_ways_l734_734133

-- Let the total number of blocks be 128
def total_blocks : ℕ := 128

-- Define the number of distinct types for each attribute
def materials : ℕ := 2
def sizes : ℕ := 4
def colors : ℕ := 4
def shapes : ℕ := 4

-- Define the generating functions for the variables
def generating_function_material : ℕ := 1 + 1
def generating_function_size : ℕ := 1 + 3
def generating_function_color : ℕ := 1 + 3
def generating_function_shape : ℕ := 1 + 3

-- Calculate the number of ways to differ in exactly 3 ways from 'plastic medium red circle'
def num_ways_differ_3 : ℕ :=
  let gf := generating_function_material * generating_function_size *
            generating_function_color * generating_function_shape in
  45 -- Coefficient of x^3 in the expansion

-- The theorem stating the problem
theorem blocks_differ_in_3_ways : num_ways_differ_3 = 45 := by
  sorry

end blocks_differ_in_3_ways_l734_734133


namespace sin_double_angle_value_l734_734779

variable (θ : ℝ)

-- θ is an angle in the third quadrant
def theta_in_third_quadrant : Prop :=
  π < θ ∧ θ < 3 * π / 2

-- Given condition
def given_condition : Prop :=
  (Real.sin θ) ^ 4 + (Real.cos θ) ^ 4 = 5 / 9

-- Proof statement
theorem sin_double_angle_value (h1 : theta_in_third_quadrant θ) (h2 : given_condition θ) : 
    Real.sin (2 * θ) = 2 * Real.sqrt 2 / 3 :=
  by
  sorry

end sin_double_angle_value_l734_734779


namespace expected_value_paths_l734_734725

theorem expected_value_paths : 
  let a := 7
  let b := 242
  100 * a + b = 942 :=
by
  sorry

end expected_value_paths_l734_734725


namespace samantha_routes_l734_734905

/-
Samantha lives 3 blocks west and 2 blocks south of the southwest corner of City Park.
Her school is 3 blocks east and 3 blocks north of the northeast corner of City Park.
On school days, she bikes to the southwest corner of City Park, takes a diagonal path
through the park to the northeast corner, and then bikes to school. Prove that the
total number of different shortest routes she can take is 200.
-/

theorem samantha_routes : 
  let house_to_sw_park := (Nat.choose 5 2) in
  let ne_park_to_school := (Nat.choose 6 3) in
  house_to_sw_park * 1 * ne_park_to_school = 200 :=
by
  let house_to_sw_park := 10
  let ne_park_to_school := 20
  show 10 * 1 * 20 = 200
  sorry

end samantha_routes_l734_734905


namespace max_x_plus_y_l734_734304

theorem max_x_plus_y (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : x + y ≤ 2 * Real.sqrt (3) / 3 :=
sorry

end max_x_plus_y_l734_734304


namespace non_isosceles_triangles_in_decagon_l734_734176

theorem non_isosceles_triangles_in_decagon : 
  let total_triangles := Nat.choose 10 3 in
  let isosceles_triangles := 50 in
  total_triangles - isosceles_triangles = 70 :=
by
  sorry

end non_isosceles_triangles_in_decagon_l734_734176


namespace sqrt_product_l734_734694

theorem sqrt_product (a b c : ℝ) (h1 : a = real.sqrt 72) (h2 : b = real.sqrt 18) (h3 : c = real.sqrt 8) :
  a * b * c = 72 * real.sqrt 2 :=
by
  rw [h1, h2, h3]
  sorry

end sqrt_product_l734_734694


namespace min_omega_symmetry_l734_734501

theorem min_omega_symmetry :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * x + ω * (π / 2) + π / 3) = sin ((-ω) * x + ω * (π / 2) + π / 3)) →
  ω = 1 / 3 :=
by {
  sorry
}

end min_omega_symmetry_l734_734501


namespace remainder_when_150_divided_by_k_is_2_l734_734215

theorem remainder_when_150_divided_by_k_is_2
  (k : ℕ) (q : ℤ)
  (hk_pos : k > 0)
  (hk_condition : 120 = q * k^2 + 8) :
  150 % k = 2 :=
sorry

end remainder_when_150_divided_by_k_is_2_l734_734215


namespace second_child_sweets_l734_734613

theorem second_child_sweets :
  (mother_kept : ℕ) (children_sweets : ℕ) (eldest_got : ℕ) (youngest_got : ℕ) (second_child_got : ℕ) 
  (h1 : mother_kept = 27 / 3)
  (h2 : children_sweets = 27 - mother_kept)
  (h3 : eldest_got = 8)
  (h4 : youngest_got = eldest_got / 2)
  (h5 : second_child_got = children_sweets - eldest_got - youngest_got) :
  second_child_got = 6 :=
by
  sorry

end second_child_sweets_l734_734613


namespace number_of_grouping_methods_l734_734619

theorem number_of_grouping_methods : 
  let males := 5
  let females := 3
  let groups := 2
  let select_males := Nat.choose males groups
  let select_females := Nat.choose females groups
  let permute := Nat.factorial groups
  select_males * select_females * permute * permute = 60 :=
by 
  sorry

end number_of_grouping_methods_l734_734619


namespace solve_for_x_l734_734911

-- Define the given equation as a hypothesis
def equation (x : ℝ) : Prop :=
  0.05 * x - 0.09 * (25 - x) = 5.4

-- State the theorem that x = 54.6428571 satisfies the given equation
theorem solve_for_x : (x : ℝ) → equation x → x = 54.6428571 :=
by
  sorry

end solve_for_x_l734_734911


namespace reflection_of_A_over_y_axis_l734_734510

-- Definition to represent a point and its reflection over the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in (-x, y)

-- The point A to be reflected
def point_A : ℝ × ℝ := (1, 2)

-- The statement of the proof problem
theorem reflection_of_A_over_y_axis : reflect_y_axis point_A = (-1, 2) := 
  by
    sorry

end reflection_of_A_over_y_axis_l734_734510


namespace first_box_oranges_l734_734635

theorem first_box_oranges (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) = 120) : x = 11 :=
sorry

end first_box_oranges_l734_734635


namespace pascal_triangle_third_number_l734_734093

theorem pascal_triangle_third_number (n : ℕ) (h : n + 1 = 52) : (nat.choose n 2) = 1275 := by
  have h_n : n = 51 := by
    linarith
  rw [h_n]
  norm_num

end pascal_triangle_third_number_l734_734093


namespace third_number_in_pascals_triangle_row_51_l734_734088

theorem third_number_in_pascals_triangle_row_51 :
  let n := 51 in 
  ∃ result, result = (n * (n - 1)) / 2 ∧ result = 1275 :=
by
  let n := 51
  use (n * (n - 1)) / 2
  split
  . rfl
  . exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num; ring)
  sorry -- This 'sorry' is provided to formally conclude the directive


end third_number_in_pascals_triangle_row_51_l734_734088


namespace largest_unreachable_integer_l734_734973

theorem largest_unreachable_integer : ∃ n : ℕ, (¬ ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = n)
  ∧ ∀ m : ℕ, m > n → (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = m) := sorry

end largest_unreachable_integer_l734_734973


namespace pure_imaginary_value_l734_734824

theorem pure_imaginary_value (m : ℂ) (z : ℂ) (h : z = (m^2 - 1) + (m + 1) * complex.I) 
  (hz_imag : z.im ≠ 0) (hz_real : z.re = 0) : m = 1 := 
by sorry

end pure_imaginary_value_l734_734824


namespace polynomial_has_two_distinct_real_roots_l734_734729

-- Define the polynomial
def P (p x : ℝ) := x^4 + 2 * p * x^3 - x^2 + 2 * p * x + 1

-- The main theorem statement
theorem polynomial_has_two_distinct_real_roots (p : ℝ) :
  (--P(p, x) = 0)
/- has at least two distinct real roots if and only if -/
  (-((P(p, x), 0)).root_coun ≥ 2 ) ↔ p ∈ set.Icc (-3/4) (-1/4) :=
sorry

end polynomial_has_two_distinct_real_roots_l734_734729


namespace five_people_lineup_count_l734_734336

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l734_734336


namespace evaluate_expression_l734_734562

theorem evaluate_expression (m n : ℝ) (h : 4 * m - 4 + n = 2) : 
  (m * (-2)^2 - 2 * (-2) + n = 10) :=
by
  sorry

end evaluate_expression_l734_734562


namespace total_markers_l734_734994

/-- 
Connie has 2315 red markers and 1028 blue markers.
Prove that the total number of markers Connie has equals 3343.
-/
theorem total_markers (red_markers : ℕ) (blue_markers : ℕ) (h1 : red_markers = 2315) (h2 : blue_markers = 1028) : 
    red_markers + blue_markers = 3343 := 
by
  rw [h1, h2]
  norm_num
  sorry

end total_markers_l734_734994


namespace revenue_increase_is_44_percent_l734_734010

variable (P Q : ℝ)

def original_revenue := P * Q
def new_price := (1.80 : ℝ) * P
def new_quantity := (0.80 : ℝ) * Q
def new_revenue := new_price * new_quantity
def revenue_increase := new_revenue - original_revenue

theorem revenue_increase_is_44_percent :
  revenue_increase = 0.44 * original_revenue :=
by
  -- Compute and verify the revenue increase
  sorry

end revenue_increase_is_44_percent_l734_734010


namespace rental_plans_count_l734_734513

-- Define the number of large buses, medium buses, and the total number of people.
def num_large_buses := 42
def num_medium_buses := 25
def total_people := 1511

-- State the theorem to prove that there are exactly 2 valid rental plans.
theorem rental_plans_count (x y : ℕ) :
  (num_large_buses * x + num_medium_buses * y = total_people) →
  (∃! (x y : ℕ), num_large_buses * x + num_medium_buses * y = total_people) :=
by
  sorry

end rental_plans_count_l734_734513


namespace minimum_omega_l734_734495

theorem minimum_omega {ω : ℝ} (hω : ω > 0)
    (symmetry : ∃ k : ℤ, ∀ x : ℝ, 
      (sin (ω * x + ω * π / 2 + π / 3) = sin (-ω * x + ω * π / 2 + π / 3))) 
    : ω = 1 / 3 :=
by
  sorry

end minimum_omega_l734_734495


namespace siblings_of_Bob_l734_734287

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (age : Nat)

def children : List Child :=
  [ { name := "Bob", eyeColor := "Green", hairColor := "Black", age := 12 }
  , { name := "Anna", eyeColor := "Green", hairColor := "Black", age := 10 }
  , { name := "Mark", eyeColor := "Brown", hairColor := "Black", age := 14 }
  , { name := "Lucy", eyeColor := "Green", hairColor := "Brown", age := 12 }
  , { name := "Olivia", eyeColor := "Green", hairColor := "Brown", age := 8 }
  , { name := "John", eyeColor := "Green", hairColor := "Black", age := 14 }
  , { name := "Eve", eyeColor := "Brown", hairColor := "Black", age := 10 }
  , { name := "Sophie", eyeColor := "Brown", hairColor := "Brown", age := 12 }
  ]

def sharedCharacteristics (c1 c2 : Child) : Nat :=
  [c1.eyeColor = c2.eyeColor, c1.hairColor = c2.hairColor, c1.age = c2.age].count id

theorem siblings_of_Bob (sibling1 sibling2 : Child) :
  (List.get children 0).name = "Bob"
  → sharedCharacteristics (List.get children 0) sibling1 ≥ 2
  → sharedCharacteristics (List.get children 0) sibling2 ≥ 2
  → sibling1.name = "Anna"
  → sibling2.name = "John"
  := by
  sorry

end siblings_of_Bob_l734_734287


namespace shirt_pants_outfits_l734_734564

theorem shirt_pants_outfits
  (num_shirts : ℕ) (num_pants : ℕ) (num_formal_pants : ℕ) (num_casual_pants : ℕ) (num_assignee_shirts : ℕ) :
  num_shirts = 5 →
  num_pants = 6 →
  num_formal_pants = 3 →
  num_casual_pants = 3 →
  num_assignee_shirts = 3 →
  (num_casual_pants * num_shirts) + (num_formal_pants * num_assignee_shirts) = 24 :=
by
  intros h_shirts h_pants h_formal h_casual h_assignee
  sorry

end shirt_pants_outfits_l734_734564


namespace tom_age_ratio_l734_734525

theorem tom_age_ratio (T N : ℝ) (h1 : T - N = 3 * (T - 4 * N)) : T / N = 5.5 :=
by
  sorry

end tom_age_ratio_l734_734525


namespace count_lineup_excluding_youngest_l734_734350

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l734_734350


namespace smallest_of_powers_l734_734110

theorem smallest_of_powers :
  min (2^55) (min (3^44) (min (5^33) (6^22))) = 2^55 :=
by
  sorry

end smallest_of_powers_l734_734110


namespace count_lineup_excluding_youngest_l734_734348

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l734_734348


namespace original_price_of_laptop_l734_734192

theorem original_price_of_laptop
  (P : ℝ)
  (h_discount_total : (0.10 * P + 0.20 * (0.90 * P) + 0.35 * (0.72 * P) + 0.40 * (0.468 * P) + 0.25 * (0.2808 * P)) = 1500) :
  P ≈ 1899.64 :=    
sorry

end original_price_of_laptop_l734_734192


namespace alcohol_mixture_l734_734910

theorem alcohol_mixture:
  ∃ (x y z: ℝ), 
    0.10 * x + 0.30 * y + 0.50 * z = 157.5 ∧
    x + y + z = 450 ∧
    x = y ∧
    x = 112.5 ∧
    y = 112.5 ∧
    z = 225 :=
sorry

end alcohol_mixture_l734_734910


namespace day_100_M_minus_1_is_Tuesday_l734_734382

variable {M : ℕ}

-- Given conditions
def day_200_M_is_Monday (M : ℕ) : Prop :=
  ((200 % 7) = 6)

def day_300_M_plus_2_is_Monday (M : ℕ) : Prop :=
  ((300 % 7) = 6)

-- Statement to prove
theorem day_100_M_minus_1_is_Tuesday (M : ℕ) 
  (h1 : day_200_M_is_Monday M) 
  (h2 : day_300_M_plus_2_is_Monday M) 
  : (((100 + (365 - 200)) % 7 + 7 - 1) % 7 = 2) :=
sorry

end day_100_M_minus_1_is_Tuesday_l734_734382


namespace line_up_ways_l734_734358

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l734_734358


namespace period_of_y_l734_734552

def y (x : ℝ) : ℝ := 2 * Real.sin x + 3 * Real.cos x

theorem period_of_y : ∀ x, y(x + 2 * Real.pi) = y(x) := sorry

end period_of_y_l734_734552


namespace div_by_1963_iff_odd_l734_734912

-- Define the given condition and statement
theorem div_by_1963_iff_odd (n : ℕ) :
  (1963 ∣ (82^n + 454 * 69^n)) ↔ (n % 2 = 1) :=
sorry

end div_by_1963_iff_odd_l734_734912


namespace points_on_same_circle_l734_734422

open Complex

noncomputable def a1 : ℂ := sorry
noncomputable def a2 : ℂ := sorry
noncomputable def a3 : ℂ := sorry
noncomputable def a4 : ℂ := sorry
noncomputable def a5 : ℂ := sorry
noncomputable def q : ℂ := sorry
noncomputable def S : ℝ := sorry

axiom nonzero (a : ℂ) (h : a ≠ 0) : Prop

axiom conditions :
  nonzero a1 (by sorry) ∧ nonzero a2 (by sorry) ∧ nonzero a3 (by sorry) ∧ nonzero a4 (by sorry) ∧ nonzero a5 (by sorry) ∧
  a2 / a1 = q ∧
  a3 / a2 = q ∧
  a4 / a3 = q ∧
  a5 / a4 = q ∧
  a1 + a2 + a3 + a4 + a5 = 4 * (1 / a1 + 1 / a2 + 1 / a3 + 1 / a4 + 1 / a5) + S ∧
  abs S ≤ 2

theorem points_on_same_circle :
  ∃ r : ℝ, ∃ c : ℂ, (Complex.abs (a1 - c) = r ∧ Complex.abs (a2 - c) = r ∧ Complex.abs (a3 - c) = r ∧ Complex.abs (a4 - c) = r ∧ Complex.abs (a5 - c) = r) :=
by
  sorry

end points_on_same_circle_l734_734422


namespace min_omega_symmetry_l734_734503

theorem min_omega_symmetry :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * x + ω * (π / 2) + π / 3) = sin ((-ω) * x + ω * (π / 2) + π / 3)) →
  ω = 1 / 3 :=
by {
  sorry
}

end min_omega_symmetry_l734_734503


namespace minimum_omega_for_symmetric_curve_l734_734473

theorem minimum_omega_for_symmetric_curve (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 2) + π / 3) = sin (-ω * (x + π / 2) + π / 3)) ↔ ω = 1 / 3 :=
by
  sorry

end minimum_omega_for_symmetric_curve_l734_734473


namespace harmonic_le_geometric_harmonic_eq_geometric_iff_l734_734969

theorem harmonic_le_geometric {n : ℕ} {a : Fin n → ℝ}
  (h_pos : ∀ i, 0 < a i) :
  (n / (Finset.univ.sum (λ i, 1 / a i))) ≤ (Finset.univ.prod (λ i, a i))^(1 / (n : ℝ)) :=
by sorry

theorem harmonic_eq_geometric_iff {n : ℕ} {a : Fin n → ℝ}
  (h_pos : ∀ i, 0 < a i) :
  (n / (Finset.univ.sum (λ i, 1 / a i))) = (Finset.univ.prod (λ i, a i))^(1 / (n : ℝ))
  ↔ ∀ i j, a i = a j :=
by sorry

end harmonic_le_geometric_harmonic_eq_geometric_iff_l734_734969


namespace contrapositive_proof_l734_734046

variable {p q : Prop}

theorem contrapositive_proof : (p → q) ↔ (¬q → ¬p) :=
  by sorry

end contrapositive_proof_l734_734046


namespace multiply_97_103_eq_9991_l734_734706

theorem multiply_97_103_eq_9991 : (97 * 103 = 9991) :=
by
  have h1 : 97 = 100 - 3 := rfl
  have h2 : 103 = 100 + 3 := rfl
  calc
    97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
    ... = 100^2 - 3^2 : by rw [mul_add, add_mul, sub_mul, add_sub_cancel, sub_add_cancel]
    ... = 10000 - 9 : by norm_num
    ... = 9991 : by norm_num

end multiply_97_103_eq_9991_l734_734706


namespace product_of_number_and_its_digits_sum_l734_734836

theorem product_of_number_and_its_digits_sum :
  ∃ (n : ℕ), (n = 24 ∧ (n % 10) = ((n / 10) % 10) + 2) ∧ (n * (n % 10 + (n / 10) % 10) = 144) :=
by
  sorry

end product_of_number_and_its_digits_sum_l734_734836


namespace sector_area_l734_734257

-- Given conditions
variables {l r : ℝ}

-- Definitions (conditions from the problem)
def arc_length (l : ℝ) := l
def radius (r : ℝ) := r

-- Problem statement
theorem sector_area (l r : ℝ) : 
    (1 / 2) * l * r = (1 / 2) * l * r :=
by
  sorry

end sector_area_l734_734257


namespace doubled_dimensions_volume_l734_734627

theorem doubled_dimensions_volume (original_volume : ℝ) (length_factor width_factor height_factor : ℝ) 
  (h : original_volume = 3) 
  (hl : length_factor = 2)
  (hw : width_factor = 2)
  (hh : height_factor = 2) : 
  original_volume * length_factor * width_factor * height_factor = 24 :=
by
  sorry

end doubled_dimensions_volume_l734_734627


namespace solve_problem_l734_734410

noncomputable def f (a b c x : ℝ) := a * x^2 + b * x + c

variables {a b c x1 x2 : ℝ}

axiom f1 : f a b c 1 = -a / 2
axiom f2 : 3 * a > 2 * c
axiom f3 : 2 * c > 2 * b

theorem solve_problem :
  (a > 0 ∧ -3 < b / a ∧ b / a < -3 / 4) ∧
  ∃ x, 0 < x ∧ x < 2 ∧ f a b c x = 0 ∧
  ∃ x1 x2, f a b c x1 = 0 ∧ f a b c x2 = 0 ∧ sqrt 2 ≤ | x1 - x2 | ∧ | x1 - x2 | < sqrt 57 / 4 :=
sorry

end solve_problem_l734_734410


namespace complex_conjugate_l734_734878

theorem complex_conjugate (z : ℂ) (h : (1 + complex.i) * z = 4 - 2 * complex.i) : complex.conj z = 1 + 3 * complex.i :=
by
  sorry

end complex_conjugate_l734_734878


namespace max_of_expression_l734_734736

theorem max_of_expression (a b c : ℝ) (hbc : b > c) (hca : c > a) (ha : a > 0) (hb : b > 0) (hc : c > 0) (ha_nonzero : a ≠ 0) :
  ∃ (max_val : ℝ), max_val = 44 ∧ (∀ x, x = (2*a + b)^2 + (b - 2*c)^2 + (c - a)^2 → x ≤ max_val) := 
sorry

end max_of_expression_l734_734736


namespace mean_of_prime_numbers_l734_734200

open Nat

-- Define a list of numbers
def numbers : List ℕ := [24, 25, 27, 29, 31]

-- Define a predicate for checking prime numbers
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Extract prime numbers from the list
def prime_numbers (lst : List ℕ) : List ℕ :=
lst.filter is_prime

-- Calculate the arithmetic mean of a list of numbers
def arithmetic_mean (lst : List ℕ) : ℚ :=
(lst.sum : ℚ) / lst.length

theorem mean_of_prime_numbers :
  arithmetic_mean (prime_numbers numbers) = 30 := by
  sorry

end mean_of_prime_numbers_l734_734200


namespace find_f_of_3_div_4_l734_734291

def h (x : ℝ) : ℝ := 1 - 2 * x ^ 2

def f (y : ℝ) (hx : y ≠ 0): ℝ := (1 - 2 * (y / x) ^ 2) / (x ^ 2)

theorem find_f_of_3_div_4 : f (3/4) hx = 6 :=
sorry

end find_f_of_3_div_4_l734_734291


namespace minimum_omega_l734_734499

theorem minimum_omega {ω : ℝ} (hω : ω > 0)
    (symmetry : ∃ k : ℤ, ∀ x : ℝ, 
      (sin (ω * x + ω * π / 2 + π / 3) = sin (-ω * x + ω * π / 2 + π / 3))) 
    : ω = 1 / 3 :=
by
  sorry

end minimum_omega_l734_734499


namespace part_a_part_b_l734_734906

def canHaveThreeWheels : Prop :=
  Exists (λ (spokes : ℕ) (wheels : ℕ), wheels = 3 ∧ spokes >= 7 ∧ ∀ wheel, wheel <= 3)

def cannotHaveTwoWheels : Prop :=
  ¬ Exists (λ (spokes : ℕ) (wheels : ℕ), wheels = 2 ∧ spokes >= 7 ∧ ∀ wheel, wheel <= 3)

theorem part_a : canHaveThreeWheels :=
  sorry

theorem part_b : cannotHaveTwoWheels :=
  sorry

end part_a_part_b_l734_734906


namespace find_FC_l734_734224

theorem find_FC (DC CB AD : ℝ) (AB : ℝ) (ED : ℝ) :
  DC = 13 → CB = 9 → AB = (1/3) * AD → ED = (2/3) * AD → 
  let CA := CB + AB in 
  let FC := ED * CA / AD in
  FC = 40 / 3 :=
by
  intros hDC hCB hAB hED
  let DA := AD
  let CA := CB + AB
  let FC := ED * CA / AD
  sorry

end find_FC_l734_734224


namespace ratio_of_discretionary_income_l734_734193

variables (D : ℝ) (salary : ℝ)
constants (D_is_discretionary_income : D > 0) (salary_is_net : salary = 3300)

axiom h1 : 0.15 * D = 99
axiom h2 : salary = 3300

theorem ratio_of_discretionary_income (h1 : 0.15 * D = 99) (h2 : salary = 3300) :
  D / salary = 1 / 5 :=
by sorry

end ratio_of_discretionary_income_l734_734193


namespace abs_diff_solutions_l734_734086

theorem abs_diff_solutions : 
  ∃ x1 x2 : ℝ, |x1 - 3| = 15 ∧ |x2 - 3| = 15 ∧ x1 ≠ x2 ∧ (|x1 - x2| = 30) :=
by
  exists 18
  exists (-12)
  split
  { rw [Real.abs_of_nonneg (by norm_num)]
    exact rfl }
  split
  { rw [Real.abs_of_nonpos (by norm_num)]
    norm_num }
  split
  { norm_num }
  norm_num [Real.abs_sub]
  sorry

end abs_diff_solutions_l734_734086


namespace net_investment_change_l734_734132

def initial_investment : ℝ := 100
def first_year_increase (init : ℝ) : ℝ := init * 1.50
def second_year_decrease (value : ℝ) : ℝ := value * 0.70

theorem net_investment_change :
  second_year_decrease (first_year_increase initial_investment) - initial_investment = 5 :=
by
  -- This will be placeholder proof
  sorry

end net_investment_change_l734_734132


namespace koala_fiber_intake_l734_734392

theorem koala_fiber_intake (r a : ℝ) (hr : r = 0.20) (ha : a = 8) : (a / r) = 40 :=
by
  sorry

end koala_fiber_intake_l734_734392


namespace arithmetic_mean_first_2n_squares_l734_734167

theorem arithmetic_mean_first_2n_squares (n : ℕ) :
  (2 * n + 1) * (4 * n + 1) / 6 = ∑ i in finset.range (2 * n), (i + 1) ^ 2 / (2 * n) := 
sorry

end arithmetic_mean_first_2n_squares_l734_734167


namespace min_waiting_time_max_waiting_time_expected_waiting_time_l734_734991

open Nat

noncomputable def C : ℕ → ℕ → ℕ
| n, 0     => 1
| 0, k     => 0
| n+1, k+1 => C n k + C n (k+1)

def a := 1
def b := 5
def n := 5
def m := 3

def T_min := a * C (n - 1) 2 + m * n * a + b * C m 2
def T_max := a * C n 2 + b * m * n + b * C m 2
def E_T := C (n + m) 2 * (b * m + a * n) / (m + n)

theorem min_waiting_time : T_min = 40 := by
  sorry

theorem max_waiting_time : T_max = 100 := by
  sorry

theorem expected_waiting_time : E_T = 70 := by
  sorry

end min_waiting_time_max_waiting_time_expected_waiting_time_l734_734991


namespace savings_of_person_l734_734123

-- Definitions as given in the problem
def income := 18000
def ratio_income_expenditure := 5 / 4

-- Implied definitions based on the conditions and problem context
noncomputable def expenditure := income * (4/5)
noncomputable def savings := income - expenditure

-- Theorem statement
theorem savings_of_person : savings = 3600 :=
by
  -- Placeholder for proof
  sorry

end savings_of_person_l734_734123


namespace average_age_of_all_l734_734460

theorem average_age_of_all (students parents : ℕ) (student_avg parent_avg : ℚ) 
  (h_students: students = 40) 
  (h_student_avg: student_avg = 12) 
  (h_parents: parents = 60) 
  (h_parent_avg: parent_avg = 36)
  : (students * student_avg + parents * parent_avg) / (students + parents) = 26.4 :=
by
  sorry

end average_age_of_all_l734_734460


namespace vitya_older_than_masha_l734_734538

theorem vitya_older_than_masha :
  (∃ (days_in_month : ℕ) (total_pairs : ℕ) (favorable_pairs : ℕ)
     (p : ℚ),
    days_in_month = 30 ∧
    total_pairs = days_in_month * days_in_month ∧
    favorable_pairs = ∑ i in Finset.range(30), i ∧
    p = favorable_pairs / total_pairs ∧
    p = 29 / 60) :=
begin
  let days_in_month := 30,
  let total_pairs := days_in_month * days_in_month,
  let favorable_pairs := ∑ i in Finset.range(days_in_month), i,
  let p := favorable_pairs / total_pairs,
  use [days_in_month, total_pairs, favorable_pairs, p],
  split,
  { refl, },
  split,
  { refl, },
  split,
  { sorry, },
  split,
  { sorry, },
end

end vitya_older_than_masha_l734_734538


namespace number_of_valid_permutations_l734_734330

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l734_734330


namespace muffin_machine_completion_time_l734_734596

theorem muffin_machine_completion_time :
  let start_time := 9 * 60 -- minutes
  let partial_completion_time := (12 * 60) + 15 -- minutes
  let partial_duration := partial_completion_time - start_time
  let fraction_of_day := 1 / 4
  let total_duration := partial_duration / fraction_of_day
  start_time + total_duration = (22 * 60) := -- 10:00 PM in minutes
by
  sorry

end muffin_machine_completion_time_l734_734596


namespace bird_twigs_circle_l734_734597

theorem bird_twigs_circle (x : ℕ) :
  (2 * x = 6 * x - 48) → x = 12 :=
by { intro h, linarith, }

end bird_twigs_circle_l734_734597


namespace doubled_volume_l734_734630

theorem doubled_volume (V₀ V₁ : ℝ) (hV₀ : V₀ = 3) (h_double : V₁ = V₀ * 8) : V₁ = 24 :=
by 
  rw [hV₀] at h_double
  exact h_double

end doubled_volume_l734_734630


namespace find_angle_A_l734_734307

theorem find_angle_A (a b c : ℝ) (A : ℝ) (h : a^2 = b^2 - b * c + c^2) : A = 60 :=
sorry

end find_angle_A_l734_734307


namespace solution_exists_for_any_y_l734_734218

theorem solution_exists_for_any_y (z : ℝ) : (∀ y : ℝ, ∃ x : ℝ, x^2 + y^2 + 4*z^2 + 2*x*y*z - 9 = 0) ↔ |z| ≤ 3 / 2 := 
sorry

end solution_exists_for_any_y_l734_734218


namespace geometric_sequence_k_value_l734_734237

theorem geometric_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (hS : ∀ n, S n = k + 3^n)
  (h_geom : ∀ n, a (n+1) = S (n+1) - S n)
  (h_geo_seq : ∀ n, a (n+2) / a (n+1) = a (n+1) / a n) :
  k = -1 := by
  sorry

end geometric_sequence_k_value_l734_734237


namespace find_m_l734_734930

theorem find_m (m : ℝ) 
  (f g : ℝ → ℝ) 
  (x : ℝ) 
  (hf : f x = x^2 - 3 * x + m) 
  (hg : g x = x^2 - 3 * x + 5 * m) 
  (hx : x = 5) 
  (h_eq : 3 * f x = 2 * g x) :
  m = 10 / 7 := 
sorry

end find_m_l734_734930


namespace vitya_older_than_masha_probability_l734_734540

-- Define the problem conditions
def total_days_in_june : ℕ := 30
def total_possible_pairs : ℕ := total_days_in_june * total_days_in_june

-- Define the calculation of favorable pairs
def favorable_pairs : ℕ :=
  (finset.range (total_days_in_june)).sum (λ d_V, if d_V = 0 then 0 else (d_V))

-- Define the probability calculation
def probability_vitya_older_than_masha : ℚ :=
  favorable_pairs / total_possible_pairs

-- Statement of the proof problem
theorem vitya_older_than_masha_probability :
  probability_vitya_older_than_masha = 29 / 60 := 
by
  -- The proof is omitted for now
  sorry

end vitya_older_than_masha_probability_l734_734540


namespace vasya_fraction_is_0_4_l734_734650

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l734_734650


namespace lcm_of_three_numbers_l734_734052

theorem lcm_of_three_numbers :
  ∀ (a b c : ℕ) (hcf : ℕ), hcf = Nat.gcd (Nat.gcd a b) c → a = 136 → b = 144 → c = 168 → hcf = 8 →
  Nat.lcm (Nat.lcm a b) c = 411264 :=
by
  intros a b c hcf h1 h2 h3 h4
  rw [h2, h3, h4]
  sorry

end lcm_of_three_numbers_l734_734052


namespace total_seeds_in_watermelon_l734_734968

theorem total_seeds_in_watermelon :
  let slices := 40
  let black_seeds_per_slice := 20
  let white_seeds_per_slice := 20
  let total_black_seeds := black_seeds_per_slice * slices
  let total_white_seeds := white_seeds_per_slice * slices
  total_black_seeds + total_white_seeds = 1600 := by
  sorry

end total_seeds_in_watermelon_l734_734968


namespace JohnReadsIn90Minutes_l734_734390

-- Conditions from the problem
variable (MarkTime : ℕ := 180) -- Mark's reading time in minutes
variable (JohnSpeedMultiplier : ℕ := 2) -- John's speed multiplier with respect to Mark

-- Define the time it takes John to read the book
def JohnTime := MarkTime / JohnSpeedMultiplier

-- Statement: Prove that JohnTime equals 90 minutes.
theorem JohnReadsIn90Minutes : JohnTime = 90 := by
  unfold JohnTime
  rw [Nat.div_eq_of_lt (Nat.succ_pos')]
  sorry -- proof omitted

end JohnReadsIn90Minutes_l734_734390


namespace not_equilateral_triangle_intersection_l734_734317

-- Definition of a triangle being acute-angled and scalene
structure AcuteScaleneTriangle (A B C : Type) [IncidencePlane A B C] :=
(acute : ∀ (a b c : A), ∃ (h : Hit b c ∥ h.ftype b c == 60))
(scalene : ∀ (a b c : A), ∃ (h : Hit a b c ∥ h)))

-- Definitions of altitude, median, and angle bisector
structure Line (A B C : Type) [IncidencePlane A B C] :=
(Alt : A → C → Prop) -- Altitude from A to BC
(Med : B → C → Prop) -- Median from B to midpoint of AC
(AngBis : C → A → Prop) -- Angle bisector from C to AB

-- Condition that lines intersect in the respective points forming triangles
structure IntersectingPoints (A B C P Q R : Type) [IncidencePlane P Q R] :=
(interAH_BM : Line.Alt A C P ∧ Line.Med B Q ∧ Intersect A B C P)
(interAH_CL : Line.Alt A B R ∧ Line.AngBis C R ∧ Intersect A B C R)
(interBM_CL : Line.Med B C Q ∧ Line.AngBis C Q ∧ Intersect A B C Q)

-- Main statement to prove the intersection points can't form an equilateral triangle
theorem not_equilateral_triangle_intersection 
  (A B C P Q R : Type) [IncidencePlane A B C] [IncidencePlane P Q R]
  (h : AcuteScaleneTriangle A B C)
  (alt : Line.Alt A B)
  (med : Line.Med B Q)
  (bis : Line.AngBis C R)
  (int_pts : IntersectingPoints A B C P Q R) :
  ¬ EquilateralTriangle P Q R :=
sorry

end not_equilateral_triangle_intersection_l734_734317


namespace mean_and_median_change_l734_734162

-- Define initial and corrected data counts
def initial_counts := [25, 30, 20, 25, 20]
def corrected_counts := [25, 30, 25, 25, 20]

-- Define means and medians
def initial_mean : ℕ := (initial_counts.sum) / initial_counts.length
def corrected_mean : ℕ := (corrected_counts.sum) / corrected_counts.length
def initial_median : ℕ := initial_counts.insertsort.nth ((initial_counts.length / 2)).getOrElse 0
def corrected_median : ℕ := corrected_counts.insertsort.nth ((corrected_counts.length / 2)).getOrElse 0

-- State the theorem to prove
theorem mean_and_median_change :
  (corrected_mean = initial_mean + 1) ∧ (corrected_median = initial_median) := sorry

end mean_and_median_change_l734_734162


namespace repeated_pair_exists_l734_734399

theorem repeated_pair_exists (a : Fin 99 → Fin 10)
  (h1 : ∀ n : Fin 98, a n = 1 → a (n + 1) ≠ 2)
  (h2 : ∀ n : Fin 98, a n = 3 → a (n + 1) ≠ 4) :
  ∃ k l : Fin 98, k ≠ l ∧ a k = a l ∧ a (k + 1) = a (l + 1) :=
sorry

end repeated_pair_exists_l734_734399


namespace vasya_drove_0_4_of_total_distance_l734_734645

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l734_734645


namespace pasha_encoded_expression_l734_734892

theorem pasha_encoded_expression :
  2065 + 5 - 47 = 2023 :=
by
  sorry

end pasha_encoded_expression_l734_734892


namespace sculpture_cost_in_cny_l734_734012

-- Define the equivalence rates
def usd_to_nad : ℝ := 8
def usd_to_cny : ℝ := 8

-- Define the cost of the sculpture in Namibian dollars
def sculpture_cost_nad : ℝ := 160

-- Theorem: Given the conversion rates, the sculpture cost in Chinese yuan is 160
theorem sculpture_cost_in_cny : (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 160 :=
by sorry

end sculpture_cost_in_cny_l734_734012


namespace mul_97_103_l734_734701

theorem mul_97_103 : (97:ℤ) = 100 - 3 → (103:ℤ) = 100 + 3 → 97 * 103 = 9991 := by
  intros h1 h2
  sorry

end mul_97_103_l734_734701


namespace tetrahedral_inequality_l734_734131

theorem tetrahedral_inequality (n : ℕ) (b : ℝ) (a b : Fin n → ℝ) 
  (h_constant_sum : (Finset.univ.sum (λ i, (a i + b i))) = k) :
  ∃ h : ℝ, (Finset.univ.sum (λ i, sqrt ((a i)^2 - (b / 2)^2))) ≤ n * sqrt ((k / (2 * n))^2 - (b / 2)^2) := sorry

end tetrahedral_inequality_l734_734131


namespace domain_of_f_l734_734922

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - real.log x / real.log 2)

theorem domain_of_f : {x : ℝ | 0 < x ∧ x ≤ 2} = {x | f x ≥ 0} :=
by 
  sorry

end domain_of_f_l734_734922


namespace number_of_valid_permutations_l734_734329

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l734_734329


namespace omega_min_value_l734_734486

theorem omega_min_value (ω : ℝ) (hω : ω > 0)
    (hSymmetry : ∀ x : ℝ, sin (ω * x + ω * π / 2 + π / 3) = sin (ω * -x + ω * π / 2 + π / 3)) :
    ω = 1 / 3 :=
begin
  sorry
end

end omega_min_value_l734_734486


namespace sector_radius_l734_734784

theorem sector_radius (r : ℝ) (h1 : r > 0) 
  (h2 : ∀ (l : ℝ), l = r → 
    (3 * r) / (1 / 2 * r^2) = 2) : r = 3 := 
sorry

end sector_radius_l734_734784


namespace range_of_x_coordinate_of_Q_l734_734761

def Point := ℝ × ℝ

def parabola (P : Point) : Prop :=
  P.2 = P.1 ^ 2

def vector (P Q : Point) : Point :=
  (Q.1 - P.1, Q.2 - P.2)

def dot_product (u v : Point) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (P Q R : Point) : Prop :=
  dot_product (vector P Q) (vector P R) = 0

theorem range_of_x_coordinate_of_Q:
  ∀ (A P Q: Point), 
    A = (-1, 1) →
    parabola P →
    parabola Q →
    perpendicular P A Q →
    (Q.1 ≤ -3 ∨ Q.1 ≥ 1) :=
by
  intros A P Q hA hParabP hParabQ hPerp
  sorry

end range_of_x_coordinate_of_Q_l734_734761


namespace sum_q_t_8_eq_128_l734_734406

def T : Set (Fin 8 → ℕ) := {f | ∀ n, n < 8 → (f n = 0 ∨ f n = 1)}

-- Defining the polynomial q_t corresponding to each tuple t
def q_t (t : Fin 8 → ℕ) : Polynomial ℕ :=
  sorry -- The polynomial construction mapping to each tuple will go here

-- Sum of q_t(8) over all t in T equals 128
theorem sum_q_t_8_eq_128 : 
  ∑ t in T, q_t t 8 = 128 :=
by
  sorry -- The proof steps will go here

end sum_q_t_8_eq_128_l734_734406


namespace image_of_2_in_set_B_l734_734593

theorem image_of_2_in_set_B (f : ℤ → ℤ) (h : ∀ x, f x = 2 * x + 1) : f 2 = 5 :=
by
  apply h

end image_of_2_in_set_B_l734_734593


namespace strictly_increasing_intervals_l734_734718

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem strictly_increasing_intervals :
  { x : ℝ | ∃ U : set ℝ, U ⊆ set.Icc (f x (x ≤ -1/3 ∨ 1 ≤ x) } ≠ ∅ := by
  sorry

end strictly_increasing_intervals_l734_734718


namespace nonnegative_rational_function_solve_l734_734742

theorem nonnegative_rational_function_solve :
  {x : ℝ | 2 * x - 5 * x^2 + 6 * x^3 ≥ 0} ∩ {x : ℝ | 9 - x^3 ≠ 0} = set.Ico 0 3 :=
sorry

end nonnegative_rational_function_solve_l734_734742


namespace smallest_prime_dividing_polynomial_l734_734294

theorem smallest_prime_dividing_polynomial :
  ∃ p : ℕ, Nat.Prime p ∧ 
           (∃ n : ℤ, ↑p ∣ n^2 + 5 * n + 23) ∧ 
           (∀ q : ℕ, Nat.Prime q ∧ (∃ m : ℤ, ↑q ∣ m^2 + 5 * m + 23) → p ≤ q) :=
begin
  use 17,
  split,
  { exact Nat.Prime.mk 17 (by norm_num) (by norm_num) },
  split,
  { use -2,
    norm_num },
  { intros q hq,
    by_cases h : q = 17,
    { rw h },
    { have hne: q ≠ 0 := Nat.Prime.ne_zero hq.1,
      have := Nat.Prime.ne_one hq.1,
      have : q ∣ 67 := sorry, -- the contradiction or argument that q does not divide the polynomial
      exact sorry } -- would involve more concrete arguments, here we use the fact-only
  }
end

end smallest_prime_dividing_polynomial_l734_734294


namespace general_formula_a_sum_first_n_T_l734_734764

-- Define the sequence {a_n}
def sequence_a (n : ℕ) : ℕ → ℝ := λ n, (1 / (2 ^ n))

-- Define the partial sum S_n
def partial_sum_S (n : ℕ) : ℝ := (1 : ℝ) - sequence_a n

-- Define the sequence {b_n}
def sequence_b (n : ℕ) : ℕ → ℝ := λ n, ∑ k in range (n + 1), log (sequence_a k) / log 4

-- Define the sequence {1/a_n + 1/b_n}
def sequence_c (n : ℕ) : ℕ → ℝ := λ n, (1 / sequence_a n) + (1 / sequence_b n)

-- Define the sum of the first n terms, T_n
def sum_T (n : ℕ) : ℝ :=
  ∑ k in range (n + 1), sequence_c k

-- Prove the general formula for the sequence {a_n}
theorem general_formula_a (n : ℕ) : sequence_a n = (1 / (2^n)) := by
  sorry

-- Prove the sum of the first n terms T_n for the sequence {1/a_n + 1/b_n}
theorem sum_first_n_T (n : ℕ) : sum_T n = (2 ^ (n + 1)) + (4 / (n + 1)) - 6 := by
  sorry

end general_formula_a_sum_first_n_T_l734_734764


namespace pets_after_one_month_l734_734857

def initial_dogs := 30
def initial_cats := 28
def initial_lizards := 20
def adoption_rate_dogs := 0.5
def adoption_rate_cats := 0.25
def adoption_rate_lizards := 0.2
def new_pets := 13

theorem pets_after_one_month :
  (initial_dogs - (initial_dogs * adoption_rate_dogs) +
   initial_cats - (initial_cats * adoption_rate_cats) +
   initial_lizards - (initial_lizards * adoption_rate_lizards) +
   new_pets) = 65 :=
by 
  -- proof goes here
  sorry

end pets_after_one_month_l734_734857


namespace equal_areas_BIF_CIE_l734_734765

noncomputable theory
open Real

variables {A B C I E F : Point}
variables (h1 : Incenter I A B C)
variables (h2 : OnRay B I E)
variables (h3 : OnRay C I F)
variables (h4 : Dist A I = Dist A E)
variables (h5 : Dist A I = Dist A F)

theorem equal_areas_BIF_CIE :
  area (△ B I F) = area (△ C I E) :=
sorry

end equal_areas_BIF_CIE_l734_734765


namespace shaded_area_percentage_l734_734841

-- Define the given conditions
def square_area := 6 * 6
def shaded_area_left := (1 / 2) * 2 * 6
def shaded_area_right := (1 / 2) * 4 * 6
def total_shaded_area := shaded_area_left + shaded_area_right

-- State the theorem
theorem shaded_area_percentage : (total_shaded_area / square_area) * 100 = 50 := by
  sorry

end shaded_area_percentage_l734_734841


namespace inequality_one_solution_inequality_two_solution_l734_734454

theorem inequality_one_solution (x : ℝ) :
  (x + 1)^2 + 3 * (x + 1) - 4 > 0 ↔ (x < -5 ∨ x > 0) :=
sorry

theorem inequality_two_solution (x : ℝ) :
  x^4 - 2 * x^2 + 1 > x^2 - 1 ↔ (x < -real.sqrt 2 ∨ (-1 < x ∧ x < 1) ∨ x > real.sqrt 2) :=
sorry

end inequality_one_solution_inequality_two_solution_l734_734454


namespace cos_angle_AND_l734_734129

-- Define points in a regular tetrahedron
variables {A B C D N : Point}

-- Given conditions
def regular_tetrahedron (A B C D : Point) : Prop := 
  ∀ (E F : Point), dist A B = dist A C ∧ dist A B = dist A D ∧ dist B C = dist B D ∧ dist C D = dist B C

def midpoint (N B C : Point) : Prop := 
  N = (B + C) / 2

-- The theorem to prove
theorem cos_angle_AND {A B C D N : Point} 
  (h_tetra : regular_tetrahedron A B C D) 
  (h_mid : midpoint N B C) : 
  cos (∠ (A - N) (N - D)) = 2 / 3 :=
sorry

end cos_angle_AND_l734_734129


namespace g_increasing_l734_734254

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2

noncomputable def translated_f (x : ℝ) : ℝ := (Real.cos (x - Real.pi / 4))^2

noncomputable def g (x : ℝ) : ℝ := -1/2 * Real.sin(2 * x - 2 * Real.pi / 3) + 1/2

def increasing_intervals (k : ℤ) : Set ℝ := 
  {x | x ∈ Icc (-5 * Real.pi / 12 + k * Real.pi) (Real.pi / 12 + k * Real.pi)}

theorem g_increasing  :
  ∀ k : ℤ, ∀ x : ℝ, x ∈ increasing_intervals k → 0 < g' x :=
sorry

end g_increasing_l734_734254


namespace two_prime_numbers_l734_734966

-- Definitions from the conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def has_sum_and_product (a b : ℕ) (sum prod : ℕ) : Prop :=
  a + b = sum ∧ a * b = prod

-- The math proof problem statement
theorem two_prime_numbers (a b : ℕ) :
  is_prime a → is_prime b → has_sum_and_product a b 10 21 → (a = 3 ∧ b = 7) ∨ (a = 7 ∧ b = 3) :=
by
  intros
  sorry

end two_prime_numbers_l734_734966


namespace total_flowers_l734_734884

def pieces (f : String) : Nat :=
  if f == "roses" ∨ f == "lilies" ∨ f == "sunflowers" ∨ f == "daisies" then 40 else 0

theorem total_flowers : 
  pieces "roses" + pieces "lilies" + pieces "sunflowers" + pieces "daisies" = 160 := 
by
  sorry


end total_flowers_l734_734884


namespace sum_of_digits_B_l734_734270

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_B (n : ℕ) (h : n = 4444^4444) : digit_sum (digit_sum (digit_sum n)) = 7 :=
by
  sorry

end sum_of_digits_B_l734_734270


namespace area_of_triangle_XYZ_l734_734633

-- Definitions for the conditions
def base : ℝ := 4  -- Base of the triangle XZ in kilometers
def height : ℝ := 2  -- Height from point Y to the line XZ in kilometers

-- The theorem statement for the proof problem
theorem area_of_triangle_XYZ : (1 / 2) * base * height = 4 := by
  sorry

end area_of_triangle_XYZ_l734_734633


namespace probability_each_delegate_next_to_another_country_l734_734077

-- Given definitions
def num_delegates : ℕ := 12
def delegates_per_country : ℕ := 3
def num_countries : ℕ := 4

-- Main statement
theorem probability_each_delegate_next_to_another_country :
  let total_ways := 12! / (3! ^ 4)
  let correct_ways := 6570576
  let probability := (correct_ways : ℚ) / total_ways
  probability = 163 / 165 :=
by
  sorry

end probability_each_delegate_next_to_another_country_l734_734077


namespace ratio_of_area_to_perimeter_squared_l734_734974

noncomputable def equilateral_triangle := 
  { side_length : ℝ // side_length = 10 }

def area (tri : equilateral_triangle) : ℝ := 
  (Real.sqrt 3 / 4) * tri.val ^ 2

def perimeter (tri : equilateral_triangle) : ℝ :=
  3 * tri.val

def perimeter_squared (tri : equilateral_triangle) : ℝ :=
  perimeter tri ^ 2

def area_per_perimeter_squared (tri : equilateral_triangle) : ℝ :=
  area tri / perimeter_squared tri

theorem ratio_of_area_to_perimeter_squared (tri : equilateral_triangle) :
  area_per_perimeter_squared tri = Real.sqrt 3 / 36 :=
sorry

end ratio_of_area_to_perimeter_squared_l734_734974


namespace printer_z_time_l734_734438

theorem printer_z_time (t_z : ℝ)
  (hx : (∀ (p : ℝ), p = 16))
  (hy : (∀ (q : ℝ), q = 12))
  (ratio : (16 / (1 /  ((1 / 12) + (1 / t_z)))) = 10 / 3) :
  t_z = 8 := by
  sorry

end printer_z_time_l734_734438


namespace nine_point_circle_midpoint_of_HD_l734_734396

open EuclideanGeometry

variables {A B C E D H : Point}
variable {circumcircleABC : Circle}
variable {ninePointCircleABC : Circle}

-- Given that AE is a diameter of the circumcircle of triangle ABC
axiom AE_diameter (h: IsDiameter circumcircleABC A E) : E = Antipode A circumcircleABC
-- H is the orthocenter of triangle ABC
axiom H_orthocenter : is_Orthocenter H A B C
-- EH intersects the circumcircle at D
axiom EH_intersects_at_D (h : Line E H = Line.extension E H) (hD : D ∈ circumcircleABC) : D ∈ Line E H

-- Propositional statement
theorem nine_point_circle_midpoint_of_HD (h_midpoint_HD: is_midpoint (Midpoint H D) H D) :
  Midpoint H D ∈ ninePointCircleABC := 
sorry

end nine_point_circle_midpoint_of_HD_l734_734396


namespace jessica_total_payment_l734_734387

-- Definitions based on the conditions
def basic_cable_cost : Nat := 15
def movie_channels_cost : Nat := 12
def sports_channels_cost : Nat := movie_channels_cost - 3

-- Definition of the total monthly payment given Jessica adds both movie and sports channels
def total_monthly_payment : Nat :=
  basic_cable_cost + (movie_channels_cost + sports_channels_cost)

-- The proof statement
theorem jessica_total_payment : total_monthly_payment = 36 :=
by
  -- skip the proof
  sorry

end jessica_total_payment_l734_734387


namespace recover_vertex_l734_734432

axiom hexagon_vertices (x1 x2 x3 x4 x5 x6 : ℝ) : 
  let A := x1 + x6 in
  let B := x1 + x2 in
  let C := x2 + x3 in
  let D := x3 + x4 in
  let E := x4 + x5 in
  let F := x5 + x6 in
  A + C + E = B + D + F

theorem recover_vertex (x1 x2 x3 x4 x5 x6 : ℝ) : 
  let A := x1 + x6 in
  let B := x1 + x2 in
  let C := x2 + x3 in
  let D := x3 + x4 in
  let E := x4 + x5 in
  let F := x5 + x6 in
  hexagon_vertices x1 x2 x3 x4 x5 x6 →
  A = (B + D + F) - (C + E) :=
by
  intros h
  rw [show A = x1 + x6, by unfold A ]
  rw [show B = x1 + x2, by unfold B ]
  rw [show C = x2 + x3, by unfold C ]
  rw [show D = x3 + x4, by unfold D ]
  rw [show E = x4 + x5, by unfold E ]
  rw [show F = x5 + x6, by unfold F ]
  exact sorry

end recover_vertex_l734_734432


namespace vitya_older_than_masha_probability_l734_734539

-- Define the problem conditions
def total_days_in_june : ℕ := 30
def total_possible_pairs : ℕ := total_days_in_june * total_days_in_june

-- Define the calculation of favorable pairs
def favorable_pairs : ℕ :=
  (finset.range (total_days_in_june)).sum (λ d_V, if d_V = 0 then 0 else (d_V))

-- Define the probability calculation
def probability_vitya_older_than_masha : ℚ :=
  favorable_pairs / total_possible_pairs

-- Statement of the proof problem
theorem vitya_older_than_masha_probability :
  probability_vitya_older_than_masha = 29 / 60 := 
by
  -- The proof is omitted for now
  sorry

end vitya_older_than_masha_probability_l734_734539


namespace count_valid_arrangements_l734_734323

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l734_734323


namespace Vasya_distance_fraction_l734_734660

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l734_734660


namespace angle_rotation_BNC_to_CMA_l734_734020

-- Mathematical definitions and conditions
variables (B N C M A : Type)
variable [inner_product_space ℝ B]
variables {BC AC : ℝ} -- lengths of BC and AC
variables {hBNC hCMA : ℝ} -- heights of triangles BNC and CMA

-- Conditions
def BNC_CMA_equal_heights (h1 : ℝ) (h2 : ℝ) (BCeq : BC = AC) : Prop :=
  h1 = h2 ∧ BC = AC

-- The main theorem
theorem angle_rotation_BNC_to_CMA
  (h1 h2 : ℝ)
  (h_eq : BNC_CMA_equal_heights h1 h2 BCeq) : 
  ∃ θ : ℝ, θ = 120 :=
by
  sorry

end angle_rotation_BNC_to_CMA_l734_734020


namespace bus_speed_excluding_stoppages_l734_734197

theorem bus_speed_excluding_stoppages (S : ℝ) (h₀ : 0 < S) (h₁ : 36 = (2/3) * S) : S = 54 :=
by 
  sorry

end bus_speed_excluding_stoppages_l734_734197


namespace exists_divisible_term_l734_734400

def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else seq (n - 1) + seq (nat.floor (real.sqrt (n - 1)))

theorem exists_divisible_term (k : ℕ) (hk : k > 0) :
  ∃ i, k ∣ seq i :=
  by
  sorry

end exists_divisible_term_l734_734400


namespace minimum_omega_l734_734488

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l734_734488


namespace log_of_a_l734_734794

noncomputable def log_base_2 := real.logb 2

theorem log_of_a
  (a x1 x2 x3 : ℝ)
  (h1 : 0 ≤ x2) 
  (h2 : x2 ≤ x3)
  (h3 : x1 < x2 ∧ x2 < x3)
  (H1 : x1 + x2 = π)
  (H2 : x2 + x3 = 3 * π)
  (H3 : x2^2 = x1 * x3)
  (a_def : a = real.sin (3 * π / 4)) :
  log_base_2 a = -1 / 2 :=
by {
  sorry
}

end log_of_a_l734_734794


namespace arithmetic_square_root_of_4_l734_734083

theorem arithmetic_square_root_of_4 : ∃ x : ℝ, x^2 = 4 ∧ x ≥ 0 ∧ x = 2 :=
begin
  use 2,
  split,
  { norm_num },
  split,
  { norm_num },
  { refl }
end

end arithmetic_square_root_of_4_l734_734083


namespace xy_relationship_l734_734177

theorem xy_relationship : 
  (∀ x y, (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 9) ∨ (x = 4 ∧ y = 16) ∨ (x = 5 ∧ y = 25) 
  → y = x * x) :=
by {
  sorry
}

end xy_relationship_l734_734177


namespace fg_of_2_eq_36_l734_734289

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 4*x - 2

theorem fg_of_2_eq_36 : f(g(2)) = 36 := by
  sorry

end fg_of_2_eq_36_l734_734289


namespace minimum_omega_l734_734490

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l734_734490


namespace flour_baking_soda_ratio_correct_l734_734377

def sugar_flour_ratio (S F : ℕ) : Prop := S / 5 = F / 4
def modified_flour_baking_soda_ratio (F B : ℕ) : Prop := F / (B + 60) = 8
def flour_baking_soda_ratio (F B : ℕ) : ℕ := F / B

theorem flour_baking_soda_ratio_correct :
  ∀ (S F B : ℕ), sugar_flour_ratio S F → modified_flour_baking_soda_ratio F B → S = 3000 → flour_baking_soda_ratio F B = 10 :=
begin
  intros S F B h1 h2 h3,
  sorry
end

end flour_baking_soda_ratio_correct_l734_734377


namespace face_opposite_orange_is_azure_l734_734885

-- Define the colors
inductive Color
| R | B | P | Y | G | W | Z | M | O

open Color

-- Define the condition that hinges together the squares to form a cube
def valid_configuration : (Color → Color) → Prop :=
  λ f, (f O = Z) ∧ 
       (f R ≠ f B) ∧
       (f W ≠ O) ∧
       (f P ≠ Z) ∧ 
       (f G ≠ B) -- Other conditions to satisfy cube properties can be added as required

theorem face_opposite_orange_is_azure (f : Color → Color) (h : valid_configuration f) : 
  f O = Z :=
by
  -- Placeholder for the actual proof
  sorry

end face_opposite_orange_is_azure_l734_734885


namespace count_valid_arrangements_l734_734321

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l734_734321


namespace ratio_of_donated_clothes_is_3_to_1_l734_734156

/-- Define the initial number of clothes Amara had. -/
def initial_clothes : ℕ := 100

/-- Define the number of clothes donated to the first orphanage home. -/
def donated_first : ℕ := 5

/-- Define the number of clothes thrown away. -/
def thrown_away : ℕ := 15

/-- Define the remaining number of clothes Amara has. -/
def remaining_clothes : ℕ := 65

/-- Total number of clothes no longer with Amara -/
def total_no_longer_with (initial remaining donated_first thrown_away : ℕ) : ℕ :=
  initial - remaining

/-- Number of clothes donated to the second orphanage home -/
def donated_second (total_no_longer_with donated_first thrown_away : ℕ) : ℕ :=
  total_no_longer_with - donated_first - thrown_away

/-- The ratio of the number of clothes donated to the second orphanage home 
    and the first orphanage home is equal to 3:1. -/
theorem ratio_of_donated_clothes_is_3_to_1 :
  let total := total_no_longer_with initial_clothes remaining_clothes donated_first thrown_away,
      second := donated_second total donated_first thrown_away in
    second / donated_first = 3 :=
by 
  -- Proof is omitted
  sorry

end ratio_of_donated_clothes_is_3_to_1_l734_734156


namespace population_correct_individual_correct_sample_correct_sample_size_correct_l734_734524

-- Definitions based on the problem conditions
def Population : Type := {s : String // s = "all seventh-grade students in the city"}
def Individual : Type := {s : String // s = "each seventh-grade student in the city"}
def Sample : Type := {s : String // s = "the 500 students that were drawn"}
def SampleSize : ℕ := 500

-- Prove given conditions
theorem population_correct (p : Population) : p.1 = "all seventh-grade students in the city" :=
by sorry

theorem individual_correct (i : Individual) : i.1 = "each seventh-grade student in the city" :=
by sorry

theorem sample_correct (s : Sample) : s.1 = "the 500 students that were drawn" :=
by sorry

theorem sample_size_correct : SampleSize = 500 :=
by sorry

end population_correct_individual_correct_sample_correct_sample_size_correct_l734_734524


namespace students_not_playing_any_sport_l734_734311

variable (Total Students Soccer Players Volleyball Players Exactly One Sport : ℕ)
variable (h1: Total = 40)
variable (h2: Soccer Players = 20)
variable (h3: Volleyball Players = 19)
variable (h4: Exactly One Sport = 15)

theorem students_not_playing_any_sport (h1: Total = 40) (h2: Soccer Players = 20) (h3: Volleyball Players = 19) (h4: Exactly One Sport = 15) : 
  ∃ StudentsNotPlayingAnySport, StudentsNotPlayingAnySport = 13 :=
by
  sorry

end students_not_playing_any_sport_l734_734311


namespace ab2_ac2_sum_l734_734394

theorem ab2_ac2_sum 
  (A B C E F G : Type) 
  [triangle ABC] 
  [area ABC = 5] 
  [BC = 10]
  [midpoint E AC]
  [midpoint F AB]
  [intersect BE CF G]
  [cyclic_quad A E G F] :
  AB^2 + AC^2 = 200 := 
  sorry

end ab2_ac2_sum_l734_734394


namespace probability_factor_less_than_eight_l734_734087

theorem probability_factor_less_than_eight (n : ℕ) (h72 : n = 72) :
  (∃ k < 8, k ∣ n) →
  (∃ p q, p/q = 5/12) :=
by
  sorry

end probability_factor_less_than_eight_l734_734087


namespace train_speed_kmph_l734_734147

theorem train_speed_kmph (length : ℝ) (time : ℝ) (speed_conversion : ℝ) (speed_kmph : ℝ) :
  length = 100.008 → time = 4 → speed_conversion = 3.6 →
  speed_kmph = (length / time) * speed_conversion → speed_kmph = 90.0072 :=
by
  sorry

end train_speed_kmph_l734_734147


namespace hexagon_perimeter_is_24_l734_734370

-- Conditions given in the problem
def AB : ℝ := 3
def EF : ℝ := 3
def BE : ℝ := 4
def AF : ℝ := 4
def CD : ℝ := 5
def DF : ℝ := 5

-- Statement to show that the perimeter is 24 units
theorem hexagon_perimeter_is_24 :
  AB + BE + CD + DF + EF + AF = 24 :=
by
  sorry

end hexagon_perimeter_is_24_l734_734370


namespace jam_bought_2_boxes_of_popcorn_l734_734073

theorem jam_bought_2_boxes_of_popcorn (P : ℝ) : 
  (let mitch_expenses := 3 * 7 in
   let jay_expenses := 3 * 3 in
   mitch_expenses + jay_expenses + 1.5 * P = 33) ↔ P = 2 :=
by
  let mitch_expenses := 3 * 7
  let jay_expenses := 3 * 3
  have h : mitch_expenses + jay_expenses + 1.5 * P = 33 ↔ 1.5 * P = 3 :=
    by
      calc
        (mitch_expenses + jay_expenses + 1.5 * P = 33) 
          ↔ (21 + 9 + 1.5 * P = 33) : by rw [←mitch_expenses, ←jay_expenses]
          ... ↔ (30 + 1.5 * P = 33) : by norm_num
          ... ↔ (1.5 * P = 3) : by linarith
  exact h.trans (by rw div_eq_iff_mul_eq; norm_num)

end jam_bought_2_boxes_of_popcorn_l734_734073


namespace period_of_trig_sum_l734_734549

theorem period_of_trig_sum : ∀ x : ℝ, 2 * Real.sin x + 3 * Real.cos x = 2 * Real.sin (x + 2 * Real.pi) + 3 * Real.cos (x + 2 * Real.pi) := 
sorry

end period_of_trig_sum_l734_734549


namespace proof_exactly_3_hits_proof_expected_score_l734_734255

noncomputable def probability_exactly_3_hits : ℚ :=
  (nat.choose 10 3) * (0.5 ^ 3) * (0.5 ^ (10 - 3))

noncomputable def expected_score : ℚ :=
  10 * 0.5 * 2

theorem proof_exactly_3_hits :
  probability_exactly_3_hits = 15 / 128 :=
by
  sorry

theorem proof_expected_score :
  expected_score = 10 :=
by
  sorry

end proof_exactly_3_hits_proof_expected_score_l734_734255


namespace circle_eq_focus_l734_734202

theorem circle_eq_focus (a b e : ℝ) 
  (h1 : a^2 = 3) 
  (h2 : b^2 = 1) 
  (h3 : e = Real.sqrt (1 + a^2 / b^2)) 
  (h4 : 3*f = e*e) :
  (Exists f : ℝ, (f = 2) ∧ circle_eq1 : x^2 + (y - 2)^2 = 4 ∨ circle_eq2 : x^2 + (y + 2)^2 = 4) :=
by
sorry

end circle_eq_focus_l734_734202


namespace vector_sum_parallel_to_y_axis_l734_734130

open Real

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (-x, x^2)

-- Define the vector sum a + b
def vector_sum (x : ℝ) : ℝ × ℝ := (vector_a x).1 + (vector_b x).1, (vector_a x).2 + (vector_b x).2

-- Define a predicate for being parallel to the y-axis
def is_parallel_to_y_axis (v : ℝ × ℝ) : Prop := v.1 = 0 ∧ v.2 ≠ 0

-- Main theorem statement
theorem vector_sum_parallel_to_y_axis (x : ℝ) :
  is_parallel_to_y_axis (vector_sum x) :=
sorry

end vector_sum_parallel_to_y_axis_l734_734130


namespace vitya_older_than_masha_probability_l734_734541

-- Define the problem conditions
def total_days_in_june : ℕ := 30
def total_possible_pairs : ℕ := total_days_in_june * total_days_in_june

-- Define the calculation of favorable pairs
def favorable_pairs : ℕ :=
  (finset.range (total_days_in_june)).sum (λ d_V, if d_V = 0 then 0 else (d_V))

-- Define the probability calculation
def probability_vitya_older_than_masha : ℚ :=
  favorable_pairs / total_possible_pairs

-- Statement of the proof problem
theorem vitya_older_than_masha_probability :
  probability_vitya_older_than_masha = 29 / 60 := 
by
  -- The proof is omitted for now
  sorry

end vitya_older_than_masha_probability_l734_734541


namespace correct_choice_l734_734242

def proposition_p : Prop := ∀ (x : ℝ), 2^x > x^2
def proposition_q : Prop := ∃ (x_0 : ℝ), x_0 - 2 > 0

theorem correct_choice : ¬proposition_p ∧ proposition_q :=
by
  sorry

end correct_choice_l734_734242


namespace dihedral_angle_formula_l734_734920

theorem dihedral_angle_formula (n : ℕ) (α : ℝ) (h : ℝ > 0) (a : ℝ > 0) 
  (S : ℝ) (cos_φ2 : ℝ) (sin_α : ℝ := real.sin α) (sin_πn : ℝ := real.sin (real.pi / n)) :
  S * cos_φ2 = sin_α * (a * h * sin_πn / 2) → 
  cos_φ2 = sin_α * sin_πn :=
sorry

end dihedral_angle_formula_l734_734920


namespace angle_B_lt_pi_div_two_l734_734594

theorem angle_B_lt_pi_div_two 
  (a b c : ℝ) (B : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : B = π / 2 - B)
  (h5 : 2 / b = 1 / a + 1 / c)
  : B < π / 2 := sorry

end angle_B_lt_pi_div_two_l734_734594


namespace chord_length_sqrt6_l734_734800

noncomputable def line_symmetric_axis (k : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 + p.2 + 4 = 0}

noncomputable def circle_C : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 + 4 * p.1 - 4 * p.2 + 6 = 0}

noncomputable def line_m (k : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + k}

theorem chord_length_sqrt6 (k : ℝ) 
  (hl : ∃ p ∈ circle_C, line_symmetric_axis k p)
  (m_passes_A : (0, k) ∈ line_m k) : 
  ∃ l : ℝ, l = sqrt 6 :=
begin
  sorry
end

end chord_length_sqrt6_l734_734800


namespace problem_proof_l734_734860

theorem problem_proof (p a b c : ℤ) (hp : nat.prime p) 
  (ha : Int.gcd a p = 1) (hb : Int.gcd b p = 1) (hc : Int.gcd c p = 1) : 
  ∃ (x1 x2 x3 x4 : ℤ), |x1| < (Int.sqrt p) ∧ |x2| < (Int.sqrt p) ∧ |x3| < (Int.sqrt p) ∧ |x4| < (Int.sqrt p) ∧ (a * x1 * x2 + b * x3 * x4 ≡ c [ZMOD p]) := 
sorry

end problem_proof_l734_734860


namespace multiply_97_103_eq_9991_l734_734705

theorem multiply_97_103_eq_9991 : (97 * 103 = 9991) :=
by
  have h1 : 97 = 100 - 3 := rfl
  have h2 : 103 = 100 + 3 := rfl
  calc
    97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
    ... = 100^2 - 3^2 : by rw [mul_add, add_mul, sub_mul, add_sub_cancel, sub_add_cancel]
    ... = 10000 - 9 : by norm_num
    ... = 9991 : by norm_num

end multiply_97_103_eq_9991_l734_734705


namespace problem_sequence_problem_sum_l734_734425

noncomputable def a (n : ℕ) : ℕ := if n = 0 then 1 else 2^(n-1)
noncomputable def b (n : ℕ) : ℕ := 3 * n - 2
noncomputable def S (n : ℕ) : ℕ := (Finset.range n).sum (λ k, a k)

theorem problem_sequence (λ : ℕ ≠ -1) :
  (∀ n > 0, a (n + 1) = λ * S n + 1) →
  (∀ n > 0, a n = 2^(n-1) ∧ b n = 3 * n - 2) :=
sorry

theorem problem_sum (λ : ℕ ≠ -1) :
  (∀ n > 0, a (n + 1) = λ * S n + 1) →
  (∀ n > 0, (Finset.range n).sum (λ k, a k * b k) = (3 * n - 5) * 2^n + 5) :=
sorry

end problem_sequence_problem_sum_l734_734425


namespace pete_ran_least_distance_l734_734934

theorem pete_ran_least_distance
  (phil_distance : ℕ := 4)
  (tom_distance : ℕ := 6)
  (pete_distance : ℕ := 2)
  (amal_distance : ℕ := 8)
  (sanjay_distance : ℕ := 7) :
  pete_distance ≤ phil_distance ∧
  pete_distance ≤ tom_distance ∧
  pete_distance ≤ amal_distance ∧
  pete_distance ≤ sanjay_distance :=
by {
  sorry
}

end pete_ran_least_distance_l734_734934


namespace unique_solution_of_functional_equation_l734_734198

theorem unique_solution_of_functional_equation
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f (x + y)) = f x + y) :
  ∀ x : ℝ, f x = x := 
sorry

end unique_solution_of_functional_equation_l734_734198


namespace increase_in_volume_eq_l734_734043

theorem increase_in_volume_eq (x : ℝ) (l w h : ℝ) (h₀ : l = 6) (h₁ : w = 4) (h₂ : h = 5) :
  (6 + x) * 4 * 5 = 6 * 4 * (5 + x) :=
by
  sorry

end increase_in_volume_eq_l734_734043


namespace correct_sampling_method_l734_734075

axiom vision_conditions (primary juniorHigh highSchool : Prop) : Prop :=
  ∃ (vision_conditions : Prop), (primary ∧ juniorHigh ∧ highSchool)

noncomputable def significant_difference (primary juniorHigh highSchool : Prop) : Prop :=
  primary ∧ juniorHigh ∧ highSchool

noncomputable def not_significant_difference (male female : Prop) : Prop :=
  ¬ (male ∧ female)

theorem correct_sampling_method
  (primary juniorHigh highSchool male female : Prop)
  (H1 : significant_difference primary juniorHigh highSchool)
  (H2 : not_significant_difference male female) :
  ∃ method : Prop, (method = stratified_sampling_by_educational_stage) :=
by
  sorry

end correct_sampling_method_l734_734075


namespace binomial_coefficient_condition_l734_734367

theorem binomial_coefficient_condition (n : ℕ) (h : binomial_expansion_condition (3 + x) n) :
  n = 10 :=
by
  sorry

end binomial_coefficient_condition_l734_734367


namespace range_of_a_l734_734795

def f (a x : ℝ) : ℝ := a * x^2 + log (x + 1) + x

theorem range_of_a {x1 x2 a : ℝ} (h1 : 1 < x1) (h2 : 1 < x2) 
    (h_condition : ∀ (x1 x2 : ℝ), 1 < x1 → 1 < x2 → (f a x1 - f a x2) / (x1 - x2) < 1) :
    a ≤ -1 / 4 :=
sorry

end range_of_a_l734_734795


namespace circle_reflection_problem_l734_734758

-- Conditions: Circle C passes through points (1, 2) and (2, 1) with center on line x + y - 4 = 0.
def is_circle_center (x y : ℝ) : Prop := 
  (x + y = 4) ∧ ((x - 1)^2 + (y - 2)^2 = (x - 2)^2 + (y - 1)^2)

def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 1

-- Conditions involving the reflection of light and the point M(a, 0)
def reflection_condition (a : ℝ) (M : ℝ × ℝ) : Prop := 
  M = (a, 0) ∧ (-4 / 3 * (a + 3) = 3 ∨ -3 / 4 * (a + 3) = 3)

-- Translating the proof problem into a Lean 4 statement.
theorem circle_reflection_problem :
  (∃ x y : ℝ, is_circle_center x y ∧ circle_equation x y) ∧ 
  (∃ a : ℝ, -3 / 4 ≤ a ∧ a ≤ 1 ∧ (M : ℝ × ℝ) (reflection_condition a M)) :=
sorry

end circle_reflection_problem_l734_734758


namespace vitya_masha_probability_l734_734545

theorem vitya_masha_probability :
  let total_days := 30
  let total_pairs := total_days * total_days
  let favourable_pairs := (∑ k in Finset.range total_days, k)
  total_pairs = 900 ∧ favourable_pairs = 435 ∧
  probability (Vitya at_least_one_day_older_than_Masha) = favourable_pairs / total_pairs :=
by {
  let total_days := 30,
  let total_pairs := total_days * total_days,
  let favourable_pairs := (∑ k in Finset.range total_days, k),
  
  have h1: total_pairs = 900 := by norm_num,
  have h2: favourable_pairs = 435 := by norm_num,

  have probability := 435.0 / 900.0,
  norm_num at top,
  simp,
}

end vitya_masha_probability_l734_734545


namespace sqrt_simplification_proof_l734_734028

noncomputable def sqrt_simplification : Real := Real.sqrt (Real.cbrt (Real.sqrt (1 / 32768)))

theorem sqrt_simplification_proof : sqrt_simplification = 1 / (2 * Real.sqrt 2) := 
by
  -- we will assume that the necessary simplifications steps can be properly carried out in Lean
  sorry

end sqrt_simplification_proof_l734_734028


namespace whitewash_cost_correct_l734_734573

def wall_area (length height: ℝ) : ℝ := 2 * (length + 20) * height
def door_area (num_doors: ℝ) : ℝ := num_doors * (7 * 4)
def window_area (num_windows: ℝ) : ℝ := num_windows * (5 * 3)
def openings_area (num_doors num_windows: ℝ) : ℝ := door_area(num_doors) + window_area(num_windows)

noncomputable def whitewash_cost (length width height cost_per_sqft: ℝ) (num_doors num_windows: ℝ) : ℝ :=
  let walls_area := 2 * (length * height) + 2 * (width * height)
  let open_area := openings_area(num_doors, num_windows)
  let paintable_area := walls_area - open_area
  paintable_area * cost_per_sqft

theorem whitewash_cost_correct :
  whitewash_cost 30 20 15 5 2 6 = 6770 := by
  sorry

end whitewash_cost_correct_l734_734573


namespace minimum_omega_l734_734481

theorem minimum_omega (ω : ℝ) (h_omega_pos : ω > 0) :
    (∃ y : ℝ → ℝ, (∀ x, y x = sin (ω * x + ω * (π / 2) + (π / 3))) ∧ 
    (∀ x, y x = y (-x))) →
    (ω = 1 / 3) :=
sorry

end minimum_omega_l734_734481


namespace vasya_drove_0_4_of_total_distance_l734_734648

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l734_734648


namespace tetrahedron_non_existent_l734_734190

-- Define the points in 3D space
structure Point3D :=
(x y z : ℝ)

-- Define distances between points
def distance (A B : Point3D) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)

-- State the conditions as equations
def condition1 (A B C D : Point3D) :=
  distance A B = distance A C ∧ 
  distance A C = distance A D ∧ 
  distance A D = distance B C

def condition2 (α β γ δ : ℝ) :=
  α + β = 90 ∧
  γ + δ = 90 ∧
  α + β + γ + δ = 180

-- The main theorem stating the non-existence of the tetrahedron under given conditions
theorem tetrahedron_non_existent :
  ∀ (A B C D : Point3D) (α β γ δ : ℝ),
    condition1 A B C D →
    condition2 α β γ δ →
    False := sorry

end tetrahedron_non_existent_l734_734190


namespace probability_contemporaries_correct_l734_734078

def alice_lifespan : ℝ := 150
def bob_lifespan : ℝ := 150
def total_years : ℝ := 800

noncomputable def probability_contemporaries : ℝ :=
  let unshaded_tri_area := (650 * 150) / 2
  let unshaded_area := 2 * unshaded_tri_area
  let total_area := total_years * total_years
  let shaded_area := total_area - unshaded_area
  shaded_area / total_area

theorem probability_contemporaries_correct : 
  probability_contemporaries = 27125 / 32000 :=
by
  sorry

end probability_contemporaries_correct_l734_734078


namespace value_of_angle_C_perimeter_range_l734_734767

-- Part (1): Prove angle C value
theorem value_of_angle_C
  {a b c : ℝ} {A B C : ℝ}
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (m : ℝ × ℝ := (Real.sin C, Real.cos C))
  (n : ℝ × ℝ := (2 * Real.sin A - Real.cos B, -Real.sin B))
  (orthogonal_mn : m.1 * n.1 + m.2 * n.2 = 0) 
  : C = π / 6 := sorry

-- Part (2): Prove perimeter range
theorem perimeter_range
  {a b c : ℝ} {A B C : ℝ}
  (A_range : π / 3 < A ∧ A < π / 2)
  (C_value : C = π / 6)
  (a_value : a = 2)
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  : 3 + 2 * Real.sqrt 3 < a + b + c ∧ a + b + c < 2 + 3 * Real.sqrt 3 := sorry

end value_of_angle_C_perimeter_range_l734_734767


namespace projection_of_a_on_b_l734_734278

variables (a b : ℝ × ℝ)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def projection (u v : ℝ × ℝ) : ℝ := dot_product u v / magnitude v

theorem projection_of_a_on_b (ha : a = (3, -4)) (hb : b = (0, 2)) :
  projection a b = -4 :=
by
  sorry

end projection_of_a_on_b_l734_734278


namespace five_people_lineup_count_l734_734335

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l734_734335


namespace proof_problem_l734_734263

-- Define the elliptic equation and other conditions
def ellipse_eq (x y : ℝ) (a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1

def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

def line_eq (x y m : ℝ) : Prop :=
  x = m * y + 3

-- Prove the updated/rephrased math problem
theorem proof_problem (a : ℝ) (m x1 y1 x2 y2 : ℝ) (h1 : ellipse_eq x1 y1 a) (h2 : ellipse_eq x2 y2 a) (h3 : circle_eq 3 0) (ha : a > sqrt 10) (hmne : m ≠ 0) (hline1 : line_eq x1 y1 m) (hline2 : line_eq x2 y2 m) (hperp: (x1 * x2 + y1 * y2 = 0)) : 
  (ellipse_eq 12 3 a ∧ m = sqrt (11 / 4) ∨ m = -sqrt (11 / 4) ∧ ∃ P, P.x = 4 ∧ P.y = 0 ∧ area_triangle PMN = 1) :=
begin
  -- Use sorry to skip the actual proof, only statement is required
  sorry,
end

end proof_problem_l734_734263


namespace joneal_stops_in_quarter_A_l734_734430

theorem joneal_stops_in_quarter_A
  (circumference : ℕ)
  (total_distance : ℕ)
  (start : ℕ)
  (quarters : ℕ)
  (quarters_label : Fin quarters → String)
  (start_label : String)
  (label_at_distance : ℕ → String) :
  circumference = 100 →
  total_distance = 10000 →
  start = 0 →
  quarters = 4 →
  quarters_label 0 = "A" →
  quarters_label 1 = "B" →
  quarters_label 2 = "C" →
  quarters_label 3 = "D" →
  start_label = "S" →
  label_at_distance (total_distance % circumference) = quarters_label 0 :=
begin
  intros h_circ h_dist h_start h_quarters h_q0 h_q1 h_q2 h_q3 h_start_lbl,
  have laps : total_distance / circumference = 100, by sorry,
  have remainder : total_distance % circumference = 0, by sorry,
  rw [remainder, h_q0],
  exact h_q0
end

end joneal_stops_in_quarter_A_l734_734430


namespace jack_buys_eight_hardcovers_l734_734194

theorem jack_buys_eight_hardcovers :
  ∃ (h p : ℕ), h + p = 12 ∧ 30 * h + 18 * p = 312 ∧ h = 8 :=
by
  use 8
  use 4
  sorry

end jack_buys_eight_hardcovers_l734_734194


namespace abs_ratio_diff_le_one_l734_734426

def a : ℕ → ℕ
| 1     := 1
| 2     := 2
| (n + 1) := if h : n ≥ 2 then a (n) + a (n - 1) else 0

theorem abs_ratio_diff_le_one (n k : ℕ) (hn : n ≥ 1) (hk : k ≥ 1) :
  |(a (n + 1) / a n) - (a (k + 1) / a k)| ≤ 1 := sorry

end abs_ratio_diff_le_one_l734_734426


namespace chicken_nugget_ratio_l734_734637

theorem chicken_nugget_ratio (k d a t : ℕ) (h1 : a = 20) (h2 : t = 100) (h3 : k + d + a = t) : (k + d) / a = 4 :=
by
  sorry

end chicken_nugget_ratio_l734_734637


namespace meaningful_fraction_l734_734108

theorem meaningful_fraction (x : ℝ) (hx : x ≠ 2) :
  (∀ (f : ℝ → ℝ), f = (λ x, 1 / (2 * x - 4)) → True) ∧
  (∀ (f : ℝ → ℝ), f = (λ x, 1 / (x + 2)) → x ≠ -2) ∧
  (∀ (f : ℝ → ℝ), f = (λ x, x / (x + 2)) → x ≠ -2) ∧
  (∀ (f : ℝ → ℝ), f = (λ x, (x - 2) / (x - 1)) → x ≠ 1) :=
by
  sorry

end meaningful_fraction_l734_734108


namespace find_P_at_1_l734_734034

noncomputable def P (x : ℝ) : ℝ := x ^ 2 + x + 1008

theorem find_P_at_1 :
  (∀ x : ℝ, P (P x) - (P x) ^ 2 = x ^ 2 + x + 2016) →
  P 1 = 1010 := by
  intros H
  sorry

end find_P_at_1_l734_734034


namespace opposite_of_neg_six_l734_734940

theorem opposite_of_neg_six : ∃ x : ℤ, -6 + x = 0 ∧ x = 6 :=
by {
  use 6,
  split,
  {
    linarith, 
  },
  {
    refl,
  }
}

end opposite_of_neg_six_l734_734940


namespace right_triangle_congruence_by_leg_and_angle_l734_734899

theorem right_triangle_congruence_by_leg_and_angle
  (A B C A1 B1 C1 : Type)
  [triangle ABC] [triangle A1B1C1]
  (right_triangle ABC) (right_triangle A1B1C1)
  (h_leg : AC = A1C1)
  (h_angle : ∠B = ∠B1) :
  ABC ≅ A1B1C1 :=
sorry

end right_triangle_congruence_by_leg_and_angle_l734_734899


namespace range_of_a_l734_734305

theorem range_of_a (a : ℝ) :
  (∀ p : ℝ × ℝ, (p.1 - 2 * a) ^ 2 + (p.2 - (a + 3)) ^ 2 = 4 → p.1 ^ 2 + p.2 ^ 2 = 1) →
  -1 < a ∧ a < 0 := 
sorry

end range_of_a_l734_734305


namespace sqrt_product_l734_734697

theorem sqrt_product (a b c : ℝ) (ha : a = 72) (hb : b = 18) (hc : c = 8) :
  (Real.sqrt a) * (Real.sqrt b) * (Real.sqrt c) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l734_734697


namespace graph_passes_through_fixed_point_l734_734716

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = 1 + Real.log a x :=
by
  use [1, 1]
  constructor
  · refl
  constructor
  · refl
  · sorry

end graph_passes_through_fixed_point_l734_734716


namespace largest_area_right_triangle_l734_734115

theorem largest_area_right_triangle (c : ℝ) (c_pos : 0 < c) :
  ∃ (a b : ℝ), a ≠ b ∧ a^2 + b^2 = c^2 ∧ 1/2 * a * b < 1/2 * (c/√2)^2 :=
sorry

end largest_area_right_triangle_l734_734115


namespace find_a6_l734_734256

noncomputable def a (n : ℕ) : ℝ := sorry

axiom geom_seq_inc :
  ∀ n : ℕ, a n < a (n + 1)

axiom root_eqn_a2_a4 :
  ∃ a2 a4 : ℝ, (a 2 = a2) ∧ (a 4 = a4) ∧ (a2^2 - 6 * a2 + 5 = 0) ∧ (a4^2 - 6 * a4 + 5 = 0)

theorem find_a6 : a 6 = 25 := 
sorry

end find_a6_l734_734256


namespace round_2748397_542_nearest_integer_l734_734021

theorem round_2748397_542_nearest_integer :
  let n := 2748397.542
  let int_part := 2748397
  let decimal_part := 0.542
  (n.round = 2748398) :=
by
  sorry

end round_2748397_542_nearest_integer_l734_734021


namespace find_m_l734_734931

theorem find_m (m : ℝ) 
  (f g : ℝ → ℝ) 
  (x : ℝ) 
  (hf : f x = x^2 - 3 * x + m) 
  (hg : g x = x^2 - 3 * x + 5 * m) 
  (hx : x = 5) 
  (h_eq : 3 * f x = 2 * g x) :
  m = 10 / 7 := 
sorry

end find_m_l734_734931


namespace number_of_valid_permutations_l734_734332

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l734_734332


namespace third_offense_fraction_l734_734222

-- Define the conditions
def sentence_assault : ℕ := 3
def sentence_poisoning : ℕ := 24
def total_sentence : ℕ := 36

-- The main theorem to prove
theorem third_offense_fraction :
  (total_sentence - (sentence_assault + sentence_poisoning)) / (sentence_assault + sentence_poisoning) = 1 / 3 := by
  sorry

end third_offense_fraction_l734_734222


namespace exists_divisible_by_sum_of_digits_l734_734025

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_divisible_by_sum_of_digits :
  ∀(n : ℕ), (n + 17 ≤ 2016) → (∃ m ∈ finset.range (18), ∃ k ∈ finset.range (n + 18), (k + n).digits 10.sum ∣ (k + n)) :=
  by
  intro n hn
  use sorry -- Proof goes here

end exists_divisible_by_sum_of_digits_l734_734025


namespace sum_log_geom_seq_l734_734786

-- Defining the geometric sum condition
def geom_sequence_sum (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ k, k ≤ n → a k = 2^(k - 1)

-- Defining the \( S_n = 2^n - 1 \) condition
def geom_sequence_sum_property (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  S n = 2^n - 1

-- Question: Prove that the sum of the first 12 terms of the sequence {log_2 a_n} == 66
theorem sum_log_geom_seq :
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    geom_sequence_sum_property 12 S a →
    geom_sequence_sum 12 a →
    (Finset.range 12).sum (λ k, Nat.log2 (a k)) = 66 :=
by
  intros a S hS_property h_geom_seq
  sorry

end sum_log_geom_seq_l734_734786


namespace number_of_red_pencils_l734_734069

theorem number_of_red_pencils (B R G : ℕ) (h1 : B + R + G = 20) (h2 : B = 6 * G) (h3 : R < B) : R = 6 :=
by
  sorry

end number_of_red_pencils_l734_734069


namespace parabola_properties_l734_734271

theorem parabola_properties (p : ℝ) :
  (∀ (x y : ℝ), (x, y) = (4, 4) → y^2 = 2 * p * x) →
  (let focus_distance := Real.sqrt ((4 - 1)^2 + (4 - 0)^2) in
   let directrix := (-1 : ℝ) in
   focus_distance = 5 ∧ directrix = -1) :=
by
  sorry

end parabola_properties_l734_734271


namespace solve_for_x_l734_734450

theorem solve_for_x (x : ℝ) : (2^x + 10 = 3 * 2^x - 20) → x = Real.log2 15 :=
by
  intro h
  sorry

end solve_for_x_l734_734450


namespace inverse_variation_l734_734445

theorem inverse_variation (p q : ℝ) (h1 : 500 * 2.8 = 1400) (h2 : p = 1250) : 
  q = 1.12 :=
  by
    have h3 : p * q = 1400 := by rw [←h2, h1]
    sorry

end inverse_variation_l734_734445


namespace refrigerator_price_paid_l734_734900

theorem refrigerator_price_paid
  (labelled_price : ℝ)
  (transport_cost : ℝ)
  (installation_cost : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (selling_price : ℝ) :
  discount_rate = 0.20 →
  transport_cost = 125 →
  installation_cost = 250 →
  profit_rate = 0.12 →
  selling_price = 17920 →
  1.12 * labelled_price = selling_price →
  let discounted_price := 0.80 * labelled_price
  let total_price_paid := discounted_price + transport_cost + installation_cost in
  total_price_paid = 13175 :=
by
  intros h_discount h_transport h_installation h_profit h_selling h_equation
  let discounted_price := 0.80 * labelled_price
  let total_price_paid := discounted_price + transport_cost + installation_cost
  sorry

end refrigerator_price_paid_l734_734900


namespace vasya_drives_fraction_l734_734663

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l734_734663


namespace f_max_value_f_max_min_on_interval_l734_734783

def f (x : ℝ) : ℝ := √2 * Real.sin (2 * x - π / 3)

theorem f_max_value :
  (∀ x, f x ≤ √2) ∧ f (π / 3) = √2 / √3 ∧ (|π / 3| ∈ Ioo 0 (π / 2)) → 
  f (π / 3) = √6 / 2 :=
begin
  sorry
end

theorem f_max_min_on_interval : 
  (∀ x ∈ Icc (π / 6) (π / 2), 0 ≤ f x ∧ f x ≤ √2) ∧ f (π / 6) = 0 ∧ 
  f (5 * π / 12) = √2 :=
begin
  sorry
end

end f_max_value_f_max_min_on_interval_l734_734783


namespace min_value_expression_l734_734737

theorem min_value_expression (x y : ℝ) : 
  ∃ x, ∀ y, x^2 - 6 * x * sin y - 9 * cos y ^ 2 ≥ -9 := sorry

end min_value_expression_l734_734737


namespace S_gt_1001_l734_734812

noncomputable def triangular_number (n : ℕ) : ℚ :=
(n * (n + 1)) / 2

noncomputable def a_n (n : ℕ) : ℚ :=
∑ k in Finset.range (n + 1), 1 / (triangular_number k)

noncomputable def S : ℚ :=
∑ i in Finset.range 1996, 1 / a_n i

theorem S_gt_1001 : S > 1001 :=
sorry

end S_gt_1001_l734_734812


namespace acai_juice_cost_l734_734936

noncomputable def cost_per_litre_juice (x : ℝ) : Prop :=
  let total_cost_cocktail := 1399.45 * 53.333333333333332
  let cost_mixed_fruit_juice := 32 * 262.85
  let cost_acai_juice := 21.333333333333332 * x
  total_cost_cocktail = cost_mixed_fruit_juice + cost_acai_juice

/-- The cost per litre of the açaí berry juice is $3105.00 given the specified conditions. -/
theorem acai_juice_cost : cost_per_litre_juice 3105.00 :=
  sorry

end acai_juice_cost_l734_734936


namespace total_liars_l734_734153

-- Define a dwarf type
inductive Dwarf
| liar
| knight

-- The board is a 4x4 grid of dwarves
def board := Fin 4 × Fin 4 → Dwarf

-- Define the property that a dwarf's statement about its neighbors being half liars and half knights
def equal_neighbors (b : board) (x y : Fin 4) : Prop :=
  let neighbors := [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] -- Neighbors by edge (validity to be checked within 4x4 bounds)
  let valid_neighbors := neighbors.filter (λ xy, ∀ i j, (0 ≤ i ∧ i < 4) ∧ (0 ≤ j ∧ j < 4) → xy = (i, j))
  let liars := valid_neighbors.count (λ (xy : Fin 4 × Fin 4), b xy = Dwarf.liar)
  let knights := valid_neighbors.count (λ (xy : Fin 4 × Fin 4), b xy = Dwarf.knight)
  liars = knights

-- The main theorem to prove: There are 12 liars in the board.
theorem total_liars : ∃ b : board, (∀ x y, b (x, y) = Dwarf.liar ∨ b (x, y) = Dwarf.knight)
  ∧ (∃ x y, b (x, y) = Dwarf.liar)
  ∧ (∃ x y, b (x, y) = Dwarf.knight)
  ∧ (∀ x y, equal_neighbors b x y)
  ∧ (Finset.univ.sum (λ (xy : Fin 4 × Fin 4), if b xy = Dwarf.liar then 1 else 0) = 12) :=
by
  sorry

end total_liars_l734_734153


namespace fair_eight_sided_die_probability_l734_734604

def prob_at_least_seven_at_least_four_times (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem fair_eight_sided_die_probability : prob_at_least_seven_at_least_four_times 5 4 (1 / 4) + (1 / 4) ^ 5 = 1 / 64 :=
by
  sorry

end fair_eight_sided_die_probability_l734_734604


namespace tan_B_eq_3_tan_A_A_value_l734_734380

section triangle_problem

variables {A B C : ℝ}

-- Condition 1
axiom condition1 : (|vector AB| * |vector AC| * cos A) = 3 * (|vector BA| * |vector BC| * cos B)

-- Prove that tan B = 3 tan A
theorem tan_B_eq_3_tan_A (h : condition1) : tan B = 3 * tan A := sorry

-- Condition 2
axiom condition2 : cos C = (sqrt 5) / 5

-- Given tan B = 3 tan A, find the value of A
theorem A_value (h : tan B = 3 * tan A) (h1 : condition2) : A = π / 4 := sorry

end triangle_problem

end tan_B_eq_3_tan_A_A_value_l734_734380


namespace min_waiting_time_max_waiting_time_expected_waiting_time_l734_734989

open Nat

noncomputable def C : ℕ → ℕ → ℕ
| n, 0     => 1
| 0, k     => 0
| n+1, k+1 => C n k + C n (k+1)

def a := 1
def b := 5
def n := 5
def m := 3

def T_min := a * C (n - 1) 2 + m * n * a + b * C m 2
def T_max := a * C n 2 + b * m * n + b * C m 2
def E_T := C (n + m) 2 * (b * m + a * n) / (m + n)

theorem min_waiting_time : T_min = 40 := by
  sorry

theorem max_waiting_time : T_max = 100 := by
  sorry

theorem expected_waiting_time : E_T = 70 := by
  sorry

end min_waiting_time_max_waiting_time_expected_waiting_time_l734_734989


namespace tangent_AC_l734_734845

theorem tangent_AC {A B C L P : Point} 
  (h_triangle_ABC : Triangle A B C)
  (h_BL_bisector : IsAngleBisector B L C)
  (h_tangent_L : TangentThroughPoint L (Circumcircle B L C) (Line P L))
  (h_intersects_AB_P : Intersection (Line A B) (Line P L) P) :
  Tangent (Circumcircle B P L) (Line A C) :=
sorry

end tangent_AC_l734_734845


namespace log_sum_l734_734823

theorem log_sum (x y z : ℝ) (h1 : Real.log 3 (Real.log 4 (Real.log 5 x)) = 0)
                          (h2 : Real.log 4 (Real.log 5 (Real.log 3 y)) = 0)
                          (h3 : Real.log 5 (Real.log 3 (Real.log 4 z)) = 0) :
  x + y + z = 932 := by
  sorry

end log_sum_l734_734823


namespace problem1_problem2_problem3_l734_734837

-- Problem 1
theorem problem1 (P : ℕ → ℤ × ℤ) (h₀ : P 0 = (0, 1)) 
    (h₁ : ∀ k, k > 0 → |(P k).1 - (P (k-1)).1| * |(P k).2 - (P (k-1)).2| = 2) 
    (h₂ : (P 1).2 - (P 0).2 > (P 1).1 - (P 0).1 > 0) : 
    P 1 = (1, 3) :=
sorry

-- Problem 2
theorem problem2 (P : ℕ → ℤ × ℤ) (h₀ : P 0 = (0, 1)) 
    (h₁ : ∀ k, k > 0 → (P k).1 - (P (k-1)).1 = 1) 
    (h₂ : ∀ k, k > 0 → y : ℕ, y > k → (P y).2 > (P k).2) 
    (h₃ : ∃ n, P n = (n, 1 + 2 * n) ∧ 1 + 2 * n = 3 * n - 8) : 
    ∃ n, n = 9 :=
sorry

-- Problem 3
theorem problem3 (P : ℕ → ℤ × ℤ) (h₀ : P 0 = (0, 0)) 
    (h₁ : (P 2016).2 = 100) 
    (h₂ : ∀ k, k > 0 → |(P k).1 - (P (k-1)).1| * |(P k).2 - (P (k-1)).2| = 2) : 
    ∃ M, M = 4066272 ∧ Σ k in (finset.range (2017)), (P k).1 = M :=
sorry

end problem1_problem2_problem3_l734_734837


namespace units_digit_sum_of_series_l734_734186

theorem units_digit_sum_of_series :
  let s := (Finset.range 11).sum (λ n, (n + 1)! + (n + 1)) in
  s % 10 = 9 :=
by
  -- Definitions
  let terms := (Finset.range 11).map (λ n, (n + 1)! + (n + 1))
  let units_digits := terms.map (λ x, x % 10)
  let sum_units_digits := units_digits.sum
  have mod_condition : ∀ n ≥ 5, (n! % 10) = 0 := sorry
  have fact1 := calc (1! % 10) = 1 : by norm_num
  have fact2 := calc (2! % 10) = 2 : by norm_num
  have fact3 := calc (3! % 10) = 6 : by norm_num
  have fact4 := calc (4! % 10) = 4 : by norm_num
  -- Main proof (skipped)
  sorry

end units_digit_sum_of_series_l734_734186


namespace range_of_a_l734_734250

theorem range_of_a (a : ℝ) (h : ∀ x, a ≤ x ∧ x ≤ a + 2 → |x + a| ≥ 2 * |x|) : a ≤ -3 / 2 := 
by
  sorry

end range_of_a_l734_734250


namespace Vasya_distance_fraction_l734_734655

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l734_734655


namespace sum_seven_terms_l734_734785

section
variable {α : Type} [LinearOrderedField α]
variable (a : ℕ → α) (d : α)

-- Definitions and conditions from the problem
def is_arithmetic_sequence (a : ℕ → α) := ∃ d, ∀ n, a (n + 1) = a n + d
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 1 + a 4 + a 7 = 12

-- Theorem to prove the result
theorem sum_seven_terms : (∑ n in Finset.range 7, a (n + 1)) = 28 :=
by
    sorry
end

end sum_seven_terms_l734_734785


namespace div_of_power_diff_div_l734_734848

theorem div_of_power_diff_div (a b n : ℕ) (h : a ≠ b) (h₀ : n ∣ (a^n - b^n)) : n ∣ (a^n - b^n) / (a - b) :=
  sorry

end div_of_power_diff_div_l734_734848


namespace vasya_fraction_is_0_4_l734_734649

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l734_734649


namespace no_such_integer_n_exists_l734_734728

theorem no_such_integer_n_exists :
  ¬ ∃ n : ℕ, n > 0 ∧ 
    (∃ s1 s2 : Finset ℕ, 
      s1 ∪ s2 = Finset.range 6 ∧
      s1 ≠ ∅ ∧ s2 ≠ ∅ ∧
      (s1.prod (λ x, n + x) = s2.prod (λ x, n + x))) := 
begin
  sorry
end

end no_such_integer_n_exists_l734_734728


namespace sin_sum_cos_l734_734026

theorem sin_sum_cos (n : ℕ) (h : n > 2) : 
  let α := Real.pi / n in
  ∑ i in Finset.range (n - 1), 
    if (i + 1) % 2 = 0 then (Real.sin ((i + 1) * α) * Real.sin ((i + 2) * α)) else 0 = 
  (n / 2) * Real.cos α :=
by
  sorry

end sin_sum_cos_l734_734026


namespace cody_discount_l734_734173

theorem cody_discount (initial_cost tax_rate cody_paid total_paid price_before_discount discount: ℝ) 
  (h1 : initial_cost = 40)
  (h2 : tax_rate = 0.05)
  (h3 : cody_paid = 17)
  (h4 : total_paid = 2 * cody_paid)
  (h5 : price_before_discount = initial_cost * (1 + tax_rate))
  (h6 : discount = price_before_discount - total_paid) :
  discount = 8 := by
  sorry

end cody_discount_l734_734173


namespace sixth_grade_charts_are_specific_l734_734105

def sixth_grade_statistical_charts (grade: ℕ) : list string :=
  if grade = 6 then ["bar charts", "line charts", "pie charts"] else []

theorem sixth_grade_charts_are_specific : 
  sixth_grade_statistical_charts 6 = ["bar charts", "line charts", "pie charts"] :=
by 
  -- Mathematical equivalent of the problem translated into Lean
  -- statement without involving solution steps
  sorry

end sixth_grade_charts_are_specific_l734_734105


namespace general_formula_Sn_range_l734_734175

variables {a : ℕ → ℝ} [1, b : ℕ → ℝ]

-- Define the constant conditions and general term
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n = a 1 * (1/3)^(n-1)

noncomputable def sequence_cond1 (a : ℕ → ℝ) : Prop :=
2 * a 1 + 3 * a 2 = 1

noncomputable def sequence_cond2 (a : ℕ → ℝ) : Prop :=
a 3^2 = 9 * a 2 * a 6

-- Assert the first part of the proof problem: the general formula for a_n
theorem general_formula (a : ℕ → ℝ) [is_geometric_sequence a]
  (h1 : sequence_cond1 a) (h2 : sequence_cond2 a) : 
  ∀ n : ℕ, n ≥ 1 → a n = (1/3)^n :=
sorry

-- Define b_n as the sum of logarithms base 3
def b (n : ℕ) (a : ℕ → ℝ) : ℝ :=
∑ i in range n, real.logb 3 (a (i + 1))

-- Assert the second part of the proof problem: the range of S_n
theorem Sn_range (a : ℕ → ℝ) (S_n : ℕ → ℝ) [is_geometric_sequence a]
  (h1 : sequence_cond1 a) (h2 : sequence_cond2 a) :
  ∀ n ≥ 1, S_n n = -2 * (1 - 1 / real.of_nat (n + 1)) ∧ -2 < S_n n ∧ S_n n ≤ -1 :=
sorry

end general_formula_Sn_range_l734_734175


namespace complex_expression_l734_734082

theorem complex_expression : 
  let a := (3 : ℝ) + (2 : ℂ).i 
      b := (2 : ℝ) - (3 : ℂ).i
  in 3 * a + 4 * b = (17 : ℂ) - (6 : ℂ).i :=
by
  -- add steps of the proof here
  sorry

end complex_expression_l734_734082


namespace pascal_triangle_third_number_l734_734103

theorem pascal_triangle_third_number {n k : ℕ} (h : n = 51) (hk : k = 2) : Nat.choose n k = 1275 :=
by
  rw [h, hk]
  norm_num

#check pascal_triangle_third_number

end pascal_triangle_third_number_l734_734103


namespace problem_statement_l734_734803

def M : Set ℝ := {x | (x + 3) / (x - 1) < 0}

def N : Set ℝ := {x | x ≤ -3}

def Prop : Prop := (complement (M ∪ N) = {x | x ≥ 1})

theorem problem_statement : Prop :=
by
  sorry

end problem_statement_l734_734803


namespace five_people_lineup_count_l734_734337

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l734_734337


namespace range_of_m_l734_734826

theorem range_of_m (m : ℝ) : 
  (∃ x ∈ set.Icc 1 2, x^2 - m * x + 2 = 0) ↔ 2 * Real.sqrt 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l734_734826


namespace diameter_of_sphere_with_triple_volume_l734_734688

noncomputable def sphere_volume (r : ℝ) := (4 / 3) * Real.pi * r^3

theorem diameter_of_sphere_with_triple_volume (r d : ℝ)
    (h1 : r = 6)
    (h2 : d = 12 * Real.cbrt 3) :
    d = 12 * Real.cbrt 3 :=
by
  sorry

end diameter_of_sphere_with_triple_volume_l734_734688


namespace proof_n_times_s_l734_734411

def g : ℝ → ℝ := sorry

axiom g_eqn : ∀ (x y : ℝ), g (x^2 - y^2) = (x - y) * (g x + g y)

axiom g1 : g 1 = 2

theorem proof_n_times_s : 
  ∃ (n s : ℕ), 
    n = (finset.univ.filter (λ x, g 3 = x)).card ∧ 
    s = (finset.univ.filter (λ x, g 3 = x)).sum id ∧ 
    n * s = 6 := 
sorry

end proof_n_times_s_l734_734411


namespace ellipse_properties_l734_734239

noncomputable def ellipse_equation (e : ℝ) (b : ℝ) : Prop :=
a = sqrt (b^2 + (e * a)^2) ∧ (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def line_L_through_P (L : ℝ → ℝ) : Prop :=
∀ x: ℝ, L x = slope * x + 2

noncomputable def tangent_line_to_ellipse (L: ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
∀ x : ℝ, (x^2 / a^2) + (L x)^2 = 1 → (slope = 1 ∨ slope = -1)

noncomputable def maximum_area_triangle (a : ℝ) (b : ℝ) (slope : ℝ) : ℝ :=
(3 * sqrt 3) / 4

theorem ellipse_properties 
(e : ℝ) 
(b : ℝ) 
(L: ℝ → ℝ) 
(slope : ℝ) 
(h_eccentricity : e = sqrt 6 / 3) 
(h_b : b = 1) 
(h_line_through_P : line_L_through_P L) : 
  ellipse_equation e b → 
  tangent_line_to_ellipse L (sqrt 3) 1 → 
  maximum_area_triangle (sqrt 3) 1 (sqrt 21 / 3) = (3 * sqrt 3) / 4 := 
by
  intros
  unfold ellipse_equation
  unfold tangent_line_to_ellipse
  unfold maximum_area_triangle
  sorry

end ellipse_properties_l734_734239


namespace count_valid_arrangements_l734_734322

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l734_734322


namespace distinct_four_integers_sum_count_is_integer_l734_734217

theorem distinct_four_integers_sum_count_is_integer :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∑ (x : ℕ) in {a, b, c, d}, (x - 1) / x = 4 - k ∧ k = 1 ∧ finset.card (finset {a, b, c, d}) = 4 :=
begin
  sorry
end

end distinct_four_integers_sum_count_is_integer_l734_734217


namespace max_real_roots_l734_734183

-- The maximum number of real roots for the polynomial given specific conditions
theorem max_real_roots (n : ℕ) (k : ℝ) (hn : n > 0) (hk : k ≠ 0) : 
  (∃ x : ℝ, x ^ n + x ^ (n - 1) + ... + x + k = 0) → 
  (n % 2 = 1 ∧ k = -1 → ∃ x : ℝ, x = -1 ∧ x ^ n + x ^ (n - 1) + ... + x - 1 = 0) :=
by
  sorry

end max_real_roots_l734_734183


namespace min_val_square_l734_734889

open Complex Real

noncomputable def z := r * (cos θ + I * sin θ)

theorem min_val_square {r θ : ℝ} (hr : 0 < r) (h_area : abs (sin 2 * θ) = 12 / 13) :
  (min {d : ℝ | d = complex.abs (z + (1 / z))})^2 = 16 / 13 :=
sorry

end min_val_square_l734_734889


namespace mashas_end_number_is_17_smallest_starting_number_ends_with_09_l734_734000

def mashas_operation (n : ℕ) : ℕ :=
  let y := n % 10
  let x := n / 10
  3 * x + 2 * y

def mashas_stable_result (n : ℕ) : Prop :=
  mashas_operation n = n

theorem mashas_end_number_is_17 :
  ∃ n : ℕ, mashas_stable_result n ∧ n = 17 :=
sorry

def is_smallest_starting_number (n : ℕ) : Prop :=
  (nat.gcd n 17 = 17) ∧ (nat.log 10 n = 2014) ∧ (n % 100 = 9)

theorem smallest_starting_number_ends_with_09 :
  ∃ n : ℕ, is_smallest_starting_number n :=
sorry

end mashas_end_number_is_17_smallest_starting_number_ends_with_09_l734_734000


namespace term_containing_x_neg2_l734_734258

variable (a b : ℝ) (n : ℕ)

-- Conditions
axiom a_positive : a > 0
axiom b_positive : b > 0
axiom n_positive : n = 8 ∧ n ∈ ℕ \ {0}
axiom A_eq : 2^n = 256
axiom B_eq : (a + b) ^ n = 256
axiom C_eq : (binomial (2*n) n) * (a ^ n) * (b ^ n) = 70
axiom sum_eq : a + b = 2
axiom product_eq : a^4 * b^4 = 1

-- Goal
theorem term_containing_x_neg2 : 
  ∃ r, (8 - 2 * r = -2) ∧ ((nat.choose 8 r) * (a^ (8-r)) * (b^r) = 56) := 
by 
  sorry

end term_containing_x_neg2_l734_734258


namespace bus_ride_cost_l734_734570

variable (cost_bus cost_train : ℝ)

-- Condition 1: cost_train = cost_bus + 2.35
#check (cost_train = cost_bus + 2.35)

-- Condition 2: cost_bus + cost_train = 9.85
#check (cost_bus + cost_train = 9.85)

theorem bus_ride_cost :
  (∃ (cost_bus cost_train : ℝ),
    cost_train = cost_bus + 2.35 ∧
    cost_bus + cost_train = 9.85) →
  cost_bus = 3.75 :=
sorry

end bus_ride_cost_l734_734570


namespace students_present_l734_734953

-- Define the total number of students in the class
def total_students : ℕ := 50

-- Define the percentage of students who are absent
def absent_percentage : ℝ := 0.14

-- Define the expected number of students present in the class
def expected_present_students : ℕ := 43

-- Prove that the number of present students is 43 given the conditions
theorem students_present (h1 : 0 ≤ absent_percentage ∧ absent_percentage ≤ 1) : 
  (total_students : ℝ) * (1 - absent_percentage) = (expected_present_students : ℝ) :=
by
  rw [← nat.cast_mul, nat.cast_bit0, nat.cast_bit0, nat.cast_bit1],
  norm_num,
  sorry

end students_present_l734_734953


namespace oak_trees_in_park_l734_734521

theorem oak_trees_in_park (planting_today : ℕ) (total_trees : ℕ) 
  (h1 : planting_today = 4) (h2 : total_trees = 9) : 
  total_trees - planting_today = 5 :=
by
  -- proof goes here
  sorry

end oak_trees_in_park_l734_734521


namespace sum_of_x_coordinates_l734_734463

-- Define segment points
def p1 : ℝ × ℝ := (-5, -3)
def p2 : ℝ × ℝ := (-3, 0)
def p3 : ℝ × ℝ := (0, -1)
def p4 : ℝ × ℝ := (2, 3)
def p5 : ℝ × ℝ := (5, 1)

-- Function defining the four line segments as piecewise functions
noncomputable def g : ℝ → ℝ
| x := if x ∈ set.Icc (-5) (-3) then (3/2 : ℝ)*x + 15/2
       else if x ∈ set.Icc (-3) (0) then -(1/3 : ℝ)*x
       else if x ∈ set.Icc (0) (2) then 2*x - 1
       else if x ∈ set.Icc (2) (5) then -x + 5
       else (0 : ℝ)

-- The proof problem
theorem sum_of_x_coordinates : ∑ x in {-17, -3/4, 0, 3}, x = -14.75 :=
sorry

end sum_of_x_coordinates_l734_734463


namespace theta_interval_l734_734788

noncomputable def f (x θ: ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_interval (θ: ℝ) (k: ℤ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x θ > 0) → 
  (2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12) := 
by
  sorry

end theta_interval_l734_734788


namespace rosie_pies_l734_734903

def number_of_pies (apples : ℕ) : ℕ := sorry

theorem rosie_pies (h : number_of_pies 9 = 2) : number_of_pies 27 = 6 :=
by sorry

end rosie_pies_l734_734903


namespace vasya_drives_fraction_l734_734665

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l734_734665


namespace det_A_is_half_l734_734949

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
![![Real.cos (20 * Real.pi / 180), Real.sin (40 * Real.pi / 180)], ![Real.sin (20 * Real.pi / 180), Real.cos (40 * Real.pi / 180)]]

theorem det_A_is_half : A.det = 1 / 2 := by
  sorry

end det_A_is_half_l734_734949


namespace sum_of_two_consecutive_squares_l734_734898

variable {k m A : ℕ}

theorem sum_of_two_consecutive_squares :
  (∃ k : ℕ, A^2 = (k+1)^3 - k^3) → (∃ m : ℕ, A = m^2 + (m+1)^2) :=
by sorry

end sum_of_two_consecutive_squares_l734_734898


namespace sufficient_not_necessary_condition_l734_734228

variable (a : ℝ)

def M := {1, a}
def N := {-1, 0, 1}

theorem sufficient_not_necessary_condition : 
  (M ⊆ N ↔ (a = 0 ∨ a = -1)) → (M ⊆ N) ∧ (a = 0 → M ⊆ N) ∧ ¬(a = 0 → ¬M ⊆ N) :=
by 
sorry

end sufficient_not_necessary_condition_l734_734228


namespace transform_111_to_777_in_10_operations_l734_734805

-- Definition of the given operation: changing any 2 digits to the units digit of their sum
def change_digits (n : ℕ) (i j : ℕ) : ℕ :=
  let d1 := (n / 100) % 10 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  let sum := (if i == 1 then d1 else if i == 2 then d2 else d3) + (if j == 1 then d1 else if j == 2 then d2 else d3) in
  let u := sum % 10 in
  if i == 1 then if j == 2 then u * 100 + u * 10 + d3
                 else u * 100 + d2 * 10 + u
             else if j == 2 then d1 * 100 + u * 10 + u
                           else d1 * 100 + d2 * 10 + u

-- The minimum number of operations required to change 111 to 777
def min_operations_111_to_777 : ℕ :=
  10

-- The main theorem to prove
theorem transform_111_to_777_in_10_operations :
  ∃ (seq : list ℕ), seq.length = min_operations_111_to_777 ∧
  seq.head = 111 ∧ seq.last = 777 ∧
  (∀ (i : ℕ), i < seq.length - 1 → ∃ (x y : ℕ), seq.nth i = change_digits (seq.nth i).get_or_else 0 x y) :=
  sorry

end transform_111_to_777_in_10_operations_l734_734805


namespace determine_OP_l734_734024

theorem determine_OP 
  (a b c d e f : ℝ)
  (h1 : 2 * a ≠ 15 * e + 20 * f) :
  ∃ x : ℝ, 12 * d ≤ x ∧ x ≤ 15 * e ∧
    x = (300 * a * e - 240 * d * f) / (2 * a - 15 * e + 20 * f) :=
begin
  use (300 * a * e - 240 * d * f) / (2 * a - 15 * e + 20 * f),
  split,
  sorry, -- Proofs for x being within the range [12d, 15e]
  split,
  sorry, -- Proof of x being within the range [12d, 15e]
  apply eq.refl, -- Equality of x
end

end determine_OP_l734_734024


namespace PQ_is_10_5_l734_734515

noncomputable def PQ_length_proof_problem : Prop := 
  ∃ (PQ : ℝ),
    PQ = 10.5 ∧ 
    ∃ (ST : ℝ) (SU : ℝ),
      ST = 4.5 ∧ SU = 7.5 ∧ 
      ∃ (QR : ℝ) (PR : ℝ),
        QR = 21 ∧ PR = 15 ∧ 
        ∃ (angle_PQR angle_STU : ℝ),
          angle_PQR = 120 ∧ angle_STU = 120 ∧ 
          PQ / ST = PR / SU

theorem PQ_is_10_5 :
  PQ_length_proof_problem := sorry

end PQ_is_10_5_l734_734515


namespace number_of_valid_permutations_l734_734326

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l734_734326


namespace no_positive_integers_k_m_l734_734443

theorem no_positive_integers_k_m (k m : ℕ) (hk : k > 0) (hm : m > 0) : k! + 48 ≠ 48 * (k + 1)^m :=
begin
  sorry
end

end no_positive_integers_k_m_l734_734443


namespace two_common_points_with_x_axis_l734_734798

noncomputable def func (x d : ℝ) : ℝ := x^3 - 3 * x + d

theorem two_common_points_with_x_axis (d : ℝ) :
(∃ x1 x2 : ℝ, x1 ≠ x2 ∧ func x1 d = 0 ∧ func x2 d = 0) ↔ (d = 2 ∨ d = -2) :=
by
  sorry

end two_common_points_with_x_axis_l734_734798


namespace second_longest_piece_length_l734_734914

theorem second_longest_piece_length (total_length : ℝ) (ratios : list ℝ) (length_of_second_longest : ℝ) :
  total_length = 142.75 →
  ratios = [real.sqrt 2, 6, 4 / 3, 9, 1 / 2] →
  length_of_second_longest = 46.938 :=
by
  sorry

end second_longest_piece_length_l734_734914


namespace avg_marks_first_class_l734_734041

theorem avg_marks_first_class (avg1 avg2 avg_combined : ℝ) (n2 total_students : ℕ) (marks1_sum marks2_sum : ℝ) :
  avg1 = 45 ∧ avg2 = 65 ∧ avg_combined = 57.22222222222222 ∧
  n2 = 55 ∧ total_students = n2 + (marks1_sum / avg1) ∧
  marks1_sum + marks2_sum = total_students * avg_combined ∧
  marks2_sum = avg2 * n2 →
  (marks1_sum / avg1) ≈ 551 :=
by sorry

end avg_marks_first_class_l734_734041


namespace number_of_integers_between_roots_l734_734280

theorem number_of_integers_between_roots :
  let sqrt8 := Real.sqrt 8
      sqrt80 := Real.sqrt 80 in
  3 ≤ sqrt8 ∧ sqrt8 < 3 ∧ 8 < sqrt80 ∧ sqrt80 ≤ 9 →
  (Finset.card (Finset.filter (λ n, sqrt8 < n ∧ n < sqrt80) (Finset.Icc 3 8))) = 6 :=
by
  sorry

end number_of_integers_between_roots_l734_734280


namespace pascal_triangle_third_number_l734_734100

theorem pascal_triangle_third_number {n k : ℕ} (h : n = 51) (hk : k = 2) : Nat.choose n k = 1275 :=
by
  rw [h, hk]
  norm_num

#check pascal_triangle_third_number

end pascal_triangle_third_number_l734_734100


namespace solve_inequality_l734_734791

namespace Problem

-- Define polynomial f(x) = x^2 + ax + b
def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

-- Conditions: f has zeroes at -2 and 3
lemma zeroes_of_f (a b : ℝ) : 
  f (-2) a b = 0 ∧ f 3 a b = 0 :=
by sorry

-- Coefficients a = -1 and b = -6
def a := -1
def b := -6

-- Polynomial becomes f(x) = x^2 - x - 6
lemma polynomial_expansion : f x a b = x^2 - x - 6 := 
by sorry

-- Set interval (-3, 2)
def interval := set.Ioo (-3 : ℝ) 2

-- Prove the inequality bf(ax) > 0 results in x ∈ (-3, 2)
theorem solve_inequality : ∀ x : ℝ, (b * (f (-x) a b)) > 0 ↔ x ∈ interval :=
by sorry

end Problem

end solve_inequality_l734_734791


namespace circle_equation_correct_l734_734743

-- Definitions of points of intersection and line equation
def line_eq (x y : ℝ) := 3 * x - 2 * y + 12 = 0

-- Intersection points
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 6)

-- Midpoint of points A and B
def M0 : ℝ × ℝ := ((-4 + 0) / 2, (0 + 6) / 2)

-- Radius of the circle
def radius : ℝ := Real.sqrt (2 ^ 2 + 3 ^ 2)

-- Equation of the circle using the center and radius
def circle_eq (x y : ℝ) := (x + 2) ^ 2 + (y - 3) ^ 2 = (Real.sqrt 13) ^ 2

-- Mathematical proof problem statement
theorem circle_equation_correct : ∀ x y, circle_eq x y ↔ (x + 2) ^ 2 + (y - 3) ^ 2 = 13 :=
by sorry

end circle_equation_correct_l734_734743


namespace distance_travelled_downstream_l734_734518

theorem distance_travelled_downstream :
  ∀ (boat_speed current_rate : ℝ) (travel_time_minutes : ℝ),
  boat_speed = 42 →
  current_rate = 4 →
  travel_time_minutes = 44 →
  let effective_speed := boat_speed + current_rate,
      travel_time_hours := travel_time_minutes / 60,
      distance := effective_speed * travel_time_hours
  in distance = 33.7318 := by
  sorry

end distance_travelled_downstream_l734_734518


namespace total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l734_734987

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l734_734987


namespace triangle_sin_cos_relation_l734_734830

variable {A B : ℝ}
variable {α β : ℝ} -- Corresponding angles

-- Let A and B be the sides opposite to angles α and β in Δ ABC.
def condition_1 : sin α > sin β := sorry
def condition_2 : cos α < cos β := sorry
def necessary_but_not_sufficient_condition (α β : ℝ) : Prop :=
  condition_1 α β → condition_2 α β ∧ ¬(condition_2 α β → condition_1 α β)

theorem triangle_sin_cos_relation (α β : ℝ) (h : α + β < π) :
  necessary_but_not_sufficient_condition α β :=
sorry

end triangle_sin_cos_relation_l734_734830


namespace seq_remainder_l734_734145

noncomputable def b : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 1
| (n+3) := b n + b (n+1) + b (n+2)

theorem seq_remainder :
  b 28 = 6090307 → b 29 = 11201821 → b 30 = 20603361 →
  (∑ k in Finset.range 30, b k) % 500 = 216 :=
by
  intros h28 h29 h30
  have h_b_28 : b 28 = 6090307 := h28
  have h_b_29 : b 29 = 11201821 := h29
  have h_b_30 : b 30 = 20603361 := h30
  sorry

end seq_remainder_l734_734145


namespace max_value_ln_x_minus_x_on_interval_l734_734927

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem max_value_ln_x_minus_x_on_interval : 
  ∃ x ∈ Set.Ioc 0 (Real.exp 1), ∀ y ∈ Set.Ioc 0 (Real.exp 1), f y ≤ f x ∧ f x = -1 :=
by
  sorry

end max_value_ln_x_minus_x_on_interval_l734_734927


namespace polynomial_satisfies_conditions_l734_734419

noncomputable def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (∀ x y z : ℝ, f x (z^2) y + f x (y^2) z = 0) ∧ 
  (∀ x y z : ℝ, f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end polynomial_satisfies_conditions_l734_734419


namespace cost_of_jeans_l734_734816

theorem cost_of_jeans 
  (price_socks : ℕ)
  (price_tshirt : ℕ)
  (price_jeans : ℕ)
  (h1 : price_socks = 5)
  (h2 : price_tshirt = price_socks + 10)
  (h3 : price_jeans = 2 * price_tshirt) :
  price_jeans = 30 :=
  by
    -- Sorry skips the proof, complies with the instructions
    sorry

end cost_of_jeans_l734_734816


namespace line_up_ways_l734_734359

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l734_734359


namespace minimum_omega_for_symmetric_curve_l734_734471

theorem minimum_omega_for_symmetric_curve (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 2) + π / 3) = sin (-ω * (x + π / 2) + π / 3)) ↔ ω = 1 / 3 :=
by
  sorry

end minimum_omega_for_symmetric_curve_l734_734471


namespace find_number_l734_734918

-- Define the conditions and the theorem
theorem find_number (number : ℝ)
  (h₁ : ∃ w : ℝ, w = (69.28 * number) / 0.03 ∧ abs (w - 9.237333333333334) ≤ 1e-10) :
  abs (number - 0.004) ≤ 1e-10 :=
by
  sorry

end find_number_l734_734918


namespace reflect_polar_point_l734_734375

-- Define the given conditions
structure PolarPoint where
  rho : ℝ
  theta : ℝ

-- Define the reflection function
def reflectAcrossPole (p : PolarPoint) : PolarPoint :=
  PolarPoint.mk p.rho (p.theta + Real.pi)

-- The theorem to prove the problem statement
theorem reflect_polar_point (p : PolarPoint) : reflectAcrossPole p = PolarPoint.mk p.rho (p.theta + Real.pi) :=
by
  -- The proof is omitted (replace this with the actual proof when needed)
  sorry

end reflect_polar_point_l734_734375


namespace candidate_marks_percentage_l734_734599

theorem candidate_marks_percentage (T : ℝ) (passing_marks : ℝ) (candidate1_marks_fraction : ℝ) 
  (candidate1_failing_margin : ℝ) (candidate2_over_passing_marks : ℝ)
  (h1 : candidate1_marks_fraction = 0.20)
  (h2 : candidate1_failing_margin = 40)
  (h3 : passing_marks = 160)
  (h4 : T = 600) :
  ((passing_marks + candidate2_over_passing_marks) / T) * 100 = 30 :=
by
  sorry

end candidate_marks_percentage_l734_734599


namespace complex_ordered_pairs_count_l734_734208

theorem complex_ordered_pairs_count :
  (⇑ (Σ' (x y : ℂ), x ^ 4 * y ^ 3 = 1 ∧ x ^ 8 * y = 1)).nonempty.card = 40 :=
sorry

end complex_ordered_pairs_count_l734_734208


namespace tailor_buttons_l734_734624

theorem tailor_buttons (G : ℕ) (yellow_buttons : ℕ) (blue_buttons : ℕ) 
(h1 : yellow_buttons = G + 10) (h2 : blue_buttons = G - 5) 
(h3 : G + yellow_buttons + blue_buttons = 275) : G = 90 :=
sorry

end tailor_buttons_l734_734624


namespace analytical_expression_of_f_l734_734796

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b

theorem analytical_expression_of_f (a b : ℝ) (h_a : a > 0)
  (h_max : (∃ x_max : ℝ, f x_max a b = 5 ∧ (∀ x : ℝ, f x_max a b ≥ f x a b)))
  (h_min : (∃ x_min : ℝ, f x_min a b = 1 ∧ (∀ x : ℝ, f x_min a b ≤ f x a b))) :
  f x 3 1 = x^3 + 3 * x^2 + 1 := 
sorry

end analytical_expression_of_f_l734_734796


namespace necessary_but_not_sufficient_l734_734592

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 0) : a > 0 ↔ ((a > 0) ∧ (a < 2) → (a^2 - 2 * a < 0)) :=
by
    sorry

end necessary_but_not_sufficient_l734_734592


namespace math_problem_real_solution_l734_734414

theorem math_problem_real_solution (x y : ℝ) (h : x^2 * y^2 - x * y - x / y - y / x = 4) : 
  (x - 2) * (y - 2) = 3 - 2 * Real.sqrt 2 :=
sorry

end math_problem_real_solution_l734_734414


namespace count_valid_arrangements_l734_734319

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l734_734319


namespace third_roll_six_probability_l734_734692

noncomputable def Die_A_six_prob : ℚ := 1 / 6
noncomputable def Die_B_six_prob : ℚ := 1 / 2
noncomputable def Die_C_one_prob : ℚ := 3 / 5
noncomputable def Die_B_not_six_prob : ℚ := 1 / 10
noncomputable def Die_C_not_one_prob : ℚ := 1 / 15

noncomputable def prob_two_sixes_die_A : ℚ := Die_A_six_prob ^ 2
noncomputable def prob_two_sixes_die_B : ℚ := Die_B_six_prob ^ 2
noncomputable def prob_two_sixes_die_C : ℚ := Die_C_not_one_prob ^ 2

noncomputable def total_prob_two_sixes : ℚ := 
  (1 / 3) * (prob_two_sixes_die_A + prob_two_sixes_die_B + prob_two_sixes_die_C)

noncomputable def cond_prob_die_A_given_two_sixes : ℚ := prob_two_sixes_die_A / total_prob_two_sixes
noncomputable def cond_prob_die_B_given_two_sixes : ℚ := prob_two_sixes_die_B / total_prob_two_sixes
noncomputable def cond_prob_die_C_given_two_sixes : ℚ := prob_two_sixes_die_C / total_prob_two_sixes

noncomputable def prob_third_six : ℚ := 
  cond_prob_die_A_given_two_sixes * Die_A_six_prob + 
  cond_prob_die_B_given_two_sixes * Die_B_six_prob + 
  cond_prob_die_C_given_two_sixes * Die_C_not_one_prob

theorem third_roll_six_probability : 
  prob_third_six = sorry := 
  sorry

end third_roll_six_probability_l734_734692


namespace E_72_eq_9_l734_734403

def E (n : ℕ) : ℕ :=
  -- Assume a function definition counting representations
  -- (this function body is a placeholder, as the exact implementation
  -- is not part of the problem statement)
  sorry

theorem E_72_eq_9 :
  E 72 = 9 :=
sorry

end E_72_eq_9_l734_734403


namespace prop_neg_or_not_l734_734302

theorem prop_neg_or_not (p q : Prop) (h : ¬(p ∨ ¬ q)) : ¬ p ∧ q :=
by
  sorry

end prop_neg_or_not_l734_734302


namespace opposite_of_neg_six_l734_734938

theorem opposite_of_neg_six :
  ∃ x : ℤ, -6 + x = 0 ∧ x = 6 :=
by
  use 6
  split
  · simp
  · rfl

end opposite_of_neg_six_l734_734938


namespace circumcenter_property_l734_734734

theorem circumcenter_property (n : ℕ) (h : n ≥ 5) : ∃ p : Finset (EuclideanSpace ℝ 2), p.card = n ∧ (∀ x ∈ p, ∃ a b c ∈ p, x = circumcenter a b c) → n ≥ 6 :=
by 
  sorry

end circumcenter_property_l734_734734


namespace minimum_value_8m_n_l734_734773

noncomputable def min_value (m n : ℝ) : ℝ :=
  8 * m + n

theorem minimum_value_8m_n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (8 / n) = 4) : 
  min_value m n = 8 :=
sorry

end minimum_value_8m_n_l734_734773


namespace segment_AB_length_l734_734049

def radius1 : ℝ := 1
def radius2 : ℝ := 2
def radius3 : ℝ := 3

def P : EuclideanSpace ℝ (Fin 2) := (0, 0)
def Q : EuclideanSpace ℝ (Fin 2) := (3, 0)
def R : EuclideanSpace ℝ (Fin 2) := (0, 4)

noncomputable def length_of_segment_AB (P Q R : EuclideanSpace ℝ (Fin 2)) (r1 r3 : ℝ) : ℝ := 
  if (dist P Q + dist Q R = dist P R ∧ dist P Q = r1 + r3) then
      r1 * Real.sqrt 2
  else 
    sorry

theorem segment_AB_length {P Q R : EuclideanSpace ℝ (Fin 2)}
  (hTangent1 : dist P Q = radius1 + radius2)
  (hTangent2 : dist P R = radius1 + radius3)
  (hTangent3 : dist Q R = radius2 + radius3)
  (hRightTriangle : (dist P Q)^2 + (dist P R)^2 = (dist Q R)^2) :
  length_of_segment_AB P Q R radius1 radius3 = Real.sqrt 2 := 
by
  sorry

end segment_AB_length_l734_734049


namespace product_mod_5_l734_734738

theorem product_mod_5:
  ∀ (x : ℕ), (10 ≤ x) → (x ≤ 99) → (x % 10 = 2 ∨ x % 10 = 3) →
  (∏ (x in Finset.range 100) (h : x % 10 = 2 ∨ x % 10 = 3), x) % 5 = 1 :=
by
  sorry

end product_mod_5_l734_734738


namespace negative_numbers_in_daily_life_l734_734977

-- Define the context and the assertion
def temperature_scenario := "The temperature on a winter day ranges from -2°C to 5°C."

-- Proof statement
theorem negative_numbers_in_daily_life : 
  "Write down a scenario in daily life where negative numbers are used" ∧
  temperature_scenario :=
by 
  split
  -- Assert that the question requires an example scenario with negative numbers
  . exact "Write down a scenario in daily life where negative numbers are used"
  -- Assert providing the temperature scenario as a valid answer
  . exact temperature_scenario

-- Proof (placeholder)
sorry

end negative_numbers_in_daily_life_l734_734977


namespace sin_cos_sum_l734_734720

open Real

theorem sin_cos_sum : sin (47 : ℝ) * cos (43 : ℝ) + cos (47 : ℝ) * sin (43 : ℝ) = 1 :=
by
  sorry

end sin_cos_sum_l734_734720


namespace sqrt_three_minus_a_is_real_l734_734827

theorem sqrt_three_minus_a_is_real (a : ℝ) : (∃ b : ℝ, b = √3 - a) → ∃ a : ℝ, a = 1 := 
by {
  intro h, 
  existsi (1 : ℝ), 
  sorry
}

end sqrt_three_minus_a_is_real_l734_734827


namespace minimum_omega_l734_734477

theorem minimum_omega (ω : ℝ) (h_omega_pos : ω > 0) :
    (∃ y : ℝ → ℝ, (∀ x, y x = sin (ω * x + ω * (π / 2) + (π / 3))) ∧ 
    (∀ x, y x = y (-x))) →
    (ω = 1 / 3) :=
sorry

end minimum_omega_l734_734477


namespace function_8_5_l734_734465

theorem function_8_5 (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f(x) = f(-x))
  (h_shiftodd : ∀ x : ℝ, f(x-1) = - f(1-x))
  (h_val : f(0.5) = 9) :
  f(8.5) = 9 :=
by
  sorry

end function_8_5_l734_734465


namespace jessica_total_monthly_payment_l734_734388

-- Definitions for the conditions
def basicCableCost : ℕ := 15
def movieChannelsCost : ℕ := 12
def sportsChannelsCost : ℕ := movieChannelsCost - 3

-- The statement to be proven
theorem jessica_total_monthly_payment :
  basicCableCost + movieChannelsCost + sportsChannelsCost = 36 := 
by
  sorry

end jessica_total_monthly_payment_l734_734388


namespace center_trajectory_l734_734277

-- Definitions
noncomputable def Circle (center : ℝ × ℝ) (radius : ℝ) := 
{c: ℝ × ℝ | dist c center = radius}

-- Given Conditions
variables (O1_center O2_center : ℝ × ℝ) (O1_radius O2_radius : ℝ)
variable (P_radius : ℝ)

-- The moving circle P is tangent to both fixed circles O1 and O2.
def is_tangent (A : ℝ × ℝ) (B_center : ℝ × ℝ) (B_radius : ℝ) : Prop :=
dist A B_center = P_radius + B_radius ∨ dist A B_center = abs (P_radius - B_radius)

-- The center of circle P follows a locus defined by its tangency to O1 and O2.
def center_locus (P_center : ℝ × ℝ) : Prop :=
is_tangent P_center O1_center O1_radius ∧ is_tangent P_center O2_center O2_radius

-- Proof problem: We prove that the locus where the center of P can be is either hyperbolas or a hyperbola and a straight line.
theorem center_trajectory :
  (∃ P_centers : set (ℝ × ℝ), ∀ P_center ∈ P_centers, center_locus P_center) →
  (∃ P_centers : set (ℝ × ℝ), 
    (∀ P_center ∈ P_centers, ∃ a b : ℝ × ℝ, (hyperbola P_center a b) ∨ (hyperbola P_center a b ∨ straight_line P_center a b)) ∧
   (P_centers.nonempty)) := 
sorry

-- Define hyperbola and straight line for completeness. 
-- Note: These are simple placeholders and need full definitions as per mathematical requirements.
def hyperbola (c : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
dist c a - dist c b = 1 

def straight_line (c : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
c = midpoint a b

end center_trajectory_l734_734277


namespace smallest_m_l734_734863

def complex_in_set_T (z : ℂ) : Prop :=
  ∃ x y : ℝ, z = x + y * complex.I ∧ (1/2 ≤ x ∧ x ≤ real.sqrt 3 / 2) ∧ (real.sqrt 2 / 2 ≤ y ∧ y ≤ 1)

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (z : ℂ) (x y : ℝ), z = x + y * complex.I ∧ complex_in_set_T z ∧ z^n = complex.I

theorem smallest_m :
  ∃ m : ℕ, (∀ n : ℕ, n ≥ m → satisfies_condition n) ∧ m = 6 :=
by
  sorry

end smallest_m_l734_734863


namespace find_equidistant_point_in_xy_plane_l734_734206

open Real

noncomputable def equidistant_point : Prop :=
  ∃ (x y : ℝ), 
  (x = 3.7 ∧ y = -0.6 ∧ 
   let p := (x, y, 0) in
   let d₁ := dist p (1, 0, -1) in
   let d₂ := dist p (2, 2, 1) in
   let d₃ := dist p (4, 1, -2) in
   d₁ = d₂ ∧ d₂ = d₃)

theorem find_equidistant_point_in_xy_plane : equidistant_point :=
sorry

end find_equidistant_point_in_xy_plane_l734_734206


namespace digits_divisible_by_7_digits_divisible_by_43_digits_divisible_by_41_digits_divisible_by_301_digits_divisible_by_3_digits_divisible_by_21_l734_734143

-- Case a): Divisibility by 7
theorem digits_divisible_by_7 (n : ℕ) (N : ℕ) (h1 : N = n * 7) (h2 : ∀k, N = 777...7 (written with identical digits)):
  ∃ k, n = 6 * k :=
sorry

-- Case b): Divisibility by 43
theorem digits_divisible_by_43 (n : ℕ) (N : ℕ) (h1 : N = n * 43) (h2 : ∀k, N = 111...1 (written with identical digits)):
  ∃ m, n = 21 * m :=
sorry

-- Case c): Divisibility by 41
theorem digits_divisible_by_41 (n : ℕ) (N : ℕ) (h1 : N = n * 41) (h2 : ∀k, N = 111...1 (written with identical digits)):
  ∃ m, n = 5 * m :=
sorry

-- Case d): Divisibility by 301
theorem digits_divisible_by_301 (n : ℕ) (N : ℕ) (h1 : N = n * 301) (h2 : ∀k, N = 777...7 (written with identical digits)):
  ∃ l, n = 42 * l :=
sorry

-- Case e): Divisibility by 3
theorem digits_divisible_by_3 (n : ℕ) (N : ℕ) (h1 : N = n * 3) (h2 : ∀k, N = 111...1 (written with identical digits)):
  ∃ s, n = 3 * s :=
sorry

-- Case f): Divisibility by 21
theorem digits_divisible_by_21 (n : ℕ) (N : ℕ) (h1 : N = n * 21) (hj : j = 7 ∨ j ≠ 7) (h2 : ∀k, N = jjj...j (written with identical digits)):
  (∃ s, n = 3 * s ∧ j = 7) ∨ (∃ k, n = 6 * k) :=
sorry

end digits_divisible_by_7_digits_divisible_by_43_digits_divisible_by_41_digits_divisible_by_301_digits_divisible_by_3_digits_divisible_by_21_l734_734143


namespace expression_change_l734_734415

variable (x b : ℝ)

-- The conditions
def expression (x : ℝ) : ℝ := x^3 - 5 * x + 1
def expr_change_plus (x b : ℝ) : ℝ := (x + b)^3 - 5 * (x + b) + 1
def expr_change_minus (x b : ℝ) : ℝ := (x - b)^3 - 5 * (x - b) + 1

-- The Lean statement to prove
theorem expression_change (h_b_pos : 0 < b) :
  expr_change_plus x b - expression x = 3 * b * x^2 + 3 * b^2 * x + b^3 - 5 * b ∨ 
  expr_change_minus x b - expression x = -3 * b * x^2 + 3 * b^2 * x - b^3 + 5 * b := 
by
  sorry

end expression_change_l734_734415


namespace prod_97_103_l734_734704

theorem prod_97_103 : (97 * 103) = 9991 := 
by 
  have h1 : 97 = 100 - 3 := by rfl
  have h2 : 103 = 100 + 3 := by rfl
  calc
    97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
         ... = 100^2 - 3^2 : by rw (mul_sub (100:ℤ) 3 3)
         ... = 10000 - 9 : by norm_num
         ... = 9991 : by norm_num
 
end prod_97_103_l734_734704


namespace mul_97_103_l734_734699

theorem mul_97_103 : (97:ℤ) = 100 - 3 → (103:ℤ) = 100 + 3 → 97 * 103 = 9991 := by
  intros h1 h2
  sorry

end mul_97_103_l734_734699


namespace vector_magnitude_sum_three_eq_l734_734404

variables (a b c : ℝ) (F : ℝ × ℝ) (A B C : ℝ × ℝ)

-- Definitions from conditions
def ellipse := ∀ x y : ℝ, (x^2 / 16 + y^2 / 4 = 1)
def F_focus := F = (-2 * real.sqrt 3, 0)
def points_on_ellipse := (A.1^2 / 16 + A.2^2 / 4 = 1) ∧ (B.1^2 / 16 + B.2^2 / 4 = 1) ∧ (C.1^2 / 16 + C.2^2 / 4 = 1)
def vector_sum_zero := (A.1 + 2 * real.sqrt 3, A.2) + (B.1 + 2 * real.sqrt 3, B.2) + (C.1 + 2 * real.sqrt 3, C.2) = (0, 0)

-- Function to calculate |FA| for given point
noncomputable def magnitude_FA (X : ℝ × ℝ) := real.sqrt ((X.1 + 2 * real.sqrt 3)^2 + X.2^2)

theorem vector_magnitude_sum_three_eq :
  ellipse ∧ F_focus ∧ points_on_ellipse ∧ vector_sum_zero →
  magnitude_FA A + magnitude_FA B + magnitude_FA C = 3 :=
by
  sorry

end vector_magnitude_sum_three_eq_l734_734404


namespace solve_system_l734_734205

theorem solve_system :
  ∃ (x y : ℚ), (4 * x - 35 * y = -1) ∧ (3 * y - x = 5) ∧ (x = -172 / 23) ∧ (y = -19 / 23) :=
by
  sorry

end solve_system_l734_734205


namespace relationship_x_y_l734_734231

variable (a b x y : ℝ)

theorem relationship_x_y (h1: 0 < a) (h2: a < b)
  (hx : x = (Real.sqrt (a + b) - Real.sqrt b))
  (hy : y = (Real.sqrt b - Real.sqrt (b - a))) :
  x < y :=
  sorry

end relationship_x_y_l734_734231


namespace compatibility_condition_l734_734439

theorem compatibility_condition (a b c d x : ℝ) 
  (h1 : a * x + b = 0) (h2 : c * x + d = 0) : a * d - b * c = 0 :=
sorry

end compatibility_condition_l734_734439


namespace num_ways_to_score_19_points_from_14_matches_l734_734514

/-- 
Given the following conditions:
1. Each win scores 3 points.
2. Each draw scores 1 point.
3. Each loss scores 0 points.
4. A team plays 14 matches.

Prove that there are exactly 4 ways for a team to accumulate a total of 19 points from these matches.
-/
def ways_to_score_19_points_from_14_matches {wins draws losses : ℕ} : Prop :=
  wins * 3 + draws * 1 = 19 ∧ wins + draws + losses = 14

theorem num_ways_to_score_19_points_from_14_matches : 
  (∃ wins draws losses, ways_to_score_19_points_from_14_matches wins draws losses) = 4 :=
sorry

end num_ways_to_score_19_points_from_14_matches_l734_734514


namespace area_of_circumscribed_triangle_l734_734921

-- Define the conditions
def distance_to_chord : ℝ := 15
def chord_length : ℝ := 16
def triangle_perimeter : ℝ := 200

-- The goal is to prove that the area of the triangle is 1700 cm².
theorem area_of_circumscribed_triangle :
  let R := Real.sqrt (distance_to_chord^2 + (chord_length / 2)^2) in
  let p := triangle_perimeter / 2 in
  (R * p) = 1700 :=
by
  sorry

end area_of_circumscribed_triangle_l734_734921


namespace unit_prices_min_total_cost_l734_734219

-- Part (1): Proving the unit prices of ingredients A and B.
theorem unit_prices (x y : ℝ)
    (h₁ : x + y = 68)
    (h₂ : 5 * x + 3 * y = 280) :
    x = 38 ∧ y = 30 :=
by
  -- Sorry, proof not provided
  sorry

-- Part (2): Proving the minimum cost calculation.
theorem min_total_cost (m : ℝ)
    (h₁ : m + (36 - m) = 36)
    (h₂ : m ≥ 2 * (36 - m)) :
    (38 * m + 30 * (36 - m)) = 1272 :=
by
  -- Sorry, proof not provided
  sorry

end unit_prices_min_total_cost_l734_734219


namespace tankB_one_third_full_depth_tankA_when_tankB_full_equal_depths_depth_TankD_when_volumes_equal_l734_734458

-- Declaring cetain units for easy usage
variables (time : Type*) [linear_ordered_field time] -- We are using linear_ordered_field for simplicity here

-- Dimensions Conditions for Tanks A and B (rectangular prisms)
def TankA_volume : ℝ := 10 * 8 * 6

def TankB_volume : ℝ := 5 * 9 * 8

-- Rates
def drain_fill_rate := 4.0 -- cm³/s for both filling Tank B and draining Tank A

-- Problem (i)
theorem tankB_one_third_full (t : ℝ) (h : t = 30) : 
  4 * t = (TankB_volume / 3) :=
sorry

-- Problem (ii)
theorem depth_tankA_when_tankB_full : 
  ∀ (h : ℝ), h = (TankA_volume - TankB_volume) / (10 * 8) → 
  h = 1.5 :=
sorry

-- Problem (iii)
theorem equal_depths : 
  ∀ (h : ℝ), 10 * 8 * h = 5 * 9 * h → 
  10 * 8 * h = 5 * 9 * h := 
sorry

-- Dimensions Conditions for Tank C and D
def TankC_volume : ℝ := 31 * 4 * 4

def TankD_height := 10
def TankD_base_side := 20

-- Filling Rates
def fill_rate_D := 1.0 -- cm³/s for filling Tank D
def drain_rate_C := 2.0 -- cm³/s for draining Tank C after 2s

-- Problem (iv)
theorem depth_TankD_when_volumes_equal (t : ℝ) (hD : ℝ) : 
  496.0 - 2.0*(t - 2.0) = 1/3 * (20 * 20 * hD) ∧ t = 10 → 
  496.0 - 2.0*(t - 2.0) = 1/3 * (20 * 20 * hD) := 
sorry

end tankB_one_third_full_depth_tankA_when_tankB_full_equal_depths_depth_TankD_when_volumes_equal_l734_734458


namespace min_omega_symmetry_l734_734502

theorem min_omega_symmetry :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * x + ω * (π / 2) + π / 3) = sin ((-ω) * x + ω * (π / 2) + π / 3)) →
  ω = 1 / 3 :=
by {
  sorry
}

end min_omega_symmetry_l734_734502


namespace coordinates_of_point_P_l734_734015

theorem coordinates_of_point_P : 
  (∃ x : ℝ, (x = 3 ∨ x = -3) ∧ (0 = 0) ∧ (real.sqrt (x^2 + 0^2) = 3)) →
  (P : ℝ × ℝ) → P = (3, 0) ∨ P = (-3, 0) :=
by
  sorry

end coordinates_of_point_P_l734_734015


namespace volume_of_large_ball_l734_734829

theorem volume_of_large_ball (r : ℝ) (V_small : ℝ) (h1 : 1 = r / (2 * r)) (h2 : V_small = (4 / 3) * Real.pi * r^3) : 
  8 * V_small = 288 :=
by
  sorry

end volume_of_large_ball_l734_734829


namespace maximize_profit_at_six_l734_734061

-- Defining the functions (conditions)
def y1 (x : ℝ) : ℝ := 17 * x^2
def y2 (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := y1 x - y2 x

-- The condition x > 0
def x_pos (x : ℝ) : Prop := x > 0

-- Proving the maximum profit is achieved at x = 6 (question == answer)
theorem maximize_profit_at_six : ∀ x > 0, (∀ y > 0, y = profit x → x = 6) :=
by 
  intros x hx y hy
  sorry

end maximize_profit_at_six_l734_734061


namespace number_of_factors_27648_l734_734283

-- Define the number in question
def n : ℕ := 27648

-- State the prime factorization
def n_prime_factors : Nat := 2^10 * 3^3

-- State the theorem to be proven
theorem number_of_factors_27648 : 
  ∃ (f : ℕ), 
  (f = (10+1) * (3+1)) ∧ (f = 44) :=
by
  -- Placeholder for the proof
  sorry

end number_of_factors_27648_l734_734283


namespace find_third_number_l734_734461

theorem find_third_number (x : ℕ) :
  let avg := (10 + 70 + 19) / 3 in
  let new_avg := avg + 7 in
  let sum := 20 + 40 + x in
  (sum / 3 = new_avg) → x = 60 :=
by
  intros avg new_avg sum h
  sorry

end find_third_number_l734_734461


namespace minimum_omega_for_symmetric_curve_l734_734472

theorem minimum_omega_for_symmetric_curve (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 2) + π / 3) = sin (-ω * (x + π / 2) + π / 3)) ↔ ω = 1 / 3 :=
by
  sorry

end minimum_omega_for_symmetric_curve_l734_734472


namespace cos_squared_sum_eq_l734_734693

theorem cos_squared_sum_eq :
  ∑ i in finset.range (91), (float.cos (45 + i : ℝ))^2 = 68.25 :=
by
  sorry

end cos_squared_sum_eq_l734_734693


namespace miles_driven_l734_734853

def total_miles : ℕ := 1200
def remaining_miles : ℕ := 432

theorem miles_driven : total_miles - remaining_miles = 768 := by
  sorry

end miles_driven_l734_734853


namespace find_m_l734_734585

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l734_734585


namespace length_of_BD_is_8_9_l734_734378

noncomputable def length_of_BD (AB AC BE : ℝ) (h1 : AB = BE) (h2 : AC = 13) (h3 : BE = 12) : ℝ :=
  let x := Real.sqrt (AC^2 + BE^2)
  let BD := x / 2
  BD

theorem length_of_BD_is_8_9 : Real.round (length_of_BD 13 12) = 8.9 := by
  sorry

end length_of_BD_is_8_9_l734_734378


namespace equation_of_C3_l734_734253

variable {X Y : Type}
variable (f : X → Y)
variable (f_inv : Y → X)

-- Given that f has an inverse
axiom h1 : ∀ y, f (f_inv y) = y
axiom h2 : ∀ x, f_inv (f x) = x

-- Define the transformation functions
def translation_left (y : Y) : Y := f (y + 1)
def translation_up (y : Y) : Y := y + 1
def reflection (y : Y) : X := f_inv y

-- The final function after transformations
def C3 (x : X) : Y := f_inv (x - 1) - 1

theorem equation_of_C3 :
  ∀ x, C3 f f_inv x = f_inv (x - 1) - 1 :=
by
  intro x
  sorry

end equation_of_C3_l734_734253


namespace carrie_phone_charges_l734_734888

def total_miles (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def charges_needed (total_miles charge_miles : ℕ) : ℕ :=
  total_miles / charge_miles + if total_miles % charge_miles = 0 then 0 else 1

theorem carrie_phone_charges :
  let d1 := 135
  let d2 := 135 + 124
  let d3 := 159
  let d4 := 189
  let charge_miles := 106
  charges_needed (total_miles d1 d2 d3 d4) charge_miles = 7 :=
by
  sorry

end carrie_phone_charges_l734_734888


namespace Mary_younger_by_14_l734_734151

variable (Betty_age : ℕ) (Albert_age : ℕ) (Mary_age : ℕ)

theorem Mary_younger_by_14 :
  (Betty_age = 7) →
  (Albert_age = 4 * Betty_age) →
  (Albert_age = 2 * Mary_age) →
  (Albert_age - Mary_age = 14) :=
by
  intros
  sorry

end Mary_younger_by_14_l734_734151


namespace distance_between_centers_l734_734835

-- Given Conditions
def right_triangle (A B C : Type) (right_angle : B → Prop) := ∃ (angle_A : B → Prop), angle_A A = 30

def bisector_of_acute_angle (B D : Type) :=  ∃ A C : Type, right_triangle A B C ∧ ∃ (P : B → Prop), (P D) = (P B)/2

def centers_distance (A B C D : Type) (inscribed_circle : B → Prop) :=
  inscribed_circle A ∧ inscribed_circle B ∧ inscribed_circle C ∧ inscribed_circle D

theorem distance_between_centers {A B C D : Type}
  (hT : right_triangle A B C)
  (hB : bisector_of_acute_angle B D)
  (hBD : centers_distance A B C D)
  : distance_centers(A:Type,B:Type,Inscribed_circle)= \(\frac{\sqrt{96-54\sqrt{3}}}{3}\):
  ∃ (hE : centers_distance A B) (hE_eq : distance_centers(A:Type,B:Type,Inscribed_circle)= \(\frac{\sqrt{96-54\sqrt{3}}}{3}\)):  sorry

end distance_between_centers_l734_734835


namespace number_of_possible_medians_l734_734417

-- We define S as a set of eleven distinct integers such that 
-- it includes at least the elements 5, 7, 8, 10, 12, and 15.
def exists_set_with_conditions : Prop :=
  ∃ (S : Set ℤ), S.card = 11 ∧ 
                 5 ∈ S ∧ 7 ∈ S ∧ 8 ∈ S ∧ 10 ∈ S ∧ 12 ∈ S ∧ 15 ∈ S

-- We now define the main theorem which states that the number of possible medians
-- of such a set S is exactly 6.
theorem number_of_possible_medians (S : Set ℤ) (h1 : S.card = 11) 
                                   (h2 : 5 ∈ S) (h3 : 7 ∈ S) (h4 : 8 ∈ S)
                                   (h5 : 10 ∈ S) (h6 : 12 ∈ S) (h7 : 15 ∈ S) 
                                   : ∃ (N : ℕ), N = 6 :=
  sorry

end number_of_possible_medians_l734_734417


namespace sum_of_x_coords_intersection_with_y_eq_x_plus_2_l734_734045

open List

def segment (p1 p2 : (ℝ × ℝ)) := ∃ a b, (0 ≤ a ∧ a ≤ 1) ∧ (p1.1 + a * (p2.1 - p1.1), p1.2 + a * (p2.2 - p1.2)) = (p1.1 + a * (p2.1 - p1.1), p1.2 + a * (p2.2 - p1.2))

def g_segments : List (ℝ × ℝ) := [(-4, -3), (-2, 0), (-1, -2), (1, 3), (2, 2), (4, 6)]

theorem sum_of_x_coords_intersection_with_y_eq_x_plus_2 : 
  let intersections := filter (λ p, p.2 = p.1 + 2) (g_segments.zip (tail g_segments))
  sum (map Prod.fst intersections) = 3 := by
  sorry

end sum_of_x_coords_intersection_with_y_eq_x_plus_2_l734_734045


namespace count_lineup_excluding_youngest_l734_734352

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l734_734352


namespace degree_Q_Q_sqrt_x_roots_l734_734128

variables {n : ℕ} {a_0 a_1 a_n_1 : ℤ} {P Q : ℤ[X]}
variables {x1 x2 ... xn : ℂ} {x : ℂ}

noncomputable def P (x : ℂ) := ∑ i in 0..n, a_i * x^i
noncomputable def Q (x : ℂ) := P(x) * P(-x)

-- Statement a: Proving that Q(x) is a polynomial of degree 2n and contains only even powers of x.
theorem degree_Q (P : Polynomial ℂ) (Q : Polynomial ℂ) :
  Q.degree = 2 * n ∧ (∀ k, Q.coeff k = 0 → odd k) := sorry

-- Statement b: Proving that Q(√x) is a polynomial with roots x1², x2², ..., xn².
theorem Q_sqrt_x_roots (x1 x2 ... xn : ℂ) :
  (Q (complex.sqrt x)).roots = [x1^2, x2^2, ..., xn^2] := sorry

end degree_Q_Q_sqrt_x_roots_l734_734128


namespace find_angle_between_line_and_base_l734_734042

noncomputable def angle_between_line_and_base {a b : ℝ} (h1 : a > 0) (h2 : b > 0) :=
  let angle := (1 / 2 : ℝ) * Real.arccos (-(4 * b^2) / a^2) in
  angle

theorem find_angle_between_line_and_base (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (AB_len : a)
  (distance_from_axis : b) :
  angle_between_line_and_base ha hb = (1 / 2 : ℝ) * Real.arccos (-(4 * b^2) / a^2) :=
begin
  sorry
end

end find_angle_between_line_and_base_l734_734042


namespace budget_percentage_l734_734393

-- Define the given conditions
def basic_salary_per_hour : ℝ := 7.50
def commission_rate : ℝ := 0.16
def hours_worked : ℝ := 160
def total_sales : ℝ := 25000
def amount_for_insurance : ℝ := 260

-- Define the basic salary, commission, and total earnings
def basic_salary : ℝ := basic_salary_per_hour * hours_worked
def commission : ℝ := commission_rate * total_sales
def total_earnings : ℝ := basic_salary + commission
def amount_for_budget : ℝ := total_earnings - amount_for_insurance

-- Define the proof problem
theorem budget_percentage : (amount_for_budget / total_earnings) * 100 = 95 := by
  simp [basic_salary, commission, total_earnings, amount_for_budget]
  sorry

end budget_percentage_l734_734393


namespace rainfall_tuesday_l734_734849

theorem rainfall_tuesday :
  let hours_monday := 7 in
  let rate_monday := 1 in
  let hours_wednesday := 2 in
  let rate_tuesday := 2 in
  let rate_wednesday := 2 * rate_tuesday in
  let total_rainfall := 23 in
  let rainfall_monday := hours_monday * rate_monday in
  let rainfall_wednesday := hours_wednesday * rate_wednesday in
  let rainfall_tuesday := total_rainfall - (rainfall_monday + rainfall_wednesday) in
  rainfall_tuesday / rate_tuesday = 4 :=
  sorry

end rainfall_tuesday_l734_734849


namespace fraction_simplification_l734_734698

theorem fraction_simplification :
  (1722 ^ 2 - 1715 ^ 2) / (1729 ^ 2 - 1708 ^ 2) = 1 / 3 := by
  sorry

end fraction_simplification_l734_734698


namespace find_f2_l734_734754

noncomputable def f : ℝ → ℝ := sorry

-- Assume f is a monotonic function
axiom monotonic_f : Monotonic f

-- Given condition
axiom condition : ∀ x : ℝ, f (f x - real.exp x) = real.exp 1 + 1

-- Prove that f(2) = e^2 + 1
theorem find_f2 : f 2 = real.exp 2 + 1 := 
by sorry

end find_f2_l734_734754


namespace quadrilateral_inscribed_in_circle_l734_734444

theorem quadrilateral_inscribed_in_circle (A B C D : Point) (k : ℝ) (angle_ABC : ℝ) 
  (h1 : inscribed_in_circle A B C D)
  (h2 : angle_ABC = 60) 
  (h3 : dist B C = k) 
  (h4 : dist C D = k) : 
  dist A B = dist A D + dist D C :=
sorry

end quadrilateral_inscribed_in_circle_l734_734444


namespace measure_angle_RSQ_72_l734_734840

open Real

-- Definitions from conditions
variable (P Q R S : Point)
variable (angle : Angle)
variable (right_angle : angle P Q R = 90)
variable (angle_QPR : angle Q P R = 54)
variable (S_on_PQ : lies_on S P Q)
variable (PRS_eq_QRS : angle P R S = angle Q R S)

-- Hypothesis and result statement
theorem measure_angle_RSQ_72 : angle R S Q = 72 := by
  sorry

end measure_angle_RSQ_72_l734_734840


namespace numbers_between_200_and_599_with_3_or_5_l734_734286

def hasDigit3Or5 (n : ℕ) : Prop :=
  let s := n.toString
  s.contains '3' ∨ s.contains '5'

def totalNumbersWith3Or5InRange200To599 : ℕ :=
  (200..600).filter hasDigit3Or5 |>.length

theorem numbers_between_200_and_599_with_3_or_5 :
  totalNumbersWith3Or5InRange200To599 = 320 := sorry

end numbers_between_200_and_599_with_3_or_5_l734_734286


namespace square_side_length_increase_l734_734507

theorem square_side_length_increase (a : ℝ) (c : ℝ):
  let b := 2 * a in
  let sum_areas := a * a + b * b in
  1.45 * sum_areas + sum_areas = c * c → 
  (c / b - 1) * 100 = 75 :=
by
  sorry

end square_side_length_increase_l734_734507


namespace roots_n_not_divisible_by_5_for_any_n_l734_734016

theorem roots_n_not_divisible_by_5_for_any_n (x1 x2 : ℝ) (n : ℕ)
  (hx : x1^2 - 6 * x1 + 1 = 0)
  (hy : x2^2 - 6 * x2 + 1 = 0)
  : ¬(∃ (k : ℕ), (x1^k + x2^k) % 5 = 0) :=
sorry

end roots_n_not_divisible_by_5_for_any_n_l734_734016


namespace omega_min_value_l734_734487

theorem omega_min_value (ω : ℝ) (hω : ω > 0)
    (hSymmetry : ∀ x : ℝ, sin (ω * x + ω * π / 2 + π / 3) = sin (ω * -x + ω * π / 2 + π / 3)) :
    ω = 1 / 3 :=
begin
  sorry
end

end omega_min_value_l734_734487


namespace binomial_problem_l734_734174

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The problem statement: prove that binomial(13, 11) * 2 = 156
theorem binomial_problem : binomial 13 11 * 2 = 156 := by
  sorry

end binomial_problem_l734_734174


namespace volume_of_rectangular_prism_l734_734621

theorem volume_of_rectangular_prism
  (l w h : ℝ)
  (Hlw : l * w = 10)
  (Hwh : w * h = 15)
  (Hlh : l * h = 6) : l * w * h = 30 := 
by
  sorry

end volume_of_rectangular_prism_l734_734621


namespace alternating_sum_l734_734558

theorem alternating_sum : (Finset.range 10002).sum (λ n, if n % 2 = 0 then n + 1 else -(n + 1)) = -5001 := 
by
  sorry

end alternating_sum_l734_734558


namespace consecutive_integers_sum_and_difference_l734_734943

theorem consecutive_integers_sum_and_difference (x y : ℕ) 
(h1 : y = x + 1) 
(h2 : x * y = 552) 
: x + y = 47 ∧ y - x = 1 :=
by {
  sorry
}

end consecutive_integers_sum_and_difference_l734_734943


namespace sufficient_not_necessary_condition_l734_734227

variable (a : ℝ)

def M := {1, a}
def N := {-1, 0, 1}

theorem sufficient_not_necessary_condition : 
  (M ⊆ N ↔ (a = 0 ∨ a = -1)) → (M ⊆ N) ∧ (a = 0 → M ⊆ N) ∧ ¬(a = 0 → ¬M ⊆ N) :=
by 
sorry

end sufficient_not_necessary_condition_l734_734227


namespace g_sum_l734_734421

def g (x : ℝ) : ℝ :=
if x > 3 then x^2 - 4 else
if -3 ≤ x ∧ x ≤ 3 then 5 * x + 2 else 0

theorem g_sum :
  g (-4) + g 0 + g 4 = 14 :=
by {
  -- Define the function g and regions for clarity
  have h1 : g (-4) = 0,
  { simp [g], -- x < -3 region
    split_ifs,
    exact h },
    
  have h2 : g 0 = 2,
  { simp [g], -- -3 ≤ x ≤ 3 region
    split_ifs,
    exact h,
    linarith [h] },
    
  have h3 : g 4 = 12,
  { simp [g], -- x > 3 region
    split_ifs,
    exact h,
    linarith [h] },
  -- Combine results
  linarith [h1, h2, h3]
}

#print g_sum

end g_sum_l734_734421


namespace triangle_sides_inequality_l734_734831

theorem triangle_sides_inequality
  {A B C a b c : ℝ}
  (triangle_ABC : ∃ (a b c : ℝ), a = b * (sin A / sin 60) ∧ c = b * (sin C / sin 60))
  (angle_B_eq_60 : B = 60)
  (side_sum_eq_one : a + c = 1)
  (A_range : 0 < A ∧ A < 120)
  : 0.5 ≤ b ∧ b < 1 :=
  sorry

end triangle_sides_inequality_l734_734831


namespace find_y_l734_734556

theorem find_y (y : ℝ) : (60 / 100 = sqrt ((y + 20) / 100)) → y = 16 :=
by
  -- Skip actual proof steps with 'sorry'
  sorry

end find_y_l734_734556


namespace smallest_divisible_by_15_16_18_l734_734740

def factors_of_15 : Prop := 15 = 3 * 5
def factors_of_16 : Prop := 16 = 2^4
def factors_of_18 : Prop := 18 = 2 * 3^2

theorem smallest_divisible_by_15_16_18 (h1: factors_of_15) (h2: factors_of_16) (h3: factors_of_18) : 
  ∃ n, n > 0 ∧ n % 15 = 0 ∧ n % 16 = 0 ∧ n % 18 = 0 ∧ n = 720 :=
by
  sorry

end smallest_divisible_by_15_16_18_l734_734740


namespace binomial_coeff_n_eq_10_l734_734369

theorem binomial_coeff_n_eq_10 (n : ℕ) (h : binomial (n, 2) * 3^(n-2) = 5 * 3^n) : n = 10 :=
by
  sorry

end binomial_coeff_n_eq_10_l734_734369


namespace ratio_of_squares_side_lengths_l734_734945

theorem ratio_of_squares_side_lengths :
  ∀ (a b c : ℕ), (∃ (ratio : ℚ), ratio = 270 / 125 ∧ (∃ (side_ratio : ℚ), side_ratio = real.sqrt ratio ∧
  (a = 3 ∧ b = 30 ∧ c = 25 ∧ side_ratio = a * real.sqrt b / c))) →
  a + b + c = 58 :=
begin
  intros a b c h,
  obtain ⟨ratio, hr1, ⟨side_ratio, hs1, ha⟩⟩ := h,
  have ha1 := ha.1,
  have hb := ha.2.1,
  have hc := ha.2.2.1,
  have hs2 := ha.2.2.2,
  rw [ha1, hb, hc],
  exact (3 + 30 + 25),
end

end ratio_of_squares_side_lengths_l734_734945


namespace minimize_sum_of_squares_l734_734591

noncomputable def minimumAP2BP2CP2DP2EP2 {α : Type*} [NormedSpace α ℝ] (A B C D E P : α) :=
  ∥P - A∥^2 + ∥P - B∥^2 + ∥P - C∥^2 + ∥P - D∥^2 + ∥P - E∥^2

theorem minimize_sum_of_squares :
  let A := (0 : ℝ), B := 1, C := 4, D := 8, E := 13 in
  ∃ P : ℝ, minimumAP2BP2CP2DP2EP2 A B C D E P = 114.8 :=
by
  sorry

end minimize_sum_of_squares_l734_734591


namespace four_units_away_l734_734011

theorem four_units_away (x : ℤ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 :=
by
  sorry

end four_units_away_l734_734011


namespace count_restricted_arrangements_l734_734344

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l734_734344


namespace rajas_income_l734_734124

theorem rajas_income (I : ℝ) 
  (h1 : 0.60 * I + 0.10 * I + 0.10 * I + 5000 = I) : I = 25000 :=
by
  sorry

end rajas_income_l734_734124


namespace vasya_fraction_l734_734667

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l734_734667


namespace correct_calculation_result_l734_734975

theorem correct_calculation_result :
  (∃ x : ℤ, 14 * x = 70) → (5 - 6 = -1) :=
by
  sorry

end correct_calculation_result_l734_734975


namespace value_expression_at_5_l734_734292

theorem value_expression_at_5 (x : ℕ) (hx : x = 5) : 2 * x^2 + 4 = 54 :=
by
  -- Adding sorry to skip the proof.
  sorry

end value_expression_at_5_l734_734292


namespace period_of_y_l734_734551

def y (x : ℝ) : ℝ := 2 * Real.sin x + 3 * Real.cos x

theorem period_of_y : ∀ x, y(x + 2 * Real.pi) = y(x) := sorry

end period_of_y_l734_734551


namespace max_dist_between_complex_vals_l734_734871

noncomputable def max_distance (z : ℂ) (hz : ∥z∥ = 1) : ℝ :=
  let a := (1 : ℂ) + (2 : ℂ) * complex.I in
  ∥a - z^2∥

theorem max_dist_between_complex_vals (z : ℂ) (hz : ∥z∥ = 1) :
  max_distance z hz = real.sqrt 5 + 1 :=
  sorry

end max_dist_between_complex_vals_l734_734871


namespace sort_descending_in_operations_l734_734520

theorem sort_descending_in_operations (n : ℕ) (h : n > 0) :
  ∀ (cards : list ℕ), (∀ x ∈ cards, x ∈ finset.range n.succ) ∧ (cards.nodup) →
  ∃ (ops : list (ℕ × ℕ)), ops.length ≤ (n * (n - 1)) / 2 ∧
  sorted (≥) (apply_swaps ops cards) :=
by
  sorry

end sort_descending_in_operations_l734_734520


namespace product_of_slopes_constant_value_l734_734769

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  x^2 / 4 + y^2 / 3 = 1

theorem product_of_slopes_constant_value (a b M A B x1 x2 y1 y2 k K1 K2: ℝ)
  (h1 : 2 * a = 4)
  (h2 : a > b)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : c = 1) 
  (h6 : a = 2) 
  (h7: b^2 = 3)
  (h8 : y1 = k * x1)
  (h9 : y2 = k * x2)
  (h10 : ellipse_eq M.fst M.snd)
  (h11 : ellipse_eq x1 y1)
  (h12 : ellipse_eq x2 y2) :
  K1 * K2 = -3 / 4 := sorry 

end product_of_slopes_constant_value_l734_734769


namespace solve_eq_l734_734451

noncomputable def final_solution (x : ℝ) : Prop :=
  ∃ n : ℤ, x = -real.pi / 14 + (2 * real.pi * n / 7) ∨ ∃ s : ℤ, x = 2 * real.pi * s

theorem solve_eq : 
  ( ∃ (x : ℝ), sin (7 * x) + real.sqrt (real.sqrt (1 - (cos (3 * x))^11 * (cos (7 * x))^2)) = 0 ) → 
  ∃ x : ℝ, final_solution x :=
sorry

end solve_eq_l734_734451


namespace find_p_q_of_divisibility_l734_734293

theorem find_p_q_of_divisibility 
  (p q : ℤ) 
  (h1 : (x + 3) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  (h2 : (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  : p = -31 ∧ q = -71 :=
by
  sorry

end find_p_q_of_divisibility_l734_734293


namespace gcd_p_q_l734_734869

noncomputable def p : ℕ := 33333333
noncomputable def q : ℕ := 777777777

theorem gcd_p_q : p.gcd(q) = 2 := by
  sorry

end gcd_p_q_l734_734869


namespace hexagon_angle_in_a_progression_l734_734915

theorem hexagon_angle_in_a_progression (x : ℝ) (common_diff : ℝ := 10) (total_sum : ℝ := 720) (num_sides : ℕ := 6) 
  (sum_formula : ℝ := (num_sides - 2) * 180) :
  x + (x + 10) + (x + 20) + (x + 30) + (x + 40) + (x + 50) = total_sum → x = 95 :=
begin
  assume h,
  have hexagon_sum : 6 * x + 150 = total_sum,
  { linarith, },
  have hexagon_solution : x = 95,
  { sorry, },
  exact hexagon_solution
end

end hexagon_angle_in_a_progression_l734_734915


namespace proof_intersection_l734_734273

def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Real := {x | -2 < x ∧ x < 2}

theorem proof_intersection (A_inter_B : Set Int) :
  (A_inter_B = {x ∈ A | x ∈ B}) -> 
  A_inter_B = {-1, 0, 1} :=
by 
  sorry

end proof_intersection_l734_734273


namespace number_of_true_propositions_is_three_l734_734160

-- Definitions for the propositions
def proposition1 (p q : Prop) : Prop :=
  (¬ (p ∧ q)) → (¬ p ∧ ¬ q)

def proposition2 : Prop :=
  ¬(∀ x: ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x: ℝ, x^3 - x^2 + 1 > 0)

def proposition3 (p q : Prop) : Prop := 
  ¬p → q

-- The condition involving the normal distribution
def proposition4 (X : ℝ → ℝ) (C : ℝ) : Prop :=
  (∀ C, P(X > C+1) = P(X < C-1)) → C = 3

-- The main theorem to prove
theorem number_of_true_propositions_is_three
: ∀ (p q : Prop) (X : ℝ → ℝ) (P : set ℝ → ℝ) (C : ℝ), 
  (proposition1 p q → False) ∧
  (proposition2) ∧
  ((¬p → q) ∧ (q → ¬p) → False) ∧
  (proposition4 X C)
  →
  (count_true [proposition1 p q, proposition2, proposition3 p q, proposition4 X C] = 3) :=
sorry

end number_of_true_propositions_is_three_l734_734160


namespace molecular_weight_of_one_mole_l734_734085

-- Definitions derived from the conditions in the problem:

def molecular_weight_nine_moles (w : ℕ) : ℕ :=
  2664

def molecular_weight_one_mole (w : ℕ) : ℕ :=
  w / 9

-- The theorem to prove, based on the above definitions and conditions:
theorem molecular_weight_of_one_mole (w : ℕ) (hw : molecular_weight_nine_moles w = 2664) :
  molecular_weight_one_mole w = 296 :=
sorry

end molecular_weight_of_one_mole_l734_734085


namespace smallest_of_powers_l734_734112

theorem smallest_of_powers :
  (2:ℤ)^(55) < (3:ℤ)^(44) ∧ (2:ℤ)^(55) < (5:ℤ)^(33) ∧ (2:ℤ)^(55) < (6:ℤ)^(22) := by
  sorry

end smallest_of_powers_l734_734112


namespace vitya_older_than_masha_probability_l734_734531

-- Definition of Days in June
def days_in_june : ℕ := 30

-- Total number of possible outcomes for birth dates (30 days for Vitya × 30 days for Masha)
def total_outcomes : ℕ := days_in_june * days_in_june

-- Sum of favorable outcomes where Vitya is at least one day older than Masha
def favorable_outcomes : ℕ := (1 to (days_in_june - 1)).sum

-- The probability calculation function
noncomputable def probability (n f : ℕ) : ℚ := ⟨f, n⟩

-- The statement of the theorem
theorem vitya_older_than_masha_probability :
  probability total_outcomes favorable_outcomes = 29 / 60 := sorry

end vitya_older_than_masha_probability_l734_734531


namespace find_smallest_positive_c_l734_734164

theorem find_smallest_positive_c : 
  ∃ c : ℝ, 0 < c ∧ (∀ x, (3 * sin(2 * x + c) + 1 = 1) → (x = π / 4)) ∧ c = π :=
begin
  sorry
end

end find_smallest_positive_c_l734_734164


namespace smallest_of_powers_l734_734111

theorem smallest_of_powers :
  min (2^55) (min (3^44) (min (5^33) (6^22))) = 2^55 :=
by
  sorry

end smallest_of_powers_l734_734111


namespace count_restricted_arrangements_l734_734342

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l734_734342


namespace factorial_expression_l734_734685

theorem factorial_expression : 8! - 7 * 7! - 2 * 6! = 3600 := by
  sorry

end factorial_expression_l734_734685


namespace sum_of_digits_at_positions_l734_734004

noncomputable theory

def resulting_sequence (s : List ℕ) : List ℕ :=
  let s1 := s.enum.filter (λ x, x.1 % 2 ≠ 1).unzip.2
  let s2 := s1.enum.filter (λ x, x.1 % 3 ≠ 2).unzip.2
  s2.enum.filter (λ x, x.1 % 4 ≠ 3).unzip.2

def repeating_sequence := List.range' 1 9

def final_sequence := (List.replicate (2024/8 + 1) (resulting_sequence (repeating_sequence))).join

def sum_positions (n1 n2 n3 : ℕ) (s : List ℕ) := s.get (n1 - 1) + s.get (n2 - 1) + s.get (n3 - 1)

theorem sum_of_digits_at_positions :
  sum_positions 2022 2023 2024 final_sequence = 13 :=
sorry

end sum_of_digits_at_positions_l734_734004


namespace pascal_triangle_third_number_l734_734102

theorem pascal_triangle_third_number {n k : ℕ} (h : n = 51) (hk : k = 2) : Nat.choose n k = 1275 :=
by
  rw [h, hk]
  norm_num

#check pascal_triangle_third_number

end pascal_triangle_third_number_l734_734102


namespace probability_of_Z_l734_734148

/-
  Given: 
  - P(W) = 3 / 8
  - P(X) = 1 / 4
  - P(Y) = 1 / 8

  Prove: 
  - P(Z) = 1 / 4 when P(Z) = 1 - (P(W) + P(X) + P(Y))
-/

theorem probability_of_Z (P_W P_X P_Y P_Z : ℚ) (h_W : P_W = 3 / 8) (h_X : P_X = 1 / 4) (h_Y : P_Y = 1 / 8) (h_Z : P_Z = 1 - (P_W + P_X + P_Y)) : 
  P_Z = 1 / 4 :=
by
  -- We can write the whole Lean Math proof here. However, per the instructions, we'll conclude with sorry.
  sorry

end probability_of_Z_l734_734148


namespace count_restricted_arrangements_l734_734341

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l734_734341


namespace closest_weight_to_standard_l734_734954

def weights : List Int := [+2, -3, +3, -4]

theorem closest_weight_to_standard (standard_weight : Int) (ws : List Int) : 
    standard_weight = 200 → ws = weights → 
    ∃ (w : Int), w = +2 ∧ (∀ v ∈ ws, Int.abs v ≥ Int.abs w) :=
by
  intro h_standard h_weights
  let w := +2
  use w
  split
  next
    rfl
  next
    intros v hv
    fin_cases hv
    any_goals { simp [weights] }
    any_goals { sorry }

end closest_weight_to_standard_l734_734954


namespace function_passes_through_point_l734_734212

theorem function_passes_through_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  ∃ x : ℝ, f x = 4 ∧ x = 1 :=
by
  -- Define the function f
  let f := λ x : ℝ, a^(x - 1) + 3
  -- Prove that f passes through the point (1,4)
  use 1
  split
  · reflexivity
  · sorry

end function_passes_through_point_l734_734212


namespace northern_village_population_is_4206_l734_734039

-- Defining the conditions
def northern_village_population := x : ℕ
def western_village_population := 7488
def southern_village_population := 6912
def total_conscripted := 300
def northern_village_conscripted := 108

-- Proving the main proposition
theorem northern_village_population_is_4206 (x : ℕ)
  (h1 : x / (x + western_village_population + southern_village_population) = northern_village_conscripted / total_conscripted) :
  x = 4206 :=
by
  sorry

end northern_village_population_is_4206_l734_734039


namespace quadratic_root_form_l734_734948

theorem quadratic_root_form {a b : ℂ} (h : 6 * a ^ 2 - 5 * a + 18 = 0 ∧ a.im = 0 ∧ b.im = 0) : 
  a + b^2 = (467:ℚ) / 144 :=
by
  sorry

end quadratic_root_form_l734_734948


namespace total_bugs_eaten_integer_l734_734371

def gecko_eats : ℝ := 18.5
def lizard_eats : ℝ := (2 / 3) * gecko_eats
def frog_eats : ℝ := 3.5 * lizard_eats
def tortoise_eats : ℝ := gecko_eats - (0.32 * gecko_eats)
def toad_eats : ℝ := (1.5 * frog_eats) - 3
def crocodile_eats : ℝ := gecko_eats + toad_eats + (0.10 * gecko_eats)
def turtle_eats : ℝ := (1 / 3) * crocodile_eats
def chameleon_eats : ℝ := 3 * (gecko_eats - tortoise_eats)

def total_bugs_eaten : ℝ := gecko_eats + lizard_eats + frog_eats + tortoise_eats + toad_eats + crocodile_eats + turtle_eats + chameleon_eats

theorem total_bugs_eaten_integer : total_bugs_eaten ≈ 276 := by
  sorry

end total_bugs_eaten_integer_l734_734371


namespace power_equality_l734_734588

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l734_734588


namespace exponent_proof_l734_734582

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l734_734582


namespace solve_quadratic_calculate_expression_l734_734997

-- Part 1: Solving the quadratic equation

theorem solve_quadratic (x : ℝ) : x^2 - 4 * x - 3 = 0 ↔ (x = 2 + real.sqrt 7) ∨ (x = 2 - real.sqrt 7) := 
sorry

-- Part 2: Calculating the given expression

theorem calculate_expression : 
  abs (-3) - 4 * real.sin (real.pi / 4) + real.sqrt 8 + (real.pi - 3)^0 = 4 := 
sorry

end solve_quadratic_calculate_expression_l734_734997


namespace arithmetic_seq_sixth_term_l734_734038

theorem arithmetic_seq_sixth_term
  (a d : ℤ)
  (h1 : a + d = 14)
  (h2 : a + 3 * d = 32) : a + 5 * d = 50 := 
by
  sorry

end arithmetic_seq_sixth_term_l734_734038


namespace value_range_of_a_l734_734951

variable (a : ℝ)
variable (suff_not_necess : ∀ x, x ∈ ({3, a} : Set ℝ) → 2 * x^2 - 5 * x - 3 ≥ 0)

theorem value_range_of_a :
  (a ≤ -1/2 ∨ a > 3) :=
sorry

end value_range_of_a_l734_734951


namespace female_officers_count_l734_734572

theorem female_officers_count 
  (percent_on_duty : ℝ) 
  (total_on_duty : ℕ)
  (half_total_on_duty : ℕ = total_on_duty / 2)
  (percent_on_duty_is_15 : percent_on_duty = 0.15)
  (female_on_duty : ℕ = half_total_on_duty) :
  ∃ (total_females : ℕ), total_females = 1000 :=
by
  have : total_on_duty / 2 = 150 := by sorry
  have : percent_on_duty * (female_on_duty / percent_on_duty) = female_on_duty := by sorry
  use ((fraction_on_duty : ℝ):total_females),
  ... := 100.pkg,
  sorry


end female_officers_count_l734_734572


namespace solve_system_of_equations_l734_734455

/-- Definition representing our system of linear equations. --/
def system_of_equations (x1 x2 : ℚ) : Prop :=
  (3 * x1 - 5 * x2 = 2) ∧ (2 * x1 + 4 * x2 = 5)

/-- The main theorem stating the solution to our system of equations. --/
theorem solve_system_of_equations : 
  ∃ x1 x2 : ℚ, system_of_equations x1 x2 ∧ x1 = 3/2 ∧ x2 = 1/2 :=
by
  sorry

end solve_system_of_equations_l734_734455


namespace smallest_term_abs_l734_734238

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem smallest_term_abs {a : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 > 0)
  (hS12 : (12 / 2) * (2 * a 1 + 11 * (a 2 - a 1)) > 0)
  (hS13 : (13 / 2) * (2 * a 1 + 12 * (a 2 - a 1)) < 0) :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 13 → n ≠ 7 → abs (a 6) > abs (a 1 + 6 * (a 2 - a 1)) :=
sorry

end smallest_term_abs_l734_734238


namespace milton_zoology_books_l734_734881

variable (Z : ℕ)
variable (total_books botany_books : ℕ)

theorem milton_zoology_books (h1 : total_books = 960)
    (h2 : botany_books = 7 * Z)
    (h3 : total_books = Z + botany_books) :
    Z = 120 := by
  sorry

end milton_zoology_books_l734_734881


namespace min_flight_routes_needed_l734_734575

theorem min_flight_routes_needed (n : ℕ) (h : n = 2021) :
  let G := { V : Type } → (V → V → Prop) → Prop
  let T_graph := ∀ {V : Type} (G : V → V → Prop), 
    (∀ (u v : V), G u v → ∃ w, G u w ∧ G v w) → 
    (∀ u v, ∃ p, List.Chain G u p ∧ List.Last p = v) →
    ∃ edges, nat.le (List.length edges) (b n)
  in T_graph G → 3030 ≤ (⌊ (3 * n - 2) / 2⌋) :=
by {
  intros,
  have min_edges := (⌊ (3 * 2021 - 2) / 2⌋ : ℕ),
  simp [min_edges],
  exact nat.le_refl 3030,
}

end min_flight_routes_needed_l734_734575


namespace minimum_omega_l734_734491

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l734_734491


namespace simplify_complex_expression_l734_734908

theorem simplify_complex_expression : 
  (7 * (4 - 2 * Complex.i) + 4 * Complex.i * (7 - 3 * Complex.i) + 2 * (5 + Complex.i)) = 50 + 16 * Complex.i := 
by 
  -- The actual proof is omitted
  sorry

end simplify_complex_expression_l734_734908


namespace monotonic_intervals_solve_inequality_max_value_on_interval_l734_734265

-- Given function definition
def f (x : ℝ) : ℝ := x * |x - 2|

-- Prove monotonic intervals
theorem monotonic_intervals :
  (∀ x, x ≤ 1 ∨ x ≥ 2 → monotonic (f x)) ∧
  (∀ x, 1 < x ∧ x < 2 → ¬monotonic (f x)) :=
sorry

-- Prove solution set for the inequality f(x) < 3
theorem solve_inequality :
  { x : ℝ | f x < 3 } = { x | x < 3 } :=
sorry

-- Prove the maximum value of f(x) on [0,a]
theorem max_value_on_interval (a: ℝ) (h : 0 < a ∧ a ≤ 2) :
  ∃ b, b = (if 0 < a ∧ a ≤ 1 then a * (2 - a) else 1) ∧
  (∀ x ∈ set.Icc 0 a, f x ≤ b) :=
sorry

end monotonic_intervals_solve_inequality_max_value_on_interval_l734_734265


namespace multiply_97_103_eq_9991_l734_734707

theorem multiply_97_103_eq_9991 : (97 * 103 = 9991) :=
by
  have h1 : 97 = 100 - 3 := rfl
  have h2 : 103 = 100 + 3 := rfl
  calc
    97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
    ... = 100^2 - 3^2 : by rw [mul_add, add_mul, sub_mul, add_sub_cancel, sub_add_cancel]
    ... = 10000 - 9 : by norm_num
    ... = 9991 : by norm_num

end multiply_97_103_eq_9991_l734_734707


namespace find_arrangements_zero_l734_734166

structure Square (color : String) where
  length : ℕ
  width : ℕ

def bobbySquares : List (Square String) :=
  [Square.mk "red" 8 5,
   Square.mk "blue" 10 7,
   Square.mk "green" 5 5,
   Square.mk "yellow" 6 4,
   Square.mk "white" 12 8]

def flagRequiredLength : ℕ := 20
def flagRequiredHeight : ℕ := 12
def colorPattern : List String := ["red", "blue", "green", "yellow", "white"]

theorem find_arrangements_zero : ∀ (arr : List (Square String)), 
  (∀ s ∈ arr, s ∈ bobbySquares) →
  (arr.length = 5) →
  (∃ perm : List (Square String), (perm = arr ∧ perm.map (λ s => s.color) = colorPattern)) →
  (flagRequiredLength = 20 ∧ flagRequiredHeight = 12) →
  False :=
by
  assume arr H1 H2 H3 H4
  sorry

end find_arrangements_zero_l734_734166


namespace volume_of_spherical_sector_l734_734867

variable {R h : ℝ}

def volume_spherical_sector (R h : ℝ) : ℝ :=
  (2 * Real.pi * R^2 * h) / 3

theorem volume_of_spherical_sector (S : ℝ) (h R : ℝ) (h1 : S = 2 * Real.pi * R * h)
  (h2 : volume_spherical_sector R h = (S * R) / 3) :
  volume_spherical_sector R h = (2 * Real.pi * R^2 * h) / 3 :=
by
  -- Proof is omitted for this statement generation
  sorry

end volume_of_spherical_sector_l734_734867


namespace choosing_officers_l734_734241

theorem choosing_officers :
  let members := {Alice, Bob, Carol, Dave}
  let qualifications (m : member) := (m = Dave) -> true 
  let eligible_for_president (m : member) := (m = Dave)
  let officers := {president, secretary, treasurer} 
  let ways_to_assign_roles := 1 * (Nat.choose 3 2) * 2 
  ways_to_assign_roles = 6 :=
by
  sorry

end choosing_officers_l734_734241


namespace function_d_has_no_boundary_point_l734_734211

def is_boundary_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∃ x₁ < x₀, f x₁ = 0) ∧ (∃ x₂ > x₀, f x₂ = 0)

def f_a (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x - 2
def f_b (x : ℝ) : ℝ := abs (x^2 - 3)
def f_c (x : ℝ) : ℝ := 1 - abs (x - 2)
def f_d (x : ℝ) : ℝ := x^3 + x

theorem function_d_has_no_boundary_point :
  ¬ ∃ x₀ : ℝ, is_boundary_point f_d x₀ :=
sorry

end function_d_has_no_boundary_point_l734_734211


namespace second_alloy_amount_l734_734361

theorem second_alloy_amount (x : ℝ) :
  (0.12 * 15 + 0.08 * x = 0.092 * (15 + x)) → x = 35 :=
by
  sorry

end second_alloy_amount_l734_734361


namespace midpoint_lies_on_circumcircle_of_ADZ_l734_734872

variable {α β γ : Type}
variable [EuclideanGeometry α]

-- Definitions of points and segments
variables (A B C D Z M : α)
variable {triangle : Triangle α A B C}
variables {circumcircle : Circle α D (circumcenter α A B C)}
variable {AB_midpoint : LineSegment α A B}

def midpoint (A B : α) : α := (A + B) / 2

axiom AB_less_AC : length (LineSegment AB) < length (LineSegment AC)
axiom D_on_internal_bisector : lies_on D (internal_bisector (Triangle α A B C) ∪ circumcircle)
axiom Z_on_perpendicular_bisector : 
  lies_on Z (perpendicular_bisector (LineSegment α A C)) ∧
  lies_on Z (external_bisector (Triangle α A B C))

theorem midpoint_lies_on_circumcircle_of_ADZ :
  lies_on (midpoint A B) (circumcircle_of_triangle ⟨A, D, Z⟩) :=
by
  sorry

end midpoint_lies_on_circumcircle_of_ADZ_l734_734872


namespace non_empty_proper_subsets_range_of_m_l734_734402

-- Define the set A
def A (x : ℝ) : Prop := x ∈ (Set.Icc (-2) 5)

-- Define the set B
def B (m x : ℝ) : Prop := x^2 - 3 * m * x + 2 * m^2 - m - 1 < 0

-- Question 1: For x ∈ ℕ, find the number of non-empty proper subsets of A
theorem non_empty_proper_subsets (A_nat : Set ℕ) (A_nat_def : A_nat = {n | A n}) : 
  (A_nat.card.to_nat - 2 = 62) :=
sorry

-- Question 2: If A ∩ B = B, find the range of real number m
theorem range_of_m (m : ℝ) (h : ∀ x, B m x → A x) : 
  (-1 ≤ m ∧ m ≤ 2) ∨ (m = -2) :=
sorry

end non_empty_proper_subsets_range_of_m_l734_734402


namespace domain_of_function_l734_734732

theorem domain_of_function :
  { x : ℝ | 0 ≤ 2 * x - 10 ∧ 2 * x - 10 ≠ 0 } = { x : ℝ | x > 5 } :=
by
  sorry

end domain_of_function_l734_734732


namespace vasya_fraction_is_0_4_l734_734652

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l734_734652


namespace choose_3_of_9_colors_l734_734318

-- Define the combination function
noncomputable def combination (n k : ℕ) := n.choose k

-- Noncomputable because factorial and combination require division.
noncomputable def combination_9_3 := combination 9 3

-- State the theorem we are proving
theorem choose_3_of_9_colors : combination_9_3 = 84 :=
by
  -- Proof skipped
  sorry

end choose_3_of_9_colors_l734_734318


namespace chunky_count_between_13_and_113_l734_734161

/-- An integer is called chunky if it consists only of non-zero digits and each digit
    is a divisor of the entire number. -/
def is_chunky (n : ℕ) : Prop :=
  ∀ d ∈ nat.digits 10 n, d ≠ 0 ∧ n % d = 0

/-- The proof problem: Prove that the number of chunky integers between 13 and 113 inclusive is 13. -/
theorem chunky_count_between_13_and_113 : 
  finset.filter is_chunky (finset.range (113 + 1)) \ {0..12}.card = 13 := 
sorry

end chunky_count_between_13_and_113_l734_734161


namespace distance_circumcenter_centroid_inequality_l734_734874

variable {R r d : ℝ}

theorem distance_circumcenter_centroid_inequality 
  (h1 : d = distance_circumcenter_to_centroid)
  (h2 : R = circumradius)
  (h3 : r = inradius) : d^2 ≤ R * (R - 2 * r) := 
sorry

end distance_circumcenter_centroid_inequality_l734_734874


namespace crossing_time_opposite_directions_l734_734126

theorem crossing_time_opposite_directions
  (length_train : ℝ)
  (speed_train1_kmh : ℝ)
  (speed_train2_kmh : ℝ)
  (time_same_direction_sec : ℝ)
  (h1 : speed_train1_kmh = 60)
  (h2 : speed_train2_kmh = 40)
  (h3 : time_same_direction_sec = 36)
  (combined_length_m : length_train * 2 = (speed_train1_kmh - speed_train2_kmh) * 5/18 * time_same_direction_sec)
  :
  let speed_opposite_mps := (speed_train1_kmh + speed_train2_kmh) * 5 / 18 in
  let time_opposite_direction_sec := (length_train * 2) / speed_opposite_mps in
  time_opposite_direction_sec = 200.16 / 27.78 := 
sorry

end crossing_time_opposite_directions_l734_734126


namespace domain_of_negative_scaled_function_l734_734300

theorem domain_of_negative_scaled_function {f : ℝ → ℝ} (h : ∀ x, 0 < x ∧ x < 2 → f x) :
  ∀ x, -1 < x ∧ x < 0 → f (-2 * x) :=
by
  -- Here goes the actual proof which we are not providing
  sorry

end domain_of_negative_scaled_function_l734_734300


namespace circumscribe_square_l734_734441

def convex_shape {ℝ : Type*} [real ℝ] (shape : set (ℝ × ℝ)) :=
  -- Definition of a bounded convex shape in the plane
  bounded shape ∧ convex shape

noncomputable def d {α : Type*} (shape : set (ℝ × ℝ)) (theta : ℝ) : ℝ :=
  -- Distance between two supporting lines of the shape that are parallel 
  -- to the vector obtained by rotating the initial vector by theta
  sorry

noncomputable def f {ℝ : Type*} [real ℝ] (shape : set (ℝ × ℝ)) (alpha : ℝ) : ℝ :=
  d shape alpha - d shape (alpha + (π / 2))

theorem circumscribe_square (shape : set (ℝ × ℝ)) :
  convex_shape shape → ∃ alpha : ℝ, f shape alpha = 0 :=
by
  intro h
  sorry

end circumscribe_square_l734_734441


namespace parallel_lines_l734_734437

theorem parallel_lines
  {A B C D Q R : Type}
  (circle : set (point))
  (on_circle : ∀ p, p ∈ {A, B, C, D} → p ∈ circle)
  (AB_eq_BD : AB = BD)
  (tangent_Q : tangent_at circle A intersects line BC at Q)
  (intersection_R : intersection_of_lines (line AB) (line CD) = R) :
  parallel (line QR) (line AD) :=
sorry

end parallel_lines_l734_734437


namespace vasya_drives_fraction_l734_734661

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l734_734661


namespace range_of_m_for_distance_l734_734374

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (|x1 - x2|) + 2 * (|y1 - y2|)

theorem range_of_m_for_distance (m : ℝ) : 
  distance 2 1 (-1) m ≤ 5 ↔ 0 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_for_distance_l734_734374


namespace wheel_turns_time_l734_734149

theorem wheel_turns_time (h : ∀ (n : ℕ), wheel_turns n seconds = 6 → time_for n turns = 30) : let turns_per_hour := 1440 / 2,
    seconds_per_hour := 3600,
    turns_per_hour = 720,
    time_per_turn := seconds_per_hour / turns_per_hour,
    time_per_turn = 5,
    turns_needed := 6,
    total_time := turns_needed * time_per_turn
  in total_time = 30 := by
  intros
  sorry

end wheel_turns_time_l734_734149


namespace tangent_line_tangent_points_extreme_values_l734_734797

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x - 3

theorem tangent_line_tangent_points :
  (∃ x₀ : ℝ, f'(x₀) = -9 ∧ (∀ x : ℝ, (x₀ = 0 → f(x) = -3 ∧ b = -3) ∨ (x₀ = 2 → f(x) = -25 ∧ b = -7))) := by
sorrry

theorem extreme_values :
  (∀ x : ℝ, (x = -1 → f(x) = 2) ∨ (x = 3 → f(x) = -30)) := by
sorry

end tangent_line_tangent_points_extreme_values_l734_734797


namespace evaluate_powers_l734_734195

theorem evaluate_powers : (81^(1/2:ℝ) * 64^(-1/3:ℝ) * 49^(1/4:ℝ) = 9 * (1/4) * Real.sqrt 7) :=
by
  sorry

end evaluate_powers_l734_734195


namespace range_of_a_l734_734306

noncomputable def equation_has_two_roots (a m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ + a * (2 * x₁ + 2 * m - 4 * Real.exp 1 * x₁) * (Real.log (x₁ + m) - Real.log x₁) = 0 ∧ 
    x₂ + a * (2 * x₂ + 2 * m - 4 * Real.exp 1 * x₂) * (Real.log (x₂ + m) - Real.log x₂) = 0

theorem range_of_a (m : ℝ) (hm : 0 < m) : 
  (∃ a, equation_has_two_roots a m) ↔ (a < 0 ∨ a > 1 / (2 * Real.exp 1)) := 
sorry

end range_of_a_l734_734306


namespace sum_of_coefficients_weighted_sum_of_coefficients_l734_734756

theorem sum_of_coefficients (a : ℕ → ℤ) :
  (∀ x : ℤ, (1 - 2 * x) ^ 2023 = ∑ i in finset.range 2024, (a i) * x ^ i) →
  (∑ i in finset.range 2024, a i) = -1 :=
by
  sorry

theorem weighted_sum_of_coefficients (a : ℕ → ℤ) :
  (∀ x : ℤ, (1 - 2 * x) ^ 2023 = ∑ i in finset.range 2024, (a i) * x ^ i) →
  (∑ i in finset.range 2024, a i * (2 ^ (i - 1))) = -1 :=
by
  sorry

end sum_of_coefficients_weighted_sum_of_coefficients_l734_734756


namespace intersection_A_B_l734_734296

def A (x : ℝ) : Prop := (2 * x - 1 > 0)
def B (x : ℝ) : Prop := (x * (x - 2) < 0)

theorem intersection_A_B :
  {x : ℝ | A x ∧ B x} = {x : ℝ | 1 / 2 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l734_734296


namespace train_initial_speed_l734_734048

variables (x t : ℝ)

theorem train_initial_speed (h1 : x * t + (x * t + 23) = 103)
                           (h2 : (x + 4) * (t + 1/4) = x * t + 23) :
                           x = 80 :=
begin
  sorry
end

end train_initial_speed_l734_734048


namespace max_area_triangle_ABC_l734_734054

noncomputable def max_area_triangle (AB AC BC : ℝ) : ℝ :=
  if AC = sqrt 3 * BC ∧ AB = 2 then sqrt 3 else 0

theorem max_area_triangle_ABC : 
  ∀ (AB AC BC : ℝ), AB = 2 → AC = sqrt 3 * BC → 
  max_area_triangle AB AC BC = sqrt 3 :=
by
  intros AB AC BC hAB hAC
  unfold max_area_triangle
  rw [hAC, hAB]
  simp
  sorry

end max_area_triangle_ABC_l734_734054


namespace minimum_omega_l734_734489

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l734_734489


namespace cubic_inequality_solution_l734_734216

theorem cubic_inequality_solution :
  ∀ x : ℝ, (x + 1) * (x + 2)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
by 
  sorry

end cubic_inequality_solution_l734_734216


namespace lebesgue_stieltjes_measure_of_countable_set_is_zero_l734_734423

noncomputable def Lebesgue_Stieltjes_measure {α : Type*} (F : α → ℝ)
  (hF_cont : ∀ x, continuous_at F x) : (set α → ℝ) :=
  sorry   -- Definition of Lebesgue-Stieltjes measure, omitting for brevity

variables {α : Type*}
variables (ν : set α → ℝ) (A : set α)

-- Assume ν is a Lebesgue-Stieltjes measure associated with a continuous function on ℝ
axiom h_ν : ∃ F : α → ℝ, (∀ x, continuous_at F x) ∧ ν = Lebesgue_Stieltjes_measure F _

-- Theorem statement
theorem lebesgue_stieltjes_measure_of_countable_set_is_zero
  (hA : set.countable A) : ν A = 0 :=
sorry

end lebesgue_stieltjes_measure_of_countable_set_is_zero_l734_734423


namespace count_lineup_excluding_youngest_l734_734347

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l734_734347


namespace parabola_vertex_l734_734947

theorem parabola_vertex (c d : ℝ) (h : ∀ x : ℝ, - x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  (∃ v : ℝ × ℝ, v = (5, 1)) :=
sorry

end parabola_vertex_l734_734947


namespace minimum_omega_l734_734494

theorem minimum_omega {ω : ℝ} (hω : ω > 0)
    (symmetry : ∃ k : ℤ, ∀ x : ℝ, 
      (sin (ω * x + ω * π / 2 + π / 3) = sin (-ω * x + ω * π / 2 + π / 3))) 
    : ω = 1 / 3 :=
by
  sorry

end minimum_omega_l734_734494


namespace ellipse_perimeter_correct_l734_734923

noncomputable def ellipse_perimeter : Prop :=
  let a : ℝ := 3
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 - b^2)
  ∀ (P : ℝ × ℝ),
    (P.1^2 / 9 + P.2^2 / 5 = 1) → 
      let F1 : ℝ × ℝ := (c, 0)
      let F2 : ℝ × ℝ := (-c, 0)
      abs (dist P F1 + dist P F2 + dist F1 F2) = 10

theorem ellipse_perimeter_correct : ellipse_perimeter :=
by
  -- Proof outline
  sorry

end ellipse_perimeter_correct_l734_734923


namespace initial_fuel_is_38_l734_734191

-- Definitions of given conditions
def full_capacity : ℕ := 150
def total_money_given : ℕ := 350
def change_received : ℕ := 14
def cost_per_liter : ℕ := 3

-- The goal is to prove that initial_fuel = 38
theorem initial_fuel_is_38 
  (full_capacity : ℕ)
  (spent : ℕ)
  (cost_per_liter : ℕ)
  : ∃ (initial_fuel : ℕ), initial_fuel = full_capacity - spent / cost_per_liter ∧ spent = total_money_given - change_received :=
by
  let fuel_bought := (total_money_given - change_received) / cost_per_liter
  let initial_fuel := full_capacity - fuel_bought
  use initial_fuel
  split
  · exact rfl
  · sorry

end initial_fuel_is_38_l734_734191


namespace solution_is_three_l734_734453

def equation (x : ℝ) : Prop := 
  Real.sqrt (4 - 3 * Real.sqrt (10 - 3 * x)) = x - 2

theorem solution_is_three : equation 3 :=
by sorry

end solution_is_three_l734_734453


namespace vasya_fraction_l734_734670

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l734_734670


namespace pump_fills_tank_without_leak_l734_734640

theorem pump_fills_tank_without_leak (T : ℝ) (h1 : 1 / 12 = 1 / T - 1 / 12) : T = 6 :=
sorry

end pump_fills_tank_without_leak_l734_734640


namespace b_gets_more_than_c_l734_734568

-- Define A, B, and C as real numbers
variables (A B C : ℝ)

theorem b_gets_more_than_c 
  (h1 : A = 3 * B)
  (h2 : B = C + 25)
  (h3 : A + B + C = 645)
  (h4 : B = 134) : 
  B - C = 25 :=
by
  -- Using the conditions from the problem
  sorry

end b_gets_more_than_c_l734_734568


namespace coordinates_after_100_steps_l734_734362

/-- In the Cartesian coordinate system, Xiaoming plays a chess game where the piece starts at the origin.
The movement rules are as follows: on the 1st step, move 1 unit to the right; on the 2nd step, move 2 units
to the right; on the 3rd step, move 1 unit up; for the nth step: when n is divisible by 3, move 1 unit up;
when the remainder of n divided by 3 is 1, move 1 unit to the right; when the remainder of n divided by 3 is
2, move 2 units to the right. After the 100th step, the coordinates of the piece are (100, 33). -/

def final_coordinates_after_100_steps : ℕ × ℕ := (100, 33)

theorem coordinates_after_100_steps :
  (λ (final_coordinates_after_100_steps)  
    .=   
    let steps := 100,
        cycles := steps / 3,
        remainder := steps % 3,
        horizontal := cycles * 3 + (if remainder = 1 then 1 else if remainder = 2 then 2 else 0),
        vertical := cycles * 1 + (if remainder = 0 then 1 else 0)
    in (horizontal, vertical)) final_coordinates == (100, 33) :=
sorry

end coordinates_after_100_steps_l734_734362


namespace line_up_ways_l734_734360

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l734_734360


namespace equal_remainders_prime_condition_l734_734233

theorem equal_remainders_prime_condition {p x : ℕ} (hp : Nat.Prime p) (hx_pos : 0 < x) 
  (h1 : ∃ r, x % p = r ∧ p^2 % x = r) :
  ∃ r, r = 0 ∨ r = 1 where
    r = x % p :=
by
  sorry

end equal_remainders_prime_condition_l734_734233


namespace joyce_new_property_is_10_times_larger_l734_734855

theorem joyce_new_property_is_10_times_larger :
  let previous_property := 2
  let suitable_acres := 19
  let pond := 1
  let new_property := suitable_acres + pond
  new_property / previous_property = 10 := by {
    let previous_property := 2
    let suitable_acres := 19
    let pond := 1
    let new_property := suitable_acres + pond
    sorry
  }

end joyce_new_property_is_10_times_larger_l734_734855


namespace exists_b_two_solutions_l734_734730

theorem exists_b_two_solutions (a : ℝ) :
  (∃ b : ℝ, ∀ (x y : ℝ), arcsin ((a + y) / 2) = arcsin ((x + 3) / 3) ∧ x^2 + y^2 + 6 * x + 6 * y = b) ↔
  a ∈ Ioo (-7 / 2) (19 / 2) :=
by
  sorry

end exists_b_two_solutions_l734_734730


namespace count_restricted_arrangements_l734_734343

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l734_734343


namespace ordered_pairs_satisfying_inequalities_l734_734281

theorem ordered_pairs_satisfying_inequalities :
  {p : ℤ × ℤ | p.1^2 + p.2^2 < 16 ∧ p.1^2 + p.2^2 < 8 * p.1 ∧ p.1^2 + p.2^2 < 8 * p.2}.to_finset.card = 6 :=
by
  sorry

end ordered_pairs_satisfying_inequalities_l734_734281


namespace Vasya_distance_fraction_l734_734657

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l734_734657


namespace mom_needs_12_packages_l734_734976

def packages_needed (desired_shirts : ℝ) (shirts_per_package : ℝ) : ℕ :=
  if desired_shirts ≤ 0 ∨ shirts_per_package ≤ 0 then 0
  else ⌈desired_shirts / shirts_per_package⌉.to_nat

theorem mom_needs_12_packages :
  packages_needed 71 6 = 12 :=
by
  unfold packages_needed
  norm_num
  sorry

end mom_needs_12_packages_l734_734976


namespace sum_of_integers_l734_734047

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 6) (h2 : x * y = 112) (h3 : x > y) : x + y = 22 :=
sorry

end sum_of_integers_l734_734047


namespace area_of_hexagon_is_half_area_of_triangle_l734_734220

-- Given definitions and conditions
variables {A B C A1 B1 C1 : Type} [plane_geometry A B C]

-- Define midpoints and perpendiculars (conditions)
def is_midpoint (m p q : Type) [plane_geometry m p q] : Prop := sorry
def is_perpendicular (a b c d : Type) [plane_geometry a b c d] : Prop := sorry
def triangle (a b c : Type) [plane_geometry a b c] : Prop := sorry

-- Given triangle ABC is acute-angled
axiom acute_triangle_ABC : triangle A B C ∧ is_acute A B C

-- Midpoints and perpendiculars
axiom mid_A1_on_BC : is_midpoint A1 B C
axiom mid_B1_on_CA : is_midpoint B1 C A
axiom mid_C1_on_AB : is_midpoint C1 A B
axiom perp_from_A1_to_CA_and_AB : is_perpendicular A1 C A ∧ is_perpendicular A1 B C
axiom perp_from_B1_to_AB_and_CA : is_perpendicular B1 A B ∧ is_perpendicular B1 C A
axiom perp_from_C1_to_BC_and_AB : is_perpendicular C1 B C ∧ is_perpendicular C1 A B

-- Prove that the area of the hexagon is half the area of the triangle
theorem area_of_hexagon_is_half_area_of_triangle (S_ABC S_hexagon : ℝ) :
  acute_triangle_ABC →
  mid_A1_on_BC →
  mid_B1_on_CA →
  mid_C1_on_AB →
  perp_from_A1_to_CA_and_AB →
  perp_from_B1_to_AB_and_CA →
  perp_from_C1_to_BC_and_AB →
  S_hexagon = (1 / 2) * S_ABC :=
begin
  sorry
end

end area_of_hexagon_is_half_area_of_triangle_l734_734220


namespace minimum_omega_for_symmetric_curve_l734_734470

theorem minimum_omega_for_symmetric_curve (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 2) + π / 3) = sin (-ω * (x + π / 2) + π / 3)) ↔ ω = 1 / 3 :=
by
  sorry

end minimum_omega_for_symmetric_curve_l734_734470


namespace decagon_vertex_sum_bound_l734_734446

open Function

-- Define the cyclic structure of the decagon indices
def decagon_index (n : ℕ) : Fin 10 := ⟨n % 10, Nat.mod_lt n dec_trivial⟩

-- Statement of the problem as a Lean theorem
theorem decagon_vertex_sum_bound (a : Fin 10 → ℕ) (h_labels : ∀ i, ∃ j, a j = i + 1) :
  ∃ i, a i + a (decagon_index (i + 1)) + a (decagon_index (i + 9)) ≥ 17 :=
by
  let sums := λ i, a i + a (decagon_index (i + 1)) + a (decagon_index (i + 9))
  have sum_all_sums : ∑ i, sums i = 165 := 
    sorry -- This would be derived as in the steps of the solution based on the given conditions.
  by_contradiction
  assume h : ∀ i, sums i < 17 
  have : ∑ i, sums i < 17 * 10 := 
    sorry -- Use the contradiction hypothesis to argue this
  linarith

end decagon_vertex_sum_bound_l734_734446


namespace determine_range_of_a_l734_734303

theorem determine_range_of_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ a * x^2 - x + 2 = 0 ∧ a * y^2 - y + 2 = 0) : 
  a < 1 / 8 ∧ a ≠ 0 :=
sorry

end determine_range_of_a_l734_734303


namespace constant_term_product_l734_734084

-- Define the polynomials
noncomputable def poly1 : Polynomial ℝ := Polynomial.mk [6, 0, 1, 1]
noncomputable def poly2 : Polynomial ℝ := Polynomial.mk [7, 0, 3, 0, 2]

-- Prove the constant term of the product of poly1 and poly2 is 42
theorem constant_term_product : (poly1 * poly2).coeff 0 = 42 :=
by sorry

end constant_term_product_l734_734084


namespace arbelos_collinearity_l734_734602

noncomputable def point_on_line (P Q R : Point) : Prop :=
  ∃ l : Line, P ∈ l ∧ Q ∈ l ∧ R ∈ l

theorem arbelos_collinearity 
  (A B C D M N : Point)
  (h_circle_DB : Diameter (Circle.mk D B))
  (h_intersects_M : M ∈ (small_arbelos_circles h_circle_DB))
  (h_intersects_N : N ∈ (small_arbelos_circles h_circle_DB))
  (h_diam_opposite : diametrically_opposite D B) :
  point_on_line A M B ∧ point_on_line B N C :=
by
  -- missing proof
  sorry

end arbelos_collinearity_l734_734602


namespace find_x_value_l734_734713

noncomputable theory

def diamondsuit (a b : ℝ) : ℝ := a / b

axiom diamondsuit_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamondsuit a (diamondsuit b c) = (diamondsuit a b) * c

axiom diamondsuit_self (a : ℝ) (ha : a ≠ 0) : diamondsuit a a = 1

theorem find_x_value : ∃ x : ℝ, (50 ≠ 0) → (5 ≠ 0) → (x ≠ 0) → diamondsuit 50 (diamondsuit 5 x) = 200 ↔ x = 20 :=
by
  use 20
  intro h50 h5 h20
  sorry

end find_x_value_l734_734713


namespace largest_number_l734_734821

-- Definitions from the problem conditions
variables a b : ℝ
variables h1 : 0 < a
variables h2 : a < b
variables h3 : a + b = 1

-- The statement that we need to prove
theorem largest_number (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > max (max (1 / 2) (2 * a * b)) (a^2 + b^2) :=
by
  sorry

end largest_number_l734_734821


namespace repeating_decimal_sum_l734_734184

theorem repeating_decimal_sum (x : ℚ) (hx : x = 47 / 99) : x.num.natAbs + x.denom = 146 := 
by
  rw hx
  dsimp
  have hnum : (47 / 99).num.natAbs = 47 := rfl
  have hdenom : (47 / 99).denom = 99 := rfl
  rw [hnum, hdenom]
  norm_num

end repeating_decimal_sum_l734_734184


namespace evaluate_expr_correct_l734_734107

-- Define the expression as a function
def expr (a b c : ℕ) : ℕ := a - b * c

-- The main theorem to be proven
theorem evaluate_expr_correct : expr 65 13 2 = 39 := 
by
  -- Arithmetic operations follow the usual precedence
  -- Multiplication 13 * 2 is performed first to get 26
  -- Then subtraction 65 - 26 is performed to get 39
  calc
  expr 65 13 2 = 65 - 13 * 2 : rfl
  ... = 65 - 26 : by simp
  ... = 39 : by simp

end evaluate_expr_correct_l734_734107


namespace f_of_5_eq_2_l734_734753

def f : ℤ → ℤ :=
λ x, if x ≥ 6 then x - 5 else f (x + 2)

theorem f_of_5_eq_2 : f 5 = 2 := 
by sorry

end f_of_5_eq_2_l734_734753


namespace committee_ways_l734_734137

theorem committee_ways (total_members leadership_experience : ℕ) (committee_size : ℕ) (H1 : total_members = 30) (H2 : leadership_experience = 8) (H3 : committee_size = 5) :
  ∃ (n : ℕ), n = 116172 ∧ 
  (∑ i in finset.range (committee_size + 1), if i = 0 then 0 else (nat.choose leadership_experience i) * (nat.choose (total_members - leadership_experience) (committee_size - i)) = n) := 
by {
  sorry
}

end committee_ways_l734_734137


namespace quadrilateral_has_circumscribed_circle_l734_734468

-- Defining the given lines forming the quadrilateral
def l1 (x y : ℝ) : Prop := x + 3 * y - 15 = 0
def l2 (x y k : ℝ) : Prop := k * x - y - 6 = 0
def l3 (x y : ℝ) : Prop := x + 5 * y = 0
def l4 (y : ℝ) : Prop := y = 0

-- The value of k such that the quadrilateral has a circumscribed circle
def k_val : ℝ := -8 / 15

-- The equation of the circumscribed circle
def circumscribed_circle (x y : ℝ) : Prop := x^2 + y^2 - 15 * x - 159 * y = 0

-- Prove that the quadrilateral with the given lines has a circumscribed circle
theorem quadrilateral_has_circumscribed_circle :
  ∀ (x y : ℝ), 
    (l1 x y ∧ l2 x y k_val ∧ l3 x y ∧ l4 y) → 
    circumscribed_circle x y :=
by
  sorry

end quadrilateral_has_circumscribed_circle_l734_734468


namespace vitya_masha_probability_l734_734546

theorem vitya_masha_probability :
  let total_days := 30
  let total_pairs := total_days * total_days
  let favourable_pairs := (∑ k in Finset.range total_days, k)
  total_pairs = 900 ∧ favourable_pairs = 435 ∧
  probability (Vitya at_least_one_day_older_than_Masha) = favourable_pairs / total_pairs :=
by {
  let total_days := 30,
  let total_pairs := total_days * total_days,
  let favourable_pairs := (∑ k in Finset.range total_days, k),
  
  have h1: total_pairs = 900 := by norm_num,
  have h2: favourable_pairs = 435 := by norm_num,

  have probability := 435.0 / 900.0,
  norm_num at top,
  simp,
}

end vitya_masha_probability_l734_734546


namespace volume_of_milk_l734_734138

def radius (diameter : ℝ) : ℝ := diameter / 2
def height (diameter : ℝ) : ℝ := (3 / 4) * diameter
def sector_area (radius : ℝ) (angle_deg : ℝ) : ℝ := (angle_deg / 360) * Math.pi * radius^2
def triangle_area (radius : ℝ) : ℝ := (sqrt 3 / 4) * radius^2

theorem volume_of_milk (d l : ℝ) (h : ℝ)
  (hd : d = 180)
  (hl : l = 400)
  (hh : h = (3 / 4) * d) :
  let
    r := radius d,
    sa := sector_area r 240,
    ta := triangle_area r,
    filled_area := sa + ta,
    volume := filled_area * l
  in
  volume = 82 * 1000 :=
by
  sorry

end volume_of_milk_l734_734138


namespace annuity_payment_l734_734081

variable (P : ℝ) (A : ℝ) (i : ℝ) (n1 n2 : ℕ)

-- Condition: Principal amount
axiom principal_amount : P = 24000

-- Condition: Annual installment for the first 5 years
axiom annual_installment : A = 1500 

-- Condition: Annual interest rate
axiom interest_rate : i = 0.045 

-- Condition: Years before equal annual installments
axiom years_before_installment : n1 = 5 

-- Condition: Years for repayment after the first 5 years
axiom repayment_years : n2 = 7 

-- Remaining debt after n1 years
noncomputable def remaining_debt_after_n1 : ℝ :=
  P * (1 + i) ^ n1 - A * ((1 + i) ^ n1 - 1) / i

-- Annual payment for n2 years to repay the remaining debt
noncomputable def annual_payment (D : ℝ) : ℝ :=
  D * (1 + i) ^ n2 / (((1 + i) ^ n2 - 1) / i)

axiom remaining_debt_amount : remaining_debt_after_n1 P A i n1 = 21698.685 

theorem annuity_payment : annual_payment (remaining_debt_after_n1 P A i n1) = 3582 := by
  sorry

end annuity_payment_l734_734081


namespace sum_series_eq_l734_734408

theorem sum_series_eq (a b : ℕ) (h_rel_prime : Nat.coprime a b) :
  (a : ℚ) / (b : ℚ) = (∑' n : ℕ, (2 * n + 1 : ℚ) / (2^(2*(n+1)) : ℚ)) + (∑' n : ℕ, (2 * (n + 1) : ℚ) / (3^(2*(n+1)+1) : ℚ)) →
  a + b = 14 :=
by
  sorry

end sum_series_eq_l734_734408


namespace conditional_probabilities_l734_734530

def PA : ℝ := 0.20
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

theorem conditional_probabilities :
  PAB / PB = 2 / 3 ∧ PAB / PA = 3 / 5 := by
  sorry

end conditional_probabilities_l734_734530


namespace part3_conclusion_l734_734363

-- Definitions and conditions for the problem
def quadratic_function (a x : ℝ) : ℝ := (x - a)^2 + a - 1

-- Part 1: Given condition that (1, 2) lies on the graph of the quadratic function
def part1_condition (a : ℝ) := (quadratic_function a 1) = 2

-- Part 2: Given condition that the function has a minimum value of 2 for 1 ≤ x ≤ 4
def part2_condition (a : ℝ) := ∀ x, 1 ≤ x ∧ x ≤ 4 → quadratic_function a x ≥ 2

-- Part 3: Given condition (m, n) on the graph where m > 0 and m > 2a
def part3_condition (a m n : ℝ) := m > 0 ∧ m > 2 * a ∧ quadratic_function a m = n

-- Conclusion for Part 3: Prove that n > -5/4
theorem part3_conclusion (a m n : ℝ) (h : part3_condition a m n) : n > -5/4 := 
sorry  -- Proof required here

end part3_conclusion_l734_734363


namespace jessica_total_payment_l734_734386

-- Definitions based on the conditions
def basic_cable_cost : Nat := 15
def movie_channels_cost : Nat := 12
def sports_channels_cost : Nat := movie_channels_cost - 3

-- Definition of the total monthly payment given Jessica adds both movie and sports channels
def total_monthly_payment : Nat :=
  basic_cable_cost + (movie_channels_cost + sports_channels_cost)

-- The proof statement
theorem jessica_total_payment : total_monthly_payment = 36 :=
by
  -- skip the proof
  sorry

end jessica_total_payment_l734_734386


namespace minimum_value_of_f_l734_734793

theorem minimum_value_of_f :
  ∀ (f : ℝ → ℝ) (ϕ : ℝ),
    (∀ x, f x = cos (2 * x - ϕ) - sqrt 3 * sin (2 * x - ϕ)) →
    abs ϕ < π / 2 →
    (∀ x, f x = 2 * cos (2 * (x - π / 12) - ϕ + π / 3)) →
    Symmetric about y-axis (λ x, f x) →
    ∀ x ∈ Icc (-π / 2) 0, f x ≥ -sqrt 3 := sorry

end minimum_value_of_f_l734_734793


namespace floor_sequence_inequality_l734_734876

theorem floor_sequence_inequality (x : ℝ) (n : ℕ) : 
  (⌊n * x⌋ : ℝ) ≥ (∑ i in finset.range n, (⌊(i + 1) * x⌋ : ℝ) / (i + 1)) :=
sorry

end floor_sequence_inequality_l734_734876


namespace train_crossing_time_l734_734284

def train_length : ℝ := 100 -- meters
def bridge_length : ℝ := 170 -- meters
def speed_kmph : ℝ := 36 -- kmph
def conversion_factor : ℝ := 1000 / 3600 -- conversion from kmph to m/s
def speed_mps : ℝ := speed_kmph * conversion_factor -- speed in m/s

def total_distance : ℝ := train_length + bridge_length -- total distance

def expected_time : ℝ := 27 -- expected time in seconds

theorem train_crossing_time :
  total_distance / speed_mps = expected_time :=
by
  sorry

end train_crossing_time_l734_734284


namespace temperature_median_mode_l734_734891

noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (f - 32) * 5 / 9

def temp_fahrenheit : List ℝ := [-36.5, 13.75, -15.25, -10.5, -15.25]
def temp_celsius : List ℝ := temp_fahrenheit.map fahrenheit_to_celsius

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 1 then sorted.sorted.nth (sorted.length / 2)
  else (sorted.sorted.nth (sorted.length / 2 - 1) + sorted.sorted.nth (sorted.length / 2)) / 2

def mode (l : List ℝ) : ℝ :=
  let frequencies := l.foldl (λ counts temp => counts.insert temp (counts.find temp |>.getD 0 + 1)) (RBMap.ofList [])
  frequencies.toList.maxBy (·.snd) |>.fst

theorem temperature_median_mode :
  median temp_celsius = -26.25 ∧ mode temp_celsius = -26.25 :=
by
  sorry

end temperature_median_mode_l734_734891


namespace part1_part2_l734_734266

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 5 * Real.log x + a * x^2 - 6 * x
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := 5 / x + 2 * a * x - 6

theorem part1 (a : ℝ) (h_tangent : f_prime 1 a = 0) : a = 1 / 2 :=
by {
  sorry
}

theorem part2 (a : ℝ) (h_a : a = 1/2) :
  (∀ x, 0 < x → x < 1 → f_prime x a > 0) ∧
  (∀ x, 5 < x → f_prime x a > 0) ∧
  (∀ x, 1 < x → x < 5 → f_prime x a < 0) :=
by {
  sorry
}

end part1_part2_l734_734266


namespace distance_point_to_plane_l734_734731

variables {A B C D x₀ y₀ z₀ : ℝ}

theorem distance_point_to_plane :
  ∀ (A B C D x₀ y₀ z₀ : ℝ), 
  let ρ := (| A * x₀ + B * y₀ + C * z₀ + D |) / (sqrt (A^2 + B^2 + C^2)) 
  in true :=
by intros; sorry

end distance_point_to_plane_l734_734731


namespace value_does_not_appear_l734_734209

theorem value_does_not_appear : 
  let f : ℕ → ℕ := fun x => 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1
  let x := 2
  let values := [14, 31, 64, 129, 259]
  127 ∉ values :=
by
  sorry

end value_does_not_appear_l734_734209


namespace limit_sin_exp_l734_734690

open Real

theorem limit_sin_exp (h1 : ∀ x, sin(5 * (x + π)) = -sin(5 * x))
  (h2 : Tendsto (λ x, (exp (3 * x) - 1) / (3 * x)) (𝓝 0) (𝓝 1))
  (h3 : Tendsto (λ x, sin(5 * x) / (5 * x)) (𝓝 0) (𝓝 1)) :
  Tendsto (λ x, (sin (5 * (x + π))) / (exp (3 * x) - 1)) (𝓝 0) (𝓝 (-5 / 3)) := by
  sorry

end limit_sin_exp_l734_734690


namespace preimage_of_4_3_is_2_1_l734_734801

theorem preimage_of_4_3_is_2_1 :
  ∃ (a b : ℝ), (a + 2 * b = 4) ∧ (2 * a - b = 3) ∧ (a = 2) ∧ (b = 1) :=
by
  exists 2
  exists 1
  constructor
  { sorry }
  constructor
  { sorry }
  constructor
  { sorry }
  { sorry }


end preimage_of_4_3_is_2_1_l734_734801


namespace find_B_values_l734_734970

theorem find_B_values (A B : ℤ) (h1 : 800 < A) (h2 : A < 1300) (h3 : B > 1) (h4 : A = B ^ 4) : B = 5 ∨ B = 6 := 
sorry

end find_B_values_l734_734970


namespace modulus_of_complex_number_l734_734298

open Complex

-- Let z be a complex number
variable (z : ℂ)

-- Condition: z satisfies 3z - conjugate(z) = 2 + 4i
def condition : Prop := 3 * z - conj(z) = (2 + 4 * Complex.i)

-- Statement: If z satisfies the condition, then |z| = sqrt(2)
theorem modulus_of_complex_number (hz : condition z) : Complex.abs z = Real.sqrt 2 := 
by
sorry

end modulus_of_complex_number_l734_734298


namespace exists_digit_sum_div_by_11_l734_734440

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_digit_sum_div_by_11 (n : ℕ) :
  ∃ k ∈ (finset.range 39).image (λ i, n + i), sum_of_digits k % 11 = 0 :=
sorry

end exists_digit_sum_div_by_11_l734_734440


namespace paths_O_to_P_paths_O_to_P_via_A_paths_O_to_P_via_A_avoiding_B_paths_O_to_P_via_A_avoiding_BC_l734_734605

open Nat

-- 1. Number of paths from O(0,0) to P(9,8)
theorem paths_O_to_P : (Nat.choose 17 9) = 24310 := 
by {
  sorry
}

-- 2. Number of paths from O(0,0) to P(9,8) via A(3,2)
theorem paths_O_to_P_via_A : (Nat.choose 5 3) * (Nat.choose 12 6) = 9240 := 
by {
  sorry
}

-- 3. Number of paths from O(0,0) to P(9,8) via A(3,2) avoiding B(6,5)
theorem paths_O_to_P_via_A_avoiding_B : ( (Nat.choose 12 6) - (Nat.choose 6 3) * (Nat.choose 6 3) ) * (Nat.choose 5 3) = 5240 := 
by {
  sorry
}

-- 4. Number of paths from O(0,0) to P(9,8) via A(3,2) avoiding BC (B(6,5) to C(8,5))
theorem paths_O_to_P_via_A_avoiding_BC : ( (Nat.choose 12 6) - (Nat.choose 7 4) * (Nat.choose 5 2) ) * (Nat.choose 5 3) = 5740 := 
by {
  sorry
}

end paths_O_to_P_paths_O_to_P_via_A_paths_O_to_P_via_A_avoiding_B_paths_O_to_P_via_A_avoiding_BC_l734_734605


namespace express_scientific_notation_l734_734314

def isScientificNotation (n : ℕ) : Prop := 
  ∃ (a : ℝ) (b : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ n = a * (10 : ℝ) ^ b

def roundToHundredThousand (n : ℝ) : ℝ :=
  ((Real.toRat n).round (10^5)).val.toReal

theorem express_scientific_notation :
  let yuan := 3185800.0 in
  roundToHundredThousand yuan = 3200000.0
  → isScientificNotation 3200000
  → 3200000 = 3.2 * (10 : ℝ) ^ 6 := by
  intros h₁ h₂
  rw [h₁]
  exact h₂

end express_scientific_notation_l734_734314


namespace spring_extension_l734_734106

theorem spring_extension (A1 A2 : ℝ) (x1 x2 : ℝ) (hA1 : A1 = 29.43) (hx1 : x1 = 0.05) (hA2 : A2 = 9.81) : x2 = 0.029 :=
by 
  sorry

end spring_extension_l734_734106


namespace solution_set_of_inequality_l734_734517

theorem solution_set_of_inequality (x : ℝ) :
  2 * |x - 1| - 1 < 0 ↔ (1 / 2) < x ∧ x < (3 / 2) :=
  sorry

end solution_set_of_inequality_l734_734517


namespace simplify_and_evaluate_expression_l734_734030

theorem simplify_and_evaluate_expression :
  let x := Real.sqrt 2 - 1 in
  (x + 2) / (x^2 - 2 * x) / ((8 * x) / (x - 2) + x - 2) = 1 :=
by
  let x := Real.sqrt 2 - 1
  sorry

end simplify_and_evaluate_expression_l734_734030


namespace goldfish_same_after_8_months_l734_734683

-- Definitions based on given conditions
def B (n : ℕ) : ℕ := 3 * 3^n
def G (n : ℕ) : ℕ := 96 * 2^n

-- Lean statement to prove the problem
theorem goldfish_same_after_8_months :
  ∃ n : ℕ, B n = G n ∧ n = 8 :=
by {
  use 8,
  show B 8 = G 8,
  sorry -- Proof of equality would go here
}

end goldfish_same_after_8_months_l734_734683


namespace georgina_parrot_days_l734_734746

theorem georgina_parrot_days
  (total_phrases : ℕ)
  (phrases_per_week : ℕ)
  (initial_phrases : ℕ)
  (phrases_now : total_phrases = 17)
  (teaching_rate : phrases_per_week = 2)
  (initial_known : initial_phrases = 3) :
  (49 : ℕ) = (((17 - 3) / 2) * 7) :=
by
  -- proof will be here
  sorry

end georgina_parrot_days_l734_734746


namespace power_equality_l734_734587

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l734_734587


namespace percent_of_total_is_correct_l734_734600

theorem percent_of_total_is_correct :
  (6.620000000000001 / 100 * 1000 = 66.2) :=
by
  sorry

end percent_of_total_is_correct_l734_734600


namespace sweets_distribution_l734_734615

/-- A mother buys a box of sweets. She kept 1/3 of the sweets and divided the rest between her 3 children.
The eldest got 8 sweets while the youngest got half as many. If there are 27 pieces of sweets in the box,
prove that the second child gets 6 sweets. -/
theorem sweets_distribution
  (total_sweets : ℕ)
  (mother_ratio : ℚ)
  (eldest_sweets : ℕ)
  (youngest_ratio : ℚ)
  (total_sweets_eq : total_sweets = 27)
  (mother_ratio_eq : mother_ratio = 1/3)
  (eldest_sweets_eq : eldest_sweets = 8)
  (youngest_ratio_eq : youngest_ratio = 1/2)
  (children_sweets : ℕ)
  (mother_kept : ℕ)
  (youngest_sweets : ℕ)
  (second_child_sweets : ℕ)
  (mother_kept_eq : mother_kept = total_sweets * (mother_ratio.num / mother_ratio.denom))
  (children_sweets_eq : children_sweets = total_sweets - mother_kept)
  (youngest_sweets_eq : youngest_sweets = eldest_sweets * (youngest_ratio.num / youngest_ratio.denom))
  (second_child_sweets_eq : second_child_sweets = children_sweets - eldest_sweets - youngest_sweets) :
  second_child_sweets = 6 := 
sorry

end sweets_distribution_l734_734615


namespace smallest_sum_arith_geo_seq_l734_734511

theorem smallest_sum_arith_geo_seq (A B C D : ℕ) 
  (h1 : A + B + C + D > 0)
  (h2 : 2 * B = A + C)
  (h3 : 16 * C = 7 * B)
  (h4 : 16 * D = 49 * B) :
  A + B + C + D = 97 :=
sorry

end smallest_sum_arith_geo_seq_l734_734511


namespace smallest_unreadable_number_l734_734847

theorem smallest_unreadable_number :
  ∃ (n : ℕ), ∃ (a b : ℕ),
    (n = 10 * a + b ∧
    34 + 21 + 63 + n = 4 * (3 + 4 + 2 + 1 + 6 + 3 + a + b) ∧
    n ≥ 0 ∧
    n < 100) ∧
    ∀ (m : ℕ), m < n → ¬ ∃ (c d : ℕ), (m = 10 * c + d ∧ 34 + 21 + 63 + m = 4 * (3 + 4 + 2 + 1 + 6 + 3 + c + d) ∧ m ≥ 0 ∧ m < 100) :=
begin
  -- proof will be here
  sorry
end

end smallest_unreadable_number_l734_734847


namespace find_x_l734_734384

noncomputable def proportional_var (k y z : ℕ) : ℕ := k * y / z^2

theorem find_x (k : ℕ) :
  proportional_var k 16 7 = 160 :=
by
  let k := 490
  have h1 : proportional_var k 4 14 = 10 := by sorry
  have h2 : proportional_var k 16 7 = 160 := by sorry
  exact h2

end find_x_l734_734384


namespace infinite_points_p_l734_734868

noncomputable def points_on_circle (P : ℝ × ℝ) : Prop :=
  let A := (0, √2)
  let B := (0, -√2)
  let C := (0, 0) -- Center of the circle
  (P.1)^2 + (P.2)^2 ≤ 2 ∧ -- Point P is inside the circle
  ((P.1 - A.1)^2 + (P.2 - A.2)^2) + ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 4 ∧ -- Sum of squares of distances
  P.2 > 0 -- P is above the x-axis (line perpendicular to diameter through the center)

theorem infinite_points_p : set.infinite {P : ℝ × ℝ | points_on_circle P} :=
sorry

end infinite_points_p_l734_734868


namespace six_digit_numbers_parity_constraint_l734_734261

def six_digit_number_count : Nat :=
  40

-- Given the digits 1, 2, 3, 4, 5, 6 form a six-digit number (without repeating any digit),
-- requiring that any two adjacent digits have different parity,
-- and digits 1 and 2 are adjacent,
-- prove that the number of such six-digit numbers is 40.
theorem six_digit_numbers_parity_constraint : 
  (∃ (digits : List Nat), digits = [1, 2, 3, 4, 5, 6] ∧ 
   (∀ (i j : Nat), i ≠ j → digits.nth i ≠ digits.nth j) ∧ 
   (∀ (i : Nat), i < 5 → (digits.nth i % 2 ≠ digits.nth (i+1) % 2)) ∧ 
   (∃ (i : Nat), i < 5 ∧ (digits.nth i = 1 ∧ digits.nth (i+1) = 2) ∨ (digits.nth i = 2 ∧ digits.nth (i+1) = 1))) →
  @Finset.card (List Nat) (@Finset.filter (List Nat) (λ digits, (digits.nodup ∧ (∀ i, i < 5 → ((digits.nth i % 2) ≠ (digits.nth (i+1) % 2))) ∧ (∃ i, i < 5 ∧ ((digits.nth i = 1 ∧ digits.nth (i+1) = 2) ∨ (digits.nth i = 2 ∧ digits.nth (i+1) = 1)))) (Finset.univ : Finset (List Nat))) = 40 := 
sorry

end six_digit_numbers_parity_constraint_l734_734261


namespace range_of_a_l734_734709

def f (a x : ℝ) : ℝ :=
  if 1 < x then 2 * f a (x - 2)
  else if -1 <= x && x <= 1 then 1 - |x|
  else 0

def g (a x : ℝ) : ℝ := f a x - Real.log x / Real.log a

theorem range_of_a {a : ℝ} (h₁ : 0 < a) (h₂ : a ≠ 1) :
  (∃ n : ℕ, n.succ = 5) →
  (∃ l : List ℝ, l.pairwise (≠) ∧ l.length = 5 ∧ ∀ x ∈ l, x ∈ Set.Icc 1 5 ∧ g a x = 0) ↔
  (a ∈ Set.Ioo (Real.sqrt 2) (Real.sqrt Real.exp1)) :=
by
  sorry

end range_of_a_l734_734709


namespace age_ratio_l734_734642

theorem age_ratio (A B : ℕ) 
  (h1 : A = 39) 
  (h2 : B = 16) 
  (h3 : (A - 5) + (B - 5) = 45) 
  (h4 : A + 5 = 44) : A / B = 39 / 16 := 
by 
  sorry

end age_ratio_l734_734642


namespace find_f_of_2005_l734_734420

noncomputable def f : ℕ → ℕ := sorry

axiom cond (m n : ℕ) (hm : m > 0) (hn : n > 0) : f (f m + f n) = m + n

theorem find_f_of_2005 : f 2005 = 2005 :=
sorry

end find_f_of_2005_l734_734420


namespace range_of_m_l734_734274

open Set

def A : Set ℝ := { x | x ≥ 2 }
def B (m : ℝ) : Set ℝ := { x | x ≥ m }

theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : 2 ≤ m :=
by {
  sorry
}

end range_of_m_l734_734274


namespace math_problem_l734_734249

noncomputable def problem_statement (α : ℝ) : Prop :=
  α ∈ set.Ioo 0 (real.pi / 2) → 
  cos(α + real.pi / 6) = 3 / 5 → 
  sin(2 * α + real.pi / 3) = 24 / 25

theorem math_problem (α : ℝ) : problem_statement α :=
by
  intros hα hcos
  sorry

end math_problem_l734_734249


namespace exponent_proof_l734_734579

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l734_734579


namespace intersection_M_N_l734_734244

def SetM : Set ℝ := { y | y ≥ 1 }
def SetN : Set ℝ := { x | ∃ (y : ℝ), y = Real.sqrt ((2 - x) / x) }
def IntersectionMN : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

theorem intersection_M_N : ∀ x, x ∈ SetM ∧ x ∈ SetN ↔ x ∈ IntersectionMN := sorry

end intersection_M_N_l734_734244


namespace third_number_in_pascals_triangle_row_51_l734_734091

theorem third_number_in_pascals_triangle_row_51 :
  let n := 51 in 
  ∃ result, result = (n * (n - 1)) / 2 ∧ result = 1275 :=
by
  let n := 51
  use (n * (n - 1)) / 2
  split
  . rfl
  . exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num; ring)
  sorry -- This 'sorry' is provided to formally conclude the directive


end third_number_in_pascals_triangle_row_51_l734_734091


namespace find_m_l734_734586

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l734_734586


namespace trajectory_of_M_l734_734770

open Real

theorem trajectory_of_M (x y : ℝ) : 
  let C : set (ℝ × ℝ) := {p | (p.1 + 1)^2 + p.2^2 = 16}
  let A := (1 : ℝ, 0 : ℝ)
  exists M in C, 
    let Q : ℝ × ℝ := _
    (M.1 = x ∧ M.2 = y) → ┃ (M, A) ┃ = ┃ (M, Q) ┃ ∧ ┃ (Q, (-1, 0)) ┃ + ┃ (M, A) ┃ = 4 →
    (x^2 / 4 + y^2 / 3 = 1) :=
sorry

end trajectory_of_M_l734_734770


namespace vasya_fraction_l734_734678

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l734_734678


namespace solution_of_system_l734_734032

variable (x y : ℝ) 

def equation1 (x y : ℝ) : Prop := 3 * |x| + 5 * y + 9 = 0
def equation2 (x y : ℝ) : Prop := 2 * x - |y| - 7 = 0

theorem solution_of_system : ∃ y : ℝ, equation1 0 y ∧ equation2 0 y := by
  sorry

end solution_of_system_l734_734032


namespace log_expr_1_log_expr_2_l734_734689

section LogarithmExpressions

theorem log_expr_1 : log 10 4 + log 10 500 - log 10 2 = 3 := sorry

theorem log_expr_2 : 
  ((1 / 27) ^ (-1 / 3)) + (log 3 16) * (log 2 (1 / 9)) = -5 := sorry

end LogarithmExpressions

end log_expr_1_log_expr_2_l734_734689


namespace problem_2010_Greek_Mathematical_Olympiad_l734_734225

noncomputable def circumcenter (ABC: Triangle) : Point := sorry
noncomputable def orthocenter (ABC: Triangle) : Point := sorry
noncomputable def reflection (P: Point) (line: Line) : Point := sorry
noncomputable def circle (center: Point) (radius: ℝ) : Set Point := sorry

variables {ABC: Triangle} (R: ℝ)
variables {BC CA AB : Line} -- sides of triangle ABC
variables {O : Point} -- circumcenter of ABC
variables {O₁ O₂ O₃ : Point} -- reflections of O!

-- Defining the circumcircle, reflections, and the circles centered at reflections
def circumcircle : Set Point := circle O R
def O₁ := reflection O BC
def O₂ := reflection O CA
def O₃ := reflection O AB

def circle₁ := circle O₁ R
def circle₂ := circle O₂ R
def circle₃ := circle O₃ R

-- Lean statement to be proved
theorem problem_2010_Greek_Mathematical_Olympiad :
  ∃ T: Point,
  (T ∈ circle₁ ∧ T ∈ circle₂ ∧ T ∈ circle₃) ∧
  let M₁ M₂ M₃ := sorry, sorry, sorry in
  ∀ L: Line (intersects L T),
  let intersect₁ := L ∩ circle₁,
      intersect₂ := L ∩ circle₂,
      intersect₃ := L ∩ circle₃ in
  ∃ P: Point,
  (P ∈ perp M₁ BC ∧ P ∈ perp M₂ CA ∧ P ∈ perp M₃ AB) :=
sorry

end problem_2010_Greek_Mathematical_Olympiad_l734_734225


namespace quadratic_inequality_contains_conditional_branch_l734_734158

def Algorithm (alg : Type) :=
  alg = "Calculating the product of two numbers"
  ∨ alg = "Calculating the distance from a point to a line"
  ∨ alg = "Solving a quadratic inequality"
  ∨ alg = "Calculating the area of a trapezoid given the lengths of its bases and height"

theorem quadratic_inequality_contains_conditional_branch (alg : Type) (h : Algorithm alg) :
  alg = "Solving a quadratic inequality" → contains_conditional_branch alg := sorry

end quadratic_inequality_contains_conditional_branch_l734_734158


namespace sum_totient_over_divisors_eq_prod_sum_totients_sum_totient_is_n_l734_734902

-- Define necessary conditions and concepts
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def phi (m : ℕ) : ℕ := m.coprime_count

lemma phi_multiplicative {a b : ℕ} (h : is_coprime a b) : 
  phi (a * b) = phi a * phi b := sorry

-- The main proof goal combining the conditions and the correct answers
theorem sum_totient_over_divisors_eq_prod_sum_totients (n : ℕ) (p : ℕ → ℕ) (r : ℕ) :
  ∀ (v : ℕ → ℕ) (pdivs : fin r → ℕ) (hp : ∀ i, nat.prime (pdivs i)) (hn : n = ∏ i : fin r, (pdivs i) ^ (v (pdivs i))),
  (∑ d in divisors n, phi d) = ∏ k in finset.range r, (∑ j in finset.range (v (pdivs k) + 1), phi ((pdivs k) ^ j)) :=
sorry

theorem sum_totient_is_n (n : ℕ) (p : ℕ → ℕ) (r : ℕ) :
  ∀ (v : ℕ → ℕ) (pdivs : fin r → ℕ) (hp : ∀ i, nat.prime (pdivs i)) (hn : n = ∏ i : fin r, (pdivs i) ^ (v (pdivs i))),
  (∑ d in divisors n, phi d) = n :=
sorry

end sum_totient_over_divisors_eq_prod_sum_totients_sum_totient_is_n_l734_734902


namespace value_of_function_gt_zero_l734_734563

theorem value_of_function_gt_zero (x : ℝ) : 
  (x^2 + x - 12 > 0) ↔ (x ∈ Set.Ioo (real.of_rat (-4)) (real.of_rat (-∞)) ∪ Set.Ioo (real.of_rat (3)) (real.of_rat (+∞))) :=
sorry

end value_of_function_gt_zero_l734_734563


namespace coin_bag_value_l734_734608

theorem coin_bag_value (p : ℕ) 
  (h1 : ∀ k, k = 3 * p) 
  (h2 : ∀ d, d = 4 * 3 * p) : 
  0.01 * p + 0.05 * (3 * p) + 0.10 * (4 * 3 * p) = 408 :=
by
  sorry

end coin_bag_value_l734_734608


namespace mouse_cannot_eat_all_but_one_l734_734005

-- Define the cube and the properties
def is_unit_cube (x y z : ℕ) : Prop := x ∈ {1, 2, 3} ∧ y ∈ {1, 2, 3} ∧ z ∈ {1, 2, 3}
def shares_face (c1 c2 : (ℕ × ℕ × ℕ)) : Prop :=
  let ⟨x1, y1, z1⟩ := c1 in
  let ⟨x2, y2, z2⟩ := c2 in
  (abs (x1 - x2) = 1 ∧ y1 = y2 ∧ z1 = z2) ∨
  (x1 = x2 ∧ abs (y1 - y2) = 1 ∧ z1 = z2) ∨
  (x1 = x2 ∧ y1 = y2 ∧ abs (z1 - z2) = 1)

def is_white (c : ℕ × ℕ × ℕ) : Prop :=
  let ⟨x, y, z⟩ := c in (x + y + z) % 2 = 0

def count_white_and_black_cubes : (ℕ × ℕ) :=
  (({(x, y, z) | is_unit_cube x y z ∧ is_white (x, y, z)}.card),
   ({(x, y, z) | is_unit_cube x y z ∧ ¬is_white (x, y, z)}.card))

theorem mouse_cannot_eat_all_but_one :
  let total_white := {c | is_unit_cube c.1 c.2 c.3 ∧ is_white c}.to_finset.card in
  let total_black := {c | is_unit_cube c.1 c.2 c.3 ∧ ¬is_white c}.to_finset.card in
  total_white = 14 ∧ total_black = 13 →
  ¬(∃ path : list (ℕ × ℕ × ℕ),
    path.nodup ∧
    (∀ p ∈ path, is_unit_cube p.1 p.2 p.3) ∧
    (∀ i, i < path.length - 1 → shares_face (path.nth_le i sorry) (path.nth_le (i + 1) sorry)) ∧
    path.filter (λ c, c = (2, 2, 2)) = [] ∧
    path.length = 26) :=
by
  intro h_card
  sorry

end mouse_cannot_eat_all_but_one_l734_734005


namespace tony_rope_length_l734_734076

-- Define the lengths of the individual ropes.
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]

-- Define the total number of ropes Tony has.
def num_ropes : ℕ := rope_lengths.length

-- Calculate the total length of ropes before tying them together.
def total_length_before_tying : ℝ := rope_lengths.sum

-- Define the length lost per knot.
def length_lost_per_knot : ℝ := 1.2

-- Calculate the total number of knots needed.
def num_knots : ℕ := num_ropes - 1

-- Calculate the total length lost due to knots.
def total_length_lost : ℝ := num_knots * length_lost_per_knot

-- Calculate the total length of the rope after tying them all together.
def total_length_after_tying : ℝ := total_length_before_tying - total_length_lost

-- The theorem we want to prove.
theorem tony_rope_length : total_length_after_tying = 35 :=
by sorry

end tony_rope_length_l734_734076


namespace andys_soda_cost_l734_734641

theorem andys_soda_cost :
  ∃ (S : ℕ), 
  (let andy_spending := 4 + S in
   let bob_spending := 6 + 2 in
   andy_spending = bob_spending ∧ S = 4) := 
begin
  use 4,
  simp [add_comm, add_assoc],
  split,
  {
    have andy_spending_eq_8 : 4 + 4 = 8, from rfl,
    have bob_spending_eq_8 : 6 + 2 = 8, from rfl,
    rw [andy_spending_eq_8, bob_spending_eq_8],
  },
  rfl,
end

end andys_soda_cost_l734_734641


namespace area_of_hexagon_l734_734245

-- Define the problem and conditions
def ABC_is_right_isosceles_triangle (A B C : Point) : Prop :=
 ∃ (AB BC : ℝ), AB = BC ∧ distance A B = AB ∧ distance B C = BC ∧ angle A B C = π / 2

def rectangle (B D E F : Point) : Prop :=
 ∃ (BD DE : ℝ), distance B D = BD ∧ distance D E = DE ∧ distance E F = BD ∧ distance F B = DE ∧ 
   angle B D E = π / 2 ∧ angle D E F = π / 2 ∧ angle E F B = π / 2 ∧ angle F B D = π / 2

def side_lengths (AB BC : ℝ) : Prop :=
  AB = 1 ∧ BC = 1

def diag_and_side_lengths (BD DE : ℝ) : Prop :=
  BD = 2 + sqrt 3 ∧ DE = 2

def side_length_of_hexagon (s : ℝ) : Prop :=
  s = 1

theorem area_of_hexagon :
  ∀ (A B C D E F : Point),
    ABC_is_right_isosceles_triangle A B C →
    rectangle B D E F →
    side_lengths (distance A B) (distance B C) →
    diag_and_side_lengths (distance B D) (distance D E) →
    side_length_of_hexagon 1 →
    regular_hexagon_area = 3 * sqrt 3 / 2 :=
by
  sorry

end area_of_hexagon_l734_734245


namespace quotient_real_iff_quotient_purely_imaginary_iff_l734_734055

variables {a b c d : ℝ} -- Declare real number variables

-- Problem 1: Proving the necessary and sufficient condition for the quotient to be a real number
theorem quotient_real_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ i : ℝ, ∃ r : ℝ, a/c = r ∧ b/d = 0) ↔ (a * d - b * c = 0) := 
by sorry -- Proof to be filled in

-- Problem 2: Proving the necessary and sufficient condition for the quotient to be a purely imaginary number
theorem quotient_purely_imaginary_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ r : ℝ, ∃ i : ℝ, a/c = 0 ∧ b/d = i) ↔ (a * c + b * d = 0) := 
by sorry -- Proof to be filled in

end quotient_real_iff_quotient_purely_imaginary_iff_l734_734055


namespace polynomial_result_l734_734577

variable (x : ℝ)

-- Given polynomials P(x) and Q(x)
def P : Polynomial ℝ := 2 * X ^ 3 - 5 * X ^ 2 + 7 * X - 8
def Q : Polynomial ℝ := 4 * X ^ 2 + 10 * X + 11

-- Resulting polynomial after multiplication
def R := P * Q

-- Requirements: R does not contain x^4 or x^3 terms
theorem polynomial_result :
  R = 8 * X ^ 5 - 17 * X ^ 2 - 3 * X - 88 := by
  sorry

end polynomial_result_l734_734577


namespace wax_calculation_l734_734883

def total_wax : ℕ := 353
def additional_wax_needed : ℕ := 22
def already_has_wax (total_wax additional_wax_needed : ℕ) : ℕ := total_wax - additional_wax_needed

theorem wax_calculation :
  already_has_wax total_wax additional_wax_needed = 331 :=
by
  have h : already_has_wax 353 22 = 353 - 22 := rfl
  have h1 : 353 - 22 = 331 := rfl
  rw [h, h1]
  sorry

end wax_calculation_l734_734883


namespace ball_returns_to_bob_after_13_throws_l734_734956

theorem ball_returns_to_bob_after_13_throws:
  ∃ n : ℕ, n = 13 ∧ (∀ k, k < 13 → (1 + 3 * k) % 13 = 0) :=
sorry

end ball_returns_to_bob_after_13_throws_l734_734956


namespace digit_seven_count_in_range_l734_734634

theorem digit_seven_count_in_range : 
  (finset.range 101).sum (λ n, (nat.digits 10 n).count 7) = 20 := sorry

end digit_seven_count_in_range_l734_734634


namespace minimum_omega_l734_734493

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l734_734493


namespace range_of_a_l734_734187

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (1 - x)

-- Given the condition f achieves its maximum > 2a - 2, we need to show a is in the range (0, 1)
theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, 0 < x ∧ ((∃ y : ℝ, 0 < y ∧ f y a = f x a ∧ f y a > 2 * a - 2))) :
  a ∈ Ioo 0 1 :=
sorry

end range_of_a_l734_734187


namespace eq_twelve_l734_734066

theorem eq_twelve : (2 + 3)^2 - (2^2 + 3^2) = 12 :=
by
  calc
    (2 + 3)^2 - (2^2 + 3^2) = 5^2 - (4 + 9) : by rw [sq]; norm_num
                          ... = 25 - 13 : by norm_num
                          ... = 12 : by norm_num

end eq_twelve_l734_734066


namespace eval_expression_l734_734724

theorem eval_expression : 3^13 / 3^3 + 2^3 = 59057 := by
  sorry

end eval_expression_l734_734724


namespace cistern_depth_l734_734603

theorem cistern_depth :
  ∀ (length width wet_surface_area depth : ℝ),
  length = 5 ∧ width = 4 ∧ wet_surface_area = 42.5 →
  length * width + 2 * depth * length + 2 * depth * width = wet_surface_area →
  depth = 1.25 :=
by
  intros l w wsa d h hw
  cases h
  cases hw
  calc
    l * w + 2 * d * l + 2 * d * w
        = 5 * 4 + 2 * d * 5 + 2 * d * 4 : by rw [←h_left, ←h_right]
    ... = 20 + 10 * d + 8 * d : by norm_num
    ... = 42.5 : by norm_num at hw_right
    ... ↔ 18 * d = 22.5 : by linarith
    ... ↔ d = 1.25 : by norm_num

end cistern_depth_l734_734603


namespace triangle_inequality_l734_734766

theorem triangle_inequality (a b c x y z : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : ∀ {α β γ : ℝ}, α + β + γ = π) 
  : bc + ca - ab < bc * cos x + ca * cos y + ab * cos z ∧ bc * cos x + ca * cos y + ab * cos z ≤ (a^2 + b^2 + c^2) / 2 := 
  sorry

end triangle_inequality_l734_734766


namespace largest_diff_neighboring_cells_l734_734044

theorem largest_diff_neighboring_cells :
  ∀ (labelling : Fin 2011.succ × Fin 2011.succ → ℕ),
  (∀ i, ∃ j, labelling (i, j) = j + 1 ∧ labelling (i, j) ≤ (2011 : ℕ)^2) →
  ∃ (i₁ i₂ : Fin 2011.succ × Fin 2011.succ),
  (i₁.1 = i₂.1 ∧ (i₁.2 - i₂.2).nat_abs = 1) ∨ (i₁.2 = i₂.2 ∧ (i₁.1 - i₂.1).nat_abs = 1) →
  (labelling i₁ - labelling i₂).nat_abs ≥ 4021 := sorry

end largest_diff_neighboring_cells_l734_734044


namespace unique_f_l734_734405

-- Let S be the set of nonzero real numbers
def S := {x : ℝ | x ≠ 0 }

-- Define a function f : S → ℝ such that:
noncomputable def f (x : S) : ℝ := sorry

-- Conditions
axiom f_1 : f ⟨1, by norm_num⟩ = 1
axiom f_2 : ∀ (x y : S), f ⟨1 / x.1 + 1 / y.1, sorry⟩ = f ⟨1 / x.1, sorry⟩ + f ⟨1 / y.1, sorry⟩
axiom f_3 : ∀ (x y : S), (x + y).1 * f ⟨(x + y).1, sorry⟩ = (x + y).1 * f x * f y

-- Conclusion
theorem unique_f : ∀ x : S, f x = 2 := sorry

end unique_f_l734_734405


namespace arithmetic_sequence_seq_general_term_sum_of_sequence_l734_734236

-- Problem (1)
theorem arithmetic_sequence {a : ℕ → ℕ} {S : ℕ → ℕ} (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n > 0 → n * S (n + 1) - (n + 1) * S n = n * (n + 1)) :
  ∀ n : ℕ, n > 0 → (S n) / n - (S (n - 1)) / (n - 1) = 1 :=
sorry

-- General term formula
theorem seq_general_term {a : ℕ → ℕ} (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n > 0 → a (n + 2) - a (n + 1) = 2) :
  ∀ n : ℕ, a n = 2 * n :=
sorry

-- Problem (2)
theorem sum_of_sequence {a b : ℕ → ℕ} (S : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → n * S (n + 1) - (n + 1) * S n = n * (n + 1))
  (h2 : a 1 = 2)
  (h3 : ∀ n : ℕ, b n = 3^(n-1) + (-1)^n * a n)
  (Sn_formula : ∀ n : ℕ, S n = n * (n + 1))
  (an_formula : ∀ n : ℕ, a n = 2 * n) :
  ∀ n : ℕ, T n = (1 / 2 : ℚ) * ((-1)^n * (2 * n + 1) + 3^n) - 1 :=
sorry

end arithmetic_sequence_seq_general_term_sum_of_sequence_l734_734236


namespace three_pow_twentyfour_mod_ten_l734_734553

-- Definition of the conditions as local lemmas
lemma three_pow_one_mod_ten : 3^1 ≡ 3 [MOD 10] := by sorry
lemma three_pow_two_mod_ten : 3^2 ≡ 9 [MOD 10] := by sorry
lemma three_pow_three_mod_ten : 3^3 ≡ 27 [MOD 10] := by sorry
lemma three_pow_four_mod_ten : 3^4 ≡ 81 [MOD 10] := by sorry

-- Main theorem to prove
theorem three_pow_twentyfour_mod_ten : 3^24 ≡ 1 [MOD 10] := by
  have h_cycle : 3^4 ≡ 1 [MOD 10] := by sorry
  have h23 : 3^24 = (3^4)^6 := by sorry
  rw [h23, pow_six, ←h_cycle]
  exact one_pow_six

end three_pow_twentyfour_mod_ten_l734_734553


namespace minimum_omega_l734_734496

theorem minimum_omega {ω : ℝ} (hω : ω > 0)
    (symmetry : ∃ k : ℤ, ∀ x : ℝ, 
      (sin (ω * x + ω * π / 2 + π / 3) = sin (-ω * x + ω * π / 2 + π / 3))) 
    : ω = 1 / 3 :=
by
  sorry

end minimum_omega_l734_734496


namespace determine_m_l734_734929

variables (m x : ℝ)
noncomputable def f (x : ℝ) := x^2 - 3*x + m
noncomputable def g (x : ℝ) := x^2 - 3*x + 5*m

theorem determine_m (h : 3 * f 5 = 2 * g 5) : m = 10 / 7 :=
by
  sorry

end determine_m_l734_734929


namespace solve_a_minus_b_l734_734924

theorem solve_a_minus_b:
  ∃ a b : ℤ, (4y + a) * (y + b) = 4*y^2 - 9*y - 36 ∧ a - b = 13 :=
  sorry

end solve_a_minus_b_l734_734924


namespace tangent_ML_circumHMN_l734_734528

variable (A B C D L M Q P N H : Type)
variable [IsTriangle B C A] [IsInscribed B C A]
variable (Ω : Circle) [IsCircumscribedBy ABC Ω]
variable [IsOnCircle A Ω]
variable [IsAngleBisector AD A]
variable [IsOnLine D BC]
variable [IsOnCircle L Ω]
variable [L ≠ A]
variable [IsMidpoint M BC]
variable [IsCircumCircle ADM Ω]
variable (PQ = Q ++ P)
variable [IsMidpoint N PQ]
variable [IsFootPerpendicular H L ND]

theorem tangent_ML_circumHMN (ABC Ω AD L M Q P N H):
  IsTangent ML (CircumCircle HMN) := sorry

end tangent_ML_circumHMN_l734_734528


namespace number_of_liars_l734_734155

def dwarf := nat -> nat -> Prop  -- Define properties of dwarf

-- Conditions
def liar (d : dwarf) := ∀ x y, true -- Placeholder definition
def knight (d : dwarf) := ∀ x y, true -- Placeholder definition
def board := fin 4 × fin 4 -- Define a 4x4 board

-- Each cell has a dwarf that is either a liar or a knight
def cell (b : board) := liar b ∨ knight b

-- Both liars and knights are present on the board
axiom liars_knights_present : ∃ b : board, liar b ∧ ∃ b' : board, knight b'

-- All dwarves state: "Among my neighbors (by edge), there are an equal number of liars and knights."
axiom dwarf_statements : ∀ b : board, 
  (dwarf b (1,0) ∧ dwarf b (0,1) ∧ dwarf b (1,2) ∧ dwarf b (2,1)) ↔ 
  (dwarf b (3, 2) ∧ dwarf b (2, 3) ∧ dwarf b (3,0) ∧ dwarf b (0,2))

-- The correct answer: Number of liars is 12
theorem number_of_liars : ∀ b : board, ∃ n : nat, n = 12 :=
sorry

end number_of_liars_l734_734155


namespace number_of_valid_permutations_l734_734331

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l734_734331


namespace person_b_days_work_alone_l734_734014

theorem person_b_days_work_alone (B : ℕ) (h1 : (1 : ℚ) / 40 + 1 / B = 1 / 24) : B = 60 := 
by
  sorry

end person_b_days_work_alone_l734_734014


namespace intersection_or_parallel_lines_l734_734523

structure Triangle (Point : Type) :=
  (A B C : Point)

structure Plane (Point : Type) :=
  (P1 P2 P3 P4 : Point)

variables {Point : Type}
variables (triABC triA1B1C1 : Triangle Point)
variables (plane1 plane2 plane3 : Plane Point)

-- Intersection conditions
variable (AB_intersects_A1B1 : (triABC.A, triABC.B) = (triA1B1C1.A, triA1B1C1.B))
variable (BC_intersects_B1C1 : (triABC.B, triABC.C) = (triA1B1C1.B, triA1B1C1.C))
variable (CA_intersects_C1A1 : (triABC.C, triABC.A) = (triA1B1C1.C, triA1B1C1.A))

theorem intersection_or_parallel_lines :
  ∃ P : Point, (
    (∃ A1 : Point, (triABC.A, A1) = (P, P)) ∧
    (∃ B1 : Point, (triABC.B, B1) = (P, P)) ∧
    (∃ C1 : Point, (triABC.C, C1) = (P, P))
  ) ∨ (
    (∃ d1 d2 d3 : Point, 
      (∀ A1 B1 C1 : Point,
        (triABC.A, A1) = (d1, d1) ∧ 
        (triABC.B, B1) = (d2, d2) ∧ 
        (triABC.C, C1) = (d3, d3)
      )
    )
  ) := by
  sorry

end intersection_or_parallel_lines_l734_734523


namespace grade_11_sample_count_l734_734607

noncomputable def high_school_students : ℕ := 1470
noncomputable def sampled_students : ℕ := 49
noncomputable def grade_10_students : ℕ := 495
noncomputable def grade_11_students : ℕ := 493
noncomputable def grade_12_students : ℕ := 482
noncomputable def first_drawn_number : ℕ := 23

theorem grade_11_sample_count : 
  ∃ n, n = 17 ∧
    systematic_sample_count high_school_students sampled_students grade_10_students grade_11_students grade_12_students first_drawn_number n :=
sorry

end grade_11_sample_count_l734_734607


namespace abs_diff_is_60_l734_734040

theorem abs_diff_is_60 
  (x y : ℕ) 
  (hxy : x ≠ y)
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9)
  (hc : 0 ≤ c ∧ c ≤ 9)
  (h_am : (x + y)/2 = 100 * a + 10 * b + c)
  (h_gm : sqrt (x * y) = 100 * c + 10 * b + a)
  (hx5 : x % 5 = 0 ∨ y % 5 = 0) : 
  abs (x - y) = 60 :=
sorry

end abs_diff_is_60_l734_734040


namespace prob1_prob2_prob3_l734_734726

-- Problem (1)
theorem prob1 (a b : ℝ) :
  ((a / 4 - 1) + 2 * (b / 3 + 2) = 4) ∧ (2 * (a / 4 - 1) + (b / 3 + 2) = 5) →
  a = 12 ∧ b = -3 :=
by { sorry }

-- Problem (2)
theorem prob2 (m n x y a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  (x = 10) ∧ (y = 6) ∧ 
  (5 * a₁ * (m - 3) + 3 * b₁ * (n + 2) = c₁) ∧ (5 * a₂ * (m - 3) + 3 * b₂ * (n + 2) = c₂) →
  (m = 5) ∧ (n = 0) :=
by { sorry }

-- Problem (3)
theorem prob3 (x y z : ℝ) :
  (3 * x - 2 * z + 12 * y = 47) ∧ (2 * x + z + 8 * y = 36) → z = 2 :=
by { sorry }

end prob1_prob2_prob3_l734_734726


namespace number_of_valid_permutations_l734_734327

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l734_734327


namespace count_of_primes_with_conditions_l734_734820

def is_prime (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def greater_than_50 (n : ℕ) : Prop :=
  n > 50

-- Define the set of two-digit primes greater than 50 with ones digit of 3
def nums : list ℕ := [53, 63, 73, 83, 93]

-- State the main theorem where the result is the count of such prime numbers
theorem count_of_primes_with_conditions : (nums.filter (λ n, is_prime n ∧ ones_digit_is_3 n ∧ is_two_digit n ∧ greater_than_50 n)).length = 3 :=
by sorry

end count_of_primes_with_conditions_l734_734820


namespace simplify_and_evaluate_l734_734909

theorem simplify_and_evaluate : 
  let x := (-1 : ℚ) / 3 
  let y := (-1 : ℚ) / 2 
  2 * (3 * x^3 - x + 3 * y) - (x - 2 * y + 6 * x^3) = -3 := 
by 
  let x := (-1 : ℚ) / 3 
  let y := (-1 : ℚ) / 2 
  calc 
    2 * (3 * x^3 - x + 3 * y) - (x - 2 * y + 6 * x^3)
     = 2 * (3 * (x^3) - (x) + 3 * (y)) - ((x) - 2 * (y) + 6 * (x^3)) : by rfl 
  ... = -3 : sorry

end simplify_and_evaluate_l734_734909


namespace min_omega_symmetry_l734_734504

theorem min_omega_symmetry :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * x + ω * (π / 2) + π / 3) = sin ((-ω) * x + ω * (π / 2) + π / 3)) →
  ω = 1 / 3 :=
by {
  sorry
}

end min_omega_symmetry_l734_734504


namespace line_up_ways_l734_734357

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l734_734357


namespace problem_quadratic_function_l734_734762

open Real

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 - 4 * x

theorem problem_quadratic_function : 
  (∀ x, f x = -2 * (x + 1)^2 + 2) ∧
  (∀ x, f 0 = 0) ∧ 
  (∀ x, x ∈ Ioi (0 : ℝ) → f (2 * x) < 0) := 
by
  sorry

end problem_quadratic_function_l734_734762


namespace jacob_needs_more_marshmallows_l734_734851

def s'mores_problem : Prop :=
  ∀ (graham_crackers marshmallows : ℕ), graham_crackers = 48 → marshmallows = 6 →
  let s'mores_with_crackers := graham_crackers / 2 in
  let s'mores_with_marshmallows := marshmallows in
  let s'mores_total := s'mores_with_crackers in
  let marshmallows_needed := s'mores_total in
  let marshmallows_to_buy := marshmallows_needed - marshmallows in
  marshmallows_to_buy = 18

theorem jacob_needs_more_marshmallows : s'mores_problem := 
by
  intros graham_crackers marshmallows hc hm
  simp only [s'mores_problem] at *
  sorry

end jacob_needs_more_marshmallows_l734_734851


namespace monotonic_function_range_l734_734790

theorem monotonic_function_range (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 + 2 * a * x - 1 ≤ 0) → -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by
  sorry

end monotonic_function_range_l734_734790


namespace Catherine_wins_with_optimal_play_l734_734013

-- Definitions for the game conditions
def circle := ℝ
def points_on_circle : ℕ := 100
def player := { Peter, Catherine : Type }

theorem Catherine_wins_with_optimal_play :
  ∀ (Peter_has_initial_move : Prop)
  (take_turns_picking_triangles : ℕ × player → Prop)
  (all_chosen_triangles_have_common_interior_point : Prop)
  (triangles_cannot_be_repeated : Prop),
  (∃ (winning_strategy : player → Prop), winning_strategy Catherine) :=
  sorry -- The proof will go here, but it's not required now

end Catherine_wins_with_optimal_play_l734_734013


namespace dishonest_shopkeeper_gain_l734_734567

theorem dishonest_shopkeeper_gain :
  let true_weight := 1000
  let false_weight := 960
  let gain := true_weight - false_weight
  let gain_percentage := (gain / true_weight.toFloat) * 100 in
  gain_percentage = 4 := by
sorry

end dishonest_shopkeeper_gain_l734_734567


namespace solution_set_inequality_l734_734760

variable {f : ℝ → ℝ}

-- Definition: f is a function defined on ℝ such that f(3) = 3 and for any x ∈ ℝ, f''(x) < 2
def function_properties (f : ℝ → ℝ) (h1 : f 3 = 3) (h2 : ∀ x : ℝ, second_derivative f x < 2) : Prop :=
  sorry

-- Main statement (math proof problem): Prove that f(x) > 2x - 3 has the solution set (-∞,3)
theorem solution_set_inequality (f : ℝ → ℝ) (h1 : f 3 = 3) (h2 : ∀ x : ℝ, second_derivative f x < 2) :
  { x : ℝ | f x > 2 * x - 3 } = set.Iio 3 :=
begin
  sorry
end

end solution_set_inequality_l734_734760


namespace function_monotonically_decreasing_on_interval_l734_734925

-- The function we are interested in
def f (x k : ℝ) : ℝ := Real.log x - k * x - k

-- The derivative of f with respect to x
def f' (x k : ℝ) : ℝ := (1 / x) - k

-- The goal is to prove the range of k for which f is monotonically decreasing on [2,5]
theorem function_monotonically_decreasing_on_interval :
  (∀ k : ℝ, (∀ x ∈ Set.Icc (2 : ℝ) 5, f' x k ≤ 0) ↔ k ≥ 1 / 2) := sorry

end function_monotonically_decreasing_on_interval_l734_734925


namespace total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l734_734985

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l734_734985


namespace characteristic_function_representation_l734_734448

theorem characteristic_function_representation
  (f : ℝ → ℝ)
  (h₁ : ∫ x in ℝ, sqrt (f x) < ∞)
  (h₂ : ∫ t in ℝ, abs (∫ x in ℝ, exp (complex.I * t * x) * sqrt (f x)) < ∞):
  ∃ (φ : ℝ → ℂ), 
    (∫ s in ℝ, |φ s|^2 = 1) ∧ 
    (∀ t ∈ ℝ, ∫ s in ℝ, φ (t + s) * complex.conj (φ s) = ∫ x in ℝ, exp (complex.I * t * x) * f x) := sorry

end characteristic_function_representation_l734_734448


namespace vasya_drove_0_4_of_total_distance_l734_734644

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l734_734644


namespace negation_of_proposition_l734_734057
open Real

theorem negation_of_proposition :
  ¬ (∃ x₀ : ℝ, (2/x₀) + log x₀ ≤ 0) ↔ ∀ x : ℝ, (2/x) + log x > 0 :=
by
  sorry

end negation_of_proposition_l734_734057


namespace angle_B_length_of_median_on_BC_perimeter_l734_734379

noncomputable def angleB (c b : ℝ) : ℝ :=
  Real.arccos (c / (2 * b))

theorem angle_B (c b : ℝ) (C : ℝ) (hb : b ≠ 0) (hc : c = 2 * b * Real.cos (angleB c b)) (hC : C = 2 * Real.pi / 3) :
  angleB c b = Real.pi / 6 := by
  sorry

theorem length_of_median_on_BC_perimeter (a b c : ℝ) (C : ℝ) (p : ℝ)
  (hab : a = b)
  (hC : C = 2 * Real.pi / 3)
  (hB : Real.arccos (c / (2 * b)) = Real.pi / 6)
  (hp : a + b + c = 4 + 2 * Real.sqrt 3) :
  let R := 2 in
  let AD := Real.sqrt 7 in
  (AD = Real.sqrt 7) := by
  sorry

end angle_B_length_of_median_on_BC_perimeter_l734_734379


namespace triangle_side_length_condition_l734_734946

theorem triangle_side_length_condition (a : ℝ) (h₁ : a > 0) (h₂ : a + 2 > a + 5) (h₃ : a + 5 > a + 2) (h₄ : a + 2 + a + 5 > a) : a > 3 :=
by
  sorry

end triangle_side_length_condition_l734_734946


namespace wet_surface_area_l734_734979

def length := 12  -- length of the cistern in meters
def width := 14   -- width of the cistern in meters
def height := 1.25 -- depth of the water in meters

def area_bottom : ℝ := length * width
def area_long_walls : ℝ := 2 * (length * height)
def area_short_walls : ℝ := 2 * (width * height)
def total_wet_area : ℝ := area_bottom + area_long_walls + area_short_walls

theorem wet_surface_area : total_wet_area = 233 :=
by
  sorry

end wet_surface_area_l734_734979


namespace sum_of_cubes_greater_than_40_l734_734065

theorem sum_of_cubes_greater_than_40
  (n : ℕ) (a : Fin n → ℝ) (h₁ : (∑ i, a i) = 10)
  (h₂ : (∑ i, (a i)^2) > 20) :
  (∑ i, (a i)^3) > 40 :=
begin
  sorry
end

end sum_of_cubes_greater_than_40_l734_734065


namespace vitya_older_than_masha_probability_l734_734532

-- Definition of Days in June
def days_in_june : ℕ := 30

-- Total number of possible outcomes for birth dates (30 days for Vitya × 30 days for Masha)
def total_outcomes : ℕ := days_in_june * days_in_june

-- Sum of favorable outcomes where Vitya is at least one day older than Masha
def favorable_outcomes : ℕ := (1 to (days_in_june - 1)).sum

-- The probability calculation function
noncomputable def probability (n f : ℕ) : ℚ := ⟨f, n⟩

-- The statement of the theorem
theorem vitya_older_than_masha_probability :
  probability total_outcomes favorable_outcomes = 29 / 60 := sorry

end vitya_older_than_masha_probability_l734_734532


namespace polynomial_factors_sum_abs_zero_l734_734862

theorem polynomial_factors_sum_abs_zero :
  let P := ∑ c in {c : ℤ | ∃ u v : ℤ, u + v = -c ∧ u * v = 4032 * c}, c
  in |P| = 0 :=
by
  /- Proof Here -/
  sorry

end polynomial_factors_sum_abs_zero_l734_734862


namespace shortest_distance_from_C1_to_C2_l734_734063

-- Define the parametric equations of Curve C₁
def C1 (θ : Real) : Real × Real :=
  (1 + Real.cos θ, Real.sin θ)

-- Define the parametric equations of Curve C₂
def C2 (t : Real) : Real × Real :=
  (-2 * Real.sqrt 2 + 0.5 * t, 1 - 0.5 * t)

-- Define the Cartesian equation of the circle derived from Curve C₁
def circle (x y : Real) : Prop :=
  (x - 1) ^ 2 + y ^ 2 = 1

-- Define the Cartesian equation of the line derived from Curve C₂
def line (x y : Real) : Prop :=
  x + y + 2 * Real.sqrt 2 - 1 = 0

-- Define the center of the circle
def circle_center : Real × Real := (1, 0)

-- Use point-to-line distance formula to find the distance from the center to the line
def point_to_line_distance (x y : Real) (a b c : Real) : Real :=
  (Real.abs (a * x + b * y + c)) / Real.sqrt (a ^ 2 + b ^ 2)

-- Prove that the shortest distance from a point on curve C₁ to a point on curve C₂ is 1
theorem shortest_distance_from_C1_to_C2 : 
  (circle_center.1, circle_center.2) = (1, 0) ∧
  line circle_center.1 circle_center.2 →
  point_to_line_distance 
    circle_center.1
    circle_center.2
    1
    1
    (2 * Real.sqrt 2 - 1) - 
  1 = 1 :=
by 
  sorry

end shortest_distance_from_C1_to_C2_l734_734063


namespace converse_proposition_converse_proposition_true_l734_734464

theorem converse_proposition (x : ℝ) (h : x > 0) : x^2 - 1 > 0 :=
by sorry

theorem converse_proposition_true (x : ℝ) (h : x^2 - 1 > 0) : x > 0 :=
by sorry

end converse_proposition_converse_proposition_true_l734_734464


namespace circle_radius_and_triangle_area_l734_734134

theorem circle_radius_and_triangle_area {CLE : Triangle} {Ω : Circle} (R : ℝ)
  (N : Point) (CE EL : Line)
  (H_circum : is_circumscribed_circle Ω CLE)
  (H_midpoint : is_midpoint_of_arc N Ω CE L)
  (H_dist_CE : distance_to_line N CE = 6)
  (H_dist_EL : distance_to_line N EL = 9)
  (H_isosceles : is_isosceles CLE)
  (H_acute : is_acute CLE) :
  R = 8 ∧ Triangle.area CLE = 15 * real.sqrt 15 := by
  sorry

end circle_radius_and_triangle_area_l734_734134


namespace power_equality_l734_734590

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l734_734590


namespace negation_proposition_l734_734056

theorem negation_proposition (l : ℝ) (h : l = 1) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) = (∀ x : ℝ, x + l < 0) := by 
  sorry

end negation_proposition_l734_734056


namespace sum_of_first_n_terms_c_n_l734_734768

noncomputable def a_n : ℕ → ℝ := λ n, 2 * n - 1

noncomputable def b_n : ℕ → ℝ := λ n, 2 ^ (n - 1)

noncomputable def c_n : ℕ → ℝ := λ n, a_n n / b_n n

noncomputable def T_n : ℕ → ℝ := λ n, (Finset.range n).sum (λ i, c_n (i + 1))

theorem sum_of_first_n_terms_c_n (n : ℕ) : T_n n = 6 - (2 * n + 3) / 2 ^ (n-1) :=
sorry

end sum_of_first_n_terms_c_n_l734_734768


namespace ron_friends_count_l734_734019

theorem ron_friends_count 
  (total_slices : ℕ) 
  (slices_per_person : ℕ) 
  (total_people : ℕ) 
  (ron_included : ℕ) 
  (total_slices_eq : total_slices = 12) 
  (slices_per_person_eq : slices_per_person = 4) 
  (total_people_eq : total_people = total_slices / slices_per_person) 
  (ron_included_eq : ron_included = 1) : 
  total_people - ron_included = 2 := 
by
  rw [total_slices_eq, slices_per_person_eq] at total_people_eq
  have h1 : total_people = 3 := total_people_eq
  rw [h1, ron_included_eq]
  sorry

end ron_friends_count_l734_734019


namespace meeting_time_l734_734566

theorem meeting_time (start_time : ℕ) (speed_a speed_b initial_distance : ℕ) (h1 : start_time = 13) (h2 : speed_a = 5) (h3 : speed_b = 7) (h4 : initial_distance = 24) : start_time + initial_distance / (speed_a + speed_b) = 15 := 
by 
  have combined_speed : ℕ := speed_a + speed_b 
  have meeting_time_hrs : ℕ := initial_distance / combined_speed 
  rw [h1, h2, h3, h4] 
  simp [combined_speed]
  exact rfl

end meeting_time_l734_734566


namespace mean_of_five_numbers_is_correct_l734_734064

-- Define the sum of the five numbers
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers
def number_of_numbers : ℚ := 5

-- Define the mean
def mean_of_five_numbers := sum_of_five_numbers / number_of_numbers

-- State the theorem
theorem mean_of_five_numbers_is_correct : mean_of_five_numbers = 3 / 20 :=
by
  -- The proof is omitted, use sorry to indicate this.
  sorry

end mean_of_five_numbers_is_correct_l734_734064


namespace medal_award_ways_l734_734067

theorem medal_award_ways :
    let total_sprinters := 10
    let canadians := 4
    let medals := 3
    -- Define the total number of ways to award medals if at most one Canadian receives a medal
    number_of_ways total_sprinters canadians medals = 360 :=
by
  let total_sprinters := 10
  let canadians := 4
  let non_canadians := total_sprinters - canadians
  let medals := 3
  let no_canadian_case := (non_canadians) * (non_canadians - 1) * (non_canadians - 2)
  let one_canadian_case := canadians * medals * (non_canadians - 1) * (non_canadians - 2)
  let total_ways := no_canadian_case + one_canadian_case

  -- Declaration of a function to find number of ways
  def number_of_ways (total_sprinters canadians medals : ℕ) : ℕ :=
    let non_canadians := total_sprinters - canadians
    let no_canadian_case := (non_canadians) * (non_canadians - 1) * (non_canadians - 2)
    let one_canadian_case := canadians * medals * (non_canadians) * (non_canadians - 1)
    no_canadian_case + one_canadian_case

  -- Prove the theorem
  have h : number_of_ways total_sprinters canadians medals = total_ways by
    -- By the conditions and separate cases, follows from proofs
    sorry

  exact h

end medal_award_ways_l734_734067


namespace vitya_older_than_masha_l734_734535

theorem vitya_older_than_masha :
  (∃ (days_in_month : ℕ) (total_pairs : ℕ) (favorable_pairs : ℕ)
     (p : ℚ),
    days_in_month = 30 ∧
    total_pairs = days_in_month * days_in_month ∧
    favorable_pairs = ∑ i in Finset.range(30), i ∧
    p = favorable_pairs / total_pairs ∧
    p = 29 / 60) :=
begin
  let days_in_month := 30,
  let total_pairs := days_in_month * days_in_month,
  let favorable_pairs := ∑ i in Finset.range(days_in_month), i,
  let p := favorable_pairs / total_pairs,
  use [days_in_month, total_pairs, favorable_pairs, p],
  split,
  { refl, },
  split,
  { refl, },
  split,
  { sorry, },
  split,
  { sorry, },
end

end vitya_older_than_masha_l734_734535


namespace fraction_of_darker_tiles_l734_734146

theorem fraction_of_darker_tiles (h : ∀ (i j : ℕ) (pattern : ℕ → ℕ → ℕ), 
  let repeating_unit := (8 * 8 : ℕ),
  let corner_unit := (4 * 4 : ℕ),
  let dark_tiles_in_corner := 9 in
  pattern i j = pattern (i + 8) (j + 8) ∧ 
  (∀ x y, 0 <= x < 4 -> 0 <= y < 4 -> pattern x y = 1) ∧ dark_tiles_in_corner = 9
  ) : 
  let total_tiles := 8 * 8,
  let dark_tiles := 9 * 4 in
  dark_tiles / total_tiles = 9 / 16 := 
by sorry

end fraction_of_darker_tiles_l734_734146


namespace javier_savings_l734_734163

theorem javier_savings (regular_price : ℕ) (discount1 : ℕ) (discount2 : ℕ) : 
  (regular_price = 50) 
  ∧ (discount1 = 40)
  ∧ (discount2 = 50) 
  → (30 = (100 * (regular_price * 3 - (regular_price + (regular_price * (100 - discount1) / 100) + regular_price / 2)) / (regular_price * 3))) :=
by
  intros h
  sorry

end javier_savings_l734_734163


namespace part_a_part_b_l734_734576

-- Define the context of the problem
variables (A B C K K1 K2 K3 O O1 O2 : Point)
variables (r R : ℝ)

-- Conditions
axiom AK1_eq_4 : dist A K1 = 4
axiom BK2_eq_6 : dist B K2 = 6
axiom AB_eq_16 : dist A B = 16
axiom Radius_eq : r > 0 ∧ R > 0 ∧ R = 8 / 5 * r

-- The problem statements as theorems we need to prove
theorem part_a : dist A K = 32 / 5 :=
sorry

theorem part_b : angle A C B = real.arccos (7 / 25) :=
sorry

end part_a_part_b_l734_734576


namespace minimum_perimeter_of_polygon_with_roots_of_Q_l734_734873

noncomputable def Q (z : ℂ) : ℂ := z^6 + (3 * real.sqrt 3 + 4) * z^3 - (3 * real.sqrt 3 + 5)

theorem minimum_perimeter_of_polygon_with_roots_of_Q :
  ∃ P : ℂ, P = 3 * real.sqrt 3 + 3 * (-cbrt (3 * real.sqrt 3 + 5) - 1) ∧
  ∀ p : ℂ, p = ∑ z in (roots Q), abs (z - w) / 2 → P ≤ p :=
sorry

end minimum_perimeter_of_polygon_with_roots_of_Q_l734_734873


namespace convex_polygon_inside_rectangle_parallelogram_in_convex_polygon_l734_734981

-- Definition of a convex polygon and its properties
variables {M : Type*} [polygon M] (convexM : is_convex M) (areaM : area M = S)

-- Part (a)
theorem convex_polygon_inside_rectangle
  (hS : area M = S) : ∃ (R : rectangle), area R ≤ 2 * S :=
sorry

-- Part (b)
theorem parallelogram_in_convex_polygon
  (hS : area M = S) : ∃ (P : parallelogram), area P ≥ S / 2 :=
sorry

end convex_polygon_inside_rectangle_parallelogram_in_convex_polygon_l734_734981


namespace same_function_group_A_proof_problem_solution_l734_734638

def f_a (x : ℝ) : ℝ := abs x
def g_a (x : ℝ) : ℝ := sqrt (x^2)
def f_b (x : ℝ) : ℝ := x
def g_b (x : ℝ) : ℝ := (sqrt x)^2
def f_c (x : ℝ) : ℝ := (if x = 1 then 0 else (x^2 - 1) / (x - 1))
def g_c (x : ℝ) : ℝ := x + 1
def f_d (x : ℝ) : ℝ := 1
def g_d (x : ℝ) : ℝ := 1

theorem same_function_group_A : ∀ x : ℝ, f_a x = g_a x :=
by
  intros x
  -- proof should show that f_a x == g_a x
  sorry

theorem proof_problem : ¬ (∀ x : ℝ, f_b x = g_b x) ∧
                ¬ (∀ x : ℝ, f_c x = g_c x) ∧
                ¬ (∀ x : ℝ, f_d x = g_d x) :=
by
  split
  { -- proof should show that f_b x and g_b x cannot be the same for all x
    sorry },
  split
  { -- proof should show that f_c x and g_c x cannot be the same for all x
    sorry },
  { -- proof should show that f_d x and g_d x cannot be the same for all x
    sorry }

theorem solution : same_function_group_A ∧ proof_problem :=
by
  have A := same_function_group_A
  have P := proof_problem
  split
  { -- acknowledge that same_function_group_A is true
    exact A },
  { -- acknowledge that proof_problem is true
    exact P }

end same_function_group_A_proof_problem_solution_l734_734638


namespace vasya_fraction_l734_734676

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l734_734676


namespace pascals_triangle_third_number_l734_734097

theorem pascals_triangle_third_number (n : ℕ) (k : ℕ) (hnk : n = 51) (hk : k = 2) :
  (nat.choose n k) = 1275 :=
by {
  subst hnk,
  subst hk,
  sorry
}

end pascals_triangle_third_number_l734_734097


namespace tangent_curve_value_l734_734260

theorem tangent_curve_value (a : ℝ) : 
  (∃ (m : ℝ), 
    (y : ℝ) := (m^2 - m + a), 
    (y = y) → (y = m + 1)) → 
  a = 2 := 
by
  sorry

end tangent_curve_value_l734_734260


namespace evaluate_expression_l734_734686

theorem evaluate_expression : 8! - 7 * 7! - 2 * 6! = 4200 := by
  sorry -- Proof placeholder

end evaluate_expression_l734_734686


namespace sequence_sum_l734_734182

noncomputable def y_sequence : ℕ → ℕ
| 1       := 85
| (k + 1) := (y_sequence k)^2 - y_sequence k

theorem sequence_sum :
  (∑' n : ℕ, 1 / (y_sequence (n + 1) + 1)) = 1 / 85 :=
by
  sorry

end sequence_sum_l734_734182


namespace triangle_ABC_angle_A_triangle_ABC_max_area_l734_734789

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + sqrt 3 * (2 * cos x ^ 2 - 1)

-- 1. Prove measure of the acute angle A
theorem triangle_ABC_angle_A (A : ℝ) 
  (h1 : f (A / 2 - π / 6) = sqrt 3)
  (hA_acute : 0 < A ∧ A < π / 2) :
  A = π / 3 :=
sorry

-- 2. Prove maximum area S of triangle ABC with given conditions
theorem triangle_ABC_max_area
  (a b c : ℝ) (A : ℝ) (R : ℝ)
  (hA : A = π / 3) (hR : R = 1)
  (h1 : f (A / 2 - π / 6) = sqrt 3)
  (ha : a = 2 * sin A)
  :
  let S := (1 / 2) * b * c * sin A in
  S ≤ 3 * sqrt 3 / 4 :=
sorry

end triangle_ABC_angle_A_triangle_ABC_max_area_l734_734789


namespace rhombus_area_4_sqrt_2_l734_734963

theorem rhombus_area_4_sqrt_2 :
  ∀ (a : ℝ) (θ : ℝ),
    a = 4 →
    θ = 45 →
    let s := a in
    let h := s * (1 / Real.sqrt 2) in
    let area := (1 / 2) * s * h in
    area = 4 * Real.sqrt 2 :=
by
  intros a θ ha hθ
  let s := a
  let h := s * (1 / Real.sqrt 2)
  let area := (1 / 2) * s * h
  rw [ha, hθ]
  have sqrt2_nonneg : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  norm_num at area
  sorry

end rhombus_area_4_sqrt_2_l734_734963


namespace line_up_ways_l734_734356

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l734_734356


namespace vasya_fraction_is_0_4_l734_734651

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l734_734651


namespace geometric_sequence_third_term_l734_734140

theorem geometric_sequence_third_term :
  ∀ (a r : ℕ), a = 2 ∧ a * r ^ 3 = 162 → a * r ^ 2 = 18 :=
by
  intros a r
  intro h
  have ha : a = 2 := h.1
  have h_fourth_term : a * r ^ 3 = 162 := h.2
  sorry

end geometric_sequence_third_term_l734_734140


namespace number_of_liars_l734_734154

def dwarf := nat -> nat -> Prop  -- Define properties of dwarf

-- Conditions
def liar (d : dwarf) := ∀ x y, true -- Placeholder definition
def knight (d : dwarf) := ∀ x y, true -- Placeholder definition
def board := fin 4 × fin 4 -- Define a 4x4 board

-- Each cell has a dwarf that is either a liar or a knight
def cell (b : board) := liar b ∨ knight b

-- Both liars and knights are present on the board
axiom liars_knights_present : ∃ b : board, liar b ∧ ∃ b' : board, knight b'

-- All dwarves state: "Among my neighbors (by edge), there are an equal number of liars and knights."
axiom dwarf_statements : ∀ b : board, 
  (dwarf b (1,0) ∧ dwarf b (0,1) ∧ dwarf b (1,2) ∧ dwarf b (2,1)) ↔ 
  (dwarf b (3, 2) ∧ dwarf b (2, 3) ∧ dwarf b (3,0) ∧ dwarf b (0,2))

-- The correct answer: Number of liars is 12
theorem number_of_liars : ∀ b : board, ∃ n : nat, n = 12 :=
sorry

end number_of_liars_l734_734154


namespace problem1_problem2_l734_734879

noncomputable def f (x : ℝ) : ℝ :=
let m := (2 * Real.cos x, 1)
let n := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
m.1 * n.1 + m.2 * n.2

theorem problem1 :
  ( ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi ) ∧
  ∀ k : ℤ, ∀ x ∈ Set.Icc ((1 : ℝ) * Real.pi / 6 + k * Real.pi) ((2 : ℝ) * Real.pi / 3 + k * Real.pi),
  f x < f (x + (Real.pi / 3)) :=
sorry

theorem problem2 (A : ℝ) (a b c : ℝ) :
  a ≠ 0 ∧ b = 1 ∧ f A = 2 ∧
  0 < A ∧ A < Real.pi ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2  →
  a = Real.sqrt 3 :=
sorry

end problem1_problem2_l734_734879


namespace vasya_fraction_l734_734671

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l734_734671


namespace coefficient_x2y2_in_expansion_l734_734462

theorem coefficient_x2y2_in_expansion :
  let c₃₂ := Nat.choose 3 2
  let c₄₂ := Nat.choose 4 2
  c₃₂ * c₄₂ = 18 :=
by
  have c₃₂ := Nat.choose 3 2
  have c₄₂ := Nat.choose 4 2
  show c₃₂ * c₄₂ = 18
  sorry

end coefficient_x2y2_in_expansion_l734_734462


namespace number_of_n_l734_734214

def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem number_of_n (k : ℕ) :
  (k = 30) ↔ (∃ L : Finset ℕ, (∀ n ∈ L, divisible_by (factorial n) ((n * (n + 2)) / 3)) ∧ L.card = k ∧ (∀ n : ℕ, n ∈ L → n ≤ 30 ∧ 0 < n)) :=
by
  sorry

end number_of_n_l734_734214


namespace mul_97_103_l734_734700

theorem mul_97_103 : (97:ℤ) = 100 - 3 → (103:ℤ) = 100 + 3 → 97 * 103 = 9991 := by
  intros h1 h2
  sorry

end mul_97_103_l734_734700


namespace mashas_end_number_is_17_smallest_starting_number_ends_with_09_l734_734001

def mashas_operation (n : ℕ) : ℕ :=
  let y := n % 10
  let x := n / 10
  3 * x + 2 * y

def mashas_stable_result (n : ℕ) : Prop :=
  mashas_operation n = n

theorem mashas_end_number_is_17 :
  ∃ n : ℕ, mashas_stable_result n ∧ n = 17 :=
sorry

def is_smallest_starting_number (n : ℕ) : Prop :=
  (nat.gcd n 17 = 17) ∧ (nat.log 10 n = 2014) ∧ (n % 100 = 9)

theorem smallest_starting_number_ends_with_09 :
  ∃ n : ℕ, is_smallest_starting_number n :=
sorry

end mashas_end_number_is_17_smallest_starting_number_ends_with_09_l734_734001


namespace complement_B_A_when_m_eq_2_necessary_not_sufficient_condition_range_l734_734775

section

def A : set ℝ := {x | 4 < x ∧ x ≤ 8}
def B (m : ℝ) : set ℝ := {x | 5 - m^2 ≤ x ∧ x ≤ 5 + m^2}

theorem complement_B_A_when_m_eq_2 : 
  (B 2) \ A = {x | (1 ≤ x ∧ x ≤ 4) ∨ (8 < x ∧ x ≤ 9)} :=
by 
  sorry

theorem necessary_not_sufficient_condition_range : 
  {m : ℝ | ∀ x, x ∈ A → x ∈ B m} = {m : ℝ | -1 < m ∧ m < 1} :=
by 
  sorry

end

end complement_B_A_when_m_eq_2_necessary_not_sufficient_condition_range_l734_734775


namespace complex_number_solution_l734_734752

variable (z : ℂ)
variable (i : ℂ)

theorem complex_number_solution (h : (1 - i)^2 / z = 1 + i) (hi : i^2 = -1) : z = -1 - i :=
sorry

end complex_number_solution_l734_734752


namespace car_dealership_math_problem_l734_734308

theorem car_dealership_math_problem
    (total_cars : ℕ)
    (hybrid_percentage : ℕ)
    (hybrid_one_headlight_percentage : ℕ)
    (hybrid_upgraded_led_percentage : ℕ)
    (non_hybrid_upgraded_led_percentage : ℕ)
    : total_cars = 600 →
      hybrid_percentage = 60 →
      hybrid_one_headlight_percentage = 40 →
      hybrid_upgraded_led_percentage = 15 →
      non_hybrid_upgraded_led_percentage = 30 →
      let hybrids := (hybrid_percentage * total_cars) / 100 in
      let hybrids_one_headlight := (hybrid_one_headlight_percentage * hybrids) / 100 in
      let hybrids_upgraded_led := (hybrid_upgraded_led_percentage * hybrids) / 100 in
      let hybrids_full_headlights := hybrids - hybrids_one_headlight - hybrids_upgraded_led in
      let non_hybrids := total_cars - hybrids in
      let non_hybrids_upgraded_led := (non_hybrid_upgraded_led_percentage * non_hybrids) / 100 in
      hybrids_full_headlights = 162 ∧ non_hybrids_upgraded_led = 72 :=
by
  intros h1 h2 h3 h4 h5
  let hybrids := (hybrid_percentage * total_cars) / 100
  have h6 : hybrids = (60 * 600) / 100 := by rw [h1, h2]
  have h7 : hybrids = 360 := by norm_num at h6
  let hybrids_one_headlight := (hybrid_one_headlight_percentage * hybrids) / 100
  have h8 : hybrids_one_headlight = (40 * 360) / 100 := by rw [h7, h3]
  have h9 : hybrids_one_headlight = 144 := by norm_num at h8
  let hybrids_upgraded_led := (hybrid_upgraded_led_percentage * hybrids) / 100
  have h10 : hybrids_upgraded_led = (15 * 360) / 100 := by rw [h7, h4]
  have h11 : hybrids_upgraded_led = 54 := by norm_num at h10
  let hybrids_full_headlights := hybrids - hybrids_one_headlight - hybrids_upgraded_led
  have h12 : hybrids_full_headlights = 360 - 144 - 54 := by rw [h7, h9, h11]
  have h13 : hybrids_full_headlights = 162 := by norm_num at h12
  let non_hybrids := total_cars - hybrids
  have h14 : non_hybrids = 600 - 360 := by rw [h1, h7]
  have h15 : non_hybrids = 240 := by norm_num at h14
  let non_hybrids_upgraded_led := (non_hybrid_upgraded_led_percentage * non_hybrids) / 100
  have h16 : non_hybrids_upgraded_led = (30 * 240) / 100 := by rw [h15, h5]
  have h17 : non_hybrids_upgraded_led = 72 := by norm_num at h16
  exact ⟨h13, h17⟩

end car_dealership_math_problem_l734_734308


namespace preston_total_received_l734_734895

theorem preston_total_received :
  let cost_per_sandwich := 5
  let cost_per_side_dish := 3
  let cost_per_drink := 1.5
  let delivery_fee := 20
  let service_charge_rate := 0.05
  let discount_on_order_threshold := 50
  let early_bird_discount_rate := 0.10
  let group_discount_rate := 0.20
  let voucher_discount_rate := 0.15
  let tip_rate := 0.10
  let num_sandwiches := 18
  let num_side_dishes := 10
  let num_drinks := 15

  let initial_cost_sandwiches := num_sandwiches * cost_per_sandwich
  let initial_cost_side_dishes := num_side_dishes * cost_per_side_dish
  let initial_cost_drinks := num_drinks * cost_per_drink

  let discounted_cost_sandwiches := initial_cost_sandwiches * (1 - early_bird_discount_rate)
  let discounted_cost_side_dishes := initial_cost_side_dishes * (1 - group_discount_rate)
  let total_cost_before_voucher := discounted_cost_sandwiches + discounted_cost_side_dishes + initial_cost_drinks

  let discount_voucher := total_cost_before_voucher * voucher_discount_rate
  let total_cost_after_voucher := total_cost_before_voucher - discount_voucher

  let total_cost_with_delivery := total_cost_after_voucher + delivery_fee
  let service_charge := if total_cost_with_delivery > discount_on_order_threshold then total_cost_with_delivery * service_charge_rate else 0
  let total_cost_with_service_charge := total_cost_with_delivery + service_charge

  let tip := total_cost_with_service_charge * tip_rate
  let total_received := total_cost_with_service_charge + tip

  total_received = 148.27 := 
begin
  sorry
end

end preston_total_received_l734_734895


namespace omega_min_value_l734_734484

theorem omega_min_value (ω : ℝ) (hω : ω > 0)
    (hSymmetry : ∀ x : ℝ, sin (ω * x + ω * π / 2 + π / 3) = sin (ω * -x + ω * π / 2 + π / 3)) :
    ω = 1 / 3 :=
begin
  sorry
end

end omega_min_value_l734_734484


namespace min_waiting_time_max_waiting_time_expected_waiting_time_l734_734990

open Nat

noncomputable def C : ℕ → ℕ → ℕ
| n, 0     => 1
| 0, k     => 0
| n+1, k+1 => C n k + C n (k+1)

def a := 1
def b := 5
def n := 5
def m := 3

def T_min := a * C (n - 1) 2 + m * n * a + b * C m 2
def T_max := a * C n 2 + b * m * n + b * C m 2
def E_T := C (n + m) 2 * (b * m + a * n) / (m + n)

theorem min_waiting_time : T_min = 40 := by
  sorry

theorem max_waiting_time : T_max = 100 := by
  sorry

theorem expected_waiting_time : E_T = 70 := by
  sorry

end min_waiting_time_max_waiting_time_expected_waiting_time_l734_734990


namespace period_of_tan_2x_minus_pi_over_4_l734_734942

theorem period_of_tan_2x_minus_pi_over_4 : 
  ∀ x, y = tan (2 * x - π / 4) → 
  (∃ T > 0, ∀ x, tan (2 * x - π / 4) = tan (2 * (x + T) - π / 4)) :=
sorry

end period_of_tan_2x_minus_pi_over_4_l734_734942


namespace calculate_f_l734_734424

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 1 + Real.logb 2 (2 - x)
else Real.ppow 2 (x - 1)

theorem calculate_f : f (-2) + f (Real.logb 2 12) = 9 :=
by
  sorry

end calculate_f_l734_734424


namespace coin_toss_sequences_l734_734315

theorem coin_toss_sequences :
  ∃ S : Finset (List Bool), 
    (∀ s ∈ S, s.length = 18 ∧ 
      (count_subsequence s [tt, tt] = 6) ∧ 
      (count_subsequence s [tt, ff] = 5) ∧ 
      (count_subsequence s [ff, tt] = 3) ∧ 
      (count_subsequence s [ff, ff] = 3)) ∧ 
    S.card = 840 := 
sorry

end coin_toss_sequences_l734_734315


namespace age_of_b_l734_734119

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 32) : b = 12 :=
by sorry

end age_of_b_l734_734119


namespace area_of_triangle_PQR_l734_734917

theorem area_of_triangle_PQR 
  (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 8)
  (P_is_center : ∃ P : ℝ, True) -- Simplified assumption that P exists
  (bases_on_same_line : True) -- Assumed true, as touching condition implies it
  : ∃ area : ℝ, area = 20 := 
by
  sorry

end area_of_triangle_PQR_l734_734917


namespace Jungkook_red_balls_count_l734_734721

-- Define the conditions
def red_balls_per_box : ℕ := 3
def boxes_Jungkook_has : ℕ := 2

-- Statement to prove
theorem Jungkook_red_balls_count : red_balls_per_box * boxes_Jungkook_has = 6 :=
by sorry

end Jungkook_red_balls_count_l734_734721


namespace Vasya_distance_fraction_l734_734656

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l734_734656


namespace find_u5_l734_734457

noncomputable def u : ℕ → ℝ

axiom seq_is_real : ∀ (n : ℕ), u n ∈ ℝ

axiom recurrence_relation : ∀ (n : ℕ), u (n + 2) = 3 * u (n + 1) + 2 * u n

axiom initial_condition_3 : u 3 = 10

axiom initial_condition_6 : u 6 = 244

theorem find_u5 : u 5 = 68 :=
by sorry

end find_u5_l734_734457


namespace five_people_lineup_count_l734_734334

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l734_734334


namespace minimum_omega_l734_734492

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l734_734492


namespace decimal_to_fraction_correct_l734_734710

-- Define a structure representing our initial decimal to fraction conversion
structure DecimalFractionConversion :=
  (decimal: ℚ)
  (vulgar_fraction: ℚ)
  (simplified_fraction: ℚ)

-- Define the conditions provided in the problem
def conversion_conditions : DecimalFractionConversion :=
  { decimal := 35 / 100,
    vulgar_fraction := 35 / 100,
    simplified_fraction := 7 / 20 }

-- State the theorem we aim to prove
theorem decimal_to_fraction_correct :
  conversion_conditions.simplified_fraction = 7 / 20 := by
  sorry

end decimal_to_fraction_correct_l734_734710


namespace find_b_l734_734288

theorem find_b (b : ℚ) (h : ∃ (P : ℚ[X]), 9 * X^3 + b * X^2 + 17 * X - 76 = (3 * X - 4) * P) : b = -17 / 6 :=
by
  sorry

end find_b_l734_734288


namespace extreme_values_a_4_find_a_minimum_minus_5_l734_734268

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem extreme_values_a_4 :
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≤ 11) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 11) ∧
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≥ 3) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 3) :=
  sorry

theorem find_a_minimum_minus_5 :
  ∀ (a : ℝ), (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x a = -5) -> (a = -12 ∨ a = 9) :=
  sorry

end extreme_values_a_4_find_a_minimum_minus_5_l734_734268


namespace tan_alpha_value_l734_734247

theorem tan_alpha_value (α : ℝ) (h : tan (π / 4 - α) = 1 / 5) : tan α = 2 / 3 :=
by
  sorry

end tan_alpha_value_l734_734247


namespace Sarah_is_26_l734_734023

noncomputable def Sarah_age (mark_age billy_age ana_age : ℕ): ℕ :=
  3 * mark_age - 4

def Mark_age (billy_age : ℕ): ℕ :=
  billy_age + 4

def Billy_age (ana_age : ℕ): ℕ :=
  ana_age / 2

def Ana_age : ℕ := 15 - 3

theorem Sarah_is_26 : Sarah_age (Mark_age (Billy_age Ana_age)) (Billy_age Ana_age) Ana_age = 26 := 
by
  sorry

end Sarah_is_26_l734_734023


namespace third_number_in_pascals_triangle_row_51_l734_734090

theorem third_number_in_pascals_triangle_row_51 :
  let n := 51 in 
  ∃ result, result = (n * (n - 1)) / 2 ∧ result = 1275 :=
by
  let n := 51
  use (n * (n - 1)) / 2
  split
  . rfl
  . exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num; ring)
  sorry -- This 'sorry' is provided to formally conclude the directive


end third_number_in_pascals_triangle_row_51_l734_734090


namespace original_cost_of_article_l734_734611

theorem original_cost_of_article : ∃ C : ℝ, 
  (∀ S : ℝ, S = 1.35 * C) ∧
  (∀ C_new : ℝ, C_new = 0.75 * C) ∧
  (∀ S_new : ℝ, (S_new = 1.35 * C - 25) ∧ (S_new = 1.0875 * C)) ∧
  (C = 95.24) :=
sorry

end original_cost_of_article_l734_734611


namespace trajectory_of_point_P_l734_734617

noncomputable def circle_equation (x y : ℝ) : Prop :=
x^2 + y^2 = 1

def is_tangent_line (x y a b : ℝ) : Prop :=
(a = x ∧ b ≠ y) ∨ (b = y ∧ a ≠ x)

def angle_condition (angle : ℝ) : Prop :=
angle = real.pi / 3

theorem trajectory_of_point_P (P : ℝ × ℝ) (A B : ℝ × ℝ)
  (h1 : ∀ x y, circle_equation x y → is_tangent_line A.1 A.2 x y)
  (h2 : ∀ x y, circle_equation x y → is_tangent_line B.1 B.2 x y)
  (h3 : angle_condition (real.angle (P.1 - A.1) (P.2 - A.2) (P.1 - B.1) (P.2 - B.2))) :
  P.1^2 + P.2^2 = 4 :=
sorry

end trajectory_of_point_P_l734_734617


namespace radius_of_large_circle_l734_734210

/-- Five circles are described with the given properties. -/
def small_circle_radius : ℝ := 2

/-- The angle between any centers of the small circles is 72 degrees due to equal spacing. -/
def angle_between_centers : ℝ := 72

/-- The final theorem states that the radius of the larger circle is as follows. -/
theorem radius_of_large_circle (number_of_circles : ℕ)
        (radius_small : ℝ)
        (angle : ℝ)
        (internally_tangent : ∀ (i : ℕ), i < number_of_circles → Prop)
        (externally_tangent : ∀ (i j : ℕ), i ≠ j → i < number_of_circles → j < number_of_circles → Prop) :
  number_of_circles = 5 →
  radius_small = small_circle_radius →
  angle = angle_between_centers →
  (∃ R : ℝ, R = 4 * Real.sqrt 5 - 2) 
:= by
  -- mathematical proof goes here
  sorry

end radius_of_large_circle_l734_734210


namespace opposite_of_neg_six_l734_734939

theorem opposite_of_neg_six :
  ∃ x : ℤ, -6 + x = 0 ∧ x = 6 :=
by
  use 6
  split
  · simp
  · rfl

end opposite_of_neg_six_l734_734939


namespace max_probability_through_intersections_l734_734141

theorem max_probability_through_intersections (x : ℕ) (P : ℕ → ℚ) :
  (∀ (t1 t2 : ℕ), t1 + 30 = t2 + x → P(t1) = t1 / (t1 + 30) * 120 / (x + 120)) →
  (x = 60 → P x = 4 / 9) :=
by
  sorry

end max_probability_through_intersections_l734_734141


namespace sum_of_values_of_N_l734_734512

theorem sum_of_values_of_N (N : ℂ) : (N * (N - 8) = 12) → (∃ x y : ℂ, N = x ∨ N = y ∧ x + y = 8) :=
by
  sorry

end sum_of_values_of_N_l734_734512


namespace operation_evaluation_l734_734554

theorem operation_evaluation : 65 + 5 * 12 / (180 / 3) = 66 :=
by
  -- Parentheses
  have h1 : 180 / 3 = 60 := by sorry
  -- Multiplication and Division
  have h2 : 5 * 12 = 60 := by sorry
  have h3 : 60 / 60 = 1 := by sorry
  -- Addition
  exact sorry

end operation_evaluation_l734_734554


namespace range_of_m_l734_734774

variables {m : ℝ} {x : ℝ}

def p : Prop := ∃ x,  m * x^2 + 1 ≤ 0
def q : Prop := ∀ x, x^2 + m * x + 1 > 0

theorem range_of_m (h : ¬(p ∨ q)) : m ≥ 2 :=
sorry

end range_of_m_l734_734774


namespace lacy_correct_percentage_l734_734009

theorem lacy_correct_percentage (x : ℕ) (hx : x > 0) :
  let total_problems := 4 * x,
      missed_problems := 2 * x,
      correct_problems := total_problems - missed_problems in
  (correct_problems : ℝ) / total_problems * 100 = 50 :=
by
  sorry

end lacy_correct_percentage_l734_734009


namespace domino_placement_l734_734887

theorem domino_placement (board : Fin 6 × Fin 6 → bool)
  (h_board_size : ∀ (i : Fin 6) (j : Fin 6), board (i, j) = false ∨ board (i, j) = true)
  (h_dominoes : ∃ dominos : Fin 11 → (Fin 6 × Fin 6) × (Fin 6 × Fin 6),
     ∀ i : Fin 11, board (dominos i).1 = true ∧ board (dominos i).2 = true ∧ 
       (dominos i).1 ≠ (dominos i).2): 
  (∃ new_domino : (Fin 6 × Fin 6) × (Fin 6 × Fin 6), 
     (board new_domino.1 = false ∧ board new_domino.2 = false) ∧ 
     new_domino.1 ≠ new_domino.2) :=
sorry

end domino_placement_l734_734887


namespace tip_percentage_correct_l734_734957

def lunch_cost := 50.20
def total_spent := 60.24
def tip_percentage := ((total_spent - lunch_cost) / lunch_cost) * 100

theorem tip_percentage_correct : tip_percentage = 19.96 := 
by
  sorry

end tip_percentage_correct_l734_734957


namespace necessary_but_not_sufficient_condition_for_extremum_l734_734578

variable (f : ℝ → ℝ)

theorem necessary_but_not_sufficient_condition_for_extremum :
  (∃ x, is_local_extremum f x) → (∃ x, deriv f x = 0) ∧ (¬ (∃ x, deriv f x = 0 → is_local_extremum f x)) :=
by
  sorry

end necessary_but_not_sufficient_condition_for_extremum_l734_734578


namespace A_inter_B_cardinality_l734_734776

noncomputable def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
noncomputable def B : Set ℕ := { x | 0 ≤ x ∧ x < 5 }

noncomputable def A_inter_B : Set ℝ := A ∩ (B : Set ℝ)

theorem A_inter_B_cardinality : (A_inter_B.to_finset.card = 4) :=
by {
  sorry
}

end A_inter_B_cardinality_l734_734776


namespace Sarah_is_26_l734_734022

noncomputable def Sarah_age (mark_age billy_age ana_age : ℕ): ℕ :=
  3 * mark_age - 4

def Mark_age (billy_age : ℕ): ℕ :=
  billy_age + 4

def Billy_age (ana_age : ℕ): ℕ :=
  ana_age / 2

def Ana_age : ℕ := 15 - 3

theorem Sarah_is_26 : Sarah_age (Mark_age (Billy_age Ana_age)) (Billy_age Ana_age) Ana_age = 26 := 
by
  sorry

end Sarah_is_26_l734_734022


namespace perpendicular_vectors_cos2theta_zero_l734_734787

theorem perpendicular_vectors_cos2theta_zero (θ : ℝ)
  (h : (1 : ℝ) * (-1 : ℝ) + (cos θ) * (2 * cos θ) = 0) : 
  cos (2 * θ) = 0 := 
sorry

end perpendicular_vectors_cos2theta_zero_l734_734787


namespace tan_QDE_eq_rat_l734_734435

theorem tan_QDE_eq_rat : 
  ∃ Q : ℝ × ℝ, 
  ∃ (φ : ℝ), 
    let D := (0 : ℝ × ℝ);
    let E := (15 : ℝ × ℝ);
    let F := (9, 12 : ℝ × ℝ) in
    dist D E = 15 ∧
    dist E F = 17 ∧
    dist F D = 16 ∧ 
    ∠ QDE = φ ∧ ∠ QEF = φ ∧ ∠ QFD = φ ∧
    Real.tan φ = 168 / 385 :=
sorry

end tan_QDE_eq_rat_l734_734435


namespace transformed_curve_l734_734958

def scaling_transformation (f : ℝ → ℝ) (x' y' : ℝ) : Prop :=
  ∃ x y : ℝ, f x = y ∧ x' = 3 * x ∧ y' = 2 * y

theorem transformed_curve :
  scaling_transformation (λ x, cos (6 * x)) x' y' ↔ y' = 2 * cos (2 * x') :=
by
  sorry

end transformed_curve_l734_734958


namespace pascal_triangle_third_number_l734_734094

theorem pascal_triangle_third_number (n : ℕ) (h : n + 1 = 52) : (nat.choose n 2) = 1275 := by
  have h_n : n = 51 := by
    linarith
  rw [h_n]
  norm_num

end pascal_triangle_third_number_l734_734094


namespace arithmetic_geometric_seq_l734_734251

open Real

theorem arithmetic_geometric_seq (a b : ℝ) (n : ℕ) (h1 : 2 * a = 1 + b) (h2 : (2 * a + 2) ^ 2 = 4 * (3 * b + 1)) :
  (∀ n, a_n = 2 * n - 1 ∧ b_n = 2^(n + 1)) → 
  (∀ n, c_n = 2 / (a_n * (log (√2) b_n - 1))) →
  ∑ i in range n, c i = 2 * n / (2 * n + 1) :=
begin
  -- initial steps to establish required sequences 
  sorry
end

end arithmetic_geometric_seq_l734_734251


namespace profit_is_correct_l734_734636

-- Define the constants for expenses
def cost_of_lemons : ℕ := 10
def cost_of_sugar : ℕ := 5
def cost_of_cups : ℕ := 3

-- Define the cost per cup of lemonade
def price_per_cup : ℕ := 4

-- Define the number of cups sold
def cups_sold : ℕ := 21

-- Define the total revenue
def total_revenue : ℕ := cups_sold * price_per_cup

-- Define the total expenses
def total_expenses : ℕ := cost_of_lemons + cost_of_sugar + cost_of_cups

-- Define the profit
def profit : ℕ := total_revenue - total_expenses

-- The theorem stating the profit
theorem profit_is_correct : profit = 66 := by
  sorry

end profit_is_correct_l734_734636


namespace seq_50_is_1284_l734_734053

-- Define the sequence of positive integers which are either powers of 4 or sums of distinct powers of 4.
def seq : ℕ → ℕ
| 0     := 1
| (n+1) := seq n + if binary_digit_sum' n 4 then 4^n else 0

-- Define the term of the sequence at position 50
def seq_50_term : ℕ := seq 49

-- Statement: Prove that the 50th term of the given sequence is 1284
theorem seq_50_is_1284 : seq_50_term = 1284 :=
by sorry

end seq_50_is_1284_l734_734053


namespace min_omega_symmetry_l734_734500

theorem min_omega_symmetry :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * x + ω * (π / 2) + π / 3) = sin ((-ω) * x + ω * (π / 2) + π / 3)) →
  ω = 1 / 3 :=
by {
  sorry
}

end min_omega_symmetry_l734_734500


namespace binary_to_octal_conversion_l734_734179

-- A definition to convert the binary number 11011100 to a natural number
def binary_to_decimal (b : List ℕ) : ℕ :=
  b.foldr (λ x acc, x + 2 * acc) 0

-- A definition to convert a natural number to an octal representation (as a list of digits)
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else aux (n / 8) ((n % 8) :: acc)
  aux n []

-- The proof statement
theorem binary_to_octal_conversion :
  binary_to_decimal [1,1,0,1,1,1,0,0] = 220 ∧ decimal_to_octal 220 = [3,3,4] :=
by {
  -- Proof goes here
  sorry
}

end binary_to_octal_conversion_l734_734179


namespace vitya_older_than_masha_probability_l734_734533

-- Definition of Days in June
def days_in_june : ℕ := 30

-- Total number of possible outcomes for birth dates (30 days for Vitya × 30 days for Masha)
def total_outcomes : ℕ := days_in_june * days_in_june

-- Sum of favorable outcomes where Vitya is at least one day older than Masha
def favorable_outcomes : ℕ := (1 to (days_in_june - 1)).sum

-- The probability calculation function
noncomputable def probability (n f : ℕ) : ℚ := ⟨f, n⟩

-- The statement of the theorem
theorem vitya_older_than_masha_probability :
  probability total_outcomes favorable_outcomes = 29 / 60 := sorry

end vitya_older_than_masha_probability_l734_734533


namespace deductive_reasoning_correct_l734_734639

theorem deductive_reasoning_correct :
    (deductive_reasoning : Prop) →
    (syllogism_form : Prop) →
    (conclusion_correct : Prop) →
    (premise_correctness : Prop) →
    (general_to_specific → syllogism_form ∧ conclusion_correct ∧ general_to_specific ∧ premise_correctness) →
    (syllogism_form) :=
begin
  intros deductive_reasoning syllogism_form conclusion_correct premise_correctness H,
  exact H.2.1,
end

end deductive_reasoning_correct_l734_734639


namespace shoe_discount_is_40_l734_734181

variable p_shoes : ℝ  -- Original price of one pair of shoes
variable p_dress : ℝ  -- Original price of the dress
variable d : ℝ       -- Discount percentage on the dress
variable s : ℝ       -- Total amount Daniela spends

-- Define the conditions
def shoes_price := 50 : ℝ
def dress_price := 100 : ℝ
def dress_discount := 20 : ℝ
def total_spent := 140 : ℝ

theorem shoe_discount_is_40 :
  2 * (p_shoes - (x / 100) * p_shoes) + p_dress * (1 - d / 100) = s -> x = 40 :=
  by
  intros
  sorry

-- Assigning the conditions to the variables
#eval p_shoes = shoes_price
#eval p_dress = dress_price
#eval d = dress_discount
#eval s = total_spent

end shoe_discount_is_40_l734_734181


namespace sqrt_product_l734_734696

theorem sqrt_product (a b c : ℝ) (ha : a = 72) (hb : b = 18) (hc : c = 8) :
  (Real.sqrt a) * (Real.sqrt b) * (Real.sqrt c) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l734_734696


namespace minimum_omega_l734_734480

theorem minimum_omega (ω : ℝ) (h_omega_pos : ω > 0) :
    (∃ y : ℝ → ℝ, (∀ x, y x = sin (ω * x + ω * (π / 2) + (π / 3))) ∧ 
    (∀ x, y x = y (-x))) →
    (ω = 1 / 3) :=
sorry

end minimum_omega_l734_734480


namespace complex_number_is_real_implies_m_three_l734_734825

theorem complex_number_is_real_implies_m_three (m : ℝ) :
  ∀ z : ℂ, z = (m ^ 2 - 5 * m + 6) + (m - 3) * complex.I → z.im = 0 → m = 3 :=
by
  intro z h_eq h_im
  sorry

end complex_number_is_real_implies_m_three_l734_734825


namespace vitya_masha_probability_l734_734544

theorem vitya_masha_probability :
  let total_days := 30
  let total_pairs := total_days * total_days
  let favourable_pairs := (∑ k in Finset.range total_days, k)
  total_pairs = 900 ∧ favourable_pairs = 435 ∧
  probability (Vitya at_least_one_day_older_than_Masha) = favourable_pairs / total_pairs :=
by {
  let total_days := 30,
  let total_pairs := total_days * total_days,
  let favourable_pairs := (∑ k in Finset.range total_days, k),
  
  have h1: total_pairs = 900 := by norm_num,
  have h2: favourable_pairs = 435 := by norm_num,

  have probability := 435.0 / 900.0,
  norm_num at top,
  simp,
}

end vitya_masha_probability_l734_734544


namespace five_people_lineup_count_l734_734338

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l734_734338


namespace fibonacci_mod_10_2006_l734_734279

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_mod_10_2006 : fibonacci 2006 % 10 = 3 :=
sorry

end fibonacci_mod_10_2006_l734_734279


namespace smallest_divisor_of_odd_five_digit_number_l734_734854

theorem smallest_divisor_of_odd_five_digit_number (m : ℕ) (h_odd : odd m) (h_digits : 10000 ≤ m ∧ m < 100000) (h_div : 437 ∣ m) : 
  ∃ n, n > 437 ∧ n ∣ m ∧ ∀ k, (k > 437 ∧ k ∣ m) → n ≤ k :=
  sorry

end smallest_divisor_of_odd_five_digit_number_l734_734854


namespace angle_difference_is_2x_minus_80_l734_734865

theorem angle_difference_is_2x_minus_80 (x : ℝ) (A B C D : Type)
  (triangle_isosceles : AB = AC) (angle_ABC : ∠ B = 50)
  (line_perpendicular : line_through C ⊥ AB) (angle_ACD : ∠ ACD = x)
  (angle_A : ∠ A = 50) (angle_C1 : ∠ C1 = x) 
  : (angle_C1 - (80 - x) = 2x - 80) :=
sorry

end angle_difference_is_2x_minus_80_l734_734865


namespace range_of_a_l734_734802

-- Define the sets A and B as described in the problem
def setA : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def setB (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}

-- State the theorem that we need to prove
theorem range_of_a (a : ℝ) : (A ∪ setB(a) = A) ↔ (a ∈ Iic (-4) ∨ a ∈ Ici 2) :=
by {
  sorry
}

end range_of_a_l734_734802


namespace ratio_of_auto_finance_companies_credit_l734_734681

theorem ratio_of_auto_finance_companies_credit
    (total_consumer_credit : ℝ)
    (percent_auto_installment_credit : ℝ)
    (credit_by_auto_finance_companies : ℝ)
    (total_auto_credit : ℝ)
    (hc1 : total_consumer_credit = 855)
    (hc2 : percent_auto_installment_credit = 0.20)
    (hc3 : credit_by_auto_finance_companies = 57)
    (htotal_auto_credit : total_auto_credit = percent_auto_installment_credit * total_consumer_credit) :
    (credit_by_auto_finance_companies / total_auto_credit) = (1 / 3) := 
by
  sorry

end ratio_of_auto_finance_companies_credit_l734_734681


namespace part1_solution_set_part2_range_b_l734_734870

-- Part (1)
theorem part1_solution_set (a : ℝ) (a_pos : 0 < a) (h1 : a < 1 ∨ a = 1 ∨ 1 < a) :
∃ s : Set ℝ, 
  (h1 = Or.inl (And.intro h1_left h1_right) → s = Set.Icc 2 (2 / a)) ∧ 
  (h1 = Or.inr (Or.inl h1_center) → s = {2}) ∧ 
  (h1 = Or.inr (Or.inr h1_right) → s = Set.Icc (2 / a) 2) := sorry

-- Part (2)
theorem part2_range_b (b : ℝ) (ineq : 1 ≤ x ∧ x ≤ 5 → 2 * x^2 - b * x + 2 ≥ 0) :
b ≤ 4 * Real.sqrt 2 := sorry

end part1_solution_set_part2_range_b_l734_734870


namespace ice_cream_total_volume_l734_734506

/-- 
  The interior of a right, circular cone is 12 inches tall with a 3-inch radius at the opening.
  The interior of the cone is filled with ice cream.
  The cone has a hemisphere of ice cream exactly covering the opening of the cone.
  On top of this hemisphere, there is a cylindrical layer of ice cream of height 2 inches 
  and the same radius as the hemisphere (3 inches).
  Prove that the total volume of ice cream is 72π cubic inches.
-/
theorem ice_cream_total_volume :
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  V_cone + V_hemisphere + V_cylinder = 72 * Real.pi :=
by {
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  sorry
}

end ice_cream_total_volume_l734_734506


namespace correct_quadratic_equation_l734_734967

theorem correct_quadratic_equation :
  (∀ a b c : ℤ, ((a ≠ 1 ∧ (4 + -3 = 1 ∧ 4 * -3 = -12) ∧ (b = -a) ∧ (c = -12 * a)) ∨ 
  (7 + 3 = 10 ∧ 7 * 3 = 21 ∧ (b = -10 * a) ∧ (c = 21 * a))) → 
  (a = 1 ∧ b = -10 ∧ c = 21)) → 
  ∃ a b c : ℤ, (correct_eq : a * (x^2) + b * x + c = 0 ∧ (correct_eq = (x^2 + 10 * x + 21 = 0))) :=
begin
  sorry
end

end correct_quadratic_equation_l734_734967


namespace locus_of_orthocenter_is_reflection_l734_734232

noncomputable def orthocenter (A B C H : Point) : Prop :=
(Line.through (A, B) ⊥ Line.through (H, C)) ∧ 
(Line.through (A, C) ⊥ Line.through (H, B))

variables {Γ : Circle} {B C : Point} (H_locus : Circle)
variable (A : Point)
variable (H : Point)

-- Given conditions:
variables (A_on_Γ : A ∈ Γ)
variables (B_on_Γ : B ∈ Γ)
variables (C_on_Γ : C ∈ Γ)

theorem locus_of_orthocenter_is_reflection :
  is_locus_of_orthocenter H_locus Γ B C A_on_Γ B_on_Γ C_on_Γ :=
sorry

end locus_of_orthocenter_is_reflection_l734_734232


namespace rectangle_perimeter_l734_734071

variable (w l A P : ℝ)

-- Conditions
def width_condition : Prop := w = 4
def area_condition : Prop := A = 44
def area_formula : Prop := A = w * l
def length_calc : Prop := l = 44 / 4

-- Statement to prove
theorem rectangle_perimeter (h1 : width_condition) (h2 : area_condition) (h3 : area_formula) (h4 : length_calc) : P = 2 * w + 2 * l := by
  sorry

end rectangle_perimeter_l734_734071


namespace henry_jeans_cost_l734_734814

-- Let P be the price of the socks, T be the price of the t-shirt, and J be the price of the jeans
def P := 5
def T := P + 10
def J := 2 * T

-- Goal: Prove that J = 30
theorem henry_jeans_cost : J = 30 :=
by
  unfold P T J -- unfold all the definitions
  simp -- simplify the expression
  exact rfl

end henry_jeans_cost_l734_734814


namespace determine_m_l734_734928

variables (m x : ℝ)
noncomputable def f (x : ℝ) := x^2 - 3*x + m
noncomputable def g (x : ℝ) := x^2 - 3*x + 5*m

theorem determine_m (h : 3 * f 5 = 2 * g 5) : m = 10 / 7 :=
by
  sorry

end determine_m_l734_734928


namespace train_crossing_man_time_l734_734980

theorem train_crossing_man_time
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (man_speed_kmph : ℝ)
  (relative_speed_mps : ℝ)
  (time : ℝ) :
  train_length = 210 →
  train_speed_kmph = 25 →
  man_speed_kmph = 2 →
  relative_speed_mps = (train_speed_kmph + man_speed_kmph) * 1000 / 3600 →
  time = train_length / relative_speed_mps →
  time = 28 :=
by
  intros h_train_length h_train_speed h_man_speed h_relative_speed h_time
  rw [h_train_length, h_train_speed, h_man_speed] at *
  have rel_speed : relative_speed_mps = (25 + 2) * 1000 / 3600 := by rw [h_relative_speed]
  rw rel_speed at *
  have rel_speed_simplified : relative_speed_mps = 7.5 := by norm_num
  rw rel_speed_simplified at *
  have time_computation : time = 210 / 7.5 := by rw [h_time]
  norm_num at time_computation
  exact time_computation
  done

end train_crossing_man_time_l734_734980


namespace no_corner_cut_possible_l734_734383

-- Define the cube and the triangle sides
def cube_edge_length : ℝ := 15
def triangle_side1 : ℝ := 5
def triangle_side2 : ℝ := 6
def triangle_side3 : ℝ := 8

-- Main statement: Prove that it's not possible to cut off a corner of the cube to form the given triangle
theorem no_corner_cut_possible :
  ¬ (∃ (a b c : ℝ),
    a^2 + b^2 = triangle_side1^2 ∧
    b^2 + c^2 = triangle_side2^2 ∧
    c^2 + a^2 = triangle_side3^2 ∧
    a^2 + b^2 + c^2 = 62.5) :=
sorry

end no_corner_cut_possible_l734_734383


namespace mike_total_work_time_l734_734385

theorem mike_total_work_time :
  let wash_time := 10
  let oil_change_time := 15
  let tire_change_time := 30
  let paint_time := 45
  let engine_service_time := 60

  let num_wash := 9
  let num_oil_change := 6
  let num_tire_change := 2
  let num_paint := 4
  let num_engine_service := 3
  
  let total_minutes := 
        num_wash * wash_time +
        num_oil_change * oil_change_time +
        num_tire_change * tire_change_time +
        num_paint * paint_time +
        num_engine_service * engine_service_time

  let total_hours := total_minutes / 60

  total_hours = 10 :=
  by
    -- Definitions of times per task
    let wash_time := 10
    let oil_change_time := 15
    let tire_change_time := 30
    let paint_time := 45
    let engine_service_time := 60

    -- Definitions of number of tasks performed
    let num_wash := 9
    let num_oil_change := 6
    let num_tire_change := 2
    let num_paint := 4
    let num_engine_service := 3

    -- Calculate total minutes
    let total_minutes := 
      num_wash * wash_time +
      num_oil_change * oil_change_time +
      num_tire_change * tire_change_time +
      num_paint * paint_time +
      num_engine_service * engine_service_time
    
    -- Calculate total hours
    let total_hours := total_minutes / 60

    -- Required equality to prove
    have : total_hours = 10 := sorry
    exact this

end mike_total_work_time_l734_734385


namespace rod_division_l734_734609

theorem rod_division (parts10 parts12 parts15 : ℕ) 
  (h1 : parts10 = 10) (h2 : parts12 = 12) (h3 : parts15 = 15) :
  let markings10 := parts10 - 1 in
  let markings12 := parts12 - 1 in
  let markings15 := parts15 - 1 in
  let common10_12 := Nat.lcm parts10 parts12 / Nat.gcd parts10 parts12 in
  let common10_15 := Nat.lcm parts10 parts15 / Nat.gcd parts10 parts15 in
  let common12_15 := Nat.lcm parts12 parts15 / Nat.gcd parts12 parts15 in
  let total_markings := markings10 + markings12 + markings15 
                       - (common10_12 + common10_15 + common12_15) in
  total_markings + 1 = 28 :=
by
  sorry

end rod_division_l734_734609


namespace tangerines_taken_l734_734074

theorem tangerines_taken : 
  ∀ (oranges_initial tangerines_initial oranges_taken tangerines_left : ℕ), 
    oranges_initial = 5 → tangerines_initial = 17 → oranges_taken = 2 → 
    (tangerines_left = tangerines_initial - 10) →
    (tangerines_left = (oranges_initial - oranges_taken) + 4) → 
    tangerines_left = 7 :=
by {
  intros oranges_initial tangerines_initial oranges_taken tangerines_left,
  sorry
}

end tangerines_taken_l734_734074


namespace angle_AC_BD_eq_90_l734_734861

-- Define the problem conditions and struct
open EuclideanGeometry

-- Define the segments and midpoints
variables {P Q R : Point}

-- Given conditions
axiom PQ_length : P.dist Q = 2
axiom QR_length : Q.dist R = Real.sqrt 5
axiom PR_length : P.dist R = 3

-- Prove the statement
theorem angle_AC_BD_eq_90 :
  ∠AC = ∠BD := 90 :=
by 
  sorry

end angle_AC_BD_eq_90_l734_734861


namespace Vasya_distance_fraction_l734_734658

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l734_734658


namespace find_m_l734_734583

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l734_734583


namespace count_triplets_solution_l734_734213

theorem count_triplets (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  (a * b * c = 2 * (a + b + c) + 4) → (a = 1 ∧ (b, c) = (3, 12) ∨ a = 1 ∧ (b, c) = (4, 7) ∨ a = 2 ∧ (b, c) = (2, 6)) :=
sorry

theorem solution :
  ∃ (triplets : Finset (ℕ × ℕ × ℕ)), (∀ triplet ∈ triplets, 0 < triplet.1 ∧ 0 < triplet.2.1 ∧ 0 < triplet.2.2 ∧ triplet.1 * triplet.2.1 * triplet.2.2 = 2 * (triplet.1 + triplet.2.1 + triplet.2.2) + 4) ∧ (triplets.card = 3) :=
begin
  let triplets : Finset (ℕ × ℕ × ℕ) := {(1, 3, 12), (1, 4, 7), (2, 2, 6)},
  use triplets,
  split,
  {
    intros triplet h_triplet,
    rcases h_triplet with ⟨h11, h12, h13⟩ | ⟨h11, h12, h13⟩ | ⟨h11, h12, h13⟩;
    repeat { split; norm_num };
    { exact nat.pos_of_ne_zero },
  },
  {
    exact rfl,
  }
end

end count_triplets_solution_l734_734213


namespace tesseract_parallel_edges_l734_734282

noncomputable def num_parallel_edges_in_tesseract : Nat :=
  36

theorem tesseract_parallel_edges 
  (tesseract : Type)
  [has_edges tesseract]
  (num_edges_tesseract : has_edges.num_edges tesseract = 32) :
  num_pairs_parallel_edges tesseract = num_parallel_edges_in_tesseract :=
sorry

end tesseract_parallel_edges_l734_734282


namespace pascals_triangle_third_number_l734_734099

theorem pascals_triangle_third_number (n : ℕ) (k : ℕ) (hnk : n = 51) (hk : k = 2) :
  (nat.choose n k) = 1275 :=
by {
  subst hnk,
  subst hk,
  sorry
}

end pascals_triangle_third_number_l734_734099


namespace disk_sum_inequality_l734_734844

theorem disk_sum_inequality (n : ℕ) (R : Fin n → ℝ) (P : Fin n → ℝ × ℝ) (O : ℝ × ℝ)
  (h_n : 6 ≤ n)
  (h_rad: ∀ i j, i < j → R i ≥ R j)
  (h_disk: ∀ i, dist O (P i) ≤ R i) :
  (∑ i, dist O (P i)) ≥ (∑ i in Finset.range (n - 6), R ⟨i + 6, sorry⟩) :=
begin
  sorry
end

end disk_sum_inequality_l734_734844


namespace minimum_omega_l734_734479

theorem minimum_omega (ω : ℝ) (h_omega_pos : ω > 0) :
    (∃ y : ℝ → ℝ, (∀ x, y x = sin (ω * x + ω * (π / 2) + (π / 3))) ∧ 
    (∀ x, y x = y (-x))) →
    (ω = 1 / 3) :=
sorry

end minimum_omega_l734_734479


namespace range_of_a_l734_734799

-- Definitions of the line and circle, and the required distance condition
def line (a : ℝ) (t : ℝ): ℝ × ℝ := (a * t, 1 - 2 * t)

def circle (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 2 * (Real.cos (θ + Real.pi / 4))
  (ρ * Real.cos θ, ρ * Real.sin θ)

def distance_from_line (a : ℝ) (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  |2 - a - a| / Real.sqrt (4 + a^2)

theorem range_of_a (a : ℝ) : 
  (∃ (P : ℝ × ℝ), (is_on_circle P (1, -1) (Real.sqrt 2)) ∧ distance_from_line a P = (Real.sqrt 2) / 2) 
  → (ℝ -> ℝ) :
  ∀ t θ, 
    let line_point := line a t in
    let circle_point := circle θ in
    distance_from_line a circle_point <= Real.sqrt 2 / 2 ↔ a ∈ interval ⟨2/7, 2⟩ 
:= sorry

end range_of_a_l734_734799


namespace inequality_solution_set_a_eq_1_smallest_positive_integer_a_l734_734269

-- Definition of the inequality and conditions
def inequality (x a : ℝ) : Prop := 2 * |x - 3| + |x - 4| < a^2 + a

-- Statement for Question 1
theorem inequality_solution_set_a_eq_1 :
  { x : ℝ | 2 * |x - 3| + |x - 4| < 2 } = { x : ℝ | 8 / 3 < x ∧ x < 4 } :=
sorry

-- Statement for Question 2
theorem smallest_positive_integer_a :
  ∃ (a : ℕ), 
   (a > 0) ∧ (∃ x : ℝ, 2 * |x - 3| + |x - 4| < a^2 + a) ∧ 
   ∀ b : ℕ, (b > 0) ∧ (∀ x : ℝ, 2 * |x - 3| + |x - 4| < b^2 + b → 2 * |x - 3| + |x - 4| < a^2 + a) :=
∃ a : ℕ, a = 1 :=
sorry

end inequality_solution_set_a_eq_1_smallest_positive_integer_a_l734_734269


namespace smallest_of_powers_l734_734113

theorem smallest_of_powers :
  (2:ℤ)^(55) < (3:ℤ)^(44) ∧ (2:ℤ)^(55) < (5:ℤ)^(33) ∧ (2:ℤ)^(55) < (6:ℤ)^(22) := by
  sorry

end smallest_of_powers_l734_734113


namespace count_restricted_arrangements_l734_734345

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l734_734345


namespace football_game_attendance_l734_734955

theorem football_game_attendance :
  (∃ S : ℕ, 
    let M := S - 20 in
    let W := M + 50 in
    let F := S + M in
    let expected_total := 350 in
    let actual_total := expected_total + 40 in
    (S + M + W + F) = actual_total) ↔ S = 80 := 
by 
  sorry

end football_game_attendance_l734_734955


namespace georgina_parrot_days_l734_734748

theorem georgina_parrot_days 
    (initial_phrases : ℕ := 3) 
    (current_phrases : ℕ := 17) 
    (phrases_per_week : ℕ := 2) 
    (days_per_week : ℕ := 7) 
    : nat :=
  let new_phrases := current_phrases - initial_phrases in
  let weeks := new_phrases / phrases_per_week in
  let days := weeks * days_per_week in
  days = 49
by
  sorry

end georgina_parrot_days_l734_734748


namespace principal_argument_l734_734259

theorem principal_argument (z : ℂ) (h : ∥2 * z + 1 / z∥ = 1) : 
  ∃ k : ℤ, let θ := complex.arg z in 
    k * real.pi + real.pi / 2 - real.arccos (3 / 4) / 2 ≤ θ ∧ 
    θ ≤ k * real.pi + real.pi / 2 + real.arccos (3 / 4) / 2 :=
sorry

end principal_argument_l734_734259


namespace angle_P_is_180_degrees_l734_734846

-- Define the triangle PQR with PQ = PR.
variables {P Q R S : Type*} [metric_space P] [metric_space Q] [metric_space R] [metric_space S]

-- Define PQ = PR (isosceles triangle)
def is_isosceles_triangle (P Q R : Type*) [metric_space P] [metric_space Q] [metric_space R] :=
  dist P Q = dist P R

-- Define that S is a point on PR such that QS bisects ∠PQR
def bisects_angle (Q S P R : Type*) [metric_space Q] [metric_space S] [metric_space P] [metric_space R] :=
  true -- This is a placeholder for actual bisecting condition 

-- Define QS = QR (isosceles triangle)
def is_isosceles_triangle_QS_QR (S Q R : Type*) [metric_space S] [metric_space Q] [metric_space R] :=
  dist Q S = dist Q R

-- Given conditions
variables (P Q R S : Type*) [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
variable [is_isosceles_triangle P Q R]
variable [bisects_angle Q S P R]
variable [is_isosceles_triangle_QS_QR S Q R]

-- Goal: Prove that ∠P = 180 degrees
theorem angle_P_is_180_degrees :
  ∠P = 180 :=
sorry

end angle_P_is_180_degrees_l734_734846


namespace binomial_coeff_n_eq_10_l734_734368

theorem binomial_coeff_n_eq_10 (n : ℕ) (h : binomial (n, 2) * 3^(n-2) = 5 * 3^n) : n = 10 :=
by
  sorry

end binomial_coeff_n_eq_10_l734_734368


namespace opposite_of_neg_six_l734_734941

theorem opposite_of_neg_six : ∃ x : ℤ, -6 + x = 0 ∧ x = 6 :=
by {
  use 6,
  split,
  {
    linarith, 
  },
  {
    refl,
  }
}

end opposite_of_neg_six_l734_734941


namespace baron_munchausen_claim_l734_734682

noncomputable def polygon_with_internal_point_dividing_it_into_three (P : Polygon) (O : Point) : Prop :=
  ∀ L : Line, O ∈ L → ∃ L1 L2 L3 : Line, (∀ x ∈ L1 ∧ x ∈ L2 ∧ x ∈ L3, polygon_region_divided_by (P, O, L, L1, L2, L3) = 3)

theorem baron_munchausen_claim (P : Polygon) (O : Point) :
  polygon_with_internal_point_dividing_it_into_three P O :=
sorry

end baron_munchausen_claim_l734_734682


namespace hyperbola_focus_coordinates_l734_734935

theorem hyperbola_focus_coordinates
  (hx : ∃ x ∈ {0}, ∃ y ∈ {y | y = 2}, (x, y)) -- One focus on the y-axis
  (center : (x,2) = (-1, 2)) -- Center is at (-1,2)
  :  (∃ x' ∈ {-2}, ∃ y' ∈ {y | y = 2}, (x', y')) := -- Other focus at (-2,2)
sorry

end hyperbola_focus_coordinates_l734_734935


namespace minimum_omega_l734_734497

theorem minimum_omega {ω : ℝ} (hω : ω > 0)
    (symmetry : ∃ k : ℤ, ∀ x : ℝ, 
      (sin (ω * x + ω * π / 2 + π / 3) = sin (-ω * x + ω * π / 2 + π / 3))) 
    : ω = 1 / 3 :=
by
  sorry

end minimum_omega_l734_734497


namespace collinear_points_l734_734395

variables {A B C P M N Q : Type} [geometry.equilateral_triangle A B C]
variables (P : geometry.point_on_circumcircle A B C) 
variables (M N Q : geo.parallel_LinesIntersect A B C P)

theorem collinear_points (hM : geometry.is_parallel (line_through P M) (line_through C A))
                        (hN : geometry.is_parallel (line_through P N) (line_through A B))
                        (hQ : geometry.is_parallel (line_through P Q) (line_through B C)) :
  geometry.collinear M N Q := 
by sorry

end collinear_points_l734_734395


namespace vasya_drives_fraction_l734_734666

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l734_734666


namespace parabola_axis_of_symmetry_range_l734_734364

theorem parabola_axis_of_symmetry_range
  (a b c m n t : ℝ)
  (h₀ : 0 < a)
  (h₁ : m = a * 1^2 + b * 1 + c)
  (h₂ : n = a * 3^2 + b * 3 + c)
  (h₃ : m < n)
  (h₄ : n < c)
  (h_t : t = -b / (2 * a)) :
  (3 / 2) < t ∧ t < 2 :=
sorry

end parabola_axis_of_symmetry_range_l734_734364


namespace count_restricted_arrangements_l734_734340

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l734_734340


namespace find_75th_element_l734_734467

namespace Sequence75

-- Definition for the element of row i
def row_elem (i : ℕ) : ℕ := 3 * i

-- Definition for the total number of elements in row i
def row_count (i : ℕ) : ℕ := 2 * i

-- Definition for calculating the total number of elements up to row n
def total_elements_up_to (n : ℕ) : ℕ :=
  (List.range (n + 1)).sum (λ i, row_count i)

-- The main theorem we want to prove
theorem find_75th_element : 
  (∃ r, total_elements_up_to r < 75 ∧ 75 ≤ total_elements_up_to (r + 1)) →
  let r := Nat.find (λ r, total_elements_up_to r < 75 ∧ 75 ≤ total_elements_up_to (r + 1))
  -- row where the 75th element resides
  75 - total_elements_up_to r ≤ row_count (r + 1) →
  row_elem (r + 1) = 27 := sorry
 
end Sequence75

end find_75th_element_l734_734467


namespace pyramid_volume_eq_one_twelfth_l734_734234

noncomputable theory

def volume_of_regular_triangular_pyramid : ℝ :=
let length_of_side := 1 in
let centroid_to_vertex := (2 / 3 : ℝ) * (Real.sqrt 3 / 2) * length_of_side in
let base_area := (Real.sqrt 3 / 4) * (length_of_side^2) in
(1 / 3) * base_area * centroid_to_vertex

theorem pyramid_volume_eq_one_twelfth :
  let length_of_side := 1 in
  let centroid_to_vertex := (2 / 3 : ℝ) * (Real.sqrt 3 / 2) * length_of_side in
  let base_area := (Real.sqrt 3 / 4) * (length_of_side^2) in
  (1 / 3) * base_area * centroid_to_vertex = 1 / 12 :=
by
  let length_of_side := 1
  let centroid_to_vertex := (2 / 3 : ℝ) * (Real.sqrt 3 / 2) * length_of_side
  let base_area := (Real.sqrt 3 / 4) * (length_of_side^2)
  have vol := (1 / 3) * base_area * centroid_to_vertex
  show vol = 1 / 12
  sorry

end pyramid_volume_eq_one_twelfth_l734_734234


namespace shaded_square_percentage_l734_734104

def square_area (side : ℕ) : ℕ := side * side

def shaded_area : ℕ := 2 * 2 + (4 * 4 - 2 * 2) + (6 * 6 - 4 * 4)

def shaded_percentage (total_area : ℕ) (shaded_area : ℕ) : ℚ :=
  (shaded_area * 100) / total_area

theorem shaded_square_percentage (side : ℕ) (h_side : side = 7) :
  shaded_percentage (square_area side) shaded_area ≈ 73.47 :=
begin
  -- Definitions based on given conditions
  let main_area := square_area 7,
  let shaded := shaded_area,

  -- Calculation of percentage
  let percent := (shaded * 100 : ℚ) / main_area,
  have h : percent ≈ 73.47,
  { sorry }, -- Proof to be filled

  -- Conclusion
  exact h,
end

end shaded_square_percentage_l734_734104


namespace vasya_fraction_is_0_4_l734_734654

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l734_734654


namespace construct_point_c_max_angle_l734_734772

open EuclideanGeometry

variables {A B O C : Point} 

-- Definitions of points and the distances
variables (dOA dOB dOC : ℝ)
variable (circleABC : Circle A B)

-- Conditions are given: A, B on one side of an angle with vertex O, and C must be on the other side
def point_of_max_angle (A B O : Point) (dOA dOB : ℝ) : ℝ :=
  dOA * dOB

theorem construct_point_c_max_angle (A B O C : Point) (dOA dOB : ℝ) : 
  dist O C = Real.sqrt (dOA * dOB) := by
  sorry

end construct_point_c_max_angle_l734_734772


namespace probability_of_prime_l734_734722

-- Define the set of numbers on the balls
def balls : Finset ℕ := {4, 5, 6, 7, 8, 9, 10, 11}

-- Define the predicate for prime numbers
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the set of prime numbers in our set of balls
def prime_balls : Finset ℕ := balls.filter is_prime

-- State the theorem for the probability of picking a prime number
theorem probability_of_prime : 
  prime_balls.card / balls.card = 3 / 8 :=
by
  sorry

end probability_of_prime_l734_734722


namespace parabola_transformation_roots_sum_l734_734933

theorem parabola_transformation_roots_sum :
  (let a := 3 in let b := 4 in let c := 7 in
   let p := c + Real.sqrt (b + 1) in
   let q := c - Real.sqrt (b + 1) in
   p + q = 14) :=
sorry

end parabola_transformation_roots_sum_l734_734933


namespace valid_sentences_in_gnollish_l734_734459

def words : List String := ["splargh", "glumph", "amr"]

def isValidSentence (sentence : List String) : Bool :=
  ¬(sentence = ["splargh", "glumph"] ++ sentence.drop 2) ∧
  ¬(sentence = sentence.take 2 ++ ["glumph", "amr"])

def countValidSentences : Nat :=
  let totalSentences := List.product (List.product words words) words
  let validSentences := totalSentences.filter isValidSentence
  validSentences.length

theorem valid_sentences_in_gnollish : countValidSentences = 16 :=
  by
    sorry

end valid_sentences_in_gnollish_l734_734459


namespace tiles_ABABABA_probability_l734_734744

theorem tiles_ABABABA_probability :
  let A := 4  -- Number of A tiles
  let B := 3  -- Number of B tiles
  let total_tiles := 7
  let favorable_outcomes := 1  -- Only one arrangement ABABABA
  let total_arrangements := Nat.choose total_tiles (A) in
  (favorable_outcomes / total_arrangements : ℚ) = 1 / 35 := by
  sorry

end tiles_ABABABA_probability_l734_734744


namespace pets_after_one_month_l734_734856

def initial_dogs := 30
def initial_cats := 28
def initial_lizards := 20
def adoption_rate_dogs := 0.5
def adoption_rate_cats := 0.25
def adoption_rate_lizards := 0.2
def new_pets := 13

theorem pets_after_one_month :
  (initial_dogs - (initial_dogs * adoption_rate_dogs) +
   initial_cats - (initial_cats * adoption_rate_cats) +
   initial_lizards - (initial_lizards * adoption_rate_lizards) +
   new_pets) = 65 :=
by 
  -- proof goes here
  sorry

end pets_after_one_month_l734_734856


namespace total_liars_l734_734152

-- Define a dwarf type
inductive Dwarf
| liar
| knight

-- The board is a 4x4 grid of dwarves
def board := Fin 4 × Fin 4 → Dwarf

-- Define the property that a dwarf's statement about its neighbors being half liars and half knights
def equal_neighbors (b : board) (x y : Fin 4) : Prop :=
  let neighbors := [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] -- Neighbors by edge (validity to be checked within 4x4 bounds)
  let valid_neighbors := neighbors.filter (λ xy, ∀ i j, (0 ≤ i ∧ i < 4) ∧ (0 ≤ j ∧ j < 4) → xy = (i, j))
  let liars := valid_neighbors.count (λ (xy : Fin 4 × Fin 4), b xy = Dwarf.liar)
  let knights := valid_neighbors.count (λ (xy : Fin 4 × Fin 4), b xy = Dwarf.knight)
  liars = knights

-- The main theorem to prove: There are 12 liars in the board.
theorem total_liars : ∃ b : board, (∀ x y, b (x, y) = Dwarf.liar ∨ b (x, y) = Dwarf.knight)
  ∧ (∃ x y, b (x, y) = Dwarf.liar)
  ∧ (∃ x y, b (x, y) = Dwarf.knight)
  ∧ (∀ x y, equal_neighbors b x y)
  ∧ (Finset.univ.sum (λ (xy : Fin 4 × Fin 4), if b xy = Dwarf.liar then 1 else 0) = 12) :=
by
  sorry

end total_liars_l734_734152


namespace binomial_coefficient_condition_l734_734366

theorem binomial_coefficient_condition (n : ℕ) (h : binomial_expansion_condition (3 + x) n) :
  n = 10 :=
by
  sorry

end binomial_coefficient_condition_l734_734366


namespace probability_neither_motor_requires_attention_l734_734316

namespace Workshop

variable (Ω : Type) [ProbabilitySpace Ω]

-- Define events A and B.
variable (A B : Event Ω)

-- Define the probabilities.
variable (P_A : ℝ)
variable (P_B : ℝ)

-- Define the independence condition.
variable (indep : Independent A B)

-- Given: P(A) = 0.85, P(B) = 0.8
axiom prob_A : P(A) = 0.85
axiom prob_B : P(B) = 0.8

-- Prove the probability that neither motor will require attention within one hour.
theorem probability_neither_motor_requires_attention : P(A ∩ B) = 0.68 := by
  -- Apply the independence of events.
  have h : P(A ∩ B) = P(A) * P(B), from indep
  
  -- Substitute the given probabilities.
  rw [prob_A, prob_B] at h
   
  -- Simplify and conclude.
  sorry

end Workshop

end probability_neither_motor_requires_attention_l734_734316


namespace number_of_type_II_color_patterns_correct_l734_734180

noncomputable def number_of_type_II_color_patterns (n m : ℕ) : ℚ :=
if h : n % 2 = 0 then 
  1 / (2 * n) * ((∑ d in nat.divisors n, nat.totient d * m ^ (n / d)) : ℚ) + 1 / 4 * m ^ (n / 2) * (1 + m)
else 
  1 / (2 * n) * ((∑ d in nat.divisors n, nat.totient d * m ^ (n / d)) : ℚ) + 1 / 2 * m ^ ((n + 1) / 2)

-- Prove that the defined number of Type II color patterns is indeed correct
theorem number_of_type_II_color_patterns_correct (n m : ℕ) : 
  number_of_type_II_color_patterns n m = 
  if n % 2 = 0 then 
    1 / (2 * n) * ((∑ d in nat.divisors n, nat.totient d * m ^ (n / d)) : ℚ) + 1 / 4 * m ^ (n / 2) * (1 + m)
  else 
    1 / (2 * n) * ((∑ d in nat.divisors n, nat.totient d * m ^ (n / d)) : ℚ) + 1 / 2 * m ^ ((n + 1) / 2) :=
sorry

end number_of_type_II_color_patterns_correct_l734_734180


namespace henry_jeans_cost_l734_734813

-- Let P be the price of the socks, T be the price of the t-shirt, and J be the price of the jeans
def P := 5
def T := P + 10
def J := 2 * T

-- Goal: Prove that J = 30
theorem henry_jeans_cost : J = 30 :=
by
  unfold P T J -- unfold all the definitions
  simp -- simplify the expression
  exact rfl

end henry_jeans_cost_l734_734813


namespace vasya_fraction_l734_734673

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l734_734673


namespace exists_plane_with_line_and_parallel_l734_734809

open Plane

-- Definitions for lines and planes
variable (Point : Type)
variable (Line : Type)
variable (Plane : Type)
variables (a b : Line)
variables [Nonintersection : ∀ (x y : Line), x ≠ y → ¬∃ p : Point, p ∈ x ∧ p ∈ y]
variables [ContainsLine : Plane → Line → Prop]
variables [ParallelLine : Line → Plane → Prop]

-- Proposition that needs to be proven
theorem exists_plane_with_line_and_parallel
  (h_skew : ∀ (p : Point), ¬(p ∈ a ∧ p ∈ b)) :
  ∃ (P : Plane), ContainsLine P a ∧ ParallelLine b P := 
  sorry

end exists_plane_with_line_and_parallel_l734_734809


namespace polygon_circumscribed_around_circle_has_triangle_l734_734442

noncomputable def circumscribed_polygon (n : ℕ) (h : n ≥ 4) : Prop :=
  ∀ (sides : Fin n → ℝ), (∃ a b c : ℝ, (a ∈ {sides 0, sides 1, sides 2, sides 3}) ∧
                                      (b ∈ {sides 0, sides 1, sides 2, sides 3}) ∧
                                      (c ∈ {sides 0, sides 1, sides 2, sides 3}) ∧
                                      a + b > c ∧
                                      a + c > b ∧
                                      b + c > a)

theorem polygon_circumscribed_around_circle_has_triangle (n : ℕ) (h : n ≥ 4) :
  circumscribed_polygon n h := sorry

end polygon_circumscribed_around_circle_has_triangle_l734_734442


namespace root_increases_implies_m_neg7_l734_734741

theorem root_increases_implies_m_neg7 
  (m : ℝ) 
  (h : ∃ x : ℝ, x ≠ 3 ∧ x = -m - 4 → x = 3) 
  : m = -7 := by
  sorry

end root_increases_implies_m_neg7_l734_734741


namespace max_k_strictly_increasing_sequence_l734_734778

open Nat

theorem max_k_strictly_increasing_sequence :
  ∀ (a : ℕ → ℕ),
    (a 1 = choose 10 0) ∧
    (a 2 = choose 10 1) ∧
    (a 3 = choose 10 2) ∧
    (a 4 = choose 10 3) ∧
    (a 5 = choose 10 4) ∧
    (a 6 = choose 10 5) ∧
    (a 7 = choose 10 6) ∧
    (a 8 = choose 10 7) ∧
    (a 9 = choose 10 8) ∧
    (a 10 = choose 10 9) ∧
    (a 11 = choose 10 10) →
  ∃ k : ℕ, k = 6 ∧ ∀ n : ℕ, (1 ≤ n ∧ n < k) → a n < a (n + 1) :=
by
  sorry

end max_k_strictly_increasing_sequence_l734_734778


namespace segment_length_C_C_l734_734959

-- Define the points C and C''.
def C : ℝ × ℝ := (-3, 2)
def C'' : ℝ × ℝ := (-3, -2)

-- State the theorem that the length of the segment from C to C'' is 4.
theorem segment_length_C_C'' : dist C C'' = 4 := by
  sorry

end segment_length_C_C_l734_734959


namespace pieces_picked_by_edward_l734_734886

theorem pieces_picked_by_edward (pieces_olivia : ℕ) (total_pieces : ℕ)
  (h₀ : pieces_olivia = 16)
  (h₁ : total_pieces = 19) :
  total_pieces - pieces_olivia = 3 :=
by
  rw [h₀, h₁]
  norm_num
  sorry

end pieces_picked_by_edward_l734_734886


namespace sweets_distribution_l734_734616

/-- A mother buys a box of sweets. She kept 1/3 of the sweets and divided the rest between her 3 children.
The eldest got 8 sweets while the youngest got half as many. If there are 27 pieces of sweets in the box,
prove that the second child gets 6 sweets. -/
theorem sweets_distribution
  (total_sweets : ℕ)
  (mother_ratio : ℚ)
  (eldest_sweets : ℕ)
  (youngest_ratio : ℚ)
  (total_sweets_eq : total_sweets = 27)
  (mother_ratio_eq : mother_ratio = 1/3)
  (eldest_sweets_eq : eldest_sweets = 8)
  (youngest_ratio_eq : youngest_ratio = 1/2)
  (children_sweets : ℕ)
  (mother_kept : ℕ)
  (youngest_sweets : ℕ)
  (second_child_sweets : ℕ)
  (mother_kept_eq : mother_kept = total_sweets * (mother_ratio.num / mother_ratio.denom))
  (children_sweets_eq : children_sweets = total_sweets - mother_kept)
  (youngest_sweets_eq : youngest_sweets = eldest_sweets * (youngest_ratio.num / youngest_ratio.denom))
  (second_child_sweets_eq : second_child_sweets = children_sweets - eldest_sweets - youngest_sweets) :
  second_child_sweets = 6 := 
sorry

end sweets_distribution_l734_734616


namespace total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l734_734986

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l734_734986


namespace vasya_fraction_l734_734672

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l734_734672


namespace area_sin_curve_l734_734916

open Real

theorem area_sin_curve :
  (2 * ∫ x in 0..π, sin x) = 4 :=
by
  sorry

end area_sin_curve_l734_734916


namespace sin_cos_alpha_value_sin2alpha_over_diff_sin_cos_l734_734811

theorem sin_cos_alpha_value (α : ℝ) (hα : α ∈ set.Icc (-(Real.pi / 2)) 0)
  (h_collinear : ∃ k : ℝ, (cos α - (Real.sqrt 2) / 3, -1) = k * (sin α, 1)) :
  sin α + cos α = (Real.sqrt 2) / 3 :=
by
  sorry

theorem sin2alpha_over_diff_sin_cos (α : ℝ) (hα : α ∈ set.Icc (-(Real.pi / 2)) 0)
  (h_collinear : ∃ k : ℝ, (cos α - (Real.sqrt 2) / 3, -1) = k * (sin α, 1)) :
  (sin (2 * α)) / (sin α - cos α) = 7 / 12 :=
by
  sorry

end sin_cos_alpha_value_sin2alpha_over_diff_sin_cos_l734_734811


namespace proj_composition_l734_734864

-- Define the projection matrices
def proj_matrix_u : Matrix (Fin 2) (Fin 2) ℚ :=
  (20⁻¹ : ℚ) • Matrix.of' ![![16, 8], ![8, 4]]

def proj_matrix_v : Matrix (Fin 2) (Fin 2) ℚ :=
  (5⁻¹ : ℚ) • Matrix.of' ![![1, -2], ![-2, 4]]

-- Define the product of the two projection matrices
def result_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  proj_matrix_v ⬝ proj_matrix_u

-- The theorem stating the result of the projections
theorem proj_composition :
  result_matrix = Matrix.of' ![![(-2 : ℚ) / 25, (-4 : ℚ) / 25], [(-4 : ℚ) / 25, 14 / 25]] :=
  sorry

end proj_composition_l734_734864


namespace pascal_triangle_third_number_l734_734101

theorem pascal_triangle_third_number {n k : ℕ} (h : n = 51) (hk : k = 2) : Nat.choose n k = 1275 :=
by
  rw [h, hk]
  norm_num

#check pascal_triangle_third_number

end pascal_triangle_third_number_l734_734101


namespace length_segment_QR_ge_b_l734_734127

theorem length_segment_QR_ge_b 
  (a b : ℝ) (h : a > b) (P : ℝ × ℝ) (θ : ℝ) :
  let A_1 := (-a, 0)
      A_2 := (a, 0)
      c := real.sqrt (a^2 - b^2)
      F_1 := (-c, 0)
      F_2 := (c, 0)
      P := (a * real.cos θ, b * real.sin θ)
      Q := (-a * real.cos θ, - (a^2 * real.sin θ) / b)
      R := (-a * real.cos θ, - (a^2 * real.sin θ) / b + b / real.sin θ) 
  in (dist Q R) ≥ b :=
sorry

end length_segment_QR_ge_b_l734_734127


namespace rhombus_area_l734_734961

noncomputable def area_of_rhombus (s : ℝ) (θ : ℝ) : ℝ :=
  s * s * Math.sin θ

theorem rhombus_area :
  area_of_rhombus 4 (Math.pi / 4) = 8 * Real.sqrt 2 :=
by
  sorry

end rhombus_area_l734_734961


namespace range_of_a_l734_734792

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x + 2 else -x^2 + 2*x + 2

theorem range_of_a (a : ℝ) (h : f (a^2 - 4*a) + f (-4) > 15) : a < -1 ∨ a > 5 :=
  sorry

end range_of_a_l734_734792


namespace sum_base6_l734_734170

theorem sum_base6 : 
  ∀ (a b : ℕ) (h1 : a = 4532) (h2 : b = 3412),
  (a + b = 10414) :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end sum_base6_l734_734170


namespace final_price_percentage_l734_734121

theorem final_price_percentage (original_price sale_price final_price : ℝ) (h1 : sale_price = 0.9 * original_price) 
(h2 : final_price = sale_price - 0.1 * sale_price) : final_price / original_price = 0.81 :=
by
  sorry

end final_price_percentage_l734_734121


namespace contrapositive_of_if_a2_b2_eq_zero_then_a_and_b_eq_zero_l734_734919

theorem contrapositive_of_if_a2_b2_eq_zero_then_a_and_b_eq_zero :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) :=
begin
  sorry
end

end contrapositive_of_if_a2_b2_eq_zero_then_a_and_b_eq_zero_l734_734919


namespace smallest_of_powers_l734_734114

theorem smallest_of_powers :
  (2:ℤ)^(55) < (3:ℤ)^(44) ∧ (2:ℤ)^(55) < (5:ℤ)^(33) ∧ (2:ℤ)^(55) < (6:ℤ)^(22) := by
  sorry

end smallest_of_powers_l734_734114


namespace find_ck_l734_734062

def arithmetic_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
def geometric_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
def c_seq (a_seq : ℕ → ℕ) (b_seq : ℕ → ℕ) (n : ℕ) := a_seq n + b_seq n

theorem find_ck (d r k : ℕ) (a_seq := arithmetic_seq d) (b_seq := geometric_seq r) :
  c_seq a_seq b_seq (k - 1) = 200 →
  c_seq a_seq b_seq (k + 1) = 400 →
  c_seq a_seq b_seq k = 322 :=
by
  sorry

end find_ck_l734_734062


namespace quadratic_general_form_l734_734711

theorem quadratic_general_form (x : ℝ) :
    (x + 3)^2 = x * (3 * x - 1) →
    2 * x^2 - 7 * x - 9 = 0 :=
by
  intros h
  sorry

end quadratic_general_form_l734_734711


namespace min_omega_symmetry_l734_734505

theorem min_omega_symmetry :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * x + ω * (π / 2) + π / 3) = sin ((-ω) * x + ω * (π / 2) + π / 3)) →
  ω = 1 / 3 :=
by {
  sorry
}

end min_omega_symmetry_l734_734505


namespace mirror_to_wall_area_ratio_l734_734623

theorem mirror_to_wall_area_ratio :
  let side_length := 24
  let width := 42
  let length := 27.428571428571427
  let A_mirror := side_length * side_length
  let A_wall := width * length
  (A_mirror / A_wall) = 0.5 :=
by
  let side_length := 24
  let width := 42
  let length := 27.428571428571427
  let A_mirror := side_length * side_length
  let A_wall := width * length
  have h1 : A_mirror = 576 := by sorry
  have h2 : A_wall = 1152 := by sorry
  show (A_mirror / A_wall) = 0.5
  from by
    rw [h1, h2]
    norm_num

end mirror_to_wall_area_ratio_l734_734623


namespace probability_X_greater_2_l734_734751

variable {σ : ℝ} (X : ℝ)

noncomputable def normal_distribution (x : ℝ) : ℝ := sorry

axiom normal_0_sigma : normal_distribution X = (0, σ^2)

axiom probability_condition : P (-2 ≤ X ∧ X ≤ 0) = 0.4

theorem probability_X_greater_2 : P (2 < X) = 0.1 :=
by
  sorry

end probability_X_greater_2_l734_734751


namespace term_300_is_neg_8_l734_734372

noncomputable def geom_seq (a r : ℤ) : ℕ → ℤ
| 0       => a
| (n + 1) => r * geom_seq a r n

-- First term and second term are given as conditions.
def a1 : ℤ := 8
def a2 : ℤ := -8

-- Define the common ratio based on the conditions
def r : ℤ := a2 / a1

-- Theorem stating the 300th term is -8
theorem term_300_is_neg_8 : geom_seq a1 r 299 = -8 :=
by
  have h_r : r = -1 := by
    rw [r, a2, a1]
    norm_num
  rw [h_r]
  sorry

end term_300_is_neg_8_l734_734372


namespace perimeter_of_Triangle_PXY_l734_734527

open Triangle

noncomputable def TrianglePXYPerimeter : ℝ :=
  let P := (0, 0)
  let Q := (13, 0)
  let R := (21, 20)
  let I := incenter (triangle.mk P Q R)
  let X := intersection_point (line_parallel_to I P Q) (line_through P R)
  let Y := intersection_point (line_parallel_to I P Q) (line_through Q R)
  length P X + length X Y + length Y P

theorem perimeter_of_Triangle_PXY :
  let P := (0, 0)
  let Q := (13, 0)
  let R := (21, 20)
  let I := incenter (triangle.mk P Q R)
  let X := intersection_point (line_parallel_to I P Q) (line_through P R)
  let Y := intersection_point (line_parallel_to I P Q) (line_through Q R)
  TrianglePXYPerimeter (triangle.mk P X Y) = 34 := by
  sorry

end perimeter_of_Triangle_PXY_l734_734527


namespace integer_sided_triangles_count_l734_734819

theorem integer_sided_triangles_count :
  (∃ n : ℕ, 
    n = 9 ∧ 
    (∀ (a b c : ℕ), 
      0 < a ∧ 
      0 < b ∧ 
      0 < c ∧ 
      a + b + c < 20 ∧ 
      a + b > c ∧ 
      a ≠ b ∧ 
      a ≠ c ∧ 
      b ≠ c ∧ 
      a^2 + b^2 ≠ c^2 →
      (∃ unique_abc : (ℕ × ℕ × ℕ) → Prop, unique_abc (a, b, c) ∧ triangle_in (a, b, c))))
:=
sorry

end integer_sided_triangles_count_l734_734819


namespace sqrt_product_l734_734695

theorem sqrt_product (a b c : ℝ) (h1 : a = real.sqrt 72) (h2 : b = real.sqrt 18) (h3 : c = real.sqrt 8) :
  a * b * c = 72 * real.sqrt 2 :=
by
  rw [h1, h2, h3]
  sorry

end sqrt_product_l734_734695


namespace vitya_older_than_masha_l734_734536

theorem vitya_older_than_masha :
  (∃ (days_in_month : ℕ) (total_pairs : ℕ) (favorable_pairs : ℕ)
     (p : ℚ),
    days_in_month = 30 ∧
    total_pairs = days_in_month * days_in_month ∧
    favorable_pairs = ∑ i in Finset.range(30), i ∧
    p = favorable_pairs / total_pairs ∧
    p = 29 / 60) :=
begin
  let days_in_month := 30,
  let total_pairs := days_in_month * days_in_month,
  let favorable_pairs := ∑ i in Finset.range(days_in_month), i,
  let p := favorable_pairs / total_pairs,
  use [days_in_month, total_pairs, favorable_pairs, p],
  split,
  { refl, },
  split,
  { refl, },
  split,
  { sorry, },
  split,
  { sorry, },
end

end vitya_older_than_masha_l734_734536


namespace georgina_parrot_days_l734_734747

theorem georgina_parrot_days
  (total_phrases : ℕ)
  (phrases_per_week : ℕ)
  (initial_phrases : ℕ)
  (phrases_now : total_phrases = 17)
  (teaching_rate : phrases_per_week = 2)
  (initial_known : initial_phrases = 3) :
  (49 : ℕ) = (((17 - 3) / 2) * 7) :=
by
  -- proof will be here
  sorry

end georgina_parrot_days_l734_734747


namespace cos_product_identity_l734_734018

theorem cos_product_identity (x : ℝ) (n : ℕ) : 
  (finset.range n).prod (λ k, real.cos (2^k * x)) = real.sin (2^n * x) / (2^n * real.sin x) := 
  sorry

end cos_product_identity_l734_734018


namespace negation_statement_l734_734937

variables (Students : Type) (LeftHanded InChessClub : Students → Prop)

theorem negation_statement :
  (¬ ∃ x, LeftHanded x ∧ InChessClub x) ↔ (∃ x, LeftHanded x ∧ InChessClub x) :=
by
  sorry

end negation_statement_l734_734937


namespace math_problem_l734_734262

-- Definitions
def ellipse_eq (a b : ℝ) : set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
def circle_eq (r : ℝ) : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}
def line_eq (k : ℝ) (p : ℝ × ℝ) : ℝ := k * p.1 + 1
def focal_length (a b : ℝ) : ℝ := real.sqrt(a^2 - b^2)
def eccentricity (a b : ℝ) : ℝ := real.sqrt(1 - b^2 / a^2)
def upper_vertex (a b : ℝ) : ℝ × ℝ := (0, b)
constant pi : ℝ

-- Condition definitions
def a : ℝ := real.sqrt 3
def b : ℝ := 1
def r : ℝ := 2
def c : ℝ := real.sqrt 2
def e : ℝ := real.sqrt 6 / 3
def k : ℝ := 1
def AB : ℝ := (6 * |k| * real.sqrt (k^2 + 1)) / (3 * k^2 + 1)
def EF (k : ℝ) : ℝ := 2 * real.sqrt ((4 * k^2 + 3) / (k^2 + 1))

-- Proof statement (without solution)
theorem math_problem :
  ellipse_eq a b = {p | (p.1^2 / 3) + p.2^2 = 1} ∧
  (|AB| * |EF k| = 3 * real.sqrt 7) →
  k = 1 →
  line_eq k (0, b) = b + 1 ∧
  ∃ F2 : ℝ × ℝ, let d' := (real.sqrt 2 + 1) / real.sqrt 2 in
  let S := 0.5 * ∃t, AB * d' in
  S = (3 * (real.sqrt 2 + 1)) / 4
:= sorry

end math_problem_l734_734262


namespace minimum_omega_l734_734498

theorem minimum_omega {ω : ℝ} (hω : ω > 0)
    (symmetry : ∃ k : ℤ, ∀ x : ℝ, 
      (sin (ω * x + ω * π / 2 + π / 3) = sin (-ω * x + ω * π / 2 + π / 3))) 
    : ω = 1 / 3 :=
by
  sorry

end minimum_omega_l734_734498


namespace sufficient_but_not_necessary_pi_l734_734944

theorem sufficient_but_not_necessary_pi (x : ℝ) : 
  (x = Real.pi → Real.sin x = 0) ∧ (Real.sin x = 0 → ∃ k : ℤ, x = k * Real.pi) → ¬(Real.sin x = 0 → x = Real.pi) :=
by
  sorry

end sufficient_but_not_necessary_pi_l734_734944


namespace count_lineup_excluding_youngest_l734_734353

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l734_734353


namespace proposition_one_proposition_two_l734_734782

-- Define the function and its derivative on the real numbers
variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Condition: Function f is both defined and differentiable on R
def differentiable_on_R : Prop :=
∀ x : ℝ, Differentiable ℝ f

-- Proposition ①: If f is an odd function, then its derivative f' is an even function.
theorem proposition_one (h_diff : differentiable_on_R) (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  ∀ x : ℝ, f' x = f' (-x) :=
by
  sorry

-- Proposition ②: If f is a strictly increasing function, then f' is a strictly increasing function.
theorem proposition_two (h_diff : differentiable_on_R) (h_increasing : ∀ x y : ℝ, x < y → f x < f y) :
  ¬ (∀ x y : ℝ, x < y → f' x < f' y) :=
by
  sorry

end proposition_one_proposition_two_l734_734782


namespace doubled_volume_l734_734629

theorem doubled_volume (V₀ V₁ : ℝ) (hV₀ : V₀ = 3) (h_double : V₁ = V₀ * 8) : V₁ = 24 :=
by 
  rw [hV₀] at h_double
  exact h_double

end doubled_volume_l734_734629


namespace charlyn_visible_area_l734_734172

noncomputable def visible_area (length width sight_distance : ℝ) : ℝ :=
  let interior_length := length - 2 * sight_distance
  let interior_width := width - 2 * sight_distance
  let interior_area := interior_length * interior_width
  let horizontal_strip_area := 2 * (length * sight_distance)
  let vertical_strip_area := 2 * (width * sight_distance)
  let circle_area := π * (sight_distance ^ 2)
  interior_area + horizontal_strip_area + vertical_strip_area + circle_area / 2

theorem charlyn_visible_area :
  visible_area 8 4 1.5 ≈ 48 :=
by
  sorry

end charlyn_visible_area_l734_734172


namespace dihedral_angle_measure_l734_734843

-- Cube and its vertices defined
structure Cube (α : Type) [LinearOrder α] :=
  (A B C D A₁ B₁ C₁ D₁ : α)

-- Define the side length of the cube
def side_length {α : Type} [LinearOrder α] (c : Cube α) : α := 1

-- Perpendicular projection and relevant congruent triangles
theorem dihedral_angle_measure (c : Cube Real)
  (h1 : side_length c = 1)
  (h2 : is_congruent (triangle c.A₁ c.D c.C) (triangle c.A₁ c.B c.C))
  (h3 : perpendicular_to c.D c.A₁ c.C) :
  dihedral_angle (plane c.B c.A₁ c.C) (plane c.A₁ c.C c.D) = 120 := sorry

end dihedral_angle_measure_l734_734843


namespace divisor_of_7_l734_734418

theorem divisor_of_7 (a n : ℤ) (h1 : a ≥ 1) (h2 : a ∣ (n + 2)) (h3 : a ∣ (n^2 + n + 5)) : a = 1 ∨ a = 7 :=
by
  sorry

end divisor_of_7_l734_734418


namespace number_of_valid_permutations_l734_734328

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l734_734328


namespace total_wall_area_l734_734569

variable (L W : ℝ) -- Length and width of the regular tile
variable (R : ℕ) -- Number of regular tiles

-- Conditions:
-- 1. The area covered by regular tiles is 70 square feet.
axiom regular_tiles_cover_area : R * (L * W) = 70

-- 2. Jumbo tiles make up 1/3 of the total tiles, and each jumbo tile has an area three times that of a regular tile.
axiom length_ratio : ∀ jumbo_tiles, 3 * (jumbo_tiles * (L * W)) = 105

theorem total_wall_area (L W : ℝ) (R : ℕ) 
  (regular_tiles_cover_area : R * (L * W) = 70) 
  (length_ratio : ∀ jumbo_tiles, 3 * (jumbo_tiles * (L * W)) = 105) : 
  (R * (L * W)) + (3 * (R / 2) * (L * W)) = 175 :=
by
  sorry

end total_wall_area_l734_734569


namespace proof_inequality_l734_734757

noncomputable def problem_statement (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : Prop :=
  a + b + c ≤ (a ^ 4 + b ^ 4 + c ^ 4) / (a * b * c)

theorem proof_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  problem_statement a b c h_a h_b h_c :=
by
  sorry

end proof_inequality_l734_734757


namespace sqrt_simplification_proof_l734_734029

noncomputable def sqrt_simplification : Real := Real.sqrt (Real.cbrt (Real.sqrt (1 / 32768)))

theorem sqrt_simplification_proof : sqrt_simplification = 1 / (2 * Real.sqrt 2) := 
by
  -- we will assume that the necessary simplifications steps can be properly carried out in Lean
  sorry

end sqrt_simplification_proof_l734_734029


namespace seq_b_arithmetic_diff_seq_a_general_term_l734_734272

variable {n : ℕ}

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n / (a n + 2)

def seq_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 1 / a n

theorem seq_b_arithmetic_diff (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_a : seq_a a) (h_b : seq_b a b) :
  ∀ n, b (n + 1) - b n = 1 / 2 :=
by
  sorry

theorem seq_a_general_term (a : ℕ → ℝ) (h_a : seq_a a) :
  ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end seq_b_arithmetic_diff_seq_a_general_term_l734_734272


namespace masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l734_734003

/-- 
Part (a): Define the problem statement where, given the iterative process on a number,
it stabilizes at 17.
-/
theorem masha_final_number_stabilizes (x y : ℕ) (n : ℕ) (h_stable : ∀ x y, 10 * x + y = 3 * x + 2 * y) :
  n = 17 :=
by
  sorry

/--
Part (b): Define the problem statement to find the smallest 2015-digit number ending with the
digits 09 that eventually stabilizes to 17.
-/
theorem masha_smallest_initial_number_ends_with_09 :
  ∃ (n : ℕ), n ≥ 10^2014 ∧ n % 100 = 9 ∧ (∃ k : ℕ, 10^2014 + k = n ∧ (10 * ((n - k) / 10) + (n % 10)) = 17) :=
by
  sorry

end masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l734_734003


namespace second_child_sweets_l734_734614

theorem second_child_sweets :
  (mother_kept : ℕ) (children_sweets : ℕ) (eldest_got : ℕ) (youngest_got : ℕ) (second_child_got : ℕ) 
  (h1 : mother_kept = 27 / 3)
  (h2 : children_sweets = 27 - mother_kept)
  (h3 : eldest_got = 8)
  (h4 : youngest_got = eldest_got / 2)
  (h5 : second_child_got = children_sweets - eldest_got - youngest_got) :
  second_child_got = 6 :=
by
  sorry

end second_child_sweets_l734_734614


namespace xiao_ying_correct_answers_at_least_l734_734832

def total_questions : ℕ := 20
def points_correct : ℕ := 5
def points_incorrect : ℕ := 2
def excellent_points : ℕ := 80

theorem xiao_ying_correct_answers_at_least (x : ℕ) :
  (5 * x - 2 * (total_questions - x)) ≥ excellent_points → x ≥ 18 := by
  sorry

end xiao_ying_correct_answers_at_least_l734_734832


namespace mutually_exclusive_necessary_but_not_sufficient_for_complementary_l734_734230

axiom events (Ω : Type) : Type
axiom A₁ A₂ : events Ω
axiom mutually_exclusive (A₁ A₂ : events Ω) : Prop :=
  ∀ ω, (A₁ ω) ∧ (A₂ ω) → false
axiom complementary (A₁ A₂ : events Ω) : Prop :=
  ∀ ω, (A₁ ω) ↔ ¬ (A₂ ω)

theorem mutually_exclusive_necessary_but_not_sufficient_for_complementary
  (hA : mutually_exclusive A₁ A₂) :
  (∀ ω, (complementary A₁ A₂) → (mutually_exclusive A₁ A₂)) ∧
  ¬ (∀ ω, (mutually_exclusive A₁ A₂) → (complementary A₁ A₂)) :=
begin
  sorry
end

end mutually_exclusive_necessary_but_not_sufficient_for_complementary_l734_734230


namespace find_distance_from_B_to_center_l734_734894

-- Define points in a Euclidean space
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define a function to calculate the Euclidean distance between two points
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Declare that our function uses real numbers
noncomputable def OB_distance (A B C O : Point) (r : ℝ) :=
  ∃ (d : ℝ), (distance A O = r) ∧ (distance C O = r) ∧ (distance A B = 6) ∧ (distance B C = 2) ∧
  (angle A B C = π / 2) ∧ (distance B O = d) ∧ (d = real.sqrt 26)

-- Now writing the proposition in Lean
theorem find_distance_from_B_to_center (A B C O : Point) :
  ∀ r : ℝ, 
  (distance A O = r) → (distance C O = r) → (distance A B = 6) → (distance B C = 2) → 
  (angle A B C = π / 2) →
  (distance B O = real.sqrt 26) :=
sorry

end find_distance_from_B_to_center_l734_734894


namespace angles_cosine_sum_l734_734035

theorem angles_cosine_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1)
  (h2 : Real.cos A + Real.cos B = 0) :
  12 * Real.cos (2 * A) + 4 * Real.cos (2 * B) = 8 :=
sorry

end angles_cosine_sum_l734_734035


namespace mishaPhoneNumber_l734_734993

-- Given definitions based on the conditions
def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s == s.reverse

def consecutiveDigits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ (digits.get 1 = digits.head.succ ∧ digits.get 1 = digits.last.pred)

def divisibleBy9 (n : ℕ) : Prop :=
  n % 9 = 0

def containsThreeConsecutiveOnes (n : ℕ) : Prop :=
  "111".isSubstringOf n.toString

def onlyOnePrime (a b : ℕ) : Prop :=
  (Nat.Prime a ∧ ¬Nat.Prime b) ∨ (¬Nat.Prime a ∧ Nat.Prime b)

-- The final theorem statement
theorem mishaPhoneNumber : Σ (n : ℕ), 1000000 ≤ n ∧ n < 10000000 ∧
  isPalindrome (n / 100) ∧ 
  consecutiveDigits (n % 1000) ∧ 
  divisibleBy9 (n / 10000) ∧ 
  containsThreeConsecutiveOnes n ∧ 
  onlyOnePrime (n / 10 % 100) (n % 100) :=
⟨7111765, ⟨rfl, (Nat.le_refl 1000000), rfl, (Nat.lt_succ 10000000),
 sorry, sorry, sorry, sorry⟩⟩

end mishaPhoneNumber_l734_734993


namespace units_digits_of_squares_l734_734178

theorem units_digits_of_squares (d : Fin 10) :
  d^2 % 10 ∈ {0, 1, 4, 5, 6, 9} := by
  sorry

end units_digits_of_squares_l734_734178


namespace total_goals_l734_734116

theorem total_goals (Bruce_goals : ℕ) (Michael_goals : ℕ) (total_goals : ℕ) 
  (h1 : Bruce_goals = 4)
  (h2 : Michael_goals = 3 * Bruce_goals) : total_goals = 16 := 
by 
  have h3 : Michael_goals = 3 * 4 := by rwa [h1] at h2
  have h4 : total_goals = 4 + Michael_goals := by rwa [h1]
  have h5 : total_goals = 4 + 12 := by rwa [h3] at h4
  rw h5
  exact rfl

end total_goals_l734_734116


namespace max_geometric_mean_of_six_numbers_l734_734449

-- Define the six numbers in terms of given conditions
variables {a1 a2 a3 a4 a5 a6 A : ℝ}

-- Given conditions as definitions
def one_unit_exists : Prop := a3 = 1
def equal_arithmetic_mean (a b c d e f : ℝ) : Prop :=
  (a + b + c) / 3 = (b + c + d) / 3 ∧ (b + c + d) / 3 = (c + d + e) / 3 ∧ (c + d + e) / 3 = (d + e + f) / 3
def arithmetic_mean_A (a b c d e f A : ℝ) : Prop := (a + b + c + d + e + f) / 6 = A

-- Translate to summarizing the geometric mean of any three consecutive numbers
noncomputable def geometric_mean_max (a b c : ℝ) : ℝ := (a * b * c)^(1/3)

-- The theorem to prove the maximum value of the geometric mean given the conditions
theorem max_geometric_mean_of_six_numbers 
  {a1 a2 a3 a4 a5 a6 A : ℝ}
  (h1 : one_unit_exists)
  (h2 : equal_arithmetic_mean a1 a2 a3 a4 a5 a6)
  (h3 : arithmetic_mean_A a1 a2 a3 a4 a5 a6 A) :
  geometric_mean_max a4 a5 a6 ≤ (3 * A - 1)^2 / 4 ^ (1 / 3) :=
  sorry

end max_geometric_mean_of_six_numbers_l734_734449


namespace count_valid_arrangements_l734_734324

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l734_734324


namespace length_of_PQ_area_of_trapezoid_APRO_l734_734708

section GeometryProblem

variables {A B C D P Q R O : Point}
variables {square : Quadrilateral}
variables {side_length : ℝ}
variables {AP_length CQ_length OP_length OQ_length OR_length : ℝ}

-- Defining the conditions
def is_square (quad : Quadrilateral) : Prop :=
  is_square quad

def side_length_is_2 (s : ℝ) : Prop :=
  s = 2

def AP_is_1 (P : Point) (A : Point) : Prop :=
  dist A P = 1

def CQ_is_1_5 (Q : Point) (C : Point) : Prop :=
  dist C Q = 1.5

def center_O (O : Point) (s : Quadrilateral) : Prop :=
  center_of_square O s

def OP_is_1 (P, O : Point) : Prop :=
  dist O P = 1

def OQ_is_1 (Q, O : Point) : Prop :=
  dist O Q = 1

def OR_is_1 (R, O : Point) : Prop :=
  dist O R = 1

-- Proof statement for the length of PQ
theorem length_of_PQ 
    (h1 : is_square square) 
    (h2 : side_length_is_2 side_length)
    (h3 : AP_is_1 P A)
    (h4 : CQ_is_1_5 Q C)
    (h5 : center_O O square)
    (h6 : OP_is_1 P O)
    (h7 : OQ_is_1 Q O)
    (h8 : OR_is_1 R O) :
    dist P Q = 0.5 :=
sorry

-- Proof statement for the area of trapezoid APRO
theorem area_of_trapezoid_APRO 
    (h1 : is_square square) 
    (h2 : side_length_is_2 side_length)
    (h3 : AP_is_1 P A)
    (h4 : CQ_is_1_5 Q C)
    (h5 : center_O O square)
    (h6 : OP_is_1 P O)
    (h7 : OQ_is_1 Q O)
    (h8 : OR_is_1 R O) :
    area (trapezoid A P R O) = 1 :=
sorry

end GeometryProblem

end length_of_PQ_area_of_trapezoid_APRO_l734_734708


namespace prod_97_103_l734_734703

theorem prod_97_103 : (97 * 103) = 9991 := 
by 
  have h1 : 97 = 100 - 3 := by rfl
  have h2 : 103 = 100 + 3 := by rfl
  calc
    97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
         ... = 100^2 - 3^2 : by rw (mul_sub (100:ℤ) 3 3)
         ... = 10000 - 9 : by norm_num
         ... = 9991 : by norm_num
 
end prod_97_103_l734_734703


namespace bill_drew_12_triangles_l734_734165

theorem bill_drew_12_triangles 
  (T : ℕ)
  (total_lines : T * 3 + 8 * 4 + 4 * 5 = 88) : 
  T = 12 :=
sorry

end bill_drew_12_triangles_l734_734165


namespace rhombus_area_4_sqrt_2_l734_734962

theorem rhombus_area_4_sqrt_2 :
  ∀ (a : ℝ) (θ : ℝ),
    a = 4 →
    θ = 45 →
    let s := a in
    let h := s * (1 / Real.sqrt 2) in
    let area := (1 / 2) * s * h in
    area = 4 * Real.sqrt 2 :=
by
  intros a θ ha hθ
  let s := a
  let h := s * (1 / Real.sqrt 2)
  let area := (1 / 2) * s * h
  rw [ha, hθ]
  have sqrt2_nonneg : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  norm_num at area
  sorry

end rhombus_area_4_sqrt_2_l734_734962


namespace fraction_addition_l734_734169

theorem fraction_addition (x : ℝ) (h : x + 1 ≠ 0) : (x / (x + 1) + 1 / (x + 1) = 1) :=
sorry

end fraction_addition_l734_734169


namespace final_result_l734_734547

-- define the initial matrix M
def M : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![53, 158, 53],
    ![23, 93, 53],
    ![50, 170, 53]
  ]

-- define the equivalence conditions
def cond_1 : 53 % 2 = 1 := by norm_num
def cond_2 : 53 % 3 = 2 := by norm_num
def cond_3 : 53 % 5 = 3 := by norm_num
def cond_4 : 53 % 7 = 4 := by norm_num

-- define the equivalence of Z_210 with Z_2 × Z_3 × Z_5 × Z_7
def Z210_eq : ZMod 210 ≃ ZMod 2 × ZMod 3 × ZMod 5 × ZMod 7 := sorry

-- Combining modulo operations.
def M_mod_2 : Matrix (Fin 3) (Fin 3) (ZMod 2) := M.map (fun x => x % 2)
def M_mod_3 : Matrix (Fin 3) (Fin 3) (ZMod 3) := M.map (fun x => x % 3)
def M_mod_5 : Matrix (Fin 3) (Fin 3) (ZMod 5) := M.map (fun x => x % 5)
def M_mod_7 : Matrix (Fin 3) (Fin 3) (ZMod 7) := M.map (fun x => x % 7)

-- prove that the final result is 1234
theorem final_result : (M_mod_2, M_mod_3, M_mod_5, M_mod_7) = 1234 := sorry

end final_result_l734_734547


namespace find_number_l734_734890

theorem find_number (x : ℝ) (h : (4 / 3) * x = 48) : x = 36 :=
sorry

end find_number_l734_734890


namespace perimeter_of_ABCDEFG_l734_734526

-- Definitions: Points and distances
variables {A B C D E F G : Type}

-- Given conditions
def is_equilateral (T : Type) (a b c : T) := sorry
def distance (p q : Type) := sorry

axiom ABC_equilateral : is_equilateral Triangle A B C
axiom ADE_equilateral : is_equilateral Triangle A D E
axiom EFG_equilateral : is_equilateral Triangle E F G

axiom D_on_AC : distance A D = (1/3) * distance A C
axiom G_midpoint_AE : distance A G = (1/2) * distance A E
axiom AB_equals_6 : distance A B = 6

-- Goal: Prove the perimeter of the figure $ABCDEFG$ equals 21
theorem perimeter_of_ABCDEFG : 
  distance A B + distance B C + distance C D + distance D E + distance E F + distance F G + distance G A = 21 := 
by
  sorry

end perimeter_of_ABCDEFG_l734_734526


namespace vasya_fraction_l734_734668

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l734_734668


namespace georgina_parrot_days_l734_734749

theorem georgina_parrot_days 
    (initial_phrases : ℕ := 3) 
    (current_phrases : ℕ := 17) 
    (phrases_per_week : ℕ := 2) 
    (days_per_week : ℕ := 7) 
    : nat :=
  let new_phrases := current_phrases - initial_phrases in
  let weeks := new_phrases / phrases_per_week in
  let days := weeks * days_per_week in
  days = 49
by
  sorry

end georgina_parrot_days_l734_734749


namespace trigonometric_identity_l734_734027

theorem trigonometric_identity (x y : ℝ) : 
  cos x ^ 2 + cos (x - y) ^ 2 - 2 * cos x * cos y * cos (x - y) = cos x ^ 2 + sin y ^ 2 :=
by 
  sorry  -- Proof to be provided

end trigonometric_identity_l734_734027


namespace alternating_sum_10002_l734_734560

def alternating_sum (n : ℕ) : ℤ :=
  (List.range (n+1)).map (λ k, if even k then (k : ℤ) else -(k : ℤ)).sum

theorem alternating_sum_10002 : alternating_sum 10002 = -5001 := by
  sorry

end alternating_sum_10002_l734_734560


namespace wall_width_l734_734601

theorem wall_width (brick_length brick_height brick_depth : ℝ)
    (wall_length wall_height : ℝ)
    (num_bricks : ℝ)
    (total_bricks_volume : ℝ)
    (total_wall_volume : ℝ) :
    brick_length = 25 →
    brick_height = 11.25 →
    brick_depth = 6 →
    wall_length = 800 →
    wall_height = 600 →
    num_bricks = 6400 →
    total_bricks_volume = num_bricks * (brick_length * brick_height * brick_depth) →
    total_wall_volume = wall_length * wall_height * (total_bricks_volume / (brick_length * brick_height * brick_depth)) →
    (total_bricks_volume / (wall_length * wall_height) = 22.5) :=
by
  intros
  sorry -- proof not required

end wall_width_l734_734601


namespace unique_solution_l734_734727

theorem unique_solution (p : ℕ) (a b n : ℕ) : 
  p.Prime → 2^a + p^b = n^(p-1) → (p, a, b, n) = (3, 0, 1, 2) ∨ (p = 2) :=
by {
  sorry
}

end unique_solution_l734_734727


namespace zoo_individuals_remaining_l734_734606

noncomputable def initial_students_class1 := 10
noncomputable def initial_students_class2 := 10
noncomputable def chaperones := 5
noncomputable def teachers := 2
noncomputable def students_left := 10
noncomputable def chaperones_left := 2

theorem zoo_individuals_remaining :
  let total_initial_individuals := initial_students_class1 + initial_students_class2 + chaperones + teachers
  let total_left := students_left + chaperones_left
  total_initial_individuals - total_left = 15 := by
  sorry

end zoo_individuals_remaining_l734_734606


namespace factorial_expression_l734_734684

theorem factorial_expression : 8! - 7 * 7! - 2 * 6! = 3600 := by
  sorry

end factorial_expression_l734_734684


namespace coin_flip_probability_l734_734983

theorem coin_flip_probability :
  let p_tails := (1 : ℚ) / 2 in
  let p_heads := 1 - p_tails in
  p_tails * p_tails * p_heads * p_heads = 1 / 16 :=
by sorry

end coin_flip_probability_l734_734983


namespace sufficient_but_not_necessary_l734_734466

noncomputable def φ := Real.pi / 2

def f (x : ℝ) : ℝ := Real.cos x
def g (x : ℝ) (φ : ℝ) : ℝ := Real.sin (x + φ)

theorem sufficient_but_not_necessary (x : ℝ) (hφ : φ = Real.pi / 2) : 
  g x φ = f x ∧ (∃ k : ℤ, φ ≠ 2 * k * Real.pi + Real.pi / 2) := 
by
  sorry

end sufficient_but_not_necessary_l734_734466


namespace vasya_fraction_l734_734669

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l734_734669


namespace three_digit_number_divisible_by_8_and_even_tens_digit_l734_734561

theorem three_digit_number_divisible_by_8_and_even_tens_digit (d : ℕ) (hd : d % 2 = 0) (hdiv : (100 * 5 + 10 * d + 4) % 8 = 0) :
  100 * 5 + 10 * d + 4 = 544 :=
by
  sorry

end three_digit_number_divisible_by_8_and_even_tens_digit_l734_734561


namespace roots_squared_sum_l734_734982

noncomputable def roots_of_quadratic (p q : ℝ) : set ℝ :=
  { x | x^2 + p * x + q = 0 }

theorem roots_squared_sum (a b : ℝ) (h : a ∈ roots_of_quadratic (-8) 8) 
  (h2 : b ∈ roots_of_quadratic (-8) 8) (hab : a ≠ b): a^2 + b^2 = 48 :=
by
  sorry

end roots_squared_sum_l734_734982


namespace negation_of_sine_proposition_l734_734896

theorem negation_of_sine_proposition :
  ¬ (∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 :=
by
-- Proof omitted
sorry

end negation_of_sine_proposition_l734_734896


namespace length_of_bridge_l734_734125

theorem length_of_bridge : 
  ∀ (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_sec : ℝ), 
  train_length = 170 → 
  train_speed_kmph = 45 → 
  crossing_time_sec = 30 → 
  let train_speed_mps := train_speed_kmph * (1000 / 3600) in
  let distance_traveled := train_speed_mps * crossing_time_sec in
  let bridge_length := distance_traveled - train_length in
  bridge_length = 205 :=
by
  intros train_length train_speed_kmph crossing_time_sec h1 h2 h3
  simp [h1, h2, h3]
  let train_speed_mps := 45 * (1000 / 3600)
  let distance_traveled := train_speed_mps * 30
  let bridge_length := distance_traveled - 170
  show bridge_length = 205
  sorry

end length_of_bridge_l734_734125


namespace eighth_root_of_unity_l734_734717

theorem eighth_root_of_unity (h : (∀ x, x = tan (π / 8))) :
  (tan (π / 8) + Complex.i) / (tan (π / 8) - Complex.i) = Complex.exp (Complex.i * (π / 4)) :=
by
  sorry

end eighth_root_of_unity_l734_734717


namespace total_cost_sandwiches_sodas_l734_734679

theorem total_cost_sandwiches_sodas (cost_per_sandwich cost_per_soda : ℝ) 
  (num_sandwiches num_sodas : ℕ) (discount_rate : ℝ) (total_items : ℕ) :
  cost_per_sandwich = 4 → 
  cost_per_soda = 3 → 
  num_sandwiches = 6 → 
  num_sodas = 7 → 
  discount_rate = 0.10 → 
  total_items = num_sandwiches + num_sodas → 
  total_items > 10 → 
  (num_sandwiches * cost_per_sandwich + num_sodas * cost_per_soda) * (1 - discount_rate) = 40.5 :=
by
  intros
  sorry

end total_cost_sandwiches_sodas_l734_734679


namespace walnuts_amount_l734_734391

theorem walnuts_amount (w : ℝ) (total_nuts : ℝ) (almonds : ℝ) (h1 : total_nuts = 0.5) (h2 : almonds = 0.25) (h3 : w + almonds = total_nuts) : w = 0.25 :=
by
  sorry

end walnuts_amount_l734_734391


namespace basic_astrophysics_degrees_l734_734135

theorem basic_astrophysics_degrees 
    (p1 : ℝ := 10) (p2 : ℝ := 24) (p3 : ℝ := 15) (p4 : ℝ := 29) (p5 : ℝ := 8) : 
    (100 - (p1 + p2 + p3 + p4 + p5)) * 360 / 100 = 50.4 := 
by
  -- Define the percentages
  let total_percentage := p1 + p2 + p3 + p4 + p5
  -- Calculate the percentage for basic astrophysics
  let basic_astrophysics_percentage := 100 - total_percentage
  -- Calculate the degrees representing basic astrophysics
  let degrees := basic_astrophysics_percentage * 360 / 100
  -- Prove the theorem
  have h : degrees = 50.4 := rfl
  exact h

end basic_astrophysics_degrees_l734_734135


namespace doubled_dimensions_volume_l734_734625

theorem doubled_dimensions_volume (original_volume : ℝ) (length_factor width_factor height_factor : ℝ) 
  (h : original_volume = 3) 
  (hl : length_factor = 2)
  (hw : width_factor = 2)
  (hh : height_factor = 2) : 
  original_volume * length_factor * width_factor * height_factor = 24 :=
by
  sorry

end doubled_dimensions_volume_l734_734625


namespace find_number_l734_734295

theorem find_number (a : ℕ) (h : a = 105) : 
  a^3 / (49 * 45 * 25) = 21 :=
by
  sorry

end find_number_l734_734295


namespace five_people_lineup_count_l734_734339

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l734_734339


namespace line_up_ways_l734_734355

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l734_734355


namespace sequence_formula_l734_734235

theorem sequence_formula (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a n ≠ 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (n + 1 + 1 / (a n))) :
  ∀ n : ℕ, n > 0 → a n = 2 / ((n : ℝ) ^ 2 - n + 2) :=
by
  sorry

end sequence_formula_l734_734235


namespace find_sample_size_l734_734833

-- Definitions based on conditions
def ratio_students : ℕ := 2 + 3 + 5
def grade12_ratio : ℚ := 5 / ratio_students
def sample_grade12_students : ℕ := 150

-- The goal is to find n such that the proportion is maintained
theorem find_sample_size (n : ℕ) (h : grade12_ratio = sample_grade12_students / ↑n) : n = 300 :=
by sorry


end find_sample_size_l734_734833


namespace minimum_omega_for_symmetric_curve_l734_734475

theorem minimum_omega_for_symmetric_curve (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 2) + π / 3) = sin (-ω * (x + π / 2) + π / 3)) ↔ ω = 1 / 3 :=
by
  sorry

end minimum_omega_for_symmetric_curve_l734_734475


namespace second_concert_attendance_correct_l734_734882

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119
def second_concert_attendance : ℕ := 66018

theorem second_concert_attendance_correct :
  first_concert_attendance + additional_people = second_concert_attendance :=
by sorry

end second_concert_attendance_correct_l734_734882


namespace product_fractions_product_l734_734719

theorem product_fractions_product : 
  (\prod_{n=2}^{100} (1 - (1 / n))) = (1 / 100) := 
by
  sorry

end product_fractions_product_l734_734719


namespace find_x_value_l734_734365

theorem find_x_value (x : ℝ) (h : 150 + 90 + x + 90 = 360) : x = 30 := by
  sorry

end find_x_value_l734_734365


namespace find_BD_over_BO_l734_734436

open real

noncomputable def ratio_BD_BO (A B C O D : Point) : ℝ :=
  if h : circle_centered_at O A ∧ circle_centered_at O C ∧ tangent B A O ∧ tangent B C O ∧ isosceles_triangle A B C 50 ∧ intersects BO D then
    1 - sin (40 * pi / 180)
  else
    0

theorem find_BD_over_BO (A B C O D : Point) :
  (circle_centered_at O A ∧ circle_centered_at O C ∧ tangent B A O ∧ tangent B C O ∧ isosceles_triangle A B C 50 ∧ intersects BO D) →
  ratio_BD_BO A B C O D = 1 - sin (40 * pi / 180) :=
by
  intro h
  simp [ratio_BD_BO, h]
  sorry

end find_BD_over_BO_l734_734436


namespace triangle_angle_proof_l734_734381

variables (A B C P : Type) [Triangle A B C] [Point P]
variables (BP PC : P = point (midpoint B C)) 
variables (AP_eq_PC : AP = PC) (AB_eq_AC : AB = AC)
variables (angle_APC : MeasureAngle A P C = 2 * x) 
variables (angle_APB : MeasureAngle A P B = 3 * x)

theorem triangle_angle_proof 
: MeasureAngle B A C = (180 / 7) := sorry

end triangle_angle_proof_l734_734381


namespace omega_min_value_l734_734483

theorem omega_min_value (ω : ℝ) (hω : ω > 0)
    (hSymmetry : ∀ x : ℝ, sin (ω * x + ω * π / 2 + π / 3) = sin (ω * -x + ω * π / 2 + π / 3)) :
    ω = 1 / 3 :=
begin
  sorry
end

end omega_min_value_l734_734483


namespace tangent_slope_at_point_l734_734739

theorem tangent_slope_at_point (x : ℝ) (h : x = 2) : 
  (deriv (λ x:ℝ, (1/5:ℝ) * x^2) x) = (4/5:ℝ) :=
by sorry

end tangent_slope_at_point_l734_734739


namespace tangent_line_at_one_l734_734203

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_one : ∀ x y, (x = 1 ∧ y = 0) → (x - y - 1 = 0) :=
by 
  intro x y h
  sorry

end tangent_line_at_one_l734_734203


namespace evaluate_expression_l734_734196

open Nat

theorem evaluate_expression : 
  (3 * 4 * 5 * 6) * (1 / 3 + 1 / 4 + 1 / 5 + 1 / 6) = 342 := by
  sorry

end evaluate_expression_l734_734196


namespace binom_prob_X_eq_3_l734_734763

noncomputable def binom_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

def X : ℕ → ℚ → ℕ → Prop := λ n p k, p = 1/2 ∧ n = 6 ∧ k = 3

theorem binom_prob_X_eq_3 :
  ∀ n k : ℕ, ∀ p : ℚ,
  X n p k →
  binom_pmf n k p = 5 / 16 :=
by
  intros n k p h
  rcases h with ⟨hp, hn, hk⟩
  subst hp
  subst hn
  subst hk
  sorry

end binom_prob_X_eq_3_l734_734763


namespace cannot_determine_right_triangle_l734_734808

-- Define what a right triangle is
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the conditions
def condition_A (A B C : ℕ) : Prop :=
  A / B = 3 / 4 ∧ A / C = 3 / 5 ∧ B / C = 4 / 5

def condition_B (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13

def condition_C (A B C : ℕ) : Prop :=
  A - B = C

def condition_D (a b c : ℕ) : Prop :=
  a^2 = b^2 - c^2

-- Define the problem in Lean
theorem cannot_determine_right_triangle :
  (∃ A B C, condition_A A B C → ¬is_right_triangle A B C) ∧
  (∀ (a b c : ℕ), condition_B a b c → is_right_triangle a b c) ∧
  (∀ A B C, condition_C A B C → A = 90) ∧
  (∀ (a b c : ℕ),  condition_D a b c → is_right_triangle a b c)
:=
by sorry

end cannot_determine_right_triangle_l734_734808


namespace count_valid_arrangements_l734_734320

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l734_734320


namespace exists_real_c_for_infinitely_many_primes_l734_734995

noncomputable def fractional_part (x : ℝ) : ℝ := x - Real.floor x

def a_p (p : ℕ) (n : ℕ) [Fact (Nat.Prime p)] [Fact (Odd p)] [Fact (Odd n)] : ℝ :=
  (1 / (p-1:ℝ)) * (∑ k in Finset.range ((p-1) / 2 + 1), fractional_part ((k^ (2:nat*n): ℝ)/ p))

theorem exists_real_c_for_infinitely_many_primes (n : ℕ) [Fact (Odd n)] :
  ∃ (c : ℝ), (∀ᶠ p in Filter.atTop, Nat.Prime p → a_p p n = c) := by
  use 1 / 2
  sorry

end exists_real_c_for_infinitely_many_primes_l734_734995


namespace average_jump_of_winner_l734_734529

-- Define the jump distances for the athletes
def athlete1_long_jump : ℕ := 26
def athlete1_triple_jump : ℕ := 30
def athlete1_high_jump : ℕ := 7

def athlete2_long_jump : ℕ := 24
def athlete2_triple_jump : ℕ := 34
def athlete2_high_jump : ℕ := 8

-- Define the total jumps for each athlete
def athlete1_total_jump : ℕ := athlete1_long_jump + athlete1_triple_jump + athlete1_high_jump
def athlete2_total_jump : ℕ := athlete2_long_jump + athlete2_triple_jump + athlete2_high_jump

-- Define the average jumps for each athlete
def athlete1_avg_jump : ℕ := athlete1_total_jump / 3
def athlete2_avg_jump : ℕ := athlete2_total_jump / 3

-- The theorem statement
theorem average_jump_of_winner : athlete2_avg_jump = 22 :=
by
  noncomputable def athlete1_long_jump := 26
  noncomputable def athlete1_triple_jump := 30
  noncomputable def athlete1_high_jump := 7
  
  noncomputable def athlete2_long_jump := 24
  noncomputable def athlete2_triple_jump := 34
  noncomputable def athlete2_high_jump := 8
  
  noncomputable def athlete1_total_jump := athlete1_long_jump + athlete1_triple_jump + athlete1_high_jump
  noncomputable def athlete2_total_jump := athlete2_long_jump + athlete2_triple_jump + athlete2_high_jump
  
  noncomputable def athlete1_avg_jump := athlete1_total_jump / 3
  noncomputable def athlete2_avg_jump := athlete2_total_jump / 3
  sorry

end average_jump_of_winner_l734_734529


namespace exponent_proof_l734_734580

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l734_734580


namespace dot_product_value_l734_734810

-- Define vectors a and b, and the condition of their linear combination
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def a : Vector2D := ⟨-1, 2⟩
def b (m : ℝ) : Vector2D := ⟨m, 1⟩

-- Define the condition that vector a + 2b is parallel to 2a - b
def parallel (v w : Vector2D) : Prop := ∃ k : ℝ, v.x = k * w.x ∧ v.y = k * w.y

def vector_add (v w : Vector2D) : Vector2D := ⟨v.x + w.x, v.y + w.y⟩
def scalar_mul (c : ℝ) (v : Vector2D) : Vector2D := ⟨c * v.x, c * v.y⟩

-- Dot product definition
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

-- The theorem to prove
theorem dot_product_value (m : ℝ)
  (h : parallel (vector_add a (scalar_mul 2 (b m))) (vector_add (scalar_mul 2 a) (scalar_mul (-1) (b m)))) :
  dot_product a (b m) = 5 / 2 :=
sorry

end dot_product_value_l734_734810


namespace count_restricted_arrangements_l734_734346

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l734_734346


namespace regular_price_proof_l734_734509

noncomputable def regular_price_of_tire (cost_paid : ℝ) : ℝ :=
  let x := 295 / 3 in x

theorem regular_price_proof (h1 : ∀x : ℝ, cost_paid = 3 * x + 5 - 10)
  (cost_paid : ℝ) (h2 : cost_paid = 290) : regular_price_of_tire cost_paid = 98.33 :=
by
  sorry

end regular_price_proof_l734_734509


namespace calcium_oxide_moles_l734_734548

theorem calcium_oxide_moles
    (atomic_weight_Ca : ℝ)
    (atomic_weight_O : ℝ)
    (total_weight_CaO : ℝ)
    (molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O)
    (number_of_moles : ℝ := total_weight_CaO / molecular_weight_CaO) :
  atomic_weight_Ca = 40.08 →
  atomic_weight_O = 16.00 →
  total_weight_CaO = 560 →
  number_of_moles ≈ 9.99 :=
by
  intros hCa hO htotal
  sorry

end calcium_oxide_moles_l734_734548


namespace part_a_part_b_l734_734171

-- Define a number T as persistent with the given conditions.
def persistent (T : ℝ) : Prop :=
  ∀ (a b c d : ℝ), 
    a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
    a ≠ 1 → b ≠ 1 → c ≠ 1 → d ≠ 1 →
    a + b + c + d = T → 
    (1 / a + 1 / b + 1 / c + 1 / d = T →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) + 1 / (1 - d) = T))

-- Part (a): Prove that if T is persistent, then T = 2.
theorem part_a (T : ℝ) (h : persistent T) : T = 2 :=
  sorry

-- Part (b): Prove that T = 2 is persistent.
theorem part_b : persistent 2 :=
  by 
  intros a b c d ha0 hb0 hc0 hd0 ha1 hb1 hc1 hd1 habcd h. 
  -- use given info to show the desired conclusion
  sorry

end part_a_part_b_l734_734171


namespace vasya_drives_fraction_l734_734664

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l734_734664


namespace emery_total_alteration_cost_l734_734723

-- Definition of the initial conditions
def num_pairs_of_shoes := 17
def cost_per_shoe := 29
def shoes_per_pair := 2

-- Proving the total cost
theorem emery_total_alteration_cost : num_pairs_of_shoes * shoes_per_pair * cost_per_shoe = 986 := by
  sorry

end emery_total_alteration_cost_l734_734723


namespace graph_is_empty_l734_734714

theorem graph_is_empty :
  ¬∃ x y : ℝ, 4 * x^2 + 9 * y^2 - 16 * x - 36 * y + 64 = 0 :=
by
  -- the proof logic will go here
  sorry

end graph_is_empty_l734_734714


namespace pool_buckets_l734_734221

theorem pool_buckets (buckets_george_per_round buckets_harry_per_round rounds : ℕ) 
  (h_george : buckets_george_per_round = 2) 
  (h_harry : buckets_harry_per_round = 3) 
  (h_rounds : rounds = 22) : 
  buckets_george_per_round + buckets_harry_per_round * rounds = 110 := 
by 
  sorry

end pool_buckets_l734_734221


namespace largest_price_drop_in_March_l734_734072

def January_change := -1.25
def February_change := 0.75
def March_change := -3.00
def April_change := 0.25

theorem largest_price_drop_in_March : 
  (March_change < January_change) ∧ 
  (March_change < February_change) ∧ 
  (March_change < April_change) := 
by {
  have h1 : March_change < January_change := by sorry,
  have h2 : March_change < February_change := by sorry,
  have h3 : March_change < April_change := by sorry,
  exact ⟨h1, h2, h3⟩
}

end largest_price_drop_in_March_l734_734072


namespace doubled_volume_l734_734628

theorem doubled_volume (V₀ V₁ : ℝ) (hV₀ : V₀ = 3) (h_double : V₁ = V₀ * 8) : V₁ = 24 :=
by 
  rw [hV₀] at h_double
  exact h_double

end doubled_volume_l734_734628


namespace total_sequences_l734_734429

theorem total_sequences (class1_students class2_students sessions_per_week : ℕ) (h1 : class1_students = 12) (h2 : class2_students = 13) (h3 : sessions_per_week = 3) : 
  (class1_students * class2_students) ^ sessions_per_week = 3_796_416 :=
by
  rw [h1, h2, h3]
  norm_num

end total_sequences_l734_734429


namespace find_m_l734_734584

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l734_734584


namespace average_of_r_s_t_l734_734822

theorem average_of_r_s_t
  (r s t : ℝ)
  (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end average_of_r_s_t_l734_734822


namespace min_f_gt_two_min_h_is_zero_l734_734264

variables {x : ℝ} (f : ℝ → ℝ) (g : ℝ → ℝ) (h : ℝ → ℝ) (m : ℝ)

-- Given the function f(x) = exp(x) - log(x), where x > 0
def f (x : ℝ) : ℝ := exp x - log x

-- (1) Prove that g(x) = f'(x) is monotonically increasing on (0, +∞)
def g (x : ℝ) : ℝ := exp x - (1 / x)

-- (2) Prove that the minimum value m of f(x) is greater than 2
theorem min_f_gt_two (x : ℝ) (hx : 0 < x) : 
  let m := inf {f x | x > 0} in
  m > 2 := sorry

-- (3) For the function h(x) = exp(x) - exp(m) * log(x), find the minimum value is 0
def h (x m : ℝ) : ℝ := exp x - exp m * log x

theorem min_h_is_zero (x m : ℝ) (hx : 0 < x) (hm_gt_two : m > 2) :
  let h_min := inf {h x m | x > 0} in
  h_min = 0 := sorry

end min_f_gt_two_min_h_is_zero_l734_734264


namespace base_radius_of_cone_l734_734433

-- Definitions of the conditions
def R1 : ℕ := 5
def R2 : ℕ := 4
def R3 : ℕ := 4
def height_radius_ratio := 4 / 3

-- Main theorem statement
theorem base_radius_of_cone : 
  (R1 = 5) → (R2 = 4) → (R3 = 4) → (height_radius_ratio = 4 / 3) → 
  ∃ r : ℚ, r = 169 / 60 :=
by 
  intros hR1 hR2 hR3 hRatio
  sorry

end base_radius_of_cone_l734_734433


namespace original_price_of_color_TV_l734_734142

theorem original_price_of_color_TV
  (x : ℝ)  -- Let the variable x represent the original price
  (h1 : x * 1.4 * 0.8 - x = 144)  -- Condition as equation
  : x = 1200 := 
sorry  -- Proof to be filled in later

end original_price_of_color_TV_l734_734142


namespace negation_P_eq_Q_l734_734508

-- Define the proposition P: For any x ∈ ℝ, x^2 - 2x - 3 ≤ 0
def P : Prop := ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0

-- Define its negation which is the proposition Q
def Q : Prop := ∃ x : ℝ, x^2 - 2*x - 3 > 0

-- Prove that the negation of P is equivalent to Q
theorem negation_P_eq_Q : ¬P = Q :=
  by
  sorry

end negation_P_eq_Q_l734_734508


namespace polygon_area_l734_734376

-- Define conditions as Lean definitions
def sides : ℕ := 32
def perimeter (s : ℝ) : ℝ := sides * s
def side_length : ℝ := 2
def area_of_single_square (s : ℝ) : ℝ := s * s
def fully_divided_area (num_squares : ℕ) (s : ℝ) : ℝ := num_squares * area_of_single_square(s)
def num_squares : ℕ := 36

-- Given conditions
def conditions := (sides = 32) ∧ (perimeter side_length = 64) ∧ (side_length = 2) ∧ (num_squares = 36)

-- Prove the area of the polygon is 144
theorem polygon_area : conditions → fully_divided_area num_squares side_length = 144 :=
by
  intro h
  sorry

end polygon_area_l734_734376


namespace hexagonal_pyramid_edge_length_l734_734620

noncomputable def hexagonal_pyramid_edge_sum (s h : ℝ) : ℝ :=
  let perimeter := 6 * s
  let center_to_vertex := s * (1 / 2) * Real.sqrt 3
  let slant_height := Real.sqrt (h^2 + center_to_vertex^2)
  let edge_sum := perimeter + 6 * slant_height
  edge_sum

theorem hexagonal_pyramid_edge_length (s h : ℝ) (a : ℝ) :
  s = 8 →
  h = 15 →
  a = 48 + 6 * Real.sqrt 273 →
  hexagonal_pyramid_edge_sum s h = a :=
by
  intros
  sorry

end hexagonal_pyramid_edge_length_l734_734620


namespace difference_in_cents_l734_734434

theorem difference_in_cents (pennies dimes : ℕ) (h : pennies + dimes = 5050) (hpennies : 1 ≤ pennies) (hdimes : 1 ≤ dimes) : 
  let total_value := pennies + 10 * dimes
  let max_value := 50500 - 9 * 1
  let min_value := 50500 - 9 * 5049
  max_value - min_value = 45432 := 
by 
  -- proof goes here
  sorry

end difference_in_cents_l734_734434


namespace upper_limit_of_first_range_l734_734144

theorem upper_limit_of_first_range (A : ℝ) :
  (∃ x : ℝ, x > 3 ∧ x < A ∧ x = 7) ∧ (∃ x : ℝ, x > 6 ∧ x < 10 ∧ x = 7) →
  A > 7 :=
by {
  intro h,
  cases h,
  cases h_left with x hx,
  cases h_right with x' hx',
  have hx7 : x = 7 := hx.2.2,
  have hA : x < A := hx.2.1,
  rw hx7 at hA,
  linarith,
  sorry
}

end upper_limit_of_first_range_l734_734144


namespace count_lines_passing_through_four_points_l734_734817

-- Define the conditions
def is_valid_point (i j k : ℕ) : Prop := 
  1 ≤ i ∧ i ≤ 4 ∧ 
  1 ≤ j ∧ j ≤ 4 ∧ 
  1 ≤ k ∧ k ≤ 4

-- Define what it means for points (i, j, k) to be collinear
def collinear_points (p1 p2 p3 p4 : ℕ × ℕ × ℕ) : Prop :=
  ∃ (a b c : ℤ), 
    ¬(a = 0 ∧ b = 0 ∧ c = 0) ∧
    (p2 = (p1.1 + a, p1.2 + b, p1.3 + c)) ∧
    (p3 = (p1.1 + 2*a, p1.2 + 2*b, p1.3 + 2*c)) ∧
    (p4 = (p1.1 + 3*a, p1.2 + 3*b, p1.3 + 3*c))

-- Define the number of valid lines passing through four distinct points in the grid
noncomputable def num_valid_lines : ℕ := 76

-- The proof problem
theorem count_lines_passing_through_four_points : 
  (∃ (points : Finset (ℕ × ℕ × ℕ)),
    points.card = 4 ∧ 
    ∀ (p ∈ points), 
      is_valid_point p.1 p.2 p.3 ∧ 
    ∃ (p1 p2 p3 p4 : ℕ × ℕ × ℕ) ∈ points, 
      collinear_points p1 p2 p3 p4) →
    num_valid_lines = 76 := 
sorry

end count_lines_passing_through_four_points_l734_734817


namespace part1_part2_l734_734246

variable {a b c : ℚ}

theorem part1 (ha : a < 0) : (a / |a|) = -1 :=
sorry

theorem part2 (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  min (a * b / |a * b| + |b * c| / (b * c) + a * c / |a * c| + |a * b * c| / (a * b * c)) (-2) = -2 :=
sorry

end part1_part2_l734_734246


namespace solid2_solid4_views_identical_l734_734839

-- Define the solids and their orthographic views
structure Solid :=
  (top_view : String)
  (front_view : String)
  (side_view : String)

-- Given solids as provided by the problem
def solid1 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid2 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid3 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid4 : Solid := { top_view := "...", front_view := "...", side_view := "..." }

-- Function to compare two solids' views
def views_identical (s1 s2 : Solid) : Prop :=
  (s1.top_view = s2.top_view ∧ s1.front_view = s2.front_view) ∨
  (s1.top_view = s2.top_view ∧ s1.side_view = s2.side_view) ∨
  (s1.front_view = s2.front_view ∧ s1.side_view = s2.side_view)

-- Theorem statement
theorem solid2_solid4_views_identical : views_identical solid2 solid4 := 
sorry

end solid2_solid4_views_identical_l734_734839


namespace laundry_problem_l734_734008

def problem_statement : Prop := 
  ∃ (S : ℕ),
  (let T := 9 in 
   let returned_sweaters := 3 in
   let returned_t_shirts := 27 in
   let missing_items := 15 in
   let original_items := returned_sweaters + returned_t_shirts + missing_items in
   let total_t_shirts := T in
   let total_sweaters := S in
   original_items = total_sweaters + total_t_shirts) ∧
  (S = 18) ∧ 
  (2 * T = S)

theorem laundry_problem : problem_statement := sorry

end laundry_problem_l734_734008


namespace magnitude_a_plus_b_l734_734229

open Real

/-- Given x and y are real numbers, with vectors a = (x, 1), b = (1, y), 
    and c = (2, -4) such that a is perpendicular to c and b is parallel to c. 
    Prove that the magnitude of (a + b) is √10. -/
theorem magnitude_a_plus_b (x y : ℝ) (h1 : ((2 * x) - 4) = 0) 
    (h2 : (1 / 2) = (y / -4)) : 
    (∥(x + 1, 1 + y)∥ = sqrt 10) := by
  sorry

end magnitude_a_plus_b_l734_734229


namespace oatmeal_cookies_divisible_by_6_l734_734858

theorem oatmeal_cookies_divisible_by_6 (O : ℕ) (h1 : 48 % 6 = 0) (h2 : O % 6 = 0) :
    ∃ x : ℕ, O = 6 * x :=
by sorry

end oatmeal_cookies_divisible_by_6_l734_734858


namespace spherical_distance_AC_l734_734834

-- Define the vertices and distances in the problem
variables {A B C D A' B' C' D' : Type}

-- Define the conditions from the problem
def is_regular_quadrilateral_prism (A B C D A' B' C' D' : Type) : Prop :=
  -- Conditions to be filled appropriately based on geometrical definitions 
  -- of a regular quadrilateral prism inscribed in a sphere.
  sorry

-- Distances given in the problem
def AB_dist (AB: Real): AB = 1 := sorry
def AA'_dist (AA': Real): AA' = sqrt(2) := sorry

-- The main theorem statement
theorem spherical_distance_AC :
  is_regular_quadrilateral_prism A B C D A' B' C' D' →
  AB = 1 →
  AA' = sqrt(2) →
  -- The spherical distance formula result
  spherical_distance A C = π / 2 :=
sorry

end spherical_distance_AC_l734_734834


namespace pascal_triangle_third_number_l734_734092

theorem pascal_triangle_third_number (n : ℕ) (h : n + 1 = 52) : (nat.choose n 2) = 1275 := by
  have h_n : n = 51 := by
    linarith
  rw [h_n]
  norm_num

end pascal_triangle_third_number_l734_734092


namespace find_f_neg_two_l734_734252

-- Definition of an odd function
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, h(-x) = -h(x)

-- Given conditions
variables (h : ℝ → ℝ) (f : ℝ → ℝ)
variable (h_odd : is_odd h)
variable (h_def : ∀ x : ℝ, f(x) = h(x) + 2)
variable (h_zero_at_two : f(2) = 0)

-- To prove
theorem find_f_neg_two : f(-2) = 4 :=
by sorry

end find_f_neg_two_l734_734252


namespace height_three_years_ago_l734_734431

theorem height_three_years_ago (H : ℝ) (H_three_years_ago : 1.06 * 1.04 * 1.05 * H = 126) :
  H ≈ 113.28 :=
by
  unfold gives_H ≈
  sorry

end height_three_years_ago_l734_734431


namespace pyramid_rearrangement_l734_734901

noncomputable def pyramid_cubes : Prop :=
  ∃ (initial_pyramid : Fin 10 → Fin 10) (rearranged_pyramid : Fin 10 → Fin 10),
    (∀ i j, initial_pyramid i ≠ rearranged_pyramid j ∧ i ≠ j) ∧ 
    (∀ i j, pyramida_structure i j → pyramida_structure (rearranged_pyramid i) (rearranged_pyramid j))

theorem pyramid_rearrangement : pyramid_cubes :=
sorry

end pyramid_rearrangement_l734_734901


namespace vitya_masha_probability_l734_734543

theorem vitya_masha_probability :
  let total_days := 30
  let total_pairs := total_days * total_days
  let favourable_pairs := (∑ k in Finset.range total_days, k)
  total_pairs = 900 ∧ favourable_pairs = 435 ∧
  probability (Vitya at_least_one_day_older_than_Masha) = favourable_pairs / total_pairs :=
by {
  let total_days := 30,
  let total_pairs := total_days * total_days,
  let favourable_pairs := (∑ k in Finset.range total_days, k),
  
  have h1: total_pairs = 900 := by norm_num,
  have h2: favourable_pairs = 435 := by norm_num,

  have probability := 435.0 / 900.0,
  norm_num at top,
  simp,
}

end vitya_masha_probability_l734_734543


namespace chess_group_players_l734_734522

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 1225) : n = 50 :=
sorry

end chess_group_players_l734_734522


namespace sequence_sum_l734_734058

theorem sequence_sum (n k : ℕ) (h_n_ge_4 : n ≥ 4) :
  (∃ (a : ℕ → ℕ), 
    (a 0 = 1) ∧ 
    (∀ i, 0 ≤ i ∧ i < n - 1 → a (i + 1) = k * ∑ j in Finset.range (i + 1), a j) ∧ 
    ∑ i in Finset.range n, a i = 531441) ↔ 
    (k = 80 ∧ n = 4) ∨ 
    (k = 26 ∧ n = 5) ∨ 
    (k = 8 ∧ n = 7) ∨ 
    (k = 2 ∧ n = 13) :=
by
  sorry

end sequence_sum_l734_734058


namespace sophia_daily_saving_l734_734913

theorem sophia_daily_saving (total_days : ℕ) (total_saving : ℝ) (h1 : total_days = 20) (h2 : total_saving = 0.20) : 
  (total_saving / total_days) = 0.01 :=
by
  sorry

end sophia_daily_saving_l734_734913


namespace line_up_ways_l734_734354

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l734_734354


namespace find_f_g_3_l734_734806

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem find_f_g_3 : f (g 3) = 51 := 
by 
  sorry

end find_f_g_3_l734_734806


namespace solve_quadratic_calculate_expression_l734_734998

-- Problem 1
theorem solve_quadratic : 
  ∀ x : ℝ, x^2 - 4 * x - 3 = 0 ↔ (x = 2 + real.sqrt 7) ∨ (x = 2 - real.sqrt 7) :=
by 
  sorry

-- Problem 2
theorem calculate_expression : 
  |(-3: ℝ)| - 4 * real.sin (real.pi / 4) + real.sqrt 8 + (real.pi - 3)^0 = 4 :=
by 
  sorry

end solve_quadratic_calculate_expression_l734_734998


namespace part1_part2_l734_734267

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x
noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := 1 / x - m
noncomputable def f'' (x : ℝ) (m : ℝ) : ℝ := -1 / (x * x)

theorem part1 (m : ℝ) (h : ∀ x > 0, f m x ≤ -1) : m = 1 :=
sorry

theorem part2 (x1 x2 m : ℝ) (h₁ : f m x1 = 0) (h₂ : f m x2 = 0) (h₃ : exp x1 ≤ x2) :
  (x1 - x2) * f'' (x1 + x2) m ≥ 2 / (1 + Real.exp 1) :=
  sorry

end part1_part2_l734_734267


namespace sum_of_first_n_terms_l734_734842

variable {a : ℕ → ℝ}
variable {n : ℕ}

-- Definitions based on conditions in the problem
def geometric_sequence := ∃ q a1, q ≠ 0 ∧ a 1 = a1 ∧ a (n + 1) = a n * q
def condition1 := a 2 + a 4 = 20
def condition2 := a 3 + a 5 = 40

theorem sum_of_first_n_terms (h : geometric_sequence) (h1 : condition1) (h2 : condition2) : 
  (finset.range n).sum (λ i, a (i+1)) = 2^(n + 1) - 2 := 
sorry

end sum_of_first_n_terms_l734_734842


namespace greatest_integer_m_l734_734972

theorem greatest_integer_m (n : ℤ) (m : ℕ) (h₁ : n = 14) (h₂ : m = 5) : 
  ∃ k : ℤ, 40.factorial / (n ^ m) = k :=
by
  sorry

end greatest_integer_m_l734_734972


namespace no_such_subset_M_exists_l734_734189

theorem no_such_subset_M_exists :
  ¬ (∃ M : set ℝ, 
    M.nonempty ∧ (∀ r > 0, ∀ a ∈ M, ∃! b ∈ M, |a - b| = r)) :=
sorry

end no_such_subset_M_exists_l734_734189


namespace value_of_s_l734_734875

theorem value_of_s (s : ℝ) : 
  (3 * (-1)^4 - 2 * (-1)^3 + 4 * (-1)^2 - 5 * (-1) + s = 0) → s = -14 :=
by
  sorry

end value_of_s_l734_734875


namespace soccer_team_wins_l734_734037

theorem soccer_team_wins :
  ∃ W D : ℕ, 
    (W + 2 + D = 20) ∧  -- total games
    (3 * W + D = 46) ∧  -- total points
    (W = 14) :=         -- correct answer
by
  sorry

end soccer_team_wins_l734_734037


namespace third_number_in_pascals_triangle_row_51_l734_734089

theorem third_number_in_pascals_triangle_row_51 :
  let n := 51 in 
  ∃ result, result = (n * (n - 1)) / 2 ∧ result = 1275 :=
by
  let n := 51
  use (n * (n - 1)) / 2
  split
  . rfl
  . exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num; ring)
  sorry -- This 'sorry' is provided to formally conclude the directive


end third_number_in_pascals_triangle_row_51_l734_734089


namespace min_waiting_time_max_waiting_time_expected_waiting_time_l734_734992

open Nat

noncomputable def C : ℕ → ℕ → ℕ
| n, 0     => 1
| 0, k     => 0
| n+1, k+1 => C n k + C n (k+1)

def a := 1
def b := 5
def n := 5
def m := 3

def T_min := a * C (n - 1) 2 + m * n * a + b * C m 2
def T_max := a * C n 2 + b * m * n + b * C m 2
def E_T := C (n + m) 2 * (b * m + a * n) / (m + n)

theorem min_waiting_time : T_min = 40 := by
  sorry

theorem max_waiting_time : T_max = 100 := by
  sorry

theorem expected_waiting_time : E_T = 70 := by
  sorry

end min_waiting_time_max_waiting_time_expected_waiting_time_l734_734992


namespace positive_integer_solutions_to_inequality_l734_734059

theorem positive_integer_solutions_to_inequality : 
  let S := {x : ℕ | (0 < x) ∧ (0.5 * (8 - x) > 2)} in S.card = 3 := 
  sorry

end positive_integer_solutions_to_inequality_l734_734059


namespace greenville_state_univ_l734_734574

noncomputable def min_cost_on_boxes
  (box_length : ℕ)
  (box_width : ℕ)
  (box_height : ℕ)
  (box_cost : ℝ)
  (total_volume : ℕ)
  : ℝ :=
  let volume_one_box := box_length * box_width * box_height
  let number_of_boxes := Int.ceil ↑(total_volume / volume_one_box)
  number_of_boxes * box_cost

theorem greenville_state_univ
  (box_length box_width box_height : ℕ)
  (box_cost : ℝ)
  (total_volume : ℕ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : box_cost = 1.20)
  (h5 : total_volume = 3060000)
  : min_cost_on_boxes box_length box_width box_height box_cost total_volume = 612 :=
by
  rw [h1, h2, h3, h4, h5]
  sorry

end greenville_state_univ_l734_734574


namespace Vasya_distance_fraction_l734_734659

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l734_734659


namespace parameter_a_values_l734_734199

theorem parameter_a_values (a : ℝ) :
  (∃ x y : ℝ, |x + y + 8| + |x - y + 8| = 16 ∧ ((|x| - 8)^2 + (|y| - 15)^2 = a) ∧
    (∀ x₁ y₁ x₂ y₂ : ℝ, |x₁ + y₁ + 8| + |x₁ - y₁ + 8| = 16 →
      (|x₁| - 8)^2 + (|y₁| - 15)^2 = a →
      |x₂ + y₂ + 8| + |x₂ - y₂ + 8| = 16 →
      (|x₂| - 8)^2 + (|y₂| - 15)^2 = a →
      (x₁, y₁) = (x₂, y₂) ∨ (x₁, y₁) = (y₂, x₂))) ↔ a = 49 ∨ a = 289 :=
by sorry

end parameter_a_values_l734_734199


namespace shorter_train_length_l734_734080

noncomputable def length_of_shorter_train
  (v1_in_kmph : ℝ) (v2_in_kmph : ℝ) (L_longer : ℝ) (t : ℝ) : ℝ :=
let v1 := v1_in_kmph * 1000 / 3600 in
let v2 := v2_in_kmph * 1000 / 3600 in
let v_relative := v1 + v2 in
let D_total := v_relative * t in
D_total - L_longer

theorem shorter_train_length
  (v1 : ℝ) (v2 : ℝ) (L_longer : ℝ) (t : ℝ) :
  v1 = 80 → v2 = 65 → L_longer = 165 → t = 7.348377647029618 →
  length_of_shorter_train v1 v2 L_longer t ≈ 130.999 :=
by
  intros h1 h2 h3 h4
  sorry

end shorter_train_length_l734_734080


namespace find_f_neg_one_l734_734780

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then x^2 + 2 * x else - ( (x^2) + (2 * x))

theorem find_f_neg_one : 
  f (-1) = -3 :=
by 
  sorry

end find_f_neg_one_l734_734780


namespace parallel_lines_l734_734373

variables {A B C D E : Type} [Point A] [Point B] [Point C] [Point D] [Point E]

structure Triangle (A B C : Type) :=
(is_isosceles : B ≠ C ∧ distance A B = distance B C)

structure Incircle (A B C O E M : Type) :=
(center_incenter : Center O (InscribedCircle (Triangle A B C)))
(touch_ab : Touch AB E)
(touch_ac : Touch AC M)
(midpoint_ac : Midpoint M A C)

noncomputable def extension_point (A C D : Type) :=
AD_half_AC : distance A D = (1/2) * distance A C

theorem parallel_lines (A B C D E O M : Type) [triangle : Triangle A B C]
  [incircle : Incircle A B C O E M] [ext_point : extension_point A C D] :
  parallel DE AO := sorry

end parallel_lines_l734_734373


namespace count_valid_arrangements_l734_734325

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l734_734325


namespace fourth_term_binomial_expansion_l734_734733

theorem fourth_term_binomial_expansion (x y : ℝ) :
  let a := x
  let b := -2*y
  let n := 7
  let r := 3
  (binomial n r) * a^(n-r) * b^r = -280*x^4*y^3 := by
  sorry

end fourth_term_binomial_expansion_l734_734733


namespace divisibility_of_poly_l734_734877

theorem divisibility_of_poly (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x):
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * (y-z) * (z-x) * (x-y) * k :=
by
  sorry

end divisibility_of_poly_l734_734877


namespace max_value_of_expression_l734_734204

noncomputable def max_value_expr : ℝ :=
  5 / 2

theorem max_value_of_expression
  (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) :
  (sin θ₁ * cos θ₂) + (sin θ₂ * cos θ₃) + (sin θ₃ * cos θ₄) + (sin θ₄ * cos θ₅) + (sin θ₅ * cos θ₁) ≤ max_value_expr :=
sorry

end max_value_of_expression_l734_734204


namespace line_through_midpoint_C_l734_734964

open Set

theorem line_through_midpoint_C {O : Type} [MetricSpace O] {circle : O → ℝ → Set O} 
  (A B C M N : O) 
  {r1 r2 : ℝ} 
  (h_circle_AB : CircleCircumference circle A B C) -- representing the given circle circumference with AB segment
  (h_C_midpoint : IsMidpoint C A B) -- C is the midpoint of arc AB excluding the segment AB
  (O1 O2 : O) 
  (h_circ1 : circle O1 r1 ⊆ segment A B) 
  (h_circ2 : circle O2 r2 ⊆ segment A B)
  (h_intersect : (circle O1 r1 ∩ circle O2 r2) = {M, N}) :
  LineThrough C M ∧ LineThrough C N :=
sorry

noncomputable def CircleCircumference {O : Type} (circle : O → ℝ → Set O) (A B C : O) : Prop := sorry
noncomputable def IsMidpoint {O : Type} (C A B : O) : Prop := sorry
noncomputable def segment {O : Type} [MetricSpace O] (A B : O) : Set O := sorry
noncomputable def LineThrough {O : Type} [MetricSpace O] (C M : O) : Prop := sorry

end line_through_midpoint_C_l734_734964


namespace find_b_l734_734297

-- Definitions of conditions
def is_imaginary (z : ℂ) : Prop := z.re = 0
def complex_expression (b : ℝ) : ℂ := (1 + b * complex.I) * (2 - complex.I)

-- Theorem statement
theorem find_b (b : ℝ) (h : is_imaginary (complex_expression b)) : b = -2 :=
sorry

end find_b_l734_734297


namespace max_square_size_is_4_l734_734893

def can_assemble_square (n : ℕ) : Prop :=
∀(color : ℕ → ℕ → bool), 
  ∃ (rows cols : finset (fin n)), 
    rows.card = 2 ∧ cols.card = 2 ∧ 
      (rows.to_list.product cols.to_list).all (λ ⟨r, c⟩, color r c) ∨
      (rows.to_list.product cols.to_list).all (λ ⟨r, c⟩, ¬color r c)

theorem max_square_size_is_4 :
  ∀n : ℕ, (¬can_assemble_square n → n ≤ 4) :=
by
  assume n : ℕ,
  -- The details of the proof would go here, but we skip it with 'sorry'
  sorry

end max_square_size_is_4_l734_734893


namespace minimum_omega_for_symmetric_curve_l734_734474

theorem minimum_omega_for_symmetric_curve (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 2) + π / 3) = sin (-ω * (x + π / 2) + π / 3)) ↔ ω = 1 / 3 :=
by
  sorry

end minimum_omega_for_symmetric_curve_l734_734474


namespace masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l734_734002

/-- 
Part (a): Define the problem statement where, given the iterative process on a number,
it stabilizes at 17.
-/
theorem masha_final_number_stabilizes (x y : ℕ) (n : ℕ) (h_stable : ∀ x y, 10 * x + y = 3 * x + 2 * y) :
  n = 17 :=
by
  sorry

/--
Part (b): Define the problem statement to find the smallest 2015-digit number ending with the
digits 09 that eventually stabilizes to 17.
-/
theorem masha_smallest_initial_number_ends_with_09 :
  ∃ (n : ℕ), n ≥ 10^2014 ∧ n % 100 = 9 ∧ (∃ k : ℕ, 10^2014 + k = n ∧ (10 * ((n - k) / 10) + (n % 10)) = 17) :=
by
  sorry

end masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l734_734002


namespace vasya_drove_0_4_of_total_distance_l734_734647

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l734_734647


namespace unique_solution_pair_l734_734185

theorem unique_solution_pair (x y : ℝ) :
  (4 * x ^ 2 + 6 * x + 4) * (4 * y ^ 2 - 12 * y + 25) = 28 →
  (x, y) = (-3 / 4, 3 / 2) := by
  intro h
  sorry

end unique_solution_pair_l734_734185


namespace proposition_p_true_proposition_q_false_proposition_2_3_true_l734_734243

variables {x y : ℝ}

def proposition_p : Prop := ∀ x y : ℝ, x > y → -x < -y
def proposition_q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

theorem proposition_p_true : proposition_p :=
by {
  intros x y h,
  exact neg_lt_neg h,
  sorry,
}

theorem proposition_q_false :
  proposition_q :=
by {
  intro h,
  exfalso,
  sorry,
}

theorem proposition_2_3_true :
  ∀ x y : ℝ, proposition_p ∨ proposition_q ∧ proposition_p ∧ ¬proposition_q :=
by {
  intros,
  split,
  sorry,
  sorry,
}

end proposition_p_true_proposition_q_false_proposition_2_3_true_l734_734243


namespace sum_of_squares_with_signs_l734_734401

theorem sum_of_squares_with_signs (n : ℤ) : 
  ∃ (k : ℕ) (s : Fin k → ℤ), (∀ i : Fin k, s i = 1 ∨ s i = -1) ∧ n = ∑ i : Fin k, s i * ((i + 1) * (i + 1)) := sorry

end sum_of_squares_with_signs_l734_734401


namespace glove_selection_count_l734_734157

theorem glove_selection_count :
  let n := 6 in
  let pair_count := n in
  let ways_to_choose_one_pair := pair_count.choose 1 in
  let remaining_pairs := pair_count - 1 in
  let ways_to_choose_three_pairs := remaining_pairs.choose 3 in
  let ways_to_choose_one_from_each_pair := 2 ^ 3 in
  let total_ways := ways_to_choose_one_pair * (ways_to_choose_three_pairs * ways_to_choose_one_from_each_pair) in
  total_ways = 480 :=
by
  sorry

end glove_selection_count_l734_734157


namespace doubled_dimensions_volume_l734_734626

theorem doubled_dimensions_volume (original_volume : ℝ) (length_factor width_factor height_factor : ℝ) 
  (h : original_volume = 3) 
  (hl : length_factor = 2)
  (hw : width_factor = 2)
  (hh : height_factor = 2) : 
  original_volume * length_factor * width_factor * height_factor = 24 :=
by
  sorry

end doubled_dimensions_volume_l734_734626


namespace solve_quadratic_calculate_expression_l734_734999

-- Problem 1
theorem solve_quadratic : 
  ∀ x : ℝ, x^2 - 4 * x - 3 = 0 ↔ (x = 2 + real.sqrt 7) ∨ (x = 2 - real.sqrt 7) :=
by 
  sorry

-- Problem 2
theorem calculate_expression : 
  |(-3: ℝ)| - 4 * real.sin (real.pi / 4) + real.sqrt 8 + (real.pi - 3)^0 = 4 :=
by 
  sorry

end solve_quadratic_calculate_expression_l734_734999


namespace triangle_angle_C_l734_734771

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π) (h3 : A + B + C = π)
  (h4 : (a + b + c) * (Real.sin A + Real.sin B - Real.sin C) = a * Real.sin B)
  (h5 : a = 2 * sin A) (h6 : b = 2 * sin B) (h7 : c = 2 * sin C):
  C = 2 * real.pi / 3 :=
  sorry

end triangle_angle_C_l734_734771


namespace sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l734_734519

open Real

theorem sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq :
  (∀ α : ℝ, sin α = cos α → ∃ k : ℤ, α = (k : ℝ) * π + π / 4) ∧
  (¬ ∀ k : ℤ, ∀ α : ℝ, α = (k : ℝ) * π + π / 4 → sin α = cos α) :=
by
  sorry

end sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l734_734519


namespace total_profit_is_8800_l734_734150

variable (A B C : Type) [CommRing A] [CommRing B] [CommRing C]

variable (investment_A investment_B investment_C : ℝ)
variable (total_profit : ℝ)

-- Conditions
def A_investment_three_times_B (investment_A investment_B : ℝ) : Prop :=
  investment_A = 3 * investment_B

def B_invest_two_thirds_C (investment_B investment_C : ℝ) : Prop :=
  investment_B = 2 / 3 * investment_C

def B_share_is_1600 (investment_B total_profit : ℝ) : Prop :=
  1600 = (2 / 11) * total_profit

theorem total_profit_is_8800 :
  A_investment_three_times_B investment_A investment_B →
  B_invest_two_thirds_C investment_B investment_C →
  B_share_is_1600 investment_B total_profit →
  total_profit = 8800 :=
by
  intros
  sorry

end total_profit_is_8800_l734_734150


namespace john_experience_when_mike_started_l734_734852

-- Definitions from the conditions
variable (J O M : ℕ)
variable (h1 : J = 20) -- James currently has 20 years of experience
variable (h2 : O - 8 = 2 * (J - 8)) -- 8 years ago, John had twice as much experience as James
variable (h3 : J + O + M = 68) -- Combined experience is 68 years

-- Theorem to prove
theorem john_experience_when_mike_started : O - M = 16 := 
by
  -- Proof steps go here
  sorry

end john_experience_when_mike_started_l734_734852


namespace sum_of_2nd_and_3rd_smallest_l734_734070

theorem sum_of_2nd_and_3rd_smallest :
  let numbers := [10, 11, 12, 13, 14] in
  let sorted_numbers := List.sort (· ≤ ·) numbers in
  sorted_numbers.nth 1 + sorted_numbers.nth 2 = 23 :=
by
  let numbers := [10, 11, 12, 13, 14]
  let sorted_numbers := List.sort (· ≤ ·) numbers
  show sorted_numbers.nth 1 + sorted_numbers.nth 2 = 23
  sorry

end sum_of_2nd_and_3rd_smallest_l734_734070


namespace arrangement_count_equals_36_l734_734745

-- Define the problem conditions and statement
def select_and_arrange_letters : ℕ :=
  let possible_choices := (4.choose 2) in -- Choosing 2 more letters from 4
  let arrangements_of_3 := (3!) in -- Arranging the 3 units where one unit is 'ab'
  possible_choices * arrangements_of_3

theorem arrangement_count_equals_36 : 
  select_and_arrange_letters = 36 := 
by 
  -- Proof goes here, but we'll use 'sorry' as instructed
  sorry

end arrangement_count_equals_36_l734_734745


namespace smallest_of_powers_l734_734109

theorem smallest_of_powers :
  min (2^55) (min (3^44) (min (5^33) (6^22))) = 2^55 :=
by
  sorry

end smallest_of_powers_l734_734109


namespace pascal_triangle_third_number_l734_734095

theorem pascal_triangle_third_number (n : ℕ) (h : n + 1 = 52) : (nat.choose n 2) = 1275 := by
  have h_n : n = 51 := by
    linarith
  rw [h_n]
  norm_num

end pascal_triangle_third_number_l734_734095


namespace vasya_fraction_l734_734677

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l734_734677


namespace find_a_and_b_l734_734807

variable {α : Type*} [DecidableEq α]

theorem find_a_and_b (U A compU_A : set ℝ) (a b : ℝ) (hU : U = {2, 3, a ^ 2 + 2 * a - 3}) (hA : A = {b, 2}) (hCompU_A : compU_A = {5}) : 
    (a = -4 ∨ a = 2) ∧ b = 3 := 
by
  sorry

end find_a_and_b_l734_734807


namespace numFunctions_l734_734416

namespace FunctionCyclic
open Finset

def E := {a, b, c, d}
def f (f : E → E) := ∀ x ∈ E, f (f (f x)) = x

noncomputable def numValidFunctions : Nat := 
  { f : E → E // f f} .card

theorem numFunctions (h : numValidFunctions) : 
  h = 9 := sorry

end FunctionCyclic

end numFunctions_l734_734416


namespace smallest_perfect_cube_of_n_l734_734240

-- Additional imports for primes and powers could be necessary
open Nat

-- Suppose p, q, and r are distinct primes
variables (p q r : ℕ)
variable (hp : Prime p)
variable (hq : Prime q)
variable (hr : Prime r)
variable (hpq : p ≠ q)
variable (hpr : p ≠ r)
variable (hqr : q ≠ r)

-- Define n as specified in the problem
def n := p^2 * q^3 * r^5

-- We need to prove that the smallest perfect cube that divides n is (pqr^2)^3
theorem smallest_perfect_cube_of_n : 
  ∃ m, m^3 = (p * q * r^2)^3 ∧ (p^2 * q^3 * r^5 ∣ m^3) :=
by 
  sorry

end smallest_perfect_cube_of_n_l734_734240


namespace five_people_lineup_count_l734_734333

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l734_734333


namespace total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l734_734988

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l734_734988


namespace seven_convex_polygons_eight_convex_polygons_l734_734122

-- Define a structure for a convex polygon
structure ConvexPolygon where
  vertices : List (ℝ × ℝ) -- This represents the vertices of the polygon in 2D

-- Define the problem statement for Part (a)
theorem seven_convex_polygons (polygons : List ConvexPolygon)
  (h1 : polygons.length = 7)
  (h2 : ∀ six_polygons ⊆ polygons, six_polygons.length = 6 → ∃ p₁ p₂ : ℝ × ℝ, (∀ polygon ∈ six_polygons, ∃ s, p₁ ∈ s ∧ p₂ ∈ s))
  : ∀ p₁ p₂ : ℝ × ℝ, ¬ (∀ polygon ∈ polygons, ∃ s, p₁ ∈ s ∧ p₂ ∈ s) :=
sorry

-- Define the problem statement for Part (b)
theorem eight_convex_polygons (polygons : List ConvexPolygon)
  (h1 : polygons.length = 8)
  (h2 : ∀ seven_polygons ⊆ polygons, seven_polygons.length = 7 → ∃ p₁ p₂ : ℝ × ℝ, (∀ polygon ∈ seven_polygons, ∃ s, p₁ ∈ s ∧ p₂ ∈ s))
  : ∀ p₁ p₂ : ℝ × ℝ, ¬ (∀ polygon ∈ polygons, ∃ s, p₁ ∈ s ∧ p₂ ∈ s) :=
sorry

end seven_convex_polygons_eight_convex_polygons_l734_734122


namespace minimum_omega_l734_734478

theorem minimum_omega (ω : ℝ) (h_omega_pos : ω > 0) :
    (∃ y : ℝ → ℝ, (∀ x, y x = sin (ω * x + ω * (π / 2) + (π / 3))) ∧ 
    (∀ x, y x = y (-x))) →
    (ω = 1 / 3) :=
sorry

end minimum_omega_l734_734478


namespace vertical_line_division_l734_734838

theorem vertical_line_division (A B C : ℝ × ℝ)
    (hA : A = (0, 2)) (hB : B = (0, 0)) (hC : C = (6, 0))
    (a : ℝ) (h_area_half : 1 / 2 * 6 * 2 / 2 = 3) :
    a = 3 :=
sorry

end vertical_line_division_l734_734838


namespace exists_zero_in_interval_l734_734301

open Set Real

theorem exists_zero_in_interval (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Icc a b)) 
  (h_pos : f a * f b > 0) : ∃ c ∈ Ioo a b, f c = 0 := sorry

end exists_zero_in_interval_l734_734301


namespace systematic_sampling_l734_734139

theorem systematic_sampling (total_parts sample_size : ℕ) (sample : set ℕ) 
  (h1 : total_parts = 60) 
  (h2 : sample_size = 5) 
  (h3 : {4, 16, 40, 52} ⊆ sample) 
  (h4 : ∀ x ∈ sample, (x % 12) = 4 % 12):
  (28 ∈ sample) :=
by
  sorry

end systematic_sampling_l734_734139


namespace largest_rational_number_largest_rational_number_is_0_08_l734_734159

theorem largest_rational_number :
  ∀ (x : ℚ), (x = -1 ∨ x = 0 ∨ x = -3 ∨ x = 0.08) → x ≤ 0.08 :=
by
  sorry

theorem largest_rational_number_is_0_08 :
  ∃ (x : ℚ), x = 0.08 ∧ ∀ (y : ℚ), (y = -1 ∨ y = 0 ∨ y = -3 ∨ y = 0.08) → y ≤ x :=
by
  use 0.08
  split
  · rfl
  sorry

end largest_rational_number_largest_rational_number_is_0_08_l734_734159


namespace pills_left_l734_734117

theorem pills_left (initial_pills : ℕ) (daily_pills : ℕ) (days : ℕ) (pills_taken : ℕ) : initial_pills = 200 → daily_pills = 12 → days = 14 → pills_taken = daily_pills * days → (initial_pills - pills_taken) = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end pills_left_l734_734117


namespace cost_of_jeans_l734_734815

theorem cost_of_jeans 
  (price_socks : ℕ)
  (price_tshirt : ℕ)
  (price_jeans : ℕ)
  (h1 : price_socks = 5)
  (h2 : price_tshirt = price_socks + 10)
  (h3 : price_jeans = 2 * price_tshirt) :
  price_jeans = 30 :=
  by
    -- Sorry skips the proof, complies with the instructions
    sorry

end cost_of_jeans_l734_734815


namespace time_to_cross_platform_l734_734631

variable (l t p : ℝ) -- Define relevant variables

-- Conditions as definitions in Lean 4
def length_of_train := l
def time_to_pass_man := t
def length_of_platform := p

-- Assume given values in the problem
def cond1 : length_of_train = 186 := by sorry
def cond2 : time_to_pass_man = 8 := by sorry
def cond3 : length_of_platform = 279 := by sorry

-- Statement that represents the target theorem to be proved
theorem time_to_cross_platform (h₁ : length_of_train = 186) (h₂ : time_to_pass_man = 8) (h₃ : length_of_platform = 279) : 
  let speed := length_of_train / time_to_pass_man
  let total_distance := length_of_train + length_of_platform
  let time_to_cross := total_distance / speed
  time_to_cross = 20 :=
by sorry

end time_to_cross_platform_l734_734631


namespace mn_equals_neg3_l734_734926

noncomputable def function_with_extreme_value (m n : ℝ) : Prop :=
  let f := λ x : ℝ => m * x^3 + n * x
  let f' := λ x : ℝ => 3 * m * x^2 + n
  f' (1 / m) = 0

theorem mn_equals_neg3 (m n : ℝ) (h : function_with_extreme_value m n) : m * n = -3 :=
sorry

end mn_equals_neg3_l734_734926


namespace vasya_drove_0_4_of_total_distance_l734_734646

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l734_734646


namespace value_of_x_l734_734950

theorem value_of_x : ∀ x : ℝ, (x^2 - 4) / (x - 2) = 0 → x ≠ 2 → x = -2 := by
  intros x h1 h2
  sorry

end value_of_x_l734_734950


namespace largest_square_area_with_4_interior_lattice_points_l734_734618

/-- 
A point (x, y) in the plane is called a lattice point if both x and y are integers.
The largest square that contains exactly four lattice points solely in its interior
has an area of 9.
-/
theorem largest_square_area_with_4_interior_lattice_points : 
  ∃ s : ℝ, ∀ (x y : ℤ), 
  (1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → s^2 = 9 := 
sorry

end largest_square_area_with_4_interior_lattice_points_l734_734618


namespace problem_l734_734777

open Set

theorem problem (M : Set ℤ) (N : Set ℤ) (hM : M = {1, 2, 3, 4}) (hN : N = {-2, 2}) : 
  M ∩ N = {2} :=
by
  sorry

end problem_l734_734777


namespace problem_solution_l734_734407

noncomputable def omega : ℂ := sorry -- as omega will be a specific complex number

theorem problem_solution :
  (∀ (ω : ℂ), ω ^ 9 = 1 ∧ ω ≠ 1 →
    (let α := ω + ω^3 + ω^5,
         β := ω^2 + ω^4 + ω^6 in
     α + β = -1 ∧ α * β = 3)) :=
by
  intros ω h,
  have h1 : ω^9 = 1 := h.1,
  have h2 : ω ≠ 1 := h.2,
  let α := ω + ω^3 + ω^5,
  let β := ω^2 + ω^4 + ω^6,
  -- Prove that α + β = -1
  sorry,
  -- Prove that α * β = 3
  sorry

end problem_solution_l734_734407


namespace range_of_m_l734_734248

theorem range_of_m (x y m : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 / x + 1 / y = 1) (h4 : x + 2 * y > m^2 + 2 * m) :
  -4 < m ∧ m < 2 :=
sorry

end range_of_m_l734_734248


namespace find_symmetric_point_l734_734880

-- We define the nat numbers and the conditions of red and blue points
variables (x y : ℕ)

-- Define the point on the line as red if it has the form 81x + 100y
def is_red : ℕ → Prop := λ n, ∃ x y : ℕ, n = 81 * x + 100 * y

-- A point is blue otherwise
def is_blue (n : ℕ) : Prop := ¬ is_red n

-- Define the point of interest
def symmetric_point (a b : ℕ) : ℕ := (a * b + a + b) / 2

-- Lean statement to find the point c such that symmetric points relative to it have different colors
theorem find_symmetric_point :
  let a := 81 in
  let b := 100 in
  let c := symmetric_point a b in
  c = 4190.5 ∧ 
  (∀ n, (n < 81 * 100 + 81 + 100) → 
  (is_red n ↔ is_blue (81 * 100 + 81 + 100 - n)) ∨        (is_blue n ↔ is_red (81 * 100 + 81 + 100 - n))) :=
by
  sorry

end find_symmetric_point_l734_734880


namespace min_distance_ineq_l734_734397

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

theorem min_distance_ineq (a b c PA PB PC : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : PA > 0) (h₅ : PB > 0) (h₆ : PC > 0) (P_interior : PA + PB + PC < a + b + c) : 
  let p := semiperimeter a b c in
  min (PA / (p - a)) (min (PB / (p - b)) (PC / (p - c))) ≤ 2 / Real.sqrt 3 :=
sorry

end min_distance_ineq_l734_734397


namespace divides_expression_for_y_greater_than_1_l734_734907

theorem divides_expression_for_y_greater_than_1 (y : ℤ) (h : y > 1) :
    (y - 1) ∣ (y^(y^2 - y + 2) - 4*y + y^2016 + 3*y^2 - 1) :=
sorry

end divides_expression_for_y_greater_than_1_l734_734907


namespace rectangle_area_l734_734299

theorem rectangle_area (l w : ℕ) (h_diagonal : l^2 + w^2 = 17^2) (h_perimeter : l + w = 23) : l * w = 120 :=
by
  sorry

end rectangle_area_l734_734299


namespace fewest_apples_l734_734978

-- Definitions based on the conditions
def Yoongi_apples : Nat := 4
def Jungkook_initial_apples : Nat := 6
def Jungkook_additional_apples : Nat := 3
def Jungkook_apples : Nat := Jungkook_initial_apples + Jungkook_additional_apples
def Yuna_apples : Nat := 5

-- Main theorem based on the question and the correct answer
theorem fewest_apples : Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end fewest_apples_l734_734978


namespace average_apples_per_hour_l734_734007

theorem average_apples_per_hour :
  (5.0 / 3.0) = 1.67 := 
sorry

end average_apples_per_hour_l734_734007


namespace jacob_needs_more_marshmallows_l734_734850

def s'mores_problem : Prop :=
  ∀ (graham_crackers marshmallows : ℕ), graham_crackers = 48 → marshmallows = 6 →
  let s'mores_with_crackers := graham_crackers / 2 in
  let s'mores_with_marshmallows := marshmallows in
  let s'mores_total := s'mores_with_crackers in
  let marshmallows_needed := s'mores_total in
  let marshmallows_to_buy := marshmallows_needed - marshmallows in
  marshmallows_to_buy = 18

theorem jacob_needs_more_marshmallows : s'mores_problem := 
by
  intros graham_crackers marshmallows hc hm
  simp only [s'mores_problem] at *
  sorry

end jacob_needs_more_marshmallows_l734_734850


namespace apples_to_mangos_equivalent_l734_734456

-- Definitions and conditions
def apples_worth_mangos (a b : ℝ) : Prop := (5 / 4) * 16 * a = 10 * b

-- Theorem statement
theorem apples_to_mangos_equivalent : 
  ∀ (a b : ℝ), apples_worth_mangos a b → (3 / 4) * 12 * a = 4.5 * b :=
by
  intro a b
  intro h
  sorry

end apples_to_mangos_equivalent_l734_734456


namespace period_of_trig_sum_l734_734550

theorem period_of_trig_sum : ∀ x : ℝ, 2 * Real.sin x + 3 * Real.cos x = 2 * Real.sin (x + 2 * Real.pi) + 3 * Real.cos (x + 2 * Real.pi) := 
sorry

end period_of_trig_sum_l734_734550


namespace group_size_l734_734312

def total_people (I N B Ne : ℕ) : ℕ := I + N - B + B + Ne

theorem group_size :
  let I := 55
  let N := 43
  let B := 61
  let Ne := 63
  total_people I N B Ne = 161 :=
by
  sorry

end group_size_l734_734312


namespace earnings_per_widget_l734_734118

theorem earnings_per_widget (W_h : ℝ) (H_w : ℕ) (W_t : ℕ) (E_w : ℝ) (E : ℝ) :
  W_h = 12.50 ∧ H_w = 40 ∧ W_t = 1000 ∧ E_w = 660 →
  E = 0.16 :=
by
  sorry

end earnings_per_widget_l734_734118


namespace ratio_of_ks_l734_734866

noncomputable def quadratic_roots (k x : ℝ) := k * (x^2 - x) + x + 7 = 0

theorem ratio_of_ks
  (a b k k1 k2 : ℝ)
  (ha : quadratic_roots k a = 0)
  (hb : quadratic_roots k b = 0)
  (h_ab_ratio : a / b + b / a = 5 / 6)
  (h_k1 : quadratic_roots k1 a = 0)
  (h_k2 : quadratic_roots k2 b = 0) :
  (k1 / k2 + k2 / k1) = 433 / 36 :=
sorry

end ratio_of_ks_l734_734866


namespace problem1_problem2_problem3_problem4_l734_734031

-- Problem 1: Prove X = 93 given X - 12 = 81
theorem problem1 (X : ℝ) (h : X - 12 = 81) : X = 93 :=
by
  sorry

-- Problem 2: Prove X = 5.4 given 5.1 + X = 10.5
theorem problem2 (X : ℝ) (h : 5.1 + X = 10.5) : X = 5.4 :=
by
  sorry

-- Problem 3: Prove X = 0.7 given 6X = 4.2
theorem problem3 (X : ℝ) (h : 6 * X = 4.2) : X = 0.7 :=
by
  sorry

-- Problem 4: Prove X = 5 given X ÷ 0.4 = 12.5
theorem problem4 (X : ℝ) (h : X / 0.4 = 12.5) : X = 5 :=
by
  sorry

end problem1_problem2_problem3_problem4_l734_734031


namespace man_is_older_by_16_l734_734610

variable (M S : ℕ)

-- Condition: The present age of the son is 14.
def son_age := S = 14

-- Condition: In two years, the man's age will be twice the son's age.
def age_relation := M + 2 = 2 * (S + 2)

-- Theorem: Prove that the man is 16 years older than his son.
theorem man_is_older_by_16 (h1 : son_age S) (h2 : age_relation M S) : M - S = 16 := 
sorry

end man_is_older_by_16_l734_734610


namespace vitya_older_than_masha_probability_l734_734542

-- Define the problem conditions
def total_days_in_june : ℕ := 30
def total_possible_pairs : ℕ := total_days_in_june * total_days_in_june

-- Define the calculation of favorable pairs
def favorable_pairs : ℕ :=
  (finset.range (total_days_in_june)).sum (λ d_V, if d_V = 0 then 0 else (d_V))

-- Define the probability calculation
def probability_vitya_older_than_masha : ℚ :=
  favorable_pairs / total_possible_pairs

-- Statement of the proof problem
theorem vitya_older_than_masha_probability :
  probability_vitya_older_than_masha = 29 / 60 := 
by
  -- The proof is omitted for now
  sorry

end vitya_older_than_masha_probability_l734_734542


namespace even_function_maximum_value_l734_734469

noncomputable def f (x : ℝ) : ℝ := cos x - cos (2 * x)

/-- f(x) is an even function. -/
theorem even_function : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  have h1 : cos (-x) = cos x := cos_neg x
  have h2 : cos (-2 * x) = cos (2 * x) := by rw [neg_mul, cos_neg]
  rw [f, f, h1, h2]

/-- The maximum value of f(x) is 9/8. -/
theorem maximum_value : ∃ x : ℝ, f x = 9 / 8 :=
by
  use real.acos (1 / 4)
  sorry -- detailed proof of maximum value is beyond the scope of this translation

end even_function_maximum_value_l734_734469


namespace no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l734_734188

theorem no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100 :
  ¬ ∃ (a b c d : ℕ), a + b + c + d = 2^100 ∧ a * b * c * d = 17^100 :=
by
  sorry

end no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l734_734188


namespace pascals_triangle_third_number_l734_734096

theorem pascals_triangle_third_number (n : ℕ) (k : ℕ) (hnk : n = 51) (hk : k = 2) :
  (nat.choose n k) = 1275 :=
by {
  subst hnk,
  subst hk,
  sorry
}

end pascals_triangle_third_number_l734_734096


namespace gcd_38_23_is_1_l734_734168

theorem gcd_38_23_is_1 : Nat.gcd 38 23 = 1 := by
  sorry

end gcd_38_23_is_1_l734_734168


namespace alternating_sum_l734_734557

theorem alternating_sum : (Finset.range 10002).sum (λ n, if n % 2 = 0 then n + 1 else -(n + 1)) = -5001 := 
by
  sorry

end alternating_sum_l734_734557


namespace largest_descendants_l734_734612

-- Definition of a descendant relationship in terms of doctoral advisor lineage
def is_descendant (M M' : Type) (advisor : M' → M → Prop) : Prop :=
  ∃ (seq : List M), seq.head = some M ∧ seq.getLast == some M' ∧ ∀ (i : ℕ), i < seq.length - 1 → advisor (seq.get i) (seq.get (i+1))

-- Let M be the set of all mathematicians.
variable {M : Type}

-- Given the historical context from the 1300s to present records.
-- Given the definition of the descendant relationship.
-- We aim to prove that the largest number of descendants a single mathematician has had is 82310.
theorem largest_descendants (M : Type) (advisor : M → M → Prop) :
  ∃ (x : ℕ), x = 82310 ∧ (∀ m, (∃ desc_list : List M, desc_list.length = x ∧ is_descendant m desc_list.head.some advisor)) := sorry

end largest_descendants_l734_734612


namespace total_pencils_is_60_l734_734952

def original_pencils : ℕ := 33
def added_pencils : ℕ := 27
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_is_60 : total_pencils = 60 := by
  sorry

end total_pencils_is_60_l734_734952


namespace area_of_quadrilateral_l734_734136

-- Define necessary geometrical entities and relationships
variables (A B C D : Type) [point A] [point B] [point C] [point D]
variables (radius : ℝ) (AB BC CD DA : ℝ) (angleDAB : ℝ)
noncomputable theory
open_locale real

-- Given conditions
def inscribed_circle_radius := radius = 2
def right_angle_DAB := angleDAB = π / 2
def side_AB := AB = 5
def side_BC := BC = 6

-- Objective is to declare the area of quadrilateral ABCD
theorem area_of_quadrilateral 
  (inscribed_circle_radius : inscribed_circle_radius)
  (right_angle_DAB : right_angle_DAB)
  (side_AB : side_AB)
  (side_BC : side_BC) :
  area ABCD = 300 / 17 :=
sorry

end area_of_quadrilateral_l734_734136


namespace gcd_of_bd_is_two_l734_734398

theorem gcd_of_bd_is_two (a b c d : ℕ) (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: d > 0) 
  (h₅: (∃ ab_pairs : ℕ, ab_pairs = 2004 ∧ ∀ (x y : ℤ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 → 
  (a * x + b * y).denom = 1 ∧ (c * x + d * y).denom = 1)) 
  (h₆: Nat.gcd a c = 6) : Nat.gcd b d = 2 := 
by 
  sorry

end gcd_of_bd_is_two_l734_734398


namespace distance_between_vertices_of_hyperbola_l734_734715

def hyperbola_eq (x y : ℝ) : Prop :=
  4 * x^2 + 8 * x - 3 * y^2 + 6 * y - 17 = 0

theorem distance_between_vertices_of_hyperbola :
  (∀ x y : ℝ, hyperbola_eq x y) →
  distance = 3 * real.sqrt 2 :=
sorry

end distance_between_vertices_of_hyperbola_l734_734715


namespace train_ticket_sequence_l734_734310

theorem train_ticket_sequence (n : ℕ) (C : Finset (Fin n)) 
  (T : C → C → ℝ) 
  (symm_price : ∀ x y : C, T x y = T y x) 
  (unique_price : ∀ x y u v : C, (x ≠ u ∨ y ≠ v) → T x y ≠ T u v) :
  ∀ (start : C), ∃ (path : List C), path.length = n - 1 ∧ 
  ∀ i : Fin (n-1), T (path[i]) (path[i+1]) < T (path[i+1]) (path[i+2]) := sorry

end train_ticket_sequence_l734_734310


namespace coeff_x4_in_x2_mul_1_minus_x_to_6_l734_734201

theorem coeff_x4_in_x2_mul_1_minus_x_to_6 :
  ∀ (R : Type) [CommRing R] (x : R), coeff x^4 (x^2 * (1 - x)^6) = 15 := by
  sorry

end coeff_x4_in_x2_mul_1_minus_x_to_6_l734_734201


namespace cos_nx_minus_sin_nx_eq_one_l734_734452

theorem cos_nx_minus_sin_nx_eq_one (n : ℕ) (x : ℝ) :
  (∃ k : ℤ, x = 2 * k * Real.pi) ∨ (∃ k : ℤ, n % 2 = 0 ∧ x = (2 * k + 1) * Real.pi) ↔ cos x ^ n - sin x ^ n = 1 :=
sorry

end cos_nx_minus_sin_nx_eq_one_l734_734452


namespace vasya_fraction_is_0_4_l734_734653

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l734_734653


namespace smallest_possible_value_of_N_l734_734409

theorem smallest_possible_value_of_N :
  ∀ (a b c d e f : ℕ), a + b + c + d + e + f = 3015 → (0 < a) → (0 < b) → (0 < c) → (0 < d) → (0 < e) → (0 < f) →
  (∃ N : ℕ, N = max (max (max (max (a + b) (b + c)) (c + d)) (d + e)) (e + f) ∧ N = 604) := 
by
  sorry

end smallest_possible_value_of_N_l734_734409


namespace omega_min_value_l734_734485

theorem omega_min_value (ω : ℝ) (hω : ω > 0)
    (hSymmetry : ∀ x : ℝ, sin (ω * x + ω * π / 2 + π / 3) = sin (ω * -x + ω * π / 2 + π / 3)) :
    ω = 1 / 3 :=
begin
  sorry
end

end omega_min_value_l734_734485


namespace perimeter_not_55_l734_734079

def is_valid_perimeter (a b p : ℕ) : Prop :=
  ∃ x : ℕ, a + b > x ∧ a + x > b ∧ b + x > a ∧ p = a + b + x

theorem perimeter_not_55 (a b : ℕ) (h1 : a = 18) (h2 : b = 10) : ¬ is_valid_perimeter a b 55 :=
by
  rw [h1, h2]
  sorry

end perimeter_not_55_l734_734079


namespace smallest_solution_to_equation_l734_734207

theorem smallest_solution_to_equation :
  ∃ (x : ℝ), (x * |x| = 4 * x + 3) ∧ (∀ y : ℝ, (y * |y| = 4 * y + 3) → x ≤ y) → x = -3 :=
begin
  sorry
end

end smallest_solution_to_equation_l734_734207


namespace Karen_has_fewer_nail_polishes_than_Kim_l734_734859

theorem Karen_has_fewer_nail_polishes_than_Kim :
  ∀ (Kim Heidi Karen : ℕ), Kim = 12 → Heidi = Kim + 5 → Karen + Heidi = 25 → (Kim - Karen) = 4 :=
by
  intros Kim Heidi Karen hK hH hKH
  sorry

end Karen_has_fewer_nail_polishes_than_Kim_l734_734859


namespace problem_statement_l734_734223

noncomputable def theta (h1 : 2 * Real.cos θ + Real.sin θ = 0) (h2 : 0 < θ ∧ θ < Real.pi) : Real :=
θ

noncomputable def varphi (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) : Real :=
φ

theorem problem_statement
  (θ : Real) (φ : Real)
  (h1 : 2 * Real.cos θ + Real.sin θ = 0)
  (h2 : 0 < θ ∧ θ < Real.pi)
  (h3 : Real.sin (θ - φ) = Real.sqrt 10 / 10)
  (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) :
  Real.tan θ = -2 ∧
  Real.sin θ = (2 * Real.sqrt 5) / 5 ∧
  Real.cos θ = -Real.sqrt 5 / 5 ∧
  Real.cos φ = -Real.sqrt 2 / 10 :=
by
  sorry

end problem_statement_l734_734223


namespace value_of_g_g_2_l734_734290

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_g_g_2 : g (g 2) = 1447 := by
  sorry

end value_of_g_g_2_l734_734290


namespace opposite_of_neg3_l734_734060

theorem opposite_of_neg3 : ∃ x : ℤ, -3 + x = 0 ∧ x = 3 :=
by
  use 3
  split
  { norm_num }
  { refl }

end opposite_of_neg3_l734_734060


namespace planes_perpendicular_if_line_conditions_l734_734412

variables (m n : ℝ^3) (α β : set (ℝ^3))

-- Definitions
def parallel (l1 l2 : ℝ^3) : Prop := ∃ t, l1 = t • l2 ∨ l2 = t • l1
def perpendicular (l1 l2 : ℝ^3) : Prop := dot_product l1 l2 = 0 
def line_parallel_to_plane (l : ℝ^3) (p : set (ℝ^3)) : Prop := ∃ v ∈ p, parallel l v
def line_perpendicular_to_plane (l : ℝ^3) (p : set (ℝ^3)) : Prop := ∃ w ∈ p, perpendicular l w
def planes_perpendicular (p1 p2 : set (ℝ^3)) : Prop := ∀ v ∈ p1, ∀ w ∈ p2, perpendicular v w

-- The Lean 4 problem statement
theorem planes_perpendicular_if_line_conditions (par_m_n : parallel m n)
(line_par_m_alpha : line_parallel_to_plane m α)
(line_perp_n_beta : line_perpendicular_to_plane n β) :
planes_perpendicular α β :=
sorry

end planes_perpendicular_if_line_conditions_l734_734412


namespace fifth_term_is_zero_l734_734050

variables {x y : ℝ}

noncomputable def term1 := x + 2y
noncomputable def term2 := x - 2y
noncomputable def term3 := x^2 - 4y^2
noncomputable def term4 := x / (2 * y)

-- Define the constant difference d assuming an arithmetic sequence
noncomputable def d := term2 - term1

-- Define the fifth term based on the sequence
noncomputable def term5 := term4 + d

theorem fifth_term_is_zero (h1 : term1 = x + 2y) (h2 : term2 = x - 2y) (h3 : term3 = x^2 - 4y^2)
    (h4 : term4 = x / (2 * y)) (hd : d = -4y) : term5 = 0 :=
by
  sorry

end fifth_term_is_zero_l734_734050


namespace evaluate_expression_l734_734687

theorem evaluate_expression : 8! - 7 * 7! - 2 * 6! = 4200 := by
  sorry -- Proof placeholder

end evaluate_expression_l734_734687


namespace vasya_drives_fraction_l734_734662

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l734_734662


namespace part1_part2_l734_734804

variable {x m : ℝ}

def P (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def S (x : ℝ) (m : ℝ) : Prop := -m + 1 ≤ x ∧ x ≤ m + 1

theorem part1 (h : ∀ x, P x → P x ∨ S x m) : m ≤ 0 :=
sorry

theorem part2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (P x ↔ S x m) :=
sorry

end part1_part2_l734_734804


namespace gcd_not_perfect_square_l734_734781

theorem gcd_not_perfect_square
  (m n : ℕ)
  (h1 : (m % 3 = 0 ∨ n % 3 = 0) ∧ ¬(m % 3 = 0 ∧ n % 3 = 0))
  : ¬ ∃ k : ℕ, k * k = Nat.gcd (m^2 + n^2 + 2) (m^2 * n^2 + 3) :=
by
  sorry

end gcd_not_perfect_square_l734_734781


namespace information_inequality_holds_l734_734017

open MeasureTheory

noncomputable def informationInequality (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x > 0 ∧ g x > 0) ∧
  (measureTheory.Measure.measure (λ x, f x ≠ g x) ≠ 0) → 
  ∫ x, f x * log (f x / g x) ∂MeasureTheory.measureSpace.volume ≥ 0

-- Here's the main statement to be proven
theorem information_inequality_holds (f g : ℝ → ℝ) (hf : ∀ x, f x > 0) (hg : ∀ x, g x > 0) 
  (h_diff_measure : measureTheory.Measure.measure (λ x, f x ≠ g x) ≠ 0) : 
  ∫ x, f x * log (f x / g x) ∂MeasureTheory.measureSpace.volume ≥ 0 := 
sorry

end information_inequality_holds_l734_734017


namespace crackers_per_person_l734_734427

theorem crackers_per_person:
  ∀ (total_crackers friends : ℕ), total_crackers = 36 → friends = 18 → total_crackers / friends = 2 :=
by
  intros total_crackers friends h1 h2
  sorry

end crackers_per_person_l734_734427


namespace capacity_ratio_l734_734036

noncomputable def radius_from_circumference (circumference : ℝ) : ℝ :=
  circumference / (2 * Real.pi)

noncomputable def volume_of_cylinder (radius height : ℝ) : ℝ :=
  Real.pi * radius^2 * height

theorem capacity_ratio :
  let hA := 7
      cA := 8
      hB := 8
      cB := 10
      rA := radius_from_circumference cA
      rB := radius_from_circumference cB
      VA := volume_of_cylinder rA hA
      VB := volume_of_cylinder rB hB
  in (VA / VB) = 0.56 :=
by
  sorry

end capacity_ratio_l734_734036


namespace subset_count_with_even_number_l734_734750

theorem subset_count_with_even_number :
  let S := {1, 2, 3, 4}
  ∃ A ⊆ S, (∃ x ∈ A, x % 2 = 0) ∧ (set.finite S) ∧ (card {A | A ⊆ S ∧ ∃ x ∈ A, x % 2 = 0} = 12) :=
sorry

end subset_count_with_even_number_l734_734750


namespace calculate_expression_l734_734691

theorem calculate_expression :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = (3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end calculate_expression_l734_734691
