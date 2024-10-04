import Mathlib
import Mathlib.Algebra.ArithmeticSeries
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Probability
import Mathlib.Tactic.Basic

namespace primes_between_50_and_60_l400_400601

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400601


namespace primes_between_50_and_60_l400_400638

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400638


namespace find_x_in_isosceles_triangle_l400_400700

def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)

def triangle_inequality (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem find_x_in_isosceles_triangle (x : ℝ) :
  is_isosceles (x + 3) (2 * x + 1) 11 ∧ triangle_inequality (x + 3) (2 * x + 1) 11 →
  (x = 8) ∨ (x = 5) :=
sorry

end find_x_in_isosceles_triangle_l400_400700


namespace wilson_theorem_l400_400971

-- Definition of Wilson's Theorem for a prime number.
theorem wilson_theorem (p : ℕ) [hp : fact (nat.prime p)] : (p - 1)! ≡ -1 [ZMOD p] :=
sorry

-- Given conditions: 17 is a prime number and we need to show 14! ≡ 8 [ZMOD 17]
example : (14! : ℤ) ≡ 8 [ZMOD 17] :=
  by
  have prime_17 : nat.prime 17 := by norm_num
  have fact_prime_17 : fact (nat.prime 17) := ⟨prime_17⟩
  -- Use Wilson's Theorem
  have wilson_result := wilson_theorem 17
  sorry

end wilson_theorem_l400_400971


namespace primes_between_50_and_60_l400_400536

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400536


namespace find_line_equation_l400_400129

variable (x y : ℝ)

theorem find_line_equation (hx : x = -5) (hy : y = 2)
  (line_through_point : ∃ a b c : ℝ, a * x + b * y + c = 0)
  (x_intercept_twice_y_intercept : ∀ a b c : ℝ, c ≠ 0 → b ≠ 0 → (a / c) = 2 * (c / b)) :
  ∃ a b c : ℝ, (a * x + b * y + c = 0 ∧ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 1 ∧ b = 2 ∧ c = 1)) :=
sorry

end find_line_equation_l400_400129


namespace alcohol_percentage_in_original_mixture_l400_400883

/-- A 20 litres mixture contains a certain percentage of alcohol and the rest water.
If 3 litres of water are mixed with it, the percentage of alcohol in the new mixture is 17.391304347826086%.
Prove that the percentage of alcohol in the original mixture was 20%. -/
theorem alcohol_percentage_in_original_mixture:
  ∀ (A : ℝ), 
    let original_volume := 20 in
    let added_water := 3 in
    let new_volume := original_volume + added_water in
    let new_percentage := 17.391304347826086 in
    (A / 100) * original_volume / new_volume * 100 = new_percentage → 
    A = 20 :=
by sorry

end alcohol_percentage_in_original_mixture_l400_400883


namespace num_primes_between_50_and_60_l400_400614

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400614


namespace value_after_increase_l400_400065

-- Definition of original number and percentage increase
def original_number : ℝ := 600
def percentage_increase : ℝ := 0.10

-- Theorem stating that after a 10% increase, the value is 660
theorem value_after_increase : original_number * (1 + percentage_increase) = 660 := by
  sorry

end value_after_increase_l400_400065


namespace primes_between_50_and_60_l400_400425

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400425


namespace ellipse_problems_l400_400175

noncomputable def ellipse_equation : Prop :=
  let a := sqrt 2 in
  let b := 1 in
  let c := 1 in
  (a^2 = 2) ∧ (b^2 = 1) ∧ (c = 1) ∧ (a = sqrt 2 * b) → 
  (∀ {x y : ℝ}, (x^2) / 2 + y^2 = 1 → true)

noncomputable def min_lambda (P : ℝ × ℝ) (F : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  let P := (2, 0) in
  let F := (if 2 > 0 then -1 else 1, 0) in
  let f (A B: (ℝ × ℝ)) := ((A.1 - 2) * (B.1 - 2) + A.2 * B.2) in
  (∀ l, (∃ A B, line passes through F and intersects ellipse C at A and B) → 
  (f A B ≤ 17 / 2)) ∧ ∃ λ ∈ ℝ, λ = 17 / 2

-- Main theorem combining both parts
theorem ellipse_problems : Prop :=
  ellipse_equation ∧ min_lambda (2, 0) (-1, 0) (0, 0) (0, 0)

end ellipse_problems_l400_400175


namespace a14_eq_33_l400_400709

variable {a : ℕ → ℝ}
variables (d : ℝ) (a1 : ℝ)

-- Defining the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℝ := a1 + n * d

-- Given conditions
axiom a5_eq_6 : arithmetic_sequence 4 = 6
axiom a8_eq_15 : arithmetic_sequence 7 = 15

-- Theorem statement
theorem a14_eq_33 : arithmetic_sequence 13 = 33 :=
by
  -- Proof skipped
  sorry

end a14_eq_33_l400_400709


namespace remaining_to_be_paid_is_2625_l400_400754

-- Define the conditions
def part_payment : ℝ := 875
def percentage_paid : ℝ := 0.25
def total_cost : ℝ := part_payment / percentage_paid
def remaining_amount : ℝ := total_cost - part_payment

-- State the theorem to prove
theorem remaining_to_be_paid_is_2625 : remaining_amount = 2625 := 
by
  sorry

end remaining_to_be_paid_is_2625_l400_400754


namespace rhombus_area_l400_400838

theorem rhombus_area (a : ℝ) (theta : ℝ) (h_side_length : a = 3) (h_angle : theta = π / 4) :
  let area := a * a * sin(theta)
  in area = 9 * Real.sqrt 2 / 2 :=
by sorry

end rhombus_area_l400_400838


namespace trapezoid_perimeter_l400_400877

def isosceles_trapezoid (A B C D : ℝ) : Prop :=
  A = D ∧ B = C

def perimeter_of_isosceles_trapezoid (AD BC h x : ℝ) : ℝ :=
  AD + BC + 2 * x

theorem trapezoid_perimeter :
  ∀ (A B C D : ℝ), isosceles_trapezoid A B C D →
  AD = 98 → BC = 62 → h = 21 →
  let x := Real.sqrt (18^2 + 21^2) in
  perimeter_of_isosceles_trapezoid AD BC h x = 160 + 2 * Real.sqrt 765 :=
by {
  intros A B C D h1 hAD hBC hh,
  let x := Real.sqrt (18^2 + 21^2),
  sorry
}

end trapezoid_perimeter_l400_400877


namespace sqrt_two_squared_l400_400044

theorem sqrt_two_squared : (real.sqrt 2) ^ 2 = 2 := by
  sorry

end sqrt_two_squared_l400_400044


namespace pete_does_not_need_to_return_any_bottles_l400_400007

-- Definitions of conditions
def owes : ℕ := 90
def twenty_bill_count : ℕ := 2
def twenty_bill_value : ℕ := 20
def ten_bill_count : ℕ := 5
def ten_bill_value : ℕ := 10
def pounds_value : ℕ := 7

-- Calculating total money Pete has
def total_money : ℕ :=
  twenty_bill_count * twenty_bill_value + ten_bill_count * ten_bill_value + pounds_value

-- Definition of store's bottle return rate
def bottle_rate : ℕ := 25 -- in cents

-- Problem statement
theorem pete_does_not_need_to_return_any_bottles (h : owes ≤ total_money) : 0 = 0 :=
by {
  have h1 : owes = 90 := rfl,
  have h2 : total_money = 97 := rfl,
  have h3 : owes ≤ total_money := by linarith,
  apply h,
}

end pete_does_not_need_to_return_any_bottles_l400_400007


namespace part1_part2_distribution_part2_expected_value_l400_400014

noncomputable def prob_A_win : ℝ := 2 / 3
noncomputable def prob_B_win : ℝ := 1 / 3

def prob_B_one_A_win_match : ℝ :=
  (prob_B_win * prob_A_win * prob_A_win) +
  (prob_A_win * prob_B_win * prob_A_win * prob_A_win)

theorem part1 : prob_B_one_A_win_match = 20 / 81 := 
  sorry

def prob_X (n : ℕ) : ℝ :=
  match n with
  | 2 => (prob_A_win * prob_A_win) + (prob_B_win * prob_B_win)
  | 3 => 2 * (prob_B_win * prob_A_win * prob_A_win)
  | 4 => 2 * (prob_A_win * prob_B_win * prob_A_win * prob_A_win)
  | 5 => 2 * (prob_B_win * prob_A_win * prob_B_win * prob_A_win)
  | _ => 0

theorem part2_distribution :
  prob_X 2 = 5 / 9 ∧ 
  prob_X 3 = 2 / 9 ∧ 
  prob_X 4 = 10 / 81 ∧ 
  prob_X 5 = 8 / 81 := 
  sorry

def expected_value_X : ℝ :=
  2 * prob_X 2 + 3 * prob_X 3 + 4 * prob_X 4 + 5 * prob_X 5

theorem part2_expected_value :
  expected_value_X = 224 / 81 :=
  sorry

end part1_part2_distribution_part2_expected_value_l400_400014


namespace watch_correction_needed_l400_400078

def watch_loses_rate : ℚ := 15 / 4  -- rate of loss per day in minutes
def initial_set_time : ℕ := 15  -- March 15th at 10 A.M.
def report_time : ℕ := 24  -- March 24th at 4 P.M.
def correction (loss_rate per_day min_hrs : ℚ) (days_hrs : ℚ) : ℚ :=
  (days_hrs * (loss_rate / (per_day * min_hrs)))

theorem watch_correction_needed :
  correction watch_loses_rate 24 60 (222) = 34.6875 := 
sorry

end watch_correction_needed_l400_400078


namespace polynomial_equal_terms_l400_400820

noncomputable def polynomial := (x + y) ^ 10

theorem polynomial_equal_terms (p q : ℝ) (h : p + q = 1) :
  (∃ p q : ℝ, p + q = 1 ∧ (binomial 10 2) * p^8 * q^2 = (binomial 10 3) * p^7 * q^3) →
  p = 8 / 11 :=
by
  sorry

end polynomial_equal_terms_l400_400820


namespace primes_between_50_and_60_l400_400236

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400236


namespace distance_traveled_correct_gas_station_times_l400_400693

noncomputable def s (t : ℝ) : ℝ :=
if 0 ≤ t ∧ t ≤ 3 then -5 * t * (t - 13)
else if 3 < t ∧ t ≤ 8 then 150
else if 8 < t ∧ t ≤ 10.5 then 60 * t - 330
else 0 -- default value for t outside the range

theorem distance_traveled_correct {t : ℝ} :
  (0 ≤ t ∧ t ≤ 3 → s(t) = -5 * t * (t - 13)) ∧
  (3 < t ∧ t ≤ 8 → s(t) = 150) ∧
  (8 < t ∧ t ≤ 10.5 → s(t) = 60 * t - 330) :=
by sorry

theorem gas_station_times : ∃ t1 t2, s t1 = 60 ∧ s t2 = 60 ∧
  (0 ≤ t1 ∧ t1 ≤ 3 ∧ 8 < t2 ∧ t2 ≤ 10.5) ∧
  t1 = 1 ∧ t2 = 9.5 :=
by sorry

end distance_traveled_correct_gas_station_times_l400_400693


namespace percentage_change_fall_to_spring_l400_400868

theorem percentage_change_fall_to_spring :
  ∀ (X : ℝ), (X > 0) → let fall_members := X * 1.04 in
  let spring_members := fall_members * 0.81 in
  let percentage_change := ((spring_members - X) / X) * 100 in
  percentage_change = -15.76 :=
by
  intros X hX
  let fall_members := X * 1.04
  let spring_members := fall_members * 0.81
  let percentage_change := ((spring_members - X) / X) * 100
  sorry

end percentage_change_fall_to_spring_l400_400868


namespace system_solution_l400_400792

theorem system_solution (x y : ℝ) : 
  (3^y * 81 = 9^(x^2)) ∧ (Real.log10 y = Real.log10 x - Real.log10 0.5) → 
  x = 2 ∧ y = 4 :=
by
  sorry

end system_solution_l400_400792


namespace no_x_intercepts_in_interval_l400_400135

noncomputable def x_intercepts_in_interval := 0

theorem no_x_intercepts_in_interval :
  ∀ x ∈ (set.Ioo 0.00005 0.0005), cos (1 / x) ≠ 0 :=
begin
  intros x hx,
  let k := floor ((2 / (π * x) - 1) / 2),
  have h1 : 0.00005 < x := hx.1,
  have h2 : x < 0.0005 := hx.2,
  have h_bdd : 636.82 < k → k < 6368.85,
  {
    sorry, -- Steps to show that no integer k satisfies the range bounds.
  },
  -- The result follows from the range of possible k values.
  sorry
end

end no_x_intercepts_in_interval_l400_400135


namespace primes_between_50_and_60_l400_400643

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400643


namespace avg_weight_all_boys_l400_400871

def avg_weight_class (weight1 : ℕ → ℚ) (weight2 : ℕ → ℚ) : ℚ :=
  let total_weight1 := 16 * (weight1 16)
  let total_weight2 := 8 * (weight2 8)
  let total_weight := total_weight1 + total_weight2
  total_weight / 24

theorem avg_weight_all_boys :
  avg_weight_class (λ _, 50.25) (λ _, 45.15) = 48.55 := by
  sorry

end avg_weight_all_boys_l400_400871


namespace five_points_opposite_sides_l400_400749

theorem five_points_opposite_sides:
  ∀ (A B C D E : ℝ × ℝ),
  (¬ (∃ collinear : fin 3 → ℝ × ℝ, ∀ i, ∃ j, collinear i = [A, B, C, D, E].nth j) ∧ 
   ¬ (∃ cocyclic : fin 4 → ℝ × ℝ, ∀ i, ∃ j, cocyclic i = [A, B, C, D, E].nth j)) →
  ∃ (P Q R : ℝ × ℝ) (X Y : ℝ × ℝ), 
    (X ≠ P ∧ X ≠ Q ∧ X ≠ R) ∧ (Y ≠ P ∧ Y ≠ Q ∧ Y ≠ R) ∧ 
    opposite_side_of_circle P Q R X Y :=
sorry

end five_points_opposite_sides_l400_400749


namespace count_primes_between_50_and_60_l400_400467

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400467


namespace count_primes_50_60_l400_400491

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400491


namespace find_a_l400_400205

theorem find_a (a : ℝ) (x y : ℝ) 
    (h1 : 4 * x + 3 * y = 10) 
    (h2 : 2 * x - y = 10) 
    (h3 : a * x + 2 * y + 8 = 0) : 
    a = -1 :=
begin
  -- proof goes here
  sorry
end

end find_a_l400_400205


namespace count_primes_between_50_and_60_l400_400451

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400451


namespace percentage_apples_sold_l400_400055

theorem percentage_apples_sold
  (original_apples : ℝ)
  (remaining_apples : ℝ)
  (num_sold : ℝ)
  (percentage_sold : ℝ) :
  original_apples = 2499.9987500006246 →
  remaining_apples = 500 →
  num_sold = (original_apples - remaining_apples) →
  percentage_sold = (num_sold / original_apples) * 100 →
  percentage_sold = 80 :=
by
  intros h_original h_remaining h_num_sold h_percentage_sold
  rw [h_original, h_remaining] at h_num_sold
  rw [h_num_sold] at h_percentage_sold
  linarith

end percentage_apples_sold_l400_400055


namespace rect_inscribed_circle_perpendicular_projections_l400_400710

-- Define the problem setting and theorem to be proved
theorem rect_inscribed_circle_perpendicular_projections
  (A B C D : ℂ) -- A, B, C, D are points on the complex plane representing the rectangle
  (M P Q R S : ℂ) -- M is on the arc AB and P, Q, R, S are projections of M
  : (x : ℂ) -- Additional parameters can be added as required
    (ABCD_is_rectangle : 
  ((A - B) * conj(A - B)) = ((A - D) * conj(A - D)) ∧ ((B - C) * conj(B - C)) = ((D - C) * conj(D - C)) ) -- Rectangle equality conditions
  (M_on_arc : ℂ) -- M is a point on arc AB distinct from A and B, additional conditions can be specified 
  (P_proj_AD : P = complex.proj (A - D)) -- Projections on the line AD
  (Q_proj_AB : Q = complex.proj (A - B)) -- Projections on the line AB
  (R_proj_BC : R = complex.proj (B - C)) -- Projections on the line BC
  (S_proj_CD : S = complex.proj (C - D)) -- Projections on the line CD
  :
  -- Proof that lines PQ and RS are perpendicular
  (PQ_dot_RS : (P - Q) * conj(P - Q) = 0 ∧ (R - S) * conj(R - S) = 0) :=
  sorry

end rect_inscribed_circle_perpendicular_projections_l400_400710


namespace bisecting_circle_diameter_correct_l400_400056

noncomputable def bisecting_circle_diameter (R r : ℝ) : ℝ :=
  2 * (Real.root3 ((R^3 + r^3) / 2))

theorem bisecting_circle_diameter_correct (R r : ℝ) :
  ∃ ζ,  ζ = bisecting_circle_diameter R r ∧ ζ = 2 * (Real.root3 ((R^3 + r^3) / 2)) := by
  sorry

end bisecting_circle_diameter_correct_l400_400056


namespace gain_percent_is_66_67_l400_400038

variables (C S : ℝ)

-- Conditions
def condition1 : Prop := 50 * C = 30 * S

def gain_percent (C S : ℝ) : ℝ := ((S - C) / C) * 100

-- Question: Prove the gain percent
theorem gain_percent_is_66_67 (h : 50 * C = 30 * S) : gain_percent C S = 66.67 := by
  sorry

end gain_percent_is_66_67_l400_400038


namespace count_primes_between_50_and_60_l400_400256

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400256


namespace prime_count_between_50_and_60_l400_400572

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400572


namespace count_primes_between_50_and_60_l400_400243

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400243


namespace area_of_triangle_PQR_l400_400926

-- Define the vertices P, Q, and R
def P : (Int × Int) := (-3, 2)
def Q : (Int × Int) := (1, 7)
def R : (Int × Int) := (3, -1)

-- Define the formula for the area of a triangle given vertices
def triangle_area (A B C : Int × Int) : Real :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Define the statement to prove
theorem area_of_triangle_PQR : triangle_area P Q R = 21 := 
  sorry

end area_of_triangle_PQR_l400_400926


namespace ellipse_standard_equation_min_lambda_inequality_l400_400177

noncomputable def ellipse_equation :=
  ∃ a b : ℝ, a > b > 0 ∧ 2 * b * a = b^2 + 1 ∧ (λ (x y: ℝ), x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a = sqrt 2 ∧ b = 1 ∧ (λ (x y: ℝ), x^2 / 2 + y^2 = 1) :=
sorry

def PA_dot_PB (A B : ℝ × ℝ) : ℝ := (A.1 - 2, A.2) • (B.1 - 2, B.2)

theorem min_lambda_inequality :
  ∃ λ : ℝ, (∀ A B : ℝ × ℝ, PA_dot_PB A B ≤ λ) ∧ λ = 17 / 2 :=
sorry

end ellipse_standard_equation_min_lambda_inequality_l400_400177


namespace bankers_gain_correct_l400_400801

-- Given conditions
def BD : ℝ := 340
def r : ℝ := 12
def t : ℝ := 3

-- Definition of Banker's Gain
def BG (BD r t : ℝ) : ℝ :=
  (BD * r * t) / (100 + (r * t))

-- Theorem statement
theorem bankers_gain_correct : BG BD r t = 90 := 
  by
  sorry

end bankers_gain_correct_l400_400801


namespace volume_of_cone_l400_400687

theorem volume_of_cone :
  let l := 2,
      r := 1,
      h := Real.sqrt (l ^ 2 - r ^ 2),
      S_base := Real.pi * r ^ 2,
      V_cone := (1 / 3) * S_base * h in
  (1 / 2) * Real.pi * l ^ 2 = 2 * Real.pi →
  (1 / 2) * 2 * Real.pi * l = 2 * Real.pi →
  V_cone = (Real.sqrt 3 / 3) * Real.pi 
:= by
  intros
  sorry

end volume_of_cone_l400_400687


namespace correct_answer_is_C_l400_400861

-- Define the conditions
def reasoning1 : Prop := "Inferring the properties of a sphere from the properties of a circle."
def reasoning2 : Prop := "Sum of interior angles generalization from specific triangles to all triangles."
def reasoning3 : Prop := "Inference of all students' scores from one student's 100 points."
def reasoning4 : Prop := "Sum of interior angles of convex polygon generalized from triangles, quadrilaterals, and pentagons."

-- Define the problem statement
def logic_answer_correct : Prop :=
  (reasoning1 ∧ reasoning2 ∧ reasoning4) ∧ ¬ reason3

-- Assert the proof goal that (C) is the correct answer
theorem correct_answer_is_C : logic_answer_correct :=
  sorry

end correct_answer_is_C_l400_400861


namespace primes_between_50_and_60_l400_400647

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400647


namespace option_C_is_polynomial_l400_400029

def is_polynomial (expr : Expr) : Prop := sorry -- Define polynomial condition here properly

-- Define each expression
def expr_C1 := a - b / 3
def expr_C2 := -2 * b / 7
def expr_C3 := 3 * a - 5 * b / 2

-- The proof statement
theorem option_C_is_polynomial :
  is_polynomial expr_C1 ∧ is_polynomial expr_C2 ∧ is_polynomial expr_C3 :=
sorry

end option_C_is_polynomial_l400_400029


namespace num_primes_between_50_and_60_l400_400609

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400609


namespace num_primes_between_50_and_60_l400_400610

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400610


namespace tom_total_amount_after_saving_l400_400005

theorem tom_total_amount_after_saving :
  let hourly_rate := 6.50
  let work_hours := 31
  let saving_rate := 0.10
  let total_earnings := hourly_rate * work_hours
  let amount_set_aside := total_earnings * saving_rate
  let amount_for_purchases := total_earnings - amount_set_aside
  amount_for_purchases = 181.35 :=
by
  sorry

end tom_total_amount_after_saving_l400_400005


namespace problem_part1_problem_part2_area_height_l400_400163

theorem problem_part1 (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) : 
  x * y ^ 2 - x ^ 2 * y = -32 * Real.sqrt 2 := 
  sorry

theorem problem_part2_area_height (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) :
  let side_length := Real.sqrt 12
  let area := (1 / 2) * x * y
  let height := area / side_length
  area = 4 ∧ height = (2 * Real.sqrt 3) / 3 := 
  sorry

end problem_part1_problem_part2_area_height_l400_400163


namespace primes_between_50_and_60_l400_400646

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400646


namespace podcast_distribution_l400_400904

theorem podcast_distribution 
  (total_content : ℕ) 
  (max_per_day : ℕ)
  (total_minutes : total_content = 500)
  (max_daily_minutes : max_per_day = 65) :
  ∃ (days : ℕ) (minutes_per_day : ℕ), 
    days = 8 ∧ minutes_per_day = 62.5 ∧ total_content ≤ days * max_per_day ∧ 
    total_content / days = minutes_per_day := 
sorry

end podcast_distribution_l400_400904


namespace primes_between_50_and_60_l400_400408

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400408


namespace minimum_even_numbers_in_circle_l400_400755

theorem minimum_even_numbers_in_circle (circle : Fin 103 → ℕ) 
    (h1 : ∀ (i : Fin 103), ∃ j k : Fin 5, 0 ≤ j.k < 5 ∧ j ≠ k ∧ circle ((i + j) % 103) % 2 = 0 ∧ circle ((i + k) % 103) % 2 = 0) : 
    ∃ (n : ℕ), n ≥ 42 ∧ ∃ s : Finset (Fin 103), (s.card = n) ∧ ∀ x ∈ s, circle x % 2 = 0 :=
sorry

end minimum_even_numbers_in_circle_l400_400755


namespace primes_between_50_and_60_l400_400400

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400400


namespace largest_among_three_l400_400084

theorem largest_among_three :
  (2:ℝ)^(1/3) > (1/3:ℝ)^(0.2) ∧ (2:ℝ)^(1/3) > Real.log 3 / Real.log (1/2) :=
by
  -- Add the necessary mathematically equivalent proof steps here
  sorry

end largest_among_three_l400_400084


namespace count_primes_between_50_and_60_l400_400662

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400662


namespace primes_between_50_and_60_l400_400319

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400319


namespace num_primes_between_50_and_60_l400_400618

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400618


namespace problem_sum_of_digits_N_l400_400815

noncomputable def N : ℕ := (30^36) * (2^50)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem problem_sum_of_digits_N :
  N^2 = 36^50 * 50^36 ∧
  sum_of_digits N = 10 :=
by
  sorry

end problem_sum_of_digits_N_l400_400815


namespace river_current_speed_l400_400062

noncomputable section

variables {d r w : ℝ}

def time_equation_normal_speed (d r w : ℝ) : Prop :=
  (d / (r + w)) + 4 = (d / (r - w))

def time_equation_tripled_speed (d r w : ℝ) : Prop :=
  (d / (3 * r + w)) + 2 = (d / (3 * r - w))

theorem river_current_speed (d r : ℝ) (h1 : time_equation_normal_speed d r w) (h2 : time_equation_tripled_speed d r w) : w = 2 :=
sorry

end river_current_speed_l400_400062


namespace primes_between_50_and_60_l400_400221

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400221


namespace count_primes_between_50_and_60_l400_400262

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400262


namespace geometric_sequence_sufficient_condition_l400_400985

theorem geometric_sequence_sufficient_condition 
  (a_1 : ℝ) (q : ℝ) (h_a1 : a_1 < 0) (h_q : 0 < q ∧ q < 1) :
  ∀ n : ℕ, n > 0 -> a_1 * q^(n-1) < a_1 * q^n :=
sorry

end geometric_sequence_sufficient_condition_l400_400985


namespace primes_between_50_and_60_l400_400298

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400298


namespace original_acid_percentage_l400_400915

-- Let x be the initial ounces of water and y be the initial ounces of acid.
variables (x y : ℕ)

-- Condition 1: After adding 1 ounce of water, the new mixture has 20% acid.
def condition1 : Prop :=
  y / (x + y + 1) = 1 / 5

-- Condition 2: After adding 1 ounce of acid to the resulting mixture, the acid concentration becomes 33⅓%.
def condition2 : Prop :=
  (y + 1) / (x + y + 2) = 1 / 3

-- Prove that the percentage of acid in the original mixture is 25%
theorem original_acid_percentage : condition1 x y ∧ condition2 x y → 
  (y / (x + y) = 1 / 4) :=
begin
  sorry
end

end original_acid_percentage_l400_400915


namespace solve_quadratic_inequality_l400_400951

theorem solve_quadratic_inequality :
    {x : ℝ | x^2 - x - 30 < 0} = set.Ioo (-5) 6 :=
by
  sorry

end solve_quadratic_inequality_l400_400951


namespace larger_acute_angle_right_triangle_l400_400697

theorem larger_acute_angle_right_triangle (x : ℝ) (hx : 0 < x) (h_triangle : x + 2 * x + 90 = 180) :
  2 * x = 60 :=
by
  have h1 : 3 * x = 90 := by linarith,
  have h2 : x = 30 := by linarith,
  show 2 * x = 60, by linarith

end larger_acute_angle_right_triangle_l400_400697


namespace count_primes_between_50_and_60_l400_400672

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400672


namespace line_equation_through_point_with_intercepts_conditions_l400_400131

theorem line_equation_through_point_with_intercepts_conditions :
  ∃ (a b : ℚ) (m c : ℚ), 
    (-5) * m + c = 2 ∧ -- The line passes through A(-5, 2)
    a = 2 * b ∧       -- x-intercept is twice the y-intercept
    (a * m + c = 0 ∨ ((1/m)*a + (1/m)^2 * c+1 = 0)) :=         -- Equations of the line
sorry

end line_equation_through_point_with_intercepts_conditions_l400_400131


namespace primes_between_50_and_60_l400_400649

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400649


namespace primes_between_50_and_60_l400_400405

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400405


namespace angle_between_OB_OC_tan_alpha_given_perpendicular_l400_400181

noncomputable def points_and_conditions (α: ℝ) :=
  let A := (2, 0)
  let B := (0, 2)
  let C := (Real.cos α, Real.sin α)
  ∃ O : (ℝ × ℝ) (hOA_hOC : 0 < α ∧ α < π ∧ 
    let vec_OA := (2, 0)
    let vec_OC := (Real.cos α, Real.sin α)
    let sum_vec := (2 + Real.cos α, Real.sin α)
    ∥sum_vec∥ = sqrt 7),
  True

theorem angle_between_OB_OC (α: ℝ) (cond : points_and_conditions α) : 
  let B := (0, 2)
  let C := (Real.cos α, Real.sin α)
  ∃ θ : ℝ, θ = π / 6 := sorry

theorem tan_alpha_given_perpendicular (α: ℝ) (cond : points_and_conditions α) :
  let A := (2, 0)
  let B := (0, 2)
  let C := (Real.cos α, Real.sin α)
  (let AC := (Real.cos α - 2, Real.sin α)
  let BC := (Real.cos α, Real.sin α - 2)
  AC.1 * BC.1 + AC.2 * BC.2 = 0) →
  tan (α : ℝ) = - (4 + sqrt 7) / 3 := sorry

end angle_between_OB_OC_tan_alpha_given_perpendicular_l400_400181


namespace monomial_2024_l400_400910

def monomial (n : ℕ) : ℤ × ℕ := ((-1)^(n + 1) * (2 * n - 1), n)

theorem monomial_2024 :
  monomial 2024 = (-4047, 2024) :=
sorry

end monomial_2024_l400_400910


namespace max_xy_on_line_segment_A_B_l400_400993

noncomputable def maxValueOnLineSegment (A B : ℝ × ℝ) := A.1 * B.2 / (A.1 + B.2)

theorem max_xy_on_line_segment_A_B :
  let A := (2 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 1 : ℝ)
  let P := (x, y) : ℝ × ℝ
  (P ∈ line_segment ℝ A B) → x * y ≤ 1 / 2 :=
by
  sorry

end max_xy_on_line_segment_A_B_l400_400993


namespace prime_count_between_50_and_60_l400_400568

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400568


namespace graham_crackers_for_cheesecake_l400_400751

theorem graham_crackers_for_cheesecake
  (initial_graham_boxes : ℕ)
  (initial_oreo_packets : ℕ)
  (oreo_per_cheesecake : ℕ)
  (remaining_graham_boxes : ℕ)
  (cheesecakes_made : ℕ)
  (graham_used : ℕ) :
  initial_graham_boxes = 14 →
  initial_oreo_packets = 15 →
  ore_per_cheesecake = 3 →
  remaining_graham_boxes = 4 →
  cheesecakes_made = initial_oreo_packets / ore_per_cheesecake →
  graham_used = initial_graham_boxes - remaining_graham_boxes →
  graham_per_cheesecake = graham_used / cheesecakes_made →
  graham_per_cheesecake = 2 :=
begin
  intros,
  sorry
end

end graham_crackers_for_cheesecake_l400_400751


namespace primes_between_50_and_60_l400_400237

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400237


namespace existence_of_same_remainder_mod_36_l400_400726

theorem existence_of_same_remainder_mod_36
  (a : Fin 7 → ℕ) :
  ∃ (i j k l : Fin 7), i < j ∧ k < l ∧ (a i)^2 + (a j)^2 % 36 = (a k)^2 + (a l)^2 % 36 := by
  sorry

end existence_of_same_remainder_mod_36_l400_400726


namespace number_of_tangent_lines_l400_400735

def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 + a

def on_line (a x y : ℝ) : Prop := 3 * x + y = a + 1

theorem number_of_tangent_lines (a m : ℝ) (h1 : on_line a m (a + 1 - 3 * m)) :
  ∃ n : ℤ, n = 1 ∨ n = 2 :=
sorry

end number_of_tangent_lines_l400_400735


namespace primes_between_50_and_60_l400_400591

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400591


namespace primes_between_50_and_60_l400_400598

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400598


namespace a_and_b_work_together_l400_400865
noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem a_and_b_work_together (A_days B_days : ℕ) (hA : A_days = 32) (hB : B_days = 32) :
  (1 / work_rate A_days + 1 / work_rate B_days) = 16 := by
  sorry

end a_and_b_work_together_l400_400865


namespace problem_statement_l400_400733

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 1988 a b α β = 3) : f 2013 a b α β = 5 :=
by 
  sorry

end problem_statement_l400_400733


namespace nutty_buddy_days_l400_400970

theorem nutty_buddy_days :
  (∀ n, n ∈ {1, 2, 3, 4, 5} → ∀ w, w ∈ {1, 2, 3, ..., 6} →
    (if n = 1 ∨ n = 3 ∨ n = 5 then cost(w, n) = 2 else
    if n = 2 ∨ n = 4 then cost(w, n) = 1.5 else
    cost(w, n) = 0)) ∧
  (∑ w in {1, 2, 3, ..., 6}, ∑ n in {1, 2, 3, 4, 5, 6, 7}, cost(w, n) = 90) →
  (∀ w, w ∈ {1, 2, 3, ..., 6} → cost(w, 6) + cost(w, 7) = 0 ∨ cost(w, 6) = 3 ∧ cost(w, 7) = 3) :=
by
  sorry

end nutty_buddy_days_l400_400970


namespace primes_between_50_and_60_l400_400387

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400387


namespace count_primes_between_50_and_60_l400_400462

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400462


namespace primes_between_50_and_60_l400_400445

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400445


namespace primes_between_50_and_60_l400_400443

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400443


namespace count_primes_between_50_and_60_l400_400257

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400257


namespace maximize_profit_l400_400027

noncomputable def selling_price_to_maximize_profit (original_price selling_price : ℝ) (units units_sold_decrease : ℝ) : ℝ :=
  let x := 5
  let optimal_selling_price := selling_price + x
  optimal_selling_price

theorem maximize_profit :
  selling_price_to_maximize_profit 80 90 400 20 = 95 :=
by
  sorry

end maximize_profit_l400_400027


namespace carA_speed_proof_l400_400804

/-- Define the given conditions -/
variables (distance time : ℝ) (carA_speed carB_speed : ℝ)
variables (h_distance : distance = 450) (h_time : time = 3)
variables (h_carA_twice_carB : carA_speed = 2 * carB_speed)
variables (h_meeting_condition : carA_speed + carB_speed = distance / time)

/-- The statement to be proven -/
theorem carA_speed_proof : carA_speed = 100 :=
by
  -- This is a placeholder to indicate the proof is to be completed
  sorry

end carA_speed_proof_l400_400804


namespace primes_between_50_and_60_l400_400391

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400391


namespace primes_between_50_and_60_l400_400636

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400636


namespace primes_between_50_and_60_l400_400427

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400427


namespace olympic_selections_l400_400153

theorem olympic_selections :
  let male_athletes := 4
  let female_athletes := 5
  let total_selections := (choose male_athletes 1) * (choose female_athletes 2) + (choose male_athletes 2) * (choose female_athletes 1)
  total_selections = 70 :=
by
  sorry

end olympic_selections_l400_400153


namespace count_primes_between_50_and_60_l400_400669

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400669


namespace max_area_of_rect_D_l400_400918

theorem max_area_of_rect_D (perimeter_A perimeter_B perimeter_C : ℕ) 
  (hA : perimeter_A = 10) 
  (hB : perimeter_B = 12)
  (hC : perimeter_C = 14) : 
  ∃ (D : ℕ), D = 16 :=
by
  use 16
  sorry

end max_area_of_rect_D_l400_400918


namespace number_of_primes_between_50_and_60_l400_400364

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400364


namespace minutes_before_4_angle_same_as_4_l400_400215

def hour_hand_angle_at_4 := 120
def minute_hand_angle_at_4 := 0
def minute_hand_angle_per_minute := 6
def hour_hand_angle_per_minute := 0.5

theorem minutes_before_4_angle_same_as_4 :
  ∃ m : ℚ, abs (hour_hand_angle_at_4 - 5.5 * m) = hour_hand_angle_at_4 ∧ 
           (60 - m) = 21 + 9 / 11 := by
  sorry

end minutes_before_4_angle_same_as_4_l400_400215


namespace count_primes_between_50_and_60_l400_400459

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400459


namespace expression_B_between_2_and_3_l400_400977

variable (a b : ℝ)
variable (h : 3 * a = 5 * b)

theorem expression_B_between_2_and_3 : 2 < (|a + b| / b) ∧ (|a + b| / b) < 3 :=
by sorry

end expression_B_between_2_and_3_l400_400977


namespace distance_from_home_to_school_l400_400944

theorem distance_from_home_to_school
  (x y : ℝ)
  (h1 : x = y / 3)
  (h2 : x = (y + 18) / 5) : x = 9 := 
by
  sorry

end distance_from_home_to_school_l400_400944


namespace num_primes_between_50_and_60_l400_400626

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400626


namespace square_area_eq_fourty_nine_l400_400819

theorem square_area_eq_fourty_nine :
  ∀ (x1 y1 x2 y2 : ℝ),
  x1 = -4 → y1 = 1 → x2 = 3 → y2 = -6 →
  let d := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) in
  let s := d / real.sqrt 2 in
  s ^ 2 = 49 :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2
  dsimp only
  rw [hx1, hy1, hx2, hy2]
  sorry

end square_area_eq_fourty_nine_l400_400819


namespace primes_between_50_and_60_l400_400441

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400441


namespace count_primes_between_50_and_60_l400_400258

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400258


namespace prime_count_between_50_and_60_l400_400566

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400566


namespace num_primes_50_60_l400_400281

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400281


namespace tan_double_angle_l400_400976

theorem tan_double_angle (α : ℝ) (h : (2 * sin α + cos α) / (sin α - cos α) = 3) :
  tan (2 * α) = -8 / 15 :=
  sorry

end tan_double_angle_l400_400976


namespace primes_between_50_and_60_l400_400289

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400289


namespace find_b_in_triangle_l400_400690

noncomputable def sin_30 : ℝ := 1 / 2
noncomputable def sin_15 : ℝ := (Real.sqrt 2 - Real.sqrt 6) / 4

theorem find_b_in_triangle (a : ℝ) (A C : ℝ)
  (ha : a = 2)
  (hA : A = Real.pi / 6) -- 30 degrees in radians
  (hC : C = 3 * Real.pi / 4) -- 135 degrees in radians
  : ∃ b : ℝ, b = (Real.sqrt 2 - Real.sqrt 6) / 2 :=
by
  have hB : B = Real.pi - A - C := by sorry
  have h_law_of_sines : (a / sin A) = (b / sin B) := by sorry
  exact ⟨(Real.sqrt 2 - Real.sqrt 6) / 2, by sorry⟩


end find_b_in_triangle_l400_400690


namespace primes_between_50_and_60_l400_400225

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400225


namespace prime_count_50_to_60_l400_400508

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400508


namespace determine_omega_phi_l400_400202

noncomputable def f : ℝ → ℝ := sorry

theorem determine_omega_phi (ω φ : ℝ) (h1 : 0 < ω)
  (h2 : -real.pi / 2 < φ) (h3 : φ < real.pi / 2)
  (h4 : ∀ x, f(x) = real.sin(x))
  (h5 : ∀ x, f(2 * (x - real.pi / 6)) = real.sin(x)) :
  ω = 1 / 2 ∧ φ = real.pi / 6 :=
begin
  sorry
end

end determine_omega_phi_l400_400202


namespace grid_filling_at_most_m_plus_n_minus_1_l400_400995

theorem grid_filling_at_most_m_plus_n_minus_1 (a : Fin m → ℝ) (b : Fin n → ℝ)
  (h1 : ∀ i, a i > 0) (h2 : ∀ j, b j > 0)
  (h_sum : ∑ i, a i = ∑ j, b j) :
  ∃ (x : Fin m → Fin n → ℝ), 
    (∀ i, ∑ j, x i j = a i) ∧ (∀ j, ∑ i, x i j = b j) ∧ 
    (∃ s, s.card ≤ m + n - 1 ∧ ∀ (i : Fin m) (j : Fin n), x i j > 0 ↔ (i, j) ∈ s) :=
by
  sorry

end grid_filling_at_most_m_plus_n_minus_1_l400_400995


namespace primes_between_50_and_60_l400_400238

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400238


namespace count_primes_between_50_and_60_l400_400664

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400664


namespace hyperbola_eccentricity_correct_l400_400987

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (θ_deg : ℝ) (hθ : θ_deg = 60) : ℝ :=
  let θ_rad := θ_deg * real.pi / 180
  let tan_θ := real.tan θ_rad
  let e := if tan_θ = b / a then real.sqrt (1 + (b / a)^2) else 0
  e

theorem hyperbola_eccentricity_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) (θ_deg : ℝ) (hθ : θ_deg = 60) :
  hyperbola_eccentricity a b ha hb θ_deg hθ = 2 ∨ hyperbola_eccentricity a b ha hb θ_deg hθ = 2 * real.sqrt 3 / 3 :=
sorry

end hyperbola_eccentricity_correct_l400_400987


namespace no_solution_cos4x_tan5x_eq_cot5x_sin6x_l400_400965

theorem no_solution_cos4x_tan5x_eq_cot5x_sin6x :
  ∀ x : ℝ, (-π < x ∧ x < π) → ¬ (cos (4 * x) - tan (5 * x) = cot (5 * x) - sin (6 * x)) :=
by
  sorry

end no_solution_cos4x_tan5x_eq_cot5x_sin6x_l400_400965


namespace num_primes_50_60_l400_400268

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400268


namespace value_of_a5_l400_400728

def sequence (a : ℕ → ℝ) : Prop :=
a 0 = 3 / 2 ∧
(∀ n, a (n + 1) = 2 * (finset.sum (finset.range (n + 1)) a) - 2^n)

theorem value_of_a5 (a : ℕ → ℝ) (h : sequence a) : a 4 = -11 :=
sorry

end value_of_a5_l400_400728


namespace primes_count_between_50_and_60_l400_400559

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400559


namespace primes_between_50_and_60_l400_400416

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400416


namespace find_sales_tax_percentage_l400_400067

noncomputable def salesTaxPercentage (price_with_tax : ℝ) (price_difference : ℝ) : ℝ :=
  (price_difference * 100) / (price_with_tax - price_difference)

theorem find_sales_tax_percentage :
  salesTaxPercentage 2468 161.46 = 7 := by
  sorry

end find_sales_tax_percentage_l400_400067


namespace primes_between_50_and_60_l400_400228

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400228


namespace change_in_nickels_l400_400752

theorem change_in_nickels (cost_bread cost_cheese given_amount : ℝ) (quarters dimes : ℕ) (nickel_value : ℝ) 
  (h1 : cost_bread = 4.2) (h2 : cost_cheese = 2.05) (h3 : given_amount = 7.0)
  (h4 : quarters = 1) (h5 : dimes = 1) (hnickel_value : nickel_value = 0.05) : 
  ∃ n : ℕ, n = 8 :=
by
  sorry

end change_in_nickels_l400_400752


namespace primes_between_50_and_60_l400_400650

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400650


namespace ellipse_problem_l400_400748

open Real

variables {a b : ℝ}

-- Defining the conditions
def is_ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ (exists (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)
def is_top_vertex (a b : ℝ) (A : ℝ × ℝ) : Prop := A = (0, b)
def is_left_focus (a b : ℝ) (F : ℝ × ℝ) : Prop := F = (-c, 0) ∧ c = sqrt (a^2 - b^2)
def incident_ray (A F B : ℝ × ℝ) : Prop := 
  ∃ (ξ : ℝ), (B = (a, -ξ)) ∧ (A = (0, b)) ∧ (AF : ℝ × ℝ, incident_angle : ℝ), angle_at (AF B) = 45
def is_tangency_circle (A B F : ℝ × ℝ) (tang_line : ℝ → ℝ) : Prop := 
  ∃ (C : ℝ × ℝ) (r : ℝ), C = ((b/2), -(b/2)) ∧ r = (sqrt ((AF C)^2 + (C B)^2) / 2) ∧ 
  tangent (circle_through A B F with_center C and_radius r) tang_line

-- Statement of the problem as a Lean theorem
theorem ellipse_problem 
  (h_ellipse : is_ellipse a b)
  (h_top_vertex : ∃ A, is_top_vertex a b A)
  (h_left_focus : ∃ F, is_left_focus a b F)
  (h_incident_ray : ∃ A F B, incident_ray A F B)
  (h_tangency_circle : ∃ A B F, is_tangency_circle A B F (λ x, 3 * x - y + 3)) :
  (∃ e, e = √2 / 2 ∧ (∃ a b, (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a = √2 ∧ b = 1)) :=
by {
  sorry
}

end ellipse_problem_l400_400748


namespace primes_between_50_and_60_l400_400294

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400294


namespace range_of_n_l400_400199

noncomputable theory
open Real

def f (x : ℝ) : ℝ := exp x + (1/2) * x^2 - x

theorem range_of_n (n : ℝ) (H : ∃ m : ℝ, f m ≤ 2 * n^2 - n) : 
  n ∈ set.Iic (-1 / 2) ∪ set.Ici 1 := by
  sorry

end range_of_n_l400_400199


namespace two_lines_perpendicular_to_same_plane_are_parallel_l400_400880

/- 
Problem: Let a, b be two lines, and α be a plane. Prove that if a ⊥ α and b ⊥ α, then a ∥ b.
-/

variables {Line Plane : Type} 

def is_parallel (l1 l2 : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry
def is_contained_in (l : Line) (p : Plane) : Prop := sorry

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane)
  (ha_perpendicular : is_perpendicular a α)
  (hb_perpendicular : is_perpendicular b α) :
  is_parallel a b :=
by
  sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l400_400880


namespace rect_area_correct_l400_400039

-- Defining the function to calculate the area of a rectangle given the coordinates of its vertices
noncomputable def rect_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℤ) : ℤ :=
  let length := abs (x2 - x1)
  let width := abs (y1 - y3)
  length * width

-- The vertices of the rectangle
def x1 : ℤ := -8
def y1 : ℤ := 1
def x2 : ℤ := 1
def y2 : ℤ := 1
def x3 : ℤ := 1
def y3 : ℤ := -7
def x4 : ℤ := -8
def y4 : ℤ := -7

-- Proving that the area of the rectangle is 72 square units
theorem rect_area_correct : rect_area x1 y1 x2 y2 x3 y3 x4 y4 = 72 := by
  sorry

end rect_area_correct_l400_400039


namespace prime_count_between_50_and_60_l400_400346

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400346


namespace number_of_primes_between_50_and_60_l400_400366

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400366


namespace count_primes_between_50_and_60_l400_400255

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400255


namespace num_primes_50_60_l400_400275

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400275


namespace inequality_solution_l400_400782

theorem inequality_solution (x : ℝ) : 
  (x / (x + 5) ≥ 0) ↔ (x ∈ (Set.Iio (-5)).union (Set.Ici 0)) :=
by
  sorry

end inequality_solution_l400_400782


namespace primes_between_50_and_60_l400_400321

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400321


namespace primes_between_50_and_60_l400_400607

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400607


namespace count_primes_between_50_and_60_l400_400455

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400455


namespace num_primes_50_60_l400_400267

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400267


namespace primes_between_50_and_60_l400_400428

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400428


namespace percentage_of_female_employees_l400_400701

theorem percentage_of_female_employees (E : ℕ) (hE : E = 1400) 
  (pct_computer_literate : ℚ) (hpct : pct_computer_literate = 0.62)
  (female_computer_literate : ℕ) (hfcl : female_computer_literate = 588)
  (pct_male_computer_literate : ℚ) (hmcl : pct_male_computer_literate = 0.5) :
  100 * (840 / 1400) = 60 := 
by
  sorry

end percentage_of_female_employees_l400_400701


namespace cos_alpha_minus_pi_l400_400162

theorem cos_alpha_minus_pi (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 4) : 
  Real.cos (α - Real.pi) = -5 / 8 :=
sorry

end cos_alpha_minus_pi_l400_400162


namespace num_primes_50_60_l400_400282

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400282


namespace equation1_solution_equation2_solution_l400_400783

theorem equation1_solution (x : ℝ) (h : 2 * (x - 1) = 2 - 5 * (x + 2)) : x = -6 / 7 :=
sorry

theorem equation2_solution (x : ℝ) (h : (5 * x + 1) / 2 - (6 * x + 2) / 4 = 1) : x = 1 :=
sorry

end equation1_solution_equation2_solution_l400_400783


namespace warehouse_millet_difference_after_10_nights_l400_400015

theorem warehouse_millet_difference_after_10_nights 
  (x1 x2 : ℝ) 
  (h_initial : x1 = x2 + 16) 
  (x1'_nth_night x2'_nth_night : ℕ → ℝ) 
  (h_nightly_operation : ∀ n, 
    x1'_nth_night (n+1) = x1'_nth_night n + 1/4 * x2'_nth_night n - 1/4 * x1'_nth_night n ∧ 
    x2'_nth_night (n+1) = x2'_nth_night n + 1/4 * x1'_nth_night n - 1/4 * x2'_nth_night n) 
  (h_initial_night: x1'_nth_night 0 = x1) 
  (h_initial_night_2: x2'_nth_night 0 = x2) : 
  x1'_nth_night 10 = x2'_nth_night 10 + 2^(-6) :=
sorry

end warehouse_millet_difference_after_10_nights_l400_400015


namespace num_primes_50_60_l400_400270

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400270


namespace range_of_x_l400_400193

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else if x < 0 then Real.log (-x) / Real.log (1 / 2)
  else 0

theorem range_of_x (x : ℝ) : (f (-x) > f x) ↔ x ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo 0 1 :=
by
  sorry

end range_of_x_l400_400193


namespace prime_count_50_to_60_l400_400509

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400509


namespace simplify_and_evaluate_expr_l400_400775

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l400_400775


namespace integral_equality_l400_400774

noncomputable def f : ℝ → ℝ := sorry 
noncomputable def g (x : ℝ) : ℝ := f x - 2 * real.exp (-x) * ∫ (t : ℝ) in 0..x, real.exp t * f t

theorem integral_equality (hf : continuous f)
  (H : ∫ (x : ℝ) in 0..∞, (f x)^2 < ∞) :
  ∫ (x : ℝ) in 0..∞, (g x)^2 = ∫ (x : ℝ) in 0..∞, (f x)^2 :=
by sorry

end integral_equality_l400_400774


namespace primes_between_50_and_60_l400_400302

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400302


namespace sum_of_roots_of_polynomials_l400_400966

theorem sum_of_roots_of_polynomials :
  let p1 := 2 * x^3 + x^2 - 8 * x + 20,
      p2 := 5 * x^3 - 25 * x^2 + 19,
      S1 := -1 / 2,
      S2 := 5
  in (fun (x : ℂ) => p1 = 0 → p2 = 0 → S1 + S2 = 9 / 2) sorry

end sum_of_roots_of_polynomials_l400_400966


namespace distribute_classes_to_factories_l400_400909

theorem distribute_classes_to_factories :
  ∃ (n : ℕ), n = 240 ∧
  ((∃ (f : ℕ → ℕ), (f 0 + f 1 + f 2 + f 3 = 5 ∧ ∀ i, f i ≥ 1)) →
  n) :=
sorry

end distribute_classes_to_factories_l400_400909


namespace count_primes_50_60_l400_400475

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400475


namespace count_primes_50_60_l400_400470

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400470


namespace distance_between_intersections_l400_400813

-- Definitions according to conditions
def line (x : ℝ) : ℝ := -2
def parabola (x : ℝ) : ℝ := x^2 - 5 * x + 3

-- Intersection points
def intersection_points : set ℝ := {x : ℝ | line x = parabola x}

-- The Lean statement for the proof problem
theorem distance_between_intersections : 
  ∃ (x1 x2 : ℝ), x1 ∈ intersection_points ∧ x2 ∈ intersection_points ∧ (x1 ≠ x2) ∧ ((| x1 - x2 | = sqrt 5)) :=
by
  sorry

end distance_between_intersections_l400_400813


namespace primes_between_50_and_60_l400_400393

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400393


namespace mode_median_ages_l400_400796

theorem mode_median_ages (h₁ : 2 * 19 + 5 * 20 + 2 * 21 + 2 * 22 + 1 * 23 = 12) :
  let ages := list.replicate 2 19 ++ list.replicate 5 20 ++ list.replicate 2 21 ++
              list.replicate 2 22 ++ list.replicate 1 23 in
  list.mode ages = some 20 ∧ list.median ages = some 20 :=
by
  let ages := list.replicate 2 19 ++ list.replicate 5 20 ++ list.replicate 2 21 ++
              list.replicate 2 22 ++ list.replicate 1 23
  have h_total : ages.length = 12 := by sorry
  have h_median : list.median ages = some 20 := by sorry
  have h_mode : list.mode ages = some 20 := by sorry
  exact ⟨h_mode, h_median⟩

end mode_median_ages_l400_400796


namespace cyclic_quadrilateral_l400_400873

variables {A B C H D E F S T U : Type}
-- Assuming some necessary definitions and axioms for the geometric constructs
-- like orthocenter, circumcircle, parallel lines, and reflections
axiom orthocenter (A B C H : Type) : Prop
axiom circumcircle (A B C D E F : Type) : Prop
axiom parallel (AD BE CF : Type) : Prop
axiom reflection (D E F S T U BC CA AB : Type) : Prop
axiom cyclic (P Q R S : Type) : Prop

-- Statement of the theorem
theorem cyclic_quadrilateral (A B C H D E F S T U : Type)
  (orth : orthocenter A B C H)
  (circ : circumcircle A B C D E F)
  (parallel : parallel AD BE CF)
  (reflect : reflection D E F S T U BC CA AB) :
  cyclic S T U H :=
begin
  sorry
end

end cyclic_quadrilateral_l400_400873


namespace sum_of_arithmetic_series_l400_400191

theorem sum_of_arithmetic_series (a1 an : ℕ) (d n : ℕ) (s : ℕ) :
  a1 = 2 ∧ an = 100 ∧ d = 2 ∧ n = (an - a1) / d + 1 ∧ s = n * (a1 + an) / 2 → s = 2550 :=
by
  sorry

end sum_of_arithmetic_series_l400_400191


namespace primes_between_50_and_60_l400_400529

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400529


namespace count_primes_50_60_l400_400485

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400485


namespace primes_between_50_and_60_l400_400430

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400430


namespace max_value_of_f_l400_400746

-- Define the function f
noncomputable def f (p x : ℝ) : ℝ := log 2 ((x + 1) / (x - 1)) + log 2 (x - 1) + log 2 (p - x)

-- Prove that the domain of f(x) with given conditions is (1, p) when p > 1
lemma domain_of_f (p : ℝ) (h1 : p > 1) : set.Ioc 1 p = { x : ℝ | 1 < x ∧ x < p } :=
by {
  simp [set.Ioc],
}

-- Prove that when p > 3, the maximum value of f(x) is 2 log₂(p+1) - 2 and there is no minimum value.
theorem max_value_of_f (p : ℝ) (h2 : p > 3) :
  ∃ x ∈ (set.Ioc 1 p), f p x = 2 * (log 2 (p + 1)) - 2 ∧
  ∀ y ∈ (set.Ioc 1 p), f p y ≤ 2 * (log 2 (p + 1)) - 2 ∧
  (∀ z, z ∈ (1, p) -> (f p z)  ≠ (2 * (log 2 (p + 1)) - 2 -> false)) :=
by {
  sorry -- Pending proof details
}

end max_value_of_f_l400_400746


namespace green_valley_absent_percentage_l400_400088

theorem green_valley_absent_percentage :
  let total_students := 130
  let boys := 75
  let girls := 55
  let absent_boys := boys / 5
  let absent_girls := round (girls / 4)
  let total_absent := absent_boys + absent_girls
  let absent_percentage := (total_absent / total_students) * 100
  in absent_percentage = 22 :=
by
  sorry

end green_valley_absent_percentage_l400_400088


namespace third_beats_fifth_l400_400695

-- Definitions of the conditions
-- 1. Eight players participated in the chess tournament, and each player played every other once.
def num_players := 8
def total_games := (num_players * (num_players - 1)) / 2    -- Total number of games played
def total_points := 28  -- Sum of points from 28 games

-- 2. Each player has a different number of points (distinct elements in a set of scores)
variables (score : Fin num_players → ℕ)
def distinct_points := Function.Injective score

-- 3. The second-place player's points equal the total points of the bottom four players combined.
def second_place := score ⟨1, by norm_num⟩  -- The 2nd place player's score
def bottom_four_points := (score ⟨4, by norm_num⟩) + (score ⟨5, by norm_num⟩) + (score ⟨6, by norm_num⟩) + (score ⟨7, by norm_num⟩)

axiom second_place_condition : second_place = bottom_four_points

-- The theorem we want to prove
theorem third_beats_fifth (second_place_condition : second_place = bottom_four_points) :
  score ⟨2, by norm_num⟩ > score ⟨4, by norm_num⟩ :=
  sorry

end third_beats_fifth_l400_400695


namespace find_omega_and_range_l400_400203

noncomputable def f (ω : ℝ) (x : ℝ) := (Real.sin (ω * x))^2 + (Real.sqrt 3) * (Real.sin (ω * x)) * (Real.sin (ω * x + Real.pi / 2))

theorem find_omega_and_range :
  ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x = (Real.sin (2 * ω * x - Real.pi / 6) + 1/2)) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
      f 1 x ∈ Set.Icc ((1 - Real.sqrt 3) / 2) (3 / 2)) :=
by
  sorry

end find_omega_and_range_l400_400203


namespace factorization_of_expression_l400_400120

theorem factorization_of_expression (x y : ℝ) : x^2 - x * y = x * (x - y) := 
by
  sorry

end factorization_of_expression_l400_400120


namespace angle_DFE_75_l400_400052

-- Definitions and conditions
variables (O D E F : Type)
variables [is_circumscribed_around △ DEF O]
variables (angle_DOE : angle D O E = 120)
variables (angle_EOF : angle E O F = 90)

-- Theorem statement: prove that ∠DFE = 75°
theorem angle_DFE_75 : angle D F E = 75 :=
by
  sorry

end angle_DFE_75_l400_400052


namespace count_primes_between_50_and_60_l400_400447

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400447


namespace count_primes_50_60_l400_400479

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400479


namespace integral_correct_l400_400925

noncomputable def integral_problem : ℝ :=
  ∫ (x: ℝ) in 0..2, (Real.sqrt (4 - x^2) - 2 * x)

theorem integral_correct : integral_problem = Real.pi - 4 := 
by
  sorry

end integral_correct_l400_400925


namespace miles_from_second_friend_to_work_l400_400939
variable (distance_to_first_friend := 8)
variable (distance_to_second_friend := distance_to_first_friend / 2)
variable (total_distance_to_second_friend := distance_to_first_friend + distance_to_second_friend)
variable (distance_to_work := 3 * total_distance_to_second_friend)

theorem miles_from_second_friend_to_work :
  distance_to_work = 36 := 
by
  sorry

end miles_from_second_friend_to_work_l400_400939


namespace trail_mix_total_weight_l400_400922

def peanuts : ℝ := 0.16666666666666666
def chocolate_chips : ℝ := 0.16666666666666666
def raisins : ℝ := 0.08333333333333333
def trail_mix_weight : ℝ := 0.41666666666666663

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = trail_mix_weight :=
sorry

end trail_mix_total_weight_l400_400922


namespace num_primes_between_50_and_60_l400_400613

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400613


namespace primes_between_50_and_60_l400_400599

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400599


namespace count_primes_between_50_and_60_l400_400245

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400245


namespace min_value_y_l400_400854

theorem min_value_y (x : ℝ) : ∃ (x : ℝ), y = x^2 + 16 * x + 20 → y ≥ -44 :=
begin
  sorry
end

end min_value_y_l400_400854


namespace n_times_s_eq_neg_two_l400_400741

-- Define existence of function g
variable (g : ℝ → ℝ)

-- The given condition for the function g: ℝ -> ℝ
axiom g_cond : ∀ x y : ℝ, g (g x - y) = 2 * g x + g (g y - g (-x)) + y

-- Define n and s as per the conditions mentioned in the problem
def n : ℕ := 1 -- Based on the solution, there's only one possible value
def s : ℝ := -2 -- Sum of all possible values

-- The main statement to prove
theorem n_times_s_eq_neg_two : (n * s) = -2 := by
  sorry

end n_times_s_eq_neg_two_l400_400741


namespace pumpkin_patch_pie_filling_l400_400070

def pumpkin_cans (small_pumpkins : ℕ) (large_pumpkins : ℕ) (sales : ℕ) (small_price : ℕ) (large_price : ℕ) : ℕ :=
  let remaining_small_pumpkins := small_pumpkins
  let remaining_large_pumpkins := large_pumpkins
  let small_cans := remaining_small_pumpkins / 2
  let large_cans := remaining_large_pumpkins
  small_cans + large_cans

#eval pumpkin_cans 50 33 120 3 5 -- This evaluates the function with the given data to ensure the logic matches the question

theorem pumpkin_patch_pie_filling : pumpkin_cans 50 33 120 3 5 = 58 := by sorry

end pumpkin_patch_pie_filling_l400_400070


namespace double_series_evaluation_l400_400946

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, if h : n ≥ m then 1 / (m * n * (m + n + 2)) else 0) = (Real.pi ^ 2) / 6 := sorry

end double_series_evaluation_l400_400946


namespace primes_count_between_50_and_60_l400_400561

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400561


namespace primes_between_50_and_60_l400_400586

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400586


namespace prime_count_between_50_and_60_l400_400341

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400341


namespace num_primes_between_50_and_60_l400_400621

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400621


namespace sum_xy_sum_inv_squared_geq_nine_four_l400_400982

variable {x y z : ℝ}

theorem sum_xy_sum_inv_squared_geq_nine_four (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z + z * x) * (1 / (x + y)^2 + 1 / (y + z)^2 + 1 / (z + x)^2) ≥ 9 / 4 :=
by sorry

end sum_xy_sum_inv_squared_geq_nine_four_l400_400982


namespace natural_numbers_division_l400_400063

theorem natural_numbers_division (n : ℕ) (q r : ℕ) (h1 : n = 8 * q + r)
  (h2 : q + r = 13) (h3 : 0 ≤ r ∧ r < 8) :
  n ∈ {108, 100, 92, 84, 76, 68, 60, 52, 44} :=
sorry

end natural_numbers_division_l400_400063


namespace cost_price_of_watch_l400_400079

theorem cost_price_of_watch (C : ℝ) (h1 : ∃ C, 0.91 * C + 220 = 1.04 * C) : C = 1692.31 :=
sorry  -- proof to be provided

end cost_price_of_watch_l400_400079


namespace count_4_digit_integers_with_conditions_l400_400211

theorem count_4_digit_integers_with_conditions :
  ∃ (n : ℕ), n = 60 ∧
    (∀ d1 d2 d3 d4 : ℕ, 
      n = if (d1 ≠ 0 ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
              d2 ≠ d3 ∧ d2 ≠ d4 ∧ 
              d3 ≠ d4 ∧ 
              d4 = 7 ∧ 
              (d1 * 1000 + d2 * 100 + d3 * 10 + d4).is_multiple_of 3) then 1 else 0) :=
sorry

end count_4_digit_integers_with_conditions_l400_400211


namespace find_pairs_in_range_l400_400121

-- Define the natural number range and the equality condition
def nat_in_range (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)

-- Main theorem to state the problem
theorem find_pairs_in_range : {x : ℕ // nat_in_range x} =
                              {y : ℕ // nat_in_range y} ^ 2 :=
by
  sorry

end find_pairs_in_range_l400_400121


namespace total_path_length_l400_400085

-- Definitions related to the problem
def equilateral_triangle (side_length : ℝ) := ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = side_length

def square (side_length : ℝ) := ∀ (w x y z : ℝ), w = x ∧ x = y ∧ y = z ∧ z = side_length

-- Conditions for the problem
def conditions (E_on_WZ : Prop) (rotation_angle : ℝ) := 
  equilateral_triangle 3 ∧ 
  square 6 ∧ 
  E_on_WZ ∧ 
  rotation_angle = (90 : ℝ) * (π / 180)

-- Theorem statement
theorem total_path_length (E_on_WZ : Prop) (rotation_angle : ℝ) (path_length : ℝ) :
  conditions E_on_WZ rotation_angle → path_length = 6 * π :=
by
  intros,
  sorry

end total_path_length_l400_400085


namespace prime_count_between_50_and_60_l400_400574

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400574


namespace Tye_bills_received_l400_400843

theorem Tye_bills_received :
  ∀ (withdrawal_per_bank : ℕ) (number_of_banks : ℕ) (denomination : ℕ),
    withdrawal_per_bank = 300 →
    number_of_banks = 2 →
    denomination = 20 →
    (withdrawal_per_bank * number_of_banks) / denomination = 30 :=
by
  intros withdrawal_per_bank number_of_banks denomination hw hb hd
  rw [hw, hb, hd]
  simp
  apply nat.div_eq_of_lt (show 600 < nat.succ 599, from dec_trivial)
  have h : 600 = 20 * 30 := by norm_num
  rw h
  exact nat.mul_div_cancel_left 30 (nat.succ_pos 19)

end Tye_bills_received_l400_400843


namespace primes_between_50_and_60_l400_400534

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400534


namespace primes_between_50_and_60_l400_400652

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400652


namespace prime_count_between_50_and_60_l400_400333

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400333


namespace num_primes_between_50_and_60_l400_400611

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400611


namespace primes_between_50_and_60_l400_400412

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400412


namespace primes_count_between_50_and_60_l400_400556

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400556


namespace log_expression_l400_400999

theorem log_expression (x : ℝ) (h1 : x > 1) (h2 : (Real.log10 x) ^ 2 - Real.log10 (x ^ 2) = 18) :
  (Real.log10 x) ^ 3 - Real.log10 (x ^ 3) = 198 :=
by
  -- Proof goes here
  sorry

end log_expression_l400_400999


namespace arithmetic_series_sum_is_1620_l400_400096

open ArithmeticSeries

/-- Conditions for the arithmetic series -/
def a1 : ℚ := 10
def an : ℚ := 30
def d : ℚ := 1 / 4

/--Calculating the number of terms in the series -/
def n : ℕ := ((an - a1) / d).to_nat + 1

/--Proving the sum of the arithmetic series is 1620 -/
theorem arithmetic_series_sum_is_1620 : 
  (arithmetic_series_sum a1 an d n) = 1620 :=
by
  sorry

end arithmetic_series_sum_is_1620_l400_400096


namespace prime_count_between_50_and_60_l400_400338

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400338


namespace primes_between_50_and_60_l400_400522

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400522


namespace bmw_cars_sold_l400_400888

theorem bmw_cars_sold (total_cars : ℕ) (mercedes_percent nissan_percent ford_percent chevrolet_percent : ℕ) :
  total_cars = 300 →
  mercedes_percent = 20 →
  nissan_percent = 25 →
  ford_percent = 10 →
  chevrolet_percent = 18 →
  let bmw_percent := 100 - (mercedes_percent + nissan_percent + ford_percent + chevrolet_percent) in
  let bmw_cars := total_cars * bmw_percent / 100 in
  bmw_cars = 81 :=
by
  intros 
  simp
  sorry

end bmw_cars_sold_l400_400888


namespace primes_between_50_and_60_l400_400526

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400526


namespace primes_between_50_and_60_l400_400329

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400329


namespace primes_between_50_and_60_l400_400231

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400231


namespace prime_count_between_50_and_60_l400_400571

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400571


namespace measure_of_angle_BAC_l400_400016

-- Definition of the problem in Lean 4
theorem measure_of_angle_BAC
  (circle : Type)
  (center : circle → Prop)
  (tangent_from : circle → circle → Prop)
  (point_A B C : circle)
  (arc_ratio_3_to_5 : (ratio_of_arcs BC CB' = 3 / 5)) :
  measure_of_angle BAC = 67.5 :=
by
  -- Formal proof would go here
  sorry

end measure_of_angle_BAC_l400_400016


namespace oblique_lines_projection_angle_l400_400761

noncomputable def congruent_oblique_lines_angle_proof : Prop :=
∀ (P A B P' : Type)
(PA PB : P → A → B → ℝ)
(P'_proj : P → P' → A → B → ℝ),
  PA = PB →
  P'_proj P = P' →
  P'_proj PA = P'A →
  P'_proj PB = P'B →
  PA = PB →
  angle APB < angle AP'B

theorem oblique_lines_projection_angle : congruent_oblique_lines_angle_proof :=
sorry

end oblique_lines_projection_angle_l400_400761


namespace primes_count_between_50_and_60_l400_400558

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400558


namespace count_primes_between_50_and_60_l400_400246

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400246


namespace primes_between_50_and_60_l400_400313

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400313


namespace calculation_correct_solve_system_of_inequalities_correct_l400_400042

theorem calculation_correct :
  -1^2 + real.cbrt (-8) - abs (1 - real.sqrt 2) + real.sqrt ((-2)^2) = -real.sqrt 2 :=
sorry

theorem solve_system_of_inequalities_correct :
  {x : ℤ // 0 ≤ x} → (x - 3 * (x - 2) ≥ 4) ∧ ((2 * x - 1) / 5 < (x + 1) / 2) → (x = 0 ∨ x = 1) :=
sorry

end calculation_correct_solve_system_of_inequalities_correct_l400_400042


namespace primes_between_50_and_60_l400_400383

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400383


namespace construct_circle_center_impossible_l400_400758

-- Definitions for the geometric context
variable (O : Type) [plane_geometry O]

/-- Statement of the theorem -/
theorem construct_circle_center_impossible (C : circle O) : 
  ¬ (exists (f : construction_technique), uses_only_straightedge f ∧ f.constructs_center C) := 
sorry

end construct_circle_center_impossible_l400_400758


namespace multiply_same_exponents_l400_400094

theorem multiply_same_exponents (x : ℝ) : (x^3) * (x^3) = x^6 :=
by sorry

end multiply_same_exponents_l400_400094


namespace graph_transformation_l400_400003

theorem graph_transformation : 
  ∀ (x : ℝ), 
  (sin (2 * (x + π / 4)) - 1 = sin (2 * x + π / 2) - 1) :=
by
  intros x
  have h1 : 2 * (x + π / 4) = 2 * x + π / 2, by ring
  rw [h1]
  sorry

end graph_transformation_l400_400003


namespace count_primes_between_50_and_60_l400_400261

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400261


namespace num_primes_between_50_and_60_l400_400620

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400620


namespace prime_count_between_50_and_60_l400_400337

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400337


namespace amount_spent_on_belt_correct_l400_400082

variable (budget shirt pants coat socks shoes remaining : ℕ)

-- Given conditions
def initial_budget : ℕ := 200
def spent_shirt : ℕ := 30
def spent_pants : ℕ := 46
def spent_coat : ℕ := 38
def spent_socks : ℕ := 11
def spent_shoes : ℕ := 41
def remaining_amount : ℕ := 16

-- The amount spent on the belt
def amount_spent_on_belt : ℕ :=
  budget - remaining - (shirt + pants + coat + socks + shoes)

-- The theorem statement we need to prove
theorem amount_spent_on_belt_correct :
  initial_budget = budget →
  spent_shirt = shirt →
  spent_pants = pants →
  spent_coat = coat →
  spent_socks = socks →
  spent_shoes = shoes →
  remaining_amount = remaining →
  amount_spent_on_belt budget shirt pants coat socks shoes remaining = 18 := by
    simp [initial_budget, spent_shirt, spent_pants, spent_coat, spent_socks, spent_shoes, remaining_amount, amount_spent_on_belt]
    sorry

end amount_spent_on_belt_correct_l400_400082


namespace count_primes_between_50_and_60_l400_400673

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400673


namespace expectation_decreases_l400_400862

variables (ξ : Type) [MeasurableSpace ξ] [ProbabilitySpace ξ]
noncomputable def P_zero (x : ℝ) : prob_measure ξ := sorry
noncomputable def P_one (x : ℝ) : prob_measure ξ := sorry

axiom P_sum (x : ℝ) (hx : 0 < x ∧ x < 1/2) : p P_zero x + p P_one x = 1

theorem expectation_decreases (x y : ℝ) (hx : 0 < x ∧ x < 1/2) (hy : 0 < y ∧ y < 1/2) (hxy : x < y) : 
  E (λ ξ, if ξ = 0 then P_zero x else P_one x) > E (λ ξ, if ξ = 0 then P_zero y else P_one y) := 
sorry

end expectation_decreases_l400_400862


namespace count_primes_between_50_and_60_l400_400456

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400456


namespace gcd_80_180_450_l400_400020

theorem gcd_80_180_450 : Int.gcd (Int.gcd 80 180) 450 = 10 := by
  sorry

end gcd_80_180_450_l400_400020


namespace max_length_of_tape_l400_400702

noncomputable def tape_thickness : ℝ := 0.0075
noncomputable def spool_diameter : ℝ := 11
noncomputable def center_distance : ℝ := 42

def radius (diameter : ℝ) := diameter / 2

def total_area_of_tape (R r : ℝ) := 2 * Real.pi * (R^2 - r^2)

def length_of_tape (area thickness : ℝ) := area / thickness

theorem max_length_of_tape :
  let r := radius spool_diameter
  let R := center_distance / 2
  let area_of_tape := total_area_of_tape R r
  let length_of_tape := length_of_tape area_of_tape tape_thickness
  length_of_tape ≈ 344 := 
by
  sorry

end max_length_of_tape_l400_400702


namespace find_constant_c_l400_400685

theorem find_constant_c (c : ℝ) (h1 : ∀ x : ℝ, f x = x * (x - c)^2) (h2 : has_local_min f 2) : c = 2 := 
begin
  sorry
end

end find_constant_c_l400_400685


namespace sum_distances_on_circumcircle_l400_400069

theorem sum_distances_on_circumcircle (A B C M : Point) (h_eq_triangle : IsEquilateralTriangle A B C)
  (h_M_on_circumcircle : OnCircumcircle M A B C) :
  let d1 := min (dist M A) (min (dist M B) (dist M C)),
      d2 := (if dist M A = d1 then min (dist M B) (dist M C) else if dist M B = d1 then min (dist M A) (dist M C) else min (dist M A) (dist M B)),
      d3 := (if d1 = dist M A ∨ d2 = dist M A then max (dist M B) (dist M C) else if d1 = dist M B ∨ d2 = dist M B then max (dist M A) (dist M C) else max (dist M A) (dist M B)) in
  d1 + d2 = d3 :=
sorry

end sum_distances_on_circumcircle_l400_400069


namespace budget_utilities_percentage_l400_400053

theorem budget_utilities_percentage :
  let transportation_percent := 15
      research_and_development_percent := 9
      equipment_percent := 4
      supplies_percent := 2
      salaries_degrees := 234
      full_circle_degrees := 360
      full_budget_percent := 100
      salaries_percent := (salaries_degrees * 100) / full_circle_degrees
  in full_budget_percent - (transportation_percent + research_and_development_percent + equipment_percent + supplies_percent + salaries_percent) = 5 :=
by
  let transportation_percent := 15
  let research_and_development_percent := 9
  let equipment_percent := 4
  let supplies_percent := 2
  let salaries_degrees := 234
  let full_circle_degrees := 360
  let full_budget_percent := 100
  let salaries_percent := (salaries_degrees * 100) / full_circle_degrees
  sorry

end budget_utilities_percentage_l400_400053


namespace primes_between_50_and_60_l400_400296

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400296


namespace primes_between_50_and_60_l400_400317

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400317


namespace monomials_like_terms_l400_400187

theorem monomials_like_terms (a b : ℝ) (m n : ℤ) 
  (h1 : 2 * (a^4) * (b^(-2 * m + 7)) = 3 * (a^(2 * m)) * (b^(n + 2))) :
  m + n = 3 := 
by {
  -- Our proof will be placed here
  sorry
}

end monomials_like_terms_l400_400187


namespace count_primes_between_50_and_60_l400_400659

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400659


namespace primes_count_between_50_and_60_l400_400542

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400542


namespace primes_between_50_and_60_l400_400632

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400632


namespace solve_system_l400_400789

namespace SolutionProof

-- Conditions
def equation1 (x y : ℝ) : Prop := 3^y * 81 = 9^(x^2)
def equation2 (x y : ℝ) : Prop := log 10 y = log 10 x - log 10 0.5

-- Proof problem statement
theorem solve_system : ∃ x y : ℝ, equation1 x y ∧ equation2 x y ∧ x = 2 ∧ y = 4 :=
by {
  sorry 
}

end SolutionProof

end solve_system_l400_400789


namespace count_primes_between_50_and_60_l400_400242

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400242


namespace pyramid_volume_and_base_area_l400_400699

noncomputable def pyramidEdges : ℝ := sqrt 70
def pyramidEdges2 : ℝ := sqrt 99
def pyramidEdges3 : ℝ := sqrt 126

theorem pyramid_volume_and_base_area
  (PA PB PC : ℝ)
  (h₁ : PA = sqrt 70)
  (h₂ : PB = sqrt 99)
  (h₃ : PC = sqrt 126) :
  (1 / 6) * (real.sqrt (PA * PB * PC * (PA + PB + PC))) = 21 * real.sqrt 55 ∧
  (real.sqrt (PA * PB * (PA + PB))) = 84 := 
sorry

end pyramid_volume_and_base_area_l400_400699


namespace combinatorial_identity_l400_400973

open Nat

theorem combinatorial_identity
  (n m k : ℕ)
  (h1 : 1 ≤ k)
  (h2 : k < m)
  (h3 : m ≤ n)
  (h4 : 0 < m) :
  (nat.choose n m + ∑ i in range k, nat.choose k i * nat.choose n (m-i)) = nat.choose (n+k) m := 
sorry

end combinatorial_identity_l400_400973


namespace count_primes_between_50_and_60_l400_400658

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400658


namespace min_y_in_quadratic_l400_400853

theorem min_y_in_quadratic (x : ℝ) : ∃ y : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ y', (y' = x^2 + 16 * x + 20) → y ≤ y' := 
sorry

end min_y_in_quadratic_l400_400853


namespace primes_between_50_and_60_l400_400326

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400326


namespace good_numbers_l400_400899

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_odd_prime (n : ℕ) : Prop :=
  Prime n ∧ n % 2 = 1

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_divisor (d + 1) (n + 1)

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ n = 1 ∨ is_odd_prime n :=
sorry

end good_numbers_l400_400899


namespace sum_first_n_terms_bn_eq_contraction_sequence_bn_find_sequences_l400_400146

def contraction_sequence (a : ℕ → ℤ) (b : ℕ → ℤ) :=
  ∀ i, b i = (finset.range (i + 1)).max (λ j, a j) - (finset.range (i + 1)).min (λ j, a j)

def sum_first_n_terms (b : ℕ → ℤ) (n : ℕ) :=
  finset.sum (finset.range n) b

theorem sum_first_n_terms_bn_eq (a : ℕ → ℤ) (b : ℕ → ℤ) (h : contraction_sequence a b) :
  (∀ n, a n = 2 * n + 1) →
  ∀ n, sum_first_n_terms b n = n * (n - 1) :=
sorry

theorem contraction_sequence_bn (a : ℕ → ℤ) (b : ℕ → ℤ) :
  contraction_sequence a b →
  contraction_sequence b b :=
sorry

theorem find_sequences (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (h1 : ∀ (n : ℕ), S n = finset.sum (finset.range (n + 1)) a) 
  (h2 : ∀ (n : ℕ), S_1 + S_2 + ⋯ + S_n = n*(n+1) / 2 * a 1 + (n*(n-1) / 2 * b n))
  (b : ℕ → ℤ) :
  contraction_sequence a b →
  ∀ n, a n = if n = 1 then a 1 else a 2 ∧ a 2 ≥ a 1 :=
sorry

end sum_first_n_terms_bn_eq_contraction_sequence_bn_find_sequences_l400_400146


namespace primes_between_50_and_60_l400_400381

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400381


namespace num_primes_50_60_l400_400263

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400263


namespace primes_between_50_and_60_l400_400219

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400219


namespace largest_possible_M_l400_400959

theorem largest_possible_M (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_cond : x * y + y * z + z * x = 1) :
    ∃ M, ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + y * z + z * x = 1 → 
    (x / (1 + yz/x) + y / (1 + zx/y) + z / (1 + xy/z) ≥ M) → 
        M = 3 / (Real.sqrt 3 + 1) :=
by
  sorry        

end largest_possible_M_l400_400959


namespace primes_between_50_and_60_l400_400530

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400530


namespace number_of_primes_between_50_and_60_l400_400356

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400356


namespace find_a1_l400_400794

def sequence (a : ℕ → ℝ) : Prop :=
a 0 = 1 ∧ ∀ n : ℕ, a n = a (n + 1) + a (n + 2)

theorem find_a1 (a : ℕ → ℝ) (h : sequence a) : a 1 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end find_a1_l400_400794


namespace affine_transformation_iff_equalities_l400_400989

variables {n : ℕ} (A : fin n → ℝ × ℝ) (O : ℝ × ℝ)

def is_regular_polygon (A : fin n → ℝ × ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : fin n, (A i).fst = r * cos (2 * π * i / n) ∧ (A i).snd = r * sin (2 * π * i / n)

def affine_transformation_exists (A : fin n → ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  ∃ B : fin n → ℝ × ℝ, is_regular_polygon B ∧ 
  ∃ L : (ℝ × ℝ → ℝ × ℝ), ∀ i, L (B i) = A i ∧ L O = O

theorem affine_transformation_iff_equalities 
  (A : fin n → ℝ × ℝ) (O : ℝ × ℝ) :
  affine_transformation_exists A O ↔ 
  (∀ i : fin n, let i1 := (i + 1) % n, i2 := (i + 2) % n in
  (A i, O) + (A i1, O) = (2 * cos (2 * π / n)) • (A i2, O)) :=
sorry

end affine_transformation_iff_equalities_l400_400989


namespace prime_count_50_to_60_l400_400497

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400497


namespace primes_between_50_and_60_l400_400291

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400291


namespace prime_count_between_50_and_60_l400_400342

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400342


namespace smallest_integer_10010_l400_400140

-- Definitions and conditions
def has_same_digit_sum (s : ℕ) (l : List ℕ) : Prop :=
  (∀ x ∈ l, x.digitSum = s)

-- The main theorem
theorem smallest_integer_10010 :
  ∃ (n : ℕ), 
    (∃ l₁ : List ℕ, l₁.length = 2002 ∧ has_same_digit_sum l₁.head l₁ ∧ n = l₁.sum) ∧
    (∃ l₂ : List ℕ, l₂.length = 2003 ∧ has_same_digit_sum l₂.head l₂ ∧ n = l₂.sum) ∧
    n = 10010 :=
sorry

end smallest_integer_10010_l400_400140


namespace primes_between_50_and_60_l400_400585

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400585


namespace primes_between_50_and_60_l400_400396

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400396


namespace primes_between_50_and_60_l400_400642

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400642


namespace count_primes_50_60_l400_400490

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400490


namespace quadratic_root_l400_400682

theorem quadratic_root (a b c : ℝ) (h : 9 * a - 3 * b + c = 0) : 
  a * (-3)^2 + b * (-3) + c = 0 :=
by
  sorry

end quadratic_root_l400_400682


namespace primes_count_between_50_and_60_l400_400560

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400560


namespace num_primes_between_50_and_60_l400_400630

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400630


namespace find_φ_l400_400983

-- Definitions 
def ω (ω_val : ℝ) := ω_val > 0
def φ_range (φ_val : ℝ) := 0 < φ_val ∧ φ_val < π
def symmetry_lines (f : ℝ → ℝ) :=
  (∀ x, f (x) = sin (ω x + φ) → f (π/4) = f (5π/4))

-- Main theorem
theorem find_φ (ω : ℝ) (φ : ℝ) :
  (ω > 0) →
  (0 < φ ∧ φ < π) →
  (∀ x, sin (ω x + φ) = sin (ω (x + π))) →
  φ = π/4 :=
begin
  intros hω hφ hsym,
  -- Proof steps
  sorry
end

end find_φ_l400_400983


namespace quadratic_positive_intervals_l400_400689

-- Problem setup
def quadratic (x : ℝ) : ℝ := x^2 - x - 6

-- Define the roots of the quadratic function
def is_root (a b : ℝ) (f : ℝ → ℝ) := f a = 0 ∧ f b = 0

-- Proving the intervals where the quadratic function is greater than 0
theorem quadratic_positive_intervals :
  is_root (-2) 3 quadratic →
  { x : ℝ | quadratic x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 3 } :=
by
  sorry

end quadratic_positive_intervals_l400_400689


namespace tangent_line_through_P_is_correct_l400_400075

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the tangent line equation to prove
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 10 = 0

-- Problem statement in Lean 4
theorem tangent_line_through_P_is_correct :
  C P.1 P.2 → tangent_line P.1 P.2 :=
by
  intros hC
  sorry

end tangent_line_through_P_is_correct_l400_400075


namespace count_primes_between_50_and_60_l400_400660

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400660


namespace primes_between_50_and_60_l400_400424

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400424


namespace num_primes_50_60_l400_400284

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400284


namespace f_640_minus_f_320_l400_400141

def sigma (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, n % d = 0).sum

def f (n : ℕ) : ℚ := (sigma n : ℚ) / n

theorem f_640_minus_f_320 : f 640 - f 320 = 3 / 320 := by
  sorry

end f_640_minus_f_320_l400_400141


namespace fraction_difference_l400_400924

theorem fraction_difference : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := 
  sorry

end fraction_difference_l400_400924


namespace ellipse_standard_equation_min_lambda_inequality_l400_400176

noncomputable def ellipse_equation :=
  ∃ a b : ℝ, a > b > 0 ∧ 2 * b * a = b^2 + 1 ∧ (λ (x y: ℝ), x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a = sqrt 2 ∧ b = 1 ∧ (λ (x y: ℝ), x^2 / 2 + y^2 = 1) :=
sorry

def PA_dot_PB (A B : ℝ × ℝ) : ℝ := (A.1 - 2, A.2) • (B.1 - 2, B.2)

theorem min_lambda_inequality :
  ∃ λ : ℝ, (∀ A B : ℝ × ℝ, PA_dot_PB A B ≤ λ) ∧ λ = 17 / 2 :=
sorry

end ellipse_standard_equation_min_lambda_inequality_l400_400176


namespace percent_c_of_b_l400_400681

variable (a b c : ℝ)

theorem percent_c_of_b (h1 : c = 0.20 * a) (h2 : b = 2 * a) : 
  ∃ x : ℝ, c = (x / 100) * b ∧ x = 10 :=
by
  sorry

end percent_c_of_b_l400_400681


namespace primes_between_50_and_60_l400_400304

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400304


namespace minimize_expression_l400_400961

-- We define the conditions.
def problem (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x * y = 4) : Prop :=
  ∀ z, z = (x^3 / (y - 1) + y^3 / (x - 1)) → z ≥ 16

-- We assert that the minimum value of the given expression under the specified conditions is 16.
theorem minimize_expression : problem :=
begin
  sorry
end

end minimize_expression_l400_400961


namespace sin_double_angle_l400_400975

theorem sin_double_angle 
  (A : ℝ) 
  (h1 : π / 2 < A) 
  (h2 : A < π) 
  (h3 : real.sin A = 4 / 5) : 
  real.sin (2 * A) = -24 / 25 := 
by 
sorry

end sin_double_angle_l400_400975


namespace number_of_proper_subsets_of_set_l400_400045

theorem number_of_proper_subsets_of_set :
  (finset.powerset (finset.from_list [1, 2, 3])).card - 1 = 7 :=
by
  sorry

end number_of_proper_subsets_of_set_l400_400045


namespace equation_transformation_l400_400837

theorem equation_transformation (x y: ℝ) (h : 2 * x - 3 * y = 6) : 
  y = (2 * x - 6) / 3 := 
by
  sorry

end equation_transformation_l400_400837


namespace primes_between_50_and_60_l400_400648

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400648


namespace count_primes_between_50_and_60_l400_400675

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400675


namespace simplify_and_evaluate_l400_400777

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3)) - (x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  sorry

end simplify_and_evaluate_l400_400777


namespace sqrt_three_irrational_l400_400860

theorem sqrt_three_irrational : irrational (Real.sqrt 3) := by
  sorry

end sqrt_three_irrational_l400_400860


namespace prime_count_50_to_60_l400_400496

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400496


namespace primes_between_50_and_60_l400_400414

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400414


namespace car_owners_without_motorcycles_l400_400698

/-- Number of car owners who do not own a motorcycle given the conditions:

1. Total number of adults = 351
2. Number of car owners = 331
3. Number of motorcycle owners = 45
-/
theorem car_owners_without_motorcycles (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) 
  (h1 : total_adults = 351) (h2 : car_owners = 331) (h3 : motorcycle_owners = 45) : 
  car_owners - (car_owners + motorcycle_owners - total_adults) = 306 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end car_owners_without_motorcycles_l400_400698


namespace primes_between_50_and_60_l400_400442

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400442


namespace num_primes_50_60_l400_400277

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400277


namespace primes_between_50_and_60_l400_400600

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400600


namespace combined_prob_orange_yellow_l400_400894

variable (P_red : ℝ) (P_green : ℝ) (P_orange_yellow : ℝ)

axiom prob_red : P_red = 0.25
axiom prob_green : P_green = 0.35
axiom sum_prob : P_red + P_green + P_orange_yellow = 1

theorem combined_prob_orange_yellow : P_orange_yellow = 0.4 := by
  have h1 : P_red + P_green = 0.25 + 0.35 := by rw [prob_red, prob_green]
  have h2 : 0.25 + 0.35 = 0.6 := by norm_num
  rw [h2] at h1
  have h3 : P_red + P_green + P_orange_yellow = 0.6 + P_orange_yellow := by rw [h1]
  rw [h3] at sum_prob
  have h4 : 0.6 + P_orange_yellow = 1 := by rw [sum_prob]
  have h5 : P_orange_yellow = 1 - 0.6 := by rw [← h4]
  norm_num at h5
  exact h5 or sorry

end combined_prob_orange_yellow_l400_400894


namespace range_of_k_l400_400809

noncomputable def log_function (a x k : ℝ) : ℝ :=
  log (a) (x - k * a) + log (a) (x^2 - a^2)

theorem range_of_k (a : ℝ) (h : a ≠ 0) : 
  (∀ x, x > a → x > k * a) → 
  (∀ x, x > a → x^2 > a^2) → 
  k ∈ set.Icc (-1 : ℝ) 1 := 
by
  -- Proof is omitted and replaced by sorry
  sorry

end range_of_k_l400_400809


namespace prime_count_between_50_and_60_l400_400345

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400345


namespace bowling_ball_weight_l400_400795

variable {b c : ℝ}

theorem bowling_ball_weight :
  (10 * b = 4 * c) ∧ (3 * c = 108) → b = 14.4 :=
by
  sorry

end bowling_ball_weight_l400_400795


namespace number_of_primes_between_50_and_60_l400_400371

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400371


namespace boat_distance_against_stream_l400_400704

/-- 
  Given:
  1. The boat goes 13 km along the stream in one hour.
  2. The speed of the boat in still water is 11 km/hr.

  Prove:
  The distance the boat goes against the stream in one hour is 9 km.
-/
theorem boat_distance_against_stream (v_s : ℝ) (distance_along_stream time : ℝ) (v_still : ℝ) :
  distance_along_stream = 13 ∧ time = 1 ∧ v_still = 11 ∧ (v_still + v_s) = 13 → 
  (v_still - v_s) * time = 9 := by
  sorry

end boat_distance_against_stream_l400_400704


namespace primes_between_50_and_60_l400_400410

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400410


namespace hannah_total_spending_l400_400210

def sweatshirt_price : ℕ := 15
def sweatshirt_quantity : ℕ := 3
def t_shirt_price : ℕ := 10
def t_shirt_quantity : ℕ := 2
def socks_price : ℕ := 5
def socks_quantity : ℕ := 4
def jacket_price : ℕ := 50
def discount_rate : ℚ := 0.10

noncomputable def total_cost_before_discount : ℕ :=
  (sweatshirt_quantity * sweatshirt_price) +
  (t_shirt_quantity * t_shirt_price) +
  (socks_quantity * socks_price) +
  jacket_price

noncomputable def total_cost_after_discount : ℚ :=
  total_cost_before_discount - (discount_rate * total_cost_before_discount)

theorem hannah_total_spending : total_cost_after_discount = 121.50 := by
  sorry

end hannah_total_spending_l400_400210


namespace area_PQR_l400_400818

-- Defining the points based on the conditions
def P := (5, 3)
def Q := (-5, 3)
def R := (-3, -5)

-- Formula for area of a triangle given 3 points
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_PQR :
  area_of_triangle P Q R = 40 :=
by
  -- Proof can be added here
  sorry

end area_PQR_l400_400818


namespace conjugate_z_is_i_l400_400984

constant z : ℂ
axiom h : (1 - z) / (1 + z) = complex.I

theorem conjugate_z_is_i : complex.conj z = complex.I :=
by
  sorry

end conjugate_z_is_i_l400_400984


namespace sale_price_lower_than_original_l400_400006

theorem sale_price_lower_than_original (x : ℝ) (hx : 0 ≤ x) : 
  let increased_price := 1.30 * x in 
  let sale_price := 0.75 * increased_price in 
  sale_price < x :=
by
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price
  sorry

end sale_price_lower_than_original_l400_400006


namespace inequality_solution_l400_400781

theorem inequality_solution (x : ℝ) : 
  (x / (x + 5) ≥ 0) ↔ (x ∈ (Set.Iio (-5)).union (Set.Ici 0)) :=
by
  sorry

end inequality_solution_l400_400781


namespace graph_of_y_eq_f_abs_x_is_A_graph_of_y_eq_f_abs_neg_x_is_A_graph_A_is_reflection_l400_400810

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then real.sqrt(4 - (x - 2) ^ 2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

theorem graph_of_y_eq_f_abs_x_is_A :
  ∀ x : ℝ, 0 ≤ x → f (|x|) = f x :=
begin
  intro x,
  intro hx,
  have : |x| = x,
  { exact abs_of_nonneg hx },
  rw this,
  refl,
end

theorem graph_of_y_eq_f_abs_neg_x_is_A :
  ∀ x : ℝ, x < 0 → f (|x|) = f (-x) :=
begin
  intro x,
  intro hx,
  have : |x| = -x,
  { exact abs_of_neg hx },
  rw this,
  refl,
end

theorem graph_A_is_reflection :
  ∀ x : ℝ, f (|x|) = if x ≥ 0 then f x else f (-x) :=
begin
  intro x,
  split_ifs,
  { exact graph_of_y_eq_f_abs_x_is_A x h },
  { exact graph_of_y_eq_f_abs_neg_x_is_A x h },
end

end graph_of_y_eq_f_abs_x_is_A_graph_of_y_eq_f_abs_neg_x_is_A_graph_A_is_reflection_l400_400810


namespace time_for_B_alone_l400_400866

theorem time_for_B_alone (W_A W_B : ℝ) (h1 : W_A = 2 * W_B) (h2 : W_A + W_B = 1/6) : 1 / W_B = 18 := by
  sorry

end time_for_B_alone_l400_400866


namespace radius_of_original_bubble_l400_400074

-- Define the radius of the hemisphere
def r_hemisphere : ℝ := 5

-- Define the volume of a sphere
def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

-- Define the volume of a hemisphere
def volume_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

-- State the theorem
theorem radius_of_original_bubble :
  ∀ r_sphere, volume_sphere r_sphere = volume_hemisphere r_hemisphere →
    r_sphere = 5 / Real.cbrt 2 := 
by
  sorry

end radius_of_original_bubble_l400_400074


namespace minimal_plate_diameter_l400_400886

theorem minimal_plate_diameter (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) 
  (sides_eq : {a, b, c} = {19, 20, 21}) : 
    ∃ (d : ℕ), (∀ (cut : set (ℕ × ℕ × ℕ)), cut ⊆ { t : ℕ × ℕ × ℕ | t.1 = a ∨ t.2 = a ∨ t.3 = a } 
                 → ∀ (x ∈ cut) (y ∈ cut), disjoint x.to_finset y.to_finset 
                         → (∀ (z ∈ {x, y}), ∀ (w ∈ z.to_finset), w ≤ d) 
                         ∧ (∀ (z ∈ {x, y}), ∃ (r : ℝ), r ≤ d / 2 ∧ (w ∈ z.to_finset → (w : ℝ)^2 ≤ r^2)) 
              ) 
           ∧ minimal {p : ℕ | ∀ (a b : ℕ), a = b → p ≥ a} d 
           ∧ d = 21 := 
sorry

end minimal_plate_diameter_l400_400886


namespace Parallelepiped_intersection_vector_l400_400717

noncomputable def vector_equal_to_B1M (a b c : ℝ^3) : Prop :=
  let A1B1 := a
  let A1D1 := b
  let A1A := c
  ∃ M : ℝ^3, M = (-1/2 : ℝ) • a + (1/2 : ℝ) • b + c

theorem Parallelepiped_intersection_vector (a b c : ℝ^3) : 
  vector_equal_to_B1M a b c := 
by 
  sorry

end Parallelepiped_intersection_vector_l400_400717


namespace count_primes_50_60_l400_400481

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400481


namespace prime_count_50_to_60_l400_400513

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400513


namespace primes_between_50_and_60_l400_400644

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400644


namespace primes_between_50_and_60_l400_400218

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400218


namespace count_primes_between_50_and_60_l400_400671

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400671


namespace primes_between_50_and_60_l400_400310

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400310


namespace danny_wrappers_more_than_soda_cans_l400_400936

theorem danny_wrappers_more_than_soda_cans :
  (67 - 22 = 45) := sorry

end danny_wrappers_more_than_soda_cans_l400_400936


namespace calc_expression_l400_400930

theorem calc_expression : 3 ^ 2022 * (1 / 3) ^ 2023 = 1 / 3 :=
by
  sorry

end calc_expression_l400_400930


namespace prime_count_between_50_and_60_l400_400577

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400577


namespace area_of_rectangle_l400_400040

theorem area_of_rectangle
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_a : a = 16)
  (h_c : c = 17)
  (h_diag : a^2 + b^2 = c^2) :
  abs (a * b - 91.9136) < 0.0001 :=
by
  sorry

end area_of_rectangle_l400_400040


namespace continuous_stripe_probability_l400_400945

def cube_stripe_probability : ℚ :=
  let stripe_combinations_per_face := 8
  let total_combinations := stripe_combinations_per_face ^ 6
  let valid_combinations := 4 * 3 * 8 * 64
  let probability := valid_combinations / total_combinations
  probability

theorem continuous_stripe_probability :
  cube_stripe_probability = 3 / 128 := by
  sorry

end continuous_stripe_probability_l400_400945


namespace negation_of_prop_l400_400814

-- Definitions based on the conditions
def pos_real (x : ℝ) : Prop := x > 0
def frac_term (x : ℝ) : ℝ := x / (x - 1)
def prop (x : ℝ) : Prop := frac_term x > 0

-- Theorem statement for the negation of the original proposition
theorem negation_of_prop : (¬ ∀ x : ℝ, pos_real x → prop x) ↔ ∃ x : ℝ, pos_real x ∧ frac_term x ≤ 0 ∧ 0 ≤ x ∧ x < 1 :=
by
  sorry

end negation_of_prop_l400_400814


namespace number_of_primes_between_50_and_60_l400_400363

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400363


namespace primes_between_50_and_60_l400_400592

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400592


namespace prime_count_50_to_60_l400_400493

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400493


namespace min_value_y_l400_400850

theorem min_value_y : ∃ x : ℝ, (∀ y : ℝ, y = x^2 + 16 * x + 20 → y ≥ -44) :=
begin
  use -8,
  intro y,
  intro hy,
  suffices : y = (x + 8)^2 - 44,
  { rw this,
    exact sub_nonneg_of_le (sq_nonneg (x + 8)) },
  sorry
end

end min_value_y_l400_400850


namespace max_angle_ACB_l400_400797

theorem max_angle_ACB (A B C D E I : ℝ) 
  (h1 : ∠ACB = 60 ∧ (triangle_area A B I = quadrilateral_area C D I E)) :
  ∠ACB = 60 :=
sorry

end max_angle_ACB_l400_400797


namespace original_cow_offspring_25_years_l400_400793

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem original_cow_offspring_25_years :
  fibonacci 25 = 75025 := 
sorry

end original_cow_offspring_25_years_l400_400793


namespace primes_between_50_and_60_l400_400415

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400415


namespace geometric_probability_l400_400064

open Set

noncomputable def probability (s : Set ℝ) : ℝ := (volume s).toReal

theorem geometric_probability :
  let s := {x : ℝ | 0 ≤ x ∧ x ≤ 1}
  let t := {x : ℝ | 3 * x - 1 > 0 ∧ 0 ≤ x ∧ x ≤ 1}
  probability t / probability s = 2 / 3 :=
by
  let s := {x : ℝ | 0 ≤ x ∧ x ≤ 1}
  let t := {x : ℝ | 3 * x - 1 > 0 ∧ 0 ≤ x ∧ x ≤ 1}
  sorry

end geometric_probability_l400_400064


namespace repetend_of_7_over_29_l400_400124

theorem repetend_of_7_over_29 : ∃ r : ℕ, r = 241379 ∧ (∃ m : ℕ, ∃ n : ℕ, r = (7 * 10^n) % 29 ∧ (7 * 10 ^ m / 29).decimalRepeats 6) :=
sorry

end repetend_of_7_over_29_l400_400124


namespace primes_between_50_and_60_l400_400532

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400532


namespace primes_count_between_50_and_60_l400_400547

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400547


namespace primes_between_50_and_60_l400_400226

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400226


namespace prime_count_between_50_and_60_l400_400569

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400569


namespace area_of_triangle_DEF_l400_400905

noncomputable def area_u1 : ℝ := 25
noncomputable def area_u2 : ℝ := 16
noncomputable def area_u3 : ℝ := 36

theorem area_of_triangle_DEF :
  ∃ (Q : Point), ∃ (DEF u1 u2 u3 : Triangle), 
    is_inside Q DEF ∧
    is_parallel DEF u1 u2 u3 Q ∧
    area u1 = area_u1 ∧
    area u2 = area_u2 ∧
    area u3 = area_u3 → 
    area DEF = 144 :=
sorry

end area_of_triangle_DEF_l400_400905


namespace factorize_x_squared_sub_xy_l400_400117

theorem factorize_x_squared_sub_xy (x y : ℝ) : x^2 - x * y = x * (x - y) :=
sorry

end factorize_x_squared_sub_xy_l400_400117


namespace carmen_four_numbers_sum_eleven_largest_is_five_l400_400098

theorem carmen_four_numbers_sum_eleven_largest_is_five
  (a b c l : ℕ)
  (h_range : {a, b, c, l} ⊆ {1, 2, 3, 4, 5, 6, 7})
  (h_distinct : ∀ x y, x ≠ y → x ∈ {a, b, c, l} → y ∈ {a, b, c, l} → x ≠ y)
  (h_sum : a + b + c + l = 11)
  (h_order : a < b ∧ b < c ∧ c < l) :
  l = 5 :=
sorry

end carmen_four_numbers_sum_eleven_largest_is_five_l400_400098


namespace complex_number_in_second_quadrant_l400_400025

noncomputable def complex_number (m : ℝ) : ℂ :=
  (m - 1) + (3 * m - 2) * complex.I

theorem complex_number_in_second_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) :
  0 < (3 * m - 2) ∧ (m - 1) < 0 :=
by
  sorry

end complex_number_in_second_quadrant_l400_400025


namespace true_proposition_l400_400182

variable (p q : Prop)

-- Proposition p: ∀ x ∈ (-∞, 0), 2^x > 3^x
def proposition_p : Prop :=
  ∀ x : ℝ, x < 0 → 2^x > 3^x

-- Proposition q: ∃ x ∈ (0, π/2), sin x > x
def proposition_q : Prop :=
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ sin x > x

theorem true_proposition (hp : proposition_p p) (hq : proposition_q q) : p ∧ ¬q := 
sorry

end true_proposition_l400_400182


namespace prime_count_50_to_60_l400_400512

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400512


namespace lying_turtle_possible_line_l400_400834

def turtles_in_line (T1_statement T2_statement T3_statement : Prop) : Prop :=
  (T1_statement = (∃ T2 T3, T2 ≠ T1 ∧ T3 ≠ T1 ∧ T2 ≠ T3)) ∧ 
  (T2_statement = (T1 ∧ T3 ≠ T2 ∧ T2 ≠ T1)) ∧ 
  (T3_statement = (T1 ∧ T2 ≠ T3 ∧ T3 ≠ T1)) ∧
  ∃ some_turtle_is_lying, some_turtle_is_lying ∧ ¬(T1_statement ∧ T2_statement ∧ T3_statement)

theorem lying_turtle_possible_line (T1_statement T2_statement T3_statement : Prop) :
  turtles_in_line T1_statement T2_statement T3_statement :=
sorry

end lying_turtle_possible_line_l400_400834


namespace profit_ratio_l400_400822

-- Define sale price (SP) and cost price (CP)
variables (SP CP : ℝ)

-- Given condition: the ratio between sale price and cost price is 6:2
axiom ratio_SP_CP : SP = 3 * CP

-- Define profit (P) as the difference between sale price and cost price
noncomputable def P : ℝ := SP - CP

-- Theorem stating the ratio of the profit (P) to the cost price (CP) is 2:1
theorem profit_ratio : (P CP ratio_SP_CP) / CP = 2 :=
by
  sorry

end profit_ratio_l400_400822


namespace prime_count_between_50_and_60_l400_400340

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400340


namespace pet_store_parakeets_l400_400068

variable (c : ℝ) (p : ℝ) (avg : ℝ)

-- Define the number of cages, parrots, and the average number of birds per cage.
def number_of_cages : ℝ := 6.0
def number_of_parrots : ℝ := 6.0
def birds_per_cage : ℝ := 1.333333333

theorem pet_store_parakeets :
  let totalBirds := c * avg in
  let parakeets := totalBirds - p in
  c = 6.0 → p = 6.0 → avg = 1.333333333 → parakeets = 2.0 :=
by
  sorry

end pet_store_parakeets_l400_400068


namespace num_primes_between_50_and_60_l400_400628

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400628


namespace diagonal_le_2_l400_400167

open EuclideanGeometry

-- Definitions of vertices and lengths
variables {A B C D E F : Point}

-- Condition: Hexagon sides have length at most 1 unit
axiom side_AB_le_1 : dist A B ≤ 1
axiom side_BC_le_1 : dist B C ≤ 1
axiom side_CD_le_1 : dist C D ≤ 1
axiom side_DE_le_1 : dist D E ≤ 1
axiom side_EF_le_1 : dist E F ≤ 1
axiom side_FA_le_1 : dist F A ≤ 1

-- Hexagon ABCDEF is convex
axiom convex_hexagon : ConvexHexagon A B C D E F

-- Goal: Prove that at least one of the diagonals AD, BE, or CF is less than or equal to 2 units
theorem diagonal_le_2 : dist A D ≤ 2 ∨ dist B E ≤ 2 ∨ dist C F ≤ 2 :=
sorry

end diagonal_le_2_l400_400167


namespace sum_of_consecutive_even_integers_l400_400023

theorem sum_of_consecutive_even_integers
  (a1 a2 a3 a4 : ℤ)
  (h1 : a2 = a1 + 2)
  (h2 : a3 = a1 + 4)
  (h3 : a4 = a1 + 6)
  (h_sum : a1 + a3 = 146) :
  a1 + a2 + a3 + a4 = 296 :=
by sorry

end sum_of_consecutive_even_integers_l400_400023


namespace part_a_arrangement_part_b_no_shared_numbers_between_boys_and_girls_l400_400046

-- Define the Lean 4 theorem statement for part (a)
theorem part_a_arrangement (students : Fin 14 → Set ℕ) :
  (∀ s, (students s).card ≥ 2) → ∃ (arrangement : Fin 14 → ℕ),
  (∀ i : Fin 14, arrangement i ∈ students i) ∧ (∀ i : Fin 14, arrangement i ≠ arrangement (i + 1)) :=
sorry

-- Define the Lean 4 theorem statement for part (b)
theorem part_b_no_shared_numbers_between_boys_and_girls 
  (boys girls : Fin 7 → Set ℕ) :
  (∀ b, (boys b).card ≥ 4) → (∀ g, (girls g).card ≥ 4) →
  (∀ b g, b ≠ g → disjoint (boys b) (girls g)) → 
  ∃ (assignment : Fin 14 → ℕ), 
    (∀ i : Fin 7, assignment i ∈ boys i) ∧ 
    (∀ j : (Fin 14) // j.val ≥ 7, assignment j ∈ girls (⟨j.val - 7, by simp⟩)) :=
sorry

end part_a_arrangement_part_b_no_shared_numbers_between_boys_and_girls_l400_400046


namespace primes_between_50_and_60_l400_400234

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400234


namespace primes_between_50_and_60_l400_400518

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400518


namespace primes_count_between_50_and_60_l400_400545

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400545


namespace fred_money_left_l400_400150

theorem fred_money_left 
  (initial_dollars : ℕ)
  (number_of_books : ℕ)
  (average_cost_per_book : ℕ)
  (total_cost_books : ℕ)
  (money_left : ℕ)
  (h_initial : initial_dollars = 236)
  (h_books : number_of_books = 6)
  (h_cost : average_cost_per_book = 37)
  (h_total_cost : total_cost_books = number_of_books * average_cost_per_book)
  (h_total_cost_value : total_cost_books = 222)
  (h_money_left : money_left = initial_dollars - total_cost_books)
  (h_money_left_value : money_left = 14):
  money_left = 14 :=
by 
  rw [h_initial, h_books, h_cost, h_total_cost, h_total_cost_value, h_money_left]
  exact h_money_left_value

end fred_money_left_l400_400150


namespace prime_count_50_to_60_l400_400500

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400500


namespace num_integer_solutions_eq_2_l400_400736

def polynomial (x : ℤ) : ℤ :=
  x^4 + 4 * x^3 + 10 * x^2 + 4 * x + 24

theorem num_integer_solutions_eq_2 :
  {x : ℤ | ∃ c : ℤ, c^2 = polynomial x}.to_finset.card = 2 :=
by sorry

end num_integer_solutions_eq_2_l400_400736


namespace primes_between_50_and_60_l400_400386

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400386


namespace prime_count_between_50_and_60_l400_400565

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400565


namespace top_card_spades_or_diamonds_l400_400911

theorem top_card_spades_or_diamonds :
  let suits := ["♠", "♣", "♥", "♦"]
  let ranks := (1: Fin 14 → String) -- Assume ranks are represented by 1 to 13
  let deck := List.product (List.ofFn ranks) suits
  ∃ p : ℚ, p = 1/2 :=
by
  sorry

end top_card_spades_or_diamonds_l400_400911


namespace SinTransformation_l400_400116

theorem SinTransformation :
  (∀ x, y = sin x →
    (∀ x, y = sin (x / 2) →
      (∀ x, y = sin (x / 2 - π / 6) → true))) :=
by
  intro x hx
  sorry

end SinTransformation_l400_400116


namespace num_primes_between_50_and_60_l400_400619

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400619


namespace discount_per_person_correct_l400_400083

noncomputable def price_per_person : ℕ := 147
noncomputable def total_people : ℕ := 2
noncomputable def total_cost_with_discount : ℕ := 266

theorem discount_per_person_correct :
  let total_cost_without_discount := price_per_person * total_people
  let total_discount := total_cost_without_discount - total_cost_with_discount
  let discount_per_person := total_discount / total_people
  discount_per_person = 14 := by
  sorry

end discount_per_person_correct_l400_400083


namespace jasmine_percent_after_addition_l400_400049

-- Variables definition based on the problem
def original_volume : ℕ := 90
def original_jasmine_percent : ℚ := 0.05
def added_jasmine : ℕ := 8
def added_water : ℕ := 2

-- Total jasmine amount calculation in original solution
def original_jasmine_amount : ℚ := original_jasmine_percent * original_volume

-- New total jasmine amount after addition
def new_jasmine_amount : ℚ := original_jasmine_amount + added_jasmine

-- New total volume calculation after addition
def new_total_volume : ℕ := original_volume + added_jasmine + added_water

-- New jasmine percent in the solution
def new_jasmine_percent : ℚ := (new_jasmine_amount / new_total_volume) * 100

-- The proof statement
theorem jasmine_percent_after_addition : new_jasmine_percent = 12.5 :=
by
  sorry

end jasmine_percent_after_addition_l400_400049


namespace line_and_parabola_intersection_l400_400171

theorem line_and_parabola_intersection (A B M : ℝ × ℝ) (L : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop)
  (hC : ∀ x y, C x y ↔ y^2 = 4 * x)
  (hM : M = (3, 2))
  (hL : ∀ x y, L x y ↔ x - y - 1 = 0) :
  (∃ P Q : ℝ × ℝ, C P.1 P.2 ∧ C Q.1 Q.2 ∧ L P.1 P.2 ∧ L Q.1 Q.2 ∧ ((P.1 + Q.1) / 2 = 3) ∧ ((P.2 + Q.2) / 2 = 2)) ∧
  (let P := (1 + 2 * (3 - 1), 4 - 2) in 
   dist (P.1, P.2) (7 - P.1, 6 - P.2) = 8) := 
sorry

end line_and_parabola_intersection_l400_400171


namespace primes_between_50_and_60_l400_400517

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400517


namespace RSVP_yes_percentage_l400_400771

variable (total_guests : ℕ :=
variable (percentage_no_response : ℕ :=  percentage_no_response : 16
 
def percentage_yes_response (total_guests percentage_no_response num_not_responded : ℕ) : ℕ :=
(total_guests - (0.09 * total_guests + num_not_responded)) * 100 / total_guests

theorem RSVP_yes_percentage (h1 : total_guests = 200) (h2 : percentage_no_response = 9) (h3 : num_not_responded = 16) :
    percentage_yes_response total_guests percentage_no_response num_not_responded = 83 :=
by
  sorry

end RSVP_yes_percentage_l400_400771


namespace primes_between_50_and_60_l400_400380

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400380


namespace time_for_A_to_complete_work_l400_400887

theorem time_for_A_to_complete_work (W : ℝ) (A B C : ℝ) (W_pos : 0 < W) (B_work : B = W / 40) (C_work : C = W / 20) : 
  (10 * (W / A) + 10 * (W / B) + 10 * (W / C) = W) → A = W / 40 :=
by 
  sorry

end time_for_A_to_complete_work_l400_400887


namespace primes_between_50_and_60_l400_400233

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400233


namespace num_primes_between_50_and_60_l400_400629

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400629


namespace eval_expression_l400_400115

theorem eval_expression : 81^(-1/4) - 16^(-3/4) = (5 / 24) := by
  sorry

end eval_expression_l400_400115


namespace solve_system_l400_400784

theorem solve_system (x y : ℝ) (h₁ : 3^y * 81 = 9^(x^2)) (h₂ : log 10 y = log 10 x - log 10 0.5) :
  x = 2 ∧ y = 4 :=
by
  sorry

end solve_system_l400_400784


namespace primes_between_50_and_60_l400_400330

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400330


namespace chef_earns_2_60_less_l400_400089

/--
At Joe's Steakhouse, the hourly wage for a chef is 20% greater than that of a dishwasher,
and the hourly wage of a dishwasher is half as much as the hourly wage of a manager.
If a manager's wage is $6.50 per hour, prove that a chef earns $2.60 less per hour than a manager.
-/
theorem chef_earns_2_60_less {w_manager w_dishwasher w_chef : ℝ} 
  (h1 : w_dishwasher = w_manager / 2)
  (h2 : w_chef = w_dishwasher * 1.20)
  (h3 : w_manager = 6.50) :
  w_manager - w_chef = 2.60 :=
by
  sorry

end chef_earns_2_60_less_l400_400089


namespace f_increasing_on_0_2pi_l400_400808

def f (x : ℝ) := 1 + x - sin x

theorem f_increasing_on_0_2pi : ∀ x y ∈ set.Ioo 0 (2 * Real.pi), x < y → f x < f y :=
by
  intros x hx y hy hxy
  sorry

end f_increasing_on_0_2pi_l400_400808


namespace count_primes_between_50_and_60_l400_400240

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400240


namespace cd_bisects_ab_l400_400207

noncomputable def radical_axis_of_two_circles (Γ₁ Γ₂ : Circle) : Set Point := 
  {M | pow Γ₁ M = pow Γ₂ M}

theorem cd_bisects_ab 
    (Γ₁ Γ₂ : Circle) 
    (C D A B : Point) 
    (tangent_to_Γ₁_at_A : Tangent Γ₁ A)
    (tangent_to_Γ₂_at_B : Tangent Γ₂ B)
    (intersection_C : PointOn C Γ₁ ∧ PointOn C Γ₂)
    (intersection_D : PointOn D Γ₁ ∧ PointOn D Γ₂)
    (radical_axis_CD : LineThrough CD ∈ radical_axis_of_two_circles Γ₁ Γ₂)
    (M := LineIntersection (LineThrough C D) (LineThrough A B)) :
  midpoint M A B :=
sorry

end cd_bisects_ab_l400_400207


namespace insurance_covers_90_percent_l400_400724

-- We firstly define the variables according to the conditions.
def adoption_fee : ℕ := 150
def training_cost_per_week : ℕ := 250
def training_weeks : ℕ := 12
def certification_cost : ℕ := 3000
def total_out_of_pocket_cost : ℕ := 3450

-- We now compute intermediate results based on the conditions provided.
def total_training_cost : ℕ := training_cost_per_week * training_weeks
def out_of_pocket_cert_cost : ℕ := total_out_of_pocket_cost - adoption_fee - total_training_cost
def insurance_coverage_amount : ℕ := certification_cost - out_of_pocket_cert_cost
def insurance_coverage_percentage : ℕ := (insurance_coverage_amount * 100) / certification_cost

-- Now, we state the theorem that needs to be proven.
theorem insurance_covers_90_percent : insurance_coverage_percentage = 90 := by
  sorry

end insurance_covers_90_percent_l400_400724


namespace primes_between_50_and_60_l400_400384

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400384


namespace good_numbers_l400_400898

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ (n = 1 ∨ (nat.prime n ∧ n % 2 = 1)) :=
by
  sorry

end good_numbers_l400_400898


namespace exists_parallel_plane_l400_400148

-- Define the basic structures and properties in Lean
structure Line (α : Type*) := (points : set α)
structure Plane (α : Type*) := (points : set α)

-- Non-intersecting (skew) lines a and b
variable {α : Type*} [LinearOrder α]
variables (a b : Line α) [SkewLines a b]

-- the statement to be proven
theorem exists_parallel_plane (a b : Line α) [SkewLines a b] :
  ∃ (α : Plane α), Parallel (a, α) ∧ Parallel (b, α) :=
sorry

end exists_parallel_plane_l400_400148


namespace derivative_problem_1_derivative_problem_2_derivative_problem_3_derivative_problem_4_derivative_problem_5_derivative_problem_6_l400_400844

-- Proof problem based on problem 1
theorem derivative_problem_1 :
  ∀ x : ℝ, deriv (λ x, x^(5/6) + 7) x = (5/6) * x^(-1/6) :=
by
  sorry

-- Proof problem based on problem 2
theorem derivative_problem_2 :
  ∀ x : ℝ, deriv (λ x, 3 * x^(13/5) - 9 * x^(-2/3) + 2 * x^(5/6)) x = 
    (39/5) * x^(8/5) + 6 * x^(-5/3) + (5/3) * x^(-1/6) :=
by
  sorry

-- Proof problem based on problem 3
theorem derivative_problem_3 :
  ∀ x : ℝ, deriv (λ x, x^(4/3) + x^(-1/2) + 0.1 * x^10) x = 
    (4/3) * x^(1/3) - (1/2) * x^(-3/2) + 1 * x^9 :=
by
  sorry

-- Proof problem based on problem 4
theorem derivative_problem_4 :
  ∀ x : ℝ, deriv (λ x, (2 * x^3 + sqrt 5) * 7^x) x = 
    (6 * x^2 + (2 * x^3 + sqrt 5) * log 7) * 7^x :=
by
  sorry

-- Proof problem based on problem 5
theorem derivative_problem_5 :
  ∀ x : ℝ, deriv (λ x, x / (2 - cos x) - x^3 / sqrt 7) x = 
    (2 - cos x - x * sin x) / (2 - cos x)^2 - (3 * x^2) / sqrt 7 :=
by
  sorry

-- Proof problem based on problem 6
theorem derivative_problem_6 :
  ∀ x : ℝ, deriv (λ x, 5 / sin x + log x / x^2) x = 
    - (5 * cos x) / (sin x)^2 + (1 - 2 * log x) / x^3 :=
by
  sorry

end derivative_problem_1_derivative_problem_2_derivative_problem_3_derivative_problem_4_derivative_problem_5_derivative_problem_6_l400_400844


namespace min_dot_product_tangent_points_l400_400104

noncomputable def circle_equation (t : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1 - t)^2 + (p.2 - t + 2)^2 = 1 }

def pointP : ℝ × ℝ := (-1, 1)

theorem min_dot_product_tangent_points :
  ∀ t ∈ ℝ, ∃ A B : ℝ × ℝ, 
   A ∈ circle_equation t ∧ B ∈ circle_equation t ∧
   (tangent_from P A) ∧ (tangent_from P B) ∧
   ∃ m : ℝ, (dot_product (vector PA) (vector PB)) = m ∧
   m = 21 / 4 :=
begin
  sorry
end

end min_dot_product_tangent_points_l400_400104


namespace primes_between_50_and_60_l400_400533

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400533


namespace primes_between_50_and_60_l400_400395

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400395


namespace solve_system_l400_400787

namespace SolutionProof

-- Conditions
def equation1 (x y : ℝ) : Prop := 3^y * 81 = 9^(x^2)
def equation2 (x y : ℝ) : Prop := log 10 y = log 10 x - log 10 0.5

-- Proof problem statement
theorem solve_system : ∃ x y : ℝ, equation1 x y ∧ equation2 x y ∧ x = 2 ∧ y = 4 :=
by {
  sorry 
}

end SolutionProof

end solve_system_l400_400787


namespace count_primes_between_50_and_60_l400_400454

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400454


namespace part_I_part_II_l400_400161

section part_I

variable (m : ℝ)

def prop_p (x : ℝ) := m^2 - 3 * m + x - 1 ≤ 0
def prop_q (x : ℝ) (a : ℝ) := m ≤ a * x

theorem part_I (a == 1) :
  ¬(∀ x ∈ (-1:ℝ)..1, prop_p m x) ∧ ¬(∃ x ∈ (-1..1), prop_q m x 1) ∧ 
  ((∀ x ∈ (-1:ℝ)..1, prop_p m x) ∨ (∃ x ∈ (-1..1), prop_q m x 1)) →
  m ∈ (Set.Iic 1 ∪ Set.Ioc 1 2) := sorry

end part_I


section part_II

variable (m a : ℝ)

theorem part_II :
  (∀ x ∈ (-1:ℝ)..1, prop_p m x) → (∃ x ∈ (-1..1), prop_q m x a) ∧
  (∃ x ∈ (-1..1), prop_q m x a) → (¬∀ x ∈ (-1:ℝ)..1, prop_p m x) →
  a ∈ (Set.Iic (-2) ∪ Set.Ici 2) := sorry

end part_II

end part_I_part_II_l400_400161


namespace winner_determined_in_7th_game_l400_400885

def prob_win_given_game (p : ℚ) (i : ℕ) : ℚ := 
  if i = 1 then p else (1 - p) * prob_win_given_game p (i - 1)

def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

def probability_7th_game_series_ends (p : ℚ) : ℚ :=
  let q := 1 - p in
  let mathletes_win_first_6_count := binomial_coefficient 6 4 in
  let mathletes_win_first_6_prob := mathletes_win_first_6_count * p^4 * q^2 in
  let opponent_win_first_6_count := binomial_coefficient 6 4 in
  let opponent_win_first_6_prob := opponent_win_first_6_count * q^4 * p^2 in
  let mathletes_win_7th_game_prob := p in
  let opponent_win_7th_game_prob := q in
  (mathletes_win_first_6_prob * mathletes_win_7th_game_prob) +
  (opponent_win_first_6_prob * opponent_win_7th_game_prob)

theorem winner_determined_in_7th_game :
  probability_7th_game_series_ends (2/3) = 20 / 81 :=
by 
  sorry

end winner_determined_in_7th_game_l400_400885


namespace dozen_bagels_cost_l400_400800

-- Define the cost per bagel
def cost_per_bagel : ℝ := 2.25

-- Define the saving per bagel when buying a dozen
def saving_per_bagel : ℝ := 0.25

-- Define the number of bagels in a dozen
def dozen : ℝ := 12

-- Define the cost of a dozen bagels before savings
def original_cost_of_a_dozen_bagels : ℝ := cost_per_bagel * dozen

-- Define the total savings for a dozen bagels
def total_savings : ℝ := saving_per_bagel * dozen

-- Define the final cost of a dozen bagels after savings
def cost_of_a_dozen_bagels_with_savings : ℝ := original_cost_of_a_dozen_bagels - total_savings

-- The theorem proving the cost of a dozen bagels after savings
theorem dozen_bagels_cost : cost_of_a_dozen_bagels_with_savings = 24 :=
by
  -- This is where the proof would go
  sorry

#eval dozen_bagels_cost  -- This should output true if the theorem is correct

end dozen_bagels_cost_l400_400800


namespace count_primes_50_60_l400_400487

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400487


namespace prime_count_between_50_and_60_l400_400579

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400579


namespace felix_trees_chopped_l400_400950

-- Given conditions
def cost_per_sharpening : ℕ := 8
def total_spent : ℕ := 48
def trees_per_sharpening : ℕ := 25

-- Lean statement of the problem
theorem felix_trees_chopped (h : total_spent / cost_per_sharpening * trees_per_sharpening >= 150) : True :=
by {
  -- This is just a placeholder for the proof.
  sorry
}

end felix_trees_chopped_l400_400950


namespace min_value_y_l400_400856

theorem min_value_y (x : ℝ) : ∃ (x : ℝ), y = x^2 + 16 * x + 20 → y ≥ -44 :=
begin
  sorry
end

end min_value_y_l400_400856


namespace prime_count_between_50_and_60_l400_400570

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400570


namespace general_form_line_eq_line_passes_fixed_point_l400_400169

-- (Ⅰ) Prove that if m = 1/2 and point P (1/2, 2), the general form equation of line l is 2x - y + 1 = 0
theorem general_form_line_eq (m n : ℝ) (h1 : m = 1/2) (h2 : n = 1 / (1 - m)) (h3 : n = 2) (P : (ℝ × ℝ)) (hP : P = (1/2, 2)) :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 1 := sorry

-- (Ⅱ) Prove that if point P(m,n) is on the line l0, then the line mx + (n-1)y + n + 5 = 0 passes through a fixed point, coordinates (1,1)
theorem line_passes_fixed_point (m n : ℝ) (h1 : m + 2 * n + 4 = 0) :
  ∀ (x y : ℝ), (m * x + (n - 1) * y + n + 5 = 0) ↔ (x = 1) ∧ (y = 1) := sorry

end general_form_line_eq_line_passes_fixed_point_l400_400169


namespace prime_count_between_50_and_60_l400_400564

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400564


namespace primes_between_50_and_60_l400_400590

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400590


namespace primes_between_50_and_60_l400_400292

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400292


namespace chris_dana_shared_rest_days_l400_400931

/-- Chris's and Dana's working schedules -/
structure work_schedule where
  work_days : ℕ
  rest_days : ℕ

/-- Define Chris's and Dana's schedules -/
def Chris_schedule : work_schedule := { work_days := 5, rest_days := 2 }
def Dana_schedule : work_schedule := { work_days := 6, rest_days := 1 }

/-- Number of days to consider -/
def total_days : ℕ := 1200

/-- Combinatorial function to calculate the number of coinciding rest-days -/
noncomputable def coinciding_rest_days (schedule1 schedule2 : work_schedule) (days : ℕ) : ℕ :=
  (days / (Nat.lcm (schedule1.work_days + schedule1.rest_days) (schedule2.work_days + schedule2.rest_days)))

/-- The proof problem statement -/
theorem chris_dana_shared_rest_days : 
coinciding_rest_days Chris_schedule Dana_schedule total_days = 171 :=
by sorry

end chris_dana_shared_rest_days_l400_400931


namespace multiply_same_exponents_l400_400095

theorem multiply_same_exponents (x : ℝ) : (x^3) * (x^3) = x^6 :=
by sorry

end multiply_same_exponents_l400_400095


namespace probability_first_heart_second_ace_l400_400842

noncomputable def deck := finset.range 51

noncomputable def card (n : ℕ) : Prop :=
      true -- This dummy definition can be refined to accurate card representation 

noncomputable def is_heart (n : ℕ) : Prop :=
      true -- This dummy definition can be refined to accurate card representation 

noncomputable def is_ace (n : ℕ) : Prop :=
      true -- This dummy definition can be refined to accurate card representation 

theorem probability_first_heart_second_ace : 
  ∀ (deck : finset ℕ), (deck.card = 51) → 
  (Prob (λ c d : ℕ, is_heart c ∧ is_ace d) deck) = 1 / 98 :=
begin
  sorry
end

end probability_first_heart_second_ace_l400_400842


namespace max_mn_square_proof_l400_400109

noncomputable def max_mn_square (m n : ℕ) : ℕ :=
m^2 + n^2

theorem max_mn_square_proof (m n : ℕ) (h1 : 1 ≤ m ∧ m ≤ 2005) (h2 : 1 ≤ n ∧ n ≤ 2005) (h3 : (n^2 + 2 * m * n - 2 * m^2)^2 = 1) : 
max_mn_square m n ≤ 702036 :=
sorry

end max_mn_square_proof_l400_400109


namespace primes_between_50_and_60_l400_400606

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400606


namespace count_primes_50_60_l400_400480

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400480


namespace ratio_of_side_lengths_l400_400823

theorem ratio_of_side_lengths (a b c : ℕ) (h : a * a * b * b = 18 * c * c * 50 * c * c) :
  (12 = 1800000) ->  (15 = 1500) -> (10 > 0):=
by
  sorry

end ratio_of_side_lengths_l400_400823


namespace primes_between_50_and_60_l400_400527

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400527


namespace inequality_x_y_l400_400727

theorem inequality_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : x + y ≥ 2 := 
  sorry

end inequality_x_y_l400_400727


namespace primes_between_50_and_60_l400_400325

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400325


namespace imo_2001_sum_largest_l400_400719

noncomputable def I : ℕ := 667
noncomputable def M : ℕ := 3
noncomputable def O : ℕ := 1

theorem imo_2001_sum_largest : I * M * O = 2001 ∧ distinct [I, M, O] → I + M + O = 671 := 
by
  sorry

end imo_2001_sum_largest_l400_400719


namespace system_solution_l400_400791

theorem system_solution (x y : ℝ) : 
  (3^y * 81 = 9^(x^2)) ∧ (Real.log10 y = Real.log10 x - Real.log10 0.5) → 
  x = 2 ∧ y = 4 :=
by
  sorry

end system_solution_l400_400791


namespace primes_between_50_and_60_l400_400328

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400328


namespace bob_needs_to_improve_l400_400091

def bobs_distance_km : ℝ := 5
def bobs_time_minutes : ℝ := 26.5
def conversion_factor : ℝ := 0.621371
def sisters_distance_miles : ℝ := 3
def sisters_time_minutes : ℝ := 23.75

def bobs_distance_miles : ℝ := bobs_distance_km * conversion_factor
def bobs_pace : ℝ := bobs_time_minutes / bobs_distance_miles
def sisters_pace : ℝ := sisters_time_minutes / sisters_distance_miles
def improvement_needed : ℝ := (bobs_pace - sisters_pace) / bobs_pace * 100

theorem bob_needs_to_improve : abs (improvement_needed - 7.16) < 0.01 := sorry

end bob_needs_to_improve_l400_400091


namespace transform_polynomial_l400_400981

variables {x y : ℝ}

theorem transform_polynomial (h : y = x - 1 / x) :
  (x^6 + x^5 - 5 * x^4 + 2 * x^3 - 5 * x^2 + x + 1 = 0) ↔ (x^2 * (y^2 + y - 3) = 0) :=
sorry

end transform_polynomial_l400_400981


namespace inscribed_circle_has_correct_equation_PA_dot_PB_in_range_l400_400718

variables {real : Type*} [linear_ordered_field real]

-- Conditions
def condition1 (x y : real) : Prop := x - (sqrt 3) * y + 4 ≥ 0
def condition2 (x y : real) : Prop := x + (sqrt 3) * y + 4 ≥ 0
def condition3 (x : real) : Prop := x ≤ 2

-- Definitions for the main problem
def inscribed_circle_in_region (x y : real) : Prop :=
condition1 x y ∧ condition2 x y ∧ condition3 x → x^2 + y^2 = 4

-- Definitions for the geometric sequence problem
def inside_circle (x y : real) (r : real) : Prop := x^2 + y^2 < r^2
def PA_dot_PB_range (x y : real) : Prop :=
inside_circle x y 2 → let PA := sqrt (x + 2)^2 + y^2; PB := sqrt (x - 2)^2 + y^2; PM := sqrt x^2 + y^2 in
PA, PM, and PB form geometric sequence → -2 ≤ x - y - y^2 - 4 < 0

-- Theorems to prove
theorem inscribed_circle_has_correct_equation (x y : real) :
inscribed_circle_in_region x y := sorry

theorem PA_dot_PB_in_range (x y : real) :
PA_dot_PB_range x y := sorry

end inscribed_circle_has_correct_equation_PA_dot_PB_in_range_l400_400718


namespace average_cost_per_pen_l400_400891

theorem average_cost_per_pen 
(company_pens: ℕ) 
(price: ℕ) 
(shipping_cost: ℕ) 
(discount_rate : ℝ):
company_pens = 150 ∧ price = 1500 ∧ shipping_cost = 550 ∧ discount_rate = 0.10 
→ (price + shipping_cost) * (1 - discount_rate) / company_pens = 12 :=
by
  intros h
  cases h with h1 h23
  cases h23 with h2 h34
  cases h34 with h3 h4
  sorry

end average_cost_per_pen_l400_400891


namespace prime_count_between_50_and_60_l400_400582

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400582


namespace prime_count_between_50_and_60_l400_400573

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400573


namespace prime_count_50_to_60_l400_400495

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400495


namespace prime_count_between_50_and_60_l400_400344

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400344


namespace num_primes_between_50_and_60_l400_400623

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400623


namespace Barkley_bones_l400_400920

def bones_per_month : ℕ := 10
def months : ℕ := 5
def bones_received : ℕ := bones_per_month * months
def bones_buried : ℕ := 42
def bones_available : ℕ := 8

theorem Barkley_bones :
  bones_received - bones_buried = bones_available := by sorry

end Barkley_bones_l400_400920


namespace primes_between_50_and_60_l400_400587

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400587


namespace sin_cos_sum_neg_one_l400_400157

theorem sin_cos_sum_neg_one (x : ℝ) (h : Math.sin x + Math.cos x = -1) :
  (Math.sin x) ^ 2005 + (Math.cos x) ^ 2005 = -1 :=
sorry

end sin_cos_sum_neg_one_l400_400157


namespace number_of_primes_between_50_and_60_l400_400369

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400369


namespace num_primes_between_50_and_60_l400_400622

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400622


namespace peter_pizza_fraction_l400_400903

def pizza_slices : ℕ := 16
def peter_initial_slices : ℕ := 2
def shared_slices : ℕ := 2
def shared_with_paul : ℕ := shared_slices / 2
def total_slices_peter_ate := peter_initial_slices + shared_with_paul
def fraction_peter_ate : ℚ := total_slices_peter_ate / pizza_slices

theorem peter_pizza_fraction :
  fraction_peter_ate = 3 / 16 :=
by
  -- Leave space for the proof, which is not required.
  sorry

end peter_pizza_fraction_l400_400903


namespace expected_value_correct_prob_abs_diff_ge_1_correct_l400_400058

/-- Probability distribution for a single die roll -/
def prob_score (n : ℕ) : ℚ :=
  if n = 1 then 1/2 else if n = 2 then 1/3 else if n = 3 then 1/6 else 0

/-- Expected value based on the given probability distribution -/
def expected_value : ℚ := 
  (1 * prob_score 1) + (2 * prob_score 2) + (3 * prob_score 3)

/-- Proving the expected value calculation -/
theorem expected_value_correct : expected_value = 7/6 :=
  by sorry

/-- Calculate the probability of score difference being at least 1 between two players -/
def prob_abs_diff_ge_1 (x y : ℕ) : ℚ :=
  -- Implementation would involve detailed probability combinations that result in diff >= 1
  sorry

/-- Prove the probability of |x - y| being at least 1 -/
theorem prob_abs_diff_ge_1_correct : 
  ∀ (x y : ℕ), prob_abs_diff_ge_1 x y < 1 :=
  by sorry

end expected_value_correct_prob_abs_diff_ge_1_correct_l400_400058


namespace prime_count_between_50_and_60_l400_400562

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400562


namespace arithmetic_and_geometric_mean_l400_400766

theorem arithmetic_and_geometric_mean (a b : ℝ) (h1 : a + b = 40) (h2 : a * b = 100) : a^2 + b^2 = 1400 := by
  sorry

end arithmetic_and_geometric_mean_l400_400766


namespace number_of_primes_between_50_and_60_l400_400360

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400360


namespace primes_between_50_and_60_l400_400323

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400323


namespace prime_count_between_50_and_60_l400_400351

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400351


namespace y_exceeds_x_by_35_percent_l400_400869

theorem y_exceeds_x_by_35_percent {x y : ℝ} (h : x = 0.65 * y) : ((y - x) / x) * 100 = 35 :=
by
  sorry

end y_exceeds_x_by_35_percent_l400_400869


namespace count_primes_50_60_l400_400478

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400478


namespace appropriate_selection_method_l400_400835

theorem appropriate_selection_method 
  (A : Prop := "Selecting students from one class of the school is not representative")
  (B : Prop := "Randomly selecting 50 students from various grades in the school is more representative")
  (C : Prop := "Selecting 50 male students from the school is not representative")
  (D : Prop := "Selecting 50 female students from the school is not representative") 
  (most_representative : B) : B := 
by 
  sorry

end appropriate_selection_method_l400_400835


namespace prime_count_50_to_60_l400_400502

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400502


namespace primes_count_between_50_and_60_l400_400544

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400544


namespace tan_15_degrees_theta_range_valid_max_f_value_l400_400723

-- Define the dot product condition
def dot_product_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  AB * BC * (Real.cos θ) = 6

-- Define the sine inequality condition
def sine_inequality_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  6 * (2 - Real.sqrt 3) ≤ AB * BC * (Real.sin θ) ∧ AB * BC * (Real.sin θ) ≤ 6 * Real.sqrt 3

-- Define the maximum value function
noncomputable def f (θ : ℝ) : ℝ :=
  (1 - Real.sqrt 2 * Real.cos (2 * θ - Real.pi / 4)) / (Real.sin θ)

-- Proof that tan 15 degrees is equal to 2 - sqrt(3)
theorem tan_15_degrees : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 := 
  by sorry

-- Proof for the range of θ
theorem theta_range_valid (AB BC : ℝ) (θ : ℝ) 
  (h1 : dot_product_condition AB BC θ)
  (h2 : sine_inequality_condition AB BC θ) : 
  (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3) := 
  by sorry

-- Proof for the maximum value of the function
theorem max_f_value (θ : ℝ) 
  (h : (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3)) : 
  f θ ≤ Real.sqrt 3 - 1 := 
  by sorry

end tan_15_degrees_theta_range_valid_max_f_value_l400_400723


namespace truncated_cone_volume_partition_l400_400812

-- Define the given constants and conditions of the problem
variables (h : ℝ) (R r : ℝ)
def volume_truncated_cone (h : ℝ) (R r : ℝ) : ℝ := (1 / 3) * π * h * (R^2 + R * r + r^2)

-- Prove that the cone volume equals 7π under the given height and radii
theorem truncated_cone_volume_partition :
  let h := 3
  let R := 2
  let r := 1
  let total_volume := volume_truncated_cone h R r
  in total_volume = 7 * π ∧
     let v1 := (2 / 12) * total_volume
     let v2 := (3 / 12) * total_volume
     let v3 := (7 / 12) * total_volume
     ∃ (h1 h2 : ℝ) (R1 R2 : ℝ),
       volume_truncated_cone h1 R R1 = v1 ∧
       volume_truncated_cone h2 R1 r = v2 ∧
       volume_truncated_cone (h - h1 - h2) R2 r = v3 :=
by
  sorry

end truncated_cone_volume_partition_l400_400812


namespace rhombus_area_l400_400841

def is_rhombus_with_45_deg_angle (a : ℝ) (θ : ℝ) : Prop :=
  θ = π / 4 ∧ ∃ (s : ℝ), s = 3

def area_rhombus (s : ℝ) : ℝ :=
  let base := s * √2
  let height := s / √2
  base * height

theorem rhombus_area (s : ℝ) (θ : ℝ) (h : is_rhombus_with_45_deg_angle s θ) : area_rhombus s = 9 :=
by
  -- Placeholder for actual proof
  sorry

end rhombus_area_l400_400841


namespace find_constants_l400_400952

variable (x : ℝ)

theorem find_constants 
  (h : ∀ x, (6 * x^2 + 3 * x) / ((x - 4) * (x - 2)^3) = 
  (13.5 / (x - 4)) + (-27 / (x - 2)) + (-15 / (x - 2)^3)) :
  true :=
by {
  sorry
}

end find_constants_l400_400952


namespace primes_between_50_and_60_l400_400232

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400232


namespace number_of_primes_between_50_and_60_l400_400357

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400357


namespace sufficient_but_not_necessary_condition_l400_400996

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (|a - b^2| + |b - a^2| ≤ 1) → ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ 
  ∃ (a b : ℝ), ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ ¬ (|a - b^2| + |b - a^2| ≤ 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l400_400996


namespace primes_between_50_and_60_l400_400392

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400392


namespace prime_count_between_50_and_60_l400_400576

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400576


namespace count_primes_between_50_and_60_l400_400464

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400464


namespace primes_between_50_and_60_l400_400303

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400303


namespace expression_value_l400_400929

noncomputable def expr1 := (3 / 2)^(-1 / 3)
noncomputable def expr2 := (-4 / 5)^0
noncomputable def expr3 := 8^(1 / 4) * 42
noncomputable def expr4 := (32 * real.sqrt 3)^6
noncomputable def expr5 := real.sqrt ((-2 / 3)^(2 / 3))

-- Given expression
noncomputable def given_expr := expr1 * expr2 + expr3 + expr4 - expr5

-- The theorem we need to prove
theorem expression_value : given_expr = 110 := by
  -- This is where the proof would go, but we insert 'sorry' for now.
  sorry

end expression_value_l400_400929


namespace circles_tangent_condition_l400_400100

theorem circles_tangent_condition (C1 C2 : ℝ → ℝ → Prop)
    (h1 : C1 10 8) (h2 : C2 10 8) (r1 r2 : ℝ)
    (h3 : r1 * r2 = 75)
    (m : ℝ) (h4 : m > 0)
    (h5 : ∀ x y, C1 x y → y = 2 * m * x)
    (h6 : ∀ x y, C2 x y → y = 2 * m * x)
    (h7 : ∀ x ≥ 0, C1 x 0 ↔ x ≥ 0)
    (h8 : ∀ x ≥ 0, C2 x 0 ↔ x ≥ 0)
    (a b c : ℕ)
    (h9 : a = 2) (h10 : b = 109) (h11 : c = 41)
    (h12 : b % (p ^ 2) ≠ 0 ∀ prime p)
    (h13 : nat.coprime a c) :
  a + b + c = 152 := sorry

end circles_tangent_condition_l400_400100


namespace primes_between_50_and_60_l400_400604

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400604


namespace eccentricity_of_ellipse_l400_400178

theorem eccentricity_of_ellipse (a b : ℝ) (h : a > b > 0) :
  ∃ c : ℝ, let e := c / a in 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
  let F := (c, 0) in
  let P := (0, -b) in
  ∀ m n : ℝ, x = m ∧ y = n → 
    let FP := (-c, -b) in
    let MF := (c-m, -n) in
    FP = (2 * c, 2 * -n) →
    e = √3 / 3) := sorry

end eccentricity_of_ellipse_l400_400178


namespace number_of_primes_between_50_and_60_l400_400361

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400361


namespace prime_count_between_50_and_60_l400_400343

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400343


namespace primes_between_50_and_60_l400_400645

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400645


namespace solve_quadratic_l400_400826

theorem solve_quadratic (x : ℝ) : (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3) :=
by
  sorry

end solve_quadratic_l400_400826


namespace find_coefficients_l400_400978

def A (x : ℝ) := x^3 + 3*x^2 + 2*x > 0
def B (x : ℝ) (a b : ℝ) := x^2 + a*x + b ≤ 0

theorem find_coefficients : 
  (A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2}) ∧ 
  (A ∪ B = {x : ℝ | x > -2}) → 
  (∃ a b : ℝ, a = -1 ∧ b = -2) :=
by
  intros h
  exists -1, -2
  split
  · refl
  · refl
  sorry

end find_coefficients_l400_400978


namespace primes_between_50_and_60_l400_400224

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400224


namespace count_primes_50_60_l400_400482

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400482


namespace find_I_l400_400720

-- Define each letter as a separate digit.
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

-- Variables representing the digits for each letter
variables (F I V E G H T : ℕ)

-- Given conditions as hypotheses
axiom digits_distinct : ∀ x y ∈ {F, I, V, E, G, H, T}, x ≠ y → x ∈ (range 10) → y ∈ (range 10)
axiom F_val : F = 8
axiom V_odd : V % 2 = 1

-- The main statement to prove
theorem find_I : I = 2 :=
by
  -- Proof steps would go here. We'll use sorry to indicate we skip proof steps
  sorry

end find_I_l400_400720


namespace num_primes_50_60_l400_400276

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400276


namespace primes_between_50_and_60_l400_400316

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400316


namespace count_primes_50_60_l400_400492

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400492


namespace repetend_of_7_over_29_l400_400125

theorem repetend_of_7_over_29 : ∃ r : ℕ, r = 241379 ∧ (∃ m : ℕ, ∃ n : ℕ, r = (7 * 10^n) % 29 ∧ (7 * 10 ^ m / 29).decimalRepeats 6) :=
sorry

end repetend_of_7_over_29_l400_400125


namespace four_digit_arithmetic_sequence_l400_400893

theorem four_digit_arithmetic_sequence :
  ∃ (a b c d : ℕ), 1000 * a + 100 * b + 10 * c + d = 5555 ∨ 1000 * a + 100 * b + 10 * c + d = 2468 ∧
  (a + d = 10) ∧ (b + c = 10) ∧ (2 * b = a + c) ∧ (c - b = b - a) ∧ (d - c = c - b) ∧
  (1000 * d + 100 * c + 10 * b + a + 1000 * a + 100 * b + 10 * c + d = 11110) :=
sorry

end four_digit_arithmetic_sequence_l400_400893


namespace num_primes_50_60_l400_400280

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400280


namespace num_primes_50_60_l400_400273

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400273


namespace overlapping_area_of_triangles_l400_400071

theorem overlapping_area_of_triangles
  (hexagon : Type)
  (A B C D E F : hexagon)
  (hexagon_area : Real)
  (h_reg_hex : is_regular_hexagon A B C D E F)
  (h_area : hexagon_area = 36) :
  ∃ overlap_area : Real, overlap_area = 9 := sorry

end overlapping_area_of_triangles_l400_400071


namespace primes_count_between_50_and_60_l400_400540

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400540


namespace count_primes_between_50_and_60_l400_400460

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400460


namespace number_of_primes_between_50_and_60_l400_400376

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400376


namespace number_of_primes_between_50_and_60_l400_400368

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400368


namespace neg_p_equiv_l400_400204

theorem neg_p_equiv :
  (¬ (∀ x : ℝ, x > 0 → x - Real.log x > 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0 - Real.log x_0 ≤ 0) :=
by
  sorry

end neg_p_equiv_l400_400204


namespace incorrect_relations_count_l400_400807

theorem incorrect_relations_count :
  let r1 := 0 ∈ 0,
      r2 := 0 ⊇ ∅,
      r3 := 0.3 ∉ ℚ,
      r4 := 0 ∈ ℕ,
      r5 := {a, b} ⊆ {b, a},
      r6 := {x | x^2 - 2 = 0 ∧ x ∈ ℤ} = ∅ in
  (¬r1 ∧ ¬r2 ∧ ¬r3 ∧ r4 ∧ r5 ∧ r6) → (card {i | ¬[r1, r2, r3, r4, r5, r6].i} = 3)
:= sorry

end incorrect_relations_count_l400_400807


namespace quadratic_two_distinct_real_roots_l400_400768

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  let a := 1
  let b := k - 3
  let c := 1 - k
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  have a := 1
  have b := k - 3
  have c := 1 - k
  let Δ := (k - 3)^2 - 4 * 1 * (1 - k)
  have Δ_simplified : Δ = (k - 1)^2 + 4 := by 
    calc Δ = (k - 3)^2 - 4 * 1 * (1 - k) : by rfl
         ... = k^2 - 6*k + 9 + 4 - 4*k : by ring
         ... = k^2 - 10*k + 13 : by ring
         ... = (k - 1)^2 + 4 : by ring
  show Δ > 0 from by
    calc Δ = (k - 1)^2 + 4 : Δ_simplified
         ... > 0 : by linarith [pow_two_nonneg (k - 1)]

end quadratic_two_distinct_real_roots_l400_400768


namespace min_y_in_quadratic_l400_400851

theorem min_y_in_quadratic (x : ℝ) : ∃ y : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ y', (y' = x^2 + 16 * x + 20) → y ≤ y' := 
sorry

end min_y_in_quadratic_l400_400851


namespace no_primes_in_factorial_range_l400_400142

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) : 
  ∀ p, prime p → ¬ (n! < p ∧ p ≤ n! + (n + 1)) :=
by sorry

end no_primes_in_factorial_range_l400_400142


namespace count_primes_between_50_and_60_l400_400667

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400667


namespace cylindrical_granary_circumference_l400_400705

def height_zhang := 1
def height_chi := 3
def height_total_chi := height_zhang * 10 + height_chi -- total height in chi

def volume_hu := 1950
def cubic_chi_per_hu := 1.62
def volume_cubic_chi := volume_hu * cubic_chi_per_hu -- volume in cubic chi

def pi_approx := 3

theorem cylindrical_granary_circumference :
  ∃ r : ℝ, pi_approx * r^2 * height_total_chi = volume_cubic_chi ∧ 2 * pi_approx * r = 54 :=
begin
  let r := 9,
  use r,
  split,
  { -- Proof part π * r^2 * 13 = 1950 * 1.62
    calc
      pi_approx * 9^2 * 13 = 3 * 81 * 13 : by norm_num
      ... = 3 * 1053 : by norm_num
      ... = 3159 : by norm_num
      ... = 1950 * 1.62 : by norm_num,
    sorry
  },
  { -- Proof part 2 * π * r = 54
    calc
      2 * pi_approx * 9 = 2 * 3 * 9 : by norm_num
      ... = 54 : by norm_num,
    sorry
  }
end

end cylindrical_granary_circumference_l400_400705


namespace number_of_selections_even_sum_is_66_l400_400772

-- Define the set of numbers
def set_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a function to count the number of ways to select four numbers with an even sum
noncomputable def count_even_sum_selections : ℕ :=
  (Finset.choose 5 4) + ((Finset.choose 5 2) * (Finset.choose 4 2)) + (Finset.choose 4 4)

-- State the theorem that needs to be proved
theorem number_of_selections_even_sum_is_66 : count_even_sum_selections = 66 := 
by
  sorry

end number_of_selections_even_sum_is_66_l400_400772


namespace primes_between_50_and_60_l400_400403

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400403


namespace primes_count_between_50_and_60_l400_400541

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400541


namespace danny_distance_to_work_l400_400937

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end danny_distance_to_work_l400_400937


namespace simplify_and_evaluate_l400_400778

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3)) - (x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  sorry

end simplify_and_evaluate_l400_400778


namespace num_primes_50_60_l400_400283

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400283


namespace range_condition_l400_400715

noncomputable def range_of_independent_variable (x : ℝ) (y : ℝ) : Prop :=
  y = real.sqrt (6 - 2 * x) -> x ≤ 3

theorem range_condition (x : ℝ) (y : ℝ) : y = real.sqrt (6 - 2 * x) → x ≤ 3 := by
  sorry

end range_condition_l400_400715


namespace least_number_subtracted_l400_400024

/--
  What least number must be subtracted from 9671 so that the remaining number is divisible by 5, 7, and 11?
-/
theorem least_number_subtracted
  (x : ℕ) :
  (9671 - x) % 5 = 0 ∧ (9671 - x) % 7 = 0 ∧ (9671 - x) % 11 = 0 ↔ x = 46 :=
sorry

end least_number_subtracted_l400_400024


namespace identical_parts_form_rectangle_l400_400107

theorem identical_parts_form_rectangle
  (shape : ℕ -> ℕ -> Prop) -- shape is defined by a proposition on the grid points.
  (identical : (ℕ -> ℕ -> Prop) -> (ℕ -> ℕ -> Prop) -> Prop) -- identical parts if they overlap after rotation/reflection.
  (part : ℕ -> ℕ -> Prop) -- each part is a subset of the grid points.
  (area : (ℕ -> ℕ -> Prop) -> ℕ) -- function to calculate the area of a part.
  (total_area : ℕ)
  (num_parts : ℕ) :
  total_area = 30 ∧ num_parts = 6 ∧ 
  area shape = total_area ∧ 
  (∀p, identical p part → area p = 5) ∧ 
  (∃ arrangements, arrangements part (5 * 6)) :=
by
  sorry

end identical_parts_form_rectangle_l400_400107


namespace primes_between_50_and_60_l400_400597

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400597


namespace primes_between_50_and_60_l400_400440

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400440


namespace congruent_triangles_l400_400739

def point_on_circle (S : set (ℝ × ℝ)) (A : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, (A.1 - r) ^ 2 + (A.2 - r) ^ 2 = 1  -- S is assumed to be a circle of radius 1 centered at (r, r)

def triangle_on_circle (S : set (ℝ × ℝ)) (A B C : ℝ × ℝ) : Prop :=
  point_on_circle S A ∧ point_on_circle S B ∧ point_on_circle S C

def parallel (A B C D : ℝ × ℝ) : Prop :=
  (A.2 - B.2) * (C.1 - D.1) = (A.1 - B.1) * (C.2 - D.2)

def next_triangle (S : set (ℝ × ℝ)) (A_r B_r C_r A_r1 B_r1 C_r1 : ℝ × ℝ) : Prop :=
  triangle_on_circle S A_r B_r C_r ∧ triangle_on_circle S A_r1 B_r1 C_r1 ∧
  parallel A_r1 A_r B_r C_r ∧ parallel B_r1 B_r C_r A_r ∧ parallel C_r1 C_r A_r B_r

def angles_not_multiples_of_45 (α β γ : ℤ) : Prop :=
  ¬ (45 ∣ α) ∧ ¬ (45 ∣ β) ∧ ¬ (45 ∣ γ)

def angles_of_triangle (A B C : ℝ × ℝ) : Type :=
  ℤ × ℤ × ℤ -- This is a placeholder. Proper type depends on further geometric definitions.

noncomputable def initialize_triangle (Δ₁ : Type) : Type :=
  Σ (A₁ B₁ C₁ : ℝ × ℝ), angles_of_triangle A₁ B₁ C₁

theorem congruent_triangles
  (S : set (ℝ × ℝ))
  (initial_triangle : Σ (A₁ B₁ C₁ : ℝ × ℝ), angles_of_triangle A₁ B₁ C₁)
  (h_conditions : ∃ (A₁ B₁ C₁ : ℝ × ℝ), ∃ (α₁ β₁ γ₁ : ℤ), 
    angles_not_multiples_of_45 α₁ β₁ γ₁ ∧ initial_triangle.2 = (α₁, β₁, γ₁) ∧ 
    next_triangle S A₁ B₁ C₁ A₁ B₁ C₁) :
  ∃ (r s : ℕ), 1 ≤ r ∧ r < s ∧ s ≤ 15 ∧
  ∃ (A_r B_r C_r A_s B_s C_s : ℝ × ℝ), Σ (α_r β_r γ_r α_s β_s γ_s : ℤ), 
    next_triangle S A_r B_r C_r A_s B_s C_s ∧ 
    initial_triangle.2 = (α_r, β_r, γ_r) ∧ 
    (α_r, β_r, γ_r) = (α_s, β_s, γ_s) := 
sorry

end congruent_triangles_l400_400739


namespace minimum_value_inequality_l400_400738

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 1) * (y^2 + 5 * y + 1) * (z^2 + 5 * y + 1) / (x * y * z) ≥ 343 :=
by sorry

end minimum_value_inequality_l400_400738


namespace min_value_y_l400_400849

theorem min_value_y : ∃ x : ℝ, (∀ y : ℝ, y = x^2 + 16 * x + 20 → y ≥ -44) :=
begin
  use -8,
  intro y,
  intro hy,
  suffices : y = (x + 8)^2 - 44,
  { rw this,
    exact sub_nonneg_of_le (sq_nonneg (x + 8)) },
  sorry
end

end min_value_y_l400_400849


namespace number_of_sequences_l400_400110

theorem number_of_sequences (n m : ℕ) (h : n ≥ 2 * m) :
  ∃ k, k = nat.choose (n - m) m :=
sorry

end number_of_sequences_l400_400110


namespace functional_square_for_all_n_l400_400744

theorem functional_square_for_all_n (f : ℕ → ℕ) :
  (∀ m n : ℕ, ∃ k : ℕ, (f m + n) * (m + f n) = k ^ 2) ↔ ∃ c : ℕ, ∀ n : ℕ, f n = n + c := 
sorry

end functional_square_for_all_n_l400_400744


namespace smallest_positive_angle_l400_400139

theorem smallest_positive_angle (x : ℝ) (hx : 0 < x) (h : sin (4 * x) * sin (6 * x) = cos (4 * x) * cos (6 * x)) :
  x = 9 := sorry

end smallest_positive_angle_l400_400139


namespace tan_ϕ_eq_neg_sqrt3_l400_400156

   variable {ϕ : ℝ}

   theorem tan_ϕ_eq_neg_sqrt3 
     (h1 : (Real.cos (π / 2 + ϕ)) = (sqrt 3 / 2))
     (h2 : abs ϕ < (π / 2)) : 
     Real.tan ϕ = -sqrt 3 := 
   sorry
   
end tan_ϕ_eq_neg_sqrt3_l400_400156


namespace probability_of_sum_9_is_two_ninths_l400_400009

-- Definitions based on the given problem conditions
def numbers_dice : list ℕ := [1, 2, 3, 4, 5, 6]

def pairs_sum_9 : list (ℕ × ℕ) :=
  [(3, 6), (4, 5), (5, 4), (6, 3)]

-- The total number of possible outcomes when rolling two dice
def total_outcomes := (numbers_dice.length) * (numbers_dice.length)

-- The total number of favorable outcomes for pairs summing to 9
def favorable_outcomes := pairs_sum_9.length

-- Define the probability as a fraction
def probability_sum_9 := (favorable_outcomes : ℚ) / total_outcomes

-- The main theorem to state and prove
theorem probability_of_sum_9_is_two_ninths :
  probability_sum_9 = (2 / 9 : ℚ) :=
by
  sorry

end probability_of_sum_9_is_two_ninths_l400_400009


namespace primes_between_50_and_60_l400_400404

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400404


namespace sum_x_le_n_div_3_l400_400742

variables {n : ℕ} (x : ℕ → ℝ)

noncomputable theory

theorem sum_x_le_n_div_3
  (hn : n ≥ 3)
  (hx_bounds : ∀ i, i < n → x i ∈ Set.Icc (-1 : ℝ) 1)
  (hx_cube_sum_zero : ∑ i in Finset.range n, (x i)^3 = 0)
  : (∑ i in Finset.range n, x i) ≤ (n / 3 : ℝ) := 
sorry

end sum_x_le_n_div_3_l400_400742


namespace roger_individual_pouches_per_pack_l400_400769

variable (members : ℕ) (coaches : ℕ) (helpers : ℕ) (packs : ℕ)

-- Given conditions
def total_people (members coaches helpers : ℕ) : ℕ := members + coaches + helpers
def pouches_per_pack (total_people packs : ℕ) : ℕ := total_people / packs

-- Specific values from the problem
def roger_total_people : ℕ := total_people 13 3 2
def roger_packs : ℕ := 3

-- The problem statement to prove:
theorem roger_individual_pouches_per_pack : pouches_per_pack roger_total_people roger_packs = 6 :=
by
  sorry

end roger_individual_pouches_per_pack_l400_400769


namespace primes_between_50_and_60_l400_400322

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400322


namespace primes_between_50_and_60_l400_400287

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400287


namespace correct_transformation_l400_400863

namespace TransformationProof

structure Point where
  x : ℝ
  y : ℝ

def scale (p : Point) (factor : ℝ) : Point := 
  { x := p.x * factor, y := p.y * factor }

def reflect_x (p : Point) : Point := 
  { x := p.x, y := -p.y }

def transformation (p : Point) : Point := 
  reflect_x (scale p 3)

def A := { x := 1, y := 2 }
def A' := { x := 3, y := -6 }
def B := { x := 2, y := 3 }
def B' := { x := 6, y := -9 }

theorem correct_transformation :
  transformation A = A' ∧ transformation B = B' :=
by
  sorry

end TransformationProof

end correct_transformation_l400_400863


namespace primes_between_50_and_60_l400_400306

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400306


namespace primes_between_50_and_60_l400_400423

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400423


namespace primes_between_50_and_60_l400_400222

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400222


namespace isosceles_trapezoid_area_l400_400686

theorem isosceles_trapezoid_area :
  ∀ (bottom_angle : ℝ) (leg_length : ℝ) (top_base_length : ℝ),
  bottom_angle = 60 ∧ leg_length = 1 ∧ top_base_length = 1 →
  area_original_figure bottom_angle leg_length top_base_length = (3 * real.sqrt 6) / 2 :=
by
  sorry

end isosceles_trapezoid_area_l400_400686


namespace count_primes_between_50_and_60_l400_400259

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400259


namespace sum_intersection_x_coords_l400_400986

theorem sum_intersection_x_coords :
  let segments := [
    ((-4 : ℝ), (-5 : ℝ), (-2 : ℝ), (0 : ℝ)),
    ((-2 : ℝ), (0 : ℝ), (-1 : ℝ), (-1 : ℝ)),
    ((-1 : ℝ), (-1 : ℝ), (1 : ℝ), (3 : ℝ)),
    ((1 : ℝ), (3 : ℝ), (2 : ℝ), (2 : ℝ)),
    ((2 : ℝ), (2 : ℝ), (4 : ℝ), (6 : ℝ))
  ] in
  let y_intersect := 1.5 in
  let slopes_and_intercepts := segments.map (λ ⟨x1, y1, x2, y2⟩,
    let slope := (y2 - y1) / (x2 - x1) in
    let intercept := y1 - slope * x1 in
    (slope, intercept)
  ) in
  let intersections := slopes_and_intercepts.filterMap (λ ⟨m, b⟩,
    if m * b = 0 then none else -- i.e., m ≠ 0
    let x := (y_intersect - b) / m in
    if x ∈ set.Icc (-4.0) (4.0) then some x else none
  ) in
  (2.6 : ℝ) = intersections.sum :=
begin
  -- Skipping the proof steps, inserting sorry
  sorry
end

end sum_intersection_x_coords_l400_400986


namespace prime_count_50_to_60_l400_400501

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400501


namespace count_primes_between_50_and_60_l400_400457

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400457


namespace count_primes_50_60_l400_400474

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400474


namespace equivalent_form_of_f_l400_400732

def f (x : ℝ) : ℝ :=
  (Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2)) - (Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2))

theorem equivalent_form_of_f (x : ℝ) : f x = Real.cos (2 * x) :=
  sorry

end equivalent_form_of_f_l400_400732


namespace primes_count_between_50_and_60_l400_400548

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400548


namespace number_of_primes_between_50_and_60_l400_400372

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400372


namespace sum_first_3m_l400_400830

variable {α : Type*}
variable [LinearOrderedField α]

-- Definitions as per conditions
def S (n : ℕ) : α := ∑ i in range (n + 1), (a i) -- To represent the sum of the first n terms of the sequence

-- Problem conditions
variable (a : ℕ → α)
variable (m : ℕ)
variable (h1 : S a m = 30)
variable (h2 : S a (2 * m) = 100)

theorem sum_first_3m (m : ℕ) (a : ℕ → α) (h1 : S a m = 30) (h2 : S a (2 * m) = 100) : S a (3 * m) = 210 :=
sorry

end sum_first_3m_l400_400830


namespace emily_caught_4_trout_l400_400114

def number_of_trout (T : ℕ) : Prop :=
  let weight_of_trout := 2 * T in
  let weight_of_catfish := 3 * 1.5 in
  let weight_of_bluegills := 5 * 2.5 in
  let total_weight := weight_of_trout + weight_of_catfish + weight_of_bluegills in
  total_weight = 25

theorem emily_caught_4_trout : ∃ T : ℕ, number_of_trout T ∧ T = 4 :=
by {
  sorry
}

end emily_caught_4_trout_l400_400114


namespace product_mb_l400_400103

noncomputable def line_eq (m b : ℝ) : ℝ → ℝ :=
  λ x, m * x + b

theorem product_mb {m b : ℝ} :
  (line_eq m b 0 = -2) →
  (line_eq m b 2 = 4) →
  (m * b) = -6 :=
by
  intro h1 h2
  sorry

end product_mb_l400_400103


namespace same_function_representation_l400_400030

theorem same_function_representation : 
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = x^2 - 2*x - 1) ∧ (∀ m, g m = m^2 - 2*m - 1) →
    (f = g) :=
by
  sorry

end same_function_representation_l400_400030


namespace quadrilateral_area_proof_l400_400907

noncomputable def quadrilateral_area_statement : Prop :=
  ∀ (a b : ℤ), a > b ∧ b > 0 ∧ 8 * (a - b) * (a - b) = 32 → a + b = 4

theorem quadrilateral_area_proof : quadrilateral_area_statement :=
sorry

end quadrilateral_area_proof_l400_400907


namespace path_area_and_cost_l400_400867

-- Definitions based on conditions
def length_field : ℝ := 60
def width_field : ℝ := 55
def width_path : ℝ := 2.5
def cost_per_sqm : ℝ := 2

-- Theorem statement
theorem path_area_and_cost :
  let length_total := length_field + 2 * width_path,
      width_total := width_field + 2 * width_path,
      area_total := length_total * width_total,
      area_field := length_field * width_field,
      area_path := area_total - area_field,
      cost_path := area_path * cost_per_sqm
  in area_path = 600 ∧ cost_path = 1200 := by sorry

end path_area_and_cost_l400_400867


namespace honor_students_in_class_l400_400000

theorem honor_students_in_class : 
  (G B G_excellent B_excellent : ℕ) 
  (h1 : G < 30) 
  (h2 : B < 30) 
  (h3 : G + B < 30)
  (h4 : G_excellent = 3 * G / 13) 
  (h5 : B_excellent = 4 * B / 11)
  (h6 : G + B = 24) : 
  G_excellent + B_excellent = 7 := by
  sorry

end honor_students_in_class_l400_400000


namespace xc_divides_de_l400_400890

noncomputable theory

open_locale classical

-- Definitions based on conditions
variables {A B C D E X : Type}

-- Assuming given conditions
axiom h1 : circle_inscribed_in_angle A B C
axiom h2 : line_through A intersects_circle_at D E
axiom h3 : chord_parallel B X D E

-- Lean theorem statement
theorem xc_divides_de (A B C D E X : Type)
                      [h1 : circle_inscribed_in_angle A B C]
                      [h2 : line_through A intersects_circle_at D E]
                      [h3 : chord_parallel B X D E] :
  divides_in_ratio XC DE 1 1 :=
sorry

end xc_divides_de_l400_400890


namespace count_primes_between_50_and_60_l400_400668

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400668


namespace solve_fraction_inequality_l400_400780

theorem solve_fraction_inequality :
  { x : ℝ | x / (x + 5) ≥ 0 } = { x : ℝ | x < -5 } ∪ { x : ℝ | x ≥ 0 } := by
  sorry

end solve_fraction_inequality_l400_400780


namespace minimum_area_l400_400145

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

noncomputable def T (α : ℝ) : ℝ :=
  arctan α + arctan (1 / α) - (α / (1 + α^2))

theorem minimum_area :
  ∃ (α : ℝ), α > 0 ∧ T α = (π / 2 - 1 / 2) :=
begin
  -- proof omitted
  sorry
end

end minimum_area_l400_400145


namespace maximum_value_l400_400737

noncomputable def M_value (x y z v w : ℝ) : ℝ :=
  x * z + 3 * y * z + 2 * z * v + 7 * z * w

theorem maximum_value (x y z v w x_M y_M z_M v_M w_M : ℝ)
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : v > 0) (h5 : w > 0)
  (h6 : x^2 + y^2 + z^2 + v^2 + w^2 = 2023)
  (hM : ∀ (x' y' z' v' w' : ℝ), M_value x y z v w ≤ M_value x' y' z' v' w') :
  M_value x_M y_M z_M v_M w_M + x_M + y_M + z_M + v_M + w_M = 
  sqrt(1011.5) + 3136.5 * sqrt(7) :=
sorry

end maximum_value_l400_400737


namespace primes_between_50_and_60_l400_400394

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400394


namespace ellipse_problems_l400_400174

noncomputable def ellipse_equation : Prop :=
  let a := sqrt 2 in
  let b := 1 in
  let c := 1 in
  (a^2 = 2) ∧ (b^2 = 1) ∧ (c = 1) ∧ (a = sqrt 2 * b) → 
  (∀ {x y : ℝ}, (x^2) / 2 + y^2 = 1 → true)

noncomputable def min_lambda (P : ℝ × ℝ) (F : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  let P := (2, 0) in
  let F := (if 2 > 0 then -1 else 1, 0) in
  let f (A B: (ℝ × ℝ)) := ((A.1 - 2) * (B.1 - 2) + A.2 * B.2) in
  (∀ l, (∃ A B, line passes through F and intersects ellipse C at A and B) → 
  (f A B ≤ 17 / 2)) ∧ ∃ λ ∈ ℝ, λ = 17 / 2

-- Main theorem combining both parts
theorem ellipse_problems : Prop :=
  ellipse_equation ∧ min_lambda (2, 0) (-1, 0) (0, 0) (0, 0)

end ellipse_problems_l400_400174


namespace count_primes_50_60_l400_400488

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400488


namespace primes_between_50_and_60_l400_400593

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400593


namespace positions_of_M_path_of_N_l400_400906

-- Variables and conditions
variables (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (k : Set A) [IsoscelesRightTriangle A B C] [Circumcircle k A B C]
variables (NB MB NC MA BC : ℝ)
-- Conditions
variables (condition1 : AC = BC) (condition2 : Center k = Midpoint (A, B)) (condition3 : N ∈ Plane A B C)
variables (condition4 : NB = MB) (condition5 : NC = MA)

-- Propositions
theorem positions_of_M (M : A) : 
  (M ∈ circumcircle k) → ∀ (MA MB BC : ℝ), 
  (triangle_ineq MA MB BC) → 
  positions_of_M_segments M :=
sorry

theorem path_of_N (N : A) : 
  (N ∈ Plane A B C) → 
  (NB = MB) →
  (NC = MA) →
  path_of_N_circle_center_midpoint_radius N (Midpoint B C) (sqrt 3 / 2 * BC) :=
sorry

end positions_of_M_path_of_N_l400_400906


namespace prime_count_50_to_60_l400_400514

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400514


namespace number_of_primes_between_50_and_60_l400_400355

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400355


namespace inequality_solution_set_l400_400824

theorem inequality_solution_set (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end inequality_solution_set_l400_400824


namespace part1_part2_l400_400180

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a * real.log x) / x + x
noncomputable def g (x : ℝ) : ℝ := 2 * x - 1 / x

theorem part1 (a : ℝ): 
    (∃! x : ℝ, 1 ≤ x ∧ x ≤ real.exp 1 ∧ f a x = g x) ↔ 
    a ∈ set.Iic 1 ∪ set.Ioi ((real.exp 2 - 1) / 2) := sorry

theorem part2 (a : ℝ): 
    (∃ t0 ∈ set.Icc 1 (real.exp 1), ∀ x ≥ t0, x * f a x > x^2 + 2 * (a + 1) / x + 2 * x) ↔ 
    a ∈ set.Iio (-2) ∪ set.Ioi ((real.exp 2 + 1) / (real.exp 1 - 1)) := sorry

end part1_part2_l400_400180


namespace minimize_PR_plus_RQ_l400_400994

open Real       -- To work with real numbers

-- Define point P
def P : (ℝ × ℝ) := (-2, -4)

-- Define point Q
def Q : (ℝ × ℝ) := (5, 3)

-- Function to find the y-coordinate of point R on the line PQ given x-coordinate 2
def y_of_R_on_PQ (m : ℝ) : Prop := 
  let slope := (Q.snd - P.snd) / (Q.fst - P.fst)
  let line_equation := λ x, slope * (x - P.fst) + P.snd
  line_equation 2 = m

-- Theorem to prove m = 0 minimizes PR + RQ
theorem minimize_PR_plus_RQ (m : ℝ) (R : (ℝ × ℝ)) (hR : R = (2, m)) : y_of_R_on_PQ m → m = 0 := by
  sorry

end minimize_PR_plus_RQ_l400_400994


namespace remainder_1534_base12_div_by_9_l400_400028

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 5 * 12^2 + 3 * 12 + 4

theorem remainder_1534_base12_div_by_9 :
  (base12_to_base10 1534) % 9 = 4 :=
by
  sorry

end remainder_1534_base12_div_by_9_l400_400028


namespace primes_between_50_and_60_l400_400531

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400531


namespace interior_angle_of_convex_hexagon_l400_400054

theorem interior_angle_of_convex_hexagon
  (A B C D E F : Type)
  (hexagon : ConvexHexagon A B C D E F)
  (AB_eq_BC : length AB = length BC)
  (BC_eq_CD : length BC = length CD)
  (CD_eq_DE : length CD = length DE)
  (DE_eq_EF : length DE = length EF)
  (EF_eq_FA : length EF = length FA)
  (angle_A : measure_angle A = 134)
  (angle_B : measure_angle B = 106)
  (angle_C : measure_angle C = 134) :
  measure_angle E = 134 := 
sorry

end interior_angle_of_convex_hexagon_l400_400054


namespace primes_between_50_and_60_l400_400413

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400413


namespace prime_count_50_to_60_l400_400515

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400515


namespace altitude_division_l400_400013

variables {A B C D E : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E]

theorem altitude_division 
  (AD DC CE EB y : ℝ)
  (hAD : AD = 6)
  (hDC : DC = 4)
  (hCE : CE = 3)
  (hEB : EB = y)
  (h_similarity : CE / DC = (AD + DC) / (y + CE)) : 
  y = 31 / 3 :=
by
  sorry

end altitude_division_l400_400013


namespace number_of_primes_between_50_and_60_l400_400375

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400375


namespace count_primes_50_60_l400_400486

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400486


namespace ticket_cost_exceeds_budget_l400_400149

theorem ticket_cost_exceeds_budget (cost_per_ticket : ℕ) (num_students : ℕ) (budget : ℕ) (h_cost : cost_per_ticket = 29) (h_students : num_students = 498) (h_budget : budget = 1500) :
  (cost_per_ticket * num_students > budget) :=
by {
  have h1 : cost_per_ticket * num_students = 29 * 498, {
    rw [h_cost, h_students]
  },
  have h2 : 29 * 498 = 14442 := by norm_num,
  rw [h1, h2],
  show 14442 > 1500,
  norm_num
}

end ticket_cost_exceeds_budget_l400_400149


namespace cos_double_angle_l400_400998

-- Let α be a real number such that cos α = 2√5 / 5
variable {α : ℝ}
variable (cos_alpha : cos α = (2 * real.sqrt 5) / 5)

-- Prove that cos 2α = 3 / 5
theorem cos_double_angle :
  cos 2 * α = 3 / 5 :=
sorry

end cos_double_angle_l400_400998


namespace count_primes_between_50_and_60_l400_400251

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400251


namespace num_primes_50_60_l400_400285

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400285


namespace fraction_meaningful_cond_l400_400836

theorem fraction_meaningful_cond (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) := 
by
  sorry

end fraction_meaningful_cond_l400_400836


namespace primes_between_50_and_60_l400_400596

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400596


namespace original_ratio_l400_400010

theorem original_ratio (x y : ℤ)
  (h1 : y = 48)
  (h2 : (x + 12) * 2 = y) :
  x * 4 = y := sorry

end original_ratio_l400_400010


namespace mul_inv_5_mod_31_l400_400134

theorem mul_inv_5_mod_31 : gcd 5 31 = 1 → ∃ x : ℤ, x * 5 % 31 = 1 ∧ x ≡ 25 [MOD 31] :=
by
  intro h
  -- skip the proof
  sorry

end mul_inv_5_mod_31_l400_400134


namespace count_primes_50_60_l400_400471

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400471


namespace polynomial_divisibility_l400_400972

theorem polynomial_divisibility (a b : ℚ) :
  (∀ x : ℚ, (a + b) * x^5 + a * b * x^2 + 1 = 0 → (x - 1) * (x - 2) = 0) ↔ 
  ({a, b} = {-1, 31/28} ∨ {a, b} = {31/28, -1}) :=
by
  sorry

end polynomial_divisibility_l400_400972


namespace num_primes_between_50_and_60_l400_400625

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400625


namespace line_intersects_circle_at_two_points_min_length_chord_eqn_l400_400166

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25
noncomputable def line_eq (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_intersects_circle_at_two_points (m : ℝ) :
  ∃ x y : ℝ, circle_eq x y ∧ line_eq m x y  ∧ ¬(∃! p : ℝ × ℝ, circle_eq p.1 p.2 ∧ line_eq m p.1 p.2) := sorry

theorem min_length_chord_eqn : ∃ m : ℝ → ℝ, ∀ x y : ℝ, circle_eq x y → 
  line_eq (m (x - 3) (y - 1)) x y ∧ m (x - 3) (y - 1) = 2  := sorry

end line_intersects_circle_at_two_points_min_length_chord_eqn_l400_400166


namespace sector_area_is_correct_l400_400684

-- Definitions extracted from the conditions
def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ
def sector_area (r : ℝ) (θ : ℝ) : ℝ := (1 / 2) * r^2 * θ

-- Given conditions
def l : ℝ := 2
def θ : ℝ := 2

-- Proof statement (theorem) to be proved
theorem sector_area_is_correct : ∃ r : ℝ, arc_length r θ = l ∧ sector_area r θ = 1 :=
by
  sorry

end sector_area_is_correct_l400_400684


namespace goods_train_speed_l400_400035

noncomputable def goodsTrainSpeed (manTrainSpeed_kmph : ℝ) (goodsTrainLength_m : ℝ) (passingTime_s : ℝ) : ℝ :=
  let manTrainSpeed_mps := manTrainSpeed_kmph * 1000 / 3600
  let relativeSpeed_mps := goodsTrainLength_m / passingTime_s
  let goodsTrainSpeed_mps := relativeSpeed_mps - manTrainSpeed_mps
  goodsTrainSpeed_mps * 3600 / 1000

theorem goods_train_speed (manTrainSpeed_kmph : ℝ) (goodsTrainLength_m : ℝ) (passingTime_s : ℝ) :
  manTrainSpeed_kmph = 64 → goodsTrainLength_m = 420 → passingTime_s = 18 →
  goodsTrainSpeed manTrainSpeed_kmph goodsTrainLength_m passingTime_s ≈ 19.98 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end goods_train_speed_l400_400035


namespace number_of_primes_between_50_and_60_l400_400367

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400367


namespace primes_between_50_and_60_l400_400407

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400407


namespace acceptable_points_parallel_l400_400018

-- Define the basic set up of the problem, including the points and intersections
variable {α : Type} [EuclideanGeometry α]

-- Define points and the properties as assumed from the problem
variables (A B C D M P E F : α)
variables [Ne : P ≠ A] [Nb : P ≠ B] [Nc : P ≠ C] [Nd : P ≠ D] [Nm : P ≠ M]

-- Define the square and its properties
variables (Hsquare : is_square A B C D)
variables (Hcenter : M = midpoint A C)

-- Define the intersections as described in the problem
variables (H_E : E = intersection (line_through P D) (line_through A C))
variables (H_F : F = intersection (line_through P C) (line_through B D))

-- Define the condition for E and F to exist
variables (H_E_exists : ∃ E, E = intersection (line_through P D) (line_through A C))
variables (H_F_exists : ∃ F, F = intersection (line_through P C) (line_through B D))

-- The circumcircle of triangle DOC where O is the center of square
variables (O : α) (HcenterO : O = M) (Hperp : ∠ D P C = 90)

-- Define the set of acceptable points
def acceptable_points (P : α) : Prop :=
  P ∈ circumcircle (triangle D O C) ∧ P ≠ O ∧ P ≠ C ∧ P ≠ D ∧ P ≠ reflection O CD

-- Prove that for any acceptable point P, the line EF is parallel to AD
theorem acceptable_points_parallel (P : α) (Hacc : acceptable_points P):
  parallel (line_through E F) (line_through A D) :=
  sorry

end acceptable_points_parallel_l400_400018


namespace find_415th_digit_of_18_div_47_l400_400019

theorem find_415th_digit_of_18_div_47 :
  let seq := "3829787234042553191489361702127659574468085106382978723404255319148936170212765957446808510638297872340425531914893617021276595744680851063829787234042553191489361702127659574468085106382978723404255319148936170212765957446808510638297872340425531914893617021276595744680851063829787234042553191489361702127"
  char_of seq 38 = '1' :=    -- Note that Lean indexing starts at 0, so 39th digit is index 38
by
  sorry

end find_415th_digit_of_18_div_47_l400_400019


namespace primes_between_50_and_60_l400_400406

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400406


namespace prime_count_50_to_60_l400_400498

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400498


namespace number_of_primes_between_50_and_60_l400_400359

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400359


namespace primes_between_50_and_60_l400_400429

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400429


namespace prime_count_between_50_and_60_l400_400354

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400354


namespace correct_statements_count_l400_400160

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (3 * x + Real.pi / 4)

def statement1 : Prop := Real.periodic_fun ((2 * Real.sin) ∘ (3 * .)) (2 * Real.pi)
def statement2 : Prop := ∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 6), 
  f (x + (1 / (3 * 1))) > f x
def statement3 : Prop := ∀ x ∈ Set.Icc (-Real.pi / 9) (Real.pi / 4),
  -Real.sqrt 3 ≤ f x ∧ f x ≤ Real.sqrt 3
def statement4 : Prop := ∀ x : ℝ, f x = g (x - Real.pi / 12)

theorem correct_statements_count :
  (Bool.toNat (statement2)) = 1 ∧
  (Bool.toNat (¬ statement1)) + (Bool.toNat (¬ statement3)) + (Bool.toNat (¬ statement4)) = 3 := by
  sorry

end correct_statements_count_l400_400160


namespace primes_between_50_and_60_l400_400217

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400217


namespace primes_between_50_and_60_l400_400439

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400439


namespace fraction_repetend_l400_400123

-- Lean statement to prove that the repetend of the fraction \frac{7}{29} is "241379".

theorem fraction_repetend (n d : ℕ) (h : d ≠ 0) (num : n = 7) (den : d = 29) :
  (decimal_repetend n d h) = "241379" := by
  sorry

end fraction_repetend_l400_400123


namespace convert_to_base_8_l400_400105

theorem convert_to_base_8 (n : ℕ) (hn : n = 3050) : 
  ∃ d1 d2 d3 d4 : ℕ, d1 = 5 ∧ d2 = 7 ∧ d3 = 5 ∧ d4 = 2 ∧ n = d1 * 8^3 + d2 * 8^2 + d3 * 8^1 + d4 * 8^0 :=
by 
  use 5, 7, 5, 2
  sorry

end convert_to_base_8_l400_400105


namespace primes_between_50_and_60_l400_400388

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400388


namespace primes_between_50_and_60_l400_400220

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400220


namespace max_value_min_fx_gx_l400_400143

-- Definition for min function based on the problem statement
def min (a b : ℝ) : ℝ := if a < b then a else b

-- Definition of f(x) and g(x)
def f (x : ℝ) : ℝ := 4 - x^2
def g (x : ℝ) : ℝ := 3 * x

-- Theorem statement for the maximum value of min(f(x), g(x))
theorem max_value_min_fx_gx : ∃ (M : ℝ), M = 3 ∧ ∀ x : ℝ, min (f x) (g x) ≤ M :=
sorry

end max_value_min_fx_gx_l400_400143


namespace num_primes_between_50_and_60_l400_400616

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400616


namespace min_y_in_quadratic_l400_400852

theorem min_y_in_quadratic (x : ℝ) : ∃ y : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ y', (y' = x^2 + 16 * x + 20) → y ≤ y' := 
sorry

end min_y_in_quadratic_l400_400852


namespace primes_between_50_and_60_l400_400411

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400411


namespace prime_count_between_50_and_60_l400_400335

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400335


namespace solve_fraction_inequality_l400_400779

theorem solve_fraction_inequality :
  { x : ℝ | x / (x + 5) ≥ 0 } = { x : ℝ | x < -5 } ∪ { x : ℝ | x ≥ 0 } := by
  sorry

end solve_fraction_inequality_l400_400779


namespace primes_between_50_and_60_l400_400227

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400227


namespace monotonicity_of_f_bound_on_f_when_k_negative_l400_400200

def f (x k : ℝ) : ℝ := Real.log x + k * x^2 + (2 * k + 1) * x

noncomputable def f'_positive (x k : ℝ) (h_pos : 0 < x) (h_nonneg : 0 ≤ k) : Prop :=
  0 < (2 * k * x^2 + (2 * k + 1) * x + 1) / x

noncomputable def f'_negative (x k : ℝ) (h_pos : 0 < x) (h_neg : k < 0) : Prop :=
  (2 * k * x + 1) * (x + 1) = 0

theorem monotonicity_of_f (k : ℝ) :
  (∀ x > 0, f'_positive x k → (0 ≤ k → ∀ y > 0, f y k ≥ f x k ∧ f' y k > 0)) ∧
  (∀ x > 0, f'_negative x k → (k < 0 → ∃ x0 > 0, x0 = -1 / (2 * k) ∧
  ∀ y > 0, y < x0 → f y k > f x0 k ∧ y > x0 → f y k < f x0 k )) :=
sorry

theorem bound_on_f_when_k_negative (k : ℝ) (h_neg : k < 0) :
  ∀ x > 0, f x k ≤ 3 / (4 * k) - 2 :=
sorry

end monotonicity_of_f_bound_on_f_when_k_negative_l400_400200


namespace prime_count_50_to_60_l400_400504

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400504


namespace original_people_count_l400_400759

theorem original_people_count (x : ℕ) 
  (H1 : (x - x / 3) / 2 = 15) : x = 45 := by
  sorry

end original_people_count_l400_400759


namespace diameter_of_circle_C_l400_400099

-- Define the initial conditions
def radius_of_D : ℝ := 10  -- Radius of circle D
def area_of_D : ℝ := π * radius_of_D^2  -- Area of circle D
def ratio : ℝ := 7  -- Ratio given in the problem

noncomputable def radius_of_C_sqrt : ℝ := (radius_of_D^2 / (ratio + 1)).sqrt
noncomputable def diameter_of_C : ℝ := 2 * radius_of_C_sqrt

-- The theorem statement
theorem diameter_of_circle_C :
  diameter_of_C = 2 * real.sqrt (radius_of_D^2 / (ratio + 1)) :=
by
  sorry

end diameter_of_circle_C_l400_400099


namespace primes_count_between_50_and_60_l400_400552

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400552


namespace ball_travel_distance_l400_400048

/-- A super ball is dropped from a window 20 meters above the ground.
    On each bounce, it rises 2/3 of the height it reached after the previous bounce.
    The ball is caught when it reaches the high point after hitting the ground for the fourth time.
    To the nearest meter, how far has it traveled? -/
theorem ball_travel_distance :
  let initial_height := 20
  let ratio := (2 : ℝ) / 3
  let descent1 := initial_height
  let descent2 := initial_height * ratio
  let descent3 := descent2 * ratio
  let descent4 := descent3 * ratio
  let total_descent := descent1 + descent2 + descent3 + descent4
  let ascent1 := descent2
  let ascent2 := ascent1 * ratio
  let ascent3 := ascent2 * ratio
  let ascent4 := ascent3 * ratio
  let total_ascent := ascent1 + ascent2 + ascent3 + ascent4
  let total_distance := total_descent + total_ascent
  in round total_distance = 80 :=
by
  sorry

end ball_travel_distance_l400_400048


namespace prime_count_between_50_and_60_l400_400567

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400567


namespace primes_between_50_and_60_l400_400312

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400312


namespace primes_between_50_and_60_l400_400418

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400418


namespace primes_between_50_and_60_l400_400432

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400432


namespace parallel_lines_m_value_l400_400980

theorem parallel_lines_m_value
  (m : ℤ) 
  (h1 : ∀ x y : ℤ, x + (1 + m) * y = 2 - m)
  (h2 : ∀ x y : ℤ, 2 * m * x + 4 * y + 16 = 0)
  (parallel : slope (x + (1 + m) * y = 2 - m) = slope (2 * m * x + 4 * y + 16 = 0)) :
  m = 1 := 
sorry

-- Note: slope function and parallelism criteria need to be defined or assumed.

end parallel_lines_m_value_l400_400980


namespace primes_between_50_and_60_l400_400537

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400537


namespace hex_numbers_under_1000_sum_l400_400059

open Nat

/- Given conditions translated into Lean definitions -/
def isHexadecimal (d : ℕ) : Prop := d < 16
def hexDigitSum (n : ℕ) : ℕ := n.digits 10 |>.sum

/- Prove that the count of valid numbers and their digit sum -/
theorem hex_numbers_under_1000_sum :
  let n := (Finset.range 1000).filter (λ x, isHexadecimal (x / 16 ^ 2) ∧ isHexadecimal (x / 16 % 16) ∧ isHexadecimal (x % 16))
  ∑ d in n, d = 21 :=
by
  -- Here we acknowledge the theorem but do not provide the proof.
  sorry

end hex_numbers_under_1000_sum_l400_400059


namespace primes_between_50_and_60_l400_400595

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400595


namespace median_parallel_to_BC_median_to_BC_l400_400158

variables {A B C : Point ℝ}
variables (A : Point ℝ) [Coord A (1, -4)]
variables (B : Point ℝ) [Coord B (6, 6)]
variables (C : Point ℝ) [Coord C (-2, 0)]

theorem median_parallel_to_BC :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, on_median_parallel_to_BC A B C x y) → 6*x - 8*y - 13 = 0 := 
sorry

theorem median_to_BC :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, on_median_to_BC A B C x y) → 7*x - y - 11 = 0 := 
sorry

end median_parallel_to_BC_median_to_BC_l400_400158


namespace graph_of_neg_g_l400_400811

def g (x : ℝ) : ℝ :=
  if x ∈ Icc (-5 : ℝ) (-2) then x + 3
  else if x ∈ Icc (-2) (1) then -real.sqrt (9 - (x + 1)^2) + 1
  else if x ∈ Icc (1) (4) then 2 * (x - 1) - 2
  else 0

theorem graph_of_neg_g :
  ∃ (graph : set (ℝ × ℝ)),
  (∀ x ∈ Icc (2 : ℝ) (5), (λ x, -x + 3) (x, g (-x))) ∧
  (∀ x ∈ Icc (-1) (2), (λ x, -real.sqrt (9 - (-x - 1)^2) + 1) (x, g (-x))) ∧
  (∀ x ∈ Icc (-4) (-1), (λ x, -2 * (x + 1) + 2) (x, g (-x))) := sorry

end graph_of_neg_g_l400_400811


namespace intersection_point_exists_l400_400011

noncomputable def line1 (t : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + 6 * t)
noncomputable def line2 (u : ℝ) : ℝ × ℝ := (3 + u, 8 + 3 * u)

theorem intersection_point_exists :
  ∃ t u : ℝ, line1 t = (1, 2) ∧ line2 u = (1, 2) := 
by
  sorry

end intersection_point_exists_l400_400011


namespace primes_between_50_and_60_l400_400528

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400528


namespace unique_triple_l400_400136

def sign (a : ℝ) : ℝ :=
if a > 0 then 1 else if a = 0 then 0 else -1

theorem unique_triple :
  ∃! (x y z : ℝ), 
    x = 2023 - 2024 * (sign (y + z - 1)) ∧
    y = 2023 - 2024 * (sign (x + z - 1)) ∧
    z = 2023 - 2024 * (sign (x + y - 1)) :=
sorry

end unique_triple_l400_400136


namespace shaded_area_l400_400942

-- Definitions based on conditions
def diameter_smaller_circle : ℝ := 6
def radius_smaller_circle : ℝ := diameter_smaller_circle / 2
def radius_larger_circle : ℝ := 5 * radius_smaller_circle

-- Theorem statement to prove
theorem shaded_area {π : ℝ} [Real π] :
  let area_larger_circle := π * radius_larger_circle^2
  let area_smaller_circle := π * radius_smaller_circle^2
  area_larger_circle - area_smaller_circle = 216 * π :=
by
  sorry

end shaded_area_l400_400942


namespace collinear_points_find_k_right_angle_triangle_find_k_l400_400209

-- Given vectors
def vector_PA (k : ℝ) : ℝ × ℝ := (k, 12)
def vector_PB : ℝ × ℝ := (4, 5)
def vector_PC (k : ℝ) : ℝ × ℝ := (10, k)

-- Problem 1: Collinearity of points A, B, and C
theorem collinear_points_find_k (k : ℝ) (λ : ℝ) :
    (4 - k, -7) = λ • (6, k - 5) → (k = 11 ∨ k = -2) :=
  sorry

-- Problem 2: Right-angled triangle with vertices A, B, C
theorem right_angle_triangle_find_k (k : ℝ) :
    (6 * (4 - k) - 7 * (k - 5) = 0 → k = 59 / 13) ∧
    ((6, k - 5).fst * (10 - k, k - 12).fst + (6, k - 5).snd * (10 - k, k - 12).snd = 0 → (k = 8 ∨ k = 15)) :=
  sorry

end collinear_points_find_k_right_angle_triangle_find_k_l400_400209


namespace primes_between_50_and_60_l400_400436

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400436


namespace number_of_primes_between_50_and_60_l400_400370

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400370


namespace prime_count_50_to_60_l400_400503

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400503


namespace min_value_y_l400_400848

theorem min_value_y : ∃ x : ℝ, (∀ y : ℝ, y = x^2 + 16 * x + 20 → y ≥ -44) :=
begin
  use -8,
  intro y,
  intro hy,
  suffices : y = (x + 8)^2 - 44,
  { rw this,
    exact sub_nonneg_of_le (sq_nonneg (x + 8)) },
  sorry
end

end min_value_y_l400_400848


namespace primes_between_50_and_60_l400_400641

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400641


namespace solve_system_l400_400785

theorem solve_system (x y : ℝ) (h₁ : 3^y * 81 = 9^(x^2)) (h₂ : log 10 y = log 10 x - log 10 0.5) :
  x = 2 ∧ y = 4 :=
by
  sorry

end solve_system_l400_400785


namespace sum_largest_smallest_angles_l400_400927

theorem sum_largest_smallest_angles (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 8) :
  ∃ θ₁ θ₃ : ℝ, θ₁ + θ₃ = 120 ∧
               θ₁ = 60 ∧
               (θ₁ < θ₂ ∧ θ₃ < θ₂ ∨ θ₂ < θ₁ ∧ θ₂ < θ₃) ∧
               (∀ θ₁ θ₂ θ₃ : ℝ, θ₁ = acos ((b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c)) ∧
                                 θ₂ = acos ((a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c)) ∧
                                 θ₃ = acos ((a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)) ∧
                                 θ₁ + θ₂ + θ₃ = 180)
:= sorry

end sum_largest_smallest_angles_l400_400927


namespace arcsin_sin_2000_l400_400878

-- Define the relevant angles
def angle_2000 := 2000 * (Real.pi / 180)
def angle_360 := 360 * (Real.pi / 180)
def angle_180 := 180 * (Real.pi / 180)
def angle_20 := 20 * (Real.pi / 180)

-- Define the conditions
lemma condition1 : angle_2000 = 5 * angle_360 + angle_200 := by sorry
lemma condition2 : Real.sin angle_2000 = Real.sin angle_200 := by sorry
lemma condition3 : Real.sin angle_200 = Real.sin (angle_180 + angle_20) := by sorry
lemma condition4 : Real.sin (angle_180 + angle_20) = - Real.sin angle_20 := by sorry
lemma condition5 : Real.arcsin (Real.sin (- angle_20)) = - angle_20 := by sorry

-- Prove the main statement
theorem arcsin_sin_2000 : Real.arcsin (Real.sin angle_2000) = - angle_20 :=
  by
  have h1 := condition1
  have h2 := condition2
  have h3 := condition3
  have h4 := condition4
  have h5 := condition5
  sorry

end arcsin_sin_2000_l400_400878


namespace pair_solution_l400_400941

theorem pair_solution (a b : ℕ) (h_b_ne_1 : b ≠ 1) :
  (a + 1 ∣ a^3 * b - 1) → (b - 1 ∣ b^3 * a + 1) →
  (a, b) = (0, 0) ∨ (a, b) = (0, 2) ∨ (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3) :=
by
  sorry

end pair_solution_l400_400941


namespace angle_CBD_eq_130_l400_400879

-- Define angles in degrees
def angle := ℤ

-- Define the problem conditions
variables {A B C D : Type}
variables (m∠ABC m∠BAC m∠C m∠CBD : angle)

-- Condition: Triangle ABC is right-angled at B
def right_triangle_at_B (m∠ABC : angle) : Prop := (m∠ABC = 90)

-- Condition: m∠C is given as 40 degrees
def angle_C (m∠C : angle) : Prop := (m∠C = 40)

-- Given that D lies on the extension of line AB
def extension_point (m∠CBD : angle) (m∠BAC : angle) : Prop :=
  (m∠CBD = 180 - m∠BAC)

-- Theorem: To prove m∠CBD is 130 degrees
theorem angle_CBD_eq_130 
  (h1 : right_triangle_at_B m∠ABC)
  (h2 : angle_C m∠C)
  (h3 : extension_point m∠CBD m∠BAC)
  : m∠CBD = 130 :=
by
  -- Definition to formalize the conditions into derivable facts
  have h4 : m∠ABC = 90, from h1,
  have h5 : m∠C = 40, from h2,

  -- Use the sum of angles in triangle property for ∠BAC
  let m∠BAC := 180 - (m∠ABC + m∠C),

  -- Substitute the known values
  have h6 : m∠BAC = 50,
  { calc
      m∠BAC = 180 - (90 + 40) : by sorry
      ... = 50                : by sorry },

  -- Now use the exterior angle property to get m∠CBD
  show m∠CBD = 130,
  { calc
      m∠CBD = 180 - m∠BAC : by sorry
      ... = 180 - 50      : by sorry
      ... = 130           : by sorry }

end angle_CBD_eq_130_l400_400879


namespace distance_to_Tianbo_Mountain_l400_400112

theorem distance_to_Tianbo_Mountain : ∀ (x y : ℝ), 
  (x ≠ 0) ∧ 
  (y = 3) ∧ 
  (∀ v, v = (4 * y + x) * ((2 * x - 8) / v)) ∧ 
  (2 * (y * x) = 8 * y + x^2 - 4 * x) 
  → 
  (x + y = 9) := 
by
  sorry

end distance_to_Tianbo_Mountain_l400_400112


namespace tulip_count_l400_400969

theorem tulip_count (total_flowers : ℕ) (daisies : ℕ) (roses_ratio : ℚ)
  (tulip_count : ℕ) :
  total_flowers = 102 →
  daisies = 6 →
  roses_ratio = 5 / 6 →
  tulip_count = (total_flowers - daisies) * (1 - roses_ratio) →
  tulip_count = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end tulip_count_l400_400969


namespace factorize_x_squared_sub_xy_l400_400118

theorem factorize_x_squared_sub_xy (x y : ℝ) : x^2 - x * y = x * (x - y) :=
sorry

end factorize_x_squared_sub_xy_l400_400118


namespace num_primes_between_50_and_60_l400_400615

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400615


namespace count_primes_between_50_and_60_l400_400674

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400674


namespace primes_between_50_and_60_l400_400431

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400431


namespace probability_of_x_ge_1_5_in_interval_1_3_l400_400988

theorem probability_of_x_ge_1_5_in_interval_1_3 :
  let interval := set.Icc (1 : ℝ) 3 
  let event_set := set.Icc (1.5 : ℝ) 3 
  (event_set.measure / interval.measure) = 0.75 := by
  let interval_length : ℝ := 3 - 1
  let event_length : ℝ := 3 - 1.5
  have interval_measure : ℝ := interval_length
  have event_measure : ℝ := event_length
  sorry

end probability_of_x_ge_1_5_in_interval_1_3_l400_400988


namespace line_slope_and_point_l400_400896

noncomputable def line_equation (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem line_slope_and_point (m b : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 5) (h₃ : y₀ = 2) (h₄ : y₀ = line_equation x₀ m b) :
  m + b = 14 :=
by
  sorry

end line_slope_and_point_l400_400896


namespace M_is_positive_integers_l400_400170

theorem M_is_positive_integers
  (M : Set ℕ)
  (h1 : 2018 ∈ M)
  (h2 : ∀ (m ∈ M) (d : ℕ), d > 0 → d ∣ m → d ∈ M)
  (h3 : ∀ (k m ∈ M), 1 < k → k < m → k * m + 1 ∈ M) :
  M = {n : ℕ | n > 0} :=
begin
  sorry
end

end M_is_positive_integers_l400_400170


namespace primes_between_50_and_60_l400_400635

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400635


namespace num_primes_50_60_l400_400271

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400271


namespace count_primes_between_50_and_60_l400_400670

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400670


namespace danny_watermelon_slices_l400_400935

theorem danny_watermelon_slices : 
  ∀ (x : ℕ), 3 * x + 15 = 45 -> x = 10 := by
  intros x h
  sorry

end danny_watermelon_slices_l400_400935


namespace isosceles_right_triangle_center_of_mass_l400_400126

def centerOfMass (a : ℝ) (k : ℝ) : ℝ × ℝ :=
⟨0, a / 2⟩

theorem isosceles_right_triangle_center_of_mass
  (a k : ℝ)
  (triangle_vertices : ℝ × ℝ → Prop)
  (H : ∀ (x y : ℝ), triangle_vertices (x, y) ↔ (y ≤ x + a ∧ y ≤ a - x ∧ 0 ≤ y ∧ -a ≤ x ∧ x ≤ a))
  (density : ℝ × ℝ → ℝ)
  (H_density : ∀ (x y : ℝ), density (x,y) = k * y)
  : centerOfMass a k = (0, a / 2) :=
sorry

end isosceles_right_triangle_center_of_mass_l400_400126


namespace sum_x_coordinates_eq_4_5_l400_400932

-- Define the line segments of the function
def segment1 (x : ℝ) : ℝ := if x ≥ -4 ∧ x ≤ -2 then (2 * x + 3) else 0
def segment2 (x : ℝ) : ℝ := if x ≥ -2 ∧ x <= -1 then (-x -1) else 0
def segment3 (x : ℝ) : ℝ := if x ≥ -1 ∧ x ≤ 1 then (-x + 2) else 0
def segment4 (x : ℝ) : ℝ := if x ≥ 1 ∧ x <= 2 then (-x + 2) else 0
def segment5 (x : ℝ) : ℝ := if x ≥ 2 ∧ x ≤ 4 then (2 * x -3) else 0

-- Combine the segments to define the complete function
def f (x : ℝ) : ℝ := segment1 x + segment2 x + segment3 x + segment4 x + segment5 x

-- Theorem to prove the sum of x-coordinates where f(x) = 2 is 4.5
theorem sum_x_coordinates_eq_4_5 :
  (∃ x1 x2 x3, f x1 = 2 ∧ f x2 = 2 ∧ f x3 = 2 ∧ x1 + x2 + x3 = 4.5) :=
sorry

end sum_x_coordinates_eq_4_5_l400_400932


namespace primes_between_50_and_60_l400_400639

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400639


namespace primes_between_50_and_60_l400_400653

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400653


namespace num_primes_50_60_l400_400269

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400269


namespace train_pass_time_l400_400076

theorem train_pass_time (train_length : ℕ) (platform_length : ℕ) (speed : ℕ) (h1 : train_length = 50) (h2 : platform_length = 100) (h3 : speed = 15) : 
  (train_length + platform_length) / speed = 10 :=
by
  sorry

end train_pass_time_l400_400076


namespace median_length_AD_l400_400186

variable {α : Type*} [Real α]

-- Define the conditions: angles and side lengths
def angle_sum (A B C : α) : Prop := A + B + C = real.pi
def angle_relation (A B C : α) : Prop := 2 * B = A + C 
def side_lengths (AB BC : α) : Prop := AB = 1 ∧ BC = 4

-- Define the target: length of the median AD
def median_length (AB BD AD : α) (B : α) : Prop :=
  AD^2 = AB^2 + BD^2 - 2 * AB * BD * real.cos(B)

theorem median_length_AD (A B C : α) (AB BC BD AD : α) 
  (h1 : angle_sum A B C) 
  (h2 : angle_relation A B C) 
  (h3 : side_lengths AB BC) 
  (h4 : BD = BC / 2)
  (h5 : B = real.pi / 3) :
  AD = real.sqrt 3 :=
by
  sorry

end median_length_AD_l400_400186


namespace pencil_length_total_l400_400902

theorem pencil_length_total :
  (1.5 + 0.5 + 2 + 1.25 + 0.75 + 1.8 + 2.5 = 10.3) :=
by
  sorry

end pencil_length_total_l400_400902


namespace primes_between_50_and_60_l400_400594

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400594


namespace primes_between_50_and_60_l400_400290

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400290


namespace prime_count_50_to_60_l400_400505

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400505


namespace claire_earnings_l400_400101

noncomputable theory

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def total_roses : ℕ := total_flowers - tulips
def white_roses : ℕ := 80
def red_roses : ℕ := total_roses - white_roses
def small_red_roses : ℕ := 40
def medium_red_roses : ℕ := 60
def large_red_roses : ℕ := red_roses - small_red_roses - medium_red_roses
def small_red_rose_price : ℝ := 0.75
def medium_red_rose_price : ℝ := 1
def large_red_rose_price : ℝ := 1.25

def sell_fraction : ℝ := 1 / 2

def small_red_roses_to_sell : ℕ := (sell_fraction * small_red_roses).to_nat
def medium_red_roses_to_sell : ℕ := (sell_fraction * medium_red_roses).to_nat
def large_red_roses_to_sell : ℕ := (sell_fraction * large_red_roses).to_nat

def total_earnings : ℝ :=
  (small_red_roses_to_sell * small_red_rose_price) +
  (medium_red_roses_to_sell * medium_red_rose_price) +
  (large_red_roses_to_sell * large_red_rose_price)

theorem claire_earnings : total_earnings = 107.50 := 
  by 
    sorry -- Proof omitted.

end claire_earnings_l400_400101


namespace matching_socks_probability_correct_l400_400032

/-- Xiao Ming has 3 pairs of black socks and 1 pair of white socks. What is the probability of randomly selecting 2 socks that exactly match? Proof that this probability is 2/7. -/
def matching_socks_probability : ℚ :=
  let total_socks := 8
  let black_socks := 6
  let white_socks := 2
  let total_outcomes := (Nat.choose total_socks 2 : ℚ)
  let favorable_black := (Nat.choose black_socks 2 : ℚ)
  let favorable_white := 1
  let favorable_outcomes := favorable_black + favorable_white
  (favorable_outcomes / total_outcomes)

theorem matching_socks_probability_correct :
  matching_socks_probability = 2 / 7 :=
by
  have total_socks := 8
  have black_socks := 6
  have white_socks := 2
  have total_outcomes := (Nat.choose total_socks 2 : ℚ)
  have favorable_black := (Nat.choose black_socks 2 : ℚ)
  have favorable_white := (1 : ℚ)
  have favorable_outcomes := favorable_black + favorable_white
  have probability := favorable_outcomes / total_outcomes
  have h1 : total_outcomes = 28 :=
    by norm_num [Nat.choose]
  have h2 : favorable_black = 15 :=
    by norm_num [Nat.choose]
  have h3 : favorable_outcomes = 16 :=
    by norm_num; exact (h2 + favorable_white).symm
  have h4 : probability = 2 / 7 :=
    by rw [h1, h3]; norm_num
  exact h4

end matching_socks_probability_correct_l400_400032


namespace primes_between_50_and_60_l400_400637

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400637


namespace num_primes_between_50_and_60_l400_400608

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400608


namespace shaded_area_equality_l400_400711

-- Definitions based on conditions
def θ (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

-- Main theorem statement
theorem shaded_area_equality {θ : ℝ} (hθ : 0 < θ ∧ θ < π / 2) :
  tan θ = 2 * θ :=
sorry

end shaded_area_equality_l400_400711


namespace prime_count_between_50_and_60_l400_400350

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400350


namespace line_equation_through_point_with_intercepts_conditions_l400_400132

theorem line_equation_through_point_with_intercepts_conditions :
  ∃ (a b : ℚ) (m c : ℚ), 
    (-5) * m + c = 2 ∧ -- The line passes through A(-5, 2)
    a = 2 * b ∧       -- x-intercept is twice the y-intercept
    (a * m + c = 0 ∨ ((1/m)*a + (1/m)^2 * c+1 = 0)) :=         -- Equations of the line
sorry

end line_equation_through_point_with_intercepts_conditions_l400_400132


namespace primes_between_50_and_60_l400_400315

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400315


namespace cube_hole_pass_l400_400762

theorem cube_hole_pass (a : ℝ) (ha : a = 1) :
  ∃ (hole : set ℝ^3),
    (∀ (c : set ℝ^3) (hc : cube c = a), cube_pass_through (a, hole, c)) :=
sorry

end cube_hole_pass_l400_400762


namespace angle_of_inclination_l400_400189

variable {a : ℕ → ℝ}   -- The arithmetic sequence

-- Conditions
axiom h1 : a 4 = 15
axiom h2 : (∑ i in finset.range 5, a i) = 55

-- Definition of arithmetic sequence
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Prove: angle of inclination
theorem angle_of_inclination (h_arith : is_arithmetic a) :
  let P := (4 : ℕ, a 2010)
  let Q := (3 : ℕ, a 2011)
  let slope := (a 2011 - a 2010) / (3 - 4 : ℝ)
  let θ := Real.arctan (-slope)
  θ = π - arctan 4 := by
  sorry

end angle_of_inclination_l400_400189


namespace count_distinct_prime_products_l400_400859

-- Define a condition to check if a number is a prime.
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a condition that specifies the sum of three primes equals 118.
def prime_sum_condition (p1 p2 p3 : ℕ) : Prop := 
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (p1 + p2 + p3 = 118)

-- Define the main theorem to count such valid products.
theorem count_distinct_prime_products : 
  ∃ (count : ℕ), count = (finset.filter (λ n, ∃ (p1 p2 p3 : ℕ), prime_sum_condition p1 p2 p3 ∧ n = p1 * p2 * p3) (finset.range 10000)).card :=
sorry

end count_distinct_prime_products_l400_400859


namespace primes_between_50_and_60_l400_400435

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400435


namespace zero_in_interval_l400_400798

open Real 

noncomputable def f (x : ℝ) : ℝ := log (x + 2) / log 2 - 3 / x

theorem zero_in_interval : ∃ c ∈ Ioo 1 2, f c = 0 :=
by
  sorry

end zero_in_interval_l400_400798


namespace seventh_grade_male_students_l400_400072

theorem seventh_grade_male_students:
  ∃ x : ℤ, (48 = x + (4*x)/5 + 3) ∧ x = 25 :=
by
  sorry

end seventh_grade_male_students_l400_400072


namespace trig_identity_1_trig_identity_2_trig_identity_3_trig_identity_4_trig_identity_5_l400_400111

theorem trig_identity_1 (x y : ℝ) : ¬ (sin (x + y) = sin x + sin y) :=
sorry

theorem trig_identity_2 (x y : ℝ) : ¬ (sin (x - y) = sin x - sin y) :=
sorry

theorem trig_identity_3 (x y : ℝ) : (tan (x + y) = (tan x + tan y) / (1 - tan x * tan y)) :=
sorry

theorem trig_identity_4 (x y : ℝ) : (cos (x + y) = cos x * cos y - sin x * sin y) :=
sorry

theorem trig_identity_5 (x : ℝ) : (cos (2 * x) = 2 * cos x ^ 2 - 1) :=
sorry

end trig_identity_1_trig_identity_2_trig_identity_3_trig_identity_4_trig_identity_5_l400_400111


namespace num_primes_between_50_and_60_l400_400624

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400624


namespace lcm_18_28_l400_400847
open Nat

/-- 
The least common multiple (LCM) of 18 and 28 is 252.
-/
theorem lcm_18_28 : lcm 18 28 = 252 :=
by
  sorry -- The proof is omitted

end lcm_18_28_l400_400847


namespace problem_1_l400_400874

variable (x : ℝ) (a : ℝ)

theorem problem_1 (h1 : x - 1/x = 3) (h2 : a = x^2 + 1/x^2) : a = 11 := sorry

end problem_1_l400_400874


namespace count_primes_between_50_and_60_l400_400453

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400453


namespace min_value_fraction_sum_l400_400803

theorem min_value_fraction_sum
  (a b : ℝ) (h0 : a > 0) (h1 : b > 0)
  (h2 : ∀ x y : ℝ, x^2 + y^2 + 2*x - 6*y + 1 = 0 → ax - by + 3 = 0)
  : (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 3 ∧ (∀ x y : ℝ, x^2 + y^2 + 2 * x - 6 * y + 1 = 0 → ax - by + 3 = 0)) 
     → (∃ k : ℝ, k = 16/3) :=
sorry

end min_value_fraction_sum_l400_400803


namespace age_problem_solution_l400_400968

theorem age_problem_solution :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
  a1 + a2 + a3 = 54 ∧
  a5 - a4 = 5 ∧
  a3 + a4 + a5 = 78 ∧
  a2 - a1 = 7 ∧
  a1 + a5 = 44 ∧
  a1 = 13 ∧
  a2 = 20 ∧
  a3 = 21 ∧
  a4 = 26 ∧
  a5 = 31 :=
by
  -- We should skip the implementation because the solution is provided in the original problem.
  sorry

end age_problem_solution_l400_400968


namespace smallest_unit_of_money_correct_l400_400047

noncomputable def smallest_unit_of_money (friends : ℕ) (total_bill paid_amount : ℚ) : ℚ :=
  if (total_bill % friends : ℚ) = 0 then
    total_bill / friends
  else
    1 % 100

theorem smallest_unit_of_money_correct :
  smallest_unit_of_money 9 124.15 124.11 = 1 % 100 := 
by
  sorry

end smallest_unit_of_money_correct_l400_400047


namespace primes_count_between_50_and_60_l400_400555

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400555


namespace primes_between_50_and_60_l400_400605

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400605


namespace mutually_exclusive_events_l400_400151

-- Define the bag, balls, and events
def bag := (5, 3) -- (red balls, white balls)

def draws (r w : Nat) := (r + w = 3)

def event_A (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.1 = 3 -- At least one red ball and all red balls
def event_B (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.2 = 3 -- At least one red ball and all white balls
def event_C (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.2 ≥ 1 -- At least one red ball and at least one white ball
def event_D (draw : ℕ × ℕ) := (draw.1 = 1 ∨ draw.1 = 2) ∧ draws draw.1 draw.2 -- Exactly one red ball and exactly two red balls

theorem mutually_exclusive_events : 
  ∀ draw : ℕ × ℕ, 
  (event_A draw ∨ event_B draw ∨ event_C draw ∨ event_D draw) → 
  (event_D draw ↔ (draw.1 = 1 ∧ draw.2 = 2) ∨ (draw.1 = 2 ∧ draw.2 = 1)) :=
by
  sorry

end mutually_exclusive_events_l400_400151


namespace primes_between_50_and_60_l400_400286

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400286


namespace unique_t_digit_l400_400001

theorem unique_t_digit (t : ℕ) (ht : t < 100) (ht2 : 10 ≤ t) (h : 13 * t ≡ 42 [MOD 100]) : t = 34 := 
by
-- Proof is omitted
sorry

end unique_t_digit_l400_400001


namespace count_primes_between_50_and_60_l400_400663

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400663


namespace broken_line_length_lt_sum_of_adj_sides_l400_400164

theorem broken_line_length_lt_sum_of_adj_sides
  (x : ℕ → ℝ)
  (n : ℕ)
  (a b : ℝ)
  (lines_intersect_once : ∀ k : ℕ, k < n → (∀ l : ℕ, l < n → (x k ≠ x l ∨ (x k = x l ∧ k = l))))
  (a_i_sum : ∑ i in finset.range n, x i < a)
  (b_i_sum : ∑ i in finset.range n, x i < b) :
  ∑ i in finset.range n, x i < a + b := by
  sorry

end broken_line_length_lt_sum_of_adj_sides_l400_400164


namespace binomial_expansion_coefficient_l400_400190

theorem binomial_expansion_coefficient (n : ℕ) (C : ℕ → ℕ → ℕ)
  (h_sum_coeffs : ∑ k in Finset.range (n + 1), C n k * (2:ℝ)^(n - k) * ((-1:ℝ)^k) = 128) :
  let k := 5 in C 7 k * (2:ℝ)^(7 - k) * ((-1:ℝ)^k) = -84 :=
by
  sorry

end binomial_expansion_coefficient_l400_400190


namespace shaniqua_style_income_correct_l400_400773

def shaniqua_income_per_style (haircut_income : ℕ) (total_income : ℕ) (number_of_haircuts : ℕ) (number_of_styles : ℕ) : ℕ :=
  (total_income - (number_of_haircuts * haircut_income)) / number_of_styles

theorem shaniqua_style_income_correct :
  shaniqua_income_per_style 12 221 8 5 = 25 :=
by
  sorry

end shaniqua_style_income_correct_l400_400773


namespace pentagon_angle_D_measure_l400_400703

theorem pentagon_angle_D_measure :
  ∀ (A B C D E : ℝ),
    A = B ∧ B = C ∧ D = E ∧ A + 50 = D ∧ A + B + C + D + E = 540 →
    D = 138 :=
by
  intros A B C D E h
  cases h with h1 htemp
  cases htemp with h2 hcopy
  cases hcopy with h3 h4
  cases h4 with h5 h6
  sorry

end pentagon_angle_D_measure_l400_400703


namespace product_1_to_1000_correct_l400_400763

-- Define conditions from the problem.

-- Initial values
def initial_i := 1
def initial_S := 1

-- The final product calculation function
def product (n : ℕ) : ℕ :=
  if n = 1 then 1 else n * product (n - 1)

-- Proof goal: the final value of S corresponds to the product of the first 1000 natural numbers.
theorem product_1_to_1000_correct :
  let S := product 1000 in 
  S = ∏ i in (Finset.range 1000 + 1), i :=
begin
  sorry
end

end product_1_to_1000_correct_l400_400763


namespace primes_between_50_and_60_l400_400603

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400603


namespace primes_between_50_and_60_l400_400301

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400301


namespace train_speed_kmh_l400_400913

variable (length_of_train_meters : ℕ) (time_to_cross_seconds : ℕ)

theorem train_speed_kmh (h1 : length_of_train_meters = 50) (h2 : time_to_cross_seconds = 6) :
  (length_of_train_meters * 3600) / (time_to_cross_seconds * 1000) = 30 :=
by
  sorry

end train_speed_kmh_l400_400913


namespace primes_count_between_50_and_60_l400_400553

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400553


namespace count_primes_between_50_and_60_l400_400241

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400241


namespace primes_count_between_50_and_60_l400_400539

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400539


namespace simplify_and_evaluate_expr_l400_400776

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l400_400776


namespace rhombus_area_l400_400839

theorem rhombus_area (a : ℝ) (theta : ℝ) (h_side_length : a = 3) (h_angle : theta = π / 4) :
  let area := a * a * sin(theta)
  in area = 9 * Real.sqrt 2 / 2 :=
by sorry

end rhombus_area_l400_400839


namespace part1_part2_l400_400991

-- Define conditions based on the problem statement
theorem part1 (a t : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < t → x^2 - 2 * x + a < 0) → 
  a = -8 ∧ t = 4 := 
by 
  sorry

theorem part2 (c : ℝ) : 
  (∀ x : ℝ, (7 < c ∧ c ≤ 8) → (-(8 - c)x^2 + 2(8 - c)x - 1 < 0)) → 
  ∀ x : ℝ, (c + (-8))x^2 + 2(c + (-8))x - 1 < 0 → c ∈ Ioi 7 ∧ c ∈ Iic 8 := 
by 
  sorry

end part1_part2_l400_400991


namespace tan_A_tan_B_range_l400_400691

theorem tan_A_tan_B_range (a b c A B C : ℝ) (h : 3 * a ^ 2 = c ^ 2 - b ^ 2)
  (ha : a = b * cos C) : 0 < (tan A) * (tan B) ∧ (tan A) * (tan B) < 1 / 2 :=
sorry

end tan_A_tan_B_range_l400_400691


namespace max_sum_d_floor_l400_400990

def max_sum_d (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  ∑ k in Finset.range n, min (abs (x k - ⌊x k⌋)) (abs (x k - (⌊x k⌋ + 1)))

theorem max_sum_d_floor (n : ℕ) (x : ℕ → ℝ) (h_sum_int : ∑ k in Finset.range n, x k ∈ ℤ) : 
  max_sum_d n x = ⌊n / 2⌋ :=
by
  /\*sorry\*/

end max_sum_d_floor_l400_400990


namespace sector_angle_l400_400799

theorem sector_angle (A : ℝ) (r : ℝ) (θ : ℝ) (h1 : A = 1) (h2 : r = 1) : θ = 2 :=
by
  have eq1 : A = π * r^2 * (θ / (2 * π)), from sorry,
  rw [h1, h2] at eq1,
  solve by assumption or further simplification
  sorry

end sector_angle_l400_400799


namespace count_primes_50_60_l400_400483

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400483


namespace primes_between_50_and_60_l400_400417

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400417


namespace primes_between_50_and_60_l400_400426

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400426


namespace primes_between_50_and_60_l400_400308

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400308


namespace speed_in_still_water_l400_400061

-- Define the given conditions
def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

-- State the theorem to be proven
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 40 := by
  -- Proof omitted
  sorry

end speed_in_still_water_l400_400061


namespace primes_between_50_and_60_l400_400235

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400235


namespace primes_between_50_and_60_l400_400385

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400385


namespace primes_between_50_and_60_l400_400378

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400378


namespace shipping_percentage_proof_l400_400917

noncomputable def shipping_percentage (p : ℝ) (total_bill : ℝ) : ℝ :=
  (total_bill - p) / p * 100

theorem shipping_percentage_proof :
  let total_purchase_price : ℝ := 3 * 12 + 5 + 2 * 15 + 14 in
  let total_bill : ℝ := 102 in
  total_purchase_price = 85 ∧
  total_bill = 102 →
  total_purchase_price > 50 →
  shipping_percentage total_purchase_price total_bill = 20 :=
by
  -- define conditions and calculations
  intros
  sorry

end shipping_percentage_proof_l400_400917


namespace no_valid_fill_l400_400806

def fill_possible : Prop :=
  ∃ f : Fin 16 → Fin 9,
    ∀ (t₁ t₂ : Fin 16 → Fin 3),
      (t₁ ≠ t₂) → (∑ i in Finset.univ, f (t₁ i) ≠ ∑ j in Finset.univ, f (t₂ j))

theorem no_valid_fill :
  ¬ fill_possible := 
sorry

end no_valid_fill_l400_400806


namespace number_of_primes_between_50_and_60_l400_400377

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400377


namespace largest_amount_received_back_l400_400057

-- Define the conditions
def initial_amount_chips : ℕ := 3000
def num_denominations_20_100 := 20 ∧ 100
def total_chips_lost := 13
def chips_lost_variation (x y : ℕ) := (x = y + 3 ∨ x = y - 3)

-- Define the goal
theorem largest_amount_received_back (x y : ℕ) :
  x + y = total_chips_lost →
  chips_lost_variation x y →
  initial_amount_chips - (x * 20 + y * 100) = 2340 :=
begin
  intros h1 h2,
  sorry
end

end largest_amount_received_back_l400_400057


namespace probability_is_two_thirds_l400_400152

/-- A group of 5 people consisting of 3 boys and 2 girls -/
def total_people := 5
def boys := 3
def girls := 2

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := Nat.binom n k

/-- The number of ways to choose 1 girl out of 2 -/
def ways_to_choose_one_girl := choose girls 1

/-- The number of ways to choose 2 boys out of 3 -/
def ways_to_choose_two_boys := choose boys 2

/-- The number of ways to choose 2 girls out of 2 -/
def ways_to_choose_two_girls := choose girls 2

/-- The number of ways to choose 1 boy out of 3 -/
def ways_to_choose_one_boy := choose boys 1

/-- The total number of favorable outcomes for selecting 1 girl and 2 boys -/
def favorable_outcomes := ways_to_choose_one_girl * ways_to_choose_two_boys

/-- The total number of ways to select 3 people with at least 1 girl -/
def total_outcomes := favorable_outcomes + (ways_to_choose_two_girls * ways_to_choose_one_boy)

/-- The probability of selecting 1 girl and 2 boys -/
def probability := favorable_outcomes.toFloat / total_outcomes.toFloat

theorem probability_is_two_thirds :
  probability = 2 / 3 := by
  sorry

end probability_is_two_thirds_l400_400152


namespace time_to_traverse_nth_mile_l400_400066

theorem time_to_traverse_nth_mile (n : ℕ) (n_pos : n > 1) :
  let k := (1 / 2 : ℝ)
  let s_n := k / ((n-1) * (2 ^ (n-2)))
  let t_n := 1 / s_n
  t_n = 2 * (n-1) * 2^(n-2) := 
by sorry

end time_to_traverse_nth_mile_l400_400066


namespace primes_between_50_and_60_l400_400307

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400307


namespace band_width_sqrt2_l400_400757

-- Definitions for points and distance on a plane
structure Point :=
(x : ℝ) (y : ℝ)

def distance (p₁ p₂ : Point) : ℝ :=
real.sqrt ((p₂.x - p₁.x)^2 + (p₂.y - p₁.y)^2)

-- Definition of a band with width d
def band (d : ℝ) (l : ℝ) (p₁ p₂ : Point) :=
forall (p : Point), distance p p₁ <= (d / 2) ∧ distance p p₂ <= (d / 2)

-- Condition: There exists a band with width 1 for any three points
axiom exists_band_width_1 (A B C D : Point) :
  (∃ l, band 1 l A B) ∧
  (∃ l, band 1 l A C) ∧
  (∃ l, band 1 l A D) ∧
  (∃ l, band 1 l B C) ∧
  (∃ l, band 1 l B D) ∧
  (∃ l, band 1 l C D)

-- Theorem to prove: There exists a band with width √2 that contains all four points A, B, C, D
theorem band_width_sqrt2 (A B C D : Point) :
  exists_band_width_1 A B C D →
  ∃ l, band (real.sqrt 2) l A B ∧ band (real.sqrt 2) l C D ∧ band (real.sqrt 2) l A C D :=
sorry

end band_width_sqrt2_l400_400757


namespace simplify_cube_root_2700_l400_400026

theorem simplify_cube_root_2700 : ∃ (a b : ℕ), a = 3 ∧ b = 100 ∧ a + b = 103 :=
by {
  use 3,
  use 100,
  split, 
  { refl },
  split,
  { refl },
  { rw [Nat.add_comm], refl }
}

end simplify_cube_root_2700_l400_400026


namespace area_AEFC_l400_400043

open Real EuclideanGeometry

noncomputable def Point := ℝ × ℝ   -- using cartesian coordinates for points

structure Triangle :=
  (A B C : Point)
  (right_angle : ∠ A C B = π / 2)
  (AB : dist A B = 3)
  (BC : dist B C = 4)

structure ExtendedPoint (T : Triangle) :=
  (D : Point)
  (AD : dist T.A D = 2 * 3)

structure Midpoint (T : Triangle) :=
  (E : Point)
  (midpoint_CB : dist E T.C = dist E T.B / 2)

structure IntersectPoint (T : Triangle) (E D : Point) :=
  (F : Point)
  (line_ED_inter_AC : Collinear [E, D, F] ∧ Collinear [T.A, T.C, F])

def area_quadrilateral (A E F C : Point) : ℝ := area_triangle A E F + area_triangle E F C

theorem area_AEFC (tri : Triangle) (ext : ExtendedPoint tri) (mid : Midpoint tri)
 (inter : IntersectPoint tri mid.E ext.D) :
  area_quadrilateral tri.A mid.E inter.F tri.C = 12 := sorry

end area_AEFC_l400_400043


namespace exists_x_such_that_ceil_minus_x_eq_half_l400_400729

def ceil (x : ℝ) : ℤ := int.ceil x

theorem exists_x_such_that_ceil_minus_x_eq_half : ∃ (x : ℝ), ↑(ceil x) - x = 0.5 := by
  sorry

end exists_x_such_that_ceil_minus_x_eq_half_l400_400729


namespace find_k_l400_400183

variables {a b : ℝ^3}

def unit_vector (v : ℝ^3) : Prop := v.dot v = 1

def perpendicular (v w : ℝ^3) : Prop := v.dot w = 0

noncomputable def k (a b : ℝ^3) (h_unit_a : unit_vector a) (h_unit_b : unit_vector b) (h_non_collinear : a ≠ b) (h_perpendicular : perpendicular (a + b) (a - b)) : ℝ :=
1

theorem find_k {a b : ℝ^3} (h_unit_a : unit_vector a) (h_unit_b : unit_vector b) (h_non_collinear : a ≠ b) (h_perpendicular : perpendicular (a + b) (a - b)) :
  k a b h_unit_a h_unit_b h_non_collinear h_perpendicular = 1 :=
sorry

end find_k_l400_400183


namespace primes_between_50_and_60_l400_400433

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400433


namespace primes_between_50_and_60_l400_400389

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400389


namespace soccer_ball_max_height_l400_400073

theorem soccer_ball_max_height :
  (∃ t, (∀ t', s t' ≤ s t) ∧ s t = 405) :=
by
  let s := λ t : ℝ, 180 * t - 20 * t^2
  use 4.5
  split
  sorry
  sorry

end soccer_ball_max_height_l400_400073


namespace count_primes_between_50_and_60_l400_400450

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400450


namespace num_primes_between_50_and_60_l400_400612

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400612


namespace primes_between_50_and_60_l400_400446

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400446


namespace radius_of_KN_l400_400087

-- Define the problem conditions
variables (ABCD : Type) (A B C D M N K : Point)
variables (radius_AN radius_KN : ℝ)

-- Conditions in the problem
def rectangle (ABCD : Type) [t : Rectangle ABCD] : Prop := 
  t.is_rectangle

def midpoint_M (M : Point) (A B : Point) : Prop :=
  is_midpoint M A B

def arc_AN (N : Point) (D : Point) (radius : ℝ) : Prop :=
  is_arc N D radius

def area_shaded_region (area : ℝ) (radius_AN : ℝ) (radius_KN : ℝ) : Prop :=
  area = 10 - (5 * Real.pi) / 2

-- The statement to be proved
theorem radius_of_KN (h1 : rectangle ABCD) 
                     (h2 : midpoint_M M A B) 
                     (h3 : arc_AN N D 3) 
                     (h4 : area_shaded_region 10 - (5 * Real.pi) / 2 3 radius_KN) : 
                     radius_KN = 1 :=
sorry

end radius_of_KN_l400_400087


namespace system_solution_l400_400790

theorem system_solution (x y : ℝ) : 
  (3^y * 81 = 9^(x^2)) ∧ (Real.log10 y = Real.log10 x - Real.log10 0.5) → 
  x = 2 ∧ y = 4 :=
by
  sorry

end system_solution_l400_400790


namespace fraction_repetend_l400_400122

-- Lean statement to prove that the repetend of the fraction \frac{7}{29} is "241379".

theorem fraction_repetend (n d : ℕ) (h : d ≠ 0) (num : n = 7) (den : d = 29) :
  (decimal_repetend n d h) = "241379" := by
  sorry

end fraction_repetend_l400_400122


namespace number_of_mappings_number_of_mappings_with_condition_l400_400731

-- Define the sets A and B and the function f
def A : Set := {a, b, c, d}
def B : Set := {0, 1, 2}

noncomputable def f : A → B

-- Define the statement proving the number of mappings from A to B
theorem number_of_mappings : (fin (cardinal.mk A) ^ (cardinal.mk B)).to_nat = 81 :=
by
  -- here the steps would be detailed in the proof, but we are assuming the result is correct
  sorry

-- Define a condition for the second theorem
def sum_condition (f : A → B) := (f a + f b + f c + f d = 4)

-- Define the statement proving the number of mappings that satisfy the condition
theorem number_of_mappings_with_condition : 
  (card {f : (A → B) // sum_condition f}).to_nat = 19 :=
by
  -- here the steps would be detailed in the proof, but we are assuming the result is correct
  sorry

end number_of_mappings_number_of_mappings_with_condition_l400_400731


namespace average_of_original_numbers_is_2013_l400_400912

theorem average_of_original_numbers_is_2013 
  (a : ℚ) 
  (h_sum : ∑ i in (finRange 2012), a = a * 2012)
  (h_avg_new_2013 : (a * 2012 + a) / 2013 = 2013) : 
  a = 2013 := 
sorry

end average_of_original_numbers_is_2013_l400_400912


namespace sum_of_angles_of_pentagon_and_triangle_l400_400721

-- Conditions
def is_polygon (n : ℕ) : Prop := n ≥ 3

def interior_angle (n : ℕ) [fact (3 ≤ n)] : ℝ := 180 * (n - 2) / n

-- Problem statement
theorem sum_of_angles_of_pentagon_and_triangle :
  ∑ (n = 5) (interior_angle(5)) + ∑ n = 3 (interior_angle(3)) = 168 :=
by sorry

end sum_of_angles_of_pentagon_and_triangle_l400_400721


namespace prime_count_between_50_and_60_l400_400332

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400332


namespace primes_between_50_and_60_l400_400588

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400588


namespace primes_between_50_and_60_l400_400520

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400520


namespace parabola_eq_exists_minimum_area_triangle_l400_400165

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y
noncomputable def circle (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 9 / 4
noncomputable def is_on_parabola (a b : ℝ) : Prop := a^2 = 2 * p * b
noncomputable def passes_through_origin (a b : ℝ) : Prop := (0 - a)^2 + (0 - b)^2 = 9 / 4
noncomputable def tangent_to_directrix (a b p : ℝ) : Prop := -- Define tangent condition here if necessary

theorem parabola_eq_exists
  (a b p : ℝ)
  (h1 : is_on_parabola a b)
  (h2 : passes_through_origin a b)
  (h3 : tangent_to_directrix a b p) :
  ∃ (x y : ℝ), parabola x y :=
begin
  sorry
end

noncomputable def minimum_area (a b p : ℝ) : ℝ := -- Define minimum area calculation here if necessary
noncomputable def line_eq (x : ℝ) : ℝ := -sqrt 6/3 * x + sqrt 2

theorem minimum_area_triangle
  (a b p : ℝ)
  (h1 : is_on_parabola a b)
  (h2 : passes_through_origin a b)
  (h3 : tangent_to_directrix a b p) :
  minimum_area a b p = 9 * sqrt 3 / 4 ∧ line_eq = (λ x, -√6/3 * x + √2) :=
begin
  sorry
end

end parabola_eq_exists_minimum_area_triangle_l400_400165


namespace minutes_before_4_angle_same_as_4_l400_400214

def hour_hand_angle_at_4 := 120
def minute_hand_angle_at_4 := 0
def minute_hand_angle_per_minute := 6
def hour_hand_angle_per_minute := 0.5

theorem minutes_before_4_angle_same_as_4 :
  ∃ m : ℚ, abs (hour_hand_angle_at_4 - 5.5 * m) = hour_hand_angle_at_4 ∧ 
           (60 - m) = 21 + 9 / 11 := by
  sorry

end minutes_before_4_angle_same_as_4_l400_400214


namespace lions_seen_l400_400921

-- Definitions based on conditions
def chimps_seen : ℕ := 12
def lizards_seen : ℕ := 5
def tarantulas_seen : ℕ := 125
def total_legs_goal : ℕ := 1100

-- Assumption about the number of legs each animal type has
def legs_per_chimp : ℕ := 4
def legs_per_lizard : ℕ := 4
def legs_per_tarantula : ℕ := 8
def legs_per_lion : ℕ := 4

theorem lions_seen (chimps_seen = 12) (lizards_seen = 5) (tarantulas_seen = 125) (total_legs_goal = 1100) :
  let legs_from_chimps := chimps_seen * legs_per_chimp
  let legs_from_lizards := lizards_seen * legs_per_lizard
  let legs_from_tarantulas := tarantulas_seen * legs_per_tarantula
  let total_legs_seen := legs_from_chimps + legs_from_lizards + legs_from_tarantulas
  let legs_needed := total_legs_goal - total_legs_seen
  let lions_seen := legs_needed / legs_per_lion
  lions_seen = 8 :=
by
  -- Placeholder to skip the actual proof
  sorry

end lions_seen_l400_400921


namespace count_primes_between_50_and_60_l400_400656

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400656


namespace count_primes_50_60_l400_400476

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400476


namespace primes_between_50_and_60_l400_400399

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400399


namespace primes_between_50_and_60_l400_400419

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400419


namespace hyperbola_through_focus_and_asymptotes_l400_400805

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 1

def asymptotes_holds (x y : ℝ) : Prop :=
  (x + y = 0) ∨ (x - y = 0)

theorem hyperbola_through_focus_and_asymptotes :
  hyperbola parabola_focus.1 parabola_focus.2 ∧ asymptotes_holds parabola_focus.1 parabola_focus.2 :=
sorry

end hyperbola_through_focus_and_asymptotes_l400_400805


namespace count_primes_between_50_and_60_l400_400448

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400448


namespace primes_between_50_and_60_l400_400521

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400521


namespace primes_between_50_and_60_l400_400318

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400318


namespace find_positive_integers_l400_400964

theorem find_positive_integers 
    (a b : ℕ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (h1 : ∃ k1 : ℤ, (a^3 * b - 1) = k1 * (a + 1))
    (h2 : ∃ k2 : ℤ, (b^3 * a + 1) = k2 * (b - 1)) : 
    (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
sorry

end find_positive_integers_l400_400964


namespace fruit_eating_orders_l400_400106

theorem fruit_eating_orders:
  let apples := 4
  let oranges := 3
  let bananas := 2
  let days := 7
  ∀ (orderings : List (List ℕ)) (valid_ordering : List ℕ → Prop),
  (∀ order ∈ orderings, valid_ordering order) →
  (valid_ordering = λ order, 
    order.length = days ∧ 
    (∀ i, i < days → order.nth i ≠ some 0 ∨ 
    (∃ j, j ≠ i ∧ j < days ∧ order.nth j ≠ some 0 ∧ order.nth j ≠ some 1))) → 
  (orderings.length = 150) :=
begin
  intros orderings valid_ordering valid_property valid_spec,
  sorry,
end

end fruit_eating_orders_l400_400106


namespace count_primes_between_50_and_60_l400_400248

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400248


namespace primes_between_50_and_60_l400_400444

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400444


namespace primes_between_50_and_60_l400_400401

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400401


namespace part_a_part_b_part_c_part_d_l400_400036

-- Part (a)
axiom convex_polyhedron (P : Type) : Prop
axiom diameter (P : Type) : ℝ
axiom edge_lengths_sum (P : Type) : ℝ

theorem part_a (P : Type) [convex_polyhedron P] :
  edge_lengths_sum P > 3 * diameter P :=
sorry

-- Part (b)
variables {P : Type} [convex_polyhedron P]
axiom vertex (P : Type) : Type
axiom edge (P : Type) : Type

def path (P : Type) (A B : vertex P) : Type := { l : list (edge P) // -- properties of the path -- }

theorem part_b (A B : vertex P) :
  ∃ p₁ p₂ p₃ : path P A B, 
  (∀ (e : edge P), e ∈ p₁.val → e ∉ p₂.val ∧ e ∉ p₃.val) ∧
  (∀ (e : edge P), e ∈ p₂.val → e ∉ p₃.val) :=
sorry

-- Part (c)
theorem part_c (A B : vertex P) (e₁ e₂ : edge P):
  ∃ p : path P A B,
  e₁ ∉ p.val ∧ e₂ ∉ p.val :=
sorry

-- Part (d)
theorem part_d (A B : vertex P) :
  ∃ p₁ p₂ p₃ : path P A B, 
  (∀ (v : vertex P), v ≠ A ∧ v ≠ B → v ∉ p₁.val ∧ v ∉ p₂.val ∧ v ∉ p₃.val) :=
sorry

end part_a_part_b_part_c_part_d_l400_400036


namespace complementary_event_l400_400817

theorem complementary_event (A : Prop) (B : Prop) 
  [decidable A] [decidable B]
  (missing_both := ¬A ∧ ¬B)
  (hitting_at_least_one := A ∨ B) :
  (missing_both → ¬hitting_at_least_one) ∧ (¬missing_both → hitting_at_least_one) :=
by
  sorry

end complementary_event_l400_400817


namespace primes_between_50_and_60_l400_400437

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400437


namespace primes_between_50_and_60_l400_400409

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400409


namespace count_perfect_squares_with_desired_last_digit_l400_400216

-- Define the condition to check the last digit of a number
def has_desired_last_digit (n : ℕ) : Prop :=
  let last_digit := n % 10
  last_digit = 1 ∨ last_digit = 2 ∨ last_digit = 4

-- Define the condition for being a perfect square less than 5000
def is_perfect_square_less_than_5000 (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m ∧ m < 71  -- Since ⌊sqrt(5000)⌋ = 70

-- Define the target set to count
def desired_squares : set ℕ :=
  { n | is_perfect_square_less_than_5000 n ∧ has_desired_last_digit n }

-- The statement to prove
theorem count_perfect_squares_with_desired_last_digit :
  (desired_squares : set ℕ).to_finset.card = 28 :=
begin
  sorry
end

end count_perfect_squares_with_desired_last_digit_l400_400216


namespace primes_between_50_and_60_l400_400379

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400379


namespace probability_sum_even_is_11_div_21_l400_400974

open_locale classical
noncomputable theory

def set_of_numbers : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}.to_finset
def num_ways_choose_3 : ℕ := (set_of_numbers.card.choose 3).to_nat

def num_even : ℕ := (set_of_numbers.filter even).card
def num_odd : ℕ := (set_of_numbers.filter (λ x, ¬ even x)).card

def num_event_A : ℕ :=
  (num_even.choose 3).to_nat + ((num_even.choose 1).to_nat * (num_odd.choose 2).to_nat)

def probability_A : ℚ := (num_event_A : ℚ) / (num_ways_choose_3 : ℚ)

theorem probability_sum_even_is_11_div_21 :
  probability_A = 11 / 21 :=
begin
  sorry,
end

end probability_sum_even_is_11_div_21_l400_400974


namespace total_sum_of_k_p_l400_400184

def sum_of_indices (n : ℕ) (h1 : n ≡ 1 [MOD 4]) (h2 : n > 1) : ℕ :=
  (1 / 2 * (n - 1) * (Nat.factorial n)).to_nat -- representing the correct answer

theorem total_sum_of_k_p (n : ℕ) (h1 : n ≡ 1 [MOD 4]) (h2 : n > 1) :
  ∑ P in Finset.univ, k_p P = sum_of_indices n h1 h2 :=
sorry

end total_sum_of_k_p_l400_400184


namespace employed_females_are_28_125_percent_of_employed_l400_400870

-- Define the given conditions as constants
def population : ℝ := 100
def employed_percent : ℝ := 64
def employed_male_percent : ℝ := 46

-- Define employed female percent based on conditions
def employed_female_percent : ℝ := employed_percent - employed_male_percent

-- Define proportion of employed females out of all employed people
def employed_female_proportion : ℝ := (employed_female_percent / employed_percent) * 100

-- Define expected answer
def expected_proportion : ℝ := 28.125

-- The final theorem to state
theorem employed_females_are_28_125_percent_of_employed :
  employed_female_proportion = expected_proportion :=
by
  sorry

end employed_females_are_28_125_percent_of_employed_l400_400870


namespace cost_of_refrigerator_l400_400764

constants (R : ℝ) (cost_mobile: ℝ) (loss_refrigerator: ℝ) (profit_mobile: ℝ) (overall_profit: ℝ)

-- Definitions based on the conditions
def cost_mobile := 8000
def loss_refrigerator := 0.04 * R
def profit_mobile := 0.10 * 8000
def overall_profit := 200

-- Given these conditions, prove that the cost of the refrigerator R is 25000
theorem cost_of_refrigerator (R : ℝ) (h1 : cost_mobile = 8000) (h2 : loss_refrigerator = 0.04 * R) 
  (h3 : profit_mobile = 0.10 * 8000) (h4 : overall_profit = 200) : 
  R = 25000 := 
by 
  sorry

end cost_of_refrigerator_l400_400764


namespace sequence_recurrent_sum_floor_equals_two_l400_400041

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n + 1) + sequence (n - 1)

theorem sequence_recurrent (n : ℕ) : sequence (n + 1) = (n + 1) + sequence n :=
by
  sorry

theorem sum_floor_equals_two : (Real.floor (Finset.sum (Finset.range 2018) (λ k : ℕ, (1 / sequence (k + 1) : ℝ)))) = 2 :=
by
  sorry

end sequence_recurrent_sum_floor_equals_two_l400_400041


namespace count_primes_between_50_and_60_l400_400654

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400654


namespace num_primes_50_60_l400_400279

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400279


namespace find_line_equation_l400_400130

variable (x y : ℝ)

theorem find_line_equation (hx : x = -5) (hy : y = 2)
  (line_through_point : ∃ a b c : ℝ, a * x + b * y + c = 0)
  (x_intercept_twice_y_intercept : ∀ a b c : ℝ, c ≠ 0 → b ≠ 0 → (a / c) = 2 * (c / b)) :
  ∃ a b c : ℝ, (a * x + b * y + c = 0 ∧ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 1 ∧ b = 2 ∧ c = 1)) :=
sorry

end find_line_equation_l400_400130


namespace unique_rhombus_property_l400_400821

-- Definitions of properties
structure Rhombus where
  sides_equal_and_parallel : ∀ (a b : ℝ), a = b → parallel a b
  opposite_angles_equal : ∀ (A B : ℝ), A = B
  diagonals_perpendicular : ∀ (d₁ d₂ : ℝ), perpendicular d₁ d₂
  diagonals_bisect_angles : ∀ (d₁ d₂ : ℝ), bisects_angles d₁ d₂

structure Parallelogram where
  sides_equal_and_parallel : ∀ (a b : ℝ), a = b → parallel a b
  opposite_angles_equal : ∀ (A B : ℝ), A = B
  diagonals_bisect_each_other : ∀ (d₁ d₂ : ℝ), bisect_each_other d₁ d₂

theorem unique_rhombus_property (r : Rhombus) (p : Parallelogram) : 
  (∀ d₁ d₂, bisects_angles d₁ d₂) ∧ ¬ (∀ d₁ d₂, bisects_angles d₁ d₂ → bisect_each_other d₁ d₂) :=
by
  sorry

end unique_rhombus_property_l400_400821


namespace derivative_condition_l400_400734

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem derivative_condition (x₀ : ℝ) (h : (deriv f x₀) = 3 / 2) : x₀ = Real.sqrt Real.exp := 
by
  sorry

end derivative_condition_l400_400734


namespace smallest_prime_factor_of_1471_l400_400857

theorem smallest_prime_factor_of_1471 : 
  (¬ (1471 % 2 = 0)) ∧ 
  (¬ (1471 % 3 = 0)) ∧ 
  (¬ (1471 % 5 = 0)) ∧ 
  (¬ (1471 % 7 = 0)) ∧ 
  (¬ (1471 % 11 = 0)) ∧ 
  (1471 % 13 = 0) → 
  ∃ p : ℕ, (Nat.prime p) ∧ (p ∣ 1471) ∧ (∀ q : ℕ, (Nat.prime q) → (q ∣ 1471) → p ≤ q) :=
by
  sorry

end smallest_prime_factor_of_1471_l400_400857


namespace primes_count_between_50_and_60_l400_400543

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400543


namespace tax_rate_percentage_l400_400694

/-- In Township K, where property values and taxes are given, prove the tax rate. -/
theorem tax_rate_percentage (r : ℝ) (h1 : 20000 * r : ℝ) (h2 : 28000 * r : ℝ) (h3 : 28000 * r - 20000 * r = 800):
  r * 100 = 10 :=
by
  sorry

end tax_rate_percentage_l400_400694


namespace expected_score_for_A_probability_of_A_and_B_selected_l400_400884

noncomputable theory

-- Definitions and conditions for the problem
def prob_A_correct := 3/5
def questions_total := 10
def prob_B_correct := 5 / 10 / (5 / 10 * 4 / 9 * 3 / 8)

def expected_score_A (n : ℕ) (p : ℚ) : ℚ :=
  let E_X := n * p
  E_X * 10 - (n - E_X) * 5

def prob_A_selected (correct_prob : ℚ) : ℚ :=
(3.choose 2) * (correct_prob ^ 2) * (1 - correct_prob) + (correct_prob ^ 3)

-- The expected score for A
theorem expected_score_for_A : expected_score_A 3 prob_A_correct = 12 :=
by sorry

-- The probability of both A and B being selected
theorem probability_of_A_and_B_selected (prob_B : ℚ) : 
  prob_A_selected prob_A_correct * prob_B = 81 / 250 :=
by sorry

end expected_score_for_A_probability_of_A_and_B_selected_l400_400884


namespace count_primes_between_50_and_60_l400_400452

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400452


namespace evaluate_expression_l400_400933

def S (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n + 1) / 2
  else -n / 2

theorem evaluate_expression : S 19 * S 31 + S 48 = 136 :=
by sorry

end evaluate_expression_l400_400933


namespace prime_count_between_50_and_60_l400_400353

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400353


namespace train_length_l400_400077

theorem train_length (speed_kmph : ℕ) (bridge_length : ℕ) (time_seconds : ℕ) : bridge_length = 255 ∧ speed_kmph = 45 ∧ time_seconds = 30 → train_length = 120 :=
by
  -- Declare the variables based on the conditions
  let speed_mps : ℝ := (speed_kmph : ℝ) * (1000 / 1) / 3600
  let total_distance : ℝ := speed_mps * time_seconds
  let train_length : ℝ := total_distance - bridge_length
  -- Use Lean's features to enforce the conditions and prove the required length
  sorry

end train_length_l400_400077


namespace sqrt_sine_tan_domain_l400_400127

open Real

noncomputable def domain_sqrt_sine_tan : Set ℝ :=
  {x | ∃ (k : ℤ), (-π / 2 + 2 * k * π < x ∧ x < π / 2 + 2 * k * π) ∨ x = k * π}

theorem sqrt_sine_tan_domain (x : ℝ) :
  (sin x * tan x ≥ 0) ↔ x ∈ domain_sqrt_sine_tan :=
by
  sorry

end sqrt_sine_tan_domain_l400_400127


namespace fuzhou_2014_quality_inspection_l400_400706

noncomputable theory
open_locale classical

variables (α : set ℝ) (m n : set ℝ)

def coplanar (m n : set ℝ) (α : set ℝ) : Prop :=
  ∃ p q r, p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ p ∈ α ∧ q ∈ α ∧ r ∈ α ∧ m ⊆ α ∧ n ⊆ α

-- Statement A
def statement_A (α m n : set ℝ) : Prop :=
  (∀ θ_m θ_n, θ_m = θ_m ∧ θ_n = θ_n) → (m = n ∨ m ∩ n = ∅)

-- Statement B
def statement_B (α m n : set ℝ) : Prop :=
  m ∥ α ∧ n ∥ α → m ∥ n

-- Statement C
def statement_C (α m n : set ℝ) : Prop :=
  m ⟂ α ∧ m ⟂ n → n ∥ α

-- Statement D
def statement_D (α m n : set ℝ) : Prop :=
  m ⊆ α ∧ n ∥ α → m ∥ n

theorem fuzhou_2014_quality_inspection (α : set ℝ) (m n : set ℝ)
  (h_coplanar : coplanar m n α) : statement_D α m n :=
begin
  sorry
end

end fuzhou_2014_quality_inspection_l400_400706


namespace number_of_primes_between_50_and_60_l400_400373

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400373


namespace identify_false_statement_l400_400031

-- Definitions for given conditions
def vertical_angles_equal (V_A V_B : Prop) : Prop := V_A = V_B

def perpendicular_to_two_lines_implies_parallel (a b c : Prop) : Prop :=
  (a ⊥ b) ∧ (a ⊥ c) → ¬(b ∥ c)

def unique_parallel_line (l m : Prop) (P : Prop) : Prop :=
  P ∉ l → ∃! (k : Prop), k ∥ l ∧ P ∈ k

def sqrt_equality (a b : ℝ) : Prop :=
  a = b → (0 ≤ a ∧ 0 ≤ b) → sqrt a = sqrt b

-- Main theorem to identify the false statement
theorem identify_false_statement (V_A V_B a b c l m P : Prop) (cond1 : vertical_angles_equal V_A V_B)
  (cond2 : perpendicular_to_two_lines_implies_parallel a b c)
  (cond3 : unique_parallel_line l m P)
  (cond4 : ∀ (a b : ℝ), a = b → (0 ≤ a ∧ 0 ≤ b) → sqrt a = sqrt b) :
  (∃ (S : Prop), S = cond2) :=
sorry

end identify_false_statement_l400_400031


namespace count_primes_between_50_and_60_l400_400249

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400249


namespace primes_between_50_and_60_l400_400398

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400398


namespace circle_symmetric_about_line_l400_400802

-- The main proof statement
theorem circle_symmetric_about_line (x y : ℝ) (k : ℝ) :
  (x - 1)^2 + (y - 1)^2 = 2 ∧ y = k * x + 3 → k = -2 :=
by
  sorry

end circle_symmetric_about_line_l400_400802


namespace condition_sufficient_but_not_necessary_l400_400678

theorem condition_sufficient_but_not_necessary (a : ℝ) : (a > 9 → (1 / a < 1 / 9)) ∧ ¬(1 / a < 1 / 9 → a > 9) :=
by 
  sorry

end condition_sufficient_but_not_necessary_l400_400678


namespace initial_amount_correct_l400_400008

-- Definitions
def spent_on_meat := 17
def spent_on_chicken := 22
def spent_on_veggies := 43
def spent_on_eggs := 5
def spent_on_dogs_food := 45
def spent_on_cats_food_before_discount := 18
def discount_rate := 0.10
def amount_left := 35

-- Total spent before discount
def total_spent_before_discount := 
  spent_on_meat + spent_on_chicken + spent_on_veggies + spent_on_eggs + spent_on_dogs_food + spent_on_cats_food_before_discount

-- Discount amount on cat's food
def discount_amount := discount_rate * spent_on_cats_food_before_discount

-- Total spent after discount
def total_spent_after_discount := total_spent_before_discount - discount_amount

-- Initial amount of money Trisha brought
def initial_amount := total_spent_after_discount + amount_left

-- Goal: prove the initial amount is $183.20
theorem initial_amount_correct : initial_amount = 183.20 := by
  sorry

end initial_amount_correct_l400_400008


namespace count_primes_between_50_and_60_l400_400244

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400244


namespace primes_between_50_and_60_l400_400299

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400299


namespace garden_perimeter_is_correct_l400_400872

-- Definitions of the problem conditions
def playground_length : ℕ := 16
def playground_width : ℕ := 12
def garden_width : ℕ := 16
def playground_area : ℕ := playground_length * playground_width

-- The goal is to prove
theorem garden_perimeter_is_correct :
  ∃ (garden_length : ℕ), (garden_length * garden_width = playground_area) → 
  let garden_perimeter := 2 * garden_length + 2 * garden_width in 
  garden_perimeter = 56 :=
by 
  sorry

end garden_perimeter_is_correct_l400_400872


namespace primes_between_50_and_60_l400_400640

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400640


namespace remainder_when_160_divided_by_k_l400_400144

-- Define k to be a positive integer
def positive_integer (n : ℕ) := n > 0

-- Given conditions in the problem
def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def problem_condition (k : ℕ) := positive_integer k ∧ (120 % (k * k) = 12)

-- Prove the main statement
theorem remainder_when_160_divided_by_k (k : ℕ) (h : problem_condition k) : 160 % k = 4 := 
sorry  -- Proof here

end remainder_when_160_divided_by_k_l400_400144


namespace find_b_for_perpendicular_lines_l400_400967

noncomputable def line1 := λ x : ℝ, -3 * x + 7
noncomputable def line2 := λ x y : ℝ, 4 * y + b * x = 12

theorem find_b_for_perpendicular_lines (b : ℝ) :
  (∀ x : ℝ, line2 x ((-3 * x + 7) / 4) = 3) → b = -4 / 3 :=
by
  assume key_cond : ∀ x : ℝ, 4 * (-3 * x + 7) / 4 + b * x = 12 
  sorry

end find_b_for_perpendicular_lines_l400_400967


namespace min_perimeter_triangle_l400_400168

theorem min_perimeter_triangle 
  (α β : Plane) 
  (P : Point)
  (d_alpha d_beta : ℝ) 
  (angle_αlβ : ℝ)
  (h_dist_alpha : dist_to_plane P α = d_alpha)
  (h_dist_beta : dist_to_plane P β = d_beta)
  (h_angle_αlβ : angle_αlβ = 60)
  (d_alpha_eq_3 : d_alpha = 3)
  (d_beta_eq_5 : d_beta = 5) : 
  (∃ A B : Point, A ∈ α ∧ B ∈ β ∧ perimeter_triangle P A B = 14) :=
sorry

end min_perimeter_triangle_l400_400168


namespace proof_centroid_orthocenter_circumcenter_l400_400206

noncomputable def centroid : ℝ × ℝ := (16/3, 8/3)
noncomputable def orthocenter : ℝ × ℝ := (6, 3)
noncomputable def circumcenter : ℝ × ℝ := (5, 5/2)

def are_collinear (G H C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := G
  let (x₂, y₂) := H
  let (x₃, y₃) := C
  x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂) = 0

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem proof_centroid_orthocenter_circumcenter :
  are_collinear centroid orthocenter circumcenter ∧ 
  distance circumcenter orthocenter = 3 * distance centroid circumcenter :=
by
  sorry

end proof_centroid_orthocenter_circumcenter_l400_400206


namespace particle_traversal_time_l400_400901

theorem particle_traversal_time (n : ℕ) (h1 : n ≥ 3) (h2 : (∀ m ≥ 2, ∃ k, (m != 3 → sorry), (s m = k / (m-1)^2))) (h3 : ∃ k, (s 3 = 1/4)) : 
  time n = (n-1)^2 :=
by sorry

end particle_traversal_time_l400_400901


namespace max_real_roots_polynomial_l400_400133

noncomputable
def max_real_roots (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else 0

theorem max_real_roots_polynomial (n : ℕ) (h : n > 0) :
  max_real_roots n = 2 ↔ (∃ (x : ℝ), x^n - x^(n-1) + x^(n-2) - ... + (-1)^(n-1)*x + 1 = 0) ∧ (n % 2 = 0) :=
sorry

end max_real_roots_polynomial_l400_400133


namespace series_sum_approx_l400_400928

noncomputable def alternating_series_sum : ℝ :=
  ∑' n, (-1)^(n+1) / n^5

def partial_sum (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (-1)^(i+1 : ℕ) / (i+1 : ℕ)^5

theorem series_sum_approx : 
  | partial_sum 3 - 0.973 | < 10^(-3) := 
by 
  sorry

end series_sum_approx_l400_400928


namespace minimum_nonzero_area_l400_400876

-- Definitions of points A, B and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (42, 18)
def C (p q : ℤ) : ℝ × ℝ := (p, q)

-- Shoelace formula area computation
def shoelace_area (p q : ℤ) : ℝ := 
  21 * (Real.abs ((q : ℝ) - 18))

-- The minimum area of the triangle ABC
theorem minimum_nonzero_area : 
  (∃ p q : ℤ, (shoelace_area p q) = 21) :=
by
  sorry

end minimum_nonzero_area_l400_400876


namespace factor_expression_l400_400948

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) :=
by
  sorry

end factor_expression_l400_400948


namespace possible_values_of_n_in_circle_l400_400831

theorem possible_values_of_n_in_circle (n : ℕ) (a : ℤ) (m : ℕ) (int_list : List ℤ)
  (h1 : int_list.length = n)
  (h2 : int_list.sum = 94)
  (h3 : ∀ i, int_list.nth i % n = | int_list.nth (i+1) % n - int_list.nth (i+2) % n |) :
  n = 3 ∨ n = 141 :=
by
  sorry

end possible_values_of_n_in_circle_l400_400831


namespace parallel_lines_value_of_a_l400_400208

theorem parallel_lines_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, ax + (a+2)*y + 2 = 0 → x + a*y + 1 = 0 → ∀ m n : ℝ, ax + (a + 2)*n + 2 = 0 → x + a*n + 1 = 0) →
  a = -1 := 
sorry

end parallel_lines_value_of_a_l400_400208


namespace max_n_for_positive_sum_l400_400173

theorem max_n_for_positive_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) 
  (h_sum : S = λ n, (n * (a 1 + a n)) / 2) 
  (h_root1 : has_root (polynomial.of_eq (x ^ 2 - 2012 * x - 2011)) (a 1006))
  (h_root2 : has_root (polynomial.of_eq (x ^ 2 - 2012 * x - 2011)) (a 1007)) 
  : 2011 = max n, S 2011 > 0 := sorry

end max_n_for_positive_sum_l400_400173


namespace num_intersections_circle_line_eq_two_l400_400816

theorem num_intersections_circle_line_eq_two :
  ∃ (points : Finset (ℝ × ℝ)), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25 ∧ p.1 = 3} = points ∧ points.card = 2 :=
by
  sorry

end num_intersections_circle_line_eq_two_l400_400816


namespace point_H_lies_on_line_B_l400_400051

noncomputable def Point : Type := sorry
noncomputable def Line : Type := sorry

variables {A B C I M H B' C' : Point}
variables (ω : Point → Prop)
variables (midpoint : Point → Point → Point)
variables (perpendicular : Point → Line → Line)
variables (triangle_bound : Line → Point → Point → (Point → Prop))
variables (circumcircle : (Point → Prop) → Line → Prop)
variables (in_circle : Point → (Point → Prop) → Prop)
variables (incenter : Point → Point → Point → Point)
variables (circumcenter : Point → Point → Point → Point)
variables (radical_axis : (Point → Prop) → (Point → Prop) → Line)
variables (collinear : Point → Point → Point → Prop)

-- Given Conditions:
axiom circumscribed (hω : ω A) (hω : ω B) (hω : ω C) : circumcircle ω (line_through A B C)
axiom AB_lt_AC : A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ AB < AC
axiom angle_bisectors_intersection (hI : I = incenter A B C)
axiom midpoint_of_BC (hM : M = midpoint B C)
axiom perpendicular_to_AI (hMH : ∀ p, in_circle p (triangle M A H) → perpendicular M (line_through A I) (line_through M p))
axiom triangles_defined (hTb : ∀ p, in_circle p (triangle_bound (line_through M H) B AB))
axiom triangles_defined2 (hTc : ∀ p, in_circle p (triangle_bound (line_through M H) C AC))
axiom circumcircles_intersect (h_intersB_B : ∃ P, in_circle P (circumcircle hTb (line_through B I)) ∧ in_circle P ω ∧ B' = P)
axiom circumcircles_intersect (h_intersC_C : ∃ P, in_circle P (circumcircle hTc (line_through C I)) ∧ in_circle P ω ∧ C' = P)

-- Theorem to prove:
theorem point_H_lies_on_line_B'C' :
  collinear H B' C' := sorry

end point_H_lies_on_line_B_l400_400051


namespace primes_between_50_and_60_l400_400402

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400402


namespace primes_count_between_50_and_60_l400_400549

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400549


namespace count_primes_between_50_and_60_l400_400458

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400458


namespace prime_count_between_50_and_60_l400_400339

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400339


namespace primes_between_50_and_60_l400_400382

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400382


namespace factorization_of_expression_l400_400119

theorem factorization_of_expression (x y : ℝ) : x^2 - x * y = x * (x - y) := 
by
  sorry

end factorization_of_expression_l400_400119


namespace least_sum_of_exponents_l400_400960

theorem least_sum_of_exponents (A : ℕ) (hA : A = 2023) :
  ∃ (S : Finset ℕ), (∑ i in S, 2^i = A) ∧ (∑ i in S, i = 52) ∧ (∀ x y ∈ S, x < y → x + y < 10) :=
by
  have h2023_binary : 2023 = 2^10 + 2^9 + 2^8 + 2^7 + 2^6 + 2^5 + 2^4 + 2^2 + 2^1 := by sorry
  use {1, 2, 4, 5, 6, 7, 8, 9, 10}
  split
  norm_num at h2023_binary
  exact h2023_binary
  split
  norm_num
  sorry
  intros x y hx hy hxy
  norm_num at hx hy
  sorry

end least_sum_of_exponents_l400_400960


namespace primes_count_between_50_and_60_l400_400546

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400546


namespace count_primes_between_50_and_60_l400_400469

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400469


namespace primes_between_50_and_60_l400_400589

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400589


namespace count_primes_50_60_l400_400473

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400473


namespace probability_ECE1_l400_400713

-- Define the conditions
def is_vowel (ch : Char) : Prop :=
  ch = 'A' ∨ ch = 'E' ∨ ch = 'I'

def is_consonant (ch : Char) : Prop :=
  (ch >= 'A' ∧ ch <= 'Z') ∧ ¬is_vowel ch ∧ (ch ≠ 'Y')

def is_digit (ch : Char) : Prop :=
  ch >= '0' ∧ ch <= '9'

def valid_license_plate (s : String) : Prop :=
  s.length = 4 ∧
  is_vowel s[0] ∧
  is_consonant s[1] ∧
  is_vowel s[2] ∧
  is_digit s[3]

-- Define the problem statement
theorem probability_ECE1 :
  (∃ (s : String), valid_license_plate s ∧ s = "ECE1") →
  (∃ (N : Nat), N = 1890 ∧ (1 : ℚ) / N = 1 / 1890) :=
by sorry

end probability_ECE1_l400_400713


namespace nesting_doll_height_l400_400750

variable (H₀ : ℝ) (n : ℕ)

theorem nesting_doll_height (H₀ : ℝ) (Hₙ : ℝ) (H₁ : H₀ = 243) (H₂ : ∀ n : ℕ, Hₙ = H₀ * (2 / 3) ^ n) (H₃ : Hₙ = 32) : n = 4 :=
by
  sorry

end nesting_doll_height_l400_400750


namespace determine_event_C_l400_400833

variable (A B C : Prop)
variable (Tallest Shortest : Prop)
variable (Running LongJump ShotPut : Prop)

variables (part_A_Running part_A_LongJump part_A_ShotPut
           part_B_Running part_B_LongJump part_B_ShotPut
           part_C_Running part_C_LongJump part_C_ShotPut : Prop)

variable (not_tallest_A : ¬Tallest → A)
variable (not_tallest_ShotPut : Tallest → ¬ShotPut)
variable (shortest_LongJump : Shortest → LongJump)
variable (not_shortest_B : ¬Shortest → B)
variable (not_running_B : ¬Running → B)

theorem determine_event_C :
  (¬Tallest → A) →
  (Tallest → ¬ShotPut) →
  (Shortest → LongJump) →
  (¬Shortest → B) →
  (¬Running → B) →
  part_C_Running :=
by
  intros h1 h2 h3 h4 h5
  sorry

end determine_event_C_l400_400833


namespace correct_statements_l400_400179

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 2 then log (x + 1) / log 2 
  else if 2 < x ∧ x ≤ 6 then - log (x - 3) / log 2
  else if -2 ≤ x ∧ x < 0 then - log (-x + 1) / log 2 
  else if -6 ≤ x ∧ x < -2 then log (-x - 3) / log 2 
  else 0

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic (-8 ≦ x + 8) (x : ℝ) : f (x + 4) = - f x

theorem correct_statements :
  (f 3 = 1) ∧ (∀ (m : ℝ), 0 < m ∧ m < 1 → ∑ x in (set.filter (λ x, f(x) = m) (set.Icc (-8 : ℝ) (16))), x = 12) := sorry

end correct_statements_l400_400179


namespace trigonometric_relation_l400_400997

theorem trigonometric_relation (α β : ℝ) (h : (sin α)^2 / (cos β)^2 + (cos α)^2 / (sin β)^2 = 4) :
  (cos β)^2 / (sin α)^2 + (sin β)^2 / (cos α)^2 = -1 := 
by
  sorry

end trigonometric_relation_l400_400997


namespace miles_from_second_friend_to_work_l400_400940
variable (distance_to_first_friend := 8)
variable (distance_to_second_friend := distance_to_first_friend / 2)
variable (total_distance_to_second_friend := distance_to_first_friend + distance_to_second_friend)
variable (distance_to_work := 3 * total_distance_to_second_friend)

theorem miles_from_second_friend_to_work :
  distance_to_work = 36 := 
by
  sorry

end miles_from_second_friend_to_work_l400_400940


namespace count_primes_50_60_l400_400472

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400472


namespace primes_between_50_and_60_l400_400314

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400314


namespace intersection_proof_l400_400707

-- Define the polar equation of circle and the parametric equation of the line
def polar_circle_equation : ∀ (ρ θ : ℝ), ρ = 6 * Real.sin θ := sorry

def parametric_line_equation (t : ℝ) : (ℝ × ℝ) := (1 + (Real.sqrt 2 / 2) * t, 2 + (Real.sqrt 2 / 2) * t)

-- Translate the polar equation to the cartesian form
noncomputable def cartesian_circle_equation : ∀ (x y : ℝ), x^2 + (y - 3)^2 = 9 := sorry

-- Intersection points with line and distance computation
noncomputable def points_distance (P A B : ℝ × ℝ) : ℝ :=
  let t := P.1 - A.1 in
  let t1 := Real.sqrt 7 in
  let t2 := -Real.sqrt 7 in
  let PA := Real.abs (t1 - P.1) in
  let PB := Real.abs (t2 - P.1) in
  PA + PB

theorem intersection_proof :
  (∀ (ρ θ : ℝ), ρ = 6 * Real.sin θ →
    (x y : ℝ → x^2 + (y - 3)^2 = 9)) ∧
  (∃ A B : ℝ × ℝ, parametric_line_equation t = (x, y) → 
    points_distance (1, 2) A B = 2 * Real.sqrt 7) :=
by
  sorry

end intersection_proof_l400_400707


namespace primes_between_50_and_60_l400_400331

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400331


namespace primes_between_50_and_60_l400_400311

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400311


namespace number_of_primes_between_50_and_60_l400_400362

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400362


namespace count_primes_between_50_and_60_l400_400655

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400655


namespace triangle_area_l400_400956

variable {x : ℝ}
variable {α : ℝ}
variable {CD BC AC AD : ℝ}
variable {cosα sinα : ℝ}
variable (h1 : CD = x)
variable (h2 : α = real.arccos (sqrt(2) / sqrt(3)))
variable (h3 : BC = 4 * x)
variable (h4 : cosα = sqrt(2) / sqrt(3))
variable (h5 : sinα = 1 / sqrt(3))
variable (h6 : AD = 3 / 4)

theorem triangle_area (x : ℝ) (α : ℝ) (BC : ℝ) (AC : ℝ) (AD : ℝ) (cosα : ℝ) (sinα : ℝ)
    (h1 : CD = x) (h2 : α = real.arccos (sqrt(2) / sqrt(3))) (h3 : BC = 4 * x)
    (h4 : cosα = sqrt(2) / sqrt(3)) (h5 : sinα = 1 / sqrt(3)) (h6 : AD = 3 / 4) : 
    real.sqrt(2) / 11 = S_ABC :=
by
  sorry

end triangle_area_l400_400956


namespace expression_value_one_l400_400858

theorem expression_value_one : 
  (144^2 - 12^2 = (144 - 12) * (144 + 12)) ∧ 
  (120^2 - 18^2 = (120 - 18) * (120 + 18)) →
  (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) = 1 :=
by 
  intro h,
  cases h with h1 h2,
  sorry

end expression_value_one_l400_400858


namespace combined_weight_chihuahua_pitbull_greatdane_l400_400889

noncomputable def chihuahua_pitbull_greatdane_combined_weight (C P G : ℕ) : ℕ :=
  C + P + G

theorem combined_weight_chihuahua_pitbull_greatdane :
  ∀ (C P G : ℕ), P = 3 * C → G = 3 * P + 10 → G = 307 → chihuahua_pitbull_greatdane_combined_weight C P G = 439 :=
by
  intros C P G h1 h2 h3
  sorry

end combined_weight_chihuahua_pitbull_greatdane_l400_400889


namespace pencil_length_eq_eight_l400_400679

theorem pencil_length_eq_eight (L : ℝ) 
  (h1 : (1/8) * L + (1/2) * ((7/8) * L) + (7/2) = L) : 
  L = 8 :=
by
  sorry

end pencil_length_eq_eight_l400_400679


namespace count_primes_between_50_and_60_l400_400260

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400260


namespace good_numbers_l400_400897

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ (n = 1 ∨ (nat.prime n ∧ n % 2 = 1)) :=
by
  sorry

end good_numbers_l400_400897


namespace primes_between_50_and_60_l400_400327

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400327


namespace solve_quadratic_equation_l400_400828

theorem solve_quadratic_equation :
  ∀ (x : ℝ), ((x - 2) * (x + 3) = 0) ↔ (x = 2 ∨ x = -3) :=
by
  intro x
  sorry

end solve_quadratic_equation_l400_400828


namespace danny_distance_to_work_l400_400938

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end danny_distance_to_work_l400_400938


namespace find_angle_A_find_area_l400_400722

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {R : ℝ}

/-- Given the conditions of the triangle, angle A is π/3 -/
theorem find_angle_A (h1 : a * cos (B - C) + a * cos A = 2 * sqrt 3 * b * sin C * cos A) :
  A = π / 3 :=
sorry

/-- Given the perimeter and circumradius, the area of ΔABC is 4sqrt(3)/3 -/
theorem find_area (h2 : a + b + c = 8) (h3 : R = sqrt 3) (hA : A = π / 3) :
  (1 / 2) * b * c * (sin A) = 4 * sqrt 3 / 3 :=
sorry

end find_angle_A_find_area_l400_400722


namespace extreme_value_when_a_is_1_monotonic_increasing_condition_l400_400201

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a / x + a / x^2

-- Statement for Part 1
theorem extreme_value_when_a_is_1 : 
  ∃ x : ℝ, f x 1 = 0 ∧ 
      ∀ y : ℝ, (0 < y ∧ y ≠ x) → f y 1 ≥ f x 1 := 
sorry

-- Statement for Part 2
theorem monotonic_increasing_condition (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x → differentiable_at ℝ (f x a) x ∧ deriv (f x a) x ≥ 0) 
  → -2 ≤ a ∧ a ≤ 1 := 
sorry

end extreme_value_when_a_is_1_monotonic_increasing_condition_l400_400201


namespace maximum_area_of_triangle_l400_400953

variable (AB AC : ℝ) (angle_BAC : ℝ)
variable (h0 : 0 < AB) (h1 : 0 < AC) (h2 : 0 ≤ angle_BAC ∧ angle_BAC ≤ π)

theorem maximum_area_of_triangle :
  (∀θ, 0 ≤ θ ∧ θ ≤ π → (1/2) * AB * AC * real.sin θ ≤ (1/2) * AB * AC) ∧ 
  ((1/2) * AB * AC * real.sin (π/2) = (1/2) * AB * AC) := 
by
  sorry

end maximum_area_of_triangle_l400_400953


namespace primes_between_50_and_60_l400_400438

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400438


namespace range_of_a_l400_400829

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, (x^2 - (a + 1) * x + a < 0) → ∑ x in finset.filter (λ x, x^2 - (a + 1) * x + a < 0) (finset.Icc 1 (⌊a⌋ : ℤ)), x = 27) →
  a ∈ set.Ioc (7:ℝ) 8 :=
by
  sorry

end range_of_a_l400_400829


namespace number_of_equilateral_triangles_l400_400688

noncomputable def parabola_equilateral_triangles (y x : ℝ) : Prop :=
  y^2 = 4 * x

theorem number_of_equilateral_triangles : ∃ n : ℕ, n = 2 ∧
  ∀ (a b c d e : ℝ), 
    (parabola_equilateral_triangles (a - 1) b) ∧ 
    (parabola_equilateral_triangles (c - 1) d) ∧ 
    ((a = e ∧ b = 0) ∨ (c = e ∧ d = 0)) → n = 2 :=
by 
  sorry

end number_of_equilateral_triangles_l400_400688


namespace primes_count_between_50_and_60_l400_400554

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400554


namespace count_primes_50_60_l400_400477

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400477


namespace number_of_primes_between_50_and_60_l400_400358

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400358


namespace three_students_received_A_l400_400916

variables (A B C E D : Prop)
variables (h1 : A → B) (h2 : B → C) (h3 : C → E) (h4 : E → D)

theorem three_students_received_A :
  (A ∨ ¬A) ∧ (B ∨ ¬B) ∧ (C ∨ ¬C) ∧ (E ∨ ¬E) ∧ (D ∨ ¬D) ∧ (¬A ∧ ¬B) → (C ∧ E ∧ D) ∧ ¬A ∧ ¬B :=
by sorry

end three_students_received_A_l400_400916


namespace prime_count_50_to_60_l400_400506

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400506


namespace prime_count_between_50_and_60_l400_400349

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400349


namespace prime_count_between_50_and_60_l400_400581

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400581


namespace gcd_n3_n3_eq_one_l400_400760

theorem gcd_n3_n3_eq_one (n : ℕ) (h1 : 0 < n) (h2 : (nat.divisors (n^2 - 9)).length = 6) : Nat.gcd (n - 3) (n + 3) = 1 :=
by
  sorry

end gcd_n3_n3_eq_one_l400_400760


namespace clock_angle_l400_400212

theorem clock_angle :
  ∃ (M : ℚ), 
    (M = 21 + 9 / 11 ∨ M ≈ 21.82) ∧ 
    let angle_4_00 := 120 in 
    ∃ (M1 : ℚ), 120 - 5.5 * M1 = 120 ∧ 
                4 * 30 + 0.5 * M - 6 * M = angle_4_00 ∧ 
                60 - M = M1 :=
sorry

end clock_angle_l400_400212


namespace blue_faces_cube_l400_400914

theorem blue_faces_cube (n : ℕ) (h1 : n > 0) (h2 : (6 * n^2) = 1 / 3 * 6 * n^3) : n = 3 :=
by
  -- we only need the statement for now; the proof is omitted.
  sorry

end blue_faces_cube_l400_400914


namespace segment_BE_approx_equal_l400_400947

noncomputable def circle : Type := sorry
def diameter (c: Type) : Type := sorry
def trisection (p: Type × Type) : Type := sorry
def chord (c: Type × Type × Type) : Type := sorry
def intersection (c: Type × Type) : Type := sorry
def radius (c: Type × Type) : Type := sorry
def side_length_17gon (r : ℝ) : ℝ := 2 * real.sin (real.pi / 17)

theorem segment_BE_approx_equal (k : circle)
    (A B : Type) 
    (h_diameter : diameter k = AB) 
    (C : Type) (h_trisection : trisection (A, B) = AC)
    (h_trisection2 : AC = 2 * CB) 
    (D : Type) (h_chord : chord (D, A, B) = CD)
    (E : Type) (h_intersection : intersection (A, AD) = E) 
    (h_radius : radius k = AD) :
    BE ≈ 0.3670 :=
sorry

end segment_BE_approx_equal_l400_400947


namespace prime_count_50_to_60_l400_400499

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400499


namespace percentage_of_students_on_trip_l400_400680

variable (students : ℕ) -- Total number of students at the school
variable (students_trip_and_more_than_100 : ℕ) -- Number of students who went to the camping trip and took more than $100
variable (percent_trip_and_more_than_100 : ℚ) -- Percent of students who went to camping trip and took more than $100

-- Given Conditions
def cond1 : students_trip_and_more_than_100 = (percent_trip_and_more_than_100 * students) := 
  by
    sorry  -- This will represent the first condition: 18% of students went to a camping trip and took more than $100.

variable (percent_did_not_take_more_than_100 : ℚ) -- Percent of students who went to camping trip and did not take more than $100

-- second condition
def cond2 : percent_did_not_take_more_than_100 = 0.75 := 
  by
    sorry  -- Represent the second condition: 75% of students who went to the camping trip did not take more than $100.

-- Prove
theorem percentage_of_students_on_trip : 
  (students_trip_and_more_than_100 / (0.25 * students)) * 100 = (72 : ℚ) := 
  by
    sorry

end percentage_of_students_on_trip_l400_400680


namespace percentage_error_in_calculated_area_l400_400037

theorem percentage_error_in_calculated_area 
  (s : ℝ) 
  (measured_side : ℝ) 
  (h : measured_side = s * 1.04) :
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 8.16 :=
by
  sorry

end percentage_error_in_calculated_area_l400_400037


namespace find_larger_number_l400_400012

theorem find_larger_number (x y : ℤ) (h1 : x - y = 7) (h2 : x + y = 41) : x = 24 :=
by sorry

end find_larger_number_l400_400012


namespace polynomial_sum_of_squares_l400_400081

open Polynomial

noncomputable def homogeneous (P : Polynomial ℝ → ℝ → ℝ) (d : ℕ) : Prop :=
  ∀ (c : ℝ) (x y : ℝ), P (c * x) (c * y) = c ^ d * P x y

theorem polynomial_sum_of_squares 
  (A B C R : Polynomial ℝ → ℝ → ℝ) 
  (deg3 : ∀ x y, homogeneous (B) 3) 
  (deg4 : ∀ x y, homogeneous (C) 4)
  (deg2 : ∀ x y, homogeneous (A) 2)
  (HR : ∀ x y, B x y ^ 2 - 4 * A x y * C x y = - R x y ^ 2) 
  (Hnonneg : ∀ x y z : ℝ, A x y * z^2 + B x y * z + C x y ≥ 0) :
  ∃ F G : Polynomial ℝ → ℝ → ℝ → ℝ,
  ∀ x y z, F x y z ^ 2 + G x y z ^ 2 = A x y * z^2 + B x y * z + C x y := 
sorry

end polynomial_sum_of_squares_l400_400081


namespace arithmetic_sequence_d_range_l400_400708

variable {d : ℝ}

-- Conditions
def a₁ : ℝ := -6
def Sn (n : ℕ) : ℝ := -6 * n + (n * (n - 1)) / 2 * d

-- Theorem
theorem arithmetic_sequence_d_range (hd : 1 < d ∧ d < 6 / 5) :
  (Sn 7 < Sn 6) ∧ (Sn 5 < Sn 6) :=
sorry

end arithmetic_sequence_d_range_l400_400708


namespace sum_of_abs_ineq_l400_400147

noncomputable def b (a i j k : ℝ) : ℝ :=
  (1 + (a * i) / (a - i)) * (1 + (a * k) / (a - k))

theorem sum_of_abs_ineq (a1 a2 a3 : ℝ)
  (h_distinct : a1 ≠ a2 ∧ a2 ≠ a3 ∧ a1 ≠ a3) :
  let b1 := b a1 a2 a3,
      b2 := b a2 a1 a3,
      b3 := b a3 a1 a2 in
  (1 + |a1 * b1 + a2 * b2 + a3 * b3|) ≤ (1 + |a1|) * (1 + |a2|) * (1 + |a3|) ∧ 
  ((1 + |a1 * b1 + a2 * b2 + a3 * b3|) = (1 + |a1|) * (1 + |a2|) * (1 + |a3|) ↔ 
  (a1 ≥ 0 ∧ a2 ≥ 0 ∧ a3 ≥ 0) ∨ (a1 ≤ 0 ∧ a2 ≤ 0 ∧ a3 ≤ 0)) :=
by
  sorry

end sum_of_abs_ineq_l400_400147


namespace proof_product_eq_l400_400864

theorem proof_product_eq (a b c d : ℚ) (h1 : 2 * a + 3 * b + 5 * c + 7 * d = 42)
    (h2 : 4 * (d + c) = b) (h3 : 2 * b + 2 * c = a) (h4 : c - 2 = d) :
    a * b * c * d = -26880 / 729 := by
  sorry

end proof_product_eq_l400_400864


namespace logarithmic_expression_equals_3_l400_400093

open Real

noncomputable def log_base (b x : ℝ) : ℝ := log x / log b

theorem logarithmic_expression_equals_3 :
  log_base 3 27 + log 0.01 / log 10 + log (√Real.exp 1 / 2) + 2 ^ ((-1) + log 3 / log 2) = 3 := by
  sorry

end logarithmic_expression_equals_3_l400_400093


namespace prime_count_between_50_and_60_l400_400348

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400348


namespace center_gravity_spherical_segment_correct_l400_400957

noncomputable def center_gravity_spherical_segment
  (R h : ℝ) 
  (k : ℝ)
  (density : ℝ → ℝ := λ z, k * (z - R + h))
  (sphere_eq : ℝ → ℝ → ℝ → Prop := λ x y z, x^2 + y^2 + z^2 = R^2)
  (plane_eq : ℝ := R - h) : ℝ :=
  (20 * R^2 - 15 * R * h + 3 * h^2) / (5 * (4 * R - h))

theorem center_gravity_spherical_segment_correct
  (R h : ℝ)
  (k : ℝ)
  (density : ℝ → ℝ := λ z, k * (z - R + h))
  (sphere_eq : ℝ → ℝ → ℝ → Prop := λ x y z, x^2 + y^2 + z^2 = R^2)
  (plane_eq : ℝ := R - h) : 
  center_gravity_spherical_segment R h k = (20 * R^2 - 15 * R * h + 3 * h^2) / (5 * (4 * R - h)) :=
sorry

end center_gravity_spherical_segment_correct_l400_400957


namespace count_primes_50_60_l400_400489

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400489


namespace primes_between_50_and_60_l400_400229

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400229


namespace num_primes_50_60_l400_400266

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400266


namespace median_of_special_list_l400_400714

theorem median_of_special_list :
  let list := (List.range 100).bind (λ n, List.replicate (n+1) (n+1)) -- creates the list [1, 2, 2, 3, 3, 3, ..., 100, 100, ..., 100]
  (List.length list = 5050) →
  let sorted_list := list.sort
  let median := (sorted_list.nthLe (2524) sorry + sorted_list.nthLe (2525) sorry) / 2
  median = 71 :=
by
  sorry

end median_of_special_list_l400_400714


namespace actual_price_of_food_l400_400033

theorem actual_price_of_food (total_amount spent tax_rate tip_rate : ℝ) (h1 : total_amount = 184.80) (h2 : tax_rate = 0.10) (h3 : tip_rate = 0.20) :
  let actual_price := 140 in
  let price_with_tax : ℝ := actual_price * (1 + tax_rate) in
  let total_cost : ℝ := price_with_tax * (1 + tip_rate) in
  total_cost = total_amount :=
by 
  sorry

end actual_price_of_food_l400_400033


namespace prime_count_between_50_and_60_l400_400575

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400575


namespace compound_interest_rate_l400_400086

theorem compound_interest_rate :
  ∀ (A P : ℝ) (t : ℕ),
  A = 4840.000000000001 ->
  P = 4000 ->
  t = 2 ->
  A = P * (1 + 0.1)^t :=
by
  intros A P t hA hP ht
  rw [hA, hP, ht]
  norm_num
  sorry

end compound_interest_rate_l400_400086


namespace perimeter_of_quadrilateral_l400_400022

theorem perimeter_of_quadrilateral 
  (EF FG GH : ℝ)
  (h1 : EF = 10)
  (h2 : FG = 15)
  (h3 : GH = 3)
  (h4 : EF ⊥ FG)
  (h5 : GH ⊥ FG) :
  ∃ EH, EH = Real.sqrt (7^2 + 15^2) ∧
  (EF + FG + GH + EH = 28 + Real.sqrt 274) := sorry

end perimeter_of_quadrilateral_l400_400022


namespace speed_of_A_is_7_l400_400050

theorem speed_of_A_is_7
  (x : ℝ)
  (h1 : ∀ t : ℝ, t = 1)
  (h2 : ∀ y : ℝ, y = 3)
  (h3 : ∀ n : ℕ, n = 10)
  (h4 : x + 3 = 10) :
  x = 7 := by
  sorry

end speed_of_A_is_7_l400_400050


namespace primes_count_between_50_and_60_l400_400551

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400551


namespace prime_count_between_50_and_60_l400_400578

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400578


namespace primes_between_50_and_60_l400_400320

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400320


namespace primes_between_50_and_60_l400_400422

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400422


namespace prime_count_50_to_60_l400_400494

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400494


namespace factor_expression_l400_400949

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) :=
by
  sorry

end factor_expression_l400_400949


namespace inverse_function_solution_l400_400194

theorem inverse_function_solution (a : ℝ) (f : ℝ → ℝ) (finv : ℝ → ℝ) :
  f = λ x, real.sqrt (x - a) →
  (∀ y, y ≥ 0 → finv y = y^2 + a) →
  finv 0 = 1 →
  ∀ x, finv x = 2 → x = 1 :=
begin
  intros hf hfinv ha x heq,
  sorry
end

end inverse_function_solution_l400_400194


namespace part1_part2_l400_400197

/- Define the function f(x) = |x-1| + |x-a| -/
def f (x a : ℝ) := abs (x - 1) + abs (x - a)

/- Part 1: Prove that if f(x) ≥ 2 implies the solution set {x | x ≤ 1/2 or x ≥ 5/2}, then a = 2 -/
theorem part1 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 → (x ≤ 1/2 ∨ x ≥ 5/2)) : a = 2 :=
  sorry

/- Part 2: Prove that for all x ∈ ℝ, f(x) + |x-1| ≥ 1 implies a ∈ [2, +∞) -/
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a + abs (x - 1) ≥ 1) : 2 ≤ a :=
  sorry

end part1_part2_l400_400197


namespace partitions_of_set_into_nonempty_subsets_with_conditions_l400_400963

theorem partitions_of_set_into_nonempty_subsets_with_conditions : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  in (∃ A B C : set ℕ, 
        A ∪ B ∪ C = S ∧ 
        A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ 
        (∀ x ∈ A, ∀ y ∈ A, x ≠ y → |x - y| ≠ 1) ∧
        (∀ x ∈ B, ∀ y ∈ B, x ≠ y → |x - y| ≠ 1) ∧
        (∀ x ∈ C, ∀ y ∈ C, x ≠ y → |x - y| ≠ 1)) ∧ 
     fintype.card {p | p ∈ set.powerset S ∧ 
                       ∃ A B C : set ℕ, 
                          p = {A, B, C} ∧ 
                          A ≠ ∅ ∧ 
                          B ≠ ∅ ∧ 
                          C ≠ ∅ ∧ 
                          (∀ x ∈ A, ∀ y ∈ A, x ≠ y → |x - y| ≠ 1) ∧ 
                          (∀ x ∈ B, ∀ y ∈ B, x ≠ y → |x - y| ≠ 1) ∧ 
                          (∀ x ∈ C, ∀ y ∈ C, x ≠ y → |x - y| ≠ 1)} = 1023 :=
by sorry

end partitions_of_set_into_nonempty_subsets_with_conditions_l400_400963


namespace num_primes_50_60_l400_400272

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400272


namespace harmonic_mean_pairs_eq_67_l400_400962

theorem harmonic_mean_pairs_eq_67 : 
  ∃ (n : ℕ), n = 67 ∧ ∃ (a b : ℕ), a < b ∧ harmonic_mean a b = 12^4 ∧ (a, b).pairs.length = n := 
sorry

def harmonic_mean (a b : ℕ) : ℕ :=
  2 * a * b / (a + b)

end harmonic_mean_pairs_eq_67_l400_400962


namespace prime_count_between_50_and_60_l400_400352

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400352


namespace prime_count_between_50_and_60_l400_400583

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400583


namespace no_value_of_b_l400_400108

theorem no_value_of_b (b : ℤ) : ¬ ∃ (n : ℤ), 2 * b^2 + 3 * b + 2 = n^2 := 
sorry

end no_value_of_b_l400_400108


namespace magnitude_circumcenter_vector_sum_l400_400740

-- Definitions and conditions
variable {V : Type*} [inner_product_space ℝ V]
variables (A B C H O : V)

-- Conditions in the problem
def is_orthocenter (H A B C : V) : Prop := sorry  -- Definition of orthocenter
def is_circumcenter (O A B C : V) : Prop := sorry  -- Definition of circumcenter
def vector_sum_H_eq_2 (H A B C : V) : Prop := 
  ‖(H - A) + (H - B) + (H - C)‖ = 2

-- Statement of the theorem
theorem magnitude_circumcenter_vector_sum
  (H A B C O : V)
  (hH : is_orthocenter H A B C)
  (hO : is_circumcenter O A B C)
  (hSumH : vector_sum_H_eq_2 H A B C) :
  ‖(O - A) + (O - B) + (O - C)‖ = 1 := 
sorry

end magnitude_circumcenter_vector_sum_l400_400740


namespace apples_distribution_l400_400881

theorem apples_distribution (baskets : ℕ → ℕ) (total_apples : ℕ)
  (h1 : total_apples = ∑ n in (finset.range (2000)), baskets n) :
  ∃ (remaining_baskets : ℕ → ℕ) (remaining_total : ℕ),
    remaining_total >= 100 ∧ 
    (∀ i j, remaining_baskets i = remaining_baskets j) ∧
    remaining_total = ∑ n in (finset.range (2000)), remaining_baskets n :=
sorry

end apples_distribution_l400_400881


namespace half_abs_diff_squares_l400_400845

theorem half_abs_diff_squares (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  (1 / 2) * |a^2 - b^2| = 112.5 :=
sorry

end half_abs_diff_squares_l400_400845


namespace primes_between_50_and_60_l400_400390

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400390


namespace prime_count_between_50_and_60_l400_400334

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400334


namespace minimum_S_n_values_l400_400172

/-- 
Given an arithmetic sequence {a_n} with sum S_n, where a_2 = -2 and S_4 = -4,
prove that the values of n that minimize S_n are n = 2 or n = 3.
-/
theorem minimum_S_n_values (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)
  (h1 : a_n 2 = -2) (h2 : S_n 4 = -4) : S_n 2 ≤ S_n 3 ∧ ∀ n, S_n n ≤ S_n (n + 1) → n = 2 ∨ n = 3 :=
begin
  sorry
end

end minimum_S_n_values_l400_400172


namespace probability_of_union_l400_400002

-- Define the event probabilities and the condition of mutual exclusivity
def event_prob_A : ℚ := 1/2
def event_prob_B : ℚ := 1/6
def mutually_exclusive : Prop := true  -- Since A and B have no common outcomes

-- Define the statement to be proved
theorem probability_of_union {A B : Prop} (P_A : ℚ) (P_B : ℚ) (h_exclusive : mutually_exclusive) : 
  P_A = event_prob_A ∧ P_B = event_prob_B → (P_A + P_B) = 2/3 := by
  assume h : P_A = event_prob_A ∧ P_B = event_prob_B
  sorry

end probability_of_union_l400_400002


namespace count_primes_between_50_and_60_l400_400254

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400254


namespace example_theorem_l400_400979

variable (a : ℝ)
hypothesis h : a^3 + 2 * a = -2

theorem example_theorem : 3 * a^6 + 12 * a^4 - a^3 + 12 * a^2 - 2 * a - 4 = 10 :=
by
  sorry

end example_theorem_l400_400979


namespace proof_problem_l400_400113

-- Definition of parametric equations for line l
def parametric_x (t : ℝ) := t
def parametric_y (t : ℝ) := 1 + 2 * t

-- Standard form of the line l
def standard_line (x y : ℝ) := y - 2 * x - 1 = 0

-- Polar equation of circle C
def polar_circle (ρ θ : ℝ) := ρ = 2 * Real.sqrt 2 * Real.sin (θ + π / 4)

-- Cartesian coordinate equation of circle C
def cartesian_circle (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 2

-- Proof problem statement
theorem proof_problem (t x y ρ θ : ℝ) :
  (∃ t, parametric_x t = x ∧ parametric_y t = y) →
  polar_circle ρ θ →
  (∃ x y, standard_line x y ∧ cartesian_circle x y ∧ (ρ^2 = (x^2 + y^2) ∧ 2 * ρ * Real.sin θ = 2 * (y - 1) ∧ 2 * ρ * Real.cos θ = 2 * (x - 1))) →
  True := by sorry

end proof_problem_l400_400113


namespace primes_between_50_and_60_l400_400230

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400230


namespace primes_between_50_and_60_l400_400535

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400535


namespace primes_between_50_and_60_l400_400634

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400634


namespace count_primes_between_50_and_60_l400_400466

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400466


namespace num_primes_50_60_l400_400274

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400274


namespace f_even_f_decreasing_on_nonnegative_real_solve_inequality_l400_400192

def f (x : ℝ) := 4 - x^2

theorem f_even : ∀ x : ℝ, f (-x) = f x :=
by sorry

theorem f_decreasing_on_nonnegative_real : ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 < x2 → f x1 > f x2 :=
by sorry

theorem solve_inequality : {x : ℝ | f x ≥ 3 * x} = {x : ℕ | -4 ≤ x ∧ x ≤ 1} :=
by sorry

end f_even_f_decreasing_on_nonnegative_real_solve_inequality_l400_400192


namespace tangent_and_normal_eqns_l400_400958

theorem tangent_and_normal_eqns (a : ℝ) :
  let t₀ := Real.pi / 6 in
  let x := a * Real.sin t₀ ^ 3 in
  let y := a * Real.cos t₀ ^ 3 in
  (∃ (tangent_eqn normal_eqn : ℝ → ℝ),
    (∀ x, tangent_eqn x = -Real.sqrt 3 * x + Real.sqrt 3 * a / 2) ∧ 
    (∀ x, normal_eqn x = x / Real.sqrt 3 + a / Real.sqrt 3)) :=
by
  let t₀ := Real.pi / 6
  let x := a * Real.sin t₀ ^ 3
  let y := a * Real.cos t₀ ^ 3
  existsi (λ x, -Real.sqrt 3 * x + Real.sqrt 3 * a / 2)
  existsi (λ x, x / Real.sqrt 3 + a / Real.sqrt 3)
  split
  assumption
  assumption
  sorry

end tangent_and_normal_eqns_l400_400958


namespace primes_between_50_and_60_l400_400309

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400309


namespace primes_between_50_and_60_l400_400523

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400523


namespace optimal_play_results_in_draw_l400_400716

-- Define the game state and basic rules
inductive Player
| white
| black

-- Define what it means for a game to be a draw
noncomputable def game_is_draw (initial_state : Type) (move : initial_state → initial_state) (win_condition : initial_state → Option Player) : Prop :=
  ∀ s : initial_state, win_condition s = none

-- Assume the initial state and move function are specified
variables (initial_state : Type) (move : initial_state → initial_state) (win_condition : initial_state → Option Player)

-- Now, state the main theorem
theorem optimal_play_results_in_draw :
  game_is_draw initial_state move win_condition :=
begin
  sorry,
end

end optimal_play_results_in_draw_l400_400716


namespace remainder_of_12_pow_2012_mod_5_l400_400137

theorem remainder_of_12_pow_2012_mod_5 : (12 ^ 2012) % 5 = 1 :=
by
  sorry

end remainder_of_12_pow_2012_mod_5_l400_400137


namespace half_abs_diff_squares_l400_400846

theorem half_abs_diff_squares (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  (1 / 2) * |a^2 - b^2| = 112.5 :=
sorry

end half_abs_diff_squares_l400_400846


namespace count_primes_between_50_and_60_l400_400252

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400252


namespace primes_between_50_and_60_l400_400420

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400420


namespace perimeter_triangle_PQR_l400_400692

-- Definitions of the given conditions
variables (P Q R X Y Z W : Type) [plane P Q R X Y Z W] 
variables (triangle_PQR : triangle P Q R)
hypothesis angle_R : ∠ R = 90
hypothesis len_PQ : PQ = 10
hypothesis sq_PQXY : square PQ X Y
hypothesis sq_PRWZ : square PR W Z
hypothesis circle_XYZW : circle {X, Y, Z, W}

-- Statement of the proof problem
theorem perimeter_triangle_PQR : perimeter triangle_PQR = 10 + 10 * sqrt 2 :=
sorry -- Proof to be constructed

end perimeter_triangle_PQR_l400_400692


namespace count_primes_between_50_and_60_l400_400449

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400449


namespace puffy_more_than_muffy_l400_400923

-- Definitions and assumptions
def Scruffy_weight : ℕ := 12
def Muffy_weight (S : ℕ) : ℕ := S - 3
def total_weight (P M : ℕ) : ℕ := P + M

-- Theorem stating the problem
theorem puffy_more_than_muffy :
  let S := Scruffy_weight in
  let M := Muffy_weight S in
  let P := 23 - M in
  P - M = 5 :=
by
  let S := Scruffy_weight
  let M := Muffy_weight S
  let P := 23 - M
  have : P - M = 5 := sorry
  exact this

end puffy_more_than_muffy_l400_400923


namespace interior_points_collinear_l400_400895

-- Define a lattice point as a point with integer coordinates.
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define a triangle with vertices as lattice points.
structure Triangle :=
  (A B C : LatticePoint)
  (no_other_lattice_points_on_sides : ¬ ∃ (P : LatticePoint), 
      (P ≠ A ∧ P ≠ B ∧ P ≠ C) ∧ 
      collinear [A, B, P] ∨ collinear [A, C, P] ∨ collinear [B, C, P])

-- Define the four lattice points inside the triangle.
def four_lattice_points_in_interior (T : Triangle) : Prop :=
  ∃ (P₁ P₂ P₃ P₄ : LatticePoint), 
    ¬ collinear [T.A, T.B, P₁] ∧ ¬ collinear [T.A, T.B, P₂] ∧ 
    ¬ collinear [T.A, T.B, P₃] ∧ ¬ collinear [T.A, T.B, P₄] ∧ 
    inside_triangle T P₁ ∧ inside_triangle T P₂ ∧ 
    inside_triangle T P₃ ∧ inside_triangle T P₄

-- Define the collinearity of points.
def points_collinear (P₁ P₂ P₃ : LatticePoint) : Prop :=
  (P₁.x * P₂.y + P₂.x * P₃.y + P₃.x * P₁.y - P₁.y * P₂.x - P₂.y * P₃.x - P₃.y * P₁.x = 0)

-- Define an interior point of a triangle.
def inside_triangle (T : Triangle) (P : LatticePoint) : Prop :=
  http://Example.com

theorem interior_points_collinear (T : Triangle)
  (h₁ : four_lattice_points_in_interior T) : ∃ (P₁ P₂ P₃ P₄ : LatticePoint), 
  points_collinear P₁ P₂ P₃ ∧ points_collinear P₂ P₃ P₄ ∧ T.contains P₁ ∧ T.contains P₂ ∧ T.contains P₃ ∧ T.contains P₄ :=
sorry

end interior_points_collinear_l400_400895


namespace problem1_problem2_l400_400195

-- Stating the proof problems in Lean 4

-- Problem (I)
theorem problem1 (x : ℝ) (hx : 0 < x) : 
    let f := λ (x : ℝ), x^2 + log x in 
    f x ≥ (x^3 + x - 1) / x := 
sorry

-- Problem (II)
theorem problem2 : 
  let f := λ (x : ℝ), x^2 - 2 * log x in
  ∃ (xmin : ℝ) (xmax : ℝ), 
    xmin = 1 ∧ xmax = 4 - 2 * log 2 ∧ 
    ∀ (x : ℝ), (1 / 2 ≤ x ∧ x ≤ 2) → (f 1 ≤ f x ∧ f x ≤ f 2) :=
sorry

end problem1_problem2_l400_400195


namespace fence_remaining_l400_400004

noncomputable def totalFence : Float := 150.0
noncomputable def ben_whitewashed : Float := 20.0

-- Remaining fence after Ben's contribution
noncomputable def remaining_after_ben : Float := totalFence - ben_whitewashed

noncomputable def billy_fraction : Float := 1.0 / 5.0
noncomputable def billy_whitewashed : Float := billy_fraction * remaining_after_ben

-- Remaining fence after Billy's contribution
noncomputable def remaining_after_billy : Float := remaining_after_ben - billy_whitewashed

noncomputable def johnny_fraction : Float := 1.0 / 3.0
noncomputable def johnny_whitewashed : Float := johnny_fraction * remaining_after_billy

-- Remaining fence after Johnny's contribution
noncomputable def remaining_after_johnny : Float := remaining_after_billy - johnny_whitewashed

noncomputable def timmy_percentage : Float := 15.0 / 100.0
noncomputable def timmy_whitewashed : Float := timmy_percentage * remaining_after_johnny

-- Remaining fence after Timmy's contribution
noncomputable def remaining_after_timmy : Float := remaining_after_johnny - timmy_whitewashed

noncomputable def alice_fraction : Float := 1.0 / 8.0
noncomputable def alice_whitewashed : Float := alice_fraction * remaining_after_timmy

-- Remaining fence after Alice's contribution
noncomputable def remaining_fence : Float := remaining_after_timmy - alice_whitewashed

theorem fence_remaining : remaining_fence = 51.56 :=
by
    -- Placeholder for actual proof
    sorry

end fence_remaining_l400_400004


namespace number_of_primes_between_50_and_60_l400_400365

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400365


namespace sum_double_series_l400_400102

theorem sum_double_series :
  (∑' n : ℕ, ∑ k in finset.range (n + 1), (k ^ 2 : ℚ) / 3 ^ (n + k + 2)) = 3645 / 41552 :=
by
  sorry

end sum_double_series_l400_400102


namespace min_AC_value_l400_400934

theorem min_AC_value :
  ∀ (A B C D : Type) (a b c d : ℕ),
  (a + b = c) →
  (d^2 = 57) → 
  ∃ (k : ℕ), a + b = 11 :=
begin
  sorry
end

end min_AC_value_l400_400934


namespace count_primes_between_50_and_60_l400_400661

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400661


namespace log_product_eq_l400_400097

theorem log_product_eq :
  (1 / 2 * log 2 3) * (1 / 2 * log 9 8) = 3 / 8 :=
by
  sorry

end log_product_eq_l400_400097


namespace geometric_sequence_log_sum_l400_400696

theorem geometric_sequence_log_sum {a : ℕ → ℝ} (h_seq : ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) (h_positive : ∀ n : ℕ, 0 < a n) (h_condition : a 2 * a 5 = 10) :
  log (a 3) + log (a 4) = 1 :=
sorry

end geometric_sequence_log_sum_l400_400696


namespace count_primes_between_50_and_60_l400_400468

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400468


namespace primes_between_50_and_60_l400_400293

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400293


namespace num_primes_between_50_and_60_l400_400627

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400627


namespace conference_handshakes_l400_400090

theorem conference_handshakes :
  ∃ (n : ℕ), n = 40 ∧
  ∃ (groupA : ℕ), groupA = 25 ∧
  ∃ (groupB : ℕ), groupB = 15 ∧
  ( ∃ (handshakes_between : ℕ), handshakes_between = groupA * groupB ∧
    ∃ (handshakes_within : ℕ), handshakes_within = (groupB * (groupB - 1)) / 2 ∧
    (handshakes_between + handshakes_within = 480) ) :=
begin
  sorry -- No proof required
end

end conference_handshakes_l400_400090


namespace first_new_player_weight_l400_400832

theorem first_new_player_weight (X : ℝ) :
  (∃ X : ℝ,
    let original_players := 7 in
    let average_weight_original := 76 in
    let new_player2 := 60 in
    let new_players := 2 in
    let average_weight_new := 78 in
    let total_weight_original := original_players * average_weight_original in
    let total_weight_new := (original_players + new_players) * average_weight_new in
    total_weight_new = total_weight_original + X + new_player2
  ) → X = 110 :=
by
  intro h
  sorry

end first_new_player_weight_l400_400832


namespace cos_double_angle_l400_400154

theorem cos_double_angle (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = -7 / 8 :=
sorry

end cos_double_angle_l400_400154


namespace sum_f_1_to_2016_l400_400198

theorem sum_f_1_to_2016 :
  ∃ A ω φ : ℝ, 
    0 < A ∧ 0 < ω ∧ 0 < φ ∧ φ < π / 2 ∧
    (∀ x, f x = A * cos(ω * x + φ)^2 + 1) ∧
    (∀ x, (A > 0 ∧ ω > 0 ∧ 0 < φ < π / 2) → (∀ x, f(x) ≤ 3) ∧ f(0) = 2) ∧
    (distance_between_axes = 2) → 
    (f 1 + f 2 + f 3 + ... + f 2016 = 4032) := 
sorry

end sum_f_1_to_2016_l400_400198


namespace clock_angle_l400_400213

theorem clock_angle :
  ∃ (M : ℚ), 
    (M = 21 + 9 / 11 ∨ M ≈ 21.82) ∧ 
    let angle_4_00 := 120 in 
    ∃ (M1 : ℚ), 120 - 5.5 * M1 = 120 ∧ 
                4 * 30 + 0.5 * M - 6 * M = angle_4_00 ∧ 
                60 - M = M1 :=
sorry

end clock_angle_l400_400213


namespace prime_count_between_50_and_60_l400_400563

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400563


namespace count_primes_between_50_and_60_l400_400465

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400465


namespace count_primes_between_50_and_60_l400_400665

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400665


namespace find_e_and_l_l400_400730

theorem find_e_and_l (e l : ℝ) 
  (B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 7, e])
  (h : inverse B = l • B) :
  (e, l) = (-3, 1/19) := by
  sorry

end find_e_and_l_l400_400730


namespace primes_between_50_and_60_l400_400305

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400305


namespace primes_between_50_and_60_l400_400633

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400633


namespace prime_count_50_to_60_l400_400511

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400511


namespace num_primes_50_60_l400_400264

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400264


namespace prime_count_between_50_and_60_l400_400580

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400580


namespace find_x_in_triangle_l400_400712

theorem find_x_in_triangle 
  (P Q R S: Type) 
  (PQS_is_straight: PQS) 
  (angle_PQR: ℝ)
  (h1: angle_PQR = 110) 
  (angle_RQS : ℝ)
  (h2: angle_RQS = 70)
  (angle_QRS : ℝ)
  (h3: angle_QRS = 3 * angle_x)
  (angle_QSR : ℝ)
  (h4: angle_QSR = angle_x + 14) 
  (triangle_angles_sum : ∀ (a b c: ℝ), a + b + c = 180) : 
  angle_x = 24 :=
by
  sorry

end find_x_in_triangle_l400_400712


namespace problem1_problem2_problem3_l400_400196

def f (a x : ℝ) : ℝ := x^3 - a * x - 1

theorem problem1 (a : ℝ) : (∀ x : ℝ, (3 * x ^ 2 - a) ≥ 0) → a ≤ 0 :=
sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, x ∈ Ioo (-1) 1 → (3 * x ^ 2 - a) < 0) → a ≥ 3 :=
sorry

theorem problem3 (a : ℝ) : ¬ (∀ x : ℝ, x^3 - a * x - 1 > a) :=
sorry

end problem1_problem2_problem3_l400_400196


namespace primes_between_50_and_60_l400_400651

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400651


namespace primes_between_50_and_60_l400_400602

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp is_prime

theorem primes_between_50_and_60 : 
  count_primes_in_range 50 60 = 2 :=
by sorry

end primes_between_50_and_60_l400_400602


namespace sum_x_coordinates_Q4_is_3000_l400_400882

-- Let Q1 be a 150-gon with vertices having x-coordinates summing to 3000
def Q1_x_sum := 3000
def Q2_x_sum := Q1_x_sum
def Q3_x_sum := Q2_x_sum
def Q4_x_sum := Q3_x_sum

-- Theorem to prove the sum of the x-coordinates of the vertices of Q4 is 3000
theorem sum_x_coordinates_Q4_is_3000 : Q4_x_sum = 3000 := by
  sorry

end sum_x_coordinates_Q4_is_3000_l400_400882


namespace rock_age_possibilities_l400_400080

theorem rock_age_possibilities :
  let digits := [2, 2, 2, 3, 7, 9],
      odd_digits := [3, 7, 9],
      factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)
  in ∃ n : Nat, n = (odd_digits.length * (factorial 5 / factorial 3)) ∧ n = 60 :=
begin
  sorry
end

end rock_age_possibilities_l400_400080


namespace primes_between_50_and_60_l400_400223

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400223


namespace count_primes_50_60_l400_400484

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem count_primes_50_60 : 
  let primes_between := List.filter isPrime (List.range 10).map (λ n, 51 + n)
  List.length primes_between = 2 :=
by 
  sorry

end count_primes_50_60_l400_400484


namespace part1_part2_l400_400747

theorem part1 (a x : ℝ) (h : |x - a| < 1) :
  let f := λ x : ℝ, x^2 - x - 15 in
  |f x| > 5 ↔ (x < -4 ∨ x > 5 ∨ (1 - real.sqrt 41) / 2 < x ∧ x < (1 + real.sqrt 41) / 2) :=
sorry

theorem part2 (a x : ℝ) (h : |x - a| < 1) :
  let f := λ x : ℝ, x^2 - x - 15 in
  |f x - f a| < 2 * (|a| + 1) :=
sorry

end part1_part2_l400_400747


namespace prove_perpendicular_planes_l400_400767

-- Defining the non-coincident lines m and n
variables {m n : Set Point} {α β : Set Point}

-- Lines and plane relationship definitions
def parallel (x y : Set Point) : Prop := sorry
def perpendicular (x y : Set Point) : Prop := sorry
def subset (x y : Set Point) : Prop := sorry

-- Given conditions
axiom h1 : parallel m n
axiom h2 : subset m α
axiom h3 : perpendicular n β

-- Prove that α is perpendicular to β
theorem prove_perpendicular_planes :
  perpendicular α β :=
  sorry

end prove_perpendicular_planes_l400_400767


namespace rhombus_area_l400_400840

def is_rhombus_with_45_deg_angle (a : ℝ) (θ : ℝ) : Prop :=
  θ = π / 4 ∧ ∃ (s : ℝ), s = 3

def area_rhombus (s : ℝ) : ℝ :=
  let base := s * √2
  let height := s / √2
  base * height

theorem rhombus_area (s : ℝ) (θ : ℝ) (h : is_rhombus_with_45_deg_angle s θ) : area_rhombus s = 9 :=
by
  -- Placeholder for actual proof
  sorry

end rhombus_area_l400_400840


namespace count_primes_between_50_and_60_l400_400461

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400461


namespace prime_count_between_50_and_60_l400_400336

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400336


namespace prime_count_50_to_60_l400_400507

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400507


namespace count_primes_between_50_and_60_l400_400250

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400250


namespace primes_between_50_and_60_l400_400538

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400538


namespace probability_floor_log_10_eq_approx_l400_400765

noncomputable def probability_floor_log_eq : ℝ :=
  probability ((uniform (0, 1)) × (uniform (0, 1))) 
  {xy | ∃ n : ℤ, (xy.1 ∈ set.Ico (10^(n-1)) (10^n))
                ∧ (xy.2 ∈ set.Ico (10^(n-1)) (10^n))}

theorem probability_floor_log_10_eq_approx :
  probability_floor_log_eq ≈ 0.81818 := 
sorry

end probability_floor_log_10_eq_approx_l400_400765


namespace first_number_is_45_l400_400908

noncomputable section

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b -- Define L.C.M. function

theorem first_number_is_45 (x : ℕ) (h1 : 3 * x * 4 = 180) : 3 * x = 45 := by
  have h2 : 12 * x = 180 := by
    rw [mul_assoc, h1]
  let x := 180 / 12 -- Solve for x
  have hx : x = 15 := by
    rw [Nat.div_eq_of_eq_mul_right (by decide)]
  show 3 * x = 45 from
    calc
      3 * x = 3 * 15 := congr_arg ((·) * 3) hx
      ... = 45 := by
        norm_num -- Simplify the multiplication
  sorry -- add 'sorry' to complete the proof outline and allow the code to compile.

end first_number_is_45_l400_400908


namespace count_primes_between_50_and_60_l400_400253

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400253


namespace prime_count_between_50_and_60_l400_400347

open Nat

theorem prime_count_between_50_and_60 : 
  (∃! p : ℕ, 50 < p ∧ p < 60 ∧ Prime p) = 2 := 
by
  sorry

end prime_count_between_50_and_60_l400_400347


namespace base_8_calculation_l400_400092

theorem base_8_calculation : ∀ (a b c d e : Nat), 
  to_nat a = 4 ∧ to_nat b = 5 ∧ to_nat c = 2 ∧
  to_nat d = 127 ∧ to_nat e = 237  →
  (a - b + c = d) :=
by
  intro a b c d e
  sorry

end base_8_calculation_l400_400092


namespace primes_between_50_and_60_l400_400300

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400300


namespace distinct_triple_identity_l400_400745

theorem distinct_triple_identity (p q r : ℝ) 
  (h1 : p ≠ q) 
  (h2 : q ≠ r) 
  (h3 : r ≠ p)
  (h : (p / (q - r)) + (q / (r - p)) + (r / (p - q)) = 3) : 
  (p^2 / (q - r)^2) + (q^2 / (r - p)^2) + (r^2 / (p - q)^2) = 3 :=
by 
  sorry

end distinct_triple_identity_l400_400745


namespace maximum_area_of_triangle_l400_400954

variable (AB AC : ℝ) (angle_BAC : ℝ)
variable (h0 : 0 < AB) (h1 : 0 < AC) (h2 : 0 ≤ angle_BAC ∧ angle_BAC ≤ π)

theorem maximum_area_of_triangle :
  (∀θ, 0 ≤ θ ∧ θ ≤ π → (1/2) * AB * AC * real.sin θ ≤ (1/2) * AB * AC) ∧ 
  ((1/2) * AB * AC * real.sin (π/2) = (1/2) * AB * AC) := 
by
  sorry

end maximum_area_of_triangle_l400_400954


namespace primes_between_50_and_60_l400_400324

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Set.filter (λ n, is_prime n) (Set.Icc a b)).toFinset.card 

theorem primes_between_50_and_60 : count_primes_in_range 51 60 = 2 := by
  sorry

end primes_between_50_and_60_l400_400324


namespace man_speed_against_stream_l400_400060

variable (S : ℝ)

-- The man's speed with the stream
axiom with_stream : 7 + S = 26

-- Man’s effective speed against the stream
def against_stream_speed := 7 - S

-- Abs function to ensure speed cannot be negative
def abs_speed (x : ℝ) := if x < 0 then -x else x

-- The theorem statement
theorem man_speed_against_stream : abs_speed (against_stream_speed S) = 12 :=
by
  have hS : S = 19 := 
    calc
      S = 26 - 7 : by rw [with_stream]
      ... = 19 : by norm_num
  -- Substitute speed of stream
  have h1 : against_stream_speed S = 7 - S := rfl
  rw [h1, hS]
  -- Calculate absolute value of the resulting speed
  simp [abs_speed]
  sorry

end man_speed_against_stream_l400_400060


namespace primes_between_50_and_60_l400_400519

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400519


namespace remainder_product_div_6_l400_400138

theorem remainder_product_div_6 :
  (3 * 7 * 13 * 17 * 23 * 27 * 33 * 37 * 43 * 47 * 53 * 57 * 63 * 67 * 73 * 77 * 83 * 87 * 93 * 97 
   * 103 * 107 * 113 * 117 * 123 * 127 * 133 * 137 * 143 * 147 * 153 * 157 * 163 * 167 * 173 
   * 177 * 183 * 187 * 193 * 197) % 6 = 3 := 
by 
  -- basic info about modulo arithmetic and properties of sequences
  sorry

end remainder_product_div_6_l400_400138


namespace count_primes_between_50_and_60_l400_400657

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400657


namespace count_primes_between_50_and_60_l400_400247

theorem count_primes_between_50_and_60 :
  (finset.filter nat.prime (finset.Ico 51 60)).card = 2 :=
by {
  -- the proof goes here
  sorry
}

end count_primes_between_50_and_60_l400_400247


namespace primes_between_50_and_60_l400_400288

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400288


namespace distance_in_miles_l400_400756

theorem distance_in_miles :
  (∀ (x : ℝ), (x = 47 / 2.54)) → 
  (∀ (y : ℝ), (y = x * (24 / 1.5))) → 
  y ≈ 296.06 :=
by
  intros x h_x y h_y
  sorry

end distance_in_miles_l400_400756


namespace primes_between_50_and_60_l400_400295

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400295


namespace primes_between_50_and_60_l400_400524

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400524


namespace primes_between_50_and_60_l400_400239

open Nat

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : (finset.filter is_prime (finset.range (60 - 51 + 1)).map (λ n, n + 51)).card = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400239


namespace sequence_count_zeros_ones_15_l400_400677

-- Definition of the problem
def count_sequences (n : Nat) : Nat := sorry -- Function calculating the number of valid sequences

-- The theorem stating that for sequence length 15, the number of such sequences is 266
theorem sequence_count_zeros_ones_15 : count_sequences 15 = 266 := 
by {
  sorry -- Proof goes here
}

end sequence_count_zeros_ones_15_l400_400677


namespace prime_count_50_to_60_l400_400510

theorem prime_count_50_to_60 : (finset.filter nat.prime (finset.range 61)).filter (λ n, 50 < n ∧ n < 61) = {53, 59} :=
by
  sorry

end prime_count_50_to_60_l400_400510


namespace num_primes_50_60_l400_400265

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400265


namespace num_primes_between_50_and_60_l400_400617

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def count_primes_in_range : Nat :=
  let numbers_to_check := [51, 52, 53, 54, 55, 56, 57, 58, 59]
  let primes := numbers_to_check.filter is_prime
  primes.length

theorem num_primes_between_50_and_60 : count_primes_in_range = 2 := 
  sorry

end num_primes_between_50_and_60_l400_400617


namespace molecular_weight_of_compound_l400_400021

-- Definitions of the conditions
def atomic_weight_N : Float := 14.01 -- Atomic weight of Nitrogen (N)
def atomic_weight_O : Float := 16.00 -- Atomic weight of Oxygen (O)
def num_atoms_N : Int := 2 -- Number of Nitrogen atoms
def num_atoms_O : Int := 3 -- Number of Oxygen atoms

-- The theorem we want to prove
theorem molecular_weight_of_compound :
  (num_atoms_N * atomic_weight_N + num_atoms_O * atomic_weight_O) = 76.02 := by sorry

end molecular_weight_of_compound_l400_400021


namespace horner_method_poly_at_neg2_l400_400017

-- Define the polynomial using the given conditions and Horner's method transformation
def polynomial : ℤ → ℤ := fun x => (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2

-- State the theorem
theorem horner_method_poly_at_neg2 : polynomial (-2) = -40 := by
  sorry

end horner_method_poly_at_neg2_l400_400017


namespace primes_count_between_50_and_60_l400_400550

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400550


namespace complement_is_empty_l400_400155

def U : Set ℕ := {1, 3}
def A : Set ℕ := {1, 3}

theorem complement_is_empty : (U \ A) = ∅ := 
by 
  sorry

end complement_is_empty_l400_400155


namespace count_primes_between_50_and_60_l400_400463

open Nat

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.Ico 51 60)).card = 2 := 
sorry

end count_primes_between_50_and_60_l400_400463


namespace cos_product_identity_l400_400875

noncomputable def L : ℝ := 3.418 * (Real.cos (2 * Real.pi / 31)) *
                               (Real.cos (4 * Real.pi / 31)) *
                               (Real.cos (8 * Real.pi / 31)) *
                               (Real.cos (16 * Real.pi / 31)) *
                               (Real.cos (32 * Real.pi / 31))

theorem cos_product_identity : L = 1 / 32 := by
  sorry

end cos_product_identity_l400_400875


namespace max_missable_problems_l400_400919

theorem max_missable_problems (total_problems : ℕ) (passing_score_pct : ℝ) 
  (condition1 : total_problems = 50) (condition2 : passing_score_pct = 85.0) : 
  ∃ max_missable : ℕ, max_missable = 7 :=
by
  have h_fraction : 1 - passing_score_pct / 100 = 0.15 := by sorry
  have h_missable_fraction : 0.15 * (total_problems : ℝ) = 7.5 := by sorry
  have h_max_missable : ∃ max_missable : ℕ, max_missable = 7 := by
    use 7
    sorry
  exact h_max_missable

end max_missable_problems_l400_400919


namespace primes_between_50_and_60_l400_400434

open Nat

theorem primes_between_50_and_60 : {p ∈ Set.range Nat.succ | p ≥ 50 ∧ p ≤ 60 ∧ Prime p}.card = 2 :=
by sorry

end primes_between_50_and_60_l400_400434


namespace solve_system_l400_400788

namespace SolutionProof

-- Conditions
def equation1 (x y : ℝ) : Prop := 3^y * 81 = 9^(x^2)
def equation2 (x y : ℝ) : Prop := log 10 y = log 10 x - log 10 0.5

-- Proof problem statement
theorem solve_system : ∃ x y : ℝ, equation1 x y ∧ equation2 x y ∧ x = 2 ∧ y = 4 :=
by {
  sorry 
}

end SolutionProof

end solve_system_l400_400788


namespace symmetric_points_sum_eq_l400_400683

-- Conditions
variables {m n : ℤ}

-- Definitions for points and symmetry
def is_symmetric_with_respect_to_origin (p q : ℤ × ℤ) : Prop :=
  p = (-q.1, -q.2)

-- Points P and Q with their coordinates
def P : ℤ × ℤ := (m, 5)
def Q : ℤ × ℤ := (3, n)

-- Lean statement to prove the given mathematical problem
theorem symmetric_points_sum_eq :
  is_symmetric_with_respect_to_origin P Q → m + n = -8 := 
by
  sorry

end symmetric_points_sum_eq_l400_400683


namespace round_nearer_hundredth_l400_400770

def roundToNearestHundredth (x : ℝ) : ℝ :=
  (Float.ofReal (Float.round (x * 100.0)) / 100.0).toReal 

theorem round_nearer_hundredth : roundToNearestHundredth (-23.4985) = -23.50 := 
  sorry

end round_nearer_hundredth_l400_400770


namespace count_primes_between_50_and_60_l400_400676

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400676


namespace math_problem_l400_400943

open Real

noncomputable def log_10_neg2 : ℝ := log 10 ^ (-2)
noncomputable def log_5: ℝ := log 5
noncomputable def log_20: ℝ := log 20
noncomputable def log_2: ℝ := log 2

theorem math_problem : log_10_neg2 + log_5 * log_20 + (log_2)^2 = -1 :=
by sorry

end math_problem_l400_400943


namespace primes_between_50_and_60_l400_400631

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_50_and_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by {
  sorry
}

end primes_between_50_and_60_l400_400631


namespace least_number_of_cookies_l400_400753

theorem least_number_of_cookies :
  ∃ x : ℕ, x % 6 = 4 ∧ x % 5 = 3 ∧ x % 8 = 6 ∧ x % 9 = 7 ∧ x = 208 :=
by
  sorry

end least_number_of_cookies_l400_400753


namespace number_of_primes_between_50_and_60_l400_400374

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem number_of_primes_between_50_and_60 : primes_between 50 60 = [53, 59] :=
sorry

end number_of_primes_between_50_and_60_l400_400374


namespace exists_k_plus_one_element_subset_l400_400992

theorem exists_k_plus_one_element_subset 
  {X : Type} {n K m α : ℕ} (S : finset (finset (fin X))) 
  (h1 : S.card = m)
  (h2 : ∀ s ∈ S, s.card = K)
  (hm : m > ((k - 1) * (n - k) + k) / k^2 * (n.choose (K-1))) :
  ∃ (T : finset (fin X)), T.card = k + 1 ∧ (∀ s ∈ T.powerset.filter (λ x, x.card = K), s ∈ S) :=
sorry

end exists_k_plus_one_element_subset_l400_400992


namespace solve_system_l400_400786

theorem solve_system (x y : ℝ) (h₁ : 3^y * 81 = 9^(x^2)) (h₂ : log 10 y = log 10 x - log 10 0.5) :
  x = 2 ∧ y = 4 :=
by
  sorry

end solve_system_l400_400786


namespace equation_of_tangent_circle_l400_400128

theorem equation_of_tangent_circle 
  (h_center : (1, -1)) 
  (h_line : ∀ (x y : ℝ), x + 2 = 0) : 
  ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 9 :=
  sorry

end equation_of_tangent_circle_l400_400128


namespace primes_between_50_and_60_l400_400525

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400525


namespace range_of_a_add_b_l400_400188

-- Define the problem and assumptions
variables (a b : ℝ)
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom ab_eq_a_add_b_add_3 : a * b = a + b + 3

-- Define the theorem to prove
theorem range_of_a_add_b : a + b ≥ 6 :=
sorry

end range_of_a_add_b_l400_400188


namespace root_of_quadratic_l400_400185

theorem root_of_quadratic (m : ℝ) (h : 3*1^2 - 1 + m = 0) : m = -2 :=
by {
  sorry
}

end root_of_quadratic_l400_400185


namespace count_primes_between_50_and_60_l400_400666

theorem count_primes_between_50_and_60 : 
  (Finset.filter Nat.prime (Finset.range 61 \ Finset.range 51)).card = 2 :=
by
  sorry

end count_primes_between_50_and_60_l400_400666


namespace primes_count_between_50_and_60_l400_400557

/-- A number is prime if it has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the set of numbers between 50 and 60. -/
def numbers_between_50_and_60 : finset ℕ :=
  finset.Ico 51 61

/-- The set of prime numbers between 50 and 60. -/
def primes_between_50_and_60 : finset ℕ :=
  numbers_between_50_and_60.filter is_prime

/-- The number of prime numbers between 50 and 60 is 2. -/
theorem primes_count_between_50_and_60 : primes_between_50_and_60.card = 2 :=
  sorry

end primes_count_between_50_and_60_l400_400557


namespace num_primes_50_60_l400_400278

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_primes_50_60 : finset.card (finset.filter is_prime (finset.range 61 \ finset.range 51)) = 2 :=
by
  sorry

end num_primes_50_60_l400_400278


namespace rounds_to_one_fifth_l400_400892

noncomputable def remaining_water (n : ℕ) : ℚ := 
  ∏ k in Finset.range n, (2 * (k + 1) : ℚ) / (2 * (k + 1) + 1)

theorem rounds_to_one_fifth :
  (remaining_water 6) = 1 / 5 :=
sorry

end rounds_to_one_fifth_l400_400892


namespace primes_between_50_and_60_l400_400397

theorem primes_between_50_and_60 : 
  let is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ¬∃ m < n, m > 1 ∧ n % m = 0 in
  let count_primes (a b : ℕ) : ℕ :=
    (a..b).filter is_prime).length in
  count_primes 51 59 = 2 :=
  sorry

end primes_between_50_and_60_l400_400397


namespace inequality_proof_l400_400725

theorem inequality_proof (n : ℕ) (x y : Fin n → ℝ)
  (h0 : ∀ i, 0 < x i)
  (hx : ∀ i, x (i + 1) ≤ (x i) / (i + 1))
  (hy : ∀ i j, j ≤ i → y i ≤ y j) :
  (∑ i in Finset.range n, x i * y i)^2 ≤
  (∑ i in Finset.range n, y i) * (∑ i in Finset.range n, (x i ^ 2 - 1/4 * x i * x (i - 1)) * y i) :=
by
  sorry

end inequality_proof_l400_400725


namespace smallest_integer_to_multiply_y_to_make_perfect_square_l400_400743

noncomputable def y : ℕ :=
  3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_multiply_y_to_make_perfect_square :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (k * y) = m^2) ∧ k = 3 := by
  sorry

end smallest_integer_to_multiply_y_to_make_perfect_square_l400_400743


namespace primes_between_50_and_60_l400_400421

open Nat

-- Define the range
def range : List ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

-- Define the primality check function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (m ∣ n)

-- Extract primes in the range
def primes_in_range : List ℕ :=
  range.filter is_prime

-- Define the proof goal
theorem primes_between_50_and_60 : primes_in_range.length = 2 :=
sorry

end primes_between_50_and_60_l400_400421


namespace price_of_uniform_l400_400034

theorem price_of_uniform (total_salary uniform_cost: ℝ) (service_time fraction_served rs_received: ℝ) 
    (h1: total_salary = 900)
    (h2: fraction_served = 9 / 12)
    (h3: rs_received = 650) 
    (h4: uniform_cost = total_salary * fraction_served - rs_received): 
  uniform_cost = 25 :=
by
  have h5 : fraction_served = 3 / 4 := by simp [h2]
  have h6 : total_salary * fraction_served = 900 * (3 / 4) := by rw [h1, h5]
  have h7 : 900 * (3 / 4) = 675 := by norm_num
  have h8 : uniform_cost = 675 - 650 := by rw [h4, h6, h3, h7]
  have h9 : 25 = 675 - 650 := by norm_num
  simp [h9, h8]

-- Provide sorry as the full proof is not required

end price_of_uniform_l400_400034


namespace primes_between_50_and_60_l400_400516

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : Nat) : Nat :=
  (Finset.range (b - a + 1)).filter (λ n => is_prime (n + a)).card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400516


namespace solve_quadratic_l400_400825

theorem solve_quadratic (x : ℝ) : (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3) :=
by
  sorry

end solve_quadratic_l400_400825


namespace problem_statement_l400_400159

theorem problem_statement (a b c : ℝ) (h₁ : a = 2 ^ 0.2) (h₂ : b = 0.4 ^ 0.2) (h₃ : c = 0.4 ^ 0.6) : a > b ∧ b > c :=
by {
  sorry
}

end problem_statement_l400_400159


namespace primes_between_50_and_60_l400_400297

theorem primes_between_50_and_60 : ∃ (S : Set ℕ), S = {53, 59} ∧ S.card = 2 :=
by
  sorry

end primes_between_50_and_60_l400_400297


namespace solve_quadratic_equation_l400_400827

theorem solve_quadratic_equation :
  ∀ (x : ℝ), ((x - 2) * (x + 3) = 0) ↔ (x = 2 ∨ x = -3) :=
by
  intro x
  sorry

end solve_quadratic_equation_l400_400827


namespace triangle_area_l400_400955

variable {x : ℝ}
variable {α : ℝ}
variable {CD BC AC AD : ℝ}
variable {cosα sinα : ℝ}
variable (h1 : CD = x)
variable (h2 : α = real.arccos (sqrt(2) / sqrt(3)))
variable (h3 : BC = 4 * x)
variable (h4 : cosα = sqrt(2) / sqrt(3))
variable (h5 : sinα = 1 / sqrt(3))
variable (h6 : AD = 3 / 4)

theorem triangle_area (x : ℝ) (α : ℝ) (BC : ℝ) (AC : ℝ) (AD : ℝ) (cosα : ℝ) (sinα : ℝ)
    (h1 : CD = x) (h2 : α = real.arccos (sqrt(2) / sqrt(3))) (h3 : BC = 4 * x)
    (h4 : cosα = sqrt(2) / sqrt(3)) (h5 : sinα = 1 / sqrt(3)) (h6 : AD = 3 / 4) : 
    real.sqrt(2) / 11 = S_ABC :=
by
  sorry

end triangle_area_l400_400955


namespace prime_count_between_50_and_60_l400_400584

-- Definitions:
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime

-- Statement:
theorem prime_count_between_50_and_60 : primes_between 50 60 = [53, 59] :=
by sorry

end prime_count_between_50_and_60_l400_400584


namespace good_numbers_l400_400900

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_odd_prime (n : ℕ) : Prop :=
  Prime n ∧ n % 2 = 1

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_divisor (d + 1) (n + 1)

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ n = 1 ∨ is_odd_prime n :=
sorry

end good_numbers_l400_400900


namespace min_value_y_l400_400855

theorem min_value_y (x : ℝ) : ∃ (x : ℝ), y = x^2 + 16 * x + 20 → y ≥ -44 :=
begin
  sorry
end

end min_value_y_l400_400855
