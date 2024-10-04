import Mathlib
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.EuclideanSpace
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.NormedSpace.LieGroup
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.ProbabilityTheory.ProbabilityMassFunction.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Eigenspace
import Mathlib.LinearAlgebra.Matrix
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.GCD.Basic
import Mathlib.NumberTheory.Powers
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Normal
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Geometry
import analysis.normed_space.real_inner_product
import data.real.basic

namespace sum_of_series_l452_452632

def telescoping_series_sum : ℕ → ℝ
| n := 1 / ((n + 1) * Real.sqrt n + n * Real.sqrt (n + 1))

-- Define the problem statement
theorem sum_of_series :
  (∑ k in Finset.range 99 + 1, telescoping_series_sum k) = 9/10 :=
sorry

end sum_of_series_l452_452632


namespace bill_head_circumference_l452_452328

theorem bill_head_circumference (jack_head_circumference charlie_head_circumference bill_head_circumference : ℝ) :
  jack_head_circumference = 12 →
  charlie_head_circumference = (1 / 2 * jack_head_circumference) + 9 →
  bill_head_circumference = (2 / 3 * charlie_head_circumference) →
  bill_head_circumference = 10 :=
by
  intro hj hc hb
  sorry

end bill_head_circumference_l452_452328


namespace trig_identity_l452_452619

theorem trig_identity
  (h₁ : Real.cos (80 * Real.pi / 180))
  (h₂ : Real.cos (160 * Real.pi / 180))
  (h₃ : Real.cos (70 * Real.pi / 180)) :
  (2 * h₁ + h₂) / h₃ = -Real.sqrt 3 :=
sorry

end trig_identity_l452_452619


namespace satisfying_integers_l452_452592

theorem satisfying_integers (a b : ℤ) :
  a^4 + (a + b)^4 + b^4 = x^2 → a = 0 ∧ b = 0 :=
by
  -- Proof is required to be filled in here.
  sorry

end satisfying_integers_l452_452592


namespace bamboo_consumption_correct_l452_452906

-- Define the daily bamboo consumption for adult and baby pandas
def adult_daily_bamboo : ℕ := 138
def baby_daily_bamboo : ℕ := 50

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total bamboo consumed by an adult panda in a week
def adult_weekly_bamboo := adult_daily_bamboo * days_in_week

-- Define the total bamboo consumed by a baby panda in a week
def baby_weekly_bamboo := baby_daily_bamboo * days_in_week

-- Define the total bamboo consumed by both pandas in a week
def total_bamboo_consumed := adult_weekly_bamboo + baby_weekly_bamboo

-- The theorem states that the total bamboo consumption in a week is 1316 pounds
theorem bamboo_consumption_correct : total_bamboo_consumed = 1316 := by
  sorry

end bamboo_consumption_correct_l452_452906


namespace initial_amount_l452_452792

variable (r : ℝ) (A : ℝ)

theorem initial_amount {P : ℝ} (h : r = 0.1) (hA : A = 99) : P = 90 :=
by
  have h1 : 1 + r = 1.1 := by rw [h]; norm_num
  have h2 : (1 + r) * P = A := by rw [h1]; sorry
  suffices P = 90 by exact this
  field_simp [h1, hA]
  linarith

end initial_amount_l452_452792


namespace factory_X_bulbs_percentage_l452_452941

theorem factory_X_bulbs_percentage (p : ℝ) (hx : 0.59 * p + 0.65 * (1 - p) = 0.62) : p = 0.5 :=
sorry

end factory_X_bulbs_percentage_l452_452941


namespace angle_representation_l452_452818

theorem angle_representation (k : ℤ) : 
  ∃ α : ℝ, let α := k * real.pi - real.pi / 4 in
  (∃ O : ℝ × ℝ, O = (0, 0)) ∧
  (∃ A : ℝ × ℝ, A = (1, 0)) ∧
  (∃ B : ℝ × ℝ, B = (-1 * 1, 1)) ∧
  α = k * real.pi - real.pi / 4 := 
sorry

end angle_representation_l452_452818


namespace B_card_le_k_l452_452737

variable (A : Type*) [Fintype A]
variable (rel : A → A → Prop)
variable (h1 : ∀ a b c : A, a ≠ b → rel a c → rel b c → (rel a b ∨ rel b a))
variable (B : Finset A)
variable (h2 : ∀ a ∈ (Finset.univ \ B), ∃ b ∈ B, rel a b ∨ rel b a)
variable (k : ℕ)
variable (h3 : ∀ S : Finset A, (∀ a b ∈ S, ¬ rel a b ∧ ¬ rel b a) → S.card ≤ k)

theorem B_card_le_k : B.card ≤ k :=
sorry

end B_card_le_k_l452_452737


namespace enclosed_area_l452_452795

noncomputable def region := { p : ℝ × ℝ | abs (p.1 - 1) + abs (p.2 - 1) = 1 }

theorem enclosed_area :
  let A : set (ℝ × ℝ) := region in
  measure_theory.measure_finset_volume (A) = 2 :=
begin
  sorry
end

end enclosed_area_l452_452795


namespace find_x_l452_452697

theorem find_x (x : ℝ) (hx : 0 < x) 
  (h : (Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) * Real.sqrt (2 * x) = 60)) : 
  x = 1 / (2 * Real.sqrt (Real.nthRoot 2 5)) :=
by
  sorry

end find_x_l452_452697


namespace intersection_is_zero_l452_452738

def M : Set ℝ := { x | |x - 3| < 4 }
def N : Set ℤ := { x | x^2 + x - 2 < 0 }

theorem intersection_is_zero : ∀ x, x ∈ (M ∩ N : Set ℝ) ↔ x = 0 := by
  sorry

end intersection_is_zero_l452_452738


namespace perimeter_difference_is_two_l452_452442

-- Define the first rectangle's dimensions
def rect1_length : ℕ := 5
def rect1_width : ℕ := 3

-- Define the second figure made of two overlapping rectangles
def rect2a_length : ℕ := 4
def rect2a_width : ℕ := 2
def rect2b_length : ℕ := 2
def rect2b_width : ℕ := 1
def overlap : ℕ := 1

-- Define the perimeters
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Calculate the perimeters
def P1 : ℕ := perimeter rect1_length rect1_width
def P2a : ℕ := perimeter rect2a_length rect2a_width
def P2b : ℕ := perimeter rect2b_length rect2b_width

-- The adjusted perimeter of the second figure due to overlap
def P2 : ℕ := (P2a - 2 * overlap) + (P2b - 2 * overlap)

-- The positive difference in perimeters
def difference : ℕ := abs (P1 - P2)

-- The Lean theorem statement
theorem perimeter_difference_is_two :
  difference = 2 := 
by
  sorry

end perimeter_difference_is_two_l452_452442


namespace carmen_candles_needed_l452_452105

-- Definitions based on the conditions

def candle_lifespan_1_hour : Nat := 8  -- a candle lasts 8 nights when burned 1 hour each night
def nights_total : Nat := 24  -- total nights

-- Question: How many candles are needed if burned 2 hours a night?

theorem carmen_candles_needed (h : candle_lifespan_1_hour / 2 = 4) :
  nights_total / 4 = 6 := 
  sorry

end carmen_candles_needed_l452_452105


namespace layoffs_payment_l452_452269

theorem layoffs_payment :
  let total_employees := 450
  let salary_2000_employees := 150
  let salary_2500_employees := 200
  let salary_3000_employees := 100
  let first_round_2000_layoffs := 0.20 * salary_2000_employees
  let first_round_2500_layoffs := 0.25 * salary_2500_employees
  let first_round_3000_layoffs := 0.15 * salary_3000_employees
  let remaining_2000_after_first_round := salary_2000_employees - first_round_2000_layoffs
  let remaining_2500_after_first_round := salary_2500_employees - first_round_2500_layoffs
  let remaining_3000_after_first_round := salary_3000_employees - first_round_3000_layoffs
  let second_round_2000_layoffs := 0.10 * remaining_2000_after_first_round
  let second_round_2500_layoffs := 0.15 * remaining_2500_after_first_round
  let second_round_3000_layoffs := 0.05 * remaining_3000_after_first_round
  let remaining_2000_after_second_round := remaining_2000_after_first_round - second_round_2000_layoffs
  let remaining_2500_after_second_round := remaining_2500_after_first_round - second_round_2500_layoffs
  let remaining_3000_after_second_round := remaining_3000_after_first_round - second_round_3000_layoffs
  let total_payment := remaining_2000_after_second_round * 2000 + remaining_2500_after_second_round * 2500 + remaining_3000_after_second_round * 3000
  total_payment = 776500 := sorry

end layoffs_payment_l452_452269


namespace greatest_ratio_l452_452936

-- Definitions of points lying on the circle with integer coordinates
def point (x y : ℤ) : Prop := x^2 + y^2 = 169

-- Points P, Q, R, S having integer coordinates on circle x^2 + y^2 = 169
def P : Prop := point (-12) 5
def Q : Prop := point 5 (-12)
def R : Prop := point 5 12
def S : Prop := point 12 5

-- Distance formula
def distance (x1 y1 x2 y2 : ℤ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Definition of distances PQ and RS
def PQ : ℝ := distance (-12) 5 5 (-12)
def RS : ℝ := distance 5 12 12 5

-- Main theorem statement
theorem greatest_ratio : (PQ / RS) = 2.4 :=
by
  sorry

end greatest_ratio_l452_452936


namespace factorize_expression_l452_452607

theorem factorize_expression (x : ℝ) : 2 * x ^ 3 - 4 * x ^ 2 - 6 * x = 2 * x * (x - 3) * (x + 1) :=
by
  sorry

end factorize_expression_l452_452607


namespace cos_a2_plus_a8_eq_neg_half_l452_452652

noncomputable def a_n (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem cos_a2_plus_a8_eq_neg_half 
  (a₁ d : ℝ) 
  (h : a₁ + a_n 5 a₁ d + a_n 9 a₁ d = 5 * Real.pi)
  : Real.cos (a_n 2 a₁ d + a_n 8 a₁ d) = -1 / 2 :=
by
  sorry

end cos_a2_plus_a8_eq_neg_half_l452_452652


namespace triangle_angle_B_l452_452425

variables {a b c : ℝ} 

theorem triangle_angle_B (h : (c^2 / (a + b)) + (a^2 / (b + c)) = b) : 
  ∃ B : ℝ, B = 60 ∧ angle_opposite_side_b a b c B :=
begin
  sorry
end

end triangle_angle_B_l452_452425


namespace sugar_for_recipe_l452_452534

theorem sugar_for_recipe (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by
  sorry

end sugar_for_recipe_l452_452534


namespace exact_fare_impossible_less_coins_exact_fare_possible_exact_coins_l452_452780

noncomputable def exact_fare (n : ℕ) : Prop :=
  ∃ (coins : ℕ), (coins < 5 * n) ∧ ( ∀ fare, fare = 5 * n → ∃ c, c = 10 ∧ exact (fare - c) = 5 * n )

theorem exact_fare_impossible_less_coins (k : ℕ) (h₁ : 10 ∣ k) : ¬ exact_fare (k - 1) :=
sorry

theorem exact_fare_possible_exact_coins (k : ℕ) (h₁ : 10 ∣ k) : exact_fare k :=
sorry

end exact_fare_impossible_less_coins_exact_fare_possible_exact_coins_l452_452780


namespace rent_percentage_l452_452733

noncomputable def condition1 (E : ℝ) : ℝ := 0.25 * E
noncomputable def condition2 (E : ℝ) : ℝ := 1.35 * E
noncomputable def condition3 (E' : ℝ) : ℝ := 0.40 * E'

theorem rent_percentage (E R R' : ℝ) (hR : R = condition1 E) (hE' : E = condition2 E) (hR' : R' = condition3 E) :
  (R' / R) * 100 = 216 :=
sorry

end rent_percentage_l452_452733


namespace center_number_in_grid_is_3_l452_452083

theorem center_number_in_grid_is_3 :
  ∃ (grid : matrix (fin 3) (fin 3) ℕ),
    (∀ i j, grid i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ -- all numbers from 1 to 9 are used
    (∀ i j k l, grid i j = grid k l → (i, j) = (k, l)) ∧ -- all numbers are unique
    (∀ i j, (i > 0 → grid i j.succ = grid (i - 1) j) ∧ 
            (i < 2 → grid i j = grid (i + 1) j)) ∧ -- consecutive numbers share an edge
    (grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 20) → -- sum of corners is 20
    grid 1 1 = 3 := -- the center number is 3
sorry

end center_number_in_grid_is_3_l452_452083


namespace problem_statement_l452_452922

theorem problem_statement :
  (∏ k in finset.range 1 12, 1 + (13 / k) : ℝ) / (∏ k in finset.range 1 10, 1 + (15 / k) : ℝ) = 1 := 
by
  sorry

end problem_statement_l452_452922


namespace find_x_l452_452643

open Real

noncomputable def sequence (x : ℝ) : List ℝ := (List.range 51).map (λ n, x^n)

def A (seq : List ℝ) : List ℝ :=
  (List.range (seq.length - 1)).map (λ n, (seq.nth n.getOrElse 0 + seq.nth (n + 1).getOrElse 0) / 2)

def A_iter (m : ℕ) (seq : List ℝ) : List ℝ :=
  Nat.iterate m A seq

theorem find_x
  (S : List ℝ)
  (x : ℝ)
  (h1 : sequence x = S)
  (h2 : x > 0)
  (h3 : A_iter 50 S = [1 / 2^25]) :
  x = sqrt 2 - 1 := sorry

end find_x_l452_452643


namespace product_A_percent_decrease_product_B_percent_decrease_product_C_percent_decrease_l452_452888

def percentage_decrease (first_ratio next_ratio : ℚ) : ℚ :=
  ((first_ratio - next_ratio) / first_ratio) * 100

theorem product_A_percent_decrease :
  percentage_decrease (8 / 20 : ℚ) (9 / 108 : ℚ) ≈ 79.175 := sorry

theorem product_B_percent_decrease :
  percentage_decrease (10 / 25 : ℚ) (12 / 150 : ℚ) = 80 := sorry

theorem product_C_percent_decrease :
  percentage_decrease (5 / 10 : ℚ) (6 / 80 : ℚ) = 85 := sorry

end product_A_percent_decrease_product_B_percent_decrease_product_C_percent_decrease_l452_452888


namespace more_flour_than_sugar_l452_452384

def cups_of_flour : Nat := 9
def cups_of_sugar : Nat := 6
def flour_added : Nat := 2
def flour_needed : Nat := cups_of_flour - flour_added -- 9 - 2 = 7

theorem more_flour_than_sugar : flour_needed - cups_of_sugar = 1 :=
by
  sorry

end more_flour_than_sugar_l452_452384


namespace number_of_selected_in_interval_l452_452254

noncomputable def systematic_sampling_group := (420: ℕ)
noncomputable def selected_people := (21: ℕ)
noncomputable def interval_start := (241: ℕ)
noncomputable def interval_end := (360: ℕ)
noncomputable def sampling_interval := systematic_sampling_group / selected_people
noncomputable def interval_length := interval_end - interval_start + 1

theorem number_of_selected_in_interval :
  interval_length / sampling_interval = 6 :=
by
  -- Placeholder for the proof
  sorry

end number_of_selected_in_interval_l452_452254


namespace arithmetic_mean_of_primes_l452_452961

open Real

theorem arithmetic_mean_of_primes :
  let numbers := [33, 37, 39, 41, 43]
  let primes := numbers.filter Prime
  let sum_primes := (37 + 41 + 43 : ℤ)
  let count_primes := (3 : ℤ)
  let mean := (sum_primes / count_primes : ℚ)
  mean = 40.33 := by
sorry

end arithmetic_mean_of_primes_l452_452961


namespace manny_marbles_l452_452446

theorem manny_marbles (total_marbles : ℕ) (ratio_m : ℕ) (ratio_n : ℕ) (manny_gives : ℕ) 
  (h_total : total_marbles = 36) (h_ratio_m : ratio_m = 4) (h_ratio_n : ratio_n = 5) (h_manny_gives : manny_gives = 2) : 
  (total_marbles * ratio_n / (ratio_m + ratio_n)) - manny_gives = 18 :=
by
  sorry

end manny_marbles_l452_452446


namespace second_smallest_pack_count_l452_452129

theorem second_smallest_pack_count : ∃ m : ℕ, (∃ j : ℤ, m = 7 * j + 3) ∧ (m % 7 = 6) ∧ m = 10 :=
by
  exists 10
  split
  exists 1
  norm_num
  split
  norm_num
  refl

end second_smallest_pack_count_l452_452129


namespace total_meals_per_week_l452_452679

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end total_meals_per_week_l452_452679


namespace find_smallest_n_right_angled_l452_452353

/-- 
  Define the initial triangle with specified angles.
  Represent the iterative steps constructing feet of altitudes.
  Prove that the smallest n such that the resulting triangle is right-angled is 13.
-/
structure Triangle :=
  (A B C : Type)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

def initial_triangle : Triangle :=
  { A := Unit, B := Unit, C := Unit, angle_A := 58, angle_B := 61, angle_C := 61 }

def altitude_foot (prev_triangle : Triangle) (n : ℕ) : Triangle :=
  sorry -- Define the recursion for feet of altitudes

noncomputable def resulting_triangle (n : ℕ) : Triangle :=
  nat.rec_on n initial_triangle (fun n prev_triangle => altitude_foot prev_triangle n)

def is_right_angled (T : Triangle) : Prop :=
  T.angle_A = 90 ∨ T.angle_B = 90 ∨ T.angle_C = 90

theorem find_smallest_n_right_angled : ∃ n: ℕ, n > 0 ∧ is_right_angled (resulting_triangle n) ∧ (∀ m: ℕ, m < n → ¬is_right_angled (resulting_triangle m)) :=
by {
  use 13,
  split,
  { exact nat.succ_pos 12 },
  split,
  { sorry }, -- Prove that the triangle is right-angled when n = 13
  { intros m hm,
    sorry }, -- Prove there is no right-angled triangle for all m < 13
}

end find_smallest_n_right_angled_l452_452353


namespace five_point_eight_one_million_in_scientific_notation_l452_452174

theorem five_point_eight_one_million_in_scientific_notation :
  5.81 * 10^6 = 5.81e6 :=
sorry

end five_point_eight_one_million_in_scientific_notation_l452_452174


namespace maximum_zero_coefficients_l452_452890

-- The defined polynomial P(x) has degree 10 and three distinct roots
noncomputable def P (x : ℝ) : ℝ := x^10 - x^8 - 0 * x^9 + 0 * x^7 + 0 * x^6 + 0 * x^5 + 
                                 0 * x^4 + 0 * x^3 + 0 * x^2 + 0 * x + 0

-- The main theorem stating the maximum number of zero coefficients
theorem maximum_zero_coefficients : 
  ∃ (P : ℝ → ℝ) (deg : ℕ) (coeff_count : ℕ), 
    deg = 10 ∧ 
    (∀ x : ℝ, P x = 0 → x = 0 ∨ x = 1 ∨ x = -1) → 
    coeff_count = 9 :=
begin
  sorry
end

end maximum_zero_coefficients_l452_452890


namespace product_of_midpoint_is_minus_4_l452_452481

-- Coordinates of the endpoints
def endpoint1 : ℝ × ℝ := (4, -3)
def endpoint2 : ℝ × ℝ := (-8, 7)

-- Function to compute the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Coordinates of the midpoint
def midpoint_coords := midpoint endpoint1 endpoint2

-- Product of the coordinates of the midpoint
def product_of_midpoint_coords (mp : ℝ × ℝ) : ℝ :=
  mp.1 * mp.2

-- Statement of the theorem to be proven
theorem product_of_midpoint_is_minus_4 : 
  product_of_midpoint_coords midpoint_coords = -4 := 
by
  sorry

end product_of_midpoint_is_minus_4_l452_452481


namespace prime_list_mean_l452_452955

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def prime_list_mean_proof : Prop :=
  let nums := [33, 37, 39, 41, 43] in
  let primes := filter is_prime nums in
  primes = [37, 41, 43] ∧
  (primes.sum : ℤ) / primes.length = 40.33

theorem prime_list_mean : prime_list_mean_proof :=
by
  sorry

end prime_list_mean_l452_452955


namespace greatest_integer_less_PS_l452_452288

theorem greatest_integer_less_PS 
  (PQ PS T : ℝ)
  (midpoint_TPS : T = PS / 2)
  (perpendicular_PT_QT : (PQ ^ 2 = (PS / 2) ^ 2 + (PS / 2) ^ 2))
  (PQ_value : PQ = 150) :
  ⌊ PS ⌋ = 212 :=
by
  sorry

end greatest_integer_less_PS_l452_452288


namespace permutation_exists_25_permutation_exists_1000_l452_452325

-- Define a function that checks if a permutation satisfies the condition
def valid_permutation (perm : List ℕ) : Prop :=
  (∀ i < perm.length - 1, let diff := (perm[i] - perm[i+1]).abs in diff = 3 ∨ diff = 5)

-- Proof problem for n = 25
theorem permutation_exists_25 : 
  ∃ perm : List ℕ, perm.perm (List.range 25).map (· + 1) ∧ valid_permutation perm := 
sorry

-- Proof problem for n = 1000
theorem permutation_exists_1000 : 
  ∃ perm : List ℕ, perm.perm (List.range 1000).map (· + 1) ∧ valid_permutation perm := 
sorry

end permutation_exists_25_permutation_exists_1000_l452_452325


namespace find_min_value_of_expression_l452_452749

noncomputable def min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
    (h : (2 / (x + 4)) + (1 / (y + 3)) = 1 / 4) : ℝ :=
3 * x + y

theorem find_min_value_of_expression :
    ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ (2 / (x + 4) + 1 / (y + 3) = 1 / 4) ∧
    min_value_of_expression x y (sorry) (sorry) (sorry) = -8 + 20 * real.sqrt 2 :=
sorry

end find_min_value_of_expression_l452_452749


namespace linda_hourly_rate_l452_452371

def babysitting_rate_per_hour (application_fee_per_college : ℕ) (num_colleges : ℕ) (hours_babysitting : ℕ) : ℕ :=
(application_fee_per_college * num_colleges) / hours_babysitting

theorem linda_hourly_rate :
  babysitting_rate_per_hour 25 6 15 = 10 :=
by
  unfold babysitting_rate_per_hour
  norm_num
  sorry

end linda_hourly_rate_l452_452371


namespace inconsistency_l452_452732

noncomputable def joshua_rocks := sorry
noncomputable def jose_rocks := joshua_rocks - 14
noncomputable def albert_rocks := jose_rocks + 20

theorem inconsistency : joshua_rocks + 6 = albert_rocks → False :=
by
  sorry

end inconsistency_l452_452732


namespace collinear_points_division_l452_452112

variable (a b c : ℝ)
variable (A : Point a 0)
variable (B : Point b b)
variable (C : Point c (2 * c))
variable (D : Point a a)
variable (E : Point (2 * a) (2 * a))
variable (F : Point (2 * a) (4 * a))

theorem collinear_points_division (h1 : A.x = a ∧ A.y = 0)
    (h2 : B.y = B.x ∧ B ∈ (Segment A C))
    (h3 : C.y = 2 * C.x ∧ C ∈ (Segment A B))
    (h4 : ∥A - B∥ / ∥B - C∥ = 2)
    (h5 : D = (a, a))
    (h6 : E ∈ (Circumcircle (Triangle (A) (D) (C))) ∧ E.y = E.x)
    (h7 : Intersects (Ray (A) (E)) (y = 2 * x) = F) :
    ∥A - E∥ / ∥E - F∥ = Real.sqrt(2) / 2 := by
sorry

end collinear_points_division_l452_452112


namespace find_matrix_M_find_line_l_l452_452995

open Matrix

-- Given conditions
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ := !![6, 2; 4, 4]
def eigenvector : Fin 2 → ℝ := λ i, if i = 0 then 1 else 1
def point_A : Fin 2 → ℝ := λ i, if i = 0 then -1 else 2
def point_A_prime : Fin 2 → ℝ := λ i, if i = 0 then -2 else 4
def line_m (x y : ℝ) : Prop := x - y = 6

-- Matrix satisfies eigenvalue-eigenvector equation
def matrix_eigenvalue : Prop := 
  ∃ λ = 8, (matrix_M.mulVec eigenvector = λ • eigenvector)

-- Matrix transforms point A to A'
def transformation_A_to_A_prime : Prop := 
  matrix_M.mulVec point_A = point_A_prime

-- Find matrix M and verify its characteristics
theorem find_matrix_M : matrix_M = !![6, 2; 4, 4]
∧ matrix_eigenvalue
∧ transformation_A_to_A_prime := 
begin
  sorry -- Proof is not required
end

-- Find the original line l given M^-1 transforms l to m
def line_l (x y : ℝ) : Prop := x - y = 12

-- Statement considering the inverse transformation
theorem find_line_l (x y : ℝ) : Prop :=
  (M⁻¹).mulVec !![x, y] = !![x_m', y_m'] →
  (line_m x_m' y_m') → (line_l x y) := 
begin
  sorry -- Proof is not required
end

end find_matrix_M_find_line_l_l452_452995


namespace arithmetic_sequence_seventh_term_l452_452660

variable {α : Type*} [LinearOrderedField α]

variable (a : ℕ → α)

def sum_first_n (n : ℕ) (a : ℕ → α) := ∑ i in finset.range n, a (i + 1)

theorem arithmetic_sequence_seventh_term (h_sum : sum_first_n 10 a = 165) (h_a4 : a 4 = 12) : a 7 = 21 := 
by
  sorry

end arithmetic_sequence_seventh_term_l452_452660


namespace sample_size_l452_452879

theorem sample_size (k n : ℕ) (h_ratio : 3 * n / (3 + 4 + 7) = 9) : n = 42 :=
by
  sorry

end sample_size_l452_452879


namespace find_minimum_f_l452_452152

noncomputable def f (x : ℝ) : ℝ :=
x^2 + 2 * x / (x^2 + 1) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem find_minimum_f (x : ℝ) (hx : x > 0) : 
  ∃ y, ∀ z, z > 0 → f z ≥ y :=
begin
  sorry
end

end find_minimum_f_l452_452152


namespace swimmers_speed_in_still_water_l452_452074

theorem swimmers_speed_in_still_water
  (v : ℝ) -- swimmer's speed in still water
  (current_speed : ℝ) -- speed of the water current
  (time : ℝ) -- time taken to swim against the current
  (distance : ℝ) -- distance swum against the current
  (h_current_speed : current_speed = 2)
  (h_time : time = 3.5)
  (h_distance : distance = 7)
  (h_eqn : time = distance / (v - current_speed)) :
  v = 4 :=
by
  sorry

end swimmers_speed_in_still_water_l452_452074


namespace smaller_of_two_digit_product_l452_452429

theorem smaller_of_two_digit_product (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha' : a < 100) (hb' : b < 100) 
  (hprod : a * b = 4680) : min a b = 52 :=
by
  sorry

end smaller_of_two_digit_product_l452_452429


namespace length_PN_l452_452186

noncomputable def parabola_C (x y : ℝ) (p : ℝ) (hp : p > 0) : Prop :=
  y^2 = 2 * p * x

noncomputable def focus_F : Prop :=
  (1,0) ∈ {F : Π (x y : ℝ), F x y}

noncomputable def directrix (x : ℝ) : Prop :=
  x = -1

noncomputable def point_on_directrix (M : Π (x y : ℝ), Prop) : Prop :=
  ∃ M, M.1 = -1 ∧ M.2 < 0

noncomputable def line_passing_through_MF (l : Π (x y : ℝ), Prop) : Prop :=
  ∃ M F, M.1 = -1 ∧ M.2 < 0 ∧ F = (1,0) ∧ l = λ x y, ∃ a, y = a * (x - 1)

noncomputable def midpoint (F M N : Π (x y : ℝ), Prop) : Prop :=
  F = (1,0) ∧ ∃ M N, F.1 = (M.1 + N.1) / 2 ∧ F.2 = (M.2 + N.2) / 2

noncomputable def line_intersects_parabola (l : Π (x y : ℝ), Prop) (C : Π (x y : ℝ), Prop) : Prop :=
  ∃ N P, l = λ x y, y = sqrt(3) * (x - 1) ∧ C = λ x y, y^2 = 4 * x ∧ 
  N = (3, 2 * sqrt(3)) ∧ P = (1/3, -2 * sqrt(3) / 3)

noncomputable def length_segment_PN (N P : Π (x y : ℝ), Prop) : ℝ :=
  sqrt((P.1 - N.1)^2 + (P.2 - N.2)^2)

theorem length_PN : ∀ (p : ℝ) (hp : p > 0) (M F N P : Π (x y : ℝ), Prop) (l C : Π (x y : ℝ), Prop),
  parabola_C (N.fst) (N.snd) p hp →
  focus_F → 
  directrix M.fst → 
  point_on_directrix M →
  line_passing_through_MF l →
  midpoint F M N →
  line_intersects_parabola l C →
  length_segment_PN N P = 16/3 :=
begin
  sorry
end

end length_PN_l452_452186


namespace greatest_integer_less_than_PS_l452_452282

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ), PQ = 150 → 
  PS = 150 * Real.sqrt 3 → 
  (⌊PS⌋ = 259) := 
by
  intros PQ PS hPQ hPS
  sorry

end greatest_integer_less_than_PS_l452_452282


namespace train_stops_15_minutes_per_hour_l452_452524

def train_speed_including_stoppages : ℕ := 90
def train_speed_excluding_stoppages : ℕ := 120

theorem train_stops_15_minutes_per_hour (v_incl v_excl : ℕ) (h1 : v_incl = train_speed_including_stoppages) (h2 : v_excl = train_speed_excluding_stoppages) :
  let distance_due_to_stoppages := v_excl - v_incl in
  let time_without_stoppages_km_per_min := v_excl / 60 in
  let stopping_time := distance_due_to_stoppages / time_without_stoppages_km_per_min in
  stopping_time = 15 :=
by
  sorry

end train_stops_15_minutes_per_hour_l452_452524


namespace revenue_from_small_lemonades_l452_452453

theorem revenue_from_small_lemonades (S : ℝ) (total_revenue : ℝ) (medium_revenue : ℝ) (num_large_cups : ℕ) (price_large_cup : ℝ) :
  total_revenue = S + medium_revenue + (num_large_cups * price_large_cup) →
  (num_large_cups = 5) →
  (price_large_cup = 3) →
  (medium_revenue = 24) →
  (total_revenue = 50) →
  S = 11 :=
by
  intros h1 h2 h3 h4 h5
  rw [h2, h3] at h1
  rw [h4] at h1
  rw [h5] at h1
  exact eq_of_add_eq_left h1


end revenue_from_small_lemonades_l452_452453


namespace abs_value_difference_l452_452694

theorem abs_value_difference (x y : ℤ) (h1 : |x| = 7) (h2 : |y| = 9) (h3 : |x + y| = -(x + y)) :
  x - y = 16 ∨ x - y = -16 :=
sorry

end abs_value_difference_l452_452694


namespace carolyn_flute_practice_l452_452577

theorem carolyn_flute_practice : 
  (∀ (days_in_month even_days odd_days weekend_days : ℕ) 
      (piano_even piano_odd violin_factor flute_divisor : ℝ),
    days_in_month = 30 → 
    even_days = 15 →
    odd_days = 15 →
    weekend_days = 8 →
    piano_even = 25 →
    piano_odd = 30 →
    violin_factor = 3 →
    flute_divisor = 2 →
      let total_piano := even_days * piano_even + odd_days * piano_odd in
      let total_violin := total_piano * violin_factor in
      let avg_violin_per_day := total_violin / (even_days + odd_days) in
      let total_flute := weekend_days * (avg_violin_per_day / flute_divisor) in
      total_flute = 330) :=
begin
  intros days_in_month even_days odd_days weekend_days piano_even piano_odd violin_factor flute_divisor,
  assume h_days_in_month h_even_days h_odd_days h_weekend_days h_piano_even h_piano_odd h_violin_factor h_flute_divisor,
  simp only [h_days_in_month, h_even_days, h_odd_days, h_weekend_days, h_piano_even, h_piano_odd, h_violin_factor, h_flute_divisor],
  let total_piano := (even_days * piano_even + odd_days * piano_odd),
  let total_violin := total_piano * violin_factor,
  let avg_violin_per_day := total_violin / (even_days + odd_days),
  let total_flute := weekend_days * (avg_violin_per_day / flute_divisor),
  rw [h_even_days, h_odd_days],
  have h_total_piano : total_piano = 825 := by norm_num,
  have h_total_violin : total_violin = 2475 := by norm_num [h_total_piano],
  have h_avg_violin_per_day : avg_violin_per_day = 82.5 := by norm_num [h_total_violin],
  have h_total_flute : total_flute = 330 := by norm_num [h_avg_violin_per_day, h_weekend_days],
  exact h_total_flute
end

end carolyn_flute_practice_l452_452577


namespace tim_total_points_l452_452071

-- Definitions based on the conditions
def points_single : ℕ := 1000
def points_tetris : ℕ := 8 * points_single
def singles_scored : ℕ := 6
def tetrises_scored : ℕ := 4

-- Theorem stating the total points scored by Tim
theorem tim_total_points : singles_scored * points_single + tetrises_scored * points_tetris = 38000 := by
  sorry

end tim_total_points_l452_452071


namespace johns_age_13_l452_452338

def johns_age (days_worked : ℕ) (total_earning : ℝ) : ℕ :=
  let weekly_bonus := 5 in
  let weekly_working_days := 5 in
  let daily_hours := 3 in
  let pay_per_hour_per_year := 0.5 in
  let working_weeks := days_worked / weekly_working_days in
  let total_bonus := working_weeks * weekly_bonus in
  let adjusted_earning := total_earning - total_bonus in
  let daily_earning age := daily_hours * pay_per_hour_per_year * age in
  let total_days := 75 in
  let solve_age age := adjusted_earning / (daily_earning age * total_days) in
  if solve_age 13 == 1 then 13 else 0 -- only 13 is the solution in this logic.

theorem johns_age_13 :
  johns_age 75 900 = 13 :=
by
  unfold johns_age
  sorry

end johns_age_13_l452_452338


namespace gcd_of_256_180_600_l452_452835

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l452_452835


namespace probability_all_girls_l452_452540

theorem probability_all_girls (total_members boys girls : ℕ) (total_members = 15) (boys = 6) (girls = 9)
    (choose_3_total : ℕ := (Nat.choose total_members 3)) (choose_3_girls : ℕ := (Nat.choose girls 3)) :
    (choose_3_total = 455) →
    (choose_3_girls = 84) →
    (choose_3_girls.toRat / choose_3_total.toRat = (12 / 65) : ℚ) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end probability_all_girls_l452_452540


namespace time_via_route_B_l452_452731

-- Given conditions
def time_via_route_A : ℕ := 5
def time_saved_round_trip : ℕ := 6

-- Defining the proof problem
theorem time_via_route_B : time_via_route_A - (time_saved_round_trip / 2) = 2 :=
by
  -- Expected proof here
  sorry

end time_via_route_B_l452_452731


namespace min_value_expr_l452_452356

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / b + b / c + c / a + real.sqrt ((a / b)^2 + (b / c)^2 + (c / a)^2)) ≥ 3 + real.sqrt 3 := 
sorry

end min_value_expr_l452_452356


namespace cuboid_surface_area_from_cubes_l452_452877

theorem cuboid_surface_area_from_cubes
  (cube_length : ℝ)
  (num_cubes : ℕ)
  (cube_length_eq : cube_length = 8)
  (num_cubes_eq : num_cubes = 3) :
  let L := num_cubes * cube_length,
      W := cube_length,
      H := cube_length,
      SA := 2 * (L * W + L * H + W * H)
  in SA = 896 := 
sorry

end cuboid_surface_area_from_cubes_l452_452877


namespace mary_has_more_l452_452379

theorem mary_has_more (marco_initial mary_initial : ℕ) (h1 : marco_initial = 24) (h2 : mary_initial = 15) :
  let marco_final := marco_initial - 12,
      mary_final := mary_initial + 12 - 5 in
  mary_final = marco_final + 10 :=
by
  sorry

end mary_has_more_l452_452379


namespace draw_balls_ways_l452_452704

theorem draw_balls_ways :
  let balls := {balls | (count b in balls = 2) ∧ (count w in balls = 6) ∧ (length balls = 8)},
      draw_two, -- function that handles drawing two balls without replacement
      draw := (draw_two balls) 
  in
  count_ways draw (2 black balls) = 10 :=
by sorry

end draw_balls_ways_l452_452704


namespace false_proposition_p_and_q_l452_452674

open Classical

-- Define the propositions
def p (a b c : ℝ) : Prop := b * b = a * c
def q (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- We provide the conditions specified in the problem
variable (a b c : ℝ)
variable (f : ℝ → ℝ)
axiom hq : ∀ x, f x = f (-x)
axiom hp : ¬ (∀ a b c, p a b c ↔ (b ≠ 0 ∧ c ≠ 0 ∧ b * b = a * c))

-- The false proposition among the given options is "p and q"
theorem false_proposition_p_and_q : ¬ (∀ a b c (f : ℝ → ℝ), p a b c ∧ q f) :=
by
  -- This is where the proof would go, but is marked as a placeholder
  sorry

end false_proposition_p_and_q_l452_452674


namespace ratio_of_profits_l452_452431

-- Define the investments and time durations
variables (p_investment q_investment p_time q_time : ℕ)
variables (p_ratio q_ratio : ℕ)

-- Conditions
def investment_ratio_condition (p_ratio q_ratio : ℕ) := p_ratio = 7 ∧ q_ratio = 5
def time_duration_condition (p_time q_time : ℕ) := p_time = 20 ∧ q_time = 40

-- Calculate products
def product_p (p_investment p_time : ℕ) := p_investment * p_time
def product_q (q_investment q_time : ℕ) := q_investment * q_time

-- Define investments using ratios
def investment_p (x : ℕ) := 7 * x
def investment_q (x : ℕ) := 5 * x

-- Main theorem: Ratio of profits is 7 : 10
theorem ratio_of_profits (x : ℕ) :
  investment_ratio_condition 7 5 →
  time_duration_condition 20 40 →
  let p_investment := investment_p x in
  let q_investment := investment_q x in
  product_p p_investment 20 * 200 = product_q q_investment 40 * 140 :=
sorry

end ratio_of_profits_l452_452431


namespace num_true_propositions_l452_452905

-- Define the four propositions
def prop1 : Prop := ∀ (a b : ℝ), consecutive_interior_angles a b → supplementary a b
def prop2 : Prop := ∀ (n : ℝ), n < 1 → n^2 - 1 < 0
def prop3 : Prop := ∀ (a b : ℝ), is_right_angle a → is_right_angle b → a = b
def prop4 : Prop := ∀ (a b : ℝ), equal_angles a b → vertical_angles a b

-- Define the main theorem to prove the number of true propositions is 1
theorem num_true_propositions : 
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) → ∃ k : ℕ, k = 1 :=
by
  intro h
  -- Here we would provide the appropriate proof
  sorry

end num_true_propositions_l452_452905


namespace arithmetic_mean_prime_numbers_l452_452962

-- Define the list of numbers
def num_list : List ℕ := [33, 37, 39, 41, 43]

-- Define a predicate for primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Extract list of prime numbers
def prime_numbers : List ℕ := num_list.filter is_prime

-- Compute the arithmetic mean of a list of natural numbers
def arithmetic_mean (nums : List ℕ) : ℚ :=
  nums.foldr Nat.add 0 / nums.length

-- The main theorem: Proving the arithmetic mean of prime numbers in the list
theorem arithmetic_mean_prime_numbers :
  arithmetic_mean prime_numbers = 121 / 3 :=
by 
  -- Include proof steps here
  sorry

end arithmetic_mean_prime_numbers_l452_452962


namespace find_angle_B_find_area_ABC_l452_452263

-- Define the geometric context and conditions for the problem
variables (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ)
variables (sin : ℝ → ℝ) (cos : ℝ → ℝ) (tan : ℝ → ℝ)

-- Establish conditions as hypothesis
hypothesis h1 : b * sin A = sqrt 3 * a * cos B
hypothesis h2 : b = 3
hypothesis h3 : sin C = 2 * sin A

-- Define the geometric and trigonometric relationships
def angle_B : Prop := B = π / 3
def area_ABC : ℝ := 1 / 2 * a * 2 * a * sin (π / 3)

-- The required lean statements
theorem find_angle_B (B : ℝ) (sin cos tan : ℝ → ℝ) (h1 : b * sin A = sqrt 3 * a * cos B) :
  B = π / 3 := sorry

theorem find_area_ABC (a b c : ℕ) (A B C : ℝ) (sin cos tan : ℝ → ℝ)
  (h2 : b = 3) (h3 : sin C = 2 * sin A) :
  1 / 2 * a * 2 * a * sin (π / 3) = 3 * sqrt 3 / 2 := sorry

end find_angle_B_find_area_ABC_l452_452263


namespace probability_xy_minus_x_minus_y_even_l452_452465

open Nat

theorem probability_xy_minus_x_minus_y_even :
  let S := {1,2,3,4,5,6,7,8,9,10,11,12}
  let evens := {2, 4, 6, 8, 10, 12}
  let total_pairs := (Finset.card S).choose 2
  let even_pairs := (Finset.card evens).choose 2
  even_pairs / total_pairs = 5 / 22 :=
by
  sorry

end probability_xy_minus_x_minus_y_even_l452_452465


namespace correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l452_452033

-- Define the constants k and b
variables (k b : ℝ)

-- Define the function y = k * t + b
def linear_func (t : ℝ) : ℝ := k * t + b

-- Define the data points as conditions
axiom data_point1 : linear_func k b 1 = 7
axiom data_point2 : linear_func k b 2 = 12
axiom data_point3 : linear_func k b 3 = 17
axiom data_point4 : linear_func k b 4 = 22
axiom data_point5 : linear_func k b 5 = 27

-- Define the water consumption rate and total minutes in a day
def daily_water_consumption : ℝ := 1500
def minutes_in_one_day : ℝ := 1440
def days_in_month : ℝ := 30

-- The expression y = 5t + 2
theorem correct_functional_relationship : (k = 5) ∧ (b = 2) :=
by
  sorry

-- Estimated water amount at the 20th minute
theorem water_amount_20th_minute (t : ℝ) (ht : t = 20) : linear_func 5 2 t = 102 :=
by
  sorry

-- The water leaked in a month (30 days) can supply the number of days
theorem water_amount_supply_days : (linear_func 5 2 (minutes_in_one_day * days_in_month)) / daily_water_consumption = 144 :=
by
  sorry

end correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l452_452033


namespace arithmetic_mean_of_primes_l452_452951

-- Define the list of numbers
def num_list : List ℕ := [33, 37, 39, 41, 43]

-- Define a predicate that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the list of prime numbers extracted from the list
def prime_list : List ℕ := (num_list.filter is_prime)

-- Define the sum of the prime numbers
def prime_sum : ℕ := prime_list.foldr (· + ·) 0

-- Define the count of prime numbers
def prime_count : ℕ := prime_list.length

-- Define the arithmetic mean of the prime numbers
def prime_mean : ℚ := prime_sum / prime_count

theorem arithmetic_mean_of_primes :
  prime_mean = 40 + 1 / 3 := sorry

end arithmetic_mean_of_primes_l452_452951


namespace smallest_multiple_of_9_and_6_is_18_l452_452484

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l452_452484


namespace midpoint_product_l452_452474

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 4) (hy1 : y1 = -3) (hx2 : x2 = -8) (hy2 : y2 = 7) :
  let midx := (x1 + x2) / 2
  let midy := (y1 + y2) / 2
  midx * midy = -4 :=
by
  sorry

end midpoint_product_l452_452474


namespace repeating_decimal_fraction_difference_l452_452349

theorem repeating_decimal_fraction_difference :
  ∀ (F : ℚ),
  F = 817 / 999 → (999 - 817 = 182) :=
by
  sorry

end repeating_decimal_fraction_difference_l452_452349


namespace minimum_operations_to_transfer_beer_l452_452456

-- Definition of the initial conditions
structure InitialState where
  barrel_quarts : ℕ := 108
  seven_quart_vessel : ℕ := 0
  five_quart_vessel : ℕ := 0

-- Definition of the desired final state after minimum steps
structure FinalState where
  operations : ℕ := 17

-- Main theorem statement
theorem minimum_operations_to_transfer_beer (s : InitialState) : FinalState :=
  sorry

end minimum_operations_to_transfer_beer_l452_452456


namespace jenny_best_neighborhood_earnings_l452_452334

theorem jenny_best_neighborhood_earnings :
  let cost_per_box := 2
  let neighborhood_a_homes := 10
  let neighborhood_a_boxes_per_home := 2
  let neighborhood_b_homes := 5
  let neighborhood_b_boxes_per_home := 5
  let earnings_a := neighborhood_a_homes * neighborhood_a_boxes_per_home * cost_per_box
  let earnings_b := neighborhood_b_homes * neighborhood_b_boxes_per_home * cost_per_box
  max earnings_a earnings_b = 50
:= by
  sorry

end jenny_best_neighborhood_earnings_l452_452334


namespace triangular_flowerbed_area_is_38_l452_452067

def rectangular_park_dimensions : ℝ := 15
def rectangular_park_length : ℝ := 45
def flowerbed_short_side_factor : ℝ := 1 / 3

noncomputable def triangular_flowerbed_area (width length : ℝ) : ℝ :=
  let base := flowerbed_short_side_factor * length
  let height := length
  (1 / 2) * base * height

theorem triangular_flowerbed_area_is_38 :
  triangular_flowerbed_area rectangular_park_dimensions rectangular_park_length ≈ 38 :=
by 
  let width := rectangular_park_dimensions
  let length := rectangular_park_length
  let base := flowerbed_short_side_factor * length
  let height := length
  let area := (1 / 2) * base * height
  have h_area : area = 37.5 := by sorry
  have rounded_area : Real.floor (area + 0.5) = 38 := by sorry
  exact rounded_area

end triangular_flowerbed_area_is_38_l452_452067


namespace problem_1_problem_2_problem_3_l452_452997

-- Definition of set M
def in_set_M (f : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, f(t + 2) = f(t) + f(2)

-- Problem 1
theorem problem_1 : ¬ in_set_M (λ x : ℝ, 3 * x + 2) := 
sorry

-- Problem 2
theorem problem_2 (a : ℝ) : in_set_M (λ x : ℝ, real.log (a / (x^2 + 2))) → 
  12 - 6 * real.sqrt 3 ≤ a ∧ a ≤ 12 + 6 * real.sqrt 3 :=
sorry

-- Problem 3
theorem problem_3 (b : ℝ) : in_set_M (λ x : ℝ, 2 ^ x + b * x^2) :=
sorry

end problem_1_problem_2_problem_3_l452_452997


namespace johns_distance_to_park_l452_452730

-- Define the conditions in Lean
def speed_kmph : ℝ := 9
def time_min : ℝ := 2
def time_hr : ℝ := time_min / 60

-- Define the theorem to prove the question equals the correct answer
theorem johns_distance_to_park : speed_kmph * time_hr = 0.3 := by
  -- Proof steps are omitted as indicated by 'sorry'
  sorry

end johns_distance_to_park_l452_452730


namespace toms_total_score_is_correct_l452_452451

/-- Tom's game score calculation conditions -/
def game_conditions (points_per_enemy : ℕ) (enemy_count : ℕ) (bonus_percent : ℝ) : ℝ :=
  let initial_score := points_per_enemy * enemy_count
      bonus := if enemy_count ≥ 100 then bonus_percent * initial_score else 0
  in initial_score + bonus

/-- Proof that Tom's total score is 2250 points given the conditions. -/
theorem toms_total_score_is_correct : 
  game_conditions 10 150 0.5 = 2250 :=
by
  sorry

end toms_total_score_is_correct_l452_452451


namespace election_1002nd_k_election_1001st_k_l452_452720

variable (k : ℕ)

noncomputable def election_in_1002nd_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 2001 → -- The conditions include the number of candidates 'n', and specifying that 'k' being the maximum initially means k ≤ 2001.
  true

noncomputable def election_in_1001st_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 1 → -- Similarly, these conditions specify the initial maximum placement as 1 when elected in 1001st round.
  true

-- Definitions specifying the problem to identify max k for given rounds
theorem election_1002nd_k : election_in_1002nd_round_max_k k := sorry

theorem election_1001st_k : election_in_1001st_round_max_k k := sorry

end election_1002nd_k_election_1001st_k_l452_452720


namespace boat_travel_distance_l452_452872

theorem boat_travel_distance
  (D : ℝ) -- Distance traveled in both directions
  (t : ℝ) -- Time in hours it takes to travel upstream
  (speed_boat : ℝ) -- Speed of the boat in still water
  (speed_stream : ℝ) -- Speed of the stream
  (time_diff : ℝ) -- Difference in time between downstream and upstream travel
  (h1 : speed_boat = 10)
  (h2 : speed_stream = 2)
  (h3 : time_diff = 1.5)
  (h4 : D = 8 * t)
  (h5 : D = 12 * (t - time_diff)) :
  D = 36 := by
  sorry

end boat_travel_distance_l452_452872


namespace candy_distribution_l452_452773

theorem candy_distribution (c : ℕ) (b : ℕ) :
  c = 10 → b = 5 →
  ∃ n, n = 34 ∧ ∀ (candy : ℕ) (boxes : Fin b → ℕ),
    sum boxes = c ∧ (∀ i, 0 < candy i) ∧ (∀ j, candy j > 0 → j < b - 1 → candy (j+1) > 0) :=
sorry

end candy_distribution_l452_452773


namespace pharmacy_coverage_l452_452062

-- Define the adjacency relation as a list of pairs
def adj : List (Char × Char) :=
  [('a', 'b'), ('a', 'd'), ('b', 'a'), ('b', 'c'), ('b', 'd'), ('c', 'b'), 
   ('d', 'a'), ('d', 'b'), ('d', 'f'), ('d', 'e'), ('e', 'd'), ('e', 'f'), 
   ('e', 'j'), ('e', 'l'), ('f', 'd'), ('f', 'e'), ('f', 'j'), ('f', 'i'), 
   ('f', 'g'), ('g', 'f'), ('g', 'i'), ('g', 'h'), ('h', 'g'), ('h', 'i'), 
   ('i', 'f'), ('i', 'g'), ('i', 'j'), ('i', 'h'), ('j', 'e'), ('j', 'f'), 
   ('j', 'i'), ('j', 'k'), ('k', 'l'), ('k', 'j'), ('l', 'k'), ('l', 'e')]

-- Define the groups that have pharmacies
def pharmacy_locations : List Char := ['b', 'i', 'l', 'm']

-- The main theorem stating that the given pharmacy locations cover all groups
theorem pharmacy_coverage :
  ∀ house : Char, house ∈ ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'] →
  ∃ p : Char, p ∈ pharmacy_locations ∧ (house = p ∨ (house, p) ∈ adj) :=
by
  sorry

end pharmacy_coverage_l452_452062


namespace work_together_days_l452_452049

theorem work_together_days
  (a_days : ℝ) (ha : a_days = 18)
  (b_days : ℝ) (hb : b_days = 30)
  (c_days : ℝ) (hc : c_days = 45)
  (combined_days : ℝ) :
  (combined_days = 1 / ((1 / a_days) + (1 / b_days) + (1 / c_days))) → combined_days = 9 := 
by
  sorry

end work_together_days_l452_452049


namespace correct_option_l452_452856

def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k / x

theorem correct_option :
  inverse_proportion x y → 
  (y = x + 3 ∨ y = x / 3 ∨ y = 3 / (x ^ 2) ∨ y = 3 / x) → 
  y = 3 / x :=
by
  sorry

end correct_option_l452_452856


namespace true_propositions_l452_452114

-- Define the propositions
def Proposition1 (a b : ℝ) := ¬(a > b → a^2 > b^2)
def Proposition2 (x y : ℝ) := (x + y = 0 → x = -y) → (x = -y → x + y = 0)
def Proposition3 (x : ℝ) := (x^2 < 4 → -2 < x ∧ x < 2) ∧ (-2 < x ∧ x < 2 → x^2 < 4)

-- Define the overall proposition checking
def correct_propositions : Prop :=
  ¬ Proposition1 ∧ Proposition2 ∧ Proposition3

-- Final proof statement
theorem true_propositions : correct_propositions :=
by
  -- Negation of Proposition 1 is false
  apply and.intro
  { sorry }
  -- Proposition 2 is true
  apply and.intro
  { sorry }
  -- Proposition 3 is true
  { sorry }

end true_propositions_l452_452114


namespace mary_has_more_money_than_marco_l452_452374

variable (Marco Mary : ℕ)

theorem mary_has_more_money_than_marco
    (h1 : Marco = 24)
    (h2 : Mary = 15)
    (h3 : ∃ maryNew : ℕ, maryNew = Mary + Marco / 2 - 5)
    (h4 : ∃ marcoNew : ℕ, marcoNew = Marco / 2) :
    ∃ diff : ℕ, diff = maryNew - marcoNew ∧ diff = 10 := 
by 
    sorry

end mary_has_more_money_than_marco_l452_452374


namespace conditional_probability_l452_452407

noncomputable def card_numbers : Finset ℕ := {3, 4, 5, 6, 7}

def event_A (cards : Finset ℕ) : Prop :=
  (cards.sum id) % 2 = 0

def event_B (cards : Finset ℕ) : Prop :=
  ∀ (n ∈ cards), n % 2 = 1

theorem conditional_probability :
  (∃(S : Finset (Finset ℕ)), S.card = 10 ∧
    (∃ (A_sub : Finset (Finset ℕ)), A_sub.card = 4 ∧
      (∃ (B_sub : Finset (Finset ℕ)), B_sub.card = 3 ∧
        ∀ (b ∈ B_sub), event_B b ∧ event_A b ∧
        P (B_sub ∩ A_sub) = (3 / 10 : ℚ) ∧
          P (A_sub) = (2 / 5 : ℚ) ∧
          P (B_sub ∩ A_sub) / P (A_sub) = (3 / 4 : ℚ)))) :=
sorry

end conditional_probability_l452_452407


namespace binomial_coefficient_third_term_l452_452799

theorem binomial_coefficient_third_term (a b : ℕ) (n : ℕ) 
  (h_coeff : 2 * nat.choose n 1 = 8) : nat.choose n 2 = 6 :=
by
  sorry

end binomial_coefficient_third_term_l452_452799


namespace yellow_block_weight_proof_l452_452342

-- Define the weights and the relationship between them
def green_block_weight : ℝ := 0.4
def additional_weight : ℝ := 0.2
def yellow_block_weight : ℝ := green_block_weight + additional_weight

-- The theorem to prove
theorem yellow_block_weight_proof : yellow_block_weight = 0.6 :=
by
  -- Proof will be supplied here
  sorry

end yellow_block_weight_proof_l452_452342


namespace intersection_of_sets_l452_452700

def setA (x : ℝ) : Prop := 2 * x + 1 > 0
def setB (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by 
  sorry  -- Placeholder for the proof

end intersection_of_sets_l452_452700


namespace gcd_256_180_600_l452_452838

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l452_452838


namespace redistributed_gnomes_l452_452413

def WestervilleWoods : ℕ := 20
def RavenswoodForest := 4 * WestervilleWoods
def GreenwoodGrove := (5 * RavenswoodForest) / 4
def OwnerTakes (f: ℕ) (p: ℚ) := p * f

def RemainingGnomes (initial: ℕ) (p: ℚ) := initial - (OwnerTakes initial p)

def TotalRemainingGnomes := 
  (RemainingGnomes RavenswoodForest (40 / 100)) + 
  (RemainingGnomes WestervilleWoods (30 / 100)) + 
  (RemainingGnomes GreenwoodGrove (50 / 100))

def GnomesPerForest := TotalRemainingGnomes / 3

theorem redistributed_gnomes : 
  2 * 37 + 38 = TotalRemainingGnomes := by
  sorry

end redistributed_gnomes_l452_452413


namespace greatest_integer_less_PS_l452_452289

theorem greatest_integer_less_PS 
  (PQ PS T : ℝ)
  (midpoint_TPS : T = PS / 2)
  (perpendicular_PT_QT : (PQ ^ 2 = (PS / 2) ^ 2 + (PS / 2) ^ 2))
  (PQ_value : PQ = 150) :
  ⌊ PS ⌋ = 212 :=
by
  sorry

end greatest_integer_less_PS_l452_452289


namespace log_function_domain_l452_452802

theorem log_function_domain (x : ℝ) : 
  (∃ y : ℝ, y = log (1/2) (2*x - 1)) ↔ (x > 1/2) :=
by
  sorry

end log_function_domain_l452_452802


namespace minimum_embrasure_length_l452_452565

theorem minimum_embrasure_length : ∀ (s : ℝ), 
  (∀ t : ℝ, (∃ k : ℤ, t = k / 2 ∧ k % 2 = 0) ∨ (∃ k : ℤ, t = (k + 1) / 2 ∧ k % 2 = 1)) → 
  (∃ z : ℝ, z = 2 / 3) := 
sorry

end minimum_embrasure_length_l452_452565


namespace speed_of_first_train_l452_452002

-- Definitions for speed calculation and ratios
def speed_of_second_train : ℝ := 400 / 4
def ratio_speed (S1 S2 : ℝ) : Prop := S1 / S2 = 7 / 8

-- The theorem stating the problem
theorem speed_of_first_train (S2 : ℝ) (h1 : speed_of_second_train = 100) (h2 : ratio_speed S1 S2) : S1 = 87.5 :=
by
  sorry

end speed_of_first_train_l452_452002


namespace compute_expression_l452_452246

-- Given condition
def condition (x : ℝ) : Prop := x + 1/x = 3

-- Theorem to prove
theorem compute_expression (x : ℝ) (hx : condition x) : (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 8 := 
by
  sorry

end compute_expression_l452_452246


namespace perfect_square_seq_l452_452588

def seq (a : ℕ) : ℕ → ℕ
| 1     := 4
| 2     := (a^2 - 2)^2
| 3     := (a^2 - 2)^2
| (n+1) := seq n * seq (n-1) - 2 * (seq n + seq (n-1)) - seq (n-2) + 8

theorem perfect_square_seq (a : ℕ) (h_gt2 : a > 2) (n : ℕ) : 
  ∃ k : ℕ, 2 + (nat.sqrt (seq a (n + 1))) = k^2 :=
sorry

end perfect_square_seq_l452_452588


namespace volume_tetrahedron_ABCD_l452_452722

noncomputable def volume_tetrahedron (AB CD distance angle : ℝ) : ℝ :=
  if (AB = 1 ∧ CD = sqrt 3 ∧ distance = 2 ∧ angle = π/3) then (sqrt 3) / 3 else 0

theorem volume_tetrahedron_ABCD :
  volume_tetrahedron 1 (sqrt 3) 2 (π/3) = (sqrt 3) / 3 :=
sorry

end volume_tetrahedron_ABCD_l452_452722


namespace domain_of_f_l452_452419

-- Define the function f
def f (x : ℝ) : ℝ := real.sqrt (x + 1) + real.logb 2016 (2 - x)

-- The domain condition for the function
def domain_condition (x : ℝ) : Prop := x + 1 ≥ 0 ∧ 2 - x > 0

-- The main theorem to prove the domain
theorem domain_of_f : {x : ℝ | domain_condition x} = set.Icc (-1) 2 \ {2} :=
by
  sorry

end domain_of_f_l452_452419


namespace equivalence_of_expressions_l452_452572

-- Define the expression on the left-hand side
def lhs := (real.sqrt ((real.sqrt 5) ^ 5)) ^ 6

-- Define the expression on the right-hand side
noncomputable def rhs := 78125 * real.sqrt 5

-- The theorem to prove
theorem equivalence_of_expressions : lhs = rhs :=
by
  sorry

end equivalence_of_expressions_l452_452572


namespace polynomial_neg_intervals_l452_452626

noncomputable def polynomial := λ x : ℝ, x^3 - 12*x^2 + 35*x + 48

theorem polynomial_neg_intervals :
  {x : ℝ | polynomial x < 0} = {x : ℝ | x ∈ Ioo (-1 : ℝ) 3 ∪ Ioi 16} :=
sorry

end polynomial_neg_intervals_l452_452626


namespace find_natural_x_l452_452942

-- Definition of the sum of digits
def sum_of_digits (x : ℕ) : ℕ :=
  (x.toDigits 10).sum

-- Definition of the product of digits
def product_of_digits (x : ℕ) : ℕ :=
  (x.toDigits 10).foldl (*) 1

theorem find_natural_x (x : ℕ) : 
  (product_of_digits x = 44 * x - 86868) ∧ (∃ k : ℕ, sum_of_digits x = k^3) :=
sorry

end find_natural_x_l452_452942


namespace intersection_A_B_l452_452203

noncomputable def A : Set ℝ := { x | (x - 1) / (x + 3) < 0 }
noncomputable def B : Set ℝ := { x | abs x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l452_452203


namespace marked_cells_below_diagonal_l452_452277

theorem marked_cells_below_diagonal (n : ℕ) (A : Matrix (Fin n) (Fin n) ℚ) 
  (marked_count : from_enum (λ (i j : Fin n), A i j ≠ 0) ≤ n - 1) :
  ∃ B : Matrix (Fin n) (Fin n) ℚ, (∃ row_perms col_perms : List (Perm (Fin n)),
  (∀ (i j : Fin n), (i ≥ j) → B i j = A i j) ∧
    B = Matrix.mul (Matrix.mul (Matrix.from_perm col_perms) A) (Matrix.from_perm row_perms)) := sorry

end marked_cells_below_diagonal_l452_452277


namespace leading_coefficient_is_15_l452_452969

-- Define the polynomials involved
def poly1 (x : ℝ) : ℝ := 5 * (x^5 - 2 * x^4 + 3 * x^2)
def poly2 (x : ℝ) : ℝ := -8 * (x^5 + x^3 - x)
def poly3 (x : ℝ) : ℝ := 6 * (3 * x^5 - x^4 + 2)

-- Define the given polynomial
def given_poly (x : ℝ) : ℝ := poly1 x + poly2 x + poly3 x

-- State the problem: The leading coefficient of the given polynomial is 15
theorem leading_coefficient_is_15 :
  ∀ x : ℝ, (leading_coefficient (given_poly x)) = 15 := by sorry

end leading_coefficient_is_15_l452_452969


namespace domain_of_g_l452_452255

variable (f : ℝ → ℝ)

def is_domain (df : Set ℝ) : Prop := 
  ∀ x, f x ≠ 0 ↔ x ∈ df

def within_domain (x : ℝ) (df : Set ℝ) : Prop := 
  x ∈ df

theorem domain_of_g (h : is_domain f (-2, 4)) : 
  is_domain (λ x: ℝ, f (x + 1) + f (-x)) (-3, 2) :=
by
  sorry

end domain_of_g_l452_452255


namespace gear_ratios_l452_452928

variable (x y z w : ℝ)
variable (ω_A ω_B ω_C ω_D : ℝ)
variable (k : ℝ)

theorem gear_ratios (h : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D) : 
    ω_A/ω_B = yzw/xzw ∧ ω_B/ω_C = xzw/xyw ∧ ω_C/ω_D = xyw/xyz ∧ ω_A/ω_C = yzw/xyw := 
sorry

end gear_ratios_l452_452928


namespace rearrange_numbers_25_rearrange_numbers_1000_l452_452314

theorem rearrange_numbers_25 (n : ℕ) (h : n = 25) : 
  ∃ (f : Fin n → Fin n), ∀ i : Fin (n - 1), (f i.succ.val - f i.val = 3 ∨ f i.succ.val - f i.val = 5 ∨ f i.val - f i.succ.val = 3 ∨ f i.val - f i.succ.val = 5) ∧ (f i).val ∈ Finset.range n := by
  sorry

theorem rearrange_numbers_1000 (n : ℕ) (h : n = 1000) : 
  ∃ (f : Fin n → Fin n), ∀ i : Fin (n - 1), (f i.succ.val - f i.val = 3 ∨ f i.succ.val - f i.val = 5 ∨ f i.val - f i.succ.val = 3 ∨ f i.val - f i.succ.val = 5) ∧ (f i).val ∈ Finset.range n := by
  sorry

end rearrange_numbers_25_rearrange_numbers_1000_l452_452314


namespace smallest_multiple_of_9_and_6_l452_452502

theorem smallest_multiple_of_9_and_6 : ∃ n : ℕ, (n > 0) ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) := 
begin
  use 18,
  split,
  { -- n > 0
    exact nat.succ_pos',
  },
  split,
  { -- n % 9 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 9),
  },
  split,
  { -- n % 6 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 6),
  },
  { -- ∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m
    intros m h_pos h_multiple9 h_multiple6,
    exact le_of_dvd h_pos (nat.lcm_dvd_prime_multiples 6 9),
  },
  sorry, -- Since full proof capabilities are not required here, "sorry" is used to skip the proof process.
end

end smallest_multiple_of_9_and_6_l452_452502


namespace sin3x_plus_sin7x_l452_452140

theorem sin3x_plus_sin7x (x : ℝ) : 
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by sorry

end sin3x_plus_sin7x_l452_452140


namespace smallest_common_multiple_of_9_and_6_l452_452488

theorem smallest_common_multiple_of_9_and_6 : 
  ∃ x : ℕ, (x > 0 ∧ x % 9 = 0 ∧ x % 6 = 0) ∧ 
           ∀ y : ℕ, (y > 0 ∧ y % 9 = 0 ∧ y % 6 = 0) → x ≤ y :=
begin
  use 18,
  split,
  { split,
    { exact nat.succ_pos 17, },
    { split,
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_right 9 6), },
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_left 9 6), } } },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    exact lcm.dvd_iff.1 (nat.dvd_of_mod_eq_zero hy3) }
end

end smallest_common_multiple_of_9_and_6_l452_452488


namespace ratio_of_triangle_areas_l452_452308

theorem ratio_of_triangle_areas (A B C D E O : Type) 
  (hAD_2DC : AD = 2 * DC)
  (hABD_area : area ABD = 3)
  (hAED_area : area AED = 1)
  (hA_BC : ∃ E, E ∈ line_segment B C)
  (hAO_BO : ∃ O, O = line_intersection (line A E) (line B D)) :
  area ABO / area OED = 9 :=
sorry

end ratio_of_triangle_areas_l452_452308


namespace problem_I_solution_set_problem_II_min_a_l452_452223

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (2 * x + 3)

-- Problem (I): Solution set of f(x) ≥ x
theorem problem_I_solution_set : {x : ℝ | f x ≥ x} = set.Ici (4 / 5) :=
by sorry

-- Problem (II): Minimum value of a such that f(x) ≤ my + a / my holds for any x, y
theorem problem_II_min_a (m : ℝ) (y : ℝ) (h : 0 < m) 
  : ∀ x : ℝ, f x ≤ m^y + a / m^y → a = 3 :=
by sorry

end problem_I_solution_set_problem_II_min_a_l452_452223


namespace smallest_common_multiple_of_9_and_6_l452_452492

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l452_452492


namespace solve_for_a_l452_452170

theorem solve_for_a (a : ℝ) (h_pos : a > 0) 
  (h_roots : ∀ x, x^2 - 2*a*x - 3*a^2 = 0 → (x = -a ∨ x = 3*a)) 
  (h_diff : |(-a) - (3*a)| = 8) : a = 2 := 
sorry

end solve_for_a_l452_452170


namespace cohen_saw_1300_fish_eater_birds_l452_452512

theorem cohen_saw_1300_fish_eater_birds :
  let day1 := 300
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3 = 1300 :=
by
  sorry

end cohen_saw_1300_fish_eater_birds_l452_452512


namespace solve_for_r_l452_452408

theorem solve_for_r (r : ℝ) : 
  (r^2 - 3) / 3 = (5 - r) / 2 ↔ 
  r = (-3 + Real.sqrt 177) / 4 ∨ r = (-3 - Real.sqrt 177) / 4 :=
by
  sorry

end solve_for_r_l452_452408


namespace union_of_M_and_N_l452_452367

def M : Set ℝ := {x | x^2 + 3x + 2 > 0}
def N : Set ℝ := {x | (1 / 2)^x ≤ 4}

theorem union_of_M_and_N : M ∪ N = Set.univ := by
  sorry

end union_of_M_and_N_l452_452367


namespace minimal_sum_arithmetic_seq_l452_452292

-- Definition of the arithmetic sequence
def arithmetic_seq (a₄ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
a₄ + d * (n - 4)

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a₁ an : ℤ) (n : ℕ) : ℤ :=
n * (a₁ + an) / 2

theorem minimal_sum_arithmetic_seq :
  let a₄ := -14 in
  let d := 3 in
  (arithmetic_seq a₄ d 1 = -23) ∧
  (arithmetic_seq a₄ d 8 = -2) ∧
  sum_arithmetic_seq (-23) (-2) 8 = -100 :=
by
  -- Definitions and assertions
  sorry

end minimal_sum_arithmetic_seq_l452_452292


namespace rationalize_denominator_l452_452396

theorem rationalize_denominator :
  (1 / (real.sqrt 3 - 2)) = -(real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l452_452396


namespace lcm_1540_2310_l452_452470

theorem lcm_1540_2310 : Nat.lcm 1540 2310 = 4620 :=
by sorry

end lcm_1540_2310_l452_452470


namespace coefficient_x55_l452_452613

def polynomial : ℕ → Polynomial ℤ
| 1 := Polynomial.X - 1
| k := Polynomial.X ^ k - k

def expansion := (Finset.range 12).prod polynomial

theorem coefficient_x55 : expansion.coeff 55 = 99 :=
by
  sorry

end coefficient_x55_l452_452613


namespace rotation_150_positions_l452_452086

/-
Define the initial positions and the shapes involved.
-/
noncomputable def initial_positions := ["A", "B", "C", "D"]
noncomputable def initial_order := ["triangle", "smaller_circle", "square", "pentagon"]

def rotate_clockwise_150 (pos : List String) : List String :=
  -- 1 full position and two-thirds into the next position
  [pos.get! 1, pos.get! 2, pos.get! 3, pos.get! 0]

theorem rotation_150_positions :
  rotate_clockwise_150 initial_positions = ["Triangle between B and C", 
                                            "Smaller circle between C and D", 
                                            "Square between D and A", 
                                            "Pentagon between A and B"] :=
by sorry

end rotation_150_positions_l452_452086


namespace translated_point_satisfies_conditions_l452_452454

theorem translated_point_satisfies_conditions 
  (s : ℝ) (h_pos : s > 0) 
  (hP : ∃ t : ℝ, (π/4, t) ∈ {p : ℝ × ℝ | p.2 = sin (p.1 - π/12)})
  (hP' : ∃ t : ℝ, (π/4 - s, t) ∈ {p : ℝ × ℝ | p.2 = sin (2 * p.1)}) :
  ∃ t : ℝ, t = 1/2 ∧ s = π/6 :=
by
  sorry

end translated_point_satisfies_conditions_l452_452454


namespace polynomial_abs_sum_roots_l452_452931

theorem polynomial_abs_sum_roots (p q r m : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2500) (h3 : p * q * r = -m) :
  |p| + |q| + |r| = 100 :=
sorry

end polynomial_abs_sum_roots_l452_452931


namespace investment_ratio_proof_l452_452078

noncomputable def investment_ratio (x : ℝ) (m : ℝ) : Prop :=
  let A_investment_time := x * 12
  let B_investment_time := 2 * x * 6
  let C_investment_time := m * x * 4
  let total_investment_time := A_investment_time + B_investment_time + C_investment_time
  let A_share := (A_investment_time / total_investment_time) * 12000
  A_share = 4000 ∧ m = 3

-- Statement of the proof problem
theorem investment_ratio_proof (x : ℝ) (m : ℝ) (A_investment : ℝ) 
    (B_investment : ℝ) (C_investment : ℝ) (annual_profit : ℝ) (A_share : ℝ) :
  A_investment = x →
  B_investment = 2 * x → 
  C_investment = m * x →
  annual_profit = 12000 →
  A_share = 4000 →
  investment_ratio x m := by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end investment_ratio_proof_l452_452078


namespace first_shipment_weight_l452_452532

variable (first_shipment : ℕ)
variable (total_dishes_made : ℕ := 13)
variable (couscous_per_dish : ℕ := 5)
variable (second_shipment : ℕ := 45)
variable (same_day_shipment : ℕ := 13)

theorem first_shipment_weight :
  13 * 5 = 65 → second_shipment ≠ first_shipment → 
  first_shipment + same_day_shipment = 65 →
  first_shipment = 65 :=
by
  sorry

end first_shipment_weight_l452_452532


namespace parallelogram_ratio_l452_452280

theorem parallelogram_ratio
  (EFGH : parallelogram)
  (E F G H J K L : point)
  (hJ : J ∈ segment E F ∧ (distance E J) / (distance E F) = 1 / 4)
  (hK : K ∈ segment E H ∧ (distance E K) / (distance E H) = 1 / 5)
  (hL : L ∈ line_through J K)
  (h_inter : L ∈ line_through F G) :
  (distance F G) / (distance F L) = 2 := by
  sorry

end parallelogram_ratio_l452_452280


namespace sum_of_fractions_l452_452026

theorem sum_of_fractions :
  (1 / 3) + (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-9 / 20) = -9 / 20 := 
by
  sorry

end sum_of_fractions_l452_452026


namespace remainder_of_b47_mod_49_l452_452753

theorem remainder_of_b47_mod_49:
  let b (n: ℕ) := 7^n + 9^n in
  b 47 % 49 = 14 :=
by
  sorry

end remainder_of_b47_mod_49_l452_452753


namespace find_a_l452_452058

def is_tangent (line : ℝ → Point → Point → Prop) (circle : Circle) :=
  ∃ (P Q : Point), line m P Q ∧ P = Q ∧ ∀ R, R ∈ circle → ¬ collinear P R Q

def perpendicular (l1 l2 : Line) :=
  ∃ (m1 m2 : ℝ), m1 * m2 = -1

def circle := {center : Point // radius : ℝ}

def Point := ℝ × ℝ
def Line := Point → Point → Prop

theorem find_a :
  ∀ a : ℝ,
  (∀ (P : Point), P = (1, 2) → is_tangent (λ m P Q, m = (fst Q - fst P) / (snd Q - snd P)) (mk_circle O 4)) →
  (∀ (L : Line), L = (λ P Q, ∃ a, ∀ (x y : ℝ), x - a * y + 1 = 0) → perpendicular L (λ P Q, True)) →
  a = -3 / 4 :=
by
  sorry

end find_a_l452_452058


namespace probability_xy_minus_x_minus_y_even_l452_452464

open Nat

theorem probability_xy_minus_x_minus_y_even :
  let S := {1,2,3,4,5,6,7,8,9,10,11,12}
  let evens := {2, 4, 6, 8, 10, 12}
  let total_pairs := (Finset.card S).choose 2
  let even_pairs := (Finset.card evens).choose 2
  even_pairs / total_pairs = 5 / 22 :=
by
  sorry

end probability_xy_minus_x_minus_y_even_l452_452464


namespace sin_equality_l452_452968

noncomputable def find_angle (θ : ℝ) : ℝ :=
  let multiple_of_360 := (θ / 360).floor
  θ - multiple_of_360 * 360

theorem sin_equality (n : ℤ) (hn1 : -180 ≤ n) (hn2 : n ≤ 180) :
  find_angle 750 = 30 → sin (n : ℝ) = sin 750 → n = 30 := 
by
  intro hangle hsin
  sorry

end sin_equality_l452_452968


namespace polynomial_satisfies_conditions_l452_452754

noncomputable def f (x y z : ℝ) : ℝ :=
  (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (f x (z^2) y + f x (y^2) z = 0) ∧ (f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end polynomial_satisfies_conditions_l452_452754


namespace total_people_counted_l452_452092

-- Definitions based on conditions
def people_second_day : ℕ := 500
def people_first_day : ℕ := 2 * people_second_day

-- Theorem statement
theorem total_people_counted : people_first_day + people_second_day = 1500 := 
by 
  sorry

end total_people_counted_l452_452092


namespace gcd_256_180_600_l452_452837

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l452_452837


namespace sequence_inequality_l452_452019

theorem sequence_inequality (k : ℕ) (m : ℕ) (b : ℕ → ℝ) (a : ℕ → ℝ)
  (h_a1a2 : a 1 = k ∧ a 2 = k)
  (h_an : ∀ n, a (n + 2) = a (n + 1) + a n)
  (h_b_cond : b 2 ≥ b 1 ∧ ∀ n, b (n + 2) ≥ b (n + 1) + b n)
  (h_sum_eq : ∑ i in finset.range m, a (i + 1) = ∑ i in finset.range m, b (i + 1)) : 
  a (m + 1) ≤ b m + b (m - 1) :=
by
  sorry

end sequence_inequality_l452_452019


namespace total_capsules_in_july_l452_452726

theorem total_capsules_in_july : 
  let mondays := 4
  let tuesdays := 5
  let wednesdays := 5
  let thursdays := 4
  let fridays := 4
  let saturdays := 4
  let sundays := 5

  let capsules_monday := mondays * 2
  let capsules_tuesday := tuesdays * 3
  let capsules_wednesday := wednesdays * 2
  let capsules_thursday := thursdays * 3
  let capsules_friday := fridays * 2
  let capsules_saturday := saturdays * 4
  let capsules_sunday := sundays * 4

  let total_capsules := capsules_monday + capsules_tuesday + capsules_wednesday + capsules_thursday + capsules_friday + capsules_saturday + capsules_sunday

  let missed_capsules_tuesday := 3
  let missed_capsules_sunday := 4

  let total_missed_capsules := missed_capsules_tuesday + missed_capsules_sunday

  let total_consumed_capsules := total_capsules - total_missed_capsules
  total_consumed_capsules = 82 := 
by
  -- Details omitted, proof goes here
  sorry

end total_capsules_in_july_l452_452726


namespace sin_cos_product_l452_452242

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l452_452242


namespace imaginary_part_of_z_is_1_l452_452630

-- Define the complex number z
def z : ℂ := (3 - complex.I) * (2 + complex.I)

-- Statement to prove the imaginary part of z is 1
theorem imaginary_part_of_z_is_1 : z.im = 1 :=
sorry

end imaginary_part_of_z_is_1_l452_452630


namespace permutation_exists_25_permutation_exists_1000_l452_452326

-- Define a function that checks if a permutation satisfies the condition
def valid_permutation (perm : List ℕ) : Prop :=
  (∀ i < perm.length - 1, let diff := (perm[i] - perm[i+1]).abs in diff = 3 ∨ diff = 5)

-- Proof problem for n = 25
theorem permutation_exists_25 : 
  ∃ perm : List ℕ, perm.perm (List.range 25).map (· + 1) ∧ valid_permutation perm := 
sorry

-- Proof problem for n = 1000
theorem permutation_exists_1000 : 
  ∃ perm : List ℕ, perm.perm (List.range 1000).map (· + 1) ∧ valid_permutation perm := 
sorry

end permutation_exists_25_permutation_exists_1000_l452_452326


namespace girl_walking_speed_l452_452880

-- Definitions of the conditions
def distance := 30 -- in kilometers
def time := 6 -- in hours

-- Definition of the walking speed function
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- The theorem we want to prove
theorem girl_walking_speed : speed distance time = 5 := by
  sorry

end girl_walking_speed_l452_452880


namespace correct_exponentiation_l452_452030

theorem correct_exponentiation : ∀ (a : ℝ) (ha : a ≠ 0), 
  (a^2 + a^4 ≠ a^6) ∧ (a^2 * a^4 ≠ a^8) ∧ ((a^3)^2 ≠ a^9) ∧ (a^6 / a^2 = a^4) :=
by
  intro a ha
  split
  { sorry } -- Proof for a^2 + a^4 ≠ a^6
  split
  { sorry } -- Proof for a^2 * a^4 ≠ a^8
  split
  { sorry } -- Proof for (a^3)^2 ≠ a^9
  { sorry } -- Proof for a^6 / a^2 = a^4

end correct_exponentiation_l452_452030


namespace smallest_multiple_9_and_6_l452_452499

theorem smallest_multiple_9_and_6 : ∃ n : ℕ, n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m :=
by
  have h := Nat.lcm 9 6
  use h
  split
  sorry

end smallest_multiple_9_and_6_l452_452499


namespace find_g_l452_452141

/-
  Problem: Given the equation 2x^5 - 4x^3 + 3x + g(x) = 7x^4 - 2x^2 - 5x + 4,
  prove that g(x) = -2x^5 + 7x^4 + 4x^3 - 2x^2 - 8x + 4.
-/
theorem find_g (x : ℝ) (g : ℝ → ℝ) :
  2*x^5 - 4*x^3 + 3*x + g(x) = 7*x^4 - 2*x^2 - 5*x + 4 →
  g(x) = -2*x^5 + 7*x^4 + 4*x^3 - 2*x^2 - 8*x + 4 :=
by
  intro h
  sorry

end find_g_l452_452141


namespace min_sum_arithmetic_sequence_l452_452999

theorem min_sum_arithmetic_sequence 
  (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith_seq : ∀ n, a (n+1) - a n = a 1 - a 0)
  (h_sum_def : S = λ n, n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0))
  (h_cond1 : a 4 + a 7 + a 10 = 9)
  (h_cond2 : S 14 - S 3 = 77):
  ∃ n, (∀ m, S n ≤ S m) ∧ n = 5 :=
by 
  sorry

end min_sum_arithmetic_sequence_l452_452999


namespace rationalize_denominator_l452_452400

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 2) = -(Real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l452_452400


namespace smallest_multiple_of_9_and_6_is_18_l452_452486

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l452_452486


namespace find_angle_between_vectors_l452_452208

open Real

variables (a b : EuclideanSpace ℝ (Fin 3)) -- Assume 3D Euclidean space for vectors

def magnitude (v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sqrt (inner v v)

noncomputable def angle_between (u v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  real.acos ((inner u v) / (magnitude u * magnitude v))

-- Statement of the problem
theorem find_angle_between_vectors 
  (ha : magnitude a = 1) 
  (hb : magnitude b = 2) 
  (h_dot : inner a (a + b) = 0) : 
  angle_between a b = Real.pi * (2/3) :=
sorry -- The proof is not required.

end find_angle_between_vectors_l452_452208


namespace range_of_f_l452_452925

noncomputable def f (x : ℝ) : ℝ := |x + 8| - |x - 3|

theorem range_of_f : set.range f = set.Icc (-11:ℝ) (11:ℝ) := sorry

end range_of_f_l452_452925


namespace carmen_candle_usage_l452_452102

-- Define the duration a candle lasts when burned for 1 hour every night.
def candle_duration_1_hour_per_night : ℕ := 8

-- Define the number of hours Carmen burns a candle each night.
def hours_burned_per_night : ℕ := 2

-- Define the number of nights over which we want to calculate the number of candles needed.
def number_of_nights : ℕ := 24

-- We want to show that given these conditions, Carmen will use 6 candles.
theorem carmen_candle_usage :
  (number_of_nights / (candle_duration_1_hour_per_night / hours_burned_per_night)) = 6 :=
by
  sorry

end carmen_candle_usage_l452_452102


namespace find_original_concentration_l452_452696

noncomputable def original_vinegar_concentration (C : ℝ) : Prop :=
  let amount_pure_vinegar_original := C / 100 * 12
  let amount_pure_vinegar_diluted := 0.07 * (12 + 50)
  amount_pure_vinegar_original = amount_pure_vinegar_diluted

theorem find_original_concentration : ∃ C : ℝ, original_vinegar_concentration C ∧ C ≈ 36.1667 :=
by
  sorry

end find_original_concentration_l452_452696


namespace similarity_and_ratio_l452_452276

-- Definitions and conditions
variables {A B C : Point}
variables {M_a M_b M_c : Point}
variables {T_a T_b T_c : Point}
variables (ω_a ω_b ω_c : Circle)
variables (p_a p_b p_c : Line)

-- Assume midpoints of sides
axiom midpoints (M_aM_bM_c : M_a = midpoint B C ∧ M_b = midpoint C A ∧ M_c = midpoint A B)

-- Assume midpoints of arcs of circumcircle
axiom arc_midpoints : T_a = arc_midpoint A B C ∧ T_b = arc_midpoint B C A ∧ T_c = arc_midpoint C A B

-- Assume circles with diameters M_i T_i
axiom circle_diameters : ∀ i ∈ {a, b, c}, ω_i = circle_diameter (M_i, T_i)

-- Assume common external tangents
axiom tangents : ∀ i ∈ {a, b, c}, ∃ j k ∈ {a, b, c}, j ≠ k ≠ i ∧ 
                                 tangent_to ω_j ω_k (p_i) ∧ side_of ω_j ω_k ω_i (p_i)

-- Prove similarity and find ratio of similitude
theorem similarity_and_ratio : similar (triangle p_a p_b p_c) (triangle A B C) (1 / 4) := sorry

end similarity_and_ratio_l452_452276


namespace arithmetic_square_root_l452_452031

theorem arithmetic_square_root (a : ℝ) (h : a > 0) : real.sqrt(a) = a^(1/2) :=
sorry

end arithmetic_square_root_l452_452031


namespace pages_with_same_units_digit_count_l452_452533

theorem pages_with_same_units_digit_count :
  let pages := { x : ℕ | 1 ≤ x ∧ x ≤ 100 }
  let same_units_digit :=
    { x ∈ pages | (x % 10) = ((101 - x) % 10) }
  same_units_digit.card = 20 := 
by {
  sorry
}

end pages_with_same_units_digit_count_l452_452533


namespace exists_two_interns_with_same_acquaintances_l452_452784

theorem exists_two_interns_with_same_acquaintances
  {n : ℕ} (h : symmetric (λ (a b : fin n), a ≠ b)) :
  ∃ (a b : fin n), a ≠ b ∧ (number_of_acquaintances a) = (number_of_acquaintances b) :=
by
  sorry

noncomputable def number_of_acquaintances (a : fin n) : fin n := sorry

end exists_two_interns_with_same_acquaintances_l452_452784


namespace number_of_terms_in_simplified_expression_l452_452918

theorem number_of_terms_in_simplified_expression :
  let expr := (x + y + z) ^ 2008 + (x - y - z) ^ 2008
  (number_of_terms expr = 1_010_025) :=
by
  sorry

end number_of_terms_in_simplified_expression_l452_452918


namespace car_travel_distance_l452_452536

theorem car_travel_distance (b t : ℕ) (h1 : t ≠ 0) :
  let rate := b / 4 in
  let time_seconds := 5 * 60 in
  let distance_feet := (rate * time_seconds / t) in
  let distance_yards := distance_feet / 3 in
  distance_yards = 25 * b / t :=
  by
  sorry

end car_travel_distance_l452_452536


namespace inverse_proportion_function_sol_l452_452422

theorem inverse_proportion_function_sol (k m x : ℝ) (h1 : k ≠ 0) (h2 : (m - 1) * x ^ (m ^ 2 - 2) = k / x) : m = -1 :=
by
  sorry

end inverse_proportion_function_sol_l452_452422


namespace b_2016_val_l452_452230

open Nat

noncomputable def a : ℕ → ℝ
| 0       := 1/2
| (n + 1) := 1 - b n

noncomputable def b : ℕ → ℝ
| 0       := 1/2
| (n + 1) := b n / (1 - (a n)^2)

theorem b_2016_val : b 2016 = 2016 / 2017 := by
  sorry

end b_2016_val_l452_452230


namespace find_a1_find_a2_cn_geometric_Tn_sum_l452_452761

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {c : ℕ → ℝ}

-- Given conditions
def hSn (n : ℕ) : S n = 2 * a n - 2^n := sorry
def hcn (n : ℕ) : c n = a (n + 1) - 2 * a n := sorry
def hcn_geometric (n : ℕ) : c n = 2^n := sorry

-- Proving the outcomes
theorem find_a1 : a 1 = 2 :=
sorry

theorem find_a2 : a 2 = 6 :=
sorry

theorem cn_geometric : ∀ n, c (n + 1) / c n = 2 :=
by
  intro n
  sorry

theorem Tn_sum (n : ℕ) : let T_n := ∑ i in Finset.range n, (i + 1) / (2 * c i)
  in T_n = 3 / 2 - 1 / (2^n) - (n + 1) / (2^(n + 1)) :=
by
  intro n
  let T_n := ∑ i in Finset.range n, (i + 1) / (2 * c i)
  sorry

end find_a1_find_a2_cn_geometric_Tn_sum_l452_452761


namespace find_a_plus_b_l452_452667

-- Conditions for the lines
def line_l0 (x y : ℝ) : Prop := x - y + 1 = 0
def line_l1 (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y + 1 = 0
def line_l2 (b : ℝ) (x y : ℝ) : Prop := x + b * y + 3 = 0

-- Perpendicularity condition for l1 to l0
def perpendicular (a : ℝ) : Prop := 1 * a + (-1) * (-2) = 0

-- Parallel condition for l2 to l0
def parallel (b : ℝ) : Prop := 1 * b = (-1) * 1

-- Prove the value of a + b given the conditions
theorem find_a_plus_b (a b : ℝ) 
  (h1 : perpendicular a)
  (h2 : parallel b) : a + b = -3 :=
sorry

end find_a_plus_b_l452_452667


namespace ConeVolumeSphereRatio_l452_452272

-- Definitions and conditions.
def Point := ℝ × ℝ × ℝ

def SA := 2.0
def AB := 2.0
def BC := 1.0
def angle_ABC := 90 -- degrees
def right_triangle_cone (S A B C : Point) := true -- Placeholder

noncomputable def volume_cone (S A B C : Point) := (1:ℝ) / 3 * (1 / 2 * AB * BC) * SA
noncomputable def radius_sphere := 3.0 / 2
noncomputable def volume_sphere := (4:ℝ) / 3 * Real.pi * (radius_sphere ^ 3)

-- The proof statement.
theorem ConeVolumeSphereRatio
  (S A B C : Point)
  (h1 : SA = 2)
  (h2 : AB = 2)
  (h3 : BC = 1)
  (h4 : angle_ABC = 90)
  (h5 : right_triangle_cone S A B C) :
  volume_cone S A B C / volume_sphere = 4 / (27 * Real.pi) :=
by
   sorry

end ConeVolumeSphereRatio_l452_452272


namespace tangent_line_at_p_l452_452421

-- Definitions for the conditions
def f (x : ℝ) : ℝ := x / (x - 2)
def p : ℝ × ℝ := (1, -1)

-- The theorem statement we need to prove
theorem tangent_line_at_p :
  let tangent_line (x : ℝ) := -2 * x + 1 in
  ∃ (k b : ℝ), (∀ x y : ℝ, y = f x → y + 1 = k * (x - 1) + b) ∧
                tangent_line = λ x, k * x + b :=
sorry

end tangent_line_at_p_l452_452421


namespace number_real_root_quadratics_l452_452929

theorem number_real_root_quadratics :
  let s := {-3, -2, -1, 0, 1, 2, 3}
  let realRoots (b c : ℤ) := b^2 - 4 * c ≥ 0
  (s.product s).count (λ p, realRoots p.1 p.2) = 34 := 
by {
  -- Definitions and logic to be done here
  sorry
}

end number_real_root_quadratics_l452_452929


namespace expression_undefined_count_l452_452623

theorem expression_undefined_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x : ℝ,
  ((x = x1 ∨ x = x2) ↔ (x^2 - 2*x - 3 = 0 ∨ x - 3 = 0)) ∧ 
  ((x^2 - 2*x - 3) * (x - 3) = 0 → (x = x1 ∨ x = x2)) :=
by
  sorry

end expression_undefined_count_l452_452623


namespace quadratic_real_roots_range_k_l452_452994

-- Define the quadratic function
def quadratic_eq (k x : ℝ) : ℝ := k * x^2 - 6 * x + 9

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the conditions for the quadratic equation to have distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ discriminant a b c > 0

theorem quadratic_real_roots_range_k (k : ℝ) :
  has_two_distinct_real_roots k (-6) 9 ↔ k < 1 ∧ k ≠ 0 := 
by
  sorry

end quadratic_real_roots_range_k_l452_452994


namespace equilateral_triangle_hyperbola_distinct_branch_find_coordinates_Q_R_l452_452115

-- Problem 1: Prove vertices of an equilateral triangle cannot all lie on the same branch of the hyperbola xy = 1.
theorem equilateral_triangle_hyperbola_distinct_branch
  (P Q R : ℝ × ℝ)
  (hP : P.1 * P.2 = 1)
  (hQ : Q.1 * Q.2 = 1)
  (hR : R.1 * R.2 = 1)
  (equilateral : dist P Q = dist Q R ∧ dist Q R = dist R P) :
  ¬ (P.1 > 0 ∧ Q.1 > 0 ∧ R.1 > 0 ∨ P.1 < 0 ∧ Q.1 < 0 ∧ R.1 < 0) := sorry

-- Problem 2: Given specific point P on C2 and Q, R on C1, find the coordinates of Q and R.
theorem find_coordinates_Q_R
  (P : ℝ × ℝ)
  (hP : P = (-1, -1))
  (Q R : ℝ × ℝ)
  (hQ : Q.1 * Q.2 = 1 ∧ Q.1 > 0)
  (hR : R.1 * R.2 = 1 ∧ R.1 > 0)
  (equilateral : dist P Q = dist Q R ∧ dist Q R = dist R P) :
  Q = (2 - real.sqrt 3, 2 + real.sqrt 3) ∧ R = (2 + real.sqrt 3, 2 - real.sqrt 3) := sorry

end equilateral_triangle_hyperbola_distinct_branch_find_coordinates_Q_R_l452_452115


namespace total_meals_per_week_l452_452678

-- Definitions of the conditions
def meals_per_day_r1 : ℕ := 20
def meals_per_day_r2 : ℕ := 40
def meals_per_day_r3 : ℕ := 50
def days_per_week : ℕ := 7

-- The proof goal
theorem total_meals_per_week : 
  (meals_per_day_r1 * days_per_week) + 
  (meals_per_day_r2 * days_per_week) + 
  (meals_per_day_r3 * days_per_week) = 770 :=
by
  sorry

end total_meals_per_week_l452_452678


namespace smallest_multiple_of_9_and_6_l452_452503

theorem smallest_multiple_of_9_and_6 : ∃ n : ℕ, (n > 0) ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) := 
begin
  use 18,
  split,
  { -- n > 0
    exact nat.succ_pos',
  },
  split,
  { -- n % 9 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 9),
  },
  split,
  { -- n % 6 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 6),
  },
  { -- ∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m
    intros m h_pos h_multiple9 h_multiple6,
    exact le_of_dvd h_pos (nat.lcm_dvd_prime_multiples 6 9),
  },
  sorry, -- Since full proof capabilities are not required here, "sorry" is used to skip the proof process.
end

end smallest_multiple_of_9_and_6_l452_452503


namespace student_failed_by_l452_452554

theorem student_failed_by :
  ∀ (total_marks obtained_marks passing_percentage : ℕ),
  total_marks = 700 →
  obtained_marks = 175 →
  passing_percentage = 33 →
  (passing_percentage * total_marks) / 100 - obtained_marks = 56 :=
by
  intros total_marks obtained_marks passing_percentage h1 h2 h3
  sorry

end student_failed_by_l452_452554


namespace temperature_in_quebec_city_is_negative_8_l452_452007

def temperature_vancouver : ℝ := 22
def temperature_calgary (temperature_vancouver : ℝ) : ℝ := temperature_vancouver - 19
def temperature_quebec_city (temperature_calgary : ℝ) : ℝ := temperature_calgary - 11

theorem temperature_in_quebec_city_is_negative_8 :
  temperature_quebec_city (temperature_calgary temperature_vancouver) = -8 := by
  sorry

end temperature_in_quebec_city_is_negative_8_l452_452007


namespace smaller_of_two_digit_product_4680_l452_452427

theorem smaller_of_two_digit_product_4680 (a b : ℕ) (h1 : a * b = 4680) (h2 : 10 ≤ a) (h3 : a < 100) (h4 : 10 ≤ b) (h5 : b < 100): min a b = 40 :=
sorry

end smaller_of_two_digit_product_4680_l452_452427


namespace find_a_l452_452672

-- Define sets A and B
def A : Set ℕ := {1, 2, 5}
def B (a : ℕ) : Set ℕ := {2, a}

-- Given condition: A ∪ B = {1, 2, 3, 5}
def union_condition (a : ℕ) : Prop := A ∪ B a = {1, 2, 3, 5}

-- Theorem we want to prove
theorem find_a (a : ℕ) : union_condition a → a = 3 :=
by
  intro h
  sorry

end find_a_l452_452672


namespace largest_uncovered_squares_l452_452912

theorem largest_uncovered_squares (board_size : ℕ) (total_squares : ℕ) (domino_size : ℕ) 
  (odd_property : ∀ (n : ℕ), n % 2 = 1 → (n - domino_size) % 2 = 1)
  (can_place_more : ∀ (placed_squares odd_squares : ℕ), placed_squares + domino_size ≤ total_squares → odd_squares - domino_size % 2 = 1 → odd_squares ≥ 0)
  : ∃ max_uncovered : ℕ, max_uncovered = 7 := by
  sorry

end largest_uncovered_squares_l452_452912


namespace average_salary_for_company_l452_452052

theorem average_salary_for_company
    (number_of_managers : Nat)
    (number_of_associates : Nat)
    (average_salary_managers : Nat)
    (average_salary_associates : Nat)
    (hnum_managers : number_of_managers = 15)
    (hnum_associates : number_of_associates = 75)
    (has_managers : average_salary_managers = 90000)
    (has_associates : average_salary_associates = 30000) : 
    (number_of_managers * average_salary_managers + number_of_associates * average_salary_associates) / 
    (number_of_managers + number_of_associates) = 40000 := 
    by
    sorry

end average_salary_for_company_l452_452052


namespace smallest_distance_l452_452348

open Real

/-- Let A be a point on the circle (x-3)^2 + (y-4)^2 = 16,
and let B be a point on the parabola x^2 = 8y.
The smallest possible distance AB is √34 - 4. -/
theorem smallest_distance 
  (A B : ℝ × ℝ)
  (hA : (A.1 - 3)^2 + (A.2 - 4)^2 = 16)
  (hB : (B.1)^2 = 8 * B.2) :
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ sqrt 34 - 4 := 
sorry

end smallest_distance_l452_452348


namespace symmetry_axis_of_f_triangle_side_b_l452_452221

noncomputable def symmetry_axis (k : ℤ) : ℝ := (k * Real.pi / 2) + (Real.pi / 3)

theorem symmetry_axis_of_f :
  ∀ k : ℤ, ∃ x : ℝ, symmetry_axis k = x :=
by 
  sorry

noncomputable def g_of_x (x : ℝ) : ℝ := Real.sin (x + (Real.pi / 6)) - 1

def cosine_rule (a b c B : ℝ) : ℝ := (a^2 + c^2 - 2 * a * c * Real.cos B)

theorem triangle_side_b (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 4) (gB_zero : g_of_x B = 0) :
  ∃ b : ℝ, b = Real.sqrt (cosine_rule a b c B) :=
by 
  sorry

end symmetry_axis_of_f_triangle_side_b_l452_452221


namespace mary_has_more_money_than_marco_l452_452376

variable (Marco Mary : ℕ)

theorem mary_has_more_money_than_marco
    (h1 : Marco = 24)
    (h2 : Mary = 15)
    (h3 : ∃ maryNew : ℕ, maryNew = Mary + Marco / 2 - 5)
    (h4 : ∃ marcoNew : ℕ, marcoNew = Marco / 2) :
    ∃ diff : ℕ, diff = maryNew - marcoNew ∧ diff = 10 := 
by 
    sorry

end mary_has_more_money_than_marco_l452_452376


namespace circumcircle_radius_is_correct_l452_452415

noncomputable def radius_of_circumcircle (a b l : ℝ) (h : a > b) : ℝ :=
  if h : 4 * l^2 > (a - b)^2 then
    (real.sqrt (l^2 + a * b)) / (real.sqrt (4 * l^2 - (a - b)^2))
  else
    0  -- define to be zero in degenerate cases

theorem circumcircle_radius_is_correct (a b l : ℝ) (h : a > b) :
  radius_of_circumcircle a b l h =
    (real.sqrt (l^2 + a * b)) / (real.sqrt (4 * l^2 - (a - b)^2)) :=
sorry

end circumcircle_radius_is_correct_l452_452415


namespace find_apple_in_box_B_l452_452821

def apple_in_box_B (A_noting: Prop) (B_noting: Prop) (C_noting: Prop) (one_truth: Prop) : Prop :=
  (B_noting = true) ∧ (A_noting = false) ∧ (C_noting = false) ∧ one_truth

theorem find_apple_in_box_B :
  (notes: Box → Prop)
  (one_true: ∀ b:Box, notes b = true ∨ (∀ x:Box, x ≠ b → notes x = false))
  (notes Box.A = "The apple is in this box") → 
  (notes Box.B = "The apple is not in this box") → 
  (notes Box.C = "The apple is not in Box A") → 
  apple_in_box_B (notes Box.A) (notes Box.B) (notes Box.C) one_true: Prop :=
sorry

end find_apple_in_box_B_l452_452821


namespace probability_even_diff_l452_452462

theorem probability_even_diff (x y : ℕ) (hx : x ≠ y) (hx_set : x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)) (hy_set : y ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)) :
  (∃ p : ℚ, p = 5 / 22 ∧ 
    (let xy_diff_even := xy - x - y mod 2 = 0 
     in (xy_diff_even --> True))) :=
sorry

end probability_even_diff_l452_452462


namespace area_triangle_ECD_correct_l452_452410

noncomputable theory

open_locale classical

-- Definitions from conditions 
def square (a : ℝ) := a * a
def point (x y : ℝ) := (x, y)
def midpoint (p1 p2 : ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def area_triangle (A B C : ℝ × ℝ) := 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
def area_quadrilateral (A B C D : ℝ × ℝ) := area_triangle A B C + area_triangle A C D

-- Given values and points from problem conditions
def A := point 0 0
def B := point 12 0
def C := point 12 12
def D := point 0 12
def x : ℝ := 1
def E := point (3 * x) 12
def F := midpoint A B
def G := midpoint E B

-- Areas from problem conditions
def area_AFC : ℝ := 20

-- Problem statement requiring proof
theorem area_triangle_ECD_correct :
  area_triangle E C D = 115 := 
sorry

end area_triangle_ECD_correct_l452_452410


namespace circle_tangent_intersection_l452_452233

theorem circle_tangent_intersection
    {k1 k2 : Circle}
    {E X1 Y1 : Point}
    {X2 Y2 : Point}
    (touches : k1.touch k2 E)
    (on_k1 : X1 ∈ k1 ∧ Y1 ∈ k1)
    (on_k2 : X2 ∈ k2 ∧ Y2 ∈ k2)
    (intersect : ∃ M, M ∈ tangent k1 k2 ∧ Line(X1, Y1).intersects M ∧ Line(X2, Y2).intersects M) :
    ∃ M, M ∈ tangent k1 k2 ∧ 
    (∃ circle1, on_circle circle1 X1 X2 E ∧ on_circle circle1 Y1 Y2 E) ∧
    (∃ circle2, on_circle circle2 X1 Y2 E ∧ on_circle circle2 X2 Y1 E) ∧
    M ∈ intersection_point(circle1) ∧ M ∈ intersection_point(circle2) :=
sorry

end circle_tangent_intersection_l452_452233


namespace arithmetic_mean_of_primes_l452_452953

-- Define the list of numbers
def num_list : List ℕ := [33, 37, 39, 41, 43]

-- Define a predicate that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the list of prime numbers extracted from the list
def prime_list : List ℕ := (num_list.filter is_prime)

-- Define the sum of the prime numbers
def prime_sum : ℕ := prime_list.foldr (· + ·) 0

-- Define the count of prime numbers
def prime_count : ℕ := prime_list.length

-- Define the arithmetic mean of the prime numbers
def prime_mean : ℚ := prime_sum / prime_count

theorem arithmetic_mean_of_primes :
  prime_mean = 40 + 1 / 3 := sorry

end arithmetic_mean_of_primes_l452_452953


namespace find_x_l452_452691

def g (x: ℝ) := Real.sqrt ((x + 5) / 5)

theorem find_x (x: ℝ) (h: g (2 * x) = 3 * g x) : x = - (40 / 7) :=
by
  sorry

end find_x_l452_452691


namespace correct_statements_l452_452662

theorem correct_statements (a b c m x y : ℝ) :
  (¬ (a > b → ac^2 > bc^2) = false) →
  (m > 0 → (∃ x, x^2 + x - m = 0)) →
  ((x + y = 5) → (x^2 - y^2 - 3x + 7y = 10) ∧  ¬(x^2 - y^2 - 3x + 7y = 10 → x + y = 5)) →
  ({2, 3} = {2, 3}) :=
by
  intros h1 h2 h3
  have h: ({2, 3} = {2, 3}) := rfl
  exact h

end correct_statements_l452_452662


namespace circle_directrix_tangent_l452_452122

noncomputable def parabola_vertex := (0 : ℝ, 0 : ℝ)
noncomputable def parabola_focus (p : ℝ) := (0 : ℝ, p)
noncomputable def parabola_directrix (p : ℝ) := {y : ℝ | y = -p}

def on_parabola (x y p : ℝ) : Prop := y = x^2 / (4 * p)

def focal_chord (x₁ y₁ x₂ y₂ p : ℝ) : Prop := 
x₁ = -x₂ ∧ y₁ = y₂ ∧ y₁ = on_parabola x₁ y₁ p ∧ x₁^2 = 4 * p^2

theorem circle_directrix_tangent (p : ℝ) (h: p > 0) :
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
  focal_chord x₁ y₁ x₂ y₂ p ∧ 
  circle (parabola_focus p) (p) ⊆ tangent parabola_directrix p :=
sorry

end circle_directrix_tangent_l452_452122


namespace pizza_eaten_after_six_trips_l452_452034

theorem pizza_eaten_after_six_trips : 
  (∑ i in finRange 6, 1/2 ^ (i + 1)) = 63 / 64 :=
by 
  sorry

end pizza_eaten_after_six_trips_l452_452034


namespace rationalize_denominator_l452_452399

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 2) = -(Real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l452_452399


namespace total_vegetarian_is_33_l452_452706

-- Definitions of the quantities involved
def only_vegetarian : Nat := 19
def both_vegetarian_non_vegetarian : Nat := 12
def vegan_strictly_vegetarian : Nat := 3
def vegan_non_vegetarian : Nat := 2

-- The total number of people consuming vegetarian dishes
def total_vegetarian_consumers : Nat := only_vegetarian + both_vegetarian_non_vegetarian + vegan_non_vegetarian

-- Prove the number of people consuming vegetarian dishes
theorem total_vegetarian_is_33 :
  total_vegetarian_consumers = 33 :=
sorry

end total_vegetarian_is_33_l452_452706


namespace max_value_Sn_l452_452993

theorem max_value_Sn (a₁ : ℚ) (r : ℚ) (S : ℕ → ℚ)
  (h₀ : a₁ = 3 / 2)
  (h₁ : r = -1 / 2)
  (h₂ : ∀ n, S n = a₁ * (1 - r ^ n) / (1 - r))
  : ∀ n, S n ≤ 3 / 2 ∧ (∃ m, S m = 3 / 2) :=
by sorry

end max_value_Sn_l452_452993


namespace number_of_proper_subsets_is_seven_l452_452811

-- Define the set
def mySet : Set ℕ := {1, 2, 3}

-- Define the condition that will be proven
theorem number_of_proper_subsets_is_seven : 
  (Set.powerset mySet).card - 1 = 7 := by 
  sorry

end number_of_proper_subsets_is_seven_l452_452811


namespace range_of_x_l452_452257

theorem range_of_x (x m : ℝ) (h₁ : 1 ≤ m) (h₂ : m ≤ 3) (h₃ : x + 3 * m + 5 > 0) : x > -14 := 
sorry

end range_of_x_l452_452257


namespace inverse_point_l452_452224

theorem inverse_point :
  ∀ (f : ℝ → ℝ), f 0 = 4 → (∃ g : ℝ → ℝ, y = f(x + 1) → g (f (-1)) = -1).
Proof
  assume f f0_eq_4,
  sorry

end inverse_point_l452_452224


namespace find_smaller_circle_radius_l452_452771

noncomputable def radius_of_smaller_circle (R_sphere : ℝ) (R_circles : ℝ) : ℝ :=
  if R_sphere = 2 ∧ R_circles = 1 then 1 - real.sqrt (2 / 3) else 0

theorem find_smaller_circle_radius :
  radius_of_smaller_circle 2 1 = 1 - real.sqrt (2 / 3) :=
by sorry

end find_smaller_circle_radius_l452_452771


namespace min_value_of_f_l452_452971

noncomputable def f (x : ℝ) := (x^2 + 2) / (x - 1)

theorem min_value_of_f :
  ∃ x, x > 1 ∧ ∀ y, (∃ x, x > 1 ∧ y = f x) → y ≥ 2 * Real.sqrt 3 + 2 :=
begin
  -- Conditions
  have h1 : ∀ x, x > 1 → (f x = (x - 1) + 3 / (x - 1) + 2),
  { intro x, intro h,
    -- Function rewrite
    sorry
  },
  
  -- Apply AM-GM inequality
  have h2 : ∀ x, x > 1 → ((x - 1) + 3 / (x - 1) ≥ 2 * Real.sqrt 3),
  { intro x, intro h,
    -- Applying AM-GM Inequality
    sorry
  },
  
  -- Conclude minimum value
  use (Real.sqrt 3 + 1),
  split,
  { linarith [Real.sqrt 3_pos], },
  { intros y hy,
    obtain ⟨x, hx1, rfl⟩ := hy,
    linarith [h1 x hx1, h2 x hx1], }
end

end min_value_of_f_l452_452971


namespace circle_radius_l452_452216

theorem circle_radius (x y : ℝ) : (x^2 + y^2 + 2*x = 0) → ∃ r, r = 1 :=
by sorry

end circle_radius_l452_452216


namespace solve_for_z_l452_452945

theorem solve_for_z : ∃ z : ℚ, (sqrt (10 + 3 * z) = 12) ∧ (z = 134 / 3) :=
by
  sorry

end solve_for_z_l452_452945


namespace greatest_integer_less_than_PS_l452_452281

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ), PQ = 150 → 
  PS = 150 * Real.sqrt 3 → 
  (⌊PS⌋ = 259) := 
by
  intros PQ PS hPQ hPS
  sorry

end greatest_integer_less_than_PS_l452_452281


namespace triangle_incircle_angles_l452_452724

theorem triangle_incircle_angles
  (XYZ : Triangle)
  (O : Point)
  (h1 : XYZ.sides_are_tangent_to_circle O)
  (h2 : XYZ.angle XYZ.XYZ = 75)
  (h3 : XYZ.angle XYZ.YXO = 45) :
  XYZ.angle XYZ.YZX = 15 :=
sorry

end triangle_incircle_angles_l452_452724


namespace max_min_values_of_f_l452_452809

def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem max_min_values_of_f :
  ∃ (max_val min_val : ℝ), max_val = 11 ∧ min_val = 2 ∧
    (∀ x ∈ Icc (1 : ℝ) 5, f x ≤ max_val ∧ f x ≥ min_val) :=
begin
  sorry
end

end max_min_values_of_f_l452_452809


namespace product_of_repeating_decimal_and_five_l452_452924

noncomputable def repeating_decimal : ℚ :=
  456 / 999

theorem product_of_repeating_decimal_and_five : 
  (repeating_decimal * 5) = 760 / 333 :=
by
  -- The proof is omitted.
  sorry

end product_of_repeating_decimal_and_five_l452_452924


namespace water_lilies_half_pond_l452_452891

theorem water_lilies_half_pond (growth_rate : ℕ → ℕ) (start_day : ℕ) (full_covered_day : ℕ) 
  (h_growth : ∀ n, growth_rate (n + 1) = 2 * growth_rate n) 
  (h_start : growth_rate start_day = 1) 
  (h_full_covered : growth_rate full_covered_day = 2^(full_covered_day - start_day)) : 
  growth_rate (full_covered_day - 1) = 2^(full_covered_day - start_day - 1) :=
by
  sorry

end water_lilies_half_pond_l452_452891


namespace gcd_of_256_180_600_l452_452834

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l452_452834


namespace product_sum_divisibility_l452_452812

theorem product_sum_divisibility (m n : ℕ) (h : (m + n) ∣ (m * n)) (hm : 0 < m) (hn : 0 < n) : m + n ≤ n^2 :=
sorry

end product_sum_divisibility_l452_452812


namespace remainder_of_M_mod_1000_l452_452739

def M : ℕ := Nat.choose 9 8

theorem remainder_of_M_mod_1000 : M % 1000 = 9 := by
  sorry

end remainder_of_M_mod_1000_l452_452739


namespace common_ratio_of_geometric_sequence_l452_452298

variables {a₁ a₄ q : ℝ}

-- Define the initial conditions
def a₁ : ℝ := 2 / 3
def a₄ : ℝ := ∫ x in 1..4, 1 + 2 * x

-- Define the target statement to be proved
theorem common_ratio_of_geometric_sequence :
  a₁ = 2 / 3 →
  a₄ = ∫ x in 1..4, 1 + 2 * x →
  q = 3 ↔ q^3 = (a₄ / a₁) :=
by
  intros h1 h2
  -- This is where you would normally begin the proof.
  sorry

end common_ratio_of_geometric_sequence_l452_452298


namespace f_value_l452_452167

noncomputable def f (n : ℕ) : ℕ :=
  if n = 1 then 0 else
  if n = 2 then 3 else n + 2

theorem f_value (n : ℕ) : 
  ∀ (A : Finset (Finset ℕ)), (∀ i j ∈ A, i ≠ j → ¬ (i ⊆ j) ∧ i.card ≠ j.card) → 
  |A.bind id| = f n :=
by sorry

end f_value_l452_452167


namespace mary_has_more_l452_452377

theorem mary_has_more (marco_initial mary_initial : ℕ) (h1 : marco_initial = 24) (h2 : mary_initial = 15) :
  let marco_final := marco_initial - 12,
      mary_final := mary_initial + 12 - 5 in
  mary_final = marco_final + 10 :=
by
  sorry

end mary_has_more_l452_452377


namespace measure_angle_MAB_l452_452628

section EquilateralTriangle

variables {A B C M : Type} [AffineSpace A] [Triangle ABC : Equilateral A B C]

-- Declaring that the angles AMB and AMC have specific measures.
axiom angle_AMB : angle A M B = 20
axiom angle_AMC : angle A M C = 30

-- Main theorem we need to prove
theorem measure_angle_MAB (h : EquilateralTriangle A B C) (external_M : external_triangle M ABC) :
  angle M A B = 20 :=
by
  sorry

end EquilateralTriangle

end measure_angle_MAB_l452_452628


namespace wings_area_l452_452783

-- Define the areas of the two cut triangles
def A1 : ℕ := 4
def A2 : ℕ := 9

-- Define the area of the wings (remaining two triangles)
def W : ℕ := 12

-- The proof goal
theorem wings_area (A1 A2 : ℕ) (W : ℕ) : A1 = 4 → A2 = 9 → W = 12 → A1 + A2 = 13 → W = 12 :=
by
  intros hA1 hA2 hW hTotal
  -- Sorry is used as a placeholder for the proof steps
  sorry

end wings_area_l452_452783


namespace ball_placement_problem_l452_452774

noncomputable def num_ways_to_place_balls : ℕ :=
(choose 8 5) * (choose 3 2) * (choose 1 1) * (factorial 3) +
(choose 8 4) * (choose 4 3) * (choose 1 1) * (factorial 3)

theorem ball_placement_problem : num_ways_to_place_balls = 2688 :=
by sorry

end ball_placement_problem_l452_452774


namespace monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l452_452665

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin (2 * x + (Real.pi / 4)))

theorem monotonic_intervals_increasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + Real.pi / 8) → f x ≤ f y :=
sorry

theorem monotonic_intervals_decreasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + 5 * Real.pi / 8) → f x ≥ f y :=
sorry

theorem maximum_value (k : ℤ) :
  f (k * Real.pi + Real.pi / 8) = 3 :=
sorry

theorem minimum_value (k : ℤ) :
  f (k * Real.pi - 3 * Real.pi / 8) = -3 :=
sorry

end monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l452_452665


namespace evaluate_expression_l452_452132

theorem evaluate_expression : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end evaluate_expression_l452_452132


namespace sum_of_integers_satisfying_condition_l452_452006

theorem sum_of_integers_satisfying_condition :
  (∑ x in {x : ℤ | x * x = x + 245}.toFinset, x) = 3 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_integers_satisfying_condition_l452_452006


namespace john_has_more_pennies_l452_452340

theorem john_has_more_pennies (kate_pennies john_pennies : ℕ) (h_kate : kate_pennies = 223) (h_john : john_pennies = 388) : john_pennies - kate_pennies = 165 :=
by
  rw [h_kate, h_john]
  norm_num

end john_has_more_pennies_l452_452340


namespace Find_N_l452_452689

noncomputable def N := 
  (sqrt (sqrt 7 + 3) + sqrt (sqrt 7 - 3)) / sqrt (sqrt 7 + 2) - sqrt (5 - 2 * sqrt 6)

theorem Find_N : N = 1 - (sqrt 6 - sqrt 2) := by
  sorry

end Find_N_l452_452689


namespace number_of_toys_bought_l452_452391

def toy_cost (T : ℕ) : ℕ := 10 * T
def card_cost : ℕ := 2 * 5
def shirt_cost : ℕ := 5 * 6
def total_cost (T : ℕ) : ℕ := toy_cost T + card_cost + shirt_cost

theorem number_of_toys_bought (T : ℕ) : total_cost T = 70 → T = 3 :=
by
  intro h
  sorry

end number_of_toys_bought_l452_452391


namespace prob_dist_of_ξ_expected_value_of_ξ_prob_ξ_leq_1_l452_452627

open ProbabilityMassFunction

-- define the context
def students : Finset ℕ := {0, 1, 2, 3, 4, 5} -- 0-3 are males, 4-5 are females
def males : Finset ℕ := {0, 1, 2, 3}
def females : Finset ℕ := {4, 5}
def choose3 : Finset (Finset ℕ) := students.powerset.filter (λ s, s.card = 3)

-- define the random variable ξ (xi)
def ξ (s : Finset ℕ) : ℕ := s.card ∩ females.card

-- Total ways to choose 3 out of 6 students
def total_ways := choose3.card

-- Define pmf for ξ
def pmf_ξ (k : ℕ) : ℚ := 
  (choose3.filter (λ s, ξ s = k)).card / total_ways

-- Proof problem statements
theorem prob_dist_of_ξ : 
  pmf_ξ 0 = 1/5 ∧ pmf_ξ 1 = 3/5 ∧ pmf_ξ 2 = 1/5 :=
by sorry

theorem expected_value_of_ξ : 
  (0 * pmf_ξ 0 + 1 * pmf_ξ 1 + 2 * pmf_ξ 2) = 1 :=
by sorry

theorem prob_ξ_leq_1 : 
  (pmf_ξ 0 + pmf_ξ 1) = 4/5 :=
by sorry

end prob_dist_of_ξ_expected_value_of_ξ_prob_ξ_leq_1_l452_452627


namespace shopkeeper_gain_percent_l452_452070

theorem shopkeeper_gain_percent
    (SP₁ SP₂ CP : ℝ)
    (h₁ : SP₁ = 187)
    (h₂ : SP₂ = 264)
    (h₃ : SP₁ = 0.85 * CP) :
    ((SP₂ - CP) / CP) * 100 = 20 := by 
  sorry

end shopkeeper_gain_percent_l452_452070


namespace train_length_is_approximately_110_l452_452559

noncomputable def length_of_train 
  (speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (time_seconds : ℝ) : ℝ := 
  let speed_mps := speed_kmph * 1000 / 3600 in
  let total_distance := speed_mps * time_seconds in
  total_distance - bridge_length

theorem train_length_is_approximately_110 
  (h1 : speed_kmph = 60) 
  (h2 : bridge_length = 390) 
  (h3 : time_seconds = 29.997600191984642) : 
  |length_of_train speed_kmph bridge_length time_seconds - 110| < 1 * 10^(-9) :=
by 
  sorry

end train_length_is_approximately_110_l452_452559


namespace count_linear_equations_l452_452566

-- Defining the conditions:
def is_linear_equation_in_two_variables (eq : String) : Bool :=
  eq = "2x - 3y = 5" || eq = "x + 3 / y = 6" || eq = "3x - y + 2z = 0" || 
  eq = "2x + 4y" || eq = "5x - y > 0"

-- The actual equations to check
def equations : List String := ["2x - 3y = 5", "x + 3 / y = 6", "3x - y + 2z = 0", 
                                "2x + 4y", "5x - y > 0"]

-- The property of being a linear equation in two variables
theorem count_linear_equations : Nat := by
  let count := equations.foldl (λ acc eq, if eq = "ax + by = c" then acc + 1 else acc) 0
  have h : count = 1 := by sorry
  exact h

end count_linear_equations_l452_452566


namespace max_magnitude_of_c_l452_452651

open Real

variables (a b c : ℝ ^ 2)

/-- Mutually perpendicular unit vectors on a plane. -/
def is_unit_perpendicular (a b : ℝ ^ 2) : Prop := (‖a‖ = 1) ∧ (‖b‖ = 1) ∧ (a ⬝ b = 0)

/-- Condition of the problem. -/
def given_condition (a b c : ℝ ^ 2) : Prop := (c - a) ⬝ (c - b) = 0

/-- The maximum value of |c|. -/
theorem max_magnitude_of_c (h1 : is_unit_perpendicular a b) (h2 : given_condition a b c) :
  ‖c‖ ≤ √2 :=
sorry

end max_magnitude_of_c_l452_452651


namespace parallelogram_area_l452_452583

/-- Given four points A = (2, -20), B = (1010, 90), C = (1012, 100), and D = (4, y)
forming a parallelogram ABCD with AD parallel to BC, and the slope of AB is 1/5,
prove that the area of ABCD is 220. -/
theorem parallelogram_area
  (y : ℝ)
  (h_parallelogram : ∃ (AD BC : ℝ × ℝ), AD = ⟨4 - 2, y + 20⟩ ∧ BC = ⟨1002, 10⟩ ∧ (AD.2 / AD.1) = (BC.2 / BC.1))
  (h_slope_AB : (90 - (-20)) / (1010 - 2) = 1 / 5) :
  let AB : ℝ × ℝ := (1010 - 2, 90 + 20)
      AD : ℝ × ℝ := (2, 0)
    in abs (AB.1 * AD.2 - AB.2 * AD.1) = 220 := by
  sorry

end parallelogram_area_l452_452583


namespace construct_rectangles_on_sides_l452_452054

structure Hexagon :=
  (A B C D E F : Type)
  (convex : Prop)

def vector_sum_property (hex : Hexagon) (BC DE FA : hex.C) : Prop :=
  BC + DE + FA = 0

def cosine_property (hex : Hexagon) (αA αB αC αD αE αF : ℝ) : Prop :=
  cos αA * cos αC * cos αE = cos αB * cos αD * cos αF

theorem construct_rectangles_on_sides
  (hex : Hexagon)
  (BC DE FA : hex.C)
  (AB CD EF : hex.C)
  (αA αB αC αD αE αF : ℝ)
  (h1 : vector_sum_property hex BC DE FA)
  (h2 : cosine_property hex αA αB αC αD αE αF) :
  vector_sum_property hex AB CD EF ∧ cosine_property hex αA αB αC αD αE :=
sorry

end construct_rectangles_on_sides_l452_452054


namespace B_completes_in_30_days_l452_452858

-- Definitions of conditions
def A_completes_in (n : ℕ) : Prop := n = 15
def A_and_B_completes_in (n : ℕ) : Prop := n = 10

-- Main theorem to prove
theorem B_completes_in_30_days : 
  (A_completes_in 15 ∧ A_and_B_completes_in 10) → ∃ x : ℕ, x = 30 :=
by {
  intro h,
  sorry
}

end B_completes_in_30_days_l452_452858


namespace sqrt_calc_correct_l452_452855

theorem sqrt_calc_correct : 
  (sqrt 3 * sqrt 5 = sqrt 15) := 
by
  sorry

end sqrt_calc_correct_l452_452855


namespace simplify_fraction_l452_452785

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (10 * x * y^2) / (5 * x * y) = 2 * y := 
by
  sorry

end simplify_fraction_l452_452785


namespace product_of_midpoint_coordinates_l452_452478

theorem product_of_midpoint_coordinates
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 4) (h2 : y1 = -3) (h3 : x2 = -8) (h4 : y2 = 7) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx * my = -4) :=
by
  -- Here we would carry out the proof.
  sorry

end product_of_midpoint_coordinates_l452_452478


namespace unit_vector_AB_l452_452198

def point := ℝ × ℝ

def vector_sub (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def magnitude (v : point) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def unit_vector (v : point) : point := (v.1 / magnitude v, v.2 / magnitude v)

def A : point := (1, 3)
def B : point := (4, -1)

def AB : point := vector_sub A B

theorem unit_vector_AB : unit_vector AB = (3/5, -4/5) := sorry

end unit_vector_AB_l452_452198


namespace slope_of_perpendicular_line_l452_452159

-- Define the line equation as a condition
def line_eqn (x y : ℝ) : Prop := 4 * x - 6 * y = 12

-- Define the slope of the given line from its equation
noncomputable def original_slope : ℝ := 2 / 3

-- Define the negative reciprocal of the original slope
noncomputable def perp_slope (m : ℝ) : ℝ := -1 / m

-- State the theorem
theorem slope_of_perpendicular_line : perp_slope original_slope = -3 / 2 :=
by 
  sorry

end slope_of_perpendicular_line_l452_452159


namespace isosceles_triangle_possible_values_of_x_l452_452432

open Real

-- Define the main statement
theorem isosceles_triangle_possible_values_of_x :
  ∀ x : ℝ, 
  (0 < x ∧ x < 90) ∧ 
  (sin (3*x) = sin (2*x) ∧ 
   sin (9*x) = sin (2*x)) 
  → x = 0 ∨ x = 180/11 ∨ x = 540/11 :=
by
  sorry

end isosceles_triangle_possible_values_of_x_l452_452432


namespace exists_m_for_all_digits_l452_452647

theorem exists_m_for_all_digits (n : ℕ) (hn : 0 < n) : ∃ m : ℕ, (∀ d : ℕ, d < 10 → (∃ q : ℕ, q = 10^d) ∧ (∀ k : ℕ, k < q → (mn = k → (mn % 10) = d)))) := 
sorry

end exists_m_for_all_digits_l452_452647


namespace tetrahedron_edge_length_CD_l452_452557

theorem tetrahedron_edge_length_CD 
    (AB AC AD BC BD CD : ℕ)
    (edges : {7, 13, 18, 27, 36, 41} = {AB, AC, AD, BC, BD, CD})
    (hAB : AB = 41) : CD = 13 := 
by 
  sorry

end tetrahedron_edge_length_CD_l452_452557


namespace water_left_after_four_hours_l452_452683

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def added_water_hour3 : ℕ := 1
def added_water_hour4 : ℕ := 3
def total_hours : ℕ := 4

theorem water_left_after_four_hours :
  initial_water - (water_loss_per_hour * total_hours) + (added_water_hour3 + added_water_hour4) = 36 := by
  sorry

end water_left_after_four_hours_l452_452683


namespace sum_an_4999_l452_452975

def a (n : ℕ) : ℕ :=
  if n % 18 = 0 ∧ n % 21 = 0 then 15
  else if n % 21 = 0 ∧ n % 15 = 0 then 18
  else if n % 15 = 0 ∧ n % 18 = 0 then 21
  else 0

def sum_an_up_to (k : ℕ) : ℕ :=
  ∑ n in Finset.range (k + 1), a n

theorem sum_an_4999 : sum_an_up_to 4999 = 2586 :=
  by sorry

end sum_an_4999_l452_452975


namespace evaluate_expression_l452_452133

theorem evaluate_expression : 500 * (500 ^ 500) * 500 = 500 ^ 502 := by
  sorry

end evaluate_expression_l452_452133


namespace sum_of_odds_eq_square_l452_452531

theorem sum_of_odds_eq_square (n : ℕ) (h : 0 < n) : List.sum (List.map (λ i, 2 * i - 1) (List.range n)) = n^2 := by
  sorry

end sum_of_odds_eq_square_l452_452531


namespace quadrant_of_z_l452_452935

-- Define the imaginary unit
def i : ℂ := complex.I

-- Define the complex number z
def z : ℂ := i + (i ^ 2)

-- Prove that z is in the second quadrant
theorem quadrant_of_z : z.re < 0 ∧ z.im > 0 := 
by {
  -- Value of i and i^2 (Lean's complex.I is already defined correctly)
  have hi2 : i^2 = -1, from by simp [complex.I_mul_I],
  -- Calculation of z
  have hz : z = i + (-1), by simp [hi2],
  -- Splitting z into real and imaginary parts
  split_ifs,
  -- Real part condition
  have hz_re : z.re = -1, by simp [hz],
  -- Imaginary part condition
  have hz_im : z.im = 1, by simp [hz],
  -- Asserting final conditions
  rw [hz_re, hz_im]
}

end quadrant_of_z_l452_452935


namespace rearrange_numbers_25_rearrange_numbers_1000_l452_452313

theorem rearrange_numbers_25 (n : ℕ) (h : n = 25) : 
  ∃ (f : Fin n → Fin n), ∀ i : Fin (n - 1), (f i.succ.val - f i.val = 3 ∨ f i.succ.val - f i.val = 5 ∨ f i.val - f i.succ.val = 3 ∨ f i.val - f i.succ.val = 5) ∧ (f i).val ∈ Finset.range n := by
  sorry

theorem rearrange_numbers_1000 (n : ℕ) (h : n = 1000) : 
  ∃ (f : Fin n → Fin n), ∀ i : Fin (n - 1), (f i.succ.val - f i.val = 3 ∨ f i.succ.val - f i.val = 5 ∨ f i.val - f i.succ.val = 3 ∨ f i.val - f i.succ.val = 5) ∧ (f i).val ∈ Finset.range n := by
  sorry

end rearrange_numbers_25_rearrange_numbers_1000_l452_452313


namespace correct_statement_unit_vector_l452_452511

variables {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem correct_statement_unit_vector :
  (∀ u : V, ∥u∥ = 1 → ∥u∥ = 1) :=
by
  intro u
  intro h
  exact h

#check correct_statement_unit_vector -- Ensure the statement can be built successfully

end correct_statement_unit_vector_l452_452511


namespace inequalities_not_equivalent_l452_452087

theorem inequalities_not_equivalent (x : ℝ) : 
  (log (x^2 - 4) > log (4 * x - 7)) ≠ (x^2 - 4 > 4 * x - 7) :=
by
  sorry

end inequalities_not_equivalent_l452_452087


namespace root_product_squared_eq_3600_l452_452097

theorem root_product_squared_eq_3600:
  (∛(125) * (16^(1/4)) * sqrt 36)^2 = 3600 :=
by
  have h1: ∛(125) = 5 := by norm_num [root_of, pow_succ],
  have h2: (16^(1/4)) = 2 := by norm_num [pow_succ],
  have h3: sqrt 36 = 6 := by norm_num [sqrt_eq_iff_sq_eq],
  calc (∛(125) * (16^(1/4)) * sqrt 36)^2
      = (5 * 2 * 6)^2      : by rw [h1, h2, h3]
  ... = 60^2               : by norm_num
  ... = 3600               : by norm_num

end root_product_squared_eq_3600_l452_452097


namespace smallest_y_absolute_value_equation_l452_452161

theorem smallest_y_absolute_value_equation :
  ∃ y : ℚ, (|5 * y - 9| = 55) ∧ y = -46 / 5 :=
by
  sorry

end smallest_y_absolute_value_equation_l452_452161


namespace volume_of_pyramid_PABCDEFGH_l452_452403

-- Definitions from the conditions
def regular_octagon (A B C D E F G H : Point) : Prop := 
  -- A formal definition of a regular octagon would be here
  sorry

def equilateral_triangle (P A D : Point) (side_length : ℝ) : Prop := 
  -- A formal definition of an equilateral triangle would be here
  sorry

def right_pyramid (P A B C D E F G H : Point) : Prop := 
  -- A formal definition of a right pyramid would be here
  sorry

-- Given conditions
constants 
  (P A B C D E F G H : Point)
  (side_length : ℝ)
  (h1 : regular_octagon A B C D E F G H)
  (h2 : right_pyramid P A B C D E F G H)
  (h3 : equilateral_triangle P A D 10)

-- Prove the volume of the pyramid
theorem volume_of_pyramid_PABCDEFGH : 
  volume_of_pyramid P A B C D E F G H = 1000 := 
sorry

end volume_of_pyramid_PABCDEFGH_l452_452403


namespace inappropriate_survey_method_l452_452032

-- Define the conditions
def option_A : Prop := "Conduct a sampling survey to understand the water quality of the Lishui River."
def option_B : Prop := "Conduct a comprehensive survey to understand the service life of a batch of light bulbs."
def option_C : Prop := "Conduct a sampling survey to understand the sleep time of students in Zhangjiajie City."
def option_D : Prop := "Conduct a comprehensive survey to understand the math scores of a class of students."

-- Define the proposition stating the conditions
def conditions : Prop := 
  (option_A → true) ∧ 
  (option_B → false) ∧
  (option_C → true) ∧
  (option_D → true)

-- State the theorem
theorem inappropriate_survey_method : conditions → option_B := 
by
  intros,
  sorry

end inappropriate_survey_method_l452_452032


namespace intersection_sine_dot_product_l452_452656

theorem intersection_sine_dot_product (a b c θ : ℝ) (hP : ∃ k ∈ ℤ, a = (π * k - θ) / 2 )
  (hA : b = a + π / 4) (hB : c = a + 3 * π / 4) : 
  let PA := (b - a, 1)
      PB := (c - a, -1) in
  PA.1 * PB.1 + PA.2 * PB.2 = 3 * π^2 / 16 - 1 :=
by {
  sorry
}

end intersection_sine_dot_product_l452_452656


namespace value_of_y_at_x_8_l452_452693

theorem value_of_y_at_x_8 (k : ℝ) (x y : ℝ) 
  (hx1 : y = k * x^(1/3)) 
  (hx2 : y = 4 * Real.sqrt 3) 
  (hx3 : x = 64) 
  (hx4 : 8^(1/3) = 2) : 
  (y = 2 * Real.sqrt 3) := 
by 
  sorry

end value_of_y_at_x_8_l452_452693


namespace symmetry_center_of_tangent_l452_452816

noncomputable def symmetry_center (k : ℤ) : ℝ × ℝ :=
  (k * Real.pi / 4, 0)

theorem symmetry_center_of_tangent (k : ℤ) :
  let f : ℝ → ℝ := λ x, 3 * Real.tan (2 * x) in
  symmetry_center k = (k * Real.pi / 4, 0) :=
by
  -- Sorry, skipping the actual proof
  sorry

end symmetry_center_of_tangent_l452_452816


namespace rationalize_denominator_l452_452401

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 2) = -(Real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l452_452401


namespace num_elements_in_S_l452_452747

noncomputable def f (x : ℝ) : ℝ :=
  (x + 5) / x

def f_seq : ℕ → (ℝ → ℝ)
| 1       := f
| (n + 1) := f ∘ f_seq n

def is_fixed_point (f_n : ℝ → ℝ) (x : ℝ) : Prop :=
  f_n x = x

def S : set ℝ := { x | ∃ n : ℕ, n > 0 ∧ is_fixed_point (f_seq n) x }

theorem num_elements_in_S : S.card = 2 :=
sorry

end num_elements_in_S_l452_452747


namespace tv_program_reform_confidence_l452_452822

theorem tv_program_reform_confidence :
  ∀ (k : ℝ) (threshold : ℝ),
    k = 6.89 →
    threshold = 6.635 →
    k > threshold →
    (∃ (confidence_level : ℝ), confidence_level = 0.99 ∧
      (confidence_level = 0.99 → 
        "There is 99% confidence that whether the TV program is excellent is related to the reform")
    ) :=
by
  intros k threshold hk hthreshold hcomp
  use 0.99
  split
  repeat { sorry }

end tv_program_reform_confidence_l452_452822


namespace symmetric_point_correct_l452_452867

-- Define the coordinates of point A
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D :=
  { x := -3
    y := -4
    z := 5 }

-- Define the symmetry function with respect to the plane xOz
def symmetric_xOz (p : Point3D) : Point3D :=
  { p with y := -p.y }

-- The expected coordinates of the point symmetric to A with respect to the plane xOz
def D_expected : Point3D :=
  { x := -3
    y := 4
    z := 5 }

-- Theorem stating that the symmetric point of A with respect to the plane xOz is D_expected
theorem symmetric_point_correct :
  symmetric_xOz A = D_expected := 
by 
  sorry

end symmetric_point_correct_l452_452867


namespace sin_eq_sqrt3_div_2_range_l452_452299

theorem sin_eq_sqrt3_div_2_range :
  {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x ≥ Real.sqrt 3 / 2} = 
  {x | Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3} :=
sorry

end sin_eq_sqrt3_div_2_range_l452_452299


namespace T_seq_correct_l452_452366

def a_seq (n: ℕ) : ℕ := 2^n
def b_seq (n : ℕ) : ℕ := 2^n + 3 * n - 2
def T_seq (n : ℕ) : ℕ := 2^(n+1) - 2 + (3 * n^2 / 2) - (n / 2)

theorem T_seq_correct (n : ℕ) (hB : ∀ n, b_seq n - a_seq n = 3 * n - 2) : 
  (λ n, T_seq n) = (λ n, 2^(n+1) - 2 + (3 * n^2 / 2) - (n / 2)) :=
by {
  -- Using the provided conditions and the known values for sequences, we prove the formula for T_seq.
  sorry
}

end T_seq_correct_l452_452366


namespace gcd_of_256_180_600_l452_452845

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l452_452845


namespace circle_passes_first_and_second_quadrants_l452_452715

theorem circle_passes_first_and_second_quadrants :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = 4 → ((x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≥ 0)) :=
by
  sorry

end circle_passes_first_and_second_quadrants_l452_452715


namespace product_of_midpoint_coordinates_l452_452477

theorem product_of_midpoint_coordinates
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 4) (h2 : y1 = -3) (h3 : x2 = -8) (h4 : y2 = 7) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx * my = -4) :=
by
  -- Here we would carry out the proof.
  sorry

end product_of_midpoint_coordinates_l452_452477


namespace unit_vector_same_direction_l452_452196

-- Define the coordinates of points A and B as given in the conditions
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Define the vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of vector AB
noncomputable def magnitudeAB : ℝ := Real.sqrt (vectorAB.1^2 + vectorAB.2^2)

-- Define the unit vector in the direction of AB
noncomputable def unitVectorAB : ℝ × ℝ := (vectorAB.1 / magnitudeAB, vectorAB.2 / magnitudeAB)

-- The theorem we want to prove
theorem unit_vector_same_direction :
  unitVectorAB = (3 / 5, -4 / 5) :=
sorry

end unit_vector_same_direction_l452_452196


namespace circumcircle_common_point_triangle_EKG_isosceles_l452_452717

-- Definitions of the geometrical objects and properties
variable (A B C O D E K F G : Point)
variable (hAcute : AcuteAngleTriangle A B C)
variable (hO : Circumcenter O A B C)
variable (hPerpendicular : Perpendicular O AO)
variable (hD : IntersectAt D AB AO)
variable (hE : IntersectAt E AC AO)
variable (hKO : IsDiameter O K AO)
variable (hEXTCO : ExtendIntersectsAt C O F (Circumcircle A B C))
variable (hEXTED : ExtendIntersectsAt E D G (LineSegment F K))

-- The proof goals, represented as Lean theorems
theorem circumcircle_common_point :
  ∃ K, OnCircumcircle K A B C ∧ OnCircumcircle K O D B ∧ OnCircumcircle K O E C := sorry

theorem triangle_EKG_isosceles :
  IsIsoscelesTriangle E K G := sorry

end circumcircle_common_point_triangle_EKG_isosceles_l452_452717


namespace exists_balanced_sequence_set_l452_452621

theorem exists_balanced_sequence_set (n : ℕ) (h_pos : 0 < n) :
  ∃ S : Finset (Fin (2 * n) → Fin 2), 
    S.card ≤ Nat.choose (2 * n) n / (n + 1) ∧
    ∀ a : Fin (2 * n) → Fin 2, 
      (is_balanced a n) → 
      (a ∈ S ∨ ∃ b : Fin (2 * n) → Fin 2, (b ∈ S ∧ is_neighbor a b)) :=
sorry

-- Auxiliary definitions:
def is_balanced (a : Fin (2 * n) → Fin 2) (n : ℕ) : Prop :=
  a.to_multiset.count 0 = n ∧ a.to_multiset.count 1 = n

def is_neighbor (a b : Fin (2 * n) → Fin 2) : Prop :=
  ∃ i j : Fin (2 * n), a j = b i ∧ 
                        ∀ k : Fin (2 * n), k ≠ i → 
                          (a k = b k ∧ (i = j ∨ a i = b j))

end exists_balanced_sequence_set_l452_452621


namespace problem_l452_452352

noncomputable def roots1 : Set ℝ := { α | α^2 - 2*α + 1 = 0 }
noncomputable def roots2 : Set ℝ := { γ | γ^2 - 3*γ + 1 = 0 }

theorem problem 
  (α β γ δ : ℝ) 
  (hαβ : α ∈ roots1 ∧ β ∈ roots1)
  (hγδ : γ ∈ roots2 ∧ δ ∈ roots2) : 
  (α - γ)^2 * (β - δ)^2 = 1 := 
sorry

end problem_l452_452352


namespace bills_head_circumference_l452_452331

/-- Jack is ordering custom baseball caps for him and his two best friends, and we need to prove the circumference of Bill's head. -/
theorem bills_head_circumference (Jack : ℝ) (Charlie : ℝ) (Bill : ℝ)
  (h1 : Jack = 12)
  (h2 : Charlie = (1 / 2) * Jack + 9)
  (h3 : Bill = (2 / 3) * Charlie) :
  Bill = 10 :=
by sorry

end bills_head_circumference_l452_452331


namespace max_f_l452_452187

noncomputable def f (a x : ℝ) : ℝ := a * real.sqrt (1 - x^2) + real.sqrt (1 + x) + real.sqrt (1 - x)

-- Define g(a) as given in the solution
noncomputable def g (a : ℝ) : ℝ :=
if a ≤ -real.sqrt 2 / 2 then real.sqrt 2
else if -real.sqrt 2 / 2 < a ∧ a ≤ -1 / 2 then -a - 1 / (2 * a)
else a + 2

-- State the theorem
theorem max_f (a : ℝ) (h_neg_a : a < 0) : 
  ∃ x ∈ set.Icc (-1 : ℝ) 1, f a x = g a :=
sorry

end max_f_l452_452187


namespace limit_sin_sq_l452_452527

theorem limit_sin_sq:
  (tendsto (λ x : ℝ, (2 * (sin x)^2 + sin x - 1) / (2 * (sin x)^2 - 3 * sin x + 1))
           (𝓝 (π / 6))
           (𝓝 (-3))) := 
sorry

end limit_sin_sq_l452_452527


namespace sum_of_projections_equals_half_nR2_OX2_l452_452895

-- Define vector operations and related structures
variables {n : ℕ} (R : ℝ) (OX : ℝ) (θ : ℝ)

-- Define the condition that O A_i are the position vectors of the vertices of a regular n-gon.
def regular_ngon_position_vector (R : ℝ) (i : ℕ) : ℝ := R * (Real.cos ((2 * i * π) / n) + Real.sin ((2 * i * π) / n))

-- Define the arbitrary vector O X
variable (x : ℝ)

-- Define the expression to sum over the projection
def sum_of_projections (OX : ℝ) (R : ℝ) : ℝ :=
  ∑ i in finset.range n, (R * ((Real.cos ((2 * i * π) / n) * OX) + (Real.sin ((2 * i * π) / n) * OX))) ^ 2

theorem sum_of_projections_equals_half_nR2_OX2 (n : ℕ) (R : ℝ) (OX : ℝ) :
  sum_of_projections OX R = (n * R^2 * OX^2) / 2 := sorry

end sum_of_projections_equals_half_nR2_OX2_l452_452895


namespace triangle_inequality_l452_452752

-- Let α, β, γ be the angles of a triangle opposite to its sides with lengths a, b, and c, respectively.
variables (α β γ a b c : ℝ)

-- Assume that α, β, γ are positive.
axiom positive_angles : α > 0 ∧ β > 0 ∧ γ > 0
-- Assume that a, b, c are the sides opposite to angles α, β, γ respectively.
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_inequality :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 
  2 * (a / α + b / β + c / γ) :=
sorry

end triangle_inequality_l452_452752


namespace fraction_equality_l452_452173

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) → x = -1 :=
by
  sorry

end fraction_equality_l452_452173


namespace mooney_ate_correct_l452_452386

-- Define initial conditions
def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mother_added : ℕ := 24
def final_brownies : ℕ := 36

-- Define Mooney ate some brownies
variable (mooney_ate : ℕ)

-- Prove that Mooney ate 4 brownies
theorem mooney_ate_correct :
  (initial_brownies - father_ate) - mooney_ate + mother_added = final_brownies →
  mooney_ate = 4 :=
by
  sorry

end mooney_ate_correct_l452_452386


namespace four_people_same_number_of_flips_probability_l452_452081

theorem four_people_same_number_of_flips_probability :
  let p := (1:ℝ) / 2
  in ∑' n : ℕ, p^(4 * (n + 1)) = 1 / 15 :=
by
  sorry

end four_people_same_number_of_flips_probability_l452_452081


namespace geometric_series_convergence_l452_452932

theorem geometric_series_convergence :
  let a := 3
  let r := (1:ℝ)/4
  let S := ∑' n : ℕ, a * r^n
  (S = 4) ∧ ((∀ ε > 0, |S - 4| < ε) ∧ (∃ l, S = l)) :=
by
  let a := 3
  let r := (1:ℝ)/4
  let S := ∑' n : ℕ, a * r^n
  have S_eq : S = a / (1 - r) := sorry
  have two_limit_statements : (∀ ε > 0, |S - 4| < ε) ∧ (∃ l, S = l) := sorry
  exact ⟨S_eq, two_limit_statements⟩

end geometric_series_convergence_l452_452932


namespace det_value_l452_452604

open Matrix

noncomputable def det_example (α β : ℝ) : ℝ :=
  det ![
    ![0, Real.cos α, Real.sin α],
    ![Real.sin α, 0, Real.cos β],
    ![-Real.cos α, -Real.sin β, 0]
  ]

theorem det_value (α β : ℝ) : 
  det_example α β = -(Real.cos β * Real.cos α ^ 2 + Real.sin β * Real.sin α ^ 2) :=
by
  sorry

end det_value_l452_452604


namespace PQRS_perimeter_PQRS_area_PQRS_properties_l452_452757

-- Definitions for the points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (3, 6)
def R : ℝ × ℝ := (6, 3)
def S : ℝ × ℝ := (8, 1)

-- Function to calculate the distance between two points
def distance (A B : ℝ × ℝ) : ℝ :=
real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- The calculated distances
def PQ := distance P Q
def QR := distance Q R
def RS := distance R S
def SP := distance S P

-- Statement for the perimeter
theorem PQRS_perimeter : PQ + QR + RS + SP = 10 * real.sqrt 2 + 2 * real.sqrt 5 :=
sorry

-- Shoelace formula for the area of a quadrilateral
def shoelace_area (A B C D : ℝ × ℝ) : ℝ :=
0.5 * (abs (A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2 - (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1)))

-- Validating the area using the shoelace formula
theorem PQRS_area : shoelace_area P Q R S = 15 :=
sorry

-- Final combined statement
theorem PQRS_properties : (PQ + QR + RS + SP = 10 * real.sqrt 2 + 2 * real.sqrt 5) ∧ (shoelace_area P Q R S = 15) :=
⟨PQRS_perimeter, PQRS_area⟩

end PQRS_perimeter_PQRS_area_PQRS_properties_l452_452757


namespace mary_more_than_marco_l452_452382

def marco_initial : ℕ := 24
def mary_initial : ℕ := 15
def half_marco : ℕ := marco_initial / 2
def mary_after_give : ℕ := mary_initial + half_marco
def mary_after_spend : ℕ := mary_after_give - 5
def marco_final : ℕ := marco_initial - half_marco

theorem mary_more_than_marco :
  mary_after_spend - marco_final = 10 := by
  sorry

end mary_more_than_marco_l452_452382


namespace ants_no_collision_probability_l452_452937

/-- Eight ants are standing simultaneously on the eight vertices of a regular cube.
Each ant independently moves from its vertex to one of the three adjacent vertices. 
This lemma proves that the probability of no two ants arriving at the same vertex is 240/6561. -/
theorem ants_no_collision_probability : (240 : ℚ) / 6561 = sorry

end ants_no_collision_probability_l452_452937


namespace matrix_max_sum_l452_452582

def B (x y z : ℤ) : Matrix (Fin 2) (Fin 2) ℚ :=
  (1 / 7 : ℚ) • ![![ -4, x], ![ y, z]]

def I2 : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, 0], ![0, 2]]

theorem matrix_max_sum (x y z : ℤ) (H : (B x y z) ⬝ (B x y z) = I2) : 
  ∃ x y z, x + y + z = 87 :=
sorry

end matrix_max_sum_l452_452582


namespace set_intersection_eq_singleton_l452_452232

-- Conditions
def M : Set ℝ := { x | abs (x - 3) < 4 }
def N : Set ℤ := { x | (x : ℝ)^2 + (x : ℝ) - 2 < 0 }

-- Goal
theorem set_intersection_eq_singleton :
  M ∩ (N : Set ℝ) = {0} :=
sorry

end set_intersection_eq_singleton_l452_452232


namespace inverse_proportion_y_relation_l452_452200

theorem inverse_proportion_y_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (hA : y₁ = -4 / x₁) 
  (hB : y₂ = -4 / x₂)
  (h₁ : x₁ < 0) 
  (h₂ : 0 < x₂) : 
  y₁ > y₂ := 
sorry

end inverse_proportion_y_relation_l452_452200


namespace dogs_neither_long_furred_nor_brown_l452_452267

theorem dogs_neither_long_furred_nor_brown :
  (∀ (total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown : ℕ),
     total_dogs = 45 →
     long_furred_dogs = 26 →
     brown_dogs = 22 →
     both_long_furred_and_brown = 11 →
     neither_long_furred_nor_brown = total_dogs - (long_furred_dogs + brown_dogs - both_long_furred_and_brown) → 
     neither_long_furred_nor_brown = 8) :=
by
  intros total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown
  sorry

end dogs_neither_long_furred_nor_brown_l452_452267


namespace range_of_x_l452_452624

theorem range_of_x (a : ℝ) (x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + ax > 4x + a - 3) ↔ (x > 3 ∨ x < -1) :=
sorry

end range_of_x_l452_452624


namespace common_element_in_all_subsets_l452_452751

open Set

variable {α : Type*}

-- Define the set K
def K : Set (Fin 5) := {0, 1, 2, 3, 4}

-- Given conditions:
-- F is a collection of 16 different subsets of K
variable (F : Set (Set (Fin 5)))
-- Condition: F has 16 elements
variable (hF_card : F.card = 16)
-- Condition: Any three members of F have at least one element in common
variable (h_common : ∀ (A B C : Set (Fin 5)), A ∈ F → B ∈ F → C ∈ F → (A ∩ B ∩ C).Nonempty)

-- Goal: Prove that there exists exactly one element common to all subsets of F
theorem common_element_in_all_subsets :
  ∃ x : Fin 5, ∀ A ∈ F, x ∈ A := by
  sorry

end common_element_in_all_subsets_l452_452751


namespace find_x_l452_452258

theorem find_x (x : ℝ) :
  (∃ m : ℝ, m = (3 - 5) / (x - 1) ∧ m = (-3 - 5) / (5 - 1)) → x = 2 :=
by
  intro h
  have h_slope : (-2 = (3 - 5) / (x - 1)) := h.elim (λ m hm, eq.symm hm.1.trans hm.2 ▸ rfl)
  sorry

end find_x_l452_452258


namespace calculate_total_cookies_l452_452519

theorem calculate_total_cookies (cookies_per_person : ℝ) (people : ℝ) (total_cookies : ℝ) : 
  cookies_per_person = 24.0 → 
  people = 6.0 → 
  total_cookies = cookies_per_person * people → 
  total_cookies = 144.0 := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3 
  exact h3

end calculate_total_cookies_l452_452519


namespace functional_transformation_l452_452225

theorem functional_transformation (f : ℝ → ℝ)
  (h : ∀ (x : ℝ), f (2 * (x - π / 3)) = sin x) :
  f x = sin (2 * x + 2 * π / 3) :=
sorry

end functional_transformation_l452_452225


namespace taxi_fare_distance_l452_452416

variable (x : ℝ)

theorem taxi_fare_distance (h1 : 0 ≤ x - 2) (h2 : 3 + 1.2 * (x - 2) = 9) : x = 7 := by
  sorry

end taxi_fare_distance_l452_452416


namespace part1_part2_l452_452758

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0 }

noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0 }

-- Part (1): Prove a = 1 given A ∪ B = B
theorem part1 (a : ℝ) (h : A ∪ B a = B a) : a = 1 :=
sorry

-- Part (2): Prove the set C composed of the values of a given A ∩ B = B
def C : Set ℝ := {a | a ≤ -1 ∨ a = 1}

theorem part2 (h : ∀ a, A ∩ B a = B a ↔ a ∈ C) : forall a, A ∩ B a = B a ↔ a ∈ C :=
sorry

end part1_part2_l452_452758


namespace function_value_l452_452609

theorem function_value (f : ℝ → ℝ) (h : ∀ x, x + 17 = 60 * f x) : f 3 = 1 / 3 :=
by
  sorry

end function_value_l452_452609


namespace rightmost_box_balls_l452_452009

theorem rightmost_box_balls :
  ∃ red_10 blue_10 : ℕ, 
  red_10 = 0 ∧ blue_10 = 4 ∧
  (∀ (boxes : Fin 10 → ℕ × ℕ),
     (∑ i, (boxes i).1) = 10 ∧
     (∑ i, (boxes i).2) = 14 ∧
     (∀ i j, i < j → (boxes i).1 + (boxes i).2 ≤ (boxes j).1 + (boxes j).2) ∧
     (∀ i j, i ≠ j → (boxes i) ≠ (boxes j)) →
     boxes ⟨9, Nat.lt_succ_self 9⟩ = (red_10, blue_10)) :=
by
  sorry

end rightmost_box_balls_l452_452009


namespace greatest_integer_less_than_PS_l452_452285

noncomputable def rectangle_problem (PQ PS : ℝ) (T : ℝ) (PT QP : ℝ) : ℝ := real.sqrt (PQ * PQ + PS * PS) / 2

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ) (hPQ : PQ = 150)
  (T_midpoint : T = PS / 2)
  (PT_perpendicular_QT : PT * PT + T * T = PQ * PQ),
  int.floor (PS) = 212 :=
by
  intros PQ PS hPQ T_midpoint PT_perpendicular_QT
  have h₁ : PS = rectangle_problem PQ PS T PQ,
  {
    sorry
  }
  have h₂ : 150 * real.sqrt 2,
  {
    sorry
  }
  have h₃ : (⌊150 * real.sqrt 2⌋ : ℤ) = 212,
  {
    sorry
  }
  exact h₃

end greatest_integer_less_than_PS_l452_452285


namespace simplify_trig1_simplify_trig2_l452_452530

-- Problem 1
theorem simplify_trig1 : 
  (sqrt (1 - 2 * sin (130 * (Real.pi / 180)) * cos (130 * (Real.pi / 180))) / 
   (sin (130 * (Real.pi / 180)) + sqrt (1 - sin (130 * (Real.pi / 180)) ^ 2))) = 1 := 
by
  sorry

-- Problem 2
theorem simplify_trig2 (α : ℝ) (h : Real.pi / 2 < α ∧ α < Real.pi) : 
  (cos α * sqrt ((1 - sin α) / (1 + sin α)) + 
   sin α * sqrt ((1 - cos α) / (1 + cos α))) = (sin α - cos α) := 
by
  sorry

end simplify_trig1_simplify_trig2_l452_452530


namespace temperature_conversion_l452_452525

-- Define the boiling point and melting point equivalences
def boiling_point_C := 100 -- °C
def boiling_point_F := 212 -- °F
def melting_point_C := 0 -- °C
def melting_point_F := 32 -- °F

-- Given temperature of the pot in °C
def temp_pot_C := 45 -- °C

-- Formula to convert °C to °F
def celsius_to_fahrenheit (c : ℝ) : ℝ := (c * 9 / 5) + 32

-- The theorem to prove
theorem temperature_conversion :
  celsius_to_fahrenheit temp_pot_C = 113 := by
  sorry

end temperature_conversion_l452_452525


namespace hyperbola_eccentricity_l452_452866

noncomputable def eccentricity_of_hyperbola 
  (a b c : ℝ)
  (h_eq : a^2 + b^2 = c^2)
  (h_perp : (a / 3) = c)
  : ℝ := c / a

theorem hyperbola_eccentricity (a b c : ℝ) (h_eq : a^2 + b^2 = c^2) (h_perp : (c / 3) = a) :
  eccentricity_of_hyperbola a b c h_eq h_perp = 3 :=
by
  sorry

end hyperbola_eccentricity_l452_452866


namespace compute_binomial_factorial_l452_452923

def binomial (n k : ℕ) : ℕ := n.choose k

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * factorial n

theorem compute_binomial_factorial : binomial 10 4 * factorial 6 = 151200 := by
  sorry

end compute_binomial_factorial_l452_452923


namespace repeating_decimal_sum_l452_452926

theorem repeating_decimal_sum : (0.66666666 : ℝ) + (0.33333333 : ℝ) = 1 := 
by 
-- Define x and y as repeating decimals
let x := (0.66666666 : ℝ)
let y := (0.33333333 : ℝ)
-- Apply the definition of repeating decimals
have hx : x = 2 / 3 := sorry
have hy : y = 1 / 3 := sorry
-- Sum the fractions and show they add to 1
have hsum : x + y = 1 := by {
  rw [hx, hy],
  norm_num,
}
exact hsum

end repeating_decimal_sum_l452_452926


namespace toms_total_score_is_correct_l452_452452

/-- Tom's game score calculation conditions -/
def game_conditions (points_per_enemy : ℕ) (enemy_count : ℕ) (bonus_percent : ℝ) : ℝ :=
  let initial_score := points_per_enemy * enemy_count
      bonus := if enemy_count ≥ 100 then bonus_percent * initial_score else 0
  in initial_score + bonus

/-- Proof that Tom's total score is 2250 points given the conditions. -/
theorem toms_total_score_is_correct : 
  game_conditions 10 150 0.5 = 2250 :=
by
  sorry

end toms_total_score_is_correct_l452_452452


namespace inclination_of_line_through_vertex_equilateral_triangle_l452_452883

noncomputable def inclination_angles (x : ℝ) : ℝ × ℝ :=
  let α := Real.arctan (Real.sqrt 3 / 5)
  in (α, 60 - α)

theorem inclination_of_line_through_vertex_equilateral_triangle (x : ℝ) (α β : ℝ)
  (h_divides_base : 2 * x = AD ∧ x = CD ∧ AC = AD + 2 * CD)
  (h_angles : α = Real.arctan (Real.sqrt 3 / 5) ∧ β = 60 - α) :
  inclination_angles x = (α, β) :=
by
  sorry

end inclination_of_line_through_vertex_equilateral_triangle_l452_452883


namespace consecutive_sum_divisible_by_12_l452_452251

theorem consecutive_sum_divisible_by_12 
  (b : ℤ) 
  (a : ℤ := b - 1) 
  (c : ℤ := b + 1) 
  (d : ℤ := b + 2) :
  ∃ k : ℤ, ab + ac + ad + bc + bd + cd + 1 = 12 * k := by
  sorry

end consecutive_sum_divisible_by_12_l452_452251


namespace point_F_lies_on_line_DE_l452_452798

-- Problem statement
theorem point_F_lies_on_line_DE 
  (ABC : Triangle) (O : Point) (D E K L F : Point)
  (is_circumcenter : circumcenter ABC O)
  (midpoint_D : midpoint D (line_segment ABC.A ABC.B))
  (midpoint_E : midpoint E (line_segment ABC.A ABC.C))
  (OE_intersects_BC_at_K : ∃ K, intersect (line_segment O E) (line_segment ABC.B ABC.C) = intersect_at K)
  (circumcircle_OKB_intersects_OD_at_L : ∃ L, second_intersection (circumcircle (triangle OKB)) (line_segment O D) = intersect_at L)
  (F_is_foot_of_altitude : foot_of_altitude F ABC.A KL)
  (AB_lt_BC : length (line_segment ABC.A ABC.B) < length (line_segment ABC.B ABC.C)) :
  collinear [D, E, F] := sorry

end point_F_lies_on_line_DE_l452_452798


namespace probability_two_red_cards_l452_452899

theorem probability_two_red_cards 
  (num_suits : ℕ) (cards_per_suit : ℕ) (num_red_suits : ℕ)
  (deck_size := num_suits * cards_per_suit) 
  (num_red_cards := num_red_suits * cards_per_suit) 
  (num_black_suits := num_suits - num_red_suits) :
  num_suits = 5 →
  cards_per_suit = 13 →
  num_red_suits = 3 →
  num_black_suits = 2 →
  deck_size = 65 →
  num_red_cards = 39 →
  (Nat.choose deck_size 2) = 2080 →
  (Nat.choose num_red_cards 2) = 741 →
  ((Nat.choose num_red_cards 2) : ℚ / (Nat.choose deck_size 2) : ℚ) = 741 / 2080 :=
by 
  intros _ _ _ _ _ _ _ _ 
  sorry

end probability_two_red_cards_l452_452899


namespace max_triangular_faces_of_pentahedron_l452_452024

theorem max_triangular_faces_of_pentahedron (P : Polyhedron) (hP : P.faces = 5) : ∃ t ≤ 4, is_triangular_faces P t := sorry

end max_triangular_faces_of_pentahedron_l452_452024


namespace vertex_of_parabola_l452_452433

theorem vertex_of_parabola (c d : ℝ) :
  (∀ x, -2 * x^2 + c * x + d ≤ 0 ↔ x ≥ -7 / 2) →
  ∃ k, k = (-7 / 2 : ℝ) ∧ y = -2 * (x + 7 / 2)^2 + 0 := 
sorry

end vertex_of_parabola_l452_452433


namespace parabola_distance_l452_452212

theorem parabola_distance (x y : ℝ) (h_parabola : y^2 = 8 * x)
  (h_distance_focus : ∀ x y, (x - 2)^2 + y^2 = 6^2) :
  abs x = 4 :=
by sorry

end parabola_distance_l452_452212


namespace gcd_of_256_180_600_l452_452842

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l452_452842


namespace find_divisor_l452_452522

theorem find_divisor (D Q R : ℤ) (hD : D = 217) (hQ : Q = 54) (hR : R = 1) : ∃ Div : ℤ, D = (Div * Q) + R ∧ Div = 4 :=
by
  -- Define the divisor
  let Div := 4
  -- Check the division property
  have h1 : D = (Div * Q) + R,
  {
    -- Place holder for the proof
    sorry
  }
  -- State that Divisor should be 4
  use Div
  split
  exact h1
  -- Prove Div = 4
  refl

end find_divisor_l452_452522


namespace geometric_sequence_problem_l452_452209

-- Assume {a_n} is a geometric sequence with positive terms
variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Condition: all terms are positive numbers in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 0 * r ^ n

-- Condition: a_1 * a_9 = 16
def condition1 (a : ℕ → ℝ) : Prop :=
  a 1 * a 9 = 16

-- Question to prove: a_2 * a_5 * a_8 = 64
theorem geometric_sequence_problem
  (h_geom : is_geometric_sequence a r)
  (h_pos : ∀ n, 0 < a n)
  (h_cond1 : condition1 a) :
  a 2 * a 5 * a 8 = 64 :=
by
  sorry

end geometric_sequence_problem_l452_452209


namespace new_recipe_sugar_amount_l452_452430

-- Define the ratios and equivalence
def original_flour_to_water (f w : ℕ) : Prop := f = 11 ∧ w = 8
def original_flour_to_sugar (f s : ℕ) : Prop := f = 11 ∧ s = 1
def new_flour_to_water (f w : ℕ) : Prop := f = 22 ∧ w = 8
def new_flour_to_sugar (f s : ℕ) : Prop := f = 22 ∧ s = 1

-- Define the condition where water is 4 cups
def new_recipe_water_is_4 (w : ℕ) : Prop := w = 4

-- Formulate the final statement to prove the required amount of sugar
theorem new_recipe_sugar_amount (h1 : original_flour_to_water 11 8) (h2 : original_flour_to_sugar 11 1)
  (h3 : new_flour_to_water 22 8) (h4 : new_flour_to_sugar 22 1) (h5 : new_recipe_water_is_4 4) :
  ∃ s : ℕ, s = 0.5 :=
by sorry

end new_recipe_sugar_amount_l452_452430


namespace fraction_calculation_l452_452101

theorem fraction_calculation :
  ((1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25) := 
by 
  sorry

end fraction_calculation_l452_452101


namespace range_of_piecewise_function_l452_452000

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2 * x - x^2
  else if -2 ≤ x ∧ x ≤ 0 then x^2 + 6 * x
  else 0  -- The function is only defined on [-2, 3], so technically should match these intervals.

theorem range_of_piecewise_function : 
  set.range f = set.Icc (-8 : ℝ) 1 :=
sorry

end range_of_piecewise_function_l452_452000


namespace roots_product_of_polynomials_l452_452145

theorem roots_product_of_polynomials :
  ∃ (b c : ℤ), (∀ r : ℂ, r ^ 2 - 2 * r - 1 = 0 → r ^ 5 - b * r - c = 0) ∧ b * c = 348 :=
by 
  sorry

end roots_product_of_polynomials_l452_452145


namespace angle_C_measure_triangle_perimeter_unit_circle_l452_452703

-- Definitions for the first part of the problem
variables {A B C : ℝ}
variable {a b c : ℝ}
variables {m n : ℝ × ℝ}
variables {sin : ℝ → ℝ}

-- 1. Proving the measure of angle C
def angle_C_condition (A B C a b c : ℝ) (m n : ℝ × ℝ) : Prop :=
  m = (sin B + sin C, sin A + sin B) ∧
  n = (sin B - sin C, sin A) ∧
  m.1 * n.1 + m.2 * n.2 = 0

theorem angle_C_measure {A B C a b c : ℝ} {m n : ℝ × ℝ} (h : angle_C_condition A B C a b c m n) : 
  C = 2 * π / 3 := 
by sorry

-- Definitions for the second part of the problem
def triangle_isosceles (a b c : ℝ) : Prop :=
  a = b

def circumcircle_unit (A B C : ℝ) (a b c : ℝ) : Prop :=
  (a / sin A = 2) ∧ (b / sin B = 2) ∧ (c / sin C = 2)

-- 2. Proving the perimeter
theorem triangle_perimeter_unit_circle {A B C a b c : ℝ} (h1 : triangle_isosceles a b c) (h2 : circumcircle_unit A B C a b c) : 
  a + b + c = 2 + Real.sqrt 3 := 
by sorry

end angle_C_measure_triangle_perimeter_unit_circle_l452_452703


namespace ones_digit_of_3_pow_26_l452_452025

theorem ones_digit_of_3_pow_26 : (3 ^ 26) % 10 = 9 :=
by
  have cycle : list ℕ := [3, 9, 7, 1]
  have h : 26 % 4 = 2 := rfl
  have ones_digit : (3 ^ 26) % 10 = cycle.nth (26 % 4 - 1) := sorry
  exact sorry

end ones_digit_of_3_pow_26_l452_452025


namespace intersection_result_l452_452368

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x : ℝ | abs (x - 1) > 2 }

-- Define set B
def B : Set ℝ := { x : ℝ | -x^2 + 6 * x - 8 > 0 }

-- Define the complement of A in U
def compl_A : Set ℝ := U \ A

-- Define the intersection of compl_A and B
def inter_complA_B : Set ℝ := compl_A ∩ B

-- Prove that the intersection is equal to the given set
theorem intersection_result : inter_complA_B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_result_l452_452368


namespace probability_convex_quadrilateral_l452_452130

-- Definition of the given conditions
def eight_points_on_circle : Set ℝ × Set ℝ := {(1, 0), (0.7071, 0.7071), (0, 1), (-0.7071, 0.7071), (-1, 0), (-0.7071, -0.7071), (0, -1), (0.7071, -0.7071)}

-- Definition of the total number of chords and selecting 4 chords
def total_chords (n : ℕ) : ℕ := Nat.choose n 2
def selected_chords (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem stating that the probability of four randomly selected chords forming a convex quadrilateral, given eight points on a circle.
theorem probability_convex_quadrilateral (n : ℕ) (k m : ℕ) (hn : n = 8) (hk : k = 4) (hm : m = 4) :
  (selected_chords (total_chords n) k : ℚ) ⁻¹ * ↑(selected_chords n m) = (2 / 585 : ℚ) :=
by
  sorry

end probability_convex_quadrilateral_l452_452130


namespace question1_question2_l452_452204

def A : Set ℝ := { x | x^2 - 2 * x - 15 ≤ 0 }

def B (m : ℝ) : Set ℝ := { x | x^2 - (2 * m - 9) * x + m^2 - 9 * m ≥ 0 }

def C_ℝ (m : ℝ) : Set ℝ := { x | m - 9 < x ∧ x < m }

-- Question 1
theorem question1 (m : ℝ) :
  (A ∩ B m = Icc (-3 : ℝ) 3) → m = 12 :=
sorry

-- Question 2
theorem question2 (m : ℝ) :
  (A ⊆ C_ℝ m) → 5 < m ∧ m < 6 :=
sorry

end question1_question2_l452_452204


namespace circumcircle_tangent_to_AC_l452_452278

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def foot_of_altitude (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def tangent_at_point (C : Circle) (P : Point) : Line := sorry

variables (A B C M P : Point)
variables (h_abc_acute : acute_triangle A B C)
variables (h_m_midpoint : M = midpoint A B)
variables (h_p_altitude : P = foot_of_altitude A B C)
variables (h_cond : AC + BC = sqrt 2 * AB)

theorem circumcircle_tangent_to_AC :
  tangent_at_point (circumcircle B M P) (intersection_point (line_segment A C) (line_segment C P)) = line_segment A C :=
sorry

end circumcircle_tangent_to_AC_l452_452278


namespace john_speed_first_part_l452_452729

theorem john_speed_first_part (S : ℝ) (h1 : 2 * S + 3 * 55 = 255) : S = 45 :=
by
  sorry

end john_speed_first_part_l452_452729


namespace meal_center_adults_l452_452061

theorem meal_center_adults (cans : ℕ) (children_served : ℕ) (adults_served : ℕ) (total_children : ℕ) 
  (initial_cans : cans = 10) 
  (children_per_can : children_served = 7) 
  (adults_per_can : adults_served = 4) 
  (children_to_feed : total_children = 21) : 
  (cans - (total_children / children_served)) * adults_served = 28 := by
  have h1: 3 = total_children / children_served := by
    sorry
  have h2: 7 = cans - 3 := by
    sorry
  have h3: 28 = 7 * adults_served := by
    sorry
  have h4: adults_served = 4 := by
    sorry
  sorry

end meal_center_adults_l452_452061


namespace neither_of_form_3k_minus_1_l452_452395

theorem neither_of_form_3k_minus_1 (n : ℕ) (k : ℤ) : 
  (∃ k : ℕ, (n * (n - 1)) / 2 = 3 * k - 1) ↔ False ∧ 
  (∃ k : ℕ, n^2 = 3 * k - 1) ↔ False :=
begin
  -- This is the theorem statement only
  sorry
end

end neither_of_form_3k_minus_1_l452_452395


namespace range_of_x_for_f_le_2_l452_452760

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.log2 x

theorem range_of_x_for_f_le_2 : 
  {x : ℝ | f x ≤ 2} = {x : ℝ | 0 ≤ x} :=
by
  sorry

end range_of_x_for_f_le_2_l452_452760


namespace people_counted_l452_452090

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end people_counted_l452_452090


namespace tom_spent_correct_amount_l452_452016

-- Define the prices of the games
def batman_game_price : ℝ := 13.6
def superman_game_price : ℝ := 5.06

-- Define the total amount spent calculation
def total_spent := batman_game_price + superman_game_price

-- The main statement to prove
theorem tom_spent_correct_amount : total_spent = 18.66 := by
  -- Proof (intended)
  sorry

end tom_spent_correct_amount_l452_452016


namespace number_of_sequences_of_length_100_l452_452393

def sequence_count (n : ℕ) : ℕ :=
  3^n - 2^n

theorem number_of_sequences_of_length_100 :
  sequence_count 100 = 3^100 - 2^100 :=
by
  sorry

end number_of_sequences_of_length_100_l452_452393


namespace distance_ST_l452_452439

-- Define the points
def P : ℝ × ℝ × ℝ := (0, 3, 0)
def Q : ℝ × ℝ × ℝ := (2, 0, 0)
def R : ℝ × ℝ × ℝ := (2, 6, 6)

-- Equation of the plane: 3x + 2y - z = 6
def plane (x y z : ℝ) : ℝ := 3 * x + 2 * y - z

-- Define the cube vertices
def cube_vertices : List (ℝ × ℝ × ℝ) := 
  [(0,0,0), (0,0,6), (0,6,0), (0,6,6), 
   (6,0,0), (6,0,6), (6,6,0), (6,6,6)]

-- Intersection points S and T on specific edges based on given conditions
def S : ℝ × ℝ × ℝ := (4, 0, 6)
def T : ℝ × ℝ × ℝ := (0, 6, 6)

-- Distance formula in 3D
def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := a
  let (x2, y2, z2) := b
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Theorem statement
theorem distance_ST : distance S T = 2 * Real.sqrt 13 := by
  sorry

end distance_ST_l452_452439


namespace part1_problem_part2_problem_l452_452051

/-- Given initial conditions and price adjustment, prove the expected number of helmets sold and the monthly profit. -/
theorem part1_problem (initial_price : ℕ) (initial_sales : ℕ) 
(price_reduction : ℕ) (sales_per_reduction : ℕ) (cost_price : ℕ) : 
  initial_price = 80 → initial_sales = 200 → price_reduction = 10 → 
  sales_per_reduction = 20 → cost_price = 50 → 
  (initial_sales + price_reduction * sales_per_reduction = 400) ∧ 
  ((initial_price - price_reduction - cost_price) * 
  (initial_sales + price_reduction * sales_per_reduction) = 8000) :=
by
  intros
  sorry

/-- Given initial conditions and profit target, prove the expected selling price of helmets. -/
theorem part2_problem (initial_price : ℕ) (initial_sales : ℕ) 
(cost_price : ℕ) (profit_target : ℕ) (x : ℕ) :
  initial_price = 80 → initial_sales = 200 → cost_price = 50 → 
  profit_target = 7500 → (x = 15) → 
  (initial_price - x = 65) :=
by
  intros
  sorry

end part1_problem_part2_problem_l452_452051


namespace find_natural_k_l452_452142

def product_of_first_k_primes (k : ℕ) : ℕ :=
  (List.range k).map Nat.prime_from_nat : list ℕ |>.product

theorem find_natural_k (k : ℕ) (a n : ℕ) (h : k > 0 ∧ k = 1 → product_of_first_k_primes k - 1 = a ^ n) :
  k = 1 :=
by sorry

end find_natural_k_l452_452142


namespace probability_even_xy_sub_xy_even_l452_452459

theorem probability_even_xy_sub_xy_even :
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let evens := {2, 4, 6, 8, 10, 12}
  let total_ways := (s.card.choose 2)
  let even_ways := (evens.card.choose 2)
  even_ways.toRat / total_ways.toRat = 5 / 22 :=
by
  sorry

end probability_even_xy_sub_xy_even_l452_452459


namespace equivalence_of_expressions_l452_452573

-- Define the expression on the left-hand side
def lhs := (real.sqrt ((real.sqrt 5) ^ 5)) ^ 6

-- Define the expression on the right-hand side
noncomputable def rhs := 78125 * real.sqrt 5

-- The theorem to prove
theorem equivalence_of_expressions : lhs = rhs :=
by
  sorry

end equivalence_of_expressions_l452_452573


namespace mutually_exclusive_and_probability_conditional_probability_independence_l452_452192

variables (Ω : Type*) [ProbabilityMeasure Ω]
variable {A B : Set Ω}
variable [MeasurableSet A] [MeasurableSet B]

-- Given conditions
variable (hA : ℙ A = 0.5)
variable (hB : ℙ B = 0.2)

-- Proof goals
theorem mutually_exclusive_and_probability (h : Disjoint A B) : ℙ (A ∪ B) = 0.7 :=
by
  sorry

theorem conditional_probability_independence (h : ℙ(B ∩ A) / ℙ A = 0.2) : Independent A B :=
by
  sorry

end mutually_exclusive_and_probability_conditional_probability_independence_l452_452192


namespace red_team_score_l452_452719

theorem red_team_score (C R : ℕ) (h1 : C = 95) (h2 : C - R = 19) : R = 76 :=
by
  sorry

end red_team_score_l452_452719


namespace find_c_plus_d_l452_452695

theorem find_c_plus_d (a b c d : ℤ) (h1 : a + b = 14) (h2 : b + c = 9) (h3 : a + d = 8) : c + d = 3 := 
by
  sorry

end find_c_plus_d_l452_452695


namespace union_sets_l452_452671

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_sets : A ∪ B = {2, 3, 5, 6} := by
  sorry

end union_sets_l452_452671


namespace final_price_is_81_percent_of_original_l452_452897

-- Define the original price and markdowns
variable (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)

-- Define the conditions
def first_sale_price (original_price first_discount : ℝ) : ℝ :=
  original_price * (1 - first_discount)

def final_sale_price (first_sale_price second_discount : ℝ) : ℝ :=
  first_sale_price * (1 - second_discount)

-- The theorem to prove: final price is 81% of the original price
theorem final_price_is_81_percent_of_original :
  ∀ original_price : ℝ, first_discount = 0.1 → second_discount = 0.1 →
    final_sale_price (first_sale_price original_price first_discount) second_discount = original_price * 0.81 := 
by 
  intros original_price first_discount second_discount h1 h2
  rw [h1, h2]
  simp [first_sale_price, final_sale_price]
  sorry

end final_price_is_81_percent_of_original_l452_452897


namespace simplest_quadratic_radical_l452_452510

-- Define the radicals as given in the conditions
def optionA : ℝ := sqrt 0.5
def optionB : ℝ := sqrt 8
def optionC : ℝ := sqrt 27
def optionD (a : ℝ) : ℝ := sqrt (a^2 + 1)

-- State that option D is the simplest quadratic radical among the options
theorem simplest_quadratic_radical : 
  (∀ a : ℝ, ∃ (radical : ℝ), radical = sqrt (a^2 + 1)) ∧ -- Option D: √(a² + 1)
  ¬(∃ (radical : ℝ), radical = sqrt 0.5) ∧                 -- Option A: √0.5
  ¬(∃ (radical : ℝ), radical = sqrt 8) ∧                   -- Option B: √8
  ¬(∃ (radical : ℝ), radical = sqrt 27)                    -- Option C: √27
  := 
begin
  sorry
end

end simplest_quadratic_radical_l452_452510


namespace bucky_needs_to_save_more_l452_452096

def video_game_cost : ℕ := 60
def last_weekend_earnings : ℕ := 35
def trout_earning : ℕ := 5
def blue_gill_earning : ℕ := 4
def fish_caught : ℕ := 5
def trout_percentage : ℚ := 0.6

theorem bucky_needs_to_save_more : 
  let trout_caught := (trout_percentage * fish_caught).to_nat in
  let blue_gill_caught := fish_caught - trout_caught in
  let trout_income := trout_caught * trout_earning in
  let blue_gill_income := blue_gill_caught * blue_gill_earning in
  let sunday_earnings := trout_income + blue_gill_income in
  let total_savings := last_weekend_earnings + sunday_earnings in
  video_game_cost - total_savings = 2 :=
by 
  sorry

end bucky_needs_to_save_more_l452_452096


namespace how_many_engineers_l452_452708

theorem how_many_engineers (n : ℕ) (h₁ : 3 ≤ 8) (h₂ : 5 + 3 = 8) (h₃ : n > 0) 
  (h₄ : (Nat.choose 8 n) - (Nat.choose 5 n) = 46) : n = 3 :=
by 
  sorry

end how_many_engineers_l452_452708


namespace power_of_point_eq_f_l452_452874

-- Define the function representing the circle
def f (x y : ℝ) (a b c : ℝ) : ℝ := x^2 + y^2 + a * x + b * y + c

-- Define the power of a point (x0, y0) with respect to the circle
def power_of_point (x0 y0 a b c : ℝ) : ℝ := 
  x0^2 + y0^2 - (-a / 2)^2 - (-b / 2)^2 - c

-- Statement of the theorem
theorem power_of_point_eq_f (x0 y0 a b c : ℝ) : 
  power_of_point x0 y0 a b c = f x0 y0 a b c :=
sorry

end power_of_point_eq_f_l452_452874


namespace triangle_identity_l452_452307

variables {A B C a b c : ℝ}

-- Conditions of the problem
axiom triangle_ABC : A + B + C = π
axiom angle_A_side_a : a = 2 * A   -- Usage of sin and cosine rules are inferred, simplified here for brevity
axiom angle_B_side_b : b = 2 * B
axiom angle_C_side_c : c = 2 * C

theorem triangle_identity 
  (h1 : A + B + C = π)
  (h2 : a = 2 * A)
  (h3 : b = 2 * B)
  (h4 : c = 2 * C) :
  (a + b) * tan ((A - B) / 2) + (b + c) * tan ((B - C) / 2) + (c + a) * tan ((C - A) / 2) = 0 := 
sorry

end triangle_identity_l452_452307


namespace tournament_games_l452_452871

/-- In a tournament where each player plays every other player twice,
    if there are 6 players, the total number of games to be played is 60. -/
theorem tournament_games (n : ℕ) (h : n = 6) (h_opponents : ∀ i j, i ≠ j → 2) : 
  2 * n * (n - 1) = 60 :=
by
  rw h
  norm_num
  sorry

end tournament_games_l452_452871


namespace consecutive_odd_integers_sum_l452_452434

theorem consecutive_odd_integers_sum (n : ℤ) (h : (n - 2) + (n + 2) = 150) : n = 75 := 
by
  sorry

end consecutive_odd_integers_sum_l452_452434


namespace transformed_function_properties_l452_452658

theorem transformed_function_properties :
  (∀ x, f x = sin (2 * x + π / 6)) →
  (∀ x, y' = 3 * y ↔ y = f x → y' = 3 * sin (2 * x + π / 6)) →
  (∀ k : ℤ, f (π + k * π) = sin (2 * (π + k * π) + π / 6)) ∧
  (∀ k : ℤ, y = f (π / 6 + k * π) → y = 1 ∨ y = f (-π / 3 + k * π) → y = -1) ∧
  (∃ y, y = sin (2 * x + π / 6) → ( x = π / 6 + k * π ∨ x = -π / 3 + k * π )) :=
begin
  sorry
end

end transformed_function_properties_l452_452658


namespace cost_of_fish_proof_l452_452248

noncomputable section

variables (F : ℝ) (P : ℝ)
variables (cost_of_fish : ℝ) (peso1 peso2 : ℝ)

-- Conditions
def condition1 : Prop := peso1 = 530 ∧ peso1 = 4 * F + 2 * P
def condition2 : Prop := peso2 = 875 ∧ peso2 = 7 * F + (875 - 7 * cost_of_fish) / P
def fish_price_given : Prop := cost_of_fish = 80

-- Prove that the cost of fish equals 80 pesos
theorem cost_of_fish_proof (h1 : condition1) (h2 : condition2) (h3 : fish_price_given) : F = 80 :=
by
  sorry

end cost_of_fish_proof_l452_452248


namespace total_birds_is_1300_l452_452514

def initial_birds : ℕ := 300
def birds_doubled (b : ℕ) : ℕ := 2 * b
def birds_reduced (b : ℕ) : ℕ := b - 200
def total_birds_three_days : ℕ := initial_birds + birds_doubled initial_birds + birds_reduced (birds_doubled initial_birds)

theorem total_birds_is_1300 : total_birds_three_days = 1300 :=
by
  unfold total_birds_three_days initial_birds birds_doubled birds_reduced
  simp
  done

end total_birds_is_1300_l452_452514


namespace arthur_walked_distance_l452_452908

-- Define the conditions
def total_blocks_walking (east_blocks north_blocks south_return_blocks : ℕ) : ℕ :=
  east_blocks + north_blocks - south_return_blocks

def blocks_to_miles (blocks : ℕ) (block_length : ℚ) : ℚ :=
  blocks * block_length

-- The main theorem
theorem arthur_walked_distance :
  total_blocks_walking 8 15 5 = 18 → 
  blocks_to_miles 18 (1/4 : ℚ) = 4.5 := 
by 
  intros h
  rw [total_blocks_walking] at h
  rw [blocks_to_miles, h]
  norm_num
  sorry

end arthur_walked_distance_l452_452908


namespace grid_max_Xs_six_l452_452775

-- Define the condition for a valid placement of X's on a 4x4 grid
def valid_grid (grid : ℕ → ℕ → Prop) : Prop :=
  ∀ i j k, i < 4 ∧ j < 4 ∧ k < 4 ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (grid i k → ¬ (grid i j ∧ grid (j - i + k) k) ∧ ¬ (grid i (k - j - i + k) ∧ grid j k))

-- Maximize the number of X's placed on the grid
noncomputable def max_Xs (grid : ℕ → ℕ → Prop) : ℕ :=
  if h : valid_grid grid then
    (finset.iota 16).sum (λ x, if grid (x % 4) (x / 4) then 1 else 0)
  else 0

-- Prove that the maximum number of X's is 6
theorem grid_max_Xs_six : ∀ (grid : ℕ → ℕ → Prop),
  valid_grid grid →
  max_Xs grid = 6 :=
sorry

end grid_max_Xs_six_l452_452775


namespace ad_bc_ratio_l452_452303

theorem ad_bc_ratio (AB AD BC CD AE E : ℝ) (h_perp1 : AB ⊥ AD) (h_perp2 : AB ⊥ BC)
  (h_midpoint : 2 * E = CD) (h_ae_eq : AE = 2 * AB) (h_perp3 : AE ⊥ CD) :
  AD / BC = 8 / 7 :=
by
  -- The proof goes here
  sorry

end ad_bc_ratio_l452_452303


namespace pizza_options_l452_452889

theorem pizza_options (n : ℕ) (h : n = 8) : (∑ k in {0, 1, 2}, Nat.choose n k) = 37 := by
  rw h
  simp [Nat.choose, Finset.sum]
  sorry

end pizza_options_l452_452889


namespace percentage_uninsured_part_time_l452_452274

theorem percentage_uninsured_part_time
  (T : ℕ) (U : ℕ) (P : ℕ)
  (Pr : ℚ)
  (hT : T = 345)
  (hU : U = 104)
  (hP : P = 54)
  (hPr : Pr = 5797101449275363 / 10000000000000000) :
  ((U + P + ⌊Pr * T⌋ - T : ℚ) / U) * 100 ≈ 12.5 := 
by sorry

end percentage_uninsured_part_time_l452_452274


namespace smallest_multiple_of_9_and_6_is_18_l452_452482

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l452_452482


namespace aidan_height_end_summer_l452_452911

theorem aidan_height_end_summer (initial_height : ℝ) (percent_increase : ℝ) : (initial_height = 160) ∧ (percent_increase = 0.05) → (initial_height * (1 + percent_increase) = 168) :=
by
  intro h
  cases h with h1 h2
  rw [h1, h2]
  norm_num
  sorry

end aidan_height_end_summer_l452_452911


namespace exists_sequence_n_25_exists_sequence_n_1000_l452_452322

theorem exists_sequence_n_25 : 
  ∃ (l : List ℕ), l.perm (List.range 25) ∧ 
                  ∀ (i : ℕ), i < l.length - 1 → (l[i + 1] - l[i]).natAbs = 3 ∨ (l[i + 1] - l[i]).natAbs = 5 := 
  sorry

theorem exists_sequence_n_1000 : 
  ∃ (l : List ℕ), l.perm (List.range 1000) ∧ 
                  ∀ (i : ℕ), i < l.length - 1 → (l[i + 1] - l[i]).natAbs = 3 ∨ (l[i + 1] - l[i]).natAbs = 5 := 
  sorry

end exists_sequence_n_25_exists_sequence_n_1000_l452_452322


namespace smallest_common_multiple_of_9_and_6_l452_452487

theorem smallest_common_multiple_of_9_and_6 : 
  ∃ x : ℕ, (x > 0 ∧ x % 9 = 0 ∧ x % 6 = 0) ∧ 
           ∀ y : ℕ, (y > 0 ∧ y % 9 = 0 ∧ y % 6 = 0) → x ≤ y :=
begin
  use 18,
  split,
  { split,
    { exact nat.succ_pos 17, },
    { split,
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_right 9 6), },
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_left 9 6), } } },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    exact lcm.dvd_iff.1 (nat.dvd_of_mod_eq_zero hy3) }
end

end smallest_common_multiple_of_9_and_6_l452_452487


namespace area_CDM_l452_452017

noncomputable def AC := 8
noncomputable def BC := 15
noncomputable def AB := 17
noncomputable def M := (AC + BC) / 2
noncomputable def AD := 17
noncomputable def BD := 17

theorem area_CDM (h₁ : AC = 8)
                 (h₂ : BC = 15)
                 (h₃ : AB = 17)
                 (h₄ : AD = 17)
                 (h₅ : BD = 17)
                 : ∃ (m n p : ℕ),
                   m = 121 ∧
                   n = 867 ∧
                   p = 136 ∧
                   m + n + p = 1124 ∧
                   ∃ (area_CDM : ℚ), 
                   area_CDM = (121 * Real.sqrt 867) / 136 :=
by
  sorry

end area_CDM_l452_452017


namespace tan_B_of_arithmetic_seq_l452_452306

variable {α : Type*}

theorem tan_B_of_arithmetic_seq
  (a b c : ℝ)
  (h1 : a^2 + c^2 = 2 * b^2)
  (h2 : 1/2 * a * c * Real.sin B = b^2 / 3)
  (A B C : α) 
  (ha : a = ∥A∥) 
  (hb : b = ∥B∥)
  (hc : c = ∥C∥) :
  Real.tan B = 4/3 := by
  sorry

end tan_B_of_arithmetic_seq_l452_452306


namespace sum_of_simplest_form_75_200_l452_452854

theorem sum_of_simplest_form_75_200 :
  let num := 75
  let den := 200
  let simplest_num := 3
  let simplest_den := 8
  (num / gcd num den = simplest_num) ∧ (den / gcd num den = simplest_den) →
  simplest_num + simplest_den = 11 :=
by {
  sorry,
}

end sum_of_simplest_form_75_200_l452_452854


namespace highway_speed_correct_l452_452050

-- Definitions based on the conditions
def distance_local : ℝ := 60
def speed_local : ℝ := 30
def distance_highway : ℝ := 65
def total_distance : ℝ := distance_local + distance_highway
def average_speed : ℝ := 41.67

-- Required to create a noncomputable context due to division of real numbers.
noncomputable def total_time : ℝ := total_distance / average_speed
def time_local : ℝ := distance_local / speed_local
noncomputable def time_highway : ℝ := total_time - time_local
noncomputable def speed_highway : ℝ := distance_highway / time_highway

-- The theorem to prove
theorem highway_speed_correct : speed_highway = 65 := by
  sorry

end highway_speed_correct_l452_452050


namespace movies_in_series_l452_452819

theorem movies_in_series :
  -- conditions 
  let number_books := 10
  let books_read := 14
  let book_read_vs_movies_extra := 5
  (∀ number_movies : ℕ, 
  (books_read = number_movies + book_read_vs_movies_extra) →
  -- question
  number_movies = 9) := sorry

end movies_in_series_l452_452819


namespace area_multiplier_l452_452998

-- Definitions based on conditions
variables (θ : ℝ) (a b : ℝ)
def original_area (a b : ℝ) (θ : ℝ) : ℝ := 0.5 * a * b * Math.sin θ
def new_area (a b : ℝ) (θ : ℝ) : ℝ := 0.5 * (3 * a) * (3 * b) * Math.sin θ

-- Theorem statement
theorem area_multiplier (θ : ℝ) (a b : ℝ) : new_area a b θ = 9 * original_area a b θ :=
sorry

end area_multiplier_l452_452998


namespace solve_for_x_l452_452163

theorem solve_for_x (x : ℝ) : (sqrt (x^2 + 16) = 12) ↔ (x = 8 * sqrt 2) ∨ (x = -8 * sqrt 2) :=
by
  sorry

end solve_for_x_l452_452163


namespace min_sum_MB_MC_l452_452711

theorem min_sum_MB_MC
  (A B C M : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
  (angle_BAC : Angle B A C)
  (inside_M : ∃ (p : Point), p ∈ interior angle_BAC)
  (equal_segments : dist A B = dist A C) :
  ∃ (B C : Point), is_on_side B angle_BAC ∧ is_on_side C angle_BAC ∧ dist A B = dist A C ∧ (∀ (B' C' : Point), is_on_side B' angle_BAC ∧ is_on_side C' angle_BAC → dist M B' + dist M C' ≥ dist M B + dist M C) :=
sorry

end min_sum_MB_MC_l452_452711


namespace marble_sculpture_l452_452900

theorem marble_sculpture (w1 w2 w3 w_final : ℝ) (p1 p3 : ℝ):
  w1 = 300 →
  p1 = 0.30 →
  p3 = 0.15 →
  w_final = 124.95 →
  w2 = (1 - p1) * w1 →
  w3 = (1 - p3) * w2 →
  w_final = w3 →
  ∃ (p2 : ℝ),
    w1 = 300 ∧
    p2 = 0.30 ∧   -- This is the input we are solving for
    p2 / 100 * w2 = w2 * (1 - p2 / 100)
:= begin
  sorry
end

end marble_sculpture_l452_452900


namespace cone_csa_approx_l452_452814

theorem cone_csa_approx (l r : ℝ) (hc_l : l = 10) (hc_r : r = 5) : 
  let π_approx := 3.14159 in
  abs (π * r * l - 157.08) < 0.01 := 
by 
  sorry

end cone_csa_approx_l452_452814


namespace f_monotonically_decreasing_intervals_g_range_on_interval_l452_452675

variable {x : ℝ}

def a : ℝ × ℝ := (Real.sin x, -2 * Real.cos x)
def b : ℝ × ℝ := (2 * Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + 1

theorem f_monotonically_decreasing_intervals :
  ∀ k : ℤ, ∀ x, x ∈ Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6) →
  ∃ε > 0, (∀ y, ∀ h : abs (y - x) < ε, f y < f x) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x + Real.pi / 6)

theorem g_range_on_interval : 
  Set.range (λ x : ℝ, g x) = Set.Icc (-1 : ℝ) (2 : ℝ) :=
sorry

end f_monotonically_decreasing_intervals_g_range_on_interval_l452_452675


namespace gcd_and_sum_of_1729_and_867_l452_452469

-- Given numbers
def a := 1729
def b := 867

-- Define the problem statement
theorem gcd_and_sum_of_1729_and_867 : Nat.gcd a b = 1 ∧ a + b = 2596 := by
  sorry

end gcd_and_sum_of_1729_and_867_l452_452469


namespace common_elements_count_l452_452351

open Nat Set

def U : Set ℕ := {n | ∃ k ≤ 3000, n = 5 * k}
def V : Set ℕ := {n | ∃ k ≤ 3000, n = 7 * k}

theorem common_elements_count : (U ∩ V).card = 428 := sorry

end common_elements_count_l452_452351


namespace euler_line_of_antipedal_triangle_l452_452346

namespace EulerLineProblem

variables {α : Type*} [linear_ordered_field α]

/-- Given a triangle ΔABC with incenter I, the antigonal point T of I w.r.t ΔABC, and the antipedal triangle ΔXYZ of T w.r.t ΔABC, 
    prove that TI is the Euler line of ΔXYZ. --/
theorem euler_line_of_antipedal_triangle (ΔABC : triangle α)
  (I : point α) (T : point α) (XYZ : antipedal_triangle α) 
  (h_incenter : is_incenter ΔABC I) (h_antigonal : antigonal_point ΔABC I = T)
  (h_antipedal_triangle : antipedal_triangle_of T w.r.t ΔABC = XYZ)
  : is_euler_line_of TI XYZ :=
sorry

end EulerLineProblem

end euler_line_of_antipedal_triangle_l452_452346


namespace juwella_reads_pages_l452_452135

theorem juwella_reads_pages :
  let pages_three_nights_ago := 15 in
  let pages_two_nights_ago := 2 * pages_three_nights_ago in
  let pages_last_night := pages_two_nights_ago + 5 in
  let total_pages := 100 in
  let pages_read_so_far := pages_three_nights_ago + pages_two_nights_ago + pages_last_night in
  let pages_remaining := total_pages - pages_read_so_far in
  pages_remaining = 20 :=
by
  sorry

end juwella_reads_pages_l452_452135


namespace probability_heads_even_l452_452518

/-- The probability that the total number of heads is even
    when Coin A with probability of heads 3/4 is tossed 40 times
    and Coin B with probability of heads 1/2 is tossed 10 times -/
theorem probability_heads_even :
  let p_A := 3 / 4 in
  let p_B := 1 / 2 in
  let P_n (p : ℝ) (n : ℕ) : ℝ := 
    (1 / 2) * (1 + (1 - 2 * p) / (1 - p)) ^ n in
  let P_40_A := P_n p_A 40 in
  let P_10_B := P_n p_B 10 in
  let P_total := P_40_A * P_10_B + (1 - P_40_A) * (1 - P_10_B) in
  P_total = 1 / 2 :=
by
  sorry

end probability_heads_even_l452_452518


namespace area_is_2_ln_2_minus_1_over_2_l452_452794

noncomputable def area_of_closed_figure : ℝ :=
  ∫ x in 1..2, (2 / x - x + 1)

theorem area_is_2_ln_2_minus_1_over_2 : area_of_closed_figure = 2 * Real.log 2 - 1 / 2 :=
by
  sorry

end area_is_2_ln_2_minus_1_over_2_l452_452794


namespace solve_inequality_l452_452004

theorem solve_inequality (x : ℝ) : (2^(x - 2) ≤ 2^(-1)) ↔ (x ≤ 1) :=
by 
  sorry

end solve_inequality_l452_452004


namespace table_move_within_room_l452_452894

theorem table_move_within_room (L W : ℝ) (hL : L ≥ W) :
  let d := real.sqrt (9^2 + 12^2) in W ≥ d ↔ W ≥ 15 :=
by
  let d := real.sqrt (9^2 + 12^2)
  have h : d = 15 := by sorry
  rw h
  exact Iff.rfl

end table_move_within_room_l452_452894


namespace theta_in_second_quadrant_l452_452991

theorem theta_in_second_quadrant
  (θ : ℝ)
  (h1 : Real.sin θ > 0)
  (h2 : Real.tan θ < 0) :
  (π / 2 < θ) ∧ (θ < π) :=
by
  sorry

end theta_in_second_quadrant_l452_452991


namespace definite_integral_value_l452_452038

theorem definite_integral_value :
  ∫ x in 0..2 * Real.arctan (1 / 2), (1 + Real.sin x) / (1 - Real.sin x) ^ 2 = 26 / 3 :=
by sorry

end definite_integral_value_l452_452038


namespace bills_head_circumference_l452_452330

/-- Jack is ordering custom baseball caps for him and his two best friends, and we need to prove the circumference of Bill's head. -/
theorem bills_head_circumference (Jack : ℝ) (Charlie : ℝ) (Bill : ℝ)
  (h1 : Jack = 12)
  (h2 : Charlie = (1 / 2) * Jack + 9)
  (h3 : Bill = (2 / 3) * Charlie) :
  Bill = 10 :=
by sorry

end bills_head_circumference_l452_452330


namespace unique_intersection_point_l452_452344

def f (x : ℝ) := x^3 + 6 * x^2 + 28 * x + 24

theorem unique_intersection_point : ∃! (a b : ℝ), f a = a ∧ b = a ∧ f b = b :=
by { use -3, split, { simp [f], ring }, split, refl, simp [f], ring, sorry }

end unique_intersection_point_l452_452344


namespace yellow_balls_count_l452_452537

-- Define the conditions
variable (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ)
variable (prob_neither_red_purple : ℚ)

-- Set the given conditions
def conditions : Prop :=
  total_balls = 60 ∧
  white_balls = 22 ∧
  green_balls = 18 ∧
  red_balls = 15 ∧
  purple_balls = 3 ∧
  prob_neither_red_purple = 0.7

-- State the theorem
theorem yellow_balls_count (h : conditions total_balls white_balls green_balls yellow_balls red_balls purple_balls prob_neither_red_purple) : 
  yellow_balls = 2 := 
sorry

end yellow_balls_count_l452_452537


namespace smallest_n_l452_452618

def gcd(a b c d : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) (Nat.gcd c d)
def lcm(a b c d : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

theorem smallest_n :
  (∃ a b c d : ℕ, gcd a b c d = 65 ∧ lcm a b c d = 4459000 ∧ -- counts the ordered quadruplets, this is conceptual here.
  sorry : ∃ (a b c d : tuple), length_of_ordered_quadruples = 50000 ) :=
begin
  sorry,
end

end smallest_n_l452_452618


namespace inequality_min_value_l452_452806

theorem inequality_min_value (a : ℝ) : 
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ a) → (a ≤ 3) := 
by
  sorry

end inequality_min_value_l452_452806


namespace log_equation_satisfied_l452_452972

noncomputable def find_positive_x : ℝ :=
  let x := Real.sqrt 7 in
  x

theorem log_equation_satisfied (x : ℝ) (h1 : x = Real.sqrt 7) :
  Real.log x - 2 + Real.log (x ^ 2 - 2) / Real.log (Real.sqrt 5) + Real.log (x - 2) / Real.log 5⁻¹ = 2 :=
by {
  rw [h1, Real.sqrt_eq_rpow, Real.sqrt_sq],
  have hl1 : Real.log 7 = 2 * Real.log 5, from sorry,
  have hl2 : Real.log 5⁻¹ = -Real.log 5, from sorry,
  have hl3 : Real.log (Real.sqrt 5) = (1 / 2) * Real.log 5, from sorry,
  rw [Real.log_eq_rlim, Real.log_2_eq_rlim, Real.log_eq_rlim, hl1, hl2, hl3],
  exact sorry,
}

end log_equation_satisfied_l452_452972


namespace construct_equal_angle_l452_452829

noncomputable def point := ℝ × ℝ

structure triangle :=
(A B C : point)

variable O : point
variable A B M N K : point
variable r : ℝ

-- Conditions
def given_angle : Prop := 
  (∃ r : ℝ, r > 0 ∧ 
   let circle := {p : point | (p.1 - O.1) ^ 2 + (p.2 - O.2) ^ 2 = r^2} in
   A ∈ circle ∧ B ∈ circle)

-- Congruence of triangles by SSS 
def congruent_triangles : Prop :=
  dist M N = dist A B ∧ 
  dist N K = dist B O ∧ 
  dist K M = dist O A ∧
  dist A B = dist M N ∧
  dist B O = dist N K ∧
  dist O A = dist K M

theorem construct_equal_angle (h1 : given_angle) (h2 : congruent_triangles) : 
  ∠MNK = ∠AOB :=
by 
  sorry

end construct_equal_angle_l452_452829


namespace part1_part2_l452_452650

open Set

variable (A : Set ℝ) (B : Set ℝ) (C : ℝ → Set ℝ) (a : ℝ)

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 7}
def B := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) := {x : ℝ | x < a}
def C_R (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem part1 :
  (C_R A) ∩ B = {x : ℝ | 7 < x ∧ x < 10} := by
    sorry

theorem part2 :
  (A ∩ C(a) ≠ ∅) → a > 1 := by
    sorry

end part1_part2_l452_452650


namespace sale_in_fifth_month_l452_452055

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 : ℕ) (avg : ℕ) (months : ℕ) (total_sales : ℕ)
    (known_sales : sale1 = 6335 ∧ sale2 = 6927 ∧ sale3 = 6855 ∧ sale4 = 7230 ∧ sale6 = 5091)
    (avg_condition : avg = 6500)
    (months_condition : months = 6)
    (total_sales_condition : total_sales = avg * months) :
    total_sales - (sale1 + sale2 + sale3 + sale4 + sale6) = 6562 :=
by
  sorry

end sale_in_fifth_month_l452_452055


namespace sqrt_pow_simplification_l452_452575

theorem sqrt_pow_simplification :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * (5 ^ (3 / 4)) :=
by
  sorry

end sqrt_pow_simplification_l452_452575


namespace list_price_is_40_l452_452080

-- Let x be the list price of the item
variables (x : ℝ)

-- Definitions of selling prices
def alice_selling_price := x - 15
def bob_selling_price := x - 25
def charlie_selling_price := x - 20

-- Definitions of commissions
def alice_commission := 0.15 * alice_selling_price
def bob_commission := 0.25 * bob_selling_price
def charlie_commission := 0.20 * charlie_selling_price

-- Problem statement: they all receive the same commission
theorem list_price_is_40 
  (h1 : alice_commission = bob_commission)
  (h2 : bob_commission = charlie_commission) : 
  x = 40 := 
by sorry

end list_price_is_40_l452_452080


namespace find_arithmetic_sequence_an_find_geometric_sequence_Sn_l452_452980

-- Definition for the arithmetic sequence problem
def arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + (n - 1) * a 2 + ∑ i in finset.range n, (n - i) * a (i + 1)

-- Theorem for the arithmetic sequence
theorem find_arithmetic_sequence_an (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : S 1 = 5) (h2 : S 2 = 18)
  (h3 : arithmetic_sequence S a) : ∀ n, a n = 3 * n + 2 := 
sorry

-- Definition for the geometric sequence problem
def geometric_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + ∑ i in finset.range n, (n - i) * a (i + 1)

-- Theorem for the geometric sequence
theorem find_geometric_sequence_Sn (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : S 1 = 3) (h2 : S 2 = 15)
  (h3 : geometric_sequence S a) : ∀ n, S n = (3 ^ (n + 2) - 6 * n - 9) / 4 := 
sorry

end find_arithmetic_sequence_an_find_geometric_sequence_Sn_l452_452980


namespace rationalize_denominator_l452_452398

theorem rationalize_denominator :
  (1 / (real.sqrt 3 - 2)) = -(real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l452_452398


namespace prove_orthocenter_l452_452394

variable {V : Type} [inner_product_space ℝ V]

noncomputable def orthocenter_of_triangle (A B C O : V) : Prop :=
  (inner_product (O - A) (O - B) = 0) ∧ 
  (inner_product (O - B) (O - C) = 0) ∧ 
  (inner_product (O - C) (O - A) = 0)

theorem prove_orthocenter (A B C O : V) 
  (h₁ : inner_product (O - A) (O - B) = 0) 
  (h₂ : inner_product (O - B) (O - C) = 0) 
  (h₃ : inner_product (O - C) (O - A) = 0) : 
  orthocenter_of_triangle A B C O :=
by {
  rw [orthocenter_of_triangle],
  exact ⟨h₁, h₂, h₃⟩,
  sorry,
}

end prove_orthocenter_l452_452394


namespace consortium_freshman_psychology_percentage_l452_452910

theorem consortium_freshman_psychology_percentage (T : ℝ) :
  let P_A := 0.40 * 0.80 * 0.60 * 0.50 * T in
  let P_B := 0.35 * 0.70 * 0.50 * 0.40 * T in
  let P_C := 0.25 * 0.60 * 0.40 * 0.30 * T in
  P_A + P_B + P_C = 0.163 * T :=
by
  sorry

end consortium_freshman_psychology_percentage_l452_452910


namespace find_speed_train1_l452_452045

-- Definitions of conditions
def length_train1 : ℝ := 270 / 1000   -- Convert length from meters to kilometers
def length_train2 : ℝ := 230 / 1000   -- Convert length from meters to kilometers
def speed_train2 : ℝ := 80            -- Speed of the second train in km/h
def crossing_time : ℝ := 9 / 3600     -- Convert time from seconds to hours

-- Statement of the problem to be proven
theorem find_speed_train1 (V_1 : ℝ) :
  (V_1 + speed_train2) = (length_train1 + length_train2) / crossing_time →
  V_1 = 120 :=
begin
  sorry
end

end find_speed_train1_l452_452045


namespace Cole_drive_time_to_work_l452_452111

theorem Cole_drive_time_to_work :
  ∀ (D T_work T_home : ℝ),
    (T_work = D / 80) →
    (T_home = D / 120) →
    (T_work + T_home = 3) →
    (T_work * 60 = 108) :=
by
  intros D T_work T_home h1 h2 h3
  sorry

end Cole_drive_time_to_work_l452_452111


namespace sin_neg_three_halves_pi_l452_452608

theorem sin_neg_three_halves_pi : Real.sin (-3 * Real.pi / 2) = 1 := sorry

end sin_neg_three_halves_pi_l452_452608


namespace polynomial_gcd_condition_l452_452467

theorem polynomial_gcd_condition (P : Polynomial ℤ) : 
  (∀ a b : ℤ, (Int.gcd a b = 1) → (Int.gcd (P.eval a) (P.eval b) = 1)) →
  ∃ a : ℕ, P = Polynomial.C (Int.coe_nat a) * Polynomial.X^a ∨ P = Polynomial.C (-1) * Polynomial.X^a :=
by
  sorry

end polynomial_gcd_condition_l452_452467


namespace pieces_from_rod_l452_452237

noncomputable def number_of_pieces (rod_length_meters : ℝ) (piece_length_cm : ℚ) : ℕ :=
  let rod_length_cm := rod_length_meters * 100
  let piece_length := piece_length_cm.to_real
  (rod_length_cm / piece_length).to_nat

theorem pieces_from_rod :
  number_of_pieces 58.75 (137 + 2/3) = 14 :=
by
  unfold number_of_pieces
  sorry

end pieces_from_rod_l452_452237


namespace gcd_of_256_180_600_l452_452843

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l452_452843


namespace domain_of_f_l452_452591

noncomputable def f (x : ℝ) : ℝ := real.sqrt (5 * x ^ 2 - 17 * x - 6)

theorem domain_of_f :
  {x : ℝ | 5 * x ^ 2 - 17 * x - 6 ≥ 0} = {x : ℝ | x ≤ -2 / 5} ∪ {x : ℝ | x ≥ 3} :=
by
  sorry

end domain_of_f_l452_452591


namespace lorry_weight_l452_452884

theorem lorry_weight : 
  let empty_lorry_weight := 500
  let apples_weight := 10 * 55
  let oranges_weight := 5 * 45
  let watermelons_weight := 3 * 125
  let firewood_weight := 2 * 75
  let loaded_items_weight := apples_weight + oranges_weight + watermelons_weight + firewood_weight
  let total_weight := empty_lorry_weight + loaded_items_weight
  total_weight = 1800 :=
by 
  sorry

end lorry_weight_l452_452884


namespace water_left_after_four_hours_l452_452682

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def added_water_hour3 : ℕ := 1
def added_water_hour4 : ℕ := 3
def total_hours : ℕ := 4

theorem water_left_after_four_hours :
  initial_water - (water_loss_per_hour * total_hours) + (added_water_hour3 + added_water_hour4) = 36 := by
  sorry

end water_left_after_four_hours_l452_452682


namespace range_of_function_l452_452001

noncomputable def function_y (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem range_of_function : 
  ∃ (a b : ℝ), a = -12 ∧ b = 4 ∧ 
  (∀ y, (∃ x, -5 ≤ x ∧ x ≤ 0 ∧ y = function_y x) ↔ a ≤ y ∧ y ≤ b) :=
sorry

end range_of_function_l452_452001


namespace revolutions_per_minute_is_correct_l452_452862

-- Define the given conditions
def radius : ℝ := 100  -- radius of the wheel in cm
def speed_kmh : ℝ := 66  -- speed of the bus in km/h

-- Define additional necessary constants
def pi : ℝ := 3.1416

-- Convert the speed from km/h to cm/min
def speed_cmm : ℝ := (speed_kmh * 100000) / 60

-- Calculate the circumference of the wheel
def circumference : ℝ := 2 * pi * radius

-- Calculate revolutions per minute
def revolutions_per_minute : ℝ := speed_cmm / circumference

-- The statement to prove
theorem revolutions_per_minute_is_correct :
  revolutions_per_minute ≈ 1750.48 := by
  sorry

end revolutions_per_minute_is_correct_l452_452862


namespace clock_angle_l452_452468

theorem clock_angle (h_degrees : ∀ x : ℕ, x * 30 = x * 30)
  (minute_hand_pos : 0 = 0)
  (hour_hand_pos : 8 * 30 = 240)
  (full_circle : 360 = 360)
  (smaller_angle : 360 - 240 = 120) :
  ∃ angle : ℝ, angle = 120 :=
by
  use 120
  exact smaller_angle

end clock_angle_l452_452468


namespace sum_of_sequence_b_l452_452642

open Real

noncomputable def sequence_b_arithmetic (a : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ (n : ℕ), b n = log (a n) ∧ (b (n+1) - b n = b 1 - log (a 1))

noncomputable def given_product (a : ℕ → ℝ) : Prop :=
  (a 3) * (a 1010) = exp 4

noncomputable def sum_arithmetic_sequence (b : ℕ → ℝ) (m : ℕ) : ℝ :=
    ∑ i in (range m), b i

theorem sum_of_sequence_b (a b : ℕ → ℝ) (m : ℕ) 
  (arithmetic_sequence_b : sequence_b_arithmetic a b)
  (product_condition : given_product a)
  (sequence_length : m = 1012) :
  sum_arithmetic_sequence b m = 2024 :=
sorry
 
end sum_of_sequence_b_l452_452642


namespace solution_set_of_inequality_l452_452005

theorem solution_set_of_inequality (x : ℝ) : 
  (1/2)^(2*x^2 - 6*x + 9) ≤ (1/2)^(x^2 + 3*x + 19) ↔ x ∈ set.Iic (-1) ∪ set.Ici 10 :=
by
  sorry

end solution_set_of_inequality_l452_452005


namespace perpendicular_slope_l452_452157

theorem perpendicular_slope (a b c : ℝ) (h : 4 * a - 6 * b = c) :
  let m := - (3 / 2) in
  ∃ k : ℝ, (k = m) :=
by
  sorry

end perpendicular_slope_l452_452157


namespace largest_value_4x_plus_3y_l452_452987

/-
Given the equation x^2 + y^2 = 16x + 8y + 10, prove that the largest possible value of 4x + 3y is 32.
-/

theorem largest_value_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 16x + 8y + 10) :
  4 * x + 3 * y ≤ 32 :=
sorry

end largest_value_4x_plus_3y_l452_452987


namespace find_mystery_number_l452_452702

theorem find_mystery_number (x : ℕ) (h : x + 45 = 92) : x = 47 :=
sorry

end find_mystery_number_l452_452702


namespace time_to_travel_A_to_C_is_6_l452_452064

-- Assume the existence of a real number t representing the time taken
-- Assume constant speed r for the river current and p for the power boat relative to the river.
variables (t r p : ℝ)

-- Conditions
axiom condition1 : p > 0
axiom condition2 : r > 0
axiom condition3 : t * (1.5 * (p + r)) + (p - r) * (12 - t) = 12 * r

-- Define the time taken for the power boat to travel from A to C
def time_from_A_to_C : ℝ := t

-- The proof problem: Prove time_from_A_to_C = 6 under the given conditions
theorem time_to_travel_A_to_C_is_6 : time_from_A_to_C = 6 := by
  sorry

end time_to_travel_A_to_C_is_6_l452_452064


namespace find_x_plus_y_l452_452402

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
by
  sorry

end find_x_plus_y_l452_452402


namespace sequence_properties_l452_452641

noncomputable def a : ℕ → ℕ
| 0     := 0  -- Handling a₀ which is actually not needed in proof
| 1     := 1
| (n+1) := if (n+1) % 2 = 0 then a n / 2 + 1 else 2 * a n

theorem sequence_properties (n : ℕ) :
(n > 0) → 
(a 2 = 2) ∧ 
(a 3 = 3) ∧ 
(∀ n ≥ 1, (a (2 * n - 1) + 1) = 2^(n + 1) - 1) ∧
(∀ n : ℕ, S (2 * n + 1) = 2^(n + 3) + 2^(n + 1) - 3 * n - 7) := 
by {
    sorry
}

end sequence_properties_l452_452641


namespace smallest_common_multiple_of_9_and_6_l452_452490

theorem smallest_common_multiple_of_9_and_6 : 
  ∃ x : ℕ, (x > 0 ∧ x % 9 = 0 ∧ x % 6 = 0) ∧ 
           ∀ y : ℕ, (y > 0 ∧ y % 9 = 0 ∧ y % 6 = 0) → x ≤ y :=
begin
  use 18,
  split,
  { split,
    { exact nat.succ_pos 17, },
    { split,
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_right 9 6), },
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_left 9 6), } } },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    exact lcm.dvd_iff.1 (nat.dvd_of_mod_eq_zero hy3) }
end

end smallest_common_multiple_of_9_and_6_l452_452490


namespace total_people_counted_l452_452093

-- Definitions based on conditions
def people_second_day : ℕ := 500
def people_first_day : ℕ := 2 * people_second_day

-- Theorem statement
theorem total_people_counted : people_first_day + people_second_day = 1500 := 
by 
  sorry

end total_people_counted_l452_452093


namespace cohen_saw_1300_fish_eater_birds_l452_452513

theorem cohen_saw_1300_fish_eater_birds :
  let day1 := 300
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3 = 1300 :=
by
  sorry

end cohen_saw_1300_fish_eater_birds_l452_452513


namespace min_value_of_f_l452_452153

def f (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 0 ∧ f 2 = 0 :=
  by sorry

end min_value_of_f_l452_452153


namespace permutation_exists_25_permutation_exists_1000_l452_452323

-- Define a function that checks if a permutation satisfies the condition
def valid_permutation (perm : List ℕ) : Prop :=
  (∀ i < perm.length - 1, let diff := (perm[i] - perm[i+1]).abs in diff = 3 ∨ diff = 5)

-- Proof problem for n = 25
theorem permutation_exists_25 : 
  ∃ perm : List ℕ, perm.perm (List.range 25).map (· + 1) ∧ valid_permutation perm := 
sorry

-- Proof problem for n = 1000
theorem permutation_exists_1000 : 
  ∃ perm : List ℕ, perm.perm (List.range 1000).map (· + 1) ∧ valid_permutation perm := 
sorry

end permutation_exists_25_permutation_exists_1000_l452_452323


namespace mango_distribution_l452_452389

theorem mango_distribution (harvested_mangoes : ℕ) (sold_fraction : ℕ) (received_per_neighbor : ℕ)
  (h_harvested : harvested_mangoes = 560)
  (h_sold_fraction : sold_fraction = 2)
  (h_received_per_neighbor : received_per_neighbor = 35) :
  (harvested_mangoes / sold_fraction) = (harvested_mangoes / sold_fraction) / received_per_neighbor :=
by
  sorry

end mango_distribution_l452_452389


namespace maximum_value_F_possible_values_s_l452_452663

-- Define the function f(x)
def f (x : ℝ) : ℝ := (Real.log x) / x

-- Problem (Ⅰ)
-- Define the function F(x)
def F (x : ℝ) : ℝ := x^2 - x * f x

-- Theorem for part (Ⅰ)
theorem maximum_value_F :
  let I := Set.Icc (1 / 2) 2
  ∃ x ∈ I, ∀ y ∈ I, F y ≤ F x ∧ F x = 4 - Real.log 2 := by
  let I := Set.Icc (1/2) 2
  exists (2 : ℝ)
  intro y hy
  sorry

-- Problem (Ⅱ)
-- Define the piecewise function H(x)
def H (s : ℝ) (x : ℝ) : ℝ :=
  if x ≥ s then x / (2 * Real.exp 1) else f x

-- Theorem for part (Ⅱ)
theorem possible_values_s :
  {s : ℝ | ∀ k ∈ Set.univ, ∃ x0 : ℝ, H s x0 = k} = {Real.sqrt (Real.exp 1)} := by
  sorry

end maximum_value_F_possible_values_s_l452_452663


namespace relationship_of_y_coordinates_l452_452253

theorem relationship_of_y_coordinates :
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ),
  A = (-3, -2 / -3) →
  B = (-2, -2 / -2) →
  C = (3, -2 / 3) →
  let y1 := A.2,
      y2 := B.2,
      y3 := C.2
  in y3 < y1 ∧ y1 < y2 :=
by
  -- Definitions of points
  let A : ℝ × ℝ := (-3, -2 / (-3))
  let B : ℝ × ℝ := (-2, -2 / (-2))
  let C : ℝ × ℝ := (3, -2 / 3)
  
  -- Extract y-coordinates
  let y1 := A.2
  let y2 := B.2
  let y3 := C.2 

  -- Prove the relationship y3 < y1 ∧ y1 < y2
  have h1 : y1 = 2 / 3 := by simp [A, y1]
  have h2 : y2 = 1 := by simp [B, y2]
  have h3 : y3 = -2 / 3 := by simp [C, y3]
  
  have y3_lt_y1 : y3 < y1 := by linarith [h1, h3]
  have y1_lt_y2 : y1 < y2 := by linarith [h1, h2]
  
  exact ⟨y3_lt_y1, y1_lt_y2⟩

end relationship_of_y_coordinates_l452_452253


namespace train_speed_l452_452558

/--
Given:
1. A train is 100 meters long.
2. A bridge is 300 meters long.
3. The train crosses the bridge completely in 24 seconds.

Prove that the speed of the train is 16.67 meters per second.
-/
theorem train_speed 
  (train_length : ℕ) 
  (bridge_length : ℕ)
  (time_crossing : ℕ)
  (h1: train_length = 100)
  (h2: bridge_length = 300)
  (h3: time_crossing = 24) : 
  (train_length + bridge_length) / time_crossing = 16.67 :=
by sorry

end train_speed_l452_452558


namespace total_meals_per_week_l452_452680

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end total_meals_per_week_l452_452680


namespace parabola_vertex_point_l452_452805

theorem parabola_vertex_point (a b c : ℝ) 
    (h_vertex : ∃ a b c : ℝ, ∀ x : ℝ, y = a * x^2 + b * x + c) 
    (h_vertex_coord : ∃ (h k : ℝ), h = 3 ∧ k = -5) 
    (h_pass : ∃ (x y : ℝ), x = 0 ∧ y = -2) :
    c = -2 := by
  sorry

end parabola_vertex_point_l452_452805


namespace teachers_without_conditions_l452_452555

variable {t hbp ht both : ℕ}

theorem teachers_without_conditions (htotal hhp hht hboth : ℕ) 
  (htotal = 150) 
  (hhp = 80) 
  (hht = 60) 
  (hboth = 30) : 
  (((htotal - ((hhp - hboth) + (hht - hboth) + hboth)) / htotal.toFloat) * 100).round == 26.67 := 
sorry

end teachers_without_conditions_l452_452555


namespace correct_propositions_l452_452217

-- Given the following propositions:
def proposition1 : Prop :=
∀ x : ℝ, (3^x) = (log3 (exp3 x))

def proposition2 : Prop :=
real.is_periodic (abs ∘ real.sin) 2 * real.pi

def proposition3 : Prop :=
∃ x : real, ∀ x, (tan (2 * x + real.pi / 3)) = (tan (2 * (x + real.pi / 3)))

def proposition4 : Prop :=
∀ x : ℝ, x ∈ set.Icc (-2 * real.pi) (2 * real.pi) → x ∈ set.Icc (-real.pi / 3) (5 * real.pi / 3) ↔ 
(real.derivative (λ x, 2 * real.sin (real.pi / 3 - 1/2 * x)) x) < 0

-- Prove that the correct propositions are (1), (3), and (4):
theorem correct_propositions : 
  proposition1 ∧ proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l452_452217


namespace max_points_top_teams_l452_452275

theorem max_points_top_teams
  (teams : Fin 8 → Type)
  (games : (Fin 8 × Fin 8) → ℕ)
  (points : ℕ → ℕ)
  (win_points : ℕ := 3)
  (draw_points : ℕ := 1)
  (loss_points : ℕ := 0)
  (total_teams : ∀ i j : Fin 8, i ≠ j → games (i, j) = 3)
  (top_teams : Fin 8 → ℕ → Prop)
  (equal_points : ∀ i j, i ∈ {0, 1, 2, 3} → j ∈ {0, 1, 2, 3} → top_teams i p = top_teams j p)
  (total_points_distributed : ∑ i j, points (games (i, j)) = 189) :
  ∃ p, p = 72 := sorry

end max_points_top_teams_l452_452275


namespace stephanie_speed_l452_452789

noncomputable def distance : ℝ := 15
noncomputable def time : ℝ := 3

theorem stephanie_speed :
  distance / time = 5 := 
sorry

end stephanie_speed_l452_452789


namespace product_of_values_l452_452155

theorem product_of_values (x : ℝ) (h : |5 * x| + 7 = 47) : ∃ a b : ℝ, 
  a = 8 ∧ b = -8 ∧ x = a ∨ x = b ∧ a * b = -64 := 
by {
  have h₁ : |5 * x| = 40,
  { linarith, },
  cases abs_choice (5 * x)  with h_left h_right,
  { use [8, -8],
    split;
    { norm_num } },
  { use [8, -8],
    split;
    { norm_num } },
  split; { tauto }
}

end product_of_values_l452_452155


namespace initial_time_between_maintenance_checks_l452_452538

theorem initial_time_between_maintenance_checks (x : ℝ) (h1 : 1.20 * x = 30) : x = 25 := by
  sorry

end initial_time_between_maintenance_checks_l452_452538


namespace altitude_segment_length_l452_452827

/-- Given an acute triangle with altitudes forming segments of lengths 7, 4, 3, and y units respectively,
prove that y = 16/3. -/
theorem altitude_segment_length (AD DC BE EC : ℝ) (hAD : AD = 7) (hBE : BE = 4) (hEC : EC = 3)
(h_similar : ∀ A D E B C, AD * EC = BE * DC) : DC = 16 / 3 :=
by 
  -- from the conditions
  rw [hAD, hBE, hEC]
  -- we state the similarity equation
  have h_proportion : 3 * (4 + DC) = 28 := by 
    calc 
      3 * (4 + DC) = 3 * (4 + DC) : by sorry 
  -- solve for DC
  sorry

end altitude_segment_length_l452_452827


namespace compare_a_b_c_l452_452986

def f (x : ℝ) : ℝ := -|x|

def a : ℝ := f (Real.log (1 / Real.pi))

def b : ℝ := f (Real.logBase Real.pi (1 / Real.exp 1))

def c : ℝ := f (Real.log (1 / (Real.pi^2)))

theorem compare_a_b_c : b > a ∧ a > c :=
by
  sorry

end compare_a_b_c_l452_452986


namespace smallest_multiple_of_9_and_6_is_18_l452_452485

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l452_452485


namespace collinearity_B_K1_K2_l452_452713

-- Definitions for geometry entities
variables {A B C B1 B2 K1 K2 : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
noncomputable def line (p q : Point) := {r : Point // r ∈ (Affine.line p q)}
variable incircle : Circle
variable center_incircle : Point

-- Define points
variable I : Point -- Center of the incircle of triangle ABC
variable D : Point -- Point where incircle touches AC

-- Conditions from the problem statement
axiom scalene_triangle : ¬(A = B) ∧ ¬(B = C) ∧ ¬(C = A)
axiom bisector_intersect : on_line A C B1 ∧ on_line A C B2
axiom tangents_touch_incircle : tangent B1 incircle K1 ∧ tangent B2 incircle K2

-- The final statement
theorem collinearity_B_K1_K2 : collinear B K1 K2 := 
sorry

end collinearity_B_K1_K2_l452_452713


namespace tan_x_neg_one_half_l452_452984

theorem tan_x_neg_one_half (x : ℝ) (h₁ : sin x = (sqrt 5) / 5) (h₂ : x > π / 2 ∧ x < 3 * π / 2) :
  tan x = -1 / 2 := 
sorry

end tan_x_neg_one_half_l452_452984


namespace line_plane_positional_relationship_l452_452655

-- Definitions
variable {Point : Type} [EuclideanGeometry Point]
variable {a b : Line Point} {α : Plane Point}

-- Given Conditions
variable (h1 : a ∥ b)
variable (h2 : b ∥ α)

-- Theorem Statement
theorem line_plane_positional_relationship : a ∥ α ∨ a ⊆ α :=
by sorry

end line_plane_positional_relationship_l452_452655


namespace cost_price_per_meter_eq_l452_452521

variable (s : ℝ) (l : ℝ) (n : ℕ) (total_sp : ℝ)

def selling_price_per_meter (total_sp n : ℝ) : ℝ := total_sp / n

theorem cost_price_per_meter_eq
  (h1 : total_sp = 18000)
  (h2 : n = 400)
  (h3 : l = 5)
  (h4 : s = selling_price_per_meter total_sp n) :
  s + l = 50 := 
by
  sorry

end cost_price_per_meter_eq_l452_452521


namespace smallest_multiple_9_and_6_l452_452500

theorem smallest_multiple_9_and_6 : ∃ n : ℕ, n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m :=
by
  have h := Nat.lcm 9 6
  use h
  split
  sorry

end smallest_multiple_9_and_6_l452_452500


namespace test_score_range_l452_452709

theorem test_score_range
  (mark_score : ℕ) (least_score : ℕ) (highest_score : ℕ)
  (twice_least_score : mark_score = 2 * least_score)
  (mark_fixed : mark_score = 46)
  (highest_fixed : highest_score = 98) :
  (highest_score - least_score) = 75 :=
by
  sorry

end test_score_range_l452_452709


namespace find_y_when_x_is_8_l452_452438

theorem find_y_when_x_is_8 (x y : ℕ) (k : ℕ) (h1 : x + y = 36) (h2 : x - y = 12) (h3 : x * y = k) (h4 : k = 288) : y = 36 :=
by
  -- Given the conditions
  sorry

end find_y_when_x_is_8_l452_452438


namespace score_analysis_l452_452875

open Real

noncomputable def deviations : List ℝ := [8, -3, 12, -7, -10, -4, -8, 1, 0, 10]
def benchmark : ℝ := 85

theorem score_analysis :
  let highest_score := benchmark + List.maximum deviations
  let lowest_score := benchmark + List.minimum deviations
  let sum_deviations := List.sum deviations
  let average_deviation := sum_deviations / List.length deviations
  let average_score := benchmark + average_deviation
  highest_score = 97 ∧ lowest_score = 75 ∧ average_score = 84.9 :=
by
  sorry -- This is the placeholder for the proof

end score_analysis_l452_452875


namespace sum_coeff_a₅_val_l452_452629

variable {a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ}

def poly_eqn (x : ℂ) : ℂ :=
  (2*x + 1) * (x - 2)^6 - (a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7)

theorem sum_coeff : (poly_eqn 1 = 0) ∧ (poly_eqn 0 = 0) → a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -61 :=
by sorry

theorem a₅_val : (poly_eqn 2 = 0) → a₅ = 108 :=
by sorry

end sum_coeff_a₅_val_l452_452629


namespace gcd_of_256_180_600_l452_452832

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l452_452832


namespace midpoint_product_l452_452475

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 4) (hy1 : y1 = -3) (hx2 : x2 = -8) (hy2 : y2 = 7) :
  let midx := (x1 + x2) / 2
  let midy := (y1 + y2) / 2
  midx * midy = -4 :=
by
  sorry

end midpoint_product_l452_452475


namespace probability_even_diff_l452_452460

theorem probability_even_diff (x y : ℕ) (hx : x ≠ y) (hx_set : x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)) (hy_set : y ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)) :
  (∃ p : ℚ, p = 5 / 22 ∧ 
    (let xy_diff_even := xy - x - y mod 2 = 0 
     in (xy_diff_even --> True))) :=
sorry

end probability_even_diff_l452_452460


namespace x_days_to_finish_full_work_l452_452526

noncomputable def y_days_to_finish_work := 15
noncomputable def y_worked_days := 9
noncomputable def x_days_finish_remaining := 8

theorem x_days_to_finish_full_work : 
  let y_rate := 1 / y_days_to_finish_work in
  let y_done := y_rate * y_worked_days in
  let remaining_work := 1 - y_done in
  let x_rate := remaining_work / x_days_finish_remaining in
  let x_days := 1 / x_rate in
  x_days = 20 :=
by
  sorry

end x_days_to_finish_full_work_l452_452526


namespace count_positive_integers_x_satisfying_inequality_l452_452978

theorem count_positive_integers_x_satisfying_inequality :
  ∃ n : ℕ, n = 6 ∧ (∀ x : ℕ, (144 ≤ x^2 ∧ x^2 ≤ 289) → (x = 12 ∨ x = 13 ∨ x = 14 ∨ x = 15 ∨ x = 16 ∨ x = 17)) :=
sorry

end count_positive_integers_x_satisfying_inequality_l452_452978


namespace symmetrical_sequence_b_symmetrical_sequence_c_sum_l452_452635

-- Problem 1: Symmetrical sequence b_n
theorem symmetrical_sequence_b :
  ∃ (b : ℕ → ℤ), (∀ i, b i = b (8 - i)) ∧  -- Symmetrical condition
    b 1 = 2 ∧ b 4 = 11 ∧
    ∀ i, b (i + 1) = b 1 + i * 3 ∧ b 7 = b 1
:= by sorry

-- Definition of the symmetrical sequence b and its values
def b : ℕ → ℤ
| 1 := 2
| 2 := 5
| 3 := 8
| 4 := 11
| 5 := 8
| 6 := 5
| 7 := 2
| _ := 0 -- Default value for other terms

-- Problem 2: Sum of the symmetrical sequence c_n
theorem symmetrical_sequence_c_sum :
  ∃ (c : ℕ → ℤ), (∀ i, c i = c (50 - i)) ∧  -- Symmetrical condition
    (c 25 = 1 ∧ ∀ i, c (25 + i) = c (25) * 2^i) ∧  -- Geometric sequence from c_25 onwards
    let S := ∑ k in (finset.range 50).erase 0, c k 
    in S = 2^26 - 3  -- The sum of the sequence
:= by sorry

-- Definition of the symmetrical sequence c
def c : ℕ → ℤ :=
  λ n, if n ≤ 24 then 2^n else 2^(49 - n)

-- Sum S of the defined sequence c
noncomputable def S : ℤ := ∑ k in (finset.range 50).erase 0, c k

end symmetrical_sequence_b_symmetrical_sequence_c_sum_l452_452635


namespace trapezoid_area_l452_452075

theorem trapezoid_area
  (a b h : ℝ)
  (a = 6)
  (h = sqrt ((13^2 - 6^2) / 4))
  (trapezoid_has_inscribed_circle : true)
  (segments_on_non_parallel_side_sum_to_13 : 9 + 4 = 13) :
  1/2 * (a + b) * h = 198 := 
sorry

end trapezoid_area_l452_452075


namespace athlete_more_stable_l452_452551

theorem athlete_more_stable (var_A var_B : ℝ) 
                                (h1 : var_A = 0.024) 
                                (h2 : var_B = 0.008) 
                                (h3 : var_A > var_B) : 
  var_B < var_A :=
by
  exact h3

end athlete_more_stable_l452_452551


namespace proof_of_problem_l452_452529

noncomputable def problem : Prop :=
  (1 + Real.cos (20 * Real.pi / 180)) / (2 * Real.sin (20 * Real.pi / 180)) -
  (Real.sin (10 * Real.pi / 180) * 
  (1 / Real.tan (5 * Real.pi / 180) - Real.tan (5 * Real.pi / 180))) =
  (Real.sqrt 3) / 2

theorem proof_of_problem : problem :=
by
  sorry

end proof_of_problem_l452_452529


namespace exists_sequence_n_25_exists_sequence_n_1000_l452_452321

theorem exists_sequence_n_25 : 
  ∃ (l : List ℕ), l.perm (List.range 25) ∧ 
                  ∀ (i : ℕ), i < l.length - 1 → (l[i + 1] - l[i]).natAbs = 3 ∨ (l[i + 1] - l[i]).natAbs = 5 := 
  sorry

theorem exists_sequence_n_1000 : 
  ∃ (l : List ℕ), l.perm (List.range 1000) ∧ 
                  ∀ (i : ℕ), i < l.length - 1 → (l[i + 1] - l[i]).natAbs = 3 ∨ (l[i + 1] - l[i]).natAbs = 5 := 
  sorry

end exists_sequence_n_25_exists_sequence_n_1000_l452_452321


namespace rearrange_for_25_rearrange_for_1000_l452_452317

open Finset

noncomputable def canRearrange (n : ℕ) (s : Finset ℕ) : Prop :=
  ∃ lst : (Σ' m, Vector ℕ m), lst.2.toList.perm (s.toList) ∧
    (∀ i : Fin (lst.1 - 1), ((lst.2.nth i.succ) - (lst.2.nth i)) ∈ {3, 5})

theorem rearrange_for_25 : canRearrange 25 (range 1 26) :=
sorry

theorem rearrange_for_1000 : canRearrange 1000 (range 1 1001) :=
sorry

end rearrange_for_25_rearrange_for_1000_l452_452317


namespace johns_total_earnings_l452_452337

noncomputable def total_earnings_per_week (baskets_monday : ℕ) (baskets_thursday : ℕ) (small_crabs_per_basket : ℕ) (large_crabs_per_basket : ℕ) (price_small_crab : ℕ) (price_large_crab : ℕ) : ℕ :=
  let small_crabs := baskets_monday * small_crabs_per_basket
  let large_crabs := baskets_thursday * large_crabs_per_basket
  (small_crabs * price_small_crab) + (large_crabs * price_large_crab)

theorem johns_total_earnings :
  total_earnings_per_week 3 4 4 5 3 5 = 136 :=
by
  sorry

end johns_total_earnings_l452_452337


namespace smallest_number_of_three_integers_l452_452012

theorem smallest_number_of_three_integers 
  (a b c : ℕ) 
  (hpos1 : 0 < a) (hpos2 : 0 < b) (hpos3 : 0 < c) 
  (hmean : (a + b + c) / 3 = 24)
  (hmed : b = 23)
  (hlargest : b + 4 = c) 
  : a = 22 :=
by
  sorry

end smallest_number_of_three_integers_l452_452012


namespace f_values_sum_l452_452118

noncomputable def f : ℝ → ℝ := sorry

-- defining the properties
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- given conditions
axiom f_odd : is_odd f
axiom f_periodic : is_periodic f 2

-- statement to prove
theorem f_values_sum : f 1 + f 2 + f 3 = 0 :=
by
  sorry

end f_values_sum_l452_452118


namespace female_managers_l452_452705

theorem female_managers (E M F FM : ℕ) (hF : F = 625)
  (h1 : 2 * E = 5 * (E - F) + 5 * FM)
  (h2 : E = M + F) :
  FM = 250 :=
by
  simp * at *
  sorry

end female_managers_l452_452705


namespace largest_angle_of_scalene_triangle_l452_452455

theorem largest_angle_of_scalene_triangle (XYZ : Type) 
  (triangle_XYZ : triangle XYZ)
  (scalene_XYZ : scalene triangle_XYZ)
  (angle_Y : measure_of_angle triangle_XYZ Y = 25)
  (angle_Z : measure_of_angle triangle_XYZ Z = 100)
  (sum_of_angles : ∀ (X Y Z : angle triangle_XYZ), measure_of_angle X + measure_of_angle Y + measure_of_angle Z = 180) :
  largest_internal_angle triangle_XYZ = 100 := sorry

end largest_angle_of_scalene_triangle_l452_452455


namespace probability_even_xy_sub_xy_even_l452_452457

theorem probability_even_xy_sub_xy_even :
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let evens := {2, 4, 6, 8, 10, 12}
  let total_ways := (s.card.choose 2)
  let even_ways := (evens.card.choose 2)
  even_ways.toRat / total_ways.toRat = 5 / 22 :=
by
  sorry

end probability_even_xy_sub_xy_even_l452_452457


namespace mary_more_than_marco_l452_452381

def marco_initial : ℕ := 24
def mary_initial : ℕ := 15
def half_marco : ℕ := marco_initial / 2
def mary_after_give : ℕ := mary_initial + half_marco
def mary_after_spend : ℕ := mary_after_give - 5
def marco_final : ℕ := marco_initial - half_marco

theorem mary_more_than_marco :
  mary_after_spend - marco_final = 10 := by
  sorry

end mary_more_than_marco_l452_452381


namespace jenny_earnings_at_better_neighborhood_l452_452332

noncomputable def homes_in_A := 10
noncomputable def boxes_per_home_A := 2
noncomputable def homes_in_B := 5
noncomputable def boxes_per_home_B := 5
noncomputable def price_per_box := 2

theorem jenny_earnings_at_better_neighborhood :
  let total_boxes_A := homes_in_A * boxes_per_home_A in
  let total_boxes_B := homes_in_B * boxes_per_home_B in
  let better_choice := if total_boxes_A > total_boxes_B then total_boxes_A else total_boxes_B in
  let total_earnings := better_choice * price_per_box in
  total_earnings = 50 :=
by
  sorry

end jenny_earnings_at_better_neighborhood_l452_452332


namespace table_tennis_prices_and_cost_effectiveness_l452_452014

theorem table_tennis_prices_and_cost_effectiveness :
  (∃ (x y : ℕ), 
  2 * x + 3 * y = 75 ∧ 
  3 * x + 2 * y = 100 ∧ 
  20 * x + 30 * y * 0.9 < 20 * x + 30 * y - (20 / 2) * y + 20 * y) :=
  sorry

end table_tennis_prices_and_cost_effectiveness_l452_452014


namespace rearrange_numbers_25_rearrange_numbers_1000_l452_452311

theorem rearrange_numbers_25 (n : ℕ) (h : n = 25) : 
  ∃ (f : Fin n → Fin n), ∀ i : Fin (n - 1), (f i.succ.val - f i.val = 3 ∨ f i.succ.val - f i.val = 5 ∨ f i.val - f i.succ.val = 3 ∨ f i.val - f i.succ.val = 5) ∧ (f i).val ∈ Finset.range n := by
  sorry

theorem rearrange_numbers_1000 (n : ℕ) (h : n = 1000) : 
  ∃ (f : Fin n → Fin n), ∀ i : Fin (n - 1), (f i.succ.val - f i.val = 3 ∨ f i.succ.val - f i.val = 5 ∨ f i.val - f i.succ.val = 3 ∨ f i.val - f i.succ.val = 5) ∧ (f i).val ∈ Finset.range n := by
  sorry

end rearrange_numbers_25_rearrange_numbers_1000_l452_452311


namespace arithmetic_mean_prime_numbers_l452_452965

-- Define the list of numbers
def num_list : List ℕ := [33, 37, 39, 41, 43]

-- Define a predicate for primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Extract list of prime numbers
def prime_numbers : List ℕ := num_list.filter is_prime

-- Compute the arithmetic mean of a list of natural numbers
def arithmetic_mean (nums : List ℕ) : ℚ :=
  nums.foldr Nat.add 0 / nums.length

-- The main theorem: Proving the arithmetic mean of prime numbers in the list
theorem arithmetic_mean_prime_numbers :
  arithmetic_mean prime_numbers = 121 / 3 :=
by 
  -- Include proof steps here
  sorry

end arithmetic_mean_prime_numbers_l452_452965


namespace f_2010_l452_452178

def f (x : ℝ) : ℝ := (1 + x) / (1 - x)
def f1 (x : ℝ) : ℝ := f x
def fn (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then x
  else if n = 1 then f x
  else if n = 2 then f (f x)
  else if n = 3 then f (f (f x))
  else x -- This is a placeholder to show the structure

axiom cycle_length : ∀ x : ℝ, f (f (f (f x))) = x

theorem f_2010 (x : ℝ) : fn 2010 x = - (1 / x) :=
by
  have h : 2010 % 4 = 2 := by norm_num
  rw h
  sorry

end f_2010_l452_452178


namespace triangle_area_fraction_l452_452776

section
variables (A B C X Y Z : ℝ × ℝ)
-- Coordinates for points A, B, C, X, Y, Z
variables hA : A = (2, 0)
variables hB : B = (8, 12)
variables hC : C = (14, 0)
variables hX : X = (6, 0)
variables hY : Y = (8, 4)
variables hZ : Z = (10, 0)

theorem triangle_area_fraction :
  let base_abc := (C.1 - A.1)
  let height_abc := B.2
  let area_abc := (base_abc * height_abc) / 2
  let base_xyz := (Z.1 - X.1)
  let height_xyz := Y.2
  let area_xyz := (base_xyz * height_xyz) / 2
  (area_xyz / area_abc) = (1 / 9) :=
by
  sorry
end

end triangle_area_fraction_l452_452776


namespace Dima_grades_and_instrument_l452_452466

-- Definitions for the students and grades
inductive Student where
  | Vasya
  | Dima
  | Kolya
  | Sergey
  deriving DecidableEq

inductive Grade where
  | grade5
  | grade6
  | grade7
  | grade8
  deriving DecidableEq

inductive Instrument where
  | Saxophone
  | Keyboard
  | Drums
  | Guitar
  deriving DecidableEq

-- Initial conditions
axiom Vasya_plays_Saxophone : ∀ (i : Instrument), i = Instrument.Saxophone -> Student.Vasya = i
axiom Vasya_not_grade8 : ∀ (g : Grade), g = Grade.grade8 -> Student.Vasya ≠ g
axiom Keyboardist_in_grade6 : ∀ (s : Student), ∀ (g : Grade), ∀ (i : Instrument), i = Instrument.Keyboard -> g = Grade.grade6 -> s = i <-> s = g
axiom Dima_not_Drummer : ∀ (s : Student), ∀ (i : Instrument), s = Student.Dima -> i = Instrument.Drums -> s ≠ i
axiom Sergey_not_Keyboardist : ∀ (s : Student), ∀ (i : Instrument), s = Student.Sergey -> i = Instrument.Keyboard -> s ≠ i
axiom Sergey_not_grade5 : ∀ (s : Student), ∀ (g : Grade), s = Student.Sergey -> g = Grade.grade5 -> s ≠ g
axiom Dima_not_grade6 : ∀ (s : Student), ∀ (g : Grade), s = Student.Dima -> g = Grade.grade6 -> s ≠ g
axiom Drummer_not_grade8 : ∀ (s : Student), ∀ (g : Grade), ∀ (i : Instrument), i = Instrument.Drums -> g = Grade.grade8 -> s = i -> s ≠ g

-- Theorem to prove
theorem Dima_grades_and_instrument :
  ∀ (g : Grade), ∀ (i : Instrument), g = Grade.grade8 -> i = Instrument.Guitar -> (Student.Dima = g) ∧ (Student.Dima = i) :=
by
  sorry

end Dima_grades_and_instrument_l452_452466


namespace remainder_of_x_div_1000_l452_452743

theorem remainder_of_x_div_1000 (a x : ℝ) (ha1 : 1 < a) (hx1 : 1 < x)
  (h1 : log a (log a (log a 2 + log a 24 - 128)) = 128)
  (h2 : log a (log a x) = 256) :
  (x : ℝ) % 1000 = 896 :=
begin
  sorry
end

end remainder_of_x_div_1000_l452_452743


namespace coin_pile_problem_l452_452820

theorem coin_pile_problem (x y z : ℕ) (h1 : 2 * (x - y) = 16) (h2 : 2 * y - z = 16) (h3 : 2 * z - x + y = 16) :
  x = 22 ∧ y = 14 ∧ z = 12 :=
by
  sorry

end coin_pile_problem_l452_452820


namespace term_containing_x_cubed_is_7th_l452_452437

-- Define the function representing our binomial expansion term
def general_term (n r : ℕ) (x : Real) : Real :=
  binomial n r * (x ^ ((n - r) / 2)) * ((2 ^ r) / (x ^ ((r / 3))))

-- Stating the specific case where n = 16 and the term containing x^3 is looked for
theorem term_containing_x_cubed_is_7th :
  ∃ r : ℕ, (general_term 16 r x) = 3 ∧ r = 6 ∧ x = x :=
sorry

end term_containing_x_cubed_is_7th_l452_452437


namespace log_frac_l452_452690

theorem log_frac (x : ℝ) (h : log 36 (x - 6) = 1/2) : (1 / log x 6) = 3 / 2 :=
by
  sorry

end log_frac_l452_452690


namespace smallest_multiple_9_and_6_l452_452498

theorem smallest_multiple_9_and_6 : ∃ n : ℕ, n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m :=
by
  have h := Nat.lcm 9 6
  use h
  split
  sorry

end smallest_multiple_9_and_6_l452_452498


namespace cups_per_visit_l452_452383

theorem cups_per_visit 
  (cups_per_day : ℕ) (visits_per_day : ℕ)
  (h1 : cups_per_day = 6) (h2 : visits_per_day = 2) :
  cups_per_day / visits_per_day = 3 :=
by
  rw [h1, h2]
  norm_num

end cups_per_visit_l452_452383


namespace sequence_an_value_l452_452742

theorem sequence_an_value (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hS : ∀ n, 4 * S n = (a n - 1) * (a n + 3))
  (h_pos : ∀ n, 0 < a n)
  (n_nondec : ∀ n, a (n + 1) - a n = 2) :
  a 1005 = 2011 := 
sorry

end sequence_an_value_l452_452742


namespace max_min_distance_product_l452_452185

noncomputable def circle : set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 + 1)^2 = 2}
def line (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0

theorem max_min_distance_product :
  let center := (1 : ℝ, -1 : ℝ) in
  let radius := real.sqrt 2 in
  let d := abs ((1 : ℝ) - (-1)) / real.sqrt (1^2 + (-1)^2) in
  let max_distance := d + radius in
  let min_distance := d - radius in
  (max_distance * min_distance = 5 / 2) :=
begin
  sorry
end

end max_min_distance_product_l452_452185


namespace tom_total_score_l452_452449

theorem tom_total_score (points_per_enemy : ℕ) (enemy_count : ℕ) (minimum_enemies_for_bonus : ℕ) (bonus_rate : ℝ) 
(initial_points : ℕ) (bonus_points : ℝ) (total_points : ℝ) :
  points_per_enemy = 10 → 
  enemy_count = 150 → 
  minimum_enemies_for_bonus = 100 → 
  bonus_rate = 0.5 → 
  initial_points = points_per_enemy * enemy_count →
  bonus_points = if enemy_count ≥ minimum_enemies_for_bonus then initial_points * bonus_rate else 0 →
  total_points = initial_points + bonus_points →
  total_points = 2250 :=
by
  sorry

end tom_total_score_l452_452449


namespace nested_sigma_l452_452171

def sigma_1 {n : ℕ} (h : 0 < n) : ℕ :=
  (finset.range n).filter (λ d, n % d = 0).sum

def sigma {n : ℕ} (h : 0 < n) : ℕ :=
  sigma_1 h - n

theorem nested_sigma (h : 0 < 8) : sigma (sigma (sigma h (by decide)) (by decide)) (by decide) = 0 := 
sorry

end nested_sigma_l452_452171


namespace problem_isosceles_triangle_l452_452270

def midpoint (A B : (ℤ × ℤ)) : (ℤ × ℤ) :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def is_perpendicular (A B C : (ℤ × ℤ)) : Prop :=
  let midAB := midpoint A B in
  C.1 ≠ A.1 ∧ C.2 = midAB.2

def is_two_units_away (A C : (ℤ × ℤ)) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 4

def is_valid_point (C : (ℤ × ℤ)) : Prop :=
  C ≠ (3, 2) ∧ C ≠ (3, 4) ∧ 0 ≤ C.1 ∧ C.1 ≤ 5 ∧ 0 ≤ C.2 ∧ C.2 ≤ 5

theorem problem_isosceles_triangle :
  let A := (3,2) in
  let B := (3,4) in
  (finset.filter (λ (C : ℤ × ℤ),
    (is_perpendicular A B C ∨ is_two_units_away A C ∨ is_two_units_away B C) ∧ is_valid_point C)
    (finset.product (finset.range 6) (finset.range 6))).card = 4 := sorry

end problem_isosceles_triangle_l452_452270


namespace gcd_256_180_600_l452_452839

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l452_452839


namespace hyperbola_eccentricity_range_l452_452666

theorem hyperbola_eccentricity_range (a b c : ℝ) (h1 : b > a) (h2 : a > 0) : 
  (∃ F : ℝ × ℝ, F = (c, 0)) → 
  (∃ l : ℝ → ℝ, ∀ (x y : ℝ), l x = y → (l x - c) * l x + b^2 - a^2 > 0) → 
  (∀ (A B : ℝ × ℝ), A = (x1, y1) ∧ B = (x2, y2) → (x1 * x2 + y1 * y2 = 0)) → 
  (∃ e : ℝ, (1 + real.sqrt 5) / 2 ≤ e ∧ e < real.sqrt 3) :=
sorry

end hyperbola_eccentricity_range_l452_452666


namespace bisection_method_next_value_l452_452169

noncomputable def f (x : ℝ) : ℝ := x - 2 - Real.log x

theorem bisection_method_next_value :
  f 3 < 0 → f 4 > 0 → f 3.5 > 0 → ∃ x, x = 3.25 :=
by
  intros
  use 3.25
  rw [f]
  sorry

end bisection_method_next_value_l452_452169


namespace angles_proof_l452_452569

noncomputable def solveAngles : (ℝ × ℝ × ℝ) :=
  let A := 154.29
  let B := 25.71
  let C := 64.29
  (A, B, C)

theorem angles_proof:
  ∃ (A B C : ℝ), A = 154.29 ∧ B = 25.71 ∧ C = 64.29 ∧
  A + B = 180 ∧ A = 6 * B ∧ B + C = 90 :=
by {
  use 154.29, 25.71, 64.29,
  repeat { split },
  exact rfl,
  exact rfl,
  exact rfl,
  norm_num,
  norm_num,
  norm_num
}

end angles_proof_l452_452569


namespace find_balcony_seat_cost_l452_452903

-- Definitions based on conditions
variable (O B : ℕ) -- Number of orchestra tickets and cost of balcony ticket
def orchestra_ticket_cost : ℕ := 12
def total_tickets : ℕ := 370
def total_cost : ℕ := 3320
def tickets_difference : ℕ := 190

-- Lean statement to prove the cost of a balcony seat
theorem find_balcony_seat_cost :
  (2 * O + tickets_difference = total_tickets) ∧
  (orchestra_ticket_cost * O + B * (O + tickets_difference) = total_cost) →
  B = 8 :=
by
  sorry

end find_balcony_seat_cost_l452_452903


namespace greatest_y_l452_452411

theorem greatest_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -9) : y ≤ -2 :=
by {
  sorry
}

end greatest_y_l452_452411


namespace billy_total_problems_solved_l452_452914

theorem billy_total_problems_solved :
  ∃ (Q : ℕ), (3 * Q = 132) ∧ ((Q) + (2 * Q) + (3 * Q) = 264) :=
by
  sorry

end billy_total_problems_solved_l452_452914


namespace minimum_ugolki_l452_452848

-- Define the grid size
def grid_size : Nat := 5

-- Define an L-shaped figure ("ugolok") as three cells forming an 'L' shape
structure Cell :=
  (x : Nat)
  (y : Nat)

structure Ugolok :=
  (c1 : Cell)
  (c2 : Cell)
  (c3 : Cell)

-- Define the predicate for a valid 'L' shape within the grid
def is_valid_ugolok (u : Ugolok) : Prop :=
  (u.c1 ≠ u.c2 ∧ u.c2 ≠ u.c3 ∧ u.c1 ≠ u.c3) ∧
  (
    -- Check if cells form an 'L' shape, considering symmetry and rotation
    (u.c1.x = u.c2.x ∧ u.c1.y + 1 = u.c2.y ∧ u.c2.x + 1 = u.c3.x ∧ u.c2.y = u.c3.y) ∨
    (u.c1.y = u.c2.y ∧ u.c1.x + 1 = u.c2.x ∧ u.c2.y + 1 = u.c3.y ∧ u.c2.x = u.c3.x) ∨
    -- Additional rotations and reflections
    (u.c1.x = u.c2.x ∧ u.c1.y - 1 = u.c2.y ∧ u.c2.x + 1 = u.c3.x ∧ u.c2.y = u.c3.y) ∨
    (u.c1.y = u.c2.y ∧ u.c1.x - 1 = u.c2.x ∧ u.c2.y + 1 = u.c3.y ∧ u.c2.x = u.c3.x)
  )

-- Define a predicate to check that 'ugolok's do not overlap
def no_overlap (us : List Ugolok) : Prop :=
  ∀ (u1 u2 : Ugolok), u1 ∈ us → u2 ∈ us → u1 ≠ u2 →
  ({u1.c1, u1.c2, u1.c3} ∩ {u2.c1, u2.c2, u2.c3} = ∅)

-- Theorem statement
theorem minimum_ugolki (us : List Ugolok) 
  (h1 : ∀ u ∈ us, is_valid_ugolok u)
  (h2 : no_overlap us)
  (h3 : us.length = 4) : 
  ∀ (v : Ugolok), v ∉ us → ¬ is_valid_ugolok v :=
sorry

end minimum_ugolki_l452_452848


namespace sum_first_fifteen_excellent_numbers_l452_452907

/-
Excellent numbers are defined as either:
1. the cube of a prime number,
2. the product of two distinct primes,
3. the square of a prime multiplied by another distinct prime.

This theorem states that the sum of the first fifteen excellent natural numbers is 448.
-/
theorem sum_first_fifteen_excellent_numbers : 
  let excellent (n : ℕ) : Prop :=
    ∃ p q : ℕ, Prime p ∧ Prime q ∧ 
      ((n = p^3) ∨ (n = p * q ∧ p ≠ q) ∨ (n = p^2 * q ∧ p ≠ q)) in
  ∃ (l : List ℕ), l.length = 15 ∧ (∀ n ∈ l, excellent n) ∧ l.sum = 448 :=
by
  sorry

end sum_first_fifteen_excellent_numbers_l452_452907


namespace smallest_multiple_of_9_and_6_l452_452506

theorem smallest_multiple_of_9_and_6 : ∃ n : ℕ, (n > 0) ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) := 
begin
  use 18,
  split,
  { -- n > 0
    exact nat.succ_pos',
  },
  split,
  { -- n % 9 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 9),
  },
  split,
  { -- n % 6 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 6),
  },
  { -- ∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m
    intros m h_pos h_multiple9 h_multiple6,
    exact le_of_dvd h_pos (nat.lcm_dvd_prime_multiples 6 9),
  },
  sorry, -- Since full proof capabilities are not required here, "sorry" is used to skip the proof process.
end

end smallest_multiple_of_9_and_6_l452_452506


namespace func_odd_extrema_values_l452_452636

variable (f : ℝ → ℝ)

-- Conditions
axiom functional_eq : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom neg_pos_cond : ∀ x : ℝ, x > 0 → f(x) < 0
axiom initial_val : f(1) = -2

-- Theorem 1: Proving the function is odd
theorem func_odd : ∀ x : ℝ, f(-x) = -f(x) :=
by
  sorry

-- Theorem 2: Determining maximum and minimum values on the interval [-3, 3]
theorem extrema_values : f(-3) = 6 ∧ f(3) = -6 :=
by
  sorry

end func_odd_extrema_values_l452_452636


namespace find_q_l452_452423

theorem find_q : 
  ∃ (p q r s : ℚ), 
    (-8 * p + 4 * q - 2 * r + s = 0) ∧ 
    (s = -3) ∧ 
    (8 * p + 4 * q + 2 * r - 3 = 0) → 
    q = 3 / 4 :=
begin 
  sorry
end

end find_q_l452_452423


namespace net_salary_change_l452_452077

-- Problem: Prove that the net change in the worker's salary is 13.99% given the series of salary adjustments.

-- Definition for initial salary
def initial_salary (S : ℝ) := S

-- Conditions:
def after_first_increase (S : ℝ) := 1.20 * S
def after_first_reduction (S : ℝ) := 1.08 * S
def after_second_increase (S : ℝ) := 1.242 * S
def after_second_reduction (S : ℝ) := 1.1799 * S
def performance_bonus (S : ℝ) := 0.08 * S
def tax_deduction (S : ℝ) := 0.12 * S

-- Net salary after all adjustments
def net_salary (S : ℝ) := after_second_reduction S + performance_bonus S - tax_deduction S

-- Net change in percentage
def net_change_percentage (S : ℝ) := ((net_salary S - initial_salary S) / initial_salary S) * 100

-- Theorem: Net change in the worker's salary is 13.99%
theorem net_salary_change (S : ℝ) : net_change_percentage S = 13.99 :=
by
  sorry

end net_salary_change_l452_452077


namespace problem1_l452_452868

theorem problem1 (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * f 1 * x) : f' 1 = 1 :=
sorry

end problem1_l452_452868


namespace total_birds_is_1300_l452_452515

def initial_birds : ℕ := 300
def birds_doubled (b : ℕ) : ℕ := 2 * b
def birds_reduced (b : ℕ) : ℕ := b - 200
def total_birds_three_days : ℕ := initial_birds + birds_doubled initial_birds + birds_reduced (birds_doubled initial_birds)

theorem total_birds_is_1300 : total_birds_three_days = 1300 :=
by
  unfold total_birds_three_days initial_birds birds_doubled birds_reduced
  simp
  done

end total_birds_is_1300_l452_452515


namespace average_new_data_set_is_5_l452_452644

variable {x1 x2 x3 x4 : ℝ}
variable (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0)
variable (var_sqr : ℝ) (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16))

theorem average_new_data_set_is_5 (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16)) : 
  (x1 + 3 + x2 + 3 + x3 + 3 + x4 + 3) / 4 = 5 := 
by 
  sorry

end average_new_data_set_is_5_l452_452644


namespace prime_list_mean_l452_452956

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def prime_list_mean_proof : Prop :=
  let nums := [33, 37, 39, 41, 43] in
  let primes := filter is_prime nums in
  primes = [37, 41, 43] ∧
  (primes.sum : ℤ) / primes.length = 40.33

theorem prime_list_mean : prime_list_mean_proof :=
by
  sorry

end prime_list_mean_l452_452956


namespace hours_learning_english_each_day_l452_452939

theorem hours_learning_english_each_day (E : ℕ) 
  (h_chinese_each_day : ∀ (d : ℕ), d = 7) 
  (days : ℕ) 
  (h_total_days : days = 5) 
  (h_total_hours : ∀ (t : ℕ), t = 65) 
  (total_learning_time : 5 * (E + 7) = 65) :
  E = 6 :=
by
  sorry

end hours_learning_english_each_day_l452_452939


namespace bill_head_circumference_l452_452329

theorem bill_head_circumference (jack_head_circumference charlie_head_circumference bill_head_circumference : ℝ) :
  jack_head_circumference = 12 →
  charlie_head_circumference = (1 / 2 * jack_head_circumference) + 9 →
  bill_head_circumference = (2 / 3 * charlie_head_circumference) →
  bill_head_circumference = 10 :=
by
  intro hj hc hb
  sorry

end bill_head_circumference_l452_452329


namespace daily_reading_goal_l452_452095

-- Define the constants for pages read each day
def pages_on_sunday : ℕ := 43
def pages_on_monday : ℕ := 65
def pages_on_tuesday : ℕ := 28
def pages_on_wednesday : ℕ := 0
def pages_on_thursday : ℕ := 70
def pages_on_friday : ℕ := 56
def pages_on_saturday : ℕ := 88

-- Define the total pages read in the week
def total_pages := pages_on_sunday + pages_on_monday + pages_on_tuesday + pages_on_wednesday 
                    + pages_on_thursday + pages_on_friday + pages_on_saturday

-- The theorem that expresses Berry's daily reading goal
theorem daily_reading_goal : total_pages / 7 = 50 :=
by
  sorry

end daily_reading_goal_l452_452095


namespace number_of_managers_in_sample_l452_452561

def totalStaff : ℕ := 160
def salespeople : ℕ := 104
def managers : ℕ := 32
def logisticsPersonnel : ℕ := 24
def sampleSize : ℕ := 20

theorem number_of_managers_in_sample : 
  (managers * (sampleSize / totalStaff) = 4) := by
  sorry

end number_of_managers_in_sample_l452_452561


namespace simplify_expression_l452_452516

theorem simplify_expression : (-5) - (-4) + (-7) - (2) = -5 + 4 - 7 - 2 := 
by
  sorry

end simplify_expression_l452_452516


namespace range_of_a_l452_452256
variable (a : ℝ)

def f (x : ℝ) : ℝ :=
if x ≥ 1 then a^x else (4 - a / 2) * x + 2

theorem range_of_a (h : ∀ x y, x < y → f a x ≤ f a y) : 4 ≤ a ∧ a < 8 :=
by
  sorry

end range_of_a_l452_452256


namespace triangle_angle_B_l452_452262

noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
def rad_to_deg (θ : ℝ) : ℝ := θ * 180 / Real.pi

theorem triangle_angle_B {a b : ℝ} (A B : ℝ) :
  a = 2 * Real.sqrt 3 ∧ b = 2 ∧ rad_to_deg B = 30 ∧ rad_to_deg A = 60 →
  B = Real.arcsin (b * sin (A * Real.pi / 180) / a) :=
by
  sorry

end triangle_angle_B_l452_452262


namespace rectangle_non_covering_sequence_exists_l452_452606

theorem rectangle_non_covering_sequence_exists :
  ∃ (s : ℕ → ℝ) (l : ℕ → ℝ),
  (s 1 = 1) ∧ 
  (∀ k, k ≥ 2 → s k = s (k - 1) / 4) ∧ 
  (∀ k, k ≥ 2 → l k > 2 * ∑ i in Finset.range (k - 1), real.sqrt (2)) ∧ 
  ¬ ∃ n, s n / 2 < ∑ i in Finset.range (100 - n + 1), s (i + n) := 
sorry

end rectangle_non_covering_sequence_exists_l452_452606


namespace minimum_value_on_parabola_l452_452668

open Real

/-- The parabola C is defined by the equation y^2 = 6x -/
def parabola (x y : ℝ) : Prop :=
  y^2 = 6 * x

/-- The function to calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The focus of the parabola y^2 = 6x is at point F=(3/2, 0) -/
def focus : ℝ × ℝ := (3 / 2, 0)

/-- The fixed point M is given as (3, 1/2) -/
def M : ℝ × ℝ := (3, 1/2)

/-- The minimum value of |PM| + |PF| for all points P on parabola y^2 = 6x -/
theorem minimum_value_on_parabola (P : ℝ × ℝ) (hP : parabola P.1 P.2) :
  (distance P M + distance P focus) = 9 / 2 :=
begin
  -- proof goes here
  sorry
end

end minimum_value_on_parabola_l452_452668


namespace stratified_sampling_household_l452_452770

/-
  Given:
  - Total valid questionnaires: 500,000.
  - Number of people who purchased:
    - clothing, shoes, and hats: 198,000,
    - household goods: 94,000,
    - cosmetics: 116,000,
    - home appliances: 92,000.
  - Number of questionnaires selected from the "cosmetics" category: 116.
  
  Prove:
  - The number of questionnaires that should be selected from the "household goods" category is 94.
-/

theorem stratified_sampling_household (total_valid: ℕ)
  (clothing_shoes_hats: ℕ)
  (household_goods: ℕ)
  (cosmetics: ℕ)
  (home_appliances: ℕ)
  (sample_cosmetics: ℕ) :
  total_valid = 500000 →
  clothing_shoes_hats = 198000 →
  household_goods = 94000 →
  cosmetics = 116000 →
  home_appliances = 92000 →
  sample_cosmetics = 116 →
  (116 * household_goods = sample_cosmetics * cosmetics) →
  116 * 94000 = 116 * 116000 →
  94000 = 116000 →
  94 = 94 := by
  intros
  sorry

end stratified_sampling_household_l452_452770


namespace smallest_common_multiple_of_9_and_6_l452_452493

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l452_452493


namespace jenny_earnings_at_better_neighborhood_l452_452333

noncomputable def homes_in_A := 10
noncomputable def boxes_per_home_A := 2
noncomputable def homes_in_B := 5
noncomputable def boxes_per_home_B := 5
noncomputable def price_per_box := 2

theorem jenny_earnings_at_better_neighborhood :
  let total_boxes_A := homes_in_A * boxes_per_home_A in
  let total_boxes_B := homes_in_B * boxes_per_home_B in
  let better_choice := if total_boxes_A > total_boxes_B then total_boxes_A else total_boxes_B in
  let total_earnings := better_choice * price_per_box in
  total_earnings = 50 :=
by
  sorry

end jenny_earnings_at_better_neighborhood_l452_452333


namespace find_principal_l452_452260

theorem find_principal
  (P : ℝ)
  (R : ℝ := 4)
  (T : ℝ := 5)
  (SI : ℝ := (P * R * T) / 100) 
  (h : SI = P - 2400) : 
  P = 3000 := 
sorry

end find_principal_l452_452260


namespace smallest_product_is_zero_l452_452585

def smallest_product (s : Set ℤ) : ℤ :=
  if s.size < 2 then 0 else s.prod

theorem smallest_product_is_zero (s : Set ℤ) (h : s = {-10, -6, 0, 2, 5}) :
  smallest_product (s × s) = 0 :=
by
  sorry

end smallest_product_is_zero_l452_452585


namespace ratio_of_cakes_l452_452107

/-- Define the usual number of cheesecakes, muffins, and red velvet cakes baked in a week -/
def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet_cakes : ℕ := 8

/-- Define the total number of cakes usually baked in a week -/
def usual_cakes : ℕ := usual_cheesecakes + usual_muffins + usual_red_velvet_cakes

/-- Assume Carter baked this week a multiple of usual cakes, denoted as x -/
def multiple (x : ℕ) : Prop := usual_cakes * x = usual_cakes + 38

/-- Assume he baked usual_cakes + 38 equals 57 cakes -/
def total_cakes_this_week : ℕ := 57

/-- The theorem stating the problem: proving the ratio is 3:1 -/
theorem ratio_of_cakes (x : ℕ) (hx : multiple x) : 
  (total_cakes_this_week : ℚ) / (usual_cakes : ℚ) = (3 : ℚ) :=
by
  sorry

end ratio_of_cakes_l452_452107


namespace total_pebbles_count_l452_452390

def white_pebbles : ℕ := 20
def red_pebbles : ℕ := white_pebbles / 2
def blue_pebbles : ℕ := red_pebbles / 3
def green_pebbles : ℕ := blue_pebbles + 5

theorem total_pebbles_count : white_pebbles + red_pebbles + blue_pebbles + green_pebbles = 41 := by
  sorry

end total_pebbles_count_l452_452390


namespace minimum_words_to_learn_for_90_percent_l452_452239

-- Define the conditions
def total_vocabulary_words : ℕ := 800
def minimum_percentage_required : ℚ := 0.90

-- Define the proof goal
theorem minimum_words_to_learn_for_90_percent (x : ℕ) (h1 : (x : ℚ) / total_vocabulary_words ≥ minimum_percentage_required) : x ≥ 720 :=
sorry

end minimum_words_to_learn_for_90_percent_l452_452239


namespace sum_even_num_even_nums_even_sum_even_num_odd_nums_even_sum_odd_num_even_nums_even_sum_odd_num_odd_nums_odd_l452_452128

-- Define the even property
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define the odd property
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Prove that the sum of an even number of even numbers is even
theorem sum_even_num_even_nums_even (n : ℤ) (a : ℤ → ℤ) (h_even_n : is_even n) (h_even_a : ∀ i, is_even (a i)) : 
  is_even (∑ i in finset.range (nat_abs n).to_nat, a i) := 
sorry

-- Prove that the sum of an even number of odd numbers is even
theorem sum_even_num_odd_nums_even (n : ℤ) (b : ℤ → ℤ) (h_even_n : is_even n) (h_odd_b : ∀ i, is_odd (b i)) : 
  is_even (∑ i in finset.range (nat_abs n).to_nat, b i) := 
sorry

-- Prove that the sum of an odd number of even numbers is even
theorem sum_odd_num_even_nums_even (n : ℤ) (c : ℤ → ℤ) (h_odd_n : is_odd n) (h_even_c : ∀ i, is_even (c i)) : 
  is_even (∑ i in finset.range (nat_abs n).to_nat, c i) := 
sorry

-- Prove that the sum of an odd number of odd numbers is odd
theorem sum_odd_num_odd_nums_odd (n : ℤ) (d : ℤ → ℤ) (h_odd_n : is_odd n) (h_odd_d : ∀ i, is_odd (d i)) : 
  is_odd (∑ i in finset.range (nat_abs n).to_nat, d i) := 
sorry

end sum_even_num_even_nums_even_sum_even_num_odd_nums_even_sum_odd_num_even_nums_even_sum_odd_num_odd_nums_odd_l452_452128


namespace linear_map_scalar_identity_l452_452830

noncomputable def V (n : ℕ) [field 𝔽] := vector_space 𝔽 (fin n)

theorem linear_map_scalar_identity
  (𝔽 : Type*) [field 𝔽]
  (n : ℕ) (V : Type*) [vector_space 𝔽 V] [finite_dimensional 𝔽 V]
  (A : V →ₗ[𝔽] V)
  (eigenvecs : list V)
  (eigenvals : list 𝔽)
  (h1 : eigenvecs.length = n + 1)
  (h2 : ∀ i, i ∈ eigenvecs → ∃ λ, A (i) = λ • i)
  (h3 : ∀ (s : finset V), s ⊆ eigenvecs.to_finset → s.card = n → linear_independent 𝔽 (s : set V)) :
  ∃ λ : 𝔽, A = λ • linear_map.id :=
sorry

end linear_map_scalar_identity_l452_452830


namespace kira_night_songs_l452_452343

-- Definitions for the conditions
def morning_songs : ℕ := 10
def later_songs : ℕ := 15
def song_size_mb : ℕ := 5
def total_new_songs_memory_mb : ℕ := 140

-- Assert the number of songs Kira downloaded at night
theorem kira_night_songs : (total_new_songs_memory_mb - (morning_songs * song_size_mb + later_songs * song_size_mb)) / song_size_mb = 3 :=
by
  sorry

end kira_night_songs_l452_452343


namespace circumcircle_tangent_to_AB_l452_452181

variables {A B C H M A' : Type*} [InnerProductSpace ℝ A]
variables [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables [InnerProductSpace ℝ H] [InnerProductSpace ℝ M] [InnerProductSpace ℝ A']

-- The centroid of an acute-angled triangle ABC
def is_centroid (A B C M : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ M] :=
  -- Definition of centroid goes here, assumed available in library

-- The altitude AH of triangle ABC
def is_altitude (A B C H : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ H] :=
  -- Definition of altitude goes here, assumed available in library

-- A' is the intersection of ray MH with circumcircle
def intersects_circumcircle (A B C M H A' : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ M] [InnerProductSpace ℝ H] [InnerProductSpace ℝ A'] :=
  -- Definition intersection with circumcircle goes here, assumed available in library

-- The circumcircle of a triangle is tangent to a line
def circumcircle_tangent (A B H A' : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ H] [InnerProductSpace ℝ A'] :=
  -- Definition of circumcircle tangency goes here, assumed available in library

theorem circumcircle_tangent_to_AB
  (A B C H M A' : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ H] [InnerProductSpace ℝ M] [InnerProductSpace ℝ A']
  (h1 : AB < AC)
  (h2 : is_centroid A B C M)
  (h3 : is_altitude A B C H)
  (h4 : intersects_circumcircle A B C M H A') :
  circumcircle_tangent A B H A' :=
sorry

end circumcircle_tangent_to_AB_l452_452181


namespace simplify_expression_l452_452631

theorem simplify_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
    a * (1 / b + 1 / c) + b * (1 / a + 1 / c) + c * (1 / a + 1 / b) = -3 :=
by
  sorry

end simplify_expression_l452_452631


namespace Robie_gave_away_boxes_l452_452405

theorem Robie_gave_away_boxes :
  ∀ (total_cards cards_per_box boxes_with_him remaining_cards : ℕ)
  (h_total_cards : total_cards = 75)
  (h_cards_per_box : cards_per_box = 10)
  (h_boxes_with_him : boxes_with_him = 5)
  (h_remaining_cards : remaining_cards = 5),
  (total_cards / cards_per_box) - boxes_with_him = 2 :=
by
  intros total_cards cards_per_box boxes_with_him remaining_cards
  intros h_total_cards h_cards_per_box h_boxes_with_him h_remaining_cards
  sorry

end Robie_gave_away_boxes_l452_452405


namespace green_beans_more_than_sugar_l452_452373

def weight_of_green_beans : ℝ := 60
def total_remaining_stock : ℝ := 120
def weight_lost_fraction_rice : ℝ := 1/3
def weight_lost_fraction_sugar : ℝ := 1/5
def rice_weight_loss : ℝ := weight_of_green_beans - 30

theorem green_beans_more_than_sugar : ∃ S : ℝ,
  total_remaining_stock = 20 + (4/5 * S) + weight_of_green_beans ∧
  weight_of_green_beans - S = 10 :=
by
  sorry

end green_beans_more_than_sugar_l452_452373


namespace triangle_inequality_l452_452177

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = π) :
    (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = π ∧ 
    a = 2 * arcsin (sin (A / 2)) ∧ b = 2 * arcsin (sin (B / 2)) ∧ c = 2 * arcsin (sin (C / 2))) →
      (sqrt (sin A * sin B) / sin (C / 2) +   
      sqrt (sin B * sin C) / sin (A / 2) + 
      sqrt (sin C * sin A) / sin (B / 2)) ≥ 3 * sqrt 3 := 
by
  sorry

end triangle_inequality_l452_452177


namespace plant_marker_matching_probability_l452_452570

theorem plant_marker_matching_probability :
  let total_possible_arrangements := 4!
  let number_of_correct_arrangements := 1
  (number_of_correct_arrangements : ℚ) / (total_possible_arrangements : ℚ) = 1 / 24 :=
by
  sorry

end plant_marker_matching_probability_l452_452570


namespace right_triangles_count_l452_452781

theorem right_triangles_count :
  let xs := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
  let ys := {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
  (∃ (P Q R : ℤ × ℤ), 
    P.1 ∈ xs ∧ P.2 ∈ ys ∧ 
    R.1 ∈ xs ∧ R.1 ≠ P.1 ∧ R.2 = P.2 ∧ 
    Q.1 = P.1 ∧ Q.2 ∈ ys ∧ Q.2 ≠ P.2) → 
  (9900 = set.univ.filter (λ ⟨P, Q, R⟩, 
    P.1 ∈ xs ∧ P.2 ∈ ys ∧ 
    R.1 ∈ xs ∧ R.1 ≠ P.1 ∧ R.2 = P.2 ∧ 
    Q.1 = P.1 ∧ Q.2 ∈ ys ∧ Q.2 ≠ P.2
  ).count sorry)

end right_triangles_count_l452_452781


namespace fair_dice_can_be_six_l452_452029

def fair_dice_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem fair_dice_can_be_six : 6 ∈ fair_dice_outcomes :=
by {
  -- This formally states that 6 is a possible outcome when throwing a fair dice
  sorry
}

end fair_dice_can_be_six_l452_452029


namespace sale_price_after_discounts_l452_452882

def original_price : ℝ := 400.00
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.10

theorem sale_price_after_discounts (orig : ℝ) (d1 d2 d3 : ℝ) :
  orig = original_price →
  d1 = discount1 →
  d2 = discount2 →
  d3 = discount3 →
  orig * (1 - d1) * (1 - d2) * (1 - d3) = 243.00 := by
  sorry

end sale_price_after_discounts_l452_452882


namespace exists_sequence_n_25_exists_sequence_n_1000_l452_452320

theorem exists_sequence_n_25 : 
  ∃ (l : List ℕ), l.perm (List.range 25) ∧ 
                  ∀ (i : ℕ), i < l.length - 1 → (l[i + 1] - l[i]).natAbs = 3 ∨ (l[i + 1] - l[i]).natAbs = 5 := 
  sorry

theorem exists_sequence_n_1000 : 
  ∃ (l : List ℕ), l.perm (List.range 1000) ∧ 
                  ∀ (i : ℕ), i < l.length - 1 → (l[i + 1] - l[i]).natAbs = 3 ∨ (l[i + 1] - l[i]).natAbs = 5 := 
  sorry

end exists_sequence_n_25_exists_sequence_n_1000_l452_452320


namespace imaginary_part_of_conjugate_l452_452184

variable (z : ℂ)
variable (i : ℂ)
variable (conj_z : ℂ)

def z_val : z = 1 + 2 * Complex.i := by sorry
def conj_z_val : conj_z = Complex.conj (1 + 2 * Complex.i) := by sorry
def imaginary_part_conj_z := Complex.im conj_z

theorem imaginary_part_of_conjugate :
  imaginary_part_conj_z = -2 := by sorry

end imaginary_part_of_conjugate_l452_452184


namespace intersecting_point_is_4_l452_452424

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := -2 * x + c

noncomputable def f_inv (x : ℝ) (c : ℝ) : ℝ := (c - x) / 2

theorem intersecting_point_is_4 {c d : ℝ} 
  (h1 : f 4 c = d)
  (h2 : f_inv 4 c = d)
  (h3 : ∀ x, f_inv (f x c) c = x)
  (h4 : c ∈ ℤ)
  (h5 : d ∈ ℤ) :
  d = 4 :=
by
  sorry

end intersecting_point_is_4_l452_452424


namespace donna_received_total_interest_l452_452597

-- Donna's investment conditions
def totalInvestment : ℝ := 33000
def investmentAt4Percent : ℝ := 13000
def investmentAt225Percent : ℝ := totalInvestment - investmentAt4Percent
def rate4Percent : ℝ := 0.04
def rate225Percent : ℝ := 0.0225

-- The interest calculation
def interestFrom4PercentInvestment : ℝ := investmentAt4Percent * rate4Percent
def interestFrom225PercentInvestment : ℝ := investmentAt225Percent * rate225Percent
def totalInterest : ℝ := interestFrom4PercentInvestment + interestFrom225PercentInvestment

-- The proof statement
theorem donna_received_total_interest :
  totalInterest = 970 := by
sorry

end donna_received_total_interest_l452_452597


namespace line_intercept_form_l452_452151

theorem line_intercept_form 
  (P : ℝ × ℝ) 
  (a : ℝ × ℝ) 
  (l_eq : ∃ m : ℝ, ∀ x y : ℝ, (x, y) = P → y - 3 = m * (x - 2))
  (P_coord : P = (2, 3)) 
  (a_vect : a = (2, -6)) 
  : ∀ x y : ℝ, y - 3 = (-3) * (x - 2) → 3 * x + y - 9 = 0 →  ∃ a' b' : ℝ, a' ≠ 0 ∧ b' ≠ 0 ∧ x / 3 + y / 9 = 1 :=
by
  sorry

end line_intercept_form_l452_452151


namespace smallest_multiple_9_and_6_l452_452497

theorem smallest_multiple_9_and_6 : ∃ n : ℕ, n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m :=
by
  have h := Nat.lcm 9 6
  use h
  split
  sorry

end smallest_multiple_9_and_6_l452_452497


namespace find_initial_solution_liters_l452_452870

-- Define the conditions
def percentage_initial_solution_alcohol := 0.26
def added_water := 5
def percentage_new_mixture_alcohol := 0.195

-- Define the initial amount of the solution
def initial_solution_liters (x : ℝ) : Prop :=
  0.26 * x = 0.195 * (x + 5)

-- State the proof problem
theorem find_initial_solution_liters : initial_solution_liters 15 :=
by
  sorry

end find_initial_solution_liters_l452_452870


namespace incorrect_conclusion_l452_452648

variable {a b c : ℝ}

theorem incorrect_conclusion
  (h1 : a^2 + a * b = c)
  (h2 : a * b + b^2 = c + 5) :
  ¬(2 * c + 5 < 0) ∧ ¬(∃ k, a^2 - b^2 ≠ k) ∧ ¬(a = b ∨ a = -b) ∧ ¬(b / a > 1) :=
by sorry

end incorrect_conclusion_l452_452648


namespace rectangle_inscribed_circle_hypotenuse_l452_452443

open Real

theorem rectangle_inscribed_circle_hypotenuse
  (AB BC : ℝ)
  (h_AB : AB = 20)
  (h_BC : BC = 10)
  (r : ℝ)
  (h_r : r = 10 / 3) :
  sqrt ((AB - 2 * r) ^ 2 + BC ^ 2) = 50 / 3 :=
by {
  sorry
}

end rectangle_inscribed_circle_hypotenuse_l452_452443


namespace scale_division_l452_452520

/-- Given a scale that is 6 feet and 8 inches long and is divided into 4 equal parts, 
prove that each part is 20 inches long. -/
theorem scale_division :
  let feet_to_inches := 12 in
  let scale_feet := 6 * feet_to_inches in
  let scale_inches := 8 in
  let total_inches := scale_feet + scale_inches in
  total_inches / 4 = 20 :=
by
  let feet_to_inches := 12
  let scale_feet := 6 * feet_to_inches
  let scale_inches := 8
  let total_inches := scale_feet + scale_inches
  show total_inches / 4 = 20
  sorry

end scale_division_l452_452520


namespace value_of_series_l452_452126

theorem value_of_series :
  (∑ k in Finset.range 2022, (2023 - (k + 1)) / (k + 1)) / (∑ k in Finset.Ico 2 2024, 1 / k) = 2023 :=
by
  sorry

end value_of_series_l452_452126


namespace arithmetic_mean_of_primes_l452_452959

open Real

theorem arithmetic_mean_of_primes :
  let numbers := [33, 37, 39, 41, 43]
  let primes := numbers.filter Prime
  let sum_primes := (37 + 41 + 43 : ℤ)
  let count_primes := (3 : ℤ)
  let mean := (sum_primes / count_primes : ℚ)
  mean = 40.33 := by
sorry

end arithmetic_mean_of_primes_l452_452959


namespace three_lines_concur_l452_452909

theorem three_lines_concur 
  {A B C A1 A2 B1 B2 : Type*}
  [add_group A] [add_group B] [add_group C]
  [add_group A1] [add_group A2] [add_group B1] [add_group B2]
  (xA yA xB yB xC yC xA1 yA1 xA2 yA2 xB1 yB1 xB2 yB2 : ℝ)
  (hA1 : xA1 = xC + yA - yC ∧ yA1 = yC + xC - xA)
  (hA2 : xA2 = xA + yA - yC ∧ yA2 = yA + xC - xA)
  (hB1 : xB1 = xC - yB + yC ∧ yB1 = yC - xC + xB)
  (hB2 : xB2 = xB - yB + yC ∧ yB2 = yB - xC + xB)
  : ∃ P : Type*, ∃ (hA1B : ∃ x y, |{A1, B} := P),
              ∃ (hA2B2 : ∃ x y, |{A2, B2} := P),
              ∃ (hAB1 : ∃ x y, |{A, B1} := P), 
                true :=
begin
  sorry
end

end three_lines_concur_l452_452909


namespace neils_cookies_l452_452766

theorem neils_cookies (total_cookies : ℕ) (fraction_given_to_friend : ℚ) (remaining_cookies : ℕ) :
  total_cookies = 20 →
  fraction_given_to_friend = 2 / 5 →
  remaining_cookies = total_cookies - (total_cookies * (fraction_given_to_friend.num : ℕ) / fraction_given_to_friend.denom) →
  remaining_cookies = 12 :=
by
  intros h_total h_fraction h_remaining
  rw [h_total, h_fraction, h_remaining]
  norm_num

end neils_cookies_l452_452766


namespace coloring_ways_l452_452940

-- Definitions for colors
inductive Color
| red
| green

open Color

-- Definition of the coloring function
def color (n : ℕ) : Color := sorry

-- Conditions:
-- 1. Each positive integer is colored either red or green
def condition1 (n : ℕ) : n > 0 → (color n = red ∨ color n = green) := sorry

-- 2. The sum of any two different red numbers is a red number
def condition2 (r1 r2 : ℕ) : r1 ≠ r2 → color r1 = red → color r2 = red → color (r1 + r2) = red := sorry

-- 3. The sum of any two different green numbers is a green number
def condition3 (g1 g2 : ℕ) : g1 ≠ g2 → color g1 = green → color g2 = green → color (g1 + g2) = green := sorry

-- The required theorem
theorem coloring_ways : ∃! (f : ℕ → Color), 
  (∀ n, n > 0 → (f n = red ∨ f n = green)) ∧ 
  (∀ r1 r2, r1 ≠ r2 → f r1 = red → f r2 = red → f (r1 + r2) = red) ∧
  (∀ g1 g2, g1 ≠ g2 → f g1 = green → f g2 = green → f (g1 + g2) = green) :=
sorry

end coloring_ways_l452_452940


namespace proof_problem_l452_452664

section MathProof

variable (ω : ℝ) (A B : ℝ) (a b : ℝ)
variable (hω_pos : 0 < ω)
variable (hb : b = 2)
variable (hfA : f A = sqrt 3 - 1)
variable (hsin : sqrt 3 * a = 2 * b * sin A)
variable (f : ℝ → ℝ)
variable (xmin xmax : ℝ)

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (ω * x) - 2 * (sin (ω * x / 2))^2

axiom is_periodic : is_periodic f (3 * π)

-- Minimum and maximum values in the given interval
def min_value : ℝ := -sqrt 3 - 1
def max_value : ℝ := 1

-- Area of triangle ABC
def area_triangle := (3 + sqrt 3) / 3

-- Theorem statement
theorem proof_problem :
  (∀ x ∈ Icc (-3 * π / 4) π, min_value ≤ f x ∧ f x ≤ max_value)
  ∧ ((1 / 2) * a * b * sin (π - A - B) = area_triangle) :=
by
  sorry

end MathProof

end proof_problem_l452_452664


namespace ants_rice_transport_l452_452327

/-- 
Given:
  1) 12 ants can move 24 grains of rice in 6 trips.

Prove:
  How many grains of rice can 9 ants move in 9 trips?
-/
theorem ants_rice_transport :
  (9 * 9 * (24 / (12 * 6))) = 27 := 
sorry

end ants_rice_transport_l452_452327


namespace polynomial_sat_condition_l452_452345

theorem polynomial_sat_condition (P : Polynomial ℝ) (k : ℕ) (hk : 0 < k) :
  (P.comp P = P ^ k) →
  (P = 0 ∨ P = 1 ∨ (k % 2 = 1 ∧ P = -1) ∨ P = Polynomial.X ^ k) :=
sorry

end polynomial_sat_condition_l452_452345


namespace part_A_part_B_part_D_l452_452412

variables (c d : ℤ)

def multiple_of_5 (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k
def multiple_of_10 (x : ℤ) : Prop := ∃ k : ℤ, x = 10 * k

-- Given conditions
axiom h1 : multiple_of_5 c
axiom h2 : multiple_of_10 d

-- Problems to prove
theorem part_A : multiple_of_5 d := by sorry
theorem part_B : multiple_of_5 (c - d) := by sorry
theorem part_D : multiple_of_5 (c + d) := by sorry

end part_A_part_B_part_D_l452_452412


namespace cycloid_length_l452_452916

/-- The parametric equations of a cycloid. -/
def cycloid_x (a t : ℝ) : ℝ := a * (t - sin t)
def cycloid_y (a t : ℝ) : ℝ := a * (1 - cos t)

/-- The length of one branch of the cycloid computed by arc length integral. -/
theorem cycloid_length (a : ℝ) : 
  (∫ t in 0..(2*π), sqrt (((deriv (λ t, cycloid_x a t)) t)^2 + ((deriv (λ t, cycloid_y a t)) t)^2)) = 8 * a :=
by
  sorry

end cycloid_length_l452_452916


namespace coloring_ways_l452_452803

/-
  Given a map consisting of five states and three available colors
  (green, blue, yellow), there are exactly 6 ways to color the map
  such that no two neighboring states share the same color.
-/

def color_map := 
  ∃ (A B C D E : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧
  B ≠ C ∧ B ≠ E ∧
  C ≠ D ∧
  D ≠ E ∧
  (A = 1 ∨ A = 2 ∨ A = 3) ∧
  (B = 1 ∨ B = 2 ∨ B = 3) ∧
  (C = 1 ∨ C = 2 ∨ C = 3) ∧
  (D = 1 ∨ D = 2 ∨ D = 3) ∧
  (E = 1 ∨ E = 2 ∨ E = 3)

theorem coloring_ways (h : color_map) : 6 := 
  sorry

end coloring_ways_l452_452803


namespace value_of_y_at_x_8_l452_452692

theorem value_of_y_at_x_8 (k : ℝ) (x y : ℝ) 
  (hx1 : y = k * x^(1/3)) 
  (hx2 : y = 4 * Real.sqrt 3) 
  (hx3 : x = 64) 
  (hx4 : 8^(1/3) = 2) : 
  (y = 2 * Real.sqrt 3) := 
by 
  sorry

end value_of_y_at_x_8_l452_452692


namespace exists_zero_in_interval_l452_452807

theorem exists_zero_in_interval :
  ∃ x ∈ (2 : ℝ, 3), (λ x : ℝ, Real.log x - 1) x = 0 :=
by
  -- Define the function f(x)
  let f := λ x : ℝ, Real.log x - 1
  -- Monotonicity condition (not necessary to state explicitly, but can be mentioned)
  have monotonic : ∀ x y : ℝ, x < y → f x < f y := 
    λ x y hx, Real.log_lt_log hx
  -- Conditions at the interval ends
  have f2 : f 2 < 0 := by norm_num; exact Real.log_lt_one_of_lt (by norm_num)
  have f3 : f 3 > 0 := by norm_num; exact Real.one_lt_log_of_lt (by norm_num)
  -- Conclude that there is a zero in the interval (2, 3)
  have exists_zero := IntermediateValueTheorem exists_zero_in_interval_monotonic f 2 3 f2 f3 monotonic sorry
  exact exists_zero

end exists_zero_in_interval_l452_452807


namespace increased_amount_is_30_l452_452003

noncomputable def F : ℝ := (3 / 2) * 179.99999999999991
noncomputable def F' : ℝ := (5 / 3) * 179.99999999999991
noncomputable def J : ℝ := 179.99999999999991
noncomputable def increased_amount : ℝ := F' - F

theorem increased_amount_is_30 : increased_amount = 30 :=
by
  -- Placeholder for proof. Actual proof goes here.
  sorry

end increased_amount_is_30_l452_452003


namespace collinear_X_Y_Z_l452_452364

open EuclideanGeometry

variables {Point : Type*} [MetricSpace Point]
variables (A C H I J Z X Y : Point)

-- Given a rectangle JHIZ
def rectangle (J H I Z : Point) : Prop := 
  (J, H, I, Z form a rectangle)

-- Given points A and C on sides ZI and ZJ respectively
def points_on_sides (A C H I J Z : Point) : Prop := 
  (A ∈ line_segment Z I) ∧ (C ∈ line_segment Z J)

-- Given perpendiculars from A on CH intersecting HI at X and from C on AH intersecting HJ at Y
def perpendicular_intersections (A C H I J Z X Y : Point) : Prop :=
  perpendicular A (line_segment C H) ∧ intersects (line_of A X) (line_seg H I) X ∧
  perpendicular C (line_segment A H) ∧ intersects (line_of C Y) (line_seg H J) Y

-- The theorem to prove
theorem collinear_X_Y_Z (J H I Z A C X Y : Point)
  (rect : rectangle J H I Z)
  (pts : points_on_sides A C H I J Z)
  (perp : perpendicular_intersections A C H I J Z X Y) : collinear X Y Z :=
sorry

end collinear_X_Y_Z_l452_452364


namespace cos_pi_plus_2alpha_l452_452175

theorem cos_pi_plus_2alpha (α : ℝ) (h : sin α = 1 / 3) : cos (Real.pi + 2 * α) = -7 / 9 :=
by
  sorry

end cos_pi_plus_2alpha_l452_452175


namespace four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l452_452688

-- Number of four-digit numbers greater than 3999 such that the product of the middle two digits > 12 is 4260
theorem four_digit_numbers_greater_3999_with_middle_product_exceeding_12
  {d1 d2 d3 d4 : ℕ}
  (h1 : 4 ≤ d1 ∧ d1 ≤ 9)
  (h2 : 0 ≤ d4 ∧ d4 ≤ 9)
  (h3 : 1 ≤ d2 ∧ d2 ≤ 9)
  (h4 : 1 ≤ d3 ∧ d3 ≤ 9)
  (h5 : d2 * d3 > 12) :
  (6 * 71 * 10 = 4260) :=
by
  sorry

end four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l452_452688


namespace red_balls_removal_l452_452268

theorem red_balls_removal (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (x : ℕ) :
  total_balls = 600 →
  red_balls = 420 →
  blue_balls = 180 →
  (red_balls - x) / (total_balls - x : ℚ) = 3 / 5 ↔ x = 150 :=
by 
  intros;
  sorry

end red_balls_removal_l452_452268


namespace KBrO3_decomposition_and_weight_l452_452120

def potassium_atomic_weight : ℝ := 39.10
def bromine_atomic_weight : ℝ := 79.90
def oxygen_atomic_weight : ℝ := 16.00
def KBrO3_molecular_weight : ℝ :=
  potassium_atomic_weight + bromine_atomic_weight + 3 * oxygen_atomic_weight

theorem KBrO3_decomposition_and_weight (K Br O : Type) [element K] [element Br] [element O]
  (K_wt : real) (Br_wt : real) (O_wt : real) :
  K_wt = 39.10 ∧ Br_wt = 79.90 ∧ O_wt = 16.00 →
  (KBrO3_molecular_weight = 167.00 ∧ (2 * K • 1 * Br • 3 * O) = (2 * K • Br + 3 * O)) :=
  by
    intros h_wt
    cases h_wt with hK wt_rest
    cases wt_rest with hBr hO
    sorry


end KBrO3_decomposition_and_weight_l452_452120


namespace probability_red_white_not_yellow_correct_l452_452598

noncomputable def probability_red_white_not_yellow : ℚ :=
  (choose 3 1) * (1 / 2) * (1 / 3)^(2:ℚ) + (choose 3 2) * (1 / 2)^(2:ℚ) * (1 / 3)

theorem probability_red_white_not_yellow_correct :
  probability_red_white_not_yellow = 5 / 12 :=
by
  sorry

end probability_red_white_not_yellow_correct_l452_452598


namespace arcsin_cos_arcsin_rel_arccos_sin_arccos_l452_452156

theorem arcsin_cos_arcsin_rel_arccos_sin_arccos (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) :
    let α := Real.arcsin (Real.cos (Real.arcsin x))
    let β := Real.arccos (Real.sin (Real.arccos x))
    (Real.arcsin x + Real.arccos x = π / 2) → α + β = π / 2 :=
by
  let α := Real.arcsin (Real.cos (Real.arcsin x))
  let β := Real.arccos (Real.sin (Real.arccos x))
  intro h_arcsin_arccos_eq
  sorry

end arcsin_cos_arcsin_rel_arccos_sin_arccos_l452_452156


namespace maxRemovableCubes_l452_452863

-- Define the conditions for the \(3 \times 3 \times 3\) cube and the connectivity requirements
def isCubeConnected (cubes : Set (ℤ × ℤ × ℤ)) : Prop :=
  ∀ (a b : ℤ × ℤ × ℤ), a ∈ cubes → b ∈ cubes → ∃ (path : List (ℤ × ℤ × ℤ)),
    path.head = a ∧ path.last = b ∧ ∀ {p1 p2 : ℤ × ℤ × ℤ}, (p1, p2) ∈ path.zip path.tail → 
    (abs (p1.1 - p2.1) + abs (p1.2 - p2.2) + abs (p1.3 - p2.3)) = 1

def allFacesVisible (cubes : Set (ℤ × ℤ × ℤ)) : Prop :=
  ∀ i ∈ Finset.range 3, ∀ j ∈ Finset.range 3, 
    (∃ k ∈ Finset.range 3, (i, j, k) ∈ cubes) ∧ 
    (∃ k ∈ Finset.range 3, (i, k, j) ∈ cubes) ∧ 
    (∃ k ∈ Finset.range 3, (k, i, j) ∈ cubes)

-- The main statement that captures the problem
theorem maxRemovableCubes : 
  ∃ (cubes : Set (ℤ × ℤ × ℤ)), 
    cubes ⊆ {(x, y, z) | x ∈ Finset.range 3 ∧ y ∈ Finset.range 3 ∧ z ∈ Finset.range 3} ∧
    allFacesVisible cubes ∧ isCubeConnected cubes ∧ 
    cubes.card = 13 := 
begin
  -- Placeholder for the proof
  sorry
end

end maxRemovableCubes_l452_452863


namespace gcd_of_256_180_600_l452_452841

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l452_452841


namespace initial_speed_is_sixty_l452_452887

variable (D T : ℝ)

-- Condition: Two-thirds of the distance is covered in one-third of the total time.
def two_thirds_distance_in_one_third_time (V : ℝ) : Prop :=
  (2 * D / 3) / V = T / 3

-- Condition: The remaining distance is covered at 15 kmph.
def remaining_distance_at_fifteen_kmph : Prop :=
  (D / 3) / 15 = T - T / 3

-- Given that 30T = D from simplification in the solution.
def distance_time_relationship : Prop :=
  D = 30 * T

-- Prove that the initial speed V is 60 kmph.
theorem initial_speed_is_sixty (V : ℝ) (h1 : two_thirds_distance_in_one_third_time D T V) (h2 : remaining_distance_at_fifteen_kmph D T) (h3 : distance_time_relationship D T) : V = 60 := 
  sorry

end initial_speed_is_sixty_l452_452887


namespace sum_of_vectors_center_zero_sum_of_vectors_arbitrary_point_l452_452355

variable {V : Type} [AddCommGroup V] [Module ℝ V]

noncomputable def center_of_regular_n_gon (n : ℕ) (i : ℕ) : V :=
sorry -- assume we have a function that gives the i-th vertex of a regular n-gon

theorem sum_of_vectors_center_zero (n : ℕ) (O : V) :
  (∑ i in Finset.range n, center_of_regular_n_gon n i) = (0 : V) :=
sorry

theorem sum_of_vectors_arbitrary_point (n : ℕ) (O X : V) :
  (∑ i in Finset.range n, ((X + center_of_regular_n_gon n i) - O)) = n • (X - O) :=
sorry

end sum_of_vectors_center_zero_sum_of_vectors_arbitrary_point_l452_452355


namespace nails_per_plank_l452_452172

theorem nails_per_plank (total_nails planks : ℕ) (h₁ : total_nails = 32) (h₂ : planks = 16) : total_nails / planks = 2 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end nails_per_plank_l452_452172


namespace equidistant_points_locus_eq_l452_452615

noncomputable theory
open_locale classical

variables {R : Type*} [linear_ordered_field R]

-- Definition of the vertices
def A : euclidean_space R (fin 3) := (0, 0, 0)
def B : euclidean_space R (fin 3) := (1, 0, 0)
def D : euclidean_space R (fin 3) := (0, 1, 0)
def A' : euclidean_space R (fin 3) := (0, 0, 1)

-- Definition of the edges
def BB' : set (euclidean_space R (fin 3)) := { p | p = (1, 0, z) for some z }
def CD : set (euclidean_space R (fin 3)) := { p | p = (0, 1, z) for some z }
def A'D' : set (euclidean_space R (fin 3)) := { p | p = (x, 0, 1) for some x }

-- The main theorem
theorem equidistant_points_locus_eq : 
  ∀ (M : euclidean_space R (fin 3)), 
  (dist M BB') = (dist M CD) ∧ (dist M CD) = (dist M A'D') ↔ M = (x, x, x) :=
sorry

end equidistant_points_locus_eq_l452_452615


namespace quadratic_expression_value_l452_452249

theorem quadratic_expression_value :
  ∀ x1 x2 : ℝ, (x1^2 - 4 * x1 - 2020 = 0) ∧ (x2^2 - 4 * x2 - 2020 = 0) →
  (x1^2 - 2 * x1 + 2 * x2 = 2028) :=
by
  intros x1 x2 h
  sorry

end quadratic_expression_value_l452_452249


namespace discount_problem_l452_452901

theorem discount_problem :
  ∃ n : ℕ, (n > 37) ∧ 
           (1 - n / 100 > 0.64) ∧ 
           (1 - n / 100 > 0.681472) ∧ 
           (1 - n / 100 > 0.63) :=
by
  sorry

end discount_problem_l452_452901


namespace kh_perpendicular_cd_l452_452293

noncomputable def convex_quad := sorry
noncomputable def is_midpoint (P Q R : Point) := sorry
noncomputable def parallel (L₁ L₂ : Line) := sorry
noncomputable def orthocenter(T: Triangle) : Point := sorry
noncomputable def is_midpoint_of_segment (P₁ P₂: Point) (L : Line) : Bool := sorry
noncomputable def perpendicular (L₁ L₂ : Line) : Prop := sorry

theorem kh_perpendicular_cd (A B C D M N K H : Point)
  (h_convex_quad : convex_quad A B C D)
  (h_equal_angles : ∠ A = ∠ C)
  (h_points_on_sides : on_side M A B ∧ on_side N B C)
  (h_parallel : parallel (line_through M N) (line_through A D))
  (h_double_length : length (segment M N) = 2 * length (segment A D))
  (h_is_midpoint : is_midpoint K M N)
  (h_orthocenter : H = orthocenter (triangle A B C)) :
  perpendicular (line_through K H) (line_through C D) :=
sorry

end kh_perpendicular_cd_l452_452293


namespace max_halls_visited_15_l452_452441

theorem max_halls_visited_15 :
  ∀ (halls : Fin 16 → Bool) (adj : Fin 16 → Fin 16 → Prop) (A B : Fin 16),
    (∀ i, ∃! j, adj i j) →
    (A ≠ B) →
    (halls A = true) →
    (halls B = true) →
    (∀ i, (halls i = true ∧ ∃! j, adj i j ∧ halls j = false) ∨
          (halls i = false ∧ ∃! j, adj i j ∧ halls j = true)) →
    (∀ p : List (Fin 16), p.head = some A → p.last = some B → (∀ i ∈ p, ∀ j ∈ p, i ≠ j → ¬ adj i j) →
                          List.length p ≤ 15) :=
by
  sorry

end max_halls_visited_15_l452_452441


namespace locus_of_M_is_circle_with_diameter_CN_l452_452865

-- Define the setup of the points A, B, C, and D
variables {A B C D P Q M N L : Type}
variables [point A] [point B] [point C] [point D]
variables [line AB] [line CD] [line AD] [line BD]
variables [line PQ] [line MN] [line L]

-- Conditions: A, B, C are collinear, and D is not collinear with A, B, C
axiom A_B_C_collinear: collinear A B C
axiom D_not_collinear_ABC: ¬ collinear A B C D

-- Definitions of P and Q based on given conditions
axiom P_on_BD_parallel_AD: ∃ P, parallel (line_through C D) (line_through A D)
axiom Q_on_AD_parallel_BD: ∃ Q, parallel (line_through C D) (line_through B D)

-- Definition of the intersection N
axiom N_intersects_PQ_and_AB: intersects PQ AB

-- Definition of M as the foot of the perpendicular from C to PQ
axiom M_perpendicular_C_to_PQ: perpendicular (line_through C M) PQ

-- Main theorem statement
theorem locus_of_M_is_circle_with_diameter_CN:
  ∀ (A B C D P Q M N : point), 
  collinear A B C ∧ ¬ collinear A B C D ∧ 
  (∃ P, parallel (line_through C D) (line_through A D)) ∧
  (∃ Q, parallel (line_through C D) (line_through B D)) ∧
  intersects PQ AB ∧ 
  perpendicular (line_through C M) PQ → 
  locus (foot_of_perpendicular C PQ) = circle_with_diameter C N
:= sorry

end locus_of_M_is_circle_with_diameter_CN_l452_452865


namespace arithmetic_mean_prime_numbers_l452_452963

-- Define the list of numbers
def num_list : List ℕ := [33, 37, 39, 41, 43]

-- Define a predicate for primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Extract list of prime numbers
def prime_numbers : List ℕ := num_list.filter is_prime

-- Compute the arithmetic mean of a list of natural numbers
def arithmetic_mean (nums : List ℕ) : ℚ :=
  nums.foldr Nat.add 0 / nums.length

-- The main theorem: Proving the arithmetic mean of prime numbers in the list
theorem arithmetic_mean_prime_numbers :
  arithmetic_mean prime_numbers = 121 / 3 :=
by 
  -- Include proof steps here
  sorry

end arithmetic_mean_prime_numbers_l452_452963


namespace gcd_of_256_180_600_l452_452831

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l452_452831


namespace find_thin_mints_price_l452_452106

def price_of_samoas : ℕ := 4
def number_of_samoas_boxes : ℕ := 3
def price_of_fudge_delights : ℕ := 5
def number_of_fudge_delights_boxes : ℕ := 1
def price_of_sugar_cookies : ℕ := 2
def number_of_sugar_cookies_boxes : ℕ := 9
def total_amount_made : ℕ := 42
def number_of_thin_mints_boxes : ℕ := 2

-- This unchanged does satisfy the profit condition.
noncomputable def price_of_each_thin_mint := 
  (total_amount_made - 
  (number_of_samoas_boxes * price_of_samoas + 
  number_of_fudge_delights_boxes * price_of_fudge_delights + 
  number_of_sugar_cookies_boxes * price_of_sugar_cookies)) / number_of_thin_mints_boxes

theorem find_thin_mints_price : price_of_each_thin_mint = 3.5 := 
by
  sorry

end find_thin_mints_price_l452_452106


namespace minimum_a_l452_452290

theorem minimum_a (a : ℝ) (h : a > 0) :
  (∀ (N : ℝ × ℝ), (N.1 - a)^2 + (N.2 + a - 3)^2 = 1 → 
   dist (N.1, N.2) (0, 0) ≥ 2) → a ≥ 3 :=
by
  sorry

end minimum_a_l452_452290


namespace part_I_part_II_l452_452231

-- Part I: Prove that the value of \( a \) is -2
theorem part_I (a : ℝ) : 
  B ⊆ A → 
  a = -2 :=
by
  -- define the sets A and B
  let A := {x : ℝ | 0 ≤ Real.log 2 x ∧ Real.log 2 x ≤ 2}
  let B := {a ^ 2 + 4 * a + 8, a + 3, 3 * Real.log 2 (abs a)}
  assume h : B ⊆ A
  -- solution still required
  sorry

-- Part II: Prove that the range of \( m \) is [0, 1]
theorem part_II (m : ℝ) : 
  (∀ x ∈ A, x ^ m ∈ A) → 
  0 ≤ m ∧ m ≤ 1 :=
by
  -- define the sets A and C
  let A := {x : ℝ | 0 ≤ Real.log 2 x ∧ Real.log 2 x ≤ 2}
  let C := {y : ℝ | ∃ x ∈ A, y = x ^ m}
  assume h : ∀ x ∈ A, x ^ m ∈ A
  -- solution still required
  sorry

end part_I_part_II_l452_452231


namespace center_is_five_l452_452082

def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def in_grid (n : ℕ) (grid : ℕ × ℕ → ℕ) : Prop :=
  ∃ (i j : ℕ), grid (i, j) = n

def consecutive_adjacent (grid : ℕ × ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), ∀ (di dj : ℕ), (di, dj) ∈ {(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)} →
    i + di ∈ {0, 1, 2} ∧ j + dj ∈ {0, 1, 2} → (abs (grid (i, j) - grid (i + di, j + dj)) = 1)

noncomputable def center_square_value (grid : ℕ × ℕ → ℕ) : ℕ :=
  grid (1, 1)

def corner_positions : List (ℕ × ℕ) :=
  [(0, 0), (0, 2), (2, 0), (2, 2)]

axiom corner_values (grid : ℕ × ℕ → ℕ) : 
  ∀ (p : ℕ × ℕ), p ∈ corner_positions → grid p ∈ {2, 4, 6, 8}

theorem center_is_five (grid : ℕ × ℕ → ℕ) 
  (h1 : ∀ n ∈ numbers, in_grid n grid)
  (h2 : consecutive_adjacent grid)
  (h3 : corner_values grid)
  : center_square_value grid = 5 :=
sorry

end center_is_five_l452_452082


namespace second_player_wins_l452_452076

structure Chessboard :=
  (size : ℕ)
  (is_valid_square : ℕ × ℕ → Prop)

def initial_positions : Chessboard → (ℕ × ℕ) × (ℕ × ℕ) := 
  λ cb, ((2, 2), (3, 4))

def legal_move (cb : Chessboard) (pos : ℕ × ℕ) (new_pos : ℕ × ℕ) (opponent_pos : ℕ × ℕ) : Prop :=
  cb.is_valid_square new_pos ∧
  ((fst pos = fst new_pos ∨ snd pos = snd new_pos) ∧ 
   new_pos ≠ opponent_pos)

noncomputable def can_second_player_win (cb : Chessboard) (pos1 pos2 : ℕ × ℕ) : Prop := sorry

theorem second_player_wins :
  ∀ (cb : Chessboard) (pos1 pos2 : ℕ × ℕ),
    cb.size = 8 →
    cb.is_valid_square = (λ (p : ℕ × ℕ), 1 ≤ p.1 ∧ p.1 ≤ 8 ∧ 1 ≤ p.2 ∧ p.2 ≤ 8) →
    pos1 = (2, 2) →
    pos2 = (3, 4) →
    can_second_player_win cb pos1 pos2 :=
begin
  intros cb pos1 pos2 size_constraint board_constraint white_position black_position,
  sorry
end

end second_player_wins_l452_452076


namespace Missy_first_year_savings_l452_452764

theorem Missy_first_year_savings : 
  ∃ x : ℝ, (x + 2 * x + 4 * x + 8 * x = 450) ∧ x = 30 :=
begin
  sorry
end

end Missy_first_year_savings_l452_452764


namespace max_value_Tn_l452_452927

noncomputable def geom_seq (a : ℕ → ℝ) : Prop := 
∀ n : ℕ, a (n+1) = 2 * a n

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a 0 * (1 - (2 : ℝ)^n) / (1 - (2 : ℝ))

noncomputable def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(9 * sum_first_n_terms a n - sum_first_n_terms a (2 * n)) / (a n * (2 : ℝ)^n)

theorem max_value_Tn (a : ℕ → ℝ) (h : geom_seq a) : 
  ∃ n, T_n a n ≤ 3 :=
sorry

end max_value_Tn_l452_452927


namespace area_of_parallelogram_l452_452146

def parallelogram_base : ℝ := 26
def parallelogram_height : ℝ := 14

theorem area_of_parallelogram : parallelogram_base * parallelogram_height = 364 := by
  sorry

end area_of_parallelogram_l452_452146


namespace compute_p_plus_q_l452_452361

noncomputable def series_sum : ℚ := 
  let series1_sum := Σ' (n : ℕ), (2 * n + 1) / (2 ^ (2 * n + 1))
  let series2_sum := Σ' (n : ℕ), (2 * (n + 1)) / (4 ^ (2 * (n + 1)))
  series1_sum + series2_sum

theorem compute_p_plus_q :
  ∃ (p q : ℕ), p.gcd q = 1 ∧ series_sum = p / q ∧ p + q = 169 := 
sorry

end compute_p_plus_q_l452_452361


namespace geometric_sequence_new_product_l452_452637

theorem geometric_sequence_new_product 
  (a r : ℝ) (n : ℕ) (h_even : n % 2 = 0)
  (P S S' : ℝ)
  (hP : P = a^n * r^(n * (n-1) / 2))
  (hS : S = a * (1 - r^n) / (1 - r))
  (hS' : S' = (1 - r^n) / (a * (1 - r))) :
  (2^n * a^n * r^(n * (n-1) / 2)) = (S * S')^(n / 2) :=
sorry

end geometric_sequence_new_product_l452_452637


namespace min_value_of_expr_l452_452243

theorem min_value_of_expr (a : ℝ) (h : a > 3) : ∃ m, (∀ b > 3, b + 4 / (b - 3) ≥ m) ∧ m = 7 :=
sorry

end min_value_of_expr_l452_452243


namespace average_salary_for_company_l452_452053

theorem average_salary_for_company
    (number_of_managers : Nat)
    (number_of_associates : Nat)
    (average_salary_managers : Nat)
    (average_salary_associates : Nat)
    (hnum_managers : number_of_managers = 15)
    (hnum_associates : number_of_associates = 75)
    (has_managers : average_salary_managers = 90000)
    (has_associates : average_salary_associates = 30000) : 
    (number_of_managers * average_salary_managers + number_of_associates * average_salary_associates) / 
    (number_of_managers + number_of_associates) = 40000 := 
    by
    sorry

end average_salary_for_company_l452_452053


namespace total_delegates_l452_452089

theorem total_delegates 
  (D: ℕ) 
  (h1: 16 ≤ D)
  (h2: (D - 16) % 2 = 0)
  (h3: 10 ≤ D - 16) : D = 36 := 
sorry

end total_delegates_l452_452089


namespace a7_b7_ratio_l452_452979

variable (S_n T_n : ℕ → ℚ)
variable (a b : ℕ → ℚ)
variable (n : ℕ)

-- Conditions
hypothesis hS : ∀ n, S_n n = (n / 2) * (a 1 + a (n + 1))
hypothesis hT : ∀ n, T_n n = (n / 2) * (b 1 + b (n + 1))
hypothesis hRatio : ∀ n, S_n n / T_n n = (2 * n + 1) / (n + 3)

-- Question
theorem a7_b7_ratio : a 7 / b 7 = 27 / 16 :=
by
  sorry

end a7_b7_ratio_l452_452979


namespace polynomial_mod_pow_p_l452_452736

-- Definitions of the conditions
def Gamma := { f : Polynomial ℤ | true } -- Polynomials in x with integer coefficients

def coeff_mod (f g : Polynomial ℤ) (m : ℤ) : Prop := 
  ∀ (i : ℕ), (f.coeff i - g.coeff i) % m = 0

variables {n p : ℕ} (hp : Nat.Prime p)
variables (f g h r s : Polynomial ℤ)
variables (hf : coeff_mod (r * f + s * g) 1 p)
variables (hg : coeff_mod (f * g) h p)

theorem polynomial_mod_pow_p :
  ∃ F G : Polynomial ℤ, coeff_mod F f p ∧ coeff_mod G g p ∧ coeff_mod (F * G) h (p ^ n) := by
  sorry

end polynomial_mod_pow_p_l452_452736


namespace volume_of_region_l452_452620

noncomputable def f (x y z : ℝ) : ℝ :=
|x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region :
  (∫ x in 0..2, ∫ y in 0..1, ∫ z in 0..1, if f x y z ≤ 6 then 1 else 0) = 2 :=
by
  sorry

end volume_of_region_l452_452620


namespace disc_minutes_l452_452904

-- Define the conditions given in the problem.
def total_minutes : Nat := 385
def disc_capacity : Nat := 75
def num_discs : Nat := (total_minutes.toNat / disc_capacity.toNat) + 1

-- State the goal to be proved.
theorem disc_minutes : total_minutes / num_discs = 64 := by
  -- Proof to be provided
  sorry

end disc_minutes_l452_452904


namespace point_M_on_diagonal_AC_l452_452864

open EuclideanGeometry

noncomputable theory

variables {A B C D G E F H I M : Point}
variables (AB AD BC CD : Line)
variables (Gamma : Circle)

theorem point_M_on_diagonal_AC
  (hG : G ∈ parallelogram A B C D)
  (hGamma : Γ = circle_passing_through_two_points A G)
  (hE : E ∈ (AB : Set Point) ∧ E ≠ A ∧ E ≠ B ∧ E ∈ Γ)
  (hF : F ∈ (AD : Set Point) ∧ F ≠ A ∧ F ≠ D ∧ F ∈ Γ)
  (hH : H ∈ (BC : Set Point) ∧ H ≠ B ∧ H ≠ C ∧ collinear' F G H)
  (hI : I ∈ (CD : Set Point) ∧ I ≠ C ∧ I ≠ D ∧ collinear' E G I)
  (hM : M ≠ G ∧ M ∈ circumcircle_of_triangle H G I ∧ M ∈ Γ) :
  collinear' A M C :=
sorry

end point_M_on_diagonal_AC_l452_452864


namespace equal_final_temperatures_l452_452804

def convertFtoC (F : ℤ) : ℤ := (5 * (F - 32)) / 9

def convertCtoF (C : ℤ) : ℤ := (9 * C) / 5 + 32

def final_convertFtoC (F : ℤ) : ℤ := convertFtoC (convertCtoF (convertFtoC F))

theorem equal_final_temperatures :
  (Finset.range (1201 - 30)).filter (λ n, 
    let F := n + 30 in F = final_convertFtoC F
  ).card = 130 := 
sorry

end equal_final_temperatures_l452_452804


namespace incorrect_statement_D_l452_452791

-- Assume that the weight y and height x have a linear correlation.
-- The regression line equation is given by y = 0.85x - 85.71.
variable (x y : ℝ) (n : ℕ)

def is_linear_correlation (xy : ℕ → (ℝ × ℝ)) : Prop :=
  ∃ a b, (∀ i, xy i = (xy i).fst * a + b) ∧ a = 0.85 ∧ b = -85.71

-- The regression line equation is proven to be y = 0.85x - 85.71 using least squares method.
def regression_line (x : ℝ) : ℝ := 0.85 * x - 85.71

-- The incorrect conclusion statement D: "If a high school girl has a height of x = 160 cm,
-- it can be determined that her weight must be 50.29 kg."
def incorrect_conclusion_statement : Prop :=
  regression_line 160 ≠ 50.29

theorem incorrect_statement_D 
  (xy : ℕ → (ℝ × ℝ))
  (h_linear : is_linear_correlation xy) :
  incorrect_conclusion_statement := 
sorry

end incorrect_statement_D_l452_452791


namespace find_a_b_l452_452220

noncomputable def f (x : ℝ) := 3^x + x - 5

theorem find_a_b (a b : ℕ) (h_root : ∃ x_0, x_0 ∈ set.Icc ↑a ↑b ∧ f x_0 = 0) (h_diff : b - a = 1) (h_natural : a > 0 ∧ b > 0) : a + b = 3 :=
by
  sorry

end find_a_b_l452_452220


namespace hyperbola_eccentricity_l452_452227

-- Define the conditions
def hyperbola (a b : ℝ) (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def parabola (x y : ℝ) := y^2 = 4 * x
def focusF2 := (1, 0)
def pointA := (1, 2)

-- Main theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (e : ℝ) : 
  (a > 0) ∧ (b > 0) ∧ 
  hyperbola a b 1 2 ∧ 
  |((1 : ℝ), (2 : ℝ)) - focusF2| = 2 ∧ 
  e = (1 / (ℝ.sqrt 2 - 1)) :=
e = ℝ.sqrt 2 + 1

end hyperbola_eccentricity_l452_452227


namespace cassie_nails_l452_452108

-- Define the number of pets
def num_dogs := 4
def num_parrots := 8
def num_cats := 2
def num_rabbits := 6

-- Define the number of nails/claws/toes per pet
def nails_per_dog := 4 * 4
def common_claws_per_parrot := 2 * 3
def extra_toed_parrot_claws := 2 * 4
def toes_per_cat := 2 * 5 + 2 * 4
def rear_nails_per_rabbit := 2 * 5
def front_nails_per_rabbit := 3 + 4

-- Calculations
def total_dog_nails := num_dogs * nails_per_dog
def total_parrot_claws := 7 * common_claws_per_parrot + extra_toed_parrot_claws
def total_cat_toes := num_cats * toes_per_cat
def total_rabbit_nails := num_rabbits * (rear_nails_per_rabbit + front_nails_per_rabbit)

-- Total nails/claws/toes
def total_nails := total_dog_nails + total_parrot_claws + total_cat_toes + total_rabbit_nails

-- Theorem stating the problem
theorem cassie_nails : total_nails = 252 :=
by
  -- Here we would normally have the proof, but we'll skip it with sorry
  sorry

end cassie_nails_l452_452108


namespace lexus_sold_l452_452878

-- Definitions based on the problem conditions
def total_cars_sold := 300
def percentage_audi := 0.10
def percentage_toyota := 0.25
def percentage_bmw := 0.15
def percentage_acura := 0.30

-- Theorem that states the number of Lexuses sold
theorem lexus_sold : 
  ∃ lexuses_sold : ℕ, lexuses_sold = total_cars_sold * (1 - (percentage_audi + percentage_toyota + percentage_bmw + percentage_acura)) :=
by 
  -- Calculating the remaining percentage for Lexuses
  let remaining_percentage := 1 - (percentage_audi + percentage_toyota + percentage_bmw + percentage_acura)
  -- Calculating the number of Lexuses sold
  let lexuses_sold := total_cars_sold * remaining_percentage
  existsi nat.floor (total_cars_sold * remaining_percentage)
  sorry

end lexus_sold_l452_452878


namespace pow_div_pow_l452_452099

variable (a : ℝ)
variable (A B : ℕ)

theorem pow_div_pow (a : ℝ) (A B : ℕ) : a^A / a^B = a^(A - B) :=
  sorry

example : a^6 / a^2 = a^4 :=
  pow_div_pow a 6 2

end pow_div_pow_l452_452099


namespace ratio_relation_l452_452013

variables {A B C D P Q : Type}
variables [parallelogram A B C D] (secant_through_A : line A -> A) 
variables (P : intersection_of secant_through_A diagonal_BD) 
variables (Q : intersection_of secant_through_A side_CD) 

theorem ratio_relation (k : ℝ) (h : (DQ / QC) = (1 / k)) : (PD / PB) = (1 / (k + 1)) :=
sorry

end ratio_relation_l452_452013


namespace max_x_y3_z4_l452_452756

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_x_y3_z4 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  max_value_expression x y z ≤ 1 :=
sorry

end max_x_y3_z4_l452_452756


namespace line_equation_l452_452639

-- Definitions based on conditions
def Point (α : Type) := α × α
def Line (α : Type) := α × α × α

def passesThrough (l : Line ℝ) (P : Point ℝ) : Prop := 
  let (a, b, c) := l
  let (x, y) := P
  a * x + b * y + c = 0

def parallel (l₁ l₂ : Line ℝ) : Prop :=
  let (a₁, b₁, _) := l₁
  let (a₂, b₂, _) := l₂
  a₁ * b₂ = a₂ * b₁

-- Given conditions
def P := (2 : ℝ, 1 : ℝ)
def l_par := (2 : ℝ, -1 : ℝ, 2 : ℝ)

-- Theorem to prove
theorem line_equation (l : Line ℝ) 
  (h₁ : passesThrough l P) 
  (h₂ : parallel l l_par) : 
  l = (2, -1, -3) :=
sorry

end line_equation_l452_452639


namespace cookies_left_l452_452769

theorem cookies_left (total_cookies : ℕ) (fraction_given : ℚ) (given_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 20)
  (h2 : fraction_given = 2/5)
  (h3 : given_cookies = fraction_given * total_cookies)
  (h4 : remaining_cookies = total_cookies - given_cookies) :
  remaining_cookies = 12 :=
by
  sorry

end cookies_left_l452_452769


namespace diameter_of_lake_l452_452015

theorem diameter_of_lake (d : ℝ) (pi : ℝ) (h1 : pi = 3.14) 
  (h2 : 3.14 * d - d = 1.14) : d = 0.5327 :=
by
  sorry

end diameter_of_lake_l452_452015


namespace sum_abcd_l452_452745

variable {a b c d : ℚ}

theorem sum_abcd 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 4) : 
  a + b + c + d = -2/3 :=
by sorry

end sum_abcd_l452_452745


namespace final_cup_order_l452_452072

theorem final_cup_order :
  let initial_order := ['A', 'B', 'C']
  let move_sequence (cups : List Char) := 
    let step1 := [cups[1], cups[0], cups[2]]
    let step2 := [step1[0], step1[2], step1[1]]
    [step2[2], step2[1], step2[0]] -- Final swap of the 3-step move sequence
  let repeated_moves (times : Nat) (cups : List Char) := 
    Nat.iterate move_sequence times cups
  repeated_moves 9 ['A', 'B', 'C'] = ['A', 'C', 'B'] := sorry

end final_cup_order_l452_452072


namespace negation_of_proposition_l452_452116

theorem negation_of_proposition (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a = 1 → a + b = 1)) ↔ (∃ a b : ℝ, a = 1 ∧ a + b ≠ 1) :=
by
  sorry

end negation_of_proposition_l452_452116


namespace exists_m_divisible_l452_452989

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x + 2

-- Define the 100th iterate of f
def f_iter (n : ℕ) : ℕ := 3^n

-- Define the condition that needs to be proven
theorem exists_m_divisible : ∃ m : ℕ, (3^100 * m + (3^100 - 1)) % 1988 = 0 :=
sorry

end exists_m_divisible_l452_452989


namespace blaine_fish_caught_l452_452341

theorem blaine_fish_caught (B : ℕ) (cond1 : B + 2 * B = 15) : B = 5 := by 
  sorry

end blaine_fish_caught_l452_452341


namespace side_length_of_smaller_square_l452_452898

theorem side_length_of_smaller_square (s : ℝ) (A1 A2 : ℝ) (h1 : 5 * 5 = A1 + A2) (h2 : 2 * A2 = A1 + 25)  : s = 5 * Real.sqrt 3 / 3 :=
by
  sorry

end side_length_of_smaller_square_l452_452898


namespace perpendicular_slope_l452_452158

theorem perpendicular_slope (a b c : ℝ) (h : 4 * a - 6 * b = c) :
  let m := - (3 / 2) in
  ∃ k : ℝ, (k = m) :=
by
  sorry

end perpendicular_slope_l452_452158


namespace part1_part2_l452_452215

variables (α β : ℝ)

-- Conditions
axiom h1 : π < α ∧ α < 3 * π / 2
axiom h2 : π < β ∧ β < 3 * π / 2
axiom h3 : Real.sin α = - Real.sqrt 5 / 5
axiom h4 : Real.cos β = - Real.sqrt 10 / 10

-- Questions to be answered
noncomputable def α_minus_β : ℝ :=
  let expr := Real.acos (Real.sqrt 2 / 2) in -π/4

noncomputable def tan_2α_minus_β : ℝ := 
  -1/3

theorem part1 : α - β = -π/4 := sorry

theorem part2 : Real.tan (2*α - β) = -1/3 := sorry

end part1_part2_l452_452215


namespace incircle_area_l452_452190

noncomputable def hyperbola : ℝ → ℝ → Prop := λ x y, x^2 - y^2 / 24 = 1

structure Points :=
(P : ℝ × ℝ) -- P (first quadrant point on hyperbola)
(F1 : ℝ × ℝ) -- left focus
(F2 : ℝ × ℝ) -- right focus
(ratio : ℝ)   -- ratio between distances

axiom conditions
  (P : ℝ × ℝ)
  (F1 : ℝ × ℝ)
  (F2 : ℝ × ℝ)
  (on_hyperbola : hyperbola P.1 P.2)
  (F1_focus : F1.1 = -5 ∧ F1.2 = 0) -- F1 as left focus of hyperbola
  (F2_focus : F2.1 = 5 ∧ F2.2 = 0)  -- F2 as right focus of hyperbola
  (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
  (distance_ratio : dist P F1 / dist P F2 = 5 / 4)

theorem incircle_area : ∀ P F1 F2, on_hyperbola → F1_focus → F2_focus → in_first_quadrant → distance_ratio → 
  let semi_perimeter := (dist P F1 + dist P F2 + dist F1 F2) / 2
  let area := (8 * sqrt 21) -- triangle area
  let r := 16 * sqrt 21 / (dist P F1 + dist P F2 + dist F1 F2) -- in-circle radius
  let incircle_area := π * r^2
  incircle_area = 48 * π / 7 := 
sorry

end incircle_area_l452_452190


namespace set_intersection_eq_l452_452369

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5}
def ComplementU (S : Set ℕ) : Set ℕ := U \ S

theorem set_intersection_eq : 
  A ∩ (ComplementU B) = {1, 3} := 
by
  sorry

end set_intersection_eq_l452_452369


namespace angle_in_third_quadrant_l452_452817

/-- 
Given that the terminal side of angle α is in the third quadrant,
prove that the terminal side of α/3 cannot be in the second quadrant.
-/
theorem angle_in_third_quadrant (α : ℝ) (k : ℤ)
  (h : π + 2 * k * π < α ∧ α < 3 / 2 * π + 2 * k * π) :
  ¬ (π / 2 < α / 3 ∧ α / 3 < π) :=
sorry

end angle_in_third_quadrant_l452_452817


namespace sqrt_pow_simplification_l452_452574

theorem sqrt_pow_simplification :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * (5 ^ (3 / 4)) :=
by
  sorry

end sqrt_pow_simplification_l452_452574


namespace outfit_combination_count_l452_452241

theorem outfit_combination_count (c : ℕ) (s p h sh : ℕ) (c_eq_6 : c = 6) (s_eq_c : s = c) (p_eq_c : p = c) (h_eq_c : h = c) (sh_eq_c : sh = c) :
  (c^4) - c = 1290 :=
by
  sorry

end outfit_combination_count_l452_452241


namespace transformation_matrix_correct_l452_452970

-- Definitions for the dilation and rotation matrices
def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 0], ![0, 2]]
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![0, -1], ![1, 0]]

-- The transformation matrix composed of dilation followed by rotation
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := rotation_matrix ⬝ dilation_matrix

-- The expected resulting matrix after applying both transformations
def expected_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![0, -2], ![2, 0]]

-- The theorem that needs to be proved
theorem transformation_matrix_correct :
  transformation_matrix = expected_matrix :=
by {
  -- Placeholder for actual proof
  sorry
}

end transformation_matrix_correct_l452_452970


namespace probability_even_diff_l452_452461

theorem probability_even_diff (x y : ℕ) (hx : x ≠ y) (hx_set : x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)) (hy_set : y ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)) :
  (∃ p : ℚ, p = 5 / 22 ∧ 
    (let xy_diff_even := xy - x - y mod 2 = 0 
     in (xy_diff_even --> True))) :=
sorry

end probability_even_diff_l452_452461


namespace remainder_x2023_plus_1_l452_452617

theorem remainder_x2023_plus_1:
  ∀ (R : Type) [CommRing R] (x : R),
  let P := x^2023 + 1
  let Q := x^12 - x^9 + x^6 - x^3 + 1
  let Rm := -x^15 + 1
  (P % Q) = Rm :=
by
  -- Definitions of P, Q, and Rm:
  let P := x^2023 + 1
  let Q := x^12 - x^9 + x^6 - x^3 + 1
  let Rm := -x^15 + 1
  -- Proof goes here
  sorry

end remainder_x2023_plus_1_l452_452617


namespace max_area_triangle_l452_452568

theorem max_area_triangle (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  ∃ m, (m = (1/2) * a * b) ∧ (m = (1/2) * a * m ⇔ b * m = a * a + b * b) :=
sorry

end max_area_triangle_l452_452568


namespace gcd_of_256_180_600_l452_452844

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l452_452844


namespace geometric_series_sum_l452_452853

theorem geometric_series_sum :
  let a := (1 / 4 : ℝ)
  let r := (-1 / 4 : ℝ)
  let n := 6
  (∑ k in finset.range n, a * r ^ k) = 4095 / 5120 :=
by
  sorry

end geometric_series_sum_l452_452853


namespace minimum_dot_product_l452_452418

variables {V : Type*} [inner_product_space ℝ V]

structure Semicircle (V : Type*) [inner_product_space ℝ V] :=
(O A B : V)
(middle_point : (O + O) = (A + B))
(diameter : dist A B = 4)

structure PointOnRadius (V : Type*) [inner_product_space ℝ V] (semicircle : Semicircle V) :=
(P C : V)
(on_radius : ∃ t ∈ (Icc 0 1), P = t • C + (1 - t) • semicircle.O)

noncomputable def minimum_value (s : Semicircle V) (p : PointOnRadius V s) : ℝ :=
  (∥p.P - s.A∥ + ∥p.P - s.B∥) * ∥p.P - p.C∥

theorem minimum_dot_product (s : Semicircle V) (p : PointOnRadius V s) :
  ∃ (P : V), minimum_value (s, P) = -2 :=
sorry

end minimum_dot_product_l452_452418


namespace circle_equation_l452_452420

theorem circle_equation (x y : ℝ) (h k r : ℝ) :
  h = 3 → k = -1 → r = 4 → (x - 3)^2 + (y + 1)^2 = 16 :=
by
  intros h_eq k_eq r_eq
  rw [h_eq, k_eq, r_eq]
  sorry

end circle_equation_l452_452420


namespace PQ_length_l452_452295

noncomputable def length_of_PQ : ℝ := 9

-- Definitions of the isosceles property for triangles PQR and QRS
def is_isosceles (a b c : ℝ) (angle : ℝ) : Prop :=
  a = b ∧ angle = 120

def triangle_PQR_isosceles : Prop := is_isosceles (length_of_PQ) (length_of_PQ) 10 120

def triangle_QRS_isosceles (a : ℝ) : Prop := is_isosceles a a 10 60

-- Given conditions
axiom PQR_isosceles : triangle_PQR_isosceles
axiom QRS_isosceles (a : ℝ) : triangle_QRS_isosceles a
axiom perimeter_QRS : 2 * a + 10 = 24
axiom perimeter_PQR : 2 * length_of_PQ + 10 = 28

-- Theorem to be proven
theorem PQ_length : length_of_PQ = 9 :=
by sorry

end PQ_length_l452_452295


namespace total_area_of_triangular_faces_l452_452123

theorem total_area_of_triangular_faces (base_edge lateral_edge : ℝ) 
  (hb : base_edge = 8) (hl : lateral_edge = 9) :
  let h := Math.sqrt (lateral_edge^2 - (base_edge / 2)^2) in
  let area_of_one_face := (1 / 2) * base_edge * h in
  let total_area := 4 * area_of_one_face in
  total_area = 16 * Math.sqrt 65 :=
by
  sorry

end total_area_of_triangular_faces_l452_452123


namespace add_decimals_l452_452079

theorem add_decimals :
  5.623 + 4.76 = 10.383 :=
by sorry

end add_decimals_l452_452079


namespace calc_expression_l452_452576

theorem calc_expression :
  10 - 9 + 8 * 7 + 6 - 5 * 4 / 2 + 3 - 1 = 55 :=
by
  -- Performing each step can be done manually
  have h1 : 8 * 7 = 56 := by norm_num,
  have h2 : 5 * 4 = 20 := by norm_num,
  have h3 : 20 / 2 = 10 := by norm_num,
  calc
    10 - 9 + 8 * 7 + 6 - 5 * 4 / 2 + 3 - 1
        = 10 - 9 + 56 + 6 - 10 + 3 - 1 : by rw [h1, h2, h3]
    ... = 1 + 56 + 6 - 10 + 3 - 1       : by norm_num
    ... = 1 + 62 - 10 + 3 - 1           : by norm_num
    ... = 1 + 52 + 3 - 1                : by norm_num
    ... = 1 + 55 - 1                    : by norm_num
    ... = 1 + 54                        : by norm_num
    ... = 55                            : by norm_num

end calc_expression_l452_452576


namespace distinct_solution_count_l452_452020

theorem distinct_solution_count
  (n : ℕ)
  (x y : ℕ)
  (h1 : x ≠ y)
  (h2 : x ≠ 2 * y)
  (h3 : y ≠ 2 * x)
  (h4 : x^2 - x * y + y^2 = n) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 12 ∧ ∀ (a b : ℕ), (a, b) ∈ pairs → a^2 - a * b + b^2 = n :=
sorry

end distinct_solution_count_l452_452020


namespace angle_sum_is_540_degrees_l452_452264

def ∠ := ℝ -- Define angle as a Real number degree

variables (FAD GBC BCE ADG CEF AFE DGB : ∠)

theorem angle_sum_is_540_degrees 
  (h : FAD + GBC + BCE + ADG + CEF + AFE + DGB = r) : r = 540 := 
sorry

end angle_sum_is_540_degrees_l452_452264


namespace point_D_is_in_fourth_quadrant_l452_452716

-- Define the Cartesian coordinate system and the fourth quadrant condition
def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (0, -3)
def C : ℝ × ℝ := (-2, -3)
def D : ℝ × ℝ := (Real.pi, -1)

-- The theorem stating which point is in the fourth quadrant
theorem point_D_is_in_fourth_quadrant : is_in_fourth_quadrant D := by
  -- The proof will follow here
  sorry

end point_D_is_in_fourth_quadrant_l452_452716


namespace additional_people_needed_correct_l452_452788

-- Conditions: A flight requires more than 15 people to depart.
def min_people_required : ℕ := 16

-- There are currently 9 people on the plane.
def currently_on_plane : ℕ := 9

-- The number of additional people needed to board before departure.
def additional_people_needed (min_people required currently_on_plane: ℕ) : ℕ := 
  min_people_required - currently_on_plane

theorem additional_people_needed_correct (min_people_required = 16) (currently_on_plane = 9) :
  additional_people_needed min_people_required currently_on_plane = 7 :=
by
  -- Proof goes here.
  sorry

end additional_people_needed_correct_l452_452788


namespace alex_buys_15_pounds_of_wheat_l452_452508

theorem alex_buys_15_pounds_of_wheat (w o : ℝ) (h1 : w + o = 30) (h2 : 72 * w + 36 * o = 1620) : w = 15 :=
by
  sorry

end alex_buys_15_pounds_of_wheat_l452_452508


namespace sqrt_of_16_eq_pm_4_l452_452815

theorem sqrt_of_16_eq_pm_4 (h₁ : 16 = 4^2) (h₂ : 16 = (-4)^2) : (∃ x, x = 4 ∧ ∃ y, y = -4 ∧ √16 = x ∨ √16 = y) :=
by
  sorry

end sqrt_of_16_eq_pm_4_l452_452815


namespace sqrt_eq_solution_l452_452946

theorem sqrt_eq_solution (z : ℝ) : z = (134 / 3) ↔ sqrt (10 + 3 * z) = 12 := 
by
  sorry

end sqrt_eq_solution_l452_452946


namespace cookies_left_l452_452768

theorem cookies_left (total_cookies : ℕ) (fraction_given : ℚ) (given_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 20)
  (h2 : fraction_given = 2/5)
  (h3 : given_cookies = fraction_given * total_cookies)
  (h4 : remaining_cookies = total_cookies - given_cookies) :
  remaining_cookies = 12 :=
by
  sorry

end cookies_left_l452_452768


namespace product_of_nonreal_roots_l452_452973

theorem product_of_nonreal_roots :
  let P := λ x: ℂ, x^4 - 6 * x^3 + 15 * x^2 - 20 * x - 396
  let nonreal_roots := {x : ℂ | P x = 0 ∧ x.im ≠ 0}
  let product_nonreal_roots := (∀ (a b : ℂ), a ∈ nonreal_roots → b ∈ nonreal_roots → a ≠ b → a * b)
  product_nonreal_roots = 4 + complex.sqrt 412 :=
sorry

end product_of_nonreal_roots_l452_452973


namespace grid_segments_divisible_by_4_l452_452553

-- Definition: square grid where each cell has a side length of 1
structure SquareGrid (n : ℕ) :=
  (segments : ℕ)

-- Condition: Function to calculate the total length of segments in the grid
def total_length {n : ℕ} (Q : SquareGrid n) : ℕ := Q.segments

-- Lean 4 statement: Prove that for any grid, the total length is divisible by 4
theorem grid_segments_divisible_by_4 {n : ℕ} (Q : SquareGrid n) :
  total_length Q % 4 = 0 :=
sorry

end grid_segments_divisible_by_4_l452_452553


namespace minimum_value_of_f_l452_452616

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 3)

theorem minimum_value_of_f :
  ∃ x > 3, ∀ y > 3, f(y) ≥ f(x) ∧ f(x) = 5 :=
sorry

end minimum_value_of_f_l452_452616


namespace ratio_of_waist_to_hem_l452_452921

theorem ratio_of_waist_to_hem
  (cuffs_length : ℕ)
  (hem_length : ℕ)
  (ruffles_length : ℕ)
  (num_ruffles : ℕ)
  (lace_cost_per_meter : ℕ)
  (total_spent : ℕ)
  (waist_length : ℕ) :
  cuffs_length = 50 →
  hem_length = 300 →
  ruffles_length = 20 →
  num_ruffles = 5 →
  lace_cost_per_meter = 6 →
  total_spent = 36 →
  waist_length = (total_spent / lace_cost_per_meter * 100) -
                (2 * cuffs_length + hem_length + num_ruffles * ruffles_length) →
  waist_length / hem_length = 1 / 3 :=
by
  sorry

end ratio_of_waist_to_hem_l452_452921


namespace value_of_a_l452_452085

theorem value_of_a (a k: ℝ) (h1: k = 44) (h2: ∃ x: ℝ, x = 4 ∧ a * x^2 + 3 * x - k = 0) : a = 2 :=
by
  obtain ⟨x, hx1, hx2⟩ := h2
  subst hx1
  rw [hx1, h1] at hx2
  sorry

end value_of_a_l452_452085


namespace sale_price_each_machine_l452_452896

variables (P : ℝ) (commission_first_100 commission_next_30 total_commission : ℝ)

-- Given conditions:
def commission_first_100 := 100 * 0.03 * P
def commission_next_30 := 30 * 0.04 * P
def total_commission := 42000

-- Theorem to prove:
theorem sale_price_each_machine : 
  commission_first_100 + commission_next_30 = total_commission → P = 10000 :=
by
  have h1 : commission_first_100 = 3 * P := by sorry
  have h2 : commission_next_30 = 1.2 * P := by sorry
  have h3 : 3 * P + 1.2 * P = 42000 := by sorry
  have h4 : 4.2 * P = 42000 := by sorry
  have h5 : P = 42000 / 4.2 := by sorry
  sorry

end sale_price_each_machine_l452_452896


namespace LindaCandiesLeft_l452_452762

variable (initialCandies : ℝ)
variable (candiesGiven : ℝ)

theorem LindaCandiesLeft (h1 : initialCandies = 34.0) (h2 : candiesGiven = 28.0) : initialCandies - candiesGiven = 6.0 := by
  sorry

end LindaCandiesLeft_l452_452762


namespace max_distance_to_line_l452_452291

theorem max_distance_to_line (θ m : ℝ) :
  let P := (Real.cos θ, Real.sin θ),
      line : ℝ → ℝ → Prop := λ x y, x - m * y + 3 * m - 4 = 0,
      D := (4, 3) in
  ∃ d : ℝ, d = 6 ∧ 
      (∀ x y, line x y → dist (x, y) P ≤ d) ∧ 
      ∃ x y, line x y ∧ dist (x, y) P = d :=
sorry

end max_distance_to_line_l452_452291


namespace smallest_common_multiple_of_9_and_6_l452_452491

theorem smallest_common_multiple_of_9_and_6 : 
  ∃ x : ℕ, (x > 0 ∧ x % 9 = 0 ∧ x % 6 = 0) ∧ 
           ∀ y : ℕ, (y > 0 ∧ y % 9 = 0 ∧ y % 6 = 0) → x ≤ y :=
begin
  use 18,
  split,
  { split,
    { exact nat.succ_pos 17, },
    { split,
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_right 9 6), },
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_left 9 6), } } },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    exact lcm.dvd_iff.1 (nat.dvd_of_mod_eq_zero hy3) }
end

end smallest_common_multiple_of_9_and_6_l452_452491


namespace find_P_coordinates_l452_452011

-- Definitions following from the conditions
def A : ℝ × ℝ := (4, -1)
def B : ℝ × ℝ := (3, 4)
def line_l (P : ℝ × ℝ) := 2 * P.1 - P.2 - 4 = 0

-- Theorem statement
theorem find_P_coordinates : 
  ∃ P : ℝ × ℝ, line_l P ∧ (∀ Q : ℝ × ℝ, line_l Q → 
  dist_diff A B P ≥ dist_diff A B Q) ∧ P = (5, 6) := 
sorry

-- Helper function to calculate the distance difference
noncomputable def dist_diff (A B P : ℝ × ℝ) : ℝ :=
  abs (dist A P - dist B P)

end find_P_coordinates_l452_452011


namespace gcd_256_180_600_l452_452836

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l452_452836


namespace gcd_256_180_600_l452_452840

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l452_452840


namespace probability_other_side_green_l452_452873

-- Definitions based on the conditions
def Card : Type := ℕ
def num_cards : ℕ := 8
def blue_blue : ℕ := 4
def blue_green : ℕ := 2
def green_green : ℕ := 2

def total_green_sides : ℕ := (green_green * 2) + blue_green
def green_opposite_green_side : ℕ := green_green * 2

theorem probability_other_side_green (h_total_green_sides : total_green_sides = 6)
(h_green_opposite_green_side : green_opposite_green_side = 4) :
  (green_opposite_green_side / total_green_sides : ℚ) = 2 / 3 := 
by
  sorry

end probability_other_side_green_l452_452873


namespace group_photo_arrangement_l452_452265

theorem group_photo_arrangement :
  let coach := 1
  let a := 1
  let b_and_c_adj := 2
  let b_and_d_not_adj := 3
  let total_arrangements := 72 in
  (coach = 2 ∧ a = 2 ∧ b_and_c_adj = 2 ∧ b_and_d_not_adj = 12) →
  total_arrangements = 72 :=
sorry

end group_photo_arrangement_l452_452265


namespace simplify_expression_l452_452744

theorem simplify_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h : a^4 + b^4 = a^2 + b^2) :
  (a / b + b / a - 1 / (a * b)) = 3 :=
  sorry

end simplify_expression_l452_452744


namespace sphere_cone_radius_ratio_l452_452550

-- Define the problem using given conditions and expected outcome.
theorem sphere_cone_radius_ratio (r R h : ℝ)
  (h1 : h = 2 * r)
  (h2 : (1/3) * π * R^2 * h = 3 * (4/3) * π * r^3) :
  r / R = 1 / Real.sqrt 6 :=
by
  sorry

end sphere_cone_radius_ratio_l452_452550


namespace remaining_amoeba_is_blue_l452_452892

-- Initial counts of amoebas
def n1 : ℕ := 47  -- number of red amoebas
def n2 : ℕ := 40  -- number of blue amoebas
def n3 : ℕ := 53  -- number of yellow amoebas

-- Invariant based on merging rules: parity of absolute differences
def epsilon_12 : ℕ := (abs (n1 - n2)) % 2
def epsilon_13 : ℕ := (abs (n1 - n3)) % 2
def epsilon_23 : ℕ := (abs (n2 - n3)) % 2

-- Since in the end, only one amoeba is left, and the parity differences indicate the final result
theorem remaining_amoeba_is_blue :
  epsilon_12 = 1 ∧ epsilon_13 = 0 ∧ epsilon_23 = 1 → 
  (n2 % 2 = 0) :=
by
  -- The exact proof is omitted here
  sorry

end remaining_amoeba_is_blue_l452_452892


namespace hyperbola_eccentricity_range_l452_452252

theorem hyperbola_eccentricity_range (a : ℝ) (ha : a > 1) : 
  let e := Real.sqrt (1 + 1 / a^2) in 
  1 < e ∧ e < Real.sqrt 2 :=
by
  let e := Real.sqrt (1 + 1 / a^2)
  have h1 : 0 < 1 / a^2 := sorry
  have h2 : 1 < 1 + 1 / a^2 := sorry
  exact ⟨h2, sorry⟩

end hyperbola_eccentricity_range_l452_452252


namespace max_intersection_points_l452_452763

-- Define the problem conditions and the theorem
theorem max_intersection_points :
  let lines : fin 200 -> Type :=
    λ n : fin 200, if (n + 1) % 5 = 0 then 'parallel'
                   else if (n + 1) % 5 = 1 then 'passes_through_point'
                   else 'distinct' in
  -- Constraint that lines with (n + 1) % 5 = 0 are parallel
  (∀ n m : fin 200, (n + 1) % 5 = 0 ∧ (m + 1) % 5 = 0 -> lines n = lines m) ∧
  -- Constraint that lines with (n + 1) % 5 = 1 pass through the given point B
  (∀ n : fin 200, (n + 1) % 5 = 1 -> lines n = 'passes_through_point') ∧
  -- All lines are distinct
  (∀ n m : fin 200, n ≠ m -> 'distinct') ->
  -- The maximum number of points of intersections
  ∃ M : ℕ, M = 18341 :=
sorry  -- Proof will be filled in subsequently.

end max_intersection_points_l452_452763


namespace prime_list_mean_l452_452957

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def prime_list_mean_proof : Prop :=
  let nums := [33, 37, 39, 41, 43] in
  let primes := filter is_prime nums in
  primes = [37, 41, 43] ∧
  (primes.sum : ℤ) / primes.length = 40.33

theorem prime_list_mean : prime_list_mean_proof :=
by
  sorry

end prime_list_mean_l452_452957


namespace minimum_percentage_increase_l452_452472

def set_S : Set Int := { -4, -1, 0, 6, 9 }

def is_prime (n : Int) : Prop := Nat.Prime (n.natAbs)  -- primes only make sense in natural numbers, so we use absolute value

-- For simplicity, defining smallest primes
def smallest_primes : List Int := [2, 3]

noncomputable def mean (s : Set Int) : Float :=
  (s.toList.sum.toFloat) / (s.card.toFloat)

def percentage_increase_in_mean (original_set new_set : Set Int) : Float :=
  let mean_original := mean original_set
  let mean_new := mean new_set
  ((mean_new - mean_original) / mean_original) * 100

theorem minimum_percentage_increase :
  let new_set := {2, 3, 0, 6, 9}
  percentage_increase_in_mean set_S new_set = 100 :=
by
  sorry

end minimum_percentage_increase_l452_452472


namespace find_lambda_l452_452211

variables {𝕍 : Type*} [inner_product_space ℝ 𝕍]

theorem find_lambda (a b : 𝕍) (λ : ℝ) (ha_eq_hb : ∥a∥ = ∥b∥)
  (cos_angle_ab : real.cos (inner_product_space.angle a b) = 1 / 3)
  (perpendicular : inner_product_space.inner (a + 2 • b) (a + λ • b) = 0) :
  λ = -5 / 7 :=
sorry

end find_lambda_l452_452211


namespace sin_half_pi_minus_2alpha_l452_452981

noncomputable def alpha : ℝ := sorry

lemma cos_alpha_eq : cos alpha = 1 / 3 := sorry

theorem sin_half_pi_minus_2alpha :
  sin (π / 2 - 2 * alpha) = -7 / 9 := 
by 
  have h := cos_alpha_eq
  sorry

end sin_half_pi_minus_2alpha_l452_452981


namespace smallest_common_multiple_of_9_and_6_l452_452494

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l452_452494


namespace hyperbola_eccentricity_proof_l452_452638

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  let C := { p : ℝ × ℝ | (p.1 - 3) ^ 2 + p.2 ^ 2 = 9 }
  ∀ (chord_length : ℝ), chord_length = 2 * Real.sqrt 5 →
    let asymptote := { p : ℝ × ℝ | b * p.1 + a * p.2 = 0 }
    ∀ (c : ℝ), c = Real.sqrt (a^2 + b^2) →
      let e := c / a in
      chord_length = 2 * Real.sqrt ((3 ^ 2 - (3 * b / Real.sqrt (b^2 + a^2)) ^ 2)) →
        e = 3 * Real.sqrt 5 / 5

theorem hyperbola_eccentricity_proof (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  hyperbola_eccentricity a b h_a h_b := by
  sorry

end hyperbola_eccentricity_proof_l452_452638


namespace total_meals_per_week_l452_452677

-- Definitions of the conditions
def meals_per_day_r1 : ℕ := 20
def meals_per_day_r2 : ℕ := 40
def meals_per_day_r3 : ℕ := 50
def days_per_week : ℕ := 7

-- The proof goal
theorem total_meals_per_week : 
  (meals_per_day_r1 * days_per_week) + 
  (meals_per_day_r2 * days_per_week) + 
  (meals_per_day_r3 * days_per_week) = 770 :=
by
  sorry

end total_meals_per_week_l452_452677


namespace hour_hand_ratio_l452_452876

-- Define the lengths of the hour, minute, and second hands.
variables (L_h L_m L_s : ℝ)

-- Define the areas swept by the hands in one minute.
def area_h := (1 / 2) * L_h^2 * (Real.pi / 360)
def area_m := (1 / 2) * L_m^2 * (Real.pi / 30)
def area_s := (1 / 2) * L_s^2 * (2 * Real.pi)

-- Assume that the areas swept by each hand are the same.
axiom areas_equal : area_h = area_m ∧ area_m = area_s

-- State the problem as a theorem to prove.
theorem hour_hand_ratio (L_h L_m L_s : ℝ) (h : area_h = area_m ∧ area_m = area_s) : (L_h / L_s) = 12 * Real.sqrt 5 :=
by 
  sorry -- Proof will be provided here

end hour_hand_ratio_l452_452876


namespace angle_of_inclination_l452_452948

theorem angle_of_inclination (t : ℝ) : 
  let l : ℝ → ℝ × ℝ := λ t, (-real.sqrt 3 * t, 1 + 3 * t)
  ∃ θ, θ = 2 * real.pi / 3 :=
by
  sorry

end angle_of_inclination_l452_452948


namespace min_packs_to_100_l452_452786

  /--
  Given that soda is sold in packs of 8, 14, and 28 cans, prove that the minimum number of packs needed to buy exactly 100 cans of soda is 5.
  -/
  theorem min_packs_to_100 : 
    ∃ (packs : List ℕ) (target : ℕ), packs = [8, 14, 28] ∧ target = 100 ∧
    min_packs packs target = 5 :=
  by
    sorry
  
end min_packs_to_100_l452_452786


namespace max_modulus_l452_452988

open Complex

theorem max_modulus (z : ℂ) (h : abs z = 1) : ∃ M, M = 6 ∧ ∀ w, abs (z - w) ≤ M :=
by
  use 6
  sorry

end max_modulus_l452_452988


namespace units_digit_difference_l452_452996

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n : ℕ, a (n + 1) = 2 * a n + 1

theorem units_digit_difference (a : ℕ → ℕ) (h : sequence a) :
  (a 2004 - a 2003) % 10 = 8 :=
sorry

end units_digit_difference_l452_452996


namespace f_at_2_eq_6_l452_452985

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^5 + b * x^3 + c * x + 8

theorem f_at_2_eq_6 (a b c : ℝ) (h : f a b c (-2) = 10) : f a b c 2 = 6 :=
by
  -- We outline the conditions 
  simp at h,
  -- Since information about the left side, with simplified equation
  have h_eq: -32 * a - 8 * b - 2 * c + 8 = 10,
    sorry,
  -- Thus, we can simplify it as 32a + 8b + 2c = -2
  sorry,
  -- And substituting into f(2), we get the final simplification
  exact 32 * a + 8 * b + 2 * c + 8,
  -- Concluding the theorem.
  sorry

end f_at_2_eq_6_l452_452985


namespace total_missed_questions_l452_452772

-- Definitions
def missed_by_you : ℕ := 36
def missed_by_friend : ℕ := 7
def missed_by_you_friends : ℕ := missed_by_you + missed_by_friend

-- Theorem
theorem total_missed_questions (h1 : missed_by_you = 5 * missed_by_friend) :
  missed_by_you_friends = 43 :=
by
  sorry

end total_missed_questions_l452_452772


namespace smallest_multiple_of_9_and_6_is_18_l452_452483

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l452_452483


namespace product_of_midpoint_is_minus_4_l452_452479

-- Coordinates of the endpoints
def endpoint1 : ℝ × ℝ := (4, -3)
def endpoint2 : ℝ × ℝ := (-8, 7)

-- Function to compute the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Coordinates of the midpoint
def midpoint_coords := midpoint endpoint1 endpoint2

-- Product of the coordinates of the midpoint
def product_of_midpoint_coords (mp : ℝ × ℝ) : ℝ :=
  mp.1 * mp.2

-- Statement of the theorem to be proven
theorem product_of_midpoint_is_minus_4 : 
  product_of_midpoint_coords midpoint_coords = -4 := 
by
  sorry

end product_of_midpoint_is_minus_4_l452_452479


namespace Eunice_seed_count_l452_452603

variable (pots : ℕ) (seedsFirst : ℕ) (seedsFourth : ℕ)

def totalSeeds (seedsFirst seedsFourth : ℕ) : ℕ :=
  seedsFirst + seedsFourth

theorem Eunice_seed_count (pots : ℕ) (seedsFirst : ℕ) (seedsFourth : ℕ) (h_pots : pots = 4) (h_seedsFirst : seedsFirst = 3) (h_seedsFourth : seedsFourth = 1) :
  totalSeeds seedsFirst seedsFourth = 4 := by
  rw [totalSeeds, h_seedsFirst, h_seedsFourth]
  simp
  sorry

end Eunice_seed_count_l452_452603


namespace valid_friendship_configurations_l452_452564

-- Define the set of individuals
inductive Person 
| Alice | Bob | Cindy | Dave | Emma
deriving DecidableEq

-- Define friendship relation
def is_friend (p1 p2 : Person) : Prop := sorry

-- Define the condition of having exactly 2 friends
def has_two_friends (p : Person) : Prop := sorry

-- Total number of valid configurations where each person has exactly 2 friends forming a cycle
theorem valid_friendship_configurations : 
    (∀ p : Person, has_two_friends p ∧ dist_pairwise (λ p1 p2 : Person, is_friend p1 p2)) → 
    num_valid_configurations = 12 := 
by
    sorry

end valid_friendship_configurations_l452_452564


namespace cells_at_day_10_l452_452545

-- Define a function to compute the number of cells given initial cells, tripling rate, intervals, and total time.
def number_of_cells (initial_cells : ℕ) (ratio : ℕ) (interval : ℕ) (total_time : ℕ) : ℕ :=
  let n := total_time / interval + 1
  initial_cells * ratio^(n-1)

-- State the main theorem
theorem cells_at_day_10 :
  number_of_cells 5 3 2 10 = 1215 := by
  sorry

end cells_at_day_10_l452_452545


namespace water_left_after_four_hours_l452_452684

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def water_added_hour3 : ℕ := 1
def water_added_hour4 : ℕ := 3

theorem water_left_after_four_hours :
    initial_water - 2 * 2 - 2 + water_added_hour3 - 2 + water_added_hour4 - 2 = 36 := 
by
    sorry

end water_left_after_four_hours_l452_452684


namespace strictly_increasing_interval_l452_452218

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 
  √3 * Real.sin (π - ω * x) - Real.sin ((5 * π / 2) + ω * x)

variable (α β : ℝ)
variable (ω : ℝ)

theorem strictly_increasing_interval :
  (f α ω = 2) →
  (f β ω = 0) →
  (|α - β| = π / 2) →
  (ω = 1) →
  ∀ k : ℤ,
  ∃ I : Set ℝ, I = Set.Icc (2 * k * π - π / 3) (2 * k * π + 2 * π / 3) ∧ StrictMonoOn (λ x, f x ω) I :=
sorry

end strictly_increasing_interval_l452_452218


namespace sum_of_roots_l452_452162

theorem sum_of_roots (x : ℝ) :
  ∑ x in {x | 2 * x ^ 2 + 2006 * x = 2007}, x = -1003 := 
sorry

end sum_of_roots_l452_452162


namespace chess_group_games_l452_452037

theorem chess_group_games (n : ℕ) (h : n = 9) : (n.choose 2 = 36) :=
by
  rw h
  simp
  exact dec_trivial

end chess_group_games_l452_452037


namespace rectangular_solid_surface_area_l452_452893

theorem rectangular_solid_surface_area 
  (a b c : ℕ) 
  (ha : nat.prime a) 
  (hb : nat.prime b) 
  (hc : nat.prime c) 
  (vol : a * b * c = 221) : 
  2 * (a * b + b * c + c * a) = 502 := 
by 
  sorry

end rectangular_solid_surface_area_l452_452893


namespace min_digits_right_of_decimal_l452_452849

theorem min_digits_right_of_decimal 
  (n : ℤ) (k1 k2 : ℕ) (h_eq : n = 987654321) 
  (h_k1 : k1 = 30) (h_k2 : k2 = 6) 
  (h_fract : n / (2^k1 * 5^k2)) :
  minimum_digits_right_of_decimal n k1 k2 = 30 :=
sorry

end min_digits_right_of_decimal_l452_452849


namespace bisection_of_arc_by_line_CD_l452_452018

theorem bisection_of_arc_by_line_CD
  (A B C D E F : Point) 
  (γ1 γ2 : Circle)
  (h1 : Center γ1 = A)
  (h2 : Center γ2 = B)
  (h3 : Intersect γ1 γ2 = {C, D})
  (circ_ABC : Circle) 
  (h4 : Circumcircle (Triangle.mk A B C) = circ_ABC)
  (h5 : Intersect circ_ABC γ1 = {C, E})
  (h6 : Intersect circ_ABC γ2 = {C, F})
  (h7 : Arc EF circ_ABC ∉ γ1 ∧ Arc EF circ_ABC ∉ γ2 ∧ C ∉ Arc EF circ_ABC) :
  BisectionLine CD (Arc EF circ_ABC) :=
sorry

end bisection_of_arc_by_line_CD_l452_452018


namespace speed_in_still_water_l452_452886

/-- 
 The speed of a man in still water, given upstream and downstream speeds.
 -/
theorem speed_in_still_water (U D : ℝ) (hU : U = 27) (hD : D = 35) : 
  let S := (U + D) / 2 in S = 31 :=
by
  sorry

end speed_in_still_water_l452_452886


namespace inequality_proof_l452_452977

theorem inequality_proof (x y z : ℝ) (h : x * y * z + x + y + z = 4) : 
    (y * z + 6)^2 + (z * x + 6)^2 + (x * y + 6)^2 ≥ 8 * (x * y * z + 5) :=
by
  sorry

end inequality_proof_l452_452977


namespace books_in_spanish_l452_452339

noncomputable def total_books (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 300 ∧
  ∃ e f i s : ℕ, 
    e = n / 5 ∧ f = n / 7 ∧ i = n / 4 ∧
    s = n - (e + f + i) ∧ s = 114

theorem books_in_spanish (n : ℕ) (h : total_books n) : 
  ∃ s : ℕ, s = n - (n / 5 + n / 7 + n / 4) ∧ s = 114 :=
begin
  sorry
end

end books_in_spanish_l452_452339


namespace gale_favorite_ones_digits_count_l452_452589

theorem gale_favorite_ones_digits_count :
  {d : Fin 10 | ∃ (n : ℕ), n % 12 = 0 ∧ n % 10 = d}.card = 5 :=
sorry

end gale_favorite_ones_digits_count_l452_452589


namespace unit_vector_AB_l452_452199

def point := ℝ × ℝ

def vector_sub (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def magnitude (v : point) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def unit_vector (v : point) : point := (v.1 / magnitude v, v.2 / magnitude v)

def A : point := (1, 3)
def B : point := (4, -1)

def AB : point := vector_sub A B

theorem unit_vector_AB : unit_vector AB = (3/5, -4/5) := sorry

end unit_vector_AB_l452_452199


namespace Dan_found_money_l452_452586

theorem Dan_found_money (cost_of_snake_toy : ℝ) (cost_of_cage : ℝ) (total_cost : ℝ) :
  cost_of_snake_toy = 11.76 ∧ cost_of_cage = 14.54 ∧ total_cost = 26.3 →
  (total_cost - cost_of_snake_toy - cost_of_cage = 0) :=
by
  intro h
  cases h with hs h
  cases h with hc ht
  simp [hs, hc, ht]
  sorry

end Dan_found_money_l452_452586


namespace meeting_time_of_Lata_and_Geeta_l452_452861

theorem meeting_time_of_Lata_and_Geeta:
  ∀ (circumference : ℝ) (lata_speed_kmh geeta_speed_kmh : ℝ),
  circumference = 640 →
  lata_speed_kmh = 4.2 →
  geeta_speed_kmh = 3.8 →
  (lata_speed_kmh * 1000 / 60 + geeta_speed_kmh * 1000 / 60) / circumference = 1 / 4.8 :=
begin
  intros circumference lata_speed_kmh geeta_speed_kmh h_circ h_lata h_geeta,
  sorry
end

end meeting_time_of_Lata_and_Geeta_l452_452861


namespace smaller_acute_angle_l452_452712

theorem smaller_acute_angle (x : ℝ) (h : 5 * x + 4 * x = 90) : 4 * x = 40 :=
by 
  -- proof steps can be added here, but are omitted as per the instructions
  sorry

end smaller_acute_angle_l452_452712


namespace eulers_formula_convex_polyhedron_l452_452131

theorem eulers_formula_convex_polyhedron :
  ∀ (V E F T H : ℕ),
  (V - E + F = 2) →
  (F = 24) →
  (E = (3 * T + 6 * H) / 2) →
  100 * H + 10 * T + V = 240 :=
by
  intros V E F T H h1 h2 h3
  sorry

end eulers_formula_convex_polyhedron_l452_452131


namespace reinforcement_arrival_l452_452859

def provisions (men days : ℕ) : ℕ := men * days

theorem reinforcement_arrival (R : ℕ) :
  let initial_men := 1850
  let initial_days := 28
  let days_elapsed := 12
  let days_remaining_after_elapsed := initial_days - days_elapsed
  let days_remaining_after_reinforcement := 10
  provisions initial_men initial_days = provisions initial_men days_remaining_after_elapsed →
  provisions initial_men days_remaining_after_elapsed = provisions (initial_men + R) days_remaining_after_reinforcement →
  R = 1110 :=
by
  intros
  rw [provisions, provisions] at a,
  exact sorry

end reinforcement_arrival_l452_452859


namespace multiply_reciprocal_fractions_l452_452021

theorem multiply_reciprocal_fractions :
  12 * (1 / 3 + 1 / 6 + 1 / 4)⁻¹ = 16 :=
by
  sorry

end multiply_reciprocal_fractions_l452_452021


namespace area_of_circle_2pi_distance_AB_sqrt6_l452_452301

/- Definition of the circle in polar coordinates -/
def circle_polar := ∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

/- Definition of the line in polar coordinates -/
def line_polar := ∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0

/- The area of the circle -/
theorem area_of_circle_2pi : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  ∃ A : ℝ, A = 2 * Real.pi :=
by
  intro h
  sorry

/- The distance between two intersection points A and B -/
theorem distance_AB_sqrt6 : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  (∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0) → 
  ∃ d : ℝ, d = Real.sqrt 6 :=
by
  intros h1 h2
  sorry

end area_of_circle_2pi_distance_AB_sqrt6_l452_452301


namespace age_sum_l452_452406

variable {S R K : ℝ}

theorem age_sum 
  (h1 : S = R + 10)
  (h2 : S + 12 = 3 * (R - 5))
  (h3 : K = R / 2) :
  S + R + K = 56.25 := 
by 
  sorry

end age_sum_l452_452406


namespace probability_sum_8_two_dice_l452_452850

theorem probability_sum_8_two_dice : 
  (let possible_outcomes := 36 in
   let favorable_outcomes := 5 in
   let probability := favorable_outcomes / possible_outcomes in
   probability = 5 / 36) :=
by sorry

end probability_sum_8_two_dice_l452_452850


namespace infinite_primes_dividing_sum_of_powers_l452_452357

theorem infinite_primes_dividing_sum_of_powers 
  (n : ℕ) (hn : n > 1) (a : Fin n → ℕ) 
  (h_distinct : Function.Injective a) 
  (h_coprime : ∀ i j, i ≠ j → Nat.coprime (a i) (a j)) :
  ∃ᶠ p in Filter.atTop, ∃ k : ℕ, p ∣ (a 0 ^ k + a 1 ^ k + ... + a (n-1) ^ k) := 
  sorry

end infinite_primes_dividing_sum_of_powers_l452_452357


namespace diagonals_in_nine_sided_polygon_l452_452063

-- Define the conditions
def sides : ℕ := 9
def right_angles : ℕ := 2

-- The function to calculate the number of diagonals for a polygon
def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The theorem to prove
theorem diagonals_in_nine_sided_polygon : number_of_diagonals sides = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l452_452063


namespace five_letter_word_with_at_least_one_consonant_l452_452236

def letter_set : Set Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Set Char := {'B', 'C', 'D', 'F'}
def vowels : Set Char := {'A', 'E'}

-- Calculate the total number of 5-letter words using the letter set
def total_words : ℕ := 6^5

-- Calculate the number of 5-letter words using only vowels
def vowel_only_words : ℕ := 2^5

-- Number of 5-letter words with at least one consonant
def words_with_consonant : ℕ := total_words - vowel_only_words

theorem five_letter_word_with_at_least_one_consonant :
  words_with_consonant = 7744 :=
by
  sorry

end five_letter_word_with_at_least_one_consonant_l452_452236


namespace cube_edge_length_l452_452562

-- Conditions as definitions
def has_edge_length (n : ℕ) := n > 2

def top_bottom_faces_painted (n : ℕ) := 
  let unpainted_cubes := (n - 2) * n^2 in
  let one_face_painted_cubes := 2 * (n - 2)^2 in
  2 * one_face_painted_cubes = unpainted_cubes

-- Theorem statement
theorem cube_edge_length (n : ℕ) (h1 : has_edge_length n) (h2 : top_bottom_faces_painted n) : 
  n = 3 :=
sorry

end cube_edge_length_l452_452562


namespace find_line_equation_l452_452149

def point := (-2, 2)
def area : ℝ := 1
def line1 : ℝ → (ℝ × ℝ) := λ x, (x, -2*x - 2)
def line2 : ℝ → (ℝ × ℝ) := λ x, (x, (2 - x) / 2)

theorem find_line_equation (l : ℝ → (ℝ × ℝ)) :
  ((point.1, point.2) ∈ set.range l) →
  (let t := (l 0).2 in (t / 2) = area) →
  (l = line1 ∨ l = line2) :=
begin
  sorry
end

end find_line_equation_l452_452149


namespace arithmetic_mean_of_primes_l452_452952

-- Define the list of numbers
def num_list : List ℕ := [33, 37, 39, 41, 43]

-- Define a predicate that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the list of prime numbers extracted from the list
def prime_list : List ℕ := (num_list.filter is_prime)

-- Define the sum of the prime numbers
def prime_sum : ℕ := prime_list.foldr (· + ·) 0

-- Define the count of prime numbers
def prime_count : ℕ := prime_list.length

-- Define the arithmetic mean of the prime numbers
def prime_mean : ℚ := prime_sum / prime_count

theorem arithmetic_mean_of_primes :
  prime_mean = 40 + 1 / 3 := sorry

end arithmetic_mean_of_primes_l452_452952


namespace probability_two_heads_with_second_tail_l452_452117

/-- The probability that Debra gets two heads in a row but sees a second tail before she sees a second head
    when repeatedly flipping a fair coin and stops flipping when she gets two heads in a row or two tails
    in a row is 1/24. --/
theorem probability_two_heads_with_second_tail :
  let coin := prob_choice (1 / 2) tt ff in
  let outcome := coin_repeatedly coin in
  let stop_condition := λ (s : list bool), (s.tail ++ [true, true] = s) ∨ (s.tail ++ [false, false] = s) in
  Pr (λ s, (stop_condition s) ∧ (λ s, s = [false, true, false])) = 1 / 24 := 
sorry

end probability_two_heads_with_second_tail_l452_452117


namespace total_children_with_cats_l452_452860

variable (D C B : ℕ)
variable (h1 : D = 18)
variable (h2 : B = 6)
variable (h3 : D + C + B = 30)

theorem total_children_with_cats : C + B = 12 := by
  sorry

end total_children_with_cats_l452_452860


namespace find_x_equals_21_over_4_l452_452625

def vector_proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let (vx, vy) := v
  let (wx, wy) := w
  let vw_dot := vx * wx + vy * wy
  let ww_dot := wx * wx + wy * wy
  (vw_dot / ww_dot * wx, vw_dot / ww_dot * wy)

theorem find_x_equals_21_over_4
  (x : ℝ)
  (v : ℝ × ℝ := (x, 4))
  (w : ℝ × ℝ := (8, -2))
  (proj_v_w : ℝ × ℝ := vector_proj v w) :
  proj_v_w = (4, -1) → x = 21 / 4 :=
by
  intro h
  sorry

end find_x_equals_21_over_4_l452_452625


namespace carmen_candles_needed_l452_452104

-- Definitions based on the conditions

def candle_lifespan_1_hour : Nat := 8  -- a candle lasts 8 nights when burned 1 hour each night
def nights_total : Nat := 24  -- total nights

-- Question: How many candles are needed if burned 2 hours a night?

theorem carmen_candles_needed (h : candle_lifespan_1_hour / 2 = 4) :
  nights_total / 4 = 6 := 
  sorry

end carmen_candles_needed_l452_452104


namespace red_or_black_prob_red_black_or_white_prob_l452_452048

-- Defining the probabilities
def prob_red : ℚ := 5 / 12
def prob_black : ℚ := 4 / 12
def prob_white : ℚ := 2 / 12
def prob_green : ℚ := 1 / 12

-- Question 1: Probability of drawing a red or black ball
theorem red_or_black_prob : prob_red + prob_black = 3 / 4 :=
by sorry

-- Question 2: Probability of drawing a red, black, or white ball
theorem red_black_or_white_prob : prob_red + prob_black + prob_white = 11 / 12 :=
by sorry

end red_or_black_prob_red_black_or_white_prob_l452_452048


namespace quadratic_root_probability_f_one_gt_zero_prob_l452_452219

def quadratic_has_root_probability (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 4) (hb : 0 ≤ b ∧ b ≤ 4) : Prop :=
  a^2 ≥ 4 * b

theorem quadratic_root_probability : ∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 
  quadratic_has_root_probability a b → (12 : ℚ) / 25 = 0.48 := 
sorry

def f_at_one_gt_zero (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 4) (hb : 0 ≤ b ∧ b ≤ 4) : Prop :=
  a - b > 1

theorem f_one_gt_zero_prob : ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 
  f_at_one_gt_zero a b → (9 : ℚ) / 32 = 0.28125 := 
sorry

end quadratic_root_probability_f_one_gt_zero_prob_l452_452219


namespace relatively_prime_days_in_june_l452_452548

def relatively_prime_to (m : ℕ) (n : ℕ) : Prop :=
  Nat.gcd m n = 1

theorem relatively_prime_days_in_june : 
  (finset.filter (λ d : ℕ, relatively_prime_to 6 d) (finset.range 31)).card = 10 :=
by sorry

end relatively_prime_days_in_june_l452_452548


namespace ellipse_properties_slope_constant_l452_452646

variables (a b : ℝ)
def ellipse_eq (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

variables (F1 F2 P Q A B : ℝ × ℝ)
def F1_coord := F1 = (-2, 0)
def perpendicular_through_F1 := ∃ y1 y2, ellipse_eq (-2) y1 ∧ ellipse_eq (-2) y2 ∧ F1 = (-2, 0) ∧ P = (-2, y1) ∧ Q = (-2, y2)
def line_PF2 := ∃ m b1, (P = (-2, b1)) ∧ (m = -3 / 4) ∧ line_eq P F2 (0, 3/2)
def points_on_ellipse := ellipse_eq (prod.fst A) (prod.snd A) ∧ ellipse_eq (prod.fst B) (prod.snd B)
def angle_equal := ∠APQ = ∠BPQ

theorem ellipse_properties :
  F1_coord ∧ perpendicular_through_F1 ∧ line_PF2 ∧ points_on_ellipse
  → ∃ e : ℝ, e = 1 / 2 ∧ (ellipse_eq 16 12) :=
sorry

theorem slope_constant (h : angle_equal) :
  F1_coord ∧ perpendicular_through_F1 ∧ line_PF2 ∧ points_on_ellipse
  → ∃ K : ℝ, K = -1 / 2 :=
sorry

end ellipse_properties_slope_constant_l452_452646


namespace chromium_percentage_in_second_alloy_l452_452279

theorem chromium_percentage_in_second_alloy : 
  ∀ (x : ℝ),
    let alloy1_chromium := 12 / 100
    let alloy2_chromium := x / 100
    let mixed_weight := 15 + 35 
    let mixed_chromium_percentage := 9.2 / 100 
    let chromium_content_alloy1 := (alloy1_chromium * 15)
    let chromium_content_alloy2 := (alloy2_chromium * 35)
    let total_chromium := mixed_chromium_percentage * mixed_weight in
    chromium_content_alloy1 + chromium_content_alloy2 = total_chromium → x = 8 :=
by
  intro x alloy1_chromium alloy2_chromium mixed_weight mixed_chromium_percentage chromium_content_alloy1 chromium_content_alloy2 total_chromium h
  sorry

end chromium_percentage_in_second_alloy_l452_452279


namespace calculate_value_is_neg_seventeen_l452_452919

theorem calculate_value_is_neg_seventeen : -3^2 + (-2)^3 = -17 :=
by
  sorry

end calculate_value_is_neg_seventeen_l452_452919


namespace log_base_change_l452_452983

theorem log_base_change (p q r : ℝ)
  (h1 : Real.logBase 8 3 = p)
  (h2 : Real.logBase 3 5 = q)
  (h3 : Real.logBase 4 7 = r) :
  Real.logBase 10 7 = (2 * r) / (1 + 4 * q * p) :=
begin
  sorry -- Proof not required as per the instructions
end

end log_base_change_l452_452983


namespace circumscribe_hexagon_l452_452778

structure Hexagon (Point : Type) :=
(A B C D E F : Point)
(opposite_sides_parallel : (A ≠ B → C ≠ D → parallel (A, B) (D, C)) ∧ 
                           (C ≠ D → E ≠ F → parallel (C, D) (F, E)) ∧ 
                           (E ≠ F → A ≠ B → parallel (E, F) (B, A)))
(equal_opposite_diagonals : dist A D = dist B E ∧ 
                            dist B E = dist C F ∧ 
                            dist C F = dist D A ∧ 
                            dist D A = dist E B ∧ 
                            dist E B = dist F C ∧ 
                            dist F C = dist A D)

theorem circumscribe_hexagon {Point : Type} [MetricSpace Point] :
  ∀ (hex : Hexagon Point), ∃ (O : Point) (r : ℝ), ∀ v ∈ {hex.A, hex.B, hex.C, hex.D, hex.E, hex.F}, dist O v = r := 
by 
  intros hex
  sorry

end circumscribe_hexagon_l452_452778


namespace number_of_solutions_abs_eq_a_l452_452686

theorem number_of_solutions_abs_eq_a (a : ℝ) :
  (0 = {x : ℝ | abs (x + 2) - abs (2 * x + 8) = a}.to_finset.card) ∧ (a > 2) ∨
  (1 = {x : ℝ | abs (x + 2) - abs (2 * x + 8) = a}.to_finset.card) ∧ (a = 2) ∨
  (2 = {x : ℝ | abs (x + 2) - abs (2 * x + 8) = a}.to_finset.card) ∧ (a < 2) :=
sorry

end number_of_solutions_abs_eq_a_l452_452686


namespace parallel_lines_k_tangent_line_k_l452_452228

-- Define the lines and circle
def line_l (k : ℝ) : ℝ → ℝ → Prop := λ x y, k * x - y + 1 = 0
def line_parallel (k : ℝ) : ℝ → ℝ → Prop := λ x y, x - k * y + 2 = 0
def circle_C : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2 * x = 0

-- Define the center and radius of the circle
def center_C : ℝ × ℝ := (1, 0)
def radius_C : ℝ := 1

-- Prove (1): If the line is parallel, then k = ±1
theorem parallel_lines_k (k : ℝ) (h : ∀ x y, line_l k x y → line_parallel k x y) :
  k = 1 ∨ k = -1 :=
sorry

-- Prove (2): If the line is tangent to the circle, then k = 1
theorem tangent_line_k (k : ℝ) (h : ∀ x y (hx : circle_C center_C.fst center_C.snd), 
                            line_l k x y ∧ (center_C.fst, center_C.snd, radius_C) ∈ set_of (λ p, k * p.fst - p.snd + 1 = radius_C)) :
    k = 1 :=
sorry

end parallel_lines_k_tangent_line_k_l452_452228


namespace coefficient_of_x9_is_84_l452_452612

noncomputable def coefficient_of_x9 : ℚ :=
  let trinomial := (x^3 - (x^2 / 2) + (2 / x)) 
  let expansion := trinomial^9 
  coeff expansion 9

theorem coefficient_of_x9_is_84 : coefficient_of_x9 = 84 := 
by
  sorry

end coefficient_of_x9_is_84_l452_452612


namespace divisibility_of_powers_l452_452654

theorem divisibility_of_powers (a b c d m : ℤ) (h_odd : m % 2 = 1)
  (h_sum_div : m ∣ (a + b + c + d))
  (h_sum_squares_div : m ∣ (a^2 + b^2 + c^2 + d^2)) : 
  m ∣ (a^4 + b^4 + c^4 + d^4 + 4 * a * b * c * d) :=
sorry

end divisibility_of_powers_l452_452654


namespace correct_percentage_changes_l452_452060

noncomputable theory

-- Define the response rate calculation
def response_rate (responses : ℕ) (customers : ℕ) : ℝ :=
  (responses.toReal / customers.toReal) * 100

-- Define the percentage change calculation
def percentage_change (old_rate new_rate : ℝ) : ℝ :=
  ((new_rate - old_rate) / old_rate) * 100

-- Conditions
def responses_A := 15
def customers_A := 100
def responses_B := 27
def customers_B := 120
def responses_C := 39
def customers_C := 140
def responses_D := 56
def customers_D := 160

-- Response rates
def rate_A := response_rate responses_A customers_A
def rate_B := response_rate responses_B customers_B
def rate_C := response_rate responses_C customers_C
def rate_D := response_rate responses_D customers_D

-- Percentage changes
def change_A_to_B := percentage_change rate_A rate_B
def change_A_to_C := percentage_change rate_A rate_C
def change_A_to_D := percentage_change rate_A rate_D

-- Theorem to prove the correctness of the required calculations
theorem correct_percentage_changes :
  change_A_to_B = 50 ∧
  change_A_to_C ≈ 85.71 ∧
  change_A_to_D ≈ 133.33 ∧
  change_A_to_D = max change_A_to_B (max change_A_to_C change_A_to_D) :=
by sorry

end correct_percentage_changes_l452_452060


namespace sequence_bound_l452_452734

noncomputable def B_exists (a : ℕ → ℝ) (A : ℝ) (B : ℝ) : Prop :=
∃ B > 0, ∀ n ≥ 1, a n ≤ B / n

theorem sequence_bound (a : ℕ → ℝ) (A : ℝ) (h1 : ∀ n, a n ≥ 0)
                      (h2 : ∀ m, a m - a (m + 1) ≥ A * (a m) ^ 2) :
  ∃ B > 0, ∀ n ≥ 1, a n ≤ B / n :=
begin
  sorry
end

end sequence_bound_l452_452734


namespace smallest_common_multiple_of_9_and_6_l452_452496

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l452_452496


namespace snow_first_day_eq_six_l452_452571

variable (snow_first_day snow_second_day snow_fourth_day snow_fifth_day : ℤ)

theorem snow_first_day_eq_six
  (h1 : snow_second_day = snow_first_day + 8)
  (h2 : snow_fourth_day = snow_second_day - 2)
  (h3 : snow_fifth_day = snow_fourth_day + 2 * snow_first_day)
  (h4 : snow_fifth_day = 24) :
  snow_first_day = 6 := by
  sorry

end snow_first_day_eq_six_l452_452571


namespace max_value_of_m_l452_452359

variable (n : ℕ) (a : ℕ → ℝ)
hypothesis (h_n_geq_3 : n ≥ 3)
hypothesis (h_sum_squares : ∑ i in finset.range n, (a i) ^ 2 = 1)
hypothesis (h_m_defined : ∃ m, ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → |a i - a j| ≥ m)

theorem max_value_of_m : 
  ∃ m, m = sqrt (12 / (n * (n - 1) * (n ^ 2 - 1))) ∧ 
       ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → |a i - a j| ≥ m := 
by
  sorry

end max_value_of_m_l452_452359


namespace coefficient_x2_sum_expansion_l452_452098

theorem coefficient_x2_sum_expansion :
  (finset.sum (finset.range 10) (λ n, nat.choose n 2)) = 120 :=
by sorry

end coefficient_x2_sum_expansion_l452_452098


namespace quadratic_root_diff_l452_452594

noncomputable def delta (a b c : ℚ) : ℚ := b^2 - 4 * a * c

noncomputable def root_diff (a b : ℚ) (Δ : ℚ) : ℚ :=
  abs ((b + sqrt Δ) / (2 * a) - (b - sqrt Δ) / (2 * a))

theorem quadratic_root_diff:
  ∀ (a b c : ℚ),
  a = 5 → b = -9 → c = 1 →
  let Δ := delta a b c in
  Δ = 61 →
  let diff := root_diff a b Δ in
  ∃ p q : ℚ, diff = (sqrt p) / q ∧ p + q = 66 :=
by
  intros
  sorry

end quadratic_root_diff_l452_452594


namespace inequality_and_equality_conditions_l452_452990

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ abc ≤ 1 ∧ ((a + b + c = 3) → (a = 1 ∧ b = 1 ∧ c = 1)) := 
by 
  sorry

end inequality_and_equality_conditions_l452_452990


namespace find_smallest_number_l452_452436

theorem find_smallest_number (a b c : ℚ) (h₁ : a + b + c = 73) (h₂ : c - b = 5) (h₃ : b - a = 6) : a = 56 / 3 :=
begin
  sorry
end

end find_smallest_number_l452_452436


namespace Dan_team_lost_games_l452_452587

/-- Dan's high school played eighteen baseball games this year.
Two were at night and they won 15 games. Prove that they lost 3 games. -/
theorem Dan_team_lost_games (total_games won_games : ℕ) (h_total : total_games = 18) (h_won : won_games = 15) :
  total_games - won_games = 3 :=
by {
  sorry
}

end Dan_team_lost_games_l452_452587


namespace domain_of_g_l452_452148

noncomputable def g (x : ℝ) : ℝ := real.sqrt (4 - real.sqrt (6 - real.sqrt (7 - x)))

theorem domain_of_g : {x : ℝ | x ∈ set.Icc (-29) 7} = {x : ℝ | 0 <= 4 - real.sqrt (6 - real.sqrt (7 - x))} :=
sorry

end domain_of_g_l452_452148


namespace min_time_for_keys_return_l452_452777

theorem min_time_for_keys_return (v_pedestrian v_cyclist_road v_cyclist_alley: ℝ) 
    (distance_AB : ℝ) (time_walked : ℝ) (pi : ℝ) : 
    v_pedestrian = 7 ∧ v_cyclist_road = 15 ∧ v_cyclist_alley = 20 ∧ distance_AB = 4 ∧ time_walked = 1 
    → let d_pedestrian := v_pedestrian * time_walked,
          circumference := pi * distance_AB,
          distance_on_alley := circumference - d_pedestrian,
          t_highway := distance_AB / v_cyclist_road,
          t_alley := distance_on_alley / (v_cyclist_alley + v_pedestrian)
      in (t_highway + t_alley) = (4 / 15) + (4 * pi - 7) / 27 :=
begin
    intros h,
    cases h with h1 h_rest,
    cases h_rest with h2 h_rest,
    cases h_rest with h3 h_rest,
    cases h_rest with h4 h5,
    sorry
end

end min_time_for_keys_return_l452_452777


namespace eggs_collected_l452_452602

def total_eggs_collected (b1 e1 b2 e2 : ℕ) : ℕ :=
  b1 * e1 + b2 * e2

theorem eggs_collected :
  total_eggs_collected 450 36 405 42 = 33210 :=
by
  sorry

end eggs_collected_l452_452602


namespace eccentricity_of_hyperbola_l452_452645

variables {a b c : ℝ}
variable (h : ℝ)

-- Condition definitions
def ellipse_eccentricity : ℝ := (Real.sqrt 6) / 3
def hyperbola_eccentricity (b a : ℝ) : ℝ := Real.sqrt (1 + (b / a) ^ 2)
def is_perpendicular (x y : ℝ) : Prop := x * y = 0

-- Main theorem statement
theorem eccentricity_of_hyperbola (h : ℝ) (a b c : ℝ)
    (h1 : h = c / 2)
    (h2 : ellipse_eccentricity = Real.sqrt 6 / 3)
    (h3 : is_perpendicular (sqrt (a^2 - b^2)) c)
    (h4 : hyperbola_eccentricity b a = Real.sqrt (1 + (Real.sqrt 3 / 3)^2)) :
    hyperbola_eccentricity b a = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l452_452645


namespace terminating_decimal_representation_count_l452_452622

theorem terminating_decimal_representation_count : 
  (∃ (s : Set ℕ), s = { n : ℕ | 1 ≤ n ∧ n ≤ 349 ∧ 7 ∣ n } ∧ s.card = 49) :=
by
  sorry

end terminating_decimal_representation_count_l452_452622


namespace value_of_F_at_4_l452_452913

def F (x : ℝ) : ℝ := sqrt (abs (x - 2)) + (10 / Real.pi) * atan (sqrt (abs (x - 1)))

theorem value_of_F_at_4 : F 4 = 4 := by
  -- The graph (4,4) is given as a point on the curve y=F(x)
  have h : (4, 4) ∈ set_of (λ p : ℝ × ℝ, p.2 = F p.1) := by
    simp [F]
    exact rfl
  sorry

end value_of_F_at_4_l452_452913


namespace sqrt_eq_solution_l452_452947

theorem sqrt_eq_solution (z : ℝ) : z = (134 / 3) ↔ sqrt (10 + 3 * z) = 12 := 
by
  sorry

end sqrt_eq_solution_l452_452947


namespace find_p_q_l452_452247

variable {R : Type*} [CommRing R]

theorem find_p_q (p q : R) :
  (X^5 - X^4 + X^3 - p * X^2 + q * X + 4 : Polynomial R) % (X - 2) = 0 ∧ 
  (X^5 - X^4 + X^3 - p * X^2 + q * X + 4 : Polynomial R) % (X + 1) = 0 → 
  (p = 5 ∧ q = -4) :=
by
  sorry

end find_p_q_l452_452247


namespace product_of_midpoint_coordinates_l452_452476

theorem product_of_midpoint_coordinates
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 4) (h2 : y1 = -3) (h3 : x2 = -8) (h4 : y2 = 7) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx * my = -4) :=
by
  -- Here we would carry out the proof.
  sorry

end product_of_midpoint_coordinates_l452_452476


namespace point_P_moves_distance_l452_452541

theorem point_P_moves_distance (P: ℝ × ℝ) (B: ℝ × ℝ) (B': ℝ × ℝ) (r r': ℝ) (k: ℝ)
  (hB: B = (3, 3))
  (hB': B' = (10, 12))
  (hr: r = 3)
  (hr': r' = 5)
  (hk: k = r' / r)
  (hP: P = (1, 1))
  (center_of_dilation: ℝ × ℝ)
  (d_center: center_of_dilation = (-1, -1))
  (d_0: ℝ)
  (d_1: ℝ)
  (hd_0: d_0 = Real.sqrt ((P.1 + 1)^2 + (P.2 + 1)^2))
  (hd_1: d_1 = k * d_0) :
  Real.sqrt(4 + 4) = 2 * Real.sqrt 2 ∧
  d_0 = 2 * Real.sqrt 2 ∧
  d_1 - d_0 = (k - 1) * d_0 ∧
  d_1 - d_0 = (4 / 3) * Real.sqrt 2 :=
sorry

end point_P_moves_distance_l452_452541


namespace swimming_pool_width_l452_452556

theorem swimming_pool_width (L D1 D2 V : ℝ) (W : ℝ) (h : L = 12) (h1 : D1 = 1) (h2 : D2 = 4) (hV : V = 270) : W = 9 :=
  by
    -- We begin by stating the formula for the volume of 
    -- a trapezoidal prism: Volume = (1/2) * (D1 + D2) * L * W
    
    -- According to the problem, we have the following conditions:
    have hVolume : V = (1/2) * (D1 + D2) * L * W :=
      by sorry

    -- Substitute the provided values into the volume equation:
    -- 270 = (1/2) * (1 + 4) * 12 * W
    
    -- Simplify and solve for W
    simp at hVolume
    exact sorry

end swimming_pool_width_l452_452556


namespace least_distinct_values_l452_452059

theorem least_distinct_values
  (n : ℕ)
  (mode_count : ℕ)
  (total_count : ℕ)
  (hn : total_count = 2023)
  (hmode : mode_count = 11)
  (hlist : total_count = n) 
  (h : ∀ x, x ≠ mode_count → x < 11)
  : ∃ m : ℕ, m = 203 ∧ n ≥ 11 + 10 * (m - 1) :=
by
  have h1 : total_count - mode_count = 2012 := by linarith
  have h2 : 10 * (m - 1) ≥ 2012 := 
    by linarith
    exact ⟨203, by linarith⟩

end least_distinct_values_l452_452059


namespace initial_sale_price_percent_l452_452552

theorem initial_sale_price_percent (P S : ℝ) (h1 : S * 0.90 = 0.63 * P) :
  S = 0.70 * P :=
by
  sorry

end initial_sale_price_percent_l452_452552


namespace minimize_f_l452_452676

variables {V : Type*} [inner_product_space ℝ V]

/-- Given vectors a, b, c, d in an inner product space over the reals, and the function
  f(t) = |t * a + c|^2 + |t * b + d|^2, the value of t that minimizes f(t) is
  t = -(⟪a, c⟫ + ⟪b, d⟫) / (‖a‖^2 + ‖b‖^2), assuming that at least one of a or b is non-zero. -/
theorem minimize_f (a b c d : V) (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ t : ℝ, ∀ t', (‖(t' • a + c)‖^2 + ‖(t' • b + d)‖^2) ≥ (‖((-((⟪a, c⟫ + ⟪b, d⟫) / (‖a‖^2 + ‖b‖^2)) • a + c)‖^2 + ‖((-((⟪a, c⟫ + ⟪b, d⟫) / (‖a‖^2 + ‖b‖^2)) • b + d)‖^2)) :=
  sorry

end minimize_f_l452_452676


namespace greatest_integer_less_than_PS_l452_452286

noncomputable def rectangle_problem (PQ PS : ℝ) (T : ℝ) (PT QP : ℝ) : ℝ := real.sqrt (PQ * PQ + PS * PS) / 2

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ) (hPQ : PQ = 150)
  (T_midpoint : T = PS / 2)
  (PT_perpendicular_QT : PT * PT + T * T = PQ * PQ),
  int.floor (PS) = 212 :=
by
  intros PQ PS hPQ T_midpoint PT_perpendicular_QT
  have h₁ : PS = rectangle_problem PQ PS T PQ,
  {
    sorry
  }
  have h₂ : 150 * real.sqrt 2,
  {
    sorry
  }
  have h₃ : (⌊150 * real.sqrt 2⌋ : ℤ) = 212,
  {
    sorry
  }
  exact h₃

end greatest_integer_less_than_PS_l452_452286


namespace walking_distance_l452_452698

theorem walking_distance (D : ℕ) (h : D / 15 = (D + 60) / 30) : D = 60 :=
by
  sorry

end walking_distance_l452_452698


namespace triangle_side_length_l452_452723

theorem triangle_side_length (A B C : Type) [RealField A] [RealField B] [RealField C]
  (angle_A : Real) (angle_B : Real) (AC : Real)
  (hangle_A : angle_A = 45) (hangle_B : angle_B = 75) (hAC : AC = 6) :
  ∃ BC : Real, BC = 4.3926 := by
  sorry

end triangle_side_length_l452_452723


namespace slope_of_midpoints_l452_452100

-- Definitions of points
structure Point :=
(x : ℝ)
(y : ℝ)

-- Function to calculate midpoint of a segment
def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

-- Function to calculate the slope between two points
def slope (P Q : Point) : ℝ :=
  if P.x ≠ Q.x then (Q.y - P.y) / (Q.x - P.x) else 0

-- The endpoints of the two segments
def A := { x := 1, y := -1 } : Point
def B := { x := 3, y := 4 } : Point
def C := { x := 7, y := -1 } : Point
def D := { x := 9, y := 4 } : Point

-- The midpoints of the two segments
def M1 := midpoint A B
def M2 := midpoint C D

-- The slope of the line connecting the midpoints
theorem slope_of_midpoints : slope M1 M2 = 0 :=
by
  rw [M1, M2, midpoint],
  -- Calculating the coordinates of M1 and M2
  have hM1 : M1 = { x := 2, y := 1.5 }, by simp [midpoint];
  have hM2 : M2 = { x := 8, y := 1.5 }, by simp [midpoint];
  rw [hM1, hM2, slope],
  -- Since y-coordinates are the same, the slope is 0
  rw if_neg, simp, norm_num,
  -- x-coordinates are different, therefore the slope calculation is valid
  exact dec_trivial,
  exact hM1,
  exact hM2


end slope_of_midpoints_l452_452100


namespace count_numbers_without_digit_3_l452_452238

def does_not_contain_digit (n : ℕ) (d : ℕ) : Prop :=
  ¬(d ∈ (nat.digits 10 n))

theorem count_numbers_without_digit_3 : 
  ∃ n, n = 242 ∧ n = (finset.card (finset.filter (λ x, does_not_contain_digit x 3) (finset.range 501))) :=
begin
  -- placeholder for the proof
  sorry
end

end count_numbers_without_digit_3_l452_452238


namespace shop_owner_profitable_l452_452069

noncomputable def shop_owner_profit (CP_SP_difference_percentage: ℚ) (CP: ℚ) (buy_cheat_percentage: ℚ) (sell_cheat_percentage: ℚ) (buy_discount_percentage: ℚ) (sell_markup_percentage: ℚ) : ℚ := 
  CP_SP_difference_percentage * 100

theorem shop_owner_profitable :
  shop_owner_profit ((114 * (110 / 80 / 100) - 90) / 90) 1 0.14 0.20 0.10 0.10 = 74.17 := 
by
  sorry

end shop_owner_profitable_l452_452069


namespace largest_integer_less_log_expr_l452_452847

theorem largest_integer_less_log_expr : 
  let expr := (∑ k in Finset.range 3010, log 3 (k + 1 / k))
  ∃ (n : ℤ), n = 6 ∧ (n < log 3 3010) ∧ (6 < log 3 3010) ∧ (log 3 3010 < 7) :=
by
  sorry

end largest_integer_less_log_expr_l452_452847


namespace total_candy_l452_452915

theorem total_candy (candy1 candy2 chocolate : ℕ) (h1 : candy1 = 28) (h2 : candy2 = 42) (h3 : chocolate = 63) :
  (candy1 + candy2 + chocolate) = 133 :=
by
  rw [h1, h2, h3]
  exact rfl

end total_candy_l452_452915


namespace last_two_digits_of_100_factorial_l452_452121

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_nonzero_digits (n : ℕ) : ℕ := sorry

theorem last_two_digits_of_100_factorial :
  last_two_nonzero_digits (factorial 100) = 24 :=
sorry

end last_two_digits_of_100_factorial_l452_452121


namespace rearrange_numbers_25_rearrange_numbers_1000_l452_452312

theorem rearrange_numbers_25 (n : ℕ) (h : n = 25) : 
  ∃ (f : Fin n → Fin n), ∀ i : Fin (n - 1), (f i.succ.val - f i.val = 3 ∨ f i.succ.val - f i.val = 5 ∨ f i.val - f i.succ.val = 3 ∨ f i.val - f i.succ.val = 5) ∧ (f i).val ∈ Finset.range n := by
  sorry

theorem rearrange_numbers_1000 (n : ℕ) (h : n = 1000) : 
  ∃ (f : Fin n → Fin n), ∀ i : Fin (n - 1), (f i.succ.val - f i.val = 3 ∨ f i.succ.val - f i.val = 5 ∨ f i.val - f i.succ.val = 3 ∨ f i.val - f i.succ.val = 5) ∧ (f i).val ∈ Finset.range n := by
  sorry

end rearrange_numbers_25_rearrange_numbers_1000_l452_452312


namespace brad_started_after_maxwell_l452_452385

theorem brad_started_after_maxwell :
  ∀ (distance maxwell_speed brad_speed maxwell_time : ℕ),
  distance = 94 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  maxwell_time = 10 →
  (distance - maxwell_speed * maxwell_time) / brad_speed = 9 := 
by
  intros distance maxwell_speed brad_speed maxwell_time h_dist h_m_speed h_b_speed h_m_time
  sorry

end brad_started_after_maxwell_l452_452385


namespace smallest_n_for_infinite_operations_l452_452828

theorem smallest_n_for_infinite_operations (p q : ℕ) (hp : p > 0) (hq : q > 0) (gcd_pq : Nat.gcd p q = 1) : ∃ n, n ≥ p + q ∧ (∀ m, m < p + q → ¬infinite_operations_possible m p q) :=
by
  -- Define and prove necessary conditions
  sorry

-- Auxiliary function to define if infinite operations are possible
def infinite_operations_possible (n p q : ℕ) : Prop := 
  sorry

end smallest_n_for_infinite_operations_l452_452828


namespace measure_of_angle_BYZ_l452_452579

-- Definitions based on conditions
def angle_A : ℝ := 50
def angle_B : ℝ := 70
def angle_C : ℝ := 60

-- Theorem statement based on the proof problem
theorem measure_of_angle_BYZ 
 (h1 : ∃ γ : Type, incircle (triangle ABC) γ ∧ circumcircle (triangle XYZ) γ)
 (h2 : point_on_line_segment X BC)
 (h3 : point_on_line_segment Y AB)
 (h4 : point_on_line_segment Z AC)
 (h5 : ∠A = angle_A)
 (h6 : ∠B = angle_B)
 (h7 : ∠C = angle_C) : 
 ∠BYZ = 60 :=
sorry

end measure_of_angle_BYZ_l452_452579


namespace sum_of_distinct_prime_factors_of_number_is_10_l452_452934

-- Define the constant number 9720
def number : ℕ := 9720

-- Define the distinct prime factors of 9720
def distinct_prime_factors_of_number : List ℕ := [2, 3, 5]

-- Sum function for the list of distinct prime factors
def sum_of_distinct_prime_factors (lst : List ℕ) : ℕ :=
  lst.foldr (.+.) 0

-- The main theorem to prove
theorem sum_of_distinct_prime_factors_of_number_is_10 :
  sum_of_distinct_prime_factors distinct_prime_factors_of_number = 10 := by
  sorry

end sum_of_distinct_prime_factors_of_number_is_10_l452_452934


namespace sqrt_expression_value_l452_452507

theorem sqrt_expression_value :
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 25 * Real.sqrt 5 :=
by
  sorry

end sqrt_expression_value_l452_452507


namespace periodic_function_solution_l452_452370

noncomputable def periodic_function_problem
  (f : ℝ → ℝ)
  (periodic_f : ∀ x, f (x + 1) = f x) : Prop :=
  
  let g := λ x, f x + 2 * x in
  
  (set.image g (set.Icc 1 2) = set.Icc (-1 : ℝ) 5) →
  (set.image g (set.Icc (-2020 : ℝ) 2020) = set.Icc (-4043 : ℝ) 4041)

theorem periodic_function_solution
  (f : ℝ → ℝ)
  (periodic_f : ∀ x, f (x + 1) = f x)
  (h : set.image (λ x, f x + 2 * x) (set.Icc 1 2) = set.Icc (-1 : ℝ) 5) :
  set.image (λ x, f x + 2 * x) (set.Icc (-2020 : ℝ) 2020) = set.Icc (-4043 : ℝ) 4041 :=
sorry

end periodic_function_solution_l452_452370


namespace probability_of_sum_leq_9_on_two_dice_is_5_over_6_l452_452826

def probability_sum_leq_9 (n : ℕ) (m : ℕ) : ℚ :=
  if n ∈ {1, 2, 3, 4, 5, 6} ∧ m ∈ {1, 2, 3, 4, 5, 6}
  then (36 - 6) / 36 else 0 -- considering the total favorable outcomes (30 out of 36)

theorem probability_of_sum_leq_9_on_two_dice_is_5_over_6 :
  probability_sum_leq_9 6 6 = 5 / 6 :=
by 
  sorry

end probability_of_sum_leq_9_on_two_dice_is_5_over_6_l452_452826


namespace intersection_of_A_and_B_l452_452205

def set_A : Set ℝ := {x | x >= 1 ∨ x <= -2}
def set_B : Set ℝ := {x | -3 < x ∧ x < 2}

def set_C : Set ℝ := {x | (-3 < x ∧ x <= -2) ∨ (1 <= x ∧ x < 2)}

theorem intersection_of_A_and_B (x : ℝ) : x ∈ set_A ∧ x ∈ set_B ↔ x ∈ set_C :=
  by
  sorry

end intersection_of_A_and_B_l452_452205


namespace double_age_in_years_l452_452544

-- Defining the conditions given in the problem
def present_age_student: ℕ := 24
def age_difference: ℕ := 26
def student_to_man_age (S: ℕ) : ℕ := S + age_difference
def years_to_double_age (M S Y: ℕ) : Prop := M + Y = 2 * (S + Y)

-- Proof problem statement
theorem double_age_in_years:
  ∃ (Y: ℕ), 
  let S := present_age_student in
  let M := student_to_man_age S in
  years_to_double_age M S Y := 
  by
  { 
    existsi 2,
    unfold present_age_student student_to_man_age years_to_double_age,
    sorry
  }

end double_age_in_years_l452_452544


namespace set_M_real_l452_452250

noncomputable def set_M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem set_M_real :
  set_M = {z : ℂ | ∃ x : ℝ, z = x} :=
by
  sorry

end set_M_real_l452_452250


namespace problem_sum_fraction_result_l452_452124

theorem problem_sum_fraction_result : 
  let N := ∑ i in Finset.range 2022, (2023 - (i + 1)) / (i + 1 : ℝ)
  let D := ∑ j in Finset.range (2023 - 2 + 1), 1 / (j + 2 : ℝ)
  in N / D = 2023 := 
by
  sorry

end problem_sum_fraction_result_l452_452124


namespace sum_b_n_l452_452302

theorem sum_b_n (a : ℕ → ℕ) (b : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a 1 = 1 ∧ a n + a (n + 1) = 2 * n + 1 ∧ (∀ b_n, x^2 - (2 * n + 1) * x + 1 / b_n = 0)
  ∧ (∀ n, b n = 1 / (n * (n + 1))))
  → (∀ n, S n = (finset.range n).sum (λ k, b (k + 1)))
  → ∀ n, S n = n / (n + 1) := 
by
  sorry

end sum_b_n_l452_452302


namespace triangle_area_expr_l452_452039

theorem triangle_area_expr (a b c : ℝ) (α β γ : ℝ) (t : ℝ)
  (h_cosine1 : a^2 = b^2 + c^2 - 2 * b * c * real.cos α)
  (h_cosine2 : b^2 = c^2 + a^2 - 2 * c * a * real.cos β)
  (h_cosine3 : c^2 = a^2 + b^2 - 2 * a * b * real.cos γ)
  (h_area1 : t = 1/2 * b * c * real.sin α)
  (h_area2 : t = 1/2 * c * a * real.sin β)
  (h_area3 : t = 1/2 * a * b * real.sin γ) :
  t = (a^2 + b^2 + c^2) / (4 * (real.cot α + real.cot β + real.cot γ)) := sorry

end triangle_area_expr_l452_452039


namespace divisibility_condition_l452_452358

def C (s : ℕ) : ℕ := s * (s + 1)

theorem divisibility_condition (k m n : ℕ) 
  (k_pos : 0 < k) (m_pos : 0 < m) (n_pos : 0 < n) 
  (prime_m_k : Nat.Prime (m + k + 1)) (prime_gt_n : m + k + 1 > n + 1) :
  (∏ i in Finset.range n, C (m + i + 1) - C k) ∣ (∏ j in Finset.range n, C (j + 1)) :=
by
  sorry

end divisibility_condition_l452_452358


namespace no_pairs_xy_perfect_square_l452_452143

theorem no_pairs_xy_perfect_square :
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ ∃ k : ℕ, (xy + 1) * (xy + x + 2) = k^2 := 
by {
  sorry
}

end no_pairs_xy_perfect_square_l452_452143


namespace rationalize_denominator_l452_452397

theorem rationalize_denominator :
  (1 / (real.sqrt 3 - 2)) = -(real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l452_452397


namespace carmen_candle_usage_l452_452103

-- Define the duration a candle lasts when burned for 1 hour every night.
def candle_duration_1_hour_per_night : ℕ := 8

-- Define the number of hours Carmen burns a candle each night.
def hours_burned_per_night : ℕ := 2

-- Define the number of nights over which we want to calculate the number of candles needed.
def number_of_nights : ℕ := 24

-- We want to show that given these conditions, Carmen will use 6 candles.
theorem carmen_candle_usage :
  (number_of_nights / (candle_duration_1_hour_per_night / hours_burned_per_night)) = 6 :=
by
  sorry

end carmen_candle_usage_l452_452103


namespace area_of_parallelogram_l452_452673

-- Points definitions
def O : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (1, real.sqrt 3, 2)
def B : ℝ × ℝ × ℝ := (real.sqrt 3, -1, 2)

-- Vectors definitions
def vector_OA : ℝ × ℝ × ℝ := (1, real.sqrt 3, 2)
def vector_OB : ℝ × ℝ × ℝ := (real.sqrt 3, -1, 2)

-- Function to calculate the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Function to calculate the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def angle_cosine (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

noncomputable def angle_sine (cosine : ℝ) : ℝ :=
  real.sqrt (1 - cosine ^ 2)

noncomputable def parallelogram_area (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  magnitude v1 * magnitude v2 * angle_sine (angle_cosine v1 v2)

theorem area_of_parallelogram :
  parallelogram_area vector_OA vector_OB = 4 * real.sqrt 3 :=
by
  sorry

end area_of_parallelogram_l452_452673


namespace kelsey_distance_l452_452523

theorem kelsey_distance :
  ∃ D : ℝ, (D / 2) / 25 + (D / 2) / 40 = 10 ∧ D ≈ 154 := 
by
  sorry

end kelsey_distance_l452_452523


namespace find_ratio_AP_PD_l452_452725

noncomputable def triangle_ratios (AB AC : ℝ) (P D : Point) (AP PD : ℝ) : Prop :=
  AB = 7 ∧ AC = 5 ∧ (exists (AD BM : Line) (P : Point), ad_is_angle_bisector AD P D ∧ bm_is_median BM P ∧ P_on AD ∧ P_on BM) →
  AP / PD = 12 / 7

theorem find_ratio_AP_PD : triangle_ratios 7 5 P D AP PD := 
  sorry

end find_ratio_AP_PD_l452_452725


namespace only_D_is_even_l452_452857

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def f_A (x : ℝ) : ℝ := x^3
def f_B (x : ℝ) : ℝ := 2^x
def f_C (x : ℝ) : ℝ := abs (x + 1)
def f_D (x : ℝ) : ℝ := abs x + 1

theorem only_D_is_even :
  is_even_function f_D ∧ ¬ is_even_function f_A ∧ ¬ is_even_function f_B ∧ ¬ is_even_function f_C :=
by
  sorry

end only_D_is_even_l452_452857


namespace number_of_elements_in_C_l452_452649

open Set

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {z | ∃ x ∈ A, ∃ y ∈ B, z = x * y}

theorem number_of_elements_in_C : (C.finite_to_set.card = 11) := 
  sorry

end number_of_elements_in_C_l452_452649


namespace modulus_of_z_l452_452661

theorem modulus_of_z (z : ℂ) (h : (3 + 4 * complex.I) * z = 5 * complex.I ^ 2016) : |z| = 1 :=
sorry

end modulus_of_z_l452_452661


namespace odd_function_and_range_l452_452222

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

theorem odd_function_and_range {a : ℝ}
    (h1 : ∀ x : ℝ, f a (-x) = -f a x) :
    a = 1 / 2 ∧ ∀ y : ℝ, y ∈ set.range (f (1 / 2)) → y > -1 / 2 ∧ y < 1 / 2 :=
by
  sorry

end odd_function_and_range_l452_452222


namespace prob_first_red_light_at_C_is_correct_expected_waiting_time_is_correct_l452_452073

noncomputable def prob_red_light_first_time_at_C : ℚ :=
  let prob_A := 1 / 3;
  let prob_B := 1 / 4;
  let prob_C := 3 / 4;
  (1 - prob_A) * (1 - prob_B) * prob_C

theorem prob_first_red_light_at_C_is_correct :
  prob_red_light_first_time_at_C = 3 / 8 :=
by 
  -- Proof needs to be provided here
  sorry

noncomputable def expected_waiting_time : ℚ :=
  let prob_A := 1 / 3;
  let prob_B := 1 / 4;
  let prob_C := 3 / 4;
  let times := [(0, (1 - prob_A) * (1 - prob_B) * (1 - prob_C)),
                (40, prob_A * (1 - prob_B) * (1 - prob_C)),
                (20, (1 - prob_A) * prob_B * (1 - prob_C)),
                (80, (1 - prob_A) * (1 - prob_B) * prob_C),
                (60, prob_A * prob_B * (1 - prob_C)),
                (100, (1 - prob_A) * prob_B * prob_C),
                (120, prob_A * (1 - prob_B) * prob_C),
                (140, prob_A * prob_B * prob_C)];
  times.foldl (λ acc (t, p), acc + t * p) 0

theorem expected_waiting_time_is_correct : 
  expected_waiting_time = 235 / 3 :=
by 
  -- Proof needs to be provided here
  sorry

end prob_first_red_light_at_C_is_correct_expected_waiting_time_is_correct_l452_452073


namespace tetrahedron_median_planes_parallelepiped_median_planes_distinct_parallelepipeds_l452_452166

-- Definition of median plane
def is_median_plane (polyhedron : Type) (plane : Type) : Prop :=
  ∀ v : polyhedron, ∃ d : ℝ, dist v plane = d

-- Part (1): Number of distinct median planes in a tetrahedron
theorem tetrahedron_median_planes : 
  ∃ (α : Type), ∀ (T : α), is_tetrahedron T → (∃ (planes : Finset Type), planes.card = 4 ∧ ∀ (plane : Type) (H : plane ∈ planes), is_median_plane T plane) := 
sorry

-- Part (2): Number of distinct median planes in a parallelepiped
theorem parallelepiped_median_planes : 
  ∃ (α : Type), ∀ (P : α), is_parallelepiped P → (∃ (planes : Finset Type), planes.card = 3 ∧ ∀ (plane : Type) (H : plane ∈ planes), is_median_plane P plane ) :=
sorry

-- Part (3): Number of distinct parallelepipeds formed by four non-coplanar points
theorem distinct_parallelepipeds :
  ∀ (points : Finset (ℝ × ℝ × ℝ)), points.card = 4 ∧ ¬∃ (plane : Type), (∀ p ∈ points, p ∈ plane) → 
  ∃ (parallelepipeds : Finset Type), parallelepipeds.card = 3 :=
sorry

end tetrahedron_median_planes_parallelepiped_median_planes_distinct_parallelepipeds_l452_452166


namespace product_of_solutions_l452_452933

theorem product_of_solutions (α β : ℝ) (h : 2 * α^2 + 8 * α - 45 = 0 ∧ 2 * β^2 + 8 * β - 45 = 0 ∧ α ≠ β) :
  α * β = -22.5 :=
sorry

end product_of_solutions_l452_452933


namespace evaluate_expression_l452_452938

def x : ℚ := 1 / 4
def y : ℚ := 1 / 3
def z : ℚ := 12

theorem evaluate_expression : x^3 * y^4 * z = 1 / 432 := 
by
  sorry

end evaluate_expression_l452_452938


namespace f_eq_g_l452_452790

def pos_int := { n : ℕ // n > 0 }

variables (f g : pos_int → pos_int)

axiom f_g_eqn : ∀ n : pos_int, f (g n) = ⟨f n.val + 1, nat.succ_pos' _⟩
axiom g_f_eqn : ∀ n : pos_int, g (f n) = ⟨g n.val + 1, nat.succ_pos' _⟩

theorem f_eq_g : ∀ n : pos_int, f n = g n := 
by
  sorry

end f_eq_g_l452_452790


namespace find_p_l452_452210

noncomputable def circle := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 4}
noncomputable def parabola_directrix (p : ℝ) := {x : ℝ | x = -p / 2}

theorem find_p (p : ℝ) (h : p > 0)
  (h1 : ∀ x : ℝ, x ∈ parabola_directrix p → (1 + x = 2)) : p = 2 :=
sorry

end find_p_l452_452210


namespace line_cannot_pass_through_fourth_quadrant_l452_452653

theorem line_cannot_pass_through_fourth_quadrant
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b < 0)
  (hc : c > 0) :
  ¬(∃ x y : ℝ, x > 0 ∧ y < 0 ∧ ax + by + c = 0) :=
sorry

end line_cannot_pass_through_fourth_quadrant_l452_452653


namespace Juwella_reads_pages_l452_452136

theorem Juwella_reads_pages (p1 p2 p3 p_total p_tonight : ℕ) 
                            (h1 : p1 = 15)
                            (h2 : p2 = 2 * p1)
                            (h3 : p3 = p2 + 5)
                            (h4 : p_total = 100) 
                            (h5 : p_total = p1 + p2 + p3 + p_tonight) :
  p_tonight = 20 := 
sorry

end Juwella_reads_pages_l452_452136


namespace swimming_pool_volume_l452_452687

theorem swimming_pool_volume (d h : ℝ) (π : ℝ) (r : ℝ) (V : ℝ) 
  (diameter_condition: d = 16) (depth_condition: h = 4) 
  (radius_condition: r = d / 2) (volume_formula: V = π * r^2 * h) :
  V = 256 * π := 
by 
  tactic.assumption_sorry

end swimming_pool_volume_l452_452687


namespace smallest_common_multiple_of_9_and_6_l452_452495

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l452_452495


namespace scientific_notation_of_75500000_l452_452266

theorem scientific_notation_of_75500000 :
  ∃ (a : ℝ) (n : ℤ), 75500000 = a * 10 ^ n ∧ a = 7.55 ∧ n = 7 :=
by {
  sorry
}

end scientific_notation_of_75500000_l452_452266


namespace number_of_pairings_l452_452578

-- We introduce the notion of bowls and glasses as sets of 5 distinct elements each.
def bowls : Finset ℕ := {0, 1, 2, 3, 4}
def glasses : Finset ℕ := {0, 1, 2, 3, 4}

theorem number_of_pairings : (bowls.card * glasses.card = 25) :=
by
  have bowls_card : bowls.card = 5 := by
    sorry
  have glasses_card : glasses.card = 5 := by
    sorry
  rw [bowls_card, glasses_card]
  exact (5 * 5) = 25

end number_of_pairings_l452_452578


namespace min_value_of_m_l452_452188

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def b_n (n : ℕ) : ℝ := 2 * n - 9
noncomputable def c_n (n : ℕ) : ℝ := b_n n / a_n n

theorem min_value_of_m (m : ℝ) : (∀ n : ℕ, c_n n ≤ m) → m ≥ 1/162 :=
by
  sorry

end min_value_of_m_l452_452188


namespace expected_value_is_1025_l452_452404

noncomputable def E (x: ℕ) : ℕ :=
if x % 100 = 0 then 0 else sorry

noncomputable def expected_value_final_sum : ℕ :=
41 * 99 / 95 + ∑ i in finset.range 99, E i

theorem expected_value_is_1025 (final_value : ℕ) : final_value = 1025 :=
begin
  let final_value := expected_value_final_sum,
  sorry
end

end expected_value_is_1025_l452_452404


namespace prime_list_mean_l452_452954

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def prime_list_mean_proof : Prop :=
  let nums := [33, 37, 39, 41, 43] in
  let primes := filter is_prime nums in
  primes = [37, 41, 43] ∧
  (primes.sum : ℤ) / primes.length = 40.33

theorem prime_list_mean : prime_list_mean_proof :=
by
  sorry

end prime_list_mean_l452_452954


namespace number_of_ways_to_elect_at_least_one_girl_l452_452542

theorem number_of_ways_to_elect_at_least_one_girl :
  let students := 10
  let girls := 3
  let boys := students - girls
  let choose := λ n k, n.choose k
  (choose girls 1 * choose boys 1 + choose girls 2) = 24 :=
by
  sorry

end number_of_ways_to_elect_at_least_one_girl_l452_452542


namespace find_width_of_rectangular_field_l452_452066

noncomputable def rectangular_field_width : Prop :=
  ∃ (w l : ℝ), 
    l = (7 / 5) * w ∧
    2 * l + 2 * w = 288 ∧
    l * w = 2592 ∧
    let d := Real.sqrt (l ^ 2 + w ^ 2) in
    Real.arccos (w / d) = Real.pi / 3 ∧ -- Angle in radians (60 degrees)
    w = 60

theorem find_width_of_rectangular_field : rectangular_field_width :=
sorry

end find_width_of_rectangular_field_l452_452066


namespace sin_ratio_proof_l452_452305

noncomputable def sin_ratio
  (X Y Z W : Type)
  (hXY : X ≠ Y) (hXZ : X ≠ Z) (hYW : Y ≠ W) (hZW : Z ≠ W)
  (angle_Y : Real := 75 * pi / 180) (angle_Z : Real := 30 * pi / 180)
  (divide_ratio : Real := 2/3)
  : Real :=
  sorry

theorem sin_ratio_proof
  (X Y Z W : Type)
  (hXY : X ≠ Y) (hXZ : X ≠ Z) (hYW : Y ≠ W) (hZW : Z ≠ W)
  (angle_Y_def : ∠ Y = 75)
  (angle_Z_def : ∠ Z = 30)
  (divide_ratio_def : divides YZ W = 2/3)
  : sin_ratio X Y Z W hXY hXZ hYW hZW angle_Y_def angle_Z_def divide_ratio_def
  = (Real.sqrt 6 + Real.sqrt 2) / 8 :=
sorry

end sin_ratio_proof_l452_452305


namespace cos_angle_A1P_B1Q_l452_452294

-- Define coordinates of points A1, P, B1, Q in a 3-dimensional space
def A1 := (2, 0, 2)
def P := (1, 2, 0)
def B1 := (2, 2, 2)
def Q := (0, 1, 1)

-- Define vectors A1P and B1Q
def vecA1P := (-1, 2, -2)
def vecB1Q := (-2, -1, -1)

noncomputable def dot_product (u v : ℕ × ℕ × ℕ) : ℤ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def norm (u : ℕ × ℕ × ℕ) : ℝ :=
  real.sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

noncomputable def cos_angle (u v : ℕ × ℕ × ℕ) : ℝ :=
  (dot_product u v) / (norm u * norm v)

-- Main theorem statement
theorem cos_angle_A1P_B1Q : cos_angle vecA1P vecB1Q = 1 / 6 := by
  sorry

end cos_angle_A1P_B1Q_l452_452294


namespace base9_to_base10_l452_452823

theorem base9_to_base10 (n : ℕ) (h : n = 3618) : 
  nat.digits 9 n = 2690 :=
by 
  sorry

end base9_to_base10_l452_452823


namespace imaginary_part_of_reciprocal_l452_452701

theorem imaginary_part_of_reciprocal (a : ℝ) 
  (h : ∃ z : ℂ, z = (a^2 - 1) + (a + 1) * complex.i ∧ z = (a + 1) * complex.i) :
  complex.imag_part (1 / ((a^2 - 1) + (a + 1) * complex.i + a)) = -2/5 :=
by
  sorry

end imaginary_part_of_reciprocal_l452_452701


namespace area_of_parallelogram_7_12_60_l452_452147

noncomputable def sqrt3_approx : ℝ := 1.732

def area_parallelogram (a b : ℝ) (theta : ℝ) : ℝ :=
  a * b * real.sin theta

theorem area_of_parallelogram_7_12_60 : 
  area_parallelogram 12 7 (real.pi / 3) ≈ 72.744 :=
by
  have height := 7 * (sqrt3_approx / 2)
  have area := 12 * height
  calc area ≈ 72.744 : by norm_num
  sorry

end area_of_parallelogram_7_12_60_l452_452147


namespace vectors_parallel_l452_452235

theorem vectors_parallel (m : ℝ) : 
    (∃ k : ℝ, (m, 4) = (k * 5, k * -2)) → m = -10 := 
by
  sorry

end vectors_parallel_l452_452235


namespace minimum_value_expr_l452_452727

theorem minimum_value_expr (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  (1 + (1 / m)) * (1 + (1 / n)) = 9 :=
sorry

end minimum_value_expr_l452_452727


namespace problem_sum_fraction_result_l452_452125

theorem problem_sum_fraction_result : 
  let N := ∑ i in Finset.range 2022, (2023 - (i + 1)) / (i + 1 : ℝ)
  let D := ∑ j in Finset.range (2023 - 2 + 1), 1 / (j + 2 : ℝ)
  in N / D = 2023 := 
by
  sorry

end problem_sum_fraction_result_l452_452125


namespace sum_triplets_l452_452580

noncomputable def geometric_sum : Rat := 
  ∑ x in Finset.range (∞ + 1), 
  ∑ y in Finset.range (∞ + 1), 
  ∑ z in Finset.range (∞ + 1), 
    1 / (105^x * 35^y * 7^z)

theorem sum_triplets {a b c : ℕ} (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) : 
    (∑ (a b c : ℕ) in {n | 1 ≤ a ∧ a < b ∧ b < c}.toFinset, 
       1 / (3^a * 5^b * 7^c)) = 1 / 21312 := 
by
  sorry

end sum_triplets_l452_452580


namespace shaded_area_correct_l452_452721

-- Define the radii of the semicircles
def radius_ADB : ℝ := 2
def radius_BEC : ℝ := 3

-- Define the area of a semicircle
def area_of_semicircle (r : ℝ) : ℝ := (1 / 2) * real.pi * r^2

-- Calculate the areas of the semicircles ADB and BEC
def area_ADB : ℝ := area_of_semicircle radius_ADB
def area_BEC : ℝ := area_of_semicircle (radius_BEC / 2)

-- Define the radius of the quarter-circle DFE
def radius_DFE : ℝ := (radius_ADB + (radius_BEC / 2)) / 2

-- Define the area of a quarter-circle
def area_of_quarter_circle (r : ℝ) : ℝ := (1 / 4) * real.pi * r^2

-- Calculate the area of the quarter-circle DFE
def area_DFE : ℝ := area_of_quarter_circle radius_DFE

-- Calculate the shaded area
def shaded_area : ℝ := (area_ADB + area_BEC) - area_DFE

-- State the theorem to be proven
theorem shaded_area_correct : shaded_area = 2.359375 * real.pi :=
by
  sorry

end shaded_area_correct_l452_452721


namespace complex_magnitude_eq_3_div_8_l452_452362

noncomputable def complex_problem (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) : ℝ :=
  | 1 / z + 1 / w |

theorem complex_magnitude_eq_3_div_8 (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  complex_problem z w hz hw hzw = 3 / 8 :=
sorry

end complex_magnitude_eq_3_div_8_l452_452362


namespace largest_C_l452_452640

theorem largest_C
  (n : ℕ)
  (h : n ≥ 2)
  (x : Fin n → ℝ) :
  ∃ (C : ℝ), C = 12 / (n * (n + 1) * (n + 2) * (3 * n + 1)) ∧
  ∑ i, (x i) ^ 2 ≥ (1 / (n + 1)) * (∑ i, x i) ^ 2 + C * (∑ i, ↑i * x i) ^ 2 :=
by
  sorry

end largest_C_l452_452640


namespace yoki_cans_correct_l452_452600

def total_cans := 85
def ladonna_cans := 25
def prikya_cans := 2 * ladonna_cans
def yoki_cans := total_cans - ladonna_cans - prikya_cans

theorem yoki_cans_correct : yoki_cans = 10 :=
by
  sorry

end yoki_cans_correct_l452_452600


namespace greatest_integer_less_than_PS_l452_452283

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ), PQ = 150 → 
  PS = 150 * Real.sqrt 3 → 
  (⌊PS⌋ = 259) := 
by
  intros PQ PS hPQ hPS
  sorry

end greatest_integer_less_than_PS_l452_452283


namespace sin3x_plus_sin7x_l452_452138

theorem sin3x_plus_sin7x (x : ℝ) : 
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by sorry

end sin3x_plus_sin7x_l452_452138


namespace real_number_solutions_l452_452611

theorem real_number_solutions (x : ℝ) :
  ([2 * x].toReal + [3 * x].toReal = 8 * x - 7 / 2) ↔ (x = 13 / 16 ∨ x = 17 / 16) := by
  sorry

end real_number_solutions_l452_452611


namespace rectangle_ratio_l452_452273

theorem rectangle_ratio (side_length : ℝ)
    (h1 : side_length = 3)
    (E F : ℝ)
    (h2 : E = side_length / 2)
    (h3 : F = side_length / 2)
    (AG_perpendicular_BF : Prop)
    (rearranged_area : Prop)
    : (let XY := (side_length ^ 2) / 12 in
       let YZ := 12 in 
       XY / YZ = 1 / 16) :=
by 
  sorry

end rectangle_ratio_l452_452273


namespace manny_marbles_l452_452444

theorem manny_marbles (total_marbles : ℕ) (ratio_m : ℕ) (ratio_n : ℕ) (manny_gives : ℕ) 
  (h_total : total_marbles = 36) (h_ratio_m : ratio_m = 4) (h_ratio_n : ratio_n = 5) (h_manny_gives : manny_gives = 2) : 
  (total_marbles * ratio_n / (ratio_m + ratio_n)) - manny_gives = 18 :=
by
  sorry

end manny_marbles_l452_452444


namespace people_counted_l452_452091

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end people_counted_l452_452091


namespace scientific_notation_214_7_billion_l452_452296

theorem scientific_notation_214_7_billion :
  let billion := 10^9 in
  214.7 * billion = 2.147 * 10^11 := 
by
  let billion := 10^9
  sorry

end scientific_notation_214_7_billion_l452_452296


namespace constant_term_of_polynomial_is_correct_l452_452207

noncomputable def integral_expr : ℝ :=
  3 * ∫ x in -1..1, x^2 + Real.sin x

theorem constant_term_of_polynomial_is_correct :
  m = integral_expr → 
  m = 2 →
  (∃ (k : ℕ), k = 2 ∧ ∑ i in Finset.range 7, (Nat.choose 6 i) * (1 / 2) ^ (6 - i) * (x ^ (i - 6/2)) = (15 / 16)) :=
begin
  intro m_def,
  intro m_val,
  use 2,
  split,
  { refl },
  sorry -- skip the proof as instructed
end

end constant_term_of_polynomial_is_correct_l452_452207


namespace mean_value_of_quadrilateral_l452_452593

theorem mean_value_of_quadrilateral {n : ℕ} (hn : n = 4) : 
    (let sum_of_interior_angles := (n - 2) * 180 in sum_of_interior_angles / 4 = 90) :=
by
    sorry

end mean_value_of_quadrilateral_l452_452593


namespace abs_x_minus_one_lt_two_iff_x_times_3_minus_x_gt_zero_iff_abs_x_minus_one_lt_two_is_necessary_but_not_sufficient_l452_452800

theorem abs_x_minus_one_lt_two_iff : ∀ x, (|x - 1| < 2) ↔ (-1 < x ∧ x < 3) :=
by
  sorry

theorem x_times_3_minus_x_gt_zero_iff : ∀ x, (x * (3 - x) > 0) ↔ (0 < x ∧ x < 3) :=
by
  sorry

theorem abs_x_minus_one_lt_two_is_necessary_but_not_sufficient : 
  (∀ x, |x - 1| < 2 → x * (3 - x) > 0) ∧ ¬(∀ x, x * (3 - x) > 0 → |x - 1| < 2) :=
by
  sorry

end abs_x_minus_one_lt_two_iff_x_times_3_minus_x_gt_zero_iff_abs_x_minus_one_lt_two_is_necessary_but_not_sufficient_l452_452800


namespace max_soap_boxes_in_carton_l452_452035

def carton_volume (length width height : ℕ) : ℕ :=
  length * width * height

def soap_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def max_soap_boxes (carton_volume soap_box_volume : ℕ) : ℕ :=
  carton_volume / soap_box_volume

theorem max_soap_boxes_in_carton :
  max_soap_boxes (carton_volume 25 42 60) (soap_box_volume 7 6 6) = 250 :=
by
  sorry

end max_soap_boxes_in_carton_l452_452035


namespace part_a_part_b_l452_452347

theorem part_a (n : ℕ) (hn : n % 2 = 1) (h_pos : n > 0) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n-1 ∧ ∃ f : (ℕ → ℕ), f k ≥ (n - 1) / 2 :=
sorry

theorem part_b : ∃ᶠ n in at_top, ∃ f : (ℕ → ℕ), ∀ k : ℕ, 1 ≤ k ∧ k ≤ n-1 → f k ≤ (n - 1) / 2 :=
sorry

end part_a_part_b_l452_452347


namespace sqrt_range_l452_452974

theorem sqrt_range (a : ℝ) : 2 * a - 1 ≥ 0 ↔ a ≥ 1 / 2 :=
by sorry

end sqrt_range_l452_452974


namespace speed_of_stream_l452_452047

theorem speed_of_stream (v_s : ℝ) : (v_s : ℝ) = 3 :=
  let boat_speed := 15
  let downstream_distance := boat_speed + v_s
  let upstream_distance := boat_speed - v_s
  let downstream_time := 1
  let upstream_time := 1.5
  let D_downstream := downstream_distance * downstream_time
  let D_upstream := upstream_distance * upstream_time
  have H : D_downstream = D_upstream := by simp [downstream_distance, downstream_time, upstream_distance, upstream_time, mul_add, mul_sub]
  calc
    15 + v_s = 22.5 - 1.5 * v_s : by exact H
    ...     = 22.5 - 1.5 * v_s : by sorry

end speed_of_stream_l452_452047


namespace trivia_team_absentees_l452_452560

theorem trivia_team_absentees (total_members : ℕ) (total_points : ℕ) (points_per_member : ℕ) 
  (h1 : total_members = 5) 
  (h2 : total_points = 6) 
  (h3 : points_per_member = 2) : 
  total_members - (total_points / points_per_member) = 2 := 
by 
  sorry

end trivia_team_absentees_l452_452560


namespace solution_set_g_inequality_l452_452213

noncomputable def f : ℝ → ℝ := sorry  -- Assume f is an even function defined on ℝ

def g (x : ℝ) : ℝ := x^2 * f x

lemma even_function (x : ℝ) : f(-x) = f(x) := sorry

lemma condition_inequality (x : ℝ) (hx : 0 ≤ x) : (x / 2) * deriv (deriv f x) + f(-x) ≤ 0 := sorry

theorem solution_set_g_inequality :
  { x : ℝ | g x < g (1 - 2 * x) } = set.Ioo (1 / 3 : ℝ) 1 :=
begin
  sorry
end

end solution_set_g_inequality_l452_452213


namespace remainder_of_power_mod_l452_452851

theorem remainder_of_power_mod 
  (n : ℕ)
  (h₁ : 7 ≡ 1 [MOD 6]) : 7^51 ≡ 1 [MOD 6] := 
sorry

end remainder_of_power_mod_l452_452851


namespace equation_of_line_l452_452699

def line_equation (x y : ℝ) : Prop := 3 * x - y - 3 = 0

theorem equation_of_line :
  ∃ (x1 y1 m : ℝ), (x1 = 1 ∧ y1 = 0 ∧ m = 3) → 
  (∀ (x y : ℝ), (y - y1 = m * (x - x1)) → line_equation x y) :=
by
  -- place a placeholder proof here
  intro x1 y1 m h,
  sorry

end equation_of_line_l452_452699


namespace henry_twice_jill_l452_452435

-- Conditions
def Henry := 29
def Jill := 19
def sum_ages : Nat := Henry + Jill

-- Prove the statement
theorem henry_twice_jill (Y : Nat) (H J : Nat) (h_sum : H + J = 48) (h_H : H = 29) (h_J : J = 19) :
  H - Y = 2 * (J - Y) ↔ Y = 9 :=
by {
  -- Here, we would provide the proof, but we'll skip that with sorry.
  sorry
}

end henry_twice_jill_l452_452435


namespace centroid_incenter_relation_l452_452350

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions for points and vectors
variables (A B C P : V)
variables (a b c : ℝ) -- side lengths

-- Define centroid G
def G : V := (A + B + C) / 3

-- Define incenter I
def I : V := (a * A + b * B + c * C) / (a + b + c)

-- Required distances (squared)
def PA2 := ∥P - A∥ ^ 2
def PB2 := ∥P - B∥ ^ 2
def PC2 := ∥P - C∥ ^ 2
def PG2 := ∥P - G A B C∥ ^ 2
def PI2 := ∥P - I A B C a b c∥ ^ 2

def GA2 := ∥G A B C - A∥ ^ 2
def GB2 := ∥G A B C - B∥ ^ 2
def GC2 := ∥G A B C - C∥ ^ 2
def GI2 := ∥G A B C - I A B C a b c∥ ^ 2

-- Prove or disprove the relationship given a constant k
theorem centroid_incenter_relation (k : ℝ) : 
  PA2 A B C P + PB2 A B C P + PC2 A B C P + PI2 A B C a b c P =
  k * (PG2 A B C P + GA2 A B C + GB2 A B C + GC2 A B C + GI2 A B C a b c) :=
sorry

end centroid_incenter_relation_l452_452350


namespace sum_inequality_l452_452183

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

theorem sum_inequality (a : ℕ → ℝ) (n : ℕ) (h : ∀ i : ℕ, 0 ≤ a i) :
  (∑ i in Finset.range n, a i / (2 * S a n - a i)) ≥ n / (2 * n - 1) :=
sorry

end sum_inequality_l452_452183


namespace maximum_cosine_value_l452_452191

noncomputable def maximum_cosine_argument_difference (z ω : ℂ) (h1 : z + ω + 3 = 0) (h2 : ∃ a b : ℝ, |z| = a ∧ |\ω| = b ∧ a, 2, b formArithmeticSequence) : ℝ :=
  if ∃ x : ℝ, cos (arg z - arg ω) = x then 1 / 8 else 0

-- Alternative Lean statement with a proof goal:
theorem maximum_cosine_value (z ω : ℂ) (h1 : z + ω + 3 = 0) (h2 : |z| + |\ω| = 4) : cos (arg z - arg ω) = 1 / 8 := 
begin
  sorry
end

end maximum_cosine_value_l452_452191


namespace roots_polynomial_sum_l452_452748

theorem roots_polynomial_sum :
  ∀ (p q r : ℂ), (p^3 - 3*p^2 - p + 3 = 0) ∧ (q^3 - 3*q^2 - q + 3 = 0) ∧ (r^3 - 3*r^2 - r + 3 = 0) →
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 1) :=
by
  intros p q r h
  sorry

end roots_polynomial_sum_l452_452748


namespace find_a_coefficient_expansion_l452_452164

theorem find_a_coefficient_expansion :
  ∃ (a : ℤ), 
    (∑ k in Finset.range 7, 
      if k = 1 then 1 else
      if k = -a * ((6.choose 2 : ℤ)) then -a * ((6.choose 2 : ℤ))
      else 0
    ) = 31 ∧ a = -2 :=
sorry

end find_a_coefficient_expansion_l452_452164


namespace ratio_of_volumes_l452_452549

-- Define the geometric properties
variables {h r : ℝ} (h_pos : 0 < h) (r_pos : 0 < r)

-- Define the volumes of the cones
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Volume of the smallest piece (Cone A)
def V_A : ℝ := volume_cone r h

-- Volume of the largest piece (Cone C)
def V_C : ℝ := volume_cone (3 * r) (3 * h)

-- The proof statement
theorem ratio_of_volumes : V_A / V_C = 1 / 27 := by
  sorry

end ratio_of_volumes_l452_452549


namespace smallest_a_divisible_by_1984_l452_452595

theorem smallest_a_divisible_by_1984 :
  ∃ a : ℕ, (∀ n : ℕ, n % 2 = 1 → 1984 ∣ (47^n + a * 15^n)) ∧ a = 1055 := 
by 
  sorry

end smallest_a_divisible_by_1984_l452_452595


namespace scientific_notation_l452_452563

noncomputable def billion := 10^9

theorem scientific_notation (n : ℝ) (h : n = 19.4 * billion) : 1.94 * 10^10 = 19.4 * billion :=
by
  sorry

end scientific_notation_l452_452563


namespace greatest_integer_less_PS_l452_452287

theorem greatest_integer_less_PS 
  (PQ PS T : ℝ)
  (midpoint_TPS : T = PS / 2)
  (perpendicular_PT_QT : (PQ ^ 2 = (PS / 2) ^ 2 + (PS / 2) ^ 2))
  (PQ_value : PQ = 150) :
  ⌊ PS ⌋ = 212 :=
by
  sorry

end greatest_integer_less_PS_l452_452287


namespace max_z_conjugate_self_proof_l452_452633

noncomputable def max_z_conjugate_self (z : ℂ) (hz : abs (z - (1 + I)) = 3) : ℝ :=
  (11 + 6 * Real.sqrt 2)

theorem max_z_conjugate_self_proof (z : ℂ) (hz : abs (z - (1 + I)) = 3) : z * conj z ≤ max_z_conjugate_self z hz :=
  sorry

end max_z_conjugate_self_proof_l452_452633


namespace rearrange_for_25_rearrange_for_1000_l452_452316

open Finset

noncomputable def canRearrange (n : ℕ) (s : Finset ℕ) : Prop :=
  ∃ lst : (Σ' m, Vector ℕ m), lst.2.toList.perm (s.toList) ∧
    (∀ i : Fin (lst.1 - 1), ((lst.2.nth i.succ) - (lst.2.nth i)) ∈ {3, 5})

theorem rearrange_for_25 : canRearrange 25 (range 1 26) :=
sorry

theorem rearrange_for_1000 : canRearrange 1000 (range 1 1001) :=
sorry

end rearrange_for_25_rearrange_for_1000_l452_452316


namespace q_f_digit_div_36_l452_452036

theorem q_f_digit_div_36 (q f : ℕ) (hq : q ≠ f) (hq_digit: q < 10) (hf_digit: f < 10) :
    (457 * 10000 + q * 1000 + 89 * 10 + f) % 36 = 0 → q + f = 6 :=
sorry

end q_f_digit_div_36_l452_452036


namespace yoki_cans_correct_l452_452601

def total_cans := 85
def ladonna_cans := 25
def prikya_cans := 2 * ladonna_cans
def yoki_cans := total_cans - ladonna_cans - prikya_cans

theorem yoki_cans_correct : yoki_cans = 10 :=
by
  sorry

end yoki_cans_correct_l452_452601


namespace angle_AED_90_degrees_length_CD_ratio_of_areas_l452_452297

-- Defining the conditions
def equilateral_triangle (A B C : Type) (AB BC CA : ℝ) := (AB = BC) ∧ (BC = CA)

def folded_triangle (A B C E F D : Type) (AB BC CA : ℝ) (DF : Type) :=
  equilateral_triangle A B C AB BC CA ∧
  ((DF ⊥ BC) ∧ (A is_folded_on BC at D))

-- The proof statements
theorem angle_AED_90_degrees (A B C E F D : Type) (AB BC CA : ℝ) 
  (h_fold : folded_triangle A B C E F D AB BC CA) :
  ∠AED = 90 := sorry

theorem length_CD (A B C E F D : Type) (AB BC CA : ℝ) 
  (h_fold : folded_triangle A B C E F D AB BC CA) :
  CD = 2 - sqrt(3) := sorry

theorem ratio_of_areas (A B C E F D : Type) (AB BC CA : ℝ) 
  (h_fold : folded_triangle A B C E F D AB BC CA) :
  area A E F / area A B C = (9 * sqrt(3) - 15) / 2 := sorry

end angle_AED_90_degrees_length_CD_ratio_of_areas_l452_452297


namespace largest_subset_size_l452_452741

def max_elements (T : Set ℕ) : ℕ :=
  if h1 : ∀ t1 t2 ∈ T, t1 ≠ t2 → (t1 ≠ t2 + 5) ∧ (t1 ≠ t2 + 8) ∧ (t2 ≠ t1 + 5) ∧ (t2 ≠ t1 + 8)
  then T.size
  else 0

theorem largest_subset_size : 
  ∃ T : Set ℕ, T ⊆ {1, ..., 2022} ∧ (∀ t1 t2 ∈ T, t1 ≠ t2 → (t1 ≠ t2 + 5) ∧ (t1 ≠ t2 + 8)) ∧ max_elements T = 778 :=
by {
  sorry
}

end largest_subset_size_l452_452741


namespace lisa_children_l452_452372

theorem lisa_children (C : ℕ) 
  (h1 : 5 * 52 = 260)
  (h2 : (2 * C + 3 + 2) * 260 = 3380) : 
  C = 4 := 
by
  sorry

end lisa_children_l452_452372


namespace total_pieces_of_junk_mail_l452_452885

-- Definition of the problem based on given conditions
def pieces_per_house : ℕ := 4
def number_of_blocks : ℕ := 16
def houses_per_block : ℕ := 17

-- Statement of the theorem to prove the total number of pieces of junk mail
theorem total_pieces_of_junk_mail :
  (houses_per_block * pieces_per_house * number_of_blocks) = 1088 :=
by
  sorry

end total_pieces_of_junk_mail_l452_452885


namespace trajectory_of_point_l452_452234

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem trajectory_of_point (F1 F2 : ℝ × ℝ) (a : ℝ) (h : a ≥ 0) :
  ∃ P : ℝ × ℝ, distance P F1 + distance P F2 = 2 * a →
  ∃ trajectory : string, 
    (trajectory = "line segment" ∨ trajectory = "ellipse" ∨ trajectory = "does not exist") :=
by
  sorry

end trajectory_of_point_l452_452234


namespace arithmetic_mean_prime_numbers_l452_452964

-- Define the list of numbers
def num_list : List ℕ := [33, 37, 39, 41, 43]

-- Define a predicate for primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Extract list of prime numbers
def prime_numbers : List ℕ := num_list.filter is_prime

-- Compute the arithmetic mean of a list of natural numbers
def arithmetic_mean (nums : List ℕ) : ℚ :=
  nums.foldr Nat.add 0 / nums.length

-- The main theorem: Proving the arithmetic mean of prime numbers in the list
theorem arithmetic_mean_prime_numbers :
  arithmetic_mean prime_numbers = 121 / 3 :=
by 
  -- Include proof steps here
  sorry

end arithmetic_mean_prime_numbers_l452_452964


namespace find_pairs_l452_452943

theorem find_pairs (x y : ℤ) (h : x^2 - nat.factorial (y.nat_abs) = 2001) :
  (x = 45 ∧ y = 4) ∨ (x = -45 ∧ y = 4) :=
sorry

end find_pairs_l452_452943


namespace slope_of_perpendicular_line_l452_452160

-- Define the line equation as a condition
def line_eqn (x y : ℝ) : Prop := 4 * x - 6 * y = 12

-- Define the slope of the given line from its equation
noncomputable def original_slope : ℝ := 2 / 3

-- Define the negative reciprocal of the original slope
noncomputable def perp_slope (m : ℝ) : ℝ := -1 / m

-- State the theorem
theorem slope_of_perpendicular_line : perp_slope original_slope = -3 / 2 :=
by 
  sorry

end slope_of_perpendicular_line_l452_452160


namespace prob_two_tangents_l452_452244

open Real

noncomputable def circle (k : ℝ): (x : ℝ) → (y : ℝ) → ℝ :=
  λ x y, x^2 + y^2 + k*x - 2*y - (5/4)*k

noncomputable def is_well_defined_circle (k : ℝ) : Prop :=
  let r_squared := 1 + (5/4)*k + (k^2)/4
  r_squared > 0

def point_A_outside_circle (k : ℝ) : Prop :=
  circle k 1 1 > 0

def valid_k_interval (k : ℝ) : Prop :=
  k ∈ Icc (-2 : ℝ) 2

def valid_tangent_interval (k : ℝ) : Prop :=
  k < -4 ∨ (-1 < k ∧ k < 0)

theorem prob_two_tangents (k : ℝ) :
  valid_tangent_interval k →
  ∑ k in (interval_integral.measure_Icc (-2 : ℝ) 2), valid_tangent_interval k / (interval_integral.measure_Icc (-2 : ℝ) 2) = 1/4 :=
sorry

end prob_two_tangents_l452_452244


namespace even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l452_452976

theorem even_ngon_parallel_edges (n : ℕ) (h : n % 2 = 0) :
  ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

theorem odd_ngon_no_two_parallel_edges (n : ℕ) (h : n % 2 = 1) :
  ¬ ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

end even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l452_452976


namespace arithmetic_sequence_middle_term_l452_452718

theorem arithmetic_sequence_middle_term (x y z : ℝ) :
  (20 + 50) / 2 = 35 →
  ∃ a b c d : ℝ, 
    a = 20 ∧ b = x ∧ c = y ∧ d = z ∧ (a + 2 * c + d) = 4 * c :=
by
  intro hy
  use [20, x, 35, z]
  repeat { split };
  try { refl };
  assumption

end arithmetic_sequence_middle_term_l452_452718


namespace number_of_ways_to_place_rooks_l452_452154

-- This statement defines a theorem for the number of ways to place 8 rooks on an 8x8 chessboard
-- such that every square is attacked by at least one rook, using the described conditions
theorem number_of_ways_to_place_rooks : 
  let n := 8 in
  let ways := (2 * n^n) - n.factorial in
  ways = 33514112 :=
by
  -- We have skipped the proof part using sorry
  sorry

end number_of_ways_to_place_rooks_l452_452154


namespace plane_tiling_with_single_line_cut_l452_452310

-- Define a tiling of the plane with rectangles
noncomputable def plane_tiling := set (set ℝ × set ℝ)

-- Define a straight-line cut in the plane
noncomputable def straight_line_cut := set (ℝ × ℝ)

-- Define a property for a tiling such that a single straight-line cut intersects each rectangle
def tiling_intersected_by_line (T : plane_tiling) (L : straight_line_cut) : Prop :=
  ∀ rectangle ∈ T, ∃ p ∈ rectangle, p ∈ L

-- Main theorem statement
theorem plane_tiling_with_single_line_cut :
  ∃ T : plane_tiling, ∃ L : straight_line_cut, tiling_intersected_by_line T L :=
sorry

end plane_tiling_with_single_line_cut_l452_452310


namespace consecutive_numbers_equation_l452_452214

theorem consecutive_numbers_equation (x y z : ℤ) (h1 : z = 3) (h2 : y = z + 1) (h3 : x = y + 1) 
(h4 : 2 * x + 3 * y + 3 * z = 5 * y + n) : n = 11 :=
by
  sorry

end consecutive_numbers_equation_l452_452214


namespace terminal_side_angle_l452_452195

theorem terminal_side_angle (θ : ℝ) (hθ : θ ∈ set.Ico 0 (2 * π)) 
  (hP : (sin (3 * π / 4), cos (3 * π / 4)) = (sin θ, cos θ)) : θ = 7 * π / 4 :=
sorry

end terminal_side_angle_l452_452195


namespace fourth_grade_students_l452_452040

theorem fourth_grade_students (initial_students : ℕ) (students_left : ℕ) (new_students : ℕ) 
  (h_initial : initial_students = 35) (h_left : students_left = 10) (h_new : new_students = 10) :
  initial_students - students_left + new_students = 35 :=
by
  -- The proof goes here
  sorry

end fourth_grade_students_l452_452040


namespace only_line_can_tile_4x7_l452_452168

-- Definitions for tetromino types
inductive Tetromino
| line : Tetromino   -- 1x4 straight line
| tee : Tetromino    -- T-shaped tetromino
| el : Tetromino     -- L-shaped tetromino
| square : Tetromino -- 2x2 square

-- Definitions for tiling conditions
def can_tile (rectangle : ℕ × ℕ) (tile : Tetromino) : Prop :=
  match tile with
  | Tetromino.line => rectangle = (4, 7)  -- Only valid tiling for 4x7 rectangle using 1x4 tetromino
  | Tetromino.tee => false                -- Impossible for T-shaped
  | Tetromino.el => false                 -- Impossible for L-shaped
  | Tetromino.square => false             -- Impossible for 2x2 square
  end

-- Main theorem statement
theorem only_line_can_tile_4x7 :
  ∀ t : Tetromino, can_tile (4, 7) t ↔ t = Tetromino.line :=
by sorry

end only_line_can_tile_4x7_l452_452168


namespace ticket_cost_correct_l452_452094

noncomputable def calculate_ticket_cost : ℝ :=
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  10 * x + 8 * child_price + 5 * senior_price

theorem ticket_cost_correct :
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  (4 * x + 3 * child_price + 2 * senior_price = 35) →
  (10 * x + 8 * child_price + 5 * senior_price = 88.75) :=
by
  intros
  sorry

end ticket_cost_correct_l452_452094


namespace prob_xi_leq_neg2_l452_452659

variable {σ : ℝ} (ξ : ℝ → ℝ)

axiom h1 : ∀ x, ξ x ~ Normal 1 σ^2
axiom h2 : Pξ (ξ ≤ 4) = 0.84

theorem prob_xi_leq_neg2 : Pξ (ξ ≤ -2) = 0.16 := by
  sorry

end prob_xi_leq_neg2_l452_452659


namespace rearrange_for_25_rearrange_for_1000_l452_452315

open Finset

noncomputable def canRearrange (n : ℕ) (s : Finset ℕ) : Prop :=
  ∃ lst : (Σ' m, Vector ℕ m), lst.2.toList.perm (s.toList) ∧
    (∀ i : Fin (lst.1 - 1), ((lst.2.nth i.succ) - (lst.2.nth i)) ∈ {3, 5})

theorem rearrange_for_25 : canRearrange 25 (range 1 26) :=
sorry

theorem rearrange_for_1000 : canRearrange 1000 (range 1 1001) :=
sorry

end rearrange_for_25_rearrange_for_1000_l452_452315


namespace equation_of_line_l_find_a_l452_452543

section
variables {x y a : ℝ}

-- Define lines l1, l2, l3 and point P
def l1 (x y : ℝ) := 2 * x - y = 1
def l2 (x y : ℝ) := x + 2 * y = 3
def l3 (x y : ℝ) := x - y + 1 = 0
def P := (1 : ℝ, 1 : ℝ)

-- Define line l passing through P and perpendicular to l3
def l (x y : ℝ) := x + y - 2 = 0

-- Define circle C and the tangent condition
def Circle (a : ℝ) := ∀ (x y : ℝ), (x - a) ^ 2 + y ^ 2 = 8
def tangent_condition (a : ℝ) := (abs (a - 2)) / Real.sqrt 2 = 2 * Real.sqrt 2

-- Proof goals
theorem equation_of_line_l : 
  l (1 : ℝ) (1 : ℝ) ∧ ∀ (x y : ℝ), l x y ↔ (x = y - 1) :=
sorry

theorem find_a (a : ℝ) (h1 : Circle a) (h2 : tangent_condition a) (h3 : a > 0) : 
  a = 6 :=
sorry
end

end equation_of_line_l_find_a_l452_452543


namespace value_of_series_l452_452127

theorem value_of_series :
  (∑ k in Finset.range 2022, (2023 - (k + 1)) / (k + 1)) / (∑ k in Finset.Ico 2 2024, 1 / k) = 2023 :=
by
  sorry

end value_of_series_l452_452127


namespace campers_went_rowing_and_hiking_in_all_l452_452044

def C_rm : Nat := 41
def C_hm : Nat := 4
def C_ra : Nat := 26

theorem campers_went_rowing_and_hiking_in_all : (C_rm + C_ra) + C_hm = 71 :=
by
  sorry

end campers_went_rowing_and_hiking_in_all_l452_452044


namespace relationship_among_abc_l452_452746

noncomputable def a : ℝ := 7 ^ 0.3
noncomputable def b : ℝ := 0.3 ^ 7
noncomputable def c : ℝ := Real.logBase 7 0.3

theorem relationship_among_abc : c < b ∧ b < a := by
  sorry

end relationship_among_abc_l452_452746


namespace product_of_midpoint_is_minus_4_l452_452480

-- Coordinates of the endpoints
def endpoint1 : ℝ × ℝ := (4, -3)
def endpoint2 : ℝ × ℝ := (-8, 7)

-- Function to compute the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Coordinates of the midpoint
def midpoint_coords := midpoint endpoint1 endpoint2

-- Product of the coordinates of the midpoint
def product_of_midpoint_coords (mp : ℝ × ℝ) : ℝ :=
  mp.1 * mp.2

-- Statement of the theorem to be proven
theorem product_of_midpoint_is_minus_4 : 
  product_of_midpoint_coords midpoint_coords = -4 := 
by
  sorry

end product_of_midpoint_is_minus_4_l452_452480


namespace smallest_multiple_9_and_6_l452_452501

theorem smallest_multiple_9_and_6 : ∃ n : ℕ, n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m :=
by
  have h := Nat.lcm 9 6
  use h
  split
  sorry

end smallest_multiple_9_and_6_l452_452501


namespace smallest_angle_A2B2C2_l452_452735

noncomputable def smallest_angle_ABC := 40

theorem smallest_angle_A2B2C2 (ABC : Triangle) (Ω : Circle) 
  (A1 B1 C1 A2 B2 C2 : Point) (A B C : Angle) 
  (h1 : internal_bisector (angle_A ABC) A Ω = A1)
  (h2 : internal_bisector (angle_B ABC) B Ω = B1)
  (h3 : internal_bisector (angle_C ABC) C Ω = C1)
  (h4 : internal_bisector (angle_A A1A2A3) A1 Ω = A2)
  (h5 : internal_bisector (angle_B B1B2B3) B1 Ω = B2)
  (h6 : internal_bisector (angle_C C1C2C3) C1 Ω = C2)
  (h7 : smallest_angle_ABC = 40) : 
  smallest_angle (triangle_A2B2C2 A2 B2 C2) = 65 :=
by 
  sorry

end smallest_angle_A2B2C2_l452_452735


namespace probability_more_than_70_l452_452710

-- Definitions based on problem conditions
def P_A : ℝ := 0.15
def P_B : ℝ := 0.45
def P_C : ℝ := 0.25

-- Theorem to state that the probability of scoring more than 70 points is 0.85
theorem probability_more_than_70 (hA : P_A = 0.15) (hB : P_B = 0.45) (hC : P_C = 0.25):
  P_A + P_B + P_C = 0.85 :=
by
  rw [hA, hB, hC]
  sorry

end probability_more_than_70_l452_452710


namespace Juwella_reads_pages_l452_452137

theorem Juwella_reads_pages (p1 p2 p3 p_total p_tonight : ℕ) 
                            (h1 : p1 = 15)
                            (h2 : p2 = 2 * p1)
                            (h3 : p3 = p2 + 5)
                            (h4 : p_total = 100) 
                            (h5 : p_total = p1 + p2 + p3 + p_tonight) :
  p_tonight = 20 := 
sorry

end Juwella_reads_pages_l452_452137


namespace _l452_452189

-- Part (1): Proving the equation of the ellipse E
noncomputable def part1_equation_of_ellipse (a b c : ℝ) (x y : ℝ) (h1 : a = 2) (h2 : b * c = Real.sqrt 3) (h3 : a > b) (h4 : b > c) (h5 : c > 0) : Prop :=
  ((x / 2)^2 + (y / Real.sqrt(3))^2 = 1)

-- Part (2): Proving the existence of the line l
noncomputable def part2_existence_of_line (x y k : ℝ) (h6 : (x = 2) ∧ (y = 1)) (h7 : k = 1 / 2) (h8 : (x - 0)^2 + (y - 0)^2 = 4 * ((x - 2)^2 + (y - k * (x - 2) + 1)^2)) : Prop :=
  (y = k * x)

-- Final problem statements for Lean theorem prover
noncomputable def problem_statement : Prop :=
  part1_equation_of_ellipse 2 (Real.sqrt 3) 1 x y (by rfl) (by { simp, ring }) (by linarith) (by linarith) (by linarith) ∧
  part2_existence_of_line 2 1 (1 / 2) (by exact ⟨rfl, rfl⟩) (by rfl) sorry

end _l452_452189


namespace perp_condition_l452_452194

section

variables {a : ℝ}

def l1 (a : ℝ) : ℝ × ℝ → Prop := λ p, p.1 + (a - 2) * p.2 - 2 = 0
def l2 (a : ℝ) : ℝ × ℝ → Prop := λ p, (a - 2) * p.1 + a * p.2 - 1 = 0

theorem perp_condition (a : ℝ) : (∀ p q : ℝ × ℝ, l1 a p → l2 a q → (a = -1 → 
  (p.1 * q.1 + p.2 * q.2 = 0)) ∧ (p.1 * q.1 + p.2 * q.2 = 0 → (a = -1 ∨ a = 2))) :=
by
  sorry

end

end perp_condition_l452_452194


namespace outfit_combination_count_l452_452240

theorem outfit_combination_count (c : ℕ) (s p h sh : ℕ) (c_eq_6 : c = 6) (s_eq_c : s = c) (p_eq_c : p = c) (h_eq_c : h = c) (sh_eq_c : sh = c) :
  (c^4) - c = 1290 :=
by
  sorry

end outfit_combination_count_l452_452240


namespace no_such_function_exists_l452_452590

noncomputable def func_a (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ ∀ n : ℕ, a n = n - a (a n)

theorem no_such_function_exists : ¬ ∃ a : ℕ → ℕ, func_a a :=
by
  sorry

end no_such_function_exists_l452_452590


namespace third_altitude_less_than_30_l452_452825

theorem third_altitude_less_than_30 (ha hb : ℝ) (h₁ : ha = 12) (h₂ : hb = 20) : ∃ hc : ℝ, hc < 30 :=
by
  use 30 - epsilon,
  sorry

end third_altitude_less_than_30_l452_452825


namespace intersection_A_B_l452_452202

open Set

def SetA : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def SetB : Set ℤ := {x | 0 ≤ x ∧ x ≤ 4}

theorem intersection_A_B :
  (SetA ∩ SetB) = ( {0, 2, 4} : Set ℤ ) :=
by
  sorry

end intersection_A_B_l452_452202


namespace unique_center_l452_452750

noncomputable def is_center (P : ℝ × ℝ × ℝ) (cube_center : ℝ × ℝ × ℝ) : Prop :=
  P = cube_center

noncomputable def equal_distances_to_planes (P : ℝ × ℝ × ℝ)
  (plane_ABC plane_ABA1 plane_ADA1 : ℝ × ℝ × ℝ → ℝ) : Prop :=
  plane_ABC P = plane_ABA1 P ∧ plane_ABA1 P = plane_ADA1 P

theorem unique_center 
  (P : ℝ × ℝ × ℝ) 
  (cube_center : ℝ × ℝ × ℝ)
  (plane_ABC plane_ABA1 plane_ADA1 : ℝ × ℝ × ℝ → ℝ)
  (H : ∀ P, P ∈ diagonal_face_BDD1B1 → equal_distances_to_planes P plane_ABC plane_ABA1 plane_ADA1) :
  ∃! P, P ∈ diagonal_face_BDD1B1 ∧ equal_distances_to_planes P plane_ABC plane_ABA1 plane_ADA1 := 
sorry

end unique_center_l452_452750


namespace find_k_value_l452_452740

theorem find_k_value (a : ℕ → ℕ) (k : ℕ) (S : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 3 = 5) 
  (h₃ : S (k + 2) - S k = 36) : 
  k = 8 := 
by 
  sorry

end find_k_value_l452_452740


namespace pascal_triangle_43rd_element_in_51_row_l452_452022

theorem pascal_triangle_43rd_element_in_51_row :
  (Nat.choose 50 42) = 10272278170 :=
  by
  -- proof construction here
  sorry

end pascal_triangle_43rd_element_in_51_row_l452_452022


namespace floor_add_eq_20_25_l452_452144

theorem floor_add_eq_20_25 (r : Real) (hn : Real.floor r = ⌊r⌋) (hf : r = ⌊r⌋ + (r - ⌊r⌋)) :
  ⌊r⌋ + r = 20.25 ↔ r = 10.25 :=
by
  sorry

end floor_add_eq_20_25_l452_452144


namespace potatoes_yield_l452_452387

theorem potatoes_yield (steps_length : ℕ) (steps_width : ℕ) (step_size : ℕ) (yield_per_sqft : ℚ) 
  (h_steps_length : steps_length = 18) 
  (h_steps_width : steps_width = 25) 
  (h_step_size : step_size = 3) 
  (h_yield_per_sqft : yield_per_sqft = 1/3) 
  : (steps_length * step_size) * (steps_width * step_size) * yield_per_sqft = 1350 := 
by 
  sorry

end potatoes_yield_l452_452387


namespace logarithmic_identity_l452_452201

noncomputable def log2 (x : ℝ) : ℝ := log x / log 2
noncomputable def log3 (x : ℝ) : ℝ := log x / log 3
noncomputable def log6 (x : ℝ) : ℝ := log x / log 6

theorem logarithmic_identity (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_cond : log6 (2 * a + 3 * b) = log3 b + log6 9 - 1 ∧ log6 (2 * a + 3 * b) = log2 a + log6 9 - log2 3) :
  log10 (2 * a + 3 * b) - log10 (10 * a) - log10 (10 * b) = -2 :=
by
  sorry

end logarithmic_identity_l452_452201


namespace monotonicity_f_max_m_l452_452226

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + x
def g (x : ℝ) (m : ℝ) : ℝ := (x - 2) * Real.exp x - x^2 + m

-- Monotonicity problem for f
theorem monotonicity_f (a : ℝ) (h_le : a ≤ 0) : 
  if a ≤ -1/8 then 
    ∀ x : ℝ, 0 < x → (f a)' x ≤ 0
  else 
    ∀ x : ℝ, 
      ((0 < x ∧ x < (1 - Real.sqrt (1 + 8 * a)) / 4) → (f a)' x < 0) ∧ 
      ((x = (1 - Real.sqrt (1 + 8 * a)) / 4 ∨ x = (1 + Real.sqrt (1 + 8 * a)) / 4) → (f a)' x = 0) ∧
      ((x > (1 - Real.sqrt (1 + 8 * a)) / 4 ∧ x < (1 + Real.sqrt (1 + 8 * a)) / 4) → (f a)' x > 0) ∧ 
      ((x > (1 + Real.sqrt (1 + 8 * a)) / 4) → (f a)' x < 0)
:= sorry

-- Maximum value of m for f(x) > g(x)
theorem max_m (m : ℝ) : 
  ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f (-1) x > g x m → m ≤ 3
:= sorry

end monotonicity_f_max_m_l452_452226


namespace number_of_colorings_l452_452336

theorem number_of_colorings (colors : Finset ℕ) (adj : (ℕ × ℕ) → (ℕ × ℕ) → Prop):
  colors.card = 3 →
  (∀ (x y : ℕ × ℕ), adj x y → x ≠ y → ∀ (c1 c2 : ℕ), c1 ∈ colors → c2 ∈ colors → c1 ≠ c2) →
  ∃ (f : (ℕ × ℕ) → ℕ), (∀ (x y : ℕ × ℕ), adj x y → f x ≠ f y) ∧ (f '' { (i,j) | i < 3 ∧ j < 3 }).card = 3^3 →
  (∃ (count : ℕ), count = 256) :=
begin
  sorry
end

end number_of_colorings_l452_452336


namespace apples_in_baskets_l452_452388

theorem apples_in_baskets (total_apples : ℕ) (first_basket : ℕ) (increase : ℕ) (baskets : ℕ) :
  total_apples = 495 ∧ first_basket = 25 ∧ increase = 2 ∧
  (total_apples = (baskets / 2) * (2 * first_basket + (baskets - 1) * increase)) -> baskets = 13 :=
by sorry

end apples_in_baskets_l452_452388


namespace smallest_n_for_integer_y_l452_452113

def y (n : ℕ) : ℝ :=
  if n = 1 then (4 : ℝ) ^ (1 / 4)
  else y (n - 1) ^ (4 : ℝ) ^ (1 / 4)

theorem smallest_n_for_integer_y :
  ∃ n : ℕ, n = 4 ∧ y n ∈ ℤ :=
by
  sorry

end smallest_n_for_integer_y_l452_452113


namespace total_cost_of_horse_and_saddle_l452_452056

noncomputable def saddle_cost : ℝ := 1000
noncomputable def horse_cost : ℝ := 4 * saddle_cost
noncomputable def total_cost : ℝ := saddle_cost + horse_cost

theorem total_cost_of_horse_and_saddle :
    total_cost = 5000 := by
  sorry

end total_cost_of_horse_and_saddle_l452_452056


namespace solution_set_inequality_l452_452759

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Conditions
axiom differentiable_on_f : IsDifferentiableOn ℝ f (set.Iio 0)
axiom derivative_f' : ∀ (x : ℝ), f' x = deriv f x
axiom condition_f : ∀ (x : ℝ), x ∈ set.Iio 0 → 3 * f x + x * f' x < 0

-- Statement of the proof problem
theorem solution_set_inequality (x : ℝ) : 
  ( ( (x + 2016) ^ 3 ) * f (x + 2016) + 8 * f (-2) < 0 ) ↔ x ∈ set.Ioo (-2018) (-2016) :=
  sorry

end solution_set_inequality_l452_452759


namespace digit_H_value_l452_452808

theorem digit_H_value (E F G H : ℕ) (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (cond1 : 10 * E + F + 10 * G + E = 10 * H + E)
  (cond2 : 10 * E + F - (10 * G + E) = E)
  (cond3 : E + G = H + 1) : H = 8 :=
sorry

end digit_H_value_l452_452808


namespace num_elements_B_l452_452670

def A : Set ℚ := {1, 2, 3, 4, 6}
def B : Set ℚ := {z | ∃ x ∈ A, ∃ y ∈ A, z = x / y}

theorem num_elements_B : B.toFinset.card = 13 := by
  sorry

end num_elements_B_l452_452670


namespace num_valid_permutations_l452_452354

-- Define the sequence number (seq_num)
def seq_num (a : List ℕ) (i : ℕ) : ℕ :=
  (List.filter (λ j, j < a.nthLe i sorry) (a.take i)).length

-- Define the conditions for the permutation
def is_valid_permutation (a : List ℕ) : Prop :=
  a.perm (List.range 1 9) ∧  -- a is a permutation of [1, 2, ..., 8]
  seq_num a 2 = 2 ∧  -- sequence number of 8 is 2 (a[2] = 8)
  seq_num a 4 = 3 ∧  -- sequence number of 7 is 3 (a[4] = 7)
  seq_num a 3 = 3    -- sequence number of 5 is 3 (a[3] = 5)

-- The main proof statement
theorem num_valid_permutations : 
  ∃ n, (∃ (l : List (List ℕ)), (∀ a, a ∈ l ↔ is_valid_permutation a) ∧ l.length = n) ∧ n = 144 :=
sorry

end num_valid_permutations_l452_452354


namespace quadrilateral_diagonals_perpendicular_cyclic_feet_of_perpendiculars_l452_452779

variables {α : Type*} [normed_group α] [normed_space ℝ α] [inner_product_space ℝ α]

structure ConvexQuadrilateral (P : ℕ → α) := 
  (convex : convex ℝ (λ i, P i) {0,1,2,3})
  (diagonals_perpendicular : inner (P 0 - P 2) (P 1 - P 3) = 0)

-- Define the feet of perpendiculars dropped from intersection point of diagonals
noncomputable def feet_of_perpendiculars (P : ℕ → α) : fin 4 → α :=
λ i, projection (line_span ℝ {P i, P ((i+1) % 4)}) (P 0 + P 1 - P 2 - P 3)

theorem quadrilateral_diagonals_perpendicular_cyclic_feet_of_perpendiculars 
  (P : ℕ → α) (hP : ConvexQuadrilateral P) :
  ∃ C, ∀ i ∈ (finset.univ : finset (fin 4)), (feet_of_perpendiculars P i) ∈ C :=
sorry

end quadrilateral_diagonals_perpendicular_cyclic_feet_of_perpendiculars_l452_452779


namespace q1_correct_q2_correct_l452_452920

-- Defining the necessary operations
def q1_lhs := 8 / (-2) - (-4) * (-3)
def q2_lhs := (-2) ^ 3 / 4 * (5 - (-3) ^ 2)

-- Theorem statements to prove that they are equal to 8
theorem q1_correct : q1_lhs = 8 := sorry
theorem q2_correct : q2_lhs = 8 := sorry

end q1_correct_q2_correct_l452_452920


namespace percentage_increase_painting_l452_452765

/-
Problem:
Given:
1. The original cost of jewelry is $30 each.
2. The original cost of paintings is $100 each.
3. The new cost of jewelry is $40 each.
4. The new cost of paintings is $100 + ($100 * P / 100).
5. A buyer purchased 2 pieces of jewelry and 5 paintings for $680.

Prove:
The percentage increase in the cost of each painting (P) is 20%.
-/

theorem percentage_increase_painting (P : ℝ) :
  let jewelry_price := 30
  let painting_price := 100
  let new_jewelry_price := 40
  let new_painting_price := 100 * (1 + P / 100)
  let total_cost := 2 * new_jewelry_price + 5 * new_painting_price
  total_cost = 680 → P = 20 := by
sorry

end percentage_increase_painting_l452_452765


namespace arithmetic_mean_of_primes_l452_452950

-- Define the list of numbers
def num_list : List ℕ := [33, 37, 39, 41, 43]

-- Define a predicate that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the list of prime numbers extracted from the list
def prime_list : List ℕ := (num_list.filter is_prime)

-- Define the sum of the prime numbers
def prime_sum : ℕ := prime_list.foldr (· + ·) 0

-- Define the count of prime numbers
def prime_count : ℕ := prime_list.length

-- Define the arithmetic mean of the prime numbers
def prime_mean : ℚ := prime_sum / prime_count

theorem arithmetic_mean_of_primes :
  prime_mean = 40 + 1 / 3 := sorry

end arithmetic_mean_of_primes_l452_452950


namespace arithmetic_mean_of_primes_l452_452960

open Real

theorem arithmetic_mean_of_primes :
  let numbers := [33, 37, 39, 41, 43]
  let primes := numbers.filter Prime
  let sum_primes := (37 + 41 + 43 : ℤ)
  let count_primes := (3 : ℤ)
  let mean := (sum_primes / count_primes : ℚ)
  mean = 40.33 := by
sorry

end arithmetic_mean_of_primes_l452_452960


namespace f_sqrt_45_l452_452363

noncomputable def f : ℝ → ℝ :=
  λ x, if x = floor x then 7 * x + 3 else floor x + 7

theorem f_sqrt_45 : f (sqrt 45) = 13 :=
by 
  sorry

end f_sqrt_45_l452_452363


namespace find_n_l452_452150

theorem find_n :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -567 [MOD 13] :=
by
  use 5
  apply And.intro
  apply le_refl
  apply And.intro
  apply le_of_lt
  exact (show 5 < 13, from dec_trivial)
  apply int.mod_eq_of_lt
  linarith
  exact (show -567 % 13 = 5, from sorry)

end find_n_l452_452150


namespace distinct_fib_sum_2017_l452_452119

-- Define the Fibonacci sequence as given.
def fib : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => (fib (n+1)) + (fib n)

-- Define the predicate for representing a number as a sum of distinct Fibonacci numbers.
def can_be_written_as_sum_of_distinct_fibs (n : ℕ) : Prop :=
  ∃ s : Finset ℕ, (s.sum fib = n) ∧ (∀ (i j : ℕ), i ≠ j → i ∉ s → j ∉ s)

theorem distinct_fib_sum_2017 : ∃! s : Finset ℕ, s.sum fib = 2017 ∧ (∀ (i j : ℕ), i ≠ j → i ≠ j → i ∉ s → j ∉ s) :=
sorry

end distinct_fib_sum_2017_l452_452119


namespace probability_sum_5_l452_452546

theorem probability_sum_5 :
  let total_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
by
  -- proof omitted
  sorry

end probability_sum_5_l452_452546


namespace mary_more_than_marco_l452_452380

def marco_initial : ℕ := 24
def mary_initial : ℕ := 15
def half_marco : ℕ := marco_initial / 2
def mary_after_give : ℕ := mary_initial + half_marco
def mary_after_spend : ℕ := mary_after_give - 5
def marco_final : ℕ := marco_initial - half_marco

theorem mary_more_than_marco :
  mary_after_spend - marco_final = 10 := by
  sorry

end mary_more_than_marco_l452_452380


namespace sine_tangent_not_possible_1_sine_tangent_not_possible_2_l452_452309

theorem sine_tangent_not_possible_1 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.27413 ∧ Real.tan θ = 0.25719) :=
sorry

theorem sine_tangent_not_possible_2 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.25719 ∧ Real.tan θ = 0.27413) :=
sorry

end sine_tangent_not_possible_1_sine_tangent_not_possible_2_l452_452309


namespace cos_angle_bisector_l452_452304

theorem cos_angle_bisector
  (P Q R S : Type)
  [metric_space P] [metric_space Q] [metric_space R]
  (dist_PQ : dist P Q = 4)
  (dist_PR : dist P R = 9)
  (dist_QR : dist Q R = 12)
  (on_bisector : ∀ S, S ∈ line_segment Q R ∧ angle_bisector P Q R S) :
  cos (angle P Q S) = 5 / 12 :=
sorry

end cos_angle_bisector_l452_452304


namespace positive_difference_between_A_and_B_l452_452930

def A : ℕ := ∑ i in finset.range 20, (2*i+2) * (2*i+3) + 40
def B : ℕ := 3 + ∑ i in finset.range 19, (2*i+3) * (2*i+4) + 42

theorem positive_difference_between_A_and_B :
  |A - B| = 77 := by
  sorry

end positive_difference_between_A_and_B_l452_452930


namespace min_value_expr_l452_452917

-- Define the given expression
def given_expr (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x) + 200

-- Define the minimum value we need to prove
def min_value : ℝ :=
  -6290.25

-- The statement of the theorem
theorem min_value_expr :
  ∃ x : ℝ, ∀ y : ℝ, given_expr y ≥ min_value := by
  sorry

end min_value_expr_l452_452917


namespace tom_total_score_l452_452450

theorem tom_total_score (points_per_enemy : ℕ) (enemy_count : ℕ) (minimum_enemies_for_bonus : ℕ) (bonus_rate : ℝ) 
(initial_points : ℕ) (bonus_points : ℝ) (total_points : ℝ) :
  points_per_enemy = 10 → 
  enemy_count = 150 → 
  minimum_enemies_for_bonus = 100 → 
  bonus_rate = 0.5 → 
  initial_points = points_per_enemy * enemy_count →
  bonus_points = if enemy_count ≥ minimum_enemies_for_bonus then initial_points * bonus_rate else 0 →
  total_points = initial_points + bonus_points →
  total_points = 2250 :=
by
  sorry

end tom_total_score_l452_452450


namespace find_p_plus_q_l452_452801

theorem find_p_plus_q : ∃ (p q : ℕ), 
  let r := λ r : ℝ, 14 / (2 * (Real.sqrt 3 + 1)) := 
  let simplified_ratio := 1 / 2 * (Real.sqrt (7 * 7 * 3) - 7) in
  simplified_ratio = r →
  p + q = 154 :=
begin
  sorry
end

end find_p_plus_q_l452_452801


namespace find_m_of_parabola_and_line_l452_452229

theorem find_m_of_parabola_and_line (k m x1 x2 : ℝ) 
  (h_parabola_line : ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2} → 
                                   y = k * x + m → true)
  (h_intersection : x1 * x2 = -4) : m = 1 := 
sorry

end find_m_of_parabola_and_line_l452_452229


namespace exists_k_l452_452008

def satisfies_condition (a b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (a n - a (n-1)) * (a n - a (n-2)) + (b n - b (n-1)) * (b n - b (n-2)) = 0

theorem exists_k (a b : ℕ → ℤ) 
  (h : satisfies_condition a b) : 
  ∃ k : ℕ, k > 0 ∧ a k = a (k + 2008) :=
sorry

end exists_k_l452_452008


namespace product_of_lengths_l452_452634

noncomputable def convex_polygon (A : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
∀ i j k : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → 1 ≤ k ∧ k ≤ n → 
  (segment A i A (i+1 % n)).is_line_segment (A j) (A k) 

def condition_points (A B D : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
  (B i).x > (A i).x ∧ (B i).y > (A i).y ∧
   (D (i + 1 % n)).x < (A (i + 1 % n)).x ∧
   (D (i + 1 % n)).y < (A (i + 1 % n)).y 

def parallelograms (A B C D : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
  (C i).x = (B i).x + (D i).x - (A i).x ∧
   (C i).y = (B i).y + (D i).y - (A i).y

def intersect_at_single_point (A C : ℕ → ℝ × ℝ) (O : ℝ × ℝ) (n : ℕ) : Prop :=
∀ i j : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n →
  (A i).x + (C i).x = 2 * O.x ∧
  (A i).y + (C i).y = 2 * O.y

theorem product_of_lengths (A B D : ℕ → ℝ × ℝ) (n : ℕ) (O : ℝ × ℝ) :
  convex_polygon A n →
  condition_points A B D n →
  parallelograms A B (λ i, (B i).x + (D i).x - (A i).x, (B i).y + (D i).y - (A i).y) D n →
  intersect_at_single_point A (λ i, (B i).x + (D i).x - (A i).x, (B i).y + (D i).y - (A i).y) O n →
  (∏ i in range n, dist (A i) (B i)) = (∏ i in range n, dist (A i) (D i)) :=
sorry

end product_of_lengths_l452_452634


namespace minimum_marked_cells_l452_452471

theorem minimum_marked_cells (n : ℕ) (h : n = 7) : 
  ∃ (s : set (ℕ × ℕ)),
    (∀ (i j : ℕ), i ≤ 15 → j ≤ 15 → 
      (s.count (i, j) ≥ 2)) ∧ s.card = 28 := sorry

end minimum_marked_cells_l452_452471


namespace particle_position_2023_l452_452547

-- Define the structure and movement function

structure Position where
  x : ℕ
  y : ℕ

def moveSquare (pos : Position) (n : ℕ) : Position :=
  -- Computes the new position after a square of size n is moved
  -- This function should include the detailed perimeter and diagonal movements
  sorry -- placeholder for the actual implementation

def totalTime (n : ℕ) : ℕ :=
  4 * n + (⌈ n * Real.sqrt 2 ⌉).toNat

def particlePosition (time : ℕ) : Position :=
  -- Computes the final position after certain minutes
  sorry -- placeholder for the actual implementation

-- Main Theorem
theorem particle_position_2023 : particlePosition 2023 = Position.mk 59 57 :=
by
  sorry -- placeholder for proof

end particle_position_2023_l452_452547


namespace triangle_ratio_correct_l452_452261

theorem triangle_ratio_correct
  (A B C : Type) [HasAngle ABC] [HasLength ABC] [HasArea ABC]
  (angle_A : ang ABC = 60)
  (b_eq_1 : b ABC = 1)
  (area_sqrt3 : area ABC = sqrt 3) :
  (a ABC + b ABC + c ABC) / (sin (angle A) + sin (angle B) + sin (angle C)) = (2 / 3) * sqrt 39 := 
  sorry

end triangle_ratio_correct_l452_452261


namespace new_average_age_l452_452414

theorem new_average_age (n : ℕ) (avg_old : ℕ) (new_person_age : ℕ) (new_avg_age : ℕ)
  (h1 : avg_old = 14)
  (h2 : n = 9)
  (h3 : new_person_age = 34)
  (h4 : new_avg_age = 16) :
  (n * avg_old + new_person_age) / (n + 1) = new_avg_age :=
sorry

end new_average_age_l452_452414


namespace alcohol_replacement_percentage_l452_452057

namespace WhiskyProblem

def initial_alcohol_percentage := 0.40
def replacement_fraction := 2 / 3
def final_alcohol_percentage := 0.26
def percentage_to_prove := 0.19

theorem alcohol_replacement_percentage :
  let remaining_alcohol := (1 - replacement_fraction) * initial_alcohol_percentage
  let added_alcohol := replacement_fraction * percentage_to_prove
  remaining_alcohol + added_alcohol = final_alcohol_percentage :=
  sorry

end WhiskyProblem

end alcohol_replacement_percentage_l452_452057


namespace manny_marbles_l452_452445

theorem manny_marbles (total_marbles : ℕ) (ratio_m : ℕ) (ratio_n : ℕ) (manny_gives : ℕ) 
  (h_total : total_marbles = 36) (h_ratio_m : ratio_m = 4) (h_ratio_n : ratio_n = 5) (h_manny_gives : manny_gives = 2) : 
  (total_marbles * ratio_n / (ratio_m + ratio_n)) - manny_gives = 18 :=
by
  sorry

end manny_marbles_l452_452445


namespace smaller_of_two_digit_product_l452_452428

theorem smaller_of_two_digit_product (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha' : a < 100) (hb' : b < 100) 
  (hprod : a * b = 4680) : min a b = 52 :=
by
  sorry

end smaller_of_two_digit_product_l452_452428


namespace permutation_exists_25_permutation_exists_1000_l452_452324

-- Define a function that checks if a permutation satisfies the condition
def valid_permutation (perm : List ℕ) : Prop :=
  (∀ i < perm.length - 1, let diff := (perm[i] - perm[i+1]).abs in diff = 3 ∨ diff = 5)

-- Proof problem for n = 25
theorem permutation_exists_25 : 
  ∃ perm : List ℕ, perm.perm (List.range 25).map (· + 1) ∧ valid_permutation perm := 
sorry

-- Proof problem for n = 1000
theorem permutation_exists_1000 : 
  ∃ perm : List ℕ, perm.perm (List.range 1000).map (· + 1) ∧ valid_permutation perm := 
sorry

end permutation_exists_25_permutation_exists_1000_l452_452324


namespace compute_expression_l452_452245

-- Given condition
def condition (x : ℝ) : Prop := x + 1/x = 3

-- Theorem to prove
theorem compute_expression (x : ℝ) (hx : condition x) : (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 8 := 
by
  sorry

end compute_expression_l452_452245


namespace find_a_range_l452_452206

variable (a k : ℝ)
variable (x : ℝ) (hx : x > 0)

def p := ∀ x > 0, x + a / x ≥ 2
def q := ∀ k : ℝ, ∃ x y : ℝ, k * x - y + 2 = 0 ∧ x^2 + y^2 / a^2 = 1

theorem find_a_range :
  (a > 0) →
  ((p a) ∨ (q a)) ∧ ¬ ((p a) ∧ (q a)) ↔ 1 ≤ a ∧ a < 2 :=
sorry

end find_a_range_l452_452206


namespace mary_has_more_l452_452378

theorem mary_has_more (marco_initial mary_initial : ℕ) (h1 : marco_initial = 24) (h2 : mary_initial = 15) :
  let marco_final := marco_initial - 12,
      mary_final := mary_initial + 12 - 5 in
  mary_final = marco_final + 10 :=
by
  sorry

end mary_has_more_l452_452378


namespace prove_x1_x2_greater_than_two_l452_452179

def f (x a : ℝ) : ℝ := x * Real.log x - a * x^2 + (2 * a - 1) * x

def g (x a : ℝ) : ℝ := Real.log x - 2 * a * x + 2 * a

def g' (x a : ℝ) : ℝ := (1 - 2 * a * x) / x

theorem prove_x1_x2_greater_than_two 
  (a t x1 x2 : ℝ) 
  (h_a_leq_zero : a ≤ 0) 
  (h_t : -1 < t ∧ t < 0) 
  (h_g_monotonic : ∀ x, x > 0 → (g' x a > 0))
  (h_intersects : f x1 a = t ∧ f x2 a = t)
  (h_x1_lt_x2 : x1 < x2) :
  x1 + x2 > 2 :=
sorry

end prove_x1_x2_greater_than_two_l452_452179


namespace money_left_after_purchases_l452_452728

variable (initial_money : ℝ) (fraction_for_cupcakes : ℝ) (money_spent_on_milkshake : ℝ)

theorem money_left_after_purchases (h_initial : initial_money = 10)
  (h_fraction : fraction_for_cupcakes = 1/5)
  (h_milkshake : money_spent_on_milkshake = 5) :
  initial_money - (initial_money * fraction_for_cupcakes) - money_spent_on_milkshake = 3 := 
by
  sorry

end money_left_after_purchases_l452_452728


namespace implicit_derivative_l452_452614

noncomputable section

open Real

section ImplicitDifferentiation

variable {x : ℝ} {y : ℝ → ℝ}

def f (x y : ℝ) : ℝ := y^2 + x^2 - 1

theorem implicit_derivative (h : f x (y x) = 0) :
  deriv y x = -x / y x :=
  sorry

end ImplicitDifferentiation

end implicit_derivative_l452_452614


namespace angle_between_a_and_b_is_zero_l452_452360

section problem

variables {a b c : ℝ^3}
-- Given conditions
variable (unit_a : ‖a‖ = 1)
variable (unit_b : ‖b‖ = 1)
variable (unit_c : ‖c‖ = 1)
variable (vec_sum : a + b + 2 * c = 0)

-- Proof statement: the angle between a and b is 0 degrees.
theorem angle_between_a_and_b_is_zero :
  real.angle a b = 0 :=
sorry

end problem

end angle_between_a_and_b_is_zero_l452_452360


namespace juwella_reads_pages_l452_452134

theorem juwella_reads_pages :
  let pages_three_nights_ago := 15 in
  let pages_two_nights_ago := 2 * pages_three_nights_ago in
  let pages_last_night := pages_two_nights_ago + 5 in
  let total_pages := 100 in
  let pages_read_so_far := pages_three_nights_ago + pages_two_nights_ago + pages_last_night in
  let pages_remaining := total_pages - pages_read_so_far in
  pages_remaining = 20 :=
by
  sorry

end juwella_reads_pages_l452_452134


namespace tree_planting_problem_l452_452448

noncomputable def total_trees_needed (length width tree_distance : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let intervals := perimeter / tree_distance
  intervals

theorem tree_planting_problem : total_trees_needed 150 60 10 = 42 :=
by
  sorry

end tree_planting_problem_l452_452448


namespace quadratic_poly_product_ge_one_l452_452065

variable {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1)
variable {n : ℕ} {x : Fin n → ℝ} (hx_pos : ∀ i, x i > 0) (hx_prod : (∏ i, x i) = 1)

theorem quadratic_poly_product_ge_one :
  (∏ i, a * (x i)^2 + b * (x i) + c) ≥ 1 :=
sorry

end quadratic_poly_product_ge_one_l452_452065


namespace rearrange_for_25_rearrange_for_1000_l452_452318

open Finset

noncomputable def canRearrange (n : ℕ) (s : Finset ℕ) : Prop :=
  ∃ lst : (Σ' m, Vector ℕ m), lst.2.toList.perm (s.toList) ∧
    (∀ i : Fin (lst.1 - 1), ((lst.2.nth i.succ) - (lst.2.nth i)) ∈ {3, 5})

theorem rearrange_for_25 : canRearrange 25 (range 1 26) :=
sorry

theorem rearrange_for_1000 : canRearrange 1000 (range 1 1001) :=
sorry

end rearrange_for_25_rearrange_for_1000_l452_452318


namespace connected_if_any_town_closed_l452_452440

open Finset Fintype

-- Definitions and Conditions
def towns := Fin 1000
def d : towns → ℕ := sorry

axiom degrees_non_decreasing : ∀ i j : towns, i ≤ j → d i ≤ d j
axiom degrees_bound : ∀ j : ℕ, j < 999 - d (⟨999, sorry⟩ : towns) → d (⟨j, sorry⟩ : towns) ≥ j + 1

-- Theorem to be proved
theorem connected_if_any_town_closed (k : towns) : 
  ∀ i j : towns, i ≠ k ∧ j ≠ k → 
  (∃ (G : SimpleGraph towns), G.Adj.symm ∧ 
   (∀ i : towns, G.degree i = d i) ∧ 
   ∀ (i j : towns), G.Connected if k ≠ i ∧ j ≠ k) :=
sorry

end connected_if_any_town_closed_l452_452440


namespace unit_vector_same_direction_l452_452197

-- Define the coordinates of points A and B as given in the conditions
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Define the vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of vector AB
noncomputable def magnitudeAB : ℝ := Real.sqrt (vectorAB.1^2 + vectorAB.2^2)

-- Define the unit vector in the direction of AB
noncomputable def unitVectorAB : ℝ × ℝ := (vectorAB.1 / magnitudeAB, vectorAB.2 / magnitudeAB)

-- The theorem we want to prove
theorem unit_vector_same_direction :
  unitVectorAB = (3 / 5, -4 / 5) :=
sorry

end unit_vector_same_direction_l452_452197


namespace probability_xy_minus_x_minus_y_even_l452_452463

open Nat

theorem probability_xy_minus_x_minus_y_even :
  let S := {1,2,3,4,5,6,7,8,9,10,11,12}
  let evens := {2, 4, 6, 8, 10, 12}
  let total_pairs := (Finset.card S).choose 2
  let even_pairs := (Finset.card evens).choose 2
  even_pairs / total_pairs = 5 / 22 :=
by
  sorry

end probability_xy_minus_x_minus_y_even_l452_452463


namespace smallest_integer_among_options_l452_452084

theorem smallest_integer_among_options :
  let A := 3 * 4^2 + 2 * 4^1 + 1 * 4^0 in
  let B := 58 in
  let C := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0 in
  let D := 7 * 8^1 + 3 * 8^0 in
  A = 57 ∧ B = 58 ∧ C = 56 ∧ D = 59 → C = 56 :=
begin
  sorry
end

end smallest_integer_among_options_l452_452084


namespace find_n_times_s_l452_452755

def nonzero_real_set := {x : ℝ // x ≠ 0}

def functional_eq (f : nonzero_real_set → nonzero_real_set) : Prop :=
  ∀ (x y : nonzero_real_set), ↑x + ↑y ≠ 0 → f x + f y = f ⟨(↑x * ↑y) / (↑x + ↑y), by
    have h1 : ↑x * ↑y ≠ 0 := mul_ne_zero x.prop y.prop
    have h2 : ↑x + ↑y ≠ 0 := by assumption
    exact div_ne_zero h1 h2⟩

def f_two (f : nonzero_real_set → nonzero_real_set) : ℝ := (f ⟨2, by norm_num⟩ : ℝ)

theorem find_n_times_s (f : nonzero_real_set → nonzero_real_set)
  (h : functional_eq f) :
  let possible_values := {v : ℝ | ∃ (x : ℝ) (hx : x ≠ 0), f ⟨x, hx⟩ = ⟨v, sorry⟩} in
  let n := possible_values.card in
  let s := possible_values.sum id in
  n * s = 1 / 2 :=
sorry

end find_n_times_s_l452_452755


namespace difference_of_squares_evaluation_l452_452605

theorem difference_of_squares_evaluation :
  49^2 - 16^2 = 2145 :=
by sorry

end difference_of_squares_evaluation_l452_452605


namespace how_many_raisins_did_bryce_receive_l452_452681

def raisins_problem : Prop :=
  ∃ (B C : ℕ), B = C + 8 ∧ C = B / 3 ∧ B + C = 44 ∧ B = 33

theorem how_many_raisins_did_bryce_receive : raisins_problem :=
sorry

end how_many_raisins_did_bryce_receive_l452_452681


namespace solve_safari_lions_l452_452782

def safari_lions : ℕ := sorry -- Number of lions in Safari National Park

axiom condition1 : ∀ (L : ℕ), (∃ S : ℕ, S = L / 2) -> (∃ G : ℕ, G = S - 10)

axiom condition2 : ∃ L S G : ℕ,
  S = L / 2 ∧
  G = S - 10 ∧
  2 * L + 3 * S + (G + 20) = 410

theorem solve_safari_lions : safari_lions = 72 :=
by
  sorry

end solve_safari_lions_l452_452782


namespace sin3x_plus_sin7x_l452_452139

theorem sin3x_plus_sin7x (x : ℝ) : 
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by sorry

end sin3x_plus_sin7x_l452_452139


namespace graduating_class_total_students_l452_452707

variables (G B d x total : ℕ)

-- Define the conditions
def students_took_geometry := G = 144
def students_took_biology := B = 119
def difference_between_greatest_and_smallest := d = 88
def students_took_both := x = 31

-- Define the total number of students
def total_students := total = (G - x) + (B - x) + x

-- Statement to prove
theorem graduating_class_total_students : 
  students_took_geometry G ∧ students_took_biology B ∧ 
  difference_between_greatest_and_smallest d ∧ students_took_both x →
  total_students total →
  total = 232 :=
by
  intros h1 h2 h3 h4,
  sorry

end graduating_class_total_students_l452_452707


namespace proposition_3_true_proposition_4_true_l452_452584

theorem proposition_3_true :
  ∀ (x : ℝ), sin (2/3 * x + π/2) = cos (2/3 * x)
  := by ext x; simp [cos, sin, Int.cast_add, Int.cast_bit0, Int.cast_one]; ring

theorem proposition_4_true {A B C : ℝ} (hABC : A + B + C = π)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hA_acute : A < π/2) (hB_acute : B < π/2) (hC_acute : C < π/2) :
  sin A > cos B
  := by
  have h : A + B + C = π - A - B - C => by rwa [← hABC]
  have hb_eq_pi_minus_c : B = π/2 - C := by linarith
  have hpi_div_2_minus_C_gt_zero := sub_pos_of_lt hC_acute
  have hb_eq_pi_over_2 := hb_eq_pi_minus_c ▸ (le_antisymm (by linarith [le_refl]) (by linarith [h]))
  rw [sin_pos_of_pos_of_lt_pi hb_eq_pi_over_2] at *,
  rw [cos_pi_div_two_sub] at *,
  exact lt_of_lt_of_le (le_trans zero_le_one hb_eq_pi_over_2) hb_eq_pi_over_2

end proposition_3_true_proposition_4_true_l452_452584


namespace arithmetic_mean_of_primes_l452_452958

open Real

theorem arithmetic_mean_of_primes :
  let numbers := [33, 37, 39, 41, 43]
  let primes := numbers.filter Prime
  let sum_primes := (37 + 41 + 43 : ℤ)
  let count_primes := (3 : ℤ)
  let mean := (sum_primes / count_primes : ℚ)
  mean = 40.33 := by
sorry

end arithmetic_mean_of_primes_l452_452958


namespace circle_is_axisymmetric_and_centrally_symmetric_l452_452567

structure Shape where
  isAxisymmetric : Prop
  isCentrallySymmetric : Prop

theorem circle_is_axisymmetric_and_centrally_symmetric :
  ∃ (s : Shape), s.isAxisymmetric ∧ s.isCentrallySymmetric :=
by
  sorry

end circle_is_axisymmetric_and_centrally_symmetric_l452_452567


namespace mary_has_more_money_than_marco_l452_452375

variable (Marco Mary : ℕ)

theorem mary_has_more_money_than_marco
    (h1 : Marco = 24)
    (h2 : Mary = 15)
    (h3 : ∃ maryNew : ℕ, maryNew = Mary + Marco / 2 - 5)
    (h4 : ∃ marcoNew : ℕ, marcoNew = Marco / 2) :
    ∃ diff : ℕ, diff = maryNew - marcoNew ∧ diff = 10 := 
by 
    sorry

end mary_has_more_money_than_marco_l452_452375


namespace smallest_multiple_of_9_and_6_l452_452505

theorem smallest_multiple_of_9_and_6 : ∃ n : ℕ, (n > 0) ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) := 
begin
  use 18,
  split,
  { -- n > 0
    exact nat.succ_pos',
  },
  split,
  { -- n % 9 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 9),
  },
  split,
  { -- n % 6 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 6),
  },
  { -- ∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m
    intros m h_pos h_multiple9 h_multiple6,
    exact le_of_dvd h_pos (nat.lcm_dvd_prime_multiples 6 9),
  },
  sorry, -- Since full proof capabilities are not required here, "sorry" is used to skip the proof process.
end

end smallest_multiple_of_9_and_6_l452_452505


namespace starting_percentage_66_inch_player_l452_452417

/-
  The following definitions and theorem aim to formalize the given problem conditions 
  and prove that the starting percentage for a 66-inch tall player is indeed 10%.
-/

-- Define the initial height, the height gain, and the final height.
def initial_height := 65
def height_gain := 3
def final_height := initial_height + height_gain

-- Define the height at which the percentage starts and the increase per inch.
def baseline_height := 66
def increase_per_inch := 0.1

-- Define Devin's final chance of making the basketball team.
def devin_final_chance := 0.3

-- Calculate the height difference and the percentage increase due to the height difference.
def height_difference := final_height - baseline_height
def percentage_increase := height_difference * increase_per_inch

-- Define the theorem to prove the starting percentage for a 66-inch tall player.
theorem starting_percentage_66_inch_player : 
  ∀ (initial_height height_gain baseline_height : ℕ) 
    (increase_per_inch devin_final_chance percentage_increase : ℝ), 
    initial_height = 65 → 
    height_gain = 3 → 
    baseline_height = 66 → 
    increase_per_inch = 0.1 → 
    devin_final_chance = 0.3 → 
    percentage_increase = (initial_height + height_gain - baseline_height) * increase_per_inch → 
    (devin_final_chance - percentage_increase) = 0.1 := by
  intros
  sorry

end starting_percentage_66_inch_player_l452_452417


namespace no_such_function_exists_l452_452596

-- Let's define the assumptions as conditions
def condition1 (f : ℝ → ℝ) := ∀ x : ℝ, f (x^2) - (f x)^2 ≥ 1 / 4
def distinct_values (f : ℝ → ℝ) := ∀ x y : ℝ, x ≠ y → f x ≠ f y

-- Now we state the main theorem
theorem no_such_function_exists : ¬ ∃ f : ℝ → ℝ, condition1 f ∧ distinct_values f :=
sorry

end no_such_function_exists_l452_452596


namespace original_equation_no_solution_l452_452409

noncomputable def no_solution (x : ℝ) : Prop :=
  (x - 2) / (x + 2) - 1 ≠ 16 / (x^2 - 4)

theorem original_equation_no_solution :
  ∀ x : ℝ, x ≠ 2 → x ≠ -2 → no_solution x :=
begin
  intros x hx1 hx2,
  unfold no_solution,
  sorry,
end

end original_equation_no_solution_l452_452409


namespace smaller_of_two_digit_product_4680_l452_452426

theorem smaller_of_two_digit_product_4680 (a b : ℕ) (h1 : a * b = 4680) (h2 : 10 ≤ a) (h3 : a < 100) (h4 : 10 ≤ b) (h5 : b < 100): min a b = 40 :=
sorry

end smaller_of_two_digit_product_4680_l452_452426


namespace Tom_needs_11_25_hours_per_week_l452_452824

theorem Tom_needs_11_25_hours_per_week
  (summer_weeks: ℕ) (summer_weeks_val: summer_weeks = 8)
  (summer_hours_per_week: ℕ) (summer_hours_per_week_val: summer_hours_per_week = 45)
  (summer_earnings: ℝ) (summer_earnings_val: summer_earnings = 3600)
  (rest_weeks: ℕ) (rest_weeks_val: rest_weeks = 40)
  (rest_earnings_goal: ℝ) (rest_earnings_goal_val: rest_earnings_goal = 4500) :
  (rest_earnings_goal / (summer_earnings / (summer_hours_per_week * summer_weeks))) / rest_weeks = 11.25 :=
by
  simp [summer_earnings_val, rest_earnings_goal_val, summer_hours_per_week_val, summer_weeks_val]
  sorry

end Tom_needs_11_25_hours_per_week_l452_452824


namespace intersection_interval_l452_452365

-- Definitions
def f (x : ℝ) := x^3 - 2^(2 - x)

-- The statement to prove
theorem intersection_interval (x : ℝ) (y : ℝ) (h_inter : y = x^3 ∧ y = 2^(2 - x)) : 1 < x ∧ x < 2 := by
  sorry

end intersection_interval_l452_452365


namespace pyramid_pattern_l452_452539

theorem pyramid_pattern
  (R : ℕ → ℕ)  -- a function representing the number of blocks in each row
  (R₁ : R 1 = 9)  -- the first row has 9 blocks
  (sum_eq : R 1 + R 2 + R 3 + R 4 + R 5 = 25)  -- the total number of blocks is 25
  (pattern : ∀ n, 1 ≤ n ∧ n < 5 → R (n + 1) = R n - 2) : ∃ d, d = 2 :=
by
  have pattern_valid : R 1 = 9 ∧ R 2 = 7 ∧ R 3 = 5 ∧ R 4 = 3 ∧ R 5 = 1 :=
    sorry  -- Proof omitted
  exact ⟨2, rfl⟩

end pyramid_pattern_l452_452539


namespace total_students_in_class_l452_452028

theorem total_students_in_class
    (students_in_front : ℕ)
    (students_in_back : ℕ)
    (lines : ℕ)
    (total_students_line : ℕ)
    (total_class : ℕ)
    (h_front: students_in_front = 2)
    (h_back: students_in_back = 5)
    (h_lines: lines = 3)
    (h_students_line : total_students_line = students_in_front + 1 + students_in_back)
    (h_total_class : total_class = lines * total_students_line) :
  total_class = 24 := by
  sorry

end total_students_in_class_l452_452028


namespace candy_cost_l452_452109

-- Definitions from conditions
def game_cost : ℕ := 60
def hourly_rate : ℕ := 8
def hours_worked : ℕ := 9
def money_left : ℕ := 7

-- Calculation and assertion
theorem candy_cost
  (game_cost_eq : game_cost = 60)
  (hourly_rate_eq : hourly_rate = 8)
  (hours_worked_eq : hours_worked = 9)
  (money_left_eq : money_left = 7) :
  let total_earned := hourly_rate * hours_worked,
      total_spent := total_earned - money_left,
      candy_cost := total_spent - game_cost in
  candy_cost = 5 :=
by
  sorry

end candy_cost_l452_452109


namespace probability_even_xy_sub_xy_even_l452_452458

theorem probability_even_xy_sub_xy_even :
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let evens := {2, 4, 6, 8, 10, 12}
  let total_ways := (s.card.choose 2)
  let even_ways := (evens.card.choose 2)
  even_ways.toRat / total_ways.toRat = 5 / 22 :=
by
  sorry

end probability_even_xy_sub_xy_even_l452_452458


namespace solve_for_z_l452_452944

theorem solve_for_z : ∃ z : ℚ, (sqrt (10 + 3 * z) = 12) ∧ (z = 134 / 3) :=
by
  sorry

end solve_for_z_l452_452944


namespace a_2008_lt_5_l452_452813

noncomputable def a (h : ℕ → ℕ) : ℕ → ℝ
| 0        := 1
| (n + 1)  := 1 / b h n * (1 + a h n + a h n * h n)

noncomputable def b (h : ℕ → ℕ) : ℕ → ℝ
| 0        := 2
| (n + 1)  := 1 / a h n * (1 + b h n + a h n * h n)

theorem a_2008_lt_5 (h : ℕ → ℕ) : a h 2008 < 5 :=
by
  sorry

end a_2008_lt_5_l452_452813


namespace problem1_problem2_problem3_problem4_l452_452041

-- Problem 1
theorem problem1 : (1 / 9) * Real.sqrt 9 - Real.cbrt 27 + Real.cbrt (-64) + Real.sqrt ((- (1 / 2))^2) = -6 - 1/6 := 
by sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (4 * x^2 * y - 8 * x * y^2 + 4 * x * y) / (-2 * x * y) = -2 * x + 4 * y - 2 := 
by sorry

-- Problem 3
theorem problem3 (x : ℝ) : (2 * x + 3) * (3 * x - 2) = 6 * x^2 + 5 * x - 6 := 
by sorry

-- Problem 4
theorem problem4 (x y : ℝ) : (3 * x - y)^2 - (3 * x + 2 * y) * (3 * x - 2 * y) = 5 * y^2 - 6 * x * y := 
by sorry

end problem1_problem2_problem3_problem4_l452_452041


namespace product_in_terms_of_sum_and_sum_of_squares_l452_452088

-- Define the geometric progression and the related sums
variables {a r : ℝ} -- first term and common ratio
variables {n : ℕ} -- number of terms
variables (P S S'' : ℝ) -- product, sum, and sum of squares

-- Defining the conditions in the problem
def is_geometric_progression (a r : ℝ) (n : ℕ) := ∀ k, k < n → ∃ i, a * r^i = k

def sum_of_gp (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

def sum_of_squares_of_gp (a r : ℝ) (n : ℕ) : ℝ := a^2 * (1 - r^(2*n)) / (1 - r^2)

def product_of_gp (a r : ℝ) (n : ℕ) : ℝ := a^n * r^(n * (n - 1) / 2)

-- The main theorem to prove
theorem product_in_terms_of_sum_and_sum_of_squares
  (h1: is_geometric_progression a r n)
  (h2: S = sum_of_gp a r n)
  (h3: S'' = sum_of_squares_of_gp a r n)
  (h4: P = product_of_gp a r n) :
  P = sqrt (a^(2*n) * (S^2 / S'')) :=
sorry

end product_in_terms_of_sum_and_sum_of_squares_l452_452088


namespace negation_of_proposition_l452_452810

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^2 + x_0 - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l452_452810


namespace symmetric_line_equation_l452_452967

theorem symmetric_line_equation (x y : ℝ) :
  (∀ x y, 2 * x - y + 1 = 0 → ∃ x' y', 2 * (2 - x') - y' + 1 = 0 ∧ x = 2 - x') →
  (2 * x + y - 5 = 0) :=
by
  intros h
  specialize h 1 ((y + 1 - 2 * x) / 1)
  cases h with x_sym symmetric_point_cond
  have : x = 2 - x_sym := symmetric_point_cond.2
  rw [this] at symmetric_point_cond.1
  sorry

end symmetric_line_equation_l452_452967


namespace triangle_area_eq_l452_452949

noncomputable theory
open Real

-- Given: A circle with radius R = 2 cm
def circle (R : ℝ) := R = 2

-- Given: Two angles of the triangle
def angle1 : ℝ := π / 3
def angle2 : ℝ := π / 4

-- To prove: The area of the triangle is sqrt(3) + 3 cm^2
theorem triangle_area_eq : ∀ {R : ℝ}, circle R → 
  (area_of_triangle_inscribed R angle1 angle2) = (sqrt 3 + 3) :=
by sorry

end triangle_area_eq_l452_452949


namespace pentagon_largest_angle_l452_452271

theorem pentagon_largest_angle
  (F G H I J : ℝ)
  (hF : F = 90)
  (hG : G = 70)
  (hH_eq_I : H = I)
  (hJ : J = 2 * H + 20)
  (sum_angles : F + G + H + I + J = 540) :
  max F (max G (max H (max I J))) = 200 :=
by
  sorry

end pentagon_largest_angle_l452_452271


namespace garden_area_increase_l452_452046

noncomputable def area_increase_due_to_shape_change : Prop :=
  let length := 60
  let width := 12
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  let original_area := length * width
  let new_area := Real.pi * radius^2
  let area_difference := new_area - original_area
  -- Using an approximate value of area_difference calculated
  area_difference ≈ 929

theorem garden_area_increase : area_increase_due_to_shape_change := by
  sorry

end garden_area_increase_l452_452046


namespace midpoint_product_l452_452473

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 4) (hy1 : y1 = -3) (hx2 : x2 = -8) (hy2 : y2 = 7) :
  let midx := (x1 + x2) / 2
  let midy := (y1 + y2) / 2
  midx * midy = -4 :=
by
  sorry

end midpoint_product_l452_452473


namespace astronaut_suitable_crew_l452_452881

theorem astronaut_suitable_crew :
  ∀ (A : Finset ℕ) (hA : A.card = 11000)
    (H : ∀ (s : Finset ℕ), s ⊆ A → s.card = 4 → ∃ (t : Finset ℕ), t ⊆ s ∧ t.card = 3 ∧ suitable_crew t),
  ∃ (B : Finset ℕ), B ⊆ A ∧ B.card = 5 ∧ ∀ (s : Finset ℕ), s ⊆ B → s.card = 3 → suitable_crew s := 
sorry

-- Declare an opaque definition for "suitable_crew" that can be refined later
noncomputable def suitable_crew (s : Finset ℕ) : Prop := 
  sorry

end astronaut_suitable_crew_l452_452881


namespace neils_cookies_l452_452767

theorem neils_cookies (total_cookies : ℕ) (fraction_given_to_friend : ℚ) (remaining_cookies : ℕ) :
  total_cookies = 20 →
  fraction_given_to_friend = 2 / 5 →
  remaining_cookies = total_cookies - (total_cookies * (fraction_given_to_friend.num : ℕ) / fraction_given_to_friend.denom) →
  remaining_cookies = 12 :=
by
  intros h_total h_fraction h_remaining
  rw [h_total, h_fraction, h_remaining]
  norm_num

end neils_cookies_l452_452767


namespace domain_of_f_l452_452966

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (2 - real.exp (x * real.log 2)) + 1 / (real.log x / real.log 3)

theorem domain_of_f :
  (∀ x : ℝ, 0 < x ∧ x < 1 → (2 - real.exp (x * real.log 2) ≥ 0) ∧ (x ≠ 1) →  0 < x ∧ x < 1) :=
by
  intros x hx
  sorry -- Proof will go here

end domain_of_f_l452_452966


namespace distance_between_skew_lines_l452_452714

variables {P : Type} [Plane P] {A B C : P} {l a r : ℝ} (h_isosceles : dist A B = l ∧ dist B C = l) 
          (h_AC : dist A C = 2 * a) {sphere : Type} [Sphere sphere] 
          (h_sphere : sphere.radius = r ∧ sphere.tangent_point = B)
          (h_angle : ∀ {L1 L2 : Line}, L1.contains A ∧ L1.tangent_to sphere ∧ L2.contains C ∧ L2.tangent_to sphere → angle_between L1 P = α ∧ angle_between L2 P = α)

theorem distance_between_skew_lines :
    distance_since_point A → distance_since_point C → 
    dist_sk_lines (L1, A) (L2, C) = (2 * a * tan α * sqrt (2 * r * l * sin α - (l^2 + r^2) * sin^2 α)) / sqrt (l^2 - a^2 * cos^2 α) := 
sorry

end distance_between_skew_lines_l452_452714


namespace find_E_coordinates_l452_452300

structure Point where
  x : ℚ
  y : ℚ

def A : Point := {x := -2, y := 1}
def B : Point := {x := 1, y := 4}
def C : Point := {x := 4, y := -3}
def D : Point := {x := (-2 * 1 + 1 * (-2)) / (1 + 2), y := (1 * 4 + 2 * 1) / (1 + 2)}

def externalDivision (P1 P2 : Point) (m n : ℚ) : Point :=
  {x := (m * P2.x - n * P1.x) / (m - n), y := (m * P2.y - n * P1.y) / (m - n)}

theorem find_E_coordinates :
  let E := externalDivision D C 1 4
  E.x = -8 / 3 ∧ E.y = 11 / 3 := 
by 
  let E := externalDivision D C 1 4
  sorry

end find_E_coordinates_l452_452300


namespace locus_of_C_l452_452669

open EuclideanSpace

variables {E : Type*} [normed_add_comm_group E] [inner_product_space ℝ E]
variables (A B C : E)
variables [ordered_ring ℝ]

def midpoint (A B : E) : E := (A + B) / 2

-- Assuming the segment AB has midpoint D
let D := midpoint A B

-- E is the perpendicular projection of C onto AB
def projection (C A B : E) : E := 
  A + (((C - A).dot_product (B - A)) / ((B - A).dot_product (B - A))) • (B - A)

let E := projection C A B

-- The internal angle bisector of ∠ACB bisects the segment DE
def angle_bisector_bisects (A B C : E) : Prop :=
  let F := midpoint (angle_bisector_point C A B) D in
  let DE := dist D E in
  dist D F = DE / 2

def is_perpendicular_bisector (C : E) (A B : E) : Prop :=
  let m := (A + B) / 2 in
  dist C m = sqrt (dist A B) / 2 

def is_ellipse (C : E) (A B : E) : Prop :=
  (norm (C - A) + norm (C - B)) = sqrt (2) * norm (B - A)

theorem locus_of_C (h : angle_bisector_bisects A B C) : 
  is_perpendicular_bisector C A B ∨ is_ellipse C A B :=
sorry

end locus_of_C_l452_452669


namespace m_squared_plus_n_squared_l452_452992

noncomputable def xi_values (m n : ℝ) : Prop :=
  ∃ (p q : ℝ), ∑ (i : ℝ) in {m, n}, i * (if (i = m) then n else m) = 3 / 8

theorem m_squared_plus_n_squared (m n : ℝ) (h1 : m + n = 1) (h2 : 2 * m * n = 3 / 8) : 
  m^2 + n^2 = 5 / 8 :=
by
  sorry

end m_squared_plus_n_squared_l452_452992


namespace largest_fraction_among_given_l452_452509

theorem largest_fraction_among_given (f1 f2 f3 f4 f5 : ℚ)
  (h1 : f1 = 2/5) 
  (h2 : f2 = 4/9) 
  (h3 : f3 = 7/15) 
  (h4 : f4 = 11/18) 
  (h5 : f5 = 16/35) 
  : f1 < f4 ∧ f2 < f4 ∧ f3 < f4 ∧ f5 < f4 :=
by
  sorry

end largest_fraction_among_given_l452_452509


namespace tenth_monomial_in_sequence_l452_452392

theorem tenth_monomial_in_sequence (a : ℕ) : 
  let nth_monomial (n : ℕ) := (-2) ^ (n - 1) * (a ^ n)
  in nth_monomial 10 = -2^9 * a^10 :=
by 
  let nth_monomial : ℕ → ℤ := (λ n, (-2)^(n-1) * (a^n))
  show nth_monomial 10 = -2^9 * a^10
  sorry

end tenth_monomial_in_sequence_l452_452392


namespace common_intersection_of_four_circles_l452_452797

noncomputable def commonIntersectionArea (a : ℝ) : ℝ :=
  a^2 * (π + 3 - 3 * Real.sqrt 3) / 3

theorem common_intersection_of_four_circles (a : ℝ) :
  let centers := [(0, 0), (a, 0), (0, a), (a, a)],
      radii := [a, a, a, a]
  in commonIntersectionArea a = a^2 * (π + 3 - 3 * Real.sqrt 3) / 3 :=
by 
  let centers := [(0, 0), (a, 0), (0, a), (a, a)],
      radii := [a, a, a, a]
  exact sorry

end common_intersection_of_four_circles_l452_452797


namespace product_of_two_larger_numbers_is_115_l452_452010

noncomputable def proofProblem : Prop :=
  ∃ (A B C : ℝ), B = 10 ∧ (C - B = B - A) ∧ (A * B = 85) ∧ (B * C = 115)

theorem product_of_two_larger_numbers_is_115 : proofProblem :=
by
  sorry

end product_of_two_larger_numbers_is_115_l452_452010


namespace quadratic_distinct_roots_l452_452259

theorem quadratic_distinct_roots (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
sorry

end quadratic_distinct_roots_l452_452259


namespace sin_double_angle_l452_452043

variable (α p : ℝ)
variable (h : Math.sin α - Math.cos α = p)

theorem sin_double_angle (h : Math.sin α - Math.cos α = p) : Math.sin (2 * α) = 1 - p^2 :=
sorry

end sin_double_angle_l452_452043


namespace greatest_candy_count_l452_452902

theorem greatest_candy_count 
  (students : ℕ) (mean_candy : ℕ) (min_candy : ℕ) 
  (h1 : students = 40) 
  (h2 : mean_candy = 4) 
  (h3 : min_candy = 2) 
  (h4 : ∀ s ∈ finset.range students, ∃ ps : ℕ, ps >= min_candy) :
  ∃ ps : ℕ, ps = 82 :=
by
  let total_candy := mean_candy * students
  have total_candy_calculation : total_candy = 160 := by rw [h1, h2]; exact rfl
  let min_for_39 := (students - 1) * min_candy
  have min_for_39_calculation : min_for_39 = 78 := by rw [h1, h3]; exact rfl
  let max_candy := total_candy - min_for_39
  have max_candy_calculation : max_candy = 82 := by rw [total_candy_calculation, min_for_39_calculation]; exact rfl
  use 82
  exact max_candy_calculation

end greatest_candy_count_l452_452902


namespace gcd_of_256_180_600_l452_452833

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l452_452833


namespace polynomial_division_l452_452027

open Polynomial

def P : Polynomial ℤ := 4 * X^4 - 3 * X^3 + 5 * X^2 - 7 * X + 6
def D : Polynomial ℤ := 4 * X + 7

theorem polynomial_division :
  ∃ (q r : Polynomial ℤ), P = D * q + r ∧ degree r < degree D :=
by
  sorry

end polynomial_division_l452_452027


namespace exists_sequence_n_25_exists_sequence_n_1000_l452_452319

theorem exists_sequence_n_25 : 
  ∃ (l : List ℕ), l.perm (List.range 25) ∧ 
                  ∀ (i : ℕ), i < l.length - 1 → (l[i + 1] - l[i]).natAbs = 3 ∨ (l[i + 1] - l[i]).natAbs = 5 := 
  sorry

theorem exists_sequence_n_1000 : 
  ∃ (l : List ℕ), l.perm (List.range 1000) ∧ 
                  ∀ (i : ℕ), i < l.length - 1 → (l[i + 1] - l[i]).natAbs = 3 ∨ (l[i + 1] - l[i]).natAbs = 5 := 
  sorry

end exists_sequence_n_25_exists_sequence_n_1000_l452_452319


namespace claudia_groupings_l452_452110

-- Definition of combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def candles_combinations : ℕ := combination 6 3
def flowers_combinations : ℕ := combination 15 12

-- Lean statement
theorem claudia_groupings : candles_combinations * flowers_combinations = 9100 :=
by
  sorry

end claudia_groupings_l452_452110


namespace largest_integer_solution_l452_452846

theorem largest_integer_solution : ∃ x : ℤ, (x ≤ 10) ∧ (∀ y : ℤ, (y > 10 → (y / 4 + 5 / 6 < 7 / 2) = false)) :=
sorry

end largest_integer_solution_l452_452846


namespace find_x_in_exponential_equation_l452_452852

theorem find_x_in_exponential_equation (x : ℝ) : 
  (5 ^ 3 + 5 ^ 3 + 5 ^ 3 = 15 ^ x) -> 
  x = 2 :=
  sorry

end find_x_in_exponential_equation_l452_452852


namespace garment_factory_optimization_l452_452517

variable (x : ℕ) (y : ℕ)

theorem garment_factory_optimization :
  (70 ≥ 1.1 * x + 0.6 * (80 - x)) ∧ (52 ≥ 0.4 * x + 0.9 * (80 - x)) ∧
  (40 ≤ x ∧ x ≤ 44) →
  y = 5 * x + 3600 ∧ 
  x ∈ {40, 41, 42, 43, 44} ∧ 
  (x = 44 → y = 3820) :=
by
  sorry

end garment_factory_optimization_l452_452517


namespace club_membership_l452_452068

theorem club_membership:
  ∃ n : ℕ, n < 50 ∧ n % 6 = 4 ∧ n % 5 = 2 :=
begin
  sorry,
end

end club_membership_l452_452068


namespace subset_choice_count_l452_452447

theorem subset_choice_count :
  ∃ (S : finset (finset ℤ)), S.card = 20 ∧ 
    ∀ (A B C : finset ℤ), (A ∈ S ∧ B ∈ S ∧ C ∈ S) → 
      (A ∪ B ∪ C = {0, 1, 2, 3, 4} : finset ℤ) ∧ 
      (A ∩ B ∩ C).card = 3 ∧ 
      multiset.card (multiset.of_finset (finset.image id S)) = 1 := sorry

end subset_choice_count_l452_452447


namespace establish_relation_l452_452528

open MeasureTheory

noncomputable def characteristic_function (X : ℕ → ℂ) (t : ℂ) := 
  ∑ n in ℕ, ℂ.exp(t * X n)

noncomputable def sum_S (X : ℕ → ℂ) (n : ℕ) := 
  ∑ i in finset.range n, X i

noncomputable def τ (X : ℕ → ℂ) : ℕ := 
  Inf {n : ℕ | n > 0 ∧ sum_S X n > 0}

theorem establish_relation (X : ℕ → ℂ) (varphi : ℂ → ℂ) (t s : ℂ):
  (∀ i j : ℕ, X i = X j) → 
  (∀ n : ℕ, varphi t = characteristic_function (λ k, X k) t) →
  (1 - ∑ n in measure_theory.prob_of (λω, ℂ.exp(it * sum_S X τ * s^τ ω))) = 
  ℂ.exp (- ∑ n in ℕ, s^n / n * ∑ n in measure_theory.prob_of (λω, ℂ.exp(it * sum_S X n * indicator (λω, sum_S X n > 0) ω))) :=
sorry

end establish_relation_l452_452528


namespace relationship_x_y_q_z_l452_452182

variable (a c b d : ℝ) (x y q z : ℝ)

-- Define the conditions as Lean assumptions
axiom condition1 : a ^ (2 * x) = c ^ (3 * q) = b
axiom condition2 : c ^ (4 * y) = a ^ (3 * z) = d

theorem relationship_x_y_q_z : 9 * q * z = 8 * x * y := by
  -- Formal proof would go here
  sorry

end relationship_x_y_q_z_l452_452182


namespace first_term_of_geometric_sequence_l452_452165

theorem first_term_of_geometric_sequence (r : ℕ) (a : ℕ): 
  (243 = 81 * r) → (81 = a * r^4) → a = 1 :=
by 
  intros h1 h2
  have h_r : r = 3 := by sorry
  rw h_r at h2
  exact eq_of_mul_eq_mul_right (pow_ne_zero 4 (by norm_num)) h2

end first_term_of_geometric_sequence_l452_452165


namespace ratio_ivanna_dorothy_l452_452793

noncomputable def T (I : ℝ) : ℝ := 2 * I
noncomputable def D : ℝ := 90
noncomputable def avg : ℝ := 84
noncomputable def total_score (T I D : ℝ) : ℝ := T + I + D
noncomputable def I_fraction_of_D (x : ℝ) (D : ℝ) : ℝ := x * D

theorem ratio_ivanna_dorothy (x : ℝ) (hx : 0 < x ∧ x < 1) :
  let I := I_fraction_of_D x D in
  let T := T I in
  total_score T I D = 3 * avg →
  I / D = 3 / 5 :=
by
  sorry

end ratio_ivanna_dorothy_l452_452793


namespace water_left_after_four_hours_l452_452685

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def water_added_hour3 : ℕ := 1
def water_added_hour4 : ℕ := 3

theorem water_left_after_four_hours :
    initial_water - 2 * 2 - 2 + water_added_hour3 - 2 + water_added_hour4 - 2 = 36 := 
by
    sorry

end water_left_after_four_hours_l452_452685


namespace num_employees_excluding_manager_l452_452796

/-- 
If the average monthly salary of employees is Rs. 1500, 
and adding a manager with salary Rs. 14100 increases 
the average salary by Rs. 600, prove that the number 
of employees (excluding the manager) is 20.
-/
theorem num_employees_excluding_manager 
  (avg_salary : ℕ) 
  (manager_salary : ℕ) 
  (new_avg_increase : ℕ) : 
  (∃ n : ℕ, 
    avg_salary = 1500 ∧ 
    manager_salary = 14100 ∧ 
    new_avg_increase = 600 ∧ 
    n = 20) := 
sorry

end num_employees_excluding_manager_l452_452796


namespace smallest_multiple_of_9_and_6_l452_452504

theorem smallest_multiple_of_9_and_6 : ∃ n : ℕ, (n > 0) ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) := 
begin
  use 18,
  split,
  { -- n > 0
    exact nat.succ_pos',
  },
  split,
  { -- n % 9 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 9),
  },
  split,
  { -- n % 6 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 6),
  },
  { -- ∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m
    intros m h_pos h_multiple9 h_multiple6,
    exact le_of_dvd h_pos (nat.lcm_dvd_prime_multiples 6 9),
  },
  sorry, -- Since full proof capabilities are not required here, "sorry" is used to skip the proof process.
end

end smallest_multiple_of_9_and_6_l452_452504


namespace perpendicular_condition_l452_452193

variables {R : Type*} [real_field R] 
variables (a b m : vector R) (α : set (vector R))

-- Define what it means for a line to be perpendicular to another line or a plane.
def perp_to_line (m l : vector R) : Prop := m.dot l = 0
def perp_to_plane (m : vector R) (α : set (vector R)) : Prop := ∀ a ∈ α, m.dot a = 0

-- Given conditions
axiom a_in_alpha : a ∈ α
axiom b_in_alpha : b ∈ α

-- The statement to prove
theorem perpendicular_condition (hma : perp_to_line m a) (hmb : perp_to_line m b) :
  (perp_to_plane m α) ↔ (perp_to_line m a ∧ perp_to_line m b) :=
sorry

end perpendicular_condition_l452_452193


namespace geometric_mean_of_sequence_l452_452023

theorem geometric_mean_of_sequence (y : ℤ) : y = 27 :=
by
  have h1 : 9 = 3 ^ 2 := by norm_num
  have h2 : 81 = 3 ^ 4 := by norm_num
  have h3 : y = Int.sqrt (9 * 81) := by sorry
  have h4 : Int.sqrt 729 = 27 := by norm_num
  rw [mul_eq, h1, h2] at h3
  rw h4 at h3
  exact h3

end geometric_mean_of_sequence_l452_452023


namespace find_x_set_l452_452657

-- Define the even function and its properties
variables (f : ℝ → ℝ)
hypothesis (h_even : ∀ x, f(x) = f(-x))
hypothesis (h_decreasing : ∀ ⦃a b⦄, a < b → b < 0 → f(b) < f(a))

-- Main theorem statement
theorem find_x_set (x : ℝ) (h : x < -1) : f(x^2 + 2*x + 3) > f(-x^2 - 4*x - 5) :=
sorry

end find_x_set_l452_452657


namespace part1_l452_452042

theorem part1 (m n p : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : p > 0) : 
  (n / m) < (n + p) / (m + p) := 
sorry

end part1_l452_452042


namespace trigonometric_identity_l452_452176

theorem trigonometric_identity (θ : ℝ) (h : sin (θ - π / 4) = 1 / 5) : cos (θ + π / 4) = -1 / 5 := 
by sorry

end trigonometric_identity_l452_452176


namespace interval_a_l452_452180

-- Definitions and conditions from the problem
def A (x : ℝ) : Prop := abs x * (x^2 - 4 * x + 3) < 0
def B (x : ℝ) (a : ℝ) : Prop := (2^(1 - x) + a ≤ 0) ∧ (x^2 - 2 * (a + 7) * x + 5 ≤ 0)

-- Statement we need to prove
theorem interval_a (a : ℝ) : (∀ x, A x → B x a) → -4 ≤ a ∧ a ≤ -1 :=
sorry

end interval_a_l452_452180


namespace case_D_has_two_solutions_l452_452787

-- Definitions for the conditions of each case
structure CaseA :=
(b : ℝ) (A : ℝ) (B : ℝ)

structure CaseB :=
(a : ℝ) (c : ℝ) (B : ℝ)

structure CaseC :=
(a : ℝ) (b : ℝ) (A : ℝ)

structure CaseD :=
(a : ℝ) (b : ℝ) (A : ℝ)

-- Setting the values based on the given conditions
def caseA := CaseA.mk 10 45 70
def caseB := CaseB.mk 60 48 100
def caseC := CaseC.mk 14 16 45
def caseD := CaseD.mk 7 5 80

-- Define a function that checks if a case has two solutions
def has_two_solutions (a b c : ℝ) (A B : ℝ) : Prop := sorry

-- The theorem to prove that out of the given cases, only Case D has two solutions
theorem case_D_has_two_solutions :
  has_two_solutions caseA.b caseB.B caseC.a caseC.b caseC.A = false →
  has_two_solutions caseB.a caseB.c caseB.B caseC.b caseC.A = false →
  has_two_solutions caseC.a caseC.b caseC.A caseD.a caseD.b = false →
  has_two_solutions caseD.a caseD.b caseD.A caseA.b caseA.A = true :=
sorry

end case_D_has_two_solutions_l452_452787


namespace most_people_can_attend_on_most_days_l452_452599

-- Define the days of the week as a type
inductive Day
| Mon | Tues | Wed | Thurs | Fri

open Day

-- Define the availability of each person
def is_available (person : String) (day : Day) : Prop :=
  match person, day with
  | "Anna", Mon => False
  | "Anna", Wed => False
  | "Anna", Fri => False
  | "Bill", Tues => False
  | "Bill", Thurs => False
  | "Bill", Fri => False
  | "Carl", Mon => False
  | "Carl", Tues => False
  | "Carl", Thurs => False
  | "Diana", Wed => False
  | "Diana", Fri => False
  | _, _ => True

-- Prove the result
theorem most_people_can_attend_on_most_days :
  {d : Day | d ∈ [Mon, Tues, Wed]} = {d : Day | ∀p : String, is_available p d → p ∈ ["Bill", "Carl", "Diana"] ∨ p ∉ ["Anna", "Bill"]} :=
sorry

end most_people_can_attend_on_most_days_l452_452599


namespace work_days_l452_452535

theorem work_days (x : ℕ) (hx : 0 < x) :
  (1 / (x : ℚ) + 1 / 20) = 1 / 15 → x = 60 := by
sorry

end work_days_l452_452535


namespace jenny_best_neighborhood_earnings_l452_452335

theorem jenny_best_neighborhood_earnings :
  let cost_per_box := 2
  let neighborhood_a_homes := 10
  let neighborhood_a_boxes_per_home := 2
  let neighborhood_b_homes := 5
  let neighborhood_b_boxes_per_home := 5
  let earnings_a := neighborhood_a_homes * neighborhood_a_boxes_per_home * cost_per_box
  let earnings_b := neighborhood_b_homes * neighborhood_b_boxes_per_home * cost_per_box
  max earnings_a earnings_b = 50
:= by
  sorry

end jenny_best_neighborhood_earnings_l452_452335


namespace greatest_integer_less_than_PS_l452_452284

noncomputable def rectangle_problem (PQ PS : ℝ) (T : ℝ) (PT QP : ℝ) : ℝ := real.sqrt (PQ * PQ + PS * PS) / 2

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ) (hPQ : PQ = 150)
  (T_midpoint : T = PS / 2)
  (PT_perpendicular_QT : PT * PT + T * T = PQ * PQ),
  int.floor (PS) = 212 :=
by
  intros PQ PS hPQ T_midpoint PT_perpendicular_QT
  have h₁ : PS = rectangle_problem PQ PS T PQ,
  {
    sorry
  }
  have h₂ : 150 * real.sqrt 2,
  {
    sorry
  }
  have h₃ : (⌊150 * real.sqrt 2⌋ : ℤ) = 212,
  {
    sorry
  }
  exact h₃

end greatest_integer_less_than_PS_l452_452284


namespace solve_trig_inequality_l452_452610

noncomputable def sin_triple_angle_identity (x : ℝ) : ℝ :=
  3 * (Real.sin x) - 4 * (Real.sin x) ^ 3

theorem solve_trig_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
  (8 / (3 * Real.sin x - sin_triple_angle_identity x) + 3 * (Real.sin x) ^ 2) ≤ 5 ↔
  x = Real.pi / 2 :=
by
  sorry

end solve_trig_inequality_l452_452610


namespace reach_final_round_l452_452869

def programmers := Fin 16

constant skills : ∀ (p : programmers), ℕ
axiom skill_different : ∀ (p1 p2 : programmers), p1 ≠ p2 → skills p1 ≠ skills p2

-- Each player has a different skill level, and when two play against each other, 
-- the one with the higher skill level will always win.
axiom match_winner : ∀ (p1 p2 : programmers), p1 ≠ p2 → (skills p1 < skills p2 ↔ skills p2 > skills p1)

-- Each round, each programmer plays a match against another, and the loser is eliminated. 
-- This continues until only one remains.
def single_elimination (P : Fin 2 → programmers) : Prop :=
  ∀ i j, i ≠ j → (skills (P i) > skills (P j) ∨ skills (P j) > skills (P i))

theorem reach_final_round :
  (∃ P : Fin 2 → programmers, single_elimination P) →
  (∃ S : Fin 9 → programmers, ∀ s1 s2, s1 ≠ s2 → skills (S s1) ≠ skills (S s2)) :=
sorry

end reach_final_round_l452_452869


namespace multiplicative_inverse_101_mod_401_l452_452581

theorem multiplicative_inverse_101_mod_401 : 
  ∃ a : ℤ, 0 ≤ a ∧ a < 401 ∧ (101 * a ≡ 1 [MOD 401]) :=
begin
  use 135,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num,
    exact_mod_cast int.modeq_of_modeq_mul 1 101 1 401 (by norm_num : 101 * 135 ≡ 1 [MOD 401]) }
end

end multiplicative_inverse_101_mod_401_l452_452581


namespace smallest_common_multiple_of_9_and_6_l452_452489

theorem smallest_common_multiple_of_9_and_6 : 
  ∃ x : ℕ, (x > 0 ∧ x % 9 = 0 ∧ x % 6 = 0) ∧ 
           ∀ y : ℕ, (y > 0 ∧ y % 9 = 0 ∧ y % 6 = 0) → x ≤ y :=
begin
  use 18,
  split,
  { split,
    { exact nat.succ_pos 17, },
    { split,
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_right 9 6), },
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_left 9 6), } } },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    exact lcm.dvd_iff.1 (nat.dvd_of_mod_eq_zero hy3) }
end

end smallest_common_multiple_of_9_and_6_l452_452489


namespace eccentricity_of_ellipse_l452_452982

theorem eccentricity_of_ellipse (m n : ℝ) (h1 : 1 / m + 2 / n = 1) (h2 : 0 < m) (h3 : 0 < n) (h4 : m * n = 8) :
  let a := n
  let b := m
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  e = Real.sqrt 3 / 2 := 
sorry

end eccentricity_of_ellipse_l452_452982
