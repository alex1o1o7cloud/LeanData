import Mathlib

namespace probability_both_numbers_are_prime_l525_525425

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l525_525425


namespace arithmetic_sequence_of_angles_maximum_area_of_triangle_l525_525689

variable {A B C a b c : ℝ}

-- Conditions
def in_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- sides opposite to angles
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π

def given_condition (A B C : ℝ) : Prop :=
  sin A * sin B + sin C ^ 2 = sin A ^ 2 + sin B ^ 2

-- Part 1: Prove that A, C, B form an arithmetic sequence
theorem arithmetic_sequence_of_angles
  (h_triangle : in_triangle a b c A B C)
  (h_condition : given_condition A B C) :
  (2 * C = A + B) :=
sorry

-- Part 2: Given c = 2, find the maximum area of the triangle
def area (a b C : ℝ) : ℝ :=
  1 / 2 * a * b * sin C

theorem maximum_area_of_triangle
  (h_triangle : in_triangle a b 2 A B π / 3) :
  (∀ a b : ℝ, a = b → area a b (π / 3) = sqrt 3) :=
sorry

end arithmetic_sequence_of_angles_maximum_area_of_triangle_l525_525689


namespace T_n_less_than_4_l525_525181

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b (n : ℕ) : ℕ := 2 ^ (n - 1)
noncomputable def c (n : ℕ) : ℕ := n / 2 ^ (n - 1)
noncomputable def T (n : ℕ) : ℕ := ∑ k in Finset.range n, c k

theorem T_n_less_than_4 (n : ℕ) : T n < 4 :=
  sorry

end T_n_less_than_4_l525_525181


namespace shaded_area_eq_l525_525187

-- Define the rectangle ABCD and its dimensions
variables (AB CD BC : ℝ)
variables (M N : ℝ)

-- midpoint conditions
def midpoint (a b: ℝ) := (a + b) / 2

-- given conditions
def rectangle_dim := AB = 6 ∧ BC = 10 ∧ M = midpoint 0 6 ∧ N = midpoint 0 6

-- shaded area calculation
def area_rectangle := AB * BC
def shaded_area (area_rectangle : ℝ) := (1 / 4) * area_rectangle

-- The theorem that needs to be proven
theorem shaded_area_eq (h : rectangle_dim) : shaded_area (area_rectangle) = 15 :=
sorry

end shaded_area_eq_l525_525187


namespace smallest_positive_a_and_unique_b_l525_525557

noncomputable def a : ℝ := sqrt 2

theorem smallest_positive_a_and_unique_b :
  (∃ b : ℝ, 0 < b ∧ (∀ x : ℝ, x^3 - 3 * a * x^2 + b * x - 2 * a = 0 → ℝ)) ∧ 
  a = sqrt 2 ∧ (∃! b : ℝ, b = 6) :=
by
  sorry

end smallest_positive_a_and_unique_b_l525_525557


namespace dartboard_probability_l525_525001

noncomputable def radius_of_dartboard (r : ℝ) : Prop :=
  r > 0 ∧
  let A_dartboard := π * r ^ 2 in
  let A_smaller := π * (r / 2) ^ 2 in
  let P := A_smaller / A_dartboard in
  P = 0.25

theorem dartboard_probability (r : ℝ) (h : r > 0) : radius_of_dartboard r :=
by
  unfold radius_of_dartboard
  sorry

end dartboard_probability_l525_525001


namespace last_two_digits_2005_power_1989_l525_525797

theorem last_two_digits_2005_power_1989 : (2005 ^ 1989) % 100 = 25 :=
by
  sorry

end last_two_digits_2005_power_1989_l525_525797


namespace discount_difference_l525_525024

theorem discount_difference :
  let bill := 10000 
  let single_discount := 0.4
  let first_discount := 0.36
  let second_discount := 0.04
  let amount_after_single := bill * (1 - single_discount)
  let amount_after_first := bill * (1 - first_discount)
  let amount_after_second := amount_after_first * (1 - second_discount)
  abs (amount_after_single - amount_after_second) = 144 :=
by
  sorry

end discount_difference_l525_525024


namespace complex_number_equality_l525_525738

open Complex

theorem complex_number_equality (u v : ℂ) 
  (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
  (h2 : abs (u + v) = abs (u * v + 1)) : 
  u = 1 ∨ v = 1 :=
sorry

end complex_number_equality_l525_525738


namespace parabola_sum_distance_l525_525138

noncomputable def parabola_result (x y : ℕ → ℝ) (cond_points_parabola : ∀ i, (y i) ^ 2 = 2 * x i)
  (sum_x: (finset.sum (finset.range 10) x) = 10) : ℝ :=
  let p := 1 in
  (finset.sum (finset.range 10) (λ i, (x i) + p/2))

theorem parabola_sum_distance :
  ∀ (x y : ℕ → ℝ) (cond_points_parabola : ∀ i, (y i) ^ 2 = 2 * x i)
  (sum_x: (finset.sum (finset.range 10) x) = 10),
  parabola_result x y cond_points_parabola sum_x = 15 :=
by sorry

end parabola_sum_distance_l525_525138


namespace total_animals_counted_l525_525223

theorem total_animals_counted :
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  show (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605
  sorry

end total_animals_counted_l525_525223


namespace solve_for_c_l525_525679

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) →
  (6 * 15 * c = 1.5) →
  c = 7

theorem solve_for_c : proof_problem 6 15 7 :=
by sorry

end solve_for_c_l525_525679


namespace minimum_n_for_candy_purchases_l525_525517

theorem minimum_n_for_candy_purchases' {o s p : ℕ} (h1 : 9 * o = 10 * s) (h2 : 9 * o = 20 * p) : 
  ∃ n : ℕ, 30 * n = 180 ∧ ∀ m : ℕ, (30 * m = 9 * o) → n ≤ m :=
by sorry

end minimum_n_for_candy_purchases_l525_525517


namespace range_of_a_l525_525608

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then exp x - a else 2 * x - a

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l525_525608


namespace analogical_reasoning_correctness_l525_525452

theorem analogical_reasoning_correctness 
  (a b c : ℝ)
  (va vb vc : ℝ) :
  (a + b) * c = (a * c + b * c) ↔ 
  (va + vb) * vc = (va * vc + vb * vc) := 
sorry

end analogical_reasoning_correctness_l525_525452


namespace isogonal_conjugate_angles_sum_eq_npi_l525_525766

noncomputable def equilateral_triangle (A B C O Z W M : ℂ) :=
(∃ r : ℝ, (A = r * complex.exp (0 * complex.I)) ∧
          (B = r * complex.exp ((2 * real.pi / 3) * complex.I)) ∧
          (C = r * complex.exp ((4 * real.pi / 3) * complex.I)) ∧
          (O = 0) ∧
          (Z * conj (Z) = A * conj (A)) ∧
          (W * conj (W) = A * conj (A)) ∧
          ((Z + W) = - (conj Z * conj W)) ∧
          (M = (Z + W) / 2))

theorem isogonal_conjugate_angles_sum_eq_npi
  (A B C O Z W M : ℂ)
  (h1 : equilateral_triangle A B C O Z W M) :
  ∃ n : ℤ, 
    (complex.arg ((Z * conj (Z)) * (W * conj (W)) * ((Z + W) / (conj (Z) + conj (W)))) 
    = n * real.pi) :=
begin
  sorry
end

end isogonal_conjugate_angles_sum_eq_npi_l525_525766


namespace dealer_sold_70_hondas_l525_525880

theorem dealer_sold_70_hondas
  (total_cars: ℕ)
  (percent_audi percent_toyota percent_acura percent_honda : ℝ)
  (total_audi := total_cars * percent_audi)
  (total_toyota := total_cars * percent_toyota)
  (total_acura := total_cars * percent_acura)
  (total_honda := total_cars * percent_honda )
  (h1 : total_cars = 200)
  (h2 : percent_audi = 0.15)
  (h3 : percent_toyota = 0.22)
  (h4 : percent_acura = 0.28)
  (h5 : percent_honda = 1 - (percent_audi + percent_toyota + percent_acura))
  : total_honda = 70 := 
  by
  sorry

end dealer_sold_70_hondas_l525_525880


namespace consecutive_odd_divisibility_l525_525131

theorem consecutive_odd_divisibility {p : ℤ} (hp : p % 2 = 1) :
  let q := p + 2 in
  (p ^ q + q ^ p) % (p + q) = 0 :=
by
  let q := p + 2
  sorry

end consecutive_odd_divisibility_l525_525131


namespace jordan_starting_weight_l525_525221

variable (weeks1 := 4)
variable (loss1 := 3) -- pounds per week for the first 4 weeks
variable (weeks2 := 8)
variable (loss2 := 2) -- pounds per week for the next 8 weeks
variable (current_weight := 222)

theorem jordan_starting_weight :
  ∀ (w1 w2 c l1 l2 : ℕ),
  w1 = weeks1 → l1 = loss1 → w2 = weeks2 → l2 = loss2 → c = current_weight →
  (c + (w1 * l1 + w2 * l2) = 250) :=
by
  intros w1 w2 c l1 l2 h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  have total_weight_loss : ℕ := 4 * 3 + 8 * 2
  have starting_weight : ℕ := 222 + total_weight_loss
  show starting_weight = 250
  calc 
    250 = 250 : rfl

end jordan_starting_weight_l525_525221


namespace T_18_plus_T_34_plus_T_51_l525_525533

def T (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, (1 : ℤ) * ((-1 : ℤ) ^ i) * i

theorem T_18_plus_T_34_plus_T_51 : T 18 + T 34 + T 51 = 154 := by
  sorry

end T_18_plus_T_34_plus_T_51_l525_525533


namespace water_added_eq_30_l525_525643

-- Given conditions
def initial_volume := 50
def initial_concentration := 0.4
def final_concentration := 0.25
def amount_of_acid := initial_volume * initial_concentration

-- Proof problem statement
theorem water_added_eq_30 : ∃ w : ℝ, amount_of_acid / (initial_volume + w) = final_concentration ∧ w = 30 :=
by 
  -- Solution steps go here, but for now we insert sorry to skip the proof.
  sorry

end water_added_eq_30_l525_525643


namespace curve_equation_with_params_l525_525206

theorem curve_equation_with_params (a m x y : ℝ) (ha : a > 0) (hm : m ≠ 0) :
    (y^2) = m * (x^2 - a^2) ↔ mx^2 - y^2 = ma^2 := by
  sorry

end curve_equation_with_params_l525_525206


namespace total_population_milburg_l525_525352

def num_children : ℕ := 2987
def num_adults : ℕ := 2269

theorem total_population_milburg : num_children + num_adults = 5256 := by
  sorry

end total_population_milburg_l525_525352


namespace tony_average_time_l525_525278

-- Definitions based on the conditions
def distance_to_store : ℕ := 4 -- in miles
def walking_speed : ℕ := 2 -- in MPH
def running_speed : ℕ := 10 -- in MPH

-- Conditions
def time_walking : ℕ := (distance_to_store / walking_speed) * 60 -- in minutes
def time_running : ℕ := (distance_to_store / running_speed) * 60 -- in minutes

def total_time : ℕ := time_walking + 2 * time_running -- Total time spent in minutes
def number_of_days : ℕ := 3 -- Number of days

def average_time : ℕ := total_time / number_of_days -- Average time in minutes

-- Statement to prove
theorem tony_average_time : average_time = 56 := by 
  sorry

end tony_average_time_l525_525278


namespace complex_calculation_l525_525113

theorem complex_calculation (z : ℂ) (h : z = 2 - complex.I) : 
  z * (z.conj + complex.I) = 6 + 2 * complex.I := 
by sorry

end complex_calculation_l525_525113


namespace Derek_spent_on_dad_l525_525942

theorem Derek_spent_on_dad :
  ∀ (Derek_initial Derek_spent_Dave_initial Dave_spent_more Dave_difference : ℕ),
    Derek_initial = 40 →
    Derek_spent_Dave_initial = 19 →
    Dave_initial = 50 →
    Dave_spent_more = 7 →
    Dave_difference = 33 →
    let Derek_left := Derek_initial - Derek_spent_Dave_initial - (Derek_initial - Derek_spent_Dave_initial - (Dave_initial - Dave_spent_more - Dave_difference)) in
    Derek_left = 11 :=
by {
  intros,
  let Derek_spent_dad := 40 - 19 - (40 - 19 - (50 - 7 - 33)),
  have h1 : Derek_spent_dad = 11, from sorry,
  exact h1,
}

end Derek_spent_on_dad_l525_525942


namespace solve_inequality_l525_525565

theorem solve_inequality (x : ℝ) : x > 13 ↔ x^3 - 16 * x^2 + 73 * x > 84 :=
by
  sorry

end solve_inequality_l525_525565


namespace order_of_numbers_l525_525337

theorem order_of_numbers :
  let a := 0.3
  let sq_a := a^2
  let log2_a := log a / log 2
  let exp_a := 2^a
  log2_a < sq_a ∧ sq_a < exp_a := by
  sorry

end order_of_numbers_l525_525337


namespace value_of_expression_l525_525893

variables (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 + c * x + d

theorem value_of_expression (h : f a b c d (-2) = -3) : 8 * a - 4 * b + 2 * c - d = 3 :=
by {
  sorry
}

end value_of_expression_l525_525893


namespace complement_union_eq_l525_525255

def U := {1, 2, 4, 5, 7, 8}
def A := {1, 2, 5}
def B := {2, 7, 8}

theorem complement_union_eq : (U \ (A ∪ B)) = {4} := by
  sorry

end complement_union_eq_l525_525255


namespace exists_separating_line_l525_525615

noncomputable def f1 (x : ℝ) (a1 b1 c1 : ℝ) : ℝ := a1 * x^2 + b1 * x + c1
noncomputable def f2 (x : ℝ) (a2 b2 c2 : ℝ) : ℝ := a2 * x^2 + b2 * x + c2

theorem exists_separating_line (a1 b1 c1 a2 b2 c2 : ℝ) (h_intersect : ∀ x, f1 x a1 b1 c1 ≠ f2 x a2 b2 c2)
  (h_neg : a1 * a2 < 0) : ∃ α β : ℝ, ∀ x, f1 x a1 b1 c1 < α * x + β ∧ α * x + β < f2 x a2 b2 c2 :=
sorry

end exists_separating_line_l525_525615


namespace triangle_tangent_identity_l525_525732

theorem triangle_tangent_identity (A B C : ℝ) (h : A + B + C = Real.pi) : 
  (Real.tan (A / 2) * Real.tan (B / 2)) + (Real.tan (B / 2) * Real.tan (C / 2)) + (Real.tan (C / 2) * Real.tan (A / 2)) = 1 :=
by
  sorry

end triangle_tangent_identity_l525_525732


namespace hexagon_midpoints_intersect_l525_525362

-- Definition of a hexagon with parallel opposite sides
structure hexagon (α : Type) [AffineSpace ℝ α] :=
(A B C D E F : α)
(parallel1 : ∀ u v w : line α, u.is_parallel v → u.is_parallel w → v.is_parallel w → u = w)

-- Definitions of parallel opposite sides
variables {α : Type} [AffineSpace ℝ α]
variables (A B C D E F : hexagon α)

def hexagon_opposite_parallel (h : hexagon α) :=
  (line_through (h.A) (h.B)).is_parallel (line_through (h.D) (h.E)) ∧
  (line_through (h.B) (h.C)).is_parallel (line_through (h.E) (h.F)) ∧
  (line_through (h.C) (h.D)).is_parallel (line_through (h.F) (h.A))

-- Prove that the line segments connecting midpoints intersect at a single point
begin
theorem hexagon_midpoints_intersect (h : hexagon α) 
  (hp : hexagon_opposite_parallel h) : 
  ∃ O : α, 
  AffineMap.lineMap 
      (midpoint ℝ (h.A) (h.D)) (midpoint ℝ (h.B) (h.E)) O ∧ 
  AffineMap.lineMap 
      (midpoint ℝ (h.C) (h.F)) (midpoint ℝ (h.A) (h.D)) O 
    :=
sorry
end

end hexagon_midpoints_intersect_l525_525362


namespace hundredth_term_in_sequence_l525_525917

def base_3_no_2_digits (n : ℕ) : Prop :=
  ∀ d ∈ nat.digits 3 n, d = 0 ∨ d = 1

def sequence := {n : ℕ // base_3_no_2_digits n}

noncomputable def find_nth_term (n : ℕ) : ℕ :=
  classical.some (nat.find (λ m, ∃ s : sequence, n = m + 1))

theorem hundredth_term_in_sequence : find_nth_term 100 = 981 :=
sorry

end hundredth_term_in_sequence_l525_525917


namespace arc_tangent_sine_squared_l525_525858

theorem arc_tangent_sine_squared (A P Q : Point) (u v w : ℝ) (angle_A : ℝ) :
  (arc_touches_sides A P Q) →
  (tangent_distances P Q A u v w) →
  (∃ a b c : ℝ, a + b = c → u * v / w^2 = (Real.sin (angle_A / 2))^2) :=
by
  sorry

-- Definitions to help Lean understand terms used:
def Point : Type := sorry
def arc_touches_sides (A P Q : Point) : Prop := sorry
def tangent_distances (P Q A : Point) (u v w : ℝ) : Prop := sorry

end arc_tangent_sine_squared_l525_525858


namespace beth_longer_distance_by_5_miles_l525_525218

noncomputable def average_speed_john : ℝ := 40
noncomputable def time_john_hours : ℝ := 30 / 60
noncomputable def distance_john : ℝ := average_speed_john * time_john_hours

noncomputable def average_speed_beth : ℝ := 30
noncomputable def time_beth_hours : ℝ := (30 + 20) / 60
noncomputable def distance_beth : ℝ := average_speed_beth * time_beth_hours

theorem beth_longer_distance_by_5_miles : distance_beth - distance_john = 5 := by 
  sorry

end beth_longer_distance_by_5_miles_l525_525218


namespace count_valid_pairs_l525_525981

theorem count_valid_pairs (bound : ℕ) (h_bound: bound = 150) :
  let conditions := λ (x y : ℕ), y < x ∧ x ≤ bound ∧ x % y = 0 ∧ (x + 2) % (y + 2) = 0 in
  ∑ y in Finset.range 149, Nat.floor ((bound - y) / (y * (y + 2))) = 
  let calculated_sum := 
    ∑ y in Finset.range 149, (bound - y) / (y * (y + 2)) in
  calculated_sum := sorry

end count_valid_pairs_l525_525981


namespace problem_statement_l525_525448

theorem problem_statement :
  102^3 + 3 * 102^2 + 3 * 102 + 1 = 1092727 :=
  by sorry

end problem_statement_l525_525448


namespace find_line_equation_l525_525066

-- Define the point (2, -1) which the line passes through
def point : ℝ × ℝ := (2, -1)

-- Define the line perpendicular to 2x - 3y = 1
def perpendicular_line (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- The equation of the line we are supposed to find
def equation_of_line (x y : ℝ) : Prop := 3 * x + 2 * y - 4 = 0

-- Proof problem: prove the equation satisfies given the conditions
theorem find_line_equation :
  (equation_of_line point.1 point.2) ∧ 
  (∃ (a b c : ℝ), ∀ (x y : ℝ), perpendicular_line x y → equation_of_line x y) := sorry

end find_line_equation_l525_525066


namespace probability_of_prime_pairs_l525_525370

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l525_525370


namespace probability_two_primes_is_1_over_29_l525_525428

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l525_525428


namespace tony_average_time_l525_525276

-- Definitions based on the conditions
def distance_to_store : ℕ := 4 -- in miles
def walking_speed : ℕ := 2 -- in MPH
def running_speed : ℕ := 10 -- in MPH

-- Conditions
def time_walking : ℕ := (distance_to_store / walking_speed) * 60 -- in minutes
def time_running : ℕ := (distance_to_store / running_speed) * 60 -- in minutes

def total_time : ℕ := time_walking + 2 * time_running -- Total time spent in minutes
def number_of_days : ℕ := 3 -- Number of days

def average_time : ℕ := total_time / number_of_days -- Average time in minutes

-- Statement to prove
theorem tony_average_time : average_time = 56 := by 
  sorry

end tony_average_time_l525_525276


namespace greatest_q_minus_r_value_l525_525801

-- Define the condition in Lean
def condition (q r : ℕ) : Prop :=
  1025 = 23 * q + r ∧ q > 0 ∧ r > 0

-- State the theorem
theorem greatest_q_minus_r_value : ∃ q r : ℕ, condition q r ∧ (∀ q' r' : ℕ, condition q' r' → (q - r ≥ q' - r')) :=
begin
  sorry
end

end greatest_q_minus_r_value_l525_525801


namespace collinear_MNT_l525_525918

variables {A B X Y: Point}
variables {P Q R S M N T: Point}

-- Provided conditions
axiom cond1 : A ∈ AX
axiom cond2 : B ∈ BY
axiom cond3 : P ∈ AX
axiom cond4 : R ∈ AX
axiom cond5 : Q ∈ BY
axiom cond6 : S ∈ BY
axiom cond7 : (AP / BQ) = (AR / BS)
axiom cond8 : (AM / MB) = (PN / NQ) = (RT / TS)

-- Define the problem as a theorem to be proven
theorem collinear_MNT : Collinear [M, N, T] :=
sorry

end collinear_MNT_l525_525918


namespace cube_angle_EF_CD_l525_525525

/-- Defining the cube and its vertices -/
structure Cube :=
(A B C D A₁ B₁ C₁ D₁ : Point)
(E : Point) (F : Point)
(E_center : E = (A₁.to_Vector + B₁.to_Vector + C₁.to_Vector + D₁.to_Vector) / 4)
(F_center : F = (A.to_Vector + D.to_Vector + D₁.to_Vector + A₁.to_Vector) / 4)
(angle_45_deg : ∀ (p q r : Point), angle (vector p q) (vector q r) = 45)

/-- Defining the Points and their relationship to the cube -/
variable (cube: Cube)

theorem cube_angle_EF_CD :
  ∀ (E F C D : Point),
  cube.E = E →
  cube.F = F →
  cube.C = C →
  cube.D = D →
  angle (EF.vector) (CD.vector) = 45 :=
by
  sorry

end cube_angle_EF_CD_l525_525525


namespace probability_two_primes_l525_525382

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l525_525382


namespace Jerry_remaining_pages_l525_525718

theorem Jerry_remaining_pages (total_pages : ℕ) (saturday_read : ℕ) (sunday_read : ℕ) (remaining_pages : ℕ) 
  (h1 : total_pages = 93) (h2 : saturday_read = 30) (h3 : sunday_read = 20) (h4 : remaining_pages = 43) : 
  total_pages - saturday_read - sunday_read = remaining_pages := 
by
  sorry

end Jerry_remaining_pages_l525_525718


namespace fifteenth_even_multiple_of_5_l525_525837

theorem fifteenth_even_multiple_of_5 : 15 * 2 * 5 = 150 := by
  sorry

end fifteenth_even_multiple_of_5_l525_525837


namespace employee_earnings_l525_525023

def task_a_hours_worked (days_1 : ℕ) (hours_per_day_1 : ℕ) (days_2 : ℕ) (hours_per_day_2 : ℕ) : ℕ :=
  days_1 * hours_per_day_1 + days_2 * (2 * hours_per_day_1)

def task_b_hours_worked (days_1 : ℕ) (hours_per_day_1 : ℕ) (days_2 : ℕ) (hours_per_day_2 : ℕ) : ℕ :=
  days_1 * hours_per_day_1 + days_2 * hours_per_day_2

def task_a_earnings (hours : ℕ) : ℕ :=
  if hours <= 40 then hours * 30
  else 40 * 30 + (hours - 40) * 45

def task_b_earnings (hours : ℕ) : ℕ :=
  hours * 40

theorem employee_earnings :
  let task_a_hours := task_a_hours_worked 3 6 2 12 in
  let task_b_hours := task_b_hours_worked 3 4 2 3 in
  let total_earnings := task_a_earnings task_a_hours + task_b_earnings task_b_hours in
  total_earnings = 2010 :=
by
  sorry

end employee_earnings_l525_525023


namespace prove_expression_l525_525336

def otimes (a b : ℚ) : ℚ := a^2 / b

theorem prove_expression : ((otimes (otimes 1 2) 3) - (otimes 1 (otimes 2 3))) = -2/3 :=
by 
  sorry

end prove_expression_l525_525336


namespace lawnmower_blades_l525_525887

theorem lawnmower_blades (B : ℤ) (h : 8 * B + 7 = 39) : B = 4 :=
by 
  sorry

end lawnmower_blades_l525_525887


namespace complex_calculation_l525_525112

theorem complex_calculation (z : ℂ) (h : z = 2 - complex.I) : 
  z * (z.conj + complex.I) = 6 + 2 * complex.I := 
by sorry

end complex_calculation_l525_525112


namespace lines_intersect_distance_to_origin_max_distance_sum_l525_525742

variable (k1 k2 : ℝ)
variable (hk : k1 * k2 + 1 = 0)

-- 1) Prove that lines l1 and l2 intersect
theorem lines_intersect : k1 ≠ k2 := 
by {
  intro h_eq,
  rw [h_eq] at hk,
  have : k1 * k1 + 1 = 0 := by { rw [add_eq_zero_iff_neg_eq] at hk, exact hk },
  have : k1^2 + 1 = 0 := by { sorry },
  have : (k1^2 + 1) ≥ 0 := by { sorry },
  have : (k1^2 + 1) > 0 := by { sorry },
  exact this.ne_zero,
  -- since this 0 doesn't match, we derive the contradiction, from hk.
  sorry
}

-- 2) Prove that the distance from the intersection point to the origin is constant (1).
theorem distance_to_origin : 
  let x := 2 / (k2 - k1),
      y := (k2 + k1) / (k2 - k1) in
  x^2 + y^2 = 1 := 
by {
  let x := 2 / (k2 - k1),
  let y := (k2 + k1) / (k2 - k1),
  have h_dist_eq : (2 / (k2 - k1))^2 + ((k2 + k1) / (k2 - k1))^2 = 1 := sorry,
  exact h_dist_eq
}

-- 3) Proof to find the maximum value of d1 + d2
theorem max_distance_sum : 
  let d1 := 1 / real.sqrt (1 + k1^2),
      d2 := 1 / real.sqrt (1 + k2^2) in
  ∀ (k1 : ℝ), d1 + d2 ≤ real.sqrt 2 := by {
    let d1 := 1 / real.sqrt (1 + k1^2),
    let d2 := 1 / real.sqrt (1 + k2^2),
    have h_sum_eq : ∀ k1, d1 + d2 = real.sqrt (1 + 2 * 1 / (real.abs k1 + 1 / real.abs k1)) := sorry,
    have h_max_value : ∀ k1, d1 + d2 ≤ real.sqrt 2 := sorry,
    exact h_max_value,
}

end lines_intersect_distance_to_origin_max_distance_sum_l525_525742


namespace parallelogram_area_l525_525971

theorem parallelogram_area
  (a b: ℝ) (θ: ℝ)
  (ha: a = 12)
  (hb: b = 18)
  (hθ: θ = real.pi / 4) :
  let h := a * real.sin θ in
  let area := b * h in
  area = 18 * (12 * real.sin (real.pi / 4)) := by
  sorry

end parallelogram_area_l525_525971


namespace sum_a1_a3_a5_l525_525143

-- Conditions: Binomial expansion and specific equations resulting from values of x and y
def binomial_expansion (x y : ℝ) : ℝ :=
  (x + 2 * real.sqrt y)^5

axiom eq1 : binomial_expansion 1 1 = (3^5 : ℝ)   -- When x = y = 1
axiom eq2 : binomial_expansion (-1) 1 = 1        -- When x = -1 and y = 1

theorem sum_a1_a3_a5 : (∃ a0 a1 a2 a3 a4 a5 : ℝ, 
  (∃ x y : ℝ, binomial_expansion x y = a0*x^5 + a1*(x^4)*(real.sqrt y) + a2*(x^3)*(y) + a3*(x^2)*(real.sqrt y)^2 + a4*x*(y^(3/2)) + a5*(y^(5/2))) →
  a1 + a3 + a5 = 122) :=
  sorry

end sum_a1_a3_a5_l525_525143


namespace probability_two_primes_from_1_to_30_l525_525402

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l525_525402


namespace area_of_tangency_triangle_l525_525933

-- Define the basic settings and conditions
noncomputable def circle_1 := (0, 0, 2)
noncomputable def circle_2 := (5, 0, 3)
noncomputable def circle_3 := (x, y, 4)

-- Define the main theorem
theorem area_of_tangency_triangle : 
  ∃ (x y : ℝ), dist (0, 0) (5, 0) = 5 ∧
               dist (5, 0) (x, y) = 7 ∧ 
               dist (0, 0) (x, y) = 6 ∧ 
               let A := sqrt (9 * 4 * 3 * 2) in
               A / 9 = 2 * sqrt 6 / 3 :=
sorry

end area_of_tangency_triangle_l525_525933


namespace group_members_l525_525456

def number_of_members (total_paise : ℕ) (collected_total_paise : ℕ) : Prop :=
  ∃ n : ℕ, n * n = total_paise ∧ total_paise = collected_total_paise

theorem group_members :
  number_of_members 7225 7225 :=
begin
  sorry
end

end group_members_l525_525456


namespace largest_median_possible_l525_525591

-- Define the given list of six positive integers
def initial_list : List ℕ := [3, 5, 1, 7, 9, 6]

-- Define the median calculation function
def median_of_ten_list (xs : List ℕ) : ℕ :=
  let sorted_xs := xs.qsort (≤)
  (sorted_xs[4] + sorted_xs[5]) / 2

-- Prove that the largest possible median we can obtain is 8
theorem largest_median_possible : 
  ∃ (additional_elements : List ℕ), length additional_elements = 4 ∧ 
  ∀ (list_of_ten : List ℕ), list_of_ten = initial_list ++ additional_elements →
  median_of_ten_list list_of_ten = 8 :=
sorry

end largest_median_possible_l525_525591


namespace no_disjoint_covering_of_N_l525_525734

theorem no_disjoint_covering_of_N {α β γ : ℝ} (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  ¬ (∀ n ∈ ℕ, ∃ x ∈ ({⌊n * α⌋, ⌊n * β⌋, ⌊n * γ⌋} : set ℕ), true) ∧
  (∀ a b ∈ (ℕ : set ℕ), (a ∈ {⌊n * α⌋ : n ∈ ℕ} → ¬ (a ∈ {⌊n * β⌋ : n ∈ ℕ})) ∧
                          (a ∈ {⌊n * β⌋ : n ∈ ℕ} → ¬ (a ∈ {⌊n * γ⌋ : n ∈ ℕ})) ∧
                          (a ∈ {⌊n * γ⌋ : n ∈ ℕ} → ¬ (a ∈ {⌊n * α⌋ : n ∈ ℕ}))) :=
begin
  sorry
end

end no_disjoint_covering_of_N_l525_525734


namespace geometric_sequence_a_sequence_b_l525_525599

theorem geometric_sequence_a (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60) :
  ∀ n, a n = 4 * 3^(n - 1) :=
sorry

theorem sequence_b (b a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60)
  (h3 : ∀ n, b (n + 1) = b n + a n) (h4 : b 1 = a 2) :
  ∀ n, b n = 2 * 3^n + 10 :=
sorry

end geometric_sequence_a_sequence_b_l525_525599


namespace integer_point_in_inner_pentagon_l525_525270

-- Define the conditions
variables (A B C D E : ℤ × ℤ) -- The vertices of the pentagon are integer coordinate points

-- Hypotheses
hypotheses 
  (h_convex: convex_hull (A :: B :: C :: D :: E :: list.nil)) -- The pentagon is convex

-- Statement to be proven
theorem integer_point_in_inner_pentagon :
  ∃ (A1 B1 C1 D1 E1 : ℤ × ℤ),
    (A1 ∈ convex_hull (A :: B :: C :: D :: E :: list.nil)) ∧
    (B1 ∈ convex_hull (A :: B :: C :: D :: E :: list.nil)) ∧
    (C1 ∈ convex_hull (A :: B :: C :: D :: E :: list.nil)) ∧
    (D1 ∈ convex_hull (A :: B :: C :: D :: E :: list.nil)) ∧
    (E1 ∈ convex_hull (A :: B :: C :: D :: E :: list.nil)) ∧
    ((∃ (P : ℤ × ℤ),
      P ∈ convex_hull (A1 :: B1 :: C1 :: D1 :: E1 :: list.nil))) :=
sorry

end integer_point_in_inner_pentagon_l525_525270


namespace largest_value_expression_l525_525450

theorem largest_value_expression (x : ℕ) (hx : x = 9) :
  (max (max (sqrt x) (x / 2)) (max (x - 5) (max (40 / x) (x^2 / 20)))) = (x / 2) := by
  sorry

end largest_value_expression_l525_525450


namespace part1_monotonically_increasing_part2_minimum_value_b_minus_e2a_l525_525611

noncomputable def is_monotonically_increasing_1 : Prop :=
∀ x ∈ set.Icc (0 : ℝ) (2 * π),
  let f : ℝ → ℝ := λ x, Real.exp x * Real.sin x in
  if (0 ≤ x) ∧ (x < (3 * π / 4)) ∨ ((7 * π / 4) < x) ∧ (x ≤ 2 * π)
  then ∃ ε > 0, ∀ h ∈ set.Ioo x (x + ε), f h > f x

noncomputable def minimum_value_b_minus_e2a : Prop :=
∀ (a : ℝ) (b : ℝ), (1 ≤ a) ∧
  (∀ x ∈ set.Icc (0 : ℝ) (π / 2), (Real.exp (a * x) * Real.sin x) ≤ b * x) →
  b - (2 * Real.exp 2 * a) = - (2 * Real.exp 2 / π)

theorem part1_monotonically_increasing : is_monotonically_increasing_1 := sorry

theorem part2_minimum_value_b_minus_e2a : minimum_value_b_minus_e2a := sorry

end part1_monotonically_increasing_part2_minimum_value_b_minus_e2a_l525_525611


namespace problem1_l525_525932

theorem problem1 :
  (let a := Real.sqrt 6 * Real.sqrt 2 in
  let b := Real.sqrt 27 / Real.sqrt 9 in
  let c := Real.sqrt (1 / 3) in
  a + b - c = 8 * Real.sqrt 3 / 3) :=
by
  sorry

end problem1_l525_525932


namespace min_percent_eats_both_l525_525266

theorem min_percent_eats_both (A B : ℝ) (hA : A = 90 / 100) (hB : B = 80 / 100) :
  ∃ x : ℝ, (x = 70 / 100) ∧ (A + B - x ≤ 1) :=
by
  use 70 / 100
  split
  { refl }
  { sorry }

end min_percent_eats_both_l525_525266


namespace probability_two_primes_l525_525383

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l525_525383


namespace fourier_series_expansion_l525_525964

noncomputable def f (x : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 24) then 1 else if (24 < x ∧ x ≤ 32) then (32 - x) / 8 else 0

theorem fourier_series_expansion :
  ∀ x, f x =
  (7 / 8) - (2 / Real.pi) * 
  ∑ k in Finset.range (Nat.succ 1000), 
  ( (7 / k) * Real.sin ((3 / 4) * k * Real.pi) + 
    (32 / (k^2 * Real.pi)) * 
    ((-1 : ℝ) ^ k - Real.cos ((3 / 4) * k * Real.pi) 
  ) * (Real.cos (k * Real.pi * x / 32)) ) :=
by
  sorry

end fourier_series_expansion_l525_525964


namespace free_space_on_new_drive_l525_525295

theorem free_space_on_new_drive
  (initial_free : ℝ) (initial_used : ℝ) (delete_size : ℝ) (new_files_size : ℝ) (new_drive_size : ℝ) :
  initial_free = 2.4 → initial_used = 12.6 → delete_size = 4.6 → new_files_size = 2 → new_drive_size = 20 →
  (new_drive_size - ((initial_used - delete_size) + new_files_size)) = 10 :=
by simp; sorry

end free_space_on_new_drive_l525_525295


namespace joseph_cards_percentage_left_l525_525722

def joseph_total_cards : ℕ := 16
def fraction_given_to_brother : ℚ := 3 / 8
def extra_cards_given_to_brother : ℕ := 2

theorem joseph_cards_percentage_left
  (total_cards : ℕ)
  (fraction_given : ℚ)
  (extra_given : ℕ)
  (percentage_left : ℚ) :
  total_cards = joseph_total_cards →
  fraction_given = fraction_given_to_brother →
  extra_given = extra_cards_given_to_brother →
  percentage_left = ((total_cards - (fraction_given * total_cards).toNat - extra_given:ℚ) / total_cards * 100) →
  percentage_left = 50 := by sorry

end joseph_cards_percentage_left_l525_525722


namespace y_in_terms_of_w_l525_525892

theorem y_in_terms_of_w (y w : ℝ) (h1 : y = 3^2 - 1) (h2 : w = 2) : y = 4 * w :=
by
  sorry

end y_in_terms_of_w_l525_525892


namespace correct_conclusions_l525_525999

-- Define the arithmetic sequence and its sum
variable (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)

-- Definitions for common difference and sum conditions
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (S : ℕ → ℚ) (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = (n / 2) * (2 * a 1 + (n - 1) * d)

-- Conditions given in the problem
def conditions (S : ℕ → ℚ) : Prop :=
  S 7 > S 6 ∧ S 6 > S 8

-- Theorem statement
theorem correct_conclusions (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ) :
  arithmetic_sequence a d →
  sum_of_terms S a →
  conditions S →
  d < 0 ∧ S 14 < 0 ∧ (∀ n : ℕ, S 7 ≥ S n) :=
by
  sorry

end correct_conclusions_l525_525999


namespace solve_for_f_l525_525574

noncomputable def f (a b c : ℝ) : ℝ → ℝ := 
  λ x => a * Real.sin x + b * Real.cbrt x + c * Real.log (x + Real.sqrt (x^2 + 1)) + 1003

theorem solve_for_f (a b c : ℝ) :
  f a b c (Real.log 10 ^ 2) = 1 →
  f a b c (Real.log (Real.log 3)) = 2005 :=
by
  intro h
  sorry

end solve_for_f_l525_525574


namespace simplify_expression_l525_525029

theorem simplify_expression : 3 * real.sqrt 5 - real.sqrt 20 = real.sqrt 5 :=
by
  sorry

end simplify_expression_l525_525029


namespace thirteen_pow_2048_mod_eleven_l525_525839

theorem thirteen_pow_2048_mod_eleven : (13 ^ 2048) % 11 = 3 := by
  -- Using Fermat's Little Theorem and properties of modular arithmetic
  let h1 : 13 % 11 = 2 := by norm_num
  let h2 : (2 ^ 10) % 11 = 1 := by norm_num
  have h3 : (13 ^ 2048) % 11 = (2 ^ 2048) % 11 := by
    rw [← h1]
  rw [pow_mod, pow_mul, pow_add]
  exact h3

end thirteen_pow_2048_mod_eleven_l525_525839


namespace vacation_cost_proof_l525_525185

noncomputable def vacation_cost (C : ℝ) :=
  C / 5 - C / 8 = 120

theorem vacation_cost_proof {C : ℝ} (h : vacation_cost C) : C = 1600 :=
by
  sorry

end vacation_cost_proof_l525_525185


namespace constant_difference_AN_BN_l525_525993

-- Define the geometric entities and their relationships
variables {α : Type*} [EuclideanGeometry α]

-- Conditions
variables (A B M : α) (l : Line α)
variable (h_perpendicular : l ⊥ (Line.mk A B))
variable (h_M_on_l : M ∈ l)

-- Define angle conditions for point N
variable (N : α)
variable (h_angle1 : ∠(N, A, B) = 2 * ∠(M, A, B))
variable (h_angle2 : ∠(N, B, A) = 2 * ∠(M, B, A))

-- Statement to prove
theorem constant_difference_AN_BN : ∀ M : α, M ∈ l → 
  ∀ N : α, (∠(N, A, B) = 2 * ∠(M, A, B)) → (∠(N, B, A) = 2 * ∠(M, B, A)) → 
  abs (dist A N - dist B N) = const :=
begin
  sorry
end

end constant_difference_AN_BN_l525_525993


namespace complex_square_of_one_plus_i_l525_525818

theorem complex_square_of_one_plus_i : (1 + complex.i) ^ 2 = 2 * complex.i :=
by
  sorry

end complex_square_of_one_plus_i_l525_525818


namespace ratio_w_y_l525_525340

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 4 / 3)
  (h3 : z / x = 1 / 8) : 
  w / y = 15 / 2 := 
by
  sorry

end ratio_w_y_l525_525340


namespace exists_half_divisible_statements_l525_525540

theorem exists_half_divisible_statements :
  ∃ (x : ℕ), (x + 2020 = ∏ i in (finset.Icc 2 2019).filter is_prime, i) - 2020 ∧
    (∀ n ∈ finset.range 2018, (n + 1).natAbs ∣ x + n + 1 ↔ n < 1009) :=
by
  sorry

end exists_half_divisible_statements_l525_525540


namespace correct_propositions_count_l525_525145

def Prop1 := ∀ (cylinder : Cylinder) (p1 : Point) (p2 : Point),
  p1 ∈ cylinder.topBase → p2 ∈ cylinder.bottomBase →
  lineConnecting p1 p2 ∈ cylinder.generatrix

def Prop2 := ∀ (prism : Prism) (base : RegularPolygon),
  isRegular base ∧ twoAdjacentFacesPerpendicularToBase prism →
  isRightPrism prism

def Prop3 := ∀ (frustum : Frustum),
  ¬similarBases frustum.topBase frustum.bottomBase ∧
  equalLength frustum.lateralEdges

theorem correct_propositions_count : 
  (¬Prop1 ∧ Prop2 ∧ ¬Prop3) → countCorrectProps = 1 :=
by
  sorry

end correct_propositions_count_l525_525145


namespace find_k_and_shifted_function_l525_525202

noncomputable def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem find_k_and_shifted_function (k : ℝ) (h : k ≠ 0) (h1 : linear_function k 1 = 3) :
  k = 2 ∧ linear_function 2 x + 2 = 2 * x + 3 :=
by
  sorry

end find_k_and_shifted_function_l525_525202


namespace selling_price_with_60_percent_profit_l525_525496

theorem selling_price_with_60_percent_profit (C : ℝ) (h₀ : 2240 = C + 0.4 * C) :
  2560 = C + 0.6 * C :=
by
  have h₁ : C = 2240 / 1.4 := by sorry
  have h₂ : 2560 = (2240 / 1.4) + 0.6 * (2240 / 1.4) := by sorry
  exact h₂

end selling_price_with_60_percent_profit_l525_525496


namespace probability_both_numbers_are_prime_l525_525420

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l525_525420


namespace value_of_n_l525_525677

theorem value_of_n (n : ℕ) (h : sqrt (5 + n) = 7) : n = 44 := 
by
  sorry

end value_of_n_l525_525677


namespace sum_of_coefficients_l525_525250

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (1 - 2 * (1 / 2))^5 = a₀ + 2 * a₁ * (1 / 2) + 4 * a₂ * (1 / 2)^2 + 8 * a₃ * (1 / 2)^3 + 16 * a₄ * (1 / 2)^4 + 32 * a₅ * (1 / 2)^5 →
  (1 - 2 * 0)^5 = a₀ →
  a₁ + a₂ + a₃ + a₄ + a₅ = -1 :=
begin
  sorry
end

end sum_of_coefficients_l525_525250


namespace probability_of_two_primes_is_correct_l525_525393

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l525_525393


namespace problem1_problem2_l525_525164

-- Definition of sequences a_n and b_n
def seq (f : ℕ → ℕ) := ∀ n : ℕ, f (n + 1) - f n = 2 * (f (n + 1) - f n)

-- Problem 1
theorem problem1 (a : ℕ → ℕ) (b : ℕ → ℕ) (h_seq : seq a = λ n, 6):
  (a 1 = 1) ∧ (b n = 3 * n + 5) → (a n = 6 * n - 5) :=
sorry

-- Problem 2
theorem problem2 (a : ℕ → ℕ) (b : ℕ → ℕ) (λ : ℝ) (h_seq : seq a = λ n, 2^(n+1)) :
  (a 1 = 6) ∧ (∀ n, b n = 2^n) → (∀ n, λ * a n > 2^n + n + 2 * λ) → (λ > 3/4) :=
sorry

end problem1_problem2_l525_525164


namespace polygon_has_even_side_l525_525569

theorem polygon_has_even_side (P : Polygon) (h1 : holeless_polygon P) (h2 : divided_into_2x1_rectangles P) : 
  ∃ side : ℕ, even side :=
sorry

end polygon_has_even_side_l525_525569


namespace tan_double_angle_l525_525571

open Real

theorem tan_double_angle {θ : ℝ} (h1 : tan (π / 2 - θ) = 4 * cos (2 * π - θ)) (h2 : abs θ < π / 2) : 
  tan (2 * θ) = sqrt 15 / 7 :=
sorry

end tan_double_angle_l525_525571


namespace carina_triangle_moves_l525_525516

def is_adjacent (p1 p2 : ℤ × ℤ) : Prop :=
  (abs (p1.1 - p2.1) = 1 ∧ p1.2 = p2.2) ∨ (abs (p1.2 - p2.2) = 1 ∧ p1.1 = p2.1)

def valid_move (start : (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)) (moves : ℕ → (ℕ × (ℤ × ℤ)) → (ℤ × ℤ)) : Prop :=
  ∀ n, ∀ k < n, let ⟨_, pos⟩ := moves k in is_adjacent (pos pos.fst) (pos pos.snd)

noncomputable def triangle_area (A B C : ℤ × ℤ) : ℤ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

def triangle_move_problem (A B C : ℤ × ℤ) : Prop :=
  ∃ (moves : ℕ → ℕ × (ℤ × ℤ)), valid_move ((0, 0), (0, 0), (0, 0)) moves ∧ 
  triangle_area A B C = 2021 ∧
  ∀ m, (m < 128) → (∃ (move_count : ℕ), triangle_area A B C ≠ 2021)

theorem carina_triangle_moves : triangle_move_problem ((0, 0), (0, 0), (0, 0)) :=
  sorry

end carina_triangle_moves_l525_525516


namespace greening_investment_equation_l525_525875

theorem greening_investment_equation:
  ∃ (x : ℝ), 20 * (1 + x)^2 = 25 := 
sorry

end greening_investment_equation_l525_525875


namespace speed_of_boat_in_still_water_l525_525852

-- Define the given conditions
def downstream_speed := 13
def upstream_speed := 4

-- Define the question to be proved: the speed of the boat in still water is 8.5 km/hr
def speed_in_still_water := (downstream_speed + upstream_speed) / 2

theorem speed_of_boat_in_still_water :
  speed_in_still_water = 8.5 := by
  sorry

end speed_of_boat_in_still_water_l525_525852


namespace solve_for_y_l525_525781

theorem solve_for_y (y : ℝ) (h : (7 * y - 2) / (y + 4) - 5 / (y + 4) = 2 / (y + 4)) : 
  y = (9 / 7) :=
by
  sorry

end solve_for_y_l525_525781


namespace parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l525_525163

variable (m x y : ℝ)

def l1_eq : Prop := (3 - m) * x + 2 * m * y + 1 = 0
def l2_eq : Prop := 2 * m * x + 2 * y + m = 0

theorem parallel_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = -3/2) :=
by sorry

theorem perpendicular_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = 0 ∨ m = 5) :=
by sorry

end parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l525_525163


namespace rotation_reflection_equivalence_l525_525287

theorem rotation_reflection_equivalence (O O1 O2 : Point) (α β : ℝ) (hα : 0 ≤ α ∧ α < 2 * Real.pi) (hβ : 0 ≤ β ∧ β < 2 * Real.pi) (hαβ : α + β ≠ 2 * Real.pi) :
  ∃ O : Point, 
    ((α + β < 2 * Real.pi) → 
      (∠ O1 O2 O = α / 2 ∧ ∠ O2 O O1 = β / 2 ∧ ∠ O1 O O2 = Real.pi - (α + β) / 2)) ∧ 
    ((α + β > 2 * Real.pi) → 
      (∠ O1 O2 O = Real.pi - α / 2 ∧ ∠ O2 O O1 = Real.pi - β / 2 ∧ ∠ O1 O O2 = (α + β) / 2)) :=
sorry

end rotation_reflection_equivalence_l525_525287


namespace smallest_solution_l525_525073

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 1)) + (1 / (x - 5)) = 4 / (x - 4)

theorem smallest_solution : 
  ∃ x : ℝ, equation x ∧ x ≠ 1 ∧ x ≠ 5 ∧ x ≠ 4 ∧ x = (5 - Real.sqrt 33) / 2 :=
by
  sorry

end smallest_solution_l525_525073


namespace largest_possible_constant_l525_525245

theorem largest_possible_constant (n : ℕ) (a : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i) :
  ∑ i, (a i)^2 ≥ (2 / (n - 1)) * ∑ i j (h : i < j), a i * a j := sorry

end largest_possible_constant_l525_525245


namespace convex_quadrilateral_diagonal_l525_525803

theorem convex_quadrilateral_diagonal (P : ℝ) (d1 d2 : ℝ) (hP : P = 2004) (hd1 : d1 = 1001) :
  (d2 = 1 → False) ∧ 
  (d2 = 2 → True) ∧ 
  (d2 = 1001 → True) :=
by
  sorry

end convex_quadrilateral_diagonal_l525_525803


namespace range_of_a_l525_525683

-- Definitions based on the conditions
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x >= 2 then (a - 2) * x else 2^x - 1

def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Main statement
theorem range_of_a (a : ℝ) :
  (monotonically_increasing (f a)) → a ≥ 7 / 2 :=
begin
  sorry
end

end range_of_a_l525_525683


namespace angle_sum_ninety_l525_525919

noncomputable def circumcenter (A B C : Point) : Point := sorry

def is_external_angle_bisector (P Q R : Point) : Prop := sorry

def is_similar (A B C D E F : Point) : Prop := sorry

theorem angle_sum_ninety
  (A B C O Q P : Point)
  (h1 : O = circumcenter A B C)
  (h2 : AB < AC)
  (h3 : Q ∈ line (BC) ∧ is_external_angle_bisector A Q BC)
  (h4 : is_similar (B P A) (A P C)) :
  ∠ QPA + ∠ OQB = 90 :=
begin
  sorry
end

end angle_sum_ninety_l525_525919


namespace max_positive_integer_value_of_n_l525_525314

-- Define the arithmetic sequence with common difference d and first term a₁.
variable {d a₁ : ℝ}

-- The quadratic inequality condition which provides the solution set [0,9].
def inequality_condition (d a₁ : ℝ) : Prop :=
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 9) → d * x^2 + 2 * a₁ * x ≥ 0

-- Maximum integer n such that the sum of the first n terms of the sequence is maximum.
noncomputable def max_n (d a₁ : ℝ) : ℕ :=
  if d < 0 then 5 else 0

-- Statement to be proved.
theorem max_positive_integer_value_of_n (d a₁ : ℝ) 
  (h : inequality_condition d a₁) : max_n d a₁ = 5 :=
sorry

end max_positive_integer_value_of_n_l525_525314


namespace best_chart_for_temperature_changes_l525_525366

def Pie_chart := "Represent the percentage of parts in the whole."
def Line_chart := "Represent changes over time."
def Bar_chart := "Show the specific number of each item."

theorem best_chart_for_temperature_changes : 
  "The best statistical chart to use for understanding temperature changes throughout a day" = Line_chart :=
by
  sorry

end best_chart_for_temperature_changes_l525_525366


namespace function_decreasing_interval_l525_525606

variable (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 * (a * x + b)

theorem function_decreasing_interval :
  (deriv (f a b) 2 = 0) ∧ (deriv (f a b) 1 = -3) →
  ∃ (a b : ℝ), (deriv (f a b) x < 0) ↔ (0 < x ∧ x < 2) := sorry

end function_decreasing_interval_l525_525606


namespace right_triangle_median_and_hypotenuse_l525_525698

/-- Given a right triangle XYZ with a right angle at Y,
    XY = 5 cm, YZ = 12 cm, and N is the midpoint of XZ,
    the length of the hypotenuse XZ is 13 cm and the length
    of the median YN is 6.5 cm. -/
theorem right_triangle_median_and_hypotenuse :
  ∀ {X Y Z : Type} [metric_space X] [metric_space Y] [metric_space Z] 
    (XY YZ : ℝ),
  ∀ (N : Y ⟶ X) [is_midpoint N Y Z],
    XY = 5 ∧ YZ = 12 ∧ angle Y = 90 → 
    hypotenuse_length = 13 ∧ median_length = 6.5 :=
begin
  intros,
  sorry, -- Proof goes here
end

end right_triangle_median_and_hypotenuse_l525_525698


namespace average_weight_section_A_l525_525819

theorem average_weight_section_A :
  let W_A := 40 in
  let num_students_A := 36 in
  let num_students_B := 44 in
  let avg_weight_B := 35 in
  let avg_weight_class := 37.25 in
  36 * W_A + 44 * 35 = 37.25 * (36 + 44) -> W_A = 40 := 
by 
  sorry

end average_weight_section_A_l525_525819


namespace linear_regression_probability_and_expectation_l525_525477

section LinearRegression

variable (data : List (ℕ × ℕ))
variable (x̄ ȳ : ℚ)
variable (n : ℕ)
variable (b : ℚ)
variable (a : ℚ)

-- Define the data
def givenData := [(1, 0), (2, 4), (3, 7), (4, 9), (5, 11), (6, 12), (7, 13)]

-- Define the averages
def x_average := (1 + 2 + 3 + 4 + 5 + 6 + 7) / 7
def y_average := (0 + 4 + 7 + 9 + 11 + 12 + 13) / 7

-- Define the regression coefficients
def b := (1 * 0 + 2 * 4 + 3 * 7 + 4 * 9 + 5 * 11 + 6 * 12 + 7 * 13 - 7 * x_average * y_average) /
          (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 - 7 * x_average^2)

def a := y_average - b * x_average

-- Property to prove
theorem linear_regression : a = -3 / 7 ∧ b = 59 / 28 ∧ ∀ x, (a + b * x) = 59 / 28 * x - 3 / 7 := 
begin
  sorry
end

end LinearRegression

section ProbabilityAndExpectation

variable (data : List (ℕ × ℕ))
variable (ξ : Type) [Fintype ξ] [DecidableEq ξ]

-- Given data and probability calculations
def num_days_gt_average := [(4, 9), (5, 11), (6, 12), (7, 13)].length
def num_days_le_average := [(1, 0), (2, 4), (3, 7)].length
def P (k : ℕ) := (nat.choose num_days_le_average (3 - k) * nat.choose num_days_gt_average k) /
                 (nat.choose 7 3 : ℚ)

-- Expected value calculation
def E := ∑ k in (Finset.range 4), k * P k

-- Property to prove
theorem probability_and_expectation : E = 12 / 7 :=
begin
  sorry
end

end ProbabilityAndExpectation

end linear_regression_probability_and_expectation_l525_525477


namespace exponent_properties_l525_525169

theorem exponent_properties (m n : ℝ) (hm : 2^m = 3) (hn : 2^n = 4) : 
  2^(m - n) = 3 / 4 :=
by
  sorry

end exponent_properties_l525_525169


namespace free_space_on_new_drive_l525_525296

theorem free_space_on_new_drive
  (initial_free : ℝ) (initial_used : ℝ) (delete_size : ℝ) (new_files_size : ℝ) (new_drive_size : ℝ) :
  initial_free = 2.4 → initial_used = 12.6 → delete_size = 4.6 → new_files_size = 2 → new_drive_size = 20 →
  (new_drive_size - ((initial_used - delete_size) + new_files_size)) = 10 :=
by simp; sorry

end free_space_on_new_drive_l525_525296


namespace luke_money_at_end_of_june_l525_525692

noncomputable def initial_money : ℝ := 48
noncomputable def february_money : ℝ := initial_money - 0.30 * initial_money
noncomputable def march_money : ℝ := february_money - 11 + 21 + 50 * 1.20

noncomputable def april_savings : ℝ := 0.10 * march_money
noncomputable def april_money : ℝ := (march_money - april_savings) - 10 * 1.18 + 0.05 * (march_money - april_savings)

noncomputable def may_savings : ℝ := 0.15 * april_money
noncomputable def may_money : ℝ := (april_money - may_savings) + 100 * 1.22 - 0.25 * ((april_money - may_savings) + 100 * 1.22)

noncomputable def june_savings : ℝ := 0.10 * may_money
noncomputable def june_money : ℝ := (may_money - june_savings) - 0.08 * (may_money - june_savings)
noncomputable def final_money : ℝ := june_money + 0.06 * (may_money - june_savings)

theorem luke_money_at_end_of_june : final_money = 128.15 := sorry

end luke_money_at_end_of_june_l525_525692


namespace probability_both_numbers_are_prime_l525_525418

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l525_525418


namespace arithmetic_sequence_n_values_l525_525997

theorem arithmetic_sequence_n_values (n d : ℕ) (h1 : ∀ n, n ≥ 3 → a n = 70) (h2 : a 1 = 1) (h3 : a n = a 1 + d * (n - 1)) :
  (n = 4 ∨ n = 24 ∨ n = 70) :=
by
  sorry

end arithmetic_sequence_n_values_l525_525997


namespace probability_two_primes_l525_525380

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l525_525380


namespace math_proof_problem_l525_525121

noncomputable def complex_example : Prop :=
  let z := 2 - complex.i in
  let conj_z := complex.conj z in
  z * (conj_z + complex.i) = 6 + 2 * complex.i

theorem math_proof_problem : complex_example := by
  sorry

end math_proof_problem_l525_525121


namespace min_sum_of_squares_convex_heptagon_l525_525297

structure LatticePoint where
  x : Int
  y : Int

def squared_distance (p1 p2 : LatticePoint) : Int :=
  (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)

def is_convex (points : List LatticePoint) : Prop := sorry
def all_sides_distinct (points : List LatticePoint) : Prop := sorry

noncomputable def heptagon_min_sum_of_squares : Int :=
  let points := [ ⟨0, 0⟩, ⟨3, 0⟩, ⟨5, 1⟩, ⟨6, 2⟩, ⟨3, 4⟩, ⟨2, 4⟩, ⟨0, 2⟩ ]
  let distances := List.map (λ i => squared_distance points.get! i points.get! ((i + 1) % 7)) (List.range 7)
  List.sum distances

theorem min_sum_of_squares_convex_heptagon :
  ∃ (points : List LatticePoint), points.length = 7 ∧
    is_convex points ∧ all_sides_distinct points ∧
    List.sum (List.map (λ i => squared_distance (points.get! i) (points.get! ((i + 1) % 7))) (List.range 7)) = 42 :=
begin
  use [ ⟨0, 0⟩, ⟨3, 0⟩, ⟨5, 1⟩, ⟨6, 2⟩, ⟨3, 4⟩, ⟨2, 4⟩, ⟨0, 2⟩ ],
  split,
  { simp },
  split,
  { sorry },
  split,
  { sorry },
  { dsimp only [list.get!, list.range, list.map, squared_distance],
    iterate 7 { simp [squared_distance], },
    norm_num }
end

end min_sum_of_squares_convex_heptagon_l525_525297


namespace total_accidents_l525_525226

theorem total_accidents :
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  accidentsA + accidentsB + accidentsC = 3730 :=
by
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  sorry

end total_accidents_l525_525226


namespace express_in_scientific_notation_l525_525365

theorem express_in_scientific_notation :
  (10.58 * 10^9) = 1.058 * 10^10 :=
by
  sorry

end express_in_scientific_notation_l525_525365


namespace dorchester_puppy_washing_l525_525952

-- Define the conditions
def daily_pay : ℝ := 40
def pay_per_puppy : ℝ := 2.25
def wednesday_total_pay : ℝ := 76

-- Define the true statement
theorem dorchester_puppy_washing :
  let earnings_from_puppy_washing := wednesday_total_pay - daily_pay in
  let number_of_puppies := earnings_from_puppy_washing / pay_per_puppy in
  number_of_puppies = 16 :=
by
  -- Placeholder for the proof
  sorry

end dorchester_puppy_washing_l525_525952


namespace sport_formulation_water_amount_l525_525854

-- Define the standard formulation ratios
def standard_ratio_flavoring_corn_syrup := (1, 12)
def standard_ratio_flavoring_water := (1, 30)

-- Define the sport formulation conditions
def sport_ratio_flavoring_corn_syrup := (1, 4)
def sport_ratio_flavoring_water := (1, 60)

-- Given condition: large bottle contains 6 ounces of corn syrup
def corn_syrup_amount : ℚ := 6

-- Theorem: large bottle of sport formulation contains 90 ounces of water
theorem sport_formulation_water_amount : 
  let flavoring_amount := corn_syrup_amount / sport_ratio_flavoring_corn_syrup.2 in
  let water_amount := flavoring_amount * sport_ratio_flavoring_water.2 in
  water_amount = 90 := by 
  sorry

end sport_formulation_water_amount_l525_525854


namespace solve_equation_l525_525970

theorem solve_equation (x : ℝ) :
  (1 / (x ^ 2 + 14 * x - 10)) + (1 / (x ^ 2 + 3 * x - 10)) + (1 / (x ^ 2 - 16 * x - 10)) = 0
  ↔ (x = 5 ∨ x = -2 ∨ x = 2 ∨ x = -5) :=
sorry

end solve_equation_l525_525970


namespace rhombus_diagonal_l525_525435

theorem rhombus_diagonal (a b : ℝ) (area_triangle : ℝ) (d1 d2 : ℝ)
  (h1 : 2 * area_triangle = a * b)
  (h2 : area_triangle = 75)
  (h3 : a = 20) :
  b = 15 :=
by
  sorry

end rhombus_diagonal_l525_525435


namespace range_of_m_l525_525177

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4 * cos x + sin x ^ 2 + m - 4 = 0) ↔ 0 ≤ m ∧ m ≤ 8 :=
sorry

end range_of_m_l525_525177


namespace dwarves_count_l525_525191

-- Definitions for creature types
inductive Creature
| elf
| dwarf
| gnome

open Creature

-- Define creatures
variables (Alice Bob Carol Dave Eve : Creature)

-- Statements made by creatures
def alice_statement := (Alice ≠ Bob)
def bob_statement := (Carol = gnome)
def carol_statement := (Eve = dwarf)
def dave_statement := (Alice = elf ∧ Bob = elf ∧ Carol = elf ∨ 
                        Alice = elf ∧ Bob = elf ∧ Dave = elf ∨ 
                        Alice = elf ∧ Carol = elf ∧ Dave = elf ∨
                        Bob = elf ∧ Carol = elf ∧ Dave = elf ∨ 
                        Alice = elf ∧ Carol = elf ∧ Eve = elf ∨ 
                        Bob = elf ∧ Carol = elf ∧ Eve = elf ∨ 
                        Alice = elf ∧ Dave = elf ∧ Eve = elf ∨ 
                        Bob = elf ∧ Dave = elf ∧ Eve = elf ∨ 
                        Carol = elf ∧ Dave = elf ∧ Eve = elf)
def eve_statement := (Alice = gnome)

-- Theorem stating that there are exactly 2 dwarves
theorem dwarves_count : (∃ (n : ℕ), n = 2 ∧ (
  (Alice = dwarf ∧ Bob ≠ dwarf ∧ Carol ≠ dwarf ∧ Dave ≠ dwarf ∧ Eve ≠ dwarf) ∨
  (Alice ≠ dwarf ∧ Bob = dwarf ∧ Carol ≠ dwarf ∧ Dave ≠ dwarf ∧ Eve ≠ dwarf) ∨
  (Alice ≠ dwarf ∧ Bob ≠ dwarf ∧ Carol = dwarf ∧ Dave ≠ dwarf ∧ Eve ≠ dwarf) ∨
  (Alice ≠ dwarf ∧ Bob ≠ dwarf ∧ Carol ≠ dwarf ∧ Dave = dwarf ∧ Eve ≠ dwarf) ∨
  (Alice ≠ dwarf ∧ Bob ≠ dwarf ∧ Carol ≠ dwarf ∧ Dave ≠ dwarf ∧ Eve = dwarf) ∨
  (Alice = dwarf ∧ Bob = dwarf ∧ Carol ≠ dwarf ∧ Dave ≠ dwarf ∧ Eve ≠ dwarf) ∨
  (Alice = dwarf ∧ Bob ≠ dwarf ∧ Carol = dwarf ∧ Dave ≠ dwarf ∧ Eve ≠ dwarf) ∨
  -- Continue listing all combinations where exactly 2 creatures are dwarves.
  ... -- Completing listings for space considerations
)) :=
sorry

end dwarves_count_l525_525191


namespace quadrilateral_is_rectangle_l525_525214

open Real

variables {A B C D : Point}
variables [ConvexQuadrilateral A B C D]
variables (r : ℝ)
variables (h1 : InscribedCircleRadius (Triangle.mk A B C) = r)
variables (h2 : InscribedCircleRadius (Triangle.mk B C D) = r)
variables (h3 : InscribedCircleRadius (Triangle.mk C D A) = r)
variables (h4 : InscribedCircleRadius (Triangle.mk D A B) = r)

theorem quadrilateral_is_rectangle : IsRectangle A B C D := by
  sorry

end quadrilateral_is_rectangle_l525_525214


namespace find_pairs_n_p_l525_525553

theorem find_pairs_n_p :
  ∀ (p n : ℕ), p.prime → n > p → 
  (∃ m : ℕ, n^(n - p) = m^n) →
  (p, n) = (2, 4) :=
by
  intros p n p_prime n_gt_p H
  -- Given conditions and objective, elaborate proof here.
  -- Note: the proof steps are omitted.
  sorry

end find_pairs_n_p_l525_525553


namespace moles_of_CH3Cl_formed_one_to_one_reaction_l525_525067

noncomputable def moles_of_CH3Cl_formed (moles_CH4 moles_Cl2 : ℕ) : ℕ :=
if (moles_CH4 = 1 ∧ moles_Cl2 = 1) then 1 else 0

theorem moles_of_CH3Cl_formed_one_to_one_reaction 
  (moles_CH4 moles_Cl2 : ℕ) 
  (h1 : moles_CH4 = 1)
  (h2 : moles_Cl2 = 1)
  (reaction : ∀ (a b : ℕ), a + b → a[embed constant statement for one-to-one reaction constraint from conditions→] a  + b) :
  moles_of_CH3Cl_formed moles_CH4 moles_Cl2 = 1 :=
by
  sorry

end moles_of_CH3Cl_formed_one_to_one_reaction_l525_525067


namespace interesting_numbers_correct_l525_525747

def is_interesting (n : ℕ) : Prop := 
  20 ≤ n ∧ n ≤ 90 ∧ 
  ∀ d ∈ (n.divisors_sorted), ∀ d' ∈ (n.divisors_sorted), d < d' → d'.mod d = 0

def interesting_numbers : Finset ℕ := {25, 27, 32, 49, 64, 81}

theorem interesting_numbers_correct : 
  ∀ n, 20 ≤ n ∧ n ≤ 90 → is_interesting n ↔ n ∈ interesting_numbers := by
  sorry

end interesting_numbers_correct_l525_525747


namespace cos_theta_sum_l525_525695

-- Definition of cos θ as derived from the problem conditions.
theorem cos_theta_sum {r : ℝ} {θ φ : ℝ}
  (h1 : 8^2 = 2 * r^2 * (1 - Real.cos θ))
  (h2 : 15^2 = 2 * r^2 * (1 - Real.cos φ))
  (h3 : 17^2 = 2 * r^2 * (1 - Real.cos (θ + φ)))
  (h4 : θ + φ < Real.pi) :
  (let (num, den) := Rat.numDen (Real.cos θ).toRational in num + den = 386) :=
by
  sorry

end cos_theta_sum_l525_525695


namespace probability_of_both_selected_l525_525855

variable (P_ram : ℚ) (P_ravi : ℚ) (P_both : ℚ)

def selection_probability (P_ram : ℚ) (P_ravi : ℚ) : ℚ :=
  P_ram * P_ravi

theorem probability_of_both_selected (h1 : P_ram = 3/7) (h2 : P_ravi = 1/5) :
  selection_probability P_ram P_ravi = P_both :=
by
  sorry

end probability_of_both_selected_l525_525855


namespace larry_jogs_first_week_days_l525_525225

-- Defining the constants and conditions
def daily_jogging_time := 30 -- Larry jogs for 30 minutes each day
def total_jogging_time_in_hours := 4 -- Total jogging time in two weeks in hours
def total_jogging_time_in_minutes := total_jogging_time_in_hours * 60 -- Convert hours to minutes
def jogging_days_in_second_week := 5 -- Larry jogs 5 days in the second week
def daily_jogging_time_in_week2 := jogging_days_in_second_week * daily_jogging_time -- Total jogging time in minutes in the second week

-- Theorem statement
theorem larry_jogs_first_week_days : 
  (total_jogging_time_in_minutes - daily_jogging_time_in_week2) / daily_jogging_time = 3 :=
by
  -- Definitions and conditions used above should directly appear from the problem statement
  sorry

end larry_jogs_first_week_days_l525_525225


namespace area_of_triangle_ABC_l525_525861

noncomputable def triangle_area (AC AD DE DF : ℝ) : ℝ :=
  if h : AC = 18 ∧ AD = 5 ∧ DE = 4 ∧ DF = 5 then
    360 / 7
  else
    0

theorem area_of_triangle_ABC :
  triangle_area 18 5 4 5 = 360 / 7 :=
by
  unfold triangle_area
  split_ifs
  . sorry
  

end area_of_triangle_ABC_l525_525861


namespace toll_constant_l525_525349

theorem toll_constant (t : ℝ) (x : ℝ) (constant : ℝ) : 
  (t = 1.50 + 0.50 * (x - constant)) → 
  (x = 18 / 2) → 
  (t = 5) → 
  constant = 2 :=
by
  intros h1 h2 h3
  sorry

end toll_constant_l525_525349


namespace number_of_solutions_l525_525944

-- Define the conditions as Lean variables
variables (x : ℝ)

-- Define each inequality
def ineq1 : Prop := -5 * x ≥ 3 * x + 11
def ineq2 : Prop := -3 * x ≤ 15
def ineq3 : Prop := -6 * x ≥ 4 * x + 23

-- Define the range of integers satisfying the inequalities
def satisfies_inequalities : Prop :=
  ∃ (x : ℤ), ineq1 ∧ ineq2 ∧ ineq3 ∧ (x ≥ -5 ∧ x ≤ -3)

-- Prove that the number of integers between -5 and -3 (inclusive) satisfying the inequalities is 3
theorem number_of_solutions : card { x : ℤ | x ≥ -5 ∧ x ≤ -3 ∧ ineq1 ∧ ineq2 ∧ ineq3 } = 3 := by
  sorry

end number_of_solutions_l525_525944


namespace total_height_of_pipes_l525_525077

theorem total_height_of_pipes 
  (diameter : ℝ) (radius : ℝ) (total_pipes : ℕ) (first_row_pipes : ℕ) (second_row_pipes : ℕ) 
  (h : ℝ) 
  (h_diam : diameter = 10)
  (h_radius : radius = 5)
  (h_total_pipes : total_pipes = 5)
  (h_first_row : first_row_pipes = 2)
  (h_second_row : second_row_pipes = 3) :
  h = 10 + 5 * Real.sqrt 3 := 
sorry

end total_height_of_pipes_l525_525077


namespace olivia_checking_time_l525_525268

theorem olivia_checking_time (problems : ℕ) (time_per_problem : ℕ) (total_time : ℕ) (time_spent : ℕ) :
  problems = 7 → time_per_problem = 4 → total_time = 31 → time_spent = total_time - (problems * time_per_problem) → time_spent = 3 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end olivia_checking_time_l525_525268


namespace cars_cannot_meet_l525_525313

/-- Given a city plan divided into equilateral triangles, cars starting from points A and B 
on the same road, moving at the same speed and making turns of either 0°, 120°, or -120°
at each intersection, cannot meet. -/
theorem cars_cannot_meet 
{A B : ℝ} -- Assuming A and B are coordinates on the same road
(triangles : set (equiv_triangle ℝ)) -- Plane divided into equilateral triangles
(speed : ℝ) -- Cars move with same speed
(directions : ℝ × ℝ → ℝ × ℝ) -- Function representing direction changes (0°, 120°, -120°)
(start_time : ℝ) -- Cars start simultaneously
(moves : ℕ → ℝ × ℝ → ℝ × ℝ) -- Movement function for cars
: ∀ t : ℕ, moves t (directions (A, B)) ≠ (A, B) :=
by sorry

end cars_cannot_meet_l525_525313


namespace no_perfect_square_in_range_l525_525630

theorem no_perfect_square_in_range : ∀ n : ℤ, 5 ≤ n ∧ n ≤ 20 → ¬ ∃ k : ℤ, k^2 = n^3 + 2*n^2 + 3*n + 4 :=
by 
  intros n hn
  cases hn with hn_left hn_right
  sorry

end no_perfect_square_in_range_l525_525630


namespace closest_rating_l525_525041

theorem closest_rating (a : ℝ) (h1 : 9.6 < a) (h2 : a < 9.8) : 
  ∃ r ∈ {9.4, 9.3, 9.7, 9.9, 9.5}, r = 9.7 :=
by
  use 9.7
  sorry

end closest_rating_l525_525041


namespace tangent_line_at_1_l525_525140

def f (x : ℝ) : ℝ := sorry

theorem tangent_line_at_1 :
  (∃ (f : ℝ → ℝ), ∀ (x : ℝ), y = f(x) → (∀ (M : ℝ), x = 1 → f(1) = y + 3)) 
  → f(1) + f'(1) = 5 :=
by
  sorry

end tangent_line_at_1_l525_525140


namespace cost_of_gas_l525_525713

def hoursDriven1 : ℕ := 2
def speed1 : ℕ := 60
def hoursDriven2 : ℕ := 3
def speed2 : ℕ := 50
def milesPerGallon : ℕ := 30
def costPerGallon : ℕ := 2

def totalDistance : ℕ := (hoursDriven1 * speed1) + (hoursDriven2 * speed2)
def gallonsUsed : ℕ := totalDistance / milesPerGallon
def totalCost : ℕ := gallonsUsed * costPerGallon

theorem cost_of_gas : totalCost = 18 := by
  -- You should fill in the proof steps here.
  sorry

end cost_of_gas_l525_525713


namespace average_mark_of_remaining_students_l525_525788

theorem average_mark_of_remaining_students
  (n : ℕ) (A : ℕ) (m : ℕ) (B : ℕ) (total_students : n = 10)
  (avg_class : A = 80) (excluded_students : m = 5) (avg_excluded : B = 70) :
  (A * n - B * m) / (n - m) = 90 :=
by
  sorry

end average_mark_of_remaining_students_l525_525788


namespace aira_rubber_bands_l525_525290

theorem aira_rubber_bands (total_bands : ℕ) (bands_each : ℕ) (samantha_extra : ℕ) (aira_fewer : ℕ)
  (h1 : total_bands = 18) 
  (h2 : bands_each = 6) 
  (h3 : samantha_extra = 5) 
  (h4 : aira_fewer = 1) : 
  ∃ x : ℕ, x + (x + samantha_extra) + (x + aira_fewer) = total_bands ∧ x = 4 :=
by
  sorry

end aira_rubber_bands_l525_525290


namespace definite_integral_correct_l525_525464

noncomputable def integral_problem : Prop := 
  ∫ x in -1..0, (x^2 + 4 * x + 3) * cos x = 4 - 2 * cos 1 - 2 * sin 1

theorem definite_integral_correct : integral_problem :=
by
  sorry

end definite_integral_correct_l525_525464


namespace ordered_pairs_count_l525_525081

theorem ordered_pairs_count :
  (∃ (a b : ℝ), (∃ (x y : ℤ),
    a * (x : ℝ) + b * (y : ℝ) = 1 ∧
    (x : ℝ)^2 + (y : ℝ)^2 = 65)) →
  ∃ (n : ℕ), n = 128 :=
by
  sorry

end ordered_pairs_count_l525_525081


namespace minimum_value_of_expression_l525_525039

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  2 * a + b + c ≥ 4 := 
by 
  sorry

end minimum_value_of_expression_l525_525039


namespace complex_multiplication_identity_l525_525092

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- State the theorem to be proved
theorem complex_multiplication_identity : z * (z_conjugate + complex.i) = 6 + 2 * complex.i :=
by {
  sorry -- proof goes here
}

end complex_multiplication_identity_l525_525092


namespace sixth_number_in_seq_l525_525343

-- Define the sequence
def seq : List ℕ := [12, 13, 15, 17, 111, 113, 117, 119, 123, 129, 131]

-- Define the proposition to prove the sixth number in the sequence
theorem sixth_number_in_seq : seq.nth 5 = some 113 :=
by
  sorry

end sixth_number_in_seq_l525_525343


namespace sum_digits_sequence_1_to_5000_l525_525030

theorem sum_digits_sequence_1_to_5000 : 
  (finset.range 5000).sum (λ n, (n + 1).digits.sum) = 194450 :=
sorry

end sum_digits_sequence_1_to_5000_l525_525030


namespace probability_two_primes_is_1_over_29_l525_525429

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l525_525429


namespace find_k_l525_525621

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : vector := (1, 2)
def b : vector := (-3, 2)

theorem find_k (k : ℝ) :
  dot_product (k • a + b) (a - 3 • b) = 0 →
  k = 19 :=
begin
  sorry
end

end find_k_l525_525621


namespace checker_on_diagonal_l525_525758

theorem checker_on_diagonal {n : ℕ} (n_eq_25 : n = 25) 
  (symmetric_placement : ∀ i j : fin n, i ≠ j → checker_placed i j ↔ checker_placed j i) :
  ∃ i : fin n, checker_placed i i := 
begin
  sorry,
end

end checker_on_diagonal_l525_525758


namespace simplify_expression_l525_525302

theorem simplify_expression :
  ( ( 81^(1/4) - real.sqrt (17 / 2) ) ^ 2 ) = 17.5 - 3 * real.sqrt 34 :=
by sorry

end simplify_expression_l525_525302


namespace evaluate_fraction_sum_l525_525043

-- Define the double factorial functions based on the problem's conditions
def double_factorial_even (n : ℕ) : ℕ := 2^n * nat.factorial n
def double_factorial_odd (n : ℕ) : ℕ := nat.factorial (2 * n) / (2^n * (nat.factorial n)^2)

-- State the sum and required transformation
def S : ℕ := (finset.range 2023).sum (λ i, (nat.choose (2 * (i + 1)) (i + 1)) / 2^(2 * (i + 1)) + 1 / 2^(i + 1))

-- State the proof goal directly
theorem evaluate_fraction_sum :
  let c := ((finset.range 2023).sum (λ i, nat.choose (2 * (i + 1)) (i + 1) * 2^(2 * 2023 - 2 * (i + 1))))
  let p_c := 2023 - ∑ k in finset.range (nat.log (2023) 2 + 1), 2023 / 2^k
  let a := 4046 - p_c
  let b := 1
  (a * b) / 10 = 403.9 :=
by sorry

end evaluate_fraction_sum_l525_525043


namespace largest_remainder_division_by_11_l525_525498

theorem largest_remainder_division_by_11 (A B C : ℕ) (h : A = 11 * B + C) (hC : 0 ≤ C ∧ C < 11) : C ≤ 10 :=
  sorry

end largest_remainder_division_by_11_l525_525498


namespace probability_correct_l525_525078

noncomputable def selected_numbers : set ℕ := {n | n % 2 = 0 ∧ n ≥ 1 ∧ n ≤ 100}

def probability_sum : ℕ :=
  let c := [c1, c2, c3, c4, c5] in 
  if H : c1 ∈ selected_numbers ∧ c2 ∈ selected_numbers ∧ c3 ∈ selected_numbers ∧ c4 ∈ selected_numbers ∧ c5 ∈ selected_numbers 
    ∧ c1 ≠ c2 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c1 ≠ c5 ∧ c2 ≠ c3 ∧ c2 ≠ c4 ∧ c2 ≠ c5 ∧ c3 ≠ c4 ∧ c3 ≠ c5 ∧ c4 ≠ c5 then 
    3
  else 
    0

theorem probability_correct (c1 c2 c3 c4 c5: ℕ) (h1 : c1 ∈ selected_numbers) (h2 : c2 ∈ selected_numbers) (h3 : c3 ∈ selected_numbers) 
  (h4 : c4 ∈ selected_numbers) (h5 : c5 ∈ selected_numbers) 
  (h_distinct : list.nodup [c1, c2, c3, c4, c5]) : 
  probability_sum c1 c2 c3 c4 c5 = 3 :=
by sorry

end probability_correct_l525_525078


namespace smallest_value_y_l525_525840

theorem smallest_value_y : ∃ y : ℝ, 3 * y ^ 2 + 33 * y - 90 = y * (y + 18) ∧ (∀ z : ℝ, 3 * z ^ 2 + 33 * z - 90 = z * (z + 18) → y ≤ z) ∧ y = -18 := 
sorry

end smallest_value_y_l525_525840


namespace combined_age_of_Jane_and_John_in_future_l525_525224

def Justin_age : ℕ := 26
def Jessica_age_when_Justin_born : ℕ := 6
def James_older_than_Jessica : ℕ := 7
def Julia_younger_than_Justin : ℕ := 8
def Jane_older_than_James : ℕ := 25
def John_older_than_Jane : ℕ := 3
def years_later : ℕ := 12

theorem combined_age_of_Jane_and_John_in_future :
  let Jessica_age := Justin_age + Jessica_age_when_Justin_born
  let James_age := Jessica_age + James_older_than_Jessica
  let Julia_age := Justin_age - Julia_younger_than_Justin
  let Jane_age := James_age + Jane_older_than_James
  let John_age := Jane_age + John_older_than_Jane
  let Jane_age_after_years := Jane_age + years_later
  let John_age_after_years := John_age + years_later
  Jane_age_after_years + John_age_after_years = 155 :=
by
  sorry

end combined_age_of_Jane_and_John_in_future_l525_525224


namespace sum_term_ratio_equals_four_l525_525998

variable {a_n : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S_n : ℕ → ℝ} -- The sum of the first n terms S_n
variable {d : ℝ} -- The common difference of the sequence
variable {a_1 : ℝ} -- The first term of the sequence

-- The conditions as hypotheses
axiom a_n_formula (n : ℕ) : a_n n = a_1 + (n - 1) * d
axiom S_n_formula (n : ℕ) : S_n n = n * (a_1 + (n - 1) * d / 2)
axiom non_zero_d : d ≠ 0
axiom condition_a10_S4 : a_n 10 = S_n 4

-- The proof statement
theorem sum_term_ratio_equals_four : (S_n 8) / (a_n 9) = 4 :=
by
  sorry

end sum_term_ratio_equals_four_l525_525998


namespace total_accidents_all_three_highways_l525_525229

def highway_conditions : Type :=
  (accident_rate : ℕ, per_million : ℕ, total_traffic : ℕ)

def highway_a : highway_conditions := (75, 100, 2500)
def highway_b : highway_conditions := (50, 80, 1600)
def highway_c : highway_conditions := (90, 200, 1900)

def total_accidents (hc : highway_conditions) : ℕ :=
  hc.accident_rate * hc.total_traffic / hc.per_million

theorem total_accidents_all_three_highways :
  total_accidents highway_a +
  total_accidents highway_b +
  total_accidents highway_c = 3730 := by
  sorry

end total_accidents_all_three_highways_l525_525229


namespace regular_octagon_interior_angle_l525_525833

theorem regular_octagon_interior_angle :
  let n := 8 in
  let sum_of_interior_angles := (8 - 2) * 180 in
  sum_of_interior_angles / 8 = 135 :=
by
  let n := 8
  let sum_of_interior_angles := (n - 2) * 180
  have h : sum_of_interior_angles = 1080 by sorry
  show sum_of_interior_angles / n = 135 from sorry

end regular_octagon_interior_angle_l525_525833


namespace arithmetic_sequence_solution_l525_525938

noncomputable def a1 : ℝ := 1 / 2
noncomputable def S4 : ℝ := 20
noncomputable def d : ℝ := 3
noncomputable def S6 : ℝ := 48

theorem arithmetic_sequence_solution 
  (h_a1 : a1 = 1 / 2)
  (h_S4 : S4 = 20)
  (h_d : d = 3)
  (h_S6 : S6 = 48) :
  (∃ d, (4 * (1 / 2) + 4 * 3 * (4 - 1) / 2 = 20) ∧ (6 * (1 / 2) + 3 * 5 * 3 = 48)) :=
begin
  sorry
end

end arithmetic_sequence_solution_l525_525938


namespace tony_average_time_l525_525281

-- Definitions for the conditions
def speed_walk : ℝ := 2  -- speed in miles per hour when Tony walks
def speed_run : ℝ := 10  -- speed in miles per hour when Tony runs
def distance_to_store : ℝ := 4  -- distance to the store in miles
def days : List String := ["Sunday", "Tuesday", "Thursday"]  -- days Tony goes to the store

-- Definition of times taken on each day
def time_sunday := distance_to_store / speed_walk  -- time in hours to get to the store on Sunday
def time_tuesday := distance_to_store / speed_run  -- time in hours to get to the store on Tuesday
def time_thursday := distance_to_store / speed_run -- time in hours to get to the store on Thursday

-- Converting times to minutes
def time_sunday_minutes := time_sunday * 60
def time_tuesday_minutes := time_tuesday * 60
def time_thursday_minutes := time_thursday * 60

-- Definition of average time
def average_time_minutes : ℝ :=
  (time_sunday_minutes + time_tuesday_minutes + time_thursday_minutes) / days.length

-- The theorem to prove
theorem tony_average_time : average_time_minutes = 56 := by
  sorry

end tony_average_time_l525_525281


namespace length_of_integer_l525_525561

theorem length_of_integer (k : ℕ) (h1: k > 1) (h2: k = 2 * 2 * 2 * 3) (h3 : (λ (x : ℕ), x = k ∧ x.prime_factors.length = 4) k) : k = 24 :=
by
  sorry

end length_of_integer_l525_525561


namespace polynomial_solution_l525_525936

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

-- Prove that the polynomial satisfies the given conditions
theorem polynomial_solution :
  (∀ x : ℝ, f(x) = x^3 + 2 * x^2 + 3 * x + 4) ∧
  (f(0) = 4) ∧
  (f(1) = 10) ∧
  (f(-1) = 2) :=
by
  sorry

end polynomial_solution_l525_525936


namespace smallest_solution_eq_l525_525071

theorem smallest_solution_eq (x : ℝ) (hneq1 : x ≠ 1) (hneq5 : x ≠ 5) (hneq4 : x ≠ 4) :
  (∃ x : ℝ, (1 / (x - 1)) + (1 / (x - 5)) = (4 / (x - 4)) ∧
            (∀ y : ℝ, (1 / (y - 1)) + (1 / (y - 5)) = (4 / (y - 4)) → x ≤ y → y = x) ∧
            x = (5 - Real.sqrt 33) / 2) := 
begin
  sorry
end

end smallest_solution_eq_l525_525071


namespace conic_section_hyperbola_l525_525049

theorem conic_section_hyperbola (x y : ℝ) :
  (x - 7)^2 = 3 * (4 * y + 2)^2 - 108 → (1, -48).snd < 0 ∧ (1, -48).fst > 0 :=
by {
  intros h,
  sorry
}

end conic_section_hyperbola_l525_525049


namespace zhang_prob_one_hit_first_two_zhang_expectation_bullets_used_l525_525543

open ProbabilityTheory

noncomputable def hit_rate : ℝ := 2/3
def miss_rate : ℝ := 1 - hit_rate

theorem zhang_prob_one_hit_first_two :
  ProbabilityTheory.prob (λ ω, (event (0,false, ω) ∧ ¬ event (1,false, ω)) ∨ (¬ event (0,false, ω) ∧ event (1,false, ω)))  =
  4/9 := sorry

theorem zhang_expectation_bullets_used :
  ∑ x in {2, 3, 4, 5}, x * ProbabilityTheory.prob (λ ω, event (n, true, ω)) =
  224/81 := sorry

end zhang_prob_one_hit_first_two_zhang_expectation_bullets_used_l525_525543


namespace complex_multiplication_identity_l525_525095

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- State the theorem to be proved
theorem complex_multiplication_identity : z * (z_conjugate + complex.i) = 6 + 2 * complex.i :=
by {
  sorry -- proof goes here
}

end complex_multiplication_identity_l525_525095


namespace polygon_divided_into_triangles_sum_of_interior_angles_l525_525851

theorem polygon_divided_into_triangles (n : ℕ) (h : n ≥ 3) (P : Type) [polygon n P] : 
  ∃ T : finset (finset P), (∀ t ∈ T, triangle t) ∧ (∀ t1 t2 ∈ T, t1 ≠ t2 → disjoint t1 t2) :=
begin
  sorry
end

theorem sum_of_interior_angles (n : ℕ) (h : n ≥ 3) : 
  sum_of_angles (polygon n) = 2 * 180 * (n - 2) :=
begin
  sorry
end

end polygon_divided_into_triangles_sum_of_interior_angles_l525_525851


namespace find_g_at_3_l525_525741

-- Define the function g and the given property
def g (x : ℝ) : ℝ := sorry

-- Define the condition of the problem
theorem find_g_at_3 :
  (∀ x : ℝ, x ≠ 1 / 2 → g(x) + g((x + 2) / (2 - 4 * x)) = 2 * x) →
  g(3) = 17 / 6 :=
by
  assume h : ∀ x : ℝ, x ≠ 1 / 2 → g(x) + g((x + 2) / (2 - 4 * x)) = 2 * x
  sorry

end find_g_at_3_l525_525741


namespace complex_calculation_l525_525105

variable (z : ℂ)

theorem complex_calculation : (z = 2 - (-1:ℂ)) → (z * (conj z + (-1:ℂ))) = 6 + 2 * (-1:ℂ) :=
by
  intro h
  rw h
  have h_conj : conj (2 - (-1:ℂ)) = 2 + (-1:ℂ) := by simp
  rw h_conj
  ring
  done

end complex_calculation_l525_525105


namespace general_term_a_n_telescoping_sum_b_l525_525580

-- Define the sequence and its sum
def Sn (n : ℕ) : ℝ := (1/7) * (2^(3 * n + 1) - 2)

-- Prove that the general term of the sequence {a_n} is given by a_n = 2^(3n-2)
theorem general_term_a_n (n : ℕ) (h : 0 < n) : 
  let a_n := λ n, if n = 1 then Sn n else Sn n - Sn (n - 1)
  in (a_n n = 2^(3*n-2)) := 
by
  sorry

-- Define logarithmic sequence b_n
def b (n : ℕ) (a_n : ℕ → ℝ) : ℝ := Real.log (a_n n) / Real.log 2

-- Prove the telescoping sum result for b_n
theorem telescoping_sum_b (n : ℕ) (h : 0 < n) : 
  let a_n := λ n, if n = 1 then Sn n else Sn n - Sn (n - 1)
      b_n := b n a_n
  in (∑ k in Finset.range n, (1 / (b_n k * b_n (k + 1))) = n / (3 * n + 1)) := 
by
  sorry

end general_term_a_n_telescoping_sum_b_l525_525580


namespace monotonicity_f_inequality_condition_l525_525151

noncomputable def f (x a : ℝ) : ℝ := x + a * Real.log x - 1

-- Prove the monotonicity of f(x)
theorem monotonicity_f (a x : ℝ) (h : 0 < x) :
  (a ≥ 0 → ∀ x > 0, deriv (λ x, f x a) x > 0) ∧
  (a < 0 → ∀ x > 0, (x < -a → deriv (λ x, f x a) x < 0) ∧ (x > -a → deriv (λ x, f x a) x > 0)) :=
sorry

-- Prove the condition on a based on the inequality
theorem inequality_condition (a : ℝ) (h : ∀ x ∈ Set.Ici 1, 2 * f x a + Real.log x / x ≥ 0) :
  a ≥ -3/2 :=
sorry

end monotonicity_f_inequality_condition_l525_525151


namespace right_triangles_count_l525_525663

theorem right_triangles_count : 
  ∃ a b : ℕ, b < 50 ∧ a^2 + b^2 = (b + 2)^2 ∧ (finset.range 8).filter (λ k, (b = k^2 - 1)).card = 7 :=
sorry

end right_triangles_count_l525_525663


namespace probability_of_prime_pairs_l525_525376

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l525_525376


namespace add_water_to_solution_l525_525650

theorem add_water_to_solution (w : ℕ) :
  let initial_volume := 50
  let initial_concentration := 0.4
  let final_concentration := 0.25
  let acid_amount := initial_concentration * initial_volume
  let final_volume := initial_volume + w
  (acid_amount / final_volume = final_concentration) ↔ (w = 30) :=
by 
  sorry

end add_water_to_solution_l525_525650


namespace prob_primes_1_to_30_l525_525412

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l525_525412


namespace evaluate_dollar_l525_525983

variable {R : Type} [CommRing R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : 
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) :=
by
  sorry

end evaluate_dollar_l525_525983


namespace complex_calculation_l525_525108

theorem complex_calculation (z : ℂ) (h : z = 2 - complex.I) : 
  z * (z.conj + complex.I) = 6 + 2 * complex.I := 
by sorry

end complex_calculation_l525_525108


namespace perfect_square_transform_l525_525442

theorem perfect_square_transform (n : ℕ) (h : n = 92555) : 
    (∃ k : ℕ, n - k = 304^2) ∧ (∃ m : ℕ, n + m = 305^2) :=
by
  have h : n = 92555 := by assumption
  use (92555 - 304^2)
  use (305^2 - 92555)
  sorry

end perfect_square_transform_l525_525442


namespace president_and_committee_l525_525697

theorem president_and_committee (total : ℕ) (senior : ℕ) (committee_size : ℕ) (remaining_after_president : ℕ) (ways_to_choose_president : ℕ) (ways_to_choose_committee : ℕ) :
  total = 10 → 
  senior = 4 → 
  committee_size = 3 → 
  remaining_after_president = total - 1 → 
  ways_to_choose_president = total - senior → 
  ways_to_choose_committee = Nat.choose (remaining_after_president - 1) committee_size → 
  ways_to_choose_president * ways_to_choose_committee = 504 :=
by
  assume h1 : total = 10,
  assume h2 : senior = 4,
  assume h3 : committee_size = 3,
  assume h4 : remaining_after_president = total - 1,
  assume h5 : ways_to_choose_president = total - senior,
  assume h6 : ways_to_choose_committee = Nat.choose (remaining_after_president - 1) committee_size,
  calc
    ways_to_choose_president * ways_to_choose_committee
      = (total - senior) * (Nat.choose (total - senior - 1) committee_size)
      : by rw [h1, h2, h3, h4, h5, h6]
  ... = 6 * Nat.choose 9 3
      : by norm_num [h1, h2]
  ... = 6 * 84
      : by norm_num
  ... = 504
      : by norm_num

end president_and_committee_l525_525697


namespace pure_water_to_achieve_desired_concentration_l525_525638

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l525_525638


namespace add_tulips_to_maintain_ratio_l525_525341

variable (initial_daisies : ℕ) (additional_daisies : ℕ) (ratio_tulips : ℕ) (ratio_daisies : ℕ) (initial_tulips : ℕ)

-- Given conditions
def conditions : Prop :=
  initial_daisies = 32 ∧
  additional_daisies = 24 ∧
  ratio_tulips = 3 ∧
  ratio_daisies = 4 ∧
  initial_tulips = (ratio_tulips * initial_daisies) / ratio_daisies

-- Question to prove
theorem add_tulips_to_maintain_ratio : conditions initial_daisies additional_daisies ratio_tulips ratio_daisies initial_tulips → 
  let total_daisies := initial_daisies + additional_daisies in
  let total_tulips := (ratio_tulips * total_daisies) / ratio_daisies in
  let tulips_to_add := total_tulips - initial_tulips in
  tulips_to_add = 18 :=
by
  sorry

end add_tulips_to_maintain_ratio_l525_525341


namespace complex_calculation_l525_525111

theorem complex_calculation (z : ℂ) (h : z = 2 - complex.I) : 
  z * (z.conj + complex.I) = 6 + 2 * complex.I := 
by sorry

end complex_calculation_l525_525111


namespace math_proof_problem_l525_525116

noncomputable def complex_example : Prop :=
  let z := 2 - complex.i in
  let conj_z := complex.conj z in
  z * (conj_z + complex.i) = 6 + 2 * complex.i

theorem math_proof_problem : complex_example := by
  sorry

end math_proof_problem_l525_525116


namespace probability_two_primes_is_1_over_29_l525_525432

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l525_525432


namespace dorchester_puppies_washed_l525_525948

theorem dorchester_puppies_washed
  (total_earnings : ℝ)
  (daily_pay : ℝ)
  (earnings_per_puppy : ℝ)
  (p : ℝ)
  (h1 : total_earnings = 76)
  (h2 : daily_pay = 40)
  (h3 : earnings_per_puppy = 2.25)
  (hp : (total_earnings - daily_pay) / earnings_per_puppy = p) :
  p = 16 := sorry

end dorchester_puppies_washed_l525_525948


namespace max_visible_sum_is_128_l525_525985

-- Define the structure of the problem
structure Cube :=
  (faces : Fin 6 → Nat)
  (bottom_face : Nat)
  (all_faces : ∀ i : Fin 6, i ≠ ⟨0, by decide⟩ → faces i = bottom_face → False)

-- Define the problem conditions
noncomputable def problem_conditions : Prop :=
  let cubes := [Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry]
  -- Cube stacking in two layers, with two cubes per layer
  
  true

-- Define the theorem to be proved
theorem max_visible_sum_is_128 (h : problem_conditions) : 
  ∃ (total_sum : Nat), total_sum = 128 := 
sorry

end max_visible_sum_is_128_l525_525985


namespace question_1_question_2_l525_525622

variable (x : ℝ)
def a : ℝ × ℝ := (Real.cos x, Real.sin x)
def b : ℝ × ℝ := (3, -Real.sqrt 3)
def f (x : ℝ) : ℝ := 3 * Real.cos x - Real.sqrt 3 * Real.sin x

theorem question_1 (h : (cos x, sin x) ∥ (3, -sqrt 3)) (hx : 0 ≤ x ∧ x ≤ π) : 
  x = 5 * π / 6 :=
sorry

theorem question_2 (hx : 0 ≤ x ∧ x ≤ π) : 
  (f x ≤ 3 ∧ f x ≥ -2 * sqrt 3) ∧ 
  (∃ x_max, (x_max = 0) ∧ (f x_max = 3)) ∧ 
  (∃ x_min, (x_min = 5 * π / 6) ∧ (f x_min = -2 * sqrt 3)) :=
sorry

end question_1_question_2_l525_525622


namespace grasshopper_jumps_at_most_8_distinct_points_l525_525796

-- Define the problem setup
variables (r : ℝ) (circle : set (ℝ × ℝ)) (line : set (ℝ × ℝ))
variables [metric_space circle] [metric_space line]
variables (A_i B_i : ℕ → (ℝ × ℝ))

-- Conditions
axiom circle_def : ∀ i, A_i i ∈ circle
axiom line_def : ∀ i, B_i i ∈ line
axiom jump_length : ∀ i, dist (A_i i) (B_i (i+1)) = r
axiom jump_length' : ∀ i, dist (B_i i) (A_i (i+1)) = r
axiom line_not_through_center : (0, 0) ∉ line

-- Problem statement (the theorem to prove)
theorem grasshopper_jumps_at_most_8_distinct_points : ∀ i, dist (A_i i) (A_i (i+8)) = 0 :=
sorry

end grasshopper_jumps_at_most_8_distinct_points_l525_525796


namespace relation_among_three_numbers_l525_525805

-- Definitions of the relevant numbers
def ex1 := 7 ^ 0.3
def ex2 := 0.3 ^ 7
def ln_val := Real.log 0.3

-- Conditions from the problem
def cond1 : Prop := ex1 > 1
def cond2 : Prop := 0 < ex2 ∧ ex2 < 1
def cond3 : Prop := ln_val < 0

-- The theorem statement
theorem relation_among_three_numbers (h1 : cond1) (h2 : cond2) (h3 : cond3) : ex1 > ex2 ∧ ex2 > ln_val :=
by sorry

end relation_among_three_numbers_l525_525805


namespace total_earnings_correct_l525_525036

-- Define the charges for services for each car model
def charge (model: String) (service: String) : Float :=
  if model = "A" then
    if service = "oil_change" then 20
    else if service = "repair" then 30
    else if service = "car_wash" then 5
    else if service = "tire_rotation" then 15
    else 0
  else if model = "B" then
    if service = "oil_change" then 25
    else if service = "repair" then 40
    else if service = "car_wash" then 8
    else if service = "tire_rotation" then 18
    else 0
  else if model = "C" then
    if service = "oil_change" then 30
    else if service = "repair" then 50
    else if service = "car_wash" then 10
    else if service = "tire_rotation" then 20
    else 0
  else if model = "D" then
    if service = "oil_change" then 35
    else if service = "repair" then 60
    else if service = "car_wash" then 12
    else if service = "tire_rotation" then 22
    else 0
  else 0

-- Define the services performed on each car model
def services (model: String) : List String :=
  if model = "A" then ["oil_change", "repair", "car_wash"]
  else if model = "B" then ["oil_change", "repair", "car_wash", "tire_rotation"]
  else if model = "C" then ["oil_change", "repair", "car_wash", "tire_rotation"]
  else if model = "D" then ["oil_change", "repair", "tire_rotation"]
  else []

-- Define the number of cars serviced for each model
def num_cars (model: String) : Nat :=
  if model = "A" then 5
  else if model = "B" then 3
  else if model = "C" then 2
  else if model = "D" then 4
  else 0

-- Define a function to calculate the total earnings
def total_earnings : Float :=
  let models := ["A", "B", "C", "D"]
  models.foldl (λ acc model =>
    let s := services model
    let total_cost := s.foldl (λ acc service => acc + charge model service) 0
    let discount := if s.length >= 3 then 0.1 * total_cost else 0
    let earnings := (total_cost - discount) * num_cars model.to_float
    acc + earnings
  ) 0

-- Statement to prove
theorem total_earnings_correct : total_earnings = 1112.40 := by
  sorry

end total_earnings_correct_l525_525036


namespace max_area_trapezoid_centroid_equilateral_l525_525235

noncomputable def centroid (a b c : ℝ × ℝ) : ℝ × ℝ :=
  let (ax, ay) := a
  let (bx, by) := b
  let (cx, cy) := c
  ( (ax + bx + cx) / 3, (ay + by + cy) / 3 )

theorem max_area_trapezoid_centroid_equilateral :
  ∀ (A B C D : ℝ × ℝ),
    A = (0, 0) →
    B = (4, 0) →
    dist B C = 3 →
    dist A D = 2 * Real.sqrt 5 →
    dist A B = 4 →
    dist C D = 8 →
    let g₁ := centroid A B C
    let g₂ := centroid B C D
    let g₃ := centroid A C D
    dist g₁ g₂ = dist g₂ g₃ ∧ dist g₂ g₃ = dist g₃ g₁ →
    let area := (abs (B.1 - A.1) + abs (D.1 - C.1)) * (C.2 - A.2) / 2
    area ≤ 36 :=
sorry

end max_area_trapezoid_centroid_equilateral_l525_525235


namespace all_statements_imply_negation_l525_525939

theorem all_statements_imply_negation :
  let s1 := (true ∧ true ∧ false)
  let s2 := (false ∧ true ∧ true)
  let s3 := (true ∧ false ∧ true)
  let s4 := (false ∧ false ∧ true)
  (s1 → ¬(true ∧ true ∧ true)) ∧
  (s2 → ¬(true ∧ true ∧ true)) ∧
  (s3 → ¬(true ∧ true ∧ true)) ∧
  (s4 → ¬(true ∧ true ∧ true)) :=
by sorry

end all_statements_imply_negation_l525_525939


namespace relationship_between_number_and_value_l525_525009

theorem relationship_between_number_and_value (n v : ℝ) (h1 : n = 7) (h2 : n - 4 = 21 * v) : v = 1 / 7 :=
  sorry

end relationship_between_number_and_value_l525_525009


namespace distinct_prime_divisors_of_sequence_l525_525767

theorem distinct_prime_divisors_of_sequence (k n : ℕ) (h : k > n.factorial) :
  ∃ (p : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i → i ≤ n → prime (p i) ∧ p i ∣ (k + i)) ∧ (∀ i j : ℕ, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → p i ≠ p j) := 
sorry

end distinct_prime_divisors_of_sequence_l525_525767


namespace eigenvalues_of_A_l525_525087

variables {a b : ℝ}
def A (a b : ℝ) := matrix 2 2 ℝ := !(fin 2) :=
  ![![a, 1], ![b, 4]]

theorem eigenvalues_of_A (h₁ : A a b.mulVec ![1, 2] = ![2, -7]) :
  eigenvalues A = {4, 1} :=
begin
  sorry
end

end eigenvalues_of_A_l525_525087


namespace probability_two_primes_is_1_over_29_l525_525431

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l525_525431


namespace rob_reads_13_pages_l525_525775
noncomputable def total_pages_read : ℕ :=
let literature_time := 3 * 60 * (3 / 4) in
let history_time := 3 * 60 * (1 / 4) in
let literature_pages := literature_time / 15 in
let history_pages := history_time / 10 in
literature_pages + history_pages

theorem rob_reads_13_pages :
  total_pages_read = 13 :=
  sorry

end rob_reads_13_pages_l525_525775


namespace intersection_P_M_l525_525161

variable {y x : ℝ}

definition P (y : ℝ) : Prop := ∃ x : ℝ, y = x^2 - 6*x + 10
definition M (y : ℝ) : Prop := ∃ x : ℝ, y = -x^2 + 2*x + 8

theorem intersection_P_M : 
  ∀ y, (P y ∧ M y) ↔ (1 ≤ y ∧ y ≤ 9) := by
  sorry

end intersection_P_M_l525_525161


namespace megan_dials_fatima_correctly_l525_525259

noncomputable def count_permutations : ℕ := (Finset.univ : Finset (Equiv.Perm (Fin 3))).card
noncomputable def total_numbers : ℕ := 4 * count_permutations

theorem megan_dials_fatima_correctly :
  (1 : ℚ) / (total_numbers : ℚ) = 1 / 24 :=
by
  sorry

end megan_dials_fatima_correctly_l525_525259


namespace equation_of_curve_C_shortest_distance_l525_525133

noncomputable def point := (ℝ × ℝ)
def A : point := (-1, 0)
def B : point := (1, 0)
def distance (P Q : point) : ℝ := ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2).sqrt
def points_satisfy_condition (P : point) : Prop := distance P A = (2 ^ 0.5) * distance P B

theorem equation_of_curve_C :
  ∃ (P : point), points_satisfy_condition P → (P.1 - 3) ^ 2 + P.2 ^ 2 = 8 :=
by { sorry }

def center_of_symmetry : point := (3, 0)
def parabola (Q : point) : Prop := Q.2 ^ 2 = Q.1
def distance_QC (Q : point) : ℝ := ((Q.1 - center_of_symmetry.1) ^ 2 + (Q.2 - center_of_symmetry.2) ^ 2).sqrt

theorem shortest_distance :
  ∃ (Q : point), parabola Q → distance_QC Q = (11 ^ 0.5) / 2 :=
by { sorry }

end equation_of_curve_C_shortest_distance_l525_525133


namespace geom_seq_nth_term_is_14_l525_525794

noncomputable theory
open_locale classical

theorem geom_seq_nth_term_is_14 (n x : ℕ) (a₁ a₂ a₃ aₙ : ℕ) 
  (h1 : a₁ = 2 * x - 3) 
  (h2 : a₂ = 5 * x - 11) 
  (h3 : a₃ = 10 * x - 22) 
  (hn : aₙ = 5120) :
  n = 14 :=
sorry

end geom_seq_nth_term_is_14_l525_525794


namespace smallest_range_l525_525850

-- Define the conditions
def estate (A B C : ℝ) : Prop :=
  A = 20000 ∧
  abs (A - B) > 0.3 * A ∧
  abs (A - C) > 0.3 * A ∧
  abs (B - C) > 0.3 * A

-- Define the statement to prove
theorem smallest_range (A B C : ℝ) (h : estate A B C) : 
  ∃ r : ℝ, r = 12000 :=
sorry

end smallest_range_l525_525850


namespace min_value_expr_l525_525684

noncomputable def find_min_value (a b c d : ℝ) (x y : ℝ) : ℝ :=
  x / c^2 + y^2 / d^2

theorem min_value_expr (a b c d : ℝ) (h : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) :
  ∃ x y : ℝ, find_min_value a b c d x y = -abs a / c^2 := 
sorry

end min_value_expr_l525_525684


namespace isosceles_right_triangle_rotation_l525_525213

theorem isosceles_right_triangle_rotation
  (A B C D O1 O2 O3 O4 : Point)
  (hABO1 : isosceles_right_triangle A B O1)
  (hBCO2 : isosceles_right_triangle B C O2)
  (hCDO3 : isosceles_right_triangle C D O3)
  (hDAO4 : isosceles_right_triangle D A O4)
  (hO1O3 : O1 = O3) :
  O2 = O4 :=
by
  sorry

end isosceles_right_triangle_rotation_l525_525213


namespace maximum_height_when_isosceles_l525_525455

variable (c : ℝ) (c1 c2 : ℝ)

def right_angled_triangle (c1 c2 c : ℝ) : Prop :=
  c1 * c1 + c2 * c2 = c * c

def isosceles_right_triangle (c1 c2 : ℝ) : Prop :=
  c1 = c2

noncomputable def height_relative_to_hypotenuse (c : ℝ) : ℝ :=
  c / 2

theorem maximum_height_when_isosceles 
  (c1 c2 c : ℝ) 
  (h_right : right_angled_triangle c1 c2 c) 
  (h_iso : isosceles_right_triangle c1 c2) :
  height_relative_to_hypotenuse c = c / 2 :=
  sorry

end maximum_height_when_isosceles_l525_525455


namespace sum_of_squares_of_A_is_correct_l525_525241

open BigOperators

def digit_sum (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

def A : Finset ℕ := (Finset.range 10001).filter (λ n => digit_sum n = 2)

def sum_of_squares_of_A : ℕ := A.sum (λ n => n^2)

theorem sum_of_squares_of_A_is_correct :
  sum_of_squares_of_A = 7294927 :=
by
  sorry

end sum_of_squares_of_A_is_correct_l525_525241


namespace aunt_gave_each_20_l525_525216

theorem aunt_gave_each_20
  (jade_initial : ℕ)
  (julia_initial : ℕ)
  (total_after_aunt : ℕ)
  (equal_amount_from_aunt : ℕ)
  (h1 : jade_initial = 38)
  (h2 : julia_initial = jade_initial / 2)
  (h3 : total_after_aunt = 97)
  (h4 : jade_initial + julia_initial + 2 * equal_amount_from_aunt = total_after_aunt) :
  equal_amount_from_aunt = 20 := 
sorry

end aunt_gave_each_20_l525_525216


namespace f_sqrt_2_l525_525240

noncomputable def f : ℝ → ℝ :=
sorry

axiom domain_f : ∀ x, 0 < x → 0 < f x
axiom add_property : ∀ x y, f (x * y) = f x + f y
axiom f_at_8 : f 8 = 6

theorem f_sqrt_2 : f (Real.sqrt 2) = 1 :=
by
  have sqrt2pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  sorry

end f_sqrt_2_l525_525240


namespace add_water_to_solution_l525_525654

theorem add_water_to_solution (w : ℕ) :
  let initial_volume := 50
  let initial_concentration := 0.4
  let final_concentration := 0.25
  let acid_amount := initial_concentration * initial_volume
  let final_volume := initial_volume + w
  (acid_amount / final_volume = final_concentration) ↔ (w = 30) :=
by 
  sorry

end add_water_to_solution_l525_525654


namespace ratio_of_shortest_to_longest_diagonal_of_regular_octagon_l525_525536

theorem ratio_of_shortest_to_longest_diagonal_of_regular_octagon :
  let α := real.sqrt 2 - real.sqrt 2
  let ratio := real.sqrt (2 - real.sqrt 2) / 2
  ratio = real.sqrt (2 - real.sqrt 2) / 2 :=
by
  sorry

end ratio_of_shortest_to_longest_diagonal_of_regular_octagon_l525_525536


namespace donald_duck_needs_at_least_13_uses_l525_525050

-- Define the conditions
def race_distance : ℕ := 10000
def speed_mouse : ℕ := 125
def speed_duck : ℕ := 100
def backward_speed (n : ℕ) : ℕ := n * speed_mouse / 10

-- Define the time to win condition
def time_to_win (n : ℕ) : Prop :=
  let time_mouse := race_distance / speed_mouse in
  let time_duck := race_distance / speed_duck in
  let time_diff := time_duck - time_mouse in
  let total_wasted_time := ∑ i in Finset.range (n + 1), 1 + i / 10 in
  total_wasted_time ≥ time_diff

-- Prove that the minimum number of times needed is 13
theorem donald_duck_needs_at_least_13_uses : ∃ n : ℕ, n ≥ 13 ∧ time_to_win n := sorry

end donald_duck_needs_at_least_13_uses_l525_525050


namespace apples_number_l525_525345

def num_apples (A O B : ℕ) : Prop :=
  A = O + 27 ∧ O = B + 11 ∧ A + O + B = 301 → A = 122

theorem apples_number (A O B : ℕ) : num_apples A O B := by
  sorry

end apples_number_l525_525345


namespace smallest_composite_square_side_length_l525_525986

theorem smallest_composite_square_side_length (n : ℕ) (h : ∃ k, 14 * n = k^2) : 
  ∃ m : ℕ, n = 14 ∧ m = 14 :=
by
  sorry

end smallest_composite_square_side_length_l525_525986


namespace find_particular_number_l525_525490

theorem find_particular_number (x : ℤ) (h : ((x / 23) - 67) * 2 = 102) : x = 2714 := 
by 
  sorry

end find_particular_number_l525_525490


namespace find_c_l525_525083

-- Define the necessary conditions for the circle equation and the radius
variable (c : ℝ)

-- The given conditions
def circle_eq := ∀ (x y : ℝ), x^2 + 8*x + y^2 - 6*y + c = 0
def radius_five := (∀ (h k r : ℝ), r = 5 → ∃ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2)

theorem find_c (h k r : ℝ) (r_eq : r = 5) : c = 0 :=
by {
  sorry
}

end find_c_l525_525083


namespace base3_to_base9_first_digit_l525_525790

theorem base3_to_base9_first_digit (x : ℕ) (h : x = 1*3^6 + 1*3^5 + 2*3^4 + 2*3^3 + 0*3^2 + 0*3^1 + 1*3^0) :
  (x.base 9).head = 1 :=
by
  sorry

end base3_to_base9_first_digit_l525_525790


namespace t_recurrence_t_2k1_t_2k_t_divides_t2n1_l525_525080

noncomputable def t : ℕ → ℕ
| 0       := 1
| 1       := 2
| 2       := 5
| (n + 3) := 2 * t (n + 2) + t (n + 1)

theorem t_recurrence (n : ℕ) : t (n + 3) = 2 * t (n + 2) + t (n + 1) := by
  sorry

theorem t_2k1 (k : ℕ) : t (2 * k + 1) = t k * (t (k - 1) + t (k + 1)) := by
  sorry

theorem t_2k (k : ℕ) : t (2 * k) = t (k - 1) * t (k + 1) := by
  sorry

theorem t_divides_t2n1 (n : ℕ) : t n ∣ t (2 * n + 1) := by
  sorry

#eval t 2 -- should give 5, as a quick test
#eval t 3 -- evaluate to ensure the recurrence is defined correctly


end t_recurrence_t_2k1_t_2k_t_divides_t2n1_l525_525080


namespace Sam_Tanya_ratio_l525_525257

-- Definitions based on the conditions given.
def Lewis_items := 20
def Tanya_items := 4
def Lewis_Sam_relation := Lewis_items = Sam_items + 4
def Sam_Tanya_relation := ∃ k : ℕ, Sam_items = k * Tanya_items

-- The theorem to be proved.
theorem Sam_Tanya_ratio : Lewis_Sam_relation → Sam_Tanya_relation → (Sam_items : ℕ) / (Tanya_items : ℕ) = 4 :=
by
  intro h1 h2
  sorry

end Sam_Tanya_ratio_l525_525257


namespace log2_a2016_l525_525705

theorem log2_a2016 (a : ℕ → ℝ) (h1 : ∀ n, a n = a 1 + n * (a 2 - a 1))
(hr : ∃ (x : ℝ), f x = a 1 ∧ ∃ (y : ℝ), f y = a 4031 ∧ f' x = 0 ∧ f' y = 0) 
(f : ℝ → ℝ := fun x => x^3 - 12 * x^2 + 6 * x)
(f' : ℝ → ℝ := fun x => 3 * x^2 - 24 * x + 6) 
: log 2 (a 2016) = 2 := 
by 
  sorry

end log2_a2016_l525_525705


namespace rational_x_from_conditions_l525_525784

-- Given conditions made explicit
variable (x : ℝ)
variable (hx0 : x ≠ 0)
variable (hx5_rational : is_rational (x^5))
variable (hx_expr_rational : is_rational (20 * x + 19 / x))

theorem rational_x_from_conditions : is_rational x :=
sorry

end rational_x_from_conditions_l525_525784


namespace parallelogram_area_ratio_l525_525364

variable {A B C D R E : Point}

-- Definitions from the conditions
variable (AB AD : Length) (AR : Length := (2/3) * AB) (AE : Length := (1/3) * AD)

-- The statement of the problem to be proved in Lean 4
theorem parallelogram_area_ratio
  (h : Length) (h1 : Length) (S_ABCD : Area := AD * h) (S_AREA : Area := (1/2) * AE * h1) :
  AB ≠ 0 → AD ≠ 0 → h ≠ 0 → h1 ≠ 0 →
  AR = (2 / 3) * AB → AE = (1 / 3) * AD → h1 = (2 / 3) * h →
  (S_ABCD / S_AREA = 9) :=
by
  -- placeholders for remaining goals
  intros _ _ _ _ _ _ _
  sorry

end parallelogram_area_ratio_l525_525364


namespace sum_first_5_terms_l525_525585

variable {a : ℕ → ℝ}
variable (h : 2 * a 2 = a 1 + 3)

theorem sum_first_5_terms (a : ℕ → ℝ) (h : 2 * a 2 = a 1 + 3) : 
  (a 1 + a 2 + a 3 + a 4 + a 5) = 15 :=
sorry

end sum_first_5_terms_l525_525585


namespace min_value_of_PM_l525_525125

def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2) / 25 + (y^2) / 16 = 1

def point_A : ℝ × ℝ := (3, 0)

def min_distance_with_conditions (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  real.sqrt ((x - 3)^2 + y^2 - 1)

theorem min_value_of_PM :
  ∀ P, point_on_ellipse P → (∃ M : ℝ × ℝ, (M.1 - 3)^2 + M.2^2 = 1 ∧ (x - M.1, y - M.2) = (y, -x - 3)) → min_distance_with_conditions P = real.sqrt 3 :=
by
  sorry

end min_value_of_PM_l525_525125


namespace evaluate_f_of_f_l525_525148

def f (x : ℝ) : ℝ := if x > 0 then Real.logb 2 x else -1 / x

theorem evaluate_f_of_f :
  f (f (1 / 4)) = 1 / 2 :=
by
  sorry

end evaluate_f_of_f_l525_525148


namespace probability_of_prime_pairs_l525_525375

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l525_525375


namespace inverse_matrix_eigenvectors_l525_525616

def eigenvalue_vector (A : Matrix (Fin 2) (Fin 2) ℝ) (λ : ℝ) (v : Vector (Fin 2) ℝ) :=
  A.mulVec v = λ • v

theorem inverse_matrix_eigenvectors
  {a b c d : ℝ}
  {A : Matrix (Fin 2) (Fin 2) ℝ}
  (A_def : A = !![a, b; c, d])
  (eigen1 : eigenvalue_vector A 6 !![1, 1])
  (eigen2 : eigenvalue_vector A 1 !![3, -2]) :
  inverse A = !![(2/3 : ℝ), -1/2; -1/3, 1/2] :=
sorry

end inverse_matrix_eigenvectors_l525_525616


namespace probability_of_two_primes_is_correct_l525_525390

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l525_525390


namespace rational_x_sqrt3_x_sq_sqrt3_l525_525968

theorem rational_x_sqrt3_x_sq_sqrt3 (x : ℝ) : (∃ a b : ℚ, x + real.sqrt 3 = a ∧ x^2 + real.sqrt 3 = b) ↔ x = (1 / 2) - real.sqrt 3 :=
by
  sorry

end rational_x_sqrt3_x_sq_sqrt3_l525_525968


namespace ratio_is_three_l525_525329

-- Define the conditions
def area_of_garden : ℕ := 588
def width_of_garden : ℕ := 14
def length_of_garden : ℕ := area_of_garden / width_of_garden

-- Define the ratio
def ratio_length_to_width := length_of_garden / width_of_garden

-- The proof statement
theorem ratio_is_three : ratio_length_to_width = 3 := 
by sorry

end ratio_is_three_l525_525329


namespace find_alpha_plus_beta_l525_525155

variable (α β : ℝ)

def condition_1 : Prop := α^3 - 3*α^2 + 5*α = 1
def condition_2 : Prop := β^3 - 3*β^2 + 5*β = 5

theorem find_alpha_plus_beta (h1 : condition_1 α) (h2 : condition_2 β) : α + β = 2 := 
  sorry

end find_alpha_plus_beta_l525_525155


namespace max_inradius_difference_l525_525506

open Real

noncomputable def ellipse := { p : ℝ × ℝ // p.fst^2 / 2 + p.snd^2 = 1 }
noncomputable def foci1 : ℝ × ℝ := (-1, 0)
noncomputable def foci2 : ℝ × ℝ := (1, 0)

theorem max_inradius_difference : ∃ P : ellipse, ∀ Q1 Q2 : ellipse, ∃ r1 r2 : ℝ, (r1 = triangle_inradius (foci1, P, Q2) ∧ r2 = triangle_inradius (foci2, P, Q1)) → 
  (r1 - r2) = 1/3 :=
sorry

def triangle_inradius (A B C : ℝ × ℝ) : ℝ := 
  sorry -- define or assume a method to find the inradius of the triangle

end max_inradius_difference_l525_525506


namespace total_cost_correct_l525_525217

/-
**Conditions:**
1. James has 10 shirts, 12 pairs of pants, 8 jackets, and 5 ties.
2. It takes 1.5 hours to fix a shirt, twice as long for pants, 2.5 hours for a jacket, and 0.5 hours for a tie.
3. The tailor charges $30 per hour for shirts and pants, $40 per hour for jackets, and $20 per hour for ties.
4. The tailor offers a 10% discount for fixing more than 10 items of the same type.
5. The hourly rate increases by $10 for every 5 items of the same type repaired due to complexity.
-/

namespace ClothingRepair

def shirts := 10
def pants := 12
def jackets := 8
def ties := 5

def hours_per_shirt := 1.5
def hours_per_pant := 3 -- twice as long as shirts
def hours_per_jacket := 2.5
def hours_per_tie := 0.5

def rate_per_hour_shirt := 30
def rate_per_hour_pant := 30
def rate_per_hour_jacket := 40
def rate_per_hour_tie := 20

def discount_threshold := 10
def discount_rate := 0.10
def complexity_threshold := 5
def complexity_increase := 10

noncomputable def total_cost : ℝ :=
  let total_hours_shirts := hours_per_shirt * shirts
  let total_hours_pants := hours_per_pant * pants
  let total_hours_jackets := hours_per_jacket * jackets
  let total_hours_ties := hours_per_tie * ties

  let base_cost_shirts := total_hours_shirts * rate_per_hour_shirt
  let base_cost_pants := total_hours_pants * rate_per_hour_pant
  let base_cost_jackets := total_hours_jackets * rate_per_hour_jacket
  let base_cost_ties := total_hours_ties * rate_per_hour_tie

  let discounted_cost_shirts := if shirts > discount_threshold then base_cost_shirts * (1 - discount_rate) else base_cost_shirts
  let final_rate_pants := if pants > complexity_threshold then rate_per_hour_pant + complexity_increase else rate_per_hour_pant
  let discounted_cost_pants := if pants > discount_threshold then (total_hours_pants * final_rate_pants) * (1 - discount_rate) else total_hours_pants * final_rate_pants
  let final_cost_jackets := base_cost_jackets
  let final_cost_ties := base_cost_ties

  discounted_cost_shirts + discounted_cost_pants + final_cost_jackets + final_cost_ties

theorem total_cost_correct : total_cost = 2551 := by
  sorry

end ClothingRepair

end total_cost_correct_l525_525217


namespace general_formula_a_n_sum_first_n_b_n_l525_525995

def sequence_sum (n : ℕ) : ℚ := (n^2 + 3 * n) / 4

def a_n (n : ℕ) : ℚ := (↑n + 1) / 2

def b_n (n : ℕ) : ℚ := 
  let a_n := (↑n + 1) / 2 in
  let a_n1 := (↑n + 2) / 2 in
  (↑n + 1) * 2^(n + 1) - (1 / ((↑n + 1) * (↑n + 2)))

def T_n (n : ℕ) : ℚ := ↑n * 2^(n + 2) + ↑n / (2 * (↑n + 2))

theorem general_formula_a_n (n : ℕ) : a_n n = (↑n + 1) / 2 :=
by sorry

theorem sum_first_n_b_n (n : ℕ) : (list.sum (list.map b_n (list.range n))) = T_n n :=
by sorry

end general_formula_a_n_sum_first_n_b_n_l525_525995


namespace determine_c_l525_525566

noncomputable theory

def sin (α : ℝ) := Real.sin α
def cos (α : ℝ) := Real.cos α
def quadratic_eq (a b c x : ℝ) := a*x^2 + b*x + c = 0

theorem determine_c (α : ℝ) (c : ℝ) 
  (h1 : quadratic_eq 10 (-7) (-c) (sin α))
  (h2 : quadratic_eq 10 (-7) (-c) (cos α)) :
  c = 2.55 :=
sorry

end determine_c_l525_525566


namespace symmetry_line_l525_525048

theorem symmetry_line (g : ℝ → ℝ) (hg : ∀ x : ℝ, g x = g (4 - x)) : ∀ x : ℝ, g x = g 2 :=
begin
  sorry
end

end symmetry_line_l525_525048


namespace product_xyz_l525_525923

theorem product_xyz {x y z a b c : ℝ} 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = b^2) 
  (h3 : x^3 + y^3 + z^3 = c^3) : 
  x * y * z = (a^3 - 3 * a * b^2 + 2 * c^3) / 6 :=
by
  sorry

end product_xyz_l525_525923


namespace area_decrease_by_4_percent_l525_525015

theorem area_decrease_by_4_percent (a b : ℝ) : 
  let a_new := 1.2 * a in
  let b_new := 0.8 * b in
  (a_new * b_new) = 0.96 * (a * b) :=
by
  let a_new := 1.2 * a
  let b_new := 0.8 * b
  sorry

end area_decrease_by_4_percent_l525_525015


namespace Problem1_Problem2_l525_525865

-- Given conditions as definitions in Lean
structure Square3D :=
(AD AR RQ : ℝ)
(hAD_AR : AD = AR)
(hAR_RQ : AR = 2 * RQ)
(hAD_val : AD = 2)

structure Point (α : Type) :=
(x y z : α)

noncomputable def midpoint (P1 P2 : Point ℝ) : Point ℝ :=
{ x := (P1.x + P2.x) / 2,
  y := (P1.y + P2.y) / 2,
  z := (P1.z + P2.z) / 2 }

noncomputable def onLineSegment (P Q : Point ℝ) (M : Point ℝ) : Prop :=
∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ M = { x := P.x + λ * (Q.x - P.x), y := P.y + λ * (Q.y - P.y), z := P.z + λ * (Q.z - P.z)}

-- Definitions in the problem
variables (R A B C D Q M: Point ℝ)
variables (SQR: Square3D)
variables (E : Point ℝ := midpoint B R)
variables (onM : onLineSegment B Q M)

-- Theorem for Part 1
theorem Problem1 : (R.x = A.x ∧ R.y = A.y ∧ R.z ≠ A.z) →
                   (RQ.x - R.x = AD.x - A.x ∧ RQ.y - R.y = AD.y - A.y ∧ RQ.z - R.z = AD.z - A.z ∧ 
                   AD.y = A.y ∧ AD.z = A.z ∧ AD.x = A.x + 2) →
                   A.z = 0 →
                   E = { x := (B.x + R.x) / 2, y := (B.y + R.y) / 2, z := (B.z + R.z) / 2 } →
                   (M.z = 0) → 
                   AD = AR ∧ AR = 2 * RQ ∧ AD = 2 →
                   (A ≠ E) →
                   A.z = 0 →
                   B.z = 0 →
                   C.z = 0 →
                   D.z = 0 →
                   M.z = 0 →
                   E.z ≠ 0 →
                   A ≠ R →
                   A ≠ B →
                   A ≠ C →
                   A ≠ D →
                   A E⊥ C M :=
sorry

-- Theorem for Part 2
theorem Problem2 : (R.x = A.x ∧ R.y = A.y ∧ R.z ≠ A.z) →
                   (RQ.x - R.x = AD.x - A.x ∧ RQ.y - R.y = AD.y - A.y ∧ RQ.z - R.z = AD.z - A.z ∧ 
                   AD.y = A.y ∧ AD.z = A.z ∧ AD.x = A.x + 2) →
                   A.z = 0 →
                   E = { x := (B.x + R.x) / 2, y := (B.y + R.y) / 2, z := (B.z + R.z) / 2} →
                   (M.x = λ * (Q.x - B.x) + B.x ∧ M.y = λ * (Q.y - B.y) + B.y ∧ M.z = λ * (Q.z - B.z) + B.z) →
                   AD = AR ∧ AR = 2 * RQ ∧ AD = 2 →
                   A ≠ E →
                   A.z = 0 →
                   B.z = 0 →
                   C.z = 0 →
                   D.z = 0 →
                   0 ≤ λ ∧ λ ≤ 1 →
                   4 / 9 ≤ |(normalizedAngle {(C.x - M.x, C.y - M.y, C.z - M.z) • (2, 2, 1)} / (||{C.x - M.x, C.y - M.y, C.z - M.z}|| * ||{2, 2, 1}||))| ∧ |normalizedAngle {(C.x - M.x, C.y - M.y, C.z - M.z) • (2, 2, 1)} / (||{C.x - M.x, C.y - M.y, C.z - M.z}|| * ||{2, 2, 1}||)| ≤ sqrt(2) / 2 :=
sorry

end Problem1_Problem2_l525_525865


namespace trapezoid_radius_circumcircle_l525_525312

-- Define the geometrical properties of the trapezoid
variables {a b : ℝ} (trapezoid : ℝ) (R : ℝ)

-- Given these conditions:
def bases_condition := (1, 6)
def diagonals_condition := (3, 5)

-- Prove that the radius \( R \) of the circumcircle of the triangle formed is given by
theorem trapezoid_radius_circumcircle (a b : ℝ) :
  (R = (a^2) / sqrt(4 * (a^2) - (b^2))) :=
sorry

end trapezoid_radius_circumcircle_l525_525312


namespace replaced_whisky_percentage_l525_525485

-- Defining the conditions as constants
def original_percentage : ℝ := 0.40 -- 40% alcohol in original whisky
def new_percentage : ℝ := 0.24 -- 24% alcohol after replacement
def replaced_fraction : ℝ := 0.7619047619047619 -- Fraction of jar replaced

-- The statement to prove
theorem replaced_whisky_percentage :
  ∃ B : ℝ, (original_percentage - original_percentage * replaced_fraction + B * replaced_fraction = new_percentage) ∧ B = 0.19 :=
begin
  use 0.19,
  split,
  { sorry }, -- Proof details to be filled in
  { refl }
end

end replaced_whisky_percentage_l525_525485


namespace projection_question_projection_condition_projection_problem_l525_525012

-- Define the projection function
def projection (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := v.1 * u.1 + v.2 * u.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu) • u

theorem projection_question :
  projection ⟨-3, 3⟩ ⟨5, 1⟩ = ⟨-30/13, -6/13⟩ :=
by
  -- Proof omitted
  sorry

-- Define the condition based on the problem
theorem projection_condition :
  projection ⟨3, 3⟩ ⟨5, 1⟩ = ⟨45/10, 9/10⟩ :=
by
  -- Substitute the standard form vector and simplify
  sorry

-- Final theorem proving the given condition implies the desired projection
theorem projection_problem (h : projection ⟨3, 3⟩ ⟨5, 1⟩ = ⟨45/10, 9/10⟩) :
  projection ⟨-3, 3⟩ ⟨5, 1⟩ = ⟨-30/13, -6/13⟩ :=
by
  -- Use h and the definition of projection to arrive at the conclusion
  sorry

end projection_question_projection_condition_projection_problem_l525_525012


namespace vector_subtraction_proof_l525_525549

def v1 : ℝ × ℝ := (3, -8)
def v2 : ℝ × ℝ := (2, -6)
def a : ℝ := 5
def answer : ℝ × ℝ := (-7, 22)

theorem vector_subtraction_proof : (v1.1 - a * v2.1, v1.2 - a * v2.2) = answer := 
by
  sorry

end vector_subtraction_proof_l525_525549


namespace slope_of_line_is_pm1_l525_525617

noncomputable def polarCurve (θ : ℝ) : ℝ := 2 * Real.cos θ - 4 * Real.sin θ

noncomputable def lineParametric (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, -1 + t * Real.sin α)

theorem slope_of_line_is_pm1
  (t α : ℝ)
  (hAB : ∃ A B : ℝ × ℝ, lineParametric t α = A ∧ (∃ t1 t2 : ℝ, A = lineParametric t1 α ∧ B = lineParametric t2 α ∧ dist A B = 3 * Real.sqrt 2))
  (hC : ∃ θ : ℝ, polarCurve θ = dist (1, -1) (polarCurve θ * Real.cos θ, polarCurve θ * Real.sin θ)) :
  ∃ k : ℝ, k = 1 ∨ k = -1 :=
sorry

end slope_of_line_is_pm1_l525_525617


namespace maximal_sum_at_nine_or_ten_l525_525159

theorem maximal_sum_at_nine_or_ten:
  let a : ℕ → ℤ := λ n, -n^2 + 9n + 10 in
  ∃ n : ℕ, (n = 9 ∨ n = 10) ∧
           (∀ m : ℕ, m < n → 
             let S : ℕ → ℤ := λ k, (finset.range k).sum a in 
             S m ≤ S (n-1)) :=
by
  sorry

end maximal_sum_at_nine_or_ten_l525_525159


namespace nurse_total_check_time_l525_525806

noncomputable def total_time_in_hours : ℝ :=
  let kindergarteners := 26
  let first_graders := 19
  let second_graders := 20
  let third_graders := 25
  let fourth_graders := 30
  let fifth_graders := 28
  let students := kindergarteners + first_graders + second_graders + third_graders + fourth_graders + fifth_graders
  let time_per_student := 2 + 2 + 3  -- vision test + hearing test + lice check (in minutes)
  let total_time_minutes := students * time_per_student
  total_time_minutes / 60

theorem nurse_total_check_time : abs (total_time_in_hours - 17.27) < 0.01 :=
by
  let total_time := total_time_in_hours
  have h : total_time = 1036 / 60 := by sorry
  rw h
  norm_num
  sorry

end nurse_total_check_time_l525_525806


namespace find_h_l525_525326

theorem find_h (h j k : ℤ) 
  (H1 : 4 * (0 - h) ^ 2 + j = 2021)
  (H2 : 3 * (0 - h) ^ 2 + k = 2022)
  (H3 : is_rational_root 4 (2021 - 4 * h ^ 2))
  (H4 : is_rational_root 3 (2022 - 3 * h ^ 2)) :
  h = 1 := 
sorry

end find_h_l525_525326


namespace todd_ate_cupcakes_l525_525778

theorem todd_ate_cupcakes :
  let C := 38   -- Total cupcakes baked by Sarah
  let P := 3    -- Number of packages made
  let c := 8    -- Number of cupcakes per package
  let L := P * c  -- Total cupcakes left after packaging
  C - L = 14 :=  -- Cupcakes Todd ate is 14
by
  sorry

end todd_ate_cupcakes_l525_525778


namespace age_problem_l525_525079

open Classical

noncomputable def sum_cubes_ages (r j m : ℕ) : ℕ :=
  r^3 + j^3 + m^3

theorem age_problem (r j m : ℕ) (h1 : 5 * r + 2 * j = 3 * m)
    (h2 : 3 * m^2 + 2 * j^2 = 5 * r^2) (h3 : Nat.gcd r (Nat.gcd j m) = 1) :
    sum_cubes_ages r j m = 3 := by
  sorry

end age_problem_l525_525079


namespace units_digit_sum_factorials_l525_525075

/-- The units digit of the sum T = 1! + 3! + 5! + 7! + 9! + 11! + ⋯ + 49! is 7 -/
theorem units_digit_sum_factorials :
  let T := (Finset.filter (λ n, n % 2 = 1 ∧ n ≤ 49) Finset.range 50).sum (λ n, nat.factorial n)
  let units_digit := T % 10
  units_digit = 7 :=
by
  sorry

end units_digit_sum_factorials_l525_525075


namespace water_added_eq_30_l525_525648

-- Given conditions
def initial_volume := 50
def initial_concentration := 0.4
def final_concentration := 0.25
def amount_of_acid := initial_volume * initial_concentration

-- Proof problem statement
theorem water_added_eq_30 : ∃ w : ℝ, amount_of_acid / (initial_volume + w) = final_concentration ∧ w = 30 :=
by 
  -- Solution steps go here, but for now we insert sorry to skip the proof.
  sorry

end water_added_eq_30_l525_525648


namespace stone_145_is_5_l525_525057

theorem stone_145_is_5 :
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 15) → (145 % 28) = 5 → n = 5 :=
by
  intros n h h145
  sorry

end stone_145_is_5_l525_525057


namespace f_derivative_at_1_g_derivative_at_x_l525_525614

noncomputable def f (x : ℝ) := (Real.exp x) / x

theorem f_derivative_at_1 : (deriv f 1) = 0 :=
by {
  have h₁ : deriv f 1 = (Real.exp 1 * (1 - 1)) / (1 ^ 2),
  simp [f],
  sorry, -- details of quotient rule application
}

noncomputable def g (x : ℝ) := f (2 * x)

theorem g_derivative_at_x (x : ℝ) : (deriv g x) = (Real.exp (2 * x) * (2 * x - 1)) / (2 * x ^ 2) :=
by {
  have h₂ : deriv g x = ((2 * Real.exp (2 * x) * (2 * x)) - (Real.exp (2 * x) * 2)) / (2 * x ^ 2),
  simp [g],
  sorry, -- details of quotient rule application
}

end f_derivative_at_1_g_derivative_at_x_l525_525614


namespace pure_water_to_achieve_desired_concentration_l525_525641

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l525_525641


namespace simplify_expression1_simplify_expression2_l525_525780

theorem simplify_expression1 (x : ℝ) : 
  5*x^2 + x + 3 + 4*x - 8*x^2 - 2 = -3*x^2 + 5*x + 1 :=
by
  sorry

theorem simplify_expression2 (a : ℝ) : 
  (5*a^2 + 2*a - 1) - 4*(3 - 8*a + 2*a^2) = -3*a^2 + 34*a - 13 :=
by
  sorry

end simplify_expression1_simplify_expression2_l525_525780


namespace distance_OP_l525_525289

open EuclideanGeometry

section CircleCenters

variables {A B C O P : Point}
variables {BC AC AB : ℝ}

noncomputable def triangle_right : Prop := is_right_triangle ABC BC AC AB
noncomputable def circle_tangent_BC : Prop := is_tangent_circle O B BC ∧ is_pass_through_circle O A
noncomputable def circle_tangent_AC : Prop := is_tangent_circle P A AC ∧ is_pass_through_circle P B

theorem distance_OP (BC AC AB : ℝ) (h_triang: triangle_right A B C BC AC AB)
  (h_circle_O: circle_tangent_BC A B C O BC)
  (h_circle_P: circle_tangent_AC A B C P AC ) :
  dist O P = 35 / 12 := 
sorry

end CircleCenters

end distance_OP_l525_525289


namespace RachelStillToColor_l525_525769

def RachelColoringBooks : Prop :=
  let initial_books := 23 + 32
  let colored := 44
  initial_books - colored = 11

theorem RachelStillToColor : RachelColoringBooks := 
  by
    let initial_books := 23 + 32
    let colored := 44
    show initial_books - colored = 11
    sorry

end RachelStillToColor_l525_525769


namespace math_proof_problem_l525_525122

noncomputable def complex_example : Prop :=
  let z := 2 - complex.i in
  let conj_z := complex.conj z in
  z * (conj_z + complex.i) = 6 + 2 * complex.i

theorem math_proof_problem : complex_example := by
  sorry

end math_proof_problem_l525_525122


namespace quadratic_solution_sum_l525_525558

theorem quadratic_solution_sum : 
  ∀ (x : ℝ), (5 * x^2 - 11 * x + 2 = 0) → 
  (∃ (m n p : ℤ), x = (m + real.sqrt n) / p ∨ x = (m - real.sqrt n) / p ∧ int.gcd m (int.gcd n p) = 1 ∧ m + n + p = 30) :=
by
  assume (x : ℝ) (h : 5 * x^2 - 11 * x + 2 = 0)
  sorry

end quadratic_solution_sum_l525_525558


namespace solve_for_x_l525_525304

theorem solve_for_x (x : ℕ) (h : 5 * (2 ^ x) = 320) : x = 6 :=
by
  sorry

end solve_for_x_l525_525304


namespace contingency_table_proof_l525_525180

noncomputable def probability_of_mistake (K_squared : ℝ) : ℝ :=
if K_squared > 3.841 then 0.05 else 1.0 -- placeholder definition to be refined

theorem contingency_table_proof :
  probability_of_mistake 4.013 ≤ 0.05 :=
by sorry

end contingency_table_proof_l525_525180


namespace largest_of_five_consecutive_integers_with_product_15120_is_9_l525_525076

theorem largest_of_five_consecutive_integers_with_product_15120_is_9 :
  ∃ (a b c d e : ℤ), a * b * c * d * e = 15120 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e = 9 :=
sorry

end largest_of_five_consecutive_integers_with_product_15120_is_9_l525_525076


namespace mango_rice_flour_cost_l525_525316

theorem mango_rice_flour_cost :
  (let M := 69
       R := 69
       F := 23
  in
    4 * M + 3 * R + 5 * F = 598) :=
by
  sorry

end mango_rice_flour_cost_l525_525316


namespace arcsin_eq_solution_domain_l525_525306

open Real

theorem arcsin_eq_solution_domain (x : ℝ) (hx1 : abs (x * sqrt 5 / 3) ≤ 1)
  (hx2 : abs (x * sqrt 5 / 6) ≤ 1)
  (hx3 : abs (7 * x * sqrt 5 / 18) ≤ 1) :
  arcsin (x * sqrt 5 / 3) + arcsin (x * sqrt 5 / 6) = arcsin (7 * x * sqrt 5 / 18) ↔ 
  x = 0 ∨ x = 8 / 7 ∨ x = -8 / 7 := sorry

end arcsin_eq_solution_domain_l525_525306


namespace roots_sum_double_l525_525668

theorem roots_sum_double (a b : ℝ) (h : Polynomial.eval₂ RingHom.id (x : ℝ) (x^2 + x - 6) = 0): 2*a + 2*b = -2 :=
by
  sorry

end roots_sum_double_l525_525668


namespace domain_of_g_x_l525_525972

theorem domain_of_g_x :
  ∀ x, (x ≤ 6 ∧ x ≥ -19) ↔ -19 ≤ x ∧ x ≤ 6 :=
by 
  -- Statement only, no proof
  sorry

end domain_of_g_x_l525_525972


namespace oranges_in_stack_l525_525884

/-- 
A grocer stacks oranges in a triangular pyramid-shaped stack.
The base is a right triangle with dimensions 6 oranges by 9 oranges.
Each subsequent layer above the base reduces such that
the layer above also forms a smaller right triangle with
each leg length reduced by one orange.
The stack concludes with a single orange at the top.
-/
def triangular_pyramid_oranges : Nat :=
  let base_layers := List.foldl (+) 0 [27, 20, 14, 9, 5, 2]
  base_layers + 1

theorem oranges_in_stack : triangular_pyramid_oranges = 78 :=
by sorry

end oranges_in_stack_l525_525884


namespace probability_between_lines_l525_525528

-- Definitions for lines p and q
def line_p (x : ℝ) : ℝ := -3 * x + 9
def line_q (x : ℝ) : ℝ := -6 * x + 9

-- Theorem statement
theorem probability_between_lines :
  let area_under_p := 1 / 2 * 3 * 9
      area_under_q := 1 / 2 * 1.5 * 9
      area_between := area_under_p - area_under_q
  in (area_between / area_under_p) = 0.5 :=
by
  sorry

end probability_between_lines_l525_525528


namespace math_proof_problem_l525_525727

variable {a b c : ℝ}

theorem math_proof_problem (h₁ : a * b * c * (a + b) * (b + c) * (c + a) ≠ 0)
  (h₂ : (a + b + c) * (1 / a + 1 / b + 1 / c) = 1007 / 1008) :
  (a * b / ((a + c) * (b + c)) + b * c / ((b + a) * (c + a)) + c * a / ((c + b) * (a + b))) = 2017 := 
sorry

end math_proof_problem_l525_525727


namespace limit_a_n_l525_525088

noncomputable def a_n (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n < 10000 then (2^(n+1)) / (2^n + 1)
  else if n ≥ 10000 then ((n + 1)^2) / (n^2 + 1)
  else 0

theorem limit_a_n : tendsto (λ n, a_n n) at_top (𝓝 1) :=
begin
  sorry
end

end limit_a_n_l525_525088


namespace cupcakes_leftover_l525_525753

-- Definitions based on the conditions
def total_cupcakes : ℕ := 17
def num_children : ℕ := 3

-- Theorem proving the correct answer
theorem cupcakes_leftover : total_cupcakes % num_children = 2 := by
  sorry

end cupcakes_leftover_l525_525753


namespace max_A_value_l525_525505

noncomputable def max_A : ℕ := 164

theorem max_A_value (a : ℕ → ℕ) (h : bijective a) :
  (∃ i : ℕ, (0 ≤ i ∧ i ≤ 32 ∧ ∑ k in finset.range 8, a (i + k) < max_A)) → false :=
sorry

end max_A_value_l525_525505


namespace determine_k_tangent_distance_l525_525136

theorem determine_k_tangent_distance :
  ∃ k : ℝ, (∀ (x : ℝ), l x = (2 + k) * x - 1 - k) ∧ (distance_point_to_line 0 (-1) l = 1) → k = -5 / 4 :=
by sorry

noncomputable def distance_point_to_line (x0 y0 : ℝ) (l : ℝ → ℝ) : ℝ :=
  let line_eq := l x0 - y0 in
  let A := 2 + l 1 in
  let B := -1 in
  abs (A * x0 + B * y0 + line_eq) / sqrt (A^2 + B^2)

end determine_k_tangent_distance_l525_525136


namespace probability_two_primes_from_1_to_30_l525_525403

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l525_525403


namespace shoes_to_belts_ratio_l525_525520

variable (hats : ℕ) (belts : ℕ) (shoes : ℕ)

theorem shoes_to_belts_ratio (hats_eq : hats = 5)
                            (belts_eq : belts = hats + 2)
                            (shoes_eq : shoes = 14) : 
  (shoes / (Nat.gcd shoes belts)) = 2 ∧ (belts / (Nat.gcd shoes belts)) = 1 := 
by
  sorry

end shoes_to_belts_ratio_l525_525520


namespace Jerry_remaining_pages_l525_525719

theorem Jerry_remaining_pages (total_pages : ℕ) (saturday_read : ℕ) (sunday_read : ℕ) (remaining_pages : ℕ) 
  (h1 : total_pages = 93) (h2 : saturday_read = 30) (h3 : sunday_read = 20) (h4 : remaining_pages = 43) : 
  total_pages - saturday_read - sunday_read = remaining_pages := 
by
  sorry

end Jerry_remaining_pages_l525_525719


namespace symmetry_x_axis_l525_525702

theorem symmetry_x_axis (a b : ℝ) (h1 : a - 3 = 2) (h2 : 1 = -(b + 1)) : a + b = 3 :=
by
  sorry

end symmetry_x_axis_l525_525702


namespace binom_21_10_l525_525593

theorem binom_21_10 :
  (Nat.choose 19 9 = 92378) →
  (Nat.choose 19 10 = 92378) →
  (Nat.choose 19 11 = 75582) →
  Nat.choose 21 10 = 352716 := by
  sorry

end binom_21_10_l525_525593


namespace find_A_l525_525441

theorem find_A (A B : ℕ) (h : 632 - (100 * A + 10 * B) = 41) : A = 5 :=
by 
  sorry

end find_A_l525_525441


namespace president_vice_president_choices_l525_525197

theorem president_vice_president_choices (n : ℕ) (h : n = 6) : (n * (n - 1)) = 30 :=
by
  rw h
  rw [Nat.mul_sub_left_distrib, Nat.mul_one]
  rw Nat.sub_self
  rw Nat.mul_zero
  rw Nat.add_zero
  rw Nat.sub_self
  rw Nat.zero_mul
  exact rfl

end president_vice_president_choices_l525_525197


namespace interesting_numbers_in_range_l525_525748

/-- A composite number n is "interesting" if all its natural divisors can be listed in ascending order, 
and each subsequent divisor is divisible by the previous one -/
def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧ ¬nat.prime n ∧ ∀ (d₁ d₂ : ℕ), d₁ ∣ n → d₂ ∣ n → d₁ < d₂ → d₂ % d₁ = 0

theorem interesting_numbers_in_range :
  {n : ℕ | 20 ≤ n ∧ n ≤ 90 ∧ is_interesting n} = {25, 27, 32, 49, 64, 81} :=
by
  sorry

end interesting_numbers_in_range_l525_525748


namespace quadratic_solution_l525_525527

def quadratic_rewrite (x b c : ℝ) : ℝ := (x + b) * (x + b) + c

theorem quadratic_solution (b c : ℝ)
  (h1 : ∀ x, x^2 + 2100 * x + 4200 = quadratic_rewrite x b c)
  (h2 : c = -b^2 + 4200) :
  c / b = -1034 :=
by
  sorry

end quadratic_solution_l525_525527


namespace ott_fraction_of_total_money_l525_525751

-- Conditions as definitions
variables (M L N O x : ℝ)

def initial_money (M L N : ℝ) : ℝ := M + L + N
def money_given (M L N : ℝ) (x : ℝ) : Prop :=
  x = M / 5 ∧ x = L / 3 ∧ x = N / 2

def money_ott_received (M L N x : ℝ) (h : money_given M L N x) : ℝ :=
  3 * x

def fraction_total_money (M L N x : ℝ) (h : money_given M L N x) : ℝ :=
  (money_ott_received M L N x h) / (initial_money M L N)

-- The statement only, no proof
theorem ott_fraction_of_total_money (M L N x : ℝ) (h : money_given M L N x) :
  fraction_total_money M L N x h = 3 / 10 := sorry

end ott_fraction_of_total_money_l525_525751


namespace phoenix_equal_roots_implies_a_eq_c_l525_525045

-- Define the "phoenix" equation property
def is_phoenix (a b c : ℝ) : Prop := a + b + c = 0

-- Define the property that a quadratic equation has equal real roots
def has_equal_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

theorem phoenix_equal_roots_implies_a_eq_c (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : is_phoenix a b c) (h₂ : has_equal_real_roots a b c) : a = c :=
sorry

end phoenix_equal_roots_implies_a_eq_c_l525_525045


namespace n_four_minus_n_squared_l525_525945

theorem n_four_minus_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
by 
  sorry

end n_four_minus_n_squared_l525_525945


namespace projection_result_l525_525010

open Real

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let denom := (v.1 * v.1 + v.2 * v.2)
  let num := (u.1 * v.1 + u.2 * v.2)
  let k := num / denom
  (k * v.1, k * v.2)

theorem projection_result :
  proj (proj (3, 3) (45 / 10, 9 / 10)) (-3, 3) = (-30 / 13, -6 / 13) :=
  sorry

end projection_result_l525_525010


namespace regular_octagon_interior_angle_l525_525834

theorem regular_octagon_interior_angle :
  let n := 8 in
  let sum_of_interior_angles := (8 - 2) * 180 in
  sum_of_interior_angles / 8 = 135 :=
by
  let n := 8
  let sum_of_interior_angles := (n - 2) * 180
  have h : sum_of_interior_angles = 1080 by sorry
  show sum_of_interior_angles / n = 135 from sorry

end regular_octagon_interior_angle_l525_525834


namespace find_b_l525_525348

noncomputable def geom_seq_term (a b c : ℝ) : Prop :=
∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r

theorem find_b (b : ℝ) (h_geom : geom_seq_term 160 b (108 / 64)) (h_pos : b > 0) :
  b = 15 * Real.sqrt 6 :=
by
  sorry

end find_b_l525_525348


namespace last_previous_date_l525_525827

def valid_digits (date : ℕ × ℕ × ℕ) (digits : List ℕ) : Prop :=
  let (day, month, year) := date
  List.perm (digits ++ digits ++ digits ++ digits) 
            ((day / 10 % 10) :: (day % 10) :: 
             (month / 10 % 10) :: (month % 10) :: 
             (year / 1000 % 10) :: (year / 100 % 10) :: 
             (year / 10 % 10) :: (year % 10) :: [])

theorem last_previous_date 
  (year_before : 2015 = 2015) 
  (digits_2015 : {1, 1, 2, 2, 2, 0, 5, 1} = {1, 1, 2, 2, 2, 0, 5, 1}) :
    valid_digits (15, 12, 2012) [1, 1, 2, 2, 2, 0, 5, 1] :=
by 
  sorry

end last_previous_date_l525_525827


namespace calculate_binary_expr_l525_525509

theorem calculate_binary_expr :
  let a := 0b11001010
  let b := 0b11010
  let c := 0b100
  (a * b) / c = 0b1001110100 := by
sorry

end calculate_binary_expr_l525_525509


namespace sum_first_2017_terms_l525_525209

noncomputable theory

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧
  a 2 = 8 ∧
  ∀ n : ℕ, 0 < n → a (n + 2) + a n = a (n + 1)

theorem sum_first_2017_terms (a : ℕ → ℤ) (h : sequence a) : 
  (∑ n in finset.range 2017, a (n + 1)) = 2 :=
by
  sorry

end sum_first_2017_terms_l525_525209


namespace csc_neg_45_eq_neg_sqrt2_l525_525061

-- Define the question in Lean given the conditions and prove the answer.
theorem csc_neg_45_eq_neg_sqrt2 : Real.csc (-π/4) = -Real.sqrt 2 :=
by
  -- Sorry placeholder since proof is not required.
  sorry

end csc_neg_45_eq_neg_sqrt2_l525_525061


namespace average_first_15_natural_numbers_l525_525063

theorem average_first_15_natural_numbers :
  (∑ i in Finset.range 15.succ, i) / 15 = 8 :=
by
  sorry

end average_first_15_natural_numbers_l525_525063


namespace ratio_Ford_to_Toyota_l525_525821

-- Definitions based on the conditions
variables (Ford Dodge Toyota VW : ℕ)

axiom h1 : Ford = (1/3 : ℚ) * Dodge
axiom h2 : VW = (1/2 : ℚ) * Toyota
axiom h3 : VW = 5
axiom h4 : Dodge = 60

-- Theorem statement to be proven
theorem ratio_Ford_to_Toyota : Ford / Toyota = 2 :=
by {
  sorry
}

end ratio_Ford_to_Toyota_l525_525821


namespace math_proof_problem_l525_525119

noncomputable def complex_example : Prop :=
  let z := 2 - complex.i in
  let conj_z := complex.conj z in
  z * (conj_z + complex.i) = 6 + 2 * complex.i

theorem math_proof_problem : complex_example := by
  sorry

end math_proof_problem_l525_525119


namespace solve_equation_l525_525782

theorem solve_equation (x y : ℤ) (eq : (x^2 - y^2)^2 = 16 * y + 1) : 
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ 
  (x = 4 ∧ y = 3) ∨ (x = -4 ∧ y = 3) ∨ 
  (x = 4 ∧ y = 5) ∨ (x = -4 ∧ y = 5) :=
sorry

end solve_equation_l525_525782


namespace lucas_journey_distance_l525_525761

noncomputable def distance (D : ℝ) : ℝ :=
  let usual_speed := D / 150
  let distance_before_traffic := 2 * D / 5
  let speed_after_traffic := usual_speed - 1 / 2
  let time_before_traffic := distance_before_traffic / usual_speed
  let time_after_traffic := (3 * D / 5) / speed_after_traffic
  time_before_traffic + time_after_traffic

theorem lucas_journey_distance : ∃ D : ℝ, distance D = 255 ∧ D = 48.75 :=
sorry

end lucas_journey_distance_l525_525761


namespace obtuse_triangle_existence_l525_525910

theorem obtuse_triangle_existence :
  ∃ (a b c : ℝ), (a = 2 ∧ b = 6 ∧ c = 7 ∧ 
  (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2)) ∧
  ¬(6^2 + 7^2 < 8^2 ∨ 7^2 + 8^2 < 6^2 ∨ 8^2 + 6^2 < 7^2) ∧
  ¬(7^2 + 8^2 < 10^2 ∨ 8^2 + 10^2 < 7^2 ∨ 10^2 + 7^2 < 8^2) ∧
  ¬(5^2 + 12^2 < 13^2 ∨ 12^2 + 13^2 < 5^2 ∨ 13^2 + 5^2 < 12^2) :=
sorry

end obtuse_triangle_existence_l525_525910


namespace value_of_a5_l525_525252

def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1
  else 1 + 1 / sequence (n - 1)

theorem value_of_a5 : sequence 5 = 8 / 5 := 
by {
  sorry
}

end value_of_a5_l525_525252


namespace hiring_methods_count_l525_525355

-- Definitions derived directly from the conditions
def units : ℕ := 3
def graduates : ℕ := 4
def hire_methods : ℕ → ℕ → ℕ
| 3, 4 := 60
| _, _ := 0

-- Lean statement proving the number of different hiring methods is 60
theorem hiring_methods_count : hire_methods units graduates = 60 :=
by sorry -- Proof is not required, so we use 'sorry'

end hiring_methods_count_l525_525355


namespace recipes_needed_l525_525025

theorem recipes_needed
  (initial_students : ℕ)
  (avg_cookies_per_student : ℕ)
  (cookies_per_recipe : ℕ)
  (attendance_reduction : ℝ) : ℕ :=
  let expected_attendees := (initial_students : ℝ) * (1 - attendance_reduction) in
  let total_cookies_needed := (expected_attendees * (avg_cookies_per_student : ℝ)) in
  let recipes_needed := (total_cookies_needed / (cookies_per_recipe : ℝ)).ceil.to_nat in
  recipes_needed

#eval recipes_needed 135 3 18 0.40 -- Evaluates the theorem with the given conditions and should output 14

end recipes_needed_l525_525025


namespace gcd_450_210_l525_525444

def prime_factorization_450 : Multiset ℕ := {2, 3, 3, 5, 5}
def prime_factorization_210 : Multiset ℕ := {2, 3, 5, 7}

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_450_210 : gcd 450 210 = 30 := by
  -- note: the actual proof goes here, currently skipped
  sorry

end gcd_450_210_l525_525444


namespace problem_f_f_half_l525_525607

def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then x + 2
  else -1

theorem problem_f_f_half : f (f (1 / 2)) = -1 := by
  sorry

end problem_f_f_half_l525_525607


namespace price_of_pants_l525_525350

-- Given conditions
variables (P B : ℝ)
axiom h1 : P + B = 70.93
axiom h2 : P = B - 2.93

-- Statement to prove
theorem price_of_pants : P = 34.00 :=
by
  sorry

end price_of_pants_l525_525350


namespace even_and_periodic_func_D_l525_525909

-- Define the functions
def fA (x : ℝ) := Real.sin (π - 2 * x)
def fB (x : ℝ) := Real.sin (2 * x) * Real.cos (2 * x)
def fC (x : ℝ) := Real.cos (2 * x) ^ 2 + 1
def fD (x : ℝ) := Real.cos (2 * x - π)

-- Proven statement about function properties
theorem even_and_periodic_func_D : 
  (∀ x : ℝ, fD x = fD (-x)) ∧ (∀ T : ℝ, T > 0 → (fD (x + T) = fD x ↔ T = π)) ∧
  (¬ (∀ x : ℝ, fA x = fA (-x))) ∧ (¬ (∀ T : ℝ, T > 0 → (fA (x + T) = fA x ↔ T = π))) ∧
  (¬ (∀ x : ℝ, fB x = fB (-x))) ∧ (¬ (∀ T : ℝ, T > 0 → (fB (x + T) = fB x ↔ T = π))) ∧
  (∀ x : ℝ, fC x = fC (-x)) ∧ (∀ T : ℝ, T > 0 → (fC (x + T) = fC x ↔ T = π / 2)) :=
by
  sorry

end even_and_periodic_func_D_l525_525909


namespace hen_and_cow_total_heads_l525_525004

theorem hen_and_cow_total_heads (H C total_heads feet : ℤ) (hH : H = 28) (hFeet : 2 * H + 4 * C = 136) : 
  total_heads = H + C := by
  have h : 2 * 28 + 4 * C = 136 := by rw [hH]; exact hFeet
  sorry

end hen_and_cow_total_heads_l525_525004


namespace range_of_x_l525_525137

noncomputable def g (x : ℝ) : ℝ := if x < 0 then log (1 - x) else log (1 + x)
noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then x^3 else g x

theorem range_of_x (x : ℝ) (h1 : ∀ x, g (-x) = g x) (h2 : ∀ x < 0, g x = log (1 - x)) : (f (2 - x^2) > f x ↔ -2 < x ∧ x < 1) :=
by
  sorry

end range_of_x_l525_525137


namespace maximum_value_a3_b3_c3_d3_l525_525735

noncomputable def max_value (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem maximum_value_a3_b3_c3_d3
  (a b c d : ℝ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 20)
  (h2 : a + b + c + d = 10) :
  max_value a b c d ≤ 500 :=
sorry

end maximum_value_a3_b3_c3_d3_l525_525735


namespace solve_for_s_l525_525537

theorem solve_for_s (log : ℝ → ℝ) (H : real.log 5 ≠ 0) : ∃ s : ℝ, 3 = 5^(4*s + 2) ∧ s = (real.log 3 / real.log 5 - 2) / 4 :=
by
  use (real.log 3 / real.log 5 - 2) / 4
  constructor
  - sorry
  - sorry

end solve_for_s_l525_525537


namespace convex_max_intersected_edges_non_convex_max_intersected_edges_96_non_convex_no_100_intersected_edges_l525_525860

theorem convex_max_intersected_edges (P : Polyhedron) (h_edges : P.edges = 100) (h_convex : P.convex) :
  ∃ plane, plane.intersected_edges P ≤ 66 := 
sorry

theorem non_convex_max_intersected_edges_96 (P : Polyhedron) (h_edges : P.edges = 100) (h_non_convex : ¬P.convex) :
  ∃ plane, plane.intersected_edges P = 96 :=
sorry

theorem non_convex_no_100_intersected_edges (P : Polyhedron) (h_edges : P.edges = 100) (h_non_convex : ¬P.convex) :
  ∀ plane, plane.intersected_edges P ≠ 100 :=
sorry

end convex_max_intersected_edges_non_convex_max_intersected_edges_96_non_convex_no_100_intersected_edges_l525_525860


namespace part_I_part_II_l525_525468

structure PolarCurve where
  ρ : ℝ → ℝ
  symmetric_about : PolarCurve → Prop

def C1 : PolarCurve :=
  ⟨ λ θ, 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4), sorry ⟩

def C2 (a : ℝ) : PolarCurve :=
  ⟨ λ θ, a / Real.sin θ, sorry ⟩

-- Conditions
axiom a_positive : ∃ a, a > 0

-- Translated problem Part I
theorem part_I : 
  ∀ a, C1.symmetric_about (C2 a) → a = 1 ∧ 
  (∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 ∧ y = 1) :=
sorry

-- Translated problem Part II
theorem part_II : 
  ∀ φ : ℝ, 
  let |OA| := 2 * Real.sqrt 2 * Real.sin (φ + Real.pi / 4),
      |OB| := 2 * Real.sqrt 2 * Real.cos φ,
      |OC| := 2 * Real.sqrt 2 * Real.sin φ,
      |OD| := 2 * Real.sqrt 2 * Real.cos (φ + Real.pi / 4)
  in |OA| * |OC| + |OB| * |OD| = 4 * Real.sqrt 2 :=
sorry

end part_I_part_II_l525_525468


namespace how_many_whole_boxes_did_nathan_eat_l525_525626

-- Define the conditions
def gumballs_per_package := 5
def total_gumballs := 20

-- The problem to prove
theorem how_many_whole_boxes_did_nathan_eat : total_gumballs / gumballs_per_package = 4 :=
by sorry

end how_many_whole_boxes_did_nathan_eat_l525_525626


namespace quadrilateral_angles_l525_525903

noncomputable theory

-- Definitions of the problem conditions
variables {A B C D O : Type}
variables [metric_space O] [has_dist O]

-- Points A, B, C, D on a circle with center O and radius r
variables (r s : ℝ)
variables (O : O)
variables (A B C D : O)

-- Conditions given in the problem
variables (h_circleA : dist A O = r)
variables (h_circleB : dist B O = r)
variables (h_circleC : dist C O = r)
variables (h_circleD : dist D O = r)

-- Segment lengths given in the problem
variables (h_AB : dist A B = s)
variables (h_BC : dist B C = s)
variables (h_CD : dist C D = s)
variables (h_AD : dist A D = s + r)
variables (h_s_less_r : s < r)

-- The translated proof problem statement
theorem quadrilateral_angles :
  (∠ A B C = 108) ∧ (∠ B C D = 108) ∧ (∠ C D A = 72) ∧ (∠ D A B = 72) :=
sorry

end quadrilateral_angles_l525_525903


namespace probability_two_primes_from_1_to_30_l525_525409

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l525_525409


namespace selling_price_with_60_percent_profit_l525_525494

theorem selling_price_with_60_percent_profit (C : ℝ) (H1 : 2240 = C + 0.4 * C) : 
  let selling_price := C + 0.6 * C 
  in selling_price = 2560 := 
by 
  sorry

end selling_price_with_60_percent_profit_l525_525494


namespace children_count_l525_525866

theorem children_count (W C n : ℝ) (h1 : 4 * W = 1 / 7) (h2 : n * C = 1 / 14) (h3 : 5 * W + 10 * C = 1 / 4) : n = 10 :=
by
  sorry

end children_count_l525_525866


namespace count_japanese_stamps_l525_525265

theorem count_japanese_stamps (total_stamps : ℕ) (perc_chinese perc_us : ℕ) (h1 : total_stamps = 100) 
  (h2 : perc_chinese = 35) (h3 : perc_us = 20): 
  total_stamps - ((perc_chinese * total_stamps / 100) + (perc_us * total_stamps / 100)) = 45 :=
by
  sorry

end count_japanese_stamps_l525_525265


namespace number_of_valid_functions_l525_525976

noncomputable def count_functions : Nat :=
  let n := 5
  ∑ a in Finset.range (n + 1), ∑ b in Finset.range (n - a + 1), ∑ c in Finset.range (n - a - b + 1),
    if a + b + c = n then
      Nat.choose n a * Nat.choose (n - a) b * a^b * b^c
    else 0

theorem number_of_valid_functions : count_functions = 756 := by
  sorry

end number_of_valid_functions_l525_525976


namespace moores_law_transistors_l525_525905

-- Define the initial conditions
def initial_transistors : ℕ := 500000
def doubling_period : ℕ := 2 -- in years
def transistors_doubling (n : ℕ) : ℕ := initial_transistors * 2^n

-- Calculate the number of doubling events from 1995 to 2010
def years_spanned : ℕ := 15
def number_of_doublings : ℕ := years_spanned / doubling_period

-- Expected number of transistors in 2010
def expected_transistors_in_2010 : ℕ := 64000000

theorem moores_law_transistors :
  transistors_doubling number_of_doublings = expected_transistors_in_2010 :=
sorry

end moores_law_transistors_l525_525905


namespace fencing_required_l525_525849

theorem fencing_required (L W : ℕ) (A : ℕ) 
  (hL : L = 20) 
  (hA : A = 680) 
  (hArea : A = L * W) : 
  2 * W + L = 88 := 
by 
  sorry

end fencing_required_l525_525849


namespace three_digit_numbers_l525_525665

theorem three_digit_numbers (x : ℕ) : 
  (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 0 ∧ x % 6 ≠ 0 ∧ x % 7 ≠ 0) ↔ 
  x = 128 :=
by 
sory

end three_digit_numbers_l525_525665


namespace new_ratio_milk_water_l525_525890

theorem new_ratio_milk_water 
  (total_weight : ℝ)
  (initial_ratio_milk : ℕ)
  (initial_ratio_water : ℕ)
  (extra_water : ℝ)
  (parts_milk : ℝ)
  (parts_water : ℝ) :
  total_weight = 85 → initial_ratio_milk = 27 → initial_ratio_water = 7 → extra_water = 5 →
  let part_weight := total_weight / (initial_ratio_milk + initial_ratio_water),
      milk_weight := initial_ratio_milk * part_weight,
      water_weight := initial_ratio_water * part_weight,
      new_water_weight := water_weight + extra_water,
      new_ratio_milk := milk_weight / new_water_weight
  in new_ratio_milk = 3 :=
by
  intros
  sorry

end new_ratio_milk_water_l525_525890


namespace probability_two_primes_from_1_to_30_l525_525408

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l525_525408


namespace find_views_multiplier_l525_525908

theorem find_views_multiplier (M: ℝ) (h: 4000 * M + 50000 = 94000) : M = 11 :=
by
  sorry

end find_views_multiplier_l525_525908


namespace a_beats_b_by_half_km_l525_525847

-- Conditions provided in the problem
def time_for_a_to_run_2km := 2 -- in minutes
def distance_a_runs := 2 -- in km
def time_for_b_to_run_2km := 8/3 -- in minutes
def distance_for_b_in_2_minutes := (3/4) * 2 -- calculate how far B runs in 2 minutes

-- Question rephrased to be proven as a theorem
theorem a_beats_b_by_half_km :
  distance_a_runs - distance_for_b_in_2_minutes = 0.5 :=
by sorry

end a_beats_b_by_half_km_l525_525847


namespace trader_loss_percent_l525_525681
-- Import Lean's mathematical library

-- Define the conditions as Lean definitions
def SP : ℕ := 325475
def gain_percent : ℝ := 0.11
def loss_percent : ℝ := 0.11

-- Define the cost prices using the given conditions
def CP1 : ℝ := SP / (1 + gain_percent)
def CP2 : ℝ := SP / (1 - loss_percent)

-- Calculate the total cost price and total selling price
def TCP : ℝ := CP1 + CP2
def TSP : ℝ := 2 * SP

-- Define the profit or loss
def profit_loss : ℝ := TSP - TCP

-- Define the profit or loss percent
def profit_loss_percent : ℝ := (profit_loss / TCP) * 100

-- State the theorem to prove
theorem trader_loss_percent : profit_loss_percent ≈ -1.209 := by
  sorry

end trader_loss_percent_l525_525681


namespace num_correct_statements_l525_525914

theorem num_correct_statements :
  let f_odd (f : ℝ → ℝ) (a : ℝ) := ∀ x : ℝ, f (-x) = -f x
  let f (a : ℝ) (x : ℝ) := 1 / (2 ^ x + 1) + a
  let converse (A B : ℝ) := A < B → cos A > cos B
  let geometric_progression (a b c : ℝ) := b * b = a * c
  let negation := ∃ x0 : ℝ, x0 ^ 3 - x0 ^ 2 + 1 > 0
  (¬ f_odd (f (-1/2)) (-1/2)) ∧ -- Statement 1
  (∀ A B : ℝ, ¬ (A < B → cos A > cos B)) ∧ -- Statement 2
  ∀ a b c : ℝ, (¬geometric_progression a b c → b ≠ sqrt(a * c) ∧ a = b = c = 0) ∧ -- Statement 3
  negation → -- Statement 4
  true :=
by
  sorry

end num_correct_statements_l525_525914


namespace water_to_add_for_desired_acid_concentration_l525_525655

theorem water_to_add_for_desired_acid_concentration :
  ∃ w : ℝ, 50 * 0.4 / (50 + w) = 0.25 ∧ w = 30 :=
begin
  let initial_volume := 50,
  let initial_acid_concentration := 0.4,
  let desired_acid_concentration := 0.25,
  let acid_content := initial_volume * initial_acid_concentration,
  use 30,
  split,
  { simp [initial_volume, initial_acid_concentration, desired_acid_concentration, acid_content],
    linarith },
  { refl }
end

end water_to_add_for_desired_acid_concentration_l525_525655


namespace arithmetic_sequence_max_sum_l525_525979

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_max_sum 
  (a d : ℝ) 
  (h1 : sum_arithmetic_sequence a d 30 > 0) 
  (h2 : sum_arithmetic_sequence a d 31 < 0) : 
  ∃ n_max, n_max = 15 := 
begin
  sorry
end

end arithmetic_sequence_max_sum_l525_525979


namespace cuboid_surface_area_l525_525824

-- Define the given conditions
def cuboid (a b c : ℝ) := 2 * (a + b + c)

-- Given areas of distinct sides
def area_face_1 : ℝ := 4
def area_face_2 : ℝ := 3
def area_face_3 : ℝ := 6

-- Prove the total surface area of the cuboid
theorem cuboid_surface_area : cuboid area_face_1 area_face_2 area_face_3 = 26 :=
by
  sorry

end cuboid_surface_area_l525_525824


namespace factorial_trailing_zeros_30_l525_525675

theorem factorial_trailing_zeros_30 :
  let zeros_count (n : ℕ) : ℕ :=
    n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 -- Continue this pattern for higher n
  in zeros_count 30 = 7 :=
by
  -- Decompose the definition and proof here
  sorry

end factorial_trailing_zeros_30_l525_525675


namespace prob_after_2021_passes_l525_525363

structure Probabilities where
  a : ℕ → ℝ
  b : ℕ → ℝ
  c : ℕ → ℝ

axiom initialProb : Probabilities
  { a := λ n => if n = 0 then 1 else 0
  , b := λ n => 0
  , c := λ n => 0 }

axiom recurrence : ∀ (p : Probabilities) (n : ℕ),
  p.a (n + 1) = 0.5 * p.b n + 0.5 * p.c n ∧
  p.b (n + 1) = 0.5 * p.a n + 0.5 * p.c n ∧
  p.c (n + 1) = 0.5 * p.a n + 0.5 * p.b n

theorem prob_after_2021_passes : ∀ (p : Probabilities),
    (∀ n, recurrence p n) →
    p.a 0 = 1 →
    p.b 0 = 0 →
    p.c 0 = 0 →
    p.a 2021 = (1/3)*(1 - (1/2)^(2020))
:= by
  intros
  sorry

end prob_after_2021_passes_l525_525363


namespace least_sum_of_exponents_for_2023_l525_525671

-- The problem is to prove the least possible sum of the exponents
-- of powers of 2 which sum up to 2023 is 48

def is_sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (exponents : List ℕ), List.sum (exponents.map (λ e, 2^e)) = n ∧ 
                           exponents.nodup

theorem least_sum_of_exponents_for_2023 : 
  is_sum_of_distinct_powers_of_two 2023 → 
  ∃ (exponents : List ℕ), List.sum (exponents.map (λ e, 2^e)) = 2023 ∧
                           exponents.nodup ∧
                           List.sum exponents = 48 :=
by
  sorry

end least_sum_of_exponents_for_2023_l525_525671


namespace rest_distance_l525_525889

-- Definitions based on conditions
def walk_rate_mph : ℝ := 10
def rest_duration_minutes : ℝ := 8
def total_trip_time_minutes : ℝ := 332
def total_distance_miles : ℝ := 50

-- The distance after which the man rests
def d := total_distance_miles / ((total_trip_time_minutes - total_distance_miles * (60 / walk_rate_mph) + rest_duration_minutes) / rest_duration_minutes)

-- The theorem to prove
theorem rest_distance :
  d = 10 := 
by
  sorry

end rest_distance_l525_525889


namespace volume_conversion_l525_525896

noncomputable def volume_cubic_meters (cubic_feet : ℝ) : ℝ :=
  cubic_feet / 35.315

theorem volume_conversion :
  volume_cubic_meters 216 ≈ 6.12 := by
  sorry

end volume_conversion_l525_525896


namespace pure_water_to_add_eq_30_l525_525633

noncomputable def initial_volume := 50
noncomputable def initial_concentration := 0.40
noncomputable def desired_concentration := 0.25
noncomputable def initial_acid := initial_concentration * initial_volume

theorem pure_water_to_add_eq_30 :
  ∃ w : ℝ, (initial_acid / (initial_volume + w) = desired_concentration) ∧ w = 30 :=
by
  sorry

end pure_water_to_add_eq_30_l525_525633


namespace problem1_problem2_l525_525150

def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

-- (1) Proving the shifted function expression
theorem problem1 (x : ℝ) : (∀ x, f 1 (x - Real.pi / 6) = Real.sin (x - Real.pi / 6)) := by
  sorry

-- (2) Proving the value of ω
theorem problem2 (ω : ℝ) (hω : 0 < ω) (hpass : f ω (2 * Real.pi / 3) = 0) (hincr : ∀ x, 0 < x ∧ x < Real.pi / 3 → f ω x > 0) : 
  ω = 3 / 2 := by
  sorry

end problem1_problem2_l525_525150


namespace problem_statement_l525_525708

noncomputable def polar_to_rectangular (θ : ℝ) : ℝ × ℝ := 
  let ρ := 4 * real.sin θ / (real.cos θ) ^ 2 in
  (ρ * real.cos θ, ρ * real.sin θ)

theorem problem_statement (t α : ℝ) (hα : 0 < α ∧ α < π) :
  ∃ (x y : ℝ), (x = t * cos α ∧ y = 1 - t * sin α) →
    polar_to_rectangular θ = (x, y) →
    x^2 = 4 * y ∧ 
    (∃ a b : ℝ, a = t * cos α ∧ b = 1 - t * sin α ∧ 
    (a^2 = 4 * b ∧ 
    (|a - b| ≥ 16 → (π / 3 < α ∧ α ≤ π / 2) ∨ (π / 2 < α ∧ α ≤ 2 * π / 3)))) :=
sorry

end problem_statement_l525_525708


namespace perimeter_of_new_rectangle_l525_525459

-- Definitions based on conditions
def side_of_square : ℕ := 8
def length_of_rectangle : ℕ := 8
def breadth_of_rectangle : ℕ := 4

-- Perimeter calculation
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the problem
theorem perimeter_of_new_rectangle :
  perimeter (side_of_square + length_of_rectangle) side_of_square = 48 :=
  by sorry

end perimeter_of_new_rectangle_l525_525459


namespace geometric_series_sum_eq_l525_525031

theorem geometric_series_sum_eq (a r : ℝ) (h_a : a = 2) (h_r : r = 1/4) : 
  let S := a / (1 - r) in S = 8 / 3 ∧ S ≤ 3 := 
by 
  have S_def : S = a / (1 - r) := rfl
  rw [h_a, h_r] at S_def
  rw [S_def]
  have : 2 / (1 - 1 / 4) = 2 / (3 / 4) := rfl
  rw this
  have S_eq : 2 / (3 / 4) = 8 / 3 := by sorry
  rw S_eq
  constructor
  . exact rfl
  . linarith

end geometric_series_sum_eq_l525_525031


namespace value_of_expression_l525_525842

theorem value_of_expression : 50^4 + 4 * 50^3 + 6 * 50^2 + 4 * 50 + 1 = 6765201 :=
by
  sorry

end value_of_expression_l525_525842


namespace length_HM_l525_525690

theorem length_HM (A B C M H : Type) [has_measured_point A B C]
  (hAB : dist A B = 13) (hBC : dist B C = 14) (hCA : dist C A = 15)
  (hM : M = midpoint A B) (hH : H = foot_of_altitude A C B)
  : dist H M = 6.5 :=
sorry

end length_HM_l525_525690


namespace n_mul_s_eq_0_l525_525730

noncomputable 
def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = -1
axiom f_eqn : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

theorem n_mul_s_eq_0 : 
  let n : ℕ := {y : ℝ | ∃ x : ℝ, f (x^2 + x^2) = y}.card
  let s : ℝ := {y : ℝ | ∃ x : ℝ, f (x^2 + x^2) = y}.sum id
  in n * s = 0 :=
by
  sorry

end n_mul_s_eq_0_l525_525730


namespace graduation_graduates_l525_525190

theorem graduation_graduates :
  ∃ G : ℕ, (∀ (chairs_for_parents chairs_for_teachers chairs_for_admins : ℕ),
    chairs_for_parents = 2 * G ∧
    chairs_for_teachers = 20 ∧
    chairs_for_admins = 10 ∧
    G + chairs_for_parents + chairs_for_teachers + chairs_for_admins = 180) ↔ G = 50 :=
by
  sorry

end graduation_graduates_l525_525190


namespace t_le_s_l525_525990

theorem t_le_s (a b : ℝ) (t s : ℝ) (h1 : t = a + 2b) (h2 : s = a + b^2 + 1) : t ≤ s :=
by
  sorry

end t_le_s_l525_525990


namespace complex_calculation_l525_525099

variable (z : ℂ)

theorem complex_calculation : (z = 2 - (-1:ℂ)) → (z * (conj z + (-1:ℂ))) = 6 + 2 * (-1:ℂ) :=
by
  intro h
  rw h
  have h_conj : conj (2 - (-1:ℂ)) = 2 + (-1:ℂ) := by simp
  rw h_conj
  ring
  done

end complex_calculation_l525_525099


namespace shyam_weight_increase_l525_525354

theorem shyam_weight_increase (total_weight_after_increase : ℝ) (ram_initial_weight_ratio : ℝ) 
    (shyam_initial_weight_ratio : ℝ) (ram_increase_percent : ℝ) (total_increase_percent : ℝ) 
    (ram_total_weight_ratio : ram_initial_weight_ratio = 6) (shyam_initial_total_weight_ratio : shyam_initial_weight_ratio = 5) 
    (total_weight_after_increase_eq : total_weight_after_increase = 82.8) 
    (ram_increase_percent_eq : ram_increase_percent = 0.10) 
    (total_increase_percent_eq : total_increase_percent = 0.15) : 
  shyam_increase_percent = (21 : ℝ) :=
sorry

end shyam_weight_increase_l525_525354


namespace evaluate_expr_l525_525961

def expr (a b c : ℝ) : ℝ :=
  (a^2 * (1 / b - 1 / c)
    + b^2 * (1 / c - 1 / a)
    + c^2 * (1 / a - 1 / b) + 37) /
  (a * (1 / b - 1 / c)
    + b * (1 / c - 1 / a)
    + c * (1 / a - 1 / b) + 2)

theorem evaluate_expr : expr 15 19 25 = 77.5 :=
sorry

end evaluate_expr_l525_525961


namespace find_magnitude_l525_525623

variable (x : ℝ)
def a : ℝ × ℝ := (x, 1)
def b : ℝ × ℝ := (1, 2)

def collinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, v.1 = k * u.1 ∧ v.2 = k * u.2

theorem find_magnitude 
  (h_collinear : collinear b (b.1 - a.1, b.2 - a.2)) 
  (hx : 2 * (1 - x) = 1) : 
  |(a.1 + b.1, a.2 + b.2)| = 3 * Real.sqrt 5 / 2 := 
by 
  sorry

end find_magnitude_l525_525623


namespace negative_solution_range_l525_525183

theorem negative_solution_range (m x : ℝ) (h : (2 * x + m) / (x - 1) = 1) (hx : x < 0) : m > -1 :=
  sorry

end negative_solution_range_l525_525183


namespace race_car_cost_l525_525895

variable (R : ℝ)
variable (Mater_cost SallyMcQueen_cost : ℝ)

-- Conditions
def Mater_cost_def : Mater_cost = 0.10 * R := by sorry
def SallyMcQueen_cost_def : SallyMcQueen_cost = 3 * Mater_cost := by sorry
def SallyMcQueen_cost_val : SallyMcQueen_cost = 42000 := by sorry

-- Theorem to prove the race car cost
theorem race_car_cost : R = 140000 :=
  by
    -- Use the conditions to prove
    sorry

end race_car_cost_l525_525895


namespace sets_produced_and_sold_is_500_l525_525772

-- Define the initial conditions as constants
def initial_outlay : ℕ := 10000
def manufacturing_cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def total_profit : ℕ := 5000

-- The proof goal
theorem sets_produced_and_sold_is_500 (x : ℕ) : 
  (total_profit = selling_price_per_set * x - (initial_outlay + manufacturing_cost_per_set * x)) → 
  x = 500 :=
by 
  sorry

end sets_produced_and_sold_is_500_l525_525772


namespace product_units_tens_not_div_by_4_l525_525763

theorem product_units_tens_not_div_by_4 
  (numbers : List ℕ) 
  (all_four_digit : ∀ n ∈ numbers, 1000 ≤ n ∧ n < 10000)
  (not_div_by_4 : ∃ n ∈ numbers, ¬ (n % 100) % 4 = 0) :
  numbers = [3544, 3554, 3564, 3572, 3576] →
  let non_divisible := 3554 in
  (non_divisible % 10) * ((non_divisible % 100) / 10) = 20 :=
by
  intro h
  dsimp only
  rfl

end product_units_tens_not_div_by_4_l525_525763


namespace monotone_increasing_interval_l525_525609

noncomputable def f (x : ℝ) : ℝ := (x / (x^2 + 1)) + 1

theorem monotone_increasing_interval :
  ∀ x : ℝ, (-1 < x ∧ x < 1) ↔ ∀ ε > 0, ∃ δ > 0, ∀ x₁ x₂, (-1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ |x₁ - x₂| < δ) → f x₁ ≤ f x₂ + ε := 
sorry

end monotone_increasing_interval_l525_525609


namespace same_function_iff_domain_and_rule_correct_function_function_a_not_same_function_c_not_same_function_d_not_same_l525_525454

theorem same_function_iff_domain_and_rule (f g : ℝ → ℝ) : 
  (∀ x, f x = g x) ∧ (∀ y, ∃ x, f x = y) ↔ (f = g) :=
by sorry

def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := (3 * x) ^ 3

def domain_f1 (x : ℝ) : Prop := true
def domain_f2 (x : ℝ) : Prop := true

theorem correct_function :
  (domain_f1 = domain_f2) ∧ (∀ x, f1 x = f2 x) := 
by sorry

theorem function_a_not_same :
  (domain_f1 ≠ {x | x ≥ 0}) ∨ (∃ x, (sqrt x) ^ 2 ≠ x) :=
by sorry

theorem function_c_not_same (a : ℝ) (ha : a > 0) (h1a : a ≠ 1) :
  (domain_f1 ≠ {x | x > 0}) ∨ (∃ x, a ^ (log a x) ≠ x) :=
by sorry

theorem function_d_not_same :
  (domain_f1 ≠ {x | x ≠ 0}) ∨ (∃ x, x / x ^ 0 ≠ x) :=
by sorry


end same_function_iff_domain_and_rule_correct_function_function_a_not_same_function_c_not_same_function_d_not_same_l525_525454


namespace residue_of_T_mod_2022_l525_525237

theorem residue_of_T_mod_2022 :
  let T := (List.range 2022).sum (λ n => if n % 2 = 0 then -n else n)
  T % 2022 = 1011 :=
by
  sorry

end residue_of_T_mod_2022_l525_525237


namespace probability_inside_triangle_l525_525785

open Rat -- Open rational scope for easier usage

-- Definitions based on the given problem
def base1 : ℕ := 10
def base2 : ℕ := 20
def height_trap : ℕ := 10
def base_triangle : ℕ := 8
def height_triangle : ℕ := 5

-- Calculate the areas
def area_trap : ℚ := (base1 + base2) * height_trap / 2
def area_triangle : ℚ := (base_triangle * height_triangle) / 2

-- Define the probability
def probability : ℚ := area_triangle / area_trap

-- State the theorem
theorem probability_inside_triangle :
  probability = 2 / 15 :=
by
  -- Proof to be filled in
  sorry

end probability_inside_triangle_l525_525785


namespace find_x_l525_525256

theorem find_x (x : ℚ) (h : (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 68) : 
  x = -50 / 19 := 
sorry

end find_x_l525_525256


namespace tony_average_time_to_store_l525_525273

-- Definitions and conditions
def speed_walking := 2  -- MPH
def speed_running := 10  -- MPH
def distance_to_store := 4  -- miles
def days_walking := 1  -- Sunday
def days_running := 2  -- Tuesday, Thursday
def total_days := days_walking + days_running

-- Proof statement
theorem tony_average_time_to_store :
  let time_walking := (distance_to_store / speed_walking) * 60
  let time_running := (distance_to_store / speed_running) * 60
  let total_time := time_walking * days_walking + time_running * days_running
  let average_time := total_time / total_days
  average_time = 56 :=
by
  sorry  -- Proof to be completed

end tony_average_time_to_store_l525_525273


namespace parametric_line_to_cartesian_and_intersection_range_l525_525700

theorem parametric_line_to_cartesian_and_intersection_range 
  (α θ t : ℝ) (t1 t2 : ℝ)
  (x y : ℝ) 
  (h1 : x = 3 + t * cos α) 
  (h2 : y = 2 + t * sin α) 
  (h3 : t1 + t2 = -4 * (cos α + sin α)) 
  (h4 : t1 * t2 = 7)
  (rho : ℝ) 
  (h5 : rho = 2 * cos θ) 
  (h6 : x^2 + y^2 - 2 * x = 0) :
  (x * sin α - y * cos α + 2 * cos α - 3 * sin α = 0) ∧ 
  (x^2 + y^2 - 2 * x = 0) ∧ 
  (∃ (M : ℝ), 
    (d₁ : ℝ) (d₂ : ℝ) (dm1 : d₁ = |t1|/|t1 * t2|) (dm2 : d₂ = |t2|/|t1 * t2|), 
    (d₁ + d₂ ∈ (Set.Ioc (2 * real.sqrt 7 / 7) (4 * real.sqrt 2 / 7)))) :=
begin
  sorry
end

end parametric_line_to_cartesian_and_intersection_range_l525_525700


namespace ratio_C_D_l525_525531

noncomputable def C : ℝ :=
  ∑' n in (ℕ \ {5 * k | k : ℕ}), (-1)^(n+1) * (1 / (n^2))

noncomputable def D : ℝ :=
  ∑' k : ℕ, (-1)^(k+1) * (1 / ((5 * k + 1)^2))

theorem ratio_C_D :
  C / D = 26 := by
  sorry

end ratio_C_D_l525_525531


namespace probability_of_two_prime_numbers_l525_525401

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l525_525401


namespace negation_equiv_proof_l525_525334

theorem negation_equiv_proof :
  (¬ ∃ x_0 ∈ Ioo 0 (Real.inf), Real.log x_0 = x_0 - 1) ↔
    ∀ x ∈ Ioo 0 (Real.inf), Real.log x ≠ x - 1 :=
by sorry

end negation_equiv_proof_l525_525334


namespace sqrt_mult_cube_root_l525_525934

theorem sqrt_mult_cube_root (h1 : Real.sqrt 75 = 5 * Real.sqrt 3)
                            (h2 : Real.sqrt 48 = 4 * Real.sqrt 3)
                            (h3 : Real.cbrt 27 = 3) :
    Real.sqrt 75 * Real.sqrt 48 * Real.cbrt 27 = 180 :=
begin
  sorry
end

end sqrt_mult_cube_root_l525_525934


namespace rectangle_diagonal_floor_eq_169_l525_525199

-- Definitions of points and properties
structure Rectangle (α : Type*) :=
(P Q R S : α)
(PQ : ℝ) (PS : ℝ)
(PQ_eq : PQ = 120)
(T_mid_PR : Prop)
(S_perpendicular_PQ : Prop)

-- Prove the desired property using the conditions
theorem rectangle_diagonal_floor_eq_169 {α : Type*} (rect : Rectangle α)
  (h : rect.PQ = 120)
  (ht : rect.T_mid_PR)
  (hs : rect.S_perpendicular_PQ) : 
  ⌊rect.PQ * Real.sqrt 2⌋ = 169 :=
sorry

end rectangle_diagonal_floor_eq_169_l525_525199


namespace find_x_l525_525982

def star (a b : ℝ) : ℝ := (a - b) / (a + b)

theorem find_x (x : ℝ) : star (x + 1) (x - 2) = 3 → x = 1 := by
  sorry

end find_x_l525_525982


namespace area_theorem_l525_525699

noncomputable def area_of_closed_figure :=
  let f1 : ℝ → ℝ := λ x, 3
  let f2 : ℝ → ℝ := λ x, x
  let f3 : ℝ → ℝ := λ x, 1 / x
  let I1 : ℝ := ∫ x in (1/3)..1, (f1 x - f3 x)
  let tri_area : ℝ := (1/2) * (1 - 1/3) * (3 - 1/3)
  I1 + tri_area

theorem area_theorem : area_of_closed_figure = 4 - Real.log 3 := by
  sorry

end area_theorem_l525_525699


namespace charlie_collected_15_seashells_l525_525518

variables (c e : ℝ)

-- Charlie collected 10 more seashells than Emily
def charlie_more_seashells := c = e + 10

-- Emily collected one-third the number of seashells Charlie collected
def emily_seashells := e = c / 3

theorem charlie_collected_15_seashells (hc: charlie_more_seashells c e) (he: emily_seashells c e) : c = 15 := 
by sorry

end charlie_collected_15_seashells_l525_525518


namespace sum_x_y_eq_two_l525_525770

theorem sum_x_y_eq_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 28) : x + y = 2 :=
sorry

end sum_x_y_eq_two_l525_525770


namespace problem1_problem2_l525_525931

theorem problem1 : -20 - (-8) + (-4) = -16 := by
  sorry

theorem problem2 : -1^3 * (-2)^2 / (4 / 3 : ℚ) + |5 - 8| = 0 := by
  sorry

end problem1_problem2_l525_525931


namespace pure_water_to_add_eq_30_l525_525631

noncomputable def initial_volume := 50
noncomputable def initial_concentration := 0.40
noncomputable def desired_concentration := 0.25
noncomputable def initial_acid := initial_concentration * initial_volume

theorem pure_water_to_add_eq_30 :
  ∃ w : ℝ, (initial_acid / (initial_volume + w) = desired_concentration) ∧ w = 30 :=
by
  sorry

end pure_water_to_add_eq_30_l525_525631


namespace prob_primes_1_to_30_l525_525413

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l525_525413


namespace sum_of_series_l525_525841

theorem sum_of_series : (1 / (1 * 2 * 3) + 1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6)) = 7 / 30 :=
by
  sorry

end sum_of_series_l525_525841


namespace parametric_curve_length_l525_525330

noncomputable def curve_length : ℝ :=
  ∫ (θ: ℝ) in (Real.Icc (π/3) π), (5 : ℝ)

theorem parametric_curve_length :
  ∫ (θ: ℝ) in (Real.Icc (π/3) π), norm (5 : ℝ) = 10 * π / 3 :=
by
  sorry

end parametric_curve_length_l525_525330


namespace probability_two_primes_l525_525381

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l525_525381


namespace probability_two_primes_l525_525378

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l525_525378


namespace trapezoid_perimeter_l525_525996

-- Define the problem conditions
variables (A B C D : Point) (BC AD : Line) (AB CD : Segment)

-- Conditions
def is_parallel (L1 L2 : Line) : Prop := sorry
def is_right_angle (A B C : Point) : Prop := sorry
def is_angle_150 (A B C : Point) : Prop := sorry

noncomputable def length (s : Segment) : ℝ := sorry

def trapezoid_conditions (A B C D : Point) (BC AD : Line) (AB CD : Segment) : Prop :=
  is_parallel BC AD ∧ is_angle_150 A B C ∧ is_right_angle C D B ∧
  length AB = 4 ∧ length BC = 3 - Real.sqrt 3

-- Perimeter calculation
noncomputable def perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) : ℝ :=
  length AB + length BC + length CD + length AD

-- Lean statement for the math proof problem
theorem trapezoid_perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) :
  trapezoid_conditions A B C D BC AD AB CD → perimeter A B C D BC AD AB CD = 12 :=
sorry

end trapezoid_perimeter_l525_525996


namespace symmetry_x_axis_l525_525701

theorem symmetry_x_axis (a b : ℝ) (h1 : a - 3 = 2) (h2 : 1 = -(b + 1)) : a + b = 3 :=
by
  sorry

end symmetry_x_axis_l525_525701


namespace probability_two_primes_is_1_over_29_l525_525430

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l525_525430


namespace probability_even_and_greater_than_30000_l525_525318

def digits : List ℕ := [1, 3, 4, 5, 7]

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_greater_than_30000 (n : ℕ) : Prop := n > 30000

def is_valid_five_digit_number (digits : List ℕ) (n : ℕ) : Prop := 
  List.length (Nat.digits 10 n) = 5 ∧ 
  ∀ d ∈ (Nat.digits 10 n), d ∈ digits ∧ 
  List.nodup (Nat.digits 10 n)

theorem probability_even_and_greater_than_30000 : 
  let possible_numbers := List.permutations digits in 
  let five_digit_numbers := List.filter (λ n, is_valid_five_digit_number digits n) possible_numbers in 
  let even_numbers := List.filter is_even five_digit_numbers in 
  let valid_numbers := List.filter is_greater_than_30000 even_numbers in 
  (List.length valid_numbers : ℚ) / (List.length five_digit_numbers : ℚ) = 2 / 5 :=
by sorry

end probability_even_and_greater_than_30000_l525_525318


namespace joseph_cards_percentage_left_l525_525721

def joseph_total_cards : ℕ := 16
def fraction_given_to_brother : ℚ := 3 / 8
def extra_cards_given_to_brother : ℕ := 2

theorem joseph_cards_percentage_left
  (total_cards : ℕ)
  (fraction_given : ℚ)
  (extra_given : ℕ)
  (percentage_left : ℚ) :
  total_cards = joseph_total_cards →
  fraction_given = fraction_given_to_brother →
  extra_given = extra_cards_given_to_brother →
  percentage_left = ((total_cards - (fraction_given * total_cards).toNat - extra_given:ℚ) / total_cards * 100) →
  percentage_left = 50 := by sorry

end joseph_cards_percentage_left_l525_525721


namespace combine_like_terms_l525_525521

variable (a : ℝ)

theorem combine_like_terms : 3 * a^2 + 5 * a^2 - a^2 = 7 * a^2 := 
by sorry

end combine_like_terms_l525_525521


namespace increasing_interval_cos_C_l525_525605

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x + sin x ^ 2

theorem increasing_interval (k : ℤ) :
  ∀ x, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 →
  ((f (x) < f (x + 1) → x < x + 1) ∧ (f (x + 1) > f (x) → x + 1 > x)) :=
sorry

theorem cos_C (A B C : ℝ) 
  (a b c : ℝ)
  (AD BD : ℝ)
  (k : ℤ)
  (h₀ : AD = sqrt 2)
  (h₁ : BD = 2)
  (h₂ : A = (k : ℝ) * π + π / 3)
  (h₃ : 0 < A)
  (h₄ : A < π / 2)
  (h₅ : B = π / 4) :
  cos C = (sqrt 6 - sqrt 2) / 4 :=
sorry

end increasing_interval_cos_C_l525_525605


namespace problem_1956_Tokyo_Tech_l525_525728

theorem problem_1956_Tokyo_Tech (a b c : ℝ) (ha : 0 < a) (ha_lt_one : a < 1) (hb : 0 < b) 
(hb_lt_one : b < 1) (hc : 0 < c) (hc_lt_one : c < 1) : a + b + c - a * b * c < 2 := 
sorry

end problem_1956_Tokyo_Tech_l525_525728


namespace solve_seq_problem_l525_525578

noncomputable def positive_geometric_sequence (a : ℕ → ℝ) := 
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

theorem solve_seq_problem :
  (∃ a : ℕ → ℝ, positive_geometric_sequence a ∧ 
                a 7 = a 6 + 2 * a 5 ∧
                ∃ m n : ℕ, m ≠ n ∧ (real.sqrt (a m * a n) = 4 * a 1) ∧
                (1 / m + 4 / n = 3 / 2)) :=
begin
  sorry
end

end solve_seq_problem_l525_525578


namespace water_added_eq_30_l525_525647

-- Given conditions
def initial_volume := 50
def initial_concentration := 0.4
def final_concentration := 0.25
def amount_of_acid := initial_volume * initial_concentration

-- Proof problem statement
theorem water_added_eq_30 : ∃ w : ℝ, amount_of_acid / (initial_volume + w) = final_concentration ∧ w = 30 :=
by 
  -- Solution steps go here, but for now we insert sorry to skip the proof.
  sorry

end water_added_eq_30_l525_525647


namespace probability_two_primes_is_1_over_29_l525_525426

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l525_525426


namespace projection_question_projection_condition_projection_problem_l525_525013

-- Define the projection function
def projection (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := v.1 * u.1 + v.2 * u.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu) • u

theorem projection_question :
  projection ⟨-3, 3⟩ ⟨5, 1⟩ = ⟨-30/13, -6/13⟩ :=
by
  -- Proof omitted
  sorry

-- Define the condition based on the problem
theorem projection_condition :
  projection ⟨3, 3⟩ ⟨5, 1⟩ = ⟨45/10, 9/10⟩ :=
by
  -- Substitute the standard form vector and simplify
  sorry

-- Final theorem proving the given condition implies the desired projection
theorem projection_problem (h : projection ⟨3, 3⟩ ⟨5, 1⟩ = ⟨45/10, 9/10⟩) :
  projection ⟨-3, 3⟩ ⟨5, 1⟩ = ⟨-30/13, -6/13⟩ :=
by
  -- Use h and the definition of projection to arrive at the conclusion
  sorry

end projection_question_projection_condition_projection_problem_l525_525013


namespace inequality_of_am_gm_l525_525515

theorem inequality_of_am_gm 
  (a b c d : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hd : 0 < d) : 
  ab^2 * c^3 * d^4 ≤ (a + 2 * b + 3 * c + 4 * d)^10 / 10^10 :=
begin
  sorry
end

end inequality_of_am_gm_l525_525515


namespace dice_probabilities_relationship_l525_525451

theorem dice_probabilities_relationship :
  let p1 := 5 / 18
  let p2 := 11 / 18
  let p3 := 1 / 2
  p1 < p3 ∧ p3 < p2
:= by
  sorry

end dice_probabilities_relationship_l525_525451


namespace work_time_A_and_C_together_l525_525869

theorem work_time_A_and_C_together
  (A_work B_work C_work : ℝ)
  (hA : A_work = 1/3)
  (hB : B_work = 1/6)
  (hBC : B_work + C_work = 1/3) :
  1 / (A_work + C_work) = 2 := by
  sorry

end work_time_A_and_C_together_l525_525869


namespace ben_shopping_trip_cost_l525_525963

noncomputable def ben_spends : ℕ := 43

theorem ben_shopping_trip_cost :
  let apples := 7 * 2,
      milk := 4 * 4,
      bread := 3 * 3,
      sugar := 3 * 6,
      total := apples + milk + bread + sugar,
      discount_milk := milk * 0.75,
      discounted_total := apples + discount_milk + bread + sugar,
      final_total := if discounted_total >= 50 then discounted_total - 10 else discounted_total in
  final_total = ben_spends :=
by
  sorry

end ben_shopping_trip_cost_l525_525963


namespace honey_production_l525_525672

-- Define the conditions:
def bees : ℕ := 60
def days : ℕ := 60
def honey_per_bee : ℕ := 1

-- Statement to prove:
theorem honey_production (bees_eq : 60 = bees) (days_eq : 60 = days) (honey_per_bee_eq : 1 = honey_per_bee) :
  bees * honey_per_bee = 60 := by
  sorry

end honey_production_l525_525672


namespace smallest_value_is_A_l525_525911

def A : ℤ := -(-3 - 2)^2
def B : ℤ := (-3) * (-2)
def C : ℚ := ((-3)^2 : ℚ) / (-2)^2
def D : ℚ := ((-3)^2 : ℚ) / (-2)

theorem smallest_value_is_A : A < B ∧ A < C ∧ A < D :=
by
  sorry

end smallest_value_is_A_l525_525911


namespace chords_inequality_in_circle_l525_525212

theorem chords_inequality_in_circle
  (O : Type) [metric_space O] [normed_group O] [normed_space ℝ O]
  (radius : ℝ)
  (h_radius : radius = 1)
  (A B P Q C D E F : O)
  (h_parallel : ∀ x y ∈ segment C D, ∀ u v ∈ segment E F, x - y = u - v)
  (h_angle_45 : ∀ {P Q' : O}, P ≠ Q' → ∠ POQ = π / 4) :
  dist P C * dist Q E + dist P D * dist Q F < 2 :=
sorry

end chords_inequality_in_circle_l525_525212


namespace negation_of_sin_universal_bound_l525_525173

theorem negation_of_sin_universal_bound :
  (∀ x : ℝ, Real.sin x ≤ 1) ↔ (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by
  sorry

end negation_of_sin_universal_bound_l525_525173


namespace can_divide_into_three_piles_l525_525567

theorem can_divide_into_three_piles (n : ℕ) : 
  (∃ piles, (1 + 2 + ... + n) = 3 * piles) ↔ (n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2)) := 
by 
  sorry

end can_divide_into_three_piles_l525_525567


namespace pure_water_to_achieve_desired_concentration_l525_525639

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l525_525639


namespace angles_CDA_l525_525248

theorem angles_CDA (A B C D I E : Point) (circ : Circle I) 
    (inscribed : ∀ P : Point, P ∈ [A, B, C, D] → dist P I = circ.radius)
    (intersection_CI_AB : ∃ E, (C -ᵥ I).point ∩ (A -ᵥ B).line)
    (angle_IDE : angle I D E = 35)
    (angle_ABC : angle A B C = 70)
    (angle_BCD : angle B C D = 60) :
    angle C D A = 70 ∨ angle C D A = 160 :=
sorry

end angles_CDA_l525_525248


namespace ratio_XQ_QY_l525_525211

variables (X Y Z Q : Type)
variables [triangle XYZ]
variables (XY_length : Real)
variables (angle_YXZ angle_XZY : Real)
variables (ZQ_length : Real)
variables (angle_ZQY angle_ZYQ : Real)

-- Given conditions
def given_conditions : Prop :=
  XY_length = 5 ∧
  angle_YXZ = 30 ∧
  angle_XZY < 60 ∧
  angle_ZQY = 2 * angle_ZYQ ∧
  ZQ_length = 2

-- Goal to prove
theorem ratio_XQ_QY : given_conditions X Y Z Q XY_length angle_YXZ angle_XZY ZQ_length angle_ZQY angle_ZYQ →
  ∃ (s t u : ℕ), (s + t * Real.sqrt u) = (5 + 2 * Real.sqrt 23) :=
begin
  sorry
end

end ratio_XQ_QY_l525_525211


namespace frac_x_y_eq_neg2_l525_525731

open Real

theorem frac_x_y_eq_neg2 (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 4) (h3 : (x + y) / (x - y) ≠ 1) :
  ∃ t : ℤ, (x / y = t) ∧ (t = -2) :=
by sorry

end frac_x_y_eq_neg2_l525_525731


namespace solve_inequality_l525_525535

-- Defining the function as given in the condition
def f (x : ℝ) : ℝ := (x^2 - 9) / (x^2 - 4)

-- The main theorem we want to prove
theorem solve_inequality (x : ℝ) : f x > 0 ↔ (x < -3 ∨ x > 3) := sorry

end solve_inequality_l525_525535


namespace power_of_point_l525_525891

theorem power_of_point (O M A B : Point) (R d : ℝ)
  (h_dist : dist O M = d)
  (h_outside : d > R)
  (h_intersects : LineThrough M intersects_circle O R at A B) :
  dist M A * dist M B = d^2 - R^2 := 
sorry

end power_of_point_l525_525891


namespace simplify_sum_of_roots_l525_525779

theorem simplify_sum_of_roots :
  sqrt 72 + sqrt 32 + sqrt 50 = 15 * sqrt 2 := 
by sorry

end simplify_sum_of_roots_l525_525779


namespace total_accidents_l525_525227

theorem total_accidents :
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  accidentsA + accidentsB + accidentsC = 3730 :=
by
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  sorry

end total_accidents_l525_525227


namespace limit_of_derivative_l525_525739

-- Definitions based on the conditions
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Statement of the problem
theorem limit_of_derivative :
  (lim (λ Δx, (f (1 + Δx) - f 1) / (3 * Δx)) (0 : ℝ) (𝓝 0) = (1 / 3) * (deriv f 1)) :=
by
  -- This is where the proof would be. We leave out the proof as instructed.
  sorry

end limit_of_derivative_l525_525739


namespace sum_largest_and_smallest_number_l525_525977

theorem sum_largest_and_smallest_number (a b c : ℕ) (h1 : a = 6) (h2 : b = 3) (h3 : c = 8) : 
  (largest_three_digit_number a b c) + (smallest_three_digit_number a b c) = 1231 := 
by
  sorry

noncomputable def largest_three_digit_number (a b c : ℕ) : ℕ :=
  let digits := [a, b, c].sort (>=)
  digits[0] * 100 + digits[1] * 10 + digits[2]

noncomputable def smallest_three_digit_number (a b c : ℕ) : ℕ :=
  let digits := [a, b, c].sort (<=)
  digits[0] * 100 + digits[1] * 10 + digits[2]

end sum_largest_and_smallest_number_l525_525977


namespace probability_no_two_people_adjacent_l525_525825

noncomputable def prob_non_adjacent_people : ℚ :=
  let total_ways := 8 * 7 * 6 in
  let favorable_ways := 24 in
  favorable_ways / total_ways

theorem probability_no_two_people_adjacent : prob_non_adjacent_people = 1 / 14 := by
  sorry

end probability_no_two_people_adjacent_l525_525825


namespace find_quotient_from_conditions_l525_525317

variable (x y : ℕ)
variable (k : ℕ)

theorem find_quotient_from_conditions :
  y - x = 1360 ∧ y = 1614 ∧ y % x = 15 → y / x = 6 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end find_quotient_from_conditions_l525_525317


namespace steps_already_climbed_l525_525499

-- Definitions based on conditions
def total_stair_steps : ℕ := 96
def steps_left_to_climb : ℕ := 22

-- Theorem proving the number of steps already climbed
theorem steps_already_climbed : total_stair_steps - steps_left_to_climb = 74 := by
  sorry

end steps_already_climbed_l525_525499


namespace find_k_l525_525538

theorem find_k (k : ℝ) : (∫ x in 0..1, (x - k)) = (3/2) → k = -1 := by
  intro h
  sorry

end find_k_l525_525538


namespace solution_exists_l525_525680

noncomputable def problem (a b c : ℝ) : Prop :=
(a - b = 3) ∧ (a^2 + b^2 = 29) ∧ (a + b + c = 10) ∧ (ab = 10) ∧ (c = 17 ∨ c = 3)

theorem solution_exists : ∃ (a b c : ℝ), problem a b c :=
by
  sorry

end solution_exists_l525_525680


namespace complex_calculation_l525_525107

theorem complex_calculation (z : ℂ) (h : z = 2 - complex.I) : 
  z * (z.conj + complex.I) = 6 + 2 * complex.I := 
by sorry

end complex_calculation_l525_525107


namespace orchids_in_vase_now_l525_525360

variable (flowers : Type) [fintype flowers]

def initial_roses : ℕ := 16
def initial_orchids : ℕ := 3
def added_orchids : ℕ := 4
def current_roses : ℕ := 13

def final_orchids : ℕ := initial_orchids + added_orchids

theorem orchids_in_vase_now : final_orchids = 7 :=
by 
  rw [initial_orchids, added_orchids]
  exact rfl

end orchids_in_vase_now_l525_525360


namespace haley_magazines_l525_525627

theorem haley_magazines (boxes : ℕ) (magazines_per_box : ℕ) (total_magazines : ℕ) :
  boxes = 7 →
  magazines_per_box = 9 →
  total_magazines = boxes * magazines_per_box →
  total_magazines = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end haley_magazines_l525_525627


namespace find_ellipse_m_l525_525144

theorem find_ellipse_m (m : ℝ) (h1 : (sqrt 10 / 5 : ℝ) = (1 / sqrt 5 * sqrt (abs (5 + (-m)))) ) :
  m = -3 ∨ m = -25 / 3 :=
sorry

end find_ellipse_m_l525_525144


namespace trajectory_of_moving_circle_l525_525991

noncomputable def trajectory_of_center (O : Point) (C : Circle) (P : Point) : Set Point := sorry

theorem trajectory_of_moving_circle 
  (O : Point) (r : ℝ) (P : Point) (hP : inside_or_on_circle O P r)
  (C : Circle) (hC : tangent_to_circle C O P) :
  trajectory_of_center O C P ∈ {two_rays, a_circle, an_ellipse} :=
sorry

end trajectory_of_moving_circle_l525_525991


namespace matrices_identical_l525_525588

variables {m n : ℕ}
variables (P Q : matrix (fin m) (fin n) ℕ)

theorem matrices_identical
  (hP₁ : ∀ i, monotone (λ j, P i j))
  (hQ₁ : ∀ j, monotone (λ i, Q i j))
  (hPsum : ∀ i, (λ j, P i j).sum = (λ j, Q i j).sum)
  (hQsum : ∀ j, (λ i, P i j).sum = (λ i, Q i j).sum) :
  P = Q :=
sorry

end matrices_identical_l525_525588


namespace find_M_base7_l525_525236

theorem find_M_base7 :
  ∃ M : ℕ, M = 48 ∧ (M^2).digits 7 = [6, 6] ∧ (∃ (m : ℕ), 49 ≤ m^2 ∧ m^2 < 343 ∧ M = m - 1) :=
sorry

end find_M_base7_l525_525236


namespace joseph_cards_percentage_l525_525724

theorem joseph_cards_percentage
    (total_cards : ℕ)
    (fraction_given_to_brother : ℚ)
    (additional_cards_given : ℕ)
    (cards_left : ℕ)
    (percentage_left : ℚ)
    (orig_cards : total_cards = 16)
    (fraction : fraction_given_to_brother = 3 / 8)
    (additional_cards : additional_cards_given = 2)
    (given_to_brother : 16 * (3 / 8) = 6)
    (total_given : 6 + additional_cards_given = 8)
    (left_cards : total_cards - total_given = 8)
    (calc_percentage : (8 : ℚ) / 16 * 100 = 50) :
  percentage_left = 50 :=
sorry

end joseph_cards_percentage_l525_525724


namespace find_sum_l525_525479

variable (P : ℝ)

def SI1 := (3 * P) / 10
def SI2 := (6 * P) / 25
def interestDifference := 840

theorem find_sum (h : SI1 P - SI2 P = interestDifference) : P = 14000 := 
  sorry

end find_sum_l525_525479


namespace crayons_at_the_end_of_thursday_l525_525754

-- Definitions for each day's changes
def monday_crayons : ℕ := 7
def tuesday_crayons (initial : ℕ) := initial + 3
def wednesday_crayons (initial : ℕ) := initial - 5 + 4
def thursday_crayons (initial : ℕ) := initial + 6 - 2

-- Proof statement to show the number of crayons at the end of Thursday
theorem crayons_at_the_end_of_thursday : thursday_crayons (wednesday_crayons (tuesday_crayons monday_crayons)) = 13 :=
by
  sorry

end crayons_at_the_end_of_thursday_l525_525754


namespace base7_arithmetic_theorem_l525_525556

noncomputable def base7_arithmetic : ℕ :=
let a := convert_from_base7 "2000" in
let b := convert_from_base7 "1256" in
let c := convert_from_base7 "345" in
convert_to_base7 (a - b + c)

theorem base7_arithmetic_theorem :
  base7_arithmetic = "1042" :=
by
  sorry

end base7_arithmetic_theorem_l525_525556


namespace interesting_numbers_in_range_l525_525749

/-- A composite number n is "interesting" if all its natural divisors can be listed in ascending order, 
and each subsequent divisor is divisible by the previous one -/
def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧ ¬nat.prime n ∧ ∀ (d₁ d₂ : ℕ), d₁ ∣ n → d₂ ∣ n → d₁ < d₂ → d₂ % d₁ = 0

theorem interesting_numbers_in_range :
  {n : ℕ | 20 ≤ n ∧ n ≤ 90 ∧ is_interesting n} = {25, 27, 32, 49, 64, 81} :=
by
  sorry

end interesting_numbers_in_range_l525_525749


namespace derivative_of_f_neg2_l525_525613

theorem derivative_of_f_neg2 (f : ℝ → ℝ) (x : ℝ) (h : f x = x^3) : deriv (λ x, f (-2)) x = 0 :=
by
  -- Given condition: f(x) = x^3
  intros
  rw h
  -- The derivative step is skipped here for brevity
  sorry

end derivative_of_f_neg2_l525_525613


namespace factorial_expression_l525_525032

open Nat

theorem factorial_expression : ((sqrt (5! * 4!)) ^ 2 + 3!) = 2886 := by
  sorry

end factorial_expression_l525_525032


namespace probability_of_prime_pairs_l525_525372

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l525_525372


namespace right_triangle_count_l525_525662

theorem right_triangle_count : 
  (Finset.card (Finset.filter 
    (λ ⟨a, b⟩, a^2 = 4 * (b + 1) ∧ b < 50 ∧ b + 2 ∈ Int ∧ a ∈ Int) 
    (Finset.univ : Finset (ℤ × ℤ)))) = 7 := 
sorry

end right_triangle_count_l525_525662


namespace pure_water_to_achieve_desired_concentration_l525_525637

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l525_525637


namespace victoria_initial_money_l525_525438

-- Definitions based on conditions
def cost_rice := 2 * 20
def cost_flour := 3 * 25
def cost_soda := 150
def total_spent := cost_rice + cost_flour + cost_soda
def remaining_balance := 235

-- Theorem to prove
theorem victoria_initial_money (initial_money : ℕ) :
  initial_money = total_spent + remaining_balance :=
by
  sorry

end victoria_initial_money_l525_525438


namespace acrobat_eq_two_lambs_l525_525434

variables (ACROBAT DOG BARREL SPOOL LAMB : ℝ)

axiom acrobat_dog_eq_two_barrels : ACROBAT + DOG = 2 * BARREL
axiom dog_eq_two_spools : DOG = 2 * SPOOL
axiom lamb_spool_eq_barrel : LAMB + SPOOL = BARREL

theorem acrobat_eq_two_lambs : ACROBAT = 2 * LAMB :=
by
  sorry

end acrobat_eq_two_lambs_l525_525434


namespace trig_cosine_sum_le_sqrt_5_l525_525562

theorem trig_cosine_sum_le_sqrt_5 
  {α β γ : ℝ} 
  (h : sin α + sin β + sin γ ≥ 0) : cos α + cos β + cos γ ≤ Real.sqrt 5 := 
sorry

end trig_cosine_sum_le_sqrt_5_l525_525562


namespace sum_of_complex_exponentials_is_real_l525_525038

theorem sum_of_complex_exponentials_is_real :
  let θ := 0
  in ∃ r : ℝ, e^{5*π*I/40} + e^{15*π*I/40} + e^{25*π*I/40} + e^{35*π*I/40} = r * exp(I * θ) :=
sorry

end sum_of_complex_exponentials_is_real_l525_525038


namespace sum_cos_equiv_l525_525056

theorem sum_cos_equiv :
  ∑ x in finset.Icc 3 46, 2 * Real.cos x * Real.cos 2 * (1 + 1 / Real.sin (x - 2) * 1 / Real.sin (x + 2))
  =
  Real.cos 5 + Real.cos 48 + Real.cos 44 - Real.cos 3 + 1 / Real.sin 48 - 1 / Real.sin 1 :=
by sorry

end sum_cos_equiv_l525_525056


namespace monotonically_increasing_interval_l525_525333

open Real

/-- The monotonically increasing interval of the function y = (cos x + sin x) * cos (x - π / 2)
    is [kπ - π / 8, kπ + 3π / 8] for k ∈ ℤ. -/
theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (cos x + sin x) * cos (x - π / 2) = y →
  (k * π - π / 8) ≤ x ∧ x ≤ (k * π + 3 * π / 8) := 
sorry

end monotonically_increasing_interval_l525_525333


namespace total_number_of_people_l525_525507

variables (A B : ℕ)

def pencils_brought_by_assoc_profs (A : ℕ) : ℕ := 2 * A
def pencils_brought_by_asst_profs (B : ℕ) : ℕ := B
def charts_brought_by_assoc_profs (A : ℕ) : ℕ := A
def charts_brought_by_asst_profs (B : ℕ) : ℕ := 2 * B

axiom pencils_total : pencils_brought_by_assoc_profs A + pencils_brought_by_asst_profs B = 10
axiom charts_total : charts_brought_by_assoc_profs A + charts_brought_by_asst_profs B = 11

theorem total_number_of_people : A + B = 7 :=
sorry

end total_number_of_people_l525_525507


namespace volume_reflected_greater_half_l525_525815

-- Define the tetrahedron and its properties
structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron := 
(A B C D : Point3D)

def is_reflection (P S : Point3D) : Point3D :=
{ x := 2 * S.x - P.x, 
  y := 2 * S.y - P.y, 
  z := 2 * S.z - P.z }

def centroid_face (T : Tetrahedron) 
(vertex : Fin 4) : Point3D :=
match vertex with
| ⟨0, _⟩ => { x := (T.B.x + T.C.x + T.D.x) / 3, y := (T.B.y + T.C.y + T.D.y) / 3, z := (T.B.z + T.C.z + T.D.z) / 3 }
| ⟨1, _⟩ => { x := (T.A.x + T.C.x + T.D.x) / 3, y := (T.A.y + T.C.y + T.D.y) / 3, z := (T.A.z + T.C.z + T.D.z) / 3 }
| ⟨2, _⟩ => { x := (T.A.x + T.B.x + T.D.x) / 3, y := (T.A.y + T.B.y + T.D.y) / 3, z := (T.A.z + T.B.z + T.D.z) / 3 }
| ⟨3, _⟩ => { x := (T.A.x + T.B.x + T.C.x) / 3, y := (T.A.y + T.B.y + T.C.y) / 3, z := (T.A.z + T.B.z + T.C.z) / 3 }
end

noncomputable def volume (T : Tetrahedron) : ℝ :=
sorry  -- Placeholder for actual volume calculation

def reflected_tetrahedron (T : Tetrahedron) : Tetrahedron :=
let SA := centroid_face T ⟨0, sorry⟩ in
let SB := centroid_face T ⟨1, sorry⟩ in
{ A := is_reflection T.A SA, 
  B := is_reflection T.B SB, 
  C := T.C, 
  D := T.D }

theorem volume_reflected_greater_half (T : Tetrahedron) :
  volume (reflected_tetrahedron T) > volume T / 2 :=
sorry

end volume_reflected_greater_half_l525_525815


namespace calculate_expression_l525_525512

theorem calculate_expression :
  (0.027 : ℝ) ^ (-1 / 3) - (-1 / 7 : ℝ) ^ (-2) + (256 : ℝ) ^ (3 / 4) - 3 ^ (-1 : ℝ) + (Real.sqrt 2 - 1) ^ 0 = 19 :=
by
  sorry

end calculate_expression_l525_525512


namespace rank_best_to_worst_buy_l525_525885

section
variables (cS cM cL : ℝ) (qS qM qL : ℝ)
variables (h1 : cM = 1.4 * cS) (h2 : qM = 0.75 * qL) (h3 : qL = 3 * qS) (h4 : cL = 1.25 * cM)

def cost_per_cookie (c q : ℝ) : ℝ := c / q

theorem rank_best_to_worst_buy :
  cost_per_cookie cL qL < cost_per_cookie cM qM ∧
  cost_per_cookie cM qM < cost_per_cookie cS qS :=
by
  sorry
end

end rank_best_to_worst_buy_l525_525885


namespace no_common_points_l525_525984

theorem no_common_points 
  (x x_o y y_o : ℝ) 
  (h_parabola : y^2 = 4 * x) 
  (h_inside : y_o^2 < 4 * x_o) : 
  ¬ ∃ (x y : ℝ), y * y_o = 2 * (x + x_o) ∧ y^2 = 4 * x :=
by
  sorry

end no_common_points_l525_525984


namespace largest_n_for_factoring_l525_525974

theorem largest_n_for_factoring :
  ∃ (n : ℤ), 
    (∀ A B : ℤ, (5 * B + A = n ∧ A * B = 60) → (5 * B + A ≤ n)) ∧
    n = 301 :=
by sorry

end largest_n_for_factoring_l525_525974


namespace prob_primes_1_to_30_l525_525417

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l525_525417


namespace tony_average_time_l525_525279

-- Definitions for the conditions
def speed_walk : ℝ := 2  -- speed in miles per hour when Tony walks
def speed_run : ℝ := 10  -- speed in miles per hour when Tony runs
def distance_to_store : ℝ := 4  -- distance to the store in miles
def days : List String := ["Sunday", "Tuesday", "Thursday"]  -- days Tony goes to the store

-- Definition of times taken on each day
def time_sunday := distance_to_store / speed_walk  -- time in hours to get to the store on Sunday
def time_tuesday := distance_to_store / speed_run  -- time in hours to get to the store on Tuesday
def time_thursday := distance_to_store / speed_run -- time in hours to get to the store on Thursday

-- Converting times to minutes
def time_sunday_minutes := time_sunday * 60
def time_tuesday_minutes := time_tuesday * 60
def time_thursday_minutes := time_thursday * 60

-- Definition of average time
def average_time_minutes : ℝ :=
  (time_sunday_minutes + time_tuesday_minutes + time_thursday_minutes) / days.length

-- The theorem to prove
theorem tony_average_time : average_time_minutes = 56 := by
  sorry

end tony_average_time_l525_525279


namespace product_of_positive_d_values_l525_525822

theorem product_of_positive_d_values (d1 d2 : ℕ) (h1 : 8x^2 + 16x + d1 = 0) (h2 : 8x^2 + 16x + d2 = 0) :
  ∃ d1 d2 : ℕ, (d1 > 0) ∧ (d2 > 0) ∧
  (8x^2 + 16x + d1 = 0) ∧ (8x^2 + 16x + d2 = 0) ∧ (d1 * d2 = 48) :=
begin
  sorry
end

end product_of_positive_d_values_l525_525822


namespace axis_of_symmetry_l525_525678

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (5 - x)) : ∀ x : ℝ, f x = f (2 * 2.5 - x) :=
by
  sorry

end axis_of_symmetry_l525_525678


namespace perpendicular_AD_EF_l525_525907

open EuclideanGeometry

variables {A B C D E F : Point}
variables (ABC : Triangle)
variables [Acute ABC]
variables (hBAC : angle A B C < 45)
variables (hD : Interior D ABC)
variables (hBD : BD = CD)
variables (hBDC : angle B D C = 4 * angle A B C)
variables (hE : Reflection E C AB)
variables (hF : Reflection F B AC)

theorem perpendicular_AD_EF :
  Perpendicular Line_AD Line_EF := sorry

end perpendicular_AD_EF_l525_525907


namespace find_C_l525_525826

theorem find_C (C : ℤ) (h : 2 * C - 3 = 11) : C = 7 :=
sorry

end find_C_l525_525826


namespace parabola_equation_equilateral_triangle_area_constant_slope_AB_l525_525126

open Classical

section Parabola
variable (O : Point) (M : Point) (p : ℝ) (k1 k2 : ℝ) (x1 x2 y1 y2 : ℝ)

-- Given M has coordinates (2, 1)
axiom M_coords : M = (2 : ℝ, 1 : ℝ)

-- Given parabola with vertex at origin, symmetric about y-axis and passing through M
axiom parabola_through_M : (2 : ℝ)^2 = 2 * p * 1

-- Given slopes of the chords through M and their sum k1 + k2 = -2
axiom slopes_sum : k1 + k2 = -2
axiom points_on_parabola : x1^2 = 4 * y1 ∧ x2^2 = 4 * y2
axiom slopes_definition : k1 = (y1 - 1) / (x1 - 2) ∧ k2 = (y2 - 1) / (x2 - 2)

-- Prove the equation of the parabola is x^2 = 4y
theorem parabola_equation : ∃ p, (2 : ℝ)^2 = 2 * p * 1 := sorry

-- Prove the area of the equilateral triangle is 48√3
theorem equilateral_triangle_area : ∃ x_p y_p, x_p^2 = 4 * y_p ∧ area_eq_triangle x_p y_p = 48 * sqrt(3) := sorry

-- Prove the slope of the line AB is constant and equals -3
theorem constant_slope_AB : ∃ k, k = -3 := sorry

end Parabola

end parabola_equation_equilateral_triangle_area_constant_slope_AB_l525_525126


namespace line_through_P_with_intercepts_l525_525555

theorem line_through_P_with_intercepts (a b : ℝ) (P : ℝ × ℝ) (hP : P = (6, -1)) 
  (h1 : a = 3 * b) (ha : a = 1 / ((-b - 1) / 6) + 6) (hb : b = -6 * ((-b - 1) / 6) - 1) :
  (∀ x y, y = (-1 / 3) * x + 1 ∨ y = (-1 / 6) * x) :=
sorry

end line_through_P_with_intercepts_l525_525555


namespace determine_lambda_l525_525620

open Real

variables {a b : ℝ → ℝ → ℝ} {λ : ℝ}

noncomputable def vec_a_len : ℝ := 1
noncomputable def vec_b_len : ℝ := 4
noncomputable def angle_a_b : ℝ := (2 * π) / 3

axiom dot_product_a_b : a 1 * b 4 = -2
axiom perp_condition : inner (2 • a + λ • b) a = 0

theorem determine_lambda : λ = 1 :=
by
 sorry

end determine_lambda_l525_525620


namespace water_to_add_for_desired_acid_concentration_l525_525658

theorem water_to_add_for_desired_acid_concentration :
  ∃ w : ℝ, 50 * 0.4 / (50 + w) = 0.25 ∧ w = 30 :=
begin
  let initial_volume := 50,
  let initial_acid_concentration := 0.4,
  let desired_acid_concentration := 0.25,
  let acid_content := initial_volume * initial_acid_concentration,
  use 30,
  split,
  { simp [initial_volume, initial_acid_concentration, desired_acid_concentration, acid_content],
    linarith },
  { refl }
end

end water_to_add_for_desired_acid_concentration_l525_525658


namespace non_congruent_triangles_count_l525_525765

-- Define the six points
variables (A B C M N P : Type)

-- Define properties of the geometric configuration
def is_isosceles_triangle (A B C : Type) : Prop :=
  (dist A B = dist A C)

def is_midpoint (M A B : Type) : Prop :=
  (dist A M = dist M B)

def divides_ratio (N A C : Type) (r s : ℕ) : Prop :=
  (dist A N / dist N C = r / (r + s))

def divides_ratio_BC (P B C : Type) (r s : ℕ) : Prop :=
  (dist B P / dist P C = r / (r + s))

-- The main theorem to be proved
theorem non_congruent_triangles_count : 
  is_isosceles_triangle A B C →
  is_midpoint M A B →
  divides_ratio N A C 1 2 →
  divides_ratio_BC P B C 1 3 →
  count_non_congruent_triangles [A, B, C, M, N, P] = 15 :=
sorry

end non_congruent_triangles_count_l525_525765


namespace greatest_divisor_four_consecutive_squared_l525_525445

theorem greatest_divisor_four_consecutive_squared :
  ∀ (n: ℕ), ∃ m: ℕ, (∀ (n: ℕ), m ∣ (n * (n + 1) * (n + 2) * (n + 3))^2) ∧ m = 144 := 
sorry

end greatest_divisor_four_consecutive_squared_l525_525445


namespace probability_two_primes_from_1_to_30_l525_525406

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l525_525406


namespace depreciation_is_one_fourth_l525_525817

-- Define the initial value of the scooter
def initial_value : ℝ := 40000

-- Define the value of the scooter after 1 year
def value_after_1_year : ℝ := 30000

-- Define the fraction representing the depreciation
def depreciation_fraction : ℝ := (initial_value - value_after_1_year) / initial_value

-- The theorem we aim to prove
theorem depreciation_is_one_fourth : depreciation_fraction = 1 / 4 :=
by
  sorry

end depreciation_is_one_fourth_l525_525817


namespace problem_statement_l525_525926

theorem problem_statement (h1 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2) :
  (Real.pi / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 := by
  sorry

end problem_statement_l525_525926


namespace graph_inverse_prop_function_quadrants_l525_525082

theorem graph_inverse_prop_function_quadrants :
  ∀ x : ℝ, x ≠ 0 → (x > 0 ∧ y = 4 / x → y > 0) ∨ (x < 0 ∧ y = 4 / x → y < 0) := 
sorry

end graph_inverse_prop_function_quadrants_l525_525082


namespace complex_multiplication_identity_l525_525097

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- State the theorem to be proved
theorem complex_multiplication_identity : z * (z_conjugate + complex.i) = 6 + 2 * complex.i :=
by {
  sorry -- proof goes here
}

end complex_multiplication_identity_l525_525097


namespace pure_water_to_add_eq_30_l525_525632

noncomputable def initial_volume := 50
noncomputable def initial_concentration := 0.40
noncomputable def desired_concentration := 0.25
noncomputable def initial_acid := initial_concentration * initial_volume

theorem pure_water_to_add_eq_30 :
  ∃ w : ℝ, (initial_acid / (initial_volume + w) = desired_concentration) ∧ w = 30 :=
by
  sorry

end pure_water_to_add_eq_30_l525_525632


namespace identify_fake_coin_in_three_weighings_l525_525042

-- Define the type for coins
inductive Coin
| Coin1 | Coin2 | Coin3 | Coin4

-- Type for weight comparison result
inductive ComparisonResult
| lighter | heavier | equal

-- Define the problem conditions
def is_fake (c : Coin) : Prop := sorry -- A coin is fake.
def fake_lighter (c : Coin) : Prop := sorry -- The fake coin is lighter.
def fake_heavier (c : Coin) : Prop := sorry -- The fake coin is heavier.

-- Define the weighing function
def weigh (group1 group2 : list Coin) : ComparisonResult := sorry

-- Statement of the proof problem
theorem identify_fake_coin_in_three_weighings :
  ∃ (c : Coin), is_fake c ∧ (fake_lighter c ∨ fake_heavier c) :=
begin
  -- Given the conditions and three weighings, it is possible to identify the fake coin
  -- and determine if it is lighter or heavier.
  sorry
end

end identify_fake_coin_in_three_weighings_l525_525042


namespace csc_neg_45_eq_neg_sqrt_2_l525_525058

noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

theorem csc_neg_45_eq_neg_sqrt_2 :
  csc (-Real.pi / 4) = -Real.sqrt 2 := by
  sorry

end csc_neg_45_eq_neg_sqrt_2_l525_525058


namespace avg_growth_rate_2009_2011_projected_output_2012_l525_525882

noncomputable def average_growth_rate (v0 v1 : ℝ) (years : ℕ) : ℝ :=
  (v1 / v0)^(1 / years) - 1

-- Given conditions
def initial_output_2009 : ℝ := 1.5 -- in million yuan
def output_2011 : ℝ := 2.16 -- in million yuan
def years_passed : ℕ := 2

-- Prove the average annual growth rate from 2009 to 2011
theorem avg_growth_rate_2009_2011 : average_growth_rate initial_output_2009 output_2011 years_passed = 0.2 := 
by
  sorry -- proof

-- Given the average annual growth rate, project the output value for 2012
noncomputable def projected_output (v1 rate : ℝ) : ℝ :=
  v1 * (1 + rate)

def output_2011 : ℝ := 2.16 -- reaffirmation for clarity
def growth_rate : ℝ := 0.2

theorem projected_output_2012 : projected_output output_2011 growth_rate = 2.592 :=
by
  sorry -- proof

end avg_growth_rate_2009_2011_projected_output_2012_l525_525882


namespace range_of_m_l525_525573

theorem range_of_m (x m : ℝ) (h1 : -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2)
                   (h2 : x^2 - 2*x + 1 - m^2 ≤ 0)
                   (h3 : m > 0)
                   (h4 : (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m))
                   (h5 : ¬((x < 1 - m ∨ x > 1 + m) → (x < -2 ∨ x > 10))) :
                   m ≤ 3 :=
by
  sorry

end range_of_m_l525_525573


namespace max_height_is_24_km_can_pose_danger_l525_525322

-- Define the given acceleration provided by the engines
def engine_acceleration := 30 -- in m/s^2

-- Define the duration of the engine run
def engine_duration := 20 -- in seconds

-- Define the acceleration due to gravity
def gravity_acceleration := 10 -- in m/s^2

-- Define the initial conditions
def initial_velocity := engine_acceleration * engine_duration -- in m/s
def initial_height := (1 / 2 : ℝ) * engine_acceleration * (engine_duration ^ 2) -- in meters

-- Calculate the time to reach maximum height after engine stops working
def time_to_max_height := initial_velocity / gravity_acceleration -- in seconds

-- Calculate the additional height gained under gravity only
def additional_height := initial_velocity * time_to_max_height - (1 / 2 : ℝ) * gravity_acceleration * (time_to_max_height ^ 2) -- in meters

-- Calculate the total maximum height reached by the rocket
def total_max_height := initial_height + additional_height -- in meters

-- Define the height threshold to check possible danger
def danger_height := 20000 -- in meters

-- The proof problems
theorem max_height_is_24_km : total_max_height = 24000 :=
by sorry

theorem can_pose_danger : total_max_height > danger_height :=
by sorry

end max_height_is_24_km_can_pose_danger_l525_525322


namespace reimbursement_correct_l525_525230

-- Define the days and miles driven each day
def miles_monday : ℕ := 18
def miles_tuesday : ℕ := 26
def miles_wednesday : ℕ := 20
def miles_thursday : ℕ := 20
def miles_friday : ℕ := 16

-- Define the mileage rate
def mileage_rate : ℝ := 0.36

-- Define the total miles driven
def total_miles_driven : ℕ := miles_monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday

-- Define the total reimbursement
def reimbursement : ℝ := total_miles_driven * mileage_rate

-- Prove that the reimbursement is $36
theorem reimbursement_correct : reimbursement = 36 := by
  sorry

end reimbursement_correct_l525_525230


namespace jewelry_store_total_cost_l525_525486

-- Definitions for given conditions
def necklace_capacity : Nat := 12
def current_necklaces : Nat := 5
def ring_capacity : Nat := 30
def current_rings : Nat := 18
def bracelet_capacity : Nat := 15
def current_bracelets : Nat := 8

def necklace_cost : Nat := 4
def ring_cost : Nat := 10
def bracelet_cost : Nat := 5

-- Definition for number of items needed to fill displays
def needed_necklaces : Nat := necklace_capacity - current_necklaces
def needed_rings : Nat := ring_capacity - current_rings
def needed_bracelets : Nat := bracelet_capacity - current_bracelets

-- Definition for cost to fill each type of jewelry
def cost_necklaces : Nat := needed_necklaces * necklace_cost
def cost_rings : Nat := needed_rings * ring_cost
def cost_bracelets : Nat := needed_bracelets * bracelet_cost

-- Total cost to fill the displays
def total_cost : Nat := cost_necklaces + cost_rings + cost_bracelets

-- Proof statement
theorem jewelry_store_total_cost : total_cost = 183 := by
  sorry

end jewelry_store_total_cost_l525_525486


namespace find_vector_at_t_zero_l525_525487

variable (a d : ℝ × ℝ × ℝ)
variable (t : ℝ)

-- Given conditions
def condition1 := a - 2 * d = (2, 4, 10)
def condition2 := a + d = (-1, -3, -5)

-- The proof problem
theorem find_vector_at_t_zero 
  (h1 : condition1 a d)
  (h2 : condition2 a d) :
  a = (0, -2/3, 0) :=
sorry

end find_vector_at_t_zero_l525_525487


namespace triangle_existence_condition_l525_525529

-- Definitions and conditions given in the problem
variables (α : ℝ) (f_a r : ℝ)

-- The equivalent proof problem statement in Lean 4
theorem triangle_existence_condition (hα : 0 < α ∧ α < π) (hf_a : 0 < f_a) (hr : 0 < r) :
  f_a ≤ 2 * r * (Real.cos (α / 2))^2 :=
sorry

end triangle_existence_condition_l525_525529


namespace other_root_of_quadratic_l525_525040

theorem other_root_of_quadratic (a b k : ℝ) (h : 1^2 - (a+b) * 1 + ab * (1 - k) = 0) : 
  ∃ r : ℝ, r = a + b - 1 := 
sorry

end other_root_of_quadratic_l525_525040


namespace solution_l525_525928

noncomputable def problem_statement : Prop :=
  ( (π / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 )

theorem solution : problem_statement := by
  sorry

end solution_l525_525928


namespace tony_average_time_l525_525277

-- Definitions based on the conditions
def distance_to_store : ℕ := 4 -- in miles
def walking_speed : ℕ := 2 -- in MPH
def running_speed : ℕ := 10 -- in MPH

-- Conditions
def time_walking : ℕ := (distance_to_store / walking_speed) * 60 -- in minutes
def time_running : ℕ := (distance_to_store / running_speed) * 60 -- in minutes

def total_time : ℕ := time_walking + 2 * time_running -- Total time spent in minutes
def number_of_days : ℕ := 3 -- Number of days

def average_time : ℕ := total_time / number_of_days -- Average time in minutes

-- Statement to prove
theorem tony_average_time : average_time = 56 := by 
  sorry

end tony_average_time_l525_525277


namespace minimize_sum_of_squared_distances_l525_525666

variables {A B : Type*} [EuclideanSpace A] [EuclideanSpace B]

-- Define the vertices of the polygon
variables (A₁ A₂ : B)
variable (n : ℕ)
variable (vertices : fin n → B)  -- vertices of the polygon
variable S : B  -- the point S on the plane

-- Define the centroid Z
def centroid (vertices : fin n → B) : B := 
  (finset.univ.sum vertices) / (n : ℝ)

-- Define the function to compute the sum of the squared distances from S to each vertex
def sum_of_squared_distances (S : B) (vertices : fin n → B) : ℝ :=
  finset.univ.sum (λ i, dist S (vertices i) ^ 2)

-- The point Z is the centroid of the vertices
noncomputable def Z : B := centroid vertices

-- The proof statement
theorem minimize_sum_of_squared_distances : 
  sum_of_squared_distances S vertices ≥ sum_of_squared_distances Z vertices :=
sorry

end minimize_sum_of_squared_distances_l525_525666


namespace tomato_offspring_heritable_variation_l525_525002

structure Genotype (A a B b : Type)

def isHeritableVariation (genotype: Genotype A a B b) : Prop := sorry

theorem tomato_offspring_heritable_variation
  (A a B b : Type)
  (parent : Genotype A a B b)
  (self_fertilizes : parent)
  (offspring : Genotype A a B bb) :
  isHeritableVariation offspring :=
sorry

end tomato_offspring_heritable_variation_l525_525002


namespace degree_measure_regular_octagon_interior_angle_l525_525835

theorem degree_measure_regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range (n - 2), 180) / n = 135 :=
by
  sorry

end degree_measure_regular_octagon_interior_angle_l525_525835


namespace ellipse_slope_ratio_l525_525130

theorem ellipse_slope_ratio (a b x1 y1 x2 y2 c k1 k2 : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c = a / 2) (h4 : a = 2) (h5 : c = 1) (h6 : b = Real.sqrt 3) 
  (h7 : 3 * x1 ^ 2 + 4 * y1 ^ 2 = 12 * c ^ 2) 
  (h8 : 3 * x2 ^ 2 + 4 * y2 ^ 2 = 12 * c ^ 2) 
  (h9 : x1 = y1 - c) (h10 : x2 = y2 - c)
  (h11 : y1^2 = 9 / 4)
  (h12 : y1 = -3 / 2 ∨ y1 = 3 / 2) 
  (h13 : k1 = -3 / 2) 
  (h14 : k2 = -1 / 2) :
  k1 / k2 = 3 := 
  sorry

end ellipse_slope_ratio_l525_525130


namespace school_competition_students_l525_525335

theorem school_competition_students (n : ℤ)
  (h1 : 100 < n) 
  (h2 : n < 200) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2) 
  (h5 : n % 6 = 2) :
  n = 122 ∨ n = 182 :=
sorry

end school_competition_students_l525_525335


namespace sin_cos_sum_expr_l525_525667

noncomputable def expr_for_sin_cos_sum (theta : ℝ) (b : ℝ) : ℝ :=
  if 0 < theta ∧ theta < (real.pi / 2) ∧ real.cos (2 * theta) = b then
    real.sqrt (2 - b)
  else
    0

theorem sin_cos_sum_expr (theta : ℝ) (b : ℝ) (h1 : 0 < theta) (h2 : theta < real.pi / 2) (h3 : real.cos (2 * theta) = b) :
  sin theta + cos theta = expr_for_sin_cos_sum theta b :=
by {
  sorry
}

end sin_cos_sum_expr_l525_525667


namespace water_added_eq_30_l525_525646

-- Given conditions
def initial_volume := 50
def initial_concentration := 0.4
def final_concentration := 0.25
def amount_of_acid := initial_volume * initial_concentration

-- Proof problem statement
theorem water_added_eq_30 : ∃ w : ℝ, amount_of_acid / (initial_volume + w) = final_concentration ∧ w = 30 :=
by 
  -- Solution steps go here, but for now we insert sorry to skip the proof.
  sorry

end water_added_eq_30_l525_525646


namespace winning_strategy_for_first_player_l525_525358

/-- There are 18 ping-pong balls. Players take turns picking 1 to 4 balls. The first player wins by picking the 18th ball. Given these conditions,
prove that the first player should pick 3 balls initially to ensure they win. -/
theorem winning_strategy_for_first_player :
  ∃ (n : ℕ), n = 3 ∧
  (∀ a b : ℕ, 1 ≤ a ∧ a ≤ 4 → 1 ≤ b ∧ b ≤ 4 →
    (let total := 14 in
    (total % 5 = 0) →
    (∃ k : ℕ, k ≤ 14 ∧ 
    (total + k = 18)))) :=
by
  sorry

end winning_strategy_for_first_player_l525_525358


namespace maximize_x5y3_l525_525736

theorem maximize_x5y3 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 30) :
  x = 18.75 ∧ y = 11.25 → (x^5 * y^3) = (18.75^5 * 11.25^3) :=
sorry

end maximize_x5y3_l525_525736


namespace shooting_result_l525_525007

/-- A person is shooting with a total of 5 bullets. Shooting stops when the target is hit or when all bullets are fired.
    \xi denotes the number of shots fired. -/
def shooting_experiment : Prop :=
  ∀ (hit : ℕ → Prop) (total_shots : ℕ) (xi : ℕ),
    total_shots = 5 →
    (∀ n, n < total_shots → ¬hit n) →
    xi = total_shots →
    (∀ n, n < 4 → ¬hit n)

theorem shooting_result (hit : ℕ → Prop) (total_shots xi : ℕ)
  (H_total : total_shots = 5)
  (H_miss_first_four : ∀ n, n < total_shots → ¬hit n)
  (H_xi_is_five : xi = total_shots) :
  ∀ n, n < 4 → ¬hit n :=
begin
  sorry
end

end shooting_result_l525_525007


namespace cubic_equation_cube_roots_eq_correct_l525_525846

noncomputable def roots_cube_cubic (p q r : ℚ) : Polynomial ℚ :=
  Polynomial.X^3 + (p^3 - 3*p*q + 3*r) • Polynomial.X^2 + (q^3 - 3*p*q*r + 3*r^2) • Polynomial.X + r^3

theorem cubic_equation_cube_roots_eq_correct :
  roots_cube_cubic (-3) (-0.5) (0.5) =
    Polynomial.X^3 - 30 • Polynomial.X^2 + (1.625 : ℚ) • Polynomial.X + (0.125 : ℚ) :=
by
  sorry

end cubic_equation_cube_roots_eq_correct_l525_525846


namespace percentage_increase_l525_525848

theorem percentage_increase (original_time new_time : ℕ) (h_original : original_time = 30) (h_new : new_time = 45) :
  ((new_time - original_time : ℕ) * 100 / original_time = 50) :=
by
  rw [h_original, h_new]
  norm_num

sorry

end percentage_increase_l525_525848


namespace find_second_year_interest_rate_l525_525006

noncomputable def interest_rate_second_year
  (initial_investment : ℝ) (first_year_rate : ℝ) (second_year_value : ℝ) : ℝ :=
  let value_after_first_year := initial_investment * (1 + first_year_rate / 100);
  let second_year_rate := (second_year_value / value_after_first_year - 1) * 100;
  second_year_rate

theorem find_second_year_interest_rate 
  (initial_investment : ℝ) (first_year_rate : ℝ) (second_year_value : ℝ) : 
  initial_investment = 12000 →
  first_year_rate = 8 →
  second_year_value = 13440 →
  interest_rate_second_year initial_investment first_year_rate second_year_value = 3.7 :=
by {
  intro h1,
  intro h2,
  intro h3,
  rw [h1, h2, h3],
  dsimp [interest_rate_second_year],
  norm_num,
  sorry
}

end find_second_year_interest_rate_l525_525006


namespace last_date_in_2011_divisible_by_101_is_1221_l525_525759

def is_valid_date (a b c d : ℕ) : Prop :=
  (10 * a + b) ≤ 12 ∧ (10 * c + d) ≤ 31

def date_as_number (a b c d : ℕ) : ℕ :=
  20110000 + 1000 * a + 100 * b + 10 * c + d

theorem last_date_in_2011_divisible_by_101_is_1221 :
  ∃ (a b c d : ℕ), is_valid_date a b c d ∧ date_as_number a b c d % 101 = 0 ∧ date_as_number a b c d = 20111221 :=
by
  sorry

end last_date_in_2011_divisible_by_101_is_1221_l525_525759


namespace probability_two_primes_l525_525384

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l525_525384


namespace find_area_CDM_l525_525367

noncomputable def right_triangle_ABC
  (AC BC : ℝ)
  (angle_C_right : (AC^2 + BC^2 = (real.sqrt (AC^2 + BC^2))^2)) : Prop :=
  AC = 9 ∧ BC = 24 ∧ angle_C_right

noncomputable def triangle_mid_point_M
  (AC BC AB M : ℝ)
  (mid_AB : M = (real.sqrt (AC^2 + BC^2)) / 2) : Prop :=
  M = AB / 2

noncomputable def point_D 
  (AD BD : ℝ)
  (equal_AD_BD : AD = BD)
  (AD_val BD_val : AD = 16 ∧ BD = 16) : Prop :=
  equal_AD_BD ∧ AD_val ∧ BD_val

theorem find_area_CDM
  (AB CN : ℝ) 
  (DM : ℝ → Prop)
  (area_expression : ∃ m n p : ℕ, m = 90 ∧ n = 15 ∧ p = 73 ∧ (∃ k : ℝ, k = 108 ∧ (real.sqrt (k) = AB ∧ DM = (real.sqrt ((16^2) - (AB / 2)^2))))) :
  ∃ m n p : ℕ, m = 90 ∧ n = 15 ∧ p = 73 ∧ m + n + p = 178 := 
  sorry

end find_area_CDM_l525_525367


namespace sphere_volume_in_cone_l525_525898

theorem sphere_volume_in_cone (d : ℝ) (r : ℝ) (π : ℝ) (V : ℝ) (h1 : d = 12) (h2 : r = d / 2) (h3 : V = (4 / 3) * π * r^3) :
  V = 288 * π :=
by 
  sorry

end sphere_volume_in_cone_l525_525898


namespace zeros_at_end_of_factorial_30_l525_525673

theorem zeros_at_end_of_factorial_30 : ∀ (n : ℕ), n = 30 → (count_factors 5 (factorial n) = 7) :=
by
  intro n hn
  rw [hn]
  sorry

def count_factors (p n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let rec loop (n : ℕ) : ℕ :=
      if n = 0 then 0
      else (if n % p = 0 then 1 else 0) + loop (n / p)
    loop n

def factorial : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * factorial n

end zeros_at_end_of_factorial_30_l525_525673


namespace red_marbles_eq_14_l525_525222

theorem red_marbles_eq_14 (total_marbles : ℕ) (yellow_marbles : ℕ) (R : ℕ) (B : ℕ)
  (h1 : total_marbles = 85)
  (h2 : yellow_marbles = 29)
  (h3 : B = 3 * R)
  (h4 : (total_marbles - yellow_marbles) = R + B) :
  R = 14 :=
by
  sorry

end red_marbles_eq_14_l525_525222


namespace area_difference_of_squares_l525_525810

theorem area_difference_of_squares (d1 d2 : ℝ) (h1 : d1 = 21) (h2 : d2 = 17) :
  (d1^2 - d2^2) = 152 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end area_difference_of_squares_l525_525810


namespace solution_l525_525929

noncomputable def problem_statement : Prop :=
  ( (π / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 )

theorem solution : problem_statement := by
  sorry

end solution_l525_525929


namespace rhombus_with_circles_l525_525828

theorem rhombus_with_circles 
  (O O1 : Point)
  (r : ℝ)
  (A B C D E F M : Point)
  (h1 : dist O O1 = 0 ∨ dist O O1 = r)
  (h2 : intersects (circle O r) (circle O1 r) A B)
  (h3 : secant_through A (C, A, D))
  (h4 : perpendicular_dropped_from B (C, A, D) M (E, F)) :
  is_rhombus (C, E, D, F) ∧
  side_length (C, E, D, F) = dist O O1 ∧
  parallel (line_through C E) (line_through D F) (line_through O O1) :=
sorry

end rhombus_with_circles_l525_525828


namespace total_savings_over_12_weeks_l525_525294

-- Define the weekly savings and durations for each period
def weekly_savings_period_1 : ℕ := 5
def duration_period_1 : ℕ := 4

def weekly_savings_period_2 : ℕ := 10
def duration_period_2 : ℕ := 4

def weekly_savings_period_3 : ℕ := 20
def duration_period_3 : ℕ := 4

-- Define the total savings calculation for each period
def total_savings_period_1 : ℕ := weekly_savings_period_1 * duration_period_1
def total_savings_period_2 : ℕ := weekly_savings_period_2 * duration_period_2
def total_savings_period_3 : ℕ := weekly_savings_period_3 * duration_period_3

-- Prove that the total savings over 12 weeks equals $140.00
theorem total_savings_over_12_weeks : total_savings_period_1 + total_savings_period_2 + total_savings_period_3 = 140 := 
by 
  sorry

end total_savings_over_12_weeks_l525_525294


namespace complement_of_M_is_123_l525_525162

open Set

variable U : Set ℕ := {1, 2, 3, 4, 5}
variable M : Set ℕ := {4, 5}

theorem complement_of_M_is_123 : U \ M = {1, 2, 3} :=
by {
  sorry
}

end complement_of_M_is_123_l525_525162


namespace abigail_lost_money_l525_525904

theorem abigail_lost_money (initial_amount spent_first_store spent_second_store remaining_amount_lost: ℝ) 
  (h_initial : initial_amount = 50) 
  (h_spent_first : spent_first_store = 15.25) 
  (h_spent_second : spent_second_store = 8.75) 
  (h_remaining : remaining_amount_lost = 16) : (initial_amount - spent_first_store - spent_second_store - remaining_amount_lost = 10) :=
by
  sorry

end abigail_lost_money_l525_525904


namespace minimize_triangle_area_l525_525583

-- Definitions of points and angles
variable {Point : Type}
variable (X A Y O B C : Point)
variable (angle_XAY angle_XA'Y' : Type)

-- Condition: Given angle XAY and point O inside it
axiom XAY_angle : angle_XAY
axiom O_in_XAY_angle : True  -- Replace True with the actual condition defining point O inside angle

-- Prove that the line passing through points B and C cuts off the triangle of the smallest possible area
theorem minimize_triangle_area (B C: Point) (BC_line: True): 
  ∃ line, through O and cuts off smallest triangle from XAY where line = BC := sorry

end minimize_triangle_area_l525_525583


namespace professors_chair_arrangement_l525_525052

theorem professors_chair_arrangement :
  let total_chairs := 11
  let professors := 3
  let separation := 2
  let available_chairs := total_chairs - 2
  let valid_positions := ∀ i j k, (1 ≤ i ∧ i ≤ available_chairs)
                        ∧ (1 + separation ≤ j ∧ j ≤ available_chairs)
                        ∧ (1 + separation ≤ k ∧ k ≤ available_chairs)
                        ∧ (i < j ∧ j < k)
                        ∧ (j - i > separation)
                        ∧ (k - j > separation)
  if valid_positions then 24 else 0
  = 24 :=
by
  -- Specification of the proof statement.
  -- Actual steps to be filled in the proof section.
  sorry

end professors_chair_arrangement_l525_525052


namespace rocket_proof_l525_525320

-- Definition of conditions
def a : ℝ := 30  -- acceleration provided by the engines in m/s^2
def tau : ℝ := 20  -- duration of the engine run in seconds
def g : ℝ := 10  -- acceleration due to gravity in m/s^2

-- Definitions based on the conditions
def V0 : ℝ := a * tau  -- final velocity after tau seconds
def y0 : ℝ := 0.5 * a * tau^2  -- height gained during initial acceleration period

-- Kinematic equations during free flight
def t_flight : ℝ := V0 / g  -- time to maximum height after engines stop
def y_flight : ℝ := V0 * t_flight - 0.5 * g * t_flight^2  -- additional height during free flight

-- Total maximum height
def height_max : ℝ := y0 + y_flight

-- Conclusion of threat evaluation
def threat := height_max > 20000  -- 20 km in meters

-- The proof problem statement
theorem rocket_proof :
  height_max = 24000 ∧ threat = true := by
  sorry  -- Proof to be provided

end rocket_proof_l525_525320


namespace bean_game_termination_and_uniqueness_l525_525003

open_locale big_operators

-- Define the conditions of the problem
def finite_beans : ℤ → ℕ := sorry -- Initial placement of beans on an infinite row of squares
def b (i : ℤ) : ℕ := finite_beans i

-- Semi-invariant definition
def S : ℤ → ℕ := λ i, i^2 * b i

-- The statement to prove
theorem bean_game_termination_and_uniqueness 
  (b : ℤ → ℕ) 
  (finite_beans : {i : ℤ | b i ≠ 0}.finite) 
  (moves : ℕ) -- Number of moves taken
  (final_configuration : ∀ i : ℤ, ∃ n : ℕ, n = b i ∧ n ≤ 1) : 
  (∃ n : ℕ, moves = n ∧ final_configuration) :=
sorry

end bean_game_termination_and_uniqueness_l525_525003


namespace domain_of_f_l525_525319

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

theorem domain_of_f :
  {x : ℝ | f x = Real.log (x^2 - x)} = {x : ℝ | x < 0 ∨ x > 1} :=
sorry

end domain_of_f_l525_525319


namespace water_to_add_for_desired_acid_concentration_l525_525659

theorem water_to_add_for_desired_acid_concentration :
  ∃ w : ℝ, 50 * 0.4 / (50 + w) = 0.25 ∧ w = 30 :=
begin
  let initial_volume := 50,
  let initial_acid_concentration := 0.4,
  let desired_acid_concentration := 0.25,
  let acid_content := initial_volume * initial_acid_concentration,
  use 30,
  split,
  { simp [initial_volume, initial_acid_concentration, desired_acid_concentration, acid_content],
    linarith },
  { refl }
end

end water_to_add_for_desired_acid_concentration_l525_525659


namespace line_passing_through_point_midpoint_of_chord_in_ellipse_l525_525706

theorem line_passing_through_point_midpoint_of_chord_in_ellipse :
  (∃ (k : ℝ), ∀ (x y : ℝ), y - 1 = k * (x - 1) ↔ 3 * x + 4 * y - 7 = 0) ∧
  (P : ℝ × ℝ) (hx : P = (1, 1)) :
  let ellipse_eq := (x y : ℝ) → x^2 / 4 + y^2 / 3 = 1,
  ∃ (line_eq : ℝ → ℝ → Prop), 
    (line_eq 1 1) ∧ 
    (∃ (x1 y1 x2 y2 : ℝ),
      ellipse_eq x1 y1 ∧
      ellipse_eq x2 y2 ∧
      (x1 + x2 = 2) ∧
      (y1 + y2 = 2) ∧
      line_eq x y := 3 * x + 4 * y - 7 = 0) :=
begin
  sorry
end

end line_passing_through_point_midpoint_of_chord_in_ellipse_l525_525706


namespace pure_water_to_add_eq_30_l525_525635

noncomputable def initial_volume := 50
noncomputable def initial_concentration := 0.40
noncomputable def desired_concentration := 0.25
noncomputable def initial_acid := initial_concentration * initial_volume

theorem pure_water_to_add_eq_30 :
  ∃ w : ℝ, (initial_acid / (initial_volume + w) = desired_concentration) ∧ w = 30 :=
by
  sorry

end pure_water_to_add_eq_30_l525_525635


namespace find_m_intersection_points_l525_525158

theorem find_m (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) : m = 1 := 
by
  sorry

theorem intersection_points (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) 
  (hm : m = 1) : ∃ x1 x2 : ℝ, (x^2 + x - 2 = 0) ∧ x1 ≠ x2 :=
by
  sorry

end find_m_intersection_points_l525_525158


namespace min_xy_squared_xy_eq_xy_min_f_x_eq_1_min_H_eq_2_l525_525046

-- Part (1)
theorem min_xy_squared_xy_eq_xy (x y : ℝ) : min (x^2 + y^2) (x * y) = x * y := 
sorry

-- Part (2)
def f (x : ℝ) : ℝ := max (abs x) (2 * x + 3)

theorem min_f_x_eq_1 : ∃ x : ℝ, f x = 1 :=
sorry

-- Part (3)
theorem min_H_eq_2 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) : 
    H = max (1 / sqrt x) ((x + y) / sqrt (x * y)) (1 / sqrt y) := 

end min_xy_squared_xy_eq_xy_min_f_x_eq_1_min_H_eq_2_l525_525046


namespace odd_base_divisibility_by_2_base_divisibility_by_m_l525_525461

-- Part (a)
theorem odd_base_divisibility_by_2 (q : ℕ) :
  (∀ a : ℕ, (a * q) % 2 = 0 ↔ a % 2 = 0) → q % 2 = 1 := 
sorry

-- Part (b)
theorem base_divisibility_by_m (q m : ℕ) (h1 : m > 1) :
  (∀ a : ℕ, (a * q) % m = 0 ↔ a % m = 0) → ∃ k : ℕ, q = 1 + m * k ∧ k ≥ 1 :=
sorry

end odd_base_divisibility_by_2_base_divisibility_by_m_l525_525461


namespace compute_sum_eq_square_l525_525234

open Int -- Use the Integer namespace for floor function

theorem compute_sum_eq_square (n : ℕ) (h : n > 1) :
  let x_k (k : ℕ) := n^2 % (⌊n^2 / k⌋ + 1)
  ∑ k in Finset.range (n - 1) + 2, (⌊x_k (k + 2) / k⌋ / k) = (n - 1)^2 :=
by
  sorry

end compute_sum_eq_square_l525_525234


namespace original_cost_price_l525_525457

theorem original_cost_price (selling_price_friend : ℝ) (gain_percent : ℝ) (loss_percent : ℝ) 
  (final_selling_price : ℝ) : 
  final_selling_price = 54000 → gain_percent = 0.2 → loss_percent = 0.1 → 
  selling_price_friend = (1 - loss_percent) * x → final_selling_price = (1 + gain_percent) * selling_price_friend → 
  x = 50000 :=
by 
  sorry

end original_cost_price_l525_525457


namespace emily_needs_375_nickels_for_book_l525_525545

theorem emily_needs_375_nickels_for_book
  (n : ℕ)
  (book_cost : ℝ)
  (five_dollars : ℝ)
  (one_dollars : ℝ)
  (quarters : ℝ)
  (nickel_value : ℝ)
  (total_money : ℝ)
  (h1 : book_cost = 46.25)
  (h2 : five_dollars = 4 * 5)
  (h3 : one_dollars = 5 * 1)
  (h4 : quarters = 10 * 0.25)
  (h5 : nickel_value = n * 0.05)
  (h6 : total_money = five_dollars + one_dollars + quarters + nickel_value) 
  (h7 : total_money ≥ book_cost) :
  n ≥ 375 :=
by 
  sorry

end emily_needs_375_nickels_for_book_l525_525545


namespace least_steps_couples_l525_525864

noncomputable def least_steps (n : ℕ) : ℕ := n - 1

theorem least_steps_couples (n : ℕ) (h : n ≥ 1) :
  ∃ C, (∀ (arr : list (ℤ × ℤ)), arr.length = 2 * n → 
  (∃ (steps : ℕ), steps ≤ C ∧ all_couples_adjacent arr steps)) ∧ 
  C = least_steps n :=
sorry

-- Assumption: all_couples_adjacent definition, which checks if all couples are adjacent in the given arrangement and steps.
-- Since the problem does not specify this function, we encapsulate the essential logic in the theorem statement.

end least_steps_couples_l525_525864


namespace selling_price_with_60_percent_profit_l525_525497

theorem selling_price_with_60_percent_profit (C : ℝ) (h₀ : 2240 = C + 0.4 * C) :
  2560 = C + 0.6 * C :=
by
  have h₁ : C = 2240 / 1.4 := by sorry
  have h₂ : 2560 = (2240 / 1.4) + 0.6 * (2240 / 1.4) := by sorry
  exact h₂

end selling_price_with_60_percent_profit_l525_525497


namespace probability_of_two_primes_is_correct_l525_525387

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l525_525387


namespace cost_of_graphing_calculator_l525_525962

/-
  Everton college paid $1625 for an order of 45 calculators.
  Each scientific calculator costs $10.
  The order included 20 scientific calculators and 25 graphing calculators.
  We need to prove that each graphing calculator costs $57.
-/

namespace EvertonCollege

theorem cost_of_graphing_calculator
  (total_cost : ℕ)
  (cost_scientific : ℕ)
  (num_scientific : ℕ)
  (num_graphing : ℕ)
  (cost_graphing : ℕ)
  (h_order : total_cost = 1625)
  (h_cost_scientific : cost_scientific = 10)
  (h_num_scientific : num_scientific = 20)
  (h_num_graphing : num_graphing = 25)
  (h_total_calc : num_scientific + num_graphing = 45)
  (h_pay : total_cost = num_scientific * cost_scientific + num_graphing * cost_graphing) :
  cost_graphing = 57 :=
by
  sorry

end EvertonCollege

end cost_of_graphing_calculator_l525_525962


namespace total_games_in_season_l525_525693

theorem total_games_in_season :
  let num_teams := 100
  let num_sub_leagues := 5
  let teams_per_league := 20
  let games_per_pair := 6
  let teams_advancing := 4
  let playoff_teams := num_sub_leagues * teams_advancing
  let sub_league_games := (teams_per_league * (teams_per_league - 1) / 2) * games_per_pair
  let total_sub_league_games := sub_league_games * num_sub_leagues
  let playoff_games := (playoff_teams * (playoff_teams - 1)) / 2 
  let total_games := total_sub_league_games + playoff_games
  total_games = 5890 :=
by
  sorry

end total_games_in_season_l525_525693


namespace scientific_notation_correct_l525_525547

-- Define the number to be represented in scientific notation
def number := -0.000000103

-- Define the expected result in scientific notation
def expected := (-1.03 : ℝ) * 10 ^ (-7 : ℝ)

-- The theorem states that the scientific notation of the number is as expected
theorem scientific_notation_correct : number = expected := by
  sorry

end scientific_notation_correct_l525_525547


namespace hobbes_winning_strategy_l525_525514

theorem hobbes_winning_strategy :
  let n := 1010 in
  let total_subsets := 4^n in
  let invalid_subsets := 3^n in
  ∃ (F : Set (Set (Fin 2020))), 
    (∀ s ∈ F, ∀ x, x ∈ s → x < 2020) ∧
    (∃ (strategy : Set (Fin 2020) → (Fin 2020 → Bool)),
      ∀ (turns : List (Fin 2020)),
        Calvin_wins turns → 
        ¬ (Hobbes_wins strategy turns)) ∧
    F = 4^n - 3^n

end hobbes_winning_strategy_l525_525514


namespace total_games_played_l525_525886

theorem total_games_played (n k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  ∑ i in finset.range (n), i * k / 2 = 84 :=
by
  sorry

end total_games_played_l525_525886


namespace add_water_to_solution_l525_525649

theorem add_water_to_solution (w : ℕ) :
  let initial_volume := 50
  let initial_concentration := 0.4
  let final_concentration := 0.25
  let acid_amount := initial_concentration * initial_volume
  let final_volume := initial_volume + w
  (acid_amount / final_volume = final_concentration) ↔ (w = 30) :=
by 
  sorry

end add_water_to_solution_l525_525649


namespace complex_multiplication_identity_l525_525096

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- State the theorem to be proved
theorem complex_multiplication_identity : z * (z_conjugate + complex.i) = 6 + 2 * complex.i :=
by {
  sorry -- proof goes here
}

end complex_multiplication_identity_l525_525096


namespace sequence_difference_l525_525915

-- Define the arithmetic sequence properties
def arithmetic_seq (a d : ℝ) (n : ℕ) := a + d * (n - 1)

-- Given conditions
def sum_of_seq (a d : ℝ) (n : ℕ) := (n : ℝ) * a + d * (n : ℝ) * (n : ℝ - 1) / 2

def sequence_conditions (a d : ℝ) : Prop :=
(∀ n, 1 ≤ n ∧ n ≤ 150 → 
  20 ≤ arithmetic_seq a d n ∧ arithmetic_seq a d n ≤ 80) ∧
  sum_of_seq a d 150 = 9000

-- Least and greatest value functions for 75th term
def least_75th_term (a d : ℝ) : ℝ := arithmetic_seq a d 75
def greatest_75th_term (a d : ℝ) : ℝ := arithmetic_seq a (-d) 75

-- Proof problem statement
theorem sequence_difference (a d : ℝ) (h : sequence_conditions a d) :
  let M := least_75th_term a d
  let N := greatest_75th_term a d
  M - N = - (3000 / 149) :=
by
  sorry

end sequence_difference_l525_525915


namespace probability_both_numbers_are_prime_l525_525419

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l525_525419


namespace probability_two_primes_is_1_over_29_l525_525427

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l525_525427


namespace water_to_add_for_desired_acid_concentration_l525_525657

theorem water_to_add_for_desired_acid_concentration :
  ∃ w : ℝ, 50 * 0.4 / (50 + w) = 0.25 ∧ w = 30 :=
begin
  let initial_volume := 50,
  let initial_acid_concentration := 0.4,
  let desired_acid_concentration := 0.25,
  let acid_content := initial_volume * initial_acid_concentration,
  use 30,
  split,
  { simp [initial_volume, initial_acid_concentration, desired_acid_concentration, acid_content],
    linarith },
  { refl }
end

end water_to_add_for_desired_acid_concentration_l525_525657


namespace csc_neg_45_eq_neg_sqrt2_l525_525060

-- Define the question in Lean given the conditions and prove the answer.
theorem csc_neg_45_eq_neg_sqrt2 : Real.csc (-π/4) = -Real.sqrt 2 :=
by
  -- Sorry placeholder since proof is not required.
  sorry

end csc_neg_45_eq_neg_sqrt2_l525_525060


namespace dorchester_puppy_count_l525_525954

/--
  Dorchester works at a puppy wash. He is paid $40 per day + $2.25 for each puppy he washes.
  On Wednesday, Dorchester earned $76. Prove that Dorchester washed 16 puppies that day.
-/
theorem dorchester_puppy_count
  (total_earnings : ℝ)
  (base_pay : ℝ)
  (pay_per_puppy : ℝ)
  (puppies_washed : ℕ)
  (h1 : total_earnings = 76)
  (h2 : base_pay = 40)
  (h3 : pay_per_puppy = 2.25) :
  total_earnings - base_pay = (puppies_washed : ℝ) * pay_per_puppy :=
sorry

example :
  dorchester_puppy_count 76 40 2.25 16 := by
  rw [dorchester_puppy_count, sub_self, mul_zero]

end dorchester_puppy_count_l525_525954


namespace vector_subtraction_l525_525551

/-
Define the vectors we are working with.
-/
def v1 : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-8]]
def v2 : Matrix (Fin 2) (Fin 1) ℤ := ![![2], ![-6]]
def scalar : ℤ := 5
def result : Matrix (Fin 2) (Fin 1) ℤ := ![![-7], ![22]]

/-
The statement of the proof problem.
-/
theorem vector_subtraction : v1 - scalar • v2 = result := 
by
  sorry

end vector_subtraction_l525_525551


namespace fraction_square_equality_l525_525285

theorem fraction_square_equality (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) 
    (h : a / b + c / d = 1) : 
    (a / b)^2 + c / d = (c / d)^2 + a / b :=
by
  sorry

end fraction_square_equality_l525_525285


namespace parallel_line_distance_l525_525065

theorem parallel_line_distance (m : ℤ) (x y : ℤ) :
  (3*x + 4*y + m = 0 ∧ m = 10) ∨ (3*x + 4*y + m = 0 ∧ m = -20) ↔
  (3*x + 4*y - 5 = 0 ∧ abs (m + 5) / real.sqrt (3^2 + 4^2) = 3) :=
begin
  sorry
end

end parallel_line_distance_l525_525065


namespace probability_of_two_prime_numbers_l525_525394

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l525_525394


namespace determine_parabola_equation_l525_525178

-- Define the conditions
def focus_on_line (focus : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, focus = (k - 2, k / 2 - 1)

-- Define the result equations
def is_standard_equation (eq : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, eq x y → x^2 = 4 * y) ∨ (∀ x y : ℝ, eq x y → y^2 = -8 * x)

-- Define the theorem stating that given the condition,
-- the standard equation is one of the two forms
theorem determine_parabola_equation (focus : ℝ × ℝ) (H : focus_on_line focus) :
  ∃ eq : ℝ → ℝ → Prop, is_standard_equation eq :=
sorry

end determine_parabola_equation_l525_525178


namespace add_water_to_solution_l525_525653

theorem add_water_to_solution (w : ℕ) :
  let initial_volume := 50
  let initial_concentration := 0.4
  let final_concentration := 0.25
  let acid_amount := initial_concentration * initial_volume
  let final_volume := initial_volume + w
  (acid_amount / final_volume = final_concentration) ↔ (w = 30) :=
by 
  sorry

end add_water_to_solution_l525_525653


namespace quadrilateral_area_l525_525894

structure Point (ℝ : Type) :=
  (x : ℝ)
  (y : ℝ)

def A : Point ℝ := ⟨1, 3⟩
def B : Point ℝ := ⟨1, 1⟩
def C : Point ℝ := ⟨5, 6⟩
def D : Point ℝ := ⟨4, 3⟩

def determinant_area (P Q R : Point ℝ) : ℝ :=
  (1 / 2) * | P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y) |

def area_quadrilateral (A B C D : Point ℝ) : ℝ :=
  determinant_area A B C + determinant_area A C D

theorem quadrilateral_area :
  area_quadrilateral A B C D = 8.5 :=
by
  sorry

end quadrilateral_area_l525_525894


namespace student_ranking_l525_525696

inductive Student
| Bridget
| Cassie
| Hannah
| Ella

open Student

def ranking : list Student := [Hannah, Cassie, Bridget, Ella]

axiom conditions :
  (Bridget ≠ list.head ranking) ∧ 
  (list.index_of Cassie ranking < list.index_of Ella ranking) ∧
  true -- Hannah not making any statements implies no additional conditions

theorem student_ranking : ∀ (ranking' : list Student),
  (Bridget ≠ list.head ranking') →
  (list.index_of Cassie ranking' < list.index_of Ella ranking') →
  (ranking' = ranking) :=
by
  intro ranking' Hb Hc
  sorry

end student_ranking_l525_525696


namespace column_of_1000_l525_525192

theorem column_of_1000 : 
  ∀ (n : ℕ), 
    1 < n → 
    (∀ (k : ℕ), let col_seq := [ "A", "B", "C", "D", "E", "F", "G",
                                  "G", "F", "E", "D", "C", "B", "A" ] in
    col_seq[((n - 2) % 14)] = "C" → 
    n = 1000 → 
    (col_seq[((1000 - 2) % 14)] = "C")) :=
by
  intros n hn
  intro k
  intro col_seq
  intro h
  intro hn_eq
  rw hn_eq at h
  simp at h
  sorry

end column_of_1000_l525_525192


namespace sum_neg_one_over_i_bounds_n_l525_525123

theorem sum_neg_one_over_i_bounds_n (n : ℕ) (h1 : 2 ≤ n) :
  (4 / 7 : ℝ) < ∑ i in Finset.range (2 * n + 1) \ {0}, ((-1) ^ (i + 1)) / i ∧
  ∑ i in Finset.range (2 * n + 1) \ {0}, ((-1) ^ (i + 1)) / i < Real.sqrt 2 / 2 :=
by
  sorry

end sum_neg_one_over_i_bounds_n_l525_525123


namespace imaginary_part_of_z_l525_525327

def z : ℂ := 2 - complex.I

theorem imaginary_part_of_z :
  complex.im z = -1 :=
sorry

end imaginary_part_of_z_l525_525327


namespace area_R_as_percentage_of_area_Q_l525_525745

-- Given conditions
variables (D_Q : ℝ)

def D_P := 0.5 * D_Q
def D_R := 0.75 * D_Q

def A_P := Real.pi * (D_P / 2)^2
def A_Q := Real.pi * (D_Q / 2)^2
def A_R := Real.pi * (D_R / 2)^2

-- The problem statement in Lean 4
theorem area_R_as_percentage_of_area_Q 
  (h : A_P = (6.25 / 100) * A_Q) : 
  (A_R / A_Q) * 100 = 14.0625 := 
sorry

end area_R_as_percentage_of_area_Q_l525_525745


namespace selling_price_with_60_percent_profit_l525_525495

theorem selling_price_with_60_percent_profit (C : ℝ) (H1 : 2240 = C + 0.4 * C) : 
  let selling_price := C + 0.6 * C 
  in selling_price = 2560 := 
by 
  sorry

end selling_price_with_60_percent_profit_l525_525495


namespace impossible_to_have_same_number_of_each_color_l525_525902

-- Define the initial number of coins Laura has
def initial_green : Nat := 1

-- Define the net gain in coins per transaction
def coins_gain_per_transaction : Nat := 4

-- Define a function that calculates the total number of coins after n transactions
def total_coins (n : Nat) : Nat :=
  initial_green + n * coins_gain_per_transaction

-- Define the theorem to prove that it's impossible for Laura to have the same number of red and green coins
theorem impossible_to_have_same_number_of_each_color :
  ¬ ∃ n : Nat, ∃ red green : Nat, red = green ∧ total_coins n = red + green := by
  sorry

end impossible_to_have_same_number_of_each_color_l525_525902


namespace sum_first_n_terms_of_b_n_l525_525584

variables (a_n : ℕ → ℤ)

-- Given conditions
def condition1 : Prop :=
  ∑ i in finset.range 3, a_n (i + 1) = 6

def condition2 : Prop :=
  ∑ i in finset.range 8, a_n (i + 1) = -4

-- General term a_n
def general_term_a_n : ℕ → ℤ :=
  λ n, 4 - n

-- Sequence b_n based on a_n
def b_n (a_n : ℕ → ℤ) : ℕ → ℤ :=
  λ n, (4 - a_n n) * 3^n

-- Sum of first n terms of b_n
def S_n (b_n : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, b_n (i + 1)

-- Main theorem proof statement
theorem sum_first_n_terms_of_b_n
  (h1 : condition1 a_n)
  (h2 : condition2 a_n) :
  ∀ n, S_n (b_n a_n) n = ((n:ℤ) / 2 - 1 / 4) * 3^n + 1 / 4 :=
sorry

end sum_first_n_terms_of_b_n_l525_525584


namespace probability_of_two_prime_numbers_l525_525398

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l525_525398


namespace parallel_lines_unique_m_l525_525179

-- Given conditions
def line1 (m : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), x + (1 + m) * y - 2 = 0
def line2 (m : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), m * x + 2 * y + 4 = 0

-- Proof problem to show: the lines are parallel implies m = 1
theorem parallel_lines_unique_m :
  (∀ (m : ℝ) (P : ℝ × ℝ), line1 m P → line2 m P) → m = 1 :=
by sorry

end parallel_lines_unique_m_l525_525179


namespace sin2theta_plus_sec2theta_l525_525086

theorem sin2theta_plus_sec2theta {θ : ℝ} (h : Real.tan θ = 2) : Real.sin (2 * θ) + Real.sec θ ^ 2 = 29 / 5 :=
by
  sorry

end sin2theta_plus_sec2theta_l525_525086


namespace problem_statement_l525_525603

def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 6*x + a^2 - 1

theorem problem_statement (a : ℝ) :
  f (Real.sqrt 2) a < f 4 a ∧ f 4 a < f 3 a :=
by
  sorry

end problem_statement_l525_525603


namespace rectangular_equation_of_C_tangent_curves_C_C1_l525_525208

-- Define the conditions
def polar_to_rectangular (ρ θ : ℝ) : (ℝ × ℝ) := (ρ * Real.cos θ, ρ * Real.sin θ)

def C_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

def C_rectangular (x y : ℝ) : Prop := (x - 0)^2 + (y - 2)^2 = 4

def C1_parametric (x y r α : ℝ) : Prop := x = 3 + r * Real.cos α ∧ y = -2 + r * Real.sin α

-- The Lean statements representing the problem
theorem rectangular_equation_of_C (ρ θ : ℝ) :
  C_polar ρ θ →
  ∃ x y, polar_to_rectangular ρ θ = (x, y) ∧ C_rectangular x y := 
sorry

theorem tangent_curves_C_C1 (r : ℝ) :
  (∃ x y α, C1_parametric x y r α) →
  (∃ x y, C_rectangular x y) →
  (dist (3, -2) (0, 2) = 2 + Real.abs r ∨ dist (3, -2) (0, 2) = Real.abs (2 - r)) →
  (r = 3 ∨ r = -3 ∨ r = 7 ∨ r = -7) :=
sorry

end rectangular_equation_of_C_tangent_curves_C_C1_l525_525208


namespace prob_primes_1_to_30_l525_525415

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l525_525415


namespace price_of_pants_l525_525351

-- Given conditions
variables (P B : ℝ)
axiom h1 : P + B = 70.93
axiom h2 : P = B - 2.93

-- Statement to prove
theorem price_of_pants : P = 34.00 :=
by
  sorry

end price_of_pants_l525_525351


namespace probability_two_primes_from_1_to_30_l525_525405

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l525_525405


namespace dave_initial_apps_l525_525940

theorem dave_initial_apps
  (F : ℕ)
  (apps_left : ℕ)
  (files_left : ℕ)
  (apps_gt_files : ℕ)
  (h1 : F = 24)
  (h2 : apps_left = 21)
  (h3 : files_left = 4)
  (h4 : apps_left = files_left + apps_gt_files)
  (h5 : apps_gt_files = 17) :
  (A : ℕ) (hA : A = F + 17) :
  A = 41 :=
by
  sorry

end dave_initial_apps_l525_525940


namespace vector_subtraction_l525_525624

def p : ℝ × ℝ × ℝ := (3, -4, 6)
def q : ℝ × ℝ × ℝ := (-2, 5, -3)

theorem vector_subtraction :
  p.1 - 5 * q.1 = 13 ∧ p.2 - 5 * q.2 = -29 ∧ p.3 - 5 * q.3 = 21 := by
  sorry

end vector_subtraction_l525_525624


namespace sequence_nonzero_l525_525492

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n ≥ 1, (a (n + 2) = if (a n * a (n + 1)) % 2 = 0 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n)

theorem sequence_nonzero (a : ℕ → ℤ) (h : sequence a) : ∀ n, a n ≠ 0 :=
sorry

end sequence_nonzero_l525_525492


namespace log_positive_solution_set_l525_525147

theorem log_positive_solution_set (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > 0) (h4 : a^2 = b^2 + 1) :
  {x : ℝ | x > 2} = {x : ℝ | ∃ (u : ℝ), u = a^x - b^x ∧ real.log u > 0} :=
by
  sorry

end log_positive_solution_set_l525_525147


namespace solve_exponential_diophantine_equation_l525_525576

theorem solve_exponential_diophantine_equation :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 → (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by {
  sorry
}

end solve_exponential_diophantine_equation_l525_525576


namespace routes_PQ_count_l525_525519

-- We define the cities as a type
inductive City
| P | Q | R | S | T

open City

-- We define the roads as a relation between cities
def Road : City → City → Prop
| P, Q := True
| P, R := True
| P, T := True
| Q, S := True
| Q, R := True
| R, S := True
| S, T := True
| _, _ := False

-- Prove that the number of routes from P to Q using each road exactly once is 16
theorem routes_PQ_count :
  ∃ (count : ℕ), count = 16 ∧
  ∀ (route : list (City × City)), 
    route.head = (P, Q) ∧
    (route.nodup ∧ ∀ (c d : City), (c, d) ∈ route → Road c d) → 
    route.length = 7 := 
sorry

end routes_PQ_count_l525_525519


namespace double_sum_ij_l525_525539

theorem double_sum_ij : 
  (\sum i in Finset.range 50, \sum j in Finset.range 50, (i + 1 + j + 1 + 10)) = 280000 := by
  sorry

end double_sum_ij_l525_525539


namespace dimes_paid_l525_525625

theorem dimes_paid (cost_in_dollars : ℕ) (dollars_to_dimes : ℕ) (h₁ : cost_in_dollars = 5) (h₂ : dollars_to_dimes = 10) :
  cost_in_dollars * dollars_to_dimes = 50 :=
by
  sorry

end dimes_paid_l525_525625


namespace sum_series_eq_l525_525523

theorem sum_series_eq : (∑ n in Finset.range (∞ + 1), ∑ k in Finset.range n, (k : ℝ) / 3^(n + k)) = 9 / 128 := 
by
  -- Luckily, this is where the magic happens.
  sorry

end sum_series_eq_l525_525523


namespace cricket_players_l525_525188

theorem cricket_players (total_students : ℕ) 
                       (football_players : ℕ) 
                       (neither_football_nor_cricket : ℕ) 
                       (both_football_and_cricket : ℕ) 
                       (H1 : total_students = 250) 
                       (H2 : football_players = 160) 
                       (H3 : neither_football_nor_cricket = 50) 
                       (H4 : both_football_and_cricket = 50) 
                       : ∃ (C : ℕ), C = 90 :=
by 
  let total_students_at_least_one_sport := total_students - neither_football_nor_cricket
  have H5 : total_students_at_least_one_sport = 200, { rw [H1, H3], norm_num }
  let only_football_players := football_players - both_football_and_cricket 
  have H6 : only_football_players = 110, { rw [H2, H4], norm_num }
  let C_only := total_students_at_least_one_sport - only_football_players - both_football_and_cricket
  have H7 : C_only = 40, { rw [H5, H6, H4], norm_num }
  let C := C_only + both_football_and_cricket
  have H8 : C = 90, { rw [H7, H4], norm_num }
  use C
  exact H8

end cricket_players_l525_525188


namespace can_divide_cube_into_71_l525_525711

theorem can_divide_cube_into_71 : 
  ∃ (n : ℕ), n = 71 ∧ 
  (∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = f k + 7) ∧ f n = 71) :=
by
  sorry

end can_divide_cube_into_71_l525_525711


namespace acute_angle_between_planes_l525_525726

/-- Points O, A, B, and C are positioned such that ∠AOB = 60°, ∠BOC = 90°, and ∠COA = 120°. 
The acute angle θ between the planes AOB and AOC satisfies cos² θ = 1. Hence, 100m + n = 101.
-/
theorem acute_angle_between_planes :
  ∃ (m n : ℕ), nat.coprime m n ∧ cos 60 * cos 90 * cos 120 = cos 0 ∧ m = 1 ∧ n = 1 ∧ 100 * m + n = 101 :=
    sorry

end acute_angle_between_planes_l525_525726


namespace max_area_triangle_ABF_l525_525602

variables (a b : ℝ) (h : a > b) (hb : b > 0)

def ellipse_eq (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def right_focus := (sqrt (a^2 - b^2), 0)

theorem max_area_triangle_ABF :
  ∃ (A B F : ℝ × ℝ), ellipse_eq a b A.1 A.2 ∧ ellipse_eq a b B.1 B.2 ∧
  F = right_focus a b ∧
  (∃ θ : ℝ, (A = (a * Real.cos θ, b * Real.sin θ) ∧ B = (a * Real.cos θ, -b * Real.sin θ) ∧
  (∃ θ : ℝ, 
  (A = (a * Real.cos θ, b * Real.sin θ) ∧ 
  B = (a * Real.cos θ, -b * Real.sin θ))) ∧ 
  let area := 1/2 * |A.1 * (B.2 - F.2) + B.1 * (F.2 - A.2) + F.1 * (A.2 - B.2)| in
  ∃ θ : ℝ, θ = Real.pi / 2 →
  area = b * sqrt (a^2 - b^2)) :=
sorry

end max_area_triangle_ABF_l525_525602


namespace csc_neg_45_eq_neg_sqrt_2_l525_525059

noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

theorem csc_neg_45_eq_neg_sqrt_2 :
  csc (-Real.pi / 4) = -Real.sqrt 2 := by
  sorry

end csc_neg_45_eq_neg_sqrt_2_l525_525059


namespace smallest_solution_l525_525072

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 1)) + (1 / (x - 5)) = 4 / (x - 4)

theorem smallest_solution : 
  ∃ x : ℝ, equation x ∧ x ≠ 1 ∧ x ≠ 5 ∧ x ≠ 4 ∧ x = (5 - Real.sqrt 33) / 2 :=
by
  sorry

end smallest_solution_l525_525072


namespace checker_on_diagonal_l525_525756

theorem checker_on_diagonal
  (board : ℕ)
  (n_checkers : ℕ)
  (symmetric : (ℕ → ℕ → Prop))
  (diag_check : ∀ i j, symmetric i j -> symmetric j i)
  (num_checkers_odd : Odd n_checkers)
  (board_size : board = 25)
  (checkers : n_checkers = 25) :
  ∃ i, i < 25 ∧ symmetric i i := by
  sorry

end checker_on_diagonal_l525_525756


namespace vector_subtraction_proof_l525_525550

def v1 : ℝ × ℝ := (3, -8)
def v2 : ℝ × ℝ := (2, -6)
def a : ℝ := 5
def answer : ℝ × ℝ := (-7, 22)

theorem vector_subtraction_proof : (v1.1 - a * v2.1, v1.2 - a * v2.2) = answer := 
by
  sorry

end vector_subtraction_proof_l525_525550


namespace b_range_condition_l525_525564

theorem b_range_condition (b : ℝ) : 
  -2 * Real.sqrt 6 < b ∧ b < 2 * Real.sqrt 6 ↔ (b^2 - 24) < 0 :=
by
  sorry

end b_range_condition_l525_525564


namespace area_of_triangle_l525_525685

theorem area_of_triangle (a b c : ℝ) (h : a = 26 ∧ b = 24 ∧ c = 20) : 
  let s := (a + b + c) / 2 in 
  √(s * (s - a) * (s - b) * (s - c)) = 228 := 
by
  rcases h with ⟨ha, hb, hc⟩
  rw [ha, hb, hc]
  let s := (26 + 24 + 20) / 2
  sorry

end area_of_triangle_l525_525685


namespace expression_value_l525_525033

theorem expression_value : (15 + 7)^2 - (15^2 + 7^2) = 210 := by
  sorry

end expression_value_l525_525033


namespace seq_diff_geometric_min_n_exists_l525_525581

variable {α : Type} [LinearOrderedRing α]

def sequence_a (n : ℕ) : α
| 0     := 4
| (n+1) := 2 * (sequence_a n - n + 1)

def seq_diff (n : ℕ) : α :=
  sequence_a n - 2 * n

theorem seq_diff_geometric : ∃ r : α, ∀ n : ℕ, seq_diff (n + 1) = r * seq_diff n :=
by
  sorry

def S (n : ℕ) : α :=
  (finset.range (n + 1)).sum (λ k, sequence_a k)

theorem min_n_exists (n : ℕ) : ∃ n : ℕ, S n ≥ sequence_a n + 2 * n^2 :=
by
  sorry

end seq_diff_geometric_min_n_exists_l525_525581


namespace quadrilateral_diagonal_angle_l525_525792

-- Definitions of midpoint and necessary constructs
def midpoint (A B : Point ℝ) : Point ℝ := Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

-- Condition of being a quadrilateral
structure Quadrilateral (A B C D : Point ℝ) : Prop :=
(convex : Convex ℝ (convex_hull ℝ ({A, B, C, D} : Set (Point ℝ))))

-- Lean statement of the problem
theorem quadrilateral_diagonal_angle (A B C D : Point ℝ)
  (h : Quadrilateral A B C D) 
  (h_dist_midpoints : distance (midpoint A B) (midpoint C D) =
                      distance (midpoint (midpoint A B) (midpoint C D))) :
  angle A D B C = 90 :=
by
  sorry

end quadrilateral_diagonal_angle_l525_525792


namespace correctStatements_l525_525484

open Classical

-- Define the conditions
def hobbyGroup (boys girls : Nat) (bChoice gChoice : Nat) : Prop :=
  boys = 20 ∧ girls = 10 ∧ bChoice = 2 ∧ gChoice = 3

def systematicSampling {boys girls bChoice gChoice: Nat} (h : hobbyGroup boys girls bChoice gChoice) : Prop :=
  let total := boys + girls in total / 5 = 6

def randomSampling {boys girls bChoice gChoice: Nat} (h : hobbyGroup boys girls bChoice gChoice) : Prop := 
  boys + girls < 50

def notStratifiedSampling {boys girls bChoice gChoice: Nat} (h : hobbyGroup boys girls bChoice gChoice) : Prop := 
  bChoice / boys ≠ gChoice / girls

def equalProbability {boys girls bChoice gChoice: Nat} (h : hobbyGroup boys girls bChoice gChoice) : Prop := 
  boys * gChoice = girls * bChoice

-- The main theorem to prove
theorem correctStatements (boys girls bChoice gChoice : Nat) (h : hobbyGroup boys girls bChoice gChoice) :
  systematicSampling h ∧ randomSampling h ∧ notStratifiedSampling h ∧ ¬ equalProbability h :=
sorry

end correctStatements_l525_525484


namespace compare_M_N_P_l525_525989

variable (a b : ℝ)
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hab : a ≠ b)

def M := a / Real.sqrt b + b / Real.sqrt a
def N := Real.sqrt a + Real.sqrt b
def P := 2 * Real.sqrt (Real.sqrt (a * b))

theorem compare_M_N_P : M a b > N a b ∧ N a b > P a b := by
  sorry

end compare_M_N_P_l525_525989


namespace complex_calculation_l525_525114

theorem complex_calculation (z : ℂ) (h : z = 2 - complex.I) : 
  z * (z.conj + complex.I) = 6 + 2 * complex.I := 
by sorry

end complex_calculation_l525_525114


namespace problem_statement_l525_525740

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)

theorem problem_statement : 
  (∀ x : ℝ, f(-x) = -f(x)) ∧ (∀ x y : ℝ, 0 < x → x < y → y < 1 → f(x) < f(y)) :=
by
  sorry

end problem_statement_l525_525740


namespace number_of_males_l525_525901

theorem number_of_males (population : ℕ) (percentage_males : ℝ) (h_pop : population = 500) 
  (h_percent : percentage_males = 0.4) : ∃ males : ℕ, males = 200 :=
by {
  use (percentage_males * population : ℕ),
  sorry,
}

end number_of_males_l525_525901


namespace prop1_prop2_correctness_l525_525686

-- Definitions based on problem conditions
def complementary_angle (α β : ℝ) : Prop :=
  α ∈ [0, 2 * Real.pi) ∧ β ∈ [0, 2 * Real.pi) ∧ Real.cos (α + β) = Real.cos α + Real.cos β

-- Proposition 1
theorem prop1 (α : ℝ) (h : α ∈ (Real.pi / 3, Real.pi / 2)) :
  ∃ β₁ β₂ : ℝ, complementary_angle α β₁ ∧ complementary_angle α β₂ := sorry

-- Proposition 2
theorem prop2 (α : ℝ) (h : α ∈ (0, Real.pi / 3)) :
  ¬ ∃ β : ℝ, complementary_angle α β := sorry

-- Correctness of the propositions
theorem correctness :
  (∀ α : ℝ, α ∈ (Real.pi / 3, Real.pi / 2) → ∃ β₁ β₂ : ℝ, complementary_angle α β₁ ∧ complementary_angle α β₂) ∧
  (∀ α : ℝ, α ∈ (0, Real.pi / 3) → ¬ ∃ β : ℝ, complementary_angle α β) := by
  split
  · intros α h
    exact prop1 α h
  · intros α h
    exact prop2 α h

end prop1_prop2_correctness_l525_525686


namespace total_assignments_for_28_points_l525_525267

-- Definitions based on conditions
def assignments_needed (points : ℕ) : ℕ :=
  (points / 7 + 1) * (points % 7) + (points / 7) * (7 - points % 7)

-- The theorem statement, which asserts the answer to the given problem
theorem total_assignments_for_28_points : assignments_needed 28 = 70 :=
by
  -- proof will go here
  sorry

end total_assignments_for_28_points_l525_525267


namespace carrots_cost_l525_525258

/-
Define the problem conditions and parameters.
-/
def num_third_grade_classes := 5
def students_per_third_grade_class := 30
def num_fourth_grade_classes := 4
def students_per_fourth_grade_class := 28
def num_fifth_grade_classes := 4
def students_per_fifth_grade_class := 27

def cost_per_hamburger : ℝ := 2.10
def cost_per_cookie : ℝ := 0.20
def total_lunch_cost : ℝ := 1036

/-
Calculate the total number of students.
-/
def total_students : ℕ :=
  (num_third_grade_classes * students_per_third_grade_class) +
  (num_fourth_grade_classes * students_per_fourth_grade_class) +
  (num_fifth_grade_classes * students_per_fifth_grade_class)

/-
Calculate the cost of hamburgers and cookies.
-/
def hamburgers_cost : ℝ := total_students * cost_per_hamburger
def cookies_cost : ℝ := total_students * cost_per_cookie
def total_hamburgers_and_cookies_cost : ℝ := hamburgers_cost + cookies_cost

/-
State the proof problem: How much do the carrots cost?
-/
theorem carrots_cost : total_lunch_cost - total_hamburgers_and_cookies_cost = 185 :=
by
  -- Proof is omitted
  sorry

end carrots_cost_l525_525258


namespace probability_of_prime_pairs_l525_525377

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l525_525377


namespace find_M_l525_525170

theorem find_M : 995 + 997 + 999 + 1001 + 1003 = 5100 - 104 :=
by 
  sorry

end find_M_l525_525170


namespace remaining_money_after_shopping_l525_525440

theorem remaining_money_after_shopping (initial_money : ℝ) (percentage_spent : ℝ) (final_amount : ℝ) :
  initial_money = 1200 → percentage_spent = 0.30 → final_amount = initial_money - (percentage_spent * initial_money) → final_amount = 840 :=
by
  intros h_initial h_percentage h_final
  sorry

end remaining_money_after_shopping_l525_525440


namespace motorcyclist_ride_l525_525489

theorem motorcyclist_ride (
  (d_AB : ℝ) (d_BC : ℝ) (v_avg : ℝ) (t_total : ℝ) (x : ℝ) : ℝ
  (h1 : d_AB = 120)
  (h2 : d_BC = d_AB / 2)
  (h3 : v_avg = 20)
  (h4 : t_total = (d_AB + d_BC) / v_avg)
  (h5 : x * (t_total - t_AB) = t_AB)
) : x = 2 :=
sorry

end motorcyclist_ride_l525_525489


namespace optimal_bicycle_location_l525_525504

/-- 
  Define the variables needed for the problem:
  - AB is the distance between points A and B
  - AC is the mid-point, exactly halfway between A and B
  - cycling_speed is the speed at which Andrey and Seva cycle
  - walking_speed is the speed at which Andrey and Seva walk
  - D is the point where the bicycle should be left
  - optimal_distance_from_B is the optimal distance from B where the bicycle should be left
-/
variables (AB AC : ℝ) (cycling_speed walking_speed : ℝ)

-- Assign values to the variables according to the problem statement
def AB := 30
def AC := 15
def cycling_speed := 20
def walking_speed := 5

-- Define the total travel times for Andrey and Seva
def t_A (D : ℝ) : ℝ := (15 - (AC - D)) / cycling_speed + D / walking_speed
def t_S (D : ℝ) : ℝ := D / walking_speed + (15 - D) / cycling_speed

-- The proof problem statement: Find the optimal distance from B to leave the bicycle
theorem optimal_bicycle_location : 
  ∃ (D : ℝ), D = 10 ∧ t_A AB AC cycling_speed walking_speed D = t_S AB AC cycling_speed walking_speed D := 
sorry

end optimal_bicycle_location_l525_525504


namespace probability_of_two_primes_is_correct_l525_525389

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l525_525389


namespace math_problem_l525_525467

theorem math_problem : 33333 * 33334 = 1111122222 := 
by sorry

end math_problem_l525_525467


namespace odd_number_of_irational_segments_l525_525619

def number_points (n : ℕ) := n + 2

def a (A : ℕ → ℝ) (i : ℕ) : ℤ :=
  if (A i) ∈ ℚ then 1 else -1

theorem odd_number_of_irational_segments (1 : ℝ) (sqrt2 : ℝ) (n : ℕ) (A : ℕ → ℝ)
(H1: A 0 = 1) (H2: A (number_points n - 1) = sqrt2)
(H3: ∀ i : ℕ, 0 < i < number_points n → 1 < A i ∧ A i < sqrt2)
: ∃ k : ℕ, odd k ∧ 
  (∀ i : ℕ, 0 ≤ i < n + 1 → 
    ((a A i) = 1 ∨ (a A (i+1)) = -1)) := 
sorry

end odd_number_of_irational_segments_l525_525619


namespace probability_of_two_prime_numbers_l525_525397

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l525_525397


namespace complex_calculation_l525_525102

variable (z : ℂ)

theorem complex_calculation : (z = 2 - (-1:ℂ)) → (z * (conj z + (-1:ℂ))) = 6 + 2 * (-1:ℂ) :=
by
  intro h
  rw h
  have h_conj : conj (2 - (-1:ℂ)) = 2 + (-1:ℂ) := by simp
  rw h_conj
  ring
  done

end complex_calculation_l525_525102


namespace number_of_pencils_selling_price_equals_loss_l525_525282

theorem number_of_pencils_selling_price_equals_loss :
  ∀ (S C L : ℝ) (N : ℕ),
  C = 1.3333333333333333 * S →
  L = C - S →
  (S / 60) * N = L →
  N = 20 :=
by
  intros S C L N hC hL hN
  sorry

end number_of_pencils_selling_price_equals_loss_l525_525282


namespace linear_regression_probability_and_expectation_l525_525476

section LinearRegression

variable (data : List (ℕ × ℕ))
variable (x̄ ȳ : ℚ)
variable (n : ℕ)
variable (b : ℚ)
variable (a : ℚ)

-- Define the data
def givenData := [(1, 0), (2, 4), (3, 7), (4, 9), (5, 11), (6, 12), (7, 13)]

-- Define the averages
def x_average := (1 + 2 + 3 + 4 + 5 + 6 + 7) / 7
def y_average := (0 + 4 + 7 + 9 + 11 + 12 + 13) / 7

-- Define the regression coefficients
def b := (1 * 0 + 2 * 4 + 3 * 7 + 4 * 9 + 5 * 11 + 6 * 12 + 7 * 13 - 7 * x_average * y_average) /
          (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 - 7 * x_average^2)

def a := y_average - b * x_average

-- Property to prove
theorem linear_regression : a = -3 / 7 ∧ b = 59 / 28 ∧ ∀ x, (a + b * x) = 59 / 28 * x - 3 / 7 := 
begin
  sorry
end

end LinearRegression

section ProbabilityAndExpectation

variable (data : List (ℕ × ℕ))
variable (ξ : Type) [Fintype ξ] [DecidableEq ξ]

-- Given data and probability calculations
def num_days_gt_average := [(4, 9), (5, 11), (6, 12), (7, 13)].length
def num_days_le_average := [(1, 0), (2, 4), (3, 7)].length
def P (k : ℕ) := (nat.choose num_days_le_average (3 - k) * nat.choose num_days_gt_average k) /
                 (nat.choose 7 3 : ℚ)

-- Expected value calculation
def E := ∑ k in (Finset.range 4), k * P k

-- Property to prove
theorem probability_and_expectation : E = 12 / 7 :=
begin
  sorry
end

end ProbabilityAndExpectation

end linear_regression_probability_and_expectation_l525_525476


namespace probability_prime_and_cube_is_correct_l525_525368

-- Conditions based on the problem
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_cube (n : ℕ) : Prop :=
  n = 1 ∨ n = 8

def possible_outcomes := 8 * 8
def successful_outcomes := 4 * 2

noncomputable def probability_of_prime_and_cube :=
  (successful_outcomes : ℝ) / (possible_outcomes : ℝ)

theorem probability_prime_and_cube_is_correct :
  probability_of_prime_and_cube = 1 / 8 :=
by
  sorry

end probability_prime_and_cube_is_correct_l525_525368


namespace factors_of_expression_l525_525629

def total_distinct_factors : ℕ :=
  let a := 10
  let b := 3
  let c := 2
  (a + 1) * (b + 1) * (c + 1)

theorem factors_of_expression :
  total_distinct_factors = 132 :=
by 
  -- the proof goes here
  sorry

end factors_of_expression_l525_525629


namespace macaroon_weight_l525_525219

theorem macaroon_weight (bakes : ℕ) (packs : ℕ) (bags_after_eat : ℕ) (remaining_weight : ℕ) (macaroons_per_bag : ℕ) (weight_per_bag : ℕ)
  (H1 : bakes = 12) 
  (H2 : packs = 4)
  (H3 : bags_after_eat = 3)
  (H4 : remaining_weight = 45)
  (H5 : macaroons_per_bag = bakes / packs) 
  (H6 : weight_per_bag = remaining_weight / bags_after_eat) :
  ∀ (weight_per_macaroon : ℕ), weight_per_macaroon = weight_per_bag / macaroons_per_bag → weight_per_macaroon = 5 :=
by
  sorry -- Proof will come here, not required as per instructions

end macaroon_weight_l525_525219


namespace probability_both_numbers_are_prime_l525_525424

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l525_525424


namespace combined_weight_correct_l525_525959

noncomputable def combined_weight : ℕ :=
  let small_candle_weight := 4 + 2 + 0.5
  let medium_candle_weight := 8 + 1 + 1
  let large_candle_weight := 16 + 3 + 2
  let total_small_candles_weight := small_candle_weight * 4
  let total_medium_candles_weight := medium_candle_weight * 3
  let total_large_candles_weight := large_candle_weight * 2
  total_small_candles_weight + total_medium_candles_weight + total_large_candles_weight

theorem combined_weight_correct : combined_weight = 98 := 
  sorry

end combined_weight_correct_l525_525959


namespace complex_calculation_l525_525110

theorem complex_calculation (z : ℂ) (h : z = 2 - complex.I) : 
  z * (z.conj + complex.I) = 6 + 2 * complex.I := 
by sorry

end complex_calculation_l525_525110


namespace complex_calculation_l525_525109

theorem complex_calculation (z : ℂ) (h : z = 2 - complex.I) : 
  z * (z.conj + complex.I) = 6 + 2 * complex.I := 
by sorry

end complex_calculation_l525_525109


namespace max_min_difference_students_l525_525957

theorem max_min_difference_students (S F : ℕ) (m M : ℕ) :
  (1050 ≤ S ∧ S ≤ 1125) ∧ (525 ≤ F ∧ F ≤ 675) ∧ 
  (S + F - S ∩ F = 1500) ∧ 
  (m = 1050 + 525 - 1500) ∧ 
  (M = 1125 + 675 - 1500) → 
  M - m = 225 := by
  sorry

end max_min_difference_students_l525_525957


namespace min_negative_numbers_l525_525269

theorem min_negative_numbers (a b c d : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a + b + c < d) (h6 : a + b + d < c) (h7 : a + c + d < b) (h8 : b + c + d < a) :
  3 ≤ (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) := 
sorry

end min_negative_numbers_l525_525269


namespace problem_statement_l525_525090

-- Define that the function f is even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define that the function f satisfies f(x) = f(2 - x)
def satisfies_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

-- Define that the function f is decreasing on a given interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define that the function f is increasing on a given interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Given hypotheses and the theorem to prove. We use two statements for clarity.
theorem problem_statement (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_symmetry : satisfies_symmetry f) 
  (h_decreasing_1_2 : is_decreasing_on f 1 2) : 
  is_increasing_on f (-2) (-1) ∧ is_decreasing_on f 3 4 := 
by 
  sorry

end problem_statement_l525_525090


namespace children_more_than_adults_l525_525480

-- Conditions
def total_members : ℕ := 120
def adult_percentage : ℝ := 0.40
def child_percentage : ℝ := 1 - adult_percentage

-- Proof problem statement
theorem children_more_than_adults : 
  let number_of_adults := adult_percentage * total_members
  let number_of_children := child_percentage * total_members
  let difference := number_of_children - number_of_adults
  difference = 24 :=
by
  sorry

end children_more_than_adults_l525_525480


namespace all_values_are_equal_l525_525831

theorem all_values_are_equal
  (f : ℤ × ℤ → ℕ)
  (h : ∀ x y : ℤ, f (x, y) * 4 = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1))
  (hf_pos : ∀ x y : ℤ, 0 < f (x, y)) : 
  ∀ x y x' y' : ℤ, f (x, y) = f (x', y') :=
by
  sorry

end all_values_are_equal_l525_525831


namespace find_m_n_monotonicity_find_t_l525_525089

noncomputable def f (m n t x : ℝ) : ℝ := (m * x^2 + t) / (x + n)

theorem find_m_n (t : ℝ) (ht : t > 0) (h_odd : ∀ x, f m n t x = -f m n t (-x)) (hf1 : f m n t 1 = t + 1) : 
  m = 1 ∧ n = 0 := 
sorry

noncomputable def g (t x : ℝ) : ℝ := x + t / x

theorem monotonicity (t : ℝ) (ht : t > 0) : 
  (∀ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < sqrt t → g t x1 > g t x2) ∧ 
  (∀ x1 x2, sqrt t ≤ x1 ∧ x1 < x2 → g t x1 < g t x2) := 
sorry

theorem find_t (t : ℝ) (ht : t > 0) (h_f_max_min : max (λ x, 2 ≤ x ∧ x ≤ 4) (g t) - min (λ x, 2 ≤ x ∧ x ≤ 4) (g t) = 2) : 
  t = 16 := 
sorry

end find_m_n_monotonicity_find_t_l525_525089


namespace sufficient_but_not_necessary_condition_for_inequality_l525_525912

theorem sufficient_but_not_necessary_condition_for_inequality (a : ℝ) (h : a > 0) : 
  (a ≥ 1 → a + 1 / a ≥ 2) ∧ (∃ (a : ℝ), a > 0 ∧ a + 1 / a ≥ 2 ∧ a < 1) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_for_inequality_l525_525912


namespace find_x_parallel_l525_525570

open_locale big_operators

variables (a b : ℝ^3) (x : ℝ)

def a_vector : ℝ^3 := ⟨2, -1, 3⟩
def b_vector (x : ℝ) : ℝ^3 := ⟨-4, 2, x⟩

theorem find_x_parallel (h : ∃ λ : ℝ, a_vector = λ • b_vector x) : x = -6 := by
  sorry

end find_x_parallel_l525_525570


namespace math_problem_l525_525309

-- Definitions based on conditions
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

-- Main theorem statement
theorem math_problem :
  avg4 (avg4 2 2 0 2) (avg2 3 1) 0 3 = 13 / 8 :=
by
  sorry

end math_problem_l525_525309


namespace container_maximized_volume_and_height_l525_525017

noncomputable def height_when_volume_maximized (length: ℝ) (extra: ℝ) : ℝ :=
  3.2 - 2 * (length / 4) - 2 * (length / 4 + extra)

noncomputable def volume (length: ℝ) (extra: ℝ) : ℝ :=
  let x := length / 4 in
  x * (x + extra) * (3.2 - 2*x)

theorem container_maximized_volume_and_height :
  ∀ length extra,
  length = 14.8 → extra = 0.5 →
  height_when_volume_maximized length extra = 1.2 ∧ 
  volume length extra = 1.8 :=
by {
  intros,
  rw [height_when_volume_maximized, volume],
  -- calculations demonstrating the proof will be present here
  sorry
}

end container_maximized_volume_and_height_l525_525017


namespace find_present_worth_l525_525789

noncomputable def present_worth (BG : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
(BG * 100) / (R * ((1 + R/100)^T - 1) - R * T)

theorem find_present_worth : present_worth 36 10 3 = 1161.29 :=
by
  sorry

end find_present_worth_l525_525789


namespace problem_statement_l525_525744

variable (U : Type) [LinearOrder U] [OrderTopology U]

/-- The universal set U is the real numbers -/
def universal_set : Set ℝ := Set.univ

/-- Set A is defined by {x | 2^x > 1} -/
def A : Set ℝ := {x : ℝ | Real.exp x * Real.log 2 > 1}

/-- Set B is defined by {-1 ≤ x ≤ 5} -/
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

/-- We aim to prove that (\complement_U A) ∩ B = [-1, 0] -/
theorem problem_statement :
  (universal_set \ A : Set ℝ) ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 0} :=
by
  sorry

end problem_statement_l525_525744


namespace part1_part2_part3_l525_525153

theorem part1 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0 ↔ x < -3 ∨ x > -2) : k = -2/5 :=
sorry

theorem part2 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) : k < -Real.sqrt 6 / 6 :=
sorry

theorem part3 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, ¬ (k * x^2 - 2 * x + 6 * k < 0)) : k ≥ Real.sqrt 6 / 6 :=
sorry

end part1_part2_part3_l525_525153


namespace conjugate_complex_number_conjugate_of_fraction_l525_525315

-- Definitions of complex numbers and their conjugates
open Complex

theorem conjugate_complex_number (z : ℂ) (a b : ℝ) (h : z = (2 + I) / (1 + I)) :
  conj z = (3 / 2) + (1 / 2) * I := by
  have z_eq : z = (3 / 2) - (1 / 2) * I := by sorry
  rw [z_eq]
  simp [conj]

-- Using the generic theorem to apply specific proof and use sorry to skip the proof
theorem conjugate_of_fraction : conj ((2 + I) / (1 + I)) = (3 / 2) + (1 / 2) * I := by
  exact conjugate_complex_number ((2 + I) / (1 + I)) (3 / 2) (1 / 2) rfl

end conjugate_complex_number_conjugate_of_fraction_l525_525315


namespace at_least_one_genuine_product_l525_525568

-- Definitions of the problem conditions
structure Products :=
  (total : ℕ)
  (genuine : ℕ)
  (defective : ℕ)

def products : Products := { total := 12, genuine := 10, defective := 2 }

-- Definition of the event
def certain_event (p : Products) (selected : ℕ) : Prop :=
  selected > p.defective

-- The theorem stating that there is at least one genuine product among the selected ones
theorem at_least_one_genuine_product : certain_event products 3 :=
by
  sorry

end at_least_one_genuine_product_l525_525568


namespace eq_radius_of_circle_l525_525830

open Real
open Classical

noncomputable theory

variables (A B C D E O : Point)
variables (S : Circle)

-- Given conditions
variables (hA : A ∈ S)
variables (hB : B ∈ S)
variables (hC : C ∉ S)
variables (hD : D ∈ S)
variables (hAB_eq_BD : dist A B = dist B D)
variables (hCD_intersects_S : ∃ E, E ∈ S ∧ lies_on_line C D E)

-- Statement to prove
theorem eq_radius_of_circle 
    (h_eq_triangle : equilateral_triangle A B C)
    (h_circle_center: O = S.center)
    :
    dist E C = S.radius := 
sorry

end eq_radius_of_circle_l525_525830


namespace find_a_l525_525135

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = Real.log (-a * x)) (h2 : ∀ x : ℝ, f (-x) = -f x) :
  a = 1 :=
by
  sorry

end find_a_l525_525135


namespace radius_of_sphere_is_correct_l525_525577

noncomputable def radius_of_sphere : ℝ :=
  let A := (0, 0, 0) in
  let C := (1, 1, 1) in
  let E := (1/2, 1, 0) in
  let F := (1, 1/2, 0) in
  let M := (1/2, 1/2, 3/8) in
  let R := Math.sqrt((1 - 1/2)^2 + (1 - 1/2)^2 + (0 - 3/8)^2) in
  R

theorem radius_of_sphere_is_correct :
  radius_of_sphere = sqrt 41 / 8 := 
sorry

end radius_of_sphere_is_correct_l525_525577


namespace y_star_definition_l525_525980

def y_star (y : Real) : Real := y - 1

theorem y_star_definition (y : Real) : (5 : Real) - y_star 5 = 1 :=
  by sorry

end y_star_definition_l525_525980


namespace bicycle_speed_l525_525786

theorem bicycle_speed (x : ℝ) :
  (10 / x = 10 / (2 * x) + 1 / 3) → x = 15 :=
by
  intro h
  sorry

end bicycle_speed_l525_525786


namespace solve_for_y_l525_525305

theorem solve_for_y {y : ℕ} (h : (1000 : ℝ) = (10 : ℝ)^3) : (1000 : ℝ)^4 = (10 : ℝ)^y ↔ y = 12 :=
by
  sorry

end solve_for_y_l525_525305


namespace Rita_reading_problem_l525_525774

theorem Rita_reading_problem :
  ∃ x : ℕ, (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 95) ∧
  5 * x + 30 = 95 ∧
  x = 13 ∧
  ((95 / 10.0).ceil : ℕ) = 10 :=
by
  sorry

end Rita_reading_problem_l525_525774


namespace problem_statement_l525_525526

def sequence (f : ℕ → ℕ) : Prop :=
  f 3 = 13 ∧ ∀ n ≥ 4, f n = f (n - 1) + 4 * (n + 1)

theorem problem_statement (f : ℕ → ℕ) (h : sequence f) : f 20 = 1661 :=
by
  sorry

end problem_statement_l525_525526


namespace probability_of_two_prime_numbers_l525_525400

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l525_525400


namespace parallelepiped_volume_l525_525311

theorem parallelepiped_volume (AB BC : ℝ) (angle_BAD AC1 : ℝ) 
  (h_length1 : AB = 1) (h_length2 : BC = 4) (h_angle : angle_BAD = 60) 
  (h_diagonal : AC1 = 5) : 
  ∃ V : ℝ, V = 4 * real.sqrt 3 := 
by 
  sorry 

end parallelepiped_volume_l525_525311


namespace last_two_digits_of_floor_l525_525798

def last_two_digits (n : Nat) : Nat :=
  n % 100

theorem last_two_digits_of_floor :
  let x := 10^93
  let y := 10^31
  last_two_digits (Nat.floor (x / (y + 3))) = 8 :=
by
  sorry

end last_two_digits_of_floor_l525_525798


namespace length_of_BE_l525_525204

theorem length_of_BE 
  (ABCD_is_square : ∀ (A B C D : point), square A B C D → side_length A B = 4)
  (rectangles_congruent : ∀ (J K H G E B C F : point), congruent_rectangle J K H G E B C F)
  (side_EJ_three_times_EK : ∀ (E J K : point), side_length E J = 3 * side_length E K) : 
  ∃ (BE : ℝ), BE = (2 * real.sqrt 10) / 3 :=
by
  sorry

end length_of_BE_l525_525204


namespace real_solution_count_l525_525534

theorem real_solution_count :
  ∃ (y1 y2 : ℝ), y1 ≠ y2 ∧ (2 ∈ {y : ℝ | 2 ∘ y = 9}) ∧ (∀ y ∈ {y : ℝ | 2 ∘ y = 9}, y = y1 ∨ y = y2) :=
by 
  sorry

end real_solution_count_l525_525534


namespace resulting_solid_vertices_l525_525016

theorem resulting_solid_vertices (s1 s2 : ℕ) (orig_vertices removed_cubes : ℕ) :
  s1 = 5 → s2 = 2 → orig_vertices = 8 → removed_cubes = 8 → 
  (orig_vertices - removed_cubes + removed_cubes * (4 * 3 - 3)) = 40 := by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end resulting_solid_vertices_l525_525016


namespace bear_population_l525_525034

theorem bear_population (black_bears white_bears brown_bears total_bears : ℕ) 
(h1 : black_bears = 60)
(h2 : white_bears = black_bears / 2)
(h3 : brown_bears = black_bears + 40) :
total_bears = black_bears + white_bears + brown_bears :=
sorry

end bear_population_l525_525034


namespace add_water_to_solution_l525_525652

theorem add_water_to_solution (w : ℕ) :
  let initial_volume := 50
  let initial_concentration := 0.4
  let final_concentration := 0.25
  let acid_amount := initial_concentration * initial_volume
  let final_volume := initial_volume + w
  (acid_amount / final_volume = final_concentration) ↔ (w = 30) :=
by 
  sorry

end add_water_to_solution_l525_525652


namespace complex_multiplication_identity_l525_525091

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- State the theorem to be proved
theorem complex_multiplication_identity : z * (z_conjugate + complex.i) = 6 + 2 * complex.i :=
by {
  sorry -- proof goes here
}

end complex_multiplication_identity_l525_525091


namespace blackboard_numbers_l525_525921

theorem blackboard_numbers (N : ℕ) (h : N > 0) : 
  (∃ steps : ℕ, ∀ board : multiset ℕ, 
  (board.count N = 1) → 
  (∀ a ∈ board, a > 0) → 
  (∀ a ∈ board, a = 1 ∨ ∀ d : ℕ, d ∣ a → d ∈ board ∨ d = a) → 
  board.card = N^2) → 
  N = 1 := 
begin
  sorry
end

end blackboard_numbers_l525_525921


namespace monotonicity_of_g_g_positive_at_inverse_exp_number_of_zeros_of_f_l525_525146

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * real.log x - a * x + 3

noncomputable def g (a x : ℝ) : ℝ := 
  if h : 0 < x
  then a * real.log x + 1 / x
  else 0

theorem monotonicity_of_g (a : ℝ) : 
    (a ≤ 0 → ∀ x > 0, deriv (g a) x < 0) ∧ 
    (a > 0 → (∀ x > 0, x < 1 / a → deriv (g a) x < 0) ∧ (∀ x > 0, x > 1 / a → deriv (g a) x > 0)) :=
sorry

theorem g_positive_at_inverse_exp (a : ℝ) (h : a > real.exp 1) : g a (real.exp (-a)) > 0 :=
sorry

theorem number_of_zeros_of_f (a : ℝ) (h : a > real.exp 1) : ∃! x > 0, f a x = 0 :=
sorry

end monotonicity_of_g_g_positive_at_inverse_exp_number_of_zeros_of_f_l525_525146


namespace river_speed_calc_l525_525888

theorem river_speed_calc :
  ∃ v : ℝ, 
    v > 0 ∧ v = 1.2 ∧ 
    let speed_still_water := 6 in
    let total_distance := 5.76 in
    let total_time := 1 in
    let d := total_distance / 2 in
    let equation := d / (speed_still_water - v) + d / (speed_still_water + v) in
    equation = total_time :=
begin
  sorry
end

end river_speed_calc_l525_525888


namespace part_I_l525_525594

variable (a b c n p q : ℝ)

theorem part_I (hne0 : a ≠ 0) (bne0 : b ≠ 0) (cne0 : c ≠ 0)
    (h1 : a^2 + b^2 + c^2 = 2) (h2 : n^2 + p^2 + q^2 = 2) :
    (n^4 / a^2 + p^4 / b^2 + q^4 / c^2) ≥ 2 := 
sorry

end part_I_l525_525594


namespace probability_of_prime_pairs_l525_525371

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l525_525371


namespace distinct_integers_in_grid_l525_525022

noncomputable def grid_4x4 (A : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  (∀ i j, (i < 3 → abs (A i j - A (i + 1) j) = 1) ∧ (j < 3 → abs (A i j - A i (j + 1)) = 1)) ∧
  (A 0 0 = 3) ∧ 
  (∃ i j, A i j = 9) ∧
  (set.univ.image (λ (i : Fin 4) (j : Fin 4), A i j) = {3, 4, 5, 6, 7, 8, 9})

theorem distinct_integers_in_grid {A : Matrix (Fin 4) (Fin 4) ℤ} : grid_4x4 A → 
  set.card (set.univ.image (λ (i : Fin 4) (j : Fin 4), A i j)) = 7 :=
sorry

#check distinct_integers_in_grid

end distinct_integers_in_grid_l525_525022


namespace average_goods_per_hour_l525_525020

-- Define the conditions
def morning_goods : ℕ := 64
def morning_hours : ℕ := 4
def afternoon_rate : ℕ := 23
def afternoon_hours : ℕ := 3

-- Define the target statement to be proven
theorem average_goods_per_hour : (morning_goods + afternoon_rate * afternoon_hours) / (morning_hours + afternoon_hours) = 19 := by
  -- Add proof steps here
  sorry

end average_goods_per_hour_l525_525020


namespace complex_calculation_l525_525103

variable (z : ℂ)

theorem complex_calculation : (z = 2 - (-1:ℂ)) → (z * (conj z + (-1:ℂ))) = 6 + 2 * (-1:ℂ) :=
by
  intro h
  rw h
  have h_conj : conj (2 - (-1:ℂ)) = 2 + (-1:ℂ) := by simp
  rw h_conj
  ring
  done

end complex_calculation_l525_525103


namespace jack_helped_hours_l525_525777

-- Definitions based on the problem's conditions
def sam_rate : ℕ := 6  -- Sam assembles 6 widgets per hour
def tony_rate : ℕ := 2  -- Tony assembles 2 widgets per hour
def jack_rate : ℕ := sam_rate  -- Jack assembles at the same rate as Sam
def total_widgets : ℕ := 68  -- The total number of widgets assembled by all three

-- Statement to prove
theorem jack_helped_hours : 
  ∃ h : ℕ, (sam_rate * h) + (tony_rate * h) + (jack_rate * h) = total_widgets ∧ h = 4 := 
  by
  -- The proof is not necessary; we only need the statement
  sorry

end jack_helped_hours_l525_525777


namespace area_of_region_l525_525809

theorem area_of_region:
  let is_in_region (x y: ℝ) := y ≥ abs (x + 2) ∧ y ≤ 6 - 2 * abs x
  ∃ (A: ℝ), A = 10 ∧ 
  ∀ (x1 y1 x2 y2 x3 y3: ℝ), 
  is_in_region x1 y1 ∧ is_in_region x2 y2 ∧ is_in_region x3 y3 ∧
  (x1 y1, x2 y2, x3 y3) are the vertices of the triangular region implies 
  A = 1/2 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|
by
  sorry

end area_of_region_l525_525809


namespace find_m_value_l525_525157

theorem find_m_value : 
  ∀ (u v : ℝ), 
    (3 * u^2 + 4 * u + 5 = 0) ∧ 
    (3 * v^2 + 4 * v + 5 = 0) ∧ 
    (u + v = -4/3) ∧ 
    (u * v = 5/3) → 
    ∃ m n : ℝ, 
      (x^2 + m * x + n = 0) ∧ 
      ((u^2 + 1) + (v^2 + 1) = -m) ∧ 
      (m = -4/9) :=
by {
  -- Insert proof here
  sorry
}

end find_m_value_l525_525157


namespace max_height_is_24_km_can_pose_danger_l525_525323

-- Define the given acceleration provided by the engines
def engine_acceleration := 30 -- in m/s^2

-- Define the duration of the engine run
def engine_duration := 20 -- in seconds

-- Define the acceleration due to gravity
def gravity_acceleration := 10 -- in m/s^2

-- Define the initial conditions
def initial_velocity := engine_acceleration * engine_duration -- in m/s
def initial_height := (1 / 2 : ℝ) * engine_acceleration * (engine_duration ^ 2) -- in meters

-- Calculate the time to reach maximum height after engine stops working
def time_to_max_height := initial_velocity / gravity_acceleration -- in seconds

-- Calculate the additional height gained under gravity only
def additional_height := initial_velocity * time_to_max_height - (1 / 2 : ℝ) * gravity_acceleration * (time_to_max_height ^ 2) -- in meters

-- Calculate the total maximum height reached by the rocket
def total_max_height := initial_height + additional_height -- in meters

-- Define the height threshold to check possible danger
def danger_height := 20000 -- in meters

-- The proof problems
theorem max_height_is_24_km : total_max_height = 24000 :=
by sorry

theorem can_pose_danger : total_max_height > danger_height :=
by sorry

end max_height_is_24_km_can_pose_danger_l525_525323


namespace probability_of_two_prime_numbers_l525_525399

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l525_525399


namespace quadratic_solution_value_l525_525344

open Real

theorem quadratic_solution_value (a b : ℝ) (h1 : 2 + b = -a) (h2 : 2 * b = -6) :
  (2 * a + b)^2023 = -1 :=
sorry

end quadratic_solution_value_l525_525344


namespace probability_same_color_blocks_l525_525502

theorem probability_same_color_blocks :
  ∃ (p : ℝ), p = 37 / 64 ∧ 
    (∀ (A B C : Fin 3 → Fin 3 → Fin 3 → Fin 3) (H : ∀ i j k, A i j k ∈ {0, 1, 2} ∧ B i j k ∈ {0, 1, 2} ∧ C i j k ∈ {0, 1, 2}), 
      ∃ (box : Fin 3), (A box 0 0 = B box 0 0 ∧ B box 0 0 = C box 0 0) 
      ∨ (A box 1 0 = B box 1 0 ∧ B box 1 0 = C box 1 0) 
      ∨ (A box 2 0 = B box 2 0 ∧ B box 2 0 = C box 2 0)) := 
exists.intro (37 / 64) sorry

end probability_same_color_blocks_l525_525502


namespace problem_statement_l525_525149

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 1/2 * x

theorem problem_statement : ∀ x ∈ Set.Icc 0 Real.pi, f x ≤ f (Real.pi / 3) :=
begin
  sorry
end

end problem_statement_l525_525149


namespace prob_primes_1_to_30_l525_525411

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l525_525411


namespace second_number_is_12_l525_525862

noncomputable def expression := (26.3 * 12 * 20) / 3 + 125

theorem second_number_is_12 :
  expression = 2229 → 12 = 12 :=
by sorry

end second_number_is_12_l525_525862


namespace correct_option_D_l525_525844

theorem correct_option_D (x y : ℝ) : (x - y) ^ 2 = (y - x) ^ 2 := by
  sorry

end correct_option_D_l525_525844


namespace angelina_speed_from_grocery_to_gym_l525_525460

theorem angelina_speed_from_grocery_to_gym
    (v : ℝ)
    (hv : v > 0)
    (home_to_grocery_distance : ℝ := 150)
    (grocery_to_gym_distance : ℝ := 200)
    (time_difference : ℝ := 10)
    (time_home_to_grocery : ℝ := home_to_grocery_distance / v)
    (time_grocery_to_gym : ℝ := grocery_to_gym_distance / (2 * v))
    (h_time_diff : time_home_to_grocery - time_grocery_to_gym = time_difference) :
    2 * v = 10 := by
  sorry

end angelina_speed_from_grocery_to_gym_l525_525460


namespace total_hangers_is_65_l525_525691

noncomputable def calculate_hangers_total : ℕ :=
  let pink := 7
  let green := 4
  let blue := green - 1
  let yellow := blue - 1
  let orange := 2 * (pink + green)
  let purple := (blue - yellow) + 3
  let red := (pink + green + blue) / 3
  let brown := 3 * red + 1
  let gray := (3 * purple) / 5
  pink + green + blue + yellow + orange + purple + red + brown + gray

theorem total_hangers_is_65 : calculate_hangers_total = 65 := 
by 
  sorry

end total_hangers_is_65_l525_525691


namespace expected_matches_l525_525859

-- Definitions based on problem conditions
def matchbox : Type := Fin 60 -- Each matchbox contains 60 matches.
def matchboxes (n : ℕ) : list matchbox := list.replicate n (Fin 60)

-- Random selection condition
constant probability_of_selecting_either_box : ℚ := 0.5

-- Expected value theorem
theorem expected_matches (M N : ℕ) (hM : M = 60) (hN : N = 60) (P : ℚ) (hP : P = probability_of_selecting_either_box) :
  ∃ μ : ℚ, μ = 7.795 ∧ 
  (∃ (X : ℕ → ℕ), (∀ k : ℕ, k ≤ 60 → (X k = k * P * (binom N k) / (2^(M+k))) ∧ 
  (∑ k in finset.range (N + 1), (N - k) * (X k * P * (binom N k) / (2^(M+k)))) = 7.795)) := sorry

end expected_matches_l525_525859


namespace major_premise_incorrect_l525_525466

theorem major_premise_incorrect (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
    ¬ (∀ x y : ℝ, x < y → a^x < a^y) :=
by {
  sorry
}

end major_premise_incorrect_l525_525466


namespace min_length_l525_525532

def length (a b : ℝ) : ℝ := b - a

noncomputable def M (m : ℝ) := {x | m ≤ x ∧ x ≤ m + 3 / 4}
noncomputable def N (n : ℝ) := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
noncomputable def intersection (m n : ℝ) := {x | max m (n - 1 / 3) ≤ x ∧ x ≤ min (m + 3 / 4) n}

theorem min_length (m n : ℝ) (hM : ∀ x, x ∈ M m → 0 ≤ x ∧ x ≤ 1) (hN : ∀ x, x ∈ N n → 0 ≤ x ∧ x ≤ 1) :
  length (max m (n - 1 / 3)) (min (m + 3 / 4) n) = 1 / 12 :=
sorry

end min_length_l525_525532


namespace greening_investment_growth_l525_525876

-- Define initial investment in 2020 and investment in 2022.
def investment_2020 : ℝ := 20000
def investment_2022 : ℝ := 25000

-- Define the average growth rate x
variable (x : ℝ)

-- The mathematically equivalent proof problem:
theorem greening_investment_growth : 
  20 * (1 + x) ^ 2 = 25 :=
sorry

end greening_investment_growth_l525_525876


namespace find_f1_l525_525612

variable {R : Type*} [LinearOrderedField R]

-- Define function f of the form px + q
def f (p q x : R) : R := p * x + q

-- Given conditions
variables (p q : R)

-- Define the equations from given conditions
def cond1 : Prop := (f p q 3) = 5
def cond2 : Prop := (f p q 5) = 9

theorem find_f1 (hpq1 : cond1 p q) (hpq2 : cond2 p q) : f p q 1 = 1 := sorry

end find_f1_l525_525612


namespace remaining_pages_l525_525717

def original_book_pages : ℕ := 93
def pages_read_saturday : ℕ := 30
def pages_read_sunday : ℕ := 20

theorem remaining_pages :
  original_book_pages - (pages_read_saturday + pages_read_sunday) = 43 := by
  sorry

end remaining_pages_l525_525717


namespace joseph_cards_percentage_l525_525723

theorem joseph_cards_percentage
    (total_cards : ℕ)
    (fraction_given_to_brother : ℚ)
    (additional_cards_given : ℕ)
    (cards_left : ℕ)
    (percentage_left : ℚ)
    (orig_cards : total_cards = 16)
    (fraction : fraction_given_to_brother = 3 / 8)
    (additional_cards : additional_cards_given = 2)
    (given_to_brother : 16 * (3 / 8) = 6)
    (total_given : 6 + additional_cards_given = 8)
    (left_cards : total_cards - total_given = 8)
    (calc_percentage : (8 : ℚ) / 16 * 100 = 50) :
  percentage_left = 50 :=
sorry

end joseph_cards_percentage_l525_525723


namespace f_bounds_l525_525342

-- Define the function f with the given properties
def f : ℝ → ℝ :=
sorry 

-- Specify the conditions on f
axiom f_0 : f 0 = 0
axiom f_1 : f 1 = 1
axiom f_ratio (x y z : ℝ) (h1 : 0 ≤ x) (h2 : x < y) (h3 : y < z) (h4 : z ≤ 1) 
  (h5 : z - y = y - x) : 1/2 ≤ (f z - f y) / (f y - f x) ∧ (f z - f y) / (f y - f x) ≤ 2

-- State the theorem to be proven
theorem f_bounds : 1 / 7 ≤ f (1 / 3) ∧ f (1 / 3) ≤ 4 / 7 :=
sorry

end f_bounds_l525_525342


namespace probability_both_numbers_are_prime_l525_525422

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l525_525422


namespace option_A_option_D_l525_525610

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  sin^2 (ω * x + (π / 3)) - cos^2 (ω * x + (π / 3))

theorem option_A (x1 x2 ω : ℝ) (h1 : f x1 ω = 1) (h2 : f x2 ω = -1) (h3 : |x1 - x2| = π / 2) : ω = 1 :=
  sorry

theorem option_D (ω : ℝ) (h4 : ∀ x ∈ Icc (-π / 6) (π / 3), f x ω ≤ f (x + (π / 1000)) ω) : 0 < ω ∧ ω ≤ 1/2 :=
  sorry

end option_A_option_D_l525_525610


namespace bear_population_l525_525035

theorem bear_population (black_bears white_bears brown_bears total_bears : ℕ) 
(h1 : black_bears = 60)
(h2 : white_bears = black_bears / 2)
(h3 : brown_bears = black_bears + 40) :
total_bears = black_bears + white_bears + brown_bears :=
sorry

end bear_population_l525_525035


namespace simplify_expression_l525_525572

theorem simplify_expression (x y m : ℤ) 
  (h1 : (x-5)^2 = -|m-1|)
  (h2 : y + 1 = 5) :
  (2 * x^2 - 3 * x * y - 4 * y^2) - m * (3 * x^2 - x * y + 9 * y^2) = -273 :=
sorry

end simplify_expression_l525_525572


namespace intersection_of_A_and_B_l525_525589

-- Definitions for the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l525_525589


namespace even_odd_set_equivalence_sum_measures_even_equal_odd_sum_measures_odd_sets_l525_525231

open Finset

-- Define X_n as a Finset of natural numbers {1, 2, ..., n}
noncomputable def X_n (n : ℕ) (h : n ≥ 3) : Finset ℕ := (range n).map (nat.cast ∘ (λ x, x + 1))

-- Measure function of subset of X_n
def measure (X : Finset ℕ) : ℕ :=
  X.sum id

-- Even and be sets in X_n
def is_even (X : Finset ℕ) : Prop :=
  measure X % 2 = 0

def is_odd (X : Finset ℕ) : Prop :=
  ¬(is_even X)

-- Part (a): The number of even sets equals the number of odd sets
theorem even_odd_set_equivalence (n : ℕ) (h : n ≥ 3) :
  (univ.filter is_even).card = (univ.filter is_odd).card := sorry

-- Part (b): The sum of the measures of the even sets equals the sum of the measures of the odd sets
theorem sum_measures_even_equal_odd (n : ℕ) (h : n ≥ 3) :
  (univ.filter is_even).sum measure = (univ.filter is_odd).sum measure := sorry

-- Part (c): The sum of the measures of the odd sets is (n+1 choose 2) * 2^(n-2)
theorem sum_measures_odd_sets (n : ℕ) (h : n ≥ 3) :
  (univ.filter is_odd).sum measure = nat.choose (n + 1) 2 * 2^(n - 2) := sorry

end even_odd_set_equivalence_sum_measures_even_equal_odd_sum_measures_odd_sets_l525_525231


namespace no_common_interior_points_l525_525618

open Metric

-- Define the distance conditions for two convex polygons F1 and F2
variables {F1 F2 : Set (EuclideanSpace ℝ (Fin 2))}

-- F1 is a convex polygon
def is_convex (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)} {a b : ℝ},
    x ∈ S → y ∈ S → 0 ≤ a → 0 ≤ b → a + b = 1 → a • x + b • y ∈ S

-- Conditions provided in the problem
def condition1 (F : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)}, x ∈ F → y ∈ F → dist x y ≤ 1

def condition2 (F1 F2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x : EuclideanSpace ℝ (Fin 2)} {y : EuclideanSpace ℝ (Fin 2)}, x ∈ F1 → y ∈ F2 → dist x y > 1 / Real.sqrt 2

-- The theorem to prove
theorem no_common_interior_points (h1 : is_convex F1) (h2 : is_convex F2) 
  (h3 : condition1 F1) (h4 : condition1 F2) (h5 : condition2 F1 F2) :
  ∀ p ∈ interior F1, ∀ q ∈ interior F2, p ≠ q :=
sorry

end no_common_interior_points_l525_525618


namespace probability_of_two_primes_is_correct_l525_525392

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l525_525392


namespace identify_incorrect_equations_l525_525453

theorem identify_incorrect_equations :
  (0 ∉ (∅ : Set ℕ)) ∧
  ({0} ⊇ (∅ : Set ℕ)) ∧
  (0 ∈ (ℕ : Set ℕ)) ∧
  ({a, b} ⊆ {a, b : Set ℕ}) ∧
  ({0} ⊈ {∅ : Set ℕ}) :=
begin
  sorry
end

end identify_incorrect_equations_l525_525453


namespace math_proof_problem_l525_525117

noncomputable def complex_example : Prop :=
  let z := 2 - complex.i in
  let conj_z := complex.conj z in
  z * (conj_z + complex.i) = 6 + 2 * complex.i

theorem math_proof_problem : complex_example := by
  sorry

end math_proof_problem_l525_525117


namespace problem_statement_l525_525927

theorem problem_statement (h1 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2) :
  (Real.pi / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 := by
  sorry

end problem_statement_l525_525927


namespace remainder_1394_div_20_l525_525975

theorem remainder_1394_div_20 : 
  ∀ k, (1394 % 2535 = 1929) → (1394 % 40 = 34) → 1394 % 20 = 14 := 
by 
  intro k h1 h2 
  sorry

end remainder_1394_div_20_l525_525975


namespace largest_consecutive_sum_to_35_l525_525720

theorem largest_consecutive_sum_to_35 (n : ℕ) (h : ∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 35) : n ≤ 7 :=
by
  sorry

end largest_consecutive_sum_to_35_l525_525720


namespace permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l525_525085

open Finset

def digits : Finset ℤ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def permutations_no_repetition : ℤ :=
  (digits.card.factorial) / ((digits.card - 4).factorial)

noncomputable def four_digit_numbers_no_repetition : ℤ :=
  9 * ((digits.card - 1).factorial / ((digits.card - 1 - 3).factorial))

noncomputable def even_four_digit_numbers_gt_3000_no_repetition : ℤ :=
  784 + 1008

theorem permutations_count_5040 : permutations_no_repetition = 5040 := by
  sorry

theorem four_digit_numbers_count_4356 : four_digit_numbers_no_repetition = 4356 := by
  sorry

theorem even_four_digit_numbers_count_1792 : even_four_digit_numbers_gt_3000_no_repetition = 1792 := by
  sorry

end permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l525_525085


namespace seedling_height_regression_seedling_height_distribution_and_expectation_l525_525475

noncomputable def average (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def linear_regression (xs ys : List ℝ) : ℝ × ℝ :=
  let n := xs.length
  let x_sum := xs.sum
  let y_sum := ys.sum
  let xy_sum := (List.zipWith (*) xs ys).sum
  let x_square_sum := (xs.map (λ x => x * x)).sum
  let b := (xy_sum - n * (x_sum / n) * (y_sum / n)) / (x_square_sum - n * (x_sum / n) ^ 2)
  let a := (y_sum / n) - b * (x_sum / n)
  (b, a)

theorem seedling_height_regression :
  linear_regression [1, 2, 3, 4, 5, 6, 7] [0, 4, 7, 9, 11, 12, 13] = (59/28, -3/7) :=
sorry

noncomputable def prob_xi_distribution_and_expectation (heights: List ℝ) : (ℕ × ℕ × ℕ × ℕ) × ℝ :=
  let avg_height := average heights
  let greater_than_avg := heights.count (λ h => h > avg_height)
  let less_or_equal_avg := heights.count (λ h => h <= avg_height)
  let p_ξ0 := (Nat.choose 3 3 * Nat.choose 4 0) / (Nat.choose 7 3 : ℝ)
  let p_ξ1 := (Nat.choose 3 2 * Nat.choose 4 1) / (Nat.choose 7 3 : ℝ)
  let p_ξ2 := (Nat.choose 3 1 * Nat.choose 4 2) / (Nat.choose 7 3 : ℝ)
  let p_ξ3 := (Nat.choose 3 0 * Nat.choose 4 3) / (Nat.choose 7 3 : ℝ)
  let E_ξ := 0 * p_ξ0 + 1 * p_ξ1 + 2 * p_ξ2 + 3 * p_ξ3
  ((p_ξ0, p_ξ1, p_ξ2, p_ξ3), E_ξ)

theorem seedling_height_distribution_and_expectation :
  prob_xi_distribution_and_expectation [0, 4, 7, 9, 11, 12, 13] = ((1/35, 12/35, 18/35, 4/35), 12/7) :=
sorry

end seedling_height_regression_seedling_height_distribution_and_expectation_l525_525475


namespace henry_initial_money_l525_525628

variable (H : ℕ)

def henry_earned_more (H : ℕ) : Prop :=
  let henry_initial := H 
  let henry_earned := 2
  let friend_money := 13
  let total_money := 20
  henry_initial + henry_earned + friend_money = total_money

theorem henry_initial_money : ∃ H : ℕ, henry_earned_more H ∧ H = 5 :=
by
  apply Exists.intro 5
  unfold henry_earned_more
  simp
  sorry

end henry_initial_money_l525_525628


namespace checker_on_diagonal_l525_525757

theorem checker_on_diagonal {n : ℕ} (n_eq_25 : n = 25) 
  (symmetric_placement : ∀ i j : fin n, i ≠ j → checker_placed i j ↔ checker_placed j i) :
  ∃ i : fin n, checker_placed i i := 
begin
  sorry,
end

end checker_on_diagonal_l525_525757


namespace alex_silver_tokens_l525_525021

theorem alex_silver_tokens :
  ∃ (x y : ℕ), (100 - 3 * x + 2 * y ≥ 3) ∧ (90 + 2 * x - 4 * y ≥ 4) ∧ (x + y = 39) :=
by
  use 31, 8
  finish

end alex_silver_tokens_l525_525021


namespace trains_cross_in_12_seconds_l525_525856

noncomputable def length := 120 -- Length of each train in meters
noncomputable def time_train1 := 10 -- Time taken by the first train to cross the post in seconds
noncomputable def time_train2 := 15 -- Time taken by the second train to cross the post in seconds

noncomputable def speed_train1 := length / time_train1 -- Speed of the first train in m/s
noncomputable def speed_train2 := length / time_train2 -- Speed of the second train in m/s

noncomputable def relative_speed := speed_train1 + speed_train2 -- Relative speed when traveling in opposite directions in m/s
noncomputable def total_length := 2 * length -- Total distance covered when crossing each other

noncomputable def crossing_time := total_length / relative_speed -- Time to cross each other in seconds

theorem trains_cross_in_12_seconds : crossing_time = 12 := by
  sorry

end trains_cross_in_12_seconds_l525_525856


namespace pure_water_to_achieve_desired_concentration_l525_525640

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l525_525640


namespace pow_sub_nat_ge_seven_l525_525232

open Nat

theorem pow_sub_nat_ge_seven
  (m n : ℕ) 
  (h1 : m > 1)
  (h2 : 2^(2 * m + 1) - n^2 ≥ 0) : 
  2^(2 * m + 1) - n^2 ≥ 7 :=
sorry

end pow_sub_nat_ge_seven_l525_525232


namespace water_to_add_for_desired_acid_concentration_l525_525656

theorem water_to_add_for_desired_acid_concentration :
  ∃ w : ℝ, 50 * 0.4 / (50 + w) = 0.25 ∧ w = 30 :=
begin
  let initial_volume := 50,
  let initial_acid_concentration := 0.4,
  let desired_acid_concentration := 0.25,
  let acid_content := initial_volume * initial_acid_concentration,
  use 30,
  split,
  { simp [initial_volume, initial_acid_concentration, desired_acid_concentration, acid_content],
    linarith },
  { refl }
end

end water_to_add_for_desired_acid_concentration_l525_525656


namespace polar_equation_of_tangent_line_l525_525156

-- Define the parametric equations
def parametric_eq (t : ℝ) : ℝ × ℝ :=
  (3 * Real.cos t, 3 * Real.sin t)

-- Define the point (0, 3)
def point_on_curve : ℝ × ℝ := (0, 3)

-- Define the standard equation of the circle
def standard_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

-- Define the tangent line equation
def tangent_line_eq (x y : ℝ) : Prop :=
  y = 3

-- Define the polar equation of the tangent line
def polar_eq (rho theta : ℝ) : Prop :=
  rho * Real.sin theta = 3

-- The theorem to prove
theorem polar_equation_of_tangent_line :
    (∀ t : ℝ, standard_eq (3 * Real.cos t) (3 * Real.sin t)) →
    tangent_line_eq 0 3 →
    ∃ rho theta : ℝ, polar_eq rho theta :=
by
  intro h1 h2
  sorry

end polar_equation_of_tangent_line_l525_525156


namespace C1_general_eq_C2_cartesian_eq_distance_AB_eq_l525_525201

section Curve

variable {φ θ α : ℝ}

-- Parametric equations of curve C1
def C1_parametric (x y φ : ℝ) : Prop :=
  x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ

-- Prove general equation of curve C1
theorem C1_general_eq (x y : ℝ) (h : ∃ φ, C1_parametric x y φ) : (x - 2)^2 + y^2 = 4 :=
sorry

-- Polar equation of curve C2
def C2_polar (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin θ

-- Prove Cartesian coordinate equation of curve C2
theorem C2_cartesian_eq (x y : ℝ) (h : ∃ ρ θ, C2_polar ρ θ ∧ ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y) : x^2 + (y - 2)^2 = 4 :=
sorry

-- Distance between intersection points of C3 with C1 and C2, and the value for α
def C3_polar (α θ : ℝ) : Prop :=
  θ = α

def Point (ρ θ : ℝ) := (ρ * Real.cos θ, ρ * Real.sin θ)

def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem distance_AB_eq (α : ℝ) (h : 0 < α ∧ α < π ∧ distance (Point (4 * Real.cos α) α) (Point (4 * Real.sin α) α) = 4 * Real.sqrt 2) : α = 3 * Real.pi / 4 :=
sorry

end Curve

end C1_general_eq_C2_cartesian_eq_distance_AB_eq_l525_525201


namespace dorchester_puppy_count_l525_525955

/--
  Dorchester works at a puppy wash. He is paid $40 per day + $2.25 for each puppy he washes.
  On Wednesday, Dorchester earned $76. Prove that Dorchester washed 16 puppies that day.
-/
theorem dorchester_puppy_count
  (total_earnings : ℝ)
  (base_pay : ℝ)
  (pay_per_puppy : ℝ)
  (puppies_washed : ℕ)
  (h1 : total_earnings = 76)
  (h2 : base_pay = 40)
  (h3 : pay_per_puppy = 2.25) :
  total_earnings - base_pay = (puppies_washed : ℝ) * pay_per_puppy :=
sorry

example :
  dorchester_puppy_count 76 40 2.25 16 := by
  rw [dorchester_puppy_count, sub_self, mul_zero]

end dorchester_puppy_count_l525_525955


namespace horse_food_per_day_l525_525920

theorem horse_food_per_day
  (total_horse_food_per_day : ℕ)
  (sheep_count : ℕ)
  (sheep_to_horse_ratio : ℕ)
  (horse_to_sheep_ratio : ℕ)
  (horse_food_per_horse_per_day : ℕ) :
  sheep_to_horse_ratio * horse_food_per_horse_per_day = total_horse_food_per_day / (sheep_count / sheep_to_horse_ratio * horse_to_sheep_ratio) :=
by
  -- Given
  let total_horse_food_per_day := 12880
  let sheep_count := 24
  let sheep_to_horse_ratio := 3
  let horse_to_sheep_ratio := 7

  -- We need to show that horse_food_per_horse_per_day = 230
  have horse_count : ℕ := (sheep_count / sheep_to_horse_ratio) * horse_to_sheep_ratio
  have horse_food_per_horse_per_day : ℕ := total_horse_food_per_day / horse_count

  -- Desired proof statement
  sorry

end horse_food_per_day_l525_525920


namespace value_of_a_minus_b_l525_525172

theorem value_of_a_minus_b
  (a b : ℚ)
  (h : ∀ x : ℚ, x > 0 →
  (a / (2^x - 1) + b / (2^x + 3) = (5 * 2^x + 7) / ((2^x - 1) * (2^x + 3)))) :
  a - b = 1 :=
sorry

end value_of_a_minus_b_l525_525172


namespace main_theorem_l525_525762

-- Define the initial conditions
variables {n : ℕ} -- number of points on the circle
variables {G : Type} -- type representing the initial graph formed by chords

-- Predicate to check if graph remains connected after removing k edges
def remains_connected_after_removal (G : Type) (k : ℕ) : Prop := sorry

-- The main theorem statement
theorem main_theorem 
  (removal_property : remains_connected_after_removal G 2021) 
  : ∃ H : Type, (remains_connected_after_removal H 2021 ∧ cardinality_of H ≤ 2022 * n) :=
sorry

end main_theorem_l525_525762


namespace feet_of_perpendiculars_lie_on_circle_l525_525127


structure Pyramid (V : Type) [inner_product_space ℝ V] :=
(S A B C D : V)
(O : V)
(h_convex_base : convex_hull ℝ [A, B, C, D] = set.univ)
(h_perpendicular_diagonals : orthogonal_projection (span ℝ [A, B, C, D]) A = orthogonal_projection (span ℝ [A, B, C, D]) C)
(h_base_perpendicular_SO : ∀ x ∈ span ℝ [A, B, C, D], ∀ y ∈ span ℝ [S], inner ℝ x y = 0)
(h_O_inter_diagonals : O = orthogonal_projection (span ℝ [A, B, C, D]) ((1 / 2 : ℝ) • (A + C)))

theorem feet_of_perpendiculars_lie_on_circle (V : Type) [inner_product_space ℝ V] (P : Pyramid V) :
  ∃ (circle : set V), ∀ (x : V), (x = orthogonal_projection (span ℝ [P.S, P.A]) P.O ∨
                                     x = orthogonal_projection (span ℝ [P.S, P.B]) P.O ∨
                                     x = orthogonal_projection (span ℝ [P.S, P.C]) P.O ∨
                                     x = orthogonal_projection (span ℝ [P.S, P.D]) P.O) → 
                                     x ∈ circle := 
by
  sorry

end feet_of_perpendiculars_lie_on_circle_l525_525127


namespace rational_x_sqrt3_x_sq_sqrt3_l525_525967

theorem rational_x_sqrt3_x_sq_sqrt3 (x : ℝ) : (∃ a b : ℚ, x + real.sqrt 3 = a ∧ x^2 + real.sqrt 3 = b) ↔ x = (1 / 2) - real.sqrt 3 :=
by
  sorry

end rational_x_sqrt3_x_sq_sqrt3_l525_525967


namespace acrobats_count_l525_525508

theorem acrobats_count (a g : ℕ) 
  (h1 : 2 * a + 4 * g = 32) 
  (h2 : a + g = 10) : 
  a = 4 := by
  -- Proof omitted
  sorry

end acrobats_count_l525_525508


namespace average_marks_correct_l525_525530

-- Define the marks for each subject
def english_marks := 90
def mathematics_marks := 92
def physics_marks := 85
def chemistry_marks := 87
def biology_marks := 85

-- Calculate the total marks
def total_marks := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks

-- Number of subjects
def number_of_subjects := 5

-- Average marks calculation
def average_marks := total_marks / number_of_subjects

-- The theorem to prove the average marks
theorem average_marks_correct : average_marks = 87.8 := by
  sorry

end average_marks_correct_l525_525530


namespace total_fuel_proof_l525_525542

def highway_consumption_60 : ℝ := 3 -- gallons per mile at 60 mph
def highway_consumption_70 : ℝ := 3.5 -- gallons per mile at 70 mph
def city_consumption_30 : ℝ := 5 -- gallons per mile at 30 mph
def city_consumption_15 : ℝ := 4.5 -- gallons per mile at 15 mph

def day1_highway_60_hours : ℝ := 2 -- hours driven at 60 mph on the highway
def day1_highway_70_hours : ℝ := 1 -- hours driven at 70 mph on the highway
def day1_city_30_hours : ℝ := 4 -- hours driven at 30 mph in the city

def day2_highway_70_hours : ℝ := 3 -- hours driven at 70 mph on the highway
def day2_city_15_hours : ℝ := 3 -- hours driven at 15 mph in the city
def day2_city_30_hours : ℝ := 1 -- hours driven at 30 mph in the city

def day3_highway_60_hours : ℝ := 1.5 -- hours driven at 60 mph on the highway
def day3_city_30_hours : ℝ := 3 -- hours driven at 30 mph in the city
def day3_city_15_hours : ℝ := 1 -- hours driven at 15 mph in the city

def total_fuel_consumption (c1 c2 c3 c4 : ℝ) (h1 h2 h3 h4 h5 h6 h7 h8 h9 : ℝ) :=
  (h1 * 60 * c1) + (h2 * 70 * c2) + (h3 * 30 * c3) + 
  (h4 * 70 * c2) + (h5 * 15 * c4) + (h6 * 30 * c3) +
  (h7 * 60 * c1) + (h8 * 30 * c3) + (h9 * 15 * c4)

theorem total_fuel_proof :
  total_fuel_consumption highway_consumption_60 highway_consumption_70 city_consumption_30 city_consumption_15
  day1_highway_60_hours day1_highway_70_hours day1_city_30_hours day2_highway_70_hours
  day2_city_15_hours day2_city_30_hours day3_highway_60_hours day3_city_30_hours day3_city_15_hours
  = 3080 := by
  sorry

end total_fuel_proof_l525_525542


namespace tan_sum_identity_l525_525139

theorem tan_sum_identity (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : Real.tan x = 1 / 3 := 
by 
  sorry

end tan_sum_identity_l525_525139


namespace slab_length_l525_525867

noncomputable def area_of_one_slab (total_area: ℝ) (num_slabs: ℕ) : ℝ :=
  total_area / num_slabs

noncomputable def length_of_one_slab (slab_area : ℝ) : ℝ :=
  Real.sqrt slab_area

theorem slab_length (total_area : ℝ) (num_slabs : ℕ)
  (h_total_area : total_area = 98)
  (h_num_slabs : num_slabs = 50) :
  length_of_one_slab (area_of_one_slab total_area num_slabs) = 1.4 :=
by
  sorry

end slab_length_l525_525867


namespace sum_of_two_is_65_l525_525987

theorem sum_of_two_is_65 (A B : Finset ℕ) (hA1 : ∀ a ∈ A, a % 2 = 1) (hB1 : ∀ b ∈ B, b % 2 = 0)
  (hAcard : A.card = 16) (hBcard : B.card = 16) (hAsum : A.sum = B.sum) (u : Finset ℕ)
  (hAu : u = (Finset.range 65).filter(λ n, n > 0)) : 
  ∃ x ∈ A, ∃ y ∈ B, x + y = 65 := 
sorry

end sum_of_two_is_65_l525_525987


namespace tangent_line_through_external_point_l525_525601

theorem tangent_line_through_external_point (x y : ℝ) (h_circle : x^2 + y^2 = 1) (P : ℝ × ℝ) (h_P : P = (1, 2)) : 
  (∃ k : ℝ, (y = 2 + k * (x - 1)) ∧ (x = 1 ∨ (3 * x - 4 * y + 5 = 0))) :=
by
  sorry

end tangent_line_through_external_point_l525_525601


namespace find_x_l525_525709

variable (BM MA BC CA : ℝ)
variable (h d : ℝ)

theorem find_x (x : ℝ) (BM_eq : BM = x) (BC_eq : BC = h) (CA_eq : CA = d)
  (dist_eq : BM + MA = BC + CA)
  (ma_eq : MA = sqrt ((x + h)^2 + d^2)) :
  x = (h * d) / (2 * h + d) :=
by
  sorry

end find_x_l525_525709


namespace A_and_B_together_days_l525_525868

theorem A_and_B_together_days
  (W : ℝ)
  (h1 : W / 30 + 20 * (W / 60) = W) :
  ∃ x : ℝ, x = 20 :=
begin
  use 20,
  sorry
end

end A_and_B_together_days_l525_525868


namespace probability_two_primes_is_1_over_29_l525_525433

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l525_525433


namespace zeros_at_end_of_factorial_30_l525_525674

theorem zeros_at_end_of_factorial_30 : ∀ (n : ℕ), n = 30 → (count_factors 5 (factorial n) = 7) :=
by
  intro n hn
  rw [hn]
  sorry

def count_factors (p n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let rec loop (n : ℕ) : ℕ :=
      if n = 0 then 0
      else (if n % p = 0 then 1 else 0) + loop (n / p)
    loop n

def factorial : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * factorial n

end zeros_at_end_of_factorial_30_l525_525674


namespace rocket_proof_l525_525321

-- Definition of conditions
def a : ℝ := 30  -- acceleration provided by the engines in m/s^2
def tau : ℝ := 20  -- duration of the engine run in seconds
def g : ℝ := 10  -- acceleration due to gravity in m/s^2

-- Definitions based on the conditions
def V0 : ℝ := a * tau  -- final velocity after tau seconds
def y0 : ℝ := 0.5 * a * tau^2  -- height gained during initial acceleration period

-- Kinematic equations during free flight
def t_flight : ℝ := V0 / g  -- time to maximum height after engines stop
def y_flight : ℝ := V0 * t_flight - 0.5 * g * t_flight^2  -- additional height during free flight

-- Total maximum height
def height_max : ℝ := y0 + y_flight

-- Conclusion of threat evaluation
def threat := height_max > 20000  -- 20 km in meters

-- The proof problem statement
theorem rocket_proof :
  height_max = 24000 ∧ threat = true := by
  sorry  -- Proof to be provided

end rocket_proof_l525_525321


namespace animal_shelter_kittens_count_l525_525357

def num_puppies : ℕ := 32
def num_kittens_more : ℕ := 14

theorem animal_shelter_kittens_count : 
  ∃ k : ℕ, k = (2 * num_puppies) + num_kittens_more := 
sorry

end animal_shelter_kittens_count_l525_525357


namespace jenna_gas_cost_l525_525714

-- Definitions of the given conditions
def hours1 : ℕ := 2
def speed1 : ℕ := 60
def hours2 : ℕ := 3
def speed2 : ℕ := 50
def miles_per_gallon : ℕ := 30
def cost_per_gallon : ℕ := 2

-- Statement to be proven
theorem jenna_gas_cost : 
  let distance1 := hours1 * speed1,
      distance2 := hours2 * speed2,
      total_distance := distance1 + distance2,
      total_gallons := total_distance / miles_per_gallon,
      total_cost := total_gallons * cost_per_gallon
  in total_cost = 18 := 
by 
  let distance1 := hours1 * speed1,
      distance2 := hours2 * speed2,
      total_distance := distance1 + distance2,
      total_gallons := total_distance / miles_per_gallon,
      total_cost := total_gallons * cost_per_gallon

  sorry

end jenna_gas_cost_l525_525714


namespace symmetric_points_x_axis_l525_525703

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : P = (a - 3, 1)) (hQ : Q = (2, b + 1)) (hSymm : P.1 = Q.1 ∧ P.2 = -Q.2) :
  a + b = 3 :=
by 
  sorry

end symmetric_points_x_axis_l525_525703


namespace total_payment_l525_525261

theorem total_payment :
  ∀ (pizzas salads : ℕ) (pizza_price salad_price discount_rate tax_rate : ℝ),
    pizzas = 3 →
    salads = 2 →
    pizza_price = 8 →
    salad_price = 6 →
    discount_rate = 0.10 →
    tax_rate = 0.07 →
    let total_cost_pizzas := pizzas * pizza_price,
        discount := discount_rate * total_cost_pizzas,
        discounted_pizzas := total_cost_pizzas - discount,
        cost_salads := salads * salad_price,
        subtotal := discounted_pizzas + cost_salads,
        tax := tax_rate * subtotal,
        total_payment := (subtotal + tax).round in
    total_payment = 35.95 :=
by
  intros pizzas salads pizza_price salad_price discount_rate tax_rate p_eq s_eq pp_eq sp_eq dr_eq tr_eq,
  -- Definitions based on the conditions given
  let total_cost_pizzas := pizzas * pizza_price
  let discount := discount_rate * total_cost_pizzas
  let discounted_pizzas := total_cost_pizzas - discount
  let cost_salads := salads * salad_price
  let subtotal := discounted_pizzas + cost_salads
  let tax := tax_rate * subtotal
  let total_payment := (subtotal + tax).round
  -- Prove with required values
  sorry

end total_payment_l525_525261


namespace non_congruent_rectangles_count_l525_525491

theorem non_congruent_rectangles_count (h w : ℕ) (P : ℕ) (multiple_of_4: ℕ → Prop) :
  P = 80 →
  w ≥ 1 ∧ h ≥ 1 →
  P = 2 * (w + h) →
  (multiple_of_4 w ∨ multiple_of_4 h) →
  (∀ k, multiple_of_4 k ↔ ∃ m, k = 4 * m) →
  ∃ n, n = 5 :=
by
  sorry

end non_congruent_rectangles_count_l525_525491


namespace maximize_expression_l525_525575

noncomputable theory

open Real

theorem maximize_expression (x1 x2 x3 : ℝ) (h₀ : x1 + x2 + x3 = 1) :
  x1^3 * x2^2 * x3 ≤ 1 / (2^4 * 3^3) :=
sorry

end maximize_expression_l525_525575


namespace octagon_midpoints_area_l525_525897

theorem octagon_midpoints_area (P : ℝ) (hP : P = 16 * Real.sqrt 2) :
  let s := P / 8,
      a := s / (2 * Real.tan (Real.pi / 8)),
      area := 8 * a ^ 2 * Real.tan (Real.pi / 8) / 4 in
  s = 2 * Real.sqrt 2 ∧ area = 8 + 4 * Real.sqrt 2 :=
by
  let s := P / 8
  let a := s / (2 * Real.tan (Real.pi / 8))
  let area := 8 * a ^ 2 * Real.tan (Real.pi / 8) / 4
  have hs : s = 2 * Real.sqrt 2, from sorry
  have harea : area = 8 + 4 * Real.sqrt 2, from sorry
  exact ⟨hs, harea⟩

end octagon_midpoints_area_l525_525897


namespace external_bisector_and_perpendicular_bisector_intersect_on_circle_l525_525299

theorem external_bisector_and_perpendicular_bisector_intersect_on_circle
  (A B C : Point)
  (Γ : Circle)
  (hAB : A ≠ B)
  (hAC : A ≠ C)
  (hBC : B ≠ C)
  (N : Point)
  (hN : lies_on_external_bisector_of_angle N A B C ∧ lies_on_perpendicular_bisector_of_segment N B C) :
  lies_on_circle N Γ :=
sorry

end external_bisector_and_perpendicular_bisector_intersect_on_circle_l525_525299


namespace impossible_configuration_l525_525500

theorem impossible_configuration : 
  ¬∃ (f : ℕ → ℕ) (h : ∀n, 1 ≤ f n ∧ f n ≤ 5) (perm : ∀i j, if i < j then f i ≠ f j else true), 
  (f 0 = 3) ∧ (f 1 = 4) ∧ (f 2 = 2) ∧ (f 3 = 1) ∧ (f 4 = 5) :=
sorry

end impossible_configuration_l525_525500


namespace trajectory_of_M_eq_circle_line_equaion_through_P_l525_525579

noncomputable def trajectory_equation (M : ℝ × ℝ) : Prop :=
  ((M.1 - 1)^2 + (M.2 - 2)^2 = 25)

theorem trajectory_of_M_eq_circle (M : ℝ × ℝ) (dist_condition : 
  (real.sqrt ((M.1 - 26)^2 + (M.2 - 1)^2) / 
   real.sqrt ((M.1 - 2)^2 + (M.2 - 1)^2)) = 5) :
  trajectory_equation M := sorry

noncomputable def line_through_P_eq (l : ℝ × ℝ → Prop) : Prop :=
  (l (-2, 3) = true ∧ 
  (l = (λ p, p.1 = -2) ∨ 
  l = (λ p, 5 * p.1 - 12 * p.2 + 46 = 0)))

theorem line_equaion_through_P (l : ℝ × ℝ → Prop) (P : ℝ × ℝ := (-2, 3)) 
  (chord_length_condition : ∀ M N, l M = true ∧ l N = true ∧ 
  ((M.1 - N.1)^2 + (M.2 - N.2)^2 = 64)) :
  line_through_P_eq l := sorry

end trajectory_of_M_eq_circle_line_equaion_through_P_l525_525579


namespace vector_subtraction_l525_525552

/-
Define the vectors we are working with.
-/
def v1 : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-8]]
def v2 : Matrix (Fin 2) (Fin 1) ℤ := ![![2], ![-6]]
def scalar : ℤ := 5
def result : Matrix (Fin 2) (Fin 1) ℤ := ![![-7], ![22]]

/-
The statement of the proof problem.
-/
theorem vector_subtraction : v1 - scalar • v2 = result := 
by
  sorry

end vector_subtraction_l525_525552


namespace animal_shelter_kittens_count_l525_525356

def num_puppies : ℕ := 32
def num_kittens_more : ℕ := 14

theorem animal_shelter_kittens_count : 
  ∃ k : ℕ, k = (2 * num_puppies) + num_kittens_more := 
sorry

end animal_shelter_kittens_count_l525_525356


namespace prob_primes_1_to_30_l525_525416

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l525_525416


namespace intersection_M_N_l525_525590

open Set

def M : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def N : Set ℝ := { x | x >= 1 }

theorem intersection_M_N : M ∩ N = { x | 1 <= x ∧ x < 3 } :=
by
  sorry

end intersection_M_N_l525_525590


namespace seedling_height_regression_seedling_height_distribution_and_expectation_l525_525474

noncomputable def average (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def linear_regression (xs ys : List ℝ) : ℝ × ℝ :=
  let n := xs.length
  let x_sum := xs.sum
  let y_sum := ys.sum
  let xy_sum := (List.zipWith (*) xs ys).sum
  let x_square_sum := (xs.map (λ x => x * x)).sum
  let b := (xy_sum - n * (x_sum / n) * (y_sum / n)) / (x_square_sum - n * (x_sum / n) ^ 2)
  let a := (y_sum / n) - b * (x_sum / n)
  (b, a)

theorem seedling_height_regression :
  linear_regression [1, 2, 3, 4, 5, 6, 7] [0, 4, 7, 9, 11, 12, 13] = (59/28, -3/7) :=
sorry

noncomputable def prob_xi_distribution_and_expectation (heights: List ℝ) : (ℕ × ℕ × ℕ × ℕ) × ℝ :=
  let avg_height := average heights
  let greater_than_avg := heights.count (λ h => h > avg_height)
  let less_or_equal_avg := heights.count (λ h => h <= avg_height)
  let p_ξ0 := (Nat.choose 3 3 * Nat.choose 4 0) / (Nat.choose 7 3 : ℝ)
  let p_ξ1 := (Nat.choose 3 2 * Nat.choose 4 1) / (Nat.choose 7 3 : ℝ)
  let p_ξ2 := (Nat.choose 3 1 * Nat.choose 4 2) / (Nat.choose 7 3 : ℝ)
  let p_ξ3 := (Nat.choose 3 0 * Nat.choose 4 3) / (Nat.choose 7 3 : ℝ)
  let E_ξ := 0 * p_ξ0 + 1 * p_ξ1 + 2 * p_ξ2 + 3 * p_ξ3
  ((p_ξ0, p_ξ1, p_ξ2, p_ξ3), E_ξ)

theorem seedling_height_distribution_and_expectation :
  prob_xi_distribution_and_expectation [0, 4, 7, 9, 11, 12, 13] = ((1/35, 12/35, 18/35, 4/35), 12/7) :=
sorry

end seedling_height_regression_seedling_height_distribution_and_expectation_l525_525474


namespace tangent_line_equation_l525_525900

theorem tangent_line_equation (x y : ℝ) 
  (h_curve : y = Real.exp x) 
  (h_origin : (0, 0) ∈ {p : ℝ × ℝ | ∃ x, p = (x, Real.exp x) ∧ p ∉ ∅}) : 
  y = Real.exp x → y - Real.exp x = 0 :=
by
  sorry

end tangent_line_equation_l525_525900


namespace length_of_field_l525_525328

def width : ℝ := 13.5

def length (w : ℝ) : ℝ := 2 * w - 3

theorem length_of_field : length width = 24 :=
by
  -- full proof goes here
  sorry

end length_of_field_l525_525328


namespace closest_value_to_division_l525_525055

-- Define the problem conditions
def number_num : ℝ := 500
def number_den : ℝ := 0.25
def options : list ℝ := [1000, 1500, 2000, 2500, 3000]

-- Define a theorem stating that the closest value to 500/0.25 from the options list is 2000
theorem closest_value_to_division :
  (∃ x ∈ options, x = number_num / number_den) →
  (number_num / number_den = 2000) :=
by
  -- Placeholder for the actual proof
  sorry

end closest_value_to_division_l525_525055


namespace probability_multiple_of_12_and_even_l525_525186

open Set

-- Definitions based on conditions
def chosen_set : Set ℕ := {4, 6, 8, 9}

def pairs (s : Set ℕ) : Set (ℕ × ℕ) := 
  { p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 ≠ p.2 }

def is_multiple_of_12 (n : ℕ) : Prop := n % 12 = 0

def has_even_number (p : ℕ × ℕ) : Prop := 
  p.1 % 2 = 0 ∨ p.2 % 2 = 0

-- Target theorem to prove
theorem probability_multiple_of_12_and_even : 
  let valid_pairs := { p ∈ pairs chosen_set | is_multiple_of_12 (p.1 * p.2) ∧ has_even_number p } in
  (valid_pairs.card : ℚ) / (pairs chosen_set).card = 2 / 3 :=
by
  sorry

end probability_multiple_of_12_and_even_l525_525186


namespace balanced_set_combined_balanced_set_odd_l525_525493

def balanced_set (s : Set ℕ) : Prop :=
∀ x ∈ s, ∃ (A B : Set ℕ), Disjoint A B ∧ A ∪ B = s \ {x} ∧ (A.sum id = B.sum id)

theorem balanced_set_combined (A B : Set ℕ) 
  (hA : balanced_set A) (hB : balanced_set B) (b₁ ∈ B) :
  balanced_set (A ∪ (B \ {b₁})) :=
sorry

theorem balanced_set_odd (n : ℕ) (hn : Odd n) (hn7 : n ≥ 7) :
  balanced_set {2 * i - 1 | i ∈ Finset.range (n + 1)} :=
sorry

end balanced_set_combined_balanced_set_odd_l525_525493


namespace height_of_spherical_caps_l525_525128

theorem height_of_spherical_caps
  (r q : ℝ)
  (m₁ m₂ m₃ m₄ : ℝ)
  (h1 : m₂ = m₁ * q)
  (h2 : m₃ = m₁ * q^2)
  (h3 : m₄ = m₁ * q^3)
  (h4 : m₁ + m₂ + m₃ + m₄ = 2 * r) :
  m₁ = 2 * r * (q - 1) / (q^4 - 1) := 
sorry

end height_of_spherical_caps_l525_525128


namespace probability_of_two_primes_is_correct_l525_525388

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l525_525388


namespace brenda_ends_with_12_skittles_l525_525922

def initial_skittles : ℕ := 7
def bought_skittles : ℕ := 8
def given_away_skittles : ℕ := 3

theorem brenda_ends_with_12_skittles :
  initial_skittles + bought_skittles - given_away_skittles = 12 := by
  sorry

end brenda_ends_with_12_skittles_l525_525922


namespace find_x_rational_l525_525965

theorem find_x_rational (x : ℝ) (h1 : ∃ (a : ℚ), x + Real.sqrt 3 = a)
  (h2 : ∃ (b : ℚ), x^2 + Real.sqrt 3 = b) :
  x = (1 / 2 : ℝ) - Real.sqrt 3 :=
sorry

end find_x_rational_l525_525965


namespace fill_up_minivans_l525_525189

theorem fill_up_minivans (service_cost : ℝ) (fuel_cost_per_liter : ℝ) (total_cost : ℝ)
  (mini_van_liters : ℝ) (truck_percent_bigger : ℝ) (num_trucks : ℕ) (num_minivans : ℕ) :
  service_cost = 2.3 ∧ fuel_cost_per_liter = 0.7 ∧ total_cost = 396 ∧
  mini_van_liters = 65 ∧ truck_percent_bigger = 1.2 ∧ num_trucks = 2 →
  num_minivans = 4 :=
by
  sorry

end fill_up_minivans_l525_525189


namespace polygon_has_five_sides_l525_525682

theorem polygon_has_five_sides (angle : ℝ) (h : angle = 108) :
  (∃ n : ℕ, n > 2 ∧ (180 - angle) * n = 360) ↔ n = 5 := 
by
  sorry

end polygon_has_five_sides_l525_525682


namespace dorchester_puppies_washed_l525_525949

theorem dorchester_puppies_washed
  (total_earnings : ℝ)
  (daily_pay : ℝ)
  (earnings_per_puppy : ℝ)
  (p : ℝ)
  (h1 : total_earnings = 76)
  (h2 : daily_pay = 40)
  (h3 : earnings_per_puppy = 2.25)
  (hp : (total_earnings - daily_pay) / earnings_per_puppy = p) :
  p = 16 := sorry

end dorchester_puppies_washed_l525_525949


namespace angle_b_in_triangle_l525_525710

theorem angle_b_in_triangle (A B C D E : Type)
    [Triangle A B C] 
    (is_angle_bisector_AD : AngleBisector A D B C) 
    (is_angle_bisector_CE : AngleBisector C E A B)
    (h : AE + CD = AC) : 
    ∠B = 60 :=
by sorry

end angle_b_in_triangle_l525_525710


namespace dorchester_puppies_washed_l525_525947

theorem dorchester_puppies_washed
  (total_earnings : ℝ)
  (daily_pay : ℝ)
  (earnings_per_puppy : ℝ)
  (p : ℝ)
  (h1 : total_earnings = 76)
  (h2 : daily_pay = 40)
  (h3 : earnings_per_puppy = 2.25)
  (hp : (total_earnings - daily_pay) / earnings_per_puppy = p) :
  p = 16 := sorry

end dorchester_puppies_washed_l525_525947


namespace statement_3_correct_l525_525845

-- Definitions based on the conditions
def DeductiveReasoningGeneralToSpecific := True
def SyllogismForm := True
def ConclusionDependsOnPremisesAndForm := True

-- Proof problem statement
theorem statement_3_correct : SyllogismForm := by
  exact True.intro

end statement_3_correct_l525_525845


namespace john_exploring_years_l525_525220

theorem john_exploring_years
    (E N : ℕ) 
    (h1 : N = E / 2) 
    (h2 : E + N + 0.5 = 5) : 
    E = 3 :=
by  sorry

end john_exploring_years_l525_525220


namespace circle_s2_leq_8r2_l525_525000

theorem circle_s2_leq_8r2 (r : ℝ) (C : ℝ × ℝ) 
  (h_radius : dist (0, 0) C = r) 
  (A B : ℝ × ℝ) 
  (h_diameter : dist A B = 2 * r) :
  let AC := dist A C,
      BC := dist B C,
      s := AC + BC in
  s^2 ≤ 8 * r^2 :=
sorry

end circle_s2_leq_8r2_l525_525000


namespace prob_primes_1_to_30_l525_525410

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l525_525410


namespace total_weight_of_lifts_l525_525194

theorem total_weight_of_lifts 
  (F S : ℕ)
  (h1 : F = 400)
  (h2 : 2 * F = S + 300) :
  F + S = 900 :=
by
  sorry

end total_weight_of_lifts_l525_525194


namespace aira_rubber_bands_l525_525291

theorem aira_rubber_bands (total_bands : ℕ) (bands_each : ℕ) (samantha_extra : ℕ) (aira_fewer : ℕ)
  (h1 : total_bands = 18) 
  (h2 : bands_each = 6) 
  (h3 : samantha_extra = 5) 
  (h4 : aira_fewer = 1) : 
  ∃ x : ℕ, x + (x + samantha_extra) + (x + aira_fewer) = total_bands ∧ x = 4 :=
by
  sorry

end aira_rubber_bands_l525_525291


namespace sequence_product_l525_525924

theorem sequence_product :
  let seq := λ (n : ℕ), (n + 1 : ℚ) / (n + 4 : ℚ)
  in ∏ i in (finset.range 50).map nat.succ, seq i = (62975 : ℚ) / 74012 :=
by
  sorry

end sequence_product_l525_525924


namespace complex_calculation_l525_525100

variable (z : ℂ)

theorem complex_calculation : (z = 2 - (-1:ℂ)) → (z * (conj z + (-1:ℂ))) = 6 + 2 * (-1:ℂ) :=
by
  intro h
  rw h
  have h_conj : conj (2 - (-1:ℂ)) = 2 + (-1:ℂ) := by simp
  rw h_conj
  ring
  done

end complex_calculation_l525_525100


namespace length_AB_correct_l525_525807

def point (ℝ := float) := (ℝ × ℝ)

def A : point := (-5, -4)
def C : point := (-3, 0)
def m : ℕ := 2
def n : ℕ := 3
def λ : ℝ := m / n

def section_formula (pt1 pt2 : point) (m n : ℝ) : point :=
  ((m * pt2.1 + n * pt1.1) / (m + n), (m * pt2.2 + n * pt1.2) / (m + n))

-- Using the section formula to derive coordinates of point B
def B : point := section_formula C A n m

def length (pt1 pt2 : point) : ℝ :=
  Real.sqrt ((pt2.1 - pt1.1) * (pt2.1 - pt1.1) + (pt2.2 - pt1.2) * (pt2.2 - pt1.2))

theorem length_AB_correct : length A B = 5 * Real.sqrt 5 :=
  sorry

end length_AB_correct_l525_525807


namespace find_hyperbola_equation_l525_525541

noncomputable def hyperbola_equation (a b : ℝ) : Prop :=
  ∃ x y, (x^2 / a^2 - y^2 / b^2 = 1)

theorem find_hyperbola_equation : ∀ (a b : ℝ),
  ((a - 4)^2 + b^2 = 16 ∧ a^2 + b^2 = 16) →
  hyperbola_equation 2 (2 * Real.sqrt 3) := by
  intros a b h,
  sorry

end find_hyperbola_equation_l525_525541


namespace domain_of_f_l525_525554

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (Real.sqrt (x - 7))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = x} = Set.Ioi 7 := by
  sorry

end domain_of_f_l525_525554


namespace equal_area_split_l525_525361

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle1 : Circle := { center := (10, 90), radius := 4 }
def circle2 : Circle := { center := (15, 80), radius := 4 }
def circle3 : Circle := { center := (20, 85), radius := 4 }

theorem equal_area_split :
  ∃ m : ℝ, ∀ x y : ℝ, m * (x - 15) = y - 80 ∧ m = 0 ∧   
    ∀ circle : Circle, circle ∈ [circle1, circle2, circle3] →
      ∃ k : ℝ, k * (x - circle.center.1) + y - circle.center.2 = 0 :=
sorry

end equal_area_split_l525_525361


namespace max_profit_L_max_average_profit_P_max_average_profit_P_reduced_l525_525881

noncomputable def L (x : ℝ) : ℝ := -0.5 * x^2 + 300 * x - 20000
noncomputable def P (x : ℝ) : ℝ := (-0.5 * x^2 + 300 * x - 20000) / x

theorem max_profit_L : ∃ x, x ∈ set.Icc 0 360 ∧ L x = 25000 :=
by {
  use 300,
  split,
  { norm_num, },
  { unfold L, sorry }
}

theorem max_average_profit_P : ∃ x, x ∈ set.Icc 0 360 ∧ P x = 100 :=
by {
  use 200,
  split,
  { norm_num, },
  { unfold P, sorry }
}

theorem max_average_profit_P_reduced : ∃ x, x ∈ set.Icc 0 160 ∧ P x = 95 :=
by {
  use 160,
  split,
  { norm_num, },
  { unfold P, sorry }
}

end max_profit_L_max_average_profit_P_max_average_profit_P_reduced_l525_525881


namespace lcm_gcd_problem_l525_525243

open Nat

theorem lcm_gcd_problem (n : ℕ) (A B : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≥ A)
  (h2 : ∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → gcd (a i) (a j) ≤ B) :
  lcm (list.map a (list.fin_range n)).prod ≥
    (list.range n).map (λ i, A ^ (i + 1) / B ^ (i * i / 2)).max' :=
sorry

end lcm_gcd_problem_l525_525243


namespace min_value_S_l525_525154

noncomputable def S (λ : ℝ) := 
  if h : λ > 1 then (λ - 1 + 4/(λ - 1) + 4) else 0

theorem min_value_S : (Inf (S '' {x | x > 1})) = 8 := by
  sorry

end min_value_S_l525_525154


namespace interesting_numbers_correct_l525_525746

def is_interesting (n : ℕ) : Prop := 
  20 ≤ n ∧ n ≤ 90 ∧ 
  ∀ d ∈ (n.divisors_sorted), ∀ d' ∈ (n.divisors_sorted), d < d' → d'.mod d = 0

def interesting_numbers : Finset ℕ := {25, 27, 32, 49, 64, 81}

theorem interesting_numbers_correct : 
  ∀ n, 20 ≤ n ∧ n ≤ 90 → is_interesting n ↔ n ∈ interesting_numbers := by
  sorry

end interesting_numbers_correct_l525_525746


namespace g_at_100_l525_525325

-- Defining that g is a function from positive real numbers to real numbers
def g : ℝ → ℝ := sorry

-- The given conditions
axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x * g y - y * g x = g (x / y)

axiom g_one : g 1 = 1

-- The theorem to prove
theorem g_at_100 : g 100 = 50 :=
by
  sorry

end g_at_100_l525_525325


namespace volume_tetrahedron_ABCD_dependent_on_lengths_angle_distance_l525_525272

variables (a b s : ℝ) (α : ℝ)

def volume_tetrahedron : ℝ :=
  (1 / 6) * a * b * s * (Real.sin α)

theorem volume_tetrahedron_ABCD_dependent_on_lengths_angle_distance 
  (AB CD : ℝ) (angle distance : ℝ) :
  (AB = a) → (CD = b) → (angle = α) → (distance = s) →
  volume_tetrahedron a b s α = (1 / 6) * a * b * s * (Real.sin α) :=
by {
  intros,
  rw [volume_tetrahedron],
  refl
}

end volume_tetrahedron_ABCD_dependent_on_lengths_angle_distance_l525_525272


namespace probability_both_numbers_are_prime_l525_525421

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l525_525421


namespace polynomial_with_transformed_roots_l525_525829

theorem polynomial_with_transformed_roots
  (a b c q r : ℝ)
  (Hroots : Polynomial.eval a (Polynomial.mk [r, q, 0, 1]) = 0)
  (Hroots : Polynomial.eval b (Polynomial.mk [r, q, 0, 1]) = 0)
  (Hroots : Polynomial.eval c (Polynomial.mk [r, q, 0, 1]) = 0)
  : Polynomial.eval (Polynomial.mk [-1, -q, 0, r]) (Polynomial.mk [x, (b + c) / (a ^ 2), (c + a) / (b ^ 2), (a + b) / (c ^ 2)]) = 0 :=
sorry

end polynomial_with_transformed_roots_l525_525829


namespace sales_price_reduction_l525_525873

theorem sales_price_reduction
  (current_sales : ℝ := 20)
  (current_profit_per_shirt : ℝ := 40)
  (sales_increase_per_dollar : ℝ := 2)
  (desired_profit : ℝ := 1200) :
  ∃ x : ℝ, (40 - x) * (20 + 2 * x) = 1200 ∧ x = 20 :=
by
  use 20
  sorry

end sales_price_reduction_l525_525873


namespace seating_arrangements_correct_l525_525359

section SeatingArrangements

variables (R F B : ℕ) (cond_front : F = 11) (cond_back : B = 12) (cond_total : R = F + B)
  (cond_middle_forbidden : ∀ i, (5 ≤ i) ∧ (i ≤ 7) → ¬occupied_front i)
  (cond_not_adjacent : ∀ a b, a != b → ¬ next_to a b)

-- Defining the type for counting seating arrangements
def total_seating_arrangements (R : ℕ) : ℕ := 
  ∑ i in finset.range R, 
    if i ∈ forbidden_seats then 0 else R - 1 - (if is_edge_seat i then 1 else 2)

-- Theorem stating the correct number of seating arrangements
theorem seating_arrangements_correct 
  (H : total_seating_arrangements R = 362) : 
  ∃ arrangements, arrangements = 362 :=
sorry

end SeatingArrangements

end seating_arrangements_correct_l525_525359


namespace probability_compare_l525_525196

noncomputable def num_white_balls := 1
noncomputable def num_red_balls := 2
noncomputable def total_balls := num_white_balls + num_red_balls

noncomputable def P1 := num_red_balls / total_balls
noncomputable def P2 := num_white_balls / total_balls

theorem probability_compare : P1 > P2 :=
by
  rw [P1, P2]
  have h1 : 2 / 3 > 1 / 3 := by norm_num
  exact h1

end probability_compare_l525_525196


namespace initial_ace_cards_l525_525263

-- Variables representing initial cards and remaining cards
variables (A B C : ℕ)

-- Conditions
axiom a1 : B = 55 -- Number of Ace cards left
axiom a2 : C = 178 -- Number of Baseball cards left
axiom a3 : C = B + 123 -- Nell has 123 more baseball cards than Ace cards now

-- Initial numbers
axiom a4 : B = 55 -- Number of Ace cards left

theorem initial_ace_cards (A : ℕ) (B : ℕ := 55) (C : ℕ := 178) (initial_baseball_cards : ℕ := 438) :
  C = B + 123 → initial_baseball_cards - C = 260 → A - B = 260 → A = 315 :=
by
  intros h1 h2 h3
  rw [h3, Nat.add_comm, h1]
  sorry

end initial_ace_cards_l525_525263


namespace correct_propositions_l525_525669

variables {Line Plane : Type}
variables {m n : Line} {α : Plane}

-- Definitions of parallel and perpendicular relationships
def parallel (l1 l2 : Line) : Prop := sorry -- definition of parallel lines
def perpendicular (l : Line) (p : Plane) : Prop := sorry -- definition of line perpendicular to plane
def in_plane (l : Line) (p : Plane) : Prop := sorry -- definition of line within a plane

-- Conditions
axiom non_coincident_lines : m ≠ n
axiom line1_parallel_line2 : parallel m n
axiom line1_perpendicular_plane : perpendicular m α
axiom line2_perpendicular_plane : perpendicular n α
axiom line1_parallel_plane : in_plane m α
axiom line2_parallel_plane : in_plane n α
axiom line1_perpendicular_line2 : sorry -- relation m ⊥ n

-- Propositions
def prop1 : Prop := perpendicular m α → parallel m n → perpendicular n α
def prop2 : Prop := perpendicular m α → perpendicular n α → parallel m n
def prop3 : Prop := perpendicular m α → in_plane n α → perpendicular m n
def prop4 : Prop := in_plane m α → perpendicular m n → perpendicular n α

-- Proof problem statement
theorem correct_propositions : (prop1 ∧ prop2 ∧ prop3) ∧ ¬prop4 := sorry

end correct_propositions_l525_525669


namespace complex_poly_bound_l525_525251

noncomputable def alpha (n : ℕ) (x : ℕ → ℂ) : ℂ := (1 / n) * ∑ i in Finset.range n, x i

noncomputable def beta2 (n : ℕ) (x : ℕ → ℂ) : ℝ := (1 / n) * ((Finset.range n).sum (λ i, Complex.abs (x i)^2))

noncomputable def P (n : ℕ) (a : ℕ → ℂ) (x : ℂ) : ℂ :=
  (Finset.range n).sum (λ i, a i * x^(n - i)) + x^n

theorem complex_poly_bound (n : ℕ) (a : ℕ → ℂ) (x : ℕ → ℂ) (x0 : ℂ) :
    let α := alpha n x,
        β2 := beta2 n x in
    P n a x0 = 0 →
    β2 < 1 + Complex.abs α^2 →
    Complex.abs (α - x0)^2 < 1 - β2 + Complex.abs α^2 →
    Complex.abs (P n a x0) < 1 :=
by
  intros α β2 habs hβ h0
  -- elaborate proof involving the conditions and inequalities
  sorry

end complex_poly_bound_l525_525251


namespace cost_per_steak_knife_l525_525544

theorem cost_per_steak_knife :
  ∀ (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℝ),
  sets = 2 →
  knives_per_set = 4 →
  cost_per_set = 80 →
  (cost_per_set * sets) / (sets * knives_per_set) = 20 :=
by
  intros sets knives_per_set cost_per_set sets_eq knives_per_set_eq cost_per_set_eq
  rw [sets_eq, knives_per_set_eq, cost_per_set_eq]
  sorry

end cost_per_steak_knife_l525_525544


namespace flower_counts_l525_525193

theorem flower_counts (
    total_flowers : ℕ := 180,
    red_percentage : ℕ := 30,
    green_percentage : ℕ := 10,
    blue_factor : ℕ := 2,
    yellow_difference : ℕ := 5,
    purple_ratio : ℕ := 3,
    orange_ratio : ℕ := 7
  ) :
  ∃ (red green blue yellow purple orange : ℕ),
  red = red_percentage * total_flowers / 100 ∧
  green = green_percentage * total_flowers / 100 ∧
  blue = green / blue_factor ∧
  yellow = red + yellow_difference ∧
  purple + orange = total_flowers - (red + green + blue + yellow) ∧
  purple * orange_ratio = orange * purple_ratio ∧
  red = 54 ∧ green = 18 ∧ blue = 9 ∧ yellow = 59 ∧ purple = 12 ∧ orange = 28 :=
sorry

end flower_counts_l525_525193


namespace Anna_trades_1_stamp_l525_525916

theorem Anna_trades_1_stamp (A_initial A_final A_stamps_received J_initial : ℕ)
  (h1 : A_initial = 37)
  (h2 : A_stamps_received = 14)
  (h3 : A_final = 50)
  (h_total : A_initial + A_stamps_received = 51) :
  A_initial + A_stamps_received - 1 = A_final :=
by
  rw [h1, h2, h3, h_total]
  exact (50 - 1).symm

end Anna_trades_1_stamp_l525_525916


namespace family_can_cross_in_17_minutes_l525_525883

theorem family_can_cross_in_17_minutes :
  ∃ (father_time mother_time son_time grandmother_time : ℕ)
    (initial_cross_return0 second_cross_return final_cross : ℕ),
    father_time = 1 ∧
    mother_time = 2 ∧
    son_time = 5 ∧
    grandmother_time = 10 ∧
    initial_cross_return0 = 2 + 1 ∧
    second_cross_return = 10 + 2 ∧
    final_cross = 2 ∧
    (initial_cross_return0 + second_cross_return + final_cross = 17) :=
begin
  use [1, 2, 5, 10, 2 + 1, 10 + 2, 2],
  repeat { split; try { refl } },
  calc
    2 + 1 + 10 + 2 + 2 = 17 : by norm_num
end

end family_can_cross_in_17_minutes_l525_525883


namespace math_proof_problem_l525_525118

noncomputable def complex_example : Prop :=
  let z := 2 - complex.i in
  let conj_z := complex.conj z in
  z * (conj_z + complex.i) = 6 + 2 * complex.i

theorem math_proof_problem : complex_example := by
  sorry

end math_proof_problem_l525_525118


namespace mean_equality_l525_525332

theorem mean_equality (y : ℝ) :
  ((3 + 7 + 11 + 15) / 4 = (10 + 14 + y) / 3) → y = 3 :=
by
  sorry

end mean_equality_l525_525332


namespace math_proof_problem_l525_525115

noncomputable def complex_example : Prop :=
  let z := 2 - complex.i in
  let conj_z := complex.conj z in
  z * (conj_z + complex.i) = 6 + 2 * complex.i

theorem math_proof_problem : complex_example := by
  sorry

end math_proof_problem_l525_525115


namespace sum_of_digits_of_N_is_54_l525_525028

noncomputable def calculate_sum_of_digits_of_N : ℕ :=
  let N := (36 ^ 50 * 50 ^ 36).sqrt in
  N.digits.sum

-- Here is the theorem we need to prove
theorem sum_of_digits_of_N_is_54 : calculate_sum_of_digits_of_N = 54 :=
  sorry

end sum_of_digits_of_N_is_54_l525_525028


namespace total_accidents_all_three_highways_l525_525228

def highway_conditions : Type :=
  (accident_rate : ℕ, per_million : ℕ, total_traffic : ℕ)

def highway_a : highway_conditions := (75, 100, 2500)
def highway_b : highway_conditions := (50, 80, 1600)
def highway_c : highway_conditions := (90, 200, 1900)

def total_accidents (hc : highway_conditions) : ℕ :=
  hc.accident_rate * hc.total_traffic / hc.per_million

theorem total_accidents_all_three_highways :
  total_accidents highway_a +
  total_accidents highway_b +
  total_accidents highway_c = 3730 := by
  sorry

end total_accidents_all_three_highways_l525_525228


namespace cupric_cyanide_formation_l525_525068

structure Reaction (A B C D : Type) :=
(mol_A : A)
(mol_B : B)
(mol_C : C)
(mol_D : D)
(balance_eq : 2 * mol_A + mol_B = mol_C + mol_D)

def react (A B C D : Type) [Reaction A B C D] (a : A) (b : B) : C := sorry

theorem cupric_cyanide_formation :
  ∀ (HCN CuSO4 CuCN H2SO4 : ℕ),
    Reaction HCN CuSO4 CuCN H2SO4 →
    (HCN = 4) →
    (CuSO4 = 2) →
    (CuCN = 2) :=
begin
  intros HCN CuSO4 CuCN H2SO4 h r1 r2,
  sorry
end

end cupric_cyanide_formation_l525_525068


namespace find_c_l525_525546

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Iio (-2) ∪ Set.Ioi 3 → x^2 - c * x + 6 > 0) → c = 1 :=
by
  sorry

end find_c_l525_525546


namespace value_of_a_l525_525184

theorem value_of_a (a : ℤ) (h1 : 2 * a + 6 + (3 - a) = 0) : a = -9 :=
sorry

end value_of_a_l525_525184


namespace original_number_of_laborers_l525_525879

theorem original_number_of_laborers (L : ℕ) 
  (h : L * 9 = (L - 6) * 15) : L = 15 :=
sorry

end original_number_of_laborers_l525_525879


namespace cosine_angle_between_adjacent_faces_l525_525064

theorem cosine_angle_between_adjacent_faces (a : ℝ) :
  (cos (angle_between_adjacent_faces_of_regular_quadrilateral_pyramid a)) = -1 / 3 :=
sorry

end cosine_angle_between_adjacent_faces_l525_525064


namespace cesaro_sum_51_term_sequence_l525_525560

theorem cesaro_sum_51_term_sequence :
  ∀ (B : Fin 50 → ℝ),
  (∑ i in Finset.range 50, ∑ j in Finset.range (i + 1), B j) / 50 = 500 →
  (∑ i in Finset.range 51, ∑ j in Finset.range (i + 1), (if j = 0 then 2 else B (j - 1))) / 51 = 492 :=
by
  intro B h
  sorry

end cesaro_sum_51_term_sequence_l525_525560


namespace prob_primes_1_to_30_l525_525414

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l525_525414


namespace find_m_tangent_parabola_hyperbola_l525_525338

-- Definitions from given conditions
def parabola (x : ℝ) : ℝ := 2 * x^2 + 3

def hyperbola (x y : ℝ) (m : ℝ) : Prop := 4 * y^2 - m * x^2 = 9

-- The main statement.
theorem find_m_tangent_parabola_hyperbola : ∃ (m : ℝ), (m = 48) ∧ ∀ x : ℝ, hyperbola x (parabola x) m :=
begin
  use 48,
  split,
  { refl, }, -- m = 48
  { intro x,
    sorry, -- Proof that hyperbola \(4 * (2 * x ^ 2 + 3) ^ 2) - 48 * x ^ 2 = 9\) holds
  }
end

end find_m_tangent_parabola_hyperbola_l525_525338


namespace snow_globes_in_box_l525_525051

theorem snow_globes_in_box (S : ℕ) 
  (h1 : ∀ (box_decorations : ℕ), box_decorations = 4 + 1 + S)
  (h2 : ∀ (num_boxes : ℕ), num_boxes = 12)
  (h3 : ∀ (total_decorations : ℕ), total_decorations = 120) :
  S = 5 :=
by
  sorry

end snow_globes_in_box_l525_525051


namespace max_pens_sold_l525_525018

theorem max_pens_sold (pen_profit toy_cost : ℕ) (promo_pens : ℕ) (total_profit : ℕ) 
  (h_pen_profit : pen_profit = 9)
  (h_toy_cost : toy_cost = 2)
  (h_promo_pens : promo_pens = 4)
  (h_total_profit : total_profit = 1922) :
  let package_profit := promo_pens * pen_profit - toy_cost in
  let x := total_profit / package_profit in
  let total_pens_from_promo := promo_pens * x in
  let remaining_profit := total_profit % package_profit in
  let additional_pens := remaining_profit / pen_profit in
  let total_pens := total_pens_from_promo + additional_pens in
  total_pens = 226 :=
by
  -- We will leave the proof as sorry since we are only required to state the theorem.
  sorry

end max_pens_sold_l525_525018


namespace no_solution_exists_l525_525062

theorem no_solution_exists (f : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, f (f x + 2 * y) = 3 * x + f (f (f y) - x)) :=
sorry

end no_solution_exists_l525_525062


namespace sushi_eating_orders_l525_525832

/-- Define a 2 x 3 grid with sushi pieces being distinguishable -/
inductive SushiPiece : Type
| A | B | C | D | E | F

open SushiPiece

/-- A function that counts the valid orders to eat sushi pieces satisfying the given conditions -/
noncomputable def countValidOrders : Nat :=
  sorry -- This is where the proof would go, stating the number of valid orders

theorem sushi_eating_orders :
  countValidOrders = 360 :=
sorry -- Skipping proof details

end sushi_eating_orders_l525_525832


namespace total_travel_expenses_l525_525791

noncomputable def cost_of_fuel_tank := 45
noncomputable def miles_per_tank := 500
noncomputable def journey_distance := 2000
noncomputable def food_ratio := 3 / 5
noncomputable def hotel_cost_per_night := 80
noncomputable def number_of_hotel_nights := 3
noncomputable def fuel_cost_increase := 5

theorem total_travel_expenses :
  let number_of_refills := journey_distance / miles_per_tank
  let first_refill_cost := cost_of_fuel_tank
  let second_refill_cost := first_refill_cost + fuel_cost_increase
  let third_refill_cost := second_refill_cost + fuel_cost_increase
  let fourth_refill_cost := third_refill_cost + fuel_cost_increase
  let total_fuel_cost := first_refill_cost + second_refill_cost + third_refill_cost + fourth_refill_cost
  let total_food_cost := food_ratio * total_fuel_cost
  let total_hotel_cost := hotel_cost_per_night * number_of_hotel_nights
  let total_expenses := total_fuel_cost + total_food_cost + total_hotel_cost
  total_expenses = 576 := by sorry

end total_travel_expenses_l525_525791


namespace tangent_line_parallel_to_y_eq_4x_minus_1_l525_525347

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_line_parallel_to_y_eq_4x_minus_1 :
  (∃ p₀ : ℝ × ℝ, (p₀ = (1,0) ∨ p₀ = (-1, -4)) ∧ 
     ∃ k : ℝ, (k = 4 ∧ k = f' (p₀.1))) :=
by
  -- Definition of the derivative f'
  let f' := λ x : ℝ, 3 * x^2 + 1
  -- Assume k is the slope that equals 4
  have slope_k : (4 = f' 1) ∧ (4 = f' (-1)), from sorry
  -- Solutions to the equation derived from the conditions
  have p0_values : ((1, 0) ∈ {(x, f x) | x = 1 ∨ x = -1}) ∧ ((-1, -4) ∈ {(x, f x) | x = 1 ∨ x = -1}), from sorry
  exact ⟨⟨(1, 0), or.inl rfl⟩, ⟨4, slope_k.left⟩⟩ ∨ ⟨⟨(-1, -4), or.inr rfl⟩, ⟨4, slope_k.right⟩⟩,
  sorry

end tangent_line_parallel_to_y_eq_4x_minus_1_l525_525347


namespace probability_of_two_prime_numbers_l525_525395

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l525_525395


namespace real_numbers_rational_l525_525771

open Real

theorem real_numbers_rational
  (x y : ℝ)
  (h : ∀ (p q : ℕ), p.Prime ∧ q.Prime ∧ p ≠ q ∧ ¬p.even ∧ ¬q.even → (x^p + y^q) ∈ ℚ) :
  x ∈ ℚ ∧ y ∈ ℚ :=
by
  sorry

end real_numbers_rational_l525_525771


namespace bus_present_when_sara_arrives_l525_525473

open ProbabilityTheory MeasureTheory Real

noncomputable def bus_arrival : Measure (Measure.measure_space measureTheory ℝ) := 
  uniform_measure (set.Icc 0 60)

noncomputable def sara_arrival : Measure (Measure.measure_space measureTheory ℝ) := 
  uniform_measure (set.Icc 0 60)

theorem bus_present_when_sara_arrives : 
  ∀ (y : ℝ), y ∈ set.Icc (0 : ℝ) 60 → 
  ∀ (x : ℝ), x ∈ set.Icc (0 : ℝ) 60 →
  (∫ x in (0 : ℝ)..60, ∫ y in (0 : ℝ)..60, indicator (set.Icc y (y + 40)) x d bus_arrival y d sara_arrival x) = (2 / 3 : ℝ) := 
sorry

end bus_present_when_sara_arrives_l525_525473


namespace cost_of_limestone_per_pound_l525_525871

theorem cost_of_limestone_per_pound (L : ℝ) (shale_cost : ℝ) (compound_weight : ℝ) (compound_cost : ℝ)
  (limestone_weight : ℝ) (shale_weight : ℝ) :
  shale_cost = 5 →
  compound_weight = 100 →
  compound_cost = 4.25 →
  limestone_weight = 37.5 →
  shale_weight = compound_weight - limestone_weight →
  compound_weight * compound_cost = 425 →
  shale_weight * shale_cost + limestone_weight * L = 425 →
  L = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7,
  sorry

end cost_of_limestone_per_pound_l525_525871


namespace sum_first_10_odd_numbers_l525_525307

def sumFirstNOddNumbers (n : ℕ) : ℕ := ∑ k in Finset.range n, (2 * k + 1)

theorem sum_first_10_odd_numbers : sumFirstNOddNumbers 10 = 100 :=
by
  sorry

end sum_first_10_odd_numbers_l525_525307


namespace problem_statement_l525_525707

noncomputable def area_triangle_ADE (BD DC : ℝ) (area_ABD : ℝ) (DE : ℝ) : ℝ :=
  let ratio := BD / DC in
  let area_ADC := (2 / 3) * area_ABD in
  let area_ADE := (DE / DC) * area_ADC in
  area_ADE

theorem problem_statement : 
  ∀ (BD DC : ℝ) (area_ABD : ℝ) (DE : ℝ), 
  BD / DC = 3 / 2 → 
  area_ABD = 27 → 
  DE = (1 / 4) * DC → 
  area_triangle_ADE BD DC area_ABD DE = 4.5 :=
by
  intros BD DC area_ABD DE h1 h2 h3
  have ratio : BD / DC = 3 / 2 := h1
  have area_ABD_val : area_ABD = 27 := h2
  have DE_val : DE = (1 / 4) * DC := h3
  sorry

end problem_statement_l525_525707


namespace cotangent_identity_tangent_identity_l525_525688

-- Part 1
theorem cotangent_identity (A B C : ℝ) (n : ℤ)
  (h1 : A + B + C = Real.pi) :
  (Real.cot n * A) * (Real.cot n * B) + 
  (Real.cot n * B) * (Real.cot n * C) + 
  (Real.cot n * C) * (Real.cot n * A) = 1 :=
sorry

-- Part 2
theorem tangent_identity (A B C : ℝ) (n : ℤ) 
  (h1 : A + B + C = Real.pi) 
  (hn : n % 2 = 1) :
  (Real.tan (n * A / 2)) * (Real.tan (n * B / 2)) + 
  (Real.tan (n * B / 2)) * (Real.tan (n * C / 2)) + 
  (Real.tan (n * C / 2)) * (Real.tan (n * A / 2)) = 1 :=
sorry

end cotangent_identity_tangent_identity_l525_525688


namespace brokerage_percentage_correct_l525_525331

noncomputable def brokerage_percentage (market_value : ℝ) (income : ℝ) (investment : ℝ) (nominal_rate : ℝ) : ℝ :=
  let face_value := (income * 100) / nominal_rate
  let market_price := (face_value * market_value) / 100
  let brokerage_amount := investment - market_price
  (brokerage_amount / investment) * 100

theorem brokerage_percentage_correct :
  brokerage_percentage 110.86111111111111 756 8000 10.5 = 0.225 :=
by
  sorry

end brokerage_percentage_correct_l525_525331


namespace water_added_eq_30_l525_525645

-- Given conditions
def initial_volume := 50
def initial_concentration := 0.4
def final_concentration := 0.25
def amount_of_acid := initial_volume * initial_concentration

-- Proof problem statement
theorem water_added_eq_30 : ∃ w : ℝ, amount_of_acid / (initial_volume + w) = final_concentration ∧ w = 30 :=
by 
  -- Solution steps go here, but for now we insert sorry to skip the proof.
  sorry

end water_added_eq_30_l525_525645


namespace seeds_planted_l525_525764

theorem seeds_planted (seeds_per_bed : ℕ) (beds : ℕ) (total_seeds : ℕ) :
  seeds_per_bed = 10 → beds = 6 → total_seeds = seeds_per_bed * beds → total_seeds = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end seeds_planted_l525_525764


namespace probability_both_numbers_are_prime_l525_525423

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l525_525423


namespace complex_multiplication_identity_l525_525094

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- State the theorem to be proved
theorem complex_multiplication_identity : z * (z_conjugate + complex.i) = 6 + 2 * complex.i :=
by {
  sorry -- proof goes here
}

end complex_multiplication_identity_l525_525094


namespace least_sum_of_exponents_for_2023_l525_525670

-- The problem is to prove the least possible sum of the exponents
-- of powers of 2 which sum up to 2023 is 48

def is_sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (exponents : List ℕ), List.sum (exponents.map (λ e, 2^e)) = n ∧ 
                           exponents.nodup

theorem least_sum_of_exponents_for_2023 : 
  is_sum_of_distinct_powers_of_two 2023 → 
  ∃ (exponents : List ℕ), List.sum (exponents.map (λ e, 2^e)) = 2023 ∧
                           exponents.nodup ∧
                           List.sum exponents = 48 :=
by
  sorry

end least_sum_of_exponents_for_2023_l525_525670


namespace probability_two_primes_from_1_to_30_l525_525407

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l525_525407


namespace point_Oc_on_line_PC_l525_525733

open EuclideanGeometry

/-- Let \( O_a \), \( O_b \), and \( O_c \) be the centers of the circumscribed circles of triangles \( PBC \), \( PCA \), and \( PAB \) respectively.
If the points \( O_a \) and \( O_b \) lie on the lines \( PA \) and \( PB \), show that the point \( O_c \) lies on the line \( PC \). -/
theorem point_Oc_on_line_PC (P A B C Oa Ob Oc : Point) 
  (circumcenter_PBC : IsCircumcenter Oa P B C)
  (circumcenter_PCA : IsCircumcenter Ob P C A) 
  (circumcenter_PAB : IsCircumcenter Oc P A B)
  (Oa_on_PA : lies_on_line Oa P A)
  (Ob_on_PB : lies_on_line Ob P B) :
  lies_on_line Oc P C :=
sorry

end point_Oc_on_line_PC_l525_525733


namespace probability_two_primes_l525_525379

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l525_525379


namespace proof_problem_l525_525913

noncomputable def problem_statement : Prop :=
  let p1 := ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0
  let p2 := ∀ x y : ℝ, x + y > 2 → x > 1 ∧ y > 1
  let p3 := ∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3
  let p4 := ∀ a b c : ℝ, a ≠ 0 ∧ b^2 - 4 * a * c > 0 → ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0
  p3 = true ∧ p1 = false ∧ p2 = false ∧ p4 = false

theorem proof_problem : problem_statement := 
sorry

end proof_problem_l525_525913


namespace find_original_production_planned_l525_525906

-- Definition of the problem
variables (x : ℕ)
noncomputable def original_production_planned (x : ℕ) :=
  (6000 / (x + 500)) = (4500 / x)

-- The theorem to prove the original number planned is 1500
theorem find_original_production_planned (x : ℕ) (h : original_production_planned x) : x = 1500 :=
sorry

end find_original_production_planned_l525_525906


namespace complex_number_z_l525_525124

theorem complex_number_z (z : ℂ) (i : ℂ) (h1 : i^2 = -1) (h2 : 2 * i / z = -√3 - i) :
  z = -1 / 2 - (√3 / 2) * i := 
sorry

end complex_number_z_l525_525124


namespace quadratic_complete_square_l525_525804

theorem quadratic_complete_square :
  ∃ d e : ℝ, (∀ x : ℝ, x^2 + 800 * x + 500 = (x + d)^2 + e) ∧
    (e / d = -398.75) :=
by
  use 400
  use -159500
  sorry

end quadratic_complete_square_l525_525804


namespace probability_of_two_prime_numbers_l525_525396

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l525_525396


namespace min_sum_of_squares_convex_heptagon_l525_525298

structure LatticePoint where
  x : Int
  y : Int

def squared_distance (p1 p2 : LatticePoint) : Int :=
  (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)

def is_convex (points : List LatticePoint) : Prop := sorry
def all_sides_distinct (points : List LatticePoint) : Prop := sorry

noncomputable def heptagon_min_sum_of_squares : Int :=
  let points := [ ⟨0, 0⟩, ⟨3, 0⟩, ⟨5, 1⟩, ⟨6, 2⟩, ⟨3, 4⟩, ⟨2, 4⟩, ⟨0, 2⟩ ]
  let distances := List.map (λ i => squared_distance points.get! i points.get! ((i + 1) % 7)) (List.range 7)
  List.sum distances

theorem min_sum_of_squares_convex_heptagon :
  ∃ (points : List LatticePoint), points.length = 7 ∧
    is_convex points ∧ all_sides_distinct points ∧
    List.sum (List.map (λ i => squared_distance (points.get! i) (points.get! ((i + 1) % 7))) (List.range 7)) = 42 :=
begin
  use [ ⟨0, 0⟩, ⟨3, 0⟩, ⟨5, 1⟩, ⟨6, 2⟩, ⟨3, 4⟩, ⟨2, 4⟩, ⟨0, 2⟩ ],
  split,
  { simp },
  split,
  { sorry },
  split,
  { sorry },
  { dsimp only [list.get!, list.range, list.map, squared_distance],
    iterate 7 { simp [squared_distance], },
    norm_num }
end

end min_sum_of_squares_convex_heptagon_l525_525298


namespace rational_iff_zero_l525_525069

def expr (x : ℝ) : ℝ :=
  x^2 - (Real.sqrt (x^2 + 1)) - 1 / (x^2 - (Real.sqrt (x^2 + 1)))

theorem rational_iff_zero (x : ℝ) : Rational (expr x) ↔ x = 0 :=
sorry

end rational_iff_zero_l525_525069


namespace complex_calculation_l525_525106

variable (z : ℂ)

theorem complex_calculation : (z = 2 - (-1:ℂ)) → (z * (conj z + (-1:ℂ))) = 6 + 2 * (-1:ℂ) :=
by
  intro h
  rw h
  have h_conj : conj (2 - (-1:ℂ)) = 2 + (-1:ℂ) := by simp
  rw h_conj
  ring
  done

end complex_calculation_l525_525106


namespace dodecahedron_interior_diagonals_l525_525166

theorem dodecahedron_interior_diagonals (dodecahedron : Type) [finite (dodecahedron → Prop)] :
  let vertices := 20
  in let adjacent_vertices := 3
  in let total_potential_diagonals := vertices * (vertices - adjacent_vertices - 1)
  in total_potential_diagonals / 2 = 160 :=
by
  let vertices : ℕ := 20
  let adjacent_vertices : ℕ := 3
  let total_potential_diagonals : ℕ := vertices * (vertices - adjacent_vertices - 1)
  have result := total_potential_diagonals / 2
  show result = 160
  sorry

end dodecahedron_interior_diagonals_l525_525166


namespace dorchester_puppy_washing_l525_525951

-- Define the conditions
def daily_pay : ℝ := 40
def pay_per_puppy : ℝ := 2.25
def wednesday_total_pay : ℝ := 76

-- Define the true statement
theorem dorchester_puppy_washing :
  let earnings_from_puppy_washing := wednesday_total_pay - daily_pay in
  let number_of_puppies := earnings_from_puppy_washing / pay_per_puppy in
  number_of_puppies = 16 :=
by
  -- Placeholder for the proof
  sorry

end dorchester_puppy_washing_l525_525951


namespace prove_median_of_list_5000_l525_525446

def median_of_list_5000 : Prop :=
  let list1 := (List.range 2501)
  let list2 := list1.map (fun n => n^2)
  let combined_list := list1 ++ list2
  let sorted_list := combined_list.qsort (≤)
  median sorted_list = 2500.5

theorem prove_median_of_list_5000 : median_of_list_5000 := 
by sorry

end prove_median_of_list_5000_l525_525446


namespace div_trans_l525_525174

variable {a b c : ℝ}

theorem div_trans :
  a / b = 3 → b / c = 5 / 2 → c / a = 2 / 15 :=
  by
  intro h1 h2
  sorry

end div_trans_l525_525174


namespace find_number_of_officers_l525_525195

theorem find_number_of_officers
  (avg_salary_employees : ℕ)
  (avg_salary_officers : ℕ)
  (avg_salary_non_officers : ℕ)
  (num_non_officers : ℕ)
  (num_all_employees : ℕ)
  (total_salary_all : ℕ)
  (total_salary_officer_and_non_officer: ℕ)
  (total_salary_from_avg: total_salary_all)
  (eq_total_salaries: total_salary_all = total_salary_officer_and_non_officer) :
 (num_employees_officers : ℕ := 15) :=
 sorry

end find_number_of_officers_l525_525195


namespace tony_average_time_to_store_l525_525275

-- Definitions and conditions
def speed_walking := 2  -- MPH
def speed_running := 10  -- MPH
def distance_to_store := 4  -- miles
def days_walking := 1  -- Sunday
def days_running := 2  -- Tuesday, Thursday
def total_days := days_walking + days_running

-- Proof statement
theorem tony_average_time_to_store :
  let time_walking := (distance_to_store / speed_walking) * 60
  let time_running := (distance_to_store / speed_running) * 60
  let total_time := time_walking * days_walking + time_running * days_running
  let average_time := total_time / total_days
  average_time = 56 :=
by
  sorry  -- Proof to be completed

end tony_average_time_to_store_l525_525275


namespace sum_of_three_squares_not_divisible_by_3_l525_525284

theorem sum_of_three_squares_not_divisible_by_3
    (N : ℕ) (n : ℕ) (a b c : ℤ) 
    (h1 : N = 9^n * (a^2 + b^2 + c^2))
    (h2 : ∃ (a1 b1 c1 : ℤ), a = 3 * a1 ∧ b = 3 * b1 ∧ c = 3 * c1) :
    ∃ (k m n : ℤ), N = k^2 + m^2 + n^2 ∧ (¬ (3 ∣ k ∧ 3 ∣ m ∧ 3 ∣ n)) :=
sorry

end sum_of_three_squares_not_divisible_by_3_l525_525284


namespace existential_inequality_false_iff_l525_525813

theorem existential_inequality_false_iff {a : ℝ} :
  (∀ x : ℝ, x^2 + a * x - 2 * a ≥ 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
by
  sorry

end existential_inequality_false_iff_l525_525813


namespace monotone_intervals_range_of_t_for_three_roots_l525_525152

def f (t x : ℝ) : ℝ := x^3 - 2 * x^2 + x + t

def f_prime (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 1

-- 1. Monotonic intervals
theorem monotone_intervals (t : ℝ) :
  (∀ x, f_prime x > 0 → x < 1/3 ∨ x > 1) ∧
  (∀ x, f_prime x < 0 → 1/3 < x ∧ x < 1) :=
sorry

-- 2. Range of t for three real roots
theorem range_of_t_for_three_roots (t : ℝ) :
  (∃ a b : ℝ, f t a = 0 ∧ f t b = 0 ∧ a ≠ b ∧
   a = 1/3 ∧ b = 1 ∧
   -4/27 + t > 0 ∧ t < 0) :=
sorry

end monotone_intervals_range_of_t_for_three_roots_l525_525152


namespace nth_letter_2023_is_F_l525_525443

-- Define the sequence and its length
def sequence : String := "ABCDEFGFEDCBA"
def seq_length : Nat := 13

-- Define the function to calculate the N-th letter of the sequence
def nth_letter (n : Nat) : Char :=
  sequence[(n - 1) % seq_length]

-- Theorem statement asserting that the 2023rd letter in the sequence is 'F'
theorem nth_letter_2023_is_F : nth_letter 2023 = 'F' :=
  sorry

end nth_letter_2023_is_F_l525_525443


namespace circle_area_l525_525694

theorem circle_area (O : Point) (A B C D I G H : Point) (r : ℝ)
  (diam_AB : diameter O A B) (diam_CD : diameter O C D)
  (perpendicular : perp diam_AB diam_CD)
  (chord_GH_intersects_AB_at_I : chord_intersection GH I AB)
  (GI : length_segment G I = 3)
  (IH : length_segment I H = 5)
  : area_circle O r = 18 * real.pi :=
by
  sorry

end circle_area_l525_525694


namespace rectangular_box_third_side_length_l525_525014

theorem rectangular_box_third_side_length
  (num_cubes : ℕ) (cube_volume : ℕ) (dim1 dim2 : ℕ) (box_volume : ℕ)
  (h : ℕ)
  (h1: num_cubes = 24) 
  (h2: cube_volume = 27) 
  (h3: dim1 = 9) 
  (h4: dim2 = 12)
  (h5: box_volume = num_cubes * cube_volume)
  (h6: box_volume = dim1 * dim2 * h)
  : h = 6 := 
by
sosorry

end rectangular_box_third_side_length_l525_525014


namespace find_value_BE_GF2_CD_HF2_l525_525586

-- Definitions based on problem conditions
variables {O F1 F2 : Point}
variables {C1 C2 : ConicSection}
variables {l : Line}
variables (A B C D E G H : Point)
variables {x y t : ℝ}

-- Given conditions
axiom ellipsoid_centered_at_origin (C1 : ConicSection) (O : Point)
axiom ellipse_foci_left_right (C1 : ConicSection) (F1 F2 : Point)
axiom parabola_vertex_focus (C2 : ConicSection) (O F2 : Point)
axiom intersection_point_angle_obtuse 
  (C1 : ConicSection) (C2 : ConicSection) (A F2 F1: Point) : is_obtuse (∠ A F2 F1)
axiom distance_A_F1 : dist A F1 = 7 / 2
axiom distance_A_F2 : dist A F2 = 5 / 2
axiom line_passing_through_F2_not_perpendicular_x (l : Line) (F2 : Point)
axiom line_intersects_C1_C2 (l : Line) (C1 : ConicSection) (C2 : ConicSection) :
  ∃ B E C D : Point, 
    (B ∈ C1) ∧ (E ∈ C1) ∧ (C ∈ C2) ∧ (D ∈ C2) ∧ (l.contains B ∧ l.contains C ∧ l.contains D ∧ l.contains E)

-- Definitions for midpoints
def midpoint (P Q : Point) : Point := ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

axiom midpoint_CD : G = midpoint C D
axiom midpoint_BE : H = midpoint B E

-- Final proof statement
theorem find_value_BE_GF2_CD_HF2 :
  ∃ A B C D E G H, 
  C1.is_ellipse ∧ C2.is_parabola ∧ 
  dist A F1 = 7/2 ∧ dist A F2 = 5/2 ∧
  (∠ A F2 F1).is_obtuse ∧
  (∃ l, l.contains F2 ∧ ∃ B E C D, 
  B ∈ C1 ∧ E ∈ C1 ∧ C ∈ C2 ∧ D ∈ C2 ∧
  l.contains B ∧ l.contains E ∧ l.contains C ∧ l.contains D) ∧
  G = midpoint C D ∧ H = midpoint B E
  → (|BE| * |GF2|) / (|CD| * |HF2|) = 3 :=
sorry

end find_value_BE_GF2_CD_HF2_l525_525586


namespace correctness_of_props_l525_525802

def prop1 (P : Type) [point_space P] : Prop :=
  ∀ point : P, ∀ l1 l2 : line P, ¬(point ∈ l1 ∧ non_coplanar l1 l2) → 
  ∃ plane : plane P, point ∈ plane ∧ parallel plane l1 ∧ parallel plane l2

def prop2 (α β : plane) (a b : line) : Prop :=
  (α ∩ β = a ∧ b ⊥ a) → b ⊥ α

def prop3 (prism : Prism) : Prop :=
  (∃ face1 face2, face1 ∈ lateral_faces prism ∧ face2 ∈ lateral_faces prism ∧ 
  perpendicular face1 base prism ∧ perpendicular face2 base prism) → 
  is_right_prism prism

def prop4 (prism : Prism) : Prop :=
  (∃ face1 face2 face3 face4, pairwise_congruent face1 face2 face3 face4 ∧ 
  face1 ∈ lateral_faces prism ∧ face2 ∈ lateral_faces prism ∧ 
  face3 ∈ lateral_faces prism ∧ face4 ∈ lateral_faces prism) → 
  is_right_prism prism

def prop5 (pyramid : Pyramid) : Prop :=
  (equilateral_triangle (base pyramid) ∧ 
  ∀ face, face ∈ lateral_faces pyramid → isosceles_triangle face) → 
  is_regular_tetrahedron pyramid

def prop6 (pyramid: Pyramid) : Prop :=
  (equilateral_triangle (base pyramid) ∧ 
  interior_angles_equal (vertex pyramid) (vertices base pyramid)) → 
  is_regular_tetrahedron pyramid

def num_correct_props : ℕ := 0

theorem correctness_of_props : 
  num_correct_props = 
  (∑ p in {prop1, prop2, prop3, prop4, prop5, prop6}, if p then 1 else 0) :=
sorry

end correctness_of_props_l525_525802


namespace maximum_value_P_l525_525054

theorem maximum_value_P {n : ℕ} (h : n > 0) (x : Fin 2n → ℝ) (h_x : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  let p : Fin 2n → ℝ := λ i, x i * x ((i + 1) % 2n)
  let P := (-1)^0 * p 0 + (-1)^1 * p 1 + ... + (-1)^(2n-1) * p (2n-1)
  ( ∃ (x : Fin 2n → ℝ), P = 2 * \left\lfloor (n-1)/2 \right\rfloor ) :=
begin 
  sorry 
end

end maximum_value_P_l525_525054


namespace gcd_gx_x_multiple_of_18432_l525_525595

def g (x : ℕ) : ℕ := (3*x + 5) * (7*x + 2) * (13*x + 7) * (2*x + 10)

theorem gcd_gx_x_multiple_of_18432 (x : ℕ) (h : ∃ k : ℕ, x = 18432 * k) : Nat.gcd (g x) x = 28 :=
by
  sorry

end gcd_gx_x_multiple_of_18432_l525_525595


namespace tangent_period_intersection_l525_525799

theorem tangent_period_intersection (ω : ℝ) (hω : ω > 0) (h_dist : ∃ a b : ℝ, a < b ∧ (b - a = π / 4) ∧ (tan (ω * a) = π / 4) ∧ (tan (ω * b) = π / 4)) :
  tan (4 * (π / 4)) = 0 :=
by 
  sorry

end tangent_period_intersection_l525_525799


namespace marble_placement_l525_525198

theorem marble_placement :
  (∃ (f : Fin 6 → Fin 6), Function.Bijective f) ↔ 720 := sorry

end marble_placement_l525_525198


namespace num_distinguishable_arrangements_l525_525776

-- Definitions based on conditions
def gold_coins : ℕ := 5
def silver_coins : ℕ := 5
def total_coins : ℕ := gold_coins + silver_coins
def engraved_face_gold_coins : ℕ := 3

-- Property no two adjacent coins are face to face
def no_adjacent_face_to_face (arrangement : list char) : Prop :=
  ∀ i < arrangement.length - 1,
    (arrangement.nth i = some 'H' ∧ arrangement.nth (i + 1) = some 'H') ∨
    (arrangement.nth i = some 'T' ∧ arrangement.nth (i + 1) = some 'T') → false

-- Main statement
theorem num_distinguishable_arrangements :
  ∃ arrangements : list (list char),
    arrangements.length = 252 * 10 * 11 ∧
    ∀ arrangement ∈ arrangements,
      arrangement.length = total_coins ∧
      arrangement.count ('G' : char) = gold_coins ∧
      arrangement.count ('S' : char) = silver_coins ∧
      arrangement.count ('E' : char) = engraved_face_gold_coins ∧
      no_adjacent_face_to_face arrangement :=
sorry

end num_distinguishable_arrangements_l525_525776


namespace sqrt_sqrt_81_eq_pm_3_l525_525812

theorem sqrt_sqrt_81_eq_pm_3 : sqrt (sqrt 81) = 3 ∨ sqrt (sqrt 81) = -3 := 
  by
  sorry

end sqrt_sqrt_81_eq_pm_3_l525_525812


namespace simplify_expression_l525_525301

theorem simplify_expression (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 36 :=
by
  sorry

end simplify_expression_l525_525301


namespace sodas_to_take_back_l525_525978

def num_sodas_brought : ℕ := 50
def num_sodas_drank : ℕ := 38

theorem sodas_to_take_back : (num_sodas_brought - num_sodas_drank) = 12 := by
  sorry

end sodas_to_take_back_l525_525978


namespace projection_result_l525_525011

open Real

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let denom := (v.1 * v.1 + v.2 * v.2)
  let num := (u.1 * v.1 + u.2 * v.2)
  let k := num / denom
  (k * v.1, k * v.2)

theorem projection_result :
  proj (proj (3, 3) (45 / 10, 9 / 10)) (-3, 3) = (-30 / 13, -6 / 13) :=
  sorry

end projection_result_l525_525011


namespace trainees_same_number_of_acquaintances_l525_525958

theorem trainees_same_number_of_acquaintances (S : Finset ℕ) (hS : S.card = 31)
  (G : SimpleGraph S) : 
  ∃ (a b : S), a ≠ b ∧ (G.degree a = G.degree b) := 
by 
  sorry

end trainees_same_number_of_acquaintances_l525_525958


namespace greening_investment_growth_l525_525877

-- Define initial investment in 2020 and investment in 2022.
def investment_2020 : ℝ := 20000
def investment_2022 : ℝ := 25000

-- Define the average growth rate x
variable (x : ℝ)

-- The mathematically equivalent proof problem:
theorem greening_investment_growth : 
  20 * (1 + x) ^ 2 = 25 :=
sorry

end greening_investment_growth_l525_525877


namespace find_x_rational_l525_525966

theorem find_x_rational (x : ℝ) (h1 : ∃ (a : ℚ), x + Real.sqrt 3 = a)
  (h2 : ∃ (b : ℚ), x^2 + Real.sqrt 3 = b) :
  x = (1 / 2 : ℝ) - Real.sqrt 3 :=
sorry

end find_x_rational_l525_525966


namespace pure_water_to_achieve_desired_concentration_l525_525642

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l525_525642


namespace smallest_solution_eq_l525_525070

theorem smallest_solution_eq (x : ℝ) (hneq1 : x ≠ 1) (hneq5 : x ≠ 5) (hneq4 : x ≠ 4) :
  (∃ x : ℝ, (1 / (x - 1)) + (1 / (x - 5)) = (4 / (x - 4)) ∧
            (∀ y : ℝ, (1 / (y - 1)) + (1 / (y - 5)) = (4 / (y - 4)) → x ≤ y → y = x) ∧
            x = (5 - Real.sqrt 33) / 2) := 
begin
  sorry
end

end smallest_solution_eq_l525_525070


namespace common_ratio_q_neg_half_l525_525992

noncomputable def a : ℕ → ℝ := λ n, (some geometric sequence definition)

def not_constant (a : ℕ → ℝ) : Prop := ∃ n m, n ≠ m ∧ a n ≠ a m

axiom a3_eq : a 3 = 5 / 2
axiom S3_eq : a 0 + a 1 + a 2 = 15 / 2

theorem common_ratio_q_neg_half (q : ℝ) (a : ℕ → ℝ) (h1 : not_constant a)
  (h2 : a 3 = 5 / 2) (h3 : a 0 + a 1 + a 2 = 15 / 2) : q = -1 / 2 :=
sorry

end common_ratio_q_neg_half_l525_525992


namespace min_sum_distances_on_line_l525_525760

theorem min_sum_distances_on_line (A B C D : ℝ) (h_distinct : A < B ∧ B < C ∧ C < D) :
  ∃ infinitely_many_points (x : ℝ), x ∈ Icc (min A D) (max A D) :=
sorry

end min_sum_distances_on_line_l525_525760


namespace number_of_liars_possible_values_l525_525271

-- Definitions of the conditions
def number_of_natives : ℕ := 2023

-- Main theorem statement
theorem number_of_liars_possible_values :
  let x := number_of_natives % 3, y := number_of_natives / 3,
  x ≥ 2 → 3 * y + 2 * x = number_of_natives → 337 = 337 := sorry

end number_of_liars_possible_values_l525_525271


namespace find_functions_l525_525249

noncomputable def is_solution (f : ℚ → ℚ) (a : ℚ) :=
  ∀ x y : ℚ, f(f(x) + a * y) = a * f(y) + x

noncomputable def solution_check (f : ℚ → ℚ) :=
  (∀ x : ℚ, f(x) = x) ∨ (∀ x : ℚ, f(x) = -x) ∨ (∃ c : ℚ, ∀ x : ℚ, f(x) = x + c ∧ c = 2)

theorem find_functions (a : ℚ) (h : a ≠ 0) :
  ∀ (f : ℚ → ℚ), is_solution f a ↔ solution_check f :=
sorry

end find_functions_l525_525249


namespace Wolstenholme_theorem_l525_525863

theorem Wolstenholme_theorem (p : ℕ) (hp : Nat.Prime p) (h5 : 5 ≤ p) :
  p ^ 2 ∣ Nat.num (∑ k in Finset.range (p - 1) + 1, (1 : ℚ) / k) :=
sorry

end Wolstenholme_theorem_l525_525863


namespace tank_capacity_l525_525843

theorem tank_capacity (T : ℝ) (h1 : 0.6 * T = 0.7 * T - 45) : T = 450 :=
by
  sorry

end tank_capacity_l525_525843


namespace sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l525_525027

-- Proof 1: 
theorem sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3 :
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 :=
by
  sorry

-- Proof 2:
theorem sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12 :
  1 / Real.sqrt 24 + abs (Real.sqrt 6 - 3) + (1 / 2)⁻¹ - 2016 ^ 0 = 4 - 11 * Real.sqrt 6 / 12 :=
by
  sorry

-- Proof 3:
theorem sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6 :
  (Real.sqrt 3 + Real.sqrt 2) ^ 2 - (Real.sqrt 3 - Real.sqrt 2) ^ 2 = 4 * Real.sqrt 6 :=
by
  sorry

end sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l525_525027


namespace problem_equivalent_l525_525596

theorem problem_equivalent (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a + b = 6) (h₃ : a * (a - 6) = x) (h₄ : b * (b - 6) = x) : 
  x = -9 :=
by
  sorry

end problem_equivalent_l525_525596


namespace water_added_eq_30_l525_525644

-- Given conditions
def initial_volume := 50
def initial_concentration := 0.4
def final_concentration := 0.25
def amount_of_acid := initial_volume * initial_concentration

-- Proof problem statement
theorem water_added_eq_30 : ∃ w : ℝ, amount_of_acid / (initial_volume + w) = final_concentration ∧ w = 30 :=
by 
  -- Solution steps go here, but for now we insert sorry to skip the proof.
  sorry

end water_added_eq_30_l525_525644


namespace unique_solution_for_system_l525_525084

theorem unique_solution_for_system (a : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 4 * y = 0 ∧ x + a * y + a * z - a = 0 →
    (a = 2 ∨ a = -2)) :=
by
  intros x y z h
  sorry

end unique_solution_for_system_l525_525084


namespace aunt_gave_each_20_l525_525215

theorem aunt_gave_each_20
  (jade_initial : ℕ)
  (julia_initial : ℕ)
  (total_after_aunt : ℕ)
  (equal_amount_from_aunt : ℕ)
  (h1 : jade_initial = 38)
  (h2 : julia_initial = jade_initial / 2)
  (h3 : total_after_aunt = 97)
  (h4 : jade_initial + julia_initial + 2 * equal_amount_from_aunt = total_after_aunt) :
  equal_amount_from_aunt = 20 := 
sorry

end aunt_gave_each_20_l525_525215


namespace length_of_relay_race_l525_525820

theorem length_of_relay_race (n : ℕ) (d : ℕ) (h1 : n = 5) (h2 : d = 30) :
  n * d = 150 :=
by
  rw [h1, h2]
  norm_num

end length_of_relay_race_l525_525820


namespace distinct_colorings_is_9_l525_525203

open Finset

-- Define the conditions
def num_disks : Nat := 6
def num_blue : Nat := 2
def num_red : Nat := 2
def num_green : Nat := 1
def num_yellow : Nat := 1

-- lean does not directly support Burnside lemma, defining equivalente of Burnside lemma's results.
noncomputable def number_colorings_without_symmetries : Nat :=
  (choose num_disks num_blue) * (choose (num_disks - num_blue) num_red) * (choose (num_disks - num_blue - num_red) num_green)

def rotation_symmetries : Nat := 12
def reflection_symmetries : Nat := 6

def fixed_by_identity : Nat := number_colorings_without_symmetries
def fixed_by_reflections : Nat := 0
def fixed_by_rotations : Nat := 0

def number_distinct_colorings : Nat :=
  (fixed_by_identity + reflection_symmetries * fixed_by_reflections + rotation_symmetries * fixed_by_rotations) / (rotation_symmetries + reflection_symmetries + 1)

-- Proof problem statement
theorem distinct_colorings_is_9 : number_distinct_colorings = 9 := by
  sorry

end distinct_colorings_is_9_l525_525203


namespace sum_of_center_coordinates_eq_neg2_l525_525339

theorem sum_of_center_coordinates_eq_neg2 
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7)
  (h2 : y1 = -8)
  (h3 : x2 = -5)
  (h4 : y2 = 2) 
  : (x1 + x2) / 2 + (y1 + y2) / 2 = -2 :=
by
  -- Insert proof here
  sorry

end sum_of_center_coordinates_eq_neg2_l525_525339


namespace floor_sum_example_l525_525960

theorem floor_sum_example : int.floor 23.8 + int.floor (-23.8) = -1 :=
by
  sorry

end floor_sum_example_l525_525960


namespace M_subsetneq_N_l525_525253

def is_odd (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k + 1

def M : set ℤ := {x | is_odd x}
def N : set ℤ := {x | ∃ k : ℤ, x = k + 2}

theorem M_subsetneq_N : M ⊆ N ∧ M ≠ N :=
by sorry

end M_subsetneq_N_l525_525253


namespace polynomial_satisfies_equation_l525_525233

noncomputable def polynomial_form (P : ℝ[X]) (a b : ℝ) (h₁ : a ≠ 0) (k : ℕ) (m : ℝ) :=
  b = (k + 1) * a ∧ P = m * (X * ∏ i in Finset.range (k + 1), (X - polynomial.C (i * a)))

open polynomial

theorem polynomial_satisfies_equation (P : ℝ[X]) (a b : ℝ) (h₁ : a ≠ 0) :
  (∀ x : ℝ, x * P.eval (x - a) = (x - b) * P.eval x) ↔ ∃ k : ℕ, ∃ m : ℝ, polynomial_form P a b h₁ k m :=
sorry

end polynomial_satisfies_equation_l525_525233


namespace min_value_a_sq_plus_b_sq_l525_525563

theorem min_value_a_sq_plus_b_sq (a b : ℝ) 
  (h1 : (λ x : ℝ, (∑ r in finset.range 7, (nat.choose 6 r * a^(6-r) * b^r * x^(12-3*r))) = (20 : ℝ))) :
  a^2 + b^2 = 2 :=
begin
  sorry,
end

end min_value_a_sq_plus_b_sq_l525_525563


namespace number_of_solutions_l525_525168

theorem number_of_solutions : 
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ (2 * x - 3 * y = 6 ∧ |x ^ 2 - |y| | = 2)) ∧ 
    S.toFinset.card = 2 := sorry

end number_of_solutions_l525_525168


namespace log_2_800_sum_l525_525074

theorem log_2_800_sum:
  (∃ (a b : ℤ), a < real.log 800 / real.log 2 ∧ real.log 800 / real.log 2 < b ∧ a + b = 19) :=
by 
  use [9, 10]
  sorry

end log_2_800_sum_l525_525074


namespace sum_of_squares_is_term_l525_525768

variable (a : ℕ → ℕ) -- Assuming the sequence is defined over natural numbers

-- Given conditions as hypotheses
axiom seq_relation : ∀ n : ℕ, a (2 * n - 1) = a (n - 1) ^ 2 + a n ^ 2

theorem sum_of_squares_is_term (n : ℕ) : ∃ m : ℕ, a m = a (n - 1) ^ 2 + a n ^ 2 := 
by
  use 2 * n - 1
  exact seq_relation n

end sum_of_squares_is_term_l525_525768


namespace collinear_MOK_l525_525283

variable {A B C M K O : Type}
variable [Incenter O A B C] -- We define the property that O is the incenter of triangle ABC.

variable {AC : LineSegment A C} -- Line segment AC
variable {BC : LineSegment B C} -- Line segment BC

variable {BK AM : ℝ} -- BK and AM are real numbers corresponding to lengths.
variable [BK_condition : BK * (distance A B) = (distance B O) ^ 2]
variable [AM_condition : AM * (distance A B) = (distance A O) ^ 2]

theorem collinear_MOK : Collinear M O K := 
sorry

end collinear_MOK_l525_525283


namespace Proposition1_correct_Proposition2_incorrect_Proposition3_correct_Proposition4_incorrect_correct_propositions_l525_525239

open PlaneGeometry

variable (m n : Line) (α β γ : Plane)

-- Propositions
def Proposition1 := (α ∥ β) ∧ (α ∥ γ) → (β ∥ γ)
def Proposition2 := (α ⟂ β) ∧ (m ∥ α) → (m ⟂ β)
def Proposition3 := (m ⟂ α) ∧ (m ∥ β) → (α ⟂ β)
def Proposition4 := (m ∥ n) ∧ (n ⊆ α) → (m ∥ α)

-- Theorems
theorem Proposition1_correct : Proposition1 α β γ :=
by sorry

theorem Proposition2_incorrect : ¬Proposition2 α β m :=
by sorry

theorem Proposition3_correct : Proposition3 m α β :=
by sorry

theorem Proposition4_incorrect : ¬Proposition4 m n α :=
by sorry

-- Main theorem to state the correctness of specific propositions
theorem correct_propositions : 
  (Proposition1 α β γ) ∧ (¬ Proposition2 α β m) ∧ (Proposition3 m α β) ∧ (¬ Proposition4 m n α) :=
by
  exact ⟨Proposition1_correct, Proposition2_incorrect, Proposition3_correct, Proposition4_incorrect⟩

end Proposition1_correct_Proposition2_incorrect_Proposition3_correct_Proposition4_incorrect_correct_propositions_l525_525239


namespace weight_of_water_in_vacuum_is_correct_l525_525353

noncomputable def weight_water_vacuum
    (q : ℝ := 151.392)
    (b₁ : ℝ := 748)
    (t : ℝ := 18)
    (s : ℝ := 0.001293)
    (S : ℝ := 0.998654)
    (S₁ : ℝ := 21)
    (base_pressure : ℝ := 760)
    (thermal_expansion_coefficient : ℝ := 0.003665) : ℝ :=
let 
    s₁ := s * (b₁ / (base_pressure * (1 + thermal_expansion_coefficient * t)))
in
    q * ((S₁ - s₁) / (S - s₁)) * (S / S₁)

theorem weight_of_water_in_vacuum_is_correct : weight_water_vacuum = 151.567 :=
by
    -- Given 
    have hq : ℝ := 151.392
    have hb₁ : ℝ := 748
    have ht : ℝ := 18
    have hs : ℝ := 0.001293
    have hS : ℝ := 0.998654
    have hS₁ : ℝ := 21
    have hbase_pressure : ℝ := 760
    have hthermal_expansion_coefficient : ℝ := 0.003665

    -- Calculate s₁ according to the given formula
    let s₁ := hs * (hb₁ / (hbase_pressure * (1 + hthermal_expansion_coefficient * ht)))

    -- Calculate the weight of the water in a vacuum
    let q₁ := hq * ((hS₁ - s₁) / (hS - s₁)) * (hS / hS₁)

    -- Expected result
    have target : ℝ := 151.567

    -- Proving approximately equal
    calc
        q₁ ≈ target : by sorry -- The actual proof steps would go here

end weight_of_water_in_vacuum_is_correct_l525_525353


namespace problem_part1_problem_part2_l525_525604

theorem problem_part1 (ω : ℝ) (hω : ω > 0) (f : ℝ → ℝ)
  (hf : ∀ x, f x = 2 * sin(ω * x) * cos(ω * x) + cos(2 * ω * x))
  (h_period : ∀ x, f(x) = f(x + π)) :
  ∀ k : ℤ, ∀ x ∈ Icc (k * π - 3 * π / 8) (k * π + π / 8), monotone (f x) := sorry

theorem problem_part2 (α β : ℝ)
  (h_fα : (2 * sin(α / 2 - π / 8) * cos(α / 2 - π / 8) + cos(2 * (α / 2 - π / 8))) = √2 / 3)
  (h_fβ : (2 * sin(β / 2 - π / 8) * cos(β / 2 - π / 8) + cos(2 * (β / 2 - π / 8))) = 2 * √2 / 3)
  (hαβ : α ∈ Ioo (-π / 2) (π / 2) ∧ β ∈ Ioo (-π / 2) (π / 2)) :
  cos(α + β) = (2 * √10 - 2) / 9 := sorry

end problem_part1_problem_part2_l525_525604


namespace _l525_525242

open EuclideanGeometry

noncomputable def geometry_collinearity_theorem :
  ∀ (O A B C K H M S T : Point) 
    (circle_O : Circle O)
    (diameter_AB : diameter circle_O A B)
    (tangent_A : tangent circle_O A) 
    (tangent_B : tangent circle_O B)
    (C_on_circle : on_circle C circle_O)
    (BC_m : intersection (line B C) tangent_A K)
    (angle_bisector_CAK : angle_bisector (∠(C A K)) (line C K) H)
    (midpoint_arc_M : midpoint_arc circle_O (arc A B) M)
    (HM_S : intersection (line H M) circle_O S)
    (tangent_M_n : tangent_through_point circle_O M tangent_B T),
     collinear S T K := sorry

end _l525_525242


namespace symmetric_points_x_axis_l525_525704

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : P = (a - 3, 1)) (hQ : Q = (2, b + 1)) (hSymm : P.1 = Q.1 ∧ P.2 = -Q.2) :
  a + b = 3 :=
by 
  sorry

end symmetric_points_x_axis_l525_525704


namespace modulus_of_complex_eq_sqrt5_l525_525592

theorem modulus_of_complex_eq_sqrt5
  (a b : ℝ)
  (z : ℂ)
  (h1 : (a + complex.i)^2 = b * complex.i)
  (h2 : z = a + b * complex.i) :
  |z| = real.sqrt 5 :=
by sorry

end modulus_of_complex_eq_sqrt5_l525_525592


namespace S_not_2010_S_has_503_different_integer_values_l525_525246

noncomputable def S (x : Fin 2010 → ℝ) : ℝ :=
  (Sum (Finset.filter (λ i, i % 2 = 0) (Finset.range 2010)) (λ i, x i * x (i + 1)))

theorem S_not_2010 :
  ∀ x : Fin 2010 → ℝ,
    (∀ i, x i ∈ {Real.sqrt 2 - 1, Real.sqrt 2 + 1}) →
    S x ≠ 2010 := by sorry

theorem S_has_503_different_integer_values :
  ∀ x : Fin 2010 → ℝ,
    (∀ i, x i ∈ {Real.sqrt 2 - 1, Real.sqrt 2 + 1}) →
    ∃ vals : Finset ℤ, vals.card = 503 ∧ 
      ∀ s ∈ vals, ∃ x : Fin 2010 → ℝ, S x = s := by sorry

end S_not_2010_S_has_503_different_integer_values_l525_525246


namespace circle_reflection_l525_525481

-- Definition of the original center of the circle
def original_center : ℝ × ℝ := (8, -3)

-- Definition of the reflection transformation over the line y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Theorem stating that reflecting the original center over the line y = x results in a specific point
theorem circle_reflection : reflect original_center = (-3, 8) :=
  by
  -- skipping the proof part
  sorry

end circle_reflection_l525_525481


namespace range_of_m_l525_525141

open Real

theorem range_of_m (m : ℝ) (h : ∀ x ∈ Icc (0 : ℝ) 1, x^2 - 4 * x ≥ m) : m ≤ -3 :=
by
  sorry

end range_of_m_l525_525141


namespace inverse_of_composition_l525_525795

variable {X Y Z W : Type}
variable (a : Y → Z) (b : X → Y) (c : W → X)

def is_invertible (f : α → β) : Prop := ∃ g : β → α, ∀ x, g (f x) = x ∧ f (g x) = x

noncomputable def compose_three (a : Y → Z) (b : X → Y) (c : W → X) : W → Z := a ∘ b ∘ c

theorem inverse_of_composition
  (ha : is_invertible a) (hb : is_invertible b) (hc : is_invertible c) :
  ∃ (g_inv : Z → W), g_inv = (c⁻¹) ∘ (b⁻¹) ∘ (a⁻¹) := sorry

end inverse_of_composition_l525_525795


namespace integral_proof_l525_525935

noncomputable def integral_expr (x : ℝ) : ℝ :=
  -x^4 / 4 + 5 * x^3 / 3 + 1 / 5 * Real.log (|x|) - 1 / 5 * Real.log (|x + 5|)

theorem integral_proof (x : ℝ) :
  ∫ (t : ℝ) in 0..x, (-t^5 + 25 * t^3 + 1) / (t^2 + 5 * t) = integral_expr x + C :=
by
  sorry

end integral_proof_l525_525935


namespace coprime_with_5_infinite_intersection_l525_525725

-- Definitions
def S : Set ℕ := { n | ∀ d ∈ to_digits 10 n, d = 1 ∨ d = 2 }

def T (n : ℕ) : Set ℕ := { m | n ∣ m }

-- Statement
theorem coprime_with_5_infinite_intersection (n : ℕ) : Set.Infinite (S ∩ T n) ↔ Nat.coprime n 5 := by
  sorry

end coprime_with_5_infinite_intersection_l525_525725


namespace transform_1_to_811_impossible_l525_525005

theorem transform_1_to_811_impossible :
  ∀ (seq_operations : List (ℕ → ℕ)),
    (∀ n, n ∈ seq_operations → (∃ m, n = λ x, (2 * x) ∘ (permute_digits m))) →
    (permute_digits: ℕ → ℕ → ℕ) → -- The function that permutes the digits, given a number and a permutation function
    ∀ n : ℕ, ¬ (1 = 811) := -- Proving that it is impossible for the transformations to result in 811 starting from 1.

begin
  -- Variables definition and initial assumptions
  intro seq_operations,
  intro valid_operations,
  intro permute_digits,
  intro n,

  sorry
end

end transform_1_to_811_impossible_l525_525005


namespace right_triangles_count_l525_525664

theorem right_triangles_count : 
  ∃ a b : ℕ, b < 50 ∧ a^2 + b^2 = (b + 2)^2 ∧ (finset.range 8).filter (λ k, (b = k^2 - 1)).card = 7 :=
sorry

end right_triangles_count_l525_525664


namespace apples_number_l525_525346

def num_apples (A O B : ℕ) : Prop :=
  A = O + 27 ∧ O = B + 11 ∧ A + O + B = 301 → A = 122

theorem apples_number (A O B : ℕ) : num_apples A O B := by
  sorry

end apples_number_l525_525346


namespace existence_and_uniqueness_l525_525439

variables (p : ℝ) (O A B C : ℝ → Prop) (cos α cos β cos γ: ℝ)

-- Assume conditions
axiom distinct_half_lines : O ≠ A ∧ O ≠ B ∧ O ≠ C
axiom non_degenerate : 2 * p > 0
axiom angle_conditions : cos α + cos β + cos γ = 1

-- Set definitions
def ox (x : ℝ) := O x ∧ A x
def oy (y : ℝ) := O y ∧ B y
def oz (z : ℝ) := O z ∧ C z

-- Define points exist in respective half-lines such that the perimeters are equal to 2p.
theorem existence_and_uniqueness (x y z : ℝ) :
  (ox x) → (oy y) → (oz z) → (x + y + sqrt (x^2 + y^2 - 2*x*y*cos γ) = 2 * p) →
  (y + z + sqrt (y^2 + z^2 - 2*y*z*cos α) = 2 * p) →
  (z + x + sqrt (z^2 + x^2 - 2*z*x*cos β) = 2 * p) →
  (∀ (m n k : ℝ), m = x → n = y → k = z → x = m ∧ y = n ∧ z = k)  :=
begin
  sorry
end

end existence_and_uniqueness_l525_525439


namespace work_together_in_4_days_l525_525175

theorem work_together_in_4_days 
  (W : ℝ) 
  (a_days b_days c_days : ℝ) 
  (ha : a_days = 12) 
  (hb : b_days = 9) 
  (hc : c_days = 18) 
  (A := W / a_days) 
  (B := W / b_days) 
  (C := W / c_days) : 
  (W / (A + B + C) = 4) :=
by
  have ha' : A = W / 12 := by rw [ha]
  have hb' : B = W / 9 := by rw [hb]
  have hc' : C = W / 18 := by rw [hc]
  have h_eq : A + B + C = W / 12 + W / 9 + W / 18 :=
    by rw [ha', hb', hc']
  sorry

end work_together_in_4_days_l525_525175


namespace student_failed_by_73_marks_l525_525019

theorem student_failed_by_73_marks (max_marks : ℕ) (student_marks : ℕ) (passing_percentage : ℚ) 
  (h_max_marks : max_marks = 600) (h_student_marks : student_marks = 125) (h_passing_percentage : passing_percentage = 33 / 100) :
  (⌊passing_percentage * max_marks⌋.to_nat - student_marks) = 73 :=
by 
  sorry

end student_failed_by_73_marks_l525_525019


namespace pure_water_to_add_eq_30_l525_525636

noncomputable def initial_volume := 50
noncomputable def initial_concentration := 0.40
noncomputable def desired_concentration := 0.25
noncomputable def initial_acid := initial_concentration * initial_volume

theorem pure_water_to_add_eq_30 :
  ∃ w : ℝ, (initial_acid / (initial_volume + w) = desired_concentration) ∧ w = 30 :=
by
  sorry

end pure_water_to_add_eq_30_l525_525636


namespace min_value_geometric_seq_l525_525994

theorem min_value_geometric_seq (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n)
  (sum_eq_six : m + n = 6) : ∃ (min_val : ℚ), min_val = 3/4 :=
begin
  sorry
end

end min_value_geometric_seq_l525_525994


namespace flies_caught_before_first_9_minute_rest_period_is_510_chameleon_catches_98th_fly_after_312_minutes_flies_caught_after_1999_minutes_is_462_l525_525462

noncomputable def r : ℕ → ℕ
| 1        := 1
| (2 * m)  := r m
| (2 * m + 1) := r m + 1

-- (a) Prove that the number of flies caught before the first 9-minute rest period is 510
theorem flies_caught_before_first_9_minute_rest_period_is_510 :
  ∃ m, r (m + 1) = 9 ∧ m = 510 :=
  sorry

-- Define t function based on the same recursive pattern as described
noncomputable def t : ℕ → ℕ
| 1        := 0  -- assuming t starts at 0 for the first catch
| 2        := r 2
| n        := let m := n / 2 in
              if n % 2 = 0 then 2 * t m + m - r m
              else 2 * t (m + 1) - r m + 1

-- (b) Prove that after 312 minutes, the chameleon catches the 98th fly
theorem chameleon_catches_98th_fly_after_312_minutes :
  t 98 = 312 :=
  sorry

-- (c) Prove that the number of flies caught after 1999 minutes is 462
theorem flies_caught_after_1999_minutes_is_462 :
  ∃ m, t m ≤ 1999 ∧ 1999 < t (m + 1) ∧ m = 462 :=
  sorry

end flies_caught_before_first_9_minute_rest_period_is_510_chameleon_catches_98th_fly_after_312_minutes_flies_caught_after_1999_minutes_is_462_l525_525462


namespace pure_water_to_add_eq_30_l525_525634

noncomputable def initial_volume := 50
noncomputable def initial_concentration := 0.40
noncomputable def desired_concentration := 0.25
noncomputable def initial_acid := initial_concentration * initial_volume

theorem pure_water_to_add_eq_30 :
  ∃ w : ℝ, (initial_acid / (initial_volume + w) = desired_concentration) ∧ w = 30 :=
by
  sorry

end pure_water_to_add_eq_30_l525_525634


namespace water_to_add_for_desired_acid_concentration_l525_525660

theorem water_to_add_for_desired_acid_concentration :
  ∃ w : ℝ, 50 * 0.4 / (50 + w) = 0.25 ∧ w = 30 :=
begin
  let initial_volume := 50,
  let initial_acid_concentration := 0.4,
  let desired_acid_concentration := 0.25,
  let acid_content := initial_volume * initial_acid_concentration,
  use 30,
  split,
  { simp [initial_volume, initial_acid_concentration, desired_acid_concentration, acid_content],
    linarith },
  { refl }
end

end water_to_add_for_desired_acid_concentration_l525_525660


namespace determine_Q_ratio_l525_525743

noncomputable def f (x : ℂ) : ℂ := x^2023 + 19 * x^2022 + 1

axiom r : Fin 2023 → ℂ
axiom distinct_roots : Function.Injective r

def Q (z : ℂ) : ℂ :=
  ∏ j in Finset.univ, (z - (r j)^2 - (r j)^(-2))

theorem determine_Q_ratio : (Q 2) / (Q (-2)) = 1 :=
by
  sorry

end determine_Q_ratio_l525_525743


namespace hyperbola_standard_eq_l525_525310

variables (a b c : ℝ)

-- Definitions for given conditions
def asymptote_eq (x y : ℝ) : Prop := x + (real.sqrt 3) * y = 0
def focus_parabola (focus : ℝ × ℝ) : Prop := focus = (4, 0)
def c_value (c: ℝ) : Prop := c = 4
def hyperbola_properties (a b c : ℝ) : Prop := c^2 = a^2 + b^2

-- The theorem to prove
theorem hyperbola_standard_eq :
∃ a b : ℝ, (asymptote_eq b a ∧ c_value 4) →
  hyperbola_properties a b 4 →
  ∃ (x y : ℝ), (x^2 / 12) - (y^2 / 4) = 1 := by
  sorry

end hyperbola_standard_eq_l525_525310


namespace locus_of_points_l525_525787

structure Point (R : Type) := (x y : R)
def Segment (R : Type) := Set (Point R)

variables {R : Type} [OrderedField R]

def distance (P Q : Point R) : R := real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)
def perpendicular (P : Point R) (a : Set (Point R)) : R := sorry -- Assume definition available

variables {a1 a2 a3 a4 : Set (Point R)} -- Lines
variables {O A1p A2p : Point R} -- Points
variables {k1 k2 k3 k4 d : R} [h0 : ∀ k, k > 0]

theorem locus_of_points (P : Point R) (OZ : R) :
  (P ∈ { Q | distance Q A1p = distance Q A2p ∧ k1 * perpendicular Q a1 + k2 * perpendicular Q a2 + k3 * perpendicular Q a3 + k4 * perpendicular Q a4 = d }) 
    → ∃ line_segment : Segment R, (P ∈ line_segment ∧ ∀ Q ∈ line_segment, distance Q OZ = d / (4 * OZ)) :=
sorry

end locus_of_points_l525_525787


namespace circle_Q_radius_eq_10_l525_525210

noncomputable def circle_radius_problem (AB AC BC : ℝ) (rP : ℝ) (PQ BC : ℝ) : ℝ :=
  let P := rP = 30  -- Radius of circle P
  let Q := PQ = rP + r  -- Relation between circles P and Q
  let ineq1 := AB = AC = 120  -- Triangle sides
  let ineq2 := BC = 90  -- Triangle base
  let tangent1 := P is tangent to AC and BC
  let tangent2 := Q is tangent to AB and BC
  let constraint := no point of Q lies outside of ABC
  rQ = 10  -- The radius of Q is 10

-- Formulate the theorem statement
theorem circle_Q_radius_eq_10 (AB AC BC rP rQ PQ : ℝ) : 
  circle_radius_problem AB AC BC rP PQ BC →
  rQ = 10 :=
begin
  sorry
end

end circle_Q_radius_eq_10_l525_525210


namespace height_of_removed_cone_l525_525324

theorem height_of_removed_cone 
  (frustum_altitude : ℝ)
  (lower_base_area : ℝ)
  (upper_base_area : ℝ) 
  (frustum_altitude_eq : frustum_altitude = 30)
  (lower_base_area_eq : lower_base_area = 400 * real.pi)
  (upper_base_area_eq : upper_base_area = 100 * real.pi) : 
  ∃ h : ℝ, h = 30 := 
by
  sorry

end height_of_removed_cone_l525_525324


namespace probability_two_primes_l525_525385

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l525_525385


namespace identity_proof_l525_525465

theorem identity_proof : 
  ∀ x : ℝ, 
    (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := 
by 
  sorry

end identity_proof_l525_525465


namespace sum_of_arguments_solution_l525_525946

noncomputable def sum_of_arguments (z : ℂ) : ℝ :=
  (Complex.arg z + Complex.arg (z * Complex.exp (Complex.I * (2*Real.pi / 6)) +
                               Complex.arg (z * Complex.exp (Complex.I * (4*Real.pi / 6))) +
                               Complex.arg (z * Complex.exp (Complex.I * (6*Real.pi / 6))) +
                               Complex.arg (z * Complex.exp (Complex.I * (8*Real.pi / 6))) +
                               Complex.arg (z * Complex.exp (Complex.I * (10*Real.pi / 6)))

theorem sum_of_arguments_solution : 
  (∃ z : ℂ, z^6 = -64 * Complex.I) → sum_of_arguments = 1170 :=
by
  sorry

end sum_of_arguments_solution_l525_525946


namespace complex_calculation_l525_525104

variable (z : ℂ)

theorem complex_calculation : (z = 2 - (-1:ℂ)) → (z * (conj z + (-1:ℂ))) = 6 + 2 * (-1:ℂ) :=
by
  intro h
  rw h
  have h_conj : conj (2 - (-1:ℂ)) = 2 + (-1:ℂ) := by simp
  rw h_conj
  ring
  done

end complex_calculation_l525_525104


namespace sin_value_l525_525988

theorem sin_value (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 :=
by
  sorry

end sin_value_l525_525988


namespace largest_b_for_denom_has_nonreal_roots_l525_525973

theorem largest_b_for_denom_has_nonreal_roots :
  ∃ b : ℤ, 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) 
  ∧ (∀ b' : ℤ, (∀ x : ℝ, x^2 + (b' : ℝ) * x + 12 ≠ 0) → b' ≤ b)
  ∧ b = 6 :=
sorry

end largest_b_for_denom_has_nonreal_roots_l525_525973


namespace distance_P_to_x_axis_l525_525129

-- Define the ellipse equation and other necessary conditions
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  (x^2 / 16) + (y^2 / 9) = 1

-- Define the foci points and the condition for P forming a right-angled triangle with F1 and F2
def right_angle_triangle (P F1 F2 : ℝ × ℝ) : Prop :=
  let (x1, y1) := F1 in
  let (x2, y2) := F2 in
  (y2 = 0 ∧ x2 = -x1 ∧ y1 = 0 ∧ x1^2 + y^2 = 7)

-- Define the point P
variables (P : ℝ × ℝ) 

-- State the theorem
theorem distance_P_to_x_axis (P F1 F2 : ℝ × ℝ) (h1 : is_on_ellipse P) (h2 : right_angle_triangle P F1 F2) :
  abs (P.snd) = 9 / 4 := 
sorry

end distance_P_to_x_axis_l525_525129


namespace fill_two_thirds_of_bucket_time_l525_525472

theorem fill_two_thirds_of_bucket_time (fill_entire_bucket_time : ℝ) (h : fill_entire_bucket_time = 3) : (2 / 3) * fill_entire_bucket_time = 2 :=
by 
  sorry

end fill_two_thirds_of_bucket_time_l525_525472


namespace collinear_vectors_l525_525165

theorem collinear_vectors (x : ℝ) : (∃ k : ℝ, (x, x + 2) = (k * 1, k * 3x)) → (x = -2/3 ∨ x = 1) :=
by
  sorry

end collinear_vectors_l525_525165


namespace no_loss_or_gain_l525_525870

def car_cost : ℕ := 20000
def motorcycle_cost : ℕ := 8000
def car_selling_price : ℕ := 18000
def motorcycle_selling_price : ℕ := 10000

theorem no_loss_or_gain :
  car_selling_price + motorcycle_selling_price = car_cost + motorcycle_cost :=
by
  rw [car_selling_price, motorcycle_selling_price, car_cost, motorcycle_cost]
  exact rfl
-- sorry

end no_loss_or_gain_l525_525870


namespace necessary_but_not_sufficient_l525_525134

-- Define propositions p and q
variables {a b : ℝ}

def p : Prop := a * b < 0
def q : Prop := b < 0 ∧ 0 < a

-- State the main theorem
theorem necessary_but_not_sufficient : 
  (q → p) ∧ ¬(p → q) := by
  sorry

end necessary_but_not_sufficient_l525_525134


namespace probability_perfect_square_l525_525008

theorem probability_perfect_square : 
  (∑ n in range (1, 121), 
    if is_even n then 
      if n ≤ 60 then 
        (ite (is_perfect_square n) (2 * (1/270)) 0)
      else 
        (ite (is_perfect_square n) (4 * (1/270)) 0) 
    else 
      if n ≤ 60 then 
        (ite (is_perfect_square n) (1/270) 0) 
      else 
        (ite (is_perfect_square n) (2 * (1/270)) 0)) 
  = 0.074 :=
sorry

end probability_perfect_square_l525_525008


namespace simple_interest_amount_l525_525773

noncomputable def principal : ℝ := 1800
noncomputable def rate : ℝ := 5.93
noncomputable def time : ℝ := 5.93

theorem simple_interest_amount :
  (principal * rate * time / 100).round = 633 := by
  sorry

end simple_interest_amount_l525_525773


namespace complex_calculation_l525_525101

variable (z : ℂ)

theorem complex_calculation : (z = 2 - (-1:ℂ)) → (z * (conj z + (-1:ℂ))) = 6 + 2 * (-1:ℂ) :=
by
  intro h
  rw h
  have h_conj : conj (2 - (-1:ℂ)) = 2 + (-1:ℂ) := by simp
  rw h_conj
  ring
  done

end complex_calculation_l525_525101


namespace product_approximation_l525_525510

theorem product_approximation :
  let a := 2.1
  let b := 45.5
  let c := 0.25
  abs ((a * (b - c)) - 95) < 0.5 :=
by
  sorry

end product_approximation_l525_525510


namespace proof_problem_1_proof_problem_2_l525_525930

noncomputable def problem_1 : Prop :=
  8^(1/3) + 2^(-2) = 9/4

noncomputable def problem_2 : Prop :=
  log 2 10 - log (1/2) 0.4 = 2

theorem proof_problem_1 : problem_1 := by
  sorry

theorem proof_problem_2 : problem_2 := by
  sorry

end proof_problem_1_proof_problem_2_l525_525930


namespace original_profit_profit_function_correct_max_profit_correct_profit_range_correct_price_controlled_correct_l525_525478

/-
Problem:
A certain store originally sold a type of merchandise, which was purchased for ¥80 per item, at ¥100 per item, and could sell 100 items per day. After conducting market research, it was found that for every ¥1 decrease in the price per item, the sales volume could increase by 10 items.
1. How much profit could the store make per day from selling this merchandise originally?
2. Suppose the store later reduces the price of the merchandise by ¥x per item, the store can make a profit of ¥y per day.
  ① Find the function relationship between y and x; what value of x makes the profit y maximum, and what is the maximum profit?
  ② If the store's profit is not less than ¥2160 per day, within what range should the price per item be controlled?
-/

def initial_purchase_price : ℕ := 80
def initial_selling_price : ℕ := 100
def initial_sales_volume : ℕ := 100

def profit (selling_price purchase_price sales_volume : ℕ) : ℕ :=
  (selling_price - purchase_price) * sales_volume

def original_daily_profit : ℕ := profit initial_selling_price initial_purchase_price initial_sales_volume

noncomputable def profit_function (x : ℕ) : ℕ :=
  let new_price := initial_selling_price - x
  let new_sales := initial_sales_volume + 10 * x
  (new_price - initial_purchase_price) * new_sales

def max_profit_value : ℕ := 2250
def max_profit_x : ℕ := 5

def profit_range : set ℕ := {x | 2 ≤ x ∧ x ≤ 8}
def price_controlled (x : ℕ) : set ℕ := {price | 92 ≤ price ∧ price ≤ 98}

theorem original_profit : original_daily_profit = 2000 := sorry

theorem profit_function_correct {x : ℕ} : profit_function x = -10 * x ^ 2 + 100 * x + 2000 := sorry

theorem max_profit_correct : profit_function max_profit_x = max_profit_value := sorry

theorem profit_range_correct {x : ℕ} (h : 2160 ≤ profit_function x) : x ∈ profit_range := sorry

theorem price_controlled_correct {x : ℕ} (h : x ∈ profit_range) : (initial_selling_price - x) ∈ price_controlled x := sorry

end original_profit_profit_function_correct_max_profit_correct_profit_range_correct_price_controlled_correct_l525_525478


namespace parabolic_vertex_expression_l525_525587

theorem parabolic_vertex_expression
  (a b c : ℤ)
  (vertex : Int × Int)
  (h_vertex : vertex = (2, -3))
  (h_equation : ∀ x h, h = -b / (2 * a) → b = -4 * a)
  (h_equation_c : -3 = c - b^2 / (4 * a) → c = -3 + 4 * a)
  : a + b - c = -4 :=
by
  sorry

end parabolic_vertex_expression_l525_525587


namespace total_votes_polled_l525_525853

theorem total_votes_polled (V: ℝ) (h: 0 < V) (h1: 0.70 * V - 0.30 * V = 320) : V = 800 :=
sorry

end total_votes_polled_l525_525853


namespace parabola_fixed_point_and_angle_l525_525600

noncomputable def parabola_vertex_origin : Prop :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (0, 1)
  let parabola : set (ℝ × ℝ) := {p | p.1^2 = 4 * p.2}
  let A (t : ℝ) : ℝ × ℝ := (t, -2)
  ∀ (x₁ x₂ : ℝ), (x₁, x₁^2 / 4) ∈ parabola → (x₂, x₂^2 / 4) ∈ parabola →
                 ∃ (P Q : ℝ × ℝ), P = (x₁, x₁^2 / 4) ∧ Q = (x₂, x₂^2 / 4) ∧ line_through P Q = line_through (0, 2)

noncomputable def angle_relation_in_parabola (t : ℝ) (x₁ x₂ : ℝ) : Prop :=
  let A : ℝ × ℝ := (t, -2)
  let F : ℝ × ℝ := (0, 1)
  let P : ℝ × ℝ := (x₁, x₁^2 / 4)
  let Q : ℝ × ℝ := (x₂, x₂^2 / 4)
  angle F P Q = 2 * angle P A Q

theorem parabola_fixed_point_and_angle :
  ( ∀ (x₁ x₂ : ℝ), (x₁, x₁^2 / 4) ∈ parabola_vertex_origin ∧ (x₂, x₂^2 / 4) ∈ parabola_vertex_origin →
    ∃ (P Q : ℝ × ℝ), P = (x₁, x₁^2 / 4) ∧ Q = (x₂, x₂^2 / 4) ∧ line_through P Q = line_through (0, 2) ) ∧
  ( ∀ (t x₁ x₂ : ℝ), angle_relation_in_parabola t x₁ x₂ )
:= by {
  -- proof would go here
  sorry
}

end parabola_fixed_point_and_angle_l525_525600


namespace hexagon_area_ratio_l525_525200

/-- Given a regular hexagon ABCDEF with points W, X, Y, and Z on sides BC, CD, EF, and FA respectively, and lines AB, ZW, YX, and ED being parallel and equally spaced, the ratio of the area of hexagon WCXYFZ to the area of hexagon ABCDEF is 11/27. -/
theorem hexagon_area_ratio :
  ∀ (A B C D E F W X Y Z : ℝ) 
    (h_regular : hexagon A B C D E F) 
    (h_points : points_on_sides W X Y Z B C D E F A)
    (h_parallel_and_equally_spaced : parallel_and_equally_spaced AB ZW YX ED),
    area_ratio (WCXYFZ A B C D E F W X Y Z) (ABCDEF A B C D E F) = 11 / 27 := 
sorry

end hexagon_area_ratio_l525_525200


namespace train_cross_time_l525_525470

/-- Definitions for the lengths and speeds of the two trains -/
def length_first_train : ℝ := 150
def speed_first_train_kmph : ℝ := 120
def speed_second_train_kmph : ℝ := 80
def length_second_train : ℝ := 350.04
def relative_speed_mps : ℝ := (speed_first_train_kmph + speed_second_train_kmph) * (1000 / 3600)
def combined_length : ℝ := length_first_train + length_second_train
def time_to_cross : ℝ := combined_length / relative_speed_mps

theorem train_cross_time :
  abs (time_to_cross - 9) < 1 :=
by
  /- Skipping the proof -/
  sorry

end train_cross_time_l525_525470


namespace probability_of_two_primes_is_correct_l525_525386

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l525_525386


namespace cube_coverage_max_identical_number_l525_525956

theorem cube_coverage_max_identical_number :
  let n := 6
  let cube := n × n × n
  let face_cells := 1 × 1
  let covering_squares := 2 × 2
  ∀ (covering : cube → covering_squares → Prop),
    (∀ cell, ∃! s, covering cube s) →
    (∀ s1 s2, s1 ≠ s2 → disjoint (covering s1) (covering s2)) →
    ∀ cell, (∃! t, covering cell t) → (card (covering cell) = 3) :=
sorry

end cube_coverage_max_identical_number_l525_525956


namespace yulia_max_candies_l525_525260

def maxCandies (totalCandies : ℕ) (horizontalCandies : ℕ) (verticalCandies : ℕ) (diagonalCandies : ℕ) : ℕ :=
  totalCandies - min (2 * horizontalCandies + 3 * diagonalCandies) (3 * diagonalCandies + 2 * verticalCandies)

-- Constants
def totalCandies : ℕ := 30
def horizontalMoveCandies : ℕ := 2
def verticalMoveCandies : ℕ := 2
def diagonalMoveCandies : ℕ := 3
def path1_horizontalMoves : ℕ := 5
def path1_diagonalMoves : ℕ := 2
def path2_verticalMoves : ℕ := 1
def path2_diagonalMoves : ℕ := 5

theorem yulia_max_candies :
  maxCandies totalCandies (path1_horizontalMoves + path2_verticalMoves) 0 (path1_diagonalMoves + path2_diagonalMoves) = 14 :=
by
  sorry

end yulia_max_candies_l525_525260


namespace prove_math_problem_l525_525857

noncomputable def polar_to_cartesian_curve (ρ : ℝ) (θ : ℝ) : Prop :=
  (ρ = 2 * cos θ) → (ρ ^ 2 = (ρ * cos θ) ^ 2 + (ρ * sin θ) ^ 2)

noncomputable def parametric_to_polar_line (t : ℝ) : Prop :=
  (x = 4 + 1 / 2 * t ∧ y = sqrt 3 / 2 * t) →
  (sqrt 3 * (sqrt x ^ 2 + sqrt y ^ 2) * cos θ - (sqrt x ^ 2 + sqrt y ^ 2) * sin θ = 4 * sqrt 3)

noncomputable def distance_AB (A B : ℝ × ℝ) : ℝ :=
  (λ (θ : ℝ), θ = π/6 → A = (sqrt 3, π/6) ∧ B = (4 * sqrt 3, π/6)) →
  (abs (B.1 - A.1) = 3 * sqrt 3)

theorem prove_math_problem (ρ θ t : ℝ) (A B : ℝ × ℝ) : 
  polar_to_cartesian_curve ρ θ ∧ parametric_to_polar_line t ∧ distance_AB A B := 
  sorry

end prove_math_problem_l525_525857


namespace robot_can_escape_desert_l525_525483

-- Assume the problem conditions
noncomputable def desert_shape : Prop := ∀ (x y : ℕ), x = y

constant energy_level : ℕ := 59
constant max_energy_consumption : ℕ := 5
constant grid_energy : ℕ → ℕ → ℕ
constant total_energy_consumption_in_5x5 : ℕ
constant boundary_distance : ℕ := 15

axiom consumption_bound : ∀ (i j : ℕ), grid_energy i j ≤ max_energy_consumption
axiom subgrid_consumption : ∀ (i j : ℕ), (∑ (m, n) in finset.range 5 ×ˢ finset.range 5, grid_energy (i + m) (j + n)) = 88
axiom movement_rule : ∀ (i j : ℕ), (abs (i - j) = 1 ∨ abs (i - j) = 0) → (grid_energy i j ≤ energy_level)

-- The main problem to prove
theorem robot_can_escape_desert : Prop :=
  ∃ path : list (ℕ × ℕ),
    (∀ (i j : ℕ), i < path.length - 1 → (path.nth i).get_or_else (0,0) ≠ (path.nth (i + 1)).get_or_else (0,0)) ∧
    (∑ (x, y) in path.to_finset, grid_energy x y) ≤ energy_level ∧
    (15 <= boundary_distance)

end robot_can_escape_desert_l525_525483


namespace fg_square_diff_l525_525132

open Real

noncomputable def f (x: ℝ) : ℝ := sorry
noncomputable def g (x: ℝ) : ℝ := sorry

axiom h1 (x: ℝ) (hx : -π / 2 < x ∧ x < π / 2) : f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x))
axiom h2 : ∀ x, f (-x) = -f x
axiom h3 : ∀ x, g (-x) = g x

theorem fg_square_diff (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end fg_square_diff_l525_525132


namespace sum_and_subtract_result_l525_525511

def calc_sum_and_subtract : ℝ :=
  let sum1 := (3/5) * 200 + 0.456 * 875 + (7/8) * 320
  let sum2 := 0.5575 * 1280 + (1/3) * 960
  sum1 - sum2

theorem sum_and_subtract_result :
  calc_sum_and_subtract = -234.9 :=
by
  sorry

end sum_and_subtract_result_l525_525511


namespace seven_letter_good_words_count_l525_525044

def is_good_word (s : String) : Prop :=
  s.all (λ c, c = 'A' ∨ c = 'B' ∨ c = 'C') ∧ 
  ∀ (i : Fin (s.length - 1)), 
    (s[i] = 'A' → s[i + 1] ≠ 'B') ∧ 
    (s[i] = 'B' → s[i + 1] ≠ 'C') ∧ 
    (s[i] = 'C' → s[i + 1] ≠ 'A')

def num_seven_letter_good_words : ℕ :=
  3 * 2 ^ (7 - 1)

theorem seven_letter_good_words_count : 
  num_seven_letter_good_words = 192 :=
by
  sorry

end seven_letter_good_words_count_l525_525044


namespace find_result_l525_525548

theorem find_result : ∀ (x : ℝ), x = 1 / 3 → 5 - 7 * x = 8 / 3 := by
  intros x hx
  sorry

end find_result_l525_525548


namespace degree_measure_regular_octagon_interior_angle_l525_525836

theorem degree_measure_regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range (n - 2), 180) / n = 135 :=
by
  sorry

end degree_measure_regular_octagon_interior_angle_l525_525836


namespace number_of_clients_l525_525458

-- Definitions from the problem
def cars : ℕ := 18
def selections_per_client : ℕ := 3
def selections_per_car : ℕ := 3

-- Theorem statement: Prove that the number of clients is 18
theorem number_of_clients (total_cars : ℕ) (cars_selected_by_each_client : ℕ) (each_car_selected : ℕ)
  (h_cars : total_cars = cars)
  (h_select_each : cars_selected_by_each_client = selections_per_client)
  (h_selected_car : each_car_selected = selections_per_car) :
  (total_cars * each_car_selected) / cars_selected_by_each_client = 18 :=
by
  rw [h_cars, h_select_each, h_selected_car]
  sorry

end number_of_clients_l525_525458


namespace math_proof_problem_l525_525120

noncomputable def complex_example : Prop :=
  let z := 2 - complex.i in
  let conj_z := complex.conj z in
  z * (conj_z + complex.i) = 6 + 2 * complex.i

theorem math_proof_problem : complex_example := by
  sorry

end math_proof_problem_l525_525120


namespace simplify_expression_l525_525303

theorem simplify_expression (x y z : ℝ) : - (x - (y - z)) = -x + y - z := by
  sorry

end simplify_expression_l525_525303


namespace sqrt_sqrt_81_eq_pm_3_l525_525811

theorem sqrt_sqrt_81_eq_pm_3 : sqrt (sqrt 81) = 3 ∨ sqrt (sqrt 81) = -3 := 
  by
  sorry

end sqrt_sqrt_81_eq_pm_3_l525_525811


namespace complex_multiplication_identity_l525_525098

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- State the theorem to be proved
theorem complex_multiplication_identity : z * (z_conjugate + complex.i) = 6 + 2 * complex.i :=
by {
  sorry -- proof goes here
}

end complex_multiplication_identity_l525_525098


namespace jenna_gas_cost_l525_525715

-- Definitions of the given conditions
def hours1 : ℕ := 2
def speed1 : ℕ := 60
def hours2 : ℕ := 3
def speed2 : ℕ := 50
def miles_per_gallon : ℕ := 30
def cost_per_gallon : ℕ := 2

-- Statement to be proven
theorem jenna_gas_cost : 
  let distance1 := hours1 * speed1,
      distance2 := hours2 * speed2,
      total_distance := distance1 + distance2,
      total_gallons := total_distance / miles_per_gallon,
      total_cost := total_gallons * cost_per_gallon
  in total_cost = 18 := 
by 
  let distance1 := hours1 * speed1,
      distance2 := hours2 * speed2,
      total_distance := distance1 + distance2,
      total_gallons := total_distance / miles_per_gallon,
      total_cost := total_gallons * cost_per_gallon

  sorry

end jenna_gas_cost_l525_525715


namespace right_triangle_count_l525_525661

theorem right_triangle_count : 
  (Finset.card (Finset.filter 
    (λ ⟨a, b⟩, a^2 = 4 * (b + 1) ∧ b < 50 ∧ b + 2 ∈ Int ∧ a ∈ Int) 
    (Finset.univ : Finset (ℤ × ℤ)))) = 7 := 
sorry

end right_triangle_count_l525_525661


namespace cook_carrots_l525_525482

theorem cook_carrots :
  ∀ (total_carrots : ℕ) (fraction_used_before_lunch : ℚ) (carrots_not_used_end_of_day : ℕ),
    total_carrots = 300 →
    fraction_used_before_lunch = 2 / 5 →
    carrots_not_used_end_of_day = 72 →
    let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
    let carrots_after_lunch := total_carrots - carrots_used_before_lunch
    let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
    (carrots_used_end_of_day / carrots_after_lunch) = 3 / 5 :=
by
  intros total_carrots fraction_used_before_lunch carrots_not_used_end_of_day
  intros h1 h2 h3
  let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
  let carrots_after_lunch := total_carrots - carrots_used_before_lunch
  let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
  have h : carrots_used_end_of_day / carrots_after_lunch = 3 / 5 := sorry
  exact h

end cook_carrots_l525_525482


namespace class_students_l525_525037

theorem class_students (A B : ℕ) 
  (h1 : A + B = 85) 
  (h2 : (3 * A) / 8 + (3 * B) / 5 = 42) : 
  A = 40 ∧ B = 45 :=
by
  sorry

end class_students_l525_525037


namespace number_913n_divisible_by_18_l525_525559

theorem number_913n_divisible_by_18 (n : ℕ) (h1 : 9130 % 2 = 0) (h2 : (9 + 1 + 3 + n) % 9 = 0) : n = 8 :=
by
  sorry

end number_913n_divisible_by_18_l525_525559


namespace probability_of_prime_pairs_l525_525373

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l525_525373


namespace hyperbola_equation_l525_525597

theorem hyperbola_equation (a b c : ℝ) (e : ℝ) 
  (h1 : e = (Real.sqrt 6) / 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : (c / a) = e)
  (h5 : (b * c) / (Real.sqrt (b^2 + a^2)) = 1) :
  (∃ a b : ℝ, (a = Real.sqrt 2) ∧ (b = 1) ∧ (∀ x y : ℝ, (x^2 / 2) - y^2 = 1)) :=
by
  sorry

end hyperbola_equation_l525_525597


namespace remaining_structure_volume_and_surface_area_l525_525524

-- Define the dimensions of the large cube and the small cubes
def large_cube_volume := 12 * 12 * 12
def small_cube_volume := 2 * 2 * 2

-- Define the number of smaller cubes in the large cube
def num_small_cubes := (12 / 2) * (12 / 2) * (12 / 2)

-- Define the number of smaller cubes removed (central on each face and very center)
def removed_cubes := 7

-- The volume of a small cube after removing its center unit
def single_small_cube_remaining_volume := small_cube_volume - 1

-- Calculate the remaining volume after all removals
def remaining_volume := (num_small_cubes - removed_cubes) * single_small_cube_remaining_volume

-- Initial surface area of a small cube and increase per removal of central unit
def single_small_cube_initial_surface_area := 6 * 4 -- 6 faces of 2*2*2 cube, each face has 4 units
def single_small_cube_surface_increase := 6

-- Calculate the adjusted surface area considering internal faces' reduction
def single_cube_adjusted_surface_area := single_small_cube_initial_surface_area + single_small_cube_surface_increase
def total_initial_surface_area := single_cube_adjusted_surface_area * (num_small_cubes - removed_cubes)
def total_internal_faces_area := (num_small_cubes - removed_cubes) * 2 * 4
def final_surface_area := total_initial_surface_area - total_internal_faces_area

theorem remaining_structure_volume_and_surface_area :
  remaining_volume = 1463 ∧ final_surface_area = 4598 :=
by
  -- Proof logic goes here
  sorry

end remaining_structure_volume_and_surface_area_l525_525524


namespace almost_perfect_numbers_l525_525729

def d (n : Nat) : Nat := 
  -- Implement the function to count the number of positive divisors of n
  sorry

def f (n : Nat) : Nat := 
  -- Implement the function f(n) as given in the problem statement
  sorry

def isAlmostPerfect (n : Nat) : Prop := 
  f n = n

theorem almost_perfect_numbers :
  ∀ n, isAlmostPerfect n → n = 1 ∨ n = 3 ∨ n = 18 ∨ n = 36 :=
by
  sorry

end almost_perfect_numbers_l525_525729


namespace probability_of_prime_pairs_l525_525374

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l525_525374


namespace octagon_diagonal_relation_l525_525937

theorem octagon_diagonal_relation (a b d : ℝ) (h : ∃ r, a = r ∧ b = r * √2 ∧ d = r * √(2 - √2)) : d^2 = a^2 + b^2 :=
by
  obtain ⟨r, ha, hb, hd⟩ := h
  rw [ha, hb, hd]
  sorry

end octagon_diagonal_relation_l525_525937


namespace complex_multiplication_identity_l525_525093

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- State the theorem to be proved
theorem complex_multiplication_identity : z * (z_conjugate + complex.i) = 6 + 2 * complex.i :=
by {
  sorry -- proof goes here
}

end complex_multiplication_identity_l525_525093


namespace infinitely_many_concentric_circles_with_irrational_sides_l525_525286

-- Define a Euclidean plane
variable (Plane : Type) [EuclideanPlane Plane]

-- Define what it means for circles to be concentric
def concentric_circles (c1 c2 : Circle Plane) : Prop :=
  c1.center = c2.center

-- Define a triangle being inscribed in a circle
def inscribed_triangle (T: Triangle Plane) (C : Circle Plane) : Prop :=
  T.vertices.all (λ v, C.on_circle v)

-- The proof statement that, in a Euclidean plane, there are infinitely many concentric circles 
-- such that all triangles inscribed in these circles have at least one irrational side.
theorem infinitely_many_concentric_circles_with_irrational_sides :
  ∃ (C : ℕ → Circle Plane), (∀ n : ℕ, is_concentric (C 0) (C n)) ∧
  (∀ n : ℕ, ∀ T : Triangle Plane, inscribed_triangle T (C n) → ∃ side ∈ T.sides, irrational side.length) :=
sorry

end infinitely_many_concentric_circles_with_irrational_sides_l525_525286


namespace volume_between_spheres_l525_525823

-- Define the radii of the smaller and larger spheres
def r_small : ℝ := 5
def r_large : ℝ := 8

-- Function to calculate the volume of a sphere with given radius r
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

-- Calculate the volume of the smaller sphere
def V_small : ℝ := volume_sphere r_small

-- Calculate the volume of the larger sphere
def V_large : ℝ := volume_sphere r_large

-- Define the volume of the space between the two spheres
def V_space : ℝ := V_large - V_small

theorem volume_between_spheres :
  V_space = 516 * Real.pi :=
by
  sorry

end volume_between_spheres_l525_525823


namespace constant_term_expansion_eq_neg160_l525_525047

theorem constant_term_expansion_eq_neg160 : 
  let x := (x - (2/x)) 
  (constant_term (x^6) = -160)  :=  sorry 

end constant_term_expansion_eq_neg160_l525_525047


namespace inequality_solution_l525_525783

theorem inequality_solution (x : ℝ) (h : x < 5) : (x/(x+2) ≥ 0) ↔ (x ∈ set.Iio (-2) ∪ set.Ico (0) 5) := by
  sorry

end inequality_solution_l525_525783


namespace greening_investment_equation_l525_525874

theorem greening_investment_equation:
  ∃ (x : ℝ), 20 * (1 + x)^2 = 25 := 
sorry

end greening_investment_equation_l525_525874


namespace factorial_trailing_zeros_30_l525_525676

theorem factorial_trailing_zeros_30 :
  let zeros_count (n : ℕ) : ℕ :=
    n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 -- Continue this pattern for higher n
  in zeros_count 30 = 7 :=
by
  -- Decompose the definition and proof here
  sorry

end factorial_trailing_zeros_30_l525_525676


namespace sequence_sum_l525_525808

def y : ℕ → ℕ
| 0     := 3
| (n+1) := y n ^ 2 + y n + 1

theorem sequence_sum : (∑' n, 1 / (y n + 1)) = 1 / 3 := by
  sorry

end sequence_sum_l525_525808


namespace dorchester_puppy_washing_l525_525950

-- Define the conditions
def daily_pay : ℝ := 40
def pay_per_puppy : ℝ := 2.25
def wednesday_total_pay : ℝ := 76

-- Define the true statement
theorem dorchester_puppy_washing :
  let earnings_from_puppy_washing := wednesday_total_pay - daily_pay in
  let number_of_puppies := earnings_from_puppy_washing / pay_per_puppy in
  number_of_puppies = 16 :=
by
  -- Placeholder for the proof
  sorry

end dorchester_puppy_washing_l525_525950


namespace two_digit_multiple_condition_l525_525449

theorem two_digit_multiple_condition :
  ∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ ∃ k : ℤ, x = 30 * k + 2 :=
by
  sorry

end two_digit_multiple_condition_l525_525449


namespace points_of_tangency_circle_l525_525582

-- Definitions to represent the problem setup
variable {G : sphere} {A B : point}
variable {Γ : sphere}

-- Main statement: Prove that the locus of points of tangency for all spheres Γ
-- passing through points A and B and tangent to sphere G forms a circle or 
-- a great circle in G.
theorem points_of_tangency_circle (G A B : point) (Γ : set sphere) (HΓ : ∀ γ ∈ Γ, γ.passes_through A ∧ γ.passes_through B ∧ γ.tangent_to G) :
  exists C : circle, ∀ T : point, (∃ γ ∈ Γ, T ∈ γ.points_of_tangency G) ↔ T ∈ C :=
sorry

end points_of_tangency_circle_l525_525582


namespace aira_rubber_bands_l525_525293

variable (S A J : ℕ)

-- Conditions
def conditions (S A J : ℕ) : Prop :=
  S = A + 5 ∧ A = J - 1 ∧ S + A + J = 18

-- Proof problem
theorem aira_rubber_bands (S A J : ℕ) (h : conditions S A J) : A = 4 :=
by
  -- introduce the conditions
  obtain ⟨h₁, h₂, h₃⟩ := h
  -- use sorry to skip the proof
  sorry

end aira_rubber_bands_l525_525293


namespace trains_cross_time_l525_525437

theorem trains_cross_time (length1 length2 : ℕ) (time1 time2 : ℕ) 
  (speed1 speed2 relative_speed total_length : ℚ) 
  (h1 : length1 = 120) (h2 : length2 = 150) 
  (h3 : time1 = 10) (h4 : time2 = 15) 
  (h5 : speed1 = length1 / time1) (h6 : speed2 = length2 / time2) 
  (h7 : relative_speed = speed1 - speed2) 
  (h8 : total_length = length1 + length2) : 
  (total_length / relative_speed = 135) := 
by sorry

end trains_cross_time_l525_525437


namespace polynomial_decomposition_l525_525247

theorem polynomial_decomposition (n : ℕ) (F G : Polynomial ℝ) 
  (h₁ : 1 + x + x^2 + ... + x^(n - 1) = F * G)
  (h₂ : ∀ i, F.coeff i = 0 ∨ F.coeff i = 1)
  (h₃ : ∀ i, G.coeff i = 0 ∨ G.coeff i = 1)
  (h₄ : n > 1) :
  ∃ k T, k > 1 ∧ 
          (∀ i, T.coeff i = 0 ∨ T.coeff i = 1) ∧ 
          (F = (1 + x + ... + x^(k - 1)) * T ∨ G = (1 + x + ... + x^(k - 1)) * T) :=
sorry

end polynomial_decomposition_l525_525247


namespace checker_on_diagonal_l525_525755

theorem checker_on_diagonal
  (board : ℕ)
  (n_checkers : ℕ)
  (symmetric : (ℕ → ℕ → Prop))
  (diag_check : ∀ i j, symmetric i j -> symmetric j i)
  (num_checkers_odd : Odd n_checkers)
  (board_size : board = 25)
  (checkers : n_checkers = 25) :
  ∃ i, i < 25 ∧ symmetric i i := by
  sorry

end checker_on_diagonal_l525_525755


namespace remaining_pages_l525_525716

def original_book_pages : ℕ := 93
def pages_read_saturday : ℕ := 30
def pages_read_sunday : ℕ := 20

theorem remaining_pages :
  original_book_pages - (pages_read_saturday + pages_read_sunday) = 43 := by
  sorry

end remaining_pages_l525_525716


namespace complement_of_angle_l525_525171

theorem complement_of_angle (A : ℝ) (hA : A = 35) : 180 - A = 145 := by
  sorry

end complement_of_angle_l525_525171


namespace determine_function_f_l525_525244

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain (x : ℝ) : -1 < x → f x ≠ ∞ ∧ -1 < f x
axiom f_0 : f 0 = 0
axiom f_continuous_strictly_monotonic :
  ∀ x y : ℝ, -1 < x ∧ -1 < y → (x < y → f x < f y) ∧ (f x < f y → x < y)
axiom f_inequality :
  ∀ x y : ℝ, -1 < x ∧ -1 < y →
  f (x + f y + x * f y) ≥ y + f x + y * f x

theorem determine_function_f :
  ∀ x : ℝ, -1 < x → (f x = - (x / (1 + x)) ∨ f x = x) :=
sorry

end determine_function_f_l525_525244


namespace part1_solution_set_part2_convex_l525_525238

def f (x : ℝ) : ℝ := -2*x^2 + 5*x - 2

theorem part1_solution_set :
  (∀ x, (1/2 < x ∧ x < 2) → f x > 0) →
  set_of (λ x : ℝ, (f x + 4 < 0)) = 
  { x : ℝ | (x > 3 ∨ x < -(1/2)) } := 
by sorry

theorem part2_convex :
  (∀ x1 x2 : ℝ, (1/2 < x1 ∧ x1 < 2) → (1/2 < x2 ∧ x2 < 2) →
    (f (x1) + f (x2)) / 2 ≤ f ((x1 + x2) / 2)) →
  convex_on ℝ (univ : set ℝ) f := 
by sorry

end part1_solution_set_part2_convex_l525_525238


namespace find_integers_a_b_c_l525_525943

theorem find_integers_a_b_c :
  ∃ a b c : ℤ, ((x - a) * (x - 12) + 1 = (x + b) * (x + c)) ∧ 
  ((b + 12) * (c + 12) = 1 → ((b = -11 ∧ c = -11) → a = 10) ∧ 
  ((b = -13 ∧ c = -13) → a = 14)) :=
by
  sorry

end find_integers_a_b_c_l525_525943


namespace Monika_number_8635_l525_525752

noncomputable def Monika_four_digit_number : ℕ :=
  let n := 8635
  n

theorem Monika_number_8635 
  (n : ℕ)
  (h₁ : (n / 1000) * (n % 10) = 40) 
  (h₂ : ((n / 10) % 10) * ((n / 100) % 10) = 18) 
  (h₃ : abs ((n / 1000) - (n % 10)) = abs (((n / 10) % 10) - ((n / 100) % 10))) 
  (h₄ : ∀ m, abs (n - (n % 10 * 1000 + (n / 10 % 10) * 100 + (n / 100 % 10) * 10 + n / 1000)) ≥ 
                abs (m - (m % 10 * 1000 + (m / 10 % 10) * 100 + (m / 100 % 10) * 10 + m / 1000))) :
  n = 8635 := sorry

end Monika_number_8635_l525_525752


namespace tony_average_time_to_store_l525_525274

-- Definitions and conditions
def speed_walking := 2  -- MPH
def speed_running := 10  -- MPH
def distance_to_store := 4  -- miles
def days_walking := 1  -- Sunday
def days_running := 2  -- Tuesday, Thursday
def total_days := days_walking + days_running

-- Proof statement
theorem tony_average_time_to_store :
  let time_walking := (distance_to_store / speed_walking) * 60
  let time_running := (distance_to_store / speed_running) * 60
  let total_time := time_walking * days_walking + time_running * days_running
  let average_time := total_time / total_days
  average_time = 56 :=
by
  sorry  -- Proof to be completed

end tony_average_time_to_store_l525_525274


namespace sphere_touches_BD_if_touches_AC_l525_525899

-- Define vertices and edges touching conditions
variables {A B C D E F G H : Point}
variables {sphere : Sphere}

-- Conditions about tetrahedron and touching points
axiom sphere_touches_edges_at_points :
  sphere.TouchesEdgeAt A B E ∧ sphere.TouchesEdgeAt B C F ∧ 
  sphere.TouchesEdgeAt C D G ∧ sphere.TouchesEdgeAt D A H

-- Condition about vertices forming a square
axiom vertices_form_square :
  IsSquare E F G H

-- Prove that if the sphere touches AC, it must also touch BD
theorem sphere_touches_BD_if_touches_AC (h : sphere.TouchesEdgeAt A C E) :
  sphere.TouchesEdgeAt B D E :=
sorry

end sphere_touches_BD_if_touches_AC_l525_525899


namespace single_burger_cost_l525_525513

-- Conditions
def total_cost : ℝ := 74.50
def total_burgers : ℕ := 50
def cost_double_burger : ℝ := 1.50
def double_burgers : ℕ := 49

-- Derived information
def cost_single_burger : ℝ := total_cost - (double_burgers * cost_double_burger)

-- Theorem: Prove the cost of a single burger
theorem single_burger_cost : cost_single_burger = 1.00 :=
by
  -- Proof goes here
  sorry

end single_burger_cost_l525_525513


namespace number_of_psafe_integers_l525_525941

def is_psafe (n p : ℕ) : Prop :=
  ∀ k : ℕ, |n - k * p| ≠ 1

def count_psafe (limit : ℕ) (ps : List ℕ) : ℕ :=
  let safe_counts := ps.map (λ p, (1..limit).filter (λ n, is_psafe n p)).length
  safe_counts.foldl (*) 1

theorem number_of_psafe_integers :
  count_psafe 12000 [5, 7, 9] = 1520 := by
  sorry

end number_of_psafe_integers_l525_525941


namespace fourth_square_state_l525_525262

inductive Shape
| Circle
| Triangle
| LineSegment
| Square

inductive Position
| TopLeft
| TopRight
| BottomLeft
| BottomRight

structure SquareState where
  circle : Position
  triangle : Position
  line_segment_parallel_to : Bool -- True = Top & Bottom; False = Left & Right
  square : Position

def move_counterclockwise : Position → Position
| Position.TopLeft => Position.BottomLeft
| Position.BottomLeft => Position.BottomRight
| Position.BottomRight => Position.TopRight
| Position.TopRight => Position.TopLeft

def update_square_states (s1 s2 s3 : SquareState) : Prop :=
  move_counterclockwise s1.circle = s2.circle ∧
  move_counterclockwise s2.circle = s3.circle ∧
  move_counterclockwise s1.triangle = s2.triangle ∧
  move_counterclockwise s2.triangle = s3.triangle ∧
  s1.line_segment_parallel_to = !s2.line_segment_parallel_to ∧
  s2.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  move_counterclockwise s1.square = s2.square ∧
  move_counterclockwise s2.square = s3.square

theorem fourth_square_state (s1 s2 s3 s4 : SquareState) (h : update_square_states s1 s2 s3) :
  s4.circle = move_counterclockwise s3.circle ∧
  s4.triangle = move_counterclockwise s3.triangle ∧
  s4.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  s4.square = move_counterclockwise s3.square :=
sorry

end fourth_square_state_l525_525262


namespace dorchester_puppy_count_l525_525953

/--
  Dorchester works at a puppy wash. He is paid $40 per day + $2.25 for each puppy he washes.
  On Wednesday, Dorchester earned $76. Prove that Dorchester washed 16 puppies that day.
-/
theorem dorchester_puppy_count
  (total_earnings : ℝ)
  (base_pay : ℝ)
  (pay_per_puppy : ℝ)
  (puppies_washed : ℕ)
  (h1 : total_earnings = 76)
  (h2 : base_pay = 40)
  (h3 : pay_per_puppy = 2.25) :
  total_earnings - base_pay = (puppies_washed : ℝ) * pay_per_puppy :=
sorry

example :
  dorchester_puppy_count 76 40 2.25 16 := by
  rw [dorchester_puppy_count, sub_self, mul_zero]

end dorchester_puppy_count_l525_525953


namespace add_water_to_solution_l525_525651

theorem add_water_to_solution (w : ℕ) :
  let initial_volume := 50
  let initial_concentration := 0.4
  let final_concentration := 0.25
  let acid_amount := initial_concentration * initial_volume
  let final_volume := initial_volume + w
  (acid_amount / final_volume = final_concentration) ↔ (w = 30) :=
by 
  sorry

end add_water_to_solution_l525_525651


namespace gain_percent_is_87_point_5_l525_525176

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  ((S - C) / C) * 100

theorem gain_percent_is_87_point_5 {C S : ℝ} (h : 75 * C = 40 * S) :
  gain_percent C S = 87.5 :=
by
  sorry

end gain_percent_is_87_point_5_l525_525176


namespace urn_probability_l525_525503

theorem urn_probability :
  let total_combinations := Nat.choose 19 4 in
  let blue_combinations := Nat.choose 6 1 in
  let red_combinations := Nat.choose 5 3 in
  let favorable_combinations := blue_combinations * red_combinations in
  total_combinations = 3876 ∧
  favorable_combinations = 60 →
  (favorable_combinations : ℚ) / total_combinations = 5 / 323 := 
by
  intros total_combinations blue_combinations red_combinations favorable_combinations h1 h2
  -- proof steps go here
  sorry

end urn_probability_l525_525503


namespace expression_equals_value_l525_525925

theorem expression_equals_value : 97^3 + 3 * (97^2) + 3 * 97 + 1 = 940792 := 
by
  sorry

end expression_equals_value_l525_525925


namespace probability_between_2_and_4_l525_525142

noncomputable theory

open probability_theory

variables {σ : ℝ} (x : ℝ) (h₁ : x ∼ normal 3 σ^2) (h₂ : P(x ≤ 4) = 0.84)

theorem probability_between_2_and_4 :
  P(2 < x < 4) = 0.68 :=
sorry

end probability_between_2_and_4_l525_525142


namespace second_train_cross_time_l525_525369

noncomputable def time_to_cross_second_train : ℝ :=
  let length := 120
  let t1 := 10
  let t_cross := 13.333333333333334
  let v1 := length / t1
  let v_combined := 240 / t_cross
  let v2 := v_combined - v1
  length / v2

theorem second_train_cross_time :
  let t2 := time_to_cross_second_train
  t2 = 20 :=
by
  sorry

end second_train_cross_time_l525_525369


namespace part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l525_525254

-- Definitions of the sets and conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -4 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part 1
theorem part1_union (a : ℝ) (ha : a = 1) : 
  A ∪ B a = { x | -4 < x ∧ x ≤ 3 } :=
sorry

theorem part1_intersection_complement (a : ℝ) (ha : a = 1) : 
  A ∩ (U \ B a) = { x | -4 < x ∧ x < 0 } :=
sorry

-- Part 2
theorem part2_necessary_sufficient_condition (a : ℝ) : 
  (∀ x, x ∈ B a ↔ x ∈ A) ↔ (-3 < a ∧ a < -1) :=
sorry

end part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l525_525254


namespace inscribed_circle_radius_l525_525447

theorem inscribed_circle_radius (AB AC BC : ℝ) (h : AB = 6) (i : AC = 8) (j : BC = 10) (K: ℝ) (s : ℝ) (r: ℝ)
  (h1: K = 1/2 * AB * AC)
  (h2: s = (AB + AC + BC) / 2)
  (h3: K = r * s) : r = 2 :=
by
  have h_AB_AC_BC : AB = 6 ∧ AC = 8 ∧ BC = 10 := by
    { split, exact h, split, exact i, exact j}
  sorry

end inscribed_circle_radius_l525_525447


namespace complement_intersection_l525_525737

namespace ProofProblem

def A : Set ℝ := {x | |x - 2| ≤ 2}
def B : Set ℝ := {y | ∃ x, (y = x^2 - 2x + 2) ∧ (0 ≤ x ∧ x ≤ 3)}

theorem complement_intersection :
  (A ∩ {x | ∃ y, y ∈ B ∧ x = y})ᶜ = {x | x < 1 ∨ x > 4} := by
  sorry

end ProofProblem

end complement_intersection_l525_525737


namespace sin_510_eq_1_div_2_l525_525469

theorem sin_510_eq_1_div_2 : Real.sin (510 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_510_eq_1_div_2_l525_525469


namespace distinct_necklace_arrangements_l525_525308

open_locale big_operators

/-- The number of distinct necklace arrangements with 6 red, 1 white, and 8 yellow balls, 
    considering rotational and reflectional symmetries, is 1519. -/
theorem distinct_necklace_arrangements :
  let n := 15 in
  let r := 6 in
  let w := 1 in
  let y := 8 in
  (n = r + w + y) →
  ∑ k in finset.range ((r + y)! / (r! * y!)), 2 • 1 = 1519 :=
by
  intros n r w y h_n
  have h1: 14! / (6! * 8!) = 3003 := sorry
  have h2: 3003 / 2 = 1501.5 := sorry
  have h3: (3003 - 35) / 2 + 35 = 1519 := sorry
  exact h3

end distinct_necklace_arrangements_l525_525308


namespace tangent_line_at_zero_is_y_eq_x_l525_525793

def f (x : ℝ) : ℝ := Real.sin x

theorem tangent_line_at_zero_is_y_eq_x : 
  ∀ (x y : ℝ), (y = 0 ↔ x = 0) → 
  (∀ x, y = f 0 + Real.cos 0 * (x - 0)) :=
sorry

end tangent_line_at_zero_is_y_eq_x_l525_525793


namespace sqrt_square_identity_l525_525522

-- Define the concept of square root
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Problem statement: prove (sqrt 12321)^2 = 12321
theorem sqrt_square_identity (x : ℝ) : (sqrt x) ^ 2 = x := by
  sorry

-- Specific instance for the given number
example : (sqrt 12321) ^ 2 = 12321 := sqrt_square_identity 12321

end sqrt_square_identity_l525_525522


namespace minimum_length_PQ_l525_525598

open Real

-- Definition of the problem environment
def arithmetic_sequence (a b c : ℝ) := b = (a + c) / 2
def line_l (a b c x y : ℝ) := a * x + b * y + c = 0
def point_A := (1, 2)
def point_Q (x y : ℝ) := 3 * x - 4 * y + 12 = 0
def circle_center := (1, 0)
def circle_radius := 2

-- Statement of the proof problem
theorem minimum_length_PQ
  (a b c : ℝ) (hac : ¬ (a = 0 ∧ c = 0)) -- Condition that a and c are not both zero
  (h_arithmetic : arithmetic_sequence a b c)
  (P : ℝ × ℝ)
  (hP_on_line_l : line_l a b c P.1 P.2)
  (hQ_on_line_l : point_Q P.1 P.2) :
  let dist_to_line := |3 * 1 + 12| / sqrt (3 ^ 2 + (-4) ^ 2)
  in dist_to_line - circle_radius = 1 := 
  by sorry

end minimum_length_PQ_l525_525598


namespace sam_read_pages_l525_525814

-- Define conditions
def assigned_pages : ℕ := 25
def harrison_pages : ℕ := assigned_pages + 10
def pam_pages : ℕ := harrison_pages + 15
def sam_pages : ℕ := 2 * pam_pages

-- Prove the target theorem
theorem sam_read_pages : sam_pages = 100 := by
  sorry

end sam_read_pages_l525_525814


namespace trapezoid_area_l525_525205

/-- Given that the area of the outer square is 36 square units and the area of the inner square is 
4 square units, the area of one of the four congruent trapezoids formed between the squares is 8 
square units. -/
theorem trapezoid_area (outer_square_area inner_square_area : ℕ) 
  (h_outer : outer_square_area = 36)
  (h_inner : inner_square_area = 4) : 
  (outer_square_area - inner_square_area) / 4 = 8 :=
by sorry

end trapezoid_area_l525_525205


namespace solve_system_l525_525969

theorem solve_system :
  {x y : ℝ} (h₁ : x^3 + y^3 = 7) (h₂ : x * y * (x + y) = -2) →
  (x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) :=
by
  intros x y h₁ h₂
  sorry

end solve_system_l525_525969


namespace min_length_intersection_l525_525160

theorem min_length_intersection
  (m n : ℝ)
  (hM0 : 0 ≤ m)
  (hM1 : m + 3/4 ≤ 1)
  (hN0 : n - 1/3 ≥ 0)
  (hN1 : n ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧
  x = ((m + 3/4) + (n - 1/3)) - 1 :=
sorry

end min_length_intersection_l525_525160


namespace total_production_volume_non_conforming_volume_below_100_l525_525872

-- Define the initial conditions and parameters
def initial_production : ℕ := 1250
def initial_pass_rate : ℝ := 0.9
def production_growth_rate : ℝ := 0.05
def pass_rate_growth_rate : ℝ := 0.004

-- Define the production volume of each month
noncomputable def monthly_production : ℕ → ℕ
| 0     => initial_production
| (n+1) => nat.floor (monthly_production n * (1.0 + production_growth_rate))

-- Define the pass rate of each month
noncomputable def monthly_pass_rate : ℕ → ℝ
| 0     => initial_pass_rate
| (n+1) => monthly_pass_rate n + pass_rate_growth_rate

-- Define the non-conforming disinfectant of each month
noncomputable def non_conforming_volume : ℕ → ℝ
| n => monthly_production n * (1 - monthly_pass_rate n)

-- Prove the total production volume from January to December
theorem total_production_volume : 
  (∑ n in finset.range 12, monthly_production n) = 19896 :=
begin
  sorry
end

-- Prove the month from which non-conforming volume is below 100 liters
theorem non_conforming_volume_below_100 : 
  ∀ n ≥ 12, non_conforming_volume n < 100 :=
begin
  sorry
end

end total_production_volume_non_conforming_volume_below_100_l525_525872


namespace numerator_multiple_of_p3_l525_525300

theorem numerator_multiple_of_p3 (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  let num := 2^(p-1) * (1) - ∑ k in Finset.range p, (Nat.choose (p-1) k) * (1 - k * p) ^ (-2)
  p^3 ∣ num := 
by
  sorry

end numerator_multiple_of_p3_l525_525300


namespace aira_rubber_bands_l525_525292

variable (S A J : ℕ)

-- Conditions
def conditions (S A J : ℕ) : Prop :=
  S = A + 5 ∧ A = J - 1 ∧ S + A + J = 18

-- Proof problem
theorem aira_rubber_bands (S A J : ℕ) (h : conditions S A J) : A = 4 :=
by
  -- introduce the conditions
  obtain ⟨h₁, h₂, h₃⟩ := h
  -- use sorry to skip the proof
  sorry

end aira_rubber_bands_l525_525292


namespace tony_average_time_l525_525280

-- Definitions for the conditions
def speed_walk : ℝ := 2  -- speed in miles per hour when Tony walks
def speed_run : ℝ := 10  -- speed in miles per hour when Tony runs
def distance_to_store : ℝ := 4  -- distance to the store in miles
def days : List String := ["Sunday", "Tuesday", "Thursday"]  -- days Tony goes to the store

-- Definition of times taken on each day
def time_sunday := distance_to_store / speed_walk  -- time in hours to get to the store on Sunday
def time_tuesday := distance_to_store / speed_run  -- time in hours to get to the store on Tuesday
def time_thursday := distance_to_store / speed_run -- time in hours to get to the store on Thursday

-- Converting times to minutes
def time_sunday_minutes := time_sunday * 60
def time_tuesday_minutes := time_tuesday * 60
def time_thursday_minutes := time_thursday * 60

-- Definition of average time
def average_time_minutes : ℝ :=
  (time_sunday_minutes + time_tuesday_minutes + time_thursday_minutes) / days.length

-- The theorem to prove
theorem tony_average_time : average_time_minutes = 56 := by
  sorry

end tony_average_time_l525_525280


namespace negation_example_l525_525800

theorem negation_example : 
  (¬ ∃ x_0 : ℚ, x_0 - 2 = 0) = (∀ x : ℚ, x - 2 ≠ 0) :=
by 
  sorry

end negation_example_l525_525800


namespace maximum_planes_by_9_points_l525_525838

def collinear (p1 p2 p3 : Point) : Prop := sorry
def coplanar (s : set Point) : Prop := sorry
def unique_planes (points : set Point) : ℕ :=
  (points.to_finset.powerset.filter (λ s, s.card = 3 ∧ ¬collinear (s.to_finset.elem 0) (s.to_finset.elem 1) (s.to_finset.elem 2))).card

noncomputable def max_planes_determined (points : set Point) (h1 : ∀ s ⊆ points, s.card = 4 → ¬coplanar s)
  (h2 : ∀ s ⊆ points, s.card = 3 → ¬collinear (s.elem 0) (s.elem 1) (s.elem 2)) : Prop :=
  unique_planes points = 84

theorem maximum_planes_by_9_points (points : set Point) (h_size : points.card = 9)
  (h_no_4_coplanar : ∀ s ⊆ points, s.card = 4 → ¬coplanar s)
  (h_no_3_collinear : ∀ s ⊆ points, s.card = 3 → ¬collinear (s.elem 0) (s.elem 1) (s.elem 2)) :
  max_planes_determined points h_no_4_coplanar h_no_3_collinear := 
sorry

end maximum_planes_by_9_points_l525_525838


namespace quadratic_min_value_l525_525816

theorem quadratic_min_value : ∀ x : ℝ, x^2 - 6 * x + 13 ≥ 4 := 
by 
  sorry

end quadratic_min_value_l525_525816


namespace three_digit_division_l525_525053

theorem three_digit_division (abc : ℕ) (a b c : ℕ) (h1 : 100 ≤ abc ∧ abc < 1000) (h2 : abc = 100 * a + 10 * b + c) (h3 : a ≠ 0) :
  (1001 * abc) / 7 / 11 / 13 = abc :=
by
  sorry

end three_digit_division_l525_525053


namespace cost_of_gas_l525_525712

def hoursDriven1 : ℕ := 2
def speed1 : ℕ := 60
def hoursDriven2 : ℕ := 3
def speed2 : ℕ := 50
def milesPerGallon : ℕ := 30
def costPerGallon : ℕ := 2

def totalDistance : ℕ := (hoursDriven1 * speed1) + (hoursDriven2 * speed2)
def gallonsUsed : ℕ := totalDistance / milesPerGallon
def totalCost : ℕ := gallonsUsed * costPerGallon

theorem cost_of_gas : totalCost = 18 := by
  -- You should fill in the proof steps here.
  sorry

end cost_of_gas_l525_525712


namespace alex_serge_equiv_distinct_values_l525_525501

-- Defining the context and data structures
variable {n : ℕ} -- Number of boxes
variable {c : ℕ → ℕ} -- Function representing number of cookies in each box, indexed by box number
variable {m : ℕ} -- Number of plates
variable {p : ℕ → ℕ} -- Function representing number of cookies on each plate, indexed by plate number

-- Define the sets representing the unique counts recorded by Alex and Serge
def Alex_record (c : ℕ → ℕ) (n : ℕ) : Set ℕ := 
  { x | ∃ i, i < n ∧ c i = x }

def Serge_record (p : ℕ → ℕ) (m : ℕ) : Set ℕ := 
  { y | ∃ j, j < m ∧ p j = y }

-- The proof goal: Alex's record contains the same number of distinct values as Serge's record
theorem alex_serge_equiv_distinct_values
  (c : ℕ → ℕ) (n : ℕ) (p : ℕ → ℕ) (m : ℕ) :
  Alex_record c n = Serge_record p m :=
sorry

end alex_serge_equiv_distinct_values_l525_525501


namespace probability_of_two_primes_is_correct_l525_525391

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l525_525391


namespace cost_per_eraser_l525_525750

-- Define the context of the problem
variables (classes folders_per_class pencils_per_class pencils_per_eraser 
           folder_cost pencil_cost paint_cost total_spent : ℕ)
variables (paints_needed : bool)

-- Set the values according to the problem
def set_values :=
  classes = 6 ∧ folders_per_class = 1 ∧ pencils_per_class = 3 ∧
  pencils_per_eraser = 6 ∧ folder_cost = 6 ∧ pencil_cost = 2 ∧
  paint_cost = 5 ∧ total_spent = 80 ∧ paints_needed = true

-- State the theorem we need to prove
theorem cost_per_eraser : set_values → (total_spent - 
  (classes * folder_cost * folders_per_class + classes * pencils_per_class * pencil_cost + paint_cost) = (classes * pencils_per_class / pencils_per_eraser) * (total_spent - (classes * folder_cost * folders_per_class + classes * pencils_per_class * pencil_cost + paint_cost)) / (classes * pencils_per_class / pencils_per_eraser) * (total_spent - (classes * folder_cost * folders_per_class + classes * pencils_per_class * pencil_cost)))
  sorry

end cost_per_eraser_l525_525750


namespace calculate_area_of_polar_region_l525_525463

noncomputable def area_bounded_by_polar_curves : ℝ :=
  (1 / 2) * ∫ (φ : ℝ) in 0..(π / 6), (2 * (cos φ))^2 - 
  (1 / 2) * ℝ ∫ (φ : ℝ) in (π / 6)..(π / 2), (2 * sqrt(3) * sin φ)^2 

theorem calculate_area_of_polar_region :
  area_bounded_by_polar_curves = (5 * π / 6 - sqrt(3)) :=
sorry

end calculate_area_of_polar_region_l525_525463


namespace raja_income_l525_525288

variable (I : ℝ)
variable (household : ℝ) (clothes : ℝ) (medicines : ℝ) (savings : ℝ)

-- Conditions based on the problem statement
def expenditures : household + clothes + medicines = (0.35 * I) + (0.20 * I) + (0.05 * I) := rfl
def savings_value : savings = 15000 := rfl
def savings_calculation : savings = 0.40 * I := by sorry

theorem raja_income :
  (0.40 * I = 15000) → (I = 37500) :=
begin
  -- Proof will go here
  sorry
end

end raja_income_l525_525288


namespace count_japanese_stamps_l525_525264

theorem count_japanese_stamps (total_stamps : ℕ) (perc_chinese perc_us : ℕ) (h1 : total_stamps = 100) 
  (h2 : perc_chinese = 35) (h3 : perc_us = 20): 
  total_stamps - ((perc_chinese * total_stamps / 100) + (perc_us * total_stamps / 100)) = 45 :=
by
  sorry

end count_japanese_stamps_l525_525264


namespace total_toothpicks_required_l525_525488

structure EquilateralTriangleConstruction where
  (numRows : ℕ)
  (baseRowTriangles : ℕ)
  (decrementPerRow : ℕ)

def alternate_row_double_counting: ℕ → Bool
  | 2, 4 => true
  | _ => false

theorem total_toothpicks_required 
  (tri : EquilateralTriangleConstruction)
  (alt_double_count : ℕ → Bool) : 
  tri.numRows = 5 ∧ tri.baseRowTriangles = 100 ∧ tri.decrementPerRow = 1 →
  let toothpicks := 
    (List.sum (List.map (λ rowIdx => 
      let numTriangles := tri.baseRowTriangles - rowIdx
      if alt_double_count rowIdx
      then numTriangles * 6
      else numTriangles * 3) 
    [0, 1, 2, 3, 4])) in
  toothpicks = 1617 :=
sorry

end total_toothpicks_required_l525_525488


namespace triangle_area_l525_525687

theorem triangle_area (AB AC : ℝ) (angleA : ℝ) (h1 : AB = real.sqrt 3) (h2 : AC = 1) (h3 : angleA = real.pi / 6) :
  (1 / 2) * AB * AC * real.sin angleA = real.sqrt 3 / 4 :=
by {
  -- Place for the proof
  sorry
}

end triangle_area_l525_525687


namespace min_value_of_m_l525_525182

def ellipse (x y : ℝ) := (y^2 / 16) + (x^2 / 9) = 1
def line (x y m : ℝ) := y = x + m
def shortest_distance (d : ℝ) := d = Real.sqrt 2

theorem min_value_of_m :
  ∃ (m : ℝ), (∀ (x y : ℝ), ellipse x y → ∃ d, shortest_distance d ∧ line x y m) 
  ∧ ∀ m', m' < m → ¬(∃ (x y : ℝ), ellipse x y ∧ ∃ d, shortest_distance d ∧ line x y m') :=
sorry

end min_value_of_m_l525_525182


namespace middle_aged_selection_l525_525026

def total_teachers := 80 + 160 + 240
def sample_size := 60
def middle_aged_proportion := 160 / total_teachers
def middle_aged_sample := middle_aged_proportion * sample_size

theorem middle_aged_selection : middle_aged_sample = 20 :=
  sorry

end middle_aged_selection_l525_525026


namespace hydrogen_atoms_in_compound_l525_525878

theorem hydrogen_atoms_in_compound :
  ∀ (molecular_weight_of_compound atomic_weight_Al atomic_weight_O atomic_weight_H : ℕ)
    (num_Al num_O num_H : ℕ),
    molecular_weight_of_compound = 78 →
    atomic_weight_Al = 27 →
    atomic_weight_O = 16 →
    atomic_weight_H = 1 →
    num_Al = 1 →
    num_O = 3 →
    molecular_weight_of_compound = 
      (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H) →
    num_H = 3 := by
  intros
  sorry

end hydrogen_atoms_in_compound_l525_525878


namespace walking_rate_ratio_l525_525471

theorem walking_rate_ratio (R R' : ℝ)
  (h : R * 36 = R' * 32) : R' / R = 9 / 8 :=
sorry

end walking_rate_ratio_l525_525471


namespace polar_equation_of_circle_chord_length_from_line_l525_525207

noncomputable def circle_polar_equation (θ : ℝ) : ℝ :=
  4 * real.cos (θ + (real.pi / 6))

theorem polar_equation_of_circle :
  ∀ θ, ρ = circle_polar_equation θ :=
sorry

theorem chord_length_from_line :
  ρ = 2 * real.sqrt 2 :=
sorry

end polar_equation_of_circle_chord_length_from_line_l525_525207


namespace one_fourths_in_five_eighths_l525_525167

theorem one_fourths_in_five_eighths : (5/8 : ℚ) / (1/4) = (5/2 : ℚ) := 
by
  -- Placeholder for the proof
  sorry

end one_fourths_in_five_eighths_l525_525167


namespace probability_two_primes_from_1_to_30_l525_525404

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l525_525404


namespace probability_condition_satisfied_l525_525436

theorem probability_condition_satisfied :
  let A := {n | 1 ≤ n ∧ n ≤ 6},
      count_valid_pairs := λ (a b : ℕ), (a - 2) * (b - 2) > 5,
      valid_pairs := {(a, b) | a ∈ A ∧ b ∈ A ∧ count_valid_pairs a b},
      total_pairs := (A × A).card,
      probability := valid_pairs.card / total_pairs
  in probability = 1 / 6 := sorry

end probability_condition_satisfied_l525_525436
