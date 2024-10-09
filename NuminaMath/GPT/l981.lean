import Mathlib

namespace albert_number_l981_98107

theorem albert_number :
  ∃ (n : ℕ), (1 / (n : ℝ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) ∧ 
             ∃ m : ℕ, (1 / (m : ℝ) + 1 / 2 = 1 / 3 + 2 / (m + 1)) ∧ m ≠ n :=
sorry

end albert_number_l981_98107


namespace abcdefg_defghij_value_l981_98106

variable (a b c d e f g h i : ℚ)

theorem abcdefg_defghij_value :
  (a / b = -7 / 3) →
  (b / c = -5 / 2) →
  (c / d = 2) →
  (d / e = -3 / 2) →
  (e / f = 4 / 3) →
  (f / g = -1 / 4) →
  (g / h = 3 / -5) →
  (abcdefg / defghij = (-21 / 16) * (c / i)) :=
by
  sorry

end abcdefg_defghij_value_l981_98106


namespace largest_possible_product_is_3886_l981_98180

theorem largest_possible_product_is_3886 :
  ∃ a b c d : ℕ, 5 ≤ a ∧ a ≤ 8 ∧
               5 ≤ b ∧ b ≤ 8 ∧
               5 ≤ c ∧ c ≤ 8 ∧
               5 ≤ d ∧ d ≤ 8 ∧
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
               b ≠ c ∧ b ≠ d ∧
               c ≠ d ∧
               (max ((10 * a + b) * (10 * c + d))
                    ((10 * c + b) * (10 * a + d))) = 3886 :=
sorry

end largest_possible_product_is_3886_l981_98180


namespace tangent_identity_problem_l981_98112

theorem tangent_identity_problem 
    (α β : ℝ) 
    (h1 : Real.tan (α + β) = 1) 
    (h2 : Real.tan (α - π / 3) = 1 / 3) 
    : Real.tan (β + π / 3) = 1 / 2 := 
sorry

end tangent_identity_problem_l981_98112


namespace units_digit_of_23_mul_51_squared_l981_98133

theorem units_digit_of_23_mul_51_squared : 
  ∀ n m : ℕ, (n % 10 = 3) ∧ ((m^2 % 10) = 1) → (n * m^2 % 10) = 3 :=
by
  intros n m h
  sorry

end units_digit_of_23_mul_51_squared_l981_98133


namespace area_of_playground_l981_98126

variable (l w : ℝ)

-- Conditions:
def perimeter_eq : Prop := 2 * l + 2 * w = 90
def length_three_times_width : Prop := l = 3 * w

-- Theorem:
theorem area_of_playground (h1 : perimeter_eq l w) (h2 : length_three_times_width l w) : l * w = 379.6875 :=
  sorry

end area_of_playground_l981_98126


namespace total_matches_in_2006_world_cup_l981_98175

-- Define relevant variables and conditions
def teams := 32
def groups := 8
def top2_from_each_group := 16

-- Calculate the number of matches in Group Stage
def matches_in_group_stage :=
  let matches_per_group := 6
  matches_per_group * groups

-- Calculate the number of matches in Knockout Stage
def matches_in_knockout_stage :=
  let first_round_matches := 8
  let quarter_final_matches := 4
  let semi_final_matches := 2
  let final_and_third_place_matches := 2
  first_round_matches + quarter_final_matches + semi_final_matches + final_and_third_place_matches

-- Total number of matches
theorem total_matches_in_2006_world_cup : matches_in_group_stage + matches_in_knockout_stage = 64 := by
  sorry

end total_matches_in_2006_world_cup_l981_98175


namespace initial_sand_amount_l981_98147

theorem initial_sand_amount (lost_sand : ℝ) (arrived_sand : ℝ)
  (h1 : lost_sand = 2.4) (h2 : arrived_sand = 1.7) :
  lost_sand + arrived_sand = 4.1 :=
by
  rw [h1, h2]
  norm_num

end initial_sand_amount_l981_98147


namespace find_b_c_find_a_range_l981_98120

noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c
noncomputable def g (a b c x : ℝ) : ℝ := f a b c x + 2 * x
noncomputable def f_prime (a b x : ℝ) : ℝ := x^2 - a * x + b
noncomputable def g_prime (a b x : ℝ) : ℝ := f_prime a b x + 2

theorem find_b_c (a c : ℝ) (h_f0 : f a 0 c 0 = c) (h_tangent_y_eq_1 : 1 = c) : 
  b = 0 ∧ c = 1 :=
by
  sorry

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, g_prime a 0 x ≥ 0) ↔ a ≤ 2 * Real.sqrt 2 :=
by
  sorry

end find_b_c_find_a_range_l981_98120


namespace range_of_linear_function_l981_98105

theorem range_of_linear_function (x : ℝ) (h : -1 < x ∧ x < 1) : 
  3 < -2 * x + 5 ∧ -2 * x + 5 < 7 :=
by {
  sorry
}

end range_of_linear_function_l981_98105


namespace Bobby_ate_5_pancakes_l981_98103

theorem Bobby_ate_5_pancakes
  (total_pancakes : ℕ := 21)
  (dog_eaten : ℕ := 7)
  (leftover : ℕ := 9) :
  (total_pancakes - dog_eaten - leftover = 5) := by
  sorry

end Bobby_ate_5_pancakes_l981_98103


namespace find_m_l981_98116

theorem find_m 
  (m : ℝ) 
  (h1 : |m + 1| ≠ 0)
  (h2 : m^2 = 1) : 
  m = 1 := sorry

end find_m_l981_98116


namespace polygon_sides_eight_l981_98153

theorem polygon_sides_eight (n : ℕ) (h : 180 * (n - 2) = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_eight_l981_98153


namespace find_m_l981_98179

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 0)
def c : ℝ × ℝ := (1, -2)

-- Define the condition that a is parallel to m * b - c
def is_parallel (a : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  a.1 * v.2 = a.2 * v.1

-- The main theorem we want to prove
theorem find_m (m : ℝ) (h : is_parallel a (m * b.1 - c.1, m * b.2 - c.2)) : m = -3 :=
by {
  -- This will be filled in with the appropriate proof
  sorry
}

end find_m_l981_98179


namespace relationship_among_ys_l981_98166

-- Define the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ :=
  -2 * x + b

-- Define the points on the graph
def y1 (b : ℝ) : ℝ :=
  linear_function (-2) b

def y2 (b : ℝ) : ℝ :=
  linear_function (-1) b

def y3 (b : ℝ) : ℝ :=
  linear_function 1 b

-- Theorem to prove the relation among y1, y2, y3
theorem relationship_among_ys (b : ℝ) : y1 b > y2 b ∧ y2 b > y3 b :=
by
  sorry

end relationship_among_ys_l981_98166


namespace solution_system_solution_rational_l981_98132

-- Definitions for the system of equations
def sys_eq_1 (x y : ℤ) : Prop := 2 * x - y = 3
def sys_eq_2 (x y : ℤ) : Prop := x + y = -12

-- Theorem to prove the solution of the system of equations
theorem solution_system (x y : ℤ) (h1 : sys_eq_1 x y) (h2 : sys_eq_2 x y) : x = -3 ∧ y = -9 :=
by {
  sorry
}

-- Definition for the rational equation
def rational_eq (x : ℤ) : Prop := (2 / (1 - x) : ℚ) + 1 = (x / (1 + x) : ℚ)

-- Theorem to prove the solution of the rational equation
theorem solution_rational (x : ℤ) (h : rational_eq x) : x = -3 :=
by {
  sorry
}

end solution_system_solution_rational_l981_98132


namespace length_AB_is_4_l981_98149

section HyperbolaProof

/-- Define the hyperbola -/
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 16) - (y^2 / 8) = 1

/-- Define the line l given by x = 2√6 -/
def line_l (x : ℝ) : Prop :=
  x = 2 * Real.sqrt 6

/-- Define the condition for intersection points -/
def intersect_points (x y : ℝ) : Prop :=
  hyperbola x y ∧ line_l x

/-- Prove the length of the line segment AB is 4 -/
theorem length_AB_is_4 :
  ∀ y : ℝ, intersect_points (2 * Real.sqrt 6) y → |y| = 2 → length_AB = 4 :=
sorry

end HyperbolaProof

end length_AB_is_4_l981_98149


namespace largest_natural_number_not_sum_of_two_composites_l981_98155

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l981_98155


namespace positive_integers_n_l981_98158

theorem positive_integers_n (n a b : ℕ) (h1 : 2 < n) (h2 : n = a ^ 3 + b ^ 3) 
  (h3 : ∀ d, d > 1 ∧ d ∣ n → a ≤ d) (h4 : b ∣ n) : n = 16 ∨ n = 72 ∨ n = 520 :=
sorry

end positive_integers_n_l981_98158


namespace parallelogram_sides_l981_98104

theorem parallelogram_sides (x y : ℝ) (h₁ : 4 * x + 1 = 11) (h₂ : 10 * y - 3 = 5) : x + y = 3.3 :=
sorry

end parallelogram_sides_l981_98104


namespace andrew_paid_1428_l981_98151

-- Define the constants for the problem
def rate_per_kg_grapes : ℕ := 98
def kg_grapes : ℕ := 11

def rate_per_kg_mangoes : ℕ := 50
def kg_mangoes : ℕ := 7

-- Calculate the cost of grapes and mangoes
def cost_grapes := rate_per_kg_grapes * kg_grapes
def cost_mangoes := rate_per_kg_mangoes * kg_mangoes

-- Calculate the total amount paid
def total_amount_paid := cost_grapes + cost_mangoes

-- State the proof problem
theorem andrew_paid_1428 :
  total_amount_paid = 1428 :=
by
  -- Add the proof to verify the calculations
  sorry

end andrew_paid_1428_l981_98151


namespace remainder_of_difference_l981_98173

open Int

theorem remainder_of_difference (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 3) (h : a > b) : (a - b) % 6 = 5 :=
  sorry

end remainder_of_difference_l981_98173


namespace collinear_A₁_F_B_iff_q_eq_4_l981_98146

open Real

theorem collinear_A₁_F_B_iff_q_eq_4
  (m q : ℝ) (h_m : m ≠ 0)
  (A B : ℝ × ℝ)
  (h_A : 3 * (m * A.snd + q)^2 + 4 * A.snd^2 = 12)
  (h_B : 3 * (m * B.snd + q)^2 + 4 * B.snd^2 = 12)
  (A₁ : ℝ × ℝ := (A.fst, -A.snd))
  (F : ℝ × ℝ := (1, 0)) :
  ((q = 4) ↔ (∃ k : ℝ, k * (F.fst - A₁.fst) = F.snd - A₁.snd ∧ k * (B.fst - F.fst) = B.snd - F.snd)) :=
sorry

end collinear_A₁_F_B_iff_q_eq_4_l981_98146


namespace fraction_value_is_one_fourth_l981_98124

theorem fraction_value_is_one_fourth (k : Nat) (hk : k ≥ 1) :
  (10^k + 6 * (10^k - 1) / 9) / (60 * (10^k - 1) / 9 + 4) = 1 / 4 :=
by
  sorry

end fraction_value_is_one_fourth_l981_98124


namespace volume_box_values_l981_98157

theorem volume_box_values :
  let V := (x + 3) * (x - 3) * (x^2 - 10*x + 25)
  ∃ (x_values : Finset ℕ),
    ∀ x ∈ x_values, V < 1000 ∧ x > 0 ∧ x_values.card = 3 :=
by
  sorry

end volume_box_values_l981_98157


namespace number_of_black_boxcars_l981_98170

def red_boxcars : Nat := 3
def blue_boxcars : Nat := 4
def black_boxcar_capacity : Nat := 4000
def boxcar_total_capacity : Nat := 132000

def blue_boxcar_capacity : Nat := 2 * black_boxcar_capacity
def red_boxcar_capacity : Nat := 3 * blue_boxcar_capacity

def red_boxcar_total_capacity : Nat := red_boxcars * red_boxcar_capacity
def blue_boxcar_total_capacity : Nat := blue_boxcars * blue_boxcar_capacity

def other_total_capacity : Nat := red_boxcar_total_capacity + blue_boxcar_total_capacity
def remaining_capacity : Nat := boxcar_total_capacity - other_total_capacity
def expected_black_boxcars : Nat := remaining_capacity / black_boxcar_capacity

theorem number_of_black_boxcars :
  expected_black_boxcars = 7 := by
  sorry

end number_of_black_boxcars_l981_98170


namespace sum_of_triangle_angles_is_540_l981_98136

theorem sum_of_triangle_angles_is_540
  (A1 A3 A5 B2 B4 B6 C7 C8 C9 : ℝ)
  (H1 : A1 + A3 + A5 = 180)
  (H2 : B2 + B4 + B6 = 180)
  (H3 : C7 + C8 + C9 = 180) :
  A1 + A3 + A5 + B2 + B4 + B6 + C7 + C8 + C9 = 540 :=
by
  sorry

end sum_of_triangle_angles_is_540_l981_98136


namespace least_multiple_of_25_gt_450_correct_l981_98139

def least_multiple_of_25_gt_450 : ℕ :=
  475

theorem least_multiple_of_25_gt_450_correct (n : ℕ) (h1 : 25 ∣ n) (h2 : n > 450) : n ≥ least_multiple_of_25_gt_450 :=
by
  sorry

end least_multiple_of_25_gt_450_correct_l981_98139


namespace cost_price_of_radio_l981_98150

-- Define the conditions
def selling_price : ℝ := 1335
def loss_percentage : ℝ := 0.11

-- Define what we need to prove
theorem cost_price_of_radio (C : ℝ) (h1 : selling_price = 0.89 * C) : C = 1500 :=
by
  -- This is where we would put the proof, but we can leave it as a sorry for now.
  sorry

end cost_price_of_radio_l981_98150


namespace given_expression_equality_l981_98193

theorem given_expression_equality (x : ℝ) (A ω φ b : ℝ) (hA : 0 < A)
  (h : 2 * (Real.cos x)^2 + Real.sin (2 * x) = A * Real.sin (ω * x + φ) + b) :
  A = Real.sqrt 2 ∧ b = 1 :=
sorry

end given_expression_equality_l981_98193


namespace inequality_abcde_l981_98144

theorem inequality_abcde
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d) : 
  1 / a + 1 / b + 4 / c + 16 / d ≥ 64 / (a + b + c + d) := 
  sorry

end inequality_abcde_l981_98144


namespace range_of_q_l981_98174

variable (a_n : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
variable (hg_seq : ∀ n : ℕ, n > 0 → ∃ a_1 : ℝ, S_n n = a_1 * (1 - q ^ n) / (1 - q))
variable (pos_sum : ∀ n : ℕ, n > 0 → S_n n > 0)

theorem range_of_q : q ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (0 : ℝ) := sorry

end range_of_q_l981_98174


namespace count_even_divisors_8_l981_98127

theorem count_even_divisors_8! :
  ∃ (even_divisors total : ℕ),
    even_divisors = 84 ∧
    total = 56 :=
by
  /-
    To formulate the problem in Lean:
    We need to establish two main facts:
    1. The count of even divisors of 8! is 84.
    2. The count of those even divisors that are multiples of both 2 and 3 is 56.
  -/
  sorry

end count_even_divisors_8_l981_98127


namespace length_of_AB_l981_98154
-- Import the necessary libraries

-- Define the quadratic function
def quad (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define a predicate to state that x is a root of the quadratic
def is_root (x : ℝ) : Prop := quad x = 0

-- Define the length between the intersection points
theorem length_of_AB :
  (is_root (-1)) ∧ (is_root 3) → |3 - (-1)| = 4 :=
by {
  sorry
}

end length_of_AB_l981_98154


namespace grill_ran_for_16_hours_l981_98156

def coals_burn_time_A (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 15 * 20)) 0

def coals_burn_time_B (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 10 * 30)) 0

def total_grill_time (bags_A bags_B : List ℕ) : ℕ :=
  coals_burn_time_A bags_A + coals_burn_time_B bags_B

def bags_A : List ℕ := [60, 75, 45]
def bags_B : List ℕ := [50, 70, 40, 80]

theorem grill_ran_for_16_hours :
  total_grill_time bags_A bags_B = 960 / 60 :=
by
  unfold total_grill_time coals_burn_time_A coals_burn_time_B
  unfold bags_A bags_B
  norm_num
  sorry

end grill_ran_for_16_hours_l981_98156


namespace count_prime_sum_112_l981_98194

noncomputable def primeSum (primes : List ℕ) : ℕ :=
  if H : ∀ p ∈ primes, Nat.Prime p ∧ p > 10 then primes.sum else 0

theorem count_prime_sum_112 :
  ∃ (primes : List ℕ), primeSum primes = 112 ∧ primes.length = 6 := by
  sorry

end count_prime_sum_112_l981_98194


namespace shark_sightings_relationship_l981_98140

theorem shark_sightings_relationship (C D R : ℕ) (h₁ : C + D = 40) (h₂ : C = R - 8) (h₃ : C = 24) :
  R = 32 :=
by
  sorry

end shark_sightings_relationship_l981_98140


namespace gcf_180_270_l981_98184

theorem gcf_180_270 : Int.gcd 180 270 = 90 :=
sorry

end gcf_180_270_l981_98184


namespace obtuse_and_acute_angles_in_convex_octagon_l981_98177

theorem obtuse_and_acute_angles_in_convex_octagon (m n : ℕ) (h₀ : n + m = 8) : m > n :=
sorry

end obtuse_and_acute_angles_in_convex_octagon_l981_98177


namespace eggs_total_l981_98176

-- Definitions based on conditions
def isPackageSize (n : Nat) : Prop :=
  n = 6 ∨ n = 11

def numLargePacks : Nat := 5

def largePackSize : Nat := 11

-- Mathematical statement to prove
theorem eggs_total : ∃ totalEggs : Nat, totalEggs = numLargePacks * largePackSize :=
  by sorry

end eggs_total_l981_98176


namespace pool_capacity_l981_98135

noncomputable def total_capacity : ℝ := 1000

theorem pool_capacity
    (C : ℝ)
    (H1 : 0.75 * C = 0.45 * C + 300)
    (H2 : 300 / 0.3 = 1000)
    : C = total_capacity :=
by
  -- Solution steps are omitted, proof goes here.
  sorry

end pool_capacity_l981_98135


namespace ordering_eight_four_three_l981_98164

noncomputable def eight_pow_ten := 8 ^ 10
noncomputable def four_pow_fifteen := 4 ^ 15
noncomputable def three_pow_twenty := 3 ^ 20

theorem ordering_eight_four_three :
  eight_pow_ten < three_pow_twenty ∧ three_pow_twenty < four_pow_fifteen :=
by
  sorry

end ordering_eight_four_three_l981_98164


namespace range_of_a_l981_98138

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₀ d : ℝ), ∀ n, a n = a₀ + n * d

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a_seq) 
  (h2 : a_seq 0 = a)
  (h3 : ∀ n, b n = (1 + a_seq n) / a_seq n)
  (h4 : ∀ n : ℕ, 0 < n → b n ≥ b 8) :
  -8 < a ∧ a < -7 :=
sorry

end range_of_a_l981_98138


namespace a_100_value_l981_98128

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0     => 0    -- using 0-index for convenience
| (n+1) => a n + 4

-- Prove the value of the 100th term in the sequence
theorem a_100_value : a 100 = 397 := 
by {
  -- proof would go here
  sorry
}

end a_100_value_l981_98128


namespace ratio_is_one_half_l981_98172

-- Define the problem conditions as constants
def robert_age_in_2_years : ℕ := 30
def years_until_robert_is_30 : ℕ := 2
def patrick_current_age : ℕ := 14

-- Using the conditions, set up the definitions for the proof
def robert_current_age : ℕ := robert_age_in_2_years - years_until_robert_is_30

-- Define the target ratio
def ratio_of_ages : ℚ := patrick_current_age / robert_current_age

-- Prove that the ratio of Patrick's age to Robert's age is 1/2
theorem ratio_is_one_half : ratio_of_ages = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l981_98172


namespace length_of_notebook_is_24_l981_98129

-- Definitions
def span_of_hand : ℕ := 12
def length_of_notebook (span : ℕ) : ℕ := 2 * span

-- Theorem statement that proves the question == answer given conditions
theorem length_of_notebook_is_24 :
  length_of_notebook span_of_hand = 24 :=
sorry

end length_of_notebook_is_24_l981_98129


namespace fraction_increase_invariance_l981_98188

theorem fraction_increase_invariance (x y : ℝ) :
  (3 * (2 * y)) / (2 * x + 2 * y) = 3 * y / (x + y) :=
by
  sorry

end fraction_increase_invariance_l981_98188


namespace geometric_sequence_product_l981_98198

variable (a : ℕ → ℝ)

def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (h_geom : is_geometric_seq a) (h_a6 : a 6 = 3) :
  a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 2187 := by
  sorry

end geometric_sequence_product_l981_98198


namespace julia_tuesday_kids_l981_98101

-- Definitions based on the given conditions in the problem.
def monday_kids : ℕ := 15
def monday_tuesday_kids : ℕ := 33

-- The problem statement to prove the number of kids played with on Tuesday.
theorem julia_tuesday_kids :
  (∃ tuesday_kids : ℕ, tuesday_kids = monday_tuesday_kids - monday_kids) →
  18 = monday_tuesday_kids - monday_kids :=
by
  intro h
  sorry

end julia_tuesday_kids_l981_98101


namespace new_profit_percentage_l981_98190

theorem new_profit_percentage (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P) 
  (h2 : SP = 879.9999999999993) 
  (h3 : NP = 0.90 * P) 
  (h4 : NSP = SP + 56) : 
  (NSP - NP) / NP * 100 = 30 := 
by
  sorry

end new_profit_percentage_l981_98190


namespace common_divisor_l981_98102

theorem common_divisor (d : ℕ) (h1 : 30 % d = 3) (h2 : 40 % d = 4) : d = 9 :=
by 
  sorry

end common_divisor_l981_98102


namespace max_value_of_expression_l981_98108

noncomputable def max_expression_value (x y : ℝ) : ℝ :=
  let expr := x^2 + 6 * y + 2
  14

theorem max_value_of_expression 
  (x y : ℝ) (h : x^2 + y^2 = 4) : ∃ (M : ℝ), M = 14 ∧ ∀ x y, x^2 + y^2 = 4 → x^2 + 6 * y + 2 ≤ M :=
  by
    use 14
    sorry

end max_value_of_expression_l981_98108


namespace fraction_given_to_jerry_l981_98143

-- Define the problem conditions
def initial_apples := 2
def slices_per_apple := 8
def total_slices := initial_apples * slices_per_apple -- 2 * 8 = 16

def remaining_slices_after_eating := 5
def slices_before_eating := remaining_slices_after_eating * 2 -- 5 * 2 = 10
def slices_given_to_jerry := total_slices - slices_before_eating -- 16 - 10 = 6

-- Define the proof statement to verify that the fraction of slices given to Jerry is 3/8
theorem fraction_given_to_jerry : (slices_given_to_jerry : ℚ) / total_slices = 3 / 8 :=
by
  -- skip the actual proof, just outline the goal
  sorry

end fraction_given_to_jerry_l981_98143


namespace part1_part2_l981_98159

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 - (4 / (2 * a^x + a))

theorem part1 (h₁ : ∀ x, f a x = -f a (-x)) (h₂ : a > 0) (h₃ : a ≠ 1) : a = 2 :=
  sorry

theorem part2 (h₁ : a = 2) (x : ℝ) (hx : 0 < x ∧ x ≤ 1) (t : ℝ) :
  t * (f a x) ≥ 2^x - 2 ↔ t ≥ 0 :=
  sorry

end part1_part2_l981_98159


namespace tutoring_minutes_l981_98197

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end tutoring_minutes_l981_98197


namespace expression_value_l981_98192

theorem expression_value (a b c d : ℝ) (h1 : a * b = 1) (h2 : c + d = 0) :
  -((a * b) ^ (1/3)) + (c + d).sqrt + 1 = 0 :=
by sorry

end expression_value_l981_98192


namespace plane_split_into_four_regions_l981_98152

theorem plane_split_into_four_regions {x y : ℝ} :
  (y = 3 * x) ∨ (y = (1 / 3) * x - (2 / 3)) →
  ∃ r : ℕ, r = 4 :=
by
  intro h
  -- We must show that these lines split the plane into 4 regions
  sorry

end plane_split_into_four_regions_l981_98152


namespace find_two_digit_number_l981_98130

theorem find_two_digit_number (x : ℕ) (h1 : (x + 3) % 3 = 0) (h2 : (x + 7) % 7 = 0) (h3 : (x - 4) % 4 = 0) : x = 84 := 
by
  -- Place holder for the proof
  sorry

end find_two_digit_number_l981_98130


namespace solve_quadratic_equation_l981_98100

theorem solve_quadratic_equation (m : ℝ) : 9 * m^2 - (2 * m + 1)^2 = 0 → m = 1 ∨ m = -1/5 :=
by
  intro h
  sorry

end solve_quadratic_equation_l981_98100


namespace virginia_more_years_l981_98171

variable {V A D x : ℕ}

theorem virginia_more_years (h1 : V + A + D = 75) (h2 : D = 34) (h3 : V = A + x) (h4 : V = D - x) : x = 9 :=
by
  sorry

end virginia_more_years_l981_98171


namespace jo_age_l981_98122

theorem jo_age (j d g : ℕ) (even_j : 2 * j = j * 2) (even_d : 2 * d = d * 2) (even_g : 2 * g = g * 2)
    (h : 8 * j * d * g = 2024) : 2 * j = 46 :=
sorry

end jo_age_l981_98122


namespace isosceles_triangle_angles_l981_98118

theorem isosceles_triangle_angles (α β γ : ℝ) (h_iso : α = β ∨ α = γ ∨ β = γ) (h_angle : α + β + γ = 180) (h_40 : α = 40 ∨ β = 40 ∨ γ = 40) :
  (α = 70 ∧ β = 70 ∧ γ = 40) ∨ (α = 40 ∧ β = 100 ∧ γ = 40) ∨ (α = 40 ∧ β = 40 ∧ γ = 100) :=
by
  sorry

end isosceles_triangle_angles_l981_98118


namespace curve_is_line_segment_l981_98145

noncomputable def parametric_curve : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p.1 = Real.cos θ ^ 2 ∧ p.2 = Real.sin θ ^ 2}

theorem curve_is_line_segment :
  (∀ p ∈ parametric_curve, p.1 + p.2 = 1 ∧ p.1 ∈ Set.Icc 0 1) :=
by
  sorry

end curve_is_line_segment_l981_98145


namespace parallel_lines_slope_l981_98168

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by
  sorry

end parallel_lines_slope_l981_98168


namespace slope_angle_line_l981_98131
open Real

theorem slope_angle_line (x y : ℝ) :
  x + sqrt 3 * y - 1 = 0 → ∃ θ : ℝ, θ = 150 ∧
  ∃ (m : ℝ), m = -sqrt 3 / 3 ∧ θ = arctan m :=
by
  sorry

end slope_angle_line_l981_98131


namespace trigonometric_identity_l981_98195

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = -3) :
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 :=
  sorry

end trigonometric_identity_l981_98195


namespace train_can_speed_up_l981_98162

theorem train_can_speed_up (d t_reduced v_increased v_safe : ℝ) 
  (h1 : d = 1600) (h2 : t_reduced = 4) (h3 : v_increased = 20) (h4 : v_safe = 140) :
  ∃ x : ℝ, (x > 0) ∧ (d / x) = (d / (x + v_increased) + t_reduced) ∧ ((x + v_increased) < v_safe) :=
by 
  sorry

end train_can_speed_up_l981_98162


namespace min_value_of_expression_l981_98109

theorem min_value_of_expression (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y = 5 := 
sorry

end min_value_of_expression_l981_98109


namespace initial_average_mark_l981_98111

theorem initial_average_mark (A : ℝ) (n_total n_excluded remaining_students_avg : ℝ) 
  (h1 : n_total = 25) 
  (h2 : n_excluded = 5) 
  (h3 : remaining_students_avg = 90)
  (excluded_students_avg : ℝ)
  (h_excluded_avg : excluded_students_avg = 40)
  (A_def : (n_total * A) = (n_excluded * excluded_students_avg + (n_total - n_excluded) * remaining_students_avg)) :
  A = 80 := 
by
  sorry

end initial_average_mark_l981_98111


namespace rope_length_l981_98123

theorem rope_length (h1 : ∃ x : ℝ, 4 * x = 20) : 
  ∃ l : ℝ, l = 35 := by
sorry

end rope_length_l981_98123


namespace problem_l981_98110

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.log x + (a + 1) * (1 / x - 2)

theorem problem (a x : ℝ) (ha_pos : a > 0) :
  f a x > - (a^2 / (a + 1)) - 2 :=
sorry

end problem_l981_98110


namespace erasers_total_l981_98185

-- Define the initial amount of erasers
def initialErasers : Float := 95.0

-- Define the amount of erasers Marie buys
def boughtErasers : Float := 42.0

-- Define the total number of erasers Marie ends with
def totalErasers : Float := 137.0

-- The theorem that needs to be proven
theorem erasers_total 
  (initial : Float := initialErasers)
  (bought : Float := boughtErasers)
  (total : Float := totalErasers) :
  initial + bought = total :=
sorry

end erasers_total_l981_98185


namespace g_f_x_not_quadratic_l981_98167

open Real

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_f_x_not_quadratic (h : ∃ x : ℝ, x - f (g x) = 0) :
  ∀ x : ℝ, g (f x) ≠ x^2 + x + 1 / 5 := sorry

end g_f_x_not_quadratic_l981_98167


namespace fraction_calculation_l981_98187

theorem fraction_calculation : (8 / 24) - (5 / 72) + (3 / 8) = 23 / 36 :=
by
  sorry

end fraction_calculation_l981_98187


namespace inscribed_circle_radius_l981_98160

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (θ : ℝ) (tangent : ℝ) :
    θ = π / 3 →
    R = 5 →
    tangent = (5 : ℝ) * (Real.sqrt 2 - 1) →
    r * (1 + Real.sqrt 2) = R →
    r = 5 * (Real.sqrt 2 - 1) := 
by sorry

end inscribed_circle_radius_l981_98160


namespace polar_to_rectangular_l981_98137

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 7) (h_θ : θ = π / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) :=
by 
  -- proof goes here
  sorry

end polar_to_rectangular_l981_98137


namespace range_of_f3_l981_98163

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 1

theorem range_of_f3 {a b : ℝ}
  (h1 : -2 ≤ a - b ∧ a - b ≤ 0) 
  (h2 : -3 ≤ 4 * a + 2 * b ∧ 4 * a + 2 * b ≤ 1) :
  -7 ≤ f a b 3 ∧ f a b 3 ≤ 3 :=
sorry

end range_of_f3_l981_98163


namespace remainder_when_P_divided_by_DD_l981_98114

noncomputable def remainder (a b : ℕ) : ℕ := a % b

theorem remainder_when_P_divided_by_DD' (P D Q R D' Q'' R'' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q^2 = D' * Q'' + R'') :
  remainder P (D * D') = R :=
by {
  sorry
}

end remainder_when_P_divided_by_DD_l981_98114


namespace gears_can_rotate_l981_98161

theorem gears_can_rotate (n : ℕ) : (∃ f : ℕ → Prop, f 0 ∧ (∀ k, f (k+1) ↔ ¬f k) ∧ f n = f 0) ↔ (n % 2 = 0) :=
by
  sorry

end gears_can_rotate_l981_98161


namespace daily_evaporation_l981_98199

theorem daily_evaporation :
  ∀ (initial_amount : ℝ) (percentage_evaporated : ℝ) (days : ℕ),
  initial_amount = 10 →
  percentage_evaporated = 6 →
  days = 50 →
  (initial_amount * (percentage_evaporated / 100)) / days = 0.012 :=
by
  intros initial_amount percentage_evaporated days
  intros h_initial h_percentage h_days
  rw [h_initial, h_percentage, h_days]
  sorry

end daily_evaporation_l981_98199


namespace fraction_auto_installment_credit_extended_by_finance_companies_l981_98119

def total_consumer_installment_credit : ℝ := 291.6666666666667
def auto_instalment_percentage : ℝ := 0.36
def auto_finance_companies_credit_extended : ℝ := 35

theorem fraction_auto_installment_credit_extended_by_finance_companies :
  auto_finance_companies_credit_extended / (auto_instalment_percentage * total_consumer_installment_credit) = 1 / 3 :=
by
  sorry

end fraction_auto_installment_credit_extended_by_finance_companies_l981_98119


namespace smallest_k_exists_l981_98183

open Nat

theorem smallest_k_exists (n m k : ℕ) (hn : n > 0) (hm : 0 < m ∧ m ≤ 5) (hk : k % 3 = 0) :
  (64^k + 32^m > 4^(16 + n^2)) ↔ k = 6 :=
by
  sorry

end smallest_k_exists_l981_98183


namespace solve_equation_l981_98148

theorem solve_equation :
  ∃ (a b c d : ℚ), 
  (a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + 2 / 5 = 0) ∧ 
  (a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5) := sorry

end solve_equation_l981_98148


namespace largest_inscribed_triangle_area_l981_98134

theorem largest_inscribed_triangle_area (r : ℝ) (h_r : r = 12) : ∃ A : ℝ, A = 144 :=
by
  sorry

end largest_inscribed_triangle_area_l981_98134


namespace sufficient_condition_l981_98117

theorem sufficient_condition 
  (x y z : ℤ)
  (H : x = y ∧ y = z)
  : x * (x - y) + y * (y - z) + z * (z - x) = 0 :=
by 
  sorry

end sufficient_condition_l981_98117


namespace jen_profit_l981_98178

-- Definitions based on the conditions
def cost_per_candy := 80 -- in cents
def sell_price_per_candy := 100 -- in cents
def total_candies_bought := 50
def total_candies_sold := 48

-- Total cost and total revenue calculations
def total_cost := cost_per_candy * total_candies_bought
def total_revenue := sell_price_per_candy * total_candies_sold

-- Profit calculation
def profit := total_revenue - total_cost

-- Main theorem to prove
theorem jen_profit : profit = 800 := by
  -- Proof is skipped
  sorry

end jen_profit_l981_98178


namespace arithmetic_sequence_ratio_l981_98182

theorem arithmetic_sequence_ratio (a x b : ℝ) 
  (h1 : x - a = b - x)
  (h2 : 2 * x - b = b - x) :
  a / b = 1 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l981_98182


namespace max_value_expr_l981_98113

def point_on_line (m n : ℝ) : Prop :=
  3 * m + n = -1

def mn_positive (m n : ℝ) : Prop :=
  m * n > 0

theorem max_value_expr (m n : ℝ) (h1 : point_on_line m n) (h2 : mn_positive m n) :
  (3 / m + 1 / n) = -16 :=
sorry

end max_value_expr_l981_98113


namespace polar_eq_of_circle_product_of_distances_MA_MB_l981_98121

noncomputable def circle_center := (2, Real.pi / 3)
noncomputable def circle_radius := 2

-- Polar equation of the circle
theorem polar_eq_of_circle :
  ∀ (ρ θ : ℝ),
    (circle_center.snd = Real.pi / 3) →
    ρ = 2 * 2 * Real.cos (θ - circle_center.snd) → 
    ρ = 4 * Real.cos (θ - (Real.pi / 3)) :=
by 
  sorry

noncomputable def point_M := (1, -2)

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ := 
  (1 + 1/2 * t, -2 + Real.sqrt 3 / 2 * t)

noncomputable def cartesian_center := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
noncomputable def cartesian_radius := 2

-- Cartesian form of the circle equation from the polar coordinates
noncomputable def cartesian_eq (x y : ℝ) : Prop :=
  (x - cartesian_center.fst)^2 + (y - cartesian_center.snd)^2 = circle_radius^2

-- Product of distances |MA| * |MB|
theorem product_of_distances_MA_MB :
  ∃ (t1 t2 : ℝ),
  (∀ t, parametric_line t ∈ {p : ℝ × ℝ | cartesian_eq p.fst p.snd}) → 
  (point_M.fst, point_M.snd) = (1, -2) →
  t1 * t2 = 3 + 4 * Real.sqrt 3 :=
by
  sorry

end polar_eq_of_circle_product_of_distances_MA_MB_l981_98121


namespace lollipops_given_l981_98141

theorem lollipops_given (initial_people later_people : ℕ) (total_people groups_of_five : ℕ) :
  initial_people = 45 →
  later_people = 15 →
  total_people = initial_people + later_people →
  groups_of_five = total_people / 5 →
  total_people = 60 →
  groups_of_five = 12 :=
by intros; sorry

end lollipops_given_l981_98141


namespace value_of_x_l981_98115

theorem value_of_x (x : ℝ) (h : x = 12 + (20 / 100) * 12) : x = 14.4 :=
by sorry

end value_of_x_l981_98115


namespace exists_solution_in_interval_l981_98125

theorem exists_solution_in_interval : ∃ x ∈ (Set.Ioo (3: ℝ) (4: ℝ)), Real.log x / Real.log 2 + x - 5 = 0 :=
by
  sorry

end exists_solution_in_interval_l981_98125


namespace hamburgers_left_over_l981_98186

-- Define the conditions as constants
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove that the number of hamburgers left over is 6
theorem hamburgers_left_over : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end hamburgers_left_over_l981_98186


namespace samBill_l981_98165

def textMessageCostPerText := 8 -- cents
def extraMinuteCostPerMinute := 15 -- cents
def planBaseCost := 25 -- dollars
def includedPlanHours := 25
def centToDollar (cents: Nat) : Nat := cents / 100

def totalBill (texts: Nat) (hours: Nat) : Nat :=
  let textCost := centToDollar (texts * textMessageCostPerText)
  let extraHours := if hours > includedPlanHours then hours - includedPlanHours else 0
  let extraMinutes := extraHours * 60
  let extraMinuteCost := centToDollar (extraMinutes * extraMinuteCostPerMinute)
  planBaseCost + textCost + extraMinuteCost

theorem samBill :
  totalBill 150 26 = 46 := 
sorry

end samBill_l981_98165


namespace second_hand_angle_after_2_minutes_l981_98169

theorem second_hand_angle_after_2_minutes :
  ∀ angle_in_radians, (∀ rotations:ℝ, rotations = 2 → one_full_circle = 2 * Real.pi → angle_in_radians = - (rotations * one_full_circle)) →
  angle_in_radians = -4 * Real.pi :=
by
  intros
  sorry

end second_hand_angle_after_2_minutes_l981_98169


namespace avg_of_last_three_l981_98189

-- Define the conditions given in the problem
def avg_5 : Nat := 54
def avg_2 : Nat := 48
def num_list_length : Nat := 5
def first_two_length : Nat := 2

-- State the theorem
theorem avg_of_last_three
    (h_avg5 : 5 * avg_5 = 270)
    (h_avg2 : 2 * avg_2 = 96) :
  (270 - 96) / 3 = 58 :=
sorry

end avg_of_last_three_l981_98189


namespace initial_balls_count_l981_98191

variables (y w : ℕ)

theorem initial_balls_count (h1 : y = 2 * (w - 10)) (h2 : w - 10 = 5 * (y - 9)) :
  y = 10 ∧ w = 15 :=
sorry

end initial_balls_count_l981_98191


namespace prob_sum_7_9_11_correct_l981_98181

def die1 : List ℕ := [1, 2, 3, 3, 4, 4]
def die2 : List ℕ := [2, 2, 5, 6, 7, 8]

def prob_sum_7_9_11 : ℚ := 
  (1/6 * 1/6 + 1/6 * 1/6) + 2/6 * 3/6

theorem prob_sum_7_9_11_correct :
  prob_sum_7_9_11 = 4 / 9 := 
by
  sorry

end prob_sum_7_9_11_correct_l981_98181


namespace distance_between_parallel_lines_l981_98196

theorem distance_between_parallel_lines 
  (d : ℝ) 
  (r : ℝ)
  (h1 : (42 * 21 + (d / 2) * 42 * (d / 2) = 42 * r^2))
  (h2 : (40 * 20 + (3 * d / 2) * 40 * (3 * d / 2) = 40 * r^2)) :
  d = 3 + 3 / 8 :=
  sorry

end distance_between_parallel_lines_l981_98196


namespace green_sweets_count_l981_98142

def total_sweets := 285
def red_sweets := 49
def neither_red_nor_green_sweets := 177

theorem green_sweets_count : 
  (total_sweets - red_sweets - neither_red_nor_green_sweets) = 59 :=
by
  -- The proof will go here
  sorry

end green_sweets_count_l981_98142
