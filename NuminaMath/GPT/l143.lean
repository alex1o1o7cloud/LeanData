import Mathlib

namespace brad_weighs_more_l143_143858

theorem brad_weighs_more :
  ∀ (Billy Brad Carl : ℕ), 
    (Billy = Brad + 9) → 
    (Carl = 145) → 
    (Billy = 159) → 
    (Brad - Carl = 5) :=
by
  intros Billy Brad Carl h1 h2 h3
  sorry

end brad_weighs_more_l143_143858


namespace total_bill_correct_l143_143125

def scoop_cost : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

def pierre_total : ℕ := pierre_scoops * scoop_cost
def mom_total : ℕ := mom_scoops * scoop_cost
def total_bill : ℕ := pierre_total + mom_total

theorem total_bill_correct : total_bill = 14 :=
by
  sorry

end total_bill_correct_l143_143125


namespace john_paid_more_than_jane_by_540_l143_143111

noncomputable def original_price : ℝ := 36.000000000000036
noncomputable def discount_percentage : ℝ := 0.10
noncomputable def tip_percentage : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price * (1 - discount_percentage)
noncomputable def john_tip : ℝ := original_price * tip_percentage
noncomputable def jane_tip : ℝ := discounted_price * tip_percentage

noncomputable def john_total_payment : ℝ := discounted_price + john_tip
noncomputable def jane_total_payment : ℝ := discounted_price + jane_tip

noncomputable def difference : ℝ := john_total_payment - jane_total_payment

theorem john_paid_more_than_jane_by_540 :
  difference = 0.5400000000000023 := sorry

end john_paid_more_than_jane_by_540_l143_143111


namespace find_quarters_l143_143788

-- Define the conditions
def quarters_bounds (q : ℕ) : Prop :=
  8 < q ∧ q < 80

def stacks_mod4 (q : ℕ) : Prop :=
  q % 4 = 2

def stacks_mod6 (q : ℕ) : Prop :=
  q % 6 = 2

def stacks_mod8 (q : ℕ) : Prop :=
  q % 8 = 2

-- The theorem to prove
theorem find_quarters (q : ℕ) (h_bounds : quarters_bounds q) (h4 : stacks_mod4 q) (h6 : stacks_mod6 q) (h8 : stacks_mod8 q) : 
  q = 26 :=
by
  sorry

end find_quarters_l143_143788


namespace solution_set_l143_143768

open Real

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the differentiable function f

axiom differentiable_f : Differentiable ℝ f
axiom condition_f : ∀ x, f x > 0 ∧ x * (deriv (deriv (deriv f))) x > 0

theorem solution_set :
  {x : ℝ | 1 ≤ x ∧ x < 2} =
    {x : ℝ | f (sqrt (x + 1)) > sqrt (x - 1) * f (sqrt (x ^ 2 - 1))} :=
sorry

end solution_set_l143_143768


namespace largest_e_possible_statement_l143_143762

noncomputable def largest_e_possible 
  (P Q : ℝ) (circ : ℝ) (X Y Z : ℝ → ℝ) 
  (diam : ℝ) 
  (midpoint : ℝ) 
  (py_eq : ℝ) 
  (intersects_S : ℝ) 
  (intersects_T : ℝ) 
  : ℝ :=
  sorry

-- Statement of the problem in Lean
theorem largest_e_possible_statement 
  (P Q : ℝ) (circ : ℝ) (X Y Z : ℝ → ℝ) 
  (diam : ℝ) 
  (midpoint : ℝ) 
  (py_eq : ℝ) 
  (intersects_S : ℝ) 
  (intersects_T : ℝ) 
  (circ_diam : circ = 2)
  (X_mid : X midpoint = 1)
  (PY_val : PY_eq = 4/5)
  : largest_e_possible P Q circ X Y Z diam midpoint py_eq intersects_S intersects_T = 25 :=
  sorry

end largest_e_possible_statement_l143_143762


namespace income_ratio_l143_143558

theorem income_ratio (I1 I2 E1 E2 : ℕ)
  (hI1 : I1 = 3500)
  (hE_ratio : (E1:ℚ) / E2 = 3 / 2)
  (hSavings : ∀ (x y : ℕ), x - E1 = 1400 ∧ y - E2 = 1400 → x = I1 ∧ y = I2) :
  I1 / I2 = 5 / 4 :=
by
  -- The proof steps would go here
  sorry

end income_ratio_l143_143558


namespace smallest_nonprime_with_large_primes_l143_143766

theorem smallest_nonprime_with_large_primes
  (n : ℕ)
  (h1 : n > 1)
  (h2 : ¬ Prime n)
  (h3 : ∀ p : ℕ, Prime p → p ∣ n → p ≥ 20) :
  660 < n ∧ n ≤ 670 :=
sorry

end smallest_nonprime_with_large_primes_l143_143766


namespace smallest_interesting_number_l143_143834

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l143_143834


namespace fraction_comparisons_l143_143358

theorem fraction_comparisons :
  (1 / 8 : ℝ) * (3 / 7) < (1 / 8) ∧ 
  (9 / 8 : ℝ) * (1 / 5) > (9 / 8) * (1 / 8) ∧ 
  (2 / 3 : ℝ) < (2 / 3) / (6 / 11) := by
    sorry

end fraction_comparisons_l143_143358


namespace xy_sum_143_l143_143082

theorem xy_sum_143 (x y : ℕ) (h1 : x < 30) (h2 : y < 30) (h3 : x + y + x * y = 143) (h4 : 0 < x) (h5 : 0 < y) :
  x + y = 22 ∨ x + y = 23 ∨ x + y = 24 :=
by
  sorry

end xy_sum_143_l143_143082


namespace box_area_relation_l143_143029

theorem box_area_relation (a b c : ℕ) (h : a = b + c + 10) :
  (a * b) * (b * c) * (c * a) = (2 * (b + c) + 10)^2 := 
sorry

end box_area_relation_l143_143029


namespace percentage_increase_weekends_l143_143185

def weekday_price : ℝ := 18
def weekend_price : ℝ := 27

theorem percentage_increase_weekends : 
  (weekend_price - weekday_price) / weekday_price * 100 = 50 := by
  sorry

end percentage_increase_weekends_l143_143185


namespace largest_d_l143_143249

theorem largest_d (a b c d : ℝ) (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := sorry

end largest_d_l143_143249


namespace find_triangle_l143_143546

theorem find_triangle : ∀ (triangle : ℕ), (∀ (d : ℕ), 0 ≤ d ∧ d ≤ 9) → (5 * 3 + triangle = 12 * triangle + 4) → triangle = 1 :=
by
  sorry

end find_triangle_l143_143546


namespace polygon_sides_from_diagonals_l143_143826

/-- A theorem to prove that a regular polygon with 740 diagonals has 40 sides. -/
theorem polygon_sides_from_diagonals (n : ℕ) (h : (n * (n - 3)) / 2 = 740) : n = 40 := sorry

end polygon_sides_from_diagonals_l143_143826


namespace triangle_first_side_l143_143705

theorem triangle_first_side (x : ℕ) (h1 : 10 + 15 + x = 32) : x = 7 :=
by
  sorry

end triangle_first_side_l143_143705


namespace number_of_children_is_4_l143_143431

-- Define the conditions from the problem
def youngest_child_age : ℝ := 1.5
def sum_of_ages : ℝ := 12
def common_difference : ℝ := 1

-- Define the number of children
def n : ℕ := 4

-- Prove that the number of children is 4 given the conditions
theorem number_of_children_is_4 :
  (∃ n : ℕ, (n / 2) * (2 * youngest_child_age + (n - 1) * common_difference) = sum_of_ages) ↔ n = 4 :=
by sorry

end number_of_children_is_4_l143_143431


namespace find_b_l143_143063

theorem find_b (a b : ℝ) (x : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^5) * x^5):
  b = 40 :=
  sorry

end find_b_l143_143063


namespace part1_part2_l143_143077

variable (a : ℝ)

-- Defining the set A
def setA (a : ℝ) : Set ℝ := { x : ℝ | (x - 2) * (x - 3 * a - 1) < 0 }

-- Part 1: For a = 2, setB should be {x | 2 < x < 7}
theorem part1 : setA 2 = { x : ℝ | 2 < x ∧ x < 7 } :=
by
  sorry

-- Part 2: If setA a = setB, then a = -1
theorem part2 (B : Set ℝ) (h : setA a = B) : a = -1 :=
by
  sorry

end part1_part2_l143_143077


namespace sum_of_exponents_sqrt_l143_143542

theorem sum_of_exponents_sqrt (a b c : ℕ) : 2 + 4 + 6 = 12 := by
  sorry

end sum_of_exponents_sqrt_l143_143542


namespace probability_cello_viola_same_tree_l143_143307

noncomputable section

def cellos : ℕ := 800
def violas : ℕ := 600
def cello_viola_pairs_same_tree : ℕ := 100

theorem probability_cello_viola_same_tree : 
  (cello_viola_pairs_same_tree: ℝ) / ((cellos * violas : ℕ) : ℝ) = 1 / 4800 := 
by
  sorry

end probability_cello_viola_same_tree_l143_143307


namespace no_perpendicular_hatching_other_than_cube_l143_143121

def is_convex_polyhedron (P : Polyhedron) : Prop :=
  -- Definition of a convex polyhedron
  sorry

def number_of_faces (P : Polyhedron) : ℕ :=
  -- Function returning the number of faces of polyhedron P
  sorry

def hatching_perpendicular (P : Polyhedron) : Prop :=
  -- Definition that checks if the hatching on adjacent faces of P is perpendicular
  sorry

theorem no_perpendicular_hatching_other_than_cube :
  ∀ (P : Polyhedron), is_convex_polyhedron P ∧ number_of_faces P ≠ 6 → ¬hatching_perpendicular P :=
by
  sorry

end no_perpendicular_hatching_other_than_cube_l143_143121


namespace flowers_per_bug_l143_143256

theorem flowers_per_bug (bugs : ℝ) (flowers : ℝ) (h_bugs : bugs = 2.0) (h_flowers : flowers = 3.0) :
  flowers / bugs = 1.5 :=
by
  sorry

end flowers_per_bug_l143_143256


namespace range_of_a_l143_143909

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a-1)*x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l143_143909


namespace smallest_factor_of_36_l143_143535

theorem smallest_factor_of_36 :
  ∃ a b c : ℤ, a * b * c = 36 ∧ a + b + c = 4 ∧ min (min a b) c = -4 :=
by
  sorry

end smallest_factor_of_36_l143_143535


namespace binomial_expansion_coefficient_l143_143166

-- Theorem: The coefficient of x^3 in the expansion of (x-2)^6 is -160
theorem binomial_expansion_coefficient :
  (∑ i in Finset.range (6 + 1), (Nat.choose 6 i) * (-2)^i * (x ^ (6 - i))).coeff 3 = -160 := sorry

end binomial_expansion_coefficient_l143_143166


namespace particular_solution_ODE_l143_143359

theorem particular_solution_ODE (y : ℝ → ℝ) (h : ∀ x, deriv y x + y x * Real.tan x = 0) (h₀ : y 0 = 2) :
  ∀ x, y x = 2 * Real.cos x :=
sorry

end particular_solution_ODE_l143_143359


namespace largest_integer_less_than_100_with_remainder_5_l143_143692

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l143_143692


namespace arithmetic_sequence_150th_term_l143_143584

theorem arithmetic_sequence_150th_term :
  let a₁ := 3
  let d := 4
  a₁ + (150 - 1) * d = 599 :=
by
  sorry

end arithmetic_sequence_150th_term_l143_143584


namespace intersection_correct_l143_143252

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x < 2}

theorem intersection_correct : P ∩ Q = {1} :=
by sorry

end intersection_correct_l143_143252


namespace mul_large_numbers_l143_143189

theorem mul_large_numbers : 300000 * 300000 * 3 = 270000000000 := by
  sorry

end mul_large_numbers_l143_143189


namespace binom_12_10_l143_143350

theorem binom_12_10 : nat.choose 12 10 = 66 := by
  sorry

end binom_12_10_l143_143350


namespace largest_integer_lt_100_with_remainder_5_div_8_l143_143664

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l143_143664


namespace sin_cos_value_l143_143892

variable (α : ℝ) (a b : ℝ × ℝ)
def vectors_parallel : Prop := b = (Real.sin α, Real.cos α) ∧
a = (4, 3) ∧ (∃ k : ℝ, a = (k * (Real.sin α), k * (Real.cos α)))

theorem sin_cos_value (h : vectors_parallel α a b) : ((Real.sin α) * (Real.cos α)) = 12 / 25 :=
by
  sorry

end sin_cos_value_l143_143892


namespace age_problem_l143_143314

theorem age_problem 
  (A : ℕ) 
  (x : ℕ) 
  (h1 : 3 * (A + x) - 3 * (A - 3) = A) 
  (h2 : A = 18) : 
  x = 3 := 
by 
  sorry

end age_problem_l143_143314


namespace total_books_for_girls_l143_143434

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end total_books_for_girls_l143_143434


namespace age_of_Rahim_l143_143507

theorem age_of_Rahim (R : ℕ) (h1 : ∀ (a : ℕ), a = (R + 1) → (a + 5) = (2 * R)) (h2 : ∀ (a : ℕ), a = (R + 1) → a = R + 1) :
  R = 6 := by
  sorry

end age_of_Rahim_l143_143507


namespace triplet_not_equal_to_one_l143_143443

def A := (1/2, 1/3, 1/6)
def B := (2, -2, 1)
def C := (0.1, 0.3, 0.6)
def D := (1.1, -2.1, 1.0)
def E := (-3/2, -5/2, 5)

theorem triplet_not_equal_to_one (ha : A = (1/2, 1/3, 1/6))
                                (hb : B = (2, -2, 1))
                                (hc : C = (0.1, 0.3, 0.6))
                                (hd : D = (1.1, -2.1, 1.0))
                                (he : E = (-3/2, -5/2, 5)) :
  (1/2 + 1/3 + 1/6 = 1) ∧
  (2 + -2 + 1 = 1) ∧
  (0.1 + 0.3 + 0.6 = 1) ∧
  (1.1 + -2.1 + 1.0 ≠ 1) ∧
  (-3/2 + -5/2 + 5 = 1) :=
by {
  sorry
}

end triplet_not_equal_to_one_l143_143443


namespace arithmetic_mean_pq_l143_143793

variable (p q r : ℝ)

-- Definitions from conditions
def condition1 := (p + q) / 2 = 10
def condition2 := (q + r) / 2 = 26
def condition3 := r - p = 32

-- Theorem statement
theorem arithmetic_mean_pq : condition1 p q → condition2 q r → condition3 p r → (p + q) / 2 = 10 :=
by
  intros h1 h2 h3
  exact h1

end arithmetic_mean_pq_l143_143793


namespace find_xy_l143_143440

variable {x y : ℝ}

theorem find_xy (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end find_xy_l143_143440


namespace lcm_inequality_l143_143493

open Nat

-- Assume positive integers n and m, with n > m
theorem lcm_inequality (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n > m) :
  Nat.lcm m n + Nat.lcm (m+1) (n+1) ≥ 2 * m * Real.sqrt n := 
  sorry

end lcm_inequality_l143_143493


namespace smallest_interesting_number_l143_143836

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l143_143836


namespace terry_nora_age_relation_l143_143235

variable {N : ℕ} -- Nora's current age

theorem terry_nora_age_relation (h₁ : Terry_current_age = 30) (h₂ : Terry_future_age = 4 * N) : N = 10 :=
by
  --- additional assumptions
  have Terry_future_age_def : Terry_future_age = 30 + 10 := by sorry
  rw [Terry_future_age_def] at h₂
  linarith

end terry_nora_age_relation_l143_143235


namespace sequence_bounds_l143_143442

theorem sequence_bounds :
    ∀ (a : ℕ → ℝ), a 0 = 5 → (∀ n : ℕ, a (n + 1) = a n + 1 / a n) → 45 < a 1000 ∧ a 1000 < 45.1 :=
by
  intros a h0 h_rec
  sorry

end sequence_bounds_l143_143442


namespace rate_of_interest_per_annum_l143_143457

theorem rate_of_interest_per_annum (SI P : ℝ) (T : ℕ) (hSI : SI = 4016.25) (hP : P = 10040.625) (hT : T = 5) :
  (SI * 100) / (P * T) = 8 :=
by 
  -- Given simple interest formula
  -- SI = P * R * T / 100, solving for R we get R = (SI * 100) / (P * T)
  -- Substitute SI = 4016.25, P = 10040.625, and T = 5
  -- (4016.25 * 100) / (10040.625 * 5) = 8
  sorry

end rate_of_interest_per_annum_l143_143457


namespace proportion_problem_l143_143378

theorem proportion_problem 
  (x : ℝ) 
  (third_number : ℝ) 
  (h1 : 0.75 / x = third_number / 8) 
  (h2 : x = 0.6) 
  : third_number = 10 := 
by 
  sorry

end proportion_problem_l143_143378


namespace upper_bound_of_third_inequality_l143_143232

variable (x : ℤ)

theorem upper_bound_of_third_inequality : (3 < x ∧ x < 10) →
                                          (5 < x ∧ x < 18) →
                                          (∃ n, n > x ∧ x > -2) →
                                          (0 < x ∧ x < 8) →
                                          (x + 1 < 9) →
                                          x < 8 :=
by { sorry }

end upper_bound_of_third_inequality_l143_143232


namespace arithmetic_geometric_sequences_sequence_sum_first_terms_l143_143883

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, b (n + 1) = b n * q

noncomputable def sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) * a 1 + (n * (n + 1)) / 2

theorem arithmetic_geometric_sequences
  (a b S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : geometric_sequence b)
  (h3 : a 0 = 1)
  (h4 : b 0 = 1)
  (h5 : b 2 * S 2 = 36)
  (h6 : b 1 * S 1 = 8) :
  ((∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 2 ^ n)) ∨
  ((∀ n, a n = -(2 * n / 3) + 5 / 3) ∧ (∀ n, b n = 6 ^ n)) :=
sorry

theorem sequence_sum_first_terms
  (a : ℕ → ℤ)
  (h : ∀ n, a n = 2 * n + 1)
  (S : ℕ → ℤ)
  (T : ℕ → ℚ)
  (hS : sequence_sum a S)
  (n : ℕ) :
  T n = n / (2 * n + 1) :=
sorry

end arithmetic_geometric_sequences_sequence_sum_first_terms_l143_143883


namespace greater_number_l143_143282

theorem greater_number (x y : ℕ) (h1 : x + y = 30) (h2 : x - y = 8) (h3 : x > y) : x = 19 := 
by 
  sorry

end greater_number_l143_143282


namespace smallest_interesting_number_l143_143850

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l143_143850


namespace volume_of_set_l143_143191

theorem volume_of_set (m n p : ℕ) (h_rel_prime : Nat.gcd n p = 1) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_p : 0 < p) 
  (h_volume : (m + n * Real.pi) / p = (324 + 37 * Real.pi) / 3) : 
  m + n + p = 364 := 
  sorry

end volume_of_set_l143_143191


namespace madeline_water_intake_l143_143933

def water_bottle_capacity : ℕ := 12
def number_of_refills : ℕ := 7
def additional_water_needed : ℕ := 16
def total_water_needed : ℕ := 100

theorem madeline_water_intake : water_bottle_capacity * number_of_refills + additional_water_needed = total_water_needed :=
by
  sorry

end madeline_water_intake_l143_143933


namespace points_per_game_l143_143772

theorem points_per_game (total_points : ℝ) (num_games : ℝ) (h1 : total_points = 120.0) (h2 : num_games = 10.0) : (total_points / num_games) = 12.0 :=
by 
  rw [h1, h2]
  norm_num
  -- sorry


end points_per_game_l143_143772


namespace fraction_over_65_l143_143356

def num_people_under_21 := 33
def fraction_under_21 := 3 / 7
def total_people (N : ℕ) := N > 50 ∧ N < 100
def num_people (N : ℕ) := num_people_under_21 = fraction_under_21 * N

theorem fraction_over_65 (N : ℕ) : 
  total_people N → num_people N → N = 77 ∧ ∃ x, (x / 77) = x / 77 :=
by
  intro hN hnum
  sorry

end fraction_over_65_l143_143356


namespace area_increase_cost_increase_l143_143178

-- Given definitions based only on the conditions from part a
def original_length := 60
def original_width := 20
def original_fence_cost_per_foot := 15
def original_perimeter := 2 * (original_length + original_width)
def original_fencing_cost := original_perimeter * original_fence_cost_per_foot

def new_fence_cost_per_foot := 20
def new_square_side := original_perimeter / 4
def new_square_area := new_square_side * new_square_side
def new_fencing_cost := original_perimeter * new_fence_cost_per_foot

-- Proof statements using the conditions and correct answers from part b
theorem area_increase : new_square_area - (original_length * original_width) = 400 := by
  sorry

theorem cost_increase : new_fencing_cost - original_fencing_cost = 800 := by
  sorry

end area_increase_cost_increase_l143_143178


namespace binomial_12_10_eq_66_l143_143347

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l143_143347


namespace paper_thickness_after_2_folds_l143_143296

theorem paper_thickness_after_2_folds:
  ∀ (initial_thickness : ℝ) (folds : ℕ),
  initial_thickness = 0.1 →
  folds = 2 →
  (initial_thickness * 2^folds = 0.4) :=
by
  intros initial_thickness folds h_initial h_folds
  sorry

end paper_thickness_after_2_folds_l143_143296


namespace book_arrangement_count_l143_143827

theorem book_arrangement_count (advanced_algebra books_basic_calculus : ℕ) (total_books : ℕ) (arrangement_ways : ℕ) :
  advanced_algebra = 4 ∧ books_basic_calculus = 5 ∧ total_books = 9 ∧ arrangement_ways = Nat.choose total_books advanced_algebra →
  arrangement_ways = 126 := 
by
  sorry

end book_arrangement_count_l143_143827


namespace total_bananas_bought_l143_143779

-- Define the conditions
def went_to_store_times : ℕ := 2
def bananas_per_trip : ℕ := 10

-- State the theorem/question and provide the answer
theorem total_bananas_bought : (went_to_store_times * bananas_per_trip) = 20 :=
by
  -- Proof here
  sorry

end total_bananas_bought_l143_143779


namespace child_tickets_sold_l143_143855

-- Define variables and types
variables (A C : ℕ)

-- Main theorem to prove
theorem child_tickets_sold : A + C = 80 ∧ 12 * A + 5 * C = 519 → C = 63 :=
by
  intros
  sorry

end child_tickets_sold_l143_143855


namespace dealer_cannot_prevent_goal_l143_143028

theorem dealer_cannot_prevent_goal (m n : ℕ) :
  (m + n) % 4 = 0 :=
sorry

end dealer_cannot_prevent_goal_l143_143028


namespace olivia_nigel_remaining_money_l143_143774

theorem olivia_nigel_remaining_money :
  let olivia_money := 112
  let nigel_money := 139
  let ticket_count := 6
  let ticket_price := 28
  let total_money := olivia_money + nigel_money
  let total_cost := ticket_count * ticket_price
  total_money - total_cost = 83 := 
by 
  sorry

end olivia_nigel_remaining_money_l143_143774


namespace abs_neg_2022_eq_2022_l143_143134

theorem abs_neg_2022_eq_2022 : abs (-2022) = 2022 :=
by
  sorry

end abs_neg_2022_eq_2022_l143_143134


namespace compare_logs_l143_143716

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.logb 2 3
noncomputable def c : ℝ := Real.logb 5 8

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l143_143716


namespace cos_arcsin_eq_l143_143470

theorem cos_arcsin_eq : ∀ (x : ℝ), (x = 8 / 17) → (cos (arcsin x) = 15 / 17) := by
  intro x hx
  rw [hx]
  -- Here you can add any required steps to complete the proof.
  sorry

end cos_arcsin_eq_l143_143470


namespace complement_intersection_l143_143897

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 3}

theorem complement_intersection : (U \ N) ∩ M = {4, 5} :=
by 
  sorry

end complement_intersection_l143_143897


namespace second_divisor_13_l143_143450

theorem second_divisor_13 (N D : ℤ) (k m : ℤ) 
  (h1 : N = 39 * k + 17) 
  (h2 : N = D * m + 4) : 
  D = 13 := 
sorry

end second_divisor_13_l143_143450


namespace probability_at_least_one_woman_l143_143390

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ)
  (h1 : total_people = 10) (h2 : men = 5) (h3 : women = 5) (h4 : selected = 3) :
  (1 - (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))) = 5 / 6 :=
by
  sorry

end probability_at_least_one_woman_l143_143390


namespace evaluate_expression_l143_143482

def S (a b c : ℤ) := a + b + c

theorem evaluate_expression (a b c : ℤ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 18) :
  (144 * ((1 : ℚ) / b - (1 : ℚ) / c) + 196 * ((1 : ℚ) / c - (1 : ℚ) / a) + 324 * ((1 : ℚ) / a - (1 : ℚ) / b)) /
  (12 * ((1 : ℚ) / b - (1 : ℚ) / c) + 14 * ((1 : ℚ) / c - (1 : ℚ) / a) + 18 * ((1 : ℚ) / a - (1 : ℚ) / b)) = 44 := 
sorry

end evaluate_expression_l143_143482


namespace int_solution_count_l143_143873

def g (n : ℤ) : ℤ :=
  ⌈97 * n / 98⌉ - ⌊98 * n / 99⌋

theorem int_solution_count :
  (∃! n : ℤ, 1 + ⌊98 * n / 99⌋ = ⌈97 * n / 98⌉) :=
sorry

end int_solution_count_l143_143873


namespace tea_bags_l143_143940

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l143_143940


namespace composite_probability_l143_143981

/--
Given that a number selected at random from the first 50 natural numbers,
where 1 is neither prime nor composite,
the probability of selecting a composite number is 34/49.
-/
theorem composite_probability :
  let total_numbers := 50
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let num_primes := primes.length
  let num_composites := total_numbers - num_primes - 1
  let probability_composite := (num_composites : ℚ) / (total_numbers - 1)
  probability_composite = 34 / 49 :=
by {
  sorry
}

end composite_probability_l143_143981


namespace n_in_S_implies_n2_in_S_l143_143409

def S (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  a ≥ b ∧ c ≥ d ∧ e ≥ f ∧
  n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2

theorem n_in_S_implies_n2_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end n_in_S_implies_n2_in_S_l143_143409


namespace number_of_solutions_l143_143361

theorem number_of_solutions :
  ∃ (x y z : ℝ), 
    (x = 4036 - 4037 * Real.sign (y - z)) ∧ 
    (y = 4036 - 4037 * Real.sign (z - x)) ∧ 
    (z = 4036 - 4037 * Real.sign (x - y)) :=
sorry

end number_of_solutions_l143_143361


namespace total_hamburgers_for_lunch_l143_143317

theorem total_hamburgers_for_lunch 
  (initial_hamburgers: ℕ) 
  (additional_hamburgers: ℕ)
  (h1: initial_hamburgers = 9)
  (h2: additional_hamburgers = 3)
  : initial_hamburgers + additional_hamburgers = 12 := 
by
  sorry

end total_hamburgers_for_lunch_l143_143317


namespace total_books_for_girls_l143_143433

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end total_books_for_girls_l143_143433


namespace variance_eta_l143_143067

noncomputable def xi : ℝ := sorry -- Define ξ as a real number (will be specified later)
noncomputable def eta : ℝ := sorry -- Define η as a real number (will be specified later)

-- Conditions
axiom xi_distribution : xi = 3 + 2*Real.sqrt 4 -- ξ follows a normal distribution with mean 3 and variance 4
axiom relationship : xi = 2*eta + 3 -- Given relationship between ξ and η

-- Theorem to prove the question
theorem variance_eta : sorry := sorry

end variance_eta_l143_143067


namespace largest_integer_less_than_hundred_with_remainder_five_l143_143669

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l143_143669


namespace slope_of_line_determined_by_solutions_l143_143817

theorem slope_of_line_determined_by_solutions (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : 3 / x₁ +  4 / y₁ = 0)
  (h₂ : 3 / x₂ + 4 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4 / 3 :=
sorry

end slope_of_line_determined_by_solutions_l143_143817


namespace total_net_loss_l143_143163

theorem total_net_loss 
  (P_x P_y : ℝ)
  (h1 : 1.2 * P_x = 25000)
  (h2 : 0.8 * P_y = 25000) :
  (25000 - P_x) - (P_y - 25000) = -2083.33 :=
by 
  sorry

end total_net_loss_l143_143163


namespace fraction_of_third_is_eighth_l143_143582

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_third_is_eighth_l143_143582


namespace smallest_interesting_number_l143_143848

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l143_143848


namespace probability_white_given_popped_l143_143017

-- Define conditional probabilities and the probability calculations
open Rat

theorem probability_white_given_popped
  (P_white : ℚ := 3/4)
  (P_yellow : ℚ := 1/4)
  (P_damaged : ℚ := 1/4)
  (P_ND_white : ℚ := 3/4 * 3/4)
  (P_ND_yellow : ℚ := 1/4 * 3/4)
  (P_popped_given_ND_white : ℚ := 3/5)
  (P_popped_given_ND_yellow : ℚ := 4/5) :
  (P_ND_white * P_popped_given_ND_white) / ((P_ND_white * P_popped_given_ND_white) + (P_ND_yellow * P_popped_given_ND_yellow)) = 9/13 :=
by
  sorry

end probability_white_given_popped_l143_143017


namespace quadratic_inequality_cond_l143_143412

theorem quadratic_inequality_cond (a : ℝ) :
  (∀ x : ℝ, ax^2 - ax + 1 > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end quadratic_inequality_cond_l143_143412


namespace find_number_with_10_questions_l143_143730

theorem find_number_with_10_questions (n : ℕ) (h : n ≤ 1000) : n = 300 :=
by
  sorry

end find_number_with_10_questions_l143_143730


namespace greatest_unexpressible_sum_l143_143650

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem greatest_unexpressible_sum : 
  ∀ (n : ℕ), (∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n) → n ≤ 11 :=
by
  sorry

end greatest_unexpressible_sum_l143_143650


namespace remainder_when_divided_by_19_l143_143596

theorem remainder_when_divided_by_19 {N : ℤ} (h : N % 342 = 47) : N % 19 = 9 :=
sorry

end remainder_when_divided_by_19_l143_143596


namespace min_value_expr_l143_143475

theorem min_value_expr (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, m > 0 ∧ (forall (n : ℕ), 0 < n → (n/2 + 50/n : ℝ) ≥ 10) ∧ 
           (n = 10) → (n/2 + 50/n : ℝ) = 10 :=
by
  sorry

end min_value_expr_l143_143475


namespace length_of_longest_side_l143_143245

theorem length_of_longest_side (l w : ℝ) (h_fencing : 2 * l + 2 * w = 240) (h_area : l * w = 8 * 240) : max l w = 96 :=
by sorry

end length_of_longest_side_l143_143245


namespace profit_distribution_l143_143035

noncomputable def profit_sharing (investment_a investment_d profit: ℝ) : ℝ × ℝ :=
  let total_investment := investment_a + investment_d
  let share_a := investment_a / total_investment
  let share_d := investment_d / total_investment
  (share_a * profit, share_d * profit)

theorem profit_distribution :
  let investment_a := 22500
  let investment_d := 35000
  let first_period_profit := 9600
  let second_period_profit := 12800
  let third_period_profit := 18000
  profit_sharing investment_a investment_d first_period_profit = (3600, 6000) ∧
  profit_sharing investment_a investment_d second_period_profit = (5040, 7760) ∧
  profit_sharing investment_a investment_d third_period_profit = (7040, 10960) :=
sorry

end profit_distribution_l143_143035


namespace remainder_4063_div_97_l143_143587

theorem remainder_4063_div_97 : 4063 % 97 = 86 := 
by sorry

end remainder_4063_div_97_l143_143587


namespace sum_of_six_consecutive_odd_numbers_l143_143908

theorem sum_of_six_consecutive_odd_numbers (a b c d e f : ℕ) 
  (ha : 135135 = a * b * c * d * e * f)
  (hb : a < b) (hc : b < c) (hd : c < d) (he : d < e) (hf : e < f)
  (hzero : a % 2 = 1) (hone : b % 2 = 1) (htwo : c % 2 = 1) 
  (hthree : d % 2 = 1) (hfour : e % 2 = 1) (hfive : f % 2 = 1) :
  a + b + c + d + e + f = 48 := by
  sorry

end sum_of_six_consecutive_odd_numbers_l143_143908


namespace smallest_interesting_number_is_1800_l143_143839

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l143_143839


namespace positive_value_of_A_l143_143520

def my_relation (A B k : ℝ) : ℝ := A^2 + k * B^2

theorem positive_value_of_A (A : ℝ) (h1 : ∀ A B, my_relation A B 3 = A^2 + 3 * B^2) (h2 : my_relation A 7 3 = 196) :
  A = 7 := by
  sorry

end positive_value_of_A_l143_143520


namespace polynomial_remainder_l143_143002

-- Define the polynomial
def poly (x : ℝ) : ℝ := 3 * x^8 - x^7 - 7 * x^5 + 3 * x^3 + 4 * x^2 - 12 * x - 1

-- Define the divisor
def divisor : ℝ := 3

-- State the theorem
theorem polynomial_remainder :
  poly divisor = 15951 :=
by
  -- Proof omitted, to be filled in later
  sorry

end polynomial_remainder_l143_143002


namespace exists_odd_digit_div_by_five_power_l143_143013

theorem exists_odd_digit_div_by_five_power (n : ℕ) (h : 0 < n) : ∃ (k : ℕ), 
  (∃ (m : ℕ), k = m * 5^n) ∧ 
  (∀ (d : ℕ), (d = (k / (10^(n-1))) % 10) → d % 2 = 1) :=
sorry

end exists_odd_digit_div_by_five_power_l143_143013


namespace fractional_expression_simplification_l143_143033

theorem fractional_expression_simplification (x : ℕ) (h : x - 3 < 0) : 
  (2 * x) / (x^2 - 1) - 1 / (x - 1) = 1 / 3 :=
by {
  -- Typical proof steps would go here, adhering to the natural conditions.
  sorry
}

end fractional_expression_simplification_l143_143033


namespace sequence_is_arithmetic_l143_143760

theorem sequence_is_arithmetic {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
  (h_second_term : a 2 = 3 * a 1)
  (h_sqrt_seq_arith : ∃ d : ℝ, ∀ n, real.sqrt (∑ i in finset.range (n + 1), a i) = d * n + real.sqrt (a 0)): 
  ∃ d : ℝ, ∀ n, a n = a 0 + d * n := 
by
  sorry

end sequence_is_arithmetic_l143_143760


namespace faucet_fill_time_l143_143709

theorem faucet_fill_time (r : ℝ) (T1 T2 t : ℝ) (F1 F2 : ℕ) (h1 : T1 = 200) (h2 : t = 8) (h3 : F1 = 4) (h4 : F2 = 8) (h5 : T2 = 50) (h6 : r * F1 * t = T1) : 
(F2 * r) * t / (F1 * F2) = T2 -> by sorry := sorry

#check faucet_fill_time

end faucet_fill_time_l143_143709


namespace remaining_perimeter_of_square_with_cutouts_l143_143455

theorem remaining_perimeter_of_square_with_cutouts 
  (square_side : ℝ) (green_square_side : ℝ) (init_perimeter : ℝ) 
  (green_square_perimeter_increase : ℝ) (final_perimeter : ℝ) :
  square_side = 10 → green_square_side = 2 →
  init_perimeter = 4 * square_side → green_square_perimeter_increase = 4 * green_square_side →
  final_perimeter = init_perimeter + green_square_perimeter_increase →
  final_perimeter = 44 :=
by
  intros hsquare_side hgreen_square_side hinit_perimeter hgreen_incr hfinal_perimeter
  -- Proof steps can be added here
  sorry

end remaining_perimeter_of_square_with_cutouts_l143_143455


namespace cube_faces_edges_vertices_sum_l143_143153

theorem cube_faces_edges_vertices_sum :
  ∀ (F E V : ℕ), F = 6 → E = 12 → V = 8 → F + E + V = 26 :=
by
  intros F E V F_eq E_eq V_eq
  rw [F_eq, E_eq, V_eq]
  rfl

end cube_faces_edges_vertices_sum_l143_143153


namespace simplify_fraction_l143_143444

theorem simplify_fraction (d : ℝ) : (6 - 5 * d) / 9 - 3 = (-21 - 5 * d) / 9 :=
by
  sorry

end simplify_fraction_l143_143444


namespace batsman_average_after_12th_innings_l143_143448

-- Defining the conditions
def before_12th_innings_average (A : ℕ) : Prop :=
11 * A + 80 = 12 * (A + 2)

-- Defining the question and expected answer
def after_12th_innings_average : ℕ := 58

-- Proving the equivalence
theorem batsman_average_after_12th_innings (A : ℕ) (h : before_12th_innings_average A) : after_12th_innings_average = 58 :=
by
sorry

end batsman_average_after_12th_innings_l143_143448


namespace isosceles_triangle_vertex_angle_l143_143367

theorem isosceles_triangle_vertex_angle (B : ℝ) (V : ℝ) (h1 : B = 70) (h2 : B = B) (h3 : V + 2 * B = 180) : V = 40 ∨ V = 70 :=
by {
  sorry
}

end isosceles_triangle_vertex_angle_l143_143367


namespace geom_seq_common_ratio_l143_143099

theorem geom_seq_common_ratio {a : ℕ → ℝ} (q : ℝ) 
  (h1 : a 3 = 6) 
  (h2 : a 1 + a 2 + a 3 = 18) 
  (h_geom_seq : ∀ n, a (n + 1) = a n * q) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end geom_seq_common_ratio_l143_143099


namespace problem_l143_143369

theorem problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end problem_l143_143369


namespace binom_eq_fraction_l143_143812

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l143_143812


namespace solve_floor_fractional_l143_143494

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - floor x

theorem solve_floor_fractional (x : ℝ) :
  floor x * fractional_part x = 2019 * x ↔ x = 0 ∨ x = -1 / 2020 :=
by
  sorry

end solve_floor_fractional_l143_143494


namespace prove_arithmetic_sequence_l143_143277

def arithmetic_sequence (x : ℝ) : ℕ → ℝ
| 0 => x - 1
| 1 => x + 1
| 2 => 2 * x + 3
| n => sorry

theorem prove_arithmetic_sequence {x : ℝ} (a : ℕ → ℝ)
  (h_terms : a 0 = x - 1 ∧ a 1 = x + 1 ∧ a 2 = 2 * x + 3)
  (h_arithmetic : ∀ n, a n = a 0 + n * (a 1 - a 0)) :
  x = 0 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end prove_arithmetic_sequence_l143_143277


namespace cube_faces_edges_vertices_sum_l143_143154

theorem cube_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  sorry

end cube_faces_edges_vertices_sum_l143_143154


namespace nesbitt_inequality_nesbitt_inequality_eq_l143_143765

variable {a b c : ℝ}

theorem nesbitt_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

theorem nesbitt_inequality_eq (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ((a / (b + c)) + (b / (a + c)) + (c / (a + b)) = (3 / 2)) ↔ (a = b ∧ b = c) :=
sorry

end nesbitt_inequality_nesbitt_inequality_eq_l143_143765


namespace smallest_interesting_number_is_1800_l143_143837

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l143_143837


namespace sarah_trucks_l143_143953

-- Define the initial number of trucks denoted by T
def initial_trucks (T : ℝ) : Prop :=
  let left_after_jeff := T - 13.5
  let left_after_ashley := left_after_jeff - 0.25 * left_after_jeff
  left_after_ashley = 38

-- Theorem stating the initial number of trucks Sarah had is 64
theorem sarah_trucks : ∃ T : ℝ, initial_trucks T ∧ T = 64 :=
by
  sorry

end sarah_trucks_l143_143953


namespace cube_faces_edges_vertices_sum_l143_143155

theorem cube_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  sorry

end cube_faces_edges_vertices_sum_l143_143155


namespace sum_of_subsets_l143_143521

theorem sum_of_subsets (a1 a2 a3 : ℝ) (h : (a1 + a2 + a3) + (a1 + a2 + a1 + a3 + a2 + a3) = 12) : 
  a1 + a2 + a3 = 4 := 
by 
  sorry

end sum_of_subsets_l143_143521


namespace height_comparison_of_cylinder_and_rectangular_solid_l143_143861

theorem height_comparison_of_cylinder_and_rectangular_solid
  (V : ℝ) (A : ℝ) (h_cylinder : ℝ) (h_rectangular_solid : ℝ)
  (equal_volume : V = V)
  (equal_base_areas : A = A)
  (height_cylinder_eq : h_cylinder = V / A)
  (height_rectangular_solid_eq : h_rectangular_solid = V / A)
  : ¬ (h_cylinder > h_rectangular_solid) :=
by {
  sorry
}

end height_comparison_of_cylinder_and_rectangular_solid_l143_143861


namespace tea_bags_number_l143_143950

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l143_143950


namespace original_amount_l143_143176

variable (M : ℕ)

def initialAmountAfterFirstLoss := M - M / 3
def amountAfterFirstWin := initialAmountAfterFirstLoss M + 10
def amountAfterSecondLoss := amountAfterFirstWin M - (amountAfterFirstWin M) / 3
def amountAfterSecondWin := amountAfterSecondLoss M + 20
def finalAmount := amountAfterSecondWin M - (amountAfterSecondWin M) / 4

theorem original_amount : finalAmount M = M → M = 30 :=
by
  sorry

end original_amount_l143_143176


namespace time_difference_in_minutes_l143_143169

def speed := 60 -- speed of the car in miles per hour
def distance1 := 360 -- distance of the first trip in miles
def distance2 := 420 -- distance of the second trip in miles
def hours_to_minutes := 60 -- conversion factor from hours to minutes

theorem time_difference_in_minutes :
  ((distance2 / speed) - (distance1 / speed)) * hours_to_minutes = 60 :=
by
  -- proof to be provided
  sorry

end time_difference_in_minutes_l143_143169


namespace diagonal_pairs_forming_60_degrees_l143_143059

theorem diagonal_pairs_forming_60_degrees :
  let total_diagonals := 12
  let total_pairs := total_diagonals.choose 2
  let non_forming_pairs_per_set := 6
  let sets_of_parallel_planes := 3
  total_pairs - non_forming_pairs_per_set * sets_of_parallel_planes = 48 :=
by 
  let total_diagonals := 12
  let total_pairs := total_diagonals.choose 2
  let non_forming_pairs_per_set := 6
  let sets_of_parallel_planes := 3
  have calculation : total_pairs - non_forming_pairs_per_set * sets_of_parallel_planes = 48 := sorry
  exact calculation

end diagonal_pairs_forming_60_degrees_l143_143059


namespace domain_of_h_l143_143956

-- Definition of the function domain of f(x) and h(x)
def f_domain := Set.Icc (-10: ℝ) 6
def h_domain := Set.Icc (-2: ℝ) (10/3)

-- Definition of f and h
def f (x: ℝ) : ℝ := sorry  -- f is assumed to be defined on the interval [-10, 6]
def h (x: ℝ) : ℝ := f (-3 * x)

-- Theorem statement: Given the domain of f(x), the domain of h(x) is as follows
theorem domain_of_h :
  (∀ x, x ∈ f_domain ↔ (-3 * x) ∈ h_domain) :=
sorry

end domain_of_h_l143_143956


namespace largest_integer_less_than_100_with_remainder_5_l143_143654

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143654


namespace height_of_original_triangle_l143_143805

variable (a b c : ℝ)

theorem height_of_original_triangle (a b c : ℝ) : 
  ∃ h : ℝ, h = a + b + c :=
  sorry

end height_of_original_triangle_l143_143805


namespace count_integers_between_sqrt10_sqrt100_l143_143725

theorem count_integers_between_sqrt10_sqrt100 :
  ∃ n : ℕ, n = 7 ∧ card {x : ℤ | real.sqrt 10 < x ∧ x < real.sqrt 100} = n :=
by
  sorry

end count_integers_between_sqrt10_sqrt100_l143_143725


namespace actual_length_of_road_l143_143775

-- Define the conditions
def scale_factor : ℝ := 2500000
def length_on_map : ℝ := 6
def cm_to_km : ℝ := 100000

-- State the theorem
theorem actual_length_of_road : (length_on_map * scale_factor) / cm_to_km = 150 := by
  sorry

end actual_length_of_road_l143_143775


namespace maximum_withdraw_l143_143124

theorem maximum_withdraw (initial_amount withdraw deposit : ℕ) (h_initial : initial_amount = 500)
    (h_withdraw : withdraw = 300) (h_deposit : deposit = 198) :
    ∃ x y : ℕ, initial_amount - x * withdraw + y * deposit ≥ 0 ∧ initial_amount - x * withdraw + y * deposit = 194 ∧ initial_amount - x * withdraw = 300 := sorry

end maximum_withdraw_l143_143124


namespace decimal_to_binary_correct_l143_143641

-- Define the decimal number
def decimal_number : ℕ := 25

-- Define the binary equivalent of 25
def binary_representation : ℕ := 0b11001

-- The condition indicating how the conversion is done
def is_binary_representation (decimal : ℕ) (binary : ℕ) : Prop :=
  -- Check if the binary representation matches the manual decomposition
  decimal = (binary / 2^4) * 2^4 + 
            ((binary % 2^4) / 2^3) * 2^3 + 
            (((binary % 2^4) % 2^3) / 2^2) * 2^2 + 
            ((((binary % 2^4) % 2^3) % 2^2) / 2^1) * 2^1 + 
            (((((binary % 2^4) % 2^3) % 2^2) % 2^1) / 2^0) * 2^0

-- Proof statement
theorem decimal_to_binary_correct : is_binary_representation decimal_number binary_representation :=
  by sorry

end decimal_to_binary_correct_l143_143641


namespace minimum_value_of_z_l143_143286

theorem minimum_value_of_z : ∃ (x : ℝ), ∀ (z : ℝ), (z = 4 * x^2 + 8 * x + 16) → z ≥ 12 :=
by
  sorry

end minimum_value_of_z_l143_143286


namespace tangent_line_equation_l143_143646

theorem tangent_line_equation :
  (∃ l : ℝ → ℝ, 
   (∀ x, l x = (1 / (4 + 2 * Real.sqrt 3)) * x + (2 + Real.sqrt 3) / 2 ∨ 
         l x = (1 / (4 - 2 * Real.sqrt 3)) * x + (2 - Real.sqrt 3) / 2) ∧ 
   (l 1 = 2) ∧ 
   (∀ x, l x = Real.sqrt x)
  ) →
  (∀ x y, 
   (y = (1 / 4 + Real.sqrt 3) * x + (2 + Real.sqrt 3) / 2 ∨ 
    y = (1 / 4 - Real.sqrt 3) * x + (2 - Real.sqrt 3) / 2) ∨ 
   (x - (4 + 2 * Real.sqrt 3) * y + (7 + 4 * Real.sqrt 3) = 0 ∨ 
    x - (4 - 2 * Real.sqrt 3) * y + (7 - 4 * Real.sqrt 3) = 0)
) :=
sorry

end tangent_line_equation_l143_143646


namespace final_price_is_correct_l143_143799

-- Define the original price and the discount rate
variable (a : ℝ)

-- The final price of the product after two 10% discounts
def final_price_after_discounts (a : ℝ) : ℝ :=
  a * (0.9 ^ 2)

-- Theorem stating the final price after two consecutive 10% discounts
theorem final_price_is_correct (a : ℝ) :
  final_price_after_discounts a = a * (0.9 ^ 2) :=
by sorry

end final_price_is_correct_l143_143799


namespace pascal_sum_difference_l143_143131

open BigOperators

noncomputable def a_i (i : ℕ) := Nat.choose 3005 i
noncomputable def b_i (i : ℕ) := Nat.choose 3006 i
noncomputable def c_i (i : ℕ) := Nat.choose 3007 i

theorem pascal_sum_difference :
  (∑ i in Finset.range 3007, (b_i i) / (c_i i)) - (∑ i in Finset.range 3006, (a_i i) / (b_i i)) = 1 / 2 := by
  sorry

end pascal_sum_difference_l143_143131


namespace largest_integer_less_than_100_with_remainder_5_l143_143691

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l143_143691


namespace distance_from_diagonal_intersection_to_base_l143_143393

theorem distance_from_diagonal_intersection_to_base (AD BC AB R : ℝ) (O : ℝ → Prop) (M N Q : ℝ) :
  (AD + BC + 2 * AB = 8) ∧
  (AD + BC) = 4 ∧
  (R = 1 / 2) ∧
  (2 = R * (AD + BC) / 2) ∧
  (BC = AD + 2 * AB) ∧
  (∀ x, x * (2 - x) = (1 / 2) ^ 2)  →
  (Q = (2 - Real.sqrt 3) / 4) :=
by
  intros
  sorry

end distance_from_diagonal_intersection_to_base_l143_143393


namespace number_of_girls_l143_143617

theorem number_of_girls (d c : ℕ) (h1 : c = 2 * (d - 15)) (h2 : d - 15 = 5 * (c - 45)) : d = 40 := 
by
  sorry

end number_of_girls_l143_143617


namespace order_of_values_l143_143903

noncomputable def a : ℝ := Real.log 2 / 2
noncomputable def b : ℝ := Real.log 3 / 3
noncomputable def c : ℝ := Real.log Real.pi / Real.pi
noncomputable def d : ℝ := Real.log 2.72 / 2.72
noncomputable def f : ℝ := (Real.sqrt 10 * Real.log 10) / 20

theorem order_of_values : a < f ∧ f < c ∧ c < b ∧ b < d :=
by
  sorry

end order_of_values_l143_143903


namespace c_share_is_160_l143_143602

theorem c_share_is_160 (a b c : ℕ) (total : ℕ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 10 * c) (h_total : a + b + c = 880) : c = 160 :=
by
  sorry

end c_share_is_160_l143_143602


namespace find_cos_2beta_l143_143717

noncomputable def cos_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (htan : Real.tan α = 1 / 7) (hcos : Real.cos (α + β) = 2 * Real.sqrt 5 / 5) : Real :=
  2 * (Real.cos β)^2 - 1

theorem find_cos_2beta (α β : ℝ) (h1: 0 < α ∧ α < π / 2) (h2: 0 < β ∧ β < π / 2)
  (htan: Real.tan α = 1 / 7) (hcos: Real.cos (α + β) = 2 * Real.sqrt 5 / 5) :
  cos_2beta α β h1 h2 htan hcos = 4 / 5 := 
sorry

end find_cos_2beta_l143_143717


namespace tea_bags_count_l143_143942

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l143_143942


namespace ratio_of_volumes_l143_143293
-- Import the necessary library

-- Define the edge lengths of the cubes
def small_cube_edge : ℕ := 4
def large_cube_edge : ℕ := 24

-- Define the volumes of the cubes
def volume (edge : ℕ) : ℕ := edge ^ 3

-- Define the volumes
def V_small := volume small_cube_edge
def V_large := volume large_cube_edge

-- State the main theorem
theorem ratio_of_volumes : V_small / V_large = 1 / 216 := 
by 
  -- we skip the proof here
  sorry

end ratio_of_volumes_l143_143293


namespace xiaozhi_needs_median_for_top_10_qualification_l143_143746

-- Define a set of scores as a list of integers
def scores : List ℕ := sorry

-- Assume these scores are unique (this is a condition given in the problem)
axiom unique_scores : ∀ (a b : ℕ), a ∈ scores → b ∈ scores → a ≠ b → scores.indexOf a ≠ scores.indexOf b

-- Define the median function (in practice, you would implement this, but we're just outlining it here)
def median (scores: List ℕ) : ℕ := sorry

-- Define the position of Xiao Zhi's score
def xiaozhi_score : ℕ := sorry

-- Given that the top 10 scores are needed to advance
def top_10 (scores: List ℕ) : List ℕ := scores.take 10

-- Proposition that Xiao Zhi needs median to determine his rank in top 10
theorem xiaozhi_needs_median_for_top_10_qualification 
    (scores_median : ℕ) (zs_score : ℕ) : 
    (∀ (s: List ℕ), s = scores → scores_median = median s → zs_score ≤ scores_median → zs_score ∉ top_10 s) ∧ 
    (exists (s: List ℕ), s = scores → zs_score ∉ top_10 s → zs_score ≤ scores_median) := 
sorry

end xiaozhi_needs_median_for_top_10_qualification_l143_143746


namespace least_number_of_pairs_l143_143634

theorem least_number_of_pairs :
  let students := 100
  let messages_per_student := 50
  ∃ (pairs_of_students : ℕ), pairs_of_students = 50 := sorry

end least_number_of_pairs_l143_143634


namespace problem_solution_l143_143074

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log (2 - x) / Real.log 2 else 2 ^ (x - 1)

theorem problem_solution : f (-2) + f (Real.log 12 / Real.log 2) = 9 := by
  sorry

end problem_solution_l143_143074


namespace range_of_r_l143_143895

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

theorem range_of_r (r : ℝ) (hr: 0 < r) : (M ∩ N r = N r) → r ≤ 2 - Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_r_l143_143895


namespace polynomial_R_result_l143_143250

noncomputable def polynomial_Q_R (z : ℤ) : Prop :=
  ∃ Q R : Polynomial ℂ, 
  z ^ 2020 + 1 = (z ^ 2 - z + 1) * Q + R ∧ R.degree < 2 ∧ R = 2

theorem polynomial_R_result :
  polynomial_Q_R z :=
by 
  sorry

end polynomial_R_result_l143_143250


namespace eccentricity_hyperbola_l143_143718

-- Conditions
def is_eccentricity_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  let e := (Real.sqrt 2) / 2
  (Real.sqrt (1 - b^2 / a^2) = e)

-- Objective: Find the eccentricity of the given the hyperbola.
theorem eccentricity_hyperbola (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : is_eccentricity_ellipse a b h1 h2) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
sorry

end eccentricity_hyperbola_l143_143718


namespace bob_total_spend_in_usd_l143_143038

theorem bob_total_spend_in_usd:
  let coffee_cost_yen := 250
  let sandwich_cost_yen := 150
  let yen_to_usd := 110
  (coffee_cost_yen + sandwich_cost_yen) / yen_to_usd = 3.64 := by
  sorry

end bob_total_spend_in_usd_l143_143038


namespace gcd_lcm_sum_l143_143465

theorem gcd_lcm_sum (a b : ℕ) (ha : a = 45) (hb : b = 4050) :
  Nat.gcd a b + Nat.lcm a b = 4095 := by
  sorry

end gcd_lcm_sum_l143_143465


namespace age_difference_l143_143603

variable (Patrick_age Michael_age Monica_age : ℕ)

theorem age_difference 
  (h1 : ∃ x : ℕ, Patrick_age = 3 * x ∧ Michael_age = 5 * x)
  (h2 : ∃ y : ℕ, Michael_age = 3 * y ∧ Monica_age = 5 * y)
  (h3 : Patrick_age + Michael_age + Monica_age = 245) :
  Monica_age - Patrick_age = 80 := by 
sorry

end age_difference_l143_143603


namespace min_value_of_a2_b2_l143_143215

theorem min_value_of_a2_b2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : 
  ∃ m : ℝ, (∀ x y, x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ m) ∧ m = 8 :=
by
  sorry

end min_value_of_a2_b2_l143_143215


namespace triangle_side_length_BC_49_l143_143915

theorem triangle_side_length_BC_49
  (angle_A : ℝ)
  (AC : ℝ)
  (area_ABC : ℝ)
  (h1 : angle_A = 60)
  (h2 : AC = 16)
  (h3 : area_ABC = 220 * Real.sqrt 3) : 
  ∃ (BC : ℝ), BC = 49 :=
by
  sorry

end triangle_side_length_BC_49_l143_143915


namespace equilateral_triangle_sum_l143_143623

theorem equilateral_triangle_sum (x y : ℕ) (h1 : x + 5 = 14) (h2 : y + 11 = 14) : x + y = 12 :=
by
  sorry

end equilateral_triangle_sum_l143_143623


namespace TV_cost_difference_l143_143295

def cost_per_square_inch_difference :=
  let first_TV_width := 24
  let first_TV_height := 16
  let first_TV_original_cost_euros := 840
  let first_TV_discount_percent := 0.10
  let first_TV_tax_percent := 0.05
  let exchange_rate_first := 1.20
  let first_TV_area := first_TV_width * first_TV_height

  let discounted_price_first_TV := first_TV_original_cost_euros * (1 - first_TV_discount_percent)
  let total_cost_euros_first_TV := discounted_price_first_TV * (1 + first_TV_tax_percent)
  let total_cost_dollars_first_TV := total_cost_euros_first_TV * exchange_rate_first
  let cost_per_square_inch_first_TV := total_cost_dollars_first_TV / first_TV_area

  let new_TV_width := 48
  let new_TV_height := 32
  let new_TV_original_cost_dollars := 1800
  let new_TV_first_discount_percent := 0.20
  let new_TV_second_discount_percent := 0.15
  let new_TV_tax_percent := 0.08
  let new_TV_area := new_TV_width * new_TV_height

  let price_after_first_discount := new_TV_original_cost_dollars * (1 - new_TV_first_discount_percent)
  let price_after_second_discount := price_after_first_discount * (1 - new_TV_second_discount_percent)
  let total_cost_dollars_new_TV := price_after_second_discount * (1 + new_TV_tax_percent)
  let cost_per_square_inch_new_TV := total_cost_dollars_new_TV / new_TV_area

  let cost_difference_per_square_inch := cost_per_square_inch_first_TV - cost_per_square_inch_new_TV
  cost_difference_per_square_inch

theorem TV_cost_difference :
  cost_per_square_inch_difference = 1.62 := by
  sorry

end TV_cost_difference_l143_143295


namespace nicolai_peaches_pounds_l143_143147

-- Condition definitions
def total_fruit_pounds : ℕ := 8
def mario_oranges_ounces : ℕ := 8
def lydia_apples_ounces : ℕ := 24
def ounces_to_pounds (ounces: ℕ) : ℚ := ounces / 16 

-- Statement we want to prove
theorem nicolai_peaches_pounds :
  let mario_oranges_pounds := ounces_to_pounds mario_oranges_ounces,
      lydia_apples_pounds := ounces_to_pounds lydia_apples_ounces,
      eaten_by_m_and_l := mario_oranges_pounds + lydia_apples_pounds,
      nicolai_peaches_pounds := total_fruit_pounds - eaten_by_m_and_l in
  nicolai_peaches_pounds = 6 :=
by
  sorry

end nicolai_peaches_pounds_l143_143147


namespace discount_on_purchase_l143_143907

theorem discount_on_purchase 
  (price_cherries price_olives : ℕ)
  (num_bags_cherries num_bags_olives : ℕ)
  (total_paid : ℕ) :
  price_cherries = 5 →
  price_olives = 7 →
  num_bags_cherries = 50 →
  num_bags_olives = 50 →
  total_paid = 540 →
  (total_paid / (price_cherries * num_bags_cherries + price_olives * num_bags_olives)) * 100 = 90 :=
by
  intros h_price_cherries h_price_olives h_num_bags_cherries h_num_bags_olives h_total_paid
  sorry

end discount_on_purchase_l143_143907


namespace square_circle_radius_l143_143000

theorem square_circle_radius (a R : ℝ) (h1 : a^2 = 256) (h2 : R = 10) : R = 10 :=
sorry

end square_circle_radius_l143_143000


namespace bugs_eaten_ratio_l143_143241

theorem bugs_eaten_ratio :
  ∃ (L : ℚ), 
    12 + L + 3 * L + 4.5 * L = 63 ∧ (L / 12 = 1 / 2) :=
by {
  sorry
}

end bugs_eaten_ratio_l143_143241


namespace ratio_of_x_to_y_l143_143731

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 3 * x = 0.12 * 250 * y) : x / y = 10 :=
sorry

end ratio_of_x_to_y_l143_143731


namespace sara_has_green_marbles_l143_143541

-- Definition of the total number of green marbles and Tom's green marbles
def total_green_marbles : ℕ := 7
def tom_green_marbles : ℕ := 4

-- Definition of Sara's green marbles
def sara_green_marbles : ℕ := total_green_marbles - tom_green_marbles

-- The proof statement
theorem sara_has_green_marbles : sara_green_marbles = 3 :=
by
  -- The proof will be filled in here
  sorry

end sara_has_green_marbles_l143_143541


namespace average_six_consecutive_integers_starting_with_d_l143_143784

theorem average_six_consecutive_integers_starting_with_d (c : ℝ) (d : ℝ)
  (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5)) / 6 = c + 5 :=
by
  sorry -- Proof to be completed

end average_six_consecutive_integers_starting_with_d_l143_143784


namespace complement_set_A_in_U_l143_143923

-- Given conditions
def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {x | x ∈ U ∧ x^2 < 1}

-- Theorem to prove complement
theorem complement_set_A_in_U :
  U \ A = {-1, 1, 2} :=
by
  sorry

end complement_set_A_in_U_l143_143923


namespace quadratic_real_roots_iff_range_of_a_l143_143997

theorem quadratic_real_roots_iff_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a = 0) ↔ a ≤ 1 :=
by
  sorry

end quadratic_real_roots_iff_range_of_a_l143_143997


namespace excess_percentage_l143_143394

theorem excess_percentage (A B : ℝ) (x : ℝ) 
  (hA' : A' = A * (1 + x / 100))
  (hB' : B' = B * (1 - 5 / 100))
  (h_area_err : A' * B' = 1.007 * (A * B)) : x = 6 :=
by
  sorry

end excess_percentage_l143_143394


namespace Gage_skating_time_l143_143875

theorem Gage_skating_time :
  let min_per_hr := 60
  let skating_6_days := 6 * (1 * min_per_hr + 20)
  let skating_4_days := 4 * (1 * min_per_hr + 35)
  let needed_total := 11 * 90
  let skating_10_days := skating_6_days + skating_4_days
  let minutes_on_eleventh_day := needed_total - skating_10_days
  minutes_on_eleventh_day = 130 :=
by
  sorry

end Gage_skating_time_l143_143875


namespace cost_price_of_article_l143_143034

theorem cost_price_of_article (C MP SP : ℝ) (h1 : MP = 62.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) :
  C = 47.5 :=
sorry

end cost_price_of_article_l143_143034


namespace closest_multiple_of_18_l143_143158

def is_multiple_of_2 (n : ℤ) : Prop := n % 2 = 0
def is_multiple_of_9 (n : ℤ) : Prop := n % 9 = 0
def is_multiple_of_18 (n : ℤ) : Prop := is_multiple_of_2 n ∧ is_multiple_of_9 n

theorem closest_multiple_of_18 (n : ℤ) (h : n = 2509) : 
  ∃ k : ℤ, is_multiple_of_18 k ∧ (abs (2509 - k) = 7) :=
sorry

end closest_multiple_of_18_l143_143158


namespace negB_sufficient_for_A_l143_143248

variables {A B : Prop}

theorem negB_sufficient_for_A (h : ¬A → B) (hnotsuff : ¬(B → ¬A)) : ¬ B → A :=
by
  sorry

end negB_sufficient_for_A_l143_143248


namespace candies_problem_l143_143754

theorem candies_problem (x : ℕ) (Nina : ℕ) (Oliver : ℕ) (total_candies : ℕ) (h1 : 4 * x = Mark) (h2 : 2 * Mark = Nina) (h3 : 6 * Nina = Oliver) (h4 : x + Mark + Nina + Oliver = total_candies) :
  x = 360 / 61 :=
by
  sorry

end candies_problem_l143_143754


namespace science_book_pages_l143_143966

theorem science_book_pages {history_pages novel_pages science_pages: ℕ} (h1: novel_pages = history_pages / 2) (h2: science_pages = 4 * novel_pages) (h3: history_pages = 300):
  science_pages = 600 :=
by
  sorry

end science_book_pages_l143_143966


namespace decrypted_plaintext_l143_143806

theorem decrypted_plaintext (a b c d : ℕ) : 
  (a + 2 * b = 14) → (2 * b + c = 9) → (2 * c + 3 * d = 23) → (4 * d = 28) → 
  (a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7) :=
by 
  intros h1 h2 h3 h4
  -- Proof steps go here
  sorry

end decrypted_plaintext_l143_143806


namespace tea_bags_number_l143_143949

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l143_143949


namespace int_solution_l143_143645

theorem int_solution (n : ℕ) (h1 : n ≥ 1) (h2 : n^2 ∣ 2^n + 1) : n = 1 ∨ n = 3 :=
by
  sorry

end int_solution_l143_143645


namespace Winnie_the_Pooh_stationary_escalator_steps_l143_143820

theorem Winnie_the_Pooh_stationary_escalator_steps
  (u v L : ℝ)
  (cond1 : L * u / (u + v) = 55)
  (cond2 : L * u / (u - v) = 1155) :
  L = 105 := by
  sorry

end Winnie_the_Pooh_stationary_escalator_steps_l143_143820


namespace smallest_palindrome_in_bases_2_and_4_l143_143190

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let repr := n.digits base
  repr = repr.reverse

theorem smallest_palindrome_in_bases_2_and_4 (x : ℕ) :
  (x > 15) ∧ is_palindrome x 2 ∧ is_palindrome x 4 → x = 17 :=
by
  sorry

end smallest_palindrome_in_bases_2_and_4_l143_143190


namespace train_speed_l143_143298

theorem train_speed (distance time : ℝ) (h1 : distance = 400) (h2 : time = 10) : 
  distance / time = 40 := 
sorry

end train_speed_l143_143298


namespace expected_score_shooting_competition_l143_143392

theorem expected_score_shooting_competition (hit_rate : ℝ)
  (miss_both_score : ℝ) (hit_one_score : ℝ) (hit_both_score : ℝ)
  (prob_0 : ℝ) (prob_10 : ℝ) (prob_15 : ℝ) :
  hit_rate = 4 / 5 →
  miss_both_score = 0 →
  hit_one_score = 10 →
  hit_both_score = 15 →
  prob_0 = (1 - 4 / 5) * (1 - 4 / 5) →
  prob_10 = 2 * (4 / 5) * (1 - 4 / 5) →
  prob_15 = (4 / 5) * (4 / 5) →
  (0 * prob_0 + 10 * prob_10 + 15 * prob_15) = 12.8 :=
by
  intros h_hit_rate h_miss_both_score h_hit_one_score h_hit_both_score
         h_prob_0 h_prob_10 h_prob_15
  sorry

end expected_score_shooting_competition_l143_143392


namespace max_value_of_sum_l143_143115

open Real

theorem max_value_of_sum (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 3) :
  (ab / (a + b) + bc / (b + c) + ca / (c + a)) ≤ 3 / 2 :=
sorry

end max_value_of_sum_l143_143115


namespace nicolai_ate_6_pounds_of_peaches_l143_143146

noncomputable def total_weight_pounds : ℝ := 8
noncomputable def pound_to_ounce : ℝ := 16
noncomputable def mario_weight_ounces : ℝ := 8
noncomputable def lydia_weight_ounces : ℝ := 24

theorem nicolai_ate_6_pounds_of_peaches :
  (total_weight_pounds * pound_to_ounce - (mario_weight_ounces + lydia_weight_ounces)) / pound_to_ounce = 6 :=
by
  sorry

end nicolai_ate_6_pounds_of_peaches_l143_143146


namespace payback_duration_l143_143518

-- Define constants for the problem conditions
def C : ℝ := 25000
def R : ℝ := 4000
def E : ℝ := 1500

-- Formal statement to be proven
theorem payback_duration : C / (R - E) = 10 := 
by
  sorry

end payback_duration_l143_143518


namespace binomial_expression_value_l143_143863

theorem binomial_expression_value :
  (Nat.choose 1 2023 * 3^2023) / Nat.choose 4046 2023 = 0 := by
  sorry

end binomial_expression_value_l143_143863


namespace tangent_line_to_circle_l143_143874

theorem tangent_line_to_circle {c : ℝ} (h : c > 0) :
  (∀ x y : ℝ, x^2 + y^2 = 8 → x + y = c) ↔ c = 4 := sorry

end tangent_line_to_circle_l143_143874


namespace mean_age_of_oldest_three_l143_143549

theorem mean_age_of_oldest_three (x : ℕ) (h : (x + (x + 1) + (x + 2)) / 3 = 6) : 
  (((x + 4) + (x + 5) + (x + 6)) / 3 = 10) := 
by
  sorry

end mean_age_of_oldest_three_l143_143549


namespace radius_of_larger_ball_l143_143559

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem radius_of_larger_ball :
  (six_ball_volume : volume_of_sphere 2 * 6 = volume_of_sphere R) →
  R = 2 * Real.cbrt 3 := by
  sorry

end radius_of_larger_ball_l143_143559


namespace tetrahedron_volume_PQRS_l143_143789

noncomputable def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ :=
  let a := PQ in
  let b := PR in
  let c := PS in
  let d := QR in
  let e := QS in
  let f := RS in
  let M := λ i j, match (i, j) with
    | (0, 0) => 0   | (0, 1) => 1     | (0, 2) => 1     | (0, 3) => 1     | (0, 4) => 1
    | (1, 0) => 1   | (1, 1) => 0     | (1, 2) => a^2   | (1, 3) => b^2   | (1, 4) => c^2
    | (2, 0) => 1   | (2, 1) => a^2   | (2, 2) => 0     | (2, 3) => d^2   | (2, 4) => e^2
    | (3, 0) => 1   | (3, 1) => b^2   | (3, 2) => d^2   | (3, 3) => 0     | (3, 4) => f^2
    | (4, 0) => 1   | (4, 1) => c^2   | (4, 2) => e^2   | (4, 3) => f^2   | (4, 4) => 0
    | _ => 0
  in
  (Real.sqrt (Matrix.det (Matrix.of (Fin 5) (Fin 5) M))) / 288

theorem tetrahedron_volume_PQRS : tetrahedron_volume 6 4 3 5 7 (Real.sqrt 94) = 2 :=
by
  -- Omitted proof steps
  sorry

end tetrahedron_volume_PQRS_l143_143789


namespace number_of_female_athletes_drawn_l143_143321

def total_athletes (male female : ℕ) : ℕ := male + female
def proportion_of_females (total female : ℕ) : ℚ := (female : ℚ) / (total : ℚ)
def expected_females_drawn (proportion : ℚ) (sample_size : ℕ) : ℚ := proportion * (sample_size : ℚ)

theorem number_of_female_athletes_drawn :
  let total := total_athletes 48 36 in
  let proportion := proportion_of_females total 36 in
  let expected_drawn := expected_females_drawn proportion 21 in
  expected_drawn = 9 := 
by
  let total := total_athletes 48 36
  let proportion := proportion_of_females total 36
  let expected_drawn := expected_females_drawn proportion 21
  have h_clean: (expected_drawn : ℚ) = 9 := sorry -- proof skipped
  exact h_clean

end number_of_female_athletes_drawn_l143_143321


namespace rahul_share_of_payment_l143_143164

-- Definitions
def rahulWorkDays : ℕ := 3
def rajeshWorkDays : ℕ := 2
def totalPayment : ℤ := 355

-- Theorem statement
theorem rahul_share_of_payment :
  let rahulWorkRate := 1 / (rahulWorkDays : ℝ)
  let rajeshWorkRate := 1 / (rajeshWorkDays : ℝ)
  let combinedWorkRate := rahulWorkRate + rajeshWorkRate
  let rahulShareRatio := rahulWorkRate / combinedWorkRate
  let rahulShare := (totalPayment : ℝ) * rahulShareRatio
  rahulShare = 142 :=
by
  sorry

end rahul_share_of_payment_l143_143164


namespace boys_in_class_is_120_l143_143141

-- Definitions from conditions
def num_boys_in_class (number_of_girls number_of_boys : Nat) : Prop :=
  ∃ x : Nat, number_of_girls = 5 * x ∧ number_of_boys = 6 * x ∧
             (5 * x - 20) * 3 = 2 * (6 * x)

-- The theorem proving that given the conditions, the number of boys in the class is 120.
theorem boys_in_class_is_120 (number_of_girls number_of_boys : Nat) (h : num_boys_in_class number_of_girls number_of_boys) :
  number_of_boys = 120 :=
by
  sorry

end boys_in_class_is_120_l143_143141


namespace actual_time_when_watch_reads_11_pm_is_correct_l143_143332

-- Define the conditions
def noon := 0 -- Time when Cassandra sets her watch to the correct time
def actual_time_2_pm := 120 -- 2:00 PM in minutes
def watch_time_2_pm := 113.2 -- 1:53 PM and 12 seconds in minutes (113 minutes + 0.2 minutes)

-- Define the goal
def actual_time_watch_reads_11_pm := 731.25 -- 12:22 PM and 15 seconds in minutes from noon

-- Provide the theorem statement without proof
theorem actual_time_when_watch_reads_11_pm_is_correct :
  actual_time_watch_reads_11_pm = 731.25 :=
sorry

end actual_time_when_watch_reads_11_pm_is_correct_l143_143332


namespace tea_bags_l143_143937

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l143_143937


namespace find_f_minus3_and_f_2009_l143_143216

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Conditions
axiom h1 : is_odd f
axiom h2 : f 1 = 2
axiom h3 : ∀ x : ℝ, f (x + 6) = f x + f 3

-- Questions
theorem find_f_minus3_and_f_2009 : f (-3) = 0 ∧ f 2009 = -2 :=
by 
  sorry

end find_f_minus3_and_f_2009_l143_143216


namespace binom_12_10_eq_66_l143_143342

theorem binom_12_10_eq_66 : (nat.choose 12 10) = 66 := by
  sorry

end binom_12_10_eq_66_l143_143342


namespace area_OPA_l143_143397

variable (x : ℝ)

def y (x : ℝ) : ℝ := -x + 6

def A : ℝ × ℝ := (4, 0)
def O : ℝ × ℝ := (0, 0)
def P (x : ℝ) : ℝ × ℝ := (x, y x)

def area_triangle (O A P : ℝ × ℝ) : ℝ := 
  0.5 * abs (A.fst * P.snd + P.fst * O.snd + O.fst * A.snd - A.snd * P.fst - P.snd * O.fst - O.snd * A.fst)

theorem area_OPA : 0 < x ∧ x < 6 → area_triangle O A (P x) = 12 - 2 * x := by
  -- proof to be provided here
  sorry


end area_OPA_l143_143397


namespace larger_acute_angle_right_triangle_l143_143091

theorem larger_acute_angle_right_triangle (x : ℝ) (h1 : x > 0) (h2 : x + 5 * x = 90) : 5 * x = 75 := by
  sorry

end larger_acute_angle_right_triangle_l143_143091


namespace sum_of_faces_edges_vertices_l143_143157

def cube_faces : ℕ := 6
def cube_edges : ℕ := 12
def cube_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices :
  cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end sum_of_faces_edges_vertices_l143_143157


namespace smallest_interesting_number_l143_143831

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l143_143831


namespace bowling_tournament_l143_143036

-- Definition of the problem conditions
def playoff (num_bowlers: Nat): Nat := 
  if num_bowlers < 5 then
    0
  else
    2^(num_bowlers - 1)

-- Theorem statement to prove
theorem bowling_tournament (num_bowlers: Nat) (h: num_bowlers = 5): playoff num_bowlers = 16 := by
  sorry

end bowling_tournament_l143_143036


namespace silvia_percentage_shorter_l143_143103

theorem silvia_percentage_shorter :
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  (abs (( (j - s) / j) * 100 - 25) < 1) :=
by
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  show (abs (( (j - s) / j) * 100 - 25) < 1)
  sorry

end silvia_percentage_shorter_l143_143103


namespace instantaneous_velocity_at_t3_l143_143136

open Real

variable (t : ℝ)

def h : ℝ → ℝ := λ t, 15 * t - t ^ 2

theorem instantaneous_velocity_at_t3 : deriv h 3 = 9 := 
by
  sorry

end instantaneous_velocity_at_t3_l143_143136


namespace no_integers_satisfy_eq_l143_143633

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^2 + 1954 = n^2) := 
by
  sorry

end no_integers_satisfy_eq_l143_143633


namespace minute_hand_rotation_l143_143088

theorem minute_hand_rotation (minutes : ℕ) (degrees_per_minute : ℝ) (radian_conversion_factor : ℝ) : 
  minutes = 10 → 
  degrees_per_minute = 360 / 60 → 
  radian_conversion_factor = π / 180 → 
  (-(degrees_per_minute * minutes * radian_conversion_factor) = -(π / 3)) := 
by
  intros hminutes hdegrees hfactor
  rw [hminutes, hdegrees, hfactor]
  simp
  sorry

end minute_hand_rotation_l143_143088


namespace vector_dot_product_l143_143078

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem vector_dot_product :
  let a := (sin_deg 55, sin_deg 35)
  let b := (sin_deg 25, sin_deg 65)
  dot_product a b = (Real.sqrt 3) / 2 :=
by
  sorry

end vector_dot_product_l143_143078


namespace geom_progression_n_eq_6_l143_143910

theorem geom_progression_n_eq_6
  (a r : ℝ)
  (h_r : r = 6)
  (h_ratio : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217) :
  n = 6 :=
by
  sorry

end geom_progression_n_eq_6_l143_143910


namespace number_of_tea_bags_l143_143946

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l143_143946


namespace find_k_l143_143995

theorem find_k (k : ℕ) : (∃ n : ℕ, 2^k + 8*k + 5 = n^2) ↔ k = 2 := by
  sorry

end find_k_l143_143995


namespace tea_bags_number_l143_143952

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l143_143952


namespace fractions_order_and_non_equality_l143_143592

theorem fractions_order_and_non_equality:
  (37 / 29 < 41 / 31) ∧ (41 / 31 < 31 / 23) ∧ 
  ((37 / 29 ≠ 4 / 3) ∧ (41 / 31 ≠ 4 / 3) ∧ (31 / 23 ≠ 4 / 3)) := by
  sorry

end fractions_order_and_non_equality_l143_143592


namespace shoe_price_monday_final_price_l143_143932

theorem shoe_price_monday_final_price : 
  let thursday_price := 50
  let friday_markup_rate := 0.15
  let monday_discount_rate := 0.12
  let friday_price := thursday_price * (1 + friday_markup_rate)
  let monday_price := friday_price * (1 - monday_discount_rate)
  monday_price = 50.6 := by
  sorry

end shoe_price_monday_final_price_l143_143932


namespace big_sale_commission_l143_143771

theorem big_sale_commission (avg_increase : ℝ) (new_avg : ℝ) (num_sales : ℕ) 
  (prev_avg := new_avg - avg_increase)
  (total_prev := prev_avg * (num_sales - 1))
  (total_new := new_avg * num_sales)
  (C := total_new - total_prev) :
  avg_increase = 150 → new_avg = 250 → num_sales = 6 → C = 1000 :=
by
  intros 
  sorry

end big_sale_commission_l143_143771


namespace horse_rent_problem_l143_143598

theorem horse_rent_problem (total_rent : ℝ) (b_payment : ℝ) (a_horses b_horses c_horses : ℝ) 
  (a_months b_months c_months : ℝ) (h_total_rent : total_rent = 870) (h_b_payment : b_payment = 360)
  (h_a_horses : a_horses = 12) (h_b_horses : b_horses = 16) (h_c_horses : c_horses = 18) 
  (h_b_months : b_months = 9) (h_c_months : c_months = 6) : 
  ∃ (a_months : ℝ), (a_horses * a_months * 2.5 + b_payment + c_horses * c_months * 2.5 = total_rent) :=
by
  use 8
  sorry

end horse_rent_problem_l143_143598


namespace largest_integer_less_than_100_with_remainder_5_l143_143680

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143680


namespace length_OD1_l143_143516

-- Define the hypothesis of the problem
noncomputable def sphere_center : Point := sorry -- center O of the sphere
noncomputable def radius_sphere : ℝ := 10 -- radius of the sphere

-- Define face intersection properties
noncomputable def face_AA1D1D_radius : ℝ := 1
noncomputable def face_A1B1C1D1_radius : ℝ := 1
noncomputable def face_CDD1C1_radius : ℝ := 3

-- Define the coordinates of D1 (or in abstract form, we'll assume it is a known point)
noncomputable def segment_OD1 : ℝ := sorry -- Length of OD1 segment to be calculated

-- The main theorem to prove
theorem length_OD1 : 
  -- Given conditions
  (face_AA1D1D_radius = 1) ∧ 
  (face_A1B1C1D1_radius = 1) ∧ 
  (face_CDD1C1_radius = 3) ∧ 
  (radius_sphere = 10) →
  -- Prove the length of segment OD1 is 17
  segment_OD1 = 17 :=
by
  sorry

end length_OD1_l143_143516


namespace sequence_squared_l143_143501

theorem sequence_squared (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = a (n - 1) + 2 * (n - 1)) 
  : ∀ n, a n = n^2 := 
by
  sorry

end sequence_squared_l143_143501


namespace belfried_payroll_l143_143445

noncomputable def tax_paid (payroll : ℝ) : ℝ :=
  if payroll < 200000 then 0 else 0.002 * (payroll - 200000)

theorem belfried_payroll (payroll : ℝ) (h : tax_paid payroll = 400) : payroll = 400000 :=
by
  sorry

end belfried_payroll_l143_143445


namespace complete_square_solution_l143_143420

theorem complete_square_solution :
  ∀ (x : ℝ), (x^2 + 8*x + 9 = 0) → ((x + 4)^2 = 7) :=
by
  intro x h_eq
  sorry

end complete_square_solution_l143_143420


namespace intersect_range_of_f_l143_143794

open Real

def f (x : ℝ) : ℝ := sin x + 2 * abs (sin x)

theorem intersect_range_of_f : 
  ∀ k : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * π → f x ≠ k) → (1 < k ∧ k < 3) := 
by
  sorry

end intersect_range_of_f_l143_143794


namespace largest_integer_less_than_hundred_with_remainder_five_l143_143670

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l143_143670


namespace cos_arcsin_l143_143472

theorem cos_arcsin {θ : ℝ} (h : sin θ = 8/17) : cos θ = 15/17 :=
sorry

end cos_arcsin_l143_143472


namespace least_sum_of_exponents_520_l143_143386

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end least_sum_of_exponents_520_l143_143386


namespace wolves_total_games_l143_143329

theorem wolves_total_games
  (x y : ℕ) -- Before district play, the Wolves had won x games out of y games.
  (hx : x = 40 * y / 100) -- The Wolves had won 40% of their basketball games before district play.
  (hx' : 5 * x = 2 * y)
  (hy : 60 * (y + 10) / 100 = x + 9) -- They finished the season having won 60% of their total games.
  : y + 10 = 25 := by
  sorry

end wolves_total_games_l143_143329


namespace range_of_a_for_function_min_max_l143_143720

theorem range_of_a_for_function_min_max 
  (a : ℝ) 
  (h_min : ∀ x ∈ [-1, 1], x = -1 → x^2 + a * x + 3 ≤ y) 
  (h_max : ∀ x ∈ [-1, 1], x = 1 → x^2 + a * x + 3 ≥ y) : 
  2 ≤ a := 
sorry

end range_of_a_for_function_min_max_l143_143720


namespace cos_arcsin_eq_l143_143468

theorem cos_arcsin_eq : ∀ (x : ℝ), x = 8 / 17 → cos (arcsin x) = 15 / 17 :=
by 
  intro x hx
  have h1 : θ = arcsin x := sorry -- by definition θ = arcsin x
  have h2 : sin θ = x := sorry -- by definition sin θ = x
  have h3 : (17:ℝ)^2 = a^2 + 8^2 := sorry -- Pythagorean theorem
  have h4 : a = 15 := sorry -- solved from h3
  show cos (arcsin x) = 15 / 17 := sorry -- proven from h2 and h4

end cos_arcsin_eq_l143_143468


namespace cookies_to_milk_l143_143568

theorem cookies_to_milk (milk_quarts : ℕ) (cookies : ℕ) (cups_in_quart : ℕ) 
  (H : milk_quarts = 3) (C : cookies = 24) (Q : cups_in_quart = 4) : 
  ∃ x : ℕ, x = 3 ∧ ∀ y : ℕ, y = 6 → x = (milk_quarts * cups_in_quart * y) / cookies := 
by {
  sorry
}

end cookies_to_milk_l143_143568


namespace right_triangle_angle_l143_143094

theorem right_triangle_angle (x : ℝ) (h1 : x + 5 * x = 90) : 5 * x = 75 :=
by
  sorry

end right_triangle_angle_l143_143094


namespace jack_vertical_displacement_l143_143916

theorem jack_vertical_displacement :
  (up_flights down_flights steps_per_flight inches_per_step : ℤ) (h0 : up_flights = 3)
  (h1 : down_flights = 6) (h2 : steps_per_flight = 12) (h3 : inches_per_step = 8) :
  (down_flights - up_flights) * steps_per_flight * inches_per_step / 12 = 24 := by
  sorry

end jack_vertical_displacement_l143_143916


namespace Lei_Lei_sheep_count_l143_143755

-- Define the initial average price and number of sheep as parameters
variables (a : ℝ) (x : ℕ)

-- Conditions as hypotheses
def condition1 : Prop := ∀ a x: ℝ,
  60 * x + 2 * (a + 60) = 90 * x + 2 * (a - 90)

-- The main problem stated as a theorem to be proved
theorem Lei_Lei_sheep_count (h : condition1) : x = 10 :=
sorry


end Lei_Lei_sheep_count_l143_143755


namespace find_y_l143_143902

theorem find_y (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  y = 0.25 := 
sorry

end find_y_l143_143902


namespace sum_of_integers_l143_143273

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 4 * Real.sqrt 34 := by
  sorry

end sum_of_integers_l143_143273


namespace greatest_cds_in_box_l143_143122

theorem greatest_cds_in_box (r c p n : ℕ) (hr : r = 14) (hc : c = 12) (hp : p = 8) (hn : n = 2) :
  n = Nat.gcd r (Nat.gcd c p) :=
by
  rw [hr, hc, hp]
  sorry

end greatest_cds_in_box_l143_143122


namespace simplify_fraction_eq_one_over_thirty_nine_l143_143418

theorem simplify_fraction_eq_one_over_thirty_nine :
  let a1 := (1 / 3)^1
  let a2 := (1 / 3)^2
  let a3 := (1 / 3)^3
  (1 / (1 / a1 + 1 / a2 + 1 / a3)) = 1 / 39 :=
by
  sorry

end simplify_fraction_eq_one_over_thirty_nine_l143_143418


namespace find_B_l143_143502

variable {U : Set ℕ}

def A : Set ℕ := {1, 3, 5, 7}
def complement_A : Set ℕ := {2, 4, 6}
def complement_B : Set ℕ := {1, 4, 6}
def B : Set ℕ := {2, 3, 5, 7}

theorem find_B
  (hU : U = A ∪ complement_A)
  (A_comp : ∀ x, x ∈ complement_A ↔ x ∉ A)
  (B_comp : ∀ x, x ∈ complement_B ↔ x ∉ B) :
  B = {2, 3, 5, 7} :=
sorry

end find_B_l143_143502


namespace find_four_numbers_l143_143198

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b = 2024) 
  (h2 : a + c = 2026) 
  (h3 : a + d = 2030) 
  (h4 : b + c = 2028) 
  (h5 : b + d = 2032) 
  (h6 : c + d = 2036) : 
  (a = 1011 ∧ b = 1012 ∧ c = 1013 ∧ d = 1015) := 
sorry

end find_four_numbers_l143_143198


namespace farmer_has_42_cows_left_l143_143020

def initial_cows : ℕ := 51
def added_cows : ℕ := 5
def fraction_sold : ℚ := 1/4

theorem farmer_has_42_cows_left : 
  let total_cows := initial_cows + added_cows in
  let cows_sold := total_cows * fraction_sold in
  total_cows - cows_sold = 42 := 
by 
  -- The proof would go here, but we are only required to state the theorem.
  sorry

end farmer_has_42_cows_left_l143_143020


namespace form_of_reasoning_is_incorrect_l143_143281

-- Definitions from the conditions
def some_rational_numbers_are_fractions : Prop := 
  ∃ q : ℚ, ∃ f : ℚ, q = f / 1

def integers_are_rational_numbers : Prop :=
  ∀ z : ℤ, ∃ q : ℚ, q = z

-- The proposition to be proved
theorem form_of_reasoning_is_incorrect (h1 : some_rational_numbers_are_fractions) (h2 : integers_are_rational_numbers) : 
  ¬ ∀ z : ℤ, ∃ f : ℚ, f = z  := sorry

end form_of_reasoning_is_incorrect_l143_143281


namespace range_of_m_l143_143211
-- Import the entire math library

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0 
def q (x m : ℝ) : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0 

-- Main theorem statement
theorem range_of_m (m : ℝ) (h1 : 0 < m) 
(hsuff : ∀ x : ℝ, p x → q x m) 
(hnsuff : ¬ (∀ x : ℝ, q x m → p x)) : m ≥ 9 := 
sorry

end range_of_m_l143_143211


namespace slope_of_line_l143_143815

theorem slope_of_line (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 / x + 4 / y = 0) : 
  ∃ m : ℝ, m = -4 / 3 := 
sorry

end slope_of_line_l143_143815


namespace marks_lost_per_wrong_answer_l143_143742

theorem marks_lost_per_wrong_answer (score_per_correct : ℕ) (total_questions : ℕ) 
(total_score : ℕ) (correct_attempts : ℕ) (wrong_attempts : ℕ) (marks_lost_total : ℕ)
(H1 : score_per_correct = 4)
(H2 : total_questions = 75)
(H3 : total_score = 125)
(H4 : correct_attempts = 40)
(H5 : wrong_attempts = total_questions - correct_attempts)
(H6 : marks_lost_total = (correct_attempts * score_per_correct) - total_score)
: (marks_lost_total / wrong_attempts) = 1 := by
  sorry

end marks_lost_per_wrong_answer_l143_143742


namespace sin_value_proof_l143_143878

theorem sin_value_proof (θ : ℝ) (h : Real.cos (5 * Real.pi / 12 - θ) = 1 / 3) :
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end sin_value_proof_l143_143878


namespace fraction_saved_l143_143597

variable {P : ℝ} (hP : P > 0)

theorem fraction_saved (f : ℝ) (hf0 : 0 ≤ f) (hf1 : f ≤ 1) (condition : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  sorry

end fraction_saved_l143_143597


namespace find_parameters_infinite_solutions_l143_143868

def system_has_infinite_solutions (a b : ℝ) :=
  ∀ x y : ℝ, 2 * (a - b) * x + 6 * y = a ∧ 3 * b * x + (a - b) * b * y = 1

theorem find_parameters_infinite_solutions :
  ∀ (a b : ℝ), 
  system_has_infinite_solutions a b ↔ 
    (a = (3 + Real.sqrt 17) / 2 ∧ b = (Real.sqrt 17 - 3) / 2) ∨
    (a = (3 - Real.sqrt 17) / 2 ∧ b = (-3 - Real.sqrt 17) / 2) ∨
    (a = -2 ∧ b = 1) ∨
    (a = -1 ∧ b = 2) :=
sorry

end find_parameters_infinite_solutions_l143_143868


namespace largest_digit_divisible_by_9_l143_143514

theorem largest_digit_divisible_by_9 : ∀ (B : ℕ), B < 10 → (∃ n : ℕ, 9 * n = 5 + B + 4 + 8 + 6 + 1) → B = 9 := by
  sorry

end largest_digit_divisible_by_9_l143_143514


namespace sarah_pencils_on_tuesday_l143_143781

theorem sarah_pencils_on_tuesday 
    (x : ℤ)
    (h1 : 20 + x + 3 * x = 92) : 
    x = 18 := 
by 
    sorry

end sarah_pencils_on_tuesday_l143_143781


namespace minimize_expression_l143_143065

variables {x y : ℝ}

theorem minimize_expression : ∃ (x y : ℝ), 2 * x^2 + 2 * x * y + y^2 - 2 * x - 1 = -2 :=
by sorry

end minimize_expression_l143_143065


namespace ab_zero_l143_143426

theorem ab_zero
  (a b : ℤ)
  (h : ∀ (m n : ℕ), ∃ (k : ℤ), a * (m : ℤ) ^ 2 + b * (n : ℤ) ^ 2 = k ^ 2) :
  a * b = 0 :=
sorry

end ab_zero_l143_143426


namespace find_a_l143_143893

theorem find_a (a : ℚ) (A : Set ℚ) (h : 3 ∈ A) (hA : A = {a + 2, 2 * a^2 + a}) : a = 3 / 2 := 
by
  sorry

end find_a_l143_143893


namespace otimes_2_3_eq_23_l143_143628

-- Define the new operation
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- The proof statement
theorem otimes_2_3_eq_23 : otimes 2 3 = 23 := 
  by 
  sorry

end otimes_2_3_eq_23_l143_143628


namespace laura_annual_income_l143_143750

theorem laura_annual_income (I T : ℝ) (q : ℝ)
  (h1 : I > 50000) 
  (h2 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000))
  (h3 : T = 0.01 * (q + 0.5) * I) : I = 56000 := 
by sorry

end laura_annual_income_l143_143750


namespace farmer_has_42_cows_left_l143_143022

-- Define the conditions
def initial_cows := 51
def added_cows := 5
def sold_fraction := 1 / 4

-- Lean statement to prove the number of cows left
theorem farmer_has_42_cows_left :
  (initial_cows + added_cows) - (sold_fraction * (initial_cows + added_cows)) = 42 :=
by
  -- skipping the proof part
  sorry

end farmer_has_42_cows_left_l143_143022


namespace projectile_reaches_49_first_time_at_1_point_4_l143_143551

-- Define the equation for the height of the projectile
def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t

-- State the theorem to prove
theorem projectile_reaches_49_first_time_at_1_point_4 :
  ∃ t : ℝ, height t = 49 ∧ (∀ t' : ℝ, height t' = 49 → t ≤ t') :=
sorry

end projectile_reaches_49_first_time_at_1_point_4_l143_143551


namespace gomoku_black_pieces_l143_143575

/--
Two students, A and B, are preparing to play a game of Gomoku but find that 
the box only contains a certain number of black and white pieces, each of the
same quantity, and the total does not exceed 10. Then, they find 20 more pieces 
(only black and white) and add them to the box. At this point, the ratio of 
the total number of white to black pieces is 7:8. We want to prove that the total number
of black pieces in the box after adding is 16.
-/
theorem gomoku_black_pieces (x y : ℕ) (hx : x = 15 * y - 160) (h_total : x + y ≤ 5)
  (h_ratio : 7 * (x + y) = 8 * (x + (20 - y))) : (x + y = 16) :=
by
  sorry

end gomoku_black_pieces_l143_143575


namespace final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l143_143866

-- Define the movements as a list of integers
def movements : List ℤ := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

-- Define the function to calculate the final position
def final_position (movements : List ℤ) : ℤ :=
  movements.foldl (· + ·) 0

-- Define the function to find the total distance walked (absolute sum)
def total_distance (movements : List ℤ) : ℕ :=
  movements.foldl (fun acc x => acc + x.natAbs) 0

-- Calorie consumption rate per kilometer (1000 meters)
def calories_per_kilometer : ℕ := 7000

-- Calculate the calories consumed
def calories_consumed (total_meters : ℕ) : ℕ :=
  (total_meters / 1000) * calories_per_kilometer

-- Lean 4 theorem statements

theorem final_position_west_of_bus_stop : final_position movements = -400 := by
  sorry

theorem distance_from_bus_stop : |final_position movements| = 400 := by
  sorry

theorem total_calories_consumed : calories_consumed (total_distance movements) = 44800 := by
  sorry

end final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l143_143866


namespace probability_red_or_blue_marbles_l143_143586

theorem probability_red_or_blue_marbles (red blue green total : ℕ) (h_red : red = 4) (h_blue : blue = 3) (h_green : green = 6) (h_total : total = red + blue + green) :
  (red + blue) / total = 7 / 13 :=
by
  sorry

end probability_red_or_blue_marbles_l143_143586


namespace fraction_of_third_is_eighth_l143_143581

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_third_is_eighth_l143_143581


namespace length_of_diagonal_EG_l143_143238

theorem length_of_diagonal_EG (EF FG GH HE : ℕ) (hEF : EF = 7) (hFG : FG = 15) 
  (hGH : GH = 7) (hHE : HE = 7) (primeEG : Prime EG) : EG = 11 ∨ EG = 13 :=
by
  -- Apply conditions and proof steps here
  sorry

end length_of_diagonal_EG_l143_143238


namespace rectangle_area_from_diagonal_l143_143852

theorem rectangle_area_from_diagonal (x : ℝ) (w : ℝ) (h_lw : 3 * w = 3 * w) (h_diag : x^2 = 10 * w^2) : 
    (3 * w^2 = (3 / 10) * x^2) :=
by 
sorry

end rectangle_area_from_diagonal_l143_143852


namespace triangle_perimeter_l143_143097

theorem triangle_perimeter (r : ℝ) (A B C P Q R S T : ℝ)
  (triangle_isosceles : A = C)
  (circle_tangent : P = A ∧ Q = B ∧ R = B ∧ S = C ∧ T = C)
  (center_dist : P + Q = 2 ∧ Q + R = 2 ∧ R + S = 2 ∧ S + T = 2) :
  2 * (A + B + C) = 6 := by
  sorry

end triangle_perimeter_l143_143097


namespace album_cost_l143_143449

-- Definition of the cost variables
variable (B C A : ℝ)

-- Conditions given in the problem
axiom h1 : B = C + 4
axiom h2 : B = 18
axiom h3 : C = 0.70 * A

-- Theorem to prove the cost of the album
theorem album_cost : A = 20 := sorry

end album_cost_l143_143449


namespace find_multiplication_value_l143_143456

-- Define the given conditions
def student_chosen_number : ℤ := 63
def subtracted_value : ℤ := 142
def result_after_subtraction : ℤ := 110

-- Define the value he multiplied the number by
def multiplication_value (x : ℤ) : Prop := 
  (student_chosen_number * x) - subtracted_value = result_after_subtraction

-- Statement to prove that the value he multiplied the number by is 4
theorem find_multiplication_value : 
  ∃ x : ℤ, multiplication_value x ∧ x = 4 :=
by 
  -- Placeholder for the actual proof
  sorry

end find_multiplication_value_l143_143456


namespace yogurt_count_l143_143130

theorem yogurt_count (Y : ℕ) 
  (ice_cream_cartons : ℕ := 20)
  (cost_ice_cream_per_carton : ℕ := 6)
  (cost_yogurt_per_carton : ℕ := 1)
  (spent_more_on_ice_cream : ℕ := 118)
  (total_cost_ice_cream : ℕ := ice_cream_cartons * cost_ice_cream_per_carton)
  (total_cost_yogurt : ℕ := Y * cost_yogurt_per_carton)
  (expenditure_condition : total_cost_ice_cream = total_cost_yogurt + spent_more_on_ice_cream) :
  Y = 2 :=
by {
  sorry
}

end yogurt_count_l143_143130


namespace part1_part2_l143_143466

open Real

variable {x y a: ℝ}

-- Condition for the second proof to avoid division by zero
variable (h1 : a ≠ 1) (h2 : a ≠ 4) (h3 : a ≠ -4)

theorem part1 : (x + y)^2 + y * (3 * x - y) = x^2 + 5 * (x * y) := 
by sorry

theorem part2 (h1: a ≠ 1) (h2: a ≠ 4) (h3: a ≠ -4) : 
  ((4 - a^2) / (a - 1) + a) / ((a^2 - 16) / (a - 1)) = -1 / (a + 4) := 
by sorry

end part1_part2_l143_143466


namespace line_intersects_circle_two_points_find_value_of_m_l143_143209

open Real

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

theorem line_intersects_circle_two_points (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ),
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  x1 ≠ x2 ∨ y1 ≠ y2 := sorry

theorem find_value_of_m (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ), 
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  dist (x1, y1) (x2, y2) = sqrt 17 → 
  m = sqrt 3 ∨ m = -sqrt 3 := sorry

end line_intersects_circle_two_points_find_value_of_m_l143_143209


namespace value_of_k_l143_143749

theorem value_of_k (k m : ℝ)
    (h1 : m = k / 3)
    (h2 : 2 = k / (3 * m - 1)) :
    k = 2 := by
  sorry

end value_of_k_l143_143749


namespace number_of_tea_bags_l143_143947

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l143_143947


namespace polynomial_solution_l143_143486

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x, (x + 2019) * (P.eval x) = x * (P.eval (x + 1))) :
  ∃ C : ℝ, P = Polynomial.C C * Polynomial.X * (Polynomial.X + 2018) :=
sorry

end polynomial_solution_l143_143486


namespace eight_faucets_fill_time_in_seconds_l143_143711

open Nat

-- Definitions under the conditions
def four_faucets_rate (gallons : ℕ) (minutes : ℕ) : ℕ := gallons / minutes

def one_faucet_rate (four_faucets_rate : ℕ) : ℕ := four_faucets_rate / 4

def eight_faucets_rate (one_faucet_rate : ℕ) : ℕ := one_faucet_rate * 8

def time_to_fill (rate : ℕ) (gallons : ℕ) : ℕ := gallons / rate

-- Main theorem to prove 
theorem eight_faucets_fill_time_in_seconds (gallons_tub : ℕ) (four_faucets_time : ℕ) :
    let four_faucets_rate := four_faucets_rate 200 8
    let one_faucet_rate := one_faucet_rate four_faucets_rate
    let rate_eight_faucets := eight_faucets_rate one_faucet_rate
    let time_fill := time_to_fill rate_eight_faucets 50
    gallons_tub = 50 ∧ four_faucets_time = 8 ∧ rate_eight_faucets = 50 -> time_fill * 60 = 60 :=
by
    intros
    sorry

end eight_faucets_fill_time_in_seconds_l143_143711


namespace evaluate_sum_l143_143867

theorem evaluate_sum : (-1:ℤ) ^ 2010 + (-1:ℤ) ^ 2011 + (1:ℤ) ^ 2012 - (1:ℤ) ^ 2013 + (-1:ℤ) ^ 2014 = 0 := by
  sorry

end evaluate_sum_l143_143867


namespace number_of_tea_bags_l143_143945

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l143_143945


namespace inequality_holds_l143_143015

theorem inequality_holds (c : ℝ) (X Y : ℝ) (h1 : X^2 - c * X - c = 0) (h2 : Y^2 - c * Y - c = 0) :
    X^3 + Y^3 + (X * Y)^3 ≥ 0 :=
sorry

end inequality_holds_l143_143015


namespace total_amount_spent_l143_143016

theorem total_amount_spent (num_pigs num_hens avg_price_hen avg_price_pig : ℕ)
                          (h_num_pigs : num_pigs = 3)
                          (h_num_hens : num_hens = 10)
                          (h_avg_price_hen : avg_price_hen = 30)
                          (h_avg_price_pig : avg_price_pig = 300) :
                          num_hens * avg_price_hen + num_pigs * avg_price_pig = 1200 :=
by
  sorry

end total_amount_spent_l143_143016


namespace li_bai_initial_wine_l143_143461

theorem li_bai_initial_wine (x : ℕ) 
  (h : (((((x * 2 - 2) * 2 - 2) * 2 - 2) * 2 - 2) = 2)) : 
  x = 2 :=
by
  sorry

end li_bai_initial_wine_l143_143461


namespace rectangular_prism_volume_l143_143030

theorem rectangular_prism_volume 
(l w h : ℝ) 
(h1 : l * w = 18) 
(h2 : w * h = 32) 
(h3 : l * h = 48) : 
l * w * h = 288 :=
sorry

end rectangular_prism_volume_l143_143030


namespace polynomial_sum_coeff_l143_143484

-- Definitions for the polynomials given
def poly1 (d : ℤ) : ℤ := 15 * d^3 + 19 * d^2 + 17 * d + 18
def poly2 (d : ℤ) : ℤ := 3 * d^3 + 4 * d + 2

-- The main statement to prove
theorem polynomial_sum_coeff :
  let p := 18
  let q := 19
  let r := 21
  let s := 20
  p + q + r + s = 78 :=
by
  sorry

end polynomial_sum_coeff_l143_143484


namespace trillion_value_l143_143745

def ten_thousand : ℕ := 10^4
def million : ℕ := 10^6
def billion : ℕ := ten_thousand * million

theorem trillion_value : (ten_thousand * ten_thousand * billion) = 10^16 :=
by
  sorry

end trillion_value_l143_143745


namespace min_value_of_m_l143_143879

theorem min_value_of_m (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  a^2 + b^2 + c^2 ≥ 3 :=
sorry

end min_value_of_m_l143_143879


namespace probability_point_in_circle_l143_143741

theorem probability_point_in_circle (r : ℝ) (h: r = 2) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := Real.pi * r ^ 2
  (area_circle / area_square) = Real.pi / 4 :=
by
  sorry

end probability_point_in_circle_l143_143741


namespace gcd_lcm_product_eq_l143_143983

theorem gcd_lcm_product_eq (a b : ℕ) : gcd a b * lcm a b = a * b := by
  sorry

example : ∃ (a b : ℕ), a = 30 ∧ b = 75 ∧ gcd a b * lcm a b = a * b :=
  ⟨30, 75, rfl, rfl, gcd_lcm_product_eq 30 75⟩

end gcd_lcm_product_eq_l143_143983


namespace probability_yellow_straight_l143_143090

open ProbabilityTheory

-- Definitions and assumptions
def P_G : ℚ := 2 / 3
def P_S : ℚ := 1 / 2
def P_Y : ℚ := 1 - P_G

lemma prob_Y_and_S (independence : Indep P_Y P_S) : P_Y * P_S = 1 / 6 :=
by
  have P_Y : ℚ := 1 - P_G
  have P_S : ℚ := 1 / 2
  calc P_Y * P_S = (1 - P_G) * P_S : by rfl
             ... = (1 - 2 / 3) * 1 / 2 : by rfl
             ... = 1 / 3 * 1 / 2 : by rfl
             ... = 1 / 6 : by norm_num

-- theorem to prove
theorem probability_yellow_straight (independence : Indep P_Y P_S) : P_Y * P_S = 1 / 6 :=
prob_Y_and_S independence

end probability_yellow_straight_l143_143090


namespace new_light_wattage_l143_143311

theorem new_light_wattage (w_old : ℕ) (p : ℕ) (w_new : ℕ) (h1 : w_old = 110) (h2 : p = 30) (h3 : w_new = w_old + (p * w_old / 100)) : w_new = 143 :=
by
  -- Using the conditions provided
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end new_light_wattage_l143_143311


namespace sum_of_integers_l143_143274

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 4 * Real.sqrt 34 := by
  sorry

end sum_of_integers_l143_143274


namespace function_does_not_have_property_P_l143_143522

-- Definition of property P
def hasPropertyP (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f ((x1 + x2) / 2) = (f x1 + f x2) / 2

-- Function in question
def f (x : ℝ) : ℝ :=
  x^2

-- Statement that function f does not have property P
theorem function_does_not_have_property_P : ¬hasPropertyP f :=
  sorry

end function_does_not_have_property_P_l143_143522


namespace pascal_row_10_sum_l143_143589

-- Definition: sum of the numbers in Row n of Pascal's Triangle is 2^n
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- Theorem: sum of the numbers in Row 10 of Pascal's Triangle is 1024
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  sorry

end pascal_row_10_sum_l143_143589


namespace irreducible_fraction_l143_143204

-- Statement of the theorem
theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry -- Proof would be placed here

end irreducible_fraction_l143_143204


namespace perpendicular_lines_l143_143889

theorem perpendicular_lines :
  (∀ (x y : ℝ), (4 * y - 3 * x = 16)) ∧ 
  (∀ (x y : ℝ), (3 * y + 4 * x = 15)) → 
  (∃ (m1 m2 : ℝ), m1 * m2 = -1) :=
by
  sorry

end perpendicular_lines_l143_143889


namespace monthly_fee_for_second_plan_l143_143316

theorem monthly_fee_for_second_plan 
  (monthly_fee_first_plan : ℝ) 
  (rate_first_plan : ℝ) 
  (rate_second_plan : ℝ) 
  (minutes : ℕ) 
  (monthly_fee_second_plan : ℝ) :
  monthly_fee_first_plan = 22 -> 
  rate_first_plan = 0.13 -> 
  rate_second_plan = 0.18 -> 
  minutes = 280 -> 
  (22 + 0.13 * 280 = monthly_fee_second_plan + 0.18 * 280) -> 
  monthly_fee_second_plan = 8 := 
by
  intros h_fee_first_plan h_rate_first_plan h_rate_second_plan h_minutes h_equal_costs
  sorry

end monthly_fee_for_second_plan_l143_143316


namespace tom_books_l143_143110

-- Definitions based on the conditions
def joan_books : ℕ := 10
def total_books : ℕ := 48

-- The theorem statement: Proving that Tom has 38 books
theorem tom_books : (total_books - joan_books) = 38 := by
  -- Here we would normally provide a proof, but we use sorry to skip this.
  sorry

end tom_books_l143_143110


namespace cheryl_material_usage_l143_143556

theorem cheryl_material_usage:
  let bought := (3 / 8) + (1 / 3)
  let left := (15 / 40)
  let used := bought - left
  used = (1 / 3) := 
by
  sorry

end cheryl_material_usage_l143_143556


namespace max_groups_l143_143194

theorem max_groups (boys girls : ℕ) (h1 : boys = 120) (h2 : girls = 140) : Nat.gcd boys girls = 20 := 
  by
  rw [h1, h2]
  -- Proof steps would be here
  sorry

end max_groups_l143_143194


namespace total_action_figures_l143_143403

theorem total_action_figures (initial_figures cost_per_figure total_cost needed_figures : ℕ)
  (h1 : initial_figures = 7)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost = 72)
  (h4 : needed_figures = total_cost / cost_per_figure)
  : initial_figures + needed_figures = 16 :=
by
  sorry

end total_action_figures_l143_143403


namespace empty_seats_l143_143999

theorem empty_seats (total_seats : ℕ) (people_watching : ℕ) (h_total_seats : total_seats = 750) (h_people_watching : people_watching = 532) : 
  total_seats - people_watching = 218 :=
by
  sorry

end empty_seats_l143_143999


namespace number_of_people_who_purchased_only_book_A_l143_143604

-- Define the conditions and the problem
theorem number_of_people_who_purchased_only_book_A 
    (total_A : ℕ) (total_B : ℕ) (both_AB : ℕ) (only_B : ℕ) :
    (total_A = 2 * total_B) → 
    (both_AB = 500) → 
    (both_AB = 2 * only_B) → 
    (total_B = only_B + both_AB) → 
    (total_A - both_AB = 1000) :=
by
  sorry

end number_of_people_who_purchased_only_book_A_l143_143604


namespace number_of_integers_between_sqrt10_sqrt100_l143_143727

theorem number_of_integers_between_sqrt10_sqrt100 : 
  (set.Ico ⌈Real.sqrt 10⌉₊ (⌊Real.sqrt 100⌋₊ + 1)).card = 7 := by
  sorry

end number_of_integers_between_sqrt10_sqrt100_l143_143727


namespace largest_int_with_remainder_5_lt_100_l143_143686

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l143_143686


namespace least_sum_of_exponents_of_powers_of_2_l143_143379

theorem least_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ s : Finset ℕ, (∑ x in s, 2^x = n) ∧ (∀ t : Finset ℕ, (∑ x in t, 2^x = n) → s.sum id ≤ t.sum id) :=
sorry

end least_sum_of_exponents_of_powers_of_2_l143_143379


namespace junk_mail_per_red_or_white_house_l143_143172

noncomputable def pieces_per_house (total_pieces : ℕ) (total_houses : ℕ) : ℕ := 
  total_pieces / total_houses

noncomputable def total_pieces_for_type (pieces_per_house : ℕ) (houses_of_type : ℕ) : ℕ := 
  pieces_per_house * houses_of_type

noncomputable def total_pieces_for_red_or_white 
  (total_pieces : ℕ)
  (total_houses : ℕ)
  (white_houses : ℕ)
  (red_houses : ℕ) : ℕ :=
  let pieces_per_house := pieces_per_house total_pieces total_houses
  let pieces_for_white := total_pieces_for_type pieces_per_house white_houses
  let pieces_for_red := total_pieces_for_type pieces_per_house red_houses
  pieces_for_white + pieces_for_red

theorem junk_mail_per_red_or_white_house :
  ∀ (total_pieces : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ),
    total_pieces = 48 →
    total_houses = 8 →
    white_houses = 2 →
    red_houses = 3 →
    total_pieces_for_red_or_white total_pieces total_houses white_houses red_houses / (white_houses + red_houses) = 6 :=
by
  intros
  sorry

end junk_mail_per_red_or_white_house_l143_143172


namespace calc_exp_l143_143041

open Real

theorem calc_exp (x y : ℝ) : 
  (-(1/3) * (x^2) * y) ^ 3 = -(x^6 * y^3) / 27 := 
  sorry

end calc_exp_l143_143041


namespace multiple_of_students_in_restroom_l143_143042

theorem multiple_of_students_in_restroom 
    (num_desks_per_row : ℕ)
    (num_rows : ℕ)
    (desk_fill_fraction : ℚ)
    (total_students : ℕ)
    (students_restroom : ℕ)
    (absent_students : ℕ)
    (m : ℕ) :
    num_desks_per_row = 6 →
    num_rows = 4 →
    desk_fill_fraction = 2 / 3 →
    total_students = 23 →
    students_restroom = 2 →
    (num_rows * num_desks_per_row : ℕ) * desk_fill_fraction = 16 →
    (16 - students_restroom) = 14 →
    total_students - 14 - 2 = absent_students →
    absent_students = 7 →
    2 * m - 1 = 7 →
    m = 4
:= by
    intros;
    sorry

end multiple_of_students_in_restroom_l143_143042


namespace find_p_l143_143927

theorem find_p (p : ℤ)
  (h1 : ∀ (u v : ℤ), u > 0 → v > 0 → 5 * u ^ 2 - 5 * p * u + (66 * p - 1) = 0 ∧
    5 * v ^ 2 - 5 * p * v + (66 * p - 1) = 0) :
  p = 76 :=
sorry

end find_p_l143_143927


namespace squares_overlap_ratio_l143_143809

theorem squares_overlap_ratio (a b : ℝ) (h1 : 0.52 * a^2 = a^2 - (a^2 - 0.52 * a^2))
                             (h2 : 0.73 * b^2 = b^2 - (b^2 - 0.73 * b^2)) :
                             a / b = 3 / 4 := by
sorry

end squares_overlap_ratio_l143_143809


namespace smallest_even_n_l143_143453

theorem smallest_even_n (n : ℕ) :
  (∃ n, 0 < n ∧ n % 2 = 0 ∧ (∀ k, 1 ≤ k → k ≤ n / 2 → k = 2213 ∨ k = 3323 ∨ k = 6121) ∧ (2^k * (k!)) % (2213 * 3323 * 6121) = 0) → n = 12242 :=
sorry

end smallest_even_n_l143_143453


namespace poly_divisible_coeff_sum_eq_one_l143_143631

theorem poly_divisible_coeff_sum_eq_one (C D : ℂ) :
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^100 + C * x^2 + D * x + 1 = 0) →
  C + D = 1 :=
by
  sorry

end poly_divisible_coeff_sum_eq_one_l143_143631


namespace find_x_squared_plus_y_squared_l143_143208

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : x^2 + y^2 = 13 :=
by
  sorry

end find_x_squared_plus_y_squared_l143_143208


namespace find_r_l143_143114

theorem find_r (a b m p r : ℝ) (h1 : a * b = 4)
  (h2 : ∃ (q w : ℝ), (a + 2 / b = q ∧ b + 2 / a = w) ∧ q * w = r) :
  r = 9 :=
sorry

end find_r_l143_143114


namespace inequality_solution_l143_143891

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x^2 + 1) + x) - (2 / (Real.exp x + 1))

theorem inequality_solution :
  { x : ℝ | f x + f (2 * x - 1) > -2 } = { x : ℝ | x > 1 / 3 } :=
sorry

end inequality_solution_l143_143891


namespace three_x_minus_five_y_l143_143930

noncomputable def F : ℝ × ℝ :=
  let D := (15, 3)
  let E := (6, 8)
  ((D.1 + E.1) / 2, (D.2 + E.2) / 2)

theorem three_x_minus_five_y : (3 * F.1 - 5 * F.2) = 4 := by
  sorry

end three_x_minus_five_y_l143_143930


namespace right_triangle_angle_l143_143093

theorem right_triangle_angle (x : ℝ) (h1 : x + 5 * x = 90) : 5 * x = 75 :=
by
  sorry

end right_triangle_angle_l143_143093


namespace acute_angles_complementary_l143_143929

-- Given conditions
variables (α β : ℝ)
variables (α_acute : 0 < α ∧ α < π / 2) (β_acute : 0 < β ∧ β < π / 2)
variables (h : (sin α) ^ 2 + (sin β) ^ 2 = sin (α + β))

-- Statement we want to prove
theorem acute_angles_complementary : α + β = π / 2 :=
  sorry

end acute_angles_complementary_l143_143929


namespace max_value_of_expression_l143_143714

theorem max_value_of_expression (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 3) : 2 * x + 3 * y ≤ Real.sqrt 31 :=
sorry

end max_value_of_expression_l143_143714


namespace discrim_of_quad_l143_143629

-- Definition of the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -9
def c : ℤ := 4

-- Definition of the discriminant formula which needs to be proved as 1
def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

-- The proof problem statement
theorem discrim_of_quad : discriminant a b c = 1 := by
  sorry

end discrim_of_quad_l143_143629


namespace rice_in_each_container_l143_143415

theorem rice_in_each_container 
  (total_weight : ℚ) 
  (num_containers : ℕ)
  (conversion_factor : ℚ) 
  (equal_division : total_weight = 29 / 4 ∧ num_containers = 4 ∧ conversion_factor = 16) : 
  (total_weight / num_containers) * conversion_factor = 29 := 
by 
  sorry

end rice_in_each_container_l143_143415


namespace largest_integer_less_than_100_with_remainder_5_l143_143703

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l143_143703


namespace range_of_a_for_integer_solutions_l143_143056

theorem range_of_a_for_integer_solutions (a : ℝ) :
  (∃ x : ℤ, (a - 2 < x ∧ x ≤ 3)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_for_integer_solutions_l143_143056


namespace farmer_has_42_cows_left_l143_143021

def initial_cows : ℕ := 51
def added_cows : ℕ := 5
def fraction_sold : ℚ := 1/4

theorem farmer_has_42_cows_left : 
  let total_cows := initial_cows + added_cows in
  let cows_sold := total_cows * fraction_sold in
  total_cows - cows_sold = 42 := 
by 
  -- The proof would go here, but we are only required to state the theorem.
  sorry

end farmer_has_42_cows_left_l143_143021


namespace part_a_part_b_l143_143823

-- Part (a)
theorem part_a (n : ℕ) (h : n > 0) :
  (2 * n ∣ n * (n + 1) / 2) ↔ ∃ k : ℕ, n = 4 * k - 1 :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n > 0) :
  (2 * n + 1 ∣ n * (n + 1) / 2) ↔ (2 * n + 1 ≡ 1 [MOD 4]) ∨ (2 * n + 1 ≡ 3 [MOD 4]) :=
by sorry

end part_a_part_b_l143_143823


namespace ceil_sqrt_200_eq_15_l143_143635

theorem ceil_sqrt_200_eq_15 (h1 : Real.sqrt 196 = 14) (h2 : Real.sqrt 225 = 15) (h3 : 196 < 200 ∧ 200 < 225) : 
  Real.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l143_143635


namespace range_a_ge_one_l143_143882

theorem range_a_ge_one (a : ℝ) (x : ℝ) 
  (p : Prop := |x + 1| > 2) 
  (q : Prop := x > a) 
  (suff_not_necess_cond : ¬p → ¬q) : a ≥ 1 :=
sorry

end range_a_ge_one_l143_143882


namespace initial_maple_trees_l143_143566

theorem initial_maple_trees
  (initial_maple_trees : ℕ)
  (to_be_planted : ℕ)
  (final_maple_trees : ℕ)
  (h1 : to_be_planted = 9)
  (h2 : final_maple_trees = 11) :
  initial_maple_trees + to_be_planted = final_maple_trees → initial_maple_trees = 2 := 
by 
  sorry

end initial_maple_trees_l143_143566


namespace value_of_expression_l143_143003

theorem value_of_expression :
  (3 * (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) + 2) = 4373 :=
by
  sorry

end value_of_expression_l143_143003


namespace largest_int_with_remainder_5_lt_100_l143_143681

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l143_143681


namespace volume_ratio_l143_143291

theorem volume_ratio (a : ℕ) (b : ℕ) (ft_to_inch : ℕ) (h1 : a = 4) (h2 : b = 2 * ft_to_inch) (ft_to_inch_value : ft_to_inch = 12) :
  (a^3) / (b^3) = 1 / 216 :=
by
  sorry

end volume_ratio_l143_143291


namespace sum_of_base5_numbers_l143_143626

-- Definitions for the numbers in base 5
def n1_base5 := (1 * 5^2 + 3 * 5^1 + 2 * 5^0 : ℕ)
def n2_base5 := (2 * 5^2 + 1 * 5^1 + 4 * 5^0 : ℕ)
def n3_base5 := (3 * 5^2 + 4 * 5^1 + 1 * 5^0 : ℕ)

-- Sum the numbers in base 10
def sum_base10 := n1_base5 + n2_base5 + n3_base5

-- Define the base 5 value of the sum
def sum_base5 := 
  -- Convert the sum to base 5
  1 * 5^3 + 2 * 5^2 + 4 * 5^1 + 2 * 5^0

-- The theorem we want to prove
theorem sum_of_base5_numbers :
    (132 + 214 + 341 : ℕ) = 1242 := by
    sorry

end sum_of_base5_numbers_l143_143626


namespace total_oranges_l143_143259

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end total_oranges_l143_143259


namespace volume_of_rectangular_box_l143_143804

theorem volume_of_rectangular_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 20) 
  (h3 : x * z = 12) : 
  x * y * z = 60 := 
sorry

end volume_of_rectangular_box_l143_143804


namespace competition_end_time_is_5_35_am_l143_143309

def start_time : Nat := 15 * 60  -- 3:00 p.m. in minutes
def duration : Nat := 875  -- competition duration in minutes
def end_time : Nat := (start_time + duration) % (24 * 60)  -- competition end time in minutes

theorem competition_end_time_is_5_35_am :
  end_time = 5 * 60 + 35 :=  -- 5:35 a.m. in minutes
sorry

end competition_end_time_is_5_35_am_l143_143309


namespace mean_of_other_two_numbers_l143_143622

-- Definitions based on conditions in the problem.
def mean_of_four (numbers : List ℕ) : ℝ := 2187.25
def sum_of_numbers : ℕ := 1924 + 2057 + 2170 + 2229 + 2301 + 2365
def sum_of_four_numbers : ℝ := 4 * 2187.25
def sum_of_two_numbers := sum_of_numbers - sum_of_four_numbers

-- Theorem to assert the mean of the other two numbers.
theorem mean_of_other_two_numbers : (4297 / 2) = 2148.5 := by
  sorry

end mean_of_other_two_numbers_l143_143622


namespace fourth_month_sale_is_7200_l143_143454

-- Define the sales amounts for each month
def sale_first_month : ℕ := 6400
def sale_second_month : ℕ := 7000
def sale_third_month : ℕ := 6800
def sale_fifth_month : ℕ := 6500
def sale_sixth_month : ℕ := 5100
def average_sale : ℕ := 6500

-- Total requirements for the six months
def total_required_sales : ℕ := 6 * average_sale

-- Known sales for five months
def total_known_sales : ℕ := sale_first_month + sale_second_month + sale_third_month + sale_fifth_month + sale_sixth_month

-- Sale in the fourth month
def sale_fourth_month : ℕ := total_required_sales - total_known_sales

-- The theorem to prove
theorem fourth_month_sale_is_7200 : sale_fourth_month = 7200 :=
by
  sorry

end fourth_month_sale_is_7200_l143_143454


namespace boat_stream_ratio_l143_143008

-- Conditions: A man takes twice as long to row a distance against the stream as to row the same distance in favor of the stream.
theorem boat_stream_ratio (B S : ℝ) (h : ∀ (d : ℝ), d / (B - S) = 2 * (d / (B + S))) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l143_143008


namespace largest_integer_less_than_100_div_8_rem_5_l143_143658

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l143_143658


namespace speed_of_stream_l143_143825

-- Definitions based on given conditions
def speed_still_water := 24 -- km/hr
def distance_downstream := 140 -- km
def time_downstream := 5 -- hours

-- Proof problem statement
theorem speed_of_stream (v : ℕ) :
  24 + v = distance_downstream / time_downstream → v = 4 :=
by
  sorry

end speed_of_stream_l143_143825


namespace meeting_attendance_l143_143325

theorem meeting_attendance (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + 2 * B = 11) : A + B = 6 :=
sorry

end meeting_attendance_l143_143325


namespace cos_min_sin_eq_neg_sqrt_seven_half_l143_143880

variable (θ : ℝ)

theorem cos_min_sin_eq_neg_sqrt_seven_half (h1 : Real.sin θ + Real.cos θ = 0.5)
    (h2 : π / 2 < θ ∧ θ < π) : Real.cos θ - Real.sin θ = - Real.sqrt 7 / 2 := by
  sorry

end cos_min_sin_eq_neg_sqrt_seven_half_l143_143880


namespace rate_per_kg_mangoes_l143_143462

theorem rate_per_kg_mangoes 
  (weight_grapes : ℕ) 
  (rate_grapes : ℕ) 
  (weight_mangoes : ℕ) 
  (total_paid : ℕ)
  (total_grapes_cost : ℕ)
  (total_mangoes_cost : ℕ)
  (rate_mangoes : ℕ) 
  (h1 : weight_grapes = 14) 
  (h2 : rate_grapes = 54)
  (h3 : weight_mangoes = 10) 
  (h4 : total_paid = 1376) 
  (h5 : total_grapes_cost = weight_grapes * rate_grapes)
  (h6 : total_mangoes_cost = total_paid - total_grapes_cost) 
  (h7 : rate_mangoes = total_mangoes_cost / weight_mangoes):
  rate_mangoes = 62 :=
by
  sorry

end rate_per_kg_mangoes_l143_143462


namespace liam_finishes_on_wednesday_l143_143931

theorem liam_finishes_on_wednesday :
  let start_day := 3  -- Wednesday, where 0 represents Sunday
  let total_books := 20
  let total_days := (total_books * (total_books + 1)) / 2
  (total_days % 7) = 0 :=
by sorry

end liam_finishes_on_wednesday_l143_143931


namespace ratio_of_roots_l143_143706

variable (a b c x1 x2 : ℝ)

theorem ratio_of_roots :
  (a ≠ 0) → (x1 = 4 * x2) → (a * x1^2 + b * x1 + c = 0) → (a * x2^2 + b * x2 + c = 0) → 
  (16 * b^2 / (a * c) = 100) :=
by
  intros ha hx1 root1 root2
  sorry

end ratio_of_roots_l143_143706


namespace matrix_power_four_correct_l143_143334

theorem matrix_power_four_correct :
  let A := Matrix.of (fun i j => ![![2, -1], ![1, 1]].get i j) in
  A ^ 4 = Matrix.of (fun i j => ![![0, -9], ![9, -9]].get i j) :=
by
  sorry

end matrix_power_four_correct_l143_143334


namespace angle_invariant_under_magnification_l143_143004

theorem angle_invariant_under_magnification :
  ∀ (angle magnification : ℝ), angle = 10 → magnification = 5 → angle = 10 := by
  intros angle magnification h_angle h_magnification
  exact h_angle

end angle_invariant_under_magnification_l143_143004


namespace total_marbles_in_bag_l143_143306

theorem total_marbles_in_bag 
  (r b p : ℕ) 
  (h1 : 32 = r)
  (h2 : b = (7 * r) / 4) 
  (h3 : p = (3 * b) / 2) 
  : r + b + p = 172 := 
sorry

end total_marbles_in_bag_l143_143306


namespace sequence_sum_l143_143914

theorem sequence_sum (A B C D E F G H : ℕ) (hC : C = 7) 
    (h_sum : A + B + C = 36 ∧ B + C + D = 36 ∧ C + D + E = 36 ∧ D + E + F = 36 ∧ E + F + G = 36 ∧ F + G + H = 36) :
    A + H = 29 :=
sorry

end sequence_sum_l143_143914


namespace largest_integer_less_than_100_div_8_rem_5_l143_143659

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l143_143659


namespace taxi_fare_range_l143_143743

theorem taxi_fare_range (x : ℝ) (h : 12.5 + 2.4 * (x - 3) = 19.7) : 5 < x ∧ x ≤ 6 :=
by
  -- Given conditions and the equation, we need to prove the inequalities.
  have fare_eq : 12.5 + 2.4 * (x - 3) = 19.7 := h
  sorry

end taxi_fare_range_l143_143743


namespace problem_statement_l143_143619

def scientific_notation_correct (x : ℝ) : Prop :=
  x = 5.642 * 10 ^ 5

theorem problem_statement : scientific_notation_correct 564200 :=
by
  sorry

end problem_statement_l143_143619


namespace number_of_sixes_l143_143824

theorem number_of_sixes
  (total_runs : ℕ)
  (boundaries : ℕ)
  (percent_runs_by_running : ℚ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (runs_by_running : ℚ)
  (runs_by_boundaries : ℕ)
  (runs_by_sixes : ℕ)
  (number_of_sixes : ℕ)
  (h1 : total_runs = 120)
  (h2 : boundaries = 6)
  (h3 : percent_runs_by_running = 0.6)
  (h4 : runs_per_boundary = 4)
  (h5 : runs_per_six = 6)
  (h6 : runs_by_running = percent_runs_by_running * total_runs)
  (h7 : runs_by_boundaries = boundaries * runs_per_boundary)
  (h8 : runs_by_sixes = total_runs - (runs_by_running + runs_by_boundaries))
  (h9 : number_of_sixes = runs_by_sixes / runs_per_six)
  : number_of_sixes = 4 :=
by
  sorry

end number_of_sixes_l143_143824


namespace no_four_digit_numbers_divisible_by_11_l143_143081

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) :
  (a + b + c + d = 9) ∧ ((a + c) - (b + d)) % 11 = 0 → false :=
by
  sorry

end no_four_digit_numbers_divisible_by_11_l143_143081


namespace sqrt_of_sum_eq_l143_143985

noncomputable def cube_term : ℝ := 2 ^ 3
noncomputable def sum_cubes : ℝ := cube_term + cube_term + cube_term + cube_term
noncomputable def sqrt_sum : ℝ := Real.sqrt sum_cubes

theorem sqrt_of_sum_eq :
  sqrt_sum = 4 * Real.sqrt 2 :=
by
  sorry

end sqrt_of_sum_eq_l143_143985


namespace angies_monthly_salary_l143_143624

theorem angies_monthly_salary 
    (necessities_expense : ℕ)
    (taxes_expense : ℕ)
    (left_over : ℕ)
    (monthly_salary : ℕ) :
  necessities_expense = 42 → 
  taxes_expense = 20 → 
  left_over = 18 → 
  monthly_salary = necessities_expense + taxes_expense + left_over → 
  monthly_salary = 80 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end angies_monthly_salary_l143_143624


namespace probability_event_proof_l143_143229

noncomputable def probability_event_occur (deck_size : ℕ) (num_queens : ℕ) (num_jacks : ℕ) (num_reds : ℕ) : ℚ :=
  let prob_two_queens := (num_queens / deck_size) * ((num_queens - 1) / (deck_size - 1))
  let prob_at_least_one_jack := 
    (num_jacks / deck_size) * ((deck_size - num_jacks) / (deck_size - 1)) +
    ((deck_size - num_jacks) / deck_size) * (num_jacks / (deck_size - 1)) +
    (num_jacks / deck_size) * ((num_jacks - 1) / (deck_size - 1))
  let prob_both_red := (num_reds / deck_size) * ((num_reds - 1) / (deck_size - 1))
  prob_two_queens + prob_at_least_one_jack + prob_both_red

theorem probability_event_proof :
  probability_event_occur 52 4 4 26 = 89 / 221 :=
by
  sorry

end probability_event_proof_l143_143229


namespace heather_aprons_l143_143221

theorem heather_aprons :
  ∀ (total sewn already_sewn sewn_today half_remaining tomorrow_sew : ℕ),
    total = 150 →
    already_sewn = 13 →
    sewn_today = 3 * already_sewn →
    sewn = already_sewn + sewn_today →
    remaining = total - sewn →
    half_remaining = remaining / 2 →
    tomorrow_sew = half_remaining →
    tomorrow_sew = 49 := 
by 
  -- The proof is left as an exercise.
  sorry

end heather_aprons_l143_143221


namespace evaluate_expression_l143_143481

theorem evaluate_expression :
  let a := 12
  let b := 14
  let c := 18
  (144 * ((1:ℝ)/b - (1:ℝ)/c) + 196 * ((1:ℝ)/c - (1:ℝ)/a) + 324 * ((1:ℝ)/a - (1:ℝ)/b)) /
  (a * ((1:ℝ)/b - (1:ℝ)/c) + b * ((1:ℝ)/c - (1:ℝ)/a) + c * ((1:ℝ)/a - (1:ℝ)/b)) = a + b + c := by
  sorry

end evaluate_expression_l143_143481


namespace find_special_integers_l143_143200

theorem find_special_integers :
  ∃ count, count = (Finset.range 2018).filter (λ n, 
    (n - 2) * n * (n - 1) * (n - 7) % 7 = 0 ∧
    (n - 2) * n * (n - 1) * (n - 7) % 11 = 0 ∧
    (n - 2) * n * (n - 1) * (n - 7) % 13 = 0).card ∧
  count = 99 := sorry

end find_special_integers_l143_143200


namespace smallest_whole_number_larger_than_sum_l143_143490

theorem smallest_whole_number_larger_than_sum :
    let sum := 2 + 1 / 2 + 3 + 1 / 3 + 4 + 1 / 4 + 5 + 1 / 5 
    let smallest_whole := 16
    (sum < smallest_whole ∧ smallest_whole - 1 < sum) := 
by
    sorry

end smallest_whole_number_larger_than_sum_l143_143490


namespace proportion_of_triumphal_arch_photographs_l143_143196

-- Define the constants
variables (x y z t : ℕ) -- x = castles, y = triumphal arches, z = waterfalls, t = cathedrals

-- The conditions
axiom half_photographed : t + x + y + z = (3*y + 2*x + 2*z + y) / 2
axiom three_times_cathedrals : ∃ (a : ℕ), t = 3 * a ∧ y = a
axiom same_castles_waterfalls : ∃ (b : ℕ), t + z = x + y
axiom quarter_photographs_castles : x = (t + x + y + z) / 4
axiom second_castle_frequency : t + z = 2 * x
axiom every_triumphal_arch_photographed : ∀ (c : ℕ), y = c ∧ y = c

theorem proportion_of_triumphal_arch_photographs : 
  ∃ (p : ℚ), p = 1 / 4 ∧ p = y / ((t + x + y + z) / 2) :=
sorry

end proportion_of_triumphal_arch_photographs_l143_143196


namespace symmetric_points_on_parabola_l143_143373

theorem symmetric_points_on_parabola (x1 x2 y1 y2 m : ℝ)
  (h1: y1 = 2 * x1 ^ 2)
  (h2: y2 = 2 * x2 ^ 2)
  (h3: x1 * x2 = -1 / 2)
  (h4: y2 - y1 = 2 * (x2 ^ 2 - x1 ^ 2))
  (h5: (x1 + x2) / 2 = -1 / 4)
  (h6: (y1 + y2) / 2 = (x1 + x2) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end symmetric_points_on_parabola_l143_143373


namespace original_curve_eqn_l143_143439

-- Definitions based on conditions
def scaling_transformation_formula (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 3 * y)

def transformed_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

-- The proof problem to be shown in Lean
theorem original_curve_eqn {x y : ℝ} (h : transformed_curve (2 * x) (3 * y)) :
  4 * x^2 + 9 * y^2 = 1 :=
sorry

end original_curve_eqn_l143_143439


namespace relationship_between_a_b_c_d_l143_143713

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x)
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.sin x)

open Real

theorem relationship_between_a_b_c_d :
  ∀ (x : ℝ) (a b c d : ℝ),
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, f x ≤ a ∧ b ≤ f x) →
  (∀ x, g x ≤ c ∧ d ≤ g x) →
  a = sin 1 →
  b = -sin 1 →
  c = 1 →
  d = cos 1 →
  b < d ∧ d < a ∧ a < c := by
  sorry

end relationship_between_a_b_c_d_l143_143713


namespace prime_arithmetic_progression_difference_divisible_by_6_l143_143737

theorem prime_arithmetic_progression_difference_divisible_by_6
    (p d : ℕ) (h₀ : Prime p) (h₁ : Prime (p - d)) (h₂ : Prime (p + d))
    (p_neq_3 : p ≠ 3) :
    ∃ (k : ℕ), d = 6 * k := by
  sorry

end prime_arithmetic_progression_difference_divisible_by_6_l143_143737


namespace polynomial_roots_r_eq_18_l143_143957

theorem polynomial_roots_r_eq_18
  (a b c : ℂ) 
  (h_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C (5 : ℂ) * Polynomial.X^2 + Polynomial.C (2 : ℂ) * Polynomial.X + Polynomial.C (-8 : ℂ)) = {a, b, c}) 
  (h_ab_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C p * Polynomial.X^2 + Polynomial.C q * Polynomial.X + Polynomial.C r) = {2 * a + b, 2 * b + c, 2 * c + a}) :
  r = 18 := sorry

end polynomial_roots_r_eq_18_l143_143957


namespace smallest_interesting_number_l143_143846

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l143_143846


namespace smallest_factor_of_36_sum_4_l143_143538

theorem smallest_factor_of_36_sum_4 : ∃ a b c : ℤ, (a * b * c = 36) ∧ (a + b + c = 4) ∧ (a = -4 ∨ b = -4 ∨ c = -4) :=
by
  sorry

end smallest_factor_of_36_sum_4_l143_143538


namespace repeating_decimals_product_l143_143050

-- Definitions to represent the conditions
def repeating_decimal_03_as_frac : ℚ := 1 / 33
def repeating_decimal_36_as_frac : ℚ := 4 / 11

-- The statement to be proven
theorem repeating_decimals_product : (repeating_decimal_03_as_frac * repeating_decimal_36_as_frac) = (4 / 363) :=
by {
  sorry
}

end repeating_decimals_product_l143_143050


namespace fraction_increase_l143_143819

-- Define the problem conditions and the proof statement
theorem fraction_increase (m n : ℤ) (hnz : n ≠ 0) (hnnz : n ≠ -1) (h : m < n) :
  (m : ℚ) / n < (m + 1 : ℚ) / (n + 1) :=
by sorry

end fraction_increase_l143_143819


namespace triangle_perimeter_l143_143368

theorem triangle_perimeter (m : ℝ) (a b : ℝ) (h1 : 3 ^ 2 - 3 * (m + 1) + 2 * m = 0)
  (h2 : a ^ 2 - (m + 1) * a + 2 * m = 0)
  (h3 : b ^ 2 - (m + 1) * b + 2 * m = 0)
  (h4 : a = 3 ∨ b = 3)
  (h5 : a ≠ b ∨ a = b)
  (hAB : a ≠ b ∨ a = b) :
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ ≠ s₂ → s₁ + s₁ + s₂ = 10 ∨ s₁ + s₁ + s₂ = 11) ∨
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ = s₂ → b + b + a = 10 ∨ b + b + a = 11) := by
  sorry

end triangle_perimeter_l143_143368


namespace solve_for_x_l143_143785

theorem solve_for_x (x y : ℝ) (h1 : 9^y = x^12) (h2 : y = 6) : x = 3 :=
by
  sorry

end solve_for_x_l143_143785


namespace divisor_count_of_45_l143_143223

theorem divisor_count_of_45 : 
  ∃ (n : ℤ), n = 12 ∧ ∀ d : ℤ, d ∣ 45 → (d > 0 ∨ d < 0) := sorry

end divisor_count_of_45_l143_143223


namespace problem_l143_143862

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem problem : nabla (nabla 1 3) 2 = 67 :=
by
  sorry

end problem_l143_143862


namespace curved_surface_area_of_cone_l143_143429

noncomputable def slant_height : ℝ := 22
noncomputable def radius : ℝ := 7
noncomputable def pi : ℝ := Real.pi

theorem curved_surface_area_of_cone :
  abs (pi * radius * slant_height - 483.22) < 0.01 := 
by
  sorry

end curved_surface_area_of_cone_l143_143429


namespace num_terms_arithmetic_sequence_is_41_l143_143503

-- Definitions and conditions
def first_term : ℤ := 200
def common_difference : ℤ := -5
def last_term : ℤ := 0

-- Definition of the n-th term of arithmetic sequence
def nth_term (a : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a + (n - 1) * d

-- Statement to prove
theorem num_terms_arithmetic_sequence_is_41 : 
  ∃ n : ℕ, nth_term first_term common_difference n = 0 ∧ n = 41 :=
by 
  sorry

end num_terms_arithmetic_sequence_is_41_l143_143503


namespace find_r_x_l143_143302

open Nat

theorem find_r_x (r n : ℕ) (x : ℕ) (h_r_le_70 : r ≤ 70) (repr_x : x = (10 * r + 6) * (r ^ (2 * n) - 1) / (r ^ 2 - 1))
  (repr_x2 : x^2 = (r ^ (4 * n) - 1) / (r - 1)) :
  (r = 7 ∧ x = 26) :=
by
  sorry

end find_r_x_l143_143302


namespace line_circle_intersection_range_l143_143906

theorem line_circle_intersection_range (b : ℝ) :
    (2 - Real.sqrt 2) < b ∧ b < (2 + Real.sqrt 2) ↔
    ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ ((p1.1 - 2)^2 + p1.2^2 = 1) ∧ ((p2.1 - 2)^2 + p2.2^2 = 1) ∧ (p1.2 = p1.1 - b ∧ p2.2 = p2.1 - b) :=
by
  sorry

end line_circle_intersection_range_l143_143906


namespace ceil_sqrt_200_eq_15_l143_143638

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l143_143638


namespace least_significant_digit_base_4_189_l143_143351

theorem least_significant_digit_base_4_189 :
  ∃ d rems q, (189 = 4 * q + d) ∧ (d = 1) ∧ (list.reversed (Nat.digitList 4 189)).head = d :=
by
  sorry

end least_significant_digit_base_4_189_l143_143351


namespace n_is_square_l143_143739

theorem n_is_square (n m : ℕ) (h1 : 3 ≤ n) (h2 : m = (n * (n - 1)) / 2) (h3 : ∃ (cards : Finset ℕ), 
  (cards.card = n) ∧ (∀ i ∈ cards, i ∈ Finset.range (m + 1)) ∧ 
  (∀ (i j : ℕ) (hi : i ∈ cards) (hj : j ∈ cards), i ≠ j → 
    ((i + j) % m) ≠ ((i + j) % m))) : 
  ∃ k : ℕ, n = k * k := 
sorry

end n_is_square_l143_143739


namespace find_gamma_l143_143547

variable (γ δ : ℝ)

def directly_proportional (γ δ : ℝ) : Prop := ∃ c : ℝ, γ = c * δ

theorem find_gamma (h1 : directly_proportional γ δ) (h2 : γ = 5) (h3 : δ = -10) : δ = 25 → γ = -25 / 2 := by
  sorry

end find_gamma_l143_143547


namespace doubled_volume_l143_143616

theorem doubled_volume (V : ℕ) (h : V = 4) : 8 * V = 32 := by
  sorry

end doubled_volume_l143_143616


namespace largest_integer_lt_100_with_remainder_5_div_8_l143_143667

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l143_143667


namespace number_of_cookies_first_friend_took_l143_143113

-- Definitions of given conditions:
def initial_cookies : ℕ := 22
def eaten_by_Kristy : ℕ := 2
def given_to_brother : ℕ := 1
def taken_by_second_friend : ℕ := 5
def taken_by_third_friend : ℕ := 5
def cookies_left : ℕ := 6

noncomputable def cookies_after_Kristy_ate_and_gave_away : ℕ :=
  initial_cookies - eaten_by_Kristy - given_to_brother

noncomputable def cookies_after_second_and_third_friends : ℕ :=
  taken_by_second_friend + taken_by_third_friend

noncomputable def cookies_before_second_and_third_friends_took : ℕ :=
  cookies_left + cookies_after_second_and_third_friends

theorem number_of_cookies_first_friend_took :
  cookies_after_Kristy_ate_and_gave_away - cookies_before_second_and_third_friends_took = 3 := by
  sorry

end number_of_cookies_first_friend_took_l143_143113


namespace corner_contains_same_color_cells_l143_143308

theorem corner_contains_same_color_cells (colors : Finset (Fin 120)) :
  ∀ (coloring : Fin 2017 × Fin 2017 → Fin 120),
  ∃ (corner : Fin 2017 × Fin 2017 → Prop), 
    (∃ cell1 cell2, corner cell1 ∧ corner cell2 ∧ coloring cell1 = coloring cell2) := 
by 
  sorry

end corner_contains_same_color_cells_l143_143308


namespace binom_12_10_l143_143349

theorem binom_12_10 : nat.choose 12 10 = 66 := by
  sorry

end binom_12_10_l143_143349


namespace rank_matA_l143_143202

def matA : Matrix (Fin 4) (Fin 5) ℤ :=
  ![![5, 7, 12, 48, -14],
    ![9, 16, 24, 98, -31],
    ![14, 24, 25, 146, -45],
    ![11, 12, 24, 94, -25]]

theorem rank_matA : Matrix.rank matA = 3 :=
by
  sorry

end rank_matA_l143_143202


namespace find_y_l143_143228

theorem find_y (x y : ℝ) (h1 : x = 202) (h2 : x^3 * y - 4 * x^2 * y + 2 * x * y = 808080) : y = 1 / 10 := by
  sorry

end find_y_l143_143228


namespace earnings_in_total_l143_143007

-- Defining the conditions
def hourly_wage : ℝ := 12.50
def hours_per_week : ℝ := 40
def earnings_per_widget : ℝ := 0.16
def widgets_per_week : ℝ := 1250

-- Theorem statement
theorem earnings_in_total : 
  (hours_per_week * hourly_wage) + (widgets_per_week * earnings_per_widget) = 700 := 
by
  sorry

end earnings_in_total_l143_143007


namespace ceil_sqrt_200_eq_15_l143_143636

theorem ceil_sqrt_200_eq_15 (h1 : Real.sqrt 196 = 14) (h2 : Real.sqrt 225 = 15) (h3 : 196 < 200 ∧ 200 < 225) : 
  Real.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l143_143636


namespace largest_number_l143_143593

theorem largest_number :
  let A := (1:ℝ) / 2
      B := 37.5 / 100
      C := (7:ℝ) / 22
      D := (Real.pi) / 10
  in A > B ∧ A > C ∧ A > D := 
by
  let A : ℝ := (1:ℝ) / 2
  let B : ℝ := 37.5 / 100
  let C : ℝ := (7:ℝ) / 22
  let D : ℝ := (Real.pi) / 10
  sorry

end largest_number_l143_143593


namespace matrix_pow_four_l143_143335

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !!
  [ 2, -1,
    1,  1]

-- State the theorem with the final result
theorem matrix_pow_four :
  A ^ 4 = !!
  [ 0, -9,
    9, -9] :=
  sorry

end matrix_pow_four_l143_143335


namespace find_divisor_l143_143237

-- Condition Definitions
def dividend : ℕ := 725
def quotient : ℕ := 20
def remainder : ℕ := 5

-- Target Proof Statement
theorem find_divisor (divisor : ℕ) (h : dividend = divisor * quotient + remainder) : divisor = 36 := by
  sorry

end find_divisor_l143_143237


namespace unique_H_value_l143_143515

theorem unique_H_value :
  ∀ (T H R E F I V S : ℕ),
    T = 8 →
    E % 2 = 1 →
    E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ F ∧ E ≠ I ∧ E ≠ V ∧ E ≠ S ∧ 
    H ≠ T ∧ H ≠ R ∧ H ≠ F ∧ H ≠ I ∧ H ≠ V ∧ H ≠ S ∧
    F ≠ T ∧ F ≠ I ∧ F ≠ V ∧ F ≠ S ∧
    I ≠ T ∧ I ≠ V ∧ I ≠ S ∧
    V ≠ T ∧ V ≠ S ∧
    S ≠ T ∧
    (8 + 8) = 10 + F ∧
    (E + E) % 10 = 6 →
    H + H = 10 + 4 →
    H = 7 := 
sorry

end unique_H_value_l143_143515


namespace ratio_of_spots_to_wrinkles_l143_143517

-- Definitions
def E : ℕ := 3
def W : ℕ := 3 * E
def S : ℕ := E + W - 69

-- Theorem
theorem ratio_of_spots_to_wrinkles : S / W = 7 :=
by
  sorry

end ratio_of_spots_to_wrinkles_l143_143517


namespace cube_faces_edges_vertices_sum_l143_143152

theorem cube_faces_edges_vertices_sum :
  ∀ (F E V : ℕ), F = 6 → E = 12 → V = 8 → F + E + V = 26 :=
by
  intros F E V F_eq E_eq V_eq
  rw [F_eq, E_eq, V_eq]
  rfl

end cube_faces_edges_vertices_sum_l143_143152


namespace smallest_interesting_number_l143_143847

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l143_143847


namespace multiples_of_3_or_5_but_not_6_l143_143375

theorem multiples_of_3_or_5_but_not_6 (n : ℕ) (h1 : n ≤ 150) :
  (∃ m : ℕ, m ≤ 150 ∧ ((m % 3 = 0 ∨ m % 5 = 0) ∧ m % 6 ≠ 0)) ↔ n = 45 :=
by {
  sorry
}

end multiples_of_3_or_5_but_not_6_l143_143375


namespace find_k_l143_143896

theorem find_k (x y k : ℝ)
  (h1 : 3 * x + 2 * y = k + 1)
  (h2 : 2 * x + 3 * y = k)
  (h3 : x + y = 3) : k = 7 := sorry

end find_k_l143_143896


namespace jessica_marbles_62_l143_143246

-- Definitions based on conditions
def marbles_kurt (marbles_dennis : ℕ) : ℕ := marbles_dennis - 45
def marbles_laurie (marbles_kurt : ℕ) : ℕ := marbles_kurt + 12
def marbles_jessica (marbles_laurie : ℕ) : ℕ := marbles_laurie + 25

-- Given marbles for Dennis
def marbles_dennis : ℕ := 70

-- Proof statement: Prove that Jessica has 62 marbles given the conditions
theorem jessica_marbles_62 : marbles_jessica (marbles_laurie (marbles_kurt marbles_dennis)) = 62 := 
by
  sorry

end jessica_marbles_62_l143_143246


namespace least_sum_of_exponents_of_powers_of_2_l143_143380

theorem least_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ s : Finset ℕ, (∑ x in s, 2^x = n) ∧ (∀ t : Finset ℕ, (∑ x in t, 2^x = n) → s.sum id ≤ t.sum id) :=
sorry

end least_sum_of_exponents_of_powers_of_2_l143_143380


namespace solve_fraction_l143_143506

theorem solve_fraction (x : ℝ) (h : 2 / (x - 3) = 2) : x = 4 :=
by
  sorry

end solve_fraction_l143_143506


namespace find_triples_l143_143197

theorem find_triples : 
  { (a, b, k) : ℕ × ℕ × ℕ | 2^a * 3^b = k * (k + 1) } = 
  { (1, 0, 1), (1, 1, 2), (3, 2, 8), (2, 1, 3) } := 
by
  sorry

end find_triples_l143_143197


namespace amount_spent_on_drink_l143_143404

-- Definitions based on conditions provided
def initialAmount : ℝ := 9
def remainingAmount : ℝ := 6
def additionalSpending : ℝ := 1.25

-- Theorem to prove the amount spent on the drink
theorem amount_spent_on_drink : 
  initialAmount - remainingAmount - additionalSpending = 1.75 := 
by 
  sorry

end amount_spent_on_drink_l143_143404


namespace milo_running_distance_l143_143255

theorem milo_running_distance
  (run_speed skateboard_speed cory_speed : ℕ)
  (h1 : skateboard_speed = 2 * run_speed)
  (h2 : cory_speed = 2 * skateboard_speed)
  (h3 : cory_speed = 12) :
  run_speed * 2 = 6 :=
by
  sorry

end milo_running_distance_l143_143255


namespace largest_integer_less_than_hundred_with_remainder_five_l143_143671

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l143_143671


namespace range_of_a_l143_143395

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 4) ∧ (2 * x^2 - 9 * x + a < 0)) ↔ (a < 4) :=
by
  sorry

end range_of_a_l143_143395


namespace milo_run_distance_l143_143254

def cory_speed : ℝ := 12
def milo_roll_speed := cory_speed / 2
def milo_run_speed := milo_roll_speed / 2
def time_hours : ℝ := 2

theorem milo_run_distance : milo_run_speed * time_hours = 6 := 
by 
  /- The proof goes here -/
  sorry

end milo_run_distance_l143_143254


namespace expand_and_simplify_expression_l143_143357

variable {x y : ℝ} {i : ℂ}

-- Declare i as the imaginary unit satisfying i^2 = -1
axiom imaginary_unit : i^2 = -1

theorem expand_and_simplify_expression :
  (x + 3 + i * y) * (x + 3 - i * y) + (x - 2 + 2 * i * y) * (x - 2 - 2 * i * y)
  = 2 * x^2 + 2 * x + 13 - 5 * y^2 :=
by
  sorry

end expand_and_simplify_expression_l143_143357


namespace beth_score_l143_143098

-- Conditions
variables (B : ℕ)  -- Beth's points are some number.
def jan_points := 10 -- Jan scored 10 points.
def judy_points := 8 -- Judy scored 8 points.
def angel_points := 11 -- Angel scored 11 points.

-- First team has 3 more points than the second team
def first_team_points := B + jan_points
def second_team_points := judy_points + angel_points
def first_team_more_than_second := first_team_points = second_team_points + 3

-- Statement: Prove that B = 12
theorem beth_score : first_team_more_than_second → B = 12 :=
by
  -- Proof will be provided here
  sorry

end beth_score_l143_143098


namespace rectangle_perimeter_l143_143807

-- Conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_of_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def perimeter_of_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

-- Given conditions from the problem
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15
def width_of_rectangle : ℕ := 6

-- Main theorem
theorem rectangle_perimeter :
  is_right_triangle a b c →
  area_of_triangle a b = area_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle →
  perimeter_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle = 30 :=
by
  sorry

end rectangle_perimeter_l143_143807


namespace part1_solution_part2_solution_l143_143722

-- Define the inequality
def inequality (m x : ℝ) : Prop := (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0

-- Part (1): Prove the solution set for m = 0 is (-2, 1)
theorem part1_solution :
  (∀ x : ℝ, inequality 0 x → (-2 : ℝ) < x ∧ x < 1) := 
by
  sorry

-- Part (2): Prove the range of values for m such that the solution set is R
theorem part2_solution (m : ℝ) :
  (∀ x : ℝ, inequality m x) ↔ (1 ≤ m ∧ m < 9) := 
by
  sorry

end part1_solution_part2_solution_l143_143722


namespace tina_sales_ratio_l143_143112

theorem tina_sales_ratio (katya_sales ricky_sales t_sold_more : ℕ) 
  (h_katya : katya_sales = 8) 
  (h_ricky : ricky_sales = 9) 
  (h_tina_sold : t_sold_more = katya_sales + 26) 
  (h_tina_multiple : ∃ m : ℕ, t_sold_more = m * (katya_sales + ricky_sales)) :
  t_sold_more / (katya_sales + ricky_sales) = 2 := 
by 
  sorry

end tina_sales_ratio_l143_143112


namespace problem_solution_l143_143212

theorem problem_solution
  (x y : ℝ)
  (h1 : (x - y)^2 = 25)
  (h2 : x * y = -10) :
  x^2 + y^2 = 5 := sorry

end problem_solution_l143_143212


namespace leak_time_to_empty_tank_l143_143992

-- Define variables for the rates
variable (A L : ℝ)

-- Given conditions
def rate_pipe_A : Prop := A = 1 / 4
def combined_rate : Prop := A - L = 1 / 6

-- Theorem statement: The time it takes for the leak to empty the tank
theorem leak_time_to_empty_tank (A L : ℝ) (h1 : rate_pipe_A A) (h2 : combined_rate A L) : 1 / L = 12 :=
by 
  sorry

end leak_time_to_empty_tank_l143_143992


namespace Rachel_books_total_l143_143534

-- Define the conditions
def mystery_shelves := 6
def picture_shelves := 2
def scifi_shelves := 3
def bio_shelves := 4
def books_per_shelf := 9

-- Define the total number of books
def total_books := 
  mystery_shelves * books_per_shelf + 
  picture_shelves * books_per_shelf + 
  scifi_shelves * books_per_shelf + 
  bio_shelves * books_per_shelf

-- Statement of the problem
theorem Rachel_books_total : total_books = 135 := 
by
  -- Proof can be added here
  sorry

end Rachel_books_total_l143_143534


namespace smallest_interesting_number_l143_143835

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l143_143835


namespace find_f_five_l143_143554

-- Define the function f and the conditions as given in the problem.
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)
variable (h₁ : ∀ x y : ℝ, f (x - y) = f x * g y)
variable (h₂ : ∀ y : ℝ, g y = Real.exp (-y))
variable (h₃ : ∀ x : ℝ, f x ≠ 0)

-- Goal: Prove that f(5) = e^{2.5}.
theorem find_f_five : f 5 = Real.exp 2.5 :=
by
  -- Proof is omitted as per the instructions.
  sorry

end find_f_five_l143_143554


namespace correct_calculation_l143_143591

theorem correct_calculation (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 :=
by sorry

end correct_calculation_l143_143591


namespace geom_seq_sum_l143_143887

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geometric : ∀ n, a (n + 1) = a n * r)
variable (h_pos : ∀ n, a n > 0)
variable (h_equation : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

theorem geom_seq_sum : a 3 + a 5 = 5 :=
by sorry

end geom_seq_sum_l143_143887


namespace a_n_less_than_inverse_n_minus_1_l143_143406

theorem a_n_less_than_inverse_n_minus_1 
  (n : ℕ) (h1 : 2 ≤ n) 
  (a : ℕ → ℝ) 
  (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ n-1 → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) - a (k+1)) 
  (h3 : ∀ m : ℕ, m ≤ n → 0 < a m) : 
  a n < 1 / (n - 1) :=
sorry

end a_n_less_than_inverse_n_minus_1_l143_143406


namespace max_ratio_a_c_over_b_d_l143_143066

-- Given conditions as Lean definitions
variables {a b c d : ℝ}
variable (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ 0)
variable (h2 : (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)^2 = 3 / 8)

-- The statement to prove the maximum value of the given expression
theorem max_ratio_a_c_over_b_d : ∃ t : ℝ, t = (a + c) / (b + d) ∧ t ≤ 3 :=
by {
  -- The proof of this theorem is omitted.
  sorry
}

end max_ratio_a_c_over_b_d_l143_143066


namespace find_two_numbers_l143_143975

theorem find_two_numbers :
  ∃ (x y : ℝ), 
  (2 * (x + y) = x^2 - y^2 ∧ 2 * (x + y) = (x * y) / 4 - 56) ∧ 
  ((x = 26 ∧ y = 24) ∨ (x = -8 ∧ y = -10)) := 
sorry

end find_two_numbers_l143_143975


namespace appropriate_selling_price_l143_143854

-- Define the given conditions
def cost_per_kg : ℝ := 40
def base_price : ℝ := 50
def base_sales_volume : ℝ := 500
def sales_decrease_per_yuan : ℝ := 10
def available_capital : ℝ := 10000
def desired_profit : ℝ := 8000

-- Define the sales volume function dependent on selling price x
def sales_volume (x : ℝ) : ℝ := base_sales_volume - (x - base_price) * sales_decrease_per_yuan

-- Define the profit function dependent on selling price x
def profit (x : ℝ) : ℝ := (x - cost_per_kg) * sales_volume x

-- Prove that the appropriate selling price is 80 yuan
theorem appropriate_selling_price : 
  ∃ x : ℝ, profit x = desired_profit ∧ x = 80 :=
by
  sorry

end appropriate_selling_price_l143_143854


namespace white_red_balls_l143_143512

theorem white_red_balls (w r : ℕ) 
  (h1 : 3 * w = 5 * r)
  (h2 : w + 15 + r = 50) : 
  r = 12 :=
by
  sorry

end white_red_balls_l143_143512


namespace total_oranges_l143_143261

def oranges_from_first_tree : Nat := 80
def oranges_from_second_tree : Nat := 60
def oranges_from_third_tree : Nat := 120

theorem total_oranges : oranges_from_first_tree + oranges_from_second_tree + oranges_from_third_tree = 260 :=
by
  sorry

end total_oranges_l143_143261


namespace prove_min_period_and_max_value_l143_143135

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem prove_min_period_and_max_value :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ y : ℝ, y ≤ f y) :=
by
  -- Proof goes here
  sorry

end prove_min_period_and_max_value_l143_143135


namespace largest_integer_less_than_100_with_remainder_5_l143_143676

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143676


namespace cucumber_to_tomato_ratio_l143_143327

variable (total_rows : ℕ) (space_per_row_tomato : ℕ) (tomatoes_per_plant : ℕ) (total_tomatoes : ℕ)

/-- Aubrey's Garden -/
theorem cucumber_to_tomato_ratio (total_rows_eq : total_rows = 15)
  (space_per_row_tomato_eq : space_per_row_tomato = 8)
  (tomatoes_per_plant_eq : tomatoes_per_plant = 3)
  (total_tomatoes_eq : total_tomatoes = 120) :
  let total_tomato_plants := total_tomatoes / tomatoes_per_plant
  let rows_tomato := total_tomato_plants / space_per_row_tomato
  let rows_cucumber := total_rows - rows_tomato
  (2 * rows_tomato = rows_cucumber)
:=
by
  sorry

end cucumber_to_tomato_ratio_l143_143327


namespace no_solution_for_x_l143_143644

open Real

theorem no_solution_for_x (m : ℝ) : ¬ ∃ x : ℝ, (sin (3 * x) * cos (↑60 - x) + 1) / (sin (↑60 - 7 * x) - cos (↑30 + x) + m) = 0 :=
by
  sorry

end no_solution_for_x_l143_143644


namespace div_sub_mult_exp_eq_l143_143977

-- Lean 4 statement for the mathematical proof problem
theorem div_sub_mult_exp_eq :
  8 / 4 - 3 - 9 + 3 * 7 - 2^2 = 7 := 
sorry

end div_sub_mult_exp_eq_l143_143977


namespace dice_probability_l143_143808

def prob_at_least_one_one : ℚ :=
  let total_outcomes := 36
  let no_1_outcomes := 25
  let favorable_outcomes := total_outcomes - no_1_outcomes
  let probability := favorable_outcomes / total_outcomes
  probability

theorem dice_probability :
  prob_at_least_one_one = 11 / 36 :=
by
  sorry

end dice_probability_l143_143808


namespace evaluate_expression_l143_143480

theorem evaluate_expression :
  let a := 12
  let b := 14
  let c := 18
  (144 * ((1:ℝ)/b - (1:ℝ)/c) + 196 * ((1:ℝ)/c - (1:ℝ)/a) + 324 * ((1:ℝ)/a - (1:ℝ)/b)) /
  (a * ((1:ℝ)/b - (1:ℝ)/c) + b * ((1:ℝ)/c - (1:ℝ)/a) + c * ((1:ℝ)/a - (1:ℝ)/b)) = a + b + c := by
  sorry

end evaluate_expression_l143_143480


namespace min_value_frac_sin_cos_l143_143214

open Real

theorem min_value_frac_sin_cos (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ m : ℝ, (∀ x : ℝ, x = (1 / (sin α)^2 + 3 / (cos α)^2) → x ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
by
  have h_sin_cos : sin α ≠ 0 ∧ cos α ≠ 0 := sorry -- This is an auxiliary lemma in the process, a proof is required.
  sorry

end min_value_frac_sin_cos_l143_143214


namespace doubled_container_volume_l143_143613

theorem doubled_container_volume (original_volume : ℕ) (factor : ℕ) 
  (h1 : original_volume = 4) (h2 : factor = 8) : original_volume * factor = 32 :=
by 
  rw [h1, h2]
  norm_num

end doubled_container_volume_l143_143613


namespace no_alpha_exists_l143_143355

theorem no_alpha_exists (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) :
  ¬(∃ (a : ℕ → ℝ), (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, 1 + a (n+1) ≤ a n + (α / n.succ) * a n)) :=
by
  sorry

end no_alpha_exists_l143_143355


namespace larger_acute_angle_right_triangle_l143_143092

theorem larger_acute_angle_right_triangle (x : ℝ) (h1 : x > 0) (h2 : x + 5 * x = 90) : 5 * x = 75 := by
  sorry

end larger_acute_angle_right_triangle_l143_143092


namespace smallest_interesting_number_smallest_interesting_number_1800_l143_143845
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l143_143845


namespace arc_length_of_circle_l143_143139

theorem arc_length_of_circle (r θ : ℝ) (h1 : r = 2) (h2 : θ = 5 * Real.pi / 3) : (θ * r) = 10 * Real.pi / 3 :=
by
  rw [h1, h2]
  -- subsequent steps would go here 
  sorry

end arc_length_of_circle_l143_143139


namespace largest_integer_less_than_100_with_remainder_5_l143_143679

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143679


namespace floor_function_solution_l143_143543

def floor_eq_x_solutions : Prop :=
  ∀ x : ℤ, (⌊(x : ℝ) / 2⌋ + ⌊(x : ℝ) / 4⌋ = x) ↔ x = 0 ∨ x = -3 ∨ x = -2 ∨ x = -5

theorem floor_function_solution: floor_eq_x_solutions :=
by
  intro x
  sorry

end floor_function_solution_l143_143543


namespace equal_poly_terms_l143_143280

theorem equal_poly_terms (p q : ℝ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : 
  (7 * p^6 * q = 21 * p^5 * q^2) -> p = 3 / 4 :=
by
  sorry

end equal_poly_terms_l143_143280


namespace largest_n_digit_number_divisible_by_61_correct_l143_143285

def largest_n_digit_number (n : ℕ) : ℕ :=
10^n - 1

def largest_n_digit_number_divisible_by_61 (n : ℕ) : ℕ :=
largest_n_digit_number n - (largest_n_digit_number n % 61)

theorem largest_n_digit_number_divisible_by_61_correct (n : ℕ) :
  ∃ k : ℕ, largest_n_digit_number_divisible_by_61 n = 61 * k :=
by
  sorry

end largest_n_digit_number_divisible_by_61_correct_l143_143285


namespace matrix_A_to_power_4_l143_143339

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l143_143339


namespace ball_reaches_less_than_5_l143_143447

noncomputable def height_after_bounces (initial_height : ℕ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial_height * (ratio ^ bounces)

theorem ball_reaches_less_than_5 (initial_height : ℕ) (ratio : ℝ) (k : ℕ) (target_height : ℝ) (stop_height : ℝ) 
  (h_initial : initial_height = 500) (h_ratio : ratio = 0.6) (h_target : target_height = 5) (h_stop : stop_height = 0.1) :
  ∃ n, height_after_bounces initial_height ratio n < target_height ∧ 500 * (0.6 ^ 17) < stop_height := by
  sorry

end ball_reaches_less_than_5_l143_143447


namespace a_in_M_sufficient_not_necessary_l143_143072

-- Defining the sets M and N
def M := {x : ℝ | x^2 < 3 * x}
def N := {x : ℝ | abs (x - 1) < 2}

-- Stating that a ∈ M is a sufficient but not necessary condition for a ∈ N
theorem a_in_M_sufficient_not_necessary (a : ℝ) (h : a ∈ M) : a ∈ N :=
by sorry

end a_in_M_sufficient_not_necessary_l143_143072


namespace maximum_dn_l143_143044

-- Definitions of a_n and d_n based on the problem statement
def a (n : ℕ) : ℕ := 150 + (n + 1)^2
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Statement of the theorem
theorem maximum_dn : ∃ M, M = 2 ∧ ∀ n, d n ≤ M :=
by
  -- proof should be written here
  sorry

end maximum_dn_l143_143044


namespace packing_objects_in_boxes_l143_143926

theorem packing_objects_in_boxes 
  (n k : ℕ) (n_pos : 0 < n) (k_pos : 0 < k) 
  (objects : Fin (n * k) → Fin k) 
  (boxes : Fin k → Fin n → Fin k) :
  ∃ (pack : Fin (n * k) → Fin k), 
    (∀ i, ∃ c1 c2, 
      ∀ j, pack i = pack j → 
      (objects i = c1 ∨ objects i = c2 ∧
      objects j = c1 ∨ objects j = c2)) := 
sorry

end packing_objects_in_boxes_l143_143926


namespace line_through_point_bisects_chord_l143_143101

theorem line_through_point_bisects_chord 
  (x y : ℝ) 
  (h_parabola : y^2 = 16 * x) 
  (h_point : 8 * 2 - 1 - 15 = 0) :
  8 * x - y - 15 = 0 :=
by
  sorry

end line_through_point_bisects_chord_l143_143101


namespace not_all_ten_on_boundary_of_same_square_l143_143398

open Function

variable (points : Fin 10 → ℝ × ℝ)

def four_points_on_square (A B C D : ℝ × ℝ) : Prop :=
  -- Define your own predicate to check if 4 points A, B, C, D are on the boundary of some square
  sorry 

theorem not_all_ten_on_boundary_of_same_square :
  (∀ A B C D : Fin 10, four_points_on_square (points A) (points B) (points C) (points D)) →
  ¬ (∃ square : ℝ × ℝ → Prop, ∀ i : Fin 10, square (points i)) :=
by
  intro h
  sorry

end not_all_ten_on_boundary_of_same_square_l143_143398


namespace proper_polygons_m_lines_l143_143529

noncomputable def smallest_m := 2

theorem proper_polygons_m_lines (P : Finset (Set (ℝ × ℝ)))
  (properly_placed : ∀ (p1 p2 : Set (ℝ × ℝ)), p1 ∈ P → p2 ∈ P → ∃ l : Set (ℝ × ℝ), (0, 0) ∈ l ∧ ∀ (p : Set (ℝ × ℝ)), p ∈ P → ¬Disjoint l p) :
  ∃ (m : ℕ), m = smallest_m ∧ ∀ (lines : Finset (Set (ℝ × ℝ))), 
    (∀ l ∈ lines, (0, 0) ∈ l) → lines.card = m → ∀ p ∈ P, ∃ l ∈ lines, ¬Disjoint l p := sorry

end proper_polygons_m_lines_l143_143529


namespace largest_integer_solution_l143_143422

theorem largest_integer_solution (x : ℤ) : 
  (x - 3 * (x - 2) ≥ 4) → (2 * x + 1 < x - 1) → (x = -3) :=
by
  sorry

end largest_integer_solution_l143_143422


namespace conditional_without_else_l143_143986

def if_then_else_statement (s: String) : Prop :=
  (s = "IF—THEN" ∨ s = "IF—THEN—ELSE")

theorem conditional_without_else : if_then_else_statement "IF—THEN" :=
  sorry

end conditional_without_else_l143_143986


namespace wang_hao_not_last_l143_143396

theorem wang_hao_not_last (total_players : ℕ) (players_to_choose : ℕ) 
  (wang_hao : ℕ) (ways_to_choose_if_not_last : ℕ) : 
  total_players = 6 ∧ players_to_choose = 3 → 
  ways_to_choose_if_not_last = 100 := 
by
  sorry

end wang_hao_not_last_l143_143396


namespace smallest_factor_of_36_l143_143536

theorem smallest_factor_of_36 :
  ∃ a b c : ℤ, a * b * c = 36 ∧ a + b + c = 4 ∧ min (min a b) c = -4 :=
by
  sorry

end smallest_factor_of_36_l143_143536


namespace total_oranges_l143_143262

def oranges_from_first_tree : Nat := 80
def oranges_from_second_tree : Nat := 60
def oranges_from_third_tree : Nat := 120

theorem total_oranges : oranges_from_first_tree + oranges_from_second_tree + oranges_from_third_tree = 260 :=
by
  sorry

end total_oranges_l143_143262


namespace largest_integer_less_than_100_div_8_rem_5_l143_143660

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l143_143660


namespace slope_of_line_determined_by_solutions_l143_143818

theorem slope_of_line_determined_by_solutions (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : 3 / x₁ +  4 / y₁ = 0)
  (h₂ : 3 / x₂ + 4 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4 / 3 :=
sorry

end slope_of_line_determined_by_solutions_l143_143818


namespace number_of_black_squares_in_56th_row_l143_143310

def total_squares (n : Nat) : Nat := 3 + 2 * (n - 1)

def black_squares (n : Nat) : Nat :=
  if total_squares n % 2 == 1 then
    (total_squares n - 1) / 2
  else
    total_squares n / 2

theorem number_of_black_squares_in_56th_row :
  black_squares 56 = 56 :=
by
  sorry

end number_of_black_squares_in_56th_row_l143_143310


namespace Larry_wins_probability_l143_143919

noncomputable def probability_Larry_wins (p_larry: ℚ) (p_paul: ℚ): ℚ :=
  let q_larry := 1 - p_larry
  let q_paul := 1 - p_paul
  p_larry / (1 - q_larry * q_paul)

theorem Larry_wins_probability:
  probability_Larry_wins (1/3 : ℚ) (1/2 : ℚ) = (2/5 : ℚ) :=
by {
  sorry
}

end Larry_wins_probability_l143_143919


namespace cost_of_projector_and_whiteboard_l143_143328

variable (x : ℝ)

def cost_of_projector : ℝ := x
def cost_of_whiteboard : ℝ := x + 4000
def total_cost_eq_44000 : Prop := 4 * (x + 4000) + 3 * x = 44000

theorem cost_of_projector_and_whiteboard 
  (h : total_cost_eq_44000 x) : 
  cost_of_projector x = 4000 ∧ cost_of_whiteboard x = 8000 :=
by
  sorry

end cost_of_projector_and_whiteboard_l143_143328


namespace assisted_work_time_l143_143599

theorem assisted_work_time (a b c : ℝ) (ha : a = 1 / 11) (hb : b = 1 / 20) (hc : c = 1 / 55) :
  (1 / ((a + b) + (a + c) / 2)) = 8 :=
by
  sorry

end assisted_work_time_l143_143599


namespace sequence_a_n_l143_143089

theorem sequence_a_n {n : ℕ} (S : ℕ → ℚ) (a : ℕ → ℚ)
  (hS : ∀ n, S n = (2/3 : ℚ) * n^2 - (1/3 : ℚ) * n)
  (ha : ∀ n, a n = if n = 1 then S n else S n - S (n - 1)) :
  ∀ n, a n = (4/3 : ℚ) * n - 1 := 
by
  sorry

end sequence_a_n_l143_143089


namespace transformation_identity_l143_143279

theorem transformation_identity (a b : ℝ) 
    (h1 : ∃ a b : ℝ, ∀ x y : ℝ, (y, -x) = (-7, 3) → (x, y) = (3, 7))
    (h2 : ∃ a b : ℝ, ∀ c d : ℝ, (d, c) = (3, -7) → (c, d) = (-7, 3)) :
    b - a = 4 :=
by
    sorry

end transformation_identity_l143_143279


namespace doubled_container_volume_l143_143612

theorem doubled_container_volume (v : ℝ) (h₁ : v = 4) (h₂ : ∀ l w h : ℝ, v = l * w * h) : 8 * v = 32 := 
by
  -- The proof will go here, this is just the statement
  sorry

end doubled_container_volume_l143_143612


namespace gunther_typing_l143_143079

theorem gunther_typing :
  ∀ (wpm : ℚ), (wpm = 160 / 3) → 480 * wpm = 25598 :=
by
  intros wpm h
  sorry

end gunther_typing_l143_143079


namespace intersection_A_complement_B_l143_143213

-- Definition of real numbers
def R := ℝ

-- Definitions of sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x^2 - x - 2 > 0}

-- Definition of the complement of B in R
def B_complement := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- The final statement we need to prove
theorem intersection_A_complement_B :
  A ∩ B_complement = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_A_complement_B_l143_143213


namespace train_crossing_time_l143_143009

noncomputable def time_to_cross_bridge (l_train : ℕ) (v_train_kmh : ℕ) (l_bridge : ℕ) : ℚ :=
  let total_distance := l_train + l_bridge
  let v_train_ms := (v_train_kmh * 1000 : ℚ) / 3600
  total_distance / v_train_ms

theorem train_crossing_time :
  time_to_cross_bridge 110 72 136 = 12.3 := 
by
  sorry

end train_crossing_time_l143_143009


namespace smallest_nonneg_int_mod_15_l143_143052

theorem smallest_nonneg_int_mod_15 :
  ∃ x : ℕ, x + 7263 ≡ 3507 [MOD 15] ∧ ∀ y : ℕ, y + 7263 ≡ 3507 [MOD 15] → x ≤ y :=
by
  sorry

end smallest_nonneg_int_mod_15_l143_143052


namespace intersection_of_M_and_N_l143_143894

-- Define the sets M and N with the given conditions
def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem that the intersection of M and N is as described
theorem intersection_of_M_and_N : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 1} :=
by
  -- the proof will go here
  sorry

end intersection_of_M_and_N_l143_143894


namespace average_salary_of_all_employees_l143_143095

theorem average_salary_of_all_employees 
    (avg_salary_officers : ℝ)
    (avg_salary_non_officers : ℝ)
    (num_officers : ℕ)
    (num_non_officers : ℕ)
    (h1 : avg_salary_officers = 450)
    (h2 : avg_salary_non_officers = 110)
    (h3 : num_officers = 15)
    (h4 : num_non_officers = 495) :
    (avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers)
    / (num_officers + num_non_officers) = 120 := by
  sorry

end average_salary_of_all_employees_l143_143095


namespace cos_arcsin_eq_l143_143469

theorem cos_arcsin_eq : ∀ (x : ℝ), (x = 8 / 17) → (cos (arcsin x) = 15 / 17) := by
  intro x hx
  rw [hx]
  -- Here you can add any required steps to complete the proof.
  sorry

end cos_arcsin_eq_l143_143469


namespace bears_in_shipment_l143_143856

theorem bears_in_shipment (initial_bears shipped_bears bears_per_shelf shelves_used : ℕ) 
  (h1 : initial_bears = 4) 
  (h2 : bears_per_shelf = 7) 
  (h3 : shelves_used = 2) 
  (total_bears_on_shelves : ℕ) 
  (h4 : total_bears_on_shelves = shelves_used * bears_per_shelf) 
  (total_bears_after_shipment : ℕ) 
  (h5 : total_bears_after_shipment = total_bears_on_shelves) 
  : shipped_bears = total_bears_on_shelves - initial_bears := 
sorry

end bears_in_shipment_l143_143856


namespace third_angle_in_triangle_sum_of_angles_in_triangle_l143_143912

theorem third_angle_in_triangle (a b : ℝ) (h₁ : a = 50) (h₂ : b = 80) : 180 - a - b = 50 :=
by
  rw [h₁, h₂]
  norm_num

-- Adding this to demonstrate the constraint of the problem: Sum of angles in a triangle is 180°
theorem sum_of_angles_in_triangle (a b c : ℝ) (h₁: a + b + c = 180) : true :=
by
  trivial

end third_angle_in_triangle_sum_of_angles_in_triangle_l143_143912


namespace fraction_of_third_is_eighth_l143_143583

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_third_is_eighth_l143_143583


namespace estimate_greater_than_exact_l143_143354

namespace NasreenRounding

variables (a b c d a' b' c' d' : ℕ)

-- Conditions: a, b, c, and d are large positive integers.
-- Definitions for rounding up and down
def round_up (n : ℕ) : ℕ := n + 1  -- Simplified model for rounding up
def round_down (n : ℕ) : ℕ := n - 1  -- Simplified model for rounding down

-- Conditions: a', b', c', and d' are the rounded values of a, b, c, and d respectively.
variable (h_round_a_up : a' = round_up a)
variable (h_round_b_down : b' = round_down b)
variable (h_round_c_down : c' = round_down c)
variable (h_round_d_down : d' = round_down d)

-- Question: Show that the estimate is greater than the original
theorem estimate_greater_than_exact :
  (a' / b' - c' * d') > (a / b - c * d) :=
sorry

end NasreenRounding

end estimate_greater_than_exact_l143_143354


namespace minoxidil_percentage_l143_143315

-- Define the conditions
variable (x : ℝ) -- percentage of Minoxidil in the solution to add
def pharmacist_scenario (x : ℝ) : Prop :=
  let amt_2_percent_solution := 70 -- 70 ml of 2% solution
  let percent_in_2_percent := 0.02
  let amt_of_2_percent := percent_in_2_percent * amt_2_percent_solution
  let amt_added_solution := 35 -- 35 ml of solution to add
  let total_volume := amt_2_percent_solution + amt_added_solution -- 105 ml in total
  let desired_percent := 0.03
  let desired_amt := desired_percent * total_volume
  amt_of_2_percent + (x / 100) * amt_added_solution = desired_amt

-- Define the proof problem statement
theorem minoxidil_percentage : pharmacist_scenario 5 := by
  -- Proof goes here
  sorry

end minoxidil_percentage_l143_143315


namespace least_sum_exponents_of_520_l143_143381

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end least_sum_exponents_of_520_l143_143381


namespace largest_int_less_than_100_with_remainder_5_l143_143694

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l143_143694


namespace correct_option_B_l143_143161

theorem correct_option_B (a : ℤ) : (2 * a) ^ 3 = 8 * a ^ 3 :=
by
  sorry

end correct_option_B_l143_143161


namespace divisible_12_or_36_l143_143533

theorem divisible_12_or_36 (x : ℕ) (n : ℕ) (h1 : Nat.Prime x) (h2 : 3 < x) (h3 : x = 3 * n + 1 ∨ x = 3 * n - 1) :
  12 ∣ (x^6 - x^3 - x^2 + x) ∨ 36 ∣ (x^6 - x^3 - x^2 + x) := 
by
  sorry

end divisible_12_or_36_l143_143533


namespace largest_integer_less_than_hundred_with_remainder_five_l143_143673

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l143_143673


namespace ratio_of_volumes_l143_143292
-- Import the necessary library

-- Define the edge lengths of the cubes
def small_cube_edge : ℕ := 4
def large_cube_edge : ℕ := 24

-- Define the volumes of the cubes
def volume (edge : ℕ) : ℕ := edge ^ 3

-- Define the volumes
def V_small := volume small_cube_edge
def V_large := volume large_cube_edge

-- State the main theorem
theorem ratio_of_volumes : V_small / V_large = 1 / 216 := 
by 
  -- we skip the proof here
  sorry

end ratio_of_volumes_l143_143292


namespace solve_equations_l143_143421

-- Prove that the solutions to the given equations are correct.
theorem solve_equations :
  (∀ x : ℝ, (x * (x - 4) = 2 * x - 8) ↔ (x = 4 ∨ x = 2)) ∧
  (∀ x : ℝ, ((2 * x) / (2 * x - 3) - (4 / (2 * x + 3)) = 1) ↔ (x = 10.5)) :=
by
  sorry

end solve_equations_l143_143421


namespace n_is_square_if_m_even_l143_143740

theorem n_is_square_if_m_even
  (n : ℕ)
  (h1 : n ≥ 3)
  (m : ℕ)
  (h2 : m = (1 / 2) * n * (n - 1))
  (h3 : ∀ i j : ℕ, i ≠ j → (a_i + a_j) % m ≠ (a_j + a_k) % m)
  (h4 : even m) :
  ∃ k : ℕ, n = k * k := sorry

end n_is_square_if_m_even_l143_143740


namespace original_profit_percentage_l143_143187

theorem original_profit_percentage
  (P SP : ℝ)
  (h1 : SP = 549.9999999999995)
  (h2 : SP = P * (1 + x / 100))
  (h3 : 0.9 * P * 1.3 = SP + 35) :
  x = 10 := 
sorry

end original_profit_percentage_l143_143187


namespace fractions_proper_or_improper_l143_143058

theorem fractions_proper_or_improper : 
  ∀ (a b : ℚ), (∃ p q : ℚ, a = p / q ∧ p < q) ∨ (∃ r s : ℚ, a = r / s ∧ r ≥ s) :=
by 
  sorry

end fractions_proper_or_improper_l143_143058


namespace x_coordinate_of_first_point_l143_143400

theorem x_coordinate_of_first_point (m n : ℝ) :
  (m = 2 * n + 3) ↔ (∃ (p1 p2 : ℝ × ℝ), p1 = (m, n) ∧ p2 = (m + 2, n + 1) ∧ 
    (p1.1 = 2 * p1.2 + 3) ∧ (p2.1 = 2 * p2.2 + 3)) :=
by
  sorry

end x_coordinate_of_first_point_l143_143400


namespace mod_11_residue_l143_143001

theorem mod_11_residue :
  (312 ≡ 4 [MOD 11]) ∧
  (47 ≡ 3 [MOD 11]) ∧
  (154 ≡ 0 [MOD 11]) ∧
  (22 ≡ 0 [MOD 11]) →
  (312 + 6 * 47 + 8 * 154 + 5 * 22 ≡ 0 [MOD 11]) :=
by
  intros h
  sorry

end mod_11_residue_l143_143001


namespace farmer_cows_after_selling_l143_143024

theorem farmer_cows_after_selling
  (initial_cows : ℕ) (new_cows : ℕ) (quarter_factor : ℕ)
  (h_initial : initial_cows = 51)
  (h_new : new_cows = 5)
  (h_quarter : quarter_factor = 4) :
  initial_cows + new_cows - (initial_cows + new_cows) / quarter_factor = 42 :=
by
  sorry

end farmer_cows_after_selling_l143_143024


namespace part1_part2_l143_143076

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Part (1): Prove range for m
theorem part1 (m : ℝ) : (∃ x : ℝ, quadratic 1 (-5) m x = 0) ↔ m ≤ 25 / 4 := sorry

-- Part (2): Prove value of m given conditions on roots
theorem part2 (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : 3 * x1 - 2 * x2 = 5) : 
  m = x1 * x2 → m = 6 := sorry

end part1_part2_l143_143076


namespace largest_4_digit_number_divisible_by_12_l143_143605

theorem largest_4_digit_number_divisible_by_12 : ∃ (n : ℕ), 1000 ≤ n ∧ n ≤ 9999 ∧ n % 12 = 0 ∧ ∀ m, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 12 = 0 → m ≤ n := 
sorry

end largest_4_digit_number_divisible_by_12_l143_143605


namespace tea_bags_number_l143_143951

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l143_143951


namespace arithmetic_sequence_l143_143744

theorem arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) (h : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 :=
sorry

end arithmetic_sequence_l143_143744


namespace frequency_even_numbers_facing_up_l143_143913

theorem frequency_even_numbers_facing_up (rolls : ℕ) (event_occurrences : ℕ) (h_rolls : rolls = 100) (h_event : event_occurrences = 47) : (event_occurrences / (rolls : ℝ)) = 0.47 :=
by
  sorry

end frequency_even_numbers_facing_up_l143_143913


namespace carmen_sprigs_left_l143_143860

-- Definitions based on conditions
def initial_sprigs : ℕ := 25
def whole_sprigs_used : ℕ := 8
def half_sprigs_plates : ℕ := 12
def half_sprigs_total_used : ℕ := half_sprigs_plates / 2

-- Total sprigs used
def total_sprigs_used : ℕ := whole_sprigs_used + half_sprigs_total_used

-- Leftover sprigs computation
def sprigs_left : ℕ := initial_sprigs - total_sprigs_used

-- Statement to prove
theorem carmen_sprigs_left : sprigs_left = 11 :=
by
  sorry

end carmen_sprigs_left_l143_143860


namespace length_of_CD_l143_143532

theorem length_of_CD (C D R S : ℝ) 
  (h1 : R = C + 3/8 * (D - C))
  (h2 : S = C + 4/11 * (D - C))
  (h3 : |S - R| = 3) :
  D - C = 264 := 
sorry

end length_of_CD_l143_143532


namespace solve_equation_l143_143954

theorem solve_equation (y : ℝ) (z : ℝ) (hz : z = y^(1/3)) :
  (6 * y^(1/3) - 3 * y^(4/3) = 12 + y^(1/3) + y) ↔ (3 * z^4 + z^3 - 5 * z + 12 = 0) :=
by sorry

end solve_equation_l143_143954


namespace sanctuary_feeding_ways_l143_143853

/-- A sanctuary houses six different pairs of animals, each pair consisting of a male and female.
  The caretaker must feed the animals alternately by gender, meaning no two animals of the same gender 
  can be fed consecutively. Given the additional constraint that the male giraffe cannot be fed 
  immediately before the female giraffe and that the feeding starts with the male lion, 
  there are exactly 7200 valid ways to complete the feeding. -/
theorem sanctuary_feeding_ways : 
  ∃ ways : ℕ, ways = 7200 :=
by sorry

end sanctuary_feeding_ways_l143_143853


namespace levi_additional_baskets_to_score_l143_143253

def levi_scored_initial := 8
def brother_scored_initial := 12
def brother_likely_to_score := 3
def levi_goal_margin := 5

theorem levi_additional_baskets_to_score : 
  levi_scored_initial + 12 >= brother_scored_initial + brother_likely_to_score + levi_goal_margin :=
by
  sorry

end levi_additional_baskets_to_score_l143_143253


namespace age_difference_l143_143971

theorem age_difference (h b m : ℕ) (ratio : h = 4 * m ∧ b = 3 * m ∧ 4 * m + 3 * m + 7 * m = 126) : h - b = 9 :=
by
  -- proof will be filled here
  sorry

end age_difference_l143_143971


namespace value_of_x_l143_143365

theorem value_of_x (x y : ℕ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
sorry

end value_of_x_l143_143365


namespace podium_height_l143_143576

theorem podium_height (l w h : ℝ) (r s : ℝ) (H1 : r = l + h - w) (H2 : s = w + h - l) 
  (Hr : r = 40) (Hs : s = 34) : h = 37 :=
by
  sorry

end podium_height_l143_143576


namespace min_value_of_g_inequality_f_l143_143719

def f (x m : ℝ) : ℝ := abs (x - m)
def g (x m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem min_value_of_g (m : ℝ) (hm : m > 0) (h : ∀ x, g x m ≥ -1) : m = 1 :=
sorry

theorem inequality_f {m a b : ℝ} (hm : m > 0) (ha : abs a < m) (hb : abs b < m) (h0 : a ≠ 0) :
  f (a * b) m > abs a * f (b / a) m :=
sorry

end min_value_of_g_inequality_f_l143_143719


namespace payal_book_length_l143_143601

theorem payal_book_length (P : ℕ) 
  (h1 : (2/3 : ℚ) * P = (1/3 : ℚ) * P + 20) : P = 60 :=
sorry

end payal_book_length_l143_143601


namespace composite_probability_l143_143980

noncomputable def probability_composite : ℚ :=
  let total_numbers := 50
      number_composite := total_numbers - 15 - 1
  in number_composite / (total_numbers - 1)

theorem composite_probability :
  probability_composite = 34 / 49 :=
by
  sorry

end composite_probability_l143_143980


namespace problem_l143_143251

namespace MathProof

variable {p a b : ℕ}

theorem problem (h1 : Nat.Prime p) (h2 : p % 2 = 1) (h3 : a > 0) (h4 : b > 0) (h5 : (p + 1)^a - p^b = 1) : a = 1 ∧ b = 1 := 
sorry

end MathProof

end problem_l143_143251


namespace smallest_n_for_geometric_sequence_divisibility_l143_143524

theorem smallest_n_for_geometric_sequence_divisibility :
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬ (2 * 10 ^ 6 ∣ (30 ^ (m - 1) * (5 / 6)))) ∧ (2 * 10 ^ 6 ∣ (30 ^ (n - 1) * (5 / 6))) ∧ n = 8 :=
by
  sorry

end smallest_n_for_geometric_sequence_divisibility_l143_143524


namespace deepak_walking_speed_l143_143046

noncomputable def speed_deepak (circumference: ℕ) (wife_speed_kmph: ℚ) (meet_time_min: ℚ) : ℚ :=
  let meet_time_hr := meet_time_min / 60
  let wife_speed_mpm := wife_speed_kmph * 1000 / 60
  let distance_wife := wife_speed_mpm * meet_time_min
  let distance_deepak := circumference - distance_wife
  let deepak_speed_mpm := distance_deepak / meet_time_min
  deepak_speed_mpm * 60 / 1000

theorem deepak_walking_speed
  (circumference: ℕ) 
  (wife_speed_kmph: ℚ)
  (meet_time_min: ℚ)
  (H1: circumference = 627)
  (H2: wife_speed_kmph = 3.75)
  (H3: meet_time_min = 4.56) :
  speed_deepak circumference wife_speed_kmph meet_time_min = 4.5 :=
by
  sorry

end deepak_walking_speed_l143_143046


namespace softball_team_total_players_l143_143973

theorem softball_team_total_players 
  (M W : ℕ) 
  (h1 : W = M + 4)
  (h2 : (M : ℚ) / (W : ℚ) = 0.6666666666666666) :
  M + W = 20 :=
by sorry

end softball_team_total_players_l143_143973


namespace maximum_m_value_l143_143721

noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 - 2 * x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem maximum_m_value :
  (∀ x > 1, 2 * f' x + x * g x + 3 > m * (x - 1)) → m ≤ 4 :=
by
  sorry

end maximum_m_value_l143_143721


namespace mike_pens_given_l143_143006

noncomputable def pens_remaining (initial_pens mike_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - 19

theorem mike_pens_given 
  (initial_pens : ℕ)
  (mike_pens final_pens : ℕ) 
  (H1 : initial_pens = 7)
  (H2 : final_pens = 39) 
  (H3 : pens_remaining initial_pens mike_pens = final_pens) : 
  mike_pens = 22 := sorry

end mike_pens_given_l143_143006


namespace quarters_given_by_mom_l143_143118

theorem quarters_given_by_mom :
  let dimes := 4
  let quarters := 4
  let nickels := 7
  let value_dimes := 0.10 * dimes
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let initial_total := value_dimes + value_quarters + value_nickels
  let final_total := 3.00
  let additional_amount := final_total - initial_total
  additional_amount / 0.25 = 5 :=
by
  sorry

end quarters_given_by_mom_l143_143118


namespace adam_remaining_loads_l143_143620

-- Define the initial conditions
def total_loads : ℕ := 25
def washed_loads : ℕ := 6

-- Define the remaining loads as the total loads minus the washed loads
def remaining_loads (total_loads washed_loads : ℕ) : ℕ := total_loads - washed_loads

-- State the theorem to be proved
theorem adam_remaining_loads : remaining_loads total_loads washed_loads = 19 := by
  sorry

end adam_remaining_loads_l143_143620


namespace smallest_interesting_number_l143_143842

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l143_143842


namespace find_annual_interest_rate_l143_143326

variable (r : ℝ) -- The annual interest rate we want to prove

-- Define the conditions based on the problem statement
variable (I : ℝ := 300) -- interest earned
variable (P : ℝ := 10000) -- principal amount
variable (t : ℝ := 9 / 12) -- time in years

-- Define the simple interest formula condition
def simple_interest_formula : Prop :=
  I = P * r * t

-- The statement to prove
theorem find_annual_interest_rate : simple_interest_formula r ↔ r = 0.04 :=
  by
    unfold simple_interest_formula
    simp
    sorry

end find_annual_interest_rate_l143_143326


namespace heather_aprons_l143_143222

theorem heather_aprons :
  ∀ (total sewn already_sewn sewn_today half_remaining tomorrow_sew : ℕ),
    total = 150 →
    already_sewn = 13 →
    sewn_today = 3 * already_sewn →
    sewn = already_sewn + sewn_today →
    remaining = total - sewn →
    half_remaining = remaining / 2 →
    tomorrow_sew = half_remaining →
    tomorrow_sew = 49 := 
by 
  -- The proof is left as an exercise.
  sorry

end heather_aprons_l143_143222


namespace problem1_problem2_l143_143012

-- Problem (1)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 2) : 
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 := 
by 
  sorry

-- Problem (2)
theorem problem2 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = -4 / 3 := 
by
  sorry

end problem1_problem2_l143_143012


namespace doubled_volume_l143_143615

theorem doubled_volume (V : ℕ) (h : V = 4) : 8 * V = 32 := by
  sorry

end doubled_volume_l143_143615


namespace farmer_has_42_cows_left_l143_143023

-- Define the conditions
def initial_cows := 51
def added_cows := 5
def sold_fraction := 1 / 4

-- Lean statement to prove the number of cows left
theorem farmer_has_42_cows_left :
  (initial_cows + added_cows) - (sold_fraction * (initial_cows + added_cows)) = 42 :=
by
  -- skipping the proof part
  sorry

end farmer_has_42_cows_left_l143_143023


namespace problem_l143_143734

theorem problem (x y z : ℝ) 
  (h1 : (x - 4)^2 + (y - 3)^2 + (z - 2)^2 = 0)
  (h2 : 3 * x + 2 * y - z = 12) :
  x + y + z = 9 := 
  sorry

end problem_l143_143734


namespace smallest_a_l143_143275

theorem smallest_a (a x : ℤ) (hx : x^2 + a * x = 30) (ha_pos : a > 0) (product_gt_30 : ∃ x₁ x₂ : ℤ, x₁ * x₂ = 30 ∧ x₁ + x₂ = -a ∧ x₁ * x₂ > 30) : a = 11 :=
sorry

end smallest_a_l143_143275


namespace original_equation_solution_l143_143240

noncomputable def original_equation : Prop :=
  ∃ Y P A K P O C : ℕ,
  (Y = 5) ∧ (P = 2) ∧ (A = 0) ∧ (K = 2) ∧ (P = 4) ∧ (O = 0) ∧ (C = 0) ∧
  (Y.factorial * P.factorial * A.factorial = K * 10000 + P * 1000 + O * 100 + C * 10 + C)

theorem original_equation_solution : original_equation :=
  sorry

end original_equation_solution_l143_143240


namespace largest_integer_less_than_100_with_remainder_5_l143_143678

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143678


namespace sum_of_legs_is_104_l143_143796

theorem sum_of_legs_is_104 (x : ℕ) (h₁ : x^2 + (x + 2)^2 = 53^2) : x + (x + 2) = 104 := sorry

end sum_of_legs_is_104_l143_143796


namespace prod_f_zeta_125_l143_143922

noncomputable def f (x : ℂ) : ℂ := 1 + 2 * x + 3 * x^2 + 4 * x^3 + 5 * x^4

noncomputable def zeta : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem prod_f_zeta_125 : f(zeta) * f(zeta^2) * f(zeta^3) * f(zeta^4) = 125 := by
  sorry

end prod_f_zeta_125_l143_143922


namespace tiffany_initial_lives_l143_143567

theorem tiffany_initial_lives (x : ℕ) 
    (H1 : x - 14 + 27 = 56) : x = 43 :=
sorry

end tiffany_initial_lives_l143_143567


namespace value_two_sd_below_mean_l143_143792

theorem value_two_sd_below_mean (mean : ℝ) (std_dev : ℝ) (h_mean : mean = 17.5) (h_std_dev : std_dev = 2.5) : 
  mean - 2 * std_dev = 12.5 := by
  -- proof omitted
  sorry

end value_two_sd_below_mean_l143_143792


namespace num_teams_is_seventeen_l143_143436

-- Each team faces all other teams 10 times and there are 1360 games in total.
def total_teams (n : ℕ) : Prop := 1360 = (n * (n - 1) * 10) / 2

theorem num_teams_is_seventeen : ∃ n : ℕ, total_teams n ∧ n = 17 := 
by 
  sorry

end num_teams_is_seventeen_l143_143436


namespace scientific_notation_of_218000000_l143_143545

theorem scientific_notation_of_218000000 :
  218000000 = 2.18 * 10^8 :=
sorry

end scientific_notation_of_218000000_l143_143545


namespace binomial_expansion_b_value_l143_143061

theorem binomial_expansion_b_value (a b x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x ^ 2 + a^5 * x ^ 5) : b = 40 := 
sorry

end binomial_expansion_b_value_l143_143061


namespace sum_of_x_and_y_l143_143206

theorem sum_of_x_and_y (x y : ℝ) (h : (x + y + 2)^2 + |2 * x - 3 * y - 1| = 0) : x + y = -2 :=
by
  sorry

end sum_of_x_and_y_l143_143206


namespace largest_int_less_than_100_with_remainder_5_l143_143697

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l143_143697


namespace sum_first_n_terms_geom_seq_l143_143242

def geom_seq (n : ℕ) : ℕ :=
match n with
| 0     => 2
| k + 1 => 3 * geom_seq k

def sum_geom_seq (n : ℕ) : ℕ :=
(geom_seq 0) * (3 ^ n - 1) / (3 - 1)

theorem sum_first_n_terms_geom_seq (n : ℕ) :
sum_geom_seq n = 3 ^ n - 1 := by
sorry

end sum_first_n_terms_geom_seq_l143_143242


namespace gcd_187_119_base5_l143_143331

theorem gcd_187_119_base5 :
  ∃ b : Nat, Nat.gcd 187 119 = 17 ∧ 17 = 3 * 5 + 2 ∧ 3 = 0 * 5 + 3 ∧ b = 3 * 10 + 2 := by
  sorry

end gcd_187_119_base5_l143_143331


namespace science_book_pages_l143_143963

def history_pages := 300
def novel_pages := history_pages / 2
def science_pages := novel_pages * 4

theorem science_book_pages : science_pages = 600 := by
  -- Given conditions:
  -- The novel has half as many pages as the history book, the history book has 300 pages,
  -- and the science book has 4 times as many pages as the novel.
  sorry

end science_book_pages_l143_143963


namespace largest_integer_less_than_100_with_remainder_5_l143_143655

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143655


namespace percentage_of_students_absent_l143_143283

theorem percentage_of_students_absent (total_students : ℕ) (students_present : ℕ) 
(h_total : total_students = 50) (h_present : students_present = 43)
(absent_students := total_students - students_present) :
((absent_students : ℝ) / total_students) * 100 = 14 :=
by sorry

end percentage_of_students_absent_l143_143283


namespace percent_students_at_trip_l143_143821

variable (total_students : ℕ)
variable (students_taking_more_than_100 : ℕ := (14 * total_students) / 100)
variable (students_not_taking_more_than_100 : ℕ := (75 * total_students) / 100)
variable (students_who_went_to_trip := (students_taking_more_than_100 * 100) / 25)

/--
  If 14 percent of the students at a school went to a camping trip and took more than $100,
  and 75 percent of the students who went to the camping trip did not take more than $100,
  then 56 percent of the students at the school went to the camping trip.
-/
theorem percent_students_at_trip :
    (students_who_went_to_trip * 100) / total_students = 56 :=
sorry

end percent_students_at_trip_l143_143821


namespace find_c_l143_143271

theorem find_c (a b c : ℚ) (h1 : ∀ y : ℚ, 1 = a * (3 - 1)^2 + b * (3 - 1) + c) (h2 : ∀ y : ℚ, 4 = a * (1)^2 + b * (1) + c)
  (h3 : ∀ y : ℚ, 1 = a * (y - 1)^2 + 4) : c = 13 / 4 :=
by
  sorry

end find_c_l143_143271


namespace perfect_square_values_l143_143869

theorem perfect_square_values :
  ∀ n : ℕ, 0 < n → (∃ k : ℕ, (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by sorry

end perfect_square_values_l143_143869


namespace contradiction_proof_l143_143413

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : ¬ (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :=
by 
  sorry

end contradiction_proof_l143_143413


namespace eight_faucets_fill_time_in_seconds_l143_143712

open Nat

-- Definitions under the conditions
def four_faucets_rate (gallons : ℕ) (minutes : ℕ) : ℕ := gallons / minutes

def one_faucet_rate (four_faucets_rate : ℕ) : ℕ := four_faucets_rate / 4

def eight_faucets_rate (one_faucet_rate : ℕ) : ℕ := one_faucet_rate * 8

def time_to_fill (rate : ℕ) (gallons : ℕ) : ℕ := gallons / rate

-- Main theorem to prove 
theorem eight_faucets_fill_time_in_seconds (gallons_tub : ℕ) (four_faucets_time : ℕ) :
    let four_faucets_rate := four_faucets_rate 200 8
    let one_faucet_rate := one_faucet_rate four_faucets_rate
    let rate_eight_faucets := eight_faucets_rate one_faucet_rate
    let time_fill := time_to_fill rate_eight_faucets 50
    gallons_tub = 50 ∧ four_faucets_time = 8 ∧ rate_eight_faucets = 50 -> time_fill * 60 = 60 :=
by
    intros
    sorry

end eight_faucets_fill_time_in_seconds_l143_143712


namespace find_c_l143_143610

theorem find_c (a b c d : ℕ) (h1 : 8 = 4 * a / 100) (h2 : 4 = d * a / 100) (h3 : 8 = d * b / 100) (h4 : c = b / a) : 
  c = 2 := 
by
  sorry

end find_c_l143_143610


namespace geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l143_143068

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n ≠ 0 ∧ (a (n + 1) = a n * (a (n + 1) / a n))

theorem geometric_sequence_implies_condition (a : ℕ → ℝ) :
  is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1) := sorry

theorem counterexample_condition_does_not_imply_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a := sorry

theorem geometric_sequence_sufficient_not_necessary (a : ℕ → ℝ) :
  (is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1)) ∧
  ((∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a) := by
  exact ⟨geometric_sequence_implies_condition a, counterexample_condition_does_not_imply_geometric_sequence a⟩

end geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l143_143068


namespace thirteen_consecutive_nat_power_l143_143865

def consecutive_sum_power (N : ℕ) : ℕ :=
  (N - 6) + (N - 5) + (N - 4) + (N - 3) + (N - 2) + (N - 1) +
  N + (N + 1) + (N + 2) + (N + 3) + (N + 4) + (N + 5) + (N + 6)

theorem thirteen_consecutive_nat_power (N : ℕ) (n : ℕ) :
  N = 13^2020 →
  n = 2021 →
  consecutive_sum_power N = 13^n := by
  sorry

end thirteen_consecutive_nat_power_l143_143865


namespace village_population_equal_in_15_years_l143_143822

theorem village_population_equal_in_15_years :
  ∀ n : ℕ, (72000 - 1200 * n = 42000 + 800 * n) → n = 15 :=
by
  intros n h
  sorry

end village_population_equal_in_15_years_l143_143822


namespace Will_old_cards_l143_143005

theorem Will_old_cards (new_cards pages cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : pages = 6) (h3 : cards_per_page = 3) :
  (pages * cards_per_page) - new_cards = 10 :=
by
  sorry

end Will_old_cards_l143_143005


namespace max_hawthorns_satisfying_conditions_l143_143018

theorem max_hawthorns_satisfying_conditions :
  ∃ x : ℕ, 
    x > 100 ∧ 
    x % 3 = 1 ∧ 
    x % 4 = 2 ∧ 
    x % 5 = 3 ∧ 
    x % 6 = 4 ∧ 
    (∀ y : ℕ, 
      y > 100 ∧ 
      y % 3 = 1 ∧ 
      y % 4 = 2 ∧ 
      y % 5 = 3 ∧ 
      y % 6 = 4 → y ≤ 178) :=
sorry

end max_hawthorns_satisfying_conditions_l143_143018


namespace chemistry_class_students_l143_143032

theorem chemistry_class_students (total_students both_classes biology_class only_chemistry_class : ℕ)
    (h1: total_students = 100)
    (h2 : both_classes = 10)
    (h3 : total_students = biology_class + only_chemistry_class + both_classes)
    (h4 : only_chemistry_class = 4 * (biology_class + both_classes)) : 
    only_chemistry_class = 80 :=
by
  sorry

end chemistry_class_students_l143_143032


namespace prob_yellow_and_straight_l143_143010

-- Definitions of probabilities given in the problem
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2

-- Derived probability of picking a yellow flower
def prob_yellow : ℚ := 1 - prob_green

-- Statement to prove
theorem prob_yellow_and_straight : prob_yellow * prob_straight = 1 / 6 :=
by
  -- sorry is used here to skip the proof.
  sorry

end prob_yellow_and_straight_l143_143010


namespace fraction_of_fractions_l143_143580

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_fractions_l143_143580


namespace g_2002_value_l143_143801

noncomputable def g : ℕ → ℤ := sorry

theorem g_2002_value :
  (∀ a b n : ℕ, a + b = 2^n → g a + g b = n^3) →
  (g 2 + g 46 = 180) →
  g 2002 = 1126 := 
by
  intros h1 h2
  sorry

end g_2002_value_l143_143801


namespace volume_of_larger_part_of_pyramid_proof_l143_143996

noncomputable def volume_of_larger_part_of_pyramid (a b : ℝ) (inclined_angle : ℝ) (area_ratio : ℝ) : ℝ :=
let h_trapezoid := Real.sqrt ((2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 / 4)
let height_pyramid := (1 / 2) * h_trapezoid * Real.tan (inclined_angle)
let volume_total := (1 / 3) * (((a + b) / 2) * Real.sqrt ((a - b) ^ 2 + 4 * h_trapezoid ^ 2) * height_pyramid)
let volume_smaller := (1 / (5 + 7)) * 7 * volume_total
(volume_total - volume_smaller)

theorem volume_of_larger_part_of_pyramid_proof  :
  (volume_of_larger_part_of_pyramid 2 (Real.sqrt 3) (Real.pi / 6) (5 / 7) = 0.875) :=
by
sorry

end volume_of_larger_part_of_pyramid_proof_l143_143996


namespace algebra_simplification_l143_143904

theorem algebra_simplification (a b : ℤ) (h : ∀ x : ℤ, x^2 - 6 * x + b = (x - a)^2 - 1) : b - a = 5 := by
  sorry

end algebra_simplification_l143_143904


namespace debate_team_has_11_boys_l143_143802

def debate_team_boys_count (num_groups : Nat) (members_per_group : Nat) (num_girls : Nat) : Nat :=
  let total_members := num_groups * members_per_group
  total_members - num_girls

theorem debate_team_has_11_boys :
  debate_team_boys_count 8 7 45 = 11 :=
by
  sorry

end debate_team_has_11_boys_l143_143802


namespace problem_part1_problem_part2_problem_part3_l143_143075

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x^2 - x

theorem problem_part1 :
  (∀ x, 0 < x -> x < 1 / Real.exp 1 -> f (Real.log x + 1) < 0) ∧ 
  (∀ x, x > 1 / Real.exp 1 -> f (Real.log x + 1) > 0) ∧ 
  (f (1 / Real.exp 1) = 1 / Real.exp 1 * Real.log (1 / Real.exp 1)) :=
sorry

theorem problem_part2 (a : ℝ) :
  (∀ x, x > 0 -> f x ≤ g a x) ↔ a ≥ 1 :=
sorry

theorem problem_part3 (a : ℝ) (m : ℝ) (ha : a = 1/8) :
  (∃ m, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ (3 * f x / (4 * x) + m + g a x = 0))) ↔ 
  (7/8 < m ∧ m < (15/8 - 3/4 * Real.log 3)) :=
sorry

end problem_part1_problem_part2_problem_part3_l143_143075


namespace problem_statement_l143_143227

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (f (g (f 1))) / (g (f (g 1))) = (-23 : ℝ) / 5 :=
by 
  sorry

end problem_statement_l143_143227


namespace smallest_factor_of_36_sum_4_l143_143537

theorem smallest_factor_of_36_sum_4 : ∃ a b c : ℤ, (a * b * c = 36) ∧ (a + b + c = 4) ∧ (a = -4 ∨ b = -4 ∨ c = -4) :=
by
  sorry

end smallest_factor_of_36_sum_4_l143_143537


namespace find_other_discount_l143_143993

theorem find_other_discount (P F d1 : ℝ) (H₁ : P = 70) (H₂ : F = 61.11) (H₃ : d1 = 10) : ∃ (d2 : ℝ), d2 = 3 :=
by 
  -- The proof will be provided here.
  sorry

end find_other_discount_l143_143993


namespace largest_int_less_than_100_with_remainder_5_l143_143693

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l143_143693


namespace arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l143_143758

variable {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
variable (h_a2 : a 2 = 3 * a 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = ∑ i in Finset.range(n+1), a i)
variable (h_sqrt_S_arith : ∃ d, ∀ n, (Sqrt.sqrt (S n) - Sqrt.sqrt (S (n - 1))) = d)

theorem arithmetic_seq_of_pos_and_arithmetic_sqrt_S : 
  ∀ n, a (n+1) - a n = a 1 := 
sorry

end arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l143_143758


namespace average_age_of_class_l143_143424

theorem average_age_of_class 
  (avg_age_8 : ℕ → ℕ)
  (avg_age_6 : ℕ → ℕ)
  (age_15th : ℕ)
  (A : ℕ)
  (h1 : avg_age_8 8 = 112)
  (h2 : avg_age_6 6 = 96)
  (h3 : age_15th = 17)
  (h4 : 15 * A = (avg_age_8 8) + (avg_age_6 6) + age_15th)
  : A = 15 :=
by
  sorry

end average_age_of_class_l143_143424


namespace split_costs_evenly_l143_143570

theorem split_costs_evenly (t d : ℕ) 
  (Tom_paid : 150) (Dorothy_paid : 180) (Sammy_paid : 220) (Nick_paid : 250) 
  (total_paid : 150 + 180 + 220 + 250 = 800)
  (even_split : 800 / 4 = 200)
  (Tom_owes : Tom_paid < even_split)
  (Dorothy_owes : Dorothy_paid < even_split)
  (t_def : t = even_split - Tom_paid)
  (d_def : d = even_split - Dorothy_paid) :
  t - d = 30 := 
by
  sorry

end split_costs_evenly_l143_143570


namespace find_n_sin_eq_l143_143872

theorem find_n_sin_eq (n : ℤ) (h₁ : -180 ≤ n) (h₂ : n ≤ 180) (h₃ : Real.sin (n * Real.pi / 180) = Real.sin (680 * Real.pi / 180)) :
  n = 40 ∨ n = 140 :=
by
  sorry

end find_n_sin_eq_l143_143872


namespace tens_digit_of_even_not_divisible_by_10_l143_143764

theorem tens_digit_of_even_not_divisible_by_10 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) :
  (N ^ 20) % 100 / 10 % 10 = 7 :=
sorry

end tens_digit_of_even_not_divisible_by_10_l143_143764


namespace doubled_container_volume_l143_143614

theorem doubled_container_volume (original_volume : ℕ) (factor : ℕ) 
  (h1 : original_volume = 4) (h2 : factor = 8) : original_volume * factor = 32 :=
by 
  rw [h1, h2]
  norm_num

end doubled_container_volume_l143_143614


namespace num_integers_between_sqrt10_sqrt100_l143_143726

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l143_143726


namespace minimize_F_l143_143773

theorem minimize_F : ∃ x1 x2 x3 x4 x5 : ℝ, 
  (-2 * x1 + x2 + x3 = 2) ∧ 
  (x1 - 2 * x2 + x4 = 2) ∧ 
  (x1 + x2 + x5 = 5) ∧ 
  (x1 ≥ 0) ∧ 
  (x2 ≥ 0) ∧ 
  (x2 - x1 = -3) :=
by {
  sorry
}

end minimize_F_l143_143773


namespace junk_mail_per_red_or_white_house_l143_143173

noncomputable def pieces_per_house (total_pieces : ℕ) (total_houses : ℕ) : ℕ := 
  total_pieces / total_houses

noncomputable def total_pieces_for_type (pieces_per_house : ℕ) (houses_of_type : ℕ) : ℕ := 
  pieces_per_house * houses_of_type

noncomputable def total_pieces_for_red_or_white 
  (total_pieces : ℕ)
  (total_houses : ℕ)
  (white_houses : ℕ)
  (red_houses : ℕ) : ℕ :=
  let pieces_per_house := pieces_per_house total_pieces total_houses
  let pieces_for_white := total_pieces_for_type pieces_per_house white_houses
  let pieces_for_red := total_pieces_for_type pieces_per_house red_houses
  pieces_for_white + pieces_for_red

theorem junk_mail_per_red_or_white_house :
  ∀ (total_pieces : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ),
    total_pieces = 48 →
    total_houses = 8 →
    white_houses = 2 →
    red_houses = 3 →
    total_pieces_for_red_or_white total_pieces total_houses white_houses red_houses / (white_houses + red_houses) = 6 :=
by
  intros
  sorry

end junk_mail_per_red_or_white_house_l143_143173


namespace common_ratio_value_l143_143747

variable (a : ℕ → ℝ) -- defining the geometric sequence as a function ℕ → ℝ
variable (q : ℝ) -- defining the common ratio

-- conditions from the problem
def geo_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

axiom h1 : geo_seq a q
axiom h2 : a 2020 = 8 * a 2017

-- main statement to be proved
theorem common_ratio_value : q = 2 :=
sorry

end common_ratio_value_l143_143747


namespace three_digit_number_exists_l143_143984

theorem three_digit_number_exists : 
  ∃ (x y z : ℕ), 
  (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧ 
  (100 * x + 10 * z + y + 1 = 2 * (100 * y + 10 * z + x)) ∧ 
  (100 * x + 10 * z + y = 793) :=
by
  sorry

end three_digit_number_exists_l143_143984


namespace find_S_9_l143_143096

variable (a : ℕ → ℝ)

def arithmetic_sum_9 (S_9 : ℝ) : Prop :=
  (a 1 + a 3 + a 5 = 39) ∧ (a 5 + a 7 + a 9 = 27) ∧ (S_9 = (9 * (a 3 + a 7)) / 2)

theorem find_S_9 
  (h1 : a 1 + a 3 + a 5 = 39)
  (h2 : a 5 + a 7 + a 9 = 27) :
  ∃ S_9, arithmetic_sum_9 a S_9 ∧ S_9 = 99 := 
by
  sorry

end find_S_9_l143_143096


namespace volume_ratio_l143_143290

theorem volume_ratio (a : ℕ) (b : ℕ) (ft_to_inch : ℕ) (h1 : a = 4) (h2 : b = 2 * ft_to_inch) (ft_to_inch_value : ft_to_inch = 12) :
  (a^3) / (b^3) = 1 / 216 :=
by
  sorry

end volume_ratio_l143_143290


namespace jimmy_more_sheets_than_tommy_l143_143571

-- Definitions for the conditions
def initial_jimmy_sheets : ℕ := 58
def initial_tommy_sheets : ℕ := initial_jimmy_sheets + 25
def ashton_gives_jimmy : ℕ := 85
def jessica_gives_jimmy : ℕ := 47
def cousin_gives_tommy : ℕ := 30
def aunt_gives_tommy : ℕ := 19

-- Lean 4 statement for the proof problem
theorem jimmy_more_sheets_than_tommy :
  let final_jimmy_sheets := initial_jimmy_sheets + ashton_gives_jimmy + jessica_gives_jimmy;
  let final_tommy_sheets := initial_tommy_sheets + cousin_gives_tommy + aunt_gives_tommy;
  final_jimmy_sheets - final_tommy_sheets = 58 :=
by sorry

end jimmy_more_sheets_than_tommy_l143_143571


namespace similar_triangles_perimeter_l143_143800

theorem similar_triangles_perimeter (P_small P_large : ℝ) 
  (h_ratio : P_small / P_large = 2 / 3) 
  (h_sum : P_small + P_large = 20) : 
  P_small = 8 := 
sorry

end similar_triangles_perimeter_l143_143800


namespace petes_original_number_l143_143123

theorem petes_original_number (x : ℤ) (h : 5 * (3 * x - 6) = 195) : x = 15 :=
sorry

end petes_original_number_l143_143123


namespace parallelogram_count_l143_143864

noncomputable def num_parallelograms (n : ℕ) : ℕ :=
  3 * (n + 2).choose 4

theorem parallelogram_count {n : ℕ} : 
  (each_side_divided n) → 
  (lines_drawn_parallel_thru_points n) → 
  num_parallelograms n = 3 * (n + 2).choose 4 := 
by {
  intro h,
  sorry
}

end parallelogram_count_l143_143864


namespace radius_larger_ball_l143_143561

-- Define the volume formula for a sphere.
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Parameters for the problem
def radius_small_ball : ℝ := 2
def total_volume_small_balls : ℝ := 6 * volume_of_sphere radius_small_ball

-- Prove that the radius of the larger ball is 4 * 2^(1 / 3) (which is 4 * cube root of 2).
theorem radius_larger_ball : ∃ r : ℝ, volume_of_sphere r = total_volume_small_balls ∧ r = 4 * Real.cbrt 2 := by
  sorry

end radius_larger_ball_l143_143561


namespace jerry_weekly_earnings_l143_143105

theorem jerry_weekly_earnings:
  (tasks_per_day: ℕ) 
  (daily_earnings: ℕ)
  (weekly_earnings: ℕ) :
  (tasks_per_day = 10 / 2) ∧
  (daily_earnings = 40 * tasks_per_day) ∧
  (weekly_earnings = daily_earnings * 7) →
  weekly_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l143_143105


namespace decimal_to_binary_25_l143_143642

theorem decimal_to_binary_25 : (25 : Nat) = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_25_l143_143642


namespace solve_m_l143_143505

theorem solve_m (m : ℝ) : 
  (m - 3) * x^2 - 3 * x + m^2 = 9 → m^2 - 9 = 0 → m = -3 :=
by
  sorry

end solve_m_l143_143505


namespace prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l143_143786

-- Problem 1
theorem prob1_part1 : |-4 + 6| = 2 := sorry
theorem prob1_part2 : |-2 - 4| = 6 := sorry

-- Problem 2
theorem find_integers_x :
  {x : ℤ | |x + 2| + |x - 1| = 3} = {-2, -1, 0, 1} :=
sorry

-- Problem 3
theorem prob3 (a : ℤ) (h : -4 ≤ a ∧ a ≤ 6) : |a + 4| + |a - 6| = 10 :=
sorry

-- Problem 4
theorem min_value_prob4 :
  ∃ (a : ℤ), |a - 1| + |a + 5| + |a - 4| = 9 ∧ ∀ (b : ℤ), |b - 1| + |b + 5| + |b - 4| ≥ 9 :=
sorry

end prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l143_143786


namespace team_A_more_uniform_l143_143548

noncomputable def average_height : ℝ := 2.07

variables (S_A S_B : ℝ) (h_variance : S_A^2 < S_B^2)

theorem team_A_more_uniform : true ∧ false :=
by
  sorry

end team_A_more_uniform_l143_143548


namespace bailey_towel_set_cost_l143_143037

def guest_bathroom_sets : ℕ := 2
def master_bathroom_sets : ℕ := 4
def cost_per_guest_set : ℝ := 40.00
def cost_per_master_set : ℝ := 50.00
def discount_rate : ℝ := 0.20

def total_cost_before_discount : ℝ := 
  (guest_bathroom_sets * cost_per_guest_set) + (master_bathroom_sets * cost_per_master_set)

def discount_amount : ℝ := total_cost_before_discount * discount_rate

def final_amount_spent : ℝ := total_cost_before_discount - discount_amount

theorem bailey_towel_set_cost : final_amount_spent = 224.00 := by sorry

end bailey_towel_set_cost_l143_143037


namespace largest_integer_less_than_100_with_remainder_5_l143_143656

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143656


namespace weight_of_172_is_around_60_316_l143_143569

noncomputable def weight_prediction (x : ℝ) : ℝ := 0.849 * x - 85.712

theorem weight_of_172_is_around_60_316 :
  ∀ (x : ℝ), x = 172 → abs (weight_prediction x - 60.316) < 1 :=
by
  sorry

end weight_of_172_is_around_60_316_l143_143569


namespace trumpet_cost_l143_143527

def cost_of_song_book : Real := 5.84
def total_spent : Real := 151
def cost_of_trumpet : Real := total_spent - cost_of_song_book

theorem trumpet_cost : cost_of_trumpet = 145.16 :=
by
  sorry

end trumpet_cost_l143_143527


namespace intersection_of_A_and_B_l143_143070

def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 4} := 
by 
  sorry

end intersection_of_A_and_B_l143_143070


namespace cake_pieces_in_pan_l143_143509

theorem cake_pieces_in_pan :
  (24 * 30) / (3 * 2) = 120 := by
  sorry

end cake_pieces_in_pan_l143_143509


namespace intersection_complement_eq_l143_143069

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Define the complement of B in U
def complement_B_in_U : Set ℕ := { x ∈ U | x ∉ B }

-- The main theorem statement stating the required equality
theorem intersection_complement_eq : A ∩ complement_B_in_U = {2, 3} := by
  sorry

end intersection_complement_eq_l143_143069


namespace single_cone_scoops_l143_143859

theorem single_cone_scoops (banana_split_scoops : ℕ) (waffle_bowl_scoops : ℕ) (single_cone_scoops : ℕ) (double_cone_scoops : ℕ)
  (h1 : banana_split_scoops = 3 * single_cone_scoops)
  (h2 : waffle_bowl_scoops = banana_split_scoops + 1)
  (h3 : double_cone_scoops = 2 * single_cone_scoops)
  (h4 : single_cone_scoops + banana_split_scoops + waffle_bowl_scoops + double_cone_scoops = 10) :
  single_cone_scoops = 1 :=
by
  sorry

end single_cone_scoops_l143_143859


namespace fraction_simplification_l143_143476

theorem fraction_simplification (x : ℝ) (h : x = 0.5 * 106) : 18 / x = 18 / 53 := by
  rw [h]
  norm_num

end fraction_simplification_l143_143476


namespace mail_distribution_l143_143175

theorem mail_distribution (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : total_mail / total_houses = 6 := by
  sorry

end mail_distribution_l143_143175


namespace fraction_sum_condition_l143_143492

theorem fraction_sum_condition 
  (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0)
  (h : x + y = x * y): 
  (1/x + 1/y = 1) :=
by
  sorry

end fraction_sum_condition_l143_143492


namespace radius_condition_l143_143606

def X (x y : ℝ) : ℝ := 12 * x
def Y (x y : ℝ) : ℝ := 5 * y

def satisfies_condition (x y : ℝ) : Prop :=
  Real.sin (X x y + Y x y) = Real.sin (X x y) + Real.sin (Y x y)

def no_intersection (R : ℝ) : Prop :=
  ∀ (x y : ℝ), satisfies_condition x y → dist (0, 0) (x, y) ≥ R

theorem radius_condition :
  ∀ R : ℝ, (0 < R ∧ R < Real.pi / 15) →
  no_intersection R :=
sorry

end radius_condition_l143_143606


namespace binom_n_2_l143_143810

theorem binom_n_2 (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_n_2_l143_143810


namespace find_angle_C_find_area_triangle_l143_143234

open Real

-- Let the angles and sides of the triangle be defined as follows
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
axiom condition1 : (a^2 + b^2 - c^2) * (tan C) = sqrt 2 * a * b
axiom condition2 : c = 2
axiom condition3 : b = 2 * sqrt 2

-- Proof statements
theorem find_angle_C :
  C = pi / 4 ∨ C = 3 * pi / 4 :=
sorry

theorem find_area_triangle :
  C = pi / 4 → a = 2 → (1 / 2) * a * b * sin C = 2 :=
sorry

end find_angle_C_find_area_triangle_l143_143234


namespace kanul_cash_spending_percentage_l143_143753

theorem kanul_cash_spending_percentage :
  ∀ (spent_raw_materials spent_machinery total_amount spent_cash : ℝ),
    spent_raw_materials = 500 →
    spent_machinery = 400 →
    total_amount = 1000 →
    spent_cash = total_amount - (spent_raw_materials + spent_machinery) →
    (spent_cash / total_amount) * 100 = 10 :=
by
  intros spent_raw_materials spent_machinery total_amount spent_cash
  intro h1 h2 h3 h4
  sorry

end kanul_cash_spending_percentage_l143_143753


namespace smallest_interesting_number_is_1800_l143_143838

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l143_143838


namespace solve_for_a_l143_143474

def E (a b c : ℝ) : ℝ := a * b^2 + b * c + c

theorem solve_for_a : (E (-5/8) 3 2 = E (-5/8) 5 3) :=
by
  sorry

end solve_for_a_l143_143474


namespace smallest_interesting_number_l143_143841

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l143_143841


namespace mikes_ride_is_46_miles_l143_143411

-- Define the conditions and the question in Lean 4
variable (M : ℕ)

-- Mike's cost formula
def mikes_cost (M : ℕ) : ℚ := 2.50 + 0.25 * M

-- Annie's total cost
def annies_miles : ℕ := 26
def annies_cost : ℚ := 2.50 + 5.00 + 0.25 * annies_miles

-- The proof statement
theorem mikes_ride_is_46_miles (h : mikes_cost M = annies_cost) : M = 46 :=
by sorry

end mikes_ride_is_46_miles_l143_143411


namespace gifts_left_l143_143627

variable (initial_gifts : ℕ)
variable (gifts_sent : ℕ)

theorem gifts_left (h_initial : initial_gifts = 77) (h_sent : gifts_sent = 66) : initial_gifts - gifts_sent = 11 := by
  sorry

end gifts_left_l143_143627


namespace science_book_pages_l143_143965

theorem science_book_pages {history_pages novel_pages science_pages: ℕ} (h1: novel_pages = history_pages / 2) (h2: science_pages = 4 * novel_pages) (h3: history_pages = 300):
  science_pages = 600 :=
by
  sorry

end science_book_pages_l143_143965


namespace nicolai_peaches_pounds_l143_143148

-- Condition definitions
def total_fruit_pounds : ℕ := 8
def mario_oranges_ounces : ℕ := 8
def lydia_apples_ounces : ℕ := 24
def ounces_to_pounds (ounces: ℕ) : ℚ := ounces / 16 

-- Statement we want to prove
theorem nicolai_peaches_pounds :
  let mario_oranges_pounds := ounces_to_pounds mario_oranges_ounces,
      lydia_apples_pounds := ounces_to_pounds lydia_apples_ounces,
      eaten_by_m_and_l := mario_oranges_pounds + lydia_apples_pounds,
      nicolai_peaches_pounds := total_fruit_pounds - eaten_by_m_and_l in
  nicolai_peaches_pounds = 6 :=
by
  sorry

end nicolai_peaches_pounds_l143_143148


namespace intersection_sum_l143_143055

-- Define the conditions
def condition_1 (k : ℝ) := k > 0
def line1 (x y k : ℝ) := 50 * x + k * y = 1240
def line2 (x y k : ℝ) := k * y = 8 * x + 544
def right_angles (k : ℝ) := (-50 / k) * (8 / k) = -1

-- Define the point of intersection
def point_of_intersection (m n : ℝ) (k : ℝ) := line1 m n k ∧ line2 m n k

-- Prove that m + n = 44 under the given conditions
theorem intersection_sum (m n k : ℝ) :
  condition_1 k →
  right_angles k →
  point_of_intersection m n k →
  m + n = 44 :=
by
  sorry

end intersection_sum_l143_143055


namespace sum_p_q_l143_143555

-- Define the cubic polynomial q(x)
def cubic_q (q : ℚ) (x : ℚ) := q * x * (x - 1) * (x + 1)

-- Define the linear polynomial p(x)
def linear_p (p : ℚ) (x : ℚ) := p * x

-- Prove the result for p(x) + q(x)
theorem sum_p_q : 
  (∀ p q : ℚ, linear_p p 4 = 4 → cubic_q q 3 = 3 → (∀ x : ℚ, linear_p p x + cubic_q q x = (1 / 24) * x^3 + (23 / 24) * x)) :=
by
  intros p q hp hq x
  sorry

end sum_p_q_l143_143555


namespace larger_number_is_50_l143_143085

variable (a b : ℕ)
-- Conditions given in the problem
axiom cond1 : 4 * b = 5 * a
axiom cond2 : b - a = 10

-- The proof statement
theorem larger_number_is_50 : b = 50 :=
sorry

end larger_number_is_50_l143_143085


namespace number_of_draw_matches_eq_points_difference_l143_143511

-- Definitions based on the conditions provided
def teams : ℕ := 16
def matches_per_round : ℕ := 8
def rounds : ℕ := 16
def total_points : ℕ := 222
def total_matches : ℕ := matches_per_round * rounds
def hypothetical_points : ℕ := total_matches * 2
def points_difference : ℕ := hypothetical_points - total_points

-- Theorem stating the equivalence to be proved
theorem number_of_draw_matches_eq_points_difference : 
  points_difference = 34 := 
by
  sorry

end number_of_draw_matches_eq_points_difference_l143_143511


namespace no_root_of_equation_l143_143039

theorem no_root_of_equation : ∀ x : ℝ, x - 8 / (x - 4) ≠ 4 - 8 / (x - 4) :=
by
  intro x
  -- Original equation:
  -- x - 8 / (x - 4) = 4 - 8 / (x - 4)
  -- No valid value of x solves the above equation as shown in the given solution
  sorry

end no_root_of_equation_l143_143039


namespace find_d_l143_143054

theorem find_d {x d : ℤ} (h : (x + (x + 2) + (x + 4) + (x + 7) + (x + d)) / 5 = (x + 4) + 6) : d = 37 :=
sorry

end find_d_l143_143054


namespace parametric_to_ordinary_eq_l143_143489

variable (t : ℝ)

theorem parametric_to_ordinary_eq (h1 : x = Real.sqrt t + 1) (h2 : y = 2 * Real.sqrt t - 1) (h3 : t ≥ 0) :
    y = 2 * x - 3 ∧ x ≥ 1 := by
  sorry

end parametric_to_ordinary_eq_l143_143489


namespace slope_of_line_l143_143816

theorem slope_of_line (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 / x + 4 / y = 0) : 
  ∃ m : ℝ, m = -4 / 3 := 
sorry

end slope_of_line_l143_143816


namespace primes_sum_solutions_l143_143870

theorem primes_sum_solutions :
  ∃ (p q r : ℕ), Prime p ∧ Prime q ∧ Prime r ∧
  p + q^2 + r^3 = 200 ∧ 
  ((p = 167 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 11 ∧ r = 2) ∨ 
   (p = 23 ∧ q = 13 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 2 ∧ r = 5)) :=
sorry

end primes_sum_solutions_l143_143870


namespace number_of_f3_and_sum_of_f3_l143_143924

noncomputable def f : ℝ → ℝ := sorry
variable (a : ℝ)

theorem number_of_f3_and_sum_of_f3 (hf : ∀ x y : ℝ, f (f x - y) = f x + f (f y - f a) + x) :
  (∃! c : ℝ, f 3 = c) ∧ (∃ s : ℝ, (∀ c, f 3 = c → s = c) ∧ s = 3) :=
sorry

end number_of_f3_and_sum_of_f3_l143_143924


namespace complement_intersection_l143_143372

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 2}

-- Define the set B
def B : Set ℕ := {2, 3, 4}

-- Statement to be proven
theorem complement_intersection :
  (U \ A) ∩ B = {3, 4} :=
sorry

end complement_intersection_l143_143372


namespace scientific_notation_15510000_l143_143305

theorem scientific_notation_15510000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 15510000 = a * 10^n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end scientific_notation_15510000_l143_143305


namespace largest_integer_less_than_100_with_remainder_5_l143_143704

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l143_143704


namespace solve_inequality_l143_143053

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 2 * x - 3) * (x ^ 2 - 4 * x + 4) < 0 ↔ (-1 < x ∧ x < 3 ∧ x ≠ 2) := by
  sorry

end solve_inequality_l143_143053


namespace decimal_to_binary_25_l143_143643

theorem decimal_to_binary_25 : (25 : Nat) = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_25_l143_143643


namespace nicky_catches_up_time_l143_143299

theorem nicky_catches_up_time
  (head_start : ℕ := 12)
  (cristina_speed : ℕ := 5)
  (nicky_speed : ℕ := 3)
  (head_start_distance : ℕ := nicky_speed * head_start)
  (time_to_catch_up : ℕ := 36 / 2) -- 36 is the head start distance of 36 meters
  (total_time : ℕ := time_to_catch_up + head_start)  -- Total time Nicky runs before Cristina catches up
  : total_time = 30 := sorry

end nicky_catches_up_time_l143_143299


namespace total_books_for_girls_l143_143432

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end total_books_for_girls_l143_143432


namespace tank_filled_fraction_l143_143732

noncomputable def initial_quantity (total_capacity : ℕ) := (3 / 4 : ℚ) * total_capacity

noncomputable def final_quantity (initial : ℚ) (additional : ℚ) := initial + additional

noncomputable def fraction_of_capacity (quantity : ℚ) (total_capacity : ℕ) := quantity / total_capacity

theorem tank_filled_fraction (total_capacity : ℕ) (additional_gas : ℚ)
  (initial_fraction : ℚ) (final_fraction : ℚ) :
  initial_fraction = initial_quantity total_capacity →
  final_fraction = fraction_of_capacity (final_quantity initial_fraction additional_gas) total_capacity →
  total_capacity = 42 →
  additional_gas = 7 →
  initial_fraction = 31.5 →
  final_fraction = (833 / 909 : ℚ) :=
by
  sorry

end tank_filled_fraction_l143_143732


namespace students_answered_both_correctly_l143_143530

theorem students_answered_both_correctly 
(total_students : ℕ) 
(did_not_answer_A_correctly : ℕ) 
(answered_A_correctly_but_not_B : ℕ) 
(h1 : total_students = 50) 
(h2 : did_not_answer_A_correctly = 12) 
(h3 : answered_A_correctly_but_not_B = 30) : 
    (total_students - did_not_answer_A_correctly - answered_A_correctly_but_not_B) = 8 :=
by
    sorry

end students_answered_both_correctly_l143_143530


namespace find_a_plus_d_l143_143446

theorem find_a_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : c + d = 3) : a + d = -1 := 
by 
  -- omit proof
  sorry

end find_a_plus_d_l143_143446


namespace popsicle_sum_l143_143876

-- Gino has 63 popsicle sticks
def gino_popsicle_sticks : Nat := 63

-- I have 50 popsicle sticks
def my_popsicle_sticks : Nat := 50

-- The sum of our popsicle sticks
def total_popsicle_sticks : Nat := gino_popsicle_sticks + my_popsicle_sticks

-- Prove that the total is 113
theorem popsicle_sum : total_popsicle_sticks = 113 :=
by
  -- Proof goes here
  sorry

end popsicle_sum_l143_143876


namespace second_tree_ring_groups_l143_143188

-- Definition of the problem conditions
def group_rings (fat thin : Nat) : Nat := fat + thin

-- Conditions
def FirstTreeRingGroups : Nat := 70
def RingsPerGroup : Nat := group_rings 2 4
def FirstTreeRings : Nat := FirstTreeRingGroups * RingsPerGroup
def AgeDifference : Nat := 180

-- Calculate the total number of rings in the second tree
def SecondTreeRings : Nat := FirstTreeRings - AgeDifference

-- Prove the number of ring groups in the second tree
theorem second_tree_ring_groups : SecondTreeRings / RingsPerGroup = 40 :=
by
  sorry

end second_tree_ring_groups_l143_143188


namespace tracy_initial_candies_l143_143572

noncomputable def initial_candies : Nat := 80

theorem tracy_initial_candies
  (x : Nat)
  (hx1 : ∃ y : Nat, (1 ≤ y ∧ y ≤ 6) ∧ x = (5 * (44 + y)) / 3)
  (hx2 : x % 20 = 0) : x = initial_candies := by
  sorry

end tracy_initial_candies_l143_143572


namespace nicolai_peaches_6_pounds_l143_143143

noncomputable def amount_peaches (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ) : ℕ :=
  let total_ounces := total_pounds * 16
  let total_consumed := oz_oranges + oz_apples
  let remaining_ounces := total_ounces - total_consumed
  remaining_ounces / 16

theorem nicolai_peaches_6_pounds (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ)
  (h_total_pounds : total_pounds = 8) (h_oz_oranges : oz_oranges = 8) (h_oz_apples : oz_apples = 24) :
  amount_peaches total_pounds oz_oranges oz_apples = 6 :=
by
  rw [h_total_pounds, h_oz_oranges, h_oz_apples]
  unfold amount_peaches
  sorry

end nicolai_peaches_6_pounds_l143_143143


namespace fraction_of_fractions_l143_143579

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_fractions_l143_143579


namespace product_of_solutions_of_abs_equation_l143_143362

theorem product_of_solutions_of_abs_equation :
  (∃ x₁ x₂ : ℚ, |5 * x₁ - 2| + 7 = 52 ∧ |5 * x₂ - 2| + 7 = 52 ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ = -2021 / 25)) :=
sorry

end product_of_solutions_of_abs_equation_l143_143362


namespace toll_for_18_wheel_truck_l143_143563

-- Define the number of wheels on the front axle and the other axles
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def total_wheels : ℕ := 18

-- Define the toll formula
def toll (x : ℕ) : ℝ := 3.50 + 0.50 * (x - 2)

-- Calculate the number of axles for the 18-wheel truck
def num_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the expected toll for the given number of axles
def expected_toll : ℝ := 5.00

-- State the theorem
theorem toll_for_18_wheel_truck : toll num_axles = expected_toll := by
    sorry

end toll_for_18_wheel_truck_l143_143563


namespace problem_l143_143407

theorem problem (n : ℕ) (p : ℕ) (a b c : ℤ)
  (hn : 0 < n)
  (hp : Nat.Prime p)
  (h_eq : a^n + p * b = b^n + p * c)
  (h_eq2 : b^n + p * c = c^n + p * a) :
  a = b ∧ b = c := 
sorry

end problem_l143_143407


namespace sum_of_consecutive_integers_l143_143564

theorem sum_of_consecutive_integers (x y z : ℤ) (h1 : y = x + 1) (h2 : z = y + 1) (h3 : z = 12) :
  x + y + z = 33 :=
sorry

end sum_of_consecutive_integers_l143_143564


namespace arithmetic_sequence_sum_l143_143562

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 3 = 6)
  (h2 : S 9 = 27) :
  S 6 = 15 :=
sorry

end arithmetic_sequence_sum_l143_143562


namespace michael_payment_correct_l143_143935

def suit_price : ℕ := 430
def suit_discount : ℕ := 100
def shoes_price : ℕ := 190
def shoes_discount : ℕ := 30
def shirt_price : ℕ := 80
def tie_price: ℕ := 50
def combined_discount : ℕ := (shirt_price + tie_price) * 20 / 100

def total_price_paid : ℕ :=
    suit_price - suit_discount + shoes_price - shoes_discount + (shirt_price + tie_price - combined_discount)

theorem michael_payment_correct :
    total_price_paid = 594 :=
by
    -- skipping the proof
    sorry

end michael_payment_correct_l143_143935


namespace largest_integer_lt_100_with_remainder_5_div_8_l143_143666

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l143_143666


namespace consecutive_integer_sets_sum_27_l143_143899

theorem consecutive_integer_sets_sum_27 :
  ∃! s : Set (List ℕ), ∀ l ∈ s, 
  (∃ n a, n ≥ 3 ∧ l = List.range n ++ [a] ∧ (List.sum l) = 27)
:=
sorry

end consecutive_integer_sets_sum_27_l143_143899


namespace solution_set_inequality_l143_143500

theorem solution_set_inequality {a b c : ℝ} (h₁ : a < 0)
  (h₂ : ∀ x : ℝ, (a * x^2 + b * x + c <= 0) ↔ (x <= -(1/3) ∨ 2 <= x)) :
  (∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -3 ∨ 1/2 < x)) :=
by
  sorry

end solution_set_inequality_l143_143500


namespace largest_integer_less_than_100_with_remainder_5_l143_143690

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l143_143690


namespace molecular_weight_of_compound_l143_143151

noncomputable def atomic_weight_carbon : ℝ := 12.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.008
noncomputable def atomic_weight_oxygen : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def num_oxygen_atoms : ℕ := 1

noncomputable def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  (num_C * atomic_weight_carbon) + (num_H * atomic_weight_hydrogen) + (num_O * atomic_weight_oxygen)

theorem molecular_weight_of_compound :
  molecular_weight num_carbon_atoms num_hydrogen_atoms num_oxygen_atoms = 65.048 :=
by
  sorry

end molecular_weight_of_compound_l143_143151


namespace dino_finances_l143_143477

def earnings_per_gig (hours: ℕ) (rate: ℕ) : ℕ := hours * rate

def dino_total_income : ℕ :=
  earnings_per_gig 20 10 + -- Earnings from the first gig
  earnings_per_gig 30 20 + -- Earnings from the second gig
  earnings_per_gig 5 40    -- Earnings from the third gig

def dino_expenses : ℕ := 500

def dino_net_income : ℕ :=
  dino_total_income - dino_expenses

theorem dino_finances : 
  dino_net_income = 500 :=
by
  -- Here, the actual proof would be constructed.
  sorry

end dino_finances_l143_143477


namespace johns_total_cost_after_discount_l143_143918

/-- Price of different utensils for John's purchase --/
def forks_cost : ℕ := 25
def knives_cost : ℕ := 30
def spoons_cost : ℕ := 20
def dinner_plate_cost (silverware_cost : ℕ) : ℚ := 0.5 * silverware_cost

/-- Calculating the total cost of silverware --/
def total_silverware_cost : ℕ := forks_cost + knives_cost + spoons_cost

/-- Calculating the total cost before discount --/
def total_cost_before_discount : ℚ := total_silverware_cost + dinner_plate_cost total_silverware_cost

/-- Discount rate --/
def discount_rate : ℚ := 0.10

/-- Discount amount --/
def discount_amount (total_cost : ℚ) : ℚ := discount_rate * total_cost

/-- Total cost after applying discount --/
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount total_cost_before_discount

/-- John's total cost after the discount should be $101.25 --/
theorem johns_total_cost_after_discount : total_cost_after_discount = 101.25 := by
  sorry

end johns_total_cost_after_discount_l143_143918


namespace sin_pow_cos_pow_sum_l143_143733

namespace ProofProblem

-- Define the condition
def trig_condition (x : ℝ) : Prop :=
  3 * (Real.sin x)^3 + (Real.cos x)^3 = 3

-- State the theorem
theorem sin_pow_cos_pow_sum (x : ℝ) (h : trig_condition x) : Real.sin x ^ 2018 + Real.cos x ^ 2018 = 1 :=
by
  sorry

end ProofProblem

end sin_pow_cos_pow_sum_l143_143733


namespace area_of_triangle_le_one_fourth_l143_143405

open Real

noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_of_triangle_le_one_fourth (t : ℝ) (x y : ℝ) (h_t : 0 < t ∧ t < 1) (h_x : 0 ≤ x ∧ x ≤ 1)
  (h_y : y = t * (2 * x - t)) :
  area_triangle t (t^2) 1 0 x y ≤ 1 / 4 :=
by
  sorry

end area_of_triangle_le_one_fourth_l143_143405


namespace power_function_propositions_l143_143182

theorem power_function_propositions : (∀ n : ℤ, n > 0 → ∀ x : ℝ, x > 0 → (x^n) < x) ∧
  (∀ n : ℤ, n < 0 → ∀ x : ℝ, x > 0 → (x^n) > x) :=
by
  sorry

end power_function_propositions_l143_143182


namespace jack_further_down_l143_143917

-- Define the conditions given in the problem
def flights_up := 3
def flights_down := 6
def steps_per_flight := 12
def height_per_step_in_inches := 8
def inches_per_foot := 12

-- Define the number of steps and height calculations
def steps_up := flights_up * steps_per_flight
def steps_down := flights_down * steps_per_flight
def net_steps_down := steps_down - steps_up
def net_height_down_in_inches := net_steps_down * height_per_step_in_inches
def net_height_down_in_feet := net_height_down_in_inches / inches_per_foot

-- The proof statement to be shown
theorem jack_further_down : net_height_down_in_feet = 24 := sorry

end jack_further_down_l143_143917


namespace cos_arcsin_eq_l143_143467

theorem cos_arcsin_eq : ∀ (x : ℝ), x = 8 / 17 → cos (arcsin x) = 15 / 17 :=
by 
  intro x hx
  have h1 : θ = arcsin x := sorry -- by definition θ = arcsin x
  have h2 : sin θ = x := sorry -- by definition sin θ = x
  have h3 : (17:ℝ)^2 = a^2 + 8^2 := sorry -- Pythagorean theorem
  have h4 : a = 15 := sorry -- solved from h3
  show cos (arcsin x) = 15 / 17 := sorry -- proven from h2 and h4

end cos_arcsin_eq_l143_143467


namespace farmer_cows_after_selling_l143_143025

theorem farmer_cows_after_selling
  (initial_cows : ℕ) (new_cows : ℕ) (quarter_factor : ℕ)
  (h_initial : initial_cows = 51)
  (h_new : new_cows = 5)
  (h_quarter : quarter_factor = 4) :
  initial_cows + new_cows - (initial_cows + new_cows) / quarter_factor = 42 :=
by
  sorry

end farmer_cows_after_selling_l143_143025


namespace overall_labor_costs_l143_143402

noncomputable def construction_worker_daily_wage : ℝ := 100
noncomputable def electrician_daily_wage : ℝ := 2 * construction_worker_daily_wage
noncomputable def plumber_daily_wage : ℝ := 2.5 * construction_worker_daily_wage

noncomputable def total_construction_work : ℝ := 2 * construction_worker_daily_wage
noncomputable def total_electrician_work : ℝ := electrician_daily_wage
noncomputable def total_plumber_work : ℝ := plumber_daily_wage

theorem overall_labor_costs :
  total_construction_work + total_electrician_work + total_plumber_work = 650 :=
by
  sorry

end overall_labor_costs_l143_143402


namespace pyramid_volume_l143_143278

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1 / 3) * a * b * c * Real.sqrt 2

theorem pyramid_volume (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (A1 : ∃ x y, 1 / 2 * x * y = a^2) 
  (A2 : ∃ y z, 1 / 2 * y * z = b^2) 
  (A3 : ∃ x z, 1 / 2 * x * z = c^2)
  (h_perpendicular : True) :
  volume_of_pyramid a b c = (1 / 3) * a * b * c * Real.sqrt 2 :=
sorry

end pyramid_volume_l143_143278


namespace square_diagonal_cut_l143_143319

/--
Given a square with side length 10,
prove that cutting along the diagonal results in two 
right-angled isosceles triangles with dimensions 10, 10, 10*sqrt(2).
-/
theorem square_diagonal_cut (side_length : ℕ) (triangle_side1 triangle_side2 hypotenuse : ℝ) 
  (h_side : side_length = 10)
  (h_triangle_side1 : triangle_side1 = 10) 
  (h_triangle_side2 : triangle_side2 = 10)
  (h_hypotenuse : hypotenuse = 10 * Real.sqrt 2) : 
  triangle_side1 = side_length ∧ triangle_side2 = side_length ∧ hypotenuse = side_length * Real.sqrt 2 :=
by
  sorry

end square_diagonal_cut_l143_143319


namespace find_value_of_x_squared_and_reciprocal_squared_l143_143086

theorem find_value_of_x_squared_and_reciprocal_squared (x : ℝ) (h : x + 1/x = 2) : x^2 + (1/x)^2 = 2 := 
sorry

end find_value_of_x_squared_and_reciprocal_squared_l143_143086


namespace arithmetic_mean_of_fractions_l143_143978

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((2 / 5) + (4 / 7)) = 17 / 35 :=
by
  sorry

end arithmetic_mean_of_fractions_l143_143978


namespace three_digit_number_digits_difference_l143_143320

theorem three_digit_number_digits_difference (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : a < b) (h4 : b < c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  reversed_number - original_number = 198 := by
  sorry

end three_digit_number_digits_difference_l143_143320


namespace sin_half_inequality_l143_143777

theorem sin_half_inequality (α β γ : ℝ) : 
  1 - Real.sin (α / 2) ≥ 2 * Real.sin (β / 2) * Real.sin (γ / 2) :=
sorry

end sin_half_inequality_l143_143777


namespace hyperbola_eccentricity_range_l143_143371

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  (∃ P₁ P₂ : { p : ℝ × ℝ // p ≠ (0, b) ∧ p ≠ (c, 0) ∧ ((0, b) - p).1 * ((c, 0) - p).1 + ((0, b) - p).2 * ((c, 0) - p).2 = 0},
   true) -- This encodes the existence of the required points P₁ and P₂ on line segment BF excluding endpoints
  → 1 < (Real.sqrt ((a^2 + b^2) / a^2)) ∧ (Real.sqrt ((a^2 + b^2) / a^2)) < (Real.sqrt 5 + 1)/2 :=
sorry

end hyperbola_eccentricity_range_l143_143371


namespace greatest_nat_not_sum_of_two_composites_l143_143647

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

theorem greatest_nat_not_sum_of_two_composites :
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b ∧
  (∀ n : ℕ, n > 11 → ¬ ∃ x y : ℕ, is_composite x ∧ is_composite y ∧ n = x + y) :=
sorry

end greatest_nat_not_sum_of_two_composites_l143_143647


namespace problem_1_problem_2_problem_3_l143_143014

def range_1 : Set ℝ :=
  { y | ∃ x : ℝ, y = 1 / (x - 1) ∧ x ≠ 1 }

def range_2 : Set ℝ :=
  { y | ∃ x : ℝ, y = x^2 + 4 * x - 1 }

def range_3 : Set ℝ :=
  { y | ∃ x : ℝ, y = x + Real.sqrt (x + 1) ∧ x ≥ 0 }

theorem problem_1 : range_1 = {y | y < 0 ∨ y > 0} :=
by 
  sorry

theorem problem_2 : range_2 = {y | y ≥ -5} :=
by 
  sorry

theorem problem_3 : range_3 = {y | y ≥ -1} :=
by 
  sorry

end problem_1_problem_2_problem_3_l143_143014


namespace megan_removed_albums_l143_143526

theorem megan_removed_albums :
  ∀ (albums_in_cart : ℕ) (songs_per_album : ℕ) (total_songs_bought : ℕ),
    albums_in_cart = 8 →
    songs_per_album = 7 →
    total_songs_bought = 42 →
    albums_in_cart - (total_songs_bought / songs_per_album) = 2 :=
by
  intros albums_in_cart songs_per_album total_songs_bought h1 h2 h3
  sorry

end megan_removed_albums_l143_143526


namespace find_fayes_age_l143_143632

variable {C D E F : ℕ}

theorem find_fayes_age
  (h1 : D = E - 2)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (h4 : D = 15) :
  F = 16 := by
  sorry

end find_fayes_age_l143_143632


namespace novels_at_both_ends_l143_143233

noncomputable def arrangements_where_novels_at_both_ends (novels : ℕ) (other_books : ℕ) : ℕ :=
  let total_books := novels + other_books
  if h : novels = 2 ∧ other_books = 3 then
    (factorial 3) * 2
  else
    0

theorem novels_at_both_ends : arrangements_where_novels_at_both_ends 2 3 = 12 := 
  by
  sorry

end novels_at_both_ends_l143_143233


namespace minimum_toys_to_add_l143_143994

theorem minimum_toys_to_add {T : ℤ} (k m n : ℤ) (h1 : T = 12 * k + 3) (h2 : T = 18 * m + 3) 
  (h3 : T = 36 * n + 3) : 
  ∃ x : ℤ, (T + x) % 7 = 0 ∧ x = 4 :=
sorry

end minimum_toys_to_add_l143_143994


namespace largest_fraction_l143_143988

theorem largest_fraction :
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  sorry

end largest_fraction_l143_143988


namespace matrix_A_to_power_4_l143_143340

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l143_143340


namespace largest_integer_less_than_100_with_remainder_5_l143_143652

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143652


namespace age_difference_l143_143968

theorem age_difference (x : ℕ) 
  (h_ratio : 4 * x + 3 * x + 7 * x = 126)
  (h_halima : 4 * x = 36)
  (h_beckham : 3 * x = 27) :
  4 * x - 3 * x = 9 :=
by sorry

end age_difference_l143_143968


namespace largest_int_with_remainder_5_lt_100_l143_143683

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l143_143683


namespace negation_of_p_l143_143776

theorem negation_of_p : 
  (¬(∀ x : ℝ, |x| < 0)) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by {
  sorry
}

end negation_of_p_l143_143776


namespace tea_bags_l143_143938

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l143_143938


namespace largest_integer_less_than_hundred_with_remainder_five_l143_143674

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l143_143674


namespace points_needed_for_office_l143_143936

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25

def jerry_interruptions : ℕ := 2
def jerry_insults : ℕ := 4
def jerry_throwings : ℕ := 2

def jerry_total_points (interrupt_points insult_points throw_points : ℕ) 
                       (interruptions insults throwings : ℕ) : ℕ :=
  (interrupt_points * interruptions) +
  (insult_points * insults) +
  (throw_points * throwings)

theorem points_needed_for_office : 
  jerry_total_points points_for_interrupting points_for_insulting points_for_throwing 
                     (jerry_interruptions) 
                     (jerry_insults) 
                     (jerry_throwings) = 100 := 
  sorry

end points_needed_for_office_l143_143936


namespace general_formula_T30_sum_l143_143498

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Given conditions for the arithmetic sequence
axiom h1 : a 1 + a 4 + a 7 = -24
axiom h2 : a 2 + a 5 + a 8 = -15
axiom h3 : ∀ n : ℕ, a (n + 1) = a n + d

-- Part 1: Prove the general formula for aₙ
theorem general_formula :
  ∀ n, a n = 3 * n - 20 := by
  sorry

-- Part 2: Prove T₃₀ = 909, where T₃₀ is the sum of the first 30 terms of |aₙ|
theorem T30_sum :
  (∑ i in Finset.range 30, abs (a i)) = 909 := by
  sorry

end general_formula_T30_sum_l143_143498


namespace largest_integer_less_than_100_with_remainder_5_l143_143687

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l143_143687


namespace doubled_container_volume_l143_143611

theorem doubled_container_volume (v : ℝ) (h₁ : v = 4) (h₂ : ∀ l w h : ℝ, v = l * w * h) : 8 * v = 32 := 
by
  -- The proof will go here, this is just the statement
  sorry

end doubled_container_volume_l143_143611


namespace find_cost_price_l143_143312

theorem find_cost_price (C : ℝ) (h1 : C * 1.05 = C + 0.05 * C)
  (h2 : 0.95 * C = C - 0.05 * C)
  (h3 : 1.05 * C - 4 = 1.045 * C) :
  C = 800 := sorry

end find_cost_price_l143_143312


namespace xyz_solution_l143_143767

theorem xyz_solution (x y z : ℂ) (h1 : x * y + 5 * y = -20) 
                                 (h2 : y * z + 5 * z = -20) 
                                 (h3 : z * x + 5 * x = -20) :
  x * y * z = 200 / 3 := 
sorry

end xyz_solution_l143_143767


namespace pump_B_time_l143_143297

theorem pump_B_time (T_B : ℝ) (h1 : ∀ (h1 : T_B > 0),
  (1 / 4 + 1 / T_B = 3 / 4)) :
  T_B = 2 := 
by
  sorry

end pump_B_time_l143_143297


namespace problem_l143_143485

noncomputable def d : ℝ := -8.63

theorem problem :
  let floor_d := ⌊d⌋
  let frac_d := d - floor_d
  (3 * floor_d^2 + 20 * floor_d - 67 = 0) ∧
  (4 * frac_d^2 - 15 * frac_d + 5 = 0) → 
  d = -8.63 :=
by {
  sorry
}

end problem_l143_143485


namespace apples_distribution_l143_143998

theorem apples_distribution (total_apples : ℕ) (rotten_apples : ℕ) (boxes : ℕ) (remaining_apples : ℕ) (apples_per_box : ℕ) :
  total_apples = 40 →
  rotten_apples = 4 →
  boxes = 4 →
  remaining_apples = total_apples - rotten_apples →
  apples_per_box = remaining_apples / boxes →
  apples_per_box = 9 :=
by
  intros
  sorry

end apples_distribution_l143_143998


namespace largest_integer_less_than_100_with_remainder_5_l143_143677

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143677


namespace most_accurate_method_is_independence_test_l143_143160

-- Definitions and assumptions
inductive Methods
| contingency_table
| independence_test
| stacked_bar_chart
| others

def related_or_independent_method : Methods := Methods.independence_test

-- Proof statement
theorem most_accurate_method_is_independence_test :
  related_or_independent_method = Methods.independence_test :=
sorry

end most_accurate_method_is_independence_test_l143_143160


namespace largest_integer_lt_100_with_remainder_5_div_8_l143_143663

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l143_143663


namespace intercepts_equal_l143_143886

theorem intercepts_equal (a : ℝ) (ha : (a ≠ 0) ∧ (a ≠ 2)) : 
  (a = 1 ∨ a = 2) ↔ (a = 1 ∨ a = 2) := 
by 
  sorry


end intercepts_equal_l143_143886


namespace contrapositive_proposition_l143_143126

-- Define the necessary elements in the context of real numbers
variables {a b c d : ℝ}

-- The statement of the contrapositive
theorem contrapositive_proposition : (a + c ≠ b + d) → (a ≠ b ∨ c ≠ d) :=
sorry

end contrapositive_proposition_l143_143126


namespace tea_bags_count_l143_143944

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l143_143944


namespace largest_integer_less_than_100_with_remainder_5_l143_143699

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l143_143699


namespace relationship_among_abc_l143_143116

theorem relationship_among_abc {f : ℝ → ℝ} (hf : ∀ x > 0, x * deriv f x - f x < 0) :
  let a := f ((Real.log 5) / (Real.log 2)) / ((Real.log 5) / (Real.log 2)),
      b := f (2 ^ 0.2) / (2 ^ 0.2),
      c := f (0.2 ^ 2) / (0.2 ^ 2) in a < b ∧ b < c :=
by
  sorry

end relationship_among_abc_l143_143116


namespace placing_2_flowers_in_2_vases_l143_143437

noncomputable def num_ways_to_place_flowers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : ℕ :=
  Nat.choose n k * 2

theorem placing_2_flowers_in_2_vases :
  num_ways_to_place_flowers 5 2 rfl rfl = 20 := 
by
  sorry

end placing_2_flowers_in_2_vases_l143_143437


namespace evaluate_expression_l143_143639

theorem evaluate_expression :
  (3 + 6 + 9) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 6 + 9) = 5 / 6 :=
by sorry

end evaluate_expression_l143_143639


namespace problem_statement_l143_143083

theorem problem_statement (x y z : ℝ) 
  (h1 : x + y + z = 6) 
  (h2 : x * y + y * z + z * x = 11) 
  (h3 : x * y * z = 6) : 
  x / (y * z) + y / (z * x) + z / (x * y) = 7 / 3 := 
sorry

end problem_statement_l143_143083


namespace matrix_power_four_l143_143338

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l143_143338


namespace neither_directly_nor_inversely_proportional_A_D_l143_143100

-- Definitions for the equations where y is neither directly nor inversely proportional to x
def equationA (x y : ℝ) : Prop := x^2 + x * y = 0
def equationD (x y : ℝ) : Prop := 4 * x + y^2 = 7

-- Definition for direct or inverse proportionality
def isDirectlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x
def isInverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

-- Proposition that y is neither directly nor inversely proportional to x for equations A and D
theorem neither_directly_nor_inversely_proportional_A_D (x y : ℝ) :
  equationA x y ∧ equationD x y ∧ ¬isDirectlyProportional x y ∧ ¬isInverselyProportional x y :=
by sorry

end neither_directly_nor_inversely_proportional_A_D_l143_143100


namespace gunther_typing_l143_143080

theorem gunther_typing : 
  (∀ (number_of_words : ℕ) (minutes_per_set : ℕ) (total_working_minutes : ℕ),
    number_of_words = 160 → minutes_per_set = 3 → total_working_minutes = 480 →
    (total_working_minutes / minutes_per_set * number_of_words) = 25600) :=
begin
  intros number_of_words minutes_per_set total_working_minutes,
  intros h_words h_time h_total,
  rw [h_words, h_time, h_total],
  norm_num,
end

end gunther_typing_l143_143080


namespace phyllis_marbles_l143_143264

theorem phyllis_marbles (num_groups : ℕ) (num_marbles_per_group : ℕ) (h1 : num_groups = 32) (h2 : num_marbles_per_group = 2) : 
  num_groups * num_marbles_per_group = 64 :=
by
  sorry

end phyllis_marbles_l143_143264


namespace largest_integer_less_than_100_div_8_rem_5_l143_143661

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l143_143661


namespace calculate_selling_prices_l143_143026

noncomputable def selling_prices
  (cost1 cost2 cost3 : ℝ) (profit1 profit2 profit3 : ℝ) : ℝ × ℝ × ℝ :=
  let selling_price1 := cost1 + (profit1 / 100) * cost1
  let selling_price2 := cost2 + (profit2 / 100) * cost2
  let selling_price3 := cost3 + (profit3 / 100) * cost3
  (selling_price1, selling_price2, selling_price3)

theorem calculate_selling_prices :
  selling_prices 500 750 1000 20 25 30 = (600, 937.5, 1300) :=
by
  sorry

end calculate_selling_prices_l143_143026


namespace part1_part2_l143_143218

open Set

-- Define the sets M and N based on given conditions
def M (a : ℝ) : Set ℝ := { x | (x + a) * (x - 1) ≤ 0 }
def N : Set ℝ := { x | 4 * x^2 - 4 * x - 3 < 0 }

-- Part (1): Prove that if M ∪ N = { x | -2 ≤ x < 3 / 2 }, then a = 2
theorem part1 (a : ℝ) (h : a > 0)
  (h_union : M a ∪ N = { x | -2 ≤ x ∧ x < 3 / 2 }) : a = 2 := by
  sorry

-- Part (2): Prove that if N ∪ (compl (M a)) = univ, then 0 < a ≤ 1/2
theorem part2 (a : ℝ) (h : a > 0)
  (h_union : N ∪ compl (M a) = univ) : 0 < a ∧ a ≤ 1 / 2 := by
  sorry

end part1_part2_l143_143218


namespace dice_sum_prime_probability_l143_143387

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roll_dice_prob_prime : ℚ :=
  let total_outcomes := 6^7
  let prime_sums := [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
  let P := 80425 -- Assume pre-computed sum counts based on primes
  (P : ℚ) / total_outcomes

theorem dice_sum_prime_probability :
  roll_dice_prob_prime = 26875 / 93312 :=
by
  sorry

end dice_sum_prime_probability_l143_143387


namespace binomial_12_10_eq_66_l143_143348

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l143_143348


namespace _l143_143127

noncomputable def polynomial_divides (x : ℂ) (n : ℕ) : Prop :=
  (x - 1) ^ 3 ∣ x ^ (2 * n + 1) - (2 * n + 1) * x ^ (n + 1) + (2 * n + 1) * x ^ n - 1

lemma polynomial_division_theorem : ∀ (n : ℕ), n ≥ 1 → ∀ (x : ℂ), polynomial_divides x n :=
by
  intros n hn x
  unfold polynomial_divides
  sorry

end _l143_143127


namespace count_integers_between_sqrt10_sqrt100_l143_143728

theorem count_integers_between_sqrt10_sqrt100 : 
  ∃ (S : Set ℤ), (∀ n, n ∈ S ↔ (real.sqrt 10 < n ∧ n < real.sqrt 100)) ∧ S.card = 6 := 
by
  sorry

end count_integers_between_sqrt10_sqrt100_l143_143728


namespace prove_interest_rates_equal_l143_143752

noncomputable def interest_rates_equal : Prop :=
  let initial_savings := 1000
  let savings_simple := initial_savings / 2
  let savings_compound := initial_savings / 2
  let simple_interest_earned := 100
  let compound_interest_earned := 105
  let time := 2
  let r_s := simple_interest_earned / (savings_simple * time)
  let r_c := (compound_interest_earned / savings_compound + 1) ^ (1 / time) - 1
  r_s = r_c

theorem prove_interest_rates_equal : interest_rates_equal :=
  sorry

end prove_interest_rates_equal_l143_143752


namespace largest_integer_less_than_hundred_with_remainder_five_l143_143672

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l143_143672


namespace matrix_power_four_correct_l143_143333

theorem matrix_power_four_correct :
  let A := Matrix.of (fun i j => ![![2, -1], ![1, 1]].get i j) in
  A ^ 4 = Matrix.of (fun i j => ![![0, -9], ![9, -9]].get i j) :=
by
  sorry

end matrix_power_four_correct_l143_143333


namespace book_distribution_l143_143565

/-- There are 6 different books. -/
def books : ℕ := 6

/-- There are three individuals: A, B, and C. -/
def individuals : ℕ := 3

/-- Each individual receives exactly 2 books. -/
def books_per_individual : ℕ := 2

/-- The number of distinct ways to distribute the books is 90. -/
theorem book_distribution :
  (Nat.choose books books_per_individual) * 
  (Nat.choose (books - books_per_individual) books_per_individual) = 90 :=
by
  sorry

end book_distribution_l143_143565


namespace spherical_to_rectangular_conversion_l143_143192

theorem spherical_to_rectangular_conversion :
  ∃ x y z : ℝ, 
    x = -Real.sqrt 2 ∧ 
    y = 0 ∧ 
    z = Real.sqrt 2 ∧ 
    (∃ rho theta phi : ℝ, 
      rho = 2 ∧
      theta = π ∧
      phi = π/4 ∧
      x = rho * Real.sin phi * Real.cos theta ∧
      y = rho * Real.sin phi * Real.sin theta ∧
      z = rho * Real.cos phi) :=
by
  sorry

end spherical_to_rectangular_conversion_l143_143192


namespace binom_two_eq_l143_143811

theorem binom_two_eq (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_eq_l143_143811


namespace camels_in_caravan_l143_143911

theorem camels_in_caravan : 
  ∃ (C : ℕ), 
  (60 + 35 + 10 + C) * 1 + 60 * 2 + 35 * 4 + 10 * 2 + 4 * C - (60 + 35 + 10 + C) = 193 ∧ 
  C = 6 :=
by
  sorry

end camels_in_caravan_l143_143911


namespace find_abc_l143_143226

theorem find_abc (a b c : ℝ) (ha : a + 1 / b = 5)
                             (hb : b + 1 / c = 2)
                             (hc : c + 1 / a = 3) :
    a * b * c = 10 + 3 * Real.sqrt 11 :=
sorry

end find_abc_l143_143226


namespace find_g_four_l143_143553

theorem find_g_four (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 11 / 2 := 
by
  sorry

end find_g_four_l143_143553


namespace lauren_total_earnings_l143_143528

-- Define earnings conditions
def mondayCommercialEarnings (views : ℕ) : ℝ := views * 0.40
def mondaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 0.80

def tuesdayCommercialEarnings (views : ℕ) : ℝ := views * 0.50
def tuesdaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 1.00

def weekendMerchandiseEarnings (sales : ℝ) : ℝ := 0.10 * sales

-- Specific conditions for each day
def mondayTotalEarnings : ℝ := mondayCommercialEarnings 80 + mondaySubscriptionEarnings 20
def tuesdayTotalEarnings : ℝ := tuesdayCommercialEarnings 100 + tuesdaySubscriptionEarnings 27
def weekendTotalEarnings : ℝ := weekendMerchandiseEarnings 150

-- Total earnings for the period
def totalEarnings : ℝ := mondayTotalEarnings + tuesdayTotalEarnings + weekendTotalEarnings

-- Examining the final value
theorem lauren_total_earnings : totalEarnings = 140.00 := by
  sorry

end lauren_total_earnings_l143_143528


namespace parity_of_expression_l143_143925

theorem parity_of_expression (e m : ℕ) (he : (∃ k : ℕ, e = 2 * k)) : Odd (e ^ 2 + 3 ^ m) :=
  sorry

end parity_of_expression_l143_143925


namespace questionnaires_drawn_from_D_l143_143618

theorem questionnaires_drawn_from_D (a b c d : ℕ) (A_s B_s C_s D_s: ℕ) (common_diff: ℕ)
  (h1 : a + b + c + d = 1000)
  (h2 : b = a + common_diff)
  (h3 : c = a + 2 * common_diff)
  (h4 : d = a + 3 * common_diff)
  (h5 : A_s = 30 - common_diff)
  (h6 : B_s = 30)
  (h7 : C_s = 30 + common_diff)
  (h8 : D_s = 30 + 2 * common_diff)
  (h9 : A_s + B_s + C_s + D_s = 150)
  : D_s = 60 := sorry

end questionnaires_drawn_from_D_l143_143618


namespace nicolai_peaches_6_pounds_l143_143144

noncomputable def amount_peaches (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ) : ℕ :=
  let total_ounces := total_pounds * 16
  let total_consumed := oz_oranges + oz_apples
  let remaining_ounces := total_ounces - total_consumed
  remaining_ounces / 16

theorem nicolai_peaches_6_pounds (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ)
  (h_total_pounds : total_pounds = 8) (h_oz_oranges : oz_oranges = 8) (h_oz_apples : oz_apples = 24) :
  amount_peaches total_pounds oz_oranges oz_apples = 6 :=
by
  rw [h_total_pounds, h_oz_oranges, h_oz_apples]
  unfold amount_peaches
  sorry

end nicolai_peaches_6_pounds_l143_143144


namespace binom_two_eq_n_choose_2_l143_143813

theorem binom_two_eq_n_choose_2 (n : ℕ) (h : n ≥ 2) :
  (nat.choose n 2) = (n * (n - 1)) / 2 := by
  sorry

end binom_two_eq_n_choose_2_l143_143813


namespace table_mat_length_l143_143452

noncomputable def calculate_y (r : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / n
  let y_side := 2 * r * Real.sin (θ / 2)
  y_side

theorem table_mat_length :
  calculate_y 6 8 1 = 3 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end table_mat_length_l143_143452


namespace angle_P_measure_l143_143324

theorem angle_P_measure (P Q : ℝ) (h1 : P + Q = 180) (h2 : P = 5 * Q) : P = 150 := by
  sorry

end angle_P_measure_l143_143324


namespace largest_int_with_remainder_5_lt_100_l143_143685

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l143_143685


namespace probability_factor_90_less_than_10_l143_143814

-- Definitions from conditions
def number_factors_90 : ℕ := 12
def factors_90_less_than_10 : ℕ := 6

-- The corresponding proof problem
theorem probability_factor_90_less_than_10 : 
  (factors_90_less_than_10 / number_factors_90 : ℚ) = 1 / 2 :=
by
  sorry  -- proof to be filled in

end probability_factor_90_less_than_10_l143_143814


namespace volume_of_regular_triangular_pyramid_l143_143364

noncomputable def pyramid_volume (a b γ : ℝ) : ℝ :=
  (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2)

theorem volume_of_regular_triangular_pyramid (a b γ : ℝ) :
  pyramid_volume a b γ = (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2) :=
by
  sorry

end volume_of_regular_triangular_pyramid_l143_143364


namespace three_integers_sum_of_consecutive_odds_l143_143898

theorem three_integers_sum_of_consecutive_odds :
  {N : ℕ | N ≤ 500 ∧ (∃ j n, N = j * (2 * n + j) ∧ j ≥ 1) ∧
                   (∃! j1 j2 j3, ∃ n1 n2 n3, N = j1 * (2 * n1 + j1) ∧ N = j2 * (2 * n2 + j2) ∧ N = j3 * (2 * n3 + j3) ∧ j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3)} = {16, 18, 50} :=
by
  sorry

end three_integers_sum_of_consecutive_odds_l143_143898


namespace max_value_of_S_l143_143210

-- Define the sequence sum function
def S (n : ℕ) : ℤ :=
  -2 * (n : ℤ) ^ 3 + 21 * (n : ℤ) ^ 2 + 23 * (n : ℤ)

theorem max_value_of_S :
  ∃ (n : ℕ), S n = 504 ∧ 
             (∀ k : ℕ, S k ≤ 504) :=
sorry

end max_value_of_S_l143_143210


namespace price_each_clock_is_correct_l143_143544

-- Definitions based on the conditions
def numberOfDolls := 3
def numberOfClocks := 2
def numberOfGlasses := 5
def pricePerDoll := 5
def pricePerGlass := 4
def totalCost := 40
def profit := 25

-- The total revenue from selling dolls and glasses
def revenueFromDolls := numberOfDolls * pricePerDoll
def revenueFromGlasses := numberOfGlasses * pricePerGlass
def totalRevenueNeeded := totalCost + profit
def revenueFromDollsAndGlasses := revenueFromDolls + revenueFromGlasses

-- The required revenue from clocks
def revenueFromClocks := totalRevenueNeeded - revenueFromDollsAndGlasses

-- The price per clock
def pricePerClock := revenueFromClocks / numberOfClocks

-- Statement to prove
theorem price_each_clock_is_correct : pricePerClock = 15 := sorry

end price_each_clock_is_correct_l143_143544


namespace false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l143_143162

-- Proposition A
theorem false_proposition_A (a b c : ℝ) (hac : a > b) (hca : b > 0) : ac * c^2 = b * c^2 :=
  sorry

-- Proposition B
theorem false_proposition_B (a b : ℝ) (hab : a < b) : (1/a) < (1/b) :=
  sorry

-- Proposition C
theorem true_proposition_C (a b : ℝ) (hab : a > b) (hba : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
  sorry

-- Proposition D
theorem true_proposition_D (a b : ℝ) (hba : a > |b|) : a^2 > b^2 :=
  sorry

end false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l143_143162


namespace graph_quadrant_l143_143137

theorem graph_quadrant (x y : ℝ) : 
  y = 3 * x - 4 → ¬ ((x < 0) ∧ (y > 0)) :=
by
  intro h
  sorry

end graph_quadrant_l143_143137


namespace smallest_interesting_number_l143_143828

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l143_143828


namespace min_value_of_quadratic_l143_143288

theorem min_value_of_quadratic :
  ∀ x : ℝ, ∃ z : ℝ, z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∀ z' : ℝ, (z' = 4 * x^2 + 8 * x + 16) → z' ≥ 12) :=
by
  sorry

end min_value_of_quadratic_l143_143288


namespace probability_purple_or_orange_face_l143_143267

theorem probability_purple_or_orange_face 
  (total_faces : ℕ) (green_faces : ℕ) (purple_faces : ℕ) (orange_faces : ℕ) 
  (h_total : total_faces = 10) 
  (h_green : green_faces = 5) 
  (h_purple : purple_faces = 3) 
  (h_orange : orange_faces = 2) :
  (purple_faces + orange_faces) / total_faces = 1 / 2 :=
by 
  sorry

end probability_purple_or_orange_face_l143_143267


namespace exists_nat_expressed_as_sum_of_powers_l143_143881

theorem exists_nat_expressed_as_sum_of_powers 
  (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : ℕ, (∀ p ∈ P, ∃ a b : ℕ, x = a^p + b^p) ∧ (∀ p : ℕ, Nat.Prime p → p ∉ P → ¬∃ a b : ℕ, x = a^p + b^p) :=
by
  let x := 2^(P.val.prod + 1)
  use x
  sorry

end exists_nat_expressed_as_sum_of_powers_l143_143881


namespace largest_integer_less_than_100_with_remainder_5_l143_143675

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143675


namespace sequence_accumulating_is_arithmetic_l143_143759

noncomputable def arithmetic_sequence {α : Type*} [LinearOrderedField α]
  (a : ℕ → α) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem sequence_accumulating_is_arithmetic
  {α : Type*} [LinearOrderedField α] (a : ℕ → α) (S : ℕ → α)
  (na_gt_zero : ∀ n, a n > 0)
  (ha2 : a 2 = 3 * a 1)
  (hS_arith : arithmetic_sequence (λ n, (S n)^(1/2)))
  (hSn : ∀ n, S n = (∑ i in Finset.range (n+1), a i)) :
  arithmetic_sequence a := 
sorry

end sequence_accumulating_is_arithmetic_l143_143759


namespace smaller_triangle_perimeter_l143_143751

theorem smaller_triangle_perimeter (p : ℕ) (h : p * 3 = 120) : p = 40 :=
sorry

end smaller_triangle_perimeter_l143_143751


namespace number_of_customers_left_l143_143459

theorem number_of_customers_left (x : ℕ) (h : 14 - x + 39 = 50) : x = 3 := by
  sorry

end number_of_customers_left_l143_143459


namespace largest_integer_less_than_100_with_remainder_5_l143_143689

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l143_143689


namespace min_sum_of_exponents_of_powers_of_2_l143_143384

theorem min_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ (s : set ℕ), (∀ (k ∈ s), ∃ (m : ℕ), k = 2 ^ m) ∧ (s.sum id = 520) ∧ (s.sum id = s.card * s.card) → (s.sum id = 12) := sorry

end min_sum_of_exponents_of_powers_of_2_l143_143384


namespace items_count_l143_143180

variable (N : ℕ)

-- Conditions
def item_price : ℕ := 50
def discount_rate : ℕ := 80
def sell_percentage : ℕ := 90
def creditors_owed : ℕ := 15000
def money_left : ℕ := 3000

-- Definitions based on the conditions
def sale_price : ℕ := (item_price * (100 - discount_rate)) / 100
def money_before_paying_creditors : ℕ := money_left + creditors_owed
def total_revenue (N : ℕ) : ℕ := (sell_percentage * N * sale_price) / 100

-- Problem statement
theorem items_count : total_revenue N = money_before_paying_creditors → N = 2000 := by
  intros h
  sorry

end items_count_l143_143180


namespace modified_expression_range_l143_143708

open Int

theorem modified_expression_range (m : ℤ) :
  ∃ n_min n_max : ℤ, 1 < 4 * n_max + 7 ∧ 4 * n_min + 7 < 60 ∧ (n_max - n_min + 1 = 15) →
  ∃ k_min k_max : ℤ, 1 < m * k_max + 7 ∧ m * k_min + 7 < 60 ∧ (k_max - k_min + 1 ≥ 15) := 
sorry

end modified_expression_range_l143_143708


namespace probability_mixed_doubles_l143_143438

def num_athletes : ℕ := 6
def num_males : ℕ := 3
def num_females : ℕ := 3
def num_coaches : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select athletes
def total_ways : ℕ :=
  (choose num_athletes 2) * (choose (num_athletes - 2) 2) * (choose (num_athletes - 4) 2)

-- Number of favorable ways to select mixed doubles teams
def favorable_ways : ℕ :=
  (choose num_males 1) * (choose num_females 1) *
  (choose (num_males - 1) 1) * (choose (num_females - 1) 1) *
  (choose 1 1) * (choose 1 1)

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

theorem probability_mixed_doubles :
  probability = 2/5 :=
by
  sorry

end probability_mixed_doubles_l143_143438


namespace log_fraction_eq_l143_143900

variable (a b : ℝ)
axiom h1 : a = Real.logb 3 5
axiom h2 : b = Real.logb 5 7

theorem log_fraction_eq : Real.logb 15 (49 / 45) = (2 * (a * b) - a - 2) / (1 + a) :=
by sorry

end log_fraction_eq_l143_143900


namespace rick_gives_miguel_cards_l143_143778

/-- Rick starts with 130 cards, keeps 15 cards for himself, gives 
12 cards each to 8 friends, and gives 3 cards each to his 2 sisters. 
We need to prove that Rick gives 13 cards to Miguel. --/
theorem rick_gives_miguel_cards :
  let initial_cards := 130
  let kept_cards := 15
  let friends := 8
  let cards_per_friend := 12
  let sisters := 2
  let cards_per_sister := 3
  initial_cards - kept_cards - (friends * cards_per_friend) - (sisters * cards_per_sister) = 13 :=
by
  sorry

end rick_gives_miguel_cards_l143_143778


namespace omar_total_time_l143_143120

-- Conditions
def lap_distance : ℝ := 400
def first_segment_distance : ℝ := 200
def second_segment_distance : ℝ := 200
def speed_first_segment : ℝ := 6
def speed_second_segment : ℝ := 4
def number_of_laps : ℝ := 7

-- Correct answer we want to prove
def total_time_proven : ℝ := 9 * 60 + 23 -- in seconds

-- Theorem statement claiming total time is 9 minutes and 23 seconds
theorem omar_total_time :
  let time_first_segment := first_segment_distance / speed_first_segment
  let time_second_segment := second_segment_distance / speed_second_segment
  let single_lap_time := time_first_segment + time_second_segment
  let total_time := number_of_laps * single_lap_time
  total_time = total_time_proven := sorry

end omar_total_time_l143_143120


namespace problem_statement_l143_143414

theorem problem_statement (x y : ℝ) (h₁ : x + y = 5) (h₂ : x * y = 3) : 
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := 
sorry

end problem_statement_l143_143414


namespace binom_12_10_eq_66_l143_143343

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l143_143343


namespace least_sum_of_exponents_520_l143_143385

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end least_sum_of_exponents_520_l143_143385


namespace relationship_of_coefficients_l143_143495

theorem relationship_of_coefficients (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0) 
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end relationship_of_coefficients_l143_143495


namespace minimum_abs_a_plus_b_l143_143377

theorem minimum_abs_a_plus_b {a b : ℤ} (h1 : |a| < |b|) (h2 : |b| ≤ 4) : ∃ (a b : ℤ), |a| + b = -4 :=
by
  sorry

end minimum_abs_a_plus_b_l143_143377


namespace evaluate_expression_l143_143483

def S (a b c : ℤ) := a + b + c

theorem evaluate_expression (a b c : ℤ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 18) :
  (144 * ((1 : ℚ) / b - (1 : ℚ) / c) + 196 * ((1 : ℚ) / c - (1 : ℚ) / a) + 324 * ((1 : ℚ) / a - (1 : ℚ) / b)) /
  (12 * ((1 : ℚ) / b - (1 : ℚ) / c) + 14 * ((1 : ℚ) / c - (1 : ℚ) / a) + 18 * ((1 : ℚ) / a - (1 : ℚ) / b)) = 44 := 
sorry

end evaluate_expression_l143_143483


namespace max_stamps_l143_143231

-- Definitions based on conditions
def price_of_stamp := 28 -- in cents
def total_money := 3600 -- in cents

-- The theorem statement
theorem max_stamps (price_of_stamp total_money : ℕ) : (total_money / price_of_stamp) = 128 := by
  sorry

end max_stamps_l143_143231


namespace find_245th_digit_in_decimal_rep_of_13_div_17_l143_143294

-- Definition of the repeating sequence for the fractional division
def repeating_sequence_13_div_17 : List Char := ['7', '6', '4', '7', '0', '5', '8', '8', '2', '3', '5', '2', '9', '4', '1', '1']

-- Period of the repeating sequence
def period : ℕ := 16

-- Function to find the n-th digit in a repeating sequence
def nth_digit_in_repeating_sequence (seq : List Char) (period : ℕ) (n : ℕ) : Char :=
  seq.get! ((n - 1) % period)

-- Hypothesis: The repeating sequence of 13/17 and its period
axiom repeating_sequence_period : repeating_sequence_13_div_17.length = period

-- The theorem to prove
theorem find_245th_digit_in_decimal_rep_of_13_div_17 : nth_digit_in_repeating_sequence repeating_sequence_13_div_17 period 245 = '7' := 
  by
  sorry

end find_245th_digit_in_decimal_rep_of_13_div_17_l143_143294


namespace kelly_single_shot_decrease_l143_143600

def kelly_salary_decrease (s : ℝ) : ℝ :=
  let first_cut := s * 0.92
  let second_cut := first_cut * 0.86
  let third_cut := second_cut * 0.82
  third_cut

theorem kelly_single_shot_decrease :
  let original_salary := 1.0 -- Assume original salary is 1 for percentage calculation
  let final_salary := kelly_salary_decrease original_salary
  (100 : ℝ) - (final_salary * 100) = 34.8056 :=
by
  sorry

end kelly_single_shot_decrease_l143_143600


namespace domain_of_f_l143_143961

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x - 2) / Real.log 3 - 1)

theorem domain_of_f :
  {x : ℝ | f x = f x} = {x : ℝ | 2 < x ∧ x ≠ 5} :=
by
  sorry

end domain_of_f_l143_143961


namespace divides_8x_7y_l143_143609

theorem divides_8x_7y (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end divides_8x_7y_l143_143609


namespace equalize_money_l143_143184

theorem equalize_money (ann_money : ℕ) (bill_money : ℕ) : 
  ann_money = 777 → 
  bill_money = 1111 → 
  ∃ x, bill_money - x = ann_money + x :=
by
  sorry

end equalize_money_l143_143184


namespace no_base6_digit_d_divisible_by_7_l143_143205

theorem no_base6_digit_d_divisible_by_7 : 
∀ d : ℕ, (d < 6) → ¬ (654 + 42 * d) % 7 = 0 :=
by
  intro d h
  -- Proof is omitted as requested
  sorry

end no_base6_digit_d_divisible_by_7_l143_143205


namespace sum_of_cubes_is_nine_l143_143594

def sum_of_cubes_of_consecutive_integers (n : ℤ) : ℤ :=
  n^3 + (n + 1)^3

theorem sum_of_cubes_is_nine :
  ∃ n : ℤ, sum_of_cubes_of_consecutive_integers n = 9 :=
by
  sorry

end sum_of_cubes_is_nine_l143_143594


namespace radhika_total_games_l143_143128

-- Define the conditions
def giftsOnChristmas := 12
def giftsOnBirthday := 8
def alreadyOwned := (giftsOnChristmas + giftsOnBirthday) / 2
def totalGifts := giftsOnChristmas + giftsOnBirthday
def expectedTotalGames := totalGifts + alreadyOwned

-- Define the proof statement
theorem radhika_total_games : 
  giftsOnChristmas = 12 ∧ giftsOnBirthday = 8 ∧ alreadyOwned = 10 
  ∧ totalGifts = 20 ∧ expectedTotalGames = 30 :=
by 
  sorry

end radhika_total_games_l143_143128


namespace binom_12_10_eq_66_l143_143341

theorem binom_12_10_eq_66 : (nat.choose 12 10) = 66 := by
  sorry

end binom_12_10_eq_66_l143_143341


namespace range_of_a_l143_143087

noncomputable def f (a x : ℝ) : ℝ := x^3 + x^2 - a * x - 4
noncomputable def f_derivative (a x : ℝ) : ℝ := 3 * x^2 + 2 * x - a

def has_exactly_one_extremum_in_interval (a : ℝ) : Prop :=
  (f_derivative a (-1)) * (f_derivative a 1) < 0

theorem range_of_a (a : ℝ) :
  has_exactly_one_extremum_in_interval a ↔ (1 < a ∧ a < 5) :=
sorry

end range_of_a_l143_143087


namespace opposite_of_neg_2022_eq_2022_l143_143557

-- Define what it means to find the opposite of a number
def opposite (n : Int) : Int := -n

-- State the theorem that needs to be proved
theorem opposite_of_neg_2022_eq_2022 : opposite (-2022) = 2022 :=
by
  -- Proof would go here but we skip it with sorry
  sorry

end opposite_of_neg_2022_eq_2022_l143_143557


namespace decimal_to_binary_correct_l143_143640

-- Define the decimal number
def decimal_number : ℕ := 25

-- Define the binary equivalent of 25
def binary_representation : ℕ := 0b11001

-- The condition indicating how the conversion is done
def is_binary_representation (decimal : ℕ) (binary : ℕ) : Prop :=
  -- Check if the binary representation matches the manual decomposition
  decimal = (binary / 2^4) * 2^4 + 
            ((binary % 2^4) / 2^3) * 2^3 + 
            (((binary % 2^4) % 2^3) / 2^2) * 2^2 + 
            ((((binary % 2^4) % 2^3) % 2^2) / 2^1) * 2^1 + 
            (((((binary % 2^4) % 2^3) % 2^2) % 2^1) / 2^0) * 2^0

-- Proof statement
theorem decimal_to_binary_correct : is_binary_representation decimal_number binary_representation :=
  by sorry

end decimal_to_binary_correct_l143_143640


namespace solve_system_l143_143504

noncomputable def solutions (a b c : ℝ) : Prop :=
  a^4 - b^4 = c ∧ b^4 - c^4 = a ∧ c^4 - a^4 = b

theorem solve_system :
  { (a, b, c) | solutions a b c } =
  { (0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0) } :=
by
  sorry

end solve_system_l143_143504


namespace find_pair_r_x_l143_143301

noncomputable def x_repr (r : ℕ) (n : ℕ) (ab : ℕ) : ℕ :=
  ab * (r * (r^(2*(n-1)) - 1) / (r^2 - 1))

noncomputable def x_squared_repr (r : ℕ) (n : ℕ) : ℕ :=
  (r^(4*n) - 1) / (r - 1)

theorem find_pair_r_x (r x : ℕ) (n : ℕ) (ab : ℕ) (r_leq_70 : r ≤ 70)
  (x_consistent: x = x_repr r n ab)
  (x_squared_consistent: x^2 = x_squared_repr r n)
  : (r = 7 ∧ x = 20) :=
begin
  sorry
end

end find_pair_r_x_l143_143301


namespace largest_integer_less_than_100_with_remainder_5_l143_143651

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143651


namespace polynomial_irreducible_segment_intersect_l143_143608

-- Part (a)
theorem polynomial_irreducible 
  (f : Polynomial ℤ) 
  (h_def : f = Polynomial.C 12 + Polynomial.X * Polynomial.C 9 + Polynomial.X^2 * Polynomial.C 6 + Polynomial.X^3 * Polynomial.C 3 + Polynomial.X^4) : 
  ¬ ∃ (p q : Polynomial ℤ), (Polynomial.degree p = 2) ∧ (Polynomial.degree q = 2) ∧ (f = p * q) :=
sorry

-- Part (b)
theorem segment_intersect 
  (n : ℕ) 
  (segments : Fin (2*n+1) → Set (ℝ × ℝ)) 
  (h_intersect : ∀ i, ∃ n_indices : Finset (Fin (2*n+1)), n_indices.card = n ∧ ∀ j ∈ n_indices, (segments i ∩ segments j).Nonempty) :
  ∃ i, ∀ j, i ≠ j → (segments i ∩ segments j).Nonempty :=
sorry


end polynomial_irreducible_segment_intersect_l143_143608


namespace katherine_age_l143_143934

-- Define a Lean statement equivalent to the given problem
theorem katherine_age (K M : ℕ) (h1 : M = K - 3) (h2 : M = 21) : K = 24 := sorry

end katherine_age_l143_143934


namespace find_a_7_l143_143239

-- Define the arithmetic sequence conditions
variable {a : ℕ → ℤ} -- The sequence a_n
variable (a_4_eq : a 4 = 4)
variable (a_3_a_8_eq : a 3 + a 8 = 5)

-- Prove that a_7 = 1
theorem find_a_7 : a 7 = 1 := by
  sorry

end find_a_7_l143_143239


namespace greatest_nat_not_sum_of_two_composites_l143_143648

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

theorem greatest_nat_not_sum_of_two_composites :
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b ∧
  (∀ n : ℕ, n > 11 → ¬ ∃ x y : ℕ, is_composite x ∧ is_composite y ∧ n = x + y) :=
sorry

end greatest_nat_not_sum_of_two_composites_l143_143648


namespace log_base_10_of_2_bounds_l143_143577

theorem log_base_10_of_2_bounds :
  (10^3 = 1000) ∧ (10^4 = 10000) ∧ (2^11 = 2048) ∧ (2^14 = 16384) →
  (3 / 11 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (2 / 7 : ℝ) :=
by
  sorry

end log_base_10_of_2_bounds_l143_143577


namespace largest_integer_less_than_100_div_8_rem_5_l143_143662

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l143_143662


namespace smallest_interesting_number_smallest_interesting_number_1800_l143_143843
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l143_143843


namespace M_plus_N_eq_2_l143_143300

noncomputable def M : ℝ := 1^5 + 2^4 * 3^3 - (4^2 / 5^1)
noncomputable def N : ℝ := 1^5 - 2^4 * 3^3 + (4^2 / 5^1)

theorem M_plus_N_eq_2 : M + N = 2 := by
  sorry

end M_plus_N_eq_2_l143_143300


namespace largest_int_with_remainder_5_lt_100_l143_143682

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l143_143682


namespace chess_tournament_games_l143_143738

theorem chess_tournament_games (P : ℕ) (TotalGames : ℕ) (hP : P = 21) (hTotalGames : TotalGames = 210) : 
  ∃ G : ℕ, G = 20 ∧ TotalGames = (P * (P - 1)) / 2 :=
by
  sorry

end chess_tournament_games_l143_143738


namespace binomial_expansion_b_value_l143_143062

theorem binomial_expansion_b_value (a b x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x ^ 2 + a^5 * x ^ 5) : b = 40 := 
sorry

end binomial_expansion_b_value_l143_143062


namespace not_both_perfect_squares_l143_143265

theorem not_both_perfect_squares (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ a b : ℕ, (n+1) * 2^n = a^2 ∧ (n+3) * 2^(n + 2) = b^2) :=
sorry

end not_both_perfect_squares_l143_143265


namespace smallest_possible_value_of_other_integer_l143_143138

theorem smallest_possible_value_of_other_integer (x b : ℕ) (h_gcd_lcm : ∀ m n : ℕ, m = 36 → gcd m n = x + 5 → lcm m n = x * (x + 5)) : 
  b > 0 → ∃ b, b = 1 ∧ gcd 36 b = x + 5 ∧ lcm 36 b = x * (x + 5) := 
by {
   sorry 
}

end smallest_possible_value_of_other_integer_l143_143138


namespace radius_of_larger_ball_l143_143560

theorem radius_of_larger_ball :
  let V_small : ℝ := (4/3) * Real.pi * (2^3)
  let V_total : ℝ := 6 * V_small
  let R : ℝ := (48:ℝ)^(1/3)
  V_total = (4/3) * Real.pi * (R^3) := 
  by
  sorry

end radius_of_larger_ball_l143_143560


namespace triangle_area_is_12_5_l143_143974

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨5, 0⟩
def N : Point := ⟨0, 5⟩
noncomputable def P (x y : ℝ) (h : x + y = 8) : Point := ⟨x, y⟩

noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem triangle_area_is_12_5 (x y : ℝ) (h : x + y = 8) :
  area_triangle M N (P x y h) = 12.5 :=
sorry

end triangle_area_is_12_5_l143_143974


namespace Jerry_weekly_earnings_l143_143106

def hours_per_task : ℕ := 2
def pay_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

theorem Jerry_weekly_earnings : pay_per_task * (hours_per_day / hours_per_task) * days_per_week = 1400 :=
by
  -- Carry out the proof here
  sorry

end Jerry_weekly_earnings_l143_143106


namespace jane_mean_after_extra_credit_l143_143102

-- Define Jane's original scores
def original_scores : List ℤ := [82, 90, 88, 95, 91]

-- Define the extra credit points
def extra_credit : ℤ := 2

-- Define the mean calculation after extra credit
def mean_after_extra_credit (scores : List ℤ) (extra : ℤ) : ℚ :=
  let total_sum := scores.sum + (scores.length * extra)
  total_sum / scores.length

theorem jane_mean_after_extra_credit :
  mean_after_extra_credit original_scores extra_credit = 91.2 := by
  sorry

end jane_mean_after_extra_credit_l143_143102


namespace jerry_weekly_earnings_l143_143108

-- Definitions of the given conditions
def pay_per_task : ℕ := 40
def hours_per_task : ℕ := 2
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

-- Calculated values from the conditions
def tasks_per_day : ℕ := hours_per_day / hours_per_task
def tasks_per_week : ℕ := tasks_per_day * days_per_week
def total_earnings : ℕ := pay_per_task * tasks_per_week

-- Theorem to prove
theorem jerry_weekly_earnings : total_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l143_143108


namespace mary_investment_amount_l143_143770

theorem mary_investment_amount
  (A : ℝ := 100000) -- Future value in dollars
  (r : ℝ := 0.08) -- Annual interest rate
  (n : ℕ := 12) -- Compounded monthly
  (t : ℝ := 10) -- Time in years
  : (⌈A / (1 + r / n) ^ (n * t)⌉₊ = 45045) :=
by
  sorry

end mary_investment_amount_l143_143770


namespace zero_point_of_function_l143_143142

theorem zero_point_of_function : ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 :=
by
  sorry

end zero_point_of_function_l143_143142


namespace exists_x_such_that_f_x_eq_0_l143_143523

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then
  3 * x - 4
else
  -x^2 + 3 * x - 5

theorem exists_x_such_that_f_x_eq_0 :
  ∃ x : ℝ, f x = 0 ∧ x = 1.192 :=
sorry

end exists_x_such_that_f_x_eq_0_l143_143523


namespace range_of_values_for_a_l143_143962

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_values_for_a_l143_143962


namespace calc_triple_hash_30_l143_143352

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem calc_triple_hash_30 :
  hash_fn (hash_fn (hash_fn 30)) = 10.4 :=
by 
  -- Proof goes here
  sorry

end calc_triple_hash_30_l143_143352


namespace Jerry_weekly_earnings_l143_143107

def hours_per_task : ℕ := 2
def pay_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

theorem Jerry_weekly_earnings : pay_per_task * (hours_per_day / hours_per_task) * days_per_week = 1400 :=
by
  -- Carry out the proof here
  sorry

end Jerry_weekly_earnings_l143_143107


namespace projectile_height_49_at_t_l143_143552

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
let discriminant := b^2 - 4 * a * c in
(((-b + Real.sqrt discriminant) / (2 * a)), ((-b - Real.sqrt discriminant) / (2 * a)))

theorem projectile_height_49_at_t (y t : ℝ) (h : y = -20 * t^2 + 100 * t) : t = 0.55 :=
begin
  have h_eq : 49 = -20 * t^2 + 100 * t, from h,
  have h_eq_quad: 0 = 20 * t^2 - 100 * t + 49,
  { ring at h_eq ⊢,
    linarith, },
  let (t1, t2) := quadratic_formula 20 (-100) 49,
  have : t1 = 0.55 ∨ t2 = 0.55 := sorry,
  cases this,
  { rwa this, },
  { sorry, },
end

end projectile_height_49_at_t_l143_143552


namespace first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l143_143625

noncomputable def first_three_digits_of_decimal_part (x : ℝ) : ℕ :=
  -- here we would have the actual definition
  sorry

theorem first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8 :
  first_three_digits_of_decimal_part ((10^1001 + 1)^((9:ℝ) / 8)) = 125 :=
sorry

end first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l143_143625


namespace largest_int_less_than_100_with_remainder_5_l143_143698

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l143_143698


namespace largest_int_less_than_100_with_remainder_5_l143_143696

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l143_143696


namespace largest_integer_less_than_100_with_remainder_5_l143_143700

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l143_143700


namespace petya_run_time_l143_143531

-- Definitions
def time_petya_4_to_1 : ℕ := 12

-- Conditions
axiom time_mom_condition : ∃ (time_mom : ℕ), time_petya_4_to_1 = time_mom - 2
axiom time_mom_5_to_1_condition : ∃ (time_petya_5_to_1 : ℕ), ∀ time_mom : ℕ, time_mom = time_petya_5_to_1 - 2

-- Proof statement
theorem petya_run_time :
  ∃ (time_petya_4_to_1 : ℕ), time_petya_4_to_1 = 12 :=
sorry

end petya_run_time_l143_143531


namespace running_speed_l143_143027

theorem running_speed (walk_speed total_distance walk_time total_time run_distance : ℝ) 
  (h_walk_speed : walk_speed = 4)
  (h_total_distance : total_distance = 4)
  (h_walk_time : walk_time = 0.5)
  (h_total_time : total_time = 0.75)
  (h_run_distance : run_distance = total_distance / 2) :
  (2 / ((total_time - walk_time) - 2 / walk_speed)) = 8 := 
by
  -- To be proven
  sorry

end running_speed_l143_143027


namespace sum_of_numbers_l143_143791

theorem sum_of_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 54) (h_ratio : a / b = 2 / 3) : a + b = 45 :=
by
  sorry

end sum_of_numbers_l143_143791


namespace sum_row_10_pascal_triangle_l143_143590

theorem sum_row_10_pascal_triangle :
  (∑ k in Finset.range (11), Nat.choose 10 k) = 1024 :=
by
  sorry

end sum_row_10_pascal_triangle_l143_143590


namespace teams_working_together_l143_143423

theorem teams_working_together
    (m n : ℕ) 
    (hA : ∀ t : ℕ, t = m → (t ≥ 0)) 
    (hB : ∀ t : ℕ, t = n → (t ≥ 0)) : 
  ∃ t : ℕ, t = (m * n) / (m + n) :=
by
  sorry

end teams_working_together_l143_143423


namespace smallest_interesting_number_l143_143829

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l143_143829


namespace problem1_problem2_l143_143073

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

-- Conditions:
axiom condition1 : sin (α + π / 6) = sqrt 10 / 10
axiom condition2 : cos (α + π / 6) = 3 * sqrt 10 / 10
axiom condition3 : tan (α + β) = 2 / 5

-- Prove:
theorem problem1 : sin (2 * α + π / 6) = (3 * sqrt 3 - 4) / 10 :=
by sorry

theorem problem2 : tan (2 * β - π / 3) = 17 / 144 :=
by sorry

end problem1_problem2_l143_143073


namespace total_items_correct_l143_143464

-- Defining the number of each type of items ordered by Betty
def slippers := 6
def lipstick := 4
def hair_color := 8

-- The total number of items ordered by Betty
def total_items := slippers + lipstick + hair_color

-- The statement asserting that the total number of items is 18
theorem total_items_correct : total_items = 18 := 
by 
  -- sorry allows us to skip the proof
  sorry

end total_items_correct_l143_143464


namespace johns_subtraction_l143_143150

theorem johns_subtraction 
  (a : ℕ) 
  (h₁ : (51 : ℕ)^2 = (50 : ℕ)^2 + 101) 
  (h₂ : (49 : ℕ)^2 = (50 : ℕ)^2 - b) 
  : b = 99 := 
by 
  sorry

end johns_subtraction_l143_143150


namespace problem_l143_143496

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem problem (a b : ℝ) (h : f a b (-2) = 2) : f a b 2 = -10 :=
by
  sorry

end problem_l143_143496


namespace consecutive_probability_l143_143621

-- Define the total number of ways to choose 2 episodes out of 6
def total_combinations : ℕ := Nat.choose 6 2

-- Define the number of ways to choose consecutive episodes
def consecutive_combinations : ℕ := 5

-- Define the probability of choosing consecutive episodes
def probability_of_consecutive : ℚ := consecutive_combinations / total_combinations

-- Theorem stating that the calculated probability should equal 1/3
theorem consecutive_probability : probability_of_consecutive = 1 / 3 :=
by
  sorry

end consecutive_probability_l143_143621


namespace remaining_customers_after_some_left_l143_143460

-- Define the initial conditions and question (before proving it)
def initial_customers := 8
def new_customers := 99
def total_customers_after_new := 104

-- Define the hypothesis based on the total customers after new customers added
theorem remaining_customers_after_some_left (x : ℕ) (h : x + new_customers = total_customers_after_new) : x = 5 :=
by {
  -- Proof omitted
  sorry
}

end remaining_customers_after_some_left_l143_143460


namespace smallest_interesting_number_l143_143849

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l143_143849


namespace petya_result_less_than_one_tenth_l143_143284

theorem petya_result_less_than_one_tenth 
  (a b c d e f : ℕ) 
  (ha: a.gcd b = 1) (hb: c.gcd d = 1)
  (hc: e.gcd f = 1) 
  (vasya_correct: (a / b) + (c / d) + (e / f) = 1) :
  (a + c + e) / (b + d + f) < 1 / 10 :=
by
  -- proof goes here
  sorry

end petya_result_less_than_one_tenth_l143_143284


namespace boys_without_calculators_l143_143508

-- Definitions based on the conditions
def total_boys : Nat := 20
def students_with_calculators : Nat := 26
def girls_with_calculators : Nat := 15

-- We need to prove the number of boys who did not bring their calculators.
theorem boys_without_calculators : (total_boys - (students_with_calculators - girls_with_calculators)) = 9 :=
by {
    -- Proof goes here
    sorry
}

end boys_without_calculators_l143_143508


namespace proof_third_length_gcd_l143_143585

/-- Statement: The greatest possible length that can be used to measure the given lengths exactly is 1 cm, 
and the third length is an unspecified number of centimeters that is relatively prime to both 1234 cm and 898 cm. -/
def third_length_gcd (x : ℕ) : Prop := 
  Int.gcd 1234 898 = 1 ∧ Int.gcd (Int.gcd 1234 898) x = 1

noncomputable def greatest_possible_length : ℕ := 1

theorem proof_third_length_gcd (x : ℕ) (h : third_length_gcd x) : greatest_possible_length = 1 := by
  sorry

end proof_third_length_gcd_l143_143585


namespace cos_arcsin_l143_143471

theorem cos_arcsin {θ : ℝ} (h : sin θ = 8/17) : cos θ = 15/17 :=
sorry

end cos_arcsin_l143_143471


namespace geometric_sequence_condition_l143_143513

-- Definitions based on conditions
def S (n : ℕ) (m : ℤ) : ℤ := 3^(n + 1) + m
def a1 (m : ℤ) : ℤ := S 1 m
def a_n (n : ℕ) : ℤ := if n = 1 then a1 (-3) else 2 * 3^n

-- The proof statement
theorem geometric_sequence_condition (m : ℤ) (h1 : a1 m = 3^2 + m) (h2 : ∀ n, n ≥ 2 → a_n n = 2 * 3^n) :
  m = -3 :=
sorry

end geometric_sequence_condition_l143_143513


namespace max_group_size_l143_143427

theorem max_group_size 
  (students_class1 : ℕ) (students_class2 : ℕ) 
  (leftover_class1 : ℕ) (leftover_class2 : ℕ) 
  (h_class1 : students_class1 = 69) 
  (h_class2 : students_class2 = 86) 
  (h_leftover1 : leftover_class1 = 5) 
  (h_leftover2 : leftover_class2 = 6) : 
  Nat.gcd (students_class1 - leftover_class1) (students_class2 - leftover_class2) = 16 :=
by
  sorry

end max_group_size_l143_143427


namespace binom_12_10_eq_66_l143_143344

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l143_143344


namespace no_rational_numbers_satisfy_l143_143478

theorem no_rational_numbers_satisfy :
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
    (1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 = 2014) :=
by
  sorry

end no_rational_numbers_satisfy_l143_143478


namespace minimum_value_of_expr_l143_143497

noncomputable def expr (x : ℝ) : ℝ := x + (1 / (x - 5))

theorem minimum_value_of_expr : ∀ (x : ℝ), x > 5 → expr x ≥ 7 ∧ (expr x = 7 ↔ x = 6) := 
by 
  sorry

end minimum_value_of_expr_l143_143497


namespace six_digit_number_property_l143_143360

theorem six_digit_number_property {a b c d e f : ℕ} 
  (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 0 ≤ c ∧ c < 10) (h4 : 0 ≤ d ∧ d < 10)
  (h5 : 0 ≤ e ∧ e < 10) (h6 : 0 ≤ f ∧ f < 10) 
  (h7 : 100000 ≤ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f ∧
        a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f < 1000000) :
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 3 * (f * 10^5 + a * 10^4 + b * 10^3 + c * 10^2 + d * 10 + e)) ↔ 
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 428571 ∨ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 857142) :=
sorry

end six_digit_number_property_l143_143360


namespace perpendicular_lines_m_value_l143_143389

-- Define the first line
def line1 (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the second line
def line2 (x y : ℝ) (m : ℝ) : Prop := 6 * x - m * y - 3 = 0

-- Define the perpendicular condition for slopes of two lines
def perpendicular_slopes (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Prove the value of m for perpendicular lines
theorem perpendicular_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, line1 x y → ∃ y', line2 x y' m) →
  (∀ x y : ℝ, ∃ x', line1 x y ∧ line2 x' y m) →
  perpendicular_slopes 3 (6 / m) →
  m = -18 :=
by
  sorry

end perpendicular_lines_m_value_l143_143389


namespace Ferris_wheel_ticket_cost_l143_143573

theorem Ferris_wheel_ticket_cost
  (cost_rc : ℕ) (rides_rc : ℕ) (cost_c : ℕ) (rides_c : ℕ) (total_tickets : ℕ) (rides_fw : ℕ)
  (H1 : cost_rc = 4) (H2 : rides_rc = 3) (H3 : cost_c = 4) (H4 : rides_c = 2) (H5 : total_tickets = 21) (H6 : rides_fw = 1) :
  21 - (3 * 4 + 2 * 4) = 1 :=
by
  sorry

end Ferris_wheel_ticket_cost_l143_143573


namespace boundary_shadow_function_l143_143430

theorem boundary_shadow_function 
    (r : ℝ) (O P : ℝ × ℝ × ℝ) (f : ℝ → ℝ)
    (h_radius : r = 1)
    (h_center : O = (1, 0, 1))
    (h_light_source : P = (1, -1, 2)) :
  (∀ x, f x = (x - 1) ^ 2 / 4 - 1) := 
by 
  sorry

end boundary_shadow_function_l143_143430


namespace max_value_of_abs_z_plus_4_l143_143735

open Complex
noncomputable def max_abs_z_plus_4 {z : ℂ} (h : abs (z + 3 * I) = 5) : ℝ :=
sorry

theorem max_value_of_abs_z_plus_4 (z : ℂ) (h : abs (z + 3 * I) = 5) : abs (z + 4) ≤ 10 :=
sorry

end max_value_of_abs_z_plus_4_l143_143735


namespace problem_solution_l143_143499

variables (x y : ℝ)

def cond1 : Prop := 4 * x + y = 12
def cond2 : Prop := x + 4 * y = 18

theorem problem_solution (h1 : cond1 x y) (h2 : cond2 x y) : 20 * x^2 + 24 * x * y + 20 * y^2 = 468 :=
by
  -- Proof would go here
  sorry

end problem_solution_l143_143499


namespace inequality_abc_l143_143519

variable {a b c : ℝ}

theorem inequality_abc (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 8) :
  (a - 2) / (a + 1) + (b - 2) / (b + 1) + (c - 2) / (c + 1) ≤ 0 := by
  sorry

end inequality_abc_l143_143519


namespace smallest_interesting_number_smallest_interesting_number_1800_l143_143844
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l143_143844


namespace children_on_ferris_wheel_l143_143170

theorem children_on_ferris_wheel (x : ℕ) (h : 5 * x + 3 * 5 + 8 * 2 * 5 = 110) : x = 3 :=
sorry

end children_on_ferris_wheel_l143_143170


namespace log_sum_greater_than_two_l143_143877

variables {x y a m : ℝ}

theorem log_sum_greater_than_two
  (hx : 0 < x) (hxy : x < y) (hya : y < a) (ha1 : a < 1)
  (hm : m = Real.log x / Real.log a + Real.log y / Real.log a) : m > 2 :=
sorry

end log_sum_greater_than_two_l143_143877


namespace largest_integer_lt_100_with_remainder_5_div_8_l143_143665

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l143_143665


namespace geometric_progression_terms_l143_143201

theorem geometric_progression_terms (b1 b2 bn : ℕ) (q n : ℕ)
  (h1 : b1 = 3) 
  (h2 : b2 = 12)
  (h3 : bn = 3072)
  (h4 : b2 = b1 * q)
  (h5 : bn = b1 * q^(n-1)) : 
  n = 6 := 
by 
  sorry

end geometric_progression_terms_l143_143201


namespace original_salary_l143_143263

def final_salary_after_changes (S : ℝ) : ℝ :=
  let increased_10 := S * 1.10
  let promoted_8 := increased_10 * 1.08
  let deducted_5 := promoted_8 * 0.95
  let decreased_7 := deducted_5 * 0.93
  decreased_7

theorem original_salary (S : ℝ) (h : final_salary_after_changes S = 6270) : S = 5587.68 :=
by
  -- Proof to be completed here
  sorry

end original_salary_l143_143263


namespace all_boxcars_combined_capacity_l143_143540

theorem all_boxcars_combined_capacity :
  let black_capacity := 4000
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  let green_capacity := 1.5 * black_capacity
  let yellow_capacity := green_capacity + 2000
  let total_red := 3 * red_capacity
  let total_blue := 4 * blue_capacity
  let total_black := 7 * black_capacity
  let total_green := 2 * green_capacity
  let total_yellow := 5 * yellow_capacity
  total_red + total_blue + total_black + total_green + total_yellow = 184000 :=
by 
  -- Proof omitted
  sorry

end all_boxcars_combined_capacity_l143_143540


namespace smallest_interesting_number_l143_143830

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l143_143830


namespace commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l143_143049

def star (x y : ℕ) : ℕ := (x + 2) * (y + 2) - 2

theorem commutative_star : ∀ x y : ℕ, star x y = star y x := by
  sorry

theorem not_distributive_star : ∃ x y z : ℕ, star x (y + z) ≠ star x y + star x z := by
  sorry

theorem special_case_star_false : ∀ x : ℕ, star (x - 2) (x + 2) ≠ star x x - 2 := by
  sorry

theorem no_identity_star : ¬∃ e : ℕ, ∀ x : ℕ, star x e = x ∧ star e x = x := by
  sorry

-- Associativity requires further verification and does not have a definitive statement yet.

end commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l143_143049


namespace no_x4_term_expansion_l143_143885

-- Mathematical condition and properties
variable {R : Type*} [CommRing R]

theorem no_x4_term_expansion (a : R) (h : a ≠ 0) :
  ∃ a, (a = 8) := 
by 
  sorry

end no_x4_term_expansion_l143_143885


namespace product_of_numbers_l143_143991

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := 
sorry

end product_of_numbers_l143_143991


namespace sum_of_squares_of_coefficients_l143_143040

theorem sum_of_squares_of_coefficients :
  let p := 3 * (X^5 + 4 * X^3 + 2 * X + 1)
  let coeffs := [3, 12, 6, 3, 0, 0]
  let sum_squares := coeffs.map (λ c => c * c) |>.sum
  sum_squares = 198 := by
  sorry

end sum_of_squares_of_coefficients_l143_143040


namespace min_value_n_l143_143060

noncomputable def minN : ℕ :=
  5

theorem min_value_n :
  ∀ (S : Finset ℕ), (∀ n ∈ S, 1 ≤ n ∧ n ≤ 9) ∧ S.card = minN → 
    (∃ T ⊆ S, T ≠ ∅ ∧ 10 ∣ (T.sum id)) :=
by
  sorry

end min_value_n_l143_143060


namespace parabola_properties_and_intersection_l143_143723

-- Definition of the parabola C: y^2 = -4x
def parabola_C (x y : ℝ) : Prop := y^2 = -4 * x

-- Focus of the parabola
def focus_C : ℝ × ℝ := (-1, 0)

-- Equation of the directrix
def directrix_C (x: ℝ): Prop := x = 1

-- Distance from the focus to the directrix
def distance_focus_to_directrix : ℝ := 2

-- Line l passing through P(1, 2) with slope k
def line_l (k x y : ℝ) : Prop := y = k * x - k + 2

-- Main theorem statement
theorem parabola_properties_and_intersection (k: ℝ) :
  (focus_C = (-1, 0)) ∧
  (∀ x, directrix_C x ↔ x = 1) ∧
  (distance_focus_to_directrix = 2) ∧
  ((k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) →
    ∃ x y, parabola_C x y ∧ line_l k x y ∧
    (∀ x' y', parabola_C x' y' ∧ line_l k x' y' → x = x' ∧ y = y')) ∧
  ((1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) →
    ∃ x y x' y', x ≠ x' ∧ y ≠ y' ∧
    parabola_C x y ∧ line_l k x y ∧
    parabola_C x' y' ∧ line_l k x' y') ∧
  ((k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) →
    ∀ x y, ¬(parabola_C x y ∧ line_l k x y)) :=
by sorry

end parabola_properties_and_intersection_l143_143723


namespace min_value_of_a_plus_b_l143_143715

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : a + b = 4 :=
sorry

end min_value_of_a_plus_b_l143_143715


namespace total_children_is_11_l143_143780

noncomputable def num_of_children (b g : ℕ) := b + g

theorem total_children_is_11 (b g : ℕ) :
  (∃ c : ℕ, b * c + g * (c + 1) = 47) ∧
  (∃ m : ℕ, b * (m + 1) + g * m = 74) → 
  num_of_children b g = 11 :=
by
  -- The proof steps would go here to show that b + g = 11
  sorry

end total_children_is_11_l143_143780


namespace Tameka_sold_40_boxes_on_Friday_l143_143268

noncomputable def TamekaSalesOnFriday (F : ℕ) : Prop :=
  let SaturdaySales := 2 * F - 10
  let SundaySales := (2 * F - 10) / 2
  F + SaturdaySales + SundaySales = 145

theorem Tameka_sold_40_boxes_on_Friday : ∃ F : ℕ, TamekaSalesOnFriday F ∧ F = 40 := 
by 
  sorry

end Tameka_sold_40_boxes_on_Friday_l143_143268


namespace matrix_pow_four_l143_143336

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !!
  [ 2, -1,
    1,  1]

-- State the theorem with the final result
theorem matrix_pow_four :
  A ^ 4 = !!
  [ 0, -9,
    9, -9] :=
  sorry

end matrix_pow_four_l143_143336


namespace sum_of_coefficients_l143_143795

theorem sum_of_coefficients (a b c : ℤ) (h : a - b + c = -1) : a + b + c = -1 := sorry

end sum_of_coefficients_l143_143795


namespace age_ratio_l143_143539

/-- Given that Sandy's age after 6 years will be 30 years,
    and Molly's current age is 18 years, 
    prove that the current ratio of Sandy's age to Molly's age is 4:3. -/
theorem age_ratio (M S : ℕ) 
  (h1 : M = 18) 
  (h2 : S + 6 = 30) : 
  S / gcd S M = 4 ∧ M / gcd S M = 3 :=
by
  sorry

end age_ratio_l143_143539


namespace edge_length_of_cubical_box_l143_143463

noncomputable def volume_of_cube (edge_length_cm : ℝ) : ℝ :=
  edge_length_cm ^ 3

noncomputable def number_of_cubes : ℝ := 8000
noncomputable def edge_of_small_cube_cm : ℝ := 5

noncomputable def total_volume_of_cubes_cm3 : ℝ :=
  volume_of_cube edge_of_small_cube_cm * number_of_cubes

noncomputable def volume_of_box_cm3 : ℝ := total_volume_of_cubes_cm3
noncomputable def edge_length_of_box_m : ℝ :=
  (volume_of_box_cm3)^(1 / 3) / 100

theorem edge_length_of_cubical_box :
  edge_length_of_box_m = 1 := by 
  sorry

end edge_length_of_cubical_box_l143_143463


namespace comparison_of_f_values_l143_143890

noncomputable def f (x : ℝ) := Real.cos x - x

theorem comparison_of_f_values :
  f (8 * Real.pi / 9) > f Real.pi ∧ f Real.pi > f (10 * Real.pi / 9) :=
by
  sorry

end comparison_of_f_values_l143_143890


namespace solve_for_y_l143_143043

theorem solve_for_y : ∃ (y : ℚ), y + 2 - 2 / 3 = 4 * y - (y + 2) ∧ y = 5 / 3 :=
by
  sorry

end solve_for_y_l143_143043


namespace solve_system_eq_pos_reals_l143_143266

theorem solve_system_eq_pos_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + y^2 + x * y = 7)
  (h2 : x^2 + z^2 + x * z = 13)
  (h3 : y^2 + z^2 + y * z = 19) :
  x = 1 ∧ y = 2 ∧ z = 3 :=
sorry

end solve_system_eq_pos_reals_l143_143266


namespace smallest_interesting_number_l143_143833

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l143_143833


namespace sum_of_squares_219_l143_143140

theorem sum_of_squares_219 :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 + c^2 = 219 ∧ a + b + c = 21 := by
  sorry

end sum_of_squares_219_l143_143140


namespace equilateral_triangles_count_in_grid_of_side_4_l143_143724

-- Define a function to calculate the number of equilateral triangles in a triangular grid of side length n
def countEquilateralTriangles (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2) * (n + 3)) / 24

-- Define the problem statement for n = 4
theorem equilateral_triangles_count_in_grid_of_side_4 :
  countEquilateralTriangles 4 = 35 := by
  sorry

end equilateral_triangles_count_in_grid_of_side_4_l143_143724


namespace intersection_eq_l143_143071

open Set

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x ≤ 5 }
def B : Set ℝ := { -1, 2, 3, 6 }

-- State the proof problem
theorem intersection_eq : A ∩ B = {2, 3} := 
by 
-- placeholder for the proof steps
sorry

end intersection_eq_l143_143071


namespace ratio_of_sides_l143_143401

variable {A B C a b c : ℝ}

theorem ratio_of_sides
  (h1 : 2 * b * Real.sin (2 * A) = 3 * a * Real.sin B)
  (h2 : c = 2 * b) :
  a / b = Real.sqrt 2 := by
  sorry

end ratio_of_sides_l143_143401


namespace unique_solution_condition_l143_143871

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 4) ↔ d ≠ 4 :=
by
  sorry

end unique_solution_condition_l143_143871


namespace largest_y_l143_143488

theorem largest_y (y : ℝ) (h : (⌊y⌋ / y) = 8 / 9) : y ≤ 63 / 8 :=
sorry

end largest_y_l143_143488


namespace decimal_to_base7_l143_143045

theorem decimal_to_base7 :
    ∃ k₀ k₁ k₂ k₃ k₄, 1987 = k₀ * 7^4 + k₁ * 7^3 + k₂ * 7^2 + k₃ * 7^1 + k₄ * 7^0 ∧
    k₀ = 0 ∧
    k₁ = 5 ∧
    k₂ = 3 ∧
    k₃ = 5 ∧
    k₄ = 6 :=
by
  sorry

end decimal_to_base7_l143_143045


namespace largest_integer_less_than_100_with_remainder_5_l143_143653

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l143_143653


namespace diamond_value_l143_143353

variable {a b : ℤ}

-- Define the operation diamond following the given condition.
def diamond (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

-- Define the conditions given in the problem.
axiom h1 : a + b = 10
axiom h2 : a * b = 24

-- State the target theorem.
theorem diamond_value : diamond a b = 5 / 12 :=
by
  sorry

end diamond_value_l143_143353


namespace ceil_sqrt_200_eq_15_l143_143637

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l143_143637


namespace fraction_of_fractions_l143_143578

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_fractions_l143_143578


namespace price_reduction_l143_143428

theorem price_reduction (x : ℝ) 
  (initial_price : ℝ := 60) 
  (final_price : ℝ := 48.6) :
  initial_price * (1 - x) * (1 - x) = final_price :=
by
  sorry

end price_reduction_l143_143428


namespace factorial_divisibility_l143_143757

theorem factorial_divisibility {n : ℕ} (h : 2011^(2011) ∣ n!) : 2011^(2012) ∣ n! :=
sorry

end factorial_divisibility_l143_143757


namespace simplify_fraction_eq_one_over_thirty_nine_l143_143419

theorem simplify_fraction_eq_one_over_thirty_nine :
  let a1 := (1 / 3)^1
  let a2 := (1 / 3)^2
  let a3 := (1 / 3)^3
  (1 / (1 / a1 + 1 / a2 + 1 / a3)) = 1 / 39 :=
by
  sorry

end simplify_fraction_eq_one_over_thirty_nine_l143_143419


namespace jerry_weekly_earnings_l143_143104

theorem jerry_weekly_earnings:
  (tasks_per_day: ℕ) 
  (daily_earnings: ℕ)
  (weekly_earnings: ℕ) :
  (tasks_per_day = 10 / 2) ∧
  (daily_earnings = 40 * tasks_per_day) ∧
  (weekly_earnings = daily_earnings * 7) →
  weekly_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l143_143104


namespace Mark_has_23_kangaroos_l143_143769

theorem Mark_has_23_kangaroos :
  ∃ K G : ℕ, G = 3 * K ∧ 2 * K + 4 * G = 322 ∧ K = 23 :=
by
  sorry

end Mark_has_23_kangaroos_l143_143769


namespace not_equal_zero_equal_zero_exists_u_l143_143884

-- Definitions based on given conditions
variables {x y u : ℝ → ℝ} {t : ℝ}
variable  {x_0 y_0 : ℝ}
variables (h_cont_u : Continuous u) 
variables (hx : ∀ t, deriv x t = -2 * y t + u t)
variables (hy : ∀ t, deriv y t = -2 * x t + u t)
variables (hx_0 : x 0 = x_0) 
variables (hy_0 : y 0 = y_0)

-- Problem 1:
theorem not_equal_zero (h : x_0 ≠ y_0) : ¬(∀ t, x t = 0 ∧ y t = 0) :=
sorry

-- Problem 2:
theorem equal_zero_exists_u (h : x_0 = y_0) (T : ℝ) (hT : T > 0) : ∃ u, (∀ t, deriv x t = -2 * y t + u t) ∧ (∀ t, deriv y t = -2 * x t + u t) ∧ (x T = 0 ∧ y T = 0) :=
sorry

end not_equal_zero_equal_zero_exists_u_l143_143884


namespace ratio_female_to_total_l143_143119

theorem ratio_female_to_total:
  ∃ (F : ℕ), (6 + 7 * F - 9 = (6 + 7 * F) - 9) ∧ 
             (7 * F - 9 = 67 / 100 * ((6 + 7 * F) - 9)) → 
             F = 3 ∧ 6 = 6 → 
             1 / F = 2 / 6 :=
by sorry

end ratio_female_to_total_l143_143119


namespace area_of_grey_region_l143_143441

open Nat

theorem area_of_grey_region
  (a1 a2 b : ℕ)
  (h1 : a1 = 8 * 10)
  (h2 : a2 = 9 * 12)
  (hb : b = 37)
  : (a2 - (a1 - b) = 65) := by
  sorry

end area_of_grey_region_l143_143441


namespace minimum_a_plus_b_l143_143928

theorem minimum_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -24 :=
by sorry

end minimum_a_plus_b_l143_143928


namespace youngest_child_age_l143_143011

theorem youngest_child_age {x : ℝ} (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by
  sorry

end youngest_child_age_l143_143011


namespace train_carriages_l143_143435

theorem train_carriages (num_trains : ℕ) (total_wheels : ℕ) (rows_per_carriage : ℕ) 
  (wheels_per_row : ℕ) (carriages_per_train : ℕ) :
  num_trains = 4 →
  total_wheels = 240 →
  rows_per_carriage = 3 →
  wheels_per_row = 5 →
  carriages_per_train = 
    (total_wheels / (rows_per_carriage * wheels_per_row)) / num_trains →
  carriages_per_train = 4 :=
by
  sorry

end train_carriages_l143_143435


namespace smallest_interesting_number_l143_143851

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l143_143851


namespace aprons_to_sew_tomorrow_l143_143219

def total_aprons : ℕ := 150
def already_sewn : ℕ := 13
def sewn_today (already_sewn : ℕ) : ℕ := 3 * already_sewn
def sewn_tomorrow (total_aprons : ℕ) (already_sewn : ℕ) (sewn_today : ℕ) : ℕ :=
  let remaining := total_aprons - (already_sewn + sewn_today)
  remaining / 2

theorem aprons_to_sew_tomorrow : sewn_tomorrow total_aprons already_sewn (sewn_today already_sewn) = 49 :=
  by 
    sorry

end aprons_to_sew_tomorrow_l143_143219


namespace range_of_a_l143_143366

theorem range_of_a (a : ℝ) :
  (∀ x: ℝ, |x - a| < 4 → -x^2 + 5 * x - 6 > 0) → (-1 ≤ a ∧ a ≤ 6) :=
by
  intro h
  sorry

end range_of_a_l143_143366


namespace op_assoc_l143_143047

open Real

def op (x y : ℝ) : ℝ := x + y - x * y

theorem op_assoc (x y z : ℝ) : op (op x y) z = op x (op y z) := by
  sorry

end op_assoc_l143_143047


namespace tangent_line_of_ellipse_l143_143276

theorem tangent_line_of_ellipse 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (x₀ y₀ : ℝ) (hx₀ : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y : ℝ, x₀ * x / a^2 + y₀ * y / b^2 = 1 := 
sorry

end tangent_line_of_ellipse_l143_143276


namespace find_perimeter_correct_l143_143958

noncomputable def find_perimeter (L W : ℝ) (x : ℝ) :=
  L * W = (L + 6) * (W - 2) ∧
  L * W = (L - 12) * (W + 6) ∧
  x = 2 * (L + W)

theorem find_perimeter_correct : ∀ (L W : ℝ), L * W = (L + 6) * (W - 2) → 
                                      L * W = (L - 12) * (W + 6) → 
                                      2 * (L + W) = 132 :=
sorry

end find_perimeter_correct_l143_143958


namespace total_cups_of_mushroom_soup_l143_143783

def cups_team_1 : ℕ := 90
def cups_team_2 : ℕ := 120
def cups_team_3 : ℕ := 70

theorem total_cups_of_mushroom_soup :
  cups_team_1 + cups_team_2 + cups_team_3 = 280 :=
  by sorry

end total_cups_of_mushroom_soup_l143_143783


namespace standard_deviations_below_mean_l143_143959

theorem standard_deviations_below_mean (μ σ x : ℝ) (hμ : μ = 14.5) (hσ : σ = 1.7) (hx : x = 11.1) :
    (μ - x) / σ = 2 := by
  sorry

end standard_deviations_below_mean_l143_143959


namespace arithmetic_sequence_a1_l143_143748

theorem arithmetic_sequence_a1 (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_inc : d > 0)
  (h_a3 : a 3 = 1)
  (h_a2a4 : a 2 * a 4 = 3 / 4) : 
  a 1 = 0 :=
sorry

end arithmetic_sequence_a1_l143_143748


namespace find_integer_pair_l143_143051

theorem find_integer_pair (x y : ℤ) :
  (x + 2)^4 - x^4 = y^3 → (x = -1 ∧ y = 0) :=
by
  intro h
  sorry

end find_integer_pair_l143_143051


namespace front_view_length_l143_143171

-- Define the conditions of the problem
variables (d_body : ℝ) (d_side : ℝ) (d_top : ℝ)
variables (d_front : ℝ)

-- The given conditions
def conditions :=
  d_body = 5 * Real.sqrt 2 ∧
  d_side = 5 ∧
  d_top = Real.sqrt 34

-- The theorem to be proved
theorem front_view_length : 
  conditions d_body d_side d_top →
  d_front = Real.sqrt 41 :=
sorry

end front_view_length_l143_143171


namespace simplify_fraction_l143_143132

theorem simplify_fraction (a b c d : ℕ) (h₁ : a = 2) (h₂ : b = 462) (h₃ : c = 29) (h₄ : d = 42) :
  (a : ℚ) / (b : ℚ) + (c : ℚ) / (d : ℚ) = 107 / 154 :=
by {
  sorry
}

end simplify_fraction_l143_143132


namespace average_of_original_set_l143_143270

theorem average_of_original_set
  (A : ℝ)
  (n : ℕ)
  (B : ℝ)
  (h1 : n = 7)
  (h2 : B = 5 * A)
  (h3 : B / n = 100)
  : A = 20 :=
by
  sorry

end average_of_original_set_l143_143270


namespace largest_integer_less_than_100_with_remainder_5_l143_143701

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l143_143701


namespace transformed_conic_symmetric_eq_l143_143487

def conic_E (x y : ℝ) := x^2 + 2 * x * y + y^2 + 3 * x + y
def line_l (x y : ℝ) := 2 * x - y - 1

def transformed_conic_equation (x y : ℝ) := x^2 + 14 * x * y + 49 * y^2 - 21 * x + 103 * y + 54

theorem transformed_conic_symmetric_eq (x y : ℝ) :
  (∀ x y, conic_E x y = 0 → 
    ∃ x' y', line_l x' y' = 0 ∧ conic_E x' y' = 0 ∧ transformed_conic_equation x y = 0) :=
sorry

end transformed_conic_symmetric_eq_l143_143487


namespace sum_of_faces_edges_vertices_l143_143156

def cube_faces : ℕ := 6
def cube_edges : ℕ := 12
def cube_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices :
  cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end sum_of_faces_edges_vertices_l143_143156


namespace smallest_interesting_number_l143_143832

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l143_143832


namespace problem_travel_time_with_current_l143_143031

theorem problem_travel_time_with_current
  (D r c : ℝ) (t : ℝ)
  (h1 : (r - c) ≠ 0)
  (h2 : D / (r - c) = 60 / 7)
  (h3 : D / r = t - 7)
  (h4 : D / (r + c) = t)
  : t = 3 + 9 / 17 := 
sorry

end problem_travel_time_with_current_l143_143031


namespace find_two_angles_of_scalene_obtuse_triangle_l143_143179

def is_scalene (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_obtuse (a : ℝ) : Prop := a > 90
def is_triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem find_two_angles_of_scalene_obtuse_triangle
  (a b c : ℝ)
  (ha : is_obtuse a) (h_scalene : is_scalene a b c) 
  (h_sum : is_triangle a b c) 
  (ha_val : a = 108)
  (h_half : b = 2 * c) :
  b = 48 ∧ c = 24 :=
by
  sorry

end find_two_angles_of_scalene_obtuse_triangle_l143_143179


namespace sum_of_products_l143_143756

open Finset

noncomputable def M (n : ℕ) : Finset ℤ :=
  (range n).map (λ i, -↑(i + 1))

def products (s : Finset ℤ) : Finset ℤ := if h : s.nonempty then {s.prod id} else ∅

theorem sum_of_products (n : ℕ) (h : 1 ≤ n) :
  ∑ t in (M n).powerset.filter (λ s, s.nonempty), (t.prod id) = -1 := by
  sorry

end sum_of_products_l143_143756


namespace number_of_tea_bags_l143_143948

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l143_143948


namespace single_intersection_point_l143_143057

theorem single_intersection_point (k : ℝ) :
  (∃! x : ℝ, x^2 - 2 * x - k = 0) ↔ k = 0 :=
by
  sorry

end single_intersection_point_l143_143057


namespace find_m_n_l143_143901

theorem find_m_n (m n : ℤ) (h : |m - 2| + (n^2 - 8 * n + 16) = 0) : m = 2 ∧ n = 4 :=
by
  sorry

end find_m_n_l143_143901


namespace largest_negative_is_l143_143323

def largest_of_negatives (a b c d : ℚ) (largest : ℚ) : Prop := largest = max (max a b) (max c d)

theorem largest_negative_is (largest : ℚ) : largest_of_negatives (-2/3) (-2) (-1) (-5) largest → largest = -2/3 :=
by
  intro h
  -- We assume the definition and the theorem are sufficient to say largest = -2/3
  sorry

end largest_negative_is_l143_143323


namespace largest_integer_lt_100_with_remainder_5_div_8_l143_143668

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l143_143668


namespace rectangle_width_decrease_l143_143797

theorem rectangle_width_decrease {L W : ℝ} (A : ℝ) (hA : A = L * W) (h_new_length : A = 1.25 * L * (W * y)) : y = 0.8 :=
by sorry

end rectangle_width_decrease_l143_143797


namespace find_a_values_l143_143084

theorem find_a_values (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 27 * x^3) (h₃ : a - b = 2 * x) :
  a = 3.041 * x ∨ a = -1.041 * x :=
by
  sorry

end find_a_values_l143_143084


namespace binom_12_10_l143_143345

theorem binom_12_10 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_l143_143345


namespace problem_solving_ratio_l143_143410

theorem problem_solving_ratio 
  (total_mcqs : ℕ) (total_psqs : ℕ)
  (written_mcqs_fraction : ℚ) (total_remaining_questions : ℕ)
  (h1 : total_mcqs = 35)
  (h2 : total_psqs = 15)
  (h3 : written_mcqs_fraction = 2/5)
  (h4 : total_remaining_questions = 31) :
  (5 : ℚ) / 15 = (1 : ℚ) / 3 := 
by {
  -- given that 5 is the number of problem-solving questions already written,
  -- and 15 is the total number of problem-solving questions
  sorry
}

end problem_solving_ratio_l143_143410


namespace emily_51_49_calculations_l143_143149

theorem emily_51_49_calculations :
  (51^2 = 50^2 + 101) ∧ (49^2 = 50^2 - 99) :=
by
  sorry

end emily_51_49_calculations_l143_143149


namespace solution_in_quadrants_I_and_II_l143_143048

theorem solution_in_quadrants_I_and_II (x y : ℝ) :
  (y > 3 * x) ∧ (y > 6 - 2 * x) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
by
  sorry

end solution_in_quadrants_I_and_II_l143_143048


namespace packs_of_blue_tshirts_l143_143473

theorem packs_of_blue_tshirts (total_tshirts white_packs white_per_pack blue_per_pack : ℕ) 
  (h_white_packs : white_packs = 3) 
  (h_white_per_pack : white_per_pack = 6) 
  (h_blue_per_pack : blue_per_pack = 4) 
  (h_total_tshirts : total_tshirts = 26) : 
  (total_tshirts - white_packs * white_per_pack) / blue_per_pack = 2 := 
by
  -- Proof omitted
  sorry

end packs_of_blue_tshirts_l143_143473


namespace margo_paired_with_irma_probability_l143_143574

theorem margo_paired_with_irma_probability :
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  probability = (1 / 15) :=
by
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  have h : probability = 1 / 15 := by
    -- skipping the proof details as per instructions
    sorry
  exact h

end margo_paired_with_irma_probability_l143_143574


namespace natasha_time_to_top_l143_143258

theorem natasha_time_to_top (T : ℝ) 
  (descent_time : ℝ) 
  (whole_journey_avg_speed : ℝ) 
  (climbing_speed : ℝ) 
  (desc_time_condition : descent_time = 2) 
  (whole_journey_avg_speed_condition : whole_journey_avg_speed = 3.5) 
  (climbing_speed_condition : climbing_speed = 2.625) 
  (distance_to_top : ℝ := climbing_speed * T) 
  (avg_speed_condition : whole_journey_avg_speed = 2 * distance_to_top / (T + descent_time)) :
  T = 4 := by
  sorry

end natasha_time_to_top_l143_143258


namespace tea_bags_l143_143939

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l143_143939


namespace white_rabbit_hop_distance_per_minute_l143_143322

-- Definitions for given conditions
def brown_hop_per_minute : ℕ := 12
def total_distance_in_5_minutes : ℕ := 135
def brown_distance_in_5_minutes : ℕ := 5 * brown_hop_per_minute

-- The statement we need to prove
theorem white_rabbit_hop_distance_per_minute (W : ℕ) (h1 : brown_hop_per_minute = 12) (h2 : total_distance_in_5_minutes = 135) :
  W = 15 :=
by
  sorry

end white_rabbit_hop_distance_per_minute_l143_143322


namespace largest_e_is_23_l143_143761

open Real

-- Define the problem conditions
structure CircleDiameter (P Q X Y Z : ℝ × ℝ) where
  PQ_is_diameter : dist P Q = 2
  X_is_midpoint : dist P Q / 2 = dist P X
  PY_length : dist P Y = 4 / 5
  X_on_semicircle : (X.1 - (P.1 + Q.1) / 2)^2 + (X.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2
  Y_on_semicircle : (Y.1 - (P.1 + Q.1) / 2)^2 + (Y.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2
  Z_lies_other_semicircle : (Z.1 - (P.1 + Q.1) / 2)^2 + (Z.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2

noncomputable def largest_possible_e (P Q X Y Z : ℝ × ℝ) [CircleDiameter P Q X Y Z] : ℝ :=
let V := P -- Intersection point placeholder V
let W := Q -- Intersection point placeholder W
let e := dist V W -- Length segment e
in e 

theorem largest_e_is_23 (P Q X Y Z : ℝ × ℝ) [CircleDiameter P Q X Y Z] :
  largest_possible_e P Q X Y Z = 23 :=
sorry

end largest_e_is_23_l143_143761


namespace aprons_to_sew_tomorrow_l143_143220

def total_aprons : ℕ := 150
def already_sewn : ℕ := 13
def sewn_today (already_sewn : ℕ) : ℕ := 3 * already_sewn
def sewn_tomorrow (total_aprons : ℕ) (already_sewn : ℕ) (sewn_today : ℕ) : ℕ :=
  let remaining := total_aprons - (already_sewn + sewn_today)
  remaining / 2

theorem aprons_to_sew_tomorrow : sewn_tomorrow total_aprons already_sewn (sewn_today already_sewn) = 49 :=
  by 
    sorry

end aprons_to_sew_tomorrow_l143_143220


namespace matrix_power_four_l143_143337

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l143_143337


namespace science_book_pages_l143_143964

def history_pages := 300
def novel_pages := history_pages / 2
def science_pages := novel_pages * 4

theorem science_book_pages : science_pages = 600 := by
  -- Given conditions:
  -- The novel has half as many pages as the history book, the history book has 300 pages,
  -- and the science book has 4 times as many pages as the novel.
  sorry

end science_book_pages_l143_143964


namespace min_value_of_quadratic_l143_143289

theorem min_value_of_quadratic :
  ∀ x : ℝ, ∃ z : ℝ, z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∀ z' : ℝ, (z' = 4 * x^2 + 8 * x + 16) → z' ≥ 12) :=
by
  sorry

end min_value_of_quadratic_l143_143289


namespace largest_int_less_than_100_with_remainder_5_l143_143695

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l143_143695


namespace largest_integer_less_than_100_with_remainder_5_l143_143688

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l143_143688


namespace boat_speed_is_20_l143_143803

-- Definitions based on conditions from the problem
def boat_speed_still_water (x : ℝ) : Prop := 
  let current_speed := 5
  let downstream_distance := 8.75
  let downstream_time := 21 / 60
  let downstream_speed := x + current_speed
  downstream_speed * downstream_time = downstream_distance

-- The theorem to prove
theorem boat_speed_is_20 : boat_speed_still_water 20 :=
by 
  unfold boat_speed_still_water
  sorry

end boat_speed_is_20_l143_143803


namespace largest_integer_less_than_100_div_8_rem_5_l143_143657

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l143_143657


namespace sum_of_three_exists_l143_143408

theorem sum_of_three_exists (n : ℤ) (X : Finset ℤ) 
  (hX_card : X.card = n + 2) 
  (hX_abs : ∀ x ∈ X, abs x ≤ n) : 
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ c = a + b := 
by 
  sorry

end sum_of_three_exists_l143_143408


namespace find_common_difference_l143_143370

theorem find_common_difference
  (a_1 : ℕ := 1)
  (S : ℕ → ℕ)
  (h1 : S 5 = 20)
  (h2 : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d))
  : d = 3 / 2 := 
by 
  sorry

end find_common_difference_l143_143370


namespace number_of_toys_sold_l143_143313

theorem number_of_toys_sold (total_selling_price gain_per_toy cost_price_per_toy : ℕ)
  (h1 : total_selling_price = 25200)
  (h2 : gain_per_toy = 3 * cost_price_per_toy)
  (h3 : cost_price_per_toy = 1200) : 
  (total_selling_price - gain_per_toy) / cost_price_per_toy = 18 :=
by 
  sorry

end number_of_toys_sold_l143_143313


namespace Radhika_total_games_l143_143129

theorem Radhika_total_games :
  let christmas_gifts := 12
      birthday_gifts := 8
      total_gifts := christmas_gifts + birthday_gifts
      previously_owned := (1 / 2) * total_gifts
  in previously_owned + total_gifts = 30 :=
by
  let christmas_gifts := 12
  let birthday_gifts := 8
  let total_gifts := christmas_gifts + birthday_gifts
  let previously_owned := (1 / 2) * total_gifts
  show previously_owned + total_gifts = 30
  sorry

end Radhika_total_games_l143_143129


namespace original_book_price_l143_143967

theorem original_book_price (P : ℝ) (h : 0.85 * P * 1.40 = 476) : P = 476 / (0.85 * 1.40) :=
by
  sorry

end original_book_price_l143_143967


namespace age_difference_l143_143969

theorem age_difference (x : ℕ) 
  (h_ratio : 4 * x + 3 * x + 7 * x = 126)
  (h_halima : 4 * x = 36)
  (h_beckham : 3 * x = 27) :
  4 * x - 3 * x = 9 :=
by sorry

end age_difference_l143_143969


namespace fill_in_square_l143_143225

variable {α : Type*} [CommRing α]

theorem fill_in_square (a b : α) (square : α) (h : square * 3 * a * b = 3 * a^2 * b) : square = a :=
sorry

end fill_in_square_l143_143225


namespace sum_of_integers_is_eleven_l143_143787

theorem sum_of_integers_is_eleven (p q r s : ℤ) 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 11 := 
by
  sorry

end sum_of_integers_is_eleven_l143_143787


namespace walking_speed_l143_143177

theorem walking_speed (W : ℝ) : (1 / (1 / W + 1 / 8)) * 6 = 2.25 * (12 / 2) -> W = 4 :=
by
  intro h
  sorry

end walking_speed_l143_143177


namespace abs_neg_six_l143_143269

theorem abs_neg_six : abs (-6) = 6 :=
by
  -- Proof goes here
  sorry

end abs_neg_six_l143_143269


namespace bus_trip_distance_l143_143168

theorem bus_trip_distance
  (D S : ℕ) (H1 : S = 55)
  (H2 : D / S - 1 = D / (S + 5))
  : D = 660 :=
sorry

end bus_trip_distance_l143_143168


namespace sally_score_is_12_5_l143_143391

-- Conditions
def correctAnswers : ℕ := 15
def incorrectAnswers : ℕ := 10
def unansweredQuestions : ℕ := 5
def pointsPerCorrect : ℝ := 1.0
def pointsPerIncorrect : ℝ := -0.25
def pointsPerUnanswered : ℝ := 0.0

-- Score computation
noncomputable def sallyScore : ℝ :=
  (correctAnswers * pointsPerCorrect) + 
  (incorrectAnswers * pointsPerIncorrect) + 
  (unansweredQuestions * pointsPerUnanswered)

-- Theorem to prove Sally's score is 12.5
theorem sally_score_is_12_5 : sallyScore = 12.5 := by
  sorry

end sally_score_is_12_5_l143_143391


namespace diet_sodas_sold_l143_143318

theorem diet_sodas_sold (R D : ℕ) (h1 : R + D = 64) (h2 : R / D = 9 / 7) : D = 28 := 
by
  sorry

end diet_sodas_sold_l143_143318


namespace base_10_to_base_7_equiv_base_10_to_base_7_678_l143_143979

theorem base_10_to_base_7_equiv : (678 : ℕ) = 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 := 
by
  -- proof steps would go here
  sorry

theorem base_10_to_base_7_678 : "678 in base-7" = "1656" := 
by
  have h1 := base_10_to_base_7_equiv
  -- additional proof steps to show 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 = 1656 in base-7
  sorry

end base_10_to_base_7_equiv_base_10_to_base_7_678_l143_143979


namespace tea_bags_count_l143_143943

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l143_143943


namespace range_of_a_l143_143905

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1 ∧ (a^2 > a + 6 ∧ a + 6 > 0)) → (a > 3 ∨ (-6 < a ∧ a < -2)) :=
by
  intro h
  sorry

end range_of_a_l143_143905


namespace minimum_value_of_z_l143_143287

theorem minimum_value_of_z : ∃ (x : ℝ), ∀ (z : ℝ), (z = 4 * x^2 + 8 * x + 16) → z ≥ 12 :=
by
  sorry

end minimum_value_of_z_l143_143287


namespace ze_age_conditions_l143_143607

theorem ze_age_conditions 
  (z g t : ℕ)
  (h1 : z = 2 * g + 3 * t)
  (h2 : 2 * (z + 15) = 2 * (g + 15) + 3 * (t + 15))
  (h3 : 2 * (g + 15) = 3 * (t + 15)) :
  z = 45 ∧ t = 5 :=
by
  sorry

end ze_age_conditions_l143_143607


namespace binom_12_10_l143_143346

theorem binom_12_10 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_l143_143346


namespace find_k_l143_143888

theorem find_k (k : ℝ) (x₁ x₂ : ℝ)
  (h : x₁^2 + (2 * k - 1) * x₁ + k^2 - 1 = 0)
  (h' : x₂^2 + (2 * k - 1) * x₂ + k^2 - 1 = 0)
  (hx : x₁ ≠ x₂)
  (cond : x₁^2 + x₂^2 = 19) : k = -2 :=
sorry

end find_k_l143_143888


namespace abs_sub_eq_three_l143_143117

theorem abs_sub_eq_three {m n : ℝ} (h1 : m * n = 4) (h2 : m + n = 5) : |m - n| = 3 := 
sorry

end abs_sub_eq_three_l143_143117


namespace problem_l143_143207

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x ^ 2

noncomputable def f (gx : ℝ) (x : ℝ) : ℝ := (2 - 3 * x ^ 2) / x ^ 2

theorem problem (x : ℝ) (hx : x ≠ 0) : f (g x) x = 3 / 2 :=
  sorry

end problem_l143_143207


namespace kendra_change_and_discounts_l143_143857

-- Define the constants and conditions
def wooden_toy_price : ℝ := 20.0
def hat_price : ℝ := 10.0
def tax_rate : ℝ := 0.08
def discount_wooden_toys_2_3 : ℝ := 0.10
def discount_wooden_toys_4_or_more : ℝ := 0.15
def discount_hats_2 : ℝ := 0.05
def discount_hats_3_or_more : ℝ := 0.10
def kendra_bill : ℝ := 250.0
def kendra_wooden_toys : ℕ := 4
def kendra_hats : ℕ := 5

-- Calculate the applicable discounts based on conditions
def discount_on_wooden_toys : ℝ :=
  if kendra_wooden_toys >= 2 ∧ kendra_wooden_toys <= 3 then
    discount_wooden_toys_2_3
  else if kendra_wooden_toys >= 4 then
    discount_wooden_toys_4_or_more
  else
    0.0

def discount_on_hats : ℝ :=
  if kendra_hats = 2 then
    discount_hats_2
  else if kendra_hats >= 3 then
    discount_hats_3_or_more
  else
    0.0

-- Main theorem statement
theorem kendra_change_and_discounts :
  let total_cost_before_discounts := kendra_wooden_toys * wooden_toy_price + kendra_hats * hat_price
  let wooden_toys_discount := discount_on_wooden_toys * (kendra_wooden_toys * wooden_toy_price)
  let hats_discount := discount_on_hats * (kendra_hats * hat_price)
  let total_discounts := wooden_toys_discount + hats_discount
  let total_cost_after_discounts := total_cost_before_discounts - total_discounts
  let tax := tax_rate * total_cost_after_discounts
  let total_cost_after_tax := total_cost_after_discounts + tax
  let change_received := kendra_bill - total_cost_after_tax
  (total_discounts = 17) → 
  (change_received = 127.96) ∧ 
  (wooden_toys_discount = 12) ∧ 
  (hats_discount = 5) :=
by
  sorry

end kendra_change_and_discounts_l143_143857


namespace female_students_in_sample_l143_143236

/-- In a high school, there are 500 male students and 400 female students in the first grade. 
    If a random sample of size 45 is taken from the students of this grade using stratified sampling by gender, 
    the number of female students in the sample is 20. -/
theorem female_students_in_sample 
  (num_male : ℕ) (num_female : ℕ) (sample_size : ℕ)
  (h_male : num_male = 500)
  (h_female : num_female = 400)
  (h_sample : sample_size = 45)
  (total_students : ℕ := num_male + num_female)
  (sample_ratio : ℚ := sample_size / total_students) :
  num_female * sample_ratio = 20 := 
sorry

end female_students_in_sample_l143_143236


namespace Drew_older_than_Maya_by_5_l143_143195

variable (Maya Drew Peter John Jacob : ℕ)
variable (h1 : John = 30)
variable (h2 : John = 2 * Maya)
variable (h3 : Jacob = 11)
variable (h4 : Jacob + 2 = (Peter + 2) / 2)
variable (h5 : Peter = Drew + 4)

theorem Drew_older_than_Maya_by_5 : Drew = Maya + 5 :=
by
  have Maya_age : Maya = 30 / 2 := by sorry
  have Jacob_age_in_2_years : Jacob + 2 = 13 := by sorry
  have Peter_age_in_2_years : Peter + 2 = 2 * 13 := by sorry
  have Peter_age : Peter = 26 - 2 := by sorry
  have Drew_age : Drew = Peter - 4 := by sorry
  have Drew_older_than_Maya : Drew = Maya + 5 := by sorry
  exact Drew_older_than_Maya

end Drew_older_than_Maya_by_5_l143_143195


namespace larry_stickers_l143_143920

theorem larry_stickers (initial_stickers : ℕ) (lost_stickers : ℕ) (final_stickers : ℕ) 
  (initial_eq_93 : initial_stickers = 93) 
  (lost_eq_6 : lost_stickers = 6) 
  (final_eq : final_stickers = initial_stickers - lost_stickers) : 
  final_stickers = 87 := 
  by 
  -- proof goes here
  sorry

end larry_stickers_l143_143920


namespace count_ball_distribution_l143_143303

theorem count_ball_distribution (A B C D : ℕ) (balls : ℕ) :
  (A + B > C + D ∧ A + B + C + D = balls) → 
  (balls = 30) →
  (∃ n, n = 2600) :=
by
  intro h_ball_dist h_balls
  sorry

end count_ball_distribution_l143_143303


namespace percent_chemical_a_in_mixture_l143_143133

-- Define the given problem parameters
def percent_chemical_a_in_solution_x : ℝ := 0.30
def percent_chemical_a_in_solution_y : ℝ := 0.40
def proportion_of_solution_x_in_mixture : ℝ := 0.80
def proportion_of_solution_y_in_mixture : ℝ := 1.0 - proportion_of_solution_x_in_mixture

-- Define what we need to prove: the percentage of chemical a in the mixture
theorem percent_chemical_a_in_mixture:
  (percent_chemical_a_in_solution_x * proportion_of_solution_x_in_mixture) + 
  (percent_chemical_a_in_solution_y * proportion_of_solution_y_in_mixture) = 0.32 
:= by sorry

end percent_chemical_a_in_mixture_l143_143133


namespace nicolai_ate_6_pounds_of_peaches_l143_143145

noncomputable def total_weight_pounds : ℝ := 8
noncomputable def pound_to_ounce : ℝ := 16
noncomputable def mario_weight_ounces : ℝ := 8
noncomputable def lydia_weight_ounces : ℝ := 24

theorem nicolai_ate_6_pounds_of_peaches :
  (total_weight_pounds * pound_to_ounce - (mario_weight_ounces + lydia_weight_ounces)) / pound_to_ounce = 6 :=
by
  sorry

end nicolai_ate_6_pounds_of_peaches_l143_143145


namespace prob_composite_in_first_50_l143_143982

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (∀ m : ℕ, m > 1 → m < n → ¬ m ∣ n)

-- Define the set of first 50 natural numbers
def first_50_numbers : list ℕ :=
  (list.range 50).map (λ n, n + 1)

-- Define the set of composite numbers within the first 50 natural numbers
def composite_numbers : list ℕ :=
  first_50_numbers.filter is_composite

-- Define the probability function
noncomputable def probability_of_composite : ℚ :=
  composite_numbers.length / first_50_numbers.length

-- The theorem statement
theorem prob_composite_in_first_50 : probability_of_composite = 34 / 50 :=
by sorry

end prob_composite_in_first_50_l143_143982


namespace smallest_integer_n_l143_143630

theorem smallest_integer_n (n : ℕ) (h₁ : 50 ∣ n^2) (h₂ : 294 ∣ n^3) : n = 210 :=
sorry

end smallest_integer_n_l143_143630


namespace sum_of_coordinates_of_A_l143_143247

open Real

theorem sum_of_coordinates_of_A (A B C : ℝ × ℝ) (h1 : B = (2, 8)) (h2 : C = (5, 2))
  (h3 : ∃ (k : ℝ), A = ((2 * (B.1:ℝ) + C.1) / 3, (2 * (B.2:ℝ) + C.2) / 3) ∧ k = 1/3) :
  A.1 + A.2 = 9 :=
sorry

end sum_of_coordinates_of_A_l143_143247


namespace train_length_1080_l143_143458

def length_of_train (speed time : ℕ) : ℕ := speed * time

theorem train_length_1080 (speed time : ℕ) (h1 : speed = 108) (h2 : time = 10) : length_of_train speed time = 1080 := by
  sorry

end train_length_1080_l143_143458


namespace mice_path_count_l143_143955

theorem mice_path_count
  (x y : ℕ)
  (left_house_yesterday top_house_yesterday right_house_yesterday : ℕ)
  (left_house_today top_house_today right_house_today : ℕ)
  (h_left_yesterday : left_house_yesterday = 8)
  (h_top_yesterday : top_house_yesterday = 4)
  (h_right_yesterday : right_house_yesterday = 7)
  (h_left_today : left_house_today = 4)
  (h_top_today : top_house_today = 4)
  (h_right_today : right_house_today = 7)
  (h_eq : (left_house_yesterday - left_house_today) + 
          (right_house_yesterday - right_house_today) = 
          top_house_today - top_house_yesterday) :
  x + y = 11 :=
by
  sorry

end mice_path_count_l143_143955


namespace greatest_unexpressible_sum_l143_143649

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem greatest_unexpressible_sum : 
  ∀ (n : ℕ), (∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n) → n ≤ 11 :=
by
  sorry

end greatest_unexpressible_sum_l143_143649


namespace find_b_l143_143064

theorem find_b (a b : ℝ) (x : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^5) * x^5):
  b = 40 :=
  sorry

end find_b_l143_143064


namespace smallest_perimeter_of_square_sides_l143_143588

/-
  Define a predicate for the triangle inequality condition for squares of integers.
-/
def triangle_ineq_squares (a b c : ℕ) : Prop :=
  (a < b) ∧ (b < c) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2)

/-
  Statement that proves the smallest possible perimeter given the conditions.
-/
theorem smallest_perimeter_of_square_sides : 
  ∃ a b c : ℕ, a < b ∧ b < c ∧ triangle_ineq_squares a b c ∧ a^2 + b^2 + c^2 = 77 :=
sorry

end smallest_perimeter_of_square_sides_l143_143588


namespace smallest_a_such_that_sqrt_50a_is_integer_l143_143224

theorem smallest_a_such_that_sqrt_50a_is_integer : ∃ a : ℕ, (∀ b : ℕ, (b > 0 ∧ (∃ k : ℕ, 50 * b = k^2)) → (a ≤ b)) ∧ (∃ k : ℕ, 50 * a = k^2) ∧ a = 2 := 
by
  sorry

end smallest_a_such_that_sqrt_50a_is_integer_l143_143224


namespace carpet_rate_proof_l143_143183

noncomputable def carpet_rate (breadth_first : ℝ) (length_ratio : ℝ) (cost_second : ℝ) : ℝ :=
  let length_first := length_ratio * breadth_first
  let area_first := length_first * breadth_first
  let length_second := length_first * 1.4
  let breadth_second := breadth_first * 1.25
  let area_second := length_second * breadth_second 
  cost_second / area_second

theorem carpet_rate_proof : carpet_rate 6 1.44 4082.4 = 45 :=
by
  -- Here we provide the goal and state what needs to be proven.
  sorry

end carpet_rate_proof_l143_143183


namespace angle_difference_l143_143510

theorem angle_difference (A B : ℝ) 
  (h1 : A = 85) 
  (h2 : A + B = 180) : B - A = 10 := 
by sorry

end angle_difference_l143_143510


namespace smallest_y_l143_143363

theorem smallest_y (y : ℕ) (h : 56 * y + 8 ≡ 6 [MOD 26]) : y = 6 := by
  sorry

end smallest_y_l143_143363


namespace largest_integer_less_than_100_with_remainder_5_l143_143702

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l143_143702


namespace sum_of_elements_equal_l143_143921

-- Let X be a set of 8 consecutive positive integers
constant X : Finset ℕ
constant A B : Finset ℕ
constant n : ℕ

-- Conditions
axiom X_def : X = {n, n+1, n+2, n+3, n+4, n+5, n+6, n+7}
axiom A_B_disjoint : A ∩ B = ∅
axiom A_B_union : A ∪ B = X
axiom A_B_size : A.card = 4 ∧ B.card = 4
axiom A_B_squares_equal : (∑ x in A, x^2) = (∑ x in B, x^2)

-- Theorem to prove
theorem sum_of_elements_equal (hX : X_def) (hA : A_B_disjoint) (hB : A_B_union)
    (hC : A_B_size) (hD : A_B_squares_equal) : (∑ x in A, x) = (∑ x in B, x) :=
sorry

end sum_of_elements_equal_l143_143921


namespace scientific_notation_of_diameter_l143_143960

theorem scientific_notation_of_diameter :
  0.00000258 = 2.58 * 10^(-6) :=
by sorry

end scientific_notation_of_diameter_l143_143960


namespace jerry_weekly_earnings_l143_143109

-- Definitions of the given conditions
def pay_per_task : ℕ := 40
def hours_per_task : ℕ := 2
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

-- Calculated values from the conditions
def tasks_per_day : ℕ := hours_per_day / hours_per_task
def tasks_per_week : ℕ := tasks_per_day * days_per_week
def total_earnings : ℕ := pay_per_task * tasks_per_week

-- Theorem to prove
theorem jerry_weekly_earnings : total_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l143_143109


namespace part_a_part_b_l143_143976

-- Define n_mid_condition
def n_mid_condition (n : ℕ) : Prop := n % 2 = 1 ∧ n ∣ 2023^n - 1

-- Part a:
theorem part_a : ∃ (k₁ k₂ : ℕ), k₁ = 3 ∧ k₂ = 9 ∧ n_mid_condition k₁ ∧ n_mid_condition k₂ := by
  sorry

-- Part b:
theorem part_b : ∀ k, k ≥ 1 → n_mid_condition (3^k) := by
  sorry

end part_a_part_b_l143_143976


namespace simplify_expression_l143_143416

theorem simplify_expression :
  (1 / (1 / (1 / 3 : ℝ)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)) = (1 / 39 : ℝ) :=
by
  sorry

end simplify_expression_l143_143416


namespace hayden_ironing_weeks_l143_143330

variable (total_daily_minutes : Nat := 5 + 3)
variable (days_per_week : Nat := 5)
variable (total_minutes : Nat := 160)

def calculate_weeks (total_daily_minutes : Nat) (days_per_week : Nat) (total_minutes : Nat) : Nat :=
  total_minutes / (total_daily_minutes * days_per_week)

theorem hayden_ironing_weeks :
  calculate_weeks (5 + 3) 5 160 = 4 := 
by
  sorry

end hayden_ironing_weeks_l143_143330


namespace people_per_car_l143_143167

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 3) : total_people / num_cars = 21 :=
by
  sorry

end people_per_car_l143_143167


namespace hawks_total_points_l143_143790

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_total_points : total_points touchdowns points_per_touchdown = 21 := 
by 
  sorry

end hawks_total_points_l143_143790


namespace mean_score_is_93_l143_143244

-- Define Jane's scores as a list
def scores : List ℕ := [98, 97, 92, 85, 93]

-- Define the mean of the scores
noncomputable def mean (lst : List ℕ) : ℚ := 
  (lst.foldl (· + ·) 0 : ℚ) / lst.length

-- The theorem to prove
theorem mean_score_is_93 : mean scores = 93 := by
  sorry

end mean_score_is_93_l143_143244


namespace distance_to_origin_eq_three_l143_143425

theorem distance_to_origin_eq_three :
  let P := (1, 2, 2)
  let origin := (0, 0, 0)
  dist P origin = 3 := by
  sorry

end distance_to_origin_eq_three_l143_143425


namespace geom_seq_sum_l143_143399

theorem geom_seq_sum (a : ℕ → ℝ) (n : ℕ) (q : ℝ) (h1 : a 1 = 2) (h2 : a 1 * a 5 = 64) :
  (a 1 * (1 - q^n)) / (1 - q) = 2^(n+1) - 2 := 
sorry

end geom_seq_sum_l143_143399


namespace smallest_interesting_number_l143_143840

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l143_143840


namespace squares_centers_equal_perpendicular_l143_143272

def Square (center : (ℝ × ℝ)) (side : ℝ) := {p : ℝ × ℝ // abs (p.1 - center.1) ≤ side / 2 ∧ abs (p.2 - center.2) ≤ side / 2}

theorem squares_centers_equal_perpendicular 
  (a b : ℝ)
  (O A B C : ℝ × ℝ)
  (hA : A = (a, a))
  (hB : B = (b, 2 * a + b))
  (hC : C = (- (a + b), a + b))
  (hO_vertex : O = (0, 0)) :
  dist O B = dist A C ∧ ∃ m₁ m₂ : ℝ, (B.2 - O.2) / (B.1 - O.1) = m₁ ∧ (C.2 - A.2) / (C.1 - A.1) = m₂ ∧ m₁ * m₂ = -1 := sorry

end squares_centers_equal_perpendicular_l143_143272


namespace quadratic_sum_roots_l143_143736

-- We define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- The function f passes through points (r, k) and (s, k)
variables (a b c r s k : ℝ)
variable (ha : a ≠ 0)
variable (hr : f a b c r = k)
variable (hs : f a b c s = k)

-- What we want to prove
theorem quadratic_sum_roots :
  f a b c (r + s) = c :=
sorry

end quadratic_sum_roots_l143_143736


namespace age_difference_l143_143970

theorem age_difference (h b m : ℕ) (ratio : h = 4 * m ∧ b = 3 * m ∧ 4 * m + 3 * m + 7 * m = 126) : h - b = 9 :=
by
  -- proof will be filled here
  sorry

end age_difference_l143_143970


namespace total_books_in_series_l143_143595

-- Definitions for the conditions
def books_read : ℕ := 8
def books_to_read : ℕ := 6

-- Statement to be proved
theorem total_books_in_series : books_read + books_to_read = 14 := by
  sorry

end total_books_in_series_l143_143595


namespace money_spent_on_video_games_l143_143376

theorem money_spent_on_video_games :
  let total_money := 50
  let fraction_books := 1 / 4
  let fraction_snacks := 2 / 5
  let fraction_apps := 1 / 5
  let spent_books := fraction_books * total_money
  let spent_snacks := fraction_snacks * total_money
  let spent_apps := fraction_apps * total_money
  let spent_other := spent_books + spent_snacks + spent_apps
  let spent_video_games := total_money - spent_other
  spent_video_games = 7.5 :=
by
  sorry

end money_spent_on_video_games_l143_143376


namespace Gwen_still_has_money_in_usd_l143_143491

open Real

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def usd_gift : ℝ := 5.00
noncomputable def eur_gift : ℝ := 20.00
noncomputable def usd_spent_on_candy : ℝ := 3.25
noncomputable def eur_spent_on_toy : ℝ := 5.50

theorem Gwen_still_has_money_in_usd :
  let eur_conversion_to_usd := eur_gift / exchange_rate
  let total_usd_received := usd_gift + eur_conversion_to_usd
  let usd_spent_on_toy := eur_spent_on_toy / exchange_rate
  let total_usd_spent := usd_spent_on_candy + usd_spent_on_toy
  total_usd_received - total_usd_spent = 18.81 :=
by
  sorry

end Gwen_still_has_money_in_usd_l143_143491


namespace line_tangent_circle_iff_m_l143_143230

/-- Definition of the circle and the line -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- Prove that the line is tangent to the circle if and only if m = -3 or m = -13 -/
theorem line_tangent_circle_iff_m (m : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq x y m) ↔ m = -3 ∨ m = -13 :=
by
  sorry

end line_tangent_circle_iff_m_l143_143230


namespace find_integer_l143_143972

theorem find_integer (N : ℤ) (hN : N^2 + N = 12) (h_pos : 0 < N) : N = 3 :=
sorry

end find_integer_l143_143972


namespace person_c_completion_time_l143_143989

def job_completion_days (Ra Rb Rc : ℚ) (total_earnings b_earnings : ℚ) : ℚ :=
  Rc

theorem person_c_completion_time (Ra Rb Rc : ℚ)
  (hRa : Ra = 1 / 6)
  (hRb : Rb = 1 / 8)
  (total_earnings : ℚ)
  (b_earnings : ℚ)
  (earnings_ratio : b_earnings / total_earnings = Rb / (Ra + Rb + Rc))
  : Rc = 1 / 12 :=
sorry

end person_c_completion_time_l143_143989


namespace find_c_for_min_value_zero_l143_143199

theorem find_c_for_min_value_zero :
  ∃ c : ℝ, c = 1 ∧ (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 ≥ 0) ∧
  (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 = 0 → c = 1) :=
by
  use 1
  sorry

end find_c_for_min_value_zero_l143_143199


namespace functional_relationship_find_selling_price_maximum_profit_l143_143451

noncomputable def linear_relation (x : ℤ) : ℤ := -5 * x + 150
def profit_function (x : ℤ) : ℤ := -5 * x * x + 200 * x - 1500

theorem functional_relationship (x : ℤ) (hx : 10 ≤ x ∧ x ≤ 15) : linear_relation x = -5 * x + 150 :=
by sorry

theorem find_selling_price (h : ∃ x : ℤ, (10 ≤ x ∧ x ≤ 15) ∧ ((-5 * x + 150) * (x - 10) = 320)) :
  ∃ x : ℤ, x = 14 :=
by sorry

theorem maximum_profit (hx : 10 ≤ 15 ∧ 15 ≤ 15) : profit_function 15 = 375 :=
by sorry

end functional_relationship_find_selling_price_maximum_profit_l143_143451


namespace parallel_lines_l143_143798

theorem parallel_lines (a : ℝ) :
  (∀ x y, x + a^2 * y + 6 = 0 → (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end parallel_lines_l143_143798


namespace mail_distribution_l143_143174

theorem mail_distribution (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : total_mail / total_houses = 6 := by
  sorry

end mail_distribution_l143_143174


namespace min_sum_of_exponents_of_powers_of_2_l143_143383

theorem min_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ (s : set ℕ), (∀ (k ∈ s), ∃ (m : ℕ), k = 2 ^ m) ∧ (s.sum id = 520) ∧ (s.sum id = s.card * s.card) → (s.sum id = 12) := sorry

end min_sum_of_exponents_of_powers_of_2_l143_143383


namespace parabola_vertex_y_coordinate_l143_143193

theorem parabola_vertex_y_coordinate (x y : ℝ) :
  y = 5 * x^2 + 20 * x + 45 ∧ (∃ h k, y = 5 * (x + h)^2 + k ∧ k = 25) :=
by
  sorry

end parabola_vertex_y_coordinate_l143_143193


namespace tea_bags_count_l143_143941

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l143_143941


namespace positive_integer_count_l143_143707

/-
  Prove that the number of positive integers \( n \) for which \( \frac{n(n+1)}{2} \) divides \( 30n \) is 11.
-/

theorem positive_integer_count (n : ℕ) :
  (∃ k : ℕ, k > 0 ∧ k ≤ 11 ∧ (2 * 30 * n) % (n * (n + 1)) = 0) :=
sorry

end positive_integer_count_l143_143707


namespace perpendicular_vectors_k_zero_l143_143374

theorem perpendicular_vectors_k_zero
  (k : ℝ)
  (a : ℝ × ℝ := (3, 1))
  (b : ℝ × ℝ := (1, 3))
  (c : ℝ × ℝ := (k, 2)) 
  (h : (a.1 - c.1, a.2 - c.2).1 * b.1 + (a.1 - c.1, a.2 - c.2).2 * b.2 = 0) :
  k = 0 :=
by
  sorry

end perpendicular_vectors_k_zero_l143_143374


namespace count_integers_between_sqrt_10_and_sqrt_100_l143_143729

theorem count_integers_between_sqrt_10_and_sqrt_100 :
  ∃ n : ℕ, (∀ k : ℕ, 4 ≤ k ∧ k ≤ 10 → k ∈ set.Ico (int.floor (sqrt (10 : ℝ))) (int.ceil (sqrt (100 : ℝ)))) ∧ n = 7 :=
sorry

end count_integers_between_sqrt_10_and_sqrt_100_l143_143729


namespace adam_earnings_after_taxes_l143_143181

theorem adam_earnings_after_taxes
  (daily_earnings : ℕ) 
  (tax_pct : ℕ)
  (workdays : ℕ)
  (H1 : daily_earnings = 40) 
  (H2 : tax_pct = 10) 
  (H3 : workdays = 30) : 
  (daily_earnings - daily_earnings * tax_pct / 100) * workdays = 1080 := 
by
  -- Proof to be filled in
  sorry

end adam_earnings_after_taxes_l143_143181


namespace sequence_geometric_condition_l143_143243

theorem sequence_geometric_condition
  (a : ℕ → ℤ)
  (p q : ℤ)
  (h1 : a 1 = -1)
  (h2 : ∀ n, a (n + 1) = 2 * (a n - n + 3))
  (h3 : ∀ n, (a (n + 1) - p * (n + 1) + q) = 2 * (a n - p * n + q)) :
  a (Int.natAbs (p + q)) = 40 :=
sorry

end sequence_geometric_condition_l143_143243


namespace female_kittens_count_l143_143257

theorem female_kittens_count (initial_cats total_cats male_kittens female_kittens : ℕ)
  (h1 : initial_cats = 2)
  (h2 : total_cats = 7)
  (h3 : male_kittens = 2)
  (h4 : female_kittens = total_cats - initial_cats - male_kittens) :
  female_kittens = 3 :=
by
  sorry

end female_kittens_count_l143_143257


namespace faucet_fill_time_l143_143710

theorem faucet_fill_time (r : ℝ) (T1 T2 t : ℝ) (F1 F2 : ℕ) (h1 : T1 = 200) (h2 : t = 8) (h3 : F1 = 4) (h4 : F2 = 8) (h5 : T2 = 50) (h6 : r * F1 * t = T1) : 
(F2 * r) * t / (F1 * F2) = T2 -> by sorry := sorry

#check faucet_fill_time

end faucet_fill_time_l143_143710


namespace sum_of_squares_of_rates_equals_536_l143_143479

-- Define the biking, jogging, and swimming rates as integers.
variables (b j s : ℤ)

-- Condition: Ed's total distance equation.
def ed_distance_eq : Prop := 3 * b + 2 * j + 4 * s = 80

-- Condition: Sue's total distance equation.
def sue_distance_eq : Prop := 4 * b + 3 * j + 2 * s = 98

-- The main statement to prove.
theorem sum_of_squares_of_rates_equals_536 (hb : b ≥ 0) (hj : j ≥ 0) (hs : s ≥ 0) 
  (h1 : ed_distance_eq b j s) (h2 : sue_distance_eq b j s) :
  b^2 + j^2 + s^2 = 536 :=
by sorry

end sum_of_squares_of_rates_equals_536_l143_143479


namespace union_sets_l143_143525

-- Define the sets A and B as conditions
def A : Set ℝ := {0, 1}  -- Since lg 1 = 0
def B : Set ℝ := {-1, 0}

-- Define that A union B equals {-1, 0, 1}
theorem union_sets : A ∪ B = {-1, 0, 1} := by
  sorry

end union_sets_l143_143525


namespace sin_15_mul_sin_75_l143_143304

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end sin_15_mul_sin_75_l143_143304


namespace least_sum_exponents_of_520_l143_143382

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end least_sum_exponents_of_520_l143_143382


namespace simplify_expression_l143_143417

theorem simplify_expression :
  (1 / (1 / (1 / 3 : ℝ)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)) = (1 / 39 : ℝ) :=
by
  sorry

end simplify_expression_l143_143417


namespace smallest_value_c_zero_l143_143186

noncomputable def smallest_possible_c (a b c : ℝ) : ℝ :=
if h : (0 < a) ∧ (0 < b) ∧ (0 < c) then
  0
else
  c

theorem smallest_value_c_zero (a b c : ℝ) (h : (0 < a) ∧ (0 < b) ∧ (0 < c)) :
  smallest_possible_c a b c = 0 :=
by
  sorry

end smallest_value_c_zero_l143_143186


namespace total_number_of_apples_l143_143782

namespace Apples

def red_apples : ℕ := 7
def green_apples : ℕ := 2
def total_apples : ℕ := red_apples + green_apples

theorem total_number_of_apples : total_apples = 9 := by
  -- Definition of total_apples is used directly from conditions.
  -- Conditions state there are 7 red apples and 2 green apples.
  -- Therefore, total_apples = 7 + 2 = 9.
  sorry

end Apples

end total_number_of_apples_l143_143782


namespace range_m_graph_in_quadrants_l143_143388

theorem range_m_graph_in_quadrants (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (m + 2) / x > 0) ∧ (x < 0 → (m + 2) / x < 0))) ↔ m > -2 :=
by 
  sorry

end range_m_graph_in_quadrants_l143_143388


namespace largest_int_with_remainder_5_lt_100_l143_143684

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l143_143684


namespace remainder_43_pow_97_pow_5_plus_109_mod_163_l143_143159

theorem remainder_43_pow_97_pow_5_plus_109_mod_163 :
    (43 ^ (97 ^ 5) + 109) % 163 = 50 :=
by
  sorry

end remainder_43_pow_97_pow_5_plus_109_mod_163_l143_143159


namespace true_value_of_product_l143_143019

theorem true_value_of_product (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  let product := (100 * a + 10 * b + c) * (100 * b + 10 * c + a) * (100 * c + 10 * a + b)
  product = 2342355286 → (product % 10 = 6) → product = 328245326 :=
by
  sorry

end true_value_of_product_l143_143019


namespace mechanic_earns_on_fourth_day_l143_143550

theorem mechanic_earns_on_fourth_day 
  (E1 E2 E3 E4 E5 E6 E7 : ℝ)
  (h1 : (E1 + E2 + E3 + E4) / 4 = 18)
  (h2 : (E4 + E5 + E6 + E7) / 4 = 22)
  (h3 : (E1 + E2 + E3 + E4 + E5 + E6 + E7) / 7 = 21) 
  : E4 = 13 := 
by 
  sorry

end mechanic_earns_on_fourth_day_l143_143550


namespace smallest_natural_number_divisible_l143_143203

theorem smallest_natural_number_divisible :
  ∃ n : ℕ, (n^2 + 14 * n + 13) % 68 = 0 ∧ 
          ∀ m : ℕ, (m^2 + 14 * m + 13) % 68 = 0 → 21 ≤ m :=
by 
  sorry

end smallest_natural_number_divisible_l143_143203


namespace correct_answer_is_option_d_l143_143987

def is_quadratic (eq : String) : Prop :=
  eq = "a*x^2 + b*x + c = 0"

def OptionA : String := "1/x^2 + x - 1 = 0"
def OptionB : String := "3x + 1 = 5x + 4"
def OptionC : String := "x^2 + y = 0"
def OptionD : String := "x^2 - 2x + 1 = 0"

theorem correct_answer_is_option_d :
  is_quadratic OptionD :=
by
  sorry

end correct_answer_is_option_d_l143_143987


namespace total_oranges_l143_143260

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end total_oranges_l143_143260


namespace volume_of_hall_l143_143990

-- Define the dimensions and areas conditions
def length_hall : ℝ := 15
def breadth_hall : ℝ := 12
def area_floor_ceiling : ℝ := 2 * (length_hall * breadth_hall)
def area_walls (h : ℝ) : ℝ := 2 * (length_hall * h) + 2 * (breadth_hall * h)

-- Given condition: The sum of the areas of the floor and ceiling is equal to the sum of the areas of the four walls
def condition (h : ℝ) : Prop := area_floor_ceiling = area_walls h

-- Define the volume of the hall
def volume_hall (h : ℝ) : ℝ := length_hall * breadth_hall * h

-- The theorem to be proven: given the condition, the volume equals 8004
theorem volume_of_hall : ∃ h : ℝ, condition h ∧ volume_hall h = 8004 := by
  sorry

end volume_of_hall_l143_143990


namespace range_of_m_l143_143217

noncomputable def distance (m : ℝ) : ℝ := (|m| * Real.sqrt 2 / 2)
theorem range_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.1 + A.2 + m = 0 ∧ B.1 + B.2 + m = 0) ∧
    (A.1 ^ 2 + A.2 ^ 2 = 2 ∧ B.1 ^ 2 + B.2 ^ 2 = 2) ∧
    (Real.sqrt (A.1 ^ 2 + A.2 ^ 2) + Real.sqrt (B.1 ^ 2 + B.2 ^ 2) ≥ 
     Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)) ∧ (distance m < Real.sqrt 2)) ↔ 
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) 2 := 
sorry

end range_of_m_l143_143217


namespace gcd_3_666666666_equals_3_l143_143763

theorem gcd_3_666666666_equals_3 :
  Nat.gcd 33333333 666666666 = 3 := by
  sorry

end gcd_3_666666666_equals_3_l143_143763


namespace juan_marbles_eq_64_l143_143165

def connie_marbles : ℕ := 39
def juan_extra_marbles : ℕ := 25

theorem juan_marbles_eq_64 : (connie_marbles + juan_extra_marbles) = 64 :=
by
  -- definition and conditions handled above
  sorry

end juan_marbles_eq_64_l143_143165
