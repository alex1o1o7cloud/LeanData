import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Field
import Mathlib.Algebra.GCD.Basic
import Mathlib.Algebra.Inverses
import Mathlib.Algebra.ModularArithmetic
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinations
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circles
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Real
import algebra.big_operators.basic
import data.nat.basic
import data.real.basic
import tactic

namespace julie_hourly_rate_l107_107490

variable (daily_hours : ℕ) (weekly_days : ℕ) (monthly_weeks : ℕ) (missed_days : ℕ) (monthly_salary : ℝ)

def total_monthly_hours : ℕ := daily_hours * weekly_days * monthly_weeks - daily_hours * missed_days

theorem julie_hourly_rate : 
    daily_hours = 8 → 
    weekly_days = 6 → 
    monthly_weeks = 4 → 
    missed_days = 1 → 
    monthly_salary = 920 → 
    (monthly_salary / total_monthly_hours daily_hours weekly_days monthly_weeks missed_days) = 5 := by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end julie_hourly_rate_l107_107490


namespace correct_transformation_l107_107224

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : (a / b = 2 * a / 2 * b) :=
by
  sorry

end correct_transformation_l107_107224


namespace part_i_part_ii_l107_107734

variable (a : ℕ → ℝ)

axiom a_1 : a 1 = 1
axiom a_recurrence : ∀ n ∈ ℕ, a (n+1) * a n = 1 / n

theorem part_i (n : ℕ) (hn : n > 0) : a (n+2) / n = a n / (n+1) :=
by
  sorry

theorem part_ii (n : ℕ) (hn : n > 0) : 
  2 * ((sqrt (n+1) - 1)) ≤ (∑ i in Finset.range (n+2).filter (λ k, k ≥ 2), 1 / (i * a (1 + i))) ∧
  (∑ i in Finset.range (n+2).filter (λ k, k ≥ 2), 1 / (i * a (1 + i))) ≤ n :=
by
  sorry

end part_i_part_ii_l107_107734


namespace m_is_perfect_square_l107_107493

theorem m_is_perfect_square
  (l m n : ℕ) (p : ℕ)
  (p_prime : prime p)
  (hl : l > 0) (hm : m > 0) (hn : n > 0)
  (h : ∃ k : ℕ, p^(2*l-1) * m * (mn+1)^2 + m^2 = k^2):
  ∃ k : ℕ, m = k^2 :=
by sorry

end m_is_perfect_square_l107_107493


namespace f_two_thirds_l107_107414

def f : ℝ → ℝ
| x => if x ≤ 0 then Real.sin (Real.pi * x) else f (x - 1)

theorem f_two_thirds : f (2 / 3) = - (Real.sqrt 3 / 2) := by
  sorry

end f_two_thirds_l107_107414


namespace max_min_diff_of_c_l107_107503

-- Definitions and conditions
variables (a b c : ℝ)
def condition1 := a + b + c = 6
def condition2 := a^2 + b^2 + c^2 = 18

-- Theorem statement
theorem max_min_diff_of_c (h1 : condition1 a b c) (h2 : condition2 a b c) :
  ∃ (c_max c_min : ℝ), c_max = 6 ∧ c_min = -2 ∧ (c_max - c_min = 8) :=
by
  sorry

end max_min_diff_of_c_l107_107503


namespace graph_empty_l107_107703

theorem graph_empty (x y : ℝ) : 
  x^2 + 3 * y^2 - 4 * x - 6 * y + 9 ≠ 0 :=
by
  -- Proof omitted
  sorry

end graph_empty_l107_107703


namespace smallest_prime_less_than_square_l107_107150

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107150


namespace smallest_prime_12_less_than_square_l107_107175

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107175


namespace set_inter_union_complement_l107_107726

noncomputable theory -- Only if necessary

-- Definitions of the sets based on the given conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }
def complement_B : Set ℝ := { x | x ≤ 2 ∨ x ≥ 4 }

-- Statement of the equivalent proof problem
theorem set_inter_union_complement :
  (A ∩ B = { x | 2 < x ∧ x ≤ 3 }) ∧
  (A ∪ complement_B = { x | x ≤ 3 ∨ x ≥ 4 }) :=
by
  sorry -- Proof to be filled in

end set_inter_union_complement_l107_107726


namespace cara_age_is_40_l107_107298

-- Defining the ages of Cara, her mom, and her grandmother
variables (cara_age mom_age grandmother_age - : ℕ)
variable (h1 : cara_age = mom_age - 20)
variable (h2 : mom_age = grandmother_age - 15)
variable (h3 : grandmother_age = 75)

-- The aim is to prove that Cara's age is 40
theorem cara_age_is_40 : cara_age = 40 :=
by
  sorry

end cara_age_is_40_l107_107298


namespace toothpick_grid_count_l107_107130

/-- Toothpick grid calculation -/
theorem toothpick_grid_count : 
  (let length := 40 in
   let width := 25 in
   let verticals := (length + 1) + 1 in
   let horizontals := width + 1 in
   (verticals * width) + (horizontals * length) = 2090) :=
by
  let length := 40
  let width := 25
  let verticals := (length + 1) + 1
  let horizontals := width + 1
  calc
    (verticals * width) + (horizontals * length) = sorry

end toothpick_grid_count_l107_107130


namespace find_base_length_of_isosceles_triangle_l107_107023

noncomputable def base_length (r1 r2 : ℝ) : ℝ := 3 * Real.sqrt 6

theorem find_base_length_of_isosceles_triangle
  (a b h : ℝ)
  (isosceles_triangle : (a = a))
  (incircle_radius : b = 3)
  (smaller_circle_radius : h = 2)
  (relationship_tangent : ∀ p : ℝ, (p == incircle_radius - smaller_circle_radius)) : 
  ∃ (base : ℝ), base = base_length b h :=
by
  use 3 * Real.sqrt 6
  sorry

end find_base_length_of_isosceles_triangle_l107_107023


namespace sample_size_correct_l107_107267

def sample_size (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ) (S : ℕ) : Prop :=
  sum_frequencies = 20 ∧ frequency_sum_ratio = 0.4 → S = 50

theorem sample_size_correct :
  ∀ (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ),
    sample_size sum_frequencies frequency_sum_ratio 50 :=
by
  intros sum_frequencies frequency_sum_ratio
  sorry

end sample_size_correct_l107_107267


namespace angle_ECD_eq_50_l107_107462

theorem angle_ECD_eq_50
  (A B C D E : Type)
  (m : Type → ℝ)
  [linear_ordered_field ℝ] -- Assuming the angles are in ℝ
  (h1 : AC < BC)
  (h2 : m(D, C, B) = 50)
  (h3 : CD ∥ AB) :
  m(E, C, D) = 50 :=
by
  sorry -- Proof not required

end angle_ECD_eq_50_l107_107462


namespace sum_of_squares_of_roots_l107_107563

theorem sum_of_squares_of_roots 
  (x1 x2 : ℝ) 
  (h₁ : 5 * x1^2 - 6 * x1 - 4 = 0)
  (h₂ : 5 * x2^2 - 6 * x2 - 4 = 0)
  (h₃ : x1 ≠ x2) :
  x1^2 + x2^2 = 76 / 25 := sorry

end sum_of_squares_of_roots_l107_107563


namespace smallest_prime_less_than_perfect_square_is_13_l107_107201

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107201


namespace compute_expr_at_three_l107_107325

theorem compute_expr_at_three :
  (let x := 3 in (x^6 - 9*x^3 + 27) / (x^3 - 3) = 24.75) :=
by
  sorry

end compute_expr_at_three_l107_107325


namespace smallest_prime_less_than_perfect_square_l107_107157

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107157


namespace a_2023_le_1_l107_107512

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, (a (n+1))^2 + a n * a (n+2) ≤ a n + a (n+2))

theorem a_2023_le_1 : a 2023 ≤ 1 := by
  sorry

end a_2023_le_1_l107_107512


namespace value_of_one_TV_mixer_blender_l107_107605

variables (M T B : ℝ)

-- The given conditions
def eq1 : Prop := 2 * M + T + B = 10500
def eq2 : Prop := T + M + 2 * B = 14700

-- The problem: find the combined value of one TV, one mixer, and one blender
theorem value_of_one_TV_mixer_blender :
  eq1 M T B → eq2 M T B → (T + M + B = 18900) :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end value_of_one_TV_mixer_blender_l107_107605


namespace circle_diameter_equality_l107_107235

theorem circle_diameter_equality (r d : ℝ) (h₁ : d = 2 * r) (h₂ : π * d = π * r^2) : d = 4 :=
by
  sorry

end circle_diameter_equality_l107_107235


namespace median_equality_range_inequality_l107_107754

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l107_107754


namespace prime_solution_unique_l107_107391

theorem prime_solution_unique {x y : ℕ} 
  (hx : Nat.Prime x)
  (hy : Nat.Prime y)
  (h : x ^ y - y ^ x = x * y ^ 2 - 19) :
  (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
sorry

end prime_solution_unique_l107_107391


namespace price_of_each_large_fry_is_correct_l107_107978

def spongebob_price_of_large_fry
  (burgers_sold : ℕ := 30)
  (price_per_burger : ℝ := 2)
  (fries_sold : ℕ := 12)
  (total_earnings : ℝ := 78) : ℝ :=
let earnings_from_burgers := burgers_sold * price_per_burger,
    earnings_from_fries := total_earnings - earnings_from_burgers,
    price_per_fry := earnings_from_fries / fries_sold
in price_per_fry

theorem price_of_each_large_fry_is_correct
  (burgers_sold : ℕ := 30)
  (price_per_burger : ℝ := 2)
  (fries_sold : ℕ := 12)
  (total_earnings : ℝ := 78)
  (correct_price : ℝ := 1.5) :
  spongebob_price_of_large_fry burgers_sold price_per_burger fries_sold total_earnings = correct_price :=
by
  sorry

end price_of_each_large_fry_is_correct_l107_107978


namespace exists_tetrahedron_l107_107926

theorem exists_tetrahedron (E : Finset.Points) (h_not_plane : ¬ E ⊆ Plane)
  (h_no_collinear : ∀ {p1 p2 p3: Point}, p1 ∈ E → p2 ∈ E → p3 ∈ E → ¬ Collinear p1 p2 p3):
  ∃ (A B C D : Point), A ∈ E ∧ B ∈ E ∧ C ∈ E ∧ D ∈ E ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (∀ x, x ∈ Tetrahedron A B C D → x ∈ {A, B, C, D}) ∧
  (let A' := project_onto_plane A (Plane.mk B C D) in
    inside_triangle A' (triangle_with_midpoints (B, C, D) (E.midpoint B C) (E.midpoint C D) (E.midpoint D B))) :=
sorry

end exists_tetrahedron_l107_107926


namespace final_computation_l107_107292

noncomputable def N := (15 ^ 10 / 15 ^ 9) ^ 3 * 5 ^ 3

theorem final_computation : (N / 3 ^ 3) = 15625 := 
by 
  sorry

end final_computation_l107_107292


namespace john_annual_haircut_expense_l107_107489

noncomputable def monthly_hair_growth : ℝ := 1.5
noncomputable def initial_hair_length : ℝ := 6
noncomputable def cutoff_hair_length : ℝ := 9
noncomputable def haircut_cost : ℝ := 45
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def months_in_year : ℝ := 12

theorem john_annual_haircut_expense :
  let tip : ℝ := tip_percentage * haircut_cost,
      total_cost_per_haircut : ℝ := haircut_cost + tip,
      haircut_interval_months : ℝ := (cutoff_hair_length - initial_hair_length) / monthly_hair_growth,
      number_of_haircuts_per_year : ℝ := months_in_year / haircut_interval_months,
      annual_expense : ℝ := total_cost_per_haircut * number_of_haircuts_per_year
  in annual_expense = 324 :=
by
  sorry

end john_annual_haircut_expense_l107_107489


namespace prove_AB1_perpendicular_A1M_l107_107013

-- Define the angl_asset
def right_triangle_prism
  (A B C A1 B1 C1 M : Type)
  (angle_ACB : ℝ)
  (angle_BAC : ℝ)
  (BC : ℝ) 
  (AA1 : ℝ) 
  (is_midpoint_M : Prop) : Prop :=
  angle_ACB = 90 ∧
  angle_BAC = 30 ∧
  BC = 1 ∧
  AA1 = sqrt 6 ∧
  is_midpoint_M

-- Problem statement:
theorem prove_AB1_perpendicular_A1M
  (A B C A1 B1 C1 M : Type)
  (h : right_triangle_prism A B C A1 B1 C1 M 90 30 1 (sqrt 6) (true))
  : ∀ (AB1_perp_A1M : Prop), AB1_perp_A1M :=
  sorry

end prove_AB1_perpendicular_A1M_l107_107013


namespace quadratic_single_intersection_l107_107004

theorem quadratic_single_intersection (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + m = 0 → x^2 - 2 * x + m = (x-1)^2) :=
sorry

end quadratic_single_intersection_l107_107004


namespace v_at_one_l107_107044

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := let x := (y + 9) / 4 in x^2 + 4 * x - 5

theorem v_at_one : v 1 = 11.25 :=
by
  -- placeholder for the proof
  sorry

end v_at_one_l107_107044


namespace max_min_diff_c_l107_107501

variable (a b c : ℝ)

theorem max_min_diff_c (h1 : a + b + c = 6) (h2 : a^2 + b^2 + c^2 = 18) : 
  (4 - 0) = 4 :=
by
  sorry

end max_min_diff_c_l107_107501


namespace smallest_prime_less_than_perfect_square_is_13_l107_107198

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107198


namespace count_valid_c_values_l107_107706

theorem count_valid_c_values : 
  let floor (x : ℝ) := Int.floor x
  let ceil (x : ℝ) := Int.ceil x
  let valid_c (c : ℕ) := ∃ x : ℝ, 5 * (floor x) + 3 * (ceil x) = c
  finset.card (finset.filter valid_c (finset.Icc 0 500).val.to_finset) = 126 := 
sorry

end count_valid_c_values_l107_107706


namespace scout_troop_profit_calc_l107_107663

theorem scout_troop_profit_calc
  (candy_bars : ℕ := 1200)
  (purchase_rate : ℚ := 3/6)
  (sell_rate : ℚ := 2/3) :
  (candy_bars * sell_rate - candy_bars * purchase_rate) = 200 :=
by
  sorry

end scout_troop_profit_calc_l107_107663


namespace solved_problem_l107_107338

def double_factorial_odd (n : ℕ) : ℕ :=
  if n % 2 = 1 then (List.range' 1 (n+1) 2).foldr (· * ·) 1 else 0

def double_factorial_even (n : ℕ) : ℕ :=
  if n % 2 = 0 then (List.range' 2 (n+1) 2).foldr (· * ·) 1 else 0

def sum_S : ℚ :=
  (List.sum (List.map (λ i : ℕ, (Nat.choose (2 * i) i : ℚ) / (2 : ℚ)^(2 * i)) (List.range' 1 2010))).natAbs

theorem solved_problem (n : ℕ) (h : n = 2010) (S : ℚ) :
  let a := 4013 in let b := 1 in 
  sum_S = (List.sum (List.map (λ i : ℕ, (Nat.choose (2 * i) i : ℚ) / (2 : ℚ)^(2 * i)) (List.range' 1 2010)))
  ∧ (ab : ℚ) = (a*b) / 10
  ∧ ab = 401.3

∑ i in (Range 2010) , ((2 * i) / ( (∏ j in Range ((2 * i)-1)) / (2*i)))

end solved_problem_l107_107338


namespace greatest_possible_a_l107_107573

theorem greatest_possible_a (a : ℤ) (x : ℤ) (h_pos : 0 < a) (h_eq : x^3 + a * x^2 = -30) : 
  a ≤ 29 :=
sorry

end greatest_possible_a_l107_107573


namespace find_unknown_l107_107643

theorem find_unknown (x : ℝ) :
  300 * 2 + (x + 4) * (1 / 8) = 602 → x = 12 :=
by 
  sorry

end find_unknown_l107_107643


namespace smallest_prime_12_less_than_square_l107_107184

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107184


namespace binomial_600_600_l107_107306

theorem binomial_600_600 : nat.choose 600 600 = 1 :=
by
  -- Given the condition that binomial coefficient of n choose n is 1 for any non-negative n
  have h : ∀ n : ℕ, nat.choose n n = 1 := sorry
  -- Applying directly to the specific case n = 600
  exact h 600

end binomial_600_600_l107_107306


namespace find_AD_l107_107897

-- Given conditions
variables {A B C M N K D : Type} [equilateral_triangle ABC]
variables (a : ℝ) [side_length ABC a]
variables [midpoint M AB] [midpoint N BC] [midline MN AC]
variables [midpoint K MN]
variables [line_through AK (B C)]

-- The theorem to prove
theorem find_AD (A B C M N K D : Point) 
  (h1 : equilateral_triangle A B C)
  (h2 : side_length A B C a)
  (h3 : midpoint M A B)
  (h4 : midpoint N B C)
  (h5 : midline M N A C)
  (h6 : midpoint K M N)
  (h7 : line_through A K B C)
  (intersect : intersects (line_through A K) B C D) :
  distance A D = (a * real.sqrt 7) / 3 :=
by sorry

end find_AD_l107_107897


namespace interest_rate_per_annum_l107_107106

theorem interest_rate_per_annum (P T : ℝ) (r : ℝ) 
  (h1 : P = 15000) 
  (h2 : T = 2)
  (h3 : P * (1 + r)^T - P - (P * r * T) = 150) : 
  r = 0.1 :=
by
  sorry

end interest_rate_per_annum_l107_107106


namespace smallest_prime_12_less_than_square_l107_107186

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107186


namespace original_price_of_cricket_bat_l107_107664

theorem original_price_of_cricket_bat (P : ℝ) (final_price_in_euros : ℝ) (exchange_rate : ℝ) 
  (sales_tax_B : ℝ) (profit_B : ℝ) (discount_C : ℝ) (shipping_cost_C : ℝ) 
  (profit_D : ℝ) (vat_D : ℝ) (final_price_E : ℝ)
  (h1 : final_price_in_euros = 355)
  (h2 : exchange_rate = 0.85)
  (h3 : final_price_E = final_price_in_euros / exchange_rate)
  (h4 : P * 1.2 * (1 + sales_tax_B) * (1 + profit_B) * (1 - discount_C) + shipping_cost_C = P * 1.575 * 0.85 + shipping_cost_C)
  (h5 : (P * 1.575 * 0.85 + shipping_cost_C) * (1 + profit_D) * (1 + vat_D) = final_price_E): 
  P ≈ 210.68 :=
by
  -- Convert business problem into algebra expression sequence of transformations step by step
  let B_price := P * 1.2
  let B_price_with_tax := B_price * 1.05
  let C_price := B_price_with_tax * 1.25
  let C_price_after_discount := C_price * 0.85
  let D_price := C_price_after_discount + 10
  let D_price_with_profit := D_price * 1.3
  let E_final_price_usd := D_price_with_profit * 1.1
  have : E_final_price_usd = final_price_E := h5
  have : final_price_E = final_price_in_euros / exchange_rate := h3
  have : final_price_in_euros = 355 := h1
  have : exchange_rate = 0.85 := h2
  -- Working forward, show P ≈ 210.68 from E_final_price_usd
  sorry

end original_price_of_cricket_bat_l107_107664


namespace surface_area_of_sphere_l107_107001

theorem surface_area_of_sphere (a : Real) (h : a = 2 * Real.sqrt 3) : 
  (4 * Real.pi * ((Real.sqrt 3 * a / 2) ^ 2)) = 36 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l107_107001


namespace minimize_transfers_l107_107996

-- Define the initial number of pieces in each supermarket
def pieces_in_A := 15
def pieces_in_B := 7
def pieces_in_C := 11
def pieces_in_D := 3
def pieces_in_E := 14

-- Define the target number of pieces in each supermarket after transfers
def target_pieces := 10

-- Define a function to compute the total number of pieces
def total_pieces := pieces_in_A + pieces_in_B + pieces_in_C + pieces_in_D + pieces_in_E

-- Define the minimum number of transfers needed
def min_transfers := 12

-- The main theorem: proving that the minimum number of transfers is 12
theorem minimize_transfers : 
  total_pieces = 5 * target_pieces → 
  ∃ (transfers : ℕ), transfers = min_transfers :=
by
  -- This represents the proof section, we leave it as sorry
  sorry

end minimize_transfers_l107_107996


namespace sum_f_k_equals_768_l107_107360

-- Definitions from the given conditions
def floor (x : ℝ) : ℤ := Int.floor x
def fractional_part (x : ℝ) : ℝ := x - floor x
def f (k : ℕ) : ℤ := floor (1 / (fractional_part (Real.sqrt k)))

-- Hypothesis based on equivalent proof problem
theorem sum_f_k_equals_768 :
  (∑ k in Finset.range 241, f k) = 768 := by
  sorry

end sum_f_k_equals_768_l107_107360


namespace smallest_k_eq_gcd_ratio_l107_107047

open Int Nat

noncomputable def f (a b : ℕ) (M : ℤ) (n : ℤ) : ℤ :=
  if n < M then n + a else n - b

noncomputable def f_iter (a b : ℕ) (M : ℤ) (n : ℤ) (i : ℕ) : ℤ :=
  if i = 0 then n else f a b M (f_iter a b M n (i - 1))

theorem smallest_k_eq_gcd_ratio
  (a b : ℕ)
  (h₁ : 1 ≤ a)
  (h₂ : a ≤ b)
  (M : ℤ := (a + b) / 2)
  (k : ℕ)
  (hk : k ≥ 1 ∧ f_iter a b M 0 k = 0) :
  k = (a + b) / gcd a b :=
sorry

end smallest_k_eq_gcd_ratio_l107_107047


namespace inequality_nonneg_reals_sum_l107_107382

theorem inequality_nonneg_reals_sum (n : ℕ) (x : Fin n → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i) (h_sum : (∑ i, x i) = n) :
  ∑ i, x i / (1 + (x i)^2) ≤ ∑ i, 1 / (1 + x i) :=
by
  sorry

end inequality_nonneg_reals_sum_l107_107382


namespace ellipse_chord_length_l107_107398

theorem ellipse_chord_length (a b c : ℝ) (h1 : eccentricity_sqrt3_div2 : a > b > 0) 
  (h2 : 2 * b = 4) (eccentricity : c / a = sqrt 3 / 2) 
  (b_def : b = 2) (a_def : a = 4) :
  let ellipse_eq := (x y : ℝ) -> x^2 / 16 + y^2 / 4 = 1,
  chord_length_AB : ℝ :=
  let A := (-2, 0),
      B := (6 / 5, 16 / 5) in
  dist A B = 16 * sqrt 2 / 5 := 
sorry

end ellipse_chord_length_l107_107398


namespace fraction_of_pure_fuji_trees_l107_107654

variable (T F C : ℕ)

-- Given conditions
def condition1 := C = 0.10 * T
def condition2 := F + C = 170
def condition3 := T = F + 30 + C

theorem fraction_of_pure_fuji_trees 
  (h1 : condition1 T F C)
  (h2 : condition2 T F C)
  (h3 : condition3 T F C) : F / T = 3 / 4 :=
sorry

end fraction_of_pure_fuji_trees_l107_107654


namespace tangent_line_slope_angle_at_zero_l107_107585

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem tangent_line_slope_angle_at_zero :
  let slope := Real.exp 0 * (Real.cos 0 - Real.sin 0) in
  slope = 1 → atan slope = Real.pi / 4 :=
by
  intro slope h
  rw [Real.exp_zero, Real.cos_zero, Real.sin_zero] at h
  rw [h]
  norm_num
  exact Real.atan_one

end tangent_line_slope_angle_at_zero_l107_107585


namespace general_formula_for_sequence_l107_107427

theorem general_formula_for_sequence (a : ℕ → ℕ) (n : ℕ) :
  (a 1 = 1) ∧ (∀ k ≥ 2, a k = a 1 + ∑ i in finset.range (k-1), (a (i+2) - a (i+1))) → 
  a n = (1/2 : ℚ) * (n * (n + 1)) :=
by
  intros h
  sorry

end general_formula_for_sequence_l107_107427


namespace factorial_product_square_root_square_l107_107688

theorem factorial_product_square_root_square :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 3))^2 = 17280 := 
by
  sorry

end factorial_product_square_root_square_l107_107688


namespace exists_parallel_line_l107_107280

variable (P : ℝ × ℝ)
variable (g : ℝ × ℝ)
variable (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
variable (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0))

theorem exists_parallel_line (P : ℝ × ℝ) (g : ℝ × ℝ) (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
  (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0)) :
  ∃ a : ℝ × ℝ, (∃ d : ℝ, g = (d, 0)) ∧ (a = P) :=
sorry

end exists_parallel_line_l107_107280


namespace simplify_expression_l107_107971

theorem simplify_expression : (27 * 10^9) / (9 * 10^2) = 3000000 := 
by sorry

end simplify_expression_l107_107971


namespace problem_statement_l107_107005

theorem problem_statement (x θ : ℝ) (h : Real.logb 2 x + Real.cos θ = 2) : |x - 8| + |x + 2| = 10 :=
sorry

end problem_statement_l107_107005


namespace total_spent_l107_107521

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end total_spent_l107_107521


namespace distinct_triangles_count_l107_107436

/-- Define the set of points in a 3x3 grid -/
def grid_points : Finset (ℕ × ℕ) :=
  Finset.ofList [(0, 0), (1, 0), (2, 0),
                 (0, 1), (1, 1), (2, 1),
                 (0, 2), (1, 2), (2, 2)]

/-- Check if three points are collinear -/
def collinear {A B C : ℕ × ℕ} : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  (x2 - x1) * (y3 - y1) = (y2 - y1) * (x3 - x1)

/-- Count the number of distinct triangles from a grid of points -/
noncomputable def count_distinct_triangles : ℕ :=
  let points := grid_points in
  let triplets := points.subsetsOfSize 3 in
  (triplets.filter (λ s, ∃ A B C, s = {A, B, C} ∧ ¬collinear)).card

theorem distinct_triangles_count : count_distinct_triangles = 76 :=
  sorry

end distinct_triangles_count_l107_107436


namespace find_explicit_formula_l107_107388

noncomputable def f (a b c x : ℝ) := a * x^4 + b * x^2 + c

theorem find_explicit_formula :
  (∃ a b c : ℝ, 
    (f a b c 0 = 1) ∧ 
    (∃ y, y = x - 2 ∧ has_deriv_at (f a b c) (f' a b c 1) 1) ∧ 
    (f 1 = -1)) → 
  ∃ a b : ℝ, f a b 1 = (5/2) * x^4 - (9/2) * x^2 + 1 :=
sorry

end find_explicit_formula_l107_107388


namespace smallest_prime_less_than_perfect_square_is_13_l107_107196

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107196


namespace jazmin_dolls_correct_l107_107725

-- Define the number of dolls Geraldine has.
def geraldine_dolls : ℕ := 2186

-- Define the number of extra dolls Geraldine has compared to Jazmin.
def extra_dolls : ℕ := 977

-- Define the calculation of the number of dolls Jazmin has.
def jazmin_dolls : ℕ := geraldine_dolls - extra_dolls

-- Prove that the number of dolls Jazmin has is 1209.
theorem jazmin_dolls_correct : jazmin_dolls = 1209 := by
  -- Include the required steps in the future proof here.
  sorry

end jazmin_dolls_correct_l107_107725


namespace log_calculation_proof_l107_107293

theorem log_calculation_proof : (log10 (1 / 4) - log10 25) * 10^(1 / 2) = -10 := by
  sorry

end log_calculation_proof_l107_107293


namespace average_salary_of_all_workers_l107_107889

noncomputable def averageSalaryPerHead (total_salary : ℝ) (total_workers : ℕ) : ℝ :=
  total_salary / total_workers

theorem average_salary_of_all_workers :
  let T := 7 in
  let N := 22 in
  let R := N - T in
  let avg_salary_technicians := 1000 in
  let avg_salary_rest := 780 in
  let total_salary_technicians := avg_salary_technicians * T in
  let total_salary_rest := avg_salary_rest * R in
  let total_salary_all := total_salary_technicians + total_salary_rest in
  averageSalaryPerHead total_salary_all N = 850 :=
by {
  sorry
}

end average_salary_of_all_workers_l107_107889


namespace number_of_subsets_M_l107_107736

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the set M based on the given conditions
def M : Set ℂ := { x | ∃ n : ℕ, x = i^n + i^(-n : ℂ) }

-- Determine the number of subsets of M
theorem number_of_subsets_M : Finset.card (Finset.powerset {0, 2, -2}) = 8 := 
by {
  -- Set a proof placeholder
  sorry
}

end number_of_subsets_M_l107_107736


namespace median_equality_range_inequality_l107_107752

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l107_107752


namespace median_eq_range_le_l107_107742

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l107_107742


namespace smallest_prime_12_less_than_perfect_square_l107_107216

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107216


namespace area_of_quadrilateral_l107_107351

variable (BD AE CF : ℝ)
variable (angle_BD_AE : ℝ)
variable (isosceles_triangle_BC_BA_CA : Prop)

noncomputable def area_quadrilateral_ABCD : Prop :=
  BD = 40 ∧ AE = 11 ∧ CF = 9 ∧ angle_BD_AE = 45 ∧ isosceles_triangle_BC_BA_CA → area = 400

theorem area_of_quadrilateral : area_quadrilateral_ABCD BD AE CF angle_BD_AE isosceles_triangle_BC_BA_CA :=
by sorry

end area_of_quadrilateral_l107_107351


namespace gcd_divisors_set_l107_107700

theorem gcd_divisors_set :
  let M : Set ℕ := {d | ∃ n m : ℕ, d = Nat.gcd (2 * n + 3 * m + 13) (Nat.gcd (3 * n + 5 * m + 1) (6 * n + 6 * m - 1))} in
  M = {d | d ∣ 151} :=
by
  let M : Set ℕ := {d | ∃ n m : ℕ, d = Nat.gcd (2 * n + 3 * m + 13) (Nat.gcd (3 * n + 5 * m + 1) (6 * n + 6 * m - 1))};
  have h1 : ∀ d, d ∈ M → d ∣ 151 := by sorry;
  have h2 : ∀ d, d ∣ 151 → d ∈ M := by sorry;
  exact Set.ext (λ d, ⟨h1 d, h2 d⟩)

end gcd_divisors_set_l107_107700


namespace smallest_prime_12_less_than_perfect_square_l107_107217

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107217


namespace ratio_OC_OA_half_l107_107891

open EuclideanGeometry

variables 
  {A B C D M O : Point}
  (ABCD_rhombus : Rhombus A B C D)
  (AB_length : dist A B = 6)
  (angle_ABC_120 : ∠ ABC = 120)
  (M_on_CD : ∃ r : ℝ, 0 < r ∧ r < 1 ∧ dist C M = r * dist C D ∧ r = 1/3)
  (O_intersect : ∃ O : Point, O ∈ lineThrough A C ∧ O ∈ lineThrough B M)

noncomputable def OC_over_OA : ℝ :=
  dist O C / dist O A

theorem ratio_OC_OA_half : OC_over_OA ABCD_rhombus AB_length angle_ABC_120 M_on_CD O_intersect = 1 / 2 :=
by {
  sorry
}

end ratio_OC_OA_half_l107_107891


namespace smallest_prime_less_than_perfect_square_is_13_l107_107193

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107193


namespace safe_storage_methods_l107_107571

/-- 
The eight edges of a pyramid represent eight different chemical products.
It is dangerous to store two products that share a common vertex in the same warehouse.
It is safe to store two products that do not share a vertex in the same warehouse.
There are four warehouses labeled ①, ②, ③, and ④.
Prove the number of different safe storage methods for the eight chemical products is 48.
-/ 
theorem safe_storage_methods : 
  ∃ (assignments : fin 8 → fin 4), 
  (∀ (a b : fin 8), (a ≠ b) ∧ (shares_vertex a b → assignments a ≠ assignments b)) ∧ 
  (∀ (a b : fin 8), (¬ shares_vertex a b → assignments a = assignments b)) ∧ 
  (card assignments = 48) := 
sorry

end safe_storage_methods_l107_107571


namespace car_miles_per_gallon_l107_107690

-- Define the conditions
def distance_home : ℕ := 220
def additional_distance : ℕ := 100
def total_distance : ℕ := distance_home + additional_distance
def tank_capacity : ℕ := 16 -- in gallons
def miles_per_gallon : ℕ := total_distance / tank_capacity

-- State the goal
theorem car_miles_per_gallon : miles_per_gallon = 20 := by
  sorry

end car_miles_per_gallon_l107_107690


namespace maximum_M_l107_107352

-- Define the sides of a triangle condition
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Theorem statement
theorem maximum_M (a b c : ℝ) (h : is_triangle a b c) : 
  (a^2 + b^2) / (c^2) > (1/2) :=
sorry

end maximum_M_l107_107352


namespace find_number_l107_107362

theorem find_number (x : ℤ) (h : 72516 * x = 724797420) : x = 10001 :=
by
  sorry

end find_number_l107_107362


namespace trajectory_of_point_l107_107458

def distance_to_point (P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

def distance_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / real.sqrt (a^2 + b^2)

theorem trajectory_of_point
  (P : ℝ × ℝ)
  (F : ℝ × ℝ)
  (a b c : ℝ)
  (hf : F = (1, 1))
  (hl : a = 3) (hl1 : b = 1) (hl2 : c = -4)
  (h : distance_to_point P F = distance_to_line P a b c) :
  3 * P.1 + P.2 - 6 = 0 :=
sorry

end trajectory_of_point_l107_107458


namespace sequence_characterization_l107_107378

noncomputable def S (n : ℕ) : ℕ := 2^(n+1) - 1

def a (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2^n

theorem sequence_characterization (n : ℕ) :
  (log 2 (S n + 1) = n + 1) ∧ 
  (a 1 = 3) ∧ 
  (∀ n ≥ 2, a n = 2^n) :=
by 
  split
  sorry

end sequence_characterization_l107_107378


namespace angle_EMC_ninety_degrees_l107_107029

open EuclideanGeometry

noncomputable def incircle_tangent_points (A B C : Point) :=
  let ic := incircle ⟨A, B, C⟩
  let ⟨C1⟩ := is_incircle_tangent_to_AB A B ic
  let ⟨B1⟩ := is_incircle_tangent_to_AC A C ic
  let ⟨A1⟩ := is_incircle_tangent_to_BC B C ic
  (C1, B1, A1)

noncomputable def point_E (A A1 : Point) (ic : Circle) : Point :=
  intersection_point (line A A1) ic

noncomputable def midpoint_N (B1 A1 : Point) : Point :=
  midpoint B1 A1

noncomputable def point_M (N A A1 : Point) : Point :=
  reflection_of_point_across_line (line A A1) N

theorem angle_EMC_ninety_degrees (A B C : Point) :
  let (C1, B1, A1) := incircle_tangent_points A B C
  let ic := incircle ⟨A, B, C⟩
  let E := point_E A A1 ic
  let N := midpoint_N B1 A1
  let M := point_M N A A1
  ∠ (line E M) (line E C) = 90 :=
by sorry

end angle_EMC_ninety_degrees_l107_107029


namespace c_over_e_l107_107998

theorem c_over_e (a b c d e : ℝ) (h1 : 1 * 2 * 3 * a + 1 * 2 * 4 * a + 1 * 3 * 4 * a + 2 * 3 * 4 * a = -d)
  (h2 : 1 * 2 * 3 * 4 = e / a)
  (h3 : 1 * 2 * a + 1 * 3 * a + 1 * 4 * a + 2 * 3 * a + 2 * 4 * a + 3 * 4 * a = c) :
  c / e = 35 / 24 :=
by
  sorry

end c_over_e_l107_107998


namespace a_2017_eq_1_l107_107904

noncomputable def a : ℕ → ℤ
| 0     := 0
| 1     := 1
| 2     := 2
| (n+3) := a (n+2) - a (n+1)

theorem a_2017_eq_1 : a 2017 = 1 :=
sorry

end a_2017_eq_1_l107_107904


namespace shoe_store_ratio_l107_107666

theorem shoe_store_ratio
  (marked_price : ℝ)
  (discount : ℝ) (discount_eq : discount = 1/4)
  (cost_factor : ℝ) (cost_factor_eq : cost_factor = 2/3) :
  (cost_factor * (1 - discount) * marked_price / marked_price) = 1/2 := 
by
  -- Insert proof here
  sorry

end shoe_store_ratio_l107_107666


namespace arrangement_count1_arrangement_count2_arrangement_count3_arrangement_count4_l107_107242

-- Define the entities in the problem
inductive Participant
| Teacher
| Boy (id : Nat)
| Girl (id : Nat)

-- Define the conditions as properties or predicates
def girlsNextToEachOther (arrangement : List Participant) : Prop :=
  -- assuming the arrangement is a list of Participant
  sorry -- insert the actual condition as needed

def boysNotNextToEachOther (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def boysInDecreasingOrder (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def teacherNotInMiddle (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def girlsNotAtEnds (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

-- Problem 1: Two girls must stand next to each other
theorem arrangement_count1 : ∃ arrangements, 1440 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, girlsNextToEachOther a := sorry

-- Problem 2: Boys must not stand next to each other
theorem arrangement_count2 : ∃ arrangements, 144 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, boysNotNextToEachOther a := sorry

-- Problem 3: Boys must stand in decreasing order of height
theorem arrangement_count3 : ∃ arrangements, 210 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, boysInDecreasingOrder a := sorry

-- Problem 4: Teacher not in middle, girls not at the ends
theorem arrangement_count4 : ∃ arrangements, 2112 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, teacherNotInMiddle a ∧ girlsNotAtEnds a := sorry

end arrangement_count1_arrangement_count2_arrangement_count3_arrangement_count4_l107_107242


namespace smallest_prime_12_less_than_square_l107_107188

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107188


namespace domain_of_f_l107_107704

open Set

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_f_l107_107704


namespace cole_speed_return_l107_107691

noncomputable def cole_return_speed (v1 : ℕ) (t_total : ℕ) (t_work_min : ℕ) : ℕ :=
  let t_work := t_work_min / 60
      d_work := v1 * t_work
      round_trip_distance := 2 * d_work
      t_return := t_total - t_work
      v_return := d_work / t_return
  in v_return

theorem cole_speed_return (h1 : cole_return_speed 50 2 82.5 = 110) : cole_return_speed 50 2 82.5 = 110 :=
by
    exact h1

end cole_speed_return_l107_107691


namespace initial_number_of_persons_l107_107104

theorem initial_number_of_persons (n : ℕ) 
  (avg_increase : n * 4.5 = 36) 
  (new_weight : 101 = 65 + 36) :
  n = 8 :=
by
  sorry

end initial_number_of_persons_l107_107104


namespace area_of_triangle_PF1F2_l107_107927

noncomputable def triangle_area (a b c : ℝ) : ℝ := Real.sqrt ((a + (b + c)) * ((-a) + (b + c)) * ((a - b) + c) * ((a + b) - c)) / 4

theorem area_of_triangle_PF1F2 :
  ∃ P F1 F2 : ℝ × ℝ, 
    let ellipse := (P ∈ {p : ℝ × ℝ | (p.1^2) / 9 + (p.2^2) / 4 = 1}),
        focus_dist := 5,
        |PF1| : ℝ = 4,
        |PF2| : ℝ = 2,
        F1F2 := 2 * Real.sqrt focus_dist, 
        area := triangle_area |PF1| |PF2| F1F2 
    in ellipse ∧ |PF1| / |PF2| = 2 / 1 ∧ focus_dist = 5 ∧ area = 4 :=
begin
  sorry
end

end area_of_triangle_PF1F2_l107_107927


namespace population_multiple_of_seven_l107_107117

theorem population_multiple_of_seven 
  (a b c : ℕ) 
  (h1 : a^2 + 100 = b^2 + 1) 
  (h2 : b^2 + 1 + 100 = c^2) : 
  (∃ k : ℕ, a = 7 * k) :=
sorry

end population_multiple_of_seven_l107_107117


namespace median_eq_range_le_l107_107740

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l107_107740


namespace median_equality_range_inequality_l107_107798

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l107_107798


namespace at_least_two_cities_with_single_connection_l107_107010

-- Define the context and the proof problem
theorem at_least_two_cities_with_single_connection
  (City : Type)
  (Road : City → City → Prop)
  (H_nonempty : ∃ c1 c2 : City, c1 ≠ c2) -- more than one city
  (H_connected : ∀ c1 c2 : City, 
    ∃ p : list City, 
      p.head = some c1 ∧ 
      p.reverse.head = some c2 ∧ 
      ∀ i, i < p.length - 1 → Road (p.nth_le i (by sorry)) (p.nth_le (i + 1) (by sorry))))
  (H_no_cycle : ∀ (c1 c2 : City), ¬ (Road c1 c2 ∧ Road c2 c1)): -- roads do not form cycles
  ∃ (c1 c2 : City), (Road c1 c2) ∧ 
    (∀ (c : City), Road c c1 → c = c2) ∧ 
    (∀ (c : City), Road c c2 → c = c1) :=
by
  sorry

end at_least_two_cities_with_single_connection_l107_107010


namespace max_lessons_l107_107589

-- Declaring noncomputable variables for the number of shirts, pairs of pants, and pairs of shoes.
noncomputable def s : ℕ := sorry
noncomputable def p : ℕ := sorry
noncomputable def b : ℕ := sorry

lemma conditions_satisfied :
  2 * (s + 1) * p * b = 2 * s * p * b + 36 ∧
  2 * s * (p + 1) * b = 2 * s * p * b + 72 ∧
  2 * s * p * (b + 1) = 2 * s * p * b + 54 ∧
  s * p * b = 27 ∧
  s * b = 36 ∧
  p * b = 18 := by
  sorry

theorem max_lessons : (2 * s * p * b) = 216 :=
by
  have h := conditions_satisfied
  sorry

end max_lessons_l107_107589


namespace sum_of_possible_n_l107_107584

theorem sum_of_possible_n :
  let S := {4, 8, 12, 14}
  ∃ n : ℝ, n ∉ S ∧ 
  (median {4, 8, 12, 14, n} = (4 + 8 + 12 + 14 + n) / 5) →
  n = 2 ∨ n = 9.5 ∨ n = 22 →
  2 + 9.5 + 22 = 33.5 :=
by
  let S := {4, 8, 12, 14}
  sorry

end sum_of_possible_n_l107_107584


namespace find_a_l107_107420

theorem find_a :
  ∃ a : ℝ, (2 * x - (a * Real.exp x + x) + 1 = 0) = (a = 1) :=
by
  sorry

end find_a_l107_107420


namespace max_groups_l107_107123

-- Define the conditions
def valid_eq (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ (3 * a + b = 13)

-- The proof problem: No need for the proof body, just statement
theorem max_groups : ∃! (l : List (ℕ × ℕ)), (∀ ab ∈ l, valid_eq ab.fst ab.snd) ∧ l.length = 3 := sorry

end max_groups_l107_107123


namespace oncoming_train_speed_oncoming_train_speed_kmph_l107_107258

-- Definitions of the given problem conditions
def passenger_train_speed_kmph : ℝ := 40
def oncoming_train_length_m : ℝ := 75
def time_to_pass_seconds : ℝ := 3

-- Conversion factors
def seconds_to_hours : ℝ := time_to_pass_seconds / 3600
def meters_to_km : ℝ := oncoming_train_length_m / 1000

-- Theorem to prove the speed of the oncoming train
theorem oncoming_train_speed :
  ( (40 + x) * (3 / 3600) = 75 * (10 ^ (-3)) ) → x = 50 :=
by
  sorry

-- Including actual definitions in theorem
theorem oncoming_train_speed_kmph (x : ℝ) :
  ( (passenger_train_speed_kmph + x) * seconds_to_hours = meters_to_km ) → x = 50 :=
by
  sorry

end oncoming_train_speed_oncoming_train_speed_kmph_l107_107258


namespace find_a_l107_107476

theorem find_a (a : ℝ) : (3 : ℝ) = 3 → (4 : ℝ) = 4 → sqrt ((3 - 3)^2 + (a - 4)^2) = 3 → a > 4 → a = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end find_a_l107_107476


namespace floor_S_squared_l107_107931

noncomputable def S : ℝ :=
  ∑ i in Finset.range 3000, real.sqrt (3 + 1 / (i + 1)^2 + 1 / (i + 2)^2)

theorem floor_S_squared : (⌊S^2⌋ : ℕ) = 9012004 := by
  sorry

end floor_S_squared_l107_107931


namespace calculate_expression_l107_107687

def convert_base_to_decimal (coeffs : List ℕ) (base : ℕ) : ℕ :=
  coeffs.reverse.map_with_index (fun i a => a * base^i).sum

def calculate_fraction (num_coeffs : List ℕ) (den_coeffs : List ℕ) (base_num : ℕ) (base_den : ℕ) : ℕ :=
  let num := convert_base_to_decimal num_coeffs base_num
  let den := convert_base_to_decimal den_coeffs base_den
  num / den

theorem calculate_expression : 
  let frac1 := calculate_fraction [2, 5, 4] [1, 4] 8 4
  let frac2 := calculate_fraction [1, 3, 2] [2, 6] 5 3
  frac1 + frac2 = 25 :=
by
  let frac1 := calculate_fraction [2, 5, 4] [1, 4] 8 4
  let frac2 := calculate_fraction [1, 3, 2] [2, 6] 5 3
  show frac1 + frac2 = 25
  sorry

end calculate_expression_l107_107687


namespace min_moves_to_identical_contents_l107_107079

-- Define the conditions as Lean definitions
def round_box : List (String × Nat) := [("white", 4), ("black", 6)]
def square_box : List (String × Nat) := [("black", 10)]

def move := "take from any box and either discard or transfer to another box."

-- The Problem Statement in Lean 4
theorem min_moves_to_identical_contents :
  ∃ min_moves, min_moves = 15 ∧ round_box = [("white", 4), ("black", 6)] ∧ square_box = [("black", 10)] ∧ move = "take from any box and either discard or transfer to another box." :=
begin
  sorry
end

end min_moves_to_identical_contents_l107_107079


namespace min_value_frac_sum_l107_107456

theorem min_value_frac_sum (a b : ℝ) (hab : a + b = 1) (ha : 0 < a) (hb : 0 < b) : 
  ∃ (x : ℝ), x = 3 + 2 * Real.sqrt 2 ∧ x = (1/a + 2/b) :=
sorry

end min_value_frac_sum_l107_107456


namespace coeff_x6_in_expansion_l107_107714

noncomputable def coefficient_x6_expansion : Prop :=
  let p := (1 + 3 * Polynomial.X - Polynomial.X ^ 2) ^ 5
  in Polynomial.coeff p 6 = -370

-- statement without proof
theorem coeff_x6_in_expansion : coefficient_x6_expansion :=
  sorry

end coeff_x6_in_expansion_l107_107714


namespace students_taking_neither_l107_107940

theorem students_taking_neither (total biology chemistry both : ℕ)
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  (total - (biology + chemistry - both)) = 10 :=
by {
  sorry
}

end students_taking_neither_l107_107940


namespace quadratic_function_properties_l107_107419

theorem quadratic_function_properties (a : ℝ) (h : a ≠ 0) :
  ¬ (∃ y : ℝ, y = a * (-1)^2 - 2 * a * (-1) - 1 ∧ y = 1) ∧
  ¬ (a = -2 ∧ (4^2 - 4 * (-2) * (-1)) ≤ 0) ∧
  ¬ (a > 0 ∧ ∀ x ≥ 1, y (a * x^2 - 2 * a * x - 1) < y (a * (x + 1)^2 - 2 * a * (x + 1) - 1)) ∧
  (a < 0 ∧ ∀ x ≤ 1, y (a * x^2 - 2 * a * x - 1) < y (a * (x - 1)^2 - 2 * a * (x - 1) - 1)) :=
by sorry

end quadratic_function_properties_l107_107419


namespace triangle_inequality_l107_107627

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by {
  sorry
}

end triangle_inequality_l107_107627


namespace smallest_prime_12_less_perfect_square_l107_107143

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107143


namespace correct_transformation_l107_107223

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : (a / b = 2 * a / 2 * b) :=
by
  sorry

end correct_transformation_l107_107223


namespace no_points_C_l107_107901

theorem no_points_C :
  ∀ (C : ℝ × ℝ), ∀ (A B : ℝ × ℝ),
  (A = ⟨0, 0⟩) →
  (B = ⟨12, 0⟩) →
  (∃ (x y : ℝ), A ≠ B ∧ (x, y) = C ∧
    (12 + (sqrt (x^2 + y^2)) + (sqrt ((x-12)^2 + y^2)) = 60) ∧
    (0.5 * 12 * |y| = 144)) → false :=
by
  intros C A B hA hB hC
  sorry

end no_points_C_l107_107901


namespace sum_of_perimeters_triangles_l107_107677

theorem sum_of_perimeters_triangles (a : ℕ → ℕ) (side_length : ℕ) (P : ℕ → ℕ):
  (∀ n : ℕ, a 0 = side_length ∧ P 0 = 3 * a 0) →
  (∀ n : ℕ, a (n + 1) = a n / 2 ∧ P (n + 1) = 3 * a (n + 1)) →
  (side_length = 45) →
  ∑' n, P n = 270 :=
by
  -- the proof would continue here
  sorry

end sum_of_perimeters_triangles_l107_107677


namespace problem_l107_107588

variables {S T : ℕ → ℕ} {a b : ℕ → ℕ}

-- Conditions
-- S_n and T_n are sums of first n terms of arithmetic sequences {a_n} and {b_n}, respectively.
axiom sum_S : ∀ n, S n = n * (n + 1) / 2  -- Example: sum from 1 to n
axiom sum_T : ∀ n, T n = n * (n + 1) / 2  -- Example: sum from 1 to n

-- For any positive integer n, (S_n / T_n = (5n - 3) / (2n + 1))
axiom condition : ∀ n > 0, (S n : ℚ) / T n = (5 * n - 3 : ℚ) / (2 * n + 1)

-- Theorem to prove
theorem problem : (a 20 : ℚ) / (b 7) = 64 / 9 :=
sorry

end problem_l107_107588


namespace rocky_fights_l107_107088

theorem rocky_fights (F : ℕ) 
  (h1 : 0.50 * F = n) 
  (h2 : 0.20 * n = 19) : 
  F = 190 := 
by {
  sorry
}

end rocky_fights_l107_107088


namespace time_to_reach_ticket_window_l107_107915

variable (minutes_per_meter : ℝ) (remaining_distance : ℝ) (expected_time : ℝ)

def rate := 2 -- Kit's rate of movement in meters per minute

def remaining_distance := 65 -- Remaining distance to the ticket window in meters

def expected_time := remaining_distance / rate -- Time required to cover the remaining distance

-- Theorem to prove
theorem time_to_reach_ticket_window :
  expected_time = 32.5 :=
sorry

end time_to_reach_ticket_window_l107_107915


namespace eval_ginv_ginv_sum_l107_107099

noncomputable def g : ℤ → ℤ := sorry
noncomputable def g_inv : ℤ → ℤ := sorry

axiom g_4 : g 4 = 3
axiom g_1 : g 1 = 6
axiom g_3 : g 3 = 2
axiom g_ginv : ∀ x, g (g_inv x) = x
axiom ginv_g : ∀ x, g_inv (g x) = x 

theorem eval_ginv_ginv_sum :
  g_inv (g_inv 6 + g_inv 2) = 4 :=
by {
  have h1 : g_inv 6 = 1,
  { apply ginv_g, rw g_1 },
  have h2 : g_inv 2 = 3,
  { apply ginv_g, rw g_3 },
  rw [h1, h2, add_comm],
  exact ginv_g 4,
}

end eval_ginv_ginv_sum_l107_107099


namespace quadratic_inequality_solution_set_conclusions_l107_107873

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set_conclusions (h1 : ∀ x, -1 ≤ x ∧ x ≤ 2 → ax^2 + bx + c ≥ 0)
(h2 : ∀ x, x < -1 ∨ x > 2 → ax^2 + bx + c < 0) :
(a + b = 0) ∧ (a + b + c > 0) ∧ (c > 0) ∧ ¬ (b < 0) := by
sorry

end quadratic_inequality_solution_set_conclusions_l107_107873


namespace binomial_600_600_l107_107310

-- Define a theorem to state the binomial coefficient property and use it to prove the specific case.
theorem binomial_600_600 : nat.choose 600 600 = 1 :=
begin
  -- Binomial property: for any non-negative integer n, (n choose n) = 1
  rw nat.choose_self,
end

end binomial_600_600_l107_107310


namespace min_dot_product_l107_107869

noncomputable def ellipse_eq_p (x y : ℝ) : Prop :=
    x^2 / 9 + y^2 / 8 = 1

noncomputable def dot_product_op_fp (x y : ℝ) : ℝ :=
    x^2 + x + y^2

theorem min_dot_product : 
    (∀ x y : ℝ, ellipse_eq_p x y → dot_product_op_fp x y = 6) := 
sorry

end min_dot_product_l107_107869


namespace isosceles_triangle_angle_sum_l107_107381

theorem isosceles_triangle_angle_sum 
  (A B C : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C]
  (AC AB : ℝ) 
  (angle_ABC : ℝ)
  (isosceles : AC = AB)
  (angle_A : angle_ABC = 70) :
  (∃ angle_B : ℝ, angle_B = 55) :=
by
  sorry

end isosceles_triangle_angle_sum_l107_107381


namespace plane_BSD_divides_MN_l107_107568

-- We define points and midpoints within the geometric context
variables (S A B C D M N : Type) [AddGroup M] [VectorSpace ℝ M] 
  (parallelogram_ABC (AB : M) (BC : M) (CD : M) (DA : M) : Prop)
  (midpoint : M → M → M)

-- Midpoint properties
def midpoint_prop (A B M : M) (H : midpoint A B = M) : Prop :=
  A + B = 2 • M

-- Given conditions
axiom parallelogram_has_properties : parallelogram_ABC AB BC CD DA
axiom midpoint_AB : midpoint A B = M
axiom midpoint_SC : midpoint S C = N

-- Lean statement to prove the required proportionality
theorem plane_BSD_divides_MN (H_parallelogram : parallelogram_ABC AB BC CD DA)
  (H_midpoint_AB : midpoint A B = M) (H_midpoint_SC : midpoint S C = N) :
  ∃ ratio : ℕ, ratio = 1 :=
begin
  -- proof terms will be here
  sorry
end

end plane_BSD_divides_MN_l107_107568


namespace intersection_point_is_P_area_of_triangle_ABP_l107_107421

-- Define the equations of the lines
def line1 (x : ℝ) : ℝ := 2 * x - 5
def line2 (x : ℝ) : ℝ := x - 1

-- Define the points A, B, and P
def A := (2.5 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 0 : ℝ)
def P := (4 : ℝ, 3 : ℝ)

-- Prove the intersection of the lines at point P is correct
theorem intersection_point_is_P : (∃ x y, y = line1 x ∧ y = line2 x ∧ x = 4 ∧ y = 3) :=
by
  sorry

-- Prove the area of triangle ABP is 9/4
theorem area_of_triangle_ABP : let base := abs (A.1 - B.1), height := P.2 in (1 / 2) * base * height = 9 / 4 :=
by
  sorry

end intersection_point_is_P_area_of_triangle_ABP_l107_107421


namespace median_equality_and_range_inequality_l107_107771

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l107_107771


namespace terminal_side_in_quadrant_II_or_IV_l107_107447

theorem terminal_side_in_quadrant_II_or_IV
  (α : ℝ)
  (h1 : sin α * cos α < 0)
  (h2 : sin α * tan α > 0) :
  ∃ (k : ℤ), π / 2 + k * π < α / 2 ∧ α / 2 < π + k * π :=
by
  sorry

end terminal_side_in_quadrant_II_or_IV_l107_107447


namespace irreducible_polynomials_large_d_l107_107949

noncomputable def is_irreducible_over_Z (p : Polynomial ℚ) : Prop :=
  irreducible p

theorem irreducible_polynomials_large_d :
  ∀ ε > (0 : ℝ), ∃ (d : ℕ), ∀ (D : ℕ), D ≥ d →
    (P : Set (Polynomial ℤ)) (∀ p ∈ P, (degree p ≤ D ∧
    (∀ (i j : ℕ), i ≤ D ∧ j ≤ D → ∃ (a : ℤ), a = 1 ∨ a = -1)),
    ∃ S ⊆ P, (card S / card P : ℝ) > 0.99 → ∀ p ∈ S, is_irreducible_over_Z p) :=
begin
  sorry,
end

end irreducible_polynomials_large_d_l107_107949


namespace ratio_BD_BO_l107_107025

-- Define the isosceles triangle and circle properties
noncomputable def triangleABC (A B C O : Point) : Prop :=
  is_isosceles A B C ∧ ∠ B = 80 ∧
  tangent_line B A O ∧ tangent_line B C O ∧
  center_circle O [A, C]

-- Define the relationship on intersection of the circle with line BO
noncomputable def circle_intersect (O B D : Point) : Prop :=
  intersects_circle O (line_segment B O) D

-- Prove the required ratio
theorem ratio_BD_BO (A B C O D : Point) :
  triangleABC A B C O →
  circle_intersect O B D →
  (BD / BO) = 1 - (√2 / 2) :=
by
  sorry

end ratio_BD_BO_l107_107025


namespace smallest_prime_less_than_perfect_square_is_13_l107_107199

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107199


namespace total_animal_legs_l107_107279

def number_of_dogs : ℕ := 2
def number_of_chickens : ℕ := 1
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem total_animal_legs : number_of_dogs * legs_per_dog + number_of_chickens * legs_per_chicken = 10 :=
by
  -- The proof is skipped
  sorry

end total_animal_legs_l107_107279


namespace median_equality_range_inequality_l107_107748

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l107_107748


namespace area_of_triangle_BDE_l107_107078

noncomputable def point (α : Type) := (α × α × α)

def AB : ℝ := 3
def BC : ℝ := 3
def CD : ℝ := 3
def DE : ℝ := 3
def EA : ℝ := 3

def angle_ABC : ℝ := 60
def angle_CDE : ℝ := 60
def angle_DEA : ℝ := 60

def is_plane_perpendicular (plane1 : set (point ℝ)) (line2 : set (point ℝ)) : Prop := sorry -- Dummy definition for perpendicularity

def is_right_angle (angle : ℝ) : Prop := angle = 90

def points_A_B_C_D_E_are_correct : Prop :=
  let A : point ℝ := (-3, 0, -1.5)
  let B : point ℝ := (0, 3, 0)
  let C : point ℝ := (3, 0, 1.5)
  let D : point ℝ := (0, 0, 1.5)
  let E : point ℝ := (0, 0, -1.5) in
  -- All distance conditions
  (euclidean_distance A B = 3) ∧
  (euclidean_distance B C = 3) ∧
  (euclidean_distance C D = 3) ∧
  (euclidean_distance D E = 3) ∧
  (euclidean_distance E A = 3) ∧
  -- Angle conditions
  (angle_ABC = 60) ∧
  (angle_CDE = 60) ∧
  (angle_DEA = 60) ∧
  -- Plane perpendicularity
  (is_plane_perpendicular ({A, B, C}) ({D, E}))

def euclidean_distance (p1 p2 : point ℝ) : ℝ :=
  match p1, p2 with
  | (x1, y1, z1), (x2, y2, z2) => sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

def calculate_area (p1 p2 p3 : point ℝ) : ℝ := sorry -- Definition for area calculation

-- Main proof problem
theorem area_of_triangle_BDE : points_A_B_C_D_E_are_correct → calculate_area (0, 3, 0) (0, 0, 1.5) (0, 0, -1.5) = sqrt 91.125 :=
by
  intro _,
  exact sorry

end area_of_triangle_BDE_l107_107078


namespace arithmetic_sequence_general_formula_sum_of_first_n_bn_value_of_t_l107_107805

noncomputable def general_term_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = 2 * n

theorem arithmetic_sequence_general_formula 
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_non_zero : d ≠ 0)
  (S : ℕ → ℝ)
  (h_S3 : S 3 = a 4 + 4)
  (h_geo_seq : ∃ r : ℝ, a 2 = a 1 * r ∧ a 6 = a 1 * r^2 ∧ a 18 = a 1 * r^3) :
  general_term_arithmetic_sequence a d := 
sorry

noncomputable def sum_of_bn (b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = 4 - (2 * n + 4) / (2^n)

theorem sum_of_first_n_bn
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (T : ℕ → ℝ)
  (h_general : ∀ n : ℕ, a n = 2 * n)
  (h_b_def : ∀ n : ℕ, b n = a n / (2^n)) :
  sum_of_bn b T :=
sorry

noncomputable def find_t (c : ℕ → ℝ) (t : ℝ) (S : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, S n = n^2 + n) →  
  (∀ n : ℕ, c n = sqrt (S n + t)) →
  (∀ k m n : ℕ, 2 * c m = c k + c n) →
  t = 1 / 4

theorem value_of_t
  (S : ℕ → ℝ)
  (c : ℕ → ℝ)
  (t : ℝ)
  (h_S_def : ∀ n : ℕ, S n = n^2 + n)
  (h_c_def : ∀ n : ℕ, c n = sqrt (S n + t))
  (h_arithmetic_c : ∀ k m n : ℕ, 2 * c m = c k + c n) :
  find_t c t S :=
sorry


end arithmetic_sequence_general_formula_sum_of_first_n_bn_value_of_t_l107_107805


namespace volume_of_oil_is_correct_l107_107248

def volume_of_oil_in_tank (height : ℝ) (radius : ℝ) (fill_ratio : ℝ) (oil_to_water_ratio : ℝ) : ℝ :=
  let total_volume := Math.pi * radius^2 * height
  let liquid_volume := fill_ratio * total_volume
  let oil_ratio := oil_to_water_ratio / (oil_to_water_ratio + 1)
  oil_ratio * liquid_volume

theorem volume_of_oil_is_correct :
  volume_of_oil_in_tank 8 (3/2) 0.75 (3/7) = 4.05 * Math.pi :=
by
  sorry

end volume_of_oil_is_correct_l107_107248


namespace median_equal_range_not_greater_l107_107763

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l107_107763


namespace cos_sum_simplified_l107_107961

theorem cos_sum_simplified :
  (Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)) = ((Real.sqrt 13 - 1) / 4) :=
by
  sorry

end cos_sum_simplified_l107_107961


namespace angle_PTV_60_degrees_l107_107020

theorem angle_PTV_60_degrees
  (m n : Line)
  (P T V : Point)
  (h_parallel : Parallel m n)
  (h_perpendicular_m : Perpendicular (Line.mk T V) m)
  (h_angle_TPV : AngleMeasure (Aangle P T V) = 30)
  (h_triangle_sum : ∀ A B C : Point, TriangleSum (Triangle.mk A B C)) :
  AngleMeasure (Aangle P T V) = 60 := 
sorry

end angle_PTV_60_degrees_l107_107020


namespace find_x_log_eq_l107_107712

theorem find_x_log_eq :
  ∃ x : ℝ, log 64 (3 * x - 2) = -(1/3) ∧ x = 3/4 :=
by
  sorry

end find_x_log_eq_l107_107712


namespace barbell_percentage_increase_l107_107909

def old_barbell_cost : ℕ := 250
def new_barbell_cost : ℕ := 325

theorem barbell_percentage_increase :
  (new_barbell_cost - old_barbell_cost : ℚ) / old_barbell_cost * 100 = 30 := 
by
  sorry

end barbell_percentage_increase_l107_107909


namespace y_coordinate_midpoint_graph_l107_107925

theorem y_coordinate_midpoint_graph {f : ℝ → ℝ} (x1 x2 : ℝ) (y1 y2 : ℝ)
  (hx1 : y1 = (1/2 : ℝ) + real.log (x1 / (1 - x1)) / real.log 2)
  (hx2 : y2 = (1/2 : ℝ) + real.log (x2 / (1 - x2)) / real.log 2)
  (hmid : (x1 + x2) / 2 = 1 / 2) :
  (y1 + y2) / 2 = 1 / 2 :=
by
  sorry

end y_coordinate_midpoint_graph_l107_107925


namespace median_eq_range_le_l107_107741

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l107_107741


namespace second_smallest_three_probability_eq_one_third_l107_107554

open Set

noncomputable def probability_second_smallest_is_three : ℚ :=
  let A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let total_ways := (A.card.choose 6)
  let condition_ways := (Finset.choose 2 1 * Finset.choose 7 4)
  condition_ways / total_ways

theorem second_smallest_three_probability_eq_one_third :
  probability_second_smallest_is_three = 1/3 :=
sorry

end second_smallest_three_probability_eq_one_third_l107_107554


namespace math_problem_l107_107623

theorem math_problem : 1003^2 - 997^2 - 1001^2 + 999^2 = 8000 := by
  sorry

end math_problem_l107_107623


namespace degree_of_P_l107_107916

noncomputable def sequence (P : ℤ → ℤ) (n : ℤ) : ℕ → ℤ
| 0     := n
| (k+1) := P (sequence k)

theorem degree_of_P (P : ℤ → ℤ) (h1 : ∃ c : ℤ, ∃ d : ℤ, c ≠ 0 ∧ d ≠ 0 ∧ ∀ x : ℤ, P x = c * x + d) 
    (n : ℤ) (hn : n > 0) 
    (h2 : ∀ b : ℕ, ∃ k : ℕ, ∃ m : ℤ, m > 1 ∧ sequence P n k = m ^ b) : 
    ∃ (a b : ℤ), a ≠ 0 ∧ P = λ x, a * x + b :=
begin
  sorry
end

end degree_of_P_l107_107916


namespace value_range_a_for_two_positive_solutions_l107_107594

theorem value_range_a_for_two_positive_solutions (a : ℝ) :
  (∃ (x : ℝ), (|2 * x - 1| - a = 0) ∧ x > 0 ∧ (0 < a ∧ a < 1)) :=
by 
  sorry

end value_range_a_for_two_positive_solutions_l107_107594


namespace min_f_g_gt_l107_107713

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (16 - x^2 + 6 * x)
noncomputable def g (x : ℝ) : ℝ := abs (x - 3)

theorem min_f_g_gt (x : ℝ) : (Real.min (f x) (g x) > (5 - x) / 2) ↔ (x ∈ Set.Ioo (-1 : ℝ) 1 ∪ Set.Ico (11 / 3) 8) :=
begin
  sorry
end

end min_f_g_gt_l107_107713


namespace avg_of_xyz_l107_107443

-- Define the given condition
def given_condition (x y z : ℝ) := 
  (5 / 2) * (x + y + z) = 20

-- Define the question (and the proof target) using the given conditions.
theorem avg_of_xyz (x y z : ℝ) (h : given_condition x y z) : 
  (x + y + z) / 3 = 8 / 3 :=
sorry

end avg_of_xyz_l107_107443


namespace extended_hexagon_area_l107_107138

theorem extended_hexagon_area (original_area : ℝ) (side_length_extension : ℝ)
  (original_side_length : ℝ) (new_side_length : ℝ) :
  original_area = 18 ∧ side_length_extension = 1 ∧ original_side_length = 2 
  ∧ new_side_length = original_side_length + 2 * side_length_extension →
  36 = original_area + 6 * (0.5 * side_length_extension * (original_side_length + 1)) := 
sorry

end extended_hexagon_area_l107_107138


namespace surface_area_increase_of_cubes_l107_107702

-- Definitions of conditions
variable (a : ℝ)

-- Increase in surface area after cutting the cube
theorem surface_area_increase_of_cubes (a : ℝ) : 
  let original_surface_area := 6 * a^2
  let smaller_edge_length := a / 3
  let smaller_surface_area := 6 * (smaller_edge_length)^2 * 27
  in smaller_surface_area - original_surface_area = 12 * a^2 :=
by
  sorry

end surface_area_increase_of_cubes_l107_107702


namespace trigonometric_identity_l107_107825

theorem trigonometric_identity 
  (α : ℝ)  -- assuming that α is a real number representing the angle
  (h1 : ∃ x y : ℝ, x ≠ 0 ∧ (x + 2 * y = 0) ∧ α = real.atan (y / x)) :
    ( (real.sin α + real.cos α) / (real.sin α - real.cos α) = -1 / 3) :=
begin
  sorry
end

end trigonometric_identity_l107_107825


namespace solve_for_n_l107_107094

theorem solve_for_n (n : ℕ) : (16^n) * (16^n) * (16^n) * (16^n) = 256^4 → n = 2 :=
by
-intro h
-sorry

end solve_for_n_l107_107094


namespace max_value_of_min_of_f_l107_107839

noncomputable def f (x t : ℝ) : ℝ := x^2 - 2 * t * x + t

theorem max_value_of_min_of_f : 
  (∀ t : ℝ, (∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) → ∃ m : ℝ, (∀ y : ℝ, y ∈ set.Icc (-1 : ℝ) (1 : ℝ) → f y t ≥ m) ∧ 
  m = 
    if t ≤ -1 then 1 + 3 * t
    else if -1 < t ∧ t < 1 then - (t - 1 / 2) ^ 2 + 1 / 4
    else 1 - t) → 
  (∃ M : ℝ, ∀ t : ℝ, M ≥ 
    if t ≤ -1 then 1 + 3 * t
    else if -1 < t ∧ t < 1 then - (t - 1 / 2) ^ 2 + 1 / 4
    else 1 - t)) :=
sorry

end max_value_of_min_of_f_l107_107839


namespace curve_intersects_line_distance_condition_l107_107330

noncomputable def curve (θ : ℝ) : ℝ × ℝ := (2 + 3 * Real.cos θ, -1 + 3 * Real.sin θ)

def line (x y : ℝ) : Prop := x - 3 * y + 2 = 0

def distance (x y : ℝ) : ℝ := |(2 - 3 * -1 + 2) / Real.sqrt 10|

theorem curve_intersects_line_distance_condition :
  ∃ p : ℝ × ℝ, ∃ θ : ℝ, curve θ = p ∧ distance p.1 p.2 = 7 * Real.sqrt 10 / 10 ∧ line p.1 p.2 ∧ 
  (curve_intersect_line : ∀ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 9 → line x y → distance x y = 7 * √10 / 10) → 
  (number_points : ∃ n : ℕ, n = 2) :=
sorry

end curve_intersects_line_distance_condition_l107_107330


namespace range_of_m_l107_107417

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (m * x^2 + (m - 3) * x + 1 = 0)) →
  m ∈ Set.Iic 1 := by
  sorry

end range_of_m_l107_107417


namespace smallest_prime_12_less_than_perfect_square_l107_107212

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107212


namespace incorrect_statement_about_zero_l107_107674

theorem incorrect_statement_about_zero :
  ¬ (0 > 0) :=
by
  sorry

end incorrect_statement_about_zero_l107_107674


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107793

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107793


namespace find_fx2_plus_2_l107_107039

noncomputable def f : ℝ → ℝ :=
λ x, x^2 + 2 * x - 3  -- From the intermediate substitution f(y) = y^2 + 2y - 3

theorem find_fx2_plus_2 (x : ℝ) :
  f (x^2 + 2) = x^4 + 6 * x^2 + 1 :=
by sorry

end find_fx2_plus_2_l107_107039


namespace simplest_fraction_l107_107283

-- Definitions based on the conditions
def optionA (x y : ℝ) : ℝ := (16 * x) / (20 * y)
def optionB (x : ℝ) : ℝ := (1 - 2 * x) / (2 * x - 1)
def optionC (x y : ℝ) : ℝ := (x - y) / (x^2 + y^2)
def optionD (x y : ℝ) : ℝ := (x - y) / (x^2 - y^2)

-- Theorem to prove that option C is in simplest form
theorem simplest_fraction (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ 0) (h3 : y ≠ 0) : 
  (∀ a b : ℝ, optionA a b ≠ optionC x y) ∧
  (optionB x ≠ optionC x y) ∧
  (∀ c d : ℝ, optionD c d ≠ optionC x y) :=
by
  sorry

end simplest_fraction_l107_107283


namespace cos_sum_identity_l107_107964

theorem cos_sum_identity :
  cos (2 * Real.pi / 17) + cos (6 * Real.pi / 17) + cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_identity_l107_107964


namespace sum_squared_distances_l107_107902

-- Definitions for the conditions
def P : ℝ × ℝ := (1, 0)

def polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

def C_rect (x y : ℝ) : Prop := x^2 + y^2 = 2 * x + 2 * y

def l_param_eq (t : ℝ) : ℝ × ℝ := (1 - (Real.sqrt 2) / 2 * t, (Real.sqrt 2) / 2 * t)

-- Main proof statement
theorem sum_squared_distances {A B : ℝ × ℝ} (HA : C_rect A.fst A.snd) (HB : C_rect B.fst B.snd)
  (hlA : ∃ t1, l_param_eq t1 = A) (hlB : ∃ t2, l_param_eq t2 = B) :
  (∥(P.fst - A.fst, P.snd - A.snd)∥ ^ 2 + ∥(P.fst - B.fst, P.snd - B.snd)∥ ^ 2) = 4 := by
  sorry

end sum_squared_distances_l107_107902


namespace positive_difference_solutions_abs_l107_107615

theorem positive_difference_solutions_abs (x1 x2 : ℝ) 
  (h1 : 2 * x1 - 3 = 18 ∨ 2 * x1 - 3 = -18) 
  (h2 : 2 * x2 - 3 = 18 ∨ 2 * x2 - 3 = -18) : 
  |x1 - x2| = 18 :=
sorry

end positive_difference_solutions_abs_l107_107615


namespace find_number_l107_107598

theorem find_number (x : ℤ) (h : 35 - 3 * x = 14) : x = 7 :=
by {
  sorry -- This is where the proof would go.
}

end find_number_l107_107598


namespace sequence_periodicity_l107_107121

def a (n : ℕ) : ℝ :=
  if n = 0 then 6/7 else
    if (n % 3 = 1) || (n % 3 = 2) then
      2 * a (n-1) - 1 
    else
      2 * a (n-3)

theorem sequence_periodicity : a 2016 = 3/7 :=
  sorry -- This skips the proof for now.

end sequence_periodicity_l107_107121


namespace median_equality_and_range_inequality_l107_107765

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l107_107765


namespace sculpture_radius_l107_107529

theorem sculpture_radius (R : ℝ) 
  (V₄ : ℝ := 4/3 * π * 4^3) 
  (V₆ : ℝ := 4/3 * π * 6^3) 
  (V₈ : ℝ := 4/3 * π * 8^3) 
  (total_volume : ℝ := V₄ + V₆ + V₈) :
  (4/3 * π * R^3 = total_volume) → 
  (R = real.cbrt 792) :=
by
  sorry

end sculpture_radius_l107_107529


namespace nonnegatives_as_disjoint_historic_sets_l107_107665

def is_historic_set (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ ({z - y, y - x} = {1776, 2001})

theorem nonnegatives_as_disjoint_historic_sets :
  ∃ (S : ℕ → set ℕ), (∀ n, ∃ a b c, a < b ∧ b < c ∧ is_historic_set a b c ∧ S n = {a, b, c}) ∧
  (⋃ n, S n) = set.univ ∧
  (∀ i j, i ≠ j → S i ∩ S j = ∅) :=
by 
  sorry

end nonnegatives_as_disjoint_historic_sets_l107_107665


namespace maximum_distance_from_point_to_circle_l107_107579

noncomputable def distance (A B : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

noncomputable def circle_center_radius (a b c : ℝ) : ((ℝ × ℝ) × ℝ) :=
  ((0, -1), 1) -- The given circle equation is of form x^2 + y^2 + 2y = 0 

theorem maximum_distance_from_point_to_circle (A : (ℝ × ℝ)) (C : ((ℝ × ℝ) × ℝ)) :
  A = (2, 1) →
  C = ((0, -1), 1) →
  ∃ d : ℝ, d = (distance A (C.1).1) + C.2 ∧ d = 2 * Real.sqrt 2 + 1 :=
by
  intros
  simp *
  sorry -- proof skipped


end maximum_distance_from_point_to_circle_l107_107579


namespace number_of_cubes_l107_107630

theorem number_of_cubes (L W H V_cube : ℝ) (L_eq : L = 9) (W_eq : W = 12) (H_eq : H = 3) (V_cube_eq : V_cube = 3) :
  L * W * H / V_cube = 108 :=
by
  sorry

end number_of_cubes_l107_107630


namespace number_of_pairs_divisible_by_five_l107_107353

theorem number_of_pairs_divisible_by_five :
  (∃ n : ℕ, n = 864) ↔
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 80) ∧ (1 ≤ b ∧ b ≤ 30) →
  (a * b) % 5 = 0 → (∃ n : ℕ, n = 864) := 
sorry

end number_of_pairs_divisible_by_five_l107_107353


namespace grove_town_fall_expenditure_l107_107988

-- Define the expenditures at the end of August and November
def expenditure_end_of_august : ℝ := 3.0
def expenditure_end_of_november : ℝ := 5.5

-- Define the spending during fall months (September, October, November)
def spending_during_fall_months : ℝ := 2.5

-- Statement to be proved
theorem grove_town_fall_expenditure :
  expenditure_end_of_november - expenditure_end_of_august = spending_during_fall_months :=
by
  sorry

end grove_town_fall_expenditure_l107_107988


namespace smallest_prime_less_than_perfect_square_l107_107165

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107165


namespace ratio_u_p_l107_107861

theorem ratio_u_p : 
  (∀ (p r s u : ℝ), p / r = 4 → s / r = 8 → s / u = 1 / 4 → u / p = 8) := 
begin
  intros p r s u h1 h2 h3,
  sorry
end

end ratio_u_p_l107_107861


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107789

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107789


namespace sum_fractions_is_5_l107_107933

def f (x : ℚ) : ℚ := 4^x / (4^x + 2)

theorem sum_fractions_is_5 : 
  (∑ i in (Finset.range 10).map (λ i, (i + 1) / 11 : ℚ) f) = 5 := 
sorry

end sum_fractions_is_5_l107_107933


namespace perpendicular_vectors_l107_107846

def vector (α : Type) := (α × α)
def dot_product {α : Type} [Add α] [Mul α] (a b : vector α) : α :=
  a.1 * b.1 + a.2 * b.2

theorem perpendicular_vectors
    (a : vector ℝ) (b : vector ℝ)
    (h : dot_product a b = 0)
    (ha : a = (2, 4))
    (hb : b = (-1, n)) : 
    n = 1 / 2 := 
  sorry

end perpendicular_vectors_l107_107846


namespace point_Q_in_second_quadrant_l107_107394

theorem point_Q_in_second_quadrant (a : ℝ) (h : a < 0) : 
  let Qx := -a^2 - 1
  let Qy := -a + 1
  Qx < 0 ∧ Qy > 0 :=
by 
  let Qx := -a^2 - 1
  let Qy := -a + 1
  split
  . have : a^2 > 0 := by sorry
    sorry -- proof that Qx < 0
  . sorry -- proof that Qy > 0

end point_Q_in_second_quadrant_l107_107394


namespace distance_between_closest_points_of_tangent_circles_l107_107304

/-- The distance between the closest points of two circles with centers at
    (5, -4) and (20, 0) and tangent respectively to the line y = -3 
    is sqrt(241) - 4. -/
theorem distance_between_closest_points_of_tangent_circles :
  let c₁ := (5, -4)
  let c₂ := (20, 0)
  let l := λ p : ℝ × ℝ, p.2 + 3
  let dist (p q : ℝ × ℝ) := real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)
  l c₁ = 0 ∧ l c₂ = 0 →
  (dist c₁ c₂) - (|(-4) - (-3)| + |0 - (-3)|) = real.sqrt 241 - 4 :=
by
  sorry

end distance_between_closest_points_of_tangent_circles_l107_107304


namespace candy_total_cents_l107_107442

def candy_cost : ℕ := 8
def gumdrops : ℕ := 28
def total_cents : ℕ := 224

theorem candy_total_cents : candy_cost * gumdrops = total_cents := by
  sorry

end candy_total_cents_l107_107442


namespace length_squared_of_min_linear_graph_correct_l107_107979

noncomputable def length_squared_of_min_linear_graph (f g h m : ℝ → ℝ)
  (hf : ∀ x, f x = a₁ * x + b₁) (hg : ∀ x, g x = a₂ * x + b₂) 
  (hh : ∀ x, h x = a₃ * x + b₃) (hm : ∀ x, m x = a₄ * x + b₄)
  (j : ℝ → ℝ := λ x, max (max (f x) (g x)) (max (h x) (m x)))
  (k : ℝ → ℝ := λ x, min (min (f x) (g x)) (min (h x) (m x))) :
  ℓ : ℝ :=
  ∑ i in (finset.range 8), _ -- Assume some function of the length calculation from Lean's library.
  
theorem length_squared_of_min_linear_graph_correct (f g h m : ℝ → ℝ)
  (hf : ∀ x, f x = a₁ * x + b₁) (hg : ∀ x, g x = a₂ * x + b₂) 
  (hh : ∀ x, h x = a₃ * x + b₃) (hm : ∀ x, m x = a₄ * x + b₄)
  (j : ℝ → ℝ := λ x, max (max (f x) (g x)) (max (h x) (m x)))
  (k : ℝ → ℝ := λ x, min (min (f x) (g x)) (min (h x) (m x))) :
  (let ℓ := length_squared_of_min_linear_graph f g h m hf hg hh hm j k in ℓ^2 = 128) :=
by
  sorry

end length_squared_of_min_linear_graph_correct_l107_107979


namespace volume_of_circumscribed_sphere_l107_107826

theorem volume_of_circumscribed_sphere (PA : ℝ) (hPA : PA = 2) (right_angled_faces : ∀ (A B C : Type) (P : Type), 
(∃ (face1 : Prop), face1 = right_angled_triangle P A B) ∧
(∃ (face2 : Prop), face2 = right_angled_triangle P B C) ∧ 
(∃ (face3 : Prop), face3 = right_angled_triangle P C A)) :
  ∃ V, V = (4 / 3) * π * 1^3 :=
by {
  sorry
}

end volume_of_circumscribed_sphere_l107_107826


namespace number_of_male_athletes_sampled_l107_107271

theorem number_of_male_athletes_sampled (total_athletes : ℕ) (female_athletes : ℕ) (prob_selection : ℚ) :
  total_athletes = 98 → female_athletes = 42 → prob_selection = 2 / 7 →
  let male_athletes := total_athletes - female_athletes in
  let sampled_male_athletes := male_athletes * prob_selection in
  sampled_male_athletes = 16 :=
by intros; simp; sorry

end number_of_male_athletes_sampled_l107_107271


namespace exist_V_N_rhombus_l107_107508

-- Define the problem setup in Lean
variable {ABC : Type} [triangle : Triangle ABC]
variable (A B C E D F G : Point ABC)
variable (V N : Point ABC)

-- Assumptions
axiom altitude_AE : Altitude A E
axiom tangency_D : Tangency (Excircle A) (BC) D
axiom intersection_FG : Intersects (Excircle A) (Circumcircle ABC) F G

-- Theorem statement translated into Lean 4
theorem exist_V_N_rhombus 
  (h1 : altitude_AE)
  (h2 : tangency_D)
  (h3 : intersection_FG) :
  ∃ V N : Point ABC, OnLine V (Line D G) ∧ OnLine N (Line D F) ∧ Rhombus E V A N :=
sorry

end exist_V_N_rhombus_l107_107508


namespace roots_cubic_properties_l107_107921

theorem roots_cubic_properties (a b c : ℝ) 
    (h1 : ∀ x : ℝ, x^3 - 2 * x^2 + 3 * x - 4 = 0 → x = a ∨ x = b ∨ x = c)
    (h_sum : a + b + c = 2)
    (h_prod_sum : a * b + b * c + c * a = 3)
    (h_prod : a * b * c = 4) :
  a^3 + b^3 + c^3 = 2 := by
  sorry

end roots_cubic_properties_l107_107921


namespace center_of_circle_on_line_l107_107830

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 1 = 0

def line_parametric (t : ℝ) (x y : ℝ) : Prop := (x = 4 * t + 3) ∧ (y = 3 * t + 1)

def is_center_on_line (x y : ℝ) : Prop := (3 * x - 4 * y = 5)

theorem center_of_circle_on_line :
  ∃ x y : ℝ, circle_eq x y ∧ is_center_on_line x y :=
by {
  use (1 : ℝ), (2 : ℝ),
  split,
  -- Show that the point (1, 2) is the center of the circle
  {
    show circle_eq 1 2,
    simp [circle_eq],
  },
  -- Show that the point (1, 2) lies on the line
  {
    show is_center_on_line 1 2,
    simp [is_center_on_line],
  },
}

end center_of_circle_on_line_l107_107830


namespace parabola_transform_l107_107422

theorem parabola_transform (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + c = (x - 4)^2 - 3) → 
  b = 4 ∧ c = 6 := 
by
  sorry

end parabola_transform_l107_107422


namespace sequence_sum_l107_107640

theorem sequence_sum : 
  (∑ n in (Finset.range 50).map (λ x, 10 + 20 * x), 
   (if even x then 1990 - 20 * x else 2000 - 10 - 10 - 20 * (x + 1)))  
  + 10 = 1000 := 
sorry

end sequence_sum_l107_107640


namespace student_B_street_dance_club_l107_107301

-- Definitions for the conditions
def participated_in (student club : Prop) := student

variables {A B C : Prop}
variables {street_dance_club anime_club instrumental_music_club : Prop}

-- Condition 1: Student A has participated in more clubs than B, but A has not participated in the anime club.
def condition1 := ¬participated_in A anime_club ∧ (participated_in A street_dance_club ∧ participated_in A instrumental_music_club → ¬ (participated_in B street_dance_club ∧ participated_in B instrumental_music_club))

-- Condition 2: Student B has not participated in the instrumental music club.
def condition2 := ¬participated_in B instrumental_music_club

-- Condition 3: All three students have participated in the same club.
def condition3 := ∀ x, (participated_in A x ↔ participated_in B x) ∧ (participated_in B x ↔ participated_in C x) 

-- Theorem to prove: Student B has participated in the street dance club
theorem student_B_street_dance_club
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) :
  participated_in B street_dance_club :=
sorry

end student_B_street_dance_club_l107_107301


namespace conference_handshakes_l107_107011

def totalHandshakes (n : ℕ) (spouses : ℕ) (nonShakingGroup : ℕ) : ℕ :=
  let total_shakes = (n * (n - 1)) / 2
  let remove_spouses = spouses
  let remove_nonShaking = (nonShakingGroup * (nonShakingGroup - 1)) / 2
  total_shakes - remove_spouses - remove_nonShaking

theorem conference_handshakes :
  let n := 30
  let spouses := 15
  let nonShakingGroup := 3
  totalHandshakes n spouses nonShakingGroup = 417 :=
by
  sorry

end conference_handshakes_l107_107011


namespace zachary_pushups_l107_107626

variable (Zachary David John : ℕ)
variable (h1 : David = Zachary + 39)
variable (h2 : John = David - 13)
variable (h3 : David = 58)

theorem zachary_pushups : Zachary = 19 :=
by
  -- Proof goes here
  sorry

end zachary_pushups_l107_107626


namespace ways_to_partition_6_into_4_boxes_l107_107855

theorem ways_to_partition_6_into_4_boxes : 
  ∃ (s : Finset (Finset ℕ)), (∀ (x ∈ s), ∃ (a b c d : ℕ), x = {a, b, c, d} ∧ a + b + c + d = 6) ∧ s.card = 9 :=
sorry

end ways_to_partition_6_into_4_boxes_l107_107855


namespace bronchitis_option_D_correct_l107_107019

noncomputable def smoking_related_to_bronchitis : Prop :=
  -- Conclusion that "smoking is related to chronic bronchitis"
sorry

noncomputable def confidence_level : ℝ :=
  -- Confidence level in the conclusion
  0.99

theorem bronchitis_option_D_correct :
  smoking_related_to_bronchitis →
  (confidence_level > 0.99) →
  -- Option D is correct: "Among 100 smokers, it is possible that not a single person has chronic bronchitis"
  ∃ (P : ℕ → Prop), (∀ n : ℕ, n ≤ 100 → P n = False) :=
by sorry

end bronchitis_option_D_correct_l107_107019


namespace numWaysToChoosePairs_is_15_l107_107858

def numWaysToChoosePairs : ℕ :=
  let white := Nat.choose 5 2
  let brown := Nat.choose 3 2
  let blue := Nat.choose 2 2
  let black := Nat.choose 2 2
  white + brown + blue + black

theorem numWaysToChoosePairs_is_15 : numWaysToChoosePairs = 15 := by
  -- We will prove this theorem in actual proof
  sorry

end numWaysToChoosePairs_is_15_l107_107858


namespace largest_constant_inequality_l107_107705

theorem largest_constant_inequality :
  ∃ C, C = 3 ∧
  (∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 
  C * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂))) :=

sorry

end largest_constant_inequality_l107_107705


namespace hyperbola_eq_l107_107395

theorem hyperbola_eq (F : ℝ × ℝ) (N : ℝ × ℝ) :
  (F = (3, 0)) →
  (N = (-12, -15)) →
  ∃ a b : ℝ, (a ^ 2 + b ^ 2 = 9) ∧ (4 * b ^ 2 = 5 * a ^ 2) ∧
  (∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 ^ 2) / (a ^ 2) - (p.2 ^ 2) / (b ^ 2) = 1) ↔
    (p.1 ^ 2) / 4 - (p.2 ^ 2) / 5 = 1) :=
by
  intros
  sorry

end hyperbola_eq_l107_107395


namespace employee_salary_percentage_l107_107133

theorem employee_salary_percentage (A B : ℝ)
    (h1 : A + B = 450)
    (h2 : B = 180) : (A / B) * 100 = 150 := by
  sorry

end employee_salary_percentage_l107_107133


namespace maximize_sum_of_digits_l107_107518

-- Define distinctness of digits and the set of digits
def is_digit (x : ℕ) : Prop := x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def distinct_digits (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem maximize_sum_of_digits (A B C D : ℕ) (h_dist : distinct_digits A B C D)
    (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D) (hCD_gt1 : C + D > 1)
    (hInt : ∃ k : ℕ, A + B = k * (C + D)) : A + B = 15 :=
by  
  sorry

end maximize_sum_of_digits_l107_107518


namespace find_n_l107_107699

/-- 
The sequence defined such that the k-th block consists of a 1 followed by k 2's. 
-/
def sequence_term : ℕ → ℕ
| n => let (m, r) := (nat.sqrt(8 * n + 1) - 1) / 2, (n - m * (m + 1) / 2)
  if r = 0 then 1 else 2

def sequence_sum : ℕ → ℕ
| 0     => 0
| (n+1) => sequence_sum n + sequence_term (n+1)

theorem find_n : ∃ n, sequence_sum n = 2010 := sorry

end find_n_l107_107699


namespace cos_diff_sum_l107_107832

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + x + 1 else 2 * x + 1

theorem cos_diff_sum
  {α β γ : ℝ}
  (h1 : f (sin α + sin β + sin γ - 1) = -1)
  (h2 : f (cos α + cos β + cos γ + 1) = 3) :
  cos (α - β) + cos (β - γ) = -1 :=
by
  sorry

end cos_diff_sum_l107_107832


namespace pears_worth_l107_107562

variable (apples pears : ℚ)
variable (h : 3/4 * 16 * apples = 6 * pears)

theorem pears_worth (h : 3/4 * 16 * apples = 6 * pears) : 1 / 3 * 9 * apples = 1.5 * pears :=
by
  sorry

end pears_worth_l107_107562


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107166

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107166


namespace minkowski_convex_point_with_integer_coords_l107_107582

theorem minkowski_convex_point_with_integer_coords (F : set (ℝ × ℝ)) (h_convex : convex ℝ F)
  (h_symmetry : ∀ x ∈ F, -x ∈ F) (h_area : μ F > 4) : 
  ∃ (x : ℝ × ℝ), x ∈ F ∧ (∃ (a b : ℤ), x = (a, b)) ∧ x ≠ (0, 0) :=
sorry

end minkowski_convex_point_with_integer_coords_l107_107582


namespace tangent_line_to_parabola_l107_107361

-- Define the line and parabola equations
def line (x y k : ℝ) := 4 * x + 3 * y + k = 0
def parabola (x y : ℝ) := y ^ 2 = 16 * x

-- Prove that if the line is tangent to the parabola, then k = 9
theorem tangent_line_to_parabola (k : ℝ) :
  (∃ (x y : ℝ), line x y k ∧ parabola x y ∧ (y^2 + 12 * y + 4 * k = 0 ∧ 144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end tangent_line_to_parabola_l107_107361


namespace fraction_subtraction_simplified_l107_107347

theorem fraction_subtraction_simplified : (7 / 17) - (4 / 51) = 1 / 3 := by
  sorry

end fraction_subtraction_simplified_l107_107347


namespace equation_solution_count_l107_107438

theorem equation_solution_count : 
  let num_roots := {n | 1 ≤ n ∧ n ≤ 50}
  let invalid_roots := {1, 8, 27}
  (num_roots \ invalid_roots).card = 47 :=
by
  sorry

end equation_solution_count_l107_107438


namespace find_side_length_a_correct_l107_107463

noncomputable def find_side_length_a (A : ℝ) (b : ℝ) (S : ℝ) : ℝ :=
  let sinA := Math.sin (A * Real.pi / 180) in
  let c := 4 in -- derived from area condition
  let a := 2 * Real.sqrt 3 in
  a

theorem find_side_length_a_correct :
  ∀ (A : ℝ) (b : ℝ) (S : ℝ), A = 60 → b = 2 → S = 2 * Real.sqrt 3 → find_side_length_a A b S = 2 * Real.sqrt 3 :=
by
  intros A b S hA hb hS
  dsimp [find_side_length_a]
  rw [hA, hb, hS]
  rw [Real.sin_eq]
  norm_num
  sorry

end find_side_length_a_correct_l107_107463


namespace repeating_decimal_one_third_repeating_decimal_one_seventh_repeating_decimal_one_ninth_l107_107363

theorem repeating_decimal_one_third : has_periodic_decimal (1 / 3) (3) := sorry

theorem repeating_decimal_one_seventh : has_periodic_decimal (1 / 7) (142857) := sorry

theorem repeating_decimal_one_ninth : has_periodic_decimal (1 / 9) (1) := sorry

end repeating_decimal_one_third_repeating_decimal_one_seventh_repeating_decimal_one_ninth_l107_107363


namespace geometric_series_eq_l107_107327

theorem geometric_series_eq (y : ℕ) (h₁ : (∑' (n : ℕ), (1/3 : ℝ) ^ n) = 3 / 2) 
    (h₂ : (∑' (n : ℕ), ((-1/3 : ℝ) ^ n)) = 3 / 4) :
    (1 + ∑' (n : ℕ), (1/3 : ℝ) ^ n) * (1 - ∑' (n : ℕ), ((-1/3 : ℝ) ^ n)) =
    1 + ∑' (n : ℕ), (1 / (y : ℝ) ^ n) ↔ y = 9 :=
begin
    sorry
end

end geometric_series_eq_l107_107327


namespace find_a4_l107_107428

open Nat

def seq (a : ℕ → ℝ) := (a 1 = 1) ∧ (∀ n : ℕ, a (n + 1) = (2 * a n) / (a n + 2))

theorem find_a4 (a : ℕ → ℝ) (h : seq a) : a 4 = 2 / 5 :=
  sorry

end find_a4_l107_107428


namespace number_of_subsets_M_l107_107739

-- Define the imaginary unit with its primary property
def imaginary_unit : ℂ := Complex.i

-- Define x as given in the problem
def x (n : ℕ) := (imaginary_unit ^ n) + (imaginary_unit ^ -(n : ℤ))

-- Define the set S of all possible values of x
def S : Set ℂ := { x n | ∃ n : ℕ, x = (imaginary_unit ^ n) + (imaginary_unit ^ -↑n) }

-- Calculate the number of sets M within S
theorem number_of_subsets_M : (Set ℂ) := 
  let values := { 2, 0, -2 }
  have card_values : values.card = 3 := by sorry 
  show card (set.powerset values) = 2^3 := sorry

end number_of_subsets_M_l107_107739


namespace unique_exponential_solution_l107_107048

theorem unique_exponential_solution (a x : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hx_pos : 0 < x) :
  ∃! y : ℝ, a^y = x :=
by
  sorry

end unique_exponential_solution_l107_107048


namespace average_age_increase_by_one_l107_107985

-- Definitions based on the conditions.
def initial_average_age : ℕ := 14
def initial_students : ℕ := 10
def new_students_average_age : ℕ := 17
def new_students : ℕ := 5

-- Helper calculation for the total age of initial students.
def total_age_initial_students := initial_students * initial_average_age

-- Helper calculation for the total age of new students.
def total_age_new_students := new_students * new_students_average_age

-- Helper calculation for the total age of all students.
def total_age_all_students := total_age_initial_students + total_age_new_students

-- Helper calculation for the number of all students.
def total_students := initial_students + new_students

-- Calculate the new average age.
def new_average_age := total_age_all_students / total_students

-- The goal is to prove the increase in average age is 1 year.
theorem average_age_increase_by_one :
  new_average_age - initial_average_age = 1 :=
by
  -- Proof goes here
  sorry

end average_age_increase_by_one_l107_107985


namespace parabola_ellipse_sum_distances_l107_107659

noncomputable def sum_distances_intersection_points (b c : ℝ) : ℝ :=
  2 * Real.sqrt b + 2 * Real.sqrt c

theorem parabola_ellipse_sum_distances
  (A B : ℝ)
  (h1 : A > 0) -- semi-major axis condition implied
  (h2 : B > 0) -- semi-minor axis condition implied
  (ellipse_eq : ∀ x y, (x^2) / A^2 + (y^2) / B^2 = 1)
  (focus_shared : ∃ f : ℝ, f = Real.sqrt (A^2 - B^2))
  (directrix_parabola : ∃ d : ℝ, d = B) -- directrix condition
  (intersections : ∃ (b c : ℝ), (b > 0 ∧ c > 0)) -- existence of such intersection points
  : sum_distances_intersection_points b c = 2 * Real.sqrt b + 2 * Real.sqrt c :=
sorry  -- proof omitted

end parabola_ellipse_sum_distances_l107_107659


namespace problem1_problem2_l107_107295

-- Proof problem (1)
theorem problem1 : 
  ( (9 / 4) ^ (1 / 2) - (-9.6) ^ 0 - (27 / 8) ^ (-2 / 3) + (3 / 2) ^ (-2) ) = (1 / 2) := 
by sorry

-- Proof problem (2)
theorem problem2 : 
  ( logBase 3 (sqrt 3) + log 10 25 + log 10 4 + (7 : ℝ) ^ (logBase 7 2) ) = (9 / 2) := 
by sorry

end problem1_problem2_l107_107295


namespace system_of_inequalities_solve_equation_simplify_expression_l107_107639

-- Define the first problem that verifies the solution of the system of inequalities
theorem system_of_inequalities (x : ℝ) :
  (x - 4 < 2 * (x - 1) ∧ (1 + 2 * x) / 3 ≥ x) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

-- Define the second problem which verifies the solution to the given equation
theorem solve_equation (x : ℝ) : 
  (x - 2) / (x - 3) = 2 - 1 / (3 - x) ↔ x = 3 :=
by 
  sorry

-- Define the third problem which simplifies the given expression and substitutes x = 3
theorem simplify_expression :
  let expr := 1 - 1 / (3 + 2) / ((3 - 1) * (3 - 2)) in 
  (expr = 1 / 2) :=
by
  sorry

end system_of_inequalities_solve_equation_simplify_expression_l107_107639


namespace TrapezoidDividerSolution_l107_107482

def TrapezoidDividerProblem (A B C D M : Type) (AD BC CM : ℝ) : Prop :=
  AD = 16 ∧ BC = 9 ∧ CM = 3.2 → (8 : ℝ) / 15 = 8 / (8 + 15)

theorem TrapezoidDividerSolution {A B C D M : Type} {AD BC CM : ℝ} :
  TrapezoidDividerProblem A B C D M AD BC CM :=
begin
  intros h,
  have h1 : AD = 16 := by simp [h.1],
  have h2 : BC = 9 := by simp [h.2.1],
  have h3 : CM = 3.2 := by simp [h.2.2],
  simp [h1, h2, h3],
  sorry,
end

end TrapezoidDividerSolution_l107_107482


namespace range_of_a_l107_107410

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if 0 ≤ x then a * x ^ 2 + 1 else (a ^ 2 - 1) * Real.exp (a * x)

theorem range_of_a : ∀ (a : ℝ), 
  (∀ (x Δx : ℝ), 
    (Δx ≠ 0) → 
    (if 0 ≤ x then
      (a * (x + Δx) ^ 2 + 1 - (a * x ^ 2 + 1)) / Δx > 0
     else
      ((a ^ 2 - 1) * Real.exp (a * (x + Δx)) - (a ^ 2 - 1) * Real.exp (a * x)) / Δx > 0)
  ) → (1 < a ∧ a ≤ Real.sqrt 2) :=
by
  intro a
  sorry

end range_of_a_l107_107410


namespace no_partition_nat_greater_than_one_l107_107086

theorem no_partition_nat_greater_than_one (A B : set ℕ) (hA : ∃ x ∈ A, 2 ≤ x)
  (hB : ∃ x ∈ B, 2 ≤ x) (hU : ∀ x, x > 1 → x ∈ A ∨ x ∈ B) 
  (h_cond : ∀ a b, a ∈ A ∧ b ∈ A → (a * b - 1) ∈ A) 
  (h_cond' : ∀ a b, a ∈ B ∧ b ∈ B → (a * b - 1) ∈ B) : false :=
sorry

end no_partition_nat_greater_than_one_l107_107086


namespace parabola_tangent_focus_solution_l107_107596

noncomputable def parabola_tangent_focus (t₁ t₂ t₃ e : Line) : ℕ :=
sorry

theorem parabola_tangent_focus_solution (t₁ t₂ t₃ e : Line) :
  let circle_exists (t1 t2 t3 : Line) := Sorry,
  let intersections (circ : Circle) (e : Line) := Sorry in
  circle_exists t₁ t₂ t₃ →
  intersections (circ) e = 2 → parabola_tangent_focus t₁ t₂ t₃ e = 2 ∧
  intersections (circ) e = 1 → parabola_tangent_focus t₁ t₂ t₃ e = 1 ∧
  intersections (circ) e = 0 → parabola_tangent_focus t₁ t₂ t₃ e = 0 :=
sorry

end parabola_tangent_focus_solution_l107_107596


namespace curve_C_equation_min_area_of_triangle_QAB_l107_107423

theorem curve_C_equation (P N : ℝ × ℝ) (O : ℝ × ℝ) :
  (P.1, P.2) = (O.1 / 2, O.2 / 2) → 
  N.2^2 = 8 * N.1 → 
  P.2^2 = 4 * P.1 := 
  sorry

theorem min_area_of_triangle_QAB (x₀ y₀ : ℝ) (h₀ : x₀ ≥ 5)
  (M : set (ℝ × ℝ)) (Q : ℝ × ℝ) (A B : ℝ × ℝ) :
  (∀ x ∈ M, (x.1 - 2)^2 + x.2^2 = 4) →
  Q ∈ { P | P.2^2 = 4 * P.1 } →
  2 * x₀ * y₀ / (x₀ - 1) - 4 * y₀ / (x₀ - 1) = 25 / 2 :=
  sorry

end curve_C_equation_min_area_of_triangle_QAB_l107_107423


namespace X_investment_l107_107225

theorem X_investment (P : ℝ) 
  (Y_investment : ℝ := 42000)
  (Z_investment : ℝ := 48000)
  (Z_joins_at : ℝ := 4)
  (total_profit : ℝ := 14300)
  (Z_share : ℝ := 4160) :
  (P * 12 / (P * 12 + Y_investment * 12 + Z_investment * (12 - Z_joins_at))) * total_profit = Z_share → P = 35700 :=
by
  sorry

end X_investment_l107_107225


namespace sum_of_pos_real_solutions_l107_107359

open Real

noncomputable def cos_equation_sum_pos_real_solutions : ℝ := 1082 * π

theorem sum_of_pos_real_solutions :
  ∃ x : ℝ, (0 < x) ∧ 
    (∀ x, 2 * cos (2 * x) * (cos (2 * x) - cos ((2016 * π ^ 2) / x)) = cos (6 * x) - 1) → 
      x = cos_equation_sum_pos_real_solutions :=
sorry

end sum_of_pos_real_solutions_l107_107359


namespace number_of_people_choose_pop_l107_107467

theorem number_of_people_choose_pop (surveyed : ℕ) (pop_angle : ℝ) (total_angle : ℝ) :
  surveyed = 472 →
  pop_angle = 251 →
  total_angle = 360 →
  (surveyed * (pop_angle / total_angle)).round = 329 := 
by
  intros h_surveyed h_pop_angle h_total_angle
  rw [h_surveyed, h_pop_angle, h_total_angle]
  sorry

end number_of_people_choose_pop_l107_107467


namespace probability_three_marbles_l107_107648

theorem probability_three_marbles :
  let total_marbles := 10
      blue_marbles := 4
      green_marbles := 6
      first_draw_blue := blue_marbles / total_marbles
      second_draw_green := green_marbles / (total_marbles - 1)
      third_draw_green := (green_marbles - 1) / (total_marbles - 2)
  in (first_draw_blue * second_draw_green * third_draw_green) = 1 / 6 :=
by
  let total_marbles := 10
  let blue_marbles := 4
  let green_marbles := 6
  let first_draw_blue := blue_marbles / total_marbles
  let second_draw_green := green_marbles / (total_marbles - 1)
  let third_draw_green := (green_marbles - 1) / (total_marbles - 2)
  have h : (first_draw_blue * second_draw_green * third_draw_green) = (4 / 10) * (2 / 3) * (5 / 8), from sorry,
  have h' : (4 / 10) * (2 / 3) * (5 / 8) = 1 / 6, from sorry,
  exact h.trans h'.symm

end probability_three_marbles_l107_107648


namespace max_books_per_student_l107_107882

-- Define the variables and conditions
variables (students : ℕ) (not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 : ℕ)
variables (avg_books_per_student : ℕ)
variables (remaining_books : ℕ) (max_books : ℕ)

-- Assume given conditions
def conditions : Prop :=
  students = 100 ∧ 
  not_borrowed5 = 5 ∧ 
  borrowed1_20 = 20 ∧ 
  borrowed2_25 = 25 ∧ 
  borrowed3_30 = 30 ∧ 
  borrowed5_20 = 20 ∧ 
  avg_books_per_student = 3

-- Prove the maximum number of books any single student could have borrowed is 50
theorem max_books_per_student (students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student : ℕ) (max_books : ℕ) :
  conditions students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student →
  max_books = 50 :=
by
  sorry

end max_books_per_student_l107_107882


namespace birch_trees_probability_l107_107251

theorem birch_trees_probability :
  let n_trees := 4 + 5 + 5;
  let non_birch_trees := 4 + 5;
  let total_ways := Nat.choose n_trees 5;
  let no_adjacent_ways := Nat.choose (non_birch_trees + 1) 5;
  let probability := no_adjacent_ways / total_ways;
  let simplest_form := Rat.mkPnat no_adjacent_ways total_ways;
  m + n = 161 :=
  by
  let n_trees := 14;
  let non_birch_trees := 9;
  let total_ways := Nat.choose n_trees 5;
  let no_adjacent_ways := Nat.choose (non_birch_trees + 1) 5;
  let probability := no_adjacent_ways / total_ways;
  let simplest_form := Rat.mkPnat no_adjacent_ways total_ways;
  let m := simplest_form.num;
  let n := simplest_form.den;
  have h1 : probability = 18 / 143 := sorry,
  have h2 : m + n = 161 := sorry,
  exact h2

end birch_trees_probability_l107_107251


namespace median_divides_AA1_in_ratio_l107_107947

-- Definitions needed based on the given conditions
structure Point where
  x : ℝ
  y : ℝ

noncomputable def point_A1 (B C : Point) : Point :=
  ⟨(2 * B.x + C.x) / 3, (2 * B.y + C.y) / 3⟩

noncomputable def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def divide_ratio (A B C C1 A1 : Point) : ℝ :=
  let P := ⟨(A.x * 3 + 2 * A1.x) / 5, (A.y * 3 + 2 * A1.y) / 5⟩ -- The intersection point
  let C1_mid := midpoint A B
  if C1 = C1_mid then
    3
  else
    sorry  -- Placeholder to ensure the program can compile

theorem median_divides_AA1_in_ratio (A B C : Point) :
  let A1 := point_A1 B C
  let C1 := midpoint A B
  divide_ratio A B C C1 A1 = 3 :=
by {
  sorry  -- Proof is skipped
}

end median_divides_AA1_in_ratio_l107_107947


namespace percent_gold_coins_l107_107475

variables (total_objects : ℝ) (coins_beads_percent beads_percent gold_coins_percent : ℝ)
           (h1 : coins_beads_percent = 0.75)
           (h2 : beads_percent = 0.15)
           (h3 : gold_coins_percent = 0.60)

theorem percent_gold_coins : (gold_coins_percent * (coins_beads_percent - beads_percent)) = 0.36 :=
by
  have coins_percent := coins_beads_percent - beads_percent
  have gold_coins_total_percent := gold_coins_percent * coins_percent
  exact sorry

end percent_gold_coins_l107_107475


namespace kamal_marks_physics_correct_l107_107914

-- Definition of the conditions
def kamal_marks_english : ℕ := 76
def kamal_marks_mathematics : ℕ := 60
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 74
def kamal_num_subjects : ℕ := 5

-- Definition of the total marks
def kamal_total_marks : ℕ := kamal_average_marks * kamal_num_subjects

-- Sum of known marks
def kamal_known_marks : ℕ := kamal_marks_english + kamal_marks_mathematics + kamal_marks_chemistry + kamal_marks_biology

-- The expected result for Physics
def kamal_marks_physics : ℕ := 82

-- Proof statement
theorem kamal_marks_physics_correct :
  kamal_total_marks - kamal_known_marks = kamal_marks_physics :=
by
  simp [kamal_total_marks, kamal_known_marks, kamal_marks_physics]
  sorry

end kamal_marks_physics_correct_l107_107914


namespace initial_distance_l107_107724

-- Define conditions
def fred_speed : ℝ := 4
def sam_speed : ℝ := 4
def sam_distance_when_meet : ℝ := 20

-- States that the initial distance between Fred and Sam is 40 miles considering the given conditions.
theorem initial_distance (d : ℝ) (fred_speed_eq : fred_speed = 4) (sam_speed_eq : sam_speed = 4) (sam_distance_eq : sam_distance_when_meet = 20) :
  d = 40 :=
  sorry

end initial_distance_l107_107724


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107173

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107173


namespace polynomial_with_integer_roots_exists_l107_107124

theorem polynomial_with_integer_roots_exists :
  ∃ p q : ℤ, ∃ n : ℤ, 
  (∀ k : ℤ, 0 ≤ k < n → (p = 10 + k ∨ p = 20 - k)) ∧ 
  (q = 20 ∨ q = 10) ∧ 
  (p^2 - 4*q = 1) :=
sorry

end polynomial_with_integer_roots_exists_l107_107124


namespace least_integer_k_l107_107868

theorem least_integer_k (k : ℕ) (h : k ^ 3 ∣ 336) : k = 84 :=
sorry

end least_integer_k_l107_107868


namespace ant_rectangle_distance_l107_107675

theorem ant_rectangle_distance :
  ∀ (width height : ℝ) (angle side_midpoint : ℝ),
    width = 18 →
    height = 150 →
    angle = π / 4 →
    side_midpoint = height / 2 →
    ∃ X : ℝ, 
    X ∈ set.Icc 0 width ∧
    (nearest_corner_distance X width height angle  = 3) :=
begin
  intros width height angle side_midpoint w_eq h_eq a_eq s_eq,
  sorry
end

def nearest_corner_distance (X width height angle : ℝ) : ℝ := 
  -- The definition can be filled here based on problem geometrical interpretation
  sorry

end ant_rectangle_distance_l107_107675


namespace average_after_removal_l107_107987

-- Given the average of 12 numbers
def average_of_12_numbers (total_sum : ℝ) : ℝ := total_sum / 12

-- Define the total sum of the 12 numbers
def total_sum_12 := 90 * 12

-- Remove two specific numbers from the sum
def new_sum_excluding_two (total_sum : ℝ) (num1 num2 : ℝ) : ℝ := total_sum - num1 - num2

-- Calculate the average of the remaining 10 numbers
def average_of_remaining_10_numbers (new_sum : ℝ) : ℝ := new_sum / 10

theorem average_after_removal :
  average_of_remaining_10_numbers (new_sum_excluding_two total_sum_12 65 80) = 93.5 := by
  sorry

end average_after_removal_l107_107987


namespace proof_problem_l107_107787

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l107_107787


namespace average_of_x_y_z_l107_107446

theorem average_of_x_y_z (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := 
by 
  sorry

end average_of_x_y_z_l107_107446


namespace average_beef_sales_l107_107072

def ground_beef_sales.Thur : ℕ := 210
def ground_beef_sales.Fri : ℕ := 2 * ground_beef_sales.Thur
def ground_beef_sales.Sat : ℕ := 150
def ground_beef_sales.total : ℕ := ground_beef_sales.Thur + ground_beef_sales.Fri + ground_beef_sales.Sat
def ground_beef_sales.days : ℕ := 3
def ground_beef_sales.average : ℕ := ground_beef_sales.total / ground_beef_sales.days

theorem average_beef_sales (thur : ℕ) (fri : ℕ) (sat : ℕ) (days : ℕ) (total : ℕ) (avg : ℕ) :
  thur = 210 → 
  fri = 2 * thur → 
  sat = 150 → 
  total = thur + fri + sat → 
  days = 3 → 
  avg = total / days → 
  avg = 260 := by
    sorry

end average_beef_sales_l107_107072


namespace minimum_value_l107_107505

noncomputable def min_expr (x : ℝ) : ℝ := (x^2 + 3 - (x^4 + 9).sqrt) / x

theorem minimum_value :
  ∃ (x : ℝ) (hx : 0 < x), min_expr x = 6 / (2 * 3.sqrt + (6).sqrt) :=
sorry

end minimum_value_l107_107505


namespace sweets_remaining_l107_107257

def num_cherry := 30
def num_strawberry := 40
def num_pineapple := 50

def half (n : Nat) := n / 2

def num_eaten_cherry := half num_cherry
def num_eaten_strawberry := half num_strawberry
def num_eaten_pineapple := half num_pineapple

def num_given_away := 5

def total_initial := num_cherry + num_strawberry + num_pineapple

def total_eaten := num_eaten_cherry + num_eaten_strawberry + num_eaten_pineapple

def total_remaining_after_eating := total_initial - total_eaten
def total_remaining := total_remaining_after_eating - num_given_away

theorem sweets_remaining : total_remaining = 55 := by
  sorry

end sweets_remaining_l107_107257


namespace geometric_sequence_min_value_l107_107807

theorem geometric_sequence_min_value
  (q : ℝ) (a : ℕ → ℝ)
  (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n)
  (h_geom : ∀ k, a k = q ^ k)
  (h_eq : a m * (a n) ^ 2 = (a 4) ^ 2)
  (h_sum : m + 2 * n = 8) :
  ∀ (f : ℝ), f = (2 / m + 1 / n) → f ≥ 1 :=
by
  sorry

end geometric_sequence_min_value_l107_107807


namespace distance_between_red_lights_in_feet_l107_107973

theorem distance_between_red_lights_in_feet :
  let inches_between_lights := 6
  let pattern := [2, 3]
  let foot_in_inches := 12
  let pos_3rd_red := 6
  let pos_21st_red := 51
  let number_of_gaps := pos_21st_red - pos_3rd_red
  let total_distance_in_inches := number_of_gaps * inches_between_lights
  let total_distance_in_feet := total_distance_in_inches / foot_in_inches
  total_distance_in_feet = 22 := by
  sorry

end distance_between_red_lights_in_feet_l107_107973


namespace circle_C_equation_min_tangent_length_l107_107369

-- Define the ellipse and its foci
def ellipse (x y : ℝ) : Prop :=
  (x^2) / 4 + (y^2) / 3 = 1

def focus_1 : ℝ × ℝ := (-1, 0)
def focus_2 : ℝ × ℝ := (1, 0)

-- Definition of the symmetry line
def symmetry_line (x y : ℝ) : Prop :=
  x + y = 2

-- Define the equation of the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 1

-- Define the distance from point P(m,0) to the center of the circle C
def dist (m : ℝ) : ℝ :=
  real.sqrt ((2 - m)^2 + 2^2)

-- Length of the tangent from point P(m,0) to the circle
def tangent_length (m : ℝ) : ℝ :=
  real.sqrt ((2 - m)^2 + 2^2 - 1)

-- Theorems to prove
theorem circle_C_equation : circle_C 2 2 :=
  sorry

theorem min_tangent_length (m : ℝ) :
  (∃ m : ℝ, P.m = 2) ∧ (∃ l : ℝ, tangent_length m = real.sqrt 3) :=
  sorry

end circle_C_equation_min_tangent_length_l107_107369


namespace smallest_prime_less_than_perfect_square_l107_107161

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107161


namespace line_equation_from_conditions_l107_107820

theorem line_equation_from_conditions :
  ∃ l : ℝ → ℝ → Prop, 
    (∀ x y, l x y → (abs (3 * 3 + -1 * y - (-4))) / √(3^2 + (-1)^2) = (abs (3 * 5 + -1 * y - (-7))) / √(3^2 + (-1)^2)) ∧ 
    ∃ p : ℝ × ℝ, (p.fst = 1 ∧ p.snd = 2) ∧ l p.fst p.snd ∧ 
    (l = λ x y, 1 * x - 6 * y + 11 = 0 ∨ l = λ x y, 1 * x + 2 * y - 5 = 0) :=
by
  sorry

end line_equation_from_conditions_l107_107820


namespace binomial_600_600_l107_107312

-- Define a theorem to state the binomial coefficient property and use it to prove the specific case.
theorem binomial_600_600 : nat.choose 600 600 = 1 :=
begin
  -- Binomial property: for any non-negative integer n, (n choose n) = 1
  rw nat.choose_self,
end

end binomial_600_600_l107_107312


namespace polynomial_evaluation_l107_107375

theorem polynomial_evaluation [Ring R] [AddMonoid R] [CommMonoid R] :
  let a : Fin 11 → ℤ := λ n, (Polynomial.Coeffi (Polynomial.Coeff (n_polynomial a)) (Polynomial.Coeff (Polynomial.shift_coeffs 1))) 
  let P (x: ℤ) := Polynomial.eval x (Polynomial.Polynomial a) in
    (∀ x: ℤ, P (1 + x) = a.nat_cast +
            ∑ a_1 * Polynomial.eval (1 - x) +
            ∑ a_2 * Polynomial.eval (Polynomial.eval (1-x) ^ 2) +
            ∑ a_10 * Polynomial.eval (Polynomial.eval (1-x) ^ 10))
  → P (2:ℤ) = 3^10
     (- a_0),
      (iterate 1, sum, sum ) :=
  sorry

end polynomial_evaluation_l107_107375


namespace min_value_y_l107_107429

-- We need to define three arbitrary points A, B, C such that BC = a, CA = b, AB = c. 
-- This can be encapsulated in a record for the sides of the triangle

structure TriSides :=
  (BC : ℝ) (CA : ℝ) (AB : ℝ)

-- Given TriSides a b c, prove that the minimum value of y = c / (a + b) + b / c is sqrt(2) - 1/2
theorem min_value_y (sides : TriSides) (h1 : sides.BC <= sides.CA + sides.AB) :
  (∃ (a b c : ℝ), BC = a ∧ CA = b ∧ AB = c → 
  (y = sides.AB / (sides.BC + sides.CA) + sides.CA / sides.AB → y = sqrt 2 - 1 / 2)) := 
sorry -- proof omitted

end min_value_y_l107_107429


namespace find_y_in_triangle_l107_107906

theorem find_y_in_triangle (BAC ABC BCA : ℝ) (y : ℝ) (h1 : BAC = 90)
  (h2 : ABC = 2 * y) (h3 : BCA = y - 10) : y = 100 / 3 :=
by
  -- The proof will be left as sorry
  sorry

end find_y_in_triangle_l107_107906


namespace conjugate_of_complex_expr_l107_107934

noncomputable def complex_number (z : ℂ) : ℂ := 1 + z^2 + (2 / z)

theorem conjugate_of_complex_expr : 
  ∀ (z : ℂ), z = 1 + complex.I → conj (complex_number z) = 1 - complex.I := 
by
  intros z hz
  rw [complex_number, hz]
  sorry

end conjugate_of_complex_expr_l107_107934


namespace area_QOS_l107_107561

-- Definitions of the conditions
variables (O P Q R S : Type) (dist_PO dist_SO : ℝ) (area_POR : ℝ)
-- Hypotheses based on the given problem
hypothesis h1 : dist_PO = 3
hypothesis h2 : dist_SO = 4
hypothesis h3 : area_POR = 7

-- Theorem statement for the proof
theorem area_QOS : (∃ (area_QOS : ℝ), area_QOS = 112 / 9) :=
sorry

end area_QOS_l107_107561


namespace factor_expression_l107_107692

-- Define the expressions used in the condition
def expr1 : ℤ[X] := 4 * X^3 + 75 * X^2 - 12
def expr2 : ℤ[X] := -5 * X^3 + 3 * X^2 - 12
def combined_expr : ℤ[X] := expr1 - expr2
def factored_expr : ℤ[X] := 9 * X^2 * (X + 8)

-- State the theorem to be proven
theorem factor_expression : combined_expr = factored_expr :=
by
  -- The proof would go here
  sorry

end factor_expression_l107_107692


namespace number_of_people_l107_107874

theorem number_of_people (x : ℕ) : 
  (x % 10 = 1) ∧
  (x % 9 = 1) ∧
  (x % 8 = 1) ∧
  (x % 7 = 1) ∧
  (x % 6 = 1) ∧
  (x % 5 = 1) ∧
  (x % 4 = 1) ∧
  (x % 3 = 1) ∧
  (x % 2 = 1) ∧
  (x < 5000) →
  x = 2521 :=
sorry

end number_of_people_l107_107874


namespace vectors_coplane_if_condition_exists_l107_107954

theorem vectors_coplane_if_condition_exists (
  a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ
) : 
  (a1 * b2 * c3 + a2 * b3 * c1 + a3 * b1 * c2 = a1 * b3 * c2 + a2 * b1 * c3 + a3 * b2 * c1) ↔ 
  matrix.det ![
    ![a1, a2, a3],
    ![b1, b2, b3],
    ![c1, c2, c3]
  ] = 0 := 
sorry

end vectors_coplane_if_condition_exists_l107_107954


namespace log_expr_base_2_l107_107686

theorem log_expr_base_2 : 
  log 2 (64 * (16)^(1/3) * (32)^(1/5)) = 25 / 3 :=
by 
  have h64 : 64 = 2 ^ 6 := by norm_num
  have h16 : (16)^(1/3) = 2 ^ (4 / 3) := by norm_num
  have h32 : (32)^(1/5) = 2 := by norm_num
  sorry

end log_expr_base_2_l107_107686


namespace avg_of_xyz_l107_107444

-- Define the given condition
def given_condition (x y z : ℝ) := 
  (5 / 2) * (x + y + z) = 20

-- Define the question (and the proof target) using the given conditions.
theorem avg_of_xyz (x y z : ℝ) (h : given_condition x y z) : 
  (x + y + z) / 3 = 8 / 3 :=
sorry

end avg_of_xyz_l107_107444


namespace marie_completes_fourth_task_at_1220_l107_107064

/-
Prove that Marie finishes the fourth task at 12:20 PM, given the conditions:
1. Marie completes four equally time-consuming tasks sequentially.
2. She starts the first task at 9:00 AM.
3. She completes the third task at 11:30 AM.
-/

def time := Int

def start_first_task : time := 9 * 60  -- 9:00 AM in minutes
def end_third_task : time := 11 * 60 + 30  -- 11:30 AM in minutes

def total_tasks := 4
def tasks_completed := 3

theorem marie_completes_fourth_task_at_1220 :
  let task_duration := (end_third_task - start_first_task) / tasks_completed in
  let end_fourth_task := end_third_task + task_duration in
  end_fourth_task = 12 * 60 + 20 := by
  sorry

end marie_completes_fourth_task_at_1220_l107_107064


namespace find_m_l107_107824

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Points (0, -1) and (2, 7) on the quadratic function
def point1 : Prop := quadratic_function 0 = -1
def point2 : Prop := quadratic_function 2 = 7

-- The sum of y-values at points (-5, y1) and (m, y2) is 28
def y1 := quadratic_function (-5)
def y2 (m : ℝ) := quadratic_function m
def y_sum_condition (y1 y2 : ℝ) : Prop := y1 + y2 = 28

-- Statement of the problem
theorem find_m (m : ℝ) (h1 : point1) (h2 : point2) (h3 : y_sum_condition y1 (y2 m)) :
  m = 3 := 
sorry

end find_m_l107_107824


namespace antelope_total_distance_l107_107604

-- Definitions based on the conditions from the problem statement
def radius_smaller_circle := 15
def radius_larger_circle := 30

def semicircle_distance (radius : ℕ) : ℝ := π * radius
def radial_distance (r1 r2 : ℕ) : ℝ := |r2 - r1|

-- Problem statement as a theorem in Lean
theorem antelope_total_distance :
  let distance1 := semicircle_distance radius_smaller_circle
      distance2 := radial_distance radius_smaller_circle radius_larger_circle
      distance3 := semicircle_distance radius_larger_circle
      total_distance := distance1 + distance2 + distance3 + distance2 + distance1 in
  total_distance = 60 * π + 30 :=
  by
    sorry

end antelope_total_distance_l107_107604


namespace prove_part_I_prove_part_II_l107_107728

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (a b x : ℝ) : ℝ := (1 / 2) * a * x^2 + b * x
noncomputable def h (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x - g x

def part_I (a b : ℝ) : Prop :=
  a = 3 → b = 2 → h f (g a b) = -Real.log 3 - 5 / 6

def part_II (a b x₁ x₂ : ℝ) (f g : ℝ → ℝ) : Prop :=
  x₁ ≠ x₂ →
  0 < x₁ ∧ x₁ < x₂ →
  h f (g a b) x₁ = 0 →
  h f (g a b) x₂ = 0 →
  let x₀ := (x₁ + x₂) / 2
  in (Real.deriv (h f (g a b)) x₀ < 0)

-- We need to create a theorem for proving the statements.
theorem prove_part_I (a b : ℝ) : part_I a b :=
  by sorry

theorem prove_part_II (a b x₁ x₂ : ℝ) (f g : ℝ → ℝ) : 
  part_II a b x₁ x₂ f g :=
  by sorry

end prove_part_I_prove_part_II_l107_107728


namespace cos_sum_identity_l107_107966

theorem cos_sum_identity :
  cos (2 * Real.pi / 17) + cos (6 * Real.pi / 17) + cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_identity_l107_107966


namespace find_equation_of_line_l107_107253

-- Define the points P, M, and N
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨1, 2⟩
def M : Point := ⟨2, 3⟩
def N : Point := ⟨4, -5⟩

-- Define a line as a structure containing coefficients of the line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to calculate the distance from a point to a line
def distance_to_line (L : Line) (P : Point) : ℝ :=
  (abs (L.a * P.x + L.b * P.y + L.c)) / (sqrt (L.a ^ 2 + L.b ^ 2))

-- Define the conditions
def condition1 (L : Line) : Prop := L.a * P.x + L.b * P.y + L.c = 0
def condition2 (L : Line) : Prop := distance_to_line L M = distance_to_line L N

-- Define the proof problem
theorem find_equation_of_line (L : Line) :
  condition1 L ∧ condition2 L →
  (L.a = 4 ∧ L.b = 1 ∧ L.c = -6) ∨ (L.a = 3 ∧ L.b = 2 ∧ L.c = -7) :=
by
  sorry

end find_equation_of_line_l107_107253


namespace binom_600_eq_1_l107_107315

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l107_107315


namespace solve_rational_eq_l107_107350

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 3 * x - 18) + 1 / (x^2 - 15 * x - 12) = 0) →
  (x = 1 ∨ x = -1 ∨ x = 12 ∨ x = -12) :=
by
  intro h
  sorry

end solve_rational_eq_l107_107350


namespace exists_subset_sum_divisible_l107_107959

theorem exists_subset_sum_divisible {A : Finset ℤ} {n : ℕ} (hA : A.card = n) :
  ∃ B : Finset ℤ, B ⊆ A ∧ B.nonempty ∧ (B.sum id) % n = 0 :=
by
  sorry

end exists_subset_sum_divisible_l107_107959


namespace exists_point_with_distance_sum_ge_n_l107_107492

open Complex

-- Define the closed unit disk
def unit_disk : set ℂ := { z : ℂ | abs z ≤ 1 }

-- Define the problem statement in Lean 4
theorem exists_point_with_distance_sum_ge_n (z_1 z_2 ⋯ z_n : ℂ)
  (hzi : ∀ i, z_i ∈ unit_disk) :
  ∃ z ∈ unit_disk, (∑ i in finset.range n, abs (z - z_i)) ≥ n := sorry

end exists_point_with_distance_sum_ge_n_l107_107492


namespace hockey_league_games_l107_107595

theorem hockey_league_games (n : ℕ) (total_games : ℕ) (games_played : ℕ) (h_team_count : n = 18) (h_total_games_played : total_games = 1530) (h_games_equation : games_played = n * (n - 1)): 
  (total_games / games_played) = 5 :=
by
  -- conditions
  rw [h_team_count, h_total_games_played, h_games_equation]
  sorry

end hockey_league_games_l107_107595


namespace no_positive_integer_pairs_l107_107354

theorem no_positive_integer_pairs (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) : ¬ (x^2 + y^2 = x^3 + 2 * y) :=
by sorry

end no_positive_integer_pairs_l107_107354


namespace cube_root_of_64_is_4_l107_107993

-- Statement of the problem
theorem cube_root_of_64_is_4 : real.cbrt 64 = 4 :=
sorry

end cube_root_of_64_is_4_l107_107993


namespace example_theorems_l107_107903

def sequence (a : ℕ → ℝ) : Prop := ∀ n : ℕ, n ≠ 0 → 
  a 1 + a 2 + a 3 + ∑ j in (finset.Ico (3 + 1) (n + 1)), a j = n - a n

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := (2 - n) * (a n - 1)

theorem example_theorems {a : ℕ → ℝ} :
  sequence a →
  (a 1 = 1 / 2) ∧ (a 2 = 3 / 4) ∧ (a 3 = 7 / 8) ∧
  (∀ n : ℕ, 2 ≤ n → a n - 1 = (1 / 2) * (a (n-1) - 1)) ∧
  (∀ t : ℕ, (∀ n : ℕ, n ≠ 0 → b a n < t / 5) → 1 ≤ t) := 
by
  sorry

end example_theorems_l107_107903


namespace sum_largest_and_third_smallest_l107_107722

def digits : Set ℕ := {1, 6, 8}

def permutations (l : List ℕ) : List (List ℕ) :=
  l.permutations.filter (λ x, x ≠ [] ∧ x.length = 3)

-- selecting maximum and third minimum elements
def max_third_min (l : List ℕ) : ℕ :=
  match l.permSort! compare with
  | [] => 0
  | [a, _, c] => a + c
  | _ => 0

theorem sum_largest_and_third_smallest : max_third_min ([861, 168, 186, 618, 681, 816]) = 1479 :=
by
  intros
  dsimp [max_third_min]
  have : [861, 168, 186, 618, 681, 816].permSort! compare = [168, 186, 618, 681, 816, 861] :=
    by rfl
  simp [this]
  rfl

end sum_largest_and_third_smallest_l107_107722


namespace max_min_diff_of_c_l107_107502

-- Definitions and conditions
variables (a b c : ℝ)
def condition1 := a + b + c = 6
def condition2 := a^2 + b^2 + c^2 = 18

-- Theorem statement
theorem max_min_diff_of_c (h1 : condition1 a b c) (h2 : condition2 a b c) :
  ∃ (c_max c_min : ℝ), c_max = 6 ∧ c_min = -2 ∧ (c_max - c_min = 8) :=
by
  sorry

end max_min_diff_of_c_l107_107502


namespace value_of_m_l107_107454

def polynomial1(x : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + x - 1
def polynomial2(x : ℝ, m : ℝ) : ℝ := 3 * x^3 + 2 * m * x^2 - 5 * x + 3

theorem value_of_m (m : ℝ) :
  (∀ x : ℝ, (polynomial1 x - polynomial2 x m) ^ 2 = 0) → m = -4 :=
by
  sorry

end value_of_m_l107_107454


namespace sinusoidal_sum_equal_zero_l107_107066

theorem sinusoidal_sum_equal_zero (x : ℝ) (h : sin x ^ 2 + sin (2 * x) ^ 2 + sin (3 * x) ^ 2 + sin (4 * x) ^ 2 + sin (5 * x) ^ 2 = 5 / 2) :
  cos (6 * x) * cos (3 * x) * cos x = 0 :=
sorry

end sinusoidal_sum_equal_zero_l107_107066


namespace find_divisor_l107_107632

theorem find_divisor (d q r : ℕ) :
  (919 = d * q + r) → (q = 17) → (r = 11) → d = 53 :=
by
  sorry

end find_divisor_l107_107632


namespace smallest_prime_less_than_perfect_square_l107_107160

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107160


namespace smallest_prime_12_less_than_square_l107_107192

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107192


namespace integer_coefficient_polynomial_l107_107349

theorem integer_coefficient_polynomial {k : ℕ} (h : k ≥ 4) :
  ∀ (F : ℤ[X]),
  (∀ c ∈ Finset.range (k+2), 0 ≤ F.eval c ∧ F.eval c ≤ k) →
  (∀ c1 c2 ∈ Finset.range (k+2), F.eval c1 = F.eval c2) :=
by sorry

end integer_coefficient_polynomial_l107_107349


namespace totalCoatsCollected_l107_107558

-- Definitions from the conditions
def highSchoolCoats : Nat := 6922
def elementarySchoolCoats : Nat := 2515

-- Theorem that proves the total number of coats collected
theorem totalCoatsCollected : highSchoolCoats + elementarySchoolCoats = 9437 := by
  sorry

end totalCoatsCollected_l107_107558


namespace max_lessons_possible_l107_107591

theorem max_lessons_possible 
  (s p b : ℕ) 
  (h1 : 2 * p * b = 36) 
  (h2 : 2 * s * b = 72) 
  (h3 : 2 * s * p = 54) 
  : 2 * s * p * b = 216 :=
begin
  sorry
end

end max_lessons_possible_l107_107591


namespace overall_average_speed_is_six_l107_107336

-- Definitions of the conditions
def cycling_time := 45 / 60 -- hours
def cycling_speed := 12 -- mph
def stopping_time := 15 / 60 -- hours
def walking_time := 75 / 60 -- hours
def walking_speed := 3 -- mph

-- Problem statement: Proving that the overall average speed is 6 mph
theorem overall_average_speed_is_six : 
  (cycling_speed * cycling_time + walking_speed * walking_time) /
  (cycling_time + walking_time + stopping_time) = 6 :=
by
  sorry

end overall_average_speed_is_six_l107_107336


namespace min_value_of_polynomial_l107_107113

theorem min_value_of_polynomial (a : ℝ) : 
  (∀ x : ℝ, (2 * x^3 - 3 * x^2 + a) ≥ 5) → a = 6 :=
by
  sorry   -- Proof omitted

end min_value_of_polynomial_l107_107113


namespace find_v1_l107_107043

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := y^2 + 4 * y - 5

theorem find_v1 : v 1 = 11.25 := by
  sorry

end find_v1_l107_107043


namespace longest_arith_prime_seq_l107_107348

open Nat

def is_arith_seq (seq : List ℕ) (d : ℕ) : Prop :=
  ∀ i, i < seq.length - 1 → seq[i+1] = seq[i] + d

def is_prime_seq (seq : List ℕ) : Prop :=
  ∀ p ∈ seq, Prime p

theorem longest_arith_prime_seq :
  ∃ (seq : List ℕ), is_arith_seq seq 6 ∧ is_prime_seq seq ∧
  seq = [5, 11, 17, 23, 29] ∧ ∀ (s : List ℕ), is_arith_seq s 6 ∧ is_prime_seq s → s.length ≤ seq.length := by
  sorry

end longest_arith_prime_seq_l107_107348


namespace price_per_foot_of_fencing_l107_107655

theorem price_per_foot_of_fencing (A C : ℝ) 
  (hA : A = 289)
  (hC : C = 3808) : 
  let side_length := Real.sqrt A in
  let perimeter := 4 * side_length in
  let price_per_foot := C / perimeter in 
  price_per_foot = 56 :=
by 
  -- The proof goes here, but we skip it with sorry
  sorry

end price_per_foot_of_fencing_l107_107655


namespace larger_number_l107_107995

/-- The difference of two numbers is 1375 and the larger divided by the smaller gives a quotient of 6 and a remainder of 15. 
Prove that the larger number is 1647. -/
theorem larger_number (L S : ℕ) 
  (h1 : L - S = 1375) 
  (h2 : L = 6 * S + 15) : 
  L = 1647 := 
sorry

end larger_number_l107_107995


namespace proof_problem_l107_107786

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l107_107786


namespace value_is_20_l107_107256

-- Define the conditions
def number : ℕ := 5
def value := number + 3 * number

-- State the theorem
theorem value_is_20 : value = 20 := by
  -- Proof goes here
  sorry

end value_is_20_l107_107256


namespace complex_numbers_count_l107_107628

def mobius (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if ∃ p : ℕ, p.prime ∧ p ^ 2 ∣ n then 0
  else (-1) ^ nat.factorization.support.card n

def count_special_roots (m : ℕ) : ℕ :=
  if m = 0 then 0 else
  let sum_mobius : ℤ := ∑ k in (nat.divisors 1989), (mobius k) * (m ^ (1989 / k)) in
  sum_mobius.to_nat

theorem complex_numbers_count (m : ℕ) (h : m > 0) :
  count_special_roots m =
  m ^ 1989 - m ^ 663 - m ^ 153 - m ^ 117 + m ^ 51 + m ^ 39 + m ^ 9 - m ^ 3 :=
sorry

end complex_numbers_count_l107_107628


namespace final_output_sum_l107_107331

theorem final_output_sum :
  let S := 0 in
  let step := 2 in
  let n := 100 in
  let range := list.range (n // step) |>.map (λ i, 1 + i * step) in
  let output := range.foldl (λ acc x, acc + x^2) S in
  output = (list.range (n // step) |>.map (λ i, (1 + i * 2)^2)).sum :=
by sorry

end final_output_sum_l107_107331


namespace sum_of_sequence_l107_107120

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else a (n - 1) + n

noncomputable def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / a (k + 1))

theorem sum_of_sequence (n : ℕ) (h : 0 < n) :
  S n = 2 * n / (n + 1) :=
sorry

end sum_of_sequence_l107_107120


namespace a5_div_b5_l107_107603

noncomputable def a (n : ℕ) : ℚ := sorry -- Define the sequence {a_n}
noncomputable def b (n : ℕ) : ℚ := sorry -- Define the sequence {b_n}

axiom seq_condition : ∀ n > 0, (∑ k in finset.range n, a (k+1)) / (∑ k in finset.range n, b (k+1)) = (7 * n + 1) / (n + 2)

theorem a5_div_b5 : a 5 / b 5 = 64 / 11 := by
  sorry

end a5_div_b5_l107_107603


namespace linear_term_coefficient_l107_107367

-- Define the polynomial
def polynomial : ℕ → ℤ
| 2 := 1  -- Coefficient of x^2
| 1 := -5 -- Coefficient of x
| 0 := -6 -- Constant term
| _ := 0  -- All other terms are zero

-- Statement to prove: The coefficient of the linear term is -5.
theorem linear_term_coefficient : polynomial 1 = -5 := by
  -- Proof to be filled in later
  sorry

end linear_term_coefficient_l107_107367


namespace smallest_prime_12_less_perfect_square_l107_107142

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107142


namespace num_non_similar_regular_1000_pointed_stars_l107_107437

theorem num_non_similar_regular_1000_pointed_stars :
  let n := 1000 in
  let phi := (n * (1 - 1/2) * (1 - 1/5)) in
  phi / 2 = 200 :=
by
  let n := 1000
  let phi := (n * (1 - 1/2) * (1 - 1/5))
  have h_phi : phi = 400 := by sorry
  have h_total : phi / 2 = 200 := by sorry
  exact h_total

end num_non_similar_regular_1000_pointed_stars_l107_107437


namespace area_quadrilateral_equal_area_triangle_l107_107507

variables (A B C P O1 O2 : Type)
variables [Triangle A B C] (angle_BAC angle_ABC angle_CPB : ℝ)
variables [AngleNotEq angle_BAC 45]
variables [AngleNotEq angle_ABC 135]
variables (pointPOnAB : PointOnLine P AB)
variables [Circumcenter O1 (triangle ACP)]
variables [Circumcenter O2 (triangle BCP)]
variables (quadrilateral CO1PO2 : Quadrilateral CO1 P O2 P)

theorem area_quadrilateral_equal_area_triangle :
  area CO1PO2 = area ABC := 
sorry

end area_quadrilateral_equal_area_triangle_l107_107507


namespace cyclic_quadrilateral_sides_equal_l107_107074

theorem cyclic_quadrilateral_sides_equal
  (A B C D P : ℝ) -- Points represented as reals for simplicity
  (AB CD BC AD : ℝ) -- Lengths of sides AB, CD, BC, AD
  (a b c d e θ : ℝ) -- Various lengths and angle as given in the solution
  (h1 : a + e = b + c + d)
  (h2 : (1 / 2) * a * e * Real.sin θ = (1 / 2) * b * e * Real.sin θ + (1 / 2) * c * d * Real.sin θ) :
  c = e ∨ d = e := sorry

end cyclic_quadrilateral_sides_equal_l107_107074


namespace median_equal_range_not_greater_l107_107759

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l107_107759


namespace solve_b_l107_107484

noncomputable def x : ℂ := sorry
def b := 2 + ∑ k in finset.range 1990, x ^ k

theorem solve_b (h : 1 + x + x^2 + x^3 + x^4 = 0) : b = 2 :=
sorry

end solve_b_l107_107484


namespace circle_center_l107_107989

theorem circle_center 
    (x y : ℝ)
    (h : x^2 + y^2 - 4 * x + 6 * y = 0) :
    (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = (x^2 - 4*x + 4) + (y^2 + 6*y + 9) 
    → (x, y) = (2, -3)) :=
sorry

end circle_center_l107_107989


namespace isosceles_triangle_l107_107006

-- Problem 1: Prove that ΔABC is isosceles
theorem isosceles_triangle
  (A B C : Type)
  (a b c : ℝ)
  (h1 : c < 2 * a)
  (h2 : 3 * (sin C)^2 + 8 * (sin A)^2 = 11 * (sin A) * (sin C)) :
  a = c :=
sorry

-- Problem 2: Find the length of the median to side BC
noncomputable def median_length
  (A B C : Type)
  (a b c median_length : ℝ)
  (h3 : a = c)
  (area : ℝ := 8 * (sqrt 15))
  (sin_B : ℝ := (sqrt 15) / 4)
  (cos_B := if (sin_B)^2 <= 1 then ± sqrt (1 - (sin_B)^2) else 0) :
  median_length = 8 ∨ median_length = 4 * sqrt 6 :=
sorry

end isosceles_triangle_l107_107006


namespace find_f_2016_l107_107510

noncomputable def f : ℕ → ℕ → ℚ :=
  λ n x, if 1 ≤ x ∧ x ≤ n + 1 then 1 / (x : ℚ) else
    ∑ k in finset.range (n + 2), (x - k) / (x * 2015.factorial) + 1 / (x : ℚ)

theorem find_f_2016 : 
  let f := f 2014 in
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2015 → f k = 1 / k) → f 2016 = 1 / 1008 :=
by
  intros h
  sorry

end find_f_2016_l107_107510


namespace triangle_heights_inverse_proportional_l107_107541

theorem triangle_heights_inverse_proportional (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] (AK BL : ℝ) (BC AC : ℝ) (h₁ : ∆ABC) (h₂ : AK = h₁.height) (h₃ : BL = h₁.height) (h₄ : AC = k * BC) :
  AK = BL / k :=
by {
  sorry
}

end triangle_heights_inverse_proportional_l107_107541


namespace smallest_prime_less_than_perfect_square_l107_107163

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107163


namespace choose_2_out_of_8_l107_107018

def n : ℕ := 8
def k : ℕ := 2

theorem choose_2_out_of_8 : choose n k = 28 :=
by
  simp [n, k]
  sorry

end choose_2_out_of_8_l107_107018


namespace median_equality_and_range_inequality_l107_107769

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l107_107769


namespace fixed_point_of_function_l107_107575

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (1, 2) ∈ set_of (λ (p : ℝ × ℝ), p.snd = a^(p.fst - 1) + 1) :=
by {
  -- the proof will be written here
  sorry
}

end fixed_point_of_function_l107_107575


namespace smallest_prime_12_less_than_perfect_square_l107_107210

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107210


namespace smallest_prime_less_than_perfect_square_l107_107162

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107162


namespace limit_ratio_areas_l107_107009

-- Define the basic geometric properties and conditions.
variables (r k : ℝ) (O T U V : ℝ) (PQ RS : ℝ) (OT : ℝ)
hypothesis (h1 : OT = r - 2 * k)
hypothesis (h2 : PQ = RS)
hypothesis (h3 : T = (RS / 2))
hypothesis (h4 : (UV = VT))

-- Define the areas of the trapezoid and the rectangle.
noncomputable def area_of_rectangle (QW U V : ℝ) :=
  k * real.sqrt(r ^ 2 - (r - k) ^ 2)

noncomputable def area_of_trapezoid (QW TS : ℝ) :=
  (k / 2) * (real.sqrt(r ^ 2 - (r - k) ^ 2) + real.sqrt(r ^ 2 - (r - 2 * k) ^ 2))

-- Define the limit of the ratio of the areas as k approaches 0.
theorem limit_ratio_areas : 
  (∀ k : ℝ, ∃ ratio : ℝ, 
    (ratio = ((k / 2) * (real.sqrt(r ^ 2 - (r - k) ^ 2) + real.sqrt(r ^ 2 - (r - 2 * k) ^ 2))) /
             (k * real.sqrt(r ^ 2 - (r - k) ^ 2))) ∧ 
    (tendsto ratio (𝓝 0) (𝓝 (1 / 2 + 1 / real.sqrt 2)))
  ) := sorry

end limit_ratio_areas_l107_107009


namespace alice_walks_distance_l107_107281

theorem alice_walks_distance :
  let blocks_south := 5
  let blocks_west := 8
  let distance_per_block := 1 / 4
  let total_blocks := blocks_south + blocks_west
  let total_distance := total_blocks * distance_per_block
  total_distance = 3.25 :=
by
  sorry

end alice_walks_distance_l107_107281


namespace problem_2014_minus_4102_l107_107612

theorem problem_2014_minus_4102 : 2014 - 4102 = -2088 := 
by
  -- The proof is omitted as per the requirement
  sorry

end problem_2014_minus_4102_l107_107612


namespace arithmetic_sequence_m_value_l107_107894

theorem arithmetic_sequence_m_value {a : ℕ → ℤ} (d : ℤ) (h : d ≠ 0)
  (h_a : ∀ n, a n = a 1 + (n - 1) * d) : 
  (a 1 = 0) → 
  a 191 = a 1 + a 2 + a 3 + ⋯ + a 20 :=
by
  intro a1_eq_0
  sorry

end arithmetic_sequence_m_value_l107_107894


namespace inequality_holds_l107_107049

theorem inequality_holds (a b c : ℝ) (h : sqrt (a^2 + b^2) < c) : ∀ x : ℝ, a * sin x + b * cos x + c > 0 :=
by
  sorry

end inequality_holds_l107_107049


namespace symmetric_matrix_five_ones_l107_107729

theorem symmetric_matrix_five_ones :
  let n := 5 in
  let total_count : ℕ := 
    ((choose n 1) * (choose (n * (n - 1) / 2) 2)) +
    ((choose n 3) * (choose (n * (n - 1) / 2) 1)) +
    ((choose n 5) * (choose (n * (n - 1) / 2) 0))
  in total_count = 326 :=
by sorry

end symmetric_matrix_five_ones_l107_107729


namespace circle_integer_points_l107_107660

theorem circle_integer_points (m n : ℤ) (h : ∃ m n : ℤ, m^2 + n^2 = r ∧ 
  ∃ p q : ℤ, m^2 + n^2 = p ∧ ∃ s t : ℤ, m^2 + n^2 = q ∧ ∃ u v : ℤ, m^2 + n^2 = s ∧ 
  ∃ j k : ℤ, m^2 + n^2 = t ∧ ∃ l w : ℤ, m^2 + n^2 = u ∧ ∃ x y : ℤ, m^2 + n^2 = v ∧ 
  ∃ i b : ℤ, m^2 + n^2 = w ∧ ∃ c d : ℤ, m^2 + n^2 = b ) :
  ∃ r, r = 25 := by
    sorry

end circle_integer_points_l107_107660


namespace total_legs_correct_l107_107276

-- Define the number of animals
def num_dogs : ℕ := 2
def num_chickens : ℕ := 1

-- Define the number of legs per animal
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

-- Define the total number of legs from dogs and chickens
def total_legs : ℕ := num_dogs * legs_per_dog + num_chickens * legs_per_chicken

theorem total_legs_correct : total_legs = 10 :=
by
  -- this is where the proof would go, but we add sorry for now to skip it
  sorry

end total_legs_correct_l107_107276


namespace average_marks_first_class_l107_107102

variable (a : ℝ) -- Define the average marks of the first class
variable (total_marks : ℝ) -- Define total marks

-- Conditions
def num_students_first_class := 24
def avg_marks_second_class := 60
def num_students_second_class := 50
def combined_avg_marks := 53.513513513513516

-- Proof problem
theorem average_marks_first_class :
  let total_students := num_students_first_class + num_students_second_class in
  let total_marks_second_class := avg_marks_second_class * num_students_second_class in
  let total_combined_marks := combined_avg_marks * total_students in
  total_marks_second_class + (a * num_students_first_class) = total_combined_marks →
  a = 40.04 :=
  by
  sorry

end average_marks_first_class_l107_107102


namespace min_fence_dimensions_l107_107063

noncomputable def minimum_perimeter (l w : ℝ) (h_area : l * w ≥ 600) : ℝ :=
2 * l + 2 * w

theorem min_fence_dimensions :
  ∃ l w : ℝ, l = w ∧ l * w ≥ 600 ∧ minimum_perimeter l w (by linarith) = 40 * Real.sqrt 6 :=
begin
  use [10 * Real.sqrt 6, 10 * Real.sqrt 6],
  exact ⟨rfl, by linarith [Real.sqrt_nonneg 6], by linarith⟩,
end

end min_fence_dimensions_l107_107063


namespace total_rent_of_pasture_l107_107228

def ox_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

noncomputable def cost_per_ox_month (c_rent : ℝ) (c_ox_months : ℕ) : ℝ := c_rent / c_ox_months

theorem total_rent_of_pasture :
  let a_ox_months := ox_months 10 7 in
  let b_ox_months := ox_months 12 5 in
  let c_ox_months := ox_months 15 3 in
  let c_share := 53.99999999999999 in -- rounded to 54 in practice
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months in
  let cost_per_month := cost_per_ox_month c_share c_ox_months in
  let total_rent := total_ox_months * cost_per_month in
  total_rent = 210 :=
by
  -- Here would come the proof steps, but they are skipped as per instructions
  sorry

end total_rent_of_pasture_l107_107228


namespace proof_problem_l107_107119

-- Definitions coming from the conditions
def num_large_divisions := 12
def num_small_divisions_per_large := 5
def seconds_per_small_division := 1
def seconds_per_large_division := num_small_divisions_per_large * seconds_per_small_division
def start_position := 5
def end_position := 9
def divisions_moved := end_position - start_position
def total_seconds_actual := divisions_moved * seconds_per_large_division
def total_seconds_claimed := 4

-- The theorem stating the false claim
theorem proof_problem : total_seconds_actual ≠ total_seconds_claimed :=
by {
  -- We skip the actual proof as instructed
  sorry
}

end proof_problem_l107_107119


namespace ratio_of_boys_to_girls_l107_107231

-- Definitions directly from the conditions
variables (B G : ℕ)
axiom avg_boys : ℕ := 90
axiom avg_girls : ℕ := 96
axiom overall_avg : ℕ := 94

-- Main theorem statement
theorem ratio_of_boys_to_girls 
    (hb : avg_boys * B = 90 * B)
    (hg : avg_girls * G = 96 * G)
    (ha : overall_avg * (B + G) = 90 * B + 96 * G) : 
    2 * B = G := 
sorry

end ratio_of_boys_to_girls_l107_107231


namespace sphere_surface_area_is_36pi_l107_107002

-- Define the edge length of the cube
def edge_length : ℝ := 2 * Real.sqrt 3

-- Define the diagonal of the cube
def cube_diagonal : ℝ := edge_length * Real.sqrt 3

-- Define the radius of the sphere circumscribing the cube
def sphere_radius : ℝ := cube_diagonal / 2

-- Define the surface area of the sphere
noncomputable def sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius^2

-- Prove that the surface area of the sphere is 36π
theorem sphere_surface_area_is_36pi : sphere_surface_area = 36 * Real.pi := by
  sorry

end sphere_surface_area_is_36pi_l107_107002


namespace proof_problem_l107_107781

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l107_107781


namespace gillian_spending_l107_107547

theorem gillian_spending (sandi_initial : ℕ) (sandi_fraction : ℕ) (gillian_extra : ℕ) (sandi_final : ℕ) (gillian_total : ℕ) :
  sandi_initial = 600 → sandi_fraction = 2 → gillian_extra = 150 →
  sandi_final = sandi_initial / sandi_fraction →
  gillian_total = 3 * sandi_final + gillian_extra →
  gillian_total = 1050 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end gillian_spending_l107_107547


namespace total_students_in_college_l107_107631

theorem total_students_in_college 
  (girls : ℕ) 
  (ratio_boys : ℕ) 
  (ratio_girls : ℕ) 
  (h_ratio : ratio_boys = 8) 
  (h_ratio_girls : ratio_girls = 5) 
  (h_girls : girls = 400) 
  : (ratio_boys * (girls / ratio_girls) + girls = 1040) := 
by 
  sorry

end total_students_in_college_l107_107631


namespace at_least_nine_empty_cells_l107_107536

theorem at_least_nine_empty_cells (board : Fin 9 × Fin 9 → Bool) :
  (∀ (i j : Fin 9), (∃ (di dj : ℤ), di ∈ {-1, 1} ∧ dj ∈ {-1, 1} ∧ board (i + di, j + dj))) →
  ∃ (empty_cells : Finset (Fin 9 × Fin 9)), empty_cells.card ≥ 9 :=
begin
  -- This proof is omitted, as per the instructions.
  sorry,
end

end at_least_nine_empty_cells_l107_107536


namespace tangents_of_circle_C_through_D_max_area_triangle_PMN_l107_107376

open Real

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := { p | (p.1 - 2) ^ 2 + p.2 ^ 2 = 9 }

-- Define the point D
def point_D : ℝ × ℝ := (-1, 1)

-- Define the line l2
def line_l2 (p : ℝ × ℝ) : Prop := p.1 + sqrt 3 * p.2 - 1 = 0

-- Statement for problem part 1
theorem tangents_of_circle_C_through_D :
  ∃ l₁ : Set (ℝ × ℝ), (l₁ = {p | p.1 = -1} ∨ l₁ = {p | 4 * p.1 - 3 * p.2 + 7 = 0})
  ∧ tangent_to l₁ circle_C point_D := sorry

-- Statement for problem part 2
theorem max_area_triangle_PMN :
  ∀ P ∈ circle_C,
  ∀ M N : ℝ × ℝ, line_l2 M ∧ line_l2 N ∧ M ≠ N ∧ (M.1 - 2)^2 + M.2^2 ∈ {9} ∧ (N.1 - 2)^2 + N.2^2 ∈ {9} → 
  area_of_triangle P M N ≤ (7 * sqrt 35) / 4 := sorry

end tangents_of_circle_C_through_D_max_area_triangle_PMN_l107_107376


namespace fraction_of_displayed_pieces_is_one_third_l107_107676

-- Definitions for clarity
variables (T D N : ℕ) -- total pieces, displayed pieces, not displayed pieces

-- Conditions from the problem.
def gallery_conditions := 
  T = 900 ∧
  (2 / 3 : ℝ) * N = 400 ∧
  N + D = T ∧
  D = T - N ∧
  N = 600   -- Calculated from the given conditions

-- Goal: The fraction of displayed pieces of art
def fraction_displayed (D T : ℕ) := (D : ℝ) / (T : ℝ)

-- The main theorem to be proved
theorem fraction_of_displayed_pieces_is_one_third :
  gallery_conditions T D N →
  fraction_displayed D T = 1 / 3 :=
by
  intros h
  sorry

end fraction_of_displayed_pieces_is_one_third_l107_107676


namespace vectors_inequality_and_angle_l107_107404

open Real

variables {a b : EuclideanSpace ℝ (Fin 3)} 

-- Define the given conditions
def not_collinear (a b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ λ:ℝ, a ≠ λ • b ∧ b ≠ λ • a 

def has_maximum_value (f : ℝ → ℝ) : Prop :=
  ∃ x₀ ∈ (0, +∞), ∀ x ∈ (0, +∞), f x ≤ f x₀

def f (x : ℝ) (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  (x • a + b) ⬝ (a - x • b)

-- Rewrite the mathematical proof problem as a Lean 4 theorem statement
theorem vectors_inequality_and_angle {a b : EuclideanSpace ℝ (Fin 3)}
  (h1: not_collinear a b)
  (h2: has_maximum_value (λ x, f x a b)) :
  |a| > |b| ∧ ∀ θ : ℝ, θ ∈ Set.Ioo 0 (π/2) → 
  θ = Real.arccos ((a ⬝ b) / (|a| * |b|)) :=
sorry

end vectors_inequality_and_angle_l107_107404


namespace median_equal_range_not_greater_l107_107757

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l107_107757


namespace nat_add_ge_3_implies_at_least_one_ge_2_l107_107609

theorem nat_add_ge_3_implies_at_least_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by {
  sorry
}

end nat_add_ge_3_implies_at_least_one_ge_2_l107_107609


namespace median_equality_range_inequality_l107_107797

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l107_107797


namespace complex_arithmetic_l107_107037

-- Defining the given complex numbers and real value
def A : ℂ := 5 - 2 * Complex.I
def M : ℂ := -3 + 3 * Complex.I
def S : ℂ := 2 * Complex.I
def P : ℝ := 1 / 2

-- Stating the theorem we want to prove
theorem complex_arithmetic : (A - M + S - P : ℂ) = 7.5 - 3 * Complex.I := by
  sorry

end complex_arithmetic_l107_107037


namespace median_equality_range_inequality_l107_107751

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l107_107751


namespace find_B_find_perimeter_l107_107027

variables {a b c : ℝ} {A B C : ℝ}

-- Conditions
axiom triangle_condition_1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B
axiom b_value : b = 2 * Real.sqrt 3
axiom area_of_triangle : 1/2 * a * c * Real.sin B = 2 * Real.sqrt 3

-- Goals
theorem find_B (h1 : triangle_condition_1) (h2 : b_value) : B = Real.pi / 3 :=
sorry

theorem find_perimeter (h1 : triangle_condition_1) (h2 : b_value) (h3 : area_of_triangle) : a + b + c = 6 + 2 * Real.sqrt 3 :=
sorry

end find_B_find_perimeter_l107_107027


namespace prove_square_distance_from_b_to_center_l107_107658

noncomputable def square_distance_from_b_to_center : Prop :=
  let r := 10
  let a := 10
  let b := 2
  radius_sq = r^2 ∧ 
  (angle_eq_right (10, 2) ∧ distance ((10, 2), (10, 2+8)) = 8 ∧ distance ((10, 2), (10+4, 2)) = 4) →
  (a^2 + b^2 = 104)

theorem prove_square_distance_from_b_to_center : square_distance_from_b_to_center := by
  sorry

end prove_square_distance_from_b_to_center_l107_107658


namespace find_a_value_l107_107406

noncomputable def binomial_expression_consts (a : ℝ) : Prop :=
  let x := 1 in
  let n := 5 in
  (1 + a)^n = 32 ∧ 
  (nat.choose 5 2 * a^2 : ℝ) = 80

theorem find_a_value : ∃ a : ℝ, binomial_expression_consts a ∧ a = 2 * real.sqrt 2 :=
begin
  sorry
end

end find_a_value_l107_107406


namespace binomial_600_600_l107_107314

-- Define a theorem to state the binomial coefficient property and use it to prove the specific case.
theorem binomial_600_600 : nat.choose 600 600 = 1 :=
begin
  -- Binomial property: for any non-negative integer n, (n choose n) = 1
  rw nat.choose_self,
end

end binomial_600_600_l107_107314


namespace sum_of_digits_of_y_coordinate_of_C_l107_107930

theorem sum_of_digits_of_y_coordinate_of_C 
  (a b : ℝ) 
  (ha_ne_hb : a ≠ b) 
  (area_ABC : 1 / 2 * abs ((b - a) * (y - (a^3 + b^3) / 2)) = 1000) :
  (digits_sum : (abs (y : ℝ)).nat_digits.sum) = 18 :=
sorry

end sum_of_digits_of_y_coordinate_of_C_l107_107930


namespace binom_600_eq_1_l107_107319

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l107_107319


namespace pythagorean_numbers_l107_107070

theorem pythagorean_numbers (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  (a = 3 ∧ b = 4 ∧ c = 5) ↔ 
  "Numbers like 3, 4, 5 that can form a right-angled triangle are called Pythagorean numbers" := 
sorry

end pythagorean_numbers_l107_107070


namespace track_radius_l107_107468

-- Definitions of the given conditions
def TP : ℝ := 6
def TʹQ : ℝ := 12
def TʹʹQ : ℝ := 10

-- Distance calculation based on given conditions
def PQ := TP + TʹʹQ
def RQ := TʹQ - TP
def PR := real.sqrt (PQ ^ 2 - RQ ^ 2)
def radius := PR / 2

-- Theorem stating that the radius is 7.4165 meters
theorem track_radius : radius = 7.4165 := by
  sorry

end track_radius_l107_107468


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107788

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107788


namespace area_equals_circumradius_times_semiperimeter_l107_107951

universe u

variables {A B C: Type u} [euclidean_space A B C]
variables (A_1 B_1 C_1: Type u) [euclidean_space A_1 B_1 C_1]

def area (t: ℝ) (A B C: Point) [is_acute_triangle A B C]: ℝ := sorry

def circumradius (R: ℝ) (A B C: Point) [is_acute_triangle A B C]: ℝ := sorry

def semiperimeter (s': ℝ) (A_1 B_1 C_1: FootPoint): ℝ := sorry

theorem area_equals_circumradius_times_semiperimeter (A B C: Point) 
  (A_1 B_1 C_1: FootPoint) 
  [is_acute_triangle A B C]
  (t R s': ℝ) 
  (t_eq: t = area t A B C)
  (R_eq: R = circumradius R A B C)
  (s'_eq: s' = semiperimeter s' A_1 B_1 C_1):
  t = R * s' :=
sorry

end area_equals_circumradius_times_semiperimeter_l107_107951


namespace probability_no_shaded_rectangle_l107_107647

theorem probability_no_shaded_rectangle :
  let n := 2003
  let total_rectangles := ((n + 1) * (n + 1) * (n + 1) * (n + 1)) / 4
  let shaded_rectangles := ((n / 2 + 1) * (n / 2 + 1) * (n / 2 + 1) * (n / 2 + 1))
  let p := 1 - (shaded_rectangles / total_rectangles)
  p = (Rat.ofInt 1001) / (Rat.ofInt 2003) :=
by
  sorry

end probability_no_shaded_rectangle_l107_107647


namespace eggs_distribution_l107_107653

theorem eggs_distribution
  (total_eggs : ℕ)
  (eggs_per_adult : ℕ)
  (num_adults : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (eggs_per_girl : ℕ)
  (total_eggs_def : total_eggs = 3 * 12)
  (eggs_per_adult_def : eggs_per_adult = 3)
  (num_adults_def : num_adults = 3)
  (num_girls_def : num_girls = 7)
  (num_boys_def : num_boys = 10)
  (eggs_per_girl_def : eggs_per_girl = 1) :
  ∃ eggs_per_boy : ℕ, eggs_per_boy - eggs_per_girl = 1 :=
by {
  sorry
}

end eggs_distribution_l107_107653


namespace total_spent_l107_107522

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end total_spent_l107_107522


namespace area_of_quadrilateral_l107_107645

theorem area_of_quadrilateral 
  (side_length : ℝ)
  (midpoint1 midpoint2 center midpoint3 : ℝ × ℝ)
  (h_square : side_length = 10)
  (h_midpoint1 : midpoint1 = (5, 0))
  (h_midpoint2 : midpoint2 = (0, 5))
  (h_center : center = (5, 5))
  (h_midpoint3 : midpoint3 = (10, 5))
  : 
  let quadrilateral_area := (1 / 2) * (5 * 5) + (1 / 2) * (5 * 5) in
  quadrilateral_area = 25 :=
by
  sorry

end area_of_quadrilateral_l107_107645


namespace minimal_range_of_sample_l107_107266

theorem minimal_range_of_sample (x1 x2 x3 x4 x5 : ℝ) 
  (mean_condition : (x1 + x2 + x3 + x4 + x5) / 5 = 6) 
  (median_condition : x3 = 10) 
  (sample_order : x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5) : 
  (x5 - x1) = 10 :=
sorry

end minimal_range_of_sample_l107_107266


namespace smallest_prime_12_less_than_perfect_square_l107_107213

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107213


namespace cost_of_6_boxes_l107_107625

theorem cost_of_6_boxes (cost_per_4_boxes : ℝ) (cost_4_boxes : cost_per_4_boxes = 26) : 
  ∃ cost_6_boxes, cost_6_boxes = 6 / 4 * cost_per_4_boxes ∧ cost_6_boxes = 39 :=
by
  use (6 / 4) * cost_per_4_boxes
  constructor
  . rw [cost_4_boxes]
  . rfl

end cost_of_6_boxes_l107_107625


namespace number_of_subsets_M_l107_107738

-- Define the imaginary unit with its primary property
def imaginary_unit : ℂ := Complex.i

-- Define x as given in the problem
def x (n : ℕ) := (imaginary_unit ^ n) + (imaginary_unit ^ -(n : ℤ))

-- Define the set S of all possible values of x
def S : Set ℂ := { x n | ∃ n : ℕ, x = (imaginary_unit ^ n) + (imaginary_unit ^ -↑n) }

-- Calculate the number of sets M within S
theorem number_of_subsets_M : (Set ℂ) := 
  let values := { 2, 0, -2 }
  have card_values : values.card = 3 := by sorry 
  show card (set.powerset values) = 2^3 := sorry

end number_of_subsets_M_l107_107738


namespace functional_relationship_minimum_wage_l107_107957

/-- Problem setup and conditions --/
def total_area : ℝ := 1200
def team_A_rate : ℝ := 100
def team_B_rate : ℝ := 50
def team_A_wage : ℝ := 4000
def team_B_wage : ℝ := 3000
def min_days_A : ℝ := 3

/-- Prove Part 1: y as a function of x --/
def y_of_x (x : ℝ) : ℝ := 24 - 2 * x

theorem functional_relationship (x : ℝ) :
  100 * x + 50 * y_of_x x = total_area := by
  sorry

/-- Prove Part 2: Minimum wage calculation --/
def total_wage (a b : ℝ) : ℝ := team_A_wage * a + team_B_wage * b

theorem minimum_wage :
  ∀ (a b : ℝ), 3 ≤ a → a ≤ b → b = 24 - 2 * a → 
  total_wage a b = 56000 → a = 8 ∧ b = 8 := by
  sorry

end functional_relationship_minimum_wage_l107_107957


namespace find_v1_l107_107042

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := y^2 + 4 * y - 5

theorem find_v1 : v 1 = 11.25 := by
  sorry

end find_v1_l107_107042


namespace binomial_600_600_l107_107311

-- Define a theorem to state the binomial coefficient property and use it to prove the specific case.
theorem binomial_600_600 : nat.choose 600 600 = 1 :=
begin
  -- Binomial property: for any non-negative integer n, (n choose n) = 1
  rw nat.choose_self,
end

end binomial_600_600_l107_107311


namespace floor_neg_2_point_8_l107_107345

theorem floor_neg_2_point_8 : floor (-2.8) = -3 := 
  sorry

end floor_neg_2_point_8_l107_107345


namespace smallest_prime_less_than_perfect_square_l107_107164

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107164


namespace weight_of_new_person_l107_107633

-- Define the relevant quantities and conditions
variables (W x : ℝ)
noncomputable def avg_weight_before := W / 15
noncomputable def avg_weight_after := (W - 75 + x) / 15

-- The condition given in the problem
axiom average_increase_condition : avg_weight_after W x = avg_weight_before W + 3.2

-- The goal is to prove that x = 123
theorem weight_of_new_person : x = 123 :=
by
  have eq1 : (W - 75 + x) / 15 = avg_weight_before W + 3.2 := average_increase_condition
  have eq2 : avg_weight_before W = W / 15 := rfl
  have eq3 : (W - 75 + x) / 15 = W / 15 + 3.2 := by rw [eq1, eq2]
  sorry  

end weight_of_new_person_l107_107633


namespace _l107_107408

noncomputable theorem locus_of_focus_of_parabola : 
  ∀ (a b x y : ℝ), 
  (a^2 + b^2 = 4) ∧ 
  (x ≠ 0) ∧ 
  (x^2 + (y - 1)^2 = (a * x + b * y - 4)^2 / 4) ∧ 
  (x^2 + (y + 1)^2 = (a * x + b * y + 4)^2 / 4) 
  → ( x^2 / 3 + y^2 / 4 = 1 ) := 
by 
  sorry

end _l107_107408


namespace symmetric_lines_inv_functions_l107_107457

-- Defining the conditions
def f(x : ℝ) : ℝ := a * x + 2
def g(x : ℝ) : ℝ := 3 * x + b
def symmetry (x : ℝ) : ℝ := x

-- Main theorem statement
theorem symmetric_lines_inv_functions (a b : ℝ) (H : ∀ x : ℝ, f (symmetry x) = symmetry (g x)) :
  a = (1 : ℝ) / 3 ∧ b = -6 :=
sorry

end symmetric_lines_inv_functions_l107_107457


namespace sum_first_10_terms_arithmetic_sequence_l107_107098

noncomputable def a_n : ℕ → ℝ := sorry -- The definition of the arithmetic sequence, typically a_n = a₁ + (n-1)d

theorem sum_first_10_terms_arithmetic_sequence
  (a_3 a_8 : ℝ)
  (h₁ : a_3 = 5)
  (h₂ : a_8 = 11) :
  let a_1 := a_n 1,
      a_10 := a_n 10,
      S_10 := 10 / 2 * (a_3 + a_10)
  in S_10 = 80 :=
by
  -- The proof will go here, skipped with sorry
  sorry

end sum_first_10_terms_arithmetic_sequence_l107_107098


namespace math_question_l107_107774

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l107_107774


namespace median_equality_and_range_inequality_l107_107768

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l107_107768


namespace probability_odd_sum_row_col_l107_107114

theorem probability_odd_sum_row_col (numGrid: List (List ℕ)) (hUnique: ∀ (x y : ℕ), x ≠ y → x ∉ List.flatten numGrid → y ∉ List.flatten numGrid) :
  (∃ (grid: List (List ℕ)), (∀ i j, 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 → grid[i][j] ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) 
  ∧ (∀ row, 1 ≤ row ∧ row ≤ 4 → List.sum (grid[row]) % 2 = 1)
  ∧ (∀ col, 1 ≤ col ∧ col ≤ 4 → List.sum (List.map (λ row, grid[row][col]) (List.range 4)) % 2 = 1)) → false := 
by 
  sorry

end probability_odd_sum_row_col_l107_107114


namespace combined_weight_of_olivers_bags_l107_107942

theorem combined_weight_of_olivers_bags (w_james : ℕ) (w_oliver : ℕ) (w_combined : ℕ) 
  (h1 : w_james = 18) 
  (h2 : w_oliver = w_james / 6) 
  (h3 : w_combined = 2 * w_oliver) : 
  w_combined = 6 := 
by
  sorry

end combined_weight_of_olivers_bags_l107_107942


namespace total_spent_l107_107520

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end total_spent_l107_107520


namespace median_equality_range_inequality_l107_107755

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l107_107755


namespace cos_sum_identity_l107_107969

noncomputable theory

open Complex

theorem cos_sum_identity :
  let ω := exp (2 * π * I / 17) in
  ω^17 = 1 →
  (∃ω_conj, ω_conj = conj ω ∧ ω_conj = 1 / ω) →
  cos (2 * π / 17) + cos (6 * π / 17) + cos (8 * π / 17) = (sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_identity_l107_107969


namespace smallest_prime_12_less_than_perfect_square_l107_107214

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107214


namespace solve_equation_l107_107096

noncomputable def equation (x : ℂ) : ℂ :=
  (x^3 - 4 * x^2 * real.sqrt 3 + 12 * x - 8 * real.sqrt 3) + (2 * x - 2 * real.sqrt 3)

theorem solve_equation :
  ∀ x : ℂ, equation x = 0 ↔ (x = 2 * real.sqrt 3 ∨ x = 2 * real.sqrt 3 + complex.I * real.sqrt 2 ∨ x = 2 * real.sqrt 3 - complex.I * real.sqrt 2) :=
by
  sorry

end solve_equation_l107_107096


namespace T6_geometric_progression_l107_107024

theorem T6_geometric_progression :
  ∀ (a : ℕ → ℝ) (r : ℝ),
    (0 < r) →
    (0 < a 3) →
    (∀ n, a (n + 1) = a n * r) →
    (real.integral (1/e) e (λ x, 1/x) = 2) →
    (a 3 * a 4 = 4) →
    (a 1 * a 6 = 4 ∧ a 2 * a 5 = 4 ∧ a 3 * a 4 = 4) →
    (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 64) :=
begin
  sorry
end

end T6_geometric_progression_l107_107024


namespace find_a_l107_107370

theorem find_a 
  (a b : ℝ)
  (h1 : real.log10 a + b = -2)
  (h2 : a^b = 10) : a = 1 / 10 :=
by 
  sorry

end find_a_l107_107370


namespace smallest_prime_12_less_than_perfect_square_l107_107218

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107218


namespace Sandy_tokens_more_than_siblings_l107_107550

theorem Sandy_tokens_more_than_siblings :
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  -- Definitions as per conditions
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  -- Conclusion
  show Sandy_tokens - sibling_tokens = 375000
  sorry

end Sandy_tokens_more_than_siblings_l107_107550


namespace difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l107_107226

-- Exploration 1
theorem difference_in_square_sides (a b : ℝ) (h1 : a + b = 20) (h2 : a^2 - b^2 = 40) : a - b = 2 :=
by sorry

-- Exploration 2
theorem square_side_length (x y : ℝ) : (2 * x + 2 * y) / 4 = (x + y) / 2 :=
by sorry

theorem square_area_greater_than_rectangle (x y : ℝ) (h : x > y) : ( (x + y) / 2 ) ^ 2 > x * y :=
by sorry

end difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l107_107226


namespace temperature_on_Monday_l107_107567

theorem temperature_on_Monday 
  (M T W Th F : ℝ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : F = 36) : 
  M = 44 := 
by 
  -- Proof omitted
  sorry

end temperature_on_Monday_l107_107567


namespace arithmetic_progression_divisors_l107_107960

theorem arithmetic_progression_divisors (d : ℕ) : 
  (∀ n : ℕ, (∃ k : ℕ, 16 + n * d = k ∧ (k.divisors.count % 5 = 0))) ∧ 
  ∃ d_min : ℕ, d = 32 :=
by
  sorry

end arithmetic_progression_divisors_l107_107960


namespace right_triangle_2014th_perimeter_l107_107697

theorem right_triangle_2014th_perimeter :
  ∃ (a b c : ℕ), ∃ (d : ℕ), d = 2014 ∧
  a < b ∧ b < c ∧
  b = a + d ∧ c = a + 2 * d ∧
  a^2 + b^2 = c^2 ∧
  12 * d = 24168 := 
begin
  sorry
end

end right_triangle_2014th_perimeter_l107_107697


namespace smallest_prime_less_than_square_l107_107152

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107152


namespace median_eq_range_le_l107_107746

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l107_107746


namespace inequality_system_solution_l107_107977

theorem inequality_system_solution (x : ℝ) : 
  (6 * x + 1 ≤ 4 * (x - 1)) ∧ (1 - x / 4 > (x + 5) / 2) → x ≤ -5/2 :=
by
  sorry

end inequality_system_solution_l107_107977


namespace area_trapezoid_l107_107641

theorem area_trapezoid (h1 : ∃ a b : ℝ, a * b = 20) (h2 : ∃ D E C F : ℝ, D = 1 ∧ F = 2 ∧ E + F = 4):
  trapezoid_area EFBA = 14 :=
by
  sorry

end area_trapezoid_l107_107641


namespace median_equality_range_inequality_l107_107796

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l107_107796


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107790

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107790


namespace v_at_one_l107_107045

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := let x := (y + 9) / 4 in x^2 + 4 * x - 5

theorem v_at_one : v 1 = 11.25 :=
by
  -- placeholder for the proof
  sorry

end v_at_one_l107_107045


namespace cube_root_of_64_is_4_l107_107992

-- Statement of the problem
theorem cube_root_of_64_is_4 : real.cbrt 64 = 4 :=
sorry

end cube_root_of_64_is_4_l107_107992


namespace regular_triangular_prism_can_be_covered_l107_107342

-- Representing the notion of a regular triangular prism and its dimensions
structure Prism :=
  (a : ℝ) -- length of the base edge
  (h : ℝ) -- length of the side edges
  (base_triangles : ℝ → Prop) -- two base faces (equilateral triangles)
  (side_rectangles : ℝ → ℝ → Prop) -- three rectangular faces

-- Define the property that faces of the prism can be tiled by equilateral triangles
def can_be_tiled_with_equilateral_triangles (P : Prism) : Prop :=
  ∃ (triangles : Set ℝ), (∀ t ∈ triangles, is_equilateral t) ∧
  (∃ (tiles : Set ℝ), (∀ tile ∈ tiles, is_equilateral tile)) ∧
  ∀ x ∈ P, x ∈ triangles ∨ x ∈ tiles

-- Given conditions:
def regular_triangular_prism_condition (P : Prism) : Prop :=
  ∀ a, (P.h = a * (real.sqrt 3)) ∧ (P.base_triangles a) ∧
  ∃ b c, (P.side_rectangles b c) ∧ b = a ∧ c = (a * real.sqrt 3)

-- Question restated as a theorem
theorem regular_triangular_prism_can_be_covered :
  ∃ P : Prism, regular_triangular_prism_condition P → (can_be_tiled_with_equilateral_triangles P) :=
sorry

end regular_triangular_prism_can_be_covered_l107_107342


namespace system_has_five_real_solutions_l107_107326

theorem system_has_five_real_solutions :
  ∃ S : Finset (ℝ × ℝ × ℝ × ℝ), S.card = 5 ∧
  ∀ ⟨x, y, z, w⟩ ∈ S,
    (x = z + w + 2 * z * w * x) ∧
    (y = w + x + w * x * y) ∧
    (z = x + y + x * y * z) ∧
    (w = y + z + 2 * y * z * w) :=
sorry

end system_has_five_real_solutions_l107_107326


namespace probability_of_distance_l107_107471

noncomputable def probability_distance_greater_than_one (P : ℝ³) (O : ℝ³) (cube_edge : ℝ) : ℝ :=
  let volume_cube := cube_edge^3 in
  let volume_hemisphere := (1/2) * (4 / 3) * π * (1^3) in
  let probability := (volume_cube - volume_hemisphere) / volume_cube in
  probability

theorem probability_of_distance (P : ℝ³) (O : ℝ³) (cube_edge : ℝ) (h_edge_length : cube_edge = 2)
  (h_O_center : O = (1, 1, 0)) : 
  probability_distance_greater_than_one P O cube_edge = 1 - π / 12 := by
  sorry

end probability_of_distance_l107_107471


namespace rhombus_area_l107_107397

theorem rhombus_area (a : ℝ) (d1 d2 : ℝ) 
  (h_diag : a = sqrt 125) 
  (h_diff : abs (d1 - d2) = 8) 
  (h_perpendicular : d1 / 2 ≤ a ∧ d2 / 2 ≤ a) 
  (h_side : (d1 / 2)^2 + (d2 / 2)^2 = a^2) :
  let area := (d1 * d2) / 2
  in area = 58.5 := 
by
  sorry

end rhombus_area_l107_107397


namespace smallest_prime_less_than_perfect_square_is_13_l107_107200

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107200


namespace scientific_notation_of_12500_l107_107071

def scientific_notation (n : ℕ) : ℝ × ℤ :=
if n = 0 then (0, 0)
else let digits := n.to_digits 10 in
     let non_zero := digits.drop_while (λ d, d = 0) in
     let len := non_zero.length in
     let scale := non_zero.reverse.foldr (λ d acc, acc / 10 + d) 0 in
     (scale, len - 1)

theorem scientific_notation_of_12500 :
  scientific_notation 12500 = (1.25, 4) :=
by
  sorry

end scientific_notation_of_12500_l107_107071


namespace fg_even_l107_107399

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Ico 0 2 then real.log (x + 1) / real.log 2 else f (x - 2)

theorem fg_even (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = f x) 
  (h2 : ∀ x, 0 ≤ x → f (x + 2) = f x) 
  (h3 : ∀ x, 0 ≤ x ∧ x < 2 → f x = real.log (x + 1) / real.log 2) : 
  f (-2010) + f (2011) = 1 := 
  sorry

end fg_even_l107_107399


namespace initial_peaches_l107_107958

-- Defining the conditions
variables (picked_peaches : ℕ) (total_peaches : ℕ)

-- Given conditions
def picked_peaches := 42
def total_peaches := 55

-- Proof statement
theorem initial_peaches : total_peaches - picked_peaches = 13 :=
by {
  -- Add the proof here
  sorry
}

end initial_peaches_l107_107958


namespace range_of_a_l107_107822

variable (a : ℝ)

def discriminant (a : ℝ) : ℝ := 4 * a ^ 2 - 12

theorem range_of_a
  (h : discriminant a > 0) :
  a < -Real.sqrt 3 ∨ a > Real.sqrt 3 :=
sorry

end range_of_a_l107_107822


namespace line_passes_through_fixed_point_l107_107863

theorem line_passes_through_fixed_point (m n : ℝ) (h : m + 2 * n - 1 = 0) :
  mx + 3 * y + n = 0 → (x, y) = (1/2, -1/6) :=
by
  sorry

end line_passes_through_fixed_point_l107_107863


namespace four_transformations_of_1989_l107_107131

-- Definition of the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Initial number
def initial_number : ℕ := 1989

-- Theorem statement
theorem four_transformations_of_1989 : 
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits initial_number))) = 9 :=
by
  sorry

end four_transformations_of_1989_l107_107131


namespace sequence_increasing_lambda_range_l107_107735

theorem sequence_increasing_lambda_range (a : ℕ → ℝ) (λ : ℝ)
  (h1 : a 1 = 1) (h2 : a 2 = 2)
  (h3 : ∀ n : ℕ, n > 0 → (n * a (n + 2) = (n + 2) * a n + λ * (n^2 + 2 * n))) :
  (∀ n : ℕ, a n ≤ a (n + 1)) ↔ λ ≥ 0 := 
sorry

end sequence_increasing_lambda_range_l107_107735


namespace prizes_inequality_l107_107474

theorem prizes_inequality :
  ∀ (x y : ℕ), 20 * x + 10 * y ≤ 200 → 3 * x ≤ y → 2 ≤ x →
  (x ≤ 4 ∧ y ≤ 16 ∧ 100 ≤ 20 * x + 10 * y ∧ (∃ (x_vals : finset ℕ) (y_vals : finset ℕ), x_vals.card = 3 ∧ y_vals.card = 11 ∨ y_vals.card = 6 ∨ y_vals.card = 1 ∧ (x_vals.product y_vals).card = 18)) :=
by
  sorry

end prizes_inequality_l107_107474


namespace graph_transform_l107_107601

theorem graph_transform (x : ℝ) :
  (√3 * sin (3 * x) + cos (3 * x)) = 2 * sin (3 * (x + π / 18)) :=
by sorry

end graph_transform_l107_107601


namespace expected_number_of_digits_l107_107685

open ProbabilityTheory

def fair_dodecahedral_die := { x : ℕ // 1 ≤ x ∧ x ≤ 12 }
def num_digits (n : ℕ) : ℕ := if n < 10 then 1 else 2

theorem expected_number_of_digits :
  measure_theory.integral (measure_theory.measure_space.comap (λ n : fair_dodecahedral_die, n.val) 
  (measure_theory.measure_space.measure_univ : measure_theory.measure fair_dodecahedral_die)) num_digits = 1.25 :=
begin
  sorry
end

end expected_number_of_digits_l107_107685


namespace minimal_sum_is_vertical_l107_107008

def grid (n : ℕ) : List (List ℕ) :=
  (List.range n).map (λ i => List.range (1 + i * 10) (11 + i * 10))

noncomputable def sum_of_products (grid : List (List ℕ)) (dominoes : List (ℕ × ℕ)) : ℕ :=
  dominoes.map (λ (i, j) => i * j).sum

def minimal_sum_domino_placement (dominoes : List (ℕ × ℕ)) : Prop :=
  dominoes.length = 50 ∧
  dominoes.all (λ (i, j) => (∃ k l, i = grid.k.l ∧ j = grid.(k+1).l))

theorem minimal_sum_is_vertical :
  ∀ dominoes, minimal_sum_domino_placement dominoes → sum_of_products (grid 10) dominoes = sum_of_products (grid 10) (List.range' 1 50).map (λ k => (k, k + 10)) :=
by
  sorry

end minimal_sum_is_vertical_l107_107008


namespace count_valid_three_digit_numbers_l107_107439

theorem count_valid_three_digit_numbers : 
  let is_valid (a b c : ℕ) := 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ b = (a + c) / 2 ∧ (a + c) % 2 = 0
  ∃ n : ℕ, (∀ a b c : ℕ, is_valid a b c → n = 45) :=
sorry

end count_valid_three_digit_numbers_l107_107439


namespace proof_problem_l107_107334

noncomputable def A (n : ℤ) := n^3 + 4 * n^2 + n - 6

theorem proof_problem :
  let S := { n : ℤ | -30 ≤ n ∧ n ≤ 100 ∧ A n % 5 = 0 }
  ∃ (count : ℕ) (min_n max_n : ℤ),
    count = 78 ∧
    min_n = -29 ∧
    max_n = 98 ∧
    ∀ n ∈ S, n ≥ min_n ∧ n ≤ max_n ∧
    ∃ (c : ℕ), c = count ∧ c = S.to_finset.card :=
begin
  sorry
end

end proof_problem_l107_107334


namespace find_ellipse_equation_l107_107812

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∃ c : ℝ, a > b ∧ b > 0 ∧ 4 * a = 16 ∧ |c| = 2 ∧ a^2 = b^2 + c^2

theorem find_ellipse_equation :
  (∃ (a b : ℝ), ellipse_equation a b) → (∃ b : ℝ, (a = 4) ∧ (b > 0) ∧ (b^2 = 12) ∧ (∀ x y : ℝ, (x^2 / 16) + (y^2 / 12) = 1)) :=
by {
  sorry
}

end find_ellipse_equation_l107_107812


namespace reconstruct_triangle_from_square_centers_l107_107127

-- Define three centers of the squares as given points in the 2D plane
variables {O₁ O₂ O₃ : ℝ × ℝ}

-- Define that there exists points A, B, C such that the rotations by 90 degrees around these centers map back to the original points
theorem reconstruct_triangle_from_square_centers (O₁ O₂ O₃ : ℝ × ℝ) :
  ∃ (A B C : ℝ × ℝ), 
    (rotate_around O₁ (90 : ℝ) B = C) ∧
    (rotate_around O₂ (90 : ℝ) C = A) ∧
    (rotate_around O₃ (90 : ℝ) A = B) :=
sorry

end reconstruct_triangle_from_square_centers_l107_107127


namespace Cara_age_is_40_l107_107299

def cara_older_than_mom : ℕ := 20
def mom_older_than_grandmother : ℕ := 15
def grandmother_age : ℕ := 75

theorem Cara_age_is_40 :
  let mom_age := grandmother_age - mom_older_than_grandmother in
  let cara_age := mom_age - cara_older_than_mom in
  cara_age = 40 :=
by
  let mom_age := grandmother_age - mom_older_than_grandmother
  let cara_age := mom_age - cara_older_than_mom
  sorry

end Cara_age_is_40_l107_107299


namespace length_segment_PT_l107_107470

open EuclideanGeometry

/-- Proof that the length of segment PT is (4*sqrt(2))/3
    given conditions in the problem. -/
theorem length_segment_PT (P Q R S : Point ℝ)
  (hP : P = ⟨0, 4⟩)
  (hQ : Q = ⟨4, 0⟩)
  (hR : R = ⟨1, 0⟩)
  (hS : S = ⟨3, 3⟩)
  (T : Point ℝ) 
  (hT : segment_intersect P Q R S T) :
  dist P T = 4 * real.sqrt 2 / 3 := 
sorry

end length_segment_PT_l107_107470


namespace ways_to_partition_6_into_4_boxes_l107_107854

theorem ways_to_partition_6_into_4_boxes : 
  ∃ (s : Finset (Finset ℕ)), (∀ (x ∈ s), ∃ (a b c d : ℕ), x = {a, b, c, d} ∧ a + b + c + d = 6) ∧ s.card = 9 :=
sorry

end ways_to_partition_6_into_4_boxes_l107_107854


namespace count_valid_n_is_9_l107_107261

noncomputable def count_valid_n : ℕ :=
  let factors_of_36 := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  ∃ (x y z : ℕ), (lcm x y = 180) ∧ (lcm x z = 900) ∧ (lcm y z ∈ {n : ℕ | ∃ k ∈ factors_of_36, n = 25 * k})

theorem count_valid_n_is_9 : count_valid_n = 9 :=
  sorry

end count_valid_n_is_9_l107_107261


namespace probability_even_product_l107_107606

def disks : List (Nat × Nat) := [(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)]

def even_product (x y : Nat) : Prop :=
  x * y % 2 = 0

theorem probability_even_product : 
  ((disks.filter (λ pair, even_product pair.1 pair.2)).length : ℚ) / disks.length = 7 / 9 :=
by
  sorry

end probability_even_product_l107_107606


namespace compare_y1_y2_l107_107811

theorem compare_y1_y2 (m y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 2*(-1) + m) 
  (h2 : y2 = 2^2 - 2*2 + m) : 
  y1 > y2 := 
sorry

end compare_y1_y2_l107_107811


namespace solution_pepper_used_l107_107291

variable (initialAmount remainingAmount usedAmount : ℝ)

def pepper_problem (initialAmount remainingAmount : ℝ) := 
  initialAmount = 0.25 ∧ remainingAmount = 0.09 → 
  usedAmount = initialAmount - remainingAmount

theorem solution_pepper_used : pepper_problem 0.25 0.09 :=
  by
    intro h
    have ha := h.1
    have hr := h.2
    show usedAmount = 0.16
    sorry

end solution_pepper_used_l107_107291


namespace expression_range_l107_107494

theorem expression_range (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2)
  + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ∧ 
  (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ≤ 8 :=
sorry

end expression_range_l107_107494


namespace average_of_x_y_z_l107_107445

theorem average_of_x_y_z (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := 
by 
  sorry

end average_of_x_y_z_l107_107445


namespace box_length_is_10_l107_107252

noncomputable def box_length (volume : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  volume / (width * height)

theorem box_length_is_10:
  let width := 18 in
  let height := 4 in
  let cube_volume := 12 in
  let num_cubes := 60 in
  let total_volume := num_cubes * cube_volume in
  box_length total_volume width height = 10 := by
  sorry

end box_length_is_10_l107_107252


namespace div_pow_eq_l107_107041

theorem div_pow_eq (n : ℕ) (h : n = 16 ^ 2023) : n / 4 = 4 ^ 4045 :=
by
  rw [h]
  sorry

end div_pow_eq_l107_107041


namespace top_red_second_black_probability_l107_107560

-- Define the deck and suits
inductive Suit
| spades | hearts | diamonds | clubs 

-- Define the ranks in the customized deck
inductive Rank
| ace | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | jack | queen | king | ace1 | ace2

-- Define the deck
structure Card :=
(rank : Rank)
(suit : Suit)

-- Define the color of each suit
def is_red (s : Suit) : Bool :=
  s = Suit.hearts ∨ s = Suit.diamonds

def is_black (s : Suit) : Bool :=
  s = Suit.spades ∨ s = Suit.clubs

-- Define the number of cards in the deck
def total_cards (deck : List Card) : ℕ :=
  deck.length

-- Probability calculation
noncomputable def probability_top_red_second_black (deck : List Card) : ℚ :=
  let red_cards := deck.filter (λ c => is_red c.suit)
  let black_cards := deck.filter (λ c => is_black c.suit)
  if red_cards.length * black_cards.length = 0 then 0 else
    (red_cards.length * black_cards.length : ℚ) / ((total_cards deck) * (total_cards deck - 1))

-- The theorem we want to prove
theorem top_red_second_black_probability (deck : List Card) (h_deck : total_cards deck = 60) :
  probability_top_red_second_black deck = 15 / 59 :=
sorry

end top_red_second_black_probability_l107_107560


namespace number_of_tires_slashed_l107_107486

-- Definitions based on conditions
def cost_per_tire : ℤ := 250
def cost_window : ℤ := 700
def total_cost : ℤ := 1450

-- Proof statement
theorem number_of_tires_slashed : ∃ T : ℤ, cost_per_tire * T + cost_window = total_cost ∧ T = 3 := 
sorry

end number_of_tires_slashed_l107_107486


namespace product_of_odd_integers_lt_5000_l107_107616

theorem product_of_odd_integers_lt_5000 :
  (∏ i in finset.filter (λ n, 0 < n ∧ n < 5000 ∧ odd n) (finset.range 5000), i) = 5000! / (2 ^ 2500 * 2500!) :=
sorry

end product_of_odd_integers_lt_5000_l107_107616


namespace median_equality_range_inequality_l107_107750

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l107_107750


namespace orthocenter_coincidence_l107_107035

-- Let's define the basic setup and structures.
variables {A B C O A' B' C' : Type}
variables (BC CA AB : set (Type))
variables (circumcircleABC : set (set (Type)))

-- The hypotheses
hypothesis h1 : O ∈ circumcircleABC
hypothesis h2 : A' ∈ BC
hypothesis h3 : B' ∈ CA
hypothesis h4 : C' ∈ AB
hypothesis h5 : O ∈ circumcircle (A, B', C')
hypothesis h6 : O ∈ circumcircle (B, C', A')
hypothesis h7 : O ∈ circumcircle (C, A', B')

-- Definitions of radical axes
def l_a : set (Type) :=
  radical_axis (circle_center B' B'C) (circle_center C' C'B)
def l_b : set (Type) :=
  radical_axis (circle_center C' C'A) (circle_center A' A'C)
def l_c : set (Type) :=
  radical_axis (circle_center A' A'B) (circle_center B' B'A)

-- The final theorem
theorem orthocenter_coincidence :
  orthocenter (triangle (l_a) (l_b) (l_c)) = orthocenter (triangle A B C) :=
sorry

end orthocenter_coincidence_l107_107035


namespace oakwood_team_count_l107_107682

theorem oakwood_team_count :
  let girls := 5
  let boys := 5
  let teams := (Finset.choose girls 3).card * (Finset.choose boys 2).card
  teams = 100 :=
by
  have h_girls : (Finset.choose 5 3).card = 10 := 
    by simp [Finset.choose_card_eq]
  have h_boys : (Finset.choose 5 2).card = 10 := 
    by simp [Finset.choose_card_eq]
  calc
    teams = 10 * 10 : by simp [h_girls, h_boys]
    ... = 100 : by norm_num

end oakwood_team_count_l107_107682


namespace ratio_of_areas_l107_107899

theorem ratio_of_areas (AB BC O : ℝ) (h_diameter : AB = 4) (h_BC : BC = 3)
  (ABD DBE ABDeqDBE : Prop) (x y : ℝ) 
  (h_area_ABCD : x = 7 * y) :
  (x / y) = 7 :=
by
  sorry

end ratio_of_areas_l107_107899


namespace median_equal_range_not_greater_l107_107760

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l107_107760


namespace find_ab_l107_107859

theorem find_ab (a b : ℝ) (h : (a - 2) ^ 2 + sqrt (b + 3) = 0) : a * b = -6 :=
by 
  sorry

end find_ab_l107_107859


namespace contrapositive_sin_eq_l107_107569

theorem contrapositive_sin_eq (A B : ℝ) : (A = B → sin A = sin B) → (sin A ≠ sin B → A ≠ B) :=
sorry

end contrapositive_sin_eq_l107_107569


namespace binom_600_600_l107_107320

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l107_107320


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107170

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107170


namespace combined_weight_of_olivers_bags_l107_107944

-- Define the weights and relationship between the weights
def weight_james_bag : ℝ := 18
def ratio_olivers_to_james : ℝ := 1 / 6
def weight_oliver_one_bag : ℝ := weight_james_bag * ratio_olivers_to_james
def number_of_oliver_bags : ℝ := 2

-- The proof problem statement: proving the combined weight of both Oliver's bags
theorem combined_weight_of_olivers_bags :
  number_of_oliver_bags * weight_oliver_one_bag = 6 := by
  sorry

end combined_weight_of_olivers_bags_l107_107944


namespace find_interest_rate_l107_107461

noncomputable def interest_rate (x y : ℝ) (x_rate y_rate_diff : ℝ) : ℝ :=
  y_rate_diff / y

theorem find_interest_rate
  (total_investment : ℝ)
  (y : ℝ)
  (x_rate : ℝ)
  (income_diff : ℝ)
  (x : ℝ := total_investment - y)
  (expected_r : ℝ := 0.08)
  : interest_rate x y x_rate income_diff = expected_r :=
by
  let r := income_diff / y
  have hr : r = expected_r, from sorry
  exact hr
  
end find_interest_rate_l107_107461


namespace systematic_sampling_questionnaire_B_count_l107_107136

theorem systematic_sampling_questionnaire_B_count (n : ℕ) (N : ℕ) (first_random : ℕ) (range_A_start range_A_end range_B_start range_B_end : ℕ) 
  (h1 : n = 32) (h2 : N = 960) (h3 : first_random = 9) (h4 : range_A_start = 1) (h5 : range_A_end = 460) 
  (h6 : range_B_start = 461) (h7 : range_B_end = 761) :
  ∃ count : ℕ, count = 10 := by
  sorry

end systematic_sampling_questionnaire_B_count_l107_107136


namespace circumcircles_intersect_l107_107012

variable {K : Type*} [EuclideanGeometry K]

-- Define points
variables (A1 A2 B1 B2 C1 C2 : K)

-- Conditions
def circumcircle_intersects_at_one_point (A B C P : K) : Prop :=
  ∃ O : K, O ∈ circle A B C ∧ O ∈ circle A B P ∧ O ∈ circle B A C ∧ O ∈ circle B A P -- Just an example definition. You should adjust as needs.

-- The statement to be proven
theorem circumcircles_intersect (
  h1 : circumcircle_intersects_at_one_point A1 B1 C1 P,
  h2 : circumcircle_intersects_at_one_point A1 B2 C2 P,
  h3 : circumcircle_intersects_at_one_point A2 B1 C2 P,
  h4 : circumcircle_intersects_at_one_point A2 B2 C1 P
) : ∃ Q : K, Q ∈ circle A2 B2 C2 ∧ Q ∈ circle A2 B1 C1 ∧ Q ∈ circle A1 B2 C1 ∧ Q ∈ circle A1 B1 C2 :=
sorry

end circumcircles_intersect_l107_107012


namespace can_restore_axes_l107_107880

noncomputable def restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : Prop :=
  ∃ (B C D : ℝ×ℝ),
    (B.fst = A.fst ∧ B.snd = 0) ∧
    (C.fst = A.fst ∧ C.snd = A.snd) ∧
    (D.fst = A.fst ∧ D.snd = 3 ^ C.fst) ∧
    (∃ (extend_perpendicular : ∀ (x: ℝ), ℝ→ℝ), extend_perpendicular A.snd B.fst = D.snd)

theorem can_restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : restore_axes A hA :=
  sorry

end can_restore_axes_l107_107880


namespace union_area_XYZ_X_l107_107028

-- Given conditions
constants XY YZ ZX : ℝ
constants O X Y Z X' Y' Z' : Point
axiom XY_eq : XY = 16
axiom YZ_eq : YZ = 17
axiom ZX_eq : ZX = 18
axiom centroid_O : centroid X Y Z = O
axiom rotation : rotate_180 O X Y Z = (X', Y', Z')
-- Required area calculation
noncomputable def area_XYZ : ℝ := sorry
noncomputable def area_X'Y'Z' : ℝ := area_XYZ

-- Problem statement
theorem union_area_XYZ_X'Y'Z' : area_XYZ + area_X'Y'Z' = 248.54 := 
by
  sorry

end union_area_XYZ_X_l107_107028


namespace train_crossing_pole_time_l107_107669

/-- 
Given the conditions:
1. The train is running at a speed of 60 km/hr.
2. The length of the train is 66.66666666666667 meters.
Prove that it takes 4 seconds for the train to cross the pole.
-/
theorem train_crossing_pole_time :
  let speed_km_hr := 60
  let length_m := 66.66666666666667
  let conversion_factor := 1000 / 3600
  let speed_m_s := speed_km_hr * conversion_factor
  let time := length_m / speed_m_s
  time = 4 :=
by
  sorry

end train_crossing_pole_time_l107_107669


namespace min_phone_calls_l107_107981

theorem min_phone_calls (n : ℕ) (h : n > 1) : 
  ∀ (info : Fin n → Set (Fin n)), 
  (∀ i j, i ≠ j → info i ≠ info j) → 
  ∃ calls : ℕ, calls = 2 * n - 2 ∧ 
    (∀ (info_after_calls : Fin n → Set (Fin n)),
    (∀ i j, i ≠ j → info_after_calls i = info_after_calls j) →
    info_after_calls = Fin n (λ _, Fin n (λ x, x))) := 
begin
  sorry
end

end min_phone_calls_l107_107981


namespace books_sold_in_store_on_saturday_l107_107657

namespace BookshopInventory

def initial_inventory : ℕ := 743
def saturday_online_sales : ℕ := 128
def sunday_online_sales : ℕ := 162
def shipment_received : ℕ := 160
def final_inventory : ℕ := 502

-- Define the total number of books sold
def total_books_sold (S : ℕ) : ℕ := S + saturday_online_sales + 2 * S + sunday_online_sales

-- Net change in inventory equals total books sold minus shipment received
def net_change_in_inventory (S : ℕ) : ℕ := total_books_sold S - shipment_received

-- Prove that the difference between initial and final inventories equals the net change in inventory
theorem books_sold_in_store_on_saturday : ∃ S : ℕ, net_change_in_inventory S = initial_inventory - final_inventory ∧ S = 37 :=
by
  sorry

end BookshopInventory

end books_sold_in_store_on_saturday_l107_107657


namespace smallest_n_divides_24_and_1024_l107_107617

theorem smallest_n_divides_24_and_1024 : ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (1024 ∣ n^3) ∧ (∀ m : ℕ, (m > 0 ∧ (24 ∣ m^2) ∧ (1024 ∣ m^3)) → n ≤ m) :=
by
  sorry

end smallest_n_divides_24_and_1024_l107_107617


namespace joan_spent_half_dollars_on_wednesday_l107_107533

theorem joan_spent_half_dollars_on_wednesday (total_half_dollars spent_thursday : ℕ) 
  (h1 : total_half_dollars = 18) (h2 : spent_thursday = 14) : 
  total_half_dollars - spent_thursday = 4 :=
by
  rw [h1, h2]
  norm_num
  sorry

end joan_spent_half_dollars_on_wednesday_l107_107533


namespace functional_equation_solution_l107_107496

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem functional_equation_solution (f : ℝ → ℝ) (x y : ℝ) (h_pos_x: 0 < x) (h_pos_y: 0 < y) :
  (f x + f y = f (x * y * f (x + y))) ↔ f = λ x, 1 / x :=
by
  sorry

end functional_equation_solution_l107_107496


namespace log_floor_sum_l107_107294

theorem log_floor_sum : (∑ N in Finset.range 730, Int.floor (Real.log N / Real.log 3)) = 2460 := by
  sorry

end log_floor_sum_l107_107294


namespace chess_tournament_participants_l107_107888

theorem chess_tournament_participants (d : ℕ) (hpos : 0 < d)
    (hboys : 5 * d)
    (hpoints : 2 * (d * 5 + 1)) :
    d = 1 → (d + 5 * d) = 6 :=
by
  intro hd1
  calc
    d + 5 * d = 6 * d : by ring
            ... = 6     : by rw hd1 
    sorry

end chess_tournament_participants_l107_107888


namespace race_minimum_distance_l107_107118

-- Define the necessary points and conditions
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1000, 1500)
def wall_length : ℝ := 1500
def dist(A B : ℝ × ℝ) : ℝ := Real.sqrt((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define main statement to prove: The minimum distance from A to B touching the wall equals 1803 meters.
theorem race_minimum_distance :
  dist A B = 1803 := 
by
  sorry

end race_minimum_distance_l107_107118


namespace joe_speed_first_part_l107_107912

theorem joe_speed_first_part (v : ℝ) :
  let d1 := 420 -- distance of the first part in miles
  let d2 := 120 -- distance of the second part in miles
  let v2 := 40  -- speed during the second part in miles per hour
  let d_total := d1 + d2 -- total distance
  let avg_speed := 54 -- average speed in miles per hour
  let t1 := d1 / v -- time for the first part
  let t2 := d2 / v2 -- time for the second part
  let t_total := t1 + t2 -- total time
  (d_total / t_total) = avg_speed -> v = 60 :=
by
  intros
  sorry

end joe_speed_first_part_l107_107912


namespace math_question_l107_107773

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l107_107773


namespace mr_benson_total_payment_l107_107247

-- Definitions based on the problem conditions
def ticket_cost : ℝ := 40
def total_tickets : ℝ := 12
def discount_percentage : ℝ := 5 / 100
def discounted_tickets := total_tickets - 10
def discounted_ticket_cost := ticket_cost * (1 - discount_percentage)
def non_discounted_ticket_cost := ticket_cost
def total_discounted_tickets_cost := discounted_ticket_cost * discounted_tickets
def total_non_discounted_tickets_cost := non_discounted_ticket_cost * 10
def total_amount_paid := total_discounted_tickets_cost + total_non_discounted_tickets_cost

-- Theorem statement of the problem
theorem mr_benson_total_payment : total_amount_paid = 476 :=
by
  -- Assert the conditions given in the problem
  have h1 : ticket_cost = 40 := rfl
  have h2 : total_tickets = 12 := rfl
  have h3 : discount_percentage = 5 / 100 := rfl
  have h4 : discounted_tickets = 2 := by simp [discounted_tickets, total_tickets, h2]
  have h5 : discounted_ticket_cost = 38 := by simp [discounted_ticket_cost, ticket_cost, h1, discount_percentage, h3]
  have h6 : total_discounted_tickets_cost = 76 := by simp [total_discounted_tickets_cost, discounted_ticket_cost, h5, discounted_tickets, h4]
  have h7 : total_non_discounted_tickets_cost = 400 := by simp [total_non_discounted_tickets_cost, non_discounted_ticket_cost, ticket_cost, h1]
  have h8 : total_amount_paid = 476 := by simp [total_amount_paid, total_discounted_tickets_cost, h6, total_non_discounted_tickets_cost, h7]
  exact h8

end mr_benson_total_payment_l107_107247


namespace smallest_natural_number_exists_l107_107720

theorem smallest_natural_number_exists (n : ℕ) : (∃ n, ∃ a b c : ℕ, n = 15 ∧ 1998 = a * (5 ^ 4) + b * (3 ^ 4) + c * (1 ^ 4) ∧ a + b + c = 15) :=
sorry

end smallest_natural_number_exists_l107_107720


namespace smallest_prime_12_less_than_perfect_square_l107_107207

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107207


namespace projection_of_b_onto_a_l107_107432

def vector_a : ℝ × ℝ × ℝ := (0, 1, 1)
def vector_b : ℝ × ℝ × ℝ := (1, 1, 0)

noncomputable def projection (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let magnitude_squared := v.1^2 + v.2^2 + v.3^2
  (dot_product / magnitude_squared) * v

theorem projection_of_b_onto_a : 
  projection vector_a vector_b = (0, 0.5, 0.5) :=
sorry

end projection_of_b_onto_a_l107_107432


namespace no_integer_r_exists_l107_107710

theorem no_integer_r_exists (r : ℤ) : ¬ ∀ n : ℕ, ∃ k : ℤ, (n! : ℤ) = k * 2^(n - r) := 
sorry

end no_integer_r_exists_l107_107710


namespace smallest_positive_period_range_of_f_shifted_l107_107411

-- Function definition
noncomputable def f (x : ℝ) : ℝ :=
  |2 * Real.sin x        (Real.sqrt 3 * (Real.sin x - Real.cos x))|
                      (Real.sin x + Real.cos x       Real.cos x)|

-- Function with shifted argument
noncomputable def f_shifted (x : ℝ) : ℝ := f (x - Real.pi / 2)

-- Define the interval
def interval (a b x : ℝ) : Prop := a ≤ x ∧ x ≤ b

-- Statements
theorem smallest_positive_period (x : ℝ) : ∃ T > 0, T = Real.pi ∧ f (x + T) = f x :=
sorry

theorem range_of_f_shifted (x : ℝ) : interval 0 (Real.pi / 2) x → interval (-2) (Real.sqrt 3) (f_shifted x) :=
sorry

end smallest_positive_period_range_of_f_shifted_l107_107411


namespace count_valid_three_digit_numbers_l107_107440

def is_valid_pair (A C : Nat) : Prop :=
  A + C = 8

def is_valid_number (A B C : Nat) : Prop :=
  B = (A + C) / 2 ∧ (A + B + C = 12)

theorem count_valid_three_digit_numbers :
  ∃ A B C : Nat, A ∈ {1, 2, 3, 4, 5, 6, 7} ∧ C ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ is_valid_pair A C ∧ is_valid_number A B C :=
(Predicate.count (λ A B C, A ∈ {1, 2, 3, 4, 5, 6, 7} ∧ C ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ is_valid_pair A C ∧ is_valid_number A B C) = 7) :=
by
  -- Mathematical details skipped for brevity
  sorry

end count_valid_three_digit_numbers_l107_107440


namespace condition_s_for_q_condition_r_for_q_condition_p_for_s_l107_107485

variables {p q r s : Prop}

-- Given conditions from a)
axiom h₁ : r → p
axiom h₂ : q → r
axiom h₃ : s → r
axiom h₄ : q → s

-- The corresponding proof problems based on c)
theorem condition_s_for_q : (s ↔ q) :=
by sorry

theorem condition_r_for_q : (r ↔ q) :=
by sorry

theorem condition_p_for_s : (s → p) :=
by sorry

end condition_s_for_q_condition_r_for_q_condition_p_for_s_l107_107485


namespace remaining_apps_eq_files_plus_more_initial_apps_eq_16_l107_107335

-- Defining the initial number of files
def initial_files: ℕ := 9

-- Defining the remaining number of files and apps
def remaining_files: ℕ := 5
def remaining_apps: ℕ := 12

-- Given: Dave has 7 more apps than files left
def apps_more_than_files: ℕ := 7

-- Equating the given condition 12 = 5 + 7
theorem remaining_apps_eq_files_plus_more :
  remaining_apps = remaining_files + apps_more_than_files := by
  sorry -- This would trivially prove as 12 = 5+7

-- Proving the number of initial apps
theorem initial_apps_eq_16 (A: ℕ) (h1: initial_files = 9) (h2: remaining_files = 5) (h3: remaining_apps = 12) (h4: apps_more_than_files = 7):
  A - remaining_apps = initial_files - remaining_files → A = 16 := by
  sorry

end remaining_apps_eq_files_plus_more_initial_apps_eq_16_l107_107335


namespace part_a_part_b_l107_107636

variables {A B C B1 C1 O : Type} 

def is_altitude (A B C B1 C1 : Type) : Prop := sorry

def is_tangent_to_circumcircle (A B C l : Type) : Prop := sorry

def parallel (l1 l2 : Type) : Prop := sorry

def perpendicular (l1 l2 : Type) : Prop := sorry

def is_center_of_circumcircle (A B C O : Type) : Prop := sorry

theorem part_a (h1 : is_altitude A B C B1 C1)
  (h2 : is_tangent_to_circumcircle A B C l)
  : parallel l (segment B1 C1) := 
sorry

theorem part_b (h1 : is_center_of_circumcircle A B C O)
  (h2 : parallel l (segment B1 C1))
  : perpendicular (segment B1 C1) (segment O A) :=
sorry

end part_a_part_b_l107_107636


namespace sum_q_t_8_eq_128_l107_107920

-- Define the type of 8-tuples where each entry is either 0 or 1
def T := {t: Fin 8 → ℕ // ∀ i, t i = 0 ∨ t i = 1}

-- Define q_t as the polynomial of degree at most 7
noncomputable def q_t (t: T) (x: ℕ) :=
  Polynomial.sum (fun i : Fin 8 => if t.val i = 1 then Polynomial.monomial i 1 else 0)

-- Define the polynomial q(x)
noncomputable def q (x: ℕ) :=
  @Finset.sum (T) (q_t · x) _ Finset.univ

-- The theorem we aim to prove
theorem sum_q_t_8_eq_128:
  q 8 = 128 :=
sorry

end sum_q_t_8_eq_128_l107_107920


namespace median_equality_range_inequality_l107_107753

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l107_107753


namespace triangle_proof_l107_107465

noncomputable def triangle_problem (a b c : ℝ) (cosA : ℝ) : Prop :=
  a = 2 ∧ c = sqrt 2 ∧ cosA = -sqrt 2 / 4 →
  sinC = sqrt 7 / 4 ∧ b = 1 ∧ cos (2 * A + π / 3) = (-3 + sqrt 21) / 8

theorem triangle_proof : ∀ (a b c A : ℝ),
  a = 2 →
  c = sqrt 2 →
  cos A = - sqrt 2 / 4 →
  let sinC := sqrt 7 / 4 in
  let b := 1 in
  let cos_result := ( -3 + sqrt 21 ) / 8 in
  sinC = sqrt 7 / 4 ∧ b = 1 ∧ cos (2 * A + π / 3 ) = cos_result
:= by
  intros a b c A h₀ h₁ h₂
  sorry

end triangle_proof_l107_107465


namespace smallest_prime_12_less_than_square_l107_107189

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107189


namespace number_of_factors_of_m_l107_107504

noncomputable def m : ℕ := 2^3 * 3^3 * 5^4 * 6^5

theorem number_of_factors_of_m : ∀ (m : ℕ), 
  m = 2^3 * 3^3 * 5^4 * 6^5 → (∃ n, n = 405 ∧ (∀ a b c, (0 ≤ a ∧ a ≤ 8 ∧ 0 ≤ b ∧ b ≤ 8 ∧ 0 ≤ c ∧ c ≤ 4) →
    m = 2^a * 3^b * 5^c) → n = 9 * 9 * 5) := 
begin
  sorry
end

end number_of_factors_of_m_l107_107504


namespace sine_of_pi_minus_alpha_l107_107862

theorem sine_of_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 3) : Real.sin (π - α) = 1 / 3 :=
by
  sorry

end sine_of_pi_minus_alpha_l107_107862


namespace sum_of_second_and_third_smallest_is_804_l107_107611

noncomputable def sum_of_second_and_third_smallest : Nat :=
  let digits := [1, 6, 8]
  let second_smallest := 186
  let third_smallest := 618
  second_smallest + third_smallest

theorem sum_of_second_and_third_smallest_is_804 :
  sum_of_second_and_third_smallest = 804 :=
by
  sorry

end sum_of_second_and_third_smallest_is_804_l107_107611


namespace expression_evaluation_l107_107346

-- Using the given conditions
def a : ℕ := 3
def b : ℕ := a^2 + 2 * a + 5
def c : ℕ := b^2 - 14 * b + 45

-- We need to assume that none of the denominators are zero.
lemma non_zero_denominators : (a + 1 ≠ 0) ∧ (b - 3 ≠ 0) ∧ (c + 7 ≠ 0) :=
  by {
    -- Proof goes here
  sorry }

theorem expression_evaluation :
  (a = 3) →
  ((a^2 + 2*a + 5) = b) →
  ((b^2 - 14*b + 45) = c) →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (↑(a + 3) / ↑(a + 1) * ↑(b - 1) / ↑(b - 3) * ↑(c + 9) / ↑(c + 7) = 4923 / 2924) :=
  by {
    -- Proof goes here
  sorry }

end expression_evaluation_l107_107346


namespace lambda_value_l107_107431

-- Define the given conditions and problem
variables {λ : ℝ}
def a : ℝ × ℝ := (1, real.sqrt(3))
def b : ℝ × ℝ := (1, 0)  -- Since we assume b has a magnitude of 1 (norm b = 1)
def a_squared : ℝ := 1^2 + (real.sqrt(3))^2
def b_squared : ℝ := 1^2
def a_plus_lambda_b_zero : Prop := (1 + λ * 1 = 0) ∧ (real.sqrt(3) + λ * 0 = 0)

-- The theorem to be proven
theorem lambda_value : λ = 2 :=
begin
  -- Available conditions
  have h₁ : (1 + λ * 1 = 0) ∧ (real.sqrt(3) + λ * 0 = 0),
  { -- proof or further steps would normally follow here
    sorry },
  have h₂ : a_squared = λ^2 * b_squared,
  { -- proof or further steps would normally follow here
    sorry },
  
  -- Main goal
  show λ = 2,
  sorry
end

end lambda_value_l107_107431


namespace solution_set_l107_107389

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f(-x) = f(x)

def derivative_condition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, x * (derivative f x) > -2 * f(x)

def g_definition (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = x^2 * f x

theorem solution_set (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_even : even_function f)
  (h_deriv_cond : derivative_condition f)
  (h_g_def : g_definition g) :
  {x | g x < g 1} = {x | -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_l107_107389


namespace smallest_prime_less_than_square_l107_107154

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107154


namespace player_A_win_probability_l107_107479

theorem player_A_win_probability:
  ∃ (p : ℚ), 
  let wins_needed_for_championship := 4 in
  let player_A_current_wins := 3 in
  let player_B_current_wins := 2 in
  let total_games_needed := wins_needed_for_championship - player_A_current_wins in
  let prob_per_game := 1 / 2 in
  (player_A_current_wins + total_games_needed <= wins_needed_for_championship) ∧
  (player_B_current_wins + total_games_needed < wins_needed_for_championship) ∧
  p = 3 / 4 :=
sorry

end player_A_win_probability_l107_107479


namespace find_a1_l107_107426

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 8 = 2 ∧ ∀ n, a (n + 1) = 1 / (1 - a n)

theorem find_a1 (a : ℕ → ℝ) (h : seq a) : a 1 = 1/2 := by
sorry

end find_a1_l107_107426


namespace max_lessons_possible_l107_107592

theorem max_lessons_possible 
  (s p b : ℕ) 
  (h1 : 2 * p * b = 36) 
  (h2 : 2 * s * b = 72) 
  (h3 : 2 * s * p = 54) 
  : 2 * s * p * b = 216 :=
begin
  sorry
end

end max_lessons_possible_l107_107592


namespace bert_same_kangaroos_as_kameron_in_40_days_l107_107491

theorem bert_same_kangaroos_as_kameron_in_40_days
  (k : ℕ := 100)
  (b : ℕ := 20)
  (r : ℕ := 2) :
  ∃ t : ℕ, t = 40 ∧ b + t * r = k := by
  sorry

end bert_same_kangaroos_as_kameron_in_40_days_l107_107491


namespace expansion_terms_2002_l107_107365

theorem expansion_terms_2002 (N : ℕ) :
  (∃ N, (∑ n in finset.range 5.succ, (multiset.replicate N ()).powerset.card = 2002) ↔ N = 16) :=
by
  sorry

end expansion_terms_2002_l107_107365


namespace highest_degree_divisibility_l107_107715

-- Definition of the problem settings
def prime_number := 1991
def number_1 := 1990 ^ (1991 ^ 1002)
def number_2 := 1992 ^ (1501 ^ 1901)
def combined_number := number_1 + number_2

-- Statement of the proof to be formalized
theorem highest_degree_divisibility (k : ℕ) : k = 1001 ∧ prime_number ^ k ∣ combined_number := by
  sorry

end highest_degree_divisibility_l107_107715


namespace tangency_of_circumcircle_DFG_to_BG_l107_107540

variables {A B C D E F G : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space G] -- Define the points as variables in a metric space

-- Conditions
variable (rectangle_ABCD: ∀ (P : Point), P ∈ rectangle A B C D)
variable (diagonals_AC_BD: diagonal A C ∧ diagonal B D)
variable (intersection_E: E = intersection (bisector_angle A C D) (segment C D))
variable (midpoint_E_DF: midpoint E (segment D F))
variable (point_G_on_BC: G ∈ line_segment B C ∧ dist B G = dist A C)

-- Problem Statement
theorem tangency_of_circumcircle_DFG_to_BG : tangency (circumcircle D F G) (line_segment B G) := sorry

end tangency_of_circumcircle_DFG_to_BG_l107_107540


namespace cube_root_of_64_l107_107991

theorem cube_root_of_64 :
  ∃ (x : ℝ), x^3 = 64 ∧ x = 4 :=
by
  use 4
  split
  · norm_num
  · refl

end cube_root_of_64_l107_107991


namespace determine_z_l107_107506

noncomputable def z : ℂ := sorry

theorem determine_z (z : ℂ) (h1 : z^2 = 6 * z - 27 + 12 * complex.I) (h2 : complex.abs z ∈ ℤ) :
  z = 3 + (complex.sqrt 6 + complex.sqrt 6 * complex.I) ∨ z = 3 - (complex.sqrt 6 + complex.sqrt 6 * complex.I) :=
sorry

end determine_z_l107_107506


namespace solve_congruence_l107_107976

theorem solve_congruence (x : ℤ) : 
  (10 * x + 3) % 18 = 11 % 18 → x % 9 = 8 % 9 :=
by {
  sorry
}

end solve_congruence_l107_107976


namespace haphazardly_hung_pictures_l107_107062

theorem haphazardly_hung_pictures (total_pictures vert_hung horizontal_hung haphazardly_hung : ℕ) 
    (h_total : total_pictures = 30) (h_vert : vert_hung = 10) (h_hor : horizontal_hung = total_pictures / 2) :
    (total_pictures - vert_hung - horizontal_hung = haphazardly_hung) → haphazardly_hung = 5 :=
by
  intros h
  rw [h_total, h_vert, h_hor, Nat.div_self] at h
  norm_num at h
  exact h

end haphazardly_hung_pictures_l107_107062


namespace tony_cost_per_sqft_l107_107602

def master_bedroom_and_bath_sqft : ℝ := 500
def living_area_sqft : ℝ := 400
def in_law_suite_sqft : ℝ := 300
def base_rent : ℝ := 3000
def additional_rent_in_law_suite : ℝ := 500
def monthly_utilities_cost : ℝ := 250
def monthly_property_tax : ℝ := 100

def total_sqft : ℝ :=
  master_bedroom_and_bath_sqft + living_area_sqft + in_law_suite_sqft

def total_monthly_cost : ℝ :=
  base_rent + additional_rent_in_law_suite + monthly_utilities_cost + monthly_property_tax

def cost_per_sqft : ℝ :=
  total_monthly_cost / total_sqft

theorem tony_cost_per_sqft :
  cost_per_sqft = 3.21 :=
by
  sorry

end tony_cost_per_sqft_l107_107602


namespace quilt_shaded_fraction_l107_107586

theorem quilt_shaded_fraction :
  let original_squares := 9
  let shaded_column_squares := 3
  let fraction_shaded := shaded_column_squares / original_squares 
  fraction_shaded = 1/3 :=
by
  sorry

end quilt_shaded_fraction_l107_107586


namespace mateen_garden_area_l107_107890

theorem mateen_garden_area :
  ∃ (L W : ℝ), (20 * L = 1000) ∧ (8 * (2 * L + 2 * W) = 1000) ∧ (L * W = 625) :=
by
  sorry

end mateen_garden_area_l107_107890


namespace candy_cost_l107_107246

theorem candy_cost (x : ℝ) : 
  (15 * x + 30 * 5) / (15 + 30) = 6 -> x = 8 :=
by sorry

end candy_cost_l107_107246


namespace geometric_triangle_q_range_l107_107403

theorem geometric_triangle_q_range (a : ℝ) (q : ℝ) (h : 0 < q) 
  (h1 : a + q * a > (q ^ 2) * a)
  (h2 : q * a + (q ^ 2) * a > a)
  (h3 : a + (q ^ 2) * a > q * a) : 
  q ∈ Set.Ioo ((Real.sqrt 5 - 1) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end geometric_triangle_q_range_l107_107403


namespace quotient_of_division_l107_107535

theorem quotient_of_division (Q : ℤ) (h1 : 172 = (17 * Q) + 2) : Q = 10 :=
sorry

end quotient_of_division_l107_107535


namespace smallest_prime_less_than_perfect_square_is_13_l107_107194

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107194


namespace TV_cost_exact_l107_107061

theorem TV_cost_exact (savings : ℝ) (fraction_furniture : ℝ) (fraction_tv : ℝ) (original_savings : ℝ) (tv_cost : ℝ) :
  savings = 880 →
  fraction_furniture = 3 / 4 →
  fraction_tv = 1 - fraction_furniture →
  tv_cost = fraction_tv * savings →
  tv_cost = 220 :=
by
  sorry

end TV_cost_exact_l107_107061


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107167

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107167


namespace angle_C_pi_over_six_l107_107464

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, and C respectively,
    and given that a² + b² = c² + √3ab, we prove that angle C is π/6. -/
theorem angle_C_pi_over_six (a b c : ℝ) (A B C : ℝ)
  (h_triangle_angles: 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (h_side_rel : a^2 + b^2 = c^2 + (real.sqrt 3) * a * b) :
  C = π / 6 :=
by
  sorry

end angle_C_pi_over_six_l107_107464


namespace sqrt_sum_of_fractions_as_fraction_l107_107624

theorem sqrt_sum_of_fractions_as_fraction :
  (Real.sqrt ((36 / 49) + (16 / 9) + (1 / 16))) = (45 / 28) :=
by
  sorry

end sqrt_sum_of_fractions_as_fraction_l107_107624


namespace find_other_root_l107_107593

variables {a b c : ℝ}

theorem find_other_root
  (h_eq : ∀ x : ℝ, a * (b - c) * x^2 + b * (c - a) * x + c * (a - b) = 0)
  (root1 : a * (b - c) * 1^2 + b * (c - a) * 1 + c * (a - b) = 0) :
  ∃ k : ℝ, k = c * (a - b) / (a * (b - c)) ∧
           a * (b - c) * k^2 + b * (c - a) * k + c * (a - b) = 0 := 
sorry

end find_other_root_l107_107593


namespace simplify_expression_l107_107970

theorem simplify_expression :
  (2 : ℝ) * (2 * a) * (4 * a^2) * (3 * a^3) * (6 * a^4) = 288 * a^10 := 
by {
  sorry
}

end simplify_expression_l107_107970


namespace polynomial_is_positive_for_all_x_l107_107953

noncomputable def P (x : ℝ) : ℝ := x^12 - x^9 + x^4 - x + 1

theorem polynomial_is_positive_for_all_x (x : ℝ) : P x > 0 := 
by
  dsimp [P]
  sorry -- Proof is omitted.

end polynomial_is_positive_for_all_x_l107_107953


namespace count_integers_between_1_and_2000_diff_squares_exclude_5_l107_107852

theorem count_integers_between_1_and_2000_diff_squares_exclude_5 :
  let count := (finset.range 2001).filter (λ n, (∃ a b, n = a ^ 2 - b ^ 2) ∧ ¬ (n % 5 = 0)).card
  in count = 1100 := 
by {
  sorry
}

end count_integers_between_1_and_2000_diff_squares_exclude_5_l107_107852


namespace percentage_of_girls_l107_107879

variables (total_students : ℕ) (G_percent B_percent Bb_percent Bnb_percent Gbb : ℚ)

-- Condition 1: There are 25 students in the class.
def total_students_eq : total_students = 25 := sorry

-- Condition 2: Some percentage are girls (G%) and the rest are boys (B%).
-- Condition 3: G% + B% = 100%.
def percentages_sum_one_hundred (G_percent B_percent : ℚ) : Prop :=
G_percent + B_percent = 100

-- Condition 4: 40% of the boys like playing basketball and the rest don't.
def boys_like_basketball_percentage (Bb_percent B_percent : ℚ) : Prop :=
Bb_percent = 0.40 * B_percent

def boys_not_like_basketball_percentage (Bnb_percent B_percent : ℚ) : Prop :=
Bnb_percent = 0.60 * B_percent

-- Condition 5: The number of girls who like playing basketball is double the number of boys who don't like to.
def girls_basketball_double_boys_no_basketball (Gbb Bnb_percent : ℚ) : Prop :=
Gbb = 2 * Bnb_percent

-- Condition 6: 80% of the girls like playing basketball.
def girls_like_basketball_percentage (Gbb G_percent : ℚ) : Prop :=
Gbb = 0.80 * G_percent

-- The question and proof goal
theorem percentage_of_girls
  (h1 : total_students_eq)
  (h2 : percentages_sum_one_hundred G_percent B_percent)
  (h3 : boys_like_basketball_percentage Bb_percent B_percent)
  (h4 : boys_not_like_basketball_percentage Bnb_percent B_percent)
  (h5 : girls_basketball_double_boys_no_basketball Gbb Bnb_percent)
  (h6 : girls_like_basketball_percentage Gbb G_percent) :
  G_percent = 60 :=
sorry

end percentage_of_girls_l107_107879


namespace smallest_prime_12_less_perfect_square_l107_107141

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107141


namespace marcella_initial_pairs_l107_107938

theorem marcella_initial_pairs (lost_shoes: ℕ) (remaining_pairs: ℕ) (initial_pairs: ℕ) :
  lost_shoes = 9 →
  remaining_pairs = 15 →
  initial_pairs = remaining_pairs + lost_shoes :=
by
  intros h1 h2
  rw [h1, h2]
  exact 24

end marcella_initial_pairs_l107_107938


namespace num_squares_50_to_each_of_3_l107_107997

def distance (A B : ℕ × ℕ) : ℕ :=
  max (abs (A.1 - B.1)) (abs (A.2 - B.2))

def is_square_at_distance (A B C : ℕ × ℕ) (P : ℕ × ℕ) : Prop :=
  distance A P = 50 ∧ distance B P = 50 ∧ distance C P = 50

def num_squares_at_distance (A B C : ℕ × ℕ) : ℕ :=
  let all_squares := { P : ℕ × ℕ | is_square_at_distance A B C P }
  in fintype.card all_squares

theorem num_squares_50_to_each_of_3 (A B C : ℕ × ℕ)
  (hAB : distance A B = 100) (hBC : distance B C = 100) (hAC : distance A C = 100) :
  num_squares_at_distance A B C = 1 :=
sorry

end num_squares_50_to_each_of_3_l107_107997


namespace hyperbola_properties_l107_107515

noncomputable def hyperbola_passing_point (x y : ℝ) := (y^2 / 9) - (x^2 / 4) = 1
def passes_through_point (x y : ℝ) := (1 = x ∧ y = 3 * Real.sqrt 5 / 2)
def asymptotes (x y : ℝ) := y = (3 / 2) * x ∨ y = -(3 / 2) * x

theorem hyperbola_properties :
  ∃ (x y : ℝ), hyperbola_passing_point x y ∧ passes_through_point x y ∧ asymptotes x y ∧
    hyperbola_passing_point x y ∧
    ∀ (a b c e : ℝ), 
    (a = 3) ∧ (b = 2) ∧ (c = Real.sqrt(a^2 + b^2)) ∧ 
    (e = c / a) ∧
    (x = 0 ∧ (y = 3 ∨ y = -3))
  := sorry

end hyperbola_properties_l107_107515


namespace binomial_600_600_l107_107307

theorem binomial_600_600 : nat.choose 600 600 = 1 :=
by
  -- Given the condition that binomial coefficient of n choose n is 1 for any non-negative n
  have h : ∀ n : ℕ, nat.choose n n = 1 := sorry
  -- Applying directly to the specific case n = 600
  exact h 600

end binomial_600_600_l107_107307


namespace number_of_valid_n_l107_107853

theorem number_of_valid_n:
  ∃ ( N : Finset ℕ), 
  (∀ (n ∈ N), 
     (↑(n + 1050) / 90 : ℝ) = Real.floor (Real.sqrt (n : ℝ)) ∧ n > 0) ∧
  N.card = 5 :=
sorry

end number_of_valid_n_l107_107853


namespace find_root_k_l107_107366
noncomputable
def polynomial : ℝ → ℝ := λ x, 12 * x^3 + 6 * x^2 - 54 * x + 63

def isRootMultiplicityAtLeast (p : ℝ → ℝ) (x0 : ℝ) (m : ℕ) : Prop :=
  ∀ i : ℕ, i < m → (derivative^[i] p) x0 = 0

theorem find_root_k (k : ℝ) (h : isRootMultiplicityAtLeast polynomial k 2) : k = -11 / 9 :=
  sorry

end find_root_k_l107_107366


namespace max_value_of_sum_of_powers_l107_107635

theorem max_value_of_sum_of_powers (k : ℕ) (a : ℝ) (h : 0 < a) :
  ∀ r : ℕ, 1 ≤ r ∧ r ≤ k → 
  ∀ (k_i : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ r → k_i i ∈ ℕ) →
  (∀ i, 1 ≤ i ∧ i ≤ r → k_i i ≥ 0) →
  (∑ i in range r, k_i i) = k →
  (∑ i in range r, a^(k_i i)) ≤ max (k * a) (a^k) :=
  sorry

end max_value_of_sum_of_powers_l107_107635


namespace coefficient_of_x21_l107_107898

noncomputable theory

def coefficient_x21 (p q : Polynomial ℚ) (k : ℕ) : ℚ :=
  Polynomial.coeff (p * q) k

theorem coefficient_of_x21 :
  let p := (Polynomial.finset_sum (Finset.range 21) (λ n : ℕ, Polynomial.monomial n 1))
  let q := (Polynomial.finset_sum (Finset.range 11) (λ n : ℕ, Polynomial.monomial n 1)) in
  coefficient_x21 (p * q^2) 21 = 97 :=
 by {
  let p := Polynomial.sum (Finset.range 21) (λ n : ℕ, Polynomial.monomial n 1),
  let q := Polynomial.sum (Finset.range 11) (λ n : ℕ, Polynomial.monomial n 1),
  rw [Polynomial.mul_assoc],
  have : Polynomial.coeff (p * q ^ 2) 21 = 97, sorry,
  exact this,
  }

end coefficient_of_x21_l107_107898


namespace smallest_prime_12_less_than_square_l107_107180

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107180


namespace conjugate_of_i_pow_2017_l107_107040

def i := Complex.I -- Assume i is the imaginary unit

theorem conjugate_of_i_pow_2017 :
  Complex.conj (i ^ 2017) = -i := by
sorry

end conjugate_of_i_pow_2017_l107_107040


namespace dog_dug_up_bones_l107_107069

theorem dog_dug_up_bones {current_bones started_bones dug_up_bones : ℕ} 
  (h_current : current_bones = 860) 
  (h_started : started_bones = 493) : 
  dug_up_bones = current_bones - started_bones :=
by
  -- Definitions based on conditions
  let current_bones := 860
  let started_bones := 493
  let dug_up_bones := current_bones - started_bones
  -- Statement to prove
  have : dug_up_bones = 367 := sorry
  exact this

end dog_dug_up_bones_l107_107069


namespace similar_triangles_heights_and_bases_l107_107608

theorem similar_triangles_heights_and_bases
  (similar : ∀ {α β : Type} [h₁ : has_size α] [h₂ : has_size β], α ~ β)
  (ratio_of_areas : Real := 1 / 9)
  (height_small : Real := 5)
  (base_small : Real := 6) :
  (∃ k: Real, k^2 = 9) →
  (∃ height_large : Real, ∃ base_large : Real,
    height_large = height_small * 3 ∧
    base_large = base_small * 3) → 
  height_large = 15 ∧ base_large = 18 :=
sorry

end similar_triangles_heights_and_bases_l107_107608


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107791

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107791


namespace thabo_books_l107_107101

variable (P F H : Nat)

theorem thabo_books :
  P > 55 ∧ F = 2 * P ∧ H = 55 ∧ H + P + F = 280 → P - H = 20 :=
by
  sorry

end thabo_books_l107_107101


namespace mix_solution_l107_107093

theorem mix_solution :
  ∀ (Vx Vy : ℝ) (Ax Ay Af : ℝ), 
  Vy = 200 ∧ 
  Ax = 0.10 ∧ 
  Ay = 0.30 ∧ 
  Af = 0.18 ∧ 
  0.10 * Vx + 0.30 * 200 = 0.18 * (Vx + 200) → 
  Vx = 300 :=
by 
  intros Vx Vy Ax Ay Af h,
  rcases h with ⟨hVy, hAx, hAy, hAf, eq⟩,
  sorry

end mix_solution_l107_107093


namespace smallest_prime_12_less_than_square_l107_107176

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107176


namespace cross_number_puzzle_sum_of_outlined_digits_l107_107599

noncomputable theory

def three_digit_powers_of_2 : set ℕ := {128, 256, 512}
def four_digit_powers_of_2 : set ℕ := {1024, 2048}
def powers_of_2 : set ℕ := three_digit_powers_of_2 ∪ four_digit_powers_of_2

def three_digit_powers_of_5 : set ℕ := {125, 625}
def four_digit_powers_of_5 : set ℕ := {3125}
def powers_of_5 : set ℕ := three_digit_powers_of_5 ∪ four_digit_powers_of_5

theorem cross_number_puzzle_sum_of_outlined_digits :
  (∃ m ∈ powers_of_2, ∃ n ∈ powers_of_5, (m = 256 ∨ m = 2048) ∧ (n = 625 ∨ n = 3125) ∧ (get_outlined_digit m + get_outlined_digit n = 4)) :=
sorry

def get_outlined_digit (n : ℕ) : ℕ :=
  if n = 128 then 1
  else if n = 256 then 2
  else if n = 512 then 5
  else if n = 1024 then 0
  else if n = 2048 then 0
  else if n = 625 then 2
  else if n = 3125 then 2
  else 0

end cross_number_puzzle_sum_of_outlined_digits_l107_107599


namespace expression_divisible_by_1897_l107_107083

theorem expression_divisible_by_1897 (n : ℕ) :
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end expression_divisible_by_1897_l107_107083


namespace apple_cost_is_2_l107_107488

def total_spent (hummus_cost chicken_cost bacon_cost vegetable_cost : ℕ) : ℕ :=
  2 * hummus_cost + chicken_cost + bacon_cost + vegetable_cost

theorem apple_cost_is_2 :
  ∀ (hummus_cost chicken_cost bacon_cost vegetable_cost total_money apples_cost : ℕ),
    hummus_cost = 5 →
    chicken_cost = 20 →
    bacon_cost = 10 →
    vegetable_cost = 10 →
    total_money = 60 →
    apples_cost = 5 →
    (total_money - total_spent hummus_cost chicken_cost bacon_cost vegetable_cost) / apples_cost = 2 :=
by
  intros
  sorry

end apple_cost_is_2_l107_107488


namespace minValue_equality_l107_107922

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (a + 3 * b) * (b + 3 * c) * (a * c + 3)

theorem minValue_equality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  minValue a b c = 48 :=
sorry

end minValue_equality_l107_107922


namespace playerB_second_shot_playerA_ith_shot_expected_shots_playerA_l107_107539

-- Define the conditions
def playerA_shooting_percentage : ℝ := 0.6
def playerB_shooting_percentage : ℝ := 0.8
def first_shot_probability : ℝ := 0.5

-- Prove that the probability player B takes the second shot is 0.6
theorem playerB_second_shot : 
  (first_shot_probability * (1 - playerA_shooting_percentage) + 
   first_shot_probability * playerB_shooting_percentage) = 0.6 :=
sorry

-- Prove that the probability player A takes the i-th shot is P_i
def P_i (i : ℕ) : ℝ := ⅓ + ⅙ * (⅖)^(i-1)
theorem playerA_ith_shot (i : ℕ) : P_i i = ⅓ + ⅙ * (⅖)^(i-1) :=
sorry

-- Prove that the expected number of times player A shoots in the first n shots is E(Y)
def E_Y (n : ℕ) : ℝ := (5 / 18) * (1 - (2 / 5)^n) + n / 3
theorem expected_shots_playerA (n : ℕ) : E_Y n = (5 / 18) * (1 - (2 / 5)^n) + n / 3 :=
sorry

end playerB_second_shot_playerA_ith_shot_expected_shots_playerA_l107_107539


namespace vertical_asymptote_sum_l107_107577

theorem vertical_asymptote_sum :
  (∀ x : ℝ, 4*x^2 + 6*x + 3 = 0 → x = -1 / 2 ∨ x = -1) →
  (-1 / 2 + -1) = -3 / 2 :=
by
  intro h
  sorry

end vertical_asymptote_sum_l107_107577


namespace gillian_total_spent_l107_107548

-- Definitions from the conditions
def sandi_initial_amount : ℕ := 600
def sandi_spent := sandi_initial_amount / 2
def gillian_spent := 3 * sandi_spent + 150

-- Proof statement
theorem gillian_total_spent : gillian_spent = 1050 := 
by
  unfold sandi_initial_amount
  unfold sandi_spent
  unfold gillian_spent
  sorry

end gillian_total_spent_l107_107548


namespace second_worker_load_time_l107_107275

theorem second_worker_load_time (T : ℝ) :
  (1 / 6) + (1 / T) = (11 / 30) → T = 5 :=
by
  assume h: (1 / 6) + (1 / T) = (11 / 30)
  sorry

end second_worker_load_time_l107_107275


namespace find_b_l107_107875

-- Define the parameters of the problem
variables (a b c : ℝ) (cosB : ℝ)

-- Assume the given conditions
axiom h1 : a = 2
axiom h2 : b + c = 7
axiom h3 : cosB = -1 / 4

-- State the goal to prove
theorem find_b : b = 4 :=
by
  have : a = 2 := h1
  have : b + c = 7 := h2
  have : cosB = -1 / 4 := h3
  sorry

end find_b_l107_107875


namespace binom_600_eq_1_l107_107318

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l107_107318


namespace cos_sum_simplified_l107_107962

theorem cos_sum_simplified :
  (Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)) = ((Real.sqrt 13 - 1) / 4) :=
by
  sorry

end cos_sum_simplified_l107_107962


namespace twenty_fifth_digit_in_sum_l107_107339

theorem twenty_fifth_digit_in_sum : 
  let a := (1 / 8 : ℝ)
  let b := (1 / 4 : ℝ)
  25th_digit_after_decimal(a + b) = 0 := 
by {
  sorry 
}

-- Auxiliary function definition:
def 25th_digit_after_decimal (x : ℝ) : ℕ := 
  let s := x.to_string 
  let parts := s.split_on '.'
  match parts with
  | [_] => 0  -- No decimal part implies all digits are 0
  | [_, dec] => if dec.length < 25 then 0 else dec[24] - '0'
  | _ => 0  -- Should never reach here 
  sorry

end twenty_fifth_digit_in_sum_l107_107339


namespace smallest_prime_12_less_than_perfect_square_l107_107205

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107205


namespace remainder_of_3_pow_102_mod_101_l107_107356

theorem remainder_of_3_pow_102_mod_101 : (3^102) % 101 = 9 :=
by
  sorry

end remainder_of_3_pow_102_mod_101_l107_107356


namespace monotonic_increase_interval_l107_107821

noncomputable def given_function (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) : ℝ → ℝ :=
  λ x, 2 * sin (ω * x + φ)

theorem monotonic_increase_interval
  (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π)
  (x1 x2 : ℝ) (hx : given_function ω φ hω hφ x1 = 2 ∧ given_function ω φ hω hφ x2 = 2)
  (min_dist : abs (x2 - x1) = π) :
  ∃ I, I = Ioc (-π/2) (-π/4) ∧ 
    ∀ x y, x ∈ I ∧ y ∈ I ∧ x < y → given_function ω φ hω hφ x < given_function ω φ hω hφ y :=
by
  sorry

end monotonic_increase_interval_l107_107821


namespace smallest_prime_12_less_than_square_l107_107183

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107183


namespace largest_number_smallest_number_l107_107716

/-- Define a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Define the property of a number being divisible by 6 -/
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

/-- Define the property of a number containing the digit 7 -/
def contains_digit_7 (n : ℕ) : Prop :=
  7 ∈ (to_digits 10 n)

/-- The largest three-digit number satisfying all conditions -/
theorem largest_number : ∃ n, is_three_digit n ∧ divisible_by_6 n ∧ contains_digit_7 n ∧ (∀ m, is_three_digit m ∧ divisible_by_6 m ∧ contains_digit_7 m → m ≤ n) ∧ n = 978 := 
by 
  sorry

/-- The smallest three-digit number satisfying all conditions -/
theorem smallest_number : ∃ n, is_three_digit n ∧ divisible_by_6 n ∧ contains_digit_7 n ∧ (∀ m, is_three_digit m ∧ divisible_by_6 m ∧ contains_digit_7 m → n ≤ m) ∧ n = 174 := 
by 
  sorry

end largest_number_smallest_number_l107_107716


namespace div_by_1897_l107_107082

theorem div_by_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end div_by_1897_l107_107082


namespace find_a_l107_107999

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

theorem find_a (a : ℝ) 
  (h₁ : ∃ x ∈ set.Icc (0 : ℝ) (4 : ℝ), f a x = 3)
  (h₂ : ∀ x ∈ set.Icc (0 : ℝ) (4 : ℝ), ∀ y ∈ set.Icc (0 : ℝ) (4 : ℝ), f a x ≤ f a y → x = y) :
  a = 3 :=
sorry

end find_a_l107_107999


namespace combined_weight_of_olivers_bags_l107_107941

theorem combined_weight_of_olivers_bags (w_james : ℕ) (w_oliver : ℕ) (w_combined : ℕ) 
  (h1 : w_james = 18) 
  (h2 : w_oliver = w_james / 6) 
  (h3 : w_combined = 2 * w_oliver) : 
  w_combined = 6 := 
by
  sorry

end combined_weight_of_olivers_bags_l107_107941


namespace polynomial_root_exists_l107_107116

theorem polynomial_root_exists (a b : ℤ) (A : ℤ → ℤ := λ x, x^2 + a * x + b)
  (h : ∀ p : ℕ, prime p → ∃ k : ℤ, p ∣ A k ∧ p ∣ A (k + 1)) :
  ∃ m : ℤ, A m = 0 ∧ A (m + 1) = 0 :=
sorry

end polynomial_root_exists_l107_107116


namespace students_no_A_l107_107469

theorem students_no_A
  (total_students : ℕ)
  (A_in_history : ℕ)
  (A_in_math : ℕ)
  (A_in_science : ℕ)
  (A_in_history_and_math : ℕ)
  (A_in_history_and_science : ℕ)
  (A_in_math_and_science : ℕ)
  (A_in_all_three : ℕ)
  (h_total_students : total_students = 40)
  (h_A_in_history : A_in_history = 10)
  (h_A_in_math : A_in_math = 15)
  (h_A_in_science : A_in_science = 8)
  (h_A_in_history_and_math : A_in_history_and_math = 5)
  (h_A_in_history_and_science : A_in_history_and_science = 3)
  (h_A_in_math_and_science : A_in_math_and_science = 4)
  (h_A_in_all_three : A_in_all_three = 2) :
  total_students - (A_in_history + A_in_math + A_in_science 
    - A_in_history_and_math - A_in_history_and_science - A_in_math_and_science 
    + A_in_all_three) = 17 := 
sorry

end students_no_A_l107_107469


namespace existence_of_specified_pairs_l107_107289

-- Definitions for the problem
variables {Boy Girl : Type}
variables (Danced : Boy → Girl → Prop)

-- Hypotheses based on the problem conditions
hypothesis no_boy_danced_with_all_girls :
  ∀ (b : Boy), ∃ (g : Girl), ¬ Danced b g
hypothesis each_girl_danced_with_at_least_one_boy :
  ∀ (g : Girl), ∃ (b : Boy), Danced b g

-- Statement of the math proof problem
theorem existence_of_specified_pairs :
  ∃ (g g' : Boy) (f f' : Girl), Danced g f ∧ ¬ Danced g f' ∧ Danced g' f' ∧ ¬ Danced g' f :=
sorry

end existence_of_specified_pairs_l107_107289


namespace sphere_surface_area_is_36pi_l107_107003

-- Define the edge length of the cube
def edge_length : ℝ := 2 * Real.sqrt 3

-- Define the diagonal of the cube
def cube_diagonal : ℝ := edge_length * Real.sqrt 3

-- Define the radius of the sphere circumscribing the cube
def sphere_radius : ℝ := cube_diagonal / 2

-- Define the surface area of the sphere
noncomputable def sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius^2

-- Prove that the surface area of the sphere is 36π
theorem sphere_surface_area_is_36pi : sphere_surface_area = 36 * Real.pi := by
  sorry

end sphere_surface_area_is_36pi_l107_107003


namespace hexagon_coloring_count_l107_107711

-- Defining the conditions
def has7Colors : Type := Fin 7

-- Hexagon vertices
inductive Vertex
| A | B | C | D | E | F

-- Adjacent vertices
def adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.B => true
| Vertex.B, Vertex.C => true
| Vertex.C, Vertex.D => true
| Vertex.D, Vertex.E => true
| Vertex.E, Vertex.F => true
| Vertex.F, Vertex.A => true
| _, _ => false

-- Non-adjacent vertices (diagonals)
def non_adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.C => true
| Vertex.A, Vertex.D => true
| Vertex.B, Vertex.D => true
| Vertex.B, Vertex.E => true
| Vertex.C, Vertex.E => true
| Vertex.C, Vertex.F => true
| Vertex.D, Vertex.F => true
| Vertex.E, Vertex.A => true
| Vertex.F, Vertex.A => true
| Vertex.F, Vertex.B => true
| _, _ => false

-- Coloring function
def valid_coloring (coloring : Vertex → has7Colors) : Prop :=
  (∀ v1 v2, adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2, non_adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2 v3, adjacent v1 v2 → adjacent v2 v3 → adjacent v1 v3 → coloring v1 ≠ coloring v3)

noncomputable def count_valid_colorings : Nat :=
  -- This is a placeholder for the count function
  sorry

theorem hexagon_coloring_count : count_valid_colorings = 21000 := 
  sorry

end hexagon_coloring_count_l107_107711


namespace sum_of_powers_l107_107597

theorem sum_of_powers (m : ℕ → ℕ) (b : ℕ → ℤ) (s : ℕ) (h_b_range : ∀ k, 1 ≤ k ∧ k ≤ s → b k = 2 ∨ b k = 1 ∨ b k = -1)
    (h_m_unique : ∀ i j, 1 ≤ i ∧ i ≤ s → 1 ≤ j ∧ j ≤ s → i ≠ j → m i ≠ m j)
    (h_m_ordered : ∀ k1 k2, 1 ≤ k1 ∧ k1 < k2 ∧ k2 ≤ s → m k1 > m k2)
    (h_eq : (∑ k in finset.range s, b k * 2 ^ m k) = 4030) :
  (∑ k in finset.range s, m k) = 55 := 
by
  sorry

end sum_of_powers_l107_107597


namespace calculate_C_rent_l107_107629

def oxenMonths (oxen : ℕ) (months : ℕ) := oxen * months

theorem calculate_C_rent
  (oxen_a : ℕ) (months_a : ℕ)
  (oxen_b : ℕ) (months_b : ℕ)
  (oxen_c : ℕ) (months_c : ℕ)
  (total_rent : ℕ)
  (hA : oxen_a = 10)
  (hMA : months_a = 7)
  (hB : oxen_b = 12)
  (hMB : months_b = 5)
  (hC : oxen_c = 15)
  (hMC : months_c = 3)
  (hRent : total_rent = 175) :
  let total_oxen_months := oxenMonths oxen_a months_a + oxenMonths oxen_b months_b + oxenMonths oxen_c months_c
  in (oxenMonths oxen_c months_c * total_rent) / total_oxen_months = 45 :=
by
  sorry

end calculate_C_rent_l107_107629


namespace polynomial_integer_roots_k_zero_l107_107637

theorem polynomial_integer_roots_k_zero :
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + 0) ∨
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + k)) →
  k = 0 :=
sorry

end polynomial_integer_roots_k_zero_l107_107637


namespace range_of_m_l107_107828

-- Using noncomputable as we are dealing with real numbers
noncomputable theory

open Set Real

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|x - m| < 1 ↔ (1/3 < x ∧ x < 1/2))) ↔ (-1/2 ≤ m ∧ m ≤ 4/3) := 
sorry

end range_of_m_l107_107828


namespace math_question_l107_107779

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l107_107779


namespace min_cars_necessary_l107_107007

theorem min_cars_necessary (n : ℕ) : ℕ :=
  if n = 5 then 6
  else if n = 8 then 10
  else sorry

example : min_cars_necessary 5 = 6 := by {
  unfold min_cars_necessary,
  exact if_pos rfl
}

example : min_cars_necessary 8 = 10 := by {
  unfold min_cars_necessary,
  rw if_neg (by norm_num),
  exact if_pos rfl
}

end min_cars_necessary_l107_107007


namespace sum_of_digits_N_eq_405_l107_107333

def N : ℕ := ∑ k in (finset.range 400).image (λ k, 10^(k + 1) - 1), k

-- The statement to prove
theorem sum_of_digits_N_eq_405 : (nat.digits 10 N).sum = 405 :=
by
  sorry

end sum_of_digits_N_eq_405_l107_107333


namespace rational_right_triangle_side_lengths_pythagorean_triples_l107_107509

open scoped Classical

theorem rational_right_triangle_side_lengths (a : ℚ)
  (b c : ℚ) (h₁ : a ≠ 0) (h₂ : c^2 = a^2 + b^2) : 
  ∃ (p q : ℚ), q ≠ 0 ∧ c - b = p / q ∧ 
  c + b = a^2 * q / p :=
by sorry

theorem pythagorean_triples (m : ℕ)
  (b c r p : ℕ) (h₁ : m ≠ 0) (h₂ : m^2 = p * r)
  (h₃ : b = (r - p) / 2) (h₄ : c = (r + p) / 2) 
  (hp : parity p = parity r) :
  m^2 + b^2 = c^2 :=
by sorry

end rational_right_triangle_side_lengths_pythagorean_triples_l107_107509


namespace smallest_prime_less_than_square_l107_107156

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107156


namespace tank_fill_time_with_leak_l107_107285

theorem tank_fill_time_with_leak 
  (pump_fill_time : ℕ) (leak_empty_time : ℕ) (effective_fill_time : ℕ)
  (hp : pump_fill_time = 5)
  (hl : leak_empty_time = 10)
  (he : effective_fill_time = 10) : effective_fill_time = 10 :=
by
  sorry

end tank_fill_time_with_leak_l107_107285


namespace jeff_total_distance_l107_107911

def mondayRunTime := 1 -- hour
def mondayPace := 6 -- mph

def tuesdayRunTime := 1 -- hour
def tuesdayPace := 7 -- mph

def wednesdayRunTime := 1 -- hour
def wednesdayPace := 8 -- mph

def thursdayRunTime := (40 : ℚ) / 60 -- hours
def thursdayPace := 7.5 -- mph

def fridayRunTime := (70 : ℚ) / 60 -- hours
def fridayPace := 9 -- mph

def mondayDistance := mondayRunTime * mondayPace
def tuesdayDistance := tuesdayRunTime * tuesdayPace
def wednesdayDistance := wednesdayRunTime * wednesdayPace
def thursdayDistance := thursdayRunTime * thursdayPace
def fridayDistance := fridayRunTime * fridayPace

def totalDistance := 
  mondayDistance + 
  tuesdayDistance + 
  wednesdayDistance + 
  thursdayDistance + 
  fridayDistance

theorem jeff_total_distance : totalDistance = 36.5 := by
  sorry

end jeff_total_distance_l107_107911


namespace smallest_prime_12_less_than_square_l107_107181

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107181


namespace Connie_savings_l107_107695

theorem Connie_savings (cost_of_watch : ℕ) (extra_needed : ℕ) (saved_amount : ℕ) : 
  cost_of_watch = 55 → 
  extra_needed = 16 → 
  saved_amount = cost_of_watch - extra_needed → 
  saved_amount = 39 := 
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end Connie_savings_l107_107695


namespace f1_is_geometric_sequence_preserving_f2_is_not_geometric_sequence_preserving_f3_is_geometric_sequence_preserving_f4_is_not_geometric_sequence_preserving_geometric_sequence_preserving_functions_l107_107656

-- Definitions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n * a (n + 2) = (a (n + 1))^2

def is_geometric_sequence_preserving (f : ℝ → ℝ) : Prop :=
∀ (a : ℕ → ℝ), is_geometric_sequence a → is_geometric_sequence (λ n, f (a n))

-- Conditions
def f1 (x : ℝ) := x^2
def f2 (x : ℝ) := x^2 + 1
def f3 (x : ℝ) := real.sqrt (|x|)
def f4 (x : ℝ) := real.log (|x|)

-- Theorems to prove
theorem f1_is_geometric_sequence_preserving : is_geometric_sequence_preserving f1 := sorry
theorem f2_is_not_geometric_sequence_preserving : ¬ is_geometric_sequence_preserving f2 := sorry
theorem f3_is_geometric_sequence_preserving : is_geometric_sequence_preserving f3 := sorry
theorem f4_is_not_geometric_sequence_preserving : ¬ is_geometric_sequence_preserving f4 := sorry

-- The main proof problem
theorem geometric_sequence_preserving_functions :
  {f | f = f1 ∨ f = f3} = {f1, f3} := sorry

end f1_is_geometric_sequence_preserving_f2_is_not_geometric_sequence_preserving_f3_is_geometric_sequence_preserving_f4_is_not_geometric_sequence_preserving_geometric_sequence_preserving_functions_l107_107656


namespace smallest_prime_12_less_than_square_l107_107179

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107179


namespace smallest_prime_12_less_than_perfect_square_l107_107209

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107209


namespace smallest_prime_12_less_perfect_square_l107_107147

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107147


namespace find_sum_of_exponents_l107_107368

variables (x y : ℝ)

theorem find_sum_of_exponents (h1 : 2^x = 3) (h2 : log2 (4/3) = y) : x + y = 2 :=
sorry

end find_sum_of_exponents_l107_107368


namespace initial_horses_to_cows_ratio_l107_107073

theorem initial_horses_to_cows_ratio (H C : ℕ) (h₁ : (H - 15) / (C + 15) = 13 / 7) (h₂ : H - 15 = C + 45) :
  H / C = 4 / 1 := 
sorry

end initial_horses_to_cows_ratio_l107_107073


namespace mx_eq_my_l107_107473

theorem mx_eq_my 
  (A B C : Point) 
  (h_scalene: scalene_triangle A B C) 
  (M : Point) 
  (h_mid_M : midpoint BC M) 
  (P : Point) 
  (h_inter_P : closest_intersection_of_ray_incircle A M P) 
  (Q : Point) 
  (h_inter_Q : farthest_intersection_of_ray_excircle A M Q) 
  (X : Point) 
  (h_tangent_incircle : tangent_intersection_incircle P X) 
  (Y : Point) 
  (h_tangent_excircle : tangent_intersection_excircle Q Y) : 
  segment_length M X = segment_length M Y :=
by
  sorry

end mx_eq_my_l107_107473


namespace find_a_l107_107870

theorem find_a (a : ℝ) :
  (let expr := (x + a) * (1/x + 2*x)^5 in
   let term := (2 : ℝ)^4 * a * nat.choose 5 4 in
   term = 20 →
   a = 1/4) :=
sorry

end find_a_l107_107870


namespace smallest_prime_less_than_perfect_square_l107_107159

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107159


namespace Q_100_distance_l107_107649

noncomputable def θ : Complex := Complex.exp (Complex.pi * Complex.I / 4)

noncomputable def Q (k : ℕ) : Complex :=
  if k = 0 then 0 else
  (2 * θ^(0 : ℕ) + ∑ n in finset.range k, 2*n + 1 * θ^(n : ℕ))

-- Prove the statement
theorem Q_100_distance : Complex.abs (Q 100) = (Real.sqrt (20410 + 10205 * Real.sqrt 2)) / 2 :=
sorry

end Q_100_distance_l107_107649


namespace acute_triangle_side_range_l107_107402

theorem acute_triangle_side_range {x : ℝ} (h : ∀ a b c : ℝ, a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2) :
  2 < 4 ∧ 4 < x → (2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 5) :=
  sorry

end acute_triangle_side_range_l107_107402


namespace milk_production_and_math_students_sampling_l107_107887

def milk_production_line_situation := "In a milk production line, a bag is sampled for inspection every 30 minutes."
def math_students_selection := "From a group of 30 math enthusiasts in a middle school, 3 students are selected to understand their study load."

theorem milk_production_and_math_students_sampling :
  (milk_production_line_situation = "In a milk production line, a bag is sampled for inspection every 30 minutes.") →
  (math_students_selection = "From a group of 30 math enthusiasts in a middle school, 3 students are selected to understand their study load.") →
  (∃ (S1 S2 : String), S1 = "systematic sampling" ∧ S2 = "simple random sampling") :=
by
  intro h1 h2
  exists "systematic sampling"
  exists "simple random sampling"
  split
  case left => exact h1
  case right => exact h2

end milk_production_and_math_students_sampling_l107_107887


namespace sequence_unique_integers_l107_107055

theorem sequence_unique_integers 
  (a : ℕ → ℤ)
  (h_inf_pos : ∀ N : ℕ, ∃ n : ℕ, a n > N)
  (h_inf_neg : ∀ N : ℕ, ∃ n : ℕ, a n < -N)
  (h_distinct_mod : ∀ n : ℕ, ∀ i j : ℕ, i < n ∧ j < n ∧ i ≠ j → (a i % n) ≠ (a j % n))
  : ∀ z : ℤ, ∃! n : ℕ, a n = z :=
begin
  -- proof goes here
  sorry
end

end sequence_unique_integers_l107_107055


namespace combined_weight_of_olivers_bags_l107_107943

-- Define the weights and relationship between the weights
def weight_james_bag : ℝ := 18
def ratio_olivers_to_james : ℝ := 1 / 6
def weight_oliver_one_bag : ℝ := weight_james_bag * ratio_olivers_to_james
def number_of_oliver_bags : ℝ := 2

-- The proof problem statement: proving the combined weight of both Oliver's bags
theorem combined_weight_of_olivers_bags :
  number_of_oliver_bags * weight_oliver_one_bag = 6 := by
  sorry

end combined_weight_of_olivers_bags_l107_107943


namespace range_of_a_range_of_a_eq_l107_107384

def set_A := {x : ℝ | abs (x - 2) < 3}
def set_B (a : ℝ) := {x : ℝ | 2^x > 2^a}

theorem range_of_a (a : ℝ) (h : set_A ⊆ set_B a) : a ≤ -1 := 
sorry

theorem range_of_a_eq : {a : ℝ | range_of_a a (by sorry)} = {a : ℝ | a ≤ -1} :=
sorry

end range_of_a_range_of_a_eq_l107_107384


namespace cos_minus_sin_second_quadrant_l107_107405

theorem cos_minus_sin_second_quadrant (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : sin α + cos α = 1 / 5) :
  cos α - sin α = -7 / 5 := 
sorry

end cos_minus_sin_second_quadrant_l107_107405


namespace hyperbola_asymptotes_l107_107481

theorem hyperbola_asymptotes (b : ℝ) (h : b > 0) (p : ℝ × ℝ) 
  (hp1 : p.1 = 3) (hp2 : p.2 = 4) (hyp : p.1^2 - (p.2^2) / (b^2) = 1): 
  ∃ c : ℝ, b = Real.sqrt 2 ∧ ∀ x : ℝ, (c = Real.sqrt 2) ∧ (y = c * x ∨ y = -c * x) where
    y := x * ↑c
:= sorry

end hyperbola_asymptotes_l107_107481


namespace find_a10_b10_l107_107956

theorem find_a10_b10 (a b : ℝ) (h₁ : a^5 + b^5 = 3) (h₂ : a^{15} + b^{15} = 9) : a^{10} + b^{10} = 5 := 
sorry

end find_a10_b10_l107_107956


namespace height_of_box_l107_107696

-- Define box dimensions
def box_length := 6
def box_width := 6

-- Define spherical radii
def radius_large := 3
def radius_small := 2

-- Define coordinates
def box_volume (h : ℝ) : Prop :=
  ∃ (z : ℝ), z = 2 + Real.sqrt 23 ∧ 
  z + radius_large = h

theorem height_of_box (h : ℝ) : box_volume h ↔ h = 5 + Real.sqrt 23 := by
  sorry

end height_of_box_l107_107696


namespace perpendicular_miquel_l107_107808

variables {P Q M O : Point}
variables {l₁ l₂ l₃ l₄ : Line}
variables {A B C D : Point}
variables (h1 : l₁ ≠ l₂) (h2 : l₁ ≠ l₃) (h3 : l₁ ≠ l₄)
variables (h4 : l₃ ≠ l₂) (h5 : l₃ ≠ l₄) (h6 : l₂ ≠ l₄)
variables (h_int1 : l₁ ∩ l₂ = A) (h_int2 : l₁ ∩ l₃ = B) (h_int3 : l₁ ∩ l₄ = C)
variables (h_int4 : l₂ ∩ l₃ = D) (h_int5 : l₂ ∩ l₄ = P) (h_int6 : l₃ ∩ l₄ = Q)
variables (circle_center : circle O (radius O) {A, B, C, D})
variables (Miquel_point : MiquelPoint l₁ l₂ l₃ l₄ M)

theorem perpendicular_miquel
    (hM1 : P ≠ Q) 
    (hM2 : P ∈ Miquel_point) 
    (hM3 : Q ∈ Miquel_point) 
    (hP : ∀ l ∈ circle_center, l) :
  (line_through P Q M) ∧ (perpendicular (line_through P Q) (line_through O M)) :=
begin
  sorry
end

end perpendicular_miquel_l107_107808


namespace proof_problem_l107_107780

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l107_107780


namespace part_a_sets_eq_part_b_max_formula_part_b_formula_correct_l107_107698

def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 2 = 0 then 2 * f (n / 2)
  else (n / 2) + 2 * f (n / 2)

def L := {n : ℕ | f n < f (n + 1)}
def E := {n : ℕ | f n = f (n + 1)}
def G := {n : ℕ | f n > f (n + 1)}

theorem part_a_sets_eq :
  L = {n | ∃ k, n = 2 * k ∧ k > 0} ∧
  E = {0} ∪ {n | ∃ k, n = 4 * k + 1} ∧
  G = {n | ∃ k, n = 4 * k + 3} :=
sorry

def a_k (k : ℕ) := Nat.rec_on k 0 (λ k ak, 2 * ak + 2^(k))

theorem part_b_max_formula (k : ℕ) :
  ∀ n, n ≤ 2^k → f n ≤ a_k k :=
sorry

theorem part_b_formula_correct (k : ℕ) :
  a_k k = k * 2^(k-1) - 2^k + 1 :=
sorry

end part_a_sets_eq_part_b_max_formula_part_b_formula_correct_l107_107698


namespace both_carpenters_complete_job_in_2_5_days_l107_107652

/-- Define the work rates of the two carpenters -/
def first_carpenter_work_rate := 1 / 5
def second_carpenter_work_rate := 1 / 5
def combined_work_rate := first_carpenter_work_rate + second_carpenter_work_rate

/-- Define the total work to be done -/
def total_work := 1

/-- Prove that the job is completed in 2.5 days -/
theorem both_carpenters_complete_job_in_2_5_days : 
  (combined_work_rate * 2.5 = total_work) := by
  sorry

end both_carpenters_complete_job_in_2_5_days_l107_107652


namespace lynne_total_spending_l107_107524

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end lynne_total_spending_l107_107524


namespace min_expenditure_l107_107234

theorem min_expenditure :
  ∀ (v : ℝ), ∃ E : ℝ, 
  (∀ v > 0, 
    (E = 50/v * 500 + (1/2500) * v * 50 * 500 + 50 * 500/v) 
    → E = 1000) :=
begin 
  sorry 
end

end min_expenditure_l107_107234


namespace max_lessons_l107_107590

-- Declaring noncomputable variables for the number of shirts, pairs of pants, and pairs of shoes.
noncomputable def s : ℕ := sorry
noncomputable def p : ℕ := sorry
noncomputable def b : ℕ := sorry

lemma conditions_satisfied :
  2 * (s + 1) * p * b = 2 * s * p * b + 36 ∧
  2 * s * (p + 1) * b = 2 * s * p * b + 72 ∧
  2 * s * p * (b + 1) = 2 * s * p * b + 54 ∧
  s * p * b = 27 ∧
  s * b = 36 ∧
  p * b = 18 := by
  sorry

theorem max_lessons : (2 * s * p * b) = 216 :=
by
  have h := conditions_satisfied
  sorry

end max_lessons_l107_107590


namespace monica_expected_winnings_l107_107939

def monica_die_winnings : List ℤ := [2, 3, 5, 7, 0, 0, 0, -4]

def expected_value (values : List ℤ) : ℚ :=
  (List.sum values) / (values.length : ℚ)

theorem monica_expected_winnings :
  expected_value monica_die_winnings = 1.625 := by
  sorry

end monica_expected_winnings_l107_107939


namespace value_of_a_l107_107837

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (1 + x) + log a (3 - x)

variable {a : ℝ}

axiom a_positive_and_not_one : 0 < a ∧ a ≠ 1

theorem value_of_a (h_min : (∀ x, f a x ≥ -2) ∧ (∃ x, f a x = -2)) : a = 1 / 2 := 
sorry

end value_of_a_l107_107837


namespace range_of_m_l107_107516

theorem range_of_m (a c m : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x ∈ Icc (0 : ℝ) 1, 2 * a * (x - 1) < 0)
  (h₂ : f m = a * m ^ 2 - 2 * a * m + c) (h₃ : f 0 = c) : 0 ≤ m ∧ m ≤ 2 :=
by
  -- Proof goes here
  sorry

def f (x : ℝ) : ℝ := a * x ^ 2 - 2 * a * x + c

end range_of_m_l107_107516


namespace consecutive_integers_squares_l107_107364

theorem consecutive_integers_squares (x : ℤ) :
  (x^2 + (x + 1)^2 + (x + 2)^2 = (x + 3)^2 + (x + 4)^2) →
  ({x, x + 1, x + 2, x + 3, x + 4} = {-2, -1, 0, 1, 2} ∨ {x, x + 1, x + 2, x + 3, x + 4} = {10, 11, 12, 13, 14}) :=
by
  sorry

end consecutive_integers_squares_l107_107364


namespace math_question_l107_107777

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l107_107777


namespace assignment_ways_l107_107681

structure Assignment :=
  (teachers : List String)
  (classes : List String)

def conditions_satisfied (a : Assignment) : Prop :=
  a.teachers = ["A", "B", "C", "D"] ∧
  a.classes = ["class1", "class2", "class3"] ∧
  (∀ c, ∃ t, t ∈ a.teachers ∧ t ∈ c) ∧
  ¬ (("A" ∈ c ∧ "B" ∈ c) for any c in a.classes)

theorem assignment_ways :
  ∃ (a : Assignment), conditions_satisfied a ∧ (number_of_ways_to_assign a = 30) :=
sorry

end assignment_ways_l107_107681


namespace village_assistants_selection_l107_107282

theorem village_assistants_selection :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let A := 1
  let B := 2
  let C := 3
  let total_ways := Nat.choose 9 3 - Nat.choose 7 3
  total_ways = 49 :=
by
  sorry

end village_assistants_selection_l107_107282


namespace not_mutually_exclusive_option_D_l107_107672

-- Definitions for mutually exclusive events
def mutually_exclusive (event1 event2 : Prop) : Prop := ¬ (event1 ∧ event2)

-- Conditions as given in the problem
def eventA1 : Prop := True -- Placeholder for "score is greater than 8"
def eventA2 : Prop := True -- Placeholder for "score is less than 6"

def eventB1 : Prop := True -- Placeholder for "90 seeds germinate"
def eventB2 : Prop := True -- Placeholder for "80 seeds germinate"

def eventC1 : Prop := True -- Placeholder for "pass rate is higher than 70%"
def eventC2 : Prop := True -- Placeholder for "pass rate is 70%"

def eventD1 : Prop := True -- Placeholder for "average score is not lower than 90"
def eventD2 : Prop := True -- Placeholder for "average score is not higher than 120"

-- Lean proof statement
theorem not_mutually_exclusive_option_D :
  mutually_exclusive eventA1 eventA2 ∧
  mutually_exclusive eventB1 eventB2 ∧
  mutually_exclusive eventC1 eventC2 ∧
  ¬ mutually_exclusive eventD1 eventD2 :=
sorry

end not_mutually_exclusive_option_D_l107_107672


namespace min_abs_expression_l107_107137

theorem min_abs_expression {x : ℝ} (h : ∀ x, |x - 4| + |x + 5| + |some_expression| ≥ 10) :
  ∃ some_expression, |some_expression| = 1 :=
begin
  sorry
end

end min_abs_expression_l107_107137


namespace present_age_of_q_is_40_l107_107232

noncomputable def present_age_of_q : ℕ := 
  let P := ∃ Q, P = 3 * (Q - (P - Q)) ∧ P + Q = 100 → Q
in 40

-- The statement means that given the conditions, prove the expected value of Q.
theorem present_age_of_q_is_40 : ∃ P Q : ℕ, P = 3 * (Q - (P - Q)) ∧ P + Q = 100 ∧ Q = present_age_of_q :=
  sorry

end present_age_of_q_is_40_l107_107232


namespace smallest_prime_12_less_perfect_square_l107_107145

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107145


namespace median_eq_range_le_l107_107747

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l107_107747


namespace smallest_prime_12_less_than_square_l107_107177

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107177


namespace smallest_prime_12_less_than_perfect_square_l107_107204

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107204


namespace values_of_m_l107_107060

theorem values_of_m (m n : ℕ) (hmn : m * n = 900) (hm: m > 1) (hn: n ≥ 1) : 
  (∃ (k : ℕ), ∀ (m : ℕ), (1 < m ∧ (900 / m) ≥ 1 ∧ 900 % m = 0) ↔ k = 25) :=
sorry

end values_of_m_l107_107060


namespace median_equality_and_range_inequality_l107_107767

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l107_107767


namespace no_nat_number_divided_by_sum_of_digits_gives_2011_l107_107341

theorem no_nat_number_divided_by_sum_of_digits_gives_2011 :
  ¬ ∃ n s : ℕ, (n = 2011 * s + 2011) ∧ (s = natDigitsSum n) := 
sorry

end no_nat_number_divided_by_sum_of_digits_gives_2011_l107_107341


namespace back_option_e_invalid_l107_107270

-- Definitions and conditions based on the problem statement
inductive Hole : Type
| A | B | C | D | E | F

def front_sequence : List Hole := [Hole.A, Hole.B, Hole.C, Hole.D, Hole.E, Hole.F]

-- Possible back configurations (a) to (e), encoded as example sequences
def back_option_a : List Hole := [...]
def back_option_b : List Hole := [...]
def back_option_c : List Hole := [...]
def back_option_d : List Hole := [...]
def back_option_e : List Hole := [...]

-- Condition to match front and back configuration (this is hypothetical based on interpretation)
def matches_front (back_sequence : List Hole) : Prop :=
  -- Define how back_sequence should correspond to front_sequence
  sorry  -- Definition omitted for brevity

-- The statement to prove that option (e) is invalid
theorem back_option_e_invalid : ¬ matches_front back_option_e :=
sorry

end back_option_e_invalid_l107_107270


namespace cannot_partition_1_to_89_l107_107908

theorem cannot_partition_1_to_89 : ¬ (∃ (groups : list (list ℕ)),
  (∀ group ∈ groups, 
    4 ≤ group.length ∧ 
    (∃ x ∈ group, x = (group.sum - x))) ∧ 
  (list.range 89).erase 0 = groups.join) :=
by
  sorry

end cannot_partition_1_to_89_l107_107908


namespace can_all_cross_l107_107254

open Nat

def AliBaba : Type := Unit
def Thief : Type := Unit

structure Friends (A : AliBaba) (T : Thief) : Prop :=
(friends : Prop)

-- There are Ali-Baba and 40 thieves
constant A : AliBaba
constant T : Fin 40 → Thief

-- Conditions given:
-- The boat can carry either 2 or 3 people
def boat_carry (k : Nat) : Prop := k = 2 ∨ k = 3

-- No one is allowed to cross alone
def valid_group (k : Nat) : Prop := 2 ≤ k ∧ k ≤ 3

-- Every two adjacent people in the line are friends
axiom adjacent_friendship (i : Fin 39) : Friends (T i) (T (i + 1))

-- Ali-Baba is also friends with the thief standing two places behind him
axiom alibaba_friendship : Friends A (T 2)

-- Prove they all can cross
theorem can_all_cross :
  ∃ f : list (list (AliBaba ⊕ Thief)), 
    (∀ grp ∈ f, valid_group grp.length) ∧ 
    (∀ grp ∈ f, ∀ x ∈ grp, ∀ y ∈ grp, x ≠ y → ∃ (i j : (AliBaba ⊕ Thief)), Friends i j) ∧
    -- Initial condition
    (∀ i, A ∈ initial_island i ∧ ∀ i, T i ∈ initial_island i) ∧
    -- Final condition
    (∀ i, A ∈ final_island i ∧ ∀ i, T i ∈ final_island i) 
    → true :=
sorry

end can_all_cross_l107_107254


namespace total_students_yellow_green_blue_l107_107905

def first_class := 200
def first_class_blue := 0.20 * first_class
def first_class_remaining := first_class - first_class_blue
def first_class_yellow := 0.20 * first_class_remaining
def first_class_green := 0.25 * first_class_remaining

def second_class := 150
def second_class_blue := 0.15 * second_class
def second_class_remaining := second_class - second_class_blue
def second_class_yellow := 0.25 * second_class_remaining
def second_class_green := 0.20 * second_class_remaining

def third_class := 250
def third_class_blue := 0.30 * third_class
def third_class_remaining := third_class - third_class_blue
def third_class_yellow := 0.15 * third_class_remaining
def third_class_green := 0.30 * third_class_remaining

def total_yellow := first_class_yellow + second_class_yellow + third_class_yellow
def total_green := first_class_green + second_class_green + third_class_green
def total_blue := first_class_blue + second_class_blue + third_class_blue

theorem total_students_yellow_green_blue :
  total_yellow + total_green + total_blue = 346 := 
by
  sorry

end total_students_yellow_green_blue_l107_107905


namespace log_eqn_l107_107864

theorem log_eqn (a b : ℝ) (h1 : a = (Real.log 400 / Real.log 16))
                          (h2 : b = Real.log 20 / Real.log 2) : a = (1/2) * b :=
sorry

end log_eqn_l107_107864


namespace log_sum_real_coeffs_l107_107495

theorem log_sum_real_coeffs (T : ℝ) (hT : T = (1+1i)^(2024).re + (1-1i)^(2024).re) : 
  log 2 T = 1012 :=
  sorry

end log_sum_real_coeffs_l107_107495


namespace factor_q_changed_l107_107871

noncomputable def q (w f z : ℝ) : ℝ := 5 * w / (4 * v * f * (z^2))

theorem factor_q_changed (w f z v : ℝ) (Hf : f ≠ 0) (Hz : z ≠ 0) :
  let new_w := 4 * w,
      new_f := 2 * f,
      factor_z := 0.2222222222222222 in
  (q new_w new_f z / q w f z) = 0.4444444444444444 :=
by
  sorry

end factor_q_changed_l107_107871


namespace binomial_600_600_l107_107305

theorem binomial_600_600 : nat.choose 600 600 = 1 :=
by
  -- Given the condition that binomial coefficient of n choose n is 1 for any non-negative n
  have h : ∀ n : ℕ, nat.choose n n = 1 := sorry
  -- Applying directly to the specific case n = 600
  exact h 600

end binomial_600_600_l107_107305


namespace median_equality_and_range_inequality_l107_107766

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l107_107766


namespace sum_of_super_cool_rectangle_areas_l107_107263

def is_super_cool_rectangle (a b : ℕ) : Prop :=
  a * b = 3 * (a + b)

def possible_areas (a b : ℕ) : set ℕ :=
  { area | is_super_cool_rectangle a b ∧ area = a * b }

theorem sum_of_super_cool_rectangle_areas : 
  ∃ (S : set ℕ), 
    (∀ (a b : ℕ), is_super_cool_rectangle a b → a * b ∈ S) ∧ 
    S = {48, 36} ∧ 
    S.sum = 84 :=
by
  sorry

end sum_of_super_cool_rectangle_areas_l107_107263


namespace second_investment_value_l107_107646

theorem second_investment_value
  (a : ℝ) (r1 r2 rt : ℝ) (x : ℝ)
  (h1 : a = 500)
  (h2 : r1 = 0.07)
  (h3 : r2 = 0.09)
  (h4 : rt = 0.085)
  (h5 : r1 * a + r2 * x = rt * (a + x)) :
  x = 1500 :=
by 
  -- The proof will go here
  sorry

end second_investment_value_l107_107646


namespace max_cannot_be_expressed_l107_107056

theorem max_cannot_be_expressed (a b c : ℕ) (h1 : Nat.coprime a b) (h2 : Nat.coprime b c) (h3 : Nat.coprime a c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬ ∃ x y z : ℕ, 2 * a * b * c - a * b - b * c - c * a = x * b * c + y * c * a + z * a * b :=
by
  sorry

end max_cannot_be_expressed_l107_107056


namespace smallest_prime_12_less_than_perfect_square_l107_107215

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107215


namespace distance_from_complex_point_to_origin_l107_107896

noncomputable def complex_distance_to_origin (z : ℂ) : ℝ :=
  complex.abs z

theorem distance_from_complex_point_to_origin :
  complex_distance_to_origin (-2 * complex.I + 1) = real.sqrt 5 :=
by
  have z : ℂ := -2 * complex.I + 1
  have coords : ℂ → ℝ × ℝ := λ w, (w.re, w.im)
  have dist_formula : ℝ × ℝ → ℝ := λ p, real.sqrt (p.1 ^ 2 + p.2 ^ 2)
  change dist_formula (coords z) = real.sqrt 5
  sorry

end distance_from_complex_point_to_origin_l107_107896


namespace proposition_p_and_q_false_l107_107057

-- Definitions of propositions p and q
def p : Prop := ∃ T > 0, ∀ x, sin (2 * (x + T)) = sin (2 * x) ∧ T = π / 2
def q : Prop := ∀ k : ℤ, ∀ x, cos (k * π + (π / 2)) = cos (π / 2 - x)

-- The theorem stating that p ∧ q is false
theorem proposition_p_and_q_false : ¬ (p ∧ q) :=
by
  sorry

end proposition_p_and_q_false_l107_107057


namespace sum_of_other_endpoint_l107_107538

theorem sum_of_other_endpoint (a b : ℝ) : 
  let A := (1 : ℝ, -2 : ℝ)
  let M := (5 : ℝ, 4 : ℝ)
  let (x, y) := (2 * 5 - 1, 2 * 4 + 2)
  x + y = 19 := 
by 
    sorry

end sum_of_other_endpoint_l107_107538


namespace alice_solitaire_moves_l107_107917

theorem alice_solitaire_moves (r : ℚ) (hr : r > 1) :
  (∃ seq : List ℤ, alice_can_move_red_to_1_in_most_2021_moves r seq) ↔
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ 1010 ∧ r = (m + 1) / m := by
  sorry

-- Here alice_can_move_red_to_1_in_most_2021_moves would be a predicate or function 
-- that verifies if a given sequence of moves can bring the red bead to 1 in at most 2021 moves.

end alice_solitaire_moves_l107_107917


namespace binom_600_600_l107_107324

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l107_107324


namespace train_length_l107_107272

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (platform_length_m : ℝ) (train_length_m : ℝ) : 
  speed_km_hr = 45 →
  time_sec = 40 →
  platform_length_m = 140 →
  train_length_m = 360 :=
begin
  assume h1 h2 h3,
  -- Convert speed from km/hr to m/s
  let speed_m_s := speed_km_hr * (1000 / 3600),
  have speed_conversion : speed_m_s = 12.5, from sorry,
  -- Calculate the total distance covered in the given time
  let total_distance := speed_m_s * time_sec,
  have distance_calculation : total_distance = 500, from sorry,
  -- Use the given platform length to find the train length
  let calc_train_length := total_distance - platform_length_m,
  have train_length_result : calc_train_length = 360, from sorry,
  exact train_length_result,
end

end train_length_l107_107272


namespace missing_number_approximately_1400_l107_107975

theorem missing_number_approximately_1400 :
  ∃ x : ℤ, x * 54 = 75625 ∧ abs (x - Int.ofNat (75625 / 54)) ≤ 1 :=
by
  sorry

end missing_number_approximately_1400_l107_107975


namespace f_extreme_points_sum_l107_107416

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (1 / (2 * x)) - a * x^2 + x

theorem f_extreme_points_sum {a x1 x2 : ℝ} 
  (h_a_pos : 0 < a)
  (h_a_lt : a < 1 / 8)
  (h_x1 : x1 = (1 - Real.sqrt (1 - 8 * a)) / (4 * a))
  (h_x2 : x2 = (1 + Real.sqrt (1 - 8 * a)) / (4 * a))
  (h_f_deriv_zero : ∀ x, f x a = log (1 / (2 * x)) - a * x^2 + x) :
  f x1 a + f x2 a > 3 - 4 * log 2 :=
sorry

end f_extreme_points_sum_l107_107416


namespace water_speed_l107_107259

theorem water_speed (v : ℝ) 
  (still_water_speed : ℝ := 4)
  (distance : ℝ := 10)
  (time : ℝ := 5)
  (effective_speed : ℝ := distance / time) 
  (h : still_water_speed - v = effective_speed) :
  v = 2 :=
by
  sorry

end water_speed_l107_107259


namespace prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l107_107233

variable {A : Set Int}

-- Assuming set A is closed under subtraction
axiom A_closed_under_subtraction : ∀ x y, x ∈ A → y ∈ A → x - y ∈ A
axiom A_contains_4 : 4 ∈ A
axiom A_contains_9 : 9 ∈ A

theorem prove_0_in_A : 0 ∈ A :=
sorry

theorem prove_13_in_A : 13 ∈ A :=
sorry

theorem prove_74_in_A : 74 ∈ A :=
sorry

theorem prove_A_is_Z : A = Set.univ :=
sorry

end prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l107_107233


namespace remainder_of_number_mod_1000_l107_107919

-- Definitions according to the conditions
def num_increasing_8_digit_numbers_with_zero : ℕ := Nat.choose 17 8

-- The main statement to be proved
theorem remainder_of_number_mod_1000 : 
  (num_increasing_8_digit_numbers_with_zero % 1000) = 310 :=
by
  sorry

end remainder_of_number_mod_1000_l107_107919


namespace tetrahedron_orthocenter_and_incircle_radius_l107_107514

theorem tetrahedron_orthocenter_and_incircle_radius 
  (O A B C : Type) [inner_product_space ℝ O] 
  [inner_product_space ℝ A] 
  [inner_product_space ℝ B] 
  [inner_product_space ℝ C]
  (OA OB OC : ℝ)
  (orthocenter : O = 0)
  (H : O = H)
  (r ABC : ℝ)
  (h1 : OA * OB * OC = r (1 + sqrt 3)) :
  ∃ (OH : ℝ), OH ≤ r * (1 + sqrt 3) :=
by sorry

end tetrahedron_orthocenter_and_incircle_radius_l107_107514


namespace proof_problem_l107_107785

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l107_107785


namespace symmetric_to_y_axis_circle_l107_107110

open Real

-- Definition of the original circle's equation
def original_circle (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 3

-- Definition of the symmetric circle's equation with respect to the y-axis
def symmetric_circle (x y : ℝ) : Prop := x^2 + 2 * x + y^2 = 3

-- Theorem stating that the symmetric circle has the given equation
theorem symmetric_to_y_axis_circle (x y : ℝ) : 
  (symmetric_circle x y) ↔ (original_circle ((-x) - 2) y) :=
sorry

end symmetric_to_y_axis_circle_l107_107110


namespace polynomial_remainder_l107_107236

theorem polynomial_remainder 
  (a b c d : ℝ) (h : a ≠ b)
  (f : ℝ → ℝ) 
  (hfa : f a = c) 
  (hfb : f b = d) :
  ∃ m n : ℝ, m = (c - d) / (a - b) ∧ n = (a * d - b * c) / (a - b) ∧ 
  (∀ x : ℝ, f x mod (x - a)*(x - b) = m * x + n) := 
sorry

end polynomial_remainder_l107_107236


namespace constant_abs_difference_l107_107046

variable (a : ℕ → ℝ)

-- Define the condition for the recurrence relation
def recurrence_relation : Prop := ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n

-- State the theorem
theorem constant_abs_difference (h : recurrence_relation a) : ∃ C : ℝ, ∀ n ≥ 2, |(a n)^2 - (a (n-1)) * (a (n+1))| = C :=
    sorry

end constant_abs_difference_l107_107046


namespace determine_peter_and_liar_l107_107284

structure Brothers where
  names : Fin 2 → String
  tells_truth : Fin 2 → Bool -- true if the brother tells the truth, false if lies
  (unique_truth_teller : ∃! (i : Fin 2), tells_truth i)
  (one_is_peter : ∃ (i : Fin 2), names i = "Péter")

theorem determine_peter_and_liar (B : Brothers) : 
  ∃ (peter liar : Fin 2), B.names peter = "Péter" ∧ B.tells_truth liar = false ∧
    ∀ (p q : Fin 2), B.names p = "Péter" → B.tells_truth q = false → p = peter ∧ q = liar :=
by
  sorry

end determine_peter_and_liar_l107_107284


namespace equal_dm_dl_l107_107907

-- Definitions of needed points and their properties
variables (A B C P M L D : Point)
variables [PlaneGeometry] -- Assuming a typeclass for plane geometry

-- Conditions
variable (hPinside : inside_triangle A B C P)
variable (hM_on_AC : on_segment A C M)
variable (hL_on_BC : on_segment B C L)
variable (hPACeqPBC : angle A P C = angle B P C)
variable (hPLC90 : angle P L C = 90 * degree)
variable (hPMC90 : angle P M C = 90 * degree)
variable (hDmidAB : midpoint A B D)

-- Theorem statement
theorem equal_dm_dl : distance D M = distance D L :=
sorry

end equal_dm_dl_l107_107907


namespace P_union_Q_eq_123_l107_107844

theorem P_union_Q_eq_123 : 
  let P := {1, 2} in
  let Q := { y | ∃ a, a ∈ P ∧ y = 2 * a - 1 } in
  P ∪ Q = {1, 2, 3} :=
by
  let P := {1, 2}
  let Q := { y | ∃ a, a ∈ P ∧ y = 2 * a - 1 }
  sorry

end P_union_Q_eq_123_l107_107844


namespace cosine_of_largest_angle_l107_107876

theorem cosine_of_largest_angle
  (A B C : ℝ) (k : ℝ) 
  (sin_ratio : sin A / sin B = 3 / 2 ∧ sin B / sin C = 2 / 4 ∧  A + B + C = π) :
  cos C = -1/4 := by
  -- Proof goes here
  sorry

end cosine_of_largest_angle_l107_107876


namespace reservoir_percentage_before_storm_l107_107229

def reservoir_contents_before_storm (contents_after_storm : ℕ) (storm_deposit : ℕ) (percentage_full_after_storm : ℝ) : ℝ := 
  100 * (contents_after_storm - storm_deposit) / ((contents_after_storm - storm_deposit) + storm_deposit)

theorem reservoir_percentage_before_storm :
  let storm_deposit : ℕ := 110
  let original_contents : ℕ := 220
  let percentage_full_after_storm : ℝ := 60
  let total_capacity := original_contents + storm_deposit
  let contents_after_storm := (percentage_full_after_storm / 100) * total_capacity
  reservoir_contents_before_storm contents_after_storm storm_deposit percentage_full_after_storm = 66.67 :=
by
  sorry

end reservoir_percentage_before_storm_l107_107229


namespace Harkamal_total_payment_l107_107847

theorem Harkamal_total_payment :
  let cost_grapes := 10 * 70
  let cost_mangoes := 9 * 55
  let cost_apples := 12 * 80
  let cost_papayas := 7 * 45
  let cost_oranges := 15 * 30
  let cost_bananas := 5 * 25
  cost_grapes + cost_mangoes + cost_apples + cost_papayas + cost_oranges + cost_bananas = 3045 := by
  sorry

end Harkamal_total_payment_l107_107847


namespace uniquely_determine_a_b_l107_107126

-- Define the conditions
variables {a b : ℕ}
variable h1 : a < b
variable h2 : ∃ (n m : ℕ), n * m = 49 * 51 ∧ ∀ x y : ℕ, x * y = a * b → ∃ r, n = x * r ∧ m = y * r
variable h3 : ∃ (p q : ℕ), p * q = 99 * 101 ∧ ∀ x y : ℕ, x * y = a * b → ∃ s, p = x * s ∧ q = y * s

-- Statement that \(a = 1\) and \(b = 3\) are uniquely determined given the conditions
theorem uniquely_determine_a_b (h1 : a < b)
  (h2 : ∃ (n m : ℕ), n * m = 49 * 51 ∧ ∀ x y : ℕ, x * y = a * b → ∃ r, n = x * r ∧ m = y * r)
  (h3 : ∃ (p q : ℕ), p * q = 99 * 101 ∧ ∀ x y : ℕ, x * y = a * b → ∃ s, p = x * s ∧ q = y * s) :
  a = 1 ∧ b = 3 :=
sorry

end uniquely_determine_a_b_l107_107126


namespace median_equal_range_not_greater_l107_107758

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l107_107758


namespace lynne_total_spent_l107_107527

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end lynne_total_spent_l107_107527


namespace squares_fit_in_larger_square_l107_107087

theorem squares_fit_in_larger_square :
  ∃ (arrangement : (ℕ → ℝ × ℝ)), 
    (∀ n, 0 ≤ arrangement n.1 ∧ arrangement n.1 + 1/n ≤ 3/2) ∧
    (∀ m n, m ≠ n → disjoint (square (arrangement m) (1/m)) (square (arrangement n) (1/n))) :=
sorry

end squares_fit_in_larger_square_l107_107087


namespace quadratic_function_properties_l107_107059
-- Import the necessary library

-- Statement
theorem quadratic_function_properties
  (a b : ℝ) (x_1 x_2 : ℝ) 
  (h_a_pos : a > 0) 
  (h_min_val : ∀ x, a * x^2 + b * x + 1 ≥ -a) 
  (h_roots : a * x_1^2 + b * x_1 + 1 = 0) 
  (h_roots2 : a * x_2^2 + b * x_2 + 1 = 0) 
  (x : ℝ)
  (h_sol_set_A : ∀ x, a * x^2 + b * x + 1 < 0 →  x ∈ set.Ioo x_1 x_2)
  (h_no_min_A : ¬∀ x ∈ set.Ioo x_1 x_2, a * x^2 + (b + 2) * x + 1 ≥ -a)
  (h_x1_range : -2 < x_1 ∧ x_1 < 0) :
  (x_1 - x_2 = 2 ∨ x_1 - x_2 = -2) ∧ (0 < a ∧ a ≤ 1) ∧ (b > 3/4) :=
by sorry

end quadratic_function_properties_l107_107059


namespace sum_of_cubes_zero_l107_107499

variables {a b c : ℝ}

theorem sum_of_cubes_zero (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) : a^3 + b^3 + c^3 = 0 :=
sorry

end sum_of_cubes_zero_l107_107499


namespace counting_unit_difference_l107_107644

-- Definitions based on conditions
def magnitude_equality : Prop := 75 = 75.0
def counting_unit_75 : Nat := 1
def counting_unit_75_0 : Nat := 1 / 10

-- Proof problem stating that 75 and 75.0 do not have the same counting units.
theorem counting_unit_difference : 
  ¬ (counting_unit_75 = counting_unit_75_0) :=
by sorry

end counting_unit_difference_l107_107644


namespace projection_of_b_onto_a_l107_107433

def vector_a : ℝ × ℝ × ℝ := (0, 1, 1)
def vector_b : ℝ × ℝ × ℝ := (1, 1, 0)

noncomputable def projection (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let magnitude_squared := v.1^2 + v.2^2 + v.3^2
  (dot_product / magnitude_squared) * v

theorem projection_of_b_onto_a : 
  projection vector_a vector_b = (0, 0.5, 0.5) :=
sorry

end projection_of_b_onto_a_l107_107433


namespace max_bees_in_largest_beehive_l107_107650

def total_bees : ℕ := 2000000
def beehives : ℕ := 7
def min_ratio : ℚ := 0.7

theorem max_bees_in_largest_beehive (B_max : ℚ) : 
  (6 * (min_ratio * B_max) + B_max = total_bees) → 
  B_max <= 2000000 / 5.2 ∧ B_max.floor = 384615 :=
by
  sorry

end max_bees_in_largest_beehive_l107_107650


namespace coloring_points_l107_107015

theorem coloring_points
  (A : ℤ × ℤ) (B : ℤ × ℤ) (C : ℤ × ℤ)
  (hA : A.fst % 2 = 1 ∧ A.snd % 2 = 1)
  (hB : (B.fst % 2 = 1 ∧ B.snd % 2 = 0) ∨ (B.fst % 2 = 0 ∧ B.snd % 2 = 1))
  (hC : C.fst % 2 = 0 ∧ C.snd % 2 = 0) :
  ∃ D : ℤ × ℤ,
    (D.fst % 2 = 1 ∧ D.snd % 2 = 0) ∨ (D.fst % 2 = 0 ∧ D.snd % 2 = 1) ∧
    (A.fst + C.fst = B.fst + D.fst) ∧
    (A.snd + C.snd = B.snd + D.snd) := 
sorry

end coloring_points_l107_107015


namespace distance_between_trains_l107_107668

def first_train_length : ℝ := 100
def first_train_speed : ℝ := 10
def second_train_length : ℝ := 150
def second_train_speed : ℝ := 15
def crossing_time : ℝ := 60

theorem distance_between_trains : 
  ∃ (D : ℝ), D = 50 := 
begin
  have relative_speed := second_train_speed - first_train_speed,
  have total_distance := relative_speed * crossing_time,
  have initial_distance := total_distance - (first_train_length + second_train_length),
  use initial_distance,
  sorry
end

end distance_between_trains_l107_107668


namespace necessary_and_sufficient_condition_l107_107813

theorem necessary_and_sufficient_condition (a : ℝ) :
  (a = 1) ↔ (∀ x y : ℝ, (ax + 2y - 1 = 0 → (a + 1)x + 4y = 0) ∨ (a + 1)x + 4y = 0 → ax + 2y - 1 = 0) := 
sorry

end necessary_and_sufficient_condition_l107_107813


namespace find_uvwx_sum_l107_107497

variables {V : Type*} [inner_product_space ℝ V]
variables (x y z : V)
variables (u v w : ℝ)

-- Definitions for mutually orthogonal unit vectors
def mutually_orthogonal_unit_vectors (x y z : V) : Prop :=
  (⟪x, x⟫ = 1) ∧ (⟪y, y⟫ = 1) ∧ (⟪z, z⟫ = 1) ∧
  (⟪x, y⟫ = 0) ∧ (⟪y, z⟫ = 0) ∧ (⟪z, x⟫ = 0)

-- Placeholder for cross product definition or assumption
axiom cross_product (a b : V) : V

-- Hypothesis that involves the cross product
def hypothesis (x y z : V) (u v w : ℝ) : Prop :=
  x = u • (cross_product y z) + v • (cross_product z x) + w • (cross_product x y)

-- Given condition that x·(y x z) = 1
def given_condition (x y z : V) : Prop :=
  ⟪x, cross_product y z⟫ = 1

-- The Lean theorem statement
theorem find_uvwx_sum (h1 : mutually_orthogonal_unit_vectors x y z)
  (h2 : hypothesis x y z u v w) (h3 : given_condition x y z) : u + v + w = 1 :=
sorry

end find_uvwx_sum_l107_107497


namespace chord_length_PQ_prime_l107_107574

noncomputable def P_Q_prime_length (PQ : ℝ) (R : ℝ) : set ℝ :=
  let γ_sin := PQ / (2 * R),
      γ_cos := real.sqrt (1 - γ_sin^2)
  in { 2 * R * real.sqrt ((1 + γ_cos) / 2),
       2 * R * real.sqrt ((1 - γ_cos) / 2) }

theorem chord_length_PQ_prime :
  P_Q_prime_length 6 5 = {3 * real.sqrt 10, real.sqrt 10} :=
sorry

end chord_length_PQ_prime_l107_107574


namespace proof_problem_l107_107783

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l107_107783


namespace tan_cos_solution_count_l107_107900

theorem tan_cos_solution_count : 
  ∃ (n : ℕ), n = 5 ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.tan (2 * x) = Real.cos (x / 2) → x ∈ Set.Icc 0 (2 * Real.pi) :=
sorry

end tan_cos_solution_count_l107_107900


namespace cost_of_each_pack_l107_107487

theorem cost_of_each_pack (num_packs : ℕ) (total_paid : ℝ) (change_received : ℝ) 
(h1 : num_packs = 3) (h2 : total_paid = 20) (h3 : change_received = 11) : 
(total_paid - change_received) / num_packs = 3 := by
  sorry

end cost_of_each_pack_l107_107487


namespace optimal_cylinder_ratio_l107_107701

theorem optimal_cylinder_ratio (V a b : ℝ) (V_pos : 0 < V) (a_pos : 0 < a) (b_pos : 0 < b) :
  ∃ r h : ℝ, r = (b * V / (2 * π * a))^(1 / 3) ∧ h = V / (π * r^2) ∧ (2 * r / h) = (b / a) :=
begin
  sorry
end

end optimal_cylinder_ratio_l107_107701


namespace largest_invertible_interval_contains_1_l107_107551

def g (x : ℝ) := 3 * x^2 - 6 * x - 9

theorem largest_invertible_interval_contains_1 : 
  ∃ (a b : ℝ), a < 1 ∧ 1 ≤ b ∧ 
  (∀ x y ∈ set.Icc a b, g x = g y → x = y) ∧
  (∀ (x : ℝ), x ∈ set.Icc a b → (∃! z, g z = x)) ∧ 
  ∀ (a' b' : ℝ), a' < 1 ∧ 1 ≤ b' → b' - a' ≤ b - a → 
  (∀ x y ∈ set.Icc a' b', g x = g y → x = y) ∧ 
  (∀ (x : ℝ), x ∈ set.Icc a' b' → (∃! z, g z = x)) →
  a = -∞ ∧ b = 1 :=
sorry

end largest_invertible_interval_contains_1_l107_107551


namespace van_speed_maintain_l107_107273

theorem van_speed_maintain 
  (D : ℕ) (T T_new : ℝ) 
  (initial_distance : D = 435) 
  (initial_time : T = 5) 
  (new_time : T_new = T / 2) : 
  D / T_new = 174 := 
by 
  sorry

end van_speed_maintain_l107_107273


namespace a4_b4_c4_d4_eq_5_l107_107033

-- Define the given conditions and results
variable {𝕂 : Type*} [Field 𝕂] [Fintype 𝟜] -- Complex field, 4x4 matrix
variables (a b c d : 𝕂)

-- Matrix definition
def N : Matrix (Fin 4) (Fin 4) 𝕂 := ![
  ![a, b, c, d],
  ![b, c, d, a],
  ![c, d, a, b],
  ![d, a, b, c]
]

-- Square of matrix is identity
axiom N_squared_identity : N a b c d * N a b c d = 1

-- Product of entries is 1
axiom abcd_one : a * b * c * d = 1

-- Main theorem statement
theorem a4_b4_c4_d4_eq_5 : a^4 + b^4 + c^4 + d^4 = 5 :=
sorry

end a4_b4_c4_d4_eq_5_l107_107033


namespace median_eq_range_le_l107_107745

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l107_107745


namespace balls_in_boxes_l107_107441

theorem balls_in_boxes : 
  (number of ways to put 7 indistinguishable balls into 4 distinguishable boxes) = 132 := 
sorry

end balls_in_boxes_l107_107441


namespace find_lambda_of_coplanarity_l107_107385

-- Define the given conditions using vectors and linear algebra
variables {V : Type*} [add_comm_group V] [module ℝ V]

variables (O A B C P : V)

-- Define the vector relationship
def vector_relation (λ : ℝ) : Prop :=
  P = 0.25 • A + λ • B + (1/6) • C

-- Define the coplanarity condition of points P, A, B, and C
def coplanar (O A B C P : V) : Prop :=
  ∃ (x y z : ℝ), x • (A - O) + y • (B - O) + z • (C - O) = P - O

-- Statement of the proof problem
theorem find_lambda_of_coplanarity (λ : ℝ) (h1 : vector_relation O A B C P λ) (h2 : coplanar O A B C P) :
  λ = 7 / 12 :=
sorry

end find_lambda_of_coplanarity_l107_107385


namespace range_of_f_l107_107329

noncomputable def k : ℝ := Real.pi
noncomputable def f (x : ℝ) : ℝ := Real.floor (k * x) - k * x

theorem range_of_f :
  (set.image f set.univ) = set.Ioc (-1 : ℝ) (0 : ℝ) :=
sorry

end range_of_f_l107_107329


namespace concatenation_palindrome_l107_107274

-- Definitions given in the problem
def word : Type := String
def palindrome (w : word) : Prop := w = w.reverse

def W : ℕ → word
| 0 => "a"
| 1 => "b"
| (n+2) => W n ++ W (n+1)

-- Theorem statement to prove
theorem concatenation_palindrome (n : ℕ) (h : n ≥ 1) : 
  palindrome (List.foldl (++) "" (List.map W (List.range (n+1)))) :=
sorry

end concatenation_palindrome_l107_107274


namespace percent_of_70_is_56_l107_107243

theorem percent_of_70_is_56 : (70 / 125) * 100 = 56 := by
  sorry

end percent_of_70_is_56_l107_107243


namespace vec_a_b_c_dot_product_l107_107816

-- Define the vectors and their properties
def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b : ℝ × ℝ := (1, -2)
def vec_c : ℝ × ℝ := (3, -6)

-- Dot product operation
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Conditions
def a_perp_c (x : ℝ) : Prop := dot_product (vec_a x) vec_c = 0
def b_par_c : Prop := vec_b = (-2 : ℝ) * (vec_c.1, vec_c.2)

-- Main theorem to prove
theorem vec_a_b_c_dot_product (x : ℝ) (h1 : a_perp_c x) (h2 : b_par_c) : dot_product (vec_a x + vec_b) vec_c = 15 :=
by
  sorry

end vec_a_b_c_dot_product_l107_107816


namespace opening_ceremony_audience_correct_closing_ceremony_audience_correct_l107_107581

-- First, define the opening ceremony audience and prove it
def opening_ceremony_audience_million := 316
def opening_ceremony_audience_unit := opening_ceremony_audience_million * 1000000

-- Define the closing ceremony audience and prove the conversion
def closing_ceremony_audience_million := 236
def closing_ceremony_audience_billion := closing_ceremony_audience_million / 1000

theorem opening_ceremony_audience_correct :
  opening_ceremony_audience_unit = 316000000 :=
by
  -- Proof of the theorem
  sorry

theorem closing_ceremony_audience_correct :
  Float.floor (closing_ceremony_audience_billion * 10) / 10 = 0.2 :=
by
  -- Proof of the theorem
  sorry

end opening_ceremony_audience_correct_closing_ceremony_audience_correct_l107_107581


namespace number_of_subsets_M_l107_107737

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the set M based on the given conditions
def M : Set ℂ := { x | ∃ n : ℕ, x = i^n + i^(-n : ℂ) }

-- Determine the number of subsets of M
theorem number_of_subsets_M : Finset.card (Finset.powerset {0, 2, -2}) = 8 := 
by {
  -- Set a proof placeholder
  sorry
}

end number_of_subsets_M_l107_107737


namespace arithmetic_seq_sum_is_110_l107_107387

noncomputable def S₁₀ (a_1 : ℝ) : ℝ :=
  10 / 2 * (2 * a_1 + 9 * (-2))

theorem arithmetic_seq_sum_is_110 (a1 a3 a7 a9 : ℝ) 
  (h_diff3 : a3 = a1 - 4)
  (h_diff7 : a7 = a1 - 12)
  (h_diff9 : a9 = a1 - 16)
  (h_geom : (a1 - 12) ^ 2 = (a1 - 4) * (a1 - 16)) :
  S₁₀ a1 = 110 :=
by
  sorry

end arithmetic_seq_sum_is_110_l107_107387


namespace number_of_integers_with_6_or_7_as_digit_in_base9_l107_107851

/-- 
  There are 729 smallest positive integers written in base 9.
  We want to determine how many of these integers use the digits 6 or 7 (or both) at least once.
-/
theorem number_of_integers_with_6_or_7_as_digit_in_base9 : 
  ∃ n : ℕ, n = 729 ∧ ∃ m : ℕ, m = n - 7^3 := sorry

end number_of_integers_with_6_or_7_as_digit_in_base9_l107_107851


namespace smallest_x_for_gx_eq_1024_l107_107250

noncomputable def g : ℝ → ℝ
  | x => if 2 ≤ x ∧ x ≤ 6 then 2 - |x - 3| else 0

axiom g_property1 : ∀ x : ℝ, 0 < x → g (4 * x) = 4 * g x
axiom g_property2 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → g x = 2 - |x - 3|
axiom g_2004 : g 2004 = 1024

theorem smallest_x_for_gx_eq_1024 : ∃ x : ℝ, g x = 1024 ∧ ∀ y : ℝ, g y = 1024 → x ≤ y := sorry

end smallest_x_for_gx_eq_1024_l107_107250


namespace lynne_total_spent_l107_107526

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end lynne_total_spent_l107_107526


namespace average_speed_l107_107230

theorem average_speed (v1 v2 : ℝ) (h1 : v1 = 110) (h2 : v2 = 88) : 
  (2 * v1 * v2) / (v1 + v2) = 97.78 := 
by sorry

end average_speed_l107_107230


namespace area_of_triangle_CDF_l107_107054

-- Definitions of points and conditions
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (3, 0)
def C : (ℝ × ℝ) := (3, 1)
def D : (ℝ × ℝ) := (0, 1)
def E : (ℝ × ℝ) := ((0 + 3) / 2, (0 + 0) / 2)
def F : (ℝ × ℝ) := ((0 + 3) / 2, 0 - 1)

-- Function to calculate the area of a triangle given three points
def triangle_area (P Q R : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

-- Theorem statement
theorem area_of_triangle_CDF : triangle_area C D F = 3 := sorry

end area_of_triangle_CDF_l107_107054


namespace median_equality_range_inequality_l107_107802

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l107_107802


namespace remainder_of_division_l107_107719

noncomputable def polynomial_div_remainder (p q: Polynomial ℤ): Polynomial ℤ × Polynomial ℤ :=
let (q', r) := p.divMod q
in (q', r)
 
theorem remainder_of_division :
  polynomial_div_remainder (3 * Polynomial.X^5 + 2 * Polynomial.X^3 - 5 * Polynomial.X^2 + 9) (Polynomial.X^2 + 3 * Polynomial.X + 2) =
  (Polynomial.div (3 * Polynomial.X^5 + 2 * Polynomial.X^3 - 5 * Polynomial.X^2 + 9) (Polynomial.X^2 + 3 * Polynomial.X + 2),
   122 * Polynomial.X + 121) :=
by sorry

end remainder_of_division_l107_107719


namespace smallest_number_of_coins_l107_107610

theorem smallest_number_of_coins (coins : Finset ℕ) (hcoins : coins = {1, 5, 10, 25}) :
  ∃ (n : ℕ), n = 10 ∧ (∀ x, x ∈ Icc 1 99 → ∃ (c1 c5 c10 c25 : ℕ), c1 + 5*c5 + 10*c10 + 25*c25 = x ∧ c1 + c5 + c10 + c25 = n) :=
by
  sorry

end smallest_number_of_coins_l107_107610


namespace expected_value_area_circle_variance_area_circle_l107_107994

def uniform_pdf (a b x : ℝ) : ℝ :=
if h : a ≤ x ∧ x ≤ b then 1 / (b - a) else 0

noncomputable def expected_value_area (a b : ℝ) : ℝ :=
  (π * (b^3 - a^3) / 12) / (b - a) 

noncomputable def variance_area (a b : ℝ) : ℝ :=
  π^2 * (b^4 + b^3 * a + b^2 * a^2 + b * a^3 + a^4) / 80 - (π * (b^2 + a * b + a^2) / 12)^2

theorem expected_value_area_circle (a b : ℝ) (h : a < b) :
  expected_value_area a b = π * (b^2 + a * b + a^2) / 12 :=
by sorry

theorem variance_area_circle (a b : ℝ) (h : a < b) :
  variance_area a b = π^2 * (b^4 + b^3 * a + b^2 * a^2 + b * a^3 + a^4) / 80 - (π * (b^2 + a * b + a^2) / 12)^2 :=
by sorry

end expected_value_area_circle_variance_area_circle_l107_107994


namespace isosceles_triangle_perimeter_l107_107380

theorem isosceles_triangle_perimeter 
  (m : ℝ) 
  (h : 2 * m + 1 = 8) : 
  (m - 2) + 2 * 8 = 17.5 := 
by 
  sorry

end isosceles_triangle_perimeter_l107_107380


namespace find_equation_of_ellipse_sum_of_slopes_constant_l107_107806

-- Definitions of the Ellipse and relevant Points and Conditions
def a (e: ℝ) := 2 * (0.5:ℝ) / e
def b (a: ℝ) := (a^2 - 1^2)^(1/2)

-- Given Conditions
def ellipse : Prop :=
  ∃ (a b x y: ℝ), (a > b ∧ b > 0) ∧
   (x^2 / a^2 + y^2 / b^2 = 1) ∧
   (a = 2 ∧ b^2 = 3)

def point_M : Prop := M = (2, 0 : ℝ)
def point_Q : Prop := Q = (1, 0 : ℝ)
def point_P : Prop := P = (4, 3 : ℝ)

-- Main Statements to be Proved
theorem find_equation_of_ellipse : ellipse → ∃ a b, ∀ x y, a = 2 ∧ b = (3: ℝ)^(1/2) → x^2 / 4 + y^2 / 3 = 1 :=
by sorry

theorem sum_of_slopes_constant : (k1 k2 : ℝ) → ∀ A B, (A = (1, 3/2) ∧ B = (1, -3/2)) ∨ (k for k ∈ ℝ) → (k1 + k2 = 2) :=
by sorry

end find_equation_of_ellipse_sum_of_slopes_constant_l107_107806


namespace lynne_total_spent_l107_107528

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end lynne_total_spent_l107_107528


namespace equilateral_triangle_area_l107_107892

noncomputable def sqrt3 := Real.sqrt 3

theorem equilateral_triangle_area :
  ∀ (A B C D M N : Type) [has_dist A B C D M N]
  (h_rect : Rectangle ABCD)
  (h_AB : has_dist.dist A B = 2)
  (h_BC : has_dist.dist B C = 1)
  (h_mid_M : is_midpoint M A B)
  (h_eq_tri : is_equilateral_triangle BMN),
  0.5 * 1 * 1 = sqrt3 / 4 :=
by
  intros
  sorry

end equilateral_triangle_area_l107_107892


namespace probability_1_2_is_0_1_l107_107400

noncomputable def find_probability (ξ : ℝ → ℝ) (μ : ℝ) (σ : ℝ) : Prop :=
  let normal_dist := λ x, (1/(σ * √(2 * Math.pi))) * Real.exp(-(x - μ)^2 / (2 * σ^2))
  (∀ x, ξ(x) = normal_dist x) → 
  (μ = 2) →
  (σ > 0) →
  (Real.Integral (ξ 2 3) = 0.6) →
  (Real.Integral (ξ 1 2) = 0.1)

theorem probability_1_2_is_0_1 : find_probability (ξ) 2 σ :=
sorry

end probability_1_2_is_0_1_l107_107400


namespace circle_problem_l107_107303

theorem circle_problem (r p q : ℕ) 
  (h1 : (r : ℝ) > 0)
  (h2 : 2 * r = 6)
  (h3 : 2 * q * real.sqrt p = 14) :
  p + q = 56 :=
sorry

end circle_problem_l107_107303


namespace product_sum_even_l107_107929

theorem product_sum_even (m n : ℤ) : Even (m * n * (m + n)) := 
sorry

end product_sum_even_l107_107929


namespace initial_quantity_l107_107619

variables {A : ℝ} -- initial quantity of acidic liquid
variables {W : ℝ} -- quantity of water removed

theorem initial_quantity (h1: A * 0.6 = W + 25) (h2: W = 9) : A = 27 :=
by
  sorry

end initial_quantity_l107_107619


namespace find_dimes_in_tip_jar_l107_107600

variable (nickels_amt dimes_amt half_dollars_amt total_amt amount_per_nickel 
          amount_per_dime amount_per_half_dollar total_from_shoes 
          total_in_tip_jar from_half_dollars dimes_in_tip_jar : ℕ)

-- Conditions
def conditions : Prop :=
  nickels_amt = 3 ∧
  dimes_amt = 13 ∧
  half_dollars_amt = 9 ∧
  total_amt = 665 ∧ -- In cents to avoid decimals
  amount_per_nickel = 5 ∧
  amount_per_dime = 10 ∧
  amount_per_half_dollar = 50 ∧
  total_from_shoes = (nickels_amt * amount_per_nickel + dimes_amt * amount_per_dime) ∧
  total_in_tip_jar = (total_amt - total_from_shoes) ∧
  from_half_dollars = half_dollars_amt * amount_per_half_dollar ∧
  dimes_in_tip_jar = (total_in_tip_jar - from_half_dollars) / amount_per_dime

-- Prove the number of dimes in the tip jar
theorem find_dimes_in_tip_jar (h : conditions) : 
  dimes_in_tip_jar = 7 :=
sorry

end find_dimes_in_tip_jar_l107_107600


namespace sum_of_floor_sqrt_l107_107694

theorem sum_of_floor_sqrt : (∑ n in Finset.range 26, Int.floor (Real.sqrt n)) = 75 := by
  sorry

end sum_of_floor_sqrt_l107_107694


namespace median_equality_range_inequality_l107_107749

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l107_107749


namespace polynomial_operations_l107_107109

-- Define the given options for M, N, and P
def A (x : ℝ) : ℝ := 2 * x - 6
def B (x : ℝ) : ℝ := 3 * x + 5
def C (x : ℝ) : ℝ := -5 * x - 21

-- Define the original expression and its simplified form
def original_expr (M N : ℝ → ℝ) (x : ℝ) : ℝ :=
  2 * M x - 3 * N x

-- Define the simplified target expression
def simplified_expr (x : ℝ) : ℝ := -5 * x - 21

theorem polynomial_operations :
  ∀ (M N P : ℝ → ℝ),
  (original_expr M N = simplified_expr) →
  (M = A ∨ N = B ∨ P = C)
:= by
  intros M N P H
  sorry

end polynomial_operations_l107_107109


namespace inequality_true_l107_107450

variable (a b : ℝ)

theorem inequality_true (h1 : b > a) (h2 : a > 0) : 2 * a + b / 2 ≥ 2 * Real.sqrt(a * b) := sorry

end inequality_true_l107_107450


namespace polygon_with_interior_angles_sum_1800_is_dodecagon_l107_107122

/-- Prove that if the sum of the interior angles of a polygon is 1800 degrees, then the polygon is a dodecagon. -/
theorem polygon_with_interior_angles_sum_1800_is_dodecagon :
  ∃ (n : ℕ), (n - 2) * 180 = 1800 ∧ n = 12 :=
by
  use 12
  simp [←nat.succ_pred_eq_of_pos (show 2 < 12, by linarith)], sorry

end polygon_with_interior_angles_sum_1800_is_dodecagon_l107_107122


namespace det_n_matrix_val_l107_107708

-- Define the matrix A in terms of the constants a and b
def det_n_matrix (n : ℕ) (a b : ℝ) : matrix (fin n) (fin n) ℝ :=
  λ i j, if i = j then b else a

-- Define the theorem to state the result of the determinant
theorem det_n_matrix_val (n : ℕ) (a b : ℝ) :
  matrix.det (det_n_matrix n a b) = (b + (n - 1) * a) * (b - a)^(n - 1) :=
  sorry

end det_n_matrix_val_l107_107708


namespace find_a1_l107_107396

variable (a : ℕ → ℝ)
variable (q : ℝ)

hypothesis pos_q : q > 0
hypothesis h1 : a 3 * a 9 = 2 * (a 5)^2
hypothesis h2 : a 2 = 2

theorem find_a1 : a 1 = sqrt 2 :=
by
  sorry

end find_a1_l107_107396


namespace gillian_total_spent_l107_107549

-- Definitions from the conditions
def sandi_initial_amount : ℕ := 600
def sandi_spent := sandi_initial_amount / 2
def gillian_spent := 3 * sandi_spent + 150

-- Proof statement
theorem gillian_total_spent : gillian_spent = 1050 := 
by
  unfold sandi_initial_amount
  unfold sandi_spent
  unfold gillian_spent
  sorry

end gillian_total_spent_l107_107549


namespace chessboard_sum_eq_zero_l107_107531

theorem chessboard_sum_eq_zero :
  ∀ (a : Fin 64 → ℤ)
  (row : Fin 8 → Fin 8 → ℤ)
  (col : Fin 8 → Fin 8 → ℤ),
  (∀ i, row i = λ j, a ⟨8 * i + j, by linarith⟩) →
  (∀ j, col j = λ i, a ⟨8 * i + j, by linarith⟩) →
  (∀ i, (∑ j in Finset.range 8, if j < 4 then row i j else -row i j) = 0) →
  (∀ j, (∑ i in Finset.range 8, if i < 4 then col j i else -col j i) = 0) →
  (∑ i in Finset.range 64, a i) = 0 :=
by sorry

end chessboard_sum_eq_zero_l107_107531


namespace probability_real_number_correct_l107_107955

noncomputable def probability_real_number (a b : ℚ) : ℚ :=
  if h : (a ∈ Icc (0 : ℚ) 3 ∧ b ∈ Icc (0 : ℚ) 3) ∧
           (∃ (n1 d1 n2 d2 : ℤ), (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ 
                                      a = n1 / d1 ∧ b = n2 / d2) ∧
           ((cos (a * real.pi) + complex.i * sin (b * real.pi)) ^ 8).im = 0
  then 1
  else 0

theorem probability_real_number_correct :
  let S := {a : ℚ | ∃ (n d : ℤ), 1 ≤ d ∧ d ≤ 6 ∧ a = n / d ∧ 0 ≤ a ∧ a < 3}
  in let count := λ P, finset.card {p ∈ finset.product (S.to_finset) (S.to_finset) | 
                                     probability_real_number p.1 p.2 = 1}
  in count {a, b | probability_real_number a b = 1} / 1764 = 17 / 196 := sorry

end probability_real_number_correct_l107_107955


namespace playoff_prize_distributions_l107_107683

-- Define the number of bowlers and the structure of the playoff
def numBowlers := 7

def playoffMatchups : List (ℕ × ℕ) := [
  (7, 6),
  (winner, 5),
  (winner, 4),
  (winner, 3),
  (winner, 2),
  (winner, 1)
]

theorem playoff_prize_distributions : (2 ^ (numBowlers - 1)) = 64 := by
  -- By problem setup: there are 6 matchups, each with 2 outcomes
  -- 2^6 = 64 follows directly
  sorry

end playoff_prize_distributions_l107_107683


namespace correct_average_l107_107986

theorem correct_average (n : ℕ) (incorrect_sum incorrect_avg misread correct_read : ℝ) :
  n = 10 →
  incorrect_avg = 15 →
  incorrect_sum = incorrect_avg * n →
  misread = 26 →
  correct_read = 36 →
  let correct_sum := incorrect_sum + (correct_read - misread) in
  let correct_avg := correct_sum / n in
  correct_avg = 16 :=
by
  intros hn hi_avg hi_sum hmisread hcorrect_read
  unfold correct_sum correct_avg
  rw [hi_sum, hi_avg, hn, hmisread, hcorrect_read]
  norm_num

end correct_average_l107_107986


namespace subset_interval_m_l107_107843

theorem subset_interval_m (m : ℝ) : 
  (∀ x : ℝ, m < x ∧ x < m + 1 → -2 < x ∧ x < 2) ↔ (-2 ≤ m ∧ m ≤ 1) :=
by
  split
  {
    intro h
    split
    {
      by_contra hneg
      have h₁ := h (m - 1) ⟨lt_sub.mpr (by linarith), sub_lt_self m zero_lt_one⟩
      linarith
    },
    {
      by_contra hneg
      have h₁ := h (m + 1) ⟨lt_add_one.mpr (by linarith), add_lt_add_right (lt_add_one.mpr (by linarith)) 1⟩
      linarith
    }
  },
  {
    intro h
    intros x hx
    split
    exact lt_trans (lt_of_le_of_lt (h.1) hx.1) hx.2,
    exact lt_of_lt_of_le hx.2 (add_le_iff_nonneg_left (h.2)).mpr hx.1,
  }

end subset_interval_m_l107_107843


namespace catalan_recurrence_l107_107950

noncomputable def catalan : ℕ → ℕ
| 0     := 1
| (n+1) := (2 * (2 * n + 1) * catalan n) / (n + 2)

theorem catalan_recurrence : ∀ n, catalan n = ∑ i in finset.range(n), (catalan i) * (catalan (n - 1 - i)) := by
  intro n
  sorry

end catalan_recurrence_l107_107950


namespace probability_of_event_l107_107578

-- Define the conditions in Lean
def line1 (a x y : ℝ) : Prop := 2 * y + (x + 4) = 2 * a
def line2 (x y : ℝ) : Prop := -x + y = 3
def line3 (x y : ℝ) : Prop := x + y = 3

-- Define abscissa conditions
def abscissa1 (a x : ℝ) : Prop := x = (2 * a - 10) / 3 ∧ -3 ≤ x ∧ x ≤ 0
def abscissa2 (a x : ℝ) : Prop := x = 10 - 2 * a ∧ 0 ≤ x ∧ x ≤ 3

-- Define valid interval for 'a' based on abscissas
def valid_interval1 (a : ℝ) : Prop := 1/2 ≤ a ∧ a ≤ 5
def valid_interval2 (a : ℝ) : Prop := 7/2 ≤ a ∧ a ≤ 5

-- Define the intersection of intervals
def intersection_interval (a : ℝ) : Prop := 7/2 ≤ a ∧ a < 5

-- Define the probability calculation
def length_segment (a b : ℝ) : ℝ := b - a
def probability_event (a₁ a₂ b₁ b₂ : ℝ) : ℝ :=
  length_segment a₁ a₂ / length_segment b₁ b₂

-- Define the proof statement
theorem probability_of_event : 
  probability_event (7/2) 5 (-1) 5 = 1/4 := by 
  sorry

end probability_of_event_l107_107578


namespace sequences_meet_at_2017_l107_107075

-- Define the sequences for Paul and Penny
def paul_sequence (n : ℕ) : ℕ := 3 * n - 2
def penny_sequence (m : ℕ) : ℕ := 2022 - 5 * m

-- Statement to be proven
theorem sequences_meet_at_2017 : ∃ n m : ℕ, paul_sequence n = 2017 ∧ penny_sequence m = 2017 := by
  sorry

end sequences_meet_at_2017_l107_107075


namespace y_increase_when_x_increases_l107_107067

theorem y_increase_when_x_increases (Δx : ℝ) (Δy : ℝ) (k : ℝ) (x_inc : ℝ) (y_inc : ℝ) :
  Δy = (6 / 4) * Δx →
  Δx = 12 →
  Δy = 18 :=
by
  intro h1 h2
  rw [h2, h1]
  norm_num
  sorry

end y_increase_when_x_increases_l107_107067


namespace median_equal_range_not_greater_l107_107761

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l107_107761


namespace binom_600_600_l107_107321

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l107_107321


namespace smallest_prime_12_less_than_perfect_square_l107_107219

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107219


namespace abs_minus_five_plus_three_l107_107553

theorem abs_minus_five_plus_three : |(-5 + 3)| = 2 := 
by
  sorry

end abs_minus_five_plus_three_l107_107553


namespace expression_divisible_by_1897_l107_107084

theorem expression_divisible_by_1897 (n : ℕ) :
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end expression_divisible_by_1897_l107_107084


namespace range_of_m_l107_107415

def f (x : ℝ) : ℝ := 4 * sin^2 (π / 4 + x) - 2 * sqrt 3 * cos (2 * x) - 1

def p (x : ℝ) : Prop := π / 4 ≤ x ∧ x ≤ π / 2

def q (x m : ℝ) : Prop := abs (f x - m) < 2

theorem range_of_m (m : ℝ) : (3 < m ∧ m < 5) ↔
  (∀ x, p x → (∃ x', 2x - π/3 ∈ (1/2, 1) ∧ f x' ∈ (3, 5)) ∧ 
         (q x m → p x) / 
         (p x → ∃ x', 2x - π / 3 ∈ (1/2, 1) ∧ f x'∈ (3, 5) 
         → ∃ y, (abs (f y - m) < 2)) / 
         ¬ (p x → ∃ y, (abs (f y - m) < 2))) :=
begin
  sorry
end

end range_of_m_l107_107415


namespace problem1_problem2_l107_107840

-- (I) Prove m = 0 given f(x) = (m-1)^2 x^(m^2 - 4m + 2) is monotonically increasing in (0, + ∞)
theorem problem1 (m : ℝ)
  (f : ℝ → ℝ := λ x, (m-1)^2 * x^(m^2 - 4m + 2))
  (hf_mono : ∀ x y : ℝ, (0 < x ∧ x < y) → f x < f y) :
  m = 0 :=
sorry

-- (II) Prove the range of k is [0, 1] given the conditions
theorem problem2 (k : ℝ)
  (f : ℝ → ℝ := λ x, x^2)
  (g : ℝ → ℝ := λ x, 2^x - k)
  (A := set.Icc (1:ℝ) 4)
  (B := set.Icc (2-k) (4-k))
  (hp_q_cond : B ⊆ A) :
  0 ≤ k ∧ k ≤ 1 :=
sorry

end problem1_problem2_l107_107840


namespace trapezoid_area_eq_15_l107_107613

theorem trapezoid_area_eq_15 :
  let line1 := fun (x : ℝ) => 2 * x
  let line2 := fun (x : ℝ) => 8
  let line3 := fun (x : ℝ) => 2
  let y_axis := fun (y : ℝ) => 0
  let intersection_points := [
    (4, 8),   -- Intersection of line1 and line2
    (1, 2),   -- Intersection of line1 and line3
    (0, 8),   -- Intersection of y_axis and line2
    (0, 2)    -- Intersection of y_axis and line3
  ]
  let base1 := (4 - 0 : ℝ)  -- Length of top base 
  let base2 := (1 - 0 : ℝ)  -- Length of bottom base
  let height := (8 - 2 : ℝ) -- Vertical distance between line2 and line3
  (0.5 * (base1 + base2) * height = 15.0) := by
  sorry

end trapezoid_area_eq_15_l107_107613


namespace problem_A_problem_C_problem_D_problem_E_l107_107559

variable {a b c : ℝ}
variable (ha : a < 0) (hab : a < b) (hb : b < 0) (hc : 0 < c)

theorem problem_A (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * b > a * c :=
by sorry

theorem problem_C (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * c < b * c :=
by sorry

theorem problem_D (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a + c < b + c :=
by sorry

theorem problem_E (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : c / a > 1 :=
by sorry

end problem_A_problem_C_problem_D_problem_E_l107_107559


namespace binomial_600_600_l107_107308

theorem binomial_600_600 : nat.choose 600 600 = 1 :=
by
  -- Given the condition that binomial coefficient of n choose n is 1 for any non-negative n
  have h : ∀ n : ℕ, nat.choose n n = 1 := sorry
  -- Applying directly to the specific case n = 600
  exact h 600

end binomial_600_600_l107_107308


namespace new_car_price_is_correct_l107_107936

-- Given conditions
variables (P : ℝ) -- original price of the car
variables (proceeds additional new_car_price: ℝ)

axiom h1 : proceeds = 0.8 * P -- Liz sold her car at 80% of the original price
axiom h2 : additional = 4000 -- She needs an additional $4,000
axiom h3 : new_car_price = P - 2500 -- New car is $2,500 cheaper than original

-- Prove that the new car price is $30,000
theorem new_car_price_is_correct: new_car_price = 30000 :=
by
  have h4 : proceeds + additional = new_car_price, from sorry -- From the given conditions
  let proceeds_calculation := 0.8 * P 
  have h5 : 0.8 * P + 4000 = P - 2500, from sorry
  let P_value : ℝ := 32500
  have h6 : P = 32500, from sorry
  have new_car_price_computation : P_value -2500 = 30000, from sorry
  exact new_car_price_computation

end new_car_price_is_correct_l107_107936


namespace tangent_line_a_value_l107_107872

theorem tangent_line_a_value (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y - 1 = 0 → x^2 + y^2 + 4 * x = 0) → a = -1 / 4 :=
by
  sorry

end tangent_line_a_value_l107_107872


namespace smallest_prime_12_less_than_square_l107_107190

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107190


namespace process_time_in_hours_l107_107670

theorem process_time_in_hours (num_pictures : ℕ) (minutes_per_picture : ℕ) (minutes_per_hour : ℕ) :
  num_pictures = 960 → minutes_per_picture = 2 → minutes_per_hour = 60 →
  num_pictures * minutes_per_picture / minutes_per_hour = 32 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold num_pictures minutes_per_picture minutes_per_hour
  sorry

end process_time_in_hours_l107_107670


namespace median_equal_range_not_greater_l107_107756

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l107_107756


namespace median_equality_range_inequality_l107_107801

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l107_107801


namespace parameter_range_l107_107827

theorem parameter_range (k : ℝ) :
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ x = 0 ∧ y = 0) →
  0 < |k| ∧ |k| < 1 :=
begin
  sorry,
end

end parameter_range_l107_107827


namespace distance_center_to_plane_l107_107829

noncomputable def radius_of_sphere (surface_area : ℝ) : ℝ :=
  real.sqrt (surface_area / (4 * real.pi))

noncomputable def distance_to_plane (R r : ℝ) : ℝ :=
  real.sqrt (R^2 - r^2)

theorem distance_center_to_plane 
  (surface_area : ℝ) 
  (AB AC BC : ℝ)
  (h1 : surface_area = 20 * real.pi)
  (h2 : AB = 2)
  (h3 : AC = 2)
  (h4 : BC = 2 * real.sqrt 3) :
  distance_to_plane (radius_of_sphere surface_area) 2 = 1 := 
sorry

end distance_center_to_plane_l107_107829


namespace university_survey_analysis_l107_107245

noncomputable def probability_classical_enthusiast (num_enthusiasts : ℕ) (total_students : ℕ) : ℚ :=
  num_enthusiasts / total_students

def stratified_sampling (university_male: ℕ) (university_female: ℕ) (surveyed_male : ℕ) (surveyed_female : ℕ) : Prop :=
  university_male / university_female = surveyed_male / surveyed_female

noncomputable def chi_squared_value (n : ℕ) (a b c d : ℕ) : ℚ :=
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

def check_certainity_K2 (K2: ℚ) (threshold: ℚ): Prop := K2 > threshold

theorem university_survey_analysis:
  let university_male := 4000 in
  let university_female := 5000 in
  let surveyed_male := 40 in
  let surveyed_female := 50 in
  let num_enthusiasts := 60 in
  let total_students := 90 in
  let a := 20 in
  let b := 20 in
  let c := 40 in
  let d := 10 in
  let n := surveyed_male + surveyed_female in
  (probability_classical_enthusiast num_enthusiasts total_students = 2 / 3) ∧
  stratified_sampling university_male university_female surveyed_male surveyed_female ∧
  check_certainity_K2 (chi_squared_value n a b c d) 6.635 :=
  sorry

end university_survey_analysis_l107_107245


namespace angle_WYZ_correct_l107_107038

-- Define the angles as constants
def angle_XYZ : ℝ := 36
def angle_XYW : ℝ := 15

-- Theorem statement asserting the solution
theorem angle_WYZ_correct :
  (angle_XYZ - angle_XYW = 21) := 
by
  -- This is where the proof would go, but we use 'sorry' as instructed
  sorry

end angle_WYZ_correct_l107_107038


namespace derivative_of_volume_is_surface_area_l107_107564

open Real

noncomputable def volume (R : ℝ) := (π * R ^ 3) 

theorem derivative_of_volume_is_surface_area (R : ℝ) (h : 0 < R):
  deriv (λ R, π * R ^ 3) R = 4 * π * R ^ 2 :=
by
  sorry

end derivative_of_volume_is_surface_area_l107_107564


namespace find_number_l107_107239

theorem find_number
  (x : ℝ)
  (h : (7.5 * 7.5) + 37.5 + (x * x) = 100) :
  x = 2.5 :=
sorry

end find_number_l107_107239


namespace quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l107_107424

theorem quadratic_has_two_real_roots_for_any_m (m : ℝ) : 
  ∃ (α β : ℝ), (α^2 - 3*α + 2 - m^2 - m = 0) ∧ (β^2 - 3*β + 2 - m^2 - m = 0) :=
sorry

theorem find_m_given_roots_conditions (α β : ℝ) (m : ℝ) 
  (h1 : α^2 - 3*α + 2 - m^2 - m = 0) 
  (h2 : β^2 - 3*β + 2 - m^2 - m = 0) 
  (h3 : α^2 + β^2 = 9) : 
  m = -2 ∨ m = 1 :=
sorry

end quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l107_107424


namespace range_of_m_l107_107723

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (m / (2*x - 1) + 3 = 0) ∧ (x > 0)) ↔ (m < 3 ∧ m ≠ 0) :=
by
  sorry

end range_of_m_l107_107723


namespace cos_sum_simplified_l107_107963

theorem cos_sum_simplified :
  (Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)) = ((Real.sqrt 13 - 1) / 4) :=
by
  sorry

end cos_sum_simplified_l107_107963


namespace arithmetic_sequence_sum_l107_107935

theorem arithmetic_sequence_sum (S : ℕ → ℕ)
  (h₁ : S 3 = 9)
  (h₂ : S 6 = 36) :
  S 9 - S 6 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l107_107935


namespace Lacy_correct_percent_l107_107945

theorem Lacy_correct_percent (x : ℝ) (h1 : 7 * x > 0) : ((5 * 100) / 7) = 71.43 :=
by
  sorry

end Lacy_correct_percent_l107_107945


namespace pie_difference_l107_107472

theorem pie_difference (p1 p2 : ℚ) (h1 : p1 = 5 / 6) (h2 : p2 = 2 / 3) : p1 - p2 = 1 / 6 := 
by 
  sorry

end pie_difference_l107_107472


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107172

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107172


namespace quadrilateral_inscribed_in_circle_iff_opposite_angles_sum_180_l107_107091

theorem quadrilateral_inscribed_in_circle_iff_opposite_angles_sum_180
  (A B C D : Type)
  (angle_A angle_B angle_C angle_D : ℝ)
  (h_quadrilateral : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)
  (h_sum_A_C : angle_A + angle_C = 180)
  (h_sum_B_D : angle_B + angle_D = 180) :
  ∃ (O : Type), -- there exists a circle with center O
  quadrilateral_inscribed_in_circle A B C D O ↔
    (h_sum_A_C ∧ h_sum_B_D) :=
sorry

end quadrilateral_inscribed_in_circle_iff_opposite_angles_sum_180_l107_107091


namespace length_of_real_axis_of_hyperbola_l107_107732

theorem length_of_real_axis_of_hyperbola (a : ℝ) (A B : ℝ × ℝ) (hA : A = (-4, 2 * real.sqrt 3)) (hB : B = (-4, -2 * real.sqrt 3)) 
(h_intersect : abs (A.2 - B.2) = 4 * real.sqrt 3) 
(h_C_eq : ∀ (x y: ℝ), x^2 - y^2 = a^2 ↔ x = -4 ∧ (y = 2 * real.sqrt 3 ∨ y = -2 * real.sqrt 3)) : 
2 * a = 4 :=
by {
  -- skipping the proof
  sorry
}

end length_of_real_axis_of_hyperbola_l107_107732


namespace power_of_point_l107_107948

theorem power_of_point (R d : ℝ) (hR : 0 < R) (hMd : R < d) (M O A B : EuclideanGeometry.Point) :
  (M ≠ O) ∧ (EuclideanGeometry.dist M O = d) ∧
  (∀ (L : EuclideanGeometry.Line), EuclideanGeometry.PointOnLine M L ∧
    (∃ (A B : EuclideanGeometry.Point), EuclideanGeometry.PointOnCircle A R ∧ EuclideanGeometry.PointOnCircle B R ∧ 
      EuclideanGeometry.PointOnLine A L ∧ EuclideanGeometry.PointOnLine B L) →
  EuclideanGeometry.dist M A * EuclideanGeometry.dist M B = d^2 - R^2) := sorry

end power_of_point_l107_107948


namespace binom_600_eq_1_l107_107317

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l107_107317


namespace square_difference_divisible_by_three_l107_107085

theorem square_difference_divisible_by_three (n : ℤ) (k : ℤ) 
  (h : k = 1 ∨ k = 2) : 
  (let x := 3 * n + k in (x ^ 2 - 1) % 3 = 0) :=
by sorry

end square_difference_divisible_by_three_l107_107085


namespace lynne_total_spending_l107_107523

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end lynne_total_spending_l107_107523


namespace tan_angle_AMB_l107_107731

theorem tan_angle_AMB (p : ℝ) (hp : p > 0) :
  let F : (ℝ × ℝ) := (p / 2, 0)
  let C : (ℝ × ℝ) → Prop := λ (x y : ℝ), y^2 = 2*p*x
  let l : (ℝ × ℝ) → Prop := λ (x y : ℝ), y = x - p / 2
  let A : (ℝ × ℝ) := (3*p/2 + p*sqrt 2, p/2 + p*sqrt 2)
  let B : (ℝ × ℝ) := (3*p/2 - p*sqrt 2, p/2 - p*sqrt 2)
  let M : (ℝ × ℝ) := (-p / 2, 0)
  tan_angle_AMB = 2 * sqrt 2 :=
begin
  sorry,
end

end tan_angle_AMB_l107_107731


namespace minimum_value_expression_l107_107390

theorem minimum_value_expression (n : ℕ) (a b : fin n → ℝ) (h_sum_a : ∑ i, a i ≤ n) (h_sum_b : ∑ j, b j ≤ n) :
  ∃ (E : ℝ), E = n^2 ∧ E = sqrt 2 * (∑ i j, sqrt (1 / a i + 1 / b j)) - (∑ i j, sqrt ((a i ^ 2 + b j ^ 2) / (a i ^ 2 * b j + a i * b j ^ 2))) := 
  by
    sorry

end minimum_value_expression_l107_107390


namespace minimum_moves_black_white_swap_l107_107946

-- Define an initial setup of the chessboard
def initial_positions_black := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8)]
def initial_positions_white := [(8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8)]

-- Define chess rules, positions, and switching places
def black_to_white_target := initial_positions_white
def white_to_black_target := initial_positions_black

-- Define a function to count minimal moves (trivial here just for the purpose of this statement)
def min_moves_to_switch_positions := 23

-- The main theorem statement proving necessity of at least 23 moves
theorem minimum_moves_black_white_swap :
  ∀ (black_positions white_positions : List (ℕ × ℕ)),
  black_positions = initial_positions_black →
  white_positions = initial_positions_white →
  min_moves_to_switch_positions ≥ 23 :=
by
  sorry

end minimum_moves_black_white_swap_l107_107946


namespace uniquely_3_colorable_min_edges_l107_107244

-- Definitions and conditions
variable {V E : Type}
variable [DecidableEq V]

structure Graph (V : Type) :=
  (adj : V → V → Prop)
  (symm : ∀ {x y}, adj x y → adj y x)
  (loopless : ∀ x, ¬ adj x x)

def is_k_colorable (G : Graph V) (k : ℕ) : Prop :=
  ∃ (coloring : V → Fin k), ∀ (x y : V), G.adj x y → coloring x ≠ coloring y

def uniquely_k_colorable (G : Graph V) (k : ℕ) : Prop :=
  is_k_colorable G k ∧ ∀ (coloring₁ coloring₂ : V → Fin k),
                   (∀ x y, G.adj x y → coloring₁ x ≠ coloring₂ y) →
                   (∀ x, coloring₁ x = coloring₂ x)

noncomputable def edge_count (G : Graph V) : ℕ :=
  Fintype.card {e : V × V // G.adj e.1 e.2}

-- The theorem to be proved
theorem uniquely_3_colorable_min_edges (G : Graph V) (n : ℕ)
  (h1 : Fintype.card V = n) (h2 : uniquely_k_colorable G 3) (h3 : 3 ≤ n) :
  edge_count G ≥ 2 * n - 3 :=
sorry

end uniquely_3_colorable_min_edges_l107_107244


namespace vector_sum_solve_for_m_n_l107_107480

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Problem 1: Vector sum
theorem vector_sum : 3 • a + b - 2 • c = (0, 6) :=
by sorry

-- Problem 2: Solving for m and n
theorem solve_for_m_n (m n : ℝ) (hm : a = m • b + n • c) :
  m = 5 / 9 ∧ n = 8 / 9 :=
by sorry

end vector_sum_solve_for_m_n_l107_107480


namespace binomial_600_600_l107_107309

theorem binomial_600_600 : nat.choose 600 600 = 1 :=
by
  -- Given the condition that binomial coefficient of n choose n is 1 for any non-negative n
  have h : ∀ n : ℕ, nat.choose n n = 1 := sorry
  -- Applying directly to the specific case n = 600
  exact h 600

end binomial_600_600_l107_107309


namespace math_question_l107_107776

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l107_107776


namespace proveArithmeticSeq_l107_107477

def isArithmeticSeq (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def arithmeticSeq (a : Nat → Real) (a₁ a₈ a₁₅: Real) (d : Real) : Prop :=
a 0 = a₁ ∧ 
a 7 = a₈ ∧ 
a 14 = a₁₅ ∧
isArithmeticSeq a ∧ 
a₁ + 3 * a₈ + a₁₅ = 120

theorem proveArithmeticSeq (a : Nat → Real) (a₁ a₉ a₁₀ : Real) (d : Real) :
  (arithmeticSeq a a₁ a a₉ a₁₀) → 2 * a₉ - a₁₀ = 20 :=
by sorry

end proveArithmeticSeq_l107_107477


namespace projection_b_onto_a_l107_107435

open EuclideanSpace

def a : ℝ^3 := ![0, 1, 1]
def b : ℝ^3 := ![1, 1, 0]
def projection (u v : ℝ^3) : ℝ^3 := (inner u v / inner u u) • u

theorem projection_b_onto_a :
  projection a b = ![0, 1/2, 1/2] :=
by
  sorry

end projection_b_onto_a_l107_107435


namespace distinct_ordered_pairs_count_l107_107407

theorem distinct_ordered_pairs_count :
  {ab : ℕ × ℕ // ab.1 % 2 = 0 ∧ ab.1 + ab.2 = 52 ∧ 0 < ab.1 ∧ 0 < ab.2}.to_finset.card = 25 :=
by
  sorry

end distinct_ordered_pairs_count_l107_107407


namespace math_question_l107_107772

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l107_107772


namespace shape_area_l107_107021

-- Define the conditions as Lean definitions
def side_length : ℝ := 3
def num_squares : ℕ := 4

-- Prove that the area of the shape is 36 cm² given the conditions
theorem shape_area : num_squares * (side_length * side_length) = 36 := by
    -- The proof is skipped with sorry
    sorry

end shape_area_l107_107021


namespace median_equality_and_range_inequality_l107_107770

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l107_107770


namespace smallest_positive_period_of_f_maximum_value_of_f_zeros_of_f_l107_107412

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin (2 * x + (π / 6)) - 1

def smallest_positive_period (f : ℝ → ℝ) : ℝ :=
  π

theorem smallest_positive_period_of_f :
  (∃ T > 0, ∀ x ∈ ℝ, f (x + T) = f x) ∧
  (∀ T' > 0, T' < π → ∃ x ∈ ℝ, f (x + T') ≠ f x) :=
  sorry

theorem maximum_value_of_f :
  (∀ x ∈ ℝ, f x ≤ 1) ∧ (∃ x ∈ ℝ, f x = 1) :=
  sorry

theorem zeros_of_f :
  {x | f x = 0} = {x | ∃ k : ℤ, x = k * π ∨ x = k * π + π/3} :=
  sorry

end smallest_positive_period_of_f_maximum_value_of_f_zeros_of_f_l107_107412


namespace sqrt_six_eq_l107_107707

theorem sqrt_six_eq (x : ℝ) (hx : Real.sqrt (Real.pow x (Real.sqrt (Real.pow x 4 / 3) * 3) = 2) :
  x = Real.pow 2 (18 / 7) :=
by
  sorry

end sqrt_six_eq_l107_107707


namespace minimize_distance_postman_l107_107661

-- Let x be a function that maps house indices to coordinates.
def optimalPostOfficeLocation (n: ℕ) (x : ℕ → ℝ) : ℝ :=
  if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2)

theorem minimize_distance_postman (n: ℕ) (x : ℕ → ℝ)
  (h_sorted : ∀ i j, i < j → x i < x j) :
  optimalPostOfficeLocation n x = if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2) := 
  sorry

end minimize_distance_postman_l107_107661


namespace isosceles_triangle_AMC_l107_107804

-- Define the basic geometric objects and properties
axiom Triangle (A B C: Type) -- A, B, C are vertices of triangle ABC
axiom Exterior_angle_bisector (B D: Type) (angle_ABC: Type) : Prop -- D lies on the exterior bisector of angle ABC
axiom Midpoint (A B M: Type) : Prop -- M is the midpoint of segment AB
axiom Angle (A B C: Type) (a: ℝ) : Prop -- ∠ABC = a degrees

-- Define the specific conditions given in the problem
variables (A B C D M: Type)
variables [Triangle A B C]
variables [Exterior_angle_bisector B D (Angle B A C 60)]
variables [Angle B C D 60]
variables [Angle (Angle B A C 60)]
variables [Angle B D C 120] -- 180 - 60 = 120 (for external angle)
variables [Midpoint B D M]
variables (AB_length : ℝ)
variable [CD_eq_2AB : (CD = 2 * AB_length)]

-- Statement to prove
theorem isosceles_triangle_AMC : 
  let AMC_isosceles := Triangle_is_isosceles A M C in
  AMC_isosceles 
:= sorry

end isosceles_triangle_AMC_l107_107804


namespace max_value_of_f_l107_107340

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 2 * x + 1

theorem max_value_of_f : ∃ x ∈ Icc (-2 : ℝ) 2, f x = 9 :=
by
  use 2
  split
  { linarith }
  { norm_num }

end max_value_of_f_l107_107340


namespace correct_statements_l107_107249

noncomputable def f : ℝ → ℝ := sorry

axiom f_property1 (x : ℝ) : f (2 - x) = f (2 + x) + 4 * x
axiom f_symmetry (x : ℝ) : f (2*x + 1) = 4 - f (-2*x + 1)

theorem correct_statements :
  (∀ x, f(2 - x) = f(2 + x) + 4 * x) ∧
  (∀ x, f(2*x + 1) = 4 - f(-2*x + 1)) →
  (∀ t, 2 - t = t → f(t) = 2) ∧
  ¬ (∃ x, f(x + 4) = f(x)) ∧
  f(2) ≠ 4 ∧
  f(2023) = -4042 :=
begin
  intros h,
  -- These steps would include the necessary deductions to prove the statement
  sorry
end

end correct_statements_l107_107249


namespace largest_possible_number_divisible_by_9_l107_107537

def large_number : ℕ := 321321321321

def erase_last_two_3s (n : ℕ) : ℕ :=
  if n == 321321321321 then 32132132121 else n

lemma divisibility_by_9 (n : ℕ) : 
  (∑ digit in n.digits 10, digit) % 9 = 0 ↔ n % 9 = 0 := 
begin
  sorry
end

theorem largest_possible_number_divisible_by_9 :
  ∃ n', erase_last_two_3s large_number = n' ∧ n' % 9 = 0 :=
by {
  use 32132132121,
  split,
  {
    refl,
  },
  {
    rw divisibility_by_9,
    norm_num,
  }
}

end largest_possible_number_divisible_by_9_l107_107537


namespace tan_theta_eq_neg_two_l107_107455

theorem tan_theta_eq_neg_two (f : ℝ → ℝ) (θ : ℝ) 
  (h₁ : ∀ x, f x = Real.sin (2 * x + θ)) 
  (h₂ : ∀ x, f x + 2 * Real.cos (2 * x + θ) = -(f (-x) + 2 * Real.cos (2 * (-x) + θ))) :
  Real.tan θ = -2 :=
by
  sorry

end tan_theta_eq_neg_two_l107_107455


namespace dance_contradiction_l107_107288

variable {Boy Girl : Type}
variable {danced_with : Boy → Girl → Prop}

theorem dance_contradiction
    (H1 : ¬ ∃ g : Boy, ∀ f : Girl, danced_with g f)
    (H2 : ∀ f : Girl, ∃ g : Boy, danced_with g f) :
    ∃ (g g' : Boy) (f f' : Girl),
        danced_with g f ∧ ¬ danced_with g f' ∧
        danced_with g' f' ∧ ¬ danced_with g' f :=
by
  -- Proof will be inserted here
  sorry

end dance_contradiction_l107_107288


namespace smallest_prime_12_less_perfect_square_l107_107146

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107146


namespace candidates_selected_difference_l107_107884

theorem candidates_selected_difference :
  let candidates_state_A := 8000
  let candidates_state_B := 8000
  let selection_rate_A := 6 / 100.0
  let selection_rate_B := 7 / 100.0
  let selected_A := selection_rate_A * candidates_state_A
  let selected_B := selection_rate_B * candidates_state_B
  selected_B - selected_A = 80 :=
by
  let candidates_state_A := 8000
  let candidates_state_B := 8000
  let selection_rate_A := 6 / 100.0
  let selection_rate_B := 7 / 100.0
  let selected_A := selection_rate_A * candidates_state_A
  let selected_B := selection_rate_B * candidates_state_B
  have h : selected_B - selected_A = 80
  sorry

end candidates_selected_difference_l107_107884


namespace binary_multiplication_l107_107718

theorem binary_multiplication : (10101 : ℕ) * (101 : ℕ) = 1101001 :=
by sorry

end binary_multiplication_l107_107718


namespace circles_intersect_l107_107430

-- Definitions for the two circles
def circle1_center : ℝ × ℝ := (1, -3)
def circle1_radius : ℝ := 2

def circle2_center : ℝ × ℝ := (2, -1)
def circle2_radius : ℝ := 1

-- Distance formula between two points in the plane
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The Lean 4 statement for the proof problem
theorem circles_intersect :
  let d := distance circle1_center circle2_center in
  circle1_radius - circle2_radius < d ∧ d < circle1_radius + circle2_radius :=
by 
  let d := distance circle1_center circle2_center
  sorry

end circles_intersect_l107_107430


namespace value_of_a_minus_b_l107_107383

variables (a b : ℚ)

theorem value_of_a_minus_b (h1 : |a| = 5) (h2 : |b| = 2) (h3 : |a + b| = a + b) : a - b = 3 ∨ a - b = 7 :=
sorry

end value_of_a_minus_b_l107_107383


namespace odd_function_condition_l107_107576

noncomputable def f (x a b : ℝ) : ℝ := x * |x + a| + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ a^2 + b^2 = 0 :=
by
  sorry

end odd_function_condition_l107_107576


namespace min_additional_marbles_needed_l107_107913

-- Definition of the conditions
def friends : ℕ := 15
def initial_marbles : ℕ := 60

-- The problem statement
theorem min_additional_marbles_needed :
  ∃ (additional_marbles : ℕ), (∀ (m : ℕ), m ≥ 0 → (∑ k in finset.range(friends + 1), k) - initial_marbles + m = additional_marbles) := sorry

end min_additional_marbles_needed_l107_107913


namespace projection_b_onto_a_l107_107434

open EuclideanSpace

def a : ℝ^3 := ![0, 1, 1]
def b : ℝ^3 := ![1, 1, 0]
def projection (u v : ℝ^3) : ℝ^3 := (inner u v / inner u u) • u

theorem projection_b_onto_a :
  projection a b = ![0, 1/2, 1/2] :=
by
  sorry

end projection_b_onto_a_l107_107434


namespace smallest_prime_12_less_than_perfect_square_l107_107211

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l107_107211


namespace coefficient_of_x3_l107_107022

theorem coefficient_of_x3 (x : ℝ) : 
  let expansion := x * (1 + x)^6 in 
  let coeff_x3 := 15 in
  (natDegree (expansion.coeff 3) = coeff_x3) :=
sorry

end coefficient_of_x3_l107_107022


namespace general_term_a_general_form_b_T_n_formula_l107_107498

open Real

noncomputable def a_n (n : ℕ) : ℝ :=
if n < 1 then 0 else 2 * n - 1

noncomputable def b_n (n : ℕ) : ℝ :=
if n < 1 then 0 else (1 / 2)^(n - 1)

noncomputable def c_n (n : ℕ) : ℝ :=
if n < 1 then 0 else (2 * n - 1) * 2^(n - 1)

noncomputable def S_n (n : ℕ) : ℝ :=
∑ i in finset.range n, b_n (i + 1)

noncomputable def T_n (n : ℕ) : ℝ :=
∑ i in finset.range n, c_n (i + 1)

theorem general_term_a (n : ℕ) (h3 : a_n 3 = 5) (h5 : a_n 5 = 9) : a_n n = 2 * n - 1 := sorry

theorem general_form_b (n : ℕ) (h : ∀ n, S_n n + b_n n = 2) : b_n n = (1 / 2)^(n - 1) := sorry

theorem T_n_formula (n : ℕ) : T_n n = 3 + (2 * n - 3) * 2^n := sorry

end general_term_a_general_form_b_T_n_formula_l107_107498


namespace not_perfect_squares_l107_107709

theorem not_perfect_squares :
  (∀ x : ℝ, x * x ≠ 8 ^ 2041) ∧ (∀ y : ℝ, y * y ≠ 10 ^ 2043) :=
by
  sorry

end not_perfect_squares_l107_107709


namespace arithmetic_sequence_common_difference_l107_107893

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 1 = 13) (h4 : a_n 4 = 1) : 
  ∃ d : ℤ, d = -4 := by
  sorry

end arithmetic_sequence_common_difference_l107_107893


namespace cos_sum_identity_l107_107967

noncomputable theory

open Complex

theorem cos_sum_identity :
  let ω := exp (2 * π * I / 17) in
  ω^17 = 1 →
  (∃ω_conj, ω_conj = conj ω ∧ ω_conj = 1 / ω) →
  cos (2 * π / 17) + cos (6 * π / 17) + cos (8 * π / 17) = (sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_identity_l107_107967


namespace tetrahedron_through_hole_tetrahedron_cannot_through_hole_l107_107296

/--
A regular tetrahedron with edge length 1 can pass through a circular hole if and only if the radius \( R \) is at least 0.4478, given that the thickness of the hole can be neglected.
-/

theorem tetrahedron_through_hole (R : ℝ) (h1 : R = 0.45) : true :=
by sorry

theorem tetrahedron_cannot_through_hole (R : ℝ) (h1 : R = 0.44) : false :=
by sorry

end tetrahedron_through_hole_tetrahedron_cannot_through_hole_l107_107296


namespace initial_percentage_is_30_l107_107269

noncomputable def initial_percentage_liquid_X (P : ℝ) : ℝ :=
  let initial_weight := 10
  let liquid_X_initial := P / 100 * initial_weight
  let remaining_weight := initial_weight - 2
  let solution_after_evaporation := 8
  let added_solution := 2
  let total_weight_after_adding := solution_after_evaporation + added_solution
  let liquid_X_additional := P / 100 * added_solution
  let total_liquid_X := liquid_X_initial + liquid_X_additional
  if total_weight_after_adding = initial_weight then
    0.36 * total_weight_after_adding = total_liquid_X
  else
    0

theorem initial_percentage_is_30 : initial_percentage_liquid_X 30 = 1 :=
by
  sorry

end initial_percentage_is_30_l107_107269


namespace cara_age_is_40_l107_107297

-- Defining the ages of Cara, her mom, and her grandmother
variables (cara_age mom_age grandmother_age - : ℕ)
variable (h1 : cara_age = mom_age - 20)
variable (h2 : mom_age = grandmother_age - 15)
variable (h3 : grandmother_age = 75)

-- The aim is to prove that Cara's age is 40
theorem cara_age_is_40 : cara_age = 40 :=
by
  sorry

end cara_age_is_40_l107_107297


namespace binom_600_600_l107_107323

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l107_107323


namespace sqrt_product_simplify_l107_107552

theorem sqrt_product_simplify : 
  (sqrt (3 * 5) * sqrt (5^4 * 3^3) = 45 * sqrt 5) :=
by
  sorry

end sqrt_product_simplify_l107_107552


namespace cos_sum_identity_l107_107968

noncomputable theory

open Complex

theorem cos_sum_identity :
  let ω := exp (2 * π * I / 17) in
  ω^17 = 1 →
  (∃ω_conj, ω_conj = conj ω ∧ ω_conj = 1 / ω) →
  cos (2 * π / 17) + cos (6 * π / 17) + cos (8 * π / 17) = (sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_identity_l107_107968


namespace kiran_money_l107_107634

theorem kiran_money (R G K : ℕ) (h1: R / G = 6 / 7) (h2: G / K = 6 / 15) (h3: R = 36) : K = 105 := by
  sorry

end kiran_money_l107_107634


namespace smallest_prime_12_less_than_square_l107_107191

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107191


namespace cricketer_average_score_l107_107103

theorem cricketer_average_score
  (avg1 : ℕ)
  (matches1 : ℕ)
  (avg2 : ℕ)
  (matches2 : ℕ)
  (total_matches : ℕ)
  (total_avg : ℕ)
  (h1 : avg1 = 20)
  (h2 : matches1 = 2)
  (h3 : avg2 = 30)
  (h4 : matches2 = 3)
  (h5 : total_matches = 5)
  (h6 : total_avg = 26)
  (h_total_runs : total_avg * total_matches = avg1 * matches1 + avg2 * matches2) :
  total_avg = 26 := 
sorry

end cricketer_average_score_l107_107103


namespace min_value_of_g_l107_107815

def f (x : ℝ) : ℝ := x * Real.log x

def g (x : ℝ) : ℝ := 2 * Real.log x + x + 3 / x

theorem min_value_of_g : (∀ x : ℝ, 0 < x → 2 * f(x) + x^2 - a * x + 3 ≥ 0) → a ≤ 4 :=
by
  sorry

end min_value_of_g_l107_107815


namespace smallest_prime_12_less_perfect_square_l107_107144

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107144


namespace problem_proof_l107_107371

variable (a b : ℝ)

-- Given conditions 
hypothesis h1 : a ≠ b
hypothesis h2 : a^2 - 13 * a + 1 = 0
hypothesis h3 : b^2 - 13 * b + 1 = 0

-- The theorem to be proved
theorem problem_proof : (b / (1 + b) + (a * (a + 1)) / ((a + 1)^2) = 1) :=
by
  -- The detailed proof goes here
  sorry

end problem_proof_l107_107371


namespace find_k_value_l107_107831

theorem find_k_value (k : ℝ) (x : ℝ) :
  -x^2 - (k + 12) * x - 8 = -(x - 2) * (x - 4) → k = -18 :=
by
  intro h
  sorry

end find_k_value_l107_107831


namespace ellipse_equation_l107_107379

theorem ellipse_equation (a b : ℝ) (a_gt_b : a > b) (b_pos : b > 0) (eccentricity : a / b = real.sqrt 2 / 2) :
  a^2 = 2 ∧ b^2 = 1 → ∀ x y, (x^2 / 2 + y^2 = 1) := 
sorry

end ellipse_equation_l107_107379


namespace number_of_family_members_l107_107566

-- Definitions based on the conditions
def total_family_members (n : ℕ) : Prop := 
  ∃ S : ℕ, 
    (S = 29 * n) ∧ 
    (S = (33 * n) - 28)

-- The proof goal
theorem number_of_family_members :
  total_family_members 7 :=
by
  use 203  -- S = 29 * 7
  split
  case left =>
    -- S = 29 * 7
    exact rfl
  case right =>
    -- S = 33 * 7 - 28
    have h : 203 = 33 * 7 - 28 := by
      calc
        33 * 7 - 28 = 231 - 28  : by norm_num
                ... = 203       : by norm_num
    exact h

end number_of_family_members_l107_107566


namespace minimum_value_of_M_l107_107511

theorem minimum_value_of_M (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (∃ c : ℝ, c = 3 * y ∧ (M = 3)) where
  M = (4 * x) / (x + (3 * y)) + (3 * y) / x :=
sorry

end minimum_value_of_M_l107_107511


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107792

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107792


namespace smallest_prime_less_than_perfect_square_is_13_l107_107197

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107197


namespace dog_food_amount_l107_107680

theorem dog_food_amount (x : ℕ) (h1 : 3 * x + 6 = 15) : x = 3 :=
by {
  sorry
}

end dog_food_amount_l107_107680


namespace parallel_lines_via_plane_l107_107374

variable {α : Type} [Plane α] {m n : Line α}

theorem parallel_lines_via_plane (h₁ : ∃ α, m ∥ α) (h₂ : ∃ α, n ∥ α) : m ∥ n :=
sorry

end parallel_lines_via_plane_l107_107374


namespace polar_coordinate_equation_l107_107355

theorem polar_coordinate_equation (x y α : ℝ) (h : x * Real.cos α + y * Real.sin α = 0) :
  ∃ θ : ℝ, θ = α - π / 2 :=
begin
  sorry
end

end polar_coordinate_equation_l107_107355


namespace num_divisors_720_l107_107286

-- Define the number 720 and its prime factorization
def n : ℕ := 720
def pf : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1)]

-- Define the function to calculate the number of divisors from prime factorization
def num_divisors (pf : List (ℕ × ℕ)) : ℕ :=
  pf.foldr (λ p acc => acc * (p.snd + 1)) 1

-- Statement to prove
theorem num_divisors_720 : num_divisors pf = 30 :=
  by
  -- Placeholder for the actual proof
  sorry

end num_divisors_720_l107_107286


namespace find_B_find_perimeter_l107_107026

variables {a b c : ℝ} {A B C : ℝ}

-- Conditions
axiom triangle_condition_1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B
axiom b_value : b = 2 * Real.sqrt 3
axiom area_of_triangle : 1/2 * a * c * Real.sin B = 2 * Real.sqrt 3

-- Goals
theorem find_B (h1 : triangle_condition_1) (h2 : b_value) : B = Real.pi / 3 :=
sorry

theorem find_perimeter (h1 : triangle_condition_1) (h2 : b_value) (h3 : area_of_triangle) : a + b + c = 6 + 2 * Real.sqrt 3 :=
sorry

end find_B_find_perimeter_l107_107026


namespace value_of_a_l107_107413

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then log 3 x else x^2

theorem value_of_a (a : ℝ) (h : f (-1) = 2 * f a) : a = sqrt 3 ∨ a = - (sqrt 2 / 2) :=
by
  have h_neg1 : f (-1) = 1 := by simp [f]
  rw [h_neg1] at h
  have h_fa : f a = 1 / 2 := by linarith
  split_ifs at h_fa
  -- Case a > 0
  { have log_eq : log 3 a = 1 / 2 := h_fa
    have a_pos : a = sqrt 3 := sorry  -- Solve using log properties
    exact Or.inl a_pos
  }
  -- Case a ≤ 0
  { have sq_eq : a^2 = 1 / 2 := h_fa
    have a_neg : a = - sqrt (1 / 2) := sorry -- Solve by taking square roots
    exact Or.inr a_neg
  }

end value_of_a_l107_107413


namespace min_value_expr_ge_52_l107_107717

open Real

theorem min_value_expr_ge_52 (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (sin x + 3 * (1 / sin x)) ^ 2 + (cos x + 3 * (1 / cos x)) ^ 2 ≥ 52 := 
by
  sorry

end min_value_expr_ge_52_l107_107717


namespace sum_of_first_12_even_numbers_is_156_l107_107129

theorem sum_of_first_12_even_numbers_is_156 :
  (∑ k in Finset.range 12, 2 * (k + 1)) = 156 := by
  sorry

end sum_of_first_12_even_numbers_is_156_l107_107129


namespace discount_percentage_l107_107030

theorem discount_percentage (coach_cost sectional_cost other_cost paid : ℕ) 
  (h1 : coach_cost = 2500) 
  (h2 : sectional_cost = 3500) 
  (h3 : other_cost = 2000) 
  (h4 : paid = 7200) : 
  ((coach_cost + sectional_cost + other_cost - paid) * 100) / (coach_cost + sectional_cost + other_cost) = 10 :=
by
  sorry

end discount_percentage_l107_107030


namespace simplify_expression_l107_107227

theorem simplify_expression (a : ℝ) (h : a ≠ 2) :
    ( (root 3 (sqrt 5 - sqrt 3)) * (root 6 (8 + 2 * sqrt 15)) - (root 3 a) ) /
    ( (root 3 (sqrt 20 + sqrt 12)) * (root 6 (8 - 2 * sqrt 15)) - 2 * (root 3 (2 * a)) + (root 3 (a ^ 2)) )
    = 1 / (root 3 2 - root 3 a) := 
sorry

end simplify_expression_l107_107227


namespace difference_of_results_l107_107679

theorem difference_of_results (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (h_diff: a ≠ b) :
  (70 * a - 7 * a) - (70 * b - 7 * b) = 0 :=
by
  sorry

end difference_of_results_l107_107679


namespace samantha_more_cat_food_l107_107545

noncomputable def cans_of_cat_food : ℕ := 12 * 8
noncomputable def cans_of_dog_food : ℕ := 7 * 5
noncomputable def cans_of_bird_food : ℕ := 4 * 3

theorem samantha_more_cat_food :
  cans_of_cat_food = (cans_of_dog_food + cans_of_bird_food) + 49 :=
by
  unfold cans_of_cat_food cans_of_dog_food cans_of_bird_food
  norm_num
  -- This line would be replaced with the actual proof steps
  sorry

end samantha_more_cat_food_l107_107545


namespace shipment_cost_l107_107937

-- Define the conditions
def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def shipping_cost_per_crate : ℝ := 1.5
def surcharge_per_crate : ℝ := 0.5
def flat_fee : ℝ := 10

-- Define the question as a theorem
theorem shipment_cost : 
  let crates := total_weight / weight_per_crate
  let cost_per_crate := shipping_cost_per_crate + surcharge_per_crate
  let total_cost_crates := crates * cost_per_crate
  let total_cost := total_cost_crates + flat_fee
  total_cost = 46 := by
  -- Proof omitted
  sorry

end shipment_cost_l107_107937


namespace max_imaginary_part_angle_l107_107671

def poly (z : Complex) : Complex := z^6 - z^4 + z^2 - 1

theorem max_imaginary_part_angle :
  ∃ θ : Real, θ = 45 ∧ 
  (∃ z : Complex, poly z = 0 ∧ ∀ w : Complex, poly w = 0 → w.im ≤ z.im)
:= sorry

end max_imaginary_part_angle_l107_107671


namespace rabbits_distribution_l107_107544

def num_ways_to_distribute : ℕ :=
  20 + 390 + 150

theorem rabbits_distribution :
  num_ways_to_distribute = 560 := by
  sorry

end rabbits_distribution_l107_107544


namespace time_for_trains_to_cross_each_other_l107_107135

def train_lengths : ℝ × ℝ := (200, 300)
def train_speeds : ℝ × ℝ := (60, 40)
def km_per_hr_to_m_per_s (v : ℝ) : ℝ := v * (5/18)
def relative_speed (v1 : ℝ) (v2 : ℝ) : ℝ := km_per_hr_to_m_per_s (v1 + v2)
def total_distance (l1 : ℝ, l2 : ℝ) : ℝ := l1 + l2
def time_to_cross (d : ℝ) (v : ℝ) : ℝ := d / v

theorem time_for_trains_to_cross_each_other :
  let (l1, l2) := train_lengths
  let (v1, v2) := train_speeds
  time_to_cross (total_distance l1 l2) (relative_speed v1 v2) = 18 :=
  by
    let (l1, l2) := train_lengths
    let (v1, v2) := train_speeds
    let rel_speed := relative_speed v1 v2
    have conversion_factor : km_per_hr_to_m_per_s 1 = 5 / 18 := by simp [km_per_hr_to_m_per_s]
    have rel_speed_val : relative_speed 60 40 = (100 * (5 / 18)) := by simp [relative_speed, km_per_hr_to_m_per_s]
    have dist := total_distance l1 l2
    have t := time_to_cross dist rel_speed
    show t = 18 from sorry

end time_for_trains_to_cross_each_other_l107_107135


namespace median_eq_range_le_l107_107743

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l107_107743


namespace perimeter_of_z_shape_l107_107058

noncomputable def perimeter_of_shape (z : ℂ) : ℝ :=
  if arg(z) = arg(Complex.i * z + Complex.i) then (Real.pi / 2) else 0

theorem perimeter_of_z_shape {z : ℂ} (h : arg(z) = arg(Complex.i * z + Complex.i)) :
  perimeter_of_shape z = Real.pi / 2 :=
by
  unfold perimeter_of_shape
  simp [h]
  sorry 

end perimeter_of_z_shape_l107_107058


namespace A_k_elements_l107_107036

noncomputable def seq (k : ℕ) : ℕ → ℕ
| 0 => 1
| n + 1 => seq k n + (seq k n).root k

def A_k (k : ℕ) : set ℕ :=
  {m | ∃ n, seq k n = m}

theorem A_k_elements (k : ℕ) (hk : k ≥ 1) : A_k k = {2^r | r : ℕ} := by
  sorry

end A_k_elements_l107_107036


namespace imaginary_part_of_z_is_neg1_l107_107453

-- Defining the conditions as given
def z : ℂ := complex.I * (-1 + complex.I)

-- Stating the proof goal
theorem imaginary_part_of_z_is_neg1 (z_def : z = complex.I * (-1 + complex.I)) :
  complex.im z = -1 := 
by {
  rw z_def,
  -- Proof steps would go here, but we skip them using sorry
  sorry
}

end imaginary_part_of_z_is_neg1_l107_107453


namespace solution_set_inequality_l107_107373

def f (x : ℝ) : ℝ := if x ≥ 0 then 1 else -1

def g (x : ℝ) : ℝ := f (x + 2)

theorem solution_set_inequality :
  {x : ℝ | x + (x + 2) * g x ≤ 5} = {x : ℝ | x ≤ 3 / 2} :=
by
  sorry

end solution_set_inequality_l107_107373


namespace monotonic_decreasing_interval_l107_107580

theorem monotonic_decreasing_interval :
  let y := λ x : ℝ, 3 * Real.sin (2 * x + Real.pi / 4)
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi → (Real.pi / 8 ≤ x ∧ x ≤ 5 * Real.pi / 8) :=
sorry

end monotonic_decreasing_interval_l107_107580


namespace math_question_l107_107775

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l107_107775


namespace time_difference_between_shoes_l107_107534

-- Define the conditions
def time_per_mile_regular := 10
def time_per_mile_new := 13
def distance_miles := 5

-- Define the theorem to be proven
theorem time_difference_between_shoes :
  (distance_miles * time_per_mile_new) - (distance_miles * time_per_mile_regular) = 15 :=
by
  sorry

end time_difference_between_shoes_l107_107534


namespace alternating_cubes_l107_107980

theorem alternating_cubes (x : ℝ) (n : ℕ) (hx : x ≠ 0) : 
  ∃ y : ℝ, y = x ^ ((-9) ^ (3 ^ n)) :=
by
  use x ^ ((-9) ^ (3 ^ n))
  sorry

end alternating_cubes_l107_107980


namespace tan_sub_pi_six_l107_107727

theorem tan_sub_pi_six (x : ℝ) (h : sin(π/3 - x) = 1/2 * cos(x - π/2)) : 
  tan (x - π/6) = sqrt 3 / 9 :=
by
  sorry

end tan_sub_pi_six_l107_107727


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107168

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107168


namespace decreasing_interval_eqn_l107_107838

def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem decreasing_interval_eqn {a : ℝ} : (∀ x : ℝ, x < 6 → deriv (f a) x < 0) ↔ a ≥ 6 :=
sorry

end decreasing_interval_eqn_l107_107838


namespace no_beverages_l107_107885

noncomputable def businessmen := 30
def coffee := 15
def tea := 13
def water := 6
def coffee_tea := 7
def tea_water := 3
def coffee_water := 2
def all_three := 1

theorem no_beverages (businessmen coffee tea water coffee_tea tea_water coffee_water all_three):
  businessmen - (coffee + tea + water - coffee_tea - tea_water - coffee_water + all_three) = 7 :=
by sorry

end no_beverages_l107_107885


namespace proof_of_problem_l107_107689

def problem_statement : Prop :=
  2 * Real.cos (Real.pi / 4) + abs (Real.sqrt 2 - 3)
  - (1 / 3) ^ (-2 : ℤ) + (2021 - Real.pi) ^ 0 = -5

theorem proof_of_problem : problem_statement :=
by
  sorry

end proof_of_problem_l107_107689


namespace base_9_contains_6_or_7_count_l107_107849

def contains_digit_6_or_7 (n : ℕ) : Prop :=
  let digits := (Nat.digits 9 n) in
  List.any digits (λ d, d = 6 ∨ d = 7)

theorem base_9_contains_6_or_7_count :
  ∃ k : ℕ, k = 386 ∧ 
           k = (Finset.filter (λ n, contains_digit_6_or_7 n)
                              (Finset.range 730)).card :=
by
  sorry

end base_9_contains_6_or_7_count_l107_107849


namespace solution_set_of_inequality_l107_107357

theorem solution_set_of_inequality :
  { x : ℝ | (x - 4) / (3 - 2*x) < 0 ∧ 3 - 2*x ≠ 0 } = { x : ℝ | x < 3 / 2 ∨ x > 4 } :=
sorry

end solution_set_of_inequality_l107_107357


namespace volume_tetrahedron_formula_l107_107542

-- Definitions of the problem elements
def distance (A B C D : Point) : ℝ := sorry
def angle (A B C D : Point) : ℝ := sorry
def length (A B : Point) : ℝ := sorry

-- The problem states you need to prove the volume of the tetrahedron
noncomputable def volume_tetrahedron (A B C D : Point) : ℝ := sorry

-- Conditions
variable (A B C D : Point)
variable (d : ℝ) (phi : ℝ) -- d = distance between lines AB and CD, phi = angle between lines AB and CD

-- Question reformulated as a proof statement
theorem volume_tetrahedron_formula (h1 : d = distance A B C D)
                                   (h2 : phi = angle A B C D) :
  volume_tetrahedron A B C D = (d * length A B * length C D * Real.sin phi) / 6 :=
sorry

end volume_tetrahedron_formula_l107_107542


namespace tennis_tournament_total_matches_l107_107268

theorem tennis_tournament_total_matches :
  ∀ (n : ℕ), n = 120 →
    let players := n,
        byes := 40,
        remaining_players := players - byes,
        matches_first_round := remaining_players / 2,
        matches_second_round := matches_first_round,
        matches_third_round := matches_second_round / 2,
        matches_fourth_round := matches_third_round / 2,
        matches_fifth_round := matches_fourth_round / 2,
        matches_sixth_round := (matches_fifth_round + 1) / 2,
        matches_seventh_round := 2
    in matches_first_round + matches_second_round + matches_third_round + matches_fourth_round +
       matches_fifth_round + matches_sixth_round + matches_seventh_round = 120 :=
begin
  intros n hn, simp only [], intros,
  rw hn,
  simp only [nat.succ_eq_add_one,
             nat.add_sub_assoc (nat.le_succ 119) (nat.le_succ 79),
             nat.add_sub_cancel_left],
  sorry,
end

end tennis_tournament_total_matches_l107_107268


namespace arithmetic_sequence_general_formula_T_n_bounds_l107_107517

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ)
    (h1 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
    (h2 : S 3 = 9)
    (h3 : 0 < d)
    (h4 : ∃ g, 2 * a 1 = g * (a 3 - 1) ∧ (a 3 - 1) * g = (a 4 + 1)) :
    a = λ n, 2 * n - 1 :=
by
  sorry

theorem T_n_bounds (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℤ) (T : ℕ → ℤ)
    (d : ℤ)
    (h1 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
    (h2 : S 3 = 9)
    (h3 : 0 < d)
    (h4 : ∃ g, 2 * a 1 = g * (a 3 - 1) ∧ (a 3 - 1) * g = (a 4 + 1))
    (h5 : ∀ n, a n = 2 * n - 1)
    (h6 : ∀ n, b n = 1 / (a n * a (n + 1)))
    (h7 : ∀ n, T n = ∑ i in range n, b i) :
    ∀ n, (1 / 3 : ℚ) ≤ T n ∧ T n < (1 / 2 : ℚ) :=
by
  sorry

end arithmetic_sequence_general_formula_T_n_bounds_l107_107517


namespace probability_less_than_mean_l107_107401

open ProbabilityTheory
open MeasureTheory

noncomputable def xi : Measure ℝ := 
  Measure.normal 2 σ^2

theorem probability_less_than_mean :
  ∀ σ > 0, P(ξ < 2) = 0.5 :=
by
  sorry

end probability_less_than_mean_l107_107401


namespace arithmetic_sequence_sum_l107_107016

theorem arithmetic_sequence_sum :
  (∀ (n : ℕ), ∃ d a : ℕ, a_2 = a + d ∧ a_4 + a_7 = (a + 3d) + (a + 6d) ∧ a_4 + a_7 = 15 ∧
    (∀ n, a_n = a + (n-1) * d) ∧ b_n = 2^(a_n-2) → 
    ∑ k in range(1, 11), b k = 2046) :=
by
  sorry

end arithmetic_sequence_sum_l107_107016


namespace sum_of_distinct_solutions_l107_107332

noncomputable def g (x : ℝ) : ℝ := (x^2) / 2 + 2 * x - 1

def satisfies_g3 (x : ℝ) : Prop := g (g (g x)) = 1

def distinct_solutions : set ℝ := {x | satisfies_g3 x}

theorem sum_of_distinct_solutions : 
  (∑ x in distinct_solutions.to_finset, x) = -4 := 
by sorry

end sum_of_distinct_solutions_l107_107332


namespace find_value_simplify_expression_l107_107240

-- Define the first part of the problem
theorem find_value (α : ℝ) (h : Real.tan α = 1/3) : 
  (1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2)) = 2 / 3 := 
  sorry

-- Define the second part of the problem
theorem simplify_expression (α : ℝ) (h : Real.tan α = 1/3) : 
  (Real.tan (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / (Real.cos (-α - π) * Real.sin (-π - α)) = -1 := 
  sorry

end find_value_simplify_expression_l107_107240


namespace binomial_600_600_l107_107313

-- Define a theorem to state the binomial coefficient property and use it to prove the specific case.
theorem binomial_600_600 : nat.choose 600 600 = 1 :=
begin
  -- Binomial property: for any non-negative integer n, (n choose n) = 1
  rw nat.choose_self,
end

end binomial_600_600_l107_107313


namespace problem_solution_l107_107034

def isSolutionCorrect (k n : ℕ) (T : set (fin n → ℕ)) (d : (fin n → ℕ) → (fin n → ℕ) → ℕ) :=
  (n = 2 * k - 1 ∧ k ≥ 6) ∧
  (∀ x ∈ T, ∀ y ∈ T, d x x = 0 ∧ d x y = fintype.card { i : fin n // x i ≠ y i }) ∧
  (∃ S ⊆ T, fintype.card S = 2^k ∧ ∀ x, ∃! y ∈ S, d x y ≤ 3)

theorem problem_solution : ∀ k n : ℕ, n = 2 * k - 1 → k ≥ 6 →
  let T := { f : fin n → ℕ | ∀ i, f i = 0 ∨ f i = 1 } in
  let d := λ (x y : fin n → ℕ), fintype.card { i : fin n // x i ≠ y i } in
  isSolutionCorrect k n T d → n = 23 :=
by
  sorry

end problem_solution_l107_107034


namespace sin_product_impossible_l107_107952

theorem sin_product_impossible (α : ℝ) :
  ¬ (sin α * sin (2 * α) * sin (3 * α) = 4 / 5) :=
by sorry

end sin_product_impossible_l107_107952


namespace median_equality_and_range_inequality_l107_107764

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l107_107764


namespace inequality_problem_l107_107809

theorem inequality_problem
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : a + b + c ≤ 3) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by sorry

end inequality_problem_l107_107809


namespace smallest_prime_12_less_than_square_l107_107187

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107187


namespace find_tax_percentage_l107_107877

-- Definitions based on given conditions
def income_total : ℝ := 58000
def income_threshold : ℝ := 40000
def tax_above_threshold_percentage : ℝ := 0.2
def total_tax : ℝ := 8000

-- Let P be the percentage taxed on the first $40,000
variable (P : ℝ)

-- Formulate the problem as a proof goal
theorem find_tax_percentage (h : total_tax = 8000) :
  P = ((total_tax - (tax_above_threshold_percentage * (income_total - income_threshold))) / income_threshold) * 100 :=
by sorry

end find_tax_percentage_l107_107877


namespace part1_part2_l107_107372

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
noncomputable def g (a : ℝ) : ℝ := a^2 - a - 2

theorem part1 (x : ℝ) : f x 3 > g 3 + 2 ↔ x < -4 ∨ x > 2 := by
sorry

theorem part2 (a : ℝ) : (∀ x, x ∈ set.Ico (-a) 1 → f x a ≤ g a) ↔ a ≥ 3 := by
sorry

end part1_part2_l107_107372


namespace sasha_lives_on_seventh_floor_l107_107090

theorem sasha_lives_on_seventh_floor (N : ℕ) (x : ℕ) 
(h1 : x = (1/3 : ℝ) * N) 
(h2 : N - ((1/3 : ℝ) * N + 1) = (1/2 : ℝ) * N) :
  N + 1 = 7 := 
sorry

end sasha_lives_on_seventh_floor_l107_107090


namespace line_dn_bisects_ab_l107_107519

theorem line_dn_bisects_ab
  (l : Line)
  (A B C D N M : Point)
  (circle : Circle)
  (h1 : Perpendicular l (Segment A B))
  (h2 : OnLine B l)
  (h3 : CenterOnLine circle l)
  (h4 : PassThrough circle A)
  (h5 : Intersect l circle [C, D])
  (h6 : TangentToCircle (Segment (Point A) (Point N)) circle)
  (h7 : TangentToCircle (Segment (Point C) (Point N)) circle)
  (M_def : Intersect (Line (Point D) (Point N)) (Segment (Point A) (Point B)) M) :
  Midpoint M A B := 
sorry

end line_dn_bisects_ab_l107_107519


namespace median_equality_range_inequality_l107_107799

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l107_107799


namespace correct_transformation_l107_107221

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0): (a / b = 2 * a / (2 * b)) :=
by
  sorry

end correct_transformation_l107_107221


namespace proportion_first_number_l107_107451

theorem proportion_first_number (x : ℝ) (h : x / 5 = 0.96 / 8) : x = 0.6 :=
by
  sorry

end proportion_first_number_l107_107451


namespace semifinalists_count_l107_107886

theorem semifinalists_count (n : ℕ) (h : (n - 2) * (n - 3) * (n - 4) = 336) : n = 10 := 
by {
  sorry
}

end semifinalists_count_l107_107886


namespace find_100th_permutation_l107_107638

def five_digits := [1, 2, 3, 4, 5]

def permute (l: List ℕ) : List (List ℕ) :=
  l.permutations

def is_sorted (l: List ℕ) : Prop :=
  l = l.sorted (<=)

def sorted_permutations := (permute five_digits).sorted

def find_nth (n: ℕ) (l: List ℕ) : ℕ :=
  l.nth (n - 1)

theorem find_100th_permutation :
  find_nth 100 sorted_permutations = 51342 :=
sorry

end find_100th_permutation_l107_107638


namespace smallest_prime_12_less_perfect_square_l107_107139

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107139


namespace smallest_prime_less_than_square_l107_107151

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107151


namespace rosa_initial_flowers_l107_107089

def initial_flowers (total_flowers : ℕ) (flowers_given : ℕ) : ℕ :=
  total_flowers - flowers_given

theorem rosa_initial_flowers :
  initial_flowers 90 23 = 67 :=
by
  rw [initial_flowers]
  sorry

end rosa_initial_flowers_l107_107089


namespace inequality_seq_l107_107733

-- Define the sum of the first n terms of the sequence
def sum_seq (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in range (n + 1), a i

-- Given: ∀ n, 2 * a_n = S_n + 2 
def sequence_property (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, 2 * a n = S n + 2

-- The sequence a_n = 2^n
def sequence (n : ℕ) : ℕ := 2 ^ n

-- The sum of the sequence of first n terms
def Sn (n : ℕ) : ℕ := sum_seq sequence n

-- Prove: ∀ n ≥ 5, a_n > n^2
theorem inequality_seq : ∀ n ≥ 5, sequence n > n^2 := by
  sorry

end inequality_seq_l107_107733


namespace sum_of_angles_B_and_D_l107_107860

theorem sum_of_angles_B_and_D
  (A B C D F G : ℝ)
  (h1 : A = 30)
  (h2 : G = A + C)
  (h3 : ∠ AFG = ∠ AGF)
  (h4 : (∠ AFG + ∠ BFD) = 180)
  (exterior_angle : ∠ BFD = 180 - ∠ DFB)
  : ∠ B + ∠ D = 75 := 
by
  sorry

end sum_of_angles_B_and_D_l107_107860


namespace Cara_age_is_40_l107_107300

def cara_older_than_mom : ℕ := 20
def mom_older_than_grandmother : ℕ := 15
def grandmother_age : ℕ := 75

theorem Cara_age_is_40 :
  let mom_age := grandmother_age - mom_older_than_grandmother in
  let cara_age := mom_age - cara_older_than_mom in
  cara_age = 40 :=
by
  let mom_age := grandmother_age - mom_older_than_grandmother
  let cara_age := mom_age - cara_older_than_mom
  sorry

end Cara_age_is_40_l107_107300


namespace cute_angle_of_isosceles_cute_triangle_l107_107867

theorem cute_angle_of_isosceles_cute_triangle (A B C : ℝ) 
    (h1 : B = 2 * C)
    (h2 : A + B + C = 180)
    (h3 : A = B ∨ A = C) :
    A = 45 ∨ A = 72 :=
sorry

end cute_angle_of_isosceles_cute_triangle_l107_107867


namespace relationship_between_abc_l107_107836

theorem relationship_between_abc (f : ℝ → ℝ) (t : ℝ) (h1 : ∀ x, f (x) = real.log (abs (x - t)) / real.log 3)
  (even_f : ∀ x, f (-x) = f (x)) :
  let a := f (real.log (4) / real.log (0.3))
      b := f (pi ^ (3 / 2))
      c := f (2 - t)
  in a < c ∧ c < b :=
by
  -- Proof omitted
  sorry

end relationship_between_abc_l107_107836


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107795

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107795


namespace max_value_x_plus_half_inv_y_l107_107460

theorem max_value_x_plus_half_inv_y 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : (2 * x * y - 1)^2 = (5 * y + 2) * (y - 2)) :
  x + 1 / (2 * y) ≤ (3 / 2) * real.sqrt 2 - 1 :=
sorry

end max_value_x_plus_half_inv_y_l107_107460


namespace gcd_840_1764_l107_107111

def a : ℕ := 840
def b : ℕ := 1764

theorem gcd_840_1764 : Nat.gcd a b = 84 := by
  -- Proof omitted
  sorry

end gcd_840_1764_l107_107111


namespace area_of_shaded_region_l107_107265

/-- A regular octagon has a side length of 3. Semicircles with diameters along each side of the octagon
are drawn inside it. Calculate the area of the region inside the octagon but outside all of the semicircles. -/
theorem area_of_shaded_region :
  let s := 3 in
  let n := 8 in
  let octagon_area := (n * s^2) / (4 * Real.tan (Real.pi / n)) in
  let radius := s / 2 in
  let semicircle_area := (1 / 2) * Real.pi * radius^2 in
  let total_semicircle_area := n * semicircle_area in
  let shaded_area := octagon_area - total_semicircle_area in
  shaded_area ≈ 43.48 - 9 * Real.pi :=
by
  sorry

end area_of_shaded_region_l107_107265


namespace center_of_circle_N_is_correct_l107_107572

-- Define the points (A and B) as the endpoints of the diameter
def A : ℝ × ℝ := (3, -2)
def B : ℝ × ℝ := (9, 8)

-- Define the midpoint formula for two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the expected center
def N_center : ℝ × ℝ := (6, 3)

-- The main statement: The midpoint of A and B should equal N_center
theorem center_of_circle_N_is_correct : 
  midpoint A B = N_center :=
  by
    sorry

end center_of_circle_N_is_correct_l107_107572


namespace proof_problem_l107_107845

variables (a b : ℝ)
variable (h : a ≠ b)
variable (h1 : a * Real.exp a = b * Real.exp b)
variable (p : Prop := Real.log a + a = Real.log b + b)
variable (q : Prop := (a + 1) * (b + 1) < 0)

theorem proof_problem : p ∨ q :=
sorry

end proof_problem_l107_107845


namespace A_completes_work_in_40_days_l107_107651

theorem A_completes_work_in_40_days 
  (B_work_rate : ℚ)
  (A_work_rate : ℚ)
  (B_days_to_complete : ℕ)
  (A_days_worked : ℕ)
  (B_days_worked : ℕ)
  (total_work : ℚ) :
  B_work_rate = 1 / 60 →
  A_work_rate = 1 / A_days_to_complete →
  B_days_to_complete = 60 →
  A_days_worked = 10 →
  B_days_worked = 45 →
  total_work = 1 →
  A_days_to_complete = 40 :=
begin
  assume h1 : B_work_rate = 1 / 60,
  assume h2 : A_work_rate = 1 / A_days_worked,
  assume h3 : B_days_to_complete = 60,
  assume h4 : A_days_worked = 10,
  assume h5 : B_days_worked = 45,
  assume h6 : total_work = 1,
  sorry
end

end A_completes_work_in_40_days_l107_107651


namespace trapezoid_diagonal_intersection_l107_107642

theorem trapezoid_diagonal_intersection (AB CD AC EC : ℝ)
  (h1 : AB = 3 * CD) 
  (h2 : AC = 15) 
  (h3 : ∃ D, true) 
  (h4 : ∃ E, true)
  (h5 : ∃ (h : ℝ), h = 5) 
  : EC = 15 / 4 := 
by
  sorry

end trapezoid_diagonal_intersection_l107_107642


namespace set_D_cannot_form_triangle_l107_107673

theorem set_D_cannot_form_triangle : ¬ (∃ a b c : ℝ, a = 2 ∧ b = 4 ∧ c = 6 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a)) :=
by {
  sorry
}

end set_D_cannot_form_triangle_l107_107673


namespace smallest_prime_12_less_than_perfect_square_l107_107208

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107208


namespace smallest_prime_less_than_square_l107_107148

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107148


namespace isosceles_right_triangle_perimeter_to_area_l107_107115

noncomputable def isosceles_right_triangle_area (p : ℝ) : ℝ :=
  let x := (3 * p) / (3 + 2 * Real.sqrt 2)
  1 / 2 * (x ^ 2)

theorem isosceles_right_triangle_perimeter_to_area (p : ℝ) :
  let x := (3 * p) / (3 + 2 * Real.sqrt 2) in
  3 * x + 2 * x * Real.sqrt 2 = 3 * p →
  isosceles_right_triangle_area p = (153 - 108 * Real.sqrt 2) / 2 * p ^ 2 :=
by
  sorry

end isosceles_right_triangle_perimeter_to_area_l107_107115


namespace oldest_child_age_l107_107565

-- Definitions based on the conditions from part a).
def average_age (ages : Fin 7 → ℤ) : ℤ :=
  (∑ i, ages i) / 7

def arithmetic_sequence (ages : Fin 7 → ℤ) : Prop :=
  ∀ i : Fin 6, ages ⟨i.1 + 1, by linarith⟩ - ages i = 3

-- Statement of the problem
theorem oldest_child_age :
  ∃ (ages : Fin 7 → ℤ), average_age ages = 8 
  ∧ arithmetic_sequence ages 
  ∧ (ages 6 = 17) :=
sorry

end oldest_child_age_l107_107565


namespace existence_of_specified_pairs_l107_107290

-- Definitions for the problem
variables {Boy Girl : Type}
variables (Danced : Boy → Girl → Prop)

-- Hypotheses based on the problem conditions
hypothesis no_boy_danced_with_all_girls :
  ∀ (b : Boy), ∃ (g : Girl), ¬ Danced b g
hypothesis each_girl_danced_with_at_least_one_boy :
  ∀ (g : Girl), ∃ (b : Boy), Danced b g

-- Statement of the math proof problem
theorem existence_of_specified_pairs :
  ∃ (g g' : Boy) (f f' : Girl), Danced g f ∧ ¬ Danced g f' ∧ Danced g' f' ∧ ¬ Danced g' f :=
sorry

end existence_of_specified_pairs_l107_107290


namespace lunch_break_duration_l107_107076

variable (p h : ℝ) (L : ℝ) -- Paula's rate, assistants' combined rate, and lunch break duration

-- Conditions
def monday_work := (9 - L) * (p + h) = 0.4
def tuesday_work := (7 - L) * h = 0.3
def wednesday_work := (12 - L) * p = 0.3

-- Statement to prove
theorem lunch_break_duration :
  monday_work p h L →
  tuesday_work p h L →
  wednesday_work p h L →
  L = 0.5 :=
sorry

end lunch_break_duration_l107_107076


namespace sphere_surface_area_l107_107377

variables {a : ℝ}

theorem sphere_surface_area (a_pos : 0 < a) :
  let l := 2 * a,
      w := a,
      h := a,
      d := Real.sqrt (l^2 + w^2 + h^2)  -- diameter of the sphere
  in d = 2 * Real.sqrt (3 / 2 * a^2) →
     4 * Real.pi * (Real.sqrt (3 / 2 * a^2))^2 = 6 * Real.pi * a^2 :=
by
  sorry

end sphere_surface_area_l107_107377


namespace lynne_total_spending_l107_107525

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end lynne_total_spending_l107_107525


namespace relationship_among_abc_l107_107513

noncomputable def a : ℝ := log 4 / log 3 + log (1 / 2) / log 3
noncomputable def b : ℝ := log 5 / log (exp 1) - log (5 / 2) / log (exp 1)
noncomputable def c : ℝ := 5 ^ (1 / 3 * (log 2 / log 5))

theorem relationship_among_abc : a < b ∧ b < c := by
  sorry

end relationship_among_abc_l107_107513


namespace sum_of_coefficients_is_zero_l107_107409

noncomputable def expansion : Polynomial ℚ := (Polynomial.X^2 + Polynomial.X + 1) * (2*Polynomial.X - 2)^5

theorem sum_of_coefficients_is_zero :
  (expansion.coeff 0) + (expansion.coeff 1) + (expansion.coeff 2) + (expansion.coeff 3) + 
  (expansion.coeff 4) + (expansion.coeff 5) + (expansion.coeff 6) + (expansion.coeff 7) = 0 :=
by
  sorry

end sum_of_coefficients_is_zero_l107_107409


namespace geometric_series_first_term_l107_107678

theorem geometric_series_first_term (a : ℝ) (r : ℝ) (s : ℝ) 
  (h1 : r = -1/3) (h2 : s = 12) (h3 : s = a / (1 - r)) : a = 16 :=
by
  -- Placeholder for the proof
  sorry

end geometric_series_first_term_l107_107678


namespace tangent_line_is_x_minus_y_eq_zero_l107_107108

theorem tangent_line_is_x_minus_y_eq_zero : 
  ∀ (f : ℝ → ℝ) (x y : ℝ), 
  f x = x^3 - 2 * x → 
  (x, y) = (1, 1) → 
  (∃ (m : ℝ), m = 3 * (1:ℝ)^2 - 2 ∧ (y - 1) = m * (x - 1)) → 
  x - y = 0 :=
by
  intros f x y h_func h_point h_tangent
  sorry

end tangent_line_is_x_minus_y_eq_zero_l107_107108


namespace smallest_prime_12_less_perfect_square_l107_107140

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l107_107140


namespace find_f_of_one_third_l107_107865

-- Define g function according to given condition
def g (x : ℝ) : ℝ := 1 - x^2

-- Define f function according to given condition, valid for x ≠ 0
noncomputable def f (x : ℝ) : ℝ := (1 - x) / x

-- State the theorem we need to prove
theorem find_f_of_one_third : f (1 / 3) = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end find_f_of_one_third_l107_107865


namespace max_value_S_correct_l107_107053

noncomputable def max_value_S (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  if h : n > 2 ∧ (∀ i, x i ≥ 0) ∧ (∑ i in finset.range (n + 1), x i = n) ∧
       (∑ i in finset.range (n + 1), i * x i = 2 * n - 2) then
    n ^ 2 - 2
  else
    0

theorem max_value_S_correct (n : ℕ) (x : ℕ → ℝ) :
  n > 2 ∧ (∀ i, x i ≥ 0) ∧
  (∑ i in finset.range (n + 1), x i = n) ∧
  (∑ i in finset.range (n + 1), i * x i = 2 * n - 2) →
  max_value_S n x = n ^ 2 - 2 :=
sorry

end max_value_S_correct_l107_107053


namespace sin_A_value_l107_107918

theorem sin_A_value (A B C : ℝ)
  (h1 : sin^2 A + sin^2 B + 2 * cos A * cos B * sin C = 9 / 10)
  (h2 : sin^2 B + sin^2 C + 2 * cos B * cos C * sin A = 11 / 12)
  (h3 : sin^2 C + sin^2 A + 2 * cos C * cos A * sin B = 1 / 2)
  (h4 : 0 < A ∧ A < π / 2) :
  sin A = sqrt 2 / sqrt 5 := 
sorry

end sin_A_value_l107_107918


namespace problem_solution_l107_107050

theorem problem_solution
  {a b c d : ℝ}
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2011)
  (h3 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2011) :
  (c * d)^2012 - (a * b)^2012 = 2011 :=
by
  sorry

end problem_solution_l107_107050


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107169

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107169


namespace angle_bisectors_and_median_inequality_l107_107080

open Real

variables (A B C : Point)
variables (a b c : ℝ) -- sides of the triangle
variables (p : ℝ) -- semi-perimeter of the triangle
variables (la lb mc : ℝ) -- angle bisectors and median lengths

-- Assume the given conditions
axiom angle_bisector_la (A B C : Point) : ℝ -- lengths of the angle bisector of ∠BAC
axiom angle_bisector_lb (A B C : Point) : ℝ -- lengths of the angle bisector of ∠ABC
axiom median_mc (A B C : Point) : ℝ -- length of the median from vertex C
axiom semi_perimeter (a b c : ℝ) : ℝ -- semi-perimeter of the triangle

-- The statement of the theorem
theorem angle_bisectors_and_median_inequality (la lb mc p : ℝ) :
  la + lb + mc ≤ sqrt 3 * p :=
sorry

end angle_bisectors_and_median_inequality_l107_107080


namespace cos_beta_acos_l107_107818

theorem cos_beta_acos {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_cos_α : Real.cos α = 1 / 7) (h_cos_sum : Real.cos (α + β) = -11 / 14) :
  Real.cos β = 1 / 2 := by
  sorry

end cos_beta_acos_l107_107818


namespace angle_ADE_eq_120_l107_107466

theorem angle_ADE_eq_120 
(AD BE : Line) (A B C D E : Point)
(h1 : AB = AC)
(h2 : AB ∥ ED)
(h3 : ∠ ABC = 30)
(h4 : ∠ ADE = S) :
S = 120 :=
sorry

end angle_ADE_eq_120_l107_107466


namespace base_of_log_eq_self_l107_107614

noncomputable def log_base := sorry

theorem base_of_log_eq_self (a x : ℝ) (h : log x a = a) : x = a^(1/a) :=
by sorry

end base_of_log_eq_self_l107_107614


namespace weighted_average_weights_l107_107343

namespace FishProblem

def lake_tahoe_trout_weight : ℕ := 20 * 2
def lake_tahoe_bass_weight : ℕ := 12 * 1.5.to_nat
def lake_tahoe_carp_weight : ℕ := 8 * 3
def lake_michigan_salmon_weight : ℕ := 13 * 4
def lake_michigan_trout_weight : ℕ := 10 * 1.5.to_nat
def lake_michigan_walleye_weight : ℕ := 9 * 2
def lake_superior_perch_weight : ℕ := 17 * 2.5.to_nat
def lake_superior_northern_pike_weight : ℕ := 15 * 4
def lake_superior_whitefish_weight : ℕ := 8 * 3.5.to_nat

def lake_tahoe_total_weight : ℕ := lake_tahoe_trout_weight + lake_tahoe_bass_weight + lake_tahoe_carp_weight
def lake_michigan_total_weight : ℕ := lake_michigan_salmon_weight + lake_michigan_trout_weight + lake_michigan_walleye_weight
def lake_superior_total_weight : ℕ := lake_superior_perch_weight + lake_superior_northern_pike_weight + lake_superior_whitefish_weight

def lake_tahoe_total_fish : ℕ := 20 + 12 + 8
def lake_michigan_total_fish : ℕ := 13 + 10 + 9
def lake_superior_total_fish : ℕ := 17 + 15 + 8

def lake_tahoe_average_weight : ℚ := lake_tahoe_total_weight / lake_tahoe_total_fish
def lake_michigan_average_weight : ℚ := lake_michigan_total_weight / lake_michigan_total_fish
def lake_superior_average_weight : ℚ := lake_superior_total_weight / lake_superior_total_fish

theorem weighted_average_weights :
    lake_tahoe_average_weight = 2.05 ∧
    lake_michigan_average_weight = 2.65625 ∧
    lake_superior_average_weight = 3.2625 :=
begin
  -- leaving the proof as an exercise.
  sorry
end

end FishProblem

end weighted_average_weights_l107_107343


namespace find_box_height_l107_107621

noncomputable def box_height
  (base_length base_width : ℝ)
  (total_volume cost_per_box : ℝ)
  (minimum_spent : ℝ)
  (base_height := 15) : ℝ :=
  total_volume / (minimum_spent / cost_per_box * base_length * base_width)

theorem find_box_height
  (base_length base_width : ℝ)
  (total_volume cost_per_box : ℝ)
  (minimum_spent : ℝ)
  (base_height := 15)
  (h_base_length : base_length = 20)
  (h_base_width : base_width = 20)
  (h_total_volume : total_volume = 3.06e6)
  (h_cost_per_box : cost_per_box = 1.2)
  (h_minimum_spent : minimum_spent = 612) :
  box_height base_length base_width total_volume cost_per_box minimum_spent = base_height :=
by
  rw [h_base_length, h_base_width, h_total_volume, h_cost_per_box, h_minimum_spent]
  have h_boxes := minimum_spent / cost_per_box
  have h_volume_per_box := 20 * 20
  have h_height := (3.06e6 : ℝ) / (510 * h_volume_per_box)
  rw h_height
  norm_num
  sorry

end find_box_height_l107_107621


namespace max_min_diff_c_l107_107500

variable (a b c : ℝ)

theorem max_min_diff_c (h1 : a + b + c = 6) (h2 : a^2 + b^2 + c^2 = 18) : 
  (4 - 0) = 4 :=
by
  sorry

end max_min_diff_c_l107_107500


namespace choose_programs_l107_107667

open Function
open Finset

def courses : Finset String := {"English", "Algebra", "Geometry", "History", "Art", "Latin", "Biology"}

def is_math : String → Bool
| "Algebra"  := true
| "Geometry" := true
| _ := false

def is_science : String → Bool
| "Biology" := true
| _ := false

theorem choose_programs :
  (choose 6 4) -
  (choose 4 4) -
  (choose 5 4) + 1 = 10 :=
by sorry

end choose_programs_l107_107667


namespace correct_increasing_function_below_line_l107_107620

theorem correct_increasing_function_below_line (x : ℝ) (h : 1 < x) :
  (∀ f ∈ {λ x, x^(1/2), λ x, x^2, λ x, x^3, λ x, x^(-1)}, ∀ x, 1 < x → f x < x) →
  (∀ f ∈ {λ x, x^(1/2), λ x, x^2, λ x, x^3, λ x, x^(-1)}, ∀ a b, a < b → f a < f b) →
  (y = x^(1/2)) := 
sorry

end correct_increasing_function_below_line_l107_107620


namespace root_in_interval_l107_107983

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 2 / x

theorem root_in_interval : ∃ x ∈ Ioo (1 : ℝ) 2, f x = 0 :=
by
  -- Using intermediate value theorem or similar result
  sorry

end root_in_interval_l107_107983


namespace evaluate_x_squared_when_y_equals_4_l107_107100

open Real

theorem evaluate_x_squared_when_y_equals_4 :
  ∀ (x y k : ℝ), (x = 5) → (y = 2) → (x^2 * y^4 = k) → (y = 4) → (x^2 = 25 / 16) :=
by
  intros x y k hx hy h1 hy' 
  -- Additional constraints and steps for the proof
  sorry

end evaluate_x_squared_when_y_equals_4_l107_107100


namespace number_of_friends_l107_107337

-- Conditions/Definitions
def total_cost : ℤ := 13500
def cost_per_person : ℤ := 900

-- Prove that Dawson is going with 14 friends.
theorem number_of_friends (h1 : total_cost = 13500) (h2 : cost_per_person = 900) :
  (total_cost / cost_per_person) - 1 = 14 :=
by
  sorry

end number_of_friends_l107_107337


namespace T_n_lt_3_l107_107817

noncomputable def a_n (n : ℕ) : ℕ := 2 * n

def S_n (n : ℕ) : ℕ := n * n + n

def b_n (n : ℕ) : ℝ := (S_n n : ℝ) / (n * (2 ^ n))

def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b_n (i + 1)

theorem T_n_lt_3 (n : ℕ) : T_n n < 3 := by sorry

end T_n_lt_3_l107_107817


namespace seq_is_arithmetic_upper_bound_on_b_l107_107842

/-
First part: Prove that the sequence \{a\_n\} satisfies \(a_n = 2n\)
Conditions:
- a_1 = 2
- ∀ n ∈ ℕ, (n + 1) * a_{n+1} - (n + 2) * a_n = 2 
-/

theorem seq_is_arithmetic (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, (n + 1) * a (n + 1) - (n + 2) * a n = 2) :
  a n = 2 * n :=
sorry

/-
Second part: Prove that \{b\_n\} is bounded above by \(\frac{40}{27}\)
Conditions:
- ∀ n ∈ ℕ, S_n = a 0 + a 1 + ... + a (n - 1)
- b_n = n * (-\sqrt{6}/3)^{S_n / n}
- ∀ n ∈ ℕ, b_n ≤ M
-/

theorem upper_bound_on_b (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) (M : ℝ) 
  (h1 : ∀ n : ℕ, a n = 2 * n) 
  (h2 : ∀ n : ℕ, S n = ∑ i in range n, a i) 
  (h3 : ∀ n : ℕ, b n = n * (- (sqrt 6) / 3) ^ (S n / n)) 
  (h4 : ∀ n : ℕ, b n ≤ M) :
  M = 40/27 :=
sorry

end seq_is_arithmetic_upper_bound_on_b_l107_107842


namespace f_of_3_l107_107833

def f (x : ℚ) : ℚ := (x + 3) / (x - 6)

theorem f_of_3 : f 3 = -2 := by
  sorry

end f_of_3_l107_107833


namespace sum_of_odds_l107_107618

theorem sum_of_odds : 
  let d := 2
  let a := 15
  let l := 65
  let n := 26
  let S := 13 * 80
  in S = 1040 :=
by
  let d := 2
  let a := 15
  let l := 65
  let n := 26
  let S := 13 * 80
  show S = 1040 from sorry

end sum_of_odds_l107_107618


namespace find_length_BX_l107_107238

noncomputable def length_BX (A B C X Y Z E D F O1 O2 O3 : Type) [AddGroup A] [AddGroup B] [AddGroup C]
  (AB BC AC : ℝ)
  (h_AB : AB = 15)
  (h_BC : BC = 20)
  (h_AC : AC = 13)
  (h_angle_A : ∠ BAC = ∠ BXC) 
  (h_angle_B : ∠ ABC = ∠ BYC)
  (h_angle_C : ∠ ACB = ∠ AZB) 
  (h_symmetry : true) : 
  ℝ :=
(BX AC).symm

theorem find_length_BX (A B C X Y Z E D F O1 O2 O3 : Type) [AddGroup A] [AddGroup B] [AddGroup C]
  (AB BC AC : ℝ)
  (h_AB : AB = 15)
  (h_BC : BC = 20)
  (h_AC : AC = 13)
  (h_angle_A : ∠ BAC = ∠ BXC) 
  (h_angle_B : ∠ ABC = ∠ BYC)
  (h_angle_C : ∠ ACB = ∠ AZB) 
  (h_symmetry : true) : 
  length_BX A B C X Y Z E D F O1 O2 O3 AB BC AC h_AB h_BC h_AC h_angle_A h_angle_B h_angle_C h_symmetry = 13 / 2 :=
sorry

end find_length_BX_l107_107238


namespace miles_to_friends_house_l107_107031

-- Define the conditions as constants
def miles_per_gallon : ℕ := 19
def gallons : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_burger_restaurant : ℕ := 2
def miles_home : ℕ := 11

-- Define the total miles driven
def total_miles_driven (miles_to_friend : ℕ) :=
  miles_to_school + miles_to_softball_park + miles_to_burger_restaurant + miles_to_friend + miles_home

-- Define the total miles possible with given gallons of gas
def total_miles_possible : ℕ :=
  miles_per_gallon * gallons

-- Prove that the miles driven to the friend's house is 4
theorem miles_to_friends_house : 
  ∃ miles_to_friend, total_miles_driven miles_to_friend = total_miles_possible ∧ miles_to_friend = 4 :=
by
  sorry

end miles_to_friends_house_l107_107031


namespace smallest_prime_less_than_square_l107_107149

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107149


namespace trigonometric_expression_l107_107693

noncomputable def angle_deg := Float
noncomputable def cos_deg (θ : angle_deg) : Float := Real.cos (θ * Float.pi / 180.0)
noncomputable def sin_deg (θ : angle_deg) : Float := Real.sin (θ * Float.pi / 180.0)

theorem trigonometric_expression :
  (1 / cos_deg 70.0) - (Float.sqrt 2 / sin_deg 70.0) = 4 :=
by
  sorry

end trigonometric_expression_l107_107693


namespace remainder_of_sum_is_zero_l107_107328

-- Define the properties of m and n according to the conditions of the problem
def m : ℕ := 2 * 1004 ^ 2
def n : ℕ := 2007 * 1003

-- State the theorem that proves the remainder of (m + n) divided by 1004 is 0
theorem remainder_of_sum_is_zero : (m + n) % 1004 = 0 := by
  sorry

end remainder_of_sum_is_zero_l107_107328


namespace order_of_a_b_c_l107_107814

noncomputable def a := 2^(1/2)
noncomputable def b := Real.log 3 / Real.log π
noncomputable def c := Real.log 0.9 / Real.log 2

theorem order_of_a_b_c : a > b ∧ b > c := by
  have ha : a = 2^(1/2) := rfl
  have hb : b = Real.log 3 / Real.log π := rfl
  have hc : c = Real.log 0.9 / Real.log 2 := rfl
  sorry

end order_of_a_b_c_l107_107814


namespace complex_point_in_third_quadrant_l107_107895

def complex_conj_point_quadrant : ℂ :=
  conj ( (i / (1 + i)) + (1 + 2 * i) ^ 2 )

theorem complex_point_in_third_quadrant :
  let z := complex_conj_point_quadrant
  - (5 / 2) - (9 / 2) * i = z →
  z.im < 0 ∧ z.re < 0 :=
by
  intros h
  sorry

end complex_point_in_third_quadrant_l107_107895


namespace exists_set_with_inf_symmetry_axes_but_not_centrally_symmetric_l107_107483

noncomputable def P (x : Real) := (Real.cos x, Real.sin x)

def K : Set (Real × Real) := {p | ∃ x : Int, p = P x}

def has_inf_symmetry_axes (s : Set (Real × Real)) : Prop :=
  ∃ (f : ℤ → Line), Function.Injective f ∧
  ∀ (i : ℤ), ∃ (l : Line), l ∈ f i

def not_centrally_symmetric (s : Set (Real × Real)) : Prop :=
  ∀ (c : Real × Real), ∃ (p ∈ s), ¬ ((2 * c - p) ∈ s)

theorem exists_set_with_inf_symmetry_axes_but_not_centrally_symmetric :
  ∃ (s : Set (Real × Real)), has_inf_symmetry_axes s ∧
  ¬(not_centrally_symmetric s) ∧
  ∃ (M : Real), ∀ (p ∈ s), ∥p∥ ≤ M :=
sorry

end exists_set_with_inf_symmetry_axes_but_not_centrally_symmetric_l107_107483


namespace inequality_proof_l107_107449

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the even function f

theorem inequality_proof
  (h1 : ∀ x, f (-x) = f x) 
  (h2 : ∀ x y ∈ [-1, 0], x < y → f x < f y) 
  (ha : 0 < α ∧ α < π / 2) 
  (hb : 0 < β ∧ β < π / 2) 
  (hab : α + β > π / 2) :
  f (Real.cos α) > f (Real.sin β) :=
by
  have hβ : β > π / 2 - α := by linarith
  have hsinβ_cosα : Real.sin β > Real.cos α := sorry -- using trigonometric identities and the fact β > π / 2 - α
  have h_decreasing : ∀ x y ∈ [0, 1], x < y → f x > f y := 
    by
      intros x y hx hy hxy
      have h1x_neg : -y ∈ [-1, 0] := sorry -- hence using the even property
      have h2y_neg : -x ∈ [-1, 0] := sorry -- hence using the even property and increasing condition
      exact h2 (-y) (-x) h1x_neg h2y_neg (neg_lt_neg_iff.mpr hxy)
  exact h_decreasing (Real.cos α) (Real.sin β)
    sorry -- concluding the final proof for f (Real.cos α) > f (Real.sin β)

end inequality_proof_l107_107449


namespace fifi_pink_hangers_l107_107878

theorem fifi_pink_hangers :
  ∀ (g b y p : ℕ), 
  g = 4 →
  b = g - 1 →
  y = b - 1 →
  16 = g + b + y + p →
  p = 7 :=
by
  intros
  sorry

end fifi_pink_hangers_l107_107878


namespace trains_clear_time_l107_107134

-- Define the lengths of the trains
def length_train1 : ℝ := 131
def length_train2 : ℝ := 165

-- Define the speeds of the trains in km/h
def speed_train1_kmph : ℝ := 80
def speed_train2_kmph : ℝ := 65

-- Define the conversion factor from km/h to m/s
def kmph_to_mps (kmph : ℝ) : ℝ := (kmph * 1000) / 3600

-- Calculate the total length of the two trains combined
def total_length : ℝ := length_train1 + length_train2

-- Calculate the relative speed in km/h
def relative_speed_kmph : ℝ := speed_train1_kmph + speed_train2_kmph

-- Calculate the relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps relative_speed_kmph

-- Statement of the problem in Lean
theorem trains_clear_time : 
  total_length / relative_speed_mps ≈ 7.35 :=
sorry

end trains_clear_time_l107_107134


namespace probability_line_passes_squares_l107_107928

-- Define the rectangle R
def R := set.Icc (0:ℝ) 2 × set.Icc (0:ℝ) 1

-- Define the conditions
def P_in_R (P : ℝ × ℝ) : Prop := P.1 ∈ set.Icc 0 2 ∧ P.2 ∈ set.Icc 0 1

def line_through_point_with_slope (P : ℝ × ℝ) (m : ℝ) : set (ℝ × ℝ) :=
  {Q | Q.2 = m * (Q.1 - P.1) + P.2}

-- Define the intersection condition
def intersects_both_squares (L : set (ℝ × ℝ)): Prop :=
  ∃ (x : ℝ), (x ∈ set.Icc 0 2) ∧ (1 ∈ set.Icc 0 1) ∧ ((1, (1/2)*1 + L) ∈ L)

-- Main statement
theorem probability_line_passes_squares :
  let m := 1/2 in
  ∀ (P : ℝ × ℝ), P ∈ (set.Icc 0 2) × (set.Icc 0 1) →
  ∃ (L : set (ℝ × ℝ)), L = line_through_point_with_slope P m →
  intersects_both_squares L →
  ∃ (prob : ℝ), prob = 3 / 4.
sorry

end probability_line_passes_squares_l107_107928


namespace smallest_prime_12_less_than_perfect_square_l107_107202

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107202


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107174

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107174


namespace unique_k_for_fibonacci_roots_l107_107684

def is_fibonacci (n : ℕ) : Prop := ∃ (a b : ℕ), (a = 0 ∨ a = 1 ∨ a > 1 ∧ (∀ k < a, fib k = fib (k+1) + fib (k+2))) ∧ 
                                                (b = 0 ∨ b = 1 ∨ b > 1 ∧ (∀ k < b, fib k = fib (k+1) + fib (k+2))) ∧ 
                                                fib a = n ∧ fib b = n

theorem unique_k_for_fibonacci_roots (k : ℕ) : 
  (∃ a b : ℕ, is_fibonacci a ∧ is_fibonacci b ∧ a + b = 20 ∧ a * b = k) ↔ k = 104 := 
  sorry

end unique_k_for_fibonacci_roots_l107_107684


namespace find_x_l107_107984

theorem find_x {x : ℝ} :
  (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 → x = 6 :=
by
  intro h
  -- Solution steps would go here, but they are omitted.
  sorry

end find_x_l107_107984


namespace q_implies_k_range_p_or_q_implies_k_range_l107_107810

variable (k : ℝ)

-- Proposition p
def prop_p := k ^ 2 - 8 * k - 20 ≤ 0

-- Proposition q
def prop_q := 1 < k ∧ k < 4

-- Question I: If q is true, the range of k is 1 < k < 4
theorem q_implies_k_range : prop_q → (1 < k ∧ k < 4) :=
by sorry

-- Question II: If p ∨ q is true and p ∧ q is false, the range of k is -2 ≤ k ≤ 1 or 4 ≤ k ≤ 10
theorem p_or_q_implies_k_range : (prop_p ∨ prop_q) ∧ ¬(prop_p ∧ prop_q) → (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) :=
by sorry

end q_implies_k_range_p_or_q_implies_k_range_l107_107810


namespace sum_of_numbers_with_base3_reverse_base8_l107_107358

theorem sum_of_numbers_with_base3_reverse_base8 :
  (∑ n in {n : ℕ | (∀ d, n / (3^d) % 3 == (n / (8^d) % 8)}, n) = 23) :=
sorry

end sum_of_numbers_with_base3_reverse_base8_l107_107358


namespace olivia_worked_hours_on_wednesday_l107_107532

-- Define the conditions
def hourly_rate := 9
def hours_monday := 4
def hours_friday := 6
def total_earnings := 117
def earnings_monday := hours_monday * hourly_rate
def earnings_friday := hours_friday * hourly_rate
def earnings_wednesday := total_earnings - (earnings_monday + earnings_friday)

-- Define the number of hours worked on Wednesday
def hours_wednesday := earnings_wednesday / hourly_rate

-- The theorem to prove
theorem olivia_worked_hours_on_wednesday : hours_wednesday = 3 :=
by
  -- Skip the proof
  sorry

end olivia_worked_hours_on_wednesday_l107_107532


namespace BF_plus_p_q_r_l107_107557

def Square (A B C D : Type) :=
  (𝒜 SQ A B C D) ∧ (AB = 1000) ∧ (center O : Math)

def Meter (a : Type) : Type :=
  exists (AB == 1000) ∧ (E lies at AB) ∧ (F lies AB) (AE < BF) ∧ (E lies between A F)

def Angle (h : Type) : Type :=
  exists (m ∠ EOF = 60°) ∧ (EF = 500)

def Conditions (h : Type) : Type :=
  (Square A B C D) ∧ (AB = 1000) ∧ (coordinates exist at AB EF) (AE < BF) ∧ (arrangement EFT triangle)

theorem BF_plus_p_q_r 
  {A B C D O E F : Type}
  (h1 : Square A B C D) 
  (h2 : Meter AB) 
  (h3 : Angle EOF E F) 
  (h4 : EF = 500) 
  (h5 : ∀ p q r : ℕ, BF = p + q * √r)
  (h6 : ∀ r ≠ divisible by any square prime):
  ∃ p q r : ℕ, p + q + r = 378 :=
by
sorry

end BF_plus_p_q_r_l107_107557


namespace tiles_visited_by_bug_l107_107264

theorem tiles_visited_by_bug (width length : ℕ)
  (h_width : width = 12) (h_length : length = 19) : 
  let gcd_val := Nat.gcd width length in
  (let num_tiles_crossed := width + length - gcd_val in 
    num_tiles_crossed) = 30 :=
by
  -- Assign the values to width and length 
  have h_width_length : width = 12 ∧ length = 19 := ⟨h_width, h_length⟩
  -- Calculate gcd of width and length
  let gcd_val := Nat.gcd width length
  have h_gcd : gcd_val = 1 := Nat.gcd_eq_right (nat.prime.dvd_gcd_iff ⟨2, nat.prime_two⟩ h_width_length.1 h_width_length.2) (nat.prime.dvd_of_dvd_sub zero_le_one (by linarith))
  -- Calculate the number of tiles crossed
  let num_tiles_crossed := width + length - gcd_val
  have h_num_tiles_crossed : num_tiles_crossed = 30 := by 
    simp [h_gcd]
    rw [←h_width, ←h_length]
    norm_num
  exact h_num_tiles_crossed

end tiles_visited_by_bug_l107_107264


namespace correct_combined_area_regions_II_III_l107_107543

noncomputable def combined_area_regions_II_III
  (ABCD_is_square : ∀ (A B C D : ℝ), is_square ABCD)
  (circle_center_D : ∀ (D : ℝ), has_arc_AEC D)
  (circle_center_B : ∀ (B : ℝ), has_arc_AFC B)
  (AB_eq_4 : AB = 4)
  (regions_congruent : football_shaped_regions_congruent II III) : ℝ :=
  9.1

theorem correct_combined_area_regions_II_III
  (ABCD_is_square : ∀ (A B C D : ℝ), is_square ABCD)
  (circle_center_D : ∀ (D : ℝ), has_arc_AEC D)
  (circle_center_B : ∀ (B : ℝ), has_arc_AFC B)
  (AB_eq_4 : AB = 4)
  (regions_congruent : football_shaped_regions_congruent II III) :
  combined_area_regions_II_III ABCD_is_square circle_center_D circle_center_B AB_eq_4 regions_congruent = 9.1 :=
sorry

end correct_combined_area_regions_II_III_l107_107543


namespace div_by_1897_l107_107081

theorem div_by_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end div_by_1897_l107_107081


namespace intervals_of_monotonicity_a_eq_2_range_of_a_if_extreme_point_in_2_3_l107_107834

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * a * x^2 + 3 * x + 1

-- (1) Intervals of monotonicity for f(x) when a = 2
theorem intervals_of_monotonicity_a_eq_2 :
  (∀ x ∈ set.Iio 1, f' x 2 > 0) ∧ 
  (∀ x ∈ set.Ioo 1 3, f' x 2 < 0) ∧ 
  (∀ x ∈ set.Ioi 3, f' x 2 > 0) := sorry

-- (2) Range of values for a when f(x) has an extreme point in (2, 3)
theorem range_of_a_if_extreme_point_in_2_3 (h : ∃ x ∈ set.Ioo (2 : ℝ) 3, deriv (λ x, f x a) x = 0) :
  2 / real.sqrt 5 < a ∧ a < 3 / real.sqrt 5 := sorry

end intervals_of_monotonicity_a_eq_2_range_of_a_if_extreme_point_in_2_3_l107_107834


namespace surface_area_of_sphere_l107_107000

theorem surface_area_of_sphere (a : Real) (h : a = 2 * Real.sqrt 3) : 
  (4 * Real.pi * ((Real.sqrt 3 * a / 2) ^ 2)) = 36 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l107_107000


namespace second_order_arithmetic_sequence_20th_term_l107_107112

theorem second_order_arithmetic_sequence_20th_term :
  (∀ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 4 ∧
    a 3 = 9 ∧
    a 4 = 16 ∧
    (∀ n, 2 ≤ n → a n - a (n - 1) = 2 * n - 1) →
    a 20 = 400) :=
by 
  sorry

end second_order_arithmetic_sequence_20th_term_l107_107112


namespace city_path_between_squares_l107_107881

universe u

structure City (α : Type u) :=
(blue_square : α)
(green_squares : list α)
(streets : α → α → Prop)
(connected_to_blue : ∀ g ∈ green_squares, streets g blue_square ∨ streets blue_square g)
(connected_to_two_others : ∀ g ∈ green_squares, ∃ g1 g2 ∈ green_squares, g1 ≠ g2 ∧ (streets g g1 ∨ streets g1 g) ∧ (streets g g2 ∨ streets g2 g))

def city_strongly_connected {α : Type u} (c : City α) : Prop :=
∀ a b : α, c.streets a b ∨ c.streets b a ∨ (∃ p, p 0 = a ∧ p (p.length - 1) = b ∧ ∀ i < p.length - 1, c.streets (p i) (p (i + 1)))

theorem city_path_between_squares (α : Type u) (c : City α) :
  city_strongly_connected c :=
sorry

end city_path_between_squares_l107_107881


namespace highest_number_paper_l107_107459

theorem highest_number_paper (n : ℕ) (h : (1 : ℝ) / n = 0.010526315789473684) : n = 95 :=
sorry

end highest_number_paper_l107_107459


namespace mulch_cost_l107_107068

-- Definitions based on conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yard_to_cubic_feet : ℕ := 27
def volume_in_cubic_yards : ℕ := 7

-- Target statement to prove
theorem mulch_cost :
    (volume_in_cubic_yards * cubic_yard_to_cubic_feet) * cost_per_cubic_foot = 1512 := by
  sorry

end mulch_cost_l107_107068


namespace base_seven_sum_of_product_l107_107583

theorem base_seven_sum_of_product :
  let b7_35 := 3 * 7^1 + 5 * 7^0 in
  let b7_42 := 4 * 7^1 + 2 * 7^0 in
  let decimal_product := b7_35 * b7_42 in
  let b7_product_digits_sum := 2 + 1 + 6 + 3 in
  let base_seven_digits_sum := 1 * 7^1 + 5 * 7^0 in
  b7_product_digits_sum = base_seven_digits_sum :=
by {
  sorry
}

end base_seven_sum_of_product_l107_107583


namespace smallest_prime_12_less_than_square_l107_107185

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l107_107185


namespace balls_into_boxes_l107_107856

noncomputable def countDistributions : ℕ :=
  sorry

theorem balls_into_boxes :
  countDistributions = 8 :=
  sorry

end balls_into_boxes_l107_107856


namespace inequality_solution_intervals_l107_107556

theorem inequality_solution_intervals (x : ℝ) (h : x > 2) : 
  (x-2)^(x^2 - 6 * x + 8) > 1 ↔ (2 < x ∧ x < 3) ∨ x > 4 := 
sorry

end inequality_solution_intervals_l107_107556


namespace hexagon_diagonals_intersect_at_centroid_l107_107128

-- Definitions for the problem's conditions
variables (A B C : Type) [AffineSpace ℝ A] (triangle : Simplex ℝ A 2)
noncomputable def divides_into_three_equal_parts (p q r : ℝ) (a1 a2 b1 b2 c1 c2 : A): Prop := sorry

-- Statement of the theorem
theorem hexagon_diagonals_intersect_at_centroid (A B C : A)
  (Ha : divides_into_three_equal_parts B C A (point_on_opposite_side B C A))
  (Hb : divides_into_three_equal_parts A C B (point_on_opposite_side A C B))
  (Hc : divides_into_three_equal_parts A B C (point_on_opposite_side A B C))
  : ∃ (G : A), is_centroid A B C G ∧ ∀ (P Q : A), (are_diagonals_of_hexagon A B C P Q) → intersect_at G P Q :=
sorry

end hexagon_diagonals_intersect_at_centroid_l107_107128


namespace math_question_l107_107778

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l107_107778


namespace distance_from_vertex_to_asymptote_l107_107570

theorem distance_from_vertex_to_asymptote :
  let h : ℝ → ℝ → Prop := λ x y, x^2 / 2 - y^2 / 4 = 1 in
  let vertex := (sqrt 2, 0) in
  let asymptote := λ x y, y = sqrt 2 * x in
  let distance := (2 * sqrt 3) / 3 in
  ∀ (x y : ℝ), h x y → vertex = (sqrt 2, 0) → y = sqrt 2 * x → distance = (2 * sqrt 3) / 3 := sorry

end distance_from_vertex_to_asymptote_l107_107570


namespace problem_solution_l107_107052

def s (n : Nat) : Nat := (Nat.binaryDigits n).count 1

theorem problem_solution : (1 / 255 : ℚ) * (Finset.sum (Finset.range 16) (λ n : ℕ, 2^n * (-1)^(s n))) = 45 :=
by
  sorry

end problem_solution_l107_107052


namespace shortest_distance_is_perpendicular_l107_107622

-- Definition of point and line in an affine space
structure Point (α : Type*) := (x : α) (y : α)
structure Line (α : Type*) := (a b : α) -- line equation ax + by = 1

-- Define a function to compute distance between a point and a line
def distance (α : Type*) [linear_ordered_field α] (p : Point α) (l : Line α) : α :=
  abs (l.a * p.x + l.b * p.y - 1) / (sqrt (l.a^2 + l.b^2))

-- The proposition we want to prove
theorem shortest_distance_is_perpendicular {α : Type*} [linear_ordered_field α] :
  ∀ (p : Point α) (l : Line α), 
  ∃ (q : Point α), (q.y * l.a - q.x * l.b - 1) = 0 ∧ 
  ∀ (r : Point α), (r.y * l.a - r.x * l.b - 1) = 0 → distance α p q ≤ distance α p r :=
sorry

end shortest_distance_is_perpendicular_l107_107622


namespace fib_150_mod_9_l107_107982

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib n + fib (n + 1)

-- State the main theorem
theorem fib_150_mod_9 : (fib 150) % 9 = 8 := 
sorry

end fib_150_mod_9_l107_107982


namespace equal_numbers_in_sequence_l107_107262

theorem equal_numbers_in_sequence (a : ℕ → ℚ)
  (h : ∀ m n : ℕ, a m + a n = a (m * n)) : 
  ∃ i j : ℕ, i ≠ j ∧ a i = a j :=
sorry

end equal_numbers_in_sequence_l107_107262


namespace area_of_PQR_l107_107237

noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs ((P.1 * (Q.2 - R.2)) + (Q.1 * (R.2 - P.2)) + (R.1 * (P.2 - Q.2)))

def PQR_area_condition (A B C O P Q R : ℝ × ℝ) : Prop :=
  let is_equilateral (A B C : ℝ × ℝ) : Prop :=
    dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = 2
  ∧ let is_isosceles_with_angle (O P A: ℝ × ℝ) (angle: ℝ) : Prop :=
    dist O P = dist O A ∧ 1 / π * (π / angle / 180) = 15
  in is_equilateral A B C ∧
     is_isosceles_with_angle O A P (15) ∧
     is_isosceles_with_angle O B Q (15) ∧
     is_isosceles_with_angle O C R (15)

theorem area_of_PQR (A B C O P Q R : ℝ × ℝ)
  (hPQR : PQR_area_condition A B C O P Q R) :
  triangle_area P Q R = 2 * real.sqrt 3 - 3 :=
sorry

end area_of_PQR_l107_107237


namespace relationship_among_a_b_c_l107_107932

noncomputable def a := real.cos (50 * real.pi / 180) * real.cos (127 * real.pi / 180) + real.cos (40 * real.pi / 180) * real.cos (37 * real.pi / 180)
noncomputable def b := (real.sqrt 2 / 2) * (real.sin (56 * real.pi / 180) - real.cos (56 * real.pi / 180))
noncomputable def c := (1 - real.tan (39 * real.pi / 180) ^ 2) / (1 + real.tan (39 * real.pi / 180) ^ 2)

theorem relationship_among_a_b_c : a > c ∧ c > b :=
by
  have ha : a = real.sin (13 * real.pi / 180), from sorry,
  have hb : b = real.sin (11 * real.pi / 180), from sorry,
  have hc : c = real.sin (12 * real.pi / 180), from sorry,
  have h1 : real.sin (11 * real.pi / 180) < real.sin (12 * real.pi / 180), from sorry,
  have h2 : real.sin (12 * real.pi / 180) < real.sin (13 * real.pi / 180), from sorry,
  exact ⟨by linarith [ha, hc, h2], by linarith [hb, hc, h1]⟩

end relationship_among_a_b_c_l107_107932


namespace find_b_l107_107392

theorem find_b (a b : ℝ) (h1 : a * (a - 4) = 21) (h2 : b * (b - 4) = 21) (h3 : a + b = 4) (h4 : a ≠ b) :
  b = -3 :=
sorry

end find_b_l107_107392


namespace cars_meet_in_3_hours_l107_107132

theorem cars_meet_in_3_hours
  (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (t : ℝ)
  (h_distance: distance = 333)
  (h_speed1: speed1 = 54)
  (h_speed2: speed2 = 57)
  (h_equation: speed1 * t + speed2 * t = distance) :
  t = 3 :=
sorry

end cars_meet_in_3_hours_l107_107132


namespace smallest_prime_less_than_square_l107_107153

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107153


namespace symmetric_points_origin_l107_107819

theorem symmetric_points_origin (a b : ℝ) (h : (1, 2) = (-a, -b)) : a = -1 ∧ b = -2 :=
sorry

end symmetric_points_origin_l107_107819


namespace median_eq_range_le_l107_107744

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l107_107744


namespace complex_exp_conj_sum_l107_107448

open Complex

theorem complex_exp_conj_sum {α β : ℝ}
  (h : exp (I * α) + exp (I * β) = (2 / 5 : ℂ) + (4 / 9 : ℂ) * I) :
  exp (-I * α) + exp (-I * β) = (2 / 5 : ℂ) - (4 / 9 : ℂ) * I :=
by
  sorry

end complex_exp_conj_sum_l107_107448


namespace smallest_sum_of_digits_form_sum_of_digits_form_1999_l107_107721

-- Define the function representing the form 3n^2 + n + 1
def form (n : ℕ) := 3 * n^2 + n + 1

-- Define the function to calculate the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else 
    (n % 10) + sum_of_digits (n / 10)

-- Proof problem for the smallest possible sum of digits of 3n^2 + n + 1
theorem smallest_sum_of_digits_form (exists_lower_bound : ∃ n : ℕ, sum_of_digits (form n)) :
  ∃ n : ℕ, sum_of_digits (form n) = 3 :=
sorry

-- Proof problem for the existence of a number with sum of digits 1999
theorem sum_of_digits_form_1999 (exists_large_bound : ∃ n : ℕ, sum_of_digits (form n)) :
  ∃ n : ℕ, sum_of_digits (form n) = 1999 :=
sorry

end smallest_sum_of_digits_form_sum_of_digits_form_1999_l107_107721


namespace total_animal_legs_l107_107278

def number_of_dogs : ℕ := 2
def number_of_chickens : ℕ := 1
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem total_animal_legs : number_of_dogs * legs_per_dog + number_of_chickens * legs_per_chicken = 10 :=
by
  -- The proof is skipped
  sorry

end total_animal_legs_l107_107278


namespace median_equality_range_inequality_l107_107803

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l107_107803


namespace smallest_prime_12_less_than_square_l107_107178

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107178


namespace smallest_prime_less_than_square_l107_107155

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l107_107155


namespace valid_m_condition_l107_107823

theorem valid_m_condition (m : ℝ) (f : ℝ → ℝ) :
  (∀ x > 0, f x = (m^2 - m - 1) * x ^ (m^2 - 2m - 2)) →
  (∀ x > 0, f x < f (x + 1)) →
  m = 2 :=
by sorry

end valid_m_condition_l107_107823


namespace find_point_C_l107_107478
noncomputable theory

-- Definitions based on conditions
def A : ℂ := 2 + I
def BA : ℂ := 1 + 2*I
def BC : ℂ := 3 - I

-- Proof statement
theorem find_point_C : ∃ C : ℂ, C = (4 - 2*I) :=
  let B : ℂ := A - BA in
  let C : ℂ := B + BC in
  by {
    use C,
    have hA : A = 2 + I := rfl,
    have hBA : BA = 1 + 2*I := rfl,
    have hB : B = 2 + I - (1 + 2*I) := rfl,
    have hBC : BC = 3 - I := rfl,
    have hC : C = (2 + I - (1 + 2*I)) + (3 - I) := rfl,
    have hC_simplified : C = 4 - 2*I := by ring,
    exact hC_simplified.symm
  }

end find_point_C_l107_107478


namespace maximize_pasture_area_l107_107662

theorem maximize_pasture_area
  (barn_length fence_cost budget : ℕ)
  (barn_length_eq : barn_length = 400)
  (fence_cost_eq : fence_cost = 5)
  (budget_eq : budget = 1500) :
  ∃ x y : ℕ, y = 150 ∧
  x > 0 ∧
  2 * x + y = budget / fence_cost ∧
  y = barn_length - 2 * x ∧
  (x * y) = (75 * 150) :=
by
  sorry

end maximize_pasture_area_l107_107662


namespace proof_problem_l107_107782

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l107_107782


namespace find_mountain_altitude_mountain_top_altitude_is_1830_l107_107452

theorem find_mountain_altitude (h_incr_temp : ∀ (d : ℕ), \(d \cdot 100\) → (0.3 * d) : Prop)
  (foot_altitude: ℕ) (foot_temp: ℝ) (top_temp: ℝ) : ℕ :=
begin
  let temp_decrease := foot_temp - top_temp by sorry,
  let num_increments := temp_decrease / 0.3 by sorry,
  let altitude_increase := num_increments * 100 by sorry,
  let top_altitude := foot_altitude + altitude_increase by sorry,
  exact top_altitude
end

theorem mountain_top_altitude_is_1830
  (h_incr_temp : ∀ (d : ℕ), \(d \cdot 100\) → (0.3 * d) : Prop)
  (foot_altitude: 1230 : ℕ) (foot_temp: 18 : ℝ) (top_temp: 16.2 : ℝ) :
  mountain_top (1830 : ℕ) := sorry

end find_mountain_altitude_mountain_top_altitude_is_1830_l107_107452


namespace base_9_contains_6_or_7_count_l107_107848

def contains_digit_6_or_7 (n : ℕ) : Prop :=
  let digits := (Nat.digits 9 n) in
  List.any digits (λ d, d = 6 ∨ d = 7)

theorem base_9_contains_6_or_7_count :
  ∃ k : ℕ, k = 386 ∧ 
           k = (Finset.filter (λ n, contains_digit_6_or_7 n)
                              (Finset.range 730)).card :=
by
  sorry

end base_9_contains_6_or_7_count_l107_107848


namespace log_cut_problem_l107_107125

theorem log_cut_problem (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x + 4 * y = 100) :
  2 * x + 3 * y = 70 := by
  sorry

end log_cut_problem_l107_107125


namespace proof_problem_l107_107784

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l107_107784


namespace odd_n_divides_pow_fact_sub_one_l107_107051

theorem odd_n_divides_pow_fact_sub_one
  {n : ℕ} (hn_pos : n > 0) (hn_odd : n % 2 = 1)
  : n ∣ (2 ^ (Nat.factorial n) - 1) :=
sorry

end odd_n_divides_pow_fact_sub_one_l107_107051


namespace arithmetic_sequence_inequality_l107_107017

def exists_inequality (a : ℕ → ℕ) (d n : ℕ) : Prop :=
  n ≥ 2 → d > 0 → 
  ∀ s : Finset ℕ,
    s.card = n + 2 →
    ∃ (i j ∈ s), i ≠ j ∧ 1 < (|a i - a j| : ℚ) / (n * d) ∧ (|a i - a j| : ℚ) / (n * d) < 2

theorem arithmetic_sequence_inequality
  {a : ℕ → ℕ} {d n : ℕ} :
  ∃ a, ∃ d, ∃ n, exists_inequality a d n :=
begin
  sorry
end

end arithmetic_sequence_inequality_l107_107017


namespace initial_number_proof_l107_107097

def initial_number : ℕ := 7899665
def result : ℕ := 7899593
def factor1 : ℕ := 12
def factor2 : ℕ := 3
def factor3 : ℕ := 2

def certain_value : ℕ := (factor1 * factor2) * factor3

theorem initial_number_proof :
  initial_number - certain_value = result := by
  sorry

end initial_number_proof_l107_107097


namespace balls_into_boxes_l107_107857

noncomputable def countDistributions : ℕ :=
  sorry

theorem balls_into_boxes :
  countDistributions = 8 :=
  sorry

end balls_into_boxes_l107_107857


namespace Cindy_correct_answer_l107_107302

theorem Cindy_correct_answer (x : ℕ) (h : (x - 14) / 4 = 28) : ((x - 5) / 7) * 4 = 69 := by
  sorry

end Cindy_correct_answer_l107_107302


namespace noemi_start_amount_l107_107530

/-
  Conditions:
    lost_roulette = -600
    won_blackjack = 400
    lost_poker = -400
    won_baccarat = 500
    meal_cost = 200
    purse_end = 1800

  Prove: start_amount == 2300
-/

noncomputable def lost_roulette : Int := -600
noncomputable def won_blackjack : Int := 400
noncomputable def lost_poker : Int := -400
noncomputable def won_baccarat : Int := 500
noncomputable def meal_cost : Int := 200
noncomputable def purse_end : Int := 1800

noncomputable def net_gain : Int := lost_roulette + won_blackjack + lost_poker + won_baccarat

noncomputable def start_amount : Int := net_gain + meal_cost + purse_end

theorem noemi_start_amount : start_amount = 2300 :=
by
  sorry

end noemi_start_amount_l107_107530


namespace man_older_than_son_l107_107255

theorem man_older_than_son (S M : ℕ) (h1 : S = 23) (h2 : M + 2 = 2 * (S + 2)) : M - S = 25 :=
by
  subst h1
  linarith
  -- skip the proof
  sorry

end man_older_than_son_l107_107255


namespace ceil_neg_3_87_l107_107344

theorem ceil_neg_3_87 : ⌈-3.87⌉ = -3 := by
  sorry

end ceil_neg_3_87_l107_107344


namespace smallest_value_between_0_and_1_l107_107866

theorem smallest_value_between_0_and_1 (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∀ y ∈ { x, x^2, 3 * x, Real.sqrt x, 1 / x }, x^2 ≤ y :=
by
  intro y hy
  cases hy with
  | inl hx => rw [hx]; apply le_of_lt; exact (h.left)
  | inr hy => cases hy with
    | inl hx2 => rw [hx2]; apply le_refl
    | inr hy => cases hy with
      | inl h3x => rw [h3x]; have h_pos: 0 < 3 * x := mul_pos (by norm_num) h.left
                    exact le_of_lt h_pos
      | inr hy => cases hy with
        | inl h_sqrtx => rw [h_sqrtx]; 
          exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (by norm_num)
        | inr hfractionx => rw [hfractionx]; exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (one_div_pos_of_pos h.left)

#align smallest_value_between_0_and_1 smallest_value_between_0_and_1

end smallest_value_between_0_and_1_l107_107866


namespace dance_contradiction_l107_107287

variable {Boy Girl : Type}
variable {danced_with : Boy → Girl → Prop}

theorem dance_contradiction
    (H1 : ¬ ∃ g : Boy, ∀ f : Girl, danced_with g f)
    (H2 : ∀ f : Girl, ∃ g : Boy, danced_with g f) :
    ∃ (g g' : Boy) (f f' : Girl),
        danced_with g f ∧ ¬ danced_with g f' ∧
        danced_with g' f' ∧ ¬ danced_with g' f :=
by
  -- Proof will be inserted here
  sorry

end dance_contradiction_l107_107287


namespace find_f_22_l107_107730

noncomputable def f : ℝ → ℝ := sorry

axiom f_periodicity (x : ℝ) : f(x + 6) + f(x) = 2 * f(3)
axiom f_symmetry (x : ℝ) : f(2 - x) = -f(x - 2)
axiom f_value : f(2) = 4

theorem find_f_22 : f(22) = -4 :=
by
  sorry

end find_f_22_l107_107730


namespace correct_transformation_l107_107222

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0): (a / b = 2 * a / (2 * b)) :=
by
  sorry

end correct_transformation_l107_107222


namespace sum_of_medians_l107_107587

-- Define the median scores for players A and B
def median_A : ℕ := 28
def median_B : ℕ := 36

-- Prove that the sum of the median scores of players A and B is 64
theorem sum_of_medians (A_median : ℕ) (B_median : ℕ)
  (hA : A_median = median_A)
  (hB : B_median = median_B) :
  A_median + B_median = 64 :=
  by {
    rw [hA, hB],
    show median_A + median_B = 64,
    sorry 
  }

end sum_of_medians_l107_107587


namespace cube_root_of_64_l107_107990

theorem cube_root_of_64 :
  ∃ (x : ℝ), x^3 = 64 ∧ x = 4 :=
by
  use 4
  split
  · norm_num
  · refl

end cube_root_of_64_l107_107990


namespace tangent_line_at_point_l107_107107

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
  y = x^3 + x + 1

theorem tangent_line_at_point : 
  ∀ (x y : ℝ), tangent_line_equation x y → (x = 1 ∧ y = 3) → 4 * x - y - 1 = 0 :=
by
  intros x y eq pt,
  sorry

end tangent_line_at_point_l107_107107


namespace div_group_sub_sum_l107_107924

theorem div_group_sub_sum (n k : ℕ) (h_n_gt_one : n > 1) (h_k_gt_one : k > 1) (h_n_lt_2_pow_k : n < 2^k) :
  ∃ (S : Finset ℤ), S.card = 2 * k ∧
  ∀ (A B : Finset ℤ), A ∪ B = S → A ∩ B = ∅ →
  ∃ (C : Finset ℤ), C ⊆ A ∨ C ⊆ B ∧ (∑ x in C, x) % n = 0 :=
sorry

end div_group_sub_sum_l107_107924


namespace simplify_expression_l107_107972

theorem simplify_expression (x y : ℤ) : 1 - (2 - (3 - (4 - (5 - x)))) - y = 3 - (x + y) := 
by 
  sorry 

end simplify_expression_l107_107972


namespace problem_1_solution_set_problem_2_inequality_l107_107835

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem problem_1_solution_set (x : ℝ) : 
  f(x) >= 3 ↔ (x <= -1/2 ∨ x >= 2) :=
begin
  sorry
end

theorem problem_2_inequality (x a : ℝ) (ha : a <= -1/2 ∨ a >= 2) : 
  |x + a| + |x - (1 / a)| >= 5 / 2 :=
begin
  sorry
end

end problem_1_solution_set_problem_2_inequality_l107_107835


namespace gillian_spending_l107_107546

theorem gillian_spending (sandi_initial : ℕ) (sandi_fraction : ℕ) (gillian_extra : ℕ) (sandi_final : ℕ) (gillian_total : ℕ) :
  sandi_initial = 600 → sandi_fraction = 2 → gillian_extra = 150 →
  sandi_final = sandi_initial / sandi_fraction →
  gillian_total = 3 * sandi_final + gillian_extra →
  gillian_total = 1050 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end gillian_spending_l107_107546


namespace original_number_is_24_l107_107220

theorem original_number_is_24 (N : ℕ) 
  (h1 : (N + 1) % 25 = 0)
  (h2 : 1 = 1) : N = 24 := 
sorry

end original_number_is_24_l107_107220


namespace files_per_folder_l107_107065

theorem files_per_folder
    (initial_files : ℕ)
    (deleted_files : ℕ)
    (folders : ℕ)
    (remaining_files : ℕ)
    (files_per_folder : ℕ)
    (initial_files_eq : initial_files = 93)
    (deleted_files_eq : deleted_files = 21)
    (folders_eq : folders = 9)
    (remaining_files_eq : remaining_files = initial_files - deleted_files)
    (files_per_folder_eq : files_per_folder = remaining_files / folders) :
    files_per_folder = 8 :=
by
    -- Here, sorry is used to skip the actual proof steps 
    sorry

end files_per_folder_l107_107065


namespace binom_600_eq_1_l107_107316

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l107_107316


namespace smallest_positive_prime_12_less_than_square_is_13_l107_107171

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l107_107171


namespace mean_and_median_change_l107_107032

def participants_week_days : List ℕ := [25, 29, 26, 18, 21]

def corrected_thursday_participants : ℕ := 24

def original_total : ℕ := participants_week_days.sum

def new_total : ℕ := original_total - 18 + corrected_thursday_participants

def original_mean : ℝ := original_total / 5

def new_mean : ℝ := new_total / 5

def change_in_mean : ℝ := new_mean - original_mean

def original_median : ℕ :=
  let sorted := participants_week_days.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

def new_participants_week_days : List ℕ := [25, 29, 26, corrected_thursday_participants, 21]

def new_median : ℕ :=
  let sorted := new_participants_week_days.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

def change_in_median : ℕ := new_median - original_median

theorem mean_and_median_change :
  change_in_mean = 1.2 ∧ change_in_median = 0 :=
by
  sorry

end mean_and_median_change_l107_107032


namespace solve_quadratic_eq_l107_107974

theorem solve_quadratic_eq {x : ℝ} (h : x^2 - 5*x + 6 = 0) : x = 2 ∨ x = 3 :=
sorry

end solve_quadratic_eq_l107_107974


namespace pirate_loot_total_base10_l107_107260

/-- Convert a base-6 number to base-10 -/
def base6_to_base10 (digits : List ℕ) : ℕ :=
  digits.reverse.enum.zipWith (λ ⟨i, d⟩, d * 6 ^ i) id.sum

/-- The total dollar amount of the pirate's loot in base-10 -/
theorem pirate_loot_total_base10 :
  base6_to_base10 [2, 3, 5, 4] + base6_to_base10 [4, 5, 2, 1] + base6_to_base10 [4, 5, 6] = 1636 :=
by
  sorry

end pirate_loot_total_base10_l107_107260


namespace median_equality_range_inequality_l107_107800

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l107_107800


namespace binom_600_600_l107_107322

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l107_107322


namespace solve_eq_1_solve_eq_2_l107_107555

open Real

theorem solve_eq_1 :
  ∃ x : ℝ, x - 2 * (x - 4) = 3 * (1 - x) ∧ x = -2.5 :=
by
  sorry

theorem solve_eq_2 :
  ∃ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 60 = 1 ∧ x = 39 / 35 :=
by
  sorry

end solve_eq_1_solve_eq_2_l107_107555


namespace number_of_integers_with_6_or_7_as_digit_in_base9_l107_107850

/-- 
  There are 729 smallest positive integers written in base 9.
  We want to determine how many of these integers use the digits 6 or 7 (or both) at least once.
-/
theorem number_of_integers_with_6_or_7_as_digit_in_base9 : 
  ∃ n : ℕ, n = 729 ∧ ∃ m : ℕ, m = n - 7^3 := sorry

end number_of_integers_with_6_or_7_as_digit_in_base9_l107_107850


namespace smallest_prime_12_less_than_perfect_square_l107_107206

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107206


namespace smallest_prime_12_less_than_square_l107_107182

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l107_107182


namespace median_equal_range_not_greater_l107_107762

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l107_107762


namespace prob_draw_is_half_l107_107077

variable (Ω : Type) [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
variable (A Draw : Set Ω)

-- Conditions
variable (hA : P A = 0.3)
variable (hAnotLosing : P (A ∪ Draw) = 0.8)

-- To prove
theorem prob_draw_is_half : P Draw = 0.5 := 
by 
  -- Proof omitted
  sorry

end prob_draw_is_half_l107_107077


namespace line_eq_l107_107393

-- Definitions based on conditions
def passesThrough (x y : ℝ) (a b : ℝ) := b = mx + c

-- The statement to be proved in lean 4
theorem line_eq (a : ℝ) (h1 : passesThrough 2 3 a _) (h2 : x_intercept_eq_2_y_intercept a) :
  ∃ (m b : ℝ), m = -1/2 ∧ b = 4 ∧ y = -1/2 * x + 4 := 
begin
  sorry
end

end line_eq_l107_107393


namespace trigonometric_sum_tangent_l107_107092

theorem trigonometric_sum_tangent (x : ℝ) (n : ℕ) :
  (∑ i in (finset.range n).map (λ k, 2*k+1), sin ((2*k+1)*x)) /
  (∑ i in (finset.range n).map (λ k, 2*k+1), cos ((2*k+1)*x)) = 
  tan (n * x) :=
sorry

end trigonometric_sum_tangent_l107_107092


namespace pc_eq_cr_l107_107105

variables {α : Type*} [Field α] 
variables {A B C D O P Q R : Point α}
variables [IsParallelogram A B C D] [IsParallelogram D O C P]
variables (B P Q C Q A intersect AC) (D Q R C P intersect CP)

theorem pc_eq_cr :
  let O := intersect A C B D in
  let Q := intersect B P A C in
  let R := intersect D Q C P in
  PC = CR :=
sorry

end pc_eq_cr_l107_107105


namespace solve_trig_proof_l107_107386

noncomputable def trig_proof (x y : ℝ) : Prop :=
  sin x + sin y = 3/5 ∧ cos x + cos y = 4/5 → tan x + tan y = -4/3

theorem solve_trig_proof (x y : ℝ) : trig_proof x y :=
by
  intro h
  cases h with h1 h2
  sorry

end solve_trig_proof_l107_107386


namespace solution_set_l107_107923

noncomputable def domain := Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x, x ∈ domain → x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
axiom f_odd : ∀ x, f x + f (-x) = 0
def f' : ℝ → ℝ := sorry
axiom derivative_condition : ∀ x, 0 < x ∧ x < Real.pi / 2 → f' x * Real.cos x + f x * Real.sin x < 0

theorem solution_set :
  {x | f x < Real.sqrt 2 * f (Real.pi / 4) * Real.cos x} = {x | Real.pi / 4 < x ∧ x < Real.pi / 2} :=
sorry

end solution_set_l107_107923


namespace smallest_prime_less_than_perfect_square_is_13_l107_107195

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l107_107195


namespace correct_props_l107_107241

variables {m n : Type} {α β : Type}
variables [Parallel m Type] [Parallel n Type] [Parallel α Type] [Parallel β Type]
variables [Perpendicular m Type] [Perpendicular n Type] [Perpendicular α Type] [Perpendicular β Type]

-- Define hypothesis for propositions:
variables (h1 : ∀ (m α : Type), Parallel m α)
variables (h2 : ∀ (n β : Type), Parallel n β)
variables (h3 : ∀ (α β : Type), Parallel α β)

variables (k1 : ∀ (m α : Type), Perpendicular m α)
variables (k2 : ∀ (n β : Type), Perpendicular n β)
variables (k3 : ∀ (α β : Type), Perpendicular α β)

def prop1 := h1 m α → h2 n β → h3 α β → Parallel m n
def prop2 := k1 m α → k2 n β → k3 α β → Perpendicular m n
def prop3 := h1 m α → Parallel m n → h2 n α
def prop4 := h3 α β → k1 m α → h2 n β → Perpendicular m n

theorem correct_props :
  prop2 ∧ prop4 := 
by 
  sorry

end correct_props_l107_107241


namespace shaded_area_in_design_l107_107910

theorem shaded_area_in_design (side_length : ℝ) (radius : ℝ)
  (h1 : side_length = 30) (h2 : radius = side_length / 6)
  (h3 : 6 * (π * radius^2) = 150 * π) :
  (side_length^2) - 6 * (π * radius^2) = 900 - 150 * π := 
by
  sorry

end shaded_area_in_design_l107_107910


namespace common_ratio_of_cos_geometric_sequence_l107_107425

theorem common_ratio_of_cos_geometric_sequence 
  (a : ℕ → ℝ)
  (cos_seq_is_geom : ∃ q : ℝ, ∀ n : ℕ, cos (a (n + 1)) = q * cos (a n))
  (first_term : ℝ)
  (common_diff : ℝ)
  (h1 : a 0 = first_term)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + common_diff)
  (h_diff_range : 0 < common_diff ∧ common_diff < 2 * Real.pi) :
  ∃ q : ℝ, q = -1 :=
by
  sorry

end common_ratio_of_cos_geometric_sequence_l107_107425


namespace total_legs_correct_l107_107277

-- Define the number of animals
def num_dogs : ℕ := 2
def num_chickens : ℕ := 1

-- Define the number of legs per animal
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

-- Define the total number of legs from dogs and chickens
def total_legs : ℕ := num_dogs * legs_per_dog + num_chickens * legs_per_chicken

theorem total_legs_correct : total_legs = 10 :=
by
  -- this is where the proof would go, but we add sorry for now to skip it
  sorry

end total_legs_correct_l107_107277


namespace stratified_sampling_number_of_females_l107_107014

theorem stratified_sampling_number_of_females
  (male_students : ℕ)
  (female_students : ℕ)
  (total_sampled : ℕ)
  (h1 : male_students = 810)
  (h2 : female_students = 540)
  (h3 : total_sampled = 200) :
  ∃ (n_female : ℕ), n_female = (female_students * total_sampled) / (male_students + female_students) ∧ n_female = 80 := by
  exist 80
  sorry

end stratified_sampling_number_of_females_l107_107014


namespace solve_fractional_equation_l107_107095

theorem solve_fractional_equation : 
  ∃ x : ℝ, (x - 1) / 2 = 1 - (3 * x + 2) / 5 ↔ x = 1 := 
sorry

end solve_fractional_equation_l107_107095


namespace smallest_prime_12_less_than_perfect_square_l107_107203

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l107_107203


namespace roots_on_same_circle_l107_107841

theorem roots_on_same_circle (m : ℝ) :
  (∀ (x : ℂ), x^2 - 2 * x + 2 = 0 → (∃ r : ℝ, abs x = r))
  ∧ (∀ (x : ℂ), x^2 + 2 * m * x + 1 = 0 → (∃ r : ℝ, abs x = r))
  ↔ (-1 < m ∧ m < 1) ∨ (m = -3/2) := sorry

end roots_on_same_circle_l107_107841


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107794

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l107_107794


namespace cos_sum_identity_l107_107965

theorem cos_sum_identity :
  cos (2 * Real.pi / 17) + cos (6 * Real.pi / 17) + cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_identity_l107_107965


namespace part1_part2_l107_107418

noncomputable def f (a x: ℝ) : ℝ := a * x^2 - (a + 2) * x + real.log x

theorem part1 (h : 1 = 1) : 
  let a := 1 in
  let y1 := f a 1 in 
  let slope := (deriv (f a)) 1 in 
  y1 = -2 ∧ slope = 0 :=
by
  let a := 1
  let y1 := f a 1
  have slope := (deriv (f a)) 1
  split
  exact rfl
  exact rfl

theorem part2 (h' : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f a x1 + 2 * x1 < f a x2 + 2 * x2) :
  0 ≤ a ∧ a ≤ 8 :=
by
  sorry

end part1_part2_l107_107418


namespace coffee_students_l107_107883

variable (S : ℝ) -- Total number of students
variable (T : ℝ) -- Number of students who chose tea
variable (C : ℝ) -- Number of students who chose coffee

-- Given conditions
axiom h1 : 0.4 * S = 80   -- 40% of the students chose tea
axiom h2 : T = 80         -- Number of students who chose tea is 80
axiom h3 : 0.3 * S = C    -- 30% of the students chose coffee

-- Prove that the number of students who chose coffee is 60
theorem coffee_students : C = 60 := by
  sorry

end coffee_students_l107_107883


namespace Cody_meets_Daisy_distance_l107_107607

noncomputable def distance_Cody_skis (t : ℝ) : ℝ := 5 * t

theorem Cody_meets_Daisy_distance :
  let CD := 150
  let theta := real.pi / 4
  let speed_Cody := 5
  let speed_Daisy := 6
  let EC := distance_Cody_skis (sqrt (22500 / (61 - 30 * real.sqrt 2)))
  EC = 375 * real.sqrt (1 / (61 - 30 * real.sqrt 2)) :=
begin
  sorry
end

end Cody_meets_Daisy_distance_l107_107607


namespace smallest_prime_less_than_perfect_square_l107_107158

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l107_107158
