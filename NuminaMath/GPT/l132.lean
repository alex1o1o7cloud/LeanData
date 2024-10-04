import Complex.NumberTheory
import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.MulAction
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialNumberTheory
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Determinant
import Mathlib.Matrix
import Mathlib.NumberTheory.Primes
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassBasic
import Mathlib.Probability.StochasticProcess
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Real

namespace largest_divisor_of_consecutive_even_product_plus_two_is_one_l132_132764

theorem largest_divisor_of_consecutive_even_product_plus_two_is_one :
  ∀ n : ℕ, ∃ k : ℕ, (∀ d : ℕ, d > 1 → ¬ d ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4) + 2)) ↔ k = 1 :=
by
  intro n
  use 1
  intro d hd
  sorry

end largest_divisor_of_consecutive_even_product_plus_two_is_one_l132_132764


namespace determine_all_functions_l132_132028

-- Define the natural numbers (ℕ) as positive integers
def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

theorem determine_all_functions (g : ℕ → ℕ) :
  (∀ m n : ℕ, is_perfect_square ((g m + n) * (m + g n))) →
  ∃ c : ℕ, ∀ n : ℕ, g n = n + c :=
by
  sorry

end determine_all_functions_l132_132028


namespace lamps_on_middle_layer_l132_132523

theorem lamps_on_middle_layer (a r : ℕ) (h1 : r = 2) (h2 : a * (r^6 * r - 1) / (r - 1) = 381): 
  a * r^3 = 24 :=
by
  -- Definitions and conditions
  have h_sum_eq : a * (r^7 - 1) / (r - 1) = 381 := by
    rw [← h2]
  -- Prove the desired result
  sorry

end lamps_on_middle_layer_l132_132523


namespace sum_reciprocal_bound_l132_132530

def f : ℕ → ℝ
| 1       := 2
| (n + 1) := (f n) ^ 2 - (f n) + 1

def sum_reciprocal (n : ℕ) : ℝ :=
  ∑ i in (Finset.range n).map Nat.succ, (1 / f i)

theorem sum_reciprocal_bound (n : ℕ) (hn : n > 1) :
  1 - (1 / 2 ^ (2 * n - 1)) < sum_reciprocal n ∧ sum_reciprocal n < 1 - (1 / 2 ^ 4) :=
by
  sorry

end sum_reciprocal_bound_l132_132530


namespace not_real_range_l132_132067

def is_range_real (f : ℝ → ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f x = y

-- Given equation: ab = a + b + 3
def satisfies_condition (a b : ℝ) : Prop := a * b = a + b + 3

theorem not_real_range (a b : ℝ) (h : satisfies_condition a b) : ¬ is_range_real (λ (x : ℝ), (x - 1) * (1 + 4 / (x - 1))) :=
sorry

end not_real_range_l132_132067


namespace simplify_trig_expression_l132_132444

variable {A : ℝ}

-- Conditions
def cot (A : ℝ) : ℝ := cos A / sin A
def csc (A : ℝ) : ℝ := 1 / sin A
def tan (A : ℝ) : ℝ := sin A / cos A
def sec (A : ℝ) : ℝ := 1 / cos A

-- Proof problem
theorem simplify_trig_expression : (1 + cot A + csc A) * (1 + tan A - sec A) = 2 := by
  sorry

end simplify_trig_expression_l132_132444


namespace batteries_on_flashlights_l132_132495

variable (b_flashlights b_toys b_controllers b_total : ℕ)

theorem batteries_on_flashlights :
  b_toys = 15 → 
  b_controllers = 2 → 
  b_total = 19 → 
  b_total = b_flashlights + b_toys + b_controllers → 
  b_flashlights = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end batteries_on_flashlights_l132_132495


namespace neg_prop_p_l132_132682

variable (x : ℝ)

def prop_p : Prop := ∃ x : ℝ, sin x + cos x > real.sqrt 2

theorem neg_prop_p :
  ¬prop_p ↔ ∀ x : ℝ, sin x + cos x ≤ real.sqrt 2 := by
  sorry

end neg_prop_p_l132_132682


namespace trajectory_eq_l132_132555

theorem trajectory_eq (x y : ℝ) (h : real.sqrt (x^2 + (y - 2)^2) = abs y + 2) : 
  x^2 = 8 * y ∧ y ≥ 0 :=
by
  sorry

end trajectory_eq_l132_132555


namespace min_value_geom_seq_l132_132048

noncomputable theory

theorem min_value_geom_seq 
  (r : ℝ)
  (b1 b2 b3 : ℝ)
  (h1 : b1 = 2)
  (h2 : b2 = 2 * r)
  (h3 : b3 = 2 * r^2) :
  ∃ r, 5 * b2 + 6 * b3 = -25 / 12 :=
by sorry

end min_value_geom_seq_l132_132048


namespace solve_for_m_l132_132680

theorem solve_for_m {m : ℝ} :
  (∀ x : ℝ, -7 < x → x < -1 → mx^2 + 8mx + 28 < 0) →
  m = 4 :=
by
  intro h
  sorry

end solve_for_m_l132_132680


namespace remainder_9995_eq_25_l132_132154

theorem remainder_9995_eq_25 (x : ℕ) (hx : Nat.IsComposite x) (h : 5000 % x = 25) : 9995 % x = 25 := 
by
sory

end remainder_9995_eq_25_l132_132154


namespace options_not_equal_l132_132508

theorem options_not_equal (a b c d e : ℚ)
  (ha : a = 14 / 10)
  (hb : b = 1 + 2 / 5)
  (hc : c = 1 + 7 / 25)
  (hd : d = 1 + 2 / 10)
  (he : e = 1 + 14 / 70) :
  a = 7 / 5 ∧ b = 7 / 5 ∧ c ≠ 7 / 5 ∧ d ≠ 7 / 5 ∧ e ≠ 7 / 5 :=
by sorry

end options_not_equal_l132_132508


namespace problem_solution_l132_132325

open Real

-- Define the vectors a and b
def vector_a (α : ℝ) : ℝ × ℝ := (2 * sin α, cos α)
def vector_b : ℝ × ℝ := (1, -1)

-- Define the condition that a is perpendicular to b
def perp_condition (α : ℝ) : Prop := 
  (2 * sin α) * 1 + (cos α) * (-1) = 0

-- Define the squared magnitude of the difference of vectors
def squared_magnitude_of_difference (α : ℝ) : ℝ :=
  let a := vector_a α
  let b := vector_b
  let diff := (a.1 - b.1, a.2 - b.2)
  diff.1^2 + diff.2^2

-- The theorem to be proven
theorem problem_solution (α : ℝ) (h : perp_condition α) : 
  squared_magnitude_of_difference α = 18 / 5 :=
sorry

end problem_solution_l132_132325


namespace problem_1_problem_2_l132_132814

-- Problem 1
theorem problem_1 (n : ℝ) (h : 3 * 9^(2 * n) * 27^n = 3^(2 * n)) : n = -1 / 5 := 
sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (ha : 10^a = 5) (hb : 10^b = 6) : 10^(2 * a + 3 * b) = 5400 := 
sorry

end problem_1_problem_2_l132_132814


namespace max_real_part_sum_w_l132_132772

noncomputable def z (j : ℕ) : ℂ := 32 * complex.exp (2 * real.pi * complex.I * j / 10)

def w (j : ℕ) : ℂ :=
if real.cos (2 * real.pi * j / 10) ≥ 0 then
  z j
else
  -complex.I * z j

theorem max_real_part_sum_w :
  ∀ (z : ℕ → ℂ) (w : ℕ → ℂ),
  (∀ j, z j = 32 * complex.exp (2 * real.pi * complex.I * j / 10)) →
  (∀ j, w j = (if real.cos (2 * real.pi * j / 10) ≥ 0 then z j else -complex.I * z j)) →
  ∑ j in finrange 10, (w j).re =
  32 * (1 + (1 + real.sqrt 5) / 2 - (real.sqrt (10 - 2 * real.sqrt 5) + real.sqrt (10 + 2 * real.sqrt 5))) :=
by {
  sorry
}

end max_real_part_sum_w_l132_132772


namespace cookies_per_bag_calc_l132_132057

theorem cookies_per_bag_calc :
  let chocolate_chip_cookies := 13
  let oatmeal_cookies := 41
  let baggies := 6
  let total_cookies := chocolate_chip_cookies + oatmeal_cookies
in total_cookies / baggies = 9 := by
  sorry

end cookies_per_bag_calc_l132_132057


namespace yellow_balls_count_l132_132174

def number_of_yellow_balls (total : ℕ) (white : ℕ) (green : ℕ) (red : ℕ) (purple : ℕ) (prob_neither_red_nor_purple : ℝ) : ℕ :=
  total - (white + green + red + purple)

theorem yellow_balls_count (total white green red purple : ℕ) (prob_neither_red_nor_purple : ℝ) :
  total = 60 → white = 22 → green = 18 → red = 5 → purple = 7 → prob_neither_red_nor_purple = 0.8 →
  number_of_yellow_balls total white green red purple prob_neither_red_nor_purple = 8 :=
by
  intros h_total h_white h_green h_red h_purple h_prob
  simp [number_of_yellow_balls, h_total, h_white, h_green, h_red, h_purple, h_prob]
  rw [←sub_eq_add_neg]
  norm_num
  sorry

end yellow_balls_count_l132_132174


namespace surrounding_circles_radius_l132_132179

theorem surrounding_circles_radius :
  let r := (1 + Real.sqrt 5) / 2 in
  ∃ c : ℝ, c = 2 ∧
  let SURROUND_CNT := 6 in 
  let HYP := 2 + r in
  let LEG_OPP := r in
  let LEG_ADJ := Real.sqrt 3 * r in
  HYP * HYP = LEG_OPP * LEG_OPP + LEG_ADJ * LEG_ADJ :=
sorry

end surrounding_circles_radius_l132_132179


namespace inequality_solution_l132_132347

theorem inequality_solution {a b x : ℝ} 
  (h_sol_set : -1 < x ∧ x < 1) 
  (h1 : x - a > 2) 
  (h2 : b - 2 * x > 0) : 
  (a + b) ^ 2021 = -1 := 
by 
  sorry 

end inequality_solution_l132_132347


namespace symmetric_lines_intersect_at_single_point_l132_132804

-- Define a triangle and its orthocenter
variables {A B C H : Type} [triangle : Triangle A B C] [orthocenter : Orthocenter H A B C]
variables {H1 H2 H3 : Type} [reflectionH1 : Reflection H BC H1] [reflectionH2 : Reflection H CA H2] [reflectionH3 : Reflection H AB H3]

-- Define an arbitrary line passing through the orthocenter
variable {l : Line} [passesThroughH : PassesThrough l H]

-- Define the reflected lines with respect to the sides of the triangle
variables {l1 l2 l3 : Line} [reflectionL1 : ReflectionLine l BC l1]
                              [reflectionL2 : ReflectionLine l CA l2]
                              [reflectionL3 : ReflectionLine l AB l3]

-- Define the circumcircle of the triangle
variable {circumcircle : Circumcircle A B C}

-- Theorem statement
theorem symmetric_lines_intersect_at_single_point :
  intersect_at_single_point l1 l2 l3 :=
sorry

end symmetric_lines_intersect_at_single_point_l132_132804


namespace complex_polynomial_coefficient_matching_l132_132767

noncomputable def distinct_triple {α : Type*} [linear_ordered_field α] 
  (u v w : α) : Prop := (u ≠ v) ∧ (v ≠ w) ∧ (u ≠ w)

theorem complex_polynomial_coefficient_matching 
  (c : ℂ) (u v w : ℂ) (hf : ∀ z : ℂ, (z - u) * (z - v) * (z - w) = (z - c * u) * (z - c * v) * (z - c * w))
  (h_uvw : distinct_triple u v w) :
  finset.card {c : ℂ | c^3 = 1} = 4 :=
by
  sorry

end complex_polynomial_coefficient_matching_l132_132767


namespace weaving_problem_l132_132448

def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + (n * (n - 1) / 2) * d

theorem weaving_problem :
  ∃ (d : ℚ), d = 16/29 ∧ 
    let a := 5 in
    arithmetic_sum a d 30 = 390 ∧ 
    let a14 := a + 13 * d in
    let a15 := a + 14 * d in
    let a16 := a + 15 * d in
    let a17 := a + 16 * d in
    a14 + a15 + a16 + a17 = 52 :=
by
  sorry

end weaving_problem_l132_132448


namespace parallel_lines_l132_132974

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end parallel_lines_l132_132974


namespace latitude_approx_l132_132482

noncomputable def calculate_latitude (R h : ℝ) (θ : ℝ) : ℝ :=
  if h = 0 then θ else Real.arccos (1 / (2 * Real.pi))

theorem latitude_approx (R h θ : ℝ) (h_nonzero : h ≠ 0)
  (r1 : ℝ := R * Real.cos θ)
  (r2 : ℝ := (R + h) * Real.cos θ)
  (s : ℝ := 2 * Real.pi * h * Real.cos θ)
  (condition : s = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end latitude_approx_l132_132482


namespace investment_plans_count_l132_132185

theorem investment_plans_count :
  let binom := Nat.choose
  ∃ (cnt : Nat), cnt = binom 5 3 * 3! + binom 5 1 * binom 4 1 * 3 ∧ cnt = 120 :=
by
  sorry

end investment_plans_count_l132_132185


namespace avg_of_sequence_is_x_l132_132086

noncomputable def sum_naturals (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem avg_of_sequence_is_x (x : ℝ) :
  let n := 100
  let sum := sum_naturals n
  (sum + x) / (n + 1) = 50 * x → 
  x = 5050 / 5049 :=
by
  intro n sum h
  exact sorry

end avg_of_sequence_is_x_l132_132086


namespace number_of_hens_l132_132553

theorem number_of_hens
    (H C : ℕ) -- Hens and Cows
    (h1 : H + C = 44) -- Condition 1: The number of heads
    (h2 : 2 * H + 4 * C = 128) -- Condition 2: The number of feet
    : H = 24 :=
by
  sorry

end number_of_hens_l132_132553


namespace isosceles_triangle_perimeter_l132_132361

theorem isosceles_triangle_perimeter (a : ℝ) (h1 : 3 + a > 6) (h2 : 6 + a > 3) (h3 : 3 + 6 > a) (h4 : 6 = a):
  3 + a + 6 = 15 :=
by
  rw [h4]
  norm_num

end isosceles_triangle_perimeter_l132_132361


namespace breadth_of_room_is_6_l132_132160

theorem breadth_of_room_is_6 
(the_room_length : ℝ) 
(the_carpet_width : ℝ) 
(cost_per_meter : ℝ) 
(total_cost : ℝ) 
(h1 : the_room_length = 15) 
(h2 : the_carpet_width = 0.75) 
(h3 : cost_per_meter = 0.30) 
(h4 : total_cost = 36) : 
  ∃ (breadth_of_room : ℝ), breadth_of_room = 6 :=
sorry

end breadth_of_room_is_6_l132_132160


namespace left_square_side_length_l132_132001

theorem left_square_side_length 
  (x y z : ℝ)
  (H1 : y = x + 17)
  (H2 : z = x + 11)
  (H3 : x + y + z = 52) : 
  x = 8 := by
  sorry

end left_square_side_length_l132_132001


namespace proof_problem_l132_132339

variable (a b c m : ℝ)

-- Condition
def condition : Prop := m = (c * a * b) / (a + b)

-- Question
def question : Prop := b = (m * a) / (c * a - m)

-- Proof statement
theorem proof_problem (h : condition a b c m) : question a b c m := 
sorry

end proof_problem_l132_132339


namespace count_divisors_31752_l132_132690

theorem count_divisors_31752 : (finset.filter (λ n, 31752 % n = 0) (finset.range 10)).card = 6 :=
by sorry

end count_divisors_31752_l132_132690


namespace number_of_even_perfect_square_factors_l132_132330

theorem number_of_even_perfect_square_factors :
  let N := (2 ^ 6) * (7 ^ 10) * (3 ^ 4)
  ∃ a b c : ℕ, 
    1 ≤ a ∧ a ≤ 6 ∧ a % 2 = 0 ∧ 0 ≤ b ∧ b ≤ 10 ∧ b % 2 = 0 ∧ 0 ≤ c ∧ c ≤ 4 ∧ c % 2 = 0 ∧
    let factors_count := (3 * 6 * 3),
    factors_count = 54 :=
begin
  sorry
end

end number_of_even_perfect_square_factors_l132_132330


namespace no_carry_pairs_count_l132_132639

-- Defining a function that checks if two consecutive integers sum without carry
def no_carry (n : ℕ) : Prop := 
n ≥ 3000 ∧ n < 4000 ∧ ∀ p, (p < 10 → ((n / 10^p) % 10 + ((n + 1) / 10^p) % 10) < 10)

theorem no_carry_pairs_count : 
  {n | no_carry n}.finite.to_finset.card = 729 := by sorry

end no_carry_pairs_count_l132_132639


namespace exist_sequences_l132_132408

theorem exist_sequences 
  (a b c : ℤ) 
  (a_nonneg : a ≥ 0)
  (b_nonneg : b ≥ 0)
  (cond : a * b ≥ c^2) : 
  ∃ (n : ℕ) (x y : ℕ → ℤ), 
  (∑ i in Finset.range n, x i ^ 2 = a) ∧ 
  (∑ i in Finset.range n, y i ^ 2 = b) ∧ 
  (∑ i in Finset.range n, x i * y i = c) :=
sorry

end exist_sequences_l132_132408


namespace merchant_loss_is_15_yuan_l132_132554

noncomputable def profit_cost_price : ℝ := (180 : ℝ) / 1.2
noncomputable def loss_cost_price : ℝ := (180 : ℝ) / 0.8

theorem merchant_loss_is_15_yuan :
  (180 + 180) - (profit_cost_price + loss_cost_price) = -15 := by
  sorry

end merchant_loss_is_15_yuan_l132_132554


namespace minimum_colors_for_tessellation_l132_132498

noncomputable section

-- Define the types Hexagon and Square, and their relations.
inductive Tile
| hexagon : Tile
| square : Tile

open Tile 

-- Define adjacency relation
def shares_side (t1 t2 : Tile) : Prop :=
match t1, t2 with
| hexagon, square => true
| square, hexagon => true
| square, square => true
| _, _ => false
end

-- Define the main theorem
theorem minimum_colors_for_tessellation : ∃ c : ℕ, c = 3 ∧ ∀ (f : Tile → ℕ), (∀ t1 t2, shares_side t1 t2 → f t1 ≠ f t2) → f (hexagon) = 1 ∧ f (square) ∈ {2, 3} :=
begin
  -- The proof is omitted as required by the instructions.
  sorry
end

end minimum_colors_for_tessellation_l132_132498


namespace prime_greater_than_five_div360_l132_132430

theorem prime_greater_than_five_div360 (p : ℕ) (hp : prime p) (hgt : p > 5) : 360 ∣ (p^4 - 5 * p^2 + 4) :=
sorry

end prime_greater_than_five_div360_l132_132430


namespace equation_of_line_l132_132128

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  (dot_product / norm_sq * v.1, dot_product / norm_sq * v.2)

theorem equation_of_line (x y : ℝ) :
  projection (x, y) (7, 3) = (-7, -3) →
  y = -7/3 * x - 58/3 :=
by
  intro h
  sorry

end equation_of_line_l132_132128


namespace sum_series_and_convergence_l132_132269

theorem sum_series_and_convergence (x : ℝ) (h : -1 < x ∧ x < 1) :
  ∑' n, (n + 6) * x^(7 * n) = (6 - 5 * x^7) / (1 - x^7)^2 :=
by
  sorry

end sum_series_and_convergence_l132_132269


namespace bella_grazing_area_l132_132547

open Real

theorem bella_grazing_area:
  let leash_length := 5
  let barn_width := 4
  let barn_height := 6
  let sector_fraction := 3 / 4
  let area_circle := π * leash_length^2
  let grazed_area := sector_fraction * area_circle
  grazed_area = 75 / 4 * π := 
by
  sorry

end bella_grazing_area_l132_132547


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l132_132851

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l132_132851


namespace range_of_a_l132_132290

structure PropositionP (a : ℝ) : Prop :=
  (h : 2 * a + 1 > 5)

structure PropositionQ (a : ℝ) : Prop :=
  (h : -1 ≤ a ∧ a ≤ 3)

theorem range_of_a (a : ℝ) (hp : PropositionP a ∨ PropositionQ a) (hq : ¬(PropositionP a ∧ PropositionQ a)) :
  (-1 ≤ a ∧ a ≤ 2) ∨ (a > 3) :=
sorry

end range_of_a_l132_132290


namespace smallest_nonpalindromic_power_of_7_l132_132633

noncomputable def isPalindrome (n : ℕ) : Bool :=
  let s := n.toString
  s == s.reverse

theorem smallest_nonpalindromic_power_of_7 :
  ∃ n : ℕ, ∃ m : ℕ, m = 7^n ∧ ¬ isPalindrome m ∧ ∀ k : ℕ, k < n → (isPalindrome (7^k) → False) → n = 4 ∧ m = 2401 :=
by sorry

end smallest_nonpalindromic_power_of_7_l132_132633


namespace correct_proposition_is_B_l132_132807

variables {m n : Type} {α β : Type}

-- Define parallel and perpendicular relationships
def parallel (l₁ l₂ : Type) : Prop := sorry
def perpendicular (l₁ l₂ : Type) : Prop := sorry

def lies_in (l : Type) (p : Type) : Prop := sorry

-- The problem statement
theorem correct_proposition_is_B
  (H1 : perpendicular m α)
  (H2 : perpendicular n β)
  (H3 : perpendicular α β) :
  perpendicular m n :=
sorry

end correct_proposition_is_B_l132_132807


namespace bill_bathroom_visits_per_day_l132_132951

theorem bill_bathroom_visits_per_day
  (squares_per_use : ℕ)
  (rolls : ℕ)
  (squares_per_roll : ℕ)
  (days_supply : ℕ)
  (total_uses : squares_per_use = 5)
  (total_rolls : rolls = 1000)
  (squares_from_each_roll : squares_per_roll = 300)
  (total_days : days_supply = 20000) :
  ( (rolls * squares_per_roll) / days_supply / squares_per_use ) = 3 :=
by
  sorry

end bill_bathroom_visits_per_day_l132_132951


namespace circle_area_l132_132178

noncomputable def area_of_circle (r : ℝ) : ℝ := π * r^2

theorem circle_area (C : ℝ) (r : ℝ) (hC : C = 2 * π * r) (hcircum : C = 36) :
  area_of_circle r = 324 / π :=
by
  sorry

end circle_area_l132_132178


namespace eval_f_3_minus_f_neg_3_l132_132338

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 7 * x

-- State the theorem
theorem eval_f_3_minus_f_neg_3 : f 3 - f (-3) = 690 := by
  sorry

end eval_f_3_minus_f_neg_3_l132_132338


namespace base_three_to_ten_l132_132149

theorem base_three_to_ten : base_three_to_ten (digit3 1) (digit3 0) (digit3 2) (digit3 2) (digit3 1) = 106 := 
by sorry

end base_three_to_ten_l132_132149


namespace frequency_of_second_group_l132_132564

theorem frequency_of_second_group (total_capacity : ℕ) (freq_percentage : ℝ)
    (h_capacity : total_capacity = 80)
    (h_percentage : freq_percentage = 0.15) :
    total_capacity * freq_percentage = 12 :=
by
  sorry

end frequency_of_second_group_l132_132564


namespace inequality_solution_l132_132996

theorem inequality_solution (x : ℝ) (h : x ≠ 0) : 
  (1 / (x^2 + 1) > 2 * x^2 / x + 13 / 10) ↔ (x ∈ Set.Ioo (-1.6) 0 ∨ x ∈ Set.Ioi 0.8) :=
by sorry

end inequality_solution_l132_132996


namespace initial_water_in_canteen_l132_132329

-- Definitions from the conditions
def distance : ℕ := 7
def time : ℕ := 3
def water_rem : ℕ := 2
def leak_rate : ℕ := 1
def last_mile_consumption : ℕ := 3
def first_6_miles_consumption_rate : ℕ := 0.5

-- Mathematically equivalent proof problem
theorem initial_water_in_canteen :
  ∃ (initial_water : ℕ), initial_water = 11 :=
by
  let total_leak := time * leak_rate
  let first_6_miles_consumption := 6 * first_6_miles_consumption_rate
  let total_consumption := first_6_miles_consumption + last_mile_consumption
  let initial_water := total_consumption + total_leak + water_rem
  use initial_water
  sorry

end initial_water_in_canteen_l132_132329


namespace imag_part_of_complex_div_l132_132299

theorem imag_part_of_complex_div (i : ℂ) (h : i^2 = -1) : (im ((1 + i) / (2 * i)) = -1/2) :=
  sorry

end imag_part_of_complex_div_l132_132299


namespace certain_event_abs_nonneg_l132_132575

theorem certain_event_abs_nonneg (x : ℝ) : |x| ≥ 0 :=
by
  sorry

end certain_event_abs_nonneg_l132_132575


namespace arithmetic_sequence_formula_geometric_sequence_sum_l132_132326

noncomputable def a (n : ℕ) : ℤ := 3 * n - 2

theorem arithmetic_sequence_formula (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = a 1 + 3) :
  a n = 3 * n - 2 := by
  sorry

def b (n : ℕ) : ℤ := 4 ^ (n - 1)

theorem geometric_sequence_sum (k : ℕ) (h1 : b 1 = 1) (h2 : ∑ i in Finset.range k, b (i + 1) = 85) :
  k = 4 := by
  sorry

end arithmetic_sequence_formula_geometric_sequence_sum_l132_132326


namespace total_games_in_season_is_162_l132_132936

-- Defines the relevant parameters for the sports conference.
def num_teams := 12
def teams_per_division := 6
def intra_division_matches_per_team (teams_per_div : ℕ) := 5 * 3
def inter_division_matches_per_team (other_div_teams : ℕ) := 6 * 2
def total_matches_per_team (intra_div matches intra_division_matches_per_team inter_division_matches_per_team : ℕ) := intra_div + inter_div
def total_matches (num_teams total_matches_per_team : ℕ) := (num_teams * total_matches_per_team) / 2

-- Main theorem to verify the total number of games in the season.
theorem total_games_in_season_is_162 :
  total_matches num_teams (total_matches_per_team (intra_division_matches_per_team teams_per_division) (inter_division_matches_per_team teams_per_division)) = 162 := by
  sorry

end total_games_in_season_is_162_l132_132936


namespace number_of_solutions_l132_132333

def is_solution (x : ℕ) : Prop :=
  (2^x - x^2) % 7 = 0

theorem number_of_solutions : 
  {x : ℕ | x < 10000 ∧ is_solution x}.to_finset.card = 2857 :=
by
  sorry

end number_of_solutions_l132_132333


namespace find_x_value_l132_132008

noncomputable def point_reflect_x_axis (A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (A.1, -A.2, -A.3)

theorem find_x_value (A : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) 
  (hA : A = (-2, 1, 3))
  (hC : C = (x, 0, -2))
  (hBC : dist (point_reflect_x_axis A) C = 3 * real.sqrt 2) :
  x = 2 ∨ x = -6 := by
  sorry

end find_x_value_l132_132008


namespace max_digit_sum_10_occurrences_l132_132082

-- Definitions for the problem conditions
def is_valid_score (a b : ℕ) : Prop :=
  a ≤ 29 ∧ b ≤ 29

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def score_digit_sum_10 (a b : ℕ) : Prop :=
  digit_sum a + digit_sum b = 10

-- Statement of the theorem
theorem max_digit_sum_10_occurrences :
  ∃ T, (∀ a b : ℕ, score_digit_sum_10 a b → is_valid_score a b → T ≤ 5) :=
sorry

end max_digit_sum_10_occurrences_l132_132082


namespace hannah_late_times_l132_132327

variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)
variable (dock_per_late : ℝ)
variable (actual_pay : ℝ)

theorem hannah_late_times (h1 : hourly_rate = 30)
                          (h2 : hours_worked = 18)
                          (h3 : dock_per_late = 5)
                          (h4 : actual_pay = 525) :
  ((hourly_rate * hours_worked - actual_pay) / dock_per_late) = 3 := 
by
  sorry

end hannah_late_times_l132_132327


namespace olympiad_even_group_l132_132213

theorem olympiad_even_group (P : Type) [Fintype P] [Nonempty P] (knows : P → P → Prop)
  (h : ∀ p, (Finset.filter (knows p) Finset.univ).card ≥ 3) :
  ∃ (G : Finset P), G.card > 2 ∧ G.card % 2 = 0 ∧ ∀ p ∈ G, ∃ q₁ q₂ ∈ G, q₁ ≠ p ∧ q₂ ≠ p ∧ knows p q₁ ∧ knows p q₂ :=
by
  sorry

end olympiad_even_group_l132_132213


namespace max_abs_sum_eq_two_l132_132706

theorem max_abs_sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 2) : |x| + |y| ≤ 2 :=
by
  sorry

end max_abs_sum_eq_two_l132_132706


namespace find_x_l132_132402

noncomputable def h (x : ℚ) : ℚ :=
  (5 * ((x - 2) / 3) - 3)

theorem find_x : h (19/2) = 19/2 :=
by
  sorry

end find_x_l132_132402


namespace min_value_expression_l132_132279

-- Define the given problem conditions and statement
theorem min_value_expression :
  ∀ (x y : ℝ), 0 < x → 0 < y → 6 ≤ (y / x) + (16 * x / (2 * x + y)) :=
by
  sorry

end min_value_expression_l132_132279


namespace second_smallest_number_sum_of_78_l132_132871

theorem second_smallest_number_sum_of_78 :
  ∃ (n : ℕ), (n * (n + 1) / 2 = 78) → 
  2 = (∃ (m : ℕ), m = 2 ∧ 1 < m ∧ m ≤ n) := 
sorry

end second_smallest_number_sum_of_78_l132_132871


namespace rectangular_prism_pairs_l132_132153

def total_pairs_of_edges_in_rect_prism_different_dimensions (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : ℕ :=
66

theorem rectangular_prism_pairs (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : total_pairs_of_edges_in_rect_prism_different_dimensions length width height h1 h2 h3 = 66 := 
sorry

end rectangular_prism_pairs_l132_132153


namespace solve_inequalities_solve_fruit_purchase_l132_132908

-- Part 1: Inequalities
theorem solve_inequalities {x : ℝ} : 
  (2 * x < 16) ∧ (3 * x > 2 * x + 3) → (3 < x ∧ x < 8) := by
  sorry

-- Part 2: Fruit Purchase
theorem solve_fruit_purchase {x y : ℝ} : 
  (x + y = 7) ∧ (5 * x + 8 * y = 41) → (x = 5 ∧ y = 2) := by
  sorry

end solve_inequalities_solve_fruit_purchase_l132_132908


namespace equation_of_curve_equation_of_line_AB_minimum_AF_BF_l132_132648

-- (I) Prove the equation of curve C
theorem equation_of_curve :
  ∀ (x y : ℝ), 
    (∃ (M : ℝ → ℝ) (r : ℝ), (M x)^2 + (M y + 1)^2 = r^2 ∧ r = |y - 1| ∧ M 0 = 0 ∧ M (-1) = -1) 
    → x^2 = -4 * y :=
by
  sorry

-- (II) Prove the equation of line AB when P(x₀, y₀) is a fixed point on line l
theorem equation_of_line_AB (l : ℝ → Prop) :
  ∀ (x₀ y₀ : ℝ),
    l x₀ y₀ ∧ (∀ x y, l x y ↔ x - y + 2 = 0) 
    → ∃ (x₁ y₁ x₂ y₂ : ℝ), 
        (y₁ = -1/2 * x₁ ∧ y₂ = -1/2 * x₂) 
        ∧ (x₀ * x₁ + 2 * y₀ + 2 * y₁ = 0) 
        ∧ (x₀ * x₂ + 2 * y₀ + 2 * y₂ = 0) 
        ∧ (x₀ * x + 2 * y + 2 * y₀ = 0) :=
by
  sorry

-- (III) Prove the minimum value of |AF| * |BF|
theorem minimum_AF_BF :
    ∀ (y₀ : ℝ),
    (∀ x₀ y₀, x₀ = y₀ - 2 → ∀ y, y = (1 - y₁) * (1 - y₂) → y₁ + y₂ = -(2 * y₀ + x₀^2) ∧ y₁ * y₂ = y₀^2 ∧ (y₀ = 1/2 → ∃ z, z = 2 * (y₀ - 1/2)^2 + 9/2)) 
    →  (2 * (y₀ - 1/2)^2 + 9/2 = 9/2) :=
by
  sorry

end equation_of_curve_equation_of_line_AB_minimum_AF_BF_l132_132648


namespace number_of_factors_of_M_l132_132763

def M := 57^6 + 6*57^5 + 15*57^4 + 20*57^3 + 15*57^2 + 6*57 + 1

theorem number_of_factors_of_M : (nat.factors M).length = 49 := 
sorry

end number_of_factors_of_M_l132_132763


namespace coefficient_of_term_containing_one_over_x_squared_l132_132668

noncomputable def binomialCoefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_term_containing_one_over_x_squared 
  (x : ℝ) (n : ℕ) (h : n = 5) 
  (sum_of_coeffs : (sum (λ (r : ℕ), binomialCoefficient n r * 4^(n-r)) (range (n+1))) = 243) :
  let expansion_term r := binomialCoefficient n r * (-1)^r * 4^(n-r) * x^(-r/2)
  in expansion_term 4 = 20 :=
by
  sorry

end coefficient_of_term_containing_one_over_x_squared_l132_132668


namespace factor_expression_l132_132232

theorem factor_expression (x : ℝ) : 18 * x^2 + 9 * x - 3 = 3 * (6 * x^2 + 3 * x - 1) :=
by
  sorry

end factor_expression_l132_132232


namespace prime_twentieth_bend_l132_132788

-- Define n-th prime number
noncomputable def nth_prime (n : ℕ) : ℕ :=
  if h : n > 0 then
    Nat.find (λ p, Nat.prime p ∧ ↑n = (Finset.range (p + 1)).filter Nat.prime.card)
  else 2 -- Return 2 for the 0th element just for input validation

-- Define the 20th bend as the 20th prime number
theorem prime_twentieth_bend : nth_prime 20 = 71 := 
  sorry

end prime_twentieth_bend_l132_132788


namespace sum_of_x_values_l132_132504

theorem sum_of_x_values (h₁ : ∀ x, x ≠ -3 → 9 = (x^3 - 3 * x^2 - 9 * x) / (x + 3)) :
  ∑ x in {x | x^2 - 6 * x - 9 = 0}.to_finset = 6 :=
by {
  sorry
}

end sum_of_x_values_l132_132504


namespace ned_trips_l132_132789

theorem ned_trips : 
  ∀ (carry_capacity : ℕ) (table1 : ℕ) (table2 : ℕ) (table3 : ℕ) (table4 : ℕ),
  carry_capacity = 5 →
  table1 = 7 →
  table2 = 10 →
  table3 = 12 →
  table4 = 3 →
  (table1 + table2 + table3 + table4 + carry_capacity - 1) / carry_capacity = 8 :=
by
  intro carry_capacity table1 table2 table3 table4
  intro h1 h2 h3 h4 h5
  sorry

end ned_trips_l132_132789


namespace albert_last_three_digits_l132_132574

theorem albert_last_three_digits :
  ∃ n : ℕ, (AlbertDigits n = 2 ∧ AlbertDigits (n+1) = 0 ∧ AlbertDigits (n+2) = 5)
  ∧ n + 2 = 500 := sorry

end albert_last_three_digits_l132_132574


namespace centroid_of_S_l132_132043

def set_S (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in |x| ≤ y ∧ y ≤ |x| + 3 ∧ y ≤ 4

theorem centroid_of_S : set.centroid set_S = (0, 13/5) :=
sorry

end centroid_of_S_l132_132043


namespace reciprocal_is_correct_l132_132858

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l132_132858


namespace total_trees_planted_l132_132866

noncomputable def total_trees : ℕ :=
let t2 := 15 in
let t3 := t2 + 10 in
let t4 := 40 in
let t5 := 2 * t4 in
let t6 := 3 * t5 - 20 in
let t7 := (t4 + t6) / 2 in
let t8 := 15 + (t2 + t3) in
t2 + t3 + t4 + t5 + t6 + t7 + t8

theorem total_trees_planted : total_trees = 565 :=
by
  sorry

end total_trees_planted_l132_132866


namespace count_negative_numbers_l132_132718

-- Definitions of the expressions given in the problem
def exp1 := -(-2)
def exp2 := -|-7|
def exp3 := -|+1|
def exp4 := |-\frac{2}{3}|
def exp5 := -(+0.8)

-- The Lean statement for the proof problem
theorem count_negative_numbers :
  (exp1 < 0).nat_abs + (exp2 < 0).nat_abs + (exp3 < 0).nat_abs + (exp4 < 0).nat_abs + (exp5 < 0).nat_abs = 3 :=
  sorry

end count_negative_numbers_l132_132718


namespace evaluation_A7_l132_132272

noncomputable section
open_locale classical

def first_order_quotient (A : ℕ → ℕ) (n : ℕ) : ℕ := A (n + 1) / A n

def second_order_quotient (A : ℕ → ℕ) (n : ℕ) : ℕ := first_order_quotient A (n + 1) / first_order_quotient A n

theorem evaluation_A7 (A : ℕ → ℕ) (hA : ∀ n, second_order_quotient A n = 2) (seed : A 0 = 1) (gen : A 1 = 2) :
  A 7 = 2 ^ 21 :=
sorry

end evaluation_A7_l132_132272


namespace growth_rate_equation_l132_132355

-- Given conditions
def revenue_january : ℕ := 36
def revenue_march : ℕ := 48

-- Problem statement
theorem growth_rate_equation (x : ℝ) 
  (h_january : revenue_january = 36)
  (h_march : revenue_march = 48) :
  36 * (1 + x) ^ 2 = 48 :=
sorry

end growth_rate_equation_l132_132355


namespace polygon_diagonals_perimeter_inequality_l132_132409

theorem polygon_diagonals_perimeter_inequality (n : ℕ) (n_gt_3 : n > 3) 
  (d p : ℝ) (h_d : d = ∑ (i j : ℕ) in finset.range n \ finset.Ico i 1 j, length_diagonal i j)
  (h_p : p = ∑ (i : ℕ) in finset.range n, length_side i) :
    (n - 3) / 2 < d / p ∧ d / p < (1/2) * (floor (n/2) * floor ((n+1)/2) - 2) := 
by {
   sorry
}

end polygon_diagonals_perimeter_inequality_l132_132409


namespace concyclic_B_E_C_D_H_l132_132580

theorem concyclic_B_E_C_D_H (A B C H D E : Point)
        (h_orthocenter : Orthocenter H A B C)
        (h_D_on_AC : LiesOn D A C)
        (h_HA_eq_HD : Distance H A = Distance H D)
        (h_parallelogram_ABEH : Parallelogram A B E H) :
    Concyclic B E C D H :=
begin
  sorry,
end

end concyclic_B_E_C_D_H_l132_132580


namespace prove_BP_squared_max_l132_132032

noncomputable def circle_diameter_proof : Prop :=
  let AB := 20 in
  ∀ (ω : Type) [circle ω] (A B C P T : ω) (hAB : diameter AB)
    (hCT_tangent : tangent (C :: T))
    (hP_foot : foot_perpendicular A (C :: T)),
    ∃ m : ℝ, maximum_length BP and
    ∃ BP : ℝ, BP^2 = 425

theorem prove_BP_squared_max : circle_diameter_proof :=
  by sorry

end prove_BP_squared_max_l132_132032


namespace Chris_age_proof_l132_132582

theorem Chris_age_proof (m c : ℕ) (h1 : c = 3 * m - 22) (h2 : c + m = 70) : c = 47 := by
  sorry

end Chris_age_proof_l132_132582


namespace correct_mean_l132_132110

variable (values : Fin 100 → ℕ)
variable (incorrect_values : Fin 3 → ℕ)
variable (correct_values : Fin 3 → ℕ)

theorem correct_mean (h1 : ∑ i, values i = 23500)
  (h2 : incorrect_values 0 = 170)
  (h3 : incorrect_values 1 = 195)
  (h4 : incorrect_values 2 = 130)
  (h5 : correct_values 0 = 190)
  (h6 : correct_values 1 = 215)
  (h7 : correct_values 2 = 150) :
  (23500 - ∑ i, incorrect_values i + ∑ i, correct_values i) / 100 = 235.60 := 
by 
  sorry

end correct_mean_l132_132110


namespace reciprocal_is_correct_l132_132861

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l132_132861


namespace angle_ALK_is_acute_l132_132356

theorem angle_ALK_is_acute
  (A B C D K P L: Point)
  (h_cyclic: is_cyclic_quadrilateral A B C D)
  (h_AD_gt_BC: dist A D > dist B C)
  (h_C_D_shorter_arc_AB: on_shorter_arc C D A B)
  (h_AD_BC_inter_K: ∃ (K: Point), line_through A D ∧ line_through B C ∧ AD_meet_BC_at_K)
  (h_AC_BD_inter_P: ∃ (P: Point), line_through A C ∧ line_through B D ∧ AC_meet_BD_at_P)
  (h_KP_inter_AB_at_L: ∃ (L: Point), line_through K P ∧ KP_meet_AB_at_L):
  angle A L K < 90 :=
sorry

end angle_ALK_is_acute_l132_132356


namespace sampling_interval_l132_132145

theorem sampling_interval (N n : ℕ) (hN : N = 1003) (hn : n = 50) : (N / n).floor = 20 := by
  sorry

end sampling_interval_l132_132145


namespace arithmetic_sequence_properties_l132_132732

def a_n (a_1 d n : ℤ) : ℤ := a_1 + (n - 1) * d

def S_n (a_1 d n : ℤ) : ℤ := n * (a_1 + a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_properties :
  ∀ (a_1 d : ℤ), 
    a_n a_1 d 10 = 18 → 
    S_n a_1 d 5 = -15 → 
    (∀ n, a_n a_1 d n = 3 * n - 12) ∧ 
    (∀ n, S_n a_1 d n = (3 * n^2 - 21 * n) / 2) ∧ 
    S_n (-9 : ℤ) 3 min_S_n = -18 :=
  by
    intros a_1 d h1 h2,
    sorry

end arithmetic_sequence_properties_l132_132732


namespace sum_of_first_20_primes_l132_132590

theorem sum_of_first_20_primes :
  ( [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71].sum = 639 ) :=
by
  sorry

end sum_of_first_20_primes_l132_132590


namespace roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l132_132592

-- Part (a)
theorem roots_can_be_integers_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x y : ℤ, x * y = q ∧ x + y = p) ∧ (∃ x y : ℤ, x * y = q ∧ x + y = p + 1) :=
sorry

-- Part (b)
theorem roots_cannot_both_be_integers_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬(∃ x y z w : ℤ, x * y = q ∧ x + y = p ∧ z * w = q ∧ z + w = p + 1) :=
sorry

end roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l132_132592


namespace problem_a_plus_b_l132_132295

theorem problem_a_plus_b 
  (h2: ∀ {a b : ℝ}, √(2 + a / b) = 2 * √(a / b) → a = 2 ∧ b = 3)
  (h3: ∀ {a b : ℝ}, √(3 + a / b) = 3 * √(a / b) → a = 3 ∧ b = 8)
  (h4: ∀ {a b : ℝ}, √(4 + a / b) = 4 * √(a / b) → a = 4 ∧ b = 15)
  (a b : ℝ) 
  (h7: √(7 + a / b) = 7 * √(a / b)) :
  a + b = 55 :=
sorry

end problem_a_plus_b_l132_132295


namespace distribution_and_expectation_of_females_probability_of_selecting_B_given_A_l132_132075

-- Definitions based on the conditions
def total_students : ℕ := 6  -- 6 student council leaders
def total_males : ℕ := 4  -- 4 males
def total_females : ℕ := 2  -- 2 females
def selected : ℕ := 3  -- selecting 3 members

-- (I) Proving the distribution and expectation of the number of females selected
theorem distribution_and_expectation_of_females :
  ∃ ξ : ℕ → ℝ, (ξ 0 = 1 / 5) ∧ (ξ 1 = 3 / 5) ∧ (ξ 2 = 1 / 5) ∧ (∀ x, x > 2 → ξ x = 0) ∧ 
  (∑ i in {0, 1, 2}, i * ξ i = 1) :=
sorry

-- (II) Proving the probability of selecting female student B given male student A is selected
theorem probability_of_selecting_B_given_A :
  let P_C := (4 : ℝ) / 10 in
  P_C = 2 / 5 :=
sorry

end distribution_and_expectation_of_females_probability_of_selecting_B_given_A_l132_132075


namespace operation_result_l132_132501

theorem operation_result (x : ℕ) (h : x = 3) : 60 + 5 * 12 / (180 / x) = 61 :=
by
  rw h
  sorry

end operation_result_l132_132501


namespace eval_cube_root_sum_of_fractions_l132_132248

theorem eval_cube_root_sum_of_fractions :
  (∛((9 / 16 : ℚ) + (25 / 36 : ℚ) + (4 / 9 : ℚ)) = (∛245) / 12) :=
by sorry

end eval_cube_root_sum_of_fractions_l132_132248


namespace worker_and_robot_capacity_additional_workers_needed_l132_132367

-- Definitions and conditions
def worker_capacity (x : ℕ) : Prop :=
  (1 : ℕ) * x + 420 = 420 + x

def time_equivalence (x : ℕ) : Prop :=
  900 * 10 * x = 600 * (x + 420)

-- First part of the proof problem
theorem worker_and_robot_capacity (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  x = 30 ∧ x + 420 = 450 :=
by
  sorry

-- Second part of the proof problem
theorem additional_workers_needed (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  3 * (x + 420) * 2 < 3600 →
  2 * 30 * 15 ≥ 3600 - 2 * 3 * (x + 420) :=
by
  sorry

end worker_and_robot_capacity_additional_workers_needed_l132_132367


namespace royalty_amount_l132_132115

theorem royalty_amount (x : ℝ) (h1 : x > 800) (h2 : x ≤ 4000) (h3 : (x - 800) * 0.14 = 420) :
  x = 3800 :=
by
  sorry

end royalty_amount_l132_132115


namespace find_k_if_lines_parallel_l132_132976

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end find_k_if_lines_parallel_l132_132976


namespace snail_returns_at_integer_hours_l132_132200

def constant_speed_crawl (x : ℕ) : Prop :=
  -- The snail crawls at a constant speed along x direction

def right_angle_turn_every_15_minutes (t : ℕ) : Prop :=
  -- The snail turns at a right angle every 15 minutes

theorem snail_returns_at_integer_hours (n : ℕ) (h1 : constant_speed_crawl n) (h2 : right_angle_turn_every_15_minutes n) : ∃ k : ℕ, n = 4 * k :=
begin
  sorry
end

end snail_returns_at_integer_hours_l132_132200


namespace projectile_time_l132_132934

theorem projectile_time : ∃ t : ℝ, (60 - 8 * t - 5 * t^2 = 30) ∧ t = 1.773 := by
  sorry

end projectile_time_l132_132934


namespace find_X_l132_132914

theorem find_X (X : ℕ) : 
  (∃ k : ℕ, X = 26 * k + k) ∧ (∃ m : ℕ, X = 29 * m + m) → (X = 270 ∨ X = 540) :=
by
  sorry

end find_X_l132_132914


namespace find_xyz_l132_132413

variable (x y z : ℝ)
variable (h1 : x = 80 + 0.11 * 80)
variable (h2 : y = 120 - 0.15 * 120)
variable (h3 : z = 0.20 * (0.40 * (x + y)) + 0.40 * (x + y))

theorem find_xyz (hx : x = 88.8) (hy : y = 102) (hz : z = 91.584) : 
  x = 88.8 ∧ y = 102 ∧ z = 91.584 := by
  sorry

end find_xyz_l132_132413


namespace parallel_lines_slope_l132_132970

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end parallel_lines_slope_l132_132970


namespace value_of_z_l132_132779

-- Conditions stated as definitions in Lean 4
def x_plus_y_eq : Prop := x + y = 90
def x_minus_y_eq : Prop := x - y = 15
def x_greater_y : Prop := x > y

-- Proving that x^2 - y^2 = 1350 given the conditions
theorem value_of_z (x y : ℝ) (h1 : x_plus_y_eq) (h2 : x_minus_y_eq) (h3 : x_greater_y) : x^2 - y^2 = 1350 := by
    sorry

end value_of_z_l132_132779


namespace menelaus_theorem_l132_132884

open set

variables {A B C P Q R : Point}

-- Assume A, B, C are points forming a triangle
axiom triangle_ABC : ¬ collinear A B C

-- Assume P is on line segment BC, Q is on AC, R is on AB
axiom P_on_BC : segment B C P
axiom Q_on_AC : segment A C Q
axiom R_on_AB : segment A B R

-- Menelaus' theorem statement
theorem menelaus_theorem (collinear_PQR : collinear P Q R) :
  (BP / PC) * (CQ / QA) * (AR / RB) = -1 :=
sorry

end menelaus_theorem_l132_132884


namespace triangle_angle_C_sin_A_l132_132349

theorem triangle_angle_C (a b c A B C : ℝ) (h1 : b * sin (2 * C) = c * sin B) (h2 : C > 0) (h3 : C < real.pi) :
  C = real.pi / 3 := 
sorry

theorem sin_A (a b c A B C : ℝ) (h1 : b * sin (2 * C) = c * sin B) (h2 : C > 0) (h3 : C < real.pi) (h4 : C = real.pi / 3) (h5 : sin (B - real.pi / 3) = 3 / 5) (h6 : A + B + C = real.pi) :
  sin A = (4 * real.sqrt 3 - 3) / 10 := 
sorry

end triangle_angle_C_sin_A_l132_132349


namespace sum_of_coeffs_l132_132131

theorem sum_of_coeffs (x y : ℕ) : 
  let expr := (2 * x^2 - 3 * x * y + y^2) ^ 6 in
  (x = 1) ∧ (y = 1) → 
  sum_of_coeffs expr = 0 :=
by
  intros
  sorry

end sum_of_coeffs_l132_132131


namespace system_of_equations_l132_132569

-- Definitions and hypotheses based on the problem conditions
variables (x y : ℕ)
def group_of_five_eq := x = 5 * y + 3
def group_of_six_eq := x = 6 * y - 3

-- Final statement to prove the system of equations
theorem system_of_equations :
  (x = 5 * y + 3) → (x = 6 * y - 3) → (5 * y = x - 3 ∧ 6 * y = x + 3) :=
by 
  intros h1 h2
  split;
  linarith

end system_of_equations_l132_132569


namespace tangent_line_eq_height_l132_132773

variables {A B C O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variables {h : ℝ} {K : Type} [MetricSpace K]

noncomputable def is_tangent (A B C : K -> Prop) : Prop :=
  ∀ (x : A), is_tangent A (sides_of_eq_triangle K x)

noncomputable def equilateral_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def height_of (T : A -> Prop) (h : ℝ) : Prop :=
  height_of T = h

def length_eq_height (OC : ℝ) (h : ℝ) := OC = h

theorem tangent_line_eq_height {A B C O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
  {h : ℝ} {K : Type} [MetricSpace K]
  (T : A -> Prop) (is_tangent T K) (equilateral_triangle A B C)
  (ht : height_of T h):
  length_eq_height OC h :=
sorry

end tangent_line_eq_height_l132_132773


namespace ratio_of_volumes_l132_132146

-- Conditions of the problem
def has_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def triangle_sides := (8 : ℝ, 15 : ℝ, 17 : ℝ)
def sphere_radius := (5 : ℝ)

-- Theorem to prove the ratio of the volumes
theorem ratio_of_volumes (a b c R : ℝ) (h_right_triangle : has_right_triangle a b c) :
  R = 5 → a = 8 → b = 15 → c = 17 → 
  let V1 := (1 * (3 * (3^2) + (1^2)) * Real.pi) / 6,
      V := (4 * Real.pi * (R^3)) / 3,
      V2 := V - V1
  in V1 / V2 = 7 / 243 :=
by
  sorry

end ratio_of_volumes_l132_132146


namespace range_of_a_l132_132649

theorem range_of_a (a : ℝ) (h1 : a > 0)
  (h2 : ∃ x : ℝ, abs (Real.sin x) > a)
  (h3 : ∀ x : ℝ, x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → (Real.sin x)^2 + a * Real.sin x - 1 ≥ 0) :
  a ∈ Set.Ico (Real.sqrt 2 / 2) 1 :=
sorry

end range_of_a_l132_132649


namespace number_of_terms_in_binomial_expansion_l132_132847

theorem number_of_terms_in_binomial_expansion (a b : ℝ) (n : ℕ) :
  (number_of_terms (a + b)^(2 * n) = 2 * n + 1) :=
sorry

end number_of_terms_in_binomial_expansion_l132_132847


namespace fuel_tank_capacity_l132_132577

def ethanol_content_fuel_A (fuel_A : ℝ) : ℝ := 0.12 * fuel_A
def ethanol_content_fuel_B (fuel_B : ℝ) : ℝ := 0.16 * fuel_B

theorem fuel_tank_capacity (C : ℝ) :
  ethanol_content_fuel_A 122 + ethanol_content_fuel_B (C - 122) = 30 → C = 218 :=
by
  sorry

end fuel_tank_capacity_l132_132577


namespace minimum_value_of_sum_l132_132294

open Real

theorem minimum_value_of_sum {a b : ℝ} (h₁ : 0 < a) (h₂ : 0 < b) (h : log a / log 2 + log b / log 2 ≥ 6) :
  a + b ≥ 16 :=
sorry

end minimum_value_of_sum_l132_132294


namespace coefficients_sum_correct_l132_132644

noncomputable def poly_expr (x : ℝ) : ℝ := (x + 2)^4

def coefficients_sum (a a_1 a_2 a_3 a_4 : ℝ) : ℝ :=
  a_1 + a_2 + a_3 + a_4

theorem coefficients_sum_correct (a a_1 a_2 a_3 a_4 : ℝ) :
  poly_expr 1 = a_4 * 1 ^ 4 + a_3 * 1 ^ 3 + a_2 * 1 ^ 2 + a_1 * 1 + a →
  a = 16 → coefficients_sum a a_1 a_2 a_3 a_4 = 65 :=
by
  intro h₁ h₂
  sorry

end coefficients_sum_correct_l132_132644


namespace volume_of_tetrahedron_circumscribed_sphere_l132_132378

noncomputable def volume_of_circumscribed_sphere (A B C D : Point) (r : ℝ) :=
  4/3 * Real.pi * r^3

theorem volume_of_tetrahedron_circumscribed_sphere (A B C D : Point) 
  (h_angle_ABC : angle A B C = Real.pi / 2)
  (h_angle_BAD : angle B A D = Real.pi / 2)
  (h_AB : dist A B = 1)
  (h_CD : dist C D = 2)
  (h_angle_AD_BC : angle A D B C = Real.pi / 6) :
  volume_of_circumscribed_sphere A B C D (Real.sqrt (13) / 2) = 13 * Real.sqrt (13) / 6 * Real.pi :=
by
  sorry

end volume_of_tetrahedron_circumscribed_sphere_l132_132378


namespace count_whole_numbers_in_interval_l132_132697

theorem count_whole_numbers_in_interval : 
  ∃ n : ℕ, n = 5 ∧ (∀ x : ℕ, x ∈ set.Ioo (5/3 : ℝ) (2 * Real.pi) → 2 ≤ x ∧ x ≤ 6) :=
begin
  sorry
end

end count_whole_numbers_in_interval_l132_132697


namespace hamburgers_total_l132_132935

theorem hamburgers_total (initial_hamburgers : ℝ) (additional_hamburgers : ℝ) (h₁ : initial_hamburgers = 9.0) (h₂ : additional_hamburgers = 3.0) : initial_hamburgers + additional_hamburgers = 12.0 :=
by
  rw [h₁, h₂]
  norm_num

end hamburgers_total_l132_132935


namespace problem_2006_sum_l132_132389

noncomputable def largest_sum_of_integers (I T E S : ℕ) : ℕ :=
  2 * T + (I + E + S) + 2006

theorem problem_2006_sum :
  ∃ I T E S : ℕ, I ≠ T ∧ I ≠ E ∧ I ≠ S ∧ T ≠ E ∧ T ≠ S ∧ E ≠ S ∧
  I * T * E * S = 2006 ∧ largest_sum_of_integers I T E S = 2086 :=
by {
  -- Example instantiation of the variables satisfying the conditions
  use [2, 1, 17, 59],
  -- Distinctness conditions
  repeat {split},
  -- Conditions that they have to be distinct
  any_goals {dec_trivial},
  -- Verification of product condition
  exact dec_trivial,
  -- Calculation of sum condition
  exact dec_trivial
}

end problem_2006_sum_l132_132389


namespace arithmetic_sequence_product_l132_132040

theorem arithmetic_sequence_product (b : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ n, b (n + 1) > b n)
  (h2 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l132_132040


namespace volleyball_team_choices_l132_132062

theorem volleyball_team_choices 
  (players : Finset ℕ) 
  (triplets : Finset ℕ) 
  (h1 : players.card = 15) 
  (h2 : triplets.card = 3)
  (triplets_subset : triplets ⊆ players) :
  (players.choose 6).card - ((players \ triplets).choose 6).card = 4081 := 
by 
  sorry

end volleyball_team_choices_l132_132062


namespace derivative_at_neg_one_l132_132701

-- Define the function f
def f (x : ℝ) : ℝ := (1/3) * x^3 + 2 * x + 1

-- State the theorem that f'(-1) = 3
theorem derivative_at_neg_one : deriv f (-1) = 3 := by
  sorry

end derivative_at_neg_one_l132_132701


namespace max_sum_black_white_table_l132_132721

theorem max_sum_black_white_table (n : ℕ) (h : n = 7) : 
  ∃ (table : (Fin n) → (Fin n) → Bool), 
  let cell_number (i j : Fin n) (tbl : (Fin n) → (Fin n) → Bool) := 
    (if tbl i j then 0 else (∑ x, if tbl x j then 1 else 0) + (∑ y, if tbl i y then 1 else 0) - (if tbl i j then 1 else 0)) in
  ∑ i, ∑ j, cell_number i j table = 168 :=
by
  let table := fun (i j : Fin n) => (i + j) % 2 = 0
  use table
  sorry

end max_sum_black_white_table_l132_132721


namespace similar_triangles_third_vertices_on_circle_l132_132316

theorem similar_triangles_third_vertices_on_circle
  (A B C₁ C₂ C₃ C₄ C₅ C₆ : Point)
  (h_similar : ∀ i ∈ {1, 2, 3, 4, 5, 6},
    ∃ α β γ : ℝ,
      ∠ C_i A B = α ∧
      ∠ A B C_i = β ∧
      ∠ B C_i A = γ ∧
      (α + β + γ = 180) ∧
      similar_triangle A B C_i) : ∃ (O : Point) (r : ℝ), 
        ∀ i ∈ {C₁, C₂, C₃, C₄, C₅, C₆}, dist O i = r :=
sorry

end similar_triangles_third_vertices_on_circle_l132_132316


namespace point_value_of_other_questions_l132_132513

theorem point_value_of_other_questions (x y p : ℕ) 
  (h1 : x = 10) 
  (h2 : x + y = 40) 
  (h3 : 40 + 30 * p = 100) : 
  p = 2 := 
  sorry

end point_value_of_other_questions_l132_132513


namespace prob_two_out_of_three_l132_132873

open ProbabilityTheory MeasureTheory

noncomputable def dist_weight_cow : ℕ → ℝ → ℝ → prob := sorry
noncomputable def normal_prob_2_of_3 (μ σ : ℝ) (p k n : ℝ) : ℝ := 
  ∑ y in (finset.Icc 0 2), (Nat.choose n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_two_out_of_three (μ σ : ℝ) (prob_interval : ℝ) : 
  μ = 470 ∧ σ = 30 ∧ prob_interval = (cdf (NormalDistribution.mk μ σ^2) 530 - cdf (NormalDistribution.mk μ σ^2) 470) / ∑ y in (finset.Icc 0 2), (Nat.choose 3 2 * ((cdf (NormalDistribution.mk μ σ^2) 530 - cdf (NormalDistribution.mk μ σ^2) 470) ^ 2) * ((1 - (cdf (NormalDistribution.mk μ σ^2) 530 - cdf (NormalDistribution.mk μ σ^2) 470)) ^ 1)) :
  normal_prob_2_of_3 μ σ 2 3 = 0.357 := sorry

end prob_two_out_of_three_l132_132873


namespace sum_of_squares_of_coefficients_l132_132815

def poly1 (x : ℝ) : ℝ := 3 * (x^2 - x + 3)
def poly2 (x : ℝ) : ℝ := -6 * (x^3 - 4 * x + 2)

def simplified_poly (x : ℝ) : ℝ := poly1 x + poly2 x

theorem sum_of_squares_of_coefficients : 
  let c := [-6, 3, 21, -3] in
  (c.map (λ y => y^2)).sum = 495 := by
  sorry

end sum_of_squares_of_coefficients_l132_132815


namespace worker_and_robot_capacity_additional_workers_needed_l132_132368

-- Definitions and conditions
def worker_capacity (x : ℕ) : Prop :=
  (1 : ℕ) * x + 420 = 420 + x

def time_equivalence (x : ℕ) : Prop :=
  900 * 10 * x = 600 * (x + 420)

-- First part of the proof problem
theorem worker_and_robot_capacity (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  x = 30 ∧ x + 420 = 450 :=
by
  sorry

-- Second part of the proof problem
theorem additional_workers_needed (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  3 * (x + 420) * 2 < 3600 →
  2 * 30 * 15 ≥ 3600 - 2 * 3 * (x + 420) :=
by
  sorry

end worker_and_robot_capacity_additional_workers_needed_l132_132368


namespace opposite_of_four_l132_132849

theorem opposite_of_four : ∃ x : ℤ, 4 + x = 0 ∧ x = -4 :=
by
  use -4
  split
  · exact rfl
  · rfl
  sorry

end opposite_of_four_l132_132849


namespace wickets_in_last_match_l132_132190

def avg_before : ℝ := 12.4
def runs_last : ℝ := 26
def avg_after : ℝ := 12
def wickets_before : ℕ := 55

theorem wickets_in_last_match :
  let wkts_last := 4 in
  (avg_before * wickets_before + runs_last) = avg_after * (wickets_before + wkts_last) :=
by
  sorry

end wickets_in_last_match_l132_132190


namespace trigonometric_identity_l132_132446

theorem trigonometric_identity 
  (A : ℝ) 
  (h_cot : Real.cot A = Real.cos A / Real.sin A) 
  (h_csc : Real.csc A = 1 / Real.sin A) 
  (h_tan : Real.tan A = Real.sin A / Real.cos A) 
  (h_sec : Real.sec A = 1 / Real.cos A) :
  (1 + Real.cot A + Real.csc A) * (1 + Real.tan A - Real.sec A) = 3 - 1 / (Real.sin A * Real.cos A) :=
by
  sorry

end trigonometric_identity_l132_132446


namespace parallel_lines_slope_l132_132972

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end parallel_lines_slope_l132_132972


namespace PA_in_equilateral_triangle_l132_132023

theorem PA_in_equilateral_triangle (A B C P : Point)
  (h_eq_triangle : equilateral_triangle A B C)
  (h_on_BC : P ∈ segment B C)
  (h_PB : dist P B = 50)
  (h_PC : dist P C = 30) :
  dist P A = 70 := by
  sorry

end PA_in_equilateral_triangle_l132_132023


namespace sin_cos_sum_eq_l132_132699

variable (theta a b : ℝ)

def is_acute (theta : ℝ) : Prop := 0 < theta ∧ theta < π / 2
def sin_double_angle (theta a : ℝ) : Prop := Real.sin (2 * theta) = a
def cos_double_angle (theta b : ℝ) : Prop := Real.cos (2 * theta) = b

theorem sin_cos_sum_eq (h1 : is_acute theta)
                       (h2 : sin_double_angle theta a)
                       (h3 : cos_double_angle theta b) :
  Real.sin theta + Real.cos theta = Real.sqrt ((1 + b) / 2) + Real.sqrt ((1 - b) / 2) := sorry

end sin_cos_sum_eq_l132_132699


namespace question_1_question_2_question_3_l132_132671

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 2 * b

-- Question 1
theorem question_1 (a b : ℝ) (h : a = b) (ha : a > 0) :
  ∀ x : ℝ, (f a b x < 0) ↔ (-2 < x ∧ x < 1) :=
sorry

-- Question 2
theorem question_2 (b : ℝ) :
  (∀ x : ℝ, x < 2 → (f 1 b x ≥ 1)) → (b ≤ 2 * Real.sqrt 3 - 4) :=
sorry

-- Question 3
theorem question_3 (a b : ℝ) (h1 : |f a b (-1)| ≤ 1) (h2 : |f a b 1| ≤ 3) :
  (5 / 3 ≤ |a| + |b + 2| ∧ |a| + |b + 2| ≤ 9) :=
sorry

end question_1_question_2_question_3_l132_132671


namespace triangle_CF_length_l132_132372

theorem triangle_CF_length
  (AF : ℝ) (BF : ℝ) (CF : ℝ)
  (h1 : AF = 36)
  (h2 : BF = AF)
  (h3 : ∀ (A B F : ℝ), ∠ABF = 90)
  (h4 : ∀ (B C F : ℝ), ∠BCF = 90)
  :
  CF = 36 := 
sorry

end triangle_CF_length_l132_132372


namespace triangle_perimeter_l132_132937

-- Given conditions
variables (z x : ℝ) (h_nonneg_z : 0 ≤ z) (h_nonneg_x : 0 ≤ x)

-- Proof goal
theorem triangle_perimeter (h : ∀ z x, ∃ x, ∀ z, ∃ z, x * x + sqrt (z^2 - x^2) = z) : 
  x = z/2 →
  perimeter (triangle ((z-x)/2) ((z+x)/2) (z/2)) = 3 * z / 2 :=
by sorry

end triangle_perimeter_l132_132937


namespace number_of_counterexamples_l132_132262

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the condition on the digits
def sum_of_digits_eq_5 (n : ℕ) : Prop :=
  (n.digits 10).sum = 5

-- Define the condition that none of the digits is zero
def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

-- Define the combined condition
def valid_number (n : ℕ) : Prop :=
  sum_of_digits_eq_5 n ∧ no_zero_digits n

-- Define the counterexample condition
def is_counterexample (n : ℕ) : Prop :=
  valid_number n ∧ ¬ is_prime n

-- State the theorem
theorem number_of_counterexamples : finset.card { n : ℕ | is_counterexample n } = 9 := 
sorry

end number_of_counterexamples_l132_132262


namespace clothes_add_percentage_l132_132899

theorem clothes_add_percentage (W : ℝ) (C : ℝ) (h1 : W > 0) 
  (h2 : C = 0.0174 * W) : 
  ((C / (0.87 * W)) * 100) = 2 :=
by
  sorry

end clothes_add_percentage_l132_132899


namespace james_older_brother_age_l132_132743

def johnAge : ℕ := 39

def ageCondition (johnAge : ℕ) (jamesAgeIn6 : ℕ) : Prop :=
  johnAge - 3 = 2 * jamesAgeIn6

def jamesOlderBrother (james : ℕ) : ℕ :=
  james + 4

theorem james_older_brother_age (johnAge jamesOlderBrotherAge : ℕ) (james : ℕ) :
  johnAge = 39 →
  (johnAge - 3 = 2 * (james + 6)) →
  jamesOlderBrotherAge = jamesOlderBrother james →
  jamesOlderBrotherAge = 16 :=
by
  sorry

end james_older_brother_age_l132_132743


namespace average_speed_round_trip_36_l132_132558

variables (z : ℝ)

def eastward_speed_minutes_per_mile : ℝ := 3
def westward_speed_miles_per_minute : ℝ := 3

def total_distance (z : ℝ) : ℝ := 2 * z
def eastward_time (z : ℝ) (eastward_speed : ℝ) : ℝ := z * eastward_speed
def westward_time (z : ℝ) (westward_speed : ℝ) : ℝ := z / westward_speed
def total_time (z : ℝ) (eastward_speed : ℝ) (westward_speed : ℝ) : ℝ := eastward_time z eastward_speed + westward_time z westward_speed
def total_time_in_hours (total_time : ℝ) : ℝ := total_time / 60

def average_speed (total_distance : ℝ) (total_time_in_hours : ℝ) : ℝ := total_distance / total_time_in_hours

theorem average_speed_round_trip_36 (z : ℝ) :
  average_speed (total_distance z) (total_time_in_hours (total_time z eastward_speed_minutes_per_mile westward_speed_miles_per_minute)) = 36 := 
  sorry

end average_speed_round_trip_36_l132_132558


namespace tom_driving_speed_l132_132754

theorem tom_driving_speed
  (v : ℝ)
  (hKarenSpeed : 60 = 60) -- Karen drives at an average speed of 60 mph
  (hKarenLateStart: 4 / 60 = 1 / 15) -- Karen starts 4 minutes late, which is 1/15 hours
  (hTomDistance : 24 = 24) -- Tom drives 24 miles before Karen wins the bet
  (hTimeEquation: 24 / v = 8 / 15): -- The equation derived from given conditions
  v = 45 := 
by
  sorry

end tom_driving_speed_l132_132754


namespace estimate_mass_of_ice_floe_l132_132797

noncomputable def mass_of_ice_floe (d : ℝ) (D : ℝ) (m : ℝ) : ℝ :=
  (m * d) / (D - d)

theorem estimate_mass_of_ice_floe :
  mass_of_ice_floe 9.5 10 600 = 11400 := 
by
  sorry

end estimate_mass_of_ice_floe_l132_132797


namespace cube_root_harmonic_mean_l132_132588

def harmonic_mean (a b : ℝ) : ℝ :=
  1 / (0.5 * (1 / a + 1 / b))

def a : ℝ := (2016 + 2015) / (2016^2 + 2016 * 2015 + 2015^2)

def b : ℝ := (2016 - 2015) / (2016^2 - 2016 * 2015 + 2015^2)

def x : ℝ := harmonic_mean a b

theorem cube_root_harmonic_mean :
  (x = harmonic_mean a b) →
  real.cbrt (x / (2015 + 2016)) = 1 / 2016 :=
by
  intros h
  rw h
  sorry

end cube_root_harmonic_mean_l132_132588


namespace find_C_l132_132271

open polynomial

noncomputable def poly : polynomial ℤ := X^6 - 21 * X^5 + (A : ℤ) * X^4 + (B : ℤ) * X^3 + (C : ℤ) * X^2 + (D : ℤ) * X + 36

theorem find_C 
    (roots_pos : ∀ (r : ℤ), r ∈ (mv_polynomial.roots poly.to_mv_polynomial).to_finset → r > 0)
    (roots_sum21 : (mv_polynomial.roots poly.to_mv_polynomial).to_finset.sum id = 21) :
    C = -180 :=
sorry

end find_C_l132_132271


namespace inequality_proof_l132_132637

open Real

theorem inequality_proof (n : ℕ) (h_pos : 0 < n) (a : Fin n → ℝ) (h_a_pos : ∀ i, 0 < a i) :
  let S := (Finset.sum (Finset.univ : Finset (Fin n)) a)
  in (Finset.sum (Finset.range (n - 1)) (λ i, sqrt (a i ^ 2 + a i * a (i + 1) + a (i + 1) ^ 2))) 
     ≥ sqrt ((S - a 0) ^ 2 + (S - a 0) * (S - a (n-1)) + (S - a (n-1)) ^ 2) :=
by
  sorry

end inequality_proof_l132_132637


namespace find_second_term_l132_132099

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end find_second_term_l132_132099


namespace triangle_perimeter_problem_l132_132843

theorem triangle_perimeter_problem : 
  ∀ (c : ℝ), 20 + 15 > c ∧ 20 + c > 15 ∧ 15 + c > 20 → ¬ (35 + c = 72) :=
by
  intros c h
  sorry

end triangle_perimeter_problem_l132_132843


namespace final_result_l132_132556

theorem final_result (x : ℝ) (hx : 1.40 * x = 1680) : (0.80 * x) * 1.15 = 1104 :=
by
  have h1 : x = 1680 / 1.40 := by
    field_simp [hx]
  rw h1 at *
  field_simp
  exact (by norm_num : 1200 * 0.8 * 1.15 = 1104)
  sorry

end final_result_l132_132556


namespace vehicles_with_cd_player_but_no_pw_or_ab_l132_132518

-- Definitions based on conditions from step a)
def P : ℝ := 0.60 -- percentage of vehicles with power windows
def A : ℝ := 0.25 -- percentage of vehicles with anti-lock brakes
def C : ℝ := 0.75 -- percentage of vehicles with a CD player
def PA : ℝ := 0.10 -- percentage of vehicles with both power windows and anti-lock brakes
def AC : ℝ := 0.15 -- percentage of vehicles with both anti-lock brakes and a CD player
def PC : ℝ := 0.22 -- percentage of vehicles with both power windows and a CD player
def PAC : ℝ := 0.00 -- no vehicle has all 3 features

-- The statement we want to prove
theorem vehicles_with_cd_player_but_no_pw_or_ab : C - (PC + AC) = 0.38 := by
  sorry

end vehicles_with_cd_player_but_no_pw_or_ab_l132_132518


namespace lunch_break_duration_l132_132799

theorem lunch_break_duration
  (p h L : ℝ)
  (monday_eq : (9 - L) * (p + h) = 0.4)
  (tuesday_eq : (8 - L) * h = 0.33)
  (wednesday_eq : (12 - L) * p = 0.27) :
  L = 7.0 ∨ L * 60 = 420 :=
by
  sorry

end lunch_break_duration_l132_132799


namespace arithmetic_series_sum_l132_132105

theorem arithmetic_series_sum (k : ℕ) : 
    let a₁ := 3 * k^2 + 2
    let d := 2
    let n := 4 * k + 3
    let a_n := a₁ + (n - 1) * d
    let S_n := n / 2 * (a₁ + a_n)
in S_n = 12 * k^3 + 28 * k^2 + 28 * k + 12 :=
by
  let a₁ := 3 * k^2 + 2
  let d := 2
  let n := 4 * k + 3
  let a_n := a₁ + (n - 1) * d
  let S_n := n / 2 * (a₁ + a_n)
  exact sorry

end arithmetic_series_sum_l132_132105


namespace length_major_axis_l132_132461

-- Define the elliptical equation as a condition
def ellipse_eq (x y : ℝ) : Prop :=
  2 * x^2 + y^2 = 8

-- Prove the length of the major axis given the ellipse equation
theorem length_major_axis (x y : ℝ) (h : ellipse_eq x y) : ∃ a : ℝ, 2 * a = 4 * real.sqrt 2 :=
by
  use (2 * real.sqrt 2)
  sorry

end length_major_axis_l132_132461


namespace sodium_bicarbonate_moles_needed_l132_132332

-- Definitions for the problem.
def balanced_reaction : Prop := 
  ∀ (NaHCO₃ HCl NaCl H₂O CO₂ : Type) (moles_NaHCO₃ moles_HCl moles_NaCl moles_H₂O moles_CO₂ : Nat),
  (moles_NaHCO₃ = moles_HCl) → 
  (moles_NaCl = moles_HCl) → 
  (moles_H₂O = moles_HCl) → 
  (moles_CO₂ = moles_HCl)

-- Given condition: 3 moles of HCl
def moles_HCl : Nat := 3

-- The theorem statement
theorem sodium_bicarbonate_moles_needed : 
  balanced_reaction → moles_HCl = 3 → ∃ moles_NaHCO₃, moles_NaHCO₃ = 3 :=
by 
  -- Proof will be provided here.
  sorry

end sodium_bicarbonate_moles_needed_l132_132332


namespace find_two_digit_integers_l132_132828

theorem find_two_digit_integers :
  ∃ (m n : ℕ), 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 ∧
    (∃ (a b : ℚ), a = m ∧ b = n ∧ (a + b) / 2 = b + a / 100) ∧ (m + n < 150) ∧ m = 50 ∧ n = 49 := 
by
  sorry

end find_two_digit_integers_l132_132828


namespace infinite_squarefree_triple_l132_132066

-- Define the set A(n)
def A (n : ℕ) : Set ℕ := 
  {m | m ∈ Finset.range n ∧ ∃ p : ℕ, Prime p ∧ 2*p ≤ m ∧ (p % 2 = 1) ∧ (p ^ 2 ∣ m)}

-- Proportion condition from the problem
axiom proportion_bound (n : ℕ) (h : n > 0) : (A n).card / n < 1/4 - 1/1000

-- Statement to be proven
theorem infinite_squarefree_triple : ∃ᶠ n in at_top, ∀ i : ℕ, i ≤ 2 → ¬∃ p : ℕ, Prime p ∧ p ^ 2 ∣ (n + i) :=
begin
  -- Future proof details
  sorry,
end

end infinite_squarefree_triple_l132_132066


namespace find_theta_l132_132477

theorem find_theta (R h : ℝ) (θ : ℝ) 
  (r1_def : r1 = R * Real.cos θ)
  (r2_def : r2 = (R + h) * Real.cos θ)
  (s_def : s = 2 * π * h * Real.cos θ)
  (s_eq_h : s = h) : 
  θ = Real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l132_132477


namespace quadratic_range_l132_132124

theorem quadratic_range (f : ℝ → ℝ) (a b : ℝ) (h : a ≤ b) : 
  (∀ x ∈ set.Ioc a b, f x = x^2 - 4 * x + 3) →
  set.range (λ x, if (x ∈ set.Ioc a b) then f x else 0) = set.Icc (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end quadratic_range_l132_132124


namespace find_theta_l132_132478

theorem find_theta (R h : ℝ) (θ : ℝ) 
  (r1_def : r1 = R * Real.cos θ)
  (r2_def : r2 = (R + h) * Real.cos θ)
  (s_def : s = 2 * π * h * Real.cos θ)
  (s_eq_h : s = h) : 
  θ = Real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l132_132478


namespace sum_of_T_l132_132029

-- Define what it means to be in the set \mathcal{T}
def is_in_T (x : ℝ) : Prop := ∃ (a b : ℕ), a ≠ b ∧ a ≤ 9 ∧ b ≤ 9 ∧ x = (10 * a + b) / 99

-- Define the set \mathcal{T}
def T : set ℝ := {x | is_in_T x}

-- Prove the sum of all elements in \mathcal{T} equals 45
theorem sum_of_T : ∑ x in T.to_finset, x = 45 :=
by
  sorry

end sum_of_T_l132_132029


namespace pentagon_distance_inequality_l132_132022

theorem pentagon_distance_inequality
  (ABCDE : Polygon)
  (F : Point)
  (F_inside : F ∈ Interior ABCDE)
  (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (b_1 b_2 b_3 b_4 b_5 : ℝ)
  (F1 F2 F3 F4 F5 : Point)
  (H_AF1_EQ_AF : dist A F1 = dist A F)
  (H_BF2_EQ_BF : dist B F2 = dist B F)
  (H_CF3_EQ_CF : dist C F3 = dist C F)
  (H_DF4_EQ_DF : dist D F4 = dist D F)
  (H_EF5_EQ_EF : dist E F5 = dist E F)
  (H_a_1 : dist_to_line F (Line.mk A B) = a_1)
  (H_a_2 : dist_to_line F (Line.mk B C) = a_2)
  (H_a_3 : dist_to_line F (Line.mk C D) = a_3)
  (H_a_4 : dist_to_line F (Line.mk D E) = a_4)
  (H_a_5 : dist_to_line F (Line.mk E A) = a_5)
  (H_b_1 : dist_to_line F1 (Line.mk E A) = b_1)
  (H_b_2 : dist_to_line F2 (Line.mk A B) = b_2)
  (H_b_3 : dist_to_line F3 (Line.mk B C) = b_3)
  (H_b_4 : dist_to_line F4 (Line.mk C D) = b_4)
  (H_b_5 : dist_to_line F5 (Line.mk D E) = b_5)
  : a_1 + a_2 + a_3 + a_4 + a_5 ≤ b_1 + b_2 + b_3 + b_4 + b_5 :=
  sorry

end pentagon_distance_inequality_l132_132022


namespace questionnaire_B_count_l132_132938

theorem questionnaire_B_count : 
  let a_n n := 12 * n - 9 in
  ∃ (n_min n_max : ℤ), 301 ≤ a_n n_min ∧ a_n n_max ≤ 495 ∧ 
  (∀ n, n_min ≤ n ∧ n ≤ n_max → 301 ≤ a_n n ∧ a_n n ≤ 495) ∧
  n_max - n_min + 1 = 17 :=
by 
  sorry

end questionnaire_B_count_l132_132938


namespace find_p_l132_132468

theorem find_p (p q : ℝ) (h1 : p + 2 * q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : 10 * p^9 * q = 45 * p^8 * q^2): 
  p = 9 / 13 :=
by
  sorry

end find_p_l132_132468


namespace max_BP_squared_l132_132034

-- Definitions of the problem's elements
def Circle (ω : Type) := sorry -- Placeholder for the type representing a circle
def Point := sorry -- Placeholder for the type representing a point

variables (ω : Circle) (A B C T P : Point)
variable [geometry : geometric_context ω]

-- Length definitions
def length_AB : ℝ := 20
def radius := length_AB / 2
def O := midpoint A B
def length_OC := 30

-- Assumptions based on the problem conditions
axiom diameter_AB : is_diameter (A, B) ω
axiom on_circle_T : point_on_circle T ω
axiom CT_tangent : tangent_line C T ω
axiom perpendicular_AP_CT : perpendicular_line A P (line C T)

-- Proof goal
theorem max_BP_squared : BP^2 = 1900 / 3 := by
  sorry

end max_BP_squared_l132_132034


namespace clock_angle_at_1015_l132_132890

theorem clock_angle_at_1015 :
  let hours := 12
  let minute_deg := 360 / 12
  let minutes := 15
  let minute_hand := (minutes / 60) * 360
  let hour := 10
  let past_hour_minutes := 15
  let hour_hand := (hour * minute_deg) + ((past_hour_minutes / 60) * minute_deg)
  let angle_diff := abs (minute_hand - hour_hand)
  let acute_angle := if angle_diff <= 180 then angle_diff else 360 - angle_diff
  acute_angle = 37.5 :=
by
  sorry

end clock_angle_at_1015_l132_132890


namespace tan_double_angle_l132_132661

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 1 / 3) : Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_double_angle_l132_132661


namespace log_inequality_l132_132280

theorem log_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : x^2 + y = 0) : 
  log a (a^x + a^y) ≤ log a 2 + 1 / 8 := 
sorry

end log_inequality_l132_132280


namespace range_of_sum_of_reciprocals_l132_132278

theorem range_of_sum_of_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  ∃ (r : ℝ), ∀ t ∈ Set.Ici (3 + 2 * Real.sqrt 2), t = (1 / x + 1 / y) := 
sorry

end range_of_sum_of_reciprocals_l132_132278


namespace reciprocal_of_minus_one_over_2023_l132_132863

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l132_132863


namespace right_to_left_evaluation_correct_l132_132234

variable (a b c d : ℝ)

def right_to_left_evaluation (a b c d : ℝ) : ℝ :=
  a * (b / (c + d^2))

theorem right_to_left_evaluation_correct :
  right_to_left_evaluation a b c d = (a * b) / (c + d^2) :=
by
  sorry

end right_to_left_evaluation_correct_l132_132234


namespace vartan_recreation_percent_l132_132388

noncomputable def percent_recreation_week (last_week_wages current_week_wages current_week_recreation last_week_recreation : ℝ) : ℝ :=
  (current_week_recreation / current_week_wages) * 100

theorem vartan_recreation_percent 
  (W : ℝ) 
  (h1 : last_week_wages = W)  
  (h2 : last_week_recreation = 0.15 * W)
  (h3 : current_week_wages = 0.90 * W)
  (h4 : current_week_recreation = 1.80 * last_week_recreation) :
  percent_recreation_week last_week_wages current_week_wages current_week_recreation last_week_recreation = 30 :=
by
  sorry

end vartan_recreation_percent_l132_132388


namespace diana_probability_larger_than_apollo_l132_132244

open Classical

noncomputable def probability_diana_larger (diana_die apollo_die : Fin 6 → ℚ) : ℚ :=
  let diana_wins := Finset.univ.filter (λ x : Fin 6, Finset.univ.filter (λ y : Fin 6, x > y).card)
  diana_wins.card / 36

theorem diana_probability_larger_than_apollo :
  let diana_die := (f : Fin 6 → ℚ) (f 0 = 1 / 6) ∧ (f 1 = 1 / 6) ∧ (f 2 = 1 / 6) ∧ (f 3 = 1 / 6) ∧ (f 4 = 1 / 6) ∧ (f 5 = 1 / 6)
  let apollo_die := (f : Fin 6 → ℚ) (f 0 = 1 / 8) ∧ (f 1 = 1 / 8) ∧ (f 2 = 1 / 8) ∧ (f 3 = 1 / 8) ∧ (f 4 = 1 / 8) ∧ (f 5 = 3 / 8)
  probability_diana_larger diana_die apollo_die = 15 / 64 :=
begin
  sorry
end

end diana_probability_larger_than_apollo_l132_132244


namespace distance_and_speed_l132_132640

def distance_to_travel (s: ℝ) (g: ℝ) (g1: ℝ) (m: ℝ) (m2: ℝ) : ℝ :=
  (s / 3) * (m + g) / (m + 5 * g)

def average_speed (s: ℝ) (g: ℝ) (g1: ℝ) (m: ℝ) (m2: ℝ) : ℝ :=
  let d := (s / 3) * (m + g) / (m + 5 * g)
  let tqdm := d * (5 * m + g) / (3 * m * (m + g))
  s / (tqdm * 3)

theorem distance_and_speed (s: ℝ) (g: ℝ) (g1: ℝ) (m: ℝ) (m2: ℝ) :
  s = 60 → g = 5 → g1 = 4 → m = 30 → m2 = 25 →
  distance_to_travel s g g1 m m2 ≈ 12.73 ∧ average_speed s g g1 m m2 ≈ 8.3 := 
by
  intros h1 h2 h3 h4 h5
  unfold distance_to_travel
  unfold average_speed
  sorry

end distance_and_speed_l132_132640


namespace rhombus_region_area_l132_132071

noncomputable def region_area (s : ℝ) (angleB : ℝ) : ℝ :=
  let h := (s / 2) * (Real.sin (angleB / 2))
  let area_triangle := (1 / 2) * (s / 2) * h
  3 * area_triangle

theorem rhombus_region_area : region_area 3 150 = 0.87345 := by
    sorry

end rhombus_region_area_l132_132071


namespace concyclic_l132_132761

-- Definitions of the points and conditions
variables {A B C P Q R S : Type*}
variables [euclidean_geometry A B C P Q R S]

-- Define the conditions
def conditions : Prop :=
  P ∈ line_segment A B ∧ Q ∈ line_segment A C ∧
  dist A P = dist A Q ∧
  R ∈ line_segment B C ∧ S ∈ line_segment B C ∧
  angle B P R = angle P S R ∧
  angle C Q S = angle Q R S

-- The theorem we want to prove
theorem concyclic (h : conditions) : is_concyclic P Q R S :=
sorry

end concyclic_l132_132761


namespace inverse_log2_function_l132_132619

theorem inverse_log2_function :
  ∀ x : ℝ, x > 1 → (∃ y : ℝ, y = (2^x + 1) ↔ x = logBase 2 (y-1)) :=
by
  sorry

end inverse_log2_function_l132_132619


namespace max_value_of_determinant_is_half_l132_132260

noncomputable def max_determinant : ℝ :=
  let θ : ℝ := sorry;
  let φ : ℝ := sorry;
  let det := 
    abs (matrix.det 
      ![![1, 1, 1], 
        ![1, 1 + real.sin(θ + φ), 1], 
        ![1 + real.cos(θ + φ), 1, 1]]) in
  max det
  
theorem max_value_of_determinant_is_half : max_determinant = 1 / 2 := sorry

end max_value_of_determinant_is_half_l132_132260


namespace winning_strategy_l132_132916

def player_A_wins (n : ℕ) : Prop := even n → (first_player wins)
def player_B_wins (n : ℕ) : Prop := odd n → (second_player wins)

theorem winning_strategy (n : ℕ) : player_A_wins n ∨ player_B_wins n :=
sorry

end winning_strategy_l132_132916


namespace last_nonzero_digit_50_factorial_l132_132620

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the last non-zero digit function
def last_nonzero_digit (n : ℕ) : ℕ :=
  let digit := factorial n in
  let rec last_digit (m : ℕ) : ℕ :=
    if m % 10 = 0 then last_digit (m / 10)
    else m % 10
  in last_digit digit

-- State the theorem
theorem last_nonzero_digit_50_factorial : last_nonzero_digit 50 = 2 := 
  sorry

end last_nonzero_digit_50_factorial_l132_132620


namespace smallest_n_inverse_mod_1260_l132_132503

theorem smallest_n_inverse_mod_1260 : ∃ n : ℕ, n > 1 ∧ n.inverse_mod 1260 ≠ 0 ∧ ∀ m : ℕ, m > 1 ∧ m.inverse_mod 1260 ≠ 0 → n ≤ m := by
  sorry

end smallest_n_inverse_mod_1260_l132_132503


namespace john_umbrella_in_car_l132_132016

variable (UmbrellasInHouse : Nat)
variable (CostPerUmbrella : Nat)
variable (TotalAmountPaid : Nat)

theorem john_umbrella_in_car
  (h1 : UmbrellasInHouse = 2)
  (h2 : CostPerUmbrella = 8)
  (h3 : TotalAmountPaid = 24) :
  (TotalAmountPaid / CostPerUmbrella) - UmbrellasInHouse = 1 := by
  sorry

end john_umbrella_in_car_l132_132016


namespace lasagna_pieces_l132_132416

-- Definition of the conditions
def manny_piece := 1
def aaron_piece := 0
def kai_piece := 2 * manny_piece
def raphael_piece := 0.5 * manny_piece
def lisa_piece := 2 + 0.5 * raphael_piece

-- The main theorem statement proving the total number of pieces
theorem lasagna_pieces : 
  manny_piece + aaron_piece + kai_piece + raphael_piece + lisa_piece = 6 :=
by
  -- Proof is omitted
  sorry

end lasagna_pieces_l132_132416


namespace min_lambda_value_l132_132288

noncomputable def minimum_lambda : ℝ :=
  let d := (sqrt ((2 + sqrt 2) / 4)) in
  2 + sqrt 2

theorem min_lambda_value : minimum_lambda = 2 + sqrt 2 :=
begin
  sorry
end

end min_lambda_value_l132_132288


namespace parabola_translation_shift_downwards_l132_132118

theorem parabola_translation_shift_downwards :
  ∀ (x y : ℝ), (y = x^2 - 5) ↔ ((∃ (k : ℝ), k = -5 ∧ y = x^2 + k)) :=
by
  sorry

end parabola_translation_shift_downwards_l132_132118


namespace fraction_of_area_outside_circle_l132_132077

open Float.Real

noncomputable def fraction_area_outside_circle (r : ℝ) : ℝ :=
  let area_triangle := (3 * r^2 * sqrt 3) / 4
  let area_segment := (π * r^2) / 3 - (r^2 * sqrt 3) / 4
  let area_outside := area_triangle - 2 * area_segment
  area_outside / area_triangle

theorem fraction_of_area_outside_circle (r : ℝ) :
  fraction_area_outside_circle r = (4 / 3) - (4 * sqrt 3 * π / 27) :=
by
  sorry

end fraction_of_area_outside_circle_l132_132077


namespace probability_consecutive_and_ordered_l132_132918

-- Define the number of chips of each color
def num_tan_chips := 4
def num_pink_chips := 3
def num_violet_chips := 5
def total_chips := num_tan_chips + num_pink_chips + num_violet_chips

-- Define factorial function for convenience
noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Calculate factorials for the relevant numbers
def fact_tan := factorial num_tan_chips
def fact_pink := factorial num_pink_chips
def fact_violet := factorial num_violet_chips
def fact_total := factorial total_chips

-- Define the probability calculation
noncomputable def probability_all_consecutive_blocks : ℚ :=
  (fact_tan * fact_pink * fact_violet * 2) / fact_total

-- The theorem to be proven
theorem probability_consecutive_and_ordered :
  probability_all_consecutive_blocks = 1 / 13860 := sorry

end probability_consecutive_and_ordered_l132_132918


namespace infinite_triples_of_coprime_integers_l132_132805

open Nat

theorem infinite_triples_of_coprime_integers :
  ∃ᶠ (a b c : ℕ),
  (coprime a b) ∧ (coprime b c) ∧ (coprime c a) ∧
  (coprime (a * b + c) (b * c + a)) ∧
  (coprime (b * c + a) (c * a + b)) ∧
  (coprime (c * a + b) (a * b + c)) :=
  sorry

end infinite_triples_of_coprime_integers_l132_132805


namespace find_pure_gala_trees_l132_132927

variable (T : ℕ) -- Total number of trees
variable (cross_pollinated : ℕ) -- Number of cross-pollinated trees
variable (pure_fuji : ℕ) -- Number of pure Fuji trees
variable (pure_gala : ℕ) -- Number of pure Gala trees

-- Conditions from the problem
def condition1 : Prop := cross_pollinated = (10 * T) / 100
def condition2 : Prop := pure_fuji + cross_pollinated = 204
def condition3 : Prop := pure_fuji = (3 * T) / 4

-- Desired conclusion
def conclusion : Prop := pure_gala = T - (pure_fuji + cross_pollinated) ∧ pure_gala = 36

theorem find_pure_gala_trees :
  condition1 ∧ condition2 ∧ condition3 → conclusion :=
by
  intro h,
  cases h with h1 h23,
  cases h23 with h2 h3,
  sorry

end find_pure_gala_trees_l132_132927


namespace tangent_line_at_1_g_sum_geq_four_l132_132676

noncomputable def f (x : ℝ) : ℝ := exp x - (1/2) * x^2

theorem tangent_line_at_1 :
  ((exp 1 - 1) * (x : ℝ) - y - (1/2) = 0) :=
sorry

def g (x : ℝ) : ℝ := f x + 3 * x + 1

theorem g_sum_geq_four (x1 x2 : ℝ) (h : x1 + x2 ≥ 0) :
  g(x1) + g(x2) ≥ 4 :=
sorry

end tangent_line_at_1_g_sum_geq_four_l132_132676


namespace candidate_lost_by_1908_votes_l132_132172

theorem candidate_lost_by_1908_votes (votes_cast : ℕ) (candidate_percentage : ℝ)
  (votes_cast_total : votes_cast = 5300) (candidate_vote_percentage_32: candidate_percentage = 0.32) :
  let candidate_votes := candidate_percentage * votes_cast in
  let rival_votes := (1 - candidate_percentage) * votes_cast in
  rival_votes - candidate_votes = 1908 :=
by
  sorry

end candidate_lost_by_1908_votes_l132_132172


namespace investment_plans_count_l132_132187

theorem investment_plans_count :
  let cities : ℕ := 4
  let projects : ℕ := 3
  (Σ (dist : cities → ℕ), dist 0 + dist 1 + dist 2 + dist 3 = projects ∧
   ∀ i, dist i ≤ 2) = 60 := by sorry

end investment_plans_count_l132_132187


namespace possible_b4b7_products_l132_132037

theorem possible_b4b7_products (b : ℕ → ℤ) (d : ℤ)
  (h_arith_sequence : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_product_21 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = 21 :=
by
  sorry

end possible_b4b7_products_l132_132037


namespace exists_new_configuration_l132_132906

-- Definitions for points and their intersections
inductive Point
| A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P

-- Define a function that represents the line with exactly 4 chips
def valid_line (line : list Point) : Prop :=
  line.length = 4

-- Define the 5 lines with chips
def line1 : list Point := [Point.A, Point.F, Point.J, Point.N]
def line2 : list Point := [Point.B, Point.F, Point.K, Point.L]
def line3 : list Point := [Point.M, Point.F, Point.D, Point.G]
def line4 : list Point := [Point.A, Point.G, Point.K, Point.P]
def line5 : list Point := [Point.J, Point.F, Point.I, Point.P]

-- Statement asserting that there exist valid configurations
theorem exists_new_configuration :
  valid_line line1 ∧ valid_line line2 ∧ valid_line line3 ∧ valid_line line4 ∧ valid_line line5 :=
by { 
   unfold valid_line,
   -- Verifying that each line contains 4 elements
   repeat { simp },
   -- Ensure no line is a simple rotation or reflection of given configuration
   sorry
}

end exists_new_configuration_l132_132906


namespace ratio_isosceles_trapezoid_l132_132393

variables (AB CD : ℝ) (A B C D P : Type)

-- Define the isosceles trapezoid and its properties
structure IsoscelesTrapezoid (AB CD : ℝ) :=
(base1 : AB > CD)
(base2_parallels : parallel base1 base2)

-- Define the areas of the triangles
variables (area_PCD area_PBC area_PDA area_PAB : ℝ)
variables (PY PX : ℝ)
variables (triangle_PCD triangle_PBC triangle_PDA triangle_PAB : Triangle)

-- Assign the given areas to the triangles
def areas_of_triangles (triangle_PCD triangle_PBC triangle_PDA triangle_PAB : Triangle) : Prop :=
  area_PCD = 3 ∧ area_PBC = 4 ∧ area_PDA = 6 ∧ area_PAB = 7

-- Define the perpendicular distances
def perpendicular_distances (PY PX : ℝ) (CD AB : ℝ) : Prop :=
  PY = 6 / CD ∧ PX = 14 / AB

-- Total area of the trapezoid
def total_area_is (AB CD PY PX : ℝ) : Prop :=
  20 = (1/2) * (AB + CD) * (PX + PY)

-- Prove the ratio of AB and CD is 7/3
theorem ratio_isosceles_trapezoid (AB CD : ℝ)
  (h1 : IsoscelesTrapezoid AB CD)
  (h2 : areas_of_triangles triangle_PCD triangle_PBC triangle_PDA triangle_PAB)
  (h3 : perpendicular_distances PY PX CD AB)
  (h4 : total_area_is AB CD PY PX) :
  AB / CD = 7 / 3 :=
begin
  sorry
end

end ratio_isosceles_trapezoid_l132_132393


namespace average_percentage_taller_l132_132598

theorem average_percentage_taller 
  (h1 b1 h2 b2 h3 b3 : ℝ)
  (h1_eq : h1 = 228) (b1_eq : b1 = 200)
  (h2_eq : h2 = 120) (b2_eq : b2 = 100)
  (h3_eq : h3 = 147) (b3_eq : b3 = 140) :
  ((h1 - b1) / b1 * 100 + (h2 - b2) / b2 * 100 + (h3 - b3) / b3 * 100) / 3 = 13 := by
  rw [h1_eq, b1_eq, h2_eq, b2_eq, h3_eq, b3_eq]
  sorry

end average_percentage_taller_l132_132598


namespace sum_of_vars_l132_132342

theorem sum_of_vars (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 2 * y) : x + y + z = 7 * x := 
by 
  sorry

end sum_of_vars_l132_132342


namespace area_triangle_PZQ_correct_l132_132730

open Real

noncomputable def area_of_triangle_PZQ 
  (PQ RS QR: ℝ) (X Y: ℝ) (Z: ℝ) : ℝ :=
  if h : (PQ = 8) ∧ (QR = 4) ∧ (X = 2) ∧ (Y = 3) then
    128 / 3
  else
    0

theorem area_triangle_PZQ_correct
  (PQ RS QR: ℝ) (X Y: ℝ) (Z: ℝ)
  (h1 : PQ = 8)
  (h2 : QR = 4)
  (h3 : X = 2)
  (h4 : Y = 3) :
  area_of_triangle_PZQ PQ RS QR X Y Z = 128 / 3 :=
by 
  rw [area_of_triangle_PZQ, if_pos]
  simp [h1, h2, h3, h4]
  sorry

end area_triangle_PZQ_correct_l132_132730


namespace reflected_line_equation_l132_132919

theorem reflected_line_equation :
  ∃ m b : ℝ, (∀ (x y : ℝ), (2x - y + 2 = 0 ↔ y = m * x + b)) ∧
    (m * (-1) + b = 0) ∧
    (m * 0 + b = 2) :=
by
  sorry

end reflected_line_equation_l132_132919


namespace determine_a_range_l132_132717

open Real

theorem determine_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a ≤ 0) → a ≤ 1 :=
sorry

end determine_a_range_l132_132717


namespace prime_polynomial_l132_132392

theorem prime_polynomial (n : ℕ) (h1 : 2 ≤ n)
  (h2 : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
sorry

end prime_polynomial_l132_132392


namespace smallest_nonpalindromic_power_of_7_l132_132634

noncomputable def isPalindrome (n : ℕ) : Bool :=
  let s := n.toString
  s == s.reverse

theorem smallest_nonpalindromic_power_of_7 :
  ∃ n : ℕ, ∃ m : ℕ, m = 7^n ∧ ¬ isPalindrome m ∧ ∀ k : ℕ, k < n → (isPalindrome (7^k) → False) → n = 4 ∧ m = 2401 :=
by sorry

end smallest_nonpalindromic_power_of_7_l132_132634


namespace parallel_lines_slope_l132_132971

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end parallel_lines_slope_l132_132971


namespace option_B_is_irrational_l132_132509

theorem option_B_is_irrational : 
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ (sqrt 5) = (p / q)) ∧
  ((∃ (p q : ℤ), q ≠ 0 ∧ ((1 : ℚ) / 2) = (p / q)) ∧
   (∃ (p q : ℤ), q ≠ 0 ∧ (27 ^ (1 / 3 : ℚ) = (p / q))) ∧
   (∃ (p q : ℤ), q ≠ 0 ∧ (314 / 100 : ℚ) = (p / q))) :=
sorry

end option_B_is_irrational_l132_132509


namespace largest_integer_in_eight_consecutive_l132_132225

theorem largest_integer_in_eight_consecutive (sum_2044 : ∑ i in (finset.range 8).map (λ x, start + x) = 2044) : 
  let s := (finset.range 8).map (λ x, start + x) in
  s.max = 259 :=
by {
  let s := (finset.range 8).map (λ x, start + x),
  have : ∑ i in s = 2044 := sum_2044,
  sorry
}

end largest_integer_in_eight_consecutive_l132_132225


namespace books_finished_l132_132056

theorem books_finished (miles_traveled : ℕ) (miles_per_book : ℕ) (h_travel : miles_traveled = 6760) (h_rate : miles_per_book = 450) : (miles_traveled / miles_per_book) = 15 :=
by {
  -- Proof will be inserted here
  sorry
}

end books_finished_l132_132056


namespace equilateral_triangle_area_percentage_l132_132578

noncomputable def percentage_area_of_triangle_in_pentagon (s : ℝ) : ℝ :=
  ((4 * Real.sqrt 3 - 3) / 13) * 100

theorem equilateral_triangle_area_percentage
  (s : ℝ) :
  let pentagon_area := s^2 * (1 + Real.sqrt 3 / 4)
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  (triangle_area / pentagon_area) * 100 = percentage_area_of_triangle_in_pentagon s :=
by
  sorry

end equilateral_triangle_area_percentage_l132_132578


namespace part1_part2_l132_132009

noncomputable def a_n (n : ℕ) : ℕ -> ℚ :=
  if n = 0 then 0 else 2 * 3^(n-1) / n

theorem part1 (n : ℕ) :
  (∑ i in finset.range n, (i + 1) * a_n (i + 1)) = 3^n - 1 :=
sorry

noncomputable def b_n (n : ℕ) : ℝ :=
  n * (n + 1) * (a_n n) / 2

theorem part2 (n : ℕ) :
  (∑ i in finset.range n, b_n (i + 1)) =
  (n / 2 + 1 / 4) * 3^n - 1 / 4 :=
sorry

end part1_part2_l132_132009


namespace ordered_pair_unique_l132_132243

-- Define the conditions
variables a b : ℝ
variables (ha : a ≠ 0) (hb : b ≠ 0)
variables (h: ∃ x : ℝ, (x = a ∨ x = -b) ∧ x^2 + bx + a = 0)

-- State the theorem
theorem ordered_pair_unique :
  a = 1 ∧ b = 1 :=
sorry

end ordered_pair_unique_l132_132243


namespace B_is_brownian_motion_l132_132044

open ProbabilityTheory

-- Given: B^{ \circ} is a Brownian bridge, which is a centered Gaussian process with a specific covariance structure.
variables (B_circ : ℝ → ℝ)
  [is_gaussian B_circ]
  (h_B_circ_zero_mean : ∀ t, t ∈ set.Icc 0 1 → 𝔼[B_circ t] = 0)
  (h_B_circ_cov : ∀ s t, s ∈ set.Icc 0 1 → t ∈ set.Icc 0 1 → 𝔼[B_circ s * B_circ t] = s * (1 - t))

-- Defining the transformed process B_t
noncomputable def B (t : ℝ) : ℝ := (1 + t) * B_circ (t / (1 + t))

-- Statement: Show that the process B = (B_t)_{t ≥ 0} is a Brownian motion
theorem B_is_brownian_motion : IsBrownianMotion (λ t, B B_circ t) :=
sorry

end B_is_brownian_motion_l132_132044


namespace max_BP_squared_l132_132035

-- Definitions of the problem's elements
def Circle (ω : Type) := sorry -- Placeholder for the type representing a circle
def Point := sorry -- Placeholder for the type representing a point

variables (ω : Circle) (A B C T P : Point)
variable [geometry : geometric_context ω]

-- Length definitions
def length_AB : ℝ := 20
def radius := length_AB / 2
def O := midpoint A B
def length_OC := 30

-- Assumptions based on the problem conditions
axiom diameter_AB : is_diameter (A, B) ω
axiom on_circle_T : point_on_circle T ω
axiom CT_tangent : tangent_line C T ω
axiom perpendicular_AP_CT : perpendicular_line A P (line C T)

-- Proof goal
theorem max_BP_squared : BP^2 = 1900 / 3 := by
  sorry

end max_BP_squared_l132_132035


namespace pizza_slice_longest_segment_square_l132_132180

variables (R θ : ℝ)

def diameter := 16 -- Diameter of the pizza
def radius := diameter / 2 -- Radius of the pizza
def central_angle := 90 -- Central angle of each slice in degrees
def radian θ := θ * real.pi / 180 -- Convert degrees to radians

theorem pizza_slice_longest_segment_square : 
    ∀ (diameter : ℝ) (central_angle : ℝ), 
    diameter = 16 → 
    central_angle = 90 → 
    let R := diameter / 2 in 
    let m := 2 * R * real.sin (radian (central_angle / 2)) in
    m^2 = 128 :=
by 
  intros diameter central_angle h_diameter h_angle 
  let R := diameter / 2
  let m := 2 * R * real.sin (radian (central_angle / 2))
  have h_r : R = 8 := by linarith
  have h_m : m = 8 * real.sqrt 2 :=
    by { rw [h_r, radian, sin, real.sin_pi_div_four], simp, ring }
  rw [h_m], norm_num

end pizza_slice_longest_segment_square_l132_132180


namespace percentage_gain_is_20_percent_l132_132845

theorem percentage_gain_is_20_percent (manufacturing_cost transportation_cost total_shoes selling_price : ℝ)
(h1 : manufacturing_cost = 220)
(h2 : transportation_cost = 500)
(h3 : total_shoes = 100)
(h4 : selling_price = 270) :
  let cost_per_shoe := manufacturing_cost + transportation_cost / total_shoes
  let profit_per_shoe := selling_price - cost_per_shoe
  let percentage_gain := (profit_per_shoe / cost_per_shoe) * 100
  percentage_gain = 20 :=
by
  sorry

end percentage_gain_is_20_percent_l132_132845


namespace number_of_valid_paintings_l132_132959

-- Given an 8-sided die, numbered 1 to 8
def faces : Finset ℕ := {(1 : ℕ), 2, 3, 4, 5, 6, 7, 8}

-- Predicate for selecting 3 faces such that their sum is not 10
def valid_painting (s : Finset ℕ) : Prop := (s.card = 3) ∧ (s.sum ≠ 10)

-- Theorem stating the number of valid ways to select 3 faces so that their sum is not 10
theorem number_of_valid_paintings : 
  (faces.subsets.filter valid_painting).card = 53 :=
sorry

end number_of_valid_paintings_l132_132959


namespace maximum_value_u_l132_132683

noncomputable def f (a c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - 4 * x + c

theorem maximum_value_u (a c f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 - 4 * x + c)
  (h2 : 0 < a)
  (h3 : ∀ y : ℝ, ∃ x : ℝ, f x = y)
  (h4 : 0 ≤ f 1 ∧ f 1 ≤ 4)
  (h5 : ∀ a c : ℝ, 4 ≤ a + c ∧ a + c ≤ 8 ∧ a * c = 4) :
  ∃ u : ℝ, u = 7/4 :=
by
  sorry

end maximum_value_u_l132_132683


namespace scientific_notation_14nm_l132_132439

theorem scientific_notation_14nm :
  0.000000014 = 1.4 * 10^(-8) := 
by 
  sorry

end scientific_notation_14nm_l132_132439


namespace difference_fewer_children_than_adults_l132_132924

theorem difference_fewer_children_than_adults : 
  ∀ (C S : ℕ), 2 * C = S → 58 + C + S = 127 → (58 - C = 35) :=
by
  intros C S h1 h2
  sorry

end difference_fewer_children_than_adults_l132_132924


namespace probability_odd_sum_l132_132143

noncomputable def favorable_outcomes : ℕ := 18
noncomputable def total_outcomes : ℕ := 6 * 6

theorem probability_odd_sum :
  favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_odd_sum_l132_132143


namespace z_in_second_quadrant_l132_132715

open Complex

-- Given the condition
def satisfies_eqn (z : ℂ) : Prop := z * (1 - I) = 4 * I

-- Define the second quadrant condition
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (h : satisfies_eqn z) : in_second_quadrant z :=
  sorry

end z_in_second_quadrant_l132_132715


namespace lasagna_pieces_l132_132418

theorem lasagna_pieces (m a k r l : ℕ → ℝ)
  (hm : m 1 = 1)                -- Manny's consumption
  (ha : a 0 = 0)                -- Aaron's consumption
  (hk : ∀ n, k n = 2 * (m 1))   -- Kai's consumption
  (hr : ∀ n, r n = (1 / 2) * (m 1)) -- Raphael's consumption
  (hl : ∀ n, l n = 2 + (r n))   -- Lisa's consumption
  : m 1 + a 0 + k 1 + r 1 + l 1 = 6 :=
by
  -- Proof goes here
  sorry

end lasagna_pieces_l132_132418


namespace product_remainder_mod_11_l132_132499

theorem product_remainder_mod_11 :
  (1010 * 1011 * 1012 * 1013 * 1014) % 11 = 0 :=
by
  have h1 : 1010 % 11 = 9 := by sorry
  have h2 : 1011 % 11 = 10 := by sorry
  have h3 : 1012 % 11 = 0 := by sorry
  have h4 : 1013 % 11 = 1 := by sorry
  have h5 : 1014 % 11 = 2 := by sorry
  calc
    (1010 * 1011 * 1012 * 1013 * 1014) % 11
        = (9 * 10 * 0 * 1 * 2) % 11 : by sorry
    ... = 0 : by sorry

end product_remainder_mod_11_l132_132499


namespace total_amount_is_correct_l132_132570

variable (w x y z R : ℝ)
variable (hx : x = 0.345 * w)
variable (hy : y = 0.45625 * w)
variable (hz : z = 0.61875 * w)
variable (hy_value : y = 112.50)

theorem total_amount_is_correct :
  R = w + x + y + z → R = 596.8150684931507 := by
  sorry

end total_amount_is_correct_l132_132570


namespace part_I_part_II_1_part_II_2_l132_132651

noncomputable def sequence_a : ℕ+ → ℚ
| 1 := -1
| (n+1) := (n : ℚ+).toRat * 3 * (sequence_a n) + 4 * (n : ℚ+).toRat + 6 * n.toRat / n

noncomputable def sequence_b (n : ℕ+) : ℚ :=
3^(n - 1) / (sequence_a n + 2)

noncomputable def S (n : ℕ+) : ℚ :=
(n.range 1 (λ i, sequence_b (⟨i, nat.succ_pos i⟩))).sum.toRat

theorem part_I :
  (is_geometric (λ n, (sequence_a n) / n + 2 / n) 1 3) :=
sorry

theorem part_II_1 (n : ℕ+): 
  (finset.sum ((finset.Icc (n+1) (2*n) : finset ℕ+)).to<typename := seq_sequence_b) < 4 / 5 :=
sorry

theorem part_II_2 (n : ℕ+) (h : 2 ≤ n):
  (S (n : ℕ+) ^ 2 > 
    2 * (finset.sum ((finset.Icc 2 n).map typingainto<nat.to_type> (λ i, S (i : ℕ+) / (i: ℕ)))) :=
sorry

end part_I_part_II_1_part_II_2_l132_132651


namespace combined_equivalent_percentage_correct_l132_132236

noncomputable def equivalent_percentage_mobile_phone : ℝ :=
let increase_40 := 1 + 40 / 100,
    decrease_15 := 1 - 15 / 100 in
  increase_40 * decrease_15

noncomputable def equivalent_percentage_laptop : ℝ :=
let increase_25 := 1 + 25 / 100,
    decrease_20 := 1 - 20 / 100,
    further_increase_10 := 1 + 10 / 100 in
  increase_25 * decrease_20 * further_increase_10

noncomputable def equivalent_percentage_tablet : ℝ :=
let increase_35 := 1 + 35 / 100,
    decrease_10 := 1 - 10 / 100,
    further_decrease_5 := 1 - 5 / 100 in
  increase_35 * decrease_10 * further_decrease_5

noncomputable def combined_equivalent_percentage : ℝ :=
(equivalent_percentage_mobile_phone + equivalent_percentage_laptop + equivalent_percentage_tablet - 3) / 3 * 100

theorem combined_equivalent_percentage_correct :
  combined_equivalent_percentage = 14.7375 := 
sorry

end combined_equivalent_percentage_correct_l132_132236


namespace dragons_legs_l132_132231

theorem dragons_legs :
  ∃ (n : ℤ), ∀ (x y : ℤ), x + 3 * y = 26
                       → 40 * x + n * y = 298
                       → n = 14 :=
by
  sorry

end dragons_legs_l132_132231


namespace smallest_angle_of_quadrilateral_l132_132846

theorem smallest_angle_of_quadrilateral (a b c d : ℝ) (ratio : a:b:c:d = 3:4:5:6) (sum_angles : a + b + c + d = 360) :
  ∃ k : ℝ, a = 60 :=
by
  sorry

end smallest_angle_of_quadrilateral_l132_132846


namespace original_average_and_variance_l132_132286

variables {n : ℕ} {x : Fin n → ℝ}

def transformed_data (x : Fin n → ℝ) : Fin n → ℝ := λ i, 2 * x i - 3

theorem original_average_and_variance (h_avg : (∑ i, transformed_data x i) / n = 7)
  (h_var : (∑ i, (transformed_data x i - (7 : ℝ))^2) / n = 4) :
  (∑ i, x i) / n = 5 ∧ (∑ i, (x i - 5)^2) / n = 1 :=
begin
  sorry
end

end original_average_and_variance_l132_132286


namespace estimate_mass_of_ice_floe_l132_132796

noncomputable def mass_of_ice_floe (d : ℝ) (D : ℝ) (m : ℝ) : ℝ :=
  (m * d) / (D - d)

theorem estimate_mass_of_ice_floe :
  mass_of_ice_floe 9.5 10 600 = 11400 := 
by
  sorry

end estimate_mass_of_ice_floe_l132_132796


namespace inequality_proof_l132_132658

theorem inequality_proof (a b c d : ℕ) (h₀: a + c ≤ 1982) (h₁: (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)) (h₂: (a:ℚ)/b + (c:ℚ)/d < 1) :
  1 - (a:ℚ)/b - (c:ℚ)/d > 1 / (1983 ^ 3) :=
sorry

end inequality_proof_l132_132658


namespace quadratic_inequality_solution_l132_132685

theorem quadratic_inequality_solution (a b x : ℝ) 
  (hsol : ∀ x, -2 ≤ x ∧ x ≤ 1 → ax^2 - x + b ≥ 0) : 
  ∀ x, - (1 / 2) ≤ x ∧ x ≤ 1 → bx^2 - x + a ≤ 0 :=
begin
  sorry
end

end quadratic_inequality_solution_l132_132685


namespace toaster_popularity_l132_132595

theorem toaster_popularity
  (c₁ c₂ : ℤ) (p₁ p₂ k : ℤ)
  (h₀ : p₁ * c₁ = k)
  (h₁ : p₁ = 12)
  (h₂ : c₁ = 500)
  (h₃ : c₂ = 750)
  (h₄ : k = p₁ * c₁) :
  p₂ * c₂ = k → p₂ = 8 :=
by
  sorry

end toaster_popularity_l132_132595


namespace sum_of_c_l132_132318

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℕ :=
  2^(n - 1)

-- Define the sequence c_n
def c (n : ℕ) : ℕ :=
  a n * b n

-- Define the sum S_n of the first n terms of c_n
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => c (i + 1))

-- The main Lean statement
theorem sum_of_c (n : ℕ) : S n = 3 + (n - 1) * 2^(n + 1) :=
  sorry

end sum_of_c_l132_132318


namespace investment_plans_count_l132_132184

theorem investment_plans_count :
  ∃ (projects cities : ℕ) (no_more_than_two : ℕ → ℕ → Prop),
    no_more_than_two projects cities →
    projects = 3 →
    cities = 5 →
    (projects ≤ 2 ∧ projects > 0) →
    ( (3.choose 2) * 5 * 4 + 5.choose 3 ) = 120 :=
by
  sorry

end investment_plans_count_l132_132184


namespace maria_remaining_money_l132_132784

theorem maria_remaining_money (initial_amount ticket_cost : ℕ) (h_initial : initial_amount = 760) (h_ticket : ticket_cost = 300) :
  let hotel_cost := ticket_cost / 2
  let total_spent := ticket_cost + hotel_cost
  let remaining := initial_amount - total_spent
  remaining = 310 :=
by
  intros
  sorry

end maria_remaining_money_l132_132784


namespace algebraic_expression_value_l132_132704

theorem algebraic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 7 = -6 := by
  sorry

end algebraic_expression_value_l132_132704


namespace ramanujans_number_l132_132328

-- Define complex numbers
def h := 3 + 4 * Complex.i
def product := 16 + 24 * Complex.i
def r := 144 / 25 + (8 / 25) * Complex.i

-- The theorem restating the problem as a proof goal
theorem ramanujans_number : (∃ r : Complex, r * h = product) → r = 5.76 + 0.32 * Complex.i :=
sorry

end ramanujans_number_l132_132328


namespace tangent_line_curves_unique_intersection_l132_132317

theorem tangent_line_curves_unique_intersection (a : ℝ) (h_a : a > 0) :
  (∀ θ : ℝ, let x := a + 4 * Real.cos θ in let y := 1 + 4 * Real.sin θ in (x - a) ^ 2 + (y - 1) ^ 2 = 16) →
  (∀ ρ : ℝ, ∃ θ : ℝ, 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ = 5) →
  (∃! P : ℝ × ℝ, P ∈ ({p | ∃ θ : ℝ, p = (a + 4 * Real.cos θ, 1 + 4 * Real.sin θ)} ∩ {q | ∃ ρ : ℝ, ∃ θ : ℝ, q = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ 3 * q.1 + 4 * q.2 - 5 = 0})) →
  a = 7 := 
begin
  sorry
end

end tangent_line_curves_unique_intersection_l132_132317


namespace bella_more_than_max_l132_132824

noncomputable def num_students : ℕ := 10
noncomputable def bananas_eaten_by_bella : ℕ := 7
noncomputable def bananas_eaten_by_max : ℕ := 1

theorem bella_more_than_max : 
  bananas_eaten_by_bella - bananas_eaten_by_max = 6 :=
by
  sorry

end bella_more_than_max_l132_132824


namespace solution_set_l132_132253

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x - 2) / (x - 4) ≥ 3

theorem solution_set :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | 4 < x ∧ x ≤ 5} :=
by
  sorry

end solution_set_l132_132253


namespace road_repair_equation_l132_132369

variable (x : ℝ) 

-- Original problem conditions
def total_road_length := 150
def extra_repair_per_day := 5
def days_ahead := 5

-- The proof problem to show that the schedule differential equals 5 days ahead
theorem road_repair_equation :
  (total_road_length / x) - (total_road_length / (x + extra_repair_per_day)) = days_ahead :=
sorry

end road_repair_equation_l132_132369


namespace find_solution_pairs_l132_132611

theorem find_solution_pairs :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y - 2 * sqrt(x * y) - sqrt(y / x) + 2 = 0 ∧ 3 * x^2 * y^2 + y^4 = 84) ↔
  ((x = 1 / 3 ∧ y = 3) ∨ (x = real.root 4 (21 / 76) ∧ y = 2 * real.root 4 (84 / 19))) :=
  sorry

end find_solution_pairs_l132_132611


namespace remainder_six_pow_4032_mod_13_l132_132891

theorem remainder_six_pow_4032_mod_13 : (6 ^ 4032) % 13 = 1 := 
by
  sorry

end remainder_six_pow_4032_mod_13_l132_132891


namespace ratio_of_tagged_fish_is_1_over_25_l132_132723

-- Define the conditions
def T70 : ℕ := 70  -- Number of tagged fish first caught and tagged
def T50 : ℕ := 50  -- Total number of fish caught in the second sample
def t2 : ℕ := 2    -- Number of tagged fish in the second sample

-- State the theorem/question
theorem ratio_of_tagged_fish_is_1_over_25 : (t2 / T50) = 1 / 25 :=
by
  sorry

end ratio_of_tagged_fish_is_1_over_25_l132_132723


namespace work_finished_days_earlier_l132_132573

theorem work_finished_days_earlier
  (D : ℕ) (M : ℕ) (A : ℕ) (Work : ℕ) (D_new : ℕ) (E : ℕ)
  (hD : D = 8)
  (hM : M = 30)
  (hA : A = 10)
  (hWork : Work = M * D)
  (hTotalWork : Work = 240)
  (hD_new : D_new = Work / (M + A))
  (hDnew_calculated : D_new = 6)
  (hE : E = D - D_new)
  (hE_calculated : E = 2) : 
  E = 2 :=
by
  sorry

end work_finished_days_earlier_l132_132573


namespace x_fifth_power_sum_l132_132705

theorem x_fifth_power_sum (x : ℝ) (h : x + 1 / x = -5) : x^5 + 1 / x^5 = -2525 := by
  sorry

end x_fifth_power_sum_l132_132705


namespace seq_sum_1095_l132_132460

-- Define the sequence based on the given conditions
def seq : ℕ → ℝ
| 0       := 2
| 1       := 3
| (n + 2) := (1 + seq (n + 1)) / seq n

-- Define the sum of the first 1095 terms
noncomputable def sum_seq : ℕ → ℝ
| 0       := seq 0
| (n + 1) := seq (n + 1) + sum_seq n

-- The problem to prove
theorem seq_sum_1095 : sum_seq 1094 = 1971 := 
sorry

end seq_sum_1095_l132_132460


namespace cos_sum_identity_sin_sum_identity_l132_132064

-- Defining the given conditions
variable {α β a b : ℝ}
variable (h1 : sin α + sin β = a)
variable (h2 : cos α + cos β = b)

theorem cos_sum_identity : cos (α + β) = - (a^2 - b^2) / (a^2 + b^2) :=
by sorry

theorem sin_sum_identity : sin (α + β) = (2 * a * b) / (a^2 + b^2) :=
by sorry

end cos_sum_identity_sin_sum_identity_l132_132064


namespace f_is_correct_l132_132664

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -x^2 + x + 1
  else if x = 0 then 0
  else x^2 + x - 1

theorem f_is_correct (x : ℝ) : 
  f(x) = 
    if x > 0 then -x^2 + x + 1
    else if x = 0 then 0
    else x^2 + x - 1 := by
  sorry

end f_is_correct_l132_132664


namespace height_of_flagpole_l132_132546

theorem height_of_flagpole {FG FA HA AG : ℝ} (h1 : FG = 5) (h2 : AG = 1)
  (h3 : HA = 1.6) (h4 : ∀ F G A, ∠(F, G, A) = 90) :
  height_of_flagpole = 8 :=
  begin
    sorry
  end

end height_of_flagpole_l132_132546


namespace average_age_of_adults_l132_132453

theorem average_age_of_adults 
  (total_members : ℕ)
  (avg_age_total : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (avg_age_girls : ℕ)
  (avg_age_boys : ℕ)
  (total_sum_ages : ℕ := total_members * avg_age_total)
  (sum_ages_girls : ℕ := num_girls * avg_age_girls)
  (sum_ages_boys : ℕ := num_boys * avg_age_boys)
  (sum_ages_adults : ℕ := total_sum_ages - sum_ages_girls - sum_ages_boys)
  : (num_adults = 10) → (avg_age_total = 20) → (num_girls = 30) → (avg_age_girls = 18) → (num_boys = 20) → (avg_age_boys = 22) → (total_sum_ages = 1200) → (sum_ages_girls = 540) → (sum_ages_boys = 440) → (sum_ages_adults = 220) → (sum_ages_adults / num_adults = 22) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end average_age_of_adults_l132_132453


namespace storks_initial_count_l132_132528

theorem storks_initial_count (S : ℕ) 
  (h1 : 6 = (S + 2) + 1) : S = 3 :=
sorry

end storks_initial_count_l132_132528


namespace quadratic_trinomial_negative_value_l132_132070

theorem quadratic_trinomial_negative_value
  (a b c : ℝ)
  (h1 : b^2 ≥ 4 * c)
  (h2 : 1 ≥ 4 * a * c)
  (h3 : b^2 ≥ 4 * a) :
  ∃ x : ℝ, a * x^2 + b * x + c < 0 :=
by
  sorry

end quadratic_trinomial_negative_value_l132_132070


namespace total_cost_is_correct_l132_132422

noncomputable def total_cost_of_gifts : ℝ :=
  let polo_shirts := 3 * 26
  let necklaces := 2 * 83
  let computer_game := 90
  let socks := 4 * 7
  let books := 3 * 15
  let scarves := 2 * 22
  let mugs := 5 * 8
  let sneakers := 65

  let cost_before_discounts := polo_shirts + necklaces + computer_game + socks + books + scarves + mugs + sneakers

  let discount_books := 0.10 * books
  let discount_sneakers := 0.15 * sneakers
  let cost_after_discounts := cost_before_discounts - discount_books - discount_sneakers

  let sales_tax := 0.065 * cost_after_discounts
  let cost_after_tax := cost_after_discounts + sales_tax

  let final_cost := cost_after_tax - 12

  final_cost

theorem total_cost_is_correct :
  total_cost_of_gifts = 564.96 := by
sorry

end total_cost_is_correct_l132_132422


namespace women_doubles_tournament_handshakes_l132_132583

theorem women_doubles_tournament_handshakes :
  ∀ (teams : List (List Prop)), List.length teams = 4 → (∀ t ∈ teams, List.length t = 2) →
  (∃ (handshakes : ℕ), handshakes = 24) :=
by
  intro teams h1 h2
  -- Assume teams are disjoint and participants shake hands meeting problem conditions
  -- The lean proof will follow the logical structure used for the mathematical solution
  -- We'll now formalize the conditions and the handshake calculation
  sorry

end women_doubles_tournament_handshakes_l132_132583


namespace fish_population_may_first_l132_132532

theorem fish_population_may_first :
  ∀ (M N M_1 N_1 : ℕ),
  M = 60 →
  M_1 = M * 3 / 4 →
  N = (45 * 70 / 3 : ℤ).to_nat →
  N_1 = (N * 60 / 100 : ℤ).to_nat →
  (N_1 * 4 / 3 : ℤ).to_nat = 840 := 
begin
  sorry
end

end fish_population_may_first_l132_132532


namespace value_of_star_15_25_l132_132964

noncomputable def star (x y : ℝ) : ℝ := Real.log x / Real.log y

axiom condition1 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star (star (x^2) y) y = star x y
axiom condition2 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star x (star y y) = star (star x y) (star x 1)
axiom condition3 (h : 1 > 0) : star 1 1 = 0

theorem value_of_star_15_25 : star 15 25 = (Real.log 3 / (2 * Real.log 5)) + 1 / 2 := 
by 
  sorry

end value_of_star_15_25_l132_132964


namespace limit_nbn_l132_132239

noncomputable def M (x : ℝ) : ℝ := x - (3 * x^2) / 7

def b_n (n : ℕ) : ℝ :=
  let rec iterate (f : ℝ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
    match n with
    | 0 => x
    | k+1 => iterate f k (f x)
  in iterate M n (23 / n)

theorem limit_nbn : filter.tendsto (λ n, n * b_n n) filter.at_top (nhds (161 / 30)) :=
sorry

end limit_nbn_l132_132239


namespace promotion_savings_l132_132202

theorem promotion_savings (price : ℝ) (second_price_A : ℝ) (second_price_B : ℝ) : 
  second_price_A = price / 2 →
  second_price_B = price - 15 →
  price = 50 →
  price + second_price_A = 75 →
  price + second_price_B = 85 →
  85 - 75 = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h4, h5]
  sorry

end promotion_savings_l132_132202


namespace solve_for_x_l132_132819

theorem solve_for_x : ∃ x : ℝ, (2/7) * (1/4) * x - 3 = 5 ∧ x = 112 :=
by {
  use 112,
  split;
  sorry
}

end solve_for_x_l132_132819


namespace apex_angle_of_third_cone_l132_132135

theorem apex_angle_of_third_cone
  (A : Point)
  (cone1 cone2 cone3 : Cone)
  (plane : Plane)
  (apex_angle1 apex_angle2 : ℝ)
  (h1 : apex_angle1 = π / 3)
  (h2 : apex_angle2 = π / 3)
  (touches_externally : touches_externally cone1 cone2)
  (touches_externally2 : touches_externally cone2 cone3)
  (touches_externally3 : touches_externally cone1 cone3)
  (touches_plane1 : touches_plane cone1 plane A)
  (touches_plane2 : touches_plane cone2 plane A)
  (touches_plane3 : touches_plane cone3 plane A)
  : apex_angle cone3 = 2 * real.arccot (2 * (sqrt 3 + sqrt 2)) ∨ 
                        apex_angle cone3 = 2 * real.arccot (2 * (sqrt 3 - sqrt 2)) :=
sorry

end apex_angle_of_third_cone_l132_132135


namespace monotonicity_of_f_range_of_a_l132_132312

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - log x

theorem monotonicity_of_f (a : ℝ) : 
  if a ≤ 0 then ∀ x > 0, derivative (f a) x < 0
  else ∀ x > 0, (x < 1 / a → derivative (f a) x < 0) ∧ (x > 1 / a → derivative (f a) x > 0)
:= sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f a x ≥ 0) → a ∈ Set.Ici 1
:= sorry

end monotonicity_of_f_range_of_a_l132_132312


namespace negation_of_existence_l132_132465

theorem negation_of_existence (h : ∃ P : ℝ × ℝ, (λ (P : ℝ × ℝ), P.1^2 + P.2^2 - 1) P ≤ 0) :
  ∀ P : ℝ × ℝ, (λ (P : ℝ × ℝ), P.1^2 + P.2^2 - 1) P > 0 :=
sorry

end negation_of_existence_l132_132465


namespace distances_perimeter_inequality_l132_132060

variable {Point Polygon : Type}

-- Definitions for the conditions
variables (O : Point) (M : Polygon)
variable (ρ : ℝ) -- perimeter of M
variable (d : ℝ) -- sum of distances to each vertex of M from O
variable (h : ℝ) -- sum of distances to each side of M from O

-- The theorem statement
theorem distances_perimeter_inequality :
  d^2 - h^2 ≥ ρ^2 / 4 :=
by
  sorry

end distances_perimeter_inequality_l132_132060


namespace holds_under_condition_l132_132700

theorem holds_under_condition (a b c : ℕ) (ha : a ≤ 10) (hb : b ≤ 10) (hc : c ≤ 10) (cond : b + 11 * c = 10 * a) :
  (10 * a + b) * (10 * a + c) = 100 * a * a + 100 * a + 11 * b * c :=
by
  sorry

end holds_under_condition_l132_132700


namespace maximum_value_of_a_l132_132276

theorem maximum_value_of_a {x y a : ℝ} (hx : x > 1 / 3) (hy : y > 1) :
  (∀ x y, x > 1 / 3 → y > 1 → 9 * x^2 / (a^2 * (y - 1)) + y^2 / (a^2 * (3 * x - 1)) ≥ 1)
  ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end maximum_value_of_a_l132_132276


namespace number_of_valid_numbers_l132_132334

theorem number_of_valid_numbers :
  ∃ n : ℕ, n = { k : ℕ | k > 1 ∧ ∃ p : ℕ, p < k ∧ prime p ∧ p * k ≤ 100 }.to_finset.card ∧ n = 34 :=
by
  sorry

end number_of_valid_numbers_l132_132334


namespace age_will_be_twice_in_two_years_l132_132932

def son_age : ℕ := 35
def age_difference : ℕ := 37

def years_to_double_age (Y : ℕ) : Prop :=
  let M := son_age + age_difference in
  M + Y = 2 * (son_age + Y)

theorem age_will_be_twice_in_two_years : years_to_double_age 2 :=
by
  sorry

end age_will_be_twice_in_two_years_l132_132932


namespace candy_chocolate_cases_l132_132900

theorem candy_chocolate_cases :
  (2 + 3 = 5) :=
begin
  -- sorry, this is straightforward arithmetic:
  exact rfl,
end

end candy_chocolate_cases_l132_132900


namespace length_of_MN_l132_132109

-- Define the points M and N
def M (a : ℝ) : ℝ × ℝ := (-2, a)
def N (a : ℝ) : ℝ × ℝ := (a, 4)

-- Define the slope condition
def slope_condition (a : ℝ) : Prop := (4 - a) / (a + 2) = -1 / 2

-- Define the distance formula
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the main theorem
theorem length_of_MN (a : ℝ) (h : slope_condition a) : distance (M a) (N a) = 6 * real.sqrt 3 := 
sorry

end length_of_MN_l132_132109


namespace vertex_A_path_length_l132_132068

noncomputable def total_path_length (AB BC CD DA : ℝ) (r1 r2 r3 : ℝ) : ℝ :=
  (π * r1 / 4) + (π * r2 / 4) + (π * r3 / 4)

theorem vertex_A_path_length 
  (AB BC CD DA : ℝ) (H_AB : AB = 4) (H_BC : BC = 5) (H_CD : CD = 4) (H_DA : DA = 5)
  (r1 := real.sqrt (AB^2 + DA^2)) (r2 := real.sqrt (AB^2 + DA^2)) (r3 := 5)
  (total_length := total_path_length AB BC CD DA r1 r2 r3) :
  total_length = (1/4) * π * (2 * real.sqrt 41 + 5) :=
sorry

end vertex_A_path_length_l132_132068


namespace airplane_shot_down_l132_132137

def P_A : ℝ := 0.4
def P_B : ℝ := 0.5
def P_C : ℝ := 0.8

def P_one_hit : ℝ := 0.4
def P_two_hit : ℝ := 0.7
def P_three_hit : ℝ := 1

def P_one : ℝ := (P_A * (1 - P_B) * (1 - P_C)) + ((1 - P_A) * P_B * (1 - P_C)) + ((1 - P_A) * (1 - P_B) * P_C)
def P_two : ℝ := (P_A * P_B * (1 - P_C)) + (P_A * (1 - P_B) * P_C) + ((1 - P_A) * P_B * P_C)
def P_three : ℝ := P_A * P_B * P_C

def total_probability := (P_one * P_one_hit) + (P_two * P_two_hit) + (P_three * P_three_hit)

theorem airplane_shot_down : total_probability = 0.604 := by
  sorry

end airplane_shot_down_l132_132137


namespace question1_question2_l132_132678

noncomputable def f (x : ℝ) : ℝ :=
  if x < -4 then -x - 9
  else if x < 1 then 3 * x + 7
  else x + 9

theorem question1 (x : ℝ) (h : -10 ≤ x ∧ x ≤ -2) : f x ≤ 1 := sorry

theorem question2 (x a : ℝ) (hx : x > 1) (h : f x > -x^2 + a * x) : a < 7 := sorry

end question1_question2_l132_132678


namespace probability_odd_divisor_of_25_factorial_l132_132116

/-- Theorem: The probability that a randomly chosen divisor of 25! is odd is 1/23. -/
theorem probability_odd_divisor_of_25_factorial :
  let num_divisors := ((22 + 1) * (10 + 1) * (6 + 1) * (3 + 1) * 2 * 2 * 2 * 2 * 2)
  let num_odd_divisors := ((10 + 1) * (6 + 1) * (3 + 1) * 2 * 2 * 2 * 2 * 2)
  (num_odd_divisors : ℚ) / num_divisors = 1 / 23 :=
begin
  sorry
end

end probability_odd_divisor_of_25_factorial_l132_132116


namespace value_of_x_l132_132979

noncomputable def log10_500 : ℝ := real.log10 500

theorem value_of_x :
  ∃ x : ℝ, 10^x * 500^x = 1000000^3 ∧ x = 18 / (1 + log10_500) := 
begin
  use 18 / (1 + log10_500),
  split,
  {
    sorry -- Skip the proof of 10^x * 500^x = 1000000^3
  },
  {
    refl -- This shows that our chosen x matches the target x in the condition
  }
end

end value_of_x_l132_132979


namespace find_x_ineq_solution_l132_132254

open Set

theorem find_x_ineq_solution :
  {x : ℝ | (x - 2) / (x - 4) ≥ 3} = Ioc 4 5 := 
sorry

end find_x_ineq_solution_l132_132254


namespace Rotary_Club_Tickets_l132_132083

noncomputable def solve_adult_tickets (n_sc n_oc n_sr n_ad : ℕ) (eggs : ℕ) 
  (omelet_sc omelet_oc omelet_sr omelet_ad extra_omelets per_omelet : ℝ) : ℕ := 
let total_omelets : ℝ := n_sc * omelet_sc + n_oc * omelet_oc + n_sr * omelet_sr + extra_omelets
let total_eggs : ℝ := total_omelets * per_omelet
let remaining_eggs : ℝ := eggs - total_eggs
let ad_tickets : ℝ := remaining_eggs / (omelet_ad * per_omelet)
in nat.floor ad_tickets

theorem Rotary_Club_Tickets :
  solve_adult_tickets 53 35 37 
                      0 
                      584
                      0.5 1 1.5 2 
                      25 
                      3 = 26 := 
by sorry

end Rotary_Club_Tickets_l132_132083


namespace cost_of_each_nose_spray_l132_132587

def total_nose_sprays : ℕ := 10
def total_cost : ℝ := 15
def buy_one_get_one_free : Bool := true

theorem cost_of_each_nose_spray :
  buy_one_get_one_free = true →
  total_nose_sprays = 10 →
  total_cost = 15 →
  (total_cost / (total_nose_sprays / 2)) = 3 :=
by
  intros h1 h2 h3
  sorry

end cost_of_each_nose_spray_l132_132587


namespace selecting_officers_l132_132126

-- Definitions for conditions
def members := 30
def officers := 4
def Andy: Type := sorry
def Bree: Type := sorry
def Carlos: Type := sorry
def Dana: Type := sorry

-- The proof statement
theorem selecting_officers:
  let num_ways := 
    (26 * 25 * 24 * 23) +
    (4 * 2 * 26 * 25) +
    (4 * 2 * 26 * 25) +
    (4 * 2)
  in
  num_ways = 369208 :=
by 
  -- This is the problem statement part, not the proof steps. 
  sorry

end selecting_officers_l132_132126


namespace kindergartners_count_l132_132490

theorem kindergartners_count :
  ∀ (first_graders : ℕ) (second_graders : ℕ) (total_students : ℕ),
  first_graders = 24 →
  second_graders = 4 →
  total_students = 42 →
  (total_students - (first_graders + second_graders) = 14) := by
  intros first_graders second_graders total_students h1 h2 h3
  rw [h1, h2, h3]
  sorry

end kindergartners_count_l132_132490


namespace absolute_value_expression_l132_132608

noncomputable def approx_pi : ℝ := 3.14159
noncomputable def five_minus_three_pi_div_two : ℝ := 5 - 3 * (approx_pi / 2)

theorem absolute_value_expression :
  abs (five_minus_three_pi_div_two) ≈ 0.287615 :=
sorry

end absolute_value_expression_l132_132608


namespace simplify_trig_expression_l132_132443

variable {A : ℝ}

-- Conditions
def cot (A : ℝ) : ℝ := cos A / sin A
def csc (A : ℝ) : ℝ := 1 / sin A
def tan (A : ℝ) : ℝ := sin A / cos A
def sec (A : ℝ) : ℝ := 1 / cos A

-- Proof problem
theorem simplify_trig_expression : (1 + cot A + csc A) * (1 + tan A - sec A) = 2 := by
  sorry

end simplify_trig_expression_l132_132443


namespace geometric_seq_second_term_l132_132103

-- Definitions
def fifth_term : ℕ → ℝ := λ n, if n = 5 then 48 else 0
def sixth_term : ℕ → ℝ := λ n, if n = 6 then 72 else 0

-- Theorem Statement
theorem geometric_seq_second_term :
  let r := sixth_term 6 / fifth_term 5,
      a := (fifth_term 5) / (r ^ 4),
      a2 := a * r in
  sixth_term 6 = 72 ∧ fifth_term 5 = 48 →
  a2 = 384 / 27 := 
begin
  sorry
end

end geometric_seq_second_term_l132_132103


namespace compute_expression_l132_132406

theorem compute_expression (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 5 / 6) :
  a^3 * b^(-4) = 82944 / 214375 :=
by
  sorry

end compute_expression_l132_132406


namespace line_through_point_parallel_l132_132090

/-
Given the point P(2, 0) and a line x - 2y + 3 = 0,
prove that the equation of the line passing through 
P and parallel to the given line is 2y - x + 2 = 0.
-/
theorem line_through_point_parallel
  (P : ℝ × ℝ)
  (x y : ℝ)
  (line_eq : x - 2*y + 3 = 0)
  (P_eq : P = (2, 0)) :
  ∃ (a b c : ℝ), a * y - b * x + c = 0 :=
sorry

end line_through_point_parallel_l132_132090


namespace point_dimension_interval_dimension_rational_dimension_cantor_set_dimension_l132_132589

-- Define Minkowski (box-counting) dimension for different objects
noncomputable def box_dimension (A : Set ℝ) : ℝ :=
  sorry  -- Placeholder for the actual definition

-- Prove the Minkowski dimensions for the given objects
theorem point_dimension :
  box_dimension {0} = 0 :=
sorry

theorem interval_dimension :
  box_dimension (Set.Icc 0 1) = 1 :=
sorry

theorem rational_dimension :
  box_dimension (Set.Icc (Set.inter (Set.Icc 0 1) Set.SymmDiff ℚ)) = 1 :=
sorry

theorem cantor_set_dimension :
  box_dimension cantor_set = Real.log 2 / Real.log 3 :=
sorry

-- Define the Cantor ternary set C(3)
noncomputable def cantor_set : Set ℝ :=
  sorry  -- Placeholder for the actual definition

-- Assertions combining the goals
asserts : Prop :=
  point_dimension ∧
  interval_dimension ∧
  rational_dimension ∧
  cantor_set_dimension

end point_dimension_interval_dimension_rational_dimension_cantor_set_dimension_l132_132589


namespace percentage_of_students_with_same_grades_l132_132540

noncomputable def same_grade_percentage (students_class : ℕ) (grades_A : ℕ) (grades_B : ℕ) (grades_C : ℕ) (grades_D : ℕ) (grades_E : ℕ) : ℚ :=
  ((grades_A + grades_B + grades_C + grades_D + grades_E : ℚ) / students_class) * 100

theorem percentage_of_students_with_same_grades :
  let students_class := 40
  let grades_A := 3
  let grades_B := 5
  let grades_C := 6
  let grades_D := 2
  let grades_E := 1
  same_grade_percentage students_class grades_A grades_B grades_C grades_D grades_E = 42.5 := by
  sorry

end percentage_of_students_with_same_grades_l132_132540


namespace dogwood_trees_final_count_l132_132876

def initial_trees := 34
def trees_planted_A := 12
def trees_planted_B := 10
def trees_planted_C := 15
def trees_planted_D := 8
def trees_planted_E := 4

def lost_transplant_C := 2
def lost_transplant_D := 1

def pest_infestation_loss := 0.10

theorem dogwood_trees_final_count :
  let total_initial := initial_trees
  let total_planted := trees_planted_A + trees_planted_B + trees_planted_C + trees_planted_D + trees_planted_E
  let total_lost_transplant := lost_transplant_C + lost_transplant_D
  let total_successfully_planted := total_planted - total_lost_transplant
  let total_pest_loss := pest_infestation_loss * total_successfully_planted
  let trees_after_pest := total_successfully_planted - total_pest_loss.toInt
  total_initial + trees_after_pest = 76 := 
  by sorry

end dogwood_trees_final_count_l132_132876


namespace not_divisible_into_different_sized_triangles_l132_132229

theorem not_divisible_into_different_sized_triangles :
  ¬ ∃ (triangles : list ℝ) (h : ∀ t ∈ triangles, t > 0),
  triangle_equilateral ∧
  has_finite_length triangles ∧
  all_triangles_equilateral triangles ∧
  all_triangles_different_sizes triangles := 
sorry

end not_divisible_into_different_sized_triangles_l132_132229


namespace knight_will_be_freed_l132_132926

/-- Define a structure to hold the state of the piles -/
structure PileState where
  pile1_magical : ℕ
  pile1_non_magical : ℕ
  pile2_magical : ℕ
  pile2_non_magical : ℕ
deriving Repr

-- Function to move one coin from pile1 to pile2
def move_coin (state : PileState) : PileState :=
  if state.pile1_magical > 0 then
    { state with
      pile1_magical := state.pile1_magical - 1,
      pile2_magical := state.pile2_magical + 1 }
  else if state.pile1_non_magical > 0 then
    { state with
      pile1_non_magical := state.pile1_non_magical - 1,
      pile2_non_magical := state.pile2_non_magical + 1 }
  else
    state -- If no coins to move, the state remains unchanged

-- The initial state of the piles
def initial_state : PileState :=
  { pile1_magical := 0, pile1_non_magical := 49, pile2_magical := 50, pile2_non_magical := 1 }

-- Check if the knight can be freed (both piles have the same number of magical or non-magical coins)
def knight_free (state : PileState) : Prop :=
  state.pile1_magical = state.pile2_magical ∨ state.pile1_non_magical = state.pile2_non_magical

noncomputable def knight_can_be_freed_by_25th_day : Prop :=
  exists n : ℕ, n ≤ 25 ∧ knight_free (Nat.iterate move_coin n initial_state)

theorem knight_will_be_freed : knight_can_be_freed_by_25th_day :=
  sorry

end knight_will_be_freed_l132_132926


namespace tangent_ratio_const_l132_132177

-- Declaration of the general setup
variables {A B X P : Point} (ω : Circle)

-- Conditions provided
axiom A_not_eq_B : A ≠ B
axiom P_on_omega : P ∈ ω
axiom P_not_eq_A : P ≠ A
axiom P_not_eq_B : P ≠ B
axiom A_B_diameter : is_diameter A B ω
axiom X_on_AB : collinear A X B

-- The main statement we need to prove
theorem tangent_ratio_const :
  ∀ (P : Point), P ∈ ω ∧ P ≠ A ∧ P ≠ B →
  (tan (angle A P X)) / (tan (angle P A X)) = (dist A X) / (dist X B) :=
begin
  -- Proof to be filled in
  sorry
end

end tangent_ratio_const_l132_132177


namespace solution_set_of_inequality_l132_132612

def g (x : ℝ) : ℝ := (3 * x - 4) * (x + 2) / (x - 1)

theorem solution_set_of_inequality :
  {x : ℝ | g x ≤ 0} = {x : ℝ | x ∈ (-∞, -2] ∪ [1, (4 : ℝ) / 3]} :=
by
  sorry

end solution_set_of_inequality_l132_132612


namespace main_l132_132383

noncomputable def walking_condition_given_paths
  (rkenny_walking_speed : ℝ)
  (rjenny_walking_speed : ℝ)
  (paths_distance : ℝ)
  (building_diameter : ℝ)
  (initial_distance_blocked : ℝ) : ℝ :=
  let kenny_position := λ t : ℝ, -150 + rkenny_walking_speed * t
  let jenny_position := λ t : ℝ, -150 + rjenny_walking_speed * t
  let can_see_each_other := λ t : ℝ, 
  ⟦(kenny_position t + 100)^2 + ((building_diameter/2)^2 - 100)^2 = building_diameter^2⟧ in
  let t := Real.prod_sigma (walk_scenario t) 
  ⟦kenny_position (t, walks_parallel_to each other center_of_paths 100) , jenny_position (t,1,val 50) ⟧ in
   Real (sum (num_denomin; Normalizes ratios 200, 1))

theorem main : walking_condition_given_paths 4 2 100 100 300 = 201 := by
   sorry

end main_l132_132383


namespace solve_a1_l132_132287

noncomputable def seq_a : ℕ → ℝ := sorry
noncomputable def seq_b : ℕ → ℝ := sorry

axiom condition1 (n : ℕ) (hn : n ≥ 1) : seq_a (n + 1) + seq_b (n + 1) = (seq_a n + seq_b n) / 2
axiom condition2 (n : ℕ) (hn : n ≥ 1) : seq_a (n + 1) * seq_b (n + 1) = real.sqrt (seq_a n * seq_b n)
axiom condition3 : seq_b 2016 = 1
axiom condition4 : 0 < seq_a 1

theorem solve_a1 : seq_a 1 = 2^2015 := sorry

end solve_a1_l132_132287


namespace product_of_roots_l132_132226

open Real

theorem product_of_roots : (sqrt (Real.exp (1 / 4 * log (16)))) * (sqrt (Real.exp (1 / 6 * log (64)))) = 4 :=
by
  -- sorry is used to bypass the actual proof implementation
  sorry

end product_of_roots_l132_132226


namespace scaled_tile_height_l132_132014

theorem scaled_tile_height
  (width_original height_original width_new : ℝ)
  (h_scale_factor : width_new / width_original = 4)
  (h_dimensions : width_original = 3 ∧ height_original = 4 ∧ width_new = 12) :
  let height_new := height_original * 4 in
  height_new = 16 := by
sorry

end scaled_tile_height_l132_132014


namespace sqrt_n_eq_prod_sin_l132_132432

theorem sqrt_n_eq_prod_sin (n : ℕ) (h : n > 0) : 
  real.sqrt n = 2^(n-1) * ∏ k in finset.range (n-1), real.sin (k * real.pi / (2 * n)) :=
by
  sorry

end sqrt_n_eq_prod_sin_l132_132432


namespace solve_division_problem_l132_132818

-- Problem Conditions
def division_problem : ℚ := 0.25 / 0.005

-- Proof Problem Statement
theorem solve_division_problem : division_problem = 50 := by
  sorry

end solve_division_problem_l132_132818


namespace planet_unobserved_l132_132006

theorem planet_unobserved (n : ℕ) (h : n = 2015) 
  (dist : Fin n → Fin n → ℝ) 
  (h_distinct : ∀ i j, i ≠ j → dist i j ≠ dist j i) 
  (closest_planet : Fin n → Fin n) 
  (observes_closest : ∀ i, dist i (closest_planet i) < dist i j ∀ j ≠ closest_planet i) :
  ∃ p : Fin n, ∀ q, closest_planet q ≠ p :=
by
  sorry

end planet_unobserved_l132_132006


namespace sum_of_third_row_from_top_l132_132360

-- Define the 17x17 grid and placements
def spiral_grid (n : ℕ) : ℕ × ℕ → ℕ :=
  λ (i j : ℕ), sorry -- Assume the function that provides integer placement in the grid (to be defined)

theorem sum_of_third_row_from_top :
  let grid_size := 17
  let max_value := grid_size ^ 2
  let third_row := 3
  let middle := (grid_size // 2, grid_size // 2)

  ∀ (sr_min sr_max : ℕ),
    (sr_min = spiral_grid max_value (third_row - 1, 0)) ∧
    (sr_max = spiral_grid max_value (third_row - 1, grid_size - 1)) →
    sr_min + sr_max = 526 :=
by
  intros
  sorry

end sum_of_third_row_from_top_l132_132360


namespace geometric_seq_second_term_l132_132102

-- Definitions
def fifth_term : ℕ → ℝ := λ n, if n = 5 then 48 else 0
def sixth_term : ℕ → ℝ := λ n, if n = 6 then 72 else 0

-- Theorem Statement
theorem geometric_seq_second_term :
  let r := sixth_term 6 / fifth_term 5,
      a := (fifth_term 5) / (r ^ 4),
      a2 := a * r in
  sixth_term 6 = 72 ∧ fifth_term 5 = 48 →
  a2 = 384 / 27 := 
begin
  sorry
end

end geometric_seq_second_term_l132_132102


namespace sum_infinite_series_eq_l132_132268

theorem sum_infinite_series_eq {x : ℝ} (hx : |x| < 1) :
  (∑' n : ℕ, (n + 1) * x^n) = 1 / (1 - x)^2 :=
by
  sorry

end sum_infinite_series_eq_l132_132268


namespace first_player_optimal_strategy_l132_132492

def tokens : List ℕ := [7, 2, 3, 8, 9, 4, 1, 6, 3, 2, 4, 7, 1]

def optimal_strategy_exists : Prop :=
  ∃ strategy, ∀ (tokens : List ℕ), (∀ turn (remaining_tokens : List ℕ), turn mod 2 = 1 → (strategy turn remaining_tokens).sum ≥ (alternative_strategy turn remaining_tokens).sum) ∧
  ∀ (turn remaining_tokens : List ℕ), turn mod 2 = 0 → (strategy turn remaining_tokens).sum < (alternative_strategy turn remaining_tokens).sum

theorem first_player_optimal_strategy : optimal_strategy_exists :=
  sorry

end first_player_optimal_strategy_l132_132492


namespace second_number_is_40_l132_132475

-- Defining the problem
theorem second_number_is_40
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a = (3/4 : ℚ) * b)
  (h3 : c = (5/4 : ℚ) * b) :
  b = 40 :=
sorry

end second_number_is_40_l132_132475


namespace tangent_line_perpendicular_to_given_line_l132_132711

theorem tangent_line_perpendicular_to_given_line :
  ∃ l : ℝ → ℝ, (∀ x : ℝ, l x = 4 * x - 3) ∧ (∀ p : ℝ × ℝ, p ∈ (λ (x : ℝ), (x, x^4)) '' univ → (∃ x : ℝ, p = (x, x^4) ∧ tangent_to_curve (x^4) l ∧ perpendicular_to_line l (λ x, -1/4 * x + 502.25))) :=
by
  sorry

end tangent_line_perpendicular_to_given_line_l132_132711


namespace termination_and_uniqueness_l132_132061

noncomputable def alpha : ℝ := (1 + Real.sqrt 5) / 2

def weight (a : ℤ → ℕ) : ℝ := ∑ i in Finset.range (1000), a i * alpha^i

theorem termination_and_uniqueness (a : ℤ → ℕ) :
  ∀ actions : list ((ℤ → ℕ) → (ℤ → ℕ)),
    let final_state := (List.foldl (λ s f, f s) a actions) in
    (∀ i, final_state (i - 1) < 2 ∧ final_state (i - 1) = 0 ∨ 
          final_state i < 2 ∧ final_state i = 0 ∨ 
          final_state (i + 1) < 1 ∧ final_state i ∉ ℕ) ∧
    (∀ actions', 
      (List.foldl (λ s f, f s) a actions') = final_state) :=
begin
  sorry
end

end termination_and_uniqueness_l132_132061


namespace parallel_lines_have_equal_slopes_l132_132345

theorem parallel_lines_have_equal_slopes (a : ℝ) :
  (∃ a : ℝ, (∀ y : ℝ, 2 * a * y - 1 = 0) ∧ (∃ x y : ℝ, (3 * a - 1) * x + y - 1 = 0) 
  → (∃ a : ℝ, (1 / (2 * a)) = - (3 * a - 1))) 
→ a = 1/2 :=
by
  sorry

end parallel_lines_have_equal_slopes_l132_132345


namespace price_of_uniform_correct_l132_132931

def price_of_uniform (U : ℝ) : Prop :=
  let one_year_compensation := 500
  let nine_months_fraction := 3 / 4
  let nine_months_service_amount := nine_months_fraction * one_year_compensation
  let amount_received := 250
  U + amount_received = nine_months_service_amount

theorem price_of_uniform_correct (U : ℝ) : price_of_uniform U → U = 125 :=
by 
  intro h
  unfold price_of_uniform at h
  linarith [h]
  sorry

end price_of_uniform_correct_l132_132931


namespace radius_increase_percentage_l132_132122

theorem radius_increase_percentage (r : ℝ) (x : ℝ) :
  let A := 4 * Real.pi * r^2
  let A' := 4 * Real.pi * r^2 * (1 + x / 100)^2
  A' = 1.2100000000000002 * A →
  x ≈ 10 :=
by
  sorry

end radius_increase_percentage_l132_132122


namespace evaluate_difference_of_squares_l132_132987
-- Import necessary libraries

-- Define the specific values for a and b
def a : ℕ := 72
def b : ℕ := 48

-- State the theorem to be proved
theorem evaluate_difference_of_squares : a^2 - b^2 = (a + b) * (a - b) ∧ (a + b) * (a - b) = 2880 := 
by
  -- The proof would go here but should be omitted as per directions
  sorry

end evaluate_difference_of_squares_l132_132987


namespace unique_tiling_100x100_l132_132194

-- Define the concept of the frame of an n × n square
def frame (n : ℕ) : set (ℤ × ℤ) :=
  {p | (p.1 = 0 ∨ p.1 = n - 1 ∨ p.2 = 0 ∨ p.2 = n - 1) ∧
       0 ≤ p.1 ∧ p.1 < n ∧ 0 ≤ p.2 ∧ p.2 < n}

-- Define the specific frame problem for a 100 × 100 square
def square_100x100 : set (ℤ × ℤ) := frame 100

-- The main theorem statement that covers the problem's question and conditions
theorem unique_tiling_100x100 :
  ∃! (cover : set (set (ℤ × ℤ))), (∀ c ∈ cover, c = frame 50) ∧
                                   (⋃ c ∈ cover, c = square_100x100) ∧
                                   (∀ c₁ c₂ ∈ cover, c₁ ≠ c₂ → c₁ ∩ c₂ = ∅) :=
by sorry

end unique_tiling_100x100_l132_132194


namespace sum_of_first_six_terms_l132_132831

def geometric_seq_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem sum_of_first_six_terms (a : ℕ) (r : ℕ) (h1 : r = 2) (h2 : a * (1 + r + r^2) = 3) :
  geometric_seq_sum a r 6 = 27 :=
by
  sorry

end sum_of_first_six_terms_l132_132831


namespace xiaoming_ticket_lineup_l132_132534
-- Import all necessary libraries from Mathlib

open Finset

-- Define the Lean 4 statement of the problem 
theorem xiaoming_ticket_lineup : 
  let n := 6 in
  let count_ways (n : ℕ) := (n.choose 4) * ((n + 1)!)
  in count_ways n = 10800 :=
by
  sorry

end xiaoming_ticket_lineup_l132_132534


namespace probability_x_add_2y_lt_4_l132_132933

theorem probability_x_add_2y_lt_4 :
  let square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3
  let px2y := (λ (p : ℝ × ℝ), p.1 + 2 * p.2 < 4)
  (volume (square ∩ {p | px2y p})) / (volume square) = 3 / 4 :=
sorry

end probability_x_add_2y_lt_4_l132_132933


namespace intersection_M_N_l132_132320

def M (x : ℝ) : Prop := x^2 ≥ x

def N (x : ℝ) (y : ℝ) : Prop := y = 3^x + 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | ∃ y : ℝ, N x y ∧ y > 1} = {x : ℝ | x > 1} :=
by {
  sorry
}

end intersection_M_N_l132_132320


namespace find_principal_l132_132833

noncomputable def principal_amount (P : ℝ) : Prop :=
  let r := 0.05
  let t := 2
  let SI := P * r * t
  let CI := P * (1 + r) ^ t - P
  CI - SI = 15

theorem find_principal : principal_amount 6000 :=
by
  simp [principal_amount]
  sorry

end find_principal_l132_132833


namespace mod_13_remainder_l132_132036

theorem mod_13_remainder (b : ℤ) : (b ≡ 6 [MOD 13]) ↔ (b = (2⁻¹ + 3⁻¹ + 5⁻¹)⁻¹ % 13) := 
by sorry

end mod_13_remainder_l132_132036


namespace number_of_students_in_both_clubs_l132_132222

theorem number_of_students_in_both_clubs (
  total_students : ℕ,
  drama_club : ℕ,
  science_club : ℕ,
  at_least_one_club : ℕ)
  (h1 : total_students = 300)
  (h2 : drama_club = 80)
  (h3 : science_club = 130)
  (h4 : at_least_one_club = 190) : 
  (drama_club + science_club - at_least_one_club = 20) :=
by {
  sorry
}

end number_of_students_in_both_clubs_l132_132222


namespace find_f_2_l132_132839

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_odd (x : ℝ) : f(x) + f(-x) = 0
axiom f_period (x : ℝ) : f(3/4 + x) = f(3/4 - x)
axiom f_one : f 1 = 3

theorem find_f_2 : f 2 = -3 := by
  sorry

end find_f_2_l132_132839


namespace nine_digit_number_count_l132_132626

/--
There are 504 nine-digit numbers where:
- each digit from 1 to 9 occurs exactly once,
- the digits 1, 2, 3, 4, 5 appear in ascending order, and
- the digit 6 appears before the digit 1.
-/
theorem nine_digit_number_count : 
  ∃! (n : ℕ), n = 504 ∧ 
    (∀ digits : Finset ℕ, 
      digits = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      sorted (λ a b, a < b) [1, 2, 3, 4, 5] ∧
      6 < 1) := sorry

end nine_digit_number_count_l132_132626


namespace real_root_probability_l132_132662

theorem real_root_probability :
  let a b : ℤ := by exact a, b
  a ∈ {-1, 0, 1, 2} ∧ b ∈ {-1, 0, 1, 2} ∧
  let eq_has_real_roots : Prop := ∃ x : ℝ, a * x^2 + 2 * x + b = 0
in
  (∑ ab in ({(a, b) | a ∈ {-1, 0, 1, 2} ∧ b ∈ {-1, 0, 1, 2}} : Finset (ℤ × ℤ)).filter (λ (ab : ℤ × ℤ), ab.1 * 4 * (0 : ℤ) + (2 : ℤ) + ab.2 ≤ (4 : ℤ)).card : ℚ) / 16 = 13 / 16 := 
begin
  sorry
end

end real_root_probability_l132_132662


namespace compute_expression_l132_132407

theorem compute_expression (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 5 / 6) :
  a^3 * b^(-4) = 82944 / 214375 :=
by
  sorry

end compute_expression_l132_132407


namespace minimum_moves_l132_132901

def valid_move (p1 p2 : ℕ × ℕ) : Prop := 
  let xdiff := (p2.1 - p1.1).nat_abs
  let ydiff := (p2.2 - p1.2).nat_abs
  (xdiff = 1 ∧ ydiff = 0) ∨ (xdiff = 0 ∧ ydiff = 1)

def valid_move_sequence (seq : List (ℕ × ℕ)) : Prop :=
  seq.head = (0, 0) ∧ 
    (seq.tail.head?).isSome ∧ 
    seq.tail.head? = (1056, 1007) ∧ 
    (∀ i, i < seq.length - 1 → valid_move (seq.nth i).get (seq.nth (i + 1)).get) ∧ 
    (∀ i, i < seq.length - 2 → ¬((seq.nth i).get.1 = (seq.nth (i+1)).get.1 ∧ (seq.nth (i+1)).get.1 = (seq.nth (i+2)).get.1) ∧ 
                          ¬((seq.nth i).get.2 = (seq.nth (i+1)).get.2 ∧ (seq.nth (i+1)).get.2 = (seq.nth (i+2)).get.2))

theorem minimum_moves : ∃ seq, valid_move_sequence seq ∧ seq.length = 2111 := 
by
  sorry

end minimum_moves_l132_132901


namespace ratio_AB_WX_l132_132774

theorem ratio_AB_WX (AB WX P Q W : ℝ) (h₀ : WX < AB)
  (h₁ : let α := (sqrt 2 - 1) in α < 1 ∧ α > 0 ∧ WX = α * AB)
  (h₂ : P = 1 ∧ Q = 1 ∧ W = 1) :
  (AB / WX = sqrt 2 + 1) :=
sorry

end ratio_AB_WX_l132_132774


namespace range_f_on_interval_l132_132672

def f (x : ℝ) (m : ℝ) : ℝ := 4*x^2 - m*x + 1

theorem range_f_on_interval :
  (∀ x : ℝ, (x ≤ (-2) → f x m ≤ f (-2) m) ∧ (-2 ≤ x → f (-2) m ≤ f x m)) →
  m = -16 →
  set.range (λ x, f x (-16)) ⟨1, 2⟩ = set.Icc 21 49 := 
by
  intro h₁ h₂
  sorry

end range_f_on_interval_l132_132672


namespace fraction_of_x_l132_132708

theorem fraction_of_x (w x y f : ℝ) (h1 : 2 / w + f * x = 2 / y) (h2 : w * x = y) (h3 : (w + x) / 2 = 0.5) : f = 2 / x - 2 := 
sorry

end fraction_of_x_l132_132708


namespace triangle_b_ge_sqrt2_l132_132653

theorem triangle_b_ge_sqrt2 {a b c : ℝ} (h1 : a ≤ b) (h2 : b ≤ c) (h_area : (1/2) * a * b * sin (γ : ℝ) = 1) : b ≥ sqrt 2 := by
  sorry

end triangle_b_ge_sqrt2_l132_132653


namespace calculate_expression_l132_132953

theorem calculate_expression :
  let s1 := 3 + 6 + 9
  let s2 := 4 + 8 + 12
  s1 = 18 → s2 = 24 → (s1 / s2 + s2 / s1) = 25 / 12 :=
by
  intros
  sorry

end calculate_expression_l132_132953


namespace proof_by_contradiction_assumption_l132_132801

theorem proof_by_contradiction_assumption (a b : ℝ) (h : a > b) : ¬(a ≤ b) → false :=
by 
  assume h₁ : a ≤ b,
  sorry

end proof_by_contradiction_assumption_l132_132801


namespace xy_yz_xz_expression_l132_132120

noncomputable def express_xy_yz_xz (a b c x y z : ℝ) (h1 : x^2 + x * y + y^2 = a^2)
  (h2 : y^2 + y * z + z^2 = b^2) (h3 : x^2 + x * z + z^2 = c^2) : ℝ :=
4 * Real.sqrt (((a + b + c) / 2) * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) / 3)

theorem xy_yz_xz_expression (a b c x y z : ℝ)
  (h1 : x^2 + x * y + y^2 = a^2)
  (h2 : y^2 + y * z + z^2 = b^2)
  (h3 : x^2 + x * z + z^2 = c^2) :
  xy + yz + xz = (express_xy_yz_xz a b c x y z h1 h2 h3) :=
begin
  sorry
end

end xy_yz_xz_expression_l132_132120


namespace range_f_l132_132123

def f (x : ℝ) : ℝ := real.log ((x + 1) / (x - 1))

theorem range_f : set.range f = set.union set.Iio 0 set.Ioi 0 := sorry

end range_f_l132_132123


namespace relationship_among_a_b_c_l132_132766

noncomputable def a : ℝ := (0.6:ℝ) ^ (0.2:ℝ)
noncomputable def b : ℝ := (0.2:ℝ) ^ (0.2:ℝ)
noncomputable def c : ℝ := (0.2:ℝ) ^ (0.6:ℝ)

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  -- The proof can be added here if needed
  sorry

end relationship_among_a_b_c_l132_132766


namespace line_through_point_intersecting_circle_eq_l132_132552

theorem line_through_point_intersecting_circle_eq :
  ∃ k l : ℝ, (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0) ∧ 
    ∀ L : ℝ × ℝ,  
      (L = (-3, -3)) ∧ (x^2 + y^2 + 4*y - 21 = 0) → 
      (L = (-3,-3) → (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0)) := 
sorry

end line_through_point_intersecting_circle_eq_l132_132552


namespace max_integers_greater_than_18_l132_132869

theorem max_integers_greater_than_18 :
  ∀ (a : ℕ → ℤ), (∑ i in finset.range 10, a i = 30) →
  (∀ i ∈ finset.range 10, a i > 18 → a i ≥ 19) →
  ∃ (k : ℕ), k ≤ 10 ∧ ∀ (n : ℕ), (n > k → ∑ i in finset.range n, 19 * (a i / 19) + ∑ j in finset.range (10 - (a i / 19)), a j = 30) ∨ n = 9 :=
by sorry

end max_integers_greater_than_18_l132_132869


namespace find_X_l132_132053

variable {α : Type} -- considering sets of some type α
variables (A B X : Set α)

theorem find_X (h1 : A ∩ X = B ∩ X ∧ B ∩ X = A ∩ B)
               (h2 : A ∪ B ∪ X = A ∪ B) : X = A ∩ B :=
by {
    sorry
}

end find_X_l132_132053


namespace banana_popsicles_count_l132_132376

theorem banana_popsicles_count 
  (grape_popsicles cherry_popsicles total_popsicles : ℕ)
  (h1 : grape_popsicles = 2)
  (h2 : cherry_popsicles = 13)
  (h3 : total_popsicles = 17) :
  total_popsicles - (grape_popsicles + cherry_popsicles) = 2 := by
  sorry

end banana_popsicles_count_l132_132376


namespace coordinates_P_2008_l132_132371

theorem coordinates_P_2008 :
  let P0 := (1:ℝ, 0:ℝ)
  let rotate (θ : ℝ) (P : ℝ × ℝ) := 
    (P.1 * Real.cos θ - P.2 * Real.sin θ, P.1 * Real.sin θ + P.2 * Real.cos θ)
  let extend (P : ℝ × ℝ) (k : ℝ) := 
    (P.1 * k, P.2 * k)
  let angle := Real.pi / 4
  let rec point (n : ℕ) : ℝ × ℝ :=
    if n = 0 then P0
    else if n % 2 = 0 then extend (point (n - 1)) 2
    else rotate angle (point (n - 1))
  point 2008 = (-2^1004, 0) :=
by
  let P0 := (1, 0)
  let rotate (θ : ℝ) (P : ℝ × ℝ) := 
    (P.1 * Real.cos θ - P.2 * Real.sin θ, P.1 * Real.sin θ + P.2 * Real.cos θ)
  let extend (P : ℝ × ℝ) (k : ℝ) := 
    (P.1 * k, P.2 * k)
  let angle := Real.pi / 4
  let rec point (n : ℕ) : ℝ × ℝ :=
    if n = 0 then P0
    else if n % 2 = 0 then extend (point (n - 1)) 2
    else rotate angle (point (n - 1))
  have h : point 2008 = (-2^1004, 0) := sorry
  exact h

end coordinates_P_2008_l132_132371


namespace number_of_integers_satisfying_inequalities_l132_132693

theorem number_of_integers_satisfying_inequalities :
  {n : ℕ | 300 < n^2 ∧ n^2 < 1200}.to_finset.card = 17 :=
by
  sorry

end number_of_integers_satisfying_inequalities_l132_132693


namespace old_edition_pages_l132_132191

theorem old_edition_pages :
  ∃ (x : ℕ), let y := 3 * x^2 - 90 in
    450 = 2 * x - 230 ∧
    y ≥ ((11:ℕ) * x / 10) ∧
    y.isNat :=
by
  -- Proof will be added here
  sorry

end old_edition_pages_l132_132191


namespace find_special_integers_l132_132994

open Nat

def has_exact_divisors (m : ℕ) (d : ℕ) : Prop :=
  d ∈ divisors m ∧ divisors m = List.range (d+1)

theorem find_special_integers :
  {n : ℕ | n ≥ 1 ∧ ∃ d, has_exact_divisors (2^n - 1) d ∧ d = n} = {1, 2, 4, 6, 8, 16, 32} :=
sorry

end find_special_integers_l132_132994


namespace repeating_pattern_250th_letter_l132_132147

theorem repeating_pattern_250th_letter :
  (let f : ℕ → char := λ n, "ABC".get ⟨(n % 3), sorry⟩ in f 249) = 'A' :=
sorry

end repeating_pattern_250th_letter_l132_132147


namespace geometric_seq_second_term_l132_132104

-- Definitions
def fifth_term : ℕ → ℝ := λ n, if n = 5 then 48 else 0
def sixth_term : ℕ → ℝ := λ n, if n = 6 then 72 else 0

-- Theorem Statement
theorem geometric_seq_second_term :
  let r := sixth_term 6 / fifth_term 5,
      a := (fifth_term 5) / (r ^ 4),
      a2 := a * r in
  sixth_term 6 = 72 ∧ fifth_term 5 = 48 →
  a2 = 384 / 27 := 
begin
  sorry
end

end geometric_seq_second_term_l132_132104


namespace domain_of_function_l132_132836

noncomputable def isDomain (x : ℝ) : Prop :=
  (x - 2 ≥ 0) ∧ (x + 5 ≥ 0)

theorem domain_of_function : ∀ x : ℝ, isDomain x ↔ x ∈ set.Ici 2 :=
by sorry

end domain_of_function_l132_132836


namespace age_difference_l132_132549

theorem age_difference (A B C : ℕ) (h1 : B = 10) (h2 : B = 2 * C) (h3 : A + B + C = 27) : A - B = 2 :=
 by
  sorry

end age_difference_l132_132549


namespace cottage_arrangement_is_valid_l132_132817

-- Define the properties and distances around the circular path
def valid_cottage_arrangement (distances : List ℕ) : Prop :=
  distances.length = 6 ∧
  distances.sum = 27 ∧
  (∀ i j, i ≠ j → (1 ≤ |distances.nthLe i sorry - distances.nthLe j sorry| ∧ |distances.nthLe i sorry - distances.nthLe j sorry| ≤ 26))

-- Example distances between cottages as given in the solution
def distances : List ℕ := [1, 1, 4, 4, 3, 14]

-- The proof problem: the given cottage arrangement satisfies the conditions
theorem cottage_arrangement_is_valid : valid_cottage_arrangement distances :=
  sorry

end cottage_arrangement_is_valid_l132_132817


namespace max_distance_from_point_to_line_l132_132353

theorem max_distance_from_point_to_line (θ m : ℝ) :
  let P := (Real.cos θ, Real.sin θ)
  let d := (P.1 - m * P.2 - 2) / Real.sqrt (1 + m^2)
  ∃ (θ m : ℝ), d ≤ 3 := sorry

end max_distance_from_point_to_line_l132_132353


namespace sequence_10th_term_eq_1_div_19_l132_132739

-- The sequence definition as given in the problem
def a : ℕ → ℝ
| 0 := 1
| (n + 1) := a n / (1 + 2 * a n)

-- The statement to prove
theorem sequence_10th_term_eq_1_div_19 :
  a 9 = 1 / 19 := sorry

end sequence_10th_term_eq_1_div_19_l132_132739


namespace crayons_left_l132_132426

theorem crayons_left (initial_crayons erasers_left more_crayons_than_erasers : ℕ)
    (H1 : initial_crayons = 531)
    (H2 : erasers_left = 38)
    (H3 : more_crayons_than_erasers = 353) :
    (initial_crayons - (initial_crayons - (erasers_left + more_crayons_than_erasers)) = 391) :=
by 
  sorry

end crayons_left_l132_132426


namespace no_natural_number_exists_l132_132625

theorem no_natural_number_exists 
  (n : ℕ) : ¬ ∃ x y : ℕ, (2 * n * (n + 1) * (n + 2) * (n + 3) + 12) = x^2 + y^2 := 
by sorry

end no_natural_number_exists_l132_132625


namespace problem_probability_log_floor_l132_132435

open Real

noncomputable def probability_log_floor_eq (x y : ℝ) : ℝ :=
if Hx : 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 then
  ∑' (n : ℤ) in {n | n < 0}.to_finset, (0.9 * 10^n)^2
else
  0

theorem problem_probability_log_floor :
  let pt := probability_log_floor_eq (Real.uniform_random 0 1) (Real.uniform_random 0 1)
  in pt = 0.00818 :=
by
  sorry

end problem_probability_log_floor_l132_132435


namespace angle_of_inclination_is_30_degrees_l132_132084

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := x - sqrt 3 * y + 5 = 0

-- Define the slope of the line
def slope_of_line := (1 : ℝ) / sqrt 3

-- Define the angle of inclination θ that we need to prove as 30 degrees (π / 6 radians)
def angle_of_inclination : ℝ := Real.arctan slope_of_line

theorem angle_of_inclination_is_30_degrees :
  angle_of_inclination = Real.pi / 6 := by
  sorry

end angle_of_inclination_is_30_degrees_l132_132084


namespace count_numbers_as_sum_of_primes_l132_132336

open Nat

-- Set definition based on the given conditions
def setN : Set ℕ := { n | ∃ k : ℕ, n = 5 + 12 * k ∧ n ≤ 100 }

-- Prime number predicate
def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Check if a number can be written as the sum of two primes
def is_sum_of_two_primes (n : ℕ) : Prop := ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n

-- The main theorem statement
theorem count_numbers_as_sum_of_primes : 
  { n | n ∈ setN ∧ is_sum_of_two_primes n }.card = 6 :=
by
  sorry

end count_numbers_as_sum_of_primes_l132_132336


namespace largest_divisor_of_m_l132_132712

theorem largest_divisor_of_m (m : ℕ) (hm : m > 0) (h : 54 ∣ m^2) : 18 ∣ m :=
sorry

end largest_divisor_of_m_l132_132712


namespace defendant_C_guilty_l132_132877

-- Definitions for the defendants
inductive Defendant
| A
| B
| C

open Defendant

-- Conditions of the problem encoded as a Lean predicate
def accuses (accuser accused : Defendant) : Prop := sorry  -- Placeholder for accusation logic
def tellsTruth (def : Defendant) : Prop := sorry  -- Placeholder for truth-telling logic

-- Problem statement: if certain conditions hold, then 'C' is guilty
theorem defendant_C_guilty 
  (h1 : tellsTruth A)  -- Condition that defendant A is the only one telling the truth initially
  (h2 : accuses A B ∨ accuses A C)  -- A accuses B or C
  (h3 : ∀ (accuser accused : Defendant),
      accuses accuser accused → accuses accuser (otherDefendant accuser accused))  -- Swap condition
  (h4 : ∀ (accuser accused : Defendant), 
      accused ≠ B → tellsTruth B → False) :  -- Condition that B tells the truth if accusations are swapped
  accuses A C →  -- If A accuses C
  ¬tellsTruth B ∧ ¬tellsTruth C  -- Conclusion that C is guilty
:= sorry

end defendant_C_guilty_l132_132877


namespace lasagna_pieces_l132_132417

-- Definition of the conditions
def manny_piece := 1
def aaron_piece := 0
def kai_piece := 2 * manny_piece
def raphael_piece := 0.5 * manny_piece
def lisa_piece := 2 + 0.5 * raphael_piece

-- The main theorem statement proving the total number of pieces
theorem lasagna_pieces : 
  manny_piece + aaron_piece + kai_piece + raphael_piece + lisa_piece = 6 :=
by
  -- Proof is omitted
  sorry

end lasagna_pieces_l132_132417


namespace second_draw_white_prob_l132_132354

-- Define the initial conditions
def total_balls := 20
def red_balls_init := 10
def white_balls_init := 10

-- Define the event of drawing a red ball first and then a white ball
def draws :=
  let first_draw_red := red_balls_init > 0 in
  let balls_left := total_balls - 1 in
  let red_balls_after_first_draw := red_balls_init - 1 in
  let white_balls_left := white_balls_init in
  let second_draw_white_prob := white_balls_left / balls_left in
  second_draw_white_prob

-- Define the theorem to be proven
theorem second_draw_white_prob :
  draws = (10 / 19) := 
by
  -- Proof goes here
  sorry

end second_draw_white_prob_l132_132354


namespace gingerbreads_per_tray_l132_132245

theorem gingerbreads_per_tray (x : ℕ) (h : 4 * x + 3 * 20 = 160) : x = 25 :=
by
  sorry

end gingerbreads_per_tray_l132_132245


namespace fractional_expr_is_B_l132_132944

noncomputable def exprA : ℚ := (8 * x) / (3 * Real.pi)
noncomputable def exprB : ℚ := (x ^ 2 - y ^ 2) / (x - y)
noncomputable def exprC : ℚ := (x - y) / 5
noncomputable def exprD : ℚ := 5 / 8

theorem fractional_expr_is_B : 
  is_fractional exprB := by
  sorry

end fractional_expr_is_B_l132_132944


namespace six_pointed_star_coloring_l132_132129

noncomputable def six_pointed_star (n : ℕ) : Prop :=
  ∀ (coloring : Fin n → Bool), ∃ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (coloring a = coloring b) ∧ (coloring b = coloring c) ∧
    is_equilateral a b c

-- Prove that for n = 13 in the context of a regular six-pointed star.
theorem six_pointed_star_coloring :
  six_pointed_star 13 :=
sorry

end six_pointed_star_coloring_l132_132129


namespace alfred_bill_days_l132_132898

-- Definitions based on conditions
def combined_work_rate := 1 / 24
def alfred_to_bill_ratio := 2 / 3

-- Theorem statement
theorem alfred_bill_days (A B : ℝ) (ha : A = alfred_to_bill_ratio * B) (hcombined : A + B = combined_work_rate) : 
  A = 1 / 60 ∧ B = 1 / 40 :=
by
  sorry

end alfred_bill_days_l132_132898


namespace mixed_price_calc_add_candy_a_to_mix_equal_weight_from_each_box_l132_132722

-- Problem 1
theorem mixed_price_calc (a b : ℕ) (m n : ℕ) (h_a : a = 30) (h_b : b = 25) 
                         (h_m : m = 30) (h_n : n = 20) :
  (a * m + b * n) / (m + n) = 28 := sorry

-- Problem 2
theorem add_candy_a_to_mix (a : ℕ) (x : ℝ) (h_a : a = 30) (price_mixed : ℝ) 
                           (weight_mixed : ℕ) (price_increase : ℝ) 
                           (h_price_mixed : price_mixed = 24)
                           (h_weight_mixed : weight_mixed = 100)
                           (h_price_increase : price_increase = 0.15) :
  let price_new := price_mixed * (1 + price_increase)
  (a * x + price_mixed * weight_mixed) / (x + weight_mixed) = price_new :=
  by
  let price_new := price_mixed * (1 + price_increase)
  have h_price_new : price_new = 24 * 1.15 := rfl
  exact sorry

-- Problem 3
theorem equal_weight_from_each_box (a b : ℕ) (m n : ℕ) (y : ℝ)
                                   (h_a : a = 30) (h_b : b = 25)
                                   (h_m : m = 40) (h_n : n = 60)
                                   (h_condition : (b * y + a * (40 - y)) / 40 = (a * y + b * (60 - y)) / 60) :
  y = 24 := sorry

end mixed_price_calc_add_candy_a_to_mix_equal_weight_from_each_box_l132_132722


namespace price_of_scooter_l132_132809

variable P : ℝ
variable h : 0.20 * P = 240

theorem price_of_scooter : P = 1200 :=
by
  sorry

end price_of_scooter_l132_132809


namespace domain_of_f_l132_132241

noncomputable def f (x : ℝ) : ℝ := (2*x^4 - 8*x^3 + 12*x^2 - 8*x + 2) / (x^3 - 5*x^2 + 6*x)

theorem domain_of_f :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 3} = 
  (-∞, 0) ∪ (0, 2) ∪ (2, 3) ∪ (3, ∞) :=
by
  sorry

end domain_of_f_l132_132241


namespace number_of_black_bears_l132_132643

-- Definitions of conditions
def brown_bears := 15
def white_bears := 24
def total_bears := 66

-- The proof statement
theorem number_of_black_bears : (total_bears - (brown_bears + white_bears) = 27) := by
  sorry

end number_of_black_bears_l132_132643


namespace find_a_l132_132635

theorem find_a (a : ℝ) (h : a > 0) :
  let f := (1 + a * x)^2 * (1 - x)^5 in
  ∑ i in finset.filter (λ i, odd i) (finset.range 8), polynomial.coeff f i = -64 →
  a = 3 := 
sorry

end find_a_l132_132635


namespace solve_for_x_l132_132324

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (-2, x)
def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def sub_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def is_parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

theorem solve_for_x : ∀ x : ℝ, is_parallel (add_vectors a (b x)) (sub_vectors a (b x)) → x = -4 :=
by
  intros x h_par
  sorry

end solve_for_x_l132_132324


namespace tom_average_speed_l132_132751

theorem tom_average_speed 
  (karen_speed : ℕ) (tom_distance : ℕ) (karen_advantage : ℕ) (delay : ℚ)
  (h1 : karen_speed = 60)
  (h2 : tom_distance = 24)
  (h3 : karen_advantage = 4)
  (h4 : delay = 4/60) :
  ∃ (v : ℚ), v = 45 := by
  sorry

end tom_average_speed_l132_132751


namespace carnival_days_l132_132535

theorem carnival_days (d : ℕ) (h : 50 * d + 3 * (50 * d) - 30 * d - 75 = 895) : d = 5 :=
by
  sorry

end carnival_days_l132_132535


namespace reciprocal_is_correct_l132_132860

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l132_132860


namespace translated_function_l132_132142

def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f (x + a)

def translate_down (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ :=
  λ x, f x - b

def translate_left_and_down (f : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ :=
  translate_down (translate_left f a) b

theorem translated_function (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 2^x) →
  translate_left_and_down f 2 3 = λ x, 2^(x + 2) - 3 :=
by
  intro h
  funext x
  rw [translate_left_and_down, translate_left, translate_down]
  simp [h]
  sorry

end translated_function_l132_132142


namespace num_orderings_satisfying_condition_l132_132337

theorem num_orderings_satisfying_condition : 
  ∃ n : ℕ, n = 4608 ∧ 
  ∃ (a : fin 8 → fin 9),
  (∀ i, ∃! j, a j = i) ∧ 
  (a 0 - a 1 + a 2 - a 3 + a 4 - a 5 + a 6 - a 7 : ℤ) = 0 :=
sorry

end num_orderings_satisfying_condition_l132_132337


namespace least_positive_integer_n_l132_132293

theorem least_positive_integer_n : ∃ (n : ℕ), (1 / (n : ℝ) - 1 / (n + 1) < 1 / 100) ∧ ∀ m, m < n → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 100) :=
sorry

end least_positive_integer_n_l132_132293


namespace factorable_n_values_l132_132993

theorem factorable_n_values (n : ℤ) :
  (∃ f g : ℤ[X], f.degree > 0 ∧ g.degree > 0 ∧ f * g = X^5 - C n * X - C n - 2) ↔
  n = 10 ∨ n = 19 ∨ n = 34 ∨ n = 342 :=
by
  sorry

end factorable_n_values_l132_132993


namespace number_of_correct_statements_is_five_l132_132106

theorem number_of_correct_statements_is_five
  (h1 : ∀ (α β γ : ℕ), 2 * α + 2 * β + 2 * γ = 360 → max α (max β γ) = 90)
  (h2 : ∀ (α β : ℕ), α + β = 180 → α = β → α = 90)
  (h3 : ∀ (Δ : Type) [triangle Δ], isRight Δ ↔ altitudes_intersect_at_vertex Δ)
  (h4 : ∀ (A B C : ℕ), C = 2 * A ∧ C = 2 * B → C = 90)
  (h5 : ∀ (A B C : ℕ), A + B = C ∧ A + B + C = 180 → C = 90) : 
  5 = 5 := 
by 
  sorry

end number_of_correct_statements_is_five_l132_132106


namespace eighth_triangle_shaded_fraction_l132_132359

-- Definitions based on conditions
def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def small_triangles (n : ℕ) : ℕ := n * n
def pascal_shading_pattern (level : ℕ) : bool :=
  if (level / 2) % 2 == 0 then false else true  -- false for unshaded, true for shaded

-- Given Condition: A function to compute the shaded area for n-th triangle
def shaded_fraction (n : ℕ) : ℚ :=
  let total_small_triangles := small_triangles n in
  let levels := List.range n in
  let shaded := List.foldl (fun acc level => acc + if pascal_shading_pattern (level + 1) then level + 1 else 0) 0 levels in
  shaded / total_small_triangles

-- Theorem to prove the fraction for the eighth triangle
theorem eighth_triangle_shaded_fraction : shaded_fraction 8 = 1 / 4 := by
  sorry

end eighth_triangle_shaded_fraction_l132_132359


namespace correct_statement_is_C_l132_132576

-- Definition of conditions
def Statement_A : Prop := ∀ (P : IndustrialProduction) (E : ImmobilizedEnzyme), provide_nutrients(P, E)
def Statement_B : Prop := ∀ (I : ImmobilizedEnzyme) (Y : Yeast), utilization_efficiency(I, glucose) > utilization_efficiency(Y, glucose)
def Statement_C : Prop := ∀ (S : Sample) (D : Dilution), count_bacteria(DilutionPlate(s, S)) < actual_bacteria_count(S)
def Statement_D : Prop := ∀ (E : Experiment) (T : Temperature) (L : LaundryDetergent) (St : Stain), independent_variable(E, Stain)

-- The theorem to be proved
theorem correct_statement_is_C : (¬Statement_A) ∧ (¬Statement_B) ∧ Statement_C ∧ (¬Statement_D) :=
by 
  sorry -- Proof

end correct_statement_is_C_l132_132576


namespace first_tourist_arrives_earlier_l132_132144

-- Definitions of speeds and distances
variables {a : ℝ} -- total distance between points A and B
variables {T1 T2 : ℝ} -- times taken by tourist 1 and tourist 2

-- Define the time for the first tourist
def time_first_tourist (a : ℝ) : ℝ :=
  let x := (2*a) / 9 in
  x

-- Define the time for the second tourist
def time_second_tourist (a : ℝ) : ℝ :=
  (a / 10) + (a / 8)

-- Theorem stating that the first tourist arrives earlier than the second tourist
theorem first_tourist_arrives_earlier (a : ℝ) : 
  time_first_tourist a < time_second_tourist a := 
by
  sorry

end first_tourist_arrives_earlier_l132_132144


namespace probability_neither_square_cube_prime_l132_132117

theorem probability_neither_square_cube_prime :
  let total := 200
  let count_square := 14
  let count_cube := 5
  let overlap_square_cube := 2
  let count_prime := 46
  let overlap_prime_square := 5
  let overlap_prime_cube := 0
  let total_special := count_square + count_cube - overlap_square_cube + count_prime - overlap_prime_square - overlap_prime_cube
  let count_neither := total - total_special
  let probability := (count_neither : ℚ) / (total : ℚ)
  probability = 71 / 100 :=
by
  let total := 200
  let count_square := 14
  let count_cube := 5
  let overlap_square_cube := 2
  let count_prime := 46
  let overlap_prime_square := 5
  let overlap_prime_cube := 0
  let total_special := count_square + count_cube - overlap_square_cube + count_prime - overlap_prime_square - overlap_prime_cube
  let count_neither := total - total_special
  let probability := (count_neither : ℚ) / (total : ℚ)
  have probability_eq : probability = 71 / 100 := by sorry
  exact probability_eq

end probability_neither_square_cube_prime_l132_132117


namespace solve_for_y_l132_132079

theorem solve_for_y (y : ℚ) (h : (4 / 7) * (1 / 5) * y - 2 = 10) : y = 105 := by
  sorry

end solve_for_y_l132_132079


namespace rohan_salary_correct_l132_132903

noncomputable def rohan_monthly_salary : ℝ :=
  let S := 10000
  let food_expense := 0.4 * S
  let house_rent_expense := 0.2 * S
  let entertainment_expense := 0.1 * S
  let conveyance_expense := 0.1 * S
  let savings := 0.2 * S
  if savings = 2000 then S else 0

theorem rohan_salary_correct :
  let S := 10000 in 0.2 * S = 2000 → S = 10000 :=
by
  intros S h1
  unfold S at h1
  have hs : 0.2 * 10000 = 2000 := by norm_num
  exact hs.symm ▸ h1

end rohan_salary_correct_l132_132903


namespace ice_floe_mass_l132_132795

/-- Given conditions: 
 - The bear's mass is 600 kg
 - The diameter of the bear's trail on the ice floe is 9.5 meters
 - The observed diameter of the trajectory from the helicopter is 10 meters

 We need to prove that the mass of the ice floe is 11400 kg.
 -/
 theorem ice_floe_mass (m d D : ℝ) (hm : m = 600) (hd : d = 9.5) (hD : D = 10) :
   let M := m * d / (D - d)
   in M = 11400 := by 
 sorry

end ice_floe_mass_l132_132795


namespace range_of_f_l132_132602

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem range_of_f : set.image f {x : ℝ | x > 1} = {y : ℝ | y ≥ 3} :=
sorry

end range_of_f_l132_132602


namespace johnny_travel_time_l132_132747

-- Define the speeds and distance
def speed_jogging : ℝ := 5
def speed_bus : ℝ := 21
def distance_to_school : ℝ := 6.461538461538462

-- Define the times for each leg of the trip
def time_to_school := distance_to_school / speed_jogging
def time_back_home := distance_to_school / speed_bus

-- Define the total travel time
def total_travel_time := time_to_school + time_back_home

-- The statement to be proven
theorem johnny_travel_time : total_travel_time = 1.6 := by
  sorry

end johnny_travel_time_l132_132747


namespace amelie_wins_l132_132848

theorem amelie_wins (numbers : Finset ℕ) (n : ℕ) (h1 : numbers = Finset.range (n + 1) \ {0}) (h2 : n = 2017) :
  ∃ strategy : Π (numbers : Finset ℕ), numbers.card > 2 → ℕ,
    ∀ remaining_numbers : Finset ℕ,
      (remaining_numbers.card = 2 → (∑ x in remaining_numbers, x) % 8 = 0) :=
by
  sorry

end amelie_wins_l132_132848


namespace arithmetic_sequence_n_value_l132_132948

theorem arithmetic_sequence_n_value 
  (a : ℕ → ℤ) (d : ℤ) (n : ℕ)
  (h1 : ∑ i in finset.range (n + 1), a (2 * i + 1) = 132)
  (h2 : ∑ i in finset.range n, a (2 * (i + 1)) = 120)
  (h3 : ∀ i, a (i + 1) = a 1 + i * d) :
  n = 10 :=
sorry

end arithmetic_sequence_n_value_l132_132948


namespace find_p_l132_132710

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by
  sorry

end find_p_l132_132710


namespace geometric_sequence_second_term_l132_132093

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l132_132093


namespace square_garden_area_l132_132085

theorem square_garden_area (P A : ℕ)
  (h1 : P = 40)
  (h2 : A = 2 * P + 20) :
  A = 100 :=
by
  rw [h1] at h2 -- Substitute h1 (P = 40) into h2 (A = 2P + 20)
  norm_num at h2 -- Normalize numeric expressions in h2
  exact h2 -- Conclude by showing h2 (A = 100) holds

-- The output should be able to build successfully without solving the proof.

end square_garden_area_l132_132085


namespace smallest_power_of_7_not_palindrome_l132_132631

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_power_of_7_not_palindrome : ∃ n : ℕ, n > 0 ∧ 7^n = 2401 ∧ ¬is_palindrome (7^n) ∧ (∀ m : ℕ, m > 0 ∧ ¬is_palindrome (7^m) → 7^n ≤ 7^m) :=
by
  sorry

end smallest_power_of_7_not_palindrome_l132_132631


namespace part1_part2_l132_132412

namespace Problem

open Real

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

theorem part1 (h : p 1 x ∧ q x) : 2 < x ∧ x < 3:= 
sorry

theorem part2 (hpq : ∀ x, ¬ p a x → ¬ q x) : 
   1 < a ∧ a ≤ 2 := 
sorry

end Problem

end part1_part2_l132_132412


namespace tom_driving_speed_l132_132752

theorem tom_driving_speed
  (v : ℝ)
  (hKarenSpeed : 60 = 60) -- Karen drives at an average speed of 60 mph
  (hKarenLateStart: 4 / 60 = 1 / 15) -- Karen starts 4 minutes late, which is 1/15 hours
  (hTomDistance : 24 = 24) -- Tom drives 24 miles before Karen wins the bet
  (hTimeEquation: 24 / v = 8 / 15): -- The equation derived from given conditions
  v = 45 := 
by
  sorry

end tom_driving_speed_l132_132752


namespace ratio_of_A_to_B_l132_132073

theorem ratio_of_A_to_B (A B C : ℝ) (h1 : A + B + C = 544) (h2 : B = (1/4) * C) (hA : A = 64) (hB : B = 96) (hC : C = 384) : A / B = 2 / 3 :=
by 
  sorry

end ratio_of_A_to_B_l132_132073


namespace inequality_solution_set_l132_132267

theorem inequality_solution_set :
  { x : ℝ | -x^2 + 2*x > 0 } = { x : ℝ | 0 < x ∧ x < 2 } :=
sorry

end inequality_solution_set_l132_132267


namespace sum_areas_inscribed_rects_l132_132497

-- Define the vertices of the rectangle ABCD
variables (a b x : ℝ)
variable (K : ℝ × ℝ) -- K is a point on AB such that (0 ≤ x ≤ a) and K = (x, 0)

-- Define the vertices A, B, C, D of the rectangle ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (a, 0)
def C : ℝ × ℝ := (a, b)
def D : ℝ × ℝ := (0, b)

-- The inscribed rectangles sharing a common vertex K on AB
variables (L₁ L₂ M₁ M₂ N₁ N₂ : ℝ × ℝ)
-- Vertex coordinates constraints
variable (K_on_AB : K = (x, 0))
variable (x_range : 0 ≤ x ∧ x ≤ a)

-- Define the areas of the inscribed rectangles
def area_rect1 := x * (M₁.2)
def area_rect2 := (a - x) * (M₂.2)

-- The area of the larger rectangle ABCD
def area_ABCD := a * b

theorem sum_areas_inscribed_rects (h₁: area_rect1 = x * (M₁.2)) (h₂: area_rect2 = (a - x) * (M₂.2)) :
  area_rect1 + area_rect2 = area_ABCD :=
by
  sorry

end sum_areas_inscribed_rects_l132_132497


namespace count_possible_integer_values_l132_132289

theorem count_possible_integer_values 
  (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 20) : 
  ∃ (values : Finset ℕ), 
  (∀ x ∈ values, ∃ d, d = a + b ∧ d + c = 20 ∧ x = d / c) ∧ 
  values.card = 6 := 
by 
  -- The proof will show there are exactly 6 possible values for d / c
  sorry

end count_possible_integer_values_l132_132289


namespace element_in_set_l132_132344

theorem element_in_set (a b c : Type) : c ∈ ({a, b, c} : set Type) :=
by {
  -- Given the set M = {a, b, c}
  let M := {a, b, c},

  -- We need to prove that c ∈ M
  trivial
}

end element_in_set_l132_132344


namespace alpha_2_alpha_3_alpha_4_general_term_l132_132164

variable {n : ℕ}

def alpha_sequence : ℕ → ℝ
| 0     := 0
| 1     := (Real.pi / 5)
| (n+1) := (Real.pi - alpha_sequence (n+1)) / 2

theorem alpha_2 : alpha_sequence 2 = (2 * Real.pi / 5) := sorry

theorem alpha_3 : alpha_sequence 3 = (3 * Real.pi / 10) := sorry

theorem alpha_4 : alpha_sequence 4 = (7 * Real.pi / 20) := sorry

theorem general_term (n : ℕ) : alpha_sequence n = (Real.pi / 3) + ((-1) ^ n) * (4 * Real.pi / 15) := sorry

end alpha_2_alpha_3_alpha_4_general_term_l132_132164


namespace no_natural_number_n_exists_l132_132622

theorem no_natural_number_n_exists (n : ℕ) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 * n * (n + 1) * (n + 2) * (n + 3) + 12 := 
sorry

end no_natural_number_n_exists_l132_132622


namespace remainder_7_pow_137_mod_11_l132_132500

theorem remainder_7_pow_137_mod_11 :
    (137 = 13 * 10 + 7) →
    (7^10 ≡ 1 [MOD 11]) →
    (7^137 ≡ 6 [MOD 11]) :=
by
  intros h1 h2
  sorry

end remainder_7_pow_137_mod_11_l132_132500


namespace correct_conclusions_l132_132303

theorem correct_conclusions (a b c d x₁ x₂ : ℝ)
  (h1 : x^2 + a*x + b > 0 ∀ x ∈ ℝ, x ≠ d)
  (h2 : a > 0)
  (h3 : solution_set : ∀ x, x ≠ d ↔ x ∈ {x | x^2 + a*x + b > 0})
  (h4 : | x₁ - x₂ | = 4)
  (h5 : (x₁, x₂) = (roots_of_quadratic_ineq x^2 + a*x - b < 0) ) :
  (a^2 = 4*b) ∧ (a^2 + 1/b ≥ 4) ∧ (c = 4) :=
by
  sorry

end correct_conclusions_l132_132303


namespace man_age_difference_l132_132189

-- Definitions of conditions
def present_age_son : ℕ := 24 -- The present age of the son is 24 years
def present_age_man (M : ℕ) : Prop := M + 2 = 2 (present_age_son + 2)

-- The statement we want to prove
theorem man_age_difference (M : ℕ) (h : present_age_man M) : M - present_age_son = 26 :=
by
  sorry

end man_age_difference_l132_132189


namespace number_of_arrangements_l132_132176

-- Definitions of teachers and schools
inductive Teacher
| A | B | C | D | E

inductive School
| A | B | C | D

open Teacher School

-- Condition: Each school must have at least one teacher
def nonempty_schools (assignment : Teacher → School) : Prop :=
  (∃ t, assignment t = A) ∧ (∃ t, assignment t = B) ∧ (∃ t, assignment t = C) ∧ (∃ t, assignment t = D)

-- Condition: Teachers A, B, and C do not go to school B
def restriction (assignment : Teacher → School) : Prop :=
  (assignment A ≠ B) ∧ (assignment B ≠ B) ∧ (assignment C ≠ B)

-- The main statement
theorem number_of_arrangements : 
  ∃ (assignment : Teacher → School), nonempty_schools assignment ∧ restriction assignment :=
sorry

end number_of_arrangements_l132_132176


namespace number_of_triangles_in_polygon_with_200_sides_l132_132650

noncomputable def triangle_count (n : ℕ) (k : ℕ) : ℕ := (n.choose k)

theorem number_of_triangles_in_polygon_with_200_sides :
  triangle_count 200 3 = 1313400 :=
by
  sorry

end number_of_triangles_in_polygon_with_200_sides_l132_132650


namespace radius_of_kth_circle_l132_132581

theorem radius_of_kth_circle (k : ℕ) : 
  let r : ℕ → ℝ := λ k, 4 / (4 * k^2 - 4 * k + 9) in
  r k = 4 / (4 * k^2 - 4 * k + 9) :=
by 
  sorry

end radius_of_kth_circle_l132_132581


namespace Tabitha_age_proof_l132_132609

variable (Tabitha_age current_hair_colors: ℕ)
variable (Adds_new_color_per_year: ℕ)
variable (initial_hair_colors: ℕ)
variable (years_passed: ℕ)

theorem Tabitha_age_proof (h1: Adds_new_color_per_year = 1)
                          (h2: initial_hair_colors = 2)
                          (h3: ∀ years_passed, Tabitha_age  = 15 + years_passed)
                          (h4: Adds_new_color_per_year  = 1 )
                          (h5: current_hair_colors =  8 - 3)
                          (h6: current_hair_colors  =  initial_hair_colors + 3)
                          : Tabitha_age = 18 := 
by {
  sorry  -- Proof omitted
}

end Tabitha_age_proof_l132_132609


namespace jan_drove_more_than_ian_and_han_l132_132512

noncomputable def distances_proof : ℕ → ℕ → ℕ → Prop := 
  λ (d_H d_I d_J : ℕ), 
    ∃ (s_I t_I : ℕ),
    d_I = s_I * t_I ∧ 
    d_H = (s_I + 10) * (t_I + 2) ∧
    d_J = (s_I + 15) * (t_I + 3) ∧ 
    d_H = d_I + 100 ∧ 
    d_J = d_I + 150 ∧
    d_J = d_H + 150

theorem jan_drove_more_than_ian_and_han : 
  ∀ (d_H d_I d_J : ℕ), 
    distances_proof d_H d_I d_J → 
    (d_J = d_I + 150 ∧ d_J = d_H + 150) := 
begin
  intros d_H d_I d_J h,
  exact h.right.right.right.right,
  sorry
end

end jan_drove_more_than_ian_and_han_l132_132512


namespace two_triangles_form_nonagon_l132_132381

theorem two_triangles_form_nonagon (Δ1 Δ2 : Type) [triangle Δ1] [triangle Δ2] :
  ∃ (nonagon : Type), nonagon := sorry

end two_triangles_form_nonagon_l132_132381


namespace find_d_minus_c_l132_132559

variable (c d : ℝ)

def rotate180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (2 * cx - x, 2 * cy - y)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

def transformations (q : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_eq_x (rotate180 q (2, 3))

theorem find_d_minus_c :
  transformations (c, d) = (1, -4) → d - c = 7 :=
by
  intro h
  sorry

end find_d_minus_c_l132_132559


namespace probability_green_or_yellow_l132_132150

/--
Prove that the probability of drawing one marble
which is either green or yellow from a bag containing
4 green marbles, 3 yellow marbles, and 8 white marbles
is 7/15.
-/
theorem probability_green_or_yellow (green yellow white : ℕ) (h_green : green = 4)
  (h_yellow : yellow = 3) (h_white : white = 8) :
  (green + yellow) / (green + yellow + white) = 7 / 15 := 
by
  have h1 : green = 4 := h_green,
  have h2 : yellow = 3 := h_yellow,
  have h3 : white = 8 := h_white,
  sorry

end probability_green_or_yellow_l132_132150


namespace johns_shell_arrangements_l132_132386

-- Define the total number of arrangements without considering symmetries
def totalArrangements := Nat.factorial 12

-- Define the number of equivalent arrangements due to symmetries
def symmetries := 6 * 2

-- Define the number of distinct arrangements
def distinctArrangements : Nat := totalArrangements / symmetries

-- State the theorem
theorem johns_shell_arrangements : distinctArrangements = 479001600 :=
by
  sorry

end johns_shell_arrangements_l132_132386


namespace find_angle_A_find_side_a_l132_132350

noncomputable def triangle_A (a b c : ℝ) (B A : ℝ) : Prop :=
c = a * real.cos B + 2 * b * (real.sin (A / 2))^2

noncomputable def median_length (b : ℝ) (m : ℝ) : Prop :=
b = 4 ∧ m = real.sqrt 7

theorem find_angle_A (a b c A B : ℝ) (h : triangle_A a b c B A) : A = real.pi / 3 :=
sorry

theorem find_side_a (a b : ℝ) (m : ℝ) (h1 : median_length b m) (h2 : a^2 = b^2 + c^2 - 2 * b * c * (real.cos (real.pi / 3))) : a = real.sqrt 13 :=
sorry

end find_angle_A_find_side_a_l132_132350


namespace domain_of_f_l132_132618

noncomputable def f (x : ℝ) : ℝ := real.sqrt (2 - real.sqrt (3 - real.sqrt (4 - x)))

theorem domain_of_f : set.Icc (-5 : ℝ) (4 : ℝ) = { x : ℝ | (2 - real.sqrt (3 - real.sqrt (4 - x))) ≥ 0 } :=
sorry

end domain_of_f_l132_132618


namespace point_p_min_distance_l132_132134

noncomputable def parabola_focus (a : ℝ) : ℝ × ℝ := (0, 1 / (4 * a))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem point_p_min_distance (P : ℝ × ℝ) (hP : P.2 = 2 * P.1^2) :
  let A := (1, 3)
  let F := parabola_focus 2
  P.2 = 2 ∧ P.1 = 1 ∧ 
  ∀ Q : ℝ × ℝ, (Q.2 = 2 * Q.1 ^ 2) → Q ≠ P →
  distance P A + distance P F ≤ distance Q A + distance Q F :=
begin
  sorry
end

end point_p_min_distance_l132_132134


namespace sadie_algebra_problems_l132_132808

def total_problems : ℤ := 250
def algebra_percentage : ℝ := 0.50
def linear_equations_percentage : ℝ := 0.35
def quadratic_equations_percentage : ℝ := 0.25
def systems_of_equations_percentage : ℝ := 0.20
def polynomial_equations_percentage : ℝ := 0.20

def algebra_problems : ℤ := (algebra_percentage * total_problems).to_int
def linear_problems : ℤ := (linear_equations_percentage * algebra_problems).to_int
def quadratic_problems : ℤ := (quadratic_equations_percentage * algebra_problems).to_int
def systems_problems : ℤ := (systems_of_equations_percentage * algebra_problems).to_int
def polynomial_problems : ℤ := (polynomial_equations_percentage * algebra_problems).to_int

theorem sadie_algebra_problems : 
  linear_problems = 44 ∧ 
  quadratic_problems = 31 ∧ 
  systems_problems = 25 ∧ 
  polynomial_problems = 25 :=
by {
  sorry -- The proof is omitted as requested
}

end sadie_algebra_problems_l132_132808


namespace find_present_age_of_eldest_l132_132469

noncomputable def eldest_present_age (x : ℕ) : ℕ :=
  8 * x

theorem find_present_age_of_eldest :
  ∃ x : ℕ, 20 * x - 21 = 59 ∧ eldest_present_age x = 32 :=
by
  sorry

end find_present_age_of_eldest_l132_132469


namespace circle_area_l132_132880

theorem circle_area : 
  ∀ (A B C M O : Type) 
    (h1 : AB = 5) (h2 : AC = 5) (h3 : BC = 8) 
    (h4 : M = midpoint B C) (h5 : O = incenter A B C),
  area_of_circle_passing_through A O M = (25 * π) / 4 :=
by
  sorry

end circle_area_l132_132880


namespace find_quadratic_polynomial_l132_132315

theorem find_quadratic_polynomial (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = -2 ∨ x = 3) ∧ q 0 = 4 →
  q = λ x, (2 / 3) * x^2 - (2 / 3) * x - 4 :=
by
  sorry

end find_quadratic_polynomial_l132_132315


namespace find_m_l132_132778

theorem find_m (m : ℕ) (hm : 0 < m)
  (a : ℕ := Nat.choose (2 * m) m)
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h : 13 * a = 7 * b) : m = 6 := by
  sorry

end find_m_l132_132778


namespace represent_as_sets_l132_132208

def is_set (s : Type*) : Prop := ∃ (S : set s), true  -- Basic existence for a set

variable (A : Type*) [is_set A]
variable (B : Type*) [is_set B]
variable (C : Type*) [is_set C]
variable (D : Type*) [is_set D]

/-- Proof that only ② (all equilateral triangles) and ③ (real number solutions to \(x^2 - 4 = 0\)) can be represented as sets -/
theorem represent_as_sets : 
  ¬ is_set (prop1 A) ∧ is_set (prop2 B) ∧ is_set (prop3 C) ∧ ¬ is_set (prop4 D) → 
  ((prop2 B ∧ prop3 C ) ∨ (prop1 A ∧ prop2 B ∧ prop3 C ∧ prop4 D) := by
{ sorry }

end represent_as_sets_l132_132208


namespace max_tan_B_minus_C_l132_132351

theorem max_tan_B_minus_C (A B C a b c : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_sides : a = 2 * b * Real.cos C - 3 * c * Real.cos B) :
  Real.tan (B - C) ≤ 3 / 4 := 
sorry

end max_tan_B_minus_C_l132_132351


namespace alyssa_puppies_l132_132207

theorem alyssa_puppies (total_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : total_puppies = 7) (h2 : given_away = 5) 
  : remaining_puppies = total_puppies - given_away → remaining_puppies = 2 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end alyssa_puppies_l132_132207


namespace constant_term_eq_240_coefficient_of_middle_term_l132_132308

variable (x : ℝ) (n : ℕ)

-- Definition for problem I
def constant_term_in_expansion : ℝ :=
  let expr := (x^2 + 2/x)^6 in
  sorry -- the process to find the constant term

-- Statement for problem I
theorem constant_term_eq_240
  (h : n = 6) :
  constant_term_in_expansion x n = 240 := sorry

-- Definition for problem II
def middle_term_coefficient : ℝ :=
  let expr := (x^2 + 2/x)^8 in
  sorry -- the process to find the middle term coefficient

-- Statement for problem II
theorem coefficient_of_middle_term
  (h : binomial.coeff n 2 = binomial.coeff n 6) :
  middle_term_coefficient x n = 1120 := sorry

end constant_term_eq_240_coefficient_of_middle_term_l132_132308


namespace prove_kl_l132_132304

variables {ℝ : Type*} [Field ℝ]

-- Given that vectors a and b
variables (a b : ℝ)
-- are not collinear
(h_not_collinear : ∀ (m : ℝ), a ≠ m * b)
-- and we define AB and AC as stated
(variable k l : ℝ)
(variable AB AC : ℝ)
(h_AB_eq : AB = a + k * b)
(h_AC_eq : AC = l * a + b)
-- And AB is collinear with AC
(h_collinear : ∃ m : ℝ, AB = m * AC)

-- We need to prove k * l - 1 = 0
theorem prove_kl (h_not_collinear : ∀ (m : ℝ), a ≠ m * b) 
  (a b : ℝ) (h_AB_eq : AB = a + k * b) 
  (h_AC_eq : AC = l * a + b) 
  (h_collinear : ∃ m : ℝ, AB = m * AC) : (k * l - 1 = 0) := 
by
  sorry

end prove_kl_l132_132304


namespace mod_computation_l132_132886

theorem mod_computation (n : ℤ) : 
  0 ≤ n ∧ n < 23 ∧ 47582 % 23 = n ↔ n = 3 := 
by 
  -- Proof omitted
  sorry

end mod_computation_l132_132886


namespace area_S_inequality_l132_132965

noncomputable def F (t : ℝ) : ℝ := 2 * (t - ⌊t⌋)

def S (t : ℝ) : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 - F t) * (p.1 - F t) + p.2 * p.2 ≤ (F t) * (F t) }

theorem area_S_inequality (t : ℝ) : 0 ≤ π * (F t) ^ 2 ∧ π * (F t) ^ 2 ≤ 4 * π := 
by sorry

end area_S_inequality_l132_132965


namespace lattice_points_interval_l132_132395
open Set

def is_lattice_point (x y : ℤ) (n : ℤ) : Prop :=
  1 ≤ x ∧ x ≤ 40 ∧ 1 ≤ y ∧ y ≤ 40 ∧ y ≤ n * x

theorem lattice_points_interval (c d : ℕ) (hc : c = 1) (hd : d = 20) :
  ∃ n_interval_start n_interval_end interval_length,
  (n_interval_start = 39 / 40) ∧ (n_interval_end = 41 / 40) ∧ (interval_length = 1 / 20) ∧ (c + d = 21) :=
by
  use [39 / 40, 41 / 40, 1 / 20]
  simp [hc, hd]
  linarith
  sorry -- Proof steps for the calculation validation

end lattice_points_interval_l132_132395


namespace apples_difference_l132_132206

theorem apples_difference (Adam_apples : ℕ) (Jackie_apples : ℕ) (h1 : Adam_apples = 10) (h2 : Jackie_apples = 2) : Adam_apples - Jackie_apples = 8 :=
by
  rw [h1, h2]
  norm_num

end apples_difference_l132_132206


namespace parallel_lines_l132_132973

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end parallel_lines_l132_132973


namespace books_in_special_collection_l132_132192

theorem books_in_special_collection (
  initial_books : ℕ,
  loaned_out : ℕ,
  return_rate : ℚ
) (h1 : initial_books = 75) (h2 : loaned_out = 40) (h3 : return_rate = 0.80) : initial_books - loaned_out + (loaned_out * return_rate).to_nat = 67 :=
by
  sorry

end books_in_special_collection_l132_132192


namespace smallest_possible_a_l132_132765

theorem smallest_possible_a (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : 2 * b = a + c) (h4 : a^2 = c * b) : a = 1 :=
by
  sorry

end smallest_possible_a_l132_132765


namespace average_income_of_other_40_customers_l132_132157

/-
Given:
1. The average income of 50 customers is $45,000.
2. The average income of the wealthiest 10 customers is $55,000.

Prove:
1. The average income of the other 40 customers is $42,500.
-/

theorem average_income_of_other_40_customers 
  (avg_income_50 : ℝ)
  (wealthiest_10_avg : ℝ) 
  (total_customers : ℕ)
  (wealthiest_customers : ℕ)
  (remaining_customers : ℕ)
  (h1 : avg_income_50 = 45000)
  (h2 : wealthiest_10_avg = 55000)
  (h3 : total_customers = 50)
  (h4 : wealthiest_customers = 10)
  (h5 : remaining_customers = 40) :
  let total_income_50 := total_customers * avg_income_50
  let total_income_wealthiest_10 := wealthiest_customers * wealthiest_10_avg
  let income_remaining_customers := total_income_50 - total_income_wealthiest_10
  let avg_income_remaining := income_remaining_customers / remaining_customers
  avg_income_remaining = 42500 := 
sorry

end average_income_of_other_40_customers_l132_132157


namespace expression_value_l132_132734

theorem expression_value : 
  let x := -1/2
  in (4 * x) / (x - 1)^2 = -8 / 9 :=
by
  sorry

end expression_value_l132_132734


namespace sum_a4_a5_l132_132285

noncomputable def a : ℕ → ℚ
| 1 := 2
| 2 := 1
| (n + 1) := 
  if h : n ≥ 2 then 
    have : n + 1 ≥ 3 := by linarith,
    have h₁ : a.val.1 (n + 1) = (2 * a n - 1 / a (n + 2))⁻¹ := by sorry,
    (h₁ this).symm
  else sorry

theorem sum_a4_a5 : a 4 + a 5 = 9 / 10 := by
  sorry

end sum_a4_a5_l132_132285


namespace imaginary_part_inv_z_l132_132306

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_inv_z : Complex.im (1 / z) = 2 / 5 :=
by
  -- proof to be filled in
  sorry

end imaginary_part_inv_z_l132_132306


namespace SandyCarrotsLeft_l132_132810

def initialCarrots : ℕ := 6
def takenCarrots : ℕ := 3

theorem SandyCarrotsLeft : initialCarrots - takenCarrots = 3 := 
by 
  rw [initialCarrots, takenCarrots]
  exact rfl -- the equality holds as 6 - 3 = 3
  sorry

end SandyCarrotsLeft_l132_132810


namespace select_employees_from_A_l132_132536

-- Definitions of conditions
def e_A : Nat := 200
def e_B : Nat := 500
def e_C : Nat := 100
def t : Nat := 40
def Total : Nat := e_A + e_B + e_C
def Ratio : Rat := t / Total

-- Theorem statement
theorem select_employees_from_A : (e_A : Rat) * Ratio = 10 := by
  sorry

end select_employees_from_A_l132_132536


namespace find_k_if_lines_parallel_l132_132978

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end find_k_if_lines_parallel_l132_132978


namespace breadth_when_area_increases_by_50_breadth_when_area_decreases_by_50_l132_132842

-- Definitions from the conditions
variables {L B B' : ℝ}
def A : ℝ := L * B
def new_length : ℝ := L / 2

-- Two cases for the new area where area changes by 50%
def new_area_increase : ℝ := A * 1.5
def new_area_decrease : ℝ := A * 0.5

-- Prove the cases
theorem breadth_when_area_increases_by_50:
  new_length * B' = new_area_increase → B' = 3 * B := by
sorry

theorem breadth_when_area_decreases_by_50:
  new_length * B' = new_area_decrease → B' = B := by
sorry

end breadth_when_area_increases_by_50_breadth_when_area_decreases_by_50_l132_132842


namespace stephanie_total_oranges_l132_132080

def total_oranges (n : ℕ) : ℕ := (n + 1)

theorem stephanie_total_oranges : 
  (∑ n in Finset.range 8, total_oranges n) = 52 := 
by
  sorry

end stephanie_total_oranges_l132_132080


namespace tan_theta_equation_l132_132777

theorem tan_theta_equation (k : ℝ) (θ : ℝ) (h₁ : 0 < k) 
  (h₂ : cos θ * k + 3 = 11) (h₃ : sin θ * k + 3 = 7) : 
  tan θ = 1 / 2 := 
by
  sorry

end tan_theta_equation_l132_132777


namespace equal_if_fraction_is_positive_integer_l132_132024

theorem equal_if_fraction_is_positive_integer
  (a b : ℕ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (K : ℝ := Real.sqrt ((a^2 + b^2:ℕ)/2))
  (A : ℝ := (a + b:ℕ)/2)
  (h_int_pos : ∃ (n : ℕ), n > 0 ∧ K / A = n) :
  a = b := sorry

end equal_if_fraction_is_positive_integer_l132_132024


namespace unique_positive_integer_pair_l132_132629

theorem unique_positive_integer_pair (x y : ℕ) (hxy : x ≤ y) (h : real.sqrt x + real.sqrt y = real.sqrt 1992) : 
  (x, y) = (498, 498) :=
sorry

end unique_positive_integer_pair_l132_132629


namespace base_prime_representation_945_l132_132258

theorem base_prime_representation_945 : 
  (∃ (exponents : List ℕ), exponents = [3, 1, 1, 0] ∧ 
  (945 = (2 ^ (exponents.nth 0).getOrElse 0) * 
        (3 ^ (exponents.nth 1).getOrElse 0) * 
        (5 ^ (exponents.nth 2).getOrElse 0) * 
        (7 ^ (exponents.nth 3).getOrElse 0))) := 
by {
  -- For simplicity in the statement, we are expressing the exponents in a list and verifying the number 945.
  use [3, 1, 1, 0],
  simp,
  sorry
}

end base_prime_representation_945_l132_132258


namespace intersect_value_l132_132737

noncomputable def coord_x_c : ℝ := 1
noncomputable def curve_C1 (x y : ℝ) : Prop := (x^2 / 4 + y^2 = 1)
noncomputable def line_l (x y t : ℝ) : Prop := (y = sqrt 3 + (sqrt 3 / 2) * t) ∧ (x = (1 / 2) * t)
noncomputable def point_P : (ℝ × ℝ) := (0, sqrt 3)
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem intersect_value : ∀ (A B : ℝ × ℝ), 
  (curve_C1 A.1 A.2) ∧ (curve_C1 B.1 B.2) ∧ (∃ t : ℝ, line_l A.1 A.2 t) ∧ (∃ t : ℝ, line_l B.1 B.2 t) →
  (1 / distance point_P A) + (1 / distance point_P B) = 3 / 2 :=
by
  sorry


end intersect_value_l132_132737


namespace triangle_altitudes_line_l132_132652

theorem triangle_altitudes_line {A B C : Type} [Coord A] [Coord B] [Coord C]
  (altitude1 : Line) (altitude2 : Line) (vertexA : A) (coordA : vertexA = (1, 2))
  (eq_altitude1 : altitude1.equation = 2 * x - 3 * y + 1)
  (eq_altitude2 : altitude2.equation = x + y = 0) :
  ∃ lineBC : Line, lineBC.equation = 2 * x + 3 * y + 7 := 
begin
  sorry
end

end triangle_altitudes_line_l132_132652


namespace seating_arrangement_l132_132133

theorem seating_arrangement {n : ℕ} (h_n: n = 9) (k : ℕ) (h_k: k = 6) :
  ∃ (ways : ℕ), ways = 7200 ∧ 
  (∑ i in finset.range (k - 1), finset.choose n (k - 1) * finset.perm k (k - 1) = ways) := 
begin
  sorry
end

end seating_arrangement_l132_132133


namespace expand_expression_l132_132988

theorem expand_expression :
  (x^22 - 4 * x^7 + x^(-3) - 8 + 2 * x^3) * (-3 * x^6) =
  -3 * x^28 + 12 * x^13 - 6 * x^9 + 24 * x^6 - 3 * x^3 :=
by
  sorry

end expand_expression_l132_132988


namespace toms_speed_l132_132756

/--
Karen places a bet with Tom that she will beat Tom in a car race by 4 miles 
even if Karen starts 4 minutes late. Assuming that Karen drives at 
an average speed of 60 mph and that Tom will drive 24 miles before 
Karen wins the bet. Prove that Tom's average driving speed is \( \frac{300}{7} \) mph.
--/
theorem toms_speed (
  (karen_speed : ℕ) (karen_lateness : ℚ) (karen_beats_tom_by : ℕ) 
  (karen_distance_when_tom_drives_24_miles : ℕ) 
  (karen_speed = 60) 
  (karen_lateness = 4 / 60) 
  (karen_beats_tom_by = 4) 
  (karen_distance_when_tom_drives_24_miles = 24)) : 
  ∃ tom_speed : ℚ, tom_speed = 300 / 7 :=
begin
  sorry
end

end toms_speed_l132_132756


namespace num_five_digit_numbers_adjacent_12_l132_132488

def five_digit_numbers_adjacent_12 : Nat :=
  36

theorem num_five_digit_numbers_adjacent_12 {n : Nat} (h1 : n = 5)
  (h2 : ∀ d, d ∈ [0, 1, 2, 3, 4])
  (h3 : ∀ d d', d ≠ d')
  (h4 : ∀ i j, (i = 1 ∧ j = 2) ∨ (i = 2 ∧ j = 1) → ∃ k, (i = k ∨ j = k) ∧ (k + 1 = i ∨ k + 1 = j)) : 
  n = five_digit_numbers_adjacent_12 :=
  sorry

end num_five_digit_numbers_adjacent_12_l132_132488


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l132_132852

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l132_132852


namespace parts_drawn_l132_132274

-- Given that a sample of 30 parts is drawn and each part has a 25% chance of being drawn,
-- prove that the total number of parts N is 120.

theorem parts_drawn (N : ℕ) (h : (30 : ℚ) / N = 0.25) : N = 120 :=
sorry

end parts_drawn_l132_132274


namespace root_of_n_identity_l132_132433

theorem root_of_n_identity (n : ℕ) (h : n > 0) : 
  real.sqrt (n : ℝ) = 2^(n-1) * ∏ k in finset.range (n-1), real.sin (k * real.pi / (2 * n)) :=
by 
  -- Proof is omitted, but the statement correctly captures the given problem.
  sorry

end root_of_n_identity_l132_132433


namespace nick_charges_l132_132790

theorem nick_charges (y : ℕ) :
  let travel_cost := 7
  let hourly_rate := 10
  10 * y + 7 = travel_cost + hourly_rate * y :=
by sorry

end nick_charges_l132_132790


namespace asian_games_profit_l132_132447

/-- Given the fixed cost and selling price conditions for a product, define the profit function y
    and prove the conditions under which the profit is maximized at a given production level. -/
theorem asian_games_profit (x : ℝ) (hx₀ : 0 < x) :
  let 
    y := if x < 60 then - (1 / 2) * x^2 + 5 * x - 4
                 else - x - (81 / x) + (55 / 2)
  in 
    (0 < x ∧ x < 60) → y = - (1 / 2) * x^2 + 5 * x - 4 ∨
    (x ≥ 60) → y = - x - (81 / x) + (55 / 2)
    ∧ (∀ x, 0 < x ∧ x < 60 → y ≤ 8.5) 
    ∧ (∀ x, x ≥ 60 → y ≤ 9.5) 
    ∧ (∃ x, x = 9 → y = 9.5) := sorry


end asian_games_profit_l132_132447


namespace exists_group_round_table_l132_132210

open Finset Function

variable (P : Finset ℤ) (knows : ℤ → ℤ → Prop)

def has_at_least_three_friends (P : Finset ℤ) (knows : ℤ → ℤ → Prop) : Prop :=
  ∀ p ∈ P, (P.filter (knows p)).card ≥ 3

noncomputable def exists_even_group (P : Finset ℤ) (knows : ℤ → ℤ → Prop) : Prop :=
  ∃ S : Finset ℤ, (S ⊆ P) ∧ (2 < S.card) ∧ (Even S.card) ∧ (∀ p ∈ S, ∀ q ∈ S, Edge_connected p q knows S)

theorem exists_group_round_table (P : Finset ℤ) (knows : ℤ → ℤ → Prop) 
  (h : has_at_least_three_friends P knows) : 
  exists_even_group P knows :=
sorry

end exists_group_round_table_l132_132210


namespace arrangement_count_correct_l132_132565

def num_arrangements (people : Finset ℕ) (classes : Finset ℕ) : ℕ :=
  114

theorem arrangement_count_correct :
  ∀ (people : Finset ℕ) (classes : Finset ℕ),
    people = {0, 1, 2, 3, 4} ∧
    classes = {0, 1, 2} ∧
    (∀ p ∈ people, ∃ c ∈ classes, true) ∧ 
    ∃ (f : ℕ -> ℕ), f 0 ≠ f 1 :=
  num_arrangements people classes = 114 :=
by
  sorry

end arrangement_count_correct_l132_132565


namespace sqrt_n_eq_prod_sin_l132_132431

theorem sqrt_n_eq_prod_sin (n : ℕ) (h : n > 0) : 
  real.sqrt n = 2^(n-1) * ∏ k in finset.range (n-1), real.sin (k * real.pi / (2 * n)) :=
by
  sorry

end sqrt_n_eq_prod_sin_l132_132431


namespace sum_of_possible_values_l132_132373

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| + 2 = 4) :
  x = 7 ∨ x = 3 → x = 10 := 
by sorry

end sum_of_possible_values_l132_132373


namespace area_excluding_hole_l132_132929

def area_large_rectangle (x : ℝ) : ℝ :=
  (2 * x + 9) * (x + 6)

def area_square_hole (x : ℝ) : ℝ :=
  (x - 1) * (x - 1)

theorem area_excluding_hole (x : ℝ) : 
  area_large_rectangle x - area_square_hole x = x^2 + 23 * x + 53 :=
by
  sorry

end area_excluding_hole_l132_132929


namespace find_polynomials_l132_132429

noncomputable def polynomial_satisfying_condition (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), f(x^2) = (f(x))^2

theorem find_polynomials (f : ℝ → ℝ) :
  polynomial_satisfying_condition f →
  (∃ k : ℕ, f = λ x, x^k) ∨ f = λ x, 0 :=
by
  -- Proof steps go here
  sorry

end find_polynomials_l132_132429


namespace find_theta_l132_132479

theorem find_theta (R h : ℝ) (θ : ℝ) 
  (r1_def : r1 = R * Real.cos θ)
  (r2_def : r2 = (R + h) * Real.cos θ)
  (s_def : s = 2 * π * h * Real.cos θ)
  (s_eq_h : s = h) : 
  θ = Real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l132_132479


namespace two_pow_n_minus_one_not_divisible_by_n_l132_132656

theorem two_pow_n_minus_one_not_divisible_by_n (n : ℤ) (h : n > 1) : ¬ n ∣ (2^n - 1) := 
sorry

end two_pow_n_minus_one_not_divisible_by_n_l132_132656


namespace intersection_points_sum_square_constant_l132_132655

-- Setting up the conditions
variables (x y : ℝ)
def ellipse (a b : ℝ) (h : a > b) : Prop := 
  (a = 2) ∧ (b = Real.sqrt 3) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

-- Defining the intersection points
variables (x1 y1 x2 y2 : ℝ)
def line_intersects_ellipse (l : Rel) : Prop :=
  let ΔOPQ_area := (1/2) * |x1 - 0| * |y1 - 0|
  ΔOPQ_area = Real.sqrt 3

-- The proof goal
theorem intersection_points_sum_square_constant
  (a b : ℝ) (h : a > b)
  (hx : ellipse a b h)
  (l : Rel)
  (h_intersect : line_intersects_ellipse l) :
  x1 ^ 2 + x2 ^ 2 = 4 := sorry

end intersection_points_sum_square_constant_l132_132655


namespace count_positive_integers_in_range_l132_132691

theorem count_positive_integers_in_range : 
  {n : ℕ | 300 < n^2 ∧ n^2 < 1200}.to_finset.card = 17 :=
by
  sorry

end count_positive_integers_in_range_l132_132691


namespace min_connected_points_l132_132548

noncomputable def graph (P : Type) := P → P → Prop

variables {P : Type} [fintype P] (G : graph P) [h : fintype P] (pts : P)

theorem min_connected_points (n : ℕ) (h_n : n = 100)
  (h_cond : ∀ (A B C D : P), (∃ X : P, (X ≠ A) ∧ (X ≠ B) ∧ (X ≠ C) ∧ (X ≠ D) ∧ 
    (G X A ∧ G X B ∧ G X C ∧ G X D))) :
  ∃ (S : fin (97)), ∀ (x ∈ S), ∀ (y : P), y ≠ x → G x y :=
sorry

end min_connected_points_l132_132548


namespace estimate_probability_estimate_points_l132_132141

theorem estimate_probability (hit_frequencies : List ℝ) (h : hit_frequencies = [0.75, 0.8, 0.8, 0.81, 0.8, 0.8]) :
  (hit_frequencies.count (λ x, x = 0.8) ≥ 4) →
  ∃ p, p = 0.8 :=
by
  intros
  exact ⟨0.8, rfl⟩
  -- This is just a statement with 'sorry', no need for proof here.

theorem estimate_points (p : ℝ) (h : p = 0.8) (free_throw_opportunities : ℕ) (attempts_per_opportunity : ℕ) :
  (free_throw_opportunities = 10) → (attempts_per_opportunity = 2) →
  ∃ points, points = 16 :=
by
  intros
  exact ⟨16, rfl⟩
  -- This is just a statement with 'sorry', no need for proof here.

end estimate_probability_estimate_points_l132_132141


namespace exists_valid_circle_group_l132_132217

variable {P : Type}
variable (knows : P → P → Prop)

def knows_at_least_three (p : P) : Prop :=
  ∃ (p₁ p₂ p₃ : P), p₁ ≠ p ∧ p₂ ≠ p ∧ p₃ ≠ p ∧ knows p p₁ ∧ knows p p₂ ∧ knows p p₃

def valid_circle_group (G : List P) : Prop :=
  (2 < G.length) ∧ (G.length % 2 = 0) ∧ (∀ i, knows (G.nthLe i sorry) (G.nthLe ((i + 1) % G.length) sorry) ∧ knows (G.nthLe i sorry) (G.nthLe ((i - 1 + G.length) % G.length) sorry))

theorem exists_valid_circle_group (H : ∀ p : P, knows_at_least_three knows p) : 
  ∃ G : List P, valid_circle_group knows G := 
sorry

end exists_valid_circle_group_l132_132217


namespace simplify_expression_l132_132813

theorem simplify_expression : 4 * (12 / 9) * (36 / -45) = -12 / 5 :=
by
  sorry

end simplify_expression_l132_132813


namespace no_tetrahedron_with_all_right_triangles_l132_132981

-- Define the geometric objects and their properties
noncomputable def Tetrahedron_with_all_right_triangles := 
  ∃ (A B C D : Type) (distance : A → B → ℝ),
    -- All faces are right-angled triangles
    (triangle A B C ∧ right_triangle A B C ∧ hypotenuse B = distance A B) ∧
    (triangle A B D ∧ right_triangle A B D ∧ hypotenuse B = distance A B) ∧
    (triangle A C D ∧ right_triangle A C D ∧ hypotenuse C = distance A C) ∧
    (triangle B C D ∧ right_triangle B C D ∧ hypotenuse C = distance A C)

-- Theorem to prove such a tetrahedron does not exist
theorem no_tetrahedron_with_all_right_triangles : ¬ Tetrahedron_with_all_right_triangles := 
  by {
    sorry  -- Proof omitted
  }

end no_tetrahedron_with_all_right_triangles_l132_132981


namespace children_count_l132_132364

theorem children_count 
  (A B C : Finset ℕ)
  (hA : A.card = 7)
  (hB : B.card = 6)
  (hC : C.card = 5)
  (hA_inter_B : (A ∩ B).card = 4)
  (hA_inter_C : (A ∩ C).card = 3)
  (hB_inter_C : (B ∩ C).card = 2)
  (hA_inter_B_inter_C : (A ∩ B ∩ C).card = 1) :
  (A ∪ B ∪ C).card = 10 := 
by
  sorry

end children_count_l132_132364


namespace trapezoid_area_l132_132148

theorem trapezoid_area :
  let y := λ x : ℝ, x
  ∧ let line1 := λ x : ℝ, 15
  ∧ let line2 := λ x : ℝ, 5
  ∧ let line3 := λ y : ℝ, 5
  ∧ let base1 := 15 - 5
  ∧ let base2 := 15 - 5
  ∧ let height := 5
  in (base1 + base2) * height / 2 = 50 := by
  sorry

end trapezoid_area_l132_132148


namespace problem_1_problem_2_l132_132659

open Set

variable (A B M : Set ℝ) (a : ℝ)

def A := { x : ℝ | -2 ≤ x ∧ x ≤ 2 }
def B := { x : ℝ | 1 < x }
def M := { x : ℝ | a < x ∧ x < a + 6 }

theorem problem_1 : compl B ∩ A = { x : ℝ | -2 ≤ x ∧ x ≤ 1 } :=
sorry

theorem problem_2 (a : ℝ) : A ∪ M = M ↔ -4 < a ∧ a < -2 :=
sorry


end problem_1_problem_2_l132_132659


namespace possible_b4b7_products_l132_132038

theorem possible_b4b7_products (b : ℕ → ℤ) (d : ℤ)
  (h_arith_sequence : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_product_21 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = 21 :=
by
  sorry

end possible_b4b7_products_l132_132038


namespace find_AB_in_30_60_90_triangle_l132_132990

theorem find_AB_in_30_60_90_triangle
  (A B C : Type)
  [Triangle A B C]
  (angle_BAC : angle A B C = 60)
  (BC_length : side B C = 6) :
  side A B = 2 * sqrt 3 := 
sorry

end find_AB_in_30_60_90_triangle_l132_132990


namespace leah_daily_savings_l132_132018

theorem leah_daily_savings 
  (L : ℝ)
  (h1 : 0.25 * 24 = 6)
  (h2 : ∀ (L : ℝ), (L * 20) = 20 * L)
  (h3 : ∀ (L : ℝ), 2 * L * 12 = 24 * L)
  (h4 :  6 + 20 * L + 24 * L = 28) 
: L = 0.5 :=
by
  sorry

end leah_daily_savings_l132_132018


namespace files_remaining_l132_132218

def total_files : ℕ := 26 + 36
def deleted_files : ℕ := 48
def remaining_files : ℕ := total_files - deleted_files

theorem files_remaining (h1 : total_files = 62) (h2 : deleted_files = 48) : remaining_files = 14 := by
  rw [h1, h2]
  rfl

end files_remaining_l132_132218


namespace length_of_AB_l132_132370

theorem length_of_AB {ABCD : Type} [EuclideanGeometry ABCD]
    (A B C D P Q : ABCD → Point) (h_rectangle : Rectangle A B C D)
    (on_side : P ∈ Segment B C) (BP_eq : BP = 12) (CP_eq : CP = 6)
    (tan_APD_eq_4 : tan (angle A P D) = 4) :
    length AB = 11 := by sorry

end length_of_AB_l132_132370


namespace find_ellipse_equation_concyclic_points_and_circle_l132_132654

-- Definitions from conditions
def ellipse (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def eccentricity (a b e : ℝ) := e = (real.sqrt (a^2 - b^2)) / a
def circle (x y b : ℝ) := x^2 + y^2 = b^2
def tangent_to_line (x y b : ℝ) := real.abs (x - y + real.sqrt 2) / real.sqrt 2 = b

-- Problem: Prove the standard equation of the ellipse
theorem find_ellipse_equation
  (a b e : ℝ) (ha : a > b) (hb : b > 0) (he : eccentricity a b e) (he2 : e = real.sqrt 2 / 2)
  (tang : ∃ x y : ℝ, circle x y b ∧ tangent_to_line x y b) :
  ellipse 1 1 (real.sqrt 2) 1 :=
sorry

-- Additional Definitions from conditions for Part II
def focus_of_ellipse (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)
def line_passing_through_focus (x y k : ℝ) := y = k * (x - 1)
def vector_sum_zero (OM ON OH : ℝ × ℝ) := OM + ON + OH = (0, 0)
def symmetric_point (H O G : ℝ × ℝ) := G = -H

-- Problem: Prove concyclicity of points and find circle's center and radius
theorem concyclic_points_and_circle
  (a b e : ℝ) (ha : a > b) (hb : b > 0) (he : eccentricity a b e) (he2 : e = real.sqrt 2 / 2)
  (tang : ∃ x y : ℝ, circle x y b ∧ tangent_to_line x y b)
  (focus_F : ℝ := focus_of_ellipse a b)
  (k : ℝ := -real.sqrt 2 / 2)
  (line_l : line_passing_through_focus 1 1 k)
  (OM ON : ℝ × ℝ)
  (H : ℝ × ℝ := (-1, -real.sqrt 2 / 2))
  (G : ℝ × ℝ := (1, real.sqrt 2 / 2))
  (vided_sum_zero : vector_sum_zero OM ON H) :
  ∃ center : ℝ × ℝ, ∃ radius : ℝ, concyclic_points [OM, ON, H, G] ∧ center = (1/8, -real.sqrt 2 / 8) ∧ radius = 3 * real.sqrt 11 / 8 :=
sorry

end find_ellipse_equation_concyclic_points_and_circle_l132_132654


namespace number_of_ordered_pairs_l132_132397

noncomputable def ω : ℂ := (-1 + complex.I * real.sqrt 3) / 2

theorem number_of_ordered_pairs : (finset.card {p : ℤ × ℤ | (abs (↑p.1 * ω - ↑p.2) = 1)}) = 6 :=
sorry

end number_of_ordered_pairs_l132_132397


namespace number_of_lines_l132_132188

open Classical

noncomputable
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1

theorem number_of_lines (f : ℝ) (d : ℝ) (a b : ℝ → Prop) :
  (∃ x y, hyperbola x y ∧ f = x) →
  (∃ x1 y1 x2 y2, hyperbola x1 y1 ∧ hyperbola x2 y2 ∧ a x1 y1 ∧ b x2 y2 ∧ (x1 - x2)^2 + (y1 - y2)^2 = d^2) →
  d = 4 →
  f = sqrt(3) →
  (a x y ↔ ∃ x' y', hyperbola x' y' ∧ x = x') →
  (b x y ↔ ∃ x' y', hyperbola x' y' ∧ y = y') →
  ∃ (n : ℕ), n = 3 := 
by
  -- Proof is skipped
  sorry

end number_of_lines_l132_132188


namespace no_convex_polyhedron_with_seven_edges_exists_convex_polyhedron_with_n_edges_l132_132802

theorem no_convex_polyhedron_with_seven_edges : ¬ ∃ P : Polyhedron, P.is_convex ∧ P.edges = 7 :=
by
  sorry

theorem exists_convex_polyhedron_with_n_edges (n : ℕ) (h : n ≥ 6 ∧ n ≠ 7) :
  ∃ P : Polyhedron, P.is_convex ∧ P.edges = n :=
by
  sorry

end no_convex_polyhedron_with_seven_edges_exists_convex_polyhedron_with_n_edges_l132_132802


namespace initial_apples_count_l132_132074

-- Definitions of conditions
variables (teachers_apples friends_apples eaten_apples remaining_apples : ℕ)
variable (initial_apples : ℕ)
variable (Sarah : Prop)
variable (gave_each_teacher : ∀ t, Sarah → teachers_apples = 16)
variable (gave_each_friend : ∀ f, Sarah → friends_apples = 5)
variable (ate_apple : Sarah → eaten_apples = 1)
variable (remaining : Sarah → remaining_apples = 3)

-- Theorem stating the problem
theorem initial_apples_count (h1 : teachers_apples = 16) (h2 : friends_apples = 5) (h3 : eaten_apples = 1) (h4 : remaining_apples = 3) :
  initial_apples = teachers_apples + friends_apples + eaten_apples + remaining_apples :=
  begin
    sorry
  end

end initial_apples_count_l132_132074


namespace directrix_of_parabola_l132_132838

theorem directrix_of_parabola :
  ∀ (x : ℝ), y = x^2 / 4 → y = -1 :=
sorry

end directrix_of_parabola_l132_132838


namespace reciprocal_of_neg_one_div_2023_l132_132854

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l132_132854


namespace geometric_sequence_sums_l132_132870

theorem geometric_sequence_sums (a q : ℝ) (n : ℕ) (h_n : n = 5)
  (h1 : a * (q ^ n - 1) / (q - 1) = 11)
  (h2 : a ^ 2 * (q ^ (2 * n) - 1) / (q ^ 2 - 1) = 341)
  (h3 : a ^ 3 * (q ^ (3 * n) - 1) / (q ^ 3 - 1) = 3641) :
  (a = 1 ∧ q = -2 ∧ n = 5 ∧ (λ i, a * q ^ i) '' (fin 5) = {1, -2, 4, -8, 16})
  ∨ (a = 16 ∧ q = -1/2 ∧ n = 5 ∧ (λ i, a * q ^ i) '' (fin 5) = {16, -8, 4, -2, 1}) :=
  sorry

end geometric_sequence_sums_l132_132870


namespace avg_difference_l132_132452

def avg (a b c : ℕ) := (a + b + c) / 3

theorem avg_difference : avg 14 32 53 - avg 21 47 22 = 3 :=
by
  sorry

end avg_difference_l132_132452


namespace marys_final_amount_l132_132785

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

def final_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + simple_interest P r t

theorem marys_final_amount 
  (P : ℝ := 200)
  (A_after_2_years : ℝ := 260)
  (t1 : ℝ := 2)
  (t2 : ℝ := 6)
  (r : ℝ := (A_after_2_years - P) / (P * t1)) :
  final_amount P r t2 = 380 := 
by
  sorry

end marys_final_amount_l132_132785


namespace rate_of_current_l132_132868

/-- The speed of a boat in still water is 20 km/hr, and the rate of current is c km/hr.
    The distance travelled downstream in 24 minutes is 9.2 km. What is the rate of the current? -/
theorem rate_of_current (c : ℝ) (h : 24/60 = 0.4 ∧ 9.2 = (20 + c) * 0.4) : c = 3 :=
by
  sorry  -- Proof is not required, only the statement is necessary.

end rate_of_current_l132_132868


namespace find_p_q_r_divisibility_l132_132707

theorem find_p_q_r_divisibility 
  (p q r : ℝ)
  (h_div : ∀ x, (x^4 + 4*x^3 + 6*p*x^2 + 4*q*x + r) % (x^3 + 3*x^2 + 9*x + 3) = 0)
  : (p + q) * r = 15 :=
by
  -- Proof steps would go here
  sorry

end find_p_q_r_divisibility_l132_132707


namespace side_length_of_square_cookie_l132_132878

-- Define the type of the length and perimeter
def perimeter : ℝ := 17.8 -- Total perimeter is 17.8 centimeters
def number_of_sides : ℝ := 4 -- A square has 4 sides

-- Define the function to find the length of one side
def side_length (p : ℝ) (n : ℝ) : ℝ := p / n

-- The theorem to prove:
theorem side_length_of_square_cookie :
  side_length perimeter number_of_sides = 4.45 :=
by
  -- proof is omitted
  sorry

end side_length_of_square_cookie_l132_132878


namespace scientific_notation_of_1_300_000_l132_132205

-- Define the condition: 1.3 million equals 1,300,000
def one_point_three_million : ℝ := 1300000

-- The theorem statement for the question
theorem scientific_notation_of_1_300_000 :
  one_point_three_million = 1.3 * 10^6 :=
sorry

end scientific_notation_of_1_300_000_l132_132205


namespace price_without_and_with_coupon_l132_132542

theorem price_without_and_with_coupon
  (commission_rate sale_tax_rate discount_rate : ℝ)
  (cost producer_price shipping_fee: ℝ)
  (S: ℝ)
  (h_commission: commission_rate = 0.20)
  (h_sale_tax: sale_tax_rate = 0.08)
  (h_discount: discount_rate = 0.10)
  (h_producer_price: producer_price = 20)
  (h_shipping_fee: shipping_fee = 5)
  (h_total_cost: cost = producer_price + shipping_fee)
  (h_profit: 0.20 * cost = 5)
  (h_total_earn: cost + sale_tax_rate * S + 5 = 0.80 * S)
  (h_S: S = 41.67):
  S = 41.67 ∧ 0.90 * S = 37.50 :=
by
  sorry

end price_without_and_with_coupon_l132_132542


namespace concert_attendance_l132_132821

-- Define the given conditions
def buses : ℕ := 8
def students_per_bus : ℕ := 45

-- Statement of the problem
theorem concert_attendance :
  buses * students_per_bus = 360 :=
sorry

end concert_attendance_l132_132821


namespace least_addition_l132_132888

theorem least_addition (a b n : ℕ) (h_a : Nat.Prime a) (h_b : Nat.Prime b) (h_a_val : a = 23) (h_b_val : b = 29) (h_n : n = 1056) :
  ∃ m : ℕ, (m + n) % (a * b) = 0 ∧ m = 278 :=
by
  sorry

end least_addition_l132_132888


namespace smallest_circle_area_l132_132539

-- Define the points
def P1 : ℝ × ℝ := (-3, -2)
def P2 : ℝ × ℝ := (2, 4)

-- Calculate the square of the distance between P1 and P2
def dist_square (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x2 - x1) ^ 2 + (y2 - y1) ^ 2

-- The minimum area of the circle passing through P1 and P2
def min_area (d_squared : ℝ) : ℝ :=
  π * (d_squared / 4)

theorem smallest_circle_area : 
  min_area (dist_square (-3) (-2) 2 4) = 61 * π / 4 := by
  sorry

end smallest_circle_area_l132_132539


namespace john_twice_james_l132_132745

def john_age : ℕ := 39
def years_ago : ℕ := 3
def years_future : ℕ := 6
def age_difference : ℕ := 4

theorem john_twice_james {J : ℕ} (h : 39 - years_ago = 2 * (J + years_future)) : 
  (J + age_difference = 16) :=
by
  sorry  -- Proof steps here

end john_twice_james_l132_132745


namespace area_covered_by_three_layers_l132_132162

theorem area_covered_by_three_layers (A B C : ℕ) (total_wallpaper : ℕ := 300)
  (wall_area : ℕ := 180) (two_layer_coverage : ℕ := 30) :
  A + 2 * B + 3 * C = total_wallpaper ∧ B + C = total_wallpaper - wall_area ∧ B = two_layer_coverage → 
  C = 90 :=
by
  sorry

end area_covered_by_three_layers_l132_132162


namespace coefficient_of_monomial_degree_of_monomial_l132_132455

-- Definitions based on the given problem conditions
def monomial : ℚ := -1 / 7 * (x^2 * y)

-- The coefficient of the monomial is -1/7
theorem coefficient_of_monomial : coefficient monomial = -1 / 7 :=
by sorry

-- The degree of the monomial is 3
theorem degree_of_monomial : degree monomial = 3 :=
by sorry

end coefficient_of_monomial_degree_of_monomial_l132_132455


namespace cat_and_dog_positions_l132_132005

def cat_position_after_365_moves : Nat :=
  let cycle_length := 9
  365 % cycle_length

def dog_position_after_365_moves : Nat :=
  let cycle_length := 16
  365 % cycle_length

theorem cat_and_dog_positions :
  cat_position_after_365_moves = 5 ∧ dog_position_after_365_moves = 13 :=
by
  sorry

end cat_and_dog_positions_l132_132005


namespace find_p_tilde_one_l132_132233

noncomputable def p (x : ℝ) : ℝ :=
  let r : ℝ := -1 / 9
  let s : ℝ := 1
  x^2 - (r + s) * x + (r * s)

theorem find_p_tilde_one : p 1 = 0 := by
  sorry

end find_p_tilde_one_l132_132233


namespace root_of_n_identity_l132_132434

theorem root_of_n_identity (n : ℕ) (h : n > 0) : 
  real.sqrt (n : ℝ) = 2^(n-1) * ∏ k in finset.range (n-1), real.sin (k * real.pi / (2 * n)) :=
by 
  -- Proof is omitted, but the statement correctly captures the given problem.
  sorry

end root_of_n_identity_l132_132434


namespace jo_climb_8_stairs_l132_132250

def f : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 2
| (n+3) := f (n+2) + f (n+1) + f n

theorem jo_climb_8_stairs :
  f 8 = 81 :=
by
  sorry

end jo_climb_8_stairs_l132_132250


namespace min_value_M_proof_l132_132297

noncomputable def min_value_M (a b c d e f g M : ℝ) : Prop :=
  (∀ (a b c d e f g : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0 ∧ 
    a + b + c + d + e + f + g = 1 ∧ 
    M = max (max (max (max (a + b + c) (b + c + d)) (c + d + e)) (d + e + f)) (e + f + g)
  → M ≥ (1 / 3))

theorem min_value_M_proof : min_value_M a b c d e f g M :=
by
  sorry

end min_value_M_proof_l132_132297


namespace bread_rolls_count_l132_132875

theorem bread_rolls_count (total_items croissants bagels : Nat) 
  (h1 : total_items = 90) 
  (h2 : croissants = 19) 
  (h3 : bagels = 22) : 
  total_items - croissants - bagels = 49 := 
by
  sorry

end bread_rolls_count_l132_132875


namespace divisible_by_10_l132_132382

theorem divisible_by_10 : (11 * 21 * 31 * 41 * 51 - 1) % 10 = 0 := by
  sorry

end divisible_by_10_l132_132382


namespace chemical_transport_problem_l132_132366

theorem chemical_transport_problem :
  (∀ (w r : ℕ), r = w + 420 →
  (900 / r) = (600 / (10 * w)) →
  w = 30 ∧ r = 450) ∧ 
  (∀ (x : ℕ), x + 450 * 3 * 2 + 60 * x ≥ 3600 → x = 15) := by
  sorry

end chemical_transport_problem_l132_132366


namespace sin_alpha_plus_7pi_over_6_l132_132291

variables {α : ℝ}

theorem sin_alpha_plus_7pi_over_6 (h : cos (α - π / 6) + sin α = (4 / 5) * sqrt 3) :
  sin (α + 7 * π / 6) = -4 / 5 :=
sorry

end sin_alpha_plus_7pi_over_6_l132_132291


namespace tan_inequality_sum_l132_132806

theorem tan_inequality_sum (n : ℕ) (α : Fin n → ℝ)
  (h1 : ∀ i, 0 ≤ α i) (h2 : ∀ i, α i < (π / 2))
  (h3 : ∀ i j, i ≤ j → α i ≤ α j) :
  (Real.tan (α 0)) ≤ (∑ i, Real.sin (α i)) / (∑ i, Real.cos (α i)) ∧
  (∑ i, Real.sin (α i)) / (∑ i, Real.cos (α i)) ≤ (Real.tan (α (n - 1))) := 
by
  sorry

end tan_inequality_sum_l132_132806


namespace length_of_smaller_rectangle_l132_132078

/-- 
Given six identical rectangles arranged to form a larger rectangle PQRS.
Let the area of PQRS be 4800.
Let x be the length of the smaller rectangle.
Then the length x rounded to the nearest integer is 35.
-/
theorem length_of_smaller_rectangle (x : ℝ) (area_PQRS : ℝ) (h : area_PQRS = 4800) (hx : 4 * x^2 = area_PQRS) : 
  x ≈ 35 :=
by
  sorry

end length_of_smaller_rectangle_l132_132078


namespace exists_odd_white_2x2_square_l132_132087

structure Grid (n : ℕ) :=
(cells : Fin n × Fin n → Bool) -- True represents black, False represents white

def count_color (g : Grid 200) (color : Bool) : ℕ :=
Finset.card ((Finset.univ : Finset (Fin 200 × Fin 200)).filter (λ cell, g.cells cell = color))

theorem exists_odd_white_2x2_square (g : Grid 200) (h : count_color g true = count_color g false + 404) :
  ∃ i j : Fin 199, ¬ (Bool.xor (g.cells ⟨i, j⟩ = false) (Bool.xor (g.cells ⟨i, j.succ⟩ = false) (Bool.xor (g.cells ⟨i.succ, j⟩ = false) (g.cells ⟨i.succ, j.succ⟩ = false)))) :=
sorry

end exists_odd_white_2x2_square_l132_132087


namespace range_of_function_l132_132470

noncomputable def validDomain (x : ℝ) : Prop :=
  (x >= 0) ∧ (x ≠ 1)

theorem range_of_function :
  ∀ x : ℝ, validDomain x ↔ ((√x) / (x - 1)).isDefined :=
by
  -- Proof goes here
  sorry

end range_of_function_l132_132470


namespace james_older_brother_age_l132_132744

def johnAge : ℕ := 39

def ageCondition (johnAge : ℕ) (jamesAgeIn6 : ℕ) : Prop :=
  johnAge - 3 = 2 * jamesAgeIn6

def jamesOlderBrother (james : ℕ) : ℕ :=
  james + 4

theorem james_older_brother_age (johnAge jamesOlderBrotherAge : ℕ) (james : ℕ) :
  johnAge = 39 →
  (johnAge - 3 = 2 * (james + 6)) →
  jamesOlderBrotherAge = jamesOlderBrother james →
  jamesOlderBrotherAge = 16 :=
by
  sorry

end james_older_brother_age_l132_132744


namespace lattice_points_inequality_holds_l132_132720

open Int

def is_lattice_point (x y : ℤ) : Prop :=
  (|x| - 1)^2 + (|y| - 1)^2 < 2

def lattice_points_count : ℕ :=
  {p : ℤ × ℤ // is_lattice_point p.1 p.2}.to_finset.card

theorem lattice_points_inequality_holds :
  lattice_points_count = 16 :=
by
  sorry

end lattice_points_inequality_holds_l132_132720


namespace functional_relationship_minimum_total_cost_l132_132928

-- Define the cost function for type A fruit
def cost_A (x : ℕ) : ℕ := 
  if x <= 300 then 14 * x 
  else 4200 + 10 * (x - 300)

-- Define the cost function for type B fruit
def cost_B (y : ℕ) := 12 * y

-- Define the total cost function
def total_cost (x y : ℕ) : ℕ := 
  cost_A x + cost_B y

-- The conditions for the problem
variables (x y : ℕ)
axiom area_condition : x + y = 1200
axiom x_range : 300 < x ∧ x ≤ 900

-- The first part of the proof
theorem functional_relationship (h : 300 < x) : cost_A x = 10 * x + 1200 :=
by {
  rw cost_A,
  simp [h],
  sorry
}

-- The second part of the proof
theorem minimum_total_cost (h_x : 300 < x ∧ x ≤ 900) (h_area : x + y = 1200) : 
  total_cost x y = 13800 :=
by {
  sorry -- Skipping the detailed proof for brevity
}

end functional_relationship_minimum_total_cost_l132_132928


namespace determine_gx_l132_132240

/-
  Given two polynomials f(x) and h(x), we need to show that g(x) is a certain polynomial
  when f(x) + g(x) = h(x).
-/

def f (x : ℝ) : ℝ := 4 * x^5 + 3 * x^3 + x - 2
def h (x : ℝ) : ℝ := 7 * x^3 - 5 * x + 4
def g (x : ℝ) : ℝ := -4 * x^5 + 4 * x^3 - 4 * x + 6

theorem determine_gx (x : ℝ) : f x + g x = h x :=
by
  -- proof will go here
  sorry

end determine_gx_l132_132240


namespace tom_average_speed_l132_132750

theorem tom_average_speed 
  (karen_speed : ℕ) (tom_distance : ℕ) (karen_advantage : ℕ) (delay : ℚ)
  (h1 : karen_speed = 60)
  (h2 : tom_distance = 24)
  (h3 : karen_advantage = 4)
  (h4 : delay = 4/60) :
  ∃ (v : ℚ), v = 45 := by
  sorry

end tom_average_speed_l132_132750


namespace frustum_volume_correct_l132_132550

-- Define the base edge of the original pyramid
def base_edge_pyramid := 16

-- Define the height (altitude) of the original pyramid
def height_pyramid := 10

-- Define the base edge of the smaller pyramid after the cut
def base_edge_smaller_pyramid := 8

-- Define the function to calculate the volume of a square pyramid
def volume_square_pyramid (base_edge : ℕ) (height : ℕ) : ℚ :=
  (1 / 3) * (base_edge ^ 2) * height

-- Calculate the volume of the original pyramid
def V := volume_square_pyramid base_edge_pyramid height_pyramid

-- Calculate the volume of the smaller pyramid
def V_small := volume_square_pyramid base_edge_smaller_pyramid (height_pyramid / 2)

-- Calculate the volume of the frustum
def V_frustum := V - V_small

-- Prove that the volume of the frustum is 213.33 cubic centimeters
theorem frustum_volume_correct : V_frustum = 213.33 := by
  sorry

end frustum_volume_correct_l132_132550


namespace validPaintingsCount_l132_132169

def isValidConfiguration (grid : Array (Array Bool)) : Prop :=
  ∀ i j, i < 2 → j < 2 → (grid[i][j] = true) → (grid[i + 1][j] ≠ false ∧ grid[i][j + 1] ≠ false)

def countValidConfigurations : Nat :=
  (Array (Array Bool)).fold (Array 81 (Array 3 (Array 3 (fun _ _ => false))))
  0 (λ grid count => if isValidConfiguration grid then count + 1 else count)

theorem validPaintingsCount :
  countValidConfigurations = 12 :=
sorry

end validPaintingsCount_l132_132169


namespace no_natural_number_exists_l132_132624

theorem no_natural_number_exists 
  (n : ℕ) : ¬ ∃ x y : ℕ, (2 * n * (n + 1) * (n + 2) * (n + 3) + 12) = x^2 + y^2 := 
by sorry

end no_natural_number_exists_l132_132624


namespace michael_needs_flour_l132_132787

-- Define the given conditions
def total_flour : ℕ := 8
def measuring_cup : ℚ := 1/4
def scoops_to_remove : ℕ := 8

-- Prove the amount of flour Michael needs is 6 cups
theorem michael_needs_flour : 
  (total_flour - (scoops_to_remove * measuring_cup)) = 6 := 
by
  sorry

end michael_needs_flour_l132_132787


namespace find_theta_l132_132484

variables (R h θ : ℝ)
hypothesis h1 : (r₁ = R * cos θ)
hypothesis h2 : (r₂ = (R + h) * cos θ)
hypothesis h3 : (s = 2 * π * r₂ - 2 * π * r₁)
hypothesis h4 : (s = 2 * π * h * cos θ)
hypothesis h5 : (s = h)

theorem find_theta : θ = real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l132_132484


namespace total_candies_is_829_l132_132438

-- Conditions as definitions
def Adam : ℕ := 6
def James : ℕ := 3 * Adam
def Rubert : ℕ := 4 * James
def Lisa : ℕ := 2 * Rubert
def Chris : ℕ := Lisa + 5
def Emily : ℕ := 3 * Chris - 7

-- Total candies
def total_candies : ℕ := Adam + James + Rubert + Lisa + Chris + Emily

-- Theorem to prove
theorem total_candies_is_829 : total_candies = 829 :=
by
  -- skipping the proof
  sorry

end total_candies_is_829_l132_132438


namespace ratio_last_cd_l132_132013

noncomputable def length_first_cd : ℝ := 1.5
noncomputable def length_second_cd : ℝ := 1.5
noncomputable def total_length : ℝ := 6

theorem ratio_last_cd (a b T : ℝ) (h1 : a = 1.5) (h2 : b = 1.5) (h3 : T = 6) :
  let c := T - (a + b) in c / a = 2 := by
    sorry

end ratio_last_cd_l132_132013


namespace probability_of_odd_product_of_eight_rolls_l132_132567

theorem probability_of_odd_product_of_eight_rolls : 
  let odds := {1, 3, 5}
      rolls := 8
      single_roll_odd_probability := 1 / 2
      all_rolls_odd_probability := (1 / 2) ^ rolls
  in all_rolls_odd_probability = 1 / 256 :=
by
  -- Definitions and statements
  let odds := {1, 3, 5}
  let rolls := 8
  let single_roll_odd_probability := 1 / 2
  let all_rolls_odd_probability := (1 / 2) ^ rolls
  -- Proof (to be filled in)
  sorry

end probability_of_odd_product_of_eight_rolls_l132_132567


namespace hours_worked_on_second_day_l132_132421

theorem hours_worked_on_second_day
  (h1 h3 : ℕ) (r total : ℕ)
  (h1_eq : h1 = 10)
  (h3_eq : h3 = 15)
  (r_eq : r = 10)
  (total_eq : total = 660) :
  ∃ h2 : ℕ, 2 * h1 + 2 * h2 + 2 * h3 = total / r ∧ h2 = 8 :=
by {
  existsi 8,
  split,
  {
    calc
      2 * h1 + 2 * 8 + 2 * h3
        = 2 * 10 + 2 * 8 + 2 * 15  : by rw [h1_eq, h3_eq]
    ... = 20 + 16 + 30            : by norm_num
    ... = 66                      : by norm_num
    ... = 660 / 10                : by rw [total_eq, r_eq]; norm_num,
  },
  {
    refl,
  }
}

end hours_worked_on_second_day_l132_132421


namespace problem1_problem2_l132_132228

-- Problem 1
theorem problem1 : (sqrt 3 - 2) ^ 2 + sqrt 27 = 7 - sqrt 3 :=
by
  sorry

-- Problem 2
theorem problem2 : (sqrt 6 - 2) * (sqrt 6 + 2) - real.cbrt 8 = 0 :=
by
  sorry

end problem1_problem2_l132_132228


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l132_132853

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l132_132853


namespace avg_salary_officers_l132_132362

-- Definitions of the given conditions
def avg_salary_employees := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 495

-- The statement to be proven
theorem avg_salary_officers : (15 * (15 * X) / (15 + 495)) = 450 :=
by
  sorry

end avg_salary_officers_l132_132362


namespace sqrt_expression_eq_l132_132591

theorem sqrt_expression_eq : 
  sqrt 5 - sqrt (4 * 5) + sqrt (90 / 2) = 2 * sqrt 5 :=
by
  sorry

end sqrt_expression_eq_l132_132591


namespace midpoints_and_circumcenter_collinear_l132_132466

-- Define the points and various entities involved as Lean structures/types
variables {A B C K L O: Point}
variables {BC: Segment}

-- Assume required conditions
axiom angle_BAK : ∠ BAK = 90
axiom angle_CAL : ∠ CAL = 90
axiom K_on_BC : K ∈ BC
axiom L_on_BC : L ∈ BC
axiom circumcenter_O : circumcenter A B C = O
axiom altitude_from_A : ∃ D, altitude A D ∧ D ∈ BC

-- Define midpoints
noncomputable def midpoint (P Q : Point) : Point := sorry

-- Variables to represent specific midpoints
noncomputable def M := midpoint K L
noncomputable def N := midpoint A (classical.some altitude_from_A)

-- Finally, the theorem stating the problem
theorem midpoints_and_circumcenter_collinear : collinear ({midpoint (A, (classical.some altitude_from_A)), midpoint (K, L), circumcenter A B C}) := sorry

end midpoints_and_circumcenter_collinear_l132_132466


namespace total_element_characteristic_sum_l132_132319

open Finset

def element_characteristic_sum (M : Finset ℕ) (A : Finset ℕ) : ℤ :=
  A.sum (λ k => (-1)^k * k)

theorem total_element_characteristic_sum :
  let M := {1, 2, 3, 4, 5}
  (∑ A in M.powerset \ ∅, element_characteristic_sum M A) = -48 :=
by
  let M := {1, 2, 3, 4, 5}
  have h : (∑ A in M.powerset \ ∅, element_characteristic_sum M A) = -48 := sorry
  exact h

end total_element_characteristic_sum_l132_132319


namespace number_of_correct_conclusions_problem_statement_l132_132840

noncomputable def f (x : ℝ) : ℝ := 2 / (x + 3)

lemma conclusion_1 : ∃ y, f 0 = y :=
begin
  use (2 / 3),
  simp [f],
end

lemma conclusion_2 : ∀ x, f x = 2 / (x + 3) :=
begin
  intro x,
  simp [f],
end

lemma conclusion_3 : ∀ x, f x = 2 / (x + 3) :=
begin
  intro x,
  simp [f],
end

lemma conclusion_4 : ∃ p, p = (-3, 0) :=
begin
  use (-3, 0),
  simp,
end

theorem number_of_correct_conclusions : (1 : ℕ) + 1 + 1 = 3 :=
begin
  ring,
end

theorem problem_statement : ∃ n : ℕ, n = 3 :=
begin
  use 3,
  exact number_of_correct_conclusions,
end

end number_of_correct_conclusions_problem_statement_l132_132840


namespace father_l132_132545

-- Let s be the circumference of the circular rink.
-- Let x be the son's speed.
-- Let k be the factor by which the father's speed is greater than the son's speed.

-- Define a theorem to state that k = 3/2.
theorem father's_speed_is_3_over_2_times_son's_speed
  (s x : ℝ) (k : ℝ) (h : s / (k * x - x) = (s / (k * x + x)) * 5) :
  k = 3 / 2 :=
by {
  sorry
}

end father_l132_132545


namespace find_theta_l132_132483

variables (R h θ : ℝ)
hypothesis h1 : (r₁ = R * cos θ)
hypothesis h2 : (r₂ = (R + h) * cos θ)
hypothesis h3 : (s = 2 * π * r₂ - 2 * π * r₁)
hypothesis h4 : (s = 2 * π * h * cos θ)
hypothesis h5 : (s = h)

theorem find_theta : θ = real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l132_132483


namespace polar_intersection_l132_132377

-- Define polar line l
def polar_line (ρ θ : ℝ) : Prop := ρ * cos (θ - π / 4) = 2 * sqrt 2

-- Define polar circle C
def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * sin θ

-- Define rectangular coordinates P and Q
def P := (0, 2) : ℝ × ℝ
def Q := (1, 3) : ℝ × ℝ

-- Define parametric line equation PQ
def line_PQ (t a b : ℝ) : ℝ × ℝ := (3 * t + a, (b / 2) * (3 * t) + 1)

theorem polar_intersection : 
  (∃ θ ρ, polar_line ρ θ ∧ polar_circle ρ θ ∧ (ρ = 4 ∧ θ = π / 2) ∨ (ρ = 2 * sqrt 2 ∧ θ = π / 4)) ∧ 
  ∃ a b, (P, Q ∈ line_PQ 0 a b) ∧ (a = -1 ∧ b = 2) :=
by
  sorry

end polar_intersection_l132_132377


namespace find_x_ineq_solution_l132_132255

open Set

theorem find_x_ineq_solution :
  {x : ℝ | (x - 2) / (x - 4) ≥ 3} = Ioc 4 5 := 
sorry

end find_x_ineq_solution_l132_132255


namespace jack_trips_l132_132012

theorem jack_trips (tank_volume : ℕ) (bucket_volume : ℕ) (jill_trip_count : ℕ) (jack_trip_factor : ℕ)
  (jill_buckets_per_trip : ℕ) (jack_buckets_per_trip : ℕ): 
  tank_volume = 600 →
  bucket_volume = 5 →
  jill_trip_count = 30 →
  jill_buckets_per_trip = 1 →
  jack_buckets_per_trip = 2 →
  (jack_trip_factor = 2) →
  (jack_trip_factor / 2 * jill_buckets_per_trip * bucket_volume = jack_buckets_per_trip * bucket_volume) →
  (tank_volume - jill_trip_count * jill_buckets_per_trip * bucket_volume = 450) →
  (450 / (jack_buckets_per_trip * bucket_volume) = 45) →
  (2 / (450 / (jack_buckets_per_trip * bucket_volume)) = 10 / jack_trip_factor) →
  (jack_trip_factor = 9) :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10,
  sorry,
end

end jack_trips_l132_132012


namespace smallest_integer_in_set_l132_132114

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 157) (h2 : greatest = 169) :
  ∃ (smallest : ℤ), smallest = 145 :=
by
  -- Setup the conditions
  have set_cons_odd : True := trivial
  -- Known facts
  have h_median : median = 157 := by exact h1
  have h_greatest : greatest = 169 := by exact h2
  -- We must prove
  existsi 145
  sorry

end smallest_integer_in_set_l132_132114


namespace max_abs_sum_l132_132775

theorem max_abs_sum (n : ℕ) (a : ℕ → ℝ) 
  (h : ∀ (m : ℕ), m ≤ n → |∑ k in finset.range (m + 1), a k / (k + 1)| ≤ 1) :
  |∑ k in finset.range (n + 1), a k| ≤ 2 ^ (n - 1) - n - 1 :=
begin
  sorry
end

end max_abs_sum_l132_132775


namespace find_a_l132_132302

def is_odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f (x)

noncomputable def f (a x : ℝ) : ℝ :=
  real.log a ((x ^ 2 + 3 * a).sqrt - x)

theorem find_a (a : ℝ) :
  is_odd_function (f a) → a = 1 / 3 :=
sorry

end find_a_l132_132302


namespace ab_inequality_l132_132025

variables (a b c : ℝ)
variables (h1 : a ≠ b) (h2 : 0 < c)
variables (h3 : a^4 - 2019 * a = c) (h4 : b^4 - 2019 * b = c)

theorem ab_inequality : -real.sqrt c < a * b ∧ a * b < 0 :=
by 
  sorry

end ab_inequality_l132_132025


namespace period_of_f_extremum_of_f_parity_of_g_l132_132309

def f (x : Real) : Real :=
  2 * sin (x / 4) * cos (x / 4) - 2 * sqrt 3 * (sin (x / 4))^2 + sqrt 3

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x := 
  by 
    use (4 * Real.pi) 
    sorry

theorem extremum_of_f : (∀ x, f x ≥ -2) ∧ (∃ y, f y = -2) ∧ (∀ x, f x ≤ 2) ∧ (∃ z, f z = 2) := 
  by 
    sorry

def g (x : Real) : Real := f (x + Real.pi / 3)

theorem parity_of_g : ∀ x, g (-x) = g x :=
  by 
    sorry

end period_of_f_extremum_of_f_parity_of_g_l132_132309


namespace max_value_of_quadratic_l132_132464

theorem max_value_of_quadratic :
  ∃ y : ℚ, ∀ x : ℚ, -x^2 - 3 * x + 4 ≤ y :=
sorry

end max_value_of_quadratic_l132_132464


namespace carly_total_practice_time_l132_132230

def butterfly_hours_per_month : ℝ := 3 * 4
def backstroke_hours_per_month : ℝ := 2 * 6
def breaststroke_hours_per_month : ℝ := 1.5 * 5
def freestyle_hours_per_month : ℝ := 2.5 * 3
def underwater_hours_per_month : ℝ := 1 * 3
def relay_hours_per_month : ℝ := 4 * 2
def total_hours_before_rest : ℝ :=
  butterfly_hours_per_month + backstroke_hours_per_month + breaststroke_hours_per_month +
  freestyle_hours_per_month + underwater_hours_per_month + relay_hours_per_month

def rest_days_in_month : ℕ := 4
def holidays_in_month : ℕ := 1
def total_rest_days_and_holiday : ℕ := rest_days_in_month + holidays_in_month

def average_daily_practice_time : ℝ :=
  (butterfly_hours_per_month + backstroke_hours_per_month + breaststroke_hours_per_month +
  freestyle_hours_per_month + underwater_hours_per_month) / (4 + 6 + 5 + 3 : ℝ)

def total_time_lost_to_rest : ℝ := average_daily_practice_time * total_rest_days_and_holiday

def total_practice_time_in_month : ℝ := total_hours_before_rest - total_time_lost_to_rest

theorem carly_total_practice_time : total_practice_time_in_month = 38.35 := by
  -- Proof goes here
  sorry

end carly_total_practice_time_l132_132230


namespace box_cubes_no_green_face_l132_132092

theorem box_cubes_no_green_face (a b c : ℕ) (h_a2 : a > 2) (h_b2 : b > 2) (h_c2 : c > 2)
  (h_no_green_face : (a-2)*(b-2)*(c-2) = (a*b*c) / 3) :
  (a, b, c) = (7, 30, 4) ∨ (a, b, c) = (8, 18, 4) ∨ (a, b, c) = (9, 14, 4) ∨
  (a, b, c) = (10, 12, 4) ∨ (a, b, c) = (5, 27, 5) ∨ (a, b, c) = (6, 12, 5) ∨
  (a, b, c) = (7, 9, 5) ∨ (a, b, c) = (6, 8, 6) :=
sorry

end box_cubes_no_green_face_l132_132092


namespace proof_problem_l132_132684

open Set

theorem proof_problem (A : Set ℕ) (B : Set ℤ) (hA : A = {0, 1, 2}) (hB : B = {x - y | x ∈ A, y ∈ A}) :
  B.card = 5 := 
sorry

end proof_problem_l132_132684


namespace option_B_not_explained_by_LC_l132_132155

/-- Le Chatelier's principle states that if a change in conditions is imposed on a system at equilibrium,
the equilibrium will shift in a direction that tends to reduce that change. -/
def le_chateliers_principle (P : Prop) (shift : Prop → Prop) := ∀ (c : Prop), P = c → shift c

/-- Option A: Immediately after opening a beer bottle, a large amount of foam appears in the bottle. -/
def option_A : Prop := ∃ (foam : Prop), foam

/-- Option B: Steel rusts easily in humid air. -/
def option_B : Prop := ∃ (rust : Prop), rust

/-- Option C: In the laboratory, the method of displacing saturated brine to collect chlorine gas is commonly used. -/
def option_C : Prop := ∃ (chlorine : Prop), chlorine

/-- Option D: In the industrial production of sulfuric acid, excess air is used to improve the utilization rate of sulfur dioxide. -/
def option_D : Prop := ∃ (sulfuric_acid : Prop), sulfuric_acid

/-- Fact that we need to prove: Option B cannot be explained by Le Chatelier's principle. -/
theorem option_B_not_explained_by_LC :
  ¬ (le_chateliers_principle option_B (λ c, ¬c)) :=
sorry

end option_B_not_explained_by_LC_l132_132155


namespace gcd_three_numbers_l132_132601

def a : ℕ := 13680
def b : ℕ := 20400
def c : ℕ := 47600

theorem gcd_three_numbers (a b c : ℕ) : Nat.gcd (Nat.gcd a b) c = 80 :=
by
  sorry

end gcd_three_numbers_l132_132601


namespace no_natural_number_n_exists_l132_132623

theorem no_natural_number_n_exists (n : ℕ) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 * n * (n + 1) * (n + 2) * (n + 3) + 12 := 
sorry

end no_natural_number_n_exists_l132_132623


namespace geometric_sequence_second_term_l132_132096

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l132_132096


namespace asymptotes_of_hyperbola_l132_132827

theorem asymptotes_of_hyperbola : ∀ (x y : ℝ), (9 * x^2 - 4 * y^2 = -36) →
  (y = (3/2) * x ∨ y = -(3/2) * x) :=
begin
  intros x y h,
  sorry
end

end asymptotes_of_hyperbola_l132_132827


namespace domain_of_f_equals_0_to_4_l132_132615

def domain_of_function (x : ℝ) : Prop :=
  2 - sqrt (3 - sqrt (4 - x)) ≥ 0 ∧
  3 - sqrt (4 - x) ≥ 0 ∧
  4 - x ≥ 0

theorem domain_of_f_equals_0_to_4 :
  ∀ x, domain_of_function x ↔ (0 ≤ x ∧ x ≤ 4) := 
by 
  sorry

end domain_of_f_equals_0_to_4_l132_132615


namespace reciprocal_of_neg_one_div_2023_l132_132857

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l132_132857


namespace license_plate_count_l132_132599

theorem license_plate_count:
  let num_letters := 26 in
  let num_digits := 10 in
  let num_letter_or_digit := num_letters + num_digits in
  (num_letters * num_digits * num_letter_or_digit) = 9360 :=
by
  sorry

end license_plate_count_l132_132599


namespace sum_of_24_terms_l132_132000

variable (a_1 d : ℝ)

def a (n : ℕ) : ℝ := a_1 + (n - 1) * d

theorem sum_of_24_terms 
  (h : (a 5 + a 10 + a 15 + a 20 = 20)) : 
  (12 * (2 * a_1 + 23 * d) = 120) :=
by
  sorry

end sum_of_24_terms_l132_132000


namespace smallest_n_satisfies_congruence_l132_132152

theorem smallest_n_satisfies_congruence :
  ∃ n : ℕ, 0 < n ∧ 725 * n % 40 = 1025 * n % 40 ∧ (∀ m : ℕ, 0 < m ∧ 725 * m % 40 = 1025 * m % 40 → n ≤ m → n = 2) :=
begin
  -- Proof omitted
  sorry
end

end smallest_n_satisfies_congruence_l132_132152


namespace david_marks_in_math_l132_132237

-- Define given conditions
def marks_in_english : ℝ := 70
def marks_in_physics : ℝ := 80
def marks_in_chemistry : ℝ := 63
def marks_in_biology : ℝ := 65
def average_marks : ℝ := 68.2
def number_of_subjects : ℝ := 5

-- Define the theorem for proving the marks in Mathematics
theorem david_marks_in_math :
  let total_marks := average_marks * number_of_subjects,
      total_marks_others := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology in
  total_marks - total_marks_others = 63 := 
by
  -- Let Lean 4 perform calculations
  let total_marks := average_marks * number_of_subjects
  let total_marks_others := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology
  sorry

end david_marks_in_math_l132_132237


namespace base_conversion_and_addition_l132_132251

noncomputable def convert_base8_to_base10 (n : ℕ) : ℕ :=
  2 * 8^2 + 5 * 8^1 + 4 * 8^0

noncomputable def convert_base4_to_base10_1 (n : ℕ) : ℕ :=
  1 * 4^1 + 3 * 4^0

noncomputable def convert_base5_to_base10 (n : ℕ) : ℕ :=
  2 * 5^2 + 0 * 5^1 + 2 * 5^0

noncomputable def convert_base4_to_base10_2 (n : ℕ) : ℕ :=
  2 * 4^1 + 2 * 4^0

theorem base_conversion_and_addition :
  let a := convert_base8_to_base10 254
  let b := convert_base4_to_base10_1 13
  let c := convert_base5_to_base10 202
  let d := convert_base4_to_base10_2 22
  (a/b + c/d) ≈ 30 := sorry

end base_conversion_and_addition_l132_132251


namespace f_2015_eq_sin_l132_132770

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 := cos
| (n+1) := λ x, (f n)' x

theorem f_2015_eq_sin : ∀ x, f 2015 x = sin x := by
  sorry

end f_2015_eq_sin_l132_132770


namespace shirt_price_percentage_l132_132199

variable (original_price : ℝ) (final_price : ℝ)

def calculate_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_new_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_final_price (p : ℝ) : ℝ := 0.85 * p

theorem shirt_price_percentage :
  (original_price = 60) →
  (final_price = calculate_final_price (calculate_new_sale_price (calculate_sale_price original_price))) →
  (final_price / original_price) * 100 = 54.4 :=
by
  intros h₁ h₂
  sorry

end shirt_price_percentage_l132_132199


namespace bobs_password_probability_l132_132952

theorem bobs_password_probability :
  (Prob_Bob_password (odd_single_digit, letter, positive_single_digit) = 9 / 20) := 
sorry

-- Define the events
def non_negative_single_digit (n : ℕ) : Prop := n ≤ 9
def odd_single_digit (n : ℕ) : Prop := n % 2 = 1 ∧ non_negative_single_digit n
def positive_single_digit (n : ℕ) : Prop := n > 0 ∧ n ≤ 9

-- Define the structure of the password and the probability function
def letter (c : Char) : Prop := c.isAlpha -- Assumes we have some character type checking method

def Prob_Bob_password (part1 : ℕ → Prop, part2 : Char → Prop, part3 : ℕ → Prop) : ℝ := 
  (Prob part1) * (Prob part2) * (Prob part3)

-- Placeholder probability function, have to define them accurately
noncomputable def Prob (event : ℕ → Prop) : ℝ := sorry
noncomputable def Prob (event : Char → Prop) : ℝ := sorry

end bobs_password_probability_l132_132952


namespace part1_part2_l132_132313

def f (x : ℝ) : ℝ := 3^x
def g (x : ℝ) : ℝ := 9^x
def h (x : ℝ) : ℝ := f x / (f x + real.sqrt 3)

theorem part1 (x : ℝ) : g x - 8 * f x - g 1 = 0 ↔ x = 2 :=
by sorry

theorem part2 : (finset.range 2019).sum (λ k, if k = 0 ∨ k = 2019 then 0 else h (k / 2019)) = 1009 :=
by sorry

end part1_part2_l132_132313


namespace fifth_expression_pattern_general_expression_for_odd_n_series_value_given_condition_l132_132793

theorem fifth_expression_pattern :
  let a := 9;
  let b := 11;
  (1 : ℚ) / (a * b) = ((1 : ℚ) / 2) * ((1 : ℚ) / a - (1 : ℚ) / b) := 
by 
  sorry

theorem general_expression_for_odd_n (n : ℕ) (hn : n % 2 = 1) :
  (1 : ℚ) / (n * (n + 2)) = ((1 : ℚ) / 2) * ((1 : ℚ) / n - (1 : ℚ) / (n + 2)) := 
by 
  sorry

theorem series_value_given_condition (a b : ℚ) (ha : |a - 1| = 0) (hb : (b - 3)^2 = 0) :
  (∑ k in Finset.range 51, (1 : ℚ) / ((a + 2 * k) * (b + 2 * k))) = 51 / 103 := 
by 
  sorry

end fifth_expression_pattern_general_expression_for_odd_n_series_value_given_condition_l132_132793


namespace frequency_of_sample_in_interval_l132_132284

def sample_frequencies : List (Nat × Nat) :=
  [(10, 20), 2, (20, 30), 3, (30, 40), 4, (40, 50), 5, (50, 60), 4, (60, 70), 2]

def sample_capacity : Nat := 20

def frequency_in_interval (freqs : List (Nat × Nat)) (capacity : Nat) : Nat :=
  freqs.foldl (fun acc el => if el.1.2 ≤ 50 then acc + el.2 else acc) 0

theorem frequency_of_sample_in_interval :
  frequency_in_interval sample_frequencies sample_capacity / sample_capacity.toRat = 0.70 :=
by
  -- Proof steps would go here.
  sorry

end frequency_of_sample_in_interval_l132_132284


namespace levi_initial_score_is_eight_l132_132415

def initial_score_levi_satisfies_conditions : Prop :=
  ∃ (L : ℕ), ∀ (B : ℕ) (additional_baskets_brother needs_to_beat : ℕ), 
    B = 12 →
    additional_baskets_brother = 3 →
    needs_to_beat = 5 →
    (L + 12 = B + additional_baskets_brother + needs_to_beat) →
    L = 8

theorem levi_initial_score_is_eight : initial_score_levi_satisfies_conditions :=
by {
  use 8,
  intros B additional_baskets_brother needs_to_beat hB haditional hneeds hL,
  rw [hB, haditional, hneeds] at hL,
  linarith,
}

end levi_initial_score_is_eight_l132_132415


namespace union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l132_132781

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition definitions
def set_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 6}
def set_B : Set ℝ := {x | -2 < x ∧ x < 9}
def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Proof statement (1)
theorem union_A_B_eq_univ (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  A ∪ B = Set.univ := by sorry

theorem inter_compl_A_B_eq_interval (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  (Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6} := by sorry

-- Proof statement (2)
theorem subset_B_range_of_a (a : ℝ) (h : set_C a ⊆ set_B) :
  -2 ≤ a ∧ a ≤ 8 := by sorry

end union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l132_132781


namespace compute_expression_l132_132405

noncomputable def a : ℚ := 4 / 7
noncomputable def b : ℚ := 5 / 6

theorem compute_expression : (a ^ 3) * (b ^ (-4)) = 82944 / 214375 := by
  sorry

end compute_expression_l132_132405


namespace sum_of_diagonals_l132_132762

-- Definitions for the given conditions
def isCyclicPentagon (ABCDE : Polygon) : Prop := 
  -- Assume that the pentagon is cyclic 
  (is_cyclic ABCDE)

def lengthsOfPentagonSidesAndDiagonals (AB CD BC DE AE AC : ℝ) : Prop := 
  AB = 4 ∧ CD = 4 ∧ BC = 7 ∧ DE = 7 ∧ AE = 13 ∧ AC = 9

-- The question translated to a Lean statement
theorem sum_of_diagonals (ABCDE : Polygon) 
  (h1: isCyclicPentagon ABCDE) 
  (h2: lengthsOfPentagonSidesAndDiagonals 4 4 7 7 13 9) : 
  let BD := 107 / 9
  let CE := 101 / 9
  BD + CE = 208 / 9 :=
by
  sorry

end sum_of_diagonals_l132_132762


namespace ratio_volume_sphere_to_hemisphere_l132_132121

-- Definitions based on conditions
def radius_sphere (r : ℝ) := r
def radius_hemisphere (r : ℝ) := 3 * r

-- Definition of the volume of a sphere of radius r
def volume_sphere (r : ℝ) := (4 / 3) * Real.pi * (radius_sphere r) ^ 3

-- Definition of the volume of a hemisphere of radius 3r
def volume_hemisphere (r : ℝ) := (1 / 2) * (4 / 3) * Real.pi * (radius_hemisphere r) ^ 3

-- The statement to prove the desired ratio
theorem ratio_volume_sphere_to_hemisphere (r : ℝ) : 
  (volume_sphere r) / (volume_hemisphere r) = 2 / 27 :=
by
  sorry

end ratio_volume_sphere_to_hemisphere_l132_132121


namespace domain_of_f_l132_132457

def f (x : ℝ) : ℝ := (1 / x) + Real.sqrt (x + 1)

theorem domain_of_f :
  {x : ℝ | x ≠ 0 ∧ x + 1 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x < 0} ∪ {x : ℝ | 0 < x} :=
by
  sorry -- proof is omitted

end domain_of_f_l132_132457


namespace find_theta_l132_132485

variables (R h θ : ℝ)
hypothesis h1 : (r₁ = R * cos θ)
hypothesis h2 : (r₂ = (R + h) * cos θ)
hypothesis h3 : (s = 2 * π * r₂ - 2 * π * r₁)
hypothesis h4 : (s = 2 * π * h * cos θ)
hypothesis h5 : (s = h)

theorem find_theta : θ = real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l132_132485


namespace closed_form_a_n_l132_132026

/-- Define \( f_0 \), the function from \( \mathbb{Z}^2 \) to \{0,1\} -/
def f_0 : ℤ × ℤ → ℕ
| (0, 0) := 1
| _      := 0

/-- Define the recursive function \( f_m \) -/
def f_m : ℕ → ℤ × ℤ → ℕ
| 0     := f_0
| (m+1) := λ (x : ℤ × ℤ), (f_m m x + 
                            ∑ j in [-1, 0, 1], 
                            ∑ k in [-1, 0, 1], 
                            f_m m (x.1 + j, x.2 + k)) % 2

/-- Define \( a_n \), the number of pairs \((x,y)\) such that \( f_n(x,y) = 1 \) -/
def a_n (n : ℕ) : ℕ :=
 finset.card { p : ℤ × ℤ | f_m n p = 1 }.to_finset

/-- Prove the closed form for \( a_n \) -/
theorem closed_form_a_n (n : ℕ) : a_n n = 
  let k := nat.find (λ k, 2^k > n + 1) - 1 in
  (5 * 4^k + (-2)^(k + 1)) / 3 :=
by
  sorry

end closed_form_a_n_l132_132026


namespace sqrt_x_add_one_meaningful_l132_132714

theorem sqrt_x_add_one_meaningful (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 1)) ↔ x ≥ -1 := 
by
  sorry

end sqrt_x_add_one_meaningful_l132_132714


namespace triangle_ratio_l132_132741

theorem triangle_ratio (A B C D : Type) 
  [triangle ABC] (h_right: ∠C = 90) (h_CB_CA : CB > CA) 
  (h_point_D: point_on D BC) 
  (h_angle: ∠CAD = 2 * ∠DAB) 
  (h_AC_AD: AC / AD = 2 / 3): 
  (let m := 5 in let n := 9 in m + n = 14) :=
sorry

end triangle_ratio_l132_132741


namespace solution_set_l132_132252

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x - 2) / (x - 4) ≥ 3

theorem solution_set :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | 4 < x ∧ x ≤ 5} :=
by
  sorry

end solution_set_l132_132252


namespace mold_radius_l132_132352

-- Define the diameter of the mold
def diameter : ℝ := 4

-- Define the relationship between diameter and radius
def radius (d : ℝ) : ℝ := d / 2

-- State the theorem:
theorem mold_radius : radius diameter = 2 :=
by
  -- Proof not required, placeholder for the solution
  sorry

end mold_radius_l132_132352


namespace range_of_a_l132_132277

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∃ x : ℝ, |x - 4| + |x + 3| < a) : a > 7 :=
sorry

end range_of_a_l132_132277


namespace exponent_sum_l132_132955

theorem exponent_sum : (-2:ℝ) ^ 4 + (-2:ℝ) ^ (3 / 2) + (-2:ℝ) ^ 1 + 2 ^ 1 + 2 ^ (3 / 2) + 2 ^ 4 = 32 := by
  sorry

end exponent_sum_l132_132955


namespace fence_length_l132_132471

theorem fence_length {w l : ℕ} (h1 : l = 2 * w) (h2 : 30 = 2 * l + 2 * w) : l = 10 := by
  sorry

end fence_length_l132_132471


namespace parallel_lines_l132_132975

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end parallel_lines_l132_132975


namespace problem_statement_l132_132946

noncomputable def fA (x : ℝ) : ℝ := (1/2)^x
noncomputable def fB (x : ℝ) : ℝ := x^2 - 4 * x + 4
noncomputable def fC (x : ℝ) : ℝ := |x + 2|
noncomputable def fD (x : ℝ) : ℝ := log (1/2) x

def increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

theorem problem_statement :
  ∀ f, (f = fA ∨ f = fB ∨ f = fC ∨ f = fD) →
  (∀ x₁ x₂, (0 < x₁ ∧ 0 < x₂) → (x₁ - x₂) * (f x₁ - f x₂) > 0) ↔ (f = fC) :=
begin
  sorry
end

end problem_statement_l132_132946


namespace largest_perimeter_l132_132940

-- Define the problem's conditions
def side1 := 7
def side2 := 9
def integer_side (x : ℕ) : Prop := (x > 2) ∧ (x < 16)

-- Define the perimeter calculation
def perimeter (a b c : ℕ) := a + b + c

-- The theorem statement which we want to prove
theorem largest_perimeter : ∃ x : ℕ, integer_side x ∧ perimeter side1 side2 x = 31 :=
by
  sorry

end largest_perimeter_l132_132940


namespace symmetric_image_triangle_vertices_on_incircle_proof_l132_132020

noncomputable def symmetric_image_triangle_vertices_on_incircle
  (A B C H₁ H₂ H₃ : Point)
  (T₁ T₂ T₃ : Point)
  (h₁ : Line := Line.mk A H₁)
  (h₂ : Line := Line.mk B H₂)
  (h₃ : Line := Line.mk C H₃)
  (T₁T₂ : Line := Line.mk T₁ T₂)
  (T₂T₃ : Line := Line.mk T₂ T₃)
  (T₃T₁ : Line := Line.mk T₃ T₁)
  (incircle : Circle)
  (H₁H₂_sym : Line := reflection T₁T₂ h₁)
  (H₂H₃_sym : Line := reflection T₂T₃ h₂)
  (H₃H₁_sym : Line := reflection T₃T₁ h₃) :
  Prop :=
  ∃ (A' B' C' : Point), Triangle.mk A' B' C' ≠ Triangle.nil
  ∧ incircle ⊆ Circle.mk_center_radius (A' : Point) (B' : Point) (C' : Point)

--- abbreviated definitions for Points, Lines, and other geometrical entities ---
structure Point := (x y : ℝ)
structure Line := (p1 p2 : Point)
structure Circle := (center : Point) (radius : ℝ)
structure Triangle := | mk (A B C : Point) | nil
def Line.mk (p1 p2 : Point) : Line := Line.mk p1 p2
def Circle.mk_center_radius (A B C : Point) : Circle := Circle.mk (midpoint A B C) (distance A B C)
def reflection (l : Line) (h : Line) : Line := sorry -- Assume a reflection function exists for simplicity
def midpoint (A B C : Point) : Point := sorry -- Midpoint of triangle's vertices A, B, C
def distance (A B C : Point) : ℝ := sorry -- Distance radius function for circle

theorem symmetric_image_triangle_vertices_on_incircle_proof :
  symmetric_image_triangle_vertices_on_incircle A B C H₁ H₂ H₃ T₁ T₂ T₃ h₁ h₂ h₃ T₁T₂ T₂T₃ T₃T₁ incircle H₁H₂_sym H₂H₃_sym H₃H₁_sym := sorry

end symmetric_image_triangle_vertices_on_incircle_proof_l132_132020


namespace sum_of_consecutive_odds_mod_14_l132_132954

theorem sum_of_consecutive_odds_mod_14 :
  let n := 10999 in
  (∑ i in Finset.range 7, n + 2 * i) % 14 = 7 :=
by 
  sorry

end sum_of_consecutive_odds_mod_14_l132_132954


namespace shortest_distance_to_circle_l132_132151

noncomputable def distance := (point1 point2 : ℝ × ℝ) → 
  real.sqrt ((point1.1 - point2.1)^2 + (point1.2 - point2.2)^2)

theorem shortest_distance_to_circle : 
  let center := (9, 4)
  let radius := real.sqrt 56
  let point := (5, -1)
  let dist := distance point center in
  x^2 - 18*x + y^2 - 8*y + 153 = 0 → 
  dist - radius = real.sqrt 41 - 2 * real.sqrt 14
:= by
  sorry

end shortest_distance_to_circle_l132_132151


namespace equilateral_triangle_side_length_l132_132760

theorem equilateral_triangle_side_length
  (s : ℝ)
  (A B C D : Point)
  (h_equilateral : equilateral_triangle ABC)
  (h_midpoint_D : midpoint D B C)
  (h_Omega : circle Ω (segment A D))
  (h_area : (circle_area Ω) - (triangle_area ABC) = 800 * real.pi - 600 * real.sqrt(3))
  : s = 80 := 
sorry

end equilateral_triangle_side_length_l132_132760


namespace max_area_triangle_ABC_l132_132051

noncomputable def area_triangle (p : ℝ) : ℝ :=
  (3 / 2) * abs ((p - 2) * (p - 5))

theorem max_area_triangle_ABC :
  ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ area_triangle p = 27 / 8 :=
by
  use 7 / 2
  split
  norm_num
  split
  norm_num
  unfold area_triangle 
  norm_num
  sorry

end max_area_triangle_ABC_l132_132051


namespace distance_from_s_to_plane_of_triangle_l132_132467

theorem distance_from_s_to_plane_of_triangle :
  ∀ (P Q R S : Type) (PQ QR RP : ℝ),
    PQ = 9 →
    QR = 10 →
    RP = 11 →
    distance_from_s_to_plane (P, Q, R) S (circle_radius := 15) = (15 * sqrt 39) / 4 →
    ((15 + 39 + 4) = 58) := sorry

end distance_from_s_to_plane_of_triangle_l132_132467


namespace jemma_total_grasshoppers_l132_132015

def number_of_grasshoppers_on_plant : Nat := 7
def number_of_dozen_baby_grasshoppers : Nat := 2
def number_in_a_dozen : Nat := 12

theorem jemma_total_grasshoppers :
  number_of_grasshoppers_on_plant + number_of_dozen_baby_grasshoppers * number_in_a_dozen = 31 := by
  sorry

end jemma_total_grasshoppers_l132_132015


namespace volume_of_box_is_correct_l132_132902

def metallic_sheet_initial_length : ℕ := 48
def metallic_sheet_initial_width : ℕ := 36
def square_cut_side_length : ℕ := 8

def box_length : ℕ := metallic_sheet_initial_length - 2 * square_cut_side_length
def box_width : ℕ := metallic_sheet_initial_width - 2 * square_cut_side_length
def box_height : ℕ := square_cut_side_length

def box_volume : ℕ := box_length * box_width * box_height

theorem volume_of_box_is_correct : box_volume = 5120 := by
  sorry

end volume_of_box_is_correct_l132_132902


namespace solution_set_inequality_l132_132298

variable {f : ℝ → ℝ}

theorem solution_set_inequality (h_diff : ∀ x > 0, differentiable_at ℝ f x)
  (h_ineq : ∀ x > 0, f(x) > x * deriv f x) :
  {x : ℝ | x > 0 ∧ x^2 * f(1 / x) - f x < 0} = {x | 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_inequality_l132_132298


namespace max_k_element_subsets_l132_132021

theorem max_k_element_subsets (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  ∃ t, t = n - k + 1 ∧ (∀ S1 S2 ∈ {T : finset ℕ | T.card = k ∧ T ⊆ finset.range (n+1)},
                         S1 ≠ S2 → (S1 ∪ S2 = S1 ∨ S1 ∪ S2 = S2 → S1.card ≤ t ∧ S2.card ≤ t)) := sorry

end max_k_element_subsets_l132_132021


namespace triangle_cos_A_and_a_correct_l132_132380

noncomputable def cos_A_and_a (b c : ℝ) (area : ℝ) : ℝ × ℝ :=
  if (b = 3) ∧ (c = 1) ∧ (area = real.sqrt 2) then
    let sinA := (2 * real.sqrt 2) / 3 in
    let cosA := real.sqrt (1 - sinA^2) in
    if cosA = 1 / 3 then (cosA, 2 * real.sqrt 2) else (-cosA, 2 * real.sqrt 3)
  else (0, 0)

theorem triangle_cos_A_and_a_correct :
  cos_A_and_a 3 1 (real.sqrt 2) = (1/3, 2 * real.sqrt 2) :=
by sorry

end triangle_cos_A_and_a_correct_l132_132380


namespace age_ratio_proof_l132_132139

variable (j a x : ℕ)

/-- Given conditions about Jack and Alex's ages. -/
axiom h1 : j - 3 = 2 * (a - 3)
axiom h2 : j - 5 = 3 * (a - 5)

def age_ratio_in_years : Prop :=
  (3 * (a + x) = 2 * (j + x)) → (x = 1)

theorem age_ratio_proof : age_ratio_in_years j a x := by
  sorry

end age_ratio_proof_l132_132139


namespace sin_cos_tan_min_value_l132_132261

open Real

theorem sin_cos_tan_min_value :
  ∀ x : ℝ, (sin x)^2 + (cos x)^2 = 1 → (sin x)^4 + (cos x)^4 + (tan x)^2 ≥ 3/2 :=
by
  sorry

end sin_cos_tan_min_value_l132_132261


namespace color_entire_square_in_3_folds_l132_132740

-- Definitions based on the conditions in the problem
def InitialState : Type := sorry  -- This should encode the initial state of the grid with some cells colored.

-- The problem rephrased to a Lean theorem statement.
theorem color_entire_square_in_3_folds (initial : InitialState) : 
  exists folds : ℕ, folds ≤ 3 ∧ can_color_entire_square initial folds :=
sorry

end color_entire_square_in_3_folds_l132_132740


namespace positive_integers_solving_inequality_l132_132997

theorem positive_integers_solving_inequality (n : ℕ) (h1: 0 < n) : 25 - 5 * n < 15 → 2 < n := by
  sorry

end positive_integers_solving_inequality_l132_132997


namespace jon_found_marbles_l132_132055

-- Definitions based on the conditions
variables (M J B : ℕ)

-- Prove that Jon found 110 marbles
theorem jon_found_marbles
  (h1 : M + J = 66)
  (h2 : M = 2 * J)
  (h3 : J + B = 3 * M) :
  B = 110 :=
by
  sorry -- proof to be completed

end jon_found_marbles_l132_132055


namespace solution_set_of_abs_sum_l132_132867

open Set

theorem solution_set_of_abs_sum (x : ℝ) : (abs (x - 1) + abs (x - 2) > 3) ↔ (x ∈ Union (Iio 0) (Ioi 3)) :=
by
  sorry

end solution_set_of_abs_sum_l132_132867


namespace calculate_third_person_contribution_l132_132219

-- Define the entities and conditions
def third_person_contribution : ℕ :=
  80  -- As determined from the solution

def brittany_contribution (third_contribution : ℕ) : ℕ :=
  3 * third_contribution

def angela_contribution (brittany_contribution : ℕ) : ℕ :=
  3 * brittany_contribution

def total_contribution (third_contribution brittany_contribution angela_contribution : ℕ) :=
  third_contribution + brittany_contribution + angela_contribution

-- The proof problem statement
theorem calculate_third_person_contribution :
  ∀ (third_contribution brittany_contribution angela_contribution : ℕ),
    brittany_contribution = 3 * third_contribution →
    angela_contribution = 3 * brittany_contribution →
    total_contribution third_contribution brittany_contribution angela_contribution = 1040 →
    third_contribution = third_person_contribution :=
by
  intros third_contribution brittany_contribution angela_contribution
  assume h1 h2 h3
  sorry  -- Proof part is not required

end calculate_third_person_contribution_l132_132219


namespace exists_consecutive_useful_numbers_l132_132782

-- Definition of a useful number
def is_useful (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits ≠ [] ∧ -- Number should have digits
  ¬0 ∈ digits ∧ -- Does not contain zeros
  digits.nodup ∧ -- No repeated digits
  let sum_digits := digits.sum
  let prod_digits := digits.prod
  sum_digits ≠ 0 ∧ prod_digits % sum_digits = 0 -- Product of digits is multiple of sum

-- The main theorem
theorem exists_consecutive_useful_numbers : 
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 998 ∧ is_useful n ∧ is_useful (n + 1) :=
sorry

end exists_consecutive_useful_numbers_l132_132782


namespace vector_subtraction_l132_132323

theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (3, 5)) (h2 : b = (-2, 1)) :
  a - (2 : ℝ) • b = (7, 3) :=
by
  rw [h1, h2]
  simp
  sorry

end vector_subtraction_l132_132323


namespace bricks_needed_l132_132689

-- Define the dimensions of the brick
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the dimensions of the wall in meters and convert to centimeters
def wall_length : ℝ := 7 * 100  -- 7 meters to centimeters
def wall_height : ℝ := 6 * 100  -- 6 meters to centimeters
def wall_width : ℝ := 22.5

-- Calculate the volume of the brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Calculate the volume of the wall
def volume_wall : ℝ := wall_length * wall_height * wall_width

-- Calculate the number of bricks required
def number_of_bricks : ℝ := volume_wall / volume_brick

-- State the theorem we need to prove
theorem bricks_needed : number_of_bricks = 5600 :=
by 
  -- Placeholder for the proof
  sorry

end bricks_needed_l132_132689


namespace total_tubes_in_consignment_l132_132541

theorem total_tubes_in_consignment (N : ℕ) 
  (h : (5 / (N : ℝ)) * (4 / (N - 1 : ℝ)) = 0.05263157894736842) : 
  N = 20 := 
sorry

end total_tubes_in_consignment_l132_132541


namespace exists_group_round_table_l132_132209

open Finset Function

variable (P : Finset ℤ) (knows : ℤ → ℤ → Prop)

def has_at_least_three_friends (P : Finset ℤ) (knows : ℤ → ℤ → Prop) : Prop :=
  ∀ p ∈ P, (P.filter (knows p)).card ≥ 3

noncomputable def exists_even_group (P : Finset ℤ) (knows : ℤ → ℤ → Prop) : Prop :=
  ∃ S : Finset ℤ, (S ⊆ P) ∧ (2 < S.card) ∧ (Even S.card) ∧ (∀ p ∈ S, ∀ q ∈ S, Edge_connected p q knows S)

theorem exists_group_round_table (P : Finset ℤ) (knows : ℤ → ℤ → Prop) 
  (h : has_at_least_three_friends P knows) : 
  exists_even_group P knows :=
sorry

end exists_group_round_table_l132_132209


namespace center_number_possible_l132_132459

-- Definitions of the numbers with their prime factorizations
def num_set := { 9, 12, 18, 24, 36, 48, 96 }

-- The possible options for the center number x in the configuration
def possible_xs := { 12, 96 }

-- Lean statement for the given math problem
theorem center_number_possible (a b c d e f x : ℕ) 
  (h_mem_a : a ∈ num_set) (h_mem_b : b ∈ num_set) (h_mem_c : c ∈ num_set)
  (h_mem_d : d ∈ num_set) (h_mem_e : e ∈ num_set) (h_mem_f : f ∈ num_set)
  (h_x : x ∈ possible_xs) 
  (h_prod1 : a * x * d = b * x * e) (h_prod2 : b * x * e = c * x * f) :
  x = 12 ∨ x = 96 :=
sorry

end center_number_possible_l132_132459


namespace part1_solution_set_part2_range_of_m_l132_132310

def f (x : ℝ) : ℝ := 45 * |2 * x + 3| + |2 * x - 1|

theorem part1_solution_set (x : ℝ) : f(x) < 8 ↔ -5 / 2 < x ∧ x < 3 / 2 := 
  sorry

theorem part2_range_of_m (m : ℝ) : (∃ x : ℝ, f(x) ≤ |3 * m + 1|) ↔ (m ≤ -5 / 3 ∨ m ≥ 1) := 
  sorry

end part1_solution_set_part2_range_of_m_l132_132310


namespace paint_three_houses_time_l132_132709

-- Define the individual painting times for Sally, John, and David
def sally_time : ℝ := 4
def john_time : ℝ := 6
def david_time : ℝ := 8

-- Define the drying times for each type of paint
def dry_time_A : ℝ := 1
def dry_time_B : ℝ := 1.5
def dry_time_C : ℝ := 2

-- Calculating individual rates (houses per hour)
def sally_rate : ℝ := 1 / sally_time
def john_rate : ℝ := 1 / john_time
def david_rate : ℝ := 1 / david_time

-- Combined painting rate
def combined_rate : ℝ := sally_rate + john_rate + david_rate

-- Time to paint each house including drying time
def house1_time : ℝ := (1 / combined_rate) + dry_time_A
def house2_time : ℝ := (1 / combined_rate) + dry_time_B
def house3_time : ℝ := (1 / combined_rate) + dry_time_C

-- Total time to paint all three houses
def total_time : ℝ := house1_time + house2_time + house3_time

theorem paint_three_houses_time : total_time = 10.038 :=
by {
  -- Verify the total painting time
  rw [sally_time, john_time, david_time, dry_time_A, dry_time_B, dry_time_C],
  have sally_rate_eq : sally_rate = 0.25 := by norm_num,
  have john_rate_eq : john_rate = 0.1667 := by norm_num,
  have david_rate_eq : david_rate = 0.125 := by norm_num,

  have combined_rate_eq : combined_rate = 0.5417 := by
    rw [sally_rate_eq, john_rate_eq, david_rate_eq],
    norm_num,

  have house1_time_eq : house1_time = 2.846 := by
    rw combined_rate_eq,
    norm_num,

  have house2_time_eq : house2_time = 3.346 := by
    rw combined_rate_eq,
    norm_num,

  have house3_time_eq : house3_time = 3.846 := by
    rw combined_rate_eq,
    norm_num,

  rw [house1_time_eq, house2_time_eq, house3_time_eq],
  norm_num,
  sorry  -- Completing the proof, assuming arithmetic correctness checked above
}

end paint_three_houses_time_l132_132709


namespace quadrilateral_ABMN_perimeter_l132_132551

-- Define the setup and conditions for the problem
def triangle : Type := sorry  -- Introduce the type for triangle
def circle : Type := sorry    -- Introduce the type for circle
def line : Type := sorry      -- Introduce the type for line

axiom has_inscribed_circle (ABC : triangle) : circle
axiom line_through_center_parallel_to_base (ABC : triangle) (c : circle) : line
axiom line_intersects_sides (l : line) (ABC : triangle) : Prop

-- Given conditions
constant ABC : triangle       -- Triangle ABC
constant c : circle := has_inscribed_circle ABC  -- Circle inscribed in triangle ABC
constant lMN : line := line_through_center_parallel_to_base ABC c  -- Line MN through center and parallel to base AB
constant AB : ℝ := 5          -- Length of side AB
constant MN : ℝ := 3          -- Length of line MN

-- Define the points of intersection
noncomputable def M : line_intersects_sides lMN ABC := sorry
noncomputable def N : line_intersects_sides lMN ABC := sorry

-- Prop to state that MN is parallel to AB
axiom MN_parallel_AB {ABC : triangle} (lMN : line) : Prop := sorry

-- Define the perimeter of the quadrilateral ABMN
noncomputable def perimeter_ABMN (AB MN : ℝ) (P : Prop) : ℝ := AB + MN + (AB / 2) + (AB / 2)

-- Statement to prove
theorem quadrilateral_ABMN_perimeter (ABC : triangle) (c : circle) (lMN : line) 
                         (M : line_intersects_sides lMN ABC) (N : line_intersects_sides lMN ABC)
                         (h1 : MN_parallel_AB lMN) 
                         (h2 : AB = 5) 
                         (h3 : MN = 3) : 
perimeter_ABMN AB MN h1 = 11 := 
by
  sorry

end quadrilateral_ABMN_perimeter_l132_132551


namespace find_m_l132_132019

noncomputable def g (x m : ℝ) : ℝ := (3 * x - 5) / (m * x + 3)

theorem find_m (m : ℝ) : g(g x m) x = x ↔ m = 8 / 5 := sorry

end find_m_l132_132019


namespace Ivanka_more_months_l132_132010

variable (I : ℕ) (W : ℕ)

theorem Ivanka_more_months (hW : W = 18) (hI_W : I + W = 39) : I - W = 3 :=
by
  sorry

end Ivanka_more_months_l132_132010


namespace decrypt_messages_l132_132247

theorem decrypt_messages (enc1 enc2 enc3 : String) (orig_msg1 orig_msg2 : String) (known_msg : String) :
  enc1 = "ТПЕОИРВНТМОЛАРГЕИАВИЛЕДНМТ ААГТДЬТКУБЧКГЕИШНЕИАЯРЯ" ∧
  enc2 = "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП" ∧
  enc3 = "РТПАИОМВСВТИЕОБПРОЕННИГЪКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК" ∧
  known_msg = "МОСКВА" ∧
  (decrypt enc1 known_msg = orig_msg1) ∧
  (decrypt enc3 known_msg = orig_msg1) ∧
  (decrypt enc2 known_msg = orig_msg2) →
  (orig_msg1 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ") ∧
  (orig_msg2 = "С ЧИСТОЙ СОВЕСТЬЮ") :=
by
  sorry

end decrypt_messages_l132_132247


namespace angle_between_vectors_l132_132256

noncomputable def vector1 : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def vector2 : ℝ × ℝ × ℝ := (-2, 2, -1)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

noncomputable def cos_theta (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

noncomputable def angle_rad (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.arccos (cos_theta v1 v2)

noncomputable def angle_deg (rad : ℝ) : ℝ :=
  rad * (180 / Real.pi)

theorem angle_between_vectors :
  angle_deg (angle_rad vector1 vector2) = 127 := sorry

end angle_between_vectors_l132_132256


namespace expression_evaluation_l132_132885

theorem expression_evaluation :
  100 + (120 / 15) + (18 * 20) - 250 - (360 / 12) = 188 := by
  sorry

end expression_evaluation_l132_132885


namespace number_of_points_on_curve_l132_132427

def point := (ℝ × ℝ)

def A : point := (1, -2)
def B : point := (2, -3)
def C : point := (3, 10)

def curve (x y : ℝ) : Prop := x^2 - x * y + 2 * y + 1 = 0

def is_on_curve (p : point) : Prop := curve p.1 p.2

theorem number_of_points_on_curve : 
  (if is_on_curve A then 1 else 0) + (if is_on_curve B then 1 else 0) + (if is_on_curve C then 1 else 0) = 2 :=
sorry

end number_of_points_on_curve_l132_132427


namespace trapezoid_problem_l132_132379

theorem trapezoid_problem 
  (A B C D : Type) 
  (AB BC CD AD AC BD : ℝ)
  (h_trapezoid : -- condition for ABCD to be a trapezoid with specified perpendicular conditions)
  (h_ABC_perp : BC ⊥ AB)
  (h_BC_perp_CD : BC ⊥ CD)
  (h_AC_perp_BD : AC ⊥ BD)
  (h_AB_val : AB = sqrt 11)
  (h_AD_val : AD = sqrt 1001) 
: BC^2 = 110 :=
  sorry

end trapezoid_problem_l132_132379


namespace part_a_l132_132516

theorem part_a (a : ℕ) (b : ℕ) (h_perm : PermutationDigits a b) : a + b ≠ (10^1967 - 1) := 
sorry

end part_a_l132_132516


namespace f_diff_l132_132045

/-- Definition of a circle with center and radius -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Definition of a point -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle is called friendly if there exist 2020 tangent circles with specified tangency -/
def isFriendly (Omega : Circle) (Gamma : Circle) : Prop :=
  ∃ (ωs : Fin 2020 → Circle),
    (∀ i, tangent (ωs i) Gamma) ∧
    (∀ i, tangent (ωs i) Omega) ∧
    (∀ i, tangent (ωs i) (ωs ((i + 1) % 2020)))

/-- Function f(P) which is the sum of the areas of all friendly circles centered at P -/
def f (P : Point) : ℝ :=
  sorry -- Detailed computation involving Ω and γ here

theorem f_diff (A B : Point) (O : Point) (rA rB : ℝ) (Gamma : Circle) :
  Circle.center Gamma = O →
  Circle.radius Gamma = 1 →
  dist O A = rA →
  dist O B = rB →
  rA = 1 / 2 →
  rB = 1 / 3 →
  f A - f B = (1000 / 9) * Real.pi :=
begin
  sorry
end

end f_diff_l132_132045


namespace latitude_approx_l132_132481

noncomputable def calculate_latitude (R h : ℝ) (θ : ℝ) : ℝ :=
  if h = 0 then θ else Real.arccos (1 / (2 * Real.pi))

theorem latitude_approx (R h θ : ℝ) (h_nonzero : h ≠ 0)
  (r1 : ℝ := R * Real.cos θ)
  (r2 : ℝ := (R + h) * Real.cos θ)
  (s : ℝ := 2 * Real.pi * h * Real.cos θ)
  (condition : s = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end latitude_approx_l132_132481


namespace sum_of_star_tips_l132_132811

theorem sum_of_star_tips (n : ℕ) (h1 : n = 7) (α β γ δ ε ζ η : ℝ) :
  (α = β) ∧ (β = γ) ∧ (γ = δ) ∧ (δ = ε) ∧ (ε = ζ) ∧ (ζ = η) ∧ (η = 3 * 180 / 7) →
  7 * α = 540 :=
by
  intros h
  have α_eq := and.left h
  rw [α_eq] at h
  have β_eq := and.left (and.right h)
  rw [β_eq] at h
  have γ_eq := and.left (and.right (and.right h))
  rw [γ_eq] at h
  have δ_eq := and.left (and.right (and.right (and.right h)))
  rw [δ_eq] at h
  have ε_eq := and.left (and.right (and.right (and.right (and.right h))))
  rw [ε_eq] at h
  have ζ_eq := and.left (and.right (and.right (and.right (and.right (and.right h)))))
  rw [ζ_eq] at h
  have η_eq := and.right (and.right (and.right (and.right (and.right (and.right h)))))
  rw [←η_eq]
  sorry

end sum_of_star_tips_l132_132811


namespace tangents_use_analogical_reasoning_l132_132893

def is_tangent_line_to_circle (l : ℝ → ℝ) (c : ℝ × ℝ) (r : ℝ) :=
  ∀ t : ℝ, (l t - c.1)^2 + (l t - c.2)^2 = r^2

def is_tangent_plane_to_sphere (p : ℝ → ℝ × ℝ) (s : ℝ × ℝ × ℝ) (r : ℝ) :=
  ∀ t u : ℝ, (p t u - s.1)^2 + (p t u - s.2)^2 + (p t u - s.3)^2 = r^2

theorem tangents_use_analogical_reasoning :
  (is_tangent_line_to_circle l c r) ∧ (is_tangent_plane_to_sphere p s r) → 
  analogical_reasoning :=
sorry

end tangents_use_analogical_reasoning_l132_132893


namespace wildcats_panthers_score_difference_l132_132002

theorem wildcats_panthers_score_difference
  (wildcats_rate : ℝ) (panthers_rate : ℝ) (total_minutes : ℝ)
  (H_wildcats_rate : wildcats_rate = 2.5)
  (H_panthers_rate : panthers_rate = 1.3)
  (H_total_minutes : total_minutes = 24) :
  (wildcats_rate * total_minutes) - (panthers_rate * total_minutes) = 28.8 :=
begin
  rw [H_wildcats_rate, H_panthers_rate, H_total_minutes],
  norm_num
end

end wildcats_panthers_score_difference_l132_132002


namespace domain_of_f_l132_132617

noncomputable def f (x : ℝ) : ℝ := real.sqrt (2 - real.sqrt (3 - real.sqrt (4 - x)))

theorem domain_of_f : set.Icc (-5 : ℝ) (4 : ℝ) = { x : ℝ | (2 - real.sqrt (3 - real.sqrt (4 - x))) ≥ 0 } :=
sorry

end domain_of_f_l132_132617


namespace reciprocal_of_neg_four_l132_132264

def is_reciprocal (x y : ℚ) : Prop := x * y = 1

theorem reciprocal_of_neg_four : is_reciprocal (-4) (-1/4) :=
by
  sorry

end reciprocal_of_neg_four_l132_132264


namespace ellipse_equation_range_oa_ob_l132_132050

open Real

variables (m n : ℝ) (theta : ℝ)
noncomputable def vector_tensor (m n : ℝ) (theta : ℝ) : ℝ :=
  abs m * abs n * sin theta

def conditions := 
  let a := 3
  let b := 2 * sqrt 2
  let e := 1 / 3
  let c := 1
  let f2 := (1, 0)
  let eccentricity := e = c / a in
  vector_tensor a b (pi / 2) = 6 * sqrt 2

theorem ellipse_equation :
  conditions →
  (∀ x y : ℝ, (x^2 / 9 + y^2 / 8 = 1) ↔ (3, 2 * sqrt 2, 1)) :=
sorry

theorem range_oa_ob :
  conditions →
  (∀ OA OB : ℝ, 0 ≤ (vector_tensor OA OB theta) ∧ (vector_tensor OA OB theta) ≤ 16 / 3) :=
sorry


end ellipse_equation_range_oa_ob_l132_132050


namespace number_of_integer_points_in_intersection_l132_132321

noncomputable def A : set (ℝ × ℝ) := 
  {p | (p.1 - 3)^2 + (p.2 - 4)^2 ≤ (5 / 2)^2}

noncomputable def B : set (ℝ × ℝ) :=
  {p | (p.1 - 4)^2 + (p.2 - 5)^2 > (5 / 2)^2}

def integer_points (S : set (ℝ × ℝ)) : set (ℝ × ℝ) :=
  {p | p ∈ S ∧ p.1 ∈ ℤ ∧ p.2 ∈ ℤ}

theorem number_of_integer_points_in_intersection : 
  (integer_points (A ∩ B)).to_finset.card = 7 :=
by
  sorry

end number_of_integer_points_in_intersection_l132_132321


namespace weighted_average_l132_132238

def marks := [86, 85, 82, 87, 85]  -- Marks in subjects
def weights := [2, 3, 4, 3, 2]  -- Corresponding weights

noncomputable def weighted_sum := (List.zip_with (· * ·) marks weights).sum
noncomputable def weights_sum := weights.sum

theorem weighted_average : weighted_sum / weights_sum = 84.71 := 
by
  -- The proof will be completed here.
  sorry

end weighted_average_l132_132238


namespace nicky_total_run_time_l132_132791

theorem nicky_total_run_time (head_start_seconds : ℕ) (cristina_speed : ℕ) (nicky_speed : ℕ) 
  (distance_head_start : ℕ) (t : ℕ) (distance_cristina : ℕ) (distance_nicky_total : ℕ)
  (time_nicky : ℕ) :
  head_start_seconds = 12 → 
  cristina_speed = 5 → 
  nicky_speed = 3 → 
  distance_head_start = 36 → 
  distance_cristina = cristina_speed * t →
  distance_nicky_total = distance_head_start + nicky_speed * (t + head_start_seconds) →
  distance_cristina = distance_nicky_total →
  t = 36 →
  time_nicky = t + head_start_seconds →
  time_nicky = 48 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  rw [h1, h2, h3, h4, h7, h8] at h9
  exact h9

end nicky_total_run_time_l132_132791


namespace expected_heads_l132_132385

/-- Given 72 fair coins, where each coin is flipped up to four times until it shows heads,
    prove that the expected number of coins that show heads is 68. -/
theorem expected_heads (n : ℕ) (h : n = 72) :
  let p_heads (k : ℕ) := (1 / 2) ^ k in
  let P_heads := p_heads 1 + p_heads 2 + p_heads 3 + p_heads 4 in
  let expected_heads := n * P_heads in
  expected_heads = 68 :=
by
  sorry

end expected_heads_l132_132385


namespace unit_prices_correct_minimum_cost_l132_132450

noncomputable def unit_price_zongzi_A : ℝ := 5
noncomputable def unit_price_zongzi_B : ℝ := 6
def quantity_zongzi_A (budget : ℝ) (unit_price : ℝ) : ℝ := budget / unit_price
def quantity_zongzi_B (budget : ℝ) (unit_price : ℝ) : ℝ := budget / unit_price
def total_cost (a : ℕ) : ℝ := 5 * a + 6 * (2200 - a)

theorem unit_prices_correct :
  ∃ x y : ℝ, 
  (quantity_zongzi_A 3000 x - quantity_zongzi_B 3360 (1.2 * x) = 40) ∧
  (1.2 * y = x) ∧
  x = 5 ∧
  y = 6 := 
by {
  use [5, 6],
  simp [quantity_zongzi_A, quantity_zongzi_B],
  split,
  { field_simp,
    linarith },
  split,
  { exact mul_assoc 1.2 1 5 },
  repeat {rfl}
}

theorem minimum_cost (a : ℕ) :
  5 * a + 6 * (2200 - a) ≥ 12000 ↔ a ≤ 1200 :=
by {
  split;
  intro h,
  { linarith },
  { linarith }
}

end unit_prices_correct_minimum_cost_l132_132450


namespace olympiad_even_group_l132_132212

theorem olympiad_even_group (P : Type) [Fintype P] [Nonempty P] (knows : P → P → Prop)
  (h : ∀ p, (Finset.filter (knows p) Finset.univ).card ≥ 3) :
  ∃ (G : Finset P), G.card > 2 ∧ G.card % 2 = 0 ∧ ∀ p ∈ G, ∃ q₁ q₂ ∈ G, q₁ ≠ p ∧ q₂ ≠ p ∧ knows p q₁ ∧ knows p q₂ :=
by
  sorry

end olympiad_even_group_l132_132212


namespace triangle_construction_possible_l132_132963

noncomputable def construct_triangle (P r_a r_b : ℝ) : Prop :=
let s := P / 2 in r_a * r_b < s^2

theorem triangle_construction_possible
  (P r_a r_b : ℝ) :
  construct_triangle P r_a r_b
  ↔ (r_a * r_b < (P / 2)^2) := by
  sorry

end triangle_construction_possible_l132_132963


namespace sum_of_fourth_powers_eq_174_fourth_l132_132136

theorem sum_of_fourth_powers_eq_174_fourth :
  120 ^ 4 + 97 ^ 4 + 84 ^ 4 + 27 ^ 4 = 174 ^ 4 :=
by
  sorry

end sum_of_fourth_powers_eq_174_fourth_l132_132136


namespace reciprocal_of_neg_one_div_2023_l132_132856

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l132_132856


namespace expected_value_range_of_p_l132_132472

theorem expected_value_range_of_p (p : ℝ) (X : ℕ → ℝ) :
  (∀ n, (n = 1 → X n = p) ∧ 
        (n = 2 → X n = p * (1 - p)) ∧ 
        (n = 3 → X n = (1 - p) ^ 2)) →
  (p^2 - 3 * p + 3 > 1.75) → 
  0 < p ∧ p < 0.5 := by
  intros hprob hexp
  -- Proof would be filled in here
  sorry

end expected_value_range_of_p_l132_132472


namespace problem_statement_l132_132657

-- Definitions for the points, vector operations, and given conditions
def P : ℝ × ℝ := (real.sqrt 3, 1)
def Q (x : ℝ) : ℝ × ℝ := (real.cos x, real.sin x)
def O : ℝ × ℝ := (0, 0)

-- Vector operations
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Function f(x)
def f (x : ℝ) : ℝ := 
  dot_product (vector_sub P O) (vector_sub P (Q x))

-- The proof problem statement
theorem problem_statement : 
  (f x = 4 - 2 * real.sin (x + real.pi / 3)) ∧ 
  (∀ x, f (x + 2 * real.pi) = f x) ∧
  (∃ (A b c : ℝ), 
    A = 2 * real.pi / 3 ∧ 
    b * c = 3 ∧ 
    (b + c = 2 * real.sqrt 3) ∧ 
    (area : ℝ × ℝ × ℝ := (1/2) * b * c * real.sin A = 3 * real.sqrt 3 / 4) ∧ 
    (perimeter : ℝ := b + c + 3) = 3 + 2 * real.sqrt 3
  ) := 
sorry

end problem_statement_l132_132657


namespace cartesian_equation_line_cartesian_equation_curve_shortest_distance_point_l132_132681

theorem cartesian_equation_line 
(t : ℝ) :
  ∃ (x y : ℝ), 
  y = -3 * t + 2 ∧ 
  x = sqrt 3 * t + sqrt 3 ∧
  sqrt 3 * x + y - 5 = 0 := 
by
  exists (sqrt 3 * t + sqrt 3) (-3 * t + 2)
  simp
  sorry

theorem cartesian_equation_curve 
(θ : ℝ) 
(hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (x y : ℝ), 
  x^2 + (y - 1)^2 = 1 ∧
  x = 2 * sin θ * cos θ ∧ 
  y = 2 * sin θ * sin θ ∧
  x^2 + y^2 - 2 * y = 0 :=
by
  exists (2 * cos θ * sin θ) (2 * sin θ * sin θ)
  simp
  sorry

theorem shortest_distance_point 
(φ : ℝ) 
(hφ : 0 ≤ φ ∧ φ < 2 * Real.pi) :
  ∃ (x y : ℝ), 
  y = 1 + sin φ ∧ 
  x = cos φ ∧
  ∃ d, d = 2 - sin (φ + Real.pi / 3) ∧
  d = 1 → 
  x = sqrt 3 / 2 ∧ 
  y = 3 / 2 :=
by
  exists (cos φ) (1 + sin φ)
  simp
  cases hφ with left right
  sorry

end cartesian_equation_line_cartesian_equation_curve_shortest_distance_point_l132_132681


namespace discount_is_one_percent_l132_132198

/-
  Assuming the following:
  - market_price is the price of one pen in dollars.
  - num_pens is the number of pens bought.
  - cost_price is the total cost price paid by the retailer.
  - profit_percentage is the profit made by the retailer.
  We need to prove that the discount percentage is 1.
-/

noncomputable def discount_percentage
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (SP_per_pen : ℝ) : ℝ :=
  ((market_price - SP_per_pen) / market_price) * 100

theorem discount_is_one_percent
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (buying_condition : cost_price = (market_price * num_pens * (36 / 60)))
  (SP : ℝ)
  (selling_condition : SP = cost_price * (1 + profit_percentage / 100))
  (SP_per_pen : ℝ)
  (sp_per_pen_condition : SP_per_pen = SP / num_pens)
  (profit_condition : profit_percentage = 65) :
  discount_percentage market_price num_pens cost_price profit_percentage SP_per_pen = 1 := by
  sorry

end discount_is_one_percent_l132_132198


namespace grasshopper_at_A10_l132_132058

-- Definitions corresponding to the conditions
variable (Point : Type)
variable (Circle : Type)
variable (onCircle : Point → Circle → Prop)
variable (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : Point)
variable (center_symmetric : (∀ p q : Point, onCircle p Circle → onCircle q Circle → p ≠ q → distance p q = distance q p))
variable (initial_positions : List Point)
variable (jump : Point → Point → Point → Prop)
variable (final_positions : {x : Point // onCircle x Circle} := sorry)

axiom initial_setup : initial_positions = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]
axiom symmetric_points : center_symmetric A1 A2 ∧ center_symmetric A3 A4 ∧
                         center_symmetric A5 A6 ∧ center_symmetric A7 A8 ∧ center_symmetric A9 A10
axiom jumping_rule : ∀ (p q r : Point), jump p q r → distance p q = distance q r
axiom no_overlap : ∀ (p q r : Point), jump p q r → p ≠ q ∧ q ≠ r
axiom final_state : List Point → Point → Prop → sorry -- This axiom is required but specifics are unclear without additional proof steps

theorem grasshopper_at_A10 (C : Circle)
  (h1 : ∀ p, p ∈ initial_positions → onCircle p C)
  (h2 : initial_positions = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10])
  (h3 : List Point → Point → Prop)
  (h4 : ∀ x, final_positions x → x = A10) : final_positions A10 :=
sorry

end grasshopper_at_A10_l132_132058


namespace combined_cost_of_rose_and_daisy_l132_132220

variable (R D : ℝ)

theorem combined_cost_of_rose_and_daisy (h : 5 * R + 5 * D = 60) : R + D = 12 :=
by
  have h1: (5: ℝ) ≠ 0 := by norm_num
  calc
    R + D
        = (5 * R + 5 * D) / 5 : by ring_nf  -- Step: Divide both sides of the equation by 5
    ... = 60 / 5           : by rw [h]
    ... = 12               : by norm_num

end combined_cost_of_rose_and_daisy_l132_132220


namespace obtuse_angled_triangle_inequality_l132_132962

theorem obtuse_angled_triangle_inequality (a b c : ℝ) (A B C : ℝ) 
(h_triangle : 0 < a ∧ 0 < b ∧ 0 < c) 
(h_angle_A : 0 < A ∧ A < π) 
(h_angle_B : 0 < B ∧ B < π) 
(h_angle_C : 0 < C ∧ C < π) 
(h_obtuse : A > π / 2 ∨ B > π / 2 ∨ C > π / 2)
(h_cos_A : cos A = (b^2 + c^2 - a^2) / (2 * b * c)) 
(h_cos_B : cos B = (a^2 + c^2 - b^2) / (2 * a * c)) 
(h_cos_C : cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  a^3 * cos A + b^3 * cos B + c^3 * cos C < a * b * c := 
sorry

end obtuse_angled_triangle_inequality_l132_132962


namespace find_g_l132_132645

-- Definitions for functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := sorry -- We will define this later in the statement

theorem find_g :
  (∀ x : ℝ, g (x + 2) = f x) →
  (∀ x : ℝ, g x = 2 * x - 1) :=
by
  intros h
  sorry

end find_g_l132_132645


namespace chinese_remainder_theorem_example_l132_132600

theorem chinese_remainder_theorem_example (x : ℤ) :
  (x % 27 = 1) ∧ (x % 37 = 6) → ∃ k : ℤ, x = 110 + 999 * k := 
by
  intro h
  cases h with h1 h2
  sorry

end chinese_remainder_theorem_example_l132_132600


namespace minutes_for_90_degree_shift_l132_132331

def hourly_angle_rate : ℝ := 0.5
def minutely_angle_rate : ℝ := 6.0
def additional_angle : ℝ := 180.0
def relative_angle_rate : ℝ := minutely_angle_rate - hourly_angle_rate

theorem minutes_for_90_degree_shift : 
  (additional_angle / relative_angle_rate).round = 33 := 
by
  sorry

end minutes_for_90_degree_shift_l132_132331


namespace toms_speed_l132_132757

/--
Karen places a bet with Tom that she will beat Tom in a car race by 4 miles 
even if Karen starts 4 minutes late. Assuming that Karen drives at 
an average speed of 60 mph and that Tom will drive 24 miles before 
Karen wins the bet. Prove that Tom's average driving speed is \( \frac{300}{7} \) mph.
--/
theorem toms_speed (
  (karen_speed : ℕ) (karen_lateness : ℚ) (karen_beats_tom_by : ℕ) 
  (karen_distance_when_tom_drives_24_miles : ℕ) 
  (karen_speed = 60) 
  (karen_lateness = 4 / 60) 
  (karen_beats_tom_by = 4) 
  (karen_distance_when_tom_drives_24_miles = 24)) : 
  ∃ tom_speed : ℚ, tom_speed = 300 / 7 :=
begin
  sorry
end

end toms_speed_l132_132757


namespace num_two_digit_numbers_no_repeats_l132_132696

-- Defining the initial set of digits and the condition of non-repetition
def digits : List ℕ := [1, 2, 3, 4, 5]

-- The actual theorem to prove
theorem num_two_digit_numbers_no_repeats 
  (d : List ℕ := digits) 
  (h : ∀ (a b : ℕ), a ∈ d → b ∈ d → (a ≠ b → (a, b) ≠ (b, a))) : 
  (List.permutations d).count (λ x, x.length = 2) = 20 :=
sorry

end num_two_digit_numbers_no_repeats_l132_132696


namespace notebook_ratio_l132_132641

theorem notebook_ratio (C N : ℕ) (h1 : ∀ k, N = k / C)
  (h2 : ∃ k, N = k / (C / 2) ∧ 16 = k / (C / 2))
  (h3 : C * N = 512) : (N : ℚ) / C = 1 / 8 := 
by
  sorry

end notebook_ratio_l132_132641


namespace group_made_l132_132108

-- Definitions based on the problem's conditions
def teachers_made : Nat := 28
def total_products : Nat := 93

-- Theorem to prove that the group made 65 recycled materials
theorem group_made : total_products - teachers_made = 65 := by
  sorry

end group_made_l132_132108


namespace smallest_n_solution_exists_l132_132041

-- Definitions and conditions
def fractional_part (x : ℝ) : ℝ := x - floor x
def f (x : ℝ) : ℝ := abs (2 * fractional_part x - 1)

-- Math proof problem statement
theorem smallest_n_solution_exists : ∃ (n : ℕ), (n > 0) ∧ 
  (∀ (x : ℝ), (n * f (x * f x) = x) → ∃ k, 2012 ≤ k) := sorry

end smallest_n_solution_exists_l132_132041


namespace trajectory_correct_and_solution_exists_l132_132281

-- Define the fixed point F and the fixed line l
def F := (1, 0)
def l (x : ℝ) := x = 4

-- Define the ratio condition
def ratio_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (real.sqrt ((x - fst F) ^ 2 + y ^ 2)) / (|x - 4|) = 1 / 2

-- Define the trajectory equation
def trajectory (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x ^ 2) / 4 + (y ^ 2) / 3 = 1

-- The main proof problem statement
theorem trajectory_correct_and_solution_exists :
  (∀ P : ℝ × ℝ, ratio_condition P → trajectory P) ∧
  (∃ Q : ℝ × ℝ, Q.2 = 0 ∧ (Q = (1, 0) ∨ Q = (7, 0)) ∧
    ∃ (B C : ℝ × ℝ) (A := (-2, 0)) (M N : ℝ × ℝ),
    B ≠ C ∧
    on_trajectory B ∧ on_trajectory C ∧
    line_through F B ≠ line_through F C ∧
    tangent_at_all B C trajectory A M N Q) :=
begin
  sorry -- Proof goes here
end

end trajectory_correct_and_solution_exists_l132_132281


namespace find_second_term_l132_132098

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end find_second_term_l132_132098


namespace points_on_circle_distance_to_line_l132_132834

theorem points_on_circle_distance_to_line (x y : ℝ) :
  let circle := x^2 + y^2 + 2 * x + 4 * y - 3 = 0 in
  let line := x + y + 1 = 0 in
  let distance := sqrt 2 in
  (circle ∧ distance = sqrt 2 ∧ line) → 
  ∃ p : ℕ, p = 3 :=
by
  sorry

end points_on_circle_distance_to_line_l132_132834


namespace minimum_prism_volume_l132_132196

theorem minimum_prism_volume (l m n : ℕ) (h1 : l > 0) (h2 : m > 0) (h3 : n > 0)
    (hidden_volume_condition : (l - 1) * (m - 1) * (n - 1) = 420) :
    ∃ N : ℕ, N = l * m * n ∧ N = 630 := by
  sorry

end minimum_prism_volume_l132_132196


namespace range_of_m_l132_132524

noncomputable def f (x m : ℝ) : ℝ := -x^2 + m * x

theorem range_of_m {m : ℝ} : (∀ x y : ℝ, x ≤ y → x ≤ 1 → y ≤ 1 → f x m ≤ f y m) ↔ 2 ≤ m := 
sorry

end range_of_m_l132_132524


namespace greatest_product_three_integers_sum_2000_l132_132887

noncomputable def maxProduct (s : ℝ) : ℝ := 
  s * s * (2000 - 2 * s)

theorem greatest_product_three_integers_sum_2000 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 / 2 ∧ maxProduct x = 8000000000 / 27 := sorry

end greatest_product_three_integers_sum_2000_l132_132887


namespace chemist_target_temperature_fahrenheit_l132_132538

noncomputable def kelvinToCelsius (K : ℝ) : ℝ := K - 273.15
noncomputable def celsiusToFahrenheit (C : ℝ) : ℝ := (C * 9 / 5) + 32

theorem chemist_target_temperature_fahrenheit :
  celsiusToFahrenheit (kelvinToCelsius (373.15 - 40)) = 140 :=
by
  sorry

end chemist_target_temperature_fahrenheit_l132_132538


namespace average_of_dice_rolls_l132_132950

theorem average_of_dice_rolls (t : ℝ) (ht : t = 12) :
  let die1_faces := [1, 2, 3, 4, 5, 6]
      die2_faces := [t - 10, t, t + 10, t + 20, t + 30, t + 40]
      possible_sums := [x + y | x ← die1_faces, y ← die2_faces]
  in (list.sum possible_sums / 36) = 30.5 :=
begin
  sorry
end

end average_of_dice_rolls_l132_132950


namespace complement_of_A_in_U_l132_132780

noncomputable theory

def R := Set ℝ
def A : R := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def U : R := Set.univ

theorem complement_of_A_in_U : (U \ A = {x : ℝ | x < 1 ∨ x > 3}) :=
by {
  sorry
}

end complement_of_A_in_U_l132_132780


namespace find_k_l132_132394

-- Definitions
variable {a : ℕ → ℤ} -- The arithmetic sequence
variable {S : ℕ → ℤ} -- The sum of the first n terms of the arithmetic sequence

-- Conditions
-- (1) Definition of the sum of the first n terms for an arithmetic sequence
axiom sum_def : ∀ n, S n = ∑ i in range n, a i
-- (2) S_3 = S_8
axiom S3_eq_S8 : S 3 = S 8
-- (3) S_7 = S_k for some k ≠ 7
axiom S7_eq_Sk : ∃ k : ℕ, S 7 = S k ∧ k ≠ 7

-- Goal
theorem find_k : ∃ k : ℕ, k = 4 ∧ S 7 = S k ∧ S k ≠ 7 :=
sorry

end find_k_l132_132394


namespace inflation_costs_l132_132182

variable (L N F : ℝ)

-- Conditions
def initial_cost := L + N + F
def new_cost := 1.20 * L + 1.10 * N + 1.05 * F
def cost_difference := 97

-- Proof goal
theorem inflation_costs (h : new_cost L N F - initial_cost L N F = cost_difference) : 
  0.20 * L + 0.10 * N + 0.05 * F = 97 :=
by
  rw [new_cost, initial_cost] at h
  linarith

end inflation_costs_l132_132182


namespace tom_average_speed_l132_132749

theorem tom_average_speed 
  (karen_speed : ℕ) (tom_distance : ℕ) (karen_advantage : ℕ) (delay : ℚ)
  (h1 : karen_speed = 60)
  (h2 : tom_distance = 24)
  (h3 : karen_advantage = 4)
  (h4 : delay = 4/60) :
  ∃ (v : ℚ), v = 45 := by
  sorry

end tom_average_speed_l132_132749


namespace suraj_innings_number_l132_132823

theorem suraj_innings_number (A : ℕ) :
  (∃ A, (16 * A + 112 = 17 * (A + 6)) ∧ (A + 6 = 16)) →
  17 - 16 = 1 :=
begin
  intro h,
  cases h with A hA,
  cases hA with hEq hAvg,
  sorry,
end

end suraj_innings_number_l132_132823


namespace compute_expression_l132_132404

noncomputable def a : ℚ := 4 / 7
noncomputable def b : ℚ := 5 / 6

theorem compute_expression : (a ^ 3) * (b ^ (-4)) = 82944 / 214375 := by
  sorry

end compute_expression_l132_132404


namespace curve_C_eq_1_range_OA_MA_l132_132007

variables (ρ θ : ℝ)

-- Define the conditions
def point_M := (-2, 0 : ℝ × ℝ)
def polar_origin := (0, 0 : ℝ × ℝ)
def polar_axis := (1, 0 : ℝ × ℝ)

def point_A (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)
def point_B (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos (θ + π / 3), ρ * Real.sin (θ + π / 3))

-- Question 1
theorem curve_C_eq_1 (ρ θ : ℝ) (hB : ∃ ρ θ, ∥point_B ρ θ - point_M∥ = 1) :
  ∀ (x y : ℝ), (x, y) = point_A ρ θ → (x + 1)^2 + (y - Real.sqrt 3)^2 = 1 :=
sorry

-- Question 2
theorem range_OA_MA (ρ θ : ℝ) :
  ∃ (α : ℝ), ∀ (x y : ℝ), point_A ρ θ = (-1 + Real.cos α, Real.sqrt 3 + Real.sin α)
              ∧ 10 - 4 * Real.sqrt 3 ≤ ∥polar_origin - point_A ρ θ∥^2 + ∥point_M - point_A ρ θ∥^2
              ∧ ∥polar_origin - point_A ρ θ∥^2 + ∥point_M - point_A ρ θ∥^2 ≤ 10 + 4 * Real.sqrt 3 :=
sorry

end curve_C_eq_1_range_OA_MA_l132_132007


namespace investment_plans_count_l132_132183

theorem investment_plans_count :
  ∃ (projects cities : ℕ) (no_more_than_two : ℕ → ℕ → Prop),
    no_more_than_two projects cities →
    projects = 3 →
    cities = 5 →
    (projects ≤ 2 ∧ projects > 0) →
    ( (3.choose 2) * 5 * 4 + 5.choose 3 ) = 120 :=
by
  sorry

end investment_plans_count_l132_132183


namespace chemistry_more_than_physics_l132_132872

noncomputable def M : ℕ := sorry
noncomputable def P : ℕ := sorry
noncomputable def C : ℕ := sorry
noncomputable def x : ℕ := sorry

theorem chemistry_more_than_physics :
  M + P = 20 ∧ C = P + x ∧ (M + C) / 2 = 20 → x = 20 :=
by
  sorry

end chemistry_more_than_physics_l132_132872


namespace bucket_full_weight_l132_132171

theorem bucket_full_weight (c d : ℝ) (x y : ℝ) (h1 : x + (1 / 4) * y = c) (h2 : x + (3 / 4) * y = d) : 
  x + y = (3 * d - 3 * c) / 2 :=
by
  sorry

end bucket_full_weight_l132_132171


namespace thomas_total_blocks_l132_132493

def stack1 := 7
def stack2 := stack1 + 3
def stack3 := stack2 - 6
def stack4 := stack3 + 10
def stack5 := stack2 * 2

theorem thomas_total_blocks : stack1 + stack2 + stack3 + stack4 + stack5 = 55 := by
  sorry

end thomas_total_blocks_l132_132493


namespace hyperbola_standard_equation_cos_angle_F1PF2_l132_132300

noncomputable def ellipse : Ellipse := 
{
  a := 3,
  b := 5
}

noncomputable def hyperbola : Hyperbola := 
{
  a := 2,
  b := √12,
  c := 4
}

theorem hyperbola_standard_equation (ellipse : Ellipse) (hyperbola : Hyperbola) 
(eccentricity_sum : ellipse.eccentricity + hyperbola.eccentricity = 14/5) 
(foci_shared : ellipse.foci = (hyperbola.focus1, hyperbola.focus2)) :
  hyperbola.equation = (y^2/4 - x^2/12 = 1) := sorry

theorem cos_angle_F1PF2 (ellipse : Ellipse) (hyperbola : Hyperbola) 
(P : Point) (PF1_sum : abs (P.distance hyperbola.focus1) + abs (P.distance hyperbola.focus2) = 10)
(PF1_diff : abs (P.distance hyperbola.focus1) - abs (P.distance hyperbola.focus2) = 4) 
(F1F2_distance : hyperbola.focus1.distance hyperbola.focus2 = 8) :
  cos_angle hyperbola.focus1 P hyperbola.focus2 = -1/7 := sorry

end hyperbola_standard_equation_cos_angle_F1PF2_l132_132300


namespace find_k_if_lines_parallel_l132_132977

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end find_k_if_lines_parallel_l132_132977


namespace mixture_replacement_l132_132533

theorem mixture_replacement (A B T x : ℝ)
  (h1 : A / (A + B) = 7 / 12)
  (h2 : A = 21)
  (h3 : (A / (B + x)) = 7 / 9) :
  x = 12 :=
by
  sorry

end mixture_replacement_l132_132533


namespace combined_snakes_turtles_weight_l132_132223

noncomputable def total_alligators : ℕ := 75
noncomputable def total_monkeys : ℕ := 54
noncomputable def total_turtles : ℕ := 45
noncomputable def total_snakes : ℕ := 3

noncomputable def hiding_alligators : ℕ := 19
noncomputable def hiding_monkeys : ℕ := 32
noncomputable def hiding_turtles : ℕ := 20
noncomputable def hiding_snakes : ℕ := 82

noncomputable def total_weight : ℕ := 22800
noncomputable def snake_weight_ratio : ℕ := 2
noncomputable def alligator_weight_ratio : ℕ := 8
noncomputable def monkey_weight_ratio : ℕ := 5
noncomputable def turtle_weight_ratio : ℕ := 3

theorem combined_snakes_turtles_weight :
  (total_alligators - hiding_alligators = 56) ∧
  (total_monkeys - hiding_monkeys = 22) ∧
  (let weights_sum := snake_weight_ratio + alligator_weight_ratio + monkey_weight_ratio + turtle_weight_ratio in
   let weight_per_part := total_weight / weights_sum in
   let snakes_weight := snake_weight_ratio * weight_per_part in
   let turtles_weight := turtle_weight_ratio * weight_per_part in
   let combined_weight := snakes_weight + turtles_weight in
   combined_weight ≈ 6333.35) :=
begin
  sorry
end

end combined_snakes_turtles_weight_l132_132223


namespace no_tiling_8x8_with_triomino_no_tiling_8x8_with_one_corner_removed_l132_132168

-- Question 1: Prove that an 8x8 chessboard cannot be tiled with 3x1 triominoes
theorem no_tiling_8x8_with_triomino :
  ¬ ∃ (tiling : (fin 8 × fin 8 → bool)), 
    (∀ (x y : fin 8), tiling (x, y) = true → ∃ (dx dy : fin 8), 
    (dx + dy) % 3 = 0 ∧ tiling (x + dx, y + dy) = tiling (x + 2 * dx, y + 2 * dy) = true) := 
sorry

-- Question 2: Prove that an 8x8 chessboard with one corner removed cannot be tiled with 3x1 triominoes
theorem no_tiling_8x8_with_one_corner_removed (x y : fin 8) :
  ¬ ∃ (tiling : (fin 8 × fin 8 → bool)), 
    (tiling ((x, y)) = false ∧
    ∀ (i j : fin 8), tiling (i, j) = true → ∃ (dx dy : fin 8),
    (dx + dy) % 3 = 0 ∧ tiling (i + dx, j + dy) = tiling (i + 2 * dx, j + 2 * dy) = true) := 
sorry

end no_tiling_8x8_with_triomino_no_tiling_8x8_with_one_corner_removed_l132_132168


namespace machine_C_time_l132_132054

theorem machine_C_time (T_c : ℝ) : 
  (1/4) + (1/3) + (1/T_c) = (3/4) → T_c = 6 := 
by 
  sorry

end machine_C_time_l132_132054


namespace unique_solution_for_divisibility_l132_132995

theorem unique_solution_for_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2) ∣ (a^3 + 1) ∧ (a^2 + b^2) ∣ (b^3 + 1) → (a = 1 ∧ b = 1) :=
by
  intro h
  sorry

end unique_solution_for_divisibility_l132_132995


namespace find_digit_l132_132449

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def is_multiple_of_6 (n : ℕ) : Prop := is_even n ∧ is_multiple_of_3 n

theorem find_digit: 
  ∃ (d : ℕ), (d = 0 ∨ d = 6) ∧ is_multiple_of_6 (43560 + d) :=
by
  use d
  sorry

end find_digit_l132_132449


namespace min_time_to_cook_three_cakes_min_time_to_cook_three_cakes_proof_l132_132195

theorem min_time_to_cook_three_cakes (pots_size : Nat) (time_per_side : Nat) (cakes : Nat) : Prop :=
  (pots_size = 2) → (time_per_side = 5) → (cakes = 3) → ∃ min_time, min_time = 15

-- Theorem statement capturing our conditions
theorem min_time_to_cook_three_cakes_proof : min_time_to_cook_three_cakes 2 5 3 :=
by
  intros h1 h2 h3
  exists 15
  sorry

end min_time_to_cook_three_cakes_min_time_to_cook_three_cakes_proof_l132_132195


namespace find_numbers_l132_132883

theorem find_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : a + b = 8) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  (a = 4 + Real.sqrt 11 ∧ b = 4 - Real.sqrt 11) ∨ (a = 4 - Real.sqrt 11 ∧ b = 4 + Real.sqrt 11) :=
sorry

end find_numbers_l132_132883


namespace probability_greater_than_90_l132_132982

noncomputable def normal_distribution (μ σ : ℝ) : ProbabilityDistribution ℝ := sorry
noncomputable def normal_cdf (μ σ x : ℝ) : ℝ := sorry

def X : ProbabilityDistribution ℝ := normal_distribution 85 9

axiom P_80_to_85 : normal_cdf 85 9 85 - normal_cdf 85 9 80 = 0.35

theorem probability_greater_than_90 : (1 - normal_cdf 85 9 90) = 0.15 :=
by
  have P_85_to_90 := P_80_to_85
  sorry

end probability_greater_than_90_l132_132982


namespace inequality_on_positive_sum_one_l132_132065

theorem inequality_on_positive_sum_one 
  (n : ℕ)
  (a : Fin n → ℝ)
  (hpos : ∀ i, 0 < a i)
  (hsum : (Fin.sum Finset.univ a) = 1) :
  (Fin.sum Finset.univ (λ i, let j := (i + 1) % n in (a i) ^ 2 / (a i + a j))) ≥ 1 / 2 :=
sorry

end inequality_on_positive_sum_one_l132_132065


namespace prove_BP_squared_max_l132_132031

noncomputable def circle_diameter_proof : Prop :=
  let AB := 20 in
  ∀ (ω : Type) [circle ω] (A B C P T : ω) (hAB : diameter AB)
    (hCT_tangent : tangent (C :: T))
    (hP_foot : foot_perpendicular A (C :: T)),
    ∃ m : ℝ, maximum_length BP and
    ∃ BP : ℝ, BP^2 = 425

theorem prove_BP_squared_max : circle_diameter_proof :=
  by sorry

end prove_BP_squared_max_l132_132031


namespace reciprocal_of_minus_one_over_2023_l132_132862

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l132_132862


namespace product_max_min_sum_l132_132776

open Real

theorem product_max_min_sum (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z)
  (h₄ : 4^(sqrt (5 * x + 9 * y + 4 * z)) - 68 * 2^(sqrt (5 * x + 9 * y + 4 * z)) + 256 = 0) :
  let s := x + y + z in (max s) * (min s) = 4 :=
sorry

end product_max_min_sum_l132_132776


namespace boys_test_l132_132728

-- Define the conditions
def passing_time : ℝ := 14
def test_results : List ℝ := [0.6, -1.1, 0, -0.2, 2, 0.5]

-- Define the proof problem
theorem boys_test (number_did_not_pass : ℕ) (fastest_time : ℝ) (average_score : ℝ) :
  passing_time = 14 →
  test_results = [0.6, -1.1, 0, -0.2, 2, 0.5] →
  number_did_not_pass = 3 ∧
  fastest_time = 12.9 ∧
  average_score = 14.3 :=
by
  intros
  sorry

end boys_test_l132_132728


namespace smallest_integer_in_set_l132_132113

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 157) (h2 : greatest = 169) :
  ∃ (smallest : ℤ), smallest = 145 :=
by
  -- Setup the conditions
  have set_cons_odd : True := trivial
  -- Known facts
  have h_median : median = 157 := by exact h1
  have h_greatest : greatest = 169 := by exact h2
  -- We must prove
  existsi 145
  sorry

end smallest_integer_in_set_l132_132113


namespace product_plus_one_square_l132_132357

theorem product_plus_one_square (n : ℕ):
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 := 
  sorry

end product_plus_one_square_l132_132357


namespace probability_diff_three_l132_132942

theorem probability_diff_three (rolls : Fin 6 × Fin 6) :
  (1 / 6 : ℚ) = (6 / 36 : ℚ) :=
by
  have h_probs : ((1, 4) ∈ rolls) ∨ ((2, 5) ∈ rolls) ∨ ((3, 6) ∈ rolls) ∨ ((4, 1) ∈ rolls) ∨ ((5, 2) ∈ rolls) ∨ ((6, 3) ∈ rolls)
  sorry
  have h_total : 36 = 6 * 6
  sorry
  have h_simplify : (6 / 36 : ℚ) = (1 / 6 : ℚ)
  sorry

end probability_diff_three_l132_132942


namespace investment_plans_count_l132_132186

theorem investment_plans_count :
  let binom := Nat.choose
  ∃ (cnt : Nat), cnt = binom 5 3 * 3! + binom 5 1 * binom 4 1 * 3 ∧ cnt = 120 :=
by
  sorry

end investment_plans_count_l132_132186


namespace sin_B_sin_C_perimeter_l132_132841

-- Define the given problem's conditions and objectives in Lean

-- Definitions and assumptions for part 1
theorem sin_B_sin_C (A B C a b c : ℝ)
  (h₁ : A + B + C = π)
  (h₂ : ∀ x, 0 < x ∧ x < π)
  (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c)
  (h_area : (a * a) = 3 * (1 / 2) * a * c * sin B)
  (h_sine_rule: sin A / a = sin B / b ∧ sin A / a = sin C / c) :
  sin B * sin C = 2 / 3 := by
  sorry

-- Definitions and assumptions for part 2
theorem perimeter (A B C a b c : ℝ)
  (h₁ : A + B + C = π)
  (h₂ : ∀ x, 0 < x ∧ x < π)
  (h₃ : a = 3)
  (h_area : (a * a) = 3 * (1 / 2) * a * c * sin B)
  (h_cos : 6 * cos B * cos C = 1)
  (h_sine_rule : sin A / a = sin B / b ∧ sin A / a = sin C / c) :
  a + b + c = 3 + sqrt 33 := by
  sorry

end sin_B_sin_C_perimeter_l132_132841


namespace certain_number_minus_two_l132_132529

theorem certain_number_minus_two (x : ℝ) (h : 6 - x = 2) : x - 2 = 2 := 
sorry

end certain_number_minus_two_l132_132529


namespace area_of_triangle_PQR_l132_132956

def point := ℝ × ℝ

def P : point := (-5, 2)
def Q : point := (0, 3)
def R : point := (7, 4)

def triangle_area (p1 p2 p3 : point) : ℝ :=
  0.5 * | p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2) |

theorem area_of_triangle_PQR : triangle_area P Q R = 6 :=
by
  sorry

end area_of_triangle_PQR_l132_132956


namespace zeros_of_h_minimum_value_of_F_l132_132679

variables (x : ℝ) (n : ℕ)

def f (x : ℝ) : ℝ := (1/2) * (x + (1/x))
def g (x : ℝ) : ℝ := (1/2) * (x - (1/x))

def h (x : ℝ) : ℝ := f x + 2 * g x
def F (x : ℝ) (n : ℕ) : ℝ := (f x)^(2 * n) - (g x)^(2 * n)

theorem zeros_of_h :
  (h x = 0) → (x = sqrt 3 / 3 ∨ x = -(sqrt 3 / 3)) :=
sorry

theorem minimum_value_of_F :
  (n > 0) → (F x n ≥ 1) :=
sorry

end zeros_of_h_minimum_value_of_F_l132_132679


namespace functional_inequality_l132_132768

variable {f : ℝ → ℝ}
variable {a b : ℝ}

theorem functional_inequality (h1 : ∀ x > 0, 0 ≤ f x)
                              (h2 : ∀ x > 0, f x + x * (f' x) ≤ 0)
                              (ha : 0 < a)
                              (hb : 0 < b)
                              (h3 : a < b) :
                              a * f b ≤ b * f a := 
sorry

end functional_inequality_l132_132768


namespace ellipse_properties_l132_132307

noncomputable def ellipse_standard_eq (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def point_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  ellipse_standard_eq a b x y

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 
  2 * Real.sqrt (a^2 - b^2)

theorem ellipse_properties (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (ab_order : a > b) :
  point_on_ellipse a b 1 (Real.sqrt 3 / 2) ∧ distance_between_foci a b = 2 * Real.sqrt 3 →
  (a = 2) ∧ (b = 1) ∧ 
  (∀ x : ℝ, 0 < x → x < 2 * Real.sqrt 6 / 3) ∧
  (∀ k : ℝ, k^2 = 4 → (k = 2 ∨ k = -2)) :=
begin
  sorry
end

end ellipse_properties_l132_132307


namespace geometric_seq_second_term_l132_132101

-- Definitions
def fifth_term : ℕ → ℝ := λ n, if n = 5 then 48 else 0
def sixth_term : ℕ → ℝ := λ n, if n = 6 then 72 else 0

-- Theorem Statement
theorem geometric_seq_second_term :
  let r := sixth_term 6 / fifth_term 5,
      a := (fifth_term 5) / (r ^ 4),
      a2 := a * r in
  sixth_term 6 = 72 ∧ fifth_term 5 = 48 →
  a2 = 384 / 27 := 
begin
  sorry
end

end geometric_seq_second_term_l132_132101


namespace reciprocal_of_neg_one_div_2023_l132_132855

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l132_132855


namespace tournament_divisible_by_nine_l132_132726

theorem tournament_divisible_by_nine 
  (n : ℕ)
  (h : ∑ g_matches : ℕ, ∑ b_matches : ℕ, 4 * g_matches = 5 * b_matches) 
  (h2 : g_matches = n*(n-1)/2 + 2*n^2 - b_matches) 
  (h3 : b_matches = 2*n*(2*n - 1) + m) 
  : 9 ∣ 3 * n :=
by
  sorry

end tournament_divisible_by_nine_l132_132726


namespace derivative_and_value_l132_132991

-- Given conditions
def eqn (x y : ℝ) : Prop := 10 * x^3 + 4 * x^2 * y + y^2 = 0

-- The derivative y'
def y_prime (x y y' : ℝ) : Prop := y' = (-15 * x^2 - 4 * x * y) / (2 * x^2 + y)

-- Specific values derivatives
def y_prime_at_x_neg2_y_4 (y' : ℝ) : Prop := y' = -7 / 3

-- The main theorem
theorem derivative_and_value (x y y' : ℝ) 
  (h1 : eqn x y) (x_neg2 : x = -2) (y_4 : y = 4) : 
  y_prime x y y' ∧ y_prime_at_x_neg2_y_4 y' :=
sorry

end derivative_and_value_l132_132991


namespace final_statue_weight_l132_132515

theorem final_statue_weight
  (original_weight : ℚ)
  (first_week_cut : ℚ)
  (second_week_cut : ℚ)
  (third_week_cut : ℚ)
  (first_week_rem : ℚ := original_weight * (1 - first_week_cut))
  (second_week_rem : ℚ := first_week_rem * (1 - second_week_cut))
  (final_weight : ℚ := second_week_rem * (1 - third_week_cut)) :
  original_weight = 250 →
  first_week_cut = 0.30 →
  second_week_cut = 0.20 →
  third_week_cut = 0.25 →
  final_weight = 105 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- After first week
  have h5 : first_week_rem = 250 * (1 - 0.30), by rw h1; rw h2
  rw h5 at *
  have h6 : first_week_rem = 250 * 0.70, by rfl
  rw h6 at *
  have h7 : first_week_rem = 175, by norm_cast; norm_num
  rw h7 at *
  -- After second week
  have h8 : second_week_rem = 175 * (1 - 0.20), by rw h7; rw h3
  rw h8 at *
  have h9 : second_week_rem = 175 * 0.80, by rfl
  rw h9 at *
  have h10 : second_week_rem = 140, by norm_cast; norm_num
  rw h10 at *
  -- After third week
  have h11 : final_weight = 140 * (1 - 0.25), by rw h10; rw h4
  rw h11 at *
  have h12 : final_weight = 140 * 0.75, by rfl
  rw h12 at *
  have h13 : final_weight = 105, by norm_cast; norm_num
  exact h13

end final_statue_weight_l132_132515


namespace f_two_eq_two_f_three_eq_three_f_1999_eq_1999_l132_132282

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_mul (a b : ℕ+) (h : Nat.coprime a b) : f (a * b) = f a * f b
axiom f_add_prime (p q : ℕ+) (hp : Nat.Prime p) (hq : Nat.Prime q) : f (p + q) = f p + f q

theorem f_two_eq_two : f 2 = 2 := 
sorry

theorem f_three_eq_three : f 3 = 3 := 
sorry

theorem f_1999_eq_1999 : f 1999 = 1999 := 
sorry

end f_two_eq_two_f_three_eq_three_f_1999_eq_1999_l132_132282


namespace geometric_sequence_log_sum_l132_132003

noncomputable def geometric_sequence_sum_ten_terms (a : ℕ → ℝ) : ℝ :=
  ∑ i in finset.range 10, real.log (a (i + 1))

theorem geometric_sequence_log_sum {a : ℕ → ℝ} 
  (h_geo : ∀ m n, a(m + n) = a m * a n)
  (a3 : a 3 = 5)
  (a8 : a 8 = 2) :
  geometric_sequence_sum_ten_terms a = 5 :=
by 
  sorry

end geometric_sequence_log_sum_l132_132003


namespace mother_returns_home_at_8_05_l132_132511

noncomputable
def xiaoMing_home_time : Nat := 7 * 60 -- 7:00 AM in minutes
def xiaoMing_speed : Nat := 40 -- in meters per minute
def mother_home_time : Nat := 7 * 60 + 20 -- 7:20 AM in minutes
def meet_point : Nat := 1600 -- in meters
def stay_time : Nat := 5 -- in minutes
def return_duration_by_bike : Nat := 20 -- in minutes

theorem mother_returns_home_at_8_05 :
    (xiaoMing_home_time + (meet_point / xiaoMing_speed) + stay_time + return_duration_by_bike) = (8 * 60 + 5) :=
by
    sorry

end mother_returns_home_at_8_05_l132_132511


namespace charlie_share_l132_132905

theorem charlie_share (A B C D E : ℝ) (h1 : A = (1/3) * B)
  (h2 : B = (1/2) * C) (h3 : C = 0.75 * D) (h4 : D = 2 * E) 
  (h5 : A + B + C + D + E = 15000) : C = 15000 * (3 / 11) :=
by
  sorry

end charlie_share_l132_132905


namespace alternating_series_converges_and_sum_l132_132803

noncomputable def alternating_series (i : ℕ) : ℝ :=
  (-1)^(i + 1) * log (1 + 1/(i+1:ℝ))

theorem alternating_series_converges_and_sum :
  (∑' i, alternating_series i).summable ∧ (∑' i, alternating_series i) = log (π / 2) :=
  by sorry

end alternating_series_converges_and_sum_l132_132803


namespace min_ab_min_a_2b_l132_132296

variable {a b : ℝ}

theorem min_ab (h₀ : a > 0) (h₁ : b > 0) (h₂ : 1 / a + 2 / b = 1) : ab := 
begin
  sorry
end

theorem min_a_2b (h₀ : a > 0) (h₁ : b > 0) (h₂ : 1 / a + 2 / b = 1) : a + 2b :=
begin
  sorry
end

end min_ab_min_a_2b_l132_132296


namespace nonzero_terms_in_polynomial_l132_132335

def A (x : ℚ) : ℚ := (2 * x + 5) * (3 * x^2 + x + 6)
def B (x : ℚ) : ℚ := 4 * (x^3 - 3 * x^2 + 5 * x - 1)
def polynomial (x : ℚ) : ℚ := A x - B x

theorem nonzero_terms_in_polynomial (x : ℚ) : 
  -- The polynomial (A - B) has exactly 4 nonzero terms
  (polynomial x = 2 * x^3 + 29 * x^2 - 3 * x + 34) -> 
  list_length (list.filter (≠ 0) [2, 29, -3, 34]) = 4 :=
sorry

end nonzero_terms_in_polynomial_l132_132335


namespace total_votes_is_7000_l132_132173

-- Given conditions
variable (V : ℝ) -- Total number of votes
variable (h1 : V > 0) -- There were votes cast
variable (candidate_votes : ℝ := 0.35 * V) -- Candidate received 35% of the total votes
variable (rival_votes : ℝ := candidate_votes + 2100) -- Rival received 2100 more votes than the candidate

-- Proof statement
theorem total_votes_is_7000 (h1 : V > 0) (h2 : candidate_votes + rival_votes = V) : V = 7000 := 
by
  -- Assume V is the total number of votes
  let candidate_votes' := 0.35 * V
  let rival_votes' := candidate_votes' + 2100
  have h2' : candidate_votes' + rival_votes' = V := h2
  -- Combine like terms
  calc
    candidate_votes' + rival_votes'
      = 0.35 * V + (0.35 * V + 2100) : by sorry
    ... = 0.35 * V + 0.35 * V + 2100 : by sorry
    ... = 0.7 * V + 2100 : by sorry
    -- Simplify the equation
    have h3 : 0.7 * V + 2100 = V := h2'
    have h4 : 0.7 * V = V - 2100 := by linarith
    have h5 : 0.7 * V / 0.7 = (V - 2100) / 0.7 := by sorry
    -- Solve for V
    have h6 : V = 3000 / 0.35 := by linarith
    show V = 7000

end total_votes_is_7000_l132_132173


namespace movie_time_difference_l132_132748

theorem movie_time_difference
  (Nikki_movie : ℝ)
  (Michael_movie : ℝ)
  (Ryn_movie : ℝ)
  (Joyce_movie : ℝ)
  (total_hours : ℝ)
  (h1 : Nikki_movie = 30)
  (h2 : Michael_movie = Nikki_movie / 3)
  (h3 : Ryn_movie = (4 / 5) * Nikki_movie)
  (h4 : total_hours = 76)
  (h5 : total_hours = Michael_movie + Nikki_movie + Ryn_movie + Joyce_movie) :
  Joyce_movie - Michael_movie = 2 := 
by {
  sorry
}

end movie_time_difference_l132_132748


namespace problem_statement_l132_132663

noncomputable def f : ℝ → ℝ := λ x, if x ∈ [0, 2) then Real.log (x + 1) / Real.log 2 else f (x - 2 * floor (x / 2))

theorem problem_statement : f (-2013) + f 2014 = 1 :=
by
  -- Conditions
  have h1 : ∀ x : ℝ, f (-x) = f x := sorry,
  have h2 : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x := sorry,
  -- Proof
  sorry

end problem_statement_l132_132663


namespace rhombus_longest_diagonal_l132_132562

theorem rhombus_longest_diagonal (A : ℝ) (r1 r2 : ℝ) (hA : A = 150) (hrat : r1 / r2 = 4 / 3) :
  let x := real.sqrt (A * 2 / (r1 * r2)) in r1 * x = 20 :=
by
  -- We skip the proof as we only need to provide the problem statement.
  sorry

end rhombus_longest_diagonal_l132_132562


namespace quadratic_rewriting_l132_132605

theorem quadratic_rewriting:
  ∃ (d e f : ℤ), (∀ x : ℝ, 4 * x^2 - 28 * x + 49 = (d * x + e)^2 + f) ∧ d * e = -14 :=
by {
  sorry
}

end quadratic_rewriting_l132_132605


namespace exists_valid_circle_group_l132_132215

variable {P : Type}
variable (knows : P → P → Prop)

def knows_at_least_three (p : P) : Prop :=
  ∃ (p₁ p₂ p₃ : P), p₁ ≠ p ∧ p₂ ≠ p ∧ p₃ ≠ p ∧ knows p p₁ ∧ knows p p₂ ∧ knows p p₃

def valid_circle_group (G : List P) : Prop :=
  (2 < G.length) ∧ (G.length % 2 = 0) ∧ (∀ i, knows (G.nthLe i sorry) (G.nthLe ((i + 1) % G.length) sorry) ∧ knows (G.nthLe i sorry) (G.nthLe ((i - 1 + G.length) % G.length) sorry))

theorem exists_valid_circle_group (H : ∀ p : P, knows_at_least_three knows p) : 
  ∃ G : List P, valid_circle_group knows G := 
sorry

end exists_valid_circle_group_l132_132215


namespace problem_statement_l132_132403

theorem problem_statement (n : ℕ) (hn : \(\frac{1}{4} + \frac{1}{5} + \frac{1}{9} + \frac{1}{n} \in \mathbb{Z}\)) :
  ¬ (4 ∣ n)
 :=
begin
  sorry
end

end problem_statement_l132_132403


namespace probability_of_sum_8_9_10_l132_132881

/-- The list of face values for the first die. -/
def first_die : List ℕ := [1, 1, 3, 3, 5, 6]

/-- The list of face values for the second die. -/
def second_die : List ℕ := [1, 2, 4, 5, 7, 9]

/-- The condition to verify if the sum is 8, 9, or 10. -/
def valid_sum (s : ℕ) : Bool := s = 8 ∨ s = 9 ∨ s = 10

/-- Calculate probability of the sum being 8, 9, or 10 for the two dice. -/
def calculate_probability : ℚ :=
  let total_rolls := first_die.length * second_die.length
  let valid_rolls := 
    first_die.foldl (fun acc d1 =>
      acc + second_die.foldl (fun acc' d2 => 
        if valid_sum (d1 + d2) then acc' + 1 else acc') 0) 0
  valid_rolls / total_rolls

/-- The required probability is 7/18. -/
theorem probability_of_sum_8_9_10 : calculate_probability = 7 / 18 := 
  sorry

end probability_of_sum_8_9_10_l132_132881


namespace profit_per_unit_l132_132175

variable (a : ℝ)

theorem profit_per_unit (h₁ : increase_by (a : ℝ) 0.3 = 1.3 * h₁)
  (h₂ : discount (1.3 * a) 0.2 = 0.8 * (1.3 * a)) :
  (0.8 * (1.3 * a) - a = 0.04 * a) := by sorry

end profit_per_unit_l132_132175


namespace average_age_condition_l132_132904

theorem average_age_condition (n : ℕ) 
  (h1 : (↑n * 14) / n = 14) 
  (h2 : ((↑n * 14) + 34) / (n + 1) = 16) : 
  n = 9 := 
by 
-- Proof goes here
sorry

end average_age_condition_l132_132904


namespace correct_function_is_inverse_l132_132947

def is_decreasing (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, (0 < x1) → (0 < x2) → (x1 < x2) → (f x1 > f x2)

theorem correct_function_is_inverse :
  ∃ f : ℝ → ℝ, f = (λ x, 1 / x) ∧ is_decreasing f ∧
  ∀ g, (g = (λ x, (x - 1) ^ 2) ∨ g = (λ x, Real.exp x) ∨ g = (λ x, Real.log (x + 1))) →
  ¬ is_decreasing g :=
by
  sorry

end correct_function_is_inverse_l132_132947


namespace robin_cut_hair_l132_132072

-- Definitions as per the given conditions
def initial_length := 17
def current_length := 13

-- Statement of the proof problem
theorem robin_cut_hair : initial_length - current_length = 4 := 
by 
  sorry

end robin_cut_hair_l132_132072


namespace percentage_difference_l132_132968

variable {P Q : ℝ}

theorem percentage_difference (P Q : ℝ) : (100 * (Q - P)) / Q = ((Q - P) / Q) * 100 :=
by
  sorry

end percentage_difference_l132_132968


namespace point_to_polar_coordinates_l132_132594

theorem point_to_polar_coordinates :
  ∃ r θ : ℝ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * real.pi ∧ (let x := 2 in let y := -2 * real.sqrt 3 in 
  r = real.sqrt (x^2 + y^2) ∧ θ = if x ≠ 0 ∨ y ≠ 0 then if y/x < 0 then (2 * real.pi - real.arctan (y/x)) else real.arctan (y/x) else 0 ) ∧
  r = 4 ∧ θ = 5 * real.pi / 3 := 
begin 
  sorry 
end

end point_to_polar_coordinates_l132_132594


namespace solutions_to_equation_l132_132130

theorem solutions_to_equation :
  ∀ x : ℝ, (x + 1) * (x - 2) = x + 1 ↔ x = -1 ∨ x = 3 :=
by
  sorry

end solutions_to_equation_l132_132130


namespace solution_set_inequality_value_of_m_l132_132604

/-- 
The solution set of the inequality |x + 3| - 2x - 1 < 0 is (2, +∞).
-/
theorem solution_set_inequality : { x : ℝ | |x + 3| - 2x - 1 < 0 } = { x : ℝ | 2 < x } :=
sorry

/-- 
If the function f(x) = |x - m| + |x + 1/m| - 2 (m > 0) has a zero, then the value of m must be 1.
-/
theorem value_of_m (m : ℝ) (h₁ : 0 < m) (h₂ : ∃ x : ℝ, |x - m| + |x + 1/m| - 2 = 0) : m = 1 :=
sorry

end solution_set_inequality_value_of_m_l132_132604


namespace split_bill_evenly_l132_132915

theorem split_bill_evenly :
  let total_bill : ℝ := 514.16
  let num_people : ℕ := 9
  let smallest_unit : ℝ := 0.01
  let payment_per_person := total_bill / num_people
  floor (payment_per_person * 100 + 0.5) / 100 = 57.13 :=
  sorry

end split_bill_evenly_l132_132915


namespace product_of_solutions_positive_real_l132_132716

noncomputable def polar_form_solution : ℂ → ℂ := 
by sorry

theorem product_of_solutions_positive_real : (∏ (x in {x : ℂ | x^8 = -256 ∧ x.re > 0}), x) = 256 := 
by sorry

end product_of_solutions_positive_real_l132_132716


namespace roots_of_quadratic_l132_132966

theorem roots_of_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let Δ := b^2 - 4*a*c in
  if hΔ : Δ ≥ 0 then
    let x1 := (-b + Real.sqrt Δ) / (2*a) in
    let x2 := (-b - Real.sqrt Δ) / (2*a) in
    (a*x1^2 + b*x1 + c = 0) ∧ (a*x2^2 + b*x2 + c = 0)
  else
    let Re := -b / (2*a) in
    let Im := (Real.sqrt (-Δ)) / (2*a) in
    let x1 := Complex.mk Re Im in
    let x2 := Complex.mk Re (-Im) in
    (a * complex.norm_sq x1 + b * Re + c = 0) ∧ (a * complex.norm_sq x2 + b * Re + c = 0) :=
by
  sorry

end roots_of_quadratic_l132_132966


namespace prove_BP_squared_max_l132_132030

noncomputable def circle_diameter_proof : Prop :=
  let AB := 20 in
  ∀ (ω : Type) [circle ω] (A B C P T : ω) (hAB : diameter AB)
    (hCT_tangent : tangent (C :: T))
    (hP_foot : foot_perpendicular A (C :: T)),
    ∃ m : ℝ, maximum_length BP and
    ∃ BP : ℝ, BP^2 = 425

theorem prove_BP_squared_max : circle_diameter_proof :=
  by sorry

end prove_BP_squared_max_l132_132030


namespace mother_distance_to_timothy_l132_132140

-- Conditions
def timothy_speed : ℝ := 6
def headwind_loss : ℝ := 1
def mother_start_delay : ℝ := 1/4 -- as hours
def mother_speed : ℝ := 36

-- Effective speeds
def timothy_effective_speed : ℝ := timothy_speed - headwind_loss

-- Distance Timothy travels before mother starts
def distance_timothy_travels : ℝ := timothy_effective_speed * mother_start_delay

-- Relative speed of mother to Timothy
def relative_speed : ℝ := mother_speed - timothy_effective_speed

-- Time mother takes to reach Timothy
def time_to_reach : ℝ := distance_timothy_travels / relative_speed

-- Distance mother must drive
def distance_mother_drives : ℝ := mother_speed * time_to_reach

-- Theorem to be proved
theorem mother_distance_to_timothy : distance_mother_drives = 45 / 31 :=
by
  sorry

end mother_distance_to_timothy_l132_132140


namespace area_of_equilateral_triangle_example_l132_132424

noncomputable def area_of_equilateral_triangle_with_internal_point (a b c : ℝ) (d_pa : ℝ) (d_pb : ℝ) (d_pc : ℝ) : ℝ :=
  if h : ((d_pa = 3) ∧ (d_pb = 4) ∧ (d_pc = 5)) then
    (9 + (25 * Real.sqrt 3)/4)
  else
    0

theorem area_of_equilateral_triangle_example :
  area_of_equilateral_triangle_with_internal_point 3 4 5 3 4 5 = 9 + (25 * Real.sqrt 3)/4 :=
  by sorry

end area_of_equilateral_triangle_example_l132_132424


namespace fraction_of_solution_replaced_l132_132519

noncomputable def fraction_replaced_solution : ℝ := sorry

-- conditions as definitions
def initial_concentration : ℝ := 0.5
def replaced_concentration : ℝ := 0.3
def final_concentration : ℝ := 0.4
def total_volume : ℝ := 1.0

-- proof problem statement
theorem fraction_of_solution_replaced :
  initial_concentration * total_volume - initial_concentration * fraction_replaced_solution + replaced_concentration * fraction_replaced_solution = final_concentration * total_volume → fraction_replaced_solution = 0.5 :=
by {
  sorry
}

end fraction_of_solution_replaced_l132_132519


namespace fraction_length_BC_AD_l132_132428

theorem fraction_length_BC_AD (A B C D : Type) 
  (h : ∀ x : ℝ, x ≥ 0 → x ∈ set.univ)
  (AB BD AC CD AD BC : ℝ) :
  (AB = 3 * BD) ∧ (AC = 7 * CD) → (BC = AC - AB) → (AD = AB + BD) ∧ (AD = AC + CD) →
  (BC / AD = 1 / 8) :=
begin
  -- Conditions:
  sorry
end

end fraction_length_BC_AD_l132_132428


namespace certain_event_is_heating_water_boil_l132_132507

/-- Definition of the events given the conditions. -/
inductive Event
| toss_coin_heads : Event
| heat_water_boil : Event
| meet_acquaintance_seaside : Event
| sun_revolve_earth : Event

/-- Definition of a certain event. -/
def is_certain_event : Event → Prop
| Event.heat_water_boil := true
| _ := false

/-- The main theorem stating that heating water to 100°C at standard atmospheric pressure is a certain event. -/
theorem certain_event_is_heating_water_boil : is_certain_event Event.heat_water_boil = true :=
by
  /- The proof that heating water to 100°C at standard atmospheric pressure is a certain event. -/
  sorry

end certain_event_is_heating_water_boil_l132_132507


namespace range_of_a_l132_132835

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, sin x + a > 0) ∧ (∃ x : ℝ, log (sin x + a) = 0) ↔ 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l132_132835


namespace smallest_n_for_integer_sum_l132_132398

noncomputable def a := Real.pi / 2016

theorem smallest_n_for_integer_sum :
  let expression_sum :=
    2 * (∑ k in Finset.range n,
           (Real.cos (k^2 * a) * Real.sin (k * a)) + 
           ((Real.sin (n^2 * a)) + (Real.cos (n^2 * a)))
        )
  in (∃ n, expression_sum = Int) ∧ (∃ n, n = 84) :=
by
  sorry

end smallest_n_for_integer_sum_l132_132398


namespace weight_of_seventh_person_l132_132520

noncomputable def weight_of_six_people : ℕ := 6 * 156
noncomputable def new_average_weight (x : ℕ) : Prop := (weight_of_six_people + x) / 7 = 151

theorem weight_of_seventh_person (x : ℕ) (h : new_average_weight x) : x = 121 :=
by
  sorry

end weight_of_seventh_person_l132_132520


namespace y_increases_as_x_increases_l132_132677

-- Define the linear function y = (m^2 + 2)x
def linear_function (m x : ℝ) : ℝ := (m^2 + 2) * x

-- Prove that y increases as x increases
theorem y_increases_as_x_increases (m x1 x2 : ℝ) (h : x1 < x2) : linear_function m x1 < linear_function m x2 :=
by
  -- because m^2 + 2 is always positive, the function is strictly increasing
  have hm : 0 < m^2 + 2 := by linarith [pow_two_nonneg m]
  have hx : (m^2 + 2) * x1 < (m^2 + 2) * x2 := by exact (mul_lt_mul_left hm).mpr h
  exact hx

end y_increases_as_x_increases_l132_132677


namespace tiles_in_each_row_l132_132826

theorem tiles_in_each_row (area_sq_ft : ℕ) (tile_size_in_inches : ℕ)
  (h1 : area_sq_ft = 324) (h2 : tile_size_in_inches = 9) : 
  let side_length_in_inches := 12 * (Int.sqrt area_sq_ft)
  in (side_length_in_inches / tile_size_in_inches) = 24 :=
by
  let side_length_in_inches := 12 * (Int.sqrt area_sq_ft)
  sorry

end tiles_in_each_row_l132_132826


namespace initial_integers_is_three_l132_132930

def num_initial_integers (n m : Int) : Prop :=
  3 * n + m = 17 ∧ 2 * m + n = 23

theorem initial_integers_is_three {n m : Int} (h : num_initial_integers n m) : n = 3 :=
by
  sorry

end initial_integers_is_three_l132_132930


namespace product_of_N_values_l132_132584

-- We define the conditions given in the problem
variables {M L N : ℤ}
def minneapolis_noon_temp := L + N
def minneapolis_5pm_temp := minneapolis_noon_temp - 7
def st_louis_5pm_temp := L + 5
def temp_difference_5pm := abs ((minneapolis_noon_temp - 7) - (L + 5)) = 6

-- We state the theorem that needs to be proved
theorem product_of_N_values : 
  (minneapolis_noon_temp = L + N) ∧ 
  (minneapolis_5pm_temp = minneapolis_noon_temp - 7) ∧
  (st_louis_5pm_temp = L + 5) ∧
  temp_difference_5pm → 
  N = 6 ∨ N = 18 → 
  6 * 18 = 108 :=
by sorry

end product_of_N_values_l132_132584


namespace correct_inequalities_l132_132597

/-- Define an odd, decreasing function f : ℝ → ℝ --/
variable {f : ℝ → ℝ}
variable {a b : ℝ}

-- Assume f(x) is an odd function
axiom odd_f : ∀ x, f(x) = -f(-x)

-- Assume f(x) is a decreasing function
axiom decreasing_f : ∀ x y, x ≤ y → f(x) ≥ f(y)

-- Assume a + b ≤ 0
axiom ab_le : a + b ≤ 0

-- Inequality ① f(a) * f(-a) ≤ 0
lemma inequality_1 : f(a) * f(-a) ≤ 0 := sorry

-- Inequality ② f(b) * f(-b) ≥ 0 is false
lemma inequality_2 : ¬ (f(b) * f(-b) ≥ 0) := sorry

-- Inequality ③ f(a) + f(b) ≤ f(-a) + f(-b) is false
lemma inequality_3 : ¬ (f(a) + f(b) ≤ f(-a) + f(-b)) := sorry

-- Inequality ④ f(a) + f(b) ≥ f(-a) + f(-b)
lemma inequality_4 : f(a) + f(b) ≥ f(-a) + f(-b) := sorry

-- The correct sequence of inequalities is option B: ① and ④
theorem correct_inequalities : (inequality_1 ∧ inequality_4) := sorry

end correct_inequalities_l132_132597


namespace smallest_five_digit_number_divisible_by_primes_l132_132630

theorem smallest_five_digit_number_divisible_by_primes :
  ∃ n : ℕ, (10000 ≤ n) ∧ (n < 100000) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ (n = 11550) :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l132_132630


namespace reciprocal_is_correct_l132_132859

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l132_132859


namespace equivalent_expression_l132_132341

-- Define the conditions and the statement that needs to be proven
theorem equivalent_expression (x : ℝ) (h : x^2 - 2 * x + 1 = 0) : 2 * x^2 - 4 * x = -2 := 
  by
    sorry

end equivalent_expression_l132_132341


namespace equations_have_same_solution_l132_132091

theorem equations_have_same_solution (x c : ℝ) 
  (h1 : 3 * x + 9 = 0) (h2 : c * x + 15 = 3) : c = 4 :=
by
  sorry

end equations_have_same_solution_l132_132091


namespace sum_a_b_l132_132346

theorem sum_a_b (a b : ℕ) (h : (∏ n in finset.range (a - 3 + 1) + 3, (n + 1) / n) = 18) : a + b = 107 :=
sorry

end sum_a_b_l132_132346


namespace cost_per_chair_l132_132606

theorem cost_per_chair (total_amount_spent : ℝ) (number_of_chairs : ℝ) 
  (h1 : total_amount_spent = 180) 
  (h2 : number_of_chairs = 12) : 
  total_amount_spent / number_of_chairs = 15 := 
by 
  -- Applying the given conditions
  rw [h1, h2]
  -- Simplifying the division
  norm_num
  -- The remaining steps will show the exact division result
  sorry

end cost_per_chair_l132_132606


namespace evaluate_72_squared_minus_48_squared_l132_132984

theorem evaluate_72_squared_minus_48_squared :
  (72:ℤ)^2 - (48:ℤ)^2 = 2880 :=
by
  sorry

end evaluate_72_squared_minus_48_squared_l132_132984


namespace functional_equation_property_l132_132822

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_property (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f(x * y) + x = x * f(y) + f(x))
  (h2 : f(3) = 4) : 
  f(-27) = 56 :=
sorry

end functional_equation_property_l132_132822


namespace matrix_N_property_l132_132593

variable {x y w : ℝ}

def matrix_N : Matrix (Fin 3) (Fin 3) ℝ := 
  !![0, 3 * y, w; 
    2 * x, y, -w; 
    2 * x, -y, w]

theorem matrix_N_property : (matrix_N.transpose ⬝ matrix_N = 1) → 
  (x^2 + y^2 + w^2 = 67 / 120) :=
  by
  sorry

end matrix_N_property_l132_132593


namespace stratified_sampling_selection_l132_132642

theorem stratified_sampling_selection:
  (C 8 2) * (C 4 1) = 112 := 
by
  sorry

end stratified_sampling_selection_l132_132642


namespace eval_expr_at_neg3_l132_132983

theorem eval_expr_at_neg3 : 
  (5 + 2 * (-3) * ((-3) + 5) - 5^2) / (2 * (-3) - 5 + 2 * (-3)^3) = 32 / 65 := 
by 
  sorry

end eval_expr_at_neg3_l132_132983


namespace ventilation_duct_area_l132_132203

def cylinder_lateral_surface_area (d h : ℝ) : ℝ :=
  Real.pi * d * h

theorem ventilation_duct_area :
  cylinder_lateral_surface_area 0.2 3 = 1.884 :=
by
  sorry

end ventilation_duct_area_l132_132203


namespace true_propositions_l132_132436

noncomputable def f (x : ℝ) : ℝ := x * Real.log (|x|)

-- Predicate definitions for each proposition:
def proposition_1 : Prop :=
  ∀ x: ℝ, x < -1 / Real.exp 1 → f x = x * Real.log (-x) ∧ f x < 0

def proposition_2 : Prop :=
  ∃ xmin : ℝ, ∀ x : ℝ, x ≠ xmin → f x > f xmin

def proposition_3 : Prop :=
  ∀ x : ℝ, x > -1 ∧ x < 1 ∧ x ≠ 0 → f x > 0

def proposition_4 : Prop :=
  ∀ x1 : ℝ, ∀ y : ℝ, ∀ m : ℝ, (x1 = 1) → x1 * Real.log (|x1|) - y = m

def proposition_5 : Prop :=
  ∀ m : ℝ, ∃ znum : ℕ, znum ≤ 3 ∧ ∃ zlist : list ℝ, zlist.length = znum ∧ ∀ x ∈ zlist, (f x - m) = 0

theorem true_propositions : {1, 5} ⊆ {n | (n = 1 → proposition_1) ∧ (n = 2 → proposition_2) ∧ (n = 3 → proposition_3) ∧ (n = 4 → proposition_4) ∧ (n = 5 → proposition_5)} :=
by
  sorry

end true_propositions_l132_132436


namespace min_trig_expression_l132_132621

theorem min_trig_expression :
  ∀ (x : ℝ), (¬ ∃ n : ℤ, x = n * (π / 2)) → 
  abs (sin x + cos x + tan x + cot x + sec x + (1 / sin x)) = 2 * sqrt 2 - 1 := 
sorry

end min_trig_expression_l132_132621


namespace relationship_among_logarithm_values_l132_132698

theorem relationship_among_logarithm_values (x d : ℝ) (h₁ : 1 < x) (h₂ : x < d) :
  let a := (Real.log x / Real.log d)^2
  let b := 2 * Real.log x / Real.log d
  let c := Real.log (Real.log x / Real.log d) / Real.log d
  c < a ∧ a < b :=
by
  -- Definitions
  let log_d x := Real.log x / Real.log d

  -- Variables
  let a := (log_d x)^2
  let b := 2 * log_d x
  let c := Real.log (log_d x) / Real.log d
  
  have h₃ : 0 < log_d x := sorry
  have h₄ : log_d x < 1 := sorry
  
  -- Proofs
  have ha : 0 < a := sorry
  have hba : a < b := sorry
  have hc : c < 0 := sorry
  have hca : c < a := sorry
  
  exact And.intro hca hba

 
end relationship_among_logarithm_values_l132_132698


namespace bacteria_time_calculation_l132_132830

noncomputable def bacteria_growth_time
  (B₀ : ℝ) -- Initial bacteria count
  (r : ℝ) -- Growth rate (tripling every period)
  (p : ℝ) -- Period of time in hours for growth rate to apply
  (B : ℝ) -- Final bacteria count
  : ℝ := 
  log (B / B₀) / log r * p

-- Given
variable (B₀ := 200 : ℝ) 
variable (r := 3 : ℝ) 
variable (p := 5 : ℝ)
variable (B := 145800 : ℝ)

theorem bacteria_time_calculation : 
  bacteria_growth_time B₀ r p B = 30 := 
sorry

end bacteria_time_calculation_l132_132830


namespace jill_marathon_time_l132_132011

-- Define constants based on the problem conditions
def marathon_distance : ℝ := 42
def jack_time : ℝ := 5.5
def speed_ratio : ℝ := 0.7636363636363637

-- Define what we need to prove
theorem jill_marathon_time :
  let v_jack := marathon_distance / jack_time in
  let v_jill := v_jack / speed_ratio in
  let t_jill := marathon_distance / v_jill in
  t_jill = 4.2 :=
by
  sorry

end jill_marathon_time_l132_132011


namespace chemical_transport_problem_l132_132365

theorem chemical_transport_problem :
  (∀ (w r : ℕ), r = w + 420 →
  (900 / r) = (600 / (10 * w)) →
  w = 30 ∧ r = 450) ∧ 
  (∀ (x : ℕ), x + 450 * 3 * 2 + 60 * x ≥ 3600 → x = 15) := by
  sorry

end chemical_transport_problem_l132_132365


namespace ratio_of_hexagon_to_dodecagon_l132_132437

theorem ratio_of_hexagon_to_dodecagon (s : ℝ) (P Q : ℝ) (hP : P = 3 * s^2 * Real.cot (π / 12)) (hQ : Q = (3 * Real.sqrt 3 / 2) * s^2) : 
  Q / P = (2 * Real.sqrt 3 - 3) / 2 := by
  sorry

end ratio_of_hexagon_to_dodecagon_l132_132437


namespace increasing_function_m_leq_3_l132_132311

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + log x + m / x

noncomputable def f_prime (x : ℝ) (m : ℝ) : ℝ := 2 * x + 1 / x - m / x^2

theorem increasing_function_m_leq_3 (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f_prime x m ≥ 0) → m ≤ 3 :=
by
  intro h
  have key_inequality := h 1 (by linarith)
  simp only [f_prime, one_div, sq] at key_inequality
  linarith

end increasing_function_m_leq_3_l132_132311


namespace find_smallest_integer_in_set_l132_132112

def is_odd (n : ℤ) : Prop := n % 2 = 1
def median (s : Set ℤ) (m : ℤ) : Prop := 
  (∃ l u : Finset ℤ, 
      (∀ x ∈ l, x < m) ∧ 
      (∀ y ∈ u, y > m) ∧ 
      l.card = u.card ∧ 
      (l ∪ u).card % 2 = 0 ∧ 
      s = l ∪ {m} ∪ u ∧ 
      l.card + 1 + u.card = s.card)
      
def greatest (s : Set ℤ) (g : ℤ) : Prop :=
  ∃ x ∈ s, ∀ y ∈ s, y ≤ g ∧ x = g
  
theorem find_smallest_integer_in_set 
  (s : Set ℤ)
  (h1 : median s 157)
  (h2 : greatest s 169) : 
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = 151 := 
by 
  sorry

end find_smallest_integer_in_set_l132_132112


namespace min_pizzas_to_recover_cost_l132_132387

theorem min_pizzas_to_recover_cost (van_cost earnings_per_pizza expenses_per_pizza : ℕ) :
  van_cost = 8000 → earnings_per_pizza = 15 → expenses_per_pizza = 4 → 
  ∃ p : ℕ, 11 * p ≥ van_cost ∧ p = 728 :=
by
  intros h1 h2 h3
  use 728
  rw [h1, h2, h3]
  exact ⟨by norm_num, rfl⟩

end min_pizzas_to_recover_cost_l132_132387


namespace find_daily_rate_second_company_l132_132440

def daily_rate_second_company (x : ℝ) : Prop :=
  let total_cost_1 := 21.95 + 0.19 * 150
  let total_cost_2 := x + 0.21 * 150
  total_cost_1 = total_cost_2

theorem find_daily_rate_second_company : daily_rate_second_company 18.95 :=
  by
  unfold daily_rate_second_company
  sorry

end find_daily_rate_second_company_l132_132440


namespace f_periodic_and_odd_find_f_2019_div_2_l132_132301

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Ioo 0 1 then 3^x - 1 else
  if x ∈ set.Ioo (-1) 0 then - (3^(-x) - 1) else
  sorry -- Additional cases to fully define the periodic and odd function

theorem f_periodic_and_odd (x : ℝ) :
  f x = f (x + 2) ∧ f (-x) = -f x :=
sorry

theorem find_f_2019_div_2 :
  f (2019 / 2) = -real.sqrt 3 + 1 :=
begin
  have h1 : 2019 / 2 = 1010 - 1 / 2, by norm_num,
  rw h1,
  have h2 : f (1010 - 1/2) = f (-1/2),
    from (f_periodic_and_odd (2019/2)).1,
  rw h2,
  have h3 : f (-1/2) = -f (1/2),
    from (f_periodic_and_odd (1/2)).2,
  rw h3,
  have h4 : (1/2 ∈ set.Ioo 0 1),
    from by norm_num,
  simp [f, h4],
  norm_num,
end

end f_periodic_and_odd_find_f_2019_div_2_l132_132301


namespace sweet_numbers_count_l132_132514

def iterative_rule (n : ℕ) : ℕ :=
  if n <= 30 then 2 * n else n - 12

def is_sweet_number (G : ℕ) : Prop :=
  ∀ (n : ℕ), let seq := (nat.iterate iterative_rule n)
  in n∈finset.range (G+1) \ 14

def count_sweet_numbers (a b : ℕ) : ℕ :=
  (finset.range (b + 1)).filter (λ n, is_sweet_number n).card

theorem sweet_numbers_count : count_sweet_numbers 1 50 = 20 := sorry

end sweet_numbers_count_l132_132514


namespace functional_equation_solution_l132_132610

open Nat

theorem functional_equation_solution (f : ℕ+ → ℕ+) 
  (H : ∀ (m n : ℕ+), f (f (f m) * f (f m) + 2 * f (f n) * f (f n)) = m * m + 2 * n * n) : 
  ∀ n : ℕ+, f n = n := 
sorry

end functional_equation_solution_l132_132610


namespace necklaces_sold_correct_l132_132792

-- Define the given constants and conditions
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20
def ensembles_sold : ℕ := 2
def total_revenue : ℕ := 565

-- Define the equation to calculate the total revenue
def total_revenue_calculation (N : ℕ) : ℕ :=
  (necklace_price * N) + (bracelet_price * bracelets_sold) + (earring_price * earrings_sold) + (ensemble_price * ensembles_sold)

-- Define the proof problem
theorem necklaces_sold_correct : 
  ∃ N : ℕ, total_revenue_calculation N = total_revenue ∧ N = 5 := by
  sorry

end necklaces_sold_correct_l132_132792


namespace fraction_eaten_on_third_day_l132_132783

theorem fraction_eaten_on_third_day
  (total_pieces : ℕ)
  (first_day_fraction : ℚ)
  (second_day_fraction : ℚ)
  (remaining_after_third_day : ℕ)
  (initial_pieces : total_pieces = 200)
  (first_day_eaten : first_day_fraction = 1/4)
  (second_day_eaten : second_day_fraction = 2/5)
  (remaining_bread_after_third_day : remaining_after_third_day = 45) :
  (1 : ℚ) / 2 = 1/2 := sorry

end fraction_eaten_on_third_day_l132_132783


namespace enclosed_area_correct_l132_132257

noncomputable def enclosedArea : ℝ := ∫ x in (1 / Real.exp 1)..Real.exp 1, 1 / x

theorem enclosed_area_correct : enclosedArea = 2 := by
  sorry

end enclosed_area_correct_l132_132257


namespace minimum_reciprocal_sum_l132_132047

theorem minimum_reciprocal_sum (a : Fin 15 → ℝ) (h₀ : ∀ i, 0 < a i) (h₁ : ∑ i, a i = 1) :
  (∑ i, (1 / (a i))) ≥ 225 := 
by
  sorry

end minimum_reciprocal_sum_l132_132047


namespace smallest_number_with_conditions_l132_132265

open Nat

/- Conditions: -/
def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def digit_sum (n : ℕ) : ℕ :=
  (list.sum (list.map fun x => (x.val) (n.digits 10)))

/- Main Theorem -/
theorem smallest_number_with_conditions :
  ∃ n : ℕ, is_divisible_by_5 n ∧ digit_sum n = 100 ∧ (∀ m : ℕ, is_divisible_by_5 m ∧ digit_sum m = 100 → n ≤ m) :=
sorry

end smallest_number_with_conditions_l132_132265


namespace circumcenters_on_circle_l132_132204

-- Define the required objects and conditions
noncomputable def triangle_with_incenter (A B C I : Point) : Prop :=
  incenter I A B C

-- State the theorem
theorem circumcenters_on_circle (A B C I : Point) (h : triangle_with_incenter A B C I) :
  ∃ (O : Point), ∀ (X : Point), 
    (circumcenter I A B = X ∨ circumcenter I B C = X ∨ circumcenter I C A = X) →
    lies_on_circle X O :=
sorry

end circumcenters_on_circle_l132_132204


namespace sin_of_right_angle_l132_132731

theorem sin_of_right_angle (D E F : Type) [euclidean_geometry D E F] (h_triangle: is_right_triangle DEF) (h_angle: angle D = 90) (DE_length: DE = 12) (EF_length: EF = 18) : sin D = 1 :=
sorry

end sin_of_right_angle_l132_132731


namespace max_cars_quotient_div_10_l132_132059

theorem max_cars_quotient_div_10 (n : ℕ) (h1 : ∀ v : ℕ, v ≥ 20 * n) (h2 : ∀ d : ℕ, d = 5* (n + 1)) :
  (4000 / 10 = 400) := by
  sorry

end max_cars_quotient_div_10_l132_132059


namespace integral_evaluation_l132_132249

noncomputable def integral_result : ℝ :=
  ∫ x in -real.pi / 4..real.pi / 4, (real.cos x + (1 / 4) * x^3 + 1)

theorem integral_evaluation :
  integral_result = real.sqrt 2 + (real.pi / 2) :=
sorry

end integral_evaluation_l132_132249


namespace transformed_inequality_solution_set_l132_132322

open Set

variable (k a b c : ℝ)
variable {x : ℝ}

theorem transformed_inequality_solution_set (h : (fun x => (k / (x + a)) + ((x + b) / (x + c)) < 0) '' (Ioo (-3 : ℝ) (-1) ∪ Ioo (1) (2)) = {y | y < 0}) :
  (λ x, (kx / (ax + 1)) + ((bx + 1) / (cx + 1)) < 0) '' (Ioo (-1 : ℝ) (-1/3) ∪ Ioo (1/2) (1)) = {z | z < 0} :=
sorry

end transformed_inequality_solution_set_l132_132322


namespace elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l132_132527

-- Define the initial conditions
def sea_level_drop : ℝ := 397
def submerged_depth_initial : ℝ := 5000
def height_diff_mauna_kea_everest : ℝ := 358

-- Define intermediate calculations based on conditions
def submerged_depth_adjusted : ℝ := submerged_depth_initial - sea_level_drop
def total_height_mauna_kea : ℝ := 2 * submerged_depth_adjusted
def elevation_above_sea_level_mauna_kea : ℝ := total_height_mauna_kea - submerged_depth_initial
def elevation_mount_everest : ℝ := total_height_mauna_kea - height_diff_mauna_kea_everest

-- Define the proof statements
theorem elevation_above_sea_level_mauna_kea_correct :
  elevation_above_sea_level_mauna_kea = 4206 := by
  sorry

theorem total_height_mauna_kea_correct :
  total_height_mauna_kea = 9206 := by
  sorry

theorem elevation_mount_everest_correct :
  elevation_mount_everest = 8848 := by
  sorry

end elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l132_132527


namespace monarchy_abolished_on_friday_l132_132451

-- Define days of the week, assuming Sunday = 0, Monday = 1, ..., Saturday = 6
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

-- Define a function to compute the day of the week after a given number of days
def add_days (start_day : DayOfWeek) (days : ℕ) : DayOfWeek :=
  match start_day with
  | Sunday    => DayOfWeek.ofNat ((0 + days) % 7)
  | Monday    => DayOfWeek.ofNat ((1 + days) % 7)
  | Tuesday   => DayOfWeek.ofNat ((2 + days) % 7)
  | Wednesday => DayOfWeek.ofNat ((3 + days) % 7)
  | Thursday  => DayOfWeek.ofNat ((4 + days) % 7)
  | Friday    => DayOfWeek.ofNat ((5 + days) % 7)
  | Saturday  => DayOfWeek.ofNat ((6 + days) % 7)

-- Problem statement: Prove that 1165 days after Tuesday is a Friday
theorem monarchy_abolished_on_friday :
  add_days Tuesday 1165 = Friday :=
by
  sorry

end monarchy_abolished_on_friday_l132_132451


namespace compound_interest_third_year_l132_132832

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n - P

noncomputable def principal_from_interest (CI : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  CI / (((1 + r)^n - 1) * (1 + r))

noncomputable def CI_2 := 1200
noncomputable def r := 0.06

noncomputable def P := principal_from_interest CI_2 r 2

noncomputable def CI_3 := compound_interest P r 3

theorem compound_interest_third_year :
  CI_3 = 1858.03 :=
by
  sorry

end compound_interest_third_year_l132_132832


namespace div_T_by_12_l132_132414

def is_12_pretty (n : ℕ) : Prop := 
  (n > 0) ∧ (n % 12 = 0) ∧ (n.divisors.length = 12)

def T : ℕ := (Finset.range 1000).filter is_12_pretty |>.sum

theorem div_T_by_12 : T / 12 = 109.33 := by
  sorry

end div_T_by_12_l132_132414


namespace problem_statement_l132_132771

variable {x y z : ℝ}

-- Be noncomputable for handling non-algorithmic proofs involving real numbers.
noncomputable theory

-- Define the conditions
def conditions (x y z : ℝ) : Prop := x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2 * y + 3 * z = 12

-- The theorem to be proved
theorem problem_statement (hx : conditions x y z) : x^2 + 2 * y^3 + 3 * z^2 > 24 := 
sorry

end problem_statement_l132_132771


namespace kelly_initially_had_l132_132758

def kelly_needs_to_pick : ℕ := 49
def kelly_will_have : ℕ := 105

theorem kelly_initially_had :
  kelly_will_have - kelly_needs_to_pick = 56 :=
by
  sorry

end kelly_initially_had_l132_132758


namespace gideon_code_count_l132_132275

theorem gideon_code_count :
  let primes := {d | d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7)}
  let multiples_of_4 := {d | d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (d % 4 = 0)}
  let odds := {d | d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (d % 2 = 1)}
  let powers_of_2 := {d | d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (d = 1 ∨ d = 2 ∨ d = 4 ∨ d = 8)}
  in primes.card * multiples_of_4.card * odds.card * powers_of_2.card = 240 := by 
  sorry

end gideon_code_count_l132_132275


namespace unique_zero_point_range_of_a_l132_132107

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x^2 + a - 1

theorem unique_zero_point_range_of_a (a : ℝ) : 
  (∃! x : ℝ, 1 < x ∧ x < Real.exp 1 ∧ f x a = 0) ↔ a ∈ Ioo (-Real.exp 2) 0 := 
by
  sorry

end unique_zero_point_range_of_a_l132_132107


namespace tg_beta_tg_gamma_l132_132907

theorem tg_beta_tg_gamma 
  (α β γ : ℝ)
  (h1 : (cot α * sin α) / (sin β * sin γ) = 2 / 3)
  (h2 : α + β + γ = 180) :
  tan β * tan γ = 3 :=
  sorry

end tg_beta_tg_gamma_l132_132907


namespace perpendicular_lines_l132_132235

def equation1 (x y : ℝ) : Prop := 5 * y - 3 * x = 15
def equation2 (x y : ℝ) : Prop := -3 * x - 5 * y = 15
def equation3 (x y : ℝ) : Prop := 5 * y + 3 * x = 15
def equation4 (x y : ℝ) : Prop := 3 * y + 5 * x = 15
def equation5 (x y : ℝ) : Prop := 2 * x - 10 * y = 12

def slope (a b c : ℝ) (x y : ℝ) : ℝ := -a / b

theorem perpendicular_lines :
  let m1 := slope 3 (-5) 15 in
  let m4 := slope (-5) 3 15 in
  m1 * m4 = -1 :=
by
  sorry

end perpendicular_lines_l132_132235


namespace ratio_of_populations_l132_132423

theorem ratio_of_populations (ne_pop : ℕ) (combined_pop : ℕ) (ny_pop : ℕ) (h1 : ne_pop = 2100000) 
                            (h2 : combined_pop = 3500000) (h3 : ny_pop = combined_pop - ne_pop) :
                            (ny_pop * 3 = ne_pop * 2) :=
by
  sorry

end ratio_of_populations_l132_132423


namespace find_smallest_integer_in_set_l132_132111

def is_odd (n : ℤ) : Prop := n % 2 = 1
def median (s : Set ℤ) (m : ℤ) : Prop := 
  (∃ l u : Finset ℤ, 
      (∀ x ∈ l, x < m) ∧ 
      (∀ y ∈ u, y > m) ∧ 
      l.card = u.card ∧ 
      (l ∪ u).card % 2 = 0 ∧ 
      s = l ∪ {m} ∪ u ∧ 
      l.card + 1 + u.card = s.card)
      
def greatest (s : Set ℤ) (g : ℤ) : Prop :=
  ∃ x ∈ s, ∀ y ∈ s, y ≤ g ∧ x = g
  
theorem find_smallest_integer_in_set 
  (s : Set ℤ)
  (h1 : median s 157)
  (h2 : greatest s 169) : 
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = 151 := 
by 
  sorry

end find_smallest_integer_in_set_l132_132111


namespace total_unique_paths_l132_132922

/-- A directional hexagonal lattice with stages of travel represented by color arrows:
    red, blue, pink, green, and orange. Each color stage must be traversed in sequence
    with no retraversing allowed. The goal is to find the total number of unique paths
    from point A to point B. -/
theorem total_unique_paths (A B : Point)
    (lattice : Lattice) (arrow_stages : List Stage)
    (travel_rules : TravelRules) :
  find_total_paths A B lattice arrow_stages travel_rules = 768 := 
sorry

end total_unique_paths_l132_132922


namespace line_intersects_curve_l132_132844

theorem line_intersects_curve (k : ℝ) :
  (∃ x y : ℝ, y + k * x + 2 = 0 ∧ x^2 + y^2 = 2 * x) ↔ k ≤ -3/4 := by
  sorry

end line_intersects_curve_l132_132844


namespace simplify_expression_l132_132816

theorem simplify_expression : (Real.cos (18 * Real.pi / 180) * Real.cos (42 * Real.pi / 180) - 
                              Real.cos (72 * Real.pi / 180) * Real.sin (42 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end simplify_expression_l132_132816


namespace latitude_approx_l132_132480

noncomputable def calculate_latitude (R h : ℝ) (θ : ℝ) : ℝ :=
  if h = 0 then θ else Real.arccos (1 / (2 * Real.pi))

theorem latitude_approx (R h θ : ℝ) (h_nonzero : h ≠ 0)
  (r1 : ℝ := R * Real.cos θ)
  (r2 : ℝ := (R + h) * Real.cos θ)
  (s : ℝ := 2 * Real.pi * h * Real.cos θ)
  (condition : s = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end latitude_approx_l132_132480


namespace domain_of_f_equals_0_to_4_l132_132616

def domain_of_function (x : ℝ) : Prop :=
  2 - sqrt (3 - sqrt (4 - x)) ≥ 0 ∧
  3 - sqrt (4 - x) ≥ 0 ∧
  4 - x ≥ 0

theorem domain_of_f_equals_0_to_4 :
  ∀ x, domain_of_function x ↔ (0 ≤ x ∧ x ≤ 4) := 
by 
  sorry

end domain_of_f_equals_0_to_4_l132_132616


namespace min_value_fraction_l132_132724

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ q, (q > 0) ∧ ∀ n, a (n + 1) = q * a n

theorem min_value_fraction (a : ℕ → ℝ) (m n : ℕ) (h_seq : geometric_sequence a)
  (h_pos : ∀ k, a k > 0)
  (h_sqrt : real.sqrt (a m * a n) = 4 * a 1)
  (h_a6 : a 6 = a 5 + 2 * a 4) :
  (1 / m : ℝ) + (4 / n) ≥ (3 / 2) := sorry

end min_value_fraction_l132_132724


namespace shaded_area_fraction_l132_132089

-- Definitions as per the problem conditions:
def square (s : ℝ) := s * s
def half (x : ℝ) := x / 2
def midpoints (s : ℝ) := s / 2
def area_of_shaded (s : ℝ) := (midpoints $ half (square s))  -- quarter of a triangle's area inside the square

-- Statement to prove:
theorem shaded_area_fraction (s : ℝ) (h : s ≠ 0) : area_of_shaded s / square s = 1 / 16 := by
  sorry

end shaded_area_fraction_l132_132089


namespace exercise_l132_132912

variable (f : ℝ → ℝ)

-- Conditions
axiom even_function : ∀ x, f(x) = f(-x)
axiom increasing_on_positive : ∀ x y, 0 < x → 0 < y → x < y → f(x) < f(y)

theorem exercise :
  f(3) < f(-4) ∧ f(-4) < f(-π) ∧ f(3) < f(-π) :=
by { sorry }

end exercise_l132_132912


namespace derivative_at_minus_one_l132_132525

-- Definition of the function f
def f (x : ℝ) : ℝ := (1/3) * x^3 + 2 * x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 + 2

-- Statement to prove that f'(-1) = 3
theorem derivative_at_minus_one : f' (-1) = 3 := 
by
  -- The proof is skipped
  sorry

end derivative_at_minus_one_l132_132525


namespace eval_expr_l132_132159
open Real

theorem eval_expr : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 8000 := by
  sorry

end eval_expr_l132_132159


namespace gcd_coefficients_expansion_eq_one_l132_132960

theorem gcd_coefficients_expansion_eq_one :
  ( ∃ (a : ℕ → ℕ) (k : ℕ), 
    (1 + x + x^2 + x^3 + x^4) ^ 496 = ∑ i in (finset.range 1985), (a i) * x^i ∧
    gcd (finset.image (λ k, if k % 5 = 3 then a k else 0) (finset.range 1985)) = 1 ) :=
sorry

end gcd_coefficients_expansion_eq_one_l132_132960


namespace find_least_m_l132_132596

def sequence (x : ℕ → ℝ) : ℕ → ℝ 
| 0     := 7
| (n+1) := (x n ^ 2 + 7 * x n + 12) / (x n + 8)

theorem find_least_m :
  let x := sequence (λ n, 7) in
  ∃ m : ℕ, m > 0 ∧ x m ≤ 6 + 1 / 2^22 ∧ ∀ n < m, x n > 6 + 1 / 2^22 := sorry

end find_least_m_l132_132596


namespace sum_of_squares_of_solutions_l132_132270

theorem sum_of_squares_of_solutions :
  (∃ (x : ℝ), (|x * x - (2 * x) + (1 / 1004)| = 1 / 502)) →
  let roots_sum_of_squares (a b c d : ℝ) := 
    (a + b - 2 * (a * b / (4 * a)) + c + d - 2 * (c * d / (4 * c)))
  in ∃ a b c d : ℝ, 
       |a * a - (2 * a) + (1 / 1004)| = 1 / 502 ∨
       |c * c - (2 * c) + (1 / 1004)| = -1 / 502 ∧ 
        roots_sum_of_squares a b c d  = 8050 / 1008 :=
sorry

end sum_of_squares_of_solutions_l132_132270


namespace value_of_x_div_y_l132_132348

noncomputable def x : ℝ := 1 / 0.16666666666666666

theorem value_of_x_div_y (y : ℝ) (h1 : y = 0.16666666666666666) (h2 : 0 < y) (h3 : x * y = 1) :
  x / y = 36 :=
by
  have hxy : x = 6 := by sorry -- Derived value of x from x * y = 1 and y = 0.16666666666666666
  have h_div : x / y = 36 := by {
    rw [hxy, h1],
    norm_num,
  }
  exact h_div

end value_of_x_div_y_l132_132348


namespace blood_expiration_date_l132_132156

theorem blood_expiration_date (donation_date : String) (leap_year : Bool) (expiry_seconds : Nat) : String :=
  if donation_date = "January 3" ∧ leap_year ∧ expiry_seconds = 8! then
    "January 3"
  else
    "Unknown"

-- Now we state our specific problem:
example : blood_expiration_date "January 3" true 40320 = "January 3" :=
  by sorry

end blood_expiration_date_l132_132156


namespace lenny_remaining_amount_l132_132759

theorem lenny_remaining_amount :
  let initial_amount := 270
  let console_price := 149
  let console_discount := 0.15 * console_price
  let final_console_price := console_price - console_discount
  let groceries_price := 60
  let groceries_discount := 0.10 * groceries_price
  let final_groceries_price := groceries_price - groceries_discount
  let lunch_cost := 30
  let magazine_cost := 3.99
  let total_expenses := final_console_price + final_groceries_price + lunch_cost + magazine_cost
  initial_amount - total_expenses = 55.36 :=
by
  sorry

end lenny_remaining_amount_l132_132759


namespace comics_stacking_order_l132_132246

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def marvel_count : ℕ := 8
def dc_count : ℕ := 6
def manga_count : ℕ := 7

def marvel_permutations : ℕ := factorial marvel_count
def dc_permutations : ℕ := factorial dc_count
def manga_permutations : ℕ := factorial manga_count

def group_permutations : ℕ := factorial 3

def total_permutations : ℕ :=
  marvel_permutations * dc_permutations * manga_permutations * group_permutations

theorem comics_stacking_order :
  total_permutations = 869397504000 := by
  sorry

end comics_stacking_order_l132_132246


namespace area_quadrilateral_twice_area_rectangle_l132_132063

noncomputable theory

variables {A B C D K M L N : Type} 
-- Assuming the points A, B, C, D, K, M, L, N are in Euclidean space.

-- Point midpoint assumptions
variable (is_midpoint_KM : ∀ (A B C D : ℝ), K = ((A + B) / 2) ∧ M = ((C + D) / 2))

-- Rectangle formation condition
variable (rectangle_KLMN : ∀ (A B C D K M L N : ℝ), is_rectangle K L M N)

-- Target theorem statement
theorem area_quadrilateral_twice_area_rectangle
    (ABCD_quadrilateral : convex_quadrilateral A B C D)
    (K_M_midpoints : is_midpoint_KM A B C D)
    (KLMN_is_rectangle : rectangle_KLMN A B C D K M L N) : 
    area ABCD = 2 * area (rectangle K L M N) :=
sorry

end area_quadrilateral_twice_area_rectangle_l132_132063


namespace no_partition_exists_l132_132980

-- Definitions representing conditions
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

noncomputable def partiton_set_exists (n : ℕ) : Prop :=
  n > 1 ∧
  ∃ A : Fin n → Set ℕ,
  (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
  (⋃ i, A i) = Set.univ ∧
  ∀ (s : Fin n → ℕ),
  (∀ i j, i ≠ j → s i ∈ A i) →
  ∃ i, ∑ (j : Fin n) in (Finset.filter (λ j, j ≠ i) Finset.univ), s j ∈ A i

-- Theorem stating the proof problem
theorem no_partition_exists : ∀ n : ℕ, ¬ partiton_set_exists n := 
by
  intro n
  sorry

end no_partition_exists_l132_132980


namespace inequalities_in_quadrants_l132_132127

theorem inequalities_in_quadrants (x y : ℝ) :
  (y > - (1 / 2) * x + 6) ∧ (y > 3 * x - 4) → (x > 0) ∧ (y > 0) :=
  sorry

end inequalities_in_quadrants_l132_132127


namespace min_value_of_norms_l132_132687

variables {x y : ℝ}

def a : ℝ × ℝ := (x, 1)
def b : ℝ × ℝ := (2, y)
def c : ℝ × ℝ := (1, 1)
def is_collinear (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = k • w
def norm (v : ℝ × ℝ) : ℝ := (v.1^2 + v.2^2).sqrt

theorem min_value_of_norms (h_collinear : is_collinear (a - b) c) (h_sum : x + y = 3) : 
  norm a + 2 * norm b = 3 * (5:ℝ).sqrt := 
  sorry

end min_value_of_norms_l132_132687


namespace original_team_players_l132_132491

theorem original_team_players (n : ℕ) (W : ℝ)
    (h1 : W = n * 76)
    (h2 : (W + 110 + 60) / (n + 2) = 78) : n = 7 :=
  sorry

end original_team_players_l132_132491


namespace rectangle_area_diagonal_l132_132561

def rectangle_area (y : ℝ) : ℝ := 
  let w := (y^2 / 10)^(1 / 2)
  let l := 3 * w
  3 * w^2

theorem rectangle_area_diagonal
  (y : ℝ)
  (h_pos : y > 0)
  (h_diagonal : ∃ w l, l = 3 * w ∧ (l^2 + w^2 = y^2)) :
  rectangle_area y = (3 / 10) * y^2 :=
sorry

end rectangle_area_diagonal_l132_132561


namespace ratio_garage_to_others_l132_132442

-- Definitions of given conditions
def bedroom_bulbs := 2
def bathroom_bulbs := 1
def kitchen_bulbs := 1
def basement_bulbs := 4
def packs := 6
def bulbs_per_pack := 2

-- Define the total bulbs required in the bedroom, bathroom, kitchen, and basement.
def total_other_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs

-- Define the total number of bulbs based on packs and bulbs per pack.
def total_bulbs := packs * bulbs_per_pack

-- Define the number of bulbs in the garage.
def garage_bulbs := total_bulbs - total_other_bulbs

-- Prove the ratio of garage bulbs to total other bulbs is 1:2
theorem ratio_garage_to_others : (garage_bulbs : ℚ) / total_other_bulbs = 1 / 2 := by
  have h1 : total_other_bulbs = 8 := by
    simp [bedroom_bulbs, bathroom_bulbs, kitchen_bulbs, basement_bulbs]
  have h2 : total_bulbs = 12 := by
    simp [packs, bulbs_per_pack]
  have h3 : garage_bulbs = 4 := by
    simp [total_bulbs, total_other_bulbs, h1, h2]
  rw [h1, h3]
  norm_num
  sorry

end ratio_garage_to_others_l132_132442


namespace find_value_l132_132669

-- Definitions of the curve and the line
def curve (a b : ℝ) (P : ℝ × ℝ) : Prop := (P.1*P.1) / a - (P.2*P.2) / b = 1
def line (P : ℝ × ℝ) : Prop := P.1 + P.2 - 1 = 0

-- Definition of the dot product condition
def dot_product_zero (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

-- Theorem statement
theorem find_value (a b : ℝ) (P Q : ℝ × ℝ)
  (hc1 : curve a b P)
  (hc2 : curve a b Q)
  (hl1 : line P)
  (hl2 : line Q)
  (h_dot : dot_product_zero P Q) :
  1 / a - 1 / b = 2 :=
sorry

end find_value_l132_132669


namespace angelina_speed_grocery_to_gym_l132_132517

variable (d1 d2 t1 t2 s1 s2 : ℝ)

-- Given distances
def d1 : ℝ := 100
def d2 : ℝ := 180

-- Given times
def t1 : ℝ := d1 / s1
def t2 : ℝ := t1 - 40

-- Given speed relationships
def s2 : ℝ := 2 * s1

-- Problem statement to prove
theorem angelina_speed_grocery_to_gym :
  s2 = 1 / 2 :=
by
  sorry

end angelina_speed_grocery_to_gym_l132_132517


namespace midpoint_of_AB_l132_132725

-- Define coordinates in polar form for points A and B
def A := (10, Real.pi / 6)
def B := (10, 5 * Real.pi / 6)

-- Define a function to calculate the midpoint in polar coordinates
def midpoint_polar (r θ₁ θ₂ : ℝ) : ℝ × ℝ :=
  let θ_M := (θ₁ + θ₂) / 2
  let r_M := r * Real.cos ((θ₂ - θ₁) / 2)
  (r_M, θ_M)

-- Prove the specific midpoint for given points A and B
theorem midpoint_of_AB : midpoint_polar 10 (Real.pi / 6) (5 * Real.pi / 6) = (5, Real.pi / 2) :=
by
  sorry

end midpoint_of_AB_l132_132725


namespace distances_equal_or_l132_132476

-- Define the geometric setup
variables {A B C D E F : Point}
variable {dist : Point → Point → ℝ}

-- Conditions from the problem
def tangent_condition (C A B D : Point) : Prop :=
∃ circle : Circle, tangent circle C ∧ circumcircle circle A B C ∧ D ∈ tangent_intersection line_through A B circle

def points_condition (B C E F : Point) : Prop :=
dist B E = dist B F ∧ dist B E = abs ((dist C D)^2 - (dist B D)^2) / (dist B C)

def line_through (p1 p2 p : Point) : Prop :=  -- Placeholder for points on a line definition
true  -- needs proper line definition

-- Question to prove
theorem distances_equal_or (C A B D E F : Point) (dist : Point → Point → ℝ) 
  (hTangent : tangent_condition C A B D) (hPoints : points_condition B C E F)
  : dist E D = dist C D ∨ dist F D = dist C D := 
begin
  sorry
end

end distances_equal_or_l132_132476


namespace concyclic_quadrilateral_same_radius_exercise_special_case_l132_132165

-- Define the basic structures and conditions
variables (γ γ' : Circle) (A B T T' B* : Point)
noncomputable def tangent_common (γ γ' : Circle) : Line := sorry
noncomputable def reflection (X : Point) (l : Line) : Point := sorry

-- Hypotheses based on conditions given in the problem:
axiom circles_intersect_at_two_points : ∀ (γ γ' : Circle), ∃ (A B : Point), A ≠ B ∧ A ∈ γ ∧ A ∈ γ' ∧ B ∈ γ ∧ B ∈ γ'
axiom common_tangent_touch_points : tangent_common γ γ' = Line.through T T'
axiom tangent_contains_B : B ∈ Line.through T T'

-- Defining the reflection
axiom reflection_B : B* = reflection B (tangent_common γ γ')

-- The first proof problem statement:
theorem concyclic_quadrilateral : is_concyclic_point A T T' B* := sorry

-- The second proof problem statement:
theorem same_radius : radius_of_circumcircle A T T' = radius_of_circumcircle B T T' := sorry

-- The third proof problem statement:
theorem exercise_special_case (h : A = B) : describes_special_case_A_eq_B := sorry

end concyclic_quadrilateral_same_radius_exercise_special_case_l132_132165


namespace solution_inequality_regions_l132_132603

noncomputable def inequality_satisfied_by_xy (x y : ℝ) :=
  let f (t : ℝ) := y^2 - (arccos (cos t))^2 in
  f x * f (x + π / 6) * f (x - π / 6) < 0

theorem solution_inequality_regions :
  ∀ (x y : ℝ), inequality_satisfied_by_xy x y ↔ 
      (y = arccos (cos x) ∨ y = - arccos (cos x) ∨ 
       y = arccos (cos (x + π / 6)) ∨ y = - arccos (cos (x + π / 6)) ∨
       y = arccos (cos (x - π / 6)) ∨ y = - arccos (cos (x - π / 6))) →
      (complex description involving intervals : regions).
sorry

end solution_inequality_regions_l132_132603


namespace regular_polygon_sides_l132_132695

-- Conditions
def exterior_angle (n : ℕ) : ℝ := 360 / n
def condition (n : ℕ) : Prop := exterior_angle n = n - 9

-- Theorem to prove
theorem regular_polygon_sides (n : ℕ) (h : condition n) : n = 24 :=
by 
  sorry

end regular_polygon_sides_l132_132695


namespace benny_missed_games_l132_132224

def total_games : ℕ := 39
def attended_games : ℕ := 14
def missed_games : ℕ := total_games - attended_games

theorem benny_missed_games : missed_games = 25 := by
  sorry

end benny_missed_games_l132_132224


namespace form_rectangle_from_squares_l132_132579

-- Define the lengths of the squares
def square_sides := [2, 8, 14, 16, 18, 20, 28, 30, 36]

-- Given the sum of areas of these squares
def total_area : Nat := 
  square_sides.foldl (λ acc s, acc + s * s) 0

-- Prove that it is possible to form a rectangle with total area 4224
theorem form_rectangle_from_squares
  (sides : List Nat)
  (h : sides = square_sides)
  (area_eq : total_area = 4224):
  ∃ a b : Nat, a * b = 4224 ∧ 
    ∀ side ∈ sides, side <= max a b :=
sorry

end form_rectangle_from_squares_l132_132579


namespace find_second_term_l132_132097

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end find_second_term_l132_132097


namespace map_scale_l132_132586

theorem map_scale (map_distance : ℝ) (time : ℝ) (speed : ℝ) (actual_distance : ℝ) (scale : ℝ) 
  (h1 : map_distance = 5) 
  (h2 : time = 1.5) 
  (h3 : speed = 60) 
  (h4 : actual_distance = speed * time) 
  (h5 : scale = map_distance / actual_distance) : 
  scale = 1 / 18 :=
by 
  sorry

end map_scale_l132_132586


namespace reciprocal_of_minus_one_over_2023_l132_132865

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l132_132865


namespace problem_equivalent_proof_l132_132969

noncomputable def sum_of_real_solutions : ℝ :=
  ∑ x in ({x : ℝ | (x^2 - 4 * x + 3)^(x^2 - 5 * x + 4) = 1}.to_finset), x

theorem problem_equivalent_proof :
  sum_of_real_solutions = 10 := sorry

end problem_equivalent_proof_l132_132969


namespace john_twice_james_l132_132746

def john_age : ℕ := 39
def years_ago : ℕ := 3
def years_future : ℕ := 6
def age_difference : ℕ := 4

theorem john_twice_james {J : ℕ} (h : 39 - years_ago = 2 * (J + years_future)) : 
  (J + age_difference = 16) :=
by
  sorry  -- Proof steps here

end john_twice_james_l132_132746


namespace pat_peano_maximum_pages_l132_132425

noncomputable def count_fives_in_range : ℕ → ℕ := sorry

theorem pat_peano_maximum_pages (n : ℕ) : 
  (count_fives_in_range 54) = 15 → n ≤ 54 :=
sorry

end pat_peano_maximum_pages_l132_132425


namespace find_second_term_l132_132100

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end find_second_term_l132_132100


namespace initial_average_income_l132_132543

theorem initial_average_income 
  (initial_members : ℕ) 
  (final_members : ℕ) 
  (final_average : ℝ) 
  (deceased_income : ℝ) 
  (initial_average : ℝ) 
  (h_initial_members : initial_members = 4) 
  (h_final_members : final_members = 3) 
  (h_final_average : final_average = 590) 
  (h_deceased_income : deceased_income = 1170) 
  (h_total_income : initial_members * initial_average - deceased_income = final_members * final_average) : 
  initial_average = 735 :=
by 
  rw [h_initial_members, h_final_members, h_final_average, h_deceased_income] at h_total_income
  calc
    initial_average = (final_members * final_average + deceased_income) / initial_members : by
      field_simp [h_total_income]
    ... = 735 : by
      norm_num


end initial_average_income_l132_132543


namespace weekly_milk_production_l132_132989

theorem weekly_milk_production 
  (bess_milk_per_day : ℕ) 
  (brownie_milk_per_day : ℕ) 
  (daisy_milk_per_day : ℕ) 
  (total_milk_per_day : ℕ) 
  (total_milk_per_week : ℕ) 
  (h1 : bess_milk_per_day = 2) 
  (h2 : brownie_milk_per_day = 3 * bess_milk_per_day) 
  (h3 : daisy_milk_per_day = bess_milk_per_day + 1) 
  (h4 : total_milk_per_day = bess_milk_per_day + brownie_milk_per_day + daisy_milk_per_day)
  (h5 : total_milk_per_week = total_milk_per_day * 7) : 
  total_milk_per_week = 77 := 
by sorry

end weekly_milk_production_l132_132989


namespace equilateral_triangle_cosine_value_l132_132727

noncomputable def equilateral_triangle_cosine (A B C F G : Point)
  (h_equilateral : Equilateral A B C)
  (h_trisect : Trisect F G B C) : Real :=
  cos_angle A F G

theorem equilateral_triangle_cosine_value
  {A B C F G : Point}
  (h_equilateral : Equilateral A B C)
  (h_trisect : Trisect F G B C) :
  equilateral_triangle_cosine A B C F G h_equilateral h_trisect =
  1.5 * sqrt 3 / sqrt 13 :=
by
  sorry

end equilateral_triangle_cosine_value_l132_132727


namespace point_B_in_first_quadrant_l132_132896

def is_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_B_in_first_quadrant : is_first_quadrant (1, 2) :=
by
  sorry

end point_B_in_first_quadrant_l132_132896


namespace derivative_log_composite_l132_132456

-- Define the function.
def f (x : ℝ) := Real.log (2 * x^2 + 1)

-- State the theorem.
theorem derivative_log_composite (x : ℝ) : 
  deriv f x = (4 * x) / (2 * x^2 + 1) :=
sorry

end derivative_log_composite_l132_132456


namespace find_y_l132_132343

theorem find_y (x y : ℤ) (h1 : x = -4) (h2 : x^2 + 3 * x + 7 = y - 5) : y = 16 := 
by
  sorry

end find_y_l132_132343


namespace min_value_n_constant_term_l132_132305

theorem min_value_n_constant_term (n r : ℕ) (h : 5 * n = 6 * r) : 
  (∃ (n : ℕ), (∃ r : ℕ, (5 * n = 6 * r ∧ (5 * n ≠ 0))) ∧ n = 6) :=
by 
  choose r' hr' using h,
  existsi 6,
  use 5,
  split,
  { exact h },
  { sorry }

end min_value_n_constant_term_l132_132305


namespace solve_y4_l132_132560

theorem solve_y4 (y : ℝ) (h : 0 < y) (h1 : sqrt(1 - y^2) + sqrt(1 + y^2) = 2) : y^4 = 0 :=
by
  sorry

end solve_y4_l132_132560


namespace johns_total_cost_l132_132017

-- Definitions for the prices and quantities
def price_shirt : ℝ := 15.75
def price_tie : ℝ := 9.40
def quantity_shirts : ℕ := 3
def quantity_ties : ℕ := 2

-- Definition for the total cost calculation
def total_cost (price_shirt price_tie : ℝ) (quantity_shirts quantity_ties : ℕ) : ℝ :=
  (price_shirt * quantity_shirts) + (price_tie * quantity_ties)

-- Theorem stating the total cost calculation for John's purchase
theorem johns_total_cost : total_cost price_shirt price_tie quantity_shirts quantity_ties = 66.05 :=
by
  sorry

end johns_total_cost_l132_132017


namespace evaluate_72_squared_minus_48_squared_l132_132985

theorem evaluate_72_squared_minus_48_squared :
  (72:ℤ)^2 - (48:ℤ)^2 = 2880 :=
by
  sorry

end evaluate_72_squared_minus_48_squared_l132_132985


namespace number_of_cars_l132_132566

theorem number_of_cars (n s t C : ℕ) (h1 : n = 9) (h2 : s = 4) (h3 : t = 3) (h4 : n * s = t * C) : C = 12 :=
by
  sorry

end number_of_cars_l132_132566


namespace max_mark_paper_I_l132_132923

theorem max_mark_paper_I (M : ℝ) (h1 : 0.52 * M = 80) : M = 154 :=
by
  have hM := h1.symm
  exact eq_of_sub_eq_zero (sub_eq_zero_of_eq (hM.trans (div_eq_iff_mul_eq').mp rfl))

end max_mark_paper_I_l132_132923


namespace smallest_four_digit_number_representation_l132_132004

theorem smallest_four_digit_number_representation :
  ∃ (草 绿 花 红 春 光 明 媚 : ℕ),
    草 ≠ 绿 ∧ 草 ≠ 花 ∧ 草 ≠ 红 ∧ 草 ≠ 春 ∧ 草 ≠ 光 ∧ 草 ≠ 明 ∧ 草 ≠ 媚 ∧
    绿 ≠ 花 ∧ 绿 ≠ 红 ∧ 绿 ≠ 春 ∧ 绿 ≠ 光 ∧ 绿 ≠ 明 ∧ 绿 ≠ 媚 ∧
    花 ≠ 红 ∧ 花 ≠ 春 ∧ 花 ≠ 光 ∧ 花 ≠ 明 ∧ 花 ≠ 媚 ∧
    红 ≠ 春 ∧ 红 ≠ 光 ∧ 红 ≠ 明 ∧ 红 ≠ 媚 ∧
    春 ≠ 光 ∧ 春 ≠ 明 ∧ 春 ≠ 媚 ∧
    光 ≠ 明 ∧ 光 ≠ 媚 ∧
    明 ≠ 媚 ∧
    1 ≤ 草 ∧ 草 ≤ 9 ∧ 
    1 ≤ 绿 ∧ 绿 ≤ 9 ∧ 
    1 ≤ 花 ∧ 花 ≤ 9 ∧ 
    1 ≤ 红 ∧ 红 ≤ 9 ∧ 
    1 ≤ 春 ∧ 春 ≤ 9 ∧ 
    1 ≤ 光 ∧ 光 ≤ 9 ∧ 
    1 ≤ 明 ∧ 明 ≤ 9 ∧ 
    1 ≤ 媚 ∧ 媚 ≤ 9 ∧ 
    草 * 绿 * 花 * 红 = 4396 :=
begin
  sorry
end

end smallest_four_digit_number_representation_l132_132004


namespace probability_divisors_of_12_pow_7_l132_132396

theorem probability_divisors_of_12_pow_7 (T : Set ℕ)
  (hT : T = {d | d ∣ 12^7 ∧ d > 0})
  (a_1 a_2 a_3 : ℕ)
  (ha_1T : a_1 ∈ T) (ha_2T : a_2 ∈ T) (ha_3T : a_3 ∈ T)
  (h_a1_divides_a2 : a_1 ∣ a_2) (h_a3_divides_a1 : a_3 ∣ a_1)
  : ∃ m n : ℕ, nat.coprime m n ∧ (m, n) = (7, 405) :=
begin
  sorry
end

end probability_divisors_of_12_pow_7_l132_132396


namespace tangent_parallel_tangent_perpendicular_l132_132798

def curve (x : ℝ) : ℝ :=  4 * x^2 - 6 * x + 3

def deriv (x : ℝ) : ℝ := 8 * x - 6

theorem tangent_parallel (x y : ℝ) 
    (h_curve : y = curve x) (h_slope_parallel : deriv x = 2) : 
    (x, y) = (1, 1) :=
by {
    sorry
}

theorem tangent_perpendicular (x y : ℝ) 
    (h_curve : y = curve x) (h_slope_perpendicular : deriv x = -4) : 
    (x, y) = (1/4, 7/4) :=
by {
    sorry
}

end tangent_parallel_tangent_perpendicular_l132_132798


namespace problem1_problem2_l132_132911

-- Definition for first problem condition
def cond1 (a : ℝ) : Prop :=
∀ x : ℝ, (3 * x - abs (-2 * x + 1) ≥ a) ↔ (x ∈ set.Ici 2)

-- Proof goal for first problem
theorem problem1 : ∃ a : ℝ, cond1 a ∧ a = 3 :=
begin
  sorry
end

-- Definition for second problem condition
def cond2 (a : ℝ) : Prop :=
∀ x : ℝ, (x ∈ set.Icc 1 2) → (x - abs (x - a) ≤ 1)

-- Proof goal for second problem
theorem problem2 : ∃ a : ℝ, cond2 a ∧ (a ∈ set.Iic 1 ∨ a ∈ set.Ici 3) :=
begin
  sorry
end

end problem1_problem2_l132_132911


namespace log_inequality_l132_132400

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 2 / Real.log 5
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem log_inequality : c > a ∧ a > b := 
by
  sorry

end log_inequality_l132_132400


namespace find_function_l132_132046

theorem find_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, x + y + z = 0 → f (x^3) + f (y)^3 + f (z)^3 = 3 * x * y * z) → 
  f = id :=
by sorry

end find_function_l132_132046


namespace pens_taken_second_month_l132_132742

theorem pens_taken_second_month :
  let red_pens := 62
  let black_pens := 43
  let students := 3
  let pens_after_first_month := (students * (red_pens + black_pens)) - 37
  let pens_after_second_set := students * 79
  pens_after_first_month - pens_after_second_set = 41 := 
by
  let red_pens := 62
  let black_pens := 43
  let students := 3
  let pens_after_first_month := (students * (red_pens + black_pens)) - 37
  let pens_after_second_set := students * 79
  show pens_after_first_month - pens_after_second_set = 41,
  sorry

end pens_taken_second_month_l132_132742


namespace star_polygon_angle_l132_132197

open Int

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem star_polygon_angle (n : ℕ) (h1 : n > 5) (h2 : is_odd n) : 
  (360 * (n - 3) / n : ℝ) = 360 * (n - 3) / n :=
begin
  sorry
end

end star_polygon_angle_l132_132197


namespace divisible_by_a_minus_one_squared_l132_132076

theorem divisible_by_a_minus_one_squared (a n : ℕ) (h : n > 0) :
  (a^(n+1) - n * (a - 1) - a) % (a - 1)^2 = 0 :=
by
  sorry

end divisible_by_a_minus_one_squared_l132_132076


namespace trajectory_Q_constant_yP_trajectory_Q_constant_xP_l132_132462

variables (x_A y_A x_B y_B x_P y_P real : ℝ)

def y_line_eq1 := y_A * y_P = x_P + x_A
def y_line_eq2 := y_B * y_P = x_P + x_B

theorem trajectory_Q_constant_yP (y_A y_P y_B x_A x_B : ℝ) :
  (∀ x_P : ℝ, y_line_eq1 y_A y_P x_P x_A ∧ y_line_eq2 y_B y_P x_P x_B) →
  -- placeholder for the fact that trajectory of Q is a straight line
  sorry :=
begin
  -- Proof omitted
  sorry
end

theorem trajectory_Q_constant_xP (x_P y_A y_P y_B x_A x_B : ℝ) :
  (∀ y_P : ℝ, y_line_eq1 y_A y_P x_P x_A ∧ y_line_eq2 y_B y_P x_P x_B) →
  -- placeholder for the fact that trajectory of Q is a parabola
  sorry :=
begin
  -- Proof omitted
  sorry
end

end trajectory_Q_constant_yP_trajectory_Q_constant_xP_l132_132462


namespace train_length_l132_132939

theorem train_length (v : ℝ) (t : ℝ) (l_b : ℝ) (v_r : v = 52) (t_r : t = 34.61538461538461) (l_b_r : l_b = 140) : 
  ∃ l_t : ℝ, l_t = 360 :=
by
  have speed_ms := v * (1000 / 3600)
  have total_distance := speed_ms * t
  have length_train := total_distance - l_b
  use length_train
  sorry

end train_length_l132_132939


namespace number_of_players_l132_132820

theorem number_of_players (n : ℕ) (G : ℕ) (h : G = 2 * n * (n - 1)) : n = 19 :=
by {
  sorry
}

end number_of_players_l132_132820


namespace flight_time_is_10_hours_l132_132607

def time_watching_TV_episodes : ℕ := 3 * 25
def time_sleeping : ℕ := 4 * 60 + 30
def time_watching_movies : ℕ := 2 * (1 * 60 + 45)
def remaining_flight_time : ℕ := 45

def total_flight_time : ℕ := (time_watching_TV_episodes + time_sleeping + time_watching_movies + remaining_flight_time) / 60

theorem flight_time_is_10_hours : total_flight_time = 10 := by
  sorry

end flight_time_is_10_hours_l132_132607


namespace mean_cost_of_diesel_l132_132158

-- Define the diesel rates and the number of years.
def dieselRates : List ℝ := [1.2, 1.3, 1.8, 2.1]
def years : ℕ := 4

-- Define the mean calculation and the proof requirement.
theorem mean_cost_of_diesel (h₁ : dieselRates = [1.2, 1.3, 1.8, 2.1]) 
                               (h₂ : years = 4) : 
  (dieselRates.sum / years) = 1.6 :=
by
  sorry

end mean_cost_of_diesel_l132_132158


namespace combination_29_5_l132_132660

theorem combination_29_5
  (h1 : nat.choose 27 3 = 2925)
  (h2 : nat.choose 27 4 = 17550)
  (h3 : nat.choose 27 5 = 80730) :
  nat.choose 29 5 = 118755 :=
by
  have h4 : nat.choose 28 4 = nat.choose 27 3 + nat.choose 27 4, from nat.choose_succ_succ 27 3,
  rw [h1, h2] at h4,
  have h5 : nat.choose 28 4 = 20475, from h4,
  have h6 : nat.choose 28 5 = nat.choose 27 4 + nat.choose 27 5, from nat.choose_succ_succ 27 4,
  rw [h2, h3] at h6,
  have h7 : nat.choose 28 5 = 98280, from h6,
  have h8 : nat.choose 29 5 = nat.choose 28 4 + nat.choose 28 5, from nat.choose_succ_succ 28 4,
  rw [h5, h7] at h8,
  have h9 : nat.choose 29 5 = 118755, from h8,
  exact h9,
  sorry  -- The actual proof in Lean, which shows the steps and calculations, is skipped.

end combination_29_5_l132_132660


namespace arithmetic_sequence_product_l132_132039

theorem arithmetic_sequence_product (b : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ n, b (n + 1) > b n)
  (h2 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l132_132039


namespace part_a_b_advantage_part_b_min_people_l132_132910

def prob_not_sunday (n : ℕ) : ℝ := (6 / 7) ^ n
def prob_at_least_one_sunday (n : ℕ) : ℝ := 1 - prob_not_sunday n

def expected_winnings_A (n : ℕ) (b_bet : ℝ) : ℝ :=
  b_bet * prob_at_least_one_sunday n - 10 * prob_not_sunday n

theorem part_a_b_advantage :
  expected_winnings_A 7 5 < 0 := by
  sorry

theorem part_b_min_people (n : ℕ) :
  expected_winnings_A n 1 > 0 → n ≥ 16 := by
  sorry

end part_a_b_advantage_part_b_min_people_l132_132910


namespace percentage_increase_is_40_percent_l132_132544

variable (C : ℝ)

-- Conditions from the problem:
-- 1. The customer price is 1.82 times the manufacturing cost.
def customer_price (C : ℝ) : ℝ := 1.82 * C
-- 2. The retailer price is customer price divided by 1.30.
def retailer_price (C : ℝ) : ℝ := (customer_price C) / 1.30
-- 3. The percentage increase from the manufacturing cost to the retailer's price.
def percentage_increase (C : ℝ) : ℝ := ((retailer_price C - C) / C) * 100

-- Theorem to prove the percentage increase is 40%
theorem percentage_increase_is_40_percent (C : ℝ) : 
  percentage_increase C = 40 := 
sorry

end percentage_increase_is_40_percent_l132_132544


namespace find_dividend_and_divisor_l132_132882

-- Definitions of the conditions
def dividend (d: ℕ): ℕ := 6 * d
def conditions (d divisor dividend quotient : ℕ) : Prop :=
  quotient = dividend / divisor ∧ dividend = 6 * divisor ∧ dividend + divisor + quotient = 216

-- Problem Statement: Prove the dividend and divisor given the conditions
theorem find_dividend_and_divisor:
  ∃ d div, conditions div div (dividend div) 6 ∧ dividend div = 180 ∧ div = 30 :=
by
  sorry

end find_dividend_and_divisor_l132_132882


namespace coefficient_x3_in_expansion_l132_132374

-- Define variables
variables {x : ℕ} {n : ℕ}

-- The statement represents the mathematical problem with necessary conditions and conclusion
theorem coefficient_x3_in_expansion (h : (4 : ℕ)^n / (2 : ℕ)^n = 64) :
  let coefficient := nat.choose 6 2 * 9 in coefficient = 135 :=
by
  sorry

end coefficient_x3_in_expansion_l132_132374


namespace gcd_765432_654321_l132_132999

-- Define the two integers 765432 and 654321
def a : ℕ := 765432
def b : ℕ := 654321

-- State the main theorem to prove the gcd
theorem gcd_765432_654321 : Nat.gcd a b = 3 := 
by 
  sorry

end gcd_765432_654321_l132_132999


namespace cosine_theta_l132_132688

variables (a b : EuclideanSpace ℝ (Fin 2))

noncomputable
def θ := real.angle (a.1, a.2) (b.1, b.2)

-- Definitions according to conditions
def vector_a : EuclideanSpace ℝ (Fin 2) := ![1, 3]
def vector_b : EuclideanSpace ℝ (Fin 2) := -(vector_a) + ![-2, 6]

theorem cosine_theta :
  (a = vector_a) →
  (a + b = ![-2, 6]) →
  real.cos (θ a b) = (real.sqrt 5) / 5 :=
begin
  sorry
end

end cosine_theta_l132_132688


namespace sqrt_x_add_one_meaningful_l132_132713

theorem sqrt_x_add_one_meaningful (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 1)) ↔ x ≥ -1 := 
by
  sorry

end sqrt_x_add_one_meaningful_l132_132713


namespace EF_parallel_plane_ABC_AC_perpendicular_AB_l132_132738

variables (A B C A1 B1 C1 E F : Type) [right_prism ABC A1 B1 C1]
  (AB1 BC1 A1B AB AC : Type)

-- Condition: AB1 ⊥ BC1
axiom AB1_perp_BC1 : orthogonal AB1 BC1

-- Condition: AB = AA1
axiom AB_eq_AA1 : equal_length AB (AA1 : Type)

-- Definitions: E is midpoint of AB1, and F is midpoint of BC1
axiom E_midpoint_AB1 : midpoint E AB1
axiom F_midpoint_BC1 : midpoint F BC1

-- Define Plane
variables (plane_ABC : Type)
axiom is_plane_ABC : plane plane_ABC A B C

theorem EF_parallel_plane_ABC : parallel EF plane_ABC :=
by
  sorry

theorem AC_perpendicular_AB : orthogonal AC AB :=
by
  sorry

end EF_parallel_plane_ABC_AC_perpendicular_AB_l132_132738


namespace functions_equal_l132_132510

theorem functions_equal :
  ∀ x : ℝ, (λ x : ℝ, x - 1) x = (λ t : ℝ, t - 1) x := by
  sorry

end functions_equal_l132_132510


namespace polynomial_remainder_l132_132052

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ),
  (q.eval 2 = 8) →
  (q.eval (-3) = -10) →
  ∃ c d : ℚ, (q = (Polynomial.C (c : ℚ) * (Polynomial.X - Polynomial.C 2) * (Polynomial.X + Polynomial.C 3)) + (Polynomial.C 3.6 * Polynomial.X + Polynomial.C 0.8)) :=
by intros q h1 h2; sorry

end polynomial_remainder_l132_132052


namespace sum_f_1_to_2015_l132_132273

-- Define the function f
section
variable {f : ℝ → ℝ}
-- Define the odd function property
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
-- Additional conditions specific for this problem
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) + f (3 + x) = 0) ∧ f (-1) = 1

-- The theorem to prove the given question
theorem sum_f_1_to_2015 (f : ℝ → ℝ) (h₁ : is_odd f) (h₂ : satisfies_conditions f) :
  f 1 + f 2 + ∑ i in finset.range 2013, f (i + 3) = 0 := sorry
end

end sum_f_1_to_2015_l132_132273


namespace trigonometric_identity_l132_132445

theorem trigonometric_identity 
  (A : ℝ) 
  (h_cot : Real.cot A = Real.cos A / Real.sin A) 
  (h_csc : Real.csc A = 1 / Real.sin A) 
  (h_tan : Real.tan A = Real.sin A / Real.cos A) 
  (h_sec : Real.sec A = 1 / Real.cos A) :
  (1 + Real.cot A + Real.csc A) * (1 + Real.tan A - Real.sec A) = 3 - 1 / (Real.sin A * Real.cos A) :=
by
  sorry

end trigonometric_identity_l132_132445


namespace part1_part2_l132_132719

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the angles

-- Given conditions
axiom cos_C : cos C = 1 / 3
axiom sin_A_eq_sqrt2_cos_B : sin A = sqrt(2) * cos B
axiom c_eq_sqrt5 : c = sqrt(5)

-- Proof goals
theorem part1 : tan B = sqrt(2) :=
sorry

theorem part2 : ∃S, S = 1 / 2 * b * c * sin A ∧ S = 5 * sqrt(2) / 4 :=
sorry

end part1_part2_l132_132719


namespace lasagna_pieces_l132_132419

theorem lasagna_pieces (m a k r l : ℕ → ℝ)
  (hm : m 1 = 1)                -- Manny's consumption
  (ha : a 0 = 0)                -- Aaron's consumption
  (hk : ∀ n, k n = 2 * (m 1))   -- Kai's consumption
  (hr : ∀ n, r n = (1 / 2) * (m 1)) -- Raphael's consumption
  (hl : ∀ n, l n = 2 + (r n))   -- Lisa's consumption
  : m 1 + a 0 + k 1 + r 1 + l 1 = 6 :=
by
  -- Proof goes here
  sorry

end lasagna_pieces_l132_132419


namespace smallest_positive_integer_last_four_digits_l132_132042

theorem smallest_positive_integer_last_four_digits :
  ∃ n : ℕ,
    (n % 10000 = 4444) ∧
    (∀ m : ℕ, (m.pos ∧ (m % 4 = 0) ∧ (m % 9 = 0) ∧ (∀ d : ℕ, d ∈ (digits 10 m) → d = 4 ∨ d = 9) ∧ (count 4 (digits 10 m) ≥ 2) ∧ (count 9 (digits 10 m) ≥ 2) → n ≤ m)) :=
by 
  sorry

end smallest_positive_integer_last_four_digits_l132_132042


namespace compound_interest_rounded_l132_132259

theorem compound_interest_rounded (P : ℝ) (r : ℝ) (t : ℝ) (A : ℝ) (interest : ℝ) : 
    P = 14800 ∧ r = 0.135 ∧ n = 1 ∧ t = 2 ∧ 
    A = P * (1 + r/n)^(n*t) ∧ 
    interest = A - P ∧ 
    real.round interest = 4266 := 
by 
    sorry

end compound_interest_rounded_l132_132259


namespace combines_like_terms_l132_132895

theorem combines_like_terms (a : ℝ) : 2 * a - 5 * a = -3 * a := 
by sorry

end combines_like_terms_l132_132895


namespace team_a_daily_work_rate_l132_132081

theorem team_a_daily_work_rate
  (L : ℕ) (D1 : ℕ) (D2 : ℕ) (w : ℕ → ℕ)
  (hL : L = 8250)
  (hD1 : D1 = 4)
  (hD2 : D2 = 7)
  (hwB : ∀ (x : ℕ), w x = x + 150)
  (hwork : ∀ (x : ℕ), D1 * x + D2 * (x + (w x)) = L) :
  ∃ x : ℕ, x = 400 :=
by
  sorry

end team_a_daily_work_rate_l132_132081


namespace distance_to_Big_Rock_l132_132563

-- Define the conditions
def rower_speed := 7 -- speed in still water
def river_speed := 1 -- speed of the river current
def total_time := 1 -- total round trip time (in hours)

-- Define the effective speeds
def downstream_speed := rower_speed + river_speed
def upstream_speed := rower_speed - river_speed

-- Define the equation for distance calculation
def distance (t1 t2 : ℝ) : ℝ :=
  downstream_speed * t1 -- distance covered downstream
  = upstream_speed * t2 -- distance covered upstream

-- The main theorem to prove
theorem distance_to_Big_Rock (t1 t2 : ℝ) (h1 : t1 + t2 = total_time) (h2 : 8 * (1 - t2) = 6 * t2) :
  distance t1 t2 = (3.43 : ℝ) := by
  -- Proof is not included
  sorry

end distance_to_Big_Rock_l132_132563


namespace geometric_sequence_second_term_l132_132094

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l132_132094


namespace shortest_distance_correct_l132_132502

def eq_circle : ℝ → ℝ → Prop := 
λ x y, x^2 - 6*x + y^2 - 8*y + 15 = 0

def center : ℝ × ℝ := (3, 4)

def radius : ℝ := real.sqrt 10

def distance_from_origin_to_center : ℝ := real.sqrt (3^2 + 4^2)

def shortest_distance_from_origin_to_circle : ℝ := distance_from_origin_to_center - radius

theorem shortest_distance_correct :
  shortest_distance_from_origin_to_circle = 5 - real.sqrt 10 :=
by sorry

end shortest_distance_correct_l132_132502


namespace inequality_l132_132521

def f (n : ℕ) : ℝ := ((2 * n + 1 : ℝ) / Real.exp 1) ^ ((2 * n + 1 : ℝ) / 2)

def product_odds (n : ℕ) : ℝ := ∏ k in Finset.range n + 1, (2 * k + 1)

theorem inequality (n : ℕ) (h : 0 < n) : 
  f (n - 1) < product_odds n ∧ product_odds n < f n :=
by sorry

end inequality_l132_132521


namespace parabola_translation_shift_downwards_l132_132119

theorem parabola_translation_shift_downwards :
  ∀ (x y : ℝ), (y = x^2 - 5) ↔ ((∃ (k : ℝ), k = -5 ∧ y = x^2 + k)) :=
by
  sorry

end parabola_translation_shift_downwards_l132_132119


namespace geometric_sequence_tenth_term_l132_132961

theorem geometric_sequence_tenth_term :
  let a : ℚ := 3
  let r : ℚ := 5 / 2
  let n : ℕ := 10
  let a_n : ℚ := a * r^(n - 1)
  in a_n = 5859375 / 512 :=
by
  sorry

end geometric_sequence_tenth_term_l132_132961


namespace find_f_l132_132992

noncomputable def f : ℤ → ℤ
def P (f : ℤ → ℤ) := ∀ n : ℤ, f (f n) + f n = 2 * n + 3
def Q (f : ℤ → ℤ) := f 0 = 1
def A (f : ℤ → ℤ) := ∀ n : ℤ, f n = n + 1

theorem find_f : ∀ f : ℤ → ℤ, (P f) → (Q f) → (A f) :=
by
  sorry

end find_f_l132_132992


namespace mushroom_collectors_l132_132812

theorem mushroom_collectors :
  ∃ (n m : ℕ), 13 * n - 10 * m = 2 ∧ 9 ≤ n ∧ n ≤ 15 ∧ 11 ≤ m ∧ m ≤ 20 ∧ n = 14 ∧ m = 18 := by sorry

end mushroom_collectors_l132_132812


namespace problem_statement_l132_132411

noncomputable def prop_p (x : ℝ) : Prop := (1 / (x - 2)) < 0
def prop_q (x : ℝ) : Prop := (x^2 - 4 * x - 5) < 0

theorem problem_statement (x : ℝ) : 
  (¬ (prop_p x ∧ prop_q x) ∧ (prop_p x ∨ prop_q x)) → (x ∈ set.Ioo (-∞) (-1) ∪ set.Ico 3 5) :=
begin
  sorry
end

end problem_statement_l132_132411


namespace person_last_name_length_l132_132441

theorem person_last_name_length (samantha_lastname: ℕ) (bobbie_lastname: ℕ) (person_lastname: ℕ) 
  (h1: samantha_lastname + 3 = bobbie_lastname)
  (h2: bobbie_lastname - 2 = 2 * person_lastname)
  (h3: samantha_lastname = 7) :
  person_lastname = 4 :=
by 
  sorry

end person_last_name_length_l132_132441


namespace arrangement_count_l132_132874

theorem arrangement_count (n r : ℕ) (h : r < n / 2 - 1) : 
  let factorial : ℕ → ℕ := λ x, if x = 0 then 1 else x * factorial (x - 1)
  2 * factorial (n - 2) = 2 * (n - 2)! :=
by
  sorry

end arrangement_count_l132_132874


namespace product_increases_sum_unchanged_l132_132069

-- Define the original product of numbers from 100 to 200.
noncomputable def original_product : ℤ := ∏ i in (finset.range (200 - 100 + 1)).map (λ i, i + 100), i

-- Define the product when all values are replaced by 150.
noncomputable def replaced_product : ℤ := 150 ^ (200 - 100 + 1)

-- Define the original sum of numbers from 100 to 200.
noncomputable def original_sum : ℤ := ∑ i in (finset.range (200 - 100 + 1)).map (λ i, i + 100), i

-- Define the sum when all values are replaced by 150.
noncomputable def replaced_sum : ℤ := 150 * (200 - 100 + 1)

-- Theorems to prove
theorem product_increases : replaced_product > original_product :=
sorry

theorem sum_unchanged : replaced_sum = original_sum :=
sorry

end product_increases_sum_unchanged_l132_132069


namespace connie_initial_marbles_l132_132957

theorem connie_initial_marbles (marbles_given : ℕ) (marbles_left : ℕ) 
  (h_given : marbles_given = 73) (h_left : marbles_left = 70) : 
  marbles_given + marbles_left = 143 :=
by
  rw [h_given, h_left]
  exact rfl

end connie_initial_marbles_l132_132957


namespace triangle_area_correct_l132_132614

def point (α : Type*) := (α × α)

def triangle_area (A B C : point ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * Real.abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_correct :
  triangle_area (4, -3) (-1, 2) (2, -7) = 15 := 
sorry

end triangle_area_correct_l132_132614


namespace balls_not_fully_submerged_l132_132496

theorem balls_not_fully_submerged 
  (cylinder_diameter : ℝ) (water_volume : ℝ) 
  (ball1_diameter : ℝ) (ball2_diameter : ℝ) : 
  cylinder_diameter = 22 ∧ water_volume = 5000 ∧ ball1_diameter = 10 ∧ ball2_diameter = 14 →
  ∀ (h r₁ r₂ : ℝ), r₁ = ball1_diameter / 2 ∧ r₂ = ball2_diameter / 2 →
  let V_cylinder := (12 + 2 * (sqrt 11)) * π * ((cylinder_diameter / 2) ^ 2),
      V_spheres := (4 / 3) * π * (r₂^3 + r₁^3) in
  V_cylinder - V_spheres > water_volume :=
begin
  intros h r₁ r₂ h_eq_ball1_ball2,
  sorry
end

end balls_not_fully_submerged_l132_132496


namespace even_of_double_even_l132_132769

-- Given conditions
axiom f : ℝ → ℝ
axiom h_even : ∀ x : ℝ, f(-x) = f(x)

-- Prove that if f is even, then f(f(x)) is also even
theorem even_of_double_even (h_even : ∀ x : ℝ, f(-x) = f(x)) : ∀ x : ℝ, f(f(-x)) = f(f(x)) := by
  sorry

end even_of_double_even_l132_132769


namespace perpendicular_lines_condition_l132_132166

theorem perpendicular_lines_condition (m : ℝ) :
  (m = -1) ↔ ((m * 2 + 1 * m * (m - 1)) = 0) :=
sorry

end perpendicular_lines_condition_l132_132166


namespace molecular_weight_C7H6O2_l132_132889

noncomputable def molecular_weight_one_mole (w_9moles : ℕ) (m_9moles : ℕ) : ℕ :=
  m_9moles / w_9moles

theorem molecular_weight_C7H6O2 :
  molecular_weight_one_mole 9 1098 = 122 := by
  sorry

end molecular_weight_C7H6O2_l132_132889


namespace tangent_line_parabola_l132_132506

noncomputable def parabola := { p : ℝ × ℝ | p.2^2 - 4 * p.1 - 2 * p.2 + 1 = 0 }
noncomputable def line (k : ℝ) := { p : ℝ × ℝ | p.2 = k * p.1 + 2 }

theorem tangent_line_parabola (k : ℝ) :
  (∃ p : ℝ × ℝ, p ∈ parabola ∧ p ∈ line k ∧ 
  (∃! p' : ℝ × ℝ, p' ∈ parabola ∧ p' ∈ line k)) ↔
  (k = -2 + 2 * Real.sqrt 2) ∨ (k = -2 - 2 * Real.sqrt 2) :=
by sorry

end tangent_line_parabola_l132_132506


namespace volume_of_tetrahedron_ABCD_l132_132505

noncomputable section

open Matrix

def volume_of_tetrahedron (a b c d : ℝ^3) : ℝ :=
  (1 / 6) * abs (det.matrix ![![b - a, c - a, d - a]])

theorem volume_of_tetrahedron_ABCD :
  ∀ (A B C D : ℝ^3), 
    dist A B = 4 ∧
    dist A C = 5 ∧
    dist A D = 6 ∧
    dist B C = 2 * Real.sqrt 7 ∧
    dist B D = 5 ∧
    dist C D = Real.sqrt 34 → 
    volume_of_tetrahedron A B C D = 6 * Real.sqrt 1301 :=
by
  intros A B C D h,
  sorry

end volume_of_tetrahedron_ABCD_l132_132505


namespace reciprocal_of_minus_one_over_2023_l132_132864

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l132_132864


namespace parametric_curve_is_ray_l132_132892

theorem parametric_curve_is_ray:
  (∃ (t : ℝ), t ≥ 0 ∧ (x = sqrt t + 1 ∧ y = 1 - 2 * sqrt t)) →
  (∃ (P: ℝ × ℝ), 
    P = (1, 1) ∧ 
    ∀ (x₀ y₀ : ℝ), y₀ = 3 - 2 * x₀ → x₀ ≥ 1 → (x₀, y₀) = P ∨ (∃ k : ℝ, k ≥ 0 ∧ (x₀, y₀) = (1 + k, 1 - 2 * k))) :=
by sorry

end parametric_curve_is_ray_l132_132892


namespace sum_pairs_104_l132_132027

-- Define the arithmetic sequence
def arithmetic_sequence : finset ℤ := finset.image (λ n, 1 + (n - 1) * 3) (finset.range 34)

-- Define the statement of the problem
theorem sum_pairs_104 (A : finset ℤ) (hA : ∀ a ∈ A, a ∈ arithmetic_sequence) (h_card : A.card = 20) :
  ∃ x y ∈ A, x ≠ y ∧ x + y = 104 := sorry

end sum_pairs_104_l132_132027


namespace count_positive_integers_in_range_l132_132692

theorem count_positive_integers_in_range : 
  {n : ℕ | 300 < n^2 ∧ n^2 < 1200}.to_finset.card = 17 :=
by
  sorry

end count_positive_integers_in_range_l132_132692


namespace area_of_triangle_l132_132998

-- Define the vertices of the triangle in 3D space
def u := (0 : ℝ, 0, 0)
def v := (2 : ℝ, 4, 6)
def w := (1 : ℝ, 2, 1)

-- Compute the vectors originating from u
def v_minus_u := (2 : ℝ, 4, 6)
def w_minus_u := (1 : ℝ, 2, 1)

-- Compute the cross product of v_minus_u and w_minus_u
def cross_product := ( -8 : ℝ, 0, -6)

-- Compute the magnitude of the cross product vector
def magnitude := Real.sqrt ((-8)^2 + 0^2 + (-6)^2)

-- Prove the area of the triangle is 5
theorem area_of_triangle : (1 / 2) * magnitude = 5 := by
  -- Sorry is a placeholder for the proof
  sorry

end area_of_triangle_l132_132998


namespace distinct_points_l132_132263

namespace IntersectionPoints

open Set

theorem distinct_points :
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ x y : ℝ, ((x + 2*y - 8) * (3*x - y + 6) = 0) → (x, y) ∈ S) ∧
    (∀ x y : ℝ, ((2*x - 3*y + 2) * (4*x + y - 16) = 0) → (x, y) ∈ S) ∧
    S.card = 4 := by
  sorry

end IntersectionPoints

end distinct_points_l132_132263


namespace f_four_times_is_odd_l132_132401

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f(x)

-- Define f as an odd function
variable (f : ℝ → ℝ)
variable (H : is_odd f)

-- State the problem: Prove that f(f(f(f(x)))) is also an odd function
theorem f_four_times_is_odd : is_odd (λ x, f (f (f (f x)))) :=
sorry

end f_four_times_is_odd_l132_132401


namespace triangle_y_value_l132_132733

theorem triangle_y_value (EF GH : StraightLine)
  (angle_EMF : ∠EMF = 110)
  (angle_NMP : ∠NMP = 70)
  (angle_MNP : ∠MNP = 40) :
  y = 40 := 
by
  sorry

end triangle_y_value_l132_132733


namespace min_omega_for_shifted_graphs_l132_132674

/-- Given the function f(x) = sin(ωx + φ) and ω > 0, 
if the graph of f(x) is shifted to the left by π/3 units and overlaps with the graph 
of f(x) shifted to the right by π/6 units, then the minimum value of ω is 4. --/
theorem min_omega_for_shifted_graphs (ω φ : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 3) + φ) = sin (ω * (x - π / 6) + φ)) -> ω = 4 :=
begin
  sorry
end

end min_omega_for_shifted_graphs_l132_132674


namespace sum_f_equals_390_l132_132670

def f (x : ℝ) : ℝ := 2 * log x / log 5 * log 5 / log 2 + 14

theorem sum_f_equals_390 :
  (f 2 + f 4 + f 8 + f 16 + f 32 + f 64 + f 128 + f 256 + f 512 + f 1024) = 390 :=
sorry

end sum_f_equals_390_l132_132670


namespace group_arrangement_l132_132363

-- Definition of the problem parameters
def men : Finset ℕ := {0, 1, 2, 3}
def women : Finset ℕ := {0, 1, 2, 3, 4}

-- Definition of the proof statement
theorem group_arrangement : 
  let group_conditions := 
    (λ grp1 grp2 grp3 : Finset ℕ, grp1.card = 3 ∧ grp2.card = 2 ∧ grp3.card = 2 ∧
                                   grp1 ∩ grp2 = ∅ ∧ grp1 ∩ grp3 = ∅ ∧ grp2 ∩ grp3 = ∅ ∧
                                   ∃ (m1 m2 m3 wl : ℕ) (grp1 ⊆ men ∪ women), 
                                     grp1 = {m1, m2, wl} ∧
                                     grp2 = {men.erase m1, women.erase wl} ∧ 
                                     grp3 = {men.erase m2, women.erase wl}) in
  ∀ (grps : Finset (Finset ℕ)), grps.card = 3 → group_conditions grps = 240 :=
sorry

end group_arrangement_l132_132363


namespace a_is_perfect_square_l132_132390

theorem a_is_perfect_square {a : ℕ} (h : ∀ n : ℕ, ∃ d : ℕ, d ≠ 1 ∧ d % n = 1 ∧ d ∣ n ^ 2 * a - 1) : ∃ k : ℕ, a = k ^ 2 :=
by
  sorry

end a_is_perfect_square_l132_132390


namespace probability_of_third_round_expected_value_of_X_variance_of_X_l132_132572

-- Define the probabilities for passing each round
def P_A : ℚ := 2 / 3
def P_B : ℚ := 3 / 4
def P_C : ℚ := 4 / 5

-- Prove the probability of reaching the third round
theorem probability_of_third_round :
  P_A * P_B = 1 / 2 := sorry

-- Define the probability distribution
def P_X (x : ℕ) : ℚ :=
  if x = 1 then 1 / 3 
  else if x = 2 then 1 / 6
  else if x = 3 then 1 / 2
  else 0

-- Expected value
def EX : ℚ := 1 * (1 / 3) + 2 * (1 / 6) + 3 * (1 / 2)

theorem expected_value_of_X :
  EX = 13 / 6 := sorry

-- E(X^2) computation
def EX2 : ℚ := 1^2 * (1 / 3) + 2^2 * (1 / 6) + 3^2 * (1 / 2)

-- Variance
def variance_X : ℚ := EX2 - EX^2

theorem variance_of_X :
  variance_X = 41 / 36 := sorry

end probability_of_third_round_expected_value_of_X_variance_of_X_l132_132572


namespace ways_for_hikers_to_stay_in_two_rooms_l132_132636

theorem ways_for_hikers_to_stay_in_two_rooms : 
  let num_ways := (Nat.choose 5 3) * 2 
  in num_ways = 20 :=
by
  sorry

end ways_for_hikers_to_stay_in_two_rooms_l132_132636


namespace interval_overlap_l132_132242

theorem interval_overlap (x : ℝ) : 
  (2 / 5 < x ∧ x < 3 / 5) →
  (4 / 7 < x ∧ x < 6 / 7) →
  (4 / 7 < x ∧ x < 3 / 5) :=
by {
  intros h1 h2,
  split;
  apply lt_of_lt_of_le;
  tauto
  sorry
}

end interval_overlap_l132_132242


namespace x_intercept_is_2_l132_132487

noncomputable def x_intercept_of_line : ℝ :=
  by
  sorry -- This is where the proof would go

theorem x_intercept_is_2 :
  (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → y = 0 → x = 2) :=
  by
  intro x y H_eq H_y0
  rw [H_y0] at H_eq
  simp at H_eq
  sorry -- This is where the proof would go

end x_intercept_is_2_l132_132487


namespace distance_between_cities_l132_132193

/-
Question: 
What is the distance between cities A and B?

Conditions: 
1. Departure of passenger and freight train at the same time.
2. Meeting after 4 hours.
3. Passenger train average speed: 115 km/h.
4. Freight train average speed: 85 km/h.
-/

theorem distance_between_cities 
  (departure_simultaneous : Prop)
  (meet_time : ℝ)
  (passenger_speed : ℝ)
  (freight_speed : ℝ)
  (combined_speed : passenger_speed + freight_speed = 200)
  (total_time : meet_time = 4) :
  let distance := (passenger_speed + freight_speed) * meet_time in
  distance = 800 :=
by
  -- The proof is skipped.
  sorry

end distance_between_cities_l132_132193


namespace annual_plan_exceeded_by_13_2_percent_l132_132735

variables (x y : ℝ)
noncomputable def first_quarter_production : ℝ := 0.25 * x
noncomputable def second_quarter_production : ℝ := 0.27 * x
noncomputable def proportional_constant : ℝ := second_quarter_production / 11.25
noncomputable def third_quarter_production : ℝ := 12 * proportional_constant
noncomputable def fourth_quarter_production : ℝ := 13.5 * proportional_constant
noncomputable def total_production : ℝ := first_quarter_production + second_quarter_production + third_quarter_production + fourth_quarter_production

def overfulfillment_percentage : ℝ := ((total_production - x) / x) * 100

theorem annual_plan_exceeded_by_13_2_percent : overfulfillment_percentage = 13.2 :=
by
  sorry

end annual_plan_exceeded_by_13_2_percent_l132_132735


namespace proof_problem1_proof_problem2_proof_problem3_l132_132909

noncomputable def problem1 : Prop :=
  0.5⁻¹ + 4 ^ 0.5 = 4

noncomputable def problem2 : Prop :=
  Real.log 2 + Real.log 5 - (Real.pi / 23)^0 = 0

noncomputable def problem3 : Prop :=
  (2 - Real.sqrt 3)⁻¹ + (2 + Real.sqrt 3)⁻¹ = 1

theorem proof_problem1 : problem1 := by
  sorry

theorem proof_problem2 : problem2 := by
  sorry

theorem proof_problem3 : problem3 := by
  sorry

end proof_problem1_proof_problem2_proof_problem3_l132_132909


namespace parity_A68_is_odd_l132_132522

/-- 
Define \(A_n\) as the number of arranged \(n\)-tuples of natural numbers \((a_1, a_2, \ldots, a_n)\)
such that \(\frac{1}{a_1} + \frac{1}{a_2} + \ldots + \frac{1}{a_n} = 1\).

We need to prove that \(A_{68}\) is odd.
-/
theorem parity_A68_is_odd (A : ℕ → ℕ) (hA : ∀ n, A n = (multiset.nat_subsets n 1).card) :
  odd (A 68) :=
sorry

end parity_A68_is_odd_l132_132522


namespace pond_width_l132_132729

noncomputable def length : ℝ := 28
noncomputable def depth : ℝ := 5
noncomputable def volume : ℝ := 1400

theorem pond_width (W : ℝ) : 
  (volume = length * W * depth) → W = 10 :=
by 
  intro h,
  sorry

end pond_width_l132_132729


namespace length_of_train_l132_132571

-- Define the conditions as variables
def speed : ℝ := 39.27272727272727
def time : ℝ := 55
def length_bridge : ℝ := 480

-- Calculate the total distance using the given conditions
def total_distance : ℝ := speed * time

-- Prove that the length of the train is 1680 meters
theorem length_of_train :
  (total_distance - length_bridge) = 1680 :=
by
  sorry

end length_of_train_l132_132571


namespace correct_statement_l132_132897

theorem correct_statement : -3 > -5 := 
by {
  sorry
}

end correct_statement_l132_132897


namespace cone_lateral_surface_area_l132_132201

noncomputable def lateral_surface_area_of_cone (v : ℝ) : ℝ :=
  let r := (3 * v / (4 * π))^(1/3)
  let l := 2 * sqrt(3) * r
  let P := (l^2 * π) / 2
  in P

theorem cone_lateral_surface_area (v : ℝ) (h : v = 100) : 
  lateral_surface_area_of_cone v = 156.28 :=
by
  rw [h]
  sorry

end cone_lateral_surface_area_l132_132201


namespace least_non_lucky_multiple_of_11_l132_132967

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

def is_multiple_of_11 (n : ℕ) : Prop :=
n % 11 = 0

def is_lucky_integer (n : ℕ) : Prop :=
n % sum_of_digits n = 0

theorem least_non_lucky_multiple_of_11 : ∃ (n : ℕ), is_multiple_of_11 n ∧ ¬is_lucky_integer n ∧ (∀ m, is_multiple_of_11 m ∧ ¬is_lucky_integer m → n ≤ m) :=
by {
  use 11,
  split,
  { have h₁: 11 % 11 = 0, by norm_num,
    exact h₁ },
  split,
  { have h₂: 11 % sum_of_digits 11 ≠ 0,
    show 11 % sum_of_digits 11 ≠ 0,
    rw [sum_of_digits, sum],
    norm_num,
    norm_num,
    exact h₂ },
  { intros m hm,
    cases hm with hm₁ hm₂,
    have hsome : ∀ n, is_multiple_of_11 n → sum_of_digits 11 ≤ 2 := λ n h, sorry,
    sorry },
  exact sorry
}

end least_non_lucky_multiple_of_11_l132_132967


namespace fold_circle_creases_l132_132925

noncomputable def crease_set (R a : ℝ) (h : 0 < a < R) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let (x, y) := p in 
    ((x - a / 2) ^ 2 / (R / 2) ^ 2) + (y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2)) = 1 }

theorem fold_circle_creases (R a : ℝ) (h : 0 < a < R):
  let ellipse_eq := "((x - a / 2) ^ 2 / (R / 2) ^ 2) + (y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2)) = 1" in
  (∀ (x y : ℝ), (x, y) ∈ crease_set R a h ↔ ellipse_eq) :=
  sorry

end fold_circle_creases_l132_132925


namespace perfect_number_l132_132557

-- Definitions to indicate perfect number and necessary conditions
def isPerfectNumber (n : ℕ) : Prop :=
  n = (List.sum (List.filter (λ d, d < n) (List.range (n+1)).filter (λ d, n % d = 0)))

-- Given conditions for five numbers
def A : ℕ := 10
def B : ℕ := 13
def C : ℕ := 6
def D : ℕ := 8
def E : ℕ := 9

-- Theorem to prove that C is the perfect number among A, B, C, D, and E
theorem perfect_number :
  (isPerfectNumber A = false) ∧
  (isPerfectNumber B = false) ∧
  (isPerfectNumber C = true) ∧
  (isPerfectNumber D = false) ∧
  (isPerfectNumber E = false) :=
by
  sorry

end perfect_number_l132_132557


namespace ratio_movies_allowance_l132_132943

variable (M A : ℕ)
variable (weeklyAllowance moneyEarned endMoney : ℕ)
variable (H1 : weeklyAllowance = 8)
variable (H2 : moneyEarned = 8)
variable (H3 : endMoney = 12)
variable (H4 : weeklyAllowance + moneyEarned - M = endMoney)
variable (H5 : A = 8)
variable (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1)

theorem ratio_movies_allowance (M A : ℕ) 
  (weeklyAllowance moneyEarned endMoney : ℕ)
  (H1 : weeklyAllowance = 8)
  (H2 : moneyEarned = 8)
  (H3 : endMoney = 12)
  (H4 : weeklyAllowance + moneyEarned - M = endMoney)
  (H5 : A = 8)
  (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1) :
  M / A = 1 / 2 :=
sorry

end ratio_movies_allowance_l132_132943


namespace set_complement_union_l132_132686

open Set

theorem set_complement_union (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (hU : U = univ) (hA : A = {x | ∃ y, y = Real.sqrt (1 - x)}) (hB : B = {x | 0 < x ∧ x < 2}) :
  (U \ A) ∪ B = Ioi 0 := by
  sorry

end set_complement_union_l132_132686


namespace greek_food_students_l132_132088

def total_students : ℕ := 150
def percent_pizza : ℕ := 20
def percent_thai : ℕ := 38

theorem greek_food_students :
  let percent_greek := 100 - percent_pizza - percent_thai in
  let number_greek := (percent_greek * total_students) / 100 in
  number_greek = 63 :=
by
  sorry

end greek_food_students_l132_132088


namespace find_angle_B_of_triangle_ABC_l132_132613

-- Define the given conditions of the problem
def Triangle (A B C : Type) : Prop := 
  ∃ (α β γ : ℝ), 
    α + β + γ = 180 ∧ 
    (α = 75 ∨ β = 75 ∨ γ = 75) ∧ 
    ((CH : ℝ) = AB / 2)

-- Define the specific angles for the triangle
def TriangleAngle (A B C : Type) : Prop := 
  ∃ (α β γ : ℝ), 
    (α = 75) ∧ 
    (β = 30)

-- Prove that β is 30 given the conditions
theorem find_angle_B_of_triangle_ABC 
  (A B C : Type)
  (CH AB : ℝ)
  (h1 : Triangle A B C)
  (h2 : TriangleAngle A B C)
  : β = 30 :=
sorry

end find_angle_B_of_triangle_ABC_l132_132613


namespace number_of_functions_with_range_1_4_l132_132628

theorem number_of_functions_with_range_1_4 :
  ∃ (s : Finset (Finset Int)), (∀ f ∈ s, ∀ y ∈ ({1, 4} : Finset Int), ∃ x ∈ f, y = x^2) ∧ s.card = 9 :=
begin
  sorry,
end

end number_of_functions_with_range_1_4_l132_132628


namespace sum_midpoint_coordinates_l132_132837

-- Define the endpoints
def point1 : (ℕ × ℤ) := (4, -1)
def point2 : (ℕ × ℤ) := (12, 7)

-- Define the midpoint formula
def midpoint (p1 p2 : ℕ × ℤ) : ℚ × ℚ :=
    ((p1.1 + p2.1 : ℕ) / 2, (p1.2 + p2.2 : ℤ) / 2)

-- Prove the sum of the coordinates of the midpoint equals 11
theorem sum_midpoint_coordinates :
    let mid := midpoint point1 point2 in
    (mid.1 + mid.2 : ℚ) = 11 := by
  -- Here 'mid' is defined but 'sorry' is used to skip the proof
  sorry

end sum_midpoint_coordinates_l132_132837


namespace abc_values_l132_132647

theorem abc_values (a b c : ℝ) 
  (ha : |a| > 1) 
  (hb : |b| > 1) 
  (hc : |c| > 1) 
  (hab : b = a^2 / (2 - a^2)) 
  (hbc : c = b^2 / (2 - b^2)) 
  (hca : a = c^2 / (2 - c^2)) : 
  a + b + c = 6 ∨ a + b + c = -4 ∨ a + b + c = -6 :=
sorry

end abc_values_l132_132647


namespace average_monthly_balance_l132_132941

def balance_January : ℝ := 50
def balance_February : ℝ := 250
def balance_March : ℝ := 100
def balance_April : ℝ := 200
def balance_May : ℝ := 150
def balance_June : ℝ := 250

def average_balance (b1 b2 b3 b4 b5 b6 : ℝ) : ℝ :=
  (b1 + b2 + b3 + b4 + b5 + b6) / 6

theorem average_monthly_balance :
  average_balance balance_January balance_February balance_March balance_April balance_May balance_June = 166.67 :=
by 
  sorry

end average_monthly_balance_l132_132941


namespace area_of_triangle_ABC_l132_132786

theorem area_of_triangle_ABC (BD CE : ℝ) (angle_BD_CE : ℝ) (BD_len : BD = 9) (CE_len : CE = 15) (angle_BD_CE_deg : angle_BD_CE = 60) : 
  ∃ area : ℝ, 
    area = 90 * Real.sqrt 3 := 
by
  sorry

end area_of_triangle_ABC_l132_132786


namespace tom_driving_speed_l132_132753

theorem tom_driving_speed
  (v : ℝ)
  (hKarenSpeed : 60 = 60) -- Karen drives at an average speed of 60 mph
  (hKarenLateStart: 4 / 60 = 1 / 15) -- Karen starts 4 minutes late, which is 1/15 hours
  (hTomDistance : 24 = 24) -- Tom drives 24 miles before Karen wins the bet
  (hTimeEquation: 24 / v = 8 / 15): -- The equation derived from given conditions
  v = 45 := 
by
  sorry

end tom_driving_speed_l132_132753


namespace hf_eval_g_l132_132703

def g (x : ℕ) : ℕ := x * x
def f (x : ℕ) : ℕ := 2 * x - 1
def h (x : ℕ) : ℕ := x + 3

theorem hf_eval_g (x : ℕ) : h (f (g x)) = 20 :=
by
  have g3 := g 3           -- g(3) = 9
  have f9 := f g3          -- f(9) = 17
  have h17 := h f9         -- h(17) = 20
  exact h17

end hf_eval_g_l132_132703


namespace slope_of_line_through_A_B_l132_132667

theorem slope_of_line_through_A_B :
  let A := (2, 1)
  let B := (-1, 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -2/3 :=
by
  have A_x : Int := 2
  have A_y : Int := 1
  have B_x : Int := -1
  have B_y : Int := 3
  sorry

end slope_of_line_through_A_B_l132_132667


namespace median_of_sequence_l132_132375

theorem median_of_sequence : ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 300) →
  let list := (List.range' 1 300).bind (λ i, List.replicate i i) in
  list.length = 45150 ∧
  (list.nth_le (45150 / 2 - 1) (by simp [list.length_pos]) = 212 ∧
   list.nth_le (45150 / 2) (by simp [list.length_pos]) = 212) :=
by
  sorry

end median_of_sequence_l132_132375


namespace seq1_sum_is_correct_b_sum_is_correct_l132_132314

-- First sequence definition: a_n
def a (n : ℕ) (h : n > 0) : ℝ := 1 / (2 * n - 1)

-- Second sequence definition: (a_n + 2) / a_n
def seq1 (n : ℕ) (h : n > 0) : ℝ := (a n h + 2) / a n h

-- Sum of first n terms of seq1
def S (n : ℕ) (h : n > 0) : ℝ := (n * (3 + (n-1) * 4)) / 2 -- Sequence sum formula for (3 + 7 + 11 + ...)

theorem seq1_sum_is_correct (n : ℕ) (h : n > 0) : 
  (finset.range n).sum (λ i, seq1 (i + 1) (nat.succ_pos _)) = (2 * n^2 + n) := 
sorry

-- Third sequence definition: b_n
def b (n : ℕ) (h : n > 0) : ℝ := a n h * a (n + 1) (nat.succ_pos _)

-- Sum of first n terms of seq3
def T (n : ℕ) (h : n > 0) : ℝ := (n: ℝ) / (2 * n + 1)

theorem b_sum_is_correct (n : ℕ) (h : n > 0) : 
  (finset.range n).sum (λ i, b (i + 1) (nat.succ_pos _)) = (n:ℝ) / (2 * n + 1) := 
sorry

end seq1_sum_is_correct_b_sum_is_correct_l132_132314


namespace chef_potatoes_l132_132537

theorem chef_potatoes (total_potatoes cooked_potatoes time_per_potato rest_time: ℕ)
  (h1 : total_potatoes = 15)
  (h2 : time_per_potato = 9)
  (h3 : rest_time = 63)
  (h4 : time_per_potato * (total_potatoes - cooked_potatoes) = rest_time) :
  cooked_potatoes = 8 :=
by sorry

end chef_potatoes_l132_132537


namespace max_BP_squared_l132_132033

-- Definitions of the problem's elements
def Circle (ω : Type) := sorry -- Placeholder for the type representing a circle
def Point := sorry -- Placeholder for the type representing a point

variables (ω : Circle) (A B C T P : Point)
variable [geometry : geometric_context ω]

-- Length definitions
def length_AB : ℝ := 20
def radius := length_AB / 2
def O := midpoint A B
def length_OC := 30

-- Assumptions based on the problem conditions
axiom diameter_AB : is_diameter (A, B) ω
axiom on_circle_T : point_on_circle T ω
axiom CT_tangent : tangent_line C T ω
axiom perpendicular_AP_CT : perpendicular_line A P (line C T)

-- Proof goal
theorem max_BP_squared : BP^2 = 1900 / 3 := by
  sorry

end max_BP_squared_l132_132033


namespace number_of_functions_with_odd_sum_l132_132627

theorem number_of_functions_with_odd_sum : 
  (∑ k in finset.range 1997, if k % 2 = 1 then nat.choose 1996 k else 0) = 2^1995 :=
sorry

end number_of_functions_with_odd_sum_l132_132627


namespace x_needs_21_days_to_finish_work_alone_l132_132163

variable (W : ℝ)
variable (Wx : ℝ)
variable (Wy : ℝ)
variable (Remaining_work : ℝ)
variable (Dx : ℝ)

def work_done_by_y_in_5_days (Wy : ℝ) : ℝ :=
  5 * Wy

def remaining_work (W : ℝ) (work_done_by_y_in_5_days : ℝ) : ℝ :=
  W - work_done_by_y_in_5_days

def remaining_work_done_by_x (remaining_work : ℝ) : ℝ :=
  remaining_work / 14

def work_done_per_day_x (W : ℝ) (Dx : ℝ) : ℝ :=
  W / Dx

theorem x_needs_21_days_to_finish_work_alone :
  let Wy := W / 15
  let work_done_by_y_in_5_days := work_done_by_y_in_5_days Wy
  let remaining_work := remaining_work W work_done_by_y_in_5_days
  let remaining_work_done_by_x := remaining_work_done_by_x remaining_work
  let work_done_per_day_x := work_done_per_day_x W Dx
  (2 / 3) * (W / 14) = W / Dx → Dx = 21
:= by
  intros
  sorry

end x_needs_21_days_to_finish_work_alone_l132_132163


namespace positive_correlation_34_condition_l132_132489

-- Definitions for conditions
def correlation (a b : Type) : Type := Prop
def positive_correlation (a b : Type) : correlation a b := sorry
def negative_correlation (a b : Type) : correlation a b := sorry
def no_correlation (a b : Type) : correlation a b := sorry

-- Conditions
def condition_1 : correlation (Type := selling_price) (Type := sales_volume) := negative_correlation
def condition_2 : correlation (Type := student_id) (Type := math_score) := no_correlation
def condition_3 : correlation (Type := breakfast_consumers) (Type := stomach_disease_cases) := negative_correlation
def condition_4 : correlation (Type := temperature) (Type := cold_drink_sales) := positive_correlation
def condition_5 : correlation (Type := bicycle_weight) (Type := electricity_consumption) := positive_correlation

-- The proof problem statement
theorem positive_correlation_34_condition : 
  positive_correlation (Type := temperature) (Type := cold_drink_sales) ∧ 
  positive_correlation (Type := bicycle_weight) (Type := electricity_consumption) :=
begin
  split;
  exact condition_4,
  exact condition_5,
end

end positive_correlation_34_condition_l132_132489


namespace proof_A_spend_percentage_l132_132531

noncomputable def A_salary : ℝ := 2999.9999999999995
def total_salary : ℝ := 4000
def B_salary : ℝ := total_salary - A_salary
def B_spend_percentage : ℝ := 0.85

def B_spend : ℝ := B_spend_percentage * B_salary
def A_spend : ℝ := B_spend
def A_spend_percentage : ℝ := (A_spend / A_salary) * 100

theorem proof_A_spend_percentage :
  A_spend_percentage = 28.33333333333335 := 
by
  sorry

end proof_A_spend_percentage_l132_132531


namespace mark_total_young_fish_l132_132420

-- Define the conditions
def num_tanks : ℕ := 5
def fish_per_tank : ℕ := 6
def young_per_fish : ℕ := 25

-- Define the total number of young fish
def total_young_fish := num_tanks * fish_per_tank * young_per_fish

-- The theorem statement
theorem mark_total_young_fish : total_young_fish = 750 :=
  by
    sorry

end mark_total_young_fish_l132_132420


namespace correct_propositions_l132_132399

-- Condition definitions
variable {a b c : Type} -- Representing lines
variable {M : Type} -- Representing a plane

-- Propositions as conditions
def prop1 (a_parallel_M : a → M) (b_parallel_M : b → M) : Prop :=
  (∃ (x: a → b), x ∨ (a ∧ b) ∨ (¬ x))

def prop2 (b_in_M : b → M) (a_parallel_b : a → b) : Prop :=
  (a → M)

def prop3 (a_perp_c : ∀ c : c, a) (b_perp_c : ∀ c : c, b) : Prop :=
  (a → b)

def prop4 (a_perp_M : a → M) (b_perp_M : b → M) : Prop :=
  (a → b)

-- Question translated to Lean 4
theorem correct_propositions (a b c : Type) (M : Type)
  (a_parallel_M : a → M) (b_parallel_M : b → M)
  (b_in_M : b → M) (a_parallel_b : a → b)
  (a_perp_c : ∀ c : c, a) (b_perp_c : ∀ c : c, b)
  (a_perp_M : a → M) (b_perp_M : b → M) : 
  (prop1 a_parallel_M b_parallel_M) ∧ (prop4 a_perp_M b_perp_M) :=
by
  sorry

end correct_propositions_l132_132399


namespace geometric_sequence_second_term_l132_132095

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l132_132095


namespace solution_l132_132138

noncomputable theory

open Real EuclideanSpace

variables (a b c : EuclideanSpace ℝ (Fin 2)) -- Vectors in the plane ℝ²

-- Define that the vectors a, b, and c are unit vectors
def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖v‖ = 1

-- The main theorem
theorem solution (h_a : is_unit_vector a) (h_b : is_unit_vector b) (h_c : is_unit_vector c):
  ∃ signs : Fin 3 → bool, 
  let x := (if signs 0 then 1 else -1) • a + (if signs 1 then 1 else -1) • b + (if signs 2 then 1 else -1) • c in
  ‖x‖ ≤ sqrt 2 := 
begin
  sorry
end

end solution_l132_132138


namespace trig_expression_value_find_two_alpha_minus_beta_l132_132167

theorem trig_expression_value :
  sin^2 (120 * π / 180) + cos (180 * π / 180) + tan (45 * π / 180) - cos^2 (-330 * π / 180) + sin (-210 * π / 180) = 1 / 2 :=
by sorry

theorem find_two_alpha_minus_beta (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π) 
  (h3 : tan (α - β) = 1 / 2) (h4 : tan β = -1 / 7) :
  2 * α - β = -3 * π / 4 :=
by sorry

end trig_expression_value_find_two_alpha_minus_beta_l132_132167


namespace joe_needs_more_cars_l132_132384

-- Definitions based on conditions
def current_cars : ℕ := 50
def total_cars : ℕ := 62

-- Theorem based on the problem question and correct answer
theorem joe_needs_more_cars : (total_cars - current_cars) = 12 :=
by
  sorry

end joe_needs_more_cars_l132_132384


namespace smallest_power_of_7_not_palindrome_l132_132632

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_power_of_7_not_palindrome : ∃ n : ℕ, n > 0 ∧ 7^n = 2401 ∧ ¬is_palindrome (7^n) ∧ (∀ m : ℕ, m > 0 ∧ ¬is_palindrome (7^m) → 7^n ≤ 7^m) :=
by
  sorry

end smallest_power_of_7_not_palindrome_l132_132632


namespace value_of_expression_l132_132340

theorem value_of_expression (x y : ℝ) (h : x + 2 * y = 3) : 2^x * 4^y = 8 := by
  sorry

end value_of_expression_l132_132340


namespace probability_of_all_black_grid_is_3_over_16_l132_132958

noncomputable def probability_all_black : ℚ :=
  let probability_independent : ℕ → ℚ := λ n, (1:ℚ)/2^n
  let all_black_initial : ℚ := probability_independent 4
  let one_white_swap_black : ℚ := 4 * probability_independent 4 * (1:ℚ)/2
  all_black_initial + one_white_swap_black

theorem probability_of_all_black_grid_is_3_over_16 :
  probability_all_black = 3/16 := by
  sorry

end probability_of_all_black_grid_is_3_over_16_l132_132958


namespace proposition_correctness_l132_132945

noncomputable def proposition1 (a b : Line) (α : Plane) : Prop :=
  (a.parallel_to b ∧ a.parallel_to α) → b.parallel_to α

noncomputable def proposition2 (a b : Line) (α : Plane) : Prop :=
  (a.parallel_to α ∧ b ∈ α) → α.parallel_to b

noncomputable def proposition3 (a : Line) (α : Plane) : Prop :=
  a.parallel_to α → ∀ l ∈ α, a.parallel_to l

noncomputable def proposition4 (a b : Line) (α : Plane) : Prop :=
  (a.parallel_to α ∧ a.parallel_to b ∧ ¬(b ∈ α)) → b.parallel_to α

theorem proposition_correctness (a b : Line) (α : Plane) :
  (proposition4 a b α ∧ ¬proposition1 a b α ∧ ¬proposition2 a b α ∧ ¬proposition3 a α) := 
sorry

end proposition_correctness_l132_132945


namespace average_weight_increase_l132_132829

variable {W : ℝ} -- Total weight before replacement
variable {n : ℝ} -- Number of men in the group

theorem average_weight_increase
  (h1 : (W - 58 + 83) / n - W / n = 2.5) : n = 10 :=
by
  sorry

end average_weight_increase_l132_132829


namespace exists_group_round_table_l132_132211

open Finset Function

variable (P : Finset ℤ) (knows : ℤ → ℤ → Prop)

def has_at_least_three_friends (P : Finset ℤ) (knows : ℤ → ℤ → Prop) : Prop :=
  ∀ p ∈ P, (P.filter (knows p)).card ≥ 3

noncomputable def exists_even_group (P : Finset ℤ) (knows : ℤ → ℤ → Prop) : Prop :=
  ∃ S : Finset ℤ, (S ⊆ P) ∧ (2 < S.card) ∧ (Even S.card) ∧ (∀ p ∈ S, ∀ q ∈ S, Edge_connected p q knows S)

theorem exists_group_round_table (P : Finset ℤ) (knows : ℤ → ℤ → Prop) 
  (h : has_at_least_three_friends P knows) : 
  exists_even_group P knows :=
sorry

end exists_group_round_table_l132_132211


namespace value_of_f_x_plus_2_l132_132702

noncomputable def f : ℝ → ℝ := λ x, 2

theorem value_of_f_x_plus_2 (x : ℝ) : f (x + 2) = 2 :=
by
  -- Proof omitted
  sorry

end value_of_f_x_plus_2_l132_132702


namespace find_sin_CBD_l132_132221

noncomputable def sin_CBD (AB AC BC : ℝ) (angle_A : ℝ) (D B C : Point) : ℝ :=
  let BD := AB
  let CD := BC
  let CL := sqrt ((sqrt 3)^2 - (BD / 2)^2)
  let DH := CL / (2 * sqrt 3)
  let sin_CBD := DH / BD
  sin_CBD

theorem find_sin_CBD :
  ∀ (AB AC BC : ℝ) (angle_A : ℝ) (D B C : Point),
    AB = 1 →
    AC = 1 →
    angle_A = 120 →
    BC = sqrt 3 →
    sin_CBD AB AC BC angle_A D B C = sqrt 33 / 6 :=
by {
  intros AB AC BC angle_A D B C hAB hAC hangle hBC,
  sorry
}

end find_sin_CBD_l132_132221


namespace b_joined_after_a_l132_132568

def months_b_joined (a_investment : ℕ) (b_investment : ℕ) (profit_ratio : ℕ × ℕ) (total_months : ℕ) : ℕ :=
  let a_months := total_months
  let b_months := total_months - (b_investment / (3500 * profit_ratio.snd / profit_ratio.fst / b_investment))
  total_months - b_months

theorem b_joined_after_a (a_investment b_investment total_months : ℕ) (profit_ratio : ℕ × ℕ) (h_a_investment : a_investment = 3500)
   (h_b_investment : b_investment = 21000) (h_profit_ratio : profit_ratio = (2, 3)) : months_b_joined a_investment b_investment profit_ratio total_months = 9 := by
  sorry

end b_joined_after_a_l132_132568


namespace problem_statement_l132_132227

theorem problem_statement : (-1 : ℝ) + |2 * real.sqrt 2 - 3| + real.cbrt 8 = 4 - 2 * real.sqrt 2 := by
  sorry

end problem_statement_l132_132227


namespace shift_graph_to_left_by_2_units_l132_132494

variable {α : Type} [Add α] [Mul α] [HasEquiv α] -- Define α with Add and Mul operations.

-- Assuming a function y = f(3x)
variable (f : α → α) (x : α) 

-- Define a theorem to establish the required shift.
theorem shift_graph_to_left_by_2_units (a : α) (h : 3 * a = 6) : 
  f (3 * (x + 2)) = f (3 * x + 6) :=
by
  -- The definition of shift is based on substituting a specific value.
  -- Since the proof isn't required, we leave a placeholder.
  sorry

end shift_graph_to_left_by_2_units_l132_132494


namespace hexagon_angle_AP_l132_132825

theorem hexagon_angle_AP (a d : ℝ) :
  (∃ k ∈ {0, 1, 2, 3, 4, 5}, angle = a + k * d) ∧ 
  (6 * a + 15 * d = 720) → 
  (∃ angle k₂, (k₂ ∈ {0, 1, 2, 3, 4, 5}) ∧ ((a + k₂ * d) = 72)) :=
sorry

end hexagon_angle_AP_l132_132825


namespace yardsCatchingPasses_l132_132474

-- Definitions from conditions in a)
def totalYardage : ℕ := 150
def runningYardage : ℕ := 90

-- Problem statement (Proof will follow)
theorem yardsCatchingPasses : totalYardage - runningYardage = 60 := by
  sorry

end yardsCatchingPasses_l132_132474


namespace ice_floe_mass_l132_132794

/-- Given conditions: 
 - The bear's mass is 600 kg
 - The diameter of the bear's trail on the ice floe is 9.5 meters
 - The observed diameter of the trajectory from the helicopter is 10 meters

 We need to prove that the mass of the ice floe is 11400 kg.
 -/
 theorem ice_floe_mass (m d D : ℝ) (hm : m = 600) (hd : d = 9.5) (hD : D = 10) :
   let M := m * d / (D - d)
   in M = 11400 := by 
 sorry

end ice_floe_mass_l132_132794


namespace intersection_points_distance_polar_line_eq_through_C_l132_132736

noncomputable def curve1_polar := ∀ (ρ θ : ℝ), ρ = 2
noncomputable def curve2_polar := ∀ (ρ θ : ℝ), ρ * Real.sin (theta - Real.pi / 4) = Real.sqrt 2
noncomputable def line_through_C_parallel_to_AB := ∀ (ρ θ : ℝ), point_C : ℝ × ℝ := (1, 0), 
                                                      l_parallel_eq := sqrt 2 * ρ * sin (θ - π / 4) = 1

theorem intersection_points_distance : ∀ (A B : ℝ × ℝ),
                           let curve1_eq := curve1_polar ∀ (A', B') :
                           direct (curve1_polar, curve2_polar).A :  3, A B * ( curve2_polar ρ * ( Real.sqrt 2)) 
                           || ((x * y ).cos.  A -B )= 2 sqrt  2 sorry

theorem polar_line_eq_through_C : (C : ℝ × ℝ), C = (1, 0) : δ A B → ρ * Real.sin (θ - Real.pi / 4) = 1 : cos( sqrt( x+y)) = (A-B)
 sorry

end intersection_points_distance_polar_line_eq_through_C_l132_132736


namespace orthogonal_pairs_in_cube_is_36_l132_132638

-- Define a cube based on its properties, i.e., having vertices, edges, and faces.
structure Cube :=
(vertices : Fin 8 → Fin 3)
(edges : Fin 12 → (Fin 2 → Fin 8))
(faces : Fin 6 → (Fin 4 → Fin 8))

-- Define orthogonal pairs of a cube as an axiom.
axiom orthogonal_line_plane_pairs (c : Cube) : ℕ

-- The main theorem stating the problem's conclusion.
theorem orthogonal_pairs_in_cube_is_36 (c : Cube): orthogonal_line_plane_pairs c = 36 :=
by { sorry }

end orthogonal_pairs_in_cube_is_36_l132_132638


namespace find_n_l132_132170

-- Define the context of the problem
variable (total_personnel : ℕ) (removed_1 : ℕ) (removed_3 : ℕ) (n : ℕ)

-- Define the conditions
def condition_1 := total_personnel = 37
def condition_2 := removed_1 = 1 ∧ total_personnel - removed_1 = 36 ∧ n ∣ 36
def condition_3 := removed_3 = 3 ∧ total_personnel - removed_3 = 34 ∧ n ∣ 34

-- The theorem we intend to prove
theorem find_n : condition_1 ∧ condition_2 ∧ condition_3 → n = 18 := by
 sorry

end find_n_l132_132170


namespace olympiad_even_group_l132_132214

theorem olympiad_even_group (P : Type) [Fintype P] [Nonempty P] (knows : P → P → Prop)
  (h : ∀ p, (Finset.filter (knows p) Finset.univ).card ≥ 3) :
  ∃ (G : Finset P), G.card > 2 ∧ G.card % 2 = 0 ∧ ∀ p ∈ G, ∃ q₁ q₂ ∈ G, q₁ ≠ p ∧ q₂ ≠ p ∧ knows p q₁ ∧ knows p q₂ :=
by
  sorry

end olympiad_even_group_l132_132214


namespace unique_solution_l132_132913

def unique_ordered_pair : Prop :=
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 
               (∃ x : ℝ, x = (m : ℝ)^(1/3) - (n : ℝ)^(1/3) ∧ x^6 + 4 * x^3 - 36 * x^2 + 4 = 0) ∧
               m = 2 ∧ n = 4

theorem unique_solution : unique_ordered_pair := sorry

end unique_solution_l132_132913


namespace regular_price_of_tire_l132_132125

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 3 = 240) : x = 79 :=
by
  sorry

end regular_price_of_tire_l132_132125


namespace tangent_line_at_point_l132_132458

-- Define the curve equation
def curve (x : ℝ) : ℝ := x^3 - 3*x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (0, 0)

-- Define the derivative of the curve
def derivative (f : ℝ → ℝ) : ℝ → ℝ := λ x, 3*x^2 - 3

-- Define the equation of the tangent line
def tangent_line_eq (x : ℝ) : ℝ := -3*x

-- The proof problem statement
theorem tangent_line_at_point :
  ∀ x y : ℝ, (x, y) = point_of_tangency → curve x = y → derivative curve x = -3 → tangent_line_eq x = -3*x :=
by
  intros x y hxy hcurve hderiv
  rw [hxy, hcurve, hderiv]
  sorry

end tangent_line_at_point_l132_132458


namespace toms_speed_l132_132755

/--
Karen places a bet with Tom that she will beat Tom in a car race by 4 miles 
even if Karen starts 4 minutes late. Assuming that Karen drives at 
an average speed of 60 mph and that Tom will drive 24 miles before 
Karen wins the bet. Prove that Tom's average driving speed is \( \frac{300}{7} \) mph.
--/
theorem toms_speed (
  (karen_speed : ℕ) (karen_lateness : ℚ) (karen_beats_tom_by : ℕ) 
  (karen_distance_when_tom_drives_24_miles : ℕ) 
  (karen_speed = 60) 
  (karen_lateness = 4 / 60) 
  (karen_beats_tom_by = 4) 
  (karen_distance_when_tom_drives_24_miles = 24)) : 
  ∃ tom_speed : ℚ, tom_speed = 300 / 7 :=
begin
  sorry
end

end toms_speed_l132_132755


namespace weights_balance_l132_132132

theorem weights_balance (k : ℕ) 
    (m n : ℕ → ℝ) 
    (h1 : ∀ i : ℕ, i < k → m i > n i) 
    (h2 : ∀ i : ℕ, i < k → ∃ j : ℕ, j ≠ i ∧ (m i + n j = n i + m j 
                                               ∨ m j + n i = n j + m i)) 
    : k = 1 ∨ k = 2 := 
by sorry

end weights_balance_l132_132132


namespace trigonometric_identity_l132_132292

theorem trigonometric_identity (x : ℝ) (h : (1 + Real.sin x) / Real.cos x = -1/2) : 
  Real.cos x / (Real.sin x - 1) = 1/2 := 
sorry

end trigonometric_identity_l132_132292


namespace evaluate_difference_of_squares_l132_132986
-- Import necessary libraries

-- Define the specific values for a and b
def a : ℕ := 72
def b : ℕ := 48

-- State the theorem to be proved
theorem evaluate_difference_of_squares : a^2 - b^2 = (a + b) * (a - b) ∧ (a + b) * (a - b) = 2880 := 
by
  -- The proof would go here but should be omitted as per directions
  sorry

end evaluate_difference_of_squares_l132_132986


namespace max_value_cos_2x_plus_2sin_x_l132_132463

theorem max_value_cos_2x_plus_2sin_x : ∀ x, ∃ t : ℝ, -1 ≤ t ∧ t ≤ 1 ∧ y = cos (2 * x + 2 * sin x) → y ≤ 3 / 2 :=
by
  sorry

end max_value_cos_2x_plus_2sin_x_l132_132463


namespace number_of_integers_satisfying_inequalities_l132_132694

theorem number_of_integers_satisfying_inequalities :
  {n : ℕ | 300 < n^2 ∧ n^2 < 1200}.to_finset.card = 17 :=
by
  sorry

end number_of_integers_satisfying_inequalities_l132_132694


namespace area_of_trajectory_of_M_l132_132283

open Real

def distance (p q : ℝ × ℝ) : ℝ := sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem area_of_trajectory_of_M :
  (∀ (M : ℝ × ℝ), distance M (8, 0) = 2 * distance M (2, 0)) →
  ∃ (C : set (ℝ × ℝ)), (∀ M ∈ C, distance M (8, 0) = 2 * distance M (2, 0)) ∧
    ((∃ (r : ℝ), ∀ M ∈ C, (M.1)^2 + (M.2)^2 = r^2) ∧
      (π * 4^2) = 16 * π)：=
begin
sory
end

end area_of_trajectory_of_M_l132_132283


namespace width_of_cistern_is_6_l132_132181

-- Length of the cistern
def length : ℝ := 8

-- Breadth of the water surface
def breadth : ℝ := 1.85

-- Total wet surface area
def total_wet_surface_area : ℝ := 99.8

-- Let w be the width of the cistern
def width (w : ℝ) : Prop :=
  total_wet_surface_area = (length * w) + 2 * (length * breadth) + 2 * (w * breadth)

theorem width_of_cistern_is_6 : width 6 :=
  by
    -- This proof is omitted. The statement asserts that the width is 6 meters.
    sorry

end width_of_cistern_is_6_l132_132181


namespace curve_polar_eq_chord_length_l132_132666

def curve_parametric (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

def line_polar (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 / Real.sin (θ + Real.pi / 4)

theorem curve_polar_eq (ρ θ : ℝ) : (2 + 2 * Real.cos θ) ^ 2 + (2 * Real.sin θ)^2 = 4 ↔ ρ = 4 * Real.cos θ :=
by sorry

theorem chord_length (ρ θ : ℝ) : 
  ((2 + 2 * Real.cos θ - 2) ^ 2 + (2 * Real.sin θ - 2) ^ 2) = 4 ∧
  (2 + 2 * Real.cos θ + 2 * Real.sin θ - 4) = 0 ↔ 
  Real.sqrt ((4 - 2)^2 + (0 - 2)^2) = 2 * Real.sqrt 2 :=
by sorry

end curve_polar_eq_chord_length_l132_132666


namespace problem_statement_l132_132675

noncomputable def f (x : ℝ) := sqrt 3 * (Real.sin x * Real.cos x) - (Real.cos x) ^ 2 + 1/2

theorem problem_statement :
  ¬ (∃ x, x = π / 6 ∧ is_symmetry_axis f x) ∧
  ∀ x ∈ Icc (5 * π / 6) π, monotone_on f (Icc (5 * π / 6) π) ∧
  ∃ x, minimum ((f x) + (f (x + π / 4))) = -sqrt 2 ∧
  ∃ x, y, (f' x = sqrt 3) ∧ (f x = y) → y = sqrt 3 * x - 1/2 :=
by
  sorry

end problem_statement_l132_132675


namespace john_cannot_score_below_needed_tests_on_remaining_tests_l132_132585

-- Definitions for the conditions
def total_tests := 60
def required_percentage := 0.85
def required_tests := (required_percentage * total_tests).toNat
def tests_taken := 40
def tests_passed := 28
def tests_remaining := total_tests - tests_taken
def tests_needed := required_tests - tests_passed

-- Theorem statement: proving that it's impossible to score needed tests within remaining tests
theorem john_cannot_score_below_needed_tests_on_remaining_tests :
  tests_needed > tests_remaining :=
by sorry

end john_cannot_score_below_needed_tests_on_remaining_tests_l132_132585


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l132_132850

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l132_132850


namespace isosceles_triangle_CDE_l132_132800

theorem isosceles_triangle_CDE
  (A B C D E M : Point)
  (h1 : Collinear A B C D)
  (h2 : Midpoint M A B)
  (h3 : Midpoint M C D)
  (h4 : IsoscelesTriangle A B E) :
  IsoscelesTriangle C D E :=
sorry

end isosceles_triangle_CDE_l132_132800


namespace find_length_l132_132161

-- Let's define the conditions given in the problem
variables (b l : ℝ)

-- Length is more than breadth by 200%
def length_eq_breadth_plus_200_percent (b l : ℝ) : Prop := l = 3 * b

-- Total cost and rate per square meter
def cost_eq_area_times_rate (total_cost rate area : ℝ) : Prop := total_cost = rate * area

-- Given values
def total_cost : ℝ := 529
def rate_per_sq_meter : ℝ := 3

-- We need to prove that the length l is approximately 23 meters
theorem find_length (h1 : length_eq_breadth_plus_200_percent b l) 
    (h2 : cost_eq_area_times_rate total_cost rate_per_sq_meter (3 * b^2)) : 
    abs (l - 23) < 1 :=
by
  sorry -- Proof to be filled

end find_length_l132_132161


namespace count_M_formula_l132_132049

/-!
  Let a positive integer \( M \) in base \( n \) be represented such that all its digits are distinct,
  and each digit (except the leftmost one) differs by \(\pm 1\) from some digit to its left. 
  Prove that the number of all such positive integers \( M \) in terms of \( n \) is \( 2^{n+1} - 2n - 2 \).
-/

def distinct_and_adjacent (M : ℕ) (n : ℕ) : Prop :=
  -- M is represented in base n
  let digits := M.digits n in
  -- All digits are distinct
  digits.Nodup ∧ 
  -- Each digit except the first differs by ±1 from some digit to its left
  ∀ i, i < (digits.length - 1) → (digits[i + 1] = digits[i] + 1 ∨ digits[i + 1] = digits[i] - 1)

def count_M (n : ℕ) : ℕ :=
  -- The number of integers M in base n satisfying the conditions
  (Finset.Ico 1 (n^n)).card.filter (λ M, distinct_and_adjacent M n)

theorem count_M_formula (n : ℕ) : 
  count_M n = 2^(n+1) - 2 * n - 2 := sorry

end count_M_formula_l132_132049


namespace average_a_b_l132_132454

-- Defining the variables A, B, C
variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- The theorem stating that the average weight of a and b is 40 kg
theorem average_a_b (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : (A + B) / 2 = 40 :=
sorry

end average_a_b_l132_132454


namespace necessary_but_not_sufficient_l132_132665

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 < x) → ((x^2 < x) ↔ (0 < x ∧ x < 1)) ∧ ((1/x > 2) ↔ (0 < x ∧ x < 1/2)) := 
by 
  sorry

end necessary_but_not_sufficient_l132_132665


namespace part1_ab_next_to_each_other_part2_ab_next_to_each_other_ac_not_next_to_each_other_l132_132949

theorem part1_ab_next_to_each_other :
  let A := "A"
  let B := "B"
  let C := "C"
  let D := "D"
  let E := "E"
  (num_ways : ℕ) -- Number of ways to arrange A, B, C, D, E with A and B next to each other
  (h : num_ways = 48) : 
  ∃ arrangements : Finset (List String),
  |arrangements| = num_ways ∧ 
  ∀ arrangement ∈ arrangements, 
  let idx_A := arrangement.indexOf A
  let idx_B := arrangement.indexOf B
  (idx_A + 1 = idx_B ∨ idx_A - 1 = idx_B) :=
sorry

theorem part2_ab_next_to_each_other_ac_not_next_to_each_other :
  let A := "A"
  let B := "B"
  let C := "C"
  let D := "D"
  let E := "E"
  (num_ways : ℕ) -- Number of ways to arrange A, B, C, D, E with A and B next to each other and A and C not next to each other
  (h : num_ways = 36) : 
  ∃ arrangements : Finset (List String),
  |arrangements| = num_ways ∧ 
  ∀ arrangement ∈ arrangements, 
  let idx_A := arrangement.indexOf A
  let idx_B := arrangement.indexOf B
  let idx_C := arrangement.indexOf C
  (idx_A + 1 = idx_B ∨ idx_A - 1 = idx_B) ∧
  (idx_A + 1 ≠ idx_C ∧ idx_A - 1 ≠ idx_C) :=
sorry

end part1_ab_next_to_each_other_part2_ab_next_to_each_other_ac_not_next_to_each_other_l132_132949


namespace smallest_angle_between_lines_l132_132879

theorem smallest_angle_between_lines (r1 r2 r3 : ℝ) (S U : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) 
  (h3 : r3 = 2) (total_area : ℝ := π * (r1^2 + r2^2 + r3^2)) 
  (h4 : S = (5 / 8) * U) (h5 : S + U = total_area) : 
  ∃ θ : ℝ, θ = (5 * π) / 13 :=
by
  sorry

end smallest_angle_between_lines_l132_132879


namespace no_convex_polygon_is_hostile_hostile_polygon_exists_for_n_ge_4_l132_132391

-- Define a polygon type: a list of points in the plane (ℝ²).
structure Point :=
(x : ℝ)
(y : ℝ)

structure Polygon :=
(vertices : List Point)
(consistent : vertices.length ≥ 4)

-- Definition to check convexity of a polygon.
def is_convex (p : Polygon) : Prop := sorry -- Convexity definition of polygon

-- Definition to check hostility of a polygon.
def is_hostile (p : Polygon) : Prop :=
  ∀ (i : ℕ), i < p.vertices.length →
  let Q_i := closest_vertex p i in 
  (Q_i ≠ p.vertices[(i + 1) % p.vertices.length]) ∧ (Q_i ≠ p.vertices[(i - 1 + p.vertices.length) % p.vertices.length])

-- Define the closest vertex function
def closest_vertex (p : Polygon) (i : ℕ) : Point := sorry -- Function to find the closest vertex to p.vertices[i]

-- a) Prove that no convex polygon is hostile
theorem no_convex_polygon_is_hostile (p : Polygon) (hc : is_convex p) : ¬ is_hostile p :=
sorry

-- b) Find all n ≥ 4 for which there exists a hostile n-gon
theorem hostile_polygon_exists_for_n_ge_4 (n : ℕ) (hn : n ≥ 4) : ∃ p : Polygon, p.vertices.length = n ∧ is_hostile p :=
sorry

end no_convex_polygon_is_hostile_hostile_polygon_exists_for_n_ge_4_l132_132391


namespace water_tank_height_l132_132486

theorem water_tank_height (r h : ℝ) (V : ℝ) (V_water : ℝ) (a b : ℕ) 
  (h_tank : h = 120) (r_tank : r = 20) (V_tank : V = (1/3) * π * r^2 * h) 
  (V_water_capacity : V_water = 0.4 * V) :
  a = 48 ∧ b = 2 ∧ V = 16000 * π ∧ V_water = 6400 * π ∧ 
  h_water = 48 * (2^(1/3) / 1) ∧ (a + b = 50) :=
by
  sorry

end water_tank_height_l132_132486


namespace min_value_exp_l132_132646

theorem min_value_exp (x y : ℝ) (h : x + 2 * y = 4) : ∃ z : ℝ, (2^x + 4^y = z) ∧ (∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ z) :=
sorry

end min_value_exp_l132_132646


namespace oranges_for_profit_l132_132921

theorem oranges_for_profit (cost_buy: ℚ) (number_buy: ℚ) (cost_sell: ℚ) (number_sell: ℚ)
  (desired_profit: ℚ) (h₁: cost_buy / number_buy = 3.75) (h₂: cost_sell / number_sell = 4.5)
  (h₃: desired_profit = 120) :
  ∃ (oranges_to_sell: ℚ), oranges_to_sell = 160 ∧ (desired_profit / ((cost_sell / number_sell) - (cost_buy / number_buy))) = oranges_to_sell :=
by
  sorry

end oranges_for_profit_l132_132921


namespace f_2_eq_3_f_n_formula_l132_132410

-- Definition of good cell and f(n)
def is_good_cell (n : ℕ) (assign : fin (n+1)^2 → ℕ) (i j : fin n) : Prop :=
  let a := assign ⟨i.1, sorry⟩
  let b := assign ⟨i.1 + 1, sorry⟩
  let c := assign ⟨j.1, sorry⟩
  let d := assign ⟨j.1 + 1, sorry⟩
  (a < b ∧ b < d ∧ d < c) ∨ (b < d ∧ d < c ∧ c < a) ∨ (d < c ∧ c < a ∧ a < b) ∨ (c < a ∧ a < b ∧ b < d)

def f (n : ℕ) : ℕ := sorry

theorem f_2_eq_3 : f 2 = 3 := sorry

theorem f_n_formula (n : ℕ) : f n = (n^2 + 2*n - 1) / 2 := sorry

end f_2_eq_3_f_n_formula_l132_132410


namespace smallest_root_eq_2_l132_132266

noncomputable def smallest_root (x : ℝ) : Prop :=
  sqrt (x + 2) + 2 * sqrt (x - 1) + 3 * sqrt (3 * x - 2) = 10

theorem smallest_root_eq_2 :
  ∀ x : ℝ, (x + 2 ≥ 0) ∧ (x - 1 ≥ 0) ∧ (3 * x - 2 ≥ 0) → smallest_root x → x = 2 :=
by
  intros x hx small_root
  sorry

end smallest_root_eq_2_l132_132266


namespace exists_valid_circle_group_l132_132216

variable {P : Type}
variable (knows : P → P → Prop)

def knows_at_least_three (p : P) : Prop :=
  ∃ (p₁ p₂ p₃ : P), p₁ ≠ p ∧ p₂ ≠ p ∧ p₃ ≠ p ∧ knows p p₁ ∧ knows p p₂ ∧ knows p p₃

def valid_circle_group (G : List P) : Prop :=
  (2 < G.length) ∧ (G.length % 2 = 0) ∧ (∀ i, knows (G.nthLe i sorry) (G.nthLe ((i + 1) % G.length) sorry) ∧ knows (G.nthLe i sorry) (G.nthLe ((i - 1 + G.length) % G.length) sorry))

theorem exists_valid_circle_group (H : ∀ p : P, knows_at_least_three knows p) : 
  ∃ G : List P, valid_circle_group knows G := 
sorry

end exists_valid_circle_group_l132_132216


namespace broadcasting_methods_l132_132917

-- Definitions
def num_ads : Nat := 6
def num_comm_ads : Nat := 3
def num_olympic_ads : Nat := 2
def num_public_ads : Nat := 1

-- Conditions
def last_ad_not_commercial := ∀ α β γ δ ε ζ : Fin 6, ζ != α → ζ != β → ζ != γ
def olympic_ads_not_consecutive := ∀ α β : Fin 6, abs (α - β) > 1
def olympic_ads_not_with_public := ∀ α β γ : Fin 6, abs (α - γ) > 1 ∧ abs (β - γ) > 1

-- Theorem statement
theorem broadcasting_methods :
  (∃ (methods : Fin 108), 
    methods = list.product (finset.univ : finset (Fin num_ads)) (finset.univ : finset (Fin num_ads)) ∧
    last_ad_not_commercial ∧
    olympic_ads_not_consecutive ∧
    olympic_ads_not_with_public
  ) := sorry

end broadcasting_methods_l132_132917


namespace speed_of_stream_l132_132920

-- Definitions based on conditions
def boat_speed_still_water : ℕ := 24
def travel_time : ℕ := 4
def downstream_distance : ℕ := 112

-- Theorem statement
theorem speed_of_stream : 
  ∀ (v : ℕ), downstream_distance = travel_time * (boat_speed_still_water + v) → v = 4 :=
by
  intros v h
  -- Proof omitted
  sorry

end speed_of_stream_l132_132920


namespace positions_coincide_after_6_steps_l132_132358

-- Definitions based on the conditions
section transformations

variables {A B C D A' D' : Type*}
variables {P : ℕ → A} 

-- Assert the initial conditions with lengths of sides
variables [has_coe_to_fun A (λ _, ℝ)] -- The points can be coerced to real coordinates
variables (AB BC CD : ℝ)
variable (AD : ℝ)
variable (h1AB : AB = 1) (h1BC : BC = 1) (h1CD : CD = 1)
variable (hAD : AD ≠ 1)

-- Transformation function representing the reflection process
noncomputable def reflect (p q : A) : A := sorry  -- Define the reflection of point p across q

-- Process that updates the position of points A and D in a step
def transform_step (An Dn Bn Cn : A) : (A × A) :=
  let A_new := reflect An Dn in
  let D_new := reflect Dn A_new in
  (A_new, D_new)

-- Definition indicating positions coincide with the original after 6 steps
def positions_coincide_originally (A0 D0 A6 D6 : A) : Prop :=
  A0 = A6 ∧ D0 = D6

-- Lean statement encapsulating the equivalent proof problem
theorem positions_coincide_after_6_steps (A0 D0 B C : A) (trans : ∀ n, (A × A)) :
  ∃ n, n = 6 ∧ positions_coincide_originally (prod.fst (trans 0)) (prod.snd (trans 0)) 
  (prod.fst (trans 6)) (prod.snd (trans 6)) :=
begin
  sorry
end

end transformations

end positions_coincide_after_6_steps_l132_132358


namespace angle_in_quadrant_120_l132_132894

theorem angle_in_quadrant_120 (θ : ℝ) (h_origin : θ = 120) :
  90 < θ ∧ θ < 180 → θ ∈ second_quadrant := 
by sorry

-- Additional definitions presumed necessary for clarity
def second_quadrant := { θ : ℝ | 90 < θ ∧ θ < 180 }

end angle_in_quadrant_120_l132_132894


namespace nearest_integer_sum_of_x_for_gx_3005_l132_132526

noncomputable def g : ℝ → ℝ := sorry

def satisfies_condition (x : ℝ) : Prop :=
  3 * g x + g (1 / x) = 6 * x + 3

theorem nearest_integer_sum_of_x_for_gx_3005 :
  (∑ x in {x : ℝ | g x = 3005}, x).toInt = 1335 := 
sorry

end nearest_integer_sum_of_x_for_gx_3005_l132_132526


namespace problem_statement_l132_132673

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + 2 * x * derivative (fun x => Real.cos x + x * derivative (fun x => Real.cos x + 2 * x * derivative (fun x => Real.cos x + x)))

theorem problem_statement : 
  f (-Real.pi / 3) < f (Real.pi / 3) :=
by
  sorry

end problem_statement_l132_132673


namespace equal_distribution_in_classrooms_l132_132473

theorem equal_distribution_in_classrooms :
  ∀ (classrooms boys girls : ℕ), classrooms = 7 → boys = 68 → girls = 53 → 
  (∃ students_per_classroom, students_per_classroom = 14 ∧ 
    ∀ c, c < classrooms → equal_distribution students_per_classroom) :=
by 
  sorry

def equal_distribution (students_per_classroom : ℕ) : Prop := 
  students_per_classroom % 2 = 0

end equal_distribution_in_classrooms_l132_132473
