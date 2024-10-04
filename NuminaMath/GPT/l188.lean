import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Divisibility
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Star.Basic
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.GCD
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Probability.ProbabilitySpace
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclid
import Mathlib.GroupTheory.OrderOfElement
import Mathlib.LinearAlgebra.Matrix
import Mathlib.NumberTheory.ModularArithmetic.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Advanced
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Real

namespace greatest_value_is_B_l188_188565

def x : Int := -6

def A : Int := 2 + x
def B : Int := 2 - x
def C : Int := x - 1
def D : Int := x
def E : Int := x / 2

theorem greatest_value_is_B :
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  sorry

end greatest_value_is_B_l188_188565


namespace complement_U_A_l188_188160

open Set

def U : Set ℝ := {x | -3 < x ∧ x < 3}
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

theorem complement_U_A : 
  (U \ A) = {x | -3 < x ∧ x ≤ -2} ∪ {x | 1 < x ∧ x < 3} :=
by
  sorry

end complement_U_A_l188_188160


namespace karen_drive_l188_188229

theorem karen_drive (a b c x : ℕ) (h1 : a ≥ 1) (h2 : a + b + c ≤ 9) (h3 : 33 * (c - a) = 25 * x) :
  a^2 + b^2 + c^2 = 75 :=
sorry

end karen_drive_l188_188229


namespace product_of_solutions_l188_188458

open Int

theorem product_of_solutions (n : ℕ) (p : ℕ) (hp : Nat.Prime p) 
  (h : n^2 - 31 * n + 240 = p) : ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, m > 0 ∧ ((n^2 - 31 * n + 240 = 2) ∧ (m^2 - 31 * m + 240 = 2) ∧ (n * m = 238))) ∨ (n = 15 ∧ m = 16) :=
begin
  sorry
end

end product_of_solutions_l188_188458


namespace probability_meeting_part_a_l188_188605

theorem probability_meeting_part_a :
  ∃ p : ℝ, p = (11 : ℝ) / 36 :=
sorry

end probability_meeting_part_a_l188_188605


namespace smallest_degree_poly_with_given_roots_l188_188292

theorem smallest_degree_poly_with_given_roots :
  ∃ p : Polynomial ℚ, 
    (p.eval (4 - 3*Real.sqrt 3) = 0 ∧ p.eval (-4 - 3*Real.sqrt 3) = 0 ∧ 
     p.eval (2 + Real.sqrt 5) = 0 ∧ p.eval (2 - Real.sqrt 5) = 0) ∧
    p.degree = 6 :=
sorry

end smallest_degree_poly_with_given_roots_l188_188292


namespace nested_sqrt_eq_five_l188_188062

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l188_188062


namespace hexagon_same_length_probability_l188_188980

/-- 
Given a regular hexagon with 6 sides and 9 diagonals, if two segments are chosen randomly without 
replacement from the set of all sides and diagonals, the probability that the two chosen segments 
have the same length is \(\frac{17}{35}\). 
-/
theorem hexagon_same_length_probability : 
  let S := (finset.range 6).attach ++ (finset.range 9).attach in
  let num_ways_two_sides := (Finset.card (Finset.range 6)).choose 2 in
  let num_ways_two_diags := (Finset.card (Finset.range 9)).choose 2 in
  let total_ways := ((Finset.card S)).choose 2 in
  (num_ways_two_sides + num_ways_two_diags) / total_ways = 17 / 35 :=
by
  sorry

end hexagon_same_length_probability_l188_188980


namespace two_digit_numbers_count_l188_188913

theorem two_digit_numbers_count : 
  let S : Finset ℕ := {4, 5, 6, 7, 8} in
  let two_digit_count := S.card * (S.card - 1) in
  two_digit_count = 20 :=
by
  -- Set representing the digits to choose from
  let S : Finset ℕ := {4, 5, 6, 7, 8} in
  -- Calculate the number of ways to form two-digit numbers with different digits
  let two_digit_count := S.card * (S.card - 1) in
  -- Assert that the count is 20
  show two_digit_count = 20 from 
    sorry

end two_digit_numbers_count_l188_188913


namespace Sandra_brought_20_pairs_l188_188259

-- Definitions for given conditions
variable (S : ℕ) -- S for Sandra's pairs of socks
variable (C : ℕ) -- C for Lisa's cousin's pairs of socks

-- Conditions translated into Lean definitions
def initial_pairs : ℕ := 12
def mom_pairs : ℕ := 3 * initial_pairs + 8 -- Lisa's mom brought 8 more than three times the number of pairs Lisa started with
def cousin_pairs (S : ℕ) : ℕ := S / 5       -- Lisa's cousin brought one-fifth the number of pairs that Sandra bought
def total_pairs (S : ℕ) : ℕ := initial_pairs + S + cousin_pairs S + mom_pairs -- Total pairs of socks Lisa ended up with

-- The theorem to prove
theorem Sandra_brought_20_pairs (h : total_pairs S = 80) : S = 20 :=
by
  sorry

end Sandra_brought_20_pairs_l188_188259


namespace infinite_radical_solution_l188_188079

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l188_188079


namespace bowls_remaining_l188_188800

-- Definitions based on conditions.
def initial_collection : ℕ := 70
def reward_per_10_bowls : ℕ := 2
def total_customers : ℕ := 20
def customers_bought_20 : ℕ := total_customers / 2
def bowls_bought_per_customer : ℕ := 20
def total_bowls_bought : ℕ := customers_bought_20 * bowls_bought_per_customer
def reward_sets : ℕ := total_bowls_bought / 10
def total_reward_given : ℕ := reward_sets * reward_per_10_bowls

-- Theorem statement to be proved.
theorem bowls_remaining : initial_collection - total_reward_given = 30 :=
by
  sorry

end bowls_remaining_l188_188800


namespace chord_length_l188_188857

noncomputable def circle (x y : ℝ) (a : ℝ) := x^2 + y^2 - 2 * a * x + 4 * a * y + 5 * a^2 - 25 = 0
def line1 (x y : ℝ) := x + y + 2 = 0
def line2 (x y : ℝ) := 3 * x + 4 * y - 5 = 0

theorem chord_length 
  (a : ℝ)
  (h1 : line1 a (-2 * a))
  (h2 : distance (2, -4) line2 = 3) 
  (h3 : ∀ x y, circle x y 2 → line2 x y) 
  : 2 * Real.sqrt (25 - 9) = 8 := 
by sorry

end chord_length_l188_188857


namespace first_present_cost_is_18_l188_188230

-- Conditions as definitions
variables (x : ℕ)

-- Given conditions
def first_present_cost := x
def second_present_cost := x + 7
def third_present_cost := x - 11
def total_cost := first_present_cost x + second_present_cost x + third_present_cost x

-- Statement of the problem
theorem first_present_cost_is_18 (h : total_cost x = 50) : x = 18 :=
by {
  sorry  -- Proof omitted
}

end first_present_cost_is_18_l188_188230


namespace infinite_radical_solution_l188_188080

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l188_188080


namespace satisfy_eqn_l188_188450

/-- 
  Prove that the integer pairs (0, 1), (0, -1), (1, 0), (-1, 0), (2, 2), (-2, -2)
  are the only pairs that satisfy x^5 + y^5 = (x + y)^3
-/
theorem satisfy_eqn (x y : ℤ) : 
  (x, y) = (0, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (1, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (2, 2) ∨ (x, y) = (-2, -2) ↔ 
  x^5 + y^5 = (x + y)^3 := 
by 
  sorry

end satisfy_eqn_l188_188450


namespace relationship_among_numbers_l188_188119

noncomputable def a : ℝ := 0.2 ^ 3.5
noncomputable def b : ℝ := 0.2 ^ 4.1
noncomputable def c : ℝ := Real.exp 1.1
noncomputable def d : ℝ := Real.logBase 0.2 3

theorem relationship_among_numbers : d < b ∧ b < a ∧ a < c := by
  sorry

end relationship_among_numbers_l188_188119


namespace smallest_number_with_70_divisors_l188_188104

theorem smallest_number_with_70_divisors:
  ∃ n : ℕ, (∀ m : ℕ, (∀ d : ℕ, (d ∣ n ↔ d ∣ m) → m = n) ∧ (divisor_count n = 70 → n = 25920)) :=
sorry

noncomputable def divisor_count (n : ℕ) : ℕ :=
  ∑ d in divisors n, 1

lemma divisors (n : ℕ) : finset ℕ :=
  finset.filter (λ d, d ∣ n) (finset.range (n+1))

lemma divisor_count_correct (n : ℕ) : divisor_count n = 
  ∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n+1))), 1 :=
by sorry

end smallest_number_with_70_divisors_l188_188104


namespace smallest_value_of_Q_l188_188435

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - 4*x^2 + 2*x - 3

theorem smallest_value_of_Q :
  min (-10) (min 3 (-2)) = -10 :=
by
  -- Skip the proof
  sorry

end smallest_value_of_Q_l188_188435


namespace charlyn_viewable_area_l188_188418

noncomputable def charlyn_sees_area (side_length viewing_distance : ℝ) : ℝ :=
  let inner_viewable_area := (side_length^2 - (side_length - 2 * viewing_distance)^2)
  let rectangular_area := 4 * (side_length * viewing_distance)
  let circular_corner_area := 4 * ((viewing_distance^2 * Real.pi) / 4)
  inner_viewable_area + rectangular_area + circular_corner_area

theorem charlyn_viewable_area :
  let side_length := 7
  let viewing_distance := 1.5
  charlyn_sees_area side_length viewing_distance = 82 := 
by
  sorry

end charlyn_viewable_area_l188_188418


namespace length_of_train_correct_l188_188754

noncomputable def length_of_train (time_pass_man : ℝ) (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - man_speed_kmh
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  relative_speed_ms * time_pass_man

theorem length_of_train_correct :
  length_of_train 29.997600191984642 60 6 = 449.96400287976963 := by
  sorry

end length_of_train_correct_l188_188754


namespace exists_star_of_david_arrangement_l188_188262

noncomputable def star_of_david_arrangement : Prop :=
  ∃ (positions : Fin 7 → Fin 7 × Fin 5) (rows : Fin 5 → Set (Fin 7)) (in_row : Fin 7 ↔ Set (Fin 7)),

  -- positions map each bush (indexed by Fin 7) to a coordinate (x, y in Fin 7 × Fin 5)
  (∀ i : Fin 7, positions i ∈ (Fin 7) × (Fin 5)) ∧
  
  -- rows is a function that maps each row index (Fin 5) to a Set of bushes (indexed by Fin 7)
  (∀ j : Fin 5, rows j ⊆ set.univ ∧ set.card (rows j) = 3) ∧
  
  -- each position appears in exactly three rows
  (∀ i : Fin 7, set.card (in_row i) = 3) ∧
  
  -- each row has exactly three bushes
  (∀ j : Fin 5, set.card (rows j) = 3) ∧

  -- in_row function aligns with rows
  (∀ i : Fin 7, ∀ j : (Fin 5), i ∈ rows j ↔ in_row i = rows j)

-- Here, we state the existence of such an arrangement:
theorem exists_star_of_david_arrangement : star_of_david_arrangement :=
sorry

end exists_star_of_david_arrangement_l188_188262


namespace inscribed_triangle_area_le_half_parallelogram_l188_188283

theorem inscribed_triangle_area_le_half_parallelogram (b h : ℝ) 
  (ht : is_pos b) (hh : is_pos h) :
  ∀ (A B C : Point) (parallelogram : Parallelogram),
  (inscribed A B C parallelogram) → 
  area_triangle A B C ≤ (1 / 2) * (b * h) :=
by sorry

end inscribed_triangle_area_le_half_parallelogram_l188_188283


namespace zeros_in_140_times_5_seven_less_than_four_times_150_l188_188683

-- We define the first problem which states that the number of zeros at the end of the product of 140 and 5 is 2.
def number_of_zeros (n : ℕ) := 
  let s := n.to_string.reverse in 
  s.take_while (λ c, c = '0').length

theorem zeros_in_140_times_5 : number_of_zeros (140 * 5) = 2 := 
by
  sorry

-- We define the second problem which states that the number 7 less than 4 times 150 is 593.
theorem seven_less_than_four_times_150 : 4 * 150 - 7 = 593 := 
by
  sorry

end zeros_in_140_times_5_seven_less_than_four_times_150_l188_188683


namespace solve_for_A_in_terms_of_B_l188_188614

noncomputable def f (A B x : ℝ) := A * x - 2 * B^2
noncomputable def g (B x : ℝ) := B * x

theorem solve_for_A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 1) = 0) : A = 2 * B := by
  sorry

end solve_for_A_in_terms_of_B_l188_188614


namespace fido_reachable_area_fraction_l188_188829

def fraction_of_reachable_area (a b : ℕ) : ℝ :=
  (Real.sqrt a) / b * Real.pi

theorem fido_reachable_area_fraction {a b : ℕ} (r : ℝ)
  (h1 : r > 0) -- leash length is positive
  (h2 : fraction_of_reachable_area a b = Real.sqrt 2 / 4 * Real.pi) :
  a * b = 8 :=
sorry

end fido_reachable_area_fraction_l188_188829


namespace average_of_4_8_N_l188_188310

-- Define the condition for N
variable (N : ℝ) (cond : 7 < N ∧ N < 15)

-- State the theorem to prove
theorem average_of_4_8_N (N : ℝ) (h : 7 < N ∧ N < 15) :
  (frac12 + N) / 3 = 7 ∨ (12 + N) / 3 = 9 :=
sorry

end average_of_4_8_N_l188_188310


namespace infinite_radical_solution_l188_188078

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l188_188078


namespace ellipse_equation_line_equation_l188_188130

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def line_eq (m : ℝ) (x y : ℝ) : Prop :=
  m * y = x - 1

theorem ellipse_equation (a b c : ℝ)
  (h1 : a = 2)
  (h2 : c / a = (Real.sqrt 2) / 2)
  (h3 : a^2 = b^2 + c^2) :
  ellipse_eq 2 (Real.sqrt 2) x y ↔ ellipse_eq 2 (Real.sqrt 2) x y :=
sorry

theorem line_equation (a b c : ℝ) (m : ℝ)
  (h1 : a = 2)
  (h2 : c / a = (Real.sqrt 2) / 2)
  (h3 : a^2 = b^2 + c^2)
  (h4 : ∃ y1 y2 x1 x2 : ℝ, (y1, x1) ≠ (y2, x2) ∧ line_eq m 1 y1 ∧ line_eq m 1 y2)
  (h5 : let area_AMN : ℝ := (4 * Real.sqrt 2) / 5 in area_AMN = (4 * Real.sqrt 2) / 5) :
  line_eq (Real.sqrt 2 / 2) x y ∨ line_eq (-(Real.sqrt 2) / 2) x y :=
sorry

end ellipse_equation_line_equation_l188_188130


namespace chairlift_halfway_l188_188271

theorem chairlift_halfway (total_chairs current_chair halfway_chair : ℕ) 
  (h_total_chairs : total_chairs = 96)
  (h_current_chair : current_chair = 66) : halfway_chair = 18 :=
sorry

end chairlift_halfway_l188_188271


namespace even_function_expression_l188_188560

-- Definitions according to conditions in the problem
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 4 * x + 3 else 0

noncomputable def f_even (x : ℝ) : ℝ := f x
noncomputable def f_neg (x : ℝ) : ℝ := f (-x)

-- The theorem we aim to prove
theorem even_function_expression :
  (∀ x : ℝ, f_even x = f_neg x) →
  (∀ x : ℝ, x ≥ 0 → f x = 4 * x + 3) →
  ∀ x : ℝ, f x = if x ≥ 0 then 4 * x + 3 else -4 * x + 3 :=
by
  intros h_even h_expr x
  sorry

end even_function_expression_l188_188560


namespace more_ducks_than_four_times_chickens_l188_188224

def number_of_chickens (C : ℕ) : Prop :=
  185 = 150 + C

def number_of_ducks (C : ℕ) (MoreDucks : ℕ) : Prop :=
  150 = 4 * C + MoreDucks

theorem more_ducks_than_four_times_chickens (C MoreDucks : ℕ) (h1 : number_of_chickens C) (h2 : number_of_ducks C MoreDucks) : MoreDucks = 10 := by
  sorry

end more_ducks_than_four_times_chickens_l188_188224


namespace watermelons_last_6_weeks_l188_188225

variable (initial_watermelons : ℕ) (eaten_per_week : ℕ) (given_away_per_week : ℕ)

def watermelons_last_weeks (initial_watermelons : ℕ) (weekly_usage : ℕ) : ℕ :=
initial_watermelons / weekly_usage

theorem watermelons_last_6_weeks :
  initial_watermelons = 30 ∧ eaten_per_week = 3 ∧ given_away_per_week = 2 →
  watermelons_last_weeks initial_watermelons (eaten_per_week + given_away_per_week) = 6 := 
by
  intro h
  cases h with h_initial he 
  cases he with  h_eaten h_given
  have weekly_usage := h_eaten + h_given
  have weeks := watermelons_last_weeks h_initial weekly_usage
  sorry

end watermelons_last_6_weeks_l188_188225


namespace obtuse_triangle_probability_l188_188468

noncomputable def uniform_circle_distribution (radius : ℝ) : ProbabilitySpace (ℝ × ℝ) := sorry

def no_obtuse_triangle (points : list (ℝ × ℝ)) (center : (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k < 4 →
  ∀ (p₁ p₂ : (ℝ × ℝ)),
  (p₁ = points.nth i ∧ p₂ = points.nth j ∧ points.nth k ∉ set_obtuse_triangles p₁ p₂ center)

theorem obtuse_triangle_probability :
  let points := [⟨cos θ₀, sin θ₀⟩, ⟨cos θ₁, sin θ₁⟩, ⟨cos θ₂, sin θ₂⟩, ⟨cos θ₃, sin θ₃⟩]
  ∀ center : (ℝ × ℝ), 
  ∀ (h : set.center.circle center radius), 
  (no_obtuse_triangle points center) →
  Pr[(points)] = 3 / 32 :=
sorry

end obtuse_triangle_probability_l188_188468


namespace divisor_is_679_l188_188101

noncomputable def x : ℕ := 8
noncomputable def y : ℕ := 9
noncomputable def z : ℝ := 549.7025036818851
noncomputable def p : ℕ := x^3
noncomputable def q : ℕ := y^3
noncomputable def r : ℕ := p * q

theorem divisor_is_679 (k : ℝ) (h : r / k = z) : k = 679 := by
  sorry

end divisor_is_679_l188_188101


namespace cos_identity_15_30_degrees_l188_188039

theorem cos_identity_15_30_degrees (a b : ℝ) (h : b = 2 * a^2 - 1) : 2 * a^2 - b = 1 :=
by
  sorry

end cos_identity_15_30_degrees_l188_188039


namespace inequality_proof_l188_188996

theorem inequality_proof (x y z : ℝ) (n : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_sum : x + y + z = 1) :
  (x^4 / (y * (1 - y^n)) + y^4 / (z * (1 - z^n)) + z^4 / (x * (1 - x^n))) ≥ (3^n) / (3^(n+2) - 9) :=
by
  sorry

end inequality_proof_l188_188996


namespace min_diff_prime_composite_sum_95_l188_188371

def is_prime (n : ℕ) : Prop := nat.prime n
def is_composite (n : ℕ) : Prop := ∃ p1 p2 : ℕ, p1.prime ∧ p2.prime ∧ p1 * p2 = n

theorem min_diff_prime_composite_sum_95 :
  ∃ a b : ℕ, is_composite a ∧ is_prime b ∧ a + b = 95 ∧ (∀ a' b' : ℕ, is_composite a' → is_prime b' → a' + b' = 95 → |a' - b'| ≥ |a - b|):
  sorry

# Sanity check example:
example : ∃ a b : ℕ, is_composite a ∧ is_prime b ∧ a + b = 95 ∧ |a - b| = 1 :=
by
  use [48, 47]
  split
  . use [2, 24]
    repeat {norm_num}
  . split
  . exact nat.prime_def_lt'.mpr ⟨by norm_num, λ m hm, by norm_num⟩
  . split
  . norm_num
  . norm_num

end min_diff_prime_composite_sum_95_l188_188371


namespace craig_total_distance_l188_188426

-- Define the distances Craig walked
def dist_school_to_david : ℝ := 0.27
def dist_david_to_home : ℝ := 0.73

-- Prove the total distance walked
theorem craig_total_distance : dist_school_to_david + dist_david_to_home = 1.00 :=
by
  -- Proof goes here
  sorry

end craig_total_distance_l188_188426


namespace coefficient_highest_term_in_expansion_l188_188452

theorem coefficient_highest_term_in_expansion :
  let expr := (2 * x - 1) * ((3 - 2 * x) ^ 5)
  in ∃ c : ℤ, (∃ p : ℕ, expr.coeff p ≠ 0 ∧ (∀ n > p, expr.coeff n = 0) ∧ c = -64) := sorry

end coefficient_highest_term_in_expansion_l188_188452


namespace number_of_bookshelves_l188_188411

theorem number_of_bookshelves (books_per_shelf total_books : ℕ) (h_books_per_shelf : books_per_shelf = 2) (h_total_books : total_books = 38) :
  total_books / books_per_shelf = 19 :=
by
  rw [h_books_per_shelf, h_total_books]
  norm_num

end number_of_bookshelves_l188_188411


namespace classroom_activity_l188_188208

theorem classroom_activity (a b : ℤ) (h1 : 3 * a + 4 * b = 161) (h2 : a = 17 ∨ b = 17) : a = 31 ∨ b = 31 :=
by
  cases h2 with ha hb
  · -- case a = 17
    rw ha at h1
    have h3 : 51 + 4 * b = 161 := h1
    have h4 : 4 * b = 110 := by linarith
    have h5 : b = 27.5 := by norm_num [h4]
    contradiction
  · -- case b = 17
    rw hb at h1
    have h3 : 3 * a + 68 = 161 := h1
    have h4 : 3 * a = 93 := by linarith
    have h5 : a = 31 := by norm_num [h4]
    right
    exact h5

end classroom_activity_l188_188208


namespace katrina_cookies_sale_l188_188603

/-- 
Katrina has 120 cookies in the beginning.
She sells 36 cookies in the morning.
She sells 16 cookies in the afternoon.
She has 11 cookies left to take home at the end of the day.
Prove that she sold 57 cookies during the lunch rush.
-/
theorem katrina_cookies_sale :
  let total_cookies := 120
  let morning_sales := 36
  let afternoon_sales := 16
  let cookies_left := 11
  let cookies_sold_lunch_rush := total_cookies - morning_sales - afternoon_sales - cookies_left
  cookies_sold_lunch_rush = 57 :=
by
  sorry

end katrina_cookies_sale_l188_188603


namespace triangle_ABC_BD_length_l188_188931

open Real

noncomputable def length_BD (AB AC : ℝ) : ℝ :=
  let BC := sqrt (AB^2 + AC^2)
  let area_ABC := (1 / 2) * AB * AC
  let AD := (2 * area_ABC) / BC
  sqrt (AC^2 - AD^2)
  
theorem triangle_ABC_BD_length (AB AC : ℝ) (h1 : AB = 45) (h2 : AC = 60) :
  length_BD AB AC = 48 :=
by
  rw [h1, h2]
  simp [length_BD]
  sorry

end triangle_ABC_BD_length_l188_188931


namespace probability_y_gt_3x_eq_1_over_6_l188_188635

noncomputable def probability_y_gt_3x : ℝ :=
  let region_area := ∫ x in (0 : ℝ)..1, ∫ y in 3 * x..1, 1 in
  region_area

theorem probability_y_gt_3x_eq_1_over_6 :
  probability_y_gt_3x = 1 / 6 :=
sorry

end probability_y_gt_3x_eq_1_over_6_l188_188635


namespace find_x_l188_188732

theorem find_x (x : ℝ) (h : 0.90 * 600 = 0.50 * x) : x = 1080 :=
sorry

end find_x_l188_188732


namespace pentagon_area_proof_l188_188270

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨9, 1⟩
def B : Point := ⟨2, 0⟩
def D : Point := ⟨1, 5⟩
def E : Point := ⟨9, 7⟩

def line_eq (P Q : Point) : ℝ → ℝ :=
  let m := (Q.y - P.y) / (Q.x - P.x)
  let b := P.y - m * P.x
  λ x => m * x + b

def AD : ℝ → ℝ := line_eq A D
def BE : ℝ → ℝ := line_eq B E

def intersection (f g : ℝ → ℝ) : Point :=
  let x := (g(0) - f(0)) / (AD 1 - BE 1)
  let y := f x
  ⟨x, y⟩

def C : Point := intersection AD BE

def shoelace_area (vertices : List Point) : ℝ :=
  (0.5 * (List.sum (List.map prod vertices.zip (vertices.tail ++ [vertices.head])))
  -
   List.sum (List.map prod reverse (vertices.tail ++ [vertices.head]).zip (vertices.zip (vertices.tail)))
  ) :: List ℝ

def pentagon_area : ℝ :=
  shoelace_area [A, B, C, D, E]

theorem pentagon_area_proof : pentagon_area = 33 := by
  sorry

end pentagon_area_proof_l188_188270


namespace divisible_by_six_of_power_two_l188_188729

theorem divisible_by_six_of_power_two (n a b : ℤ) (h1 : n > 3) (h2 : 2^n = 10 * a + b) (h3 : b < 10) : 6 ∣ (a * b) :=
by
  sorry

end divisible_by_six_of_power_two_l188_188729


namespace closest_number_proof_l188_188021

noncomputable def M : ℝ := 3^361
noncomputable def N : ℝ := 10^80
noncomputable def M_div_N : ℝ := M / N

theorem closest_number_proof : M_div_N ≈ 10^93 := by
  sorry

end closest_number_proof_l188_188021


namespace hindi_speaking_children_l188_188207

-- Condition Definitions
def total_children : ℕ := 90
def percent_only_english : ℝ := 0.25
def percent_only_hindi : ℝ := 0.15
def percent_only_spanish : ℝ := 0.10
def percent_english_hindi : ℝ := 0.20
def percent_english_spanish : ℝ := 0.15
def percent_hindi_spanish : ℝ := 0.10
def percent_all_three : ℝ := 0.05

-- Question translated to a Lean statement
theorem hindi_speaking_children :
  (percent_only_hindi + percent_english_hindi + percent_hindi_spanish + percent_all_three) * total_children = 45 :=
by
  sorry

end hindi_speaking_children_l188_188207


namespace omega_max_value_l188_188704

theorem omega_max_value (f : ℝ → ℝ) (g : ℝ → ℝ) (ω : ℝ) (hω : ω > 0)
  (hf : ∀ x, f x = 2 * Real.sin (ω * x + π / 4))
  (hg : ∀ x, g x = f (x - π / (4 * ω)))
  (h_inc : ∀ x ∈ Icc (-π / 6) (π / 4), ∃ y ∈ (0 : ℝ, Real.pi / 2), ω * x = y) :
  ω = 1 :=
sorry

end omega_max_value_l188_188704


namespace exists_M_l188_188611

-- Define the initial function conditions
def f : ℕ × ℤ → ℕ
| (0, 0) := 5 ^ 2003
| (0, n) := 0
| (m+1, n) := 
    let a := f (m, n) in
    let b1 := (a / 2) in
    let b2 := (f (m, n-1) / 2) in
    let b3 := (f (m, n+1) / 2) in
    a - 2 * b1 + b2 + b3

-- Prove the existence of M
theorem exists_M (M : ℕ) :
  (∀ n : ℤ, |n| ≤ (5^2003 - 1) / 2 → f (M, n) = 1) ∧ 
  (∀ n : ℤ, |n| > (5^2003 - 1) / 2 → f (M, n) = 0) :=
sorry

end exists_M_l188_188611


namespace speed_on_flight_up_l188_188915

theorem speed_on_flight_up :
  ∃ (v : ℝ), 
  (∀ (speedHome avgSpeed : ℝ), speedHome = 90 ∧ avgSpeed = 100 →
    avgSpeed = (2 * v * speedHome) / (v + speedHome) →
    v = 112.5) :=
by
  use 112.5
  intros speedHome avgSpeed h_cond h_eq
  cases h_cond with h1 h2
  sorry

end speed_on_flight_up_l188_188915


namespace find_a3_l188_188219

noncomputable def a_n (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

theorem find_a3 (a : ℝ) (q : ℝ) (h₁ : a_n a q 4 - a_n a q 2 = 6)
  (h₂ : a_n a q 5 - a_n a q 1 = 15) : 
  a_n a q 3 = 4 ∨ a_n a q 3 = -4 :=
begin
  sorry
end

end find_a3_l188_188219


namespace conic_section_is_ellipse_l188_188721

theorem conic_section_is_ellipse :
  (∀ x y : ℝ, sqrt (x^2 + (y - 1)^2) + sqrt ((x - 5)^2 + (y + 3)^2) = 10) →
  (∃ e : conicSection, e = conicSection.ellipse) :=
by 
  sorry

end conic_section_is_ellipse_l188_188721


namespace average_speed_of_planes_l188_188329

-- Definitions for the conditions
def num_passengers_plane1 : ℕ := 50
def num_passengers_plane2 : ℕ := 60
def num_passengers_plane3 : ℕ := 40
def base_speed : ℕ := 600
def speed_reduction_per_passenger : ℕ := 2

-- Calculate speeds of each plane according to given conditions
def speed_plane1 := base_speed - num_passengers_plane1 * speed_reduction_per_passenger
def speed_plane2 := base_speed - num_passengers_plane2 * speed_reduction_per_passenger
def speed_plane3 := base_speed - num_passengers_plane3 * speed_reduction_per_passenger

-- Calculate the total speed and average speed
def total_speed := speed_plane1 + speed_plane2 + speed_plane3
def average_speed := total_speed / 3

-- The theorem to prove the average speed is 500 MPH
theorem average_speed_of_planes : average_speed = 500 := by
  sorry

end average_speed_of_planes_l188_188329


namespace polygon_area_equality_l188_188002

-- Definitions of the points and lines as given in the problem
variables (A B B' C C' D D' E E' F : Point)
variables (AB DE : Line)

-- Conditions based on the points and geometric figures
def condition1 : Prop := star_pentagon A A' B B' C C' D D' E E'
def condition2 : Prop := (AB : Line) ∧ (DE : Line) ∧ meets (AB) (DE) F

-- Areas of the respective polygons
def area_ABB'CC'DED' : ℝ := area (polygon.mk [A, B, B', C, C', D, E, D'])
def area_AD'E'F : ℝ := area (polygon.mk [A, D', E', F])

-- Theorem statement
theorem polygon_area_equality :
  condition1 ∧ condition2 → area_ABB'CC'DED' = area_AD'E'F :=
sorry

end polygon_area_equality_l188_188002


namespace percentage_allocation_industrial_lubricants_l188_188758

/-- Lean4 Statement: The percentage allocated to industrial lubricants is 8%,
    given the conditions about the percentages of other sectors and the degrees 
    allocated to basic astrophysics. -/
theorem percentage_allocation_industrial_lubricants :
  ∀ (microphotonics home_electronics food_additives gm_microorganisms : ℝ)
    (basic_astrophysics_degrees total_degrees : ℝ),
  microphotonics = 10 ∧ home_electronics = 24 ∧ food_additives = 15 ∧ 
  gm_microorganisms = 29 ∧ basic_astrophysics_degrees = 50.4 ∧ total_degrees = 360 →
  let basic_astrophysics := (basic_astrophysics_degrees / total_degrees) * 100 in
  let total_percentage_accounted := microphotonics + home_electronics + 
                                    food_additives + gm_microorganisms + 
                                    basic_astrophysics in
  let industrial_lubricants := 100 - total_percentage_accounted in
  industrial_lubricants = 8 :=
by
  intros _ _ _ _ _ _ H,
  simp only at H,
  cases H with H10 H,
  cases H with H24 H,
  cases H with H15 H,
  cases H with H29 H,
  cases H with H50.4 HT360,
  have H_basic : (50.4 / 360) * 100 = 14 := by
  -- Calculation for actual proof will go here
  sorry
  have H_total : 10 + 24 + 15 + 29 + 14 = 92 := by
  -- Calculation for actual proof will go here
  sorry
  have H_industrial : 100 - 92 = 8 := by
  -- Calculation for actual proof will go here
  sorry
  exact H_industrial

end percentage_allocation_industrial_lubricants_l188_188758


namespace average_age_is_approx_11_75_l188_188212

-- Define the conditions
def total_students : ℕ := 604
def avg_age_boys : ℕ := 12
def avg_age_girls : ℕ := 11
def num_girls : ℕ := 151

-- Derive the number of boys
def num_boys : ℕ := total_students - num_girls

-- Calculate the total age of boys and girls
def total_age_boys : ℕ := avg_age_boys * num_boys
def total_age_girls : ℕ := avg_age_girls * num_girls

-- Calculate the total age of all students
def total_age_students : ℕ := total_age_boys + total_age_girls

-- Calculate the average age of the school
noncomputable def average_age_school : ℚ := total_age_students / total_students

-- State the theorem
theorem average_age_is_approx_11_75 : average_age_school ≈ 11.75 := 
by 
  sorry

end average_age_is_approx_11_75_l188_188212


namespace ball_drawing_ways_l188_188322

theorem ball_drawing_ways :
    ∃ (r w y : ℕ), 
      0 ≤ r ∧ r ≤ 2 ∧
      0 ≤ w ∧ w ≤ 3 ∧
      0 ≤ y ∧ y ≤ 5 ∧
      r + w + y = 5 ∧
      10 ≤ 5 * r + 2 * w + y ∧ 
      5 * r + 2 * w + y ≤ 15 := 
sorry

end ball_drawing_ways_l188_188322


namespace location_of_items_l188_188723

-- Definitions for places
inductive Place
| under_pillow
| under_couch
| on_table
| under_table

-- Definitions for items
inductive Item
| notebook
| cheat_sheet
| player
| sneakers

open Place Item

-- Initial conditions as axioms
axiom item_needs_placement : ∀ (p : Place), ∃ (i : Item), true

axiom condition1_notebook_not_under_table : ¬ notebook = under_table ∧ ¬ player = under_table 
axiom condition2_cheat_sheet_not_on_floor : ¬ cheat_sheet = under_table ∧ ¬ cheat_sheet = under_couch
axiom condition3_player_not_on_table_or_under_couch : ¬ player = on_table ∧ ¬ player = under_couch

-- The theorem to be proven
theorem location_of_items :
  (notebook = under_couch ∧ cheat_sheet = on_table ∧ player = under_pillow ∧ sneakers = under_table) :=
begin
  -- The proof will be given here
  sorry
end

end location_of_items_l188_188723


namespace reach_50_from_49_l188_188382

def is_valid_move (n m : ℕ) : Prop := 
  m = 2 * n ∨ m = n / 10

def reachable (start target : ℕ) : Prop :=
  ∃ seq : List ℕ, seq.head? = some start ∧ seq.last? = some target ∧
                   ∀ (x y : ℕ), List.chain is_valid_move seq

theorem reach_50_from_49 : reachable 49 50 :=
by
  sorry

end reach_50_from_49_l188_188382


namespace infinite_radical_solution_l188_188081

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l188_188081


namespace real_root_uncertainty_l188_188572

noncomputable def f (x m : ℝ) : ℝ := m * x^2 - 2 * (m + 2) * x + m + 5
noncomputable def g (x m : ℝ) : ℝ := (m - 5) * x^2 - 2 * (m + 2) * x + m

theorem real_root_uncertainty (m : ℝ) :
  (∀ x : ℝ, f x m ≠ 0) → 
  (m ≤ 5 → ∃ x : ℝ, g x m = 0 ∧ ∀ y : ℝ, y ≠ x → g y m = 0) ∧
  (m > 5 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0) :=
sorry

end real_root_uncertainty_l188_188572


namespace jason_more_coins_l188_188787

-- Define the conditions
def jayden_coins : ℕ := 300
def total_coins : ℕ := 660

-- Prove that the difference in coins received by Jason and Jayden is 60
theorem jason_more_coins : (total_coins - jayden_coins) - jayden_coins = 60 :=
by
  -- Use corresponding calculation taken from solution steps
  have jason_coins := total_coins - jayden_coins
  show jason_coins - jayden_coins = 60
  sorry

end jason_more_coins_l188_188787


namespace sum_of_xy_l188_188159

theorem sum_of_xy (x y : ℝ) (h1 : x + 3 * y = 12) (h2 : 3 * x + y = 8) : x + y = 5 := 
by
  sorry

end sum_of_xy_l188_188159


namespace fuse_inequality_l188_188823

-- Definitions of the given conditions
def fuseLength (x : ℝ) := x > 0
def blasterSpeed := 3 
def fuseBurningSpeed := 0.2
def safeDistance := 50 

-- Lean statement
theorem fuse_inequality (x : ℝ) (hx : fuseLength x) : 
  (x / fuseBurningSpeed) > (safeDistance - x) / blasterSpeed := 
by 
  sorry

end fuse_inequality_l188_188823


namespace probability_defective_second_given_first_l188_188344

theorem probability_defective_second_given_first :
  let total_products := 20
  let defective_products := 4
  let genuine_products := 16
  let prob_A := defective_products / total_products
  let prob_AB := (defective_products * (defective_products - 1)) / (total_products * (total_products - 1))
  let prob_B_given_A := prob_AB / prob_A
  prob_B_given_A = 3 / 19 :=
by
  sorry

end probability_defective_second_given_first_l188_188344


namespace obtuse_triangle_probability_l188_188467

noncomputable def uniform_circle_distribution (radius : ℝ) : ProbabilitySpace (ℝ × ℝ) := sorry

def no_obtuse_triangle (points : list (ℝ × ℝ)) (center : (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k < 4 →
  ∀ (p₁ p₂ : (ℝ × ℝ)),
  (p₁ = points.nth i ∧ p₂ = points.nth j ∧ points.nth k ∉ set_obtuse_triangles p₁ p₂ center)

theorem obtuse_triangle_probability :
  let points := [⟨cos θ₀, sin θ₀⟩, ⟨cos θ₁, sin θ₁⟩, ⟨cos θ₂, sin θ₂⟩, ⟨cos θ₃, sin θ₃⟩]
  ∀ center : (ℝ × ℝ), 
  ∀ (h : set.center.circle center radius), 
  (no_obtuse_triangle points center) →
  Pr[(points)] = 3 / 32 :=
sorry

end obtuse_triangle_probability_l188_188467


namespace terminal_side_alpha_minus_beta_nonneg_x_axis_l188_188141

theorem terminal_side_alpha_minus_beta_nonneg_x_axis
  (α β : ℝ) (k : ℤ) (h : α = k * 360 + β) : 
  (∃ m : ℤ, α - β = m * 360) := 
sorry

end terminal_side_alpha_minus_beta_nonneg_x_axis_l188_188141


namespace opposite_exprs_have_value_l188_188716

theorem opposite_exprs_have_value (x : ℝ) : (4 * x - 8 = -(3 * x - 6)) → x = 2 :=
by
  intro h
  sorry

end opposite_exprs_have_value_l188_188716


namespace area_of_parallelogram_l188_188451

def vector_u : fin 3 → ℝ := ![2, 4, -1]
def vector_v : fin 3 → ℝ := ![5, -2, 3]

def cross_product (u v : fin 3 → ℝ) : fin 3 → ℝ :=
![u 1 * v 2 - u 2 * v 1,
  u 2 * v 0 - u 0 * v 2,
  u 0 * v 1 - u 1 * v 0]

noncomputable def vector_magnitude (v : fin 3 → ℝ) : ℝ :=
real.sqrt (v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2)

theorem area_of_parallelogram :
  vector_magnitude (cross_product vector_u vector_v) = real.sqrt 797 :=
by sorry

end area_of_parallelogram_l188_188451


namespace floor_add_self_eq_14_5_iff_r_eq_7_5_l188_188098

theorem floor_add_self_eq_14_5_iff_r_eq_7_5 (r : ℝ) : 
  (⌊r⌋ + r = 14.5) ↔ r = 7.5 :=
by
  sorry

end floor_add_self_eq_14_5_iff_r_eq_7_5_l188_188098


namespace part1_part2_part3_l188_188252

-- Define the set A and the operation ⊙
def A := (ℝ × ℝ)

def op (α β : A) : A :=
  (α.1 * β.2 + α.2 * β.1, α.2 * β.2 - α.1 * β.1)

-- The three parts of our problem:
-- Part I: Calculation
theorem part1 : op (2, 3) (-1, 4) = (5, 14) := 
  sorry

-- Part II: Commutative Law
theorem part2 (α β : A) : op α β = op β α := 
  sorry

-- Part III: Identity Element
theorem part3 : ∃ I : A, ∀ α : A, op I α = α := 
  exists.intro (0, 1) sorry

end part1_part2_part3_l188_188252


namespace range_of_a_l188_188924

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, a * (x - 2) * Real.exp x + Real.log x + 1 / x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Ioo 0 2, ∃ c ∈ Ioo 0 2, (deriv (f a) c = 0)) ↔
    (a ∈ Ioo (-∞) (-1 / Real.exp 1) ∪ Ioo (-1 / Real.exp 1) (-1 / (4 * Real.exp 2))) :=
by
  sorry

end range_of_a_l188_188924


namespace quadratic_roots_properties_l188_188833

-- Given the quadratic equation x^2 - 7x + 12 = 0
-- Prove that the absolute value of the difference of the roots is 1
-- Prove that the maximum value of the roots is 4

theorem quadratic_roots_properties :
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → abs (r1 - r2) = 1) ∧ 
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → max r1 r2 = 4) :=
by sorry

end quadratic_roots_properties_l188_188833


namespace cost_of_largest_pot_is_2_52_l188_188999

/-
Mark bought a set of 6 flower pots of different sizes at a total pre-tax cost.
Each pot cost 0.4 more than the next one below it in size.
The total cost, including a sales tax of 7.5%, was $9.80.
Prove that the cost of the largest pot before sales tax was $2.52.
-/

def cost_smallest_pot (x : ℝ) : Prop :=
  let total_cost := x + (x + 0.4) + (x + 0.8) + (x + 1.2) + (x + 1.6) + (x + 2.0)
  let pre_tax_cost := total_cost / 1.075
  let pre_tax_total_cost := (9.80 / 1.075)
  (total_cost = 6 * x + 6 ∧ total_cost = pre_tax_total_cost) →
  (x + 2.0 = 2.52)

theorem cost_of_largest_pot_is_2_52 :
  ∃ x : ℝ, cost_smallest_pot x :=
sorry

end cost_of_largest_pot_is_2_52_l188_188999


namespace positive_integers_count_l188_188112

theorem positive_integers_count (S_n : ℕ → ℕ) 
  (h : ∀ n : ℕ, S_n n = n * (n + 1) / 2) :
  { n : ℕ | S_n n ∣ 10 * n }.to_finset.card = 5 :=
by
  sorry

end positive_integers_count_l188_188112


namespace proof_t_value_proof_side_sum_value_l188_188891

open Real

-- Define the problem and the derived conditions
def function_f (x : ℝ) (t : ℝ) : ℝ :=
  2 * cos x * sin (x - π / 3) + t

noncomputable def t_value : ℝ :=
  sqrt 3 / 2

def triangle_area (b c A : ℝ) : ℝ :=
  1 / 2 * b * c * sin A

axiom a_side : ℝ := 2 * sqrt 2
axiom area_triangle : ℝ := sqrt 3

def maximum_condition (t : ℝ) : Prop :=
  ∀ x : ℝ, function_f x t ≤ 1

def side_sum_condition (a b c : ℝ) (area : ℝ) (f_A : ℝ) : Prop :=
  (a = 2 * sqrt 2) ∧ (area = sqrt 3) ∧ (f_A = sqrt 3 / 2) 

noncomputable def b_side (b c A : ℝ) : ℝ :=
  begin
    -- Since A = π / 3, sin A = sqrt 3 / 2
    let A := π / 3,
    -- Given, Area of Triangle is sqrt 3
    -- bc * sqrt 3 / 2 = sqrt 3 -> bc = 4
    -- Using Law of Cosines: 8 = (b+c)^2 - 3bc -> (b+c)^2 - 12 = 8 -> b+c = 2sqrt 5
    let b_c_sum := 2 * sqrt 5,
    exact b_c_sum
  end

theorem proof_t_value :
  ∀ t, maximum_condition t → t = t_value := sorry

theorem proof_side_sum_value (b c A : ℝ) :
  side_sum_condition a_side b c area_triangle (function_f A t_value) → b_side b c A = 2 * sqrt 5 :=
begin
  sorry
end

end proof_t_value_proof_side_sum_value_l188_188891


namespace ratio_of_patients_l188_188385

def one_in_four_zx (current_patients : ℕ) : ℕ :=
  current_patients / 4

def previous_patients : ℕ :=
  26

def diagnosed_patients : ℕ :=
  13

def current_patients : ℕ :=
  diagnosed_patients * 4

theorem ratio_of_patients : 
  one_in_four_zx current_patients = diagnosed_patients → 
  (current_patients / previous_patients) = 2 := 
by 
  sorry

end ratio_of_patients_l188_188385


namespace function_above_x_axis_l188_188894

theorem function_above_x_axis (m : ℝ) : 
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) ↔ m < 2 + 2 * Real.sqrt 2 :=
sorry

end function_above_x_axis_l188_188894


namespace units_digit_F_F_15_l188_188672

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem units_digit_F_F_15 : (fib (fib 15) % 10) = 5 := by
  sorry

end units_digit_F_F_15_l188_188672


namespace non_intersecting_polygon_possible_l188_188277

-- Definition of the problem statement and required conditions
def non_collinear (p1 p2 p3 : Point) : Prop :=
  ¬ collinear p1 p2 p3

def non_intersecting_polygon_exists (points : Finset Point) : Prop :=
  ∃ (polygon : Polygon), 
    (polygon.vertices = points) ∧ 
    (polygon.sides_non_intersecting)

theorem non_intersecting_polygon_possible (n : ℕ) (points : Finset Point) 
  (h1 : points.card = n) 
  (h2 : ∀ (p1 p2 p3 : Point), {p1, p2, p3} ⊆ points → non_collinear p1 p2 p3) : 
  non_intersecting_polygon_exists points := 
sorry

end non_intersecting_polygon_possible_l188_188277


namespace B_alone_completion_l188_188379

-- Define the conditions:
def A_efficiency_rel_to_B (A B: ℕ → Prop) : Prop :=
  ∀ (x: ℕ), B x → A (2 * x)

def together_job_completion (A B: ℕ → Prop) : Prop :=
  ∀ (t: ℕ), t = 20 → (∃ (x y : ℕ), B x ∧ A y ∧ (1/x + 1/y = 1/t))

-- Define the theorem:
theorem B_alone_completion (A B: ℕ → Prop) (h1 : A_efficiency_rel_to_B A B) (h2 : together_job_completion A B) :
  ∃ (x: ℕ), B x ∧ x = 30 :=
sorry

end B_alone_completion_l188_188379


namespace red_triangle_perimeter_l188_188315

noncomputable def perimeter_of_equilateral_triangle (s : ℕ) : ℕ :=
  3 * s

theorem red_triangle_perimeter :
  let side_length := 23 in
  perimeter_of_equilateral_triangle side_length = 69 :=
by
  sorry

end red_triangle_perimeter_l188_188315


namespace cosine_angle_l188_188524

variables {a b : EuclideanSpace ℝ (Fin 2)}

noncomputable def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 1

theorem cosine_angle (h₁ : is_unit_vector a) (h₂ : is_unit_vector b) 
  (h : ‖2 • a - b‖ = Real.sqrt 2) : 
  Real.cos (angle a b) = 3 / 4 :=
sorry

end cosine_angle_l188_188524


namespace find_x_l188_188994

noncomputable def g1 (x : ℝ) : ℝ := (3 * x - 8) / (5 * x + 2)

noncomputable def g (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then g1 x else g1 (g (n - 1) x)

theorem find_x (x : ℝ) (h : g 2022 x = x - 4) : x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 :=
  sorry

end find_x_l188_188994


namespace distance_between_A_and_B_l188_188015

noncomputable def time_from_A_to_B (D : ℝ) : ℝ := D / 200

noncomputable def time_from_B_to_A (D : ℝ) : ℝ := time_from_A_to_B D + 3

def condition (D : ℝ) : Prop := 
  D = 100 * (time_from_B_to_A D)

theorem distance_between_A_and_B :
  ∃ D : ℝ, condition D ∧ D = 600 :=
by
  sorry

end distance_between_A_and_B_l188_188015


namespace number_of_x_satisfying_cos2x_plus_3sin2x_eq_1_l188_188914

theorem number_of_x_satisfying_cos2x_plus_3sin2x_eq_1 :
  let x := by sorry
  let interval := (-25 < x < 120)
  let eq := (cos x)^2 + 3 * (sin x)^2 = 1
  interval ∧ eq → ∃ n : ℕ, n = 46 := sorry

end number_of_x_satisfying_cos2x_plus_3sin2x_eq_1_l188_188914


namespace xyz_not_divisible_by_3_l188_188233

theorem xyz_not_divisible_by_3 (x y z : ℕ) (h1 : x % 2 = 1) (h2 : y % 2 = 1) (h3 : z % 2 = 1) 
  (h4 : Nat.gcd (Nat.gcd x y) z = 1) (h5 : (x^2 + y^2 + z^2) % (x + y + z) = 0) : 
  (x + y + z - 2) % 3 ≠ 0 :=
by
  sorry

end xyz_not_divisible_by_3_l188_188233


namespace years_until_5_years_before_anniversary_l188_188187

-- Definitions
def years_built_ago := 100
def upcoming_anniversary := 200
def years_before_anniversary := 5

-- Theorem statement
theorem years_until_5_years_before_anniversary :
  let years_until_anniversary := upcoming_anniversary - years_built_ago in
  let future_years := years_until_anniversary - years_before_anniversary in
  future_years = 95 :=
by
  sorry

end years_until_5_years_before_anniversary_l188_188187


namespace earrings_ratio_l188_188035

theorem earrings_ratio :
  ∃ (M R : ℕ), 10 = M / 4 ∧ 10 + M + R = 70 ∧ M / R = 2 := by
  sorry

end earrings_ratio_l188_188035


namespace exists_term_with_digit_9_l188_188301

noncomputable def arithmetic_progression_has_digit_9 (a d : ℕ) : Prop :=
  ∃ n : ℕ, (a + n * d).digits.contains 9

theorem exists_term_with_digit_9 (a d : ℕ) (h1 : 1 ≤ d) (h2 : d < 10) : 
  arithmetic_progression_has_digit_9 a d := 
by 
  sorry

end exists_term_with_digit_9_l188_188301


namespace stamps_total_l188_188688

theorem stamps_total (x : ℕ) (a_initial : ℕ := 5 * x) (b_initial : ℕ := 4 * x)
                     (a_after : ℕ := a_initial - 5) (b_after : ℕ := b_initial + 5)
                     (h_ratio_initial : a_initial / b_initial = 5 / 4)
                     (h_ratio_final : a_after / b_after = 4 / 5) :
                     a_initial + b_initial = 45 :=
by
  sorry

end stamps_total_l188_188688


namespace differential_eq_l188_188838

variable (x : ℝ)

noncomputable def y : ℝ := Real.log (Real.cos x ^ 2 + (1 + Real.cos x ^ 4).sqrt)
def dy_dx : ℝ := -((2 * Real.cos x * Real.sin x) / (1 + Real.cos x ^ 4).sqrt)

theorem differential_eq :
  ∀ (dx : ℝ), dy_dx * dx = -((Real.sin (2 * x) * dx) / (1 + Real.cos x ^ 4).sqrt) := by
  sorry

end differential_eq_l188_188838


namespace skyscraper_anniversary_l188_188193

theorem skyscraper_anniversary 
  (years_since_built : ℕ)
  (target_years : ℕ)
  (years_before_200th : ℕ)
  (years_future : ℕ) 
  (h1 : years_since_built = 100) 
  (h2 : target_years = 200 - 5) 
  (h3 : years_future = target_years - years_since_built) : 
  years_future = 95 :=
by
  sorry

end skyscraper_anniversary_l188_188193


namespace at_least_one_of_triplet_can_be_sum_of_two_primes_l188_188424

-- Goldbach's conjecture for even integers greater than 2
def goldbach_conjecture (n : ℕ) : Prop :=
  ∃ p q : ℕ, prime p ∧ prime q ∧ n = p + q

-- Definition for triplet of consecutive even numbers greater than 2
def triplet_consecutive_even_gt_2 (n : ℕ) : Prop :=
  n > 1 ∧ goldbach_conjecture (2 * n) ∧ goldbach_conjecture (2 * n + 2) ∧ goldbach_conjecture (2 * n + 4)

-- The theorem to prove:
theorem at_least_one_of_triplet_can_be_sum_of_two_primes (n : ℕ) :
  triplet_consecutive_even_gt_2 n →
  (goldbach_conjecture (2 * n) ∨ goldbach_conjecture (2 * n + 2) ∨ goldbach_conjecture (2 * n + 4)) :=
by sorry

end at_least_one_of_triplet_can_be_sum_of_two_primes_l188_188424


namespace obtuse_triangle_probability_l188_188478

/-- 
Given four points chosen uniformly at random on a circle,
prove that the probability that no three of these points form an obtuse triangle with the circle's center is 3/32.
-/
theorem obtuse_triangle_probability : 
  let points := 4 in
  ∀ (P : Fin points → ℝ × ℝ) (random_uniform : ∀ i, random.uniform_on_circle i),
  (probability (no_three_form_obtuse_triangle P) = 3 / 32) :=
sorry

end obtuse_triangle_probability_l188_188478


namespace sqrt_floor_eq_sqrt_sqrt_floor_l188_188961

theorem sqrt_floor_eq_sqrt_sqrt_floor (a : ℝ) (h : a > 1) :
  Int.floor (Real.sqrt (Int.floor (Real.sqrt a))) = Int.floor (Real.sqrt (Real.sqrt a)) :=
sorry

end sqrt_floor_eq_sqrt_sqrt_floor_l188_188961


namespace nilpotent_sum_comm_noncomm_nilpotent_example_l188_188368

section part_a

variable {R : Type*} [CommRing R] (u v : R)
variable (p q : ℕ) (hu : u^p = 0) (hv : v^q = 0)

theorem nilpotent_sum_comm {u v : R} (h1 : IsNilpotent u) (h2 : IsNilpotent v) : IsNilpotent (u + v) :=
by sorry

end part_a

section part_b

variable {R : Type*} [NoncommRing R] 

def u := !!(0 1)

def v := !!(1 0)

theorem noncomm_nilpotent_example : ¬ IsNilpotent (u + v) :=
by sorry

end part_b

end nilpotent_sum_comm_noncomm_nilpotent_example_l188_188368


namespace seafood_regular_price_l188_188768

theorem seafood_regular_price (y : ℝ) (h : y / 4 = 4) : 2 * y = 32 := by
  sorry

end seafood_regular_price_l188_188768


namespace nested_sqrt_solution_l188_188075

noncomputable def nested_sqrt (x : ℝ) : ℝ := sqrt (20 + x)

theorem nested_sqrt_solution (x : ℝ) : nonneg_real x →
  (x = nested_sqrt x ↔ x = 5) :=
begin
  sorry
end

end nested_sqrt_solution_l188_188075


namespace nested_sqrt_solution_l188_188074

noncomputable def nested_sqrt (x : ℝ) : ℝ := sqrt (20 + x)

theorem nested_sqrt_solution (x : ℝ) : nonneg_real x →
  (x = nested_sqrt x ↔ x = 5) :=
begin
  sorry
end

end nested_sqrt_solution_l188_188074


namespace price_range_of_book_l188_188726

variable (x : ℝ)

theorem price_range_of_book (h₁ : ¬(x ≥ 15)) (h₂ : ¬(x ≤ 12)) (h₃ : ¬(x ≤ 10)) : 12 < x ∧ x < 15 := 
by
  sorry

end price_range_of_book_l188_188726


namespace almonds_received_by_amanda_l188_188653

variable (totalAlmonds : ℚ)
variable (numberOfPiles : ℚ)
variable (pilesForAmanda : ℚ)

-- Conditions
def stephanie_has_almonds := totalAlmonds = 66 / 7
def distribute_equally_into_piles := numberOfPiles = 6
def amanda_receives_piles := pilesForAmanda = 3

-- Conclusion to prove
theorem almonds_received_by_amanda :
  stephanie_has_almonds totalAlmonds →
  distribute_equally_into_piles numberOfPiles →
  amanda_receives_piles pilesForAmanda →
  (totalAlmonds / numberOfPiles) * pilesForAmanda = 33 / 7 :=
by
  sorry

end almonds_received_by_amanda_l188_188653


namespace units_digit_of_FF15_is_5_l188_188660

noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem units_digit_of_FF15_is_5 : (fib (fib 15)) % 10 = 5 :=
by
  sorry

end units_digit_of_FF15_is_5_l188_188660


namespace geometric_series_sum_l188_188414

theorem geometric_series_sum :
  let a := (3 : ℚ) / 4
  let r := (3 : ℚ) / 4
  let n := 15
  ∑ i in finset.range n, a * r ^ i = (3177884751 : ℚ) / 1073741824 := sorry

end geometric_series_sum_l188_188414


namespace log_simplification_l188_188343

theorem log_simplification :
  (log 2 (3 * log 2 256))^2 = 9 + 6 * log 2 3 + (log 2 3)^2 :=
by
  -- Define the property of logarithm with respect to its base
  have log_base_property : ∀ {b x : ℝ}, b > 0 → b ≠ 1 → log b (b ^ x) = x := sorry,
  -- Define that 256 equals 2^8
  have two_pow_eight : 256 = 2 ^ 8 := sorry,
  -- Define the logarithm property of product
  have log_product_property : ∀ {a b : ℝ}, log 2 (a * b) = log 2 a + log 2 b := sorry,
  -- Now, combine these properties to show the required equivalence
  sorry

end log_simplification_l188_188343


namespace sum_inequality_l188_188239

variable {α : Type} [LinearOrderedField α] 

theorem sum_inequality 
  (n : ℕ) 
  (α : Fin n → α) 
  (h1 : ∀ i, 0 < α i) 
  (h2 : ∑ i in Finset.univ, α i = 1) 
  (h3 : 2 ≤ n) : 
  ∑ i in Finset.univ, α i / (2 - α i) ≥ n / (2 * n - 1) :=
sorry

end sum_inequality_l188_188239


namespace same_solution_set_l188_188718

theorem same_solution_set :
  (∀ x : ℝ, (x - 1) / (x - 2) ≤ 0 ↔ (x^3 - x^2 + x - 1) / (x - 2) ≤ 0) :=
sorry

end same_solution_set_l188_188718


namespace alice_after_bob_implies_bob_before_345_l188_188022

open Set

def prob_Alice_after_Bob_Bob_before_345 : ℚ := 9 / 16

theorem alice_after_bob_implies_bob_before_345 : 
  ∀ (a b : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ a ≤ 60 ∧ b ≤ 60 ∧ a > b → (b < 45) = (9 / 16) :=
sorry

end alice_after_bob_implies_bob_before_345_l188_188022


namespace least_possible_b_l188_188290

theorem least_possible_b (a b : ℕ) (h1 : ∃ p : ℕ, nat.prime p ∧ a = p^2) 
    (h2 : ∀ d, d ∣ b → nat.count_factors d = a) 
    (h3 : b % a = 0) :
    b = 8 :=
by sorry

end least_possible_b_l188_188290


namespace cubed_z_is_i_l188_188919

def z : ℂ := complex.sin (real.pi / 3) + complex.I * complex.cos (real.pi / 3)

theorem cubed_z_is_i : z ^ 3 = complex.I := by
  sorry

end cubed_z_is_i_l188_188919


namespace path_length_traversed_by_P_l188_188282

theorem path_length_traversed_by_P :
  ∀ (A B P X Y Z : Point) (s : Square),
  (right_angle ∠B) →
  (length AB = 2) →
  (length BP = 4) →
  (square AXYZ ∧ side_length AXYZ = 6 ∧ B ∈ AX) →
  rotating_triangle (A B P) (B P A) (P A B) (A B P) →
  total_path_length P = 24 * π :=
by
  -- Define specific points
  intros A B P X Y Z s
  assume right_angle_B
  assume AB_2
  assume BP_4
  assume square_6 B_in_AX
  assume rotating_pattern
  sorry -- The proof steps would follow from here

end path_length_traversed_by_P_l188_188282


namespace quadratic_roots_sum_l188_188881

theorem quadratic_roots_sum
  (m n : ℝ)
  (h_mn_roots : (Polynomial.X ^ 2 - 8 * Polynomial.X + 5 : Polynomial ℝ).is_root m ∧
                 (Polynomial.X ^ 2 - 8 * Polynomial.X + 5 : Polynomial ℝ).is_root n) :
  (1 / (m - 1)) + (1 / (n - 1)) = -3 := by
  sorry

end quadratic_roots_sum_l188_188881


namespace probability_A_or_B_at_end_l188_188698

theorem probability_A_or_B_at_end 
  (students : Fin 4 → String)
  (h1 : (students 0 = "A") ∨ (students 3 = "A") ∨ (students 0 = "B") ∨ (students 3 = "B")) :
  4! = 24 → 2! * 2! = 4 → 
  let total_arrangements := 24 in
  let middle_arrangements := 4 in
  let favorable_arrangements := total_arrangements - middle_arrangements in
  let probability := favorable_arrangements / total_arrangements in
  probability = 5 / 6 :=
sorry

end probability_A_or_B_at_end_l188_188698


namespace combined_resistance_parallel_l188_188213

theorem combined_resistance_parallel (x y r : ℝ) (hx : x = 3) (hy : y = 5) :
  (1 / r) = (1 / x) + (1 / y) → r = 15 / 8 :=
by
  intros h
  rw [hx, hy] at h
  have h1 : (1 / r) = (1 / 3) + (1 / 5) := h
  have h2 : (1 / r) = 5 / 15 + 3 / 15 := by
     rw [←one_div, ←one_div]
     norm_num
  rw [h1, h2]
  norm_num
  sorry

end combined_resistance_parallel_l188_188213


namespace range_of_x2_plus_4y2_l188_188513

theorem range_of_x2_plus_4y2 (x y : ℝ) (h : 4 * x ^ 2 - 2 * real.sqrt 3 * x * y + 4 * y ^ 2 = 13) :
  10 - 4 * real.sqrt 3 ≤ x ^ 2 + 4 * y ^ 2 ∧ x ^ 2 + 4 * y ^ 2 ≤ 10 + 4 * real.sqrt 3 :=
sorry

end range_of_x2_plus_4y2_l188_188513


namespace geometry_table_correct_l188_188339

-- Define properties as propositions
def acute_angle : Prop := sorry
def equal_sides : Prop := sorry
def unequal_sides : Prop := sorry
def right_angle : Prop := sorry

-- Define shapes and their properties
def scalene_triangle : Prop := acute_angle ∧ ¬equal_sides ∧ ¬right_angle
def isosceles_right_triangle : Prop := acute_angle ∧ equal_sides ∧ right_angle
def rectangle : Prop := ¬acute_angle ∧ ¬equal_sides ∧ right_angle
def parallelogram : Prop := acute_angle ∧ ¬equal_sides ∧ ¬right_angle

theorem geometry_table_correct :
  let table := [[acute_angle, ¬equal_sides, unequal_sides, ¬right_angle],
                [acute_angle, equal_sides, ¬unequal_sides, right_angle],
                [¬acute_angle, ¬equal_sides, unequal_sides, right_angle],
                [acute_angle, ¬equal_sides, unequal_sides, ¬right_angle]] in
  (∀ row, (∃! col, table[row] col = false)) ∧
  (∀ col, (∃! row, table[row] col = false)) :=
sorry

end geometry_table_correct_l188_188339


namespace average_percentage_increase_l188_188600

theorem average_percentage_increase (initial_A initial_B initial_C new_A new_B new_C : ℝ)
  (hA : initial_A = 50) (hA' : new_A = 75)
  (hB : initial_B = 100) (hB' : new_B = 120)
  (hC : initial_C = 80) (hC' : new_C = 88) :
  ((new_A - initial_A) / initial_A * 100 + (new_B - initial_B) / initial_B * 100 +
   (new_C - initial_C) / initial_C * 100) / 3 = 26.67 := 
by
  calc sorry

end average_percentage_increase_l188_188600


namespace find_n_l188_188108

theorem find_n (n : ℕ) (h : n = (nat.factorial 500)^2) : 
  ∃ (m : ℕ), m = (nat.factorial 500)^2 ∧ 
  (n < (nat.factorial 500) ∧ 
  ∃ (k : ℕ), k = n + 1) := by
  sorry

end find_n_l188_188108


namespace evaluate_expression_l188_188285

-- Definitions for a and b
def a : Int := 1
def b : Int := -1

theorem evaluate_expression : 
  5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b) + 1 = -17 := by
  -- Simplification steps skipped
  sorry

end evaluate_expression_l188_188285


namespace compute_expr_l188_188559

noncomputable def c : ℝ := Real.log 25
noncomputable def d : ℝ := Real.log 36

theorem compute_expr : 5^(c/d) + 6^(d/c) = 11 := 
by 
  sorry

end compute_expr_l188_188559


namespace market_survey_l188_188933

theorem market_survey (X Y : ℕ) (h1 : X / Y = 9) (h2 : X + Y = 400) : X = 360 :=
by
  sorry

end market_survey_l188_188933


namespace triangle_to_polygons_l188_188867

-- Define what it means for a polygon to be an n-gon
def is_ngon (n : ℕ) (P : set Point) : Prop :=
  P.card = n

-- Define the statement: A triangle can be divided into a 2020-gon and a 2021-gon.
theorem triangle_to_polygons (T : set Point) (hT : is_triangle T):
  ∃ P Q : set Point, is_ngon 2020 P ∧ is_ngon 2021 Q ∧ (P ∪ Q = T) :=
sorry

end triangle_to_polygons_l188_188867


namespace rectangle_length_l188_188694

theorem rectangle_length
  (W : ℝ) (side_length_pentagon : ℝ)
  (P_pentagon : ℝ)
  (L : ℝ)
  (h1 : W = 20) 
  (h2 : side_length_pentagon = 10)
  (h3 : P_pentagon = 5 * side_length_pentagon)
  (h4 : 2 * L + 2 * W = P_pentagon) : 
  L = 5 :=
by
  rw [h2] at h3
  rw [h3] at h4
  rw [h1] at h4
  simp at h4
  exact eq_of_sub_eq_zero (eq_of_add_eq_add_right h4)

end rectangle_length_l188_188694


namespace find_side_length_l188_188631

theorem find_side_length (M E : ℝ × ℝ) (s : ℝ) (h1 : M.1 = s / 4 ∧ M.2 = 0)
  (h2 : E.2 ≥ 0)
  (h3 : let CM := ∥(M.1, M.2) - (0, 0)∥ in let MD := ∥(s, 0) - (M.1, M.2)∥
        in CM / MD = 1 / 3)
  (h4 : let AE := ∥(s, s) - (E.1, E.2)∥ in let CE := ∥(0, 0) - (E.1, E.2)∥
        in s^2 * AE * CE = 7 * 56)
  : s = 10 := sorry

end find_side_length_l188_188631


namespace units_digit_of_FF15_is_5_l188_188661

noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem units_digit_of_FF15_is_5 : (fib (fib 15)) % 10 = 5 :=
by
  sorry

end units_digit_of_FF15_is_5_l188_188661


namespace find_a_b_l188_188287

theorem find_a_b
  (f g h k : ℝ → ℝ)
  (a b : ℝ)
  (H1 : ∀ x, f x = (x - 1) * g x + 3)
  (H2 : ∀ x, f x = (x + 1) * h x + 1)
  (H3 : ∀ x, f x = (x^2 - 1) * k x + a * x + b) :
  a = 1 ∧ b = 2 :=
by
  let x := arbitrary ℝ
  sorry

end find_a_b_l188_188287


namespace find_polynomial_l188_188096

open Polynomial

theorem find_polynomial (P : ℝ[X]) (h : ∀ x : ℝ, eval (2*x) P = eval x (P.derivative) * eval x (P.derivative.derivative)) :
  P = C (4 / 9) * X^3 :=
by 
  sorry

end find_polynomial_l188_188096


namespace dice_arithmetic_progression_probability_l188_188466

theorem dice_arithmetic_progression_probability :
  (∃ f : Fin 4 → Fin 6, 
    let s := multiset.map (λ i, (f i : ℕ)) multiset.univ in 
      (∃ d ∈ ({1, 2} : set ℕ), 
         ∃ a, multiset.card (multiset.filter (λ x, ∃ n, x = a + n * d) s) = 4)) → 
  (108 / 1296 = 1 / 12) := 
by 
suffices ⊢ 108 = 1 / 12 * 1296 sorry

end dice_arithmetic_progression_probability_l188_188466


namespace shells_not_red_or_green_l188_188399

theorem shells_not_red_or_green (total_shells : ℕ) (red_shells : ℕ) (green_shells : ℕ) 
  (h_total : total_shells = 291) (h_red : red_shells = 76) (h_green : green_shells = 49) :
  total_shells - (red_shells + green_shells) = 166 :=
by
  sorry

end shells_not_red_or_green_l188_188399


namespace problem_1_problem_2_l188_188535

-- First problem: Find the solution set for the inequality |x - 1| + |x + 2| ≥ 5
theorem problem_1 (x : ℝ) : (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) :=
sorry

-- Second problem: Find the range of real number a such that |x - a| + |x + 2| ≤ |x + 4| for all x in [0, 1]
theorem problem_2 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x - a| + |x + 2| ≤ |x + 4|) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end problem_1_problem_2_l188_188535


namespace min_value_of_expr_l188_188906

noncomputable section
open Real

theorem min_value_of_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (hlog : log 2 x + log 2 y = 1) :
  ∃ b, b = 4 ∧ (∀ z, z = (x^2 + y^2) / (x - y) → z ≥ b) :=
by
  sorry

end min_value_of_expr_l188_188906


namespace exist_infinitely_many_coprime_pairs_l188_188279

theorem exist_infinitely_many_coprime_pairs (a b : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : Nat.gcd a b = 1) : 
  ∃ (a b : ℕ), (a + b).mod (a^b + b^a) = 0 :=
sorry

end exist_infinitely_many_coprime_pairs_l188_188279


namespace nested_sqrt_eq_five_l188_188065

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l188_188065


namespace min_number_to_remove_l188_188115

theorem min_number_to_remove (S : Set ℕ) (hs : S = {1, 2, 3, ..., 1982}) :
  ∃ R ⊆ S, (∀ x y ∈ (S \ R), x * y ∉ (S \ R)) ∧ |R| = 43 := 
sorry

end min_number_to_remove_l188_188115


namespace length_of_square_side_l188_188728

theorem length_of_square_side (length_of_string : ℝ) (num_sides : ℕ) (total_side_length : ℝ) 
  (h1 : length_of_string = 32) (h2 : num_sides = 4) (h3 : total_side_length = length_of_string) : 
  total_side_length / num_sides = 8 :=
by
  sorry

end length_of_square_side_l188_188728


namespace compare_abc_l188_188495

noncomputable def a : ℝ := 3^0.4
noncomputable def b : ℝ := 0.4^3
noncomputable def c : ℝ := Real.logBase 0.4 3

theorem compare_abc : c < b ∧ b < a := by
  sorry

end compare_abc_l188_188495


namespace number_of_mixed_vegetable_plates_l188_188402

theorem number_of_mixed_vegetable_plates :
  ∃ n : ℕ, n * 70 = 1051 - (16 * 6 + 5 * 45 + 6 * 40) ∧ n = 7 :=
by
  sorry

end number_of_mixed_vegetable_plates_l188_188402


namespace infinite_nested_sqrt_l188_188067

theorem infinite_nested_sqrt :
  let x := \sqrt{20 + \sqrt{20 + \sqrt{20 + \sqrt{20 + \cdots}}}} in
  x = 5 :=
begin
  let x : ℝ := sqrt(20 + sqrt(20 + sqrt(20 + sqrt(20 + ...)))),
  have h1 : x = sqrt(20 + x), from sorry,
  have h2 : x^2 = 20 + x, from sorry,
  have h3 : x^2 - x - 20 = 0, from sorry,
  have h4 : (x - 5) * (x + 4) = 0, from sorry,
  have h5 : x = 5 ∨ x = -4, from sorry,
  have h6 : x >= 0, from sorry,
  exact h5.elim (λ h, h) (λ h, (h6.elim_left h))
end

end infinite_nested_sqrt_l188_188067


namespace cube_painting_l188_188396

theorem cube_painting (n : ℕ) (h : n > 2) :
  (6 * (n - 2)^2 = (n - 2)^3) ↔ (n = 8) :=
by
  sorry

end cube_painting_l188_188396


namespace sum_of_segments_constant_l188_188851

variables {P V H A1 B1 C1 D1 : Type} 

-- Definitions of the points and segments, assuming they form a regular pyramid and the points are in the correct locations
variables (ABCD : Type) [IsRegularPyramid ABCD V] {P : ABCD}

-- The main theorem
theorem sum_of_segments_constant (P : ABCD) (h1: arbitrary_point P ABCD):
  ∃ k : ℝ, PA1 + PB1 + PC1 + PD1 = k :=
  sorry

end sum_of_segments_constant_l188_188851


namespace number_of_true_propositions_l188_188794

theorem number_of_true_propositions :
  ¬(∀ a b : ℝ, a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ log (a + b) = log a + log b) ∧
  (∃ n : ℤ, odd n ∧ ¬ prime n) ∧
  ¬(∀ (A B : ℝ) (h : 0 < A ∧ A < B ∧ B < π),
      sin A < sin B) → 2 := sorry

end number_of_true_propositions_l188_188794


namespace triangle_OAM_area_l188_188004

theorem triangle_OAM_area 
  (h_parabola : ∀ x y : ℝ, x^2 = 4 * y ↔ (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2})
  (h_focus : (0, 1) ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2})
  (h_M : (1, 0) ∈ {p : ℝ × ℝ})
  (h_O : (0, 0) ∈ {p : ℝ × ℝ}) :
  let A := classical.some (exists_intersection_of_line_and_parabola (0, 1) (1, 0) h_parabola h_focus h_M) in
  abs (1 / 2 * ((0 * (A.2 - 0) - ((A.1) - 0) * 0) - (1 * (A.2 - 0)))) = 1.5 - sqrt 2 :=
sorry

-- Helper theorem to express the existence of an intersection point A
lemma exists_intersection_of_line_and_parabola
  (F M : ℝ × ℝ)
  (h_parabola : ∀ x y, x ^ 2 = 4 * y ↔ (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2})
  (h_F : F ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2})
  (h_M : M ∈ {p : ℝ × ℝ}) :
  ∃ A : ℝ × ℝ, A ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2} ∧ is_intersection_of_line_and_parabola F M A
:= sorry

-- Helper definition to express the concept of intersection of a line and a parabola
def is_intersection_of_line_and_parabola (F M A : ℝ × ℝ) : Prop :=
∃ k b : ℝ, (F.1, F.2) ∈ line_k_b k b ∧ (M.1, M.2) ∈ line_k_b k b ∧ (-A.1 / k + b = A.2)

end triangle_OAM_area_l188_188004


namespace locus_of_Q_is_ellipse_l188_188132

noncomputable def circle_E (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16
def point_F : ℝ × ℝ := (1, 0)
def is_point_on_circle_E (P : ℝ × ℝ) : Prop := circle_E P.1 P.2
def perpendicular_bisector_intersects_radius_at_Q (P Q : ℝ × ℝ) : Prop := 
  ∃ M, ((M.1 - P.1)^2 + (M.2 - P.2)^2 = (M.1 - Q.1)^2 + (M.2 - Q.2)^2) ∧ 
       ((M.1 + 1)^2 + M.2^2 = 16)

theorem locus_of_Q_is_ellipse :
  (∃ Q : ℝ × ℝ, (∃ P : ℝ × ℝ, is_point_on_circle_E P ∧ perpendicular_bisector_intersects_radius_at_Q P Q)) →
  ∀ Q : ℝ × ℝ, (Q.1^2 / 4 + Q.2^2 / 3 = 1) :=
by
  sorry

end locus_of_Q_is_ellipse_l188_188132


namespace value_of_a_purely_imaginary_l188_188921

-- Define the conditions under which the complex number is purely imaginary
def isPurelyImaginary (a : ℝ) : Prop :=
  (a^2 + a - 2 = 0) ∧ (a^2 - 3a + 2 ≠ 0)

-- The main theorem to be proven
theorem value_of_a_purely_imaginary : ∀ a : ℝ, isPurelyImaginary a → a = -2 :=
sorry

end value_of_a_purely_imaginary_l188_188921


namespace sector_area_l188_188144

theorem sector_area (α l : ℝ) (hα : α = Real.pi / 6) (hl : l = Real.pi / 3) : 
  let r := l / α in let s := (1 / 2) * l * r in s = Real.pi / 3 := 
by
  sorry

end sector_area_l188_188144


namespace lindsey_squat_weight_l188_188258

theorem lindsey_squat_weight :
  let num_bands := 2 in
  let resistance_per_band := 5 in
  let total_resistance := num_bands * resistance_per_band in
  let dumbbell_weight := 10 in
  total_resistance + dumbbell_weight = 20 :=
by
  sorry

end lindsey_squat_weight_l188_188258


namespace range_of_k_l188_188892

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * Math.sin (ω * x + ϕ)

theorem range_of_k 
  (ω : ℝ) (ϕ : ℝ)
  (h1 : -π < ϕ) 
  (h2 : ϕ < 0) 
  (h3 : ω > 0)
  (h4 : ∀ x ∈ set.Icc (0 : ℝ) (π / 2), ∃ k : ℝ, f x ω ϕ + real.log k / real.log 2 = 0) :
  ∀ k, k ∈ set.Icc (1 / 2) 4 :=
sorry

end range_of_k_l188_188892


namespace maximize_product_of_geometric_sequence_l188_188932

def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a1 * q^n

def product_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∏ k in Finset.range n.succ, a k

theorem maximize_product_of_geometric_sequence :
  ∃ n ∈ {9, 11, 12, 13}, 
    ∀ m ∈ {9, 11, 12, 13}, 
      product_of_first_n_terms (λ n : ℕ, 1536 * (-1/2)^n) n ≥ 
      product_of_first_n_terms (λ n : ℕ, 1536 * (-1/2)^n) m := 
by
  sorry

end maximize_product_of_geometric_sequence_l188_188932


namespace wire_cut_min_area_l188_188856

theorem wire_cut_min_area :
  ∃ x : ℝ, 0 < x ∧ x < 100 ∧ S = π * (x / (2 * π))^2 + ((100 - x) / 4)^2 ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 100 → (π * (y / (2 * π))^2 + ((100 - y) / 4)^2 ≥ S)) ∧
  x = 100 * π / (16 + π) :=
sorry

end wire_cut_min_area_l188_188856


namespace sum_of_specific_primes_l188_188820

open Nat

theorem sum_of_specific_primes :
  let primes := [2, 3, 5] in
  (∀ p, p ∈ primes ↔ p.prime ∧ (gcd 40 p ≠ 1 ∨ gcd 30 p ≠ 1)) →
  ∑ p in primes, p = 10 :=
by
  sorry

end sum_of_specific_primes_l188_188820


namespace mrs_taylor_paid_l188_188267

theorem mrs_taylor_paid :
  let cost_per_tv := 650
  let num_tvs := 2
  let discount_rate := 0.25
  let total_cost_without_discount := num_tvs * cost_per_tv
  let discount_amount := discount_rate * total_cost_without_discount
  let final_price := total_cost_without_discount - discount_amount
  final_price = 975 :=
by
  let cost_per_tv := 650
  let num_tvs := 2
  let discount_rate := 0.25
  let total_cost_without_discount := num_tvs * cost_per_tv
  let discount_amount := discount_rate * total_cost_without_discount
  let final_price := total_cost_without_discount - discount_amount
  show final_price = 975 from sorry

end mrs_taylor_paid_l188_188267


namespace math_competition_l188_188582

theorem math_competition:
  ∀ (j k : ℕ), 
    j ≤ 30 → k ≤ 30 → j + k ≤ 30 →
    ∃ S : ℤ, 
      S = 4 * j - k ∧
      S ∈ {S_min : ℤ | S_min = -30} ∪ {S_max : ℤ | S_max = 120} ∪ 
        {S_range : ℤ | ∀ j_val ∈ ({0, 1, ..., 30} : set ℕ), 
            let min := 5 * j_val - 30, max := 4 * j_val in S_range ∈ (min..max).toSet} ∧
      {S_problem : ℤ | S_problem ∈ (insert -30 (finset.range 151)) \ {109, 113, 114, 117, 118, 119}}.card = 145 := 
by
  sorry

end math_competition_l188_188582


namespace nested_sqrt_solution_l188_188073

noncomputable def nested_sqrt (x : ℝ) : ℝ := sqrt (20 + x)

theorem nested_sqrt_solution (x : ℝ) : nonneg_real x →
  (x = nested_sqrt x ↔ x = 5) :=
begin
  sorry
end

end nested_sqrt_solution_l188_188073


namespace units_digit_of_fibonacci_f_15_l188_188669

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n + 1) + fibonacci n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fibonacci_f_15 :
  units_digit (fibonacci (fibonacci 15)) = 5 := 
sorry

end units_digit_of_fibonacci_f_15_l188_188669


namespace water_fee_part1_water_fee_part2_water_fee_usage_l188_188809

theorem water_fee_part1 (x : ℕ) (h : 0 < x ∧ x ≤ 6) : y = 2 * x :=
sorry

theorem water_fee_part2 (x : ℕ) (h : x > 6) : y = 3 * x - 6 :=
sorry

theorem water_fee_usage (y : ℕ) (h : y = 27) : x = 11 :=
sorry

end water_fee_part1_water_fee_part2_water_fee_usage_l188_188809


namespace find_a_l188_188150

def curve (a : ℝ) (x : ℝ) : ℝ := a * x + Real.exp x - 1

def tangent_line (x : ℝ) : ℝ := 3 * x

-- The statement to prove
theorem find_a (a : ℝ) (h : curve a 0 = 0) (h_tangent : ∀ x, tangent_line x = (curve a x - curve a 0) / x) : a = 2 :=
sorry

end find_a_l188_188150


namespace M_value_l188_188177

noncomputable def x : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)

noncomputable def y : ℝ := Real.sqrt (4 - 2 * Real.sqrt 3)

noncomputable def M : ℝ := x - y

theorem M_value :
  M = (5 / 2) * Real.sqrt 2 - Real.sqrt 3 + (3 / 2) :=
sorry

end M_value_l188_188177


namespace find_month_B_invests_l188_188020

def partnership_investment
  (x : ℕ) -- A's initial investment
  (m : ℕ) -- Months after which B invests
  (profit : ℕ := 12000) -- Total annual profit
  (a_share : ℕ := 4000) -- A's share of profit
  : Prop :=
  a_share * 3 = profit ∧
  (12 * x + 2 * x * (12 - m) + 3 * x * 4 = 3 * (12 * x))

theorem find_month_B_invests (x : ℕ) (m : ℕ) :
  partnership_investment x m → m = 6 :=
begin
  sorry
end

end find_month_B_invests_l188_188020


namespace Jan_keeps_375_feet_l188_188966

def Jan_keeps_on_hand_final_cable (total_cable : ℕ) (section_length : ℕ) (fraction_to_friend : ℚ) (fraction_to_storage : ℚ) : ℕ :=
  let total_sections := total_cable / section_length
  let sections_given := (fraction_to_friend * total_sections).natFloor
  let sections_left := total_sections - sections_given
  let sections_in_storage := (fraction_to_storage * sections_left).natFloor
  let sections_on_hand := sections_left - sections_in_storage
  sections_on_hand * section_length

theorem Jan_keeps_375_feet :
  Jan_keeps_on_hand_final_cable 1000 25 (1/4) (1/2) = 375 :=
by
  sorry

end Jan_keeps_375_feet_l188_188966


namespace transform_trig_expression_l188_188332

theorem transform_trig_expression (α : ℝ) :
  (4.24 * (cos ((5 * π / 2) - α) * sin ((π / 2) + (α / 2))) / 
  ((2 * sin ((π - α) / 2) + cos ((3 * π / 2) - α)) * (cos ((π - α) / 4))^2)) = 
  2 * tan (α / 2) :=
by
  sorry 

end transform_trig_expression_l188_188332


namespace C1_tangent_C2_l188_188817

/-- Define curve C1 in rectangular coordinates -/
def C1_rect (x y : ℝ) : Prop := y = 2

/-- Define curve C2 in rectangular coordinates -/
def C2_rect (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Define the center of the circle C2 -/
def C2_center : ℝ × ℝ := (2, 0)

/-- Define the radius of the circle C2 -/
def C2_radius : ℝ := 2

/-- Distance between a point and a line -/
def distance_point_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / sqrt (a^2 + b^2)

/-- Equation of the line C1 in standard form Ax + By + C = 0, where A = 0, B = 1, C = -2 -/
def line_C1_standard (x y : ℝ) : ℝ := 1 * y + (-2)

/-- Positional relationship between curves C1 and C2 -/
theorem C1_tangent_C2 :
  distance_point_line C2_center 0 1 (-2) = C2_radius ∧
  ∀ (x y : ℝ), C1_rect x y ∨ C2_rect x y → C1_rect x y → C2_rect x y → False := sorry

end C1_tangent_C2_l188_188817


namespace smallest_positive_angle_l188_188422

noncomputable theory

open Real

theorem smallest_positive_angle (x : ℝ) (hx : 0 < x) (H : tan (3 * x) = (sin x - cos x) / (sin x + cos x)) : x = 45 :=
by
  sorry

end smallest_positive_angle_l188_188422


namespace tan_of_angle_in_fourth_quadrant_l188_188882

-- Define the angle α in the fourth quadrant in terms of its cosine value
variable (α : Real)
variable (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) -- fourth quadrant condition
variable (h2 : Real.cos α = 4/5) -- given condition

-- Define the proof problem that tan α equals -3/4 given the conditions
theorem tan_of_angle_in_fourth_quadrant (α : Real) (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) (h2 : Real.cos α = 4/5) : 
  Real.tan α = -3/4 :=
sorry

end tan_of_angle_in_fourth_quadrant_l188_188882


namespace count_integer_roots_pairs_l188_188463

theorem count_integer_roots_pairs :
  (∑ k in Finset.range 51, (k / 2)) = 625 := by
sorry

end count_integer_roots_pairs_l188_188463


namespace equation_of_ellipse_range_of_B_x_coordinate_l188_188506

section part_I

variables (a b c : ℝ)
  (h_ab_pos : a > 0 ∧ b > 0)
  (h_a_gt_b : a > b)
  (h_eccentricity : c / a = Real.sqrt 2 / 2)
  (h_area : a * b = 2 * Real.sqrt 2)
  (h_radical_axis : a^2 = b^2 + c^2)

-- The equation of the ellipse
theorem equation_of_ellipse :
  (a = 2 ∧ b = Real.sqrt 2) →
  ∀ x y : ℝ, (x^2 / 4 + y^2 / 2 = 1) :=
sorry

end part_I

section part_II

variables (a b : ℝ) (A B : ℝ × ℝ)
  (P: ℝ × ℝ)
  (h_a : a = 2)
  (h_b : b = Real.sqrt 2)
  (h_A : A = (2, 0))
  (h_B_x : ∃ t : ℝ, B = (t, 0))
  (h_angle90 : ∀ P : ℝ × ℝ, ( (P.1 - A.1) * (t - P.1) + P.2 * (-P.2) = 0))

-- The range of B's x-coordinate
theorem range_of_B_x_coordinate :
  (h_exists_P : ∃ P : ℝ × ℝ, P ≠ A ∧ P ≠ (2, 0) ∧ (P.1^2 / 4 + P.2^2 / 2 = 1) ∧ (P.1 - 2) * (t - P.1) + P.2^2 = 0) →
  -2 < t ∧ t < 0 :=
sorry

end part_II

end equation_of_ellipse_range_of_B_x_coordinate_l188_188506


namespace ben_minimum_test_score_l188_188410

theorem ben_minimum_test_score 
  (scores : List ℕ) 
  (current_avg : ℕ) 
  (desired_increase : ℕ) 
  (lowest_score : ℕ) 
  (required_score : ℕ) 
  (h_scores : scores = [95, 85, 75, 65, 90]) 
  (h_current_avg : current_avg = 82) 
  (h_desired_increase : desired_increase = 5) 
  (h_lowest_score : lowest_score = 65) 
  (h_required_score : required_score = 112) :
  (current_avg + desired_increase) = 87 ∧ 
  (6 * (current_avg + desired_increase)) = 522 ∧ 
  required_score = (522 - (95 + 85 + 75 + 65 + 90)) ∧ 
  (522 - (95 + 85 + 75 + 65 + 90)) > lowest_score :=
by 
  sorry

end ben_minimum_test_score_l188_188410


namespace tan_y_l188_188201

theorem tan_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hy : 0 < y ∧ y < π / 2)
  (hsiny : Real.sin y = 2 * a * b / (a^2 + b^2)) :
  Real.tan y = 2 * a * b / (a^2 - b^2) :=
sorry

end tan_y_l188_188201


namespace bernoulli_inequality_gt_bernoulli_inequality_lt_l188_188641

theorem bernoulli_inequality_gt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : x > 1 ∨ x < 0) : (1 + h)^x > 1 + h * x := sorry

theorem bernoulli_inequality_lt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : 0 < x) (hx3 : x < 1) : (1 + h)^x < 1 + h * x := sorry

end bernoulli_inequality_gt_bernoulli_inequality_lt_l188_188641


namespace find_common_difference_l188_188502

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Module ℤ α] [CharZero α]

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → α) (d : α) := ∀ n, a (n + 1) = a n + d

-- Given conditions
def conditions (a : ℕ → ℤ) (d : ℤ) : Prop :=
(a 0 + a 8) * 9 / 2 = 27 ∧ a 9 = 8

-- Question: Prove that the common difference is 1 given the conditions
theorem find_common_difference (a : ℕ → ℤ) (d : ℤ) (h : conditions a d) : d = 1 :=
sorry

end find_common_difference_l188_188502


namespace two_digit_multiples_of_3_and_7_l188_188554

theorem two_digit_multiples_of_3_and_7 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ n % (3 * 7) = 0}.card = 4 :=
by
  -- The proof goes here, but it is not required.
  sorry

end two_digit_multiples_of_3_and_7_l188_188554


namespace infinite_radical_solution_l188_188077

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l188_188077


namespace trishas_test_scores_l188_188706

theorem trishas_test_scores:
  ∃ (scores: List ℕ),
    scores = [94, 92, 81, 75, 68] ∧
    (scores.nodup) ∧
    (scores.sum = 410) ∧
    (∀ s ∈ scores, s < 95) ∧
    ((92, 75, 68) ⊆ scores) ∧
    (∃ s ∈ scores, s / 10 = 6) :=
by
  sorry

end trishas_test_scores_l188_188706


namespace solve_equation_l188_188645

theorem solve_equation (x y : ℕ) (h_xy : x ≠ y) : x = 2 ∧ y = 4 ∨ x = 4 ∧ y = 2 :=
by {
  sorry -- Proof skipped
}

end solve_equation_l188_188645


namespace binom_equality_l188_188812

noncomputable def binom (a : ℚ) (b : ℕ) : ℚ :=
  (Finset.range b).prod (λ k, (a - k) / (k + 1))

theorem binom_equality :
  (binom (-1/2 : ℚ) 1000 * 8 ^ 1000) / (binom 2000 1000) = 1 / (Nat.factorial 1000) := by
  sorry

end binom_equality_l188_188812


namespace correct_equation_l188_188346

variables (a b : ℝ)

-- Conditions as definitions
def cond_A : Prop := 3 * a + 2 * a = 5 * a^2
def cond_B : Prop := 3 * a - a = 3
def cond_C : Prop := 2 * a^3 + 3 * a^2 = 5 * a^6
def cond_D : Prop := -0.25 * a * b + 0.25 * a * b = 0

theorem correct_equation : ¬cond_A ∧ ¬cond_B ∧ ¬cond_C ∧ cond_D :=
by {
  sorry
}

end correct_equation_l188_188346


namespace intersection_at_circumcenter_l188_188232

open EuclideanGeometry

variables {ω ω₁ ω₂ : Circle} {ℓ₁ ℓ₂ : Line} {A B C D E : Point}

-- Given conditions
def conditions (h₁ : Touches ω ℓ₁) (h₂ : Touches ω ℓ₂) 
               (h₃ : Touches ω₁ ℓ₁) (h₄ : Touches ω₁ ω C) 
               (h₅ : Touches ω₂ ℓ₂) (h₆ : Touches ω₂ ω D) 
               (h₇ : Touches ω₂ ω₁ E) (h₈ : ω₁.center = A ∧ ω₂.center = B) : Prop := 
  A ∈ ℓ₁ ∧ B ∈ ℓ₂ ∧ A ∈ ω₁ ∧ B ∈ ω₂

-- Main theorem statement
theorem intersection_at_circumcenter 
  (h₁ : Touches ω ℓ₁) (h₂ : Touches ω ℓ₂) 
  (h₃ : Touches ω₁ ℓ₁) (h₄ : Touches ω₁ ω C) 
  (h₅ : Touches ω₂ ℓ₂) (h₆ : Touches ω₂ ω D) 
  (h₇ : Touches ω₂ ω₁ E) (h₈ : ω₁.center = A ∧ ω₂.center = B) 
  (h_conditions : conditions h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈) : 
  let circumcenter := circumcenter_of_triangle C D E in 
  Intersect At circumcenter (Line_through A D) (Line_through B C) := 
sorry -- Proof is omitted

end intersection_at_circumcenter_l188_188232


namespace volume_of_revolved_solid_l188_188925

theorem volume_of_revolved_solid (l : Segment) (A : Solid) :
  -- Defining the line segment l joining the points (1, 0, 1) and (1, 0, 2)
  l = Segment.mk (Point.mk 1 0 1) (Point.mk 1 0 2) →
  -- Defining the figure A obtained by revolving l around the z-axis
  A = Figure.revolve l axis.z →
  -- Prove the volume of the solid obtained by revolving A around the x-axis
  volume (Solid.revolve A axis.x) = 2 * π^2 :=
sorry

end volume_of_revolved_solid_l188_188925


namespace primes_subset_eq_full_set_l188_188231

open Finset

theorem primes_subset_eq_full_set (P : Set ℕ) [hp : ∀ n, n ∈ P ↔ Prime n]
    (M : Finset ℕ) (hM : ∀ x ∈ M, x ∈ P) (h_size : 3 ≤ M.card)
    (h_prime_divisors : ∀ (A : Finset ℕ), A ≠ ∅ → A ≠ M → (∀ q, Prime q → q ∣ (A.prod id) - 1 → q ∈ M)) :
    (M : Set ℕ) = P.to_finset := sorry

end primes_subset_eq_full_set_l188_188231


namespace correct_conclusions_l188_188792

theorem correct_conclusions :
  let focus := (0, 1)
  let parabola := λ x, (1/4) * x^2
  let dist := (λ p (a b c : ℝ), (a * p.1 + b * p.2 + c) / (real.sqrt (a^2 + b^2))) (focus) (1, -1, -1)
  let f := λ x α, x ^ α
  let α := by sorry
  let f_four := f 4 α
  let exists_prop := ∃ x : ℝ, x^2 - x > 0
  let forall_prop := ∀ x : ℝ, x^2 - x < 0
  (dist = real.sqrt 2) ∧ (f 2 α = (real.sqrt 2) / 2 ∧ α = -1/2 ∧ f_four = 1/2) ∧ ¬ (forall_prop)
 :=
begin
  sorry
end

end correct_conclusions_l188_188792


namespace triangle_area_l188_188873

open Complex

noncomputable theory
def area_of_triangle_OAB (z1 z2 : ℂ) (h1 : abs z1 = 4) (h2 : 4 * z1 ^ 2 - 2 * z1 * z2 + z2 ^ 2 = 0) : ℝ :=
  8 * Real.sqrt 3

theorem triangle_area (z1 z2 : ℂ) (h1 : abs z1 = 4) (h2 : 4 * z1 ^ 2 - 2 * z1 * z2 + z2 ^ 2 = 0) :
  area_of_triangle_OAB z1 z2 h1 h2 = 8 * Real.sqrt 3 := by
  sorry

end triangle_area_l188_188873


namespace no_two_consecutive_courses_l188_188171

theorem no_two_consecutive_courses :
  let periods := 8
  let courses := 4
  let valid_arrangements := 120
  ∃ n : ℕ, periods = 8 ∧ courses = 4 ∧ n = (number_of_ways_to_schedule_courses periods courses) ∧ n = valid_arrangements := by sorry

end no_two_consecutive_courses_l188_188171


namespace Randy_blocks_used_l188_188281

theorem Randy_blocks_used (blocks_tower : ℕ) (blocks_house : ℕ) (total_blocks_used : ℕ) :
  blocks_tower = 27 → blocks_house = 53 → total_blocks_used = (blocks_tower + blocks_house) → total_blocks_used = 80 :=
by
  sorry

end Randy_blocks_used_l188_188281


namespace sequence_part_sum_l188_188900

-- Define the sequence a_n such that the cumulative sum equals n^3
def sequence_sum (n : ℕ) : ℕ :=
  if h : n > 0 then nat.rec_on n 0 (λ k sum, sum + (k + 1) ^ 3)
  else 0

-- Prove that a_6 + a_7 + a_8 + a_9 equals 604
theorem sequence_part_sum : sequence_sum 9 - sequence_sum 5 = 604 :=
by
  sorry

end sequence_part_sum_l188_188900


namespace negation_of_exists_l188_188308

theorem negation_of_exists (P : ℝ → Prop) : 
  (¬ ∃ x < 0, 2^x > 0) ↔ (∀ x < 0, 2^x ≤ 0) :=
begin
  sorry
end

end negation_of_exists_l188_188308


namespace triangle_AB_eq_2DM_l188_188954

theorem triangle_AB_eq_2DM 
  {A B C D M : Type} 
  [triangle ABC]
  (hB_angle : ∠B = 2 * ∠C)
  (hC_angle : ∠C = y)
  (h_altitude : is_altitude AD)
  (h_midpoint : is_midpoint M B C) : 
  AB = 2 * (segment DM) :=
by
  sorry

end triangle_AB_eq_2DM_l188_188954


namespace initial_cards_l188_188803

theorem initial_cards (taken left initial : ℕ) (h1 : taken = 59) (h2 : left = 17) (h3 : initial = left + taken) : initial = 76 :=
by
  sorry

end initial_cards_l188_188803


namespace total_area_of_ABHFGD_l188_188216

-- Define the geometric shapes and areas
def isSquare (s : ℝ) (area : ℝ) : Prop :=
  area = s^2

def isRectangle (l w : ℝ) (area : ℝ) : Prop :=
  area = l * w

-- Given conditions as Lean definitions
def ABDC_is_square : Prop := isSquare 5 25
def EFGD_is_rectangle (x y : ℝ) : Prop := isRectangle x y 25
def H_is_midpoint (BC EF : ℝ) : Prop :=
  BC = 5 ∧ EF * (EF / 2) = 25

-- Lean statement for the theorem to prove the total area of polygon ABHFGD
theorem total_area_of_ABHFGD :
  ABDC_is_square →
  ∃ (x y : ℝ), EFGD_is_rectangle x y →
  H_is_midpoint 5 x →
  25 :=
by
  sorry

end total_area_of_ABHFGD_l188_188216


namespace min_value_of_modulus_l188_188247

noncomputable def min_modulus (z : ℂ) : ℝ :=
  |z|

theorem min_value_of_modulus {z : ℂ} (h₁ : |z - 5 * complex.I| + |z - 2| = 10) :
  ∃ w : ℝ, min_modulus z = w ∧ w = 10 / real.sqrt 29 :=
begin
  sorry
end

end min_value_of_modulus_l188_188247


namespace find_f_find_m_l188_188879

variables {R : Type*} [OrderedRing R] [LinearOrderedField R]

def linear_increasing_function (f : R → R) :=
  ∃ a b : R, (a > 0) ∧ (f = λ x, a * x + b)

theorem find_f (f : ℝ → ℝ) (h_inc : linear_increasing_function f) (h_comp : ∀ (x : ℝ), f (f x) = 16 * x + 5) :
  f = (λ x, 4 * x + 1) :=
sorry

theorem find_m (m : ℝ) (h_max : ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 3 → (4 * x + 1) * (x + m) ≤ 13) :
  m = -2 :=
sorry

end find_f_find_m_l188_188879


namespace fraction_of_power_l188_188988

theorem fraction_of_power :
  let m := 27 ^ 1001 in
  m / 9 = 3 ^ 3001 :=
by
  sorry

end fraction_of_power_l188_188988


namespace shirts_per_minute_l188_188405

theorem shirts_per_minute (shirts_in_6_minutes : ℕ) (time_minutes : ℕ) (h1 : shirts_in_6_minutes = 36) (h2 : time_minutes = 6) : 
  ((shirts_in_6_minutes / time_minutes) = 6) :=
by
  sorry

end shirts_per_minute_l188_188405


namespace vertex_angle_measure_l188_188027

noncomputable def measure_vertex_angle (T : Triangle) : Float :=
  if is_isosceles T ∧ has_exterior_angle T 100 then 20 else if is_isosceles T ∧ has_exterior_angle T 100 then 80 else 0

-- Problem statement: Prove that if an exterior angle of an isosceles triangle is 100°, then the measure of the vertex angle must be either 20° or 80°.
theorem vertex_angle_measure (T : Triangle) (h1 : is_isosceles T) (h2 : has_exterior_angle T 100) :
  measure_vertex_angle T = 20 ∨ measure_vertex_angle T = 80 := by
sorry

end vertex_angle_measure_l188_188027


namespace range_of_m_value_of_m_l188_188897

-- Defining the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- The condition for having real roots
def has_real_roots (a b c : ℝ) := b^2 - 4 * a * c ≥ 0

-- First part: Range of values for m
theorem range_of_m (m : ℝ) : has_real_roots 1 (-2) (m - 1) ↔ m ≤ 2 := sorry

-- Second part: Finding m when x₁² + x₂² = 6x₁x₂
theorem value_of_m 
  (x₁ x₂ m : ℝ) (h₁ : quadratic_eq 1 (-2) (m - 1) x₁) (h₂ : quadratic_eq 1 (-2) (m - 1) x₂) 
  (h_sum : x₁ + x₂ = 2) (h_prod : x₁ * x₂ = m - 1) (h_condition : x₁^2 + x₂^2 = 6 * (x₁ * x₂)) : 
  m = 3 / 2 := sorry

end range_of_m_value_of_m_l188_188897


namespace ravi_overall_profit_l188_188736

-- Define basic conditions
def cost_price_refrigerator : ℝ := 15000
def cost_price_mobile : ℝ := 8000
def loss_percent_refrigerator : ℝ := 0.04
def profit_percent_mobile : ℝ := 0.11

-- Calculations based on conditions
def loss_refrigerator := loss_percent_refrigerator * cost_price_refrigerator
def selling_price_refrigerator := cost_price_refrigerator - loss_refrigerator

def profit_mobile := profit_percent_mobile * cost_price_mobile
def selling_price_mobile := cost_price_mobile + profit_mobile

def total_cost_price := cost_price_refrigerator + cost_price_mobile
def total_selling_price := selling_price_refrigerator + selling_price_mobile

def overall_profit_or_loss := total_selling_price - total_cost_price

-- Proof that the overall profit or loss is Rs. 280
theorem ravi_overall_profit : overall_profit_or_loss = 280 := by
    have : loss_refrigerator = 0.04 * 15000 := rfl
    have : loss_refrigerator = 600 := by norm_num
    have : selling_price_refrigerator = 15000 - 600 := rfl
    have : selling_price_refrigerator = 14400 := by norm_num

    have : profit_mobile = 0.11 * 8000 := rfl
    have : profit_mobile = 880 := by norm_num
    have : selling_price_mobile = 8000 + 880 := rfl
    have : selling_price_mobile = 8880 := by norm_num

    have : total_cost_price = 15000 + 8000 := rfl
    have : total_cost_price = 23000 := by norm_num

    have : total_selling_price = 14400 + 8880 := rfl
    have : total_selling_price = 23280 := by norm_num

    have : overall_profit_or_loss = 23280 - 23000 := rfl
    have : overall_profit_or_loss = 280 := by norm_num

    exact eq.refl 280

end ravi_overall_profit_l188_188736


namespace find_rate_of_interest_l188_188735

theorem find_rate_of_interest (P R : ℝ) (h1 : P * R * 2 / 100 = 325) (h2 : P * ((1 + R / 100)^2 - 1) = 340) : R ≈ 10.92 := by
  sorry

end find_rate_of_interest_l188_188735


namespace compound_interest_second_year_l188_188676

theorem compound_interest_second_year
  (P: ℝ) (r: ℝ) (CI_3 : ℝ) (CI_2 : ℝ)
  (h1 : r = 0.06)
  (h2 : CI_3 = 1272)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1200 :=
by
  sorry

end compound_interest_second_year_l188_188676


namespace ellipse_and_line_l188_188505

-- Define the ellipse condition
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def passes_through_A (x y : ℝ) : Prop :=
  x = 0 ∧ y = 2

def eccentricity (a c : ℝ) : Prop :=
  c / a = sqrt 2 / 2

def ellipse_condition (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2

-- Define properties of the line and circle
def right_focus (a : ℝ) : ℝ := a * sqrt 2 / 2
def tangent_line (x y : ℝ) : Prop := x - 2 * y - 2 = 0

theorem ellipse_and_line
  (a b c : ℝ)
  (h1 : 0 < a) (h2 : a > b) (h3 : b > 0)
  (h4 : passes_through_A 0 2)
  (h5 : eccentricity a c)
  (h6 : ellipse_condition a b c) :
  ∃ (M : ℝ × ℝ) (l : ℝ × ℝ → Prop),
    ellipse a b M.1 M.2 ∧ M ≠ (0, 2) ∧
    (∃ (C : ℝ × ℝ → Prop),
      C (right_focus a, 0) ∧ tangent_line M.1 M.2) ∧
    (∀ (x y : ℝ), l (x, y) ↔ y = -1/2 * x + 2) :=
begin
  sorry
end

end ellipse_and_line_l188_188505


namespace mean_of_remaining_l188_188675

variable (a b c : ℝ)
variable (mean_of_four : ℝ := 90)
variable (largest : ℝ := 105)

theorem mean_of_remaining (h1 : (a + b + c + largest) / 4 = mean_of_four) : (a + b + c) / 3 = 85 := by
  sorry

end mean_of_remaining_l188_188675


namespace arithmetic_square_root_l188_188295

theorem arithmetic_square_root (a : ℝ) (h : a = -4) : Real.sqrt (a ^ 2) = 4 :=
by
  have h1 : a^2 = 16 := by rw [h, pow_two, neg_mul_neg]
  rw [h1]
  exact Real.sqrt_sq (le_of_lt (lt_trans zero_lt_four (neg_lt_self zero_lt_four)))

end arithmetic_square_root_l188_188295


namespace ordered_pairs_satisfying_condition_l188_188169

theorem ordered_pairs_satisfying_condition : 
  ∃! (pairs : Finset (ℕ × ℕ)),
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ 
      m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 144) ∧ 
    pairs.card = 4 := sorry

end ordered_pairs_satisfying_condition_l188_188169


namespace solve_for_t_l188_188236

def point (t : ℝ) := (ℝ × ℝ)

def P (t : ℝ) : point t := (t - 3, 2)
def Q (t : ℝ) : point t := (-1, t + 2)

def dist (p1 p2 : ℝ × ℝ) := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem solve_for_t {t : ℝ} (h : dist (midpoint (P t) (Q t)) (P t) = t^2 + 1) :
  t = 4 ∨ t = 2 :=
sorry

end solve_for_t_l188_188236


namespace greatest_combination_bathrooms_stock_l188_188268

theorem greatest_combination_bathrooms_stock 
  (toilet_paper : ℕ) 
  (soap : ℕ) 
  (towels : ℕ) 
  (shower_gels : ℕ) 
  (h_tp : toilet_paper = 36)
  (h_soap : soap = 18)
  (h_towels : towels = 24)
  (h_shower_gels : shower_gels = 12) : 
  Nat.gcd (Nat.gcd (Nat.gcd toilet_paper soap) towels) shower_gels = 6 :=
by
  sorry

end greatest_combination_bathrooms_stock_l188_188268


namespace ben_mms_count_l188_188037

theorem ben_mms_count (S M : ℕ) (hS : S = 50) (h_diff : S = M + 30) : M = 20 := by
  sorry

end ben_mms_count_l188_188037


namespace complement_in_R_l188_188545

-- Definitions from conditions given in part a).
def U := Set.univ : Set ℝ
def P := { x : ℝ | x^2 ≤ 1 }

-- The theorem to prove the problem in c).
theorem complement_in_R :
  U \ P = { x : ℝ | x < -1 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end complement_in_R_l188_188545


namespace greatest_four_digit_divisible_by_conditions_l188_188738

-- Definitions based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = k * b

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

-- Problem statement: Finding the greatest 4-digit number divisible by 15, 25, 40, and 75
theorem greatest_four_digit_divisible_by_conditions :
  ∃ n, is_four_digit n ∧ is_divisible_by n 15 ∧ is_divisible_by n 25 ∧ is_divisible_by n 40 ∧ is_divisible_by n 75 ∧ n = 9600 :=
  sorry

end greatest_four_digit_divisible_by_conditions_l188_188738


namespace input_command_is_INPUT_l188_188948

-- Define the commands
def PRINT : String := "PRINT"
def INPUT : String := "INPUT"
def THEN : String := "THEN"
def END : String := "END"

-- Define the properties of each command
def PRINT_is_output (cmd : String) : Prop :=
  cmd = PRINT

def INPUT_is_input (cmd : String) : Prop :=
  cmd = INPUT

def THEN_is_conditional (cmd : String) : Prop :=
  cmd = THEN

def END_is_end (cmd : String) : Prop :=
  cmd = END

-- Theorem stating that INPUT is the command associated with input operation
theorem input_command_is_INPUT : INPUT_is_input INPUT :=
by
  -- Proof goes here
  sorry

end input_command_is_INPUT_l188_188948


namespace max_distance_traveled_l188_188697

theorem max_distance_traveled (fare: ℝ) (x: ℝ) :
  fare = 17.2 → 
  x > 3 →
  1.4 * (x - 3) + 6 ≤ fare → 
  x ≤ 11 := by
  sorry

end max_distance_traveled_l188_188697


namespace obtuse_triangle_probability_l188_188481

/-- 
Given four points chosen uniformly at random on a circle,
prove that the probability that no three of these points form an obtuse triangle with the circle's center is 3/32.
-/
theorem obtuse_triangle_probability : 
  let points := 4 in
  ∀ (P : Fin points → ℝ × ℝ) (random_uniform : ∀ i, random.uniform_on_circle i),
  (probability (no_three_form_obtuse_triangle P) = 3 / 32) :=
sorry

end obtuse_triangle_probability_l188_188481


namespace total_volume_of_snowballs_l188_188266

theorem total_volume_of_snowballs (r1 r2 r3 : ℝ) (π : ℝ) (h_r1 : r1 = 4) (h_r2 : r2 = 6) (h_r3 : r3 = 8) :
  (4/3) * π * (r1^3) + (4/3) * π * (r2^3) + (4/3) * π * (r3^3) = 1056 * π :=
by
  have h_total_volume : (4/3) * π * (4^3) + (4/3) * π * (6^3) + (4/3) * π * (8^3) = 1056 * π,
  {
    -- Proof would go here but is omitted by using sorry
    sorry
  }
  exact h_total_volume

end total_volume_of_snowballs_l188_188266


namespace average_speed_of_planes_l188_188328

def planePassengers (n1 n2 n3 : ℕ) := (n1, n2, n3)
def emptyPlaneSpeed : ℕ := 600
def speedReductionPerPassenger : ℕ := 2
def planeSpeed (s0 r n : ℕ) : ℕ := s0 - r * n
def averageSpeed (speeds : List ℕ) : ℕ := (List.sum speeds) / speeds.length

theorem average_speed_of_planes :
  let (n1, n2, n3) := planePassengers 50 60 40 in
  let s0 := emptyPlaneSpeed in
  let r := speedReductionPerPassenger in
  let speed1 := planeSpeed s0 r n1 in
  let speed2 := planeSpeed s0 r n2 in
  let speed3 := planeSpeed s0 r n3 in
  averageSpeed [speed1, speed2, speed3] = 500 := by
  sorry

end average_speed_of_planes_l188_188328


namespace limit_arcsin_tan_pi_l188_188421

theorem limit_arcsin_tan_pi :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, continuous_at real.arcsin x) →
  (∀ x ∈ set.Icc (-1 : ℝ) 1, continuous_at (λ x, real.tan (π * x)) x) →
  (∃ L : ℝ, filter.tendsto (λ x, (real.arcsin x) ^ (real.tan (π * x))) (nhds 1) (nhds L) ∧ L = 1) :=
by 
  sorry

end limit_arcsin_tan_pi_l188_188421


namespace min_value_3_div_a_add_2_div_b_l188_188517

/-- Given positive real numbers a and b, and the condition that the lines
(a + 1)x + 2y - 1 = 0 and 3x + (b - 2)y + 2 = 0 are perpendicular,
prove that the minimum value of 3/a + 2/b is 25, given the condition 3a + 2b = 1. -/
theorem min_value_3_div_a_add_2_div_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h : 3 * a + 2 * b = 1) : 3 / a + 2 / b ≥ 25 :=
sorry

end min_value_3_div_a_add_2_div_b_l188_188517


namespace count_integers_l188_188700

theorem count_integers (lst : List ℚ) : lst = [6, -3.1, 0.17, 0, -3, 2/3, -1, 2, -2, -17/4] → 
  (lst.filter (λ x => ∃ n : ℤ, x = n)).length = 6 := 
by 
  sorry

end count_integers_l188_188700


namespace find_k_value_l188_188849

def line (k : ℝ) (x y : ℝ) : Prop := 3 - 2 * k * x = -4 * y

def on_line (k : ℝ) : Prop := line k 5 (-2)

theorem find_k_value (k : ℝ) : on_line k → k = -0.5 :=
by
  sorry

end find_k_value_l188_188849


namespace math_problem_l188_188876

noncomputable def x : ℝ := -2

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}
def C1 : Set ℝ := {1, 3}
def C2 : Set ℝ := {3, 4}

theorem math_problem
  (h1 : B x ⊆ A x) :
  x = -2 ∧ (B x ∪ C1 = A x ∨ B x ∪ C2 = A x) :=
by
  sorry

end math_problem_l188_188876


namespace parabola_sum_of_coefficients_l188_188685

theorem parabola_sum_of_coefficients (a b c : ℝ) 
  (h₁ : ∃ (a b c : ℝ), (∀ x : ℝ, (y = a * x^2 + b * x + c) ∧ (y = 0) → (x = -3) ∨ (x = 1)) 
        (h₂ : ∃ (l : ℝ), (l = -1) → (∀ x : ℝ, (y = a * x^2 + b * x + c) ∧ (y = y(-1)) → l = x)) : 
  a + b + c = 0 :=
sorry

end parabola_sum_of_coefficients_l188_188685


namespace vasya_claim_false_l188_188699

theorem vasya_claim_false :
  ∀ (weights : List ℕ), weights = [1, 2, 3, 4, 5, 6, 7] →
  (¬ ∃ (subset : List ℕ), subset.length = 3 ∧ 1 ∈ subset ∧
  ((weights.sum - subset.sum) = 14) ∧ (14 = 14)) :=
by
  sorry

end vasya_claim_false_l188_188699


namespace simplify_fraction_144_12672_l188_188644

theorem simplify_fraction_144_12672 : (144 / 12672 : ℚ) = 1 / 88 :=
by
  sorry

end simplify_fraction_144_12672_l188_188644


namespace find_dividend_l188_188840

-- Conditions
def quotient : ℕ := 4
def divisor : ℕ := 4

-- Dividend computation
def dividend (q d : ℕ) : ℕ := q * d

-- Theorem to prove
theorem find_dividend : dividend quotient divisor = 16 := 
by
  -- Placeholder for the proof, not needed as per instructions
  sorry

end find_dividend_l188_188840


namespace inequality_am_gm_l188_188875

theorem inequality_am_gm (n : ℕ) (a : Fin n → ℝ) (h₀ : ∀ i, 0 < a i) (h₁ : (∑ i, a i) = 1) :
  (∏ i, (1 / (a i)^2 - 1)) ≥ (n^2 - 1)^n :=
by
  sorry

end inequality_am_gm_l188_188875


namespace coeff_x1_x2_x1000_is_zero_P_is_zero_when_x_in_neg1_pos1_l188_188047

def ξ (n k : ℤ) : ℤ :=
  if n ≤ k then 1 else -1

noncomputable def P (x : Fin 1000 → ℤ) : ℤ :=
  ∏ n in Finset.range 1000, ∑ k in Finset.range 1000, ξ n k * x k

theorem coeff_x1_x2_x1000_is_zero : 
  coeff_x1_x2_x1000 P = 0 := 
sorry

theorem P_is_zero_when_x_in_neg1_pos1
  (x : Fin 1000 → ℤ) (hx : ∀ i, x i ∈ {-1, 1}) : 
  P x = 0 :=
sorry

end coeff_x1_x2_x1000_is_zero_P_is_zero_when_x_in_neg1_pos1_l188_188047


namespace catering_budget_l188_188970

namespace CateringProblem

variables (s c : Nat) (cost_steak cost_chicken : Nat)

def total_guests (s c : Nat) : Prop := s + c = 80

def steak_to_chicken_ratio (s c : Nat) : Prop := s = 3 * c

def total_cost (s c cost_steak cost_chicken : Nat) : Nat := s * cost_steak + c * cost_chicken

theorem catering_budget :
  ∃ (s c : Nat), (total_guests s c) ∧ (steak_to_chicken_ratio s c) ∧ (total_cost s c 25 18) = 1860 :=
by
  sorry

end CateringProblem

end catering_budget_l188_188970


namespace club_population_in_five_years_l188_188753

def club_population : ℕ → ℕ
| 0     := 21
| (n+1) := 4 * (club_population n - 7) + 7

theorem club_population_in_five_years : club_population 5 = 14343 :=
by
  sorry

end club_population_in_five_years_l188_188753


namespace mean_weight_correct_l188_188692

def weights := [51, 60, 62, 64, 64, 65, 67, 73, 74, 74, 75, 76, 77, 78, 79]

noncomputable def mean_weight (weights : List ℕ) : ℚ :=
  (weights.sum : ℚ) / weights.length

theorem mean_weight_correct :
  mean_weight weights = 69.27 := by
  sorry

end mean_weight_correct_l188_188692


namespace heartsuit_4_6_l188_188918

-- Define the operation \heartsuit
def heartsuit (x y : ℤ) : ℤ := 5 * x + 3 * y

-- Prove that 4 \heartsuit 6 = 38 under the given operation definition
theorem heartsuit_4_6 : heartsuit 4 6 = 38 := by
  -- Using the definition of \heartsuit
  -- Calculation is straightforward and skipped by sorry
  sorry

end heartsuit_4_6_l188_188918


namespace sequence_properties_l188_188253

noncomputable def general_term (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = if q = 2 then 2 ^ (n - 1) else (-2) ^ (n - 1)

noncomputable def sum_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in finset.range n, a i)

theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) 
  (h₀ : (∀ n, a(n) ≠ 0) → (∃ q, ∀ n, a(n) = a(0) * q ^ n)) 
  (h₁ : a 1 = 1) (h₂ : a 2 * a 3 * a 4 = 64) : 
  (general_term a 2 ∨ general_term a (-2)) ∧ 
  ((∀ n, S n + λ = 2 ^ n + λ - 1 → λ = 1) ∨ 
  (∀ n, S n + λ = (-2) ^ n + λ + 1 / 3 → λ = -1 / 3)) :=
  sorry

end sequence_properties_l188_188253


namespace men_left_hostel_l188_188001

theorem men_left_hostel (x : ℕ) (x ≥ 0) (men : ℕ) 
  (initial_men := 250) (days_with_initial_men := 32) 
  (days_with_remaining_men := 40)
  (initial_provisions : men * days_with_initial_men = (men - x) * days_with_remaining_men) : 
  x = 50 := by
  sorry

end men_left_hostel_l188_188001


namespace chocolate_chip_count_l188_188264

variable (n_total_cookies : ℕ)
variable (n_chocolate_chip_cookies : ℕ)
variable (n_oatmeal_cookies : ℕ)

noncomputable def total_cookies := 6 * 3
def oatmeal_cookies := 16
def chocolate_chip_cookies := n_total_cookies - n_oatmeal_cookies

theorem chocolate_chip_count : n_chocolate_chip_cookies = 2 :=
  by
    have h1 : n_total_cookies = total_cookies := sorry
    have h2 : n_oatmeal_cookies = oatmeal_cookies := sorry
    show n_chocolate_chip_cookies = 2, from sorry

end chocolate_chip_count_l188_188264


namespace time_b_is_54_l188_188181

-- Define the time A takes to complete the work
def time_a := 27

-- Define the time B takes to complete the work as twice the time A takes
def time_b := 2 * time_a

-- Prove that B takes 54 days to complete the work
theorem time_b_is_54 : time_b = 54 :=
by
  sorry

end time_b_is_54_l188_188181


namespace rice_grains_difference_l188_188019

theorem rice_grains_difference :
  let grains (k : ℕ) := 2^k
  let sum_first_9 := ∑ i in finset.range 9, grains (i + 1)
  grains 12 - sum_first_9 = 3074 := by
  sorry

end rice_grains_difference_l188_188019


namespace num_subsets_and_proper_subsets_of_set_1_2_l188_188318

theorem num_subsets_and_proper_subsets_of_set_1_2 :
  ∃ (S : set ℕ), S = {1, 2} ∧ (S.powerset.card = 4 ∧ (S.powerset.card - 1) = 3) :=
by {
  sorry
}

end num_subsets_and_proper_subsets_of_set_1_2_l188_188318


namespace contrapositive_statement_l188_188313

theorem contrapositive_statement (a b : ℝ) :
  (a² + b² = 0 → a = 0 ∧ b = 0) →
  (a ≠ 0 ∨ b ≠ 0 → a² + b² ≠ 0) :=
by
  intro h₁ h₂
  sorry

end contrapositive_statement_l188_188313


namespace ellipse_equation_range_of_slope_l188_188503

theorem ellipse_equation (a b c: ℝ) (h1: a > b) (h2: b > 0) (h3: c = 1) (h4: a = 2) (h5: b = sqrt 3) : 
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) ):= 
sorry

theorem range_of_slope (k : ℝ) : 
  (∀ k : ℝ, (k ≤ - sqrt 6 / 4) ∨ (k ≥ sqrt 6 / 4) ⟹ ∃ l : line, is_line l ∧ slope l = k ) :=
sorry

end ellipse_equation_range_of_slope_l188_188503


namespace line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l188_188638

-- Define the set of people
def people : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : ℕ := 1 
def B : ℕ := 2
def C : ℕ := 3

-- Proof Problem 1: Prove that there are 1800 ways to line up 5 people out of 7 given A must be included.
theorem line_up_including_A : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 2: Prove that there are 1800 ways to line up 5 people out of 7 given A, B, and C are not all included.
theorem line_up_excluding_all_ABC : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 3: Prove that there are 144 ways to line up 5 people out of 7 given A, B, and C are all included, A and B are adjacent, and C is not adjacent to A or B.
theorem line_up_adjacent_AB_not_adjacent_C : Finset ℕ → ℕ :=
by
  sorry

end line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l188_188638


namespace price_reduction_l188_188755

theorem price_reduction (P : ℝ) (hP : P = 0.24) (n : ℕ) 
  (sales_doubled : ∀ n, 2 * n)
  (profit_increased : (1.5 * (n * P)) = (0.36 * n)) :
  (P - 0.18 = 0.06) :=
sorry

end price_reduction_l188_188755


namespace find_AP_l188_188276

-- Definitions for the square ABCD
def A := (0 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 0 : ℝ)
def C := (1 : ℝ, 1 : ℝ)
def D := (0 : ℝ, 1 : ℝ)

-- Definitions for midpoints M and N
def M := ((0 + 1) / 2 : ℝ, 0 : ℝ)
def N := (1 : ℝ, (0 + 1) / 2 : ℝ)

-- Definitions for equations of lines CM and DN
def line_CM (x : ℝ) : ℝ := 2 * x - 1
def line_DN (x : ℝ) : ℝ := -1 / 2 * x + 1

-- Intersection point P
def P : ℝ × ℝ := (4 / 5, 3 / 5)

-- Distance between point (0, 0) and point (x, y)
def distance (x y : ℝ) : ℝ := Real.sqrt (x ^ 2 + y ^ 2)

theorem find_AP : distance (4 / 5) (3 / 5) = 1 :=
by sorry

end find_AP_l188_188276


namespace problem_equiv_proof_l188_188850

def not_divisible_by_four (b : ℕ) : Prop := 
  (b^3 + b^2 - b + 2) % 4 ≠ 0

theorem problem_equiv_proof : 
  ∀ b ∈ ({4, 5, 6, 7, 8} : set ℕ), 
    not_divisible_by_four b ↔ b ∈ ({4, 5, 7, 8} : set ℕ) := by
  sorry

end problem_equiv_proof_l188_188850


namespace order_a_b_c_d_l188_188917

def a : ℝ := Real.sin (Real.sin (2013 * Real.pi / 180))
def b : ℝ := Real.sin (Real.cos (2013 * Real.pi / 180))
def c : ℝ := Real.cos (Real.sin (2013 * Real.pi / 180))
def d : ℝ := Real.cos (Real.cos (2013 * Real.pi / 180))

theorem order_a_b_c_d :
  b < a ∧ a < d ∧ d < c :=
sorry

end order_a_b_c_d_l188_188917


namespace watermelons_last_6_weeks_l188_188226

variable (initial_watermelons : ℕ) (eaten_per_week : ℕ) (given_away_per_week : ℕ)

def watermelons_last_weeks (initial_watermelons : ℕ) (weekly_usage : ℕ) : ℕ :=
initial_watermelons / weekly_usage

theorem watermelons_last_6_weeks :
  initial_watermelons = 30 ∧ eaten_per_week = 3 ∧ given_away_per_week = 2 →
  watermelons_last_weeks initial_watermelons (eaten_per_week + given_away_per_week) = 6 := 
by
  intro h
  cases h with h_initial he 
  cases he with  h_eaten h_given
  have weekly_usage := h_eaten + h_given
  have weeks := watermelons_last_weeks h_initial weekly_usage
  sorry

end watermelons_last_6_weeks_l188_188226


namespace part_one_part_two_l188_188134

namespace ProofProblem

def setA (a : ℝ) := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def setB := {x : ℝ | 0 < x ∧ x < 1}

theorem part_one (a : ℝ) (h : a = 1/2) : 
  setA a ∩ setB = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

theorem part_two (a : ℝ) (h_subset : setB ⊆ setA a) : 
  0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end ProofProblem

end part_one_part_two_l188_188134


namespace average_temperatures_l188_188034

theorem average_temperatures :
  let high_temps := [49, 62, 58, 57, 46, 55, 60]
  let low_temps := [40, 47, 45, 41, 39, 42, 44]
  ∑ i in (finset.range 7), high_temps[i] / 7.0 = 55.9 ∧
  ∑ i in (finset.range 7), low_temps[i] / 7.0 = 42.6 :=
by
  sorry

end average_temperatures_l188_188034


namespace isosceles_triangle_bisector_l188_188705

theorem isosceles_triangle_bisector {A B C D : Type*}
  (hABC_isosceles: is_isosceles_triangle A B C)
  (hAB: distance A B = 2)
  (hAC: distance A C = 2)
  (hAD_bisector: is_angle_bisector A D B C)
  (hD_on_BC : is_point_on_line D B C)
  : distance B D = 1 := 
sorry

end isosceles_triangle_bisector_l188_188705


namespace exists_maximum_value_of_f_l188_188642

-- Define the function f(x, y)
noncomputable def f (x y : ℝ) : ℝ := (3 * x * y + 1) * Real.exp (-(x^2 + y^2))

-- Maximum value proof statement
theorem exists_maximum_value_of_f :
  ∃ (x y : ℝ), f x y = (3 / 2) * Real.exp (-1 / 3) :=
sorry

end exists_maximum_value_of_f_l188_188642


namespace probability_x_lt_2y_in_rectangle_l188_188777

theorem probability_x_lt_2y_in_rectangle :
  let rectangle := set.Icc (0, 0) (4, 2)
  let region := {p : ℝ × ℝ | p ∈ rectangle ∧ p.1 < 2 * p.2}
  (measure_theory.measure.region.probability_of_region region rectangle = 1 / 2) :=
by
  sorry

end probability_x_lt_2y_in_rectangle_l188_188777


namespace find_possible_values_of_k_l188_188690

open Nat

noncomputable def sequence (u : ℕ → ℕ) (k : ℕ) : Prop :=
  u 0 = 1 ∧ ∀ n, u (n + 1) * u (n - 1) = k * u n

theorem find_possible_values_of_k (u : ℕ → ℕ) (k : ℕ) (h_seq : sequence u k) (h_u2000 : u 2000 = 2000) :
  k ∈ ({2000, 1000, 500, 400, 200, 100} : Finset ℕ) :=
  sorry

end find_possible_values_of_k_l188_188690


namespace parallel_lines_a_eq_3_l188_188743

theorem parallel_lines_a_eq_3
  (a : ℝ)
  (l1 : a^2 * x - y + a^2 - 3 * a = 0)
  (l2 : (4 * a - 3) * x - y - 2 = 0)
  (h : ∀ x y, a^2 * x - y + a^2 - 3 * a = (4 * a - 3) * x - y - 2) :
  a = 3 :=
by
  sorry

end parallel_lines_a_eq_3_l188_188743


namespace minimum_value_a_l188_188843

theorem minimum_value_a (a : ℝ) (h : 0 < a ∧ a < 1) :
  a = (Real.sqrt 2 / Real.pi * Real.sin (Real.pi / Real.sqrt 2)) ↔
  (∀ b, (0 < b ∧ b < 1) → (∫ x in 0..Real.pi, Real.abs (Real.sin x - b * x)) ≥ (∫ x in 0..Real.pi, Real.abs (Real.sin x - a * x))) :=
sorry

end minimum_value_a_l188_188843


namespace infinite_nested_sqrt_l188_188070

theorem infinite_nested_sqrt :
  let x := \sqrt{20 + \sqrt{20 + \sqrt{20 + \sqrt{20 + \cdots}}}} in
  x = 5 :=
begin
  let x : ℝ := sqrt(20 + sqrt(20 + sqrt(20 + sqrt(20 + ...)))),
  have h1 : x = sqrt(20 + x), from sorry,
  have h2 : x^2 = 20 + x, from sorry,
  have h3 : x^2 - x - 20 = 0, from sorry,
  have h4 : (x - 5) * (x + 4) = 0, from sorry,
  have h5 : x = 5 ∨ x = -4, from sorry,
  have h6 : x >= 0, from sorry,
  exact h5.elim (λ h, h) (λ h, (h6.elim_left h))
end

end infinite_nested_sqrt_l188_188070


namespace slope_range_PA1_l188_188044

variables {x y a b : ℝ}

def hyperbola (a b : ℝ) : Prop := (a^2 / 4) - (b^2 / 5) = 1
def slope_PA2 (a b : ℝ) : ℝ := b / (a - 2)
def slope_PA1 (a b : ℝ) : ℝ := b / (a + 2)

theorem slope_range_PA1
  (h : hyperbola a b)
  (ha : a ≠ -2) (hb : a ≠ 2)
  (hPA2 : slope_PA2 a b ∈ (1 / 2 : ℝ, 1)) :
  slope_PA1 a b ∈ (5 / 4 : ℝ, 5 / 2) :=
sorry

end slope_range_PA1_l188_188044


namespace transformation_sequence_terminates_l188_188010

-- Define the basic elements involved in the problem.
noncomputable def point := ℝ³

def is_no_four_coplanar (points: finset point) : Prop := sorry -- Exact definition omitted

def partition (s: finset point) : Type :=
  Σ (A B : finset point), (∀ x y, x ∈ A → y ∈ B → x ≠ y)

structure AB_tree (A B : finset point) :=
  (edges : finset (point × point))
  (no_closed_polyline : sorry) -- Exact definition omitted

def transformation (A B : finset point) (t : AB_tree A B) : Prop :=
  ∃ (a1 a2 : point) (b1 b2 : point), a1 ∈ A ∧ a2 ∈ A ∧ b1 ∈ B ∧ b2 ∈ B ∧
  (a1, b1) ∈ t.edges ∧ (a2, b2) ∈ t.edges ∧
  (|a1.to_real - b1.to_real| + |a2.to_real - b2.to_real| > |a1.to_real - b2.to_real| + |a2.to_real - b1.to_real|)

theorem transformation_sequence_terminates {n : ℕ} (points: finset point) (h : is_no_four_coplanar points) 
  (A B : finset point) (p : partition points) (t : AB_tree A B):
  ∀ sequence : ℕ → AB_tree A B, 
  (∀ k, transformation A B (sequence k) → sequence (k+1) = some_transformation (sequence k)) → 
  ∃ K, sequence K = sequence (K + 1) := 
sorry

end transformation_sequence_terminates_l188_188010


namespace identify_incorrect_statement_l188_188350

def is_monomial (expr : String) : Prop := sorry
def coefficient (expr : String) : Real := sorry
def degree (expr : String) : ℕ := sorry
def incorrect_statement := "The coefficient of −(2πab)/3 is −2/3"

-- Definitions corresponding to problem conditions
def option_A := is_monomial "0"
def option_B := coefficient "b" = 1 ∧ degree "b" = 1
def option_C := coefficient "1/2 * x^2 * y^2" = 1/2 ∧ degree "1/2 * x^2 * y^2" = 4
def option_D := coefficient "-(2πab)/3" = -2/3

theorem identify_incorrect_statement (A : Prop) (B : Prop) (C : Prop) (D : Prop) : 
  A ∧ B ∧ C ∧ ¬D -> incorrect_statement = "The coefficient of −(2πab)/3 is −2/3" :=
by
  intros h,
  have a := (h.1).1,
  have b := (h.1).2.1,
  have c := (h.1).2.2,
  have not_d := h.2,
  unfold incorrect_statement,
  exact (h.right)

end identify_incorrect_statement_l188_188350


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l188_188805

theorem problem_1 : 78 * 4 + 488 = 800 := by
  calc
    78 * 4 + 488 = 312 + 488 := by rw [←Nat.mul_comm 78 4]
    _ = 800 := by rw [Nat.add_comm 312 488]

theorem problem_2 : 350 * (12 + 342 / 9) = 17500 := by
  calc
    350 * (12 + 342 / 9) = 350 * (12 + (342 / 9)) := by rw [Nat.div_eq_of_lt (Nat.lt_add_one_iff.2 _) _]
    _ = 350 * (12 + 38) := by rw [←Nat.div_comm 342 9]
    _ = 350 * 50 := by rw [Nat.add_comm 12 38]
    _ = 17500 := rfl

theorem problem_3 : (3600 - 18 * 200) / 253 = 0 := by
  calc
    (3600 - 18 * 200) / 253 = (3600 - (18 * 200)) / 253 := by rw [←Nat.mul_comm 18 200]
    _ = 0 / 253 := by rw [Nat.sub_self 3600]
    _ = 0 := rfl

theorem problem_4 : 1903 - 475 * 4 = 3 := by
  calc
    1903 - 475 * 4 = 1903 - (475 * 4) := by rw [←Nat.mul_comm 475 4]
    _ = 1903 - 1900 := rfl
    _ = 3 := by rw [Nat.sub_self]

theorem problem_5 : 480 / (125 - 117) = 60 := by
  calc
    480 / (125 - 117) = 480 / (8) := by rw [Nat.sub_self]
    _ = 60 := rfl

theorem problem_6 : (243 - 162) / 27 * 380 = 1140 := by
  calc
    (243 - 162) / 27 * 380 = (81) / 27 * 380 := by rw [Nat.sub_self]
    _ = 3 * 380 := by rw [Nat.div_self]
    _ = 1140 := rfl


end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l188_188805


namespace family_reunion_kids_l188_188799

theorem family_reunion_kids (adults : ℕ) (tables : ℕ) (people_per_table : ℕ) 
  (h_adults : adults = 123) (h_tables : tables = 14) 
  (h_people_per_table : people_per_table = 12) :
  (tables * people_per_table - adults) = 45 :=
by
  sorry

end family_reunion_kids_l188_188799


namespace domino_piles_sum_l188_188909

theorem domino_piles_sum (dominoes: Fin 28 → ℕ) 
    (h_sum: (∑ i, dominoes i) = 168) 
    (h_piles: ∃ (s1 s2 s3 s4: Finset (Fin 28)),
        s1 ∪ s2 ∪ s3 ∪ s4 = Finset.univ ∧
        s1 ∩ s2 = ∅ ∧
        s1 ∩ s3 = ∅ ∧
        s1 ∩ s4 = ∅ ∧
        s2 ∩ s3 = ∅ ∧
        s2 ∩ s4 = ∅ ∧
        s3 ∩ s4 = ∅ ∧
        (∑ i in s1, dominoes i) = 37 ∧
        (∑ i in s2, dominoes i) = 41 ∧
        (∑ i in s3, dominoes i) = 43 ∧
        (∑ i in s4, dominoes i) = 47) : True :=
    sorry

end domino_piles_sum_l188_188909


namespace average_speed_of_planes_l188_188327

def planePassengers (n1 n2 n3 : ℕ) := (n1, n2, n3)
def emptyPlaneSpeed : ℕ := 600
def speedReductionPerPassenger : ℕ := 2
def planeSpeed (s0 r n : ℕ) : ℕ := s0 - r * n
def averageSpeed (speeds : List ℕ) : ℕ := (List.sum speeds) / speeds.length

theorem average_speed_of_planes :
  let (n1, n2, n3) := planePassengers 50 60 40 in
  let s0 := emptyPlaneSpeed in
  let r := speedReductionPerPassenger in
  let speed1 := planeSpeed s0 r n1 in
  let speed2 := planeSpeed s0 r n2 in
  let speed3 := planeSpeed s0 r n3 in
  averageSpeed [speed1, speed2, speed3] = 500 := by
  sorry

end average_speed_of_planes_l188_188327


namespace ratio_of_Katie_to_Cole_l188_188352

variable (K C : ℕ)

theorem ratio_of_Katie_to_Cole (h1 : 3 * K = 84) (h2 : C = 7) : K / C = 4 :=
by
  sorry

end ratio_of_Katie_to_Cole_l188_188352


namespace no_real_sol_l188_188822

open Complex

theorem no_real_sol (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (↑(x.re) ≠ x ∨ ↑(y.re) ≠ y) → (x + y) / y ≠ x / (y + x) := by
  sorry

end no_real_sol_l188_188822


namespace solve_inequality_l188_188154

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x else log x + 2

theorem solve_inequality (x : ℝ) : f x > 3 ↔ x < -3 ∨ x > Real.exp 1 := by
  sorry

end solve_inequality_l188_188154


namespace f_f_5_l188_188860

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition : ∀ x : ℝ, f(x + 2) = 1 / f(x)
axiom f_initial : f 1 = -5

theorem f_f_5 : f (f 5) = -1 / 5 :=
by
  sorry

end f_f_5_l188_188860


namespace product_of_distances_equal_l188_188725

theorem product_of_distances_equal
  (A O B P Q P' Q' : Point)
  (hAOB : angle O A B)
  (hP_on_Perp_to_OA : distance P OA = P')
  (hQ_on_Perp_to_OB : distance Q OB = Q')
  (hEqual_Angles : ∀ O' M N, angle M O A = angle N O B)
  (h_on_rays: ∀ M N, point_on_ray O M P ∧ point_on_ray O N Q)
  :
  (distance P' OA) * (distance Q' OB) = (distance P' Q' O).collab :=
by
  -- Establish intermediate steps using properties of similar triangles
  sorry

end product_of_distances_equal_l188_188725


namespace complement_of_intersection_range_of_a_l188_188254

variable (U : Set ℝ) (A B C : Set ℝ)
variable (a : ℝ)

def universal_set : Set ℝ := set.univ
def set_A : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def set_B : Set ℝ := { x | 2 * x - 4 ≥ x - 2 }
def set_C (a : ℝ) : Set ℝ := { x | 2 * x + a > 0 }

theorem complement_of_intersection :
  U = universal_set →
  A = set_A →
  B = set_B →
  (compl (A ∩ B)) = { x | x < 2 ∨ x ≥ 3 } :=
by
  intro hU hA hB
  sorry

theorem range_of_a :
  B = set_B →
  (B ∪ set_C a) = set_C a →
  a ≥ -4 :=
by
  intro hB hC
  sorry

end complement_of_intersection_range_of_a_l188_188254


namespace probability_non_obtuse_l188_188489

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l188_188489


namespace average_percentage_increase_l188_188601

theorem average_percentage_increase (initial_A initial_B initial_C new_A new_B new_C : ℝ)
  (hA : initial_A = 50) (hA' : new_A = 75)
  (hB : initial_B = 100) (hB' : new_B = 120)
  (hC : initial_C = 80) (hC' : new_C = 88) :
  ((new_A - initial_A) / initial_A * 100 + (new_B - initial_B) / initial_B * 100 +
   (new_C - initial_C) / initial_C * 100) / 3 = 26.67 := 
by
  calc sorry

end average_percentage_increase_l188_188601


namespace length_DF_l188_188408

noncomputable def point : Type := ℝ × ℝ

def A : point := (0, 0)
def B : point := (8, 0)
def C : point := (8, 8)
def D : point := (0, 8)

variable (E : point)
variable hA : real.sqrt ((E.1)^2 + (E.2)^2) = 10
variable hB : real.sqrt ((E.1 - 8)^2 + (E.2)^2) = 6

theorem length_DF :
  ∃ x y : ℝ, (E = (x, y)) → real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = 2 * real.sqrt 17 :=
sorry

end length_DF_l188_188408


namespace average_percentage_increase_l188_188599

theorem average_percentage_increase (initial_A raised_A initial_B raised_B initial_C raised_C : ℕ) 
(hA: initial_A = 50) (hRA: raised_A = 75) 
(hB: initial_B = 100) (hRB: raised_B = 120)
(hC: initial_C = 80) (hRC: raised_C = 88) : 
  ((raised_A - initial_A) * 100 / initial_A + (raised_B - initial_B) * 100 / initial_B + (raised_C - initial_C) * 100 / initial_C) / 3 = 26.67 := 
by 
  sorry

end average_percentage_increase_l188_188599


namespace perfect_square_factors_l188_188911

theorem perfect_square_factors (a b c : ℕ) (h₁ : a = 10) (h₂ : b = 12) (h₃ : c = 15) :
  let factors_of_2 := a/2 + 1,
      factors_of_3 := b/2 + 1,
      factors_of_5 := c/2 + 1
  in factors_of_2 * factors_of_3 * factors_of_5 = 336 :=
by
  have factors_of_2 := h₁ / 2 + 1,
  have factors_of_3 := h₂ / 2 + 1,
  have factors_of_5 := h₃ / 2 + 1,
  sorry

end perfect_square_factors_l188_188911


namespace complex_division_example_l188_188521

-- Given conditions
def i : ℂ := Complex.I

-- The statement we need to prove
theorem complex_division_example : (1 + 3 * i) / (1 + i) = 2 + i :=
by
  sorry

end complex_division_example_l188_188521


namespace quadratic_has_only_negative_roots_l188_188573

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_only_negative_roots (m : ℝ) :
  (∀ (x : ℝ), x^2 + (m+2)*x + (m+5) = 0 → x < 0) ↔ m ≥ 4 :=
begin
  sorry
end

end quadratic_has_only_negative_roots_l188_188573


namespace math_problem_l188_188427

def dist (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)

def curve_C := { P : ℝ × ℝ | dist P (-1, 0) * dist P (1, 0) = 4 }

def ellipse := { P : ℝ × ℝ | (P.1 ^ 2) / 4 + (P.2 ^ 2) / 3 = 1 }

theorem math_problem :
  (∀ (P : ℝ × ℝ), P ∈ curve_C → P ∈ curve_C → P.1 = -P.1 ∧ P.2 = -P.2) ∧
  (∀ (P : ℝ × ℝ), P ∈ curve_C → (½ * 4 * real.sin (real.acos (((P.1 + 1) * (1 - P.1) + 0 * 0) / (dist P (-1, 0) * dist P (1, 0)))) ≤ 2)) ∧
  (∃ (P1 P2 : ℝ × ℝ), P1 ∈ curve_C ∧ P2 ∈ curve_C ∧ P1 ∈ ellipse ∧ P2 ∈ ellipse ∧ P1 ≠ P2 ∧ 
    ∀ (P : ℝ × ℝ), P ∈ curve_C ∧ P ∈ ellipse → P = P1 ∨ P = P2) :=
by sorry

end math_problem_l188_188427


namespace convex_polyhedron_has_two_faces_same_sides_l188_188632

/-- Def. of a convex polyhedron: The intersection of finitely many open half-spaces, provided this intersection is bounded and non-empty. -/
structure ConvexPolyhedron :=
(faces : List ConvexPolygon)
(vertices edges faces_count: Nat)
(Euler: vertices - edges + faces_count = 2)

theorem convex_polyhedron_has_two_faces_same_sides (K : ConvexPolyhedron) :
  ∃ (f1 f2 : ConvexPolygon), f1 ∈ K.faces ∧ f2 ∈ K.faces ∧ f1.num_edges = f2.num_edges ∧ f1 ≠ f2 :=
by
  sorry

end convex_polyhedron_has_two_faces_same_sides_l188_188632


namespace log_odd_function_range_lt_0_l188_188245

def f (x : ℝ) : ℝ := log x

theorem log_odd_function_range_lt_0 :
  (f : ℝ → ℝ) = log →
  (∀ x : ℝ, f (-x) = -f x) →
  {x : ℝ | f x < 0} = set.Ioo (0 : ℝ) 1 :=
sorry

end log_odd_function_range_lt_0_l188_188245


namespace cos_F_cos_D_l188_188942

variables (DE DF EF : ℝ)
hypothesis h1 : DE = 9
hypothesis h2 : DF = 15
theorem cos_F (h3 : EF = Real.sqrt (DE^2 + DF^2)) : Real.cos (Real.arccos (DE / EF)) = 3 * Real.sqrt 34 / 34 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

theorem cos_D : Real.cos (π / 2) = 0 :=
by norm_num

end cos_F_cos_D_l188_188942


namespace minimum_sequence_length_l188_188993

theorem minimum_sequence_length :
  ∃ n (a : Finₓ n → Finₓ 4),
    (∀ B : Finset (Finₓ 4), B ≠ ∅ → 
      ∃ i : Finₓ (n - B.card + 1), (Finset.univ.filter (λ x, a ⟨i + x, sorry⟩) = B)) ∧ n = 8 :=
sorry

end minimum_sequence_length_l188_188993


namespace number_increase_l188_188639

theorem number_increase (S : Finset ℝ) (hS_card : S.card = 10) (hS_mean : S.sum / 10 = 6.2) (c : ℝ) (h_new_mean : (S.sum + c) / 10 = 7) : c = 8 :=
sorry

end number_increase_l188_188639


namespace tanks_difference_l188_188741

theorem tanks_difference (total_tanks german_tanks allied_tanks sanchalian_tanks : ℕ)
  (h_total : total_tanks = 115)
  (h_german_allied : german_tanks = 2 * allied_tanks + 2)
  (h_allied_sanchalian : allied_tanks = 3 * sanchalian_tanks + 1)
  (h_total_eq : german_tanks + allied_tanks + sanchalian_tanks = total_tanks) :
  german_tanks - sanchalian_tanks = 59 :=
sorry

end tanks_difference_l188_188741


namespace xyz_line_segments_total_length_l188_188936

noncomputable def total_length_XYZ : ℝ :=
  let length_X := 2 * Real.sqrt 2
  let length_Y := 2 + 2 * Real.sqrt 2
  let length_Z := 2 + Real.sqrt 2
  length_X + length_Y + length_Z

theorem xyz_line_segments_total_length : total_length_XYZ = 4 + 5 * Real.sqrt 2 := 
  sorry

end xyz_line_segments_total_length_l188_188936


namespace dabbie_turkey_cost_l188_188431

theorem dabbie_turkey_cost :
  let weight1 := 6
  let weight2 := 9
  let weight3 := 2 * weight2
  let cost_per_kg := 2
  let total_weight := weight1 + weight2 + weight3
  total_weight * cost_per_kg = 66 :=
by
  let weight1 := 6
  let weight2 := 9
  let weight3 := 2 * weight2
  let cost_per_kg := 2
  let total_weight := weight1 + weight2 + weight3
  show total_weight * cost_per_kg = 66
  rw [total_weight, cost_per_kg, 66]
  sorry

end dabbie_turkey_cost_l188_188431


namespace units_digit_F_F_15_l188_188671

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem units_digit_F_F_15 : (fib (fib 15) % 10) = 5 := by
  sorry

end units_digit_F_F_15_l188_188671


namespace complex_expression_evaluation_l188_188880

noncomputable def imaginary_i := Complex.I 

theorem complex_expression_evaluation : 
  ((2 + imaginary_i) / (1 - imaginary_i)) - (1 - imaginary_i) = -1/2 + (5/2) * imaginary_i :=
by 
  sorry

end complex_expression_evaluation_l188_188880


namespace geometric_sequence_a8_value_l188_188210

variable {a : ℕ → ℕ}

-- Assuming a is a geometric sequence, provide the condition a_3 * a_9 = 4 * a_4
def geometric_sequence_condition (a : ℕ → ℕ) :=
  (a 3) * (a 9) = 4 * (a 4)

-- Prove that a_8 = 4 under the given condition
theorem geometric_sequence_a8_value (a : ℕ → ℕ) (h : geometric_sequence_condition a) : a 8 = 4 :=
  sorry

end geometric_sequence_a8_value_l188_188210


namespace choice_first_question_range_of_P2_l188_188586

theorem choice_first_question (P1 P2 a b : ℚ) (hP1 : P1 = 1/2) (hP2 : P2 = 1/3) :
  (P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) > 0) ↔ a > b / 2 :=
sorry

theorem range_of_P2 (a b P1 P2 : ℚ) (ha : a = 10) (hb : b = 20) (hP1 : P1 = 2/5) :
  P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) ≥ 0 ↔ (0 ≤ P2 ∧ P2 ≤ P1 / (2 - P1)) :=
sorry

end choice_first_question_range_of_P2_l188_188586


namespace inequality_solution_l188_188650

theorem inequality_solution :
  {x : ℝ | |2 * x - 3| + |x + 1| < 7 ∧ x ≤ 4} = {x : ℝ | -5 / 3 < x ∧ x < 3} :=
by
  sorry

end inequality_solution_l188_188650


namespace range_of_m_l188_188175

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → x^2 - 2 * x - 3 > 0) → (0 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l188_188175


namespace car_factory_ratio_l188_188394

theorem car_factory_ratio
  (cars_yesterday : ℕ)
  (total_cars : ℕ)
  (cars_yesterday_eq : cars_yesterday = 60)
  (total_cars_eq : total_cars = 180) :
  (2 * cars_yesterday) = (total_cars - cars_yesterday) :=
by
  rw [cars_yesterday_eq, total_cars_eq]
  sorry

end car_factory_ratio_l188_188394


namespace expression_evaluates_to_three_halves_l188_188415

theorem expression_evaluates_to_three_halves :
  ( (81 / 16) ^ (-1 / 4) +
    (1 / 4 * real.log 3 / real.log (real.sqrt 2) * real.log 4 / real.log 3 * ((-1 / 3) ^ 2) ^ (1 / 2)) +
    7 ^ real.log (1 / 2) / real.log 7 ) = 3 / 2 :=
  sorry

end expression_evaluates_to_three_halves_l188_188415


namespace fencing_needed_l188_188387

-- Definitions of the conditions
def garden_length : ℝ := 60
def garden_width : ℝ := garden_length / 2
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

-- The statement we need to prove
theorem fencing_needed :
  perimeter garden_length garden_width = 180 := 
  sorry

end fencing_needed_l188_188387


namespace solve_for_x_l188_188050

theorem solve_for_x : ∃ x : ℝ, (∛(x * real.sqrt (x^3)) = 3) ∧ x = 3^(6/5) := by
  let x := 3^(6/5)
  use x
  split
  { 
    sorry
  }
  { 
    refl 
  }

end solve_for_x_l188_188050


namespace possible_values_of_y_probability_of_history_probability_of_geography_l188_188727

def P := ℝ × ℝ
def O : P := (0, 0)
def P1 : P := (-1, 0)
def P2 : P := (-1, 1)
def P3 : P := (0, 1)
def P4 : P := (1, 1)
def P5 : P := (1, 0)

def dot_product (v w : P) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem possible_values_of_y :
  ∀ v1 v2 ∈ {P1, P2, P3, P4, P5}, 
  dot_product v1 v2 ∈ {-1, 0, 1} := sorry

theorem probability_of_history :
  ∀ (v1 v2 : P) (hv1 : v1 ∈ {P1, P2, P3, P4, P5}) (hv2 : v2 ∈ {P1, P2, P3, P4, P5}),
  4 / 10 = 2 / 5 := sorry

theorem probability_of_geography :
  ∀ (v1 v2 : P) (hv1 : v1 ∈ {P1, P2, P3, P4, P5}) (hv2 : v2 ∈ {P1, P2, P3, P4, P5}),
  3 / 10 = 0.3 := sorry

end possible_values_of_y_probability_of_history_probability_of_geography_l188_188727


namespace total_candidates_l188_188940

theorem total_candidates (T : ℝ) 
  (h1 : 0.45 * T = T * 0.45)
  (h2 : 0.38 * T = T * 0.38)
  (h3 : 0.22 * T = T * 0.22)
  (h4 : 0.12 * T = T * 0.12)
  (h5 : 0.09 * T = T * 0.09)
  (h6 : 0.10 * T = T * 0.10)
  (h7 : 0.05 * T = T * 0.05)
  (h_passed_english_alone : T - (0.45 * T - 0.12 * T - 0.10 * T + 0.05 * T) = 720) :
  T = 1000 :=
by
  sorry

end total_candidates_l188_188940


namespace range_of_a_l188_188166

noncomputable
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (A a ∪ B = B) ↔ a < -4 ∨ a > 5 :=
sorry

end range_of_a_l188_188166


namespace least_positive_multiple_of_13_gt_418_l188_188712

theorem least_positive_multiple_of_13_gt_418 : ∃ (n : ℕ), n > 418 ∧ (13 ∣ n) ∧ n = 429 :=
by
  sorry

end least_positive_multiple_of_13_gt_418_l188_188712


namespace common_chord_diameter_circle_eqn_l188_188165
noncomputable theory

-- Mathematical conditions and the given circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * a * x = 0
def circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * b * y = 0

-- The derived equation of the circle whose diameter is the common chord
def desired_circle_eqn (a b x y : ℝ) : Prop := 
  (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0

-- Prove the two circles and their conditions result in the desired circle equation
theorem common_chord_diameter_circle_eqn (a b : ℝ) (hb : b ≠ 0) : 
  ∃ x y : ℝ, circle1 a x y ∧ circle2 b x y → desired_circle_eqn a b x y :=
by
  intro x y h
  sorry


end common_chord_diameter_circle_eqn_l188_188165


namespace inequality_relations_l188_188244

noncomputable def a : ℝ := 0.5 ^ 0.1
noncomputable def b : ℝ := Real.log 0.1 / Real.log 4
noncomputable def c : ℝ := 0.4 ^ 0.1

theorem inequality_relations : a > c ∧ c > b := by
  sorry

end inequality_relations_l188_188244


namespace max_of_exponential_sum_l188_188243

noncomputable def max_value_of_exponential_sum : Real :=
  let a b : Real := sorry -- a and b are real numbers
  if h : a + b = 3 then max (2^a + 2^b) else sorry -- Conditions: a + b = 3

theorem max_of_exponential_sum : max_value_of_exponential_sum = 4 := 
  sorry

end max_of_exponential_sum_l188_188243


namespace probability_no_obtuse_triangle_correct_l188_188475

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l188_188475


namespace length_DE_is_3_l188_188425

def is_circle (center : ℝ×ℝ) (radius : ℝ) (point : ℝ×ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

def point_D (D : ℝ×ℝ) : Prop :=
  is_circle (0, real.sqrt 3) (real.sqrt 5) D ∧ is_circle (-2, 0) (real.sqrt 5) D

def point_E (E : ℝ×ℝ) : Prop :=
  is_circle (0, real.sqrt 3) (real.sqrt 5) E ∧ is_circle (2, 0) (real.sqrt 5) E

theorem length_DE_is_3 (D E : ℝ×ℝ) (hD : point_D D) (hE : point_E E) : 
  real.dist D E = 3 :=
sorry

end length_DE_is_3_l188_188425


namespace complete_square_eqn_l188_188649

theorem complete_square_eqn (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 → (x + d)^2 = e) → d + e = 5 :=
by
  sorry

end complete_square_eqn_l188_188649


namespace expected_diff_tea_coffee_in_leap_year_l188_188024

/-! 
  Define the problem setup
-/

-- Defining the set of outcomes for the eight-sided die and categorize them as prime or composite.
def outcomes : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

def is_prime (n : Nat) : Bool :=
  n ∈ [2, 3, 5, 7]

def is_composite (n : Nat) : Bool :=
  n ∈ [4, 6, 8]

-- Probability of rolling a prime number, excluding 1 (since rolling 1 leads to a re-roll).
def prob_prime := 4 / 7

-- Probability of rolling a composite number, excluding 1.
def prob_composite := 3 / 7

-- The number of days in a leap year.
def days_in_leap_year := 366

-- Expected number of days drinking tea (prime numbers).
def expected_days_tea := prob_prime * days_in_leap_year

-- Expected number of days drinking coffee (composite numbers).
def expected_days_coffee := prob_composite * days_in_leap_year

-- Define the expected value of the difference in days drinking tea versus coffee.
def expected_difference := expected_days_tea - expected_days_coffee

-- We need to show that the expected difference is approximately equal to 53.
theorem expected_diff_tea_coffee_in_leap_year :
  expected_difference ≈ 53 := 
sorry

end expected_diff_tea_coffee_in_leap_year_l188_188024


namespace perimeter_triangle_ABC_l188_188202

-- Define the given data and conditions
variables (A B C X Y Z W P : Type)
variables [EuclideanGeometry A B C] [RightTriangle A B C (angle C B A = 90)] (angleC : angle A B C = 90)
variables (AB : Real) (P_on_YZ : is_point_on_line_segment P Y Z)
variables (AB : ℝ) (AB_EQ_15 : AB = 15)
variables [Square ABXY] [Square CBWZ]
variables (X_on_Circle : is_point_on_circle X (circle_sup W Y Z))
variables (Y_on_Circle : is_point_on_circle Y (circle_sup W Y Z))
variables (Z_on_Circle : is_point_on_circle Z (circle_sup W Y Z))
variables (W_on_Circle : is_point_on_circle W (circle_sup W Y Z))

-- Define the final theorem
theorem perimeter_triangle_ABC :
  ∃ (a b c : ℝ), a = 5 * sqrt 6 ∧ b = 5 * sqrt 3 ∧ c = 15 ∧ (a + b + c = 15 + 5 * sqrt 3 + 5 * sqrt 6) :=
sorry

end perimeter_triangle_ABC_l188_188202


namespace no_such_integer_a_l188_188781

noncomputable def x_sequence (a : ℤ) : ℕ → ℤ
| 0 => a
| 1 => 2
| (n + 2) => 2 * x_sequence (n + 1) * x_sequence n - x_sequence (n + 1) - x_sequence n + 1

def is_perfect_square (n : ℤ) : Prop :=
∃ m : ℤ, m * m = n

theorem no_such_integer_a : ∀ (a : ℤ), ¬ (∀ n : ℕ, n ≥ 1 → is_perfect_square (2 * x_sequence a (3 * n) - 1)) :=
by sorry

end no_such_integer_a_l188_188781


namespace decreasing_function_range_l188_188854

theorem decreasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x ≥ f y ∧ f x ≤ f y) →
  (∃ x < 1, f x = (3 * a - 1) * x + 4 * a) →
  (∃ x > 1, f x = log a x) →
  (∀ x y : ℝ, x < y → (3 * a - 1) * x + 4 * a < (3 * a - 1) * y + 4 * a) →
  (∀ x > 1, x < log a x → x ≥ log a x) →
  (3 * a - 1 < 0) →
  (0 < a < 1) →
  ((3 * a - 1) * 1 + 4 * a ≥ log a 1) →
  (frac 1 7 ≤ a ∧ a < frac 1 3) :=
by
  sorry

end decreasing_function_range_l188_188854


namespace floor_ceiling_product_l188_188082

theorem floor_ceiling_product : ( ∏ i in [1, 2, 3, 4, 5], (Int.floor (-i - 0.5) * Int.ceil (i + 0.5)) ) * Int.floor (-0.5) * Int.ceil (0.5) = -518400 := by
  sorry

end floor_ceiling_product_l188_188082


namespace find_a5_a6_l188_188949

noncomputable theory

-- Define a sequence a_n as a geometric sequence
variable (a : ℕ → ℝ)
variable (q : ℝ) -- the common ratio

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

-- Given conditions
variable (h1 : a 1 + a 2 = 20)
variable (h2 : a 3 + a 4 = 40)
variable (geom : geom_seq a q)

-- Required to prove
theorem find_a5_a6 : a 5 + a 6 = 80 :=
by
  sorry

end find_a5_a6_l188_188949


namespace value_of_a_l188_188562

theorem value_of_a (x a : ℤ) (h1 : x = 2) (h2 : 3 * x - a = -x + 7) : a = 1 :=
by
  sorry

end value_of_a_l188_188562


namespace problem1_problem2_problem3_l188_188149

def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 14 * y + 45 = 0

def point_Q : ℝ × ℝ := (-2, 3)

-- Problem 1: length and slope of segment PQ
theorem problem1 (a : ℝ) (hP : circle_C a (a + 1)) :
  let P : ℝ × ℝ := (a, a + 1),
      Q : ℝ × ℝ := point_Q,
      distPQ := (P.1 + 2)^2 + (P.2 - 3)^2,
      slopePQ := (P.2 - Q.2) / (P.1 - Q.1) in
  distPQ = 40 ∧ slopePQ = 1/3 :=
  sorry

-- Problem 2: Maximum and minimum values of |MQ| where Q is point_Q and M is any point on circle C
theorem problem2 (M : ℝ × ℝ) (hM : circle_C M.1 M.2) :
  let Q : ℝ × ℝ := point_Q,
      centerC : ℝ × ℝ := (2, 7),
      radiusC := 2 * sqrt 2,
      distQC := sqrt ((Q.1 - centerC.1)^2 + (Q.2 - centerC.2)^2),
      distMQ := sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) in
  2 * sqrt 2 ≤ distMQ ∧ distMQ ≤ 6 * sqrt 2 :=
  sorry

-- Problem 3: Maximum and minimum values of K = (n-3) / (m+2) given circle_C condition
theorem problem3 (m n : ℝ) (h : circle_C m n) :
  let K : ℝ := (n - 3) / (m + 2),
      min_K := 2 - sqrt 3,
      max_K := 2 + sqrt 3 in
  min_K ≤ K ∧ K ≤ max_K :=
  sorry

end problem1_problem2_problem3_l188_188149


namespace transformed_curve_l188_188703

variables (x y x' y' θ : ℝ)

def transformation (x y : ℝ) : ℝ × ℝ :=
  (x / 3, y / 2)

noncomputable def parametric_form (θ : ℝ) : ℝ × ℝ :=
  (sqrt 3 / 3 * cos θ, sqrt 2 / 2 * sin θ)

theorem transformed_curve (x y : ℝ) :
  x = sqrt 3 / 3 * cos θ ∧ y = sqrt 2 / 2 * sin θ ↔
  3 * (x' * 3)^2 + 2 * (y' * 2)^2 = 1 :=
by
  sorry

end transformed_curve_l188_188703


namespace problem1_problem2_l188_188417

-- Problem 1
theorem problem1 : 2 * Real.tan (Float.pi / 3) - Real.abs (Real.sqrt 3 - 2) - 3 * Real.sqrt 3 + (1 / 3)⁻¹ = 1 := 
  by
    sorry

-- Problem 2
theorem problem2 (x : ℝ) : (x - 3)^2 = 2*x - 6 ↔ (x = 3 ∨ x = 5) := 
  by
    sorry

end problem1_problem2_l188_188417


namespace fill_table_with_numbers_l188_188511

theorem fill_table_with_numbers (m n : ℕ) (a : Fin m → ℝ) (b : Fin n → ℝ) 
  (h₀ : ∀ i, 0 < a i) (h₁ : ∀ j, 0 < b j)
  (h₂ : (∑ i, a i) = (∑ j, b j)) : 
  ∃ (f : Fin m → Fin n → ℝ), 
    (∀ i, ∑ j, f i j = a i) ∧ 
    (∀ j, ∑ i, f i j = b j) ∧ 
    (∑ i j, if 0 < f i j then 1 else 0) ≤ m + n - 1 := 
sorry

end fill_table_with_numbers_l188_188511


namespace find_a_l188_188530

theorem find_a (a : ℝ) :
  (∀ x : ℝ, f (2 - x) = 2 - f x) ∧ f 0 = 2 → a = 1 :=
by
  let f := λ x : ℝ, (a * x - 2) / (x - 1)
  sorry

end find_a_l188_188530


namespace z_gets_amount_per_unit_l188_188393

-- Define the known conditions
variables (x y z : ℝ)
variables (x_share : ℝ)
variables (y_share : ℝ)
variables (z_share : ℝ)
variables (total : ℝ)

-- Assume the conditions given in the problem
axiom h1 : y_share = 54
axiom h2 : total = 234
axiom h3 : (y / x) = 0.45
axiom h4 : total = x_share + y_share + z_share

-- Prove the target statement
theorem z_gets_amount_per_unit : ((z_share / x_share) = 0.50) :=
by
  sorry

end z_gets_amount_per_unit_l188_188393


namespace inverse_proposition_holds_l188_188719

-- Definitions related to the conditions:
def corresponding_angles_equal (l₁ l₂ : Line) : Prop := ∀ (a b : Point), corresponding_angle l₁ a b = corresponding_angle l₂ a b
def vertical_angles_equal (a b : Angle) : Prop := ∃ p : Point, vertical_angle a p = vertical_angle b p
def corresponding_sides_equal (△₁ △₂ : Triangle) : Prop := ∀ (a b c : Side), corresponding_side △₁ a b c = corresponding_side △₂ a b c
def equidistant_from_sides (p : Point) (a b : AngleBisector) : Prop := distance p a = distance p b

-- Theorem stating the problem:
theorem inverse_proposition_holds (A B C D : Prop) :
  (inverse corresponding_angles_equal) → 
  (¬ inverse vertical_angles_equal) ∧ 
  (inverse corresponding_sides_equal) ∧ 
  (inverse equidistant_from_sides) := 
sorry

end inverse_proposition_holds_l188_188719


namespace length_of_arc_l188_188143

theorem length_of_arc (S : ℝ) (α : ℝ) (hS : S = 4) (hα : α = 2) : 
  ∃ l : ℝ, l = 4 :=
by
  sorry

end length_of_arc_l188_188143


namespace valid_n_values_l188_188652

theorem valid_n_values :
  {n : ℕ | ∀ a : ℕ, a^(n+1) ≡ a [MOD n]} = {1, 2, 6, 42, 1806} :=
sorry

end valid_n_values_l188_188652


namespace palindromes_with_seven_percentage_l188_188770

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def contains_digit (n digit : ℕ) : Prop :=
  digit.to_string ∈ n.to_string

theorem palindromes_with_seven_percentage :
  let palindromes := {n | 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n}
  let palindromes_with_7 := {n | 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n ∧ (contains_digit n 7)}
  ((palindromes_with_7.to_finset.card : ℚ) / palindromes.to_finset.card * 100) = 19 :=
by
  sorry

end palindromes_with_seven_percentage_l188_188770


namespace dabbies_turkey_cost_l188_188428

noncomputable def first_turkey_weight : ℕ := 6
noncomputable def second_turkey_weight : ℕ := 9
noncomputable def third_turkey_weight : ℕ := 2 * second_turkey_weight
noncomputable def cost_per_kg : ℕ := 2

noncomputable def total_cost : ℕ :=
  first_turkey_weight * cost_per_kg +
  second_turkey_weight * cost_per_kg +
  third_turkey_weight * cost_per_kg

theorem dabbies_turkey_cost : total_cost = 66 :=
by
  sorry

end dabbies_turkey_cost_l188_188428


namespace probability_no_adjacent_people_standing_l188_188657

noncomputable def probability_no_adjacent_stands : ℚ :=
  let num_people := 10
  let total_outcomes := 2^num_people
  let a8 := 47
  let a9 := 76
  let a10 := a9 + a8
  a10 / total_outcomes

theorem probability_no_adjacent_people_standing (num_people := 10) (total_outcomes := 2^num_people) (a8 := 47) (a9 := 76) (a10 := a9 + a8) :
  probability_no_adjacent_stands = 123 / 1024 :=
by
  rw [probability_no_adjacent_stands, total_outcomes, a10]
  norm_num
  sorry

end probability_no_adjacent_people_standing_l188_188657


namespace parabola_focus_l188_188841

theorem parabola_focus :
  ∃ f_x f_y : ℚ, (y = 4 * x^2 + 8 * x - 1) → (f_x, f_y) = (-1 : ℚ, -79/16 : ℚ) :=
by
  sorry

end parabola_focus_l188_188841


namespace angles_of_triangle_l188_188960

namespace TriangleAngleProblem

-- Definitions based on the conditions
def Triangle := Type
variable (A B C M K L : Triangle)

-- Axioms based on the given conditions
axiom angle_bisectors_from_A_B (A B C : Triangle): Prop
axiom median_from_C (C : Triangle): Prop
-- Intersection points forming a right triangle
axiom right_triangle_formed (K L M : Triangle): Prop

-- Statement to be proved based on the question and correct answer
theorem angles_of_triangle (A B C M K L : Triangle) 
  (h1: angle_bisectors_from_A_B A B C) 
  (h2: median_from_C C) 
  (h3: right_triangle_formed K L M) : 
  Triangle.angles A B C = (30, 60, 90) := 
sorry

end TriangleAngleProblem

end angles_of_triangle_l188_188960


namespace count_divisors_of_36_divisible_by_6_l188_188288

theorem count_divisors_of_36_divisible_by_6 :
    {b : ℕ | 0 < b ∧ b ∣ 36 ∧ 6 ∣ b}.to_finset.card = 4 :=
by sorry

end count_divisors_of_36_divisible_by_6_l188_188288


namespace D_f_g_range_a_l188_188465

section problem1

def f (x: ℝ) : ℝ := 2 * abs x
def g (x: ℝ) : ℝ := x + 3

theorem D_f_g : { x : ℝ | f x > g x } = { x : ℝ | x < -1 ∨ x > 3 } :=
by sorry

end problem1

section problem2

def f1 (x: ℝ) : ℝ := x - 1
def f2 (x: ℝ) (a: ℝ) : ℝ := (1/3)^x + a * 3^x + 1
def h (x: ℝ) : ℝ := 0

theorem range_a (a : ℝ) : {x : ℝ | f1 x > h x} ∪ {x : ℝ | f2 x a > h x} = set.univ → a > -4/9 :=
by sorry

end problem2

end D_f_g_range_a_l188_188465


namespace area_of_EMD_in_triangleABC_l188_188955

theorem area_of_EMD_in_triangleABC
  (A B C M D E : Type)
  [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
  (angle_B : B > (90 : ℝ)) -- $\angle B > 90^\circ$
  (AC AM MC : ℝ)
  (AM_eq_3MC : AM = 3 * MC)
  (area_ABC : (AC * (AC / 4) * sin angle_B) / 2 = 36)
  (MD_perp_AB : ∃ MD : ℝ, MD ⊥ AC)
  (ED_perp_AB : ∃ ED : ℝ, ED ⊥ AC) : 
  ∃ (area_EMD : ℝ), area_EMD = 9 / 8 := 
sorry

end area_of_EMD_in_triangleABC_l188_188955


namespace height_of_water_in_cylinder_l188_188009

-- Definitions of given conditions
def base_area_pyramid : ℝ := 144  -- cm^2
def height_pyramid : ℝ := 27  -- cm
def base_radius_cylinder : ℝ := 9  -- cm

-- Volume of a pyramid
def volume_pyramid : ℝ := (1 / 3) * base_area_pyramid * height_pyramid

-- Volume of a cylinder
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Theorem statement to prove the height of water in the cylindrical container
theorem height_of_water_in_cylinder :
  ∀ h : ℝ, volume_cylinder base_radius_cylinder h = volume_pyramid ↔ h = 16 / π :=
by
  -- Proof would go here
  sorry

end height_of_water_in_cylinder_l188_188009


namespace correct_operation_l188_188348

theorem correct_operation :
  (∀ a b, (sqrt a + sqrt b ≠ sqrt (a + b)) ∧
          (∀ r, r + sqrt b ≠ r * sqrt b) ∧
          (∀ r b, (r * sqrt b)^2 = r^2 * b) ∧
          (∀ n, sqrt (n^2) = abs n)) →
  (2 * sqrt 3) ^ 2 = 12 :=
by
  intro h
  sorry

end correct_operation_l188_188348


namespace probability_non_obtuse_l188_188490

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l188_188490


namespace find_largest_c_and_no_nat_n_l188_188842

noncomputable def frac (x : ℚ) : ℚ := x - x.floor

theorem find_largest_c_and_no_nat_n :
  let c := 1 / (2 * Real.sqrt 2) in
  (∀ (n : ℕ), frac (n * Real.sqrt 2) ≥ c / n) ∧ ¬(∃ (n : ℕ), frac (n * Real.sqrt 2) = c / n) :=
by
  sorry

end find_largest_c_and_no_nat_n_l188_188842


namespace imaginary_part_of_complex_l188_188303

theorem imaginary_part_of_complex :
  let z := (2 + 3 * Complex.i) / (-3 + 2 * Complex.i)
  Complex.im z = -1 :=
by
  sorry

end imaginary_part_of_complex_l188_188303


namespace area_triangle_ABC_l188_188953

theorem area_triangle_ABC {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
(h_right_angle: ∠C = 90)
(h_CA_eq_4: dist C A = 4)
(h_CD_eq_1: dist C D = 1)
(h_circle: ∃ r, r = (sqrt 5 / 2) ∧ passes_through_circle C D r ∧ tangent_at_C_to_circumcircle_triangle_ABC):
(area_triangle_ABC = 4) :=
sorry

end area_triangle_ABC_l188_188953


namespace payment_n_amount_l188_188362

def payment_m_n (m n : ℝ) : Prop :=
  m + n = 550 ∧ m = 1.2 * n

theorem payment_n_amount : ∃ n : ℝ, ∀ m : ℝ, payment_m_n m n → n = 250 :=
by
  sorry

end payment_n_amount_l188_188362


namespace smallest_n_square_average_l188_188342

theorem smallest_n_square_average (n : ℕ) (h : n > 1)
  (S : ℕ := (n * (n + 1) * (2 * n + 1)) / 6)
  (avg : ℕ := S / n) :
  (∃ k : ℕ, avg = k^2) → n = 337 := by
  sorry

end smallest_n_square_average_l188_188342


namespace bipartite_partition_count_l188_188157

def A := {1, 2, 3}

def is_bipartite_partition (A1 A2 : Set ℕ) : Prop :=
  (A1 ∪ A2 = A) ∧ (A1 ∩ A2 = ∅)

def bipartite_partitions (A : Set ℕ) : Set (Set ℕ × Set ℕ) :=
  {p | is_bipartite_partition p.1 p.2}

noncomputable def count_bipartite_partitions (A : Set ℕ) :=
  (bipartite_partitions A).toFinset.card / 2

theorem bipartite_partition_count :
  count_bipartite_partitions A = 14 :=
by
  sorry

end bipartite_partition_count_l188_188157


namespace num_values_satisfy_gg_eq_3_l188_188681

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem num_values_satisfy_gg_eq_3 :
  (Set {x : ℝ | -2 ≤ x ∧ x ≤ 5 ∧ g (g x) = 3}).card = 4 := 
by
  sorry

end num_values_satisfy_gg_eq_3_l188_188681


namespace arithmetic_geometric_sequence_l188_188520

/-- Given:
  * 1, a₁, a₂, 4 form an arithmetic sequence
  * 1, b₁, b₂, b₃, 4 form a geometric sequence
Prove that:
  (a₁ + a₂) / b₂ = 5 / 2
-/
theorem arithmetic_geometric_sequence (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (h_arith : 2 * a₁ = 1 + a₂ ∧ 2 * a₂ = a₁ + 4)
  (h_geom : b₁ * b₁ = b₂ ∧ b₁ * b₂ = b₃ ∧ b₂ * b₂ = b₃ * 4) :
  (a₁ + a₂) / b₂ = 5 / 2 :=
sorry

end arithmetic_geometric_sequence_l188_188520


namespace k_values_count_l188_188126

-- Definition of prerequisites
def k_valid (k : ℕ) : Prop :=
  k > 1 ∧ ∃ n : ℕ, k * (2 * n + k - 1) = 4000

-- Theorem stating the problem
theorem k_values_count : {k : ℕ // k_valid k}.set.count = 3 :=
sorry

end k_values_count_l188_188126


namespace eighth_term_is_79_l188_188294

variable (a d : ℤ)

def fourth_term_condition : Prop := a + 3 * d = 23
def sixth_term_condition : Prop := a + 5 * d = 51

theorem eighth_term_is_79 (h₁ : fourth_term_condition a d) (h₂ : sixth_term_condition a d) : a + 7 * d = 79 :=
sorry

end eighth_term_is_79_l188_188294


namespace infinite_arithmetic_sequence_exclusion_of_triangular_numbers_l188_188595

/-- A type representing natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n - 1) / 2

theorem infinite_arithmetic_sequence_exclusion_of_triangular_numbers :
  (∀ n : ℕ, ∃ k : ℕ, ¬ (3 * k + 2 = triangular_number n)): true :=
sorry

end infinite_arithmetic_sequence_exclusion_of_triangular_numbers_l188_188595


namespace value_of_b_in_triangle_l188_188594

variable (A B C a b c : Real)
variable (sin : ℝ → ℝ)
variable (pi : Real)

open Real

theorem value_of_b_in_triangle (hA : A = pi / 3)
                               (hB : B = pi / 4)
                               (ha : a = 3 * sqrt 2)
                               (law_of_sines : a / sin(A) = b / sin(B)) :
                               b = 2 * sqrt 3 :=
by
  sorry

end value_of_b_in_triangle_l188_188594


namespace max_distance_from_circle_to_line_l188_188456

-- Definitions based on the given conditions
def circle_eq (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 5
def line_eq (x y : ℝ) : Prop := 2 * x - y + 9 = 0

-- The maximum distance from a point on the circle to the line
theorem max_distance_from_circle_to_line : 
  ∃ P : ℝ × ℝ, circle_eq P.1 P.2 ∧ ∃ Q : ℝ × ℝ, line_eq Q.1 Q.2 ∧ 
  (∀ (x y : ℝ), circle_eq x y → 2 * x - y + 9 ≠ 0 → ∥(x, y) - (Q.1, Q.2)∥ ≤ 3 * real.sqrt 5) := 
sorry

end max_distance_from_circle_to_line_l188_188456


namespace omega_value_l188_188615

noncomputable def f (ω : ℝ) (k : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x - Real.pi / 6) + k

theorem omega_value (ω k : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, f ω k x ≤ f ω k (Real.pi / 3)) → ω = 8 :=
by sorry

end omega_value_l188_188615


namespace solve_by_completing_square_l188_188646

noncomputable def d : ℤ := -5
noncomputable def e : ℤ := 10

theorem solve_by_completing_square :
  ∃ d e : ℤ, (x^2 - 10 * x + 15 = 0 ↔ (x + d)^2 = e) ∧ (d + e = 5) :=
by
  use -5, 10
  split
  -- First part: Show the equivalence of equations
  sorry
  -- Second part: Show d + e = 5
  refl

end solve_by_completing_square_l188_188646


namespace vector_magnitude_l188_188542

def a (x : ℝ) : ℝ × ℝ := (4^x, 2^x)
def b (x : ℝ) : ℝ × ℝ := (1, (2^x - 2) / 2^x)

theorem vector_magnitude (x : ℝ) (h : (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0) : 
  real.sqrt ((a x).1 - (b x).1)^2 + ((a x).2 - (b x).2)^2 = 2 := 
by
  sorry

end vector_magnitude_l188_188542


namespace skyscraper_anniversary_l188_188194

theorem skyscraper_anniversary 
  (years_since_built : ℕ)
  (target_years : ℕ)
  (years_before_200th : ℕ)
  (years_future : ℕ) 
  (h1 : years_since_built = 100) 
  (h2 : target_years = 200 - 5) 
  (h3 : years_future = target_years - years_since_built) : 
  years_future = 95 :=
by
  sorry

end skyscraper_anniversary_l188_188194


namespace obtuse_triangle_probability_l188_188480

/-- 
Given four points chosen uniformly at random on a circle,
prove that the probability that no three of these points form an obtuse triangle with the circle's center is 3/32.
-/
theorem obtuse_triangle_probability : 
  let points := 4 in
  ∀ (P : Fin points → ℝ × ℝ) (random_uniform : ∀ i, random.uniform_on_circle i),
  (probability (no_three_form_obtuse_triangle P) = 3 / 32) :=
sorry

end obtuse_triangle_probability_l188_188480


namespace sum_of_digits_of_product_l188_188623

open BigOperators

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_product :
  ∃ (c d : ℕ), c = 777 ∧ d = 444 ∧ digit_sum (7 * c * d) = 27 :=
begin
  use 777,
  use 444,
  split,
  { refl },
  split,
  { refl },
  { sorry }
end

end sum_of_digits_of_product_l188_188623


namespace remaining_digits_product_l188_188325

theorem remaining_digits_product (a b c : ℕ)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (b + c) % 10 = a % 10)
  (h3 : (c + a) % 10 = b % 10) :
  ((a * b * c) % 1000 = 0 ∨
   (a * b * c) % 1000 = 250 ∨
   (a * b * c) % 1000 = 500 ∨
   (a * b * c) % 1000 = 750) :=
sorry

end remaining_digits_product_l188_188325


namespace sequence_of_ones_F_zero_arithmetic_l188_188235

noncomputable def F (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (∑ i in range n, (-1)^i * a (i + 1) * binomial n i)

theorem sequence_of_ones (a : ℕ → ℕ) (h : ∀ m, a m = 1) (n : ℕ) (h_n : 2 ≤ n):
  F a n = 0 :=
sorry

theorem F_zero_arithmetic (a : ℕ → ℕ) (h : ∀ n, 2 ≤ n → F a n = 0) :
  ∀ (n m : ℕ), a n = a m + a (m + 1) - a (m + 2) :=
sorry

end sequence_of_ones_F_zero_arithmetic_l188_188235


namespace december_fraction_of_yearly_sales_l188_188606

variable {A : ℝ} -- A represents the average monthly sales total for Jan through Nov

theorem december_fraction_of_yearly_sales 
  (h : ∃ (A : ℝ), ∀ (j n) : ℕ, j = 7*n → x = 11*A + 7*A → 
    y = (7*A) / (11*A + 7*A)) : 
  y = (7/18) := 
sorry

end december_fraction_of_yearly_sales_l188_188606


namespace largest_possible_median_l188_188136

theorem largest_possible_median (l : List ℕ) (h1 : l.length = 10) 
  (h2 : ∀ x ∈ l, 0 < x) (exists6l : ∃ l1 : List ℕ, l1 = [3, 4, 5, 7, 8, 9]) :
  ∃ median_val : ℝ, median_val = 8.5 := 
sorry

end largest_possible_median_l188_188136


namespace midpoint_trajectory_of_intersecting_line_l188_188526

theorem midpoint_trajectory_of_intersecting_line 
    (h₁ : ∀ x y, x^2 + 2 * y^2 = 4) 
    (h₂ : ∀ M: ℝ × ℝ, M = (4, 6)) :
    ∃ x y, (x-2)^2 / 22 + (y-3)^2 / 11 = 1 :=
sorry

end midpoint_trajectory_of_intersecting_line_l188_188526


namespace arithmetic_sequence_n_eq_8_l188_188237

theorem arithmetic_sequence_n_eq_8 (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) :
  (a 1 = 1) →
  (∀ k, a (k + 1) = a k + 2) →
  S (n + 2) - S n = 36 →
  S n = ∑ i in range (n + 1), a (i + 1) →
  S (n + 2) = ∑ i in range (n + 3), a (i + 1) →
  n = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end arithmetic_sequence_n_eq_8_l188_188237


namespace path_length_PQ_l188_188317

theorem path_length_PQ (PQ : ℕ) (h : PQ = 73) : 3 * PQ = 219 := by
  rw [h]
  norm_num

end path_length_PQ_l188_188317


namespace suitable_sampling_method_l188_188395

theorem suitable_sampling_method (elderly middle_aged young sample : ℕ) 
    (h1 : elderly = 28) 
    (h2 : middle_aged = 54) 
    (h3 : young = 81) 
    (h4 : sample = 36) : 
    ∃ method : String, method = "Remove one elderly person first, then use stratified sampling" :=
by 
  use "Remove one elderly person first, then use stratified sampling"
  sorry

end suitable_sampling_method_l188_188395


namespace mutually_exclusive_not_necessarily_complementary_and_complementary_definitely_mutually_exclusive_l188_188720

variable {Ω : Type*} (P : MeasureTheory.ProbabilityMeasure Ω)
variable {A B : Set Ω}

/-- Two events are mutually exclusive if their intersection is empty. -/
def mutually_exclusive (A B : Set Ω) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if they are disjoint and together cover the entire probability space. -/
def complementary_events (A B : Set Ω) : Prop :=
  mutually_exclusive A B ∧ (A ∪ B = set.univ)

theorem mutually_exclusive_not_necessarily_complementary_and_complementary_definitely_mutually_exclusive
  (A B : Set Ω) :
  ¬(complementary_events A B → mutually_exclusive A B) ∧ (mutually_exclusive A B → complementary_events A B) :=
begin
  sorry
end

end mutually_exclusive_not_necessarily_complementary_and_complementary_definitely_mutually_exclusive_l188_188720


namespace area_of_triangle_PTS_l188_188779

theorem area_of_triangle_PTS (PQ QR : ℝ) (hPQ : PQ = 5) (hQR : QR = 12) :
    ∃ (T : ℝ × ℝ), let PS := real.sqrt (PQ^2 + QR^2),
                       QT := (PQ * QR) / PS in
    let PT := (QT / QR) * PS in
    (1/2) * PT * QT = 150 / 13 :=
by
  sorry

end area_of_triangle_PTS_l188_188779


namespace total_surface_area_of_hemisphere_l188_188674

def base_area_of_hemisphere (r : ℝ) : Prop :=
  π * r^2 = 225 * π

theorem total_surface_area_of_hemisphere (r : ℝ) (h : base_area_of_hemisphere r) : 
  2 * π * r^2 + π * r^2 = 675 * π :=
by
  intro
  sorry

end total_surface_area_of_hemisphere_l188_188674


namespace infinitely_many_solutions_eq_l188_188571

theorem infinitely_many_solutions_eq {a b : ℝ} 
  (H : ∀ x : ℝ, a * (a - x) - b * (b - x) = 0) : a = b :=
sorry

end infinitely_many_solutions_eq_l188_188571


namespace palindromes_with_seven_percentage_l188_188769

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def contains_digit (n digit : ℕ) : Prop :=
  digit.to_string ∈ n.to_string

theorem palindromes_with_seven_percentage :
  let palindromes := {n | 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n}
  let palindromes_with_7 := {n | 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n ∧ (contains_digit n 7)}
  ((palindromes_with_7.to_finset.card : ℚ) / palindromes.to_finset.card * 100) = 19 :=
by
  sorry

end palindromes_with_seven_percentage_l188_188769


namespace find_m_l188_188509

-- Define the lines and the condition of perpendicularity
def l1 (m : ℝ) : ℝ → ℝ → Prop := 
  λ x y, (m + 3) * x + 4 * y = 5

def l2 (m : ℝ) : ℝ → ℝ → Prop := 
  λ x y, 2 * x + (m + 5) * y = 8

def perpendicular (m : ℝ) : Prop := 
  -(m + 3) / 4 * (-2 / (m + 5)) = -1

-- The theorem stating that m = -13/3 satisfies the perpendicularity condition
theorem find_m (m : ℝ) : 
  perpendicular m → m = -13 / 3 := 
sorry

end find_m_l188_188509


namespace domain_and_range_of_f_not_center_of_symmetry_f_is_increasing_on_intervals_l188_188525

noncomputable def nearest_integer (x : ℝ) : ℤ :=
  if h : ∃ m : ℤ, m - 1/2 < x ∧ x ≤ m + 1/2 then Classical.some h else 0

def f (x : ℝ) := x - nearest_integer x

theorem domain_and_range_of_f : 
  (∀ x : ℝ, x ∈ set.univ) ∧ (∀ y : ℝ, y ∈ (set.Ioc (-1/2) 1/2)) :=
sorry

theorem not_center_of_symmetry (k : ℤ) : 
  ¬ (∀ x : ℝ, f(2*k - x) = -f(x)) :=
sorry

theorem f_is_increasing_on_intervals : 
  (∀ x y : ℝ, x ∈ set.Ioc 1/2 3/2 → y ∈ set.Ioc 1/2 3/2 → x < y → f(x) < f(y)) :=
sorry

end domain_and_range_of_f_not_center_of_symmetry_f_is_increasing_on_intervals_l188_188525


namespace general_formula_an_sum_first_n_terms_cn_l188_188515
-- First, we import the necessary libraries

-- Define the conditions from part a) as Lean definitions
variables {a b : ℕ → ℤ}

-- Arithmetic sequence (a_n = a_1 + (n - 1) * d)
def is_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n, a n = a 1 + (n - 1) * d

-- Geometric sequence (b_n = b_1 * q^(n - 1))
def is_geometric_seq (b : ℕ → ℤ) (q : ℤ) : Prop := ∀ n, b n = b 1 * q^(n - 1)

-- Define the additional conditions given in the problem
axiom b2 : b 2 = 3
axiom b3 : b 3 = 9
axiom a1_eq_b1 : a 1 = b 1
axiom a14_eq_b4 : a 14 = b 4

-- Problem (1): Find the general formula for a_n
theorem general_formula_an {d : ℤ} (h_arith : is_arithmetic_seq a d) : 
  a n = 2 * n - 1 := 
sorry

-- Additional definitions for Problem (2)
def c (n : ℕ) := a n + b n
def sum_first_n_terms (c : ℕ → ℤ) (n : ℕ) := ∑ i in finset.range n, c (i + 1)

-- Problem (2): Find the sum of the first n terms of the sequence {c_n}
theorem sum_first_n_terms_cn {d : ℤ} (h_arith : is_arithmetic_seq a d) (h_geom : is_geometric_seq b 3):
  sum_first_n_terms c n = n^2 - (1 / 2) * (1 - 3^n) :=
sorry

end general_formula_an_sum_first_n_terms_cn_l188_188515


namespace find_point_B_l188_188874

theorem find_point_B (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, -5)) 
  (ha : a = (2, 3)) 
  (hAB : B - A = 3 • a) : 
  B = (5, 4) := sorry

end find_point_B_l188_188874


namespace max_value_of_a_plus_b_max_value_attained_l188_188569

theorem max_value_of_a_plus_b (a b : ℝ) (h : 2 ^ a + 2 ^ b = 1) : a + b ≤ -2 :=
sorry

theorem max_value_attained (a b : ℝ) (h : 2 ^ a + 2 ^ b = 1) : a = b = -1 → (a + b = -2) :=
sorry

end max_value_of_a_plus_b_max_value_attained_l188_188569


namespace cos_beta_value_l188_188251

variable (α β : ℝ)
variable (h₁ : 0 < α ∧ α < π)
variable (h₂ : 0 < β ∧ β < π)
variable (h₃ : Real.sin (α + β) = 5 / 13)
variable (h₄ : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_value : Real.cos β = -16 / 65 := by
  sorry

end cos_beta_value_l188_188251


namespace probability_between_0_and_2_l188_188930

-- Define the normal distribution and probability conditions
noncomputable def normalDistribution (μ σ : ℝ) : ℝ → ℝ :=
  λ x, (1 / (σ * (real.sqrt (2 * π)))) * real.exp (-((x - μ)^2) / (2 * σ^2))

-- Define the conditions: X follows N(1, σ^2) and P(X < 0) = 0.2
constant σ : ℝ
axiom σ_pos : σ > 0
constant X : ℝ → ℝ
axiom X_follows_normal : X = normalDistribution 1 σ

def P (a b : ℝ) : ℝ :=
  ∫ x in a..b, X x

@[axiom] def P_X_lt_0 : P (-∞) 0 = 0.2 := sorry  -- Given

-- Proving the desired probability P(0 < X < 2)
theorem probability_between_0_and_2 :
  P 0 2 = 0.6 :=
sorry

end probability_between_0_and_2_l188_188930


namespace probability_Jane_Albert_same_committee_l188_188334

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def comb (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def totalWaysToFormCommittees : ℕ :=
  comb 6 3

def waysJaneAlbertTogether : ℕ :=
  comb 4 1

def probabilityJaneAlbertTogether : ℚ :=
  waysJaneAlbertTogether / totalWaysToFormCommittees

theorem probability_Jane_Albert_same_committee (prob : probabilityJaneAlbertTogether) : prob = 1 / 5 := sorry

end probability_Jane_Albert_same_committee_l188_188334


namespace product_of_sum_and_reciprocal_ge_four_l188_188640

theorem product_of_sum_and_reciprocal_ge_four (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
sorry

end product_of_sum_and_reciprocal_ge_four_l188_188640


namespace find_a_range_l188_188512

theorem find_a_range (a : ℝ) : 
  ((3 + a) * (a - 1) > 0) ∧ (4 + (a - 1) ^ 2 < 8) ∧ ((¬ ((3 + a) * (a - 1) > 0) ∨ ¬ (4 + (a - 1) ^ 2 < 8))) = false ∧ (¬ (4 + (a - 1) ^ 2 < 8) = false) → (-1 < a ∧ a ≤ 1) :=
begin
 sorry
end

end find_a_range_l188_188512


namespace units_digit_F_F_15_l188_188670

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem units_digit_F_F_15 : (fib (fib 15) % 10) = 5 := by
  sorry

end units_digit_F_F_15_l188_188670


namespace sanitizer_first_spray_kills_l188_188375

theorem sanitizer_first_spray_kills (x : ℝ) 
  (H1 : 0.25 ∈ Icc 0 1) -- The second spray kills 25% of germs
  (H2 : 0.05 ∈ Icc 0 1) -- 5% of the germs they kill are the same ones
  (H3 : ∀ y₁ y₂ y₃, y₁ + y₂ - y₃ = 0.70) -- After using both sprays, 30% of germs would be left
  (H4 : ∀ y₁ y₂, y₁ * y₂ = 0.05) -- The overlap, which is the product of individual kill rates
  : x = 0.4737 :=
by {
    -- We aren't interested in the internals of the proof, hence sorry
    sorry
}

end sanitizer_first_spray_kills_l188_188375


namespace range_of_a_l188_188885

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + a * x - 4 * a < 0) : a ∈ Icc (-16 : ℝ) 0 := 
by 
  sorry

end range_of_a_l188_188885


namespace units_digit_of_F_F_15_l188_188662

def F : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := F (n + 1) + F n

theorem units_digit_of_F_F_15 : (F (F 15)) % 10 = 5 := by
  have h₁ : ∀ n, F n % 10 = ([0, 1, 1, 2, 3, 5, 8, 3, 1, 4, 5, 9, 4, 3, 7, 0, 7, 7, 4, 1, 5, 6, 1, 7, 8, 5, 3, 8, 1, 9, 0, 9, 9, 8, 7, 5, 2, 7, 9, 6, 5, 1, 6, 7, 3, 0, 3, 3, 6, 9, 5, 4, 9, 3, 2, 5, 7, 2, 9, 1].take 60)!!.get_or_else (n % 60) 0 := sorry
  have h₂ : F 15 = 610 := by
    simp [F]; -- Proof for F 15 = 610
    sorry
  calc (F (F 15)) % 10
       = F 610 % 10 := by
         rw [h₂]
       ... = 5 := h₁ 610

end units_digit_of_F_F_15_l188_188662


namespace day_of_week_197_l188_188570

/-- Prove that given the 15th day of the year 2005 falls on a Wednesday, the 197th day of the same year also falls on a Wednesday. -/
theorem day_of_week_197 (fifteenth_day_is_wednesday : ∀ d : ℕ, d = 15 → nat.mod d 7 = 3) : ∀ d : ℕ, d = 197 → nat.mod d 7 = 3 :=
by
  intro d hd
  sorry

end day_of_week_197_l188_188570


namespace find_min_and_range_of_lambda_l188_188151

-- Definitions for the functions involved
def f (x k : ℝ) : ℝ := (x - k) * Real.exp x

def g (x k : ℝ) : ℝ := f x k + (D (λ y, f y k) x)

-- Main specification statement
theorem find_min_and_range_of_lambda (k : ℝ) (λ : ℝ) :
    (∀ (x ∈ Icc 1 2), f x k ≥ f 1 k ∨ f x k ≤ f 2 k ∨ f x k = -(Real.exp (k - 1)) ) →
    (∀ (k ∈ Icc (3 / 2 : ℝ) (5 / 2 : ℝ)), ∀ (x ∈ Icc 0 1), g x k ≥ λ ) →
    λ ≤ -2 * Real.exp 1 :=
sorry

end find_min_and_range_of_lambda_l188_188151


namespace initial_fee_correct_l188_188227

-- Define the relevant values
def initialFee := 2.25
def chargePerSegment := 0.4
def totalDistance := 3.6
def totalCharge := 5.85
noncomputable def segments := (totalDistance * (5 / 2))
noncomputable def costForDistance := segments * chargePerSegment

-- Define the theorem
theorem initial_fee_correct :
  totalCharge = initialFee + costForDistance :=
by
  -- Proof is omitted.
  sorry

end initial_fee_correct_l188_188227


namespace blue_pill_cost_l188_188802

/-- Bob takes one blue pill and one red pill per day. A blue pill costs $3 more than a red pill.
    Given that the total cost for 10 days of medication is $320, we prove that the cost of one blue pill is $17.5. -/
theorem blue_pill_cost 
    (number_of_days : ℕ)
    (total_cost : ℝ)
    (blue_red_difference : ℝ)
    (per_day_cost : ℝ)
    (blue_pill_cost : ℝ)
    (red_pill_cost : ℝ) :
    (number_of_days = 10) →
    (total_cost = 320) →
    (blue_red_difference = 3) →
    (per_day_cost = total_cost / number_of_days) →
    (per_day_cost = blue_pill_cost + red_pill_cost) →
    (blue_pill_cost = red_pill_cost + blue_red_difference) →
    blue_pill_cost = 17.5 :=
begin
    sorry
end

end blue_pill_cost_l188_188802


namespace cube_edge_length_l188_188555

theorem cube_edge_length (L : ℝ) : ∃ a : ℝ, a = (L * ((Real.sqrt 3) + 1)) / 2 :=
by
  sorry

end cube_edge_length_l188_188555


namespace inequality_hold_l188_188634

theorem inequality_hold (n : ℕ) (h1 : n > 1) : 1 + n * 2^((n - 1 : ℕ) / 2) < 2^n :=
by
  sorry

end inequality_hold_l188_188634


namespace channel_width_at_top_l188_188677

theorem channel_width_at_top 
  (area : ℝ) (bottom_width : ℝ) (depth : ℝ) 
  (H1 : bottom_width = 6) 
  (H2 : area = 630) 
  (H3 : depth = 70) : 
  ∃ w : ℝ, (∃ H : w + 6 > 0, area = 1 / 2 * (w + bottom_width) * depth) ∧ w = 12 :=
by
  sorry

end channel_width_at_top_l188_188677


namespace gasoline_storage_l188_188977

noncomputable def total_distance : ℕ := 280 * 2

noncomputable def miles_per_segment : ℕ := 40

noncomputable def gasoline_consumption : ℕ := 8

noncomputable def total_segments : ℕ := total_distance / miles_per_segment

noncomputable def total_gasoline : ℕ := total_segments * gasoline_consumption

noncomputable def number_of_refills : ℕ := 14

theorem gasoline_storage (storage_capacity : ℕ) (h : number_of_refills * storage_capacity = total_gasoline) :
  storage_capacity = 8 :=
by
  sorry

end gasoline_storage_l188_188977


namespace probability_x_lt_2y_l188_188774

noncomputable def probability_x_lt_2y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1/2) * 4 * 2
  let area_rectangle : ℚ := 4 * 2
  (area_triangle / area_rectangle)

theorem probability_x_lt_2y (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : 0 ≤ y) (h4 : y ≤ 2) :
  probability_x_lt_2y_in_rectangle = 1/2 := by
  sorry

end probability_x_lt_2y_l188_188774


namespace trevor_counted_77_coins_l188_188333

-- Define the problem conditions
variables (quarters : ℕ) (total : ℕ)

-- Conditions
def condition1 := quarters = 29
def condition2 := total = quarters + 48

-- The theorem stating the required proof problem
theorem trevor_counted_77_coins (quarters total : ℕ) : 
  condition1 quarters ∧ 
  condition2 quarters total → 
  total = 77 :=
by 
  intros h
  cases h
  rw [h_left, h_right]
  exact rfl

end trevor_counted_77_coins_l188_188333


namespace units_digit_of_FF15_is_5_l188_188659

noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem units_digit_of_FF15_is_5 : (fib (fib 15)) % 10 = 5 :=
by
  sorry

end units_digit_of_FF15_is_5_l188_188659


namespace dice_four_even_four_odd_probability_l188_188824

theorem dice_four_even_four_odd_probability :
  let total_cases := (nat.choose 8 4) * (1 / 2 : ℝ) ^ 8
  (total_cases = 35 / 128 : ℝ) :=
by
  have total_cases_eq : total_cases = (nat.choose 8 4) * (1 / 256 : ℝ)
  calc
  sorry

end dice_four_even_four_odd_probability_l188_188824


namespace AI_eq_IT_l188_188161

open Triangle Circle Geometry Point Line Segment

-- Definitions based on the problem conditions
variable (A B C I A₁ D E T : Point)
variable (AIA₁D : ∃ (l1 : Line), is_parallel l1 (line_through_points B C) ∧ line_through_points A A₁ = l1)
variable (A1CircleO : ∃ (c1 : Circle), circumcircle (triangle A B C) = c1)
variable (ID : ∃ (l2 : Line), line_through_points A I = l2 ∧ (l2 ∩ line_through_points B C = set_of_point D))
variable (E : ∃ (c2 : Circle), incircle (triangle A B C) = c2)
variable (E_on_circumcircleADE : E = tangent_point_of_incircle (triangle A B C) (line_through_points B C) c2)
variable (T_on_A1circumcircleADE : ∃ (c3 : Circle), circumcircle (triangle A D E) = c3 ∧ (A₁E ∩ c3 = set_of_point T))

-- The theorem to prove
theorem AI_eq_IT :
  AI = IT :=
sorry

end AI_eq_IT_l188_188161


namespace price_of_first_metal_l188_188767

theorem price_of_first_metal (x : ℝ) 
  (h1 : (x + 96) / 2 = 82) : 
  x = 68 :=
by sorry

end price_of_first_metal_l188_188767


namespace minimum_coins_to_identify_pouch_l188_188773

theorem minimum_coins_to_identify_pouch
  (pouches : Fin 5 → Fin 30 → ℕ)
  (total_coins : ∀ i, ∑ j, pouches i j = 30)
  (gold_pouch : ∃ i, ∀ j, pouches i j = 1 ∧ ¬(∃ k, k ≠ i ∧ pouches k j = 1))
  (silver_pouch : ∃ i, ∀ j, pouches i j = 2 ∧ ¬(∃ k, k ≠ i ∧ pouches k j = 2))
  (bronze_pouch : ∃ i, ∀ j, pouches i j = 3 ∧ ¬(∃ k, k ≠ i ∧ pouches k j = 3))
  (mixed_pouches : ∃ i j, i ≠ j ∧ ∀ k, pouches i k = pouches j k ∧ pouches i k = 10) :
  ∃ chosen_pouches : Fin 5 → bool, ∑ i, if chosen_pouches i then 1 else 0 = 5 →
  ∃ idetected_pouch : Fin 5, ∀ j, chosen_pouches j → idetected_pouch = j :=
sorry

end minimum_coins_to_identify_pouch_l188_188773


namespace magic_coin_l188_188351

theorem magic_coin (m n : ℕ) (h_m_prime: Nat.gcd m n = 1)
  (h_prob : (m : ℚ) / n = 1 / 158760): m + n = 158761 := by
  sorry

end magic_coin_l188_188351


namespace jana_walks_2_25_miles_in_45_minutes_l188_188967

-- Define the conditions given in the problem
def time_to_walk_d1 : ℝ := 30 -- minutes
def distance_d1 : ℝ := 1.5 -- miles
def time_to_walk_d2 : ℝ := 45 -- minutes

-- Define the expected distance for the proof
def expected_distance : ℝ := 2.25 -- miles

-- Formalize the rate calculation
def walking_rate : ℝ := distance_d1 / time_to_walk_d1

-- Formalize the distance calculation function
def calculate_distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

-- The theorem to prove that the distance walked in 45 minutes is 2.25 miles
theorem jana_walks_2_25_miles_in_45_minutes :
  calculate_distance walking_rate time_to_walk_d2 = expected_distance := by
  sorry

end jana_walks_2_25_miles_in_45_minutes_l188_188967


namespace exists_another_perfect_seating_l188_188211

-- Definitions as per the conditions
def is_perfect_seating (n : ℕ) (G : SimpleGraph (Fin n)) (S : Fin n → Fin n) : Prop :=
  (∀ i, G.Adj (S i) (S (i + 1))) ∧ (∀ i, G.Adj (S i) (S (i - 1)))

def distinct_hc (n : ℕ) (G : SimpleGraph (Fin n)) (C₁ C₂ : Fin n → Fin n) :=
  (C₁ ≠ C₂) ∧ (∀ k, (C₁ k → C₂) ≠ (C₁ → C₂ k)) ∧ (∀ k, ((C₁ k) ≠ (C₂ (k + 1))) ∧ ((C₁ k) ≠ (C₂ (k - 1))))

-- The final theorem statement to prove existence of another perfect seating
theorem exists_another_perfect_seating (n : ℕ) (G : SimpleGraph (Fin n)) (S : Fin n → Fin n) :
  is_perfect_seating n G S →
  ∃ S', is_perfect_seating n G S' ∧ distinct_hc n G S S' :=
sorry

end exists_another_perfect_seating_l188_188211


namespace probability_two_digit_seq_num_gt_45_l188_188742

-- Define what a sequential number is in Lean
def is_sequential (n : ℕ) : Prop :=
  (n < 10 ∨ (n / 10 < n % 10)) -- digit condition for sequential numbers

-- Define the set of two-digit sequential numbers
def two_digit_sequential_numbers : set ℕ := {n | 10 ≤ n ∧ n < 100 ∧ is_sequential n}

-- Count of two-digit sequential numbers
def count_two_digit_sequential_numbers : ℕ :=
  finset.card (finset.filter (λ n => is_sequential n) (finset.range 100)) - finset.card (finset.range 10)

-- Define the set of two-digit sequential numbers greater than 45
def two_digit_seq_num_gt_45 : set ℕ := {n ∈ two_digit_sequential_numbers | n > 45}

-- The proof statement: probability of randomly selecting a two-digit sequential number greater than 45
theorem probability_two_digit_seq_num_gt_45 :
  count_two_digit_sequential_numbers = 36 → 
  ∃ (s : finset ℕ) (h : ∀ x ∈ s, x ∈ two_digit_sequential_numbers), s.card = 14 →
  (14 / 36 : ℚ) = 7 / 18 :=
sorry -- proof is not required

end probability_two_digit_seq_num_gt_45_l188_188742


namespace units_digit_of_fibonacci_f_15_l188_188667

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n + 1) + fibonacci n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fibonacci_f_15 :
  units_digit (fibonacci (fibonacci 15)) = 5 := 
sorry

end units_digit_of_fibonacci_f_15_l188_188667


namespace probability_x_lt_2y_l188_188775

noncomputable def probability_x_lt_2y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1/2) * 4 * 2
  let area_rectangle : ℚ := 4 * 2
  (area_triangle / area_rectangle)

theorem probability_x_lt_2y (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : 0 ≤ y) (h4 : y ≤ 2) :
  probability_x_lt_2y_in_rectangle = 1/2 := by
  sorry

end probability_x_lt_2y_l188_188775


namespace calculate_neg_pow_mul_l188_188412

theorem calculate_neg_pow_mul (a : ℝ) : -a^4 * a^3 = -a^7 := by
  sorry

end calculate_neg_pow_mul_l188_188412


namespace largest_multiple_of_20_condition_compute_fraction_l188_188304

theorem largest_multiple_of_20_condition (m : ℕ) 
  (h1: m % 20 = 0)
  (h2: ∀ d ∈ digits 10 m, d = 8 ∨ d = 0):
  m = 8880 :=
sorry

theorem compute_fraction (h: largest_multiple_of_20_condition m) :
  m / 20 = 444 :=
sorry

end largest_multiple_of_20_condition_compute_fraction_l188_188304


namespace passengers_logan_approx_1_6_l188_188927

noncomputable def Kennedy := (1 / 3) * 38.3
noncomputable def Miami := (1 / 2) * Kennedy
noncomputable def Logan := Miami / 4

theorem passengers_logan_approx_1_6 : Logan ≈ 1.6 :=
by
  sorry

end passengers_logan_approx_1_6_l188_188927


namespace findPrincipal_l188_188354

variable (P : ℝ) (R : ℝ := 10) (T : ℝ := 2) (D : ℝ := 51)

def simpleInterest (P R T : ℝ) : ℝ := P * R * T / 100

def compoundInterest (P R T : ℝ) : ℝ := P * (1 + R/100)^T - P

theorem findPrincipal (P : ℝ) (h : compoundInterest P R T - simpleInterest P R T = D) : P = 5100 :=
by
  sorry

end findPrincipal_l188_188354


namespace line_passes_through_fixed_point_l188_188537

theorem line_passes_through_fixed_point (k : ℝ) : (k * 2 - 1 + 1 - 2 * k = 0) :=
by
  sorry

end line_passes_through_fixed_point_l188_188537


namespace sequences_equality_l188_188622

def pos_int (n : ℕ) : Prop := ∃ m : ℕ, m > 0 ∧ n = m

def recurrence_condition (a : fin n → ℝ) (u v : fin (n+1) → ℝ) : Prop :=
  u 0 = 1 ∧ u 1 = 1 ∧ v 0 = 1 ∧ v 1 = 1 ∧
  (∀ k : fin (n-1), u (k + 2) = u (k + 1) + a k * u k) ∧
  (∀ k : fin (n-1), v (k + 2) = v (n - k - 1) + a (n - k - 2) * v (n - k - 2))

theorem sequences_equality {n : ℕ} (h : pos_int n)
(a : fin n → ℝ) (u v : fin (n+1) → ℝ)
(h_rec : recurrence_condition a u v) : 
u n = v n :=
sorry

end sequences_equality_l188_188622


namespace graph_of_equation_is_two_intersecting_lines_l188_188434

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ (x y : ℝ), (x - y)^2 = 3 * x^2 - y^2 ↔ 
  (x = (1 - Real.sqrt 5) / 2 * y) ∨ (x = (1 + Real.sqrt 5) / 2 * y) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l188_188434


namespace abs_fraction_eq_sqrt_three_over_two_l188_188566

variable (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b)

theorem abs_fraction_eq_sqrt_three_over_two (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b) : 
  |(a + b) / (a - b)| = Real.sqrt (3 / 2) := by
  sorry

end abs_fraction_eq_sqrt_three_over_two_l188_188566


namespace intersection_of_M_and_N_l188_188903

def M : Set ℝ := { x : ℝ | x^2 - x > 0 }
def N : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | x > 1 } :=
by
  sorry

end intersection_of_M_and_N_l188_188903


namespace sqrt_seq_limit_l188_188388

noncomputable def a : ℕ → ℝ
| 0     := 10
| (n+1) := a n + 18 * n + 10

def floor (x : ℝ) : ℤ := int.floor x

theorem sqrt_seq_limit :
  tendsto (λ n, real.sqrt (a n) - ↑(floor (real.sqrt (a n)))) at_top (𝓝 (1 / 6)) :=
sorry

end sqrt_seq_limit_l188_188388


namespace time_for_A_and_C_l188_188360

variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := A + B = 1 / 8
def condition2 : Prop := B + C = 1 / 12
def condition3 : Prop := A + B + C = 1 / 6

theorem time_for_A_and_C (h1 : condition1 A B)
                        (h2 : condition2 B C)
                        (h3 : condition3 A B C) :
  1 / (A + C) = 8 :=
sorry

end time_for_A_and_C_l188_188360


namespace volume_of_tetrahedron_l188_188012

-- Define the conditions of the problem
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (dist : A → B → ℝ)
variable (R : ℝ)
variables (AD : dist A D = 3) 
variable (cos_BAC : cos (angle B A C) = 4/5)
variables (cos_BAD : cos (angle B A D) = 1/Real.sqrt 2)
variables (cos_CAD : cos (angle C A D) = 1/Real.sqrt 2)

-- Define the theorem to prove the volume of the tetrahedron
theorem volume_of_tetrahedron (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (dist : A → B → ℝ) 
  (AD : dist A D = 3)
  (cos_BAC : cos (angle B A C) = 4/5)
  (cos_BAD : cos (angle B A D) = 1/Real.sqrt 2)
  (cos_CAD : cos (angle C A D) = 1/Real.sqrt 2) : 
  volume (tetrahedron A B C D dist) = 18/5 :=
sorry

end volume_of_tetrahedron_l188_188012


namespace probability_no_obtuse_triangle_correct_l188_188474

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l188_188474


namespace compare_a_b_c_l188_188116

theorem compare_a_b_c :
  let a := 2.5 ^ (-3/2)
  let b := Real.logBase (2/3) 2.5
  let c := 2.5 ^ (-2)
  a > c ∧ c > b := by
   sorry

end compare_a_b_c_l188_188116


namespace num_distinct_terms_in_expansion_l188_188049

theorem num_distinct_terms_in_expansion : 
  ∀ (a b : ℚ), 
  distinct_terms (expand (((2 * a + 4 * b) ^ 2 * (2 * a - 4 * b) ^ 2) ^ 3)) = 7 := 
by
  intros a b
  sorry

end num_distinct_terms_in_expansion_l188_188049


namespace gcd_of_powers_of_two_l188_188340

noncomputable def m := 2^2048 - 1
noncomputable def n := 2^2035 - 1

theorem gcd_of_powers_of_two : Int.gcd m n = 8191 := by
  sorry

end gcd_of_powers_of_two_l188_188340


namespace find_b_and_line_K_l188_188043

theorem find_b_and_line_K (b c : ℝ) : 
  (∀ x : ℝ, let y := x^2 + b*x + 4 in
   let dydx := deriv (λ x, x^2 + b*x + 4) in
   let slope_at_x_1 := 2 * x + b in
   let slope_K := 2 * slope_at_x_1 in
   slope_K = 1 → b = -1.5 ∧ (∀ x, x = 0 → (∃ K : ℝ → ℝ, K x = x))) :=
by
  sorry

end find_b_and_line_K_l188_188043


namespace catch_up_time_l188_188944

theorem catch_up_time (x : ℕ) : 240 * x = 150 * x + 12 * 150 := by
  sorry

end catch_up_time_l188_188944


namespace unique_polynomial_l188_188091

def satisfies_condition (P : ℝ[X]) : Prop :=
  ∀ x : ℝ, P.eval (2 * x) = (polynomial.derivative P).eval x * (polynomial.derivative (polynomial.derivative P)).eval x

theorem unique_polynomial {P : ℝ[X]} (h : satisfies_condition P) : P = (4 / 9) • polynomial.X ^ 3 :=
by
  sorry

end unique_polynomial_l188_188091


namespace circle_properties_l188_188148

noncomputable def circle_eq (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0
noncomputable def line_eq (x y : ℝ) := x + 2*y - 4 = 0
noncomputable def perpendicular (x1 y1 x2 y2 : ℝ) := 
  (x1 * x2 + y1 * y2 = 0)

theorem circle_properties (m : ℝ) (x1 y1 x2 y2 : ℝ) :
  (∀ x y, circle_eq x y m) →
  (∀ x, line_eq x (y1 + y2)) →
  perpendicular (4 - 2*y1) y1 (4 - 2*y2) y2 →
  m = 8 / 5 ∧ 
  (∀ x y, (x^2 + y^2 - (8 / 5) * x - (16 / 5) * y = 0) ↔ 
           (x - (4 - 2*(16/5))) * (x - (4 - 2*(16/5))) + (y - (16/5)) * (y - (16/5)) = 5 - (8/5)) :=
sorry

end circle_properties_l188_188148


namespace average_grade_of_male_students_l188_188296

theorem average_grade_of_male_students (M : ℝ) (H1 : (90 : ℝ) = (8 + 32 : ℝ) / 40) 
(H2 : (92 : ℝ) = 32 / 40) :
  M = 82 := 
sorry

end average_grade_of_male_students_l188_188296


namespace topping_cost_l188_188556

noncomputable def cost_of_topping (ic_cost sundae_cost number_of_toppings: ℝ) : ℝ :=
(sundae_cost - ic_cost) / number_of_toppings

theorem topping_cost
  (ic_cost : ℝ)
  (sundae_cost : ℝ)
  (number_of_toppings : ℕ)
  (h_ic_cost : ic_cost = 2)
  (h_sundae_cost : sundae_cost = 7)
  (h_number_of_toppings : number_of_toppings = 10) :
  cost_of_topping ic_cost sundae_cost number_of_toppings = 0.5 :=
  by
  -- Proof will be here
  sorry

end topping_cost_l188_188556


namespace repeating_decimal_sum_l188_188345

theorem repeating_decimal_sum (x : ℚ)
  (hx : x = 23 / 99) :
  x.denom + x.num = 122 :=
begin
  sorry
end

end repeating_decimal_sum_l188_188345


namespace find_sum_of_coordinates_of_point_D_l188_188274

noncomputable def point_M : (ℝ × ℝ) := (-2, 3)
noncomputable def point_C : (ℝ × ℝ) := (-6, 1)
noncomputable def point_D : (ℝ × ℝ) := 
  by 
    let x_d = 5  -- From midpoint formula calculation
    let y_d = 2  -- From midpoint formula calculation
    exact (2, 5)

theorem find_sum_of_coordinates_of_point_D :
  let M := point_M
  let C := point_C
  let D : (ℝ × ℝ) := point_D
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  (D.1 + D.2) = 7 :=
by
  intro h_midpoint
  -- The proof is skipped
  sorry

end find_sum_of_coordinates_of_point_D_l188_188274


namespace sum_ineq_l188_188242

theorem sum_ineq (α : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ i, 0 < α i) 
  (h2 : ∑ i in (Finset.range n), α i = 1) 
  (h3 : 2 ≤ n) :
  ∑ i in (Finset.range n), α i / (2 - α i) ≥ n / (2 * n - 1) :=
by
  sorry

end sum_ineq_l188_188242


namespace general_term_formula_sum_of_first_n_terms_l188_188500

-- Definitions for arithmetic sequence a_n with common difference d and given conditions
def a_n (n : ℕ) : ℤ := 2 * n - 1

-- Define b_n sequence based on a_n
def b_n (n : ℕ) : ℚ := 1 / ((a_n n) * (a_n (n + 1)))

-- Define the sum of the first n terms of b_n
def T_n (n : ℕ) : ℚ := (1 / 2) * (1 - (1 / (2 * n + 1)))

-- Proof statements
theorem general_term_formula (n : ℕ) : a_n n = 2 * n - 1 :=
by
  -- Further proof would be required here
  sorry

theorem sum_of_first_n_terms (n : ℕ) : (∑ i in Finset.range n, b_n i) = T_n n :=
by
  -- Further proof would be required here
  sorry

end general_term_formula_sum_of_first_n_terms_l188_188500


namespace intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l188_188538

-- Definitions based on the conditions
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x - a * y + 2 = 0
def perpendicular (a : ℝ) : Prop := a = 0
def parallel (a : ℝ) : Prop := a = 1 ∨ a = -1

-- Theorem 1: Intersection point when a = 0 is (-2, 2)
theorem intersection_point_zero_a_0 :
  ∀ x y : ℝ, l₁ 0 x y → l₂ 0 x y → (x, y) = (-2, 2) := 
by
  sorry

-- Theorem 2: Line l₁ always passes through (0, 2)
theorem l₁_passes_through_0_2 :
  ∀ a : ℝ, l₁ a 0 2 := 
by
  sorry

-- Theorem 3: l₁ is perpendicular to l₂ implies a = 0
theorem l₁_perpendicular_l₂ :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ∀ m n, (a * m + (n / a) = 0)) → (a = 0) :=
by
  sorry

-- Theorem 4: l₁ is parallel to l₂ implies a = 1 or a = -1
theorem l₁_parallel_l₂ :
  ∀ a : ℝ, parallel a → (a = 1 ∨ a = -1) :=
by
  sorry

end intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l188_188538


namespace ball_height_at_36_l188_188750

def ball_height (t : ℝ) : ℝ := -16 * t^2 + 96 * t

theorem ball_height_at_36 : ∃ t : ℝ, ball_height t = 36 ∧ t = 1.5 :=
by
  use 1.5
  simp [ball_height]
  norm_num
  exact ⟨rfl, rfl⟩

end ball_height_at_36_l188_188750


namespace length_MN_eq_semiperimeter_l188_188433

variables (A B C M N : Point)
variables (h_triangle : Triangle A B C)
variables (h_M : FootPerpendicular A (ExteriorAngleBisector B))
variables (h_N : FootPerpendicular A (ExteriorAngleBisector C))

theorem length_MN_eq_semiperimeter :
  length (segment M N) = semiperimeter (triangle A B C) :=
sorry

end length_MN_eq_semiperimeter_l188_188433


namespace polynomial_solution_l188_188088

noncomputable def is_polynomial_solution (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, P.eval (2 * x) = (Polynomial.derivative P).eval x * (Polynomial.derivative (Polynomial.derivative P)).eval x

theorem polynomial_solution :
  ∀ P : Polynomial ℝ, is_polynomial_solution P → 
    P = Polynomial.C (4/9):ℝ * Polynomial.X ^ 3 + 
        Polynomial.C (3 * 4/9):ℝ * Polynomial.X ^ 2 + 
        Polynomial.C c * Polynomial.X ∧ 
        c ∈ ℝ := by
  sorry

end polynomial_solution_l188_188088


namespace unknown_number_is_105_l188_188341

theorem unknown_number_is_105 :
  ∃ x : ℝ, x^2 + 94^2 = 19872 ∧ x = 105 :=
by
  sorry

end unknown_number_is_105_l188_188341


namespace Problem1_Problem2_l188_188905

universe u

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (C : Set ℝ)

-- Problem 1
theorem Problem1 (hU : U = Set.univ)
                 (hA : A = {x : ℝ | x^2 - 3 * x - 18 ≥ 0})
                 (hB : B = {x : ℝ | (x + 5) / (x - 14) ≤ 0}) :
  ((Set.univ \ B) ∩ A) = Set.Ioo (-∞) (-5) ∪ Set.Ici 14 := 
  sorry

-- Problem 2
theorem Problem2 (a : ℝ)
                 (hB : B = {x : ℝ | (x + 5) / (x - 14) ≤ 0})
                 (hC : C = {x : ℝ | 2 * a < x ∧ x < a + 1})
                 (hBC : B ∩ C = C) :
  -5 / 2 ≤ a := 
  sorry

end Problem1_Problem2_l188_188905


namespace determine_a_l188_188945

-- Definitions based on conditions identified
def line_l (t a : ℝ) : ℝ × ℝ := (t, t - a)
def ellipse_C (φ : ℝ) : ℝ × ℝ := (3 * Real.cos φ, 2 * Real.sin φ)
def right_vertex := (3, 0)

theorem determine_a (a : ℝ) :
  (∃ t φ, line_l t a = right_vertex ∧ ellipse_C φ = right_vertex) → a = 3 :=
by
  intro h
  sorry

end determine_a_l188_188945


namespace ratio_of_segments_of_hypotenuse_l188_188575

theorem ratio_of_segments_of_hypotenuse (k : Real) :
  let AB := 3 * k
  let BC := 2 * k
  let AC := Real.sqrt (AB^2 + BC^2)
  ∃ D : Real, 
    let BD := (2 / 3) * D
    let AD := (4 / 9) * D
    let CD := D
    ∀ AD CD, AD / CD = 4 / 9 :=
by
  sorry

end ratio_of_segments_of_hypotenuse_l188_188575


namespace smallest_n_with_70_divisors_l188_188106

theorem smallest_n_with_70_divisors : ∃ n : ℕ, 
  (∃ (k₁ k₂ k₃ : ℕ), 
    (n = 2^k₁ * 3^k₂ * 5^k₃) ∧
    (k₁ + 1) * (k₂ + 1) * (k₃ + 1) = 70 
  ) ∧ 
  (∀ m : ℕ, 
    (∃ (k₁ k₂ k₃ : ℕ), 
      (m = 2^k₁ * 3^k₂ * 5^k₃) ∧
      (k₁ + 1) * (k₂ + 1) * (k₃ + 1) = 70 
    ) → 
    n ≤ m
  ) := 
begin
  use 25920,
  split,
  { use [6, 4, 1],
    split,
    { norm_num, },
    { norm_num, }, },
  { intros m hm,
    obtain ⟨km₁, km₂, km₃, hfm, hkm⟩ := hm,
    sorry, },
end

end smallest_n_with_70_divisors_l188_188106


namespace largest_domain_of_f_l188_188302

theorem largest_domain_of_f (f : ℝ → ℝ) (dom : ℝ → Prop) :
  (∀ x : ℝ, dom x → dom (1 / x)) →
  (∀ x : ℝ, dom x → (f x + f (1 / x) = x)) →
  (∀ x : ℝ, dom x ↔ x = 1 ∨ x = -1) :=
by
  intro h1 h2
  sorry

end largest_domain_of_f_l188_188302


namespace units_digit_of_F_F_15_l188_188664

def F : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := F (n + 1) + F n

theorem units_digit_of_F_F_15 : (F (F 15)) % 10 = 5 := by
  have h₁ : ∀ n, F n % 10 = ([0, 1, 1, 2, 3, 5, 8, 3, 1, 4, 5, 9, 4, 3, 7, 0, 7, 7, 4, 1, 5, 6, 1, 7, 8, 5, 3, 8, 1, 9, 0, 9, 9, 8, 7, 5, 2, 7, 9, 6, 5, 1, 6, 7, 3, 0, 3, 3, 6, 9, 5, 4, 9, 3, 2, 5, 7, 2, 9, 1].take 60)!!.get_or_else (n % 60) 0 := sorry
  have h₂ : F 15 = 610 := by
    simp [F]; -- Proof for F 15 = 610
    sorry
  calc (F (F 15)) % 10
       = F 610 % 10 := by
         rw [h₂]
       ... = 5 := h₁ 610

end units_digit_of_F_F_15_l188_188664


namespace initial_discount_percentage_l188_188377

-- Statement of the problem
theorem initial_discount_percentage (d : ℝ) (x : ℝ)
  (h₁ : d > 0)
  (h_staff_price : d * ((100 - x) / 100) * 0.5 = 0.225 * d) :
  x = 55 := 
sorry

end initial_discount_percentage_l188_188377


namespace celina_total_spending_l188_188807

noncomputable def hoodie_cost : ℕ := 80
noncomputable def flashlight_cost : ℕ := 16
noncomputable def boots_cost : ℕ := 99
noncomputable def water_filter_cost : ℕ := 48.75
noncomputable def camping_mat_cost : ℕ := 38.25

theorem celina_total_spending :
  hoodie_cost + flashlight_cost + boots_cost + water_filter_cost + camping_mat_cost = 282 :=
by
  sorry

end celina_total_spending_l188_188807


namespace maximum_sum_of_sequence_l188_188739

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Condition 1: Define the sequence a_i with constraints > 1 and k in ℤ_+
def isSequenceAcceptable (a : ℕ → ℕ) (k : ℕ) : Prop :=
  (∀ i, 1 < a i) ∧ (k ∈ Set.Univ)

-- Condition 2: Product of factorials divides 2017!
def productOfFactorialsDivides2017Factorial (a : ℕ → ℕ) (k : ℕ) : Prop :=
  factorial(2017) % (List.prod (List.map factorial (List.ofFn a (Fin.mk k (nat.zero_le _))))) = 0

-- Definition of the maximal sum function
def maximalSum (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  List.sum (List.ofFn a (Fin.mk k (nat.zero_le _)))

theorem maximum_sum_of_sequence : 
  ∃ a : ℕ → ℕ, ∃ k : ℕ, 
  isSequenceAcceptable a k ∧ 
  productOfFactorialsDivides2017Factorial a k ∧ 
  maximalSum a k = 5024 := sorry

end maximum_sum_of_sequence_l188_188739


namespace sqrt_equation_has_solution_l188_188059

noncomputable def x : ℝ := Real.sqrt (20 + x)

theorem sqrt_equation_has_solution : x = 5 :=
by
  sorry

end sqrt_equation_has_solution_l188_188059


namespace conjugate_z_is_1_plus_i_l188_188518

noncomputable def i : ℂ := Complex.I
noncomputable def z : ℂ := (1 - i) / Complex.abs i

theorem conjugate_z_is_1_plus_i : Complex.conj z = 1 + i := 
sorry

end conjugate_z_is_1_plus_i_l188_188518


namespace conjugate_of_z_l188_188432

-- Condition definition: the determinant
def determinant {a b c d : ℂ} : ℂ :=
  a * d - b * c

-- Condition for z
def satisfies_condition (z : ℂ) : Prop :=
  determinant z i 1 i = 1 + i

-- Definition of the conjugate of a complex number
def conjugate (z : ℂ) : ℂ :=
  complex.conj z

-- The main theorem that we need to state
theorem conjugate_of_z (z : ℂ) (h : satisfies_condition z) : conjugate z = 2 + i :=
  sorry

end conjugate_of_z_l188_188432


namespace part_c_sum_l188_188740

variables {n k : ℕ}
noncomputable def sum_expression (n k : ℕ) : ℤ :=
∑ i in finset.range(n+1), (-1:ℤ) ^ i * (nat.choose n i : ℤ) * (((1:ℚ) - (i:ℚ) / (n:ℚ)) ^ k : ℤ)

theorem part_c_sum (hn : n > 0) (hk : k > 0) :
 if k < n then
   sum_expression n k = 0
 else if k = n then
   sum_expression n k = (-1)^(n-1) * nat.factorial n
 else
   false :=
begin
  sorry
end

end part_c_sum_l188_188740


namespace hours_per_day_l188_188747

theorem hours_per_day (H : ℕ) : 
  (42 * 12 * H = 30 * 14 * 6) → 
  H = 5 := by
  sorry

end hours_per_day_l188_188747


namespace catering_budget_total_l188_188971

theorem catering_budget_total 
  (total_guests : ℕ)
  (guests_want_chicken guests_want_steak : ℕ)
  (cost_steak cost_chicken : ℕ) 
  (H1 : total_guests = 80)
  (H2 : guests_want_steak = 3 * guests_want_chicken)
  (H3 : cost_steak = 25)
  (H4 : cost_chicken = 18)
  (H5 : guests_want_chicken + guests_want_steak = 80) :
  (guests_want_chicken * cost_chicken + guests_want_steak * cost_steak = 1860) := 
by
  sorry

end catering_budget_total_l188_188971


namespace log_exp_identity_l188_188852

theorem log_exp_identity 
    {a m n : Real} 
    (h1 : log a 2 = m) 
    (h2 : log a 5 = n) : 
    a^(3 * m + n) = 40 := 
by sorry

end log_exp_identity_l188_188852


namespace smallest_n_with_17_proper_factors_l188_188419

-- Definition of the problem conditions
def is_proper_factor (n : ℕ) (y : ℕ) : Prop := y > 1 ∧ y < n ∧ n % y = 0

def proper_factors (n : ℕ) : List ℕ := 
  List.filter (is_proper_factor n) (List.range n)

def valid_ys (n : ℕ) : Prop := (proper_factors n).length = 17

-- The Lean theorem statement
theorem smallest_n_with_17_proper_factors : ∃ n : ℕ, valid_ys n ∧ n = 78732 :=
begin
  sorry
end

end smallest_n_with_17_proper_factors_l188_188419


namespace focal_length_of_hyperbola_l188_188861

noncomputable def hyperbola_focal_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b = sqrt(3) * a) (e : ℝ) (h4 : e = 2) (d : ℝ) (h5 : d = sqrt(3)) : ℝ :=
  let c := e * a in
  2 * c

theorem focal_length_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b = sqrt(3) * a) (e : ℝ) (h4 : e = 2) (d : ℝ) (h5 : d = sqrt(3)) :
  hyperbola_focal_length a b h1 h2 h3 e h4 d h5 = 4 :=
sorry

end focal_length_of_hyperbola_l188_188861


namespace max_f_on_interval_l188_188183

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q
noncomputable def g (x : ℝ) : ℝ := x + 1 / (x^2)

theorem max_f_on_interval 
  {p q : ℝ} 
  (H : ∃ x ∈ set.Icc (1 : ℝ) 2, f x p q = g x ∧ ∀ y ∈ set.Icc (1 : ℝ) 2, g x ≤ g y ∧ f x (p : ℝ) (q : ℝ) ≤ f y p q) :
  ∀ x ∈ set.Icc (1 : ℝ) 2, f x p q ≤ 4 - (5 / 2) * real.cbrt (2 : ℝ) + real.cbrt (4 : ℝ) :=
sorry

end max_f_on_interval_l188_188183


namespace turning_faucet_is_rotational_l188_188707

-- Define the proposition P which states that turning a faucet by hand is not considered a rotational motion.
def P : Prop := ¬ rotational_motion "turning a faucet by hand"

-- Definition of rotational_motion.
def rotational_motion (description : String) : Prop :=
  description = "revolves around a center and direction of motion changes."

-- Given condition that turning a faucet by hand involves revolving around a center and direction of motion changes.
axiom faucet_motion : rotational_motion "turning a faucet by hand"

-- The theorem that we need to prove.
theorem turning_faucet_is_rotational : P = False :=
by
  -- Proof is skipped with sorry.
  sorry

end turning_faucet_is_rotational_l188_188707


namespace evaluate_floor_abs_neg_25_7_l188_188055

theorem evaluate_floor_abs_neg_25_7 : Int.floor (Real.abs (-25.7)) = 25 := by
  sorry

end evaluate_floor_abs_neg_25_7_l188_188055


namespace find_numbers_l188_188107

theorem find_numbers (x : ℚ) (a : ℚ) (b : ℚ) (h₁ : a = 8 * x) (h₂ : b = x^2 - 1) :
  (a * b + a = (2 * x)^3) ∧ (a * b + b = (2 * x - 1)^3) → 
  x = 14 / 13 ∧ a = 112 / 13 ∧ b = 27 / 169 :=
by
  intros h
  sorry

end find_numbers_l188_188107


namespace birds_in_nature_reserve_l188_188355

theorem birds_in_nature_reserve 
  (B : ℕ) 
  (hawks : ℕ := (0.30 * B).toNat)
  (paddyfieldWarblers : ℕ := (0.40 * (0.70 * B)).toNat)
  (kingfishers : ℕ := (0.25 * paddyfieldWarblers).toNat)
  (non_HP_wildlife : ℕ := B - hawks - paddyfieldWarblers - kingfishers) :
  (non_HP_wildlife / B : ℝ) * 100 = 35 :=
by
  sorry

end birds_in_nature_reserve_l188_188355


namespace hyperbola_eccentricity_proof_l188_188156

noncomputable def hyperbola_eccentricity_statement : Prop :=
  ∀ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c : ℝ) (F1 F2 P Q: ℝ × ℝ),
  (F1 = (-c, 0)) →
  (F2 = (c, 0)) →
  (P ≠ 0, 0) →
  (PF1=P∨PF2=P) →
  (|PQ| = |2*F1Q|) →
  (|PF_1|² + |PF_2|² = |QF_2|²) →
  let e := c/a in
  e = √5

-- Proof omitted
theorem hyperbola_eccentricity_proof : hyperbola_eccentricity_statement := sorry

end hyperbola_eccentricity_proof_l188_188156


namespace probability_open_suitcase_l188_188442

theorem probability_open_suitcase : 
  ∀ (password : ℕ), 
  (∀ n, 0 ≤ n ∧ n ≤ 9) →
  (∃ last_digit, 0 ≤ last_digit ∧ last_digit ≤ 9) →
  ∃ p : ℚ, p = 1 / 10 :=
by
  intros password h1 h2
  use (1 / 10)
  sorry

end probability_open_suitcase_l188_188442


namespace dabbie_turkey_cost_l188_188430

theorem dabbie_turkey_cost :
  let weight1 := 6
  let weight2 := 9
  let weight3 := 2 * weight2
  let cost_per_kg := 2
  let total_weight := weight1 + weight2 + weight3
  total_weight * cost_per_kg = 66 :=
by
  let weight1 := 6
  let weight2 := 9
  let weight3 := 2 * weight2
  let cost_per_kg := 2
  let total_weight := weight1 + weight2 + weight3
  show total_weight * cost_per_kg = 66
  rw [total_weight, cost_per_kg, 66]
  sorry

end dabbie_turkey_cost_l188_188430


namespace sqrt_equation_has_solution_l188_188058

noncomputable def x : ℝ := Real.sqrt (20 + x)

theorem sqrt_equation_has_solution : x = 5 :=
by
  sorry

end sqrt_equation_has_solution_l188_188058


namespace parabola_focus_directrix_distance_l188_188540

theorem parabola_focus_directrix_distance (a : ℝ) (h_pos : a > 0) (h_dist : 1 / (2 * 2 * a) = 1) : a = 1 / 4 :=
by
  sorry

end parabola_focus_directrix_distance_l188_188540


namespace probability_non_obtuse_l188_188487

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l188_188487


namespace part_I_a_part_II_l188_188139

variable {n : ℕ}
noncomputable def a_seq (n : ℕ) : ℝ := 3 * n - 2
noncomputable def b_seq (n : ℕ) : ℝ := 2 ^ (n - 1)
noncomputable def T (n : ℕ) : ℝ := 5 + (3 * n - 5) * 2 ^ n

theorem part_I_a :
  (∀ n : ℕ, a_seq n = 3 * n - 2) ∧
  (∀ n : ℕ, b_seq n = 2 ^ (n - 1)) :=
by
  split
  · intro n
    sorry
  · intro n
    sorry

theorem part_II :
  (∀ n : ℕ, T n = 5 + (3 * n - 5) * 2 ^ n) :=
by 
  intro n
  sorry

end part_I_a_part_II_l188_188139


namespace infinite_pairs_l188_188633

theorem infinite_pairs (m : ℕ) (hm : m > 0) : 
  ∃ᶠ x in (coe : ℕ -> ℤ), ∃ᶠ y in (coe : ℕ -> ℤ), 
    Int.gcd x y = 1 ∧ y ∣ (x^2 + m) ∧ x ∣ (y^2 + m) :=
by sorry

end infinite_pairs_l188_188633


namespace ali_score_difference_is_56_l188_188401

theorem ali_score_difference_is_56 (c d : Nat) : 
  ∃ n,  (|(8 * c + d) - (8 * d + c)| = 7 * n) ∧ (∃ opt ∈ {53, 54, 55, 56, 57}, opt = 56) :=
by
  sorry

end ali_score_difference_is_56_l188_188401


namespace coeff_x2_in_binomial_expansion_l188_188835

noncomputable def f (x : ℝ) : ℝ := log (x / 2) + 1 / 2
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 2)

theorem coeff_x2_in_binomial_expansion :
  -- Define the general term of the binomial expansion
  let T_r (r : ℕ) : ℝ := (Nat.choose 5 r : ℝ) * (-2)^r * x^(5 - (3/2:ℝ) * r)
  -- Find the coefficient of x^2
  let r := 2 in
  T_r r = 40 :=
sorry

end coeff_x2_in_binomial_expansion_l188_188835


namespace tan_of_acute_angle_l188_188182

open Real

theorem tan_of_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 2 * sin (α - 15 * π / 180) - 1 = 0) : tan α = 1 :=
by
  sorry

end tan_of_acute_angle_l188_188182


namespace total_fruits_correct_l188_188974

def totalOranges (joan sara carlos : Nat) : Nat := joan + sara + carlos
def totalPears (alyssa ben vanessa : Nat) : Nat := alyssa + ben + vanessa
def totalApples (tim linda : Nat) : Nat := tim + linda

def totalFruits (joan sara carlos alyssa ben vanessa tim linda : Nat) : Nat :=
  totalOranges joan sara carlos + totalPears alyssa ben vanessa + totalApples tim linda

theorem total_fruits_correct :
  totalFruits 37 10 25 30 40 20 15 10 = 187 :=
by
  unfold totalFruits
  unfold totalOranges
  unfold totalPears
  unfold totalApples
  norm_num
  sorry

end total_fruits_correct_l188_188974


namespace number_of_triangles_in_lattice_l188_188912

-- Define the triangular lattice structure
def triangular_lattice_rows : List ℕ := [1, 2, 3, 4]

-- Define the main theorem to state the number of triangles
theorem number_of_triangles_in_lattice :
  let number_of_triangles := 1 + 2 + 3 + 6 + 10
  number_of_triangles = 22 :=
by
  -- here goes the proof, which we skip with "sorry"
  sorry

end number_of_triangles_in_lattice_l188_188912


namespace find_m_value_l188_188870

noncomputable def exponential_function (x : ℝ) (a : ℝ) : ℝ := a ^ x

variables (a x m : ℝ)
variables (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1)
variables (h_function_passes_through : exponential_function 2 a = 4)
variables (h_f_m : exponential_function m a = 8)

theorem find_m_value : m = 3 :=
by
  sorry

end find_m_value_l188_188870


namespace arc_length_of_sector_l188_188522

variable (α : ℝ) (r : ℝ) (l : ℝ)

theorem arc_length_of_sector (h1 : α = 2 * real.pi / 3) (h2 : r = 3) : l = 2 * real.pi := 
by
  sorry

end arc_length_of_sector_l188_188522


namespace area_of_polygon_l188_188946

-- Definitions for square areas
def squareArea (a : ℝ) : ℝ := a ^ 2

-- Problem conditions
def A : ℝ := 25
def B : ℝ := 9

-- Midpoint proof
def midpoint (x y : ℝ) : ℝ := (x + y) / 2

-- Given squares ABCD and EFGD with the squares' side lengths inferred
def sideLengthA := Real.sqrt A
def sideLengthB := Real.sqrt B
def H := midpoint sideLengthA sideLengthB -- Midpoints of BC and EF

-- Property of H the midpoint
def is_midpoint (X : ℝ) := midpoint 0 X = X / 2

-- Total area calculation
theorem area_of_polygon (sA sB : ℝ) (H_is_midpoint: is_midpoint H) : 
    sA = sideLengthA → sB = sideLengthB → 
    (A / 2 + B / 2 + 7.5 - 4.5 = 15.5) := by
  sorry

end area_of_polygon_l188_188946


namespace player5_could_not_have_won_all_matches_l188_188461

variables (W1 L1 W2 L2 W3 L3 W4 L4 W5 L5 : ℕ)

-- Defining the conditions
def player1_condition : Prop := W1 = L1 + 4
def player2_condition : Prop := L2 = W2 + 5
def player3_condition : Prop := L3 = W3 + 5
def player4_condition : Prop := W4 = L4
def player5_condition : Prop := W5 = 4

-- Total wins and losses should be equal
def total_wins_losses_equivalence : Prop :=
  W1 + W2 + W3 + W4 + W5 = L1 + L2 + L3 + L4 + L5

-- Main theorem
theorem player5_could_not_have_won_all_matches :
  player1_condition ∧ 
  player2_condition ∧ 
  player3_condition ∧ 
  player4_condition ∧ 
  player5_condition ∧
  total_wins_losses_equivalence → 
  false := 
sorry

end player5_could_not_have_won_all_matches_l188_188461


namespace integral_sin_cos_equals_half_l188_188444

noncomputable def integral_sin_cos := ∫ t in (0 : ℝ)..(π/2), sin t * cos t

theorem integral_sin_cos_equals_half : integral_sin_cos = 1/2 :=
  sorry

end integral_sin_cos_equals_half_l188_188444


namespace bridget_bakery_profit_l188_188804

theorem bridget_bakery_profit :
  let loaves := 36
  let cost_per_loaf := 1
  let morning_sale_price := 3
  let afternoon_sale_price := 1.5
  let late_afternoon_sale_price := 1
  
  let morning_loaves := (2/3 : ℝ) * loaves
  let morning_revenue := morning_loaves * morning_sale_price
  
  let remaining_after_morning := loaves - morning_loaves
  let afternoon_loaves := (1/2 : ℝ) * remaining_after_morning
  let afternoon_revenue := afternoon_loaves * afternoon_sale_price
  
  let late_afternoon_loaves := remaining_after_morning - afternoon_loaves
  let late_afternoon_revenue := late_afternoon_loaves * late_afternoon_sale_price
  
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
  let total_cost := loaves * cost_per_loaf
  
  total_revenue - total_cost = 51 := by sorry

end bridget_bakery_profit_l188_188804


namespace infinite_nested_sqrt_l188_188069

theorem infinite_nested_sqrt :
  let x := \sqrt{20 + \sqrt{20 + \sqrt{20 + \sqrt{20 + \cdots}}}} in
  x = 5 :=
begin
  let x : ℝ := sqrt(20 + sqrt(20 + sqrt(20 + sqrt(20 + ...)))),
  have h1 : x = sqrt(20 + x), from sorry,
  have h2 : x^2 = 20 + x, from sorry,
  have h3 : x^2 - x - 20 = 0, from sorry,
  have h4 : (x - 5) * (x + 4) = 0, from sorry,
  have h5 : x = 5 ∨ x = -4, from sorry,
  have h6 : x >= 0, from sorry,
  exact h5.elim (λ h, h) (λ h, (h6.elim_left h))
end

end infinite_nested_sqrt_l188_188069


namespace polynomial_coeff_sum_l188_188493

theorem polynomial_coeff_sum (a0 a1 a2 a3 : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a3 * x^3 + a2 * x^2 + a1 * x + a0) →
  a0 + a1 + a2 + a3 = 27 :=
by
  sorry

end polynomial_coeff_sum_l188_188493


namespace infinite_pairs_in_A_not_sum_in_A_l188_188609

def A : Set ℤ := {n | ∃ a b : ℤ, b ≠ 0 ∧ n = a^2 + 13 * b^2}

theorem infinite_pairs_in_A_not_sum_in_A :
  ∃ᵐ x y : ℤ, x^13 + y^13 ∈ A ∧ x + y ∉ A :=
sorry

end infinite_pairs_in_A_not_sum_in_A_l188_188609


namespace volume_ratio_l188_188400

theorem volume_ratio (A B C : ℚ) (h1 : (3/4) * A = (2/3) * B) (h2 : (2/3) * B = (1/2) * C) :
  A / C = 2 / 3 :=
sorry

end volume_ratio_l188_188400


namespace sophia_age_in_three_years_l188_188319

def current_age_jeremy : Nat := 40
def current_age_sebastian : Nat := current_age_jeremy + 4

def sum_ages_in_three_years (age_jeremy age_sebastian age_sophia : Nat) : Nat :=
  (age_jeremy + 3) + (age_sebastian + 3) + (age_sophia + 3)

theorem sophia_age_in_three_years (age_sophia : Nat) 
  (h1 : sum_ages_in_three_years current_age_jeremy current_age_sebastian age_sophia = 150) :
  age_sophia + 3 = 60 := by
  sorry

end sophia_age_in_three_years_l188_188319


namespace units_digit_of_fibonacci_f_15_l188_188666

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n + 1) + fibonacci n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fibonacci_f_15 :
  units_digit (fibonacci (fibonacci 15)) = 5 := 
sorry

end units_digit_of_fibonacci_f_15_l188_188666


namespace smallest_value_of_AP_plus_BP_l188_188612

noncomputable def pointA : ℝ × ℝ := (Real.sqrt 5, 0)
noncomputable def pointB : ℝ × ℝ := (7, 5)
noncomputable def ellipse : ℝ × ℝ → Prop :=
  λ P, (P.1 ^ 2 / 4 + P.2 ^ 2 / 9 = 1)

theorem smallest_value_of_AP_plus_BP :
  ∃ P : ℝ × ℝ, ellipse P ∧
  ∀ Q : ℝ × ℝ, ellipse Q → dist pointA Q + dist pointB Q ≥ 6 :=
by
  sorry

end smallest_value_of_AP_plus_BP_l188_188612


namespace problem_II_l188_188214

-- Defining the conditions
def point_on_curve_C1 (a b φ : ℝ) (x y : ℝ) : Prop :=
  x = a * Real.cos φ ∧ y = b * Real.sin φ

def curve_C1 (x y : ℝ) : Prop :=
  ∃ φ : ℝ, point_on_curve_C1 2 1 φ x y

def point_on_curve_C2 (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

def point_C1_M {x y : ℝ} : Prop := 
  point_on_curve_C1 2 1 (π / 3) x y

def point_D_on_C2 (x y : ℝ) : Prop :=
  x = 1 ∧ y = Real.sqrt 3 / 2 ∧ point_on_curve_C2 x y

-- Main theorem to prove
theorem problem_II 
  (θ ρ₁ ρ₂ : ℝ)
  (A_on_C1 : curve_C1 (ρ₁ * Real.cos θ) (ρ₁ * Real.sin θ))
  (B_on_C1 : curve_C1 (ρ₂ * Real.cos (θ + π / 2)) (ρ₂ * Real.sin (θ + π / 2))) : 
  1 / ρ₁^2 + 1 / ρ₂^2 = 5 / 4 :=
  sorry

end problem_II_l188_188214


namespace find_Sum_4n_l188_188129

variable {a : ℕ → ℕ} -- Define a sequence a_n

-- Define our conditions about the sums Sn and S3n
axiom Sum_n : ℕ → ℕ 
axiom Sum_3n : ℕ → ℕ 
axiom Sum_4n : ℕ → ℕ 

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a n + a 0)) / 2

axiom h1 : is_arithmetic_sequence a
axiom h2 : Sum_n 1 = 2
axiom h3 : Sum_3n 3 = 12

theorem find_Sum_4n : Sum_4n 4 = 20 :=
sorry

end find_Sum_4n_l188_188129


namespace tank_loss_rate_after_first_repair_l188_188761

def initial_capacity : ℕ := 350000
def first_loss_rate : ℕ := 32000
def first_loss_duration : ℕ := 5
def second_loss_duration : ℕ := 10
def filling_rate : ℕ := 40000
def filling_duration : ℕ := 3
def missing_gallons : ℕ := 140000

noncomputable def first_repair_loss_rate := (initial_capacity - (first_loss_rate * first_loss_duration) + (filling_rate * filling_duration) - (initial_capacity - missing_gallons)) / second_loss_duration

theorem tank_loss_rate_after_first_repair : first_repair_loss_rate = 10000 := by sorry

end tank_loss_rate_after_first_repair_l188_188761


namespace initial_percentage_of_gold_l188_188795

theorem initial_percentage_of_gold (x : ℝ) (h₁ : 48 * x / 100 + 12 = 40 * 60 / 100) : x = 25 :=
by
  sorry

end initial_percentage_of_gold_l188_188795


namespace pair_count_l188_188910

theorem pair_count (x y : ℕ) :
  (0 ≤ x ∧ x ≤ y ∧ 5 * x^2 - 4 * x * y + 2 * x + y^2 = 624) ↔
  {p : ℕ × ℕ | 0 ≤ p.1 ∧ p.1 ≤ p.2 ∧ 5 * p.1^2 - 4 * p.1 * p.2 + 2 * p.1 + p.2^2 = 624}.card = 7 :=
by
  sorry

end pair_count_l188_188910


namespace lemma_g_inequality_l188_188621

-- Given
variables {D : Set ℝ} (g : ℝ → ℝ)
  (h1 : ∀ x1 x2 ∈ D, ∀ t ∈ Ioo (0 : ℝ) 1, g (x1^t * x2^(1 - t)) ≤ t * g x1 + (1 - t) * g x2)
  (a b : ℝ)
  (h2 : 0 < a ∧ a < b)
  (x y : ℝ)
  (h3 : x ∈ Ioo a b)
  (h4 : y ∈ Ioo a b)
  (h5 : x * y = a * b)

-- To Prove
theorem lemma_g_inequality : g a + g b ≥ g x + g y :=
sorry

end lemma_g_inequality_l188_188621


namespace count_valid_numbers_l188_188170

theorem count_valid_numbers :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 7 = 3 ∧ n % 10 = 6 ∧ n % 12 = 9}.card = 2 :=
by sorry

end count_valid_numbers_l188_188170


namespace midpoint_area_ratio_l188_188798

theorem midpoint_area_ratio
  (A B C D E F G H M N : Type) 
  (midpoints : E = midpoint A B ∧ F = midpoint B C ∧ G = midpoint C D ∧ H = midpoint D A)
  (intersections : M = intersection (line B H) (line D E) ∧ N = intersection (line B G) (line D F)) :
  let S_ABCD := area A B C D 
  let S_BMND := area B M N D 
  (S_BMND / S_ABCD = 1/3) := 
sorry

end midpoint_area_ratio_l188_188798


namespace quadratic_equation_l188_188816

theorem quadratic_equation (a b c x1 x2 : ℝ) (hx1 : a * x1^2 + b * x1 + c = 0) (hx2 : a * x2^2 + b * x2 + c = 0) :
  ∃ y : ℝ, c * y^2 + b * y + a = 0 := 
sorry

end quadratic_equation_l188_188816


namespace sphere_surface_area_l188_188763

-- Define the volume of the cube
def cube_volume : ℝ := 8

-- The relationship between the cube and the sphere
def all_vertices_on_sphere (s : ℝ) : Prop :=
  let side_len := (cube_volume)^(1/3)
  s = (side_len / 2) * real.sqrt(3)

-- Define the surface area of the sphere
def surface_area_of_sphere (r : ℝ) : ℝ :=
  4 * real.pi * r^2

-- Prove that given the conditions, the surface area is 12π cm²
theorem sphere_surface_area : ∃ r : ℝ, all_vertices_on_sphere r ∧ surface_area_of_sphere r = 12 * real.pi :=
  sorry

end sphere_surface_area_l188_188763


namespace min_calls_correct_l188_188291

-- Define a function that calculates the minimum number of calls given n people
def min_calls (n : ℕ) : ℕ :=
  2 * n - 2

-- Theorem to prove that min_calls(n) given the conditions is equal to 2n - 2
theorem min_calls_correct (n : ℕ) (h : n ≥ 2) : min_calls n = 2 * n - 2 :=
by
  sorry

end min_calls_correct_l188_188291


namespace geometry_problem_l188_188947

theorem geometry_problem
  (A B C D E : Type*)
  (BAC ABC ACB ADE ADC AEB DEB CDE : ℝ)
  (h₁ : ABC = 72)
  (h₂ : ACB = 90)
  (h₃ : CDE = 36)
  (h₄ : ADC = 180)
  (h₅ : AEB = 180) :
  DEB = 162 :=
sorry

end geometry_problem_l188_188947


namespace construct_bisecting_plane_l188_188045

noncomputable theory
open_locale classical

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  direction : Point
  point : Point

variable (P : Point) (l₁ l₂ : Line)

def intersects (l₁ l₂ : Line) : Prop :=
  -- Here, we define what it means for two lines to intersect at an inaccessible point.
  l₁.direction ≠ l₂.direction ∧
  (∃ t₁ t₂ : ℝ, l₁.point.x + t₁ * l₁.direction.x = l₂.point.x + t₂ * l₂.direction.x ∧
                l₁.point.y + t₁ * l₁.direction.y  = l₂.point.y + t₂ * l₂.direction.y ∧
                l₁.point.z + t₁ * l₁.direction.z = l₂.point.z + t₂ * l₂.direction.z)

theorem construct_bisecting_plane (h₁ : intersects l₁ l₂) :
  ∃ P₁ : Point, ∃ plane : Point → Prop, plane P ∧ plane P₁ ∧
  (∀ Q₁ Q₂ : Point, (plane Q₁ ∧ plane Q₂) → 
    (∃ l₁ l₂_angle_biased Q₁P Q₂P : Point,
      l₁.direction = Q₁P ∧ l₂.direction = Q₂P ∧
      l₁.direction ≠ l₂.direction ∧ 
      Q₁P = P₁Q.parallel ∧ Q₁P = P.parallel) :=
sorry

end construct_bisecting_plane_l188_188045


namespace basic_astrophysics_degrees_l188_188353

theorem basic_astrophysics_degrees :
  let microphotonics := 14
  let home_electronics := 19
  let food_additives := 10
  let gm_microorganisms := 24
  let industrial_lubricants := 8
  ∑ x in [microphotonics, home_electronics, food_additives, gm_microorganisms, industrial_lubricants], x = 75 →
  100 - 75 = 25 →
  0.25 * 360 = 90 :=
by
  sorry

end basic_astrophysics_degrees_l188_188353


namespace box_mass_calculation_l188_188752

variable (h₁ w₁ l₁ : ℝ) (m₁ : ℝ)
variable (h₂ w₂ l₂ density₁ density₂ : ℝ)

theorem box_mass_calculation
  (h₁_eq : h₁ = 3)
  (w₁_eq : w₁ = 4)
  (l₁_eq : l₁ = 6)
  (m₁_eq : m₁ = 72)
  (h₂_eq : h₂ = 1.5 * h₁)
  (w₂_eq : w₂ = 2.5 * w₁)
  (l₂_eq : l₂ = l₁)
  (density₂_eq : density₂ = 2 * density₁)
  (density₁_eq : density₁ = m₁ / (h₁ * w₁ * l₁)) :
  h₂ * w₂ * l₂ * density₂ = 540 := by
  sorry

end box_mass_calculation_l188_188752


namespace sum_inverse_S_eq_l188_188862

noncomputable def a (n : ℕ) : ℕ := 2 * n

noncomputable def S (n : ℕ) : ℕ := n * (n + 1)

noncomputable def inverse_S (n : ℕ) : ℚ := 1 / S n

noncomputable def T (n : ℕ) : ℚ := (∑ i in Finset.range n, inverse_S (i + 1))

theorem sum_inverse_S_eq (n : ℕ) : T n = n / (n + 1) :=
sorry

end sum_inverse_S_eq_l188_188862


namespace max_value_of_f_l188_188847

noncomputable def f (x : ℝ) : ℝ := min (min (2 * x + 3) (3 * x - 2)) (25 - x)

theorem max_value_of_f :
  ∃ x : ℝ, f x = 53 / 3 :=
by
  have key : f (22 / 3) = 53 / 3 := 
    by
      dsimp [f]
      simp
  exact ⟨22 / 3, key⟩
  sorry

end max_value_of_f_l188_188847


namespace count_integers_with_factors_12_and_7_l188_188552

theorem count_integers_with_factors_12_and_7 :
  ∃ k : ℕ, k = 4 ∧
    (∀ (n : ℕ), 500 ≤ n ∧ n ≤ 800 ∧ 12 ∣ n ∧ 7 ∣ n ↔ (84 ∣ n ∧
      n = 504 ∨ n = 588 ∨ n = 672 ∨ n = 756)) :=
sorry

end count_integers_with_factors_12_and_7_l188_188552


namespace gumballs_remaining_l188_188790

theorem gumballs_remaining (a b total eaten remaining : ℕ) 
  (hAlicia : a = 20) 
  (hPedro : b = a + 3 * a) 
  (hTotal : total = a + b) 
  (hEaten : eaten = 40 * total / 100) 
  (hRemaining : remaining = total - eaten) : 
  remaining = 60 := by
  sorry

end gumballs_remaining_l188_188790


namespace hyperbola_asymptotes_l188_188815

-- Define the data for the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (y - 1)^2 / 16 - (x + 2)^2 / 25 = 1

-- Define the two equations for the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = 4 / 5 * x + 13 / 5
def asymptote2 (x y : ℝ) : Prop := y = -4 / 5 * x + 13 / 5

-- Theorem stating that the given asymptotes are correct for the hyperbola
theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, hyperbola_eq x y → (asymptote1 x y ∨ asymptote2 x y)) := 
by
  sorry

end hyperbola_asymptotes_l188_188815


namespace value_of_a_l188_188179

theorem value_of_a (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) (h3 : a > b) (h4 : a - b = 8) : a = 10 := 
by 
sorry

end value_of_a_l188_188179


namespace solve_inequality_l188_188286

theorem solve_inequality : 
  {x : ℝ | -3 * x^2 + 9 * x + 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
by {
  sorry
}

end solve_inequality_l188_188286


namespace relationship_y1_y2_l188_188145

theorem relationship_y1_y2 :
  (y1 y2 : ℝ)(h1 : y1 = -(-2 : ℝ) + 1) (h2 : y2 = -(3 : ℝ) + 1) :
  y1 > y2 :=
by
  sorry

end relationship_y1_y2_l188_188145


namespace count_five_digit_palindromes_l188_188413

def is_five_digit_palindrome (d : ℕ) : Prop := 
  ∃ (a b c : ℕ), 
    a ≠ 0 ∧ 
    a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    d = a * 10001 + b * 1010 + c * 100

theorem count_five_digit_palindromes :
  {d : ℕ | is_five_digit_palindrome d}.to_finset.card = 900 :=
by sorry

end count_five_digit_palindromes_l188_188413


namespace probability_no_obtuse_triangle_correct_l188_188472

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l188_188472


namespace third_note_denom_is_10_l188_188766

-- Definitions according to the conditions
def total_amount : ℕ := 400
def total_notes : ℕ := 75
def num_notes_each_denom : ℕ := 25

-- Main theorem to prove the denomination of the third type of notes
theorem third_note_denom_is_10 :
  ∃ D : ℕ, (total_amount = (num_notes_each_denom * 1) + 
            (num_notes_each_denom * 5) + 
            (num_notes_each_denom * D)) ∧
            total_notes = num_notes_each_denom * 3 ∧
            D = 10 :=
begin
  use 10,
  split,
  { -- proof that total_amount matches
    sorry },
  split,
  { -- proof that total_notes match
    sorry },
  { -- proof that D = 10
    refl }
end

end third_note_denom_is_10_l188_188766


namespace analytical_expression_and_monotonicity_inequality_solution_l188_188532

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (a * x + b) / (1 + x^2)

def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) := ∀ ⦃x1 x2 : ℝ⦄, x1 ∈ I → x2 ∈ I → x1 < x2 → f x1 < f x2

theorem analytical_expression_and_monotonicity :
  (∃ a b : ℝ, is_odd_function (f a b) ∧ f (1 / 2) a b = 2 / 5) →
  ∃ a : ℝ, a = 1 ∧ (∀ b : ℝ, b = 0 → (∀ x : ℝ, -1 < x ∧ x < 1 → f a b x = x / (1 + x^2)))
  ∧ is_increasing_on (f 1 0) {x | -1 < x ∧ x < 1} :=
sorry

theorem inequality_solution :
  (∃ a b : ℝ, is_odd_function (f a b) ∧ f (1 / 2) a b = 2 / 5) →
  (∃ x : ℝ → ℝ, (set_of (λ x, f a b (x - 1) + f a b x < 0)) = {x | 0 < x ∧ x < 1 / 2}) :=
sorry

end analytical_expression_and_monotonicity_inequality_solution_l188_188532


namespace Will_Had_28_Bottles_l188_188724

-- Definitions based on conditions
-- Let days be the number of days water lasted (4 days)
def days : ℕ := 4

-- Let bottles_per_day be the number of bottles Will drank each day (7 bottles/day)
def bottles_per_day : ℕ := 7

-- Correct answer defined as total number of bottles (28 bottles)
def total_bottles : ℕ := 28

-- The proof statement to show that the total number of bottles is equal to 28
theorem Will_Had_28_Bottles :
  (bottles_per_day * days = total_bottles) :=
by
  sorry

end Will_Had_28_Bottles_l188_188724


namespace parabola_line_intersection_triangle_area_point_on_x_axis_l188_188541

theorem parabola_line_intersection 
  (m : ℝ) : 
  (∃ x y : ℝ, y^2 = 4 * x ∧ y = 2 * x + m ∧ (abs(x - x) = sqrt 15)) → 
  m = -1 := 
by 
  sorry

theorem triangle_area_point_on_x_axis 
  (a : ℝ) : 
  (let P : ℝ × ℝ := (a, 0) in 
  ∃ x1 y1 x2 y2 : ℝ, 
    y1^2 = 4 * x1 ∧ y1 = 2 * x1 - 1 ∧
    y2^2 = 4 * x2 ∧ y2 = 2 * x2 - 1 ∧ 
    dist ({x1, y1}, {x2, y2}) = sqrt 15 ∧  
    let d := abs (2 * a - 1) / sqrt 5 in 
    d = 2 * (9 * sqrt 3) / sqrt 15 ∧
    a = 5 ∨ a = -4) :=
by 
  sorry

end parabola_line_intersection_triangle_area_point_on_x_axis_l188_188541


namespace total_jumps_l188_188550

theorem total_jumps (hattie_1 : ℕ) (lorelei_1 : ℕ) (hattie_2 : ℕ) (lorelei_2 : ℕ) (hattie_3 : ℕ) (lorelei_3 : ℕ) :
  hattie_1 = 180 →
  lorelei_1 = 3 / 4 * hattie_1 →
  hattie_2 = 2 / 3 * hattie_1 →
  lorelei_2 = hattie_2 + 50 →
  hattie_3 = hattie_2 + 1 / 3 * hattie_2 →
  lorelei_3 = 4 / 5 * lorelei_1 →
  hattie_1 + hattie_2 + hattie_3 + lorelei_1 + lorelei_2 + lorelei_3 = 873 :=
by
  intros h1 l1 h2 l2 h3 l3
  sorry

end total_jumps_l188_188550


namespace units_digit_of_FF15_is_5_l188_188658

noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem units_digit_of_FF15_is_5 : (fib (fib 15)) % 10 = 5 :=
by
  sorry

end units_digit_of_FF15_is_5_l188_188658


namespace polynomial_solution_l188_188089

noncomputable def is_polynomial_solution (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, P.eval (2 * x) = (Polynomial.derivative P).eval x * (Polynomial.derivative (Polynomial.derivative P)).eval x

theorem polynomial_solution :
  ∀ P : Polynomial ℝ, is_polynomial_solution P → 
    P = Polynomial.C (4/9):ℝ * Polynomial.X ^ 3 + 
        Polynomial.C (3 * 4/9):ℝ * Polynomial.X ^ 2 + 
        Polynomial.C c * Polynomial.X ∧ 
        c ∈ ℝ := by
  sorry

end polynomial_solution_l188_188089


namespace total_students_l188_188356

theorem total_students (T : ℕ) (h1 : (1/5 : ℚ) * T + (1/4 : ℚ) * T + (1/2 : ℚ) * T + 20 = T) : 
  T = 400 :=
sorry

end total_students_l188_188356


namespace coins_on_side_l188_188042

theorem coins_on_side (circumference : ℕ) (n : ℕ) : (coins : ℕ) :=
  (circumference = 36) →
  (coins = 4 * n - 4) →
  (n = 10)

end coins_on_side_l188_188042


namespace length_of_IL_l188_188217

theorem length_of_IL (x y : ℝ) (h : x^2 + y^2 = 162) : √(2 * 162) = 18 :=
by
  sorry

end length_of_IL_l188_188217


namespace solution1_solution2_l188_188757

noncomputable def question1 (A B : ℝ × ℝ) (h : A ≠ B)
  (hyperbola : A.1 ^ 2 - A.2 ^ 2 / 3 = 1 ∧ B.1 ^ 2 - B.2 ^ 2 / 3 = 1)
  (angle_condition : ∃ m : ℝ, (A.2 - B.2) / (A.1 - B.1) = m ∧ m = (Real.sqrt 3) / 3) :
  ℝ :=
let AB := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) in
AB

noncomputable def question2 (A B F2 : ℝ × ℝ) (F2_eq : F2.1 = -F1.1 ∧ F2.2 = F1.2)
  (h : A ≠ B)
  (hyperbola : A.1 ^ 2 - A.2 ^ 2 / 3 = 1 ∧ B.1 ^ 2 - B.2 ^ 2 / 3 = 1)
  (angle_condition : ∃ m : ℝ, (A.2 - B.2) / (A.1 - B.1) = m ∧ m = (Real.sqrt 3) / 3) :
  ℝ :=
let AB := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) in
let FA := Real.sqrt ((A.1 + 2) ^ 2 + A.2 ^ 2) in
let FB := Real.sqrt ((B.1 + 2) ^ 2 + B.2 ^ 2) - 2 in
AB + FA + FB

theorem solution1 :
  ∃ A B : ℝ × ℝ, (A ≠ B) ∧
    (A.1 ^ 2 - A.2 ^ 2 / 3 = 1 ∧ B.1 ^ 2 - B.2 ^ 2 / 3 = 1) ∧
    (∃ m : ℝ, (A.2 - B.2) / (A.1 - B.1) = m ∧ m = (Real.sqrt 3) / 3) ∧
    question1 A B (by sorry) (by sorry) (by sorry) = 3 := by
  sorry

theorem solution2 :
  ∃ A B F2 : ℝ × ℝ, (A ≠ B) ∧
    (F2.1 = 2 ∧ F2.2 = 0) ∧
    (A.1 ^ 2 - A.2 ^ 2 / 3 = 1 ∧ B.1 ^ 2 - B.2 ^ 2 / 3 = 1) ∧
    (∃ m : ℝ, (A.2 - B.2) / (A.1 - B.1) = m ∧ m = (Real.sqrt 3) / 3) ∧
    question2 A B F2 (by sorry) (by sorry) (by sorry) (by sorry) = 3 + 3 * Real.sqrt 3 := by
  sorry

end solution1_solution2_l188_188757


namespace quadratic_polynomial_solution_l188_188844

theorem quadratic_polynomial_solution :
  ∃ (a b c : ℚ), (λ x : ℚ, a * x^2 + b * x + c) 1 = 3 ∧ 
                 (λ x : ℚ, a * x^2 + b * x + c) 0 = 2 ∧ 
                 (λ x : ℚ, a * x^2 + b * x + c) 3 = 18 ∧ 
                 (a = 13/6 ∧ b = -7/6 ∧ c = 2) :=
by
  use 13/6, -7/6, 2
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  repeat { sorry }

end quadratic_polynomial_solution_l188_188844


namespace vector_equality_l188_188275

variables (A B C D M : Type) [inner_product_space ℝ V] [point A B C D M] (M : inner_product_space ℝ V)

-- Definitions of position vectors
axiom pos_A : V := vector A
axiom pos_B : V := vector B
axiom pos_C : V := vector C
axiom pos_D : V := vector D

-- Condition provided in the problem
axiom condition : ∀ (M : point), inner_product (vector (M, A), vector (M, B)) ≠ inner_product (vector (M, C), vector (M, D))

-- The statement to be proved
theorem vector_equality : vector (A, C) = vector (D, B) :=
sorry

end vector_equality_l188_188275


namespace probability_of_at_least_one_consonant_l188_188363

theorem probability_of_at_least_one_consonant :
  let n_total := 8
  let n_consonants := 6
  let n_vowels := 2
  let p_no_consonants := (n_vowels / n_total) * ((n_vowels - 1) / (n_total - 1))
  let p_at_least_one_consonant := 1 - p_no_consonants
  in p_at_least_one_consonant = 27 / 28 :=
by
  intros n_total n_consonants n_vowels p_no_consonants p_at_least_one_consonant
  sorry

end probability_of_at_least_one_consonant_l188_188363


namespace skyscraper_anniversary_l188_188191

theorem skyscraper_anniversary (built_years_ago : ℕ) (anniversary_years : ℕ) (years_before : ℕ) :
    built_years_ago = 100 → anniversary_years = 200 → years_before = 5 → 
    (anniversary_years - years_before) - built_years_ago = 95 := by
  intros h1 h2 h3
  sorry

end skyscraper_anniversary_l188_188191


namespace no_primes_in_sequence_l188_188871

-- Definitions and conditions derived from the problem statement
variable (a : ℕ → ℕ) -- sequence of natural numbers
variable (increasing : ∀ n, a n < a (n + 1)) -- increasing sequence
variable (is_arith_or_geom : ∀ n, (2 * a (n + 1) = a n + a (n + 2)) ∨ (a (n + 1) ^ 2 = a n * a (n + 2))) -- arithmetic or geometric progression condition
variable (divisible_by_four : a 0 % 4 = 0 ∧ a 1 % 4 = 0) -- first two numbers divisible by 4

-- The statement to prove: no prime numbers exist in the sequence
theorem no_primes_in_sequence : ∀ n, ¬ (Nat.Prime (a n)) :=
by 
  sorry

end no_primes_in_sequence_l188_188871


namespace max_distance_of_MN_l188_188898

noncomputable def curve_C_polar (θ : ℝ) : ℝ := 2 * Real.cos θ

noncomputable def curve_C_cartesian (x y : ℝ) := x^2 + y^2 - 2 * x

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ( -1 + (Real.sqrt 5 / 5) * t, (2 * Real.sqrt 5 / 5) * t)

def point_M : ℝ × ℝ := (0, 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def center_C : ℝ × ℝ := (1, 0)

theorem max_distance_of_MN :
  ∃ N : ℝ × ℝ, 
  ∀ (θ : ℝ), N = (curve_C_polar θ * Real.cos θ, curve_C_polar θ * Real.sin θ) →
  distance point_M N ≤ Real.sqrt 5 + 1 :=
sorry

end max_distance_of_MN_l188_188898


namespace count_integers_expressed_as_sums_of_floors_l188_188553

theorem count_integers_expressed_as_sums_of_floors :
  (have h : ℝ → ℕ := λ x, Int.floor (3 * x) + Int.floor (5 * x) + Int.floor (7 * x) + Int.floor (9 * x),
  {n : ℕ | ∃ x : ℝ, h x = n}.finite.count (≤ 1500) = 1260) := 
sorry

end count_integers_expressed_as_sums_of_floors_l188_188553


namespace complete_square_eqn_l188_188648

theorem complete_square_eqn (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 → (x + d)^2 = e) → d + e = 5 :=
by
  sorry

end complete_square_eqn_l188_188648


namespace conic_section_represents_parabola_l188_188722

theorem conic_section_represents_parabola :
  ∀ x y : ℝ, (|x - 4| = real.sqrt ((y + 3)^2 + x^2)) → 
  (∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y = e * x + f * y) → True :=
by
  intro x y
  intro h
  simp
  sorry

end conic_section_represents_parabola_l188_188722


namespace tangent_line_values_graph_above_l188_188893

-- Part (Ⅰ): Prove x₀ = 1 and m = -2
theorem tangent_line_values (f : ℝ → ℝ) (x₀ : ℝ) (m : ℝ) 
    (hx₀_pos : 0 < x₀) (h_eqn : f x₀ = 2 * x₀ + m) :
  (∀ x, f x = x - 1/x) → x₀ = 1 ∧ m = -2 :=
  sorry

-- Part (Ⅱ): Prove graph of g(x) is above graph of f(x)
theorem graph_above (f g : ℝ → ℝ) 
    (h_f_eq : ∀ x, f x = x - 1/x) 
    (h_g_eq : ∀ x, g x = 1 + x * real.log x) 
    (x_pos : ∀ x, 0 < x → 0 < (g x - f x)) : ∀ x, 0 < x → g x > f x :=
  sorry

end tangent_line_values_graph_above_l188_188893


namespace father_has_nine_children_l188_188764

noncomputable def total_assets : ℝ := 81000
noncomputable def each_child_share : ℝ := 9000

theorem father_has_nine_children :
  ∃ n : ℕ, (n * each_child_share = total_assets) ∧ (n = 9) :=
by
  let n := 9
  have h1 : n * each_child_share = total_assets := by sorry
  use n
  exact ⟨h1, rfl⟩

end father_has_nine_children_l188_188764


namespace num_ways_to_write_430_is_44_l188_188613

noncomputable def num_ways_to_write_430 : ℕ :=
  let fo_expr := (b3 : ℕ) → (b2 : ℕ) → (b1 : ℕ) → (b0 : ℕ) → 
                 430 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0 ∧ 
                 0 ≤ b3 ∧ b3 ≤ 99 ∧ 
                 0 ≤ b2 ∧ b2 ≤ 99 ∧ 
                 0 ≤ b1 ∧ b1 ≤ 99 ∧ 
                 0 ≤ b0 ∧ b0 ≤ 99
  let M := {
    (b3, b2, b1, b0) ∈ ℕ×ℕ×ℕ×ℕ | fo_expr b3 b2 b1 b0
  }
  card M -- M is ensured here to be 44 by proof filled in

theorem num_ways_to_write_430_is_44 : num_ways_to_write_430 = 44 :=
  sorry

end num_ways_to_write_430_is_44_l188_188613


namespace angle_ACB_is_90_l188_188956

theorem angle_ACB_is_90 (A B C D E G : Point)
  (h1 : AB = 3 * AC)
  (h2 : Points_on_segment D AB)
  (h3 : Points_on_segment E BC)
  (h4 : ∠BAE = ∠ACD)
  (h5 : G = Intersection AE CD)
  (h6 : EquilateralTriangle BEG) :
  ∠ACB = 90 :=
sorry

end angle_ACB_is_90_l188_188956


namespace prime_p_prime_p₁₀_prime_p₁₄_l188_188744

theorem prime_p_prime_p₁₀_prime_p₁₄ (p : ℕ) (h₀p : Nat.Prime p) 
  (h₁ : Nat.Prime (p + 10)) (h₂ : Nat.Prime (p + 14)) : p = 3 := by
  sorry

end prime_p_prime_p₁₀_prime_p₁₄_l188_188744


namespace sum_combinations_l188_188038

theorem sum_combinations :
  (finset.range 17).sum (λ k, nat.choose (k + 3) 2) = 1139 :=
begin
  sorry
end

end sum_combinations_l188_188038


namespace find_distance_AC_l188_188338

noncomputable def distance_AC : ℝ :=
  let speed := 25  -- km per hour
  let angleA := 30  -- degrees
  let angleB := 135 -- degrees
  let distanceBC := 25 -- km
  (distanceBC * Real.sin (angleB * Real.pi / 180)) / (Real.sin (angleA * Real.pi / 180))

theorem find_distance_AC :
  distance_AC = 25 * Real.sqrt 2 :=
by
  sorry

end find_distance_AC_l188_188338


namespace common_difference_l188_188865

variable (a : ℕ → ℝ)

def arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ a1, ∀ n, a n = a1 + (n - 1) * d

def geometric_sequence (a1 a2 a5 : ℝ) : Prop :=
  a1 * (a1 + 4 * (a2 - a1)) = (a2 - a1)^2

theorem common_difference {d : ℝ} (hd : d ≠ 0)
  (h_arith : arithmetic a d)
  (h_sum : a 1 + a 2 + a 5 = 13)
  (h_geom : geometric_sequence (a 1) (a 2) (a 5)) :
  d = 2 :=
sorry

end common_difference_l188_188865


namespace vector_BC_correct_l188_188510

-- Define points A, B, and the vector AC
def A := (0, 1)
def B := (3, 2)
def AC := (-4, -3)

-- Define the coordinates for point C derived from vector AC and point A
def C : ℕ × ℕ := (-4, -2)

-- The vector BC is the difference in coordinates between points B and C
def vector_BC := (C.1 - B.1, C.2 - B.2)

-- The theorem to be proven:
theorem vector_BC_correct : vector_BC = (-7, -4) :=
by
  -- the proof is skipped
  sorry

end vector_BC_correct_l188_188510


namespace haley_total_spent_l188_188549

theorem haley_total_spent :
  let price_first_three := 4,
      price_additional := 3,
      count_first_three := 3,
      count_additional := 5 in
  let cost_first_three := count_first_three * price_first_three,
      cost_additional := count_additional * price_additional,
      total_cost := cost_first_three + cost_additional in
  total_cost = 27 :=
by
  let price_first_three := 4
  let price_additional := 3
  let count_first_three := 3
  let count_additional := 5
  let cost_first_three := count_first_three * price_first_three
  let cost_additional := count_additional * price_additional
  let total_cost := cost_first_three + cost_additional
  show total_cost = 27 from sorry

end haley_total_spent_l188_188549


namespace skyscraper_anniversary_l188_188189

theorem skyscraper_anniversary (built_years_ago : ℕ) (anniversary_years : ℕ) (years_before : ℕ) :
    built_years_ago = 100 → anniversary_years = 200 → years_before = 5 → 
    (anniversary_years - years_before) - built_years_ago = 95 := by
  intros h1 h2 h3
  sorry

end skyscraper_anniversary_l188_188189


namespace sum_ineq_l188_188241

theorem sum_ineq (α : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ i, 0 < α i) 
  (h2 : ∑ i in (Finset.range n), α i = 1) 
  (h3 : 2 ≤ n) :
  ∑ i in (Finset.range n), α i / (2 - α i) ≥ n / (2 * n - 1) :=
by
  sorry

end sum_ineq_l188_188241


namespace no_obtuse_triangle_probability_l188_188485

noncomputable def prob_no_obtuse_triangle : ℝ :=
  9 / 128

theorem no_obtuse_triangle_probability :
  let circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 },
      points : List (ℝ × ℝ) × ℝ := (List.replicate 4 (0, 0), 1) -- assuming uniform rand points chosen
  in 
  ∀ (A₀ A₁ A₂ A₃ : ℝ × ℝ), A₀ ∈ circle → A₁ ∈ circle → A₂ ∈ circle → A₃ ∈ circle →
  (prob_no_obtuse_triangle = 9 / 128) :=
by
  sorry

end no_obtuse_triangle_probability_l188_188485


namespace pages_for_ten_dollars_l188_188596

theorem pages_for_ten_dollars (p c pages_per_cent : ℕ) (dollars cents : ℕ) (h1 : p = 5) (h2 : c = 10) (h3 : pages_per_cent = p / c) (h4 : dollars = 10) (h5 : cents = 100 * dollars) :
  (cents * pages_per_cent) = 500 :=
by
  sorry

end pages_for_ten_dollars_l188_188596


namespace determinant_of_matrix_mul_imaginary_unit_l188_188539

theorem determinant_of_matrix_mul_imaginary_unit :
  let determinant (M : Matrix (Fin 2) (Fin 2) ℂ) : ℂ :=
    M 0 0 * M 1 1 - M 0 1 * M 1 0 in
  let M := ![
    ![Complex.i, 2],
    ![1, Complex.i]
  ] in
  determinant M * Complex.i = -3 * Complex.i :=
by
  sorry

end determinant_of_matrix_mul_imaginary_unit_l188_188539


namespace λ_value_l188_188142

variables {a b : EuclideanSpace ℝ (Fin 3)}
noncomputable def λ (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  if (3 • a + λ • b) ⋅ a = 0 then -3/2 else sorry

def angle_a_b := real.angle a b = π / 3
def norm_a := ‖a‖ = 1
def norm_b := ‖b‖ = 4

theorem λ_value : 
  angle_a_b →
  norm_a →
  norm_b →
  (λ : ℝ → ℝ) (a b) = -3/2 :=
by
  intros
  unfold λ angle_a_b norm_a norm_b
  sorry

end λ_value_l188_188142


namespace winning_percentage_is_70_l188_188939

def percentage_of_votes (P : ℝ) : Prop :=
  ∃ (P : ℝ), (7 * P - 7 * (100 - P) = 280 ∧ 0 ≤ P ∧ P ≤ 100)

theorem winning_percentage_is_70 :
  percentage_of_votes 70 :=
by
  sorry

end winning_percentage_is_70_l188_188939


namespace compare_sizes_l188_188420

noncomputable def a : ℝ := 0.2^2
noncomputable def b : ℝ := 2^0.2
noncomputable def c : ℝ := Real.log 2 / Real.log 0.2

theorem compare_sizes : b > a ∧ a > c := by
  sorry

end compare_sizes_l188_188420


namespace least_number_to_add_l188_188731

theorem least_number_to_add (n : ℕ) (H : n = 433124) : ∃ k, k = 15 ∧ (n + k) % 17 = 0 := by
  sorry

end least_number_to_add_l188_188731


namespace vector_magnitude_MN_l188_188907

/-- Given the magnitudes and dot product of vectors OA and OB, prove the magnitude of MN. -/
theorem vector_magnitude_MN (OA OB: ℝ^2) (M N A B S : ℝ^2)
  (h1 : ∥ OA ∥ = 2)
  (h2 : ∥ OB ∥ = 2)
  (h3 : OA • OB = 2)
  (h4 : A = (OA + M) / 2 )
  (h5 : S = 2 * A - M)
  (h6 : B = (OB + S) / 2)
  (h7 : N = 2 * B - S) :
  ∥(N - M)∥ = 4 :=
sorry

end vector_magnitude_MN_l188_188907


namespace range_of_a_l188_188248

-- Define the conditions
def A : Set ℝ := { x : ℝ | -2 < x ∧ x < 4 }
def B (a : ℝ) : Set ℝ := { x : ℝ | x^2 + a * x + 4 = 0 }
def one_nonempty_subset (a : ℝ) : Prop := (∀ x ∈ A ∩ B a, ¬(∃ y ∈ A ∩ B a, y ≠ x))

theorem range_of_a (a : ℝ) : one_nonempty_subset a → 
  a ∈ Iic (-5) ∨ a = -4 ∨ a ∈ Ici 4 :=
sorry

end range_of_a_l188_188248


namespace problem1_problem2_problem3_l188_188122

    -- Definitions of sequences and given conditions
    variables (a : ℕ → ℝ) (b : ℕ → ℝ)
    variable (t : ℝ)
    variable (S : ℕ → ℝ)
    
    -- Given conditions
    -- 1. ∀ n ∈ ℕ⁺, 4(n+1)a(n)² - n a(n+1)² = 0)
    axiom a_pos : ∀ n : ℕ, n > 0 → a(n) > 0
    axiom seq_cond : ∀ n : ℕ, n > 0 → 4 * (n + 1) * (a(n)^2) = n * (a(n+1) ^ 2)
    
    -- Conditions for sequence b
    def b (n : ℕ) : ℝ := a(n)^2 / (t^n)
    
    -- Problem statement
    theorem problem1 (h_pos : ∀(n : ℕ), n > 0 → a(n) > 0) (h_cond : ∀ n : ℕ, n > 0 → 4 * (n + 1) * (a n)^2 = n * (a (n + 1))^2) :
        ∀ n : ℕ, n > 0 → ∃ r : ℝ, r > 0 ∧ ∀ m : ℕ, m > 0 → (a (m + 1) / √(m + 1)) = r * (a m / √m) :=
    sorry

    theorem problem2 (h_pos : ∀(n : ℕ), n > 0 → a(n) > 0) (h_is_arith_seq : ∀ n : ℕ, n > 0 → b(n+1) - b(n) = b(n) - b(n-1)) :
        t = 4 :=
    sorry

    theorem problem3 (h_pos : ∀(n : ℕ), n > 0 → a(n) > 0) (t_eq_4 : t = 4) 
        (h_seq_arith : ∀ n : ℕ, n > 0 → ((n * (a 1) ^ 2) / 4) = S(n)) 
        (h : ∀ n : ℕ, n > 0 → ∃ m : ℕ, 8 * (a 1) ^ 2 * S (n) - (a 1) ^ 4 * n ^ 2 = 16 * b(m)) :
        ∀ k : ℕ, k > 0 → a(1) = 2 * k :=
    sorry
    
end problem1_problem2_problem3_l188_188122


namespace count_positive_numbers_l188_188404

-- Definitions for the given numbers
def num1 := -16
def num2 := 0.25
def num3 := 1 / 7
def num4 := -2003
def num5 := -3.24
def pi_val := Real.pi

-- Statement to prove that the number of positive numbers is 3
theorem count_positive_numbers :
  (num2 > 0 ∧ num3 > 0 ∧ pi_val > 0) ∧ (num1 < 0 ∧ num4 < 0 ∧ num5 < 0) →
  3 = 3 := 
by
  intros h,
  sorry

end count_positive_numbers_l188_188404


namespace probability_non_obtuse_l188_188488

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l188_188488


namespace total_amount_245_l188_188785

-- Define the conditions and the problem
theorem total_amount_245 (a : ℝ) (x y z : ℝ) (h1 : y = 0.45 * a) (h2 : z = 0.30 * a) (h3 : y = 63) :
  x + y + z = 245 := 
by
  -- Starting the proof (proof steps are unnecessary as per the procedure)
  sorry

end total_amount_245_l188_188785


namespace problem_statement_l188_188158

def A : Set ℝ := { x | x * (x - 1) < 0 }
def B : Set ℝ := { x | Real.exp x > 1 }

theorem problem_statement : (Aᶜ ∩ B) = Ici 1 :=
by
  sorry

end problem_statement_l188_188158


namespace unique_polynomial_l188_188092

def satisfies_condition (P : ℝ[X]) : Prop :=
  ∀ x : ℝ, P.eval (2 * x) = (polynomial.derivative P).eval x * (polynomial.derivative (polynomial.derivative P)).eval x

theorem unique_polynomial {P : ℝ[X]} (h : satisfies_condition P) : P = (4 / 9) • polynomial.X ^ 3 :=
by
  sorry

end unique_polynomial_l188_188092


namespace smallest_n_square_division_l188_188113

theorem smallest_n_square_division : ∃ (n : ℕ), (∀ (a b : ℕ), n = 40 * a + 49 * b → a ≥ 1 ∧ b ≥ 1) ∧ n = 2000 :=
by
  let exists_n : ∃ (n : ℕ), (∀ (a b : ℕ), n = 40 * a + 49 * b → a ≥ 1 ∧ b ≥ 1) := ⟨2000, by sorry⟩;
  exact exists.intro 2000 (and.intro (exists_n.snd) rfl)

end smallest_n_square_division_l188_188113


namespace smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l188_188713

noncomputable def smallest_not_prime_nor_square_no_prime_factor_lt_60 : ℕ :=
  4087

theorem smallest_not_prime_nor_square_no_prime_factor_lt_60_correct :
  ∀ n : ℕ, 
    (n > 0) → 
    (¬ Prime n) →
    (¬ ∃ k : ℕ, k * k = n) →
    (∀ p : ℕ, Prime p → p ∣ n → p ≥ 60) →
    n ≥ 4087 :=
sorry

end smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l188_188713


namespace area_of_triangle_25_16_l188_188834

def area_of_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

theorem area_of_triangle_25_16 :
  area_of_triangle 25 16 = 200 :=
by
  sorry

end area_of_triangle_25_16_l188_188834


namespace real_roots_approx_correct_to_4_decimal_places_l188_188048

noncomputable def f (x : ℝ) : ℝ := x^4 - (2 * 10^10 + 1) * x^2 - x + 10^20 + 10^10 - 1

theorem real_roots_approx_correct_to_4_decimal_places :
  ∃ x1 x2 : ℝ, 
  abs (x1 - 99999.9997) ≤ 0.0001 ∧ 
  abs (x2 - 100000.0003) ≤ 0.0001 ∧ 
  f x1 = 0 ∧ 
  f x2 = 0 :=
sorry

end real_roots_approx_correct_to_4_decimal_places_l188_188048


namespace distance_B_to_plane_GEF_l188_188498

noncomputable theory
open_locale classical

-- Definitions of the conditions
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (0, 4, 0)
def C : ℝ × ℝ × ℝ := (4, 4, 0)
def D : ℝ × ℝ × ℝ := (4, 0, 0)
def E : ℝ × ℝ × ℝ := (0, 2, 0) -- midpoint of AB
def F : ℝ × ℝ × ℝ := (2, 0, 0) -- midpoint of AD
def G : ℝ × ℝ × ℝ := (0, 0, 2) -- point G where OG = 2

-- Defining the problem statement
theorem distance_B_to_plane_GEF :
  let EF := (2, -2, 0) in
  let GE := (0, 2, -2) in
  let n := (1, 1, 3) -- normalized normal vector
  let BE := (0, -2, 0) in
  abs ((1, 1, 3).1 * (0, -2, 0).1 + (1, 1, 3).2 * (0, -2, 0).2 + (1, 1, 3).3 * (0, -2, 0).3) / sqrt (1^2 + 1^2 + 3^2) = 2 * sqrt 11 / 11 :=
sorry

end distance_B_to_plane_GEF_l188_188498


namespace fraction_green_knights_magical_l188_188580

variables (total_knights magical_knights green_knights yellow_knights : ℕ)
variables (p q : ℕ)

-- Conditions
def fraction_green := 3 / 8
def fraction_magical := 1 / 5
def green_knights_fraction_magical := p / q
def yellow_knights_fraction_magical := p / (3 * q)

-- Hypotheses
hypothesis (H1 : fraction_green * total_knights = green_knights)
hypothesis (H2 : total_knights - green_knights = yellow_knights)
hypothesis (H3 : fraction_magical * total_knights = magical_knights)
hypothesis (H4 : 3 * yellow_knights_fraction_magical = green_knights_fraction_magical)
hypothesis (H5 : green_knights_fraction_magical * green_knights + yellow_knights_fraction_magical * yellow_knights = magical_knights)

-- The theorem to prove
theorem fraction_green_knights_magical : green_knights_fraction_magical = 12 / 35 :=
by sorry

end fraction_green_knights_magical_l188_188580


namespace construct_segment_AB_l188_188133

-- Define the two points A and B and assume the distance between them is greater than 1 meter
variables {A B : Point} (dist_AB_gt_1m : Distance A B > 1)

-- Define the ruler length as 10 cm
def ruler_length : ℝ := 0.1

theorem construct_segment_AB 
  (h : dist_AB_gt_1m) 
  (ruler : ℝ := ruler_length) : ∃ (AB : Segment), Distance A B = AB.length ∧ AB.length > 1 :=
sorry

end construct_segment_AB_l188_188133


namespace race_order_count_l188_188551

theorem race_order_count (H H' R N : Type) [Finite H] [Finite H'] [Finite R] [Finite N] 
  (distinct: H ≠ H' ∧ H ≠ R ∧ H ≠ N ∧ H' ≠ R ∧ H' ≠ N ∧ R ≠ N) : 
  Finset.card (Finset.univ : Finset (Fin 4 → Type)) = 24 := 
by
  sorry

end race_order_count_l188_188551


namespace find_polynomial_l188_188094

open Polynomial

theorem find_polynomial (P : ℝ[X]) (h : ∀ x : ℝ, eval (2*x) P = eval x (P.derivative) * eval x (P.derivative.derivative)) :
  P = C (4 / 9) * X^3 :=
by 
  sorry

end find_polynomial_l188_188094


namespace maximum_x_plus_y_l188_188992

theorem maximum_x_plus_y (N x y : ℕ) 
  (hN : N = 19 * x + 95 * y) 
  (hp : ∃ k : ℕ, N = k^2) 
  (hN_le : N ≤ 1995) :
  x + y ≤ 86 :=
sorry

end maximum_x_plus_y_l188_188992


namespace number_of_zeros_in_interval_f_positive_l188_188155

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x + 1

noncomputable def f' (x : ℝ) : ℝ := Real.log x - 2 / x + 1

noncomputable def f'' (x : ℝ) : ℝ := 1 / x + 2 / (x * x)

theorem number_of_zeros_in_interval (h₀ : ∀ x > 0, f'' x > 0) :
  ∃! x ∈ set.Ioo 1 2, f' x = 0 := sorry

theorem f_positive (h₀ : ∀ x > 0, f'' x > 0) :
  ∀ x > 0, f x > 0 := sorry

end number_of_zeros_in_interval_f_positive_l188_188155


namespace catering_budget_l188_188969

namespace CateringProblem

variables (s c : Nat) (cost_steak cost_chicken : Nat)

def total_guests (s c : Nat) : Prop := s + c = 80

def steak_to_chicken_ratio (s c : Nat) : Prop := s = 3 * c

def total_cost (s c cost_steak cost_chicken : Nat) : Nat := s * cost_steak + c * cost_chicken

theorem catering_budget :
  ∃ (s c : Nat), (total_guests s c) ∧ (steak_to_chicken_ratio s c) ∧ (total_cost s c 25 18) = 1860 :=
by
  sorry

end CateringProblem

end catering_budget_l188_188969


namespace sum_of_interior_angles_and_diagonals_l188_188306

theorem sum_of_interior_angles_and_diagonals (h : ∀ (n : ℕ), 360 / 24 = n) :
  ∃ n : ℕ, 360 / 24 = n ∧ ((180 * (n - 2)) = 2340 ∧ (n * (n - 3)) / 2 = 90) :=
by
  use 15
  split
  { exact h 15 }
  split
  { repeat { sorry } }
  { repeat { sorry } }

end sum_of_interior_angles_and_diagonals_l188_188306


namespace alpha_when_beta_neg4_l188_188289

theorem alpha_when_beta_neg4 :
  (∀ (α β : ℝ), (β ≠ 0) → α = 5 → β = 2 → α * β^2 = α * 4) →
   ∃ (α : ℝ), α = 5 → ∃ β, β = -4 → α = 5 / 4 :=
  by
    intros h
    use 5 / 4
    sorry

end alpha_when_beta_neg4_l188_188289


namespace problem1_problem2_l188_188416

theorem problem1 (a b : ℝ) : (-(2 : ℝ) * a ^ 2 * b) ^ 3 / (-(2 * a * b)) * (1 / 3 * a ^ 2 * b ^ 3) = (4 / 3) * a ^ 7 * b ^ 5 :=
  by
  sorry

theorem problem2 (x : ℝ) : (27 * x ^ 3 + 18 * x ^ 2 - 3 * x) / -3 * x = -9 * x ^ 2 - 6 * x + 1 :=
  by
  sorry

end problem1_problem2_l188_188416


namespace sqrt_equation_has_solution_l188_188061

noncomputable def x : ℝ := Real.sqrt (20 + x)

theorem sqrt_equation_has_solution : x = 5 :=
by
  sorry

end sqrt_equation_has_solution_l188_188061


namespace complex_expression_l188_188916

noncomputable def B : ℂ := 3 + 2 * complex.I
noncomputable def Q : ℤ := -3
noncomputable def T : ℂ := 2 * complex.I
noncomputable def U : ℂ := 1 + 5 * complex.I

theorem complex_expression : 2 * B - Q + 3 * T + U = (10 : ℂ) + 15 * complex.I := by
  sorry

end complex_expression_l188_188916


namespace total_investment_after_18_years_l188_188273

def P1 : ℝ := 2000
def r1 : ℝ := 0.06
def P2 : ℝ := 2500
def r2 : ℝ := 0.085
def P3 : ℝ := 7000 - P1 - P2
def r3 : ℝ := 0.07
def t : ℝ := 18
def n : ℝ := 1

def compound_interest (P r n t : ℝ) : ℝ := P * (1 + r / n)^(n * t)

def A1 : ℝ := compound_interest P1 r1 n t
def A2 : ℝ := compound_interest P2 r2 n t
def A3 : ℝ := compound_interest P3 r3 n t

def total_amount : ℝ := A1 + A2 + A3

theorem total_investment_after_18_years : total_amount = 24605.11 := by
  sorry

end total_investment_after_18_years_l188_188273


namespace sequence_prime_bounded_l188_188384

theorem sequence_prime_bounded (c : ℕ) (h : c > 0) : 
  ∀ (p : ℕ → ℕ), (∀ k, Nat.Prime (p k)) → (p 0) = some_prime →
  (∀ k, ∃ q, Nat.Prime q ∧ q ∣ (p k + c) ∧ (∀ i < k, q ≠ p i)) → 
  (∃ N, ∀ m ≥ N, ∀ n ≥ N, p m = p n) :=
by
  sorry

end sequence_prime_bounded_l188_188384


namespace sale_coupon_discount_l188_188782

theorem sale_coupon_discount
  (original_price : ℝ)
  (sale_price : ℝ)
  (price_after_coupon : ℝ)
  (h1 : sale_price = 0.5 * original_price)
  (h2 : price_after_coupon = 0.8 * sale_price) :
  (original_price - price_after_coupon) / original_price * 100 = 60 := by
sorry

end sale_coupon_discount_l188_188782


namespace medal_award_count_l188_188941

theorem medal_award_count :
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  no_canadians_get_medals + one_canadian_gets_medal = 480 :=
by
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  show no_canadians_get_medals + one_canadian_gets_medal = 480
  -- here should be the steps skipped
  sorry

end medal_award_count_l188_188941


namespace least_sum_of_four_primes_l188_188711

def is_prime (n: ℕ) : Prop := nat.prime n

def is_sum_of_four_distinct_primes (n: ℕ) : Prop :=
  ∃ a b c d, a > 10 ∧ b > 10 ∧ c > 10 ∧ d > 10 ∧
             is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
             a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
             a + b + c + d = n

def is_multiple_of_15 (n: ℕ) : Prop :=
  n % 15 = 0

theorem least_sum_of_four_primes (n: ℕ) (h1: is_sum_of_four_distinct_primes n) (h2: is_multiple_of_15 n) : n = 60 := 
sorry

end least_sum_of_four_primes_l188_188711


namespace gumballs_remaining_l188_188788

theorem gumballs_remaining (Alicia_gumballs : ℕ) (Pedro_gumballs : ℕ) (Total_gumballs : ℕ) (Gumballs_taken_out : ℕ)
  (h1 : Alicia_gumballs = 20)
  (h2 : Pedro_gumballs = Alicia_gumballs + 3 * Alicia_gumballs)
  (h3 : Total_gumballs = Alicia_gumballs + Pedro_gumballs)
  (h4 : Gumballs_taken_out = 40 * Total_gumballs / 100) :
  Total_gumballs - Gumballs_taken_out = 60 := by
  sorry

end gumballs_remaining_l188_188788


namespace curve_eq_shapes_l188_188678

theorem curve_eq_shapes (m : ℝ) :
  (3x^2 + m * y^2 = 1 → 
    ((m = 3 ∨ m < 0 ∨ (m > 0 ∧ m ≠ 3) ∨ m = 0) ∧ 
    (m = 3 → circle) ∧ 
    (m < 0 → hyperbola) ∧ 
    ((m > 0 ∧ m ≠ 3) → ellipse) ∧ 
    (m = 0 → two lines))
    := sorry

end curve_eq_shapes_l188_188678


namespace problem_statement_l188_188987

theorem problem_statement : 
  let a := (0.2:Real) ^ 0.3 
  let b := (0.2:Real) ^ 0.2 
  let c := Real.logb 2 0.4 
  c < a ∧ a < b :=
by
  sorry

end problem_statement_l188_188987


namespace smallest_n_with_70_divisors_l188_188105

theorem smallest_n_with_70_divisors : ∃ n : ℕ, 
  (∃ (k₁ k₂ k₃ : ℕ), 
    (n = 2^k₁ * 3^k₂ * 5^k₃) ∧
    (k₁ + 1) * (k₂ + 1) * (k₃ + 1) = 70 
  ) ∧ 
  (∀ m : ℕ, 
    (∃ (k₁ k₂ k₃ : ℕ), 
      (m = 2^k₁ * 3^k₂ * 5^k₃) ∧
      (k₁ + 1) * (k₂ + 1) * (k₃ + 1) = 70 
    ) → 
    n ≤ m
  ) := 
begin
  use 25920,
  split,
  { use [6, 4, 1],
    split,
    { norm_num, },
    { norm_num, }, },
  { intros m hm,
    obtain ⟨km₁, km₂, km₃, hfm, hkm⟩ := hm,
    sorry, },
end

end smallest_n_with_70_divisors_l188_188105


namespace cos_sin_fraction_identity_l188_188445

theorem cos_sin_fraction_identity :
  (cos 27 - real.sqrt 2 * sin 18) / cos 63 = 1 := 
by
  -- Proving that the expression simplifies to 1
  sorry

end cos_sin_fraction_identity_l188_188445


namespace focus_of_parabola_l188_188100

theorem focus_of_parabola (p : ℝ) :
  (∃ p, x ^ 2 = 4 * p * y ∧ x ^ 2 = 4 * 1 * y) → (0, p) = (0, 1) :=
by
  sorry

end focus_of_parabola_l188_188100


namespace andrea_lauren_bike_reach_l188_188797

noncomputable def andrea_lauren_biking (dist_apart : ℕ) (time_till_flat : ℕ) (andrea_speed_factor : ℕ) 
  (closing_rate_kmpm : ℕ) (correct_total_time : ℕ) : Prop :=
  let lauren_speed := 15 in
  let andrea_speed := 3 * lauren_speed in
  let initial_distance := 20 in
  let new_distance := initial_distance - (closing_rate_kmpm * time_till_flat) in
  let lauren_time_after_stop := new_distance * 60 / lauren_speed in
  let total_time := time_till_flat + lauren_time_after_stop in
  total_time = correct_total_time

theorem andrea_lauren_bike_reach : 
  andrea_lauren_biking 20 5 3 1 65 :=
by 
  sorry

end andrea_lauren_bike_reach_l188_188797


namespace neg_sin_universal_proposition_l188_188543

theorem neg_sin_universal_proposition :
  (¬ ∀ x : ℝ, sin x ≤ 1) ↔ (∃ x : ℝ, sin x > 1) :=
sorry

end neg_sin_universal_proposition_l188_188543


namespace wage_on_18th_day_l188_188441

def average_wage_month (avg_wage : ℕ) (total_days : ℕ) : Prop :=
  avg_wage = 120 ∧ total_days = 30

def average_wage_week (week_avg : ℕ → ℕ) : Prop :=
  week_avg 1 = 110 ∧
  week_avg 2 = 130 ∧
  week_avg 3 = 140 ∧
  week_avg 4 = 115 

theorem wage_on_18th_day 
  (avg_month : ℕ := 120) 
  (days : ℕ := 30)
  (week_avg : ℕ → ℕ) 
  (h1 : average_wage_month avg_month days) 
  (h2 : average_wage_week week_avg) :
  ∃ wage : ℕ, wage = 140 :=
by
  use 140
  sorry

end wage_on_18th_day_l188_188441


namespace bags_already_made_l188_188709

def bags_per_batch : ℕ := 10
def customer_order : ℕ := 60
def days_to_fulfill : ℕ := 4
def batches_per_day : ℕ := 1

theorem bags_already_made :
  (customer_order - (days_to_fulfill * batches_per_day * bags_per_batch)) = 20 :=
by
  sorry

end bags_already_made_l188_188709


namespace James_age_l188_188965

-- Defining variables
variables (James John Tim : ℕ)
variables (h1 : James + 12 = John)
variables (h2 : Tim + 5 = 2 * John)
variables (h3 : Tim = 79)

-- Statement to prove James' age
theorem James_age : James = 25 :=
by {
  sorry
}

end James_age_l188_188965


namespace complement_intersection_l188_188546

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

theorem complement_intersection (hU : ∀ x, x ∈ U) (hA : ∀ x, x ∈ A) (hB : ∀ x, x ∈ B) :
    (U \ A) ∩ (U \ B) = {7, 9} :=
by
  sorry

end complement_intersection_l188_188546


namespace sum_fn_equals_364_l188_188109

def f (n : ℕ) : ℝ :=
  (4 * n + Real.sqrt (4 * n^2 - 1)) / (Real.sqrt (2 * n + 1) + Real.sqrt (2 * n - 1))

theorem sum_fn_equals_364 : 
  (∑ n in Finset.range 40, f (n + 1)) = 364 := 
by
  sorry

end sum_fn_equals_364_l188_188109


namespace find_third_divisor_l188_188459

theorem find_third_divisor 
  (h1 : ∃ (n : ℕ), n = 1014 - 3 ∧ n % 12 = 0 ∧ n % 16 = 0 ∧ n % 21 = 0 ∧ n % 28 = 0) 
  (h2 : 1011 - 3 = 1008) : 
  (∃ d, d = 3 ∧ 1008 % d = 0 ∧ 1008 % 12 = 0 ∧ 1008 % 16 = 0 ∧ 1008 % 21 = 0 ∧ 1008 % 28 = 0) :=
sorry

end find_third_divisor_l188_188459


namespace employee_department_division_l188_188579

theorem employee_department_division (G : SimpleGraph V) (isolated_count : ℕ)
  (H_isolated : isolated_count ≤ 2015) :
  ∃ (c : Coloring G 11), ∀ (v w : V), G.Adj v w → c v ≠ c w :=
by
  sorry

end employee_department_division_l188_188579


namespace triangle_perimeter_l188_188305

/-- The lengths of two sides of a triangle are 3 and 5 respectively. The third side is a root of the equation x^2 - 7x + 12 = 0. Find the perimeter of the triangle. -/
theorem triangle_perimeter :
  let side1 := 3
  let side2 := 5
  let third_side1 := 3
  let third_side2 := 4
  (third_side1 * third_side1 - 7 * third_side1 + 12 = 0) ∧
  (third_side2 * third_side2 - 7 * third_side2 + 12 = 0) →
  (side1 + side2 + third_side1 = 11 ∨ side1 + side2 + third_side2 = 12) :=
by
  sorry

end triangle_perimeter_l188_188305


namespace min_value_sum_fraction_l188_188990

theorem min_value_sum_fraction (x : Fin 150 → ℝ) 
  (hx_pos : ∀ i, 0 < x i) 
  (hx_sum_squares : ∑ i, (x i)^2 = 2) : 
  ∃ (y : ℝ), y = 3 * Real.sqrt 3 ∧ ∑ i, (x i) / (2 - (x i)^2) ≥ y := 
by 
  sorry

end min_value_sum_fraction_l188_188990


namespace units_digit_F_F_15_l188_188673

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem units_digit_F_F_15 : (fib (fib 15) % 10) = 5 := by
  sorry

end units_digit_F_F_15_l188_188673


namespace arithmetic_seq_common_difference_l188_188501

theorem arithmetic_seq_common_difference (a1 d : ℝ) (h1 : a1 + 2 * d = 10) (h2 : 4 * a1 + 6 * d = 36) : d = 2 :=
by
  sorry

end arithmetic_seq_common_difference_l188_188501


namespace count_valid_sequences_returning_rectangle_l188_188813

/-- The transformations that can be applied to the rectangle -/
inductive Transformation
| rot90   : Transformation
| rot180  : Transformation
| rot270  : Transformation
| reflYeqX  : Transformation
| reflYeqNegX : Transformation

/-- Apply a transformation to a point (x, y) -/
def apply_transformation (t : Transformation) (p : ℝ × ℝ) : ℝ × ℝ :=
match t with
| Transformation.rot90   => (-p.2,  p.1)
| Transformation.rot180  => (-p.1, -p.2)
| Transformation.rot270  => ( p.2, -p.1)
| Transformation.reflYeqX  => ( p.2,  p.1)
| Transformation.reflYeqNegX => (-p.2, -p.1)

/-- Apply a sequence of transformations to a list of points -/
def apply_sequence (seq : List Transformation) (points : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  seq.foldl (λ acc t => acc.map (apply_transformation t)) points

/-- Prove that there are exactly 12 valid sequences of three transformations that return the rectangle to its original position -/
theorem count_valid_sequences_returning_rectangle :
  let rectangle := [(0,0), (6,0), (6,2), (0,2)];
  let transformations := [Transformation.rot90, Transformation.rot180, Transformation.rot270, Transformation.reflYeqX, Transformation.reflYeqNegX];
  let seq_transformations := List.replicate 3 transformations;
  (seq_transformations.filter (λ seq => apply_sequence seq rectangle = rectangle)).length = 12 :=
sorry

end count_valid_sequences_returning_rectangle_l188_188813


namespace abcdeq_five_l188_188097

theorem abcdeq_five (a b c d : ℝ) 
    (h1 : a + b + c + d = 20) 
    (h2 : ab + ac + ad + bc + bd + cd = 150) : 
    a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5 := 
  by
  sorry

end abcdeq_five_l188_188097


namespace replacement_paint_intensity_l188_188651

theorem replacement_paint_intensity 
  (original_intensity: ℝ) 
  (fraction_replaced: ℝ) 
  (new_intensity: ℝ) 
  (h1: original_intensity = 50) 
  (h2: fraction_replaced = 1/3) 
  (h3: new_intensity = 40): 
  ∃ (I: ℝ), I = 20 :=
by
  exists 20
  sorry

end replacement_paint_intensity_l188_188651


namespace find_b_l188_188178

-- Let gcd denote the greatest common divisor

theorem find_b (b : ℕ) (gcd : ℕ → ℕ → ℕ) (h_gcdb : gcd (12, b)) (h_gcd1218 : gcd (18, 12) = 6) (h_finalgcd : gcd (gcd (12, b), 6) = 2) :
  b = 2 :=
sorry

end find_b_l188_188178


namespace factor_difference_of_squares_l188_188827

theorem factor_difference_of_squares (t : ℤ) : t^2 - 64 = (t - 8) * (t + 8) :=
by {
  sorry
}

end factor_difference_of_squares_l188_188827


namespace sqrt_equality_pattern_l188_188492

theorem sqrt_equality_pattern (a t : ℝ) (h1 : a ≠ 0) (h2 : t ≠ 0)
  (h3 : ∀ k ∈ {2, 3, 4}, sqrt (k * (k / (f k))) = k * sqrt (k / (f k))) :
  a = 6 ∧ t = 6^2 - 1 :=
by
  -- Skipping the proof with "sorry"
  sorry

end sqrt_equality_pattern_l188_188492


namespace find_hanyoung_weight_l188_188908

variable (H J : ℝ)

def hanyoung_is_lighter (H J : ℝ) : Prop := H = J - 4
def sum_of_weights (H J : ℝ) : Prop := H + J = 88

theorem find_hanyoung_weight (H J : ℝ) (h1 : hanyoung_is_lighter H J) (h2 : sum_of_weights H J) : H = 42 :=
by
  sorry

end find_hanyoung_weight_l188_188908


namespace salary_and_percentage_increase_l188_188626

def initial_salary := (S : ℝ)

def annual_raise := 1.08

def new_salary_after_bonus (S : ℝ) : ℝ :=
  S * annual_raise^3 * 1.05

def percentage_increase (initial : ℝ) (new : ℝ) : ℝ :=
  (new - initial) / initial * 100

theorem salary_and_percentage_increase (S : ℝ) :
  new_salary_after_bonus S = S * 1.3226976 ∧
  percentage_increase S (new_salary_after_bonus S) = 32.27 :=
by
  sorry

end salary_and_percentage_increase_l188_188626


namespace range_of_a_l188_188617

-- Definitions for the conditions
def p (x : ℝ) := x ≤ 2
def q (x : ℝ) (a : ℝ) := x < a + 2

-- Theorem statement
theorem range_of_a (a : ℝ) : (∀ x : ℝ, q x a → p x) → a ≤ 0 := by
  sorry

end range_of_a_l188_188617


namespace degree_f_x2_mul_g_x4_l188_188655

open Polynomial

theorem degree_f_x2_mul_g_x4 {f g : Polynomial ℝ} (hf : degree f = 4) (hg : degree g = 5) :
  degree (f.comp (X ^ 2) * g.comp (X ^ 4)) = 28 :=
sorry

end degree_f_x2_mul_g_x4_l188_188655


namespace min_5a2_plus_6a3_l188_188249

theorem min_5a2_plus_6a3 (a_1 a_2 a_3 : ℝ) (r : ℝ)
  (h1 : a_1 = 2)
  (h2 : a_2 = a_1 * r)
  (h3 : a_3 = a_1 * r^2) :
  5 * a_2 + 6 * a_3 ≥ -25 / 12 :=
by
  sorry

end min_5a2_plus_6a3_l188_188249


namespace cos_sum_diff_identity_l188_188085

theorem cos_sum_diff_identity (x y : ℝ) :
  cos (x + y) - cos (x - y) = -2 * sin (x + y) * sin (x - y) :=
by
  sorry

end cos_sum_diff_identity_l188_188085


namespace polynomial_remainder_4_l188_188436

theorem polynomial_remainder_4 (c : ℚ) : 
  (polynomial.div_mod_by_monic (8 * X^3 + 5 * X^2 - c * X - 3) (3 * X^2 - 2 * X + 1)).snd = 4 :=
by sorry

end polynomial_remainder_4_l188_188436


namespace relatively_prime_ratios_l188_188140

theorem relatively_prime_ratios (r s : ℕ) (h_coprime: Nat.gcd r s = 1) 
  (h_cond: (r : ℝ) / s = 2 * (Real.sqrt 2 + Real.sqrt 10) / (5 * Real.sqrt (3 + Real.sqrt 5))) :
  r = 4 ∧ s = 5 :=
by
  sorry

end relatively_prime_ratios_l188_188140


namespace sample_size_survey_l188_188588

theorem sample_size_survey (students_selected : ℕ) (h : students_selected = 200) : students_selected = 200 :=
by
  assumption

end sample_size_survey_l188_188588


namespace relationship_among_abc_l188_188689

def a : ℝ := 0.3^3
def b : ℝ := log 3 (0.3 : ℝ)
def c : ℝ := 3 ^ 0.3

theorem relationship_among_abc : b < a ∧ a < c := by
  sorry

end relationship_among_abc_l188_188689


namespace probability_sum_six_l188_188324

-- Define the concept of a fair six-sided die
def fair_six_sided_die : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the event of rolling three six-sided dice and the desired sum
def event_sum_six (rolls : (fair_six_sided_die × fair_six_sided_die × fair_six_sided_die)) : Prop :=
  (rolls.1.val + rolls.2.val + rolls.2.snd.val) = 6

-- Calculate the required probability
theorem probability_sum_six : 
  (∃ (total_outcomes favourable_outcomes: ℕ),
    total_outcomes = 6 * 6 * 6 ∧
    favourable_outcomes = 10 ∧
    5 / 108 = favourable_outcomes / total_outcomes) :=
sorry

end probability_sum_six_l188_188324


namespace volume_of_inscribed_cube_in_sphere_in_cube_l188_188391

def diameter_of_inscribed_sphere (edge_length : ℝ) : ℝ :=
  edge_length

def side_length_of_inscribed_cube (diameter : ℝ) : ℝ :=
  diameter / (Real.sqrt 3)

def volume_of_cube (side_length : ℝ) : ℝ :=
  side_length ^ 3

theorem volume_of_inscribed_cube_in_sphere_in_cube 
  (edge_length : ℝ) (h : edge_length = 9) :
  volume_of_cube (side_length_of_inscribed_cube (diameter_of_inscribed_sphere edge_length)) = 81 * Real.sqrt 3 :=
by
  -- The proof is omitted by using sorry
  sorry

end volume_of_inscribed_cube_in_sphere_in_cube_l188_188391


namespace min_value_h_on_negative_l188_188884

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := f(x) + g(x) - 2

axiom f_odd (x : ℝ) : f (-x) = - f x
axiom g_odd (x : ℝ) : g (-x) = - g x
axiom h_max_on_positive (x : ℝ) (h_pos: 0 < x): h(x) ≤ 6

theorem min_value_h_on_negative : ∃ x : ℝ, x < 0 ∧ h(x) = -10 := sorry

end min_value_h_on_negative_l188_188884


namespace parabola_symmetric_points_l188_188200

theorem parabola_symmetric_points (a : ℝ) (h : 0 < a) :
  (∃ (P Q : ℝ × ℝ), (P ≠ Q) ∧ ((P.fst + P.snd = 0) ∧ (Q.fst + Q.snd = 0)) ∧
    (P.snd = a * P.fst ^ 2 - 1) ∧ (Q.snd = a * Q.fst ^ 2 - 1)) ↔ (3 / 4 < a) := 
sorry

end parabola_symmetric_points_l188_188200


namespace find_A_capital_l188_188003

noncomputable def money_put_in_business : ℝ := 5000

variables (total_profit : ℝ) (a_received : ℝ) (b_capital : ℝ) (a_manage_percentage : ℝ)

def A_money_received (total_profit : ℝ) (a_manage_percentage : ℝ) (A_share : ℝ) : ℝ :=
  (a_manage_percentage / 100) * total_profit + A_share

def remaining_profit (total_profit : ℝ) (a_manage_percentage : ℝ) : ℝ :=
  total_profit - (a_manage_percentage / 100) * total_profit

def B_share (remaining_profit : ℝ) (A_share : ℝ) : ℝ :=
  remaining_profit - A_share

def proportion (A_share : ℝ) (a_capital : ℝ) (B_share : ℝ) (b_capital : ℕ) : Prop :=
  A_share / a_capital = B_share / b_capital

theorem find_A_capital (a_capital : ℝ) (h1 : total_profit = 9600) 
  (h2 : a_received = 8160) (h3 : b_capital = 1000) (h4 : a_manage_percentage = 10) :
  a_capital = money_put_in_business :=
by
  let A_share := a_received - (a_manage_percentage / 100) * total_profit
  let remaining := remaining_profit total_profit a_manage_percentage 
  let B_share_val := B_share remaining A_share  
  let proportion_holds := proportion A_share a_capital B_share_val b_capital
  let answer := A_share * b_capital / B_share_val
  simp at answer
  exact sorry

end find_A_capital_l188_188003


namespace obtuse_triangle_probability_l188_188471

noncomputable def uniform_circle_distribution (radius : ℝ) : ProbabilitySpace (ℝ × ℝ) := sorry

def no_obtuse_triangle (points : list (ℝ × ℝ)) (center : (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k < 4 →
  ∀ (p₁ p₂ : (ℝ × ℝ)),
  (p₁ = points.nth i ∧ p₂ = points.nth j ∧ points.nth k ∉ set_obtuse_triangles p₁ p₂ center)

theorem obtuse_triangle_probability :
  let points := [⟨cos θ₀, sin θ₀⟩, ⟨cos θ₁, sin θ₁⟩, ⟨cos θ₂, sin θ₂⟩, ⟨cos θ₃, sin θ₃⟩]
  ∀ center : (ℝ × ℝ), 
  ∀ (h : set.center.circle center radius), 
  (no_obtuse_triangle points center) →
  Pr[(points)] = 3 / 32 :=
sorry

end obtuse_triangle_probability_l188_188471


namespace obtuse_triangle_probability_l188_188477

/-- 
Given four points chosen uniformly at random on a circle,
prove that the probability that no three of these points form an obtuse triangle with the circle's center is 3/32.
-/
theorem obtuse_triangle_probability : 
  let points := 4 in
  ∀ (P : Fin points → ℝ × ℝ) (random_uniform : ∀ i, random.uniform_on_circle i),
  (probability (no_three_form_obtuse_triangle P) = 3 / 32) :=
sorry

end obtuse_triangle_probability_l188_188477


namespace three_dice_probability_even_l188_188331

/-- A die is represented by numbers from 1 to 6. -/
def die := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

/-- Define an event where three dice are thrown, and we check if their sum is even. -/
def three_dice_sum_even (d1 d2 d3 : die) : Prop :=
  (d1.val + d2.val + d3.val) % 2 = 0

/-- Define the probability that a single die shows an odd number. -/
def prob_odd := 1 / 2

/-- Define the probability that a single die shows an even number. -/
def prob_even := 1 / 2

/-- Define the total probability for the sum of three dice to be even. -/
def prob_sum_even : ℚ :=
  prob_even ^ 3 + (3 * prob_odd ^ 2 * prob_even)

theorem three_dice_probability_even :
  prob_sum_even = 1 / 2 :=
by
  sorry

end three_dice_probability_even_l188_188331


namespace circumradius_of_triangle_ABC_l188_188222

-- Define the angles A and B
def angle_A := 45
def angle_B := 30

-- Define the lengths and relationships given in the problem
def DE := 4 * (Real.sqrt 2 - 1)

-- Define function to find circumradius using provided conditions
-- Dummy definitions for lengths of sides
noncomputable def a := 8 * Real.sqrt 2
noncomputable def b := 8

-- Circumradius relation derived in the solution
def circumradius (a b : Real) : Real := b

-- Define the theorem to prove
theorem circumradius_of_triangle_ABC : 
  circumradius a b = 8 := 
by 
  sorry

end circumradius_of_triangle_ABC_l188_188222


namespace monotonic_intervals_and_extrema_l188_188152

def f (x : ℝ) (b : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + b

def f' (x : ℝ) : ℝ := -3 * (x + 1) * (x - 3)

theorem monotonic_intervals_and_extrema (b : ℝ) :
  (∀ x, -1 < x ∧ x < 3 → f'(x) > 0) ∧
  (∀ x, (x < -1 ∨ x > 3) → f'(x) < 0) ∧
  f (-1) b = -5 + b ∧ f 3 b = 27 + b := by
  sorry

end monotonic_intervals_and_extrema_l188_188152


namespace meaningful_expression_l188_188185

theorem meaningful_expression (x : ℝ) : 
    (x + 2 > 0 ∧ x - 1 ≠ 0) ↔ (x > -2 ∧ x ≠ 1) :=
by
  sorry

end meaningful_expression_l188_188185


namespace hyperbola_asymptotes_angle_l188_188462

theorem hyperbola_asymptotes_angle {a b : ℝ} 
  (h1 : a > b)
  (h2 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h3 : ∀ θ : ℝ, θ = 45 → ∃ m1 m2 : ℝ, tan θ = abs (m1 - m2) / (1 + m1 * m2) ∧ m1 = b / a ∧ m2 = -b / a) : 
  a / b = sqrt 3 :=
by
  sorry

end hyperbola_asymptotes_angle_l188_188462


namespace magician_starting_decks_l188_188381

def starting_decks (price_per_deck earned remaining_decks : ℕ) : ℕ :=
  earned / price_per_deck + remaining_decks

theorem magician_starting_decks :
  starting_decks 2 4 3 = 5 :=
by
  sorry

end magician_starting_decks_l188_188381


namespace polynomial_solution_l188_188090

noncomputable def is_polynomial_solution (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, P.eval (2 * x) = (Polynomial.derivative P).eval x * (Polynomial.derivative (Polynomial.derivative P)).eval x

theorem polynomial_solution :
  ∀ P : Polynomial ℝ, is_polynomial_solution P → 
    P = Polynomial.C (4/9):ℝ * Polynomial.X ^ 3 + 
        Polynomial.C (3 * 4/9):ℝ * Polynomial.X ^ 2 + 
        Polynomial.C c * Polynomial.X ∧ 
        c ∈ ℝ := by
  sorry

end polynomial_solution_l188_188090


namespace smallest_enclosing_sphere_radius_l188_188825

noncomputable def radius_of_enclosing_sphere (r : ℝ) : ℝ :=
  let s := 6 -- side length of the cube
  let d := s * Real.sqrt 3 -- space diagonal of the cube
  (d + 2 * r) / 2

theorem smallest_enclosing_sphere_radius :
  radius_of_enclosing_sphere 2 = 3 * Real.sqrt 3 + 2 :=
by
  -- skipping the proof with sorry
  sorry

end smallest_enclosing_sphere_radius_l188_188825


namespace total_cost_eq_16000_l188_188734

theorem total_cost_eq_16000 (F M T : ℕ) (n : ℕ) (hF : F = 12000) (hM : M = 200) (hT : T = 16000) :
  T = F + M * n → n = 20 :=
by
  sorry

end total_cost_eq_16000_l188_188734


namespace find_polynomial_l188_188095

open Polynomial

theorem find_polynomial (P : ℝ[X]) (h : ∀ x : ℝ, eval (2*x) P = eval x (P.derivative) * eval x (P.derivative.derivative)) :
  P = C (4 / 9) * X^3 :=
by 
  sorry

end find_polynomial_l188_188095


namespace reading_difference_l188_188167

theorem reading_difference :
  let greg_pages := (7 * 18) + (14 * 22) in
  let brad_pages := (5 * 26) + (12 * 20) in
  let emily_pages := (3 * 15) + (7 * 24) + (7 * 18) in
  let total_pages_greg_brad := greg_pages + brad_pages in
  total_pages_greg_brad - emily_pages = 465 := by
  sorry

end reading_difference_l188_188167


namespace nth_equation_l188_188269

theorem nth_equation (n : ℕ) : (n + 1)^2 - 1 = n * (n + 2) := 
by 
  sorry

end nth_equation_l188_188269


namespace problem_inequality_l188_188564

theorem problem_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : 1 < a₁) (h₂ : 1 < a₂) (h₃ : 1 < a₃) (h₄ : 1 < a₄) (h₅ : 1 < a₅) :
  (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) ≤ 16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) :=
sorry

end problem_inequality_l188_188564


namespace cos_value_l188_188516

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 4 - α) = 1 / 3) :
  Real.cos (Real.pi / 4 + α) = 1 / 3 :=
sorry

end cos_value_l188_188516


namespace probability_non_obtuse_l188_188491

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l188_188491


namespace nested_sqrt_eq_five_l188_188064

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l188_188064


namespace proof_problem_l188_188121

variables {Point : Type} {Line Plane : Type}
variables [IsLine Line Point] [IsPlane Plane Point]

-- Define the relationships of parallel and perpendicular
variables (m n : Line) (α β : Plane)
variable parallel : Line → Plane → Prop
variable perp : Line → Plane → Prop

def problem_statement : Prop :=
  ∀ (m n : Line) (α : Plane),
    (parallel m n) →
    (perp n α) →
    (perp m α)

-- Now we assume the conditions and state the theorem
theorem proof_problem (m n : Line) (α : Plane)
  (h1 : parallel m n)
  (h2 : perp n α) : perp m α :=
sorry

end proof_problem_l188_188121


namespace thirteen_pow_2023_mod_1000_l188_188369

theorem thirteen_pow_2023_mod_1000 :
  (13^2023) % 1000 = 99 :=
sorry

end thirteen_pow_2023_mod_1000_l188_188369


namespace cost_of_bananas_l188_188801

/-- We are given that the rate of bananas is $6 per 3 kilograms. -/
def rate_per_3_kg : ℝ := 6

/-- We need to find the cost for 12 kilograms of bananas. -/
def weight_in_kg : ℝ := 12

/-- We are asked to prove that the cost of 12 kilograms of bananas is $24. -/
theorem cost_of_bananas (rate_per_3_kg weight_in_kg : ℝ) :
  (weight_in_kg / 3) * rate_per_3_kg = 24 :=
by
  sorry

end cost_of_bananas_l188_188801


namespace emery_family_trip_l188_188053

theorem emery_family_trip 
  (first_part_distance : ℕ) (first_part_time : ℕ) (total_time : ℕ) (speed : ℕ) (second_part_time : ℕ) :
  first_part_distance = 100 ∧ first_part_time = 1 ∧ total_time = 4 ∧ speed = 100 ∧ second_part_time = 3 →
  second_part_time * speed = 300 :=
by 
  sorry

end emery_family_trip_l188_188053


namespace average_monthly_growth_rate_l188_188312

theorem average_monthly_growth_rate (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, (1 + x) ^ 11 = a ∧ x = real.sqrt a^(1/11) - 1 :=
by {
  sorry
}

end average_monthly_growth_rate_l188_188312


namespace cars_meet_after_32_minutes_l188_188361

theorem cars_meet_after_32_minutes :
  ∀ (distance : ℝ) (speed_ratio : ℝ) (speed_b : ℝ),
  distance = 88 ∧ speed_ratio = 5/6 ∧ speed_b = 90 →
  let speed_a := speed_ratio * speed_b in
  let relative_speed := speed_a + speed_b in
  let time_to_meet := (distance / relative_speed) * 60 in
  time_to_meet ≈ 32 := 
by
  intros distance speed_ratio speed_b h,
  rcases h with ⟨h_distance, h_ratio, h_speed_b⟩,
  have h_speed_a : speed_a = (5/6) * 90, by simp [h_ratio, h_speed_b],
  have h_relative_speed : relative_speed = (5/6) * 90 + 90, by simp [relative_speed, h_speed_a, h_speed_b],
  have h_time_to_meet : time_to_meet = (88 / relative_speed) * 60, by simp [time_to_meet, h_distance, h_relative_speed],
  have approx_time : (88 / 165) * 60 ≈ 32, by norm_num,
  sorry

end cars_meet_after_32_minutes_l188_188361


namespace common_chord_circle_eq_l188_188163

theorem common_chord_circle_eq {a b : ℝ} (hb : b ≠ 0) :
  ∃ x y : ℝ, 
    (x^2 + y^2 - 2 * a * x = 0) ∧ 
    (x^2 + y^2 - 2 * b * y = 0) ∧ 
    (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0 :=
by sorry

end common_chord_circle_eq_l188_188163


namespace negate_all_teachers_excellent_l188_188528

-- Definitions of statements
def all_students_excellent_in_math := ∀ student, excellent_in_math student
def some_students_excellent_in_math := ∃ student, excellent_in_math student
def no_teachers_excellent_in_math := ∀ teacher, ¬ excellent_in_math teacher
def all_teachers_poor_in_math := ∀ teacher, poor_in_math teacher
def at_least_one_teacher_poor_in_math := ∃ teacher, poor_in_math teacher
def all_teachers_excellent_in_math := ∀ teacher, excellent_in_math teacher

-- Negation proof statement
theorem negate_all_teachers_excellent :
  (¬ all_teachers_excellent_in_math) ↔ at_least_one_teacher_poor_in_math :=
sorry

end negate_all_teachers_excellent_l188_188528


namespace evaluate_floor_ceil_l188_188443

theorem evaluate_floor_ceil :
  (⌊0.998⌋₊ : ℤ) + (⌈4.002⌉₊ : ℤ) = 5 := 
sorry

end evaluate_floor_ceil_l188_188443


namespace common_chord_diameter_circle_eqn_l188_188164
noncomputable theory

-- Mathematical conditions and the given circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * a * x = 0
def circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * b * y = 0

-- The derived equation of the circle whose diameter is the common chord
def desired_circle_eqn (a b x y : ℝ) : Prop := 
  (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0

-- Prove the two circles and their conditions result in the desired circle equation
theorem common_chord_diameter_circle_eqn (a b : ℝ) (hb : b ≠ 0) : 
  ∃ x y : ℝ, circle1 a x y ∧ circle2 b x y → desired_circle_eqn a b x y :=
by
  intro x y h
  sorry


end common_chord_diameter_circle_eqn_l188_188164


namespace find_angle_DAF_l188_188957

-- Definitions according to the problem conditions
def triangle (A B C : Type) : Prop := ∃ (α β γ : ℝ), α + β + γ = 180

def ∠ (A B C : Type) (θ : ℝ) : Prop := θ = ∠A + ∠B + ∠C

def angle_condition_1 (A B C : Type) : Prop :=
∠ (A B C) 60

def angle_condition_2 (A B C : Type) : Prop :=
∠ (A B C) 80

def perp (A B C D : Type) : Prop :=
∃ (α : ℝ), α = 90

def center_circumscribed_triangle (O A B C : Type) : Prop :=
O = center

def midpoint_arc (F B C : Type) : Prop :=
F = midpoint arc

-- The final problem to be proved
theorem find_angle_DAF (A B C D F O : Type)
  (triangle_ABC : triangle A B C)
  (angle_ACB : angle_condition_1 A B C)
  (angle_CBA : angle_condition_2 A B C)
  (perp_AD_bc : perp A B C D)
  (center_O : center_circumscribed_triangle O A B C)
  (midpoint_F : midpoint_arc F B C)
: ∠ DAF = 20 :=
sorry

end find_angle_DAF_l188_188957


namespace probability_divisible_by_15_l188_188198
-- Import the comprehensive Mathlib library.

-- Definition of the digits set and the key properties related to divisibility.
def digits : Finset ℕ := {1, 2, 3, 4, 5, 9}

-- Statement of the proof problem.
theorem probability_divisible_by_15 :
  let sum_digits := 1 + 2 + 3 + 4 + 5 + 9,
      total_arrangements := 6!,
      favorable_arrangements := (5!)
  in
  sum_digits % 3 = 0 ∧ 5 ∈ digits ∧ favorable_arrangements / total_arrangements = 1 / 6 := 
by
  sorry

end probability_divisible_by_15_l188_188198


namespace lines_intersection_l188_188297

theorem lines_intersection :
  ∃ (x y : ℝ), 
    (x - y = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end lines_intersection_l188_188297


namespace smallest_square_side_length_l188_188180

theorem smallest_square_side_length :
  ∃ s : ℝ, (∀ (x y : ℝ), (x^2 + y^2 = 5^2 ∧ x = 4 ∧ y = 3) → s ≥ (∥(4, 3)∥)) ∧ (s = (16 * sqrt 17) / 17) :=
sorry

end smallest_square_side_length_l188_188180


namespace rhombus_area_l188_188359

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 12) (h_d2 : d2 = 10) : 
  (d1 * d2) / 2 = 60 :=
by 
  rw [h_d1, h_d2]
  norm_num
  sorry

end rhombus_area_l188_188359


namespace geometric_sequence_n_eq_46_l188_188591

theorem geometric_sequence_n_eq_46 (q : ℝ) (h1 : q ≠ 1) (h2 : q ^ 45 = ∏ i in (finset.range 10), q ^ i) :
  ∃ n : ℕ, ∏ i in finset.range 10, q^i = q^(n - 1) ∧ n = 46 := 
by 
  sorry

end geometric_sequence_n_eq_46_l188_188591


namespace travel_distance_l188_188627

theorem travel_distance (x t : ℕ) (h : t = 14400) (h_eq : 12 * x + 12 * (2 * x) = t) : x = 400 :=
by
  sorry

end travel_distance_l188_188627


namespace area_of_region_is_4_l188_188818

noncomputable def area_of_bounded_region : ℝ :=
  let f1 := λ θ : ℝ, real.tan θ in
  let f2 := λ θ : ℝ, real.cot θ in
  let line_x := 2 in
  let line_y := 2 in
  (line_x * line_y : ℝ)

theorem area_of_region_is_4 :
  area_of_bounded_region = 4 :=
by
  -- Proof is omitted
  sorry

end area_of_region_is_4_l188_188818


namespace tan_alpha_l188_188124

theorem tan_alpha (α : ℝ) (hα1 : α > π / 2) (hα2 : α < π) (h_sin : Real.sin α = 4 / 5) : Real.tan α = - (4 / 3) :=
by 
  sorry

end tan_alpha_l188_188124


namespace hexagon_chord_length_l188_188000

noncomputable def chord_length (m n : ℕ) (h : gcd m n = 1) : ℝ :=
  (m : ℝ) / (n : ℝ)

theorem hexagon_chord_length :
  ∃ (m n : ℕ), gcd m n = 1 ∧ chord_length m n = 560 / 81 ∧ (m + n = 641) :=
by
  existsi 560
  existsi 81
  split
  · sorry
  · split
    · sorry
    · sorry

end hexagon_chord_length_l188_188000


namespace kim_gallons_l188_188052

theorem kim_gallons :
  ∃ (K : ℕ), K = 24 ∧ 
  let isabella_gallons = 25 in
  let non_discounted_limit = 6 in
  let discount_rate = 0.10 in
  let isabella_discounted_gallons = isabella_gallons - non_discounted_limit in
  let kim_discounted_gallons = K - non_discounted_limit in
  let discount_ratio = 1.0857142857142861 in
  (isabella_discounted_gallons * discount_rate) = 
  (discount_ratio * kim_discounted_gallons * discount_rate) := 
begin
  sorry
end

end kim_gallons_l188_188052


namespace product_increase_l188_188952

theorem product_increase (a b c : ℕ) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 := by
  sorry

end product_increase_l188_188952


namespace tan_nine_pi_over_three_l188_188086

theorem tan_nine_pi_over_three : Real.tan (9 * Real.pi / 3) = 0 := by
  sorry

end tan_nine_pi_over_three_l188_188086


namespace range_of_a_l188_188616

theorem range_of_a (a : ℝ) (f g : ℝ → ℝ) 
  (hf : ∀ x : ℝ, f x = |Real.log x|)
  (hg : ∀ x : ℝ, g x = f x - a * x)
  (hzeros : ∃ x1 x2 x3 ∈ Ioo (0:ℝ) (Real.exp 2), g x1 = 0 ∧ g x2 = 0 ∧ g x3 = 0) :
  a ∈ Set.Ioo (2 / Real.exp 2) (1 / Real.exp 1) :=
sorry

end range_of_a_l188_188616


namespace quadrilateral_area_conditions_l188_188590

-- Define the geometric setup and conditions.
variables (AB BC CD AD BF : ℝ) (O : ℝ)

/-- Given the perpendicular conditions and tangency to the circle, we need to show 
that for the given values of AB and CD, the area of the quadrilateral is not an integer. -/
theorem quadrilateral_area_conditions (h1 : AB * CD = BF ^ 2)
    (h2 : BC * BC = 2 * BF * BF)
    (h3 : O = 0) (h4 : AD = sqrt (AB ^ 2 + CD ^ 2)) :
    (AB = 4 ∧ CD = 2) ∨ (AB = 6 ∧ CD = 3) ∨ (AB = 8 ∧ CD = 4) ∨ 
    (AB = 10 ∧ CD = 5) ∨ (AB = 12 ∧ CD = 6) → 
    ¬ (∃ k : ℤ, (AB + CD) * BF = k ^ 2) :=
by
  sorry

end quadrilateral_area_conditions_l188_188590


namespace triangle_area_l188_188017

theorem triangle_area (base height : ℕ) (h_base : base = 123) (h_height : height = 10) :
  (base * height) / 2 = 615 :=
by
  rw [h_base, h_height]
  norm_num
  -- proof would continue here
  sorry

end triangle_area_l188_188017


namespace valid_propositions_l188_188403

theorem valid_propositions :
  (∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧ (∃ n : ℝ, ∀ m : ℝ, m * n = m) :=
by
  sorry

end valid_propositions_l188_188403


namespace simplify_expression_l188_188730

theorem simplify_expression (c d : ℝ) (hc : c = 2) (hd : d = 1/4) :
  (√(c - d) / (c^2 * √(2 * c))) * (√((c - d) / (c + d)) + √((c^2 + c * d) / (c^2 - c * d))) = 1 / 3 :=
by
  rw [hc, hd]
  -- Further steps to prove the equality would go here, ending with the required result
  sorry

end simplify_expression_l188_188730


namespace factorize_expression_l188_188828

theorem factorize_expression (x y a : ℝ) : x * (a - y) - y * (y - a) = (x + y) * (a - y) := 
by 
  sorry

end factorize_expression_l188_188828


namespace closed_form_expression_l188_188051

noncomputable def f (a d : ℕ) : ℕ

theorem closed_form_expression (a d : ℕ) (ha : a ≥ 1)
  (base_case : f 2 d ≤ 2^d + 1)
  (inductive_hyp : ∀ a > 1, f (2^(a-1)) d ≤ (2^(a-1) - 1) * 2^d + 1)
  (known_inequality : ∀ a, f (2^a) d ≤ f 2 d + 2 * (f (2^(a-1)) d - 1))
  : f (2^a) d = (2^a - 1) * 2^d + 1 :=
sorry

end closed_form_expression_l188_188051


namespace base_extension_l188_188390

/-- Define the initial inclination angle and final inclination angle. -/
def initial_inclination : ℝ := 20
def final_inclination : ℝ := 10
def hypotenuse : ℝ := 1

/-- Define the base length calculations given the inclination angles. -/
def base_length (angle : ℝ) : ℝ := hypotenuse * Real.cos (angle * Real.pi / 180)

/-- Define the original and new base lengths. -/
def base_original : ℝ := base_length initial_inclination
def base_new : ℝ := base_length final_inclination

/-- Prove the extension needed to reduce the inclination from 20° to 10° is 1 kilometer. -/
theorem base_extension : base_new - base_original = 1 := sorry

end base_extension_l188_188390


namespace product_of_digits_l188_188568

theorem product_of_digits (A B : ℕ) 
  (h1 : A + B = 16) 
  (h2 : (10 * A + B) % 4 = 0) : 
  A * B = 64 := 
begin
  -- Proof is not required
  sorry
end

end product_of_digits_l188_188568


namespace common_chord_circle_eq_l188_188162

theorem common_chord_circle_eq {a b : ℝ} (hb : b ≠ 0) :
  ∃ x y : ℝ, 
    (x^2 + y^2 - 2 * a * x = 0) ∧ 
    (x^2 + y^2 - 2 * b * y = 0) ∧ 
    (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0 :=
by sorry

end common_chord_circle_eq_l188_188162


namespace steve_sold_50_fish_l188_188654

variable (x : ℕ)

/-- Steve starts with a stock of 200 fish, sells some fish, a third of the remaining fish become spoiled,
a new stock of 200 fish arrives, and he has 300 fish in stock. Prove that the number of fish sold is 50. -/
theorem steve_sold_50_fish
  (h_initial_stock : 200)
  (h_final_stock : 300)
  (h_new_stock : 200)
  (h_spoiled : (200 - x) / 3)
  (h_equation : 200 - x - (200 - x) / 3 + 200 = 300) :
  x = 50 := by
  sorry

end steve_sold_50_fish_l188_188654


namespace sum_of_digits_l188_188714

theorem sum_of_digits (a b : ℕ) (h1 : a = 3 ^ 2005) (h2 : b = 7 ^ 2007) : 
  sum_of_digits (a * b * 2) = 17 :=
by
  sorry

end sum_of_digits_l188_188714


namespace geometric_progression_t_exists_t_l188_188610

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, a n > 0)
variable (h1 : ∀ n, a (n + 2) / a n = 1 / 4)
variable (h2 : ∀ k n, k ≠ n ∧ |k - n| ≠ 1 → a (k + 1) / a k + a (n + 1) / a n = 1)

theorem geometric_progression :
  ∃ r, ∀ n, a (n + 1) = r * a n ∧ r = 1 / 2 :=
sorry

theorem t_exists_t :
  (∃ t > 0, ∀ n, sqrt (a (n + 1)) ≤ 1 / 2 * a n + t) :=
sorry

end geometric_progression_t_exists_t_l188_188610


namespace sum_projections_GS_GT_GU_l188_188593

/-- 
In triangle XYZ with side lengths XY = 5, XZ = 7, and YZ = 6, 
the medians intersect at centroid G. 
Let the projections of G onto sides YZ, XZ, and XY be denoted as S, T, and U respectively.
We want to prove that the sum of the projections GS, GT, and GU is equal to 122 * sqrt 6 / 105.
-/

noncomputable def triangle_XYZ (XY XZ YZ : ℝ) (M G S T U : Point) : Prop :=
XY = 5 ∧ XZ = 7 ∧ YZ = 6 ∧
is_centroid G (triangle XYZ) ∧
is_projection G YZ S ∧
is_projection G XZ T ∧
is_projection G XY U

theorem sum_projections_GS_GT_GU :
  ∀ {XY XZ YZ : ℝ} {M G S T U : Point},
  triangle_XYZ XY XZ YZ M G S T U →
  (dist G S + dist G T + dist G U = 122 * Real.sqrt 6 / 105) :=
by
  intros XY XZ YZ M G S T U h
  sorry

end sum_projections_GS_GT_GU_l188_188593


namespace construct_square_on_triangle_l188_188127

noncomputable theory

-- Define Triangle
structure Triangle :=
  (A B C : Point)

-- Given a triangle ABC, prove that a square can be constructed...
theorem construct_square_on_triangle (ABC : Triangle) :
  ∃ B' C' D' E' : Point, 
  ∃ AB AC BC : Line,
  (Vertex B' on ABC.AB) ∧ 
  (Vertex C' on ABC.AC) ∧ 
  (Adjacent B' C' on ABC.BC) ∧ 
  (Adjacent C' D' on ABC.BC) ∧ 
  is_square B' C' D' E' :=
sorry

end construct_square_on_triangle_l188_188127


namespace boat1_realizes_missing_cargo_in_40_minutes_l188_188365

noncomputable def minutes_until_boat1_realizes_missing : ℕ := 40

theorem boat1_realizes_missing_cargo_in_40_minutes 
  (Vα Vβ V_water : ℝ) 
  (h : Vα = 2 * Vβ) 
  (meet_time : ℕ) 
  (box1_meet_time : ℕ) 
  (box2_meet_time : ℕ) :
  meet_time = 20 →
  box1_meet_time = 10 → 
  meet_time + 2 * box1_meet_time = minutes_until_boat1_realizes_missing :=
by
  intro h_meet_time h_box1_meet_time
  unfold minutes_until_boat1_realizes_missing
  rw [h_meet_time, h_box1_meet_time]
  show 20 + 2 * 10 = 40
  rfl

end boat1_realizes_missing_cargo_in_40_minutes_l188_188365


namespace probability_no_obtuse_triangle_correct_l188_188473

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l188_188473


namespace range_of_a_l188_188533

noncomputable def f (a x : ℝ) : ℝ := Real.sin x + 0.5 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici 0, f a x ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l188_188533


namespace obtuse_triangle_probability_l188_188469

noncomputable def uniform_circle_distribution (radius : ℝ) : ProbabilitySpace (ℝ × ℝ) := sorry

def no_obtuse_triangle (points : list (ℝ × ℝ)) (center : (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k < 4 →
  ∀ (p₁ p₂ : (ℝ × ℝ)),
  (p₁ = points.nth i ∧ p₂ = points.nth j ∧ points.nth k ∉ set_obtuse_triangles p₁ p₂ center)

theorem obtuse_triangle_probability :
  let points := [⟨cos θ₀, sin θ₀⟩, ⟨cos θ₁, sin θ₁⟩, ⟨cos θ₂, sin θ₂⟩, ⟨cos θ₃, sin θ₃⟩]
  ∀ center : (ℝ × ℝ), 
  ∀ (h : set.center.circle center radius), 
  (no_obtuse_triangle points center) →
  Pr[(points)] = 3 / 32 :=
sorry

end obtuse_triangle_probability_l188_188469


namespace arithmetic_sequence_properties_l188_188138

-- Definitions and conditions
def S (n : ℕ) : ℤ := -2 * n^2 + 15 * n

-- Statement of the problem as a theorem
theorem arithmetic_sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = 17 - 4 * (n + 1)) ∧
  (∃ n : ℕ, S n = 28 ∧ ∀ m : ℕ, S m ≤ S n) :=
by {sorry}

end arithmetic_sequence_properties_l188_188138


namespace geometric_sequence_S5_equals_l188_188950

theorem geometric_sequence_S5_equals :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    a 1 = 1 → 
    (a 3 + a 4) / (a 1 + a 2) = 4 → 
    ((S5 : ℤ) = 31 ∨ (S5 : ℤ) = 11) :=
by
  sorry

end geometric_sequence_S5_equals_l188_188950


namespace distance_walked_on_first_day_l188_188008

theorem distance_walked_on_first_day (a : ℕ → ℝ) (a0 : ℝ) :
  (a 0 = a0) → 
  (∀ n, a (n+1) = a n / 2) → 
  (∑ n in Finset.range 6, a n = 378) → 
  a 0 = 192 :=
by
  -- Define the initial conditions
  intro ha0 hratio hsum
  -- Apply the given conditions and derive the solution
  sorry

end distance_walked_on_first_day_l188_188008


namespace length_of_AB_correct_l188_188298

def length_of_AB (ABCD EFGD : Square) (BF : ℝ) (area: ℝ) : ℝ :=
  let AB := (ABCD.side_length : ℝ)
  if (BF = 10) ∧ (area = 35) then 2 * Real.sqrt 2 else 0

theorem length_of_AB_correct (ABCD EFGD : Square) (BF : 10) (area: 35) : 
  length_of_AB ABCD EFGD BF area = 2 * Real.sqrt 2 := 
by sorry

end length_of_AB_correct_l188_188298


namespace common_ratio_of_geometric_sequence_l188_188589

variables (a : ℕ → ℝ) (q : ℝ)
axiom h1 : a 1 = 2
axiom h2 : ∀ n : ℕ, a (n + 1) - a n ≠ 0 -- Common difference is non-zero
axiom h3 : a 3 = (a 1) * q
axiom h4 : a 11 = (a 1) * q^2
axiom h5 : a 11 = a 1 + 5 * (a 3 - a 1)

theorem common_ratio_of_geometric_sequence : q = 4 := 
by sorry

end common_ratio_of_geometric_sequence_l188_188589


namespace prime_bound_l188_188995

-- The definition for the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry  -- placeholder for the primorial definition

-- The main theorem to prove
theorem prime_bound (n : ℕ) : nth_prime n ≤ 2 ^ 2 ^ (n - 1) := sorry

end prime_bound_l188_188995


namespace max_area_of_triangle_PQR_l188_188863

theorem max_area_of_triangle_PQR (A B C : Point) (t_ABC : ℝ) :
  (∀ Q : Point, Q ∈ line_segment B C →
    ∃ P R : Point, 
    P ∈ midpoint A B ∧ 
    R ∈ midpoint A C ∧ 
    area_of_triangle P Q R = 1/4 * t_ABC) :=
by
  sorry

end max_area_of_triangle_PQR_l188_188863


namespace area_of_right_triangle_l188_188221

-- Define the input conditions
variables (a b : ℝ)

-- Define the given right triangle ABC with specific properties
structure triangle_props :=
  (right_angle_at_C : Prop)
  (angle_bisector_CL : ℝ)
  (median_CM : ℝ)

-- The input conditions for the problem
def input_conditions : triangle_props :=
{ right_angle_at_C := true,
  angle_bisector_CL := a,
  median_CM := b }

-- Define the proof problem
theorem area_of_right_triangle (x y : ℝ) (h : triangle_props) :
  h = input_conditions →
  ∃ S, S = (a^2 + a * real.sqrt(a^2 + 8 * b^2)) / 4 :=
by
  -- We provide the condition verifying the input condition is met
  intro h_eq,
  have h_right : h.right_angle_at_C = true :=
    by rw [h_eq, input_conditions],
  use (a^2 + a * real.sqrt (a^2 + 8 * b^2) ) / 4,
  sorry -- Proof will go here (not required by the problem statement)

end area_of_right_triangle_l188_188221


namespace hexagon_segments_same_length_probability_l188_188982

theorem hexagon_segments_same_length_probability :
  let S := {s : ℝ // s = side ∨ s = diagonal}
  (length S = 15) ∧
  (∃! side ∈ S, side_length side = base_length) ∧ 
  (∃! diagonal ∈ S, diagonal_length diagonal = base_diagonal_length) ∧ 
  (6 * base_length + 9 * base_diagonal_length = 15 * side_length) →
  prob_of_same_length (S: set ℝ) = 17 / 35 := by
  sorry

end hexagon_segments_same_length_probability_l188_188982


namespace find_ratio_OD_CF_l188_188629

noncomputable def hyperbola (x : ℝ) : Prop := x > 0 ∧ ∃ y, y = 1/x

variable (a b : ℝ)
variable (A : ℝ × ℝ) (B : ℝ × ℝ)
variable (O C F : ℝ × ℝ)

def point_on_hyperbola (P : ℝ × ℝ) := hyperbola P.1 ∧ P.2 = 1 / P.1
def line_perpendicular (l1 l2 : ℝ × ℝ → Prop) := ∀ x1 y1 x2 y2, l1 (x1, y1) → l2 (x2, y2) → x1 * x2 + y1 * y2 = 0
def passes_through_origin (l : ℝ × ℝ → Prop) := l (0,0)

axiom O_midpoint_CM (C M O : ℝ × ℝ) : O = ((C.1 + M.1) / 2, (C.2 + M.2) / 2)

axiom O_is_origin : O = (0,0)

def circle_through_points (P Q R : ℝ × ℝ) (F : ℝ × ℝ) := sorry -- placeholder for circle condition

theorem find_ratio_OD_CF :
  (point_on_hyperbola A) →
  (point_on_hyperbola B) →
  (line_perpendicular (λ P, P = O) (λ P, (A.2 - B.2) * P.1 − (A.1 - B.1) * P.2)) →
  (passes_through_origin (λ P, (A.2 - B.2) * P.1 − (A.1 - B.1) * P.2)) →
  (point_on_hyperbola C) →
  (circle_through_points A B C F) →
  O = (0,0) →
  OD:CF = 1:2 :=
sorry

end find_ratio_OD_CF_l188_188629


namespace proof_problem_l188_188853

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := 
  (Real.sin (2 * x), 2 * Real.cos x ^ 2 - 1)

noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := 
  (Real.sin θ, Real.cos θ)

noncomputable def f (x θ : ℝ) : ℝ := 
  (vector_a x).1 * (vector_b θ).1 + (vector_a x).2 * (vector_b θ).2

theorem proof_problem 
  (θ : ℝ) 
  (hθ : 0 < θ ∧ θ < π) 
  (h1 : f (π / 6) θ = 1) 
  (x : ℝ) 
  (hx : -π / 6 ≤ x ∧ x ≤ π / 4) :
  θ = π / 3 ∧
  (∀ x, f x θ = f (x + π) θ) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≤ 1) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≥ -0.5) :=
by
  sorry

end proof_problem_l188_188853


namespace range_of_a_l188_188574

theorem range_of_a (a : ℝ) (h : ∀ x ∈ set.Ioo 0 1, (x^2 - a*x + 1) ≥ 0) : a ≤ 2 := by
  sorry

end range_of_a_l188_188574


namespace students_overlap_difference_l188_188733

theorem students_overlap_difference (total_students geometry_students biology_students : ℕ)
  (H1 : total_students = 232)
  (H2 : geometry_students = 144)
  (H3 : biology_students = 119) :
  let max_overlap := min geometry_students biology_students in
  let min_overlap := geometry_students + biology_students - total_students in
  max_overlap - min_overlap = 88 :=
by
  rw [H1, H2, H3]
  have max_overlap := min 144 119
  have min_overlap := 144 + 119 - 232
  show max_overlap - min_overlap = 88
  rw [min_eq_right (le_of_lt (lt_succ_self 119))]
  rw [←nat.add_sub_assoc (le_of_lt (lt_succ_self 232)) 144 119]
  norm_num
  sorry

end students_overlap_difference_l188_188733


namespace intersection_A_B_l188_188998

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | x > 0}

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {1} := 
by {
  sorry
}

end intersection_A_B_l188_188998


namespace phase_shift_sin_l188_188102

def b : ℝ := 5
def c : ℝ := 2 * Real.pi
def phaseShift (b c : ℝ) := c / b

theorem phase_shift_sin : phaseShift b c = (2 * Real.pi) / 5 := 
by 
  sorry

end phase_shift_sin_l188_188102


namespace race_outcomes_count_l188_188398

def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Frank"]

theorem race_outcomes_count : 
  let outcomes := [(p1, p2, p3) | p1 ← participants, p2 ← participants, p3 ← participants, 
                  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ p3 ≠ "Frank"] in
  outcomes.length = 120 :=
  sorry

end race_outcomes_count_l188_188398


namespace each_person_ate_slices_l188_188976

def slices_per_person (num_pizzas : ℕ) (slices_per_pizza : ℕ) (num_people : ℕ) : ℕ :=
  (num_pizzas * slices_per_pizza) / num_people

theorem each_person_ate_slices :
  ∀ (num_pizzas slices_per_pizza num_people : ℕ),
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  num_people = 6 →
  slices_per_person num_pizzas slices_per_pizza num_people = 4 :=
by
  intros num_pizzas slices_per_pizza num_people hpizzas hslices hpeople
  rw [hpizzas, hslices, hpeople]
  simp [slices_per_person]
  sorry

end each_person_ate_slices_l188_188976


namespace undefined_integer_count_l188_188111

noncomputable def expression (x : ℤ) : ℚ := (x^2 - 16) / ((x^2 - x - 6) * (x - 4))

theorem undefined_integer_count : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x^2 - x - 6) * (x - 4) = 0) ∧ S.card = 3 :=
  sorry

end undefined_integer_count_l188_188111


namespace angle_QSR_eq_angle_AFQ_l188_188499

open_locale classical
noncomputable theory

variables {A B C F Q S T P R : Point}

-- Given conditions
variables (triangle_ABC : Triangle A B C)
variables (incircle_touch : Incircle triangle_ABC (Segment A B) F)
variables (midpoints_minor_arcs : MidpointsMinorArcs A C Q B C S)
variables (A_excircle_touch : Excircle A (Segment B C) T)
variables (intersection_P : Line QS ∩ Line BC = {P})
variables (intersection_R : Circle TPS ∩ Circle ABC = {R})

-- To prove
theorem angle_QSR_eq_angle_AFQ :
  ∠QSR = ∠AFQ :=
sorry

end angle_QSR_eq_angle_AFQ_l188_188499


namespace probability_no_obtuse_triangle_correct_l188_188476

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l188_188476


namespace negation_of_sine_bound_l188_188899

theorem negation_of_sine_bound (p : ∀ x : ℝ, Real.sin x ≤ 1) : ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x₀ : ℝ, Real.sin x₀ > 1 := 
by 
  sorry

end negation_of_sine_bound_l188_188899


namespace no_obtuse_triangle_probability_l188_188486

noncomputable def prob_no_obtuse_triangle : ℝ :=
  9 / 128

theorem no_obtuse_triangle_probability :
  let circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 },
      points : List (ℝ × ℝ) × ℝ := (List.replicate 4 (0, 0), 1) -- assuming uniform rand points chosen
  in 
  ∀ (A₀ A₁ A₂ A₃ : ℝ × ℝ), A₀ ∈ circle → A₁ ∈ circle → A₂ ∈ circle → A₃ ∈ circle →
  (prob_no_obtuse_triangle = 9 / 128) :=
by
  sorry

end no_obtuse_triangle_probability_l188_188486


namespace modulus_of_z_l188_188888

open Complex

theorem modulus_of_z : 
  let z := (1 - complex.I) / (3 + 4 * complex.I)
  in complex.abs z = real.sqrt 2 / 5 :=
by
  let z := (1 - complex.I) / (3 + 4 * complex.I)
  let lhs := complex.abs z
  let rhs := real.sqrt 2 / 5
  show lhs = rhs
  sorry

end modulus_of_z_l188_188888


namespace average_speed_of_planes_l188_188330

-- Definitions for the conditions
def num_passengers_plane1 : ℕ := 50
def num_passengers_plane2 : ℕ := 60
def num_passengers_plane3 : ℕ := 40
def base_speed : ℕ := 600
def speed_reduction_per_passenger : ℕ := 2

-- Calculate speeds of each plane according to given conditions
def speed_plane1 := base_speed - num_passengers_plane1 * speed_reduction_per_passenger
def speed_plane2 := base_speed - num_passengers_plane2 * speed_reduction_per_passenger
def speed_plane3 := base_speed - num_passengers_plane3 * speed_reduction_per_passenger

-- Calculate the total speed and average speed
def total_speed := speed_plane1 + speed_plane2 + speed_plane3
def average_speed := total_speed / 3

-- The theorem to prove the average speed is 500 MPH
theorem average_speed_of_planes : average_speed = 500 := by
  sorry

end average_speed_of_planes_l188_188330


namespace curve_is_line_l188_188837

theorem curve_is_line (r θ x y : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x + y = 1 := by
  sorry

end curve_is_line_l188_188837


namespace percent_palindromes_with_seven_l188_188772

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_valid_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 2000

def count_with_seven (low high : ℕ) : ℕ :=
  (list.range' low (high - low)).count (λ x, is_palindrome x ∧ x.to_string.contains '7')

def total_palindromes (low high : ℕ) : ℕ :=
  (list.range' low (high - low)).count (λ x, is_palindrome x)

theorem percent_palindromes_with_seven : 
  1000≤ x → x < 2000 → ((count_with_seven 1000 2000) * 100 / (total_palindromes 1000 2000) = 18) := 
by sorry

end percent_palindromes_with_seven_l188_188772


namespace problem_discussion_organization_l188_188033

theorem problem_discussion_organization 
    (students : Fin 20 → Finset (Fin 20))
    (problems : Fin 20 → Finset (Fin 20))
    (h1 : ∀ s, (students s).card = 2)
    (h2 : ∀ p, (problems p).card = 2)
    (h3 : ∀ s p, s ∈ problems p ↔ p ∈ students s) : 
    ∃ (discussion : Fin 20 → Fin 20), 
        (∀ s, discussion s ∈ students s) ∧ 
        (Finset.univ.image discussion).card = 20 :=
by
  -- proof goes here
  sorry

end problem_discussion_organization_l188_188033


namespace polynomial_identity_l188_188173

theorem polynomial_identity (a b c d e f : ℤ)
  (h_eq : ∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 770 := by
  sorry

end polynomial_identity_l188_188173


namespace concyclic_points_diameter_l188_188128

theorem concyclic_points_diameter (A B C O D E F G H K L M N : Point)
  (a b c : ℝ)
  (h1 : acute_triangle A B C)
  (h2 : circumcircle A B C O)
  (h3 : foot D A B = D ∧ foot E B C = E ∧ foot F C A = F)
  (h4 : arc_intersects EF (arc_AB O) G ∧ arc_intersects EF (arc_AC O) H)
  (h5 : line_intersects DF BG K ∧ line_intersects DF BH L)
  (h6 : line_intersects DE CG M ∧ line_intersects DE CH N)
  (h7 : length BC = a)
  (h8 : length CA = b)
  (h9 : length AB = c) :
  concyclic K L M N ∧ diameter_circle K L M N = sqrt(2 * (b^2 + c^2 - a^2)) :=
sorry

end concyclic_points_diameter_l188_188128


namespace least_possible_sum_l188_188184

theorem least_possible_sum (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 21 * (q + 1)) : p + q = 5 :=
sorry

end least_possible_sum_l188_188184


namespace gumballs_remaining_l188_188789

theorem gumballs_remaining (Alicia_gumballs : ℕ) (Pedro_gumballs : ℕ) (Total_gumballs : ℕ) (Gumballs_taken_out : ℕ)
  (h1 : Alicia_gumballs = 20)
  (h2 : Pedro_gumballs = Alicia_gumballs + 3 * Alicia_gumballs)
  (h3 : Total_gumballs = Alicia_gumballs + Pedro_gumballs)
  (h4 : Gumballs_taken_out = 40 * Total_gumballs / 100) :
  Total_gumballs - Gumballs_taken_out = 60 := by
  sorry

end gumballs_remaining_l188_188789


namespace find_a_l188_188110

variable {x y : Type}

-- Definitions for the problem conditions
def linear_relationship (x y : ℝ) : Prop := y = (1/3) * x + a

def sum_conditions (x_list y_list : list ℝ) : Prop :=
  list.sum x_list = 6 ∧ list.sum y_list = 3

-- The proof problem statement
theorem find_a (x_list y_list : list ℝ) (h1 : sum_conditions x_list y_list) :
  let a := (1 / 8) in
  ∃ a, linear_relationship x_list y_list = a :=
sorry

end find_a_l188_188110


namespace each_person_ate_slices_l188_188975

def slices_per_person (num_pizzas : ℕ) (slices_per_pizza : ℕ) (num_people : ℕ) : ℕ :=
  (num_pizzas * slices_per_pizza) / num_people

theorem each_person_ate_slices :
  ∀ (num_pizzas slices_per_pizza num_people : ℕ),
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  num_people = 6 →
  slices_per_person num_pizzas slices_per_pizza num_people = 4 :=
by
  intros num_pizzas slices_per_pizza num_people hpizzas hslices hpeople
  rw [hpizzas, hslices, hpeople]
  simp [slices_per_person]
  sorry

end each_person_ate_slices_l188_188975


namespace train_trip_length_l188_188016

theorem train_trip_length (x D : ℝ) (h1 : D > 0) (h2 : x > 0) 
(h3 : 2 + 3 * (D - 2 * x) / (2 * x) + 1 = (x + 240) / x + 1 + 3 * (D - 2 * x - 120) / (2 * x) - 0.5) 
(h4 : 3 + 3 * (D - 2 * x) / (2 * x) = 7) :
  D = 640 :=
by
  sorry

end train_trip_length_l188_188016


namespace units_digit_of_fibonacci_f_15_l188_188668

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n + 1) + fibonacci n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fibonacci_f_15 :
  units_digit (fibonacci (fibonacci 15)) = 5 := 
sorry

end units_digit_of_fibonacci_f_15_l188_188668


namespace salary_increase_l188_188602

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 0.65 * S = 0.5 * S + (P / 100) * (0.5 * S)) : P = 30 := 
by
  -- proof goes here
  sorry

end salary_increase_l188_188602


namespace M_is_always_positive_l188_188172

theorem M_is_always_positive (x y : ℝ) : 
  let M := 3 * x ^ 2 - 8 * x * y + 9 * y ^ 2 - 4 * x + 6 * y + 13 in
  M > 0 :=
by
  let M := 3 * x ^ 2 - 8 * x * y + 9 * y ^ 2 - 4 * x + 6 * y + 13
  sorry

end M_is_always_positive_l188_188172


namespace find_uv_l188_188099

open Real

def vec1 : ℝ × ℝ := (3, -2)
def vec2 : ℝ × ℝ := (-1, 2)
def vec3 : ℝ × ℝ := (1, -1)
def vec4 : ℝ × ℝ := (4, -7)
def vec5 : ℝ × ℝ := (-3, 5)

theorem find_uv (u v : ℝ) :
  vec1 + ⟨4 * u, -7 * u⟩ = vec2 + ⟨-3 * v, 5 * v⟩ + vec3 →
  u = 3 / 4 ∧ v = -9 / 4 :=
by
  sorry

end find_uv_l188_188099


namespace geometric_progression_numbers_l188_188326

theorem geometric_progression_numbers (a r : ℝ) (h1: a + a * r + a * r^2 = 65) (h2: a^3 * r^3 = 3375) : 
  {s : set ℝ | s = {a, a * r, a * r^2}} = {5, 15, 45} ∨ {s : set ℝ | s = {a, a * r, a * r^2}} = {45, 15, 5} :=
sorry

end geometric_progression_numbers_l188_188326


namespace point_in_which_quadrant_l188_188563

noncomputable def quadrant_of_point (x y : ℝ) : String :=
if (x > 0) ∧ (y > 0) then
    "First"
else if (x < 0) ∧ (y > 0) then
    "Second"
else if (x < 0) ∧ (y < 0) then
    "Third"
else if (x > 0) ∧ (y < 0) then
    "Fourth"
else
    "On Axis"

theorem point_in_which_quadrant (α : ℝ) (h1 : π / 2 < α) (h2 : α < π) : quadrant_of_point (Real.sin α) (Real.cos α) = "Fourth" :=
by {
    sorry
}

end point_in_which_quadrant_l188_188563


namespace lisa_eggs_total_l188_188625

def children_mon_tue := 4 * 2 * 2
def husband_mon_tue := 3 * 2 
def lisa_mon_tue := 2 * 2
def total_mon_tue := children_mon_tue + husband_mon_tue + lisa_mon_tue

def children_wed := 4 * 3
def husband_wed := 4
def lisa_wed := 3
def total_wed := children_wed + husband_wed + lisa_wed

def children_thu := 4 * 1
def husband_thu := 2
def lisa_thu := 1
def total_thu := children_thu + husband_thu + lisa_thu

def children_fri := 4 * 2
def husband_fri := 3
def lisa_fri := 2
def total_fri := children_fri + husband_fri + lisa_fri

def total_week := total_mon_tue + total_wed + total_thu + total_fri

def weeks_per_year := 52
def yearly_eggs := total_week * weeks_per_year

def children_holidays := 4 * 2 * 8
def husband_holidays := 2 * 8
def lisa_holidays := 2 * 8
def total_holidays := children_holidays + husband_holidays + lisa_holidays

def total_annual_eggs := yearly_eggs + total_holidays

theorem lisa_eggs_total : total_annual_eggs = 3476 := by
  sorry

end lisa_eggs_total_l188_188625


namespace equilateral_triangle_division_l188_188868

theorem equilateral_triangle_division :
  ∀ (T : Triangle), 
    (is_equilateral T) → 
    (∃ (P₁ P₂ : Polygon), 
      (num_sides P₁ = 2020) ∧ 
      (num_sides P₂ = 2021) ∧ 
      (is_partition_of_triangle T P₁ P₂)) := 
by
  sorry

end equilateral_triangle_division_l188_188868


namespace sufficient_not_necessary_condition_l188_188321

theorem sufficient_not_necessary_condition (α : ℝ) :
  (∃ x y : ℝ, x + 3 * y = 0 ∧ x ≥ 0 ∧ sin (2 * α) = -3 / 5) ↔
  (sin (2 * α) = -3 / 5) :=
by
  sorry

end sufficient_not_necessary_condition_l188_188321


namespace ben_less_than_jack_l188_188054

def jack_amount := 26
def total_amount := 50
def eric_ben_difference := 10

theorem ben_less_than_jack (E B J : ℕ) (h1 : E = B - eric_ben_difference) (h2 : J = jack_amount) (h3 : E + B + J = total_amount) :
  J - B = 9 :=
by sorry

end ben_less_than_jack_l188_188054


namespace max_plus_min_eq_two_l188_188895

def f (x : ℝ) := (2 ^ x + 1) ^ 2 / (2 ^ x * x) + 1

theorem max_plus_min_eq_two :
  let M := Real.Sup (f '' (Set.Icc (-2018) (-epsilon) ∪ Set.Icc (epsilon) 2018))
  let N := Real.Inf (f '' (Set.Icc (-2018) (-epsilon) ∪ Set.Icc (epsilon) 2018))
  ∀ (epsilon : ℝ) (h : epsilon > 0),
  M + N = 2 := 
sorry

end max_plus_min_eq_two_l188_188895


namespace total_price_eq_2500_l188_188320

theorem total_price_eq_2500 (C P : ℕ)
  (hC : C = 2000)
  (hE : C + 500 + P = 6 * P)
  : C + P = 2500 := 
by
  sorry

end total_price_eq_2500_l188_188320


namespace obtuse_triangle_probability_l188_188479

/-- 
Given four points chosen uniformly at random on a circle,
prove that the probability that no three of these points form an obtuse triangle with the circle's center is 3/32.
-/
theorem obtuse_triangle_probability : 
  let points := 4 in
  ∀ (P : Fin points → ℝ × ℝ) (random_uniform : ∀ i, random.uniform_on_circle i),
  (probability (no_three_form_obtuse_triangle P) = 3 / 32) :=
sorry

end obtuse_triangle_probability_l188_188479


namespace geometric_arithmetic_sum_l188_188883

open Real

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def arithmetic_mean (x y a b c : ℝ) : Prop :=
  2 * x = a + b ∧ 2 * y = b + c

theorem geometric_arithmetic_sum
  (a b c x y : ℝ)
  (habc : geometric_sequence a b c)
  (hxy : arithmetic_mean x y a b c)
  (hx_ne_zero : x ≠ 0)
  (hy_ne_zero : y ≠ 0) :
  (a / x) + (c / y) = 2 := 
by {
  sorry -- Proof omitted as per the prompt
}

end geometric_arithmetic_sum_l188_188883


namespace dabbies_turkey_cost_l188_188429

noncomputable def first_turkey_weight : ℕ := 6
noncomputable def second_turkey_weight : ℕ := 9
noncomputable def third_turkey_weight : ℕ := 2 * second_turkey_weight
noncomputable def cost_per_kg : ℕ := 2

noncomputable def total_cost : ℕ :=
  first_turkey_weight * cost_per_kg +
  second_turkey_weight * cost_per_kg +
  third_turkey_weight * cost_per_kg

theorem dabbies_turkey_cost : total_cost = 66 :=
by
  sorry

end dabbies_turkey_cost_l188_188429


namespace partitions_are_equal_l188_188607

def is_partition_diff_parity (seq : List ℕ) (p : List (List ℕ)) : Prop :=
  ∀ (subseq ∈ p), (∀ i j ∈ subseq, (i ≠ j) → ((i % 2) ≠ (j % 2)))

def is_partition_same_parity (seq : List ℕ) (p : List (List ℕ)) : Prop :=
  ∀ (subseq ∈ p), (∀ i j ∈ subseq, (i % 2) = (j % 2))

def A (n : ℕ) : Set (List (List ℕ)) :=
  { p | is_partition_diff_parity (List.range' 1 n) p }

def B (n : ℕ) : Set (List (List ℕ)) :=
  { p | is_partition_same_parity (List.range' 1 n) p }

theorem partitions_are_equal (n : ℕ) (hn : 0 < n) :
  A n.card == B (n + 1).card := 
sorry

end partitions_are_equal_l188_188607


namespace wrapping_paper_cost_l188_188168
noncomputable def cost_per_roll (shirt_boxes XL_boxes: ℕ) (cost_total: ℝ) : ℝ :=
  let rolls_for_shirts := shirt_boxes / 5
  let rolls_for_xls := XL_boxes / 3
  let total_rolls := rolls_for_shirts + rolls_for_xls
  cost_total / total_rolls

theorem wrapping_paper_cost : cost_per_roll 20 12 32 = 4 :=
by
  sorry

end wrapping_paper_cost_l188_188168


namespace hexagon_same_length_probability_l188_188981

/-- 
Given a regular hexagon with 6 sides and 9 diagonals, if two segments are chosen randomly without 
replacement from the set of all sides and diagonals, the probability that the two chosen segments 
have the same length is \(\frac{17}{35}\). 
-/
theorem hexagon_same_length_probability : 
  let S := (finset.range 6).attach ++ (finset.range 9).attach in
  let num_ways_two_sides := (Finset.card (Finset.range 6)).choose 2 in
  let num_ways_two_diags := (Finset.card (Finset.range 9)).choose 2 in
  let total_ways := ((Finset.card S)).choose 2 in
  (num_ways_two_sides + num_ways_two_diags) / total_ways = 17 / 35 :=
by
  sorry

end hexagon_same_length_probability_l188_188981


namespace possible_single_lit_positions_l188_188701

-- Conditions Definitions
def is_on (grid : ℕ × ℕ → bool) (i j : ℕ) : bool := grid (i, j)

def adjacent (i j ni nj : ℕ) : Prop :=
  (ni = i ∧ (nj = j + 1 ∨ nj = j - 1)) ∨
  (nj = j ∧ (ni = i + 1 ∨ ni = i - 1))

def toggle (grid : ℕ × ℕ → bool) (i j : ℕ) : ℕ × ℕ → bool :=
  λ (ni, nj), if adjacent i j ni nj ∨ (ni = i ∧ nj = j)
              then not (grid (ni, nj))
              else grid (ni, nj)

-- Initial condition: all lamps are off
def all_off : ℕ × ℕ → bool := λ _, false

-- The target set of positions where only one lamp is on
def target_positions : Finset (ℕ × ℕ) := 
  {(3, 3), (2, 3), (3, 2), (4, 3), (3, 4)}

-- Proof statement
theorem possible_single_lit_positions :
  ∃ (toggles : list (ℕ × ℕ)), 
    let final_state := toggles.foldl (λ g (i, j), toggle g i j) all_off in
    (∃ (i j : ℕ), final_state (i, j) = true) ∧
    (∀ (ni nj : ℕ), (ni ≠ i ∨ nj ≠ j) → final_state (ni, nj) = false) ∧
    (i, j) ∈ target_positions :=
  sorry

end possible_single_lit_positions_l188_188701


namespace constant_term_condition_l188_188922

theorem constant_term_condition (a : ℝ) :
  (∃ (x : ℝ), (x + 2) * ((1/x - a * x)^7).constant_term = -280) ↔ a = 2 :=
by
  -- -- proof goes here
  sorry

end constant_term_condition_l188_188922


namespace sqrt_ineq_l188_188123

theorem sqrt_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (sqrt x + sqrt y) * (1 / sqrt (1 + x) + 1 / sqrt (1 + y)) > 1 + sqrt 2 / 2 := by
  sorry

end sqrt_ineq_l188_188123


namespace maximum_value_y_l188_188146

noncomputable def max_value_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x^3 + y^3 = (4 * x - 5 * y) * y ∧ (y ≤ 1 / 3)

theorem maximum_value_y : 
  ∀ (x y : ℝ), x > 0 → y > 0 → (x^3 + y^3 = (4 * x - 5 * y) * y) → y ≤ 1 / 3 :=
by
sory

end maximum_value_y_l188_188146


namespace express_in_scientific_notation_l188_188786

def scientific_notation_of_160000 : Prop :=
  160000 = 1.6 * 10^5

theorem express_in_scientific_notation : scientific_notation_of_160000 :=
  sorry

end express_in_scientific_notation_l188_188786


namespace tan_3x_strictly_increasing_l188_188195

theorem tan_3x_strictly_increasing (m : ℝ) :
  (∀ x₁ x₂, m < x₁ ∧ x₁ < x₂ ∧ x₂ < π / 6 → tan (3 * x₁) < tan (3 * x₂)) ↔ m ∈ Icc (-π / 6) (π / 6) :=
sorry

end tan_3x_strictly_increasing_l188_188195


namespace proof_statement_l188_188349

-- Conditions for statements A, B, C, and D
def statement_A_condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x : ℝ, f(x) = 2*x^2 - a*x + 3) ∧ (∀ x : ℝ, f(1-x) = f(1+x))

def statement_B_condition (k : ℝ) : Prop :=
  ∀ x : ℝ, k*x^2 - 6*k*x + k + 8 ≥ 0

def statement_C_condition (a : ℝ) : Prop :=
  let M := {1, 2}
  let N := {a^2}
  (∃ a : ℝ, a = 1 → N ⊆ M)

def statement_D_condition (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, - (π / 4) < x ∧ x ≤ 3 * (π / 4) → f(x) = Real.sin (2 * x)) ∧ 
  (∀ y : ℝ, ∃ x : ℝ, - (π / 4) < x ∧ x ≤ 3 * (π / 4) ∧ f(x) = y)

-- Statements to be proved
theorem proof_statement :
  (∀ f a, statement_A_condition f a → a = 4) ∧
  (¬(∀ k, statement_B_condition k → 0 < k ∧ k ≤ 1)) ∧
  (statement_C_condition 1) ∧
  (exists f, statement_D_condition f) :=
by
  constructor;
  try { intro, sorry };
  try { split; intro, sorry; sorry }

end proof_statement_l188_188349


namespace investment_amount_l188_188624

def compoundInterest (P: ℝ) (r: ℝ) (n: ℕ) (t: ℕ): ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem investment_amount:
  let A := 100000 in
  let r := 0.08 in
  let n := 12 in
  let t := 10 in
  let P := 45046 in
  compoundInterest P r n t = A :=
by
  sorry

end investment_amount_l188_188624


namespace heat_of_neutralization_combination_l188_188218

-- Define instruments
inductive Instrument
| Balance
| MeasuringCylinder
| Beaker
| Burette
| Thermometer
| TestTube
| AlcoholLamp

def correct_combination : List Instrument :=
  [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer]

theorem heat_of_neutralization_combination :
  correct_combination = [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer] :=
sorry

end heat_of_neutralization_combination_l188_188218


namespace work_completion_days_l188_188373

theorem work_completion_days
    (A : ℝ) (B : ℝ) (h1 : 1 / A + 1 / B = 1 / 10)
    (h2 : B = 35) :
    A = 14 :=
by
  sorry

end work_completion_days_l188_188373


namespace find_AX_l188_188628

variable (A B C D F H G X : Point)
variable (AB BC BF BH AC GD AX : ℝ)

-- Define the properties and conditions given in the problem
variable [IsParallelogram A B C D]
variable [OnSegment A B F]
variable [OnExtension B C H]
variable [FormParallelogram B F G H]

-- Given ratios and given length AC
variable (AB_ratio : AB / BF = 5)
variable (BC_ratio : BC / BH = 5)
variable (AC_len : AC = 100)

-- We need to prove the result
theorem find_AX : AX = 40 :=
by
  sorry

end find_AX_l188_188628


namespace maryHasNineMarbles_l188_188265

-- Defining the conditions
def joanMarbles : Nat := 3
def totalMarbles : Nat := 12

-- Define what we are proving
def maryMarbles : Nat := totalMarbles - joanMarbles

theorem maryHasNineMarbles : maryMarbles = 9 := by
  -- It suffices to compute the subtraction manually
  calc
    maryMarbles = totalMarbles - joanMarbles := rfl
    ... = 12 - 3 := by rw [totalMarbles, joanMarbles]
    ... = 9 := by norm_num

-- Placeholder to conclude the proof
sorry

end maryHasNineMarbles_l188_188265


namespace lindsey_squat_weight_l188_188256

-- Define the conditions
def num_bands : ℕ := 2
def resistance_per_band : ℤ := 5
def dumbbell_weight : ℤ := 10

-- Define the weight Lindsay will squat
def total_weight : ℤ := num_bands * resistance_per_band + dumbbell_weight

-- State the theorem
theorem lindsey_squat_weight : total_weight = 20 :=
by
  sorry

end lindsey_squat_weight_l188_188256


namespace sample_size_is_correct_l188_188760

variable (company_employees : ℕ)
variable (young_employees : ℕ)
variable (middle_aged_employees : ℕ)
variable (elderly_employees : ℕ)
variable (sampled_young_employees : ℕ)
variable (sample_size : ℕ)

-- Given conditions
def conditions : Prop := 
  company_employees = 750 ∧
  young_employees = 350 ∧
  middle_aged_employees = 250 ∧
  elderly_employees = 150 ∧
  sampled_young_employees = 7

-- Question to prove
theorem sample_size_is_correct (h : conditions) : sample_size = 15 := 
  sorry

end sample_size_is_correct_l188_188760


namespace number_of_k_l188_188846

theorem number_of_k (n : ℕ) (hn : n = 2016) :
  (Finset.card { k : ℕ | k ∈ Finset.range (n + 1) ∧ (k : ZMod (n + 1 + 1) ^ n = 1) }) = Nat.totient n :=
by
  sorry

end number_of_k_l188_188846


namespace max_viewing_area_l188_188702

theorem max_viewing_area (L W: ℝ) (h1: 2 * L + 2 * W = 420) (h2: L ≥ 100) (h3: W ≥ 60) : 
  (L = 105) ∧ (W = 105) ∧ (L * W = 11025) :=
by
  sorry

end max_viewing_area_l188_188702


namespace median_length_l188_188547

def point := ℝ × ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  (real.sqrt ((fst p2 - fst p1) ^ 2 + (snd p2 - snd p1) ^ 2 + ((p2.2 - p1.2) ^ 2)))

theorem median_length (A B C : point) (A_x A_y A_z B_x B_y B_z C_x C_y C_z : ℝ) 
  (hA : A = (A_x, A_y, A_z)) 
  (hB : B = (B_x, B_y, B_z)) 
  (hC : C = (C_x, C_y, C_z)) 
  (hA_coords: A = (2, -1, 4)) 
  (hB_coords: B = (3, 2, -6)) 
  (hC_coords: C = (5, 0, 2)) : 
  distance A ((B_x + C_x) / 2, (B_y + C_y) / 2, (B_z + C_z) / 2) = 2 * real.sqrt 11 := 
by 
  sorry

end median_length_l188_188547


namespace probability_x_lt_2y_in_rectangle_l188_188776

theorem probability_x_lt_2y_in_rectangle :
  let rectangle := set.Icc (0, 0) (4, 2)
  let region := {p : ℝ × ℝ | p ∈ rectangle ∧ p.1 < 2 * p.2}
  (measure_theory.measure.region.probability_of_region region rectangle = 1 / 2) :=
by
  sorry

end probability_x_lt_2y_in_rectangle_l188_188776


namespace monotonic_increasing_interval_l188_188307

-- Define the function f(x) = (2x - 3) * exp(x)
def f (x : ℝ) : ℝ := (2 * x - 3) * Real.exp x

-- Theorem statement: f is monotonic increasing on (1/2, +∞ )
theorem monotonic_increasing_interval : ∀ x ∈ Set.Ioi (1/2), 0 < (2 * x - 1) * Real.exp x :=
by
  sorry

end monotonic_increasing_interval_l188_188307


namespace total_distance_correct_l188_188018

-- Define the conditions given in the problem
def total_time := 15 -- total time in hours
def speed_first_half := 21 -- speed for the first half in km/hr
def speed_second_half := 24 -- speed for the second half in km/hr

-- Define the total distance D
def total_distance (D : ℝ) : Prop :=
  let time_first_half := (D / 2) / speed_first_half in
  let time_second_half := (D / 2) / speed_second_half in
  time_first_half + time_second_half = total_time

-- The proof statement we need to prove
theorem total_distance_correct : total_distance 336 :=
by
  sorry

end total_distance_correct_l188_188018


namespace tariffs_impact_but_no_timeframe_l188_188751

noncomputable def cost_of_wine_today : ℝ := 20.00
noncomputable def increase_percentage : ℝ := 0.25
noncomputable def bottles_count : ℕ := 5
noncomputable def price_increase_for_bottles : ℝ := 25.00

theorem tariffs_impact_but_no_timeframe :
  ¬ ∃ (t : ℝ), (cost_of_wine_today * (1 + increase_percentage) - cost_of_wine_today) * bottles_count = price_increase_for_bottles →
  (t = sorry) :=
by 
  sorry

end tariffs_impact_but_no_timeframe_l188_188751


namespace Danny_more_wrappers_than_bottle_caps_l188_188046

theorem Danny_more_wrappers_than_bottle_caps
  (initial_wrappers : ℕ)
  (initial_bottle_caps : ℕ)
  (found_wrappers : ℕ)
  (found_bottle_caps : ℕ) :
  initial_wrappers = 67 →
  initial_bottle_caps = 35 →
  found_wrappers = 18 →
  found_bottle_caps = 15 →
  (initial_wrappers + found_wrappers) - (initial_bottle_caps + found_bottle_caps) = 35 :=
by
  intros h1 h2 h3 h4
  sorry

end Danny_more_wrappers_than_bottle_caps_l188_188046


namespace man_work_alone_in_5_days_l188_188005

theorem man_work_alone_in_5_days (d : ℕ) (h1 : ∀ m : ℕ, (1 / (m : ℝ)) + 1 / 20 = 1 / 4):
  d = 5 := by
  sorry

end man_work_alone_in_5_days_l188_188005


namespace subtract_two_percent_is_multiplying_l188_188357

theorem subtract_two_percent_is_multiplying (a : ℝ) : (a - 0.02 * a) = 0.98 * a := by
  sorry

end subtract_two_percent_is_multiplying_l188_188357


namespace janet_family_needs_91_tickets_l188_188968

def janet_family_tickets (adults: ℕ) (children: ℕ) (roller_coaster_adult_tickets: ℕ) (roller_coaster_child_tickets: ℕ) 
  (giant_slide_adult_tickets: ℕ) (giant_slide_child_tickets: ℕ) (num_roller_coaster_rides_adult: ℕ) 
  (num_roller_coaster_rides_child: ℕ) (num_giant_slide_rides_adult: ℕ) (num_giant_slide_rides_child: ℕ) : ℕ := 
  (adults * roller_coaster_adult_tickets * num_roller_coaster_rides_adult) + 
  (children * roller_coaster_child_tickets * num_roller_coaster_rides_child) + 
  (1 * giant_slide_adult_tickets * num_giant_slide_rides_adult) + 
  (1 * giant_slide_child_tickets * num_giant_slide_rides_child)

theorem janet_family_needs_91_tickets :
  janet_family_tickets 2 2 7 5 4 3 3 2 5 3 = 91 := 
by 
  -- Calculations based on the given conditions (skipped in this statement)
  sorry

end janet_family_needs_91_tickets_l188_188968


namespace speed_of_faster_train_l188_188337

noncomputable def speed_of_slower_train_kmph := 36
def time_to_cross_seconds := 12
def length_of_faster_train_meters := 120

-- Speed of train V_f in kmph 
theorem speed_of_faster_train 
  (relative_speed_mps : ℝ := length_of_faster_train_meters / time_to_cross_seconds)
  (speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * (1000 / 3600))
  (speed_of_faster_train_mps : ℝ := relative_speed_mps + speed_of_slower_train_mps)
  (speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * (3600 / 1000) )
  : speed_of_faster_train_kmph = 72 := 
sorry

end speed_of_faster_train_l188_188337


namespace eval_piecewise_function_l188_188153

noncomputable def f : ℝ → ℝ :=
λ x, if h : -1 ≤ x ∧ x ≤ 0 then - (real.sqrt 2) * real.sin x
     else if h : 0 < x ∧ x ≤ 1 then real.tan (real.pi / 4 * x)
     else 0

theorem eval_piecewise_function :
  f (f (- real.pi / 4)) = 1 :=
by {
  sorry
}

end eval_piecewise_function_l188_188153


namespace max_of_five_numbers_l188_188347

theorem max_of_five_numbers : 
let a := 0.9891 
let b := 0.9799 
let c := 0.9890 
let d := 0.978 
let e := 0.979 
in max (max (max (max a b) c) d) e = a := 
by 
  let a := 0.9891
  let b := 0.9799
  let c := 0.9890
  let d := 0.978
  let e := 0.979
  sorry

end max_of_five_numbers_l188_188347


namespace obtuse_triangle_probability_l188_188470

noncomputable def uniform_circle_distribution (radius : ℝ) : ProbabilitySpace (ℝ × ℝ) := sorry

def no_obtuse_triangle (points : list (ℝ × ℝ)) (center : (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k < 4 →
  ∀ (p₁ p₂ : (ℝ × ℝ)),
  (p₁ = points.nth i ∧ p₂ = points.nth j ∧ points.nth k ∉ set_obtuse_triangles p₁ p₂ center)

theorem obtuse_triangle_probability :
  let points := [⟨cos θ₀, sin θ₀⟩, ⟨cos θ₁, sin θ₁⟩, ⟨cos θ₂, sin θ₂⟩, ⟨cos θ₃, sin θ₃⟩]
  ∀ center : (ℝ × ℝ), 
  ∀ (h : set.center.circle center radius), 
  (no_obtuse_triangle points center) →
  Pr[(points)] = 3 / 32 :=
sorry

end obtuse_triangle_probability_l188_188470


namespace curve_is_circle_l188_188836

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (2 - Real.sin θ)) : 
    ∃ (a b R : ℝ), ∀ x y : ℝ, x^2 + (y - (2 * sqrt (x^2 + y^2) - 1))^2 = (sqrt (x^2 + y^2))^2 → 
    (x - a)^2 + (y - b)^2 = R^2 :=
by sorry

end curve_is_circle_l188_188836


namespace solution_k_system_eq_l188_188199

theorem solution_k_system_eq (x y k : ℝ) 
  (h1 : x + y = 5 * k) 
  (h2 : x - y = k) 
  (h3 : 2 * x + 3 * y = 24) : k = 2 :=
by
  sorry

end solution_k_system_eq_l188_188199


namespace smallest_a_satisfies_sin_condition_l188_188997

open Real

theorem smallest_a_satisfies_sin_condition :
  ∃ (a : ℝ), (∀ x : ℤ, sin (a * x + 0) = sin (45 * x)) ∧ 0 ≤ a ∧ ∀ b : ℝ, (∀ x : ℤ, sin (b * x + 0) = sin (45 * x)) ∧ 0 ≤ b → 45 ≤ b :=
by
  -- To be proved.
  sorry

end smallest_a_satisfies_sin_condition_l188_188997


namespace highest_score_l188_188358

variables (H L : ℕ) (average_46 : ℕ := 61) (innings_46 : ℕ := 46) 
                (difference : ℕ := 150) (average_44 : ℕ := 58) (innings_44 : ℕ := 44)

theorem highest_score:
  (H - L = difference) →
  (average_46 * innings_46 = average_44 * innings_44 + H + L) →
  H = 202 :=
by
  intros h_diff total_runs_eq
  sorry

end highest_score_l188_188358


namespace nested_sqrt_solution_l188_188076

noncomputable def nested_sqrt (x : ℝ) : ℝ := sqrt (20 + x)

theorem nested_sqrt_solution (x : ℝ) : nonneg_real x →
  (x = nested_sqrt x ↔ x = 5) :=
begin
  sorry
end

end nested_sqrt_solution_l188_188076


namespace sum_f_values_l188_188890

def f (x : ℝ) : ℝ := 1 - Real.sin ((π / 3) * x + (π / 6))

theorem sum_f_values : 
  (∑ i in Finset.range 2023, f (i + 1)) = 2022 := 
by 
  sorry

end sum_f_values_l188_188890


namespace length_of_each_song_l188_188261

-- Define the conditions
def jump_per_second : ℕ := 1
def total_jumps : ℕ := 2100
def number_of_songs : ℕ := 10

-- Define the conversion factor from seconds to minutes
def seconds_per_minute : ℕ := 60

-- Define the proof problem
theorem length_of_each_song : 
  (total_jumps / seconds_per_minute) / number_of_songs = 3.5 :=
by
  sorry

end length_of_each_song_l188_188261


namespace integral_limit_l188_188366

noncomputable def continuous_function_limit (f : ℝ → ℝ) (L : ℝ) : Prop :=
  continuous_on f (set.Ici 0) ∧ filter.tendsto f filter.at_top (nhds L)

theorem integral_limit (f : ℝ → ℝ) (L : ℝ) 
  (h1 : continuous_on f (set.Ici 0))
  (h2 : filter.tendsto f filter.at_top (nhds L)) :
  filter.tendsto (λ n : ℕ, ∫ x in 0..1, f(n * x)) filter.at_top (nhds L) :=
sorry

end integral_limit_l188_188366


namespace james_age_when_tim_is_79_l188_188963

variable {James_age John_age Tim_age : ℕ}

theorem james_age_when_tim_is_79 (J_age J_age_at_23 J_age_diff J_age_j : ℕ) 
                                  (H1 : J_age = J_age_at_23 - J_age_diff)
                                  (H2 : John_age = 35)
                                  (H3 : James_age = 23)
                                  (age_diff:12: ℕ )
                                  (H4 : Tim_age = 2 * John_age - 5)
                                  (H5 : Tim_age = 79):
                                  J_age=30 :=
by
  sorry

end james_age_when_tim_is_79_l188_188963


namespace minimum_n_exists_l188_188934

theorem minimum_n_exists :
  ∀ (n : ℕ), (n ≥ 4) →
  (∀ p : ℕ, (p ∈ list.range n) →
  (∃ s : set ℕ, s.card = 4 ∧ ∀ p₁ p₂ : ℕ, (p₁ ≠ p₂ ∧ p₁ ∈ list.range n ∧ p₂ ∈ list.range n) → (∃! x, x ∈ s))) →
  (∃ N : ℕ, N ≥ 4 * n → N) →
  ∃ n, n = 14 :=
by
  sorry

end minimum_n_exists_l188_188934


namespace heidi_more_nail_polishes_l188_188604

theorem heidi_more_nail_polishes :
  ∀ (k h r : ℕ), 
    k = 12 ->
    r = k - 4 ->
    h + r = 25 ->
    h - k = 5 :=
by
  intros k h r hk hr hr_sum
  sorry

end heidi_more_nail_polishes_l188_188604


namespace lindsey_squat_weight_l188_188257

theorem lindsey_squat_weight :
  let num_bands := 2 in
  let resistance_per_band := 5 in
  let total_resistance := num_bands * resistance_per_band in
  let dumbbell_weight := 10 in
  total_resistance + dumbbell_weight = 20 :=
by
  sorry

end lindsey_squat_weight_l188_188257


namespace nested_sqrt_eq_five_l188_188063

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l188_188063


namespace cartesian_equation_of_curve_C_length_AB_l188_188943

-- Conditions
def polar_equation (ρ θ : ℝ) : Prop := ρ * real.sin θ ^ 2 = real.cos θ
def param_eq_line_L (t : ℝ) : ℝ × ℝ := (2 - (real.sqrt 2 / 2) * t, (real.sqrt 2 / 2) * t)

-- Questions and correct answers as Lean statements
theorem cartesian_equation_of_curve_C :
  ∀ (ρ θ x y : ℝ), polar_equation ρ θ →
  ρ = real.sqrt (x^2 + y^2) →
  θ = real.arctan (y / x) →
  y^2 = x :=
sorry

theorem length_AB :
  ∀ (t t₁ t₂ : ℝ), 
  (x y : ℝ) = param_eq_line_L t →
  y^2 = x →
  t₁ + t₂ = -real.sqrt 2 →
  t₁ * t₂ = -4 →
  |t₁ - t₂| = 3 * real.sqrt 2 :=
sorry

end cartesian_equation_of_curve_C_length_AB_l188_188943


namespace sin_alpha_plus_7pi_over_12_cos_2alpha_plus_pi_over_6_l188_188877

/-- 
Given α is an acute angle and satisfies cos(α + π/4) = √3 / 3,
we have the following proofs to establish:
1) sin(α + 7π/12) = (√6 + 3) / 6
2) cos(2α + π/6) = (2√6 - 1) / 6
-/

variable {α : ℝ} (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos (α + π / 4) = Real.sqrt 3 / 3)

theorem sin_alpha_plus_7pi_over_12 :
  Real.sin (α + 7 * π / 12) = (Real.sqrt 6 + 3) / 6 :=
sorry

theorem cos_2alpha_plus_pi_over_6 :
  Real.cos (2 * α + π / 6) = (2 * Real.sqrt 6 - 1) / 6 :=
sorry

end sin_alpha_plus_7pi_over_12_cos_2alpha_plus_pi_over_6_l188_188877


namespace compute_operation_l188_188684

-- Define the operation ⊕
def op (a b : ℝ) : ℝ := (a - b) / (2 - a * b)

-- Define y as the nested operation for 2 ⊕ (3 ⊕ (... ⊕ (999 ⊕ 1000) ...))
noncomputable def y : ℝ := List.foldr op 1000 (List.range 999).map (λ i => i + 2)

theorem compute_operation : 
  op 1 y = (1 - y) / (2 - y) :=
by sorry

end compute_operation_l188_188684


namespace consecutive_neg_int_abs_diff_l188_188686

theorem consecutive_neg_int_abs_diff (n : ℤ) (h1 : n * (n + 1) = 2240) (h2 : n < 0) (h3 : n + 1 < 0) : 
  |n - (n + 1)| = 1 := 
by 
  -- Conditions
  have h₄ : n * (n + 1) = -2240, from sorry,
  -- Conclusion
  sorry

end consecutive_neg_int_abs_diff_l188_188686


namespace years_until_5_years_before_anniversary_l188_188186

-- Definitions
def years_built_ago := 100
def upcoming_anniversary := 200
def years_before_anniversary := 5

-- Theorem statement
theorem years_until_5_years_before_anniversary :
  let years_until_anniversary := upcoming_anniversary - years_built_ago in
  let future_years := years_until_anniversary - years_before_anniversary in
  future_years = 95 :=
by
  sorry

end years_until_5_years_before_anniversary_l188_188186


namespace D_72_eq_48_l188_188985

def is_valid_factorization : ℕ → (List ℕ) → Prop
| n, factors => factors.prod = n ∧ ∀ f ∈ factors, f > 1

def D (n : ℕ) : ℕ :=
  List.length ([factors | factors ← List.sublists (List.range (n + 1)), is_valid_factorization n factors])

theorem D_72_eq_48 : D 72 = 48 :=
by
  sorry

end D_72_eq_48_l188_188985


namespace product_conjugate_l188_188246

variable (z : ℂ)

theorem product_conjugate (h : Complex.abs z = 7) : z * Complex.conj z = 49 := by
  sorry

end product_conjugate_l188_188246


namespace order_of_middle_four_cells_l188_188205

def Letter := Fin 6
def Pos := Fin 6

-- Conditions
structure Grid (L : Type) (P : Type) :=
  (cell : P → P → L)
  (unique_rows : ∀ r : P, ∀ l : L, ∃! c : P, cell r c = l)
  (unique_cols : ∀ c : P, ∀ l : L, ∃! r : P, cell r c = l)
  (unique_blocks : ∀ br bc : Fin 2, ∀ l : L, ∃! p : (Fin 2) × (Fin 3),
                    cell (br * 2 + p.1) (bc * 3 + p.2) = l)

-- Define the specific grid of letters A to F
def cell_grid_example : Grid (Fin 6) (Fin 6) :=
sorry  -- Complete with the specific grid satisfying the conditions

theorem order_of_middle_four_cells (G : Grid (Fin 6) (Fin 6)) :
  let r := 2  -- the third row (index is 0-based)
      left_to_right_letters := List.ofFn (λ i, G.cell r (i + 1))
  in left_to_right_letters = [1, 3, 4, 5] :=
sorry

end order_of_middle_four_cells_l188_188205


namespace solve_for_a_l188_188878

noncomputable def line_tangent_to_circle (a : ℝ) : Prop :=
  ∀ θ : ℝ, let x := 2 + 2 * Real.cos θ in
           let y := 1 + 2 * Real.sin θ in
           (a * x - y + 2 = 0) ∧ (x - 2)^2 + (y - 1)^2 = 4 → False

theorem solve_for_a (a : ℝ) : line_tangent_to_circle a ↔ a = 3 / 4 := by
  sorry

end solve_for_a_l188_188878


namespace distance_between_points_l188_188839

theorem distance_between_points : real.sqrt ((13 - 2)^2 + (4 + 3)^2) = real.sqrt 170 :=
by
  sorry

end distance_between_points_l188_188839


namespace find_x_if_opposites_l188_188557

theorem find_x_if_opposites :
  (∀ x : ℝ, 2 * x^2 + 1 = -(4 * x^2 - 2 * x - 5)) → (x = 1 ∨ x = -2/3) :=
by
  intro h
  have h_eq : 2 * x^2 + 1 + (4 * x^2 - 2 * x - 5) = 0 :=
    by simp [h x]
  have h_simplified : 6 * x^2 - 2 * x - 4 = 0 :=
    by linarith
  have h_reduced : 3 * x^2 - x - 2 = 0 :=
    by linarith
  -- Factor the quadratic equation
  rw [←eq_iff_eq_cancel_left] at h_reduced
  have h_factored : (x - 1) * (3 * x + 2) = 0 :=
    by (apply eq_zero_or_eq_zero_of_mul_eq_zero).mp
  cases h_factored
  case inl h1
    exact h1.symm ▸ Or.inl rfl
  case inr h2
    exact h2.symm ▸ Or.inr rfl
  done

end find_x_if_opposites_l188_188557


namespace polynomial_sum_of_squares_l188_188280

theorem polynomial_sum_of_squares (P : Polynomial ℝ) 
  (hP : ∀ x : ℝ, 0 ≤ P.eval x) : 
  ∃ (f g : Polynomial ℝ), P = f * f + g * g := 
sorry

end polynomial_sum_of_squares_l188_188280


namespace curve_C1_polar_equation_segment_AB_length_l188_188215

-- Definitions based on the conditions
structure ParametricCurve (a t : ℝ) where
  x : ℝ := a * Real.cos t + Real.sqrt 3
  y : ℝ := a * Real.sin t

def polar_equation (ρ θ a : ℝ) : Prop :=
  ρ^2 - 2 * Real.sqrt 3 * ρ * Real.cos θ + 3 = a^2

def intersection_in_polar_coords := ∃ (ρ1 ρ2 θ : ℝ), 
  (ρ1^2 = 2 * ρ1 * Real.sin θ + 6 ∧
   ρ2^2 = 2 * ρ2 * Real.sin θ + 6 ∧
   ρ1 ≠ ρ2)

-- Proving the polar equation
theorem curve_C1_polar_equation (a t ρ θ : ℝ) (h : a > 0)
  (h_ρ_eq : ρ^2 = (a * Real.cos t + Real.sqrt 3)^2 + (a * Real.sin t)^2)
  (h_x : (ParametricCurve a t).x = ρ * Real.cos θ)
  (h_y : (ParametricCurve a t).y = ρ * Real.sin θ) :
  polar_equation ρ θ a :=
sorry

-- Proving the length of segment AB
theorem segment_AB_length (a : ℝ) (h : a > 0)
  (h_intersect : ∀ θ : ℝ, intersection_in_polar_coords) :
  ∃ AB : ℝ, AB = 3 * Real.sqrt 3 :=
sorry

end curve_C1_polar_equation_segment_AB_length_l188_188215


namespace fraction_of_students_received_B_l188_188577

theorem fraction_of_students_received_B {total_students : ℝ}
  (fraction_A : ℝ)
  (fraction_A_or_B : ℝ)
  (h_fraction_A : fraction_A = 0.7)
  (h_fraction_A_or_B : fraction_A_or_B = 0.9) :
  fraction_A_or_B - fraction_A = 0.2 :=
by
  rw [h_fraction_A, h_fraction_A_or_B]
  sorry

end fraction_of_students_received_B_l188_188577


namespace unpainted_unit_cubes_in_4x4x4_cube_l188_188749

theorem unpainted_unit_cubes_in_4x4x4_cube :
  let painted_count : ℕ := 6 * 6,
      doubly_counted : ℕ := 12 * 2,
      painted_unit_cubes : ℕ := painted_count - (doubly_counted / 2)
  in
  64 - painted_unit_cubes = 40 :=
by
  -- proof goes here
  sorry

end unpainted_unit_cubes_in_4x4x4_cube_l188_188749


namespace shaded_area_l188_188583

open Real

-- Define the areas of the rectangles
def area_rect1 : ℝ := 3 * 2
def area_rect2 : ℝ := 4 * 6
def area_rect3 : ℝ := 5 * 3

-- Calculate the total area of the grid
def total_area_grid : ℝ := area_rect1 + area_rect2 + area_rect3

-- Define the area of the unshaded triangle
def area_triangle : ℝ := (1 / 2) * 14 * 6

-- Prove the area of the shaded region
theorem shaded_area : total_area_grid - area_triangle = 3 :=
by 
  -- substitute the calculated values
  unfold total_area_grid area_triangle
  norm_num
  sorry

end shaded_area_l188_188583


namespace tangent_line_eq_l188_188299

noncomputable def curve (x : ℝ) := x / (x + 1)
noncomputable def point_of_tangency := (-2 : ℝ)
noncomputable def slope_of_tangent_at (x : ℝ) := deriv curve x
noncomputable def tangent_point := (point_of_tangency, curve point_of_tangency)

theorem tangent_line_eq :
  let k := slope_of_tangent_at point_of_tangency,
      p := tangent_point in
  ∃ m b, (λ y x, y = k * x + b) p.2 p.1 ∧ x - y + 4 = 0 :=
by
  sorry

end tangent_line_eq_l188_188299


namespace joan_original_seashells_l188_188228

theorem joan_original_seashells (a b total: ℕ) (h1 : a = 63) (h2 : b = 16) (h3: total = a + b) : total = 79 :=
by
  rw [h1, h2] at h3
  exact h3

end joan_original_seashells_l188_188228


namespace equal_number_of_boys_and_girls_l188_188316

theorem equal_number_of_boys_and_girls
    (num_classrooms : ℕ) (girls : ℕ) (total_per_classroom : ℕ)
    (equal_boys_and_girls : ∀ (c : ℕ), c ≤ num_classrooms → (girls + boys) = total_per_classroom):
    num_classrooms = 4 → girls = 44 → total_per_classroom = 25 → boys = 44 :=
by
  sorry

end equal_number_of_boys_and_girls_l188_188316


namespace correct_option_C_l188_188901

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 5}
def P : Set ℕ := {2, 4}

theorem correct_option_C : 3 ∈ U \ (M ∪ P) :=
by
  sorry

end correct_option_C_l188_188901


namespace expansion_abs_coeff_sum_l188_188117

theorem expansion_abs_coeff_sum :
  ∀ (a a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - x)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 32 :=
by
  sorry

end expansion_abs_coeff_sum_l188_188117


namespace find_investment_C_l188_188397

noncomputable def investment_C (C : ℝ) : Prop :=
  C = 9600

lemma determine_C (C : ℝ) :
  (2400 + 7200 + C) * (1125 / 9000) = 2400 :=
  sorry

theorem find_investment_C (C : ℝ) : investment_C C :=
begin
  have h : (2400 + 7200 + C) * (1 / 8) = 2400,
  { exact determine_C C },
  have h_eq : 2400 * 8 = 2400 + 7200 + C,
  { simp [h], norm_num },
  linarith,
end

end find_investment_C_l188_188397


namespace polynomial_never_33_l188_188278

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by
  sorry

end polynomial_never_33_l188_188278


namespace verify_correct_propositions_l188_188848

variables (m n : Type) (α β γ : set m) [inner_product_space ℝ m]

-- Conditions for each proposition
def proposition_1 (m_parallel_alpha m_perp_n n_perp_alpha : Prop) : Prop :=
(m_parallel_alpha ∧ m_perp_n) → n_perp_alpha

def proposition_2 (m_perp_alpha m_perp_n n_parallel_alpha : Prop) : Prop :=
(m_perp_alpha ∧ m_perp_n) → n_parallel_alpha

def proposition_3 (alpha_perp_beta gamma_perp_beta alpha_parallel_gamma : Prop) : Prop :=
(alpha_perp_beta ∧ gamma_perp_beta) → alpha_parallel_gamma

def proposition_4 (m_perp_alpha m_parallel_n n_subset_beta alpha_perp_beta : Prop) : Prop :=
(m_perp_alpha ∧ m_parallel_n ∧ n_subset_beta) → alpha_perp_beta

-- Proposition correctness 
def correct_propositions : Prop :=
¬(proposition_1 m_parallel_alpha m_perp_n n_perp_alpha) ∧
¬(proposition_2 m_perp_alpha m_perp_n n_parallel_alpha) ∧
¬(proposition_3 alpha_perp_beta gamma_perp_beta alpha_parallel_gamma) ∧
(proposition_4 m_perp_alpha m_parallel_n n_subset_beta alpha_perp_beta)

theorem verify_correct_propositions : correct_propositions :=
begin
  sorry
end

end verify_correct_propositions_l188_188848


namespace find_k_l188_188437

noncomputable def polynomial_division_constant_remainder (k : ℝ) : Prop :=
  (∀ (x : ℝ), ((3 * x^2 - 5 * x + 2) ≠ 0) → 
    ∃ r : ℝ, (12 * x^3 - k * x^2 - 8 * x - 5) / (3 * x^2 - 5 * x + 2) = r
  ) → k = 62

theorem find_k : polynomial_division_constant_remainder 62 :=
begin
  sorry
end

end find_k_l188_188437


namespace geometric_sequence_problem_l188_188592

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6) 
  (h2 : a 3 + a 5 + a 7 = 78) :
  a 5 = 18 :=
sorry

end geometric_sequence_problem_l188_188592


namespace find_k_l188_188783

noncomputable def original_height : ℝ := 6
noncomputable def original_radius : ℝ := 5
noncomputable def smaller_cone_height : ℝ := 2
noncomputable def k : ℝ := 1 / 24

def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * (r ^ 2) * h
def cone_surface_area (r l : ℝ) : ℝ := π * r ^ 2 + π * r * l
def slant_height (r h : ℝ) : ℝ := Real.sqrt (r ^ 2 + h ^ 2)

theorem find_k :
  let original_l := slant_height original_radius original_height
      original_volume := cone_volume original_radius original_height
      original_surface_area := cone_surface_area original_radius original_l
      ratio := smaller_cone_height / original_height
      smaller_radius := ratio * original_radius
      smaller_volume := cone_volume smaller_radius smaller_cone_height
      smaller_l := slant_height smaller_radius smaller_cone_height
      smaller_surface_area := π * smaller_radius * smaller_l
      frustum_volume := original_volume - smaller_volume
      frustum_surface_area := original_surface_area - smaller_surface_area
  in smaller_volume / frustum_volume = k ∧ smaller_surface_area / frustum_surface_area = k :=
by
  sorry

end find_k_l188_188783


namespace range_of_a_l188_188534

noncomputable def f (x a : ℝ) : ℝ := exp x * (x^2 - a)

theorem range_of_a (a : ℝ) : (∀ x ∈ Ioo (-3 : ℝ) 0, deriv (λ x, f x a) x ≤ 0) ↔ 3 ≤ a := 
by
  sorry

end range_of_a_l188_188534


namespace probability_ones_digit_zero_total_outcomes_probability_of_event_ones_digit_zero_probability_l188_188336

def is_ones_digit_zero (n : ℕ) : Prop :=
  n % 10 = 0

def product (d1 d2 : ℕ) : ℕ :=
  d1 * d2

theorem probability_ones_digit_zero :
  ∑ (d1 d2 : ℕ) in ({1, 2, 3, 4, 5, 6} : Finset ℕ), if is_ones_digit_zero (product d1 d2) then 1 else 0 = 6 :=
sorry

theorem total_outcomes :
  (6 * 6) = 36 :=
rfl

theorem probability_of_event :
  (6 : ℚ) / 36 = 1 / 6 :=
by norm_num

theorem ones_digit_zero_probability : (6 / 36 : ℚ) = 1 / 6 :=
by apply probability_of_event

end probability_ones_digit_zero_total_outcomes_probability_of_event_ones_digit_zero_probability_l188_188336


namespace revenue_expression_l188_188376

noncomputable def total_revenue (x : ℕ) : ℕ :=
  1800 * x + 1600 * (30 - x) + 1600 * (20 - x) + 1200 * x

theorem revenue_expression (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 20) :
  total_revenue x = -200 * x + 80000 :=
by {
  sorry
}

end revenue_expression_l188_188376


namespace arrangement_of_competitors_l188_188581

/-- Represents a group of competitors where the largest size of any clique is even,
    and it is provable that they can be arranged into two rooms such that the size
    of the largest clique in one room is equal to the size of the largest clique
    in the other room. -/
theorem arrangement_of_competitors (G : Type) [Fintype G]
  (friend : G → G → Prop) [DecidableRel friend]
  (even_largest_clique_size : ∃ m : ℕ, ∃ M : Finset G, ∀ g g' ∈ M, friend g g' ∧ 2 * m = M.card) :
  ∃ (A B : Finset G), (Finset.card A + Finset.card B = Finset.card (Fintype.elems G)) ∧ 
  ((∀ (C : Finset G), (∀ g g' ∈ C, friend g g' → (Finset.card C ≤ Finset.card A)) ∧
    (∀ C : Finset G, (∀ g g' ∈ C, friend g g' → (Finset.card C ≤ Finset.card B))))) :=
sorry

end arrangement_of_competitors_l188_188581


namespace tic_tac_toe_azar_wins_with_fourth_X_l188_188409

-- Define the main theorem
theorem tic_tac_toe_azar_wins_with_fourth_X :
  ∃ (n : ℕ), n = 40 ∧
  (∃ (board : fin 9 → option (fin 2)), 
     -- Conditions: 
     -- Each player makes their move at random
     (∀ i, board i ≠ none) ∧ -- All cells are filled
     (∃ (azar_wins : Prop), fst (count board) = 4 ∧ -- Azar wins with fourth 'X'
      azar_wins) ∧
     (∃ (azar_starts : Prop), snd (count board) = 4 ∧ -- Carl places 'O' alternatively
      azar_starts))
 := by
  -- The proof is omitted
  sorry

end tic_tac_toe_azar_wins_with_fourth_X_l188_188409


namespace calc_variance_of_data_set_l188_188920

theorem calc_variance_of_data_set :
  (avg : ℚ) (h_avg : avg = (4 + 5 + 7 + 9 + a) / 5) -> 
  (variance : ℚ) (h_variance : variance = (1 / 5) * ((4-6)^2 + (5-6)^2 + (7-6)^2 + (9-6)^2 + (a-6)^2)) ->
  a = 5 -> 
  variance = 16 / 5 :=
by
  intro avg h_avg variance h_variance ha
  sorry

end calc_variance_of_data_set_l188_188920


namespace pq_sum_neg1_l188_188872

theorem pq_sum_neg1 (p q : ℝ) 
  (h_eq: (1, 2) ⊗ (p, q) = (5, 0))
  (h_op: ∀ (a b c d : ℝ), (a, b) ⊗ (c, d) = (a*c - b*d, a*d + b*c)) : 
  p + q = -1 := 
sorry

end pq_sum_neg1_l188_188872


namespace sin_cos_square_plot_on_line_l188_188464

theorem sin_cos_square_plot_on_line (t : ℝ) : 
  let x := Real.sin t ^ 2,
      y := Real.cos t ^ 2 in
  x + y = 1 :=
by sorry

end sin_cos_square_plot_on_line_l188_188464


namespace skyscraper_anniversary_l188_188190

theorem skyscraper_anniversary (built_years_ago : ℕ) (anniversary_years : ℕ) (years_before : ℕ) :
    built_years_ago = 100 → anniversary_years = 200 → years_before = 5 → 
    (anniversary_years - years_before) - built_years_ago = 95 := by
  intros h1 h2 h3
  sorry

end skyscraper_anniversary_l188_188190


namespace exists_reals_condition_l188_188449

-- Define the conditions in Lean
theorem exists_reals_condition (n : ℕ) (h₁ : n ≥ 3) : 
  (∃ a : Fin (n + 2) → ℝ, a 0 = a n ∧ a 1 = a (n + 1) ∧ 
  ∀ i : Fin n, a i * a (i + 1) + 1 = a (i + 2)) ↔ 3 ∣ n := 
sorry

end exists_reals_condition_l188_188449


namespace september_first_2021_was_wednesday_l188_188784

-- Defining the main theorem based on the conditions and the question
theorem september_first_2021_was_wednesday
  (doubledCapitalOnWeekdays : ∀ day : Nat, day = 0 % 7 → True)
  (sevenFiftyPercOnWeekends : ∀ day : Nat, day = 5 % 7 → True)
  (millionaireOnLastDayOfYear: ∀ day : Nat, day = 364 % 7 → True)
  : 1 % 7 = 3 % 7 := 
sorry

end september_first_2021_was_wednesday_l188_188784


namespace part1_part2_l188_188135

def setA : Set ℝ := {x | x^2 - 5*x - 14 < 0}
def setB (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3*a - 2}

theorem part1 (a : ℝ) (h : a = 4) : 
  setA ∪ setB a = {x | -2 < x ∧ x ≤ 10} := 
by {
  substitute h,
  sorry
}

theorem part2 (a : ℝ) :
  (setB a ⊆ setA) → a < 3 := 
by {
  intro h,
  sorry
}

end part1_part2_l188_188135


namespace function_decreasing_interval_l188_188682

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Lean statement to express the interval where the function is decreasing
theorem function_decreasing_interval :
  ∀ x : ℝ, f' x < 0 ↔ x ∈ set.Iic 1 := 
by
  sorry

end function_decreasing_interval_l188_188682


namespace intersection_points_l188_188636

-- Define the conditions
def regular_polygon_sides : List ℕ := [6, 7, 8, 9]

def no_shared_vertices (sides: List ℕ) : Prop :=
  ∀ n m, n ≠ m → ¬∃ v, v ∈ n ∧ v ∈ m

def no_three_sides_intersect (sides: List ℕ) : Prop :=
  ∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c → ¬∃ p, p ∈ a ∧ p ∈ b ∧ p ∈ c

-- Define the problem
theorem intersection_points :
  no_shared_vertices regular_polygon_sides ∧ no_three_sides_intersect regular_polygon_sides →
  80 = sorry :=       -- You must provide the correct proof here.
  sorry

end intersection_points_l188_188636


namespace isosceles_triangles_l188_188959

theorem isosceles_triangles (A B C C1 F1 F2 G1 G2 : Point) (h_bc_gt_ac : B.coord.y > C.coord.x) 
    (h_bisector_f : ∃ f, is_bisector_angle A C B f)
    (h_perpendicular_g : ∃ g, is_perpendicular_line_through_point g C f)
    (h_midpoint_C1 : is_midpoint_of C1 B C)
    (h_parallel_lines : ∃ p1 p2, are_parallel_lines p1 AB ∧ are_parallel_lines p2 AB ∧ 
                      p1.intersection AC = F1 ∧ p2.intersection AC = F2 ∧ 
                      p1.intersection BC = G1 ∧ p2.intersection BC = G2) :
    is_isosceles_triangle (triangle C1 F1 F2) ∧ is_isosceles_triangle (triangle C1 G1 G2) ∧
    dist C1 F1 = dist C1 F2 ∧ dist C1 F1 = (dist B C - dist A C) / 2 ∧
    dist C1 G1 = dist C1 G2 ∧ dist C1 G1 = (dist A C + dist B C) / 2 := 
by sorry

end isosceles_triangles_l188_188959


namespace abs_eq_iff_x_eq_2_l188_188715

theorem abs_eq_iff_x_eq_2 (x : ℝ) : |x - 1| = |x - 3| → x = 2 := by
  sorry

end abs_eq_iff_x_eq_2_l188_188715


namespace years_until_5_years_before_anniversary_l188_188188

-- Definitions
def years_built_ago := 100
def upcoming_anniversary := 200
def years_before_anniversary := 5

-- Theorem statement
theorem years_until_5_years_before_anniversary :
  let years_until_anniversary := upcoming_anniversary - years_built_ago in
  let future_years := years_until_anniversary - years_before_anniversary in
  future_years = 95 :=
by
  sorry

end years_until_5_years_before_anniversary_l188_188188


namespace binomial_np_sum_l188_188519

-- Definitions of variance and expectation for a binomial distribution
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)
def binomial_expectation (n : ℕ) (p : ℚ) : ℚ := n * p

-- Statement of the problem
theorem binomial_np_sum (n : ℕ) (p : ℚ) (h_var : binomial_variance n p = 4) (h_exp : binomial_expectation n p = 12) :
    n + p = 56 / 3 := by
  sorry

end binomial_np_sum_l188_188519


namespace angle_between_unit_vectors_l188_188147

variables {ℝ : Type*} [normed_ring ℝ] [normed_space ℝ ℝ]

-- Definitions of unit vectors (norm = 1).
variables (a b : ℝ)

-- Given conditions 1 and 2
def unit_vectors : Prop := ∥a∥ = 1 ∧ ∥b∥ = 1
def condition : Prop := ∥√2 • a - b∥ = √5

-- Question: prove that the angle between a and b is 135 degrees
theorem angle_between_unit_vectors (h1 : unit_vectors a b) (h2 : condition a b) :
  ∃ θ : ℝ, θ = 135 ∧ inner_product.cos θ a b := sorry

end angle_between_unit_vectors_l188_188147


namespace probability_same_gender_l188_188263

-- Condition: There are 3 male and 2 female young volunteers.
def total_volunteers : ℕ := 5
def male_volunteers : ℕ := 3
def female_volunteers : ℕ := 2

-- Condition: 2 volunteers are selected to go to the community for public welfare activities.
def total_selections : ℕ := 2

-- Proving that the probability that the selected volunteers are of the same gender is 2/5.
theorem probability_same_gender : 
  (((choose male_volunteers total_selections) + (choose female_volunteers total_selections)) / (choose total_volunteers total_selections)) = (2 / 5) :=
  sorry

end probability_same_gender_l188_188263


namespace probability_of_one_red_ball_l188_188372

open_locale big_operators

def combination (n k : ℕ) : ℕ :=
  nat.choose n k

theorem probability_of_one_red_ball :
  let total_balls := 5 in
  let red_balls := 3 in
  let yellow_balls := 2 in
  let total_ways := combination total_balls 2 in
  let ways_to_choose_one_red_and_one_yellow := combination red_balls 1 * combination yellow_balls 1 in
  (ways_to_choose_one_red_and_one_yellow : ℚ) / total_ways = 0.6 :=
by sorry

end probability_of_one_red_ball_l188_188372


namespace sum_inequality_l188_188240

variable {α : Type} [LinearOrderedField α] 

theorem sum_inequality 
  (n : ℕ) 
  (α : Fin n → α) 
  (h1 : ∀ i, 0 < α i) 
  (h2 : ∑ i in Finset.univ, α i = 1) 
  (h3 : 2 ≤ n) : 
  ∑ i in Finset.univ, α i / (2 - α i) ≥ n / (2 * n - 1) :=
sorry

end sum_inequality_l188_188240


namespace Chicago_White_Sox_loss_l188_188440

theorem Chicago_White_Sox_loss :
  ∃ (L : ℕ), (99 = L + 36) ∧ (L = 63) :=
by
  sorry

end Chicago_White_Sox_loss_l188_188440


namespace units_digit_of_F_F_15_l188_188663

def F : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := F (n + 1) + F n

theorem units_digit_of_F_F_15 : (F (F 15)) % 10 = 5 := by
  have h₁ : ∀ n, F n % 10 = ([0, 1, 1, 2, 3, 5, 8, 3, 1, 4, 5, 9, 4, 3, 7, 0, 7, 7, 4, 1, 5, 6, 1, 7, 8, 5, 3, 8, 1, 9, 0, 9, 9, 8, 7, 5, 2, 7, 9, 6, 5, 1, 6, 7, 3, 0, 3, 3, 6, 9, 5, 4, 9, 3, 2, 5, 7, 2, 9, 1].take 60)!!.get_or_else (n % 60) 0 := sorry
  have h₂ : F 15 = 610 := by
    simp [F]; -- Proof for F 15 = 610
    sorry
  calc (F (F 15)) % 10
       = F 610 % 10 := by
         rw [h₂]
       ... = 5 := h₁ 610

end units_digit_of_F_F_15_l188_188663


namespace skyscraper_anniversary_l188_188192

theorem skyscraper_anniversary 
  (years_since_built : ℕ)
  (target_years : ℕ)
  (years_before_200th : ℕ)
  (years_future : ℕ) 
  (h1 : years_since_built = 100) 
  (h2 : target_years = 200 - 5) 
  (h3 : years_future = target_years - years_since_built) : 
  years_future = 95 :=
by
  sorry

end skyscraper_anniversary_l188_188192


namespace expected_number_seconds_to_multiple_2021_l188_188041

/--
Caroline starts with the number 1. Every second, she flips a fair coin.
If it lands heads, she adds 1 to her number.
If it lands tails, she multiplies her number by 2.

This defines a Markov process over the states n % 2021, where n is Caroline's number.

We aim to prove that the expected number of seconds it takes for Caroline's number to become a multiple of 2021 is 4040.
-/
noncomputable def expected_seconds_to_multiple_2021 : ℕ := 4040

theorem expected_number_seconds_to_multiple_2021 :
  (expected_seconds_to_multiple_2021 = 4040) :=
begin
  sorry,
end

end expected_number_seconds_to_multiple_2021_l188_188041


namespace water_percentage_l188_188756

theorem water_percentage (P : ℕ) : 
  let initial_volume := 300
  let final_volume := initial_volume + 100
  let desired_water_percentage := 70
  let water_added := 100
  let final_water_amount := desired_water_percentage * final_volume / 100
  let current_water_amount := P * initial_volume / 100

  current_water_amount + water_added = final_water_amount → 
  P = 60 :=
by sorry

end water_percentage_l188_188756


namespace no_obtuse_triangle_probability_l188_188484

noncomputable def prob_no_obtuse_triangle : ℝ :=
  9 / 128

theorem no_obtuse_triangle_probability :
  let circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 },
      points : List (ℝ × ℝ) × ℝ := (List.replicate 4 (0, 0), 1) -- assuming uniform rand points chosen
  in 
  ∀ (A₀ A₁ A₂ A₃ : ℝ × ℝ), A₀ ∈ circle → A₁ ∈ circle → A₂ ∈ circle → A₃ ∈ circle →
  (prob_no_obtuse_triangle = 9 / 128) :=
by
  sorry

end no_obtuse_triangle_probability_l188_188484


namespace percent_increase_proof_l188_188978

def skateboard_cost_last_year : ℝ := 100
def protective_gear_cost_last_year : ℝ := 50
def skateboard_increase_percent : ℝ := 0.12
def protective_gear_increase_percent : ℝ := 0.20

def new_skateboard_cost : ℝ := skateboard_cost_last_year * (1 + skateboard_increase_percent)
def new_protective_gear_cost : ℝ := protective_gear_cost_last_year * (1 + protective_gear_increase_percent)

def original_total_cost : ℝ := skateboard_cost_last_year + protective_gear_cost_last_year
def new_total_cost : ℝ := new_skateboard_cost + new_protective_gear_cost

def total_increase : ℝ := new_total_cost - original_total_cost
def percent_increase : ℝ := (total_increase / original_total_cost) * 100

theorem percent_increase_proof : percent_increase = 14.67 := by
  -- The rest of the code is omitted
  sorry

end percent_increase_proof_l188_188978


namespace poster_tells_truth_for_18_hours_l188_188114

def clever_cat_schedule : Π (t : ℕ), String :=
λ t, if t % 24 < 12 then "sleeping" else "telling stories"

def poster_statement_true (t : ℕ) : Prop :=
  clever_cat_schedule t = clever_cat_schedule (t + 3)

def hours_poster_tells_truth_per_day : ℕ :=
  (Finset.filter poster_statement_true (Finset.range 24)).card

theorem poster_tells_truth_for_18_hours :
  hours_poster_tells_truth_per_day = 18 := 
sorry

end poster_tells_truth_for_18_hours_l188_188114


namespace smallest_N_value_l188_188030

noncomputable def satisfies_conditions (N : ℕ) : Prop :=
  ∃ (points : finset (ℤ × ℤ)), points.card = 13 ∧
  ∀ (p1 p2 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → 
  real.dist p1.1 p2.1 > 2 ∧ real.dist p1.2 p2.2 > 2

theorem smallest_N_value : ∃ (N : ℕ), satisfies_conditions N ∧ ∀ (M : ℕ), satisfies_conditions M → N ≤ M :=
begin
  use 8,
  split,
  { -- Proof that satisfies_conditions 8:
    sorry },
  { -- Proof that for all M, if satisfies_conditions M then 8 ≤ M
    sorry }
end

end smallest_N_value_l188_188030


namespace find_length_BE_l188_188630

noncomputable def length_BE (AF BE CB FE FD FA AB DC : ℝ) :=
  let y := CB in
  let x := BE in
  let AF := 40 * y / x in
  let FD := (40 * y - x * y) / x in
  let FD_FA_ratio := FD / FA in
  FD_FA_ratio = 18 / (x + 40)

theorem find_length_BE (EF GF : ℝ) (AF BE CB FE FD FA AB DC : ℝ) (y x : ℝ) (parallelogram : Prop) (intersect_AC_BF : Prop) (intersect_DC_BF : Prop):
  EF = 40 → GF = 18 →
  parallelogram → 
  intersect_AC_BF → 
  intersect_DC_BF → 
  let y := CB in
  let x := BE in
  let AF := 40 * y / x in
  let FD := (40 * y - x * y) / x in
  ∀ FA, FA = 40 * y / x →
  let FD := (40 * y - x * y) / x in
  FD / FA = 18 / (x + 40) →
  BE = 20 * Real.sqrt 2 :=
begin
  sorry
end

end find_length_BE_l188_188630


namespace adjusted_smallest_part_proof_l188_188014

theorem adjusted_smallest_part_proof : 
  ∀ (x : ℝ), 14 * x = 100 → x + 12 = 19 + 1 / 7 := 
by
  sorry

end adjusted_smallest_part_proof_l188_188014


namespace negation_proposition_l188_188309

  theorem negation_proposition :
    ¬ (∃ x : ℝ, 0 < x ∧ real.log10 x ≤ 1) ↔ ∀ x : ℝ, 0 < x → real.log10 x > 1 :=
  by
    sorry
  
end negation_proposition_l188_188309


namespace minimum_reliable_length_embrasure_total_length_greater_than_half_reliable_system_with_length_lt_s_l188_188025

noncomputable def tower_wall_length : ℝ := 1
noncomputable def guards_speed_ratio : ℝ := 2

def reliable_system (length_1 length_2 : ℝ) : Prop :=
  ∀ t : ℝ, (∃ k : ℤ, (k * tower_wall_length ≤ t ∧ t < (k + length_1)) ∨ 
                      (k * tower_wall_length / guards_speed_ratio ≤ t ∧ t < (k + length_2 / guards_speed_ratio)))

theorem minimum_reliable_length_embrasure : tower_wall_length / guards_speed_ratio * (guards_speed_ratio +1) / 
(guards_speed_ratio + guards_speed_ratio) = 2*tower_balance := 
by {calculate, sorry }

theorem total_length_greater_than_half : 
  ∀ (total_length : ℝ), reliable_system guards_speed_ratio (total_length) → 
    (∃ s : ℝ, 2*tower_wall_length / guards_speed_ratio * (guards_speed_ratio + 1) >  total_length := 
by {calculate, sorry }

theorem reliable_system_with_length_lt_s (s : ℝ) (h : s > tower_wall_length /  guards_speed_ratio + 1) : 
  ∃ (length_1 : ℝ) (length_2 : ℝ), reliable_system tower_wall_length (length_1 + length_2) ∧
  length_1 + length_2 < s := 
by {calculate, sorry }


end minimum_reliable_length_embrasure_total_length_greater_than_half_reliable_system_with_length_lt_s_l188_188025


namespace find_x_for_fraction_equality_l188_188460

theorem find_x_for_fraction_equality (x : ℝ) : 
  (4 + 2 * x) / (7 + x) = (2 + x) / (3 + x) ↔ (x = -2 ∨ x = 1) := by
  sorry

end find_x_for_fraction_equality_l188_188460


namespace number_of_ways_to_distribute_cards_l188_188765

open Finset

/-- There are 100 cards numbered from 1 to 100.
The cards are placed into three boxes such that each box contains at least one card.
Each box is represented by a finite set of natural numbers, where A, B, and C represent the red, white, and blue boxes respectively.
It is known that for any two sets of these cards, their pairwise sums are distinct.
Prove that the number of ways to distribute the cards into these boxes is 12.
-/
theorem number_of_ways_to_distribute_cards : 
  ∃ (A B C : Finset ℕ), 
  (A ∪ B ∪ C = { n | 1 ≤ n ∧ n ≤ 100 }) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧ 
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (∀ x ∈ A, ∀ y ∈ B, ∀ z ∈ C, x + y ≠ y + z ∧ y + z ≠ z + x ∧ z + x ≠ x + y) ∧
  (∀(S1 S2 : Finset ℕ), S1 ∪ S2 ∈ S1.card + S2.card - 1) ∧
  (finset.card {d ∈ S | (1 ≤ d ∧ d ≤ 100)} = 12) :=
  sorry

end number_of_ways_to_distribute_cards_l188_188765


namespace problem_solution_l188_188531

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 3^x else m - x^2

def p (m : ℝ) : Prop :=
∃ x, f m x = 0

def q (m : ℝ) : Prop :=
m = 1 / 9 → f m (f m (-1)) = 0

theorem problem_solution :
  ¬ (∃ m, m < 0 ∧ p m) ∧ q (1 / 9) :=
by 
  sorry

end problem_solution_l188_188531


namespace range_of_m_for_decreasing_function_l188_188819

theorem range_of_m_for_decreasing_function (m : ℝ) :
  ∀ x, (2 ≤ x) → ∀ y, y = -x^2 - 4*m*x + 1 → 
  (-∀ x > 2, 0 > (2 * m)) := sorry

end range_of_m_for_decreasing_function_l188_188819


namespace range_of_k_l188_188314

theorem range_of_k (k : ℝ) (h : -3 < k ∧ k ≤ 0) : ∀ x : ℝ, k * x^2 + 2 * k * x - 3 < 0 :=
sorry

end range_of_k_l188_188314


namespace correct_propositions_l188_188986

-- Define vectors and conditions
variables {V M : Type} [AddCommGroup V] [Module ℝ V]

-- Definitions for vectors a, b, and c in the plane M
variables (a b c : V) (λ μ : ℝ)

-- Condition for linear dependence
def linear_dependent (a b : V) := ∃ (λ μ : ℝ), λ ≠ 0 ∧ μ ≠ 0 ∧ λ • a + μ • b = 0

-- Condition for collinear vectors
def collinear (a b : V) := ∃ (λ : ℝ), λ • a = b

-- Proposition 1: If a = 2b then linear dependence
def proposition1 : Prop := a = 2 • b → linear_dependent a b

-- Proposition 2: Non-zero and perpendicular vectors imply linear independence
def perpendicular (a b : V) := ∀ (v₁ v₂ : V), inner_product_space ℝ V a b = 0
def proposition2 : Prop := ¬ (a = 0 ∧ b = 0) ∧ perpendicular a b → ¬ linear_dependent a b

-- Proposition 3: Transitivity of linear dependence (incorrect)
def proposition3 : Prop := linear_dependent a b ∧ linear_dependent b c → linear_dependent a c

-- Proposition 4: Linear dependence if and only if collinear
def proposition4 : Prop := linear_dependent a b ↔ collinear a b

-- The main theorem statement of the problem, indicating correct propositions
theorem correct_propositions : (proposition1 a b) ∧ (¬ proposition2 a b) ∧ (¬ proposition3 a b c) ∧ (proposition4 a b) :=
by sorry

end correct_propositions_l188_188986


namespace probability_even_two_digit_number_l188_188029

def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def probability_even_draw : ℚ :=
  let total_outcomes := 9 * 8 in
  let favorable_outcomes := (4 * 3) + (5 * 4) in
  favorable_outcomes / total_outcomes

theorem probability_even_two_digit_number
  (urn : Finset ℕ) (jose_draw maria_draw : urn) :
  urn = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  (∀ a ∈ urn, ∀ b ∈ urn, a ≠ b → jose_draw = a ∧ maria_draw = b → is_even (10 * a + b) → 
  probability_even_draw = 4 / 9) :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end probability_even_two_digit_number_l188_188029


namespace price_sugar_salt_l188_188311

/-- The price of two kilograms of sugar and five kilograms of salt is $5.50. If a kilogram of sugar 
    costs $1.50, then how much is the price of three kilograms of sugar and some kilograms of salt, 
    if the total price is $5? -/
theorem price_sugar_salt 
  (price_sugar_per_kg : ℝ)
  (price_total_2kg_sugar_5kg_salt : ℝ)
  (total_price : ℝ) :
  price_sugar_per_kg = 1.50 →
  price_total_2kg_sugar_5kg_salt = 5.50 →
  total_price = 5 →
  2 * price_sugar_per_kg + 5 * (price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5 = 5.50 →
  3 * price_sugar_per_kg + (total_price - 3 * price_sugar_per_kg) / ((price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5) = 1 →
  true :=
by
  sorry

end price_sugar_salt_l188_188311


namespace values_of_x_and_y_in_log_eq_l188_188438

noncomputable def log_eq_condition (x y : ℝ) : Prop :=
  log x + log y = log (x + 2 * y)

theorem values_of_x_and_y_in_log_eq (x y : ℝ) (hx : x ≠ 2) :
  log_eq_condition x y ↔ y = x / (x - 2) := by
  sorry

end values_of_x_and_y_in_log_eq_l188_188438


namespace lindsey_squat_weight_l188_188255

-- Define the conditions
def num_bands : ℕ := 2
def resistance_per_band : ℤ := 5
def dumbbell_weight : ℤ := 10

-- Define the weight Lindsay will squat
def total_weight : ℤ := num_bands * resistance_per_band + dumbbell_weight

-- State the theorem
theorem lindsey_squat_weight : total_weight = 20 :=
by
  sorry

end lindsey_squat_weight_l188_188255


namespace max_min_distance_to_Q_max_min_slope_to_Q_l188_188137

-- Definitions from the conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 14 * y + 45 = 0
def point_Q := (-2, 3)

-- Problem (I)
theorem max_min_distance_to_Q : 
  (∀ (m n : ℝ), circle_eq m n → 
    ∃ (d_max d_min : ℝ), d_max = 6 * Real.sqrt 2 ∧ d_min = 2 * Real.sqrt 2 ∧ 
    ∀ (d : ℝ), d = Real.dist (m, n) point_Q → d_min ≤ d ∧ d ≤ d_max) :=
by {
  sorry
}

-- Problem (II)
theorem max_min_slope_to_Q : 
  (∀ (m n : ℝ), circle_eq m n → 
    2 - Real.sqrt 3 ≤ (n - 3) / (m + 2) ∧ (n - 3) / (m + 2) ≤ 2 + Real.sqrt 3) :=
by {
  sorry
}

end max_min_distance_to_Q_max_min_slope_to_Q_l188_188137


namespace unshaded_area_inside_triangle_l188_188389

theorem unshaded_area_inside_triangle (r : ℝ) (h_r : r = 1) : 
  let s := 2 * r in
  let triangle_area := (real.sqrt 3 / 4) * s^2 in
  let semicircle_area := (π / 2) * r^2 in
  let shaded_segment_area := (π / 6) * r^2 in
  let unshaded_area := triangle_area - shaded_segment_area in
  unshaded_area = real.sqrt 3 - (π / 6) := 
by
  -- here we will insert the proof steps which we currently omit with sorry
  sorry

end unshaded_area_inside_triangle_l188_188389


namespace perpendicular_MN_CD_l188_188619

open_locale classical

variables {A B C D N M : ℝ²}

-- Points are cocyclic
axiom cocyclic_points : cocyclic {A, B, C, D}

-- AC is perpendicular to BD at N
axiom perp_AC_BD_at_N : ∃ N, (A, C) ⊥ (B, D) ∧ lines_intersect_at A C B D N

-- M is the midpoint of AB
axiom midpoint_M : M = midpoint A B

-- Required to prove MN is perpendicular to CD
theorem perpendicular_MN_CD : (M, N) ⊥ (C, D) :=
  sorry

end perpendicular_MN_CD_l188_188619


namespace min_sum_geometric_sequence_l188_188584

noncomputable def sequence_min_value (a : ℕ → ℝ) : ℝ :=
  a 4 + a 3 - 2 * a 2 - 2 * a 1

theorem min_sum_geometric_sequence (a : ℕ → ℝ)
  (h : sequence_min_value a = 6) :
  a 5 + a 6 = 48 := 
by
  sorry

end min_sum_geometric_sequence_l188_188584


namespace functional_equation_solution_l188_188118

theorem functional_equation_solution (α : ℚ) (hα : 0 < α) (f : ℚ+ → ℚ+) :
  (∀ (x y : ℚ+), f (x / y + y) = f x / f y + f y + α * x) →
  (α = 2 → ∀ x : ℚ+, f x = x^2) ∧ (α ≠ 2 → ¬ ∃ g : ℚ+ → ℚ+, ∀ (x y : ℚ+), g (x / y + y) = g x / g y + g y + α * x) :=
by
  sorry

end functional_equation_solution_l188_188118


namespace meal_combinations_l188_188260

theorem meal_combinations :
    let fruits := 3
    let salads := (4.choose 2)
    let desserts := 5
    fruits * salads * desserts = 90 :=
by
    let fruits := 3
    let salads := (4.choose 2)
    let desserts := 5
    have calc1 : salads = 6 := by norm_num
    have calc2 : fruits * salads * desserts = 90 := by norm_num
    exact calc2

end meal_combinations_l188_188260


namespace collinear_M_N_B_l188_188335

variables {A B M N : Point}
variables (circle1 circle2 : Circle)

-- Conditions
axiom intersection_points (h1 : A ∈ circle1) (h2 : A ∈ circle2) (h3 : B ∈ circle1) (h4 : B ∈ circle2) : Prop
axiom are_diameters (hm : M ∈ circle1.diameter_points A) (hn : N ∈ circle2.diameter_points A) : Prop

-- The proof statement
theorem collinear_M_N_B (h1 : A ∈ circle1) (h2 : A ∈ circle2) (h3 : B ∈ circle1) (h4 : B ∈ circle2)
  (hm : M ∈ circle1.diameter_points A) (hn : N ∈ circle2.diameter_points A) :
  collinear {M, N, B} :=
by
  sorry

end collinear_M_N_B_l188_188335


namespace onions_total_l188_188637

-- Define the number of onions grown by Sara, Sally, and Fred
def sara_onions : ℕ := 4
def sally_onions : ℕ := 5
def fred_onions : ℕ := 9

-- Define the total onions grown
def total_onions : ℕ := sara_onions + sally_onions + fred_onions

-- Theorem stating the total number of onions grown
theorem onions_total : total_onions = 18 := by
  sorry

end onions_total_l188_188637


namespace water_fee_part1_water_fee_part2_water_fee_usage_l188_188808

theorem water_fee_part1 (x : ℕ) (h : 0 < x ∧ x ≤ 6) : y = 2 * x :=
sorry

theorem water_fee_part2 (x : ℕ) (h : x > 6) : y = 3 * x - 6 :=
sorry

theorem water_fee_usage (y : ℕ) (h : y = 27) : x = 11 :=
sorry

end water_fee_part1_water_fee_part2_water_fee_usage_l188_188808


namespace nested_sqrt_eq_five_l188_188066

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l188_188066


namespace problem_solution_l188_188984

-- Definitions for the digits and arithmetic conditions
def is_digit (n : ℕ) : Prop := n < 10

-- Problem conditions stated in Lean
variables (A B C D E : ℕ)

-- Define the conditions
axiom digits_A : is_digit A
axiom digits_B : is_digit B
axiom digits_C : is_digit C
axiom digits_D : is_digit D
axiom digits_E : is_digit E

-- Subtraction result for second equation
axiom sub_eq : A - C = A

-- Additional conditions derived from the problem
axiom add_eq : (E + E = D)

-- Now, state the problem in Lean
theorem problem_solution : D = 8 :=
sorry

end problem_solution_l188_188984


namespace num_negative_in_list_l188_188680

def is_negative (x : ℝ) : Prop := x < 0

def list_of_numbers : List ℝ := [
  -1/2,
  -[+[-3]],
  -abs (-2),
  (-1)^2021,
  0,
  (-2) * (-3)
]

theorem num_negative_in_list :
  (list_of_numbers.filter is_negative).length = 3 := 
sorry

end num_negative_in_list_l188_188680


namespace impossible_to_fill_grid_l188_188928

theorem impossible_to_fill_grid :
  ¬ (∃ (f : Fin 4 → Fin 4 → ℕ) (a b : ℕ),
    (∀ i : Fin 4, (∑ j, f i j) = a + 2 * i) ∧
    (∀ j : Fin 4, (∑ i, f i j) = b + 3 * j)) :=
by
  sorry

end impossible_to_fill_grid_l188_188928


namespace distinct_values_x0_for_x6_eq_x0_l188_188494

-- Define the initial conditions and the sequence
def sequence (x₀ : ℝ) : ℕ → ℝ
| 0       := x₀
| (n + 1) := if 2 * sequence n < 1 then 2 * sequence n else 2 * sequence n - 1

-- Define the main theorem stating there are 64 distinct valid x₀
theorem distinct_values_x0_for_x6_eq_x0 : ∀ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ < 1 → (∃ s : set ℝ, s.card = 64 ∧ ∀ x ∈ s, sequence x 6 = x) := 
sorry

end distinct_values_x0_for_x6_eq_x0_l188_188494


namespace smallest_number_with_70_divisors_l188_188103

theorem smallest_number_with_70_divisors:
  ∃ n : ℕ, (∀ m : ℕ, (∀ d : ℕ, (d ∣ n ↔ d ∣ m) → m = n) ∧ (divisor_count n = 70 → n = 25920)) :=
sorry

noncomputable def divisor_count (n : ℕ) : ℕ :=
  ∑ d in divisors n, 1

lemma divisors (n : ℕ) : finset ℕ :=
  finset.filter (λ d, d ∣ n) (finset.range (n+1))

lemma divisor_count_correct (n : ℕ) : divisor_count n = 
  ∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n+1))), 1 :=
by sorry

end smallest_number_with_70_divisors_l188_188103


namespace yellow_percentage_l188_188013

theorem yellow_percentage (s w : ℝ) 
  (h_cross : w * w + 4 * w * (s - 2 * w) = 0.49 * s * s) : 
  (w / s) ^ 2 = 0.2514 :=
by
  sorry

end yellow_percentage_l188_188013


namespace rectangular_plot_area_l188_188780

theorem rectangular_plot_area (P : ℝ) (L W : ℝ) (h1 : P = 24) (h2 : L = 2 * W) :
    A = 32 := by
  sorry

end rectangular_plot_area_l188_188780


namespace game_winning_strategy_l188_188023

theorem game_winning_strategy (n : ℕ) (h : n ≥ 3) :
  (∃ k : ℕ, n = 3 * k + 2) → (∃ k : ℕ, n = 3 * k + 2 ∨ ∀ k : ℕ, n ≠ 3 * k + 2) :=
by
  sorry

end game_winning_strategy_l188_188023


namespace train_crossing_signal_pole_time_l188_188748

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 600.0000000000001
noncomputable def time_cross_platform : ℝ := 54
noncomputable def total_distance : ℝ := train_length + platform_length

theorem train_crossing_signal_pole_time :
  let speed := total_distance / time_cross_platform
  let expected_time := 18
  (train_length / speed).round = expected_time :=
by
  sorry

end train_crossing_signal_pole_time_l188_188748


namespace no_other_three_prime_products_l188_188717

theorem no_other_three_prime_products :
  let p := 2
  let q := 3
  let r := 1301
  ∀ (a b c : ℕ), prime a → prime b → prime c → a < b ∧ b < c → 
  a + b + c = p + q + r → (a = p ∧ b = q ∧ c = r) :=
by
  intros p q r a b c ha hb hc habc hsum
  sorry

end no_other_three_prime_products_l188_188717


namespace ellipse_eqn_is_correct_max_inradius_triangle_l188_188504

noncomputable def ellipse_params (a b : ℝ) : Prop :=
(0 < b) ∧ (b < a) ∧ (a * a = 4) ∧ (b * b = 1) ∧
(2 / 2 = 1) ∧ (a * 2 = 4) ∧ (b * a / 2 = 1/2) ∧
(3 / 4 / a / a + 1 / b / b = 1)

noncomputable def equation_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
(ellipse_params a b) ∧ (float $ (sqrt 3) y)

theorem ellipse_eqn_is_correct :
  ∃ a b, equation_ellipse a b = (frac $ (x as ℝ) ) 4 + y * y == 1
  := sorry

theorem max_inradius_triangle :
  ∃ PQ P F2 Q z, ellipse_eqn_is_correct PQ P F2 Q := λ PQ P y P Q F Triangle $ frac $ (max_of x) ) = z := sorry

end ellipse_eqn_is_correct_max_inradius_triangle_l188_188504


namespace gcd_m_n_l188_188250

-- Define the numbers m and n
def m : ℕ := 555555555
def n : ℕ := 1111111111

-- State the problem: Prove that gcd(m, n) = 1
theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Proof goes here
  sorry

end gcd_m_n_l188_188250


namespace length_of_AH_l188_188864

-- Define the given triangle ABC with specific angles and side length
noncomputable def triangle_ABC := 
  { ABC : Triangle | 
    ∠ ABC = 80 ∧ 
    ∠ ACB = 70 ∧ 
    BC = 2 }

-- Define point H as the orthocenter of triangle ABC
def orthocenter_H (T: Triangle) : Point :=
  T.orthocenter

-- The length of orthocenter AH
def AH_length (T: Triangle) (H: Point) : ℝ :=
  T.AH_length H

-- The theorem we need to prove
theorem length_of_AH {T : Triangle} (hT : T ∈ triangle_ABC) : 
  ∃ H, H = orthocenter_H T ∧ AH_length T H = 2 * sqrt 3 :=
sorry -- the proof is omitted

end length_of_AH_l188_188864


namespace no_obtuse_triangle_probability_l188_188482

noncomputable def prob_no_obtuse_triangle : ℝ :=
  9 / 128

theorem no_obtuse_triangle_probability :
  let circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 },
      points : List (ℝ × ℝ) × ℝ := (List.replicate 4 (0, 0), 1) -- assuming uniform rand points chosen
  in 
  ∀ (A₀ A₁ A₂ A₃ : ℝ × ℝ), A₀ ∈ circle → A₁ ∈ circle → A₂ ∈ circle → A₃ ∈ circle →
  (prob_no_obtuse_triangle = 9 / 128) :=
by
  sorry

end no_obtuse_triangle_probability_l188_188482


namespace exists_idempotent_l188_188364

-- Definition of the set M as the natural numbers from 1 to 1993
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1993 }

-- Operation * on M
noncomputable def star (a b : ℕ) : ℕ := sorry

-- Hypothesis: * is closed on M and (a * b) * a = b for any a, b in M
axiom star_closed (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star a b ∈ M
axiom star_property (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star (star a b) a = b

-- Goal: Prove that there exists a number a in M such that a * a = a
theorem exists_idempotent : ∃ a ∈ M, star a a = a := by
  sorry

end exists_idempotent_l188_188364


namespace sqrt_equation_has_solution_l188_188057

noncomputable def x : ℝ := Real.sqrt (20 + x)

theorem sqrt_equation_has_solution : x = 5 :=
by
  sorry

end sqrt_equation_has_solution_l188_188057


namespace find_lambda_parallel_l188_188548

theorem find_lambda_parallel (λ : ℝ) : 
  let a := (1 : ℝ, 1 : ℝ),
      b := (-1 : ℝ, 3 : ℝ),
      c := (2 : ℝ, 1 : ℝ),
      v := (1 + λ, 1 - 3 * λ)
  in (v.1 * c.2 - v.2 * c.1 = 0) → λ = 1 / 7 :=
sorry

end find_lambda_parallel_l188_188548


namespace length_GP_l188_188618

-- Definitions based on conditions
def rhombus (JH ON: ℝ) (JG GN NP: ℝ → ℝ) : Prop :=
  JH = 16 ∧ ON = 12 ∧ JG / GN = 2 / 2 ∧ GN / NP = 2 / 1

-- The main theorem to prove length of GP
theorem length_GP (JH ON: ℝ) (JG GN NP: ℝ → ℝ) (GP: ℝ) : 
  rhombus JH ON JG GN NP → GP = 3 * sqrt 17 / 2 :=
by
  sorry

end length_GP_l188_188618


namespace factor_condition_for_polynomial_l188_188448

theorem factor_condition_for_polynomial (t : ℝ) :
  (x - t) is_a_factor_of (4 * x^2 + 9 * x - 2)
  ↔ ((t = -1/4) ∨ (t = -2)) := by sorry

end factor_condition_for_polynomial_l188_188448


namespace percentage_of_students_owning_birds_l188_188585

theorem percentage_of_students_owning_birds
    (total_students : ℕ) 
    (students_owning_birds : ℕ) 
    (h_total_students : total_students = 500) 
    (h_students_owning_birds : students_owning_birds = 75) : 
    (students_owning_birds * 100) / total_students = 15 := 
by 
    sorry

end percentage_of_students_owning_birds_l188_188585


namespace winning_candidate_percentage_l188_188746

theorem winning_candidate_percentage :
  let total_votes := 1136 + 5636 + 11628 in
  let winning_votes := 11628 in
  (winning_votes / total_votes : ℝ) * 100 ≈ 63.17 := 
by 
  let total_votes := 1136 + 5636 + 11628
  have h_total_votes : total_votes = 18400 := by rfl
  have h_winning_votes : winning_votes = 11628 := by rfl
  sorry

end winning_candidate_percentage_l188_188746


namespace taller_cycle_shadow_length_l188_188708

theorem taller_cycle_shadow_length 
  (h_taller : ℝ) (h_shorter : ℝ) (shadow_shorter : ℝ) (shadow_taller : ℝ) 
  (h_taller_val : h_taller = 2.5) 
  (h_shorter_val : h_shorter = 2) 
  (shadow_shorter_val : shadow_shorter = 4)
  (similar_triangles : h_taller / shadow_taller = h_shorter / shadow_shorter) :
  shadow_taller = 5 := 
by 
  sorry

end taller_cycle_shadow_length_l188_188708


namespace no_obtuse_triangle_probability_l188_188483

noncomputable def prob_no_obtuse_triangle : ℝ :=
  9 / 128

theorem no_obtuse_triangle_probability :
  let circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 },
      points : List (ℝ × ℝ) × ℝ := (List.replicate 4 (0, 0), 1) -- assuming uniform rand points chosen
  in 
  ∀ (A₀ A₁ A₂ A₃ : ℝ × ℝ), A₀ ∈ circle → A₁ ∈ circle → A₂ ∈ circle → A₃ ∈ circle →
  (prob_no_obtuse_triangle = 9 / 128) :=
by
  sorry

end no_obtuse_triangle_probability_l188_188483


namespace find_positive_number_l188_188737

theorem find_positive_number 
  (x : ℝ) (h_pos : x > 0) 
  (h_eq : (2 / 3) * x = (16 / 216) * (1 / x)) : 
  x = 1 / 3 :=
by
  -- This is indicating that we're skipping the actual proof steps
  sorry

end find_positive_number_l188_188737


namespace james_age_when_tim_is_79_l188_188962

variable {James_age John_age Tim_age : ℕ}

theorem james_age_when_tim_is_79 (J_age J_age_at_23 J_age_diff J_age_j : ℕ) 
                                  (H1 : J_age = J_age_at_23 - J_age_diff)
                                  (H2 : John_age = 35)
                                  (H3 : James_age = 23)
                                  (age_diff:12: ℕ )
                                  (H4 : Tim_age = 2 * John_age - 5)
                                  (H5 : Tim_age = 79):
                                  J_age=30 :=
by
  sorry

end james_age_when_tim_is_79_l188_188962


namespace wave_velocity_problem_l188_188762

noncomputable def wave_velocity_ratio : ℝ := 5 / 2

theorem wave_velocity_problem 
  (T μ : ℝ)
  (h1 : ∀ (v1 : ℝ), v1 = Real.sqrt (T / μ))
  (h2 : ∀ (v2 v1 : ℝ), v2 = v1 / Real.sqrt 2 → v2 = Real.sqrt ((T / 2) / μ))
  (h3 : ∀ (v3 v1 : ℝ), v3 = v1 / 2 → v3 = Real.sqrt ((T / 4) / μ)) :
  (h4 : ∀ (v1 v3 : ℝ), v3 = v1 / 2 → ((v1 / v3) + (v3 / v1)) = wave_velocity_ratio) →
  (m n r : ℕ) (h5 : m = 5) (h6 : n = 2) (h7 : r = 1) :
  m + n + r = 8 :=
begin
  -- Proof is omitted as per the problem statement
  sorry
end

end wave_velocity_problem_l188_188762


namespace abs_le_and_interval_iff_l188_188367

variable (x : ℝ)

theorem abs_le_and_interval_iff :
  (|x - 2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end abs_le_and_interval_iff_l188_188367


namespace division_of_decimals_l188_188810

theorem division_of_decimals : (0.45 : ℝ) / (0.005 : ℝ) = 90 := 
sorry

end division_of_decimals_l188_188810


namespace annual_interest_rate_l188_188028

-- Definitions based on the conditions
def principal : ℝ := 800
def accumulated : ℝ := 882
def num_compounds_per_year : ℝ := 2
def time_in_years : ℝ := 1

-- The compound interest formula given the above conditions
theorem annual_interest_rate : 
  ∃ r : ℝ, accumulated = principal * (1 + r / num_compounds_per_year) ^ (num_compounds_per_year * time_in_years) → r = 0.10 := 
by 
  sorry

end annual_interest_rate_l188_188028


namespace candidate_lost_by_l188_188374

-- Given condition: The candidate got 30% of the votes and there were 10000 votes cast.
def candidate_votes (total_votes : ℕ) : ℕ := (30 * total_votes) / 100
def rival_votes (total_votes : ℕ) : ℕ := (70 * total_votes) / 100
def vote_difference (total_votes : ℕ) : ℕ := rival_votes total_votes - candidate_votes total_votes

-- Theorem statement: Prove that the candidate lost by 4000 votes.
theorem candidate_lost_by (total_votes : ℕ) (h : total_votes = 10000) : vote_difference total_votes = 4000 :=
by
  rw [h, vote_difference, rival_votes, candidate_votes]
  norm_num
  sorry

end candidate_lost_by_l188_188374


namespace find_length_of_BD_l188_188587

theorem find_length_of_BD
  (A B C D M : Point)
  (isosceles : (AB = 10) ∧ (AC = 12))
  (midpoint : M = midpoint A C)
  (BC_calculated : BC = sqrt (AB^2 - (AC/2)^2))
  (angle_bisector : B bisects_angle A C D)
  : BD = 5 := sorry

end find_length_of_BD_l188_188587


namespace equal_intercepts_on_both_axes_l188_188527

theorem equal_intercepts_on_both_axes (m : ℝ) :
  (5 - 2 * m ≠ 0) ∧
  (- (5 - 2 * m) / (m^2 - 2 * m - 3) = - (5 - 2 * m) / (2 * m^2 + m - 1)) ↔ m = -2 :=
by sorry

end equal_intercepts_on_both_axes_l188_188527


namespace symmetric_points_l188_188529

def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^2 + 4 * x + 1 else 2 / Real.exp x

-- Define the predicate that checks if a point (x, f(x)) is symmetric with respect to the origin.
def is_symmetric_point (x : ℝ) : Prop :=
f x = -f (-x)

theorem symmetric_points : ∃! x : ℝ, is_symmetric_point x := sorry

end symmetric_points_l188_188529


namespace same_answer_l188_188272

def Person (A_truth_teller B_liar : Prop) : Prop :=
  (∀ P : Prop, (A_truth_teller → P) ∧ (B_liar → ¬P)) →
  (∀ phi : Prop, (phi → ¬(P)) ∧ (¬phi → P)) :=
  sorry

theorem same_answer (person_A person_B : Prop) (question : Prop) :
  (person_A → person_A = (¬person_B)) →
  (person_B → person_A = person_B) →
  ∀ (phi : Prop), question = ("Would the other person say that you always tell the truth?") →
  ((∀ P : Prop, ((person_A → P) ∧ (person_B → ¬P)) →
  ((P → ¬(phi)) ∧ (¬P → phi)))
  sorry

end same_answer_l188_188272


namespace classroom_partition_l188_188578

universe u

def symmetric_relation {α : Type u} (R : α → α → Prop) :=
  ∀ ⦃x y : α⦄, R x y → R y x

noncomputable def exists_partition (α : Type u) [Fintype α] (students : α) 
  (knows : α → α → Prop)
  (h_symm : symmetric_relation knows) 
  (h_card : 4 ≤ Fintype.card α)
  (h_condition : ∀ (s : Finset α), s.card = 4 → ∃ (x : α), (∀ y ∈ s.erase x, knows x y ∨ ¬knows x y)) 
: Prop :=
  ∃ (group1 group2 : Finset α), group1 ∪ group2 = Finset.univ ∧ group1 ∩ group2 = ∅ ∧ 
    (∀ x ∈ group1, ∀ y ∈ group1, x ≠ y → knows x y) ∧ 
    (∀ x ∈ group2, ∀ y ∈ group2, x ≠ y → ¬knows x y)

theorem classroom_partition (α : Type u) [Fintype α] (knows : α → α → Prop)
  (h_symm : symmetric_relation knows) 
  (h_card : 4 ≤ Fintype.card α)
  (h_condition : ∀ (s : Finset α), s.card = 4 → ∃ (x : α), (∀ y ∈ s.erase x, knows x y ∨ ¬knows x y)) 
: exists_partition α α knows h_symm h_card h_condition := sorry

end classroom_partition_l188_188578


namespace num_valid_subsets_l188_188902

open Finset

noncomputable def setM : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def isValidSubset (A : Finset ℕ) : Prop :=
  A.sum id = 8 ∧ A ⊆ setM

theorem num_valid_subsets : (filter isValidSubset (powerset setM)).card = 6 := 
sorry

end num_valid_subsets_l188_188902


namespace third_smallest_palindromic_prime_three_digit_l188_188821

-- Define a three-digit number
def is_three_digit (x : ℕ) : Prop := 100 ≤ x ∧ x ≤ 999

-- Define a palindromic number
def is_palindromic (x : ℕ) : Prop :=
  let digits := x.to_digits 10
  digits = digits.reverse

-- Define a prime number
def is_prime (x : ℕ) : Prop := Nat.Prime x

-- Define the third-smallest three-digit palindromic prime
theorem third_smallest_palindromic_prime_three_digit :
  ∃ x : ℕ, is_three_digit x ∧ is_palindromic x ∧ is_prime x ∧ ∀ y : ℕ, (is_three_digit y ∧ is_palindromic y ∧ is_prime y → y < x → y = 131 ∨ y = 151) :=
  ∃ x : ℕ, is_three_digit x ∧ is_palindromic x ∧ is_prime x ∧ ∀ y : ℕ, (is_three_digit y ∧ is_palindromic y ∧ is_prime y → y < x → (y = 131 ∨ y = 151)) ∧ x = 181 :=
sorry

end third_smallest_palindromic_prime_three_digit_l188_188821


namespace multiplication_with_letters_l188_188370

theorem multiplication_with_letters :
  ∃ (S E A M T : ℕ),
    S ≠ 0 ∧ E ≠ 0 ∧ A ≠ 0 ∧ M ≠ 0 ∧ T ≠ 0 ∧
    S ≠ E ∧ S ≠ A ∧ S ≠ M ∧ S ≠ T ∧
    E ≠ A ∧ E ≠ M ∧ E ≠ T ∧
    A ≠ M ∧ A ≠ T ∧
    M ≠ T ∧
    S = 8 ∧ E = 9 ∧ A = 7 ∧ M = 3 ∧ T = 4 ∧
    let num := 1000 * S + 100 * E + 10 * A + M,
        multiplier := T
    in num * multiplier = 39784 :=
by
  sorry

end multiplication_with_letters_l188_188370


namespace function_inequality_l188_188620

theorem function_inequality {f : ℝ → ℝ} 
  (h1 : ∀ (x y : ℝ), 0 < x → 0 < y → f(x * y) ≤ f(x) * f(y)) 
  (h2 : ∀ x : ℝ, 0 < x → f(x) > 0) :
  ∀ (x : ℝ) (n : ℕ), 0 < x → 
  f(x ^ n) ≤ (finset.range n).sum (λ k, f (x ^ (k + 1)) ^ (1 / (k + 1))) :=
sorry

end function_inequality_l188_188620


namespace p_satisfies_conditions_l188_188989

noncomputable def p (x : ℕ) : ℕ := sorry

theorem p_satisfies_conditions (h_monic : p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5) : 
  p 6 = 126 := sorry

end p_satisfies_conditions_l188_188989


namespace infinite_nested_sqrt_l188_188071

theorem infinite_nested_sqrt :
  let x := \sqrt{20 + \sqrt{20 + \sqrt{20 + \sqrt{20 + \cdots}}}} in
  x = 5 :=
begin
  let x : ℝ := sqrt(20 + sqrt(20 + sqrt(20 + sqrt(20 + ...)))),
  have h1 : x = sqrt(20 + x), from sorry,
  have h2 : x^2 = 20 + x, from sorry,
  have h3 : x^2 - x - 20 = 0, from sorry,
  have h4 : (x - 5) * (x + 4) = 0, from sorry,
  have h5 : x = 5 ∨ x = -4, from sorry,
  have h6 : x >= 0, from sorry,
  exact h5.elim (λ h, h) (λ h, (h6.elim_left h))
end

end infinite_nested_sqrt_l188_188071


namespace vector_dot_product_condition_l188_188886

open Real

variables {a b : ℝ → ℝ → ℝ} (ha : a ≠ 0) (hb : b ≠ 0)

theorem vector_dot_product_condition (h : a • b = ∥ a ∥ * ∥ b ∥) :
  a ∥ ∥ b ∥ ∧ ¬ (a ∥ ∥ b  → a • b = ∥a∥ * ∥b∥) :=
by
sorrry

end vector_dot_product_condition_l188_188886


namespace percent_palindromes_with_seven_l188_188771

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_valid_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 2000

def count_with_seven (low high : ℕ) : ℕ :=
  (list.range' low (high - low)).count (λ x, is_palindrome x ∧ x.to_string.contains '7')

def total_palindromes (low high : ℕ) : ℕ :=
  (list.range' low (high - low)).count (λ x, is_palindrome x)

theorem percent_palindromes_with_seven : 
  1000≤ x → x < 2000 → ((count_with_seven 1000 2000) * 100 / (total_palindromes 1000 2000) = 18) := 
by sorry

end percent_palindromes_with_seven_l188_188771


namespace part_a_part_b_part_c_part_d_l188_188536

def seq_a (n : ℕ) : ℕ := 2 ^ n

def seq_b (n : ℕ) : ℕ :=
  if h : n > 0 then 
    let m := (1 + (Int.ofNat n).sqrt) in
    if 2 * m * m = n then seq_a (m.toNat)
    else
      let pos := n - m.toNat - 1
      seq_a (m.toNat - 1) + ((seq_a m.toNat - seq_a (m.toNat - 1)) * pos / (m.toNat - 1 + 1))
  else seq_a 1

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, seq_b (i + 1))

theorem part_a (n : ℕ) : (Finset.range n).sum (λ i, seq_b (i + 1)) = 3 * n * 2 ^ (n - 1) :=
sorry

theorem part_b (n m : ℕ) (h : n = 10 ∧ m = 66) : seq_a n ≠ seq_b m :=
sorry

theorem part_c (n : ℕ) (h : n = 72) : seq_b n = 3072 :=
sorry

theorem part_d (n : ℕ) (h : n = 55) : S n = 14337 :=
sorry

end part_a_part_b_part_c_part_d_l188_188536


namespace find_value_of_number_l188_188176

theorem find_value_of_number :
  ∃ x : ℕ, (400 * 7000) = (28000 * (x^1)) ∧ x = 100 :=
begin
  sorry
end

end find_value_of_number_l188_188176


namespace win_lose_area_l188_188759

noncomputable def spinner_radius : ℝ := 12
noncomputable def total_area : ℝ := π * spinner_radius^2
noncomputable def win_probability : ℝ := 1/3
noncomputable def lose_probability : ℝ := 1/2

theorem win_lose_area :
  let win_area := win_probability * total_area,
      lose_area := lose_probability * total_area
  in win_area = 48 * π ∧ lose_area = 72 * π :=
by
  let win_area := win_probability * total_area
  let lose_area := lose_probability * total_area
  have : total_area = 144 * π := sorry
  have : win_area = 48 * π := sorry
  have : lose_area = 72 * π := sorry
  exact ⟨this, this⟩

end win_lose_area_l188_188759


namespace num_mappings_l188_188234

-- Definitions for sets A and B
def A : Set ℕ := {0, 1, 2}  -- equivalent to {A, B, C}
def B : Set ℤ := {-1, 0, 1}

-- Definition for the function f that maps A to B
def f : A → B

-- Mathematical statement to be proved
theorem num_mappings (h : ∀ a b c : A, f a + f b + f c = 0) : 
  ∃! (f : A → B), 7 = fintype.card {g : A → B // h g} := 
sorry

end num_mappings_l188_188234


namespace catering_budget_total_l188_188972

theorem catering_budget_total 
  (total_guests : ℕ)
  (guests_want_chicken guests_want_steak : ℕ)
  (cost_steak cost_chicken : ℕ) 
  (H1 : total_guests = 80)
  (H2 : guests_want_steak = 3 * guests_want_chicken)
  (H3 : cost_steak = 25)
  (H4 : cost_chicken = 18)
  (H5 : guests_want_chicken + guests_want_steak = 80) :
  (guests_want_chicken * cost_chicken + guests_want_steak * cost_steak = 1860) := 
by
  sorry

end catering_budget_total_l188_188972


namespace negative_expression_l188_188793

theorem negative_expression :
  -(-1) ≠ -1 ∧ (-1)^2 ≠ -1 ∧ |(-1)| ≠ -1 ∧ -|(-1)| = -1 :=
by
  sorry

end negative_expression_l188_188793


namespace ordering_of_a_b_c_l188_188120

noncomputable def a : ℝ := (3/2)^(-0.6)
noncomputable def b : ℝ := Real.log 4 / Real.log (1/3)
noncomputable def c : ℝ := (2/3)^0.9

theorem ordering_of_a_b_c : a > c ∧ c > b :=
by
  -- These have to be derived based on the mathematical conditions given:
  -- a = (3/2)^(-0.6), b = log_{1/3}(1/4), c = (2/3)^{0.9}
  sorry

end ordering_of_a_b_c_l188_188120


namespace trajectory_midpoint_eq_equation_of_line_l188_188131

-- Proof Problem 1
theorem trajectory_midpoint_eq (x y : ℝ) (h1 : x^2 + y^2 - 3 * x = 0)
  (h2 : x^2 + y^2 - 6 * x + 5 = 0) : 
  x^2 + y^2 - 3 * x = 0 ∧ 5 / 3 < x ∧ x < 3 := sorry

-- Proof Problem 2
theorem equation_of_line (k : ℝ) (x1 x2 x3 x4 : ℝ) (y1 y2 y3 y4 : ℝ) 
  (h1 : y3 - y1 = y2 - y4 ∧ x2 + x1 = x4 + x3)
  (h2 : (4 * k ^ 2 + 1)/(k ^ 2) = (4 * k ^ 2 + 6)/(1 + k ^ 2)) :
  (k = 1 ∧ (k ≠ 0)) ∨ (k = -1 ∨ (k ≠ 0)) ∨ (x1 + x2 = x3 + x4): sorry

end trajectory_midpoint_eq_equation_of_line_l188_188131


namespace infinite_nested_sqrt_l188_188068

theorem infinite_nested_sqrt :
  let x := \sqrt{20 + \sqrt{20 + \sqrt{20 + \sqrt{20 + \cdots}}}} in
  x = 5 :=
begin
  let x : ℝ := sqrt(20 + sqrt(20 + sqrt(20 + sqrt(20 + ...)))),
  have h1 : x = sqrt(20 + x), from sorry,
  have h2 : x^2 = 20 + x, from sorry,
  have h3 : x^2 - x - 20 = 0, from sorry,
  have h4 : (x - 5) * (x + 4) = 0, from sorry,
  have h5 : x = 5 ∨ x = -4, from sorry,
  have h6 : x >= 0, from sorry,
  exact h5.elim (λ h, h) (λ h, (h6.elim_left h))
end

end infinite_nested_sqrt_l188_188068


namespace law_of_sines_equiv_l188_188576

theorem law_of_sines_equiv {α β γ a b c : ℝ} (h1 : α + β + γ = 180)
    (h2 : a / sin α = b / sin β) : a * sin β = b * sin α :=
by sorry

end law_of_sines_equiv_l188_188576


namespace probability_x_lt_y_in_rectangle_l188_188007

structure Point where
  x : ℝ
  y : ℝ

def rectangle : set Point := 
  {p | 0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 3}

def region_x_lt_y : set Point :=
  {p | p ∈ rectangle ∧ p.x < p.y}

noncomputable def area (s : set Point) : ℝ :=
  if h : ∃ a b, s = {p | 0 ≤ p.x ∧ p.x ≤ a ∧ 0 ≤ p.y ∧ p.y ≤ b}
  then let ⟨a, b, hs⟩ := h in a * b else 0

noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)

theorem probability_x_lt_y_in_rectangle : 
  let area_rectangle := area rectangle
  let area_triangle := triangle_area ⟨0,0⟩ ⟨3,3⟩ ⟨0,3⟩
  area_rectangle ≠ 0 → (area_triangle / area_rectangle) = 3 / 8 := by
  intros area_rectangle area_triangle h
  sorry

end probability_x_lt_y_in_rectangle_l188_188007


namespace construct_ngon_vertices_l188_188855

theorem construct_ngon_vertices (n : ℕ) (h_odd : n % 2 = 1) 
  (midpoints : Fin n → ℝ × ℝ) :
  ∃ vertices : Fin n → ℝ × ℝ, (∀ i : Fin n, 
    midpoints i = ((vertices i).fst + (vertices (i + 1) % n).fst) / 2 , 
                    (vertices i).snd + (vertices (i + 1) % n).snd) / 2) :=
sorry

end construct_ngon_vertices_l188_188855


namespace parabola_directrix_l188_188679

theorem parabola_directrix (p : ℝ) (h : y^2 = -8 * x) : directrix (y^2 = -8 * x) = x = 2 :=
begin
  sorry
end

end parabola_directrix_l188_188679


namespace hyperbola_correct_l188_188938

noncomputable def hyperbola_properties : Prop :=
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  let b := Real.sqrt ((c^2) - (a^2))
  (h + k + a + b = 4 * Real.sqrt 3 + 6)

theorem hyperbola_correct : hyperbola_properties :=
by
  unfold hyperbola_properties
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  have b : ℝ := Real.sqrt ((c^2) - (a^2))
  sorry

end hyperbola_correct_l188_188938


namespace sum_of_powers_of_i_equals_one_l188_188811

open Complex

noncomputable def sum_of_powers_of_i : ℂ :=
  ∑ (n : ℤ) in finset.Icc (-50) 50, (i : ℂ) ^ n

theorem sum_of_powers_of_i_equals_one :
  sum_of_powers_of_i = 1 := by
  sorry

end sum_of_powers_of_i_equals_one_l188_188811


namespace find_fraction_l188_188087

theorem find_fraction (N : ℝ) (hN : N = 24) : ∃ f : ℝ, N * f - 10 = 0.25 * N ∧ f = 2 / 3 :=
by
  use 2 / 3
  split
  { rw hN
    norm_num }
  { norm_num }

end find_fraction_l188_188087


namespace correlation_coefficient_one_l188_188937

variables {n : ℕ} (x y : Fin n → ℝ)

def line (x y: ℝ) : Prop := y = (1/2) * x + 1

theorem correlation_coefficient_one 
  (h: ∀ i, ∃ a, line (x i) (y i))
  (hne: ∃ i1 i2, i1 ≠ i2 ∧ x i1 ≠ x i2) : 
  correlation_coeff x y = 1 := 
sorry

end correlation_coefficient_one_l188_188937


namespace sum_geometric_series_l188_188887

variable (z : ℂ)

theorem sum_geometric_series
  (h : z + complex.I = 1 - z * complex.I) 
  : (finset.range 2019).sum (λ n, z ^ n) = - complex.I := 
sorry

end sum_geometric_series_l188_188887


namespace sin_cos_identity_l188_188558

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := 
by
  sorry

end sin_cos_identity_l188_188558


namespace simplified_expression_l188_188814

variable {x y : ℝ}

theorem simplified_expression 
  (P : ℝ := x^2 + y^2) 
  (Q : ℝ := x^2 - y^2) : 
  ( (P + 3 * Q) / (P - Q) - (P - 3 * Q) / (P + Q) ) = (2 * x^4 - y^4) / (x^2 * y^2) := 
  by sorry

end simplified_expression_l188_188814


namespace description_of_T_l188_188238

def T (x y : ℝ) : Prop :=
  ((x - 3 = 7 ∧ y + 5 ≥ 7) ∨ (y + 5 = 7 ∧ x - 3 ≥ 7) ∨ (x - 3 = y + 5 ∧ x - 3 ≥ 7))

theorem description_of_T :
  T = {p : ℝ × ℝ | (p.1 = 10 ∧ p.2 ≥ 2) ∨ (p.2 = 2 ∧ p.1 ≥ 10) ∨ (p.2 = p.1 - 8 ∧ p.1 ≥ 10)} :=
sorry

end description_of_T_l188_188238


namespace solve_complex_eq_l188_188831

theorem solve_complex_eq (z : ℂ) (h : z^2 = -100 - 64 * I) : z = 3.06 - 10.46 * I ∨ z = -3.06 + 10.46 * I :=
by
  sorry

end solve_complex_eq_l188_188831


namespace hyperbola_eccentricity_l188_188523

-- Definitions based on the conditions
def hyperbola (a b : ℝ) : Prop := (a > 0) ∧ (b > 0)

def distance_from_focus_to_asymptote (a b c : ℝ) : Prop :=
  (b^2 * c) / (a^2 + b^2).sqrt = b ∧ b = 2 * Real.sqrt 3

def minimum_distance_point_to_focus (a c : ℝ) : Prop :=
  c - a = 2

def eccentricity (a c e : ℝ) : Prop :=
  e = c / a

-- Problem statement
theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_hyperbola : hyperbola a b)
  (h_dist_asymptote : distance_from_focus_to_asymptote a b c)
  (h_min_dist_focus : minimum_distance_point_to_focus a c)
  (h_eccentricity : eccentricity a c e) :
  e = 2 :=
sorry

end hyperbola_eccentricity_l188_188523


namespace tennis_tournament_matches_l188_188929

theorem tennis_tournament_matches :
  let initial_tournament_players := 10
  let top_tournament_players := 5
  initial_tournament_players * (initial_tournament_players - 1) / 2 +
  top_tournament_players * (top_tournament_players - 1) / 2 = 55 :=
by
  let initial_matches := initial_tournament_players * (initial_tournament_players - 1) / 2
  let top_matches := top_tournament_players * (top_tournament_players - 1) / 2
  have total_matches := initial_matches + top_matches
  sorry

end tennis_tournament_matches_l188_188929


namespace find_m_max_min_FA_FB_l188_188496

-- Definitions of the conditions
def is_line (m t α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = m + t * (Real.cos α) ∧ y = t * (Real.sin α)

def is_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

def left_focus_ellipse (φ : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 * Real.cos φ ∧ y = Real.sqrt 3 * Real.sin φ

def focal_point := (-1 : ℝ, 0 : ℝ)

-- Proof statements
theorem find_m (m : ℝ) (h : is_line m 0 0) : m = -1 :=
sorry

theorem max_min_FA_FB (α : ℝ) (h : ∀ t, is_line (-1) t α → is_ellipse (m + t * Real.cos α) (t * Real.sin α))
  : (max (λ (F A B : (ℝ × ℝ)), ∃ t1 t2, |t1 * t2| = \frac{9}{3 * (Real.cos^2 α + 4 * Real.sin^2 α)} = 3) ∧ 
    min (λ (F A B : (ℝ × ℝ)), ∃ t1 t2, |t1 * t2| = \frac{9}{4}) := 
sorry

end find_m_max_min_FA_FB_l188_188496


namespace equal_ME_MF_l188_188407

-- Given definitions (conditions from the problem)
variables (A B C D P E F G M : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables [MetricSpace P] [MetricSpace E] [MetricSpace F] [MetricSpace G] [MetricSpace M]

-- Definitions of conditions
variable (inscribed : CyclicQuadrilateral A B C D)
variable (midpoint : P = Midpoint A B)
variable (perp_PE_AD : Perpendicular P E A D)
variable (perp_PF_BC : Perpendicular P F B C)
variable (perp_PG_CD : Perpendicular P G C D)
variable (intersection : IntersectionLine PG EF = M)

-- Goal to be proved (the question)
theorem equal_ME_MF : Distance M E = Distance M F :=
sorry -- proof omitted

end equal_ME_MF_l188_188407


namespace sqrt_equation_has_solution_l188_188060

noncomputable def x : ℝ := Real.sqrt (20 + x)

theorem sqrt_equation_has_solution : x = 5 :=
by
  sorry

end sqrt_equation_has_solution_l188_188060


namespace volume_of_circumscribed_sphere_of_regular_tetrahedron_is_correct_l188_188923

noncomputable def circumsphereVolume (edgeLength : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (sqrt (6 : ℝ) / 4) ^ 3

theorem volume_of_circumscribed_sphere_of_regular_tetrahedron_is_correct :
  circumsphereVolume 1 = (sqrt (6 : ℝ) / 8) * Real.pi :=
by 
  sorry

end volume_of_circumscribed_sphere_of_regular_tetrahedron_is_correct_l188_188923


namespace find_largest_number_l188_188696

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
  sorry

end find_largest_number_l188_188696


namespace James_age_l188_188964

-- Defining variables
variables (James John Tim : ℕ)
variables (h1 : James + 12 = John)
variables (h2 : Tim + 5 = 2 * John)
variables (h3 : Tim = 79)

-- Statement to prove James' age
theorem James_age : James = 25 :=
by {
  sorry
}

end James_age_l188_188964


namespace solve_by_completing_square_l188_188647

noncomputable def d : ℤ := -5
noncomputable def e : ℤ := 10

theorem solve_by_completing_square :
  ∃ d e : ℤ, (x^2 - 10 * x + 15 = 0 ↔ (x + d)^2 = e) ∧ (d + e = 5) :=
by
  use -5, 10
  split
  -- First part: Show the equivalence of equations
  sorry
  -- Second part: Show d + e = 5
  refl

end solve_by_completing_square_l188_188647


namespace pen_tip_movement_l188_188446

-- Definitions for the conditions
def condition_a := "Point movement becomes a line"
def condition_b := "Line movement becomes a surface"
def condition_c := "Surface movement becomes a solid"
def condition_d := "Intersection of surfaces results in a line"

-- The main statement we need to prove
theorem pen_tip_movement (phenomenon : String) : 
  phenomenon = "the pen tip quickly sliding on the paper to write the number 6" →
  condition_a = "Point movement becomes a line" :=
by
  intros
  sorry

end pen_tip_movement_l188_188446


namespace hyperbola_focal_length_l188_188454

theorem hyperbola_focal_length (m : ℝ) (a b : ℝ) (h_gt : 4 - m^2 > 0) (ha : a^2 = m^2 + 12) (hb : b^2 = 4 - m^2) :
  2 * real.sqrt (a^2 + b^2) = 8 :=
begin
  sorry
end

end hyperbola_focal_length_l188_188454


namespace area_inside_circle_outside_square_l188_188392

noncomputable def circle_square_area_difference : ℝ :=
  let side_length_square := 1
  let radius_circle : ℝ := real.sqrt 3 / 3
  let area_circle := π * (radius_circle ^ 2)
  let area_square := side_length_square ^ 2
  let area_difference := area_circle - area_square
  let segment_area := area_difference / 4
  let final_area := 4 * segment_area - side_length_square ^ 2
  final_area

theorem area_inside_circle_outside_square :
  circle_square_area_difference = (2 * π / 9) - (real.sqrt 3 / 3) :=
by
  sorry

end area_inside_circle_outside_square_l188_188392


namespace units_digit_of_F_F_15_l188_188665

def F : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := F (n + 1) + F n

theorem units_digit_of_F_F_15 : (F (F 15)) % 10 = 5 := by
  have h₁ : ∀ n, F n % 10 = ([0, 1, 1, 2, 3, 5, 8, 3, 1, 4, 5, 9, 4, 3, 7, 0, 7, 7, 4, 1, 5, 6, 1, 7, 8, 5, 3, 8, 1, 9, 0, 9, 9, 8, 7, 5, 2, 7, 9, 6, 5, 1, 6, 7, 3, 0, 3, 3, 6, 9, 5, 4, 9, 3, 2, 5, 7, 2, 9, 1].take 60)!!.get_or_else (n % 60) 0 := sorry
  have h₂ : F 15 = 610 := by
    simp [F]; -- Proof for F 15 = 610
    sorry
  calc (F (F 15)) % 10
       = F 610 % 10 := by
         rw [h₂]
       ... = 5 := h₁ 610

end units_digit_of_F_F_15_l188_188665


namespace point_in_second_quadrant_l188_188514

theorem point_in_second_quadrant (θ : ℝ) (h₁ : real.pi / 2 < θ) (h₂ : θ < real.pi) :
  ∃ q : ℕ, q = 2 ∧ (∃ P : ℝ × ℝ, P.1 = real.tan θ ∧ P.2 = real.sin θ ∧ P ∈ quadrant q) :=
begin
  sorry
end

end point_in_second_quadrant_l188_188514


namespace rectangle_overlap_l188_188935

theorem rectangle_overlap (large_rectangle_area : ℝ)
  (num_rectangles : ℕ)
  (individual_area : ℝ)
  (rectangles : fin num_rectangles → set (ℝ × ℝ))
  (h_large_area : large_rectangle_area = 5)
  (h_num_rectangles : num_rectangles = 9)
  (h_individual_area : individual_area = 1)
  (h_total_area : ∀ i, measure_theory.measure (rectangles i) = individual_area) :
  ∃ (i j : fin num_rectangles), i ≠ j ∧ measure_theory.measure (rectangles i ∩ rectangles j) ≥ 1/9 :=
begin
  sorry
end

end rectangle_overlap_l188_188935


namespace roots_sum_of_quadratic_l188_188197

theorem roots_sum_of_quadratic:
  (∃ a b : ℝ, (a ≠ b) ∧ (a * b = 5) ∧ (a + b = 8)) →
  (a + b = 8) :=
by
  sorry

end roots_sum_of_quadratic_l188_188197


namespace f_2007_value_l188_188608

def A : Set ℚ := { x | x ≠ 0 ∧ x ≠ 1 }

def f (x : ℚ) (hx : x ∈ A) : ℝ := sorry

theorem f_2007_value (h : ∀ x ∈ A, f x _ + f (1 - 1 / x) _ = Real.log (abs x)) :
  f 2007 (by { simp [A], norm_num }) = Real.log (2007 / 2006) :=
sorry

end f_2007_value_l188_188608


namespace greatest_natural_number_l188_188455

theorem greatest_natural_number (n q r : ℕ) (h1 : n = 91 * q + r)
  (h2 : r = q^2) (h3 : r < 91) : n = 900 :=
sorry

end greatest_natural_number_l188_188455


namespace magnitude_of_z_proof_l188_188859

noncomputable def magnitude_of_z (x y : ℝ) (z : ℂ) : ℝ := complex.abs z

theorem magnitude_of_z_proof
  (x y : ℝ)
  (z : ℂ)
  (h1 : z = x + y * complex.I)
  (h2 : -3 + 3 * complex.I = x + (y - 1) * complex.I) :
  magnitude_of_z x y z = 5 := by
srry

end magnitude_of_z_proof_l188_188859


namespace interval_monotonicity_f_maximum_k_l188_188896

section
variables (x : ℝ) (a k : ℝ)
noncomputable def f (x : ℝ) := (1/2) * x^2 + a * x + 2 * Real.log x
noncomputable def g (x : ℝ) := (1/2) * x^2 + k * x + (2 - x) * Real.log x - k

theorem interval_monotonicity_f {a : ℝ} (ha : a = -3) :
  (∀ x > 0, (f x ha) is increasing on (0, 1) ∪ (2, +∞)) ∧ 
  (∀ x > 0, (f x ha) is decreasing on (1, 2)) :=
sorry

theorem maximum_k (a = 1) :
  (∀ x > 1, g x < f x (1)) → k ≤ 3 :=
sorry
end

end interval_monotonicity_f_maximum_k_l188_188896


namespace inequality_solution_l188_188832

theorem inequality_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x ∈ Set.Ioo (-2 : ℝ) (-1) ∨ x ∈ Set.Ioi 2) ↔ 
  (∃ x : ℝ, (x^2 + x - 2) / (x + 2) ≥ (3 / (x - 2)) + (3 / 2)) := by
  sorry

end inequality_solution_l188_188832


namespace car_speed_first_hour_l188_188691

theorem car_speed_first_hour 
  (x : ℝ)  -- Speed of the car in the first hour.
  (s2 : ℝ)  -- Speed of the car in the second hour is fixed at 40 km/h.
  (avg_speed : ℝ)  -- Average speed over two hours is 65 km/h.
  (h1 : s2 = 40)  -- speed in the second hour is 40 km/h.
  (h2 : avg_speed = 65)  -- average speed is 65 km/h
  (h3 : avg_speed = (x + s2) / 2)  -- definition of average speed
  : x = 90 := 
  sorry

end car_speed_first_hour_l188_188691


namespace quadratic_polynomial_AT_BT_l188_188006

theorem quadratic_polynomial_AT_BT (p s : ℝ) :
  ∃ (AT BT : ℝ), (AT + BT = p + 3) ∧ (AT * BT = s^2) ∧ (∀ (x : ℝ), (x^2 - (p+3) * x + s^2) = (x - AT) * (x - BT)) := 
sorry

end quadratic_polynomial_AT_BT_l188_188006


namespace range_m_for_doubling_ln_l188_188196

def doubling_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ set.Icc a b, f x ∈ set.Icc (2 * a) (2 * b) ∧ strict_mono_on f (set.Icc a b)

def is_doubling_ln (m : ℝ) : Prop :=
  doubling_function (λ x, real.log (real.exp x + m)) (-real.sqrt 2 + 1) (real.sqrt 2 + 1)

theorem range_m_for_doubling_ln :
  ∀ m, is_doubling_ln m ↔ -1/4 < m ∧ m < 0 :=
by
  sorry

end range_m_for_doubling_ln_l188_188196


namespace cos_C_eq_neg_one_fourth_l188_188203

-- Given conditions
variables {A B C : Type}
variables [triangle_metric_space A B C]
variables {a b c : ℝ}
variables {k : ℝ}
variables (ratio : a = 2 * k ∧ b = 3 * k ∧ c = 4 * k)

-- Statement to prove
theorem cos_C_eq_neg_one_fourth (h : ratio) : ∃ k, cos (C) = - 1 / 4 :=
by
  sorry

end cos_C_eq_neg_one_fourth_l188_188203


namespace eval_expression_l188_188826

theorem eval_expression :
  sqrt ((16^10 + 8^10 + 2^30 : ℝ) / (16^4 + 8^11 + 2^20)) = 1 / 2 := by
  sorry

end eval_expression_l188_188826


namespace angle_bisector_theorem_proportion_l188_188926

/-- In triangle PQR, PS bisects ∠P and meets QR at S. Given QR = a, PR = 2b, and PQ = b,
    then the proportion x/y = 2, where x = SR and y = SQ. -/
theorem angle_bisector_theorem_proportion {a b x y : ℝ} 
    (h1 : QR = a) (h2 : PR = 2 * b) (h3 : PQ = b)
    (h_bisect : PS bisects ∠P)
    (h_x_y_sum : x + y = a) 
    (h_bisector : x / b = y / (2 * b)) : x / y = 2 :=
  by
  sorry

end angle_bisector_theorem_proportion_l188_188926


namespace silvia_shorter_route_l188_188973

theorem silvia_shorter_route :
  let jerry_distance := 3 + 4
  let silvia_distance := Real.sqrt (3^2 + 4^2)
  let percentage_reduction := ((jerry_distance - silvia_distance) / jerry_distance) * 100
  (28.5 ≤ percentage_reduction ∧ percentage_reduction < 30.5) →
  percentage_reduction = 30 := by
  intro h
  sorry

end silvia_shorter_route_l188_188973


namespace exist_v_numbers_l188_188507

theorem exist_v_numbers (u : Fin 5 -> ℝ) :
  ∃ v : Fin 5 → ℝ, (∀ i, u i - v i ∈ Set.Univ ∩ ℕ) ∧ 
  (∑ i j in Finset.Ico 0 5, if i < j then (v i - v j)^2 else 0) < 4 := 
sorry

end exist_v_numbers_l188_188507


namespace all_guests_know_each_other_l188_188323

-- Definitions for conditions
axiom Guest : Type
axiom Acquainted : Guest → Guest → Prop
axiom Mutual : ∀ g1 g2 : Guest, Acquainted g1 g2 → Acquainted g2 g1
axiom Equidistant : Guest → Guest → Prop
axiom MutualAcquaintance : ∀ g1 g2 : Guest, ∃ g3 : Guest, Acquainted g1 g3 ∧ Acquainted g2 g3

-- Statement to be proved
theorem all_guests_know_each_other (guests : set Guest) :
  ∀ g1 g2 : Guest, g1 ∈ guests → g2 ∈ guests → (∃ g3, Acquainted g1 g3 ∧ Acquainted g3 g2) → Acquainted g1 g2 := 
sorry

end all_guests_know_each_other_l188_188323


namespace fraction_sum_in_base10_l188_188040

noncomputable def base8_to_base10 (n : ℕ) : ℕ :=
  sorry -- a placeholder for the actual function

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  sorry -- a placeholder for the actual function

noncomputable def base7_to_base10 (n : ℕ) : ℕ :=
  sorry -- a placeholder for the actual function

noncomputable def base5_to_base10 (n : ℕ) : ℕ :=
  sorry -- a placeholder for the actual function

theorem fraction_sum_in_base10 :
  let n1 := base8_to_base10 254
  let d1 := base4_to_base10 16
  let n2 := base7_to_base10 232
  let d2 := base5_to_base10 34
  \((n1 / d1) + (n2 / d2) = 23.6\) :=
  sorry

end fraction_sum_in_base10_l188_188040


namespace wheat_price_rate_l188_188406

theorem wheat_price_rate (x : ℝ) :
  (30 * x + 20 * 14.25) * 1.25 = 50 * 15.75 → x = 11.50 :=
begin
  intro h,
  sorry
end

end wheat_price_rate_l188_188406


namespace recipes_needed_l188_188032

-- Defining the conditions
def number_students : ℕ := 120
def fraction_attending : ℝ := 0.60
def avg_cookies_per_student : ℕ := 3
def cookies_per_recipe : ℕ := 18

-- Expected number of students attending
def expected_attending : ℕ := (number_students * fraction_attending).toNat

-- Total cookies needed
def total_cookies_needed : ℕ := expected_attending * avg_cookies_per_student

-- Full recipes needed
def full_recipes_needed : ℕ := total_cookies_needed / cookies_per_recipe

-- The proof statement: We need 12 full recipes
theorem recipes_needed : full_recipes_needed = 12 := by
  sorry

end recipes_needed_l188_188032


namespace polynomial_sum_of_squares_l188_188284

-- Definition of a nowhere negative polynomial with real coefficients
def nowhere_negative (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ p.eval x

-- Theorem to be proven
theorem polynomial_sum_of_squares 
  (n : ℕ) (h1 : 0 < n) (p : Polynomial ℝ) 
  (h2 : nowhere_negative p) : 
 ∃ (q : Fin n → Polynomial ℝ), p = ∑ i, (q i) * (q i) := 
sorry

end polynomial_sum_of_squares_l188_188284


namespace exists_acute_triangle_l188_188508

-- Define the segments as a list of positive real numbers
variables (a b c d e : ℝ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0) (h3 : d > 0) (h4 : e > 0)
(h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e)

-- Conditions: Any three segments can form a triangle
variables (h_triangle_1 : a + b > c ∧ a + c > b ∧ b + c > a)
variables (h_triangle_2 : a + b > d ∧ a + d > b ∧ b + d > a)
variables (h_triangle_3 : a + b > e ∧ a + e > b ∧ b + e > a)
variables (h_triangle_4 : a + c > d ∧ a + d > c ∧ c + d > a)
variables (h_triangle_5 : a + c > e ∧ a + e > c ∧ c + e > a)
variables (h_triangle_6 : a + d > e ∧ a + e > d ∧ d + e > a)
variables (h_triangle_7 : b + c > d ∧ b + d > c ∧ c + d > b)
variables (h_triangle_8 : b + c > e ∧ b + e > c ∧ c + e > b)
variables (h_triangle_9 : b + d > e ∧ b + e > d ∧ d + e > b)
variables (h_triangle_10 : c + d > e ∧ c + e > d ∧ d + e > c)

-- Prove that there exists an acute-angled triangle 
theorem exists_acute_triangle : ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                                        (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                                        (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
                                        x + y > z ∧ x + z > y ∧ y + z > x ∧ 
                                        x^2 < y^2 + z^2 := 
sorry

end exists_acute_triangle_l188_188508


namespace nutritional_composition_l188_188951

variables (x y : ℝ)

-- Conditions:
-- 1. The carbohydrate content is 1.5 times that of the protein.
-- 2. The total content of carbohydrates, protein, and fat is 30 g.
def protein_content (x : ℝ) := x
def fat_content (y : ℝ) := y
def carb_content (x : ℝ) := 1.5 * x
def total_content_eq (x y : ℝ) := total_content_eq = 30

theorem nutritional_composition 
  (h1 : carb_content x = 1.5 * x)
  (h2 : protein_content x + fat_content y + carb_content x = 30) :
  (5/2) * x + y = 30 :=
sorry

end nutritional_composition_l188_188951


namespace seven_students_speaking_l188_188206

/-- There are 7 students including A and B. We need to select 4 students to speak, with 
at least one of A or B participating. The total number of different speaking orders is 720. -/
theorem seven_students_speaking {
  A B : Type
  n : ℕ
  (h1 : n = 7) (h2 : 4 ≤ n) (h3 : A ≠ B)
} : (∃ l : List Type, l.length = 4 ∧ (A ∈ l ∨ B ∈ l) ∧ l.nodup) → (List.permutations l).length = 720 :=
sorry

end seven_students_speaking_l188_188206


namespace triangle_right_iff_cos_half_angle_l188_188958

theorem triangle_right_iff_cos_half_angle (a b c A B C : ℝ) 
  (is_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (angles_sum : A + B + C = π)
  (side_relation : a = (2 * b * c * (1 - 2 * (cos(A / 2)) ^ 2)) ^ (1/2))
  (cos_half_angle_condition : (cos(A / 2)) ^ 2 = (b + c) / (2 * c)) :
  c^2 = a^2 + b^2 :=
sorry

end triangle_right_iff_cos_half_angle_l188_188958


namespace sin_14pi_over_5_eq_sin_36_degree_l188_188830

noncomputable def sin_14pi_over_5 : ℝ :=
  Real.sin (14 * Real.pi / 5)

noncomputable def sin_36_degree : ℝ :=
  Real.sin (36 * Real.pi / 180)

theorem sin_14pi_over_5_eq_sin_36_degree :
  sin_14pi_over_5 = sin_36_degree :=
sorry

end sin_14pi_over_5_eq_sin_36_degree_l188_188830


namespace gumballs_remaining_l188_188791

theorem gumballs_remaining (a b total eaten remaining : ℕ) 
  (hAlicia : a = 20) 
  (hPedro : b = a + 3 * a) 
  (hTotal : total = a + b) 
  (hEaten : eaten = 40 * total / 100) 
  (hRemaining : remaining = total - eaten) : 
  remaining = 60 := by
  sorry

end gumballs_remaining_l188_188791


namespace rainfall_difference_l188_188597

variable (M : ℕ)

def rainfall_sunday : ℕ := 4
def rainfall_monday : ℕ := M
def rainfall_tuesday : ℕ := 2 * M
def total_rainfall : ℕ := rainfall_sunday + rainfall_monday + rainfall_tuesday

theorem rainfall_difference (h : total_rainfall M = 25) : rainfall_monday M - rainfall_sunday = 3 :=
by
  sorry

end rainfall_difference_l188_188597


namespace sum_of_roots_l188_188845

-- Defining the polynomial
def P (x : ℝ) : ℝ :=
  (x - 1)^2010 + 2 * (x - 2)^2009 + 3 * (x - 3)^2008 + ∑ i in finset.range 2009, (i + 1) * (x - (i + 1))^(2010 - (i + 1))

-- Stating the theorem: Sum of the roots of the polynomial P(x)
theorem sum_of_roots : ∑ root in (P.roots), root = 2008 :=
  by
  -- proof would be here
  sorry

end sum_of_roots_l188_188845


namespace num_simple_pairs_1492_l188_188796

-- Define the concept of a "simple" pair
def is_simple_pair (m n : ℕ) : Prop :=
  ∀ i, let mi := (m / 10^i) % 10 in  -- i-th digit of m (from the right)
         let ni := (n / 10^i) % 10 in  -- i-th digit of n (from the right)
         mi + ni < 10

-- Define the condition that the sum is 1492
def sum_1492 (m n : ℕ) : Prop := m + n = 1492

-- The main theorem
theorem num_simple_pairs_1492 : 
  ∃ count : ℕ, count = 300 ∧ ∀ m n : ℕ, is_simple_pair m n ∧ sum_1492 m n → true :=
sorry

end num_simple_pairs_1492_l188_188796


namespace fifth_bowler_points_l188_188378

variable (P1 P2 P3 P4 P5 : ℝ)
variable (h1 : P1 = (5 / 12) * P3)
variable (h2 : P2 = (5 / 3) * P3)
variable (h3 : P4 = (5 / 3) * P3)
variable (h4 : P5 = (50 / 27) * P3)
variable (h5 : P3 ≤ 500)
variable (total_points : P1 + P2 + P3 + P4 + P5 = 2000)

theorem fifth_bowler_points : P5 = 561 :=
  sorry

end fifth_bowler_points_l188_188378


namespace Carolina_spent_correct_amount_l188_188806

theorem Carolina_spent_correct_amount :
  let cost_letter := 0.37
  let cost_package := 0.88
  let letters := 5
  let packages := letters - 2
  let total_cost := (letters * cost_letter) + (packages * cost_package)
  total_cost = 4.49 :=
by
  let cost_letter := 0.37
  let cost_package := 0.88
  let letters := 5
  let packages := letters - 2
  let total_cost := (letters * cost_letter) + (packages * cost_package)
  have : total_cost = 4.49 := by sorry
  exact this

end Carolina_spent_correct_amount_l188_188806


namespace common_difference_value_l188_188656

noncomputable def arithmetic_sequence : ℕ → ℚ
| n => b1 + (n - 1) * d

-- Conditions
axiom sum_first_50_terms : (∑ n in finset.range 50, arithmetic_sequence n.succ) = 150
axiom sum_last_50_terms : (∑ n in finset.range 50, arithmetic_sequence (n + 51)) = 300

theorem common_difference_value : d = 3 / 50 :=
by 
  sorry

end common_difference_value_l188_188656


namespace eval_expression_l188_188561

theorem eval_expression (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m - 3 = -1 :=
by
  sorry

end eval_expression_l188_188561


namespace side_length_square_base_l188_188386

theorem side_length_square_base 
  (height : ℕ) (volume : ℕ) (A : ℕ) (s : ℕ) 
  (h_height : height = 8) 
  (h_volume : volume = 288) 
  (h_base_area : A = volume / height) 
  (h_square_base : A = s ^ 2) :
  s = 6 :=
by
  sorry

end side_length_square_base_l188_188386


namespace hexagon_segments_same_length_probability_l188_188983

theorem hexagon_segments_same_length_probability :
  let S := {s : ℝ // s = side ∨ s = diagonal}
  (length S = 15) ∧
  (∃! side ∈ S, side_length side = base_length) ∧ 
  (∃! diagonal ∈ S, diagonal_length diagonal = base_diagonal_length) ∧ 
  (6 * base_length + 9 * base_diagonal_length = 15 * side_length) →
  prob_of_same_length (S: set ℝ) = 17 / 35 := by
  sorry

end hexagon_segments_same_length_probability_l188_188983


namespace no_four_binomial_coeffs_in_arithmetic_progression_l188_188643

theorem no_four_binomial_coeffs_in_arithmetic_progression 
  (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (h : m + 3 ≤ n) :
  ¬ (2 * (nat.choose n (m + 1)) = (nat.choose n m) + (nat.choose n (m + 2)) ∧
     2 * (nat.choose n (m + 2)) = (nat.choose n (m + 1)) + (nat.choose n (m + 3))) :=
  by sorry

end no_four_binomial_coeffs_in_arithmetic_progression_l188_188643


namespace reassignment_possible_iff_impossible_l188_188209

theorem reassignment_possible_iff_impossible :
  ∃ (rows columns : ℕ) (students chairs : ℕ), 
  rows = 5 ∧ columns = 7 ∧ students = 34 ∧ chairs = rows * columns - 1 ∧ 
  (∀ student_position : ℕ × ℕ, student_position.1 < rows ∧ student_position.2 < columns ∧ 
  student_position ≠ (2, 3) ∧ (student_position.1 = 0 ∨ student_position.1 = rows - 1 ∨
  student_position.2 = 0 ∨ student_position.2 = columns - 1) → false) :=
begin
  -- statement confirmation
  existsi (5, 7, 34, 34),
  simp,
  split,
  exact (dec_trivial : 5 = 5),
  split,
  exact (dec_trivial : 7 = 7),
  split,
  exact (dec_trivial : 34 = 34),
  split,
  exact (dec_trivial : 34 = 5 * 7 - 1),
  intros sp hsp,
  rcases hsp with ⟨hsp1, hsp2, hsp3, hsp4⟩,
  cases sp with srow scol,
  simp at *,
  rcases hsp4 with hsp4 | hsp4 | hsp4 | hsp4;
  linarith,
end

end reassignment_possible_iff_impossible_l188_188209


namespace factorize_difference_of_squares_l188_188447

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) :=
by
  sorry

end factorize_difference_of_squares_l188_188447


namespace problem_statement_l188_188979

-- Define the problem conditions
variables {α : Type*} [linear_ordered_field α]

-- Define the sequence α_i
def alpha (i : ℕ) (α : ℕ → α) : α := α i

-- Formulate the problem statement in Lean 4
theorem problem_statement (α : ℕ → ℝ) :
  let S := ∑ i in finset.range 2008, real.sin (α i) * real.cos (α ((i + 1) % 2008)) in
  S ≤ 1004 :=
sorry

end problem_statement_l188_188979


namespace find_BC_length_l188_188204

/-- Given a triangle ABC with D and E as midpoints of sides AB and AC respectively,
    such that CD and BE intersect at P with ∠BPC = 90°, and given BD = 1829 and CE = 1298,
    we are to prove that the length of BC is 2006. -/
theorem find_BC_length
  (A B C D E P : Point) -- Points involved in the triangle and midpoints
  (h1 : midpoint A B D) -- D is the midpoint of AB
  (h2 : midpoint A C E) -- E is the midpoint of AC
  (h3 : intersect CD BE P) -- CD and BE intersect at P
  (h4 : angle B P C = 90) -- angle BPC is 90 degrees
  (h5 : length B D = 1829) -- BD is 1829 units
  (h6 : length C E = 1298) -- CE is 1298 units
  : length B C = 2006 := 
sorry

end find_BC_length_l188_188204


namespace annuity_duration_l188_188383

def principal : ℝ := 5000
def interest_rate : ℝ := 0.045
def initial_years : ℕ := 26
def delay : ℕ := 3

theorem annuity_duration : ∃ (n : ℕ), 
  (n = initial_years + delay + ((log ((1 + interest_rate) ^ initial_years - 1) -
  log ((1 + interest_rate) ^ initial_years)) / log (1 + interest_rate))) :=
by
  sorry

end annuity_duration_l188_188383


namespace jack_jog_speed_l188_188223

theorem jack_jog_speed (distance_miles : ℝ) (time_hours : ℝ) (blocks : ℝ) (block_distance : ℝ): 
  (distance_miles = (block_distance * blocks)) →
  (block_distance = 1/8) →
  (blocks = 16) →
  (time_hours = 10 / 60) →
  (distance_miles / time_hours = 12) :=
begin
  intros h1 h2 h3 h4,
  rw [h2, h3] at h1,   
  rw ←h1,
  rw h4,
  norm_num,
  sorry,
end

end jack_jog_speed_l188_188223


namespace max_elements_in_A_l188_188991

noncomputable def max_elements_A (M : Set ℕ) (A : Set ℕ) : ℕ :=
if h : (∀ x ∈ A, x ∈ M) ∧ (∀ x ∈ A, 15 * x ∉ A) then 
  1970 - 125 
else 0

theorem max_elements_in_A : max_elements_A (Set.Icc 1 1995) A = 1870 :=
begin
  sorry
end

end max_elements_in_A_l188_188991


namespace equilateral_triangle_division_l188_188869

theorem equilateral_triangle_division :
  ∀ (T : Triangle), 
    (is_equilateral T) → 
    (∃ (P₁ P₂ : Polygon), 
      (num_sides P₁ = 2020) ∧ 
      (num_sides P₂ = 2021) ∧ 
      (is_partition_of_triangle T P₁ P₂)) := 
by
  sorry

end equilateral_triangle_division_l188_188869


namespace tangent_line_eqn_at_e_l188_188300

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def point : ℝ × ℝ := (Real.exp 1, f (Real.exp 1))

theorem tangent_line_eqn_at_e :
  ∀ (x y : ℝ), y - point.2 = (deriv f (point.1)) * (x - point.1) ↔ 2 * x - y - point.1 = 0 :=
by
  sorry

end tangent_line_eqn_at_e_l188_188300


namespace nested_sqrt_solution_l188_188072

noncomputable def nested_sqrt (x : ℝ) : ℝ := sqrt (20 + x)

theorem nested_sqrt_solution (x : ℝ) : nonneg_real x →
  (x = nested_sqrt x ↔ x = 5) :=
begin
  sorry
end

end nested_sqrt_solution_l188_188072


namespace evaluate_sum_expression_l188_188084

theorem evaluate_sum_expression : 
  (12 * exp (3 * real.pi * complex.I / 13) + 12 * exp (20 * real.pi * complex.I / 26)) = 
  (24 * real.cos (7 * real.pi / 26) * complex.I) := 
by
  sorry

end evaluate_sum_expression_l188_188084


namespace laptop_cost_l188_188293

theorem laptop_cost
  (C : ℝ) (down_payment := 0.2 * C + 20) (installments_paid := 65 * 4) (balance_after_4_months := 520)
  (h : C - (down_payment + installments_paid) = balance_after_4_months) :
  C = 1000 :=
by
  sorry

end laptop_cost_l188_188293


namespace intersection_P_Q_l188_188904

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l188_188904


namespace max_b_no_lattice_point_l188_188423

theorem max_b_no_lattice_point (b : ℚ) (m : ℚ) (h1 : 1 / 3 < m) (h2 : m < b) 
(h3 : ∀ (x : ℤ), 0 < x ∧ x ≤ 200 → ∀ (y : ℤ), y ≠ m * (x:ℚ) + 3) : b = 67 / 199 :=
sorry

end max_b_no_lattice_point_l188_188423


namespace max_area_right_triangle_in_semicircle_l188_188567

theorem max_area_right_triangle_in_semicircle :
  ∀ (r : ℝ), r = 1/2 → 
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ y > 0 ∧ 
  (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 ∧ y' > 0 → (1/2) * x * y ≥ (1/2) * x' * y') ∧ 
  (1/2) * x * y = 3 * Real.sqrt 3 / 32 := 
sorry

end max_area_right_triangle_in_semicircle_l188_188567


namespace area_of_ADC_l188_188220

noncomputable def area_of_triangle_ALT (base height : ℝ) : ℝ :=
  0.5 * base * height

structure TriangleData where
  BD DC ABD_area : ℝ
  BD_DC_ratio : BD / DC = 3 / 2

theorem area_of_ADC (data : TriangleData) : 
  data.BD / data.DC = 3 / 2 → 
  data.ABD_area = 27 → 
  area_of_triangle_ALT data.DC height = 18 :=
by
  sorry

end area_of_ADC_l188_188220


namespace number_of_boys_is_810_l188_188695

theorem number_of_boys_is_810 (B G : ℕ) (h1 : B + G = 900) (h2 : G = B / 900 * 100) : B = 810 :=
by
  sorry

end number_of_boys_is_810_l188_188695


namespace circle_equation_line_exists_l188_188125

-- Definitions and conditions
def is_center (c : ℝ × ℝ) : Prop := c.1 = c.2 ∧ c.2 < 0
def radius := 3
def chord_length := 2 * Real.sqrt 5

-- (Ⅰ) Prove the equation of the circle
theorem circle_equation (c : ℝ × ℝ) (h_center : is_center c) 
  (h_radius : (c.1 + 2)^2 + (c.2 + 2)^2 = radius^2) : 
  ∃ a b : ℝ, (c.1 = a ∧ c.2 = b) ∧ ((a + 2)^2 + (b + 2)^2 = 9) := sorry

-- (Ⅱ) Prove the existence of the line and find its equations
theorem line_exists (c : ℝ × ℝ) (h_center : is_center c) 
  (h_radius : (c.1 + 2)^2 + (c.2 + 2)^2 = radius^2) : 
  ∃ b : ℝ, (b = 1 ∨ b = -1) ∧ 
  ((8 + 2*b)^2 - 4*2*(b^2 + 4*b - 1) > 0) := sorry

end circle_equation_line_exists_l188_188125


namespace greatest_possible_individual_award_l188_188778

noncomputable def prize : ℝ := 2000
noncomputable def total_winners : ℕ := 50
noncomputable def min_award : ℝ := 25
noncomputable def fraction_of_prize : ℝ := 3 / 4
noncomputable def fraction_of_winners : ℝ := 2 / 5

theorem greatest_possible_individual_award : 
  let prize_for_fraction := fraction_of_prize * prize,
      winners_for_fraction := fraction_of_winners * total_winners,
      remaining_prize := prize - prize_for_fraction,
      remaining_winners := total_winners - winners_for_fraction,
      min_total_amount := remaining_winners * min_award,
      redistributed_amount := min_total_amount - remaining_prize,
      new_prize_for_fraction := prize_for_fraction - redistributed_amount,
      remaining_fraction_winners := winners_for_fraction - 1,
      max_individual_award := new_prize_for_fraction - (remaining_fraction_winners * min_award)
  in max_individual_award = 775 := 
begin 
  sorry
end

end greatest_possible_individual_award_l188_188778


namespace two_polygons_sum_of_interior_angles_l188_188693

theorem two_polygons_sum_of_interior_angles
  (n1 n2 : ℕ) (h1 : Even n1) (h2 : Even n2) 
  (h_sum : (n1 - 2) * 180 + (n2 - 2) * 180 = 1800):
  (n1 = 4 ∧ n2 = 10) ∨ (n1 = 6 ∧ n2 = 8) :=
by
  sorry

end two_polygons_sum_of_interior_angles_l188_188693


namespace find_loss_percentage_l188_188011

theorem find_loss_percentage (W : ℝ) (profit_percentage : ℝ) (remaining_percentage : ℝ)
  (overall_loss : ℝ) (stock_worth : ℝ) (L : ℝ) :
  W = 12499.99 →
  profit_percentage = 0.20 →
  remaining_percentage = 0.80 →
  overall_loss = -500 →
  0.04 * W - (L / 100) * (remaining_percentage * W) = overall_loss →
  L = 10 :=
by
  intro hW hprofit_percentage hremaining_percentage hoverall_loss heq
  -- We'll provide the proof here
  sorry

end find_loss_percentage_l188_188011


namespace binary_multiplication_correct_l188_188457

def bin_mul (a b : ℕ) : ℕ :=
  let rec helper x y acc :=
    if y = 0 then acc
    else helper (x * 2) (y / 2) (if y % 2 = 1 then acc + x else acc)
  in helper a b 0

theorem binary_multiplication_correct : bin_mul 0b1101 0b0110 = 0b1011110 := 
by
  sorry

end binary_multiplication_correct_l188_188457


namespace part_a_part_b_l188_188858

def circle (O : Point) (radius : ℝ) := {P : Point | dist O P = radius}

variable (O A B C : Point)
variable (line_l : Line)
variable (radius a : ℝ)

-- Part (a)
theorem part_a : ∃ X : Point, X ∈ circle O radius ∧ 
  ∃ M N : Point, M ∈ line_l ∧ N ∈ line_l ∧
  collinear [A, X, M] ∧ collinear [B, X, N] ∧ dist M N = a :=
sorry

-- Part (b)
theorem part_b : ∃ X : Point, X ∈ circle O radius ∧ 
  ∃ M N : Point, M ∈ line_l ∧ N ∈ line_l ∧
  collinear [A, X, M] ∧ collinear [B, X, N] ∧
  dist M C = dist C N :=
sorry

end part_a_part_b_l188_188858


namespace simplify_expression_l188_188889

theorem simplify_expression (k : ℤ) : 
  2^(-(2*k + 1)) + 2^(-(2*k - 1)) - 2^(-2*k) + 2^(-(2*k + 2)) = (7/4) * 2^(-2*k) :=
by sorry

end simplify_expression_l188_188889


namespace alternating_binomial_sum_l188_188083

theorem alternating_binomial_sum :
  \sum_{k=0}^{50} (-1:ℤ)^k * (nat.choose 50 k) = 0 := by
  sorry

end alternating_binomial_sum_l188_188083


namespace mapping_element_l188_188497

def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

def f (p : A) : B := (p.1 - p.2, p.1 + p.2)

theorem mapping_element (p : A) (h : p = (1, 3)) : f p = (-2, 4) :=
  by
  rw [h]
  rfl

end mapping_element_l188_188497


namespace number_of_irrationals_in_list_l188_188026

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def count_irrationals (l : List ℝ) : ℕ :=
  l.filter is_irrational |> List.length

theorem number_of_irrationals_in_list :
  count_irrationals [
    0,
    Real.pi / 2,
    -1 / 5,
    3.1415926,
    22 / 7,
    1.53,
    -- Representation of 1.01001000100001... would typically be more elaborated
    1.01001000100001...
  ] = 2 := by {
    sorry
  }

end number_of_irrationals_in_list_l188_188026


namespace average_percentage_increase_l188_188598

theorem average_percentage_increase (initial_A raised_A initial_B raised_B initial_C raised_C : ℕ) 
(hA: initial_A = 50) (hRA: raised_A = 75) 
(hB: initial_B = 100) (hRB: raised_B = 120)
(hC: initial_C = 80) (hRC: raised_C = 88) : 
  ((raised_A - initial_A) * 100 / initial_A + (raised_B - initial_B) * 100 / initial_B + (raised_C - initial_C) * 100 / initial_C) / 3 = 26.67 := 
by 
  sorry

end average_percentage_increase_l188_188598


namespace not_exists_polynomials_for_first_identity_exists_polynomials_for_second_identity_l188_188745

-- Problem 1: Proving the non-existence
theorem not_exists_polynomials_for_first_identity : 
  ¬ ∃ P Q R : ℚ[x, y, z], 
    ((x - y + 1) ^ 3) * P + ((y - z - 1) ^ 3) * Q + ((z - 2 * x + 1) ^ 3) * R = 1 := 
sorry

-- Problem 2: Proving the existence
theorem exists_polynomials_for_second_identity : 
  ∃ P Q R : ℚ[x, y, z], 
    ((x - y + 1) ^ 3) * P + ((y - z - 1) ^ 3) * Q + ((z - x + 1) ^ 3) * R = 1 := 
sorry

end not_exists_polynomials_for_first_identity_exists_polynomials_for_second_identity_l188_188745


namespace log_diff_l188_188056

/-
Question: Evaluate log₅(625) - log₅(1/25)
Conditions:
  625 = 5^4
  1/25 = 5⁻²
Correct Answer: 6
-/

theorem log_diff (log5 : ℕ → ℕ) (h1 : 625 = 5 ^ 4) (h2 : 1 / 25 = 5 ^ -2) : 
  log5 625 - log5 (1 / 25) = 6 := 
sorry

end log_diff_l188_188056


namespace prob_return_to_freezer_l188_188031

-- Define the probabilities of picking two pops of each flavor
def probability_same_flavor (total: ℕ) (pop1: ℕ) (pop2: ℕ) : ℚ :=
  (pop1 * pop2) / (total * (total - 1))

-- Definitions according to the problem conditions
def cherry_pops : ℕ := 4
def orange_pops : ℕ := 3
def lemon_lime_pops : ℕ := 4
def total_pops : ℕ := cherry_pops + orange_pops + lemon_lime_pops

-- Calculate the probability of picking two ice pops of the same flavor
def prob_cherry : ℚ := probability_same_flavor total_pops cherry_pops (cherry_pops - 1)
def prob_orange : ℚ := probability_same_flavor total_pops orange_pops (orange_pops - 1)
def prob_lemon_lime : ℚ := probability_same_flavor total_pops lemon_lime_pops (lemon_lime_pops - 1)

def prob_same_flavor : ℚ := prob_cherry + prob_orange + prob_lemon_lime
def prob_diff_flavor : ℚ := 1 - prob_same_flavor

-- Theorem stating the probability of needing to return to the freezer
theorem prob_return_to_freezer : prob_diff_flavor = 8 / 11 := by
  sorry

end prob_return_to_freezer_l188_188031


namespace cardinality_of_M_l188_188544

open Classical

noncomputable def M := {x : ℕ | 8 - x ∈ ℕ}

theorem cardinality_of_M : (set.to_finset M).card = 9 :=
by
  sorry

end cardinality_of_M_l188_188544


namespace exists_right_triangle_int_sides_sqrt2016_l188_188439

theorem exists_right_triangle_int_sides_sqrt2016 : 
  ∃ a b c : ℤ, (a^2 + b^2 = c^2) ∧ (a = sqrt 2016 ∨ b = sqrt 2016 ∨ c = sqrt 2016)  :=
by
  sorry

end exists_right_triangle_int_sides_sqrt2016_l188_188439


namespace compare_a_b_l188_188174

noncomputable def a : ℝ := log 2 3 + 1
noncomputable def b : ℝ := log 2 14 - 1

theorem compare_a_b : b > a := by
  sorry

end compare_a_b_l188_188174


namespace equidistant_point_l188_188453

structure Point3D (α : Type _) := 
(x : α) (y : α) (z : α)

def distance {α : Type _} [LinearOrderedField α] (p1 p2 : Point3D α) : α := 
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

theorem equidistant_point :
  let A := Point3D.mk 0 0 (-1)
  let B := Point3D.mk 3 1 3
  let C := Point3D.mk 1 4 2
  distance A B = distance A C :=
by
  sorry

end equidistant_point_l188_188453


namespace triangle_to_polygons_l188_188866

-- Define what it means for a polygon to be an n-gon
def is_ngon (n : ℕ) (P : set Point) : Prop :=
  P.card = n

-- Define the statement: A triangle can be divided into a 2020-gon and a 2021-gon.
theorem triangle_to_polygons (T : set Point) (hT : is_triangle T):
  ∃ P Q : set Point, is_ngon 2020 P ∧ is_ngon 2021 Q ∧ (P ∪ Q = T) :=
sorry

end triangle_to_polygons_l188_188866


namespace total_candy_eaten_by_bobby_l188_188036

-- Definitions based on the problem conditions
def candy_eaten_by_bobby_round1 : ℕ := 28
def candy_eaten_by_bobby_round2 : ℕ := 42
def chocolate_eaten_by_bobby : ℕ := 63

-- Define the statement to prove
theorem total_candy_eaten_by_bobby : 
  candy_eaten_by_bobby_round1 + candy_eaten_by_bobby_round2 + chocolate_eaten_by_bobby = 133 :=
  by
  -- Skipping the proof itself
  sorry

end total_candy_eaten_by_bobby_l188_188036


namespace unique_polynomial_l188_188093

def satisfies_condition (P : ℝ[X]) : Prop :=
  ∀ x : ℝ, P.eval (2 * x) = (polynomial.derivative P).eval x * (polynomial.derivative (polynomial.derivative P)).eval x

theorem unique_polynomial {P : ℝ[X]} (h : satisfies_condition P) : P = (4 / 9) • polynomial.X ^ 3 :=
by
  sorry

end unique_polynomial_l188_188093


namespace cost_per_page_first_time_l188_188687

theorem cost_per_page_first_time (x : ℕ) :
  (100 : ℕ) = 100 ∧
  (20 * 5 + 30 * 2 * 5) = 400 ∧
  100 * x + 400 = 1400 →
  x = 10 :=
by
  intro h
  cases h with h_1 h_2
  cases h_2 with h_2 h_3
  -- proof omitted
  sorry

end cost_per_page_first_time_l188_188687


namespace three_pair_probability_l188_188710

theorem three_pair_probability :
  let total_combinations := Nat.choose 52 5
  let three_pair_combinations := 13 * 4 * 12 * 4
  total_combinations = 2598960 ∧ three_pair_combinations = 2496 →
  (three_pair_combinations : ℚ) / total_combinations = 2496 / 2598960 :=
by
  -- Definitions and computations can be added here if necessary
  sorry

end three_pair_probability_l188_188710


namespace fill_pool_time_l188_188380

theorem fill_pool_time :
  let pool_capacity := 36000 in
  let hose_count := 6 in
  let flow_rate_per_hose := 3 in
  let flow_rate_per_minute := hose_count * flow_rate_per_hose in
  let flow_rate_per_hour := flow_rate_per_minute * 60 in
  let fill_time := pool_capacity / flow_rate_per_hour in
  fill_time = 100 / 3 := sorry

end fill_pool_time_l188_188380
