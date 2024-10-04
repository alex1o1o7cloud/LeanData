import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Orthogonality.Complex
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialNumberTheory
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Peirce
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Num.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.Time.Clock
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Plane
import Mathlib.Geometry.euclidean.Basic
import Mathlib.Init.Data.Real.Basic
import Mathlib.Probability.ProbabilityDensityFunction
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith

namespace least_N_with_shared_digits_l142_142093

theorem least_N_with_shared_digits :
  ∃ (N : ℕ), (N ≥ 2) ∧ (∀ (arrangement : List ℕ), arrangement.perm (List.range (N+1).tail) →
  (∀ i, i < N → (∃ d ∈ (digits 10 (arrangement.nthLe i (sorry) : ℕ) ∩ 
  digits 10 (arrangement.nthLe (i+1 % N) (sorry)) : finset ℕ)))) ∧ N = 29
  :=
sorry

end least_N_with_shared_digits_l142_142093


namespace train_passing_tree_time_l142_142550

theorem train_passing_tree_time
  (l : ℝ) (v_kmh : ℝ) (convert : ℝ) (time : ℝ) :
  l = 140 →
  v_kmh = 63 →
  convert = 5 / 18 →
  time = l / (v_kmh * convert) →
  time = 40 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith
  -- Expected proof follows from the given steps, here we skip actual proof by sorry
  sorry

end train_passing_tree_time_l142_142550


namespace inequality_solution_set_l142_142100

theorem inequality_solution_set (x : ℝ) : 
  ((x + 1) / x ≤ 3 ↔ x ∈ set.Iio 0 ∪ set.Ici (1 / 2)) := by
sorry

end inequality_solution_set_l142_142100


namespace math_proof_problem_l142_142036

theorem math_proof_problem (B2 B4 A x : ℝ) :
  (B2 = B4) →
  (B2 = B2^2 - 2) →
  (B2 = A^2 - 2) →
  (x > 0 → x + 1/x ≥ 2) →
  (x < 0 → x + 1/x < 0) →
  (B2 = -1 → A = ±1 → ¬ (x + 1/x = 1)) →
  (B2 = 2 → A = ±2 → x = ±1) :=
by
  intros hB2eqB4 hB2eq hB2A2 hPos hNeg hImpossible h2 _ _ _ _ _
  sorry

end math_proof_problem_l142_142036


namespace selection_ways_3_people_5x5_matrix_not_same_row_col_l142_142565

theorem selection_ways_3_people_5x5_matrix_not_same_row_col :
  let n := 5 in let ways := 10 * 5 * 4 * 3 in ways = 600 :=
by
  sorry

end selection_ways_3_people_5x5_matrix_not_same_row_col_l142_142565


namespace select_2n_rectangles_l142_142963

-- Definitions of the segments
def segment_partition (n : ℕ) (h : n^2 ≥ 4) : Prop :=
  ∃ (a b : fin n → ℝ), (∀ i, 0 < a i) ∧ (∀ j, 0 < b j) ∧ 
  (finset.univ.sum a = finset.univ.sum b)

-- Definitions of the rectangles
def rectangle (a b : fin n → ℝ) (i j : fin n) : Prop :=
  (0 < a i) ∧ (0 < b j)

-- Condition: Rectangles can be nested or rotated to nest
def nested (a b : fin n → ℝ) (i1 j1 i2 j2 : fin n) : Prop :=
  (a i1 ≤ a i2 ∧ b j1 ≤ b j2) ∨ (a i2 ≤ a i1 ∧ b j2 ≤ b j1) ∨ 
  (a i1 ≤ b j2 ∧ b j1 ≤ a i2) ∨ (b j2 ≤ a i1 ∧ a i2 ≤ b j1)

-- Main proof statement
theorem select_2n_rectangles (n : ℕ) (h : n^2 ≥ 4)
  (a b : fin n → ℝ) (ha : ∀ i, 0 < a i) (hb : ∀ j, 0 < b j)
  (hab_sum : finset.univ.sum a = finset.univ.sum b) :
  ∃ (I J : finset (fin n × fin n)),
  (I.card = 2 * n) ∧ (∀ (ij1 ij2 : fin n × fin n), 
    ij1 ∈ I → ij2 ∈ I → 
      nested a b ij1.fst ij1.snd ij2.fst ij2.snd) :=
by
  sorry

end select_2n_rectangles_l142_142963


namespace find_f_at_2014_l142_142563

noncomputable def f : ℤ → ℤ := sorry

-- Conditions
axiom odd_function : ∀ x, f(-x) = -f(x)
axiom periodic_function : ∀ x, f(x + 3) = f(x)
axiom f_at_2 : f(2) = 1

-- Statement to prove
theorem find_f_at_2014 : f(2014) = -1 := by
  sorry

end find_f_at_2014_l142_142563


namespace remainder_of_product_l142_142988

theorem remainder_of_product (n : ℕ) (m : ℕ) : ((67545 * 11) % 13) = 11 :=
by
  have h1 : (67545 % 13) = 1,
  sorry,
  have h2 : ( (67545 * 11) % 13 ) = ( (1 * 11) % 13 ),
  sorry,
  have h3 : ( (1 * 11) % 13 ) = 11,
  sorry,
  exact h3

end remainder_of_product_l142_142988


namespace circle_equation_l142_142065

theorem circle_equation (a b R x y : ℝ) :
  (x - a)^2 + (y - b)^2 = R^2 ↔ sqrt((x - a)^2 + (y - b)^2) = R :=
by
  sorry

end circle_equation_l142_142065


namespace find_b_l142_142763

-- Define the given conditions
variable (a b : ℝ)
variable (A B : Real.Angle)
variable (A_is_30 : A = 30)
variable (B_is_45 : B = 45)
variable (a_is_2 : a = 2)
variable (sin_A : Real.sin (A.toRad) = 1 / 2)
variable (sin_B : Real.sin (B.toRad) = Real.sqrt 2 / 2)

-- State the theorem to be proven
theorem find_b 
  (A_is_30 : A = 30)
  (B_is_45 : B = 45)
  (a_is_2 : a = 2)
  (sin_A : Real.sin (A.toRad) = 1 / 2)
  (sin_B : Real.sin (B.toRad) = Real.sqrt 2 / 2)
: b = 2 * Real.sqrt 2 := 
sorry

end find_b_l142_142763


namespace radius_smaller_circle_l142_142178

theorem radius_smaller_circle (A₁ A₂ A₃ : ℝ) (s : ℝ)
  (h1 : A₁ + A₂ = 12 * Real.pi)
  (h2 : A₃ = (Real.sqrt 3 / 4) * s^2)
  (h3 : 2 * A₂ = A₁ + A₁ + A₂ + A₃) :
  ∃ r : ℝ, r = Real.sqrt (6 - (Real.sqrt 3 / 8) * s^2) := by
  sorry

end radius_smaller_circle_l142_142178


namespace youtube_dislikes_l142_142567

def initial_dislikes (likes : ℕ) : ℕ := (likes / 2) + 100

def new_dislikes (initial : ℕ) : ℕ := initial + 1000

theorem youtube_dislikes
  (likes : ℕ)
  (h_likes : likes = 3000) :
  new_dislikes (initial_dislikes likes) = 2600 :=
by
  sorry

end youtube_dislikes_l142_142567


namespace number_of_distinct_integer_sums_of_special_fractions_l142_142984

def is_special_fraction (a b : ℕ) : Prop := a + b = 17

def is_integer (x : ℚ) : Prop := ∃ k : ℤ, x = k

def is_sum_of_special_fractions (x : ℚ) : Prop :=
  ∃ (a₁ a₂ b₁ b₂ : ℕ), is_special_fraction a₁ b₁ ∧ is_special_fraction a₂ b₂ ∧ x = (a₁ / b₁ + a₂ / b₂)

theorem number_of_distinct_integer_sums_of_special_fractions : 
  (set_of (λ x, is_integer x ∧ is_sum_of_special_fractions x)).to_finset.card = 2 := 
sorry

end number_of_distinct_integer_sums_of_special_fractions_l142_142984


namespace diagonal_of_square_on_circle_and_tangent_l142_142161

theorem diagonal_of_square_on_circle_and_tangent (R : ℝ) : 
  ∃ d : ℝ, ∀ (x : ℝ), (x ≥ 0) → 
  (∃ (A B C D : Point), 
    A ∈ Circle O R ∧ B ∈ Circle O R ∧ 
    (tangent Circle O R).contains C ∧ (tangent Circle O R).contains D ∧ 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
    distance A B = x ∧ distance C D = x ∧ 
    distance A D = x ∧ distance B C = x) →
  d = x * Real.sqrt 2 ∧ d = (8 * R * Real.sqrt 2) / 5 :=
begin
  sorry
end

end diagonal_of_square_on_circle_and_tangent_l142_142161


namespace watch_gain_percentage_l142_142200

variable (CP SP_new : ℕ) (loss_percent extra_amount : ℝ)

def gain_percentage (CP SP_new : ℕ) := 
    ((SP_new - CP) : ℝ) / (CP : ℝ) * 100

theorem watch_gain_percentage :
  CP = 2000 →
  loss_percent = 10 →
  extra_amount = 280 →
  SP_new = (2000 - (10/100) * 2000) + 280 →
  gain_percentage CP SP_new = 4 := 
by
  intros hcp hlp hea hsp
  rw [hcp, hlp, hea] at hsp
  unfold gain_percentage
  simp only [hcp]
  sorry

end watch_gain_percentage_l142_142200


namespace num_divisors_of_1386_l142_142271

theorem num_divisors_of_1386 : 
  (∀ n, n = 1386 → 
    (∃ (a b c d : ℕ), 
        n = (2^a) * (3^b) * (7^c) * (11^d) ∧ 
        a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1) → 
        (finset.range (n + 1)).filter (λ x, n % x = 0).card = 24) := 
by
  sorry

end num_divisors_of_1386_l142_142271


namespace total_sum_l142_142440

def is_subset (n : ℕ) (A : set ℕ) : Prop :=
  ∀ a ∈ A, a ∈ {k | k ≤ 1998}

def is_n_tuple_subset (n : ℕ) (A : vector (set ℕ) n) : Prop :=
  ∀ i, is_subset 1998 (A.nth i)

noncomputable def F (n : ℕ) : set (vector (set ℕ) n) :=
  { A | is_n_tuple_subset n A }

def union_cardinality (n : ℕ) (A : vector (set ℕ) n) : ℕ :=
  (A.to_list.foldr (∪) ∅).card

theorem total_sum (n : ℕ) :
  ∑ A in (F n).to_finset, union_cardinality n A = 2^(1998 * n) * 1998 * (1 - 1 / 2^n) := by
  sorry

end total_sum_l142_142440


namespace eleven_y_minus_x_eq_one_l142_142540

theorem eleven_y_minus_x_eq_one 
  (x y : ℤ) 
  (hx_pos : x > 0)
  (h1 : x = 7 * y + 3)
  (h2 : 2 * x = 6 * (3 * y) + 2) : 
  11 * y - x = 1 := 
by 
  sorry

end eleven_y_minus_x_eq_one_l142_142540


namespace circles_intersect_probability_l142_142989

-- Define the problem parameters
structure Circle (center_x : ℝ) (center_y : ℝ) (radius : ℝ)

-- Define the two circles with their centers as variables
def circle_C (C_X : ℝ) : Circle := ⟨C_X, 0, 1.5⟩
def circle_D (D_X : ℝ) : Circle := ⟨D_X, 2, 1.5⟩

-- Define that the centers are chosen uniformly from the given segments
def uniform_segment_C (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3
def uniform_segment_D (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Define the distance between the centers of the circles
noncomputable def distance_centers (C_X D_X : ℝ) : ℝ :=
  sqrt ((C_X - D_X) ^ 2 + 4)

-- Define the intersection condition
def circles_intersect (C_X D_X : ℝ) : Prop :=
  distance_centers C_X D_X ≤ 3

-- The probability that C and D intersect, computed analytically
def intersection_probability : ℝ := (8 * sqrt 5 - 5) / 9

-- The theorem we need to prove
theorem circles_intersect_probability : 
  (∫ (C_X : ℝ) in 0..3, 
    (if uniform_segment_C C_X then 
      ∫ (D_X : ℝ) in 0..3, 
        if circles_intersect C_X D_X then 1 else 0
      else 0)) = intersection_probability :=
sorry

end circles_intersect_probability_l142_142989


namespace y_relationship_range_of_x_l142_142310

-- Definitions based on conditions
variable (x : ℝ) (y : ℝ)

-- Condition: Perimeter of the isosceles triangle is 6 cm
def perimeter_is_6 (x : ℝ) (y : ℝ) : Prop :=
  2 * x + y = 6

-- Condition: Function relationship of y in terms of x
def y_function (x : ℝ) : ℝ :=
  6 - 2 * x

-- Prove the functional relationship y = 6 - 2x
theorem y_relationship (x : ℝ) : y = y_function x ↔ perimeter_is_6 x y := by
  sorry

-- Prove the range of values for x
theorem range_of_x (x : ℝ) : 3 / 2 < x ∧ x < 3 ↔ (0 < y_function x ∧ perimeter_is_6 x (y_function x)) := by
  sorry

end y_relationship_range_of_x_l142_142310


namespace find_snickers_l142_142826

theorem find_snickers (total_candy_bars : ℕ) (mars_bars : ℕ) (butterfingers : ℕ) (h : total_candy_bars = 12) (h1 : mars_bars = 2) (h2 : butterfingers = 7) : total_candy_bars - (mars_bars + butterfingers) = 3 :=
by
  rw [h, h1, h2]
  sorry

end find_snickers_l142_142826


namespace min_sum_squares_of_roots_l142_142243

theorem min_sum_squares_of_roots :
  ∃ p, (∀ p ≠ 1, let x1 := (-((2 - p)) + Math.sqrt((2 - p)^2 - 4 * 1 * (-p - 3))) / (2 * 1) in
              let x2 := (-((2 - p)) - Math.sqrt((2 - p)^2 - 4 * 1 * (-p - 3))) / (2 * 1) in
              (x1^2 + x2^2) > ((-(2 - 1))^2 - 2 * (-(1 + 3)))) ∧ 
              (let x1 := (-((2 - 1)) + Math.sqrt((2 - 1)^2 - 4 * 1 * (-1 - 3))) / (2 * 1) in
               let x2 := (-((2 - 1)) - Math.sqrt((2 - 1)^2 - 4 * 1 * (-1 - 3))) / (2 * 1) in
               (x1^2 + x2^2) = (1^2 - 2 * 1 + 10))) :=
by
  sorry

end min_sum_squares_of_roots_l142_142243


namespace natural_numbers_divisible_by_7_between_200_400_l142_142363

theorem natural_numbers_divisible_by_7_between_200_400 : 
  { n : ℕ | 200 <= n ∧ n <= 400 ∧ n % 7 = 0 }.to_finset.card = 29 := 
  sorry

end natural_numbers_divisible_by_7_between_200_400_l142_142363


namespace car_travel_distance_l142_142170

theorem car_travel_distance (h : 2.5 * 60 > 175) : ∃ d, d > 175 ∧ d = 2.5 * 60 := 
by
  use 2.5 * 60
  split
  · exact h
  · rfl
sorrry

end car_travel_distance_l142_142170


namespace mindy_earns_k_times_more_than_mork_l142_142463

-- Given the following conditions:
-- Mork's tax rate: 0.45
-- Mindy's tax rate: 0.25
-- Combined tax rate: 0.29
-- Mindy earns k times more than Mork

theorem mindy_earns_k_times_more_than_mork (M : ℝ) (k : ℝ) (hM : M > 0) :
  (0.45 * M + 0.25 * k * M) / (M * (1 + k)) = 0.29 → k = 4 :=
by
  sorry

end mindy_earns_k_times_more_than_mork_l142_142463


namespace divisor_exists_l142_142119

-- Define the initial conditions
def all_natural_numbers := { n : ℕ | 1 ≤ n ∧ n ≤ 2014 }

-- Define the remaining set after erasing 1006 numbers
def remaining_numbers (S : set ℕ) (h : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : Prop :=
  ∃ a b ∈ S, a ≠ b ∧ a ∣ b

-- The problem statement in Lean
theorem divisor_exists (S : set ℕ) (h_sub : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : 
  remaining_numbers S h_sub h_card :=
begin
  sorry
end

end divisor_exists_l142_142119


namespace prove_A_plus_B_l142_142035

variable (A B : ℝ)

theorem prove_A_plus_B (h : ∀ x : ℝ, x ≠ 2 → (A / (x - 2) + B * (x + 3) = (-5 * x^2 + 20 * x + 34) / (x - 2))) : A + B = 9 := by
  sorry

end prove_A_plus_B_l142_142035


namespace find_P_nplus1_l142_142042

-- Conditions
def P (n : ℕ) (k : ℕ) : ℚ :=
  1 / Nat.choose n k

-- Lean 4 statement for the proof
theorem find_P_nplus1 (n : ℕ) : (if Even n then P n (n+1) = 1 else P n (n+1) = 0) := by
  sorry

end find_P_nplus1_l142_142042


namespace arithmetic_geometric_seq_l142_142888

open Real

theorem arithmetic_geometric_seq (a d : ℝ) (h₀ : d ≠ 0) 
  (h₁ : (a + d) * (a + 5 * d) = (a + 2 * d) ^ 2) : 
  (a + 2 * d) / (a + d) = 3 :=
sorry

end arithmetic_geometric_seq_l142_142888


namespace max_value_part1_l142_142561

theorem max_value_part1 (a : ℝ) (h : a < 3 / 2) : 2 * a + 4 / (2 * a - 3) + 3 ≤ 2 :=
sorry

end max_value_part1_l142_142561


namespace probability_P_is_1_over_3_l142_142832

-- Definitions and conditions
def A := 0
def B := 3
def C := 1
def D := 2
def length_AB := B - A
def length_CD := D - C

-- Problem statement to prove
theorem probability_P_is_1_over_3 : (length_CD / length_AB) = 1 / 3 := by
  sorry

end probability_P_is_1_over_3_l142_142832


namespace expected_circle_area_correct_l142_142192

noncomputable def expected_circle_area (n : ℕ) (a d : ℝ) :=
  π * n * (d + a^2)

theorem expected_circle_area_correct (n : ℕ) (a d : ℝ)
  (X : ℕ → ℝ)
  (h_mean : ∀ i, E[X i] = a)
  (h_var : ∀ i, variance (X i) = d) :
  E[π * ∑ i in range n, (X i)^2] = π * n * (d + a^2) :=
by 
  sorry

end expected_circle_area_correct_l142_142192


namespace find_b_plus_m_l142_142531

def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 7
def line2 (b : ℝ) (x : ℝ) : ℝ := 4 * x + b

theorem find_b_plus_m :
  ∃ (m b : ℝ), line1 m 8 = 11 ∧ line2 b 8 = 11 ∧ b + m = -20.5 :=
sorry

end find_b_plus_m_l142_142531


namespace not_perfect_power_probability_l142_142883

def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ x^y = n

theorem not_perfect_power_probability :
  let total := 200
  let count_perfect_powers := 19
  let count_non_perfect_powers := total - count_perfect_powers
  (count_non_perfect_powers : ℚ) / total = 181 / 200 :=
by
  sorry

end not_perfect_power_probability_l142_142883


namespace geometric_sum_sequence_l142_142492

theorem geometric_sum_sequence (n : ℕ) (a : ℕ → ℕ) (a1 : a 1 = 2) (a4 : a 4 = 16) :
    (∃ q : ℕ, a 2 = a 1 * q) → (∃ S_n : ℕ, S_n = 2 * (2 ^ n - 1)) :=
by
  sorry

end geometric_sum_sequence_l142_142492


namespace goalkeeper_not_at_goal_line_total_energy_consumption_scoring_opportunities_l142_142770

def movements := [7, -3, 8, 4, -6, -8, 14, -15]
def energy_consumption_rate := 0.1
def scoring_distance := 10

-- Part 1: Prove that the goalkeeper is not exactly at the goal line at the end of the movements.
theorem goalkeeper_not_at_goal_line :
  ∑ i in movements, i ≠ 0 := by
  sorry

-- Part 2: Prove that the total energy consumption of the goalkeeper is 6.5 calories.
theorem total_energy_consumption :
  (∑ i in movements, abs i) * energy_consumption_rate = 6.5 := by
  sorry

-- Part 3: Prove that the opponent player has exactly 3 scoring opportunities during the period.
theorem scoring_opportunities :
  list (λ acc i, acc + i) movements [] |>.map abs |>.filter (> scoring_distance) |>.length = 3 := by
  sorry

end goalkeeper_not_at_goal_line_total_energy_consumption_scoring_opportunities_l142_142770


namespace find_abc_l142_142258

theorem find_abc (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by 
  sorry

end find_abc_l142_142258


namespace cone_slant_height_l142_142180

theorem cone_slant_height (radius : ℝ) (theta : ℝ) (h_radius : radius = 6) (h_theta : theta = 240) : 
  let circumference_base := 2 * Real.pi * radius in
  let arc_length := (theta / 360) * 2 * Real.pi * slant_height in 
  12 * Real.pi = arc_length → 
  slant_height = 9 := sorry

end cone_slant_height_l142_142180


namespace f_n_perfect_square_iff_l142_142443

def d_i (i k : ℕ) : ℕ := (Nat.divisors k).filter (λ d => d > i).length

def f (n : ℕ) : ℕ := 
  (Finset.range (n^2 / 2).natCeil).sum (λ i => d_i i (n^2 - i)) - 
  2 * (Finset.range (n / 2).natCeil).sum (λ i => d_i i (n - i))

theorem f_n_perfect_square_iff (n : ℕ) : 
  (∃ m : ℕ, f n = m * m) ↔ ∃ p k : ℕ, Nat.Prime p ∧ n = p^k := by
  sorry

end f_n_perfect_square_iff_l142_142443


namespace count_multiples_of_7_l142_142356

theorem count_multiples_of_7 (low high : ℕ) (hlow : low = 200) (hhigh : high = 400) : 
  (card {n | low ≤ n ∧ n ≤ high ∧ n % 7 = 0}) = 29 := by
  sorry

end count_multiples_of_7_l142_142356


namespace cook_weave_l142_142586

theorem cook_weave (Y C W OC CY CYW : ℕ) (hY : Y = 25) (hC : C = 15) (hW : W = 8) (hOC : OC = 2)
  (hCY : CY = 7) (hCYW : CYW = 3) : 
  ∃ (CW : ℕ), CW = 9 :=
by 
  have CW : ℕ := C - OC - (CY - CYW) 
  use CW
  sorry

end cook_weave_l142_142586


namespace medicine_supply_duration_l142_142801

theorem medicine_supply_duration
  (pills_per_three_days : ℚ := 1 / 3)
  (total_pills : ℕ := 60)
  (days_per_month : ℕ := 30) :
  (((total_pills : ℚ) * ( 3 / pills_per_three_days)) / days_per_month) = 18 := sorry

end medicine_supply_duration_l142_142801


namespace number_of_n_not_divisible_by_2_or_3_l142_142514

theorem number_of_n_not_divisible_by_2_or_3 :
  let count := (999 - (499 + 333 - 166))
  in count = 333 :=
sorry

end number_of_n_not_divisible_by_2_or_3_l142_142514


namespace total_peppers_weight_l142_142736

theorem total_peppers_weight :
  let green_peppers : ℝ := 1.45
  let red_peppers : ℝ := 0.68
  let yellow_peppers : ℝ := 1.6
  let jalapeno_peppers : ℝ := 2.25
  let habanero_peppers : ℝ := 3.2
  green_peppers + red_peppers + yellow_peppers + jalapeno_peppers + habanero_peppers = 9.18 :=
by {
  let green_peppers : ℝ := 1.45
  let red_peppers : ℝ := 0.68
  let yellow_peppers : ℝ := 1.6
  let jalapeno_peppers : ℝ := 2.25
  let habanero_peppers : ℝ := 3.2
  calc
  green_peppers + red_peppers + yellow_peppers + jalapeno_peppers + habanero_peppers
    = 1.45 + 0.68 + 1.6 + 2.25 + 3.2 : by refl
  ... = 9.18 : by norm_num
}. 

end total_peppers_weight_l142_142736


namespace projection_a_onto_b_l142_142345

-- Let a and b be vectors in a real inner product space.
variables (a b : ℝ^n)

-- Given conditions
variable (ha : ‖a‖ = 2)
variable (hb : ‖b‖ = 3)
variable (h2a_b : ‖(2 : ℝ) • a - b‖ = Real.sqrt 13)

-- Theorem statement
theorem projection_a_onto_b (a b : ℝ^n) 
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = 3)
  (h2a_b : ‖(2 : ℝ) • a - b‖ = Real.sqrt 13) : 
  (a ⬝ b) / (‖b‖) = 1 :=
sorry

end projection_a_onto_b_l142_142345


namespace part1_part2_l142_142030

-- Part (1)
theorem part1 (a : ℝ) (b : ℝ) (x : ℝ) (h_a : a = 1) (h_x : 0 ≤ x):
  e^x + a * sin x + b ≥ 0 → b ≥ -1 :=
by
  sorry

-- Part (2)
theorem part2 (m : ℝ) : 
  (∀ x : ℝ, e^x - 2 = (m - 2 * x) / x → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ e^x1 - 2 = (m - 2 * x1) / x1 ∧ e^x2 - 2 = (m - 2 * x2) / x2) → 
  (-1 / Real.exp(1) < m ∧ m < 0) :=
by
  sorry

end part1_part2_l142_142030


namespace beta_minus_alpha_l142_142825

variable {α β : ℝ}
axiom a_vector : (cos α, sin α)
axiom b_vector : (cos β, sin β)
axiom α_range : 0 < α ∧ α < β ∧ β < π
axiom magnitude_condition : |2 * a_vector + b_vector| = |a_vector - 2 * b_vector|

theorem beta_minus_alpha : β - α = π / 2 := by
  sorry

end beta_minus_alpha_l142_142825


namespace inequality_problem_l142_142390

theorem inequality_problem (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by {
  sorry
}

end inequality_problem_l142_142390


namespace sum_proper_divisors_600_l142_142994

def sum_of_proper_divisors (n : ℕ) : ℕ :=
  nat.divisors n |>.filter (λ d, d < n) |>.sum

theorem sum_proper_divisors_600 : sum_of_proper_divisors 600 = 1260 := by
  sorry

end sum_proper_divisors_600_l142_142994


namespace number_of_divisibles_by_7_l142_142349

theorem number_of_divisibles_by_7 (a b : ℕ) (h1 : a = 200) (h2 : b = 400) : 
  (nat.card {n | a ≤ 7 * n ∧ 7 * n ≤ b}) = 29 := 
by sorry

end number_of_divisibles_by_7_l142_142349


namespace base_of_first_term_l142_142103

theorem base_of_first_term (e : ℕ) (b : ℝ) (h : e = 35) :
  b^e * (1/4)^18 = 1/(2 * 10^35) → b = 1/5 :=
by
  sorry

end base_of_first_term_l142_142103


namespace inheritance_amount_l142_142016

theorem inheritance_amount (x : ℝ) 
  (h1 : x * 0.25 + (x - x * 0.25) * 0.12 = 13600) : x = 40000 :=
by
  -- This is where the proof would go
  sorry

end inheritance_amount_l142_142016


namespace min_cost_to_package_collection_l142_142919

theorem min_cost_to_package_collection (box_length box_width box_height : ℕ)
  (cost_per_box : ℝ)
  (total_volume_to_be_packaged : ℕ) :
  box_length = 20 → box_width = 20 → box_height = 12 →
  cost_per_box = 0.5 → total_volume_to_be_packaged = 2400000 →
  (total_volume_to_be_packaged / (box_length * box_width * box_height) * cost_per_box = 250) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  have hvolume : 20 * 20 * 12 = 4800 := by norm_num
  have hboxes : 2400000 / 4800 = 500 := by norm_num
  have hcost : 500 * 0.5 = 250 := by norm_num
  exact eq.trans (eq.trans (by norm_num) hboxes) (by norm_num)

end min_cost_to_package_collection_l142_142919


namespace pages_revised_twice_l142_142505

theorem pages_revised_twice :
  ∀ (total_pages revised_once revised_twice_no_revisions total_cost : ℕ),
  total_pages = 200 →
  revised_once = 80 →
  (total_pages = revised_once + revised_twice_no_revisions + (200 - revised_once - revised_twice_no_revisions)) →
  total_cost = 1360 →
  total_cost = (total_pages * 5) + (revised_once * 3) + (revised_twice_no_revisions * 6) →
  revised_twice_no_revisions = 20 :=
by
  intros total_pages revised_once revised_twice_no_revisions total_cost
  assume h1 h2 h3 h4 h5
  sorry

end pages_revised_twice_l142_142505


namespace altitudes_are_equal_l142_142280

-- Definitions of geometric entities involved in the problem
variables {α : Type*} [EuclideanGeometry α]
variables {A B C P P' Q Q' N : α} (Γ Γ' Γ₀ Γ₀' : Circle α)

-- Assumptions from problem conditions
variables (h_acute : Triangle A B C ∧ Triangle.isAcute A B C)
variables (h_not_equilateral : ¬Triangle.isEquilateral A B C)
variables (h_incircle : Circle.isIncircle Γ (Triangle.mk A B C))
variables (h_excircle : Circle.isExcircle Γ' (Triangle.mk A B C) A)
variables (h_tangent_P : Circle.isTangentAt Γ (Line.mk B C) P)
variables (h_tangent_P' : Circle.isTangentAt Γ' (Line.mk B C) P')
variables (h_passes_BC_Γ₀ : Circle.passesThrough Γ₀ B ∧ Circle.passesThrough Γ₀ C)
variables (h_tangent_Q : Circle.isTangentAt Γ₀ Γ Q)
variables (h_passes_BC_Γ₀' : Circle.passesThrough Γ₀' B ∧ Circle.passesThrough Γ₀' C)
variables (h_tangent_Q' : Circle.isTangentAt Γ₀' Γ' Q')
variables (h_intersect_N : Line.mk P Q ∩ Line.mk P' Q' = {N})

-- The theorem to be proved
theorem altitudes_are_equal :
  Line.perpendicular (Line.mk A N) (Line.mk B C) :=
sorry

end altitudes_are_equal_l142_142280


namespace part1_part2_part3_l142_142582

noncomputable def p1_cost (t : ℕ) : ℕ := 
  if t <= 150 then 58 else 58 + 25 * (t - 150) / 100

noncomputable def p2_cost (t : ℕ) (a : ℕ) : ℕ := 
  if t <= 350 then 88 else 88 + a * (t - 350)

-- Part 1: Prove the costs for 260 minutes
theorem part1 : p1_cost 260 = 855 / 10 ∧ p2_cost 260 30 = 88 :=
by 
  sorry

-- Part 2: Prove the existence of t for given a
theorem part2 (t : ℕ) : (a = 30) → (∃ t, p1_cost t = p2_cost t a) :=
by 
  sorry

-- Part 3: Prove a=45 and the range for which Plan 1 is cheaper
theorem part3 : 
  (a = 45) ↔ (p1_cost 450 = p2_cost 450 a) ∧ (∀ t, (0 ≤ t ∧ t < 270) ∨ (t > 450) → p1_cost t < p2_cost t 45 ) :=
by 
  sorry

end part1_part2_part3_l142_142582


namespace bisectors_form_inscribed_quadrilateral_l142_142843

noncomputable def angle_sum_opposite_bisectors {α β γ δ : ℝ} (a_bisector b_bisector c_bisector d_bisector : ℝ)
  (cond : α + β + γ + δ = 360) : Prop :=
  (a_bisector + b_bisector + c_bisector + d_bisector) = 180

theorem bisectors_form_inscribed_quadrilateral
  {α β γ δ : ℝ} (convex_quad : α + β + γ + δ = 360) :
  ∃ a_bisector b_bisector c_bisector d_bisector : ℝ,
  angle_sum_opposite_bisectors a_bisector b_bisector c_bisector d_bisector convex_quad := 
sorry

end bisectors_form_inscribed_quadrilateral_l142_142843


namespace perfect_squares_digit_4_5_6_l142_142740

theorem perfect_squares_digit_4_5_6 (n : ℕ) (hn : n^2 < 2000) : 
  (∃ k : ℕ, k = 18) :=
  sorry

end perfect_squares_digit_4_5_6_l142_142740


namespace natural_numbers_divisible_by_7_between_200_400_l142_142361

theorem natural_numbers_divisible_by_7_between_200_400 : 
  { n : ℕ | 200 <= n ∧ n <= 400 ∧ n % 7 = 0 }.to_finset.card = 29 := 
  sorry

end natural_numbers_divisible_by_7_between_200_400_l142_142361


namespace transformation_impossible_l142_142237

noncomputable def initial_sequence : List Bool :=
  List.repeat false 2003 ++ [true]

noncomputable def target_sequence : List Bool :=
  [true] ++ List.repeat false 2003

def operation_a (seq : List Bool) : List Bool :=
  if 010 occurs in seq then replace 010 with 1 in seq else seq

def operation_b (seq : List Bool) : List Bool :=
  if 1 occurs in seq then replace 1 with 010 in seq else seq

def operation_c (seq : List Bool) : List Bool :=
  if 110 occurs in seq then replace 110 with 0 in seq else seq

def operation_d (seq : List Bool) : List Bool :=
  if 0 occurs in seq then replace 0 with 110 in seq else seq

theorem transformation_impossible : 
  ∀ seq, (seq = initial_sequence) → (¬ can_transform seq target_sequence)
  where can_transform : List Bool → List Bool → Prop
  | seq, target => 
      if seq = target then true
      else can_transform (operation_a seq) target ||
           can_transform (operation_b seq) target ||
           can_transform (operation_c seq) target ||
           can_transform (operation_d seq) target
:= sorry

end transformation_impossible_l142_142237


namespace smallest_m_l142_142453

def has_pair (s : Set ℕ) : Prop :=
  ∃ a b ∈ s, a ≠ b ∧ a ^ b = b ^ a

def forall_partitions_has_pair (m : ℕ) : Prop :=
  ∀ A B : Set ℕ, (λ S, S = { x | 2 ≤ x ∧ x ≤ m }) = A ∪ B ∧ A ∩ B = ∅ → has_pair A ∨ has_pair B

theorem smallest_m (m : ℕ) : m = 16 ↔ (m ≥ 2 ∧ ∀ A B : Set ℕ, (λ S, S = { x | 2 ≤ x ∧ x ≤ m }) = A ∪ B ∧ A ∩ B = ∅ → has_pair A ∨ has_pair B) :=
by
  sorry

end smallest_m_l142_142453


namespace number_of_true_propositions_is_1_l142_142608

-- Define each proposition as a boolean value
def prop1 : Prop := ∀ (L₁ L₂ L₃ : Line), (IntersectedByThirdLine L₁ L₂ L₃) → CorrespondingAnglesEqual L₁ L₂ L₃.

def prop2 : Prop := ∀ (P : Point) (L : Line), ∃! (M : Line), (ParallelToLine P M L).

def prop3 : Prop := ∀ (P : Point) (L : Line), ∃! (M : Line), (PerpendicularToLine P M L).

def prop4 : Prop := ∀ (L₁ L₂ L₃ : Line), (InSamePlane L₁ L₂ L₃) → (PerpendicularToLine L₁ L₃ ∧ PerpendicularToLine L₂ L₃) → Parallel L₁ L₂.

-- Now, formulate the theorem to prove number of true propositions:
theorem number_of_true_propositions_is_1 : (¬ prop1) ∧ (¬ prop2) ∧ prop3 ∧ prop4 → (CountTrue [prop1, prop2, prop3, prop4] = 1) := by
  sorry

end number_of_true_propositions_is_1_l142_142608


namespace count_valid_pairs_l142_142650

theorem count_valid_pairs :
  let S := {a | 10 ≤ a ∧ a ≤ 30} in
  (∃ count : ℕ, count = 11 ∧ 
   ∃ f : ℕ × ℕ → Prop, 
   (∀ (a b : ℕ), (a ∈ S) ∧ (b ∈ S) → f (a, b)) ∧
   (∀ (a b : ℕ), f (a, b) ↔ (Nat.gcd a b + Nat.lcm a b = a + b)) ∧
   (count = S.card)) :=
sorry

end count_valid_pairs_l142_142650


namespace hypergeom_prob_formula_mle_formula_l142_142935

-- Definitions of the conditions
variables (N M n m : ℕ)

-- Function to calculate the binomial coefficient
noncomputable def binom (n k : ℕ) : ℚ := nat.choose n k

-- Part (a): The hypergeometric probability distribution
def hypergeom_prob : ℚ :=
  (binom M m * binom (N - M) (n - m)) / binom N n

-- Part (b): The maximum likelihood estimate for N
noncomputable def mle (M n m : ℕ) : ℕ :=
  if m > 0 then M * n / m else 0

-- Theorems and properties to be proven
theorem hypergeom_prob_formula :
  hypergeom_prob N M n m = (binom M m * binom (N - M) (n - m)) / binom N n := by sorry

theorem mle_formula :
  mle M n m = ⌊(M * n : ℚ) / m⌋ := by sorry

end hypergeom_prob_formula_mle_formula_l142_142935


namespace angle_between_a_b_l142_142027

noncomputable def unit_vector (v : ℝ × ℝ × ℝ) : Prop := (v.1^2 + v.2^2 + v.3^2 = 1)

def vector_zero : ℝ × ℝ × ℝ := (0, 0, 0)

def vector_add (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def scalar_mul (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (k * v.1, k * v.2, k * v.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3)

theorem angle_between_a_b 
    (a b c : ℝ × ℝ × ℝ)
    (ha : unit_vector a)
    (hb : unit_vector b)
    (hc : unit_vector c)
    (h_eq : vector_add a (vector_add (scalar_mul 2 b) (scalar_mul 2 c)) = vector_zero) :
    real.arccos (dot_product a b) = 104.48 :=
begin
  sorry,
end

end angle_between_a_b_l142_142027


namespace coordinates_of_Q_l142_142470

theorem coordinates_of_Q (m : ℤ) (P Q : ℤ × ℤ) (hP : P = (m + 2, 2 * m + 4))
  (hQ_move : Q = (P.1, P.2 + 2)) (hQ_x_axis : Q.2 = 0) : Q = (-1, 0) :=
sorry

end coordinates_of_Q_l142_142470


namespace log_inequality_solution_expression_value_l142_142940

/-
Problem 1:
Prove the solution set for the inequality log_{1/2}(x + 2) > -3 is -2 < x < 6
-/
theorem log_inequality_solution (x : ℝ) : 
  (log (1/2) (x + 2) > -3) ↔ (-2 < x) ∧ (x < 6) :=
sorry

/-
Problem 2:
Prove the value of the given expression is 221/2
-/
theorem expression_value : 
  ( ( (1/8: ℝ)^(1/3) * (-7/6)^0 ) + 8^(0.25) * (2: ℝ)^(1/4) + ((2: ℝ)^(1/3) * (3: ℝ)^(1/2))^6 ) = 221/2 :=
sorry

end log_inequality_solution_expression_value_l142_142940


namespace candidate_marks_secured_l142_142580

noncomputable def max_marks : ℝ := 185.71
noncomputable def pass_percentage : ℝ := 0.35
noncomputable def marks_failed_by : ℝ := 23
noncomputable def passing_marks : ℝ := pass_percentage * max_marks

theorem candidate_marks_secured : ∃ (x : ℝ), x + marks_failed_by = passing_marks → x = 42 :=
begin
  use 42,
  intro h,
  exact sorry
end

end candidate_marks_secured_l142_142580


namespace circle_passing_through_pole_l142_142416

noncomputable def equation_of_circle (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.cos θ

theorem circle_passing_through_pole :
  equation_of_circle 2 θ := 
sorry

end circle_passing_through_pole_l142_142416


namespace max_distinct_digits_l142_142953

theorem max_distinct_digits (n : ℕ) (digits : List ℕ) :
  (∀ d ∈ digits, d ≠ 0 ∧ d ∈ (Digits n) → n % d = 0) →
  ∃ m ≤ 9, m = 8 :=
by
  sorry

end max_distinct_digits_l142_142953


namespace min_value_of_y_l142_142394

theorem min_value_of_y (y : ℝ) (hy_pos : y > 0) (h : log 3 y ≥ log 3 9 - log 3 y) : y = 3 :=
sorry

end min_value_of_y_l142_142394


namespace trapezoid_length_l142_142785

variable (W X Y Z : ℝ)
variable (WX ZY WY XY : ℝ)

theorem trapezoid_length (h1 : WX = ZY)
  (h2 : WY * ZY ≠ 0)
  (h3 : YZ = 15)
  (h4 : Real.tan Z = 4 / 3)
  (h5 : Real.tan X = 3 / 2)
  (h6 : ∃ WY WX XY, XY^2 = WY^2 + WX^2 ∧ WX = WY / (3 / 2) ∧ WY = 15 * (4 / 3) ∧ XY > 0):
  XY = 20 * Real.sqrt 13 / 3 :=
by
  sorry

end trapezoid_length_l142_142785


namespace abs_value_of_difference_l142_142509

noncomputable def abs_diff_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 275) : ℝ :=
| x - y |

theorem abs_value_of_difference : ∃ x y : ℝ, (x + y = 36) ∧ (x * y = 275) ∧ |x - y| = 14 :=
by
  use [18 + √7, 18 - √7]
  split
  · simp
  · specialize h even; contradiction. symm; refine abs_diff_of_numbers |x - y| fact (
    calc (x - y) * (x + y) = 0 on
  specialize h even; fully_zip_ closed; symmetry; sorry

end abs_value_of_difference_l142_142509


namespace ratio_a_b_l142_142753

theorem ratio_a_b (a b c d : ℝ) 
  (h1 : b / c = 7 / 9) 
  (h2 : c / d = 5 / 7)
  (h3 : a / d = 5 / 12) : 
  a / b = 3 / 4 :=
  sorry

end ratio_a_b_l142_142753


namespace sallys_dad_nickels_l142_142853

theorem sallys_dad_nickels :
  ∀ (initial_nickels mother's_nickels total_nickels nickels_from_dad : ℕ), 
    initial_nickels = 7 → 
    mother's_nickels = 2 →
    total_nickels = 18 →
    total_nickels = initial_nickels + mother's_nickels + nickels_from_dad →
    nickels_from_dad = 9 :=
by
  intros initial_nickels mother's_nickels total_nickels nickels_from_dad
  intros h1 h2 h3 h4
  sorry

end sallys_dad_nickels_l142_142853


namespace baba_yaga_departure_and_speed_l142_142213

variables (T : ℕ) (d : ℕ)

theorem baba_yaga_departure_and_speed :
  (50 * (T + 2) = 150 * (T - 2)) →
  (12 - T = 8) ∧ (d = 50 * (T + 2)) →
  (d = 300) ∧ ((d / T) = 75) :=
by
  intros h1 h2
  sorry

end baba_yaga_departure_and_speed_l142_142213


namespace texts_sent_total_l142_142058

def texts_sent_on_monday_to_allison_and_brittney : Nat := 5 + 5
def texts_sent_on_tuesday_to_allison_and_brittney : Nat := 15 + 15

def total_texts_sent (texts_monday : Nat) (texts_tuesday : Nat) : Nat := texts_monday + texts_tuesday

theorem texts_sent_total :
  total_texts_sent texts_sent_on_monday_to_allison_and_brittney texts_sent_on_tuesday_to_allison_and_brittney = 40 :=
by
  sorry

end texts_sent_total_l142_142058


namespace diane_last_year_harvest_l142_142634

noncomputable def last_year_harvest (x y : ℕ) : Prop :=
  y = x + 6085 ∧ y = 8564

theorem diane_last_year_harvest : ∃ x : ℕ, last_year_harvest x 8564 → x = 2479 :=
by
  intro h
  use 2479
  -- proof steps will go here
  sorry

end diane_last_year_harvest_l142_142634


namespace is_real_iff_is_pure_imaginary_iff_l142_142676

section ComplexNumberConditions

variable (m : ℝ)

def complexNumber (m : ℝ) : ℂ :=
  (m^2 - 5 * m + 6 : ℂ) + (m^2 - 3 * m : ℂ) * Complex.I

theorem is_real_iff (h : complexNumber m m = (m^2 - 5 * m + 6 : ℂ)) :
  m = 0 ∨ m = 3 :=
by
  sorry

theorem is_pure_imaginary_iff (h1 : complexNumber m m = (m^2 - 3 * m : ℂ) * Complex.I)
  (h2 : m ≠ 0 ∧ m ≠ 3) :
  m = 2 :=
by
  sorry

end ComplexNumberConditions

end is_real_iff_is_pure_imaginary_iff_l142_142676


namespace unique_solution_l142_142024

noncomputable def S := { x : ℝ // x ≠ 0 }

def f (x : S) : ℝ := sorry

axiom cond1 : f ⟨2, by norm_num⟩ = 4

axiom cond2 : ∀ x y : S, x.1 + y.1 ≠ 0 → f ⟨1 / (x.1 + y.1), by norm_num⟩ = f ⟨1 / x.1, by norm_num⟩ + f ⟨1 / y.1, by norm_num⟩

axiom cond3 : ∀ x y : S, x.1 + y.1 ≠ 0 → 2 * x.1 * y.1 * f ⟨x.1 + y.1, by norm_num⟩ = (x.1 + y.1) * f x * f y

axiom cond4 : ∀ x y : S, x.1 < y.1 → f x < f y

theorem unique_solution : ∃! f : S → ℝ, 
  (f ⟨2, by norm_num⟩ = 4) ∧
  (∀ x y : S, x.1 + y.1 ≠ 0 → f ⟨1 / (x.1 + y.1), by norm_num⟩ = f ⟨1 / x.1, by norm_num⟩ + f ⟨1 / y.1, by norm_num⟩) ∧
  (∀ x y : S, x.1 + y.1 ≠ 0 → 2 * x.1 * y.1 * f ⟨x.1 + y.1, by norm_num⟩ = (x.1 + y.1) * f x * f y) ∧
  (∀ x y : S, x.1 < y.1 → f x < f y) :=
sorry

end unique_solution_l142_142024


namespace fruits_in_good_condition_percentage_l142_142549

theorem fruits_in_good_condition_percentage (total_oranges total_bananas rotten_oranges_percentage rotten_bananas_percentage : ℝ) 
  (h1 : total_oranges = 600) 
  (h2 : total_bananas = 400) 
  (h3 : rotten_oranges_percentage = 0.15) 
  (h4 : rotten_bananas_percentage = 0.08) : 
  (1 - ((rotten_oranges_percentage * total_oranges + rotten_bananas_percentage * total_bananas) / (total_oranges + total_bananas))) * 100 = 87.8 :=
by 
  sorry

end fruits_in_good_condition_percentage_l142_142549


namespace pb_length_l142_142808

noncomputable def point_outside_circle (P : Point) (O : Circle) : Prop := 
  ¬on_circle P O

def tangent_to_circle (P : Point) (O : Circle) (T : Point) : Prop :=
  tangent_line_from_point P O T

def secant_intersecting_circle (P : Point) (O : Circle) (A B : Point) : Prop :=
  secant_from_point P O A B

def segment_length {P A B : Point} (d : ℝ) : Prop := 
  distance P A = d ∧ distance P B = d

theorem pb_length :
  ∀ (P O : Type) (A B T : Point), 
    point_outside_circle P O →
    tangent_to_circle P O T →
    secant_intersecting_circle P O A B →
    (segment_length 5 P A) →
    (distance T P = 2 * (distance A B - distance P A)) →
    P A < P B → 
    P B = 8.75 :=
by sorry

end pb_length_l142_142808


namespace expression_has_8_distinct_factors_l142_142762

theorem expression_has_8_distinct_factors (x y : ℕ) (hx : Nat.Prime x ∧ x % 2 = 1) (hy : Nat.Prime y ∧ y % 2 = 1) (hxy : x < y) :
  ∃ (E : ℕ), E = x^3 * y ∧ (Nat.factors_count E = 8) :=
sorry

end expression_has_8_distinct_factors_l142_142762


namespace cost_of_iced_coffee_for_2_weeks_l142_142529

def cost_to_last_for_2_weeks (servings_per_bottle servings_per_day price_per_bottle duration_in_days : ℕ) : ℕ :=
  let total_servings_needed := servings_per_day * duration_in_days
  let bottles_needed := total_servings_needed / servings_per_bottle
  bottles_needed * price_per_bottle

theorem cost_of_iced_coffee_for_2_weeks :
  cost_to_last_for_2_weeks 6 3 3 14 = 21 :=
by
  sorry

end cost_of_iced_coffee_for_2_weeks_l142_142529


namespace track_length_l142_142217

theorem track_length (x : ℕ) 
  (diametrically_opposite : ∃ a b : ℕ, a + b = x)
  (first_meeting : ∃ b : ℕ, b = 100)
  (second_meeting : ∃ s s' : ℕ, s = 150 ∧ s' = (x / 2 - 100 + s))
  (constant_speed : ∀ t₁ t₂ : ℕ, t₁ / t₂ = 100 / (x / 2 - 100)) :
  x = 400 := 
by sorry

end track_length_l142_142217


namespace sufficient_condition_for_p_l142_142703

theorem sufficient_condition_for_p (m : ℝ) (h : 1 < m) : ∀ x : ℝ, x^2 - 2 * x + m > 0 :=
sorry

end sufficient_condition_for_p_l142_142703


namespace coefficient_x2_l142_142630

def expr : Polynomial ℤ := 3 * (2 * Polynomial.X - 2 * Polynomial.X ^ 3) - 5 * (Polynomial.X ^ 2 - Polynomial.X ^ 4 + 2 * Polynomial.X ^ 7) + 4 * (3 * Polynomial.X ^ 2 - 2 * Polynomial.X ^ 6)

theorem coefficient_x2 : expr.coeff 2 = 7 :=
by
  sorry

end coefficient_x2_l142_142630


namespace carla_zoo_l142_142225

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end carla_zoo_l142_142225


namespace chord_length_of_tangent_circles_l142_142620

theorem chord_length_of_tangent_circles 
    (C1 C2 C3 : Type)
    (O1 O2 O3 : Type)
    (r1 r2 r3 : ℝ)
    (h1 : r1 = 3)
    (h2 : r2 = 9)
    (h3 : r3 = 15)
    (T : Type)
    (T1 T2 A B : Type)
    (h4 : collinear [O1, O2, O3])
    (h5 : externally_tangent C1 C2)
    (h6 : internally_tangent C1 C3)
    (h7 : internally_tangent C2 C3) 
    (h8 : chord_of_c3_is_tangent C3 T1 T2) : 
    chord_length C3 A B = 18 :=
sorry

end chord_length_of_tangent_circles_l142_142620


namespace range_x1_x2_l142_142723

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_x1_x2 (a b c d x1 x2 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a + 2 * b + 3 * c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hx1 : f a b c x1 = 0)
  (hx2 : f a b c x2 = 0) :
  abs (x1 - x2) ∈ Set.Ico 0 (2 / 3) :=
sorry

end range_x1_x2_l142_142723


namespace count_multiples_of_7_l142_142360

theorem count_multiples_of_7 (low high : ℕ) (hlow : low = 200) (hhigh : high = 400) : 
  (card {n | low ≤ n ∧ n ≤ high ∧ n % 7 = 0}) = 29 := by
  sorry

end count_multiples_of_7_l142_142360


namespace divisible_by_7_in_range_200_to_400_l142_142374

theorem divisible_by_7_in_range_200_to_400 : 
  ∃ n : ℕ, 
    (∀ (x : ℕ), (200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ finset.range (201)) ∧ finset.card (finset.filter (λ x, 200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0) (finset.range 401)) = 29) := 
begin
  sorry
end

end divisible_by_7_in_range_200_to_400_l142_142374


namespace unique_positive_solution_eq_15_l142_142655

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l142_142655


namespace divisor_exists_l142_142123

theorem divisor_exists : ∀ (s : Finset ℕ),
  (∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) → s.card = 1008 →
  ∃ a b ∈ s, a ∣ b ∧ a ≠ b :=
by
  sorry

end divisor_exists_l142_142123


namespace f_of_2_eq_neg_26_l142_142719

/-- Given function f defined with specific conditions -/
def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

/-- Defining conditions for the problem -/
variables (a b : ℝ)
axiom h1 : f (-2) a b = 10
axiom h2 : ∀ x, f (-x) a b + f (x) a b = -16

/-- Statement to prove -/
theorem f_of_2_eq_neg_26 : f 2 a b = -26 :=
by sorry

end f_of_2_eq_neg_26_l142_142719


namespace weekly_sales_function_maximum_profit_m_range_fruit_store_problem_l142_142174

-- Setting up the conditions as definitions
def cost_price : ℝ := 40
def sales_data : List (ℝ × ℝ) := [(45, 110), (60, 80), (70, 60), (75, 50)]
def selling_price_for_profit_decrease : ℝ := 76
def profit_decrease_condition (x m : ℝ) : Prop := x > selling_price_for_profit_decrease → ∀ w : ℝ, w (x - (cost_price + m)) (-2 * x + 200)

-- First part: Function expression y = -2x + 200
theorem weekly_sales_function (x : ℝ) (y : ℝ) (h1 : (60, 80) ∈ sales_data) (h2 : (70, 60) ∈ sales_data) :
  y = -2 * x + 200 := sorry  -- This is where the proof for finding the function goes

-- Second part: Maximum profit at selling price 70 yuan/kg
theorem maximum_profit (x : ℝ) (max_price : ℝ) (max_profit : ℝ) (h1 : (x - cost_price) * (-2 * x + 200) = max_profit) :
  x = 70 ∧ max_profit = 1800 := sorry  -- This is where the vertex calculation and profit computation goes

-- Third part: Range of values for m
theorem m_range (m : ℝ) (h : profit_decrease_condition (76) m) :
  0 < m ∧ m ≤ 12 := sorry  -- This is where we prove the range for m

-- Assembling the full problem statement
theorem fruit_store_problem :
  (∀ x y, (60, 80) ∈ sales_data → (70, 60) ∈ sales_data → weekly_sales_function x y) ∧
  (∀ x max_price max_profit, (x - cost_price) * (-2 * x + 200) = max_profit → maximum_profit x max_price max_profit) ∧
  (∀ m, profit_decrease_condition (76) m → m_range m) :=
sorry  -- This is where we combine all parts into one theorem

end weekly_sales_function_maximum_profit_m_range_fruit_store_problem_l142_142174


namespace vincent_rope_length_l142_142139

def rope_length : Nat := 72
def pieces_count : Nat := 12
def shortened_length : Nat := 1
def tied_pieces : Nat := 3

theorem vincent_rope_length : 
  let piece_length := rope_length / pieces_count
  let shortened_piece_length := piece_length - shortened_length
  let final_length := shortened_piece_length * tied_pieces
  final_length = 15 := by
  let piece_length := rope_length / pieces_count
  let shortened_piece_length := piece_length - shortened_length
  let final_length := shortened_piece_length * tied_pieces
  show final_length = 15 from sorry

end vincent_rope_length_l142_142139


namespace max_area_circle_eq_l142_142086

theorem max_area_circle_eq (m : ℝ) :
  (x y : ℝ) → (x - 1) ^ 2 + (y + m) ^ 2 = -(m - 3) ^ 2 + 1 → 
  (∃ (r : ℝ), (r = (1 : ℝ)) ∧ (m = 3) ∧ ((x - 1) ^ 2 + (y + 3) ^ 2 = 1)) :=
by
  sorry

end max_area_circle_eq_l142_142086


namespace vertex_coordinates_range_of_y_quadratic_expression_l142_142294

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions for part (1)
def b1 := 4
def c1 := 3

-- First part of the problem
theorem vertex_coordinates :
    ∃ (h k : ℝ), (∀ x : ℝ, quadratic_function (-1) b1 c1 x = - (x - h)^2 + k) ∧ h = 2 ∧ k = 7 :=
begin
    sorry
end

-- Define the interval for the second part
def interval (x : ℝ) := -1 ≤ x ∧ x ≤ 3

-- Second part of the problem
theorem range_of_y :
    ∃ (min max : ℝ), (∀ x : ℝ, interval x → -2 ≤ quadratic_function (-1) b1 c1 x ∧ quadratic_function (-1) b1 c1 x ≤ 7) := 
begin
    sorry
end

-- Given conditions for part (2)
def ymax_left := 2
def ymax_right := 3

-- Third part of the problem
theorem quadratic_expression :
    ∃ b c, (∀ x ≤ 0, quadratic_function (-1) b c x ≤ ymax_left) ∧ 
           (∀ x > 0, quadratic_function (-1) b c x ≤ ymax_right)  ∧
           b = 2 ∧ c = 2 :=
begin
    sorry
end

end vertex_coordinates_range_of_y_quadratic_expression_l142_142294


namespace simplify_fraction_l142_142858

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) : 
  (9 * x^3 * y^2) / (12 * x^2 * y^4) = 9 / 16 := by
  sorry

end simplify_fraction_l142_142858


namespace inscribed_circle_radius_l142_142865

theorem inscribed_circle_radius (R : ℝ) (h : 0 < R) : 
  ∃ x : ℝ, (x = R / 3) :=
by
  -- Given conditions
  have h1 : R > 0 := h

  -- Mathematical proof statement derived from conditions
  sorry

end inscribed_circle_radius_l142_142865


namespace insufficient_information_to_determine_baking_time_l142_142829

-- Defining the given conditions as constants
constant total_cookies : ℕ := 32
constant mixing_time : ℕ := 24
constant eaten_cookies : ℕ := 9
constant leftover_cookies : ℕ := 23

-- The conclusion is that the information is insufficient to determine the baking time
theorem insufficient_information_to_determine_baking_time :
  ∀ baking_time : ℕ,
  ¬ (total_cookies = eaten_cookies + leftover_cookies) → baking_time = _ :=
by
  sorry

end insufficient_information_to_determine_baking_time_l142_142829


namespace books_sold_on_tuesday_l142_142429

theorem books_sold_on_tuesday (total_stock : ℕ) (monday_sold : ℕ) (wednesday_sold : ℕ)
  (thursday_sold : ℕ) (friday_sold : ℕ) (percent_unsold : ℚ) (tuesday_sold : ℕ) :
  total_stock = 1100 →
  monday_sold = 75 →
  wednesday_sold = 64 →
  thursday_sold = 78 →
  friday_sold = 135 →
  percent_unsold = 63.45 →
  tuesday_sold = total_stock - (monday_sold + wednesday_sold + thursday_sold + friday_sold + (total_stock * percent_unsold / 100)) :=
by sorry

end books_sold_on_tuesday_l142_142429


namespace probability_female_selection_l142_142403

theorem probability_female_selection (total_contestants : ℕ) (females : ℕ) (males : ℕ) (selection_size : ℕ) :
  total_contestants = 8 → females = 5 → males = 3 → selection_size = 2 →
  (nat.choose females selection_size) / (nat.choose total_contestants selection_size) = 5 / 14 :=
by
  intros h_total h_females h_males h_selection_size
  rw [h_total, h_females, h_males, h_selection_size]
  sorry

end probability_female_selection_l142_142403


namespace unique_positive_solution_l142_142659

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l142_142659


namespace omega_value_k_range_l142_142729

-- Definitions for Part (1)
def m (ω x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (ω * x), cos (ω * x))
def n (ω x : ℝ) : ℝ × ℝ := (cos (ω * x), -cos (ω * x))
def f (ω x : ℝ) : ℝ := (m ω x).1 * (n ω x).1 + (m ω x).2 * (n ω x).2

-- Part (1) Lean Statement
theorem omega_value (ω : ℝ) (hω : ω > 0) (hT : ∀ x, f ω x = f ω (x + π / (2 * ω))) : ω = 2 := sorry

-- Definitions for Part (2)
variables {a b c x k : ℝ}
def cos_x (a b c x : ℝ) (h : b * b = a * c) : ℝ := (a * a + c * c - a * c) / (2 * a * c)

-- Part (2) Lean Statement
theorem k_range (a b c x k : ℝ) (h : b * b = a * c) :
  -1 < k ∧ k < 0.5 → (∃ x, a > 0 ∧ c > 0 ∧ 0 < x ∧ x ≤ π / 3 ∧ f 2 x = k ∧ f 2 x ≠ k) :=
sorry

end omega_value_k_range_l142_142729


namespace denomination_of_220_coins_l142_142105

-- Given conditions
variables (x : ℕ)
def total_coins := 324
def rupees_to_paise (r : ℕ) := r * 100
def total_value := rupees_to_paise 70
def num_25_paise_coins := total_coins - 220
def total_value_of_25_paise_coins := num_25_paise_coins * 25

-- Mathematical statement to be proved
theorem denomination_of_220_coins :
  220 * x + total_value_of_25_paise_coins = total_value → x = 20 :=
begin
  sorry,
end

end denomination_of_220_coins_l142_142105


namespace equation_line_through_intersections_l142_142343

theorem equation_line_through_intersections (A1 B1 A2 B2 : ℝ)
  (h1 : 2 * A1 + 3 * B1 = 1)
  (h2 : 2 * A2 + 3 * B2 = 1) :
  ∃ (a b c : ℝ), a = 2 ∧ b = 3 ∧ c = -1 ∧ (a * x + b * y + c = 0) := 
sorry

end equation_line_through_intersections_l142_142343


namespace solve_triangle_l142_142006

-- Definition of the triangle with given conditions
structure Triangle where
  a b c : ℝ
  cosA : ℝ
  sinB : ℝ
  cosB : ℝ
  cosC : ℝ
  
def triangle_conditions (T : Triangle) : Prop :=
  √3 * T.cosA + T.a * T.sinB = √3 * T.c ∧
  T.a + 2 * T.c = 6

-- Proof problem: Prove the angle B and minimum value of b
theorem solve_triangle (T : Triangle) (h : triangle_conditions T) : 
  T.sinB = √3 * T.cosB ∧ 
  ∃ b_min, b_min = T.b ∧ T.b = 3*real.sqrt(21)/7 := 
sorry

end solve_triangle_l142_142006


namespace true_proposition_l142_142313

def proposition_p : Prop :=
  ∃ x : ℝ, x^2 - x + 2 < 0

def neg_proposition_p : Prop :=
  ∀ x : ℝ, x^2 - x + 2 ≥ 0

def proposition_q : Prop :=
  ∀ x : ℝ, 1 ≤ x → x ≤ 2 → x^2 ≥ 1

theorem true_proposition : neg_proposition_p ∧ proposition_q := 
by
  have h_neg_p : neg_proposition_p := 
    by
    intro x
    have : (x - 1 / 2) ^ 2 + 7 / 4 ≥ 0 := sq_nonneg (x - 1 / 2)
    linarith
  have h_q : proposition_q := 
    by
    intro x h1 h2
    calc x^2 ≥ 1 : by nlinarith [h1, h2]
  exact ⟨h_neg_p, h_q⟩

end true_proposition_l142_142313


namespace subsets_count_is_four_l142_142501

noncomputable def number_of_subsets_of_special_set : Nat := 
  let S := {x | x^2 - 1 = 0}
  fintype.card (set_subtype S)

theorem subsets_count_is_four : number_of_subsets_of_special_set = 4 := by
  sorry

end subsets_count_is_four_l142_142501


namespace biography_increase_l142_142930

theorem biography_increase (B N : ℝ) (hN : N = 0.35 * (B + N) - 0.20 * B):
  (N / (0.20 * B) * 100) = 115.38 :=
by
  sorry

end biography_increase_l142_142930


namespace natural_numbers_divisible_by_7_between_200_400_l142_142364

theorem natural_numbers_divisible_by_7_between_200_400 : 
  { n : ℕ | 200 <= n ∧ n <= 400 ∧ n % 7 = 0 }.to_finset.card = 29 := 
  sorry

end natural_numbers_divisible_by_7_between_200_400_l142_142364


namespace subset_intersection_l142_142769

theorem subset_intersection (S : Set (α : Type)) (n : ℕ)
  (h1 : S.card = n)
  (h2 : ∃ (T : Finset (Finset α)), T.card = 2^(n-1) ∧ (∀ (A B C : Finset α), A ∈ T → B ∈ T → C ∈ T → (A ∩ B ∩ C).nonempty))
  : ∃ (a : α), ∀ (A : Finset α), A ∈ T → a ∈ A := sorry

end subset_intersection_l142_142769


namespace range_of_a_if_p_true_range_of_a_if_p_or_q_l142_142337

noncomputable def f (a x : ℝ) : ℝ := Real.log ((a^2 - 1) * x^2 + (a - 1) * x + 1)

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, (a^2 - 1) * x^2 + (a - 1) * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f a x = y

theorem range_of_a_if_p_true (a : ℝ) : prop_p a → (a ∈ Ico (-∞) (-5/3) ∪ Ici 1) :=
sorry

theorem range_of_a_if_p_or_q (a : ℝ) : (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) → (a ∈ Iic (-1) ∪ Ici 1) :=
sorry

end range_of_a_if_p_true_range_of_a_if_p_or_q_l142_142337


namespace points_on_line_is_sufficient_but_not_necessary_l142_142560

theorem points_on_line_is_sufficient_but_not_necessary (a_n : ℕ → ℕ) :
  (∀ n : ℕ, a_n = n + 1) → (∃ d : ℤ, ∀ n : ℕ, a_n = a_n + d)
  ∧ ¬ ( ∀ n : ℕ, (∃ d : ℤ, ∀ n : ℕ, a_n = a_n + d) → (a_n = n + 1)) :=
by
  sorry

end points_on_line_is_sufficient_but_not_necessary_l142_142560


namespace quadratic_trinomial_sum_zero_discriminants_l142_142068

theorem quadratic_trinomial_sum_zero_discriminants (a b c : ℝ) :
  ∃ p q : ℝ → ℝ, (∀ x : ℝ, a * x ^ 2 + b * x + c = p x + q x) ∧
    (∀ x : ℝ, (let Δp := (d2p - d1p) ^ 2 - 4 * d0p * p2 in Δp = 0 ∧ 
                     Δq := (d2q - d1q) ^ 2 - 4 * d0q * q2 in Δq = 0)) :=
sorry

end quadratic_trinomial_sum_zero_discriminants_l142_142068


namespace polyhedron_volume_l142_142209

-- Define the conditions
variables (A A₁ B C₁ D₁ O C D: ℝ)
variables (AB : ℝ) (AA₁ : ℝ)

-- Provide known values
def cond1 : Prop := AA₁ = OC₁
def cond2 : Prop := 45° = 45° -- This means the dihedral angle is 45 degrees
def cond3 : Prop := AB = 1

-- Statement to be proved
theorem polyhedron_volume (H1 : cond1) (H2 : cond2) (H3 : cond3) : 
  let V := (polyhedron_volume A A₁ B C₁ D₁)
  V = (√2 / 2) :=
sorry

end polyhedron_volume_l142_142209


namespace sum_base8_and_convert_to_base10_l142_142664

theorem sum_base8_and_convert_to_base10 :
  let a := 2 * 8^2 + 4 * 8^1 + 5 * 8^0,
      b := 1 * 8^2 + 5 * 8^1 + 7 * 8^0,
      sum_base8 := 4 * 8^2 + 2 * 8^1 + 4 * 8^0,
      sum_base10 := 276 in
  (a + b = 276) ∧ sum_base8 = 4 * 8^2 + 2 * 8^1 + 4 * 8^0 ∧ sum_base10 = 276 :=
by {
  sorry
}

end sum_base8_and_convert_to_base10_l142_142664


namespace binary_addition_correct_l142_142606

theorem binary_addition_correct :
  (binary_to_nat "1101" + binary_to_nat "101" + binary_to_nat "111" + binary_to_nat "10001") = binary_to_nat "101010" :=
by
  sorry

end binary_addition_correct_l142_142606


namespace unique_positive_solution_eq_15_l142_142653

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l142_142653


namespace smallest_integer_proof_l142_142458

noncomputable def smallest_n : ℕ := 13

theorem smallest_integer_proof :
  ∃ (m : ℝ) (n : ℕ) (r : ℝ), n = smallest_n ∧ r > 0 ∧ r < 1/500 ∧ m = (n + r)^3 :=
by
  let n := smallest_n
  use 2198 -- m
  use n -- n
  use 1/507 -- r, chosen to be < 1/500
  split
  . exact rfl  -- n = smallest_n
  split
  . norm_num -- r > 0
  split
  . norm_num -- r < 1 / 500
  . norm_num -- verify m = (n + r)^3 holds with these values
  sorry

end smallest_integer_proof_l142_142458


namespace triangle_angle_contradiction_l142_142472

theorem triangle_angle_contradiction (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) :
  false :=
by
  have h : α + β + γ > 180 := by
  { linarith }
  linarith

end triangle_angle_contradiction_l142_142472


namespace integer_side_lengths_triangle_l142_142716

theorem integer_side_lengths_triangle :
  ∃ (a b c : ℤ), (abc = 2 * (a - 1) * (b - 1) * (c - 1)) ∧
            (a = 8 ∧ b = 7 ∧ c = 3 ∨ a = 6 ∧ b = 5 ∧ c = 4) := 
by
  sorry

end integer_side_lengths_triangle_l142_142716


namespace sum_of_angles_of_roots_l142_142886

theorem sum_of_angles_of_roots :
  let cis θ := complex.exp (θ * complex.I) in
  ∃ θ_1 θ_2 θ_3 θ_4 θ_5 : ℝ,
    (0 ≤ θ_1 ∧ θ_1 < 360 ∧
     0 ≤ θ_2 ∧ θ_2 < 360 ∧
     0 ≤ θ_3 ∧ θ_3 < 360 ∧
     0 ≤ θ_4 ∧ θ_4 < 360 ∧
     0 ≤ θ_5 ∧ θ_5 < 360 ∧
     complex.exp (5 * θ_1 * complex.I) = -1 / real.sqrt 2 + (1 / real.sqrt 2) * complex.I ∧
     complex.exp (5 * θ_2 * complex.I) = -1 / real.sqrt 2 + (1 / real.sqrt 2) * complex.I ∧
     complex.exp (5 * θ_3 * complex.I) = -1 / real.sqrt 2 + (1 / real.sqrt 2) * complex.I ∧
     complex.exp (5 * θ_4 * complex.I) = -1 / real.sqrt 2 + (1 / real.sqrt 2) * complex.I ∧
     complex.exp (5 * θ_5 * complex.I) = -1 / real.sqrt 2 + (1 / real.sqrt 2) * complex.I) →
  θ_1 + θ_2 + θ_3 + θ_4 + θ_5 = 1125 := by
  sorry

end sum_of_angles_of_roots_l142_142886


namespace relation_between_x_and_y_l142_142393

noncomputable def x : ℝ := 2 + Real.sqrt 3
noncomputable def y : ℝ := 1 / (2 - Real.sqrt 3)

theorem relation_between_x_and_y : x = y := sorry

end relation_between_x_and_y_l142_142393


namespace total_meat_supply_l142_142400

-- Definitions of the given conditions
def lion_consumption_per_day : ℕ := 25
def tiger_consumption_per_day : ℕ := 20
def duration_days : ℕ := 2

-- Statement of the proof problem
theorem total_meat_supply :
  (lion_consumption_per_day + tiger_consumption_per_day) * duration_days = 90 :=
by
  sorry

end total_meat_supply_l142_142400


namespace group_A_percentage_l142_142407

/-!
In an examination, there are 100 questions divided into 3 groups A, B, and C such that each group contains at least one question. 
Each question in group A carries 1 mark, each question in group B carries 2 marks, and each question in group C carries 3 marks. 
It is known that:
- Group B contains 23 questions
- Group C contains 1 question.
Prove that the percentage of the total marks that the questions in group A carry is 60.8%.
-/

theorem group_A_percentage :
  ∃ (a b c : ℕ), b = 23 ∧ c = 1 ∧ (a + b + c = 100) ∧ ((a * 1) + (b * 2) + (c * 3) = 125) ∧ ((a : ℝ) / 125 * 100 = 60.8) :=
by
  sorry

end group_A_percentage_l142_142407


namespace total_selections_eq_525_l142_142283

open Finset

noncomputable def count_valid_selections : ℕ :=
  let S := range 21 \ {0}
  let quadruples := { x ∈ (S.product (S.product (S.product S))).to_finset | 
                    let a := x.1
                    let b := x.2.1
                    let c := x.2.2.1
                    let d := x.2.2.2
                    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧ (a + c = b + d) 
                  }
  quadruples.card / 2

theorem total_selections_eq_525 : count_valid_selections = 525 :=
  sorry

end total_selections_eq_525_l142_142283


namespace probability_two_doors_open_l142_142998

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_doors_open :
  let total_doors := 5
  let total_combinations := 2 ^ total_doors
  let favorable_combinations := binomial total_doors 2
  let probability := favorable_combinations / total_combinations
  probability = 5 / 16 :=
by
  sorry

end probability_two_doors_open_l142_142998


namespace julie_money_left_after_purchase_l142_142803

-- Definitions based on the problem conditions
def cost_of_bike : ℕ := 2345
def savings : ℕ := 1500
def lawns : ℕ := 20
def newspapers : ℕ := 600
def dogs : ℕ := 24
def pay_per_lawn : ℕ := 20
def pay_per_newspaper : ℝ := 0.40
def pay_per_dog : ℕ := 15

-- Lean theorem stating the final desired result
theorem julie_money_left_after_purchase :
  let total_earnings := lawns * pay_per_lawn + newspapers * (pay_per_newspaper.to_nat) + dogs * pay_per_dog,
      total_money := savings + total_earnings,
      money_left := total_money - cost_of_bike
  in
  money_left = 155 := 
by {
  sorry
}

end julie_money_left_after_purchase_l142_142803


namespace vincent_rope_length_l142_142138

def rope_length : Nat := 72
def pieces_count : Nat := 12
def shortened_length : Nat := 1
def tied_pieces : Nat := 3

theorem vincent_rope_length : 
  let piece_length := rope_length / pieces_count
  let shortened_piece_length := piece_length - shortened_length
  let final_length := shortened_piece_length * tied_pieces
  final_length = 15 := by
  let piece_length := rope_length / pieces_count
  let shortened_piece_length := piece_length - shortened_length
  let final_length := shortened_piece_length * tied_pieces
  show final_length = 15 from sorry

end vincent_rope_length_l142_142138


namespace sum_of_squares_of_elements_of_matrix_is_two_l142_142749

variable {R : Type} [Field R] (a b c d : R)

def matrix_A := ![![a, b], ![c, d]]
def transpose_A := ![![a, c], ![b, d]]
def inverse_A := ![![1 / (a * d - b * c) * d, -((1 / (a * d - b * c)) * b)], 
                     ![-(1 / (a * d - b * c)) * c, (1 / (a * d - b * c)) * a]]

-- The main statement to prove
theorem sum_of_squares_of_elements_of_matrix_is_two
    (h1 : transpose_A = inverse_A)
    : a^2 + b^2 + c^2 + d^2 = 2 := 
by sorry

end sum_of_squares_of_elements_of_matrix_is_two_l142_142749


namespace triangle_ABC_right_l142_142448

-- Define the conditions
variables (r : ℝ) (A B C : ℝ×ℝ)
variable (h1 : dist A C = 3 * r)
variable (h2 : dist B C = 4 * r)
variable (h3 : dist A B = 2 * r)  -- AB is the diameter of the semicircle
variable (h4 : (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = (2 * r) ^ 2) -- Distance between A and B is 2r

-- Define the theorem
theorem triangle_ABC_right (h1 : dist A C = 3 * r) (h2 : dist B C = 4 * r) (h3 : dist A B = 2 * r)
: (dist A B = 5 * r) ∧ is_right_triangle A B C :=
by
  sorry

end triangle_ABC_right_l142_142448


namespace solve_for_x_l142_142553

theorem solve_for_x (x : ℤ) (h : x + 1 = 4) : x = 3 :=
sorry

end solve_for_x_l142_142553


namespace no_rectangle_other_than_square_exists_l142_142009

theorem no_rectangle_other_than_square_exists : 
  ∃ (p1 p2 p3 p4 : ℕ), 
  (prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4) ∧ 
  (p1 % 2 ≠ 0) ∧ (p2 % 2 ≠ 0) ∧ (p3 % 2 ≠ 0) ∧ (p4 % 2 ≠ 0) ∧ 
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) → 
  ¬ ∃ (segments : ℕ → ℝ) (rectangle_segments : ℕ → ℝ), 
    (∀ i < 100, segments i = 1 / p1 ^ i) ∧
    (∀ i < 100, segments (i + 100) = 1 / p2 ^ i) ∧
    (∀ i < 100, segments (i + 200) = 1 / p3 ^ i) ∧
    (∀ i < 100, segments (i + 300) = 1 / p4 ^ i) ∧
    ( ∀ (i j : ℕ), i ≠ j → rectangle_segments i + rectangle_segments j ≠ 1) :=
sorry

end no_rectangle_other_than_square_exists_l142_142009


namespace cone_volume_l142_142327

theorem cone_volume (l : ℝ) (circumference : ℝ) (radius : ℝ) (height : ℝ) (volume : ℝ) 
  (h1 : l = 8) 
  (h2 : circumference = 6 * Real.pi) 
  (h3 : radius = circumference / (2 * Real.pi))
  (h4 : height = Real.sqrt (l^2 - radius^2)) 
  (h5 : volume = (1 / 3) * Real.pi * radius^2 * height) :
  volume = 3 * Real.sqrt 55 * Real.pi := 
  by 
    sorry

end cone_volume_l142_142327


namespace right_triangle_third_side_l142_142757

theorem right_triangle_third_side {m n : ℝ} (h : |m - 3| + sqrt (n - 4) = 0) 
  (right_triangle : ∃ a b c : ℝ, m = a ∧ n = b ∧ right_triangle a b c) : 
  (∃ side : ℝ, (side = 5 ∨ side = sqrt 7)) := 
sorry

end right_triangle_third_side_l142_142757


namespace average_minutes_heard_by_audience_l142_142979

theorem average_minutes_heard_by_audience :
  let attendees := 100
  let duration := 70
  let full := 0.3 * attendees
  let sleep := 0.05 * attendees
  let remaining := attendees - full - sleep
  let one_fourth := 0.4 * remaining
  let two_thirds := 0.6 * remaining
  let total_minutes := (full * duration) + (sleep * 0) + (one_fourth * (duration / 4)) + (two_thirds * (2 * duration / 3))
  let average_minutes := total_minutes / attendees
  in average_minutes = 44 := by
  let attendees := 100
  let duration := 70
  let full := 0.3 * attendees
  let sleep := 0.05 * attendees
  let remaining := attendees - full - sleep
  let one_fourth := 0.4 * remaining
  let two_thirds := 0.6 * remaining
  let total_minutes := (full * duration) + (sleep * 0) + (one_fourth * (duration / 4)) + (two_thirds * (2 * duration / 3))
  have h1 : total_minutes = 4375 := by sorry
  have h2 : average_minutes = total_minutes / attendees := by sorry
  have h3 : average_minutes = 44 := by sorry
  exact h3

end average_minutes_heard_by_audience_l142_142979


namespace service_fee_calculation_l142_142012

-- Problem definitions based on conditions
def cost_food : ℝ := 50
def tip : ℝ := 5
def total_spent : ℝ := 61
def service_fee_percentage (x : ℝ) : Prop := x = (12 / 50) * 100

-- The main statement to be proven, showing that the service fee percentage is 24%
theorem service_fee_calculation : service_fee_percentage 24 :=
by {
  sorry
}

end service_fee_calculation_l142_142012


namespace area_sum_triangles_l142_142190

def vertices : List (ℝ × ℝ × ℝ) := [
  (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
  (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
  (0, 0, 2), (1, 0, 2), (0, 1, 2), (1, 1, 2)
]

def face_areas : List ℝ := List.replicate 16 (1 / 2) ++ List.replicate 16 1 ++ List.replicate 8 2

noncomputable def sum_face_areas : ℝ := face_areas.sum

theorem area_sum_triangles (v : List (ℝ × ℝ × ℝ)) (face_areas : List ℝ)
  (h : v = vertices ∧ face_areas = List.replicate 16 (1 / 2) ++ List.replicate 16 1 ++ List.replicate 8 2) :
  sum_face_areas = 40 := by
  simp [sum_face_areas]
  sorry

end area_sum_triangles_l142_142190


namespace topsoil_cost_l142_142527

-- Definitions
constant yard_to_cubic_feet : ℕ := 27
constant cost_per_cubic_foot : ℕ := 8
constant volume_in_yards : ℕ := 7

-- Proof statement
theorem topsoil_cost :
  let volume_in_cubic_feet := volume_in_yards * yard_to_cubic_feet in
  let total_cost := volume_in_cubic_feet * cost_per_cubic_foot in
  total_cost = 1512 :=
by
  sorry

end topsoil_cost_l142_142527


namespace perimeter_triangle_FEN_l142_142957

theorem perimeter_triangle_FEN (MB KC BK CF MK KF : ℕ) 
  (h1 : MB = 4) 
  (h2 : KC = 4) 
  (h3 : BK = 3) 
  (h4 : CF = 3) 
  (h5 : MK = 5) 
  (h6 : KF = 5) :
  let FEN_perimeter := 8 in
  (FEN_perimeter = 8) := 
by
  sorry

end perimeter_triangle_FEN_l142_142957


namespace abs_eq_4_reciprocal_eq_self_l142_142885

namespace RationalProofs

-- Problem 1
theorem abs_eq_4 (x : ℚ) : |x| = 4 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Problem 2
theorem reciprocal_eq_self (x : ℚ) : x ≠ 0 → x⁻¹ = x ↔ x = 1 ∨ x = -1 :=
by sorry

end RationalProofs

end abs_eq_4_reciprocal_eq_self_l142_142885


namespace minimum_value_expression_l142_142690

theorem minimum_value_expression 
  (x y z : ℝ) 
  (h1 : x ≥ 3) (h2 : y ≥ 3) (h3 : z ≥ 3) :
  let A := (x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3) in
  let B := x * y + y * z + z * x in
  A / B ≥ 1 :=
sorry

end minimum_value_expression_l142_142690


namespace fraction_equals_i_l142_142321

theorem fraction_equals_i (m n : ℝ) (i : ℂ) (h : i * i = -1) (h_cond : m * (1 + i) = (11 + n * i)) :
  (m + n * i) / (m - n * i) = i :=
sorry

end fraction_equals_i_l142_142321


namespace max_integer_value_l142_142392

open Real

theorem max_integer_value (x : ℝ) : 
    let expr := (4 * x^2 + 12 * x + 23) / (4 * x^2 + 12 * x + 9) 
    in expr ≤ 7 :=
begin
  let expr := (4 * x^2 + 12 * x + 23) / (4 * x^2 + 12 * x + 9),
  calc
    expr = 1 + 14 / (4 * x^2 + 12 * x + 9) : sorry  -- Calculation step as in solution
    ... ≤ 1 + 14 / (9 / 4) : sorry                -- Minimum value step
    ... = 65 / 9 : sorry                          -- Evaluation step
    ... ≤ 7 : by { norm_num },                    -- Comparison step
end

end max_integer_value_l142_142392


namespace vertex_coordinates_range_of_y_quadratic_expression_l142_142292

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions for part (1)
def b1 := 4
def c1 := 3

-- First part of the problem
theorem vertex_coordinates :
    ∃ (h k : ℝ), (∀ x : ℝ, quadratic_function (-1) b1 c1 x = - (x - h)^2 + k) ∧ h = 2 ∧ k = 7 :=
begin
    sorry
end

-- Define the interval for the second part
def interval (x : ℝ) := -1 ≤ x ∧ x ≤ 3

-- Second part of the problem
theorem range_of_y :
    ∃ (min max : ℝ), (∀ x : ℝ, interval x → -2 ≤ quadratic_function (-1) b1 c1 x ∧ quadratic_function (-1) b1 c1 x ≤ 7) := 
begin
    sorry
end

-- Given conditions for part (2)
def ymax_left := 2
def ymax_right := 3

-- Third part of the problem
theorem quadratic_expression :
    ∃ b c, (∀ x ≤ 0, quadratic_function (-1) b c x ≤ ymax_left) ∧ 
           (∀ x > 0, quadratic_function (-1) b c x ≤ ymax_right)  ∧
           b = 2 ∧ c = 2 :=
begin
    sorry
end

end vertex_coordinates_range_of_y_quadratic_expression_l142_142292


namespace calculate_range_of_a_l142_142222

variable (a : ℝ)
def p := ∀ x : ℝ, x ∈ set.Icc (1 : ℝ) 2 → x^2 - a ≥ 0
def q := ∃ x0 : ℝ, ∀ x : ℝ, x + (a - 1) * x0 + 1 < 0

theorem calculate_range_of_a (hpq_true : p a ∨ q a) (hpq_false : ¬(p a ∧ q a)) : a > 3 ∨ -1 ≤ a ∧ a ≤ 1 := sorry

end calculate_range_of_a_l142_142222


namespace sphere_radius_in_unit_cube_l142_142911

theorem sphere_radius_in_unit_cube : ∃ r : ℝ, r = 1 / 2 ∧
  (∃ (spheres : Set ℝ^3), spheres.card = 12 ∧ 
  (∃ v : ℝ^3, v ∈ spheres ∧ v = (0,0,0) ∧
    (∀ s ∈ spheres, s ≠ v → 
      (∃ f : ℝ^3 → Prop, (f (1,0,0) ∧ f (0,1,0) ∧ f (0,0,1)) ∧ 
          ∀ x ∈ spheres, x ≠ v → 
            let d := dist x v in 
              d = 2 * r ∧ f x ) ) ) )
sorry

end sphere_radius_in_unit_cube_l142_142911


namespace sam_annual_income_l142_142765

theorem sam_annual_income
  (q : ℝ) (I : ℝ)
  (h1 : 30000 * 0.01 * q + 15000 * 0.01 * (q + 3) + (I - 45000) * 0.01 * (q + 5) = (q + 0.35) * 0.01 * I) :
  I = 48376 := 
sorry

end sam_annual_income_l142_142765


namespace sin_2_alpha_beta_eq_l142_142725

noncomputable def sin2_alpha_beta (p : ℝ) (α β : ℝ) : ℝ :=
  2 * real.sin (α + β) * real.cos (α + β)

theorem sin_2_alpha_beta_eq (p α β : ℝ) 
  (h1 : 1 = real.tan α ^ 2 - 4 * p * real.tan α - 2)
  (h2 : 1 = real.tan β ^ 2 - 4 * p * real.tan β - 2) :
  sin2_alpha_beta p α β = (2 * p) / (p ^ 2 + 1) :=
begin
  sorry
end

end sin_2_alpha_beta_eq_l142_142725


namespace probability_within_range_l142_142715

noncomputable def ξ : ℝ → ℝ → ℝ → ℝ := sorry -- (Here, we'll assume a normal distribution function for ξ)
variables {σ : ℝ} (h₁ : ξ ~ N(1, σ^2))
variables {P : set ℝ → ℝ} (h₂ : P {x : ℝ | x > 3} = 0.023)

theorem probability_within_range :
  P {x : ℝ | -1 ≤ x ∧ x ≤ 3} = 0.954 :=
by
  sorry

end probability_within_range_l142_142715


namespace ratio_perimeters_eq_3_div_2_l142_142160

-- Define the side length of the smaller square
def s : ℝ := sorry

-- Define the diagonal of the smaller square
def diagonal_smaller_square : ℝ := s * Real.sqrt 2

-- Define the diagonal of the larger square
def diagonal_larger_square : ℝ := 1.5 * diagonal_smaller_square

-- Define the side length of the larger square
def side_larger_square : ℝ := diagonal_larger_square / Real.sqrt 2

-- Define the perimeter of the smaller square
def perimeter_smaller_square : ℝ := 4 * s

-- Define the perimeter of the larger square
def perimeter_larger_square : ℝ := 4 * side_larger_square

-- Prove that the ratio of the perimeters is 3/2
theorem ratio_perimeters_eq_3_div_2 :
  (perimeter_larger_square / perimeter_smaller_square) = (3 / 2) :=
by
  sorry

end ratio_perimeters_eq_3_div_2_l142_142160


namespace cyclic_hexagon_ratio_theorem_l142_142439

noncomputable theory

variables {A B C D E F X Y Z : Point}
variables [cyclic_hexagon A B C D E F]
variables (X_def : X = AB ∩ DE)
variables (Y_def : Y = BC ∩ EF)
variables (Z_def : Z = CD ∩ FA)
variables [non_degenerate_hexagon A B C D E F]

theorem cyclic_hexagon_ratio_theorem
  (no_opposite_sides_parallel : ∀ P Q R S : Point, ¬ (parallel (line_through P Q) (line_through R S)))
  : ∀ (XY XZ BE AD : ℝ)
    (angle_A angle_B angle_D angle_E : ℝ), 
  ∃ (sin_B_E sin_A_D : ℝ),
    |XY / XZ| = |BE / AD| * |sin (angle_B - angle_E) / sin (angle_A - angle_D)| := 
sorry

end cyclic_hexagon_ratio_theorem_l142_142439


namespace complement_union_correct_l142_142894

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_correct :
  ((U \ A) ∪ B) = {0, 2, 3, 6} :=
by
  sorry

end complement_union_correct_l142_142894


namespace trigonometric_identity_l142_142938

theorem trigonometric_identity :
  cos 28 * cos 17 - sin 28 * cos 73 = sqrt 2 / 2 :=
by 
  sorry

end trigonometric_identity_l142_142938


namespace angle_bisectors_form_inscribed_quadrilateral_l142_142845

-- Definitions for basic angle representation and quadrilateral
structure Quadrilateral (α : Type) :=
  (A B C D : α)

structure AngleBisector where
  quadrilateral : Quadrilateral ℝ
  bisectorA bisectorB bisectorC bisectorD : ℝ

def isConvex (Q : Quadrilateral ℝ) : Prop := sorry

def sumInternalAngles (Q : Quadrilateral ℝ) : ℝ :=
  let ⟨A, B, C, D⟩ := Q
  (D - A) + (B - A) + (C - D) + (D - C)

def isCyclic (Q : Quadrilateral ℝ) (a b c d : ℝ) : Prop :=
  let α := (a + b) 
  let β := (c + d) 
  α + β = 180

-- Given conditions based on the problem
axiom quadrilateral_convex (Q : Quadrilateral ℝ) : isConvex Q

axiom quadrilateral_internal_angle_sum (Q : Quadrilateral ℝ) : sumInternalAngles Q = 360

-- The main theorem statement we need to prove
theorem angle_bisectors_form_inscribed_quadrilateral (Q : Quadrilateral ℝ) (a b c d : ℝ) (h: AngleBisector) :
  isConvex Q →
  sumInternalAngles Q = 360 →
  isCyclic Q a b c d :=
begin
  sorry
end

end angle_bisectors_form_inscribed_quadrilateral_l142_142845


namespace correct_propositions_count_l142_142504

theorem correct_propositions_count (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0) ∧ -- original proposition
  (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0) ∧ -- converse proposition
  (¬(x ≠ 0 ∨ y ≠ 0) ∨ x^2 + y^2 = 0) ∧ -- negation proposition
  (¬(x^2 + y^2 = 0) ∨ x ≠ 0 ∨ y ≠ 0) -- inverse proposition
  := by
  sorry

end correct_propositions_count_l142_142504


namespace probability_less_than_8000_l142_142511

def city : Type := { Bangok, CapeTown, Honolulu, London }

def distance (c1 c2 : city) : ℕ :=
  match (c1, c2) with
  | (Bangkok, CapeTown) => 6400
  | (Bangkok, Honolulu) => 7850
  | (Bangkok, London) => 6100
  | (CapeTown, Honolulu) => 12000
  | (CapeTown, London) => 6150
  | (Honolulu, London) => 7500
  | (a, b) => distance b a -- symmetric
  | _ => 0

def pairs : List (city × city) := [(Bangkok, CapeTown), (Bangkok, Honolulu), (Bangkok, London), (CapeTown, Honolulu), (CapeTown, London), (Honolulu, London)]

def qualifying_pairs : ℕ := List.length (List.filter (λ p => distance p.fst p.snd < 8000) pairs)

theorem probability_less_than_8000 :
  qualifying_pairs = 5 → pairs.length = 6 → qualifying_pairs / pairs.length = 5 / 6 := by
  sorry

end probability_less_than_8000_l142_142511


namespace arrival_time_at_work_l142_142464

-- Conditions
def pickup_time : String := "06:00"
def travel_to_station : ℕ := 40 -- in minutes
def travel_from_station : ℕ := 140 -- in minutes

-- Prove the arrival time is 9:00 a.m.
theorem arrival_time_at_work : 
  (arrival_time : String) :=
by
  -- Assume the conditions
  let initial_time := "06:00"
  let first_station_time := "06:40"
  let work_arrival_time := "09:00"
  sorry -- Proof

end arrival_time_at_work_l142_142464


namespace quadratic_trinomial_decomposition_l142_142066

theorem quadratic_trinomial_decomposition (a b c : ℝ) :
  ∃ g h : ℝ → ℝ, (∀ x : ℝ, a*x^2 + b*x + c = g x + h x) ∧
                 (∃ g_discriminant, g_discriminant = 0) ∧
                 (∃ h_discriminant, h_discriminant = 0) :=
begin
  sorry
end

end quadratic_trinomial_decomposition_l142_142066


namespace sandy_walk_distance_l142_142854

theorem sandy_walk_distance (d : ℝ) : 
  let initial_position := (0 : ℝ, 0 : ℝ) in
  let south_walk := (0, -d) in
  let east_walk_1 := (d, -d) in
  let north_walk := (d, 0) in
  let east_walk_2 := (2 * d, 0) in
  (east_walk_2 = (2 * d, 40)) → d = 40 :=
by
  intro h
  sorry

end sandy_walk_distance_l142_142854


namespace geometric_sequence_a5_l142_142709

variable (a : ℕ → ℝ) (q : ℝ)

axiom pos_terms : ∀ n, a n > 0

axiom a1a3_eq : a 1 * a 3 = 16
axiom a3a4_eq : a 3 + a 4 = 24

theorem geometric_sequence_a5 :
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) → a 5 = 32 :=
by
  sorry

end geometric_sequence_a5_l142_142709


namespace tigers_losses_l142_142078

theorem tigers_losses (L T : ℕ) (h1 : 56 = 38 + L + T) (h2 : T = L / 2) : L = 12 :=
by sorry

end tigers_losses_l142_142078


namespace evaluate_expression_l142_142640

theorem evaluate_expression (k : ℤ): 
  2^(-(3*k+1)) - 2^(-(3*k-2)) + 2^(-(3*k)) - 2^(-(3*k+3)) = -((21:ℚ)/(8:ℚ)) * 2^(-(3*k)) := 
by 
  sorry

end evaluate_expression_l142_142640


namespace BN_plus_CN_eq_AN_l142_142798

open EuclideanGeometry
open Real

variables {A B C D I M P N : Point} (hABC : Triangle A B C)

-- Given conditions
variables (hAD_bisector : is_angle_bisector A D)
variables (hD_circumcircle : on_circumcircle D (Triangle A B C))
variables (hI_incenter : incenter I (Triangle A B C))
variables (hM_midpoint : is_midpoint M B C)
variables (hP_reflection : reflection I M P)
variables (hPD_extends_N : on_circumcircle N (Triangle A B C) ∧ intersects (line_through D P) N)
variables (hN_arc_BC : on_arc N B C (circumcircle (Triangle A B C)))

-- Question to prove
theorem BN_plus_CN_eq_AN : distance B N + distance C N = distance A N := sorry

end BN_plus_CN_eq_AN_l142_142798


namespace find_pairs_l142_142934

noncomputable def pairs_of_real_numbers (α β : ℝ) := 
  ∀ x y z w : ℝ, 0 < x → 0 < y → 0 < z → 0 < w →
    (x + y^2 + z^3 + w^6 ≥ α * (x * y * z * w)^β)

theorem find_pairs (α β : ℝ) :
  (∃ x y z w : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
    (x + y^2 + z^3 + w^6 = α * (x * y * z * w)^β))
  →
  pairs_of_real_numbers α β :=
sorry

end find_pairs_l142_142934


namespace min_value_of_a_is_five_l142_142840

-- Given: a, b, c in table satisfying the conditions
-- We are to prove that the minimum value of a is 5.
theorem min_value_of_a_is_five
  {a b c: ℤ} (h_pos: 0 < a) (hx_distinct: 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
                               a*x₁^2 + b*x₁ + c = 0 ∧ 
                               a*x₂^2 + b*x₂ + c = 0) (hb_neg: b < 0) 
                               (h_disc_pos: (b^2 - 4*a*c) > 0) : a = 5 :=
sorry

end min_value_of_a_is_five_l142_142840


namespace evaluate_fraction_l142_142638

theorem evaluate_fraction : (8 / 29) - (5 / 87) = (19 / 87) := sorry

end evaluate_fraction_l142_142638


namespace range_of_a_for_zero_point_l142_142287

theorem range_of_a_for_zero_point (a : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = x^2 - 2 * x + a)
  (h₂ : ∃ x ∈ set.Ioo (-1 : ℝ) (3 : ℝ), f x = 0) :
  (-3 : ℝ) < a ∧ a ≤ (1 : ℝ) :=
sorry

end range_of_a_for_zero_point_l142_142287


namespace total_distance_l142_142155

theorem total_distance (D : ℕ) 
  (h1 : (1 / 2 * D : ℝ) + (1 / 4 * (1 / 2 * D : ℝ)) + 105 = D) : 
  D = 280 :=
by
  sorry

end total_distance_l142_142155


namespace least_possible_value_l142_142140

theorem least_possible_value (x y : ℝ) : ∃ x y : ℝ, (xy - 2)^2 + (x + y - 1)^2 = 0 :=
by {
  use [(2 : ℝ), (-1 : ℝ)],
  use [(-1 : ℝ), (2 : ℝ)],
  sorry
} 

end least_possible_value_l142_142140


namespace reconstruct_quadrilateral_l142_142697

-- Define the points P', Q', R', S'
variables (P' Q' R' S' : ℝ)

-- Define the unknowns x, y, z, w as real numbers
variables (x y z w : ℝ)

-- State the given conditions
def conditions (P Q R S P' Q' R' S' : ℝ) : Prop :=
  (∀ P Q, P'Q = 2 * P Q) ∧
  (∀ Q R, QR' = QR) ∧
  (∀ R S, SR' = SR) ∧
  (∀ S P, PS' = 3 * P S)

-- State the question with the correct ordered quadruple (x, y, z, w)
noncomputable def ordered_quadruple_correct (x y z w : ℝ) : Prop :=
  (x = 48 / 95) ∧
  (y = 32 / 95) ∧
  (z = 19 / 95) ∧
  (w = 4 / 5)

-- The final proof statement
theorem reconstruct_quadrilateral (P Q R S P' Q' R' S' : ℝ) : 
  conditions P Q R S P' Q' R' S' →
  ordered_quadruple_correct x y z w :=
by {
  sorry
}

end reconstruct_quadrilateral_l142_142697


namespace perfect_squares_count_2000_l142_142741

theorem perfect_squares_count_2000 : 
  let count := (1 to 44).filter (λ x, 
    let ones_digit := (x * x) % 10 in 
    ones_digit = 4 ∨ ones_digit = 5 ∨ ones_digit = 6).length
  in
  count = 22 :=
by
  sorry

end perfect_squares_count_2000_l142_142741


namespace new_person_age_l142_142157

theorem new_person_age
  (initial_group_size : ℕ)
  (initial_avg_age decreased_avg_age : ℝ)
  (age_of_replaced_person : ℝ)
  (decrease_in_avg : ℝ)
  (initial_group_size = 10)
  (decrease_in_avg = 3)
  (age_of_replaced_person = 44)
  (decreased_avg_age + decrease_in_avg = initial_avg_age)
  (initial_avg_age * 10 = decreased_avg_age * 10 + age_of_replaced_person - (initial_group_size - 1) * decreased_avg_age) :
  ∃ (new_person_age : ℝ), new_person_age = 14 :=
by 
  sorry

end new_person_age_l142_142157


namespace total_lives_l142_142557

/-- Suppose there are initially 4 players, then 5 more players join. Each player has 3 lives.
    Prove that the total number of lives is equal to 27. -/
theorem total_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
  (h_initial : initial_players = 4) (h_additional : additional_players = 5) (h_lives : lives_per_player = 3) : 
  initial_players + additional_players = 9 ∧ 
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end total_lives_l142_142557


namespace repeating_decimal_primes_count_l142_142670

-- Define the interval and the property we want for the primes.
def is_repeating_decimal (n : ℕ) : Prop := 
  ¬(∀ p, Nat.Prime p → (n + 1) % p = 0 → (p = 2) ∨ (p = 5))

-- Filter the primes in the given range that satisfy the property.
noncomputable def count_repeating_decimal_primes : ℕ :=
  (Finset.range 151).filter (λ n, Nat.Prime n ∧ is_repeating_decimal n).card

-- Statement asserting the equality to the specific count k.
theorem repeating_decimal_primes_count (k : ℕ) :
  count_repeating_decimal_primes = k := 
  sorry

end repeating_decimal_primes_count_l142_142670


namespace smaller_cube_volume_is_correct_l142_142193

noncomputable def inscribed_smaller_cube_volume 
  (edge_length_outer_cube : ℝ)
  (h : edge_length_outer_cube = 12) : ℝ := 
  let diameter_sphere := edge_length_outer_cube
  let radius_sphere := diameter_sphere / 2
  let space_diagonal_smaller_cube := diameter_sphere
  let side_length_smaller_cube := space_diagonal_smaller_cube / (Real.sqrt 3)
  let volume_smaller_cube := side_length_smaller_cube ^ 3
  volume_smaller_cube

theorem smaller_cube_volume_is_correct 
  (h : 12 = 12) : inscribed_smaller_cube_volume 12 h = 192 * Real.sqrt 3 :=
by
  sorry

end smaller_cube_volume_is_correct_l142_142193


namespace num_divisible_by_7_200_to_400_l142_142367

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l142_142367


namespace largest_odd_factors_of_220_l142_142104

def number := 220

def is_factor (n d : ℕ) : Prop := d ∣ n
def is_odd (n : ℕ) : Prop := n % 2 = 1

noncomputable def odd_factors (n : ℕ) : List ℕ := 
  (List.range (n + 1)).filter (λ d, is_factor n d ∧ is_odd d)

theorem largest_odd_factors_of_220 : ∃ (a b c : ℕ), 
  {a, b, c} = {5, 11, 55} ∧ 
  a ∈ odd_factors number ∧ 
  b ∈ odd_factors number ∧ 
  c ∈ odd_factors number ∧ 
  (∀ x ∈ odd_factors number, x ≠ 1 → (x = a ∨ x = b ∨ x = c)) :=
by
  sorry

end largest_odd_factors_of_220_l142_142104


namespace texts_sent_total_l142_142057

def texts_sent_on_monday_to_allison_and_brittney : Nat := 5 + 5
def texts_sent_on_tuesday_to_allison_and_brittney : Nat := 15 + 15

def total_texts_sent (texts_monday : Nat) (texts_tuesday : Nat) : Nat := texts_monday + texts_tuesday

theorem texts_sent_total :
  total_texts_sent texts_sent_on_monday_to_allison_and_brittney texts_sent_on_tuesday_to_allison_and_brittney = 40 :=
by
  sorry

end texts_sent_total_l142_142057


namespace volume_ratio_of_cube_and_cuboid_l142_142434

theorem volume_ratio_of_cube_and_cuboid :
  let edge_length_meter := 1
  let edge_length_cm := edge_length_meter * 100 -- Convert meter to centimeters
  let cube_volume := edge_length_cm^3
  let cuboid_width := 50
  let cuboid_length := 50
  let cuboid_height := 20
  let cuboid_volume := cuboid_width * cuboid_length * cuboid_height
  cube_volume = 20 * cuboid_volume := 
by
  sorry

end volume_ratio_of_cube_and_cuboid_l142_142434


namespace video_has_2600_dislikes_l142_142570

def likes := 3000
def initial_dislikes := 1500 + 100
def additional_dislikes := 1000
def total_dislikes := initial_dislikes + additional_dislikes

theorem video_has_2600_dislikes:
  total_dislikes = 2600 :=
by
  unfold likes initial_dislikes additional_dislikes total_dislikes
  sorry

end video_has_2600_dislikes_l142_142570


namespace examine_points_l142_142221

variable (Bryan Jen Sammy mistakes : ℕ)

def problem_conditions : Prop :=
  Bryan = 20 ∧ Jen = Bryan + 10 ∧ Sammy = Jen - 2 ∧ mistakes = 7

theorem examine_points (h : problem_conditions Bryan Jen Sammy mistakes) : ∃ total_points : ℕ, total_points = Sammy + mistakes :=
by {
  sorry
}

end examine_points_l142_142221


namespace five_digit_numbers_count_l142_142282

noncomputable def count_five_digit_numbers : ℕ := 
  let first_set := {1, 3, 5, 7, 9}
  let second_set := {2, 4, 6, 8}
  let choose_three_from_first := Nat.choose 5 3
  let choose_two_from_second := Nat.choose 4 2
  let arrange_five := factorial 5
  choose_three_from_first * choose_two_from_second * arrange_five

-- Statement we want to prove
theorem five_digit_numbers_count : count_five_digit_numbers = 7200 :=
by
  unfold count_five_digit_numbers
  -- Add the necessary calculations or computational proofs here
  sorry

end five_digit_numbers_count_l142_142282


namespace mike_age_when_barbara_is_16_l142_142828

noncomputable theory
open_locale classical

def Mike_initial_age : ℕ := 16
def Barbara_initial_age : ℕ := Mike_initial_age / 2

theorem mike_age_when_barbara_is_16 (M B M' B' : ℕ) 
  (hM : M = 16)
  (hB : B = M / 2)
  (hB' : B' = 16)
  (hM' : M' = M + (B' - B)) :
  M' = 24 := 
by {
  have h1 : B = 8,
  { rw [hB, hM], norm_num, },
  rw [hM', hB'],
  rw [h1],
  norm_num,
}

end mike_age_when_barbara_is_16_l142_142828


namespace total_dogs_is_28_l142_142242

def number_of_boxes : ℕ := 7
def dogs_per_box : ℕ := 4
def total_dogs (boxes : ℕ) (dogs_in_each : ℕ) : ℕ := boxes * dogs_in_each

theorem total_dogs_is_28 : total_dogs number_of_boxes dogs_per_box = 28 :=
by
  sorry

end total_dogs_is_28_l142_142242


namespace serena_clears_driveway_l142_142478

def serena_shovels_snow (initial_rate : ℕ) (decrement : ℕ) (total_volume : ℕ) : ℕ :=
  let remaining_volume n := total_volume - (n * initial_rate - (decrement * (n * (n-1)) / 2))
  Nat.find_greatest (λ n, remaining_volume n > 0) 10

theorem serena_clears_driveway :
  serena_shovels_snow 30 2 200 = 10 :=
by
  -- Proof is to be provided here.
  sorry

end serena_clears_driveway_l142_142478


namespace hockey_season_duration_l142_142902

theorem hockey_season_duration 
  (total_games : ℕ)
  (games_per_month : ℕ)
  (h_total : total_games = 182)
  (h_monthly : games_per_month = 13) : 
  total_games / games_per_month = 14 := 
by
  sorry

end hockey_season_duration_l142_142902


namespace B_C_complete_task_in_l142_142937

variable (A B C : Type) [Field A] [Field B] [Field C] (task : A)

def work_rate_A : A := 1 / 12

def work_rate_B : A := 1.2 * work_rate_A

def work_rate_C : A := 0.75 * work_rate_A

def combined_work_rate (B_rate C_rate : A) : A := B_rate + C_rate

noncomputable def days_to_complete (combined_rate : A) : A := 1 / combined_rate

theorem B_C_complete_task_in (B_rate C_rate : A) (h1 : B_rate = 1.2 * work_rate_A) (h2 : C_rate = 0.75 * work_rate_A) :
  days_to_complete (combined_work_rate B_rate C_rate) = 80 / 13 :=
by
  unfold work_rate_A
  unfold work_rate_B
  unfold work_rate_C
  unfold combined_work_rate
  unfold days_to_complete
  sorry

end B_C_complete_task_in_l142_142937


namespace josie_initial_amount_is_correct_l142_142433

def cost_of_milk := 4.00 / 2
def cost_of_bread := 3.50
def cost_of_detergent_after_coupon := 10.25 - 1.25
def cost_of_bananas := 2 * 0.75
def total_cost := cost_of_milk + cost_of_bread + cost_of_detergent_after_coupon + cost_of_bananas
def leftover := 4.00
def initial_amount := total_cost + leftover

theorem josie_initial_amount_is_correct :
  initial_amount = 20.00 := by
  sorry

end josie_initial_amount_is_correct_l142_142433


namespace zoo_problem_l142_142227

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end zoo_problem_l142_142227


namespace complex_number_real_imaginary_equal_implies_a_is_3_l142_142039

theorem complex_number_real_imaginary_equal_implies_a_is_3 (a : ℝ) (i : ℂ) (h : i = complex.I) 
(h_equal_parts : (complex.re ((a + i) / (2 - i)) = complex.im ((a + i) / (2 - i)))) 
: a = 3 :=
sorry

end complex_number_real_imaginary_equal_implies_a_is_3_l142_142039


namespace zoo_problem_l142_142228

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end zoo_problem_l142_142228


namespace number_divisor_property_l142_142131

theorem number_divisor_property (s : Set ℕ) (h_s : s ⊆ Finset.range 2015) (h_size : s.card = 1008) :
  ∃ a b ∈ s, a ≠ b ∧ a ∣ b := 
by
  sorry

end number_divisor_property_l142_142131


namespace monkeys_more_than_giraffes_l142_142231

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end monkeys_more_than_giraffes_l142_142231


namespace birds_percentage_not_hawks_paddyfield_warblers_kingfishers_l142_142929

theorem birds_percentage_not_hawks_paddyfield_warblers_kingfishers
  (total_birds : ℕ)
  (hawks_percentage : ℝ := 0.3)
  (paddyfield_warblers_percentage : ℝ := 0.4)
  (kingfishers_ratio : ℝ := 0.25) :
  (35 : ℝ) = 100 * ( total_birds - (hawks_percentage * total_birds) 
                     - (paddyfield_warblers_percentage * (total_birds - (hawks_percentage * total_birds))) 
                     - (kingfishers_ratio * paddyfield_warblers_percentage * (total_birds - (hawks_percentage * total_birds))) )
                / total_birds :=
by
  sorry

end birds_percentage_not_hawks_paddyfield_warblers_kingfishers_l142_142929


namespace largest_base3_three_digit_to_base10_l142_142536

theorem largest_base3_three_digit_to_base10 :
  let largest_base3_number := 2 * 3^2 + 2 * 3^1 + 2 * 3^0 in
  largest_base3_number = 26 :=
by
  sorry

end largest_base3_three_digit_to_base10_l142_142536


namespace minimum_value_of_A_l142_142691

open Real

noncomputable def A (x y z : ℝ) : ℝ :=
  ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x * y + y * z + z * x)

theorem minimum_value_of_A (x y z : ℝ) (h : 3 ≤ x) (h2 : 3 ≤ y) (h3 : 3 ≤ z) :
  ∃ v : ℝ, (∀ a b c : ℝ, 3 ≤ a ∧ 3 ≤ b ∧ 3 ≤ c → A a b c ≥ v) ∧ v = 1 :=
sorry

end minimum_value_of_A_l142_142691


namespace smallest_positive_period_pi_interval_extrema_l142_142336

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos (x + Real.pi / 3) + Real.sqrt 3

theorem smallest_positive_period_pi : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem interval_extrema :
  ∃ x_max x_min : ℝ, 
  -Real.pi / 4 ≤ x_max ∧ x_max ≤ Real.pi / 6 ∧ f x_max = 2 ∧
  -Real.pi / 4 ≤ x_min ∧ x_min ≤ Real.pi / 6 ∧ f x_min = -1 ∧ 
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 2 ∧ f x ≥ -1) :=
sorry

end smallest_positive_period_pi_interval_extrema_l142_142336


namespace cost_when_q_is_2_l142_142506

-- Defining the cost function
def cost (q : ℕ) : ℕ := q^3 + q - 1

-- Theorem to prove the cost when q = 2
theorem cost_when_q_is_2 : cost 2 = 9 :=
by
  -- placeholder for the proof
  sorry

end cost_when_q_is_2_l142_142506


namespace angle_A_leq_45_contradiction_l142_142347

theorem angle_A_leq_45_contradiction (ABC : Triangle)
  (right_angle_C : ABC.C = 90)
  (angle_A_gt_angle_B : ABC.A > ABC.B) :
  (using_contradiction_assume : ABC.A ≤ 45) :=
sorry

end angle_A_leq_45_contradiction_l142_142347


namespace problem_1_problem_2_l142_142702

noncomputable def z1 : ℂ := 1 - I
noncomputable def z2 : ℂ := 4 + 6 * I

theorem problem_1 :
  z2 / z1 = -1 + 5 * I := 
sorry

variables (b : ℝ) (z : ℂ) (hz : z = 1 + b * I) (hzz1_real : z + z1 ∈ ℝ)

theorem problem_2 : 
  |z| = Real.sqrt 2 := 
sorry

end problem_1_problem_2_l142_142702


namespace tangent_line_segment_length_l142_142444

open Real

theorem tangent_line_segment_length
  (center1 : ℝ × ℝ) (r1 : ℝ) (center2 : ℝ × ℝ) (r2 : ℝ)
  (h1 : center1 = (12, 3)) (hr1 : r1 = 5)
  (h2 : center2 = (-9, -4)) (hr2 : r2 = 7) :
  let d := sqrt ((12 - (-9))^2 + (3 - (-4))^2)
  in  sqrt (d^2 - (r1 + r2)^2) = 70 :=
by
  sorry

end tangent_line_segment_length_l142_142444


namespace root_in_interval_l142_142148

theorem root_in_interval :
  ∃ x ∈ set.Ioo (-2 : ℝ) 0, (x^2 - 1) = 0 :=
sorry

end root_in_interval_l142_142148


namespace sum_mod_2008_l142_142607

theorem sum_mod_2008 (grid : ℕ → ℕ → ℕ)
  (H_grid_seq : ∀ i j : ℕ, grid (i - 1) (j - 1) = (i - 1) * 2008 + j)
  (H_no_overlap : ∀ (pick1 pick2 : fin 2008 → fin 2008),
    (∀ i j : fin 2008, i ≠ j → pick1 i ≠ pick1 j ∧ pick2 i ≠ pick2 j) →
    (∀ i : fin 2008, pick1 i ≠ pick2 i)) :
  (∑ i : fin 2008, grid (i : ℕ) ((pick1 i) : ℕ)) % 2008 = 1004 ∧
  (∑ i : fin 2008, grid (i : ℕ) ((pick2 i) : ℕ)) % 2008 = 1004 := by
  sorry

end sum_mod_2008_l142_142607


namespace eccentricity_ellipse_l142_142083

variable (a b c : ℝ)

def is_ellipse_center_origin (a b : ℝ) (h : a > b ∧ b > 0) := True

def is_focus (c : ℝ) := True

def is_line (x : ℝ) := (3 * x - 2)

def midpoint_x : ℝ := 1 / 2

def eccentricity (c a : ℝ) := c / a

theorem eccentricity_ellipse : (∀ (a b c : ℝ),
  is_ellipse_center_origin a b (a > b ∧ b > 0) →
  is_focus c →
  (midpoint_x * 2 = 1) →
  2 * a^2 = 3 * c^2 →
  eccentricity c a = Real.sqrt 6 / 3) := sorry

end eccentricity_ellipse_l142_142083


namespace remainder_product_modulo_17_l142_142143

theorem remainder_product_modulo_17 :
  (1234 % 17) = 5 ∧ (1235 % 17) = 6 ∧ (1236 % 17) = 7 ∧ (1237 % 17) = 8 ∧ (1238 % 17) = 9 →
  ((1234 * 1235 * 1236 * 1237 * 1238) % 17) = 9 :=
by
  sorry

end remainder_product_modulo_17_l142_142143


namespace tinas_extra_earnings_l142_142047

def price_per_candy_bar : ℕ := 2
def marvins_candy_bars_sold : ℕ := 35
def tinas_candy_bars_sold : ℕ := 3 * marvins_candy_bars_sold

def marvins_earnings : ℕ := marvins_candy_bars_sold * price_per_candy_bar
def tinas_earnings : ℕ := tinas_candy_bars_sold * price_per_candy_bar

theorem tinas_extra_earnings : tinas_earnings - marvins_earnings = 140 := by
  sorry

end tinas_extra_earnings_l142_142047


namespace nancy_savings_exceeds_goal_l142_142053
noncomputable theory

def nancy_first_job_daily_income : ℕ := 12 * 2
def nancy_first_job_total_income : ℕ := nancy_first_job_daily_income * 4

def second_job_daily_income (day: ℕ) : ℕ := (10 + 2 * (day - 1)) * 4
def nancy_second_job_total_income : ℕ := second_job_daily_income 1 + second_job_daily_income 2 + second_job_daily_income 3 + second_job_daily_income 4

def nancy_third_job_daily_income : ℕ := 25
def nancy_third_job_total_income : ℕ := nancy_third_job_daily_income * 4

def nancy_total_income : ℕ := nancy_first_job_total_income + nancy_second_job_total_income + nancy_third_job_total_income

theorem nancy_savings_exceeds_goal :  nancy_total_income - 300 = 104 :=
by
  let total_first_job_income := 96
  let total_second_job_income := 208
  let total_third_job_income := 100
  let total_income := total_first_job_income + total_second_job_income + total_third_job_income
  have h1 : total_income = 404 := by sorry
  have h2 : total_income - 300 = 104 := by sorry
  exact h2

end nancy_savings_exceeds_goal_l142_142053


namespace proof_max_value_l142_142730

open Real EuclideanSpace

variables {V : Type*} [InnerProductSpace ℝ V]

def max_value (m n : V) : ℝ :=
  ∥2 • m + n∥ + ∥n∥

theorem proof_max_value (m n : V) (h1 : ∥m∥ = 2) (h2 : ∥m + 2 • n∥ = 2) :
  max_value m n = (8 * sqrt 3) / 3 :=
sorry

end proof_max_value_l142_142730


namespace repeated_two_digit_divisible_by_10101_l142_142598

theorem repeated_two_digit_divisible_by_10101 (x y : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9) :
  let n := 100 * x + y in
  10101 ∣ (10000 * n + 100 * n + n) :=
by
  let n := 100 * x + y
  have : 10000 * n + 100 * n + n = 10101 * n := by sorry
  exact dvd.intro n this

end repeated_two_digit_divisible_by_10101_l142_142598


namespace problem_solution_l142_142517

def boys_A := 5
def girls_A := 3
def boys_B := 6
def girls_B := 2

noncomputable def selected_ways (boys_A girls_A boys_B girls_B : ℕ) : ℕ :=
  (nat.choose girls_A 1) * (nat.choose boys_A 1) * (nat.choose boys_B 2) + 
  (nat.choose boys_A 2) * (nat.choose girls_B 1) * (nat.choose boys_B 1)

theorem problem_solution : selected_ways boys_A girls_A boys_B girls_B = 345 := by
  sorry

end problem_solution_l142_142517


namespace sine_angle_ef_pbc_l142_142791

-- Definitions of points and tetrahedron properties
structure Tetrahedron (P A B C E F : Type) :=
(PA_perpendicular_AB : perpendicular P A B)
(PA_perpendicular_AC : perpendicular P A C)
(AB_perpendicular_AC : perpendicular A B C)
(PA_eq_AB : dist P A = dist A B)
(PA_eq_AC : dist P A = dist A C)
(mE : midpoint E A B)
(mF : midpoint F P C)

-- Define a function to compute the sine of angle between a line and a plane
def sine_angle_line_plane {P A B C E F : Type} (T : Tetrahedron P A B C E F) : ℝ :=
  (- (dist P A)^3) / ((dist P A) * sqrt 3) * (dist P A * sqrt 3)

-- The statement that needs to be proven
theorem sine_angle_ef_pbc {P A B C E F : Type} (T : Tetrahedron P A B C E F) :
  sine_angle_line_plane T = sqrt 8 / 3 :=
sorry

end sine_angle_ef_pbc_l142_142791


namespace vertices_count_edges_count_condition_a_b_l142_142559

-- Definitions of the region S
def region_S (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x + y + z ≤ 11 ∧
  2 * x + 4 * y + 3 * z ≤ 36 ∧
  2 * x + 3 * z ≤ 24

-- Proof that there are 7 vertices in S
theorem vertices_count :
  ∃ (vertices : set (ℝ × ℝ × ℝ)),
    (∀ (v : ℝ × ℝ × ℝ), v ∈ vertices → region_S v.1 v.2 v.3) ∧
    vertices.size = 7 :=
by
  sorry

-- Proof that there are 11 edges in S
theorem edges_count :
  ∃ (edges : set (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)),
    (∀ (e : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ), e ∈ edges →
      ∃ (v1 v2 : ℝ × ℝ × ℝ), e = (v1.1, v1.2, v1.3, v2.1, v2.2, v2.3) ∧
      v1 ∈ vertices ∧ v2 ∈ vertices ∧
      ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
      region_S (t * v1.1 + (1 - t) * v2.1) (t * v1.2 + (1 - t) * v2.2) (t * v1.3 + (1 - t) * v2.3)) ∧
    edges.size = 11 :=
by
  sorry

-- Proof for values of a and b ensuring ax + by + z ≤ 2a + 5b + 4 in S
theorem condition_a_b (a b : ℝ) :
  (∀ (x y z : ℝ), region_S x y z → a * x + b * y + z ≤ 2 * a + 5 * b + 4) ↔
  (2 / 3 ≤ a ∧ a ≤ 1 ∧ b = 2 - a) :=
by
  sorry

end vertices_count_edges_count_condition_a_b_l142_142559


namespace smallest_root_of_polynomial_l142_142248

theorem smallest_root_of_polynomial :
  let g : ℝ → ℝ := λ x, 12 * x^4 - 8 * x^2 + 1 in
  ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → abs x ≤ abs y ∧ x = sqrt (1 / 6) :=
begin
  sorry
end

end smallest_root_of_polynomial_l142_142248


namespace min_jumps_to_same_segment_l142_142519

theorem min_jumps_to_same_segment (P : set ℝ) (hP : ∀ p ∈ P, 0 ≤ p ∧ p ≤ 1) :
  ∃ n, n = 2 ∧ ∀ f1 f2 ∈ {0, 1}, (∃ p ∈ P, ¬jump p f1 ⟶ jump p f2)
  := sorry

/-
Definitions:
- jump: A function representing the symmetric jump of a flea over a point.
  For our simple example: jump p x = 2 * p - x

Hypotheses:
- The points in P are within the interval [0, 1].
- The minimum number of jumps required is 2.
- For any initial locations of the fleas at {0, 1}, there exists a marked point in P such that the fleas can jump and land within the same sub-segment.
-/

end min_jumps_to_same_segment_l142_142519


namespace ralph_socks_l142_142474

theorem ralph_socks
  (x y w z : ℕ)
  (h1 : x + y + w + z = 15)
  (h2 : x + 2 * y + 3 * w + 4 * z = 36)
  (hx : x ≥ 1) (hy : y ≥ 1) (hw : w ≥ 1) (hz : z ≥ 1) :
  x = 5 :=
sorry

end ralph_socks_l142_142474


namespace numbers_divisor_property_l142_142134

theorem numbers_divisor_property (S : Finset ℕ) (h₁ : Finset.card S = 1008) (h₂ : ∀ x ∈ S, x ≤ 2014) :
  ∃ a b ∈ S, a ∣ b ∨ b ∣ a :=
by
  sorry

end numbers_divisor_property_l142_142134


namespace units_digit_m_squared_plus_3_to_m_eq_5_l142_142817

theorem units_digit_m_squared_plus_3_to_m_eq_5 :
  let m := 2011^2 + 3^2011 in
  (m^2 + 3^m) % 10 = 5 := 
by
  sorry

end units_digit_m_squared_plus_3_to_m_eq_5_l142_142817


namespace num_natural_numbers_divisible_by_7_l142_142381

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l142_142381


namespace equilateral_triangle_perimeter_l142_142612

/-- Given an equilateral triangle ABC, and DE = 19, DF = 89, 
we want to prove the perimeter of triangle ABC is 216\sqrt{3} -/
theorem equilateral_triangle_perimeter
  (a b c : ℝ)
  (h_eq : a = b ∧ b = c)
  (DE DF : ℝ)
  (h_DE : DE = 19)
  (h_DF : DF = 89)
  (h_height : ∃ h: ℝ, h = DE + DF) :
  let h := 19 + 89 in
  let side_length := (2 * h) / Real.sqrt 3 in
  let perimeter := 3 * side_length in
  perimeter = 216 * Real.sqrt 3 :=
by 
  let h := 19 + 89 in
  let side_length := (2 * h) / Real.sqrt 3 in
  let perimeter := 3 * side_length in
  sorry

end equilateral_triangle_perimeter_l142_142612


namespace black_population_percentage_in_the_west_l142_142977

theorem black_population_percentage_in_the_west (NE_Blacks : ℝ) (MW_Blacks : ℝ) (South_Blacks : ℝ) (West_Blacks : ℝ) (total_Blacks : ℝ) : 
  NE_Blacks = 6 → MW_Blacks = 7 → South_Blacks = 18 → West_Blacks = 4 → 
  total_Blacks = NE_Blacks + MW_Blacks + South_Blacks + West_Blacks →
  (West_Blacks / total_Blacks) * 100 = 11 :=
by
  intros hNE hMW hSouth hWest hTotal
  have h1 : total_Blacks = 35, by 
  linarith
  have h2 : (4 / 35) * 100 ≠ 11.43, by norm_num
  have h3 : (4 / 35) * 100 ≈ 11.43 → (4 / 35) * 100 ≈ 11, by sorry
  exact h3
  sorry

end black_population_percentage_in_the_west_l142_142977


namespace clock_face_rearrangement_sum_l142_142863

theorem clock_face_rearrangement_sum (a : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ i, 1 ≤ a i ∧ a i ≤ 12) (h_sum : ∑ i in finset.range 12, a i = 78):
  ∃ i, a i + a ((i + 1) % 12) + a ((i + 2) % 12) ≥ 21 := sorry

end clock_face_rearrangement_sum_l142_142863


namespace divisor_exists_l142_142126

theorem divisor_exists : ∀ (s : Finset ℕ),
  (∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) → s.card = 1008 →
  ∃ a b ∈ s, a ∣ b ∧ a ≠ b :=
by
  sorry

end divisor_exists_l142_142126


namespace find_line_equations_l142_142263

theorem find_line_equations (P A B : ℝ × ℝ) (hP : P = (1, 2)) (hA : A = (2, 3)) (hB : B = (0, -5)) :
  (∀ x y, (y - 2 = 4 * (x - 1) → x ∈ {1}) ∨ (x = 1)) :=
by
  have slopes_are_equal : ∀ x y, (y - 2 = 4 * (x - 1)) := sorry
  have line_through_P : ∀ x y, x = 1 := sorry
  sorry

end find_line_equations_l142_142263


namespace number_of_divisibles_by_7_l142_142353

theorem number_of_divisibles_by_7 (a b : ℕ) (h1 : a = 200) (h2 : b = 400) : 
  (nat.card {n | a ≤ 7 * n ∧ 7 * n ≤ b}) = 29 := 
by sorry

end number_of_divisibles_by_7_l142_142353


namespace general_formula_correct_lambda_range_l142_142307

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {n : ℕ} {λ : ℝ}

-- Conditions from the problem
noncomputable def condition1 : Prop := 2 * a 3 - a 2 = 5
noncomputable def condition2 : Prop := S 5 - S 3 = 14

-- Desired outcomes
noncomputable def general_formula (n : ℕ) : ℝ := 4 * n - 11
noncomputable def inequality (n : ℕ) : ℝ := S n - a n + λ ≥ 0

-- Prove the general formula matches
theorem general_formula_correct (h1 : condition1) (h2 : condition2) :
  ∀ n, a n = general_formula n := 
sorry

-- Prove the range of values for λ
theorem lambda_range (h1 : ∀ n, a n = general_formula n) :
  (∀ n ∈ {k | Nat.k : ℕ+}.toFinset, S n = (λ m, m * (2 * general_formula m - 9) : ℕ → ℝ) n) →
  (∀ n, inequality n) → λ ≥ 10 :=
sorry

end general_formula_correct_lambda_range_l142_142307


namespace domino_swap_multiplication_correct_l142_142802

-- Defining the scenario where domino swapping yields correct multiplication
theorem domino_swap_multiplication_correct : 
  ∀ (a b : ℕ), (a = 4) → (b = 3) → (a * b = 12) :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  exact Nat.mul_comm 3 4 ▸ rfl

end domino_swap_multiplication_correct_l142_142802


namespace total_students_in_gym_class_l142_142967

theorem total_students_in_gym_class (group1 group2 : ℕ)
  (h1 : group1 = 34) 
  (h2 : group2 = 37) : 
  group1 + group2 = 71 := 
by {
  rw [h1, h2],
  exact rfl,
}

end total_students_in_gym_class_l142_142967


namespace positive_difference_l142_142080

def average (a b : ℕ) := (a + b) / 2

theorem positive_difference (y : ℕ) (h: average 54 y = 32) : abs (54 - y) = 44 :=
by
  sorry

end positive_difference_l142_142080


namespace general_term_formula_sum_term_b_l142_142308

-- Conditions
variables (a : ℕ → ℝ)
variables (S : ℕ → ℝ)
variables (q : ℝ) (h_q : q > 1)
variables (h2_1 : a 2 + 1 = (a 1 + a 3) / 2)
variables (h_S3 : S 3 = 14)

-- The general term formula for a_n
theorem general_term_formula : (∃ a_n_formula: ℕ → ℝ, ∀ n, a n = 2^n) :=
sorry

-- Sum of the first n terms T_n for the sequence b_n
noncomputable def b (n : ℕ) : ℝ := a n * real.logb 2 (a n)

theorem sum_term_b (n : ℕ) : (∃ T_n_formula: ℕ → ℝ, T_n_formula n = (n - 1) * 2^(n + 1) + 2) :=
sorry

end general_term_formula_sum_term_b_l142_142308


namespace max_value_of_n_l142_142326

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_cond : a 11 / a 10 < -1)
  (h_maximum : ∃ N, ∀ n > N, S n ≤ S N) :
  ∃ N, S N > 0 ∧ ∀ m, S m > 0 → m ≤ N :=
by
  sorry

end max_value_of_n_l142_142326


namespace vertex_of_f1_range_of_f1_expression_of_g_l142_142299

section
variable (x : ℝ)

-- Define the quadratic function with parameters b and c.
def quadratic (b c : ℝ) : ℝ := -x^2 + b*x + c

-- Part 1: Specific values of b and c.
def f1 : ℝ := quadratic 4 3 x

-- Problem 1: Prove that the vertex of f1 is (2, 7).
theorem vertex_of_f1 : ∃! (p : ℝ × ℝ), p = (2, 7) ∧ ∀ x, f1 ≤ f1 2 := sorry

-- Problem 2: Prove that the range of f1 on [-1, 3] is [-2, 7].
theorem range_of_f1 : ∀ y, (-2 ≤ y ∧ y ≤ 7) ↔ ∃ x, (-1 ≤ x ∧ x ≤ 3 ∧ y = f1) := sorry

-- Part 2: Given conditions on the function.
variable (y : ℝ)

-- Define a function to represent the conditions.
def g : ℝ := quadratic 2 2 x

-- When x ≤ 0, the maximum value is 2.
axiom max_value_x_leq_0 (x : ℝ) (h : x ≤ 0) : g ≤ 2

-- When x > 0, the maximum value is 3.
axiom max_value_x_gt_0 (x : ℝ) (h : x > 0) : g ≤ 3

-- Problem 3: Prove that the expression of the quadratic function is y = -x^2 + 2x + 2.
theorem expression_of_g : g = -x^2 + 2*x + 2 := sorry
end

end vertex_of_f1_range_of_f1_expression_of_g_l142_142299


namespace unique_positive_real_solution_l142_142663

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l142_142663


namespace fraction_simplification_l142_142214

theorem fraction_simplification (x : ℝ) (hx₁ : x ≠ 2) (hx₂ : x ≠ -2) :
  let a := (x^2 - 4),
      b := (x^2 - 4x + 4)
  in let c := (x^2 + 4x + 4),
         d := (2x - x^2)
  in (∃ (Y₁ Y₂ Y₃ : ℝ), (Y₁ = (x-2)^2) ∧ (Y₂ = x-2) ∧ (Y₃ = (x+2)^2) ∧
      ((a / b) * (d / c) = -x / Y₃)) :=
begin
  sorry
end

end fraction_simplification_l142_142214


namespace haley_small_gardens_l142_142163

theorem haley_small_gardens (total_seeds : ℕ) (seeds_in_big_garden : ℕ) (seeds_per_small_garden : ℕ) :
  total_seeds = 56 → seeds_in_big_garden = 35 → seeds_per_small_garden = 3 → 
  (total_seeds - seeds_in_big_garden) / seeds_per_small_garden = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end haley_small_gardens_l142_142163


namespace no_factors_x4_2x2_9_l142_142469

theorem no_factors_x4_2x2_9 (x : ℝ) : ¬(∃ (f : ℝ → ℝ), f = (λ x, x^4 + 2 * x^2 + 9) ∧ 
  (f = (λ x, (x^2 + 3) * q1 x) ∨
   f = (λ x, (x + 1) * q2 x) ∨ 
   f = (λ x, (x^2 - 3) * q3 x) ∨ 
   f = (λ x, (x^2 - 2 * x - 3) * q4 x))) :=
by { sorry }

end no_factors_x4_2x2_9_l142_142469


namespace find_m_l142_142318

theorem find_m (x : ℝ) (m : ℝ) (h1 : log 10 (tan x) + log 10 (cot x) = 0)
  (h2 : log 10 (tan x + cot x) = (1 / 2) * (log 10 m - 1)) : m = 10 := 
sorry

end find_m_l142_142318


namespace find_n_l142_142315

def P (n : ℕ) : ℕ × ℕ :=
  Nat.recOn n (1, 0) $ λ k pk, (pk.fst - pk.snd, pk.fst + pk.snd)

def a (n : ℕ) : ℕ :=
  let pn := P n
  let pn1 := P (n + 1)
  let pn2 := P (n + 2)
  (pn1.fst - pn.fst) * (pn2.fst - pn1.fst) + (pn1.snd - pn.snd) * (pn2.snd - pn1.snd)

def sum_a (n : ℕ) : ℕ :=
  (List.range n).sum a

theorem find_n (n : ℕ) :
  sum_a n > 1000 :=
  sorry

end find_n_l142_142315


namespace number_of_points_P_l142_142445

theorem number_of_points_P (x y : ℝ) : 
  let P := (x, y) in 
  let F₁ := (-4, 0),
      F₂ := (4, 0) in
  P ∈ {p : ℝ × ℝ | (p.1^2 / 25) + (p.2^2 / 9) = 1} →
  (x + 4) * (x - 4) + y^2 = 7 →
  ∃! (x y : ℝ), (x^2 / 25 + y^2 / 9 = 1 ∧ x^2 + y^2 = 23) := 
by 
  sorry

end number_of_points_P_l142_142445


namespace hockey_season_length_l142_142900

theorem hockey_season_length (total_games_per_month : ℕ) (total_games_season : ℕ) 
  (h1 : total_games_per_month = 13) (h2 : total_games_season = 182) : 
  total_games_season / total_games_per_month = 14 := 
by 
  sorry

end hockey_season_length_l142_142900


namespace pickles_per_cucumber_l142_142837

theorem pickles_per_cucumber (jars cucumbers vinegar_initial vinegar_left pickles_per_jar vinegar_per_jar total_pickles_per_cucumber : ℕ) 
    (h1 : jars = 4) 
    (h2 : cucumbers = 10) 
    (h3 : vinegar_initial = 100) 
    (h4 : vinegar_left = 60) 
    (h5 : pickles_per_jar = 12) 
    (h6 : vinegar_per_jar = 10) 
    (h7 : total_pickles_per_cucumber = 4): 
    total_pickles_per_cucumber = (vinegar_initial - vinegar_left) / vinegar_per_jar * pickles_per_jar / cucumbers := 
by 
  sorry

end pickles_per_cucumber_l142_142837


namespace sum_of_logs_eq_neg_two_l142_142823

noncomputable def tangent_line_x_intercept (n : ℕ) : ℝ :=
1 - 1 / (n : ℝ)

noncomputable def a (n : ℕ) : ℝ :=
Real.log10 (tangent_line_x_intercept n)

theorem sum_of_logs_eq_neg_two :
  (∑ n in Finset.range (99 - 1) + 1, a (n + 2)) = -2 := by
  sorry

end sum_of_logs_eq_neg_two_l142_142823


namespace divisor_exists_l142_142122

-- Define the initial conditions
def all_natural_numbers := { n : ℕ | 1 ≤ n ∧ n ≤ 2014 }

-- Define the remaining set after erasing 1006 numbers
def remaining_numbers (S : set ℕ) (h : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : Prop :=
  ∃ a b ∈ S, a ≠ b ∧ a ∣ b

-- The problem statement in Lean
theorem divisor_exists (S : set ℕ) (h_sub : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : 
  remaining_numbers S h_sub h_card :=
begin
  sorry
end

end divisor_exists_l142_142122


namespace zoo_problem_l142_142226

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end zoo_problem_l142_142226


namespace arithmetic_sequence_properties_l142_142306

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (S4 : ℤ) (T : ℕ → ℚ) :
  (∀ n, a n = a 0 + n * (a 1 - a 0)) ∧ -- Arithmetic sequence
  S4 = a 0 + a 1 + a 2 + a 3 ∧
  a 0 ≠ a 1 ∧ -- Distinct terms imply non-zero common difference
  S4 = 14 ∧
  (a 0 + 2 * (a 1 - a 0))^2 = a 0 * (a 0 + 6 * (a 1 - a 0)) ∧
  (∀ n, T n = (∑ k in finset.range n, (1/((a k)*(a k+1))))) →
  (∀ n, a n = n + 1) ∧
  (∀ n, T n = n / (2 * (n + 2))) := by
  sorry

end arithmetic_sequence_properties_l142_142306


namespace pair_points_with_intersections_l142_142288

/-- 
  Given 22 points on a plane such that no three of them are collinear, 
  it is possible to pair the points such that the connecting line segments intersect 
  at least five times.
-/
theorem pair_points_with_intersections (points : Fin 22 -> Point)
  (h_no_three_collinear : ∀ (A B C : Point), A ∈ points -> B ∈ points -> C ∈ points -> ¬ collinear {A, B, C}) :
  ∃ (pairs : Fin 11 -> (Point × Point)), 
    (∀ i j, i ≠ j -> segments_intersect (pairs i).fst (pairs i).snd (pairs j).fst (pairs j).snd) ∧
    (∃ intersects : Fin 5, ∃ (i j : Fin 11), i ≠ j ∧ 
      segments_intersect (pairs i).fst (pairs i).snd (pairs j).fst (pairs j).snd) :=
sorry

end pair_points_with_intersections_l142_142288


namespace edge_of_third_cube_l142_142085

theorem edge_of_third_cube (a b c : ℕ) (V1 V2 V3 V_total V_third_cube : ℕ) :
  a = 4 → b = 5 → c = 6 → 
  V1 = a^3 → V2 = b^3 → V3 = c^3 →
  V_total = V1 + V2 → V_third_cube = V3 - V_total → 
  V_third_cube^(1/3 : ℚ) = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end edge_of_third_cube_l142_142085


namespace interior_diagonal_length_l142_142893

theorem interior_diagonal_length 
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 150)
  (h2 : 4 * (a + b + c) = 60) :
  sqrt (a^2 + b^2 + c^2) = 5 * sqrt 3 :=
by
  sorry

end interior_diagonal_length_l142_142893


namespace trig_system_solution_l142_142075

theorem trig_system_solution :
  ∃ l k : ℤ, ∀ x y : ℝ,
    (cos x = 2 * (cos y)^3 ∧ sin x = 2 * (sin y)^3) ↔
    (x = 2 * l * Real.pi + k * Real.pi / 2 + Real.pi / 4 ∧
     y = k * Real.pi / 2 + Real.pi / 4) := by
  sorry

end trig_system_solution_l142_142075


namespace check_bag_correct_l142_142827

-- Define the conditions as variables and statements
variables (uber_to_house : ℕ) (uber_to_airport : ℕ) (check_bag : ℕ)
          (security : ℕ) (wait_for_boarding : ℕ) (wait_for_takeoff : ℕ) (total_time : ℕ)

-- Assign the given conditions
def given_conditions : Prop :=
  uber_to_house = 10 ∧
  uber_to_airport = 5 * uber_to_house ∧
  security = 3 * check_bag ∧
  wait_for_boarding = 20 ∧
  wait_for_takeoff = 2 * wait_for_boarding ∧
  total_time = 180

-- Define the question as a statement
def check_bag_time (check_bag : ℕ) : Prop :=
  check_bag = 15

-- The Lean theorem based on the problem, conditions, and answer
theorem check_bag_correct :
  given_conditions uber_to_house uber_to_airport check_bag security wait_for_boarding wait_for_takeoff total_time →
  check_bag_time check_bag :=
by
  intros h
  sorry

end check_bag_correct_l142_142827


namespace irreducible_fraction_l142_142847

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end irreducible_fraction_l142_142847


namespace part1_line_l1_part2_line_l2_1_part2_line_l2_2_l142_142718

-- Definitions for the given lines and points
def line_l (x y : ℝ) : Prop := 2 * x + y + 1 = 0
def point_A := (3, 2 : ℝ)
def point_P := (3, 0 : ℝ)

-- The equations we need to prove
def line_l1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def line_l2_1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line_l2_2 (x y : ℝ) : Prop := 2 * x + y - 11 = 0

-- Prove the equations hold using the given conditions
theorem part1_line_l1 :
  (∀ x y : ℝ, line_l x y → ∃ x y : ℝ, line_l1 x y) :=
sorry

theorem part2_line_l2_1 :
  (∀ x y : ℝ, line_l x y → ∃ x y : ℝ, line_l2_1 x y) :=
sorry

theorem part2_line_l2_2 :
  (∀ x y : ℝ, line_l x y → ∃ x y : ℝ, line_l2_2 x y) :=
sorry

end part1_line_l1_part2_line_l2_1_part2_line_l2_2_l142_142718


namespace determine_placemat_length_l142_142960

theorem determine_placemat_length :
  ∃ (y : ℝ), ∀ (r : ℝ), r = 5 →
  (∀ (n : ℕ), n = 8 →
  (∀ (w : ℝ), w = 1 →
  y = 10 * Real.sin (5 * Real.pi / 16))) :=
by
  sorry

end determine_placemat_length_l142_142960


namespace area_of_quadrilateral_APBQ_l142_142839

-- Define points, distances, and circles based on the conditions
variables {P Q A B : Type} [metric_space P] [metric_space Q] [metric_space A] [metric_space B]
variables (dist_PQ : dist P Q = 3) 
variables (radius_P : ∀ (x : P), dist x A ≤ sqrt 3)
variables (radius_Q : ∀ (y : Q), dist y A ≤ 3)
variables (intersect_circles : ∃ A B : P, dist P A = sqrt 3 ∧ dist Q A = 3 ∧ dist P B = sqrt 3 ∧ dist Q B = 3)

-- Statement for the proof required
theorem area_of_quadrilateral_APBQ : 
  ∀ (P Q A B : Type) [metric_space P] [metric_space Q] [metric_space A] [metric_space B],
  dist P Q = 3 → 
  (∀ (x : P), dist x A ≤ sqrt 3) →
  (∀ (y : Q), dist y A ≤ 3) →
  (∃ A B : P, dist P A = sqrt 3 ∧ dist Q A = 3 ∧ dist P B = sqrt 3 ∧ dist Q B = 3) →
  area APBQ = (3 * sqrt 5) / 2 := 
sorry

end area_of_quadrilateral_APBQ_l142_142839


namespace special_divisors_of_factorial_50_l142_142385

noncomputable def number_of_special_divisors : ℕ :=
let primes_up_to_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
let num_primes := primes_up_to_50.length in
let num_two_prime_products := num_primes * (num_primes - 1) / 2 in
num_primes + num_two_prime_products

theorem special_divisors_of_factorial_50 : number_of_special_divisors = 120 :=
by 
  sorry

end special_divisors_of_factorial_50_l142_142385


namespace four_buildings_possible_five_buildings_impossible_l142_142153

-- Problem for four buildings
theorem four_buildings_possible :
  ∃ (buildings : Finset Point) (P : Point), 
    (∀ (i j k l : buildings), ∃ (view_point : Point), 
      viewing_point_conditions i j k l view_point) :=
sorry

-- Problem for five buildings
theorem five_buildings_impossible :
  ¬ ∃ (buildings : Finset Point) (P : Point), 
      (∀ (i j k l m : buildings), ∃ (view_point : Point), 
        viewing_point_conditions i j k l m view_point) :=
sorry

end four_buildings_possible_five_buildings_impossible_l142_142153


namespace clock_angle_8_30_l142_142737

open Real

def minute_hand_angle (minutes : ℕ) : ℝ := 6 * minutes
def hour_hand_angle (hours : ℕ) (minutes : ℕ) : ℝ := 30 * hours + (minutes / 2)

theorem clock_angle_8_30 (minute_hand_at_6 : minute_hand_angle 30 = 180) (hour_hand_between_8_and_9 : hour_hand_angle 8 30 = 255) :
  abs (hour_hand_angle 8 30 - minute_hand_angle 30) = 75 :=
by sorry

end clock_angle_8_30_l142_142737


namespace five_digit_probability_l142_142585

theorem five_digit_probability : 
  let digits := [1, 2, 3, 4, 5] in
  let permutations := List.permutations digits in
  let isValid (l : List ℕ) := 
    (l.length = 5 ∧ 
     List.nth l 0 > List.nth l 1 ∧ 
     List.nth l 1 > List.nth l 2 ∧ 
     List.nth l 2 < List.nth l 3 ∧ 
     List.nth l 3 < List.nth l 4) in
  (List.filter isValid permutations).length / permutations.length = 1 / 20 :=
sorry

end five_digit_probability_l142_142585


namespace number_of_males_is_one_part_l142_142503

-- Define the total population
def population : ℕ := 480

-- Define the number of divided parts
def parts : ℕ := 3

-- Define the population part represented by one square.
def part_population (total_population : ℕ) (n_parts : ℕ) : ℕ :=
  total_population / n_parts

-- The Lean statement for the problem
theorem number_of_males_is_one_part : part_population population parts = 160 :=
by
  -- Proof omitted
  sorry

end number_of_males_is_one_part_l142_142503


namespace stamps_difference_l142_142500

theorem stamps_difference (x : ℕ) (h1: 5 * x / 3 * x = 5 / 3)
(h2: (5 * x - 12) / (3 * x + 12) = 4 / 3) : 
(5 * x - 12) - (3 * x + 12) = 32 := by
sorry

end stamps_difference_l142_142500


namespace max_value_sin_cos_sum_l142_142270

theorem max_value_sin_cos_sum :
  ∃ x : ℝ, 
    (∀ y : ℝ, 
      (sin (y + π/3) + cos (y - π/6)) ≤ sqrt 3) ∧
    (sin (x + π/3) + cos (x - π/6)) = sqrt 3 :=
sorry

end max_value_sin_cos_sum_l142_142270


namespace LCM_of_numbers_with_HCF_and_ratio_l142_142556

theorem LCM_of_numbers_with_HCF_and_ratio (a b x : ℕ)
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x)
  (h3 : ∀ y : ℕ, y ∣ a → y ∣ b → y ∣ x)
  (hx : x = 5) :
  Nat.lcm a b = 60 := 
by
  sorry

end LCM_of_numbers_with_HCF_and_ratio_l142_142556


namespace race_track_radius_l142_142495

theorem race_track_radius (C_inner : ℝ) (width : ℝ) (r_outer : ℝ) : 
  C_inner = 440 ∧ width = 14 ∧ r_outer = (440 / (2 * Real.pi) + 14) → r_outer = 84 :=
by
  intros
  sorry

end race_track_radius_l142_142495


namespace explicit_expression_and_inequality_l142_142712

def is_odd_function {α β : Type*} [AddGroup α] [Neg β] (f : α → β) : Prop :=
∀ x, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 1
  else 1 - x

theorem explicit_expression_and_inequality :
  (∀ x < 0, f x = -x - 1) ∧ (is_odd_function f) →
  (∀ x, f x = if x < 0 then -x - 1 else 1 - x) ∧
  ({x : ℝ | f x > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 1}) :=
by
  intros h
  sorry

end explicit_expression_and_inequality_l142_142712


namespace part1_part2_part3_l142_142824

def Sn (n : ℕ) : Set ℕ := { x | 1 ≤ x ∧ x ≤ n }

def capacity (X : Set ℕ) : ℕ := X.to_finset.sum id

def is_odd_subset (X : Set ℕ) : Prop := capacity X % 2 = 1

def is_even_subset (X : Set ℕ) : Prop := capacity X % 2 = 0

theorem part1 (n : ℕ) : 
  (Set.filter is_odd_subset (𝒫 (Sn n))).to_finset.card = 
  (Set.filter is_even_subset (𝒫 (Sn n))).to_finset.card := 
sorry

theorem part2 (n : ℕ) (h : 3 ≤ n) : 
  (Set.filter is_odd_subset (𝒫 (Sn n))).to_finset.sum capacity = 
  (Set.filter is_even_subset (𝒫 (Sn n))).to_finset.sum capacity := 
sorry

theorem part3 (n : ℕ) (h : 3 ≤ n) : 
  (Set.filter is_odd_subset (𝒫 (Sn n))).to_finset.sum capacity = 
  2^(n-3) * n * (n + 1) := 
sorry

end part1_part2_part3_l142_142824


namespace probability_each_mailbox_has_at_least_one_letter_l142_142141

noncomputable def probability_mailbox (total_letters : ℕ) (mailboxes : ℕ) : ℚ := 
  let total_ways := mailboxes ^ total_letters
  let favorable_ways := Nat.choose total_letters (mailboxes - 1) * (mailboxes - 1).factorial
  favorable_ways / total_ways

theorem probability_each_mailbox_has_at_least_one_letter :
  probability_mailbox 3 2 = 3 / 4 := by
  sorry

end probability_each_mailbox_has_at_least_one_letter_l142_142141


namespace graduation_ceremony_arrangements_l142_142872

-- Lean proof statement
theorem graduation_ceremony_arrangements :
  let events := {A, B, C, D, E, F}
  let first_three := {1, 2, 3}
  let bc_together := [{B, C}, {C, B}]
  let all_permutations := events.permutations
  (∃ p ∈ all_permutations, (A ∈ {p[1], p[2], p[3]}) ∧ (∃ x ∈ bc_together, x = {p[i], p[i+1]})) → 
  120 := sorry

end graduation_ceremony_arrangements_l142_142872


namespace number_of_flags_l142_142627

theorem number_of_flags (colors : Finset ℕ) (stripes : ℕ) (h_colors : colors.card = 3) (h_stripes : stripes = 3) : 
  (colors.card ^ stripes) = 27 := 
by
  sorry

end number_of_flags_l142_142627


namespace AC_interval_sum_l142_142005

def triangle_ABC (A B C D : Type) [HasLen A B 15] [AngleBisector A C D] [HasLen C D 5] (x : ℝ) : Prop :=
  let AC := x
  let BD := 75 / x in
  (5 < x ∧ x < 25)

theorem AC_interval_sum (A B C D : Type) [HasLen A B 15] [AngleBisector A C D] [HasLen C D 5] :
  let m := 5
  let n := 25 in
  m + n = 30 :=
begin
  sorry
end

end AC_interval_sum_l142_142005


namespace triangle_y_difference_l142_142001

theorem triangle_y_difference :
  ∀ (y : ℕ), 
    (y + 8 > 11) ∧ (y + 11 > 8) ∧ (8 + 11 > y) →
    (max (filter (λ x, (x + 8 > 11) ∧ (x + 11 > 8) ∧ (8 + 11 > x)) (list.range 20)) - 
     min (filter (λ x, (x + 8 > 11) ∧ (x + 11 > 8) ∧ (8 + 11 > x)) (list.range 20))) = 14 := 
by {
  sorry
}

end triangle_y_difference_l142_142001


namespace triangle_area_correct_l142_142646

noncomputable def triangle_area {a b C : ℝ} (h_a : a = 15) (h_b : b = 12) (h_C : C = Real.pi / 3) : ℝ :=
  1 / 2 * a * b * Real.sin C

theorem triangle_area_correct : triangle_area (by rfl) (by rfl) (by rfl) ≈ 77.94 :=
  sorry

end triangle_area_correct_l142_142646


namespace max_parts_by_two_triangles_max_parts_by_two_rectangles_max_parts_by_two_convex_ngons_l142_142008

-- a) Maximum number of parts the plane can be divided into by two triangles is 8.
theorem max_parts_by_two_triangles : ∀ (triangle1 triangle2 : Set Point),
    (∃ parts : ℕ, divide_plane_by_polygons 2 triangle1 triangle2 = parts) → 
    max_parts_by_two_triangles triangle1 triangle2 = 8 :=
begin
  sorry
end

-- b) Maximum number of parts the plane can be divided into by two rectangles is 10.
theorem max_parts_by_two_rectangles : ∀ (rectangle1 rectangle2 : Set Point),
    (∃ parts : ℕ, divide_plane_by_polygons 2 rectangle1 rectangle2 = parts) → 
    max_parts_by_two_rectangles rectangle1 rectangle2 = 10 :=
begin
  sorry
end

-- c) Maximum number of parts the plane can be divided into by two convex n-gons is 2n + 2.
theorem max_parts_by_two_convex_ngons : ∀ (n : ℕ) (ngon1 ngon2 : Set Point),
    (is_convex n ngon1) ∧ (is_convex n ngon2) →
    (∃ parts : ℕ, divide_plane_by_polygons 2 ngon1 ngon2 = parts) → 
    max_parts_by_two_convex_ngons n ngon1 ngon2 = 2 * n + 2 :=
begin
  sorry
end

end max_parts_by_two_triangles_max_parts_by_two_rectangles_max_parts_by_two_convex_ngons_l142_142008


namespace average_score_of_all_matches_is_36_l142_142043

noncomputable def average_score_of_all_matches
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) : ℝ :=
  (x + y + a + b + c) / 5

theorem average_score_of_all_matches_is_36
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) :
  average_score_of_all_matches x y a b c h1 h2 h3x h3y h3a h3b h3c h4 = 36 := 
  by 
  sorry

end average_score_of_all_matches_is_36_l142_142043


namespace monotonic_decreasing_range_l142_142724

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

theorem monotonic_decreasing_range (a : ℝ) : (-1 ≤ a ∧ a ≤ 0) ↔ ∀ x ∈ set.Iic (a / 3), deriv (g a) x ≤ 0 :=
by
  sorry

end monotonic_decreasing_range_l142_142724


namespace two_numbers_divisor_property_l142_142113

theorem two_numbers_divisor_property (s : Finset ℕ) (h1 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) (h2 : s.card = 1008) 
  : ∃ a b ∈ s, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end two_numbers_divisor_property_l142_142113


namespace first_group_men_l142_142946

theorem first_group_men (M : ℕ) :
  (∀ (men_work : ℕ → ℝ → ℝ → ℝ), (men_work 5 36 1.2 = 36 / (5 * 1.2)) → 
   (let work_per_man_per_day := men_work 5 36 1.2 in
    (48 / 2) = M * work_per_man_per_day)) → 
  (M = 4) :=
by
  sorry

end first_group_men_l142_142946


namespace eight_tuples_multiple_of_three_l142_142993

theorem eight_tuples_multiple_of_three :
  let S := {ε : Fin 8 → ℤ | ∀ i, ε i = 1 ∨ ε i = -1} in
  let sum_eps := λ (ε : Fin 8 → ℤ), ∑ i in Finset.univ, (↑i + 1) * ε i in
  (∃ t ∈ S, ∃ n ∈ S, (sum_eps t) % 3 = 0 ∨ (sum_eps n) % 3 = 0) →
  Finset.card {ε ∈ S | (sum_eps ε) % 3 = 0} = 88 := 
begin
  sorry
end

end eight_tuples_multiple_of_three_l142_142993


namespace color_change_probability_l142_142198

-- Definitions based directly on conditions in a)
def light_cycle_duration := 93
def change_intervals_duration := 15
def expected_probability := 5 / 31

-- The Lean 4 statement for the proof problem
theorem color_change_probability :
  (change_intervals_duration / light_cycle_duration) = expected_probability :=
by
  sorry

end color_change_probability_l142_142198


namespace part1_part2_l142_142302

-- Definition of the sequence a_{n} given the condition
def sequence_a (n : ℕ) : ℝ :=
if n = 1 then 1
else 1 / (2 * n - 1)

-- Condition a_{1} + 3a_{2} + ... + (2n-1)a_{n} = n
def condition (n : ℕ) : Prop := 
∑ i in finset.range n, (2 * i + 1) * sequence_a (i + 1) = n

-- Definition of the sequence c_{n} based on a_{n}
def sequence_c (n : ℕ) : ℝ :=
if n % 2 = 1 then 1 / (19 * sequence_a n)
else sequence_a n * sequence_a (n + 2)

-- Prove that \{\frac{1}{a_n}\} is an arithmetic sequence
theorem part1 (n : ℕ) (h : condition n) : 
  ∃ d : ℝ, ∀ k : ℕ, k < n → (1 / sequence_a (k + 1)) = 1 + d * ↑k :=
sorry

-- Find the sum of the first 2n terms of the sequence \{c_{n}\}
theorem part2 (n : ℕ) (h : condition (2 * n)) : 
  ∑ i in finset.range (2 * n), sequence_c (i + 1) = 
  (1 / 19) * n * (2 * n - 1) + (1 / 12) - (1 / (16 * n + 12)) :=
sorry

end part1_part2_l142_142302


namespace BoatsRUs_total_canoes_built_l142_142215

theorem BoatsRUs_total_canoes_built : 
  let a := 5
  let r := 3
  S_4 = a * (r^4 - 1) / (r - 1)
  S_1 := a
  S_2 := a * r
  S_3 := a * r^2
  S_4 := a * r^3
  (S_1 + S_2 + S_3 + S_4) = 200 :=
by 
  let a := 5
  let r := 3
  let S_1 := a
  let S_2 := a * r
  let S_3 := a * r^2
  let S_4 := a * r^3
  let total := (S_1 + S_2 + S_3 + S_4)
  have term_relation : S_4 = (a * (r^4 - 1) / (r - 1)), sorry
  exact term_relation
where total = 200

end BoatsRUs_total_canoes_built_l142_142215


namespace range_of_f_is_real_l142_142325

noncomputable def f (x : ℝ) (m : ℝ) := Real.log (5^x + 4 / 5^x + m)

theorem range_of_f_is_real (m : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x m = y) ↔ m ≤ -4 :=
sorry

end range_of_f_is_real_l142_142325


namespace primes_min_sum_l142_142518

theorem primes_min_sum : ∃ numbers : List ℕ, (∀ n ∈ numbers, Nat.Prime n) ∧ numbers.sum = 225 ∧ (List.sort (· ≤ ·) (numbers.bind Nat.digits)).dedup = List.range' 1 9 :=
by
  sorry

end primes_min_sum_l142_142518


namespace average_percent_score_l142_142052

theorem average_percent_score : 
  let num_students := 120
  let scores : List (ℕ × ℕ) := [(95, 10), (85, 20), (75, 40), (65, 30), (55, 15), (45, 5)]
  let total_weighted_score := List.sum (scores.map (fun (s, n) => s * n))
  let average_score := (total_weighted_score : ℕ) / num_students
  average_score = 72.0833 :=
by
  -- Define number of students
  let num_students := 120
  
  -- List of (score, number of students)
  let scores : List (ℕ × ℕ) := [(95, 10), (85, 20), (75, 40), (65, 30), (55, 15), (45, 5)]
  
  -- Calculate total weighted score
  let total_weighted_score := List.sum (scores.map (fun (s, n) => s * n))
  
  -- Calculate average score
  let average_score := (total_weighted_score : ℕ) / num_students
  
  -- Skip the actual proof
  sorry

end average_percent_score_l142_142052


namespace polynomial_divisibility_l142_142398

theorem polynomial_divisibility (p q : ℝ) :
    let f (x : ℝ) := x^5 - x^4 + x^3 - p * x^2 + q * x + 9 in
    f (-3) = 0 ∧ f (2) = 0 → p = -130.5 ∧ q = -277.5 :=
by
  intro f h
  sorry

end polynomial_divisibility_l142_142398


namespace maximum_smallest_angle_l142_142397

-- Definition of points on the plane
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

-- Function to calculate the angle between three points (p1, p2, p3)
def angle (p1 p2 p3 : Point2D) : ℝ := 
  -- Placeholder for the actual angle calculation
  sorry

-- Condition: Given five points on a plane
variables (A B C D E : Point2D)

-- Maximum value of the smallest angle formed by any triple is 36 degrees
theorem maximum_smallest_angle :
  ∃ α : ℝ, (∀ p1 p2 p3 : Point2D, α ≤ angle p1 p2 p3) ∧ α = 36 :=
sorry

end maximum_smallest_angle_l142_142397


namespace range_of_k_l142_142890

theorem range_of_k (k : ℝ) : 
  (∃ x y : ℝ, k * x - y + k = 0 ∧ x^2 + y^2 - 2 * x = 0) ↔ (- real.sqrt 3 / 3 ≤ k ∧ k ≤ real.sqrt 3 / 3) :=
by sorry

end range_of_k_l142_142890


namespace count_multiples_of_7_l142_142359

theorem count_multiples_of_7 (low high : ℕ) (hlow : low = 200) (hhigh : high = 400) : 
  (card {n | low ≤ n ∧ n ≤ high ∧ n % 7 = 0}) = 29 := by
  sorry

end count_multiples_of_7_l142_142359


namespace compound_interest_amount_l142_142508

noncomputable def P_SI : ℝ := 2625.0000000000027
noncomputable def r_SI : ℝ := 0.08
noncomputable def r_CI : ℝ := 0.10
noncomputable def t : ℝ := 2

theorem compound_interest_amount :
  let SI := P_SI * r_SI * t,
      CI := λ (P_CI : ℝ), P_CI * (((1 + r_CI)^t) - 1) in
  ∃ P_CI : ℝ, SI = (1/2) * CI P_CI ∧ P_CI = 4000 :=
by {
  let SI := P_SI * r_SI * t,
  let CI := λ (P_CI : ℝ), P_CI * (((1 + r_CI)^t) - 1),
  use 4000,
  split,
  sorry,
  refl
}

end compound_interest_amount_l142_142508


namespace line_through_ellipse_bisector_triangle_area_l142_142309

theorem line_through_ellipse_bisector :
  ∀ (l : ℝ → ℝ → Prop),
  (∃ A B : ℝ × ℝ, l A.1 A.2 ∧ l B.1 B.2 ∧ l = (λ x y, y - (-1 / 2) * x = 0) ∧
  (A.1 + B.1) / 2 = sqrt 3 ∧
  (A.2 + B.2) / 2 = sqrt 3 / 2 ∧
  (A.1^2 / 16 + A.2^2 / 4 = 1) ∧ (B.1^2 / 16 + B.2^2 / 4 = 1)) :=
sorry

theorem triangle_area :
  ∀ (F1 A B : ℝ × ℝ),
  F1 = (-2 * sqrt 3, 0) ∧
  (∃ (d : ℝ), d = abs (1 / 2 * F1.1 - sqrt 3) / sqrt (1 + (1 / 2)^2) ∧
  |A.1 - B.1| = 2 * sqrt 3) ∧
  (F1.1^2 / 16 + F1.2^2 / 4 = 1) ∧ (A.1^2 / 16 + A.2^2 / 4 = 1) ∧
  (B.1^2 / 16 + B.2^2 / 4 = 1) ∧
  |A.1 - B.1| = 5 ∧
  d = 4 * sqrt 15 / 5 ∧
  (abs ((A.1 * A.2) + (A.1 * B.2) + (B.1 * A.2) + (B.1 * B.2)) / 2) = 2 * sqrt 15 :=
sorry

end line_through_ellipse_bisector_triangle_area_l142_142309


namespace factorize_expression_l142_142255

theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x^2 + 8 * x = 2 * x * (x - 2) ^ 2 := 
sorry

end factorize_expression_l142_142255


namespace iterative_method_converges_to_root_l142_142423

noncomputable def seq (x : ℕ → ℝ) : ℕ → ℝ
| 0     := 1
| (n+1) := Real.sqrt (1 + 1 / x n)

theorem iterative_method_converges_to_root :
  ∃ x_0 : ℝ, x_0 > 0 ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, abs (seq seq n - x_0) < ε) ∧ x_0 ^ 3 - x_0 - 1 = 0 :=
sorry

end iterative_method_converges_to_root_l142_142423


namespace two_numbers_divisor_property_l142_142117

theorem two_numbers_divisor_property (s : Finset ℕ) (h1 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) (h2 : s.card = 1008) 
  : ∃ a b ∈ s, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end two_numbers_divisor_property_l142_142117


namespace problem1_problem2_l142_142413

section ArithmeticSequence

variable {a : ℕ → ℤ} {a1 a5 a8 a6 a4 d : ℤ}

-- Problem 1: Prove that if a_5 = -1 and a_8 = 2, then a_1 = -5 and d = 1
theorem problem1 
  (h1 : a 5 = -1) 
  (h2 : a 8 = 2)
  (h3 : ∀ n, a n = a1 + n * d) : 
  a1 = -5 ∧ d = 1 := 
sorry 

-- Problem 2: Prove that if a_1 + a_6 = 12 and a_4 = 7, then a_9 = 17
theorem problem2 
  (h1 : a1 + a 6 = 12) 
  (h2 : a 4 = 7)
  (h3 : ∀ n, a n = a1 + n * d) 
  (h4 : ∀ m (hm : m ≠ 0), a1 = a 1): 
   a 9 = 17 := 
sorry

end ArithmeticSequence

end problem1_problem2_l142_142413


namespace largest_k_for_sum_of_consecutive_integers_l142_142269

theorem largest_k_for_sum_of_consecutive_integers 
  (k : ℕ) (h : 3 ^ 12 = (finset.range k).sum(λ i, i + n)) : k = 729 :=
sorry

end largest_k_for_sum_of_consecutive_integers_l142_142269


namespace nina_jewelry_sales_l142_142054

theorem nina_jewelry_sales :
  ∃ (earrings_sold : ℕ),
  let 
    necklace_price := 25,
    bracelet_price := 15,
    earring_price := 10,
    ensemble_price := 45,
    necklaces_sold := 5,
    bracelets_sold := 10,
    ensembles_sold := 2,
    total_earnings := 565,
    sales_necklaces := necklaces_sold * necklace_price,
    sales_bracelets := bracelets_sold * bracelet_price,
    sales_ensembles := ensembles_sold * ensemble_price,
    total_known_sales := sales_necklaces + sales_bracelets + sales_ensembles,
    sales_earrings := total_earnings - total_known_sales
  in earrings_sold * earring_price = sales_earrings ∧ earrings_sold = 20 :=
by sorry

end nina_jewelry_sales_l142_142054


namespace no_five_coprime_two_digit_composites_l142_142253

open Nat

/-- A number is composite if it has more than two distinct divisors -/
def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ (∃ d : ℕ, d < n ∧ 1 < d ∧ d ∣ n)

/-- Two numbers are coprime if their gcd is 1 -/
def coprime (m n : ℕ) : Prop :=
  gcd m n = 1

/-- There do not exist five different two-digit composite numbers, 
such that each pair is coprime -/
theorem no_five_coprime_two_digit_composites : ¬ ∃ a b c d e : ℕ,
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  10 ≤ a ∧ a < 100 ∧ is_composite a ∧
  10 ≤ b ∧ b < 100 ∧ is_composite b ∧
  10 ≤ c ∧ c < 100 ∧ is_composite c ∧
  10 ≤ d ∧ d < 100 ∧ is_composite d ∧
  10 ≤ e ∧ e < 100 ∧ is_composite e ∧
  coprime a b ∧ coprime a c ∧ coprime a d ∧ coprime a e ∧
  coprime b c ∧ coprime b d ∧ coprime b e ∧
  coprime c d ∧ coprime c e ∧
  coprime d e :=
by
  sorry

end no_five_coprime_two_digit_composites_l142_142253


namespace conjugate_of_z_l142_142324

theorem conjugate_of_z (z : ℂ) (hz : z * (1 + complex.I) = 1 - complex.I) : complex.conj z = complex.I := 
sorry

end conjugate_of_z_l142_142324


namespace numbers_divisor_property_l142_142135

theorem numbers_divisor_property (S : Finset ℕ) (h₁ : Finset.card S = 1008) (h₂ : ∀ x ∈ S, x ≤ 2014) :
  ∃ a b ∈ S, a ∣ b ∨ b ∣ a :=
by
  sorry

end numbers_divisor_property_l142_142135


namespace sum_of_series_l142_142249

def i : ℂ := complex.I

def power_of_i (k : ℕ) : ℂ :=
  i^k

theorem sum_of_series :
  (∑ k in finset.range (2017 + 1), power_of_i k) = i :=
by
  sorry

end sum_of_series_l142_142249


namespace num_natural_numbers_divisible_by_7_l142_142383

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l142_142383


namespace range_of_m_l142_142727

theorem range_of_m (m : ℝ) (p : Prop) (q : Prop)
  (hp : (2 * m)^2 - 4 ≥ 0 ↔ p)
  (hq : 1 < (Real.sqrt (5 + m)) / (Real.sqrt 5) ∧ (Real.sqrt (5 + m)) / (Real.sqrt 5) < 2 ↔ q)
  (hnq : ¬q = False)
  (hpq : (p ∧ q) = False) :
  0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l142_142727


namespace find_special_numbers_l142_142644

def is_divisible (n m : ℕ) : Prop := n % m = 0

def valid_digits (l : List ℕ) : Prop :=
  l ~ [1, 2, 3, 4, 5, 6]

def valid_number (l : List ℕ) : Prop :=
  valid_digits l ∧ 
  is_divisible (l.take 1).foldr (λ x d, x + d * 10) 0 1 ∧
  is_divisible (l.take 2).foldr (λ x d, x + d * 10) 0 2 ∧
  is_divisible (l.take 3).foldr (λ x d, x + d * 10) 0 3 ∧
  is_divisible (l.take 4).foldr (λ x d, x + d * 10) 0 4 ∧
  is_divisible (l.take 5).foldr (λ x d, x + d * 10) 0 5 ∧
  is_divisible (l.take 6).foldr (λ x d, x + d * 10) 0 6

theorem find_special_numbers : 
  ∀ l : List ℕ, valid_number l → l = [1, 2, 3, 6, 5, 4] ∨ l = [3, 2, 1, 6, 5, 4] :=
by
  sorry

end find_special_numbers_l142_142644


namespace numbers_divisor_property_l142_142136

theorem numbers_divisor_property (S : Finset ℕ) (h₁ : Finset.card S = 1008) (h₂ : ∀ x ∈ S, x ≤ 2014) :
  ∃ a b ∈ S, a ∣ b ∨ b ∣ a :=
by
  sorry

end numbers_divisor_property_l142_142136


namespace conic_sections_l142_142995

theorem conic_sections (x y : ℝ) :
  y^4 - 9 * x^4 = 3 * y^2 - 4 →
  (∃ c : ℝ, (c = 5/2 ∨ c = 1) ∧ y^2 - 3 * x^2 = c) :=
by
  sorry

end conic_sections_l142_142995


namespace union_M_N_is_U_l142_142044

-- Defining the universal set as the set of real numbers
def U : Set ℝ := Set.univ

-- Defining the set M
def M : Set ℝ := {x | x > 0}

-- Defining the set N
def N : Set ℝ := {x | x^2 >= x}

-- Stating the theorem that M ∪ N = U
theorem union_M_N_is_U : M ∪ N = U :=
  sorry

end union_M_N_is_U_l142_142044


namespace cot_squared_sum_l142_142025

noncomputable def T : set ℝ := {x : ℝ | real.pi / 6 < x ∧ x < real.pi / 3}

theorem cot_squared_sum : 
  (∑ x in T, real.cot x ^ 2) = 1 :=
by 
  sorry

end cot_squared_sum_l142_142025


namespace arithmetic_sequence_20th_term_l142_142238

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 5
  let n := 20
  let a_n := a + (n - 1) * d
  a_n = 97 := by
  sorry

end arithmetic_sequence_20th_term_l142_142238


namespace simplify_expression_l142_142481

variable {s r : ℝ}

theorem simplify_expression :
  (2 * s^2 + 5 * r - 4) - (3 * s^2 + 9 * r - 7) = -s^2 - 4 * r + 3 := 
by
  sorry

end simplify_expression_l142_142481


namespace final_qualification_median_l142_142997

section
variables {scores : list ℕ} (h_scores_length : scores.length = 13)
(h_distinct : scores.nodup)
(h_sorted : scores.sorted (≤)) 
{xy_score : ℕ}

def qualifies_for_finals (xy_score : ℕ) (scores : list ℕ) : Prop :=
  xy_score ∈ scores.take 7 -- since the top 6 + 1 median are in the top 7

theorem final_qualification_median 
  (h_scores_length : scores.length = 13) 
  (h_distinct : scores.nodup) 
  (h_sorted : scores.sorted (≤)) 
  (xy_score : ℕ) :
  qualifies_for_finals xy_score scores ↔ 
  xy_score ≤ scores.nth_le 6 h_scores_length.succ_pos :=
sorry
end

end final_qualification_median_l142_142997


namespace largest_possible_number_on_blackboard_l142_142879

theorem largest_possible_number_on_blackboard (nums : List ℕ) (h : nums = [2^0, 2^1, 2^2, ... , 2^16]) :
  (∃ n, nums = [n] ∧ n ≤ 65535) → n = 65535 := sorry

end largest_possible_number_on_blackboard_l142_142879


namespace arrangement_of_athletes_l142_142072

def num_arrangements (n : ℕ) (available_tracks_for_A : ℕ) (permutations_remaining : ℕ) : ℕ :=
  n * available_tracks_for_A * permutations_remaining

theorem arrangement_of_athletes :
  num_arrangements 2 3 24 = 144 :=
by
  sorry

end arrangement_of_athletes_l142_142072


namespace two_numbers_divisor_property_l142_142114

theorem two_numbers_divisor_property (s : Finset ℕ) (h1 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) (h2 : s.card = 1008) 
  : ∃ a b ∈ s, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end two_numbers_divisor_property_l142_142114


namespace fraction_expression_of_repeating_decimal_l142_142145

theorem fraction_expression_of_repeating_decimal :
  ∃ (x : ℕ), x = 79061333 ∧ (∀ y : ℚ, y = 0.71 + 264 * (1/999900) → x / 999900 = y) :=
by
  sorry

end fraction_expression_of_repeating_decimal_l142_142145


namespace handshake_problem_l142_142920

-- Define the remainder operation
def r_mod (n : ℕ) (k : ℕ) : ℕ := n % k

-- Define the function F
def F (t : ℕ) : ℕ := r_mod (t^3) 5251

-- The lean theorem statement with the given conditions and expected results
theorem handshake_problem :
  ∃ (x y : ℕ),
    F x = 506 ∧
    F (x + 1) = 519 ∧
    F y = 229 ∧
    F (y + 1) = 231 ∧
    x = 102 ∧
    y = 72 :=
by
  sorry

end handshake_problem_l142_142920


namespace numbers_divisor_property_l142_142133

theorem numbers_divisor_property (S : Finset ℕ) (h₁ : Finset.card S = 1008) (h₂ : ∀ x ∈ S, x ≤ 2014) :
  ∃ a b ∈ S, a ∣ b ∨ b ∣ a :=
by
  sorry

end numbers_divisor_property_l142_142133


namespace CorrectStatement_l142_142542

def PositiveInteger (n : ℤ) := n > 0
def NegativeInteger (n : ℤ) := n < 0
def PositiveFraction (q : ℚ) := q > 0 ∧ q.denom ≠ 0
def NegativeFraction (q : ℚ) := q < 0 ∧ q.denom ≠ 0
def RationalNumber (q : ℚ) := True
def ZeroIsRational := ∃ q : ℚ, q = 0

theorem CorrectStatement :
  (PositiveInteger 1 ∨ PositiveFraction (1/2)) ∧ ¬(PositiveInteger 1 ∧ NegativeInteger (-1)) ∧ ¬((PositiveInteger 1 ∨ NegativeInteger (-1) ∨ PositiveFraction (1/2) ∨ NegativeFraction (-1/2)) ∧ RationalNumber 0) ∧ ZeroIsRational → (StatementA)
by
  sorry

end CorrectStatement_l142_142542


namespace count_multiples_of_7_l142_142357

theorem count_multiples_of_7 (low high : ℕ) (hlow : low = 200) (hhigh : high = 400) : 
  (card {n | low ≤ n ∧ n ≤ high ∧ n % 7 = 0}) = 29 := by
  sorry

end count_multiples_of_7_l142_142357


namespace pyramid_plane_angle_l142_142082

theorem pyramid_plane_angle (α β : ℝ) (hα0 : 0 < α) (hα90 : α < π / 2) :
    let R := 1 in  -- Assume a unit radius for the simplicity of calculation
    let OA := R in
    let AD := R * sin α in
    let OD := R * cos α in
    let SO := R * tan β in
    let EO := (1 / 2) * SO in
    let angle_EDO := arctan (tan β / (2 * cos α)) in
    angle_EDO = arctan (tan β / (2 * cos α)) :=
by
  let R := 1  -- Assumption to simplify calculations; it doesn't affect the proportionality
  let OA := R
  let AD := R * sin α
  let OD := sqrt (R^2 - AD^2)
  let SO := R * tan β
  let EO := (1 / 2) * SO
  let angle_EDO := arctan (tan β / (2 * cos α))
  exact eq.refl angle_EDO

end pyramid_plane_angle_l142_142082


namespace perfect_cube_condition_l142_142442

theorem perfect_cube_condition (a b : ℕ) (h1 : 0 < b) (h2 : b < a) (h3 : ab(a - b) ∣ a^3 + b^3 + ab) : ∃ c : ℕ, ab = c^3 := 
sorry

end perfect_cube_condition_l142_142442


namespace solution_set_l142_142734

noncomputable def truncated_interval (x : ℝ) (n : ℤ) : Prop :=
n ≤ x ∧ x < n + 1

theorem solution_set (x : ℝ) (hx : ∃ n : ℤ, n > 0 ∧ truncated_interval x n) :
  2 ≤ x ∧ x < 8 :=
sorry

end solution_set_l142_142734


namespace circumsphere_surface_area_of_cuboid_l142_142182

theorem circumsphere_surface_area_of_cuboid :
  ∀ (l w h : ℕ), l = 2 → w = 1 → h = 1 → 
  let d := Real.sqrt (l^2 + w^2 + h^2)
  let r := d / 2
  4 * Real.pi * r^2 = 6 * Real.pi :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  let d := Real.sqrt (2^2 + 1^2 + 1^2)
  let r := d / 2
  have hd : d = Real.sqrt 6 := by sorry
  rw [hd]
  have hr : r = Real.sqrt 6 / 2 := by sorry
  rw [hr]
  norm_num
  sorry

end circumsphere_surface_area_of_cuboid_l142_142182


namespace value_of_expression_l142_142918

theorem value_of_expression (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 3) :
  (2 * a - (3 * b - 4 * c)) - ((2 * a - 3 * b) - 4 * c) = 24 := by
  sorry

end value_of_expression_l142_142918


namespace cardboard_placement_l142_142611

/--
  A lemma to prove the placement of cardboard pieces on a 10x10 grid.
  Part (1): Prove that arranging the pieces in the order 1x4, 1x3, 1x2, 1x1 always succeeds.
  Part (2): Show that placing the pieces in the ascending order 1x1, 1x2, 1x3, 1x4 can fail with a specific example.
--/
theorem cardboard_placement :
  (∃ positions_1x4 positions_1x3 positions_1x2 positions_1x1 : list (fin 10 × fin 10),
    no_overlap positions_1x4 positions_1x3 positions_1x2 positions_1x1 ∧
    no_common_points positions_1x4 positions_1x3 positions_1x2 positions_1x1 ∧
    align_along_edges positions_1x4 positions_1x3 positions_1x2 positions_1x1 ∧
    each_in_grid positions_1x4 positions_1x3 positions_1x2 positions_1x1)
∧
  (¬ ∃ positions_1x1 positions_1x2 positions_1x3 positions_1x4 : list (fin 10 × fin 10),
    no_overlap positions_1x1 positions_1x2 positions_1x3 positions_1x4 ∧
    no_common_points positions_1x1 positions_1x2 positions_1x3 positions_1x4 ∧
    align_along_edges positions_1x1 positions_1x2 positions_1x3 positions_1x4 ∧
    each_in_grid positions_1x1 positions_1x2 positions_1x3 positions_1x4 ∧
    provides_example_of_failure positions_1x1 positions_1x2 positions_1x3 positions_1x4) :=
by {
  sorry
}

end cardboard_placement_l142_142611


namespace min_alterations_to_make_sums_unique_l142_142239

def initial_matrix : matrix (fin 3) (fin 3) ℕ :=
  ![![4, 9, 2], ![9, 1, 6], ![4, 5, 7]]

def row_sums (m : matrix (fin 3) (fin 3) ℕ) : fin 3 → ℕ :=
  λ i, (finset.univ.sum (λ j, m i j))

def col_sums (m : matrix (fin 3) (fin 3) ℕ) : fin 3 → ℕ :=
  λ j, (finset.univ.sum (λ i, m i j))

theorem min_alterations_to_make_sums_unique :
  ∃ m' : matrix (fin 3) (fin 3) ℕ,
    (∃ i j₁ j₂, m' = initial_matrix.update i j₁ 10 .update i j₂ 4) ∧
    (∀ i₁ i₂, i₁ ≠ i₂ → row_sums m' i₁ ≠ row_sums m' i₂) ∧
    (∀ j₁ j₂, j₁ ≠ j₂ → col_sums m' j₁ ≠ col_sums m' j₂) :=
begin
  sorry, -- proof not required
end

end min_alterations_to_make_sums_unique_l142_142239


namespace a_6_eq_15_l142_142488

def sequence (n : ℕ) : ℕ → ℕ
| 0      := 1
| (k+1)  := if sequence k - 2 ∈ Nat ∧ (sequence k - 2) ∉ {sequence i | i < k} then
                sequence k - 2 
            else 
                3 * sequence k

theorem a_6_eq_15 : sequence 5 = 15 :=
sorry

end a_6_eq_15_l142_142488


namespace perfect_squares_count_2000_l142_142743

theorem perfect_squares_count_2000 : 
  let count := (1 to 44).filter (λ x, 
    let ones_digit := (x * x) % 10 in 
    ones_digit = 4 ∨ ones_digit = 5 ∨ ones_digit = 6).length
  in
  count = 22 :=
by
  sorry

end perfect_squares_count_2000_l142_142743


namespace cos_75_deg_l142_142234

theorem cos_75_deg :
  let deg_to_rad := λ θ : ℝ, θ * Real.pi / 180
  let cos_60 := Real.cos (deg_to_rad 60)
  let sin_60 := Real.sin (deg_to_rad 60)
  let cos_15 := Real.cos (deg_to_rad 15)
  let sin_15 := Real.sin (deg_to_rad 15)
  cos_60 = 1/2 ∧
  sin_60 = Real.sqrt 3 / 2 ∧
  cos_15 = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧
  sin_15 = (Real.sqrt 6 - Real.sqrt 2) / 4 →
  Real.cos (deg_to_rad 75) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_deg_l142_142234


namespace num_divisible_by_7_200_to_400_l142_142371

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l142_142371


namespace probability_non_perfect_power_200_l142_142880

def is_perfect_power (x : ℕ) : Prop := 
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b = x

def count_perfect_powers_up_to (n : ℕ) : ℕ := 
  Finset.card (Finset.filter is_perfect_power (Finset.range (n + 1)))

def probability_not_perfect_power (n : ℕ) : ℚ :=
  let total := n in
  let perfect_powers := count_perfect_powers_up_to n in
  (total - perfect_powers) / total

theorem probability_non_perfect_power_200 :
  probability_not_perfect_power 200 = 9 / 10 :=
by {
  -- statement placeholder
  sorry
}

end probability_non_perfect_power_200_l142_142880


namespace nested_abs_eq_solutions_l142_142386

theorem nested_abs_eq_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, (|||| x - 2 |- 2| - 2| - 2) = (|||| x - 3 |- 3| - 3| - 3)) ∧ s.card = 6 :=
begin
  sorry
end

end nested_abs_eq_solutions_l142_142386


namespace area_of_quadrilateral_AEDC_l142_142795

theorem area_of_quadrilateral_AEDC :
  ∀ (PE PD DE : ℕ), PE = 3 → PD = 4 → DE = 5 →
    (let AP := 2 * PD,
         CP := 2 * PE,
         area := 1 / 2 * (PE * AP + PD * CP + PE * CP + PD * AP)
     in area = 49) :=
by
  intro PE PD DE
  intros hPE hPD hDE
  let AP := 2 * PD
  let CP := 2 * PE
  let area := 1 / 2 * (PE * AP + PD * CP + PE * CP + PD * AP)
  have h1 : area = 1 / 2 * (3 * 8 + 4 * 6 + 3 * 6 + 4 * 8) := by
    rw [hPE, hPD]
  have h2 : area = 1 / 2 * 98 := by
    simp [h1]
  show area = 49, from by
    simp [h2]
  sorry -- Proof steps skipped

end area_of_quadrilateral_AEDC_l142_142795


namespace cone_volume_l142_142328

noncomputable def volume_of_cone (l : ℝ) (A : ℝ) : ℝ :=
  let r := (A / (l * real.pi))
  let h := real.sqrt (l^2 - r^2)
  (1 / 3) * real.pi * r^2 * h

theorem cone_volume : volume_of_cone 5 (20 * real.pi) = 16 * real.pi :=
by
  sorry

end cone_volume_l142_142328


namespace circle_has_greatest_symmetry_l142_142923

-- Define the number of lines of symmetry for each of the figures
def lines_of_symmetry (figure : Type) : ℕ → Prop := 
  figure = regular_pentagon → 5 ∨ 
  figure = isosceles_triangle → 1 ∨ 
  figure = parallelogram → 0 ∨ 
  figure = non_equilateral_rhombus → 2 ∨ 
  figure = circle → ∞

-- Define that the circle has the maximum number of lines of symmetry
theorem circle_has_greatest_symmetry (figures : List Type) (circle ∈ figures) (regular_pentagon ∈ figures) (isosceles_triangle ∈ figures) (parallelogram ∈ figures) (non_equilateral_rhombus ∈ figures) : 
  ∀ (f : Type), f ∈ figures → lines_of_symmetry f → lines_of_symmetry circle := 
by 
  sorry 

end circle_has_greatest_symmetry_l142_142923


namespace hockey_season_duration_l142_142903

theorem hockey_season_duration 
  (total_games : ℕ)
  (games_per_month : ℕ)
  (h_total : total_games = 182)
  (h_monthly : games_per_month = 13) : 
  total_games / games_per_month = 14 := 
by
  sorry

end hockey_season_duration_l142_142903


namespace arithmetic_progression_general_term_l142_142246

def sequence_arithmetic_progression (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sequence_distinct (a : ℕ → ℕ) := ∀ m n : ℕ, m ≠ n → a m ≠ a n

noncomputable def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) := (0 to n).sum (λ i, a i)

theorem arithmetic_progression_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) :
  sequence_distinct a →
  sequence_arithmetic_progression a →
  (a 3 * a 5 = 3 * a 7) →
  (sum_of_first_n_terms (λ n, a n) 3 = 9) →
  ∀ n : ℕ, a n = n + 1 :=
sorry

end arithmetic_progression_general_term_l142_142246


namespace residual_at_neg1_l142_142101

noncomputable def observed_value_at_neg1 := 3
noncomputable def predicted_value_at_neg1 := (-(-1) + 2.6 : ℝ)

theorem residual_at_neg1 :
  (observed_value_at_neg1 - predicted_value_at_neg1 = -0.6) := by
  -- Given conditions deducted from the table and regression equation
  have h_steps :
    (observed_value_at_neg1 - predicted_value_at_neg1 = 3 - (1 + 2.6)) := by
    sorry
  exact h_steps

end residual_at_neg1_l142_142101


namespace rubles_greater_than_seven_l142_142471

theorem rubles_greater_than_seven (x : ℕ) (h : x > 7) : ∃ a b : ℕ, x = 3 * a + 5 * b :=
sorry

end rubles_greater_than_seven_l142_142471


namespace angle_between_AD_and_BC_l142_142405

variables {a b c : ℝ} 
variables {θ : ℝ}
variables {α β γ δ ε ζ : ℝ} -- representing the angles

-- Conditions of the problem
def conditions (a b c : ℝ) (α β γ δ ε ζ : ℝ) : Prop :=
  (α + β + γ = 180) ∧ (δ + ε + ζ = 180) ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Definition of the theorem to prove the angle between AD and BC
theorem angle_between_AD_and_BC
  (a b c : ℝ) (α β γ δ ε ζ : ℝ)
  (h : conditions a b c α β γ δ ε ζ) :
  θ = Real.arccos ((|b^2 - c^2|) / a^2) :=
sorry

end angle_between_AD_and_BC_l142_142405


namespace divisible_by_7_in_range_200_to_400_l142_142376

theorem divisible_by_7_in_range_200_to_400 : 
  ∃ n : ℕ, 
    (∀ (x : ℕ), (200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ finset.range (201)) ∧ finset.card (finset.filter (λ x, 200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0) (finset.range 401)) = 29) := 
begin
  sorry
end

end divisible_by_7_in_range_200_to_400_l142_142376


namespace total_bill_l142_142636

theorem total_bill (total_friends : ℕ) (extra_payment : ℝ) (total_bill : ℝ) (paid_by_friends : ℝ) :
  total_friends = 8 → extra_payment = 2.50 →
  (7 * ((total_bill / total_friends) + extra_payment)) = total_bill →
  total_bill = 140 :=
by
  intros h1 h2 h3
  sorry

end total_bill_l142_142636


namespace path_length_traveled_by_A_l142_142850

theorem path_length_traveled_by_A 
  (ABCD: Type) 
  (AB CD BC DA : ℝ) 
  (h1 : AB = 3) 
  (h2 : CD = 3) 
  (h3 : BC = 5) 
  (h4 : DA = 5) 
  (rot90_clockwise_D : ∀(P: ℝ × ℝ), ℝ × ℝ)
  (rot90_clockwise_new_C : ∀(P: ℝ × ℝ), ℝ × ℝ) :
  ∃ len, len = (π * (sqrt 34 + 5)) / 2 :=
by
  sorry

end path_length_traveled_by_A_l142_142850


namespace minimum_value_of_A_l142_142692

open Real

noncomputable def A (x y z : ℝ) : ℝ :=
  ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x * y + y * z + z * x)

theorem minimum_value_of_A (x y z : ℝ) (h : 3 ≤ x) (h2 : 3 ≤ y) (h3 : 3 ≤ z) :
  ∃ v : ℝ, (∀ a b c : ℝ, 3 ≤ a ∧ 3 ≤ b ∧ 3 ≤ c → A a b c ≥ v) ∧ v = 1 :=
sorry

end minimum_value_of_A_l142_142692


namespace expenditure_on_house_rent_l142_142616

variable (X : ℝ) -- Let X be Bhanu's total income in rupees

-- Condition 1: Bhanu spends 300 rupees on petrol, which is 30% of his income
def condition_on_petrol : Prop := 0.30 * X = 300

-- Definition of remaining income
def remaining_income : ℝ := X - 300

-- Definition of house rent expenditure: 10% of remaining income
def house_rent : ℝ := 0.10 * remaining_income X

-- Theorem: If the condition on petrol holds, then the house rent expenditure is 70 rupees
theorem expenditure_on_house_rent (h : condition_on_petrol X) : house_rent X = 70 :=
  sorry

end expenditure_on_house_rent_l142_142616


namespace not_perfect_power_probability_l142_142882

def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ x^y = n

theorem not_perfect_power_probability :
  let total := 200
  let count_perfect_powers := 19
  let count_non_perfect_powers := total - count_perfect_powers
  (count_non_perfect_powers : ℚ) / total = 181 / 200 :=
by
  sorry

end not_perfect_power_probability_l142_142882


namespace find_y_l142_142502

-- Definitions based on conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

-- Lean statement capturing the problem
theorem find_y
  (h1 : inversely_proportional x y)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (h4 : x = -12) :
  y = -56.25 :=
sorry  -- Proof omitted

end find_y_l142_142502


namespace number_divisor_property_l142_142128

theorem number_divisor_property (s : Set ℕ) (h_s : s ⊆ Finset.range 2015) (h_size : s.card = 1008) :
  ∃ a b ∈ s, a ≠ b ∧ a ∣ b := 
by
  sorry

end number_divisor_property_l142_142128


namespace machine_purchase_price_l142_142498

theorem machine_purchase_price (P : ℝ) 
  (h1 : ∃ v1982, v1982 = P) 
  (h2 : ∀ t : ℕ, 1 / 10 * P → v(t + 1) = v(t) - (1 / 10 * P)) 
  (h3 : v(2) = 6400) : 
  P = 8000 := 
sorry

end machine_purchase_price_l142_142498


namespace quadratic_equation_roots_l142_142079

noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def geometric_mean (a b : ℝ) : ℝ := real.sqrt (a * b)
noncomputable def quadratic_equation (a b : ℝ) : polynomial ℝ := polynomial.X^2 - polynomial.C (a + b) * polynomial.X + polynomial.C (a * b)

theorem quadratic_equation_roots (a b : ℝ) (h_am : arithmetic_mean a b = 10) (h_gm : geometric_mean a b = 10) :
  quadratic_equation a b = polynomial.X^2 - 20 * polynomial.X + 100 :=
by
  sorry

end quadratic_equation_roots_l142_142079


namespace fraction_baked_second_day_l142_142045

def total_cakes : ℕ := 60
def half_total_cakes : ℕ := total_cakes / 2
def remaining_cakes_after_first_half : ℕ := total_cakes - half_total_cakes
def cakes_baked_first_day : ℕ := remaining_cakes_after_first_half / 2
def remaining_cakes_after_first_day : ℕ := remaining_cakes_after_first_half - cakes_baked_first_day
def final_remaining_cakes : ℕ := 10
def cakes_baked_second_day : ℕ := remaining_cakes_after_first_day - final_remaining_cakes
def fraction_second_day := cakes_baked_second_day.to_rat / remaining_cakes_after_first_day.to_rat

theorem fraction_baked_second_day : fraction_second_day = 1/3 := by
    sorry

end fraction_baked_second_day_l142_142045


namespace silvia_escalator_time_l142_142424

noncomputable def total_time_standing (v s : ℝ) : ℝ := 
  let d := 80 * v
  d / s

theorem silvia_escalator_time (v s t : ℝ) (h1 : 80 * v = 28 * (v + s)) (h2 : t = total_time_standing v s) : 
  t = 43 := by
  sorry

end silvia_escalator_time_l142_142424


namespace simplify_expression_l142_142857

theorem simplify_expression (x : ℝ) : (2 * x)^3 + (3 * x) * (x^2) = 11 * x^3 := 
  sorry

end simplify_expression_l142_142857


namespace unique_positive_real_solution_l142_142661

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l142_142661


namespace triangle_inscribed_circles_rs_l142_142811

theorem triangle_inscribed_circles_rs 
  (AB AC BC : ℕ)
  (h_AB : AB = 2023)
  (h_AC : AC = 2022)
  (h_BC : BC = 2021)
  (CH : ℕ)
  (R S : ℕ)
  (inscribed_R : ∃ C:ℕ, C = AC)
  (inscribed_S : ∃ B:ℕ, B = BC):
  let RS := |2021 / 2023| in
  (2021.gcd 2023 = 1) → (2021 + 2023 = 4044) :=
by
  intros 
  sorry

end triangle_inscribed_circles_rs_l142_142811


namespace max_value_of_f_l142_142499

noncomputable def f (x : Real) := 2 * (Real.sin x) ^ 2 - (Real.tan x) ^ 2

theorem max_value_of_f : 
  ∃ (x : Real), f x = 3 - 2 * Real.sqrt 2 := 
sorry

end max_value_of_f_l142_142499


namespace degree_of_expression_l142_142147

-- Definitions of the polynomials involved
def poly1 := 2 * X^4 + 3 * X^3 + X - 14
def poly2 := 3 * X^10 - 9 * X^7 + 9 * X^4 + 30
def poly3 := X^2 + 5

-- Expression we are interested in
def expression := poly1 * poly2 - poly3 ^ 7

-- Goal: prove the degree of the resulting polynomial is 14
theorem degree_of_expression : degree expression = 14 := sorry

end degree_of_expression_l142_142147


namespace balloons_remaining_l142_142013

-- Define the initial conditions
def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2

-- State the theorem
theorem balloons_remaining : initial_balloons - lost_balloons = 7 := by
  -- Add the solution proof steps here
  sorry

end balloons_remaining_l142_142013


namespace least_positive_angle_l142_142262

theorem least_positive_angle (θ : ℝ) :
  ∃ θ = 70 ∧ cos 15 = sin 35 + sin θ :=
by
  sorry

end least_positive_angle_l142_142262


namespace can_partition_into_three_square_corners_l142_142800

-- Define a three-square corner
structure ThreeSquareCorner (n : ℕ) :=
(i1 j1 i2 j2 i3 j3 : ℕ)
(h1 : i1 < n ∧ j1 < n)
(h2 : i2 < n ∧ j2 < n)
(h3 : i3 < n ∧ j3 < n)
(adjacent : (abs (i1 - i2) + abs (j1 - j2) = 1 ∧ abs (i1 - i3) + abs (j1 - j3) = 1) 
             ∨ (abs (i2 - i3) + abs (j2 - j3) = 1 ∧ abs (i2 - i1) + abs (j2 - j1) = 1)
             ∨ (abs (i3 - i1) + abs (j3 - j1) = 1 ∧ abs (i3 - i2) + abs (j3 - j2) = 1))

-- Define the main theorem
theorem can_partition_into_three_square_corners (n : ℕ) (grid : fin n × fin n → bool) :
  (∀ (x : fin n × fin n), grid x = tt) → ∃ (corners : list (ThreeSquareCorner n)), 
    (∀ (x : fin n × fin n), ∃ (c : ThreeSquareCorner n), 
      (c.i1, c.j1) = x ∨ (c.i2, c.j2) = x ∨ (c.i3, c.j3) = x) :=
by
  sorry

end can_partition_into_three_square_corners_l142_142800


namespace distance_to_focus_2_l142_142939

-- Definition of the ellipse and the given distance to one focus
def ellipse (P : ℝ × ℝ) : Prop := (P.1^2)/25 + (P.2^2)/16 = 1
def distance_to_focus_1 (P : ℝ × ℝ) : Prop := dist P (5, 0) = 3

-- Proof problem statement
theorem distance_to_focus_2 (P : ℝ × ℝ) (h₁ : ellipse P) (h₂ : distance_to_focus_1 P) :
  dist P (-5, 0) = 7 :=
sorry

end distance_to_focus_2_l142_142939


namespace find_valid_a_l142_142991

noncomputable def equation_satisfies (a x : ℤ) : Prop :=
  (1 + 1 / x) * ∏ k in finset.range (a + 1), (1 + 1 / (x + k)) = a - x

theorem find_valid_a : ∀ (a : ℤ), (a > 0) →
  (∃ (x : ℤ), equation_satisfies a x) ↔ a = 7 :=
by sorry

end find_valid_a_l142_142991


namespace divisor_exists_l142_142125

theorem divisor_exists : ∀ (s : Finset ℕ),
  (∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) → s.card = 1008 →
  ∃ a b ∈ s, a ∣ b ∧ a ≠ b :=
by
  sorry

end divisor_exists_l142_142125


namespace ratio_black_white_l142_142601

-- Conditions from the problem
def original_black_tiles : ℕ := 10
def original_white_tiles : ℕ := 26
def original_total_tiles : ℕ := original_black_tiles + original_white_tiles
def original_side_length : ℕ := Nat.sqrt original_total_tiles

def border_width : ℕ := 2
def extended_side_length : ℕ := original_side_length + 2 * border_width
def extended_total_tiles : ℕ := extended_side_length * extended_side_length

def new_black_tiles : ℕ := extended_total_tiles - original_total_tiles
def total_black_tiles : ℕ := original_black_tiles + new_black_tiles
def total_white_tiles : ℕ := original_white_tiles

-- Prove the ratio of black tiles to white tiles in the extended pattern
theorem ratio_black_white : total_black_tiles.toRat / total_white_tiles.toRat = 37 / 13 := by
  sorry

end ratio_black_white_l142_142601


namespace place_numbers_l142_142063

/-- 
Define a 3x3 grid where each cell contains an integer, and two cells are adjacent
if they are next to each other horizontally or vertically. 
-/
structure Grid3x3 (α : Type) :=
(cells : Fin 3 → Fin 3 → α)

/-- Distinct positive integers not greater than 25. -/
def valid_numbers : Set ℤ := {x | 1 ≤ x ∧ x ≤ 25}

/-- A grid number placement is valid if all integers are distinct, within the given 
range, and adjacent cells satisfy the divisor condition. -/
def valid_placement (grid : Grid3x3 ℤ) : Prop :=
(∀ i j, (grid.cells i j) ∈ valid_numbers) ∧
(∀ i j k l, (i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ (j = l ∧ (i = k + 1 ∨ i = k - 1)) →
  let x := grid.cells i j in
  let y := grid.cells k l in
  x ≠ y ∧ (x ∣ y ∨ y ∣ x))

/-- It is possible to fill a 3x3 grid with distinct positive integers not 
greater than 25 such that in any pair of cells adjacent by side, one number 
divides the other. -/
theorem place_numbers : ∃ (grid : Grid3x3 ℤ), valid_placement grid :=
sorry

end place_numbers_l142_142063


namespace vertex_coordinates_range_of_y_quadratic_expression_l142_142293

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions for part (1)
def b1 := 4
def c1 := 3

-- First part of the problem
theorem vertex_coordinates :
    ∃ (h k : ℝ), (∀ x : ℝ, quadratic_function (-1) b1 c1 x = - (x - h)^2 + k) ∧ h = 2 ∧ k = 7 :=
begin
    sorry
end

-- Define the interval for the second part
def interval (x : ℝ) := -1 ≤ x ∧ x ≤ 3

-- Second part of the problem
theorem range_of_y :
    ∃ (min max : ℝ), (∀ x : ℝ, interval x → -2 ≤ quadratic_function (-1) b1 c1 x ∧ quadratic_function (-1) b1 c1 x ≤ 7) := 
begin
    sorry
end

-- Given conditions for part (2)
def ymax_left := 2
def ymax_right := 3

-- Third part of the problem
theorem quadratic_expression :
    ∃ b c, (∀ x ≤ 0, quadratic_function (-1) b c x ≤ ymax_left) ∧ 
           (∀ x > 0, quadratic_function (-1) b c x ≤ ymax_right)  ∧
           b = 2 ∧ c = 2 :=
begin
    sorry
end

end vertex_coordinates_range_of_y_quadratic_expression_l142_142293


namespace average_salary_degrees_is_correct_l142_142948

-- Define the percentages for each category for the three years 
def percentages_year1 := (20, 9, 5, 4, 2 : ℝ)
def percentages_year2 := (25, 12, 6, 3, 4 : ℝ)
def percentages_year3 := (18, 10, 8, 5, 3 : ℝ)

-- Calculate the percentage of budget allocated to salaries each year
def salaries_percentage (percentages : ℝ × ℝ × ℝ × ℝ × ℝ) : ℝ :=
  100 - (percentages.1 + percentages.2 + percentages.3 + percentages.4 + percentages.5)

-- Calculate the average salary percentage
def average_salary_percentage : ℝ :=
  (salaries_percentage percentages_year1 +
   salaries_percentage percentages_year2 +
   salaries_percentage percentages_year3) / 3

-- Convert the salary percentage to degrees of the circle
def salary_degrees : ℝ :=
  (average_salary_percentage / 100) * 360

theorem average_salary_degrees_is_correct :
  salary_degrees ≈ 199.19 :=
by 
  -- Placeholder for actual proof
  sorry

end average_salary_degrees_is_correct_l142_142948


namespace tan_theta_sqrt_two_l142_142346

theorem tan_theta_sqrt_two (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < 2 * Real.pi) :
  let a := (1, Real.sqrt 2)
  let b := (Real.cos θ, Real.sin θ)
  let dot_product := a.1 * b.1 + a.2 * b.2
  let a_magnitude := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
    in dot_product = a_magnitude → Real.tan θ = Real.sqrt 2 :=
by
  intros a b dot_product a_magnitude h_dot_product
  sorry

end tan_theta_sqrt_two_l142_142346


namespace sum_reciprocal_l142_142790

-- Definitions and conditions based on the problem statement
def sequence (n : ℕ) : ℕ := 2 * 3^(n - 1)

def sum_sequence (n : ℕ) : ℕ := 3^n - 1

-- Main theorem statement proving the required equality
theorem sum_reciprocal (n : ℕ) :
  (∑ i in Finset.range n, 1 / sequence (i + 1)) = (3 / 4) * (1 - (1 / 3^n)) :=
by
  -- Proof to be filled in here
  sorry

end sum_reciprocal_l142_142790


namespace geometric_sequence_general_term_l142_142096

theorem geometric_sequence_general_term (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, Real.logBase 2 (a (n + 1)) - Real.logBase 2 (a n) = 1)
  (h2 : a 1 = 1) :
  ∀ n : ℕ, a n = 2^(n - 1) :=
by
  sorry

end geometric_sequence_general_term_l142_142096


namespace tan_thirteen_pi_over_four_l142_142642

theorem tan_thirteen_pi_over_four : 
  let θ := (13 * Real.pi) / 4 in
  Real.tan θ = 1 := by
  let θ := (13 * Real.pi) / 4
  have h1 : θ = (9 * Real.pi) / 4 + Real.pi := by
    sorry
  have h2 : Real.tan θ = Real.tan ((9 * Real.pi) / 4 + Real.pi) := by
    sorry
  have h3 : Real.tan ((9 * Real.pi) / 4 + Real.pi) = Real.tan (Real.pi / 4) := by
    sorry
  have h4 : Real.tan (Real.pi / 4) = 1 := by
    exact Real.tan_pi_over_four
  exact Eq.trans h2 (Eq.trans h3 h4)

end tan_thirteen_pi_over_four_l142_142642


namespace unique_positive_real_solution_l142_142660

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l142_142660


namespace number_of_divisibles_by_7_l142_142354

theorem number_of_divisibles_by_7 (a b : ℕ) (h1 : a = 200) (h2 : b = 400) : 
  (nat.card {n | a ≤ 7 * n ∧ 7 * n ≤ b}) = 29 := 
by sorry

end number_of_divisibles_by_7_l142_142354


namespace find_a_l142_142320

variable (a : ℝ)

def pure_imaginary (z : ℂ) : Prop := 
  z.re = 0

theorem find_a (h : pure_imaginary (Complex.div (a + 2 * Complex.I) (1 + 2 * Complex.I))) : a = -4 :=
sorry

end find_a_l142_142320


namespace find_inclination_angle_l142_142259

variables (A : Point) (t1_2 : Line) (plane : Plane)

-- Definitions from the conditions
def perpendicular (p : Point) (pl : Plane) := sorry -- Definition of a perpendicular line from p to pl
def inclination_angle (l1 l2 : Line) := sorry -- Definition of the inclination angle between two lines

-- Given problem in Lean 4
theorem find_inclination_angle :
  ∃ θ, ∃ A ∈ t1_2, perpendicular A plane ∧ inclination_angle t1_2 (perpendicular A plane) = θ := 
sorry

end find_inclination_angle_l142_142259


namespace polar_equation_of_line_l_polar_equation_of_curve_C_distance_MN_l142_142780

section

noncomputable def line_l_parametric (t : ℝ) : ℝ × ℝ :=
(2 * real.sqrt 3 - (real.sqrt 3 / 2) * t, (1 / 2) * t)

noncomputable def curve_C_parametric (α : ℝ) : ℝ × ℝ :=
(real.sqrt 3 + real.sqrt 3 * real.cos α, real.sqrt 3 * real.sin α)

theorem polar_equation_of_line_l (ρ θ : ℝ) :
  (∃ t, (2 * real.sqrt 3 - (real.sqrt 3 / 2) * t = ρ * real.cos θ) ∧
    ((1 / 2) * t = ρ * real.sin θ)) ↔
  (ρ * real.cos θ +  real.sqrt 3 * ρ * real.sin θ = 2 * real.sqrt 3) :=
sorry

theorem polar_equation_of_curve_C (ρ θ : ℝ) :
  (∃ α, (real.sqrt 3 + real.sqrt 3 * real.cos α = ρ * real.cos θ) ∧
    (real.sqrt 3 * real.sin α = ρ * real.sin θ)) ↔
  (ρ = 2 * real.sqrt 3 * real.cos θ) :=
sorry

theorem distance_MN (θ : ℝ) (hθ : 0 < θ ∧ θ < real.pi / 2) :
  let M := (2, θ)
  let N := (3, θ)
  |(M.1, M.2) - (N.1, N.2)| = 1 :=
sorry

end

end polar_equation_of_line_l_polar_equation_of_curve_C_distance_MN_l142_142780


namespace sqrt_fraction_inequality_l142_142319

variable {a b c d : ℝ}

theorem sqrt_fraction_inequality (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : (sqrt (a / d) > sqrt (b / c)) :=
by sorry

end sqrt_fraction_inequality_l142_142319


namespace sample_data_max_value_l142_142171

theorem sample_data_max_value 
  (a b c d e : ℕ) 
  (h_abcd : a < b < c < d < e) 
  (h_mean : (a + b + c + d + e) = 25)
  (h_variance : ((a - 5) ^ 2 + (b - 5) ^ 2 + (c - 5) ^ 2 + (d - 5) ^ 2 + (e - 5) ^ 2) = 20) 
  : e = 8 := 
sorry

end sample_data_max_value_l142_142171


namespace minyoung_math_score_l142_142050

-- Define the conditions as hypotheses
variables (K E M : ℝ)
hypothesis h1 : (K + E) / 2 = 89
hypothesis h2 : (K + E + M) / 3 = 91

-- Define the theorem to prove M = 95
theorem minyoung_math_score : M = 95 :=
  by
  -- Place proof here
  sorry

end minyoung_math_score_l142_142050


namespace divisor_count_1386_l142_142273

theorem divisor_count_1386 : 
  let n := 1386 in
  let exps := [(2,1), (3,2), (7,1), (11,1)] in
  (n = (2^1) * (3^2) * (7^1) * (11^1)) →
  (exps.foldl (λ acc (p, e) => acc * (e + 1)) 1) = 24 :=
by
  intros n exps h
  rw h
  sorry

end divisor_count_1386_l142_142273


namespace trader_profit_percent_equal_eight_l142_142197

-- Defining the initial conditions
def original_price (P : ℝ) := P
def purchased_price (P : ℝ) := 0.60 * original_price P
def selling_price (P : ℝ) := 1.80 * purchased_price P

-- Statement to be proved
theorem trader_profit_percent_equal_eight (P : ℝ) (h : P > 0) :
  ((selling_price P - original_price P) / original_price P) * 100 = 8 :=
by
  sorry

end trader_profit_percent_equal_eight_l142_142197


namespace max_value_of_expression_l142_142959

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 = z^2) :
  ∃ t, t = (3 * Real.sqrt 2) / 2 ∧ ∀ u, u = (x + 2 * y) / z → u ≤ t := by
  sorry

end max_value_of_expression_l142_142959


namespace inequality_k_distance_comparison_l142_142936

theorem inequality_k (k : ℝ) (x : ℝ) : 
  -3 < k ∧ k ≤ 0 → 2 * k * x^2 + k * x - 3/8 < 0 := sorry

theorem distance_comparison (a b : ℝ) (hab : a ≠ b) : 
  (abs ((a^2 + b^2) / 2 - (a + b)^2 / 4) > abs (a * b - (a + b)^2 / 4)) := sorry

end inequality_k_distance_comparison_l142_142936


namespace find_perpendicular_line_equation_l142_142693

noncomputable def line_equation_perpendicular_through_point (A B C x₀ y₀ : ℝ) (ℓ : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), (∀ x y, ℓ x y ↔ y = k*x + (y₀ - k*x₀)) ∧
             (A * x + B * y + C = 0 → ∀ x y, (x₀ ≤ x → y = (- A / B) * x))

theorem find_perpendicular_line_equation (A B C x₀ y₀ : ℝ) (ℓ : ℝ → ℝ → Prop) :
  line_equation_perpendicular_through_point A B C x₀ y₀
    (λ x y, B * (x - x₀) - A * (y - y₀) = 0) :=
sorry

end find_perpendicular_line_equation_l142_142693


namespace yakob_can_place_100_red_stones_l142_142543

theorem yakob_can_place_100_red_stones : 
  ∃ (grid : ℕ × ℕ → option Prop) (yakob_can_place : nat → Prop),
  (∀ (x y : ℕ × ℕ), grid x = some true → grid y = some true → x ≠ y → (dist x y ≠ sqrt 5)) → 
  (yakob_can_place 100) :=
by
  -- grid allows stones on a 20x20 grid only
  let grid : ℕ × ℕ → option Prop := λ (x y : ℕ × ℕ), if (x < 20 ∧ y < 20) then some true else none

  -- Check if Yakob can place K stones (K = 100 in our conclusion)
  let yakob_can_place : nat → Prop := λ K, ∀ (moves : list (ℕ × ℕ)), moves.length = K → ∀ (move : ℕ × ℕ), move ∈ moves → grid move = some true

  sorry

end yakob_can_place_100_red_stones_l142_142543


namespace sum_converges_to_7_l142_142987

theorem sum_converges_to_7 : 
  (∑ k in Finset.range(1,∞), 7^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 7 :=
by
  sorry

end sum_converges_to_7_l142_142987


namespace flash_catches_ace_l142_142605

variables (x y : ℝ)
variable (v : ℝ := 1)

-- Conditions
axiom h1 : x > 1
axiom h2 : y > 0

theorem flash_catches_ace : 
  let Flash_speed := 2 * x * v,
      d := (4 * x * y) / (2 * x - 1) in
  d = (4 * x * y) / (2 * x - 1) :=
by
  sorry

end flash_catches_ace_l142_142605


namespace find_cos_F_l142_142004

-- Definitions of the given conditions
def sin_D : ℝ := 4 / 5
def cos_E : ℝ := 3 / 5

-- Statement of the proof problem
theorem find_cos_F (h1 : sin D = sin_D) (h2 : cos E = cos_E) : cos F = 7 / 25 := sorry

end find_cos_F_l142_142004


namespace distance_and_y_intercept_l142_142185

theorem distance_and_y_intercept (x1 y1 : ℝ)
  (h1 : (x1, y1) = (12, 20)) :
  let line1 := λ x, 2 * x - 24 + y1,
      line2 := λ x, -4 * x + 48 + y1,
      x_intercept1 := 2,
      x_intercept2 := 17,
      distance := |x_intercept2 - x_intercept1|,
      y_intercept1 := line1 0
  in
  distance = 15 ∧ y_intercept1 = -4 :=
by {
  sorry
}

end distance_and_y_intercept_l142_142185


namespace find_area_of_BEC_l142_142003

open_locale classical -- to suppress warning about noncomputable

noncomputable
def area_of_BEC (AD AB DC DE : ℝ) (h1 : AD = AB) (h2 : AD = 5) (h3 : DC = 10) (h4 : DE = 6) (h5 : BE_parallel_AD : True) : ℝ :=
  let EC := DC - DE in
  let BE := AD in
  1 / 2 * BE * EC

theorem find_area_of_BEC
  (AD AB DC DE : ℝ) 
  (h1 : AD = AB) 
  (h2 : AD = 5) 
  (h3 : DC = 10) 
  (h4 : DE = 6) 
  (h5 : BE_parallel_AD : True) :
  area_of_BEC AD AB DC DE h1 h2 h3 h4 h5 = 10 :=
by sorry

end find_area_of_BEC_l142_142003


namespace cricket_throwers_l142_142467

theorem cricket_throwers (T L R : ℕ) 
  (h1 : T + L + R = 55)
  (h2 : T + R = 49) 
  (h3 : L = (1/3) * (L + R))
  (h4 : R = (2/3) * (L + R)) :
  T = 37 :=
by sorry

end cricket_throwers_l142_142467


namespace Leah_coins_value_in_cents_l142_142437

theorem Leah_coins_value_in_cents (p n : ℕ) (h₁ : p + n = 15) (h₂ : p = n + 2) : p + 5 * n = 44 :=
by
  sorry

end Leah_coins_value_in_cents_l142_142437


namespace sunzi_system_l142_142414

variable (x y : ℝ)

theorem sunzi_system :
  (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  sorry

end sunzi_system_l142_142414


namespace range_for_a_l142_142761

theorem range_for_a (a : ℝ) : 
  (∃ x : ℝ, (1 - x < -1) ∧ (x - 1 > a) ∧ (x > 2)) ↔ (a ≤ 1) :=
by
  split
  {
    intro h
    cases h with x hx
    sorry
  }
  {
    intro ha
    use 3  -- arbitrary value greater than 2
    sorry
  }

end range_for_a_l142_142761


namespace reservoir_dimensions_l142_142933

noncomputable def shorter_side : ℝ := 3
noncomputable def longer_side : ℝ := 4
noncomputable def diagonal_distance : ℝ := Real.sqrt (shorter_side ^ 2 + (longer_side) ^ 2)
noncomputable def walking_distance : ℝ := shorter_side + shorter_side + 1
noncomputable def speed : ℝ := 4
noncomputable def time_difference : ℝ := 0.5

theorem reservoir_dimensions :
  2 * shorter_side + 1 = Real.sqrt (2 * shorter_side ^ 2 + 2 * shorter_side + 1) + 2 →
  shorter_side = 3 ∧ longer_side = 4 :=
by
  intro h
  rw [h]
  sorry

end reservoir_dimensions_l142_142933


namespace parabola_vertex_sum_l142_142256

theorem parabola_vertex_sum (p q r : ℝ) 
  (h1 : ∃ (a b c : ℝ), ∀ (x : ℝ), a * x ^ 2 + b * x + c = y)
  (h2 : ∃ (vertex_x vertex_y : ℝ), vertex_x = 3 ∧ vertex_y = -1)
  (h3 : ∀ (x : ℝ), y = p * x ^ 2 + q * x + r)
  (h4 : y = p * (0 - 3) ^ 2 + r - 1)
  (h5 : y = 8)
  : p + q + r = 3 := 
by
  sorry

end parabola_vertex_sum_l142_142256


namespace part1_vertex_part1_range_y_part2_expression_l142_142296

-- Part 1: Quadratic function with b=4 and c=3
def part1_questions := let b := 4; let c := 3 in
  (⟨vertex_coordinates : (2, 7)⟩, ⟨range_y : -2 ≤ fun x => -x^2 + b * x + c x ≤ 7 for -1 ≤ x ≤ 3⟩)

-- Part 2: Quadratic function with specific conditions on max values
def part2_conditions := let y := fun x => -x^2 + b * x + c in
  (⟨max_y_on_neg := 2 if x ≤ 0⟩, ⟨max_y_on_pos := 3 if x > 0⟩)

theorem part1_vertex : (part1_questions.vertex_coordinates).fst = ()


theorem part1_range_y : ∀ x, -1 ≤ x ≤ 3 → -2 ≤ -x^2 + 4 * x + 3 ≤ 7 :=
by sorry

theorem part2_expression : ∀ x, b = 2 ∧ c = 2 →

  ∃ (b = 2) ∧ (x ≤ 0 → y x = 2) ∧ (x > 0 → y x = 3)
:=
by sorry

end part1_vertex_part1_range_y_part2_expression_l142_142296


namespace min_groups_l142_142669

open Set

structure TwinSiblings (α : Type) := 
  (pairs : Set (Set α)) (members : ∀ s ∈ pairs, s.card = 2)

structure GroupActivities (α : Type) :=
  (groups : Set (Set α))
  (no_twins_same_group : ∀ g ∈ groups, ∀ pair ∈ TwinSiblings.pairs α, pair ∩ g ≠ pair)
  (each_non_twin_pair_once : ∀ g1 g2 ∈ groups, g1 ≠ g2 → ∀ x y, x ≠ y → x ∈ g1 ∩ g2 → y ∈ g1 ∩ g2 → False)
  (one_person_two_groups : ∃ x, (groups.filter (λ g, x ∈ g)).card = 2)

def problem_instance (α : Type) [Fintype α] :=
  (TwinSiblings α) × (GroupActivities α)

theorem min_groups {α : Type} [Fintype α] : 
  ∃ k, k = 14 → 
  ∃ inst : problem_instance α, 
  inst.2.groups.card = k := 
sorry

end min_groups_l142_142669


namespace hockey_season_length_l142_142901

theorem hockey_season_length (total_games_per_month : ℕ) (total_games_season : ℕ) 
  (h1 : total_games_per_month = 13) (h2 : total_games_season = 182) : 
  total_games_season / total_games_per_month = 14 := 
by 
  sorry

end hockey_season_length_l142_142901


namespace bisectors_form_inscribed_quadrilateral_l142_142842

noncomputable def angle_sum_opposite_bisectors {α β γ δ : ℝ} (a_bisector b_bisector c_bisector d_bisector : ℝ)
  (cond : α + β + γ + δ = 360) : Prop :=
  (a_bisector + b_bisector + c_bisector + d_bisector) = 180

theorem bisectors_form_inscribed_quadrilateral
  {α β γ δ : ℝ} (convex_quad : α + β + γ + δ = 360) :
  ∃ a_bisector b_bisector c_bisector d_bisector : ℝ,
  angle_sum_opposite_bisectors a_bisector b_bisector c_bisector d_bisector convex_quad := 
sorry

end bisectors_form_inscribed_quadrilateral_l142_142842


namespace max_int_value_of_a_real_roots_l142_142681

-- Definitions and theorem statement based on the above conditions
theorem max_int_value_of_a_real_roots (a : ℤ) :
  (∃ x : ℝ, (a-1) * x^2 - 2 * x + 3 = 0) ↔ a ≠ 1 ∧ a ≤ 0 := by
  sorry

end max_int_value_of_a_real_roots_l142_142681


namespace translate_sin_left_l142_142111

/-- 
  Given the function y = sin (2 * x), translating it to the left 
  by π / 3 units results in the new function
  y = sin (2 * x + 2 * π / 3). 
-/
theorem translate_sin_left (x : ℝ) : 
  (λ x, Real.sin (2 * x)) (x + π / 3) = Real.sin (2 * x + 2 * π / 3) :=
  sorry

end translate_sin_left_l142_142111


namespace find_angle_BAD_l142_142783

open set

variables {A B C D : Type} [linear_ordered_field A] 
          [linear_ordered_field B] [linear_ordered_field C] [linear_ordered_field D]

-- Conditions:
def convex_quadrilateral (A B C D : Type) := True -- Placeholder for convex quadrilateral condition.

def equal_diagonals (AC BD : Type) := AC = BD

def angle_conditions (BAC ADB CAD ADC ABD BAD : A) : Prop :=
  BAC = ADB ∧ (CAD + ADC) = ABD

-- Proof problem statement:
theorem find_angle_BAD (A B C D : Type) (AC BD : Type) (BAC ADB CAD ADC ABD BAD : A)
    (h_convex : convex_quadrilateral A B C D)
    (h_equal_diagonals : equal_diagonals AC BD)
    (h_angle_conditions : angle_conditions BAC ADB CAD ADC ABD BAD):
  BAD = 60 :=
by sorry

end find_angle_BAD_l142_142783


namespace problem_l142_142165

-- Define the main problem conditions
variables {a b c : ℝ}
axiom h1 : a^2 + b^2 + c^2 = 63
axiom h2 : 2 * a + 3 * b + 6 * c = 21 * Real.sqrt 7

-- Define the goal
theorem problem :
  (a / c) ^ (a / b) = (1 / 3) ^ (2 / 3) :=
sorry

end problem_l142_142165


namespace three_digit_numbers_are_168_l142_142515

noncomputable def three_digit_numbers_count : Nat :=
  let cards : List (Set Nat) := [{0, 1}, {2, 3}, {4, 5}, {6, 7}]
  let possible_hundreds := cards.bind id -- flatten the sets of card sides
  let possible_tens := possible_hundreds.diff {0} -- ensuring no zero at the hundreds place
  let possible_ones := possible_tens.diff {1}
  let count_ways (n : Nat) := possible_hundreds.filter (λ x => x ≠ n)
  let total_count := possible_hundreds.sum (λ h => possible_tens.sum (λ t => possible_ones.filter (λ o => o ≠ h ∧ o ≠ t).length))
  total_count

theorem three_digit_numbers_are_168 :
  three_digit_numbers_count = 168 :=
sorry

end three_digit_numbers_are_168_l142_142515


namespace donuts_selection_l142_142061

def number_of_selections (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem donuts_selection : number_of_selections 6 4 = 84 := by
  sorry

end donuts_selection_l142_142061


namespace pet_store_cages_l142_142162

-- Definitions and conditions
def initial_puppies : ℕ := 56
def sold_puppies : ℕ := 24
def puppies_per_cage : ℕ := 4
def remaining_puppies : ℕ := initial_puppies - sold_puppies
def cages_used : ℕ := remaining_puppies / puppies_per_cage

-- Theorem statement
theorem pet_store_cages : cages_used = 8 := by sorry

end pet_store_cages_l142_142162


namespace number_of_wickets_last_match_l142_142590

noncomputable def bowling_average : ℝ := 12.4
noncomputable def runs_taken_last_match : ℝ := 26
noncomputable def wickets_before_last_match : ℕ := 175
noncomputable def decrease_in_average : ℝ := 0.4
noncomputable def new_average : ℝ := bowling_average - decrease_in_average

theorem number_of_wickets_last_match (w : ℝ) :
  (175 + w) > 0 → 
  ((wickets_before_last_match * bowling_average + runs_taken_last_match) / (wickets_before_last_match + w) = new_average) →
  w = 8 := 
sorry

end number_of_wickets_last_match_l142_142590


namespace system_solution_l142_142534

theorem system_solution :
  (∀ x y : ℝ, (2 * x + 3 * y = 19) ∧ (3 * x + 4 * y = 26) → x = 2 ∧ y = 5) →
  (∃ x y : ℝ, (2 * (2 * x + 4) + 3 * (y + 3) = 19) ∧ (3 * (2 * x + 4) + 4 * (y + 3) = 26) ∧ x = -1 ∧ y = 2) :=
by
  sorry

end system_solution_l142_142534


namespace sphere_radius_zero_l142_142779

/-
  Given the points A, B, C, and D in space,
  such that:
  1. A ≠ B,
  2. Line m through A is perpendicular to AB,
  3. Line n through B is perpendicular to AB,
  4. C is on line m and C ≠ A,
  5. D is on line n and D ≠ B,
  6. The length of AB is a,
  7. The length of CD is b,
  8. The angle θ between m and n is given.

  Prove that the radius of the sphere passing through points A, B, C, and D is 0.
-/

variable {A B C D : Type*}
variable [inner_product_space ℝ (Euclidean_space ℝ)]

def perp (v w : Euclidean_space ℝ) : Prop := inner_product v w = 0

theorem sphere_radius_zero
  (A B C D : Euclidean_space ℝ)
  (a b : ℝ)
  (θ : ℝ)
  (h_distinct : A ≠ B)
  (h_perp1 : perp (A - B) (C - A))
  (h_perp2 : perp (A - B) (D - B))
  (h_C_adj : perp (C - A) (D - C))
  (h_len_AB : dist A B = a)
  (h_len_CD : dist C D = b)
  (h_angle : θ = real.arccos (inner_product (C - A) (D - B) / (∥C - A∥ * ∥D - B∥))) :
  euclidean_dist_sphere (A, B, C, D) = 0 := 
  sorry

end sphere_radius_zero_l142_142779


namespace ratio_of_part_to_whole_l142_142060

theorem ratio_of_part_to_whole (N : ℝ) :
  (2/15) * N = 14 ∧ 0.40 * N = 168 → (14 / ((1/3) * (2/5) * N)) = 1 :=
by
  -- We assume the conditions given in the problem and need to prove the ratio
  intro h
  obtain ⟨h1, h2⟩ := h
  -- Establish equality through calculations
  sorry

end ratio_of_part_to_whole_l142_142060


namespace smallest_solution_l142_142538

theorem smallest_solution (x : ℝ) (h : x^2 + 10 * x - 24 = 0) : x = -12 :=
sorry

end smallest_solution_l142_142538


namespace hancho_milk_consumption_l142_142108

theorem hancho_milk_consumption :
  ∀ (initial_yeseul_consumption gayoung_bonus liters_left initial_milk consumption_yeseul consumption_gayoung consumption_total), 
  initial_yeseul_consumption = 0.1 →
  gayoung_bonus = 0.2 →
  liters_left = 0.3 →
  initial_milk = 1 →
  consumption_yeseul = initial_yeseul_consumption →
  consumption_gayoung = initial_yeseul_consumption + gayoung_bonus →
  consumption_total = consumption_yeseul + consumption_gayoung →
  (initial_milk - (consumption_total + liters_left)) = 0.3 :=
by sorry

end hancho_milk_consumption_l142_142108


namespace diagonal_divided_into_three_equal_parts_l142_142846

-- We define the points in 3D space
variables (A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ)

-- Assumptions to ensure that we are considering a parallelepiped
-- A parallelepiped condition might be complex to curb completely in Lean directly.
-- For simplicity, I'll assert diagonals and their intersections with medians.

-- Defining the medians intersection points
def median_triangle (P Q R : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (xP, yP, zP) := P
  let (xQ, yQ, zQ) := Q
  let (xR, yR, zR) := R
  ((xP + xQ + xR) / 3, (yP + yQ + yR) / 3, (zP + zQ + zR) / 3)

def intersection_medians_A1BD (A1 B D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  median_triangle A1 B D

def intersection_medians_CB1D1 (C B1 D1 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  median_triangle C B1 D1

-- Main theorem
theorem diagonal_divided_into_three_equal_parts
  (H_parallel: ∃ (V W: ℝ × ℝ × ℝ), 
    intersection_medians_A1BD A1 B D = V ∧ 
    intersection_medians_CB1D1 C B1 D1 = W ∧ 
    let P := ((1/3:ℝ) • (A + C1)) in 
    let Q := ((2/3:ℝ) • (A + C1)) in 
    V = P ∧ W = Q ) : true := sorry

end diagonal_divided_into_three_equal_parts_l142_142846


namespace ke_eq_kd_l142_142151

/-- Given a triangle ABC, with BD as the angle bisector of ∠ABC.
Given point E such that ∠EAB = ∠ACB and AE = DC.
Let segment ED intersect segment AB at point K.
Prove that KE = KD.
--/
theorem ke_eq_kd (A B C D E K : Point)
  (h1 : is_angle_bisector B D A C)
  (h2 : ∠EAB = ∠ACB)
  (h3 : AE = DC)
  (h4 : K ∈ segment ED ∩ segment AB) :
  KE = KD :=
sorry

end ke_eq_kd_l142_142151


namespace erase_edge_forest_l142_142675

def basic (a b : ℤ) : Prop := Int.gcd a b = 1

def connected (p₁ p₂ : ℤ × ℤ) : Prop :=
  let (a₁, b₁) := p₁
  let (a₂, b₂) := p₂
  (2 * a₁ = 2 * a₂ ∧ (b₁ - b₂ = 2 * a₁ ∨ b₂ - b₁ = 2 * a₁)) ∨
  (2 * b₁ = 2 * b₂ ∧ (a₁ - a₂ = 2 * b₁ ∨ a₂ - a₁ = 2 * b₁))

theorem erase_edge_forest (vertices : Finset (ℤ × ℤ)) (edges : Finset ((ℤ × ℤ) × (ℤ × ℤ))) :
  (∀ (a b : ℤ), (a, b) ∈ vertices → basic a b) →
  (∀ (p₁ p₂ : ℤ × ℤ), (p₁, p₂) ∈ edges → connected p₁ p₂) →
  (∃ (erased_edges : Finset ((ℤ × ℤ) × (ℤ × ℤ))), erased_edges ⊆ edges ∧
    erase_edge_forest_property vertices (edges \ erased_edges)) ∧ 
  (∃ (trees : Finset (Finset (ℤ × ℤ))), forest_trees_property vertices (edges \ erased_edges) trees ∧ trees.card ≥ 1) :=
sorry

end erase_edge_forest_l142_142675


namespace new_person_weight_is_90_l142_142555

-- Define the weight of the replaced person
def replaced_person_weight : ℝ := 40

-- Define the increase in average weight when the new person replaces the replaced person
def increase_in_average_weight : ℝ := 10

-- Define the increase in total weight as 5 times the increase in average weight
def increase_in_total_weight (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

-- Define the weight of the new person
def new_person_weight (replaced_w : ℝ) (total_increase : ℝ) : ℝ := replaced_w + total_increase

-- Prove that the weight of the new person is 90 kg
theorem new_person_weight_is_90 :
  new_person_weight replaced_person_weight (increase_in_total_weight 5 increase_in_average_weight) = 90 := 
by 
  -- sorry will skip the proof, as required
  sorry

end new_person_weight_is_90_l142_142555


namespace tan_half_angle_product_l142_142628

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0) :
  (Real.tan (a / 2)) * (Real.tan (b / 2)) = 5 ∨ (Real.tan (a / 2)) * (Real.tan (b / 2)) = -5 :=
by 
  sorry

end tan_half_angle_product_l142_142628


namespace goalkeeper_not_at_goal_line_energy_consumed_correct_lob_shot_opportunities_correct_l142_142773

-- Movement record of the goalkeeper
def movements : List Int := [7, -3, 8, 4, -6, -8, 14, -15]

-- Question 1: The goalkeeper does not return to the goal line
theorem goalkeeper_not_at_goal_line : (movements.sum ≠ 0) :=
by
  -- Assume the proof here
  sorry

-- Question 2: The total energy consumed
noncomputable def total_energy_consumed : Float :=
let distances := movements.map Int.natAbs
distances.sum * 0.1

theorem energy_consumed_correct : total_energy_consumed = 6.5 :=
by
  -- Assume the proof here
  sorry

-- Question 3: Number of opportunities for a lob shot goal
def cumulative_distance(moves : List Int) : List Int :=
moves.scanl (+) 0 -- This generates the running total of the distances

def opportunities_for_lob_shot : List Int :=
(cumulative_distance movements).filter (λ x => abs x > 10)

theorem lob_shot_opportunities_correct : opportunities_for_lob_shot.length = 3 :=
by
  -- Assume the proof here
  sorry

end goalkeeper_not_at_goal_line_energy_consumed_correct_lob_shot_opportunities_correct_l142_142773


namespace num_divisible_by_7_200_to_400_l142_142372

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l142_142372


namespace fraction_B_compared_to_A_and_C_l142_142176

theorem fraction_B_compared_to_A_and_C
    (A B C : ℕ) 
    (h1 : A = (B + C) / 3) 
    (h2 : A = B + 35) 
    (h3 : A + B + C = 1260) : 
    (∃ x : ℚ, B = x * (A + C) ∧ x = 2 / 7) :=
by
  sorry

end fraction_B_compared_to_A_and_C_l142_142176


namespace complement_superset_l142_142026

open Set

variable (U : Type) (M N : Set U)
variables [UniversalSet U]

theorem complement_superset (U : Type) [UniversalSet U] (M N : Set U) 
  (h1 : M ∪ N = N) : U \ M ⊆ U \ N := 
  sorry

end complement_superset_l142_142026


namespace acute_angle_30_l142_142755

theorem acute_angle_30 (α : ℝ) (h : Real.cos (π / 6) * Real.sin α = Real.sqrt 3 / 4) : α = π / 6 := 
by 
  sorry

end acute_angle_30_l142_142755


namespace area_enclosed_by_equation_l142_142260

theorem area_enclosed_by_equation : 
  let equation : (ℝ × ℝ) → Prop := λ ⟨x, y⟩, x^2 + y^2 = 2 * (|x| + |y|)
  ∃ A : ℝ, (∀ (x y : ℝ), equation (x, y) → A = 2 * π + 8) :=
sorry

end area_enclosed_by_equation_l142_142260


namespace max_type_A_pieces_max_profit_l142_142172

noncomputable def type_A_cost := 80
noncomputable def type_A_sell := 120
noncomputable def type_B_cost := 60
noncomputable def type_B_sell := 90
noncomputable def total_clothes := 100
noncomputable def min_type_A := 65
noncomputable def max_cost := 7500

/-- The maximum number of type A clothing pieces that can be purchased --/
theorem max_type_A_pieces (x : ℕ) : 
  type_A_cost * x + type_B_cost * (total_clothes - x) ≤ max_cost → 
  x ≤ 75 := by 
sorry

variable (a : ℝ) (h_a : 0 < a ∧ a < 10)

/-- The optimal purchase strategy to maximize profit --/
theorem max_profit (x y : ℕ) : 
  (x + y = total_clothes) ∧ 
  (type_A_cost * x + type_B_cost * y ≤ max_cost) ∧
  (min_type_A ≤ x) ∧ 
  (x ≤ 75) → 
  (type_A_sell - type_A_cost - a) * x + (type_B_sell - type_B_cost) * y 
  ≤ (type_A_sell - type_A_cost - a) * 75 + (type_B_sell - type_B_cost) * 25 := by 
sorry

end max_type_A_pieces_max_profit_l142_142172


namespace mr_green_garden_yield_l142_142830

noncomputable def garden_yield (steps_length steps_width step_length yield_per_sqft : ℝ) : ℝ :=
  let length_ft := steps_length * step_length
  let width_ft := steps_width * step_length
  let area := length_ft * width_ft
  area * yield_per_sqft

theorem mr_green_garden_yield :
  garden_yield 18 25 2.5 0.5 = 1406.25 :=
by
  sorry

end mr_green_garden_yield_l142_142830


namespace squared_length_of_graph_m_l142_142624

noncomputable def p (x : ℝ) := -3/2 * x + 1
noncomputable def q (x : ℝ) := 3/2 * x + 1
noncomputable def r (x : ℝ) := (2 : ℝ)

noncomputable def m (x : ℝ) := min (min (p x) (q x)) (r x)

theorem squared_length_of_graph_m :
  (∑ i in [-4, (-2:3:ℝ), (2:3:ℝ), 4], real.lengthSegment (m i)) ^ 2 = 841 / 9 :=
sorry

end squared_length_of_graph_m_l142_142624


namespace last_number_l142_142551

theorem last_number (A B C D E F G : ℕ)
  (h1 : A + B + C + D = 52)
  (h2 : D + E + F + G = 60)
  (h3 : E + F + G = 55)
  (h4 : D^2 = G) : G = 25 :=
by
  sorry

end last_number_l142_142551


namespace decreasing_cubic_function_l142_142493

theorem decreasing_cubic_function (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → a ≤ 0 :=
sorry

end decreasing_cubic_function_l142_142493


namespace total_cost_l142_142978

-- Definitions based on the problem's conditions
def cost_hamburger : ℕ := 4
def cost_milkshake : ℕ := 3

def qty_hamburgers : ℕ := 7
def qty_milkshakes : ℕ := 6

-- The proof statement
theorem total_cost :
  (qty_hamburgers * cost_hamburger + qty_milkshakes * cost_milkshake) = 46 :=
by
  sorry

end total_cost_l142_142978


namespace cost_price_proof_1_profit_max_proof_2_l142_142915

-- Define the cost prices and conditions for Part 1
def cost_price_problem_1 (x : ℕ) : Prop :=
    (40 / (2 * x) - 10 / x = 10) → (x = 1) ∧ (2 * x = 2)

-- Prove Part 1
theorem cost_price_proof_1 : ∃ x : ℕ, cost_price_problem_1 x :=
by { existsi 1, unfold cost_price_problem_1, split; sorry }

-- Define the purchasing and profit conditions for Part 2
def profit_max_problem_2 := 
    ∀ (m : ℕ), 
    (1000 - 2 * m ≤ 3 * m) →
    (m ≥ 200) →
    (1000 - 2 * m ≥ m) →
    (1.5 * m + (1000 - 2 * m) - m = -0.5 * m + 1000) →
    ((∀ n, 200 ≤ n ∧ 1000 - 2 * n ≤ 3 * n → (1.5 * n + 1000 - 2 * n - n ≤ -0.5 * n + 1000)) ∧
    (m = 200 → -0.5 * m + 1000 = 900.0))

-- Prove Part 2
theorem profit_max_proof_2 : ∃ m : ℕ, profit_max_problem_2 :=
by { existsi 200, unfold profit_max_problem_2, split; sorry }

end cost_price_proof_1_profit_max_proof_2_l142_142915


namespace andy_tomatoes_l142_142973

theorem andy_tomatoes (P : ℕ) (h1 : ∀ P, 7 * P / 3 = 42) : P = 18 := by
  sorry

end andy_tomatoes_l142_142973


namespace solution_set_of_inequality_l142_142087

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_inequality 
  (hf_even : ∀ x : ℝ, f x = f (|x|))
  (hf_increasing : ∀ x y : ℝ, x < y → x < 0 → y < 0 → f x < f y)
  (hf_value : f 3 = 1) :
  {x : ℝ | f (x - 1) < 1} = {x : ℝ | x > 4 ∨ x < -2} := 
sorry

end solution_set_of_inequality_l142_142087


namespace relationship_inequalities_l142_142252

theorem relationship_inequalities : 
  ∀ (x : ℝ), ((x - 2) * (x - 1) > 0) ↔ (x - 2 > 0) :=
by
  intro x
  split
  . intro h₁
    -- From the given solution, we know $(x - 2) * (x - 1) > 0$ implies either $x < 1$ or $x > 2$
    sorry
  . intro h₂
    -- From the given solution, we know $(x - 2 > 0$, implies $x > 2$
    -- This indeed implies $(x - 2) * (x - 1) > 0$ since when $x > 2$, both factors are positive
    sorry

end relationship_inequalities_l142_142252


namespace coordinates_P1_conjugate_z1_max_value_z2_minus_z1_l142_142330

-- Define the complex numbers z1 and the condition for z2
def z1 : ℂ := 2 - 2 * complex.I
def z2_condition : set ℂ := {z | abs (z - complex.I) = 1}

-- Prove the necessary statements
theorem coordinates_P1 : (z1.re, z1.im) = (2, -2) :=
by sorry

theorem conjugate_z1 : conj z1 = 2 + 2 * complex.I :=
by sorry

theorem max_value_z2_minus_z1 {z2 : ℂ} (hz2 : z2 ∈ z2_condition) : 
  ∃ max_val, max_val = real.sqrt 13 + 1 ∧ 
  ∀ w ∈ z2_condition, abs (w - z1) ≤ max_val :=
by sorry

end coordinates_P1_conjugate_z1_max_value_z2_minus_z1_l142_142330


namespace area_of_triangle_BP_Q_is_24_l142_142191

open Real

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem area_of_triangle_BP_Q_is_24
  (A B C P H Q : ℝ × ℝ)
  (h_triangle_ABC_right : C.1 = 0 ∧ C.2 = 0 ∧ B.2 = 0 ∧ A.2 ≠ 0)
  (h_BC_diameter : distance B C = 26)
  (h_tangent_AP : distance P B = distance P C ∧ P ≠ C)
  (h_PH_perpendicular_BC : P.1 = H.1 ∧ H.2 = 0)
  (h_PH_intersects_AB_at_Q : H.1 = Q.1 ∧ Q.2 ≠ 0)
  (h_BH_CH_ratio : 4 * distance B H = 9 * distance C H)
  : triangle_area B P Q = 24 :=
sorry

end area_of_triangle_BP_Q_is_24_l142_142191


namespace question_equals_answer_l142_142418

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ :=
  (ρ * Real.cos θ)^2 - ρ * Real.sin θ

def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) := 
  ((-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))

noncomputable def intersection_points_distances (l : ℝ) : ℝ :=
  let (x1, x2) := quadratic_roots 1 (-l) (-2) in
  abs x1 + abs x2

def reciprocal_distances_sum (l : ℝ) : ℝ :=
  let (x1, x2) := quadratic_roots 1 (-l) (-2) in
  1 / abs x1 + 1 / abs x2

theorem question_equals_answer (m : ℝ) :
  reciprocal_distances_sum m = Real.sqrt (m^2 + 8) / 2 := 
sorry

end question_equals_answer_l142_142418


namespace largest_angle_ABC_l142_142422

theorem largest_angle_ABC (A B C : Type) [EuclideanGeometry A] 
  [Triangle A B C] (AC BC : ℝ) (angleBAC : ℝ) 
  (hAC : AC = 5 * Real.sqrt 2) 
  (hBC : BC = 5) 
  (hBAC : angleBAC = 30) : 
  ∃ (angleABC : ℝ), angleABC = 135 :=
by
  sorry

end largest_angle_ABC_l142_142422


namespace negation_of_all_boys_love_football_is_at_least_one_boy_does_not_love_football_l142_142516

-- Definitions of the propositions
def allBoysLoveFootball : Prop := ∀ (x : ℕ), x ∈ boys → lovesFootball x
def noBoyLovesFootball : Prop := ∀ (x : ℕ), x ∈ boys → ¬ lovesFootball x
def allBoysDoNotLoveFootball : Prop := ∀ (x : ℕ), x ∈ boys → ¬ lovesFootball x
def atLeastOneBoyDoesNotLoveFootball : Prop := ∃ (x : ℕ), x ∈ boys ∧ ¬ lovesFootball x
def allGirlsLoveFootball : Prop := ∀ (x : ℕ), x ∈ girls → lovesFootball x

-- The proof statement
theorem negation_of_all_boys_love_football_is_at_least_one_boy_does_not_love_football :
  ¬ allBoysLoveFootball = atLeastOneBoyDoesNotLoveFootball := 
sorry

end negation_of_all_boys_love_football_is_at_least_one_boy_does_not_love_football_l142_142516


namespace carnations_third_bouquet_l142_142910

theorem carnations_third_bouquet (bouquet1 bouquet2 bouquet3 : ℕ) 
  (h1 : bouquet1 = 9) (h2 : bouquet2 = 14) 
  (h3 : (bouquet1 + bouquet2 + bouquet3) / 3 = 12) : bouquet3 = 13 :=
by
  sorry

end carnations_third_bouquet_l142_142910


namespace find_m_n_l142_142338

def line1 := (2, 2, -1)
def line2 (n : ℝ) := (4, n, 3)
def line3 (m : ℝ) := (m, 6, 1)

def are_parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop := l1.1 * l2.2 = l1.2 * l2.1
def are_perpendicular (l1 l2 : ℝ × ℝ × ℝ) : Prop := l1.1 * l2.1 + l1.2 * l2.2 = 0

theorem find_m_n (m n : ℝ) (h_parallel : are_parallel line1 (line2 n)) (h_perpendicular : are_perpendicular line1 (line3 m)) :
  m + n = -2 :=
sorry

end find_m_n_l142_142338


namespace problem_A_problem_B_problem_C_problem_D_l142_142149

-- Problem A
theorem problem_A (n : ℕ) (p : ℝ) (h₁ : n * p = 30) (h₂ : n * p * (1 - p) = 20) : p ≠ 2 / 3 :=
by
  sorry

-- Problem B
theorem problem_B : let data := [91, 72, 75, 85, 64, 92, 76, 78, 86, 79] in
  let sorted_data := data.sort ≤ := [64, 72, 75, 76, 78, 79, 85, 86, 91, 92] in
  sorted_data.nth 4 = 78 :=
by
  sorry

-- Problem C
theorem problem_C (p : ℝ) (h₁ : 0 < p) (h₂ : p < 1) : ∀ (ξ : ℝ), P (ξ ∼ N(0,1)) (ξ > 1) = p → P (ξ ∼ N(0,1)) (-1 ≤ ξ ≤ 0) = (1 / 2 - p) :=
by
  sorry

-- Problem D
theorem problem_D (students_grade11 : ℕ) (students_grade12 : ℕ) (selected_total : ℕ) (selected_grade11 : ℕ)
  (h₁ : students_grade11 = 400)
  (h₂ : students_grade12 = 360)
  (h₃ : selected_total = 57)
  (h₄ : selected_grade11 = 20) : 
  ¬ (19 students should be selected from grade 13) := 
by
  sorry

end problem_A_problem_B_problem_C_problem_D_l142_142149


namespace seq_general_formula_sum_formula_l142_142698

open_locale big_operators

noncomputable def seq (n : ℕ) : ℕ := n + 2^(n - 1)

def S (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), seq i

theorem seq_general_formula (n : ℕ) : seq n = n + 2^(n - 1) :=
by {
  sorry
}

theorem sum_formula (n : ℕ) : S n = (n * (n + 1)) / 2 + 2^n - 1 :=
by {
  sorry
}

end seq_general_formula_sum_formula_l142_142698


namespace solve_for_x_l142_142859

theorem solve_for_x (x : ℝ) : (∃ x : ℝ, (cbrt (30 * x + cbrt (30 * x + 80)) = 27)) → x = 19603 / 30 :=
sorry

end solve_for_x_l142_142859


namespace find_f_5_point_5_l142_142815

-- f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

-- f satisfies the property f(x+2) = -f(x)
def periodic_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f(x)

-- f is given explicitly on [0, 1]
def f_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = x

-- All conditions combined
def conditions (f : ℝ → ℝ) : Prop :=
  odd_function f ∧ periodic_property f ∧ f_on_interval f

-- Prove the goal
theorem find_f_5_point_5 (f : ℝ → ℝ) (h : conditions f) : f 5.5 = 0.5 :=
sorry

end find_f_5_point_5_l142_142815


namespace range_of_M_l142_142683

theorem range_of_M (a θ : ℝ) (h : a ≠ 0) :
  let M := (a^2 - a * sin θ + 1) / (a^2 - a * cos θ + 1) in
  ∃ (M : ℝ), M ∈ set.Icc ((4 - real.sqrt 7) / 3) ((4 + real.sqrt 7) / 3) :=
by
  let M := (a^2 - a * sin θ + 1) / (a^2 - a * cos θ + 1)
  use M
  sorry

end range_of_M_l142_142683


namespace geometric_sequence_S_n_plus_1_l142_142303

noncomputable def a : ℕ → ℝ
noncomputable def S : ℕ → ℝ := λ n, ∑ i in Finset.range n, a i

axiom a_1 : a 1 = 1
axiom a_rec (n : ℕ) : a (n + 1) = (n + 2) / n * S n

theorem geometric_sequence (n : ℕ) : 
  ∃ r, r ≠ 0 ∧ (∀ k, k ≤ n → S k / k = r ^ k) ∧ S 1 / 1 = 1 :=
sorry

theorem S_n_plus_1 (n : ℕ) : S (n + 1) = 4 * a n :=
sorry

end geometric_sequence_S_n_plus_1_l142_142303


namespace number_of_adults_l142_142523

theorem number_of_adults
  (A C : ℕ)
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) :
  A = 350 :=
by
  sorry

end number_of_adults_l142_142523


namespace find_least_k_l142_142878

noncomputable def has_distinct_nonzero_digit_frequencies (s : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ d1 d2, d1 ≠ d2 → d1 ≠ 0 → d2 ≠ 0 →
  let ds := (list.repeat 0 ((k - (s k).digits.length) % 9)) ++ (list.range k).bind (λ n, (n+1).digits.reverse) in
  (list.count d1 ds) ≠ (list.count d2 ds)

theorem find_least_k : ∃ k, k = 2468 ∧ has_distinct_nonzero_digit_frequencies (λ n, n) k := sorry

end find_least_k_l142_142878


namespace total_texts_sent_l142_142055

def texts_sent_monday_allison : ℕ := 5
def texts_sent_monday_brittney : ℕ := 5
def texts_sent_tuesday_allison : ℕ := 15
def texts_sent_tuesday_brittney : ℕ := 15

theorem total_texts_sent : (texts_sent_monday_allison + texts_sent_monday_brittney) + 
                           (texts_sent_tuesday_allison + texts_sent_tuesday_brittney) = 40 :=
by
  sorry

end total_texts_sent_l142_142055


namespace lisa_needs_change_probability_l142_142521

theorem lisa_needs_change_probability :
  let quarters := 16
  let toy_prices := List.range' 2 10 |> List.map (fun n => n * 25) -- List of toy costs: (50,75,...,300)
  let favorite_toy_price := 275
  let factorial := Nat.factorial
  let favorable := (factorial 9) + 9 * (factorial 8)
  let total_permutations := factorial 10
  let p_no_change := (favorable.toFloat / total_permutations.toFloat) -- Convert to Float for probability calculations
  let p_change_needed := Float.round ((1.0 - p_no_change) * 100.0) / 100.0
  p_change_needed = 4.0 / 5.0 := sorry

end lisa_needs_change_probability_l142_142521


namespace harmonic_series_S4_eq_25_l142_142821

-- Given conditions
def first_terms : List ℕ := [3, 4, 6]

-- Harmonic sum S_n
noncomputable def S (n : ℕ) : ℚ :=
  if n = 1 then first_terms.head
  else if n = 2 then first_terms.head + first_terms.nth 1
  else if n = 3 then first_terms.head + first_terms.nth 1 + first_terms.nth 2
  else first_terms.head + first_terms.nth 1 + first_terms.nth 2 + 12 -- The fourth term calculated in the solution

/-- Main theorem statement -/
theorem harmonic_series_S4_eq_25 :
  S 4 = 25 :=
by
  sorry

end harmonic_series_S4_eq_25_l142_142821


namespace distance_between_A_and_B_l142_142592

-- Declaring the distance between points A and B
def d : ℝ := 271 / 3

-- Condition: The passenger train's initial speed
def passenger_speed_initial : ℝ := 30

-- Condition: The fast train's speed
def fast_train_speed : ℝ := 60

-- Condition: Distance covered by the passenger train before breaking down
def distance_covered_before_issue : ℝ := (2/3) * d

-- Condition: Speed of passenger train after the issue
def passenger_speed_after_issue : ℝ := 15

-- Condition: Distance at which fast train overtakes the passenger train after breaking down
def distance_when_overtaken : ℝ := d - 271 / 9

theorem distance_between_A_and_B : d = 271 / 3 := 
by
  sorry

end distance_between_A_and_B_l142_142592


namespace max_ben_cupcakes_l142_142525

theorem max_ben_cupcakes (total_cupcakes : ℕ) (ben_cupcakes charles_cupcakes diana_cupcakes : ℕ)
    (h1 : total_cupcakes = 30)
    (h2 : diana_cupcakes = 2 * ben_cupcakes)
    (h3 : charles_cupcakes = diana_cupcakes)
    (h4 : total_cupcakes = ben_cupcakes + charles_cupcakes + diana_cupcakes) :
    ben_cupcakes = 6 :=
by
  -- Proof steps would go here
  sorry

end max_ben_cupcakes_l142_142525


namespace part1_part2_l142_142686

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem part1 :
  ∀ x, (x ≤ -2 ∨ x ≥ 2) → (f' x ≥ 0) → strict_mono f :=
by
  sorry

theorem part2 :
  ∀ x ∈ Icc (0 : ℝ) 4, (f x = -4 / 3) → is_min_on f x (Icc 0 4) :=
by
  sorry

end part1_part2_l142_142686


namespace rich_avg_time_per_mile_l142_142475

-- Define the total time in minutes and the total distance
def total_minutes : ℕ := 517
def total_miles : ℕ := 50

-- Define a function to calculate the average time per mile
def avg_time_per_mile (total_time : ℕ) (distance : ℕ) : ℚ :=
  total_time / distance

-- Theorem statement
theorem rich_avg_time_per_mile :
  avg_time_per_mile total_minutes total_miles = 10.34 :=
by
  -- Proof steps go here
  sorry

end rich_avg_time_per_mile_l142_142475


namespace soccer_league_games_l142_142932

open Nat

theorem soccer_league_games (n : ℕ) (h : n = 10) : (nat.choose n 2) = 45 := by
  rw h
  simp [nat.choose, factorial]
  sorry

end soccer_league_games_l142_142932


namespace masha_number_l142_142461

theorem masha_number (x : ℝ) (n : ℤ) (ε : ℝ) (h1 : 0 ≤ ε) (h2 : ε < 1) (h3 : x = n + ε) (h4 : (n : ℝ) = 0.57 * x) : x = 100 / 57 :=
by
  sorry

end masha_number_l142_142461


namespace iron_conducts_deductively_l142_142558

axiom all_metals_conduct : ∀ (x : Type), x = "metal" → x = "conduct electricity"
axiom iron_is_metal : "iron" = "metal"

theorem iron_conducts_deductively :
  (all_metals_conduct "iron" iron_is_metal) = "conduct electricity" :=
by sorry

end iron_conducts_deductively_l142_142558


namespace additional_hours_to_station_C_l142_142485

-- Additional conditions definitions
variables (M D_AB D_BC D_AC : ℝ)
variables (time_AB time_BC : ℝ)
variables (S : ℝ)

-- Problem Conditions
def constant_speed (S : ℝ) : Prop := true

def time_to_reach_B (time_AB : ℝ) : Prop :=
  time_AB = 7

def distances (D_AB D_BC D_AC : ℝ) : Prop :=
  D_AB = D_BC + M ∧ D_AC = 6 * M

-- Main Goal Statement
theorem additional_hours_to_station_C
  (M D_AB D_BC D_AC time_AB S : ℝ)
  (h1 : constant_speed S)
  (h2 : time_to_reach_B time_AB)
  (h3 : distances D_AB D_BC D_AC)
  : time_BC = 5 := 
  sorry

end additional_hours_to_station_C_l142_142485


namespace calc_expr_l142_142981

theorem calc_expr : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 :=
by sorry

end calc_expr_l142_142981


namespace t_n_closed_form_t_2022_last_digit_l142_142672

noncomputable def t_n (n : ℕ) : ℕ :=
  (4^n - 3 * 3^n + 3 * 2^n - 1) / 6

theorem t_n_closed_form (n : ℕ) (hn : 0 < n) :
  t_n n = (4^n - 3 * 3^n + 3 * 2^n - 1) / 6 :=
by
  sorry

theorem t_2022_last_digit :
  (t_n 2022) % 10 = 1 :=
by
  sorry

end t_n_closed_form_t_2022_last_digit_l142_142672


namespace skew_lines_intersecting_lines_l142_142731

noncomputable def exists_intersecting_lines (p q : Line) (h_skew : skew p q) : Prop :=
  ∃ (A : Point) (B1 B2 : Point) (line1 line2 : Line),
    on_line A p ∧
    on_line B1 q ∧
    on_line B2 q ∧
    on_line A line1 ∧
    on_line B1 line1 ∧
    on_line A line2 ∧
    on_line B2 line2 ∧
    intersects line1 line2

theorem skew_lines_intersecting_lines (p q : Line) (h_skew : skew p q) : 
  exists_intersecting_lines p q h_skew :=
sorry

end skew_lines_intersecting_lines_l142_142731


namespace sum_of_10_smallest_divisible_by_5_l142_142674

def T_n (n : ℕ) : ℕ :=
  ((n - 1) * n * (n + 1) * (3 * n + 2)) / 24

theorem sum_of_10_smallest_divisible_by_5 :
  (Finset.range 50).filter (λ n, T_n n % 5 = 0) |>.sum = 235 := 
by
  sorry

end sum_of_10_smallest_divisible_by_5_l142_142674


namespace perimeter_of_square_B_l142_142189

theorem perimeter_of_square_B :
  ∀ (area_A : ℝ) (prob_not_in_B : ℝ),
  area_A = 30 →
  prob_not_in_B = 0.4666666666666667 →
  let area_B := (1 - prob_not_in_B) * area_A in
  let side_length_s := Real.sqrt area_B in
  let perimeter_B := 4 * side_length_s in
  perimeter_B = 16 :=
by
  intros area_A prob_not_in_B h_area_A h_prob_not_in_B
  let area_B := (1 - prob_not_in_B) * area_A
  let side_length_s := Real.sqrt area_B
  let perimeter_B := 4 * side_length_s
  have h1 : area_B = 16
  {
    calc
      area_B = (1 - prob_not_in_B) * area_A : by rw [h_area_A, h_prob_not_in_B]
      ... = (1 - 0.4666666666666667) * 30 : by rw [h_area_A, h_prob_not_in_B]
      ... = 0.5333333333333333 * 30 : rfl
      ... = 16 : by ring
  }
  have h2 : side_length_s = 4
  {
    calc
      side_length_s = Real.sqrt area_B : rfl
      ... = Real.sqrt 16 : by rw [h1]
      ... = 4 : by norm_num
  }
  have h3 : perimeter_B = 16
  {
    calc
      perimeter_B = 4 * side_length_s : rfl
      ... = 4 * 4 : by rw [h2]
      ... = 16 : by norm_num
  }
  exact h3

end perimeter_of_square_B_l142_142189


namespace possibilities_for_sets_l142_142017

noncomputable def sets_of_positive_integers  (A B : set ℕ) (H1 : (∀ n, n ∈ A ∪ B))
                                                (H2 : ∀ n m, n ∈ A → m ∈ A → n ≠ m → ¬ prime (n - m))
                                                (H3 : ∀ n m, n ∈ B → m ∈ B → n ≠ m → ¬ prime (n - m)) : Prop :=
  (∀ n, even n → n ∈ A) ∧ (∀ n, odd n → n ∈ B)

-- Formalizing the statement as a theorem
theorem possibilities_for_sets (A B : set ℕ)
  (H1: ∀ n, n ∈ A ∪ B)
  (H2: A ∩ B = ∅)
  (H3: ∀ n m, n ∈ ℕ → m ∈ ℕ → prime (n - m) → n ≠ m → (n ∈ A ∧ m ∈ B) ∨ (n ∈ B ∧ m ∈ A)):
  sets_of_positive_integers A B H1 sorry sorry := sorry

end possibilities_for_sets_l142_142017


namespace calculate_coeffs_l142_142983

noncomputable def quadratic_coeffs (p q : ℝ) : Prop :=
  if p = 1 then true else if p = -2 then q = -1 else false

theorem calculate_coeffs (p q : ℝ) :
    (∃ p q, (x^2 + p * x + q = 0) ∧ (x^2 - p^2 * x + p * q = 0)) →
    quadratic_coeffs p q :=
by sorry

end calculate_coeffs_l142_142983


namespace no_isosceles_triangles_l142_142468

structure Point where
  x : ℕ
  y : ℕ

structure Triangle where
  A : Point
  B : Point
  C : Point

def distance (p1 p2 : Point) : ℝ :=
  Math.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

def is_isosceles (t : Triangle) : Prop :=
  let side1 := distance t.A t.B
  let side2 := distance t.A t.C
  let side3 := distance t.B t.C
  side1 = side2 ∨ side1 = side3 ∨ side2 = side3

def T1 : Triangle := { A := { x := 2, y := 7 }, B := { x := 5, y := 7 }, C := { x := 5, y := 3 } }
def T2 : Triangle := { A := { x := 4, y := 2 }, B := { x := 7, y := 2 }, C := { x := 4, y := 6 } }
def T3 : Triangle := { A := { x := 2, y := 1 }, B := { x := 2, y := 4 }, C := { x := 7, y := 1 } }
def T4 : Triangle := { A := { x := 7, y := 5 }, B := { x := 9, y := 8 }, C := { x := 9, y := 9 } }
def T5 : Triangle := { A := { x := 8, y := 2 }, B := { x := 8, y := 5 }, C := { x := 10, y := 1 } }

def triangles : List Triangle := [T1, T2, T3, T4, T5]

theorem no_isosceles_triangles :
  (triangles.filter is_isosceles).length = 0 :=
by 
  sorry

end no_isosceles_triangles_l142_142468


namespace num_natural_numbers_divisible_by_7_l142_142380

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l142_142380


namespace chess_tournament_min_participants_l142_142767

open Real

noncomputable def minParticipants := 11

theorem chess_tournament_min_participants :
  ∃ (n : ℕ), (∀ (k : ℤ), 0.45 * n < k ∧ k < 0.5 * n → n ≥ minParticipants) :=
begin
  sorry,
end

end chess_tournament_min_participants_l142_142767


namespace smallest_n_for_R_eq_R_plus_2_l142_142022

def R (n : ℕ) : ℕ :=
  let primes := [11, 13, 17, 19]
  primes.foldl (λ acc p, acc + n % p) 0

theorem smallest_n_for_R_eq_R_plus_2 (n : ℕ) : (R n = R (n + 2)) ↔ n = 37 := by
  sorry

end smallest_n_for_R_eq_R_plus_2_l142_142022


namespace divisor_exists_l142_142124

theorem divisor_exists : ∀ (s : Finset ℕ),
  (∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) → s.card = 1008 →
  ∃ a b ∈ s, a ∣ b ∧ a ≠ b :=
by
  sorry

end divisor_exists_l142_142124


namespace youtube_dislikes_l142_142569

def initial_dislikes (likes : ℕ) : ℕ := (likes / 2) + 100

def new_dislikes (initial : ℕ) : ℕ := initial + 1000

theorem youtube_dislikes
  (likes : ℕ)
  (h_likes : likes = 3000) :
  new_dislikes (initial_dislikes likes) = 2600 :=
by
  sorry

end youtube_dislikes_l142_142569


namespace life_insurance_amount_is_20_l142_142617

variable (weekly_salary : ℕ)
variable (federal_tax_rate : ℚ)
variable (state_tax_rate : ℚ)
variable (health_insurance : ℕ)
variable (parking_fee : ℕ)
variable (final_amount : ℕ)

def life_insurance_removal_amount
  (weekly_salary : ℕ)
  (federal_tax_rate : ℚ)
  (state_tax_rate : ℚ)
  (health_insurance : ℕ)
  (parking_fee : ℕ)
  (final_amount : ℕ)
  : ℚ :=
  let federal_taxes := weekly_salary * federal_tax_rate in
  let state_taxes := weekly_salary * state_tax_rate in
  let total_deductions := federal_taxes + state_taxes + health_insurance + parking_fee in
  let remaining_salary := weekly_salary - total_deductions in
  remaining_salary - final_amount

theorem life_insurance_amount_is_20 :
  life_insurance_removal_amount 450 (1 / 3) 0.08 50 10 184 = 20 := 
  sorry

end life_insurance_amount_is_20_l142_142617


namespace possible_b_value_l142_142812

theorem possible_b_value (a b : ℤ) (h1 : a = 3^20) (h2 : a ≡ b [ZMOD 10]) : b = 2011 :=
by sorry

end possible_b_value_l142_142812


namespace distance_to_focus_parabola_l142_142708

theorem distance_to_focus_parabola (F P : ℝ × ℝ) (hF : F = (0, -1/2))
  (hP : P = (1, 2)) (C : ℝ × ℝ → Prop)
  (hC : ∀ x, C (x, 2 * x^2)) : dist P F = 17 / 8 := by
sorry

end distance_to_focus_parabola_l142_142708


namespace total_fencing_l142_142522

def playground_side_length : ℕ := 27
def garden_length : ℕ := 12
def garden_width : ℕ := 9

def perimeter_square (side : ℕ) : ℕ := 4 * side
def perimeter_rectangle (length width : ℕ) : ℕ := 2 * length + 2 * width

theorem total_fencing (side playground_side_length : ℕ) (garden_length garden_width : ℕ) :
  perimeter_square playground_side_length + perimeter_rectangle garden_length garden_width = 150 :=
by
  sorry

end total_fencing_l142_142522


namespace find_a_range_l142_142311

-- Definitions of sets A and B
def A (a x : ℝ) : Prop := a + 1 ≤ x ∧ x ≤ 2 * a - 1
def B (x : ℝ) : Prop := x ≤ 3 ∨ x > 5

-- Condition p: A ⊆ B
def p (a : ℝ) : Prop := ∀ x, A a x → B x

-- The function f(x) = x^2 - 2ax + 1
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Condition q: f(x) is increasing on (1/2, +∞)
def q (a : ℝ) : Prop := ∀ x y, 1/2 < x → x < y → f a x ≤ f a y

-- The given propositions
def prop1 (a : ℝ) : Prop := p a
def prop2 (a : ℝ) : Prop := q a

-- Given conditions
def given_conditions (a : ℝ) : Prop := ¬ (prop1 a ∧ prop2 a) ∧ (prop1 a ∨ prop2 a)

-- Proof statement: Find the range of values for 'a' according to the given conditions
theorem find_a_range (a : ℝ) :
  given_conditions a →
  (1/2 < a ∧ a ≤ 2) ∨ (4 < a) :=
sorry

end find_a_range_l142_142311


namespace extra_chairs_added_l142_142279

theorem extra_chairs_added (rows cols total_chairs extra_chairs : ℕ) 
  (h1 : rows = 7) 
  (h2 : cols = 12) 
  (h3 : total_chairs = 95) 
  (h4 : extra_chairs = total_chairs - rows * cols) : 
  extra_chairs = 11 := by 
  sorry

end extra_chairs_added_l142_142279


namespace perfect_squares_digit_4_5_6_l142_142738

theorem perfect_squares_digit_4_5_6 (n : ℕ) (hn : n^2 < 2000) : 
  (∃ k : ℕ, k = 18) :=
  sorry

end perfect_squares_digit_4_5_6_l142_142738


namespace non_swimmers_play_basketball_l142_142210

-- Define the total number of students as a nonnegative real number
variable {N : ℝ} (hN : 0 ≤ N)

-- Define the conditions
def play_basketball (N : ℝ) := 0.7 * N
def swim (N : ℝ) := 0.5 * N
def play_basketball_and_swim (N : ℝ) := 0.3 * (play_basketball N)

-- Prove the percentage of non-swimmers who play basketball is 98%
theorem non_swimmers_play_basketball :
  70%% * N = 0.7 * N →
  50%% * N = 0.5 * N →
  play_basketball_and_swim N = 0.3 * (play_basketball N) →
  (98% = (0.49 * N) / (0.5 * N) * 100) :=
by
  intro h1 h2 h3
  unfold play_basketball at h3
  have h4 : 0.49 * N = 0.7 * N - 0.3 * (0.7 * N) := by sorry
  have h5 : 0.5 * N = N - 0.5 * N := by sorry
  rw [h4, h5]
  sorry

end non_swimmers_play_basketball_l142_142210


namespace equation_of_parametrized_curve_l142_142084

theorem equation_of_parametrized_curve :
  ∀ t : ℝ, let x := 3 * t + 6 
           let y := 5 * t - 8 
           ∃ (m b : ℝ), y = m * x + b ∧ m = 5 / 3 ∧ b = -18 :=
by
  sorry

end equation_of_parametrized_curve_l142_142084


namespace complex_locus_is_circle_l142_142867

open Complex

theorem complex_locus_is_circle (z : ℂ) : |z - 3 * I| = 10 → ∃ c r, (c = 0 + 3 * I ∧ r = 10 ∧ |z - c| = r) :=
by
  intro h
  use (0 + 3 * I), 10
  exact ⟨by simp, by simp, h⟩

end complex_locus_is_circle_l142_142867


namespace intersect_single_point_l142_142760

theorem intersect_single_point (k : ℝ) :
  (∃ x : ℝ, (x^2 + k * x + 1 = 0) ∧
   ∀ x y : ℝ, (x^2 + k * x + 1 = 0 → y^2 + k * y + 1 = 0 → x = y))
  ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end intersect_single_point_l142_142760


namespace shanmukham_total_payment_l142_142856

def total_amount_to_pay (total_cost : ℝ) (rebate_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let rebate_amount := (rebate_percent / 100) * total_cost
  let amount_after_rebate := total_cost - rebate_amount
  let sales_tax := (tax_percent / 100) * amount_after_rebate
  amount_after_rebate + sales_tax

theorem shanmukham_total_payment :
  total_amount_to_pay 6650 6 10 = 6876.1 :=
by
  sorry

end shanmukham_total_payment_l142_142856


namespace necessary_but_not_sufficient_l142_142750

variables (p q : Prop)

theorem necessary_but_not_sufficient :
  (p → q) ∧ ¬(q → p) → (¬p → ¬q) ∧ ¬(¬q → ¬p) :=
by
  assume h : (p → q) ∧ ¬(q → p)
  sorry

end necessary_but_not_sufficient_l142_142750


namespace find_f_2017_l142_142711

def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_shifted : ∀ x : ℝ, f (1 - x) = f (x + 1)
axiom f_neg_one : f (-1) = 2

theorem find_f_2017 : f 2017 = -2 :=
by
  sorry

end find_f_2017_l142_142711


namespace average_difference_l142_142597

def student_counts : List ℕ := [60, 30, 20, 5, 3, 2]
def total_students : ℕ := 120
def total_teachers : ℕ := 6

def t (student_counts : List ℕ) (total_teachers : ℕ) : ℚ := 
  (student_counts.foldl (· + ·) 0) / total_teachers

def s (student_counts : List ℕ) (total_students : ℕ) : ℚ := 
  (student_counts.map (λ n => n * n)).foldl (· + ·) 0 / total_students

theorem average_difference (student_counts : List ℕ) (total_students total_teachers : ℕ) :
  let t := t student_counts total_teachers
  let s := s student_counts total_students
  t - s = -21 :=
by
  sorry

end average_difference_l142_142597


namespace fraction_day_loaded_box_l142_142980

theorem fraction_day_loaded_box
  (D W : ℚ)
  (N = (3 / 4) * D)
  (E = (5 / 6) * D)
  (WN = (4 / 7) * W)
  (WE = (3 / 5) * W) :
  (D * W) / ((27 / 14) * D * W) = (14 / 27) := 
by
  sorry

end fraction_day_loaded_box_l142_142980


namespace loss_is_negative_one_point_twenty_seven_percent_l142_142577

noncomputable def book_price : ℝ := 600
noncomputable def gov_tax_rate : ℝ := 0.05
noncomputable def shipping_fee : ℝ := 20
noncomputable def seller_discount_rate : ℝ := 0.03
noncomputable def selling_price : ℝ := 624

noncomputable def gov_tax : ℝ := gov_tax_rate * book_price
noncomputable def seller_discount : ℝ := seller_discount_rate * book_price
noncomputable def total_cost : ℝ := book_price + gov_tax + shipping_fee - seller_discount
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def loss_percentage : ℝ := (profit / total_cost) * 100

theorem loss_is_negative_one_point_twenty_seven_percent :
  loss_percentage = -1.27 :=
by
  sorry

end loss_is_negative_one_point_twenty_seven_percent_l142_142577


namespace volume_of_parallelepiped_l142_142098

variables (a b : ℝ) (h1 : a > 0) (h2 : b > 0)

theorem volume_of_parallelepiped (h : ∃ d, d ∈ ℝ ∧ d^2 = a^2 + b^2 + h_height^2) :
  let S := a * b,
      H := sqrt (3 * a ^ 2 - b ^ 2),
      V := S * H in
  V = a * b * sqrt (3 * a ^ 2 - b ^ 2) :=
sorry

end volume_of_parallelepiped_l142_142098


namespace volume_of_pyramid_is_1280_l142_142956

def pyramid_square_base_volume (s ABE_area CDE_area d: ℝ) : ℝ :=
  1/3 * s^2 * d

theorem volume_of_pyramid_is_1280 {s ABE_area CDE_area d : ℝ} 
  (h_base : s^2 = 256)
  (h_ABE : (2 * ABE_area) / s = 15)
  (h_CDE : (2 * CDE_area) / s = 17)
  (h_distance : d = 15): 
  pyramid_square_base_volume s ABE_area CDE_area d = 1280 := 
by 
  rw [←h_base, ←h_distance]
  sorry

end volume_of_pyramid_is_1280_l142_142956


namespace natural_numbers_divisible_by_7_between_200_400_l142_142365

theorem natural_numbers_divisible_by_7_between_200_400 : 
  { n : ℕ | 200 <= n ∧ n <= 400 ∧ n % 7 = 0 }.to_finset.card = 29 := 
  sorry

end natural_numbers_divisible_by_7_between_200_400_l142_142365


namespace isosceles_triangle_sides_l142_142609

theorem isosceles_triangle_sides :
  ∃ (a b c : ℝ), is_isosceles a b c ∧
  let S_BCD := (72 / 11 : ℝ)
      S_BAD := (55 / 11 : ℝ) 
  in (area_triangle_split_by_bisector a b c S_BCD S_BAD ∧ a = 6 ∧ b = 5 ∧ c = 5) :=
sorry

-- Definitions used in the theorem statement
def is_isosceles (a b c : ℝ) : Prop :=
  (a = b) ∨ (a = c) ∨ (b = c)

def area_triangle_split_by_bisector (a b c S_BCD S_BAD: ℝ) : Prop :=
  (S_BCD + S_BAD = (127 / 11) ∧ (S_BAD / S_BCD = 55 / 66))

end isosceles_triangle_sides_l142_142609


namespace cos_angle_A_l142_142007

noncomputable def triangle_cosine (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℚ :=
  b / c

theorem cos_angle_A (a b c : ℕ) (h : a^2 + b^2 = c^2) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  triangle_cosine a b c h = 4 / 5 :=
by
  rw [ha, hb, hc]
  -- sorry, the proof can be filled in later
  sorry

end cos_angle_A_l142_142007


namespace parabola_zero_difference_l142_142188

noncomputable def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

theorem parabola_zero_difference 
    (a b c : ℝ)
    (h_pass1 : quadratic a b c 3 = -9)
    (h_pass2 : quadratic a b c 5 = 7)
    (h_vertex : ∃ k : ℝ, ∀ x : ℝ, k = a * (x - 3)^2 - 9) :
    let zeros := {x | quadratic a b c x = 0} in
    let x1 := sup zeros in
    let x2 := inf zeros in
    x1 - x2 = 3 := 
sorry

end parabola_zero_difference_l142_142188


namespace select_stable_athlete_l142_142211

theorem select_stable_athlete :
  ∀ (S_A² S_B² S_C² S_D² : ℝ), 
  S_A² = 0.4 ∧ S_B² = 0.5 ∧ S_C² = 0.6 ∧ S_D² = 0.3 ->
  (S_D² < S_A² ∧ S_D² < S_B² ∧ S_D² < S_C²) :=
by
  intros S_A² S_B² S_C² S_D² h,
  cases h with hA hBCD,
  cases hBCD with hB hCD,
  cases hCD with hC hD,
  rw [hA, hB, hC, hD],
  split,
  { exact lt_trans (by norm_num) (by norm_num) },
  { split,
    { exact lt_trans (by norm_num) (by norm_num) },
    { exact by norm_num } }

sorry

end select_stable_athlete_l142_142211


namespace pima_initial_investment_l142_142062

/-- Pima's initial investment in Ethereum. The investment value gained 25% in the first week and 50% of its current value in the second week. The final investment value is $750. -/
theorem pima_initial_investment (I : ℝ) 
  (h1 : 1.25 * I * 1.5 = 750) : I = 400 :=
sorry

end pima_initial_investment_l142_142062


namespace proof_AD_gt_l142_142064

variables (A B C D : Type)

-- Define terms for lengths of segments
variables (AB BC DA CD AC BD AD : ℝ)

-- Define known values
def known_values (BC BD AC : ℝ): Prop :=
BC = 4 ∧ BD = 4 ∧ AC = 3

-- Define the ratio condition as stated in the problem
def ratio_condition (AB BC DA CD : ℝ): Prop :=
AB / BC = DA / CD

-- Applying Ptolemy's theorem on the segments
def ptolemys_theorem (AB CD BC DA AC BD : ℝ): Prop :=
AB * CD + BC * DA = AC * BD

-- The mathematically equivalent proof statement
theorem proof_AD_gt {A B C D : Type}
  (BC BD AC : ℝ)
  (BasicAssump : known_values BC BD AC)
  (Ratio : ratio_condition AB BC DA CD)  
  (Ptolemy : ptolemys_theorem AB CD BC DA AC BD):
  AD = 3 / 2 :=
by {
  cases BasicAssump with hBC hBD,
  cases hBD with hBD hAC,
  rw [hBC, hBD, hAC],
  sorry
}

end proof_AD_gt_l142_142064


namespace unique_positive_solution_eq_15_l142_142654

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l142_142654


namespace smaller_angle_at_7_15_is_correct_l142_142537

-- Definitions based on the conditions
def angle_per_hour_mark : ℝ := 30.0
def hour_hand_angle_at_7 : ℝ := 7 * angle_per_hour_mark
def minutes_elapsed : ℝ := 15
def minute_hand_angle_at_15 : ℝ := 90.0
def hour_hand_movement_per_minute : ℝ := angle_per_hour_mark / 60.0
def hour_hand_angle_at_7_15 : ℝ := hour_hand_angle_at_7 + (minutes_elapsed * hour_hand_movement_per_minute)
def absolute_difference (a b : ℝ) : ℝ := abs (a - b)
def smaller_angle (a : ℝ) : ℝ := if a > 180 then 360 - a else a

-- The theorem to prove the measure of the smaller angle at 7:15
theorem smaller_angle_at_7_15_is_correct : smaller_angle (absolute_difference hour_hand_angle_at_7_15 minute_hand_angle_at_15) = 127.5 := by
  sorry

end smaller_angle_at_7_15_is_correct_l142_142537


namespace graph_passes_through_point_l142_142494

theorem graph_passes_through_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :  
  ∃ x y : ℝ, (x = 0 ∧ y = 5) ∧ y = 2 * a^x + 3 := 
by {
  use [0, 5],
  split,
  { refl },
  { sorry }
}

end graph_passes_through_point_l142_142494


namespace perfect_squares_ending_in_4_5_6_less_than_2000_l142_142745

theorem perfect_squares_ending_in_4_5_6_less_than_2000 :
  let squares := { n : ℕ | n * n < 2000 ∧ (n * n % 10 = 4 ∨ n * n % 10 = 5 ∨ n * n % 10 = 6) } in
  squares.card = 23 :=
by
  sorry

end perfect_squares_ending_in_4_5_6_less_than_2000_l142_142745


namespace count_ordered_pairs_m_n_l142_142756

theorem count_ordered_pairs_m_n :
  ∃ (s : Finset (ℝ × ℕ)), s.card = 4 ∧ 
  ∀ (mn ∈ s), 
    let m := mn.1, n := mn.2 in
    n > 0 ∧ 
    (m - 1) ^ n = 2 ^ n ∧ 
    (n = 9 ∨ n = 10 ∨ n = 11) ∧ 
    (if n = 10 then (m = 3 ∨ m = -1) else m = 3) := 
sorry

end count_ordered_pairs_m_n_l142_142756


namespace range_of_m_l142_142677

open Real

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ (x : ℝ), x > 0 → log x ≤ x * exp (m^2 - m - 1)

theorem range_of_m : 
  {m : ℝ | satisfies_inequality m} = {m : ℝ | m ≤ 0 ∨ m ≥ 1} :=
by 
  sorry

end range_of_m_l142_142677


namespace no_real_roots_of_equation_l142_142992

theorem no_real_roots_of_equation :
  ¬ ∃ x : ℝ, sqrt (x + 16) + 4 / sqrt (x + 16) = 7 :=
begin
  sorry
end

end no_real_roots_of_equation_l142_142992


namespace complex_in_third_quadrant_l142_142275

def complex_number := (-1 : ℂ) - 2 * complex.I
def denominator := (1 : ℂ) - 2 * complex.I
def complex_quotient := complex_number / denominator

theorem complex_in_third_quadrant
  (complex_number / denominator = (-1 : ℂ) - (4 / 5) * complex.I) :
  complex.quotient_im < 0 ∧ complex.quotient_re < 0 :=
sorry

end complex_in_third_quadrant_l142_142275


namespace carla_zoo_l142_142223

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end carla_zoo_l142_142223


namespace bug_positions_l142_142789

def fibonacci (n : ℕ) : ℕ :=
  Nat.recOn n
    1
    (λ p f, match p with
      | 0 => 1
      | p+1 => Nat.recOn p 1 (λ _ f₀ => f + f₀))

def reachable_positions (n : ℕ) : finset ℚ :=
  let step (positions: finset ℚ) : finset ℚ :=
    positions ∪ positions.image (λ x, x + 2) ∪ positions.image (λ x, x / 2)
  (Nat.iterate step n {1 : ℚ})

theorem bug_positions (n : ℕ) : reachable_positions n = F_{n + 4} - (n + 4) :=
sorry 

end bug_positions_l142_142789


namespace medians_form_similar_triangle_l142_142199

theorem medians_form_similar_triangle (Δ : Type*) [MetricSpace Δ] [EuclideanGeometry Δ] (A B C : Δ) :
  let G := centroid A B C,
      T1 := triangle A B G,
      T2 := triangle A C G,
      T3 := triangle B C G in
  (similar T1 (triangle A B C) ∨ similar T2 (triangle A B C) ∨ similar T3 (triangle A B C)) := by
sorry

end medians_form_similar_triangle_l142_142199


namespace small_cubes_for_larger_cube_l142_142928

theorem small_cubes_for_larger_cube (VL VS : ℕ) (h : VL = 125 * VS) : (VL / VS = 125) :=
by {
    sorry
}

end small_cubes_for_larger_cube_l142_142928


namespace part1_l142_142412

theorem part1 (ABC : Triangle) (I O : Point) (h_incenter : incenter ABC I) (h_circumcenter : circumcenter ABC O)
  (h_angle_B : ∠B = 60) (h_OI_AB_AC : |OI| = |AB - AC|) : is_equilateral ABC := 
sorry

end part1_l142_142412


namespace third_bouquet_carnations_l142_142908

/--
Trevor buys three bouquets of carnations. The first included 9 carnations, and the second included 14 carnations. If the average number of carnations in the bouquets is 12, then the third bouquet contains 13 carnations.
-/
theorem third_bouquet_carnations (n1 n2 n3 : ℕ)
  (h1 : n1 = 9)
  (h2 : n2 = 14)
  (h3 : (n1 + n2 + n3) / 3 = 12) :
  n3 = 13 :=
by
  sorry

end third_bouquet_carnations_l142_142908


namespace average_of_sequence_l142_142261

theorem average_of_sequence (z : ℝ) : 
  (0 + 3 * z + 9 * z + 27 * z + 81 * z) / 5 = 24 * z :=
by
  sorry

end average_of_sequence_l142_142261


namespace intersecting_perpendicular_line_l142_142869

theorem intersecting_perpendicular_line
  (L1 : ∀ x y : ℝ, x + y - 3 = 0)
  (L2 : ∀ x y : ℝ, 2 * x - y + 6 = 0)
  (L3 : ∀ x y : ℝ, 3 * x - 2 * y + 1 = 0) :
  ∃ a b c : ℝ, a = 2 ∧ b = 3 ∧ c = -8 ∧ (∀ x y : ℝ, a * x + b * y + c = 0) :=
begin
  sorry
end

end intersecting_perpendicular_line_l142_142869


namespace valid_paths_in_grid_l142_142051

theorem valid_paths_in_grid : 
  let total_paths := Nat.choose 15 4;
  let paths_through_EF := (Nat.choose 7 2) * (Nat.choose 7 2);
  let valid_paths := total_paths - 2 * paths_through_EF;
  grid_size == (11, 4) ∧
  blocked_segments == [((5, 2), (5, 3)), ((6, 2), (6, 3))] 
  → valid_paths = 483 :=
by
  sorry

end valid_paths_in_grid_l142_142051


namespace tan_subtraction_l142_142388

theorem tan_subtraction
  (x y : ℝ)
  (h1 : sin x + sin y = 15 / 17)
  (h2 : cos x + cos y = 8 / 17) :
  tan (x - y) = 870 / 49 :=
by
  sorry

end tan_subtraction_l142_142388


namespace problem_statement_l142_142816

variables {Line : Type} {Plane : Type} [parallel : Line → Plane → Prop] [subset : Line → Plane → Prop]

-- Definitions for variables
variables {m n : Line} {α β : Plane}

-- Hypotheses
axiom parallel_planes : ∀ {α β : Plane}, α ∥ β → ∀ l, l ∥ α → l ∥ β
axiom parallel_lines : ∀ l₁ l₂ : Line, (l₁ ∥ l₂) → (l₂ ∥ l₁)
axiom line_not_subset : ∀ l : Line, ∀ p : Plane, ¬(l ⊂ p)

-- Problem statement
theorem problem_statement (h1 : α ∥ β) (h2 : m ∥ α) (h3 : n ∥ m) (h4 : ¬(n ⊂ β)) : n ∥ β :=
by {
  sorry
}

end problem_statement_l142_142816


namespace angles_equal_BAD_BCD_l142_142584

-- Given the conditions on the positions of the vertices of the convex quadrilateral,
-- we define A, B, C, and D according to the problem statement.

variables {R : Type*} [Real R]

structure Point (R : Type*) := (x : R) (y : R)

def on_hyperbola (p : Point R) : Prop := p.y = 1 / p.x

-- Define points on the hyperbola
def A : Point R := ⟨-c, -1/c⟩
def B : Point R := ⟨b, 1/b⟩
def C : Point R := ⟨c, 1/c⟩
def D : Point R := ⟨d, 1/d⟩

-- Conditions
variables (b c d : R) (hbc : 0 < b) (hcc : 0 < c) (hdc_neg : d < 0) (hdc_pos : 0 < d)
          (left_BC : b < c) (pass_origin_AC : A.x * C.y = A.y * C.x)

theorem angles_equal_BAD_BCD :
  ∃ (A B C D : Point R),
    on_hyperbola A ∧ on_hyperbola B ∧ on_hyperbola C ∧ on_hyperbola D ∧ 
    left_BC ∧ pass_origin_AC ∧ 
    ∠BAD = ∠BCD :=
sorry

end angles_equal_BAD_BCD_l142_142584


namespace height_of_point_A_l142_142322

theorem height_of_point_A
  (a α β : ℝ)
  (α_gt_β : α > β)
  (DC_eq_a : ∀ (D C B : ℝ), (D, C, B are_collinear_on_line ∧ DC = a))
  (elevation_C : angle_of_elevation A C = α)
  (elevation_D : angle_of_elevation A D = β):
  AB = a * (sin α) * (sin β) / (sin (α - β)) := 
sorry

end height_of_point_A_l142_142322


namespace abcd_mod_7_zero_l142_142813

theorem abcd_mod_7_zero
  (a b c d : ℕ)
  (h1 : a + 2 * b + 3 * c + 4 * d ≡ 1 [MOD 7])
  (h2 : 2 * a + 3 * b + c + 2 * d ≡ 5 [MOD 7])
  (h3 : 3 * a + b + 2 * c + 3 * d ≡ 3 [MOD 7])
  (h4 : 4 * a + 2 * b + d + c ≡ 2 [MOD 7])
  (ha : a < 7) (hb : b < 7) (hc : c < 7) (hd : d < 7) :
  (a * b * c * d) % 7 = 0 :=
by sorry

end abcd_mod_7_zero_l142_142813


namespace susan_age_in_5_years_l142_142167

-- Definitions of the given conditions
def james_age_in_15_years : ℕ := 37
def years_until_james_is_37 : ℕ := 15
def years_ago_james_twice_janet : ℕ := 8
def susan_born_when_janet_turned : ℕ := 3
def years_to_future_susan_age : ℕ := 5

-- Calculate the current age of people involved
def james_current_age : ℕ := james_age_in_15_years - years_until_james_is_37
def james_age_8_years_ago : ℕ := james_current_age - years_ago_james_twice_janet
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def janet_current_age : ℕ := janet_age_8_years_ago + years_ago_james_twice_janet
def susan_current_age : ℕ := janet_current_age - susan_born_when_janet_turned

-- Prove that Susan will be 17 years old in 5 years
theorem susan_age_in_5_years (james_age_future : james_age_in_15_years = 37)
  (years_until_james_37 : years_until_james_is_37 = 15)
  (years_ago_twice_janet : years_ago_james_twice_janet = 8)
  (susan_born_janet : susan_born_when_janet_turned = 3)
  (years_future : years_to_future_susan_age = 5) :
  susan_current_age + years_to_future_susan_age = 17 := by
  -- The proof is omitted
  sorry

end susan_age_in_5_years_l142_142167


namespace area_of_quadrilateral_AEDC_l142_142793
-- Import the Mathlib library for access to comprehensive mathematics capabilities.

-- Define the problem in Lean 4, considering $PE = 3$, $PD = 4$, and $DE = 5$.
noncomputable def area_quadrilateral_AEDC : ℝ :=
  let PE := 3
  let PD := 4
  let DE := 5
  let area_triangle := λ a b : ℝ, (1 / 2) * a * b
  area_triangle 8 3 + area_triangle 4 3 + area_triangle 6 4 + area_triangle 8 6

-- State the theorem that the area of quadrilateral AEDC is 54.
theorem area_of_quadrilateral_AEDC : area_quadrilateral_AEDC = 54 := by
  sorry

end area_of_quadrilateral_AEDC_l142_142793


namespace transformed_function_correct_l142_142334

open Real

noncomputable def f (x φ : ℝ) : ℝ := sqrt 3 * sin (x + φ) - cos (x + φ)

-- Conditions: 0 < φ < π
def φ_cond (φ : ℝ) := 0 < φ ∧ φ < π

-- Transformation: horizontally halved and shifted right by π/8
def g (x φ : ℝ) : ℝ := 2 * sin (2 * (x - π / 8))

theorem transformed_function_correct (φ x : ℝ) (hφ: φ_cond φ) (h_odd: ∀ x, f (-x) φ = -f x φ) : 
  g x (π / 6) = 2 * sin (2 * x - π / 4) :=
sorry

end transformed_function_correct_l142_142334


namespace time_to_complete_work_l142_142944

-- Define the conditions as variables
variables (W : ℝ) -- Total work
variables (A B C : ℝ) -- Work rates of A, B, C respectively

-- Define the conditions
def work_rate_A_B := W / 4
def work_rate_A := W / 6
def work_rate_C := W / 12
def work_rate_B := work_rate_A_B - work_rate_A -- This can be further simplified as W/12 based on given conditions

-- Define the combined work rate, which is the sum of the individual work rates
def combined_work_rate := work_rate_A + work_rate_B + work_rate_C

-- State the theorem to be proved
theorem time_to_complete_work : (W / combined_work_rate) = 3 :=
by
  -- Assume the steps to simplify the fractions and solve for time are done here
  sorry

end time_to_complete_work_l142_142944


namespace monkeys_more_than_giraffes_l142_142229

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end monkeys_more_than_giraffes_l142_142229


namespace time_first_half_l142_142169

noncomputable def initial_speed : ℝ := 14
noncomputable def increased_speed : ℝ := initial_speed + 2
noncomputable def total_distance : ℝ := 140
noncomputable def half_distance : ℝ := total_distance / 2
noncomputable def time_second_half : ℝ := 7 / 3

theorem time_first_half :
  let s1 := initial_speed,
      s2 := increased_speed,
      d := total_distance / 2,
      t2 := time_second_half
  in (d / s1) = 5 :=
by
  let s1 := 14
  let s2 := 16
  let d := 70
  let t2 := 7 / 3
  sorry

end time_first_half_l142_142169


namespace increasing_interval_range_on_interval_l142_142285

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  cos x * (a * sin x - cos x) + cos^2 (π / 2 - x)

theorem increasing_interval (a : ℝ) (h : f a (-π / 3) = f a 0) : 
  ∀ k : ℤ, strict_mono_on (f 2*sqrt(3)) (Icc (-π / 6 + k * π) (π / 3 + k * π)) :=
sorry

theorem range_on_interval (a b c : ℝ) (h : (a^2 + c^2 - b^2) / (a^2 + b^2 - c^2) = c / (2 * a - c)) : 
  range (f 2*sqrt(3)) (Ioo 0 (π/3)) = Ioo (-1 : ℝ) 2 :=
sorry

end increasing_interval_range_on_interval_l142_142285


namespace carla_zoo_l142_142224

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end carla_zoo_l142_142224


namespace multiplication_expression_l142_142621

theorem multiplication_expression : 45 * 27 + 18 * 45 = 2025 := by
  sorry

end multiplication_expression_l142_142621


namespace number_divisor_property_l142_142130

theorem number_divisor_property (s : Set ℕ) (h_s : s ⊆ Finset.range 2015) (h_size : s.card = 1008) :
  ∃ a b ∈ s, a ≠ b ∧ a ∣ b := 
by
  sorry

end number_divisor_property_l142_142130


namespace sqrt_13_parts_sqrt_13_expression_l142_142733

noncomputable theory

def integer_part (a : ℝ) : ℤ :=
⌊a⌋

def decimal_part (a : ℝ) : ℝ :=
a - ⌊a⌋

def sqrt_13_integer_part : ℤ :=
integer_part (Real.sqrt 13)

def sqrt_13_decimal_part : ℝ :=
decimal_part (Real.sqrt 13)

theorem sqrt_13_parts :
  sqrt_13_integer_part = 3 ∧ sqrt_13_decimal_part = Real.sqrt 13 - 3 :=
by
  sorry

theorem sqrt_13_expression :
  2 * sqrt_13_integer_part - sqrt_13_decimal_part + Real.sqrt 13 = 9 :=
by
  sorry

end sqrt_13_parts_sqrt_13_expression_l142_142733


namespace angle_bisectors_form_inscribed_quadrilateral_l142_142844

-- Definitions for basic angle representation and quadrilateral
structure Quadrilateral (α : Type) :=
  (A B C D : α)

structure AngleBisector where
  quadrilateral : Quadrilateral ℝ
  bisectorA bisectorB bisectorC bisectorD : ℝ

def isConvex (Q : Quadrilateral ℝ) : Prop := sorry

def sumInternalAngles (Q : Quadrilateral ℝ) : ℝ :=
  let ⟨A, B, C, D⟩ := Q
  (D - A) + (B - A) + (C - D) + (D - C)

def isCyclic (Q : Quadrilateral ℝ) (a b c d : ℝ) : Prop :=
  let α := (a + b) 
  let β := (c + d) 
  α + β = 180

-- Given conditions based on the problem
axiom quadrilateral_convex (Q : Quadrilateral ℝ) : isConvex Q

axiom quadrilateral_internal_angle_sum (Q : Quadrilateral ℝ) : sumInternalAngles Q = 360

-- The main theorem statement we need to prove
theorem angle_bisectors_form_inscribed_quadrilateral (Q : Quadrilateral ℝ) (a b c d : ℝ) (h: AngleBisector) :
  isConvex Q →
  sumInternalAngles Q = 360 →
  isCyclic Q a b c d :=
begin
  sorry
end

end angle_bisectors_form_inscribed_quadrilateral_l142_142844


namespace geometric_sequence_on_line_l142_142491

theorem geometric_sequence_on_line (a b : ℝ) (n : ℕ) (h : a ≠ 1) :
  let S_n := b * (1 - a^n) / (1 - a) in
  let S_n_plus_1 := b * (1 - a^(n + 1)) / (1 - a) in
  S_n_plus_1 = a * S_n + b :=
sorry

end geometric_sequence_on_line_l142_142491


namespace polar_eq_of_circle_product_of_distances_MA_MB_l142_142417

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

end polar_eq_of_circle_product_of_distances_MA_MB_l142_142417


namespace range_of_a_l142_142735

theorem range_of_a (a : ℝ) (h1 : 2 * a + 1 < 17) (h2 : 2 * a + 1 > 7) : 3 < a ∧ a < 8 := by
  sorry

end range_of_a_l142_142735


namespace max_integer_a_for_real_roots_l142_142680

theorem max_integer_a_for_real_roots (a : ℤ) :
  (((a - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1) → a ≤ 0 ∧ (∀ b : ℤ, ((b - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1 → b ≤ 0) :=
sorry

end max_integer_a_for_real_roots_l142_142680


namespace am_gm_inequality_l142_142029

theorem am_gm_inequality (a1 a2 a3 : ℝ) (h₀ : 0 < a1) (h₁ : 0 < a2) (h₂ : 0 < a3) (h₃ : a1 + a2 + a3 = 1) : 
  1 / a1 + 1 / a2 + 1 / a3 ≥ 9 :=
by
  sorry

end am_gm_inequality_l142_142029


namespace largest_k_for_sum_of_consecutive_integers_l142_142268

theorem largest_k_for_sum_of_consecutive_integers 
  (k : ℕ) (h : 3 ^ 12 = (finset.range k).sum(λ i, i + n)) : k = 729 :=
sorry

end largest_k_for_sum_of_consecutive_integers_l142_142268


namespace find_ab_l142_142870

-- Define the statement to be proven
theorem find_ab (a b : ℕ) (h1 : (a + b) % 3 = 2)
                           (h2 : b % 5 = 3)
                           (h3 : (b - a) % 11 = 1) :
  10 * a + b = 23 := 
sorry

end find_ab_l142_142870


namespace largest_k_consecutive_sum_l142_142266

theorem largest_k_consecutive_sum (k : ℕ) (h1 : (∃ n : ℕ, 3^12 = k * n + (k*(k-1))/2)) : k ≤ 729 :=
by
  -- Proof omitted for brevity
  sorry

end largest_k_consecutive_sum_l142_142266


namespace value_of_m_l142_142754

def a (k m : ℕ) : ℕ :=
  (2 * k + m) ^ k

theorem value_of_m (h : a (a (a 0 m) m) m = 343) : m = 1 :=
by
  have h1 : a 0 m = 1 := by
    sorry
  
  have h2 : a (a 0 m) m = 2 + m := by
    sorry

  have h3 : a (a (a 0 m) m) m = (4 + 3 * m) ^ (2 + m) := by
    sorry

  rw h1 at h
  rw h2 at h
  rw h3 at h
  sorry

end value_of_m_l142_142754


namespace number_divisor_property_l142_142129

theorem number_divisor_property (s : Set ℕ) (h_s : s ⊆ Finset.range 2015) (h_size : s.card = 1008) :
  ∃ a b ∈ s, a ≠ b ∧ a ∣ b := 
by
  sorry

end number_divisor_property_l142_142129


namespace find_f_5_l142_142861
noncomputable theory

def linear_function (a b x : ℝ) := a * x + b

theorem find_f_5 (a b : ℝ) (h1 : ∀ x, linear_function a b x = 3 * linear_function (1/a) (-b/a) x + 9)
                 (h2 : linear_function a b 2 = 5)
                 (h3 : linear_function a b 3 = 9) :
  linear_function a b 5 = 9 - 8 * real.sqrt 3 :=
sorry

end find_f_5_l142_142861


namespace find_other_endpoint_l142_142947

set_option pp.funBinderTypes true

def circle_center : (ℝ × ℝ) := (5, -2)
def diameter_endpoint1 : (ℝ × ℝ) := (1, 2)
def diameter_endpoint2 : (ℝ × ℝ) := (9, -6)

theorem find_other_endpoint (c : ℝ × ℝ) (e1 : ℝ × ℝ) (e2 : ℝ × ℝ) : 
  c = circle_center ∧ e1 = diameter_endpoint1 → e2 = diameter_endpoint2 := by
  sorry

end find_other_endpoint_l142_142947


namespace least_n_divisible_by_135_l142_142455

noncomputable def b : ℕ → ℕ
| 10       := 10
| (n + 1)  := if n ≥ 10 then 150 * b n + (n + 1) ^ 2 else 0

theorem least_n_divisible_by_135 : ∃ n > 10, b n % 135 = 0 ∧ ∀ k, (k > 10 ∧ k < n) → b k % 135 ≠ 0 :=
by {
    use 27,
    sorry
}

end least_n_divisible_by_135_l142_142455


namespace collinear_points_sum_l142_142633

theorem collinear_points_sum (x y z : ℝ) 
  (h_collinear : ∃ (a b c : Real), ∀ (t : Real), (a * t + b, b * t + c) ∈ ({(x, 1, z), (2, y, z), (x, y, 3)} : set (ℝ × ℝ × ℝ))) :
  x + y = 3 :=
sorry

end collinear_points_sum_l142_142633


namespace number_of_ways_to_select_officers_l142_142896

-- Definitions based on conditions
def boys : ℕ := 6
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def officers_to_select : ℕ := 3

-- Number of ways to choose 3 individuals out of 10
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_choices : ℕ := choose total_people officers_to_select

-- Number of ways to choose 3 boys out of 6 (0 girls)
def all_boys_choices : ℕ := choose boys officers_to_select

-- Number of ways to choose at least 1 girl
def at_least_one_girl_choices : ℕ := total_choices - all_boys_choices

-- Theorem to prove the number of ways to select the officers
theorem number_of_ways_to_select_officers :
  at_least_one_girl_choices = 100 := by
  sorry

end number_of_ways_to_select_officers_l142_142896


namespace pens_sold_l142_142836

theorem pens_sold (initial_pens : ℕ) (pens_left : ℕ) : initial_pens = 106 → pens_left = 14 → initial_pens - pens_left = 92 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end pens_sold_l142_142836


namespace parabola_ellipse_l142_142187

-- Conditions

def is_vertex_origin (p : ℝ × ℝ) : Prop :=
p = (0, 0)

def directrix_passes_focus_perpendicular
  (directrix : ℝ → Prop) (focus : ℝ × ℝ) (major_axis : ℝ → Prop) : Prop :=
directrix focus.1 ∧ (∃ k : ℝ, directrix = {x : ℝ | x = k ∧ ∀ y : ℝ, (major_axis y) → (x ≠ y)})

def is_intersection_point (p : ℝ × ℝ) : Prop :=
p = (-2/3, 2 * Real.sqrt 6 / 3)

def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Proof statement

theorem parabola_ellipse (
  h_vertex : ∃ p : ℝ × ℝ, is_vertex_origin p,
  h_directrix : ∃ f : ℝ × ℝ → Prop, ∃ focus : ℝ × ℝ, ∃ major_axis : ℝ → Prop, 
                    directrix_passes_focus_perpendicular (f focus) focus major_axis,
  h_intersection : ∃ p : ℝ × ℝ, is_intersection_point p,
  h_ellipse : ∃ a b x y : ℝ, ellipse_C a b x y
) : 
-- Equation of the parabola
(∀ (x y : ℝ), y^2 = -4 * x) ∧
-- Equation of the ellipse C
(∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) ∧
-- Equation of the hyperbola
(∃ (a₁ b₁ : ℝ), 
    (c : ℝ) = 1 ∧ 
    (y / x = 4 / 3 ∨ y / x = -4 / 3) ∧ 
    (x^2 / (a₁)^2 - y^2 / (b₁)^2 = 1) → 
    a₁ = 3 / 5 ∧ b₁ = 4 / 5 → 
    (25 * x^2 / 9 - 25 * y^2 / 16 = 1)) := 
  sorry

end parabola_ellipse_l142_142187


namespace an_geometric_l142_142286

-- Define the functions and conditions
def f (x : ℝ) (b : ℝ) : ℝ := b * x + 1

def g (n : ℕ) (b : ℝ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => f (g n b) b

-- Define the sequence a_n
def a (n : ℕ) (b : ℝ) : ℝ :=
  g (n + 1) b - g n b

-- Prove that a_n is a geometric sequence
theorem an_geometric (b : ℝ) (h : b ≠ 1) : 
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) b = q * a n b :=
sorry

end an_geometric_l142_142286


namespace sum_f_l142_142673

def f (x : ℝ) : ℝ := 2 / (x + 1)

theorem sum_f {x : ℝ} (hx : 0 < x) : 
  (f(100) + f(99) + f(98) + f(2) + f(1) + 
   f(1/2) + f(1/98) + f(1/99) + f(1/100)) = 199 :=
by
  sorry  -- The proof is omitted as per instructions

end sum_f_l142_142673


namespace trapezoid_ABMD_rhombus_line_DN_midpoint_BE_l142_142018

noncomputable def is_rhombus (A B M D : Point) : Prop :=
  dist A B = dist B M ∧ dist B M = dist M D ∧ dist M D = dist D A

noncomputable def passes_through_midpoint (D N B E : Point) (mid_BE: Point) : Prop :=
  midpoint B E mid_BE ∧ on_line D N mid_BE

theorem trapezoid_ABMD_rhombus 
  (A B C D M E O N : Point)
  (h1 : parallel AB CD)
  (h2 : 2 * dist A B = dist C D)
  (h3 : perpendicular BD BC)
  (h4 : midpoint C D M)
  (h5 : intersection BC AD E)
  (h6 : intersection AM BD O)
  (h7 : intersection OE AB N) :
  is_rhombus A B M D :=
sorry

theorem line_DN_midpoint_BE
  (A B C D M E O N : Point)
  (h1 : parallel AB CD)
  (h2 : 2 * dist A B = dist C D)
  (h3 : perpendicular BD BC)
  (h4 : midpoint C D M)
  (h5 : intersection BC AD E)
  (h6 : intersection AM BD O)
  (h7 : intersection OE AB N)
  (mid_BE : Point) (h8: midpoint B E mid_BE):
  passes_through_midpoint D N B E mid_BE :=
sorry

end trapezoid_ABMD_rhombus_line_DN_midpoint_BE_l142_142018


namespace video_has_2600_dislikes_l142_142571

def likes := 3000
def initial_dislikes := 1500 + 100
def additional_dislikes := 1000
def total_dislikes := initial_dislikes + additional_dislikes

theorem video_has_2600_dislikes:
  total_dislikes = 2600 :=
by
  unfold likes initial_dislikes additional_dislikes total_dislikes
  sorry

end video_has_2600_dislikes_l142_142571


namespace greenhouse_dimensions_l142_142999

noncomputable def height := 20
noncomputable def width := 40

theorem greenhouse_dimensions (h w : ℕ) (hw : w = 2 * h) (ha : w * h ≥ 800) : 
  h = height ∧ w = width :=
by
  sorry

end greenhouse_dimensions_l142_142999


namespace part1_vertex_part1_range_y_part2_expression_l142_142295

-- Part 1: Quadratic function with b=4 and c=3
def part1_questions := let b := 4; let c := 3 in
  (⟨vertex_coordinates : (2, 7)⟩, ⟨range_y : -2 ≤ fun x => -x^2 + b * x + c x ≤ 7 for -1 ≤ x ≤ 3⟩)

-- Part 2: Quadratic function with specific conditions on max values
def part2_conditions := let y := fun x => -x^2 + b * x + c in
  (⟨max_y_on_neg := 2 if x ≤ 0⟩, ⟨max_y_on_pos := 3 if x > 0⟩)

theorem part1_vertex : (part1_questions.vertex_coordinates).fst = ()


theorem part1_range_y : ∀ x, -1 ≤ x ≤ 3 → -2 ≤ -x^2 + 4 * x + 3 ≤ 7 :=
by sorry

theorem part2_expression : ∀ x, b = 2 ∧ c = 2 →

  ∃ (b = 2) ∧ (x ≤ 0 → y x = 2) ∧ (x > 0 → y x = 3)
:=
by sorry

end part1_vertex_part1_range_y_part2_expression_l142_142295


namespace perfect_squares_digit_4_5_6_l142_142739

theorem perfect_squares_digit_4_5_6 (n : ℕ) (hn : n^2 < 2000) : 
  (∃ k : ℕ, k = 18) :=
  sorry

end perfect_squares_digit_4_5_6_l142_142739


namespace min_distance_curve_line_l142_142446

theorem min_distance_curve_line :
  let P : ℝ × ℝ := (1, 1),
      curve_eq : ℝ → ℝ := λ x, x^2 - Real.log x,
      line_eq : ℝ → ℝ := λ y, y = x - 2,
      dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2),
      d := λ (x₁ : ℝ), (|x₁^2 - Real.log x₁ - x₁ + 2|) / (Real.sqrt 2)
  in d 1 = Real.sqrt 2 := by
    sorry


end min_distance_curve_line_l142_142446


namespace harry_walks_9_dogs_on_thursday_l142_142348

-- Define the number of dogs Harry walks on specific days
def dogs_monday : Nat := 7
def dogs_wednesday : Nat := 7
def dogs_friday : Nat := 7
def dogs_tuesday : Nat := 12

-- Define the payment per dog
def payment_per_dog : Nat := 5

-- Define total weekly earnings
def total_weekly_earnings : Nat := 210

-- Define the number of dogs Harry walks on Thursday
def dogs_thursday : Nat := 9

-- Define the total earnings for Monday, Wednesday, Friday, and Tuesday
def earnings_first_four_days : Nat := (dogs_monday + dogs_wednesday + dogs_friday + dogs_tuesday) * payment_per_dog

-- Now we state the theorem that we need to prove
theorem harry_walks_9_dogs_on_thursday :
  (total_weekly_earnings - earnings_first_four_days) / payment_per_dog = dogs_thursday :=
by
  -- Proof omitted
  sorry

end harry_walks_9_dogs_on_thursday_l142_142348


namespace xiao_dong_actual_jump_distance_l142_142410

-- Conditions are defined here
def standard_jump_distance : ℝ := 4.00
def xiao_dong_recorded_result : ℝ := -0.32

-- Here we structure our problem
theorem xiao_dong_actual_jump_distance :
  standard_jump_distance + xiao_dong_recorded_result = 3.68 :=
by
  sorry

end xiao_dong_actual_jump_distance_l142_142410


namespace prob_at_least_7_consecutive_heads_l142_142949

theorem prob_at_least_7_consecutive_heads : 
  let total_outcomes := 2^10,
      successful_outcomes := 49
  in successful_outcomes / total_outcomes = (49:ℚ) / 1024 :=
by sorry

end prob_at_least_7_consecutive_heads_l142_142949


namespace sqrt_card_subsets_le_sqrt_card_divisors_l142_142695

theorem sqrt_card_subsets_le_sqrt_card_divisors (n : ℕ) (h : 0 < n) :
  ∀ (A B : finset ℕ), (∀ a ∈ A, ∀ b ∈ B, ¬a ∣ b ∧ ¬b ∣ a) →
  (A.card : ℝ) ^ 0.5 + (B.card : ℝ) ^ 0.5 ≤ (nat.divisors n).card ^ 0.5 :=
by sorry

end sqrt_card_subsets_le_sqrt_card_divisors_l142_142695


namespace box_dimensions_sum_l142_142196

theorem box_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 30) 
  (h2 : A * C = 50)
  (h3 : B * C = 90) : 
  A + B + C = (58 * Real.sqrt 15) / 3 :=
sorry

end box_dimensions_sum_l142_142196


namespace project_selection_l142_142595

theorem project_selection (pA pB pC : ℕ) (k : ℕ) :
  pA = 5 ∧ pB = 6 ∧ pC = 4 ∧ k = 4 →
  (∃ x y z, x + y + z = k ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ x ≤ pA - 1 ∧ y ≤ pB ∧ z ≤ pC) →
  ∑ (x in finset.range (pA)), (x.choose 1) * (pB.choose (k - x)) * (pC.choose (k - x)) = 192 := 
by
  intros H1 H2
  sorry

end project_selection_l142_142595


namespace third_day_opponent_l142_142562

def Person : Type := {A, B, C, D : Person}

def plays (x y : Person) : Prop :=
  (x, y) ∈ {(A, C), (C, D)} ∨ (y, x) ∈ {(A, C), (C, D)}

theorem third_day_opponent :
  ∀ (A B C D : Person), 
  (plays A C) →
  (plays C D) →
  (¬ plays A B) →
  (¬ plays B D) →
  (∀ x, (x ≠ A ∧ x ≠ B) → (plays B x))  → 
  (plays B C) := 
by {
  intros A B C D h1 h2 h3 h4 h5,
  sorry
}

end third_day_opponent_l142_142562


namespace hay_mass_calculation_l142_142091

theorem hay_mass_calculation (grass_mass hay_mass : ℝ) (grass_moisture hay_moisture : ℝ)
    (h_grass_moisture : grass_moisture = 0.70) (h_hay_moisture : hay_moisture = 0.16) (h_hay_mass : hay_mass = 1000) :
    grass_mass = 2800 :=
by
  let dry_mass_hay := hay_mass * (1 - hay_moisture)
  have h_dry_mass_hay : dry_mass_hay = 1000 * 0.84, by simp [h_hay_mass, h_hay_moisture]
  let dry_mass_grass := grass_mass * (1 - grass_moisture)
  have h_main_eq : dry_mass_grass = dry_mass_hay, by rw [<-h_grass_moisture, <-h_hay_moisture, <-h_dry_mass_hay]
  have h_grass_mass : grass_mass * 0.30 = 840, by rwa [h_main_eq]
  have result : grass_mass = 840 / 0.30, by rw div_eq_iff (by norm_num : (0.30 : ℝ) ≠ 0); exact h_grass_mass.symm
  exact result

end hay_mass_calculation_l142_142091


namespace simplify_expansion_l142_142254

-- Define the variables and expressions
variable (x : ℝ)

-- The main statement
theorem simplify_expansion : (x + 5) * (4 * x - 12) = 4 * x^2 + 8 * x - 60 :=
by sorry

end simplify_expansion_l142_142254


namespace correspondence1Mapping_correspondence1Function_correspondence2Mapping_correspondence2Function_correspondence3Mapping_correspondence3Function_countMappings_countFunctions_l142_142333

-- Definition of the basic sets
def A : Set := { students : Type* }
def B : Set := { weights : ℝ }
def M : Set := {1, 2, 3, 4}
def N : Set := {2, 4, 6, 8}
def X : Set := ℝ
def Y : Set := { r : ℝ // r >= 0 }

-- Conditions for correspondences
def correspondence1 : A → B := λ (a : A), sorry
def correspondence2 : M → N := λ (m : ℕ), 2 * m
def correspondence3 : X → Y := λ (x : ℝ), ⟨x^3, by sorry⟩

-- Predicate for being a mapping
def isMapping (f : α → β) : Prop := ∀ a : α, ∃! b : β, f a = b

-- Predicate for being a function
def isFunction (f : α → β) : Prop := ∀ a : α, ∃! b : β, f a = b -- Same formal definition as for mapping, no extra condition.

-- Theorem statements
theorem correspondence1Mapping : isMapping correspondence1 := sorry
theorem correspondence1Function : ¬isFunction correspondence1 := sorry

theorem correspondence2Mapping : isMapping correspondence2 := sorry
theorem correspondence2Function : isFunction correspondence2 := sorry

theorem correspondence3Mapping : ¬isMapping correspondence3 := sorry
theorem correspondence3Function : ¬isFunction correspondence3 := sorry

-- Checking the total count for mappings and functions
theorem countMappings : (if correspondence1Mapping then 1 else 0) +
                          (if correspondence2Mapping then 1 else 0) +
                          (if correspondence3Mapping then 1 else 0) = 2 := sorry
                          
theorem countFunctions : (if correspondence1Function then 1 else 0) +
                           (if correspondence2Function then 1 else 0) +
                           (if correspondence3Function then 1 else 0) = 1 := sorry

end correspondence1Mapping_correspondence1Function_correspondence2Mapping_correspondence2Function_correspondence3Mapping_correspondence3Function_countMappings_countFunctions_l142_142333


namespace prime_triplets_l142_142645

theorem prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p ^ q + q ^ p = r ↔ (p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17) := by
  sorry

end prime_triplets_l142_142645


namespace sqrt_product_simplification_l142_142982

theorem sqrt_product_simplification (y : ℝ) : (sqrt (48 * y) * sqrt (3 * y) * sqrt (50 * y)) = 30 * y * sqrt (2 * y) :=
by sorry

end sqrt_product_simplification_l142_142982


namespace exactly_one_passing_at_most_one_signing_l142_142904

noncomputable def probability_of_exactly_one_passing : ℚ :=
  let pA := 1 / 4
  let pB := 1 / 3
  let pC := 1 / 3
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC)

theorem exactly_one_passing : probability_of_exactly_one_passing = 4 / 9 := sorry

noncomputable def probability_of_at_most_one_signing : ℚ :=
  let pA := 1 / 4
  let pB := 1 / 3
  let pC := 1 / 3
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  let pB_and_pC := pB * pC
  (qA * qB_and_pC) + (pA * qB_and_pC)

theorem at_most_one_signing : probability_of_at_most_one_signing = 8 / 9 := sorry

end exactly_one_passing_at_most_one_signing_l142_142904


namespace complement_intersection_l142_142340

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) (hM : M = {1, 3, 5, 7}) (hN : N = {2, 5, 8}) :
  (U \ M) ∩ N = {2, 8} :=
by
  sorry

end complement_intersection_l142_142340


namespace Ann_has_more_cards_than_Anton_l142_142975

-- Definition of Heike's cards
def HeikeCards : Type := ℕ

-- Anton has three times as many cards as Heike
def AntonCards (H : HeikeCards) : ℕ := 3 * H

-- Ann has six times as many cards as Heike
def AnnCards (H : HeikeCards) : ℕ := 6 * H

-- Proof statement
theorem Ann_has_more_cards_than_Anton (H : HeikeCards) :
  AnnCards H - AntonCards H = 3 * H :=
by
  sorry

end Ann_has_more_cards_than_Anton_l142_142975


namespace solve_for_x_l142_142482

theorem solve_for_x (x : ℝ) (h : 4 * 4^x + sqrt (16 * 16^x) = 64) : x = 3 / 2 := 
sorry

end solve_for_x_l142_142482


namespace johns_payment_ratio_is_one_half_l142_142015

-- Define the initial conditions
def num_members := 4
def join_fee_per_person := 4000
def monthly_cost_per_person := 1000
def johns_payment_per_year := 32000

-- Calculate total cost for joining
def total_join_fee := num_members * join_fee_per_person

-- Calculate total monthly cost for a year
def total_monthly_cost := num_members * monthly_cost_per_person * 12

-- Calculate total cost for the first year
def total_cost_for_year := total_join_fee + total_monthly_cost

-- The ratio of John's payment to the total cost
def johns_ratio := johns_payment_per_year / total_cost_for_year

-- The statement to be proved
theorem johns_payment_ratio_is_one_half : johns_ratio = (1 / 2) := by sorry

end johns_payment_ratio_is_one_half_l142_142015


namespace range_of_m_for_distinct_real_roots_l142_142678

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4 * x1 - m = 0 ∧ x2^2 - 4 * x2 - m = 0) ↔ m > -4 :=
by
  sorry

end range_of_m_for_distinct_real_roots_l142_142678


namespace part1_vertex_part1_range_y_part2_expression_l142_142297

-- Part 1: Quadratic function with b=4 and c=3
def part1_questions := let b := 4; let c := 3 in
  (⟨vertex_coordinates : (2, 7)⟩, ⟨range_y : -2 ≤ fun x => -x^2 + b * x + c x ≤ 7 for -1 ≤ x ≤ 3⟩)

-- Part 2: Quadratic function with specific conditions on max values
def part2_conditions := let y := fun x => -x^2 + b * x + c in
  (⟨max_y_on_neg := 2 if x ≤ 0⟩, ⟨max_y_on_pos := 3 if x > 0⟩)

theorem part1_vertex : (part1_questions.vertex_coordinates).fst = ()


theorem part1_range_y : ∀ x, -1 ≤ x ≤ 3 → -2 ≤ -x^2 + 4 * x + 3 ≤ 7 :=
by sorry

theorem part2_expression : ∀ x, b = 2 ∧ c = 2 →

  ∃ (b = 2) ∧ (x ≤ 0 → y x = 2) ∧ (x > 0 → y x = 3)
:=
by sorry

end part1_vertex_part1_range_y_part2_expression_l142_142297


namespace shelter_blocks_l142_142046

noncomputable def volume_original (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def volume_interior (length width height : ℕ) : ℕ :=
  (length - 2) * (width - 2) * (height - 2)

theorem shelter_blocks :
  let V₀ := volume_original 14 12 6 in
  let Vᵢ := volume_interior 14 12 6 in
  V₀ - Vᵢ = 528 := by
  sorry

end shelter_blocks_l142_142046


namespace angle_ABC_is_90_l142_142020

open Real

-- Define the points A, B, and C
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

-- Define the distance formula for 3D space
def dist (p1 p2 : Point3D) : ℝ :=
  sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

-- Given points A, B, and C
def A := { x := -3, y := 1, z := 5 }
def B := { x := -4, y := 0, z := 4 }
def C := { x := -5, y := 0, z := 5 }

-- Calculate distances AB, AC, BC
def AB := dist A B
def AC := dist A C
def BC := dist B C

-- Problem statement
theorem angle_ABC_is_90 :
  ∀ (A B C : Point3D), A = {x := -3, y := 1, z := 5} →
                      B = {x := -4, y := 0, z := 4} →
                      C = {x := -5, y := 0, z := 5} →
                      ∠ ABC = 90 := by
  sorry

end angle_ABC_is_90_l142_142020


namespace math_proof_problem_l142_142329

noncomputable def a_value := 1
noncomputable def b_value := 2

-- Defining the primary conditions
def condition1 (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - 3 * x + 2 > 0) ↔ (x < 1 ∨ x > b)

def condition2 (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - (2 * b - a) * x - 2 * b < 0) ↔ (-1 < x ∧ x < 4)

-- Defining the main goal
theorem math_proof_problem :
  ∃ a b : ℝ, a = a_value ∧ b = b_value ∧ condition1 a b ∧ condition2 a b := 
sorry

end math_proof_problem_l142_142329


namespace maximize_area_difference_l142_142952

noncomputable def circle_center : ℝ × ℝ := (2, 0)
noncomputable def circle_radius : ℝ := 2
noncomputable def point_P : ℝ × ℝ := (1, 1)

theorem maximize_area_difference (line_eq : ℝ × ℝ → Prop)
    (H : line_eq point_P)
    (H_max : ∀ (other_line : ℝ × ℝ → Prop), other_line point_P →
      area_difference (circle_center, circle_radius) line_eq ≥ area_difference (circle_center, circle_radius) other_line) :
    line_eq = (fun p => p.1 - p.2 = 0) :=
sorry

end maximize_area_difference_l142_142952


namespace area_of_quadrilateral_AEDC_l142_142796

theorem area_of_quadrilateral_AEDC :
  ∀ (PE PD DE : ℕ), PE = 3 → PD = 4 → DE = 5 →
    (let AP := 2 * PD,
         CP := 2 * PE,
         area := 1 / 2 * (PE * AP + PD * CP + PE * CP + PD * AP)
     in area = 49) :=
by
  intro PE PD DE
  intros hPE hPD hDE
  let AP := 2 * PD
  let CP := 2 * PE
  let area := 1 / 2 * (PE * AP + PD * CP + PE * CP + PD * AP)
  have h1 : area = 1 / 2 * (3 * 8 + 4 * 6 + 3 * 6 + 4 * 8) := by
    rw [hPE, hPD]
  have h2 : area = 1 / 2 * 98 := by
    simp [h1]
  show area = 49, from by
    simp [h2]
  sorry -- Proof steps skipped

end area_of_quadrilateral_AEDC_l142_142796


namespace cost_of_each_ticket_l142_142668

theorem cost_of_each_ticket (x : ℝ) : 
  500 * x * 0.70 = 4 * 2625 → x = 30 :=
by 
  sorry

end cost_of_each_ticket_l142_142668


namespace money_bounds_l142_142479

variables (c d : ℝ)

theorem money_bounds :
  (7 * c + d > 84) ∧ (5 * c - d = 35) → (c > 9.92 ∧ d > 14.58) :=
by
  intro h
  sorry

end money_bounds_l142_142479


namespace x_intercept_of_line_l142_142245

theorem x_intercept_of_line : ∃ x : ℚ, 4 * x + 7 * 0 = 28 ∧ (x, 0) = (7, 0) :=
by
  use 7
  simp
  norm_num
  sorry

end x_intercept_of_line_l142_142245


namespace number_of_sister_point_pairs_l142_142411

theorem number_of_sister_point_pairs :
  let f : ℝ → ℝ := λ x, if x < 0 then x^2 + 2*x else 2 / real.exp x in
  ∃! (A B : ℝ × ℝ), 
    (f A.1 = A.2 ∧ f B.1 = B.2 ∧ A.1 = -B.1 ∧ A.2 = -B.2 ∧ A ≠ B) :=
sorry

end number_of_sister_point_pairs_l142_142411


namespace club_members_count_l142_142766

theorem club_members_count (m n : ℕ) (h1 : n = 5) (h2 : ∀ i j, i ≠ j → ∃! x, mem x i ∧ mem x j) : m = 10 := 
sorry

end club_members_count_l142_142766


namespace half_less_than_reciprocal_l142_142150

theorem half_less_than_reciprocal : (1 / 2) < 2 :=
by
  linarith

end half_less_than_reciprocal_l142_142150


namespace measure_of_angle_C_length_of_side_c_value_of_sin_2B_minus_C_l142_142421

open Real

noncomputable def angle_C (a b c C : ℝ) (h : √2 * cos C * (a * cos b + b * cos a) + c = 0) : Prop :=
  C = 3 * π / 4

noncomputable def side_c (a b c : ℝ) (C : ℝ) (ha : a = √2) (hb : b = 2) (hC : C = 3 * π / 4) : Prop :=
  c = √10

noncomputable def sin_2B_minus_C (a b c C B : ℝ) (ha : a = √2) (hb : b = 2) (hC : C = 3 * π / 4) (hc : c = √10) : Prop :=
  sin(2 * B - C) = -7 * √2 / 10

variables (a b c C B : ℝ)
variable (h : √2 * cos C * (a * cos B + b * cos A) + c = 0)
variable (ha : a = √2)
variable (hb : b = 2)
variable (hC : C = 3 * π / 4)
variable (hc : c = √10)

theorem measure_of_angle_C : angle_C a b c C h := by
  sorry

theorem length_of_side_c : side_c a b c C ha hb hC := by
  sorry

theorem value_of_sin_2B_minus_C : sin_2B_minus_C a b c C B ha hb hC hc := by
  sorry

end measure_of_angle_C_length_of_side_c_value_of_sin_2B_minus_C_l142_142421


namespace natural_numbers_divisible_by_7_between_200_400_l142_142362

theorem natural_numbers_divisible_by_7_between_200_400 : 
  { n : ℕ | 200 <= n ∧ n <= 400 ∧ n % 7 = 0 }.to_finset.card = 29 := 
  sorry

end natural_numbers_divisible_by_7_between_200_400_l142_142362


namespace hyperbola_distance_property_l142_142759

theorem hyperbola_distance_property (P : ℝ × ℝ)
  (hP_on_hyperbola : (P.1 ^ 2 / 16) - (P.2 ^ 2 / 9) = 1)
  (h_dist_15 : dist P (5, 0) = 15) :
  dist P (-5, 0) = 7 ∨ dist P (-5, 0) = 23 := 
sorry

end hyperbola_distance_property_l142_142759


namespace g_x0_even_g_x0_odd_interval_monotonic_increase_l142_142721

open Real

def f (x : ℝ) : ℝ := cos (x + π / 12) ^ 2
def g (x : ℝ) : ℝ := 1 + 1 / 2 * sin (2 * x)

-- (I) Prove g(x0) = 3/4 or 5/4 depending on the parity of k
theorem g_x0_even {x0 : ℝ} (k : ℤ) (hx0 : 2 * x0 = k * π - π / 6) (hk_even : Even k) :
  g x0 = 3 / 4 := 
sorry

theorem g_x0_odd {x0 : ℝ} (k : ℤ) (hx0 : 2 * x0 = k * π - π / 6) (hk_odd : ¬ Even k) :
  g x0 = 5 / 4 := 
sorry

def h (x : ℝ) : ℝ := f x + g x

-- (II) Prove the interval of monotonic increase for h(x)
theorem interval_monotonic_increase (k : ℤ) :
  ∀ x, (k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12) → 
  ∃ ε > 0, ∀ y, (x ≤ y ∧ y ≤ x + ε) → h y > h x :=
sorry

end g_x0_even_g_x0_odd_interval_monotonic_increase_l142_142721


namespace distribute_balls_l142_142071

theorem distribute_balls :
  (∑ (x1 x2 x3 x4 : ℕ) in {x1 | x1 % 2 = 0}.to_finset × {x2 | x2 % 2 = 0}.to_finset × 
                           {x3 | x3 % 2 = 1}.to_finset × finset.range 8,
                           x1 + x2 + x3 + x4 = 7) = 2080 := 
sorry

end distribute_balls_l142_142071


namespace jeremy_travel_time_l142_142428

theorem jeremy_travel_time:
  (distance : ℕ) 
  (average_speed : ℕ)
  (rest_interval : ℕ) 
  (rest_duration : ℕ) 
  (fuel_efficiency : ℕ) 
  (tank_capacity : ℕ) 
  (refuel_time : ℕ)  
  (initial_fuel : ℕ)
  (total_time : ℕ)
  (H1 : distance = 600) -- Jeremy is driving 600 miles to visit his parents
  (H2 : average_speed = 50) -- His average speed for this trip is 50 miles per hour
  (H3 : rest_interval = 2) -- Every two hours of driving, he makes a rest stop
  (H4 : rest_duration = 15) -- Each rest stop lasts 15 minutes
  (H5 : fuel_efficiency = 18) -- His car gets 18 miles per gallon of gas
  (H6 : tank_capacity = 15) -- His gas indicator comes on when he's used 15 gallons
  (H7 : refuel_time = 10) -- It takes 10 minutes to refuel
  (H8 : initial_fuel = 1) -- He starts out with a full tank of gas
  (H9 : total_time = 815) -- The total time Jeremy takes to get to his parents' house is 815 minutes
  : total_time = 815 := 
  by
  sorry

end jeremy_travel_time_l142_142428


namespace average_weight_new_students_l142_142081

theorem average_weight_new_students :
  ∀ (avg_weight_old_students : ℝ) (num_old_students : ℕ)
    (avg_weight_new : ℝ) (num_new_students : ℕ)
    (new_avg_weight : ℝ),
  avg_weight_old_students = 65 ∧ num_old_students = 100 ∧
  num_new_students = 10 ∧ new_avg_weight = 64.6 →
  let total_weight_old_students := num_old_students * avg_weight_old_students in
  let total_weight_new_students := num_old_students * new_avg_weight + num_new_students * new_avg_weight in
  let total_weight_old_new := total_weight_old_students + total_weight_new_students in
  let avg_new_students := (total_weight_old_new - total_weight_old_students.toReal) / num_new_students in
  avg_new_students = 60.6  :=
begin
  intros avg_weight_old_students num_old_students avg_weight_new num_new_students new_avg_weight h,
  rcases h with ⟨h1, h2, h3, h4⟩,
  rw [h1, h2, h3, h4],
  sorry,
end

end average_weight_new_students_l142_142081


namespace candy_comparison_l142_142220

variable (skittles_bryan : ℕ)
variable (gummy_bears_bryan : ℕ)
variable (chocolate_bars_bryan : ℕ)
variable (mms_ben : ℕ)
variable (jelly_beans_ben : ℕ)
variable (lollipops_ben : ℕ)

def bryan_total_candies := skittles_bryan + gummy_bears_bryan + chocolate_bars_bryan
def ben_total_candies := mms_ben + jelly_beans_ben + lollipops_ben

def difference_skittles_mms := skittles_bryan - mms_ben
def difference_gummy_jelly := jelly_beans_ben - gummy_bears_bryan
def difference_choco_lollipops := chocolate_bars_bryan - lollipops_ben

def sum_of_differences := difference_skittles_mms + difference_gummy_jelly + difference_choco_lollipops

theorem candy_comparison
  (h_bryan_skittles : skittles_bryan = 50)
  (h_bryan_gummy_bears : gummy_bears_bryan = 25)
  (h_bryan_choco_bars : chocolate_bars_bryan = 15)
  (h_ben_mms : mms_ben = 20)
  (h_ben_jelly_beans : jelly_beans_ben = 30)
  (h_ben_lollipops : lollipops_ben = 10) :
  bryan_total_candies = 90 ∧
  ben_total_candies = 60 ∧
  bryan_total_candies > ben_total_candies ∧
  difference_skittles_mms = 30 ∧
  difference_gummy_jelly = 5 ∧
  difference_choco_lollipops = 5 ∧
  sum_of_differences = 40 := by
  sorry

end candy_comparison_l142_142220


namespace P_2_value_Pn_recursive_Pn_explicit_l142_142173

noncomputable def P (n : ℕ) : ℚ :=
  if n = 1 then 1 / 2
  else if n = 2 then 7 / 15
  else (P (n - 1) * (-4 / 15)) + 9 / 19

theorem P_2_value : P 2 = 7 / 15 :=
by {
  -- proof goes here
  sorry
}

theorem Pn_recursive (n : ℕ) (hn : n ≥ 2) : P n = (-4 / 15) * P (n - 1) + 3 / 5 :=
by {
  -- proof goes here
  sorry
}

theorem Pn_explicit (n : ℕ) : P n = (1 / 38) * ((-4 / 15) ^ (n - 1)) + 9 / 19 :=
by {
  -- proof goes here
  sorry
}

end P_2_value_Pn_recursive_Pn_explicit_l142_142173


namespace more_silverfish_than_goldfish_l142_142406

variable (n G S R : ℕ)

-- Condition 1: If the cat eats all the goldfish, the number of remaining fish is \(\frac{2}{3}\)n - 1
def condition1 := n - G = (2 * n) / 3 - 1

-- Condition 2: If the cat eats all the redfish, the number of remaining fish is \(\frac{2}{3}\)n + 4
def condition2 := n - R = (2 * n) / 3 + 4

-- The goal: Silverfish are more numerous than goldfish by 2
theorem more_silverfish_than_goldfish (h1 : condition1 n G) (h2 : condition2 n R) :
  S = (n / 3) + 3 → G = (n / 3) + 1 → S - G = 2 :=
by
  sorry

end more_silverfish_than_goldfish_l142_142406


namespace perp_condition_necessary_but_not_sufficient_l142_142164

theorem perp_condition_necessary_but_not_sufficient 
  (l : Type) (α : Type) [Plane α] [Line l] 
  (h : ∀ (l' : Line), l' ∈ α → perp l l') : 
  (necessary_condition (∀ (l' : Line), l' ∈ α → perp l l')) ∧ ¬ (sufficient_condition (∀ (l' : Line), l' ∈ α → perp l l')) :=
sorry

end perp_condition_necessary_but_not_sufficient_l142_142164


namespace algebra_expression_bound_l142_142758

theorem algebra_expression_bound (x y m : ℝ) 
  (h1 : x + y + m = 6) 
  (h2 : 3 * x - y + m = 4) : 
  (-2 * x * y + 1) ≤ 3 / 2 := 
by 
  sorry

end algebra_expression_bound_l142_142758


namespace stacy_savings_for_3_pairs_l142_142483

-- Define the cost per pair of shorts
def cost_per_pair : ℕ := 10

-- Define the discount percentage as a decimal
def discount_percentage : ℝ := 0.1

-- Function to calculate the total cost without discount for n pairs
def total_cost_without_discount (n : ℕ) : ℕ := cost_per_pair * n

-- Function to calculate the total cost with discount for n pairs
noncomputable def total_cost_with_discount (n : ℕ) : ℝ :=
  if n >= 3 then
    let discount := discount_percentage * (cost_per_pair * n : ℝ)
    (cost_per_pair * n : ℝ) - discount
  else
    cost_per_pair * n

-- Function to calculate the savings for buying n pairs at once compared to individually
noncomputable def savings (n : ℕ) : ℝ :=
  (total_cost_without_discount n : ℝ) - total_cost_with_discount n

-- Proof statement
theorem stacy_savings_for_3_pairs : savings 3 = 3 := by
  sorry

end stacy_savings_for_3_pairs_l142_142483


namespace bottles_needed_l142_142166

theorem bottles_needed (runners : ℕ) (bottles_needed_per_runner : ℕ) (bottles_available : ℕ)
  (h_runners : runners = 14)
  (h_bottles_needed_per_runner : bottles_needed_per_runner = 5)
  (h_bottles_available : bottles_available = 68) :
  runners * bottles_needed_per_runner - bottles_available = 2 :=
by
  sorry

end bottles_needed_l142_142166


namespace blue_to_red_ratio_l142_142233

theorem blue_to_red_ratio (r1 r2 : ℝ) (h1 : r1 = 1) (h2 : r2 = 3) :
  let A_red := π * r1^2,
      A_large := π * r2^2,
      A_blue := A_large - A_red
  in A_blue / A_red = 8 :=
by
  sorry

end blue_to_red_ratio_l142_142233


namespace limit_to_infinity_zero_l142_142456

variable (f : ℝ → ℝ)

theorem limit_to_infinity_zero (h_continuous : Continuous f)
  (h_alpha : ∀ (α : ℝ), α > 0 → Filter.Tendsto (fun n : ℕ => f (n * α)) Filter.atTop (nhds 0)) :
  Filter.Tendsto f Filter.atTop (nhds 0) :=
sorry

end limit_to_infinity_zero_l142_142456


namespace probability_blue_face_up_l142_142914

-- Definitions of the conditions
def dodecahedron_faces : ℕ := 12
def blue_faces : ℕ := 10
def red_faces : ℕ := 2

-- Expected probability
def probability_blue_face : ℚ := 5 / 6

-- Theorem to prove the probability of rolling a blue face on a dodecahedron
theorem probability_blue_face_up (total_faces blue_count red_count : ℕ)
    (h1 : total_faces = dodecahedron_faces)
    (h2 : blue_count = blue_faces)
    (h3 : red_count = red_faces) :
  blue_count / total_faces = probability_blue_face :=
by sorry

end probability_blue_face_up_l142_142914


namespace isosceles_triangle_area_l142_142618

theorem isosceles_triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 6) :
  let s := (a + b + c) / 2,
      h := real.sqrt (a * a - (c / 2) * (c / 2))
  in a = b → c = 6 → (1 / 2) * c * h = 12 := 
by
  intros _
  sorry

end isosceles_triangle_area_l142_142618


namespace sin_gt_cos_range_l142_142415

theorem sin_gt_cos_range (x : ℝ) : 
  0 < x ∧ x < 2 * Real.pi → (Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4)) := by
  sorry

end sin_gt_cos_range_l142_142415


namespace parallel_slope_l142_142871

theorem parallel_slope (k : ℝ) : 
  let A := (-6, 0)
  let B := (0, -6)
  let X := (0, 10)
  let Y := (18, k)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_XY := (Y.2 - X.2) / (Y.1 - X.1)
  (slope_AB = slope_XY) → k = -8 :=
by 
  let A := (-6 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, -6 : ℝ)
  let X := (0 : ℝ, 10 : ℝ)
  let Y := (18 : ℝ, k)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_XY := (Y.2 - X.2) / (Y.1 - X.1)
  have h1 : slope_AB = -1 := by simp [A, B]
  have h2 : slope_XY = (k - 10) / 18 := by simp [X, Y]
  assumption
sorry

end parallel_slope_l142_142871


namespace greatest_integer_S_l142_142728

-- Definitions based on the given problem
def a (n : ℕ) : ℝ := real.sqrt (1 + 1 / (n : ℝ)^2 + 1 / (n + 1 : ℝ)^2)

def S (n : ℕ) := ∑ k in finset.range n, a (k + 1)

-- The theorem to prove that the greatest integer less than or equal to Sn is n
theorem greatest_integer_S (n : ℕ) : ⌊S n⌋ = n := sorry

end greatest_integer_S_l142_142728


namespace probability_YQ_greater_8_sqrt_2_proof_l142_142797

noncomputable def probability_YQ_greater_8_sqrt_2
  (XYZ : Triangle)
  (angle_YXZ : XYZ.angle Y X Z = 90)
  (angle_XYZ : XYZ.angle X Y Z = 60)
  (XY_length : XYZ.side_length X Y = 16)
  (P : Point)
  (P_in_XYZ : XYZ.contains P)
  (Q : Point)
  (Q_on_XZ : Line_through Y P ∩ Line_through X Z = {Q})
  : ℝ :=
  sorry

theorem probability_YQ_greater_8_sqrt_2_proof
  (XYZ : Triangle)
  (angle_YXZ : XYZ.angle Y X Z = 90)
  (angle_XYZ : XYZ.angle X Y Z = 60)
  (XY_length : XYZ.side_length X Y = 16)
  (P : Point)
  (P_in_XYZ : XYZ.contains P)
  (Q : Point)
  (Q_on_XZ : Line_through Y P ∩ Line_through X Z = {Q}) :
  probability_YQ_greater_8_sqrt_2 XYZ angle_YXZ angle_XYZ XY_length P P_in_XYZ Q Q_on_XZ = (3 - real.sqrt 3) / 3 :=
begin
  sorry
end

end probability_YQ_greater_8_sqrt_2_proof_l142_142797


namespace probability_P_area_PBC_l142_142955
noncomputable theory

def point := (ℝ × ℝ)

def A : point := (0, 6)
def B : point := (9, 0)
def C : point := (0, 0)

def area_of_triangle (a b c : point) : ℝ :=
  0.5 * |(fst b - fst a) * (snd c - snd a) - (fst c - fst a) * (snd b - snd a) |

def probability_area_less_than_one_third (P : point) (A B C : point) : ℝ :=
  if snd P < 2 then 1/3 else 0

theorem probability_P_area_PBC (A B C : point) (P : point) (hP : snd P < 6 ∧ fst P < 9 ∧ 0 ≤ snd P):
  probability_area_less_than_one_third P A B C = 1/3 :=
sorry

end probability_P_area_PBC_l142_142955


namespace am_perp_h₁h₂_l142_142833

noncomputable def midpt (A B : Point) : Point := sorry

structure Triangle :=
  (A B C : Point)

structure PointProp :=
  (is_perpendicular : Point → Point → Point → Prop)
  (is_midpoint : Point → Point → Point → Prop)

variables {α : Type*} [plane_geometry α]

-- Definitions of Points P, Q, E, F, M
variables (ABC : Triangle)
variables (P Q E F M H₁ H₂ : Point)
variables (prop : PointProp)

-- Conditions extracted and formulated
axiom cond1 : prop.is_midpoint (midpt ABC.B ABC.C) P Q
axiom cond2 : prop.is_perpendicular P ABC.B ABC.C E
axiom cond3 : prop.is_perpendicular Q ABC.B ABC.C F
axiom cond4 : prop.is_perpendicular P F M
axiom cond5 : prop.is_perpendicular Q E M
axiom cond6 : orthocenter (Triangle.mk ABC.B F P) H₁
axiom cond7 : orthocenter (Triangle.mk ABC.C E Q) H₂

-- The theorem statement
theorem am_perp_h₁h₂ : prop.is_perpendicular (Triangle.mk ABC.A M) H₁ H₂ := sorry

end am_perp_h₁h₂_l142_142833


namespace distance_between_circle_centers_eq_l142_142449

theorem distance_between_circle_centers_eq : 
  let DE := 17
  let DF := 15
  let EF := 8
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := area / s
  let R := area / (s - EF)
  let DI := Real.sqrt (DF^2 - r^2)
  let DE := Real.sqrt (DF^2 + R^2)
  IE = DE - DI := 5 * Real.sqrt 10 - 6 * Real.sqrt 6 :=
begin
  sorry
end

end distance_between_circle_centers_eq_l142_142449


namespace find_p_value_l142_142622

variable (p q : ℝ)

-- Conditions in Lean definitions:
def quadratic_eq (p q : ℝ) : Prop := ∀ x : ℝ, x^2 + p * x + q = 0
def positive (p q : ℝ) : Prop := p > 0 ∧ q > 0
def roots_differ_by_two (p q : ℝ) : Prop := 
  let root1 := (-p + Real.sqrt (p^2 - 4 * q)) / 2 
  let root2 := (-p - Real.sqrt (p^2 - 4 * q)) / 2 
  Real.abs (root1 - root2) = 2

-- Lean statement to prove:
theorem find_p_value (h1 : quadratic_eq p q) (h2 : positive p q) (h3 : roots_differ_by_two p q) : 
  p = Real.sqrt (4 * q + 4) :=
sorry

end find_p_value_l142_142622


namespace max_cosine_prod_l142_142290

open Real

noncomputable def cosine_product_max_value (n : ℕ) (a : ℝ) (α : Fin n → ℝ) :=
  ∀ (i : Fin n), 0 < α i ∧ α i < π / 2 → 
  (∏ i, sin (α i)) = a →
  (∏ i, cos (α i)) ≤ sqrt ( a * (1 / (root n a) - (root n a))^n )

theorem max_cosine_prod (n : ℕ) (h : 2 ≤ n) (a : ℝ) (α : Fin n → ℝ) :
  cosine_product_max_value n a α := by
  sorry

end max_cosine_prod_l142_142290


namespace count_valid_permutations_l142_142777

def no_four_consecutive_increasing (l : List ℕ) : Prop :=
  ∀ i, i + 3 < l.length → ¬ ((l.get i < l.get (i+1)) ∧ (l.get (i+1) < l.get (i+2)) ∧ (l.get (i+2) < l.get (i+3)))

def no_four_consecutive_decreasing (l : List ℕ) : Prop :=
  ∀ i, i + 3 < l.length → ¬ ((l.get i > l.get (i+1)) ∧ (l.get (i+1) > l.get (i+2)) ∧ (l.get (i+2) > l.get (i+3)))

def satisfies_conditions (l : List ℕ) : Prop :=
  no_four_consecutive_increasing l ∧ no_four_consecutive_decreasing l

theorem count_valid_permutations : (List.permutations [1, 2, 3, 4, 5, 6]).count satisfies_conditions = 180 := 
sorry

end count_valid_permutations_l142_142777


namespace eval_integral_eq_value_l142_142639

noncomputable def eval_definite_integral : ℝ :=
  ∫ (u : ℝ) in -1..1, (2 * u^332 + u^998 + 4 * u^1664 * Real.sin (u^691)) / (1 + u^666)

theorem eval_integral_eq_value :
  eval_definite_integral = (2 / 333) * (1 + (Real.pi / 4)) :=
by
  sorry

end eval_integral_eq_value_l142_142639


namespace incenter_parallel_l142_142486

noncomputable def angle_bisector_intersect (ABC : Triangle) : Prop :=
  ∃ A1 C1 P I A0 C0,
    angle_bisector_intersect(ABC.A, A1) ∧
    angle_bisector_intersect(ABC.C, C1) ∧
    circumscribed_circle_intersect(ABC, A0) ∧
    circumscribed_circle_intersect(ABC, C0) ∧
    intersect_lines (line_through_points A1 C1) (line_through_points A0 C0) P ∧
    incenter(ABC, I) ∧
    parallel(segment_through_points I P, line_through_points ABC.A ABC.C)

theorem incenter_parallel (ABC : Triangle) :
  angle_bisector_intersect (ABC) :=
sorry

end incenter_parallel_l142_142486


namespace rice_wheat_ratio_l142_142484

theorem rice_wheat_ratio (total_shi : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) (total_sample : ℕ) : 
  total_shi = 1512 ∧ sample_size = 216 ∧ wheat_in_sample = 27 ∧ total_sample = 1512 * (wheat_in_sample / sample_size) →
  total_sample = 189 :=
by
  intros h
  sorry

end rice_wheat_ratio_l142_142484


namespace number_of_divisibles_by_7_l142_142352

theorem number_of_divisibles_by_7 (a b : ℕ) (h1 : a = 200) (h2 : b = 400) : 
  (nat.card {n | a ≤ 7 * n ∧ 7 * n ≤ b}) = 29 := 
by sorry

end number_of_divisibles_by_7_l142_142352


namespace int_squares_l142_142257

theorem int_squares (n : ℕ) (h : ∃ k : ℕ, n^4 - n^3 + 3 * n^2 + 5 = k^2) : n = 2 := by
  sorry

end int_squares_l142_142257


namespace perfect_squares_ending_in_4_5_6_less_than_2000_l142_142746

theorem perfect_squares_ending_in_4_5_6_less_than_2000 :
  let squares := { n : ℕ | n * n < 2000 ∧ (n * n % 10 = 4 ∨ n * n % 10 = 5 ∨ n * n % 10 = 6) } in
  squares.card = 23 :=
by
  sorry

end perfect_squares_ending_in_4_5_6_less_than_2000_l142_142746


namespace num_divisible_by_7_200_to_400_l142_142368

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l142_142368


namespace max_n_non_negative_sum_l142_142700

-- Definition of an arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * a 1) + (n * (n - 1) / 2) * (a 2 - a 1)

-- Given conditions
variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (d : ℤ)
variable (a1_positive : 0 < a 1)
variable (ratio_condition : (a 6) / (a 5) = 9/11)

-- Prove the maximum value of n
theorem max_n_non_negative_sum :
  sum_of_first_n_terms a S →
  arithmetic_sequence a d →
  (∀ n, 0 ≤ S n) →
  ∃ n : ℕ, n ≤ 20 :=
begin
  intros hS ha hS_nonneg,
  use 20,
  sorry -- Proof to be completed
end

end max_n_non_negative_sum_l142_142700


namespace trig_identity_l142_142925

theorem trig_identity (α : ℝ) (h1 : sin (2 * α) ≠ 0) (h2 : cos (2 * α) ≠ 0) :
    (sin (6 * α) / sin (2 * α)) + (cos (6 * α - π) / cos (2 * α)) = 2 :=
by
  sorry

end trig_identity_l142_142925


namespace delacroix_band_max_members_l142_142077

theorem delacroix_band_max_members :
  ∃ n : ℕ, 30 * n % 28 = 6 ∧ 30 * n < 1200 ∧ 30 * n = 930 :=
by
  sorry

end delacroix_band_max_members_l142_142077


namespace polynomial_p_value_l142_142751

noncomputable def polynomial_p : ℕ → Polynomial ℚ :=
sorry

theorem polynomial_p_value (n : ℕ) (h : ∀ k, k ≤ n → polynomial_p n k = (k : ℚ) / (k + 1)) :
  polynomial_p n (n + 1) = if even n then 1 else (n:ℚ) / (n + 2) :=
sorry

end polynomial_p_value_l142_142751


namespace k_value_tangent_l142_142666

-- Defining the equations
def line (k : ℝ) (x y : ℝ) : Prop := 3 * x + 5 * y + k = 0
def parabola (x y : ℝ) : Prop := y^2 = 24 * x

-- The main theorem stating that k must be 50 for the line to be tangent to the parabola
theorem k_value_tangent (k : ℝ) : (∀ x y : ℝ, line k x y → parabola x y → True) → k = 50 :=
by 
  -- The proof can be constructed based on the discriminant condition provided in the problem
  sorry

end k_value_tangent_l142_142666


namespace total_cost_of_digging_well_l142_142648

noncomputable def cost_of_digging (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := Real.pi * (radius^2) * depth
  volume * cost_per_cubic_meter

theorem total_cost_of_digging_well :
  cost_of_digging 14 3 15 = 1484.4 :=
by
  sorry

end total_cost_of_digging_well_l142_142648


namespace find_number_l142_142186

variable (x : ℝ)

theorem find_number (h : 20 * (x / 5) = 40) : x = 10 := by
  sorry

end find_number_l142_142186


namespace square_has_2_distinct_circles_l142_142023

def is_diameter (S : set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ S ∧ B ∈ S ∧ ∃ C D, C ∈ S ∧ D ∈ S ∧ S = {A, B, C, D} ∧ 
  dist A B = dist C D

def distinct_circles (S : set (ℝ × ℝ)) : Prop :=
  ∃ A B C D : ℝ × ℝ,
  A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧
  S = {A, B, C, D} ∧
  (∃ O R, ∀ E F ∈ S, E ≠ F → dist E F = 2 * R ∨ dist E F = R * sqrt 2) →
  2

theorem square_has_2_distinct_circles (S : set (ℝ × ℝ)) (hS : ∃ A B C D, A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ S = {A, B, C, D} ∧ is_square A B C D) :
  distinct_circles S :=
sorry

end square_has_2_distinct_circles_l142_142023


namespace max_area_of_triangle_rotating_lines_l142_142784

/- Definitions -/
def point (x y : ℝ) : Type := (x, y)
def slope (m : ℝ) (p : Type) := ∃ x y, p = (x, y) ∧ y = m * x

def A := point 0 0
def B := point 14 0
def C := point 25 0

def line_l_A := ∀ x, slope 2 (point x 0)
def line_l_B := ∀ x, x = 14
def line_l_C := ∀ x, slope (-2) (point x (2 * x + 25))

/- Hypotheses -/
def rotate_at_same_rate (p₁ p₂ p₃ : Type) (rate : ℝ) : Prop := sorry

/- Theorem -/
theorem max_area_of_triangle_rotating_lines : 
  rotate_at_same_rate A B C (π/180) → 
  ∃ area, area = 158.5 :=
by sorry

end max_area_of_triangle_rotating_lines_l142_142784


namespace Jason_total_money_l142_142427

theorem Jason_total_money :
  let quarter_value := 0.25
  let dime_value := 0.10
  let nickel_value := 0.05
  let initial_quarters := 49
  let initial_dimes := 32
  let initial_nickels := 18
  let additional_quarters := 25
  let additional_dimes := 15
  let additional_nickels := 10
  let initial_money := initial_quarters * quarter_value + initial_dimes * dime_value + initial_nickels * nickel_value
  let additional_money := additional_quarters * quarter_value + additional_dimes * dime_value + additional_nickels * nickel_value
  initial_money + additional_money = 24.60 :=
by
  sorry

end Jason_total_money_l142_142427


namespace track_length_is_500_l142_142218

-- Given Conditions
variables (B_speed S_speed : ℝ) -- Brenda and Sally's speeds
variable (track_length : ℝ) -- Length of the track
-- Brenda and Sally start at diametrically opposite points and run in opposite directions
-- They meet for the first time after Brenda has run 100 meters
variable h_meet_first : B_speed * 100 / (B_speed + S_speed) = track_length / 2
-- They meet for the second time after Sally has run 150 meters past their first meeting point
variable h_meet_second : S_speed * 150 / (B_speed + S_speed) + track_length / 2 = track_length

-- Theorem to be proved: the length of the track is 500 meters
theorem track_length_is_500 (h1 : B_speed ≠ 0) (h2 : S_speed ≠ 0) :
  track_length = 500 := 
sorry

end track_length_is_500_l142_142218


namespace max_vertex_sum_l142_142281

-- Define the variables and the conditions

variables (a : ℤ) (T : ℤ)
variables (hT : T ≠ 0)

-- Define the points that the parabola passes through
def passes_through_A := (0 : ℝ) ∧ (0 : ℝ)
def passes_through_B := (4 * T : ℝ) ∧ (0 : ℝ)
def passes_through_C := (4 * T + 2 : ℝ) ∧ (32 : ℝ)

-- Define the sum of the coordinates of the vertex point
def vertex_sum := 2 * T - 2 * a * T * T

-- Statement to prove that the maximum value of vertex_sum is 14
theorem max_vertex_sum (hA : passes_through_A) (hB : passes_through_B) (hC : passes_through_C) : 
  ∃ a T, is_maximum (vertex_sum a T hT) 14 :=
sorry

end max_vertex_sum_l142_142281


namespace rectangle_fraction_of_circle_radius_l142_142496

theorem rectangle_fraction_of_circle_radius 
  (area_rectangle : ℕ := 220)
  (breadth_rectangle : ℕ := 10)
  (area_square : ℕ := 3025) 
  (r : ℕ := nat.sqrt area_square)
  (L : ℕ := area_rectangle / breadth_rectangle) : 
  ((L : ℚ) / (r : ℚ)) = (2/5 : ℚ) :=
by 
  sorry

end rectangle_fraction_of_circle_radius_l142_142496


namespace weight_of_larger_pencil_l142_142954

-- Define the weights and the scale factor
def smaller_pencil_weight : Real := 10
def scale_factor : Real := 3
def larger_pencil_weight : Real := scale_factor^3 * smaller_pencil_weight

-- The theorem statement
theorem weight_of_larger_pencil :
  ∃ w : Real, w = 270 :=
by
  use larger_pencil_weight
  have : larger_pencil_weight = 270 := by
  {
    rw [pow_succ, pow_succ, mul_pow],
    norm_num,
  }
  exact this

end weight_of_larger_pencil_l142_142954


namespace broken_seashells_count_l142_142109

def total_seashells : ℕ := 7
def unbroken_seashells : ℕ := 3

theorem broken_seashells_count : (total_seashells - unbroken_seashells) = 4 := by
  sorry

end broken_seashells_count_l142_142109


namespace workers_together_complete_work_in_14_days_l142_142604

noncomputable def efficiency (Wq : ℝ) := 1.4 * Wq

def work_done_in_one_day_p (Wp : ℝ) := Wp = 1 / 24

noncomputable def work_done_in_one_day_q (Wq : ℝ) := Wq = (1 / 24) / 1.4

noncomputable def combined_work_per_day (Wp Wq : ℝ) := Wp + Wq

noncomputable def days_to_complete_work (W : ℝ) := 1 / W

theorem workers_together_complete_work_in_14_days (Wp Wq : ℝ) 
  (h1 : Wp = efficiency Wq)
  (h2 : work_done_in_one_day_p Wp)
  (h3 : work_done_in_one_day_q Wq) :
  days_to_complete_work (combined_work_per_day Wp Wq) = 14 := 
sorry

end workers_together_complete_work_in_14_days_l142_142604


namespace probability_entirely_black_after_transformation_l142_142566

noncomputable def unit_square_color : Type := bool -- true for black, false for white

def grid := matrix (fin 4) (fin 4) unit_square_color

def rotate_180 (g : grid) : grid :=
  λ i j, g (3 - i) (3 - j)

def new_color (g : grid) (i j : fin 4) : unit_square_color :=
  if g i j = false ∧ rotate_180 g i j = true then true else g i j

def transform_grid (g : grid) : grid :=
  λ i j, new_color g i j

open_locale classical
noncomputable def probability_entirely_black : ℚ :=
  finset.univ.prod $ λ _ : fin 4, (finset.univ.prod $ λ _ : fin 4, (0.5 : ℚ))

theorem probability_entirely_black_after_transformation :
  probability_entirely_black = 1 / 1024 :=
sorry

end probability_entirely_black_after_transformation_l142_142566


namespace num_divisible_by_7_200_to_400_l142_142370

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l142_142370


namespace equation_of_line_AB_l142_142342

open Real

-- Define the equations of two circles.
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 20

-- Define the line equation to be proven.
def lineAB (x y : ℝ) : Prop := x + 3y = 0

-- Statement of the proof problem.
theorem equation_of_line_AB :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y → lineAB x y :=
by
  sorry

end equation_of_line_AB_l142_142342


namespace price_per_rose_l142_142610

variable (initial_roses remaining_roses : ℕ)
variable (total_earnings : ℕ)

theorem price_per_rose
  (h1 : initial_roses = 13)
  (h2 : remaining_roses = 4)
  (h3 : total_earnings = 36) :
  total_earnings / (initial_roses - remaining_roses) = 4 :=
by
  rw [h1, h2, h3]
  sorry

end price_per_rose_l142_142610


namespace john_total_cost_l142_142945

-- Define the costs and usage details
def base_cost : ℕ := 25
def cost_per_text_cent : ℕ := 10
def cost_per_extra_minute_cent : ℕ := 15
def included_hours : ℕ := 20
def texts_sent : ℕ := 150
def hours_talked : ℕ := 22

-- Prove that the total cost John had to pay is $58
def total_cost_john : ℕ :=
  let base_cost_dollars := base_cost
  let text_cost_dollars := (texts_sent * cost_per_text_cent) / 100
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_cost_dollars := (extra_minutes * cost_per_extra_minute_cent) / 100
  base_cost_dollars + text_cost_dollars + extra_minutes_cost_dollars

theorem john_total_cost (h1 : base_cost = 25)
                        (h2 : cost_per_text_cent = 10)
                        (h3 : cost_per_extra_minute_cent = 15)
                        (h4 : included_hours = 20)
                        (h5 : texts_sent = 150)
                        (h6 : hours_talked = 22) : 
  total_cost_john = 58 := by
  sorry

end john_total_cost_l142_142945


namespace max_integer_a_for_real_roots_l142_142679

theorem max_integer_a_for_real_roots (a : ℤ) :
  (((a - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1) → a ≤ 0 ∧ (∀ b : ℤ, ((b - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1 → b ≤ 0) :=
sorry

end max_integer_a_for_real_roots_l142_142679


namespace largest_n_base_conversion_l142_142862

theorem largest_n_base_conversion :
  ∃ (n : ℕ) (A B C : ℕ), (0 ≤ A ∧ A < 8) ∧ (0 ≤ B ∧ B < 8) ∧ (0 ≤ C ∧ C < 8) ∧
    (C % 2 = 0) ∧ (n = 64 * A + 8 * B + C) ∧ (n = 81 * C + 9 * B + A) ∧
    ∀ (m : ℕ) (A' B' C' : ℕ), (0 ≤ A' ∧ A' < 8) ∧ (0 ≤ B' ∧ B' < 8) ∧ (0 ≤ C' ∧ C' < 8) ∧
      (C' % 2 = 0) ∧ (m = 64 * A' + 8 * B' + C') ∧ (m = 81 * C' + 9 * B' + A') →
    m ≤ n :=
begin
  sorry
end

end largest_n_base_conversion_l142_142862


namespace increasing_sequence_l142_142204

-- Define all given sequences
def seqA (n : ℕ) : ℕ := 1 - n
def seqB (n : ℕ) : ℝ := 1 / (4 : ℝ)^n
def seqC (n : ℕ) : ℕ := 2 * n^2 - 5 * n + 1
def seqD (n : ℕ) : ℕ :=
  if n ≤ 2 then n + 3 else 2^(n - 1)

-- Define the property of being increasing
def is_increasing (seq : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, seq (n + 1) > seq n

-- Theorem to prove: sequence C is the only increasing sequence
theorem increasing_sequence :
  is_increasing seqC ∧ ¬ is_increasing seqA ∧ ¬ is_increasing seqB ∧ ¬ is_increasing seqD :=
by
  sorry

end increasing_sequence_l142_142204


namespace boys_on_soccer_team_l142_142962

theorem boys_on_soccer_team (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : B = 15 :=
sorry

end boys_on_soccer_team_l142_142962


namespace triangle_semicircle_radius_l142_142974

theorem triangle_semicircle_radius 
  (A B C : Type*) [metric_space A] [metric_space B] [metric_space C]
  (hABC : ∠ A B C = 90)
  (area_semicircle_AB : ∀ (r : ℝ), ½ * Real.pi * r^2 = 8 * Real.pi)
  (arc_length_semicircle_AC : ∀ (r : ℝ), Real.pi * r = 8.5 * Real.pi) :
  (radius_bc : ℝ), radius_bc = 7.5 :=
begin
  sorry
end

end triangle_semicircle_radius_l142_142974


namespace integer_solutions_of_quadratic_eq_l142_142033

theorem integer_solutions_of_quadratic_eq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ x1 x2 : ℤ, x1 * x2 = q^4 ∧ x1 + x2 = -p ∧ x1 = -1 ∧ x2 = - (q^4) ∧ p = 17 ∧ q = 2 := 
sorry

end integer_solutions_of_quadratic_eq_l142_142033


namespace carol_total_peanuts_l142_142232

-- Conditions as definitions
def carol_initial_peanuts : Nat := 2
def carol_father_peanuts : Nat := 5

-- Theorem stating that the total number of peanuts Carol has is 7
theorem carol_total_peanuts : carol_initial_peanuts + carol_father_peanuts = 7 := by
  -- Proof would go here, but we use sorry to skip
  sorry

end carol_total_peanuts_l142_142232


namespace diet_soda_bottles_l142_142951

/-- Define variables for the number of bottles. -/
def total_bottles : ℕ := 38
def regular_soda : ℕ := 30

/-- Define the problem of finding the number of diet soda bottles -/
def diet_soda := total_bottles - regular_soda

/-- Claim that the number of diet soda bottles is 8 -/
theorem diet_soda_bottles : diet_soda = 8 :=
by
  sorry

end diet_soda_bottles_l142_142951


namespace sum_f_values_l142_142820

def f (x : ℝ) : ℝ := 5 * x ^ 6 - 3 * x ^ 5 + 4 * x ^ 4 + x ^ 3 - 2 * x ^ 2 - 2 * x + 8

theorem sum_f_values : f 5 + f (-5) = 68343 :=
by
  have h₁ : f (5) = 20 := by sorry
  have h₂ : f (-5) = 68323 := by sorry
  calc
    f 5 + f (-5) = 20 + 68323 : by rw [h₁, h₂]
    ... = 68343 : by norm_num

end sum_f_values_l142_142820


namespace perfect_squares_ending_in_4_5_6_less_than_2000_l142_142744

theorem perfect_squares_ending_in_4_5_6_less_than_2000 :
  let squares := { n : ℕ | n * n < 2000 ∧ (n * n % 10 = 4 ∨ n * n % 10 = 5 ∨ n * n % 10 = 6) } in
  squares.card = 23 :=
by
  sorry

end perfect_squares_ending_in_4_5_6_less_than_2000_l142_142744


namespace Bayes_theorem_2_white_balls_l142_142524

noncomputable def P (A_i : ℕ → bool) : ℝ := do 
  sorry 

noncomputable def BayesTheorem {A : Type*} (P : A → ℝ) (B : A → ℝ) : A → ℝ :=
  λ x, P x * B x / ∑ y in A, P y * B y

theorem Bayes_theorem_2_white_balls :
  let A := ℕ → bool,
      B := 2 * (∑ j in 0, 1, 2, P (λ j, true) * P (λ j : 0, 1, 2 : Φ, true)) :=
  BayesTheorem (λ A, 3 / 10 * 2 / 5) (∑ j in 0 to 2, P (λ A, true) (λ i, B)) 
    = 18 / 37 :=
  sorry

end Bayes_theorem_2_white_balls_l142_142524


namespace inverse_function_coeff_ratio_l142_142089

noncomputable def f_inv_coeff_ratio : ℝ :=
  let f (x : ℝ) := (2 * x - 1) / (x + 5)
  let a := 5
  let b := 1
  let c := -1
  let d := 2
  a / c

theorem inverse_function_coeff_ratio :
  f_inv_coeff_ratio = -5 := 
by
  sorry

end inverse_function_coeff_ratio_l142_142089


namespace alcohol_quantity_in_mixture_l142_142554

theorem alcohol_quantity_in_mixture 
  (A W : ℝ)
  (h1 : A / W = 4 / 3)
  (h2 : A / (W + 4) = 4 / 5)
  : A = 8 :=
sorry

end alcohol_quantity_in_mixture_l142_142554


namespace derivative_y1_derivative_y2_l142_142647

-- Problem 1: Define the function and prove its derivative
def y1 (x : ℝ) := (2 * x ^ 2 + 3) * (3 * x - 1)
theorem derivative_y1 : ∀ x : ℝ, deriv y1 x = 18 * x ^ 2 - 4 * x + 9 := 
by 
  sorry

-- Problem 2: Define the function and prove its derivative
def y2 (x : ℝ) := (1 - sin x) / (1 + cos x)
theorem derivative_y2 : ∀ x : ℝ, deriv y2 x = (-1 - cos x + sin x) / (1 + cos x) ^ 2 := 
by 
  sorry

end derivative_y1_derivative_y2_l142_142647


namespace log_sum_property_l142_142942

theorem log_sum_property :
  2 * log 5 10 + log 5 0.25 = 2 := 
by
  sorry

end log_sum_property_l142_142942


namespace square_side_length_and_circle_radius_l142_142600

theorem square_side_length_and_circle_radius (area : ℝ) (h : area = 1) :
  ∃ (side_length : ℝ) (radius : ℝ), side_length = 1 ∧ radius = 0.5 :=
by
  use 1, 0.5
  exact ⟨rfl, rfl⟩

end square_side_length_and_circle_radius_l142_142600


namespace triangle_is_isosceles_l142_142097

-- Define the points A, B, and C
def A : ℝ × ℝ × ℝ := (4, 3, 1)
def B : ℝ × ℝ × ℝ := (7, 1, 2)
def C : ℝ × ℝ × ℝ := (5, 2, 3)

-- Define the distance function between two points
def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2 + (Q.3 - P.3) ^ 2)

-- Theorem to prove that the triangle is isosceles
theorem triangle_is_isosceles : 
  distance A B = distance B C ∨ distance B C = distance C A ∨ distance C A = distance A B :=
by
  sorry

end triangle_is_isosceles_l142_142097


namespace triangle_side_length_l142_142420

theorem triangle_side_length (y z : ℝ) (cos_Y_minus_Z : ℝ) (h_y : y = 7) (h_z : z = 6) (h_cos : cos_Y_minus_Z = 17 / 18) : 
  ∃ x : ℝ, x = Real.sqrt 65 :=
by
  sorry

end triangle_side_length_l142_142420


namespace initial_fish_count_l142_142831

-- Definitions based on the given conditions
def Fish_given : ℝ := 22.0
def Fish_now : ℝ := 25.0

-- The goal is to prove the initial number of fish Mrs. Sheridan had.
theorem initial_fish_count : (Fish_given + Fish_now) = 47.0 := by
  sorry

end initial_fish_count_l142_142831


namespace sophomores_selected_correct_l142_142184

-- Define the number of students in each grade and the total spots for the event
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300
def totalSpots : ℕ := 40

-- Calculate the total number of students
def totalStudents : ℕ := freshmen + sophomores + juniors

-- The correct answer we want to prove
def numberOfSophomoresSelected : ℕ := (sophomores * totalSpots) / totalStudents

-- Statement to be proved
theorem sophomores_selected_correct : numberOfSophomoresSelected = 26 := by
  -- Proof is omitted
  sorry

end sophomores_selected_correct_l142_142184


namespace max_value_n_exists_l142_142401

noncomputable theory

def grid := fin 9 → fin 9 → ℕ

theorem max_value_n_exists (G : grid) :
  (∀ i : fin 9, ∃ S : set ℕ, S.card ≤ 3 ∧ ∀ j : fin 9, G i j ∈ S) →
  (∀ j : fin 9, ∃ S : set ℕ, S.card ≤ 3 ∧ ∀ i : fin 9, G i j ∈ S) →
  ∃ n : ℕ, n = 3 ∧ ∃ a : ℕ, (∃ i : fin 9, (G i = λ j, a) ∧ ∃ j : fin 9, (G i j = a)) :=
begin
  sorry,
end

end max_value_n_exists_l142_142401


namespace sin_alpha_minus_beta_l142_142752

-- Definitions based on the conditions provided
def e_i_alpha := complex.exp (complex.i * α) = (4/5 : ℝ) + (3/5 : ℝ) * complex.i
def e_i_beta := complex.exp (complex.i * β) = (12/13 : ℝ) + (5/13 : ℝ) * complex.i

-- Statement of the proof problem
theorem sin_alpha_minus_beta : 
  e_i_alpha →
  e_i_beta →
  real.sin (α - β) = - (16 / 65) :=
by
  sorry

end sin_alpha_minus_beta_l142_142752


namespace T_2016_eq_l142_142507

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def b_n (n : ℕ) : ℚ :=
  1 / ((n + 1) * a_n n)

noncomputable def T_n (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, b_n (i + 1))

theorem T_2016_eq : T_n 2016 = 2016 / 2017 := by
  sorry

end T_2016_eq_l142_142507


namespace complex_modulus_w_l142_142805

open Complex

theorem complex_modulus_w :
  let z := (⟨-7, 15⟩^2 * ⟨8, 11⟩^3) / ⟨5, 12⟩ in
  let w := conj z / z in
  abs w = 1 :=
by
  sorry

end complex_modulus_w_l142_142805


namespace pf1_pf2_plus_op_eq_25_l142_142037

-- Define the necessary structures and conditions
variables {x y : ℝ}

def ellipse (P : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), P = (x, y) ∧ (x^2 / 16 + y^2 / 9 = 1)

def foci_1 : ℝ × ℝ := (-sqrt 7, 0)
def foci_2 : ℝ × ℝ := (sqrt 7, 0)
def center : ℝ × ℝ := (0, 0)

-- Define the goal we are trying to prove
theorem pf1_pf2_plus_op_eq_25 (P : ℝ × ℝ) (hP : ellipse P) : 
  let PF1 := dist P foci_1 in
  let PF2 := dist P foci_2 in
  let OP := dist P center in
  PF1 * PF2 + OP^2 = 25 :=
sorry

end pf1_pf2_plus_op_eq_25_l142_142037


namespace find_u_l142_142240

-- Definitions for given points lying on a straight line
def point := (ℝ × ℝ)

-- Points
def p1 : point := (2, 8)
def p2 : point := (6, 20)
def p3 : point := (10, 32)

-- Function to check if point is on the line derived from p1, p2, p3
def is_on_line (x y : ℝ) : Prop :=
  ∃ m b : ℝ, y = m * x + b ∧
  p1.2 = m * p1.1 + b ∧ 
  p2.2 = m * p2.1 + b ∧
  p3.2 = m * p3.1 + b

-- Statement to prove
theorem find_u (u : ℝ) (hu : is_on_line 50 u) : u = 152 :=
sorry

end find_u_l142_142240


namespace miranda_more_delivers_in_a_month_l142_142426

theorem miranda_more_delivers_in_a_month : 
  let jake_delivers_weekly := 234
  let miranda_delivers_weekly := 2 * jake_delivers_weekly
  let weeks_in_a_month := 4
  let difference_in_weekly_delivery := miranda_delivers_weekly - jake_delivers_weekly
  let difference_in_monthly_delivery := difference_in_weekly_delivery * weeks_in_a_month
  in difference_in_monthly_delivery = 936 :=
by
  sorry

end miranda_more_delivers_in_a_month_l142_142426


namespace correct_log_values_l142_142000

theorem correct_log_values (a b c : ℝ) :
  (∃ l1 l7 : ℝ, 
    l1 = 3a - b + c - 1 ∧
    l7 = 2b + c) → 
  (∀ x, (x = 1.5 → log10 x = 3a - b + c - 1) ∧
       (x = 7 → log10 x = 2b + c)) := 
by 
  sorry

end correct_log_values_l142_142000


namespace triangle_propositions_l142_142671

/--
For a triangle ABC, the following propositions hold:
1. If \(\sin 2A = \sin 2B\), then \(\triangle ABC\) must be an isosceles triangle.
2. If \(\sin A = \sin B\), then \(\triangle ABC\) must be an isosceles triangle.
3. If \(\sin^2 A + \sin^2 B + \cos^2 C < 1\), then \(\triangle ABC\) must be an obtuse triangle.
4. If \(\tan A + \tan B + \tan C > 0\), then \(\triangle ABC\) must be an acute triangle.
We need to prove that propositions (2), (3), and (4) are correct.
-/
theorem triangle_propositions (A B C : ℝ) (h1 : sin (2 * A) = sin (2 * B))
(h2 : sin A = sin B) (h3 : sin A ^ 2 + sin B ^ 2 + cos C ^ 2 < 1) 
(h4 : tan A + tan B + tan C > 0) :
  (if sin A = sin B then is_isosceles ABC else true) ∧
  (if sin A ^ 2 + sin B ^ 2 + cos C ^ 2 < 1 then is_obtuse ABC else true) ∧
  (if tan A + tan B + tan C > 0 then is_acute ABC else true) :=
by
  sorry

end triangle_propositions_l142_142671


namespace recurrence_relation_F_l142_142031

def F : ℕ → ℕ
| 0       := 1
| (n + 1) := ∑ k in Finset.range (n + 1), F k * F (n - k)

theorem recurrence_relation_F 
  (n : ℕ) (h : n > 0) : 
  F n = ∑ k in Finset.range n, F k * F (n - 1 - k) :=
by sorry

end recurrence_relation_F_l142_142031


namespace remove_12_kings_l142_142435

-- Define the chessboard as an 8x8 grid
@[derive DecidableEq]
inductive Square : Type
| mk : Fin 8 → Fin 8 → Square

def attacks (sq1 sq2 : Square) : Prop :=
  match sq1, sq2 with
  | Square.mk x1 y1, Square.mk x2 y2 =>
    abs (x1 - x2) ≤ 1 ∧ abs (y1 - y2) ≤ 1

-- Define a configuration of n kings on the chessboard
structure Configuration (n : Nat) :=
  pos : Fin n → Square

def non_threatening (cfg : Configuration 5) : Prop :=
  ∀ i j, i ≠ j → ¬ attacks (cfg.pos i) (cfg.pos j)

-- The main problem statement
theorem remove_12_kings (init_cfg : Configuration 17) :
  ∃ (final_cfg : Configuration 5), non_threatening final_cfg :=
by
  sorry

end remove_12_kings_l142_142435


namespace geometric_sequence_magnitude_Sn_2013_l142_142694

-- Define the given vector sequence 
def vector_seq : ℕ → ℝ × ℝ
| 0     := (0, 0)
| 1     := (1, 1)
| (n+2) := let (x, y) := vector_seq (n+1) in (1/2 * (x - y), 1/2 * (x + y))

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define bn and Sn
def theta (n : ℕ) : ℝ := real.pi / 4
def bn (n : ℕ) : ℝ := real.pi / (4 * n * (n - 1) * theta n)
def Sn (n : ℕ) : ℝ := ∑ i in finset.range (n-1), bn (i + 2)

theorem geometric_sequence_magnitude : ∀ n, 
  magnitude (vector_seq (n + 2)) = (real.sqrt 2 / 2) ^ n * magnitude (vector_seq 1) :=
sorry

theorem Sn_2013 : Sn 2013 = 2012 / 2013 := 
sorry

end geometric_sequence_magnitude_Sn_2013_l142_142694


namespace perimeter_PQRS_value_c_plus_d_is_114_l142_142454

structure Point where
  x : ℝ
  y : ℝ

def distance (a b : Point) : ℝ :=
  real.sqrt ((b.x - a.x) ^ 2 + (b.y - a.y) ^ 2)

noncomputable def perimeter_of_PQRS (a b c d : Point) : ℝ :=
  distance a b + distance b c + distance c d + distance d a

theorem perimeter_PQRS_value :
  let P := {x := 0, y := 0}
  let Q := {x := 3, y := 4}
  let R := {x := 6, y := 0}
  let S := {x := 9, y := 4}
  let perim := perimeter_of_PQRS P Q R S
  perim = 15 + real.sqrt 97 := by
    simp [P, Q, R, S, perim, perimeter_of_PQRS, distance]
    sorry

theorem c_plus_d_is_114 :
  let c := 16
  let d := 98
  c + d = 114 := by
    simp [c, d]
    sorry

end perimeter_PQRS_value_c_plus_d_is_114_l142_142454


namespace percentage_books_not_sold_l142_142430

theorem percentage_books_not_sold (initial_stock : ℕ) (sold_monday : ℕ) (sold_tuesday : ℕ) 
(sold_wednesday : ℕ) (sold_thursday : ℕ) (sold_friday : ℕ) : 
(initial_stock = 1200) → 
(sold_monday = 75) → 
(sold_tuesday = 50) → 
(sold_wednesday = 64) → 
(sold_thursday = 78) → 
(sold_friday = 135) → 
(100 * (initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday)) / initial_stock = 66.5) :=
by
  intros h_initial_stock h_sold_monday h_sold_tuesday h_sold_wednesday h_sold_thursday h_sold_friday
  sorry

end percentage_books_not_sold_l142_142430


namespace min_w_l142_142247

def w (x y z : ℝ) : ℝ := x^2 + 4*y^2 + 8*x - 6*y + z - 20

theorem min_w (x y z : ℝ) : ∃ a : ℝ, (∀ x y, a ≤ w x y z) ∧ (∃ x y, a = w x y z) :=
begin
  use z - 38.25,
  split,
  { intros x y,
    have h1 : (x + 4)^2 ≥ 0 := pow_two_nonneg (x + 4),
    have h2 : 4 * (y - 0.75)^2 ≥ 0 := mul_nonneg (by norm_num) (pow_two_nonneg (y - 0.75)),
    have hw : w x y z = (x + 4)^2 + 4 * (y - 0.75)^2 + z - 38.25 := by {
      field_simp [w],
      ring,
    },
    linarith, },
  { use [-4, 0.75],
    field_simp [w],
    ring, },
end

end min_w_l142_142247


namespace unique_positive_solution_l142_142657

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l142_142657


namespace constant_term_in_binomial_expansion_l142_142782

theorem constant_term_in_binomial_expansion :
  let term := (fun (n k : ℕ) (a b : ℂ) => (↑((nat.descFactorial n k) / (nat.factorial k)) * a^(n-k) * b^k))
  let bin_exp := (fun (x : ℂ) (n : ℕ) => List.sum $ List.map (λ k => term n k x (-1 / x)) (List.range n.succ))
  (bin_exp x 8).constant = 70 := 
by 
  sorry

end constant_term_in_binomial_expansion_l142_142782


namespace unique_positive_solution_eq_15_l142_142652

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l142_142652


namespace divisible_by_7_in_range_200_to_400_l142_142373

theorem divisible_by_7_in_range_200_to_400 : 
  ∃ n : ℕ, 
    (∀ (x : ℕ), (200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ finset.range (201)) ∧ finset.card (finset.filter (λ x, 200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0) (finset.range 401)) = 29) := 
begin
  sorry
end

end divisible_by_7_in_range_200_to_400_l142_142373


namespace intervals_of_increase_cos_a_l142_142684

def f (x : ℝ) : ℝ := 2 * sin x * (sin x + cos x)

theorem intervals_of_increase (k : ℤ) : ∀ x : ℝ, 
  (k * real.pi - real.pi / 8 ≤ x ∧ x ≤ k * real.pi + 3 * real.pi / 8) ↔ 
  (∃ x0 : ℝ, f x0 = x ∧ strict_mono_incr_on f (k * real.pi - real.pi / 8) (k * real.pi + 3 * real.pi / 8)) :=
sorry

theorem cos_a (a : ℝ) (h1 : f (a / 2) = 1 + 3 * real.sqrt 2 / 5) (h2 : 3 * real.pi / 4 < a ∧ a < 5 * real.pi / 4) : 
  cos a = - 7 * real.sqrt 2 / 10 :=
sorry

end intervals_of_increase_cos_a_l142_142684


namespace complex_number_on_imaginary_axis_l142_142866

theorem complex_number_on_imaginary_axis (a : ℝ) 
(h : ∃ z : ℂ, z = (a^2 - 2 * a) + (a^2 - a - 2) * Complex.I ∧ z.re = 0) : 
a = 0 ∨ a = 2 :=
by
  sorry

end complex_number_on_imaginary_axis_l142_142866


namespace cannot_form_set_definiteness_l142_142922

/--
The problem is to determine which of the given sets cannot form a set due to indefiniteness of the elements.
The given options are:
A: All prime numbers between 1 and 20.
B: All real roots of the equation x^2 + x - 2 = 0.
C: All taller students at Xinhua High School.
D: All squares.

The correct answer is the set consisting of all taller students at Xinhua High School since it cannot be definitively formed.
-/
theorem cannot_form_set_definiteness:
    ∃ S, S = { x : Type | x = "All taller students at Xinhua High School" ∧ ¬(∃ s, s ∈ S)} :=
    sorry

end cannot_form_set_definiteness_l142_142922


namespace discount_percentage_l142_142596

/-
  A retailer buys 80 pens at the market price of 36 pens from a wholesaler.
  He sells these pens giving a certain discount and his profit is 120%.
  What is the discount percentage he gave on the pens?
-/
theorem discount_percentage
  (P : ℝ)
  (CP SP D DP : ℝ) 
  (h1 : CP = 36 * P)
  (h2 : SP = 2.2 * CP)
  (h3 : D = P - (SP / 80))
  (h4 : DP = (D / P) * 100) :
  DP = 1 := 
sorry

end discount_percentage_l142_142596


namespace num_natural_numbers_divisible_by_7_l142_142384

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l142_142384


namespace find_y_l142_142667

theorem find_y (y : ℝ) (h : |2 * y - 44| + |y - 24| = |3 * y - 66|) : y = 23 := 
by 
  sorry

end find_y_l142_142667


namespace natural_numbers_divisible_by_7_between_200_400_l142_142366

theorem natural_numbers_divisible_by_7_between_200_400 : 
  { n : ℕ | 200 <= n ∧ n <= 400 ∧ n % 7 = 0 }.to_finset.card = 29 := 
  sorry

end natural_numbers_divisible_by_7_between_200_400_l142_142366


namespace compute_Q3_Qneg3_l142_142809

noncomputable def Q (x : ℝ) (a b c m : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + m

theorem compute_Q3_Qneg3 (a b c m : ℝ)
  (h1 : Q 1 a b c m = 3 * m)
  (h2 : Q (-1) a b c m = 4 * m)
  (h3 : Q 0 a b c m = m) :
  Q 3 a b c m + Q (-3) a b c m = 47 * m :=
by
  sorry

end compute_Q3_Qneg3_l142_142809


namespace track_length_l142_142216

theorem track_length (x : ℕ) 
  (diametrically_opposite : ∃ a b : ℕ, a + b = x)
  (first_meeting : ∃ b : ℕ, b = 100)
  (second_meeting : ∃ s s' : ℕ, s = 150 ∧ s' = (x / 2 - 100 + s))
  (constant_speed : ∀ t₁ t₂ : ℕ, t₁ / t₂ = 100 / (x / 2 - 100)) :
  x = 400 := 
by sorry

end track_length_l142_142216


namespace sum_of_divisors_divisible_by_24_l142_142457

theorem sum_of_divisors_divisible_by_24 (n : ℕ) (h : n + 1 ∣ 24) : 24 ∣ ∑ d in nat.divisors n, d :=
sorry

end sum_of_divisors_divisible_by_24_l142_142457


namespace temperature_difference_l142_142924

/-- The average temperature at the top of Mount Tai. -/
def T_top : ℝ := -9

/-- The average temperature at the foot of Mount Tai. -/
def T_foot : ℝ := -1

/-- The temperature difference between the average temperature at the foot and the top of Mount Tai is 8 degrees Celsius. -/
theorem temperature_difference : T_foot - T_top = 8 := by
  sorry

end temperature_difference_l142_142924


namespace cube_surface_area_l142_142875

-- Definitions based on conditions from the problem
def edge_length : ℕ := 7
def number_of_faces : ℕ := 6

-- Definition of the problem converted to a theorem in Lean 4
theorem cube_surface_area (edge_length : ℕ) (number_of_faces : ℕ) : 
  number_of_faces * (edge_length * edge_length) = 294 :=
by
  -- Proof steps are omitted, so we put sorry to indicate that the proof is required.
  sorry

end cube_surface_area_l142_142875


namespace work_hours_l142_142431

namespace JohnnyWork

variable (dollarsPerHour : ℝ) (totalDollars : ℝ)

theorem work_hours 
  (h_wage : dollarsPerHour = 3.25)
  (h_earned : totalDollars = 26) 
  : (totalDollars / dollarsPerHour) = 8 := 
by
  rw [h_wage, h_earned]
  -- proof goes here
  sorry

end JohnnyWork

end work_hours_l142_142431


namespace sum_digits_of_three_digit_numbers_l142_142892

theorem sum_digits_of_three_digit_numbers (a c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hc : 1 ≤ c ∧ c < 10) 
  (h1 : (300 + 10 * a + 7) + 414 = 700 + 10 * c + 1)
  (h2 : ∃ k : ℤ, 700 + 10 * c + 1 = 11 * k) :
  a + c = 14 :=
by
  sorry

end sum_digits_of_three_digit_numbers_l142_142892


namespace scientific_notation_248000_l142_142864

theorem scientific_notation_248000 : (248000 : Float) = 2.48 * 10^5 := 
sorry

end scientific_notation_248000_l142_142864


namespace tracy_starting_candies_l142_142528

-- Define the initial conditions
def initial_candies (x : ℕ) (eaten_tracy_fraction : ℚ) (given_to_rachel_fraction : ℚ) (candies_eaten_by_tracy_and_mom : ℕ)
                    (min_candies_taken_by_brother : ℕ) (max_candies_taken_by_brother : ℕ) (final_candies_tracy : ℕ) :=
  tracy_fraction_remained : ℚ := 1 - eaten_tracy_fraction
  rachel_fraction_remained : ℚ := 1 - given_to_rachel_fraction
  half_candies := x * tracy_fraction_remained * rachel_fraction_remained / 1
  remaining_candies := half_candies.toNat - candies_eaten_by_tracy_and_mom
  min_remaining_candies := remaining_candies + min_candies_taken_by_brother
  max_remaining_candies := remaining_candies + max_candies_taken_by_brother

-- Statement to prove the initial number of candies is 96
theorem tracy_starting_candies : ∃ x : ℕ, initial_candies x (1/4:ℚ) (1/3:ℚ) 40 2 6 5 ∧ x = 96 := by
  sorry

end tracy_starting_candies_l142_142528


namespace condition_necessary_but_not_sufficient_l142_142389

theorem condition_necessary_but_not_sufficient (a : ℝ) :
  ((1 / a > 1) → (a < 1)) ∧ (∃ (a : ℝ), a < 1 ∧ 1 / a < 1) :=
by
  sorry

end condition_necessary_but_not_sufficient_l142_142389


namespace select_three_consecutive_circles_l142_142950

theorem select_three_consecutive_circles
    (h: ∃ (figure : List (List ℕ)), 
      (figure = [[6], [5], [4], [3], [2], [1]]) ∨ 
      (figure = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])) :
    ∃ ways : ℕ, ways = 57 :=
by
  -- We define variables to represent the given conditions
  let horizontal_rows := [[6], [5], [4], [3], [2], [1]]
  let diagonal_rows := [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]]

  -- We assert that these are the correct arrangements based on the conditions
  have H1 : ∃ (figure : List (List ℕ)), (figure = horizontal_rows ∨ figure = diagonal_rows), from h

  -- We calculate the total number of ways to select three consecutive circles
  have total_ways := 21 + 36

  -- We assert the final result
  existsi (57 : ℕ)
  exact total_ways -- Ways to select in horizontal and both diagonal directions added
  sorry

end select_three_consecutive_circles_l142_142950


namespace relationship_between_x_and_y_l142_142396

variables (x y : ℝ)

theorem relationship_between_x_and_y (h1 : x + y > 2 * x) (h2 : x - y < 2 * y) : y > x := 
sorry

end relationship_between_x_and_y_l142_142396


namespace min_value_f_l142_142851

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + 4 * x + 20) + Real.sqrt (x^2 + 2 * x + 10)

theorem min_value_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 :=
by
  sorry

end min_value_f_l142_142851


namespace log_eq_solution_range_l142_142291

theorem log_eq_solution_range (a k x : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  (∃ x, log (sqrt a) (x - a * k) = log a (x^2 - a^2)) ↔ k ∈ set.Iio (-1) ∪ set.Ioo 0 1 :=
by
  sorry

end log_eq_solution_range_l142_142291


namespace cally_pants_count_l142_142985

variable (cally_white_shirts : ℕ)
variable (cally_colored_shirts : ℕ)
variable (cally_shorts : ℕ)
variable (danny_white_shirts : ℕ)
variable (danny_colored_shirts : ℕ)
variable (danny_shorts : ℕ)
variable (danny_pants : ℕ)
variable (total_clothes_washed : ℕ)
variable (cally_pants : ℕ)

-- Given conditions
#check cally_white_shirts = 10
#check cally_colored_shirts = 5
#check cally_shorts = 7
#check danny_white_shirts = 6
#check danny_colored_shirts = 8
#check danny_shorts = 10
#check danny_pants = 6
#check total_clothes_washed = 58
#check cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed

-- Proof goal
theorem cally_pants_count (cally_white_shirts cally_colored_shirts cally_shorts danny_white_shirts danny_colored_shirts danny_shorts danny_pants cally_pants total_clothes_washed : ℕ) :
  cally_white_shirts = 10 →
  cally_colored_shirts = 5 →
  cally_shorts = 7 →
  danny_white_shirts = 6 →
  danny_colored_shirts = 8 →
  danny_shorts = 10 →
  danny_pants = 6 →
  total_clothes_washed = 58 →
  (cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed) →
  cally_pants = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end cally_pants_count_l142_142985


namespace carla_water_requirement_l142_142986

theorem carla_water_requirement (h: ℕ) (p: ℕ) (c: ℕ) (gallons_per_pig: ℕ) (horse_factor: ℕ) 
  (num_pigs: ℕ) (num_horses: ℕ) (tank_water: ℕ): 
  num_pigs = 8 ∧ num_horses = 10 ∧ gallons_per_pig = 3 ∧ horse_factor = 2 ∧ tank_water = 30 →
  h = horse_factor * gallons_per_pig ∧ p = num_pigs * gallons_per_pig ∧ c = tank_water →
  h * num_horses + p + c = 114 :=
by
  intro h1 h2
  cases h1
  cases h2
  sorry

end carla_water_requirement_l142_142986


namespace tangent_symmetry_of_triangle_l142_142806

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the points, altitudes, and necessary geometric constructions
variables {H_A H_B H_C : A → B → C → Type}  -- Altitudes of triangle A, B, C
           [Altitude H_A A B C]  -- Altitude from A to BC
           [Altitude H_B B C A]  -- Altitude from B to CA
           [Altitude H_C C A B]  -- Altitude from C to AB

-- Define the given ratio condition
variable (ratio_cond : ∀ A B C : Type, (H_BC / AC) = (H_CA / AB))

-- Define symmetrical property and tangent to circumscribed circle
variable (is_symmetric : SymmetricTo BC H_B H_C)
variable (is_tangent : TangentToCircumscribed H_B H_C A)

theorem tangent_symmetry_of_triangle (A B C : Type) 
  (altitude_A : Altitude H_A A B C)
  (altitude_B : Altitude H_B B C A)
  (altitude_C : Altitude H_C C A B)
  (ratio_cond : ∀ A B C : Type, (H_BC / AC) = (H_CA / AB)) :
  is_tangent (SymmetricTo BC H_B H_C) (Circumscribed H_B H_C A) :=
sorry

end tangent_symmetry_of_triangle_l142_142806


namespace num_natural_numbers_divisible_by_7_l142_142379

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l142_142379


namespace chocolate_bars_count_l142_142460

theorem chocolate_bars_count (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
    (h_milk : milk_chocolate = 25)
    (h_almond : almond_chocolate = 25)
    (h_white : white_chocolate = 25)
    (h_percent : milk_chocolate = almond_chocolate ∧ almond_chocolate = white_chocolate ∧ white_chocolate = dark_chocolate) :
    dark_chocolate = 25 := by
  sorry

end chocolate_bars_count_l142_142460


namespace james_total_distance_l142_142011

theorem james_total_distance
    (speed : ℝ)
    (morning_hours : ℝ)
    (afternoon_hours : ℝ)
    (total_distance : ℝ) :
    speed = 8 →
    morning_hours = 2.5 →
    afternoon_hours = 1.5 →
    total_distance = (speed * morning_hours + speed * afternoon_hours) →
    total_distance = 32.0 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end james_total_distance_l142_142011


namespace numbers_divisor_property_l142_142137

theorem numbers_divisor_property (S : Finset ℕ) (h₁ : Finset.card S = 1008) (h₂ : ∀ x ∈ S, x ≤ 2014) :
  ∃ a b ∈ S, a ∣ b ∨ b ∣ a :=
by
  sorry

end numbers_divisor_property_l142_142137


namespace contradiction_proof_l142_142533

theorem contradiction_proof (a b : ℝ) : a + b = 12 → ¬ (a < 6 ∧ b < 6) :=
by
  intro h
  intro h_contra
  sorry

end contradiction_proof_l142_142533


namespace find_a_l142_142451

theorem find_a (a : ℝ) (h₁ : ∀ x, (λ x, (real.log (3 * x + a))).derivative x = (3 / (3 * x + a))) (h₂ : 3 / a = 1) : a = 3 :=
by
  sorry

end find_a_l142_142451


namespace area_triangle_AEB_l142_142778

theorem area_triangle_AEB
  {A B C D F G E : Type}
  (h_rect : is_rectangle A B C D)
  (h_side_AB : dist A B = 6)
  (h_side_BC : dist B C = 3)
  (h_point_F : on_segment D C F)
  (h_point_G : on_segment D C G)
  (h_DF : dist D F = 1)
  (h_GC : dist G C = 2)
  (h_inter_AF_BG : intersect_at AF BG E)
  : area (triangle A E B) = 18 := 
sorry

end area_triangle_AEB_l142_142778


namespace greatest_number_of_police_officers_needed_l142_142049

-- Define the conditions within Math City
def number_of_streets : ℕ := 10
def number_of_tunnels : ℕ := 2
def intersections_without_tunnels : ℕ := (number_of_streets * (number_of_streets - 1)) / 2
def intersections_bypassed_by_tunnels : ℕ := number_of_tunnels

-- Define the number of police officers required (which is the same as the number of intersections not bypassed)
def police_officers_needed : ℕ := intersections_without_tunnels - intersections_bypassed_by_tunnels

-- The main theorem: Given the conditions, the greatest number of police officers needed is 43.
theorem greatest_number_of_police_officers_needed : police_officers_needed = 43 := 
by {
  -- Proof would go here, but we'll use sorry to indicate it's not provided.
  sorry
}

end greatest_number_of_police_officers_needed_l142_142049


namespace cosine_sine_zero_solutions_in_interval_l142_142629

theorem cosine_sine_zero_solutions_in_interval :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), cos (π / 3 * sin x) ≠ sin (π / 3 * cos x) := 
by
  sorry

end cosine_sine_zero_solutions_in_interval_l142_142629


namespace grasshopper_frog_jump_total_l142_142088

theorem grasshopper_frog_jump_total (grasshopper_jump frog_jump : ℕ) : 
  grasshopper_jump = 31 ∧ frog_jump = 35 → grasshopper_jump + frog_jump = 66 := 
by
  intro h
  cases h with hg hf
  rw [hg, hf]
  rfl

end grasshopper_frog_jump_total_l142_142088


namespace relationship_among_values_l142_142041

noncomputable theory
open_locale classical

-- Define f as a function ℝ → ℝ
variable (f : ℝ → ℝ)

-- Conditions
def even_function : Prop := ∀ x : ℝ, f (-x) = f x
def domain_real : Prop := ∀ x : ℝ, true
def increasing_on_nonneg : Prop := ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- Theorem statement
theorem relationship_among_values (h_even : even_function f)
    (h_domain : domain_real f) 
    (h_increasing : increasing_on_nonneg f) :
    f π > f (-2) ∧ f (-2) > f (-1) :=
by 
  -- By the definition of even function, we have:
  have h1 : f (-1) = f 1 := h_even 1,
  have h2 : f (-2) = f 2 := h_even 2,

  -- Since f is increasing on [0, +∞), we have:
  have h3 : f π > f 2,
  have h4 : f 2 > f 1,

  -- Therefore, combining these facts:
  sorry

end relationship_among_values_l142_142041


namespace original_price_l142_142594

theorem original_price (saving : ℝ) (percentage : ℝ) (h_saving : saving = 10) (h_percentage : percentage = 0.10) :
  ∃ OP : ℝ, OP = 100 :=
by
  sorry

end original_price_l142_142594


namespace find_BF_pqsum_l142_142860

noncomputable def square_side_length : ℝ := 900
noncomputable def EF_length : ℝ := 400
noncomputable def m_angle_EOF : ℝ := 45
noncomputable def center_mid_to_side : ℝ := square_side_length / 2

theorem find_BF_pqsum :
  let G_mid : ℝ := center_mid_to_side
  let x : ℝ := G_mid - (2 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let y : ℝ := (1 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let BF := G_mid - y
  BF = 250 + 50 * Real.sqrt 7 ->
  250 + 50 + 7 = 307 := sorry

end find_BF_pqsum_l142_142860


namespace eight_parallel_planes_eq_dist_exist_l142_142010

noncomputable def vertices : List (ℝ × ℝ × ℝ) := [
  (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
  (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)
  ]

def plane (k : ℝ) : ℝ × ℝ × ℝ → Prop :=
  λ (x : ℝ × ℝ × ℝ), x.1 + 2 * x.2 + 4 * x.3 = k

theorem eight_parallel_planes_eq_dist_exist :
  ∃ k_seq : Fin 8 → ℝ, (∀ i, plane (k_seq i) ∈ vertices) ∧ (∀ i j, abs (k_seq i - k_seq j) = 1 / Real.sqrt 21) :=
sorry

end eight_parallel_planes_eq_dist_exist_l142_142010


namespace divisor_exists_l142_142120

-- Define the initial conditions
def all_natural_numbers := { n : ℕ | 1 ≤ n ∧ n ≤ 2014 }

-- Define the remaining set after erasing 1006 numbers
def remaining_numbers (S : set ℕ) (h : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : Prop :=
  ∃ a b ∈ S, a ≠ b ∧ a ∣ b

-- The problem statement in Lean
theorem divisor_exists (S : set ℕ) (h_sub : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : 
  remaining_numbers S h_sub h_card :=
begin
  sorry
end

end divisor_exists_l142_142120


namespace two_numbers_divisor_property_l142_142115

theorem two_numbers_divisor_property (s : Finset ℕ) (h1 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) (h2 : s.card = 1008) 
  : ∃ a b ∈ s, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end two_numbers_divisor_property_l142_142115


namespace age_difference_ratio_l142_142852

def Roy_age_condition_1 (R J K : ℕ) : Prop := R = J + 8
def Roy_age_condition_2 (R J K : ℕ) : Prop := R + 2 = 3 * (J + 2)
def Roy_age_condition_3 (R J K : ℕ) : Prop := (R + 2) * (K + 2) = 96

def ratio_of_age_differences (R J K : ℕ) : ℚ := (R - J : ℚ) / (R - K)

theorem age_difference_ratio (R J K : ℕ) :
  Roy_age_condition_1 R J K →
  Roy_age_condition_2 R J K →
  Roy_age_condition_3 R J K →
  ratio_of_age_differences R J K = 2 :=
by
  sorry

end age_difference_ratio_l142_142852


namespace eccentricity_range_of_hyperbola_l142_142710

open Real

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def eccentricity_range :=
  ∀ (a b c : ℝ), 
    ∃ (e : ℝ),
      hyperbola a b (-c) 0 ∧ -- condition for point F
      (a + b > 0) ∧ -- additional conditions due to hyperbola properties
      (1 < e ∧ e < 2)
      
theorem eccentricity_range_of_hyperbola :
  eccentricity_range :=
by
  sorry

end eccentricity_range_of_hyperbola_l142_142710


namespace cube_root_of_xy_l142_142391

theorem cube_root_of_xy (x y : ℝ) (h1 : y = -x) (h2 : 3 * x - 4 * y = 7) : real.cbrt (x * y) = -1 :=
by sorry

end cube_root_of_xy_l142_142391


namespace weight_difference_l142_142921

theorem weight_difference : 
  ∀ (n : ℕ) (W : ℕ) (joe_wt mary_wt sam_wt : ℕ), 
  joe_wt = 42 →
  mary_wt = 36 →
  sam_wt = 48 →
  W = 30 * n →
  W + 42 + 36 + 48 = 31.5 * (n + 3) →
  let total_weight_after_leaving := 600 in
  let combined_weight_lost := (30 * n) + 42 + 36 + 48 - total_weight_after_leaving in
  combined_weight_lost - (joe_wt + mary_wt + sam_wt) = 30 :=
by
  sorry

end weight_difference_l142_142921


namespace subscription_monthly_cost_l142_142545

theorem subscription_monthly_cost (annual_cost_half : ℕ) (split_cost_evenly : ℕ) (monthly_duration: ℕ): 
  annual_cost_half = 84 → 
  split_cost_evenly = 2 → 
  monthly_duration = 12 → 
  (annual_cost_half * split_cost_evenly) / monthly_duration = 14 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end subscription_monthly_cost_l142_142545


namespace pirate_treasure_l142_142838

theorem pirate_treasure (x : ℕ) (hx : x = 3) :
  let paul_coins := x
  let pete_coins := x^2
  let total_coins := paul_coins + pete_coins
  total_coins = 12 :=
by
  -- considering Paul's and Pete's coins in terms of x
  let paul_coins := x,
      pete_coins := 3 * paul_coins
  -- given x = 3, we need to demonstrate the total number of coins equals 12
  have h_paul_coins : paul_coins = x := rfl,
  have h_pete_coins : pete_coins = 3 * x := rfl,
  have h_total_coins : total_coins = paul_coins + pete_coins,
  subst hx,
  have h_paul : 3 = 3 := rfl,
  calc
    total_coins = 4 * 3
              ... = 12


end pirate_treasure_l142_142838


namespace inscribed_circle_radius_l142_142059

theorem inscribed_circle_radius (r1 r2 r3 : ℝ) (h1 : r1 > r2) (h2 : r1 > r3) :
  ∃ r : ℝ, (quadrilateral_can_be_inscribed S1 S2 S3 r1 r2 r3) ∧ 
            r = r1 * r2 * r3 / (r1 * r3 + r1 * r2 - r2 * r3) :=
by sorry

end inscribed_circle_radius_l142_142059


namespace bees_12_feet_apart_l142_142530

structure Position where
  x : ℝ
  y : ℝ
  z : ℝ

def moveBeeA (pos : Position) : Position :=
  { pos with y := pos.y + 2, z := pos.z + 3 }

def moveBeeB (pos : Position) : Position :=
  { pos with y := pos.y - 1, x := pos.x - 2, z := pos.z - 1 }

noncomputable def distance (p1 p2 : Position) : ℝ :=
  ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2).sqrt

theorem bees_12_feet_apart :
  ∃ t : ℕ, let posA := (nth (iterate moveBeeA {x := 0, y := 0, z := 0}) t) in
           let posB := (nth (iterate moveBeeB {x := 0, y := 0, z := 0}) t) in
           distance posA posB = 12 :=
sorry

end bees_12_feet_apart_l142_142530


namespace probability_even_sum_of_two_cards_l142_142289

/-- Given 9 cards labeled with numbers 1, 2, 3, 4, 5, 6, 7, 8, 9, 
    the probability that the sum of the numbers on any two drawn cards is even is 4/9.
-/
theorem probability_even_sum_of_two_cards : 
  let cards := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let number_of_ways_to_pick_2 := (finset.univ.card.choose 2)
  let odd_cards := [1, 3, 5, 7]
  let even_cards := [2, 4, 6, 8, 9]
  let number_of_ways_pick_2_odd := (finset.ops.image_embedding (finset.image_embedding_of_injective cards _)) odd_cards).card.choose 2
  let number_of_ways_pick_2_even := (finset.ops.image_embedding (finset.image_embedding_of_injective cards _)) even_cards).card.choose 2
  let favorable_outcomes := number_of_ways_pick_2_odd + number_of_ways_pick_2_even
  let total_outcomes := number_of_ways_to_pick_2
  let probability := favorable_outcomes / total_outcomes
  probability = 4 / 9 :=
by
  -- here goes the proof
  sorry

end probability_even_sum_of_two_cards_l142_142289


namespace winning_strategy_l142_142107

variables {a b : ℝ} (h_ab : a ≠ b)

def volume_A := a^3
def volume_B := a^2 * b
def volume_C := b^2 * a
def volume_D := b^3

theorem winning_strategy : ∃! (strategy : finset ℕ), strategy = {1, 4} :=
by
  -- Calculate and compare volumes for different strategies to determine the unique winning strategy.
  sorry

end winning_strategy_l142_142107


namespace sixty_pair_is_5_7_l142_142717

-- Definition of the sequence conditions
def sequence_pair (n : ℕ) : ℕ × ℕ :=
  let sum := (2 * ⟨n, n - 1⟩.sqrt\<^image 2 + 1) in -- This line calculates the sum s such that sum covers the nth position.
  let pos := (n - sum * (sum - 1) / 2) in -- This calculates the position within the sum group.
  (pos, sum + 1 - pos) -- Provides the actual pair given the position within the sum group.

-- Statement to prove the 60th pair
theorem sixty_pair_is_5_7 : sequence_pair 60 = (5, 7) :=
  sorry

end sixty_pair_is_5_7_l142_142717


namespace escalator_speed_l142_142208

theorem escalator_speed (v : ℝ) :
  (v + 3) * 9 = 126 → v = 11 :=
by {
  intro h,
  -- From (v + 3) * 9 = 126, we solve for v
  have h1 : 9 * v + 27 = 126, from Eq.trans (by ring) h,
  have h2 : 9 * v = 99, from Eq.subst (by ring) h1,
  have h3 : v = 99 / 9, from (Eq.symm $ eq_div_of_mul_eq h2),
  rw [h3, div_eq_iff_mul_eq (by norm_num : 9 ≠ 0)],
  norm_num
}

end escalator_speed_l142_142208


namespace divisor_exists_l142_142118

-- Define the initial conditions
def all_natural_numbers := { n : ℕ | 1 ≤ n ∧ n ≤ 2014 }

-- Define the remaining set after erasing 1006 numbers
def remaining_numbers (S : set ℕ) (h : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : Prop :=
  ∃ a b ∈ S, a ≠ b ∧ a ∣ b

-- The problem statement in Lean
theorem divisor_exists (S : set ℕ) (h_sub : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : 
  remaining_numbers S h_sub h_card :=
begin
  sorry
end

end divisor_exists_l142_142118


namespace third_bouquet_carnations_l142_142907

/--
Trevor buys three bouquets of carnations. The first included 9 carnations, and the second included 14 carnations. If the average number of carnations in the bouquets is 12, then the third bouquet contains 13 carnations.
-/
theorem third_bouquet_carnations (n1 n2 n3 : ℕ)
  (h1 : n1 = 9)
  (h2 : n2 = 14)
  (h3 : (n1 + n2 + n3) / 3 = 12) :
  n3 = 13 :=
by
  sorry

end third_bouquet_carnations_l142_142907


namespace max_na_value_l142_142028

def A : Set ℤ :=
  {a | ∀ (i : Nat), a = i ∧ i ≥ 1 ∧ i ≤ 7}

def n_A (A : Set ℤ) : Nat :=
  (A.cartesian_product A).count (λ ⟨x, y⟩, x < y ∧ ∃ z, z ∈ A ∧ x + y = z)

theorem max_na_value : ∃ A : Set ℤ, (∀ a, a ∈ A → ∃ i:ℤ, a = i ∧ 1 ≤ i ∧ i ≤ 7) ∧ n_A A = 1880 :=
by
  sorry

end max_na_value_l142_142028


namespace unique_distribution_l142_142520

-- State representation as a tuple of balls in each box
structure State (n : Nat) where
  a : Fin (n + 1) → Nat
  deriving Repr

def initialState (n : Nat) : State n :=
  ⟨fun i => if i = 1 then n else 0⟩

def adjustedState (s : State n) (k : Nat) : State n :=
  if s.a ⟨k, Nat.lt_of_le_and_ne k.2 (Nat.succ_ne_zero k)⟩ = k then
    ⟨fun i => 
       if i = 0 then s.a 0 + 1
       else if i ≤ k - 1 then s.a i + 1 
       else if i = k then 0
       else s.a i⟩
  else s

theorem unique_distribution (n : Nat) (h : n ≥ 3) :
  ∃ unique (s : State n), ∀ (s' : State n), 
    (∃ m, Function.iterate adjustedState m.initialState = s') → 
    (Function.iterate adjustedState m.initialState = s) →
    ∃ m, Function.iterate adjustedState (m + s.a 0) s = initialState n :=
by {
  sorry
}

end unique_distribution_l142_142520


namespace tangent_line_ex_l142_142019

open Classical

variable {α : Type} [EuclideanGeometry α]

-- Let ω be a circle
variable (ω : Circle α)

-- Let C be a point outside ω
variable (C : Point α) (hC_outside : ¬ C ∈ ω)

-- Distinct points A and B are selected on ω 
variable (A B : Point α) (hA_neq_B : A ≠ B) (hA_on_ω : A ∈ ω) (hB_on_ω : B ∈ ω)

-- so that CA and CB are tangent to ω
variable (hCA_tgt_ω : Tangent C A ω)
variable (hCB_tgt_ω : Tangent C B ω)

-- Let X be the reflection of A across B
def X : Point α := reflection A B

-- Let γ be the circumcircle of triangle BXC
def γ : Circle α := circumcircle B X C

-- Suppose γ and ω meet at D ≠ B
variable (D : Point α) (hD_in_γ : D ∈ γ) (hD_in_ω : D ∈ ω) (hD_neq_B : D ≠ B)

-- Line CD intersects ω at E ≠ D
variable (E : Point α) (hE_on_ω : E ∈ ω) (hE_on_CD : E ∈ line_through C D) (hE_neq_D : E ≠ D)

-- Prove that line EX is tangent to the circle γ
theorem tangent_line_ex (hX_on_γ : X ∈ γ) (hE_not_on_X : E ≠ X) : Tangent E X γ := 
sorry

end tangent_line_ex_l142_142019


namespace proposition3_l142_142409

noncomputable def line_parallel_line {a b : Type} [AffineSpace ℝ a] : Prop :=
  parallel a b

noncomputable def line_perpendicular_plane {l : Type} [AffineSpace ℝ l] {α : Type} [AffineSpace ℝ α] : Prop :=
  perpendicular l α

noncomputable def plane_perpendicular_plane {α β : Type} [AffineSpace ℝ α] [AffineSpace ℝ β] : Prop :=
  perpendicular α β

theorem proposition3 (a b l : Type) [AffineSpace ℝ a] [AffineSpace ℝ b] [AffineSpace ℝ l] (α : Type) [AffineSpace ℝ α]
  (ha : line_parallel_line a b) (hl : line_perpendicular_plane l α) : 
  line_perpendicular_plane l b := 
sorry

end proposition3_l142_142409


namespace integer_solutions_of_quadratic_eq_l142_142032

theorem integer_solutions_of_quadratic_eq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ x1 x2 : ℤ, x1 * x2 = q^4 ∧ x1 + x2 = -p ∧ x1 = -1 ∧ x2 = - (q^4) ∧ p = 17 ∧ q = 2 := 
sorry

end integer_solutions_of_quadratic_eq_l142_142032


namespace video_has_2600_dislikes_l142_142572

def likes := 3000
def initial_dislikes := 1500 + 100
def additional_dislikes := 1000
def total_dislikes := initial_dislikes + additional_dislikes

theorem video_has_2600_dislikes:
  total_dislikes = 2600 :=
by
  unfold likes initial_dislikes additional_dislikes total_dislikes
  sorry

end video_has_2600_dislikes_l142_142572


namespace ratio_of_averages_l142_142201

theorem ratio_of_averages (x : Fin 50 → ℝ) :
  let true_average := (∑ i, x i) / 50
  new_average := (∑ i, x i + 2 * true_average) / 52
  true_average = new_average :=
by
  sorry

end ratio_of_averages_l142_142201


namespace divisor_exists_l142_142121

-- Define the initial conditions
def all_natural_numbers := { n : ℕ | 1 ≤ n ∧ n ≤ 2014 }

-- Define the remaining set after erasing 1006 numbers
def remaining_numbers (S : set ℕ) (h : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : Prop :=
  ∃ a b ∈ S, a ≠ b ∧ a ∣ b

-- The problem statement in Lean
theorem divisor_exists (S : set ℕ) (h_sub : S ⊆ all_natural_numbers) (h_card : S.card = 1008) : 
  remaining_numbers S h_sub h_card :=
begin
  sorry
end

end divisor_exists_l142_142121


namespace num_divisible_by_7_200_to_400_l142_142369

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l142_142369


namespace maddy_credits_to_graduate_l142_142459

theorem maddy_credits_to_graduate (semesters : ℕ) (credits_per_class : ℕ) (classes_per_semester : ℕ)
  (semesters_eq : semesters = 8)
  (credits_per_class_eq : credits_per_class = 3)
  (classes_per_semester_eq : classes_per_semester = 5) :
  semesters * (classes_per_semester * credits_per_class) = 120 :=
by
  -- Placeholder for proof
  sorry

end maddy_credits_to_graduate_l142_142459


namespace vertex_of_parabola_l142_142874

theorem vertex_of_parabola (a b c : ℝ) :
  (∀ x y : ℝ, (x = -2 ∧ y = 5) ∨ (x = 4 ∧ y = 5) ∨ (x = 2 ∧ y = 2) →
    y = a * x^2 + b * x + c) →
  (∃ x_vertex : ℝ, x_vertex = 1) :=
by
  sorry

end vertex_of_parabola_l142_142874


namespace parallel_lines_m_zero_perpendicular_lines_m_values_l142_142344

-- Define the lines l1 and l2
def l1 (x y : ℝ) (m : ℝ): Prop := x + m * y + 6 = 0
def l2 (x y : ℝ) (m : ℝ): Prop := (m - 2) * x + 3 * m * y + 18 = 0

-- Parallel condition definition
def parallel (l1 l2 : ℝ → ℝ → Prop): Prop :=
  ∀ (x y : ℝ), l1 x y = 0 → l2 x y = 0

-- Perpendicular condition definition
def perpendicular (l1 l2 : ℝ → ℝ → Prop): Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ (∀ (x y : ℝ), l1 x y = 0 → l2 x y = 0)

-- Theorem for the parallel case
theorem parallel_lines_m_zero (m : ℝ) :
  (parallel (l1 x y) (l2 x y)) → m = 0 :=
by
  sorry

-- Theorem for the perpendicular case
theorem perpendicular_lines_m_values (m : ℝ) :
  (perpendicular (l1 x y) (l2 x y)) → (m = -1 ∨ m = 2 / 3) :=
by
  sorry

end parallel_lines_m_zero_perpendicular_lines_m_values_l142_142344


namespace avg_remaining_two_is_correct_avg_new_set_is_correct_l142_142487

-- Definitions derived from the conditions
def avg_all_numbers := 4.60
def total_numbers := 10
def sum_all_numbers := avg_all_numbers * total_numbers

def avg_first_three := 3.4
def num_first_three := 3
def sum_first_three := avg_first_three * num_first_three

def avg_next_two := 3.8
def num_next_two := 2
def sum_next_two := avg_next_two * num_next_two

def avg_another_three := 4.2
def num_another_three := 3
def sum_another_three := avg_another_three * num_another_three

def sum_known_eight := sum_first_three + sum_next_two + sum_another_three
def sum_remaining_two := sum_all_numbers - sum_known_eight
def avg_remaining_two := sum_remaining_two / 2

def new_set_size := 3
def sum_new_set := avg_all_numbers * new_set_size
def avg_new_set := sum_new_set / new_set_size

-- Lean statements for the proof problems
theorem avg_remaining_two_is_correct : avg_remaining_two = 7.8 := by
  sorry

theorem avg_new_set_is_correct : avg_new_set = 4.60 := by
  sorry

end avg_remaining_two_is_correct_avg_new_set_is_correct_l142_142487


namespace range_of_m_l142_142332

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, 4^x + m * 2^x + m^2 - 1 = 0) ↔ - (2 * real.sqrt 3 / 3) ≤ m ∧ m < 1 :=
sorry

end range_of_m_l142_142332


namespace coefficient_x3_expansion_l142_142631

/--
Prove that the coefficient of \(x^{3}\) in the expansion of \(( \frac{x}{\sqrt{y}} - \frac{y}{\sqrt{x}})^{6}\) is \(15\).
-/
theorem coefficient_x3_expansion (x y : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ (x / y.sqrt - y / x.sqrt) ^ 6 = c * x ^ 3) :=
sorry

end coefficient_x3_expansion_l142_142631


namespace incenter_and_circumcenter_parallel_l142_142775

open EuclideanGeometry

-- Definitions based on the given problem
variables {A B C D E F I1 I2 O1 O2 : Point}

-- Proof statement based on the given conditions
theorem incenter_and_circumcenter_parallel
  (h_acute: Triangle ABC)
  (h_D: foot A B C D)
  (h_E: foot B C A E)
  (h_F: foot C A B F)
  (h_I1: incenter I1 A E F)
  (h_I2: incenter I2 B D F)
  (h_O1: circumcenter O1 A C I1)
  (h_O2: circumcenter O2 B C I2):
  parallel (line_through I1 I2) (line_through O1 O2) :=
sorry

end incenter_and_circumcenter_parallel_l142_142775


namespace increasing_intervals_l142_142264

noncomputable def f (x : ℝ) : ℝ :=
  sin (π / 3 - 1 / 2 * x)

theorem increasing_intervals :
  ∀ x : ℝ, -2 * π ≤ x ∧ x ≤ 2 * π →
  ( ( -2 * π ≤ x ∧ x ≤ -π / 3 ) ∨ ( 5 * π / 3 ≤ x ∧ x ≤ 2 * π ) ) →
  f x = sin (π / 3 - 1 / 2 * x) :=
by sorry

end increasing_intervals_l142_142264


namespace monkeys_more_than_giraffes_l142_142230

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end monkeys_more_than_giraffes_l142_142230


namespace number_of_divisibles_by_7_l142_142351

theorem number_of_divisibles_by_7 (a b : ℕ) (h1 : a = 200) (h2 : b = 400) : 
  (nat.card {n | a ≤ 7 * n ∧ 7 * n ≤ b}) = 29 := 
by sorry

end number_of_divisibles_by_7_l142_142351


namespace greatest_three_digit_multiple_of_17_l142_142535

theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ≤ 999 ∧ n ≥ 100 ∧ (∃ k : ℕ, n = 17 * k) ∧ 
  (∀ m : ℕ, m ≤ 999 → m ≥ 100 → (∃ k : ℕ, m = 17 * k) → m ≤ n) ∧ n = 986 := 
sorry

end greatest_three_digit_multiple_of_17_l142_142535


namespace complex_point_on_imaginary_axis_l142_142706

noncomputable def a : ℝ := sorry
noncomputable def z := (a + complex.i) * (1 + a * complex.i)

theorem complex_point_on_imaginary_axis :
  ∃ y : ℝ, z = complex.mk 0 y :=
by sorry

end complex_point_on_imaginary_axis_l142_142706


namespace plant_branches_l142_142175

theorem plant_branches (x : ℕ) (h : 1 + x + x^2 = 91) : 1 + x + x^2 = 91 :=
by sorry

end plant_branches_l142_142175


namespace balance_difference_l142_142641

theorem balance_difference :
  let Pv := 15000
  let r_EV := 0.04
  let Pf := 15000
  let r_FR := 0.06
  let n := 20 in
  let Av := Pv * (1 + r_EV)^n in
  let Af := Pf * (1 + n * r_FR) in
  abs (Av - Af) = 133 :=
by {
  let Pv := 15000
  let r_EV := 0.04
  let Pf := 15000
  let r_FR := 0.06
  let n := 20
  let Av := Pv * (1 + r_EV)^n
  let Af := Pf * (1 + n * r_FR)
  sorry
}

end balance_difference_l142_142641


namespace base4_product_132_12_l142_142917

def base4_mult (n m : ℕ) : ℕ := (n * m)

def to_base4 (n : ℕ) : string :=
  let rec divmod (n : ℕ) (acc : string) :=
    if n = 0 then acc else
    let (q, r) := Nat.divMod n 4
    divmod q (to_string r ++ acc)
  divmod n ""

theorem base4_product_132_12 (h1 : to_base4 30 = "132") (h2 : to_base4 6 = "12"):
  to_base4 (base4_mult 30 6) = "2310" := by
  sorry

end base4_product_132_12_l142_142917


namespace youtube_dislikes_l142_142574

theorem youtube_dislikes (likes : ℕ) (initial_dislikes : ℕ) (final_dislikes : ℕ) :
  likes = 3000 →
  initial_dislikes = likes / 2 + 100 →
  final_dislikes = initial_dislikes + 1000 →
  final_dislikes = 2600 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end youtube_dislikes_l142_142574


namespace percent_students_like_donuts_l142_142465

theorem percent_students_like_donuts (d : ℕ) (s : ℕ) (e : ℕ) (p : ℕ) :
  d = 4 * 12 → s = 30 → e = 2 → p = ((d / e) * 100) / s → p = 80 :=
by 
  intros hd hs he hp
  rw [hd, hs, he] at hp
  calc
    p = ((4 * 12 / 2) * 100) / 30 : hp
    ... = (24 * 100) / 30 : by norm_num
    ... = 2400 / 30 : by rw mul_comm
    ... = 80 : by norm_num

end percent_students_like_donuts_l142_142465


namespace number_of_divisibles_by_7_l142_142350

theorem number_of_divisibles_by_7 (a b : ℕ) (h1 : a = 200) (h2 : b = 400) : 
  (nat.card {n | a ≤ 7 * n ∧ 7 * n ≤ b}) = 29 := 
by sorry

end number_of_divisibles_by_7_l142_142350


namespace first_day_painted_l142_142615

theorem first_day_painted (days : ℕ) (h1 : days = 28 ∨ days = 30 ∨ days = 31)
  (painted : finset ℕ) (h2 : painted.card = 3)
  (h3 : ∀ (i j : ℕ), i ∈ painted → j ∈ painted → abs (i - j) ≠ 1)
  (unpainted_segments : finset ℕ)
  (h4 : ∀ segment ∈ unpainted_segments, 
          ∃ n, segment = n ∧ ∀ m, m ∈ segment → ((m < 10 ∧ m < 10) ∨ (m ≥ 10 ∧ m ≥ 10))) :
  1 ∈ painted :=
by sorry

end first_day_painted_l142_142615


namespace unique_solution_l142_142038

noncomputable def functional_condition (c : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f((c + 1) * x + f y) = f (x + 2 * y) + 2 * c * x

theorem unique_solution (c : ℝ) (f : ℝ → ℝ) (hf : functional_condition c f) (hc : c > 0) :
  ∀ x : ℝ, x > 0 → f x = 2 * x :=
sorry

end unique_solution_l142_142038


namespace postage_problem_l142_142277

theorem postage_problem (n : ℕ) (h1 : unlimited_supply_of_stamps_of_denominations 7 n (n + 2)) (h2 : greatest_postage_that_cannot_be_formed_is 104 7 n (n + 2)) :
  ∑ m in {m : ℕ | unlimited_supply_of_stamps_of_denominations 7 m (m + 2) ∧ greatest_postage_that_cannot_be_formed_is 104 7 m (m + 2), m} = 19 :=
by sorry

end postage_problem_l142_142277


namespace probability_succeeding_third_attempt_l142_142593

theorem probability_succeeding_third_attempt :
  let total_keys := 5
  let successful_keys := 2
  let attempts := 3
  let prob := successful_keys / total_keys * (successful_keys / (total_keys - 1)) * (successful_keys / (total_keys - 2))
  prob = 1 / 5 := by
sorry

end probability_succeeding_third_attempt_l142_142593


namespace quadratic_trinomial_sum_zero_discriminants_l142_142069

theorem quadratic_trinomial_sum_zero_discriminants (a b c : ℝ) :
  ∃ p q : ℝ → ℝ, (∀ x : ℝ, a * x ^ 2 + b * x + c = p x + q x) ∧
    (∀ x : ℝ, (let Δp := (d2p - d1p) ^ 2 - 4 * d0p * p2 in Δp = 0 ∧ 
                     Δq := (d2q - d1q) ^ 2 - 4 * d0q * q2 in Δq = 0)) :=
sorry

end quadratic_trinomial_sum_zero_discriminants_l142_142069


namespace quadratic_eq_has_distinct_real_roots_find_p_values_l142_142696

-- Definitions
def quadratic_eq (x p : ℝ) : Prop := (x - 3) * (x - 2) - p^2 = 0

-- Proposition 1: Prove that the quadratic equation always has two distinct real roots regardless of p.
theorem quadratic_eq_has_distinct_real_roots (p : ℝ) : 
  discriminant (λ x, x^2 - 5 * x + (6 - p^2)) > 0 :=
by 
  sorry

-- Proposition 2: Find the values of p for which the roots of the equation satisfy x1 = 4 * x2.
theorem find_p_values (p x1 x2 : ℝ) 
  (h_eq : quadratic_eq x1 p)
  (h_eq' : quadratic_eq x2 p)
  (h_sum : x1 + x2 = 5)
  (h_product : x1 * x2 = 6 - p^2)
  (h_relation : x1 = 4 * x2) : 
  p = sqrt 2 ∨ p = -sqrt 2 :=
by 
  sorry

end quadratic_eq_has_distinct_real_roots_find_p_values_l142_142696


namespace number_of_students_increased_l142_142899

theorem number_of_students_increased
  (original_number_of_students : ℕ) (increase_in_expenses : ℕ) (diminshed_average_expenditure : ℕ)
  (original_expenditure : ℕ) (increase_in_students : ℕ) :
  original_number_of_students = 35 →
  increase_in_expenses = 42 →
  diminshed_average_expenditure = 1 →
  original_expenditure = 420 →
  (35 + increase_in_students) * (12 - 1) - 420 = 42 →
  increase_in_students = 7 :=
by
  intros
  sorry

end number_of_students_increased_l142_142899


namespace vertex_of_f1_range_of_f1_expression_of_g_l142_142300

section
variable (x : ℝ)

-- Define the quadratic function with parameters b and c.
def quadratic (b c : ℝ) : ℝ := -x^2 + b*x + c

-- Part 1: Specific values of b and c.
def f1 : ℝ := quadratic 4 3 x

-- Problem 1: Prove that the vertex of f1 is (2, 7).
theorem vertex_of_f1 : ∃! (p : ℝ × ℝ), p = (2, 7) ∧ ∀ x, f1 ≤ f1 2 := sorry

-- Problem 2: Prove that the range of f1 on [-1, 3] is [-2, 7].
theorem range_of_f1 : ∀ y, (-2 ≤ y ∧ y ≤ 7) ↔ ∃ x, (-1 ≤ x ∧ x ≤ 3 ∧ y = f1) := sorry

-- Part 2: Given conditions on the function.
variable (y : ℝ)

-- Define a function to represent the conditions.
def g : ℝ := quadratic 2 2 x

-- When x ≤ 0, the maximum value is 2.
axiom max_value_x_leq_0 (x : ℝ) (h : x ≤ 0) : g ≤ 2

-- When x > 0, the maximum value is 3.
axiom max_value_x_gt_0 (x : ℝ) (h : x > 0) : g ≤ 3

-- Problem 3: Prove that the expression of the quadratic function is y = -x^2 + 2x + 2.
theorem expression_of_g : g = -x^2 + 2*x + 2 := sorry
end

end vertex_of_f1_range_of_f1_expression_of_g_l142_142300


namespace solve_trig_eq_l142_142546

theorem solve_trig_eq (x : ℝ) (k n : ℤ) :
    (cot x + tan (2 * x) + 1 = 4 * cos x ^ 2 + sin (3 * x) / sin x - 2 * cos (2 * x)) 
    ∧ (cos (2 * x) ≠ 0) 
    ∧ (sin x ≠ 0) → 
    (x = (π / 2) * (2 * k + 1) ∨ x = (π / 8) * (4 * n + 1)) :=
by 
    sorry

end solve_trig_eq_l142_142546


namespace pyramid_edges_correct_l142_142958

noncomputable def sum_of_pyramid_edges (l w h : ℝ) : ℝ :=
  let d := Real.sqrt (l^2 + w^2) in
  let slant_length := Real.sqrt (h^2 + (d / 2)^2) in
  2 * l + 2 * w + 4 * slant_length

theorem pyramid_edges_correct : sum_of_pyramid_edges 14 10 15 ≈ 117 :=
by
  sorry

end pyramid_edges_correct_l142_142958


namespace largest_k_consecutive_sum_l142_142267

theorem largest_k_consecutive_sum (k : ℕ) (h1 : (∃ n : ℕ, 3^12 = k * n + (k*(k-1))/2)) : k ≤ 729 :=
by
  -- Proof omitted for brevity
  sorry

end largest_k_consecutive_sum_l142_142267


namespace strength_order_l142_142106

-- Definitions of individual strengths as variables
variables A B C D : ℝ

-- Conditions provided in the problem
axiom cond1 : A + B = C + D
axiom cond2 : A + D > B + C
axiom cond3 : B > A + C

-- Theorem stating the order of strength from strongest to weakest
theorem strength_order : D > B ∧ B > A ∧ A > C := sorry

end strength_order_l142_142106


namespace goat_can_circle_around_tree_l142_142599

/-- 
  Given a goat tied with a rope of length 4.7 meters (L) near an old tree with a cylindrical trunk of radius 0.5 meters (R), 
  with the shortest distance from the stake to the surface of the tree being 1 meter (d), 
  prove that the minimal required rope length to encircle the tree and return to the stake is less than 
  or equal to the given rope length of 4.7 meters (L).
-/ 
theorem goat_can_circle_around_tree (L R d : ℝ) (hR : R = 0.5) (hd : d = 1) (hL : L = 4.7) : 
  ∃ L_min, L_min ≤ L := 
by
  -- Detailed proof steps omitted.
  sorry

end goat_can_circle_around_tree_l142_142599


namespace equality_one_equality_two_l142_142976

-- Proof Problem for Question 1
theorem equality_one (a : ℝ) : 
  ([-a^3 * (-a)^3]^2 + [-a^2 * (-a)^2]^3 = 0) :=
begin
  -- The proof would be placed here
  sorry
end

-- Proof Problem for Question 2
theorem equality_two (n k : ℤ) (a : ℝ) : 
  (-1)^n * a^(n + k) = (-a)^n * a^k :=
begin
  -- The proof would be placed here
  sorry
end

end equality_one_equality_two_l142_142976


namespace max_diff_in_grid_at_least_209_l142_142265

theorem max_diff_in_grid_at_least_209 :
  ∃ (N : ℕ), (∀ (grid : {g : ℕ → ℕ → ℕ // (∀ (i j : ℕ), g i j ≥ 1 ∧ g i j ≤ 400) ∧
    (∀ (i j : ℕ), i < 20 ∧ j < 20 → g i j ≠ g i (j + 1) ∧ g i j ≠ g (i + 1) j)}), 
    ∃ (r1 r2 : ℕ), r1 < 20 ∧ r2 < 20 ∧ (∃ (c1 c2 : ℕ), c1 < 20 ∧ c2 < 20 ∧ 
    (abs (grid.val r1 c1 - grid.val r2 c2) ≥ N ∨ abs (grid.val r1 c1 - grid.val r1 c2) ≥ N ∨ 
    abs (grid.val r2 c1 - grid.val r2 c2) ≥ N))) ∧ N = 209 :=
sorry

end max_diff_in_grid_at_least_209_l142_142265


namespace geometric_sequence_log_sum_l142_142891

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n m : ℕ, a (n + m + 1) = (a n) * (a m) 

theorem geometric_sequence_log_sum {a : ℕ → ℝ} (h : ∀ n, 0 < a n) 
  (h_geom : geometric_sequence a) (h_eq : a 5 * a 6 + a 4 * a 7 = 18) :
  ∑ i in Finset.range 10, Real.log 3 (a (i + 1)) = 10 :=
by
  sorry

end geometric_sequence_log_sum_l142_142891


namespace problem_statement_l142_142835

theorem problem_statement :
  (1 / 3 * 1 / 6 * P = (1 / 4 * 1 / 8 * 64) + (1 / 5 * 1 / 10 * 100)) → 
  P = 72 :=
by
  sorry

end problem_statement_l142_142835


namespace max_marks_l142_142152

theorem max_marks (total_marks : ℕ) (obtained_marks : ℕ) (failed_by : ℕ) 
    (passing_percentage : ℝ) (passing_marks : ℝ) (H1 : obtained_marks = 125)
    (H2 : failed_by = 40) (H3 : passing_percentage = 0.33) 
    (H4 : passing_marks = obtained_marks + failed_by) 
    (H5 : passing_marks = passing_percentage * total_marks) : total_marks = 500 := by
  sorry

end max_marks_l142_142152


namespace distance_M_to_AB_l142_142868

theorem distance_M_to_AB (M : Point) (A B C : Point) (d_AC d_BC d_AB : ℝ) 
    (h_d_AC : d_AC = 2) (h_d_BC : d_BC = 4) (h_AB : dist A B = 10) 
    (h_BC : dist B C = 17) (h_AC : dist A C = 21) : 
    d_AB = 29 / 5 := 
  sorry

end distance_M_to_AB_l142_142868


namespace finish_work_in_time_l142_142579

noncomputable def work_in_days_A (DA : ℕ) := DA
noncomputable def work_in_days_B (DA : ℕ) := DA / 2
noncomputable def combined_work_rate (DA : ℕ) : ℚ := 1 / work_in_days_A DA + 2 / work_in_days_A DA

theorem finish_work_in_time (DA : ℕ) (h_combined_rate : combined_work_rate DA = 0.25) : DA = 12 :=
sorry

end finish_work_in_time_l142_142579


namespace sum_of_coefficients_of_expansion_l142_142748

theorem sum_of_coefficients_of_expansion :
  let b : ℕ → ℤ := λ n, (2 ^ (5 - n) * 3 ^ n * Int.binomial 5 n)
  let sum := ∑ n in Finset.range 6, b n
  sum = 3125 :=
by
  sorry

end sum_of_coefficients_of_expansion_l142_142748


namespace youtube_dislikes_l142_142568

def initial_dislikes (likes : ℕ) : ℕ := (likes / 2) + 100

def new_dislikes (initial : ℕ) : ℕ := initial + 1000

theorem youtube_dislikes
  (likes : ℕ)
  (h_likes : likes = 3000) :
  new_dislikes (initial_dislikes likes) = 2600 :=
by
  sorry

end youtube_dislikes_l142_142568


namespace convex_ngons_tessellation_congruent_ngons_tessellation_l142_142154

theorem convex_ngons_tessellation (n : ℕ) (h : n ≥ 7) : 
  ∃ tessellation : Set (Set Point), (∀ g ∈ tessellation, convex g ∧ sides g = n) ∧ 
  (∀ g1 g2 ∈ tessellation, g1 ≠ g2 → ¬ congruent g1 g2) ∧ 
  covers_plane tessellation :=
sorry

theorem congruent_ngons_tessellation (n : ℕ) (h : n ≥ 7) : 
  ∃ tessellation : Set (Set Point), (∀ g ∈ tessellation, sides g = n) ∧ 
  (∀ g ∈ tessellation, congruent g (arbitrary_n_gon_with_sides n)) ∧ 
  covers_plane tessellation :=
sorry

end convex_ngons_tessellation_congruent_ngons_tessellation_l142_142154


namespace sara_has_8_balloons_l142_142526

-- Define the number of yellow balloons Tom has.
def tom_balloons : ℕ := 9 

-- Define the total number of yellow balloons.
def total_balloons : ℕ := 17

-- Define the number of yellow balloons Sara has.
def sara_balloons : ℕ := total_balloons - tom_balloons

-- Theorem stating that Sara has 8 yellow balloons.
theorem sara_has_8_balloons : sara_balloons = 8 := by
  -- Proof goes here. Adding sorry for now to skip the proof.
  sorry

end sara_has_8_balloons_l142_142526


namespace quadratic_trinomial_decomposition_l142_142067

theorem quadratic_trinomial_decomposition (a b c : ℝ) :
  ∃ g h : ℝ → ℝ, (∀ x : ℝ, a*x^2 + b*x + c = g x + h x) ∧
                 (∃ g_discriminant, g_discriminant = 0) ∧
                 (∃ h_discriminant, h_discriminant = 0) :=
begin
  sorry
end

end quadratic_trinomial_decomposition_l142_142067


namespace option_C_increasing_sequence_l142_142206

def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem option_C_increasing_sequence :
  is_increasing_sequence (λ n, 2 * n^2 - 5 * n + 1) :=
sorry

end option_C_increasing_sequence_l142_142206


namespace addition_result_dot_product_result_l142_142732

-- Define the given vectors
def a : ℝ × ℝ × ℝ := (1, 2, 2)
def b : ℝ × ℝ × ℝ := (6, -3, 2)

-- Proof of the first correct option
theorem addition_result :
  (a.1 + b.1, a.2 + b.2, a.3 + b.3) = (7, -1, 4) :=
sorry

-- Proof of the second correct option
theorem dot_product_result :
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 4 :=
sorry

end addition_result_dot_product_result_l142_142732


namespace modified_cube_surface_area_l142_142623

noncomputable def total_surface_area_modified_cube : ℝ :=
  let side_length := 10
  let triangle_side := 7 * Real.sqrt 2
  let tunnel_wall_area := 3 * (Real.sqrt 3 / 4 * triangle_side^2)
  let original_surface_area := 6 * side_length^2
  original_surface_area + tunnel_wall_area

theorem modified_cube_surface_area : 
  total_surface_area_modified_cube = 600 + 73.5 * Real.sqrt 3 := 
  sorry

end modified_cube_surface_area_l142_142623


namespace evaluate_monomial_l142_142651

theorem evaluate_monomial (a b : ℤ) (h₁ : a = -5) (h₂ : b = 2) :
  0.007 * a^7 * b^9 = -280000 :=
by
  sorry

end evaluate_monomial_l142_142651


namespace percentage_discount_l142_142961

theorem percentage_discount (C S S' : ℝ) (h1 : S = 1.14 * C) (h2 : S' = 2.20 * C) :
  (S' - S) / S' * 100 = 48.18 :=
by 
  sorry

end percentage_discount_l142_142961


namespace lilies_purchasable_l142_142177

theorem lilies_purchasable (cost_chrysanthemum cost_lily total_money num_chrysanthemums remaining_money : ℕ) :
  cost_chrysanthemum = 3 → cost_lily = 4 → total_money = 100 → num_chrysanthemums = 16 → 
  remaining_money = total_money - (cost_chrysanthemum * num_chrysanthemums) → (remaining_money / cost_lily) = 13 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  have total_cost := (3 * 16)
  have rem_money := (100 - total_cost)
  rw [h5] at rem_money
  simp at rem_money
  sorry

end lilies_purchasable_l142_142177


namespace line_equiv_intersect_point_l142_142588

section

def line1 (x y : ℝ) : Prop := (3 : ℝ) * (x - 2) - (4 : ℝ) * (y - 3) = 0
def line2 (x : ℝ) : ℝ := (3 / 4 : ℝ) * x + (3 / 2 : ℝ)
def line3 (x : ℝ) : ℝ := -x + 7

theorem line_equiv :
  ∀ x y : ℝ, line1 x y ↔ y = line2 x :=
by
  intros x y
  unfold line1 line2
  sorry

theorem intersect_point :
  ∃ x y : ℝ, x = 3.142857 ∧ y = line2 x ∧ y = line3 x :=
by
  unfold line2 line3
  existsi (3.142857 : ℝ), (3.857143 : ℝ)
  sorry

end

end line_equiv_intersect_point_l142_142588


namespace intersection_divides_square_into_four_congruent_triangles_l142_142235

noncomputable def pt_A : ℝ × ℝ := (0, 0)
noncomputable def pt_B : ℝ × ℝ := (0, 4)
noncomputable def pt_C : ℝ × ℝ := (4, 4)
noncomputable def pt_D : ℝ × ℝ := (4, 0)

def line_from_A : ℝ → ℝ := λ x, x
def line_from_B : ℝ → ℝ := λ x, 4 - x

theorem intersection_divides_square_into_four_congruent_triangles :
  line_from_A 2 = 2 ∧ line_from_B 2 = 2 → 
  ∃ intersect_point : ℝ × ℝ, intersect_point = (2, 2) ∧
  (∀ t1 t2 t3 t4 : ℝ × ℝ × ℝ, (t1 = ((0,0), (2,2), (2,0)) ∧
  t2 = ((0,4), (2,2), (0,2)) ∧
  t3 = ((4,4), (2,2), (4,2)) ∧
  t4 = ((4,0), (2,2), (2,0)) →
  triangle t1 ∧ triangle t2 ∧ triangle t3 ∧ triangle t4 ∧ 
  congruent_triangles t1 t2 ∧ congruent_triangles t2 t3 ∧ congruent_triangles t3 t4 ∧ congruent_triangles t4 t1)) := 
by
  intro h
  rw [line_from_A, line_from_B] at h
  use (2, 2)
  split
  repeat {assumption}
  sorry

end intersection_divides_square_into_four_congruent_triangles_l142_142235


namespace reassemble_square_with_hole_l142_142602

theorem reassemble_square_with_hole 
  (a b c d k1 k2 : ℝ)
  (h1 : a = b)
  (h2 : c = d)
  (h3 : k1 = k2) :
  ∃ (f gh ef gh' : ℝ), 
    f = a - c ∧
    gh = b - d ∧
    ef = f ∧
    gh' = gh := 
by sorry

end reassemble_square_with_hole_l142_142602


namespace sum_of_digits_9ab_l142_142776

theorem sum_of_digits_9ab (a b : ℕ) (h1: a = 8 * (10^1985 - 1) / 9) (h2: b = 5 * (10^1985 - 1) / 9) :
  let ab := a * b in let result := 9 * ab in
  sum_of_digits result = 17865 :=
by
  sorry

end sum_of_digits_9ab_l142_142776


namespace forward_voltage_on_reverse_voltage_off_l142_142613

def diode_state (I U : ℝ) : Prop :=
  I = 10^(-13) * (Real.exp (U / 0.026) - 1)

def diode_is_on (U : ℝ) : Prop :=
  let I := 10^(-13) * (Real.exp (U / 0.026) - 1)
  |I| ≥ 10^(-7)

def diode_is_off (U : ℝ) : Prop :=
  let I := 10^(-13) * (Real.exp (U / 0.026) - 1)
  |I| < 10^(-7)

theorem forward_voltage_on : diode_is_on 0.78 :=
sorry  -- proof omitted

theorem reverse_voltage_off : diode_is_off (-0.78) :=
sorry  -- proof omitted

end forward_voltage_on_reverse_voltage_off_l142_142613


namespace max_min_sum_eq_neg2_l142_142722

noncomputable def f (x : ℝ) : ℝ := (x^3) / (x^2 + 2) - 1

theorem max_min_sum_eq_neg2 : 
  let I := set.Icc (-2023 : ℝ) 2023
  let M := real.Sup (f '' I)
  let m := real.Inf (f '' I)
  in M + m = -2 :=
begin
  sorry
end

end max_min_sum_eq_neg2_l142_142722


namespace sum_gcd_lcm_75_4410_l142_142144

theorem sum_gcd_lcm_75_4410 :
  Nat.gcd 75 4410 + Nat.lcm 75 4410 = 22065 := by
  sorry

end sum_gcd_lcm_75_4410_l142_142144


namespace arithmetic_sequence_50th_term_l142_142966

-- Definitions as per the conditions
def a_1 : ℤ := 48
def d : ℤ := -2
def n : ℕ := 50

-- Statement to prove the 50th term in the series
theorem arithmetic_sequence_50th_term : a_1 + (n - 1) * d = -50 :=
by
  sorry

end arithmetic_sequence_50th_term_l142_142966


namespace unique_positive_real_solution_l142_142662

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l142_142662


namespace sum_of_circumferences_l142_142480

-- Conditions
def radius_first_circle : ℝ := 1
def area_fourth_circle : ℝ := 64 * Real.pi

-- Constants
noncomputable def radius_second_circle : ℝ := 2
noncomputable def radius_third_circle : ℝ := 4

-- Definition of circumference
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Theorem statement
theorem sum_of_circumferences : 
  circumference radius_second_circle + circumference radius_third_circle = 12 * Real.pi :=
by 
  sorry

end sum_of_circumferences_l142_142480


namespace unique_positive_solution_l142_142658

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l142_142658


namespace value_of_a_l142_142250

-- Defining the hyperbola condition.
def is_hyperbola (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x y : ℝ, (x^2 / a^2) - (y^2 / 9) = 1

-- Defining the asymptote condition.
def is_asymptote (a : ℝ) : Prop :=
  y = (3 / 5) * x

-- Theorem statement
theorem value_of_a {a : ℝ} (h1 : is_hyperbola a) (h2 : is_asymptote a) : a = 5 :=
sorry

end value_of_a_l142_142250


namespace count_multiples_of_7_l142_142358

theorem count_multiples_of_7 (low high : ℕ) (hlow : low = 200) (hhigh : high = 400) : 
  (card {n | low ≤ n ∧ n ≤ high ∧ n % 7 = 0}) = 29 := by
  sorry

end count_multiples_of_7_l142_142358


namespace probability_odd_as_n_increases_l142_142578

-- Definitions based on the conditions
def digit := {n : ℕ // n ≤ 9}

def initial_display : ℕ := 0

def operations := {op : String // op = "+" ∨ op = "*"}

-- Probabilities computations
noncomputable def p_n (n : ℕ) : ℝ :=
  if n = 0 then 0
  else (let rec_prob (m : ℕ) : ℝ := 
          if m = 0 then 0
          else (1 / 4) * rec_prob (m - 1) + 1 / 4
        in rec_prob n)

-- Statement for the probability approximation as n approaches infinity
theorem probability_odd_as_n_increases {ε : ℝ} (hε : 0 < ε) :
  ∃ N : ℕ, ∀ n ≥ N, abs (p_n n - (1 / 3)) < ε :=
sorry

end probability_odd_as_n_increases_l142_142578


namespace f_at_5_l142_142450

def f : ℝ → ℝ := λ x, 3 * x^4 - 22 * x^3 + 51 * x^2 - 58 * x + 24

theorem f_at_5 : f 5 = 134 := 
  sorry

end f_at_5_l142_142450


namespace unique_positive_solution_l142_142656

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l142_142656


namespace pythagorean_triple_correct_answer_l142_142972

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def sets := [(12, 8, 5), (30, 40, 50), (9, 13, 15), (1/6, 1/8, 1/10)]

theorem pythagorean_triple_correct_answer :
 ∃ (a b c : ℕ), (a, b, c) ∈ sets ∧ is_pythagorean_triple a b c :=
begin
  use (30, 40, 50),
  split,
  { -- verifying the set is in the choices
    have h_mem : (30, 40, 50) ∈ sets := by { tautology },
    exact h_mem,
  },
  { -- verify if the set (30, 40, 50) forms a Pythagorean triple
    exactly is_pythagorean_triple 30 40 50,
  },
end

end pythagorean_triple_correct_answer_l142_142972


namespace perfect_squares_count_2000_l142_142742

theorem perfect_squares_count_2000 : 
  let count := (1 to 44).filter (λ x, 
    let ones_digit := (x * x) % 10 in 
    ones_digit = 4 ∨ ones_digit = 5 ∨ ones_digit = 6).length
  in
  count = 22 :=
by
  sorry

end perfect_squares_count_2000_l142_142742


namespace mary_change_in_dollars_l142_142048

theorem mary_change_in_dollars :
  let cost_berries_euros := 7.94
  let cost_peaches_dollars := 6.83
  let exchange_rate := 1.2
  let money_handed_euros := 20
  let money_handed_dollars := 10
  let cost_berries_dollars := cost_berries_euros * exchange_rate
  let total_cost_dollars := cost_berries_dollars + cost_peaches_dollars
  let total_handed_dollars := (money_handed_euros * exchange_rate) + money_handed_dollars
  total_handed_dollars - total_cost_dollars = 17.642 :=
by
  intros
  sorry

end mary_change_in_dollars_l142_142048


namespace total_students_l142_142637

/-- Definition of the problem's conditions as Lean statements -/
def left_col := 8
def right_col := 14
def front_row := 7
def back_row := 15

/-- The total number of columns calculated from Eunji's column positions -/
def total_columns := left_col + right_col - 1
/-- The total number of rows calculated from Eunji's row positions -/
def total_rows := front_row + back_row - 1

/-- Lean statement showing the total number of students given the conditions -/
theorem total_students : total_columns * total_rows = 441 := by
  sorry

end total_students_l142_142637


namespace probability_alex_wins_3_mel_2_chelsea_2_sam_1_8_rounds_l142_142202

noncomputable def probability_of_wins (a m c s : ℚ) (rounds_alex rounds_mel rounds_chelsea rounds_sam total_rounds : ℕ)
    (h1 : a = 1/3) (h2 : m = 3 * s) (h3 : c = s) (h4 : rounds_alex + rounds_mel + rounds_chelsea + rounds_sam = total_rounds) : ℚ :=
  let total_prob := (a ^ rounds_alex) * (m ^ rounds_mel) * (c ^ rounds_chelsea) * (s ^ rounds_sam) in
  let arrangements := Nat.factorial total_rounds / (Nat.factorial rounds_alex * Nat.factorial rounds_mel * Nat.factorial rounds_chelsea * Nat.factorial rounds_sam) in
  arrangements * total_prob

theorem probability_alex_wins_3_mel_2_chelsea_2_sam_1_8_rounds :
  probability_of_wins 1/3 (2/5 : ℚ) (2/15 : ℚ) (2/15 : ℚ) 3 2 2 1 8 (by norm_num) (by norm_num) (by norm_num) (by norm_num) = 13440 / 455625 :=
by sorry

end probability_alex_wins_3_mel_2_chelsea_2_sam_1_8_rounds_l142_142202


namespace problem_equivalent_proof_l142_142873

lemma reflect_arccos_y_eq_neg_cos_x (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) : 
  ∃ (y : ℝ), y = -cos x :=
begin
  use -cos x,
  sorry
end

theorem problem_equivalent_proof :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ π → ∃ y, y = -cos x :=
begin
  intros x h,
  exact reflect_arccos_y_eq_neg_cos_x x h,
end

end problem_equivalent_proof_l142_142873


namespace ellipse_standard_eq_max_area_triangle_l142_142701

-- Definitions from conditions
def is_ellipse (a b : Real) (x y : Real) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def focus1 (a b : Real) : (Real × Real) := (a * sqrt(1 - (b^2 / a^2)), 0)
def focus2 (a b : Real) : (Real × Real) := (-a * sqrt(1 - (b^2 / a^2)), 0)

def dist (P Q : (Real × Real)) : Real :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def total_dist (P F1 F2 : (Real × Real)) : Real :=
  dist P F1 + dist P F2

-- Conditions for existence of ellipse
axiom C_exists (P : Real × Real) (a b : Real) (F1 F2 : (Real × Real))
  (h1 : is_ellipse a b 1 (sqrt 2 / 2))
  (h2 : total_dist P F1 F2 = 2 * sqrt 2) :
  b^2 = 1

-- Prove the standard equation of ellipse
theorem ellipse_standard_eq {a b : Real}
  (h : a = sqrt 2 ∧ b = 1) :
  ∀ x y : Real, (x^2 / 2 + y^2 = 1) :=
sorry

-- Prove the maximum area of triangle
theorem max_area_triangle {a b : Real}
  (h1 : is_ellipse a b 1 (sqrt 2 / 2))
  (h2 : a = sqrt 2 ∧ b = 1) :
  ∃ (m : Real), m = 0 ∧ (1 / 2 * dist (0, 0) (focus2 a b) * (sqrt 2 / 2 - (-sqrt 2 / 2)) = sqrt 2 / 2) :=
sorry

end ellipse_standard_eq_max_area_triangle_l142_142701


namespace smallest_n_exists_l142_142276

theorem smallest_n_exists : 
  ∃ n : ℕ, 
  (0 < n) ∧ 
  (∀ m : ℕ, 
     (0 < m < 2004) → 
     ∃ k : ℤ, 
       (m : ℚ) / 2004 < (k : ℚ) / n ∧ 
       (k : ℚ) / n < (m + 1) / 2005) ∧ 
  (n = 4009) :=
sorry

end smallest_n_exists_l142_142276


namespace zero_in_interval_2_3_l142_142512

noncomputable def f (x : ℝ) : ℝ := log (x / 2) - 1 / x

theorem zero_in_interval_2_3 :
  (∃ x, 2 < x ∧ x < 3 ∧ f x = 0) :=
by {
  have h2 : f 2 = log 1 - 1 / 2 := by simp [f, log],
  have h2_neg : f 2 < 0 := by {
    rw [h2, log_one],
    norm_num,
  },
  have h3 : f 3 = log (3 / 2) - 1 / 3 := by simp [f],
  have h3_pos : f 3 > 0 := by {
    rw [h3],
    norm_num,
    exact log_pos (by norm_num : 3 / 2 > 1),
  },
  exact exists_Ioo_zero_of_continuous f (by norm_num : 2 < 3) h2_neg h3_pos,
  sorry,
}

end zero_in_interval_2_3_l142_142512


namespace original_number_is_seven_l142_142146

theorem original_number_is_seven (x : ℕ) (h : 3 * x - 5 = 16) : x = 7 := by
sorry

end original_number_is_seven_l142_142146


namespace height_difference_is_correct_l142_142436

-- Define the heights of the trees as rational numbers.
def maple_tree_height : ℚ := 10 + 1 / 4
def spruce_tree_height : ℚ := 14 + 1 / 2

-- Prove that the spruce tree is 19 3/4 feet taller than the maple tree.
theorem height_difference_is_correct :
  spruce_tree_height - maple_tree_height = 19 + 3 / 4 := 
sorry

end height_difference_is_correct_l142_142436


namespace correct_propositions_l142_142971

-- Propositions definitions
def prop1 := ∀ x: ℝ, y = log x.toReal (x - 3) + 2 ↔ y = log x.toReal x + 2 + 3
def prop2 := ∀ x: ℝ, f x = (2 * x - 3) / (x + 1) → ∃ x: ℝ, f x = 2 * (2 - (5 / (x + 1)))
def prop3 := ∀ x: ℝ, x > 0 → x ^ (1 / 2) > x
def prop4 := ∀ {f: ℝ → ℝ} x, ∀ y: ℝ, f x = y → (f x1 = y ∧ f x2 = y) → (x1 = x2)

-- Main proof problem statement
theorem correct_propositions: (prop1 ∧ prop4) := sorry

end correct_propositions_l142_142971


namespace mark_lisa_price_difference_zero_l142_142764

theorem mark_lisa_price_difference_zero :
  let sales_tax_rate := 0.075
  let original_price := 120.0
  let discount_rate := 0.30
  let mark_total := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
  let lisa_total := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)
  mark_total = lisa_total := by
  let sales_tax_rate := 0.075
  let original_price := 120.0
  let discount_rate := 0.30
  let mark_total := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
  let lisa_total := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)
  have h : mark_total = 90.30 := sorry
  have h2 : lisa_total = 90.30 := sorry
  exact eq.trans h h2

end mark_lisa_price_difference_zero_l142_142764


namespace estimate_blue_balls_l142_142408

theorem estimate_blue_balls (total_balls : ℕ) (prob_yellow : ℚ)
  (h_total : total_balls = 80)
  (h_prob_yellow : prob_yellow = 0.25) :
  total_balls * (1 - prob_yellow) = 60 :=
by
  -- proof
  sorry

end estimate_blue_balls_l142_142408


namespace initial_bacteria_population_l142_142581

theorem initial_bacteria_population 
  (double_time : ℕ)
  (final_time : ℕ)
  (final_population : ℕ) 
  (h_double_time : double_time = 30) 
  (h_final_time : final_time = 240) 
  (h_final_population : final_population = 524288) : 
  ∃ n, n * 2 ^ (final_time / double_time) = final_population ∧ n = 2048 := 
by 
  -- Import the proof variables and assumptions
  let total_doublings := final_time / double_time
  have h_total_doublings : total_doublings = 8 := by 
    rw [h_final_time, h_double_time]
    exact nat.div_eq_of_lt (dec_trivial : 240 = 8 * 30).symm 
    
  -- Prove the initial population calculation
  use 2048
  split
  { rw [h_final_population, h_total_doublings]
    norm_num   
  }
  { triv }

end initial_bacteria_population_l142_142581


namespace regular_prism_dihedral_angle_l142_142768

noncomputable def dihedral_angle_range (n : ℕ) (h : 2 ≤ n) : set ℝ := 
  set.Ioo ((n-2) * real.pi / n) real.pi

theorem regular_prism_dihedral_angle {n : ℕ} (h : 2 ≤ n) :
  range_of_dihedral_angle_between_adjacent_faces n = dihedral_angle_range n h :=
sorry

end regular_prism_dihedral_angle_l142_142768


namespace perimeter_DEC_l142_142002

theorem perimeter_DEC :
  (∀ {T: Type} (triangle : T → T → T → Prop) (equilateral : ∀ {A B C : T}, triangle A B C → (∀{a b c: ℝ}, a = b ∧ b = c)),
    ( ∃ {A K T : T}, triangle A K T ∧ (3 * 36 = 108) ) →
    ( ∃ {D E C : T}, triangle D E C ∧ (3 * 3 = 9) )) :=
begin
  sorry
end

end perimeter_DEC_l142_142002


namespace range_of_m_l142_142720

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then 1 / (Real.exp x) + m * x^2
else Real.exp x + m * x^2

theorem range_of_m {m : ℝ} : (∀ m, ∃ x y, f x m = 0 ∧ f y m = 0 ∧ x ≠ y) ↔ m < -Real.exp 2 / 4 := by
  sorry

end range_of_m_l142_142720


namespace determine_x_1000_l142_142643

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

def smallest_composite_greater_than (n : ℕ) : ℕ :=
  Nat.find (by {
    have h: ∃ k, k > n ∧ is_composite k := sorry,
    exact h
  })

noncomputable def sequence_x : ℕ → ℕ
| 1 := 4
| 2 := 6
| n + 1 := if h : 3 ≤ n + 1 then smallest_composite_greater_than (2 * sequence_x n - sequence_x (n - 1)) else 0

theorem determine_x_1000 : sequence_x 1000 = 501500 :=
sorry

end determine_x_1000_l142_142643


namespace base8_subtraction_l142_142619

-- Defining base 8 numbers
def n1 := 0b4325 -- 4325 in base 8
def n2 := 0b2377 -- 2377 in base 8
def n3 := 0b122  -- 122 in base 8
def result := 0b1714 -- 1714 in base 8

theorem base8_subtraction :
  n1 - n2 - n3 = result := 
by
  sorry

end base8_subtraction_l142_142619


namespace number_of_people_with_fewer_than_7_cards_l142_142395

theorem number_of_people_with_fewer_than_7_cards
  (total_cards : ℕ) (total_people : ℕ)
  (h_cards : total_cards = 60) (h_people : total_people = 9) :
  ∃ num_people_with_fewer_than_7_cards, num_people_with_fewer_than_7_cards = 3 :=
by
  use 3
  sorry

end number_of_people_with_fewer_than_7_cards_l142_142395


namespace cricket_target_runs_l142_142788

def run_rate_first_20_overs : ℝ := 4.2
def overs_first_20 : ℝ := 20
def run_rate_remaining_30_overs : ℝ := 5.533333333333333
def overs_remaining_30 : ℝ := 30
def total_runs_first_20 : ℝ := run_rate_first_20_overs * overs_first_20
def total_runs_remaining_30 : ℝ := run_rate_remaining_30_overs * overs_remaining_30

theorem cricket_target_runs :
  (total_runs_first_20 + total_runs_remaining_30) = 250 :=
by
  sorry

end cricket_target_runs_l142_142788


namespace arithmetic_progression_common_difference_l142_142207

theorem arithmetic_progression_common_difference
  (a : ℕ → ℤ) (m : ℕ) (d : ℤ)
  (h_odd_sum : ∑ i in Finset.range m, a (2 * i + 1) = 90)
  (h_even_sum : ∑ i in Finset.range m, a (2 * i + 2) = 72)
  (h_diff_last_first : a (2 * m) - a 1 = -33) :
  d = -3 :=
by
  sorry

end arithmetic_progression_common_difference_l142_142207


namespace sum_of_longest_altitudes_l142_142747

theorem sum_of_longest_altitudes (a b c : ℕ) (h : a^2 + b^2 = c^2) (h₁ : a = 8) (h₂ : b = 15) (h₃ : c = 17) : a + b = 23 := 
by { rw [h₁, h₂], norm_num, }

end sum_of_longest_altitudes_l142_142747


namespace infinite_solutions_count_l142_142649

theorem infinite_solutions_count :
  {n : ℤ | ∃ (f : ℤ → ℤ → ℤ), (∀ x y : ℤ, f x y = x * y^2 + y^2 - x - y) ∧ 
  (∀ N: ℤ, ∃ᶠ x y in filter (range N), f x y = n)}.card = 3 :=
sorry

end infinite_solutions_count_l142_142649


namespace divisible_by_7_in_range_200_to_400_l142_142377

theorem divisible_by_7_in_range_200_to_400 : 
  ∃ n : ℕ, 
    (∀ (x : ℕ), (200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ finset.range (201)) ∧ finset.card (finset.filter (λ x, 200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0) (finset.range 401)) = 29) := 
begin
  sorry
end

end divisible_by_7_in_range_200_to_400_l142_142377


namespace monte_carlo_area_estimation_l142_142912

theorem monte_carlo_area_estimation :
  let a1_998 := [0.3, 0.9] in
  let b1_998 := [0.1, 0.7] in
  let inside_998 := (4 * a1_998.head! - 2) ^ 2 + 1 < 4 * b1_998.head! + 1 in
  let inside_999 := (4 * a1_998.tail!.head! - 2) ^ 2 + 1 < 4 * b1_998.tail!.head! + 1 in
  let count_in := 624 + (if inside_998 then 1 else 0) + (if inside_999 then 1 else 0) in
  count_in = 625 →
  let probability := count_in / 1000.0 in
  16 * probability = 10 :=
by
  sorry

end monte_carlo_area_estimation_l142_142912


namespace solve_equation_l142_142074

theorem solve_equation (y : ℝ) : 
  5 * (y + 2) + 9 = 3 * (1 - y) ↔ y = -2 := 
by 
  sorry

end solve_equation_l142_142074


namespace find_a_b_monotonicity_l142_142452

noncomputable def f (x a b : ℝ) : ℝ := x^3 - 3 * a * x^2 + b

theorem find_a_b (a b : ℝ) :
  f 2 a b = 8 ∧ (3 * (2:ℝ)^2 - 6 * a * 2) = 0 → a = 1 ∧ b = 12 := 
sorry

theorem monotonicity (a : ℝ) :
  (∀ x, a = 0 → (f x a 0) ≤ f (x + 1) a 0) ∧
  (∀ x, a > 0 →
    ((x < 0 ∨ x > 2 * a) → f (x) a (12 * a) < f (x + 1) a (12 * a)) ∧ 
    (x ∈ (0, 2*a) → f (x) a (12 * a) > f (x + 1) a (12 * a))) ∧
  (∀ x, a < 0 →
    ((x < 2 * a ∨ x > 0) → f (x) a (12 * a) < f (x + 1) a (12 * a)) ∧ 
    (x ∈ (2 * a, 0) → f (x) a (12 * a) > f (x + 1) a (12 * a))) :=
sorry

end find_a_b_monotonicity_l142_142452


namespace first_team_engineers_l142_142195

theorem first_team_engineers (E : ℕ) 
  (teamQ_engineers : ℕ := 16) 
  (work_days_teamQ : ℕ := 30) 
  (work_days_first_team : ℕ := 32) 
  (working_capacity_ratio : ℚ := 3 / 2) :
  E * work_days_first_team * 3 = teamQ_engineers * work_days_teamQ * 2 → 
  E = 10 :=
by
  sorry

end first_team_engineers_l142_142195


namespace f_6_plus_f_neg3_l142_142884

noncomputable def f : ℝ → ℝ := sorry

-- f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- f is increasing in the interval [3,6]
def is_increasing_interval (f : ℝ → ℝ) (a b : ℝ) := a ≤ b → ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

-- Define the given conditions
axiom h1 : is_odd_function f
axiom h2 : is_increasing_interval f 3 6
axiom h3 : f 6 = 8
axiom h4 : f 3 = -1

-- The statement to be proved
theorem f_6_plus_f_neg3 : f 6 + f (-3) = 9 :=
by
  sorry

end f_6_plus_f_neg3_l142_142884


namespace stick_markings_count_l142_142547

theorem stick_markings_count :
  ∃ (n : ℕ), (n = 9) ∧
  let marking_positions := {0, 1} ∪ (set.image (λ k : ℕ, k/3) {1, 2}) ∪ (set.image (λ k : ℕ, k/5) {1, 2, 3, 4}) in
  marking_positions.card = n :=
begin
  sorry
end

end stick_markings_count_l142_142547


namespace maximum_tied_teams_round_robin_l142_142404

noncomputable def round_robin_tournament_max_tied_teams (n : ℕ) : ℕ := 
  sorry

theorem maximum_tied_teams_round_robin (h : n = 8) : round_robin_tournament_max_tied_teams n = 7 :=
sorry

end maximum_tied_teams_round_robin_l142_142404


namespace exist_non_concentric_inner_circle_l142_142241

-- Definitions from the conditions given in the problem
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

def touches (c1 c2 : Circle) : Prop :=
dist c1.center c2.center = c1.radius + c2.radius

def is_between (c : Circle) (c1 c2 : Circle) : Prop :=
dist c1.center c.center < c1.radius ∧ dist c2.center c.center < c2.radius

-- The main proof problem statement in Lean 4
theorem exist_non_concentric_inner_circle
  (k1 : Circle) (r1 : ℝ)
  (h1 : k1.radius = r1)
  (h8 : ∃ (k2 : Circle), ¬(k2.center = k1.center) ∧ 
    ∃ (smalls : fin 8 → Circle),
      (∀ i, touches (smalls i) k1) ∧
      (∀ i, touches (smalls i) k2) ∧
      (∀ i, touches (smalls i) (smalls ((i + 1) % 8))) ∧
      is_between (smalls i) k1 k2) :
  ∃ (k2 : Circle), ¬(k2.center = k1.center) ∧
    ∃ (smalls : fin 8 → Circle),
      (∀ i, touches (smalls i) k1) ∧
      (∀ i, touches (smalls i) k2) ∧
      (∀ i, touches (smalls i) (smalls ((i + 1) % 8))) ∧
      is_between (smalls i) k1 k2 :=
sorry

end exist_non_concentric_inner_circle_l142_142241


namespace theodoreEarningsCorrect_l142_142513

noncomputable def theodoreEarnings : ℝ := 
  let s := 10
  let ps := 20
  let w := 20
  let pw := 5
  let b := 15
  let pb := 15
  let m := 150
  let l := 200
  let t := 0.10
  let totalEarnings := (s * ps) + (w * pw) + (b * pb)
  let expenses := m + l
  let earningsBeforeTaxes := totalEarnings - expenses
  let taxes := t * earningsBeforeTaxes
  earningsBeforeTaxes - taxes

theorem theodoreEarningsCorrect :
  theodoreEarnings = 157.50 :=
by sorry

end theodoreEarningsCorrect_l142_142513


namespace last_date_with_four_consecutive_digits_2012_l142_142490

noncomputable def is_four_consecutive_digits (date : ℕ) : Prop :=
  let digits := to_digits date in
  digits.length = 4 ∧ (digits[1] = digits[0] + 1) ∧ 
  (digits[2] = digits[1] + 1) ∧ (digits[3] = digits[2] + 1)

noncomputable def find_last_consecutive_digit_date (year : ℕ) : ℕ :=
  (List.range' 1 366).reverse.find (λ day, 
    let (month, day) := to_month_day year day in
    is_four_consecutive_digits (to_mmdd month day)).get_or_else 0

theorem last_date_with_four_consecutive_digits_2012 :
  find_last_consecutive_digit_date 2012 = 1230 := by
  apply congrArg
  sorry

end last_date_with_four_consecutive_digits_2012_l142_142490


namespace neg_q_true_l142_142704

theorem neg_q_true : (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end neg_q_true_l142_142704


namespace find_MN_l142_142792

noncomputable theory

variables (A B C L K M N : Type)
variables [Point A] [Point B] [Point C] [Point L] [Point K] [Point M] [Point N]
variables (AB AC BC : ℝ)
variables (AB_eq : AB = 130) (AC_eq : AC = 110) (BC_eq : BC = 120)
variables (angle_bisector_A : angle_bisector A L BC)
variables (angle_bisector_B : angle_bisector B K AC)
variables (foot_perpendicular_C_BK : foot_perpendicular C M BK)
variables (foot_perpendicular_C_AL : foot_perpendicular C N AL)

theorem find_MN : distance M N = 50 :=
sorry

end find_MN_l142_142792


namespace problem_l142_142425

noncomputable def probability_same_heads_sum (p : ℚ) : ℕ :=
  let m := 63
  let n := 200
  m + n

theorem problem {p : ℚ} (h : p = 3 / 5) :
  probability_same_heads_sum p = 263 :=
by
  rw [probability_same_heads_sum]
  sorry

end problem_l142_142425


namespace increasing_sequence_l142_142203

-- Define all given sequences
def seqA (n : ℕ) : ℕ := 1 - n
def seqB (n : ℕ) : ℝ := 1 / (4 : ℝ)^n
def seqC (n : ℕ) : ℕ := 2 * n^2 - 5 * n + 1
def seqD (n : ℕ) : ℕ :=
  if n ≤ 2 then n + 3 else 2^(n - 1)

-- Define the property of being increasing
def is_increasing (seq : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, seq (n + 1) > seq n

-- Theorem to prove: sequence C is the only increasing sequence
theorem increasing_sequence :
  is_increasing seqC ∧ ¬ is_increasing seqA ∧ ¬ is_increasing seqB ∧ ¬ is_increasing seqD :=
by
  sorry

end increasing_sequence_l142_142203


namespace collinear_ZAX_l142_142810

open scoped Classical

noncomputable def concyclic (P Q R S : Point) : Prop :=
  ∃ (circ : Circle), P ∈ circ ∧ Q ∈ circ ∧ R ∈ circ ∧ S ∈ circ

theorem collinear_ZAX {O A B C X Y Z : Point}
  (h1 : ∃ (circle1 : Circle), ∀ P ∈ {A, B, O}, P ∈ circle1)
  (h2 : ∃ (circle2 : Circle), ∀ P ∈ {B, C, O}, P ∈ circle2)
  (h3 : ∃ (circle3 : Circle), ∀ P ∈ {C, A, O}, P ∈ circle3)
  (h4 : X ∈ circle1)
  (h5 : Y ∈ line_through X B)
  (h6 : Y ∈ circle2)
  (h7 : Z ∈ line_through Y C)
  (h8 : Z ∈ circle3) :
  collinear Z A X :=
sorry

end collinear_ZAX_l142_142810


namespace dice_probability_prime_l142_142532

def dice_pairs : List (ℕ × ℕ) :=
  List.bind (List.range 8) $ λ i => 
  List.map (λ j => (i + 1, j + 1)) (List.range 8)

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

def valid_prime_sums : List (ℕ × ℕ) :=
  dice_pairs.filter (λ p => is_prime (p.fst + p.snd))

theorem dice_probability_prime : (valid_prime_sums.length :ℚ) / (dice_pairs.length : ℚ) = 17 / 64 :=
by
  -- You would construct this proof by computation and verification.
  sorry

end dice_probability_prime_l142_142532


namespace catch_up_distance_l142_142965

theorem catch_up_distance (v x y : ℝ) (hv : v > 0) (hx : x > 0.5) :
  let flash_speed := 2 * x * v in 
  let head_start := y in 
  let catch_up_distance := (2 * x * y) / (2 * x - 1) in
  catch_up_distance = (2 * x * y) / (2 * x - 1) :=
by
  sorry

end catch_up_distance_l142_142965


namespace solve_equation_l142_142073

open Real

theorem solve_equation (t : ℝ) :
  ¬cos t = 0 ∧ ¬cos (2 * t) = 0 → 
  (tan (2 * t) / (cos t)^2 - tan t / (cos (2 * t))^2 = 0 ↔ 
    (∃ k : ℤ, t = π * ↑k) ∨ (∃ n : ℤ, t = π * ↑n + π / 6) ∨ (∃ n : ℤ, t = π * ↑n - π / 6)) :=
by
  intros h
  sorry

end solve_equation_l142_142073


namespace total_journey_distance_l142_142589

-- Definitions of the conditions

def journey_time : ℝ := 40
def first_half_speed : ℝ := 20
def second_half_speed : ℝ := 30

-- Proof statement
theorem total_journey_distance : ∃ D : ℝ, (D / first_half_speed + D / second_half_speed = journey_time) ∧ (D = 960) :=
by 
  sorry

end total_journey_distance_l142_142589


namespace michael_and_truck_meet_once_l142_142462

variable (t : ℝ) -- time variable

-- Define the speeds and distances
def michael_speed := 6 -- feet per second
def truck_speed := 12 -- feet per second
def pail_distance := 300 -- feet
def truck_stop_time := 20 -- seconds

-- Position functions
def M (t : ℝ) : ℝ := michael_speed * t
def T (t : ℝ) : ℝ := truck_speed * t - pail_distance

-- Truck's cycle
def truck_cycle_time := (pail_distance / truck_speed) + truck_stop_time -- 45 seconds

-- Distance function during movement and stop
def D (t : ℝ) : ℝ :=
  if 0 ≤ t % truck_cycle_time ∧ t % truck_cycle_time < (pail_distance / truck_speed) then
    (truck_speed * t - pail_distance) - michael_speed * t
  else
    (truck_speed * (t - (t % truck_cycle_time - (pail_distance / truck_speed))) - pail_distance) - michael_speed * t

theorem michael_and_truck_meet_once :
  ∃ t₁ t₂ : ℝ,
    0 ≤ t₁ ∧ t₁ < (pail_distance / truck_speed) ∧ D t₁ = 0 ∧
    ∀ t, (t ≥ truck_cycle_time → D t > 0) :=
by
  sorry

end michael_and_truck_meet_once_l142_142462


namespace probability_matching_or_diff_colors_l142_142906

noncomputable def total_socks : ℕ := 12 + 10 + 6
noncomputable def combinations (n k : ℕ) : ℕ := nat.choose n k

noncomputable def prob_matching_or_diff_colors : ℚ :=
  let total_ways := combinations total_socks 3
  let ways_gray_pair := combinations 12 2 * 16
  let ways_white_pair := combinations 10 2 * 18
  let ways_blue_pair := combinations 6 2 * 22
  let ways_diff_colors := 12 * 10 * 6
  let favorable_ways := ways_gray_pair + ways_white_pair + ways_blue_pair + ways_diff_colors
  favorable_ways / total_ways

theorem probability_matching_or_diff_colors :
  prob_matching_or_diff_colors = 81 / 91 := sorry

end probability_matching_or_diff_colors_l142_142906


namespace part_a_intersection_part_b_section_l142_142927

section ThreeFacedAngle

variable (O a b c A B : Point)
variable (Plane : Type) [has_plane : has_plane Plane]
variable (O_a_b c : Plane)
variable [has_plane_plane : has_plane_plane O_a_b c]
variable (AB : Line)
variable (A_in_O_bc : A ∈ face O b c)
variable (B_in_O_ac : B ∈ face O a c)
variable (P_in_AB : Point)

-- Part (a)
theorem part_a_intersection (O a b c A B : Point) (A_in_O_bc : A ∈ face O b c) (B_in_O_ac : B ∈ face O a c) (AB : Line) : 
  ∃ P, intersection (line_through A B) (plane_through O a b) = Some P := 
by
  sorry

variable (A B C : Point)
variable (O_a_b O_b_c O_a_c : Plane)
variable (A_in_O_bc : A ∈ face O b c)
variable (B_in_O_ac : B ∈ face O a c)
variable (C_in_O_ab : C ∈ face O a b)

-- Part (b)
theorem part_b_section (O a b c A B C: Point) (A_in_O_bc : A ∈ face O b c) (B_in_O_ac : B ∈ face O a c) (C_in_O_ab : C ∈ face O a b) : 
  ∃ Q, intersection (plane_through P C) (plane_passing_through A B C) = Some Q :=
by
  sorry

end ThreeFacedAngle

end part_a_intersection_part_b_section_l142_142927


namespace f_1986_eq_1_l142_142990

noncomputable def f : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := 3 - f (n+1)

theorem f_1986_eq_1 : f 1986 = 1 :=
sorry

end f_1986_eq_1_l142_142990


namespace savings_after_four_weeks_l142_142070

noncomputable def hourly_wage (name : String) : ℝ :=
  match name with
  | "Robby" | "Jaylen" | "Miranda" => 10
  | "Alex" => 12
  | "Beth" => 15
  | "Chris" => 20
  | _ => 0

noncomputable def daily_hours (name : String) : ℝ :=
  match name with
  | "Robby" | "Miranda" => 10
  | "Jaylen" => 8
  | "Alex" => 6
  | "Beth" => 4
  | "Chris" => 3
  | _ => 0

noncomputable def saving_rate (name : String) : ℝ :=
  match name with
  | "Robby" => 2/5
  | "Jaylen" => 3/5
  | "Miranda" => 1/2
  | "Alex" => 1/3
  | "Beth" => 1/4
  | "Chris" => 3/4
  | _ => 0

noncomputable def weekly_earning (name : String) : ℝ :=
  hourly_wage name * daily_hours name * 5

noncomputable def weekly_saving (name : String) : ℝ :=
  weekly_earning name * saving_rate name

noncomputable def combined_savings : ℝ :=
  4 * (weekly_saving "Robby" + 
       weekly_saving "Jaylen" + 
       weekly_saving "Miranda" + 
       weekly_saving "Alex" + 
       weekly_saving "Beth" + 
       weekly_saving "Chris")

theorem savings_after_four_weeks :
  combined_savings = 4440 :=
by
  sorry

end savings_after_four_weeks_l142_142070


namespace find_number_of_students_l142_142402

variables (n : ℕ)
variables (avg_A avg_B avg_C excl_avg_A excl_avg_B excl_avg_C : ℕ)
variables (new_avg_A new_avg_B new_avg_C : ℕ)
variables (excluded_students : ℕ)

theorem find_number_of_students :
  avg_A = 80 ∧ avg_B = 85 ∧ avg_C = 75 ∧
  excl_avg_A = 20 ∧ excl_avg_B = 25 ∧ excl_avg_C = 15 ∧
  excluded_students = 5 ∧
  new_avg_A = 90 ∧ new_avg_B = 95 ∧ new_avg_C = 85 →
  n = 35 :=
by
  sorry

end find_number_of_students_l142_142402


namespace find_s_l142_142814

noncomputable def f (s : ℝ) :=
  Polynomial.monicHorner [1, -(s + 2 + s + 8), (s + 2) * (s + 8), 0, 0]

noncomputable def g (s : ℝ) :=
  Polynomial.monicHorner [1, -(s + 4 + s + 10), (s + 4) * (s + 10), 0, 0]

theorem find_s :
  ∃ s : ℝ, s = 12 ∧ ∀ x : ℝ, f s x - g s x = 2 * s :=
  by
    use 12
    sorry

end find_s_l142_142814


namespace avocados_per_serving_l142_142284

-- Definitions for the conditions
def original_avocados : ℕ := 5
def additional_avocados : ℕ := 4
def total_avocados : ℕ := original_avocados + additional_avocados
def servings : ℕ := 3

-- Theorem stating the result
theorem avocados_per_serving : (total_avocados / servings) = 3 :=
by
  sorry

end avocados_per_serving_l142_142284


namespace volume_of_cuboid_l142_142489

-- Define the edges of the cuboid
def edge1 : ℕ := 6
def edge2 : ℕ := 5
def edge3 : ℕ := 6

-- Define the volume formula for a cuboid
def volume (a b c : ℕ) : ℕ := a * b * c

-- State the theorem
theorem volume_of_cuboid : volume edge1 edge2 edge3 = 180 := by
  sorry

end volume_of_cuboid_l142_142489


namespace volume_of_solid_l142_142876

-- Definitions and Conditions
def line (M N : ℝ × ℝ) : Prop :=
  2 * M.1 - 2 * M.2 = 1 ∧ 2 * N.1 - 2 * N.2 = 1

def parabola (M N : ℝ × ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ M.2^2 = 2 * p * M.1 ∧ N.2^2 = 2 * p * N.1

def distance (M N : ℝ × ℝ) : ℝ :=
  real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)

-- Theorem statement
theorem volume_of_solid (M N S T : ℝ × ℝ) (p : ℝ) :
  line M N →
  parabola M N p →
  distance M N = 4 →
  -- Projections S and T ignored as they are implicitly used in volume computation
  -- This is the proof goal; no proof provided.
  ∃ V : ℝ, V = 9 * real.sqrt 2 * real.pi :=
begin
  sorry
end

end volume_of_solid_l142_142876


namespace baseball_player_hits_l142_142168

def total_number_of_hits (total_bats lh_bats: ℕ) (lh_avg rh_avg: ℝ) : ℕ :=
  let rh_bats := total_bats - lh_bats
  let lh_hits := lh_avg * lh_bats
  let rh_hits := rh_avg * rh_bats
  (lh_hits + rh_hits).to_nat

theorem baseball_player_hits 
  (total_bats : ℕ)
  (lh_bats : ℕ)
  (lh_avg rh_avg : ℝ)
  (h_total_bats : total_bats = 600)
  (h_lh_bats : lh_bats = 180)
  (h_lh_avg : lh_avg = 0.250)
  (h_rh_avg : rh_avg = 0.350) 
  : total_number_of_hits total_bats lh_bats lh_avg rh_avg = 192 :=
  by
    rw [h_total_bats, h_lh_bats, h_lh_avg, h_rh_avg]
    sorry

end baseball_player_hits_l142_142168


namespace sum_of_squares_l142_142905

def gcd (a b c : Nat) : Nat := (Nat.gcd (Nat.gcd a b) c)

theorem sum_of_squares {a b c : ℕ} (h1 : 3 * a + 2 * b = 4 * c)
                                   (h2 : 3 * c ^ 2 = 4 * a ^ 2 + 2 * b ^ 2)
                                   (h3 : gcd a b c = 1) :
  a^2 + b^2 + c^2 = 45 :=
by
  sorry

end sum_of_squares_l142_142905


namespace circle_with_diameter_touches_sides_l142_142799

-- Definitions 
variables {O A B : Point}
variables (angle : region) (circle1 circle2 circle_diam : Circle)
variables (r1 r2 rd : ℝ) -- radii

-- Conditions
def inside_angle (point : Point) (angle : region) : Prop := sorry -- replace with actual definition
def touching (c1 c2 : Circle) : Prop := sorry -- replace with actual definition
def center (C : Circle) (P : Point) : Prop := sorry -- replace with actual definition
def diameter (P Q : Point) (C : Circle) : Prop := sorry -- replace with actual definition

axiom circles_inside_angle : inside_angle A angle ∧ inside_angle B angle 
axiom circles_touching_each_other : touching circle1 circle2 
axiom circles_touching_sides : touching circle1 angle ∧ touching circle2 angle
axiom centers : center circle1 A ∧ center circle2 B
axiom diameter_circle : diameter A B circle_diam

-- Proof problem
theorem circle_with_diameter_touches_sides :
  touching circle_diam angle :=
sorry

end circle_with_diameter_touches_sides_l142_142799


namespace distance_AB_eq_sqrt_29_l142_142625

noncomputable def ellipse_distance : ℝ :=
  let A : ℝ × ℝ := (-2, 8)
  let B : ℝ × ℝ := (0, 3)
  real.sqrt (((B.1 - A.1)^2 + (B.2 - A.2)^2))

theorem distance_AB_eq_sqrt_29 :
  let A : ℝ × ℝ := (-2, 8)
  let B : ℝ × ℝ := (0, 3)
  ellipse_distance = real.sqrt 29 :=
by sorry

end distance_AB_eq_sqrt_29_l142_142625


namespace youtube_dislikes_l142_142573

theorem youtube_dislikes (likes : ℕ) (initial_dislikes : ℕ) (final_dislikes : ℕ) :
  likes = 3000 →
  initial_dislikes = likes / 2 + 100 →
  final_dislikes = initial_dislikes + 1000 →
  final_dislikes = 2600 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end youtube_dislikes_l142_142573


namespace positive_numbers_l142_142341

theorem positive_numbers 
    (a b c : ℝ) 
    (h1 : a + b + c > 0) 
    (h2 : ab + bc + ca > 0) 
    (h3 : abc > 0) 
    : a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end positive_numbers_l142_142341


namespace total_packs_l142_142834

noncomputable def robyn_packs : ℕ := 16
noncomputable def lucy_packs : ℕ := 19

theorem total_packs : robyn_packs + lucy_packs = 35 := by
  sorry

end total_packs_l142_142834


namespace baba_yaga_departure_and_speed_l142_142212

variables (T : ℕ) (d : ℕ)

theorem baba_yaga_departure_and_speed :
  (50 * (T + 2) = 150 * (T - 2)) →
  (12 - T = 8) ∧ (d = 50 * (T + 2)) →
  (d = 300) ∧ ((d / T) = 75) :=
by
  intros h1 h2
  sorry

end baba_yaga_departure_and_speed_l142_142212


namespace find_b_l142_142090

-- Define the curve and the line equations
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x + 1
def line (k : ℝ) (b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the conditions in the problem
def passes_through_point (a : ℝ) : Prop := curve a 2 = 3
def is_tangent_at_point (a k b : ℝ) : Prop :=
  ∀ x : ℝ, curve a x = 3 → line k b 2 = 3

-- Main theorem statement
theorem find_b (a k b : ℝ) (h1 : passes_through_point a) (h2 : is_tangent_at_point a k b) : b = -15 :=
by sorry

end find_b_l142_142090


namespace max_int_value_of_a_real_roots_l142_142682

-- Definitions and theorem statement based on the above conditions
theorem max_int_value_of_a_real_roots (a : ℤ) :
  (∃ x : ℝ, (a-1) * x^2 - 2 * x + 3 = 0) ↔ a ≠ 1 ∧ a ≤ 0 := by
  sorry

end max_int_value_of_a_real_roots_l142_142682


namespace sin_2alpha_value_l142_142317

theorem sin_2alpha_value (α : ℝ) (h : (cos (2 * α)) / (sin (α + real.pi / 4)) = 4 / 7) :
  sin (2 * α) = 41 / 49 :=
by
  sorry

end sin_2alpha_value_l142_142317


namespace minimum_value_expression_l142_142689

theorem minimum_value_expression 
  (x y z : ℝ) 
  (h1 : x ≥ 3) (h2 : y ≥ 3) (h3 : z ≥ 3) :
  let A := (x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3) in
  let B := x * y + y * z + z * x in
  A / B ≥ 1 :=
sorry

end minimum_value_expression_l142_142689


namespace chairs_to_remove_l142_142181

theorem chairs_to_remove 
  (chairs_per_row : ℕ) 
  (initial_chairs : ℕ) 
  (expected_attendees : ℕ) 
  (remaining_chairs_multiple : ℕ → ℕ) 
  (at_least_one_row_empty : ℕ → ℕ) : 
  chairs_per_row = 15 → 
  initial_chairs = 150 → 
  expected_attendees = 125 → 
  (∀ n, remaining_chairs_multiple n → n % chairs_per_row = 0) → 
  (∀ m, at_least_one_row_empty m → m < initial_chairs - expected_attendees) → 
  ∃ chairs_to_remove : ℕ, chairs_to_remove = 45 :=
by
  sorry

end chairs_to_remove_l142_142181


namespace sqrt_identity_l142_142665

theorem sqrt_identity :
  sqrt (1 - 2 * cos (Real.pi / 2 + 3) * sin (Real.pi / 2 - 3)) = -sin 3 - cos 3 :=
by sorry

end sqrt_identity_l142_142665


namespace children_tickets_sold_l142_142544

theorem children_tickets_sold {A C : ℕ} (h1 : 6 * A + 4 * C = 104) (h2 : A + C = 21) : C = 11 :=
by
  sorry

end children_tickets_sold_l142_142544


namespace intersection_points_rectangular_coordinates_tangent_line_equation_l142_142781

-- (1) Prove that the intersection points in rectangular coordinates are (1/4, ± sqrt(15)/4)
theorem intersection_points_rectangular_coordinates :
  ∀ (ρ θ : ℝ), (ρ = 1 ∧ ρ = 4 * cos θ) → 
  ∃ (x y : ℝ), x = 1 / 4 ∧ (y = sqrt (15) / 4 ∨ y = -sqrt (15) / 4) :=
by
  sorry

-- (2) Prove that the equation of the line l tangent to both curves is x ± sqrt(3)y - 2 = 0
theorem tangent_line_equation :
  ∃ (k : ℝ), (∀ (x y : ℝ), y = k * (x - 2) → x ± sqrt(3) * y - 2 = 0) :=
by
  sorry

end intersection_points_rectangular_coordinates_tangent_line_equation_l142_142781


namespace star_inequalities_not_all_true_simultaneously_l142_142841

theorem star_inequalities_not_all_true_simultaneously
  (AB BC CD DE EF FG GH HK KL LA : ℝ)
  (h1 : BC > AB)
  (h2 : DE > CD)
  (h3 : FG > EF)
  (h4 : HK > GH)
  (h5 : LA > KL) :
  False :=
  sorry

end star_inequalities_not_all_true_simultaneously_l142_142841


namespace problem_statement_l142_142849

theorem problem_statement (x : ℝ) (h : (2024 - x)^2 + (2022 - x)^2 = 4038) : 
  (2024 - x) * (2022 - x) = 2017 :=
sorry

end problem_statement_l142_142849


namespace centroid_iff_quadrilateral_areas_equal_l142_142441

noncomputable def is_midpoint (P A B : Point) : Prop :=
  dist P A = dist P B

noncomputable def is_centroid (P A B C : Point) : Prop :=
  dist P (midpoint A B) = dist P (midpoint B C) ∧ dist P (midpoint A C) = 2 * dist P (midpoint A B)

-- Define the quadrilateral areas equivalence
noncomputable def quadrilateral_areas_equal (A B C P : Point) : Prop :=
  let X := midpoint B C in
  let Y := midpoint C A in
  let Z := midpoint A B in
  area (polygon.mk [A, Z, P, Y]) = area (polygon.mk [B, X, P, Z]) ∧
  area (polygon.mk [C, Y, P, X]) = area (polygon.mk [B, X, P, Z])

theorem centroid_iff_quadrilateral_areas_equal (A B C P : Point) :
  quadrilateral_areas_equal A B C P ↔ is_centroid P A B C :=
sorry

end centroid_iff_quadrilateral_areas_equal_l142_142441


namespace miss_bus_time_l142_142913

-- Definitions (conditions extracted from part a)
def usual_time : ℕ := 24
def slowed_speed_factor : ℚ := 4 / 5

-- Theorem statement (the question and answer, proving that the miss time is 6 minutes)
theorem miss_bus_time : 
  let usual_time := usual_time 
  ∧ let slowed_speed := slowed_speed_factor 
  in (let new_time := (affected_usual_time : ℚ) := 
       (5 / 4) * (usual_time: ℚ) 
  in (new_time - usual_time) = 6) := 
begin
  sorry
end

end miss_bus_time_l142_142913


namespace x5_plus_y5_l142_142314

theorem x5_plus_y5 (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) : 
  x^5 + y^5 = 252 :=
by
  -- Placeholder for the proof
  sorry

end x5_plus_y5_l142_142314


namespace cos_pi_minus_2alpha_l142_142705

-- Defining the necessary constants and variables
variables (α : ℝ)

-- Given condition
def sin_alpha := 3 / 5

-- Required proof statement
theorem cos_pi_minus_2alpha : sin α = sin_alpha → cos (π - 2 * α) = -7 / 25 :=
by
  intro h
  sorry

end cos_pi_minus_2alpha_l142_142705


namespace find_point_C_l142_142497

-- Definitions of the conditions
def line_eq (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
def on_parabola (C : ℝ × ℝ) : Prop := parabola_eq C.1 C.2
def perpendicular_at_C (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Points A and B satisfy both the line and parabola equations
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_eq A.1 A.2 ∧ parabola_eq A.1 A.2 ∧
  line_eq B.1 B.2 ∧ parabola_eq B.1 B.2

-- Statement to be proven
theorem find_point_C (A B : ℝ × ℝ) (hA : intersection_points A B) :
  ∃ C : ℝ × ℝ, on_parabola C ∧ perpendicular_at_C A B C ∧
    (C = (1, -2) ∨ C = (9, -6)) :=
by
  sorry

end find_point_C_l142_142497


namespace largest_minus_smallest_l142_142898

-- Define the given conditions
def A : ℕ := 10 * 2 + 9
def B : ℕ := A - 16
def C : ℕ := B * 3

-- Statement to prove
theorem largest_minus_smallest : C - B = 26 := by
  sorry

end largest_minus_smallest_l142_142898


namespace area_of_rhombus_eq_208_l142_142916

variable (side : ℝ) (d1 d2 : ℝ)

-- Conditions
def rhombus_side := side = 13
def diagonals_differ_by_10 := abs (d1 - d2) = 10

-- Area of rhombus as half the product of diagonals
def rhombus_area := (1/2) * d1 * d2

theorem area_of_rhombus_eq_208 :
  rhombus_side side →
  diagonals_differ_by_10 d1 d2 →
  ∃ d, d1 = 2 * d ∧ d2 = 2 * (d + 5) ∧ d1 * d2 / 2 = 208 :=
by
  intros h_side h_diagonal_diff
  sorry

end area_of_rhombus_eq_208_l142_142916


namespace find_a_and_k_sum_b_series_l142_142699

-- Define the arithmetic sequence and the sum of the first n terms
def a₁ (a : ℕ) := a - 1
def a₂ : ℕ := 4
def a₃ (a : ℕ) := 2 * a

def Sn (a n : ℕ) : ℕ := n * a₁ a + (n * (n - 1) / 2) * (a₂ - a₁ a)

-- First part: finding a and k
theorem find_a_and_k (a k : ℕ) (h : Sn a k = 30) : a = 3 ∧ k = 5 := sorry

-- Define b_n and the required summation
def bn (a n : ℕ) : ℕ := Sn a n / n

def sum_b_3_7_11 (a n : ℕ) : ℕ :=
  ∑ i in Finset.range n, bn a (4 * i + 3)

-- Second part: finding the specific sum
theorem sum_b_series (a : ℕ) : b_n 3 (n + 1) + b_n 3 (4 + 1) + b_n 3 (4*2 + 1) + ··· + b_n 3 (4*(n-1)+1) = 2n^2 + 2n := sorry

end find_a_and_k_sum_b_series_l142_142699


namespace count_multiples_of_7_l142_142355

theorem count_multiples_of_7 (low high : ℕ) (hlow : low = 200) (hhigh : high = 400) : 
  (card {n | low ≤ n ∧ n ≤ high ∧ n % 7 = 0}) = 29 := by
  sorry

end count_multiples_of_7_l142_142355


namespace initial_pants_l142_142635

theorem initial_pants (pairs_per_year : ℕ) (pants_per_pair : ℕ) (years : ℕ) (total_pants : ℕ) 
  (h1 : pairs_per_year = 4) (h2 : pants_per_pair = 2) (h3 : years = 5) (h4 : total_pants = 90) : 
  ∃ (initial_pants : ℕ), initial_pants = total_pants - (pairs_per_year * pants_per_pair * years) :=
by
  use 50
  sorry

end initial_pants_l142_142635


namespace two_numbers_divisor_property_l142_142116

theorem two_numbers_divisor_property (s : Finset ℕ) (h1 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) (h2 : s.card = 1008) 
  : ∃ a b ∈ s, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end two_numbers_divisor_property_l142_142116


namespace num_zeros_f_in_domain_l142_142092

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) - log x

theorem num_zeros_f_in_domain :
  ∃! (x1 x2 : ℝ), (0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0) :=
sorry

end num_zeros_f_in_domain_l142_142092


namespace sum_of_numbers_l142_142278

theorem sum_of_numbers (x y : ℝ) (h1 : y = 4 * x) (h2 : x + y = 45) : x + y = 45 := 
by
  sorry

end sum_of_numbers_l142_142278


namespace congruence_solution_l142_142387

theorem congruence_solution (x : ℤ) (h : 5 * x + 11 ≡ 3 [ZMOD 19]) : 3 * x + 7 ≡ 6 [ZMOD 19] :=
sorry

end congruence_solution_l142_142387


namespace angle_C_eq_pi_over_3_sides_a_and_b_eq_2_l142_142774

-- Define the triangle and its conditions
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (ABC_is_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variable (c_eq_2 : c = 2)
variable (sqrt3a_eq_2csinA : sqrt 3 * a = 2 * c * Real.sin A)

-- Define the given condition for part (2)
variable (area_eq_sqrt3 : 1 / 2 * a * b * Real.sin C = sqrt 3)

-- Theorem for part (1)
theorem angle_C_eq_pi_over_3 : C = π / 3 := by
  sorry

-- Theorem for part (2)
theorem sides_a_and_b_eq_2 : a = 2 ∧ b = 2 := by
  sorry

end angle_C_eq_pi_over_3_sides_a_and_b_eq_2_l142_142774


namespace rodney_lifting_capacity_l142_142476

theorem rodney_lifting_capacity 
  (R O N : ℕ)
  (h1 : R + O + N = 239)
  (h2 : R = 2 * O)
  (h3 : O = 4 * N - 7) : 
  R = 146 := 
by
  sorry

end rodney_lifting_capacity_l142_142476


namespace base2_digit_difference_l142_142541

def num_digits_in_base (n b : ℕ) : ℕ :=
  (Nat.log n / Nat.log b).toNat + 1

theorem base2_digit_difference (n m : ℕ) (hn : n = 1500) (hm : m = 300) :
  num_digits_in_base 1500 2 - num_digits_in_base 300 2 = 2 := by
  sorry

end base2_digit_difference_l142_142541


namespace find_c_l142_142626

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12 * x + 3 * x^2 - 4 * x^3 + 5 * x^4
def g (x : ℝ) : ℝ := 3 - 2 * x - 6 * x^3 + 7 * x^4

-- Define the main theorem stating that c = -5/7 makes f(x) + c*g(x) have degree 3
theorem find_c (c : ℝ) (h : ∀ x : ℝ, f x + c * g x = 0) : c = -5 / 7 := by
  sorry

end find_c_l142_142626


namespace full_capacity_l142_142552

def oil_cylinder_capacity (C : ℝ) :=
  (4 / 5) * C - (3 / 4) * C = 4

theorem full_capacity : oil_cylinder_capacity 80 :=
by
  simp [oil_cylinder_capacity]
  sorry

end full_capacity_l142_142552


namespace divisor_count_1386_l142_142274

theorem divisor_count_1386 : 
  let n := 1386 in
  let exps := [(2,1), (3,2), (7,1), (11,1)] in
  (n = (2^1) * (3^2) * (7^1) * (11^1)) →
  (exps.foldl (λ acc (p, e) => acc * (e + 1)) 1) = 24 :=
by
  intros n exps h
  rw h
  sorry

end divisor_count_1386_l142_142274


namespace probability_three_reds_first_l142_142603

open ProbabilityTheory

/-- A hat contains 3 red chips and 3 green chips. Chips are drawn randomly one by one without replacement until either all 3 red chips or all 3 green chips are drawn. Prove that the probability of drawing all 3 red chips before all 3 green chips is 1/2. -/
theorem probability_three_reds_first : 
  let chips := {x : Fin 6 // x < 3 ∨ x ≥ 3} in
  let event := {ω : Finset (Fin 6) | ∀ x ∈ ω, ∀ y ∈ (Finset.univ \ ω), (x < 3 ↔ y ≥ 3)} in
  (|{ω ∈ (events Ω) | ∀ x ∈ ω, x < 3}| / |Ω|) = 1 / 2 :
  sorry

end probability_three_reds_first_l142_142603


namespace line_no_intersection_parabola_l142_142339

theorem line_no_intersection_parabola {t : ℝ} :
  let A := (0, -2)
      G := {p : ℝ × ℝ | p.1^2 = 2 * p.2}
      line := {y | ∃ (k : ℝ), y = k * t - 2}
  in (∀ p ∈ G, ∀ y ∈ line, p ≠ y)
  ↔ (t < -1 ∨ t > 1) :=
by
  -- sorry  

end line_no_intersection_parabola_l142_142339


namespace number_of_new_players_l142_142887

variable (returning_players : ℕ)
variable (groups : ℕ)
variable (players_per_group : ℕ)

theorem number_of_new_players
  (h1 : returning_players = 6)
  (h2 : groups = 9)
  (h3 : players_per_group = 6) :
  (groups * players_per_group - returning_players = 48) := 
sorry

end number_of_new_players_l142_142887


namespace math_problem_proof_l142_142305

open Real

-- Definitions and conditions
variables {A B C : ℝ} {a b c : ℝ}
variables (BA BC BD : EuclideanSpace ℝ (Fin 3))

noncomputable def triangle_acute (A B C : ℝ) : Prop :=
A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π

noncomputable def given_conditions : Prop :=
a = 3 ∧ b = sqrt 13 ∧ a * sin (2 * B) = b * sin A ∧ triangle_acute A B C

-- Prove angles and lengths
noncomputable def question1 := (B = π / 3)
noncomputable def question2 := (BD = sqrt ((1 / 3)^2 * BA^2 + (2 / 3)^2 * BC^2 + 2 * (1 / 3) * (2 / 3) * BA * BC * cos B) := (2 * sqrt 19) / 3)

theorem math_problem_proof :
given_conditions → question1 ∧ question2 :=
by
sorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrsorry

end math_problem_proof_l142_142305


namespace min_odd_integers_l142_142112

theorem min_odd_integers (a b c d e f : ℤ) 
  (h1 : a + b = 34)
  (h2 : a + b + c + d = 51)
  (h3 : a + b + c + d + e + f = 72) : 
  nat (count_odds [a, b, c, d, e, f]) ≥ 2 :=
sorry

-- Helper function to count the odd integers in a list of integers
def count_odds (lst : list ℤ) : ℤ :=
lst.count (λ n, n % 2 ≠ 0)

end min_odd_integers_l142_142112


namespace min_visible_sum_of_values_l142_142943

-- Definitions based on the problem conditions
def is_standard_die (die : ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), (i + j = 7) → (die j + die i = 7)

def corner_cubes (cubes : ℕ) : ℕ := 8
def edge_cubes (cubes : ℕ) : ℕ := 24
def face_center_cubes (cubes : ℕ) : ℕ := 24

-- The proof statement
theorem min_visible_sum_of_values
  (m : ℕ)
  (condition1 : is_standard_die m)
  (condition2 : corner_cubes 64 = 8)
  (condition3 : edge_cubes 64 = 24)
  (condition4 : face_center_cubes 64 = 24)
  (condition5 : 64 = 8 + 24 + 24 + 8): 
  m = 144 :=
sorry

end min_visible_sum_of_values_l142_142943


namespace number_of_linear_functions_l142_142877

/-- Defines a function and checks if it is linear -/
def is_linear (f : ℝ → ℝ) : Prop :=
∃ (m b: ℝ), (∀ x, f x = m*x + b)

def func1 : ℝ → ℝ := λ x, k*x + b
def func2 : ℝ → ℝ := λ x, 2*x
def func3 : ℝ → ℝ := λ x, -3/x
def func4 : ℝ → ℝ := λ x, (1/3)*x + 3
def func5 : ℝ → ℝ := λ x, x^2 - 2*x + 1

theorem number_of_linear_functions : 
  let functions := [func1, func2, func3, func4, func5] in
  (list.countp (λ f, is_linear f) functions) = 2 :=
by sorry

end number_of_linear_functions_l142_142877


namespace train_length_l142_142964

theorem train_length (t : ℝ) (v : ℝ) (h1 : t = 13) (h2 : v = 58.15384615384615) : abs (v * t - 756) < 1 :=
by
  sorry

end train_length_l142_142964


namespace actual_revenue_is_60_percent_of_projected_l142_142931

variable (R : ℝ)

-- Condition: Projected revenue is 25% more than last year's revenue
def projected_revenue (R : ℝ) : ℝ := 1.25 * R

-- Condition: Actual revenue decreased by 25% compared to last year's revenue
def actual_revenue (R : ℝ) : ℝ := 0.75 * R

-- Theorem: Prove that the actual revenue is 60% of the projected revenue
theorem actual_revenue_is_60_percent_of_projected :
  (actual_revenue R) = 0.6 * (projected_revenue R) :=
  sorry

end actual_revenue_is_60_percent_of_projected_l142_142931


namespace slope_angle_of_tangent_at_x_eq_1_l142_142099

def f (x : ℝ) : ℝ := - (Real.sqrt 3 / 3) * x^3 + 2

theorem slope_angle_of_tangent_at_x_eq_1 :
  let f' := (deriv f 1)
  let θ := Real.arctan f'
  θ = 2 * Real.pi / 3 := by
  sorry

end slope_angle_of_tangent_at_x_eq_1_l142_142099


namespace num_divisors_of_1386_l142_142272

theorem num_divisors_of_1386 : 
  (∀ n, n = 1386 → 
    (∃ (a b c d : ℕ), 
        n = (2^a) * (3^b) * (7^c) * (11^d) ∧ 
        a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1) → 
        (finset.range (n + 1)).filter (λ x, n % x = 0).card = 24) := 
by
  sorry

end num_divisors_of_1386_l142_142272


namespace goalkeeper_not_at_goal_line_total_energy_consumption_scoring_opportunities_l142_142771

def movements := [7, -3, 8, 4, -6, -8, 14, -15]
def energy_consumption_rate := 0.1
def scoring_distance := 10

-- Part 1: Prove that the goalkeeper is not exactly at the goal line at the end of the movements.
theorem goalkeeper_not_at_goal_line :
  ∑ i in movements, i ≠ 0 := by
  sorry

-- Part 2: Prove that the total energy consumption of the goalkeeper is 6.5 calories.
theorem total_energy_consumption :
  (∑ i in movements, abs i) * energy_consumption_rate = 6.5 := by
  sorry

-- Part 3: Prove that the opponent player has exactly 3 scoring opportunities during the period.
theorem scoring_opportunities :
  list (λ acc i, acc + i) movements [] |>.map abs |>.filter (> scoring_distance) |>.length = 3 := by
  sorry

end goalkeeper_not_at_goal_line_total_energy_consumption_scoring_opportunities_l142_142771


namespace carnations_third_bouquet_l142_142909

theorem carnations_third_bouquet (bouquet1 bouquet2 bouquet3 : ℕ) 
  (h1 : bouquet1 = 9) (h2 : bouquet2 = 14) 
  (h3 : (bouquet1 + bouquet2 + bouquet3) / 3 = 12) : bouquet3 = 13 :=
by
  sorry

end carnations_third_bouquet_l142_142909


namespace count_ways_to_change_15_dollars_with_nickels_and_half_dollars_l142_142818

theorem count_ways_to_change_15_dollars_with_nickels_and_half_dollars : 
  (∃ n h : ℕ, 5 * n + 50 * h = 1500 ∧ n > 0 ∧ h > 0) → 
  (finset.card (finset.filter (λ h, (5 * (300 - 10 * h) + 50 * h = 1500) ∧ (300 - 10 * h > 0) ∧ (h > 0)) (finset.range 30)) = 29) :=
by
  let n := 300
  let c := 10
  sorry

end count_ways_to_change_15_dollars_with_nickels_and_half_dollars_l142_142818


namespace angles_BD_sum_l142_142316

theorem angles_BD_sum (A B C D E F G : Type) [Geometry A B C D E F G]
  (angle_A : angle A = 30)
  (angle_AFG_eq_AGF : angle AFG = angle AGF) :
  angle B + angle D = 75 :=
sorry

end angles_BD_sum_l142_142316


namespace range_b_three_monotonic_intervals_l142_142688

theorem range_b_three_monotonic_intervals (b : ℝ) :
  (∃ y : ℝ → ℝ, y = λ x, (1/3:ℝ) * x^3 + b * x^2 + (b + 6) * x + 3 ∧ 
                 ∃ intervals : list (set ℝ), 
                   (∀ t ∈ intervals, is_monotone_in_interval y t) ∧
                   intervals.length = 3) → (b < -2) ∨ (b > 3) :=
by
  intro h
  -- Proof skipped
  sorry

end range_b_three_monotonic_intervals_l142_142688


namespace halfway_point_l142_142539

theorem halfway_point (x1 x2 : ℚ) (h1 : x1 = 1 / 6) (h2 : x2 = 5 / 6) : 
  (x1 + x2) / 2 = 1 / 2 :=
by
  sorry

end halfway_point_l142_142539


namespace triangle_angles_l142_142889

noncomputable def theta : ℝ :=
  Real.arccos (- (1 + Real.sqrt 2) / 12)

noncomputable def alpha : ℝ :=
  Real.arccos ((34 + 2 * Real.sqrt 2) / (20 + 4 * Real.sqrt 2))

noncomputable def beta : ℝ := 180 - theta - alpha

theorem triangle_angles :
  ∃ (θ α β : ℝ), θ = Real.arccos (- (1 + Real.sqrt 2) / 12)
    ∧ α = Real.arccos ((34 + 2 * Real.sqrt 2) / (20 + 4 * Real.sqrt 2))
    ∧ β = 180 - θ - α :=
begin
  use [theta, alpha, beta],
  split; [refl, split; [refl, refl]],
  sorry
end

end triangle_angles_l142_142889


namespace leftmost_blue_and_on_rightmost_red_and_off_prob_l142_142477

/-- 
Ryan has 4 red lava lamps and 4 blue lava lamps. He arranges them 
in a row on a shelf randomly, and then randomly turns on 4 of them.
What is the probability that the leftmost lamp is blue and on, and 
the rightmost lamp is red and off? 
-/
noncomputable def probability_leftmost_blue_on_rightmost_red_off : ℚ :=
let total_arrangements := (Nat.comb 8 4) * (Nat.comb 8 4) in
let favorable_arrangements :=
  (Nat.comb 7 3) * (Nat.comb 7 3) * (Nat.comb 6 3) in
favorable_arrangements / total_arrangements

theorem leftmost_blue_and_on_rightmost_red_and_off_prob 
  (h : probability_leftmost_blue_on_rightmost_red_off = 5 / 7) : 
  true := 
by sorry

end leftmost_blue_and_on_rightmost_red_and_off_prob_l142_142477


namespace number_is_0_point_5_l142_142591

theorem number_is_0_point_5 (x : ℝ) (h : x = 1/6 + 0.33333333333333337) : x = 0.5 := 
by
  -- The actual proof would go here.
  sorry

end number_is_0_point_5_l142_142591


namespace probability_leq_0_l142_142714

noncomputable def normal_distribution (μ σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.Measure.toProbabilityMeasure (MeasureTheory.Measure.gaussian μ σ)

variable (X : ℝ → ℝ)

-- Given conditions
axiom X_normal : (X ~ normal_distribution 1 (σ squared))
axiom P_leq_2 : MeasureTheory.ProbabilityMeasure.probability (set.Iic 2) = 0.72

-- The proof problem
theorem probability_leq_0 : MeasureTheory.ProbabilityMeasure.probability (set.Iic 0) = 0.28 :=
begin
  sorry
end

end probability_leq_0_l142_142714


namespace smallest_possible_sector_angle_l142_142014

theorem smallest_possible_sector_angle : ∃ a₁ d : ℕ, 2 * a₁ + 9 * d = 72 ∧ a₁ = 9 :=
by
  sorry

end smallest_possible_sector_angle_l142_142014


namespace triangle_area_difference_l142_142787

-- Definitions based on given lengths and right angles.
def GH : ℝ := 5
def HI : ℝ := 7
def FG : ℝ := 9

-- Note: Right angles are implicitly used in the area calculations and do not need to be represented directly in Lean.
-- Define areas for triangles involved.
def area_FGH : ℝ := 0.5 * FG * GH
def area_GHI : ℝ := 0.5 * GH * HI
def area_FHI : ℝ := 0.5 * FG * HI

-- Define areas of the triangles FGJ and HJI using variables.
variable (x y z : ℝ)
axiom area_FGJ : x = area_FHI - z
axiom area_HJI : y = area_GHI - z

-- The main proof statement involving the difference.
theorem triangle_area_difference : (x - y) = 14 := by
  sorry

end triangle_area_difference_l142_142787


namespace find_integer_pairs_l142_142244

theorem find_integer_pairs (x y : ℤ) (h_xy : x ≤ y) (h_eq : (1 : ℚ)/x + (1 : ℚ)/y = 1/4) :
  (x, y) = (5, 20) ∨ (x, y) = (6, 12) ∨ (x, y) = (8, 8) ∨ (x, y) = (-4, 2) ∨ (x, y) = (-12, 3) :=
sorry

end find_integer_pairs_l142_142244


namespace determine_d_l142_142632

theorem determine_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 2 = d + (a + b + c - d)^(1/3)) : d = 1/2 := by
  sorry

end determine_d_l142_142632


namespace range_of_5m_minus_n_l142_142713

-- Definitions of conditions
variable {R : Type*} [LinearOrder R] [HasAdd R] [Neg R]
variable (f : R → R)
variable {m n : R}

-- Condition: f is an odd function
def is_odd (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

-- Condition: f is decreasing on R
def is_decreasing (f : R → R) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

-- Problem conditions:
def problem_conditions (f : R → R) (m n : R) : Prop :=
  is_odd f ∧ is_decreasing f ∧ (f m + f (n - 2) ≤ 0) ∧ (f (m - n - 1) ≤ 0)

-- The theorem to be proven:
theorem range_of_5m_minus_n (f : R → R) (m n : R) :
  problem_conditions f m n → (5 * m - n ≥ 7) :=
by
  sorry

end range_of_5m_minus_n_l142_142713


namespace shaded_region_area_l142_142786

theorem shaded_region_area (r : ℝ) (h : r = 5) : 
  8 * (π * r * r / 4 - r * r / 2) / 2 = 50 * (π - 2) :=
by
  sorry

end shaded_region_area_l142_142786


namespace least_possible_value_of_n_l142_142183

noncomputable theory

variables (n d : ℕ)

def cost_per_radio (d n : ℕ) : ℕ := d / n
def income_from_contributed_radios (d n : ℕ) : ℕ := d / n
def remaining_radios (n : ℕ) : ℕ := n - 3
def selling_price_remaining_radios (d n : ℕ) : ℕ := cost_per_radio d n + 10
def income_from_remaining_radios (d n : ℕ) : ℕ :=
  remaining_radios n * selling_price_remaining_radios d n
def total_income (d n : ℕ) : ℕ := 
  income_from_contributed_radios d n + income_from_remaining_radios d n
def total_profit (d n : ℕ) : ℕ := 
  total_income d n - d

theorem least_possible_value_of_n (d : ℕ) (h : d > 0) : n = 13 :=
by
  have h1 : total_profit d n = 100 := sorry
  have h2 : 10 * n - 30 = 100 := sorry
  exact h2 ▸ h1

end least_possible_value_of_n_l142_142183


namespace option_C_increasing_sequence_l142_142205

def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem option_C_increasing_sequence :
  is_increasing_sequence (λ n, 2 * n^2 - 5 * n + 1) :=
sorry

end option_C_increasing_sequence_l142_142205


namespace triangle_area_relation_l142_142304

theorem triangle_area_relation (ABC : Triangle) 
  (A B C B₁ B₂ B₃ C₁ C₂ C₃ C₄ : Point) 
  (h_segments_AB : AB_distance_eq (AB, (AB₁, B₁B₂, B₂B₃, B₃B)) (divide AB 4))
  (h_segments_AC : AC_distance_eq (AC, (AC₁, C₁C₂, C₂C₃, C₃C₄, C₄C)) (divide AC 5))
  : 
  2 * (area_C1B1C2 + area_C2B2C3 + area_C3B3C4 + area_C4BC) = area_ABC :=
sorry

end triangle_area_relation_l142_142304


namespace constant_term_correct_l142_142941

theorem constant_term_correct:
    ∀ (a k n : ℤ), 
      (∀ x : ℤ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
      → a - n + k = 7 
      → n = -6 := 
by
    intros a k n h h2
    have h1 := h 0
    sorry

end constant_term_correct_l142_142941


namespace sum_of_smallest_two_consecutive_numbers_l142_142095

theorem sum_of_smallest_two_consecutive_numbers (n : ℕ) (h : n * (n + 1) * (n + 2) = 210) : n + (n + 1) = 11 :=
sorry

end sum_of_smallest_two_consecutive_numbers_l142_142095


namespace total_texts_sent_l142_142056

def texts_sent_monday_allison : ℕ := 5
def texts_sent_monday_brittney : ℕ := 5
def texts_sent_tuesday_allison : ℕ := 15
def texts_sent_tuesday_brittney : ℕ := 15

theorem total_texts_sent : (texts_sent_monday_allison + texts_sent_monday_brittney) + 
                           (texts_sent_tuesday_allison + texts_sent_tuesday_brittney) = 40 :=
by
  sorry

end total_texts_sent_l142_142056


namespace num_natural_numbers_divisible_by_7_l142_142382

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l142_142382


namespace divisor_exists_l142_142127

theorem divisor_exists : ∀ (s : Finset ℕ),
  (∀ x ∈ s, 1 ≤ x ∧ x ≤ 2014) → s.card = 1008 →
  ∃ a b ∈ s, a ∣ b ∧ a ≠ b :=
by
  sorry

end divisor_exists_l142_142127


namespace divisible_by_7_in_range_200_to_400_l142_142378

theorem divisible_by_7_in_range_200_to_400 : 
  ∃ n : ℕ, 
    (∀ (x : ℕ), (200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ finset.range (201)) ∧ finset.card (finset.filter (λ x, 200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0) (finset.range 401)) = 29) := 
begin
  sorry
end

end divisible_by_7_in_range_200_to_400_l142_142378


namespace monotonic_intervals_range_of_b_existence_of_b_and_x4_l142_142335

open Real

noncomputable def f (a b x : ℝ) : ℝ := (x - a) ^ 2 * (x + b) * exp x

theorem monotonic_intervals (a b : ℝ) (ha : a = 0) (hb : b = -3) :
  (strict_mono_on (λ x, f a b x) {x | x < -3} ∧
   strict_mono_on (λ x, f a b x) {x | 0 < x ∧ x < 2} ∧
   strict_mono_on (λ x, f a b x) {x | x > 3} ∧
   strict_anti_on (λ x, f a b x) {x | -3 < x ∧ x < 0} ∧
   strict_anti_on (λ x, f a b x) {x | 2 < x ∧ x < 3}) :=
sorry

theorem range_of_b (a b : ℝ) (ha : a = 0) :
  x = a → (b < 0) :=
sorry

theorem existence_of_b_and_x4 (a : ℝ) (hxa : ∃ x1 x2 x3, x1 < x2 ∧ x2 < x3 ∧ x = a) :
  ∃ (b x4 : ℝ), (x4 = a + 2 * sqrt 6 ∨ x4 = a - 2 * sqrt 6 ∨
                 x4 = a + (1 + sqrt 13) / 2 ∨ x4 = a + (1 - sqrt 13) / 2) :=
sorry

end monotonic_intervals_range_of_b_existence_of_b_and_x4_l142_142335


namespace f_2005_l142_142685

noncomputable def f : ℝ → ℝ := sorry -- Define f as a real-valued function

axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom functional_equation : ∀ x : ℝ, f(x + 6) = f(x) + f(3)
axiom initial_condition : f(1) = 2

theorem f_2005 : f(2005) = 2 := by
  sorry

end f_2005_l142_142685


namespace part1_part2_l142_142447

-- Part (1)
theorem part1 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a n ^ 2 + 2 * a n + 1 = 4 * S n) :
  ∀ n, n >= 1 → a n = 2 * n - 1 :=
by sorry

-- Part (2)
theorem part2 (a : ℕ → ℕ)
  (h1 : ∀ n, a n = 2 * n - 1) :
  ∀ n, let b k := (-1) ^ k * (4 * k) / (a k * a (k + 1))
        in (∑ k in Finset.range (n + 1), b k)
        = -1 + (-1) ^ n * (1 / (2 * n + 1)) :=
by sorry

end part1_part2_l142_142447


namespace hyperbola_equations_l142_142323

theorem hyperbola_equations (a b : ℝ) (h1 : b / a = 2 / 3) (h2 : 2 * a = 12) :
  (\forall x y : ℝ, x^2 / 36 - y^2 / 16 = 1) ∨ (\forall x y : ℝ, y^2 / 36 - x^2 / 16 = 1) :=
by
  sorry

end hyperbola_equations_l142_142323


namespace trigonometric_identity_l142_142251

theorem trigonometric_identity : 
  sin (π / 2 + π / 3) + cos (π / 2 - π / 6) = 1 :=
by
  sorry

end trigonometric_identity_l142_142251


namespace cone_volume_ratio_l142_142142

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

theorem cone_volume_ratio :
  let rA : ℝ := 14.8
  let hA : ℝ := 28.3
  let rB : ℝ := 28.3
  let hB : ℝ := 14.8 in
  (cone_volume rA hA) / (cone_volume rB hB) = 148 / 283 :=
by
  sorry

end cone_volume_ratio_l142_142142


namespace triangle_inequality_l142_142438

theorem triangle_inequality (A B C P : Point) (S : ℝ) 
  (hABC : area A B C = S) : AP + BP + CP ≥ 2 * (3 ^ (1/4)) * sqrt S := by sorry

end triangle_inequality_l142_142438


namespace surface_area_of_sphere_l142_142312

-- Define the points on the surface of the sphere
variables {O P A B C : Type} 

-- Define distances indicating pairwise perpendicularity and the distances from P to A, B, and C
variables (hpab : ∃ (A B C : Type), PA = 1 ∧ PB = 2 ∧ PC = 3 ∧ PA ⊥ PB ∧ PB ⊥ PC ∧ PC ⊥ PA)

-- The theorem statement asserting the surface area of the sphere
theorem surface_area_of_sphere (PA PB PC : ℝ) (O P A B C : Type)
  (hyp : PA = 1 ∧ PB = 2 ∧ PC = 3 ∧ PA ⊥ PB ∧ PB ⊥ PC ∧ PC ⊥ PA) : 
  4 * Real.pi * (sqrt (1^2 + 2^2 + 3^2) / 2) ^ 2 = 14 * Real.pi := 
sorry

end surface_area_of_sphere_l142_142312


namespace expected_value_m_plus_n_l142_142194

-- Define the main structures and conditions
def spinner_sectors : List ℚ := [-1.25, -1, 0, 1, 1.25]
def initial_value : ℚ := 1

-- Define a function that returns the largest expected value on the paper
noncomputable def expected_largest_written_value (sectors : List ℚ) (initial : ℚ) : ℚ :=
  -- The expected value calculation based on the problem and solution analysis
  11/6  -- This is derived from the correct solution steps not shown here

-- Define the final claim
theorem expected_value_m_plus_n :
  let m := 11
  let n := 6
  expected_largest_written_value spinner_sectors initial_value = 11/6 → m + n = 17 :=
by sorry

end expected_value_m_plus_n_l142_142194


namespace oliver_rearrangements_time_l142_142466

theorem oliver_rearrangements_time :
  let unique_letters := 6
  let rate_per_minute := 15
  let total_rearrangements := Nat.factorial unique_letters
  let time_in_minutes := total_rearrangements / rate_per_minute
  let time_in_hours := time_in_minutes / 60
  time_in_hours = 0.8 :=
by
  let unique_letters := 6
  let rate_per_minute := 15
  let total_rearrangements := Nat.factorial unique_letters
  let time_in_minutes := total_rearrangements / rate_per_minute
  let time_in_hours := time_in_minutes / 60
  sorry

end oliver_rearrangements_time_l142_142466


namespace count_paths_to_form_2005_l142_142076

/-- Define the structure of a circle label. -/
inductive CircleLabel
| two
| zero
| five

open CircleLabel

/-- Define the number of possible moves from each circle. -/
def moves_from_two : Nat := 6
def moves_from_zero_to_zero : Nat := 2
def moves_from_zero_to_five : Nat := 3

/-- Define the total number of paths to form 2005. -/
def total_paths : Nat := moves_from_two * moves_from_zero_to_zero * moves_from_zero_to_five

/-- The proof statement: The total number of different paths to form the number 2005 is 36. -/
theorem count_paths_to_form_2005 : total_paths = 36 :=
by
  sorry

end count_paths_to_form_2005_l142_142076


namespace solve_z_l142_142331

noncomputable def z1 : ℂ := 5 + 10 * complex.I
noncomputable def z2 : ℂ := 3 - 4 * complex.I
noncomputable def z_inv : ℂ := (1 / z1) + (1 / z2)

theorem solve_z : ∃ z : ℂ, z ≠ 0 ∧ z_inv = (1 / z) ∧ z = 5 - (5 / 2) * complex.I :=
by
  sorry

end solve_z_l142_142331


namespace salary_of_A_l142_142158

theorem salary_of_A (x y : ℝ) (h₁ : x + y = 4000) (h₂ : 0.05 * x = 0.15 * y) : x = 3000 :=
by {
    sorry
}

end salary_of_A_l142_142158


namespace angle_MKO_right_l142_142301

-- Define the semicircle with diameter AB and center O
variables {A B O : Point}
variables (h1 : Circle (segment A B) O)

-- Define the line intersecting the semicircle at points C and D and the line AB at point M
variables {C D M : Point}
variables (h2 : intersects C D h1) (h3 : intersects M (segment A B))
variables (h4 : distance M B < distance M A) (h5 : distance M D < distance M C)

-- Define the circumcircles of triangles AOC and DOB intersecting at K
variables {K : Point}
variables (h6 : circumcircle_intersects K (triangle A O C) (triangle D O B)) (h7 : K ≠ O)

-- The goal is to prove that angle MKO is a right angle
theorem angle_MKO_right : angle M K O = 90 :=
sorry

end angle_MKO_right_l142_142301


namespace youtube_dislikes_l142_142575

theorem youtube_dislikes (likes : ℕ) (initial_dislikes : ℕ) (final_dislikes : ℕ) :
  likes = 3000 →
  initial_dislikes = likes / 2 + 100 →
  final_dislikes = initial_dislikes + 1000 →
  final_dislikes = 2600 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end youtube_dislikes_l142_142575


namespace find_h_neg_a_l142_142707

-- Definitions of even and odd functions
def even (f: ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def odd (g: ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Given conditions
variables (f g : ℝ → ℝ) (a : ℝ)
variable (h : ℝ → ℝ)
variable h_def : ∀ x, h x = f x + g x - 1
variable f_even : even f
variable g_odd : odd g
variable f_a : f a = 2
variable g_a : g a = 3

-- Proof goal
theorem find_h_neg_a : h (-a) = -2 := by sorry

end find_h_neg_a_l142_142707


namespace function_not_satisfy_condition_l142_142970

-- Definitions of the functions
def f1 (x: ℝ) := abs x
def f2 (x: ℝ) := x - abs x
def f3 (x: ℝ) := x + 1
def f4 (x: ℝ) := -x

-- Statement to prove
theorem function_not_satisfy_condition :
  (∀ f, f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4 → (¬ (f = f3 ∧ f (2 * x) = 2 * f x))) → true :=
begin
  sorry
end

end function_not_satisfy_condition_l142_142970


namespace track_length_is_500_l142_142219

-- Given Conditions
variables (B_speed S_speed : ℝ) -- Brenda and Sally's speeds
variable (track_length : ℝ) -- Length of the track
-- Brenda and Sally start at diametrically opposite points and run in opposite directions
-- They meet for the first time after Brenda has run 100 meters
variable h_meet_first : B_speed * 100 / (B_speed + S_speed) = track_length / 2
-- They meet for the second time after Sally has run 150 meters past their first meeting point
variable h_meet_second : S_speed * 150 / (B_speed + S_speed) + track_length / 2 = track_length

-- Theorem to be proved: the length of the track is 500 meters
theorem track_length_is_500 (h1 : B_speed ≠ 0) (h2 : S_speed ≠ 0) :
  track_length = 500 := 
sorry

end track_length_is_500_l142_142219


namespace sum_series_l142_142819

noncomputable def s : ℝ := Classical.choose $ exists_unique_of_exists_of_unique
  (exists.intro (by { sorry : s > 0 -> s^3 + (1/4) * s - 1 == 0, sorry }))
  (by { intros x y (hx : x^3 + (1/4) * x = 1) (hy : y^3 + (1/4) * y = 1), sorry })

theorem sum_series (s_pos : 0 < s) (h_s : s^3 + (1/4) * s - 1 = 0) :
  (s^2 + 2 * s^5 + 3 * s^8 + 4 * s^11 + ···) = 16 :=
begin
  sorry
end

end sum_series_l142_142819


namespace range_of_m_l142_142687

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : 
  (x + y ≥ m) → m ≤ 18 :=
sorry

end range_of_m_l142_142687


namespace job_hourly_rate_l142_142432

theorem job_hourly_rate (x : ℝ) (h_daily : 3 * x + 2 * 10 + 4 * 12) (h : 5 * h_daily = 445) : x = 7 :=
by
  sorry

end job_hourly_rate_l142_142432


namespace total_students_count_l142_142564

-- Define variables for the number of friends in each category
variables (M P G MP MG PG MPG T : ℕ)

-- Given conditions
axiom h1 : M = 10
axiom h2 : P = 20
axiom h3 : G = 5
axiom h4 : MP = 4
axiom h5 : MG = 2
axiom h6 : PG = 0
axiom h7 : MPG = 2

-- Using the principle of inclusion-exclusion to calculate T
def total_students := M + P + G - MP - MG - PG + MPG

-- The theorem to prove
theorem total_students_count : total_students M P G MP MG PG MPG = 31 :=
by
  rw [h1, h2, h3, h4, h5, h6, h7]
  simp
  sorry

end total_students_count_l142_142564


namespace number_divisor_property_l142_142132

theorem number_divisor_property (s : Set ℕ) (h_s : s ⊆ Finset.range 2015) (h_size : s.card = 1008) :
  ∃ a b ∈ s, a ≠ b ∧ a ∣ b := 
by
  sorry

end number_divisor_property_l142_142132


namespace ralph_total_cost_l142_142473

def initial_cart_value := 54.00
def small_issue_item_original_cost := 20.00
def small_issue_discount_rate := 0.20
def additional_item_cost := 15.00
def coupon_discount_rate := 0.10
def sales_tax_rate := 0.05

theorem ralph_total_cost : 
    let small_issue_discount := small_issue_item_original_cost * small_issue_discount_rate in
    let discounted_small_issue_item_cost := small_issue_item_original_cost - small_issue_discount in
    let subtotal := initial_cart_value + discounted_small_issue_item_cost + additional_item_cost in
    let coupon_discount := subtotal * coupon_discount_rate in
    let after_coupon := subtotal - coupon_discount in
    let sales_tax := after_coupon * sales_tax_rate in
    let total_cost := after_coupon + sales_tax in
    total_cost = 80.33 :=
by
    sorry

end ralph_total_cost_l142_142473


namespace alan_needs_add_58_wings_in_first_minute_l142_142804

noncomputable def alan_wings_eaten (minute : ℕ) : ℝ :=
  (if minute < 8 then 5 - 0.5 * minute else 0)

noncomputable def alan_total_wings : ℝ :=
  ∑ i in Finset.range 8, alan_wings_eaten i

noncomputable def lisa_wings_eaten (minute : ℕ) : ℝ :=
  if minute < 4 then 7 else if minute < 8 then 14 else 0

noncomputable def lisa_total_wings : ℝ :=
  ∑ i in Finset.range 8, lisa_wings_eaten i

def Alan_needs_additional_wings : ℝ :=
  (lisa_total_wings - alan_total_wings)

theorem alan_needs_add_58_wings_in_first_minute :
  Alan_needs_additional_wings = 58 := by
    sorry

end alan_needs_add_58_wings_in_first_minute_l142_142804


namespace divisible_by_7_in_range_200_to_400_l142_142375

theorem divisible_by_7_in_range_200_to_400 : 
  ∃ n : ℕ, 
    (∀ (x : ℕ), (200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ finset.range (201)) ∧ finset.card (finset.filter (λ x, 200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0) (finset.range 401)) = 29) := 
begin
  sorry
end

end divisible_by_7_in_range_200_to_400_l142_142375


namespace expected_value_greater_than_median_l142_142583

noncomputable def density_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x, x < a → f x = 0) ∧ 
  (∀ x, x ≥ b → f x = 0) ∧
  (∀ x, a ≤ x → x < b → f x > 0) ∧
  (∀ x y, a ≤ x → x ≤ y → y < b → f y ≤ f x)


theorem expected_value_greater_than_median
  (X : Type) [Probability.ZeroOneOneClassDensity X]
  (f : ℝ → ℝ) (a b : ℝ)
  (h1 : density_function f a b)
  (h2 : ∃ x, is_median X x) :
  expected_value X > median X := 
sorry

end expected_value_greater_than_median_l142_142583


namespace inclination_angle_of_line_l142_142726

theorem inclination_angle_of_line (α : ℝ) (hα : α ∈ Ioo (π / 2) π) :
    ∃ θ : ℝ, θ = π - α ∧ tan θ = -tan α :=
by 
  use π - α
  sorry

end inclination_angle_of_line_l142_142726


namespace problem1_problem2_l142_142040

def f (a b : ℝ) (x : ℝ) := a * real.log x + b * x
def g (x : ℝ) := x ^ 2
def k := 2
def m := -1

theorem problem1 (a b : ℝ) (h1 : a = 1) (h2 : b = 1) (x : ℝ) :
  (f a b x ≤ k * x + m) ∧ (g x ≥ k * x + m) :=
sorry

def G (a b : ℝ) (x : ℝ) := g x - f a b x + 2
def x0 (x1 x2 : ℝ) := (x1 + x2) / 2

theorem problem2 (a b : ℝ) (x1 x2 : ℝ) (h1 : G a b x1 = 0) (h2 : G a b x2 = 0) (h3 : a > 0) :
  let x0 := (x1 + x2) / 2 in G(a b (x0)) > 0 :=
sorry

end problem1_problem2_l142_142040


namespace root_conditions_imply_sum_l142_142399

-- Define the variables a and b in the context that their values fit the given conditions.
def a : ℝ := 5
def b : ℝ := -6

-- Define the quadratic equation and conditions on roots.
def quadratic_eq (x : ℝ) := x^2 - a * x - b

-- Given that 2 and 3 are the roots of the quadratic equation.
def roots_condition := (quadratic_eq 2 = 0) ∧ (quadratic_eq 3 = 0)

-- The theorem to prove.
theorem root_conditions_imply_sum :
  roots_condition → a + b = -1 :=
by
sorry

end root_conditions_imply_sum_l142_142399


namespace initial_points_count_l142_142855

theorem initial_points_count :
  ∃ n_0 : ℕ, 
  (∃ n_1 : ℕ, n_1 = 2 * n_0 - 1) ∧ 
  (∃ n_2 : ℕ, n_2 = 2 * n_1 - 1) ∧ 
  (n_3 = 2 * n_2 - 1) ∧ 
  (n_3 = 113) ∧ 
  n_0 = 15 :=
by {
  let n_3 := 113,
  let n_2 := (113 + 1) / 2,
  let n_1 := (n_2 + 1) / 2,
  let n_0 := (n_1 + 1) / 2,
  use n_0,
  split; try {use n_1, assumption},
  split; try {use n_2, assumption},
  split, exact rfl,
  exact rfl,
  sorry
}

end initial_points_count_l142_142855


namespace raise_back_to_original_l142_142159

-- Define originalSalary and reducedSalary
def originalSalary: ℝ := 100
def reducedSalary : ℝ := originalSalary * 0.65

-- We want to verify the percentage increase x
def percentageIncreaseToOriginal (originalSalary reducedSalary : ℝ) : ℝ :=
  35 / 65 * 100

-- Proof statement
theorem raise_back_to_original (originalSalary reducedSalary : ℝ) :
  reducedSalary = originalSalary * 0.65 →
  let x := percentageIncreaseToOriginal originalSalary reducedSalary in
  reducedSalary * (1 + x / 100) = originalSalary :=
by
  sorry -- Proof is omitted

end raise_back_to_original_l142_142159


namespace small_boxes_count_l142_142587

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) (num_small_boxes : ℕ) :
  total_chocolates = 504 →
  chocolates_per_box = 28 →
  total_chocolates / chocolates_per_box = num_small_boxes →
  num_small_boxes = 18 :=
by
  intros h_total h_each h_div
  rw [h_total, h_each] at h_div
  simp at h_div
  exact h_div

end small_boxes_count_l142_142587


namespace max_non_managers_l142_142156

theorem max_non_managers (m n : ℕ) (h : m = 11) (ratio_condition : (m : ℚ) / n > 7 / 37) : n ≤ 58 :=
by
  have h1 : (11 : ℚ) / n > 7 / 37, from by rwa [h] at ratio_condition
  have h2 : (11 : ℚ) / (7 / 37) > n, from by apply h1.trans (7 / 37)
  have h3 : (11 * 37) / 7 > n, from by rw div_eq_mul_inv; apply mul_lt_mul_of_pos_right _ (inv_pos.mpr (by norm_num))
  have h4 : 407 / 7 > n, from by norm_num1
  exact nat.ceil_le h4

end max_non_managers_l142_142156


namespace area_of_quadrilateral_AEDC_l142_142794
-- Import the Mathlib library for access to comprehensive mathematics capabilities.

-- Define the problem in Lean 4, considering $PE = 3$, $PD = 4$, and $DE = 5$.
noncomputable def area_quadrilateral_AEDC : ℝ :=
  let PE := 3
  let PD := 4
  let DE := 5
  let area_triangle := λ a b : ℝ, (1 / 2) * a * b
  area_triangle 8 3 + area_triangle 4 3 + area_triangle 6 4 + area_triangle 8 6

-- State the theorem that the area of quadrilateral AEDC is 54.
theorem area_of_quadrilateral_AEDC : area_quadrilateral_AEDC = 54 := by
  sorry

end area_of_quadrilateral_AEDC_l142_142794


namespace polygon_num_sides_and_exterior_angle_l142_142102

theorem polygon_num_sides_and_exterior_angle 
  (n : ℕ) (x : ℕ) 
  (h : (n - 2) * 180 + x = 1350) 
  (hx : 0 < x ∧ x < 180) 
  : (n = 9) ∧ (x = 90) := 
by 
  sorry

end polygon_num_sides_and_exterior_angle_l142_142102


namespace functions_with_inverses_l142_142996

theorem functions_with_inverses:
    (∀ x ∈ set.Iic 3, ∃ y, sqrt (3 - y) = x)
    ∧ (∀ x (hx : x > 0), ∃ y, x = y - 2 / y)
    ∧ (∀ x ∈ set.Ici (-3), ∃ y, x = (y + 3)^2 + 1)
    ∧ (∀ x, ∃ y, x = 2^y + 9^y)
    ∧ (∀ x ∈ set.Ico (-3) 9, ∃ y, x = y / 3) := sorry

end functions_with_inverses_l142_142996


namespace probability_even_N_l142_142094

noncomputable def probability_N_even : ℚ := 181 / 361

theorem probability_even_N :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 20 },
      C := S.to_finset.powerset.filter (λ s, s.card = 2),
      uniform_sample := (C ×ˢ C).to_finset.prod (λ _, 1 / ((card C).to_rat * (card C).to_rat)) in
  ∃ (a b c d : ℕ), 
    (a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a < b ∧ c < d) ∧
    let N := { n | (a ≤ n ∧ n ≤ b ⊕ c ≤ n ∧ n ≤ d) ∧ ¬((a ≤ n ∧ n ≤ b) ∧ (c ≤ n ∧ n ≤ d)) },
        even_N := (card N) % 2 = 0 in
  ∑' (x : (S × S) × (S × S)), 
    if even_N then uniform_sample else 0 = probability_N_even :=
  sorry

end probability_even_N_l142_142094


namespace line_intersects_y_axis_at_point_l142_142614

open Function

theorem line_intersects_y_axis_at_point :
  ∃ (y : ℝ), (∃ m b : ℝ, (∀ x, y = m * 0 + b) ∧ (m = (15 - 3) / (6 - 2)) ∧ (b = 3 - m * 2)) ∧ y = -3 :=
by
  sorry

end line_intersects_y_axis_at_point_l142_142614


namespace length_of_each_part_l142_142548

theorem length_of_each_part (ft : ℕ) (inch : ℕ) (parts : ℕ) (total_length : ℕ) (part_length : ℕ) :
  ft = 6 → inch = 8 → parts = 5 → total_length = 12 * ft + inch → part_length = total_length / parts → part_length = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end length_of_each_part_l142_142548


namespace tan_fraction_power_l142_142848

noncomputable theory

open Complex Real

theorem tan_fraction_power (α : ℝ) (n : ℕ) :
  ( (1 + Complex.i * (Real.sin α / Real.cos α)) / 
    (1 - Complex.i * (Real.sin α / Real.cos α)) ) ^ n = 
  (1 + Complex.i * (Real.sin (n * α) / Real.cos (n * α))) / 
  (1 - Complex.i * (Real.sin (n * α) / Real.cos (n * α))) :=
sorry

end tan_fraction_power_l142_142848


namespace vertex_of_f1_range_of_f1_expression_of_g_l142_142298

section
variable (x : ℝ)

-- Define the quadratic function with parameters b and c.
def quadratic (b c : ℝ) : ℝ := -x^2 + b*x + c

-- Part 1: Specific values of b and c.
def f1 : ℝ := quadratic 4 3 x

-- Problem 1: Prove that the vertex of f1 is (2, 7).
theorem vertex_of_f1 : ∃! (p : ℝ × ℝ), p = (2, 7) ∧ ∀ x, f1 ≤ f1 2 := sorry

-- Problem 2: Prove that the range of f1 on [-1, 3] is [-2, 7].
theorem range_of_f1 : ∀ y, (-2 ≤ y ∧ y ≤ 7) ↔ ∃ x, (-1 ≤ x ∧ x ≤ 3 ∧ y = f1) := sorry

-- Part 2: Given conditions on the function.
variable (y : ℝ)

-- Define a function to represent the conditions.
def g : ℝ := quadratic 2 2 x

-- When x ≤ 0, the maximum value is 2.
axiom max_value_x_leq_0 (x : ℝ) (h : x ≤ 0) : g ≤ 2

-- When x > 0, the maximum value is 3.
axiom max_value_x_gt_0 (x : ℝ) (h : x > 0) : g ≤ 3

-- Problem 3: Prove that the expression of the quadratic function is y = -x^2 + 2x + 2.
theorem expression_of_g : g = -x^2 + 2*x + 2 := sorry
end

end vertex_of_f1_range_of_f1_expression_of_g_l142_142298


namespace triangle_is_right_l142_142419

variable {A B C : ℝ}
variable {a b c : ℝ} -- sides corresponding to angles A, B, C

theorem triangle_is_right 
  (h : sin (A - B) = 1 + 2 * cos (B * C) * sin (A + C)) 
  (h_non_neg: 0 ≤ A ∧ A ≤ π ∧ 0 ≤ B ∧ B ≤ π ∧ 0 ≤ C ∧ C ≤ π) 
  (h_sum_angles: A + B + C = π) : 
  C = π / 2 :=
sorry

end triangle_is_right_l142_142419


namespace distinct_vertices_1995_l142_142895

noncomputable def vertices : Fin 20 → ℂ := λ i => exp(2 * Real.pi * Complex.I * (i : ℂ) / 20)

theorem distinct_vertices_1995 :
  (Finset.univ.image (λ (i : Fin 20), (vertices i) ^ 1995)).card = 4 :=
sorry

end distinct_vertices_1995_l142_142895


namespace probability_red_card_eq_one_seventh_l142_142576

def cards := {n : ℕ | 1 ≤ n ∧ n ≤ 70}

def is_red (n : ℕ) : Prop := n % 7 = 1

def red_cards := {n : ℕ | n ∈ cards ∧ is_red n}

noncomputable def probability_of_red_card : ℚ :=
  set_size red_cards / set_size cards

theorem probability_red_card_eq_one_seventh :
  probability_of_red_card = 1 / 7 := sorry

end probability_red_card_eq_one_seventh_l142_142576


namespace ratio_of_weights_l142_142510

variable (x y : ℝ)

theorem ratio_of_weights (h : x + y = 7 * (x - y)) (h1 : x > y) : x / y = 4 / 3 :=
sorry

end ratio_of_weights_l142_142510


namespace transform_uniform_random_l142_142110

theorem transform_uniform_random (a_1 : ℝ) (h : 0 ≤ a_1 ∧ a_1 ≤ 1) : -2 ≤ a_1 * 8 - 2 ∧ a_1 * 8 - 2 ≤ 6 :=
by sorry

end transform_uniform_random_l142_142110


namespace probability_non_perfect_power_200_l142_142881

def is_perfect_power (x : ℕ) : Prop := 
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b = x

def count_perfect_powers_up_to (n : ℕ) : ℕ := 
  Finset.card (Finset.filter is_perfect_power (Finset.range (n + 1)))

def probability_not_perfect_power (n : ℕ) : ℚ :=
  let total := n in
  let perfect_powers := count_perfect_powers_up_to n in
  (total - perfect_powers) / total

theorem probability_non_perfect_power_200 :
  probability_not_perfect_power 200 = 9 / 10 :=
by {
  -- statement placeholder
  sorry
}

end probability_non_perfect_power_200_l142_142881


namespace depth_of_well_l142_142179

noncomputable def volume_of_cylinder (radius : ℝ) (depth : ℝ) : ℝ :=
  Real.pi * radius^2 * depth

theorem depth_of_well (volume depth : ℝ) (r : ℝ) : 
  r = 1 ∧ volume = 25.132741228718345 ∧ 2 * r = 2 → depth = 8 :=
by
  intros h
  sorry

end depth_of_well_l142_142179


namespace prob_defective_l142_142897

variable {Ω : Type} [ProbabilityTheory Ω]

def P (event : Set Ω) : ℝ := sorry -- Assuming a general definition for probability function

-- Conditions
variable (A B C : Set Ω)
variable (hA : P A = 0.65)
variable (hB : P B = 0.3)
variable (hAB : Disjoint A B) -- A and B are mutually exclusive

-- To show
theorem prob_defective : P C = 0.05 :=
by
  have hComplement : C = Ω \ (A ∪ B) := sorry  -- Complement event definition
  rw [hComplement]
  rw [prob_compl, prob_union_disjoint hAB]
  rw [hA, hB]
  exact sorry -- Final steps skipping the CA steps

end prob_defective_l142_142897


namespace alice_charles_meeting_probability_l142_142969

theorem alice_charles_meeting_probability :
  let arrival_time_range : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}
  let overlap_condition (t₁ t₂ : ℝ) : Prop := t₁ - 0.5 ≤ t₂ ∧ t₂ <=  t₁ + 0.5
  let overlap_area : ℝ := 3.75
  let total_area : ℝ := 4
  abs (overlap_area / total_area - (15 / 16)) < 1e-9
by sorry

end alice_charles_meeting_probability_l142_142969


namespace sum_of_arithmetic_sequence_2008_terms_l142_142968

theorem sum_of_arithmetic_sequence_2008_terms :
  let a := -1776
  let d := 11
  let n := 2008
  let l := a + (n - 1) * d
  let S := (n / 2) * (a + l)
  S = 18599100 := by
  sorry

end sum_of_arithmetic_sequence_2008_terms_l142_142968


namespace max_distance_complex_l142_142034

theorem max_distance_complex (z : ℂ) (hz : complex.abs z = 3) : 
  ∃ d, d = complex.abs ((5 + 2 * complex.I) * z^3 - z^5) ∧ d ≤ 99 :=
by sorry

end max_distance_complex_l142_142034


namespace existence_of_function_implies_a_le_1_l142_142807

open Real

noncomputable def positive_reals := { x : ℝ // 0 < x }

theorem existence_of_function_implies_a_le_1 (a : ℝ) :
  (∃ f : positive_reals → positive_reals, ∀ x : positive_reals, 3 * (f x).val^2 = 2 * (f (f x)).val + a * x.val^4) → a ≤ 1 :=
by
  sorry

end existence_of_function_implies_a_le_1_l142_142807


namespace goalkeeper_not_at_goal_line_energy_consumed_correct_lob_shot_opportunities_correct_l142_142772

-- Movement record of the goalkeeper
def movements : List Int := [7, -3, 8, 4, -6, -8, 14, -15]

-- Question 1: The goalkeeper does not return to the goal line
theorem goalkeeper_not_at_goal_line : (movements.sum ≠ 0) :=
by
  -- Assume the proof here
  sorry

-- Question 2: The total energy consumed
noncomputable def total_energy_consumed : Float :=
let distances := movements.map Int.natAbs
distances.sum * 0.1

theorem energy_consumed_correct : total_energy_consumed = 6.5 :=
by
  -- Assume the proof here
  sorry

-- Question 3: Number of opportunities for a lob shot goal
def cumulative_distance(moves : List Int) : List Int :=
moves.scanl (+) 0 -- This generates the running total of the distances

def opportunities_for_lob_shot : List Int :=
(cumulative_distance movements).filter (λ x => abs x > 10)

theorem lob_shot_opportunities_correct : opportunities_for_lob_shot.length = 3 :=
by
  -- Assume the proof here
  sorry

end goalkeeper_not_at_goal_line_energy_consumed_correct_lob_shot_opportunities_correct_l142_142772


namespace distance_from_center_to_vertex_Q_is_5_l142_142236

-- Definitions coming from conditions
variables (P Q R S O : Type) [metric_space P] [ordered_field P]
variables (P_dist_Q : dist P Q = 10) (P_dist_R : dist P R = 10)
variables (S_tangent_to_QS : dist S Q = 5) 

-- The statement to prove
theorem distance_from_center_to_vertex_Q_is_5 :
  ∃ (O : Type), dist O Q = 5 :=
begin
  sorry
end

end distance_from_center_to_vertex_Q_is_5_l142_142236


namespace reinforcement_proof_l142_142926

theorem reinforcement_proof : 
  ∀ (initial_men : Nat) (initial_days : Nat) (days_lapsed : Nat) (additional_days : Nat) (initial_provisions : Nat),
    initial_men = 1000 →
    initial_days = 60 →
    days_lapsed = 15 →
    additional_days = 20 →
    initial_provisions = initial_men * initial_days →
    ∃ (reinforcement : Nat), reinforcement = 1250 ∧
    (initial_men + reinforcement) * additional_days = initial_provisions - initial_men * days_lapsed :=
by
  intros initial_men initial_days days_lapsed additional_days initial_provisions
  assume h1: initial_men = 1000,
         h2: initial_days = 60,
         h3: days_lapsed = 15,
         h4: additional_days = 20,
         h5: initial_provisions = initial_men * initial_days
  use 1250
  split
  · sorry
  · sorry

end reinforcement_proof_l142_142926


namespace intersection_cardinality_l142_142822

open Set

theorem intersection_cardinality :
  let M : Set ℕ := {0, 1, 2, 3, 4}
  let N : Set ℕ := {1, 3, 5}
  let P : Set ℕ := M ∩ N
  P.card = 2 := 
by {
  sorry
}

end intersection_cardinality_l142_142822


namespace divisibility_of_special_number_l142_142021

theorem divisibility_of_special_number (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
    ∃ d : ℕ, 100100 * a + 10010 * b + 1001 * c = 11 * d := 
sorry

end divisibility_of_special_number_l142_142021
