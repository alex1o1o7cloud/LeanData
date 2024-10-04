import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Periodic
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Hyperbolic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.CrossProduct
import Mathlib.Probability.Basic
import Mathlib.Probability.Independent
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.VectorSpace
import Mathlib.Topology.Basic
import Mathlib.Trigonometry.Basic
import data.nat.prime
import data.real.basic

namespace complex_conjugate_l127_127356

theorem complex_conjugate (z : ℂ) (h : (z - 3) * (2 - complex.i) = 5) :
  complex.conj z = 5 - complex.i :=
sorry

end complex_conjugate_l127_127356


namespace problem_statement_l127_127211

def f (x : ℝ) : ℝ := 3^x

theorem problem_statement (x y : ℝ) : f(x) * f(y) = f(x + y) :=
by
  sorry

end problem_statement_l127_127211


namespace functional_eq_solution_l127_127505

noncomputable def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x + y) * f(x - y) = f(x)^2 + f(y)^2 - 1

theorem functional_eq_solution (f : ℝ → ℝ) (c : ℝ) :
  (functional_eq f) →
  (∀ t : ℝ, has_deriv_at f (f t) t) →
  (∃ c : ℝ, ∀ t : ℝ, deriv (deriv f) t = c^2 * f t ∨ deriv (deriv f) t = -c^2 * f t) →
  (∀ t : ℝ, f t = cos (c * t) ∨ f t = cosh (c * t) ∨ f t = - cos (c * t) ∨ f t = - cosh (c * t)) :=
sorry

end functional_eq_solution_l127_127505


namespace phone_number_value_of_A_l127_127091

variable (A B C D E F G H I J : ℕ)

theorem phone_number_value_of_A 
  (h1 : A > B) (h2 : B > C)
  (h3 : D > E) (h4 : E > F)
  (h5 : G > H) (h6 : H > I) (h7 : I > J) 
  (h8 : D ∈ {2, 4, 6, 8}) 
  (h9 : D + 2 = E) 
  (h10 : E + 2 = F) 
  (h11 : G ∈ {1, 3, 5, 7, 9}) 
  (h12 : G - 2 = H) 
  (h13 : H - 2 = I) 
  (h14 : I - 4 = J) 
  (h15 : J = 1) 
  (h16 : A + B + C = 11) 
  : A = 8 :=
sorry

end phone_number_value_of_A_l127_127091


namespace cos_105_proof_l127_127853

noncomputable def cos_105_degrees : Real :=
  cos 105 * (π / 180)

theorem cos_105_proof : cos_105_degrees = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_proof_l127_127853


namespace rabbit_catch_up_time_l127_127032

theorem rabbit_catch_up_time :
  let rabbit_speed := 25 -- miles per hour
  let cat_speed := 20 -- miles per hour
  let head_start := 15 / 60 -- hours, which is 0.25 hours
  let initial_distance := cat_speed * head_start
  let relative_speed := rabbit_speed - cat_speed
  initial_distance / relative_speed = 1 := by
  sorry

end rabbit_catch_up_time_l127_127032


namespace tiles_covering_the_floor_l127_127094

theorem tiles_covering_the_floor 
  (L W : ℕ) 
  (h1 : (∃ k, L = 10 * k) ∧ (∃ j, W = 10 * j))
  (h2 : W = 2 * L)
  (h3 : (L * L + W * W).sqrt = 45) :
  L * W = 810 :=
sorry

end tiles_covering_the_floor_l127_127094


namespace b_51_is_5151_l127_127993

def sequence_a (n : Nat) : Nat :=
  n * (n + 1) / 2

def sequence_b (n : Nat) : Nat :=
  (Nat.rec (λ _, Nat) (λ k bk hk, if (k + 1) % 2 = 0 then hk else sequence_a (k + 1))
     (n - 1) Nat.zero)

theorem b_51_is_5151 : sequence_b 51 = 5151 :=
sorry

end b_51_is_5151_l127_127993


namespace solution_set_l127_127899

def f (x : ℝ) : ℝ := (3 * x - 4) * (x - 2) * (x + 1) / (x - 1)

theorem solution_set :
  {x : ℝ | f x ≥ 0} = {x | x ≤ -1} ∪ {x | 4/3 ≤ x ∧ x ≤ 2} ∪ {x | x > 2} :=
by
  sorry

end solution_set_l127_127899


namespace permutations_of_distinct_letters_l127_127227

theorem permutations_of_distinct_letters : 
  let lett := ["S", "T1", "A1", "R1", "T2", "A2", "R2", "T3"]
  (list.permutations lett).length = 5040 :=
by
  sorry

end permutations_of_distinct_letters_l127_127227


namespace find_y_value_l127_127598

theorem find_y_value (k c x y : ℝ) (h1 : c = 3) 
                     (h2 : ∀ x : ℝ, y = k * x + c)
                     (h3 : ∃ k : ℝ, 15 = k * 5 + 3) :
  y = -21 :=
by 
  sorry

end find_y_value_l127_127598


namespace equal_play_time_l127_127797

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l127_127797


namespace equal_play_time_l127_127798

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l127_127798


namespace cos_105_eq_l127_127862

theorem cos_105_eq : (cos 105) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by 
  have h1: cos (60:ℝ) = 1 / 2 := by sorry
  have h2: cos (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h3: sin (60:ℝ) = real.sqrt 3 / 2 := by sorry
  have h4: sin (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h5: cos (105:ℝ) = cos (60 + 45) := by sorry
  have h6: cos (60 + 45) = cos 60 * cos 45 - sin 60 * sin 45 := by sorry
  have h7 := calc
    cos 105 = (cos 60) * (cos 45) - (sin 60) * (sin 45) : by sorry
    ... = (1 / 2) * (real.sqrt 2 / 2) - (real.sqrt 3 / 2) * (real.sqrt 2 / 2) : by sorry
    ... = (real.sqrt 2 / 4) - (real.sqrt 6 / 4) : by sorry
    ... = (real.sqrt 2 - real.sqrt 6) / 4 : by sorry
  exact h7

end cos_105_eq_l127_127862


namespace probability_black_balls_l127_127507

variable {m1 m2 k1 k2 : ℕ}

/-- Given conditions:
  1. The total number of balls in both urns is 25.
  2. The probability of drawing one white ball from each urn is 0.54.
To prove: The probability of both drawn balls being black is 0.04.
-/
theorem probability_black_balls : 
  m1 + m2 = 25 → 
  (k1 * k2) * 50 = 27 * m1 * m2 → 
  ((m1 - k1) * (m2 - k2) : ℚ) / (m1 * m2) = 0.04 :=
by
  intros h1 h2
  sorry

end probability_black_balls_l127_127507


namespace each_player_plays_36_minutes_l127_127805

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l127_127805


namespace andrey_valentin_distance_l127_127119

/-- Prove the distance between Andrey and Valentin is 107 meters,
    given that Andrey finishes 60 meters ahead of Boris and Boris finishes 50 meters ahead of Valentin.
    Assumptions:
    - Andrey, Boris, and Valentin participated in a 1 km race.
    - All participants ran at a constant speed. -/
theorem andrey_valentin_distance :
  ∀ (a b c : ℝ), -- Speeds of Andrey, Boris, and Valentin respectively.
  (a ≠ 0) →     -- Non-zero speed
  (b = 0.94 * a) → 
  (c = 0.95 * b) →
  (distance_a := 1000 : ℝ) →
  (distance_valentin : ℝ := c * (distance_a / a)) →
  (distance_andrey_valentin : ℝ := distance_a - distance_valentin) →
  distance_andrey_valentin = 107 :=
by {
  intros a b c ha hb hc distance_a distance_valentin distance_andrey_valentin,
  subst_vars,
  sorry
}

end andrey_valentin_distance_l127_127119


namespace negation_is_returning_complement_is_returning_l127_127531

-- Negation returning transformation
theorem negation_is_returning (a : ℝ) : -(-a) = a := sorry

-- Set complement returning transformation within a universal set U
theorem complement_is_returning (A U : Set α) : (U ⊆ A → ¬(U ⊆ A) = A) := sorry

end negation_is_returning_complement_is_returning_l127_127531


namespace common_difference_of_arithmetic_sequence_l127_127252

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 9)
  (h2 : a 5 = 33)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = 8 :=
sorry

end common_difference_of_arithmetic_sequence_l127_127252


namespace positional_relationship_parallel_or_skew_l127_127601

variable {α β : Type _} [AffinePlane α] [AffinePlane β]
variable (a : Line α) (b : Line β)

theorem positional_relationship_parallel_or_skew
  (h_parallel_planes : Parallel α β)
  (h_a_in_alpha : a ⊆ α)
  (h_b_in_beta : b ⊆ β) :
  (Parallel a b ∨ Skew a b) :=
sorry

end positional_relationship_parallel_or_skew_l127_127601


namespace solution_x_is_325_l127_127340

noncomputable def solve_equation (x : ℝ) : Prop :=
  2 * x + real.sqrt (x - 3) = 7

theorem solution_x_is_325 : ∃ x : ℝ, solve_equation x ∧ x = 3.25 :=
by
  use 3.25
  split
  . sorry
  . rfl

end solution_x_is_325_l127_127340


namespace compute_series_l127_127851

theorem compute_series :
  (∑ i in finset.range 100, (3 + (i + 1) * 9) / 3 ^ (101 - i)) = 453 := 
  sorry

end compute_series_l127_127851


namespace graph_through_fixed_point_l127_127320

theorem graph_through_fixed_point (m : ℝ) (h : m ≠ 0) :
  (∃ x y, (x = 3) ∧ (y = -2) ∧ y = m * x - (3 * m + 2)) :=
by
  use [3, -2]
  split
  . refl
  split
  . refl
  calc 
    -2 = m * 3 - (3 * m + 2) : sorry

end graph_through_fixed_point_l127_127320


namespace tan_square_proof_l127_127178

theorem tan_square_proof (θ : ℝ) (h : Real.tan θ = 2) : 
  1 / (Real.sin θ ^ 2 - Real.cos θ ^ 2) = 5 / 3 := by
  sorry

end tan_square_proof_l127_127178


namespace max_product_sum_2020_l127_127167

theorem max_product_sum_2020 :
  ∃ (a : ℕ → ℕ) (n : ℕ), (∑ i in finset.range n, a i = 2020) ∧ (∏ i in finset.range n, a i = 2^2 * 3^672) :=
sorry

end max_product_sum_2020_l127_127167


namespace segments_form_triangle_l127_127550

noncomputable def can_form_triangle (A B C : Point) (M1 M2 M3 H1 H2 H3 : Point) : Prop :=
  midpoint A B M1 ∧ midpoint B C M2 ∧ midpoint A C M3 ∧
  foot_of_altitude A B C H1 ∧ foot_of_altitude B C A H2 ∧ foot_of_altitude C A B H3 ∧
  (segment_length H1 M2 + segment_length H2 M3 > segment_length H3 M1) ∧
  (segment_length H2 M3 + segment_length H3 M1 > segment_length H1 M2) ∧
  (segment_length H3 M1 + segment_length H1 M2 > segment_length H2 M3)

theorem segments_form_triangle (A B C M1 M2 M3 H1 H2 H3 : Point) :
  can_form_triangle A B C M1 M2 M3 H1 H2 H3 := sorry

end segments_form_triangle_l127_127550


namespace equal_play_time_l127_127799

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l127_127799


namespace homologous_functions_count_l127_127235

def homologous_count : Nat :=
  let xs := {-3, -2, 2, 3}
  
  -- Helper function to check if a domain yields the required range
  let valid_domain (domain : Set ℤ) : Bool :=
    let ys := Set.image (λ x, x ^ 2) domain
    ys = {4, 9}
  
  -- Generate all subsets of xs and filter those which produce the desired range
  let valid_domains := (Finset.powerset xs).filter (λ s, valid_domain s)
  
  -- Return the number of valid domains
  valid_domains.card

theorem homologous_functions_count :
  homologous_count = 9 :=
sorry

end homologous_functions_count_l127_127235


namespace inverse_28_mod_97_l127_127957

theorem inverse_28_mod_97 :
  ∀ (a : ℤ), (47 * a ≡ 1 [MOD 97]) → (28 * 26 ≡ 1 [MOD 97]) :=
by 
  sorry

end inverse_28_mod_97_l127_127957


namespace total_claps_20th_num_l127_127913

def fibonacci (n : ℕ) : ℕ :=
  if n = 1 ∨ n = 2 then 1 else fibonacci (n - 1) + fibonacci (n - 2)

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

def count_multiples_of_3 (max_n : ℕ) : ℕ :=
  (List.range (max_n + 1)).filter (λ n, is_multiple_of_3 (fibonacci n)).length

theorem total_claps_20th_num : count_multiples_of_3 20 = 5 :=
  sorry

end total_claps_20th_num_l127_127913


namespace simplify_expr1_simplify_expr2_l127_127846

-- Problem 1
theorem simplify_expr1 (x y : ℝ) : x^2 - 5 * y - 4 * x^2 + y - 1 = -3 * x^2 - 4 * y - 1 :=
by sorry

-- Problem 2
theorem simplify_expr2 (a b : ℝ) : 7 * a + 3 * (a - 3 * b) - 2 * (b - 3 * a) = 16 * a - 11 * b :=
by sorry

end simplify_expr1_simplify_expr2_l127_127846


namespace exists_good_vertex_l127_127509

open GraphTheory

theorem exists_good_vertex (G : Graph) [planar G] 
  (colored : ∀ e : G.edge, e.color = Color.blue ∨ e.color = Color.red) :
  ∃ v : G.vertex, 
    let edges := G.edges_of v in 
    (∃ (cycle : list G.edge) (h_cycle : cycle.is_cycle edges),
    (count_color_changes cycle ≤ 2)) :=
sorry

end exists_good_vertex_l127_127509


namespace max_x2_plus_2xy_plus_3y2_l127_127344

theorem max_x2_plus_2xy_plus_3y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 18 + 9 * Real.sqrt 3 :=
sorry

end max_x2_plus_2xy_plus_3y2_l127_127344


namespace find_ages_l127_127092

-- Define that f is a polynomial with integer coefficients
noncomputable def f : ℤ → ℤ := sorry

-- Given conditions
axiom f_at_7 : f 7 = 77
axiom f_at_b : ∃ b : ℕ, f b = 85
axiom f_at_c : ∃ c : ℕ, f c = 0

-- Define what we need to prove
theorem find_ages : ∃ b c : ℕ, (b - 7 ∣ 8) ∧ (c - b ∣ 85) ∧ (c - 7 ∣ 77) ∧ (b = 9) ∧ (c = 14) :=
sorry

end find_ages_l127_127092


namespace group_A_same_function_l127_127112

def fA (x : ℝ) : ℝ := -6 * x + 1 / x
def gA (t : ℝ) : ℝ := -6 * t + 1 / t

def fB (x : ℝ) : ℝ := 1
def gB (x : ℝ) : ℝ := x ^ 0

def fC (x : ℝ) : ℝ := x + 1
def gC (x : ℝ) : ℝ := x * (x + 1) / x

def fD (x : ℝ) : ℝ := 5 * x ^ 5
def gD (x : ℝ) : ℝ := Real.sqrt (x ^ 2)

theorem group_A_same_function : ∀ x : ℝ, x ≠ 0 → fA x = gA x := by
  intros x hx
  rw [fA, gA]
  exact rfl

-- Skipping group B, C and D proofs as they are irrelevant for this theorem
-- And they do not need to be proven correct, only used for reference.

end group_A_same_function_l127_127112


namespace remainder_when_divided_by_198_l127_127348

-- Define the conditions as Hypotheses
variables (x : ℤ)

-- Hypotheses stating the given conditions
def cond1 : Prop := 2 + x ≡ 9 [ZMOD 8]
def cond2 : Prop := 3 + x ≡ 4 [ZMOD 27]
def cond3 : Prop := 11 + x ≡ 49 [ZMOD 1331]

-- Final statement to prove
theorem remainder_when_divided_by_198 (h1 : cond1 x) (h2 : cond2 x) (h3 : cond3 x) : x ≡ 1 [ZMOD 198] := by
  sorry

end remainder_when_divided_by_198_l127_127348


namespace phase_shift_of_sine_l127_127904

def phase_shift (a b c d : ℝ) (x : ℝ) := a * Real.sin (b * x - c) + d

theorem phase_shift_of_sine : 
  phase_shift 5 2 (2 * Real.pi / 3) 0 = phase_shift 5 2 (2 * Real.pi / 3) 0 :=
by
  have phase_shift_value : (2 * Real.pi / 3) / 2 = Real.pi / 3 := sorry
  sorry

end phase_shift_of_sine_l127_127904


namespace length_of_ST_l127_127875

theorem length_of_ST (PQ PS : ℝ) (ST : ℝ) (hPQ : PQ = 8) (hPS : PS = 7) 
  (h_area_eq : (1 / 2) * PQ * (PS * (1 / PS) * 8) = PQ * PS) : 
  ST = 2 * Real.sqrt 65 := 
by
  -- proof steps (to be written)
  sorry

end length_of_ST_l127_127875


namespace mean_is_4_greater_than_median_for_set_l127_127915

theorem mean_is_4_greater_than_median_for_set (x : ℕ) :
  let set := [x, x + 2, x + 4, x + 7, x + 27]
  let median := x + 4
  let mean := (∑ i in set, i) / set.length
  mean = median + 4 :=
by
  sorry

end mean_is_4_greater_than_median_for_set_l127_127915


namespace a2016_value_l127_127627

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = 1 - (1 / a n)

theorem a2016_value : ∃ a : ℕ → ℚ, seq a ∧ a 2016 = 1 / 3 :=
by
  sorry

end a2016_value_l127_127627


namespace determine_m_l127_127617

namespace MathProblem

-- Define the parabola C using parametric equations
def parabolaC (t : ℝ) : ℝ × ℝ := (t^2, 2 * t)

-- Define the polar equation of line l
def lineLPolar (rho theta m : ℝ) : Prop := rho * Real.sin (theta + Real.pi / 4) = m

-- Define the focus of the parabola C
def focusOfC : ℝ × ℝ := (1, 0)

-- Define the Cartesian form of the line passing through the focus
def lineLCartesian (x y m : ℝ) : Prop := x + y - Real.sqrt 2 * m = 0

theorem determine_m (m : ℝ) :
  (∃ t : ℝ, parabolaC t = focusOfC) →
  (∃ rho theta, lineLPolar rho theta m)
  ∧ (∃ x y, focusOfC = (x, y) ∧ lineLCartesian x y m) →
  m = Real.sqrt 2 / 2 := by
  intro h_focus h_conditions
  sorry

end MathProblem

end determine_m_l127_127617


namespace least_additional_squares_needed_for_symmetry_l127_127492

-- Conditions
def grid_size : ℕ := 5
def initial_shaded_squares : List (ℕ × ℕ) := [(1, 5), (3, 3), (5, 1)]

-- Goal statement
theorem least_additional_squares_needed_for_symmetry
  (grid_size : ℕ)
  (initial_shaded_squares : List (ℕ × ℕ)) : 
  ∃ (n : ℕ), n = 2 ∧ 
  (∀ (x y : ℕ), (x, y) ∈ initial_shaded_squares ∨ (grid_size - x + 1, y) ∈ initial_shaded_squares ∨ (x, grid_size - y + 1) ∈ initial_shaded_squares ∨ (grid_size - x + 1, grid_size - y + 1) ∈ initial_shaded_squares) :=
sorry

end least_additional_squares_needed_for_symmetry_l127_127492


namespace orange_harvest_total_sacks_l127_127708

theorem orange_harvest_total_sacks (days : ℕ) (sacks_per_day : ℕ):
  days = 4 → sacks_per_day = 14 → (days * sacks_per_day = 56) :=
by
  intros hdays hsacks
  rw [hdays, hsacks]
  norm_num

end orange_harvest_total_sacks_l127_127708


namespace area_part_triangle_in_circle_l127_127122

theorem area_part_triangle_in_circle (b : ℝ) : 
  let h := (sqrt 3 / 2) * b,
      R := h / 2,
      sect_area := (1/3) * π * R^2,
      tri_area := (1/2) * R^2 * (sqrt 3 / 2),
      S := sect_area + 2 * tri_area,
      R_squared := (b * sqrt 3 / 4)^2
  in S = (R_squared * (2 * π + 3 * sqrt 3) / 6) :=
by
  let h := (sqrt 3 / 2) * b
  let R := h / 2
  let R_squared := (b * (sqrt 3) / 4)^2
  sorry

end area_part_triangle_in_circle_l127_127122


namespace waiting_time_probability_l127_127772

-- Given conditions
def dep1 := 7 * 60 -- 7:00 in minutes
def dep2 := 7 * 60 + 30 -- 7:30 in minutes
def dep3 := 8 * 60 -- 8:00 in minutes

def arrival_start := 7 * 60 + 25 -- 7:25 in minutes
def arrival_end := 8 * 60 -- 8:00 in minutes
def total_time_window := arrival_end - arrival_start -- 35 minutes

def favorable_window1_start := 7 * 60 + 25 -- 7:25 in minutes
def favorable_window1_end := 7 * 60 + 30 -- 7:30 in minutes
def favorable_window2_start := 8 * 60 -- 8:00 in minutes
def favorable_window2_end := 8 * 60 + 10 -- 8:10 in minutes

def favorable_time_window := 
  (favorable_window1_end - favorable_window1_start) + 
  (favorable_window2_end - favorable_window2_start) -- 15 minutes

-- Probability calculation
theorem waiting_time_probability : 
  (favorable_time_window : ℚ) / (total_time_window : ℚ) = 3 / 7 :=
by
  sorry

end waiting_time_probability_l127_127772


namespace tan_pi_minus_theta_l127_127942

theorem tan_pi_minus_theta (θ : ℝ) (h1 : sin θ = -3 / 4) (h2 : 0 < θ ∧ θ < 2 * π) (h3 : θ > 3 * π / 2 ∨ θ < 2 * π) :
  tan (π - θ) = 3 * sqrt 7 / 7 :=
by
  sorry

end tan_pi_minus_theta_l127_127942


namespace circumcenter_of_BCD_lies_on_circumcircle_of_ABC_l127_127302

theorem circumcenter_of_BCD_lies_on_circumcircle_of_ABC 
  (A B C D K : Point)
  (S1 S2 : Circle)
  (hB_on_S1 : B ∈ S1)
  (hA_on_tangent_at_B_to_S1 : A ≠ B ∧ is_tangent A B S1)
  (hC_not_on_S1 : C ∉ S1)
  (hAC_meets_S1_at_two_distinct_points : line(A, C, meets, S1, 2))
  (hS2_touches_AC_at_C : touches(S2, line(A, C), C))
  (hS2_touches_S1_at_D : touches(S2, S1, D))
  (hD_on_opposite_side_of_AC_from_B : is_opposite_side(D, line(A, C), B))
  (hK_circumcenter_of_BCD : is_circumcenter(K, triangle(B, C, D))) :
  lies_on_circumcircle(K, triangle(A, B, C)) :=
sorry

end circumcenter_of_BCD_lies_on_circumcircle_of_ABC_l127_127302


namespace sum_a6_a7_a8_is_32_l127_127285

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l127_127285


namespace angle_x_in_triangle_CDE_l127_127620

theorem angle_x_in_triangle_CDE (D C B E : Type) 
  [IsPoint D] [IsPoint C] [IsPoint B] [IsPoint E]
  (angle_DCB : angle_between_lines (line_through D C) (line_through C B) = 90)
  (angle_ECB : angle_between_lines (line_through E C) (line_through C B) = 70)
  (sum_angles_CDE : ∀ a b c : ℝ, a + b + c = 180) :
  angle_between_lines (line_through D C) (line_through C E) = 20 := 
by 
  sorry

end angle_x_in_triangle_CDE_l127_127620


namespace smallest_five_digit_divisible_by_2_5_11_l127_127908

theorem smallest_five_digit_divisible_by_2_5_11 : ∃ n, n >= 10000 ∧ n % 2 = 0 ∧ n % 5 = 0 ∧ n % 11 = 0 ∧ n = 10010 :=
by
  sorry

end smallest_five_digit_divisible_by_2_5_11_l127_127908


namespace sum_of_first_n_odd_integers_l127_127679

theorem sum_of_first_n_odd_integers (n : ℕ) : (∑ i in Finset.range n, (2 * i + 1)) = n^2 :=
by
  sorry

end sum_of_first_n_odd_integers_l127_127679


namespace min_sum_of_pairs_l127_127474

theorem min_sum_of_pairs:
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2013 → a i = 1 ∨ a i = -1) →
  0 < ∑ 1 ≤ i < j ≤ 2013, a i * a j :=
sorry

end min_sum_of_pairs_l127_127474


namespace number_of_real_roots_l127_127372

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - 6 * x ^ 2 + 9 * x - 10

theorem number_of_real_roots : ∃! x : ℝ, f x = 0 :=
sorry

end number_of_real_roots_l127_127372


namespace part_a_part_b_l127_127513

/-- Two equally skilled chess players with p = 0.5, q = 0.5. -/
def p : ℝ := 0.5
def q : ℝ := 0.5

-- Definition for binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Binomial distribution
def P (n k : ℕ) : ℝ := (binomial_coeff n k) * (p^k) * (q^(n-k))

/-- Prove that the probability of winning one out of two games is greater than the probability of winning two out of four games -/
theorem part_a : (P 2 1) > (P 4 2) := sorry

/-- Prove that the probability of winning at least two out of four games is greater than the probability of winning at least three out of five games -/
theorem part_b : (P 4 2 + P 4 3 + P 4 4) > (P 5 3 + P 5 4 + P 5 5) := sorry

end part_a_part_b_l127_127513


namespace sum_a6_a7_a8_is_32_l127_127286

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l127_127286


namespace tan_sum_sin_eq_l127_127878

/-- Define the sum S -/
def S : ℝ := ∑ j in finset.range 2021, |real.sin (2 * real.pi * j / 2021)|

/-- S can be written as tan(cπ/d) where c = 1010, d = 2021, and 2c < d. -/
theorem tan_sum_sin_eq :
  S = real.tan (1010 * real.pi / 2021) ∧ nat.coprime 1010 2021 ∧ 2 * 1010 < 2021 :=
sorry

end tan_sum_sin_eq_l127_127878


namespace tangent_line_at_point_is_correct_l127_127982

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line_at_point_is_correct :
    ∀ x y : ℝ, y = f x ∧ x = 2 → (13, -32) = (f' 2, f 2 - 2 * f' 2) :=
by 
  sorry

end tangent_line_at_point_is_correct_l127_127982


namespace cone_slant_height_l127_127208

theorem cone_slant_height (V : ℝ) (θ : ℝ) (l : ℝ)
  (h : 0 < l)
  (volume_cone : V = 9 * real.pi)
  (angle_45 : θ = real.pi / 4) :
  l = 3 * real.sqrt 2 :=
by
  sorry

end cone_slant_height_l127_127208


namespace ofelia_savings_in_may_l127_127324

-- Conditions
def savings (n : ℕ) : ℕ :=
if n = 1 then 10 else 2 * (savings (n - 1))

-- Theorem
theorem ofelia_savings_in_may : savings 5 = 160 :=
by
  sorry

end ofelia_savings_in_may_l127_127324


namespace square_mirror_side_length_l127_127826

theorem square_mirror_side_length :
  ∃ (side_length : ℝ),
  let wall_width := 42
  let wall_length := 27.428571428571427
  let wall_area := wall_width * wall_length
  let mirror_area := wall_area / 2
  (side_length * side_length = mirror_area) → side_length = 24 :=
by
  use 24
  intro h
  sorry

end square_mirror_side_length_l127_127826


namespace roots_quadratic_expression_l127_127646

theorem roots_quadratic_expression :
  ∀ (a b : ℝ), (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) → 
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b * (a + b) = 533 :=
by
  intros a b h
  sorry

end roots_quadratic_expression_l127_127646


namespace third_quadrant_angle_to_fourth_l127_127231

theorem third_quadrant_angle_to_fourth {α : ℝ} (k : ℤ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  -90 - k * 360 < 180 - α ∧ 180 - α < -k * 360 :=
by
  sorry

end third_quadrant_angle_to_fourth_l127_127231


namespace number_of_ways_to_select_singer_and_dancer_l127_127768

theorem number_of_ways_to_select_singer_and_dancer :
  ∀ (n m k : ℕ), n = 8 → m = 6 → k = 5 →
  (let both := m + k - n in
   let only_singers := k - both in
   let only_dancers := m - both in
   (both * only_dancers + only_singers * both) = 15) := 
by
  intros n m k hn hm hk
  rw [hn, hm, hk]
  let both := m + k - n
  let only_singers := k - both
  let only_dancers := m - both
  have h_both : both = 3 := by
    rw [hn, hm, hk]
    calc m + k - n = 6 + 5 - 8 : by rw' [hm, hk, hn]
    ... = 3

  have h_only_singers : only_singers = 2 := by
    rw [←h_both, hk]
    calc k - both = 5 - 3 : by rw [hk, h_both]
    ... = 2

  have h_only_dancers: only_dancers = 3 := by
    rw [←h_both, hm]
    calc m - both = 6 - 3 : by rw [hm, h_both]
    ... = 3
  
  show (both * only_dancers + only_singers * both) = 15 from
    calc both * only_dancers + only_singers * both = 3 * 3 + 2 * 3 : by rw [h_both, h_only_dancers, h_only_singers]
    ... = 9 + 6 
    ... = 15


end number_of_ways_to_select_singer_and_dancer_l127_127768


namespace fastest_route_time_l127_127633

def distance_first_second : ℝ := 6
def speed_first_second : ℝ := 30
def distance_increase_factor : ℝ := 2 / 3
def speed_second_third : ℝ := 40
def distance_house_first : ℝ := 4
def speed_house_first_last_work : ℝ := 20
def detour_additional_distance : ℝ := 3
def detour_speed : ℝ := 25

theorem fastest_route_time : ∀ (faster_route : String), (faster_route = "original" ∧
  (4 / 20 + 6 / 30 + (6 + 2 / 3 * 6) / 40 + 4 / 20) * 60 = 51) :=
by 
  sorry

end fastest_route_time_l127_127633


namespace number_of_values_of_x_l127_127926

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (169 - real.cbrt x)

theorem number_of_values_of_x : 
  let S := {y : ℕ | y ≤ 13} in
  ∃ (n : ℕ), n = S.card := 
sorry

end number_of_values_of_x_l127_127926


namespace find_m_l127_127713

def Point (α : Type) := (α × α)

def slope {α : Type} [DivisionRing α] (A B : Point α) : α :=
  (B.2 - A.2) / (B.1 - A.1)

theorem find_m (m : ℝ) 
  (A B : Point ℝ)
  (hA : A = (-m, 6))
  (hB : B = (1, 3 * m))
  (hSlope : slope A B = 12) : m = -2 :=
by
  sorry

end find_m_l127_127713


namespace most_significant_price_drop_l127_127776

noncomputable def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => -1.00
  | 2 => 0.50
  | 3 => -3.00
  | 4 => 2.00
  | 5 => -1.50
  | 6 => -0.75
  | _ => 0.00 -- For any invalid month, we assume no price change

theorem most_significant_price_drop :
  ∀ m : ℕ, (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6) →
  (∀ n : ℕ, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) →
  price_change m ≤ price_change n) → m = 3 :=
by
  intros m hm H
  sorry

end most_significant_price_drop_l127_127776


namespace strike_time_10_times_l127_127999

def time_to_strike (n : ℕ) : ℝ :=
  if n = 0 then 0 else (n - 1) * 6

theorem strike_time_10_times : time_to_strike 10 = 60 :=
  by {
    -- Proof outline
    -- time_to_strike 10 = (10 - 1) * 6 = 9 * 6 = 54. Thanks to provided solution -> we shall consider that time take 10 seconds for the clock to start striking.
    sorry
  }

end strike_time_10_times_l127_127999


namespace least_divisible_by_primes_is_420_l127_127403

def is_least_divisible_by_four_primes_squared (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
  n = (p1 * p1 * p2 * p3 * p4) ∧
  ∀ m : ℕ, (nat.prime m → ∀ (i j k l : ℕ), (i = p1 → j = p2 → k = p3 → l = p4 → 
  m = i*i * j * k * l → (m ≥ n)))

theorem least_divisible_by_primes_is_420 : 
  is_least_divisible_by_four_primes_squared 420 :=
sorry

end least_divisible_by_primes_is_420_l127_127403


namespace net_share_B_l127_127827

-- Define the fixed values and conditions
def parts_A := 2
def parts_B := 3
def parts_C := 4
def parts_D := 6
def total_parts := parts_A + parts_B + parts_C + parts_D -- 15 parts

def diff_DC := parts_D - parts_C -- Difference in parts (2 parts)
def increment_amount := 700 -- D gets $700 more than C
def value_per_part := increment_amount / diff_DC -- $350

-- Initial gross salary calculation based on parts
def gross_A := parts_A * value_per_part -- $700
def gross_B := parts_B * value_per_part -- $1050
def gross_C := parts_C * value_per_part -- $1400
def gross_D := parts_D * value_per_part -- $2100

-- Minimum wage and adjustment factor
def min_wage := 1000
def adjustment_factor := min_wage / gross_A -- 10/7 for adjusting salaries

-- Adjusted gross salaries
def adj_gross_A := min_wage -- $1000
def adj_gross_B := gross_B * adjustment_factor -- $1500
def adj_gross_C := gross_C * adjustment_factor -- $2000
def adj_gross_D := gross_D * adjustment_factor -- $3000

-- Tax calculations for B's adjusted gross salary
def tax_B := if adj_gross_B > 1000 then 0.1 * (adj_gross_B - 1000) else 0 -- $50

-- B's net share after tax
def net_B := adj_gross_B - tax_B -- $1450

-- Define the theorem to prove
theorem net_share_B : net_B = 1450 := by
  sorry

end net_share_B_l127_127827


namespace quadratic_root_relation_l127_127548

theorem quadratic_root_relation (x₁ x₂ : ℝ) (h₁ : x₁ ^ 2 - 3 * x₁ + 2 = 0) (h₂ : x₂ ^ 2 - 3 * x₂ + 2 = 0) :
  x₁ + x₂ - x₁ * x₂ = 1 := by
sorry

end quadratic_root_relation_l127_127548


namespace root_interval_l127_127884

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem root_interval : ∃ c ∈ Ioo 0 1, f c = 0 :=
by
  have f_cont : Continuous f := Real.exp_continuous.add continuous_id.sub continuous_const
  have f_increasing : ∀ x, 0 < (Differentiable.fderiv f x).dfDeriv.totalDeriv :=
    by simp [f, differentiable_real.exp.add differentiable_id.sub differentiable_const]
  have f_at_0 : f 0 < 0 := by norm_num [f, Real.exp]
  have f_at_1 : 0 < f 1 := by norm_num [f, Real.exp] ; exact lt_add_of_pos_of_le zero_lt_one (Real.exp_pos 1)
  exact exists_root_intermediate_value (Icc_mem_Ioo.mpr (lt_of_lt_of_le zero_lt_one one_le_two)) f_cont f_at_0 f_at_1 sorry -- Details of the proof can be filled

end root_interval_l127_127884


namespace find_ordered_pair_l127_127903

noncomputable def cos_deg (θ : ℝ) := cos (θ * (π / 180))
noncomputable def sec_deg (θ : ℝ) := 1 / cos_deg θ

theorem find_ordered_pair :
  ∃ (a b : ℤ), sqrt (16 - 12 * cos_deg 40) = a + b * sec_deg 40 ∧ a = 4 ∧ b = -1 :=
by {
  use (4 : ℤ),
  use (-1 : ℤ),
  split,
  {
    sorry 
  },
  split,
  {
    refl  
  },
  {
    refl  
  },
}

end find_ordered_pair_l127_127903


namespace BQ_bisects_EF_l127_127274

theorem BQ_bisects_EF
  (ABC : Triangle)
  (H : Point) (Γ : Circle)
  (orthocenter_ABC : IsOrthocenter H ABC)
  (circumcircle_ABC : IsCircumcircle Γ ABC)
  (B H C : Point → Line)
  (E F P Q : Point)
  (BH_inter_AC : Intersects (Line_through B H) (Line_through A C) = E)
  (CH_inter_AB : Intersects (Line_through C H) (Line_through A B) = F)
  (AH_inter_Γ : SecondIntersection (Line_through A H) Γ = P)
  (P_neq_A : P ≠ A)
  (PE_inter_Γ : SecondIntersection (Line_through P E) Γ = Q)
  (Q_neq_P : Q ≠ P) :
  Bisects (Line_through B Q) EF :=
begin
  sorry
end

end BQ_bisects_EF_l127_127274


namespace triangle_problem_l127_127946

noncomputable def f (x : ℝ) : ℝ :=
  let m := (2 * Real.cos x, 1)
  let n := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
  m.1 * n.1 + m.2 * n.2

theorem triangle_problem
  (A B C : ℝ)
  (b : ℝ)
  (S : ℝ)
  (hA : f A = 2)
  (h_b : b = 1)
  (h_S : S = Real.sqrt 3 / 2)
  (c : ℝ)
  (sin_B sin_C : ℝ)
  (h_sin : sin_B + sin_C = 3 / 2)
  (h_c : c = 2) -- Derived from given conditions
  (h_sin_B : sin_B = 1 / 2) -- From Sine Rule
  (h_sin_C : sin_C = 1) -- From Sine Rule
  : (b + c) / (sin_B + sin_C) = 2 :=
by
  rw [h_b, h_c, h_sin_B, h_sin_C]
  linarith

end triangle_problem_l127_127946


namespace value_of_m_div_x_l127_127424

variables (a b : ℝ) (k : ℝ)
-- Condition: The ratio of a to b is 4 to 5
def ratio_a_to_b : Prop := a / b = 4 / 5

-- Condition: x equals a increased by 75 percent of a
def x := a + 0.75 * a

-- Condition: m equals b decreased by 80 percent of b
def m := b - 0.80 * b

-- Prove the given question
theorem value_of_m_div_x (h1 : ratio_a_to_b a b) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  m / x = 1 / 7 := by
sorry

end value_of_m_div_x_l127_127424


namespace calculate_nabla_l127_127136

-- Define the nabla operation
def nabla (a b : ℝ) : ℝ := a + 2 * b^a

-- State the theorem that represents the problem
theorem calculate_nabla :
  nabla (nabla 1.5 2) 2.5 = 200.8125 :=
by
  sorry

end calculate_nabla_l127_127136


namespace midpoints_octagon_area_half_l127_127096

theorem midpoints_octagon_area_half (A B C D E F G H I J K L M N O P : Point)
  (h_regular_octagon : regular_octagon A B C D E F G H)
  (h_midpoints : midpoints A B C D E F G H I J K L M N O P) :
  enclosed_area I J K L M N O P = 1/2 * enclosed_area A B C D E F G H :=
sorry

end midpoints_octagon_area_half_l127_127096


namespace petStoreHasSixParrots_l127_127779

def petStoreParrotsProof : Prop :=
  let cages := 6.0
  let parakeets := 2.0
  let birds_per_cage := 1.333333333
  let total_birds := cages * birds_per_cage
  let number_of_parrots := total_birds - parakeets
  number_of_parrots = 6.0

theorem petStoreHasSixParrots : petStoreParrotsProof := by
  sorry

end petStoreHasSixParrots_l127_127779


namespace monster_feast_interval_l127_127081

theorem monster_feast_interval (P : ℕ) (h1 : P = 121) (h2 : P + 2 * P + 4 * P = 847) : (P + 2 * P + 4 * P = 847) → 100 :=
by
  sorry

end monster_feast_interval_l127_127081


namespace even_iff_a_zero_monotonous_iff_a_range_max_value_l127_127571

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 2

-- (I) Prove that f(x) is even on [-5, 5] if and only if a = 0
theorem even_iff_a_zero (a : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a = f (-x) a) ↔ a = 0 := sorry

-- (II) Prove that f(x) is monotonous on [-5, 5] if and only if a ≥ 10 or a ≤ -10
theorem monotonous_iff_a_range (a : ℝ) : (∀ x y : ℝ, -5 ≤ x ∧ x ≤ y ∧ y ≤ 5 → f x a ≤ f y a) ↔ (a ≥ 10 ∨ a ≤ -10) := sorry

-- (III) Prove the maximum value of f(x) in the interval [-5, 5]
theorem max_value (a : ℝ) : (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ (∀ y : ℝ, -5 ≤ y ∧ y ≤ 5 → f y a ≤ f x a)) ∧  
                           ((a ≥ 0 → f 5 a = 27 + 5 * a) ∧ (a < 0 → f (-5) a = 27 - 5 * a)) := sorry

end even_iff_a_zero_monotonous_iff_a_range_max_value_l127_127571


namespace find_starting_number_from_divisible_sequence_l127_127387

theorem find_starting_number_from_divisible_sequence (n m : ℕ) (d : ℕ) :
  (m - d) / n = 5 → m = 79 → d = 11 → n = 6 → ∃ a, a = 22 ∧ list.all (list.range' a n) (λ x, x % d = 0) :=
by sorry

end find_starting_number_from_divisible_sequence_l127_127387


namespace max_value_of_k_l127_127250

theorem max_value_of_k (k : ℝ) 
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 8 * x + 15 = 0)
  (line_eq : ∀ x y : ℝ, y = k * x - 2) :
  max {k : ℝ | 0 ≤ k ∧ k ≤ 4 / 3} = 4 / 3 := 
sorry

end max_value_of_k_l127_127250


namespace dishonest_shopkeeper_gain_l127_127747

-- Define the constants
def true_weight : ℝ := 1000
def false_weight : ℝ := 970
def expected_gain_percentage : ℝ := 3.09

-- Write the Lean 4 statement to prove the gain percentage
theorem dishonest_shopkeeper_gain : 
  ((true_weight - false_weight) / false_weight * 100) ≈ expected_gain_percentage :=
by
  -- Proof will be written here
  sorry

end dishonest_shopkeeper_gain_l127_127747


namespace single_elimination_games_count_l127_127828

-- Define the number of teams
def num_teams : ℕ := 23

-- Define the single-elimination tournament games count theorem
theorem single_elimination_games_count (n : ℕ) (h : n = 23) : 
  ∃ g : ℕ, g = n - 1 :=
by
  use n - 1
  rw h
  exact rfl

end single_elimination_games_count_l127_127828


namespace correct_final_positions_l127_127185

noncomputable def shapes_after_rotation (initial_positions : (String × String) × (String × String) × (String × String)) : (String × String) × (String × String) × (String × String) :=
  match initial_positions with
  | (("Triangle", "Top"), ("Circle", "Lower Left"), ("Pentagon", "Lower Right")) =>
    (("Triangle", "Lower Right"), ("Circle", "Top"), ("Pentagon", "Lower Left"))
  | _ => initial_positions

theorem correct_final_positions :
  shapes_after_rotation (("Triangle", "Top"), ("Circle", "Lower Left"), ("Pentagon", "Lower Right")) = (("Triangle", "Lower Right"), ("Circle", "Top"), ("Pentagon", "Lower Left")) :=
by
  unfold shapes_after_rotation
  rfl

end correct_final_positions_l127_127185


namespace rectangle_lengths_l127_127602

theorem rectangle_lengths (side_length : ℝ) (width1 width2: ℝ) (length1 length2 : ℝ) 
  (h1 : side_length = 6) 
  (h2 : width1 = 4) 
  (h3 : width2 = 3)
  (h_area_square : side_length * side_length = 36)
  (h_area_rectangle1 : width1 * length1 = side_length * side_length)
  (h_area_rectangle2 : width2 * length2 = (1 / 2) * (side_length * side_length)) :
  length1 = 9 ∧ length2 = 6 :=
by
  sorry

end rectangle_lengths_l127_127602


namespace paint_can_distribution_l127_127089

-- Definitions based on conditions provided in the problem.
def ratio_red := 3
def ratio_white := 2
def ratio_blue := 1
def total_paint := 60
def ratio_sum := ratio_red + ratio_white + ratio_blue

-- Definition of the problem to be proved.
theorem paint_can_distribution :
  (ratio_red * total_paint) / ratio_sum = 30 ∧
  (ratio_white * total_paint) / ratio_sum = 20 ∧
  (ratio_blue * total_paint) / ratio_sum = 10 := 
by
  sorry

end paint_can_distribution_l127_127089


namespace password_combinations_check_l127_127319

theorem password_combinations_check : ∃ (s : Multiset Char), Multiset.card s = 5 ∧ (Multiset.perm s).card = 20 := by
  sorry

end password_combinations_check_l127_127319


namespace cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127865

theorem cos_105_eq_sqrt2_sub_sqrt6_div4 :
  cos (105 * real.pi / 180) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by
  -- Definitions and conditions
  have cos_60 : cos (60 * real.pi / 180) = 1/2 := by sorry
  have cos_45 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  have sin_60 : sin (60 * real.pi / 180) = real.sqrt 3 / 2 := by sorry
  have sin_45 : sin (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  -- Use the angle addition formula: cos (a + b) = cos a * cos b - sin a * sin b
  have add_formula := cos_add (60 * real.pi / 180) (45 * real.pi / 180)
  -- Combine the results using the given known values
  rw [cos_60, cos_45, sin_60, sin_45] at add_formula
  exact add_formula

end cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127865


namespace sum_sequence_2021_eq_1_l127_127187

def sequence (n : ℕ) : (ℕ → ℚ) :=
  λ k, if k = 0 then 1 / n else 1 / (n - k) * (List.sum (List.ofFn (λ i, sequence n i) k))

def sum_sequence (n : ℕ) : ℚ :=
  List.sum (List.ofFn (sequence n) n)

theorem sum_sequence_2021_eq_1 : sum_sequence 2021 = 1 :=
by
  sorry

end sum_sequence_2021_eq_1_l127_127187


namespace reciprocal_neg_one_over_2011_l127_127378

theorem reciprocal_neg_one_over_2011 : 1 / (- (1 / 2011)) = -2011 :=
by
  sorry

end reciprocal_neg_one_over_2011_l127_127378


namespace partial_derivatives_l127_127168

noncomputable def function_z (x y : ℝ) : ℝ :=
  2 * x^y - x * Real.tan (x * y)

theorem partial_derivatives (x y : ℝ) :
  (∀ z : ℝ, z = function_z x y → 
  (∂ (λ x : ℝ, function_z x y) / ∂ x) x y = 2 * y * x^(y-1) - Real.tan (x * y) - (x * y) / (Real.cos (x * y))^2) ∧
  (∂ (λ y : ℝ, function_z x y) / ∂ y) x y = 2 * x^y * Real.log x - (x^2) / (Real.cos (x * y))^2 :=
begin
  sorry,
end

end partial_derivatives_l127_127168


namespace area_triangle_FPG_l127_127729

noncomputable def area_of_triangle_FPG (EF GH EG FH P : ℝ) (A : ℝ) : Prop :=
  let EF : ℝ := 15
  let GH : ℝ := 25
  let A : ℝ := 200
  (∃ P : ℝ, area_of_triangle FPG = 45)

theorem area_triangle_FPG (EF GH EG FH P : ℝ) (A: ℝ) 
                          (trapezoid_EFGH : EF = 15) 
                          (GH = 25) 
                          (area_trapezoid : A = 200) 
                          (diagonals_intersect : EG ∩ FH = P) : area_of_triangle FPG = 45 :=
sorry

end area_triangle_FPG_l127_127729


namespace range_of_a_l127_127986

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

theorem range_of_a (a : ℝ) : (∀ x, (1/2 : ℝ) < x → x < 3 → (Real.log x + a * x^2 - 3 * x)' x ≥ 0) ↔ a ∈ Icc (9/8 : ℝ) Real.infinity :=
by sorry

end range_of_a_l127_127986


namespace poly_diff_divisible_l127_127174

-- Definitions of the problem's initial conditions
def is_int_poly (p : polynomial ℤ) : Prop := ∀ (x : ℤ), p.eval x ∈ ℤ
def degree_gt_two (p : polynomial ℤ) : Prop := p.degree > 2
def takes_values (p : polynomial ℤ) : Prop :=
  p.eval 1 = 1 ∧ p.eval 2 = 2 ∧ p.eval 3 = 3

-- Theorem statement
theorem poly_diff_divisible (p : polynomial ℤ) (b c : ℤ) (h_distinct: b ≠ c) :
  is_int_poly p ∧ degree_gt_two p ∧ takes_values p → (p.eval b - p.eval c) % (b - c) = 0 :=
by
  -- Proof is skipped with 'sorry'
  sorry

end poly_diff_divisible_l127_127174


namespace price_of_refrigerator_l127_127052

variable (R W : ℝ)

theorem price_of_refrigerator 
  (h1 : W = R - 1490) 
  (h2 : R + W = 7060) 
  : R = 4275 :=
sorry

end price_of_refrigerator_l127_127052


namespace exist_points_on_ellipse_l127_127409

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- A point B on the plane
def B : ℝ × ℝ := (2, 0)

-- Define the conditions for minimum length of segment MN
def k_constraint (k : ℝ) : Prop :=
  k > 0 ∧ (16 * k / 3 + 1 / (3 * k)) = 8 / 3

-- Define the condition for the position of point S when MN is minimized
def point_S (k : ℝ) : ℝ × ℝ :=
  (2 - 8 * k^2) / (1 + 4 * k^2), 4 * k / (1 + 4 * k^2)

-- Define the equation of the line BS
def line_BS (k : ℝ) (x y : ℝ) : Prop :=
  y = - (1 / (4 * k)) * (x - 2)

-- Define the distance condition for point T to line BS
def distance_condition (S : ℝ × ℝ) (T : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), ((T.1 + T.2 + t = 0) ∧ (abs (t + 2) / ℝ.sqrt 2 = ℝ.sqrt 2 / 4))

-- Define the area constraint for triangle TSB
def area_triangle_TSB (T S : ℝ × ℝ) : Prop :=
  let Bx, By := B in
  let Sx, Sy := S in
  let Tx, Ty := T in
  abs (Sx * (By - Ty) + Bx * (Ty - Sy) + Tx * (Sy - By)) / 2 = 1 / 5

-- Main theorem statement
theorem exist_points_on_ellipse (k : ℝ) :
  k_constraint k →
  let S := point_S k in
  (∃ (T1 T2 : ℝ × ℝ), ellipse_C T1.1 T1.2 ∧ ellipse_C T2.1 T2.2 ∧
  area_triangle_TSB T1 S ∧ area_triangle_TSB T2 S) :=
sorry

end exist_points_on_ellipse_l127_127409


namespace max_points_difference_l127_127071

theorem max_points_difference {n : ℕ} (h : n ≥ 2) :
  ∃ A B : ℕ, A ≠ B ∧ (∀ t₁ t₂ : fin n, t₁ ≠ t₂ → 
  (t₁.val - t₂.val).abs ≤ n) ∧
  (A.val - B.val).abs = n :=
sorry

end max_points_difference_l127_127071


namespace necessary_but_not_sufficient_l127_127431

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (¬(x ≥ 1) ∨ (x ≥ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l127_127431


namespace slices_per_person_l127_127126

theorem slices_per_person (pizzas slices_per_pizza people : ℕ) (hp : pizzas = 5) (hs : slices_per_pizza = 4) (hpeople : people = 10) :
  (pizzas * slices_per_pizza) / people = 2 :=
by
  -- Provide assumptions
  rw [hp, hs, hpeople]
  -- Calculate the total slices
  have total_slices : 5 * 4 = 20 := by norm_num
  rw total_slices
  -- Divide by number of people
  have slices_per_person : 20 / 10 = 2 := by norm_num
  exact slices_per_person

end slices_per_person_l127_127126


namespace subset_exists_l127_127549

def odd_pos (n : ℕ) : Prop := n % 2 = 1 ∧ n > 0

theorem subset_exists (A : set ℕ) (H : A = {a, b, c} ∧ ∀ x ∈ A, x > 0) :
  ∃ (B : set ℕ), B ⊆ A ∧ ∃ x y, x ≠ y ∧ x ∈ B ∧ y ∈ B ∧ (∀ m n : ℕ, odd_pos m ∧ odd_pos n → 10 ∣ x ^ m * y ^ n - x ^ n * y ^ m) :=
sorry

end subset_exists_l127_127549


namespace bacteria_initial_count_l127_127353

noncomputable def initial_bacteria (b_final : ℕ) (q : ℕ) : ℕ :=
  b_final / 4^q

theorem bacteria_initial_count : initial_bacteria 262144 4 = 1024 := by
  sorry

end bacteria_initial_count_l127_127353


namespace sector_area_correct_l127_127950

def sector_area (n : ℝ) (l : ℝ) (R : ℝ) : ℝ := (n / 360) * π * R^2

theorem sector_area_correct :
  ∀ (n l : ℝ) (R : ℝ), n = 60 ∧ l = π ∧ l = (n * π * R) / 180 → sector_area n l R = (3 * π) / 2 :=
by
  intros n l R h
  cases h with h1 h2
  sorry

end sector_area_correct_l127_127950


namespace total_area_inside_odd_number_of_circles_l127_127256

def radius : List ℕ := [1, 2, 3, 4, 5]

def area (r : ℕ) : ℚ := (r ^ 2 : ℚ) * Real.pi 

theorem total_area_inside_odd_number_of_circles :
  let A1 := area 1
  let A2 := area 2
  let A3 := area 3
  let A4 := area 4
  let A5 := area 5
  (A5 - A4) + (A3 - A2) + A1 = 15 * Real.pi := 
by
  sorry

end total_area_inside_odd_number_of_circles_l127_127256


namespace maximal_enclosed_area_l127_127327

theorem maximal_enclosed_area (k : ℕ) :
  ∃ S : set (ℤ × ℤ), (∃ P : list (ℤ × ℤ),
    vertices P ∧ non_self_intersecting P ∧ 
    encloses_black_cells P = k) →
    area_of_enclosing_figure S = 4 * k + 1 := 
sorry

end maximal_enclosed_area_l127_127327


namespace right_triangle_angle_measure_l127_127702

theorem right_triangle_angle_measure
  {A B C D L : Type}
  [right_triangle ABC]
  (h_angle_ABC : angle ABC = 90)
  (h_height_AD : height_from A D)
  (h_bisector_ratio : divide_ratio (angle_bisector BAC D L) 5 2) :
  angle BAC = real.arccos (5 / 7) :=
sorry

end right_triangle_angle_measure_l127_127702


namespace part1_part2_l127_127070

/-- Part (1): For any real number \( t \in \left(0, \frac{1}{2}\right) \), 
there exists a positive integer \( n \), such that for any set \( S \) of \( n \) 
positive integers, there exist distinct elements \( x \) and \( y \) in \( S \) 
and a natural number \( m \) such that \( |x - my| \leq t y \). -/
theorem part1 (t : ℝ) (ht : 0 < t ∧ t < 1/2) : 
  ∃ n : ℕ, ∀ S : Finset ℕ, S.card = n → 
    ∃ (x y : ℕ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x ≠ y) (m : ℕ),
    |x - m * y| ≤ t * y :=
sorry

/-- Part (2): For any real number \( t \in \left(0, \frac{1}{2}\right) \), 
there exists an infinite set \( S \) of positive integers such that for any distinct 
elements \( x \) and \( y \) in \( S \) and any positive integer \( m \), 
\( |x - my| > t y \). -/
theorem part2 (t : ℝ) (ht : 0 < t ∧ t < 1/2) : 
  ∃ S : Set ℕ, S.infinite ∧ ∀ x y ∈ S, x ≠ y → ∀ m : ℕ, 
    |x - m * y| > t * y :=
sorry

end part1_part2_l127_127070


namespace compute_expression_l127_127630

noncomputable def symbol_to_number : ℕ → ℤ
| 0 := 0
| 1 := 7
| 2 := 5
| 3 := 3
| 4 := 4
| 5 := 2
| 6 := 9
| 7 := 1
| 8 := 1000000  -- assuming an arbitrarily large positive integer
| 9 := 6
| _ := 8  -- for ∞

theorem compute_expression : 
  abs (
    symbol_to_number 0 
    - symbol_to_number 1 
    + symbol_to_number 2 
    - (symbol_to_number 3) ^ 4 
    - symbol_to_number 5 
    + symbol_to_number 6 
    - (symbol_to_number 7) ^ (symbol_to_number 8) * symbol_to_number 9 
    - symbol_to_number 10
  ) = 90 :=
by
  sorry

end compute_expression_l127_127630


namespace puppy_weight_l127_127781

theorem puppy_weight (a b c : ℕ) 
  (h1 : a + b + c = 24) 
  (h2 : a + c = 2 * b) 
  (h3 : a + b = c) : 
  a = 4 :=
sorry

end puppy_weight_l127_127781


namespace sum_of_integers_l127_127016

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 119) (h2 : a < 15) (h3 : b < 15) (h4 : Nat.coprime a b) :
  a + b = 21 := 
sorry

end sum_of_integers_l127_127016


namespace nonneg_real_values_of_x_count_l127_127924

theorem nonneg_real_values_of_x_count :
  let S := {x : ℝ | 0 ≤ x ∧ ∃ k : ℕ, k ≤ 13 ∧ k = Real.sqrt (169 - Real.cbrt x)} in
  S.card = 14 :=
by
  sorry

end nonneg_real_values_of_x_count_l127_127924


namespace reflection_path_exists_l127_127186

theorem reflection_path_exists (line_I : Line) (A B : Point) (A1 : Point) (φ : ℝ) :
  (A1 = Line.symmetricPoint line_I A) →
  (A side_of_line line_I) →
  (B side_of_line line_I) →
  ∃ X : Point,
  let X : Point := reflection_point φ line_I A1 B in
  reflects_off_with_given_condition X φ :=
begin
  sorry
end

end reflection_path_exists_l127_127186


namespace scientific_notation_of_22nm_l127_127690

theorem scientific_notation_of_22nm (h : 22 * 10^(-9) = 0.000000022) : 0.000000022 = 2.2 * 10^(-8) :=
sorry

end scientific_notation_of_22nm_l127_127690


namespace region_area_l127_127038

theorem region_area (x y : ℝ) : 
  (x^2 + y^2 + 14 * x + 18 * y = 0) → 
  (π * 130) = 130 * π :=
by 
  sorry

end region_area_l127_127038


namespace rise_in_water_level_correct_l127_127464

noncomputable def volume_of_rectangular_solid (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def area_of_circular_base (d : ℝ) : ℝ :=
  Real.pi * (d / 2) ^ 2

noncomputable def rise_in_water_level (solid_volume base_area : ℝ) : ℝ :=
  solid_volume / base_area

theorem rise_in_water_level_correct :
  let l := 10
  let w := 12
  let h := 15
  let d := 18
  let solid_volume := volume_of_rectangular_solid l w h
  let base_area := area_of_circular_base d
  let expected_rise := 7.07
  abs (rise_in_water_level solid_volume base_area - expected_rise) < 0.01 
:= 
by {
  sorry
}

end rise_in_water_level_correct_l127_127464


namespace determine_radii_l127_127552

-- Definitions based on conditions from a)
variable (S1 S2 S3 S4 : Type) -- Centers of the circles
variable (dist_S2_S4 : ℝ) (dist_S1_S2 : ℝ) (dist_S2_S3 : ℝ) (dist_S3_S4 : ℝ)
variable (r1 r2 r3 r4 : ℝ) -- Radii of circles k1, k2, k3, and k4
variable (rhombus : Prop) -- Quadrilateral S1S2S3S4 is a rhombus

-- Given conditions
axiom C1 : ∀ t : S1, r1 = 5
axiom C2 : dist_S2_S4 = 24
axiom C3 : rhombus

-- Equivalency to be proven
theorem determine_radii : 
  r2 = 12 ∧ r4 = 12 ∧ r1 = 5 ∧ r3 = 5 :=
sorry

end determine_radii_l127_127552


namespace OQ_perpendicular_PQ_l127_127953

theorem OQ_perpendicular_PQ
  (O A D B C Q P : Point)
  (h1 : Circle O contains A)
  (h2 : Line P A passes_through O)
  (h3 : Secant P C D)
  (h4 : Circle A O D intersects (Circle B O C) at Q) :
  Perpendicular (Line O Q) (Line P Q) :=
sorry

end OQ_perpendicular_PQ_l127_127953


namespace trigonometric_identity_simplify_l127_127684

theorem trigonometric_identity_simplify (x y : ℝ) : 
  sin(x + y)^2 + cos(x)^2 + cos(x + y)^2 - 2 * sin(x + y) * cos(x) * cos(y) =
  1 - sin(2 * x) * cos(y)^2 - cos(x)^2 * sin(2 * y) :=
  sorry

end trigonometric_identity_simplify_l127_127684


namespace square_area_l127_127012

open Real

def dist (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem square_area (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 0) (h2 : y1 = 3) (h3 : x2 = 4) (h4 : y2 = 0) :
  let side_length := dist x1 y1 x2 y2
  in side_length^2 = 25 :=
by
  sorry

end square_area_l127_127012


namespace constant_function_of_inequality_l127_127156

theorem constant_function_of_inequality (f : ℝ → ℝ) :
  (∀ x y z : ℝ, f(x + y) + f(y + z) + f(z + x) ≥ 3 * f(x + 2 * y + 3 * z)) →
  ∃ c : ℝ, ∀ x : ℝ, f(x) = c :=
by
  sorry

end constant_function_of_inequality_l127_127156


namespace problem_statement_l127_127135

noncomputable def f : ℝ → ℝ :=
sorry

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) : Prop :=
∀ x, f (x + 1) = f (x - 1)

def monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

theorem problem_statement :
  even_function f ∧
  periodic_function f ∧
  monotonically_increasing f (set.Icc 0 1) →
  (f (-3 / 2) < f (4 / 3) ∧ f (4 / 3) < f 1) :=
sorry

end problem_statement_l127_127135


namespace length_of_platform_proof_l127_127457

def convert_speed_to_mps (kmph : Float) : Float := kmph * (5/18)

def distance_covered (speed : Float) (time : Float) : Float := speed * time

def length_of_platform (total_distance : Float) (train_length : Float) : Float := total_distance - train_length

theorem length_of_platform_proof :
  let speed_kmph := 72.0
  let speed_mps := convert_speed_to_mps speed_kmph
  let time_seconds := 36.0
  let train_length := 470.06
  let total_distance := distance_covered speed_mps time_seconds
  length_of_platform total_distance train_length = 249.94 :=
by
  sorry

end length_of_platform_proof_l127_127457


namespace each_player_plays_36_minutes_l127_127804

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l127_127804


namespace simplify_expression_l127_127683

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x⁻¹ + 2 * y⁻¹)⁻² = (x^2 * y^2) / (3 * y + 2 * x)^2 :=
by
  sorry

end simplify_expression_l127_127683


namespace equal_playing_time_for_each_player_l127_127793

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l127_127793


namespace mean_equals_d_l127_127414

noncomputable def sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

theorem mean_equals_d
  (a b c d e : ℝ)
  (h_a : a = sqrt 2)
  (h_b : b = sqrt 18)
  (h_c : c = sqrt 200)
  (h_d : d = sqrt 32)
  (h_e : e = sqrt 8) :
  d = (a + b + c + e) / 4 := by
  -- We insert proof steps here normally
  sorry

end mean_equals_d_l127_127414


namespace octal_to_binary_conversion_l127_127132

theorem octal_to_binary_conversion : 
  octal_to_binary 127 = "1010111" := 
sorry

end octal_to_binary_conversion_l127_127132


namespace probability_of_neither_red_is_correct_l127_127072

noncomputable def probability_neither_red (total_marbles : ℕ) (red_marbles : ℕ) : ℚ :=
  let non_red_marbles := total_marbles - red_marbles
  let prob_non_red := (non_red_marbles : ℚ) / (total_marbles : ℚ)
  prob_non_red * prob_non_red

theorem probability_of_neither_red_is_correct (total_marbles : ℕ) (red_marbles : ℕ) 
  (h_total : total_marbles = 48) (h_red : red_marbles = 12) :
  probability_neither_red total_marbles red_marbles = 9 / 16 :=
by
  rw [h_total, h_red, probability_neither_red]
  norm_num
  sorry

end probability_of_neither_red_is_correct_l127_127072


namespace floor_sum_proof_l127_127650

noncomputable def floor_sum (x y z w : ℝ) : ℝ :=
  x + y + z + w

theorem floor_sum_proof
  (x y z w : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hw_pos : 0 < w)
  (h1 : x^2 + y^2 = 2010)
  (h2 : z^2 + w^2 = 2010)
  (h3 : x * z = 1008)
  (h4 : y * w = 1008) :
  ⌊floor_sum x y z w⌋ = 126 :=
by
  sorry

end floor_sum_proof_l127_127650


namespace seedling_probability_l127_127073

theorem seedling_probability (germination_rate survival_rate : ℝ)
    (h_germ : germination_rate = 0.9) (h_survival : survival_rate = 0.8) : 
    germination_rate * survival_rate = 0.72 :=
by
  rw [h_germ, h_survival]
  norm_num

end seedling_probability_l127_127073


namespace machines_produce_bottles_l127_127421

theorem machines_produce_bottles :
  (∀ (n m : ℕ) (rate : ℕ), ((6 * rate) = 300) → (n * rate) * m = 10 * rate * 4 → (10 * rate * 4 = 2000)) := 
begin
  intros n m rate h1 h2,
  rw h1 at h2,
  sorry
end

end machines_produce_bottles_l127_127421


namespace find_period_l127_127905

theorem find_period (A P R : ℕ) (I : ℕ) (T : ℚ) 
  (hA : A = 1120) 
  (hP : P = 896) 
  (hR : R = 5) 
  (hSI : I = A - P) 
  (hT : I = (P * R * T) / 100) :
  T = 5 := by 
  sorry

end find_period_l127_127905


namespace cos_105_proof_l127_127857

noncomputable def cos_105_degrees : Real :=
  cos 105 * (π / 180)

theorem cos_105_proof : cos_105_degrees = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_proof_l127_127857


namespace solve_problem_method_count_l127_127778

def total_methods (f s : ℕ) : ℕ :=
  f * s

theorem solve_problem_method_count (f s : ℕ) (hf : f = 2) (hs : s = 3) : total_methods f s = 6 := by
  rw [hf, hs]
  simp [total_methods]
  exact eq.refl 6

end solve_problem_method_count_l127_127778


namespace solve_arctan_equation_l127_127686

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan (1 / x) + Real.arctan (1 / (x^3))

theorem solve_arctan_equation (x : ℝ) (hx : x = (1 + Real.sqrt 5) / 2) :
  f x = Real.pi / 4 :=
by
  rw [hx]
  sorry

end solve_arctan_equation_l127_127686


namespace math_problem_l127_127588

-- Define the condition
def condition (x : ℝ) : Prop := (256:ℝ)^4 = (64:ℝ)^x

-- Define what needs to be proved
def problem (x : ℝ) : Prop := 2^(-x) = 1 / (2^(16 / 3))

-- State the theorem based on the above definitions
theorem math_problem (x : ℝ) (h : condition x) : problem x :=
sorry

end math_problem_l127_127588


namespace monopoly_favor_durable_machine_competitive_market_prefer_durable_l127_127753

-- Define the conditions
def consumer_valuation : ℕ := 10
def durable_cost : ℕ := 6

-- Define the monopoly decision problem: prove C > 3
theorem monopoly_favor_durable_machine (C : ℕ) : 
  consumer_valuation * 2 - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

-- Define the competitive market decision problem: prove C > 3
theorem competitive_market_prefer_durable (C : ℕ) :
  2 * consumer_valuation - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

end monopoly_favor_durable_machine_competitive_market_prefer_durable_l127_127753


namespace coeff_x2_term_l127_127139

-- Define the two polynomials
noncomputable def p : Polynomial ℝ := Polynomial.C 6 + Polynomial.X * Polynomial.C (-3) + Polynomial.X^2 * Polynomial.C 5 + Polynomial.X^3 * Polynomial.C 2
noncomputable def q : Polynomial ℝ := Polynomial.C (-5) + Polynomial.X * Polynomial.C (-2) + Polynomial.X^2 * Polynomial.C 3

-- Proof statement
theorem coeff_x2_term :
  (p * q).coeff 2 = -19 := 
sorry

end coeff_x2_term_l127_127139


namespace num_terms_rational_coefficient_l127_127144

/-- Determine the number of terms in the polynomial of x obtained from 
the expansion of (sqrt(3) * x + cbrt(2))^100 where the coefficient is rational -/
theorem num_terms_rational_coefficient :
  let x := 3;
  let y := 2;
  let n := 100;
  ∃ (count : Nat), 
    (count = 17 ∧ 
    (∀ r : Nat, (r % 3 = 0 ∧ (n - r) % 2 = 0) → 
      (C n r * (Real.sqrt x)^(n - r) * (Real.cbrt y)^r = C n r * (Real.sqrt x)^(n - r) * (Real.cbrt y)^r))) := 
sorry

end num_terms_rational_coefficient_l127_127144


namespace sum_horizontal_equals_vertical_l127_127716

variable (P : Polygon) (is_convex : IsConvex P) (are_lattice_vertices : LatticeVertices P) (no_side_along_grid_line : NoSideAlongGridLine P)

theorem sum_horizontal_equals_vertical : SumHorizontalSegments P = SumVerticalSegments P :=
sorry

end sum_horizontal_equals_vertical_l127_127716


namespace michael_earnings_l127_127312

-- Define variables for pay rates and hours.
def regular_pay_rate : ℝ := 7.00
def overtime_multiplier : ℝ := 2
def regular_hours : ℝ := 40
def overtime_hours (total_hours : ℝ) : ℝ := total_hours - regular_hours

-- Define the earnings functions.
def regular_earnings (hourly_rate : ℝ) (hours : ℝ) : ℝ := hourly_rate * hours
def overtime_earnings (hourly_rate : ℝ) (multiplier : ℝ) (hours : ℝ) : ℝ := hourly_rate * multiplier * hours

-- Total earnings calculation.
def total_earnings (total_hours : ℝ) : ℝ := 
regular_earnings regular_pay_rate regular_hours + 
overtime_earnings regular_pay_rate overtime_multiplier (overtime_hours total_hours)

-- The theorem to prove the correct earnings for 42.857142857142854 hours worked.
theorem michael_earnings : total_earnings 42.857142857142854 = 320 := by
  sorry

end michael_earnings_l127_127312


namespace compound_interest_correct_l127_127422

-- Define the given conditions in Lean 4
def P : ℝ := 6000
def r : ℝ := 0.10
def n : ℕ := 2
def t : ℝ := 1

-- Define the formula for compound interest
def compound_interest (P r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- Define the target amount to prove
def target_amount : ℝ := 6615

-- Statement of the proof problem in Lean 4
theorem compound_interest_correct : compound_interest P r n t = target_amount :=
by
  -- Insert the direct proof steps here or use "sorry" if skipping
  sorry

end compound_interest_correct_l127_127422


namespace cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127866

theorem cos_105_eq_sqrt2_sub_sqrt6_div4 :
  cos (105 * real.pi / 180) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by
  -- Definitions and conditions
  have cos_60 : cos (60 * real.pi / 180) = 1/2 := by sorry
  have cos_45 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  have sin_60 : sin (60 * real.pi / 180) = real.sqrt 3 / 2 := by sorry
  have sin_45 : sin (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  -- Use the angle addition formula: cos (a + b) = cos a * cos b - sin a * sin b
  have add_formula := cos_add (60 * real.pi / 180) (45 * real.pi / 180)
  -- Combine the results using the given known values
  rw [cos_60, cos_45, sin_60, sin_45] at add_formula
  exact add_formula

end cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127866


namespace equal_play_time_l127_127800

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l127_127800


namespace number_of_quadratic_residues_l127_127143

theorem number_of_quadratic_residues (n : ℕ) (p : ℕ) [fact (nat.prime p)] 
    (hp : p = 1009) (hn : n = 1007) :
  (∃ count : ℕ, count = (finset.filter (λ a, 
    (quadratic_residue a p) ∧ quadratic_residue (a + 1) p) 
    (finset.range n).card)
  ∧ count = 251) := sorry

end number_of_quadratic_residues_l127_127143


namespace profit_is_5000_l127_127334

namespace HorseshoeProfit

-- Defining constants and conditions
def initialOutlay : ℝ := 10000
def costPerSet : ℝ := 20
def sellingPricePerSet : ℝ := 50
def numberOfSets : ℝ := 500

-- Calculating the profit
def profit : ℝ :=
  let revenue := numberOfSets * sellingPricePerSet
  let manufacturingCosts := initialOutlay + (costPerSet * numberOfSets)
  revenue - manufacturingCosts

-- The main theorem: the profit is $5,000
theorem profit_is_5000 : profit = 5000 := by
  sorry

end HorseshoeProfit

end profit_is_5000_l127_127334


namespace equal_playing_time_l127_127810

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l127_127810


namespace line_through_p_perpendicular_to_polar_axis_l127_127207

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem line_through_p_perpendicular_to_polar_axis :
  ∃ (ρ : ℝ → ℝ), ∀ θ, ρ θ = - (1 / Real.cos θ) :=
by
  let P := polar_to_cartesian 1 Real.pi
  have P_coords : P = (-1, 0) := by
    simp [polar_to_cartesian, Real.cos_pi, Real.sin_pi]
  use (λ θ, - (1 / Real.cos θ))
  intro θ
  sorry

end line_through_p_perpendicular_to_polar_axis_l127_127207


namespace equal_play_time_l127_127795

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l127_127795


namespace range_of_m_l127_127534

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = 3 * m + 1)
  (h2 : x + 2 * y = 3)
  (h3 : 2 * x - y < 1) : 
  m < 1 := 
sorry

end range_of_m_l127_127534


namespace transformed_triangle_area_l127_127349

variable {α : Type} [LinearOrder α] [Field α]

theorem transformed_triangle_area (a b c : α) (g : α → α) :
  let S := 50
  let T := 3 / 4 * S
  area_triangle (a, g a) (b, g b) (c, g c) = S →
  area_triangle (a / 4, 3 * g a) (b / 4, 3 * g b) (c / 4, 3 * g c) = T := by
  sorry

end transformed_triangle_area_l127_127349


namespace students_team_three_classes_l127_127475

theorem students_team_three_classes
  (students : Fin 30 → Fin 3 → Fin 3)
  (h : ∀ (n : Fin 3) (team : Fin 3), ∃ (student : Fin 10), team = student) :
  ∃ (x y : Fin 30), x ≠ y ∧ (∀ (class : Fin 3), students x class = students y class) :=
begin
  sorry
end

end students_team_three_classes_l127_127475


namespace lena_candy_bars_l127_127641

/-- Lena has some candy bars. She needs 5 more candy bars to have 3 times as many as Kevin,
and Kevin has 4 candy bars less than Nicole. Lena has 5 more candy bars than Nicole.
How many candy bars does Lena have? -/
theorem lena_candy_bars (L K N : ℕ) 
  (h1 : L + 5 = 3 * K)
  (h2 : K = N - 4)
  (h3 : L = N + 5) : 
  L = 16 :=
sorry

end lena_candy_bars_l127_127641


namespace lukas_points_in_5_games_l127_127429

theorem lukas_points_in_5_games (avg_points_per_game : ℕ) (games_played : ℕ) (total_points : ℕ)
  (h_avg : avg_points_per_game = 12) (h_games : games_played = 5) : total_points = 60 :=
by
  sorry

end lukas_points_in_5_games_l127_127429


namespace bars_not_sold_l127_127892

theorem bars_not_sold (cost_per_bar : ℕ) (total_bars : ℕ) (amount_made : ℕ) (bars_sold : ℕ) (bars_not_sold : ℕ) :
  cost_per_bar = 2 → total_bars = 13 → amount_made = 18 → bars_sold = amount_made / cost_per_bar →
  bars_not_sold = total_bars - bars_sold → bars_not_sold = 4 :=
by
  intros h_cost_per_bar h_total_bars h_amount_made h_bars_sold h_bars_not_sold
  rw [h_cost_per_bar, h_total_bars, h_amount_made] at *
  simp [h_bars_sold, h_bars_not_sold]
  sorry

end bars_not_sold_l127_127892


namespace find_initial_sale_percentage_l127_127101

-- Define the variables
variable (P : ℝ) -- original price
variable (x : ℝ) -- initial sale percentage in decimal form

-- Define the conditions
def initial_sale_price := (1 - x) * P
def second_sale_price := initial_sale_price * 0.90
def final_price := 0.45 * P

-- The theorem we want to prove
theorem find_initial_sale_percentage (h : final_price = second_sale_price) : x = 0.5 := 
by
    sorry

end find_initial_sale_percentage_l127_127101


namespace expression_computation_l127_127490

theorem expression_computation : 
    ( (1 / 3)^(-1) - Real.logb 2 8 + (0.5^(-2) - 2) * ((27 / 8)^(2 / 3)) = 9 / 2 ) := by
    sorry

end expression_computation_l127_127490


namespace necessary_but_not_sufficient_condition_l127_127538

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l127_127538


namespace largest_common_divisor_462_330_l127_127041

theorem largest_common_divisor_462_330 :
  ∃ d : ℕ, (d ∣ 462) ∧ (d ∣ 330) ∧
  (∀ k : ℕ, (k ∣ 462) → (k ∣ 330) → k ≤ d) ∧ d = 66 :=
by
  have prime_factors_462 : prime_factors 462 = [2, 3, 7, 11] :=
    sorry
  have prime_factors_330 : prime_factors 330 = [2, 3, 5, 11] :=
    sorry
  have common_factors := [2, 3, 11]
  have largest_common_divisor := 2 * 3 * 11
  use 66
  split
  sorry -- d ∣ 462 and d ∣ 330 proof
  split
  sorry -- d ∣ 330 proof
  split
  sorry -- d is the largest common factor proof
  refl -- d = 66

end largest_common_divisor_462_330_l127_127041


namespace floor_sum_eq_55_l127_127643

noncomputable def x : ℝ := 9.42

theorem floor_sum_eq_55 : ∀ (x : ℝ), x = 9.42 → (⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋) = 55 := by
  intros
  sorry

end floor_sum_eq_55_l127_127643


namespace andrey_valentin_distance_l127_127118

/-- Prove the distance between Andrey and Valentin is 107 meters,
    given that Andrey finishes 60 meters ahead of Boris and Boris finishes 50 meters ahead of Valentin.
    Assumptions:
    - Andrey, Boris, and Valentin participated in a 1 km race.
    - All participants ran at a constant speed. -/
theorem andrey_valentin_distance :
  ∀ (a b c : ℝ), -- Speeds of Andrey, Boris, and Valentin respectively.
  (a ≠ 0) →     -- Non-zero speed
  (b = 0.94 * a) → 
  (c = 0.95 * b) →
  (distance_a := 1000 : ℝ) →
  (distance_valentin : ℝ := c * (distance_a / a)) →
  (distance_andrey_valentin : ℝ := distance_a - distance_valentin) →
  distance_andrey_valentin = 107 :=
by {
  intros a b c ha hb hc distance_a distance_valentin distance_andrey_valentin,
  subst_vars,
  sorry
}

end andrey_valentin_distance_l127_127118


namespace maximum_expression_value_l127_127346

theorem maximum_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 33 :=
sorry

end maximum_expression_value_l127_127346


namespace necessary_but_not_sufficient_condition_l127_127945

def p (x : ℝ) : Prop := |4 * x - 3| ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≤ 0

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, p x → q x a) ∧ ¬ (∀ x : ℝ, q x a → p x) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end necessary_but_not_sufficient_condition_l127_127945


namespace intersected_smaller_cubes_l127_127452

-- Define the conditions and final proof statement in Lean
open Real -- Assume real numbers can be used

-- Define the side length of the cube
def side_length : ℝ := 5

-- Define the number of smaller cubes in the larger cube
def num_smaller_cubes : ℤ := 125

-- Define the mid point of the diagonal of the cube
def midpoint := (side_length / 2, side_length / 2, side_length / 2)

-- Define the equation of the plane
def plane_eq (x y z : ℝ) : Prop := x + y + z = 3 * side_length / 2

-- Define the interval representing the sum k + m + n
def sum_interval (k m n : ℤ) : Prop := k + m + n = 5 ∨ k + m + n = 6 ∨ k + m + n = 7

-- Define the number of intersected smaller cubes
def num_intersected_cubes : ℤ := 55

theorem intersected_smaller_cubes : 
  ∀ (x y z : ℝ) (k m n : ℤ),
    cube.side_length = 5 ∧
    cube.num_smaller_cubes = 125 ∧
    plane_eq x y z ∧
    midpoint = (side_length / 2, side_length / 2, side_length / 2) ∧
    (0 ≤ k ∧ k ≤ 4) ∧ (0 ≤ m ∧ m ≤ 4) ∧ (0 ≤ n ∧ n ≤ 4) ∧
    sum_interval k m n →
    intersected.num_intersected_cubes = 55 := by
  sorry

end intersected_smaller_cubes_l127_127452


namespace rectangle_area_l127_127900

/-- Define the length of the rectangle -/
def length : ℝ := 5

/-- Define the width of the rectangle -/
def width : ℝ := 17 / 20

/-- Define the expected area of the rectangle -/
def expected_area : ℝ := 4.25

/-- Prove that the area of a rectangle with the given length and width is the expected area -/
theorem rectangle_area :
  (length * width) = expected_area := 
by 
  sorry

end rectangle_area_l127_127900


namespace ant_prob_in_safe_zone_l127_127114

-- Definition of variables and parameters
def side_length : ℝ := 3
def radius : ℝ := 1

-- Statement of the problem
theorem ant_prob_in_safe_zone : 
  (let area_square := side_length * side_length in
   let area_circles := 4 * (radius * radius * Real.pi) in
   let safe_area := area_square - area_circles in
   (safe_area / area_square) = 1 - (Real.pi / 9)) :=
by 
   sorry

end ant_prob_in_safe_zone_l127_127114


namespace assignment_plans_proof_l127_127249

noncomputable def total_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let positions := ["translation", "tour guide", "etiquette", "driver"]
  -- Definitions for eligible volunteers for the first two positions
  let first_positions := ["Xiao Zhang", "Xiao Zhao"]
  let remaining_positions := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Assume the computation for the exact number which results in 36
  36

theorem assignment_plans_proof : total_assignment_plans = 36 := 
  by 
  -- Proof skipped
  sorry

end assignment_plans_proof_l127_127249


namespace reduced_price_l127_127419

open Real

noncomputable def original_price : ℝ := 33.33

variables (P R: ℝ) (Q : ℝ)

theorem reduced_price
  (h1 : R = 0.75 * P)
  (h2 : P * 500 / P = 500)
  (h3 : 0.75 * P * (Q + 5) = 500)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  -- The proof will be provided here
  sorry

end reduced_price_l127_127419


namespace circle_radius_equivalence_l127_127034

theorem circle_radius_equivalence (OP_radius : ℝ) (QR : ℝ) (a : ℝ) (P : ℝ × ℝ) (S : ℝ × ℝ)
  (h1 : P = (12, 5))
  (h2 : S = (a, 0))
  (h3 : QR = 5)
  (h4 : OP_radius = 13) :
  a = 8 := 
sorry

end circle_radius_equivalence_l127_127034


namespace angle_OA_plane_ABC_l127_127473

noncomputable def sphere_radius (A B C : Type*) (O : Type*) : ℝ :=
  let surface_area : ℝ := 48 * Real.pi
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let radius := Real.sqrt (surface_area / (4 * Real.pi))
  radius

noncomputable def length_AC (A B C : Type*) : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let AC := Real.sqrt (AB ^ 2 + BC ^ 2 - 2 * AB * BC * Real.cos angle_ABC)
  AC

theorem angle_OA_plane_ABC 
(A B C O : Type*)
(radius : ℝ)
(AC : ℝ) :
radius = 2 * Real.sqrt 3 ∧
AC = 2 * Real.sqrt 3 ∧ 
(AB : ℝ) = 2 ∧ 
(BC : ℝ) = 4 ∧ 
(angle_ABC : ℝ) = Real.pi / 3
→ ∃ (angle_OA_plane_ABC : ℝ), angle_OA_plane_ABC = Real.arccos (Real.sqrt 3 / 3) :=
by
  intro h
  sorry

end angle_OA_plane_ABC_l127_127473


namespace gillian_phone_bill_next_year_l127_127936

theorem gillian_phone_bill_next_year (monthly_bill : ℝ) (increase_percentage : ℝ) (months_in_year : ℕ) :
  monthly_bill = 50 → increase_percentage = 0.10 → months_in_year = 12 → 
  (monthly_bill * (1 + increase_percentage)) * (months_in_year) = 660 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  linarith

end gillian_phone_bill_next_year_l127_127936


namespace nonneg_real_values_of_x_count_l127_127922

theorem nonneg_real_values_of_x_count :
  let S := {x : ℝ | 0 ≤ x ∧ ∃ k : ℕ, k ≤ 13 ∧ k = Real.sqrt (169 - Real.cbrt x)} in
  S.card = 14 :=
by
  sorry

end nonneg_real_values_of_x_count_l127_127922


namespace real_solutions_l127_127161

theorem real_solutions : 
  ∃ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ∧ (x = 7 ∨ x = -2) :=
sorry

end real_solutions_l127_127161


namespace exists_three_sticks_form_triangle_l127_127067

theorem exists_three_sticks_form_triangle 
  (l : Fin 5 → ℝ) 
  (h1 : ∀ i, 2 < l i) 
  (h2 : ∀ i, l i < 8) : 
  ∃ (i j k : Fin 5), i < j ∧ j < k ∧ 
    (l i + l j > l k) ∧ 
    (l j + l k > l i) ∧ 
    (l k + l i > l j) :=
sorry

end exists_three_sticks_form_triangle_l127_127067


namespace cos_alpha_add_beta_over_2_l127_127587

variable (α β : ℝ)

-- Conditions
variables (h1 : 0 < α ∧ α < π / 2)
variables (h2 : -π / 2 < β ∧ β < 0)
variables (h3 : Real.cos (π / 4 + α) = 1 / 3)
variables (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Result
theorem cos_alpha_add_beta_over_2 :
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_over_2_l127_127587


namespace fraction_simplify_l127_127483

theorem fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by sorry

end fraction_simplify_l127_127483


namespace gillian_phone_bill_next_year_l127_127937

theorem gillian_phone_bill_next_year (monthly_bill : ℝ) (increase_percentage : ℝ) (months_in_year : ℕ) :
  monthly_bill = 50 → increase_percentage = 0.10 → months_in_year = 12 → 
  (monthly_bill * (1 + increase_percentage)) * (months_in_year) = 660 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  linarith

end gillian_phone_bill_next_year_l127_127937


namespace machine_x_produces_40_percent_l127_127455

theorem machine_x_produces_40_percent (T X Y : ℝ) 
  (h1 : X + Y = T)
  (h2 : 0.009 * X + 0.004 * Y = 0.006 * T) :
  X = 0.4 * T :=
by
  sorry

end machine_x_produces_40_percent_l127_127455


namespace solve_exponential_eq_l127_127053

theorem solve_exponential_eq (x : ℝ) :
  (3 * 4^x + (1 / 3) * 9^(x + 2) = 6 * 4^(x + 1) - (1 / 2) * 9^(x + 1)) →
  x = -1/2 := by
  sorry

end solve_exponential_eq_l127_127053


namespace minimal_area_isosceles_l127_127194

def triangle (α h : ℝ) : Type :=
sorry

theorem minimal_area_isosceles (α h : ℝ) :
  ∃ (ABC : triangle α h), is_isosceles_triangle ABC ∧ (∀ t, area t ≥ area ABC) :=
sorry

end minimal_area_isosceles_l127_127194


namespace isosceles_triangle_angles_sum_l127_127732

theorem isosceles_triangle_angles_sum (x : ℝ) 
  (h_triangle_sum : ∀ a b c : ℝ, a + b + c = 180)
  (h_isosceles : ∃ a b : ℝ, (a = 50 ∧ b = x) ∨ (a = x ∧ b = 50)) :
  50 + x + (180 - 50 * 2) + 65 + 80 = 195 :=
by
  sorry

end isosceles_triangle_angles_sum_l127_127732


namespace matrix_inverse_solution_l127_127703

theorem matrix_inverse_solution :
  (∃ c d : ℚ, (∀ (A : Matrix (Fin 2) (Fin 2) ℚ), A = !![ 4, -2; c, d ] → A * A = 1) →
  (c, d) = (15 / 2, -4)) :=
begin
  sorry
end

end matrix_inverse_solution_l127_127703


namespace probability_all_from_same_tribe_l127_127377

-- Definitions based on the conditions of the problem
def total_people := 24
def tribe_count := 3
def people_per_tribe := 8
def quitters := 3

-- We assume each person has an equal chance of quitting and the quitters are chosen independently
-- The probability that all three people who quit belong to the same tribe

theorem probability_all_from_same_tribe :
  ((3 * (Nat.choose people_per_tribe quitters)) / (Nat.choose total_people quitters) : ℚ) = 1 / 12 := 
  by 
    sorry

end probability_all_from_same_tribe_l127_127377


namespace max_area_of_rectangle_l127_127182

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * (x + y) = 60) : x * y ≤ 225 :=
by sorry

end max_area_of_rectangle_l127_127182


namespace initial_population_l127_127062

theorem initial_population (P : ℝ)
  (h1 : P * 1.25 * 0.75 = 18750) : P = 20000 :=
sorry

end initial_population_l127_127062


namespace binom_20_5_l127_127489

-- Definition of the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Problem statement
theorem binom_20_5 : binomial_coefficient 20 5 = 7752 := 
by {
  -- Proof goes here
  sorry
}

end binom_20_5_l127_127489


namespace equal_perimeter_lines_concur_l127_127192

variable {α : Type}
variables {A B C : α}
variables {a b c : ℕ}
variables {A' B' C' : α}
variables {m n : ℕ}

-- Definitions and conditions from part a)
def triangle (ABC : α) (a b c : ℕ) := 
  ∃ BC CA AB : ℕ, BC = a ∧ CA = b ∧ AB = c

def equal_perimeter_line (A A' B C : α) (a b c m n : ℕ) :=
  ∃ AA' : ℕ, (c + m + AA') = (b + n + AA') ∧ m + n = a

-- The proof statement
theorem equal_perimeter_lines_concur 
  (h_triangle : triangle (A, B, C) a b c)
  (h_Aline : equal_perimeter_line A A' B C a b c m n)
  (h_Bline : equal_perimeter_line B B' C A b c a m n)
  (h_Cline : equal_perimeter_line C C' A B c a b m n) :
  ∃ P : α, 
    concur (A, A', P) ∧
    concur (B, B', P) ∧
    concur (C, C', P) :=
sorry

end equal_perimeter_lines_concur_l127_127192


namespace equal_play_time_l127_127819

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l127_127819


namespace beth_sheep_l127_127110

-- Definition: number of sheep Beth has (B)
variable (B : ℕ)

-- Condition 1: Aaron has 7 times as many sheep as Beth
def Aaron_sheep (B : ℕ) := 7 * B

-- Condition 2: Together, Aaron and Beth have 608 sheep
axiom together_sheep : B + Aaron_sheep B = 608

-- Theorem: Prove that Beth has 76 sheep
theorem beth_sheep : B = 76 :=
sorry

end beth_sheep_l127_127110


namespace Vasya_did_not_make_it_on_time_l127_127689

-- Conditions definitions
variables (v : ℝ) (v > 0)  -- Speed of the bicycle must be positive
def distance_total : ℝ := 50  -- Total distance to the institute
def distance_bicycle : ℝ := 10  -- Distance traveled by bicycle
def distance_walk : ℝ := 16  -- Distance walked after bicycle broke down
def distance_car : ℝ := 24  -- Distance traveled by car
def speed_walk : ℝ := v / 2.5  -- Walking speed
def speed_car : ℝ := 6 * v  -- Car speed

-- Calculate time taken for each segment of the journey
def time_bicycle : ℝ := distance_bicycle / v
def time_walk : ℝ := distance_walk / speed_walk
def time_car : ℝ := distance_car / speed_car

-- Total travel time with breakdown
def total_travel_time : ℝ := time_bicycle + time_walk + time_car

-- Time to travel the entire distance by bicycle
def on_time_travel : ℝ := distance_total / v

-- Proposition stating that Vasya did not make it on time
theorem Vasya_did_not_make_it_on_time : total_travel_time v > on_time_travel v :=
by
  sorry

end Vasya_did_not_make_it_on_time_l127_127689


namespace sin_600_eq_neg_sqrt3_div_2_proof_l127_127885

noncomputable def sin_600_eq_neg_sqrt3_div_2 : Prop :=
  sin (600 : ℝ) = - (Real.sqrt 3) / 2

theorem sin_600_eq_neg_sqrt3_div_2_proof :
  sin_600_eq_neg_sqrt3_div_2 := 
by
  sorry

end sin_600_eq_neg_sqrt3_div_2_proof_l127_127885


namespace Q1_Q2_l127_127952

-- Given Conditions
def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 1 else sorry -- condition for seq_a is not clearly given except for a₁ = 1

def sum_S (n : ℕ) : ℕ := sorry -- We need to define sum_S as S_n

axiom eq1 (n : ℕ) (h : n ≥ 2) : (n - 1) * sum_S n - n * sum_S (n - 1) = n^2 - n

-- Proof statement for Question 1
theorem Q1 (n : ℕ) (h : n ≥ 1) : sum_S n = n^2 :=
by sorry

-- Definition for function f(n)
def f (n : ℕ) : ℝ :=
  ((List.range (n - 1)).map (λ k => 1 - 1 / (sum_S (k + 2):ℝ))).prod

-- Proof statement for Question 2
theorem Q2 : ∀ n, n ≥ 2 → f(n) ≤ 3/4 :=
by sorry

end Q1_Q2_l127_127952


namespace factors_of_1386_with_more_than_three_factors_l127_127585

theorem factors_of_1386_with_more_than_three_factors : 
  ∃ n : ℕ, 
  (n = 1386) ∧ 
  (∃ k : ℕ, (number_of_factors (factor_count n)) > 3) ∧ 
  (count_factors_with_more_than_3_factors n = 5) 
:=
sorry

/-- Count the total number of factors of a given number based on its prime factorization -/
def number_of_factors (pf : ℕ → ℕ) : ℕ :=
  pf 1 * pf 2 * pf 3 * pf 4 -- Example. Needs actual implementation based on the prime factorization

/-- Given a number, return its prime factor counts as a function mapping prime bases to their exponents -/
def factor_count (n : ℕ) : ℕ → ℕ :=
  λ p, if p = 2 then 1 else if p = 3 then 2 else if p = 7 then 1 else if p = 11 then 1 else 0  -- for 1386

/-- For a given number n, count how many of its factors have more than 3 factors -/
def count_factors_with_more_than_3_factors (n : ℕ) : ℕ :=
  [6, 22, 42, 66, 154, 1386].filter (λ x, number_of_factors (factor_count x) > 3).length  -- Example list

end factors_of_1386_with_more_than_three_factors_l127_127585


namespace polyhedron_vertex_number_assignment_l127_127330

theorem polyhedron_vertex_number_assignment (V : Type) [fintype V] (E : set (V × V)) :
  (∃ (f : V → ℕ), (∀ (v1 v2 : V), (v1, v2) ∈ E → ¬ coprime (f v1) (f v2)) ∧ 
                    (∀ (v1 v2 : V), (v1, v2) ∉ E → coprime (f v1) (f v2))) :=
sorry

end polyhedron_vertex_number_assignment_l127_127330


namespace max_receptivity_compare_receptivity_l127_127029

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 43
  else if 10 < x ∧ x <= 16 then 59
  else if 16 < x ∧ x <= 30 then -3 * x + 107
  else 0 -- To cover the case when x is outside the given ranges

-- Problem 1
theorem max_receptivity :
  f 10 = 59 ∧ ∀ x, 10 < x ∧ x ≤ 16 → f x = 59 :=
by
  sorry

-- Problem 2
theorem compare_receptivity :
  f 5 > f 20 :=
by
  sorry

end max_receptivity_compare_receptivity_l127_127029


namespace perpendicular_slope_l127_127525

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 2 * y = 10) :
  ∀ (m' : ℝ), m' = -2 / 5 :=
by
  sorry

end perpendicular_slope_l127_127525


namespace machine_depletion_years_l127_127088

theorem machine_depletion_years (present_value : ℝ) (depletion_rate : ℝ) (final_value : ℝ) :
    present_value = 700 → depletion_rate = 0.10 → final_value = 567 → 
    real.log(final_value / present_value) / real.log(1 - depletion_rate) = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end machine_depletion_years_l127_127088


namespace find_a_l127_127971

variable (a b : ℝ)
def A : Set ℝ := {1, -2}
def B : Set ℝ := {x | x^2 + a * x + b = 0}

theorem find_a (h : A = B) : a = 1 :=
by
  sorry  -- proof steps go here

end find_a_l127_127971


namespace total_lives_l127_127389

theorem total_lives (initial_players additional_players lives_per_player : ℕ) (h1 : initial_players = 4) (h2 : additional_players = 5) (h3 : lives_per_player = 3) :
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end total_lives_l127_127389


namespace a_range_l127_127545

noncomputable def f (x a : ℝ) : ℝ := |2 * x - 1| + |x - 2 * a|

def valid_a_range (a : ℝ) : Prop :=
∀ x, 1 ≤ x ∧ x ≤ 2 → f x a ≤ 4

theorem a_range (a : ℝ) : valid_a_range a → (1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) := 
sorry

end a_range_l127_127545


namespace ellipse_properties_l127_127335

def ellipse_equation (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

def major_axis_length (a : ℝ) : ℝ := 2 * a

def foci (a b c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0))

def eccentricity (c a : ℝ) : ℝ := c / a

theorem ellipse_properties :
  (∃ x y : ℝ, ellipse_equation x y) →
  let a := 2
  let b := Real.sqrt 3
  let major_axis_len := major_axis_length a
  let c := Real.sqrt (a^2 - b^2)
  let foci_coords := foci a b c
  let ecc := eccentricity c a
  (ecc = 1/2) ∧ (foci_coords = ((-1, 0), (1, 0))) :=
by
  sorry

end ellipse_properties_l127_127335


namespace compare_cube_roots_l127_127849

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem compare_cube_roots : 2 + cube_root 7 < cube_root 60 :=
sorry

end compare_cube_roots_l127_127849


namespace force_exerted_by_pulley_on_axis_l127_127109

-- Define the basic parameters given in the problem
def m1 : ℕ := 3 -- mass 1 in kg
def m2 : ℕ := 6 -- mass 2 in kg
def g : ℕ := 10 -- acceleration due to gravity in m/s^2

-- From the problem, we know that:
def F1 : ℕ := m1 * g -- gravitational force on mass 1
def F2 : ℕ := m2 * g -- gravitational force on mass 2

-- To find the tension, setup the equations
def a := (F2 - F1) / (m1 + m2) -- solving for acceleration between the masses

def T := (m1 * a) + F1 -- solving for the tension in the rope considering mass 1

-- Define the proof statement to find the force exerted by the pulley on its axis
theorem force_exerted_by_pulley_on_axis : 2 * T = 80 :=
by
  -- Annotations or calculations can go here
  sorry

end force_exerted_by_pulley_on_axis_l127_127109


namespace total_noodles_and_pirates_l127_127623

-- Condition definitions
def pirates : ℕ := 45
def noodles : ℕ := pirates - 7

-- Theorem stating the total number of noodles and pirates
theorem total_noodles_and_pirates : (noodles + pirates) = 83 := by
  sorry

end total_noodles_and_pirates_l127_127623


namespace remaining_seat_in_sample_l127_127487

noncomputable def systematic_sampling_seat {total_students : ℕ} {sample_size : ℕ} 
  (sampled_seats : Finset ℕ) : Prop :=
  total_students = 60 ∧
  sample_size = 5 ∧
  sampled_seats = {3, 15, 45, 53} ∧
  ∃ (remaining_seat : ℕ), remaining_seat ∈ (Finset.range (25 + 12) \ Finset.range 25) ∧ 
  remaining_seat ≠ 37

theorem remaining_seat_in_sample {total_students : ℕ} {sample_size : ℕ} 
  (sampled_seats : Finset ℕ) : systematic_sampling_seat sampled_seats → 
  ∃ (remaining_seat : ℕ), remaining_seat ∈ (Finset.range (25 + 12) \ Finset.range 25) ∧ 
  remaining_seat ≠ 37 :=
by
  sorry

end remaining_seat_in_sample_l127_127487


namespace find_num_of_planets_l127_127394

def num_of_planets_total_condition {S P : ℕ} : S + P = 200

def num_of_solar_systems_per_planet_condition {S P : ℕ} : S = 8 * P

theorem find_num_of_planets (P : ℕ) (S : ℕ) 
  (h1 : num_of_solar_systems_per_planet_condition S P)
  (h2 : num_of_planets_total_condition S P) :
  P = 22 := 
sorry

end find_num_of_planets_l127_127394


namespace number_of_cookies_l127_127666

def candy : ℕ := 63
def brownies : ℕ := 21
def people : ℕ := 7
def dessert_per_person : ℕ := 18

theorem number_of_cookies : 
  (people * dessert_per_person) - (candy + brownies) = 42 := 
by
  sorry

end number_of_cookies_l127_127666


namespace correct_statements_l127_127994

-- Definitions for conditions
variables (l m : Type) [linear_ordered_add_comm_group l] [linear_ordered_add_comm_group m]

-- Define the lines l and m, and planes α and β
variables (α β : set l)
variables (α_perp_β : α ∩ β = m)
variables (l_in_β : l ⊆ β)
variables (l_perp_m : l ∩ m = ∅)

-- Define the properties as required in the statements
def perp (x y : set l) : Prop := x ∩ y = ∅ -- Define perpendicular
def parallel (x y : set l) : Prop := x ⊆ y or y ⊆ x or x = ∅ -- Define parallel

-- Statement 1 Conditions
variables (l_perp_α : perp l α) (m_parallel_β : parallel m β) (α_perp_β : perp α β)

-- Statement 2 Conditions
variables (l_parallel_m : parallel l m) (m_perp_α : perp m α) (l_perp_β : perp l β)

-- Statement 3 Conditions
variables (l_parallel_α : parallel l α) (m_parallel_β : parallel m β) (α_parallel_β : parallel α β)

-- Statement 4 Conditions
variables (α_perp_β : perp α β) (α_inter_β : α_intersect β = m) (l_subset_β : l ⊆ β) (l_perp_m : perp l m)

-- The actual proof theorem putting all of this together
theorem correct_statements :
  (∃ l m α β,
    (perp l α ∧ parallel m β ∧ perp α β → ¬ perp l m) ∧
    (parallel l m ∧ perp m α ∧ perp l β → parallel α β) ∧
    (parallel l α ∧ parallel m β ∧ parallel α β → ¬ parallel l m) ∧
    (perp α β ∧ α ∩ β = m ∧ l ⊆ β ∧ perp l m → perp l α)) :=
begin
  existsi α,
  existsi β,
  existsi l,
  existsi m,
  split,
  { rintros ⟨l_perp_α, m_parallel_β, α_perp_β⟩,
    exact not_perp_l_m_of_l_perp_α_m_parallel_β_α_perp_β l_perp_α m_parallel_β α_perp_β },
  split,
  { rintros ⟨l_parallel_m, m_perp_α, l_perp_β⟩,
    exact parallel_α_β_of_l_parallel_m_and_m_perp_α_and_l_perp_β l_parallel_m m_perp_α l_perp_β },
  split,
  { rintros ⟨l_parallel_α, m_parallel_β, α_parallel_β⟩,
    exact not_parallel_l_m_of_l_parallel_α_and_m_parallel_β_and_α_parallel_β l_parallel_α m_parallel_β α_parallel_β },
  { rintros ⟨α_perp_β, α_inter_β, l_subset_β, l_perp_m⟩,
    exact perp_l_α_of_perp_α_β_and_α_inter_β_and_l_subset_β_and_perp_l_m α_perp_β α_inter_β l_subset_β l_perp_m }
end

end correct_statements_l127_127994


namespace minimum_value_l127_127559

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : 
  (∃ x, (∀ y, y = (1 / a) + (4 / b) → y ≥ x) ∧ x = 9 / 2) :=
by
  sorry

end minimum_value_l127_127559


namespace find_numbers_l127_127529

theorem find_numbers (p q x : ℝ) (h : (p ≠ 1)) :
  ((p * x) ^ 2 - x ^ 2) / (p * x + x) = q ↔ x = q / (p - 1) ∧ p * x = (p * q) / (p - 1) := 
by
  sorry

end find_numbers_l127_127529


namespace equal_playing_time_l127_127813

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l127_127813


namespace math_classes_participating_l127_127661

theorem math_classes_participating (total_volunteers required_extra_volunteers volunteer_students_per_class volunteer_teachers : ℕ) (h1 : total_volunteers = 50) (h2 : required_extra_volunteers = 7) (h3 : volunteer_students_per_class = 5) (h4 : volunteer_teachers = 13) : 
  (total_volunteers - required_extra_volunteers) = (volunteer_students_per_class * 6 + volunteer_teachers) :=
by 
  rw [h1, h2] at *;
  have : total_volunteers - required_extra_volunteers = 43 := by norm_num;
  have : volunteer_students_per_class * 6 + volunteer_teachers = 43 := by norm_num;
  exact eq.trans this.symm sorry

end math_classes_participating_l127_127661


namespace original_pencils_correct_l127_127390

-- Define the conditions
variables (total_pencils now: ℕ) (pencils_added original_pencils: ℕ)
-- State the given conditions as assumptions
axiom pencil_conditions : total_pencils = 60 ∧ pencils_added = 27 ∧ (original_pencils + pencils_added = total_pencils)

-- State the theorem to be proved
theorem original_pencils_correct : original_pencils = 33 :=
by
  -- Get the conditions from the axiom
  have ⟨Htotal, Hadd, Htotal_eq⟩ := pencil_conditions

  -- Use the fact that (original_pencils + pencils_added = total_pencils) implies that original_pencils = total_pencils - pencils_added
  calc
    original_pencils = total_pencils - pencils_added : by sorry
    ... = 33 : by sorry

end original_pencils_correct_l127_127390


namespace find_n_l127_127590

theorem find_n (n : ℕ) (h1 : ∃ k : ℕ, 12 - n = k * k) : n = 11 := 
by sorry

end find_n_l127_127590


namespace initial_money_amount_l127_127425

theorem initial_money_amount 
  (X : ℝ) 
  (h : 0.70 * X = 350) : 
  X = 500 := 
sorry

end initial_money_amount_l127_127425


namespace determine_sold_cakes_l127_127843

def initial_cakes := 121
def new_cakes := 170
def remaining_cakes := 186
def sold_cakes (S : ℕ) : Prop := initial_cakes - S + new_cakes = remaining_cakes

theorem determine_sold_cakes : ∃ S, sold_cakes S ∧ S = 105 :=
by
  use 105
  unfold sold_cakes
  simp
  sorry

end determine_sold_cakes_l127_127843


namespace sum_first_10_terms_120_l127_127253

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * a 0 + (n * (n + 1) * (a 1 - a 0)) / 2

theorem sum_first_10_terms_120 
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_first_n_terms a 9 = 120) :
  a 1 + a 8 = 24 := 
sorry

end sum_first_10_terms_120_l127_127253


namespace sum_of_angles_of_inscribed_quadrilateral_l127_127078

/--
Given a quadrilateral EFGH inscribed in a circle, and the measures of ∠EGH = 50° and ∠GFE = 70°,
then the sum of the angles ∠EFG + ∠EHG is 60°.
-/
theorem sum_of_angles_of_inscribed_quadrilateral
  (E F G H : Type)
  (circumscribed : True) -- This is just a place holder for the circle condition
  (angle_EGH : ℝ) (angle_GFE : ℝ)
  (h1 : angle_EGH = 50)
  (h2 : angle_GFE = 70) :
  ∃ (angle_EFG angle_EHG : ℝ), angle_EFG + angle_EHG = 60 := sorry

end sum_of_angles_of_inscribed_quadrilateral_l127_127078


namespace largest_n_factorial_product_l127_127518

theorem largest_n_factorial_product (n : ℕ) :
  (∃ b : ℕ, b ≥ 5 ∧ n! = (∏ i in finset.range (n - 5 + b + 1), i + 1) / (b!)) → n = 4 :=
by
  sorry

end largest_n_factorial_product_l127_127518


namespace x_is_irrational_l127_127171

def decimal_representation (k : ℕ) : list ℕ := -- Assumed function
  sorry 

noncomputable def infinite_decimal_number (k : ℕ) : ℝ :=
  let coeffs := list.join (list.map (decimal_representation ∘ pow 1987) (list.range (k+1))) in
  let digits := 0 :: 1 :: 1 :: 9 :: 8 :: 7 :: coeffs in 
  list.foldr (λ d acc, acc / 10 + d) 0 digits / 10^k

theorem x_is_irrational : ∀ k : ℕ, ¬ is_rat (infinite_decimal_number k) :=
begin
  sorry
end

end x_is_irrational_l127_127171


namespace sum_primes_meeting_conditions_l127_127910

def is_prime (n : ℕ) : Prop := nat.prime n

def meets_conditions (q : ℕ) : Prop :=
  q % 6 = 2 ∧ q % 7 = 4

def primes_between (a b : ℕ) : list ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a)

noncomputable def sum_of_primes_meeting_conditions_between_1_and_150 : ℕ :=
  (primes_between 1 150).filter (λ q, is_prime q ∧ meets_conditions q).sum

-- The theorem we need to prove
theorem sum_primes_meeting_conditions :
  sum_of_primes_meeting_conditions_between_1_and_150 = 122 :=
sorry

end sum_primes_meeting_conditions_l127_127910


namespace part1_part2_part3_l127_127570

-- Problem Definitions

def f (m x : ℝ) : ℝ := m * x^2 - 2 * x + 1

-- Lean statement for part 1
theorem part1 (x : ℝ) (h : x ∈ set.Icc (-2:ℝ) 1) : 
  ∃ (y : ℝ), y ∈ set.range (λ x, f 1 x) ∧ y ∈ set.Icc (0:ℝ) 9 :=
by
  sorry

-- Lean statement for part 2
theorem part2 (m : ℝ) (h : f m m = 0) :
  m = 1 ∨ m = (-1 + Real.sqrt 5) / 2 ∨ m = (-1 - Real.sqrt 5) / 2 :=
by
  sorry

-- Lean statement for part 3
theorem part3 (m : ℝ) (h : ∀ (x : ℝ), x ∈ set.Icc (1:ℝ) 2 → f m x ≥ 0) : 
  1 ≤ m :=
by
  sorry

end part1_part2_part3_l127_127570


namespace exponent_multiplication_l127_127594

variable (a x y : ℝ)

theorem exponent_multiplication :
  a^x = 2 →
  a^y = 3 →
  a^(x + y) = 6 :=
by
  intros h1 h2
  sorry

end exponent_multiplication_l127_127594


namespace eighth_term_of_binomial_expansion_is_l127_127351

-- Define the binomial expansion term
def binomial_term (a b : ℤ) (n r : ℕ) (x : ℤ) : ℤ :=
  ((-1)^r * a^(n - r) * (Nat.choose n r) * x^(n - r))

-- Define the specific parameters
def a : ℤ := 2
def b : ℤ := -1
def n : ℕ := 9
def r : ℕ := 7
def x : ℤ := x

-- Statement to prove
theorem eighth_term_of_binomial_expansion_is :
  binomial_term a b n r x = -144 * x^2 :=
by
  sorry

end eighth_term_of_binomial_expansion_is_l127_127351


namespace number_of_new_trailer_homes_l127_127530

-- Definitions coming from the conditions
def initial_trailers : ℕ := 30
def initial_avg_age : ℕ := 15
def years_passed : ℕ := 5
def current_avg_age : ℕ := initial_avg_age + years_passed

-- Let 'n' be the number of new trailer homes added five years ago
variable (n : ℕ)

def new_trailer_age : ℕ := years_passed
def total_trailers : ℕ := initial_trailers + n
def total_ages : ℕ := (initial_trailers * current_avg_age) + (n * new_trailer_age)
def combined_avg_age := total_ages / total_trailers

theorem number_of_new_trailer_homes (h : combined_avg_age = 12) : n = 34 := 
sorry

end number_of_new_trailer_homes_l127_127530


namespace probability_even_sum_l127_127665

noncomputable def spinner1 := [2, 3, 7]
noncomputable def spinner2 := [5, 3, 6]

def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the probability of an event happening in a list of events
def probability (A : List ℕ) (p : ℕ → Prop) : ℚ :=
  (A.filter p).length / A.length

-- Probabilities of landing on even or odd numbers for both spinners
def P_even1 := probability spinner1 is_even
def P_even2 := probability spinner2 is_even
def P_odd1 := probability spinner1 (λ n => ¬ is_even n)
def P_odd2 := probability spinner2 (λ n => ¬ is_even n)

-- Probability of the sum being even
def P_even_sum := P_even1 * P_even2 + P_odd1 * P_odd2

theorem probability_even_sum : P_even_sum = 5/9 := 
by sorry

end probability_even_sum_l127_127665


namespace inequality_proof_l127_127544

theorem inequality_proof (a b : ℝ) (h1 : b < 0) (h2 : 0 < a) : a - b > a + b :=
sorry

end inequality_proof_l127_127544


namespace inequality_sums_products_l127_127054

open Real
open Finset

def s_k (n : ℕ) (x : Fin n → ℝ) (k : ℕ) : ℝ :=
  (univ.powerset k).sum (λ s, (s.prod (λ i, x i)))

theorem inequality_sums_products (n k : ℕ) (x : Fin n → ℝ)
  (h1 : 0 < k) (h2 : k < n) (hx : ∀ i : Fin n, 0 < x i) :
  s_k n x k * s_k n x (n - k) ≥ (Nat.choose n k) ^ 2 * s_k n x n :=
sorry

end inequality_sums_products_l127_127054


namespace nonneg_real_values_of_x_count_l127_127921

theorem nonneg_real_values_of_x_count :
  let S := {x : ℝ | 0 ≤ x ∧ ∃ k : ℕ, k ≤ 13 ∧ k = Real.sqrt (169 - Real.cbrt x)} in
  S.card = 14 :=
by
  sorry

end nonneg_real_values_of_x_count_l127_127921


namespace ball_travel_distance_l127_127439

def rebounds (n : ℕ) (h : ℝ) : ℝ :=
  if n = 0 then h
  else (3 / 4) * rebounds (n - 1) h

def total_distance : ℝ :=
  let initial_height := 160
  let drops := [160, 120, 90, 67.5, 50.625]
  let rebounds := [120, 90, 67.5, 50.625]
  (drops.sum + rebounds.sum)

theorem ball_travel_distance :
  total_distance = 816.25 :=
sorry

end ball_travel_distance_l127_127439


namespace find_smallest_number_l127_127726

theorem find_smallest_number 
  (x a b c : ℕ) 
  (h_arith_mean : (a + b + c) / 3 = 30) 
  (h_median : b = 25) 
  (h_largest : c = b + 7) 
  (h_sum : a + b + c = 90) : 
  x = 33 :=
by {
  -- Verify the smallest number is correct
  assume h1 : x = a,
  sorry
}

end find_smallest_number_l127_127726


namespace measure_angle_GAF_l127_127836

/-!
Given:
1. Triangle ABC is equilateral.
2. Triangle ABC shares a common side BC with rectangle BCFG.
3. BF = 2 * BC.

To Prove:
The measure of angle GAF is 30 degrees.
-/
noncomputable theory

open_locale real

def is_equilateral (A B C : ℝ) (ABC : ℕ) : Prop :=
  A = B ∧ B = C ∧ C = 60 -- Equilateral triangle has all angles 60 degrees

def is_rectangle (BC BF CG FG : ℝ) (BCGF : ℕ) : Prop :=
  BC = CG ∧ BF = FG ∧ BF = 2 * BC ∧ (BC ^ 2 + CG ^ 2 = BF ^ 2) -- Rectangle properties

def angle_measure (A B : ℝ) :ℝ :=
  A - B

theorem measure_angle_GAF (A B C F G : ℝ) (ABC : ℕ) (BCFG: ℕ):
  is_equilateral A B C ABC →
  is_rectangle BC BF CG FG BCFG →
  angle_measure A (B + F) = 30 :=
  by sorry

end measure_angle_GAF_l127_127836


namespace equal_playing_time_l127_127812

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l127_127812


namespace miniature_tower_height_l127_127510

-- Definitions of conditions
def actual_tower_height := 60
def actual_dome_volume := 200000 -- in liters
def miniature_dome_volume := 0.4 -- in liters

-- Goal: Prove the height of the miniature tower
theorem miniature_tower_height
  (actual_tower_height: ℝ)
  (actual_dome_volume: ℝ)
  (miniature_dome_volume: ℝ) : 
  actual_tower_height = 60 ∧ actual_dome_volume = 200000 ∧ miniature_dome_volume = 0.4 →
  (actual_tower_height / ( (actual_dome_volume / miniature_dome_volume)^(1/3) )) = 1.2 :=
by
  sorry

end miniature_tower_height_l127_127510


namespace determine_a_l127_127233

theorem determine_a (a : ℝ) : (∃ b : ℝ, (3 * (x : ℝ))^2 - 2 * 3 * b * x + b^2 = 9 * x^2 - 27 * x + a) → a = 20.25 :=
by
  sorry

end determine_a_l127_127233


namespace strings_completely_pass_each_other_l127_127733

-- Define the problem parameters
def d : ℝ := 30    -- distance between A and B in cm
def l1 : ℝ := 151  -- length of string A in cm
def l2 : ℝ := 187  -- length of string B in cm
def v1 : ℝ := 2    -- speed of string A in cm/s
def v2 : ℝ := 3    -- speed of string B in cm/s
def r1 : ℝ := 1    -- burn rate of string A in cm/s
def r2 : ℝ := 2    -- burn rate of string B in cm/s

-- The proof problem statement
theorem strings_completely_pass_each_other : ∀ (T : ℝ), T = 40 :=
by
  sorry

end strings_completely_pass_each_other_l127_127733


namespace parabola_shift_left_l127_127007

theorem parabola_shift_left (x : ℝ) :
  (λ x, x^2) (x + 2) = (x + 2)^2 := by
  sorry

end parabola_shift_left_l127_127007


namespace lyndee_friends_count_l127_127662

-- Definitions
variables (total_chicken total_garlic_bread : ℕ)
variables (lyndee_chicken lyndee_garlic_bread : ℕ)
variables (friends_large_chicken_count : ℕ)
variables (friends_large_chicken : ℕ)
variables (friend_garlic_bread_per_friend : ℕ)

def remaining_chicken (total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken : ℕ) : ℕ :=
  total_chicken - (lyndee_chicken + friends_large_chicken_count * friends_large_chicken)

def remaining_garlic_bread (total_garlic_bread lyndee_garlic_bread : ℕ) : ℕ :=
  total_garlic_bread - lyndee_garlic_bread

def total_friends (friends_large_chicken_count remaining_chicken remaining_garlic_bread friend_garlic_bread_per_friend : ℕ) : ℕ :=
  friends_large_chicken_count + remaining_chicken + remaining_garlic_bread / friend_garlic_bread_per_friend

-- Theorem statement
theorem lyndee_friends_count : 
  total_chicken = 11 → 
  total_garlic_bread = 15 →
  lyndee_chicken = 1 →
  lyndee_garlic_bread = 1 →
  friends_large_chicken_count = 3 →
  friends_large_chicken = 2 →
  friend_garlic_bread_per_friend = 3 →
  total_friends friends_large_chicken_count 
                (remaining_chicken total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken)
                (remaining_garlic_bread total_garlic_bread lyndee_garlic_bread)
                friend_garlic_bread_per_friend = 7 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof omitted
  sorry

end lyndee_friends_count_l127_127662


namespace total_is_83_l127_127621

def number_of_pirates := 45
def number_of_noodles := number_of_pirates - 7
def total_number_of_noodles_and_pirates := number_of_noodles + number_of_pirates

theorem total_is_83 : total_number_of_noodles_and_pirates = 83 := by
  sorry

end total_is_83_l127_127621


namespace painting_falls_l127_127502

-- Define the types and structures used in the problem
namespace PaintingOnNails

-- Define a type for nails
constant Nail : Type

-- Define a set of subsets representing conditions
constant Conditions : Set (Set Nail)

/-- The key property of the conditions being an anti-chain -/
def isAntichain (conditions : Set (Set Nail)) : Prop :=
  ∀ (A B : Set Nail), A ∈ conditions → B ∈ conditions → A ≠ B → ¬(A ⊆ B)

/-- The main theorem to be proven -/
theorem painting_falls (conditions : Set (Set Nail)) : 
  isAntichain conditions → 
  (∀ (nails_removed : Set Nail), 
    (∃ (C ∈ conditions), C ⊆ nails_removed) ↔ (∃ (C ∈ conditions), ∀ n ∈ C, n ∉ nails_removed)) :=
by
  sorry

end PaintingOnNails

end painting_falls_l127_127502


namespace length_of_pentagon_side_l127_127782

noncomputable def length_of_side_of_pentagon : ℝ :=
  let area := 200 in
  let tan_pi_5 := Real.tan (Real.pi / 5) in
  Real.sqrt (800 / (5 * tan_pi_5))

theorem length_of_pentagon_side :
  length_of_side_of_pentagon ≈ 14.83 :=
by
  sorry

end length_of_pentagon_side_l127_127782


namespace symmetry_axis_f_x_minus_1_plus_2_l127_127715

theorem symmetry_axis_f_x_minus_1_plus_2
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f(-x+2) = f(x+2)) :
  ∃ a : ℝ, f(x-1) + 2 = g(x) →  ∀ x : ℝ, g(a + x) = g(a - x) :=
by
  sorry

end symmetry_axis_f_x_minus_1_plus_2_l127_127715


namespace Chemistry_marks_l127_127498

theorem Chemistry_marks (english_marks mathematics_marks physics_marks biology_marks : ℕ) (avg_marks : ℝ) (num_subjects : ℕ) (total_marks : ℕ)
  (h1 : english_marks = 72)
  (h2 : mathematics_marks = 60)
  (h3 : physics_marks = 35)
  (h4 : biology_marks = 84)
  (h5 : avg_marks = 62.6)
  (h6 : num_subjects = 5)
  (h7 : total_marks = avg_marks * num_subjects) :
  (total_marks - (english_marks + mathematics_marks + physics_marks + biology_marks) = 62) :=
by
  sorry

end Chemistry_marks_l127_127498


namespace calculate_value_l127_127127

theorem calculate_value : (81 ^ 0.25) * (81 ^ 0.20) * 2 = 20.09 := 
by sorry

end calculate_value_l127_127127


namespace avg_speed_trip_l127_127764

theorem avg_speed_trip :
  let speed1 := 35 -- miles per hour for the first 4 hours
  let time1 := 4   -- hours
  let speed2 := 53 -- miles per hour for the remaining hours
  let time2 := 20  -- hours
  let total_time := 24 -- total time of the trip

  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2

  let avg_speed := total_distance / total_time
  
  avg_speed = 50 -- miles per hour
:= 
begin
  sorry
end

end avg_speed_trip_l127_127764


namespace maximum_expression_value_l127_127347

theorem maximum_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 33 :=
sorry

end maximum_expression_value_l127_127347


namespace midpoint_of_AB_l127_127618

noncomputable def midpoint (p q : Point) : Point := 
  ⟨(p.1 + q.1) / 2, (p.2 + q.2) / 2⟩

structure Point :=
(x : ℝ)
(y : ℝ)

namespace Point

theorem midpoint_of_AB (A B : Point) 
  (hA : A = ⟨-1, 2⟩) 
  (hB : B = ⟨3, 0⟩) : 
  (midpoint A B) = ⟨1, 1⟩ := by
  sorry

end Point

end midpoint_of_AB_l127_127618


namespace base_six_sub_add_l127_127037

theorem base_six_sub_add :
  let base6_to_base10 := λ (n : Nat), n.digits 6.reverse.reverse.sum (λ d i, d * (6 ^ i))
  base6_to_base10 1254 - base6_to_base10 432 + base6_to_base10 221 = base6_to_base10 1043 := 
sorry

end base_six_sub_add_l127_127037


namespace ratio_sum_constant_ratio_diff_constant_l127_127551

theorem ratio_sum_constant (O A B R : Point) (a b l : ℝ) :
  circle_passing_through O A B R →
  segment(O, A) = a →
  segment(O, B) = b →
  segment(O, R) = l →
  ∃ k : ℝ, ∀ circle_passing_through O, a + b = k * l := sorry

theorem ratio_diff_constant (O A B R : Point) (a b l : ℝ) :
  circle_passing_through O A B R →
  segment(O, A) = a →
  segment(O, B) = b →
  segment(O, R) = l →
  ∃ k : ℝ, ∀ circle_passing_through O, |a - b| = k * l := sorry

end ratio_sum_constant_ratio_diff_constant_l127_127551


namespace inequality_proof_l127_127556

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end inequality_proof_l127_127556


namespace largest_fraction_among_given_l127_127744

theorem largest_fraction_among_given (f₁ f₂ f₃ f₄ f₅ : ℚ)
  (h1 : f₁ = 5 / 12) (h2 : f₂ = 7 / 16) (h3 : f₃ = 23 / 48) (h4 : f₄ = 99 / 200) (h5 : f₅ = 201 / 400) :
  ∀ f, f ∈ {f₁, f₂, f₃, f₄, f₅} → f ≤ 201 / 400 := 
by
  sorry

end largest_fraction_among_given_l127_127744


namespace Jerrie_situps_more_than_Carrie_l127_127480

noncomputable def situps_per_minute_difference : ℕ :=
  let Barney_rate := 45 in
  let Carrie_rate := 2 * Barney_rate in
  let Jerrie_rate := Carrie_rate + x in
  let total_situps := Barney_rate * 1 + Carrie_rate * 2 + Jerrie_rate * 3 in
  if total_situps = 510 then x else 0

theorem Jerrie_situps_more_than_Carrie (x : ℕ) (H : situps_per_minute_difference = 5) : x = 5 :=
by
  sorry

end Jerrie_situps_more_than_Carrie_l127_127480


namespace value_of_coefficients_l127_127305

theorem value_of_coefficients :
  let a0 := 32 in
  let sum := 1 in
  a0 + (a1 + a2 + a3 + a4 + a5) = sum →
  (a1 + a2 + a3 + a4 + a5) = -31 :=
by
  sorry

end value_of_coefficients_l127_127305


namespace SC_perp_BC_cos_angle_SC_AB_l127_127629

variables {Point : Type} [EuclideanGeometry Point]

-- Definition of the specific points and conditions in the problem space
variables (S A B C : Point)
variables (AC BC SB : ℝ)
variables (angle_SAB angle_SAC angle_ACB : ℝ)
variables [Decidable (AC = 2)]
variables [Decidable (BC = Real.sqrt 13)]
variables [Decidable (SB = Real.sqrt 29)]
variables [Decidable (angle_SAB = 90)]
variables [Decidable (angle_SAC = 90)]
variables [Decidable (angle_ACB = 90)]

-- Additional geometrical definitions
def SC : ℝ := (SB^2 + AC^2 - (2 * SB * AC * Math.cos angle_SAC))^(1/2)

-- The hypotheses based on the given conditions
hypothesis h1 : angle_SAB = 90
hypothesis h2 : angle_SAC = 90
hypothesis h3 : angle_ACB = 90
hypothesis h4 : AC = 2
hypothesis h5 : BC = Real.sqrt 13
hypothesis h6 : SB = Real.sqrt 29

-- Proof problems
theorem SC_perp_BC : ∀ (S A B C : Point), 
  angle_SAB = 90 ∧ angle_SAC = 90 ∧ angle_ACB = 90 ∧ AC = 2 ∧ BC = Real.sqrt 13 ∧ SB = Real.sqrt 29 → EuclideanGeometry.Perpendicular (SC - BC) BC := 
by
  sorry

theorem cos_angle_SC_AB : ∀ (S A B C : Point), 
  angle_SAB = 90 ∧ angle_SAC = 90 ∧ angle_ACB = 90 ∧ AC = 2 ∧ BC = Real.sqrt 13 ∧ SB = Real.sqrt 29 → 
  ∃ α, α = angle (SC - BC) (AC + BC) ∧ Math.cos α = (Real.sqrt 17) / 17 :=
by
  sorry

end SC_perp_BC_cos_angle_SC_AB_l127_127629


namespace quadrant_of_z1_div_z2_l127_127210

def z1 : ℂ := 1 - 2 * Complex.i
def z2 : ℂ := 2 + 3 * Complex.i

theorem quadrant_of_z1_div_z2 : (z1 / z2).re < 0 ∧ (z1 / z2).im < 0 :=
by
  sorry

end quadrant_of_z1_div_z2_l127_127210


namespace no_such_alpha_exists_l127_127888

theorem no_such_alpha_exists
  (α : ℝ)
  (H1 : irrational (real.cos α))
  (H2 : rational (real.cos (2 * α)) ∧ rational (real.cos (3 * α)) ∧ rational (real.cos (4 * α)) ∧ rational (real.cos (5 * α))) :
  false :=
sorry

end no_such_alpha_exists_l127_127888


namespace sum_of_roots_of_Q_l127_127020

theorem sum_of_roots_of_Q
    (Q : ℝ → ℝ)
    (hQ : ∃ (a b c : ℝ), ∀ x : ℝ, Q x = a * x^2 + b * x + c)
    (ineq : ∀ x : ℝ, Q (x^3 - x) ≥ Q (x^2 - 1)) :
    ∃ (a b c : ℝ), (Q = λ x, a * x^2 + b * x + c) ∧ (a ≠ 0) ∧ (b = -a) ∧ ((-b / a) = 1) :=
by
  sorry

end sum_of_roots_of_Q_l127_127020


namespace molly_age_l127_127313

theorem molly_age : 14 + 6 = 20 := by
  sorry

end molly_age_l127_127313


namespace initial_savings_l127_127731

theorem initial_savings (computer_cost : ℕ) (extra_needed : ℕ) (old_computer_value : ℕ) (total_save : ℕ)
  (h1 : computer_cost = 80)
  (h2 : extra_needed = 10)
  (h3 : old_computer_value = 20)
  (h4 : total_save = computer_cost - extra_needed) :
    total_save - old_computer_value = 50 :=
by
  rw [h1, h2, h3, h4] -- Substitute given values
  norm_num -- Calculate and normalize
  exact rfl -- Confirm correctness

end initial_savings_l127_127731


namespace rent_increase_percentage_l127_127352

theorem rent_increase_percentage :
  ∀ (initial_avg new_avg rent : ℝ) (num_friends : ℝ),
    num_friends = 4 →
    initial_avg = 800 →
    new_avg = 850 →
    rent = 800 →
    ((num_friends * new_avg) - (num_friends * initial_avg)) / rent * 100 = 25 :=
by
  intros initial_avg new_avg rent num_friends h_num h_initial h_new h_rent
  sorry

end rent_increase_percentage_l127_127352


namespace equilateral_triangle_ratio_l127_127397

theorem equilateral_triangle_ratio (A a : ℝ) (hA : A ≠ a) :
  let P := 3 * A
  let p := 3 * a
  let R := (A * Real.sqrt 3) / 3
  let r := (a * Real.sqrt 3) / 3
  (P / p = R / r) :=
by
  let P := 3 * A
  let p := 3 * a
  let R := (A * Real.sqrt 3) / 3
  let r := (a * Real.sqrt 3) / 3
  have hp : p ≠ 0 := by sorry
  have hr : r ≠ 0 := by sorry
  calc
    P / p = 3 * A / (3 * a) : by sorry
       ... = A / a           : by sorry
    ... = (A * Real.sqrt 3 / 3) / (a * Real.sqrt 3 / 3) : by sorry
       ... = R / r           : by sorry

end equilateral_triangle_ratio_l127_127397


namespace rotate_150_deg_l127_127837

def initial_positions := ["Triangle at Top", "Smaller Circle at Right", "Square at Bottom", "Pentagon at Left"]
def rotation := 150

theorem rotate_150_deg (initial_positions: List String) (rotation: Int) :
  initial_positions = ["Triangle at Top", "Smaller Circle at Right", "Square at Bottom", "Pentagon at Left"] ∧
  rotation = 150 -> 
  ["Triangle at Left", "Smaller Circle at Top", "Square at Right", "Pentagon at Bottom"] := by
  sorry

end rotate_150_deg_l127_127837


namespace part1_smallest_n_zero_part2_a2j1_a2j_part3_k_and_smallest_n_l127_127876

-- Sequence definition: a_{n+1} = a_n - ⌊√a_n⌋
def seq_step (a_n : ℕ) : ℕ := a_n - (nat.sqrt a_n)

-- Part 1
theorem part1_smallest_n_zero (a : List ℕ):
  a.head = 24 →
  a[1] = seq_step a.head →
  a[2] = seq_step a[1] →
  a[3] = seq_step a[2] →
  a[4] = seq_step a[3] →
  a[5] = seq_step a[4] →
  a[6] = seq_step a[5] →
  a[7] = seq_step a[6] →
  a[8] = seq_step a[7] →
  a[9] = seq_step a[8] →
  a[9] = 0 :=
sorry

-- Part 2
theorem part2_a2j1_a2j (m j : ℕ) (H : 1 ≤ j ∧ j ≤ m) :
  a_0 = m^2 →
  ∀ i < 2 * m, 
  seq i = if i % 2 = 0 then (m - (i / 2))^2 else (m - ((i + 1) / 2)) * (m - ((i + 1) / 2)) - 1 :=
sorry

-- Part 3
theorem part3_k_and_smallest_n (m p : ℕ) (H : 1 ≤ p ∧ p ≤ m - 1) :
  (∃ k, a_k = (m - p)^2) ∧ 2 * m - 1 = n :=
sorry

end part1_smallest_n_zero_part2_a2j1_a2j_part3_k_and_smallest_n_l127_127876


namespace length_of_first_train_l127_127734

theorem length_of_first_train
  (speed1_kmph : ℝ) (speed2_kmph : ℝ)
  (time_s : ℝ) (length2_m : ℝ)
  (relative_speed_mps : ℝ := (speed1_kmph + speed2_kmph) * 1000 / 3600)
  (total_distance_m : ℝ := relative_speed_mps * time_s)
  (length1_m : ℝ := total_distance_m - length2_m) :
  speed1_kmph = 80 →
  speed2_kmph = 65 →
  time_s = 7.199424046076314 →
  length2_m = 180 →
  length1_m = 110 :=
by
  sorry

end length_of_first_train_l127_127734


namespace total_frames_l127_127269

theorem total_frames (frames_per_page : ℝ) (pages : ℝ) (h1 : frames_per_page = 143.0) (h2 : pages = 11.0) : (frames_per_page * pages = 1573.0) :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_frames_l127_127269


namespace each_player_plays_36_minutes_l127_127807

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l127_127807


namespace total_surface_area_of_cylinder_l127_127454

-- Define radius and height of the cylinder
def radius : ℝ := 5
def height : ℝ := 12

-- Theorem stating the total surface area of the cylinder
theorem total_surface_area_of_cylinder : 2 * real.pi * radius * (radius + height) = 170 * real.pi := by
  sorry

end total_surface_area_of_cylinder_l127_127454


namespace trapezoid_area_ratios_l127_127381

/-- A trapezoid with side lengths 5 cm, 15 cm, 15 cm, and 20 cm has two possible configurations. 
The quadrilateral formed by the intersection points of the angle bisectors of the trapezoid has:
- An area ratio of 1/45 relative to the area of the trapezoid in one configuration,
- An area ratio of 7/40 relative to the area of the trapezoid in the other configuration. -/
theorem trapezoid_area_ratios :
  ∃ (side_length_1 side_length_2 side_length_3 side_length_4 : ℝ), 
    side_length_1 = 5 ∧ 
    side_length_2 = 15 ∧ 
    side_length_3 = 15 ∧ 
    side_length_4 = 20 ∧
    ∃ (area_ratio_config_1 area_ratio_config_2 : ℝ), 
      area_ratio_config_1 = 1 / 45 ∧
      area_ratio_config_2 = 7 / 40 :=
begin
  -- The detailed proof steps would go here
  sorry
end

end trapezoid_area_ratios_l127_127381


namespace no_constant_term_and_integer_powers_in_expansion_l127_127204

noncomputable def sqrt (x : ℝ) := x^(1/2)

/--
Given that the absolute values of the coefficients of the first three terms of the expansion of 
(\sqrt{x} - \frac{1}{24x})^8 form an arithmetic sequence,
1. Prove that there is no constant term in the expansion.
2. Identify the terms with integer powers of x in the expansion.
-/
theorem no_constant_term_and_integer_powers_in_expansion :
  let f (x : ℝ) := (sqrt x - 1 / (24 * x)) ^ 8 in
  (∀ n : ℕ, (f x).coeff n ≠ 0 → n ≠ 0) ∧
  ((f x).coeff 0 = 1 ∧ (f x).coeff 4 = (35 / 8) * x ∧ (f x).coeff 8 = (1 / 256) * x^16) :=
by
  sorry

end no_constant_term_and_integer_powers_in_expansion_l127_127204


namespace perpendicular_lines_m_value_l127_127574

theorem perpendicular_lines_m_value :
  ∀ (m : ℝ), (∀ (x y : ℝ), x + (m^2 - m) * y = 4 * m - 1) ⟶ 
             (∀ (x y : ℝ), 2 * x - y = 5) ⟶ 
             (2 * 1 + (m^2 - m) * (-1) = 0) ⟶ (m = 2 ∨ m = -1) := by
  intros m h1 h2 h3
  sorry

end perpendicular_lines_m_value_l127_127574


namespace time_to_groom_rottweiler_l127_127270

theorem time_to_groom_rottweiler
  (R : ℕ)  -- Time to groom a rottweiler
  (B : ℕ)  -- Time to groom a border collie
  (C : ℕ)  -- Time to groom a chihuahua
  (total_time_6R_9B_1C : 6 * R + 9 * B + C = 255)  -- Total time for grooming 6 rottweilers, 9 border collies, and 1 chihuahua
  (time_to_groom_border_collie : B = 10)  -- Time to groom a border collie is 10 minutes
  (time_to_groom_chihuahua : C = 45) :  -- Time to groom a chihuahua is 45 minutes
  R = 20 :=  -- Prove that it takes 20 minutes to groom a rottweiler
by
  sorry

end time_to_groom_rottweiler_l127_127270


namespace part_one_part_two_i_part_two_ii_l127_127212

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem part_one (a b : ℝ) : 
  f (-a / 2 + 1) a b ≤ f (a^2 + 5 / 4) a b :=
sorry

theorem part_two_i (a b : ℝ) : 
  f 1 a b + f 3 a b - 2 * f 2 a b = 2 :=
sorry

theorem part_two_ii (a b : ℝ) : 
  ¬((|f 1 a b| < 1/2) ∧ (|f 2 a b| < 1/2) ∧ (|f 3 a b| < 1/2)) :=
sorry

end part_one_part_two_i_part_two_ii_l127_127212


namespace symmetric_line_exists_l127_127165

def is_symmetric_line (line₁ line₂ line_sym : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), line₁ x y →
    ∃ (a b : ℝ), 
      line₂ x a y b ∧ 
      line_sym a b

def line₁ (x y : ℝ) : Prop := y = 2 * x + 1
def line_of_symmetry (x y : ℝ) : Prop := x + y + 1 = 0
def line_sym (x y : ℝ) : Prop := x - 2 * y = 0

theorem symmetric_line_exists : is_symmetric_line line₁ line_of_symmetry line_sym := by
  sorry

end symmetric_line_exists_l127_127165


namespace each_player_plays_36_minutes_l127_127806

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l127_127806


namespace subset_families_inequality_l127_127959

theorem subset_families_inequality 
  {X : Type*} [fintype X] [decidable_eq X] (n : ℕ) (hn : fintype.card X = n)
  (𝒜 𝒝 : finset (finset X))
  (h : ∀ A ∈ 𝒜, ∀ B ∈ 𝒝, ¬(A ⊆ B ∨ B ⊆ A)) :
  (real.sqrt 𝒜.card + real.sqrt 𝒝.card ≤ 2^(7/2 : ℝ)) :=
sorry

end subset_families_inequality_l127_127959


namespace necessary_not_sufficient_l127_127434

theorem necessary_not_sufficient (x : ℝ) : (x^2 ≥ 1) ↔ (x ≥ 1 ∨ x ≤ -1) ≠ (x ≥ 1) :=
by
  sorry

end necessary_not_sufficient_l127_127434


namespace cartons_per_box_l127_127108

open Nat

theorem cartons_per_box (cartons packs sticks brown_boxes total_sticks : ℕ) 
  (h1 : cartons * (packs * sticks) * brown_boxes = total_sticks) 
  (h2 : packs = 5) 
  (h3 : sticks = 3) 
  (h4 : brown_boxes = 8) 
  (h5 : total_sticks = 480) :
  cartons = 4 := 
by 
  sorry

end cartons_per_box_l127_127108


namespace range_of_f_l127_127906

theorem range_of_f (x : ℝ) : 
  let f := λ x, (sin x)^4 - (sin x) * (cos x) + (cos x)^4 + (1/2) * (cos (2 * x))
  in -1/2 ≤ f x ∧ f x ≤ 1 := 
by
  -- Conditions from the problem
  have h1 : ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 := sorry,
  have h2 : ∀ x : ℝ, cos (2 * x) = 2 * cos x ^ 2 - 1 := sorry,
  have h3 : ∀ x : ℝ, sin x * cos x = (1 / 2) * sin (2 * x) := sorry,
  sorry

end range_of_f_l127_127906


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l127_127979

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + cos (2 * x)

theorem smallest_positive_period_of_f : (∀ x, f (x + π) = f x) ∧ (∀ ε > 0, ∃ x, 0 < x ∧ x < ε ∧ x ≠ π) :=
by sorry

theorem max_min_values_of_f_on_interval :
  (∀ x ∈ Icc 0 (π/2), f x ≤ 2) ∧ (∃ x ∈ Icc 0 (π/2), f x = 2) ∧ 
  (∀ x ∈ Icc 0 (π/2), -1 ≤ f x) ∧ (∃ x ∈ Icc 0 (π/2), f x = -1) :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l127_127979


namespace min_AB_plus_five_thirds_BF_l127_127536

theorem min_AB_plus_five_thirds_BF 
  (A : ℝ × ℝ) (onEllipse : ℝ × ℝ → Prop) (F : ℝ × ℝ)
  (B : ℝ × ℝ) (minFunction : ℝ)
  (hf : F = (-3, 0)) (hA : A = (-2,2))
  (hB : onEllipse B) :
  (∀ B', onEllipse B' → (dist A B' + 5/3 * dist B' F) ≥ minFunction) →
  minFunction = (dist A B + 5/3 * dist B F) →
  B = (-(5 * Real.sqrt 3) / 2, 2) := by
  sorry

def onEllipse (B : ℝ × ℝ) : Prop := (B.1^2) / 25 + (B.2^2) / 16 = 1

end min_AB_plus_five_thirds_BF_l127_127536


namespace probability_defective_part_correct_probability_defective_given_lathe_2_probability_defective_given_lathe_3_l127_127386

def probability_defective_part (p1 p2 p3 : ℚ) (q1 q2 q3 : ℚ) : ℚ :=
  p1 * q1 + p2 * q2 + p3 * q3

def probability_defective_given_lathe (p : ℚ) (q : ℚ) (total_defective : ℚ) : ℚ :=
  p * q / total_defective

-- Given conditions
def p1 := 0.06
def p2 := 0.05
def p3 := 0.05

def q1 := 0.25
def q2 := 0.30
def q3 := 0.45

theorem probability_defective_part_correct : 
  probability_defective_part p1 p2 p3 q1 q2 q3 = 0.0525 := by
  sorry

theorem probability_defective_given_lathe_2 :
  probability_defective_given_lathe p2 q2 (probability_defective_part p1 p2 p3 q1 q2 q3) = 2/7 := by
  sorry

theorem probability_defective_given_lathe_3 :
  probability_defective_given_lathe p3 q3 (probability_defective_part p1 p2 p3 q1 q2 q3) = 3/7 := by
  sorry

end probability_defective_part_correct_probability_defective_given_lathe_2_probability_defective_given_lathe_3_l127_127386


namespace simplify_complex_exponent_l127_127743

theorem simplify_complex_exponent :
    (- (1 / 64 : ℂ)) ^ (- 3 / 2) = -512 * Complex.i := by
    sorry

end simplify_complex_exponent_l127_127743


namespace simplify_vector_expression_l127_127685

variables (V : Type) [AddCommGroup V]

variables (A B C D : V)
variables (AC BD AB CD BC : V)

-- Define conditions
def AC_eq : AC = AB + BC := sorry
def BD_eq : BD = BC + CD := sorry

-- Translate proof problem into Lean statement
theorem simplify_vector_expression
  (h1 : AC = AB + BC)
  (h2 : BD = BC + CD) :
  AC - BD + CD - AB = 0 :=
by {
  rw [h1, h2],
  simp, -- will handle the necessary simplifications
  sorry
}

end simplify_vector_expression_l127_127685


namespace number_of_non_neg_real_values_of_x_l127_127918

noncomputable def numNonNegRealValues (x : ℝ) : ℕ :=
  ∑ k in Finset.range 14, if (169 - k^2 : ℝ) ^ 3 ≥ 0 then 1 else 0

theorem number_of_non_neg_real_values_of_x :
  numNonNegRealValues 169 = 14 :=
sorry

end number_of_non_neg_real_values_of_x_l127_127918


namespace cos_105_proof_l127_127854

noncomputable def cos_105_degrees : Real :=
  cos 105 * (π / 180)

theorem cos_105_proof : cos_105_degrees = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_proof_l127_127854


namespace parabola_shift_left_l127_127006

theorem parabola_shift_left (x : ℝ) :
  (λ x, x^2) (x + 2) = (x + 2)^2 := by
  sorry

end parabola_shift_left_l127_127006


namespace oats_per_meal_l127_127481

theorem oats_per_meal :
  let horses := 4
  let meals_per_day := 2
  let total_oats := 96
  let days := 3 in
  (total_oats / (horses * meals_per_day * days) = 4) :=
by 
  let horses := 4
  let meals_per_day := 2
  let total_oats := 96
  let days := 3
  let meals := days * meals_per_day
  let total_meals := meals * horses
  have e1 : total_meals = 24 := by norm_num
  have e2 : total_oats / total_meals = 4 := by norm_num
  exact e2

end oats_per_meal_l127_127481


namespace valid_numbers_l127_127138

-- A function to split a number into its digits and verify if they are distinct
def distinct_digits (n : ℕ) : Prop :=
  let digits := (toDigits 10 n)
  digits.length > 1 ∧ digits.nodup

-- A function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (toDigits 10 n).sum

-- A function to calculate the product of the digits of a number
def product_of_digits (n : ℕ) : ℕ :=
  (toDigits 10 n).prod

theorem valid_numbers :
  ∀ n : ℕ, distinct_digits n ∧ sum_of_digits n = product_of_digits n →
    n ∈ {123, 132, 213, 231, 312, 321} :=
by
  sorry

end valid_numbers_l127_127138


namespace sum_a6_a7_a8_is_32_l127_127288

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l127_127288


namespace sum_first_n_odd_eq_n_squared_l127_127678

theorem sum_first_n_odd_eq_n_squared (n : ℕ) : (Finset.sum (Finset.range n) (fun k => (2 * k + 1)) = n^2) := sorry

end sum_first_n_odd_eq_n_squared_l127_127678


namespace boomerang_inequality_l127_127499

variable {C : Type} [convex_polygon : ConvexPolygon C]
variable {q b s : ℕ}

def boomerang (quad : Type) : Prop := 
  ∃ (a b c d : ℝ), quadrilateral quad a b c d ∧ (internal_angle quad a b c > 180)

theorem boomerang_inequality (hC : polygon_sides C = s)
  (h_interiors : ∀ (quad1 quad2 : Type), quadrilateral_interior quad1 ∩ quadrilateral_interior quad2 = ∅)
  (h_quad_partition : ∪ i, (quadrilateral_interior i) = polygon_interior C) 
  (h_boomerang_count : count_boomerangs b) : 
  q ≥ b + (s - 2) / 2 :=
sorry

end boomerang_inequality_l127_127499


namespace gift_wrapping_combinations_l127_127084

theorem gift_wrapping_combinations 
  (wrapping_varieties : ℕ)
  (ribbon_colors : ℕ)
  (gift_card_types : ℕ)
  (H_wrapping_varieties : wrapping_varieties = 8)
  (H_ribbon_colors : ribbon_colors = 3)
  (H_gift_card_types : gift_card_types = 4) : 
  wrapping_varieties * ribbon_colors * gift_card_types = 96 := 
by
  sorry

end gift_wrapping_combinations_l127_127084


namespace positive_integer_condition_l127_127173

theorem positive_integer_condition (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℤ, n > 0 ∧ |x - |x + 3|| / x = n) ↔ x = 1 ∨ x = 3 := 
sorry

end positive_integer_condition_l127_127173


namespace find_roots_l127_127907

noncomputable def roots_of_equation : set ℂ :=
{z : ℂ | z^2 + 2*z = 16 + 8*I}

theorem find_roots : roots_of_equation = 
  { -1 + 2*real.sqrt 5 + (2*real.sqrt 5) / 5 * I, 
    -1 - 2*real.sqrt 5 - (2*real.sqrt 5) / 5 * I } :=
sorry

end find_roots_l127_127907


namespace expected_worth_is_1_33_l127_127115

noncomputable def expected_worth_of_coin_flip : ℝ :=
  let prob_heads := 2 / 3
  let profit_heads := 5
  let prob_tails := 1 / 3
  let loss_tails := -6
  (prob_heads * profit_heads + prob_tails * loss_tails)

theorem expected_worth_is_1_33 : expected_worth_of_coin_flip = 1.33 := by
  sorry

end expected_worth_is_1_33_l127_127115


namespace find_m_value_l127_127582

def vector_magnitude (x : ℝ) (y : ℝ) : ℝ :=
  real.sqrt (x*x + y*y)

theorem find_m_value (m : ℝ) :
  let a := (4, m)
  let b := (-1, 2)
  vector_magnitude (a.1 + b.1) (a.2 + b.2) = vector_magnitude (a.1 - b.1) (a.2 - b.2) →
  m = 2 :=
begin
  sorry,
end

end find_m_value_l127_127582


namespace gillian_yearly_phone_bill_l127_127934

-- Given conditions
def usual_monthly_bill : ℝ := 50
def increase_percentage : ℝ := 0.10

-- Desired result for the yearly bill after the increase
def expected_yearly_bill : ℝ := 660

-- The theorem to prove
theorem gillian_yearly_phone_bill :
  let new_monthly_bill := usual_monthly_bill * (1 + increase_percentage) in
  let yearly_bill := new_monthly_bill * 12 in
  yearly_bill = expected_yearly_bill :=
by
  sorry

end gillian_yearly_phone_bill_l127_127934


namespace line_intersection_equation_of_l4_find_a_l127_127995

theorem line_intersection (P : ℝ × ℝ)
    (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) :
  P = (-2, 2) :=
sorry

theorem equation_of_l4 (l4 : ℝ → ℝ → Prop)
    (P : ℝ × ℝ) (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) 
    (h_parallel: ∀ x y, l4 x y ↔ y = 1/2 * x + 3)
    (x y : ℝ) :
  l4 x y ↔ y = 1/2 * x + 3 :=
sorry

theorem find_a (a : ℝ) :
    (∀ x y, 2 * x + y + 2 = 0 → y = -2 * x - 2) →
    (∀ x y, a * x - 2 * y + 1 = 0 → y = 1/2 * x - 1/2) →
    a = 1 :=
sorry

end line_intersection_equation_of_l4_find_a_l127_127995


namespace sin_subtract_pi_over_6_l127_127282

theorem sin_subtract_pi_over_6 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (hcos : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_subtract_pi_over_6_l127_127282


namespace password_combinations_l127_127314

theorem password_combinations (n r k : ℕ) (hn : n = 5) (hk_fact : k.factorial = 6) (hr : r = 20) : 
  ∃ (password : list char), 
    let combinations := (n.factorial / k.factorial) in 
    combinations = r := 
begin
  sorry
end

end password_combinations_l127_127314


namespace white_white_pairs_coincide_l127_127874

theorem white_white_pairs_coincide :
  ∀ (red_half blue_half white_half : ℕ),
  red_half = 4 →
  blue_half = 7 →
  white_half = 10 →
  (∃ red_pairs blue_pairs red_white_pairs : ℕ,
    red_pairs = 3 ∧  
    blue_pairs = 4 ∧
    red_white_pairs = 3 ∧
    let total_white_pairs := white_half - red_white_pairs in
    total_white_pairs = 7) :=
by sorry

end white_white_pairs_coincide_l127_127874


namespace geometric_series_squares_sum_l127_127367

theorem geometric_series_squares_sum (a : ℝ) (r : ℝ) (h : -1 < r ∧ r < 1) :
  (∑' n : ℕ, (a * r^n)^2) = a^2 / (1 - r^2) :=
by sorry

end geometric_series_squares_sum_l127_127367


namespace selling_price_per_tire_l127_127770

theorem selling_price_per_tire 
    (cost_per_batch : ℕ := 22500) 
    (cost_per_tire : ℕ := 8) 
    (number_of_tires : ℕ := 15000) 
    (profit_per_tire : ℚ := 10.5) :
    (selling_price_per_tire : ℚ) :=
  have total_cost := cost_per_batch + cost_per_tire * number_of_tires
  have selling_price_per_tire := cost_per_tire + profit_per_tire
  selling_price_per_tire = 18.5

end selling_price_per_tire_l127_127770


namespace flour_for_original_recipe_l127_127895

theorem flour_for_original_recipe (butter_per_flour : ℝ) (multiplier : ℝ) (total_butter : ℝ) (total_flour : ℝ) :
  butter_per_flour = 2 → multiplier = 4 → total_butter = 12 → total_flour = 20 →
  (20 * butter_per_flour / (multiplier * butter_per_flour)) ≈ 13 :=
by
  intros h_butter_per_flour h_multiplier h_total_butter h_total_flour
  have h_original_butter : total_butter / multiplier = 3 := by
    rw [h_total_butter, h_multiplier]
    norm_num
  have h_proportion : butter_per_flour * total_flour = 3 * x := by
    rw [h_butter_per_flour]
    norm_num
  finish
sorry

end flour_for_original_recipe_l127_127895


namespace decimalToFrac_l127_127399

theorem decimalToFrac : (145 / 100 : ℚ) = 29 / 20 := by
  sorry

end decimalToFrac_l127_127399


namespace find_phi_l127_127591

-- Define the given condition: sqrt(3) * sin(20°)
def given_term := Real.sqrt 3 * Real.sin (20 * Real.pi / 180)

-- Define the equation to prove: cos φ - sin φ = sqrt(3) * sin(20°) and φ is acute
theorem find_phi : ∃ φ : ℝ, φ > 0 ∧ φ < Real.pi / 2 ∧ (Real.cos φ - Real.sin φ = given_term) :=
sorry

end find_phi_l127_127591


namespace parabola_shift_left_l127_127001

theorem parabola_shift_left (x : ℝ) : 
  (let y := x^2 in
  y = x^2 → 
  y = ((x + 2)^2)) :=
sorry

end parabola_shift_left_l127_127001


namespace cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127863

theorem cos_105_eq_sqrt2_sub_sqrt6_div4 :
  cos (105 * real.pi / 180) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by
  -- Definitions and conditions
  have cos_60 : cos (60 * real.pi / 180) = 1/2 := by sorry
  have cos_45 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  have sin_60 : sin (60 * real.pi / 180) = real.sqrt 3 / 2 := by sorry
  have sin_45 : sin (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  -- Use the angle addition formula: cos (a + b) = cos a * cos b - sin a * sin b
  have add_formula := cos_add (60 * real.pi / 180) (45 * real.pi / 180)
  -- Combine the results using the given known values
  rw [cos_60, cos_45, sin_60, sin_45] at add_formula
  exact add_formula

end cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127863


namespace medians_perpendicular_to_altitudes_l127_127435

noncomputable theory

/-- Let O be the intersection point of the diagonals of a convex quadrilateral ABCD.
Prove that the line passing through the points of intersection of the medians of
triangles AOB and COD is perpendicular to the line passing through the points
of intersection of the altitudes of triangles BOC and AOD. -/
theorem medians_perpendicular_to_altitudes
  (A B C D O : Type*)
  [AffinePlane A B C D]
  [IntersectionPoint (diag AC) (diag BD) O]
  (MediansIntersectAOB : IntersectionPoint (medians_intersect A O B) (medians_intersect C O D))
  (AltitudesIntersectBOC : IntersectionPoint (altitudes_intersect B O C) (altitudes_intersect A O D)) :
  is_perpendicular (line_through MediansIntersectAOB) (line_through AltitudesIntersectBOC) :=
sorry

end medians_perpendicular_to_altitudes_l127_127435


namespace farmer_brown_leg_wing_count_l127_127515

theorem farmer_brown_leg_wing_count :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let pigeons := 4
  let kangaroos := 2
  
  let chicken_legs := 2
  let chicken_wings := 2
  let sheep_legs := 4
  let grasshopper_legs := 6
  let grasshopper_wings := 2
  let spider_legs := 8
  let pigeon_legs := 2
  let pigeon_wings := 2
  let kangaroo_legs := 2

  (chickens * (chicken_legs + chicken_wings) +
  sheep * sheep_legs +
  grasshoppers * (grasshopper_legs + grasshopper_wings) +
  spiders * spider_legs +
  pigeons * (pigeon_legs + pigeon_wings) +
  kangaroos * kangaroo_legs) = 172 := 
by
  sorry

end farmer_brown_leg_wing_count_l127_127515


namespace songs_per_album_correct_l127_127332

-- Define the number of albums and total number of songs as conditions
def number_of_albums : ℕ := 8
def total_songs : ℕ := 16

-- Define the number of songs per album
def songs_per_album (albums : ℕ) (songs : ℕ) : ℕ := songs / albums

-- The main theorem stating that the number of songs per album is 2
theorem songs_per_album_correct :
  songs_per_album number_of_albums total_songs = 2 :=
by
  unfold songs_per_album
  sorry

end songs_per_album_correct_l127_127332


namespace parking_space_length_l127_127783

theorem parking_space_length {L W : ℕ} 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 126) : 
  L = 9 := 
sorry

end parking_space_length_l127_127783


namespace tangent_line_at_point_is_correct_l127_127981

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line_at_point_is_correct :
    ∀ x y : ℝ, y = f x ∧ x = 2 → (13, -32) = (f' 2, f 2 - 2 * f' 2) :=
by 
  sorry

end tangent_line_at_point_is_correct_l127_127981


namespace ruler_markings_correct_l127_127722

theorem ruler_markings_correct :
  ∃ (markings : list ℕ), 
    markings = [1, 4, 5, 14, 16, 23, 25, 31] ∧
    (∀ n : ℕ, (1 ≤ n ∧ n ≤ 33) →
      ∃ (a b : ℕ), a ∈ markings ∧ b ∈ markings ∧ n = abs (b - a)) := 
begin
  sorry
end

end ruler_markings_correct_l127_127722


namespace math_proof_problem_l127_127616

-- First part: Determining the type of curve and its polar equation.
def curve_eq (ϕ : ℝ) : Prop :=
  (λ (x y : ℝ), x = 1 + 2 * Math.cos ϕ ∧ y = sqrt 3 + 2 * Math.sin ϕ)

def polar_eq_of_curve : Prop :=
  ∀ θ : ℝ, (λ ρ : ℝ, ρ = 4 * Math.sin (θ + π / 6))

-- Second part: Finding the range of the area of triangle OAB.
def area_triangle_OAB (α : ℝ) : Prop :=
  0 < α ∧ α < π / 4 → 
  let ρ1 := 4 * Math.sin (α + π / 6) in
  let ρ2 := 4 * Math.cos (α + π / 6) in
  let area := 4 * Math.sin (2 * α + π / 3) in
  2 < area ∧ area ≤ 4

-- Complete problem statement
theorem math_proof_problem :
  (∃ (C_polar_eq : ∀ θ : ℝ, ρ = 4 * Math.sin (θ + π / 6)), ∀ ϕ : ℝ, curve_eq ϕ (1 + 2 * cos ϕ, sqrt 3 + 2 * sin ϕ)) ∧
  (∀ α : ℝ, area_triangle_OAB α) :=
by
  sorry

end math_proof_problem_l127_127616


namespace man_running_time_l127_127461

theorem man_running_time :
  let flat_distance := 2 -- in km
  let uphill_distance := 3 -- in km
  let downhill_distance := 1 -- in km
  let base_walking_speed := 5 -- in kmph
  let base_running_speed := 15 -- in kmph
  let uphill_walking_reduction := 0.25 -- 25%
  let uphill_running_reduction := 0.40 -- 40%
  let downhill_walking_reduction := 0.10 -- 10%
  let downhill_running_increase := 0.10 -- 10%
  let flat_running_speed := base_running_speed
  let uphill_running_speed := base_running_speed * (1 - uphill_running_reduction)
  let downhill_running_speed := base_running_speed * (1 + downhill_running_increase)
  let time_flat := flat_distance / flat_running_speed
  let time_uphill := uphill_distance / uphill_running_speed
  let time_downhill := downhill_distance / downhill_running_speed
  let total_time_hours := time_flat + time_uphill + time_downhill
  let total_time_minutes := total_time_hours * 60
  in total_time_minutes = 31.632 := 
by
  sorry

end man_running_time_l127_127461


namespace each_player_plays_36_minutes_l127_127809

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l127_127809


namespace student_arrangements_5x5_array_l127_127449

theorem student_arrangements_5x5_array : 
  (let rows := 5
       columns := 5
       conditions := λ r c, (r = 5 ∨ c = 5)) -- capturing the rule constraints
  in 2^rows - 1 = 31 ∧ 2^columns - 1 = 31 ∧ 31 * 31 + 1 = 962 :=
sorry

end student_arrangements_5x5_array_l127_127449


namespace midpoints_collinear_l127_127841

-- Definition of the midpoints in Lean
structure Point :=
  (x : ℝ) (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)

structure Midpoint (P Q : Point) :=
  (mid : Point)
  (mid_property : mid.x = (P.x + Q.x) / 2 ∧ mid.y = (P.y + Q.y) / 2)

-- Lean statement of the problem
theorem midpoints_collinear (A B C D E F X Y Z : Point)
  (h1 : ∃ quad : Quadrilateral, quad.A = A ∧ quad.B = B ∧ quad.C = C ∧ quad.D = D)
  (h2 : midpoint_intersection_property : (∃ P1 P2 : Point, intersection_lemma A B C D P1 = E ∧ intersection_lemma B C D A P2 = F))
  (h3 : Midpoint A C = X)
  (h4 : Midpoint B D = Y)
  (h5 : Midpoint E F = Z) : 
  collinear X Y Z :=
sorry

end midpoints_collinear_l127_127841


namespace brothers_ticket_cost_l127_127263

-- Definitions based on conditions
def ticket_cost_isabelle : ℕ := 20
def saved_brothers : ℕ := 5
def saved_isabelle : ℕ := 5
def weeks_working_isabelle : ℕ := 10
def earning_per_week : ℕ := 3
def total_earnings_isabelle : ℕ := weeks_working_isabelle * earning_per_week -- 10 * 3

-- Theorem to prove the cost of each brother's ticket
theorem brothers_ticket_cost : 
  (isabelle_total_savings := saved_isabelle + total_earnings_isabelle) = 35 →
  (remaining_for_brothers := isabelle_total_savings - ticket_cost_isabelle) = 15 →
  (combined_total_for_brothers := remaining_for_brothers + saved_brothers) = 20 →
  (cost_per_brother := combined_total_for_brothers / 2) = 10 :=
by sorry

end brothers_ticket_cost_l127_127263


namespace children_distribution_l127_127147

theorem children_distribution (a b c d N : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : a + b + c + d < 18) 
  (h5 : a * b * c * d = N) : 
  N = 120 ∧ a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 := 
by 
  sorry

end children_distribution_l127_127147


namespace cut_half_meter_from_two_thirds_l127_127998

theorem cut_half_meter_from_two_thirds (L : ℝ) (hL : L = 2 / 3) : L - 1 / 6 = 1 / 2 :=
by
  rw [hL]
  norm_num

end cut_half_meter_from_two_thirds_l127_127998


namespace sequence_a11_l127_127951

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

axiom sum_eq (n : ℕ) : n > 0 → 4 * S n = 2 * a n - n^2 + 7 * n

theorem sequence_a11 : a 11 = -2 := by
  have h1 : 4 * S 1 = 2 * a 1 - 1^2 + 7 * 1 := sum_eq 1 (by linarith)
  have h2 : a 1 = 3 := by linarith
  have h3 : ∀ n, n > 1 → a n + a (n - 1) = 4 - n := by intros; sorry
  have h4 : ∀ n, n > 2 → a (n + 1) - a (n - 1) = -1 := by intros; sorry
  have h5 : ∀ k, k % 2 = 1 → a k = 3 + (k - 1) / 2 * -1 := by intros; sorry
  have h6 : a 11 = 3 + (11 - 1) / 2 * -1 := h5 11 (by rfl)
  linarith

end sequence_a11_l127_127951


namespace max_value_of_z_l127_127586

variable (x y z : ℝ)

def condition1 : Prop := 2 * x + y ≤ 4
def condition2 : Prop := x ≤ y
def condition3 : Prop := x ≥ 1 / 2
def objective_function : ℝ := 2 * x - y

theorem max_value_of_z :
  (∀ x y, condition1 x y ∧ condition2 x y ∧ condition3 x → z = objective_function x y) →
  z ≤ 4 / 3 :=
sorry

end max_value_of_z_l127_127586


namespace four_planes_divide_space_into_fifteen_parts_l127_127931

-- Define the function that calculates the number of parts given the number of planes.
def parts_divided_by_planes (x : ℕ) : ℕ :=
  (x^3 + 5 * x + 6) / 6

-- Prove that four planes divide the space into 15 parts.
theorem four_planes_divide_space_into_fifteen_parts : parts_divided_by_planes 4 = 15 :=
by sorry

end four_planes_divide_space_into_fifteen_parts_l127_127931


namespace gillian_phone_bill_l127_127940

variable (original_monthly_bill : ℝ) (increase_percentage : ℝ) (months_per_year : ℕ)

def annual_phone_bill_after_increase (bill : ℝ) (increase : ℝ) (months : ℕ) : ℝ :=
  bill * (1 + increase / 100) * months

theorem gillian_phone_bill (h1 : original_monthly_bill = 50)
  (h2 : increase_percentage = 10)
  (h3 : months_per_year = 12) :
  annual_phone_bill_after_increase original_monthly_bill increase_percentage months_per_year = 660 := by
  sorry

end gillian_phone_bill_l127_127940


namespace consumer_credit_amount_l127_127479

theorem consumer_credit_amount
  (C A : ℝ)
  (h1 : A = 0.20 * C)
  (h2 : 57 = 1/3 * A) :
  C = 855 := by
  sorry

end consumer_credit_amount_l127_127479


namespace polynomial_simplification_l127_127699

theorem polynomial_simplification :
  ∃ A B C D : ℤ,
  (∀ x : ℤ, x ≠ D → (x^3 + 5 * x^2 + 8 * x + 4) / (x + 1) = A * x^2 + B * x + C)
  ∧ (A + B + C + D = 8) :=
sorry

end polynomial_simplification_l127_127699


namespace correct_operation_l127_127410

variable (x y : ℝ)

theorem correct_operation : 3 * x * y² - 4 * x * y² = -x * y² :=
by
  sorry

end correct_operation_l127_127410


namespace gillian_yearly_phone_bill_l127_127932

-- Given conditions
def usual_monthly_bill : ℝ := 50
def increase_percentage : ℝ := 0.10

-- Desired result for the yearly bill after the increase
def expected_yearly_bill : ℝ := 660

-- The theorem to prove
theorem gillian_yearly_phone_bill :
  let new_monthly_bill := usual_monthly_bill * (1 + increase_percentage) in
  let yearly_bill := new_monthly_bill * 12 in
  yearly_bill = expected_yearly_bill :=
by
  sorry

end gillian_yearly_phone_bill_l127_127932


namespace tan_neg_480_eq_sqrt_3_l127_127436

theorem tan_neg_480_eq_sqrt_3 : Real.tan (-8 * Real.pi / 3) = Real.sqrt 3 :=
by
  sorry

end tan_neg_480_eq_sqrt_3_l127_127436


namespace total_songs_l127_127175

theorem total_songs (h : ℕ) (m : ℕ) (a : ℕ) (t : ℕ) (P : ℕ)
  (Hh : h = 6) (Hm : m = 3) (Ha : a = 5) 
  (Htotal : P = (h + m + a + t) / 3) 
  (Hdiv : (h + m + a + t) % 3 = 0) : P = 6 := by
  sorry

end total_songs_l127_127175


namespace true_propositions_l127_127963

-- Definitions for non-coincident lines and pairwise non-coincident planes
variables {m n : Line} {α β γ : Plane}
variable (P : Prop)

-- Proposition 1: If m ⊥ α and m ⊥ β, then α ∥ β
def prop1 : Prop := ∀ (m : Line) (α β : Plane), 
  (m ⊥ α) → (m ⊥ β) → (α ∥ β)

-- Proposition 4: If m and n are skew lines, m ⊥ α, m ∥ β, n ⊥ β, n ∥ α, then α ⊥ β
def prop4 : Prop := ∀ (m n : Line) (α β : Plane), 
  skew m n → (m ⊥ α) → (m ∥ β) → (n ⊥ β) → (n ∥ α) → (α ⊥ β)

-- Neither of the incorrect propositions (2 and 3) should appear here

-- True propositions to be proven
theorem true_propositions : P → (prop1 m α β ∧ prop4 m n α β) := 
by sorry

end true_propositions_l127_127963


namespace proof_math_problem_l127_127625

-- Define the conditions
def inclination_angle (α : ℝ) : Prop := α = 60 * (Real.pi / 180)
def param_eq_line (x y t α : ℝ) : Prop := (x = 2 + t * Real.cos α) ∧ (y = t * Real.sin α)
def polar_eq_curve (ρ θ : ℝ) : Prop := ρ = ρ * (Real.cos θ) ^ 2 + 4 * Real.cos θ

-- Define the proof problem
theorem proof_math_problem (α t x y ρ θ : ℝ) (P : ℝ × ℝ) (A B D : ℝ × ℝ) 
  (h1 : inclination_angle α)
  (h2 : param_eq_line x y t α)
  (h3 : polar_eq_curve ρ θ)
  (hP : P = (2, 0))
  (hA : A = (x, y) ∧ y * y = 4 * x)
  (hB : B = (x, y) ∧ y * y = 4 * x)
  (hD : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  : (∃ A B D, (sqrt 3 * x - y - 2 * sqrt 3 = 0) ∧ (y * y = 4 * x) 
    ∧ |P.1 - D.1| / |P.1 - A.1| + |P.2 - D.2| / |P.2 - B.2| = sqrt 7 / 3) := sorry

end proof_math_problem_l127_127625


namespace solve_congruence_l127_127341

theorem solve_congruence (n : ℤ) (h : 11 * n ≡ 10 [MOD 43]) : n ≡ 40 [MOD 43] := 
sorry

end solve_congruence_l127_127341


namespace value_of_c_l127_127930

open scoped Real

theorem value_of_c (u v w : ℝ) (c d : ℝ) :
  8 * u^3 + 6 * c * u^2 + 3 * d * u + c = 0 ∧
  8 * v^3 + 6 * c * v^2 + 3 * d * v + c = 0 ∧
  8 * w^3 + 6 * c * w^2 + 3 * d * w + c = 0 ∧
  u ≠ v ∧ v ≠ w ∧ w ≠ u ∧
  0 < u ∧ 0 < v ∧ 0 < w ∧
  log 3 u + log 3 v + log 3 w = 5 
  → c = -1944 := by
  sorry

end value_of_c_l127_127930


namespace compare_powers_l127_127850

-- Definitions for the three numbers
def a : ℝ := 3 ^ 555
def b : ℝ := 4 ^ 444
def c : ℝ := 5 ^ 333

-- Statement to prove
theorem compare_powers : c < a ∧ a < b := sorry

end compare_powers_l127_127850


namespace minimum_time_is_21_l127_127469

-- Define the individual crossing times for A, B, C, and D
def cross_time_A : ℕ := 3
def cross_time_B : ℕ := 4
def cross_time_C : ℕ := 5
def cross_time_D : ℕ := 6

-- Define a function to calculate the minimum time for all to cross
def minimum_cross_time (A B C D : ℕ) : ℕ :=
  let t1 := A + B + A + C + D in
  let t2 := A + D + A + B + C in
  min t1 t2

-- The theorem stating the minimum crossing time is 21 minutes
theorem minimum_time_is_21 : minimum_cross_time cross_time_A cross_time_B cross_time_C cross_time_D = 21 := 
sorry

end minimum_time_is_21_l127_127469


namespace math_problem_l127_127533

noncomputable def M : ℝ → ℝ → ℝ := max
noncomputable def m : ℝ → ℝ → ℝ := min

theorem math_problem 
(a b c d e : ℝ) 
(h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
(h_order : a < c ∧ c < b ∧ b < e ∧ e < d) :
  M(m(b, m(c, d)), M(a, m(c, e))) = c :=
sorry

end math_problem_l127_127533


namespace range_of_a_l127_127604

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a * x + 5 < 0) ↔ (a < -2 * Real.sqrt 5 ∨ a > 2 * Real.sqrt 5) := 
by 
  sorry

end range_of_a_l127_127604


namespace solve_diophantine_l127_127342

theorem solve_diophantine : ∃ (x y : ℕ) (t : ℤ), x = 4 - 43 * t ∧ y = 6 - 65 * t ∧ t ≤ 0 ∧ 65 * x - 43 * y = 2 :=
by
  sorry

end solve_diophantine_l127_127342


namespace rectangle_ratio_l127_127535

theorem rectangle_ratio (s y x : ℝ)
  (h1 : 4 * y * x + s * s = 9 * s * s)
  (h2 : s + y + y = 3 * s)
  (h3 : y = s)
  (h4 : x + s = 3 * s) : 
  (x / y = 2) :=
sorry

end rectangle_ratio_l127_127535


namespace covering_ways_l127_127415

noncomputable def f : ℕ → ℕ 
| 0       := 1
| 1       := 0
| 2       := 1
| 3       := 1
| n       := f (n - 2) + f (n - 3)

theorem covering_ways (n : ℕ) : f 13 * f 13 + 1 = 257 := 
by {
  have f_13 : f 13 = 16 := by {
    sorry
  },
  rw [f_13],
  norm_num
}

end covering_ways_l127_127415


namespace compute_sum_l127_127852
-- Import the necessary library to have access to the required definitions and theorems.

-- Define the integers involved based on the conditions.
def a : ℕ := 157
def b : ℕ := 43
def c : ℕ := 19
def d : ℕ := 81

-- State the theorem that computes the sum of these integers and equate it to 300.
theorem compute_sum : a + b + c + d = 300 := by
  sorry

end compute_sum_l127_127852


namespace solve_for_a_l127_127238

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solve_for_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x / ((x + 1) * (x - a)))
  (h_odd : is_odd_function f) :
  a = 1 :=
sorry

end solve_for_a_l127_127238


namespace unique_function_l127_127280

-- Define the set T of positive real numbers
def T := {x : ℝ // x > 0}

-- The function g : T → ℝ
variable (g : T → ℝ)

-- The conditions provided in the problem
def condition1 : Prop := g ⟨2, by norm_num⟩ = 2
def condition2 : Prop := ∀ x y : T, g ⟨x.1^2 + y.1^2, by linarith⟩ = g ⟨x.1^2, by positivity⟩ + g ⟨y.1^2, by positivity⟩
def condition3 : Prop := ∀ x y : T, (x.1^2 + y.1^2) * g ⟨x.1^2 + y.1^2, by linarith⟩ = x.1^2 * y.1^2 * g ⟨x.1^2, by positivity⟩ * g ⟨y.1^2, by positivity⟩

-- The goal to prove that there is only one possible function g(x) = 4 / x
theorem unique_function : condition1 g ∧ condition2 g ∧ condition3 g → ∃! f : T → ℝ, f = λ x, 4 / x.1 :=
by
  sorry

end unique_function_l127_127280


namespace geom_sum_correct_limit_geom_sum_ratio_l127_127131

noncomputable def geometric_sum (n : ℕ) (q : ℝ) : ℝ :=
if q = 1 then 
  2 * n 
else 
  2 * (1 - q^n) / (1 - q)

noncomputable def alternating_geometric_sum (n : ℕ) (q : ℝ) : ℝ :=
2 * q * (1 - q^(2 * n)) / (1 - q^2)

theorem geom_sum_correct (n : ℕ) (q : ℝ) (h : q > 0) :
  geometric_sum n q = 
  if q = 1 then
    2 * n
  else
    2 * (1 - q^n) / (1 - q) :=
by sorry

theorem limit_geom_sum_ratio (q : ℝ) (h : q > 0) :
  ∀ (Sn Tn : ℕ → ℝ) (t : ∀ n, Sn n = geometric_sum n q ∧ Tn n = alternating_geometric_sum n q), 
  (if q = 1 then
    (∀ n, Sn n = 2 * n) ∧ (∀ n, Tn n = 2 * n) ∧ (∀ n, Sn n / Tn n = 1)
  else if q > 1 then
    ∀ n, Sn n / Tn n = (1 + q) / (q * (1 + q^n)) ∧ (∀ n, tendsto (λ n, Sn n / Tn n) at_top (𝓝 0))
  else
    ∀ n, Sn n / Tn n = (1 + q) / (q * (1 + q^n)) ∧ (∀ n, tendsto (λ n, Sn n / Tn n) at_top (𝓝 ((1 + q) / q)))) :=
by sorry

end geom_sum_correct_limit_geom_sum_ratio_l127_127131


namespace sample_size_six_l127_127766

-- Definitions for the conditions
def num_senior_teachers : ℕ := 18
def num_first_level_teachers : ℕ := 12
def num_top_level_teachers : ℕ := 6
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_top_level_teachers

-- The proof problem statement
theorem sample_size_six (n : ℕ) (h1 : n > 0) : 
  (∀ m : ℕ, m * n = total_teachers → 
             ((n + 1) * m - 1 = 35) → False) → n = 6 :=
sorry

end sample_size_six_l127_127766


namespace largest_common_factor_462_330_l127_127040

-- Define the factors of 462
def factors_462 : Set ℕ := {1, 2, 3, 6, 7, 14, 21, 33, 42, 66, 77, 154, 231, 462}

-- Define the factors of 330
def factors_330 : Set ℕ := {1, 2, 3, 5, 6, 10, 11, 15, 30, 33, 55, 66, 110, 165, 330}

-- Define the statement of the theorem
theorem largest_common_factor_462_330 : 
  (∀ d : ℕ, d ∈ (factors_462 ∩ factors_330) → d ≤ 66) ∧
  66 ∈ (factors_462 ∩ factors_330) :=
sorry

end largest_common_factor_462_330_l127_127040


namespace ellipse_equation_l127_127655

theorem ellipse_equation 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hM : M = (2, real.sqrt 2))
  (hN : N = (real.sqrt 6, 1))
  (hE : ∀ p : ℝ × ℝ, p = M ∨ p = N -> (p.1^2 / a^2 + p.2^2 / b^2 = 1)) :
  (a^2 = 8 ∧ b^2 = 4) ∧ 
  (∃ R : ℝ, R^2 = 8 / 3 ∧ R > 0 ∧ (∀ (k m : ℝ), 
    (m^2 > 2 ∧ 3*m^2 >= 8 ∧ R = m / real.sqrt(1 + k^2)) ∧ 
    (∀ A B : ℝ × ℝ, tangent_to_circle_at_origin R k m A ∧ tangent_to_circle_at_origin R k m B ∧ 
      (A ≠ B ∧ (A.1 * B.1 + A.2 * B.2 = 0) -> 
      (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) > 4 * real.sqrt 6 / 3 ∧ 
       sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) <= 2 * real.sqrt 3)))) :=
begin
  sorry
end

def tangent_to_circle_at_origin (R k m : ℝ) (P : ℝ × ℝ) : Prop :=
P.2 = k * P.1 + m ∧ P.1^2 + P.2^2 = R^2

end ellipse_equation_l127_127655


namespace complex_product_identity_l127_127051

variables {α1 α2 α3 : ℝ}
def z1 := complex.mk (real.cos α1) (real.sin α1)
def z2 := complex.mk (real.cos α2) (real.sin α2)
def z3 := complex.mk (real.cos α3) (real.sin α3)

theorem complex_product_identity :
  z1 * z2 * z3 = complex.mk (real.cos (α1 + α2 + α3)) (real.sin (α1 + α2 + α3)) :=
sorry

end complex_product_identity_l127_127051


namespace arthur_winning_strategy_l127_127300

theorem arthur_winning_strategy (n k : ℕ) (hn : 2 ≤ n) (hk : 1 ≤ k) :
  (∀ (A B : Finset ℕ) (hA : A.card = n) (hB : B.card = n), 
    ∃ (x : Fin k → ℕ), 
      (∀ (i : Fin k), ∃ (a ∈ A) (b ∈ B), x i = a * b) ∧
      (∀ (i1 i2 i3 : Fin k), i1 ≠ i2 → i1 ≠ i3 → i2 ≠ i3 → ¬ (x i1 ∣ x i2 * x i3))) ↔
  k ≤ 2 * n - 2 :=
by
  sorry

end arthur_winning_strategy_l127_127300


namespace problem_sol_l127_127973

theorem problem_sol (a b : ℝ) (h : ∀ x, (x > -1 ∧ x < 1/3) ↔ (ax^2 + bx + 1 > 0)) : a * b = 6 :=
sorry

end problem_sol_l127_127973


namespace Laplace_transform_ratio_l127_127281

variables {α : Type*} [MeasureSpace α] {X : ℕ → α → ℝ}
variable  {f : ℝ → ℝ}
variable  (λ : ℝ) {n : ℕ} {x y : ℝ}

-- Random variables X_i are independent and identically distributed
axiom iid_nonneg_random_vars (X : ℕ → α → ℝ) (f : ℝ → ℝ) : (∀ i, MeasureTheory.integrable (X i) ∧ ∀ i, HasPdf (λ a, f (X i a)) (f i)) ∧ ∀ i, 0 ≤ X i := sorry

-- Definitions of Sn and Mn
noncomputable def S_n := (finset.range n).sum (λ i, X i)
noncomputable def M_n := finset.max (finset.range n) (λ i, X i)

-- Definition of φ_n, the Laplace transform of S_n / M_n
noncomputable def φ_n (λ : ℝ) : ℝ :=
  n * real.exp (-λ) * ∫ (x : ℝ) in set.Ici 0, (∫ (y : ℝ) in set.Ioc 0 x, real.exp (-λ * y / x) * f y) ^ (n - 1) * f x

-- Goal statement
theorem Laplace_transform_ratio (h : iid_nonneg_random_vars X f) :
  φ_n λ = φ_n_le := sorry

end Laplace_transform_ratio_l127_127281


namespace total_sides_on_cookie_cutters_l127_127896

theorem total_sides_on_cookie_cutters :
  let top_triangles := 6 * 3,
      top_nonagon := 1 * 9,
      top_heptagons := 2 * 7,
      middle_squares := 4 * 4,
      middle_hexagons := 2 * 6,
      middle_hendecagon := 1 * 11,
      bottom_octagons := 3 * 8,
      bottom_circles := 5 * 0,
      bottom_pentagon := 1 * 5,
      bottom_nonagon := 1 * 9
  in top_triangles + top_nonagon + top_heptagons +
     middle_squares + middle_hexagons + middle_hendecagon +
     bottom_octagons + bottom_circles + bottom_pentagon + bottom_nonagon = 118 := by
  sorry

end total_sides_on_cookie_cutters_l127_127896


namespace sum_of_first_2015_terms_l127_127254

variable {a : ℕ → ℝ} -- the arithmetic sequence a_n
variable n : ℕ -- specific term index

-- Condition: a_n is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Condition: a_{1007} + a_{1008} + a_{1009} = 18
def condition_sum_terms : Prop :=
  a 1007 + a 1008 + a 1009 = 18

-- Question: Sum of the first 2015 terms is 12090
def sum_first_n (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n : ℝ) * (a 0 + a (n - 1)) / 2

theorem sum_of_first_2015_terms
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_1007_1008_1009 : condition_sum_terms) :
  sum_first_n 2015 a = 12090 :=
sorry

end sum_of_first_2015_terms_l127_127254


namespace part1_part2_l127_127758

namespace TriangleProblems

noncomputable def area_triangle_part1 (A C a : ℝ) : ℝ :=
  if A = 30 ∧ C = 45 ∧ a = 2 then 1 + real.sqrt 3 else 0

theorem part1 (A C a : ℝ) : 
  A = 30 → C = 45 → a = 2 → area_triangle_part1 A C a = 1 + real.sqrt 3 :=
begin
  intros hA hC ha,
  simp [area_triangle_part1, hA, hC, ha]
end

noncomputable def length_AB_part2 (area BC C : ℝ) : ℝ :=
  if area = real.sqrt 3 ∧ BC = 2 ∧ C = 60 then 2 else 0

theorem part2 (area BC C : ℝ) :
  area = real.sqrt 3 → BC = 2 → C = 60 → length_AB_part2 area BC C = 2 :=
begin
  intros hArea hBC hC,
  simp [length_AB_part2, hArea, hBC, hC]
end

end TriangleProblems

end part1_part2_l127_127758


namespace joanne_gave_coins_l127_127512

theorem joanne_gave_coins
  (h1 : ∃ a, a = 15)  -- Joanne collected 15 coins in the first hour
  (h2 : ∃ b, b = 70)  -- Joanne collected 35 coins over the next two hours (35 * 2 = 70)
  (h3 : ∃ c, c = 50)  -- Joanne collected 50 coins in the fourth hour
  (h4 : ∃ d, d = 120) -- Joanne had 120 coins after giving some to coworker
  : ∃ x, x = (15 + 70 + 50 - 120) :=   -- Joanne gave 15 coins to her coworker
by
  use (15 + 70 + 50 - 120)
  norm_num
  sorry

end joanne_gave_coins_l127_127512


namespace part1_part2_l127_127987

noncomputable def f (x a : ℝ) : ℝ := real.exp x - (1 / 2) * a * x^2

theorem part1 (x : ℝ) (h1 : 0 < x) : 
  (f x 2) > 1 := 
by
  sorry

theorem part2 :
  ∃ (a : ℕ), (∀ (x : ℝ), 0 < x → (real.exp x - a * x) ≥ x^2 * real.log x) ∧ 
  (∀ (b : ℕ), (∀ (x : ℝ), 0 < x → (real.exp x - b * x) ≥ x^2 * real.log x) → b ≤ 2) :=
by
  sorry

end part1_part2_l127_127987


namespace sum_a6_a7_a8_is_32_l127_127287

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l127_127287


namespace exists_positive_a_no_positive_rational_a_l127_127749

-- Part (a)
theorem exists_positive_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  ∃ n : ℕ, divides_into_two_piles a n = true :=
sorry

-- Part (b)
theorem no_positive_rational_a (m n : ℕ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : gcd m n = 1) (h₄ : m ≠ n) :
  ¬ ∃ k : ℕ, divides_into_two_piles (m / n : ℚ) k = true :=
sorry

end exists_positive_a_no_positive_rational_a_l127_127749


namespace count_polynomials_l127_127226

open Nat Polynomial

theorem count_polynomials : 
  let P : Polynomial ℤ := a_5 * X^5 + a_4 * X^4 + a_3 * X^3 + a_2 * X^2 + a_1 * X + a_0
  ∀ x ∈ {0, 1, 2, 3, 4, 5}, (0 ≤ eval x P ∧ eval x P < 120) →
  ∃! n, n = (Nat.factorial 125) / ((Nat.factorial 5) * (Nat.factorial 120)) :=
by 
  sorry

end count_polynomials_l127_127226


namespace password_combinations_check_l127_127317

theorem password_combinations_check : ∃ (s : Multiset Char), Multiset.card s = 5 ∧ (Multiset.perm s).card = 20 := by
  sorry

end password_combinations_check_l127_127317


namespace monopoly_favor_durable_machine_competitive_market_prefer_durable_l127_127754

-- Define the conditions
def consumer_valuation : ℕ := 10
def durable_cost : ℕ := 6

-- Define the monopoly decision problem: prove C > 3
theorem monopoly_favor_durable_machine (C : ℕ) : 
  consumer_valuation * 2 - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

-- Define the competitive market decision problem: prove C > 3
theorem competitive_market_prefer_durable (C : ℕ) :
  2 * consumer_valuation - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

end monopoly_favor_durable_machine_competitive_market_prefer_durable_l127_127754


namespace relationship_among_a_b_c_l127_127539

noncomputable def a := Real.logBase 3 (1 / 2)
noncomputable def b := Real.logBase 3 9.1
noncomputable def c := 2 ^ 0.8

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  sorry

end relationship_among_a_b_c_l127_127539


namespace triangle_PQ_length_l127_127033

theorem triangle_PQ_length (RP PQ : ℝ) (n : ℕ) (h_rp : RP = 2.4) (h_n : n = 25) : RP = 2.4 → PQ = 3 := by
  sorry

end triangle_PQ_length_l127_127033


namespace alcohol_water_ratio_l127_127035

theorem alcohol_water_ratio (V : ℝ) (hV_pos : V > 0) :
  let jar1_alcohol := (2 / 3) * V
  let jar1_water := (1 / 3) * V
  let jar2_alcohol := (3 / 2) * V
  let jar2_water := (1 / 2) * V
  let total_alcohol := jar1_alcohol + jar2_alcohol
  let total_water := jar1_water + jar2_water
  (total_alcohol / total_water) = (13 / 5) :=
by
  -- Placeholder for the proof
  sorry

end alcohol_water_ratio_l127_127035


namespace amounts_are_correct_l127_127306

theorem amounts_are_correct (P Q R S : ℕ) 
    (h1 : P + Q + R + S = 10000)
    (h2 : R = 2 * P)
    (h3 : R = 3 * Q)
    (h4 : S = P + Q) :
    P = 1875 ∧ Q = 1250 ∧ R = 3750 ∧ S = 3125 := by
  sorry

end amounts_are_correct_l127_127306


namespace first_group_men_l127_127688

theorem first_group_men (M : ℕ) :
  M * 25 = 20 * 18.75 → M = 15 := by
  sorry

end first_group_men_l127_127688


namespace positive_integers_solution_l127_127898

open Nat

theorem positive_integers_solution (a b m n : ℕ) (r : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h_gcd : Nat.gcd m n = 1) :
  (a^2 + b^2)^m = (a * b)^n ↔ a = 2^r ∧ b = 2^r ∧ m = 2 * r ∧ n = 2 * r + 1 :=
sorry

end positive_integers_solution_l127_127898


namespace find_positives_xyz_l127_127520

theorem find_positives_xyz (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0)
    (heq : (1 : ℚ)/x + (1 : ℚ)/y + (1 : ℚ)/z = 4 / 5) :
    (x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 5 ∧ z = 10) :=
by
  sorry

-- This theorem states that there are only two sets of positive integers (x, y, z)
-- that satisfy the equation (1/x) + (1/y) + (1/z) = 4/5, specifically:
-- (2, 4, 20) and (2, 5, 10).

end find_positives_xyz_l127_127520


namespace units_digit_of_42_pow_3_add_24_pow_3_l127_127406

theorem units_digit_of_42_pow_3_add_24_pow_3 :
    (42 ^ 3 + 24 ^ 3) % 10 = 2 :=
by
    have units_digit_42 := (42 % 10 = 2)
    have units_digit_24 := (24 % 10 = 4)
    sorry

end units_digit_of_42_pow_3_add_24_pow_3_l127_127406


namespace max_volume_is_40_l127_127393

noncomputable def max_volume (V: ℝ) : Prop :=
  let first_dilution := (V - 10) / V
  let second_dilution := (V - 10 - (8 * (V - 10) / V)) / V
  second_dilution ≤ 0.6

theorem max_volume_is_40 : ∃ V: ℝ, max_volume V ∧ V = 40 :=
begin
  use 40,
  unfold max_volume,
  dsimp,
  sorry
end

end max_volume_is_40_l127_127393


namespace find_a6_plus_a7_plus_a8_l127_127289

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l127_127289


namespace gillian_phone_bill_next_year_l127_127935

theorem gillian_phone_bill_next_year (monthly_bill : ℝ) (increase_percentage : ℝ) (months_in_year : ℕ) :
  monthly_bill = 50 → increase_percentage = 0.10 → months_in_year = 12 → 
  (monthly_bill * (1 + increase_percentage)) * (months_in_year) = 660 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  linarith

end gillian_phone_bill_next_year_l127_127935


namespace least_number_to_add_l127_127407

theorem least_number_to_add (x : ℕ) (h : 53 ∣ x ∧ 71 ∣ x) : 
  ∃ n : ℕ, x = 1357 + n ∧ n = 2406 :=
by sorry

end least_number_to_add_l127_127407


namespace find_amplitude_l127_127123

-- Define the necessary constants and conditions
constant a b c d : ℝ
constant h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

-- Define the function
def y (x : ℝ) : ℝ := a * sin (b * x + c) + d

-- Define the conditions for maximum and minimum values of the function
constant h_max : ∀ x, y x ≤ y (0) → y (0) = 5
constant h_min : ∀ x, y x ≥ y (0) → y (0) = -3

-- The theorem to prove
theorem find_amplitude : a = 4 :=
by
  sorry

end find_amplitude_l127_127123


namespace binomial_coefficient_x4_sum_3125_l127_127975

theorem binomial_coefficient_x4_sum_3125 :
  (∑ i in finset.range (5 + 1), (3:ℤ)^i * (2:ℤ)^(5 - i) * nat.choose 5 i) = 3125 ->
  nat.choose 5 1 * 3^4 * 2 = 810 :=
by
  sorry

end binomial_coefficient_x4_sum_3125_l127_127975


namespace second_boy_marbles_correct_l127_127724

-- Define the conditions
def first_boy_marbles (x : ℚ) : ℚ := 4 * x + 2
def second_boy_marbles (x : ℚ) : ℚ := 2 * x - 1
def third_boy_marbles (x : ℚ) : ℚ := 3 * x + 3

theorem second_boy_marbles_correct :
  ∀ (x : ℚ), first_boy_marbles x + second_boy_marbles x + third_boy_marbles x = 100 →
             second_boy_marbles x = 61 / 3 :=
by {
  intros,
  sorry
}

end second_boy_marbles_correct_l127_127724


namespace cover_n_plus_1_points_l127_127736

theorem cover_n_plus_1_points (n : ℕ) (points : Fin (2 * n + 1) → ℝ × ℝ)
    (H : ∀ i : Fin (2 * n + 1), ∀ j : Fin (2 * n + 1), ∀ k : Fin (2 * n + 1),
        i ≠ j → j ≠ k → i ≠ k →
        (dist (points i) (points j) < 1 ∨ dist (points j) (points k) < 1 ∨ dist (points k) (points i) < 1)) :
    ∃ (center : ℝ × ℝ), ∃ (S : Finset (Fin (2 * n + 1))), S.card = n + 1 ∧
    ∀ i ∈ S, dist (points i) center ≤ 1 :=
begin
  sorry
end

end cover_n_plus_1_points_l127_127736


namespace more_bottles_of_regular_soda_l127_127775

theorem more_bottles_of_regular_soda (reg_soda diet_soda : ℕ) (h1 : reg_soda = 79) (h2 : diet_soda = 53) :
  reg_soda - diet_soda = 26 :=
by
  sorry

end more_bottles_of_regular_soda_l127_127775


namespace positive_multiples_of_6_l127_127748

theorem positive_multiples_of_6 (k a b : ℕ) (h₁ : a = (3 + 3 * k))
  (h₂ : b = 24) (h₃ : a^2 - b^2 = 0) : k = 7 :=
sorry

end positive_multiples_of_6_l127_127748


namespace equal_playing_time_l127_127815

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l127_127815


namespace tom_gave_jessica_some_seashells_l127_127030

theorem tom_gave_jessica_some_seashells
  (original_seashells : ℕ := 5)
  (current_seashells : ℕ := 3) :
  original_seashells - current_seashells = 2 :=
by
  sorry

end tom_gave_jessica_some_seashells_l127_127030


namespace equal_play_time_l127_127820

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l127_127820


namespace wine_problem_l127_127619

theorem wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + (1 / 3) * y = 33) : x + y = 19 ∧ 3 * x + (1 / 3) * y = 33 :=
by
  sorry

end wine_problem_l127_127619


namespace three_classes_same_team_l127_127477

open Function

theorem three_classes_same_team
  (students : Finset ℕ)
  (classes : Finset ℕ)
  (teams : Fin ℕ → ℕ)
  (h_students : students.card = 30)
  (h_classes_three : classes.card = 3)
  (h_team_division : ∀ c ∈ classes, (teams c = 1) ∨ (teams c = 2) ∨ (teams c = 3))
  (h_team_size : ∀ c ∈ classes, ∀ t, (students.filter (λ s, teams s = t)).card = 10) :
  ∃ (s1 s2 : ℕ), s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ 
  ∀ c ∈ classes, teams s1 = teams s2 :=
sorry

end three_classes_same_team_l127_127477


namespace ball_radius_l127_127735

-- Definitions for the given conditions
def shadow_length (ball_shadow farthest_point : ℝ) := ball_shadow = 10
def ruler_shadow (ruler_shadow_length : ℝ) := ruler_shadow_length = 2

-- The main statement to be proven based on stated conditions
theorem ball_radius
  (ball_shadow farthest_point ruler_shadow_length : ℝ)
  (h1 : shadow_length ball_shadow farthest_point)
  (h2 : ruler_shadow ruler_shadow_length) :
    ∃ r : ℝ, r = 10 * sqrt 5 - 20 :=
by sorry

end ball_radius_l127_127735


namespace total_spent_proof_l127_127889

noncomputable def total_spent (cost_pen cost_pencil cost_notebook : ℝ) 
  (pens_robert pencils_robert notebooks_dorothy : ℕ) 
  (julia_pens_ratio robert_pens_ratio dorothy_pens_ratio : ℝ) 
  (julia_pencils_diff notebooks_julia_diff : ℕ) 
  (robert_notebooks_ratio dorothy_pencils_ratio : ℝ) : ℝ :=
    let pens_julia := robert_pens_ratio * pens_robert
    let pens_dorothy := dorothy_pens_ratio * pens_julia
    let total_pens := pens_robert + pens_julia + pens_dorothy
    let cost_pens := total_pens * cost_pen 
    
    let pencils_julia := pencils_robert - julia_pencils_diff
    let pencils_dorothy := dorothy_pencils_ratio * pencils_julia
    let total_pencils := pencils_robert + pencils_julia + pencils_dorothy
    let cost_pencils := total_pencils * cost_pencil 
        
    let notebooks_julia := notebooks_dorothy + notebooks_julia_diff
    let notebooks_robert := robert_notebooks_ratio * notebooks_julia
    let total_notebooks := notebooks_dorothy + notebooks_julia + notebooks_robert
    let cost_notebooks := total_notebooks * cost_notebook
        
    cost_pens + cost_pencils + cost_notebooks

theorem total_spent_proof 
  (cost_pen : ℝ := 1.50)
  (cost_pencil : ℝ := 0.75)
  (cost_notebook : ℝ := 4.00)
  (pens_robert : ℕ := 4)
  (pencils_robert : ℕ := 12)
  (notebooks_dorothy : ℕ := 3)
  (julia_pens_ratio : ℝ := 3)
  (robert_pens_ratio : ℝ := 3)
  (dorothy_pens_ratio : ℝ := 0.5)
  (julia_pencils_diff : ℕ := 5)
  (notebooks_julia_diff : ℕ := 1)
  (robert_notebooks_ratio : ℝ := 0.5)
  (dorothy_pencils_ratio : ℝ := 2) : 
  total_spent cost_pen cost_pencil cost_notebook pens_robert pencils_robert notebooks_dorothy 
    julia_pens_ratio robert_pens_ratio dorothy_pens_ratio julia_pencils_diff notebooks_julia_diff robert_notebooks_ratio dorothy_pencils_ratio 
    = 93.75 := 
by 
  sorry

end total_spent_proof_l127_127889


namespace locus_of_point_P_l127_127978

variable {x y m n : ℝ}

-- Condition 1: The given ellipse
def ellipse (x y : ℝ) : Prop := (x^2) / 20 + (y^2) / 16 = 1

-- Condition 2: Points of intersection with the x-axis
def point_A := (-2 * Real.sqrt 5, 0)
def point_B := (2 * Real.sqrt 5, 0)

-- Condition 3: Movable point M on the ellipse, not coinciding with A or B
def point_M (m n : ℝ) : Prop := ellipse m n ∧ m ≠ -2 * Real.sqrt 5 ∧ m ≠ 2 * Real.sqrt 5

-- Condition 4: Tangent line at M
def tangent_line (m n x y : ℝ) : Prop := (m * x) / 20 + (n * y) / 16 = 1

-- Condition 5: Intersection points C and D found from vertical lines through A and B intersecting the tangent line

-- This function finds y for given x on the tangent line
def tangent_y (m n x : ℝ) : ℝ := 16 / n + (8 * Real.sqrt 5 * m) / (5 * n) * x

def point_C (m n : ℝ) : ℝ × ℝ := (-2 * Real.sqrt 5, tangent_y m n (-2 * Real.sqrt 5))
def point_D (m n : ℝ) : ℝ × ℝ := (2 * Real.sqrt 5, tangent_y m n (2 * Real.sqrt 5))

-- Slope functions for lines AD and BC are not strictly needed, so they are skipped in the conditions

-- Intersection point Q
def point_Q (m n : ℝ) : ℝ × ℝ := (m, (40 - 2 * m^2) / (5 * n))

-- Point P is symmetric to Q concerning M
def point_P (m n : ℝ) : ℝ × ℝ := (m, (2 * m^2 + 10 * n^2 - 40) / (5 * n))

-- A proof that given the conditions, the locus of P is the ellipse with given equation
theorem locus_of_point_P : ∀ (m n : ℝ), 
  point_M m n →
  ∃ x y : ℝ, point_P m n = (x, y) ∧ (y^2) / 36 + (x^2) / 20 = 1 :=
sorry

end locus_of_point_P_l127_127978


namespace num_common_points_of_three_lines_l127_127391

def three_planes {P : Type} [AddCommGroup P] (l1 l2 l3 : Set P) : Prop :=
  let p12 := Set.univ \ (l1 ∪ l2)
  let p13 := Set.univ \ (l1 ∪ l3)
  let p23 := Set.univ \ (l2 ∪ l3)
  ∃ (pl12 pl13 pl23 : Set P), 
    p12 = pl12 ∧ p13 = pl13 ∧ p23 = pl23

theorem num_common_points_of_three_lines (l1 l2 l3 : Set ℝ) 
  (h : three_planes l1 l2 l3) : ∃ n : ℕ, n = 0 ∨ n = 1 := by
  sorry

end num_common_points_of_three_lines_l127_127391


namespace tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l127_127329

open Real

theorem tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence (α β γ : ℝ) 
  (h1 : α + β + γ = π)  -- Assuming α, β, γ are angles in a triangle
  (h2 : tan α + tan γ = 2 * tan β) :
  sin (2 * α) + sin (2 * γ) = 2 * sin (2 * β) :=
by
  sorry

end tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l127_127329


namespace password_combinations_check_l127_127318

theorem password_combinations_check : ∃ (s : Multiset Char), Multiset.card s = 5 ∧ (Multiset.perm s).card = 20 := by
  sorry

end password_combinations_check_l127_127318


namespace customers_in_other_countries_l127_127765

-- Define the given conditions

def total_customers : ℕ := 7422
def customers_us : ℕ := 723

theorem customers_in_other_countries : total_customers - customers_us = 6699 :=
by
  -- This part will contain the proof, which is not required for this task.
  sorry

end customers_in_other_countries_l127_127765


namespace SetC_Satisfies_Conditions_l127_127113

-- Definitions for problem conditions and propositions
def P := ∀ (a b : ℝ), a * b > 0 → 0 < real.angle ⟨a, b⟩ ⟨0, 0⟩ ∧ 0 < real.angle ⟨a, b⟩ ⟨b, a⟩
def Q := ∀ (a x : ℝ), a < -1 → a^2 * x^2 - 2 * x + 1 > 0

-- Theorem statement matching conditions
theorem SetC_Satisfies_Conditions : ¬ P ∧ Q :=
begin
  sorry
end

end SetC_Satisfies_Conditions_l127_127113


namespace sum_b_n_l127_127197

-- Definitions and conditions of the problem
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := a n + 2^(a n)

def S (n : ℕ) : ℕ := n * (a 1 + a n) / 2

lemma geom_seq (n : ℕ) (h : 0 < a 2 ∧ 0 < S 3) :
  let d := a 2 - a 1 in
  a 1 = 1 ∧ d > 0 ∧ (d * d + 2 * d + 1 = 3 + 3 * d)
  := sorry

-- Proving the sum of the first n terms of b_n
theorem sum_b_n (n : ℕ) : 
  ∑ k in Finset.range n, b (k + 1) = (2^(2*n+1) / 3) + n^2 - (2 / 3)
  := sorry

end sum_b_n_l127_127197


namespace bob_total_profit_l127_127844

/-- Define the cost of each dog --/
def dog_cost : ℝ := 250.0

/-- Define the number of dogs Bob bought --/
def number_of_dogs : ℕ := 2

/-- Define the total cost of the dogs --/
def total_cost_for_dogs : ℝ := dog_cost * number_of_dogs

/-- Define the selling price of each puppy --/
def puppy_selling_price : ℝ := 350.0

/-- Define the number of puppies --/
def number_of_puppies : ℕ := 6

/-- Define the total revenue from selling the puppies --/
def total_revenue_from_puppies : ℝ := puppy_selling_price * number_of_puppies

/-- Define Bob's total profit from selling the puppies --/
def total_profit : ℝ := total_revenue_from_puppies - total_cost_for_dogs

/-- The theorem stating that Bob's total profit is $1600.00 --/
theorem bob_total_profit : total_profit = 1600.0 := 
by
  /- We leave the proof out as we just need the statement -/
  sorry

end bob_total_profit_l127_127844


namespace unique_inverse_function_l127_127239

theorem unique_inverse_function 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : ∀ x y : ℝ, (y = f x) ↔ (x = a ^ y))
  (h4 : f 2 = 1) :
  f = (λ x, log 2 x) :=
by 
  sorry

end unique_inverse_function_l127_127239


namespace books_loaned_out_l127_127055

theorem books_loaned_out (initial_books loaned_books returned_percentage end_books missing_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : end_books = 66)
  (h3 : returned_percentage = 70)
  (h4 : initial_books - end_books = missing_books)
  (h5 : missing_books = (loaned_books * (100 - returned_percentage)) / 100):
  loaned_books = 30 :=
by
  sorry

end books_loaned_out_l127_127055


namespace limit_of_n_bn_l127_127134

def L (x : ℝ) : ℝ :=
  x - (x^2) / 2

def L_iterate (L : ℝ → ℝ) : ℕ → ℝ → ℝ
| 0, x => x
| (n+1), x => L (L_iterate L n x)

def b_n (n : ℕ) : ℝ :=
  L_iterate L n (20 / n)

-- Theorem that needs to be proven
theorem limit_of_n_bn : 
  ¬ ∀ n : ℕ, n > 0 → tendsto (λ n : ℕ, n * b_n n) at_top (𝓝 (40 / 21)) :=
by
  sorry

end limit_of_n_bn_l127_127134


namespace number_of_non_neg_real_values_of_x_l127_127917

noncomputable def numNonNegRealValues (x : ℝ) : ℕ :=
  ∑ k in Finset.range 14, if (169 - k^2 : ℝ) ^ 3 ≥ 0 then 1 else 0

theorem number_of_non_neg_real_values_of_x :
  numNonNegRealValues 169 = 14 :=
sorry

end number_of_non_neg_real_values_of_x_l127_127917


namespace Basel_series_l127_127121

theorem Basel_series :
  (∑' (n : ℕ+), 1 / (n : ℝ)^2) = π^2 / 6 := by sorry

end Basel_series_l127_127121


namespace length_of_AB_l127_127379

theorem length_of_AB {L : ℝ} (h : 9 * Real.pi * L + 36 * Real.pi = 216 * Real.pi) : L = 20 :=
sorry

end length_of_AB_l127_127379


namespace convex_ngon_contains_points_l127_127451

theorem convex_ngon_contains_points :
  ∀ (convex_2000_gon : fin 2000 → ℝ × ℝ), 
  ∃ (selected_points : fin 1998 → ℝ × ℝ),
  ∀ (a b c : fin 2000), a ≠ b → a ≠ c → b ≠ c →
  let triangle_contains_point (tri : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (p : ℝ × ℝ) :=
    let (v1, v2, v3) := tri in
    let (x1, y1) := v1 in
    let (x2, y2) := v2 in
    let (x3, y3) := v3 in
    let (px, py) := p in
    let _det := (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) in
    let (_alpha := ((py - y1) * (x3 - x1) - (px - x1) * (y3 - y1)) / _det) in
    let (_beta := ((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)) / -_det) in
    0 < _alpha ∧ 0 < _beta ∧ (_alpha + _beta) < 1
  in 
  ∃ (i : fin 1998), 
  triangle_contains_point (
    (convex_2000_gon a, convex_2000_gon b, convex_2000_gon c))
    (selected_points i) ∧
  ∀ (j : fin 1998),
  j ≠ i → ¬(triangle_contains_point (
    (convex_2000_gon a, convex_2000_gon b, convex_2000_gon c))
    (selected_points j)) :=
sorry

end convex_ngon_contains_points_l127_127451


namespace find_beta_find_sin_beta_minus_alpha_l127_127542

variable (α β : ℝ)
variable (a b c : ℝ × ℝ)
variable (θ1 θ2 : ℝ)

-- Conditions
axiom h1 : 0 < α ∧ α < π / 4
axiom h2 : π / 4 < β ∧ β < π / 2
axiom h3 : a = (2 * Real.cos α ^ 2, 2 * Real.sin α * Real.cos α)
axiom h4 : b = (1 + Real.sin β * Real.cos β, 1 - 2 * Real.sin β ^ 2)
axiom h5 : c = (1, 0)
axiom h6 : θ1 = Real.inner a c
axiom h7 : θ2 = Real.inner b c

-- Proof problem 1: Find β if θ2 = π / 6
theorem find_beta (h8 : θ2 = π / 6) : β = 5 * π / 12 :=
sorry

-- Proof problem 2: Find sin(β - α) if θ2 - θ1 = π / 6
theorem find_sin_beta_minus_alpha (h9 : θ2 - θ1 = π / 6) : Real.sin (β - α) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
sorry

end find_beta_find_sin_beta_minus_alpha_l127_127542


namespace eval_sqrt_log_exp_l127_127151

theorem eval_sqrt_log_exp : sqrt (2 ^ log 2 9 + 3 ^ log 3 8) = sqrt 17 := by
  have h₁ : 2 ^ log 2 9 = 9 := by
    apply pow_log at _ 
    -- numerical substitution
    ring
  have h₂ : 3 ^ log 3 8 = 8 := by
    apply pow_log at _ 
    -- numerical substitution
    ring
  calc
    sqrt (2 ^ log 2 9 + 3 ^ log 3 8)
        = sqrt (9 + 8)     : by rw [h₁, h₂]
    ... = sqrt 17          : by ring

end eval_sqrt_log_exp_l127_127151


namespace f_n_alpha_l127_127642

noncomputable def f : ℝ → ℝ := sorry

variables (α : ℝ) (hα : α^3 - α = 33^1992) 
          (hfα : f(α)^3 - f(α) = 33^1992) (n : ℕ)

lemma polynomial_rational_coefficients (x : ℝ) : Prop := sorry  -- Placeholder for the definition

theorem f_n_alpha (n : ℕ) (h : n ≥ 1) : (f^[n] α)^3 - f^[n] α = 33^1992 :=
begin
  induction n with k hk,
  { exfalso, linarith, },
  { cases k,
    { rw [function.iterate_one], exact hfα, },
    { rw [function.iterate_succ],
      calc (f^[k.succ] α)^3 - f^[k.succ] α = (f (f^[k] α))^3 - f (f^[k] α) : sorry
      ... = (f α)^3 - f α : sorry
      ... = 33^1992 : hfα,
      sorry,
    },
  },
end

end f_n_alpha_l127_127642


namespace pentagon_side_length_l127_127706

theorem pentagon_side_length:
  ∃ s: ℝ, s = 4 * real.sqrt(1 + 0.4 * real.sqrt 5) ∧
  5 * s = (1 / 4) * real.sqrt (5 * (5 + 2 * real.sqrt 5)) * s ^ 2 :=
sorry

end pentagon_side_length_l127_127706


namespace values_of_a_and_b_l127_127698

theorem values_of_a_and_b (a b : ℝ) (h_extremum : ∀ x, ∃ y, f x = y = 10 ∧ (x = 1 → f' 1 = 0)) :
  (a = 3 ∧ b = -3) ∨ (a = -4 ∧ b = 11) :=
by {
  let f := λ x : ℝ , x^3 - a*x^2 - b*x + a^2,
  let f' := λ x : ℝ ,3*x^2 - 2*a*x - b,
  have h_deriv_1 : f'(1) = 0,
    from h_extremum 1,
  have h_value_1 : f(1) = 10,
    from h_extremum 1,
  sorry
}

end values_of_a_and_b_l127_127698


namespace strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l127_127605

def f (x : ℝ) : ℝ := -x * abs x + 2 * x

theorem strictly_increasing : ∃ A : Set ℝ, A = (Set.Ioo 0 1) ∧ (∀ x y, x ∈ A → y ∈ A → x < y → f x < f y) :=
  sorry

theorem not_gamma_interval : ¬(Set.Icc (1/2) (3/2) ⊆ Set.Ioo 0 1 ∧ 
  (∀ x ∈ Set.Icc (1/2) (3/2), f x ∈ Set.Icc (1/(3/2)) (1/(1/2)))) :=
  sorry

theorem gamma_interval_within_one_inf : ∃ m n : ℝ, 1 ≤ m ∧ m < n ∧ 
  Set.Icc m n = Set.Icc 1 ((1 + Real.sqrt 5) / 2) ∧ 
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (1/n) (1/m)) :=
  sorry

end strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l127_127605


namespace equal_play_time_l127_127801

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l127_127801


namespace f_of_f_of_minus_two_l127_127943

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - real.sqrt x else 2^x

theorem f_of_f_of_minus_two : f (f (-2)) = 1 / 2 :=
  sorry

end f_of_f_of_minus_two_l127_127943


namespace prove_region_area_relationship_l127_127486

noncomputable def region_area_relationship (C D : ℝ) (side_length : ℝ) : Prop :=
  (side_length = 3) → (C = 9 * Real.pi) → (D = Real.pi * ((3 * Real.csc (Real.pi / 8))^2 - (3 * Real.cot (Real.pi / 8))^2)) → C = (4 / 3) * D

theorem prove_region_area_relationship {C D : ℝ} :
  region_area_relationship C D 3 :=
by {
  sorry
}

end prove_region_area_relationship_l127_127486


namespace log_property_l127_127961

theorem log_property (a b : ℝ) (ha : a > 2) (hb : b > 2)
  (h : (1/2) * log 2 (a + b) + log 2 (sqrt 2 / a) = (1/2) * log 2 (1 / (a + b)) + log 2 (b / sqrt 2)) :
  log 2 (a - 2) + log 2 (b - 2) = 2 :=
sorry

end log_property_l127_127961


namespace total_is_83_l127_127622

def number_of_pirates := 45
def number_of_noodles := number_of_pirates - 7
def total_number_of_noodles_and_pirates := number_of_noodles + number_of_pirates

theorem total_is_83 : total_number_of_noodles_and_pirates = 83 := by
  sorry

end total_is_83_l127_127622


namespace moses_percentage_l127_127728

theorem moses_percentage (P : ℝ) (T : ℝ) (E : ℝ) (total_amount : ℝ) (moses_more : ℝ)
  (h1 : total_amount = 50)
  (h2 : moses_more = 5)
  (h3 : T = E)
  (h4 : P / 100 * total_amount = E + moses_more)
  (h5 : 2 * E = (1 - P / 100) * total_amount) :
  P = 40 :=
by
  sorry

end moses_percentage_l127_127728


namespace parabola_shift_left_l127_127008

theorem parabola_shift_left (x : ℝ) :
  (λ x, x^2) (x + 2) = (x + 2)^2 := by
  sorry

end parabola_shift_left_l127_127008


namespace tangent_line_equation_l127_127983

/-- Definition of the function f(x) --/
def f (x : ℝ) : ℝ := x^3 + x - 16

/-- The point of interest (2, -6) --/
def point_of_interest : ℝ × ℝ := (2, -6)

/-- The derivative of f(x) --/
def f_prime (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The equation of the tangent line at the point (2, -6) --/
theorem tangent_line_equation :
  let (x0, y0) := point_of_interest in
  (y0 = f x0) ∧ (f_prime x0 = 13) →
  ∃ b : ℝ, y = 13 * x + b ∧ b = -32 :=
by
  sorry

end tangent_line_equation_l127_127983


namespace angle_A_condition_bc_range_l127_127968

variables {A B C : ℝ} {a b c : ℝ}

noncomputable def is_acute_triangle := (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ (A + B + C = π) ∧ (A < π / 2) ∧ (B < π / 2) ∧ (C < π / 2)

axiom cosine_double_half_angle (B : ℝ) : cos^2 (B / 2) = (1 + cos B) / 2

theorem angle_A_condition (h : is_acute_triangle) (h_eq : (b - 2 * c) * cos A = a - 2 * a * cos^2 (B / 2)) : 
  A = π / 3 :=
begin
  sorry
end

theorem bc_range (h : is_acute_triangle) (h_A : A = π / 3) (h_a : a = sqrt 3) : 
  3 < b + c ∧ b + c ≤ 2 * sqrt 3 :=
begin
  sorry
end

end angle_A_condition_bc_range_l127_127968


namespace translate_parabola_l127_127396

theorem translate_parabola (x : ℝ) :
  (∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ ∀ x: ℝ, y = 2*x^2 → y = 2*(x - h)^2 + k) := 
by
  use 1, 3
  sorry

end translate_parabola_l127_127396


namespace option_B_correct_option_C_correct_l127_127578

universe u
variables {α : Type u} [noncomputable_instance : topological_space α]

def U := set.univ : set ℝ
def A := {x : ℝ | (1 ≤ x ∧ x ≤ 3) ∨ (4 < x ∧ x < 6)}
def B := {x : ℝ | 2 ≤ x ∧ x < 5}
def compl (s : set ℝ) := U \ s

theorem option_B_correct :
  compl B = {x : ℝ | x < 2 ∨ 5 ≤ x} :=
sorry

theorem option_C_correct :
  A ∩ (compl B) = {x : ℝ | (1 ≤ x ∧ x < 2) ∨ (5 ≤ x ∧ x < 6)} :=
sorry

end option_B_correct_option_C_correct_l127_127578


namespace factorial_division_l127_127603

theorem factorial_division (n : ℕ) (h : n = 7) : (n + 2)! / n! = 72 := by
  sorry

end factorial_division_l127_127603


namespace distance_between_lines_eq_l127_127251

-- Define the line l in terms of slope k and y-intercept b
structure Line :=
(k : ℝ)
(b : ℝ)

-- Define conditions for the problem
def line_l : Line := { k := 3/4, b := arbitrary ℝ }

-- Translation of line l by (3,5) to get line l1
def translate (l : Line) (a b : ℝ) : Line := { k := l.k, b := l.b + l.k * -a + b }

def line_l1 := translate line_l 3 5

-- Further translation of line l1 by (1, -2) to get back to line l
def translate_back (l1 : Line) (a b : ℝ) : Line := { k := l1.k, b := l1.b + l1.k * -a + b }

def line_l2 := translate_back line_l1 1 (-2)

-- Prove that the distance between line_l and line_l1 is (11/5)
theorem distance_between_lines_eq : 
  let d : ℝ := (line_l1.b - line_l.b) / real.sqrt (1 + line_l.k^2) in 
  d = 11 / 5 :=
by
  -- Define necessary computations and proofs here
  sorry

end distance_between_lines_eq_l127_127251


namespace monopoly_durable_only_iff_competitive_market_durable_preference_iff_l127_127756

variable (C : ℝ)

-- Definitions based on conditions
def consumer_benefit_period : ℝ := 10
def durable_machine_periods : ℝ := 2
def low_quality_machine_periods : ℝ := 1
def durable_machine_cost : ℝ := 6

-- Statements based on extracted questions & correct answers
theorem monopoly_durable_only_iff (H : C > 3) :
  let durable_benefit := durable_machine_periods * consumer_benefit_period
      durable_price := durable_benefit
      durable_profit := durable_price - durable_machine_cost
      low_quality_price := consumer_benefit_period
      low_quality_profit := low_quality_price - C in
  durable_profit > durable_machine_periods * (low_quality_profit) :=
by 
  sorry

theorem competitive_market_durable_preference_iff (H : C > 3) :
  let durable_benefit := durable_machine_periods * consumer_benefit_period
      durable_surplus := durable_benefit - durable_machine_cost
      low_quality_surplus := low_quality_machine_periods * (consumer_benefit_period - C) in
  durable_surplus > durable_machine_periods * low_quality_surplus :=
by 
  sorry

end monopoly_durable_only_iff_competitive_market_durable_preference_iff_l127_127756


namespace number_of_solutions_l127_127902

theorem number_of_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℕ,
    (x < 10^2006) ∧ ((x * (x - 1)) % 10^2006 = 0) → x ≤ n :=
sorry

end number_of_solutions_l127_127902


namespace cylinder_lateral_surface_area_l127_127354

-- Define the conditions
def circumference := 5 -- in cm
def height := 2 -- in cm

-- Statement to prove
theorem cylinder_lateral_surface_area :
  let lateral_surface_area := circumference * height in
  lateral_surface_area = 10 :=
by
  sorry

end cylinder_lateral_surface_area_l127_127354


namespace min_distance_from_P_to_origin_l127_127222

noncomputable def circleA : set (ℝ × ℝ) := {p | (p.1^2 + p.2^2 = 1)}
noncomputable def circleB : set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 4}
def P (x y : ℝ) : Prop := true -- Any point P in the plane

theorem min_distance_from_P_to_origin (x y : ℝ) (hp : P x y) 
  (h : (circleA.mkP x y ≠ ∅) ∧ (circleB.mkP x y ≠ ∅) ∧ equal_distances) :
  ∃ P : ℝ × ℝ, (P.1, P.2) = (x, y) → 
  ∀ x y, sqrt((0 - x)^2 + (0 - y)^2) ≥ |3 * 0 + 4 * 0 - 11| / sqrt(3^2 + 4^2) :=
sorry

end min_distance_from_P_to_origin_l127_127222


namespace set_intersection_example_l127_127201

theorem set_intersection_example : {1, 2, 9} ∩ {1, 7} = {1} :=
by {
  sorry
}

end set_intersection_example_l127_127201


namespace find_a1_l127_127712

noncomputable def a_seq : ℕ → ℝ
| 1     := sorry -- (we'll fill this in after the proof)
| 2     := sorry -- same here
| (n+2) := sorry -- and this spot too

theorem find_a1 :
  (∀ n ≥ 2, (∑ i in Icc 1 n, a_seq i) = n^2 * a_seq n) ∧ a_seq 50 = 2 → a_seq 1 = 2550 :=
begin
  intro h,
  sorry -- proof goes here
end

end find_a1_l127_127712


namespace area_ratio_PST_to_QRST_l127_127262

-- Define the lengths and points involved
variables {P Q R S T : Type*}
variables {PQ QR PR PS PT : ℝ}
variables (PQ_pos : PQ = 30) (QR_pos : QR = 50) (PR_pos : PR = 54)
variables (PS_pos : PS = 18) (PT_pos : PT = 24)
variables (S_on_PQ : S \isin PQ) (T_on_PR : T \isin PR)

-- Define the main theorem to be proven
theorem area_ratio_PST_to_QRST : 
  (area_ratio := (area P S T / area Q R S T)) = 6 / 19 :=
sorry

end area_ratio_PST_to_QRST_l127_127262


namespace least_integer_N_exists_l127_127912

theorem least_integer_N_exists :
  ∃ (N : ℕ), N = 6097392 ∧
  ∀ (B : finset ℕ), B.card = 2016 →
    ∀ (X : finset ℕ), X ⊆ (finset.range (N+1) \ B) ∧ X.card = 2016 →
      X.sum id = N :=
sorry

end least_integer_N_exists_l127_127912


namespace jerry_pool_time_l127_127842

variables (J : ℕ) -- Denote the time Jerry was in the pool

-- Conditions
def Elaine_time := 2 * J -- Elaine stayed in the pool for twice as long as Jerry
def George_time := (2 / 3) * J -- George could only stay in the pool for one-third as long as Elaine
def Kramer_time := 0 -- Kramer did not find the pool

-- Combined total time
def total_time : ℕ := J + Elaine_time J + George_time J + Kramer_time

-- Theorem stating that J = 3 given the combined total time of 11 minutes
theorem jerry_pool_time (h : total_time J = 11) : J = 3 :=
by
  sorry

end jerry_pool_time_l127_127842


namespace ratio_of_four_numbers_exists_l127_127176

theorem ratio_of_four_numbers_exists (A B C D : ℕ) (h1 : A + B + C + D = 1344) (h2 : D = 672) : 
  ∃ rA rB rC rD, rA ≠ 0 ∧ rB ≠ 0 ∧ rC ≠ 0 ∧ rD ≠ 0 ∧ A = rA * k ∧ B = rB * k ∧ C = rC * k ∧ D = rD * k :=
by {
  sorry
}

end ratio_of_four_numbers_exists_l127_127176


namespace three_classes_same_team_l127_127478

open Function

theorem three_classes_same_team
  (students : Finset ℕ)
  (classes : Finset ℕ)
  (teams : Fin ℕ → ℕ)
  (h_students : students.card = 30)
  (h_classes_three : classes.card = 3)
  (h_team_division : ∀ c ∈ classes, (teams c = 1) ∨ (teams c = 2) ∨ (teams c = 3))
  (h_team_size : ∀ c ∈ classes, ∀ t, (students.filter (λ s, teams s = t)).card = 10) :
  ∃ (s1 s2 : ℕ), s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ 
  ∀ c ∈ classes, teams s1 = teams s2 :=
sorry

end three_classes_same_team_l127_127478


namespace geese_percentage_l127_127272

noncomputable def percentage_of_geese_among_non_swans (geese swans herons ducks : ℝ) : ℝ :=
  (geese / (100 - swans)) * 100

theorem geese_percentage (geese swans herons ducks : ℝ)
  (h1 : geese = 40)
  (h2 : swans = 20)
  (h3 : herons = 15)
  (h4 : ducks = 25) :
  percentage_of_geese_among_non_swans geese swans herons ducks = 50 :=
by
  simp [percentage_of_geese_among_non_swans, h1, h2, h3, h4]
  sorry

end geese_percentage_l127_127272


namespace pentagon_proof_l127_127773

noncomputable def pentagon_problem (ABCDE : Polygon ℝ) (AC AD : ℝ) (pts : List (ℝ × ℝ)) : Prop :=
  (∀ i ∈ [AC, AD], i ≤ Real.sqrt 3) →
  pts.length = 2011 →
  ∃ circle_center edge, edge ∈ ABCDE.edges ∧ dist circle_center pts ≥ 403

theorem pentagon_proof (ABCDE : Polygon ℝ) (AC AD : ℝ) (pts : List (ℝ × ℝ)) : 
  pentagon_problem ABCDE AC AD pts := 
begin
  sorry
end

end pentagon_proof_l127_127773


namespace find_Z_number_of_questions_l127_127460

noncomputable def minimum_questions (N : ℕ) : ℕ :=
  N - 1

theorem find_Z_number_of_questions (N : ℕ) (Z : ℕ) (H : 1 ≤ N) :
  (∀ x, x ≠ Z → knows Z x) ∧ (∀ x, x ≠ Z → ¬ knows x Z) → 
    (minimum_questions N = N - 1) := 
by {
  sorry
}

end find_Z_number_of_questions_l127_127460


namespace exists_x_y_for_2021_pow_n_l127_127157

theorem exists_x_y_for_2021_pow_n (n : ℕ) :
  (∃ x y : ℤ, 2021 ^ n = x ^ 4 - 4 * y ^ 4) ↔ ∃ m : ℕ, n = 4 * m := 
sorry

end exists_x_y_for_2021_pow_n_l127_127157


namespace shifted_parabola_l127_127003

theorem shifted_parabola (x : ℝ) : 
  let original_parabola := fun x : ℝ => x^2 in
  let shifted_parabola := fun x : ℝ => (x + 2)^2 in
  (∀ x, original_parabola (x - 2) = shifted_parabola x) :=
by 
  sorry

end shifted_parabola_l127_127003


namespace white_area_of_sign_remains_l127_127785

theorem white_area_of_sign_remains (h1 : (6 * 18 = 108))
  (h2 : 9 = 6 + 3)
  (h3 : 7.5 = 5 + 3 - 0.5)
  (h4 : 13 = 9 + 4)
  (h5 : 9 = 6 + 3)
  (h6 : 38.5 = 9 + 7.5 + 13 + 9)
  : 108 - 38.5 = 69.5 := by
  sorry

end white_area_of_sign_remains_l127_127785


namespace nonneg_real_values_of_x_count_l127_127923

theorem nonneg_real_values_of_x_count :
  let S := {x : ℝ | 0 ≤ x ∧ ∃ k : ℕ, k ≤ 13 ∧ k = Real.sqrt (169 - Real.cbrt x)} in
  S.card = 14 :=
by
  sorry

end nonneg_real_values_of_x_count_l127_127923


namespace rotate_90_degree_l127_127762

theorem rotate_90_degree (z : ℂ) (hz : z = - 4 - I) : (I * z) = 1 - 4 * I :=
by
  -- Applying a 90 degree counter-clockwise rotation
  rw [hz, mul_add, mul_neg, I_mul_I]
  sorry

end rotate_90_degree_l127_127762


namespace meal_protein_percentage_l127_127447

variable (mix_weight : ℕ) (mix_protein_pct : ℝ)
variable (meal_weight : ℕ) (cornmeal_protein_pct : ℝ)
variable (meal_protein_pct : ℝ)

-- Conditions
def mixture : Prop := mix_weight = 280 ∧ mix_protein_pct = 0.13
def meal : Prop := meal_weight = 240
def cornmeal : Prop := cornmeal_protein_pct = 0.07
def other_meal : Prop := mix_weight - meal_weight = 40

-- Theorem
theorem meal_protein_percentage (h1 : mixture) (h2 : meal) (h3 : cornmeal) (h4 : other_meal) :
  meal_protein_pct = 0.14 :=
sorry

end meal_protein_percentage_l127_127447


namespace paul_can_win_2019_paul_cannot_win_2020_l127_127328

def game_conditions (N : ℕ) : Prop :=
  ∃ (k : ℕ), N - 4 * k = 3

theorem paul_can_win_2019 : game_conditions 2019 :=
begin
  sorry
end

theorem paul_cannot_win_2020 : ¬ game_conditions 2020 :=
begin
  sorry
end

end paul_can_win_2019_paul_cannot_win_2020_l127_127328


namespace rebecca_groups_of_eggs_l127_127333

def eggs : Nat := 16
def group_size : Nat := 2

theorem rebecca_groups_of_eggs : (eggs / group_size) = 8 := by
  sorry

end rebecca_groups_of_eggs_l127_127333


namespace train_crossing_time_l127_127059

theorem train_crossing_time :
  ∀ (L_train L_platform : ℝ) (v_kmph : ℝ),
  L_train = 110 → 
  L_platform = 165 → 
  v_kmph = 132 → 
  (let v_mps := v_kmph * 1000 / 3600 in
  let total_distance := L_train + L_platform in
  let t := total_distance / v_mps in
  t ≈ 7.49) := 
by
  intros L_train L_platform v_kmph h_train h_platform h_speed
  dsimp [v_mps, total_distance, t]
  rw [h_train, h_platform, h_speed]
  norm_num
  have h : (110 + 165) / (132 * 1000 / 3600) ≈ 7.49 := by norm_num
  exact h

end train_crossing_time_l127_127059


namespace proper_fraction_and_condition_l127_127163

theorem proper_fraction_and_condition (a b : ℤ) (h1 : 1 < a) (h2 : b = 2 * a - 1) :
  0 < a ∧ a < b ∧ (a - 1 : ℚ) / (b - 1) = 1 / 2 :=
by
  sorry

end proper_fraction_and_condition_l127_127163


namespace ordered_triple_solution_l127_127517

theorem ordered_triple_solution :
  ∃ (x y z : ℤ), (3 * x + 2 * y - z = 1) ∧ (4 * x - 5 * y + 3 * z = 11) ∧ (x = 1 ∧ y = 1 ∧ z = 4) :=
by
  use 1
  use 1
  use 4
  split
  { -- Show that 3 * 1 + 2 * 1 - 4 = 1
    calc
      3 * 1 + 2 * 1 - 4 = 3 + 2 - 4 : by ring
      ... = 1 : by ring
  }
  split
  { -- Show that 4 * 1 - 5 * 1 + 3 * 4 = 11
    calc
      4 * 1 - 5 * 1 + 3 * 4 = 4 - 5 + 12 : by ring
      ... = 11 : by ring
  }
  { -- Show that (x, y, z) = (1, 1, 4)
    split
    { refl }
    split
    { refl }
    { refl }
  }

end ordered_triple_solution_l127_127517


namespace max_det_value_l127_127166

theorem max_det_value :
  ∃ θ : ℝ, 
    (1 * ((5 + Real.sin θ) * 9 - 6 * 8) 
     - 2 * (4 * 9 - 6 * (7 + Real.cos θ)) 
     + 3 * (4 * 8 - (5 + Real.sin θ) * (7 + Real.cos θ))) 
     = 93 :=
sorry

end max_det_value_l127_127166


namespace path_count_A_to_D_via_B_and_C_l127_127612

def count_paths (start finish : ℕ × ℕ) (h_steps v_steps : ℕ) : ℕ :=
  Nat.choose (h_steps + v_steps) v_steps

theorem path_count_A_to_D_via_B_and_C :
  let A := (0, 5)
  let B := (3, 3)
  let C := (4, 1)
  let D := (6, 0)
  count_paths A B 3 2 * count_paths B C 1 2 * count_paths C D 2 1 = 90 :=
by {
  let A := (0, 5),
  let B := (3, 3),
  let C := (4, 1),
  let D := (6, 0),
  suffices h1 : count_paths A B 3 2 = 10, by {
    suffices h2 : count_paths B C 1 2 = 3, by {
      suffices h3 : count_paths C D 2 1 = 3, by {
        calc
          count_paths A B 3 2 * count_paths B C 1 2 * count_paths C D 2 1
            = 10 * 3 * 3 : by rw [h1, h2, h3]
        ... = 90 : by norm_num
      }
      show count_paths C D 2 1 = 3, sorry
    }
    show count_paths B C 1 2 = 3, sorry
  }
  show count_paths A B 3 2 = 10, sorry
}

end path_count_A_to_D_via_B_and_C_l127_127612


namespace sum_roots_zero_l127_127018

-- Let Q be a quadratic polynomial
def Q (x : ℝ) : ℝ

-- Define the condition for Q
def condition (Q : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, Q(x^3 - x) ≥ Q(x^2 - 1)

-- Define the property to prove
theorem sum_roots_zero (Q : ℝ → ℝ) [quad : ∀ x ∈ ℝ, polynomial.degree Q ≤ 2] 
  (hQ : condition Q) : 
  ∑ root in (polynomial.roots Q), root = 0 := 
sorry

end sum_roots_zero_l127_127018


namespace find_pair_ab_l127_127988

theorem find_pair_ab (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : ∀ x, f x = x^2 - 2 * |x| + 4) 
  (h2 : a < b) 
  (h3 : ∀ y, y ∈ set.range f ↔ ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y) :
  (a = 1 ∧ b = 4) :=
by 
  -- We declare the conditions encapsulated in h1, h2, and h3 first,
  -- then conclude that (a, b) = (1, 4)
  sorry

end find_pair_ab_l127_127988


namespace rectangle_length_l127_127234

-- Define the area and width of the rectangle as given
def width : ℝ := 4
def area  : ℝ := 28

-- Prove that the length is 7 cm given the conditions
theorem rectangle_length : ∃ length : ℝ, length = 7 ∧ area = length * width :=
sorry

end rectangle_length_l127_127234


namespace projection_of_a_in_direction_of_e_l127_127181

variables {V : Type*} [InnerProductSpace ℝ V]
variables (a e : V)

-- Define conditions
-- |e| = 1
def norm_e_one : Prop :=
  ‖e‖ = 1

-- |a + e| = |a - 2e|
def norm_eq_cond : Prop :=
  ‖a + e‖ = ‖a - 2 • e‖

-- Question: Prove the projection of "a" in the direction of "e" is 1/2
theorem projection_of_a_in_direction_of_e
  (hnorm_e : norm_e_one e)
  (hnorm_eq : norm_eq_cond a e) :
  real_inner a e = 1 / 2 :=
sorry

end projection_of_a_in_direction_of_e_l127_127181


namespace truck_distance_l127_127832

theorem truck_distance :
  ∀ (v_t d_t v_c d_c : ℝ),
    (d_t = v_t * 8) ∧ 
    (v_c = v_t + 18) ∧ 
    (d_c = d_t + 6.5) ∧ 
    (d_c = v_c * 5.5) → 
    d_t = 296 := 
by
  /* Skip the proof */
  sorry

end truck_distance_l127_127832


namespace f_at_10_l127_127213

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

-- Prove that f(10) = 756
theorem f_at_10 : f 10 = 756 := by
  sorry

end f_at_10_l127_127213


namespace geometric_sequence_term_l127_127023

theorem geometric_sequence_term
  (r a : ℝ)
  (h1 : 180 * r = a)
  (h2 : a * r = 81 / 32)
  (h3 : a > 0) :
  a = 135 / 19 :=
by sorry

end geometric_sequence_term_l127_127023


namespace find_integer_pairs_l127_127519

theorem find_integer_pairs (x y : ℕ) (h : x ^ 5 = y ^ 5 + 10 * y ^ 2 + 20 * y + 1) : (x, y) = (1, 0) :=
  sorry

end find_integer_pairs_l127_127519


namespace red_balls_estimation_l127_127839

-- Definitions based on conditions
def balls_in_bag (red_balls : ℕ) : ℕ := red_balls + 10

def probability_black_ball (red_balls : ℕ) : ℝ := 10 / (balls_in_bag red_balls)
  
theorem red_balls_estimation (red_balls : ℕ) (h : probability_black_ball red_balls = 0.4) : red_balls = 15 := by
  sorry

end red_balls_estimation_l127_127839


namespace downstream_speed_average_l127_127098

variables (D V_d : ℝ)

/-- The average speed of the boat upstream is 3 km/h. -/
def upstream_speed : ℝ := 3

/-- The average speed for the round trip is 4.2 km/h. -/
def round_trip_avg_speed : ℝ := 4.2

/-- The equation representing the average speed of the round trip -/
def round_trip_equation (V_d : ℝ) (D : ℝ) : ℝ :=
  (2 * D) / ((D / upstream_speed) + (D / V_d))

theorem downstream_speed_average (V_d D : ℝ) :
  round_trip_avg_speed = 4.2 → upstream_speed = 3 →
  (round_trip_equation V_d D = 4.2 → V_d = 7) := by
  sorry

end downstream_speed_average_l127_127098


namespace square_area_in_right_triangle_l127_127343

theorem square_area_in_right_triangle
  (A G D B C E F : Point) -- Points in the plane defining the right triangle and the square
  (right_triangle_AGD : RightTriangle A G D)
  (square_BCFE : Square B C F E)
  (inscribed_BCFE_in_AGD : InscribedSquareInRightTriangle square_BCFE right_triangle_AGD)
  (AB_eq_36 : AB = 36)
  (CD_eq_72 : CD = 72) :
  area square_BCFE = 2592 := 
  sorry

end square_area_in_right_triangle_l127_127343


namespace length_of_AB_l127_127380

theorem length_of_AB (V : ℝ) (r : ℝ) :
  V = 216 * Real.pi →
  r = 3 →
  ∃ (len_AB : ℝ), len_AB = 20 :=
by
  intros hV hr
  have volume_cylinder := V - 36 * Real.pi
  have height_cylinder := volume_cylinder / (Real.pi * r^2)
  exists height_cylinder
  exact sorry

end length_of_AB_l127_127380


namespace cos_105_eq_fraction_l127_127871

theorem cos_105_eq_fraction : 
  cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  have h_cos_45 : cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.cos_eq_sqrt_4_inv_norm_eq_sqrt_two_div_two]
  have h_cos_60 : cos (60 * Real.pi / 180) = 1 / 2 :=
    by norm_num [Real.cos_pi_div_three]
  have h_sin_45 : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.sin_eq_sqrt_4_inv_sqrt_two_div_two]
  have h_sin_60 : sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by norm_num [Real.sin_pi_div_three]
  sorry

end cos_105_eq_fraction_l127_127871


namespace possible_to_connect_points_with_arrows_l127_127543

theorem possible_to_connect_points_with_arrows (n : ℕ) (hn : n > 4) : 
  ∃ f : Fin n × Fin n → Bool, 
    (∀ x y : Fin n, x ≠ y → 
      (f (x, y) ∨ 
        ∃ z : Fin n, f (x, z) ∧ f (z, y))) := 
begin
  sorry
end

end possible_to_connect_points_with_arrows_l127_127543


namespace rowing_distance_l127_127382

theorem rowing_distance (v_b : ℝ) (v_s : ℝ) (t_total : ℝ) (D : ℝ) :
  v_b = 9 → v_s = 1.5 → t_total = 48 → D / (v_b + v_s) + D / (v_b - v_s) = t_total → D = 210 :=
by
  intros
  sorry

end rowing_distance_l127_127382


namespace point_quadrant_I_or_IV_l127_127217

def is_point_on_line (x y : ℝ) : Prop := 4 * x + 3 * y = 12
def is_equidistant_from_axes (x y : ℝ) : Prop := |x| = |y|

def point_in_quadrant_I (x y : ℝ) : Prop := (x > 0 ∧ y > 0)
def point_in_quadrant_IV (x y : ℝ) : Prop := (x > 0 ∧ y < 0)

theorem point_quadrant_I_or_IV (x y : ℝ) 
  (h1 : is_point_on_line x y) 
  (h2 : is_equidistant_from_axes x y) :
  point_in_quadrant_I x y ∨ point_in_quadrant_IV x y :=
sorry

end point_quadrant_I_or_IV_l127_127217


namespace sum_of_first_n_odd_integers_l127_127680

theorem sum_of_first_n_odd_integers (n : ℕ) : (∑ i in Finset.range n, (2 * i + 1)) = n^2 :=
by
  sorry

end sum_of_first_n_odd_integers_l127_127680


namespace transformed_function_l127_127179

noncomputable def g (x : ℝ) : ℝ := sin (2 * x)

-- Translate the graph to the left by π/8 units
noncomputable def translated_g (x : ℝ) : ℝ := g (x + π / 8)

-- Shorten the horizontal coordinates to 1/4 of their original length
noncomputable def f (x : ℝ) : ℝ := translated_g (4 * x)

theorem transformed_function :
  f x = sin (8 * x + π / 4) := by
  sorry

end transformed_function_l127_127179


namespace avg_and_var_of_scaled_shifted_data_l127_127562

-- Definitions of average and variance
noncomputable def avg (l: List ℝ) : ℝ := (l.sum) / l.length
noncomputable def var (l: List ℝ) : ℝ := (l.map (λ x => (x - avg l) ^ 2)).sum / l.length

theorem avg_and_var_of_scaled_shifted_data
  (n : ℕ)
  (x : Fin n → ℝ)
  (h_avg : avg (List.ofFn x) = 2)
  (h_var : var (List.ofFn x) = 3) :
  avg (List.ofFn (λ i => 2 * x i + 3)) = 7 ∧ var (List.ofFn (λ i => 2 * x i + 3)) = 12 := by
  sorry

end avg_and_var_of_scaled_shifted_data_l127_127562


namespace sum_of_solutions_eq_one_l127_127304

def f (x : ℝ) : ℝ :=
if x < -3 then 2 * x + 4 else -x^2 + x + 3

theorem sum_of_solutions_eq_one : 
  (∃ x : ℝ, f x = 0) → (∀ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 → x1 + x2 = 1) :=
by
  sorry

end sum_of_solutions_eq_one_l127_127304


namespace cone_properties_l127_127450

-- Definition for a cone's characteristics.
def cone.has_surface : Prop := ∀ (c : Cone), c.surface_count = 2

-- Definition for the unfolding of a cone's lateral surface into a sector.
def cone.lateral_unfolds_into : Prop := ∀ (c : Cone), c.lateral_surface.shape = Sector

-- Statement of the theorem combining the conditions and the required proof.
theorem cone_properties (c : Cone) : cone.has_surface c ∧ cone.lateral_unfolds_into c := by
  sorry

end cone_properties_l127_127450


namespace min_value_of_n_over_m_l127_127541

theorem min_value_of_n_over_m (m n : ℝ)
  (h1 : ∀ x : ℝ, (exp(x) - m * x + n - 1) ≥ 0): 
  ∃ x : ℝ, (n / m) = 0 :=
sorry

end min_value_of_n_over_m_l127_127541


namespace ping_pong_ball_probability_l127_127893

theorem ping_pong_ball_probability :
  (∃ (balls : Finset ℕ), balls.card = 80 ∧ 
    (∀ (n : ℕ), n ∈ balls ↔ 1 ≤ n ∧ n ≤ 80) ∧
    let multiples_of_6 := balls.filter (λ n, n % 6 = 0),
        multiples_of_9 := balls.filter (λ n, n % 9 = 0),
        multiples_of_18 := balls.filter (λ n, n % 18 = 0) in
    let count_6 := multiples_of_6.card,
        count_9 := multiples_of_9.card,
        count_18 := multiples_of_18.card in
    let total_multiples := count_6 + count_9 - count_18 in
    let probability := total_multiples.to_rat / 80 in
    probability = 17 / 80
  ) :=
sorry

end ping_pong_ball_probability_l127_127893


namespace sum_of_products_two_at_a_time_l127_127714

theorem sum_of_products_two_at_a_time
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 222)
  (h2 : a + b + c = 22) :
  a * b + b * c + c * a = 131 := 
sorry

end sum_of_products_two_at_a_time_l127_127714


namespace combination_equality_l127_127180

-- open the required namespaces to use combinatorial functions and arithmetic
open nat

-- Define the given problem conditions and desired proof outcome.
theorem combination_equality (x : ℕ) (h : nat.choose 5 x = nat.choose 5 (x-1)) : x = 3 :=
sorry

end combination_equality_l127_127180


namespace average_strokes_per_hole_l127_127395

theorem average_strokes_per_hole (total_rounds : ℕ) (par_per_hole : ℕ) (over_par : ℕ) :
  total_rounds = 9 → par_per_hole = 3 → over_par = 9 →
  (total_rounds * par_per_hole + over_par) / total_rounds = 4 :=
by
  intros htr hpph hop,
  sorry

end average_strokes_per_hole_l127_127395


namespace total_cartons_packed_l127_127468

-- Define the given conditions
def cans_per_carton : ℕ := 20
def cartons_loaded : ℕ := 40
def cans_left : ℕ := 200

-- Formalize the proof problem
theorem total_cartons_packed : cartons_loaded + (cans_left / cans_per_carton) = 50 := by
  sorry

end total_cartons_packed_l127_127468


namespace university_students_grouping_l127_127752

theorem university_students_grouping
  (E F G : Finset ℕ) -- Set of students knowing English (E), French (F), German (G)
  (hE : E.card = 50)
  (hF : F.card = 50)
  (hG : G.card = 50)
  (hEFG : ∀ (e ∈ E) (f ∈ F) (g ∈ G), e ≠ f ∨ f ≠ g ∨ g ≠ e) -- Overlap condition.
  : ∃ (groups : Fin 5 → Finset ℕ), 
      (∀ i, (groups i).card = 30 ∧ 
            (groups i ∩ E).card = 10 ∧ 
            (groups i ∩ F).card = 10 ∧ 
            (groups i ∩ G).card = 10) := 
  sorry

end university_students_grouping_l127_127752


namespace max_sum_of_squares_eq_sqrt2_max_product_eq_half_l127_127557

theorem max_sum_of_squares_eq_sqrt2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 + b^2 = 1) : 
  a + b ≤ real.sqrt 2 :=
sorry

theorem max_product_eq_half (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 + b^2 = 1) : 
  a * b ≤ 1 / 2 :=
sorry

end max_sum_of_squares_eq_sqrt2_max_product_eq_half_l127_127557


namespace min_distance_from_origin_l127_127200

theorem min_distance_from_origin (a b : ℝ) (h : 3 * a + 4 * b = 20) : ∃ d, d = 4 ∧ ∀ x y : ℝ, 3 * x + 4 * y = 20 → sqrt (x^2 + y^2) ≥ d :=
by 
  use 4
  split
  -- Proofs will be here.
  sorry

end min_distance_from_origin_l127_127200


namespace number_of_ordered_pairs_l127_127873

theorem number_of_ordered_pairs :
  ∃ (n : ℕ), n = 99 ∧
  (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ (Int.gcd a b) * a + b^2 = 10000
  → ∃ (k : ℕ), k = 99) :=
sorry

end number_of_ordered_pairs_l127_127873


namespace cookies_problem_l127_127025

theorem cookies_problem (x : ℕ) 
  (h1 : ∀ n, (n = 6) ↔ (n = 7 + x) / 2) : x = 5 :=
by
  have h2 : 6 = (7 + x) / 2 := h1 6
  have h3 : 12 = 7 + x := by linarith
  sorry

end cookies_problem_l127_127025


namespace sara_golf_balls_l127_127673

theorem sara_golf_balls (golf_balls dozens : ℕ) (h1 : golf_balls = 192) (h2 : dozens = 12) : (192 / 12) = 16 :=
by
  rw [h1, h2]
  sorry

end sara_golf_balls_l127_127673


namespace max_x2_plus_2xy_plus_3y2_l127_127345

theorem max_x2_plus_2xy_plus_3y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 18 + 9 * Real.sqrt 3 :=
sorry

end max_x2_plus_2xy_plus_3y2_l127_127345


namespace minimum_perimeter_triangle_l127_127879

-- Define the given angle and point inside the angle
variables {O M N A C D : Point}
variables (angle_MON : Angle O M N)
variables (point_A_inside_angle : Angle.Mem interior O M A ∧ Angle.Mem interior O N A)

-- Define the symmetry points
def symmetric_point_across_line (P : Point) (line : Line) : Point := 
  sorry -- Implementation of symmetric point across given line

noncomputable def A1 := symmetric_point_across_line A (Line.mk O M) -- A1 is symmetric to A across OM
noncomputable def A2 := symmetric_point_across_line A (Line.mk O N) -- A2 is symmetric to A across ON

-- Define the intersection points
def intersection_point (line1 line2 : Line) : Point := 
  sorry -- Implementation of intersection of two lines

noncomputable def C := intersection_point (Line.mk A1 A2) (Line.mk O M) -- Intersection of A1A2 with OM
noncomputable def D := intersection_point (Line.mk A1 A2) (Line.mk O N) -- Intersection of A1A2 with ON

-- The main theorem stating the triangle CDA has the smallest perimeter
theorem minimum_perimeter_triangle :
  ∀ (A1 A2 C D : Point),
  A1 = symmetric_point_across_line A (Line.mk O M) ∧
  A2 = symmetric_point_across_line A (Line.mk O N) ∧
  C = intersection_point (Line.mk A1 A2) (Line.mk O M) ∧
  D = intersection_point (Line.mk A1 A2) (Line.mk O N) →
  (triangle_perimeter A C D) = min (triangle_perimeter _) :=
sorry

end minimum_perimeter_triangle_l127_127879


namespace prove_circle_tangent_radius_l127_127727

noncomputable def circle_tangent_radius (O A B C : ℝ × ℝ) : Prop :=
  let r₁ : ℝ := 10  -- radius of the given circle
  let AB : ℝ := 16  -- length of chord AB
  let r₂ : ℝ := 8   -- radius of the circle we need to prove

  -- Conditions:
  -- 1. Point A lies on the circle with center O and radius r₁
  -- 2. AB is a chord through A with length AB
  -- 3. AC is a chord perpendicular to AB, also through A

  ∃ (O₁ : ℝ × ℝ),  -- center of the second circle
    (dist O₂ O = r₂) ∧ -- second circle is tangent to the given circle
    (dist A O₂ = r₂) ∧ -- second circle is tangent to the chord AB
    (dist A O₁ = r₁ + r₂)  -- second circle is tangent to the chord AC

theorem prove_circle_tangent_radius :
  ∀ (O A B C : ℝ × ℝ), circle_tangent_radius O A B C := 
begin
  sorry  -- proof not required by the instructions
end

end prove_circle_tangent_radius_l127_127727


namespace cos_105_eq_fraction_l127_127868

theorem cos_105_eq_fraction : 
  cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  have h_cos_45 : cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.cos_eq_sqrt_4_inv_norm_eq_sqrt_two_div_two]
  have h_cos_60 : cos (60 * Real.pi / 180) = 1 / 2 :=
    by norm_num [Real.cos_pi_div_three]
  have h_sin_45 : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.sin_eq_sqrt_4_inv_sqrt_two_div_two]
  have h_sin_60 : sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by norm_num [Real.sin_pi_div_three]
  sorry

end cos_105_eq_fraction_l127_127868


namespace find_angle_B_find_side_b_l127_127631

variables {A B C a b c : ℝ} 

-- Given conditions
def condition_trig := (cos (A + C) + cos (A - C) - 1) / (cos (A - B) + cos C) = c / b
def side_c := c = 2
def point_D_condition := AD = 2 * DC
def BD_value := BD = (2 * sqrt 13) / 3

-- Proof problem 1: Finding angle B
theorem find_angle_B (A C : ℝ) (a b c : ℝ) (h1 : condition_trig) : 
  B = 2 * π / 3 := 
sorry

-- Proof problem 2: Finding b
theorem find_side_b 
  (A B C a b c AD DC BD : ℝ) 
  (h1 : B = 2 * π / 3) 
  (h2 : side_c) 
  (h3 : point_D_condition) 
  (h4 : BD_value) : 
  b = 2 * sqrt 7 := 
sorry

end find_angle_B_find_side_b_l127_127631


namespace ofelia_savings_in_may_l127_127325

-- Conditions
def savings (n : ℕ) : ℕ :=
if n = 1 then 10 else 2 * (savings (n - 1))

-- Theorem
theorem ofelia_savings_in_may : savings 5 = 160 :=
by
  sorry

end ofelia_savings_in_may_l127_127325


namespace number_of_bricks_is_1800_l127_127080

-- Define the conditions
def rate_first_bricklayer (x : ℕ) : ℕ := x / 8
def rate_second_bricklayer (x : ℕ) : ℕ := x / 12
def combined_reduced_rate (x : ℕ) : ℕ := (rate_first_bricklayer x + rate_second_bricklayer x - 15)

-- Prove that the number of bricks in the wall is 1800
theorem number_of_bricks_is_1800 :
  ∃ x : ℕ, 5 * combined_reduced_rate x = x ∧ x = 1800 :=
by
  use 1800
  sorry

end number_of_bricks_is_1800_l127_127080


namespace vip_seat_cost_is_65_l127_127467

noncomputable def cost_of_VIP_seat (G V_T V : ℕ) (cost : ℕ) : Prop :=
  G + V_T = 320 ∧
  (15 * G + V * V_T = cost) ∧
  V_T = G - 212 → V = 65

theorem vip_seat_cost_is_65 :
  ∃ (G V_T V : ℕ), cost_of_VIP_seat G V_T V 7500 :=
  sorry

end vip_seat_cost_is_65_l127_127467


namespace scientific_notation_of_360_billion_l127_127430

def number_in_scientific_notation (n : ℕ) : String :=
  match n with
  | 360000000000 => "3.6 × 10^11"
  | _ => "Unknown"

theorem scientific_notation_of_360_billion : 
  number_in_scientific_notation 360000000000 = "3.6 × 10^11" :=
by
  -- insert proof steps here
  sorry

end scientific_notation_of_360_billion_l127_127430


namespace cards_in_middle_pile_after_steps_l127_127310

theorem cards_in_middle_pile_after_steps
  (x : ℕ)
  (h : x ≥ 2) :
  let left_pile_step_1 := x - 2,
      middle_pile_step_2 := x + 3,
      middle_pile_final := middle_pile_step_2 - left_pile_step_1 in
  middle_pile_final = 5 :=
begin
  dsimp [left_pile_step_1, middle_pile_step_2, middle_pile_final],
  linarith
end

end cards_in_middle_pile_after_steps_l127_127310


namespace bill_steps_l127_127237

theorem bill_steps (step_length : ℝ) (total_distance : ℝ) (n_steps : ℕ) 
  (h_step_length : step_length = 1 / 2) 
  (h_total_distance : total_distance = 12) 
  (h_n_steps : n_steps = total_distance / step_length) : 
  n_steps = 24 :=
by sorry

end bill_steps_l127_127237


namespace geometric_sequence_a6_l127_127259

theorem geometric_sequence_a6 (a : ℕ → ℝ) (r : ℝ)
  (h₁ : a 4 = 7)
  (h₂ : a 8 = 63)
  (h_geom : ∀ n, a n = a 1 * r^(n - 1)) :
  a 6 = 21 :=
sorry

end geometric_sequence_a6_l127_127259


namespace PA_PB_PC_constant_l127_127835

open Real

variables {A B C P : Point}
variables {a : ℝ}

-- Definitions for the conditions
def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def inscribe_in_circle (ABC c : Point → Prop) : Prop :=
  ∀ (P : Point), dist P ABC circ = r

def on_circumference (P c : Point → Prop) : Prop := 
  ∃ r : ℝ, dist P c = r

-- The theorem statement
theorem PA_PB_PC_constant
  (h1 : equilateral_triangle A B C)
  (h2 : inscribe_in_circle A B C P)
  (h3 : on_circumference P A B C)
  : dist P A ^ 2 + dist P B ^ 2 + dist P C ^ 2 = 2 * (dist A B ^ 2) :=
sorry

end PA_PB_PC_constant_l127_127835


namespace sum_not_zero_l127_127015

open Int

theorem sum_not_zero {a : Fin 22 → Int} (h : (∏ i, a i) = 1) : (∑ i, a i) ≠ 0 :=
by
  sorry

end sum_not_zero_l127_127015


namespace find_m_range_l127_127537

noncomputable def range_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : Prop :=
  m ≥ 4

-- Here is the theorem statement
theorem find_m_range (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : range_m a b c m h1 h2 h3 :=
sorry

end find_m_range_l127_127537


namespace equal_playing_time_for_each_player_l127_127786

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l127_127786


namespace height_difference_l127_127670

def empireStateBuildingHeight : ℕ := 443
def petronasTowersHeight : ℕ := 452

theorem height_difference :
  petronasTowersHeight - empireStateBuildingHeight = 9 := 
sorry

end height_difference_l127_127670


namespace binomial_coeff_x3_l127_127355

theorem binomial_coeff_x3 :
  let a := 3
  let b := 2
  let n := 6
  let r := 3
  let coefficient := (Nat.choose n r) * a^(n-r) * (-b)^r
  coefficient = -4320 :=
by
  sorry

end binomial_coeff_x3_l127_127355


namespace min_radius_for_area_l127_127311

theorem min_radius_for_area (A : ℝ) (hA : A = 500) : ∃ r : ℝ, r = 13 ∧ π * r^2 ≥ A :=
by
  sorry

end min_radius_for_area_l127_127311


namespace find_angle_y_l127_127260

-- Mathematical definitions as conditions
variables {P Q R S T F : Type*}
variables [EuclideanGeometry P Q R S T F]

-- Angles given in the problem
def angle_PQR := 30
def angle_QPS := 80
def angle_TPQ := 130
def angle_PRF := 110

-- Parallel lines denomination
def PQ_parallel_ST : Prop := isParallel PQ ST

-- Definition of a straight line
def PRF_straight_line : Prop := isStraightLine PR F

theorem find_angle_y (h1 : PQ_parallel_ST) (h2 : PRF_straight_line) :
  angle (QSR) = 20 := 
sorry -- Proof

end find_angle_y_l127_127260


namespace complex_magnitude_problem_l127_127947

-- Let's define the problem in Lean 4
theorem complex_magnitude_problem (z : ℂ) (h : (z + complex.I) / (-2 * (complex.I ^ 3) - z) = complex.I) :
  abs (conj(z) + 1) = real.sqrt 2 / 2 :=
sorry

end complex_magnitude_problem_l127_127947


namespace shift_parabola_upwards_l127_127361

theorem shift_parabola_upwards (y x : ℝ) (h : y = x^2) : y + 5 = (x^2 + 5) := by 
  sorry

end shift_parabola_upwards_l127_127361


namespace yvon_combination_l127_127416

theorem yvon_combination :
  let num_notebooks := 4
  let num_pens := 5
  num_notebooks * num_pens = 20 :=
by
  sorry

end yvon_combination_l127_127416


namespace intersection_complement_eq_l127_127577

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≥ 2}

-- Define the intersection of A and complement of B
def intersection : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- The theorem to be proved
theorem intersection_complement_eq : (A ∩ complement_B) = intersection :=
sorry

end intersection_complement_eq_l127_127577


namespace relations_among_zeros_l127_127573

noncomputable def f (x : ℝ) : ℝ := 2 ^ x + x
noncomputable def g (x : ℝ) : ℝ := log 2 x + x
def h (x : ℝ) : ℝ := x ^ 3 + x

axiom zero_of_f_eq : ∃ a : ℝ, f a = 0
axiom zero_of_g_eq : ∃ b : ℝ, g b = 0
axiom zero_of_h_eq : ∃ c : ℝ, h c = 0

theorem relations_among_zeros :
  (∃ a b c : ℝ, f a = 0 ∧ g b = 0 ∧ h c = 0) →
  ∃ a b c : ℝ, a < c ∧ c < b :=
by
  intros a b c ha hb hc,
  sorry

end relations_among_zeros_l127_127573


namespace polar_equation_of_line_AB_l127_127991

noncomputable def polar_coordinates_A : ℝ × ℝ := (4, 2 * Real.pi / 3)
noncomputable def polar_coordinates_B : ℝ × ℝ := (2, Real.pi / 3)

theorem polar_equation_of_line_AB :
  ∃ ρ θ, (polar_coordinates_A, polar_coordinates_B) →
          ρ * Real.sin (θ + Real.pi / 6) = 2 := 
sorry

end polar_equation_of_line_AB_l127_127991


namespace general_term_a_sum_an_an1_sum_abs_bn_l127_127980

def f (x : ℝ) : ℝ := (7 * x + 5) / (x + 1)

def a_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * a (n + 1) - 2 * a n + a (n + 1) * a n = 0 ∧ a n ≠ 0

def b_seq (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = f 0 ∧ ∀ n > 1, b n = f (a (n - 1) - 1)

-- General term of the sequence {a_n}
theorem general_term_a {a : ℕ → ℝ} (h : a_seq a) : ∀ n, a n = 2 / (n + 1) :=
sorry

-- Sum of the first n terms of the sequence {a_n * a_{n+1}}
theorem sum_an_an1 {a : ℕ → ℝ} (h : a_seq a) :
  ∀ n, (∑ k in Finset.range n, a k * a (k + 1)) = 2 * n / (n + 2) :=
sorry

-- Sum of the first n terms of the sequence {|b_n|}
theorem sum_abs_bn {a b : ℕ → ℝ} (ha : a_seq a) (hb : b_seq b a) :
  ∀ n, ∑ k in Finset.range n, |b k| =
    if n ≤ 6 then n * (11 - n) / 2
    else (n^2 - 11 * n + 60) / 2 :=
sorry

end general_term_a_sum_an_an1_sum_abs_bn_l127_127980


namespace find_y_intercept_l127_127140

theorem find_y_intercept (m : ℝ) 
  (h1 : ∀ x y : ℝ, y = 2 * x + m)
  (h2 : ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = 2 * x + m) : 
  m = -1 := 
sorry

end find_y_intercept_l127_127140


namespace points_of_tangency_lie_on_circle_l127_127066

-- Definitions based on the conditions
variables (P : Plane) (A B : Point)
hypothesis h1 : A.side_of P = B.side_of P
hypothesis h2 : ¬Parallel (LineThrough A B) P

-- Lean theorem statement
theorem points_of_tangency_lie_on_circle (M : Point)
  (H1 : M ∈ P ∧ M ∈ LineThrough A B)
  (R : ℝ)
  (H2 : ∀ (T : Point), T ∈ P ∧ Tangent (SphereThrough A B) P T → dist M T = R) :
  ∃ (c : Circle), ∀ (T : Point), T ∈ P ∧ Tangent (SphereThrough A B) P T → T ∈ c :=
sorry

end points_of_tangency_lie_on_circle_l127_127066


namespace BD_sum_eq_99_l127_127877

def P (z : ℂ) : ℂ := z^4 + 6 * z^3 + 7 * z^2 + 3 * z + 2

def f (z : ℂ) : ℂ := 3 * complex.I * conj z

def Q (z : ℂ) : ℂ := 
  let z1 := f z1
  let z2 := f z2
  let z3 := f z3
  let z4 := f z4
  z^4 + A * z^3 + B * z^2 + C * z + D
  -- Here A, B, C, D are coefficients of polynomial Q where roots are transformed.

theorem BD_sum_eq_99 :
  let z1, z2, z3, z4 := ... -- the roots of P(z)
  let f_z1, f_z2, f_z3, f_z4 := f z1, f z2, f z3, f z4
  let Q := z^4 + A * z^3 + B * z^2 + C * z + D -- Polynomial with roots f(z1), f(z2), f(z3), f(z4)
  (B + D) = 99 :=
sorry

end BD_sum_eq_99_l127_127877


namespace train_crossing_platform_l127_127104

/-- Given a train crosses a 100 m platform in 15 seconds, and the length of the train is 350 m,
    prove that the train takes 20 seconds to cross a second platform of length 250 m. -/
theorem train_crossing_platform (dist1 dist2 l_t t1 t2 : ℝ) (h1 : dist1 = 100) (h2 : dist2 = 250) (h3 : l_t = 350) (h4 : t1 = 15) :
  t2 = 20 :=
sorry

end train_crossing_platform_l127_127104


namespace find_costs_of_accessories_max_type_a_accessories_l127_127769

theorem find_costs_of_accessories (x y : ℕ) 
  (h1 : x + 3 * y = 530) 
  (h2 : 3 * x + 2 * y = 890) : 
  x = 230 ∧ y = 100 := 
by 
  sorry

theorem max_type_a_accessories (m n : ℕ) 
  (m_n_sum : m + n = 30) 
  (cost_constraint : 230 * m + 100 * n ≤ 4180) : 
  m ≤ 9 := 
by 
  sorry

end find_costs_of_accessories_max_type_a_accessories_l127_127769


namespace find_k_l127_127996

def vector := ℝ × ℝ  -- Define a vector as a pair of real numbers

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a (k : ℝ) : vector := (k, 3)
def b : vector := (1, 4)
def c : vector := (2, 1)
def linear_combination (k : ℝ) : vector := ((2 * k - 3), -6)

theorem find_k (k : ℝ) (h : dot_product (linear_combination k) c = 0) : k = 3 := by
  sorry

end find_k_l127_127996


namespace flag_arrangements_remainder_l127_127388

theorem flag_arrangements_remainder :
  let num_blue_flags := 12
  let num_green_flags := 11
  let num_flagpoles := 2
  
  -- Condition: Each flagpole must have at least one flag.
  -- Condition: No two green flags on either pole are adjacent.
  -- Condition: No two blue flags on either pole are adjacent.
  
  -- Question: Prove the number of valid arrangements modulo 1000 is 858.
  (number_of_arrangements num_blue_flags num_green_flags num_flagpoles) % 1000 = 858 :=
sorry

end flag_arrangements_remainder_l127_127388


namespace sum_of_coefficients_eq_l127_127607

theorem sum_of_coefficients_eq :
  ∃ n : ℕ, (∀ a b : ℕ, (3 * a + 5 * b)^n = 2^15) → n = 5 :=
by
  sorry

end sum_of_coefficients_eq_l127_127607


namespace find_c_l127_127516

theorem find_c (c d : ℤ) (h_factor : (λ x : ℂ, x^2 - 2 * x - 1) ∣ (λ x : ℂ, c * x^17 + d * x^16 + 1)) : c = 987 :=
sorry

end find_c_l127_127516


namespace coloring_possible_l127_127949

variable (S : Finset (ℤ × ℤ))

theorem coloring_possible :
  ∃ f : (S : Set (ℤ × ℤ)) → {-1, 1}, ∀ (L : Finset (ℤ × ℤ)), 
    (∀ y ∈ L, (∃ k : ℤ, L = S.filter (λ p, p.2 = k)) ∨ (∃ k : ℤ, L = S.filter (λ p, p.1 = k))) → 
    ((Finset.sum L (λ p, f p)) ∈ ({-1, 0, 1} : Set ℤ)) :=
sorry

end coloring_possible_l127_127949


namespace three_equal_products_possible_l127_127611

-- Let's start by defining the problem conditions as follows:
def unique_numbers (M : matrix (fin 3) (fin 3) ℕ) : Prop :=
∀ i j, M i j ∈ finset.range 1 10 ∧ ∀ i j k l, (i ≠ k ∨ j ≠ l) → M i j ≠ M k l

-- Define a function to compute row and column products
def row_product (M : matrix (fin 3) (fin 3) ℕ) (i : fin 3) :=
(M i 0) * (M i 1) * (M i 2)

def col_product (M : matrix (fin 3) (fin 3) ℕ) (j : fin 3) :=
(M 0 j) * (M 1 j) * (M 2 j)

-- Finally, state that it is possible for three of these products to be the same
theorem three_equal_products_possible (M : matrix (fin 3) (fin 3) ℕ) 
  (hunique : unique_numbers M) : 
  ∃ (i1 i2 i3 : fin 3) (j1 j2 j3 : fin 3),
  row_product M i1 = row_product M i2 ∧ row_product M i2 = col_product M j1 ∧
  col_product M j1 = row_product M i3 ∧ row_product M i3 = col_product M j2 := 
sorry

end three_equal_products_possible_l127_127611


namespace transformed_quadratic_roots_l127_127608

-- Definitions of the conditions
def quadratic_roots (a b : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + b * x + 3 = 0 → (x = -2) ∨ (x = 3)

-- Statement of the theorem
theorem transformed_quadratic_roots (a b : ℝ) :
  quadratic_roots a b →
  ∀ x : ℝ, a * (x + 2)^2 + b * (x + 2) + 3 = 0 → (x = -4) ∨ (x = 1) :=
sorry

end transformed_quadratic_roots_l127_127608


namespace probability_even_sum_of_three_dice_l127_127725

-- Conditions
noncomputable def die_faces : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def even_faces : Set ℕ := {x ∈ die_faces | x % 2 = 0}
def odd_faces : Set ℕ := {x ∈ die_faces | x % 2 = 1}

-- Definitions of probabilities
noncomputable def prob_even : ℚ := 5 / 9
noncomputable def prob_odd : ℚ := 4 / 9

-- Question and Correct Answer
theorem probability_even_sum_of_three_dice : 
  (prob_even^3 + 3 * (prob_odd^2 * prob_even) + 3 * (prob_odd * (prob_even^2))) = 665 / 729 :=
by
  sorry

end probability_even_sum_of_three_dice_l127_127725


namespace minimum_BC_length_l127_127610

noncomputable theory

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

def is_triangle (A B C : V) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C

def angle_ABC_A120 (A B C : V) : Prop :=
  ∡ A B C = 120

def dot_product_neg_one (A B C : V) : Prop :=
  ⟪B - A, C - A⟫ = -1

theorem minimum_BC_length 
  {A B C : V} (h_ABC: is_triangle A B C) (h_angle: angle_ABC_A120 A B C) (h_dot: dot_product_neg_one A B C) : 
  ∃ x : ℝ, x = ‖B - C‖ ∧ x = sqrt 6 :=
sorry

end minimum_BC_length_l127_127610


namespace function_characterization_l127_127897

theorem function_characterization (f : ℤ → ℤ)
  (h1 : ∀ p : ℤ, prime p → f p > 0)
  (h2 : ∀ (x : ℤ) (p : ℤ), prime p → p ∣ ((f x + f p) ^ f p - x)) : 
  ∀ x : ℤ, f x = x :=
sorry

end function_characterization_l127_127897


namespace four_digit_integers_with_five_or_seven_l127_127225

theorem four_digit_integers_with_five_or_seven :
  let total_four_digit := 9000 in
  let without_five_or_seven := 7 * 8 * 8 * 8 in
  let with_five_or_seven := total_four_digit - without_five_or_seven in
  with_five_or_seven = 5416 :=
by
  let total_four_digit := 9000
  let without_five_or_seven := 7 * 8 * 8 * 8
  let with_five_or_seven := total_four_digit - without_five_or_seven
  show with_five_or_seven = 5416
  sorry

end four_digit_integers_with_five_or_seven_l127_127225


namespace aaron_carson_change_l127_127833

theorem aaron_carson_change :
  let total_money := 40 + 40 in
  let bill := (3 / 4 : ℝ) * total_money in
  let money_left_after_dinner := total_money - bill in
  let cost_per_scoop := (3 / 2 : ℝ) in
  let total_scoops := 6 + 6 in
  let total_ice_cream_cost := cost_per_scoop * total_scoops in
  let final_money_left := money_left_after_dinner - total_ice_cream_cost in
  let change_per_person := final_money_left / 2 in
  change_per_person = 1 := by
  let total_money := 40 + 40
  let bill := (3 / 4 : ℝ) * total_money
  let money_left_after_dinner := total_money - bill
  let cost_per_scoop := (3 / 2 : ℝ)
  let total_scoops := 6 + 6
  let total_ice_cream_cost := cost_per_scoop * total_scoops
  let final_money_left := money_left_after_dinner - total_ice_cream_cost
  let change_per_person := final_money_left / 2
  show change_per_person = 1, from sorry

end aaron_carson_change_l127_127833


namespace part_a_part_b_l127_127500

def trirectangular_angle (A B C : ℝ) : ℝ := A + B + C - Real.pi

theorem part_a
  (A B C : ℝ)
  (h : A + B + C > Real.pi) :
  trirectangular_angle A B C > 0 :=
sorry

theorem part_b
  (A B C A' A'' B' C' mu' mu'' : ℝ)
  (h_A : A' + A'' = A)
  (h_mu : mu' + mu'' = Real.pi)
  (h1 : trirectangular_angle A' B mu' > 0)
  (h2 : trirectangular_angle A'' C mu'' > 0) :
  trirectangular_angle A B C = trirectangular_angle A' B mu' + trirectangular_angle A'' C mu'' :=
sorry

end part_a_part_b_l127_127500


namespace calendar_sum_multiple_of_4_l127_127255

theorem calendar_sum_multiple_of_4 (a : ℕ) : 
  let top_left := a - 1
  let bottom_left := a + 6
  let bottom_right := a + 7
  top_left + a + bottom_left + bottom_right = 4 * (a + 3) :=
by
  sorry

end calendar_sum_multiple_of_4_l127_127255


namespace sum_lucky_tickets_divisible_by_13_l127_127442

-- Definition of a six-digit number
def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

-- Definition of a bus ticket number
def is_bus_ticket (A : ℕ) : Prop := A >= 0 ∧ A <= 999999

-- Definition of a lucky ticket
def is_lucky_ticket (A : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧ is_digit f ∧
  A = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧
  a + b + c = d + e + f

-- The statement to prove
theorem sum_lucky_tickets_divisible_by_13 :
  (∑ A in Finset.filter is_lucky_ticket (Finset.range 1000000), A) % 13 = 0 :=
by sorry

end sum_lucky_tickets_divisible_by_13_l127_127442


namespace number_of_ways_to_paint_circle_l127_127696

-- Define the sectors
inductive Sector
| A | B | C | D | E | F
deriving DecidableEq, Repr

-- Define the colors
inductive Color
| Red | Blue | Yellow
deriving DecidableEq, Repr

-- Adjacent sectors
def adjacent (s1 s2 : Sector) : Prop :=
  match s1, s2 with
  | Sector.A, Sector.B => true
  | Sector.A, Sector.F => true
  | Sector.B, Sector.C => true
  | Sector.C, Sector.D => true
  | Sector.D, Sector.E => true
  | Sector.E, Sector.F => true
  -- Bidirectional adjacency for completeness
  | Sector.B, Sector.A => true
  | Sector.F, Sector.A => true
  | Sector.C, Sector.B => true
  | Sector.D, Sector.C => true
  | Sector.E, Sector.D => true
  | Sector.F, Sector.E => true
  | _, _ => false
  end

-- Colors for sectors
structure Coloring :=
(sectorColor : Sector → Color)

def validColoring (c : Coloring) : Prop :=
  ∀ s₁ s₂, adjacent s₁ s₂ → c.sectorColor s₁ ≠ c.sectorColor s₂

-- Definition of the proof problem
theorem number_of_ways_to_paint_circle : ∃ (count : Nat), count = 24 ∧
  (∃ (colorings : List Coloring), colorings.length = count ∧
    ∀ coloring ∈ colorings, validColoring coloring) := 
sorry

end number_of_ways_to_paint_circle_l127_127696


namespace equal_play_time_l127_127824

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l127_127824


namespace each_player_plays_36_minutes_l127_127808

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l127_127808


namespace solve_for_x_l127_127528

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end solve_for_x_l127_127528


namespace no_100th_digit_4_l127_127739

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else (list.range (n + 1)).product

noncomputable def expression (n : ℕ) : ℕ :=
(factorial n * 5.factorial + factorial n * 3.factorial) / 2

theorem no_100th_digit_4 (n : ℕ) : 
  (∃ k, (expression n / 10^k) % 10 = 4) → false :=
by { sorry }

end no_100th_digit_4_l127_127739


namespace count_multiples_LCM_in_range_2000_3000_l127_127584

open Nat

theorem count_multiples_LCM_in_range_2000_3000 :
  let lcm_18_24_32 := Nat.lcm 18 (Nat.lcm 24 32)
  (count (fun n => 2000 ≤ n ∧ n ≤ 3000 ∧ lcm_18_24_32 ∣ n) (range 2001 1001)) = 4 := by
  let lcm_18_24 := Nat.lcm 18 24
  let lcm_18_24_32 := Nat.lcm lcm_18_24 32
  have h_eq_lcm: lcm_18_24_32 = 288 := by
    calc
      -- show calculation steps
      lcm_18_24_32 = lcm (lcm 18 24) 32 := rfl
      _ = lcm 72 32 := by conv_lhs { congr, congr, skip, congr, {rw mul_comm} }
      _ = 288 := by norm_num
  sorry

end count_multiples_LCM_in_range_2000_3000_l127_127584


namespace min_balls_cover_light_source_four_l127_127026

noncomputable theory

def min_balls_cover_light_source (n : ℕ) : Prop :=
∀ (source : Point) (balls : Fin n → Ball),
  ∀ (ray : Ray),
    intersects source ray balls

theorem min_balls_cover_light_source_four :
  min_balls_cover_light_source 4 :=
sorry

end min_balls_cover_light_source_four_l127_127026


namespace sum_of_roots_of_Q_l127_127019

theorem sum_of_roots_of_Q
    (Q : ℝ → ℝ)
    (hQ : ∃ (a b c : ℝ), ∀ x : ℝ, Q x = a * x^2 + b * x + c)
    (ineq : ∀ x : ℝ, Q (x^3 - x) ≥ Q (x^2 - 1)) :
    ∃ (a b c : ℝ), (Q = λ x, a * x^2 + b * x + c) ∧ (a ≠ 0) ∧ (b = -a) ∧ ((-b / a) = 1) :=
by
  sorry

end sum_of_roots_of_Q_l127_127019


namespace some_number_value_l127_127599

theorem some_number_value (a : ℕ) (some_number : ℕ) (h_a : a = 105)
  (h_eq : a ^ 3 = some_number * 25 * 35 * 63) : some_number = 7 := by
  sorry

end some_number_value_l127_127599


namespace equal_play_time_l127_127823

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l127_127823


namespace perpendicular_vectors_l127_127580

theorem perpendicular_vectors (x : ℝ) : (2 * x + 3 = 0) → (x = -3 / 2) :=
by
  intro h
  sorry

end perpendicular_vectors_l127_127580


namespace dog_tail_length_l127_127266

theorem dog_tail_length (b h t : ℝ) 
  (h_head : h = b / 6) 
  (h_tail : t = b / 2) 
  (h_total : b + h + t = 30) : 
  t = 9 :=
by
  sorry

end dog_tail_length_l127_127266


namespace problem_statement_l127_127609

noncomputable def sum_ints_from_60_to_80 : ℕ := ∑ i in finset.range (80 - 60 + 1), (60 + i)
noncomputable def num_even_ints_from_60_to_80 : ℕ := finset.card (finset.filter (λ i, i % 2 = 0) (finset.range (80 - 60 + 1).map (λ i, 60 + i)))

theorem problem_statement : sum_ints_from_60_to_80 + num_even_ints_from_60_to_80 = 1481 :=
by {
  sorry
}

end problem_statement_l127_127609


namespace find_b_l127_127362

theorem find_b (b : ℝ) :
  (∀ x : ℝ, 1 ≤ x → x ≤ b → f x = x ^ 2 - 2 * x + 2) ∧
  (∀ y : ℝ, 1 ≤ y → y ≤ b → (∃ x : ℝ, 1 ≤ x → x ≤ b → f x = y)) ↔ b = 2 :=
by
  sorry

end find_b_l127_127362


namespace raft_sticks_total_l127_127681

theorem raft_sticks_total : 
  let S := 45 
  let G := (3/5 * 45 : ℝ)
  let M := 45 + G + 15
  let D := 2 * M - 7
  S + G + M + D = 326 := 
by
  sorry

end raft_sticks_total_l127_127681


namespace problem_proof_l127_127960

-- Define the geometric sequence and vectors conditions
variables (a : ℕ → ℝ) (q : ℝ)
variables (h1 : ∀ n, a (n + 1) = q * a n)
variables (h2 : a 2 = a 2)
variables (h3 : a 3 = q * a 2)
variables (h4 : 3 * a 2 = 2 * a 3)

-- Statement to prove
theorem problem_proof:
  (a 2 + a 4) / (a 3 + a 5) = 2 / 3 :=
  sorry

end problem_proof_l127_127960


namespace bus_passing_time_l127_127441

noncomputable def time_for_bus_to_pass (bus_length : ℝ) (bus_speed_kph : ℝ) (man_speed_kph : ℝ) : ℝ :=
  let relative_speed_kph := bus_speed_kph + man_speed_kph
  let relative_speed_mps := (relative_speed_kph * (1000/3600))
  bus_length / relative_speed_mps

theorem bus_passing_time :
  time_for_bus_to_pass 15 40 8 = 1.125 :=
by
  sorry

end bus_passing_time_l127_127441


namespace equal_play_time_l127_127821

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l127_127821


namespace intersecting_circles_line_eq_l127_127221

theorem intersecting_circles_line_eq (A B : ℝ × ℝ)
  (h1 : A ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 + 4 * p.1 - 4 * p.2 = 0})
  (h2 : A ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 12 = 0})
  (h3 : B ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 + 4 * p.1 - 4 * p.2 = 0})
  (h4 : B ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 12 = 0}) :
  ∃ m c : ℝ, (m = -2 ∧ c = 6) ∧ ∀ p : ℝ × ℝ, p ∈ ({A, B} : set (ℝ × ℝ)) → (p.1 + m * p.2 + c = 0) :=
  by
    sorry

end intersecting_circles_line_eq_l127_127221


namespace tail_length_l127_127268

variable (Length_body Length_tail Length_head : ℝ)

-- Conditions
def tail_half_body (Length_tail Length_body : ℝ) := Length_tail = 1/2 * Length_body
def head_sixth_body (Length_head Length_body : ℝ) := Length_head = 1/6 * Length_body
def overall_length (Length_head Length_body Length_tail : ℝ) := Length_head + Length_body + Length_tail = 30

-- Theorem statement
theorem tail_length (h1 : tail_half_body Length_tail Length_body) 
                  (h2 : head_sixth_body Length_head Length_body) 
                  (h3 : overall_length Length_head Length_body Length_tail) : 
                  Length_tail = 6 := by
  sorry

end tail_length_l127_127268


namespace digit_sum_subtraction_l127_127366

theorem digit_sum_subtraction (P Q R S : ℕ) (hQ : Q + P = P) (hP : Q - P = 0) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10) (h4 : S < 10) : S = 0 := by
  sorry

end digit_sum_subtraction_l127_127366


namespace parabola_focus_distance_l127_127462

theorem parabola_focus_distance
  (p : ℝ) (h : p > 0)
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 = 3 - p / 2) 
  (h2 : x2 = 2 - p / 2)
  (h3 : y1^2 = 2 * p * x1)
  (h4 : y2^2 = 2 * p * x2)
  (h5 : y1^2 / y2^2 = x1 / x2) : 
  p = 12 / 5 := 
sorry

end parabola_focus_distance_l127_127462


namespace ranking_possibilities_l127_127914

theorem ranking_possibilities (A B C D E : Type) : 
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1 → n ≠ last)) →
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1)) →
  ∃ (positions : Finset (List ℕ)),
    positions.card = 54 :=
by
  sorry

end ranking_possibilities_l127_127914


namespace rain_on_all_three_days_l127_127709

theorem rain_on_all_three_days
    (P_F : ℝ) (P_S : ℝ) (P_Su : ℝ)
    (H1 : P_F = 0.4) 
    (H2 : P_S = 0.5) 
    (H3 : P_Su = 0.3) 
    (independence : true) : 
    P_F * P_S * P_Su = 0.06 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end rain_on_all_three_days_l127_127709


namespace perpendicular_line_to_plane_implies_perpendicular_line_to_any_line_in_plane_l127_127295

noncomputable theory

-- Declaring the basic elements in the Lean environment
variables {m n : Line} {α β : Plane}

-- The given conditions
axiom m_subset_α : m ⊆ α
axiom n_subset_β : n ⊆ β
axiom m_diff_n : m ≠ n
axiom α_diff_β : α ≠ β

-- Statement to prove
theorem perpendicular_line_to_plane_implies_perpendicular_line_to_any_line_in_plane
  (h1 : m ⊥ β) (h2 : n ⊆ β) : m ⊥ n :=
sorry

end perpendicular_line_to_plane_implies_perpendicular_line_to_any_line_in_plane_l127_127295


namespace toys_produced_per_week_l127_127774

-- Definitions corresponding to the conditions
def days_per_week : ℕ := 2
def toys_per_day : ℕ := 2170

-- Theorem statement corresponding to the question and correct answer
theorem toys_produced_per_week : days_per_week * toys_per_day = 4340 := 
by 
  -- placeholders for the proof steps
  sorry

end toys_produced_per_week_l127_127774


namespace intersection_complement_l127_127657

universe u

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 2}
def complement_U_B : Set ℤ := {x ∈ U | x ∉ B}

theorem intersection_complement :
  A ∩ complement_U_B = {0, 1} :=
by
  sorry

end intersection_complement_l127_127657


namespace shortest_hike_distance_l127_127459

-- Definitions for initial conditions
def hiker_initial_position : ℝ × ℝ := (0, -3)
def shelter_position : ℝ × ℝ := (6, -8)
def river_y_position : ℝ := 0

-- Function to calculate the Euclidean distance between two points (x1, y1) and (x2, y2)
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Problem statement: Prove the shortest distance the hiker must travel
theorem shortest_hike_distance :
  let hiker_reflected_position := (hiker_initial_position.1, -hiker_initial_position.2)
  in 3 + euclidean_distance hiker_reflected_position shelter_position = 3 + real.sqrt 157 :=
by
  sorry

end shortest_hike_distance_l127_127459


namespace triangle_proof_problem_l127_127248

-- The conditions and question programmed as a Lean theorem statement
theorem triangle_proof_problem
    (A B C : ℝ)
    (h1 : A > B)
    (S T : ℝ)
    (h2 : A = C)
    (K : ℝ)
    (arc_mid_A : K = A): -- K is midpoint of the arc A
    
    RS = K := sorry

end triangle_proof_problem_l127_127248


namespace largest_common_divisor_462_330_l127_127042

theorem largest_common_divisor_462_330 :
  ∃ d : ℕ, (d ∣ 462) ∧ (d ∣ 330) ∧
  (∀ k : ℕ, (k ∣ 462) → (k ∣ 330) → k ≤ d) ∧ d = 66 :=
by
  have prime_factors_462 : prime_factors 462 = [2, 3, 7, 11] :=
    sorry
  have prime_factors_330 : prime_factors 330 = [2, 3, 5, 11] :=
    sorry
  have common_factors := [2, 3, 11]
  have largest_common_divisor := 2 * 3 * 11
  use 66
  split
  sorry -- d ∣ 462 and d ∣ 330 proof
  split
  sorry -- d ∣ 330 proof
  split
  sorry -- d is the largest common factor proof
  refl -- d = 66

end largest_common_divisor_462_330_l127_127042


namespace exactly_two_sports_l127_127614

variables {U : Type} {A B C : Finset U}

def badminton_and_tennis (A B : Finset U) : ℕ := (A ∩ B).card
def badminton_and_soccer (A C : Finset U) : ℕ := (A ∩ C).card
def tennis_and_soccer (B C : Finset U) : ℕ := (B ∩ C).card

theorem exactly_two_sports (badminton tennis soccer : Finset U) 
  (total_members : ℕ) (badminton_card : ℕ) (tennis_card : ℕ) (soccer_card : ℕ) 
  (not_playing_any : ℕ) (badminton_tennis : ℕ) (badminton_soccer : ℕ) (tennis_soccer : ℕ) 
  (no_three_sports : (badminton ∩ tennis ∩ soccer).card = 0)
  (htotal_members : total_members = 60)
  (hbadminton_card : badminton_card = 25)
  (htennis_card : tennis_card = 32)
  (hsoccer_card : soccer_card = 14)
  (hnot_playing_any : not_playing_any = 5)
  (hbadminton_tennis : badminton_tennis = 10)
  (hbadminton_soccer : badminton_soccer = 8)
  (htennis_soccer : tennis_soccer = 6)
  : badminton_and_tennis badminton tennis + badminton_and_soccer badminton soccer + tennis_and_soccer tennis soccer = 24 :=
by
  rw [badminton_and_tennis, badminton_and_soccer, tennis_and_soccer]
  simp
  sorry

end exactly_two_sports_l127_127614


namespace upper_limit_of_set_A_is_31_l127_127674

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def set_A (upper_limit : ℕ) : set ℕ :=
  { n | 15 < n ∧ n ≤ upper_limit ∧ is_prime n }

theorem upper_limit_of_set_A_is_31 (A : set ℕ) (h1 : ∀ x, x ∈ A ↔ 15 < x ∧ x ≤ 31 ∧ is_prime x) (h2 : ∀ x ∈ A, ∃ y ∈ A, x - y = 14) :
  upper_limit 31 ∈ A := sorry

end upper_limit_of_set_A_is_31_l127_127674


namespace log2_50_between_5_and_6_l127_127384

theorem log2_50_between_5_and_6 (c d : ℤ) 
  (hlog32 : Real.logBase 2 32 = 5) 
  (hlog64 : Real.logBase 2 64 = 6)
  (hbound : 32 < 50 ∧ 50 < 64) 
  (hlog_bound : Real.logBase 2 32 < Real.logBase 2 50 ∧ Real.logBase 2 50 < Real.logBase 2 64) :
  c = 5 ∧ d = 6 → c + d = 11 := by
  sorry

end log2_50_between_5_and_6_l127_127384


namespace rotated_triangle_forms_two_cones_l127_127784

/-- Prove that the spatial geometric body formed when a right-angled triangle 
is rotated 360° around its hypotenuse is two cones. -/
theorem rotated_triangle_forms_two_cones (a b c : ℝ) (h1 : a^2 + b^2 = c^2) : 
  ∃ (cones : ℕ), cones = 2 :=
by
  sorry

end rotated_triangle_forms_two_cones_l127_127784


namespace xiao_hui_mother_second_purchase_l127_127506

def first_purchase_amount (x : ℝ) : ℝ := (5 / 8) * x
def second_purchase_amount (x : ℝ) : ℝ := 270 - (5 / 8) * x

theorem xiao_hui_mother_second_purchase (x : ℝ)
  (h1 : 230 ≤ x ∧ x < 320)
  (h2 : 13.5 / 0.05 = 270)
  (h3 : 0.1 * (270 + x) - 0.05 * x = 39.4) :
  second_purchase_amount x = 115 :=
begin
  -- proof placeholder
  sorry
end

end xiao_hui_mother_second_purchase_l127_127506


namespace speed_of_stream_l127_127777

theorem speed_of_stream (v : ℝ) (d : ℝ) :
  (∀ d : ℝ, d > 0 → (1 / (6 - v) = 2 * (1 / (6 + v)))) → v = 2 := by
  sorry

end speed_of_stream_l127_127777


namespace necessary_not_sufficient_l127_127433

theorem necessary_not_sufficient (x : ℝ) : (x^2 ≥ 1) ↔ (x ≥ 1 ∨ x ≤ -1) ≠ (x ≥ 1) :=
by
  sorry

end necessary_not_sufficient_l127_127433


namespace part1_part2_part3_l127_127566

open Real

-- Condition: Definition of f(x)
def f (x : ℝ) : ℝ := sin x + cos x

-- Question 1: Prove that f(π/2) = 1
theorem part1 : f (π / 2) = 1 :=
by
  sorry

-- Question 2: Prove that the smallest positive period of f(x) is 2π
theorem part2 : ∀ x : ℝ, f (x + 2 * π) = f x :=
by
  sorry

-- Condition: Definition of g(x)
def g (x : ℝ) : ℝ := f (x + π / 4) + f (x + 3 * π / 4)

-- Question 3: Prove that the minimum value of g(x) is -2
theorem part3 : ∀ x : ℝ, ∃ k : ℤ, g (x) = -2 ↔ x = 2 * k * π + 3 * π / 4 :=
by
  sorry

end part1_part2_part3_l127_127566


namespace original_expenditure_mess_l127_127063

theorem original_expenditure_mess : 
  ∀ (x : ℝ), 
  35 * x + 42 = 42 * (x - 1) + 35 * x → 
  35 * 12 = 420 :=
by
  intro x
  intro h
  sorry

end original_expenditure_mess_l127_127063


namespace total_carrots_l127_127232

/-- 
  If Pleasant Goat and Beautiful Goat each receive 6 carrots, and the other goats each receive 3 carrots, there will be 6 carrots left over.
  If Pleasant Goat and Beautiful Goat each receive 7 carrots, and the other goats each receive 5 carrots, there will be a shortage of 14 carrots.
  Prove the total number of carrots (n) is 45. 
--/
theorem total_carrots (X n : ℕ) 
  (h1 : n = 3 * X + 18) 
  (h2 : n = 5 * X) : 
  n = 45 := 
by
  sorry

end total_carrots_l127_127232


namespace find_f99_l127_127083

noncomputable def f : ℝ → ℝ := sorry

axiom periodicity : ∀ x : ℝ, f(x) * f(x + 2) = 2012
axiom initial_value : f(1) = 2

theorem find_f99 : f(99) = 1006 :=
sorry

end find_f99_l127_127083


namespace real_solutions_l127_127160

theorem real_solutions : 
  ∃ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ∧ (x = 7 ∨ x = -2) :=
sorry

end real_solutions_l127_127160


namespace Option_C_correct_l127_127412

theorem Option_C_correct (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = - x * y^2 :=
by
  sorry

end Option_C_correct_l127_127412


namespace collinear_points_b_value_l127_127886

theorem collinear_points_b_value :
  ∃ b : ℚ, let p1 := (4, -10 : ℚ)
           let p2 := (-b + 4, 6 : ℚ)
           let p3 := (3b + 6, 4 : ℚ)
           (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1) ∧
           b = -16 / 31 :=
by
  sorry

end collinear_points_b_value_l127_127886


namespace equal_playing_time_l127_127816

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l127_127816


namespace taxi_ride_distance_l127_127446

-- Define the conditions as per the problem statement
def total_charge : ℝ := 19.1
def initial_charge : ℝ := 3.5
def increment_charge : ℝ := 0.4

-- Calculate the distance based on the given charges
def calc_distance (total_charge initial_charge increment_charge : ℝ) : ℝ :=
  let remaining_charge := total_charge - initial_charge
  let increments := remaining_charge / increment_charge
  let total_increments := increments + 1
  total_increments * (1/5)

-- The theorem to prove the problem statement
theorem taxi_ride_distance :
  calc_distance total_charge initial_charge increment_charge = 8 := by
  sorry

end taxi_ride_distance_l127_127446


namespace each_player_plays_36_minutes_l127_127803

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l127_127803


namespace triangle_equilateral_iff_equal_inradii_l127_127654

variables {A B C A₁ B₁ C₁ M : Point}
variable {Triangle : Point → Point → Point → Prop}
variable {inscribed_circle : Point → Point → Point → ℝ → Prop}
variable {angle_bisector : Point → Point → Point → Prop}

-- Given triangle ABC with angle bisectors AA₁, BB₁, CC₁.
def bisector_intersection (ABC : Triangle A B C) (AA₁ : angle_bisector A A₁ B C) 
  (BB₁ : angle_bisector B B₁ A C) (CC₁ : angle_bisector C C₁ A B) (common_point : Point) : Prop :=
  let M := common_point
  M = intersection (AA₁ ∧ BB₁ ∧ CC₁)

-- Triangles formed and their inscribed circle with equal radii for four out of the six.
def triangles_with_equal_inradii (triangle: Triangle A B C) (common_point : Point) : Prop :=
  let M := common_point in
  ∃ r: ℝ, 
  let T1 := Triangle A M B₁,
  let T2 := Triangle B M A₁,
  let T3 := Triangle C M A₁,
  let T4 := Triangle A M C₁,
  let T5 := Triangle B M C₁,
  let T6 := Triangle C M B₁ in
  inscribed_circle A M B₁ r ∧ inscribed_circle B M A₁ r ∧ 
  inscribed_circle C M A₁ r ∧ inscribed_circle A M C₁ r

-- Proving the triangle ABC is equilateral
theorem triangle_equilateral_iff_equal_inradii (ABC: Triangle A B C) (common_point: Point)
  (h_bisectors: bisector_intersection ABC (angle_bisector A A₁ B C) (angle_bisector B B₁ A C) (angle_bisector C C₁ A B) common_point)
  (h_equal_inradii: triangles_with_equal_inradii ABC common_point):
  AB = BC ∧ BC = CA :=
sorry

end triangle_equilateral_iff_equal_inradii_l127_127654


namespace base9_to_base3_conversion_726_l127_127493

theorem base9_to_base3_conversion_726 : ∀ (n : ℕ), (n = 726) → (726₉ = 210220₃) := by
  sorry

end base9_to_base3_conversion_726_l127_127493


namespace grant_total_earnings_l127_127997

def first_month (X : ℝ) : ℝ := X

def second_month (X : ℝ) (Y : ℝ) : ℝ := 
  let first_earnings := X
  let second_earnings := (first_earnings * first_earnings) * (1 + Y / 100)
  second_earnings

noncomputable def third_month (X : ℝ) (second_earnings : ℝ) (Z : ℝ) : ℝ := 
  let product_earnings := X * second_earnings
  let third_earnings := Real.log2(product_earnings) * Z
  third_earnings

noncomputable def fourth_month (X : ℝ) (second_earnings : ℝ) (third_earnings : ℝ) (W : ℝ) : ℝ := 
  let product_earnings := X * second_earnings * third_earnings
  let geometric_mean := Real.cbrt(product_earnings)
  geometric_mean + W

noncomputable def total_earnings (X : ℝ) (Y : ℝ) (Z : ℝ) (W : ℝ) : ℝ := 
  let earnings1 := first_month X
  let earnings2 := second_month X Y
  let earnings3 := third_month X earnings2 Z
  let earnings4 := fourth_month X earnings2 earnings3 W
  earnings1 + earnings2 + earnings3 + earnings4

theorem grant_total_earnings :
  total_earnings 350 10 20 50 = 137549.94 :=
by
  sorry

end grant_total_earnings_l127_127997


namespace cos_105_eq_fraction_l127_127872

theorem cos_105_eq_fraction : 
  cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  have h_cos_45 : cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.cos_eq_sqrt_4_inv_norm_eq_sqrt_two_div_two]
  have h_cos_60 : cos (60 * Real.pi / 180) = 1 / 2 :=
    by norm_num [Real.cos_pi_div_three]
  have h_sin_45 : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.sin_eq_sqrt_4_inv_sqrt_two_div_two]
  have h_sin_60 : sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by norm_num [Real.sin_pi_div_three]
  sorry

end cos_105_eq_fraction_l127_127872


namespace subset_G_exists_l127_127276

open Set

noncomputable def sequence_of_points (A : ℕ → ℝ × ℝ) : Prop :=
∀ n : ℕ, ∀ m : ℕ, (n ≠ m) → (dist (A n) (A m) > 2^n + 2^m)

noncomputable def subset_G_property (A : ℕ → ℝ × ℝ) (G : Set (ℝ × ℝ)) (R a b : ℝ) : Prop :=
(measure G ≥ R) ∧ (∀ (P : ℝ × ℝ), P ∈ G → (a < (∑' n, 1 / dist P (A n)) ∧ (∑' n, 1 / dist P (A n)) < b))

theorem subset_G_exists {A : ℕ → ℝ × ℝ} {R a b : ℝ} (hA : sequence_of_points A) (ha_lt_b : a < b) :
  ∃ G : Set (ℝ × ℝ), subset_G_property A G R a b :=
sorry

end subset_G_exists_l127_127276


namespace solve_for_y_l127_127228

variable (y : ℚ)

-- Conditions from part a) defining the arithmetic sequence
def arithmetic_seq_condition : Prop :=
  (y - 2) - (1 / 3) = (4y) - (y - 2)

-- The theorem to prove the value of y
theorem solve_for_y (h : arithmetic_seq_condition y) : y = -13 / 6 :=
  sorry

end solve_for_y_l127_127228


namespace product_of_roots_is_four_l127_127044

noncomputable def quadratic_eq := -2 * (x:ℝ) ^ 2 - 6 * x + 8

theorem product_of_roots_is_four : 
  let a : ℝ := -2
  let b : ℝ := -6
  let c : ℝ := 8
  ∃ α β : ℝ, quadratic_eq α = 0 ∧ quadratic_eq β = 0 ∧ α * β = 4 := 
by 
  sorry

end product_of_roots_is_four_l127_127044


namespace gillian_phone_bill_l127_127939

variable (original_monthly_bill : ℝ) (increase_percentage : ℝ) (months_per_year : ℕ)

def annual_phone_bill_after_increase (bill : ℝ) (increase : ℝ) (months : ℕ) : ℝ :=
  bill * (1 + increase / 100) * months

theorem gillian_phone_bill (h1 : original_monthly_bill = 50)
  (h2 : increase_percentage = 10)
  (h3 : months_per_year = 12) :
  annual_phone_bill_after_increase original_monthly_bill increase_percentage months_per_year = 660 := by
  sorry

end gillian_phone_bill_l127_127939


namespace cos_105_eq_fraction_l127_127869

theorem cos_105_eq_fraction : 
  cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  have h_cos_45 : cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.cos_eq_sqrt_4_inv_norm_eq_sqrt_two_div_two]
  have h_cos_60 : cos (60 * Real.pi / 180) = 1 / 2 :=
    by norm_num [Real.cos_pi_div_three]
  have h_sin_45 : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.sin_eq_sqrt_4_inv_sqrt_two_div_two]
  have h_sin_60 : sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by norm_num [Real.sin_pi_div_three]
  sorry

end cos_105_eq_fraction_l127_127869


namespace polynomial_division_remainder_l127_127279

theorem polynomial_division_remainder (Q : ℝ → ℝ) :
  (Q 10 = 8) → 
  (Q 14 = 2) → 
  ∃ c d : ℝ, (∀ x : ℝ, Q(x) = (x-10)*(x-14)*R(x) + c*x + d) ∧
             (c = -3 / 2) ∧
             (d = 23) :=
by {
  sorry
}

end polynomial_division_remainder_l127_127279


namespace A_25_value_l127_127273

def f (x : ℝ) : ℝ := x^5 - 3 * x^3 + 2 * x^2 + 3 * x + 6

def A (n : ℕ) : ℝ :=
  ∏ k in Finset.range n, (4 * (k + 1) - 3) * f (4 * (k + 1) - 3) / ((4 * (k + 1) - 1) * f (4 * (k + 1) - 1))

theorem A_25_value : A 25 = 1 / 1000001 := 
  sorry

end A_25_value_l127_127273


namespace solution_set_of_inequality_l127_127021

theorem solution_set_of_inequality :
  {x : ℝ // (2 < x ∨ x < 2) ∧ x ≠ 3} =
  {x : ℝ // x < 2 ∨ 3 < x } :=
sorry

end solution_set_of_inequality_l127_127021


namespace largest_stamps_per_page_l127_127634

-- Definitions of the conditions
def stamps_book1 : ℕ := 1260
def stamps_book2 : ℕ := 1470

-- Statement to be proven: The largest number of stamps per page (gcd of 1260 and 1470)
theorem largest_stamps_per_page : Nat.gcd stamps_book1 stamps_book2 = 210 :=
by
  sorry

end largest_stamps_per_page_l127_127634


namespace find_a_l127_127565

theorem find_a (a : ℝ) (h : a < 0) : a = -5/3 :=
  let i : ℂ := complex.I
  let z : ℂ := 3 * a * i / (1 - 2 * i)
  have hz : complex.abs(z) = real.sqrt 5 := by sorry
  sorry

end find_a_l127_127565


namespace percent_profit_is_16_l127_127060

variables (C S : ℝ)

-- Define the condition
def condition := 58 * C = 50 * S

-- Define the profit
def profit := S - C

-- Define the profit percent
def profit_percent := (profit / C) * 100

-- The goal is to prove the profit percent is 16 when the condition is satisfied
theorem percent_profit_is_16 (h : condition C S) : profit_percent C S = 16 :=
by
  sorry

end percent_profit_is_16_l127_127060


namespace not_perfect_square_l127_127297

theorem not_perfect_square (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : ¬ (a^2 - b^2) % 4 = 0) : 
  ¬ ∃ k : ℤ, (a + 3*b) * (5*a + 7*b) = k^2 :=
sorry

end not_perfect_square_l127_127297


namespace vector_dot_triple_product_l127_127284

-- Definitions of vectors
variables {𝕜 : Type*} [Field 𝕜] [Module 𝕜 (Fin 3 → 𝕜)] [Fintype (Fin 3)]
variables (u v w : (Fin 3 → 𝕜))

-- Conditions
def unit_vector (a : Fin 3 → 𝕜) : Prop := ∥a∥ = 1
def condition_1 : Prop := unit_vector u ∧ unit_vector v
def condition_2 : Prop := 2 • (u ×ₗ v) + u = w
def condition_3 : Prop := w ×ₗ u = 3 • v

-- The theorem to prove
theorem vector_dot_triple_product : condition_1 u v ∧ condition_2 u v w ∧ condition_3 u v w → 
  (u ⬝ (v ×ₗ w)) = 2 :=
by
  intro h
  sorry

end vector_dot_triple_product_l127_127284


namespace rhombus_diagonal_length_l127_127097

theorem rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) (d1 : ℝ)
  (h_area : area = 432) 
  (h_d2 : d2 = 24) :
  d1 = 36 :=
by
  sorry

end rhombus_diagonal_length_l127_127097


namespace square_area_l127_127010

def dist (A B : (ℝ × ℝ)) : ℝ :=
  (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))

def area_of_square (A B : (ℝ × ℝ)) : ℝ :=
  let side := dist A B
  side * side

theorem square_area (A B : (ℝ × ℝ)) (h : A = (0, 3) ∧ B = (4, 0)) : 
  area_of_square A B = 25 :=
by
  sorry

end square_area_l127_127010


namespace value_of_m_minus_n_l127_127229

theorem value_of_m_minus_n (m n : ℝ) (h : (sqrt (1 - m))^2 + |n + 2| = 0) : m - n = 3 := by
  sorry

end value_of_m_minus_n_l127_127229


namespace units_digit_of_42_pow_3_add_24_pow_3_l127_127405

theorem units_digit_of_42_pow_3_add_24_pow_3 :
    (42 ^ 3 + 24 ^ 3) % 10 = 2 :=
by
    have units_digit_42 := (42 % 10 = 2)
    have units_digit_24 := (24 % 10 = 4)
    sorry

end units_digit_of_42_pow_3_add_24_pow_3_l127_127405


namespace min_period_f_max_value_f_l127_127214

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * sin x * cos x + 1

-- Statement to prove the minimum positive period of f(x)
theorem min_period_f : ∀ x : ℝ, f (x) = f (x + π) :=
by sorry

-- Statement to prove the maximum value of f(x)
theorem max_value_f : ∃ x : ℝ, f x = 2 :=
by sorry

end min_period_f_max_value_f_l127_127214


namespace find_k_l127_127301

theorem find_k 
  (a b : ℝ) 
  (x : ℝ) 
  (h1 : tan x = 2 * a / b) 
  (h2 : tan (2 * x) = 2 * b / (a + 2 * b)) 
: 
  let k := (-3 + Real.sqrt 13) / 2
  in x = Real.arctan k :=
sorry

end find_k_l127_127301


namespace Option_C_correct_l127_127413

theorem Option_C_correct (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = - x * y^2 :=
by
  sorry

end Option_C_correct_l127_127413


namespace exists_marked_sum_of_three_l127_127613

theorem exists_marked_sum_of_three (s : Finset ℕ) (h₀ : s.card = 22) (h₁ : ∀ x ∈ s, x ≤ 30) :
  ∃ a ∈ s, ∃ b ∈ s, ∃ c ∈ s, ∃ d ∈ s, a = b + c + d :=
by
  sorry

end exists_marked_sum_of_three_l127_127613


namespace determine_width_of_first_2_walls_l127_127511

variable (width_of_remaining_walls : ℕ) (width_of_first_2_walls : ℕ)
variable (total_masking_tape_needed : ℕ)
variable (width_of_each_remaining_wall : ℕ)
variable (num_remaining_walls : ℕ)
variable (num_first_2_walls : ℕ)

axiom walls_condition : num_remaining_walls = 2 ∧ width_of_each_remaining_wall = 6
axiom tape_condition : total_masking_tape_needed = 20
axiom first_2_walls_condition : width_of_first_2_walls = (total_masking_tape_needed - num_remaining_walls * width_of_each_remaining_wall) / num_first_2_walls

theorem determine_width_of_first_2_walls :
  width_of_first_2_walls = 4 :=
by
  have num_remaining_walls_val : num_remaining_walls = 2 := walls_condition.left
  have width_of_each_remaining_wall_val : width_of_each_remaining_wall = 6 := walls_condition.right
  have total_tape_val : total_masking_tape_needed = 20 := tape_condition
  have width_of_first_2_walls_val : width_of_first_2_walls = 4 :=
    calc
      width_of_first_2_walls
      = (20 - 2 * 6) / 2 : by rw [total_tape_val, num_remaining_walls_val, width_of_each_remaining_wall_val]
      ... = (20 - 12) / 2 : by norm_num
      ... = 8 / 2 : by norm_num
      ... = 4 : by norm_num
  exact width_of_first_2_walls_val

end determine_width_of_first_2_walls_l127_127511


namespace poly_expansion_l127_127045

theorem poly_expansion (a b : ℝ) (n : ℕ) (h : n > 0) : 
  (a - b) * (∑ i in range (n + 1), a^(n - i) * b^i) = a^(n + 1) - b^(n + 1) :=
sorry

end poly_expansion_l127_127045


namespace negation_proposition_l127_127371

theorem negation_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 + x_0 - 2 < 0) ↔ ∀ x_0 : ℝ, x_0^2 + x_0 - 2 ≥ 0 :=
by
  sorry

end negation_proposition_l127_127371


namespace triangle_is_acute_l127_127193

theorem triangle_is_acute (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let AB := real.sqrt (a^2 + b^2),
      BC := real.sqrt (b^2 + c^2),
      AC := real.sqrt (a^2 + c^2) in
  (AB^2 + AC^2 > BC^2) ∧
  (AB^2 + BC^2 > AC^2) ∧
  (BC^2 + AC^2 > AB^2) :=
by
  sorry

end triangle_is_acute_l127_127193


namespace simplify_expression_l127_127172

theorem simplify_expression (x y z : ℝ) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (hx2 : x ≠ 2) (hy3 : y ≠ 3) (hz5 : z ≠ 5) :
  ( ( (x - 2) / (3 - z) * ( (y - 3) / (5 - x) ) * ( (z - 5) / (2 - y) ) ) ^ 2 ) = 1 :=
by
  sorry

end simplify_expression_l127_127172


namespace polynomial_existence_l127_127148

theorem polynomial_existence :
  ∃ (a : Fin 20 → ℝ), (∀ i, 0 < a i) ∧
    (∀ x : ℝ, (∑ i in Finset.range 20, a i * x ^ (19 - i) + x ^ 20) ≠ 0) ∧
    (∀ i j : Fin 20, i ≠ j → ∃ x : ℝ, (∑ k in Finset.range 20, a (if k = i then j else if k = j then i else k) * x ^ (19 - k) + x ^ 20) = 0) :=
sorry

end polynomial_existence_l127_127148


namespace number_of_values_of_x_l127_127928

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (169 - real.cbrt x)

theorem number_of_values_of_x : 
  let S := {y : ℕ | y ≤ 13} in
  ∃ (n : ℕ), n = S.card := 
sorry

end number_of_values_of_x_l127_127928


namespace subset_intersection_complement_empty_l127_127219

open Set

variable (U : Type) (A B : Set U)

theorem subset_intersection_complement_empty
  (hU : U = ℝ)
  (h : A ⊆ (A ∩ B)) :
  A ∩ (U \ B) = ∅ :=
sorry

end subset_intersection_complement_empty_l127_127219


namespace base9_to_base3_conversion_726_l127_127494

theorem base9_to_base3_conversion_726 : ∀ (n : ℕ), (n = 726) → (726₉ = 210220₃) := by
  sorry

end base9_to_base3_conversion_726_l127_127494


namespace train_speed_l127_127829

theorem train_speed (train_length : ℕ) (bridge_length : ℕ) (crossing_time : ℕ) 
  (h_train_length : train_length = 120) 
  (h_bridge_length : bridge_length = 480) 
  (h_crossing_time : crossing_time = 55) : 
  (train_length + bridge_length) / crossing_time = 10.91 :=
by {
  rw [h_train_length, h_bridge_length, h_crossing_time],
  -- Lean will not be able to handle non-rational division precisely, so we provide a proof sketch here.
  -- In a real setting, we would need to handle floating points properly.
  sorry
}

end train_speed_l127_127829


namespace distance_between_andrey_and_valentin_l127_127116

-- Definitions based on conditions
def speeds_relation_andrey_boris (a b : ℝ) := b = 0.94 * a
def speeds_relation_boris_valentin (b c : ℝ) := c = 0.95 * b

theorem distance_between_andrey_and_valentin
  (a b c : ℝ)
  (h1 : speeds_relation_andrey_boris a b)
  (h2 : speeds_relation_boris_valentin b c)
  : 1000 - 1000 * c / a = 107 :=
by
  sorry

end distance_between_andrey_and_valentin_l127_127116


namespace average_first_last_numbers_l127_127373

theorem average_first_last_numbers :
  ∀ (l : List ℝ), l = [-2, 4, 6, 9, 12] →
  (∃ (f l : ℝ), f ≠ 12 ∧ l ≠ -2 ∧
    4 ≤ l.length ∧
    6 ≠ List.head l ∧ 6 ≠ List.last l (by simp [l])) →
  (f + l) / 2 = 6.5 :=
by
  intro l hl h
  -- This part would contain steps of the proof if required, skipping with 'sorry' as per the prompt
  sorry

end average_first_last_numbers_l127_127373


namespace area_ratio_l127_127730

theorem area_ratio (A H I B C D E F G : Point) 
  (h1 : EquilateralTriangle A H I)
  (h2 : Parallel BC HI ∧ Parallel DE HI ∧ Parallel FG HI)
  (h3 : dist A B = dist B D ∧ dist B D = dist D F ∧ dist D F = dist F H) :
  (area_trapezoid F G I H) / (area_triangle A H I) = 7 / 16 := 
sorry

end area_ratio_l127_127730


namespace annual_income_of_A_l127_127423

theorem annual_income_of_A :
  let C_income := 14000
  let B_income := 1.12 * C_income
  let A_income := (5 / 2) * B_income
  A_income * 12 = 470400 :=
by
  let C_income := 14000
  let B_income := 1.12 * C_income
  let A_income := (5 / 2) * B_income
  calc
    A_income * 12 = ((5 / 2) * B_income) * 12 : by rw [←mul_assoc]
                ... = ((5 / 2) * (1.12 * C_income)) * 12 : by rw B_income
                ... = ((5 / 2) * (1.12 * 14000)) * 12 : by rw C_income
                ... = 39200 * 12 : by norm_num
                ... = 470400 : by norm_num

end annual_income_of_A_l127_127423


namespace parallel_sides_l127_127245

-- Let ABCD be a convex quadrilateral
variable {P : Type} [EuclideanGeometry P]
variables (A B C D M N M' N' : P)

-- Define midpoints M and N
variable (H1 : Midpoint A C M)
variable (H2 : Midpoint B D N)

-- Define intersections M' and N'
variable (H3 : IntersectSegment MN A B M')
variable (H4 : IntersectSegment MN C D N')

-- Define the condition MM' = NN'
variable (H5 : dist M M' = dist N N')

-- Define the convexity of the quadrilateral
variable (H6 : ConvexQuadrilateral A B C D)

-- Prove that BC is parallel to AD
theorem parallel_sides : Parallel (Line.mk B C) (Line.mk A D) :=
  sorry

end parallel_sides_l127_127245


namespace find_number_l127_127761

theorem find_number (x : ℝ) (h : 61 + 5 * 12 / (180 / x) = 62): x = 3 :=
by
  sorry

end find_number_l127_127761


namespace determine_a_eq_pm1_l127_127568

def f (a : ℝ) (x : ℝ) : ℝ := cos (2 * x) + a * sin x

theorem determine_a_eq_pm1 (a : ℝ) (n : ℕ) (hn_pos : 0 < n) 
  (h_zeros : ∃ n_zero : ℝ → Prop, (∀ x, f a x = 0 ↔ n_zero x) ∧ 
     finset.card (finset.filter n_zero (finset.range n)) = 9) : 
  a = 1 ∨ a = -1 := 
sorry

end determine_a_eq_pm1_l127_127568


namespace corrected_mean_l127_127704

theorem corrected_mean (n : ℕ) (mean : ℝ) (incorrect_val correct_val : ℝ)
  (h_n : n = 50)
  (h_mean : mean = 36)
  (h_incorrect_val : incorrect_val = 21)
  (h_correct_val : correct_val = 48) :
  let initial_sum := n * mean in
  let corrected_sum := initial_sum - incorrect_val + correct_val in
  corrected_sum / n = 36.54 :=
by
  sorry

end corrected_mean_l127_127704


namespace quadratic_root_signs_l127_127697

-- Variables representation
variables {x m : ℝ}

-- Given: The quadratic equation with one positive root and one negative root
theorem quadratic_root_signs (h : ∃ a b : ℝ, 2*a*2*b + (m+1)*(a + b) + m = 0 ∧ a > 0 ∧ b < 0) : 
  m < 0 := 
sorry

end quadratic_root_signs_l127_127697


namespace new_trailer_homes_count_l127_127028

theorem new_trailer_homes_count :
  let old_trailers : ℕ := 30
  let old_avg_age : ℕ := 15
  let years_since : ℕ := 3
  let new_avg_age : ℕ := 10
  let total_age := (old_trailers * (old_avg_age + years_since)) + (3 * new_trailers)
  let total_trailers := old_trailers + new_trailers
  let total_avg_age := total_age / total_trailers
  total_avg_age = new_avg_age → new_trailers = 34 :=
by
  sorry

end new_trailer_homes_count_l127_127028


namespace exists_100_quadratic_trinomials_l127_127887

noncomputable def quadratic_polynomial (n : ℕ) : ℝ → ℝ := 
  λ x, (x - 4 * (n : ℝ))^2 - 1

def has_two_distinct_roots (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, a ≠ b ∧ f a = 0 ∧ f b = 0

def sums_of_roots_non_integer (fs : ℕ → ℝ → ℝ) (n m : ℕ) (h_n : n < 101) (h_m : m < 101) (h_nm : n ≠ m) : Prop := 
  ∀ x y, fs n x = 0 → fs m y = 0 → (x + y) ∉ ℤ

theorem exists_100_quadratic_trinomials :
  ∃ fs : ℕ → ℝ → ℝ, 
    (∀ n : ℕ, n < 101 → fs n = quadratic_polynomial n) ∧ 
    (∀ n : ℕ, n < 101 → has_two_distinct_roots (fs n)) ∧ 
    (∀ n m : ℕ, n < 101 → m < 101 → n ≠ m → sums_of_roots_non_integer fs n m (by assumption) (by assumption) (by assumption)) :=
begin
  sorry
end

end exists_100_quadratic_trinomials_l127_127887


namespace find_real_solutions_l127_127159

theorem find_real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 7 ∨ x = -2) := 
by
  sorry

end find_real_solutions_l127_127159


namespace sprockets_produced_l127_127658

variables {T : ℝ}

theorem sprockets_produced (T : ℝ) : 
  let sprockets_per_hour_X := 5.999999999999999 in
  let sprockets_per_hour_B := 6.6 in
  let time_X := T + 10 in
  let time_B := T in
  (6 * time_X + 6.6 * time_B) = 1320 :=
by
  sorry

end sprockets_produced_l127_127658


namespace range_of_a_min_value_ab_range_of_y_l127_127438
-- Import the necessary Lean library 

-- Problem 1
theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) → (-2 ≤ a ∧ a ≤ 1) := 
sorry

-- Problem 2
theorem min_value_ab (a b : ℝ) (h₁ : a + b = 1) : 
  (∀ x, |x - 1| + |x - 3| ≥ a^2 + a) → 
  (min ((1 : ℝ) / (4 * |b|) + |b| / a) = 3 / 4 ∧ (a = 2)) :=
sorry

-- Problem 3
theorem range_of_y (a : ℝ) (y : ℝ) (h₁ : a ∈ Set.Ici (2 : ℝ)) : 
  y = (2 * a) / (a^2 + 1) → 0 < y ∧ y ≤ (4 / 5) :=
sorry

end range_of_a_min_value_ab_range_of_y_l127_127438


namespace part1_part2_l127_127205

section ProofProblem

variable {f : ℝ → ℝ}
variable {a : ℝ}

def fx : ℝ → ℝ := fun x => x^3 - a * x
def dfx : ℝ → ℝ := fun x => 3 * x^2 - a

-- (1) Proving that a = 3 given f(x) has a local minimum at x = 1
theorem part1 : (dfx 1 = 0) → a = 3 := by
    sorry

-- (2) Use the method of contradiction to prove the given statement
theorem part2 (h : a = 3) : (∀ x : ℝ, 0 < x → -2 * fx x / x^2 < sqrt 3 ∧ dfx x / x < sqrt 3) → False := by
    sorry

end ProofProblem

end part1_part2_l127_127205


namespace all_edges_same_color_l127_127085

-- Define the vertices in the two pentagons and the set of all vertices
inductive vertex
| A1 | A2 | A3 | A4 | A5 | B1 | B2 | B3 | B4 | B5
open vertex

-- Predicate to identify edges between vertices
def edge (v1 v2 : vertex) : Prop :=
  match (v1, v2) with
  | (A1, A2) | (A2, A3) | (A3, A4) | (A4, A5) | (A5, A1) => true
  | (B1, B2) | (B2, B3) | (B3, B4) | (B4, B5) | (B5, B1) => true
  | (A1, B1) | (A1, B2) | (A1, B3) | (A1, B4) | (A1, B5) => true
  | (A2, B1) | (A2, B2) | (A2, B3) | (A2, B4) | (A2, B5) => true
  | (A3, B1) | (A3, B2) | (A3, B3) | (A3, B4) | (A3, B5) => true
  | (A4, B1) | (A4, B2) | (A4, B3) | (A4, B4) | (A4, B5) => true
  | (A5, B1) | (A5, B2) | (A5, B3) | (A5, B4) | (A5, B5) => true
  | _ => false

-- Edge coloring predicate 'black' or 'white'
inductive color
| black | white
open color

def edge_color (v1 v2 : vertex) : color → Prop :=
  sorry -- Coloring function needs to be defined accordingly

-- Predicate to check for monochrome triangles
def no_monochrome_triangle : Prop :=
  ∀ v1 v2 v3 : vertex,
    (edge v1 v2 ∧ edge v2 v3 ∧ edge v3 v1) →
    ¬ (∃ c : color, edge_color v1 v2 c ∧ edge_color v2 v3 c ∧ edge_color v3 v1 c)

-- Main theorem statement
theorem all_edges_same_color (no_mt : no_monochrome_triangle) :
  ∃ c : color, ∀ v1 v2 : vertex,
    (edge v1 v2 ∧ (v1 = A1 ∨ v1 = A2 ∨ v1 = A3 ∨ v1 = A4 ∨ v1 = A5) ∧
                 (v2 = A1 ∨ v2 = A2 ∨ v2 = A3 ∨ v2 = A4 ∨ v2 = A5) ) →
    edge_color v1 v2 c ∧
    (edge v1 v2 ∧ (v1 = B1 ∨ v1 = B2 ∨ v1 = B3 ∨ v1 = B4 ∨ v1 = B5) ∧
                 (v2 = B1 ∨ v2 = B2 ∨ v2 = B3 ∨ v2 = B4 ∨ v2 = B5) ) →
    edge_color v1 v2 c := sorry

end all_edges_same_color_l127_127085


namespace fruit_problem_l127_127058

variables (A O x : ℕ) -- Natural number variables for apples, oranges, and oranges put back

theorem fruit_problem :
  (A + O = 10) ∧
  (40 * A + 60 * O = 480) ∧
  (240 + 60 * (O - x) = 45 * (10 - x)) →
  A = 6 ∧ O = 4 ∧ x = 2 :=
  sorry

end fruit_problem_l127_127058


namespace expression_is_nonnegative_l127_127299

noncomputable def expression_nonnegative (a b c d e : ℝ) : Prop :=
  (a - b) * (a - c) * (a - d) * (a - e) +
  (b - a) * (b - c) * (b - d) * (b - e) +
  (c - a) * (c - b) * (c - d) * (c - e) +
  (d - a) * (d - b) * (d - c) * (d - e) +
  (e - a) * (e - b) * (e - c) * (e - d) ≥ 0

theorem expression_is_nonnegative (a b c d e : ℝ) : expression_nonnegative a b c d e := 
by 
  sorry

end expression_is_nonnegative_l127_127299


namespace blocks_differ_in_two_ways_l127_127448

theorem blocks_differ_in_two_ways :
  let materials := 3
  let sizes := 3
  let colors := 4
  let shapes := 4
  let finishes := 2
  let generating_function := (1 + 2 * X) * (1 + 2 * X) * (1 + 3 * X) * (1 + 3 * X) * (1 + X)
  let expanded_function := 1 + 6 * X + 18 * X^2 + O(X^3)
  in
  coeff X^2 expanded_function = 18 :=
by {
  -- Definitions
  let materials := 3,
  let sizes := 3,
  let colors := 4,
  let shapes := 4,
  let finishes := 2,
  let generating_function := (1 + 2 * X) * (1 + 2 * X) * (1 + 3 * X) * (1 + 3 * X) * (1 + X),
  let expanded_function := 1 + 6 * X + 18 * X^2 + O(X^3),

  -- Assertion
  have : coeff X^2 expanded_function = 18 := by sorry,
  exact this,
}

end blocks_differ_in_two_ways_l127_127448


namespace ethanol_in_fuel_A_l127_127834

def fuel_tank_volume : ℝ := 208
def fuel_A_volume : ℝ := 82
def fuel_B_volume : ℝ := fuel_tank_volume - fuel_A_volume
def ethanol_in_fuel_B : ℝ := 0.16
def total_ethanol : ℝ := 30

theorem ethanol_in_fuel_A 
  (x : ℝ) 
  (H_fuel_tank_capacity : fuel_tank_volume = 208) 
  (H_fuel_A_volume : fuel_A_volume = 82) 
  (H_fuel_B_volume : fuel_B_volume = 126) 
  (H_ethanol_in_fuel_B : ethanol_in_fuel_B = 0.16) 
  (H_total_ethanol : total_ethanol = 30) 
  : 82 * x + 0.16 * 126 = 30 → x = 0.12 := by
  sorry

end ethanol_in_fuel_A_l127_127834


namespace graph_is_pair_of_straight_lines_l127_127141

theorem graph_is_pair_of_straight_lines : ∀ (x y : ℝ), 9 * x^2 - y^2 - 6 * x = 0 → ∃ a b c : ℝ, (y = 3 * x - 2 ∨ y = 2 - 3 * x) :=
by
  intro x y h
  sorry

end graph_is_pair_of_straight_lines_l127_127141


namespace largest_n_for_negative_sum_l127_127195

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ} -- common difference of the arithmetic sequence

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

theorem largest_n_for_negative_sum
  (h_arith_seq : is_arithmetic_sequence a d)
  (h_first_term : a 0 < 0)
  (h_sum_2015_2016 : a 2014 + a 2015 > 0)
  (h_product_2015_2016 : a 2014 * a 2015 < 0) :
  (∀ n, sum_of_first_n_terms a n < 0 → n ≤ 4029) ∧ (sum_of_first_n_terms a 4029 < 0) :=
sorry

end largest_n_for_negative_sum_l127_127195


namespace final_theorem_l127_127278

-- Given an acute triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
-- and the condition a = 2b * sin A,
-- Prove that B = π / 6 and if a = 3 * sqrt 3 and c = 5, find b.

noncomputable def part1 (a b : ℝ) (A B : ℝ) (h1 : a = 2 * b * Real.sin A) : B = π / 6 :=
sorry

-- Given specific values a = 3sqrt3 and c = 5
-- and using cosine rule to find b
noncomputable def part2 (a c : ℝ) (B : ℝ) (h1 : a = 3 * Real.sqrt 3) (h2 : c = 5) 
  (h3 : B = π / 6) : ℝ := by
  let b_squared := a^2 + c^2 - 2 * a * c * Real.cos (π / 6)
  have : b_squared = 7 := sorry
  let b := Real.sqrt 7
  b

-- Final proof combining both parts
theorem final_theorem (a b c A B : ℝ) (h1 : a = 2 * b * Real.sin A) (h2 : a = 3 * Real.sqrt 3) (h3 : c = 5)
  (h4 : ∃ ABC : triangle, ∀ (x y : ℝ), acute_triangle ABC) : B = π / 6 ∧ b = Real.sqrt 7 :=
  by
    have part1_result := part1 a b A B h1
    have part2_result := part2 a c B h2 h3 part1_result
    exact ⟨part1_result, part2_result⟩

end final_theorem_l127_127278


namespace isosceles_triangle_MNP_l127_127417

variables (M N P M₁ P₁ : Point)
variable (triangle_MNP : Triangle M N P)
variable (MM₁ PP₁ : Line)
variable [angle_bisector_MM₁ : AngleBisector MM₁ (Angle MNP M₁)]
variable [angle_bisector_PP₁ : AngleBisector PP₁ (Angle NMP P₁)]
variable (N_to_MM₁_perpendicular : Perpendicular N MM₁)
variable (N_to_PP₁_perpendicular : Perpendicular N PP₁)
variable (perpendicular_length_eq : distance N MM₁ = distance N PP₁)

theorem isosceles_triangle_MNP : distance M N = distance N P :=
by
  sorry

end isosceles_triangle_MNP_l127_127417


namespace necessary_but_not_sufficient_l127_127432

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (¬(x ≥ 1) ∨ (x ≥ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l127_127432


namespace vector_dot_product_condition_l127_127615

variables {V : Type*} [InnerProductSpace ℝ V]

def isosceles_triangle (A B C : V) : Prop :=
  (A - B).norm = (A - C).norm

noncomputable def angle_BAC (A B C : V) : ℝ := (A - B) ⬝ (A - C) / 
  (∥A - B∥ * ∥A - C∥)

variables (A B C D E : V)
noncomputable def length_AB : ℝ := (A - B).norm
noncomputable def length_AC : ℝ := (A - C).norm

axiom condition1 : isosceles_triangle A B C
axiom condition2 : angle_BAC A B C = -1 / 2  -- since cos(120°) = -1/2
axiom condition3 : length_AB = 2
axiom condition4 : length_AC = 2
axiom condition5 : B - C = 2 • (B - D)
axiom condition6 : A - C = 3 • (A - E)

theorem vector_dot_product_condition :
  (A - D) ⬝ (B - E) = -2 / 3 :=
sorry

end vector_dot_product_condition_l127_127615


namespace kate_visits_cost_l127_127640

theorem kate_visits_cost (entrance_fee_first_year : Nat) (monthly_visits : Nat) (next_two_years_fee : Nat) (yearly_visits_next_two_years : Nat) (total_years : Nat) : 
  entrance_fee_first_year = 5 →
  monthly_visits = 12 →
  next_two_years_fee = 7 →
  yearly_visits_next_two_years = 4 →
  total_years = 3 →
  let first_year_cost := entrance_fee_first_year * monthly_visits in
  let subsequent_years_visits := (total_years - 1) * yearly_visits_next_two_years in
  let subsequent_years_cost := next_two_years_fee * subsequent_years_visits in
  let total_cost := first_year_cost + subsequent_years_cost in
  total_cost = 116 :=
begin
  intros h1 h2 h3 h4 h5,
  unfold first_year_cost subsequent_years_visits subsequent_years_cost total_cost,
  simp [h1, h2, h3, h4, h5],
  rfl,
end

end kate_visits_cost_l127_127640


namespace solve_sum_of_squares_l127_127199

theorem solve_sum_of_squares
  (k l m n a b c : ℕ)
  (h_cond1 : k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n)
  (h_cond2 : a * k^2 - b * k + c = 0)
  (h_cond3 : a * l^2 - b * l + c = 0)
  (h_cond4 : c * m^2 - 16 * b * m + 256 * a = 0)
  (h_cond5 : c * n^2 - 16 * b * n + 256 * a = 0) :
  k^2 + l^2 + m^2 + n^2 = 325 :=
by
  sorry

end solve_sum_of_squares_l127_127199


namespace find_a_b_find_m_l127_127563

-- Define the parabola and the points it passes through
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- The conditions based on the given problem
def condition1 (a b : ℝ) : Prop := parabola a b 1 = -2
def condition2 (a b : ℝ) : Prop := parabola a b (-2) = 13

-- Part 1: Proof for a and b
theorem find_a_b : ∃ a b : ℝ, condition1 a b ∧ condition2 a b ∧ a = 1 ∧ b = -4 :=
by sorry

-- Part 2: Given y equation and the specific points
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 1

-- Conditions for the second part
def condition3 : Prop := parabola2 5 = 6
def condition4 (m : ℝ) : Prop := parabola2 m = 12 - 6

-- Theorem statement for the second part
theorem find_m : ∃ m : ℝ, condition3 ∧ condition4 m ∧ m = -1 :=
by sorry

end find_a_b_find_m_l127_127563


namespace problem_statement_l127_127882

noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def g_inv (x : ℝ) : ℝ := (x + 4) / 3

theorem problem_statement : ∃ (x : ℝ), g(x) = g_inv(x) ∧ x = 2 :=
by
  sorry

end problem_statement_l127_127882


namespace midpoints_collinear_l127_127337

open EuclideanGeometry

variables {A B C D E F : Point}
variables {M_X M_Y M_Z : Point}

def is_midpoint (M : Point) (P Q : Point) : Prop := dist M P = dist M Q ∧ dist P Q = 2 * dist M P

theorem midpoints_collinear {quadrilateral : complete_quadrilateral A B C D E F}
  (hMx : is_midpoint M_X C D)
  (hMy : is_midpoint M_Y A B)
  (hMz : is_midpoint M_Z E F) :
  collinear {M_X, M_Y, M_Z} :=
sorry

end midpoints_collinear_l127_127337


namespace number_of_values_of_x_l127_127927

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (169 - real.cbrt x)

theorem number_of_values_of_x : 
  let S := {y : ℕ | y ≤ 13} in
  ∃ (n : ℕ), n = S.card := 
sorry

end number_of_values_of_x_l127_127927


namespace exponent_property_l127_127592

theorem exponent_property (a x y : ℝ) (hx : a ^ x = 2) (hy : a ^ y = 3) : a ^ (x + y) = 6 := by
  sorry

end exponent_property_l127_127592


namespace josh_money_remaining_l127_127636

theorem josh_money_remaining :
  let initial := 50.00
  let shirt := 7.85
  let meal := 15.49
  let magazine := 6.13
  let friends_debt := 3.27
  let cd := 11.75
  initial - shirt - meal - magazine - friends_debt - cd = 5.51 :=
by
  sorry

end josh_money_remaining_l127_127636


namespace cos_105_eq_l127_127861

theorem cos_105_eq : (cos 105) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by 
  have h1: cos (60:ℝ) = 1 / 2 := by sorry
  have h2: cos (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h3: sin (60:ℝ) = real.sqrt 3 / 2 := by sorry
  have h4: sin (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h5: cos (105:ℝ) = cos (60 + 45) := by sorry
  have h6: cos (60 + 45) = cos 60 * cos 45 - sin 60 * sin 45 := by sorry
  have h7 := calc
    cos 105 = (cos 60) * (cos 45) - (sin 60) * (sin 45) : by sorry
    ... = (1 / 2) * (real.sqrt 2 / 2) - (real.sqrt 3 / 2) * (real.sqrt 2 / 2) : by sorry
    ... = (real.sqrt 2 / 4) - (real.sqrt 6 / 4) : by sorry
    ... = (real.sqrt 2 - real.sqrt 6) / 4 : by sorry
  exact h7

end cos_105_eq_l127_127861


namespace cos_105_eq_l127_127860

theorem cos_105_eq : (cos 105) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by 
  have h1: cos (60:ℝ) = 1 / 2 := by sorry
  have h2: cos (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h3: sin (60:ℝ) = real.sqrt 3 / 2 := by sorry
  have h4: sin (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h5: cos (105:ℝ) = cos (60 + 45) := by sorry
  have h6: cos (60 + 45) = cos 60 * cos 45 - sin 60 * sin 45 := by sorry
  have h7 := calc
    cos 105 = (cos 60) * (cos 45) - (sin 60) * (sin 45) : by sorry
    ... = (1 / 2) * (real.sqrt 2 / 2) - (real.sqrt 3 / 2) * (real.sqrt 2 / 2) : by sorry
    ... = (real.sqrt 2 / 4) - (real.sqrt 6 / 4) : by sorry
    ... = (real.sqrt 2 - real.sqrt 6) / 4 : by sorry
  exact h7

end cos_105_eq_l127_127860


namespace circle_area_outside_triangle_l127_127277

/-- Given a right triangle ABC with ∠BAC = 90°, a circle is tangent to side AB and hypotenuse BC
at points X and Y, respectively. The point on the circle diametrically opposite X lies on the hypotenuse.
Given AB = 8 and BC = 10, the area of the portion of the circle that lies outside the triangle ABC is 4π - 8. -/
theorem circle_area_outside_triangle :
  ∃ (r : ℝ), ∃ (A B C X Y : ℝ), ∠BAC = 90 ∧ B = 8 ∧ C = 10 ∧ 
  (area_of_quarter_circle r - area_of_triangle X Y) = 4 * Real.pi - 8 :=
by
  sorry

end circle_area_outside_triangle_l127_127277


namespace prove_angle_A_prove_area_l127_127241

-- First, we declare the conditions as definitions.
variables (a b c A B C : ℝ)
variable (triangle_ABC : Triangle)
variable (h1 : a^2 - b^2 = sqrt 3 * b * c)
variable (h2 : sin C = 2 * sqrt 3 * sin B)

-- To prove: $\angle A = 30^\circ$
def angle_A_is_30_degrees (triangle_ABC : Triangle) (h1 : a^2 - b^2 = sqrt 3 * b * c) (h2 : sin C = 2 * sqrt 3 * sin B) : Prop :=
  A = 30

-- Additional condition for the second proof: $b = 1$
variable (hb1 : b = 1)

-- To prove: The area of $\Delta ABC$ is $\frac{\sqrt{3}}{2}$ if $b = 1$.
def area_given_b_is_1 (triangle_ABC : Triangle) (hb1 : b = 1) : Prop :=
  0.5 * b * c * (sin A) = sqrt 3 / 2

-- We now state the two proofs.
theorem prove_angle_A : angle_A_is_30_degrees triangle_ABC h1 h2 := sorry
theorem prove_area : area_given_b_is_1 triangle_ABC hb1 := sorry

end prove_angle_A_prove_area_l127_127241


namespace joy_remaining_tape_l127_127638

theorem joy_remaining_tape (total_tape length width : ℕ) (h_total_tape : total_tape = 250) (h_length : length = 60) (h_width : width = 20) :
  total_tape - 2 * (length + width) = 90 :=
by
  sorry

end joy_remaining_tape_l127_127638


namespace cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127864

theorem cos_105_eq_sqrt2_sub_sqrt6_div4 :
  cos (105 * real.pi / 180) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by
  -- Definitions and conditions
  have cos_60 : cos (60 * real.pi / 180) = 1/2 := by sorry
  have cos_45 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  have sin_60 : sin (60 * real.pi / 180) = real.sqrt 3 / 2 := by sorry
  have sin_45 : sin (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  -- Use the angle addition formula: cos (a + b) = cos a * cos b - sin a * sin b
  have add_formula := cos_add (60 * real.pi / 180) (45 * real.pi / 180)
  -- Combine the results using the given known values
  rw [cos_60, cos_45, sin_60, sin_45] at add_formula
  exact add_formula

end cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127864


namespace area_increased_by_percent_l127_127047

theorem area_increased_by_percent (l w : ℝ) :
  let original_area := l * w in
  let new_area := (1.2 * l) * (1.1 * w) in
  ((new_area - original_area) / original_area) * 100 = 32 :=
by
  let original_area := l * w
  let new_area := (1.2 * l) * (1.1 * w)
  have h : ((new_area - original_area) / original_area) * 100 = 32 := sorry
  exact h

end area_increased_by_percent_l127_127047


namespace infinite_k_Q_ineq_l127_127916

def Q (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem infinite_k_Q_ineq :
  ∃ᶠ k in at_top, Q (3 ^ k) > Q (3 ^ (k + 1)) := sorry

end infinite_k_Q_ineq_l127_127916


namespace radius_of_smaller_circle_l127_127257

open Real

-- Definitions based on the problem conditions
def large_circle_radius : ℝ := 10
def pattern := "square"

-- Statement of the problem in Lean 4
theorem radius_of_smaller_circle :
  ∀ (r : ℝ), (large_circle_radius = 10) → (pattern = "square") → r = 5 * sqrt 2 →  ∃ r, r = 5 * sqrt 2 :=
by
  sorry

end radius_of_smaller_circle_l127_127257


namespace combinations_divisible_by_10_l127_127966

theorem combinations_divisible_by_10 : 
  {n : ℕ // 1 ≤ n ∧ n ≤ 6} → 
  {n : ℕ // 1 ≤ n ∧ n ≤ 6} → 
  {n : ℕ // 1 ≤ n ∧ n ≤ 6} → 
  ∃ (count : ℕ), count = 72 :=
by
  let count := 6^3 - ((3^3 + 5^3 - 2^3)) -- Simplifying directly based on the given conditions and steps
  exact ⟨count, by norm_num⟩

end combinations_divisible_by_10_l127_127966


namespace problem1_problem2_l127_127215

def given_conditions (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ (f 1 = 8) ∧ (f 3 = 32)

def problem1_correct_answer (a b : ℝ) : Prop :=
  (a = 2) ∧ (b = 4)

def problem2_correct_answer (m : ℝ) : Prop :=
  m ≤ 3/4

theorem problem1 (a b : ℝ) (f : ℝ → ℝ) 
  (hc : given_conditions a b f) : 
  problem1_correct_answer a b := 
sorry

theorem problem2 (a b m : ℝ) 
  (h : a = 2 ∧ b = 4) 
  (hineq : ∀ x : ℝ, x ∈ set.Iic 1 → ( (1 / a) ^ x + (1 / b) ^ x - m ≥ 0)) : 
  problem2_correct_answer m :=
sorry

end problem1_problem2_l127_127215


namespace simplify_fraction_l127_127682

noncomputable def simplify_expression (x : ℂ) : Prop :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) =
  (x - 3) / (x^2 - 6*x + 8)

theorem simplify_fraction (x : ℂ) : simplify_expression x :=
by
  sorry

end simplify_fraction_l127_127682


namespace target_runs_correct_l127_127258

noncomputable def target_runs (run_rate1 : ℝ) (ovs1 : ℕ) (run_rate2 : ℝ) (ovs2 : ℕ) : ℝ :=
  (run_rate1 * ovs1) + (run_rate2 * ovs2)

theorem target_runs_correct : target_runs 4.5 12 8.052631578947368 38 = 360 :=
by
  sorry

end target_runs_correct_l127_127258


namespace sufficient_not_necessary_range_l127_127956

theorem sufficient_not_necessary_range (a : ℝ) (h : ∀ x : ℝ, x > 2 → x^2 > a ∧ ¬(x^2 > a → x > 2)) : a ≤ 4 :=
by
  sorry

end sufficient_not_necessary_range_l127_127956


namespace conjugate_of_z_l127_127209

def is_conjugate (z w : ℂ) : Prop :=
  w = ⟨z.re, -z.im⟩

theorem conjugate_of_z :
  ∃ z : ℂ, (z + 2 * complex.I) / z = 2 + 3 * complex.I ∧ is_conjugate z (⟨3/5, -1/5⟩) :=
by
  sorry

end conjugate_of_z_l127_127209


namespace initial_marbles_total_l127_127482

theorem initial_marbles_total :
  ∃ (x : ℕ), let B := 3 * x in
              let A := 5 * x in
              let J := 7 * x in
              A + (1 / 2) * B = 260 ∧ B + A + J = 600 :=
by
  sorry

end initial_marbles_total_l127_127482


namespace quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l127_127992

open Real

-- Mathematical translations of conditions and proofs
theorem quadratic_real_roots_range_of_m (m : ℝ) (h1 : ∃ x : ℝ, x^2 + 2 * x - (m - 2) = 0) :
  m ≥ 1 := by
  sorry

theorem quadratic_root_and_other_m (h1 : (1:ℝ) ^ 2 + 2 * 1 - (m - 2) = 0) :
  m = 3 ∧ ∃ x : ℝ, (x = -3) ∧ (x^2 + 2 * x - 3 = 0) := by
  sorry

end quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l127_127992


namespace exists_stable_k_l127_127466

noncomputable def is_good_sequence (a : ℕ → ℕ) : Prop :=
a 1 ∈ { x | ∃ y, y^2 = x } ∧ ∀ n ≥ 2, (∃ k, k^2 =  n * a 1 + (n - 1) * a 2 + ∑ i in range (n - 1), (i + 2) * a (i + 2) + a n)

theorem exists_stable_k (a : ℕ → ℕ) (h : is_good_sequence a) : ∃ k : ℕ, 0 < k ∧ ∀ n ≥ k, a n = a k :=
sorry

end exists_stable_k_l127_127466


namespace minimum_distance_from_curve_C_to_line_l_l127_127014

-- Define the polar curve C
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * (Real.cos θ) - 4 * ρ * (Real.sin θ) + 6 = 0

-- Define the parametric line l
def parametric_line (t : ℝ) : ℝ × ℝ :=
  let x := -2 - Real.sqrt 2 * t
  let y := 3 + Real.sqrt 2 * t
  (x, y)

-- Define the Cartesian circle obtained from the polar curve
def cartesian_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Define the Cartesian line obtained from the parametric line
def cartesian_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define the minimum distance function from point (x, y) to line x + y - 1 = 0
def point_to_line_distance (x y : ℝ) : ℝ :=
  Float.abs (x + y - 1) / Real.sqrt 2

-- Define the goal as a theorem
theorem minimum_distance_from_curve_C_to_line_l :
  let center_distance := point_to_line_distance 2 2
  let radius := Real.sqrt 2
  let minimum_distance := center_distance - radius
  minimum_distance = Real.sqrt 2 / 2 :=
sorry

end minimum_distance_from_curve_C_to_line_l_l127_127014


namespace sqrt_of_4m_add_n_is_six_l127_127941

theorem sqrt_of_4m_add_n_is_six (x y m n : ℤ) 
  (h1 : x = 2) 
  (h2 : y = 3) 
  (h3 : m * x + n * y = 28)
  (h4 : m * x - n * y = 4) :
  sqrt (4 * m + n) = 6 ∨ sqrt (4 * m + n) = -6 :=
by
  sorry

end sqrt_of_4m_add_n_is_six_l127_127941


namespace area_of_set_T_l127_127645

def omega : ℂ := -1/2 + (1/2) * complex.I * real.sqrt 3

def T : set ℂ := {z | ∃ a d : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ d ∧ d ≤ 2 ∧ z = a + d * omega}

theorem area_of_set_T : (area T) = 4 := sorry

end area_of_set_T_l127_127645


namespace equation_of_curve_and_point_M_coords_l127_127244

theorem equation_of_curve_and_point_M_coords :
  (∀ P : ℝ × ℝ, let d1 := (P.1 - 1)^2 + P.2^2 in 
                let d2 := P.1^2 in 
                d1 = d2 + 1 → 
                P.2^2 = 4 * P.1) ∧ 
  (∀ M : ℝ × ℝ, (M.2^2 = 4 * M.1) → 
                (let A := (4 / M.2^2 - M.2, 4 / M.2) in 
                 let B := (4 / M.2^2 + M.2, -4 / M.2) in 
                 let D := (x : ℝ, y : ℝ, y = M.2 / 2 * (x - 1) ∧ y^2 = 4 * x) in
                 (∃ DE_len : ℝ, DE_len = 8 -> M = (1, 2) ∨ M = (1, -2))) sorry

end equation_of_curve_and_point_M_coords_l127_127244


namespace projection_lengths_equal_l127_127309

-- Given a cubic function y = ax^3 + bx^2 + cx + d
variables {a b c d p q : ℝ}

-- Define the cubic function f
def cubic_function (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Roots of intersections with y = p and y = q
variables {x1 x2 x3 X1 X2 X3 : ℝ}

-- Assuming x1 < x2 < x3 and X1 < X2 < X3 are the roots of these polynomial equations
axiom roots_first_line (h1 : cubic_function a b c d x1 = p) (h2 : cubic_function a b c d x2 = p) (h3 : cubic_function a b c d x3 = p) : x1 < x2 ∧ x2 < x3
axiom roots_second_line (h4 : cubic_function a b c d X1 = q) (h5 : cubic_function a b c d X2 = q) (h6 : cubic_function a b c d X3 = q) : X1 < X2 ∧ X2 < X3

theorem projection_lengths_equal : (x2 - X2) = (X1 - x1) + (X3 - x3) :=
begin
  -- Proof goes here
  sorry
end

end projection_lengths_equal_l127_127309


namespace linear_regression_equation_l127_127385

theorem linear_regression_equation (x y : ℝ) (h : {(1, 2), (2, 3), (3, 4), (4, 5)} ⊆ {(x, y) | y = x + 1}) : 
  (∀ x y, (x = 1 → y = 2) ∧ (x = 2 → y = 3) ∧ (x = 3 → y = 4) ∧ (x = 4 → y = 5)) ↔ (y = x + 1) :=
by
  sorry

end linear_regression_equation_l127_127385


namespace find_x_l127_127223

theorem find_x (x : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (x, 1)) :
  ((2 * a.fst - x, 2 * a.snd + 1) • b = 0) → x = -1 ∨ x = 3 :=
by
  sorry

end find_x_l127_127223


namespace stuart_initially_had_20_l127_127111

variable (B T S : ℕ) -- Initial number of marbles for Betty, Tom, and Susan
variable (S_after : ℕ) -- Number of marbles Stuart has after receiving from Betty

-- Given conditions
axiom betty_initially : B = 150
axiom tom_initially : T = 30
axiom susan_initially : S = 20

axiom betty_to_tom : (0.20 : ℚ) * B = 30
axiom betty_to_susan : (0.10 : ℚ) * B = 15
axiom betty_to_stuart : (0.40 : ℚ) * B = 60
axiom stuart_after_receiving : S_after = 80

-- Theorem to prove Stuart initially had 20 marbles
theorem stuart_initially_had_20 : ∃ S_initial : ℕ, S_after - 60 = S_initial ∧ S_initial = 20 :=
by {
  sorry
}

end stuart_initially_had_20_l127_127111


namespace convert_726_base9_to_base3_l127_127496

-- Define the conversion of a single digit from base 9 to base 3
def convert_digit_base9_to_base3 (d : ℕ) : string :=
  if d == 7 then "21"
  else if d == 2 then "02"
  else if d == 6 then "20"
  else ""

-- Define the conversion function for the entire number
def convert_base9_to_base3 (n : ℕ) : string :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  (convert_digit_base9_to_base3 d1) ++ (convert_digit_base9_to_base3 d2) ++ (convert_digit_base9_to_base3 d3)

-- Define the proof problem
theorem convert_726_base9_to_base3 :
  convert_base9_to_base3 726 = "210220" :=
  sorry

end convert_726_base9_to_base3_l127_127496


namespace gillian_yearly_phone_bill_l127_127933

-- Given conditions
def usual_monthly_bill : ℝ := 50
def increase_percentage : ℝ := 0.10

-- Desired result for the yearly bill after the increase
def expected_yearly_bill : ℝ := 660

-- The theorem to prove
theorem gillian_yearly_phone_bill :
  let new_monthly_bill := usual_monthly_bill * (1 + increase_percentage) in
  let yearly_bill := new_monthly_bill * 12 in
  yearly_bill = expected_yearly_bill :=
by
  sorry

end gillian_yearly_phone_bill_l127_127933


namespace find_range_of_m_l127_127567

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * log x - a * x - 3

def f' (x : ℝ) (a : ℝ) : ℝ := a * (x - 1) / x

def tangent_slope_45 (a : ℝ) : Prop := f' 2 a = 1

def g (x : ℝ) (m : ℝ) : ℝ := x^3 + (m / 2 + 2) * x^2 - 2 * x

def g' (x : ℝ) (m : ℝ) : ℝ := 3 * x^2 + (m + 4) * x - 2

def non_monotonic_on_interval (m : ℝ) : Prop :=
  ∀ (t : ℝ), 1 ≤ t ∧ t ≤ 2 → g' t m < 0 ∧ g' 3 m > 0

theorem find_range_of_m : ∀ (m : ℝ), 
  tangent_slope_45 (-2) →
  non_monotonic_on_interval m →
  -37 / 3 < m ∧ m < -9 :=
by
  intros m h1 h2
  sorry

end find_range_of_m_l127_127567


namespace infinite_series_sum_eq_1_div_432_l127_127130

theorem infinite_series_sum_eq_1_div_432 :
  (∑' n : ℕ, (4 * (n + 1) + 1) / ((4 * (n + 1) - 1)^3 * (4 * (n + 1) + 3)^3)) = (1 / 432) :=
  sorry

end infinite_series_sum_eq_1_div_432_l127_127130


namespace prove_inequality_l127_127188

variable (C : ℝ) (a : ℕ → ℝ)

def sequence_conditions : Prop := 
  (C ≥ 1) ∧ 
  (∀ n : ℕ, n > 0 → a n ≥ 0) ∧ 
  (∀ x : ℝ, x ≥ 1 → abs (x * real.log x - ∑ k in finset.range (⌊x⌋₊ + 1), (⌊x / k⌋₊ * a k)) ≤ C * x)

theorem prove_inequality (y : ℝ) 
  (hC : C ≥ 1)
  (ha : ∀ n : ℕ, n > 0 → a n ≥ 0)
  (hx : ∀ x : ℝ, x ≥ 1 → abs (x * real.log x - ∑ k in finset.range (⌊x⌋₊ + 1), (⌊x / k⌋₊ * a k)) ≤ C * x):
  y ≥ 1 → (∑ k in finset.range (⌊y⌋₊ + 1), a k) < 3 * C * y :=
by
  sorry

end prove_inequality_l127_127188


namespace sin_300_equiv_l127_127128

noncomputable def sin_300_deg : Real :=
  -Real.sqrt 3 / 2

theorem sin_300_equiv :
  sin (300 * Real.pi / 180) = sin_300_deg := by
  -- Definitions of conditions
  have angle_identity : ∀ θ : Real, sin (θ + 2 * Real.pi) = sin θ :=
    by intro θ; exact Real.sin_add_two_pi θ
  have sine_neg_angle : ∀ θ : Real, sin (-θ) = -sin θ :=
    by intro θ; exact Real.sin_neg θ
  have sin_60 : sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by exact (Real.sin_pi_div_three)

  sorry -- The specific proof steps are omitted

end sin_300_equiv_l127_127128


namespace amount_of_water_per_minute_l127_127056

-- Definitions based on the problem conditions
def river_depth : ℝ := 4 -- in meters
def river_width : ℝ := 40 -- in meters
def flow_rate_kmph : ℝ := 4 -- flow rate in km/h

-- Conversion factor
def km_to_m : ℝ := 1000
def hour_to_min : ℝ := 60

-- Flow rate conversion to m/min
def flow_rate_m_per_min : ℝ := (flow_rate_kmph * km_to_m) / hour_to_min

-- Cross-sectional area of the river
def cross_sectional_area : ℝ := river_depth * river_width

-- Volume of water per minute
def volume_per_minute : ℝ := flow_rate_m_per_min * cross_sectional_area

-- Theorem that needs to be proved
theorem amount_of_water_per_minute : volume_per_minute = 10666.67 := by
  sorry

end amount_of_water_per_minute_l127_127056


namespace pyramid_sum_of_edges_l127_127465

theorem pyramid_sum_of_edges :
  ∀ (s h : ℝ),
  s = 8 →
  h = 15 →
  (4 * s + 4 * real.sqrt (h^2 + (s / real.sqrt 2)^2)) = 96.1 :=
by
  intros s h hs hh
  rw [hs, hh] -- replace s with 8 and h with 15
  -- calculation of the edge sums using provided conditions
  sorry

end pyramid_sum_of_edges_l127_127465


namespace points_per_member_l127_127107

def numMembersTotal := 12
def numMembersAbsent := 4
def totalPoints := 64

theorem points_per_member (h : numMembersTotal - numMembersAbsent = 12 - 4) :
  (totalPoints / (numMembersTotal - numMembersAbsent)) = 8 := 
  sorry

end points_per_member_l127_127107


namespace least_sum_of_bases_l127_127707

theorem least_sum_of_bases (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : 4 * a + 7 = 7 * b + 4) (h4 : 4 * a + 3 % 7 = 0) :
  a + b = 24 :=
sorry

end least_sum_of_bases_l127_127707


namespace parabola_chord_focus_l127_127958

def parabola : Set (ℝ × ℝ) := { p | ∃ y x : ℝ, p = (x, y) ∧ y^2 = 2 * p * x }

theorem parabola_chord_focus
  (p : ℝ)
  (A B : ℝ × ℝ)
  (F G : ℝ × ℝ)
  (hF : ∃ (y x : ℝ), F = (x, y) ∧ y^2 = 2 * p * x)  -- F is the focus
  (hAB_chord : A ∈ parabola ∧ B ∈ parabola ∧ ∃ M, M ∈ parabola ∧ ∃ mid, mid = ((fst A + fst B) / 2, (snd A + snd B) / 2) ∧ mid = proj x axis of chord)
  (hG_bisect : ∃ N, ∃ k, k ∈ parabola ∧ N ∈ parabola ∧ proj ⟂ AB = G axis ∈ x)
  (hFG_dist : ∥F - G∥ = λ * ∥A - B∥) :
  λ = 1 / 2 :=
sorry

end parabola_chord_focus_l127_127958


namespace tangent_line_at_0_eq_x_minus_y_plus_1_l127_127883

noncomputable def tangentLineEquation (x : ℝ) : ℝ := 
  let f := λ x : ℝ, Real.sin x + Real.cos x in
  let f' := λ x : ℝ, Real.cos x - Real.sin x in
  let x0 : ℝ := 0 in
  let y0 : ℝ := f x0 in
  let m : ℝ := f' x0 in
  if (x = x0) then y0 else m * (x - x0) + y0

theorem tangent_line_at_0_eq_x_minus_y_plus_1 :
  let point := (0, Real.sin 0 + Real.cos 0) in
  let slope := Real.cos 0 - Real.sin 0 in
  let line := λ x y : ℝ, x - y + 1 = 0 in
  tangentLineEquation 0 = line.fst point := sorry

end tangent_line_at_0_eq_x_minus_y_plus_1_l127_127883


namespace range_of_a_l127_127572

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 then (5 / 16) * x^2 else (1/2)^x + 1

theorem range_of_a {a b : ℝ}
  (h_even: ∀ x : ℝ, f (-x) = f x)
  (h_piecewise : ∀ x : ℝ, 
    (0 ≤ x ∧ x ≤ 2 → f x = (5 / 16) * x^2) ∧
    (x > 2 → f x = (1/2)^x + 1))
  (h_roots : ∀ x : ℝ, (f x)^2 + a * f x + b = 0 → 
            ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ t1 ∈ { ⋃ x, {f x} } ∧
            t2 ∈ { ⋃ x, {f x} } ∧
            ∃ s : set ℝ, s.card = 6 ∧ x ∈ { ⋃ x, {f x} }) :
  a ∈ set.Ioo (-5 / 2) (-9 / 4) ∪ set.Ioo (-9 / 4) (-1) :=
sorry

end range_of_a_l127_127572


namespace aerith_expected_rolls_l127_127470

-- Define the fair die roll and conditions for stopping
noncomputable def expected_rolls_before_stop : ℝ :=
  let p k := (Natural.binomial 6 (k - 1)) / (6:ℝ)^(k-1) in
  1 + ∑ k in finset.range 2 7, p k

theorem aerith_expected_rolls : expected_rolls_before_stop = 2.52 := 
    sorry

end aerith_expected_rolls_l127_127470


namespace equal_play_time_l127_127822

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l127_127822


namespace parallel_lines_trans_l127_127472

variables {α : Type*} [plane α] {m n : line α}

def lies_within (m : line α) (α : plane α) : Prop := ∀ (x : point α), x ∈ m → x ∈ α

def parallel_to (n : line α) (α : plane α) : Prop := ∀ (x y : point α), x ∈ n → y ∈ α → ∃ z : point α, (z ≠ x) ∧ z ≠ y ∧ z ∈ n ∧ z ∈ α

def parallel_lines (m n : line α) : Prop := ∀ (x : point α), x ∈ m → x ∈ n

theorem parallel_lines_trans {m n : line α} {α : plane α} 
  (hm : lies_within m α) (hn : parallel_to n α) : parallel_lines m n :=
sorry

end parallel_lines_trans_l127_127472


namespace guards_will_succeed_l127_127376

-- Define the basic elements and assumptions
structure Point (α : Type*) :=
(x : α) (y : α)

structure Triangle (α : Type*)
  (A B C : Point α) :=
(equilateral : ∀ (P Q : Point α), (P = A ∧ Q = B) ∨ (P = B ∧ Q = C) ∨ (P = C ∧ Q = A) → 
  dist P Q = dist A B)

def on_path {α : Type*} (pts : List (Point α)) (P : Point α) : Prop :=
P ∈ pts

def same_speed {α : Type*} [H : Field α] (G1 G2 M : Point α) : Prop :=
dist G1 M = dist G2 M ∧ dist G1 M = dist G2 M

def see_each_other {α : Type*} (G1 G2 M : Point α) : Prop :=
true  -- This condition is always true in this problem statement

-- Problem Condition Definitions
variables {α : Type*} [Field α]

def guards_catch_monkey : Prop :=
  ∀ (A B C : Point α) (triangle : Triangle α A B C) 
    (G1 G2 M : Point α),
  on_path [A, B, C] M → same_speed G1 G2 M → see_each_other G1 G2 M →
  ∃ (P : Point α), P = M

-- Theorem Statement
theorem guards_will_succeed :
  guards_catch_monkey :=
begin
  sorry
end

end guards_will_succeed_l127_127376


namespace round_5278132_point_764501_l127_127336

theorem round_5278132_point_764501 : Real.round 5278132.764501 = 5278133 := 
by
  sorry

end round_5278132_point_764501_l127_127336


namespace partial_fraction_ABC_zero_l127_127169

theorem partial_fraction_ABC_zero :
  ∀ (x A B C : ℝ),
  (x^2 - x - 20) / (x^3 - 4*x^2 + x + 6) = A/(x - 3) + B/(x + 1) + C/(x - 2) →
  (x^3 - 4*x^2 + x + 6) = (x - 3)*(x + 1)*(x - 2) →
  (A = 0) ∧ (B = -3/2) ∧ (C = 0) →
  A * B * C = 0 :=
by
  assume x A B C
  assume h₁ h₂ h₃
  sorry

end partial_fraction_ABC_zero_l127_127169


namespace coefficient_of_x_in_expansion_l127_127694

   theorem coefficient_of_x_in_expansion :
     let f := (x : ℤ) ^ 2 + 3 * x + 2
     in coeff (f ^ 5 : polynomial ℤ) 1 = 240 :=
   by
     sorry
   
end coefficient_of_x_in_expansion_l127_127694


namespace equal_play_time_l127_127818

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l127_127818


namespace geometric_sequence_sum_l127_127508

noncomputable def geometric_sum := λ (a r n : ℝ), a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (q a : ℝ) (hq : q > 0) (S10 S30 : ℝ) 
  (hS10 : S10 = geometric_sum a q 10)
  (hS30 : S30 = geometric_sum a q 30)
  (h_val10 : S10 = 10)
  (h_val30 : S30 = 70) : 
  geometric_sum a q 40 = 150 :=
by
  sorry

end geometric_sequence_sum_l127_127508


namespace parabola_point_distance_condition_l127_127547

theorem parabola_point_distance_condition (k : ℝ) (p : ℝ) (h_p_gt_0 : p > 0) (focus : ℝ × ℝ) (vertex : ℝ × ℝ) :
  vertex = (0, 0) → focus = (0, p/2) → (k^2 = -2 * p * (-2)) → dist (k, -2) focus = 4 → k = 4 ∨ k = -4 :=
by
  sorry

end parabola_point_distance_condition_l127_127547


namespace expected_value_xi_l127_127074

noncomputable def passengers_on_ground_floor : ℕ := 5
noncomputable def probability_getting_off_10th_floor : ℚ := 1 / 3
noncomputable def xi : ℕ → ℕ := λ n, binomial passengers_on_ground_floor probability_getting_off_10th_floor n

theorem expected_value_xi :
  ∑ i in finset.range (passengers_on_ground_floor + 1), i * (binomial passengers_on_ground_floor probability_getting_off_10th_floor i) = 5 / 3 :=
sorry

end expected_value_xi_l127_127074


namespace remainder_of_x15_plus_3_div_x_minus_1_eq_4_l127_127046

-- Define the polynomial
def f (x : ℕ) : ℕ := x^15 + 3

-- State the theorem using the Remainder Theorem
theorem remainder_of_x15_plus_3_div_x_minus_1_eq_4 : (f 1) = 4 := by
  have h1 : f 1 = 1 ^ 15 + 3 := rfl
  rw h1
  norm_num

end remainder_of_x15_plus_3_div_x_minus_1_eq_4_l127_127046


namespace probability_event_l127_127463

-- Define the probability space for two random points between 0 and 1
def probability_space := Set.Icc (0 : ℝ) 1 × Set.Icc (0 : ℝ) 1

-- Define the event we're interested in: x < y < 3x
def event (p : ℝ × ℝ) : Prop := p.1 < p.2 ∧ p.2 < 3 * p.1

-- Define the measure of the event
noncomputable def event_probability : ℝ :=
  (volume {p : ℝ × ℝ | event p}) / (volume probability_space)

-- The theorem to prove
theorem probability_event : event_probability = 11 / 18 :=
sorry

end probability_event_l127_127463


namespace min_num_equilateral_triangles_to_cover_isosceles_l127_127043

namespace TriangleCovering

noncomputable def area_isosceles_triangle (b h : ℝ) : ℝ :=
  1 / 2 * b * h

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  sqrt 3 / 4 * s^2

noncomputable def num_triangles_required (b h s : ℝ) : ℝ :=
  area_isosceles_triangle b h / area_equilateral_triangle s

theorem min_num_equilateral_triangles_to_cover_isosceles :
  num_triangles_required 8 15 1 = 138 :=
by
  sorry

end TriangleCovering

end min_num_equilateral_triangles_to_cover_isosceles_l127_127043


namespace classroom_ratio_l127_127717

theorem classroom_ratio (total_students : ℕ) (playground_girls : ℕ) (playground_fraction_boys : ℚ) 
    (h1 : total_students = 20) 
    (h2 : playground_girls = 10) 
    (h3 : playground_fraction_boys = (1 / 3)) : 
    let playground_students := playground_girls * (3 / 2) in
    let classroom_students := total_students - playground_students in
    (classroom_students / total_students : ℚ) = 1 / 4 :=
by
  let playground_students := playground_girls * (3 / 2)
  let classroom_students := total_students - playground_students
  have h4 : playground_students = 15 := by
    calc
      playground_students = playground_girls * (3 / 2) : rfl
      _ = 10 * (3 / 2)           : by rw [h2]
      _ = 15                     : by norm_num
  have h5 : classroom_students = 5 := by
    calc
      classroom_students = total_students - playground_students : rfl
      _ = 20 - 15                             : by rw [h4, h1]
      _ = 5                                   : by norm_num
  show (classroom_students / total_students : ℚ) = 1 / 4 from
    calc
      (classroom_students / total_students : ℚ) = (5 : ℚ) / 20 : by rw [h5, h1]
      _ = 1 / 4                                    : by norm_num

end classroom_ratio_l127_127717


namespace probability_of_distance_less_than_8000_l127_127692

-- Define distances between cities

noncomputable def distances : List (String × String × ℕ) :=
  [("Bangkok", "Cape Town", 6300),
   ("Bangkok", "Honolulu", 7609),
   ("Bangkok", "London", 5944),
   ("Bangkok", "Tokyo", 2870),
   ("Cape Town", "Honolulu", 11535),
   ("Cape Town", "London", 5989),
   ("Cape Town", "Tokyo", 13400),
   ("Honolulu", "London", 7240),
   ("Honolulu", "Tokyo", 3805),
   ("London", "Tokyo", 5950)]

-- Define the total number of pairs and the pairs with distances less than 8000 miles

noncomputable def total_pairs : ℕ := 10
noncomputable def pairs_less_than_8000 : ℕ := 7

-- Define the statement of the probability being 7/10
theorem probability_of_distance_less_than_8000 :
  pairs_less_than_8000 / total_pairs = 7 / 10 :=
by
  sorry

end probability_of_distance_less_than_8000_l127_127692


namespace hypotenuse_measurement_l127_127189

theorem hypotenuse_measurement (a b : ℕ) (h₁ : (a = 6 ∧ b = 8) ∨ (a ≠ 6 ∨ b ≠ 8)) :
  (c : ℝ) := sqrt(a^2 + b^2) ≤ 10 :=
by 
  cases h₁ with h₂ h₃
  case or.inl =>
    rw [h₂.left, h₂.right]
    norm_num
  case or.inr =>
    sorry

end hypotenuse_measurement_l127_127189


namespace joy_remaining_tape_l127_127637

theorem joy_remaining_tape (total_tape length width : ℕ) (h_total_tape : total_tape = 250) (h_length : length = 60) (h_width : width = 20) :
  total_tape - 2 * (length + width) = 90 :=
by
  sorry

end joy_remaining_tape_l127_127637


namespace entries_multiple_of_43_are_29_l127_127831

-- Condition 1: Define the function a(n, k) according to the given problem structure
def a (n k : ℕ) : ℕ := 2^(n-1) * (n + 2*k - 2)

-- Problem Statement: Prove there are exactly 29 entries in the array that are multiples of 43
theorem entries_multiple_of_43_are_29 :
  (∃ n k : ℕ, 1 ≤ n ∧ n ≤ 51 ∧ 1 ≤ k ∧ k ≤ 51 - n ∧ a n k % 43 = 0) → 
  (29 = (finset.range (57 // 2 + 1)).card) :=
sorry

end entries_multiple_of_43_are_29_l127_127831


namespace non_obtuse_triangle_perimeter_greater_than_twice_diameter_l127_127675

theorem non_obtuse_triangle_perimeter_greater_than_twice_diameter
  {A B C : Type*} [nonobtuse_triangle A B C]
  (midpoint_K : midpoint A B)
  (midpoint_L : midpoint B C)
  (midpoint_M : midpoint A C)
  (circumcenter_O : circumcenter A B C)
  (P : ℝ := perimeter A B C)
  (d : ℝ := diameter (circumcircle A B C)) :
  P > 2 * d :=
sorry

end non_obtuse_triangle_perimeter_greater_than_twice_diameter_l127_127675


namespace circle_tangent_to_AB_and_AC_is_tangent_to_circumcircle_l127_127087

-- Definitions of the elements used in the conditions
structure Triangle :=
(A : Point)
(B : Point)
(C : Point)
(incenter : Point)
(line_through_incenter_perpendicular_to_AI : Line)
(line_meets_AB_at_P : Point)
(line_meets_AC_at_Q : Point)

-- Main theorem statement
theorem circle_tangent_to_AB_and_AC_is_tangent_to_circumcircle (T : Triangle) 
(circle_tangent_to_AB_at_P_and_AC_at_Q : Circle)
(circumcircle_of_ABC : Circle) : 
tangent_to circle_tangent_to_AB_at_P_and_AC_at_Q circumcircle_of_ABC :=
sorry

end circle_tangent_to_AB_and_AC_is_tangent_to_circumcircle_l127_127087


namespace sum_even_minus_sum_odd_l127_127240

theorem sum_even_minus_sum_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 := by
sorry

end sum_even_minus_sum_odd_l127_127240


namespace four_distinct_real_solutions_l127_127303

noncomputable def polynomial (a b c d e x : ℝ) : ℝ :=
  (x - a) * (x - b) * (x - c) * (x - d) * (x - e)

noncomputable def derivative (a b c d e x : ℝ) : ℝ :=
  (x - b) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - b) * (x - d) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - d)

theorem four_distinct_real_solutions (a b c d e : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
    (derivative a b c d e x1 = 0 ∧ derivative a b c d e x2 = 0 ∧ derivative a b c d e x3 = 0 ∧ derivative a b c d e x4 = 0) :=
sorry

end four_distinct_real_solutions_l127_127303


namespace least_three_digit_with_product_l127_127404

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_product (n : ℕ) (p : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 * d2 * d3 = p

theorem least_three_digit_with_product (p : ℕ) : ∃ n : ℕ, is_three_digit n ∧ digits_product n p ∧ 
  ∀ m : ℕ, is_three_digit m ∧ digits_product m p → n ≤ m :=
by
  use 116
  sorry

end least_three_digit_with_product_l127_127404


namespace red_ball_probability_l127_127308

-- Definitions based on conditions
def numBallsA := 10
def redBallsA := 5
def greenBallsA := numBallsA - redBallsA

def numBallsBC := 10
def redBallsBC := 7
def greenBallsBC := numBallsBC - redBallsBC

def probSelectContainer := 1 / 3
def probRedBallA := redBallsA / numBallsA
def probRedBallBC := redBallsBC / numBallsBC

-- Theorem statement to be proved
theorem red_ball_probability : (probSelectContainer * probRedBallA) + (probSelectContainer * probRedBallBC) + (probSelectContainer * probRedBallBC) = 4 / 5 := 
sorry

end red_ball_probability_l127_127308


namespace correct_operation_l127_127411

variable (x y : ℝ)

theorem correct_operation : 3 * x * y² - 4 * x * y² = -x * y² :=
by
  sorry

end correct_operation_l127_127411


namespace ratio_john_maya_age_l127_127149

theorem ratio_john_maya_age :
  ∀ (john drew maya peter jacob : ℕ),
  -- Conditions:
  john = 30 ∧
  drew = maya + 5 ∧
  peter = drew + 4 ∧
  jacob = 11 ∧
  jacob + 2 = (peter + 2) / 2 →
  -- Conclusion:
  john / gcd john maya = 2 ∧ maya / gcd john maya = 1 :=
by
  sorry

end ratio_john_maya_age_l127_127149


namespace train_passing_time_l127_127103

-- Definitions and assumptions
def length_of_train := 540 -- in meters
def length_of_platform := 260 -- in meters
def speed_of_train_kmh := 60 -- in km/hr
def speed_of_train_ms := speed_of_train_kmh * (1000 / 3600) -- convert km/hr to m/s

-- Theorem statement
theorem train_passing_time (length_of_train length_of_platform : ℕ) 
  (speed_of_train_kmh : ℕ) : 
  (length_of_train + length_of_platform) / (speed_of_train_kmh * (1000 / 3600)) ≈ 48 :=
by sorry

end train_passing_time_l127_127103


namespace find_c_l127_127527

noncomputable def P (c : ℝ) (x : ℝ) : ℝ := x^3 - 3 * x^2 + c * x - 8

theorem find_c (c : ℝ) : (∀ x, P c (x + 2) = 0) → c = -14 :=
sorry

end find_c_l127_127527


namespace how_many_adults_went_to_movie_l127_127767

-- Defining the given problem in Lean 4

def num_children : ℕ := 2
def total_cost_trip : ℕ := 76
def cost_concessions : ℕ := 12
def cost_child_ticket : ℕ := 7
def cost_adult_ticket : ℕ := 10

theorem how_many_adults_went_to_movie (A : ℕ) :
  A = 5 :=
by
  have cost_children_tickets : ℕ := num_children * cost_child_ticket
  have total_ticket_cost : ℕ := total_cost_trip - cost_concessions
  have cost_adult_tickets : ℕ := total_ticket_cost - cost_children_tickets
  have number_of_adults : ℕ := cost_adult_tickets / cost_adult_ticket
  -- The problem's condition assures that A equals 5
  have h : A = number_of_adults := sorry
  -- Use these definitions and hypotheses to state the theorem equivalently
  show A = 5, from sorry


end how_many_adults_went_to_movie_l127_127767


namespace nine_digit_positive_integers_l127_127224

theorem nine_digit_positive_integers :
  (∃ n : Nat, 10^8 * 9 = n ∧ n = 900000000) :=
sorry

end nine_digit_positive_integers_l127_127224


namespace exponent_multiplication_l127_127595

variable (a x y : ℝ)

theorem exponent_multiplication :
  a^x = 2 →
  a^y = 3 →
  a^(x + y) = 6 :=
by
  intros h1 h2
  sorry

end exponent_multiplication_l127_127595


namespace average_of_data_set_l127_127691

theorem average_of_data_set :
  (7 + 5 + (-2) + 5 + 10) / 5 = 5 :=
by sorry

end average_of_data_set_l127_127691


namespace visitor_increase_l127_127891

variable (x : ℝ) -- The percentage increase each day

theorem visitor_increase (h1 : 1.2 * (1 + x)^2 = 2.5) : 1.2 * (1 + x)^2 = 2.5 :=
by exact h1

end visitor_increase_l127_127891


namespace total_cost_correct_l127_127890

noncomputable def total_cost_after_sales : ℝ :=
let
  shoes_price : ℝ := 70,
  dresses_price : ℝ := 150,
  handbag_price : ℝ := 80,
  accessories_price : ℝ := 20,
  shoes_quantity : ℕ := 4,
  dresses_quantity : ℕ := 2,
  handbag_quantity : ℕ := 1,
  accessories_quantity : ℕ := 5,
  shoes_discount : ℝ := 0.25,
  dresses_discount : ℝ := 0.30,
  handbag_discount : ℝ := 0.40,
  accessories_discount : ℝ := 0.15,
  extra_discount_threshold : ℝ := 350,
  extra_discount : ℝ := 0.10,
  sales_tax : ℝ := 0.08
in
let
  total_shoes := shoes_price * shoes_quantity,
  total_dresses := dresses_price * dresses_quantity,
  total_handbag := handbag_price * handbag_quantity,
  total_accessories := accessories_price * accessories_quantity,

  discount_shoes := total_shoes * shoes_discount,
  discount_dresses := total_dresses * dresses_discount,
  discount_handbag := total_handbag * handbag_discount,
  discount_accessories := total_accessories * accessories_discount,

  discounted_shoes := total_shoes - discount_shoes,
  discounted_dresses := total_dresses - discount_dresses,
  discounted_handbag := total_handbag - discount_handbag,
  discounted_accessories := total_accessories - discount_accessories,

  subtotal := discounted_shoes + discounted_dresses + discounted_handbag + discounted_accessories,
  
  total_after_extra_discount := if subtotal > extra_discount_threshold then
                                   subtotal * (1 - extra_discount)
                                 else
                                   subtotal,

  total_with_tax := total_after_extra_discount * (1 + sales_tax)
in
total_with_tax

theorem total_cost_correct : total_cost_after_sales = 537.52 :=
by
  -- This 'sorry' means that the proof implementation is left as an exercise.
  sorry

end total_cost_correct_l127_127890


namespace shots_cost_l127_127848

-- Define the conditions
def golden_retriever_pregnant_dogs : ℕ := 3
def golden_retriever_puppies_per_dog : ℕ := 4
def golden_retriever_shots_per_puppy : ℕ := 2
def golden_retriever_cost_per_shot : ℕ := 5

def german_shepherd_pregnant_dogs : ℕ := 2
def german_shepherd_puppies_per_dog : ℕ := 5
def german_shepherd_shots_per_puppy : ℕ := 3
def german_shepherd_cost_per_shot : ℕ := 8

def bulldog_pregnant_dogs : ℕ := 4
def bulldog_puppies_per_dog : ℕ := 3
def bulldog_shots_per_puppy : ℕ := 4
def bulldog_cost_per_shot : ℕ := 10

-- Define the total cost calculation
def total_puppies (dogs_per_breed puppies_per_dog : ℕ) : ℕ :=
  dogs_per_breed * puppies_per_dog

def total_shot_cost (puppies shots_per_puppy cost_per_shot : ℕ) : ℕ :=
  puppies * shots_per_puppy * cost_per_shot

def total_cost : ℕ :=
  let golden_retriever_puppies := total_puppies golden_retriever_pregnant_dogs golden_retriever_puppies_per_dog
  let german_shepherd_puppies := total_puppies german_shepherd_pregnant_dogs german_shepherd_puppies_per_dog
  let bulldog_puppies := total_puppies bulldog_pregnant_dogs bulldog_puppies_per_dog
  let golden_retriever_cost := total_shot_cost golden_retriever_puppies golden_retriever_shots_per_puppy golden_retriever_cost_per_shot
  let german_shepherd_cost := total_shot_cost german_shepherd_puppies german_shepherd_shots_per_puppy german_shepherd_cost_per_shot
  let bulldog_cost := total_shot_cost bulldog_puppies bulldog_shots_per_puppy bulldog_cost_per_shot
  golden_retriever_cost + german_shepherd_cost + bulldog_cost

-- Statement of the problem
theorem shots_cost (total_cost : ℕ) : total_cost = 840 := by
  -- Proof would go here
  sorry

end shots_cost_l127_127848


namespace necessary_but_not_sufficient_condition_l127_127974

variables {α : Type*} [field α] 
variables (x y : ℕ → α) (b a : α)

def satisfies_regression (x y : ℕ → α) (b a : α) (n : ℕ) := 
  ∀ i, i < n → y i = b * x i + a

def is_centroid (x y : ℕ → α) (n : ℕ) := 
  let x0 := (finset.range n).sum x / n in
  let y0 := (finset.range n).sum y / n in
  (x0, y0)

theorem necessary_but_not_sufficient_condition : 
  ∀ n, n > 0 →
  satisfies_regression x y b a n →
  let (x0, y0) := is_centroid x y n in 
  satisfies_regression x y b a n → 
  (y0 = b * x0 + a) → 
  (∃ (i : ℕ), i < n ∧ (x i, y i) ≠ (x0, y0)) ∧ 
  (x0 = (finset.range n).sum x / n) ∧ (y0 = (finset.range n).sum y / n) → 
  true := sorry

end necessary_but_not_sufficient_condition_l127_127974


namespace sum_distinct_prime_factors_4446_l127_127360

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0
noncomputable def sum_of_digits (n : ℕ) : ℕ := n.digits.sum
noncomputable def distinct_prime_factors (n : ℕ) : set ℕ := {p | nat.prime p ∧ p ∣ n}

theorem sum_distinct_prime_factors_4446 :
  is_even 4446 ∧ sum_of_digits 4446 = 18 ∧
  distinct_prime_factors 4446 ⊆ {2, 3, 13, 19} ∧
  {2, 3, 13} ⊆ distinct_prime_factors 4446 →
  (∑ p in distinct_prime_factors 4446, p) = 37 :=
begin
  intros h,
  sorry
end

end sum_distinct_prime_factors_4446_l127_127360


namespace cos_105_proof_l127_127855

noncomputable def cos_105_degrees : Real :=
  cos 105 * (π / 180)

theorem cos_105_proof : cos_105_degrees = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_proof_l127_127855


namespace area_calculation_l127_127243

-- Definitions based on problem conditions
def grid_side_cm : ℝ := 12  -- Each side of the grid in cm
def grid_area_cm2 : ℝ := grid_side_cm ^ 2  -- Total area of the grid

def small_circle_diameter_cm : ℝ := 2
def small_circle_radius_cm : ℝ := small_circle_diameter_cm / 2
def small_circle_area_cm2 : ℝ := π * (small_circle_radius_cm ^ 2)
def num_small_circles : ℝ := 4

def large_circle_diameter_cm : ℝ := 4
def large_circle_radius_cm : ℝ := large_circle_diameter_cm / 2
def large_circle_area_cm2 : ℝ := π * (large_circle_radius_cm ^ 2)
def num_large_circles : ℝ := 2

-- Total area covered by circles
def total_circle_area_cm2 : ℝ := (num_small_circles * small_circle_area_cm2) + (num_large_circles * large_circle_area_cm2)

-- Area of visible shaded region
def visible_shaded_area_cm2 : ℝ := grid_area_cm2 - total_circle_area_cm2

-- Values of A and B
def A : ℝ := grid_area_cm2
def B : ℝ := total_circle_area_cm2 / π

-- The proof statement to show that A + B = 156
theorem area_calculation : A + B = 156 := 
  sorry

end area_calculation_l127_127243


namespace max_abs_value_l127_127653

open Complex Real

theorem max_abs_value (z : ℂ) (h : abs (z - 8) + abs (z + 6 * I) = 10) : abs z ≤ 8 :=
sorry

example : ∃ z : ℂ, abs (z - 8) + abs (z + 6 * I) = 10 ∧ abs z = 8 :=
sorry

end max_abs_value_l127_127653


namespace acute_dihedral_angle_exists_l127_127668

def dihedral_angles_at_vertex (pyramid : Set (fin 4 → ℝ^3)) (vertex : fin 4) : fin 3 → ℝ := sorry

theorem acute_dihedral_angle_exists (pyramid : Set (fin 4 → ℝ^3)) (vertices : fin 4) :
  (∃ v : fin 4, ∀ i : fin 3, dihedral_angles_at_vertex pyramid v i < 90) :=
sorry

end acute_dihedral_angle_exists_l127_127668


namespace problem_1_problem_2_l127_127985

theorem problem_1 (f : ℝ → ℝ) (m : ℝ) (h_cond : ∀ x > 0, f x = ln x + m / exp x - 1)
  (h_extreme : ∃ x1 x2, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (f' x1 = 0) ∧ (f' x2 = 0))
  : m > Real.exp 1 := 
sorry

theorem problem_2 (g : ℝ → ℝ) (m : ℝ) (x1 x2 : ℝ)
  (h_cond_g : ∀ x, g x = (x - 2) * exp x - (1 / 3) * m * x^3 + (1 / 2) * m * x^2)
  (h_extreme_g : ∃ x1 x2 x3, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ 
    g' x1 = 0 ∧ g' x2 = 0 ∧ g' x3 = 0)
  (h_ratio : x1 / x2 = 1 / Real.exp 1) 
  : x1 + x2 ≥ (Real.exp 1 + 1) / (Real.exp 1 - 1) :=
sorry

end problem_1_problem_2_l127_127985


namespace sum_x_values_l127_127911

open Real

theorem sum_x_values :
  ∑ x in {x | 50 < x ∧ x < 150 ∧ cos^3 (2 * x) + cos^3 (6 * x) = 8 * cos^3 (4 * x) * cos^3 x}, x = 270 := by
  sorry

end sum_x_values_l127_127911


namespace company_needs_86_workers_l127_127079

def profit_condition (n : ℕ) : Prop :=
  147 * n > 600 + 140 * n

theorem company_needs_86_workers (n : ℕ) : profit_condition n → n ≥ 86 :=
by
  intro h
  sorry

end company_needs_86_workers_l127_127079


namespace least_cost_planting_l127_127671

theorem least_cost_planting :
  let region1_area := 3 * 1
  let region2_area := 4 * 4
  let region3_area := 7 * 2
  let region4_area := 5 * 4
  let region5_area := 5 * 6
  let easter_lilies_cost_per_sqft := 3.25
  let dahlias_cost_per_sqft := 2.75
  let cannas_cost_per_sqft := 2.25
  let begonias_cost_per_sqft := 1.75
  let asters_cost_per_sqft := 1.25
  region1_area * easter_lilies_cost_per_sqft +
  region2_area * dahlias_cost_per_sqft +
  region3_area * cannas_cost_per_sqft +
  region4_area * begonias_cost_per_sqft +
  region5_area * asters_cost_per_sqft =
  156.75 := 
sorry

end least_cost_planting_l127_127671


namespace convert_726_base9_to_base3_l127_127495

-- Define the conversion of a single digit from base 9 to base 3
def convert_digit_base9_to_base3 (d : ℕ) : string :=
  if d == 7 then "21"
  else if d == 2 then "02"
  else if d == 6 then "20"
  else ""

-- Define the conversion function for the entire number
def convert_base9_to_base3 (n : ℕ) : string :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  (convert_digit_base9_to_base3 d1) ++ (convert_digit_base9_to_base3 d2) ++ (convert_digit_base9_to_base3 d3)

-- Define the proof problem
theorem convert_726_base9_to_base3 :
  convert_base9_to_base3 726 = "210220" :=
  sorry

end convert_726_base9_to_base3_l127_127495


namespace ipad_avg_cost_l127_127840

noncomputable def avg_cost_iPad : ℝ :=
  let totalRevenue := 100 * 1000 + 20 * x + 80 * 200
  let avgCostAllProducts := 670
  let numProducts := 200
  ((numProducts * avgCostAllProducts) - (100 * 1000 + 80 * 200)) / 20

theorem ipad_avg_cost (totalRevenue : ℝ) (avgCostAllProducts : ℝ) (numProducts : ℝ) (iphoneSales : ℝ) (iphoneCost : ℝ) (ipadSales : ℝ) (ipadCost : ℝ) (tvSales : ℝ) (tvCost : ℝ) :
  totalRevenue = (iphoneSales * iphoneCost) + (ipadSales * ipadCost) + (tvSales * tvCost) →
  avgCostAllProducts = 670 →
  numProducts = 200 →
  iphoneSales = 100 →
  iphoneCost = 1000 →
  tvSales = 80 →
  tvCost = 200 →
  numProducts * avgCostAllProducts = totalRevenue →
  ipadCost = 900 :=
by
  sorry

end ipad_avg_cost_l127_127840


namespace exponent_property_l127_127593

theorem exponent_property (a x y : ℝ) (hx : a ^ x = 2) (hy : a ^ y = 3) : a ^ (x + y) = 6 := by
  sorry

end exponent_property_l127_127593


namespace probability_ephraim_fiona_same_heads_as_keiko_l127_127271

/-- Define a function to calculate the probability that Keiko, Ephraim, and Fiona get the same number of heads. -/
def probability_same_heads : ℚ :=
  let total_outcomes := (2^2) * (2^3) * (2^3)
  let successful_outcomes := 13
  successful_outcomes / total_outcomes

/-- Theorem stating the problem condition and expected probability. -/
theorem probability_ephraim_fiona_same_heads_as_keiko
  (h_keiko : ℕ := 2) -- Keiko tosses two coins
  (h_ephraim : ℕ := 3) -- Ephraim tosses three coins
  (h_fiona : ℕ := 3) -- Fiona tosses three coins
  -- Expected probability that both Ephraim and Fiona get the same number of heads as Keiko
  : probability_same_heads = 13 / 256 :=
sorry

end probability_ephraim_fiona_same_heads_as_keiko_l127_127271


namespace angle_between_vectors_length_of_diagonals_l127_127964

variables (a b : ℝ³) (θ : ℝ)

-- Conditions
axiom magnitude_a : ‖a‖ = 4
axiom magnitude_b : ‖b‖ = 3
axiom dot_product_condition : (2 • a - 3 • b) ⬝ (2 • a + b) = 61

-- Proof goals
theorem angle_between_vectors : θ = 2 * Real.pi / 3 :=
by sorry

theorem length_of_diagonals (d1 d2 : ℝ) :
  d1 = ‖a + b‖ ∧  d2 = ‖a - b‖ ∧
  d1 = Real.sqrt (4^2 + 3^2 + 2 * (-6)) ∧
  d2 = Real.sqrt (4^2 + 3^2 - 2 * (-6)) :=
by sorry

end angle_between_vectors_length_of_diagonals_l127_127964


namespace concyclic_points_theorem_l127_127970

def points_concyclic (B Q L C : Type) [inhabited B] [inhabited Q] [inhabited L] [inhabited C] : Prop :=
  ∃ A D (AD_circle : set B), 
  (B ∈ AD_circle ∧ C ∈ AD_circle) ∧ 
  (∃ P, P ∈ (segment B C) ∧ 
  (∃ M, M ∈ segment A B ∧ 
  (∃ N, N ∈ segment A C ∧ 
  par PMAN (parallelogram (1 : Type)) ∧
  is_angle_bisector P L (angle_bisector (triangle M P N))))) ∧
  (∃ PD Q, intersects PD (segment M N) at Q) ∧
  concyclic B Q L C

/-- Proof that points B, Q, L, and C are concyclic given the conditions
 points B and C are on a circle with diameter AD, AB = AC, P is any point
 on BC, PMAN is a parallelogram, PL is the angle bisector of ∠P in ΔMPN,
 and PD intersects MN at Q. -/
theorem concyclic_points_theorem (B Q L C : Type) [inhabited B] [inhabited Q] [inhabited L] [inhabited C] :
  points_concyclic B Q L C := 
sorry

end concyclic_points_theorem_l127_127970


namespace birds_reduction_on_third_day_l127_127049

theorem birds_reduction_on_third_day
  {a b c : ℕ} 
  (h1 : a = 300)
  (h2 : b = 2 * a)
  (h3 : c = 1300)
  : (b - (c - (a + b))) = 200 :=
by sorry

end birds_reduction_on_third_day_l127_127049


namespace line_perpendicular_planes_l127_127546

-- Given definitions
variables {l : Line}
variables {α β : Plane}

-- Conditions
def line_perpendicular_to_plane (l : Line) (α : Plane) : Prop := l ⊥ α
def line_included_in_plane (l : Line) (β : Plane) : Prop := l ⊆ β
def planes_perpendicular (α β : Plane) : Prop := α ⊥ β

-- Proof problem statement
theorem line_perpendicular_planes (h1 : line_perpendicular_to_plane l α) (h2 : line_included_in_plane l β) : planes_perpendicular α β :=
sorry

end line_perpendicular_planes_l127_127546


namespace Ofelia_savings_in_May_l127_127323

theorem Ofelia_savings_in_May :
  let saving_in_month (savings : ℕ → ℕ) (january_savings : ℕ) :=
  savings 0 = january_savings ∧ (∀ n : ℕ, savings (n + 1) = 2 * savings n) in
  ∀ savings : ℕ → ℕ, saving_in_month savings 10 → savings 4 = 160 :=
by
  intros savings h
  rcases h with ⟨jan, step⟩
  sorry

end Ofelia_savings_in_May_l127_127323


namespace perpendicular_slope_l127_127524

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 2 * y = 10) :
  ∀ (m' : ℝ), m' = -2 / 5 :=
by
  sorry

end perpendicular_slope_l127_127524


namespace largest_common_factor_462_330_l127_127039

-- Define the factors of 462
def factors_462 : Set ℕ := {1, 2, 3, 6, 7, 14, 21, 33, 42, 66, 77, 154, 231, 462}

-- Define the factors of 330
def factors_330 : Set ℕ := {1, 2, 3, 5, 6, 10, 11, 15, 30, 33, 55, 66, 110, 165, 330}

-- Define the statement of the theorem
theorem largest_common_factor_462_330 : 
  (∀ d : ℕ, d ∈ (factors_462 ∩ factors_330) → d ≤ 66) ∧
  66 ∈ (factors_462 ∩ factors_330) :=
sorry

end largest_common_factor_462_330_l127_127039


namespace abs_neg_two_plus_two_l127_127129

def abs (x : Int) : Int :=
  if x >= 0 then x else -x

theorem abs_neg_two_plus_two : abs (-2) + 2 = 4 := by
  sorry

end abs_neg_two_plus_two_l127_127129


namespace profit_percentage_after_decrease_l127_127880

variable (selling_price : ℝ)
variable (initial_cost : ℝ) (new_cost : ℝ)
variable (initial_profit_percent : ℝ)

-- Conditions
def conditions :=
  initial_cost = 80 ∧
  new_cost = 50 ∧
  initial_profit_percent = 0.20 ∧
  selling_price = initial_cost / (1 - initial_profit_percent)

-- Profit percentage function
def profit_percentage (price cost : ℝ) : ℝ :=
  ((price - cost) / price) * 100

-- Theorem to be proved
theorem profit_percentage_after_decrease :
  conditions selling_price initial_cost new_cost initial_profit_percent →
  profit_percentage selling_price new_cost = 50 :=
sorry

end profit_percentage_after_decrease_l127_127880


namespace solve_inequality_l127_127757

noncomputable def inequality (x : ℕ) : Prop :=
  6 * (9 : ℝ)^(1/x) - 13 * (3 : ℝ)^(1/x) * (2 : ℝ)^(1/x) + 6 * (4 : ℝ)^(1/x) ≤ 0

theorem solve_inequality (x : ℕ) (hx : 1 < x) : inequality x ↔ x ≥ 2 :=
by {
  sorry
}

end solve_inequality_l127_127757


namespace infinite_not_expressible_as_sum_of_three_squares_l127_127068

theorem infinite_not_expressible_as_sum_of_three_squares :
  ∃ (n : ℕ), ∃ (infinitely_many_n : ℕ → Prop), (∀ m:ℕ, (infinitely_many_n m ↔ m ≡ 7 [MOD 8])) ∧ ∀ a b c : ℕ, n ≠ a^2 + b^2 + c^2 := 
by
  sorry

end infinite_not_expressible_as_sum_of_three_squares_l127_127068


namespace intersection_A_B_complement_A_in_U_complement_B_in_U_l127_127307

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {5, 6, 7, 8}
def B : Set ℕ := {2, 4, 6, 8}

-- Problems to prove
theorem intersection_A_B : A ∩ B = {6, 8} := by
  sorry

theorem complement_A_in_U : U \ A = {1, 2, 3, 4} := by
  sorry

theorem complement_B_in_U : U \ B = {1, 3, 5, 7} := by
  sorry

end intersection_A_B_complement_A_in_U_complement_B_in_U_l127_127307


namespace perimeter_triangle_eq_dodecagon_distance_PA_eq_side_dodecagon_l127_127296

-- Statements based on the given problem's conditions and conclusions
theorem perimeter_triangle_eq_dodecagon (GERMANYISHOT : Type)
  [IsRegularDodecagon GERMANYISHOT]
  (G I P N M : Point)
  (GN : Line)
  (MI : Line)
  (h_intersect : intersects GN MI P) :
  perimeter (triangle G I P) = perimeter GERMANYISHOT := 
sorry

theorem distance_PA_eq_side_dodecagon (GERMANYISHOT : Type)
  [IsRegularDodecagon GERMANYISHOT]
  (P A : Point)
  (side : ℝ)
  (h_side_length : side_length GERMANYISHOT side) :
  dist P A = side := 
sorry

end perimeter_triangle_eq_dodecagon_distance_PA_eq_side_dodecagon_l127_127296


namespace conversation_occurred_on_Thursday_l127_127027

-- Definitions based on conditions
constant visitsEveryNDays : Nat → Nat → Prop
def visitsAdjustsIfWednesday (day: Nat) : Nat :=
  if day % 7 = 3 then day + 1 else day

-- Conditions
axiom A_visits_every_2_days : ∀ day : Nat, visitsEveryNDays day 2
axiom B_visits_every_3_days : ∀ day : Nat, visitsEveryNDays day 3
axiom C_visits_every_4_days : ∀ day : Nat, visitsEveryNDays day 4
axiom library_is_closed_on_Wednesday : ∀ day : Nat, day % 7 = 3 → visitsAdjustsIfWednesday day = day + 1
axiom all_meet_on_Monday : ∃ day : Nat, day % 12 = 0 ∧ day % 7 = 0 

-- The desired conclusion
theorem conversation_occurred_on_Thursday :
  ∀ day : Nat, (day = 4) ↔ (day + 12 % 7 = 0) :=
sorry

end conversation_occurred_on_Thursday_l127_127027


namespace symmetric_graph_inverse_l127_127206

def f (x : ℝ) : ℝ := sorry -- We assume f is defined accordingly somewhere, as the inverse of ln.

theorem symmetric_graph_inverse (h : ∀ x, f (f x) = x) : f 2 = Real.exp 2 := by
  sorry

end symmetric_graph_inverse_l127_127206


namespace max_diff_S_n_S_m_l127_127575

noncomputable def a_n (n : ℕ) : ℤ := -(n * n : ℤ) + 12 * (n : ℤ) - 32

noncomputable def S_n (n : ℕ) : ℤ := ∑ i in finset.range n, a_n (i + 1)

theorem max_diff_S_n_S_m : S_n 8 - S_n 4 = 10 :=
by sorry

end max_diff_S_n_S_m_l127_127575


namespace last_house_probability_l127_127718

noncomputable def santa_claus_distribution (A : ℕ) (B : ℕ) : Enat :=
if B ≠ A then 1 / 2013 else 0

theorem last_house_probability (A B : ℕ)
  (h : B ≠ A) :
  santa_claus_distribution A B = 1 / 2013 :=
by
  sorry

end last_house_probability_l127_127718


namespace line_equation_correct_l127_127437

-- Define the conditions
def slope (m : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  m = (p2.2 - p1.2) / (p2.1 - p1.1)

def passes_through (p : ℝ × ℝ) (line_eq : ℝ → ℝ) : Prop :=
  line_eq p.1 = p.2

-- Define the point and slope
def l_slope : ℝ := 2
def point : ℝ × ℝ := (0, 3)

-- Define the line equation
def line_eq (x : ℝ) : ℝ := 2 * x + 3

-- Prove that the line_eq satisfies the conditions and thus is the equation of the line
theorem line_equation_correct :
  slope l_slope (0, line_eq 0) (1, line_eq 1) ∧ passes_through point line_eq :=
by
  sorry

end line_equation_correct_l127_127437


namespace perpendicular_iff_zero_dot_product_l127_127581

open Real

def a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem perpendicular_iff_zero_dot_product (m : ℝ) :
  dot_product (a m) (b m) = 0 → m = -1 / 3 :=
by
  sorry

end perpendicular_iff_zero_dot_product_l127_127581


namespace intersection_midpoint_l127_127700

theorem intersection_midpoint (b : ℝ) : 
  (let midpoint := (1 + 5) / 2, (3 + 7) / 2 in
   let x := midpoint.1 in
   let y := midpoint.2 in
   x - y = b) → b = -2 :=
by
  let midpoint := (3, 5)
  let x := midpoint.1
  let y := midpoint.2
  have midpoint_eq : (1 + 5) / 2 = 3 ∧ (3 + 7) / 2 = 5 := by
    split
    norm_num
    norm_num
  change x - y = b → b = -2
  rw [midpoint_eq.1, midpoint_eq.2]
  intro h
  simp at h
  exact h.symm

end intersection_midpoint_l127_127700


namespace sum_pqrs_eq_one_l127_127283

-- Definitions for mutually orthogonal unit vectors
variables (a b c d : ℝ^3)
variables (p q r s : ℝ)

-- Conditions
axiom orthogonal_unit_vectors : 
  orthogonal a b ∧ orthogonal a c ∧ orthogonal a d ∧ orthogonal b c ∧ orthogonal b d ∧ orthogonal c d ∧ 
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥c∥ = 1 ∧ ∥d∥ = 1

axiom vector_equation :
  a = p * (a.cross b) + q * (b.cross c) + r * (c.cross a) + s * (d.cross a)

axiom scalar_product_condition : 
  a.dot (b.cross c) = 1

-- Theorem to prove
theorem sum_pqrs_eq_one : p + q + r + s = 1 := sorry

end sum_pqrs_eq_one_l127_127283


namespace number_of_non_neg_real_values_of_x_l127_127919

noncomputable def numNonNegRealValues (x : ℝ) : ℕ :=
  ∑ k in Finset.range 14, if (169 - k^2 : ℝ) ^ 3 ≥ 0 then 1 else 0

theorem number_of_non_neg_real_values_of_x :
  numNonNegRealValues 169 = 14 :=
sorry

end number_of_non_neg_real_values_of_x_l127_127919


namespace fraction_is_perfect_square_l127_127298

theorem fraction_is_perfect_square (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end fraction_is_perfect_square_l127_127298


namespace combined_work_time_l127_127236

theorem combined_work_time (W : ℝ) (A B C : ℝ) (ha : A = W / 12) (hb : B = W / 18) (hc : C = W / 9) : 
  1 / (A + B + C) = 4 := 
by sorry

end combined_work_time_l127_127236


namespace gillian_phone_bill_l127_127938

variable (original_monthly_bill : ℝ) (increase_percentage : ℝ) (months_per_year : ℕ)

def annual_phone_bill_after_increase (bill : ℝ) (increase : ℝ) (months : ℕ) : ℝ :=
  bill * (1 + increase / 100) * months

theorem gillian_phone_bill (h1 : original_monthly_bill = 50)
  (h2 : increase_percentage = 10)
  (h3 : months_per_year = 12) :
  annual_phone_bill_after_increase original_monthly_bill increase_percentage months_per_year = 660 := by
  sorry

end gillian_phone_bill_l127_127938


namespace kate_visits_cost_l127_127639

theorem kate_visits_cost (entrance_fee_first_year : Nat) (monthly_visits : Nat) (next_two_years_fee : Nat) (yearly_visits_next_two_years : Nat) (total_years : Nat) : 
  entrance_fee_first_year = 5 →
  monthly_visits = 12 →
  next_two_years_fee = 7 →
  yearly_visits_next_two_years = 4 →
  total_years = 3 →
  let first_year_cost := entrance_fee_first_year * monthly_visits in
  let subsequent_years_visits := (total_years - 1) * yearly_visits_next_two_years in
  let subsequent_years_cost := next_two_years_fee * subsequent_years_visits in
  let total_cost := first_year_cost + subsequent_years_cost in
  total_cost = 116 :=
begin
  intros h1 h2 h3 h4 h5,
  unfold first_year_cost subsequent_years_visits subsequent_years_cost total_cost,
  simp [h1, h2, h3, h4, h5],
  rfl,
end

end kate_visits_cost_l127_127639


namespace cos_105_eq_l127_127859

theorem cos_105_eq : (cos 105) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by 
  have h1: cos (60:ℝ) = 1 / 2 := by sorry
  have h2: cos (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h3: sin (60:ℝ) = real.sqrt 3 / 2 := by sorry
  have h4: sin (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h5: cos (105:ℝ) = cos (60 + 45) := by sorry
  have h6: cos (60 + 45) = cos 60 * cos 45 - sin 60 * sin 45 := by sorry
  have h7 := calc
    cos 105 = (cos 60) * (cos 45) - (sin 60) * (sin 45) : by sorry
    ... = (1 / 2) * (real.sqrt 2 / 2) - (real.sqrt 3 / 2) * (real.sqrt 2 / 2) : by sorry
    ... = (real.sqrt 2 / 4) - (real.sqrt 6 / 4) : by sorry
    ... = (real.sqrt 2 - real.sqrt 6) / 4 : by sorry
  exact h7

end cos_105_eq_l127_127859


namespace probability_m_eq_kn_l127_127082

/- 
Define the conditions and question in Lean 4 -/
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def valid_rolls : Finset (ℕ × ℕ) := Finset.product die_faces die_faces

def events_satisfying_condition : Finset (ℕ × ℕ) :=
  {(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (4, 1), (4, 2), (4, 4), 
   (5, 1), (5, 5), (6, 1), (6, 2), (6, 3), (6, 6)}

theorem probability_m_eq_kn (k : ℕ) (h : k > 0) :
  (events_satisfying_condition.card : ℚ) / (valid_rolls.card : ℚ) = 7/18 := by
  sorry

end probability_m_eq_kn_l127_127082


namespace fraction_integer_solution_l127_127649

theorem fraction_integer_solution (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 8) (h₃ : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = -1 := 
sorry

end fraction_integer_solution_l127_127649


namespace cos_105_eq_fraction_l127_127870

theorem cos_105_eq_fraction : 
  cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  have h_cos_45 : cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.cos_eq_sqrt_4_inv_norm_eq_sqrt_two_div_two]
  have h_cos_60 : cos (60 * Real.pi / 180) = 1 / 2 :=
    by norm_num [Real.cos_pi_div_three]
  have h_sin_45 : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num [Real.sin_eq_sqrt_4_inv_sqrt_two_div_two]
  have h_sin_60 : sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by norm_num [Real.sin_pi_div_three]
  sorry

end cos_105_eq_fraction_l127_127870


namespace sum_first_n_odd_eq_n_squared_l127_127677

theorem sum_first_n_odd_eq_n_squared (n : ℕ) : (Finset.sum (Finset.range n) (fun k => (2 * k + 1)) = n^2) := sorry

end sum_first_n_odd_eq_n_squared_l127_127677


namespace relationship_of_x1_x2_x0_l127_127656

theorem relationship_of_x1_x2_x0 (a x1 x2 x0 : ℝ) (h0 : 0 < x1) (h1 : x1 < x2) (hx0 : 0 < x0)
  (h2 : f x1 a - f x2 a = f' x0 a * (x1 - x2)) : 
  x1 + x2 > 2 * x0 :=
sorry

noncomputable def f (x a : ℝ) : ℝ := 4 * real.log x - 1 / 2 * a * x^2 + (4 - a) * x

noncomputable def f' (x a : ℝ) : ℝ := 4 / x - a * x + (4 - a)

end relationship_of_x1_x2_x0_l127_127656


namespace range_of_a_l127_127962

noncomputable def f (a x: ℝ) : ℝ :=
  if x > 1 then a ^ x else (4 - a / 2) * x + 2

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → 4 ≤ a ∧ a < 8 :=
begin
  sorry
end

end range_of_a_l127_127962


namespace distance_between_andrey_and_valentin_l127_127117

-- Definitions based on conditions
def speeds_relation_andrey_boris (a b : ℝ) := b = 0.94 * a
def speeds_relation_boris_valentin (b c : ℝ) := c = 0.95 * b

theorem distance_between_andrey_and_valentin
  (a b c : ℝ)
  (h1 : speeds_relation_andrey_boris a b)
  (h2 : speeds_relation_boris_valentin b c)
  : 1000 - 1000 * c / a = 107 :=
by
  sorry

end distance_between_andrey_and_valentin_l127_127117


namespace number_of_values_of_x_l127_127925

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (169 - real.cbrt x)

theorem number_of_values_of_x : 
  let S := {y : ℕ | y ≤ 13} in
  ∃ (n : ℕ), n = S.card := 
sorry

end number_of_values_of_x_l127_127925


namespace min_f_eq_4_sqrt_678_final_result_l127_127350

noncomputable def Tetrahedron := ℝ × ℝ × ℝ × ℝ → Prop

variables (A B C D : ℝ) (X : ℝ → ℝ)

def AD : ℝ := 28
def BC : ℝ := 28
def AC : ℝ := 44
def BD : ℝ := 44
def AB : ℝ := 52
def CD : ℝ := 52

def f (X : ℝ → ℝ) : ℝ :=
  (A.dist X + B.dist X + C.dist X + D.dist X)

theorem min_f_eq_4_sqrt_678 (X : ℝ → ℝ) : 
  ∃ m n : ℕ, m = 4 ∧ n = 678 ∧ (∀ X, f(X) ≥ 4 * Real.sqrt 678) ∧ (∃ X, f(X) = 4 * Real.sqrt 678) :=
sorry

theorem final_result (m n : ℕ) (H : m = 4 ∧ n = 678) : m + n = 682 :=
begin
  rcases H with ⟨hm, hn⟩,
  rw [hm, hn],
  exact nat.add_zero 682,
end

end min_f_eq_4_sqrt_678_final_result_l127_127350


namespace total_amount_l127_127659

def mark_dollars : ℚ := 5 / 8
def carolyn_dollars : ℚ := 2 / 5
def total_dollars : ℚ := mark_dollars + carolyn_dollars

theorem total_amount : total_dollars = 1.025 := by
  sorry

end total_amount_l127_127659


namespace radius_of_larger_circle_l127_127398

noncomputable def larger_circle_radius (r : ℝ) : ℝ := 5 * r

theorem radius_of_larger_circle
  (smaller_larger_ratio : ℝ := 2 / 5)
  (AC : ℝ)
  (BC : ℝ)
  (AB : ℝ := 10)
  (angle_ABC : ℝ := 90)
  (tangent_smaller : 2 * smaller_larger_ratio * real.sqrt(smaller_larger_ratio^2 * 10^2 - 10) = 10)
  (right_triangle : angle_ABC = 90) :
  larger_circle_radius (10 / real.sqrt(92)) ≈ 5.2 :=
begin
  sorry
end

end radius_of_larger_circle_l127_127398


namespace squirrel_climb_l127_127102

-- Define the problem conditions and the goal
variable (x : ℝ)

-- net_distance_climbed_every_two_minutes
def net_distance_climbed_every_two_minutes : ℝ := x - 2

-- distance_climbed_in_14_minutes
def distance_climbed_in_14_minutes : ℝ := 7 * (x - 2)

-- distance_climbed_in_15th_minute
def distance_climbed_in_15th_minute : ℝ := x

-- total_distance_climbed_in_15_minutes
def total_distance_climbed_in_15_minutes : ℝ := 26

-- Theorem: proving x based on the conditions
theorem squirrel_climb : 
  7 * (x - 2) + x = 26 -> x = 5 := by
  intros h
  sorry

end squirrel_climb_l127_127102


namespace invariant_intersection_l127_127579

noncomputable def fixed_points_on_line (A B C : ℝ) (hABC : A < B ∧ B < C) : Prop :=
A < B ∧ B < C

noncomputable def circle_passing_A_C (A C : ℝ) (hAC₁ : A ≠ C) : Prop :=
∃ O : ℝ × ℝ, O ≠ (A, 0) ∧ ∃ r : ℝ, r > 0 ∧ ((fst O - A)^2 + (snd O)^2 = r^2) ∧ ((fst O - C)^2 + (snd O)^2 = r^2)

noncomputable def tangents_intersect_at_P (A C : ℝ) (hAC₂ : A ≠ C) : Prop :=
∃ P : ℝ × ℝ, ∀ R : ℝ × ℝ, (fst R - A)*(fst R - A) + (snd R)^2 = (fst R - C)*(fst R - C) + (snd R)^2 → fst P ≠ A ∧ fst P ≠ C ∧ ⟪P, R⟫ = 0 

noncomputable def segment_intersects_Q (B : ℝ) (Q : ℝ × ℝ) (Γ : circle_passing_A_C) : Prop :=
∃ t : ℝ, 0 < t ∧ t < 1 ∧ Q = (t * B + (1 - t) * fst Γ.center, (1 - t) * snd Γ.center)

theorem invariant_intersection (A B C : ℝ) (Γ : circle_passing_A_C) (Q : ℝ × ℝ) (hABC : fixed_points_on_line A B C) (hΓ : circle_passing_A_C A C) 
(tangents_intersect : tangents_intersect_at_P A C) (intersect_Q : segment_intersects_Q B Q Γ) :
  ∃ R : ℝ, ∀ Γ' : circle_passing_A_C A C, angle_bisector_intersect_angle A Q C R A C :=
sorry

end invariant_intersection_l127_127579


namespace angle_AHB_of_triangle_ABC_l127_127471

theorem angle_AHB_of_triangle_ABC
  (A B C F G H : Type)
  (altitude_AF : Altitude A F B C)
  (altitude_BG : Altitude B G A C)
  (H_intersects_AF_BG : Intersection H altitude_AF altitude_BG)
  (angle_BAC : measure_angle A B C = 60)
  (angle_ABC : measure_angle A B C = 80) :
  measure_angle A H B = 140 := 
sorry

end angle_AHB_of_triangle_ABC_l127_127471


namespace series_sum_eq_l127_127667

theorem series_sum_eq (n : ℕ) : 
  ∑ k in Finset.range (n + 1), 1 / ((3 * k - 2 : ℕ) * (3 * k + 1 : ℕ)) = n / (3 * n + 1 : ℕ) := 
by
  sorry

end series_sum_eq_l127_127667


namespace solving_inequality_l127_127162

theorem solving_inequality (x : ℝ) :
  (x ∈ set.Ioo (-4:ℝ) (-2:ℝ) ∪ set.Ioc (-2:ℝ) (real.sqrt 8)) ↔
  (2/(x+2) + 4/(x+4) ≥ 1) :=
by sorry

end solving_inequality_l127_127162


namespace trains_clear_post_time_correct_l127_127093

def first_train_length : ℝ := 120
def first_train_speed : ℝ := 36 * 1000 / 3600
def second_train_length : ℝ := 150
def second_train_speed : ℝ := 45 * 1000 / 3600
def telegraph_post_length : ℝ := 30

theorem trains_clear_post_time_correct :
  first_train_length / first_train_speed = 12 ∧ second_train_length / second_train_speed = 12 :=
by
  have h1 : first_train_length / first_train_speed = 12 :=
    by sorry
  have h2 : second_train_length / second_train_speed = 12 :=
    by sorry
  exact ⟨h1, h2⟩

end trains_clear_post_time_correct_l127_127093


namespace Ofelia_savings_in_May_l127_127322

theorem Ofelia_savings_in_May :
  let saving_in_month (savings : ℕ → ℕ) (january_savings : ℕ) :=
  savings 0 = january_savings ∧ (∀ n : ℕ, savings (n + 1) = 2 * savings n) in
  ∀ savings : ℕ → ℕ, saving_in_month savings 10 → savings 4 = 160 :=
by
  intros savings h
  rcases h with ⟨jan, step⟩
  sorry

end Ofelia_savings_in_May_l127_127322


namespace part_a_part_b_l127_127065

noncomputable def cities := Fin 100
def records := Matrix cities cities ℝ

-- Given conditions
variable (d : records)
variable (n : ℕ := 100)
variable (h_condition : ∀ (A B C : cities), A ≠ B ∧ B ≠ C ∧ C ≠ A → 
  ¬ collinear ({A, B, C} : Set cities))

-- (a) If one record is erased, cannot always be uniquely restored
theorem part_a : ∃ (i j : cities), 
  (d i j).erase ≠ d :=
sorry

-- (b) The largest k such that the erased records can always be restored is 96
theorem part_b : ∀ (k : ℕ), 
  (k ≤ n - 4) ↔ ∀ (erased : Finset (Σ i j, i < j)), erased.card = k → 
  ∃ (d' : records), 
  ∀ (i j : cities), (i ≠ j → (i, j) ∉ erased → d i j = d' i j) :=
sorry

end part_a_part_b_l127_127065


namespace cos_105_proof_l127_127856

noncomputable def cos_105_degrees : Real :=
  cos 105 * (π / 180)

theorem cos_105_proof : cos_105_degrees = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_proof_l127_127856


namespace power_division_identity_l127_127738

theorem power_division_identity : 
  ∀ (a b c : ℕ), a = 3 → b = 12 → c = 2 → (3 ^ 12 / (3 ^ 2) ^ 2 = 6561) :=
by
  intros a b c h1 h2 h3
  sorry

end power_division_identity_l127_127738


namespace sum_roots_zero_l127_127017

-- Let Q be a quadratic polynomial
def Q (x : ℝ) : ℝ

-- Define the condition for Q
def condition (Q : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, Q(x^3 - x) ≥ Q(x^2 - 1)

-- Define the property to prove
theorem sum_roots_zero (Q : ℝ → ℝ) [quad : ∀ x ∈ ℝ, polynomial.degree Q ≤ 2] 
  (hQ : condition Q) : 
  ∑ root in (polynomial.roots Q), root = 0 := 
sorry

end sum_roots_zero_l127_127017


namespace angle_NQP_measure_l127_127969

theorem angle_NQP_measure (JK_par_PQ : Parallel JK PQ) 
  (angle_JKN_eq_a : ∠ JKN = a) 
  (angle_KNJ_eq_3a : ∠ KNJ = 3 * a) 
  (angle_NPQ_eq_2a : ∠ NPQ = 2 * a) : 
  ∠ NQP = 45 :=
by
  sorry

end angle_NQP_measure_l127_127969


namespace smallest_c_sequence_inequality_l127_127909

theorem smallest_c_sequence_inequality : 
  ∃ (a : ℕ → ℕ) (c : ℕ), 
    (∀ n : ℕ, n ≥ 1 → a n > 0) ∧ 
    (∀ n : ℕ, n ≥ 1 → (finset.range (n + 1)).sum a < c * a n) ∧ 
    c = 4 :=
sorry

end smallest_c_sequence_inequality_l127_127909


namespace time_to_pass_man_l127_127057

open Real

noncomputable def train_length : ℝ := 55
noncomputable def train_speed : ℝ := 60
noncomputable def man_speed : ℝ := 6
noncomputable def relative_speed : ℝ := (train_speed + man_speed) * (5 / 18) -- convert km/h to m/s
noncomputable def time_to_pass : ℝ := train_length / relative_speed

theorem time_to_pass_man (train_length train_speed man_speed : ℝ) 
  (h1 : train_length = 55) 
  (h2 : train_speed = 60) 
  (h3 : man_speed = 6) : 
  time_to_pass = 3 :=
by
  rw [h1, h2, h3]
  unfold relative_speed
  simp
  sorry

end time_to_pass_man_l127_127057


namespace binom_20_5_l127_127488

-- Definition of the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Problem statement
theorem binom_20_5 : binomial_coefficient 20 5 = 7752 := 
by {
  -- Proof goes here
  sorry
}

end binom_20_5_l127_127488


namespace cube_root_solution_l127_127339

theorem cube_root_solution (y : ℝ) (h : real.cbrt (5 + 2 / y) = 3) : y = 1 / 11 :=
by
  -- Proof omitted:
  sorry

end cube_root_solution_l127_127339


namespace smallest_nonprime_consecutive_l127_127170

theorem smallest_nonprime_consecutive :
  ∃ n : ℕ, (∀ i : ℕ, 0 ≤ i ∧ i < 5 → prime (n + i) = false) ∧ n > 90 ∧ n < 100 ∧ n = 90 :=
begin
  sorry
end

end smallest_nonprime_consecutive_l127_127170


namespace cost_of_adult_ticket_l127_127036

/-- University Theater sold a total of 529 tickets. Each senior citizen ticket costs $15. 
The total receipts were $9745, and they sold 348 senior citizen tickets. Prove the cost of an adult ticket. --/

theorem cost_of_adult_ticket 
  (total_tickets : ℕ) 
  (senior_ticket_cost : ℕ) 
  (total_receipts : ℕ) 
  (senior_tickets_sold : ℕ) 
  (adult_ticket_cost : ℕ) :
  total_tickets = 529 →
  senior_ticket_cost = 15 →
  total_receipts = 9745 →
  senior_tickets_sold = 348 →
  (total_receipts - (senior_tickets_sold * senior_ticket_cost)) / (total_tickets - senior_tickets_sold) = 25 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact sorry

end cost_of_adult_ticket_l127_127036


namespace correct_multiplication_l127_127440

theorem correct_multiplication (n : ℕ) (wrong_answer correct_answer : ℕ) 
    (h1 : wrong_answer = 559981)
    (h2 : correct_answer = 987 * n)
    (h3 : ∃ (x y : ℕ), correct_answer = 500000 + x + 901 + y ∧ x ≠ 98 ∧ y ≠ 98 ∧ (wrong_answer - correct_answer) % 10 = 0) :
    correct_answer = 559989 :=
by
  sorry

end correct_multiplication_l127_127440


namespace neg_one_pow_2019_l127_127485

theorem neg_one_pow_2019 : (-1 : ℝ)^2019 = -1 := 
begin
  sorry
end

end neg_one_pow_2019_l127_127485


namespace sum_four_variables_l127_127293

theorem sum_four_variables 
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + 2 = x)
  (h2 : b + 3 = x)
  (h3 : c + 4 = x)
  (h4 : d + 5 = x)
  (h5 : a + b + c + d + 8 = x) :
  a + b + c + d = -6 :=
by
  sorry

end sum_four_variables_l127_127293


namespace correct_multiple_l127_127597

theorem correct_multiple (n : ℝ) (m : ℝ) (h1 : n = 6) (h2 : m * n - 6 = 2 * n) : m * n = 18 :=
by
  sorry

end correct_multiple_l127_127597


namespace total_spent_by_mrs_hilt_l127_127663

-- Define the cost per set of tickets for kids.
def cost_per_set_kids : ℕ := 1
-- Define the number of tickets in a set for kids.
def tickets_per_set_kids : ℕ := 4

-- Define the cost per set of tickets for adults.
def cost_per_set_adults : ℕ := 2
-- Define the number of tickets in a set for adults.
def tickets_per_set_adults : ℕ := 3

-- Define the total number of kids' tickets purchased.
def total_kids_tickets : ℕ := 12
-- Define the total number of adults' tickets purchased.
def total_adults_tickets : ℕ := 9

-- Prove that the total amount spent by Mrs. Hilt is $9.
theorem total_spent_by_mrs_hilt :
  (total_kids_tickets / tickets_per_set_kids * cost_per_set_kids) + 
  (total_adults_tickets / tickets_per_set_adults * cost_per_set_adults) = 9 :=
by sorry

end total_spent_by_mrs_hilt_l127_127663


namespace midpoint_of_PQ_l127_127652

variables {A B C D E F P Q : Point}
variables {triangle : Triangle}

-- Definitions for the conditions given in the problem
def incircle_touches_sides (triangle : Triangle) (D E F : Point) : Prop :=
  touches triangle.incircle (triangle.sides • BC) D ∧ 
  touches triangle.incircle (triangle.sides • CA) E ∧ 
  touches triangle.incircle (triangle.sides • AB) F

def intersection_ED_perpendicular_to_EF (D E F P : Point) : Prop :=
  ∃ (l : Line), intersects (line_through E D) l P ∧ perpendicular l (line_through F E)

def intersection_EF_perpendicular_to_ED (D E F Q : Point) : Prop :=
  ∃ (m : Line), intersects (line_through F E) m Q ∧ perpendicular m (line_through D E)

-- The main theorem statement
theorem midpoint_of_PQ (triangle : Triangle) (D E F P Q : Point) :
  incircle_touches_sides triangle D E F →
  intersection_ED_perpendicular_to_EF D E F P →
  intersection_EF_perpendicular_to_ED D E F Q →
  midpoint B P Q :=
sorry

end midpoint_of_PQ_l127_127652


namespace minimum_questionnaires_l127_127120

theorem minimum_questionnaires (X : ℕ) (hX : 342 ≤ X) :
  0.70 * (0.40 * X) + 0.60 * (0.35 * X) + 0.50 * (0.25 * X) ≥ 210 := by
sorry

end minimum_questionnaires_l127_127120


namespace tail_length_l127_127267

variable (Length_body Length_tail Length_head : ℝ)

-- Conditions
def tail_half_body (Length_tail Length_body : ℝ) := Length_tail = 1/2 * Length_body
def head_sixth_body (Length_head Length_body : ℝ) := Length_head = 1/6 * Length_body
def overall_length (Length_head Length_body Length_tail : ℝ) := Length_head + Length_body + Length_tail = 30

-- Theorem statement
theorem tail_length (h1 : tail_half_body Length_tail Length_body) 
                  (h2 : head_sixth_body Length_head Length_body) 
                  (h3 : overall_length Length_head Length_body Length_tail) : 
                  Length_tail = 6 := by
  sorry

end tail_length_l127_127267


namespace calculate_lives_lost_l127_127050

-- Define the initial number of lives
def initial_lives : ℕ := 98

-- Define the remaining number of lives
def remaining_lives : ℕ := 73

-- Define the number of lives lost
def lives_lost : ℕ := initial_lives - remaining_lives

-- Prove that Kaleb lost 25 lives
theorem calculate_lives_lost : lives_lost = 25 := 
by {
  -- The proof would go here, but we'll skip it
  sorry
}

end calculate_lives_lost_l127_127050


namespace perpendicular_planes_l127_127628

-- Definitions for the coordinate planes
def yoz_plane : Type := { p : ℝ × ℝ × ℝ // p.1 = 0 }
def xoz_plane : Type := { p : ℝ × ℝ × ℝ // p.2 = 0 }
def xoy_plane : Type := { p : ℝ × ℝ × ℝ // p.3 = 0 }

-- Theorem to prove the perpendicularity of each respective plane to the coordinate axes
theorem perpendicular_planes :
  (∀ p : ℝ × ℝ × ℝ, p ∈ yoz_plane ↔ p.1 = 0) ∧
  (∀ p : ℝ × ℝ × ℝ, p ∈ xoz_plane ↔ p.2 = 0) ∧
  (∀ p : ℝ × ℝ × ℝ, p ∈ xoy_plane ↔ p.3 = 0) :=
by
  sorry

end perpendicular_planes_l127_127628


namespace functional_equation_solution_l127_127456

noncomputable def f : ℝ → ℝ := sorry 

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) →
  10 * f 2006 + f 0 = 20071 :=
by
  intros h
  sorry

end functional_equation_solution_l127_127456


namespace part1_monotonicity_part2_range_of_a_l127_127569

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x / log x - a * x

theorem part1_monotonicity (a : ℝ) (h_a : a = 0) :
  (∀ x > e, (f x a) / (f x a) > 0) ∧ 
  (∀ x ∈ (Ioo 0 1) ∪ (Ioo 1 e), (f x a) / (f x a) < 0) := 
by
  sorry

theorem part2_range_of_a (x1 : ℝ) (h_x1 : x1 ∈ Icc e (exp 2)) (h_f : f x1 0 ≤ 1/4) :
  ∃ a ≥ (1/2 - 1/(4 * (exp 1) ^ 2)), a ∈ Icc 0 (1/2 - 1/(4 * (exp 1) ^ 2)) :=
by
  sorry

end part1_monotonicity_part2_range_of_a_l127_127569


namespace equal_playing_time_for_each_player_l127_127790

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l127_127790


namespace point_in_fourth_quadrant_l127_127948

def complex_quadrant_check (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
  else "On an axis"

theorem point_in_fourth_quadrant : complex_quadrant_check ((1 - 3*complex.i) * (2 + complex.i)) = "Fourth quadrant" :=
by
  sorry

end point_in_fourth_quadrant_l127_127948


namespace find_f_2011_5_l127_127881

noncomputable def f : ℝ → ℝ :=
  sorry

lemma f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) :=
sorry

theorem find_f_2011_5 : f 2011.5 = -0.5 :=
by
  have h := f_properties
  sorry

end find_f_2011_5_l127_127881


namespace equal_playing_time_for_each_player_l127_127789

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l127_127789


namespace exists_sum_full_zero_sum_free_set_l127_127099

def fibonacci : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

def A : Set ℤ := {a | ∃ n ≥ 2, a = (-1)^n * fibonacci n}

lemma sum_full (A : Set ℤ) : A ⊆ {a + b | a b ∈ A} :=
sorry

lemma zero_sum_free (A : Set ℤ) : ∀ B : Finset ℤ, (B ⊆ A ∧ B.sum id = 0) → B = ∅ :=
sorry

theorem exists_sum_full_zero_sum_free_set : ∃ A : Set ℤ, 
  (A ⊆ {a + b | a b ∈ A}) ∧ (∀ B : Finset ℤ, (B ⊆ A ∧ B.sum id = 0) → B = ∅) :=
exists.intro A ⟨sum_full A, zero_sum_free A⟩

end exists_sum_full_zero_sum_free_set_l127_127099


namespace work_completion_days_l127_127075

theorem work_completion_days (A B C : ℕ) (work_rate_A : A = 4) (work_rate_B : B = 10) (work_rate_C : C = 20 / 3) :
  (1 / A) + (1 / B) + (3 / C) = 1 / 2 :=
by
  sorry

end work_completion_days_l127_127075


namespace magnitude_of_projection_l127_127644

variable (v w : ℝ^3)

-- Given conditions
variable (dot_vw : v ⬝ w = 6)
variable (norm_w : ∥w∥ = 4)

-- Problem statement
theorem magnitude_of_projection : ∥(v ⬝ w / ∥w∥^2) • w∥ = 6 :=
by
  sorry

end magnitude_of_projection_l127_127644


namespace shifted_parabola_l127_127005

theorem shifted_parabola (x : ℝ) : 
  let original_parabola := fun x : ℝ => x^2 in
  let shifted_parabola := fun x : ℝ => (x + 2)^2 in
  (∀ x, original_parabola (x - 2) = shifted_parabola x) :=
by 
  sorry

end shifted_parabola_l127_127005


namespace find_real_solutions_l127_127158

theorem find_real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 7 ∨ x = -2) := 
by
  sorry

end find_real_solutions_l127_127158


namespace antecedent_is_50_l127_127247

theorem antecedent_is_50 (antecedent consequent : ℕ) (h_ratio : 4 * consequent = 6 * antecedent) (h_consequent : consequent = 75) : antecedent = 50 := by
  sorry

end antecedent_is_50_l127_127247


namespace window_width_proof_l127_127359

noncomputable def window_width (length width height door_height door_width cost_per_sqft total_cost : ℝ) : ℝ :=
  let perimeter := 2 * (length + width)
  let wall_area := perimeter * height
  let door_area := door_height * door_width
  let total_area_to_be_whitewashed := wall_area - door_area - 3 * (door_height * window_width)
  let cost_eq := total_cost = cost_per_sqft * total_area_to_be_whitewashed
  let w := which_satisfies_w(cost_eq)
  w

theorem window_width_proof :
  window_width 25 15 12 6 3 8 7248 = 4 :=
by
  sorry

end window_width_proof_l127_127359


namespace factor_polynomial_l127_127514

theorem factor_polynomial :
  4 * (x + 5) * (x + 6) * (x + 10) * (x + 12) - 3 * x^2 = 
  (2 * x^2 + 35 * x + 120) * (x + 8) * (2 * x + 15) := 
by sorry

end factor_polynomial_l127_127514


namespace intersection_distance_l127_127990

-- Definitions for the parameterized curve C1
def C1 (t : ℝ) : ℝ × ℝ := ( (sqrt 5 / 5) * t, (2 * sqrt 5 / 5) * t - 1 )

-- Definition for the curve C2 in polar coordinates, converted to rectangular coordinates
def C2 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * Real.cos θ - 4 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Distance formula function
def distance_between_points (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- The theorem to be proven
theorem intersection_distance :
  ∃ t θ, 
  let p1 := C1 t in
  let p2 := C2 θ in
  (p1 = (1, -2) ⊓ p2 = (0, -1) →
  distance_between_points p1 p2 = 8 * sqrt 5 / 5) :=
sorry

end intersection_distance_l127_127990


namespace solve_rect_eq_and_chord_length_l127_127218

-- Definition of the parametric equation of line l
def line_param (t : ℝ) : ℝ × ℝ := 
  (2 + t, -1 + (sqrt 3) * t)

-- Definition of the polar equation of curve C
def curve_polar (θ : ℝ) : ℝ := 
  2 * sin θ + 4 * cos θ

-- Definition of the rectangular coordinate equation of curve C
def curve_rect_eq (x y : ℝ) : Prop := 
  (x - 2)^2 + (y - 1)^2 = 5

-- Definition of the standard equation of line l
def line_standard_eq (x y : ℝ) : Prop := 
  (sqrt 3) * x - y - 2 * (sqrt 3) - 1 = 0

-- Final proof problem: Proving the curve_rect_eq and the chord length
theorem solve_rect_eq_and_chord_length (x y t θ : ℝ) (α : ℝ) :
  (curve_polar θ = sqrt (x^2 + y^2) →
  x = curve_polar θ * cos θ →
  y = curve_polar θ * sin θ →
  curve_rect_eq x y) ∧
  (line_standard_eq (2 + t) (-1 + (sqrt 3) * t) →
  (sqrt (5) = 2 * sqrt ((2 + sqrt 5 * cos α) - 2)^2 + ((1 + sqrt 5 * sin α) - 1)^2)) →
  sqrt ((2 + sqrt 3) - 2 * (sqrt 3) - 1)^2 / 2 = 1 →
  2 * sqrt (((sqrt 5)^2 - 1)) = 4 :=
by
  sorry

end solve_rect_eq_and_chord_length_l127_127218


namespace ellipse_equation_and_max_triangle_area_l127_127558

-- Define the conditions
def eccentricity : ℝ := sqrt 2 / 2
def chord_length : ℝ := sqrt 2

theorem ellipse_equation_and_max_triangle_area :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ 
    (eccentricity = sqrt (1 - (b/a)^2)) ∧ 
    is_eq (chord_length^2 / a) sqrt 2 ∧ 
    is_eq (a, b) (sqrt 2, 1) ∧ 
    (∀ m : ℝ, abs m < sqrt 3 ∧ m ≠ 0 → 
      area_of_triangle_PAB (eq_of_ellipse a b) m = sqrt 2 / 3)) :=
begin
  -- No proof required: just the statement
  sorry
end

end ellipse_equation_and_max_triangle_area_l127_127558


namespace total_noodles_and_pirates_l127_127624

-- Condition definitions
def pirates : ℕ := 45
def noodles : ℕ := pirates - 7

-- Theorem stating the total number of noodles and pirates
theorem total_noodles_and_pirates : (noodles + pirates) = 83 := by
  sorry

end total_noodles_and_pirates_l127_127624


namespace asymptotes_of_hyperbola_l127_127901

def hyperbola_asymptotes (x y : ℝ) : Prop :=
  x^2 - 2 * y^2 = 1

theorem asymptotes_of_hyperbola :
  ∀ x y : ℝ, hyperbola_asymptotes x y → y = (√2 / 2) * x ∨ y = -(√2 / 2) * x :=
by
  intros x y h
  sorry

end asymptotes_of_hyperbola_l127_127901


namespace sum_of_roots_l127_127742

theorem sum_of_roots (a b c d : ℝ) (h : a = 1 ∧ b = -7 ∧ c = 3 ∧ d = 11) (sum_roots_eq : a * x^2 + b * x + c = d) : 
  (roots : List ℝ) [∀ x, root x ↔ sum_roots_eq] →
  roots.sum = 7 :=
by
  sorry

end sum_of_roots_l127_127742


namespace find_lower_concentration_l127_127771

-- Define initial parameters
def total_volume : ℝ := 2
def desired_concentration : ℝ := 0.08
def volume_10_percent : ℝ := 1.2
def concentration_10_percent : ℝ := 0.10

-- Define the target concentration calculation
def target_pure_acid := total_volume * desired_concentration
def pure_acid_10_percent := volume_10_percent * concentration_10_percent
def remaining_volume : ℝ := total_volume - volume_10_percent

-- Define unknown percentage in terms of x
def lower_concentration_acid (x : ℝ) := x / 100 * remaining_volume

-- The proof problem
theorem find_lower_concentration : ∃ x : ℝ, pure_acid_10_percent + lower_concentration_acid x = target_pure_acid ∧ x = 5 := sorry

-- substitute by the corresponding code to make sure that the Lean statement is correct
#eval find_lower_concentration

end find_lower_concentration_l127_127771


namespace rectangle_vertices_l127_127275

variables (C1 C2 : Circle) (A B S T P : Point)
variables (C D E F : Point)

-- Assume the initial conditions
variables (h1 : intersect C1 C2 A B)
variables (h2 : center C1 S)
variables (h3 : center C2 T)
variables (h4 : on_segment P A B)
variables (h5 : AP ≠ BP)
variables (h6 : P ≠ A ∧ P ≠ B)
variables (h7 : intersect_line_through_point_perpendicular_to SP P C D)
variables (h8 : intersect_line_through_point_perpendicular_to TP P E F)

-- We need to prove that C, D, E, and F are the vertices of a rectangle
theorem rectangle_vertices : rectangle C D E F :=
sorry

end rectangle_vertices_l127_127275


namespace probability_non_adjacent_ones_l127_127230

theorem probability_non_adjacent_ones : 
  let totalWays := Nat.choose 5 3 in
  let favorableWays := Nat.choose 4 2 in
  favorableWays.toRational / totalWays.toRational = 3 / 5 :=
by
  sorry

end probability_non_adjacent_ones_l127_127230


namespace compute_CD_length_l127_127664

noncomputable def CD_length (a : ℝ) (h : a ≥ sqrt 7) : ℝ :=
  |4 - sqrt (a^2 - 7)|

theorem compute_CD_length (BC AC AD : ℝ) (a : ℝ) (hBC : BC = 3) (hAC : AC = a) (hAD : AD = 4) (ha : a ≥ sqrt 7) :
  let AB := sqrt (a^2 + 9)
  in |AD - sqrt ((AB^2) - (AD^2))| = CD_length a ha :=
by
  intros
  simp only [BC, AC, AD, hBC, hAC, hAD]
  let BD := sqrt (a^2 - 7)
  have : BD = sqrt (a^2 - 7), sorry
  simp only [CD_length, this]
  sorry

end compute_CD_length_l127_127664


namespace equal_playing_time_for_each_player_l127_127787

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l127_127787


namespace smallest_element_in_S_l127_127191

open Set

theorem smallest_element_in_S : 
  ∃ S : Set ℕ, S ⊆ {x | 1 ≤ x ∧ x ≤ 12} ∧ S.card = 6 ∧ 
  (∀ a b, a ∈ S → b ∈ S → a < b → ¬ (b % a = 0)) ∧
  (4 = S.min' (by { sorry })) := 
by {
  -- Placeholder for proof, which is not required in this task
  sorry
}

end smallest_element_in_S_l127_127191


namespace min_value_of_squares_l127_127294

noncomputable def f (x : ℝ) : ℝ := abs (1 / 2 * x + 1) + abs x

lemma min_value_of_f : (∃ x : ℝ, f(x) = 1) ∧ (∀ x : ℝ, f(x) ≥ 1) := 
sorry

theorem min_value_of_squares (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
(h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 :=
by
  have h1 : (p^2 + q^2 + r^2) * (1^2 + 1^2 + 1^2) ≥ (p + q + r)^2, from
    -- Cauchy-Schwarz inequality for (p, q, r) and (1, 1, 1)
    sorry,
  have h2 : (p + q + r)^2 = 9, from
    calc (p + q + r)^2 = 3^2 : by rw [h]
                        ... = 9 : by norm_num,
  have h3 : (1^2 + 1^2 + 1^2) = 3, from by norm_num,
  have h4 : (p^2 + q^2 + r^2) * 3  ≥ 9, from by rwa [h3, h2] at h1,
  have h5 : (p^2 + q^2 + r^2) ≥ 3, from by linarith,
  exact h5

end min_value_of_squares_l127_127294


namespace betty_watermelons_l127_127125

theorem betty_watermelons :
  ∃ b : ℕ, 
  (b + (b + 10) + (b + 20) + (b + 30) + (b + 40) = 200) ∧
  (b + 40 = 60) :=
by
  sorry

end betty_watermelons_l127_127125


namespace x_intercept_of_line_l127_127164

theorem x_intercept_of_line (x y : ℚ) (h : 4 * x + 6 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by
  sorry

end x_intercept_of_line_l127_127164


namespace domain_of_g_l127_127401

noncomputable def g (x : ℝ) : ℝ := (x^2 + 4*x + 4) / (sqrt (2*x^2 - 5*x - 3))

theorem domain_of_g :
  { x : ℝ | ∃ y : ℝ, g x = y } = { x : ℝ | x ≤ -1/2 ∨ x ≥ 3 } :=
by
  sorry

end domain_of_g_l127_127401


namespace sum_first_n_terms_eq_l127_127190

-- Definitions based on conditions
def a_seq : ℕ → ℤ
| 0     := 2
| (n+1) := 2 * (a_seq n + 3) - 3

-- Main theorem statement
theorem sum_first_n_terms_eq (n : ℕ) : 
  (∑ i in Finset.range n, a_seq i) = 5 * 2^n - 3 * n - 5 :=
sorry

end sum_first_n_terms_eq_l127_127190


namespace compare_powers_l127_127701

theorem compare_powers (a b c d : ℝ) 
  (h1 : ∀ x, x^2 + a * x + b = x^2 + c * x + d → x = 1 → (x, 1) ∈ set_of (λ x, x^2 + a * x + b = 1)) :
  a^5 + d^6 = c^6 - b^5 :=
by
  have h_ab : a + b = 0, from 
    calc 1 = 1 + a + b : by rw [h1 1]
           ... = 1 + a + b : by simp
  have h_cd : c + d = 0, from 
    calc 1 = 1 + c + d : by rw [h1 1]
           ... = 1 + c + d : by simp
  have ha : a = -b, from eq_neg_of_add_eq_zero_left h_ab
  have hc : c = -d, from eq_neg_of_add_eq_zero_left h_cd
  calc
    a^5 + d^6 = (-b)^5 + (-c)^6 : by rw [ha, hc]
           ... = -b^5 + c^6   : by rw [neg_pow, neg_pow, even_pow_of_even_index (succ_ne_zero 6)]
           ... = c^6 - b^5    : by ring

end compare_powers_l127_127701


namespace sum_f_1_2021_l127_127561

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom equation_f : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom interval_f : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f x = Real.log (1 - x) / Real.log 2

theorem sum_f_1_2021 : (List.sum (List.map f (List.range' 1 2021))) = -1 := sorry

end sum_f_1_2021_l127_127561


namespace equal_subset_sum_probability_l127_127375

theorem equal_subset_sum_probability
  (set1 set2 : Finset ℕ)
  (h : (Finset.range 8).erase 0 = set1 ∪ set2 ∧ set1 ≠ ∅ ∧ set2 ≠ ∅) :
  Finset.sum set1 id = Finset.sum set2 id →
  (fst ((4:ℚ)/(63:ℚ)).num + (4:ℚ)/(63:ℚ).den) = 67 :=
by sorry

end equal_subset_sum_probability_l127_127375


namespace work_completion_time_l127_127745

/-- 
Given:
1. a and b together can complete the work in 10 days.
2. a alone can complete the work in 14 days.
3. c alone can complete the work in 21 days.
Then, the time it takes for a, b, and c to complete the work together is 210 / 31 days.
-/
theorem work_completion_time (W : ℕ) (t_ab t_a t_c : ℕ) (H1 : t_ab = 10) (H2 : t_a = 14) (H3 : t_c = 21) :
  ∀ t_abc : ℚ, t_abc = 210 / 31 :=
begin
  sorry,
end

end work_completion_time_l127_127745


namespace carrie_hours_per_week_l127_127847

variable (H : ℕ)

def carrie_hourly_wage : ℕ := 8
def cost_of_bike : ℕ := 400
def amount_left_over : ℕ := 720
def weeks_worked : ℕ := 4
def total_earnings : ℕ := cost_of_bike + amount_left_over

theorem carrie_hours_per_week :
  (weeks_worked * H * carrie_hourly_wage = total_earnings) →
  H = 35 := by
  sorry

end carrie_hours_per_week_l127_127847


namespace negation_equivalence_l127_127705

-- Define the original proposition P
def P : Prop := ∃ x₀ : ℝ, 2 ^ x₀ < 1 / 2 ∨ x₀ ^ 2 > x₀

-- State the equivalence of the negation of P to the universal proposition
theorem negation_equivalence : ¬ P ↔ ∀ x : ℝ, 2 ^ x ≥ 1 / 2 ∧ x ^ 2 ≤ x :=
by {
  -- Proof will be provided here.
  sorry
}

end negation_equivalence_l127_127705


namespace pumpkin_to_spiderwebs_ratio_l127_127497

theorem pumpkin_to_spiderwebs_ratio :
  let skulls := 12
  let broomsticks := 4
  let spiderwebs := 12
  let cauldron := 1
  let buy_more := 20
  let left_to_put := 10
  let total_decorations := 83
  let pumpkins := total_decorations - (skulls + broomsticks + spiderwebs + cauldron + buy_more + left_to_put)
  (pumpkins : ℕ) : 2 * spiderwebs = pumpkins := 
by
  let skulls := 12
  let broomsticks := 4
  let spiderwebs := 12
  let cauldron := 1
  let buy_more := 20
  let left_to_put := 10
  let total_decorations := 83
  let pumpkins := total_decorations - (skulls + broomsticks + spiderwebs + cauldron + buy_more + left_to_put)
  have h_pumpkins_ncalc : pumpkins = 24 := by sorry
  rw [← h_pumpkins_ncalc]
  norm_num
  sorry

end pumpkin_to_spiderwebs_ratio_l127_127497


namespace integral_equals_two_plus_pi_l127_127976

noncomputable def complex_number_is_real (a : ℝ) : Prop :=
  let z := complex.mk a (a - 2)
  z.im = 0

noncomputable def integral_expression (a : ℝ) : ℝ :=
  ∫ x in 0..a, real.sqrt (4 - x^2) + x

theorem integral_equals_two_plus_pi (a : ℝ) (h : complex_number_is_real a) :
  integral_expression a = 2 + real.pi :=
begin
  sorry
end

end integral_equals_two_plus_pi_l127_127976


namespace age_difference_l127_127242

theorem age_difference :
    ∀ (A B : ℕ), B = 70 ∧ A + 20 = 2 * (B - 20) → A - B = 10 := by
  intros A B h
  rcases h with ⟨hB, hEquation⟩
  have h1 : B = 70 := hB
  have h2 : A + 20 = 2 * (B - 20) := hEquation
  rw h1 at h2
  calc
    A = 100 - 20 : by linarith
    _ = 80 : by norm_num
    A - B = 80 - 70 : by rw h1
    _ = 10 : by norm_num

end age_difference_l127_127242


namespace trapezoid_area_l127_127427

-- Definitions of the problem
noncomputable def area_of_trapezoid (AC BD a : ℝ) :=
  let α := acos ((7/10) : ℝ) in
  let sin_α := real.sqrt (51/100 : ℝ) in
  let sin_3α := 27 * sin_α / 125 in
  (7 * a^2 * sin_3α) / 10

-- Proving the area of the trapezoid
theorem trapezoid_area (a : ℝ) (h₁ : a > 0) (h₂ : cos (real.to_angle (2 * angle.DBA)) = 7 / 10) :
  area_of_trapezoid a (7 / 5 * a) a = (42 * real.sqrt 51 / 625) * a^2 :=
begin
  sorry,
end

end trapezoid_area_l127_127427


namespace find_a_star_b_l127_127501

-- Given conditions
variables (a b : ℝ)
-- these conditions must be true for the proof to hold
def conditions := (a + b = 16) ∧ (a^2 + b^2 = 136)

-- Define the operation 
def operation (a b : ℝ) := (1 / a + 1 / b)

-- The theorem to prove
theorem find_a_star_b (h : conditions a b) : operation a b = 4 / 15 := by
  sorry

end find_a_star_b_l127_127501


namespace monopoly_durable_only_iff_competitive_market_durable_preference_iff_l127_127755

variable (C : ℝ)

-- Definitions based on conditions
def consumer_benefit_period : ℝ := 10
def durable_machine_periods : ℝ := 2
def low_quality_machine_periods : ℝ := 1
def durable_machine_cost : ℝ := 6

-- Statements based on extracted questions & correct answers
theorem monopoly_durable_only_iff (H : C > 3) :
  let durable_benefit := durable_machine_periods * consumer_benefit_period
      durable_price := durable_benefit
      durable_profit := durable_price - durable_machine_cost
      low_quality_price := consumer_benefit_period
      low_quality_profit := low_quality_price - C in
  durable_profit > durable_machine_periods * (low_quality_profit) :=
by 
  sorry

theorem competitive_market_durable_preference_iff (H : C > 3) :
  let durable_benefit := durable_machine_periods * consumer_benefit_period
      durable_surplus := durable_benefit - durable_machine_cost
      low_quality_surplus := low_quality_machine_periods * (consumer_benefit_period - C) in
  durable_surplus > durable_machine_periods * low_quality_surplus :=
by 
  sorry

end monopoly_durable_only_iff_competitive_market_durable_preference_iff_l127_127755


namespace incircle_tangent_ration_l127_127220

variable {α : Type} [IsAffine α]

/-- Given a triangle ABC with incircle tangent to BC, CA, AB at D, E, F, respectively.
    Let G be a point on EF such that DG is perpendicular to EF.
    Then prove that FG / EG = BF / CE. -/
theorem incircle_tangent_ration (A B C D E F G : α) (h : IsTriangle A B C)
  (h_tangent_D : IsIncircleTangentAt D B C)
  (h_tangent_E : IsIncircleTangentAt E C A)
  (h_tangent_F : IsIncircleTangentAt F A B) 
  (h_on_EF : LiesOnLineSeg G E F)
  (h_perp_DG_EF : Perpendicular (LineThroughPoints D G) (LineThroughPoints E F)) :
  (dist F G / dist E G) = (dist B F / dist C E) :=
by
  sorry

end incircle_tangent_ration_l127_127220


namespace series_sum_l127_127145

noncomputable def sum_signed_series : ℤ :=
  let series : ℕ → ℤ := λ n, if n % 6 < 3 then n else -n
  (∑ n in Finset.range 10004, series n)

theorem series_sum : sum_signed_series = 36118 := by
  -- Assume necessary computation steps here
  sorry

end series_sum_l127_127145


namespace largest_prime_factor_of_expression_l127_127503

theorem largest_prime_factor_of_expression : 
  ∀ (a b c : ℕ), a = 12^3 → b = 15^4 → c = 6^5 → 
  (prime_factors (a + b - c)).max = 11 := by
  sorry

end largest_prime_factor_of_expression_l127_127503


namespace total_pieces_l127_127532

-- Define the given conditions
def pieces_eaten_per_person : ℕ := 4
def num_people : ℕ := 3

-- Theorem stating the result
theorem total_pieces (h : num_people > 0) : (num_people * pieces_eaten_per_person) = 12 := 
by
  sorry

end total_pieces_l127_127532


namespace largest_lcm_value_l127_127402

-- Define the conditions as local constants 
def lcm_18_3 : ℕ := Nat.lcm 18 3
def lcm_18_6 : ℕ := Nat.lcm 18 6
def lcm_18_9 : ℕ := Nat.lcm 18 9
def lcm_18_15 : ℕ := Nat.lcm 18 15
def lcm_18_21 : ℕ := Nat.lcm 18 21
def lcm_18_27 : ℕ := Nat.lcm 18 27

-- Statement to prove
theorem largest_lcm_value : max lcm_18_3 (max lcm_18_6 (max lcm_18_9 (max lcm_18_15 (max lcm_18_21 lcm_18_27)))) = 126 :=
by
  -- We assume the necessary calculations have been made
  have h1 : lcm_18_3 = 18 := by sorry
  have h2 : lcm_18_6 = 18 := by sorry
  have h3 : lcm_18_9 = 18 := by sorry
  have h4 : lcm_18_15 = 90 := by sorry
  have h5 : lcm_18_21 = 126 := by sorry
  have h6 : lcm_18_27 = 54 := by sorry

  -- Using above results to determine the maximum
  exact (by rw [h1, h2, h3, h4, h5, h6]; exact rfl)

end largest_lcm_value_l127_127402


namespace not_perfect_square_l127_127048

theorem not_perfect_square (a b c d e : ℕ) (h1 : a = 1^6)
                                (h2 : b = 2^5)
                                (h3 : c = 3^4)
                                (h4 : d = 4^3)
                                (h5 : e = 5^2) :
  ¬(∃ k : ℕ, k^2 = b) :=
by {
  have h2_val : b = 32 := by rw[h2],
  sorry
}

end not_perfect_square_l127_127048


namespace solve_problem_l127_127600

noncomputable def f : ℝ → ℝ := sorry

axiom f_periodicity (x : ℝ) : f(x+3) = -f(x+1)
axiom f_initial_condition : f(2) = 2014

theorem solve_problem : f(f(2014) + 2) + 3 = -2011 := 
by 
  sorry 

end solve_problem_l127_127600


namespace equal_area_trapezoid_l127_127780

open Set

def isMidpoint (P A B : Point) : Prop := (P.x = (A.x + B.x) / 2) ∧ (P.y = (A.y + B.y) / 2)

theorem equal_area_trapezoid
  (A B C D M N P : Point)
  (h_trapezoid : isTrapezoid A B C D)
  (h_M_midpoint : isMidpoint M B C)
  (h_N_midpoint : isMidpoint N A D)
  (h_P_on_MN : P ∈ lineSegment M N) :
  area (triangle P A B) = area (triangle P C D) := 
sorry

end equal_area_trapezoid_l127_127780


namespace rational_linear_function_l127_127155

theorem rational_linear_function (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
sorry

end rational_linear_function_l127_127155


namespace simplify_and_evaluate_l127_127338

theorem simplify_and_evaluate:
  ∀ (a : ℝ), a ≠ -1 ∧ a ≠ 2 ∧ a = 1 → 
  ( ( (3 / (a + 1)) - a + 1 ) / ( (a^2 - 4 * a + 4) / (a + 1) ) ) = 3 :=
by
  intro a
  rintro ⟨h₁, h₂, ha⟩
  rw ha
  sorry

end simplify_and_evaluate_l127_127338


namespace midpoint_of_altitudes_l127_127369

variables {A B C I C' B' C_1 B_1 H : Point}
variables {ABC : Triangle A B C}
variables {I_center : Incenter ABC}
variables (h1 : LineThroughPerpendicular I A C' B')
variables (h2 : Altitude BC'I C'C_1)
variables (h3 : Altitude CB'I B'B_1)
variables (M : Point)

theorem midpoint_of_altitudes (h : Midpoint B_1 C_1 M) :
  LiesOn (LineThroughPerpendicular I BC) M :=
sorry

end midpoint_of_altitudes_l127_127369


namespace equation_of_common_chord_length_of_common_chord_l127_127553

-- Define a helper function for the distance from a point to a line
noncomputable def distance_point_line (A B C : ℝ) (x0 y0 : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- Define the centers and radii of the circles
def center₁ : ℝ × ℝ := (1, 0)
def center₂ : ℝ × ℝ := (0, sqrt 3)
def radius₁ : ℝ := 2
def radius₂ : ℝ := 3

theorem equation_of_common_chord :
  ∀ (x y : ℝ), (x, y) ∈ (λ p : ℝ × ℝ, (p.1 - 1)^2 + p.2^2 = 4) ∧ (x, y) ∈ (λ p : ℝ × ℝ, p.1^2 + (p.2 - sqrt 3)^2 = 9) 
  → 2 * x - 2 * sqrt 3 * y - 3 = 0 :=
sorry

theorem length_of_common_chord :
  2 * sqrt (radius₁^2 - (distance_point_line 2 (-2*sqrt 3) (-3) 1 0)^2) = (3 * sqrt 7) / 2 :=
sorry

end equation_of_common_chord_length_of_common_chord_l127_127553


namespace part1_proof_part2_proof_l127_127929

noncomputable def proof_problem_part1 
  (b a : ℝ) (hb : b > a) (ha : a > 0) 
  (f : ℝ → ℝ) (hf : ContinuousOn f (Set.Ici 0)) : 
  Prop := 
  Filter.Tendsto 
    (fun ε : ℝ => ∫ x in a * ε .. b * ε, (f x) / x) 
    (Filter.Ici 0) 
    (Filter.atBot (f 0 * Real.log (b / a)))

noncomputable def proof_problem_part2
  (b a : ℝ) (hb : b > a) (ha : a > 0) 
  (f : ℝ → ℝ) (hf : ContinuousOn f (Set.Ici 0)) (int_conv : IntegrableOn (fun x => f x / x) (Set.Ioc 1 ∞)) : 
  Prop := 
  ∫ x in Set.Ici 0, (f (b * x) - f (a * x)) / x = f 0 * Real.log (a / b)

#check @proof_problem_part1
#check @proof_problem_part2

-- proof placeholders
theorem part1_proof (b a : ℝ) (hb : b > a) (ha : a > 0) (f : ℝ → ℝ) (hf : ContinuousOn f (Set.Ici 0)) : 
  proof_problem_part1 b a hb ha f hf :=
by
  sorry

theorem part2_proof 
  (b a : ℝ) (hb : b > a) (ha : a > 0) 
  (f : ℝ → ℝ) (hf : ContinuousOn f (Set.Ici 0)) 
  (int_conv : IntegrableOn (fun x => f x / x) (Set.Ioc 1 ∞)) : 
  proof_problem_part2 b a hb ha f hf int_conv :=
by
  sorry

end part1_proof_part2_proof_l127_127929


namespace shifted_parabola_l127_127004

theorem shifted_parabola (x : ℝ) : 
  let original_parabola := fun x : ℝ => x^2 in
  let shifted_parabola := fun x : ℝ => (x + 2)^2 in
  (∀ x, original_parabola (x - 2) = shifted_parabola x) :=
by 
  sorry

end shifted_parabola_l127_127004


namespace evaluate_expression_l127_127711

theorem evaluate_expression : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 := by
  sorry

end evaluate_expression_l127_127711


namespace integer_product_l127_127146

open Real

theorem integer_product (P Q R S : ℕ) (h1 : P + Q + R + S = 48)
    (h2 : P + 3 = Q - 3) (h3 : P + 3 = R * 3) (h4 : P + 3 = S / 3) :
    P * Q * R * S = 5832 :=
sorry

end integer_product_l127_127146


namespace equal_playing_time_for_each_player_l127_127788

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l127_127788


namespace points_on_circle_or_line_l127_127554

open Set

variables {S₁ S₂ S₃ S₄ : Sphere ℝ} -- Spheres S₁, S₂, S₃, S₄
variables {A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ} -- Points A₁, A₂, A₃, A₄

--conditions of spheres touching at points
def sphere_touching_at_point (S₁ S₂ : Sphere ℝ) (A : ℝ × ℝ × ℝ) : Prop :=
  ∃ (α : ℝ), A ∈ S₁ ∧ A ∈ S₂ ∧ A = α

-- Given: Spheres S1 and S2 touch at point A1, S2 and S3 touch at point A2, etc.
axiom touch_S₁_S₂ : sphere_touching_at_point S₁ S₂ A₁
axiom touch_S₂_S₃ : sphere_touching_at_point S₂ S₃ A₂
axiom touch_S₃_S₄ : sphere_touching_at_point S₃ S₄ A₃
axiom touch_S₄_S₁ : sphere_touching_at_point S₄ S₁ A₄

theorem points_on_circle_or_line (S₁ S₂ S₃ S₄ : Sphere ℝ) 
  (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ)
  (h₁ : sphere_touching_at_point S₁ S₂ A₁)
  (h₂ : sphere_touching_at_point S₂ S₃ A₂)
  (h₃ : sphere_touching_at_point S₃ S₄ A₃)
  (h₄ : sphere_touching_at_point S₄ S₁ A₄) :
  ∃ C : Circle ℝ, A₁ ∈ C ∧ A₂ ∈ C ∧ A₃ ∈ C ∧ A₄ ∈ C ∨
  ∃ L : Line ℝ, A₁ ∈ L ∧ A₂ ∈ L ∧ A₃ ∈ L ∧ A₄ ∈ L := 
sorry

end points_on_circle_or_line_l127_127554


namespace area_ratio_of_octagons_l127_127491

theorem area_ratio_of_octagons (r : ℝ) :
  let cos_term := Real.sqrt (2 + Real.sqrt 2) / 2 in
  (cos_term ^ 2 = (2 + Real.sqrt 2) / 4) :=
by
  -- We assert cos(22.5 degrees) as cos_term
  have cos_22_5 : cos (22.5 * Real.pi / 180) = Real.sqrt (2 + Real.sqrt 2) / 2,
  -- Introduce the result and prove the theorem
  sorry

end area_ratio_of_octagons_l127_127491


namespace tangent_line_equation_l127_127984

/-- Definition of the function f(x) --/
def f (x : ℝ) : ℝ := x^3 + x - 16

/-- The point of interest (2, -6) --/
def point_of_interest : ℝ × ℝ := (2, -6)

/-- The derivative of f(x) --/
def f_prime (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The equation of the tangent line at the point (2, -6) --/
theorem tangent_line_equation :
  let (x0, y0) := point_of_interest in
  (y0 = f x0) ∧ (f_prime x0 = 13) →
  ∃ b : ℝ, y = 13 * x + b ∧ b = -32 :=
by
  sorry

end tangent_line_equation_l127_127984


namespace number_of_non_neg_real_values_of_x_l127_127920

noncomputable def numNonNegRealValues (x : ℝ) : ℕ :=
  ∑ k in Finset.range 14, if (169 - k^2 : ℝ) ^ 3 ≥ 0 then 1 else 0

theorem number_of_non_neg_real_values_of_x :
  numNonNegRealValues 169 = 14 :=
sorry

end number_of_non_neg_real_values_of_x_l127_127920


namespace translate_complex_number_l127_127106

theorem translate_complex_number :
  ∀ (z w1 w2 : ℂ), z = 1 + 3 * complex.i → w1 = 4 + 7 * complex.i → w2 = 2 - complex.i →
    (∃ t : ℂ, w1 = z + t) → ∀ t, w1 = z + t → w2 + t = 5 + 3 * complex.i :=
by
  intro z w1 w2 hz hw1 hw2 ht t h
  have h_translation : t = 3 + 4 * complex.i := by sorry
  have h_result : w2 + t = 5 + 3 * complex.i := by sorry
  exact h_result

#check translate_complex_number

end translate_complex_number_l127_127106


namespace ratio_of_ages_l127_127710

variable (F S : ℕ)

-- Condition 1: The product of father's age and son's age is 756
def cond1 := F * S = 756

-- Condition 2: The ratio of their ages after 6 years will be 2
def cond2 := (F + 6) / (S + 6) = 2

-- Theorem statement: The current ratio of the father's age to the son's age is 7:3
theorem ratio_of_ages (h1 : cond1 F S) (h2 : cond2 F S) : F / S = 7 / 3 :=
sorry

end ratio_of_ages_l127_127710


namespace number_of_arrangements_l127_127076

-- Definition of the students and communities
constant students : Finset ℕ := {0, 1, 2, 3} -- Representing {A, B, C, D}
constant communities : Finset ℕ := {0, 1, 2} -- Representing {A, B, C}

-- Conditions 
-- B ≠ A translation: student 1 cannot go to community 0
-- A and B cannot be in the same community
def valid_arrangement (assignment : ℕ → ℕ) : Prop :=
  (∀ s ∈ students, assignment s ∈ communities) ∧        -- Each student must participate in one community
  (∀ c ∈ communities, ∃ s ∈ students, assignment s = c) ∧ -- Each community must have at least one student
  assignment 1 ≠ 0 ∧                       -- Student B cannot go to community A
  assignment 0 ≠ assignment 1              -- Students A and B cannot go to the same community

-- The main theorem to be proven
theorem number_of_arrangements : 
  ∃ (assignments : Finset (ℕ → ℕ)), 
  (∀ assignment ∈ assignments, valid_arrangement assignment) 
  ∧ assignments.card = 20 :=
sorry

end number_of_arrangements_l127_127076


namespace num_integers_between_cubes_l127_127583

theorem num_integers_between_cubes :
  let n1 := Real.floor ((9.8:ℝ) ^ 3) + 1 in
  let n2 := Real.floor ((10.1:ℝ) ^ 3) in
  n2 - n1 + 1 = 89 :=
by
  let n1 := Real.floor ((9.8:ℝ) ^ 3) + 1
  let n2 := Real.floor ((10.1:ℝ) ^ 3)
  have hn1 : n1 = 942 := sorry
  have hn2 : n2 = 1030 := sorry
  calc
    n2 - n1 + 1 = 1030 - 942 + 1 : by rw [hn1, hn2]
          ... = 89 : by norm_num

end num_integers_between_cubes_l127_127583


namespace closest_integer_to_1000E_l127_127124

theorem closest_integer_to_1000E : 
  let n := 2020
  let lambda := 1
  let poisson_pmf := λ k, Mathlib.exp(-lambda) * lambda^k / Mathlib.factorial k
  let E := ∑ i in finset.range (n + 1), 
            (1 - ∑ k in finset.range i, poisson_pmf k)
  in (1000 * E).nat_ceil = 1000 := 
by 
  sorry

end closest_integer_to_1000E_l127_127124


namespace equal_playing_time_l127_127814

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l127_127814


namespace planes_parallel_in_space_l127_127759

noncomputable section

-- Defining basic geometric entities
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Plane :=
(normal : Point3D) (offset : ℝ)

-- Definition: Two planes are parallel if their normal vectors are parallel
def planes_parallel (π₁ π₂ : Plane) : Prop :=
π₁.normal.x * π₂.normal.y - π₁.normal.y * π₂.normal.x = 0 ∧
π₁.normal.x * π₂.normal.z - π₁.normal.z * π₂.normal.x = 0 ∧
π₁.normal.y * π₂.normal.z - π₁.normal.z * π₂.normal.y = 0

-- Axiom: Two lines parallel to the same line in a plane are parallel
axiom lines_parallel_in_plane (l1 l2 l3 : Plane) : 
  (planes_parallel l1 l3 ∧ planes_parallel l2 l3) → planes_parallel l1 l2

-- Theorem to be proved
theorem planes_parallel_in_space (π₁ π₂ π₃ : Plane) :
  (planes_parallel π₁ π₃ ∧ planes_parallel π₂ π₃) → planes_parallel π₁ π₂ :=
begin
  sorry
end

end planes_parallel_in_space_l127_127759


namespace least_value_f_1998_l127_127142

theorem least_value_f_1998 :
  ∀ (f : ℕ → ℕ),
  (∀ m n : ℕ, f(n^2 * f(m)) = m * (f(n))^2) →
  ∃ c : ℕ, f(1998) = c ∧ c = 120 :=
by
  intros f hf
  sorry

end least_value_f_1998_l127_127142


namespace equal_play_time_l127_127794

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l127_127794


namespace candy_weight_reduction_l127_127444

theorem candy_weight_reduction:
  ∀ (W P : ℝ), (33.333333333333314 / 100) * (P / W) = (P / (W - (1/4) * W)) →
  (1 - (W - (1/4) * W) / W) * 100 = 25 :=
by
  intros W P h
  sorry

end candy_weight_reduction_l127_127444


namespace line_AM_bisects_perimeter_of_triangle_l127_127693

theorem line_AM_bisects_perimeter_of_triangle
  (A B C M P Q : Point)
  (circle : Circle)
  (ABC : Triangle A B C)
  (tangent_AB_PM : TangentAcross (P.circle_point circle) (B.circle_point circle))
  (tangent_AC_QM : TangentAcross (Q.circle_point circle) (C.circle_point circle))
  (tangent_BC_M : TangentAcross (M.circle_point circle) (BC.circle_segment circle))
  (AP_AQ : tangent_length (P.circle_point circle) = tangent_length (Q.circle_point circle))
  (BP_BM : tangent_length (P.circle_point circle) = tangent_length (M.circle_point circle))
  (CQ_CM : tangent_length (Q.circle_point circle) = tangent_length (M.circle_point circle)) :
  length_segment (AB.line_segment) + length_segment (BM.line_segment) = 
  length_segment (AC.line_segment) + length_segment (CM.line_segment) := 
begin
  sorry
end

end line_AM_bisects_perimeter_of_triangle_l127_127693


namespace password_combinations_l127_127315

theorem password_combinations (n r k : ℕ) (hn : n = 5) (hk_fact : k.factorial = 6) (hr : r = 20) : 
  ∃ (password : list char), 
    let combinations := (n.factorial / k.factorial) in 
    combinations = r := 
begin
  sorry
end

end password_combinations_l127_127315


namespace parametric_line_l127_127368

theorem parametric_line (s m : ℤ) :
  (∀ t : ℤ, ∃ x y : ℤ, 
    y = 5 * x - 7 ∧
    x = s + 6 * t ∧ y = 3 + m * t ) → 
  (s = 2 ∧ m = 30) :=
by
  sorry

end parametric_line_l127_127368


namespace find_a6_plus_a7_plus_a8_l127_127292

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l127_127292


namespace equal_playing_time_for_each_player_l127_127791

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l127_127791


namespace side_length_of_rhombus_l127_127358

-- Define the conditions
variables {A B C D O : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]
variables {AC BD : ℝ}
variables (S : ℝ := 24) (AC : ℝ := 6)

-- Given conditions
def is_intersection_point (O : Type) (A C B D : Type) : Prop :=
true -- Placeholder definition, actual geometric relationship between points A, C, B, D, and O

def is_diagonal (A C B D : Type) (AC BD : ℝ) : Prop :=
true -- Placeholder definition for diagonals AC, BD of a rhombus

def area_of_rhombus (S AC BD : ℝ) : Prop :=
S = 1 / 2 * AC * BD

-- Proposition: Proving the side length
theorem side_length_of_rhombus (h1 : is_diagonal A C B D AC BD)
                               (h2 : is_intersection_point O A C B D)
                               (h3 : area_of_rhombus S AC BD) 
                               : ∃ x : ℝ, x = 5 :=
by
  sorry

end side_length_of_rhombus_l127_127358


namespace division_problem_l127_127408

theorem division_problem (A : ℕ) (h : 23 = (A * 3) + 2) : A = 7 :=
sorry

end division_problem_l127_127408


namespace equal_play_time_l127_127825

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l127_127825


namespace alice_paid_percentage_l127_127370

theorem alice_paid_percentage {P : ℝ} (hP : P > 0)
  (hMP : ∀ P, MP = 0.60 * P)
  (hPrice_Alice_Paid : ∀ MP, Price_Alice_Paid = 0.40 * MP) :
  (Price_Alice_Paid / P) * 100 = 24 := by
  sorry

end alice_paid_percentage_l127_127370


namespace area_of_triangle_PQR_l127_127740

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def area_of_triangle (P Q R : Point) : ℝ :=
  1 / 2 * (Q.x - P.x) * abs(R.y - P.y)

theorem area_of_triangle_PQR :
  area_of_triangle ⟨-4, 2⟩ ⟨8, 2⟩ ⟨6, -4⟩ = 36 :=
by
  sorry

end area_of_triangle_PQR_l127_127740


namespace trig_identity_proof_l127_127418

theorem trig_identity_proof (α : ℝ) :
  sin^2 (π / 4 + α) - sin^2 (π / 6 - α) - sin (π / 12) * cos (π / 12 + 2 * α) = sin (2 * α) :=
by sorry

end trig_identity_proof_l127_127418


namespace elena_deductions_in_cents_l127_127894

-- Definitions based on the conditions
def cents_per_dollar : ℕ := 100
def hourly_wage_in_dollars : ℕ := 25
def hourly_wage_in_cents : ℕ := hourly_wage_in_dollars * cents_per_dollar
def tax_rate : ℚ := 0.02
def health_benefit_rate : ℚ := 0.015

-- The problem to prove
theorem elena_deductions_in_cents:
  (tax_rate * hourly_wage_in_cents) + (health_benefit_rate * hourly_wage_in_cents) = 87.5 := 
by
  sorry

end elena_deductions_in_cents_l127_127894


namespace find_a6_plus_a7_plus_a8_l127_127290

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l127_127290


namespace pieces_of_clothing_l127_127154

theorem pieces_of_clothing (shirts trousers total : ℕ) (hshirts : shirts = 589) (htrousers : trousers = 345) :
  shirts + trousers = total → total = 934 :=
by
  intros htotal
  rw [hshirts, htrousers] at htotal
  exact htotal

end pieces_of_clothing_l127_127154


namespace cube_root_3670_l127_127589
def cube_root (x : ℝ) : ℝ := x^(1 / 3)

variable (h1 : cube_root 0.3670 = 0.7160)
variable (h2 : cube_root 3.670 = 1.542)

theorem cube_root_3670 : cube_root 3670 = 15.42 :=
by
  -- Proof would go here
  sorry

end cube_root_3670_l127_127589


namespace length_of_bridge_l127_127365

variable (train_length : ℕ)
variable (train_speed_kmh : ℕ)
variable (time_to_cross : ℕ)

theorem length_of_bridge:
  train_length = 140 →
  train_speed_kmh = 45 →
  time_to_cross = 30 →
  let train_speed := train_speed_kmh * 1000 / 3600 in
  let total_distance := train_speed * time_to_cross in
  let bridge_length := total_distance - train_length in
  bridge_length = 235 := by
  sorry

end length_of_bridge_l127_127365


namespace percentage_of_people_born_in_october_l127_127364

theorem percentage_of_people_born_in_october (total : ℕ) (october : ℕ) (h1 : total = 120) (h2 : october = 18) :
  (october.toRat / total.toRat) * 100 = 15 := by
  have h3 : (18.toRat / 120.toRat) * 100 = (18 / 120 : ℚ) * 100 := by sorry
  have h4 : (18 / 120 : ℚ) = 0.15 := by sorry
  have h5 : 0.15 * 100 = 15 := by sorry
  rw [h3, h4, h5]
  zh4fc
  exact h5

end percentage_of_people_born_in_october_l127_127364


namespace last_two_nonzero_digits_75_factorial_l127_127504

theorem last_two_nonzero_digits_75_factorial : 
  ∃ (d : ℕ), d = 76 ∧ last_two_nonzero_digits (factorial 75) = d := by
  sorry

end last_two_nonzero_digits_75_factorial_l127_127504


namespace square_area_l127_127011

def dist (A B : (ℝ × ℝ)) : ℝ :=
  (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))

def area_of_square (A B : (ℝ × ℝ)) : ℝ :=
  let side := dist A B
  side * side

theorem square_area (A B : (ℝ × ℝ)) (h : A = (0, 3) ∧ B = (4, 0)) : 
  area_of_square A B = 25 :=
by
  sorry

end square_area_l127_127011


namespace initial_action_figures_l127_127635

theorem initial_action_figures (x : ℕ) (h : x + 2 - 7 = 10) : x = 15 :=
by
  sorry

end initial_action_figures_l127_127635


namespace equal_playing_time_for_each_player_l127_127792

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l127_127792


namespace f_eval_at_8_l127_127540

-- Define the function f based on the given condition
def f : ℝ → ℝ := λ x, log 2 (x ^ (1/6))

-- The goal is to prove that f(8) = 1/2
theorem f_eval_at_8 : f 8 = 1 / 2 :=
by 
  sorry

end f_eval_at_8_l127_127540


namespace complex_quadrant_l127_127357

theorem complex_quadrant (z : ℂ) (hz : z = (2 - I) / (1 + I)) : 
  0 < z.re ∧ z.im < 0 :=
by 
  -- Assignments from conditions
  assume z hv,
  sorry

end complex_quadrant_l127_127357


namespace probability_of_sum_gt_five_is_one_third_l127_127721

noncomputable def probability_sum_gt_five : ℚ :=
let balls := {1, 2, 3, 4}
let possible_draws := ({x // x ∈ balls}.subtype) × ({y // y ∈ balls}.subtype)
let valid_draws := filter (λ (p : ({x // x ∈ balls}.subtype) × ({y // y ∈ balls}.subtype)), p.1.val ≠ p.2.val ∧ p.1.val + p.2.val > 5) possible_draws
(valid_draws.length : ℚ) / (possible_draws.length : ℚ)

theorem probability_of_sum_gt_five_is_one_third : 
  probability_sum_gt_five = 1 / 3 := 
sorry

end probability_of_sum_gt_five_is_one_third_l127_127721


namespace tom_gave_jessica_some_seashells_l127_127031

theorem tom_gave_jessica_some_seashells
  (original_seashells : ℕ := 5)
  (current_seashells : ℕ := 3) :
  original_seashells - current_seashells = 2 :=
by
  sorry

end tom_gave_jessica_some_seashells_l127_127031


namespace height_of_spheres_l127_127737

theorem height_of_spheres (R r : ℝ) (h : ℝ) :
  0 < r ∧ r < R → h = R - Real.sqrt ((3 * R^2 - 6 * R * r - r^2) / 3) :=
by
  intros h0
  sorry

end height_of_spheres_l127_127737


namespace range_of_x_sqrt_x_minus_one_div_three_l127_127606

theorem range_of_x_sqrt_x_minus_one_div_three (x : ℝ) : (∃ c : ℝ, c = sqrt (x - 1) / 3) ↔ x ≥ 1 := by
sorry

end range_of_x_sqrt_x_minus_one_div_three_l127_127606


namespace ratio_of_ages_l127_127672

theorem ratio_of_ages (Sachin_age Rahul_age : ℕ) (h1 : Sachin_age = 5) (h2 : Sachin_age + 7 = Rahul_age) : Sachin_age : Rahul_age = 5 : 12 :=
sorry

end ratio_of_ages_l127_127672


namespace sum_of_roots_of_polynomial_product_l127_127526

-- Let p1, p2, and p3 be the polynomials defined as follows.
def p1 : Polynomial ℚ := 3 * (X ^ 4) + 2 * (X ^ 3) - 9 * X + 15
def p2 : Polynomial ℚ := 4 * (X ^ 3) - 16 * (X ^ 2) + X - 5
def p3 : Polynomial ℚ := 6 * (X ^ 2) - 24 * X + 35

-- To find the sum of the roots of the product of p1, p2, and p3.
theorem sum_of_roots_of_polynomial_product :
  (roots p1).sum + (roots p2).sum + (roots p3).sum = (10 / 3 : ℚ) :=
sorry

end sum_of_roots_of_polynomial_product_l127_127526


namespace decrease_in_revenue_l127_127383

theorem decrease_in_revenue (T C : ℝ) :
  let original_revenue := T * C
  let new_tax := 0.8 * T
  let new_consumption := 1.2 * C
  let new_revenue := new_tax * new_consumption
  original_revenue ≠ 0 → 
  (original_revenue - new_revenue) / original_revenue * 100 = 4 := 
by
  intros original_revenue new_tax new_consumption new_revenue h
  let decrease := (original_revenue - new_revenue) / original_revenue * 100
  show decrease = 4 from sorry

end decrease_in_revenue_l127_127383


namespace parabola_shift_left_l127_127002

theorem parabola_shift_left (x : ℝ) : 
  (let y := x^2 in
  y = x^2 → 
  y = ((x + 2)^2)) :=
sorry

end parabola_shift_left_l127_127002


namespace angle_MBC_equals_30_l127_127632

noncomputable theory
open_locale real

variables {A B C M : Type*} [triangle A B C]
variables (H : altitude A B C) (K : median B A C M)

-- Given the altitude AH and median BM
axiom height_equal_median : H = K

-- Stating the theorem to be proven
theorem angle_MBC_equals_30 (H : altitude A B C) (K : median B A C M) 
  (height_equal_median : H = K) : ∠ MBC = 30 :=
by
  sorry

end angle_MBC_equals_30_l127_127632


namespace shoes_cost_l127_127660

theorem shoes_cost (S : ℝ) : 
  let suit := 430
  let discount := 100
  let total_paid := 520
  suit + S - discount = total_paid -> 
  S = 190 :=
by 
  intro h
  sorry

end shoes_cost_l127_127660


namespace lines_through_point_not_intersecting_graph_l127_127216

theorem lines_through_point_not_intersecting_graph :
  let f : ℝ → ℝ := λ x, if x = 0 then 0 else x + 1
  let p : ℝ × ℝ := (1, 0)
  ∃! (l1 l2 : ℝ → ℝ), (∀ x, l1 x ≠ f x) ∧ (∀ x, l2 x ≠ f x) ∧
    (l1 p.1 = p.2) ∧ (l2 p.1 = p.2) ∧ l1 ≠ l2 := sorry

end lines_through_point_not_intersecting_graph_l127_127216


namespace problem_1_problem_2_l127_127137

def op (x y : ℝ) : ℝ := 3 * x - y

theorem problem_1 (x : ℝ) : op x (op 2 3) = 1 ↔ x = 4 / 3 := by
  -- definitions from conditions
  let def_op_2_3 := op 2 3
  let eq1 := op x def_op_2_3
  -- problem in lean representation
  sorry

theorem problem_2 (x : ℝ) : op (x ^ 2) 2 = 10 ↔ x = 2 ∨ x = -2 := by
  -- problem in lean representation
  sorry

end problem_1_problem_2_l127_127137


namespace num_invalid_d_l127_127009

noncomputable def square_and_triangle_problem (d : ℕ) : Prop :=
  ∃ a b : ℕ, 3 * a - 4 * b = 1989 ∧ a - b = d ∧ b > 0

theorem num_invalid_d : ∀ (d : ℕ), (d ≤ 663) → ¬ square_and_triangle_problem d :=
by {
  sorry
}

end num_invalid_d_l127_127009


namespace total_seats_correct_l127_127061

-- Define the arithmetic sequence with initial term a1 and common difference d
def seats_in_row (n : ℕ) : ℕ := 14 + 2 * (n - 1)

-- Define the total number of seats in the theater when the number of rows is n
def total_seats (n : ℕ) : ℕ := n * (14 + seats_in_row n) / 2

theorem total_seats_correct :
  (∃ n, seats_in_row n = 56) →
  total_seats (classical.some (exists.intro 22 (by simp))) = 770 :=
by
  intro h
  have n_eq : classical.some h = 22 := by
    simp [classical.some_spec h]
  rw [n_eq, total_seats, seats_in_row]
  simp
  sorry

end total_seats_correct_l127_127061


namespace no_path_from_1_to_9_l127_127830

def is_connected (a b : ℕ) : Prop :=
  (10 * a + b) % 3 = 0

def can_travel (from to : ℕ) (reachable : ℕ → set ℕ) : Prop :=
  reachable from to

def all_connections : ℕ → set ℕ
| 1 := {2, 5, 8}
| 2 := {1, 4, 7}
| 3 := {0, 3, 6, 9}
| 4 := {2, 5, 8}
| 5 := {2, 5, 8}
| 6 := {3, 6, 9}
| 7 := {1, 4, 7}
| 8 := {1, 4, 7}
| 9 := {0, 3, 6, 9}
| _ := ∅

theorem no_path_from_1_to_9 : ¬ can_travel 1 9 all_connections :=
  sorry

end no_path_from_1_to_9_l127_127830


namespace circle_equation_l127_127184

theorem circle_equation 
  (circle_eq : ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = (x - 3)^2 + (y - 2)^2) 
  (tangent_to_line : ∀ (x y : ℝ), (2*x - y + 5) = 0 → 
    (x = -2 ∧ y = 1))
  (passes_through_N : ∀ (x y : ℝ), (x = 3 ∧ y = 2)) :
  ∀ (x y : ℝ), x^2 + y^2 - 9*x + (9/2)*y - (55/2) = 0 := 
sorry

end circle_equation_l127_127184


namespace students_team_three_classes_l127_127476

theorem students_team_three_classes
  (students : Fin 30 → Fin 3 → Fin 3)
  (h : ∀ (n : Fin 3) (team : Fin 3), ∃ (student : Fin 10), team = student) :
  ∃ (x y : Fin 30), x ≠ y ∧ (∀ (class : Fin 3), students x class = students y class) :=
begin
  sorry
end

end students_team_three_classes_l127_127476


namespace team_members_run_distance_l127_127719

-- Define the given conditions
def total_distance : ℕ := 150
def members : ℕ := 5

-- Prove the question == answer given the conditions
theorem team_members_run_distance :
  total_distance / members = 30 :=
by
  sorry

end team_members_run_distance_l127_127719


namespace parabola_shift_left_l127_127000

theorem parabola_shift_left (x : ℝ) : 
  (let y := x^2 in
  y = x^2 → 
  y = ((x + 2)^2)) :=
sorry

end parabola_shift_left_l127_127000


namespace max_people_k_2_max_people_k_776_l127_127326

-- Define the conditions of the problem
def group_conditions (n k: ℕ) :=
  (∀ (s : finset ℕ), s.card = 2 * k - 1 → ∃ (t : finset ℕ), t ⊆ s ∧ t.card = k ∧ (∀ {a b}, a ∈ t → b ∈ t → a ≠ b → friend a b)) ∧
  (∀ x : ℕ, member x → finset.card (friend_set x) ≤ 2011)

-- Define the friend relationship (undefined, will be assumed)
def friend : ℕ → ℕ → Prop := sorry

-- Define member relationship (undefined, will be assumed)
def member : ℕ → Prop := sorry

-- Define the set of friends for a member (undefined, will be assumed)
def friend_set : ℕ → finset ℕ := sorry

-- The proof statement for k = 2
theorem max_people_k_2 (n : ℕ) (h : group_conditions n 2) : n ≤ 4024 :=
sorry

-- The proof statement for k = 776
theorem max_people_k_776 (n : ℕ) (h : group_conditions n 776) : n ≤ 4024 :=
sorry

end max_people_k_2_max_people_k_776_l127_127326


namespace find_number_l127_127069

theorem find_number 
  (x : ℝ)
  (h : (258 / 100 * x) / 6 = 543.95) :
  x = 1265 :=
sorry

end find_number_l127_127069


namespace find_a1_l127_127198

noncomputable def geosequence (a: ℕ → ℝ) (q: ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

noncomputable def Sn (a: ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => Sn n a + a (n + 1)

theorem find_a1
  (a : ℕ → ℝ)
  (q : ℝ)
  (hseq : geosequence a q)
  (h_s4_s6 : a 4 + a 5 + a 6 = 2 * Sn a 3)
  (h_a7 : a 7 = 12) :
  a 1 = 3 :=
  sorry

end find_a1_l127_127198


namespace num_functions_eq_num_injections_eq_num_surjections_eq_num_bijections_eq_l127_127651

open BigOperators

noncomputable section

variables {A B : Type*} [Fintype A] [Fintype B] (N : ℕ) (M : ℕ)
  [DecidableEq A] [DecidableEq B]

def num_functions : ℕ := M ^ N

def num_injections : ℕ := ∏ i in Finset.range N, (M - i)

def num_surjections (S_N_M : ℕ) : ℕ := M.factorial * S_N_M

def num_bijections : ℕ := N.factorial

theorem num_functions_eq (hA : Fintype.card A = N) (hB : Fintype.card B = M) : 
  num_functions N M = M^N :=
by sorry

theorem num_injections_eq (hA : Fintype.card A = N) (hB : Fintype.card B = M) (h : N ≤ M) : 
  num_injections N M = ∏ i in Finset.range N, (M - i) :=
by sorry

theorem num_surjections_eq (hA : Fintype.card A = N) (hB : Fintype.card B = M) (h : N ≥ M) (S_N_M : ℕ) :
  num_surjections N M S_N_M = M.factorial * S_N_M :=
by sorry

theorem num_bijections_eq (hA : Fintype.card A = N) (hB : Fintype.card B = M) (h : N = M) :
  num_bijections N = N.factorial :=
by sorry

end num_functions_eq_num_injections_eq_num_surjections_eq_num_bijections_eq_l127_127651


namespace inequality_count_l127_127321

variable (x y a b : ℝ)
variable (hx : x < a)
variable (hy : y > b)
variable (hnx : x ≠ 0)
variable (hny : y ≠ 0)
variable (hna : a ≠ 0)
variable (hnb : b ≠ 0)

theorem inequality_count : 
  (cond1 : x + y < a + b) ∨
  (cond2 : x - y > a - b) ∨
  (cond3 : xy > ab) ∨
  (cond4 : x / y < a / b) → 
  1 = 1 :=
by
  sorry

end inequality_count_l127_127321


namespace bisectors_intersect_on_AD_l127_127261

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assume the points A, B, C, D in a Metric Space such that AB + CD = BC, and angles ∠DAB and ∠CDA are right.
variables (AB CD BC AD : ℝ) 
  (h_right_angle_A : ∠ D A B = π / 2)
  (h_right_angle_D : ∠ A D C = π / 2)
  (h_sum_sides : AB + CD = BC)
  (h_trapezoid : Trapezoid A B C D)

-- Define the internal angle bisectors originating from B and C.
noncomputable def bisector_B : Line := sorry
noncomputable def bisector_C : Line := sorry

-- The goal to prove that these bisectors intersect on the side AD.
theorem bisectors_intersect_on_AD : (bisector_B ∩ bisector_C) ∈ AD := sorry

end bisectors_intersect_on_AD_l127_127261


namespace equal_playing_time_l127_127811

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l127_127811


namespace triangle_area_l127_127400

def line1 (x : ℝ) : ℝ := 2 * x - 4
def line2 (x : ℝ) : ℝ := -3 * x + 16

theorem triangle_area : 
  let intersection := (4, 4),
      vertex1 := (0, -4),
      vertex2 := (0, 16)
  in
  1/2 * (vertex2.2 - vertex1.2) * intersection.1 = 40 := 
by
  -- Using the equations of the lines to find intersection and performing required area calculations
  sorry

end triangle_area_l127_127400


namespace curveC_equation_a_sqrt3_unique_common_point_line_slope_perpendicular_intersection_exists_l127_127203

def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)
def C (a : ℝ) : set (ℝ × ℝ) := {P | abs (dist P F1 - dist P F2) = 2 * a}

def curveC1 := {P : ℝ × ℝ | (P.1^2) / 3 - P.2^2 = 1}
def curveC3 := {P : ℝ × ℝ | P.1^2 - (P.2^2) / 3 = 1}

theorem curveC_equation_a_sqrt3 : C (sqrt 3) = curveC1 := sorry

theorem unique_common_point_line_slope : 
  ∃ l : ℝ → ℝ, (l (0) = 1) 
            ∧ (∀ P ∈ curveC1, (P.2 = l P.1) → ∃! P ∈ curveC1, P.2 = l P.1)
            ∧ (l = λ x, (sqrt 3) / 3 * x + 1 ∨ l = λ x, -((sqrt 3) / 3 * x + 1)
            ∨ l = λ x, 2 * x + 1 ∨ l = λ x, -2 * x + 1) := sorry

theorem perpendicular_intersection_exists : 
  ∃ k : ℝ, C 1 = curveC3 ∧ 
           (y = λ x, k * x + 2) ∧ 
           (∃ A B : ℝ × ℝ, A ∈ curveC3 ∧ B ∈ curveC3 ∧ 
           dist A.origin = dist B.origin ∧ 
           (A.origin ⟷ B.origin) ⊥ k)
           ∧ (k = sqrt 2 ∨ k = -sqrt 2) := sorry

end curveC_equation_a_sqrt3_unique_common_point_line_slope_perpendicular_intersection_exists_l127_127203


namespace each_player_plays_36_minutes_l127_127802

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l127_127802


namespace cos_of_sin_l127_127560

theorem cos_of_sin (α : ℝ) (h : sin (α + π) = 3 / 4) : cos (α + π / 2) = 3 / 4 :=
by 
  sorry

end cos_of_sin_l127_127560


namespace problem_statement_l127_127944

theorem problem_statement 
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x = exp x + a * x)
  (h2 : ∀ x < 0, f x = 1 / exp x - a * x)
  (h3 : ∀ x, f x = f (-x))
  (h4 : ∃ x1 x2 x3 x4, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :
  a ∈ set.Iio (-exp 1) :=
sorry

end problem_statement_l127_127944


namespace find_larger_number_l127_127022

theorem find_larger_number
  (a b : ℕ)
  (h_sum : a + b = 123456)
  (h_division : b % 8 = 0)
  (h_reduce : ∀ n : ℕ, n > 0 → n < b → Int.ofNat a = n → n ∣ 456) :
  b = 120000 :=
begin
  sorry
end

end find_larger_number_l127_127022


namespace algorithm_can_contain_all_structures_l127_127760

def sequential_structure : Prop := sorry
def conditional_structure : Prop := sorry
def loop_structure : Prop := sorry

def algorithm_contains_structure (str : Prop) : Prop := sorry

theorem algorithm_can_contain_all_structures :
  algorithm_contains_structure sequential_structure ∧
  algorithm_contains_structure conditional_structure ∧
  algorithm_contains_structure loop_structure := sorry

end algorithm_can_contain_all_structures_l127_127760


namespace initial_volume_of_solution_l127_127443

variable (V : ℝ)

theorem initial_volume_of_solution :
  (0.05 * V + 5.5 = 0.15 * (V + 10)) → (V = 40) :=
by
  intro h
  sorry

end initial_volume_of_solution_l127_127443


namespace x_squared_y_plus_xy_squared_l127_127596

-- Define the variables and their conditions
variables {x y : ℝ}

-- Define the theorem stating that if xy = 3 and x + y = 5, then x^2y + xy^2 = 15
theorem x_squared_y_plus_xy_squared (h1 : x * y = 3) (h2 : x + y = 5) : x^2 * y + x * y^2 = 15 :=
by {
  sorry
}

end x_squared_y_plus_xy_squared_l127_127596


namespace polynomial_has_three_distinct_integer_roots_l127_127426

theorem polynomial_has_three_distinct_integer_roots :
  ∀ (x : ℤ), (x^5 + 3*x^4 - 4044118*x^3 - 12132362*x^2 - 12132363*x - 2011^2 = 0) →
  ∃ n1 n2 n3 n4 n5 : ℤ, distinct {n1, n2, n3, n4, n5} ∧
  (x - n1)*(x - n2)*(x - n3)*(x - n4)*(x - n5) = x^5 + 3*x^4 - 4044118*x^3 - 12132362*x^2 - 12132363*x - 2011^2 ∧
  (count_distinct {n1, n2, n3, n4, n5} = 3) := 
sorry

end polynomial_has_three_distinct_integer_roots_l127_127426


namespace bounded_color_difference_l127_127458

noncomputable def checkerboard_coloring (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

noncomputable def red_blue_coloring (i j : ℕ) (adjacent_diff : Bool) : Prop :=
  if checkerboard_coloring i j 
  then if adjacent_diff then ((i + j) % 3 = 1) else ((i + j) % 3 = 2)
  else false

theorem bounded_color_difference 
  (l : ℝ → ℝ → Prop) -- Line l is a relation between x and y
  (not_parallel : ∀ x y, ¬ l x (y + 1) ∧ ¬ l (x + 1) y) -- l is not parallel to the sides of the cells
  : ∃ C : ℝ, ∀ segment I, |(sum of red sections of I's length) - (sum of blue sections of I's length)| ≤ C :=
by
  sorry

end bounded_color_difference_l127_127458


namespace find_a6_plus_a7_plus_a8_l127_127291

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l127_127291


namespace train_crosses_platform_l127_127763

theorem train_crosses_platform :
  ∀ (L : ℕ), 
  (300 + L) / (50 / 3) = 48 → 
  L = 500 := 
by
  sorry

end train_crosses_platform_l127_127763


namespace answer_is_correct_l127_127967

-- We define the prime checking function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

-- We define the set of candidates satisfying initial prime condition
def candidates : Set ℕ := {A | is_prime A ∧ A < 100 
                                   ∧ is_prime (A + 10) 
                                   ∧ is_prime (A - 20)
                                   ∧ is_prime (A + 30) 
                                   ∧ is_prime (A + 60) 
                                   ∧ is_prime (A + 70)}

-- The explicit set of valid answers
def valid_answers : Set ℕ := {37, 43, 79}

-- The statement that we need to prove
theorem answer_is_correct : candidates = valid_answers := 
sorry

end answer_is_correct_l127_127967


namespace largest_multiple_of_15_who_negation_greater_than_neg_150_l127_127741

theorem largest_multiple_of_15_who_negation_greater_than_neg_150 : 
  ∃ (x : ℤ), x % 15 = 0 ∧ -x > -150 ∧ ∀ (y : ℤ), y % 15 = 0 ∧ -y > -150 → x ≥ y :=
by
  sorry

end largest_multiple_of_15_who_negation_greater_than_neg_150_l127_127741


namespace dice_probability_is_correct_l127_127150

noncomputable def dice_probability : ℚ :=
  let total_outcomes := (6:ℚ) ^ 7
  let one_pair_no_three_of_a_kind := (6 * 21 * 120 : ℚ)
  let two_pairs_no_three_of_a_kind := (15 * 35 * 6 * 24 : ℚ)
  let three_pairs_one_different := (20 * 7 * 90 * 3 : ℚ)
  let successful_outcomes := one_pair_no_three_of_a_kind + two_pairs_no_three_of_a_kind + three_pairs_one_different
  successful_outcomes / total_outcomes

theorem dice_probability_is_correct : dice_probability = 6426 / 13997 := 
by sorry

end dice_probability_is_correct_l127_127150


namespace total_cost_diff_sum_l127_127090

def bags_dog := 600.5
def price_dog := 24.99
def bags_cat := 327.25
def price_cat := 19.49
def bags_bird := 415.75
def price_bird := 15.99
def bags_fish := 248.5
def price_fish := 13.89
def sales_tax := 0.065

def total_cost_after_tax (bags : ℝ) (price : ℝ) (tax : ℝ) : ℝ :=
  let total_before_tax := bags * price
  total_before_tax + (total_before_tax * tax)

def total_dog := total_cost_after_tax bags_dog price_dog sales_tax
def total_cat := total_cost_after_tax bags_cat price_cat sales_tax
def total_bird := total_cost_after_tax bags_bird price_bird sales_tax
def total_fish := total_cost_after_tax bags_fish price_fish sales_tax

def diff_dog_cat := total_dog - total_cat
def diff_cat_bird := total_cat - total_bird
def diff_bird_fish := total_bird - total_fish

def sum_of_differences := diff_dog_cat + diff_cat_bird + diff_bird_fish

theorem total_cost_diff_sum : sum_of_differences = 12301.9002 :=
by
  sorry

end total_cost_diff_sum_l127_127090


namespace deductible_increase_l127_127420

theorem deductible_increase (current_deductible : ℝ) (increase_fraction : ℝ) (next_year_deductible : ℝ) : 
  current_deductible = 3000 ∧ increase_fraction = 2 / 3 ∧ next_year_deductible = (1 + increase_fraction) * current_deductible →
  next_year_deductible - current_deductible = 2000 :=
by
  intros h
  sorry

end deductible_increase_l127_127420


namespace perpendicular_slope_l127_127523

-- Conditions
def slope_of_given_line : ℚ := 5 / 2

-- The statement
theorem perpendicular_slope (slope_of_given_line : ℚ) : (-1 / slope_of_given_line = -2 / 5) :=
by
  sorry

end perpendicular_slope_l127_127523


namespace sum_2019_l127_127196

noncomputable def a : ℕ → ℝ := sorry
def S (n : ℕ) : ℝ := sorry

axiom prop_1 : (a 2 - 1)^3 + (a 2 - 1) = 2019
axiom prop_2 : (a 2018 - 1)^3 + (a 2018 - 1) = -2019
axiom arithmetic_sequence : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom sum_formula : S 2019 = (2019 * (a 1 + a 2019)) / 2

theorem sum_2019 : S 2019 = 2019 :=
by sorry

end sum_2019_l127_127196


namespace sqrt_2_sqrt_3_minus_3_thm_l127_127676

noncomputable def sqrt_2_sqrt_3_minus_3_eq : Prop :=
  let a : ℚ := 27 / 4
  let b : ℚ := 3 / 4
  (real.sqrt (2 * real.sqrt 3 - 3)) = (real.fourth_root a) - (real.fourth_root b)

theorem sqrt_2_sqrt_3_minus_3_thm : sqrt_2_sqrt_3_minus_3_eq :=
  sorry

end sqrt_2_sqrt_3_minus_3_thm_l127_127676


namespace factorization_count_l127_127152

theorem factorization_count (x : ℤ) : 
  let p := x^15 - x
  count_factors (factorize p) = 4 := 
sorry

end factorization_count_l127_127152


namespace intersection_of_B_and_complement_R_A_l127_127954

open Set

def A : Set ℝ := { x | x^2 ≤ 3 }
def B : Set ℝ := { -2, -1, 1, 2 }
def complement_R_A : Set ℝ := { x | x < -sqrt 3 ∨ x > sqrt 3 }

theorem intersection_of_B_and_complement_R_A :
  B ∩ complement_R_A = { -2, 2 } :=
sorry

end intersection_of_B_and_complement_R_A_l127_127954


namespace cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127867

theorem cos_105_eq_sqrt2_sub_sqrt6_div4 :
  cos (105 * real.pi / 180) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by
  -- Definitions and conditions
  have cos_60 : cos (60 * real.pi / 180) = 1/2 := by sorry
  have cos_45 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  have sin_60 : sin (60 * real.pi / 180) = real.sqrt 3 / 2 := by sorry
  have sin_45 : sin (45 * real.pi / 180) = real.sqrt 2 / 2 := by sorry
  -- Use the angle addition formula: cos (a + b) = cos a * cos b - sin a * sin b
  have add_formula := cos_add (60 * real.pi / 180) (45 * real.pi / 180)
  -- Combine the results using the given known values
  rw [cos_60, cos_45, sin_60, sin_45] at add_formula
  exact add_formula

end cos_105_eq_sqrt2_sub_sqrt6_div4_l127_127867


namespace train_length_l127_127105

theorem train_length (speed_kmh : ℝ) (cross_time_s : ℝ) (bridge_length_m : ℝ) 
  (h_speed : speed_kmh = 45) (h_time : cross_time_s = 30) (h_bridge : bridge_length_m = 285) : 
  let speed_ms := speed_kmh * 1000 / 3600 
  let total_distance := speed_ms * cross_time_s
  length_of_train : total_distance - bridge_length_m = 90 := 
by
  sorry

end train_length_l127_127105


namespace dog_tail_length_l127_127265

theorem dog_tail_length (b h t : ℝ) 
  (h_head : h = b / 6) 
  (h_tail : t = b / 2) 
  (h_total : b + h + t = 30) : 
  t = 9 :=
by
  sorry

end dog_tail_length_l127_127265


namespace derivative_of_f_is_x_cos_x_l127_127695

def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem derivative_of_f_is_x_cos_x : 
  ∀ x : ℝ, deriv f x = x * Real.cos x :=
by
  intros
  sorry

end derivative_of_f_is_x_cos_x_l127_127695


namespace complex_product_conjugate_eq_one_l127_127564

-- Define the complex number and its conjugate
def Z : ℂ := (1 - complex.i * real.sqrt 3) / (complex.i + real.sqrt 3)
def Z_conjugate : ℂ := complex.conj Z

-- Prove that Z * Z_conjugate = 1
theorem complex_product_conjugate_eq_one : Z * Z_conjugate = 1 := sorry

end complex_product_conjugate_eq_one_l127_127564


namespace real_parts_squares_equal_l127_127183

noncomputable def imaginary_unit : ℂ :=
  complex.I

theorem real_parts_squares_equal (a b : ℝ) (h : (a + b * imaginary_unit) ^ 2 = 3 + 4 * imaginary_unit) : 
  a^2 + b^2 = 25 :=
by
  sorry

end real_parts_squares_equal_l127_127183


namespace equal_playing_time_l127_127817

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l127_127817


namespace solve_abs_quadratic_l127_127687

theorem solve_abs_quadratic :
  ∃ x : ℝ, (|x - 3| + x^2 = 10) ∧ 
  (x = (-1 + Real.sqrt 53) / 2 ∨ x = (1 + Real.sqrt 29) / 2 ∨ x = (1 - Real.sqrt 29) / 2) :=
by sorry

end solve_abs_quadratic_l127_127687


namespace unit_normal_vector_l127_127972

theorem unit_normal_vector
  (x y z : ℝ)
  (a : ℝ × ℝ × ℝ)
  (b : ℝ × ℝ × ℝ)
  (a_parallel : a = (1, 0, -1))
  (b_parallel : b = (1, -1, 0)) :
  (sqrt(x^2 + y^2 + z^2) = 1) →
  (x - z = 0) →
  (x - y = 0) →
  (x = y ∧ y = z) ∧ 
  ((x = sqrt(1/3) ∨ x = -sqrt(1/3)) ∧ (x = y ∧ y = z) → 
  (x = sqrt(1/3) ∧ y = sqrt(1/3) ∧ z = sqrt(1/3)) ∨ 
    (x = -sqrt(1/3) ∧ y = -sqrt(1/3) ∧ z = -sqrt(1/3))) := 
by 
  sorry

end unit_normal_vector_l127_127972


namespace square_area_l127_127013

open Real

def dist (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem square_area (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 0) (h2 : y1 = 3) (h3 : x2 = 4) (h4 : y2 = 0) :
  let side_length := dist x1 y1 x2 y2
  in side_length^2 = 25 :=
by
  sorry

end square_area_l127_127013


namespace odd_functions_domain_real_l127_127965

theorem odd_functions_domain_real 
  (α : ℚ) 
  (hα : α ∈ {-1, 1, 2, 3/5, 7/2}) : 
  (∀ x : ℝ, x ∈ ℝ → x^α = -(x^α)) ↔ α ∈ {1, 3/5} := 
by 
  sorry

end odd_functions_domain_real_l127_127965


namespace total_surface_area_of_cylinder_l127_127453

-- Define radius and height of the cylinder
def radius : ℝ := 5
def height : ℝ := 12

-- Theorem stating the total surface area of the cylinder
theorem total_surface_area_of_cylinder : 2 * real.pi * radius * (radius + height) = 170 * real.pi := by
  sorry

end total_surface_area_of_cylinder_l127_127453


namespace rectangle_area_inscribed_circle_l127_127428

theorem rectangle_area_inscribed_circle (PA PB PC PD AB BC : ℝ) 
  (h1 : PA * PB = 2) 
  (h2 : PC * PD = 18) 
  (h3 : PB * PC = 9) 
  (h4 : ∃ x y : ℝ, PA + PB + PC + PD = 18 ∧ x * y = 9 ∧ x + y = 208 √ 17 / 85) 
  : 100 * 208 + 10 * 17 + 85 = 21055 :=
by
  sorry

end rectangle_area_inscribed_circle_l127_127428


namespace cos_105_eq_l127_127858

theorem cos_105_eq : (cos 105) = (real.sqrt 2 - real.sqrt 6) / 4 :=
by 
  have h1: cos (60:ℝ) = 1 / 2 := by sorry
  have h2: cos (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h3: sin (60:ℝ) = real.sqrt 3 / 2 := by sorry
  have h4: sin (45:ℝ) = real.sqrt 2 / 2 := by sorry
  have h5: cos (105:ℝ) = cos (60 + 45) := by sorry
  have h6: cos (60 + 45) = cos 60 * cos 45 - sin 60 * sin 45 := by sorry
  have h7 := calc
    cos 105 = (cos 60) * (cos 45) - (sin 60) * (sin 45) : by sorry
    ... = (1 / 2) * (real.sqrt 2 / 2) - (real.sqrt 3 / 2) * (real.sqrt 2 / 2) : by sorry
    ... = (real.sqrt 2 / 4) - (real.sqrt 6 / 4) : by sorry
    ... = (real.sqrt 2 - real.sqrt 6) / 4 : by sorry
  exact h7

end cos_105_eq_l127_127858


namespace equal_play_time_l127_127796

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l127_127796


namespace ratio_of_democrats_l127_127720

theorem ratio_of_democrats (F M : ℕ) 
  (total_participants : F + M = 840)
  (half_females_democrats : F / 2 = 140)
  (one_quarter_males_democrats : M / 4 = 140) :
  (F / 2 + M / 4) / 840 = 1 / 3 :=
by
sory

end ratio_of_democrats_l127_127720


namespace eccentricity_hyperbola_midpoint_of_MN_area_of_triangle_isosceles_triangle_l127_127989

def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

def focus1 := (-2, 0 : ℝ)
def focus2 := (2, 0 : ℝ)

theorem eccentricity_hyperbola : ∃ e : ℝ, e = 2 :=
by
  -- Proof can be added here
  sorry

theorem midpoint_of_MN (slope_l : ℝ) (M N : ℝ × ℝ) :
  slope_l = 2 → 
  (M ∈ {p : ℝ × ℝ | hyperbola p.1 p.2}) ∧ 
  (N ∈ {p : ℝ × ℝ | hyperbola p.1 p.2}) → 
  let midpoint := (M.1 + N.1) / 2, (M.2 + N.2) / 2 
  in midpoint = (8, 12 : ℝ) :=
by
  -- Proof can be added here
  sorry

theorem area_of_triangle (M : ℝ × ℝ) : 
  ∠ focus1 focus2 M = π / 3 → 
  area focus1 focus2 M = 3 * real.sqrt 3 :=
by
  -- Proof can be added here
  sorry

theorem isosceles_triangle (M N : ℝ × ℝ) : 
  (M ∈ {p : ℝ × ℝ | hyperbola p.1 p.2}) ∧ 
  (N ∈ {p : ℝ × ℝ | hyperbola p.1 p.2}) → 
  ∃ l : ℝ, number_of_isosceles_focus1 M N = 3 :=
by
  -- Proof can be added here
  sorry

end eccentricity_hyperbola_midpoint_of_MN_area_of_triangle_isosceles_triangle_l127_127989


namespace least_number_of_marbles_l127_127100

def divisible_by (n : ℕ) (d : ℕ) : Prop := n % d = 0

theorem least_number_of_marbles 
  (n : ℕ)
  (h3 : divisible_by n 3)
  (h4 : divisible_by n 4)
  (h5 : divisible_by n 5)
  (h7 : divisible_by n 7)
  (h8 : divisible_by n 8) :
  n = 840 :=
sorry

end least_number_of_marbles_l127_127100


namespace problem_a_lt_c_lt_b_l127_127647

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem problem_a_lt_c_lt_b : a < c ∧ c < b := 
by {
  sorry
}

end problem_a_lt_c_lt_b_l127_127647


namespace perpendicular_slope_l127_127522

-- Conditions
def slope_of_given_line : ℚ := 5 / 2

-- The statement
theorem perpendicular_slope (slope_of_given_line : ℚ) : (-1 / slope_of_given_line = -2 / 5) :=
by
  sorry

end perpendicular_slope_l127_127522


namespace certain_number_is_sixteen_l127_127445

theorem certain_number_is_sixteen (x : ℝ) (h : x ^ 5 = 4 ^ 10) : x = 16 :=
by
  sorry

end certain_number_is_sixteen_l127_127445


namespace polynomial_remainder_l127_127521

theorem polynomial_remainder (x : ℂ) : 
  (3 * x ^ 1010 + x ^ 1000) % (x ^ 2 + 1) * (x - 1) = 3 * x ^ 2 + 1 := 
sorry

end polynomial_remainder_l127_127521


namespace no_integer_solutions_for_eq_l127_127331

theorem no_integer_solutions_for_eq {x y : ℤ} : ¬ (∃ x y : ℤ, (x + 7) * (x + 6) = 8 * y + 3) := by
  sorry

end no_integer_solutions_for_eq_l127_127331


namespace negation_of_p_l127_127576

def M : set ℝ := {x | -1 < x ∧ x < 2}

def p : Prop := ∃ x ∈ M, x^2 - x - 2 < 0

theorem negation_of_p : ¬p ↔ ∀ x ∈ M, x^2 - x - 2 ≥ 0 :=
by sorry

end negation_of_p_l127_127576


namespace repeating_decimal_eq_fraction_l127_127133

theorem repeating_decimal_eq_fraction : 
  let x := 0.\overline{56}
  in x = (56 / 99) := by
  sorry

end repeating_decimal_eq_fraction_l127_127133


namespace children_difference_l127_127723

-- Axiom definitions based on conditions
def initial_children : ℕ := 36
def first_stop_got_off : ℕ := 45
def first_stop_got_on : ℕ := 25
def second_stop_got_off : ℕ := 68
def final_children : ℕ := 12

-- Mathematical formulation of the problem and its proof statement
theorem children_difference :
  ∃ (x : ℕ), 
    initial_children - first_stop_got_off + first_stop_got_on - second_stop_got_off + x = final_children ∧ 
    (first_stop_got_off + second_stop_got_off) - (first_stop_got_on + x) = 24 :=
by 
  sorry

end children_difference_l127_127723


namespace gumballs_minimum_count_to_four_same_color_l127_127086

theorem gumballs_minimum_count_to_four_same_color (red white blue green : ℕ) 
    (h_red : red = 12) (h_white : white = 10) (h_blue : blue = 9) (h_green : green = 8) :
    ∃ n, (n = 13) ∧ (∀ picks : Finset (Fin (red + white + blue + green)), 
      picks.card = n → ∃ color_count : Finset ℕ, ∃ c ∈ color_count, c ≥ 4) :=
by
  sorry

end gumballs_minimum_count_to_four_same_color_l127_127086


namespace isosceles_obtuse_triangle_angle_correct_l127_127838

noncomputable def isosceles_obtuse_triangle_smallest_angle (A B C : ℝ) (h1 : A = 1.3 * 90) (h2 : B = C) (h3 : A + B + C = 180) : ℝ :=
  (180 - A) / 2

theorem isosceles_obtuse_triangle_angle_correct 
  (A B C : ℝ)
  (h1 : A = 1.3 * 90)
  (h2 : B = C)
  (h3 : A + B + C = 180) :
  isosceles_obtuse_triangle_smallest_angle A B C h1 h2 h3 = 31.5 :=
sorry

end isosceles_obtuse_triangle_angle_correct_l127_127838


namespace password_combinations_l127_127316

theorem password_combinations (n r k : ℕ) (hn : n = 5) (hk_fact : k.factorial = 6) (hr : r = 20) : 
  ∃ (password : list char), 
    let combinations := (n.factorial / k.factorial) in 
    combinations = r := 
begin
  sorry
end

end password_combinations_l127_127316


namespace multiplication_correct_l127_127484

theorem multiplication_correct :
  375680169467 * 4565579427629 = 1715110767607750737263 :=
  by sorry

end multiplication_correct_l127_127484


namespace find_q_l127_127363

theorem find_q (q : ℝ → ℝ) 
  (h1: ∀ x, q x = 0 → (x = -2 ∨ x = 2))
  (h2: ∃ a b c : ℝ, q = λ x, a * x^2 + b * x + c)
  (h3: q 3 = 10) : 
  q = λ x, 2 * x^2 - 8 :=
by 
-- Proof goes here
sorry

end find_q_l127_127363


namespace smallest_S_value_l127_127374

theorem smallest_S_value :
  ∃ (a1 a2 a3 b1 b2 b3 c1 c2 c3 d : ℕ),
  (∀ x ∈ {a1, a2, a3, b1, b2, b3, c1, c2, c3, d}, x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧
  (∀ x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, ∃! y, y ∈ {a1, a2, a3, b1, b2, b3, c1, c2, c3, d} ∧ y = x) ∧
  minimal_value a1 a2 a3 b1 b2 b3 c1 c2 c3 d = 609 :=
by
  sorry

def minimal_value (a1 a2 a3 b1 b2 b3 c1 c2 c3 d : ℕ) : ℕ :=
  a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 + d

end smallest_S_value_l127_127374


namespace calculation_correct_l127_127750

theorem calculation_correct : 469111 * 9999 = 4690428889 := 
by sorry

end calculation_correct_l127_127750


namespace rectangular_region_area_l127_127095

theorem rectangular_region_area :
  ∀ (s : ℝ), 18 * s * s = (15 * Real.sqrt 2) * (7.5 * Real.sqrt 2) :=
by
  intro s
  have h := 5 ^ 2 = 2 * s ^ 2
  have s := Real.sqrt (25 / 2)
  exact sorry

end rectangular_region_area_l127_127095


namespace external_angle_bisectors_form_rectangle_l127_127669

noncomputable theory
open_locale classical

variables {A B C D P Q R S : Type} 
variables [module ℝ A] [module ℝ B] [module ℝ C] [module ℝ D]
variables [module ℝ P] [module ℝ Q] [module ℝ R] [module ℝ S]

structure parallelogram (A B C D P Q R S : Type) :=
  (AB_eq_CD : ∥B - A∥ = ∥D - C∥)
  (AD_eq_BC : ∥D - A∥ = ∥C - B∥)
  (AB_parallel_CD : line_through A B ∥ line_through C D)
  (AD_parallel_BC : line_through A D ∥ line_through B C)
  (P_bisects_angular : line_through P ∥ line_through A B ∧ line_through P ∥ line_through B C)

theorem external_angle_bisectors_form_rectangle (p : parallelogram A B C D P Q R S) :
  let PQRS := {P, Q, R, S} in
  let diagonal_PR_sum := ∥B - A∥ + ∥D - A∥ in
  is_rectangle PQRS ∧ ∥P - R∥ = diagonal_PR_sum := 
sorry

end external_angle_bisectors_form_rectangle_l127_127669


namespace pit_width_problem_l127_127246

theorem pit_width_problem
    (field_length field_width : ℝ)
    (pit_length pit_depth : ℝ)
    (height_rise : ℝ)
    (field_area : ℝ := field_length * field_width)
    (V_pit : ℝ := pit_length * (w : ℝ) * pit_depth)
    (area_pit : ℝ := pit_length * w)
    (remaining_area : ℝ := field_area - area_pit)
    (V_spread : ℝ := remaining_area * height_rise)
    (height_rise_eqn : V_pit = V_spread) :
    w = 5 :=
by {
    sorry
}

-- Instantiate the parameters:
variables (field_length := 20.0) (field_width := 10.0)
          (pit_length := 8.0) (pit_depth := 2.0)
          (height_rise := 0.5)
#eval pit_width_problem field_length field_width pit_length pit_depth height_rise

end pit_width_problem_l127_127246


namespace fraction_color_is_20_over_x_plus_20_l127_127746

variable (x y : ℕ)

def black_and_white_films := 20 * x
def color_films := 4 * y
def selected_black_and_white_films := (y : ℚ) / (x : ℚ) * (black_and_white_films x y) / 100
def selected_color_films := color_films x y
def fraction_of_selected_films_in_color := selected_color_films x y / (selected_black_and_white_films x y + selected_color_films x y)

theorem fraction_color_is_20_over_x_plus_20 : fraction_of_selected_films_in_color x y = 20 / (x + 20) := by
  sorry

end fraction_color_is_20_over_x_plus_20_l127_127746


namespace tangent_slope_at_origin_l127_127977

-- Defining the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x - a * real.log (x + 1)

-- Statement of the problem in Lean
theorem tangent_slope_at_origin (a : ℝ) : 
  (∀ x, curve a x = 2 * x) → a = -1 :=
by
  intro h
  sorry

end tangent_slope_at_origin_l127_127977


namespace perpendicular_vectors_dot_product_zero_l127_127955

theorem perpendicular_vectors_dot_product_zero (x : ℝ) 
  (a : ℝ × ℝ × ℝ := (-3, 2, 5))
  (b : ℝ × ℝ × ℝ := (1, x, -1))
  (h : (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0) : x = 4 :=
by
  sorry

end perpendicular_vectors_dot_product_zero_l127_127955


namespace weight_mixture_is_correct_l127_127751

noncomputable def weight_mixture_in_kg (weight_a_per_liter weight_b_per_liter : ℝ)
  (ratio_a ratio_b total_volume_liters weight_conversion : ℝ) : ℝ :=
  let total_parts := ratio_a + ratio_b
  let volume_per_part := total_volume_liters / total_parts
  let volume_a := ratio_a * volume_per_part
  let volume_b := ratio_b * volume_per_part
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  total_weight_gm / weight_conversion

theorem weight_mixture_is_correct :
  weight_mixture_in_kg 900 700 3 2 4 1000 = 3.280 :=
by
  -- Calculation should follow from the def
  sorry

end weight_mixture_is_correct_l127_127751


namespace carla_book_count_l127_127392

theorem carla_book_count (tiles_count books_count : ℕ) 
  (tiles_monday : tiles_count = 38)
  (total_tuesday_count : 2 * tiles_count + 3 * books_count = 301) : 
  books_count = 75 :=
by
  sorry

end carla_book_count_l127_127392


namespace ω_range_max_area_l127_127177

-- Define the given vectors and function
def m (ω x : ℝ) : ℝ × ℝ := (sin (ω * x) + cos (ω * x), sqrt 3 * cos (ω * x))
def n (ω x : ℝ) : ℝ × ℝ := (cos (ω * x) - sin (ω * x), 2 * sin (ω * x))
def f (ω x : ℝ) : ℝ := (m ω x).1 * (n ω x).1 + (m ω x).2 * (n ω x).2

-- Condition: The distance between two adjacent axes of symmetry of f(x) is at least π/2
def symmetry_condition (ω : ℝ) : Prop :=
  ω > 0 ∧ (2 * π) / (2 * ω) ≥ π

-- The maximum area of a triangle
def max_area_of_triangle (ω A : ℝ) (b c : ℝ) : ℝ :=
  if ω = 1 ∧ 2 * sin (2 * A + π / 6) = 1
  then (1 / 2) * b * c * sin A
  else 0

-- First proof: Prove range of ω
theorem  ω_range : ∀ ω : ℝ, symmetry_condition ω → 0 < ω ∧ ω ≤ 1 := 
sorry

-- Second proof: Prove maximum area of the triangle is √3
theorem max_area : ∀ (A b c : ℝ), let ω := 1 in symmetry_condition ω → (2 * sin (2 * A + π / 6) = 1) → 
                  b = 2 → max_area_of_triangle ω A b c = √3 := 
sorry

end ω_range_max_area_l127_127177


namespace supermarket_correctness_l127_127077

-- Statements of the conditions
def supermarket := Type
def goodsA := { x : ℕ // x = 2 * goodsB - 30}
def goodsB := { x : ℕ // 30 * x + 22 * (2 * x - 30) = 6000 }
def costA : ℕ := 22
def sellA : ℕ := 29
def costB : ℕ := 30 
def sellB : ℕ := 40

-- First purchase conditions
def purchase_cost : ℕ := 6000
def sold_goodsA (x : ℕ) : ℕ := 7 * x
def sold_goodsB (x : ℕ) : ℕ := 10 * x

-- Second purchase conditions
def profit_target : ℤ := 2350
def second_purchase := {x : ℕ // x = 3 * (goodsB x)}
def discount (x : ℕ) := (sellB // 2)

-- The main theorem declaration
theorem supermarket_correctness (goodsB : ℕ) (a : ℕ) (goodsA := 150) :
    30 * goodsB + 22 * (2 * goodsB - 30) = 6000 → 
    goodsB = 90 → 150 * 7 + 90 * 10 = 1950 →
    150 * 7 + (270 - a) * 10 - 10 * a = 2350 →
    a = 70 := 
sorry

end supermarket_correctness_l127_127077


namespace modular_inverse_problem_l127_127648

theorem modular_inverse_problem :
  ∃ b : ℤ, b % 13 = 6 ∧ b ≡ (4⁻¹ + 6⁻¹ + 9⁻¹)⁻¹ [MOD 13] := sorry

end modular_inverse_problem_l127_127648


namespace initial_percentage_of_managers_l127_127024

theorem initial_percentage_of_managers (P : ℕ) (h : 0 ≤ P ∧ P ≤ 100)
  (total_employees initial_managers : ℕ) 
  (h1 : total_employees = 500) 
  (h2 : initial_managers = P * total_employees / 100) 
  (remaining_employees remaining_managers : ℕ)
  (h3 : remaining_employees = total_employees - 250)
  (h4 : remaining_managers = initial_managers - 250)
  (h5 : remaining_managers * 100 = 98 * remaining_employees) :
  P = 99 := 
by
  sorry

end initial_percentage_of_managers_l127_127024


namespace distinct_terms_count_l127_127845

/-!
  Proving the number of distinct terms in the expansion of (x + 2y)^12
-/

theorem distinct_terms_count (x y : ℕ) : 
  (x + 2 * y) ^ 12 = 13 :=
by sorry

end distinct_terms_count_l127_127845


namespace finish_remaining_work_l127_127064

theorem finish_remaining_work (x y : ℕ) (hx : x = 30) (hy : y = 15) (hy_work_days : y_work_days = 10) :
  x = 10 :=
by
  sorry

end finish_remaining_work_l127_127064


namespace length_of_cable_l127_127626

noncomputable def sphere_radius (x y z : ℝ) : ℝ := 8
noncomputable def plane_distance (v : ℝ) : ℝ := v / real.sqrt 3
noncomputable def circle_radius (R d : ℝ) : ℝ := real.sqrt (R ^ 2 - d ^ 2)

theorem length_of_cable :
  (∃ (x y z : ℝ), x + y + z = 10 ∧ x * y + y * z + x * z = 18) →
  (∃ (l : ℝ), l = 4 * real.pi * real.sqrt (23 / 3)) :=
by
  intros h
  obtain ⟨x, y, z, h₁, h₂⟩ := h
  -- Definition of values based on conditions
  let R := sphere_radius x y z
  let d := plane_distance 10
  let r := circle_radius R d
  -- Expected proof that the calculated length is indeed 4 π √(23/3)
  use 4 * real.pi * r
  sorry

end length_of_cable_l127_127626


namespace dig_days_l127_127264

theorem dig_days (m1 m2 : ℕ) (d1 d2 : ℚ) (k : ℚ) 
  (h1 : m1 * d1 = k) (h2 : m2 * d2 = k) : 
  m1 = 30 ∧ d1 = 6 ∧ m2 = 40 → d2 = 4.5 := 
by sorry

end dig_days_l127_127264


namespace min_value_M_at_a_1_find_a_for_min_value_3_l127_127555

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + x + a^2 + a
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 - x + a^2 - a
noncomputable def M (x : ℝ) (a : ℝ) : ℝ := max (f x a) (g x a)

-- Part 1: Minimum value of M(x) when a = 1
theorem min_value_M_at_a_1 : (∃ x : ℝ, x ∈ ℝ ∧ M x 1 = 7 / 4) :=
sorry

-- Part 2: Find a when min value of M(x) is 3
theorem find_a_for_min_value_3 (a : ℝ) : (∃ x : ℝ, x ∈ ℝ ∧ M x a = 3) ↔ (a = (sqrt 14 - 1) / 2 ∨ a = -(sqrt 14 - 1) / 2) :=
sorry

end min_value_M_at_a_1_find_a_for_min_value_3_l127_127555


namespace factorization_correct_l127_127153

-- Define the expression
def expression (a b : ℝ) : ℝ := 3 * a^2 - 3 * b^2

-- Define the factorized form of the expression
def factorized (a b : ℝ) : ℝ := 3 * (a + b) * (a - b)

-- The main statement we need to prove
theorem factorization_correct (a b : ℝ) : expression a b = factorized a b :=
by 
  sorry -- Proof to be filled in

end factorization_correct_l127_127153


namespace determine_xyz_l127_127202

-- Define the conditions for the variables x, y, and z
variables (x y z : ℝ)

-- State the problem as a theorem
theorem determine_xyz :
  (x + y + z) * (x * y + x * z + y * z) = 24 ∧
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 8 →
  x * y * z = 16 / 3 :=
by
  intros h
  sorry

end determine_xyz_l127_127202
