import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Limits
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Arithmetic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Arithmetic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Lemmas
import Mathlib.Data.Ratio
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.CauSeqFilter

namespace differential_solution_l568_568765

theorem differential_solution (C : ℝ) : 
  ∃ y : ℝ → ℝ, (∀ x : ℝ, y x = C * (1 + x^2)) := 
by
  sorry

end differential_solution_l568_568765


namespace length_ninth_day_l568_568431

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range n, a i

def length_2_5_8 (a : ℕ → ℕ) : ℕ :=
  a 1 + a 4 + a 7

-- Constants
noncomputable def a : ℕ → ℕ := sorry

-- Conditions
axiom h1 : is_arithmetic_sequence a 
axiom h2 : sum_first_n_terms a 7 = 49
axiom h3 : length_2_5_8 a = 27

-- The desired proof statement
theorem length_ninth_day : a 8 = 17 := 
sorry

end length_ninth_day_l568_568431


namespace train_speed_is_10_l568_568595

-- Define the problem conditions
def length_of_train : ℝ := 90  -- Length is 90 meters
def time_to_cross_pole : ℝ := 9  -- Time is 9 seconds

-- Speed is defined as distance divided by time
def speed_of_train : ℝ := length_of_train / time_to_cross_pole

-- Theorem to prove the speed of the train
theorem train_speed_is_10 : speed_of_train = 10 := by
  sorry

end train_speed_is_10_l568_568595


namespace angle_equality_l568_568692

variable {α : Type} [LinearOrder α] [AddCommGroup α] [Module ℝ α] {A B C P G E F : α}
variable (triangle : Triangle α)

-- Conditions of the problem
def condition1 (hP : Inside P triangle) : Prop := 
  ∠ BPA = ∠ CPA

def condition2 (hG : G ∈ Segment A P) : Prop := 
  True

def condition3 (E F : α) (hE : E ∈ Line AC) (hF : F ∈ Line AB) : Prop := 
  ∃ BG OG : Line α, Intersect_at BG G AC E ∧ Intersect_at OG G AB F

-- Main proof problem statement
theorem angle_equality 
  (hP : Inside P triangle)
  (hG : G ∈ Segment A P)
  (h_conditions : ∃ (E F : α) (hE : E ∈ Line AC) (hF : F ∈ Line AB),
    Intersect_at (Line_through B G) G AC E ∧ Intersect_at (Line_through O G) G AB F)
  (hBPAeqCPA : ∠ BPA = ∠ CPA) :
  ∠ BPF = ∠ CPE :=
by 
  sorry

end angle_equality_l568_568692


namespace p_is_necessary_but_not_sufficient_for_q_l568_568669

variable (x : ℝ)
def p := |x| ≤ 2
def q := 0 ≤ x ∧ x ≤ 2

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬ q x := by
  sorry

end p_is_necessary_but_not_sufficient_for_q_l568_568669


namespace students_errors_proof_l568_568611

noncomputable def students (x y0 y1 y2 y3 y4 y5 : ℕ): ℕ :=
  x + y5 + y4 + y3 + y2 + y1 + y0

noncomputable def errors (x y1 y2 y3 y4 y5 : ℕ): ℕ :=
  6 * x + 5 * y5 + 4 * y4 + 3 * y3 + 2 * y2 + y1

theorem students_errors_proof
  (x y0 y1 y2 y3 y4 y5 : ℕ)
  (h1 : students x y0 y1 y2 y3 y4 y5 = 333)
  (h2 : errors x y1 y2 y3 y4 y5 ≤ 1000) :
  x ≤ y3 + y2 + y1 + y0 :=
by
  sorry

end students_errors_proof_l568_568611


namespace limit_of_sequence_l568_568678

noncomputable theory

def sequence (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧
a 2 = 3 ∧ 
(∀ n : ℕ, 0 < n → |a (n + 1) - a n| = 2 ^ n) ∧ 
strict_mono (λ n, a (2 * n - 1)) ∧ -- strictly increasing for odd indices
strict_antimono (λ n, a (2 * n)) -- strictly decreasing for even indices

theorem limit_of_sequence (a : ℕ → ℝ) (h : sequence a) : 
  tendsto (λ n, (a (2 * n - 1) / a (2 * n))) at_top (𝓝 (-1 / 2)) :=
sorry

end limit_of_sequence_l568_568678


namespace dodecagon_diagonals_l568_568584

theorem dodecagon_diagonals (n : ℕ) (h : n = 12) : (n * (n - 3)) / 2 = 54 :=
by
  rw [h]
  norm_num
  sorry

end dodecagon_diagonals_l568_568584


namespace largest_distance_between_spheres_l568_568897

theorem largest_distance_between_spheres :
  let c1 := (4 : ℝ, -5, 10)
  let r1 := 15
  let c2 := (-6 : ℝ, 20, -10)
  let r2 := 50
  let distance := λ (p1 p2 : ℝ × ℝ × ℝ), real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)
  ∃ (A : ℝ × ℝ × ℝ) (B : ℝ × ℝ × ℝ), 
    distance c1 A = r1 ∧ 
    distance c2 B = r2 ∧ 
    distance A B = 65 + 25 * real.sqrt 3 :=
sorry

end largest_distance_between_spheres_l568_568897


namespace incorrect_statement_l568_568674

-- Definitions of points in a cube structure
variable (A B C D A1 B1 C1 D1 : Point)
variable (Cube ABCD_A1B1C1D1 : is_cube A B C D A1 B1 C1 D1)

-- Definitions of lines
def BD : Line := line_through B D
def AC1 : Line := line_through A C1
def AD : Line := line_through A D
def CB1 : Line := line_through C B1

-- Definitions of planes
def CB1D1_plane : Plane := plane_through C B1 D1

-- Properties to check (from statements A to D)
def StatementA : Prop := BD ∥ CB1D1_plane
def StatementB : Prop := AC1 ⟂ BD
def StatementC : Prop := AC1 ⟂ CB1D1_plane
def StatementD : Prop := angle_skew_lines AD CB1 = 60

-- The proof goal
theorem incorrect_statement : StatementD := 
sorry

end incorrect_statement_l568_568674


namespace problem_statement_l568_568668

theorem problem_statement {a b c : ℝ} (h₁ : 0 = 16 * a + 4 * b) (h₂ : ax^2 + bx + c) (h₃ : a > 0):
  4 * a + b = 0 ∧ a > 0 :=
sorry

end problem_statement_l568_568668


namespace describe_graph_l568_568903

theorem describe_graph :
  ∀ (x y : ℝ), ((x + y) ^ 2 = x ^ 2 + y ^ 2 + 4 * x) ↔ (x = 0 ∨ y = 2) := 
by
  sorry

end describe_graph_l568_568903


namespace function_domain_l568_568097

theorem function_domain (x : ℝ) :
  0 ≤ 9 - x^2 →
  0 < x + 1 →
  x + 1 ≠ 1 →
  (x ∈ set.interval (-1 : ℝ) 0 ∪ set.interval 0 3) :=
by
  intros h1 h2 h3
  sorry

end function_domain_l568_568097


namespace union_A_B_l568_568352

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem union_A_B :
  A ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

end union_A_B_l568_568352


namespace largest_spherical_ball_radius_in_torus_l568_568949

theorem largest_spherical_ball_radius_in_torus 
    (inner_radius outer_radius : ℝ) 
    (circle_center : ℝ × ℝ × ℝ) 
    (circle_radius : ℝ) 
    (r : ℝ)
    (h0 : inner_radius = 2)
    (h1 : outer_radius = 4)
    (h2 : circle_center = (3, 0, 1))
    (h3 : circle_radius = 1)
    (h4 : 3^2 + (r - 1)^2 = (r + 1)^2) :
    r = 9 / 4 :=
by
  sorry

end largest_spherical_ball_radius_in_torus_l568_568949


namespace prove_cos_C_prove_side_lengths_l568_568393

variables {a b c : ℝ} {A B C : ℝ}

-- Conditions from the problem
axiom angle_C_half_sine : sin(C / 2) = sqrt 10 / 4
axiom triangle_area : (1/2) * a * b * sin C = 3 * sqrt 15 / 4
axiom sine_square_relation : sin A^2 + sin B^2 = (13 / 16) * sin C^2

-- Derived condition from previous answer
axiom cos_C_value : cos C = -1 / 4

-- Proof Problem 1
theorem prove_cos_C : cos C = -1 / 4 := by sorry

-- Proof Problem 2
theorem prove_side_lengths : 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 2 ∧ c = 4) := by sorry

end prove_cos_C_prove_side_lengths_l568_568393


namespace geom_seq_sum_abs_eq_l568_568002

noncomputable def geom_seq_sum_abs (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ k in finset.range n, |a k|

theorem geom_seq_sum_abs_eq (n : ℕ) (a : ℕ → ℝ) (h1 : a 1 = 1/2) (h4 : a 4 = -4) :
  geom_seq_sum_abs a n = 2^(n-1) - 1/2 := 
sorry

end geom_seq_sum_abs_eq_l568_568002


namespace avg_student_headcount_l568_568894

def student_headcount (yr1 yr2 yr3 yr4 : ℕ) : ℕ :=
  (yr1 + yr2 + yr3 + yr4) / 4

theorem avg_student_headcount :
  student_headcount 10600 10800 10500 10400 = 10825 :=
by
  sorry

end avg_student_headcount_l568_568894


namespace find_even_digits_in_product_l568_568656

def num_even_digits (n : Nat) : Nat :=
  (n.toString.data.filter (λ c => c.isDigit ∧ (c.toNat - '0'.toNat).mod 2 = 0)).length

theorem find_even_digits_in_product :
  num_even_digits (2222222222 * 9999999999) = 11 := by
  sorry

end find_even_digits_in_product_l568_568656


namespace union_A_B_eq_C_l568_568685

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
noncomputable def C : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}

theorem union_A_B_eq_C : A ∪ B = C := by
  sorry

end union_A_B_eq_C_l568_568685


namespace range_of_x0_y0_square_l568_568733

theorem range_of_x0_y0_square {x_0 y_0 : ℝ} 
  (hP : ∃ x y : ℝ, (x - 2*y - 2 = 0) ∧ (x - x_0) = x - (x_0 - (x - 2 * y)) ∧ (y - y_0) = y - (y_0 - y))
  (hQ : ∃ x y : ℝ, (x - 2*y - 6 = 0) ∧ (x - x_0) = x - (x_0 - (x - 2 * y)) ∧ (y - y_0) = y - (y_0 - y))
  (hm : (x_0 - 2) ^ 2 + (y_0 + 1) ^ 2 ≤ 5) :
  ∃ l u : ℝ, l = 16 / 5 ∧ u = 16 ∧ l ≤ x_0^2 + y_0^2 ∧ x_0^2 + y_0^2 ≤ u := 
begin
  sorry
end

end range_of_x0_y0_square_l568_568733


namespace corrected_mean_35_25_l568_568500

theorem corrected_mean_35_25 (n : ℕ) (mean : ℚ) (x_wrong x_correct : ℚ) :
  n = 20 → mean = 36 → x_wrong = 40 → x_correct = 25 → 
  ( (mean * n - x_wrong + x_correct) / n = 35.25) :=
by
  intros h1 h2 h3 h4
  sorry

end corrected_mean_35_25_l568_568500


namespace ratio_of_typing_speed_l568_568770

variable (J : ℕ) (K : ℕ)
variable (h1 : J ≠ 0) -- Ensuring non-zero typing speed
variable (h2 : Jack_time : ℝ := 4.999999999999999)

-- John can type a set of pages (5J) in 5 hours
def john_typing_time (J : ℕ) : ℝ := 5

-- John types for 3 hours at rate J pages per hour
def john_pages (J : ℕ) : ℝ := 3 * J

-- Remaining pages to be typed by Jack in approximately 5 hours
def remaining_pages (J : ℕ) : ℝ := 5 * J - 3 * J

-- Jack's typing speed defined
def jack_typing_speed : ℝ := remaining_pages J / Jack_time

-- The ratio of Jack's typing speed to John's typing speed
def typing_speed_ratio (J : ℕ) (K : ℕ) : ℝ := jack_typing_speed J / J

theorem ratio_of_typing_speed (J : ℕ) (h1 : J ≠ 0) :
  typing_speed_ratio J (jack_typing_speed J) = 2 / 5 := by
  sorry

end ratio_of_typing_speed_l568_568770


namespace stratified_sampling_correct_l568_568200

-- Definitions based on the given conditions
def total_students : ℕ := 900
def freshmen : ℕ := 300
def sophomores : ℕ := 200
def juniors : ℕ := 400
def sample_size : ℕ := 45

-- Stratified sampling method ensures the proportion of each subgroup is maintained
def proportion : ℚ := sample_size / total_students

def freshmen_sample : ℕ := (freshmen * proportion).toNat
def sophomores_sample : ℕ := (sophomores * proportion).toNat
def juniors_sample : ℕ := (juniors * proportion).toNat

-- Theorem to prove the numbers drawn using stratified sampling
theorem stratified_sampling_correct :
  freshmen_sample = 15 ∧ sophomores_sample = 10 ∧ juniors_sample = 20 :=
by
  have h_proportion : proportion = 1 / 20 := sorry
  have h1 : freshmen_sample = (300 * (1/20)).toNat := sorry
  have h2 : sophomores_sample = (200 * (1/20)).toNat := sorry
  have h3 : juniors_sample = (400 * (1/20)).toNat := sorry
  rw [h1, h2, h3],
  split,
  all_goals { exact (by norm_num) }

end stratified_sampling_correct_l568_568200


namespace juvy_chives_l568_568772

-- Definitions based on the problem conditions
def total_rows : Nat := 20
def plants_per_row : Nat := 10
def parsley_rows : Nat := 3
def rosemary_rows : Nat := 2
def chive_rows : Nat := total_rows - (parsley_rows + rosemary_rows)

-- The statement we want to prove
theorem juvy_chives : chive_rows * plants_per_row = 150 := by
  sorry

end juvy_chives_l568_568772


namespace pies_sold_in_week_l568_568578

def daily_pies := 8
def days_in_week := 7
def total_pies := 56

theorem pies_sold_in_week : daily_pies * days_in_week = total_pies :=
by
  sorry

end pies_sold_in_week_l568_568578


namespace probability_reach_1_probability_reach_neg_1_probability_return_0_l568_568469

theorem probability_reach_1 (x : ℝ) 
  (particle_start_at_0 : true)
  (move_prob : ∀ n, (Math.random_uniform (List.range (n - 1) ((n + 1) + 1)) = 0.5))
  : x = 1 :=
sorry

theorem probability_reach_neg_1 (y : ℝ) 
  (particle_start_at_0 : true)
  (move_prob : ∀ n, (Math.random_uniform (List.range (n - 1) ((n + 1) + 1)) = 0.5))
  : y = 1 :=
sorry

theorem probability_return_0 (z : ℝ) 
  (particle_start_at_0 : true)
  (move_prob : ∀ n, (Math.random_uniform (List.range (n - 1) ((n + 1) + 1)) = 0.5))
  : z = 1 :=
sorry

end probability_reach_1_probability_reach_neg_1_probability_return_0_l568_568469


namespace mn_sum_l568_568721

theorem mn_sum {m n : ℤ} (h : ∀ x : ℤ, (x + 8) * (x - 1) = x^2 + m * x + n) : m + n = -1 :=
by
  sorry

end mn_sum_l568_568721


namespace area_T3_l568_568764

def R_1 : Type := { side_length : ℝ // side_length ^ 2 = 81 }

def T_1 (R : R_1) : Type := 
  { side_length : ℝ // side_length = R.side_length / Real.sqrt 3 }

def T_2 (T : T_1) : Type := 
  { side_length : ℝ // side_length = (Real.sqrt 3 / 4) * T.side_length }

def T_3 (T : T_2) : Type := 
  { side_length : ℝ // side_length = (1 / 2) * T.side_length }

def area (T : {side_length : ℝ}) : ℝ :=
  (Real.sqrt 3 / 4) * T.side_length ^ 2

theorem area_T3 : ∀ (T3 : T_3 { side_length := 9/(2*Real.sqrt 3) }),
  area T3 = 81 * Real.sqrt 3 / 256 :=
by 
  sorry

end area_T3_l568_568764


namespace complex_expression_equality_l568_568093

theorem complex_expression_equality :
  (1 + Complex.i)^4 / (1 - Complex.i) + 2 = -2 * Complex.i :=
  sorry

end complex_expression_equality_l568_568093


namespace probability_both_heads_on_last_flip_l568_568523

noncomputable def fair_coin_flip : probabilityₓ ℙ :=
  probabilityₓ.ofUniform [true, false]

def both_coins_heads (events : list (bool × bool)) : bool :=
  events.all (λ event, event.1 = true)

def stops_with_heads (events : list (bool × bool)) : bool :=
  events.any (λ event, event.1 = true ∨ event.2 = true)

theorem probability_both_heads_on_last_flip :
  ∀ events : list (bool × bool), probabilityₓ (fair_coin_flip ×ₗ fair_coin_flip)
  (λ event, both_coins_heads events = true ∧ stops_with_heads events = true) = 1 / 3 :=
sorry

end probability_both_heads_on_last_flip_l568_568523


namespace ratio_of_ages_l568_568520

variable (T N : ℕ)
variable (sum_ages : T = T) -- This is tautological based on the given condition; we can consider it a given sum
variable (age_condition : T - N = 3 * (T - 3 * N))

theorem ratio_of_ages (T N : ℕ) (sum_ages : T = T) (age_condition : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end ratio_of_ages_l568_568520


namespace sums_are_different_l568_568955

variable (N : ℕ)
variable (a : Fin N → ℤ)
variable (h_sum : ∑ i in Finset.range N, a i = 1)

noncomputable def S (i : Fin N) : ℤ :=
  ∑ k in Finset.range N, (k + 1) * a ((i + k) % N)

theorem sums_are_different (i j : Fin N) (h_ij : i ≠ j) : S N a i ≠ S N a j := by
  sorry

end sums_are_different_l568_568955


namespace hotel_charge_percentage_l568_568842

theorem hotel_charge_percentage (G R P : ℝ) 
  (hR : R = 1.60 * G) 
  (hP : P = 0.80 * G) : 
  ((R - P) / R) * 100 = 50 := by
  sorry

end hotel_charge_percentage_l568_568842


namespace shift_right_three_units_l568_568417

theorem shift_right_three_units (x : ℝ) : (λ x, -2 * x) (x - 3) = -2 * x + 6 :=
by
  sorry

end shift_right_three_units_l568_568417


namespace intervals_of_increase_range_of_fA_in_acute_triangle_l568_568788

noncomputable def f (x : ℝ) : ℝ := 
  cos x * (2 * sqrt 3 * sin x - cos x) + cos (π / 2 - x) ^ 2

theorem intervals_of_increase (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) :
  x ∈ [0, π] → (f x > 0 ∧ f x ≤ f (x + ε)) :=
sorry

theorem range_of_fA_in_acute_triangle 
  (a b c A B C : ℝ) -- angles are in radians
  (h1 : acute_triangle A B C)
  (h2 : a^2 + c^2 - b^2 = c * (a^2 + b^2 - c^2) / (2 * a - c)) :
  1 < 2 * sin (2 * A - π / 6) ∧ 2 * sin (2 * A - π / 6) ≤ 2 :=
sorry

end intervals_of_increase_range_of_fA_in_acute_triangle_l568_568788


namespace tetrahedron_half_volume_intersection_distance_l568_568959

theorem tetrahedron_half_volume_intersection_distance (r : ℝ) (x : ℝ) :
  (∀ (a : ℝ), a = r * sqrt 3 → 
  (∀ (m : ℝ), m = (sqrt 3 * a) / 2 → 
  1 / 2 = (x / m) ^ 3 → 
  x = r * (sqrt 2 - real.cbrt 2 ^ (1 / 3))))  :=
begin
  sorry
end

end tetrahedron_half_volume_intersection_distance_l568_568959


namespace compute_fraction_product_l568_568977

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l568_568977


namespace shift_right_linear_function_l568_568411

theorem shift_right_linear_function (x : ℝ) : 
  (∃ k b : ℝ, k ≠ 0 ∧ (∀ x : ℝ, y = -2x → y = kx + b) → (x, y) = (x - 3, -2(x-3))) → y = -2x + 6 :=
by
  sorry

end shift_right_linear_function_l568_568411


namespace composite_sequence_99_percent_l568_568251

theorem composite_sequence_99_percent : 
  let seq := λ n: ℕ, 10^n + 1 in 
  ∃ (composite_count : ℕ), composite_count = 2000 - 11 ∧ 
  (composite_count : ℕ) / 2000.0 ≥ 0.99 :=
by sorry

end composite_sequence_99_percent_l568_568251


namespace find_x_l568_568660

theorem find_x (x : ℝ) (h : sqrt (x + 16) = 12) : x = 128 :=
by
  sorry

end find_x_l568_568660


namespace solve_equations_l568_568084

theorem solve_equations :
  (∃ x1 x2 : ℝ, (x1 = 1 ∧ x2 = 3) ∧ (x1^2 - 4 * x1 + 3 = 0) ∧ (x2^2 - 4 * x2 + 3 = 0)) ∧
  (∃ y1 y2 : ℝ, (y1 = 9 ∧ y2 = 11 / 7) ∧ (4 * (2 * y1 - 5)^2 = (3 * y1 - 1)^2) ∧ (4 * (2 * y2 - 5)^2 = (3 * y2 - 1)^2)) :=
by
  sorry

end solve_equations_l568_568084


namespace total_pies_sold_l568_568577

-- Defining the conditions
def pies_per_day : ℕ := 8
def days_in_week : ℕ := 7

-- Proving the question
theorem total_pies_sold : pies_per_day * days_in_week = 56 :=
by
  sorry

end total_pies_sold_l568_568577


namespace solve_for_y_l568_568079

theorem solve_for_y (y : ℝ) (h : 5 * (y ^ (1/3)) - 3 * (y / (y ^ (2/3))) = 10 + (y ^ (1/3))) :
  y = 1000 :=
by {
  sorry
}

end solve_for_y_l568_568079


namespace prove_parallel_FG_HE_l568_568112

open EuclideanGeometry

variables {A B C D E F G H K L : Point}

-- Assume we have rhombus ABCD
def is_rhombus (A B C D : Point) : Prop := 
  is_parallelogram A B C D ∧ dist A B = dist B C

-- Assume points E, F, G, H are on the sides of the rhombus
variables (E_on_DA : OnLine E (Line.mk D A))
variables (F_on_AB : OnLine F (Line.mk A B))
variables (G_on_BC : OnLine G (Line.mk B C))
variables (H_on_CD : OnLine H (Line.mk C D))

-- Assume EF and GH are tangent to the incircle of the rhombus
variables (EF_tangent_incircle : tangent (Segment.mk E F) (incircle A B C D))
variables (GH_tangent_incircle : tangent (Segment.mk G H) (incircle A B C D))

theorem prove_parallel_FG_HE 
  (rhombus_ABCD : is_rhombus A B C D)
  (E_on_DA : OnLine E (Line.mk D A)) 
  (F_on_AB : OnLine F (Line.mk A B))
  (G_on_BC : OnLine G (Line.mk B C))
  (H_on_CD : OnLine H (Line.mk C D))
  (EF_tangent_incircle : tangent (Segment.mk E F) (incircle A B C D))
  (GH_tangent_incircle : tangent (Segment.mk G H) (incircle A B C D)) :
  Parallel (Line.mk F G) (Line.mk H E) :=
begin
  sorry
end

end prove_parallel_FG_HE_l568_568112


namespace count_valid_hex_numbers_sum_digits_l568_568602

theorem count_valid_hex_numbers_sum_digits :
  let count := (List.range 500).filter (λ n, ∀ d ∈ n.digits 16, d ≤ 9)
  in count.length = 150 ∧ (count.length.digits 10).sum = 6 :=
sorry

end count_valid_hex_numbers_sum_digits_l568_568602


namespace coeff_x3y3_in_expansion_l568_568532

theorem coeff_x3y3_in_expansion : 
  (coeff (expand (x + y) 6) (monomial 3 3)) = 20 :=
sorry

end coeff_x3y3_in_expansion_l568_568532


namespace ribbons_green_count_l568_568744

theorem ribbons_green_count
  (N : ℕ)  -- The total number of ribbons
  (red_ribbons : ℕ := N / 4)   -- Red ribbons are 1/4 of the total
  (blue_ribbons : ℕ := 3 * N / 8)   -- Blue ribbons are 3/8 of the total
  (green_ribbons : ℕ := N / 8)   -- Green ribbons are 1/8 of the total
  (white_ribbons : ℕ := 36) -- The remaining ribbons are white
  (h : N - (red_ribbons + blue_ribbons + green_ribbons) = white_ribbons) :
  green_ribbons = 18 := sorry

end ribbons_green_count_l568_568744


namespace probability_at_least_one_red_l568_568961

theorem probability_at_least_one_red (total_balls red_balls white_balls drawn_balls : ℕ) 
  (h1 : total_balls = 4) (h2 : red_balls = 2) (h3 : white_balls = 2) (h4 : drawn_balls = 2) : 
  prob_at_least_one_red (total_balls red_balls white_balls drawn_balls) = 5 / 6 := 
sorry

end probability_at_least_one_red_l568_568961


namespace area_triang_eq_area_quad_l568_568505

-- Assume we have points A, B, C forming an acute-angled triangle
variables {A B C L N K M : Point}
-- hABC states that ABC forms an acute-angled triangle
-- hAL_Bisector states that AL is the angle bisector of ∠BAC, with L on BC
-- hAN_circumcircle states that N is on the circumcircle of triangle ABC where AL extended intersects again
-- hLK_perpendicular states that LK is perpendicular to AB
-- hLM_perpendicular states that LM is perpendicular to AC

theorem area_triang_eq_area_quad (hABC : acute_triangle A B C)
    (hAL_Bisector : angle_bisector A B C L)
    (hAN_circle : on_circumcircle A L N)
    (hL_BC : L ∈ line_segment B C)
    (hLK_perpendicular : perpendicular (segment L K) (segment A B))
    (hLM_perpendicular : perpendicular (segment L M) (segment A C)) :
    area (triangle A B C) = area (quadrilateral A K N M) := 
    sorry

end area_triang_eq_area_quad_l568_568505


namespace range_of_magnitude_is_3_to_7_l568_568370

noncomputable def range_of_complex_magnitude (z : ℂ) (hz : |z| = 2) : set ℝ :=
  {r | ∃ w, |w| = 2 ∧ r = complex.abs (w + 4 - 3 * complex.I)}

theorem range_of_magnitude_is_3_to_7 (z : ℂ) (hz : |z| = 2) :
  range_of_complex_magnitude z hz = set.Icc 3 7 :=
sorry

end range_of_magnitude_is_3_to_7_l568_568370


namespace pictures_per_album_l568_568551

-- Define the problem conditions
def picturesFromPhone : Nat := 35
def picturesFromCamera : Nat := 5
def totalAlbums : Nat := 5

-- Define the total number of pictures
def totalPictures : Nat := picturesFromPhone + picturesFromCamera

-- Define what we need to prove
theorem pictures_per_album :
  totalPictures / totalAlbums = 8 := by
  sorry

end pictures_per_album_l568_568551


namespace compute_fraction_power_mul_l568_568984

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end compute_fraction_power_mul_l568_568984


namespace percentage_difference_is_14_25_l568_568086

def euro_to_dollar : ℝ := 1.5
def transaction_fee : ℝ := 0.02
def diana_dollars : ℝ := 600
def etienne_euros : ℝ := 350

theorem percentage_difference_is_14_25 :
  let etienne_dollars := etienne_euros * euro_to_dollar
  let etienne_dollars_after_fee := etienne_dollars * (1 - transaction_fee)
  let percentage_difference := (diana_dollars - etienne_dollars_after_fee) / diana_dollars * 100
  percentage_difference ≈ 14.25 :=
by
  sorry

end percentage_difference_is_14_25_l568_568086


namespace angle_BAC_is_90_degrees_l568_568599

noncomputable def point_A : ℝ × ℝ := (0, 100)
noncomputable def point_B : ℝ × ℝ := (30, -90)
noncomputable def point_C : ℝ × ℝ × ℝ := (90, 0, 2000)
def R := 6400  -- radius of Earth in km

theorem angle_BAC_is_90_degrees :
  let OA := (R * Math.cos(100 * π / 180), R * Math.sin(100 * π / 180), 0),
      OB := (R * Math.cos(30 * π / 180) * Math.cos(-90 * π / 180), R * Math.cos(30 * π / 180) * Math.sin(-90 * π / 180), R * Math.sin(30 * π / 180)),
      OC := (0, 0, R + 2)
  in
  let dot_AB := OA.1 * OB.1 + OA.2 * OB.2 + OA.3 * OB.3,
      norm_OA := R,
      norm_OB := R
  in
  ∠ ⟨OA, OB, OC⟩ = π / 2 :=
begin
  sorry
end

end angle_BAC_is_90_degrees_l568_568599


namespace max_sum_of_heights_l568_568962

-- Defining the problem in Lean 4
theorem max_sum_of_heights (T : Type) [triangle T] (m1 m2 m3 : ℝ) (h1 h2 h3 : ℝ) 
  (sum_medians : m1 + m2 + m3 = 3):
  (∃ a : ℝ, a = (2 * sqrt(3) / 3) ∧ 
           (h1 = sqrt(3)/2 * a ∧ h2 = sqrt(3)/2 * a ∧ h3 = sqrt(3)/2 * a ∧ 
           h1 + h2 + h3 = 3)) :=
by
  sorry

end max_sum_of_heights_l568_568962


namespace KC_bisects_B1E_l568_568017

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def big_arc_midpoint (A B C : Point) : Point := sorry
noncomputable def point_inside_AC (A C : Point) (BC_eq_CB1 : Bool) : Point := sorry
noncomputable def tangent_point (P Q : Point) : Point := sorry
noncomputable def intersection_of_circles (circle1 circle2 : Circle) : Point := sorry
noncomputable def bisects (P Q R : Point) : Prop := sorry

variable (A B C : Point)
variable (AC_gt_CB : AC > CB)
variable (M : Point := midpoint A B)
variable (Q : Point := big_arc_midpoint A B C)
variable (B1 : Point := point_inside_AC A C (BC = CB1))
variable (E : Point := tangent_point B1 Q)
variable (K : Point := intersection_of_circles (circle B B1 M) (circle A B C))

theorem KC_bisects_B1E : bisects K C B1 E :=
sorry

end KC_bisects_B1E_l568_568017


namespace chloe_total_score_l568_568920

def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

def score_first_level : ℕ := treasures_first_level * points_per_treasure
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := score_first_level + score_second_level

theorem chloe_total_score : total_score = 81 := by
  sorry

end chloe_total_score_l568_568920


namespace least_number_to_add_l568_568159

theorem least_number_to_add (LCM : ℕ) (a : ℕ) (x : ℕ) :
  LCM = 23 * 29 * 31 →
  a = 1076 →
  x = LCM - a →
  (a + x) % LCM = 0 :=
by
  sorry

end least_number_to_add_l568_568159


namespace folded_triangle_square_length_l568_568939

theorem folded_triangle_square_length (side_length folded_distance length_squared : ℚ) 
(h1: side_length = 15) 
(h2: folded_distance = 11) 
(h3: length_squared = 1043281/31109) :
∃ (PQ : ℚ), PQ^2 = length_squared := 
by 
  sorry

end folded_triangle_square_length_l568_568939


namespace Nicole_has_69_clothes_l568_568053

def clothingDistribution : Prop :=
  let nicole_clothes := 15
  let first_sister_clothes := nicole_clothes / 3
  let second_sister_clothes := nicole_clothes + 5
  let third_sister_clothes := 2 * first_sister_clothes
  let average_clothes := (nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes) / 4
  let oldest_sister_clothes := 1.5 * average_clothes
  let total_clothes := nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes + oldest_sister_clothes
  total_clothes = 69

theorem Nicole_has_69_clothes : clothingDistribution :=
by
  -- Proof omitted
  sorry

end Nicole_has_69_clothes_l568_568053


namespace profit_percentage_l568_568943

/-- 
A retailer bought a machine at a wholesale price of $90 and later on sold it after a 10% discount 
of the retail price. The retailer made a profit equivalent to a certain percentage of the wholesale price. 
The retail price of the machine is $120. 
-/
theorem profit_percentage (wholesale_price retail_price : ℕ) (discount_percentage : ℝ) : 
  wholesale_price = 90 → 
  retail_price = 120 → 
  discount_percentage = 0.10 → 
  let discount := discount_percentage * retail_price in 
  let selling_price := retail_price - discount in 
  let profit := selling_price - wholesale_price in 
  ((profit / wholesale_price) * 100) = 20 :=
begin
  intros,
  sorry
end

end profit_percentage_l568_568943


namespace centroid_length_ratio_l568_568450

-- Definitions of the points and conditions:
variable (A B C : Point)
variable [Triangle A B C]

-- G is the centroid of triangle ABC
variable (G : Point)
variable [Centroid G A B C]

-- M is the midpoint of segment BC
variable (M : Point)
variable [Midpoint M B C]

theorem centroid_length_ratio (A B C G M : Point)
  [Triangle A B C]
  [Centroid G A B C]
  [Midpoint M B C] :
  dist A G = (2 / 3) * dist A M := 
by 
  sorry

end centroid_length_ratio_l568_568450


namespace work_done_by_force_l568_568208

def F (x : ℝ) : ℝ := 1 - Real.exp (-x)

theorem work_done_by_force :
  ∫ x in 0..1, F x = 1 / Real.exp 1 :=
by 
  -- The steps of integration and evaluation are skipped here for brevity.
  sorry

end work_done_by_force_l568_568208


namespace number_17_more_than_5_times_X_number_less_than_5_times_22_by_Y_l568_568502

variable (X Y : ℕ)

theorem number_17_more_than_5_times_X : (5 * X) + 17 = 5 * X + 17 := by
  refl

theorem number_less_than_5_times_22_by_Y : (22 * 5) - Y = 110 - Y := by
  refl

end number_17_more_than_5_times_X_number_less_than_5_times_22_by_Y_l568_568502


namespace problem_abc_l568_568061

theorem problem_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by
  sorry

end problem_abc_l568_568061


namespace solve_for_p_l568_568552

noncomputable def f (p : ℝ) := 2 * p - 20

theorem solve_for_p : ∃ p : ℝ, f(f(f(p))) = 6 ∧ p = 18.25 :=
by {
  sorry
}

end solve_for_p_l568_568552


namespace cone_base_radius_larger_than_cylinder_l568_568155
-- Bringing in required library for basic mathematical constructs.

-- Definitions and conditions based on the problem statement.
variable {R_k R_h m_k m_h : ℝ} -- Define the variables for radii and heights.

-- Given conditions
def unit_sphere_radius : ℝ := 1

def cone_radius_eq : R_k ^ 2 + ((m_k - unit_sphere_radius) / 2) ^ 2 = unit_sphere_radius ^ 2 :=
  by sorry

def cylinder_radius_eq : R_h ^ 2 + (m_h / 2) ^ 2 = unit_sphere_radius ^ 2 :=
  by sorry

def volume_cone (R_k m_k : ℝ) : ℝ := (1 / 3) * Mathlib.pi * R_k ^ 2 * m_k

def volume_cylinder (R_h m_h : ℝ) : ℝ := Mathlib.pi * R_h ^ 2 * m_h

axiom max_cone_height : m_k = 4 / 3
axiom max_cylinder_height : m_h = sqrt (4 / 3)

-- Conclusion to be proven
theorem cone_base_radius_larger_than_cylinder : R_k > R_h :=
  by sorry

end cone_base_radius_larger_than_cylinder_l568_568155


namespace shifted_function_is_correct_l568_568420

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end shifted_function_is_correct_l568_568420


namespace barbara_other_goods_cost_in_home_currency_l568_568966

noncomputable def cost_of_tuna := 5 * 2
noncomputable def cost_of_water := 4 * 1.5
noncomputable def total_cost_of_tuna_and_water := cost_of_tuna + cost_of_water
noncomputable def paid_after_discount := 56
noncomputable def discount_rate := 0.9
noncomputable def total_cost_before_discount := paid_after_discount / discount_rate
noncomputable def cost_of_other_goods := total_cost_before_discount - total_cost_of_tuna_and_water
noncomputable def conversion_rate := 1.5

theorem barbara_other_goods_cost_in_home_currency :
  (cost_of_other_goods / conversion_rate) = 30.81 :=
by {
  rw [cost_of_tuna, cost_of_water, total_cost_of_tuna_and_water, paid_after_discount, discount_rate, total_cost_before_discount, cost_of_other_goods],
  norm_num,
  sorry
}

end barbara_other_goods_cost_in_home_currency_l568_568966


namespace sum_of_angles_eq_92_l568_568122

theorem sum_of_angles_eq_92 :
  (∑ n in {2..44}, 2 * sin n * sin 1 * (1 + (sec (n-1)) * (sec (n+1)))) 
  = ∑ n in {1, 2, 44, 45}, -(-1)^n * (sin n) / (cos(n)) :=
sorry

end sum_of_angles_eq_92_l568_568122


namespace max_product_production_l568_568181

theorem max_product_production (C_mats A_mats C_ship A_ship B_mats B_ship : ℝ)
  (cost_A cost_B ship_A ship_B : ℝ) (prod_A prod_B max_cost_mats max_cost_ship prod_max : ℝ)
  (h_prod_A : prod_A = 90)
  (h_cost_A : cost_A = 1000)
  (h_ship_A : ship_A = 500)
  (h_prod_B : prod_B = 100)
  (h_cost_B : cost_B = 1500)
  (h_ship_B : ship_B = 400)
  (h_max_cost_mats : max_cost_mats = 6000)
  (h_max_cost_ship : max_cost_ship = 2000)
  (h_prod_max : prod_max = 440)
  (H_C_mats : C_mats = cost_A * A_mats + cost_B * B_mats)
  (H_C_ship : C_ship = ship_A * A_ship + ship_B * B_ship)
  (H_A_mats_ship : A_mats = A_ship)
  (H_B_mats_ship : B_mats = B_ship)
  (H_C_mats_le : C_mats ≤ max_cost_mats)
  (H_C_ship_le : C_ship ≤ max_cost_ship) :
  prod_A * A_mats + prod_B * B_mats ≤ prod_max :=
by {
  sorry
}

end max_product_production_l568_568181


namespace dodecagon_diagonals_l568_568590

theorem dodecagon_diagonals :
  ∀ n : ℕ, n = 12 → (n * (n - 3)) / 2 = 54 :=
begin
  intros n hn,
  rw hn,
  norm_num,
end

end dodecagon_diagonals_l568_568590


namespace map_distance_proof_l568_568470

theorem map_distance_proof (scale_cm : ℝ) (scale_km : ℝ) (map_distance_cm : ℝ) (actual_distance_km : ℝ) :
  scale_cm = 0.4 → scale_km = 5.3 → map_distance_cm = 64 → actual_distance_km = (scale_km * map_distance_cm) / scale_cm →
  actual_distance_km = 848 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end map_distance_proof_l568_568470


namespace probability_three_books_in_common_l568_568803

open Nat

theorem probability_three_books_in_common :
  ∀ (total_books : ℕ) (books_to_select : ℕ) (common_books : ℕ),
  total_books = 12 → books_to_select = 6 → common_books = 3 →
  (choose total_books books_to_select * choose total_books books_to_select) ≠ 0 →
  ((choose total_books common_books * choose (total_books - common_books) (books_to_select - common_books) * choose (total_books - common_books) (books_to_select - common_books)).toRat /
  (choose total_books books_to_select * choose total_books books_to_select).toRat) = 112 / 617 := by
  sorry

end probability_three_books_in_common_l568_568803


namespace probability_lottery_sum_of_two_squares_l568_568163

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the function that counts the numbers between 1 and 90 that can be expressed as the sum of two squares
def count_sum_of_two_squares (max_val : ℕ) : ℕ := List.length ([1, 4, 9, 16, 25, 36, 49, 64, 81, 2, 5, 10, 17, 26, 37, 50, 65, 82, 8, 13, 20, 29, 40, 53, 68, 85, 18, 25, 34, 45, 58, 73, 90, 32, 41, 52, 65, 80, 50, 61, 74, 89, 72, 85].erase_dup.filter (≤ max_val))

-- Define the total number of possible lotto combinations
def total_combinations (n k : ℕ) : ℕ := binom n k

-- Define the probability calculation
def probability_of_sum_of_two_squares (n k : ℕ) (p : ℕ) : ℚ :=
  (binom p k : ℚ) / (binom n k : ℚ)

-- Prove that the probability is approximately 0.015
theorem probability_lottery_sum_of_two_squares : probability_of_sum_of_two_squares 90 5 40 = 0.015 := by
  sorry

end probability_lottery_sum_of_two_squares_l568_568163


namespace find_bounds_l568_568778

open Real

noncomputable def f (a b c : ℝ) : ℝ := 4 * (1 / a + 1 / b + 1 / c) - 1 / (a * b * c)

def M : Set (ℝ × ℝ × ℝ) := {p | 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.3 ∧ p.1 < 1 / 2 ∧ p.2 < 1 / 2 ∧ p.3 < 1 / 2 ∧ p.1 + p.2 + p.3 = 1}

theorem find_bounds : ∃ (α β : ℝ), (∀ (a b c : ℝ), (a, b, c) ∈ M → α ≤ f a b c ∧ f a b c ≤ β) ∧ α = 8 ∧ β = 9 :=
sorry

end find_bounds_l568_568778


namespace angle_GAC_eq_angle_EAC_l568_568755

open EuclideanGeometry

noncomputable def quadrilateral_bisect_angle {A B C D E F G : Point} : Prop :=
  is_quadrilateral A B C D ∧
  bisects_ac (angle A B C) A C ∧
  on_segment E C D ∧
  line_intersect B E A C F ∧
  line_intersect D F B C G

theorem angle_GAC_eq_angle_EAC {A B C D E F G : Point} (H : quadrilateral_bisect_angle A B C D E F G) :
  ∠ G A C = ∠ E A C :=
sorry

end angle_GAC_eq_angle_EAC_l568_568755


namespace overall_support_percentage_l568_568948

theorem overall_support_percentage (S_s S_t : ℕ) (P_s P_t : ℝ) 
  (hS_s : S_s = 200) (hS_t : S_t = 50) (hP_s : P_s = 0.7) (hP_t : P_t = 0.6) : 
  (0.7 * 200 + 0.6 * 50) / (200 + 50) * 100 = 68 := 
by
  rw [hS_s, hS_t, hP_s, hP_t]
  norm_num
  sorry

end overall_support_percentage_l568_568948


namespace reema_loan_time_l568_568069

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem reema_loan_time :
  ∃ T : ℝ, simple_interest 1800 5.93 T = 632 ∧ abs (T - 5.92) < 0.01 :=
by
  use 5.92
  apply and.intro
  {
    -- This would show the exact simple interest calculation using the given values
    sorry
  }
  {
    -- This shows that the computed T is approximately 5.92
    sorry
  }

end reema_loan_time_l568_568069


namespace compute_fraction_product_l568_568979

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l568_568979


namespace value_of_x2_y2_z2_l568_568790

variable (x y z : ℝ)

theorem value_of_x2_y2_z2 (h1 : x^2 + 3 * y = 4) 
                          (h2 : y^2 - 5 * z = 5) 
                          (h3 : z^2 - 7 * x = -8) : 
                          x^2 + y^2 + z^2 = 20.75 := 
by
  sorry

end value_of_x2_y2_z2_l568_568790


namespace max_value_expression_l568_568292

theorem max_value_expression (x : ℝ) : 
  ∃ m : ℝ, m = 1 / 37 ∧ ∀ x : ℝ, (x^6) / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) ≤ m :=
sorry

end max_value_expression_l568_568292


namespace norma_cards_count_l568_568467

variable (initial_cards : ℝ) (additional_cards : ℝ)
variable (total_cards : ℝ)

def norma_initial_cards : Prop := initial_cards = 88.0
def norma_additional_cards : Prop := additional_cards = 70.0
def norma_total_cards : Prop := total_cards = initial_cards + additional_cards

theorem norma_cards_count
  (h_initial: norma_initial_cards initial_cards) 
  (h_additional: norma_additional_cards additional_cards) :
  total_cards = 158.0 := 
by
  rw [norma_initial_cards, norma_additional_cards] at h_initial h_additional
  have : total_cards = initial_cards + additional_cards, from rfl
  sorry

end norma_cards_count_l568_568467


namespace area_of_quadrilateral_AXYD_l568_568036

open Real

noncomputable def area_quadrilateral_AXYD: ℝ :=
  let A := (0, 0)
  let B := (20, 0)
  let C := (20, 12)
  let D := (0, 12)
  let Z := (20, 30)
  let E := (6, 6)
  let X := (2.5, 0)
  let Y := (9.5, 12)
  let base1 := (B.1 - X.1)  -- Length from B to X
  let base2 := (Y.1 - A.1)  -- Length from D to Y
  let height := (C.2 - A.2) -- Height common for both bases
  (base1 + base2) * height / 2

theorem area_of_quadrilateral_AXYD : area_quadrilateral_AXYD = 72 :=
by
  sorry

end area_of_quadrilateral_AXYD_l568_568036


namespace find_xyz_l568_568878

theorem find_xyz (x y z : ℕ) (h : 4 * Real.sqrt (Real.cbrt 7 - Real.cbrt 6) = Real.cbrt x + Real.cbrt y - Real.cbrt z) : 
  x + y + z = 51 := 
sorry

end find_xyz_l568_568878


namespace ray_equation_and_distance_l568_568573

noncomputable def point := (-6, 7)

noncomputable def circle := (x^2 + y^2 - 8*x - 6*y + 21 = 0)

theorem ray_equation_and_distance (
  x y : ℝ
  point : (ℝ × ℝ) := (-6, 7)
  circle : ℝ := x^2 + y^2 - 8*x - 6*y + 21 = 0
) :
  ( ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧ 
    ((a = 3 ∧ b = 4 ∧ c = -10) ∨ (a = 4 ∧ b = 3 ∧ c = 3)) ) 
  ∧ 
  ( ∃ (d : ℝ), d = 14 ) := 
sorry

end ray_equation_and_distance_l568_568573


namespace coprime_composite_lcm_l568_568145

theorem coprime_composite_lcm (a b : ℕ) (ha : a > 1) (hb : b > 1) (hcoprime : Nat.gcd a b = 1) (hlcm : Nat.lcm a b = 120) : 
  Nat.gcd a b = 1 ∧ min a b = 8 := 
by 
  sorry

end coprime_composite_lcm_l568_568145


namespace smallest_nm_correct_second_smallest_nm_correct_l568_568325

noncomputable def a : ℕ := sorry
noncomputable def b : ℕ := sorry
noncomputable def m : ℕ := sorry
noncomputable def n_m : ℕ := sorry
noncomputable def F : ℕ → ℕ := sorry
noncomputable def L : ℕ → ℕ := sorry

def sequence (a b : ℕ) : ℕ × ℕ := sorry

def f (a b : ℕ) : ℕ :=
  let seq := sequence a b in
  sorry -- Smallest positive integer j such that seq.j.snd = 0

def g (n : ℕ) : ℕ :=
  max (f n) sorry -- max {f(n, k) | 1 ≤ k < n}

def smallest_nm (m : ℕ) : ℕ :=
  let n_m := sorry in
  n_m

def second_smallest_nm (m : ℕ) : ℕ :=
  let n_m := sorry in
  n_m

theorem smallest_nm_correct (m : ℕ) : smallest_nm m = F (m + 1) := sorry

theorem second_smallest_nm_correct (m : ℕ) : second_smallest_nm m = L (m + 1) := sorry

end smallest_nm_correct_second_smallest_nm_correct_l568_568325


namespace f_neg_2_eq_3_l568_568040

-- Define the piecewise function f
def f (a x : ℝ) : ℝ :=
  if x >= 0 
  then a^x 
  else Real.log (x^2 + a^2) / Real.log a

-- Given condition f(2) = 4 implies a = 2
lemma solve_for_a (a : ℝ) (h : f a 2 = 4) : a = 2 :=
begin
  sorry
end

-- Prove that f(-2) = 3 given f(2) = 4
theorem f_neg_2_eq_3 (a : ℝ) (h : f a 2 = 4) : f a (-2) = 3 :=
begin
  -- Using the lemma solve_for_a to obtain a = 2
  have ha : a = 2 := solve_for_a a h,
  -- Substituting a = 2 into f gives the desired function
  rw ha,
  -- Rewriting the function definition for f(-2)
  rw f,
  -- Since -2 < 0, we must use the second branch of f
  simp,
  -- Calculating the log
  sorry
end

end f_neg_2_eq_3_l568_568040


namespace lattice_right_triangles_incenter_origin_l568_568000

theorem lattice_right_triangles_incenter_origin :
  let I : ℤ × ℤ := (2015, 7 * 2015)
  let O : ℤ × ℤ := (0,0)
  in 
  ∃ n : ℕ, n = 54 ∧ ∀ (A B : ℤ × ℤ),
    A ≠ O ∧ B ≠ O ∧ 
    (∃ t1 t2 : ℤ, A = (4 * t1, 3 * t1) ∧ B = (-3 * t2, 4 * t2)) ∧
    let OA := (4 * t1)^2 + (3 * t1)^2
    let OB := (-3 * t2)^2 + (4 * t2)^2
    n = (1 + 1) * (2 + 1) * (2 + 1) * (2 + 1) := n = 54 :=
sorry

end lattice_right_triangles_incenter_origin_l568_568000


namespace commensurable_iff_rat_l568_568482

def commensurable (A B : ℝ) : Prop :=
  ∃ d : ℝ, ∃ m n : ℤ, A = m * d ∧ B = n * d

theorem commensurable_iff_rat (A B : ℝ) :
  commensurable A B ↔ ∃ (m n : ℤ) (h : n ≠ 0), A / B = m / n :=
by
  sorry

end commensurable_iff_rat_l568_568482


namespace p_plus_q_l568_568100

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

lemma p_linear : ∀ x, p x = x :=
sorry

lemma q_quadratic : ∀ x, q x = 2 * x * (x - 2) :=
sorry

lemma q_at_3 : q 3 = 6 :=
sorry

lemma p_at_4 : p 4 = 4 :=
sorry

lemma vertical_asymptote_at_2 : q 2 = 0 :=
sorry

lemma hole_at_0 : p 0 = 0 ∧ q 0 = 0 :=
sorry

theorem p_plus_q : ∀ x, p x + q x = 2 * x^2 - 3 * x :=
begin
  have p_def := p_linear,
  have q_def := q_quadratic,
  sorry
end

end p_plus_q_l568_568100


namespace octagon_area_sum_l568_568144

/--
Given two concentric squares centered at O,
  - the larger square has a side length of 2
  - the smaller square has a side length of 1
  - the octagon formed by the intersection points of the extended sides has a side length of 17/36.
  Prove that the area of the octagon is 17/9 and that the sum of m and n (where the area fraction is m/n in simplest form) is 26.
-/
theorem octagon_area_sum (O : Point) (s₁ s₂ : ℕ) (L : ℚ) 
  (hs₁ : s₁ = 2) (hs₂ : s₂ = 1) (hl : L = 17/36) :
  let area := 17 / 9 in
  let fraction := Rat.mk 17 9 in
  let m := fraction.num in
  let n := fraction.denom in
  m + n = 26 :=
by {
  sorry
}

end octagon_area_sum_l568_568144


namespace arithmetic_sequence_a1_value_l568_568337

   noncomputable def arithmetic_sequence_find_a1 (d a_30 : ℚ) (h_d : d = 3/4) (h_a_30 : a_30 = 63/4) : ℚ :=
   let a_1 := a_30 - 29 * d in
   a_1

   theorem arithmetic_sequence_a1_value :
     arithmetic_sequence_find_a1 (3/4) (63/4) (by norm_num) (by norm_num) = -14 := by
   sorry
   
end arithmetic_sequence_a1_value_l568_568337


namespace largest_satisfying_n_correct_l568_568276
noncomputable def largest_satisfying_n : ℕ := 4

theorem largest_satisfying_n_correct :
  ∀ n x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5) 
  → n = largest_satisfying_n ∧
  ¬ (∃ x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5 ∧ 5 < x^5 ∧ x^5 < 6)) := sorry

end largest_satisfying_n_correct_l568_568276


namespace average_of_integers_between_results_l568_568531

noncomputable def average_of_integers_between_fraction_bounds : ℚ :=
  let lower_bound : ℚ := 22 / 77
  let upper_bound : ℚ := 35 / 77
  let integers_within_bounds : List ℤ := List.range' 23 12  -- produces [23, 24, ..., 34]
  let sum_of_integers : ℤ := integers_within_bounds.sum
  (sum_of_integers : ℚ) / integers_within_bounds.length

theorem average_of_integers_between_results :
  average_of_integers_between_fraction_bounds = 28.5 :=
by
  simp only [average_of_integers_between_fraction_bounds, lower_bound, upper_bound, integers_within_bounds]
  norm_num
  sorry

end average_of_integers_between_results_l568_568531


namespace fraction_power_mult_correct_l568_568989

noncomputable def fraction_power_mult : Prop :=
  (\left(\frac{1}{3} \right)^{4}) * \left(\frac{1}{5} \right) = \left(\frac{1}{405} \right)

theorem fraction_power_mult_correct : fraction_power_mult :=
by
  -- The complete proof will be here.
  sorry

end fraction_power_mult_correct_l568_568989


namespace scientific_notation_rice_weight_l568_568140

/-- Each grain of rice weighs about 0.000035 kilograms. -/
def rice_grain_weight : ℝ := 0.000035

/-- The scientific notation of 0.000035 kilograms is 3.5 × 10⁻⁵. -/
theorem scientific_notation_rice_weight : rice_grain_weight = 3.5 * 10^(-5) :=
by
  sorry

end scientific_notation_rice_weight_l568_568140


namespace problem1_l568_568622

theorem problem1 : -1 ^ 2022 + (π - 2023) ^ 0 - (-1 / 2) ^ (-2) = -4 := 
by 
  sorry

end problem1_l568_568622


namespace problem_part1_problem_part2_l568_568349

/-- Definitions of the sequence structure -/
def sequence : ℕ → ℕ := λ n, match n with
  | 0     => 2^0
  | n + 1 => let k := nat.find (λ k, (k * (k + 1)) / 2 > n + 1) in 2^(n + 1 - k * (k - 1) / 2 - 1)
  end

/-- Definition for sum of the first N terms being a power of 2 -/
def sum_is_power_of_2 (N : ℕ) : Prop :=
  let sum := (1 to N).to_list.map (λ n, sequence (n-1)).sum in
    ∃ k : ℕ, 2^k = sum

/-- Problem statements -/
theorem problem_part1 : sequence 99 = 256 := by
  sorry

theorem problem_part2 : ∃ N : ℕ, N > 1000 ∧ sum_is_power_of_2 N ∧ (∀ M : ℕ, (M > 1000 ∧ sum_is_power_of_2 M) → N ≤ M) := by
  sorry

end problem_part1_problem_part2_l568_568349


namespace students_failed_l568_568877

theorem students_failed (Q : ℕ) (x : ℕ) (h1 : 4 * Q < 56) (h2 : x = Nat.lcm 3 (Nat.lcm 7 2)) (h3 : x < 56) :
  let R := x - (x / 3 + x / 7 + x / 2) 
  R = 1 := 
by
  sorry

end students_failed_l568_568877


namespace maximum_integer_solutions_l568_568209

-- Definitions
def skew_centered_polynomial (p : ℤ[X]) : Prop := 
  p.coeff 50 = -50 ∧ ∀ n, p.coeff n ∈ ℤ

-- Main statement
theorem maximum_integer_solutions (p : ℤ[X]) (h : skew_centered_polynomial p) : 
  ∃ n ≤ 7, ∀ x ∈ ℤ, (p x = x^2 → x ∈ {r : ℤ | r = r}) :=
sorry

end maximum_integer_solutions_l568_568209


namespace area_inside_arcs_outside_square_l568_568211

theorem area_inside_arcs_outside_square (r : ℝ) (θ : ℝ) (L : ℝ) (a b c d : ℝ) :
  r = 6 ∧ θ = 45 ∧ L = 12 ∧ a = 15 ∧ b = 0 ∧ c = 15 ∧ d = 144 →
  (a + b + c + d = 174) :=
by
  intros h
  sorry

end area_inside_arcs_outside_square_l568_568211


namespace sample_statistics_l568_568319

open Real

def sample := [10, 12, 9, 14, 13]

def sample_mean : ℝ :=
  (sample.sum : ℝ) / sample.length

def sample_variance (mean : ℝ) : ℝ :=
  (sample.map (λ x => (x - mean) ^ 2)).sum / sample.length

theorem sample_statistics :
  sample_mean = 11.6 ∧ sample_variance sample_mean = 3.44 := by
  sorry

end sample_statistics_l568_568319


namespace sum_of_arithmetic_sequence_9_terms_l568_568694

-- Define the odd function and its properties
variables {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = -f (x)) 
          (h2 : ∀ x y, x < y → f x < f y)

-- Define the shifted function g
noncomputable def g (x : ℝ) := f (x - 5)

-- Define the arithmetic sequence with non-zero common difference
variables {a : ℕ → ℝ} (d : ℝ) (h3 : d ≠ 0) 
          (h4 : ∀ n, a (n + 1) = a n + d)

-- Condition given by the problem
variable (h5 : g (a 1) + g (a 9) = 0)

-- Proof obligation
theorem sum_of_arithmetic_sequence_9_terms :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 45 :=
sorry

end sum_of_arithmetic_sequence_9_terms_l568_568694


namespace total_distance_dog_runs_l568_568171

-- Define the distance between Xiaoqiang's home and his grandmother's house in meters
def distance_home_to_grandma : ℕ := 1000

-- Define Xiaoqiang's walking speed in meters per minute
def xiaoqiang_speed : ℕ := 50

-- Define the dog's running speed in meters per minute
def dog_speed : ℕ := 200

-- Define the time Xiaoqiang takes to reach his grandmother's house
def xiaoqiang_time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the total distance the dog runs given the speeds and distances
theorem total_distance_dog_runs (d x_speed dog_speed : ℕ) 
  (hx : x_speed > 0) (hd : dog_speed > 0) : (d / x_speed) * dog_speed = 4000 :=
  sorry

end total_distance_dog_runs_l568_568171


namespace smallest_positive_period_of_f_interval_monotonically_increasing_set_of_x_values_for_f_ge_1_l568_568707

noncomputable def f (x : ℝ) : ℝ := cos(x)^4 - 2*sin(x)*cos(x) - sin(x)^4

theorem smallest_positive_period_of_f :
  ∃ p > 0, ∀ x, f (x + p) = f x := by
  use π
  sorry

theorem interval_monotonically_increasing :
  ∀ k : ℤ, ∀ x, -5*π/8 + k*π ≤ x ∧ x ≤ -π/8 + k*π → monotone_on f (Icc (-5*π/8 + k*π) (-π/8 + k*π)) := by
  sorry

theorem set_of_x_values_for_f_ge_1 :
  ∀ k : ℤ, ∀ x, -π/4 + k*π ≤ x ∧ x ≤ k*π → f(x) ≥ 1 := by
  sorry

end smallest_positive_period_of_f_interval_monotonically_increasing_set_of_x_values_for_f_ge_1_l568_568707


namespace train_stops_approx_857_minutes_per_hour_l568_568547

-- Define the speeds in km per hour
def speed_without_stoppage : ℝ := 42
def speed_with_stoppage : ℝ := 36

-- Define the reduced speed due to stoppage
def reduced_speed : ℝ := speed_without_stoppage - speed_with_stoppage

-- Calculating the time in hours the train stops
def time_stopped_in_hours : ℝ := reduced_speed / speed_without_stoppage

-- Convert time from hours to minutes
def time_stopped_in_minutes : ℝ := time_stopped_in_hours * 60

-- The theorem to prove that the train stops for approximately 8.57 minutes per hour
theorem train_stops_approx_857_minutes_per_hour : abs (time_stopped_in_minutes - 8.57) < 0.01 :=
by sorry

end train_stops_approx_857_minutes_per_hour_l568_568547


namespace visit_orders_l568_568906

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_permutations_cities (pohang busan geoncheon gimhae gyeongju : Type) : ℕ :=
  factorial 4

theorem visit_orders (pohang busan geoncheon gimhae gyeongju : Type) :
  num_permutations_cities pohang busan geoncheon gimhae gyeongju = 24 :=
by
  unfold num_permutations_cities
  norm_num
  sorry

end visit_orders_l568_568906


namespace rectangle_area_l568_568172

theorem rectangle_area
  (width : ℕ) (length : ℕ)
  (h1 : width = 7)
  (h2 : length = 4 * width) :
  length * width = 196 := by
  sorry

end rectangle_area_l568_568172


namespace train_crossing_time_l568_568557

noncomputable def speed_conversion (v_kmh : ℕ) : ℕ :=
  v_kmh * 1000 / 3600

theorem train_crossing_time :
  ∀ (L : ℕ) (v_kmh : ℕ),
    L = 400 →
    v_kmh = 144 →
    L / (speed_conversion v_kmh) = 10 :=
by
  intros L v_kmh hL hv
  rw [hL, hv]
  -- we need to show 400 / (144 * 1000 / 3600) = 10
  show 400 / speed_conversion 144 = 10
  -- evaluate speed_conversion 144
  have hs : speed_conversion 144 = 40 := by
    unfold speed_conversion
    norm_num
  rw [hs]
  -- finish the proof
  norm_num

end train_crossing_time_l568_568557


namespace negation_of_proposition_l568_568729

def quadrilateral (Q : Type) := 
  ∃ (a b c d : Q), 
  (diagonals_equal : ∀ x y, ∃ (mid : Q), x = y) ∧
  (diagonals_bisect : ∃ m : Q, eq_trans x = m) 

def is_parallelogram (Q : Type) := 
  ∃ (a b c d : Q), 
  (parallel_sides : ∀ s t, ∃ (parallel : Prop), s ∥ t) 

theorem negation_of_proposition :
  ¬ (∀ (Q : Type), 
     (∃ (a b c d : Q), 
      (diagonals_equal : ∀ x y, ∃ (mid : Q), x = y) ∧ 
      (diagonals_bisect : ∃ m : Q, eq_trans x = m) → 
      (is_parallelogram Q))) ↔
  ∃ (Q : Type), 
  (∃ (a b c d : Q), 
   (diagonals_equal : ∀ x y, ∃ (mid : Q), x = y) ∧ 
   (diagonals_bisect : ∃ m : Q, eq_trans x = m) ∧ 
   ¬ (is_parallelogram Q)).
:= sorry

end negation_of_proposition_l568_568729


namespace rhombus_side_length_l568_568743

noncomputable def functional_relationship (x : ℝ) : ℝ :=
  - (1 / 2) * x^2 + 35 * x

theorem rhombus_side_length (S : ℝ) (x : ℝ) 
  (h_sum_diagonals : x + (70 - x) = 70)
  (h_area : S = 600) :
  let d1 := x,
      d2 := 70 - x,
      side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  side = 25 := by
  sorry

end rhombus_side_length_l568_568743


namespace delivery_pattern_count_l568_568197

-- Define the concept of valid delivery patterns ensuring no four consecutive nondeliveries
def P : ℕ → ℕ
| 0     := 1 -- Basic case (considered as a single pattern)
| 1     := 2
| 2     := 4
| 3     := 8
| (n+4) := P n + P (n+1) + P (n+2) + P (n+3)

-- Proving the specific case of 12 offices
theorem delivery_pattern_count : P 12 = 927 :=
by {
  -- Fill initial base cases
  have base_cases : P 4 = 15 ∧ P 5 = 29 ∧ P 6 = 56 ∧ P 7 = 108 ∧ P 8 = 208 ∧ P 9 = 401 ∧ P 10 = 773 ∧ P 11 = 1491,
  {
    repeat {
      -- Checking computation correctness on base or intermediate cases
      sorry,
    }
  },
  -- Use definition to prove the final case
  calc
    P 12 = P 8 + P 9 + P 10 + P 11 : by simp [P]
    ... = 208 + 401 + 773 + 1491 : by rw base_cases
    ... = 2873 : by norm_num,
  rw base_cases
}

end delivery_pattern_count_l568_568197


namespace pq_square_eq_l568_568787

theorem pq_square_eq :
  let a := 2
  let b := -9
  let c := 7
  (p q : ℝ) (h_root1 : a * p ^ 2 + b * p + c = 0)
              (h_root2 : a * q ^ 2 + b * q + c = 0) :
  (p - q) ^ 2 = 6.25 :=
by
  sorry

end pq_square_eq_l568_568787


namespace integral_solution_l568_568971

noncomputable def integral_problem : ℝ :=
  ∫ x in 0..(sqrt 2 / 2), x^4 / (sqrt ((1 - x^2)^3))

theorem integral_solution :
  (∫ x in 0..(sqrt 2 / 2), x^4 / (sqrt ((1 - x^2)^3))) = (5/4) - (3*π/8) :=
by
  -- applying the conditions from our set problem
  let t := λ x, asin x
  sorry -- the proof steps go here

end integral_solution_l568_568971


namespace garden_percent_increase_l568_568195

def area (r : ℝ) : ℝ := real.pi * r^2

def percent_increase_in_area (d₁ d₂ : ℝ) : ℝ :=
  let r₁ := d₁ / 2
  let r₂ := d₂ / 2
  let A₁ := area r₁
  let A₂ := area r₂
  ((A₂ - A₁) / A₁) * 100

theorem garden_percent_increase (d₁ d₂ : ℝ) (h₁ : d₁ = 20) (h₂ : d₂ = 30) : 
  percent_increase_in_area d₁ d₂ = 125 :=
by
  rw [h₁, h₂]
  dsimp [percent_increase_in_area, area]
  norm_num
  rw [mul_div_cancel_left, mul_div_cancel_left] 
  {exact real.pi_ne_zero}
  sorry

end garden_percent_increase_l568_568195


namespace Faye_apps_left_l568_568639

theorem Faye_apps_left (total_apps gaming_apps utility_apps deleted_gaming_apps deleted_utility_apps remaining_apps : ℕ)
  (h1 : total_apps = 12) 
  (h2 : gaming_apps = 5) 
  (h3 : utility_apps = total_apps - gaming_apps) 
  (h4 : remaining_apps = total_apps - (deleted_gaming_apps + deleted_utility_apps))
  (h5 : deleted_gaming_apps = gaming_apps) 
  (h6 : deleted_utility_apps = 3) : 
  remaining_apps = 4 :=
by
  sorry

end Faye_apps_left_l568_568639


namespace pies_sold_in_week_l568_568579

def daily_pies := 8
def days_in_week := 7
def total_pies := 56

theorem pies_sold_in_week : daily_pies * days_in_week = total_pies :=
by
  sorry

end pies_sold_in_week_l568_568579


namespace probability_heads_given_heads_l568_568526

-- Definitions for fair coin flips and the stopping condition
noncomputable def fair_coin_prob (event : ℕ → Prop) : ℝ :=
  sorry -- Probability function for coin events (to be defined in proofs)

-- The main statement
theorem probability_heads_given_heads :
  let p : ℝ := 1 / 3 in
  ∃ p: ℝ, p = 1 / 3 ∧ fair_coin_prob (λ n, (n = 1 ∧ (coin_flip n = (TT)) ∧ ((coin_flip (n+1) = (HH) ∨ coin_flip (n+1) = (TH))) ∧ ¬has_heads_before n)) = p :=
sorry

end probability_heads_given_heads_l568_568526


namespace area_relationship_area_increase_l568_568388

-- Definitions based on given conditions
def upper_base_length (x : ℕ) := x
def lower_base_length : ℕ := 15
def height : ℕ := 8

-- Proving the relationship between area and upper base length
theorem area_relationship (x y : ℕ) (h1 : y = 4 * x + 60) : y = 4 * x + 60 :=
by sorry

-- Proving the increase in area when x increases by 1
theorem area_increase (x : ℕ) : 4 * (x + 1) + 60 - (4 * x + 60) = 4 :=
by simp [nat.add_sub_cancel]

end area_relationship_area_increase_l568_568388


namespace inequality_is_linear_l568_568538

theorem inequality_is_linear (k : ℝ) (h1 : (|k| - 1) = 1) (h2 : (k + 2) ≠ 0) : k = 2 :=
sorry

end inequality_is_linear_l568_568538


namespace solve_for_y_l568_568081

theorem solve_for_y : ∃ y : ℝ, (5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3)) ↔ y = 1000 := by
  sorry

end solve_for_y_l568_568081


namespace chloe_boxes_of_clothing_l568_568975

theorem chloe_boxes_of_clothing (total_clothing pieces_per_box : ℕ) (h1 : total_clothing = 32) (h2 : pieces_per_box = 2 + 6) :
  ∃ B : ℕ, B = total_clothing / pieces_per_box ∧ B = 4 :=
by
  -- Proof can be filled in here
   sorry

end chloe_boxes_of_clothing_l568_568975


namespace units_digit_of_quotient_l568_568620

theorem units_digit_of_quotient (n : ℕ) (h1 : n = 1987) : 
  (((4^n + 6^n) / 5) % 10) = 0 :=
by
  have pattern_4 : ∀ (k : ℕ), (4^k) % 10 = if k % 2 = 0 then 6 else 4 := sorry
  have pattern_6 : ∀ (k : ℕ), (6^k) % 10 = 6 := sorry
  have units_sum : (4^1987 % 10 + 6^1987 % 10) % 10 = 0 := sorry
  have multiple_of_5 : (4^1987 + 6^1987) % 5 = 0 := sorry
  sorry

end units_digit_of_quotient_l568_568620


namespace shaded_area_possible_values_l568_568055

variable (AB BC PQ SC : ℕ)

-- Conditions:
def dimensions_correct : Prop := AB * BC = 33 ∧ AB < 7 ∧ BC < 7
def length_constraint : Prop := PQ < SC

-- Theorem statement
theorem shaded_area_possible_values (h1 : dimensions_correct AB BC) (h2 : length_constraint PQ SC) :
  (AB = 3 ∧ BC = 11 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17) ∨
                      (33 - 2 * 3 - 1 * 6 = 21) ∨
                      (33 - 2 * 4 - 1 * 5 = 20))) ∨ 
  (AB = 11 ∧ BC = 3 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17))) :=
sorry

end shaded_area_possible_values_l568_568055


namespace derivative_at_zero_l568_568310

def f (n : ℕ) (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * ... * (x + n)

theorem derivative_at_zero (n : ℕ) : (deriv (f n) 0) = n! :=
sorry

end derivative_at_zero_l568_568310


namespace factorize_expr_l568_568266

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end factorize_expr_l568_568266


namespace num_other_adults_l568_568076

-- Define the variables and conditions
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9
def shonda_kids : ℕ := 2
def kids_friends : ℕ := 10
def num_participants : ℕ := (num_baskets * eggs_per_basket) / eggs_per_person

-- Prove the number of other adults at the Easter egg hunt
theorem num_other_adults : (num_participants - (shonda_kids + kids_friends + 1)) = 7 := by
  sorry

end num_other_adults_l568_568076


namespace quadrilateral_area_l568_568257

theorem quadrilateral_area :
  let a1 := 9  -- adjacent side length
  let a2 := 6  -- other adjacent side length
  let d := 20  -- diagonal
  let θ1 := 35  -- first angle in degrees
  let θ2 := 110  -- second angle in degrees
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  area_triangle1 + area_triangle2 = 108.006 := 
by
  let a1 := 9
  let a2 := 6
  let d := 20
  let θ1 := 35
  let θ2 := 110
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  show area_triangle1 + area_triangle2 = 108.006
  sorry

end quadrilateral_area_l568_568257


namespace area_FBEG_gt_area_MFE_l568_568958

noncomputable def acute_angled_triangle (A B C : Point) : Prop := sorry

noncomputable def bc_longest_side (A B C : Point) : Prop := sorry

noncomputable def intersection_points (A B C E G : Point) : Prop := sorry

noncomputable def circumscribed_circle_center (O A B E : Point) : Prop := sorry

noncomputable def perpendiculars (E M F A C B : Point) : Prop := sorry

theorem area_FBEG_gt_area_MFE
    (A B C E G O M F : Point)
    (h_acute_angled : acute_angled_triangle A B C)
    (h_bc_longest : bc_longest_side A B C)
    (h_intersections : intersection_points A B C E G)
    (h_circumcenter : circumscribed_circle_center O A B E)
    (h_perpendiculars : perpendiculars E M F A C B)
    : area (quadrilateral F B E G) > area (quadrilateral M F E) := 
begin
  sorry
end

end area_FBEG_gt_area_MFE_l568_568958


namespace median_of_sequence_l568_568427

theorem median_of_sequence : 
  let seq := List.join (List.map (λ n => List.replicate n n) (List.range (250 + 1)))
  let sorted_seq := seq.sort (· ≤ ·)
  (sorted_seq.nth ((List.length sorted_seq) / 2)).iget = 177 :=
by
  sorry

end median_of_sequence_l568_568427


namespace part1a_part1b_part2_part3_l568_568806

-- Definitions for the sequences in columns ①, ②, and ③
def col1 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1)
def col2 (n : ℕ) : ℤ := ((-1 : ℤ) ^ n * (2 * n - 1)) - 2
def col3 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1) * 3

-- Problem statements
theorem part1a : col1 10 = 19 :=
sorry

theorem part1b : col2 15 = -31 :=
sorry

theorem part2 : ¬ ∃ n : ℕ, col2 (n - 1) + col2 n + col2 (n + 1) = 1001 :=
sorry

theorem part3 : ∃ k : ℕ, col1 k + col2 k + col3 k = 599 ∧ k = 301 :=
sorry

end part1a_part1b_part2_part3_l568_568806


namespace train_length_approx_l568_568909

/-- Given the speed of the train in km/hr and the time to cross the pole in seconds, prove that
the length of the train is approximately 200 meters. -/
theorem train_length_approx (speed_kmph : ℕ) (time_s : ℕ)
  (h_speed : speed_kmph = 80) (h_time : time_s = 9) :
  (speed_kmph * (5 / 18) * time_s ≈ 200 : ℝ) :=
sorry

end train_length_approx_l568_568909


namespace triangle_area_l568_568762

theorem triangle_area (AB BC : ℕ) (cosB : ℚ) (h1 : AB = 5) (h2 : BC = 6) (h3 : cosB = 3/5) : 
  let sinB := Real.sqrt (1 - cosB^2) in
  let area := (1/2 : ℚ) * AB * BC * sinB in
  area = 12 := 
by
  sorry

end triangle_area_l568_568762


namespace find_interest_rate_l568_568071

theorem find_interest_rate (P1 P2 : ℝ) (r : ℝ) (total_amount : P1 + P2 = 1600)
  (interest_P1 : P1 = 1100) (interest_rate_P1 : 0.06)
  (total_interest : P1 * interest_rate_P1 + P2 * r = 85) : r = 0.038 :=
by
  sorry

end find_interest_rate_l568_568071


namespace factorization_result_l568_568846

theorem factorization_result (a b : ℤ) (h1 : 25 * x^2 - 160 * x - 336 = (5 * x + a) * (5 * x + b)) :
  a + 2 * b = 20 :=
by
  sorry

end factorization_result_l568_568846


namespace cubes_prob_rotated_identical_l568_568882

theorem cubes_prob_rotated_identical:
  let total_ways := 3^6,
      ways_all_one_color := 3,
      ways_one_other_five := 3 * (6 * 2 + 1),
      ways_two_colors_three_each := 3 * (choose 6 3) * 2,
      identical_ways :=  ways_all_one_color + ways_one_other_five + ways_two_colors_three_each,
      total_paintings := total_ways^3 in
  (identical_ways / total_paintings : ℚ) = 19 / 143 :=
by {
  let total_ways := 3^6,
  let ways_all_one_color := 3,
  let ways_one_other_five := 3 * (6 * 2 + 1),
  let ways_two_colors_three_each := 3 * (choose 6 3) * 2,
  let identical_ways :=  ways_all_one_color + ways_one_other_five + ways_two_colors_three_each,
  let total_paintings := total_ways^3,
  have identical_cases := (identical_ways : ℚ) / total_paintings,
  norm_num at identical_cases,
  exact sorry
}

end cubes_prob_rotated_identical_l568_568882


namespace solve_log_equation_l568_568083

theorem solve_log_equation (x : ℝ) (h : log 2 x - 3 * log 2 5 = -1) : x = 62.5 :=
sorry

end solve_log_equation_l568_568083


namespace train_speed_is_50_kmph_l568_568221

def length_of_train : ℕ := 360
def time_to_pass_bridge : ℕ := 36
def length_of_bridge : ℕ := 140

theorem train_speed_is_50_kmph :
  ((length_of_train + length_of_bridge) / time_to_pass_bridge) * 3.6 = 50 :=
by
  -- The proof will go here
  sorry

end train_speed_is_50_kmph_l568_568221


namespace add_pure_water_to_achieve_solution_l568_568718

theorem add_pure_water_to_achieve_solution
  (w : ℝ) (h_salt_content : 0.15 * 40 = 6) (h_new_concentration : 6 / (40 + w) = 0.1) :
  w = 20 :=
sorry

end add_pure_water_to_achieve_solution_l568_568718


namespace convert_1814_billion_to_scientific_notation_l568_568795

def billion := 10^9

def yuan_1814_billion := 1814 * billion

theorem convert_1814_billion_to_scientific_notation : 
  yuan_1814_billion = 1.814 * 10^12 :=
sorry

end convert_1814_billion_to_scientific_notation_l568_568795


namespace log_probability_is_one_sixth_l568_568302

noncomputable def log_is_integer_probability : Prop :=
  let nums : List ℕ := [2, 3, 8, 9]
  let pairs := nums.product nums
  let distinct_pairs := pairs.filter (λ p => p.fst ≠ p.snd)
  let count_total := List.length distinct_pairs
  let integer_log_pairs := distinct_pairs.filter (λ p => Int.log p.fst p.snd = Real.log p.fst p.snd)
  let count_integer := List.length integer_log_pairs
  count_integer / count_total = 1 / 6

-- Statement of the theorem
theorem log_probability_is_one_sixth : log_is_integer_probability := by sorry

end log_probability_is_one_sixth_l568_568302


namespace max_distance_on_curve_to_line_trajectory_of_moving_point_l568_568347

-- Problem 1: Prove maximum distance from any point M on curve C to line l is 2√2 + 1
theorem max_distance_on_curve_to_line (θ : ℝ) :
  let x := cos θ,
      y := sin θ,
      C := (x, y),
      l := {p : ℝ × ℝ | p.1 + p.2 - 4 = 0} in
  ∃ d_max : ℝ, d_max = 2 * sqrt 2 + 1 ∧ 
  ∀ M ∈ C, ∀ l_contains l M → (distance M l = d_max)
:= sorry

-- Problem 2: Prove trajectory of moving point Q is portion of the circle with center (1/8, 1/8) and radius √2/8 excluding the origin.
theorem trajectory_of_moving_point (θ : ℝ) (α : ℝ) :
  let P := (cos α, sin α),
      Q := (1/(cos α + sin α), α),
      radius := sqrt 2 / 8,
      center := (1/8, 1/8) in
  ∀ P ∈ l, ∀ Q ∈ ray OP, 
  |OP| * |OQ| = |OR|^2 →
  (center_x - Q_x)^2 + (center_y - Q_y)^2 = radius^2
:= sorry

end max_distance_on_curve_to_line_trajectory_of_moving_point_l568_568347


namespace route_A_is_quicker_l568_568466

-- Defining the conditions
def distance_A := 8  -- miles
def speed_A := 40    -- miles per hour
def distance_B_total := 7  -- miles
def construction_zone_distance := 1  -- mile
def speed_B_regular := 35  -- miles per hour
def speed_B_construction := 15  -- miles per hour

-- Calculating the time taken for Route A in minutes
def time_A := (distance_A / speed_A) * 60  -- = 12 minutes

-- Calculating the time taken for the non-construction part of Route B in minutes
def time_B1 := ((distance_B_total - construction_zone_distance) / speed_B_regular) * 60

-- Calculating the time taken for the construction part of Route B in minutes
def time_B2 := (construction_zone_distance / speed_B_construction) * 60

-- Total time for Route B
def time_B := time_B1 + time_B2  -- minutes

-- Calculate the time difference
def time_difference := time_B - time_A  -- = 2.29 minutes

-- Statement of the theorem
theorem route_A_is_quicker:
  time_difference = 2.29 := 
begin 
  sorry
end

end route_A_is_quicker_l568_568466


namespace range_of_a_l568_568698

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ x^2 - a * x + 1 < 0) → a > 2 :=
begin
  sorry
end

end range_of_a_l568_568698


namespace max_reciprocal_sum_eq_2_l568_568638

theorem max_reciprocal_sum_eq_2 (r1 r2 t q : ℝ) (h1 : r1 + r2 = t) (h2 : r1 * r2 = q)
  (h3 : ∀ n : ℕ, n > 0 → r1 + r2 = r1^n + r2^n) :
  1 / r1^2010 + 1 / r2^2010 = 2 :=
by
  sorry

end max_reciprocal_sum_eq_2_l568_568638


namespace cube_construction_possible_l568_568437

-- Define the block shape (4 unit cubes) using a shape identifier for clarity
structure Block :=
  (cells : List (Int × Int × Int))

-- Definition of the specific block given in the problem
def given_block : Block :=
  { cells := [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1)] }

-- The Lean statement for proving the problem
theorem cube_construction_possible (b : Block) (shape : b = given_block) :
  ∃ (cubes : List Block), length cubes = 16 ∧ (∀ cube ∈ cubes, cube = b) ∧ (∀ i j k, (i < 4 ∧ j < 4 ∧ k < 4) ↔ ∃ cube ∈ cubes, (i, j, k) ∈ cube.cells) :=
by
  sorry

end cube_construction_possible_l568_568437


namespace square_garden_perimeter_l568_568840

theorem square_garden_perimeter (q p : ℝ) (h : q = 2 * p + 20) : p = 40 :=
sorry

end square_garden_perimeter_l568_568840


namespace increasing_order_2011_l568_568900

theorem increasing_order_2011 :
  [Real.sqrt 2011, 2011, 2011^2].sorted (<) := by
  sorry

end increasing_order_2011_l568_568900


namespace units_painted_faces_half_total_l568_568015

noncomputable def expected_painted_faces (n : ℕ) : ℝ :=
  let P0 := 0
  let rec P (n : ℕ) : ℝ :=
    if n = 0 then P0
    else P (n-1) * (1566 / 1729) + 978
  P n

theorem units_painted_faces_half_total : ∃ n : ℕ, 
  real.abs (expected_painted_faces n - 5187) = real.abs (expected_painted_faces 7 - 5187) :=
begin
  trivial
end

end units_painted_faces_half_total_l568_568015


namespace triangle_third_side_l568_568750

noncomputable def c := sqrt (181 + 90 * Real.sqrt 3)

theorem triangle_third_side {a b : ℝ} (A : ℝ) (ha : a = 9) (hb : b = 10) (hA : A = 150) :
  c = sqrt (9^2 + 10^2 - 2 * 9 * 10 * Real.cos (A * Real.pi / 180)) := by
  rw [Real.cos_of_real (150 * Real.pi / 180)]
  -- Expecting this cosine computation is correct per original problem solution
  sorry

end triangle_third_side_l568_568750


namespace shift_right_linear_function_l568_568412

theorem shift_right_linear_function (x : ℝ) : 
  (∃ k b : ℝ, k ≠ 0 ∧ (∀ x : ℝ, y = -2x → y = kx + b) → (x, y) = (x - 3, -2(x-3))) → y = -2x + 6 :=
by
  sorry

end shift_right_linear_function_l568_568412


namespace base3_addition_correct_l568_568598

theorem base3_addition_correct :
  nat.addDigits 3 [2] + nat.addDigits 3 [1,2,1] + nat.addDigits 3 [1,2,1,2] + nat.addDigits 3 [1,2,1,2,1] = nat.addDigits 3 [2,1,1,1] :=
begin
  sorry
end

end base3_addition_correct_l568_568598


namespace find_complex_solutions_l568_568269

-- Define the given conditions
def z : Type := ℂ
def equation (z : ℂ) := z^2 = -45 - 28 * complex.I

-- State the problem as a theorem
theorem find_complex_solutions (z : ℂ) : 
  equation z ↔ (z = sqrt 7 - 2 * sqrt 7 * complex.I ∨ z = -sqrt 7 + 2 * sqrt 7 * complex.I) := 
sorry

end find_complex_solutions_l568_568269


namespace shaded_area_is_30_l568_568230

theorem shaded_area_is_30 (leg_length : ℕ) (num_small_triangles : ℕ) (num_shaded_triangles : ℕ)
  (h1 : leg_length = 10)
  (h2 : num_small_triangles = 25)
  (h3 : num_shaded_triangles = 15)
  (h_area_large : let area_large := 0.5 * (leg_length * leg_length) in area_large = 50)
  (h_area_small : let area_small := (0.5 * (leg_length * leg_length)) / num_small_triangles in area_small = 2) :
  let shaded_area := num_shaded_triangles * 2 in shaded_area = 30 :=
by {
  -- proof can be filled in here
  sorry
}

end shaded_area_is_30_l568_568230


namespace compute_fraction_product_l568_568978

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l568_568978


namespace range_of_a_l568_568335

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → abs (2 * a - 1) ≤ abs (x + 1 / x)) →
  -1 / 2 ≤ a ∧ a ≤ 3 / 2 :=
by sorry

end range_of_a_l568_568335


namespace exists_real_ge_3_l568_568904

-- Definition of the existential proposition
theorem exists_real_ge_3 : ∃ x : ℝ, x ≥ 3 :=
sorry

end exists_real_ge_3_l568_568904


namespace boats_eaten_percentage_l568_568048

-- Definitions for the problem:
def total_boats : ℕ := 30
def boats_shot : ℕ := 2
def boats_left : ℕ := 22
def boats_eaten := total_boats - boats_shot - boats_left
def percentage_eaten := (boats_eaten.to_real / total_boats.to_real) * 100

-- Statement to prove:
theorem boats_eaten_percentage : percentage_eaten = 20 := by
  sorry

end boats_eaten_percentage_l568_568048


namespace Linda_original_savings_l568_568464

-- Definition of the problem with all conditions provided.
theorem Linda_original_savings (S : ℝ) (TV_cost : ℝ) (TV_tax_rate : ℝ) (refrigerator_rate : ℝ) (furniture_discount_rate : ℝ) :
  let furniture_cost := (3 / 4) * S
  let TV_cost_with_tax := TV_cost + TV_cost * TV_tax_rate
  let refrigerator_cost := TV_cost + TV_cost * refrigerator_rate
  let remaining_savings := TV_cost_with_tax + refrigerator_cost
  let furniture_cost_after_discount := furniture_cost - furniture_cost * furniture_discount_rate
  (remaining_savings = (1 / 4) * S) →
  S = 1898.40 :=
by
  sorry


end Linda_original_savings_l568_568464


namespace missing_numbers_in_sequence_l568_568279

theorem missing_numbers_in_sequence :
  ∃ x y, (x = 25) ∧ (y = 13) ∧ 
    (λ (s : List ℕ), s = [1, 4, 3, 9, 5, 16, 7, x, 36, 11, y]) [1, 4, 3, 9, 5, 16, 7, 25, 36, 11, 13] := 
begin
  use 25,
  use 13,
  split,
  { refl },
  { split,
    { refl },
    { refl }
  }
end

end missing_numbers_in_sequence_l568_568279


namespace compute_fraction_power_mul_l568_568985

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end compute_fraction_power_mul_l568_568985


namespace Eve_age_l568_568597

theorem Eve_age (Adam_age : ℕ) (Eve_age : ℕ) (h1 : Adam_age = 9) (h2 : Eve_age = Adam_age + 5)
  (h3 : ∃ k : ℕ, Eve_age + 1 = k * (Adam_age - 4)) : Eve_age = 14 :=
sorry

end Eve_age_l568_568597


namespace functional_periodicity_l568_568029

noncomputable def is_periodic_with_period (f: ℝ → ℝ) (p: ℝ) : Prop :=
  ∀ x: ℝ, f(x) = f(x + p)

theorem functional_periodicity (f: ℝ → ℝ) (h: ∀ x: ℝ, f(x-1) + f(x+1) = real.sqrt 2 * f(x)) : 
  is_periodic_with_period f 8 :=
sorry

end functional_periodicity_l568_568029


namespace ratio_of_cone_volumes_l568_568625

noncomputable def volume_ratio_of_cones (r_C h_C r_D h_D : ℕ) : ℚ :=
  (1 / 3 * Real.pi * r_C ^ 2 * h_C) / (1 / 3 * Real.pi * r_D ^ 2 * h_D)

theorem ratio_of_cone_volumes :
  let r_C := 16
  let h_C := 42
  let r_D := 21
  let h_D := 16
  volume_ratio_of_cones r_C h_C r_D h_D = 224 / 147 :=
by
  sorry

end ratio_of_cone_volumes_l568_568625


namespace EquivalenceStatements_l568_568902

-- Define real numbers and sets P, Q
variables {x a b c : ℝ} {P Q : Set ℝ}

-- Prove the necessary equivalences
theorem EquivalenceStatements :
  ((x > 1) → (abs x > 1)) ∧ ((∃ x, x < -1) → (abs x > 1)) ∧
  ((a ∈ P ∩ Q) ↔ (a ∈ P ∧ a ∈ Q)) ∧
  (¬ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0)) ∧
  (x = 1 ↔ a + b + c = 0) :=
by
  sorry

end EquivalenceStatements_l568_568902


namespace sin_theta_plus_sqrt3_cos_theta_l568_568446

noncomputable def theta : ℝ := sorry

axiom theta_second_quadrant : π / 2 < θ ∧ θ < π

axiom tan_theta_pi_over_3 : Real.tan (θ + π / 3) = 1 / 2

theorem sin_theta_plus_sqrt3_cos_theta : 
  Real.sin θ + Real.sqrt 3 * Real.cos θ = -2 * Real.sqrt 5 / 5 :=
  sorry

end sin_theta_plus_sqrt3_cos_theta_l568_568446


namespace inequality_for_natural_n_l568_568479

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end inequality_for_natural_n_l568_568479


namespace volume_ratio_l568_568506

noncomputable def volume_of_sphere (r : ℝ) : ℝ := 
    (4 / 3 : ℝ) * Real.pi * r^3

theorem volume_ratio (a : ℝ) (h : a > 0) :
  let r_in := (1 / 2) * a in
  let r_out := (Real.sqrt 3 / 2) * a in
  volume_of_sphere r_out / volume_of_sphere r_in = 3 * Real.sqrt 3 := 
by
  let r_in := (1 / 2) * a
  let r_out := (Real.sqrt 3 / 2) * a
  have volume_in := volume_of_sphere r_in
  have volume_out := volume_of_sphere r_out
  /- Proof -/
  sorry

end volume_ratio_l568_568506


namespace place_value_diff_7669_l568_568895

theorem place_value_diff_7669 :
  let a := 6 * 10
  let b := 6 * 100
  b - a = 540 :=
by
  let a := 6 * 10
  let b := 6 * 100
  have h : b - a = 540 := by sorry
  exact h

end place_value_diff_7669_l568_568895


namespace coefficient_x3_in_expansion_l568_568493

theorem coefficient_x3_in_expansion : 
  ∀ (x : ℝ), 
  (∑ r in range.succ 6, (nat.choose 5 r) * (sqrt x)^(5 - r) * ((-2 * x)^r)) = 
  (sqrt x)^5 + 5 * (sqrt x)^4 * (-2 * x) + 10 * (sqrt x)^3 * ((-2 * x) ^ 2) +
  10 * (sqrt x)^2 * ((-2 * x) ^ 3) + 5 * (sqrt x) * ((-2 * x) ^ 4) + 
  ((-2 * x) ^ 5) := 
  ∀ x, sorry

end coefficient_x3_in_expansion_l568_568493


namespace total_volume_of_cubes_l568_568165

theorem total_volume_of_cubes (s : ℕ) (n : ℕ) (h_s : s = 5) (h_n : n = 4) : 
  n * s^3 = 500 :=
by
  sorry

end total_volume_of_cubes_l568_568165


namespace clock_hands_alignment_l568_568169

theorem clock_hands_alignment (h1 : ℕ := 62) (h2 : ℕ := 66) (h3 : ℕ := 76) :
  let d1 := 60 / (h2 - h1)  -- First and second clock alignment time
  let d2 := 60 / (h3 - h1)  -- First and third clock alignment time
in Nat.lcm d1 d2 = 30 :=
by
  -- Definition step
  let h1 := 62
  let h2 := 66
  let h3 := 76
  let d1 := 60 / (h2 - h1)  -- Time for first and second clocks to realign
  let d2 := 60 / (h3 - h1)  -- Time for first and third clocks to realign
  -- Stating the theorem
  show Nat.lcm d1 d2 = 30 from sorry

end clock_hands_alignment_l568_568169


namespace new_baking_soda_ratio_l568_568006

variables (sugar flour bakingSoda : ℕ) 

def initial_flour_ratio (sugar flour : ℕ) : Prop := sugar = flour
def initial_bakingSoda_ratio (flour bakingSoda : ℕ) : Prop := flour = 10 * bakingSoda 
def new_bakingSoda_amount (bakingSoda : ℕ) : ℕ := bakingSoda + 60 
def final_flour_ratio (flour newBakingSoda : ℕ) : Prop := flour / newBakingSoda = 8

axiom sugar_amount : sugar = 2400

theorem new_baking_soda_ratio :
  initial_flour_ratio sugar flour →
  initial_bakingSoda_ratio flour bakingSoda →
  new_bakingSoda_amount bakingSoda = newBakingSoda →
  final_flour_ratio flour newBakingSoda :=
begin
  sorry
end

end new_baking_soda_ratio_l568_568006


namespace butter_left_correct_l568_568796

-- Defining the initial amount of butter
def initial_butter : ℝ := 15

-- Defining the fractions used for different types of cookies
def fraction_chocolate_chip : ℝ := 2/5
def fraction_peanut_butter : ℝ := 1/6
def fraction_sugar : ℝ := 1/8
def fraction_oatmeal : ℝ := 1/4

-- Defining the amount of lost butter
def lost_butter : ℝ := 0.5

-- Calculating the total butter used
def butter_used : ℝ := (fraction_chocolate_chip * initial_butter) +
                        (fraction_peanut_butter * initial_butter) +
                        (fraction_sugar * initial_butter) +
                        (fraction_oatmeal * initial_butter)

-- Calculating the remaining butter before the loss
def butter_left_before_loss : ℝ := initial_butter - butter_used

-- Calculating the remaining butter after the loss
def butter_left : ℝ := butter_left_before_loss - lost_butter

-- Proving the final amount of butter left is 0.375 kg
theorem butter_left_correct : butter_left = 0.375 := by
  -- We add 'sorry' to skip the proof.
  sorry

end butter_left_correct_l568_568796


namespace find_fractional_sum_l568_568035

noncomputable def seq_a : ℕ → ℝ
| 0       := -3
| (n + 1) := seq_a n + seq_b n + 2 * real.sqrt (seq_a n ^ 2 + seq_b n ^ 2)
  
noncomputable def seq_b : ℕ → ℝ
| 0       := 2
| (n + 1) := seq_a n + seq_b n - 2 * real.sqrt (seq_a n ^ 2 + seq_b n ^ 2)

theorem find_fractional_sum :
  (1 / seq_a 2023 + 1 / seq_b 2023) = -1 / 6 :=
sorry

end find_fractional_sum_l568_568035


namespace value_of_y_l568_568240

theorem value_of_y (y : ℝ) (h : (45 / 75) = sqrt (3 * y / 75)) : y = 9 :=
sorry

end value_of_y_l568_568240


namespace horizontal_asymptote_of_f_l568_568630

open Filter Real

def f (x : ℝ) : ℝ := (7 * x^2 - 15) / (4 * x^2 + 7 * x + 3)

theorem horizontal_asymptote_of_f :
  tendsto f at_top (𝓝 (7 / 4)) :=
sorry

end horizontal_asymptote_of_f_l568_568630


namespace change_Xiaoli_should_get_back_l568_568170

theorem change_Xiaoli_should_get_back :
  let postage1 := 1.6
  let postage2 := 12.2
  let total_given := 15
  let total_postage := postage1 + postage2
  let change := total_given - total_postage
  change = 1.2 :=
by
  unfold postage1 postage2 total_given total_postage change
  have step1 : total_postage = 13.8 := by norm_num
  have step2 : change = total_given - total_postage := by rfl
  rw [←step2, step1]
  norm_num

end change_Xiaoli_should_get_back_l568_568170


namespace polynomial_degree_cancellation_l568_568626

theorem polynomial_degree_cancellation :
  let f := λ x : ℝ, 2 - 8 * x + 5 * x^2 - 3 * x^4
  let g := λ x : ℝ, 1 - x - 3 * x^2 + 4 * x^4
  let c := 3 / 4
  ∃ (h : polynomial ℝ), h.degree = 2 ∧
    (f + polynomial.C c * g) = h :=
by {
  sorry
}

end polynomial_degree_cancellation_l568_568626


namespace target_hit_probability_l568_568933

-- Defining the probabilities for A, B, and C hitting the target.
def P_A_hit := 1 / 2
def P_B_hit := 1 / 3
def P_C_hit := 1 / 4

-- Defining the probability that A, B, and C miss the target.
def P_A_miss := 1 - P_A_hit
def P_B_miss := 1 - P_B_hit
def P_C_miss := 1 - P_C_hit

-- Calculating the combined probability that none of them hit the target.
def P_none_hit := P_A_miss * P_B_miss * P_C_miss

-- Now, calculating the probability that at least one of them hits the target.
def P_hit := 1 - P_none_hit

-- Statement of the theorem.
theorem target_hit_probability : P_hit = 3 / 4 := by
  sorry

end target_hit_probability_l568_568933


namespace part1_part2_l568_568003

namespace VectorProblem

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def m := 5 / 9
def n := 8 / 9

def k := -16 / 13

-- Statement 1: Prove vectors satisfy the linear combination
theorem part1 : vector_a = (m * vector_b.1 + n * vector_c.1, m * vector_b.2 + n * vector_c.2) :=
by {
  sorry
}

-- Statement 2: Prove vectors are parallel
theorem part2 : (3 + 4 * k) * 2 + (2 + k) * 5 = 0 :=
by {
  sorry
}

end VectorProblem

end part1_part2_l568_568003


namespace line_eq_equiv_slope_intercept_find_slope_intercept_l568_568568

-- Define the given line equation in vector form
def line_eq (x y : ℝ) : Prop :=
  (⟨2, -1⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨3, -4⟩) = 0

-- Define the slope-intercept form of the line equation
def slope_intercept_eq (x y : ℝ) : Prop :=
  y = 2 * x - 10

-- Define the ordered pair (m, b)
def ordered_pair : ℝ × ℝ :=
  (2, -10)

-- The theorem that states that the given line equation is equivalent to the slope-intercept form
theorem line_eq_equiv_slope_intercept :
  ∀ x y : ℝ, line_eq x y ↔ slope_intercept_eq x y :=
sorry

-- The theorem that states the values of (m, b)
theorem find_slope_intercept :
  ordered_pair = (2, -10) :=
rfl

end line_eq_equiv_slope_intercept_find_slope_intercept_l568_568568


namespace profit_percentage_is_20_l568_568946

def wholesale_price : ℝ := 90
def retail_price : ℝ := 120
def discount_percentage : ℝ := 10

def discount_amount : ℝ := (discount_percentage / 100) * retail_price
def selling_price : ℝ := retail_price - discount_amount
def profit : ℝ := selling_price - wholesale_price
def profit_percentage : ℝ := (profit / wholesale_price) * 100

theorem profit_percentage_is_20 : profit_percentage = 20 := by
  sorry

end profit_percentage_is_20_l568_568946


namespace decomposition_l568_568025

noncomputable def R (P Q : Polynomial ℚ) : Polynomial ℚ := 
  P / Q

theorem decomposition (P Q : Polynomial ℚ)
  (h_coprime : P.coprime Q) :
  ∃ A : Polynomial ℚ, 
  ∃ (c : ℕ → ℕ → ℚ) (a : ℕ → ℚ), 
  ∀ i k : ℕ,
  R P Q = A + ∑ i k, (c i k) / (X - C a i)^k :=
sorry

end decomposition_l568_568025


namespace find_f_neg_2010_6_l568_568689

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_one (x : ℝ) : f (x + 1) + f x = 3

axiom f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = 2 - x

theorem find_f_neg_2010_6 : f (-2010.6) = 1.4 := by {
  sorry
}

end find_f_neg_2010_6_l568_568689


namespace DE_zero_l568_568020

variable (A B C D E : Type)
variable [AffineSpace ℝ A]
variable [HasDistance A ℝ]

-- Conditions
variable (h₁ : is_right_triangle A B C)
variable (h₂ : on_diameter_circle B C D AC)
variable (h₃ : on_diameter_circle A B E AC)
variable (h₄ : area_triangle A B C = 200)
variable (h₅ : dist A C = 40)

-- Statement to prove
theorem DE_zero : dist D E = 0 :=
sorry

end DE_zero_l568_568020


namespace conic_section_is_ellipse_l568_568541

-- Definitions based on the conditions given
def fixed_point1 : (ℝ × ℝ) := (0, 2)
def fixed_point2 : (ℝ × ℝ) := (6, -4)
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def conic_section_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 6)^2 + (y + 4)^2) = 14

-- Proof statement for Lean 4
theorem conic_section_is_ellipse : 
  (∀ (x y : ℝ), conic_section_equation x y → (conic_section_equation x y → choose_conic x y fixed_point1 fixed_point2 = "E")) :=
by
  sorry

-- Utility function used in the theorem definition
def choose_conic (x y : ℝ) (p1 p2 : ℝ × ℝ) : String :=
  if Real.sqrt ((x - p1.1)^2 + (y - p1.2)^2) + Real.sqrt ((x - p2.1)^2 + (y - p2.2)^2) = 14
  then "E"
  else "N"

end conic_section_is_ellipse_l568_568541


namespace g_g_g_9_equals_30_l568_568994

def g (x : ℝ) : ℝ :=
  if x < 5 then x^3 + 1 else x + 7

theorem g_g_g_9_equals_30 : g (g (g 9)) = 30 := by
  sorry

end g_g_g_9_equals_30_l568_568994


namespace problem_solution_l568_568609

def n_calculation (n : ℝ) : Prop := 3 * n + 26 = 50

def central_angle (n : ℝ) : ℝ := 360 * (2 * n / 50)

def find_median_group (n : ℝ) : char :=
  if n + 18 >= 25 then 'B' else 'other'

def average_situps (n : ℝ) : ℝ :=
  (15 * n + 26 * 18 + 34 * (2 * n) + 46 * 8) / 50

def passing_students (n : ℝ) : ℝ :=
  (18 + 2 * n + 8) / 50 * 700

theorem problem_solution : ∃ n : ℝ, n_calculation n ∧
                           central_angle n = 115.2 ∧
                           find_median_group n = 'B' ∧
                           average_situps n = 30 ∧
                           passing_students n = 588 := by
  sorry

end problem_solution_l568_568609


namespace non_periodic_sine_combination_l568_568818

theorem non_periodic_sine_combination (α : ℝ) (h_irrational : Irrational α) (h_pos : α > 0) :
  ¬(∃ T > 0, ∀ x, sin x + sin (α * x) = sin (x + T) + sin (α * (x + T))) :=
sorry

end non_periodic_sine_combination_l568_568818


namespace peaches_thrown_away_l568_568969

variables (total_peaches fresh_percentage peaches_left : ℕ) (thrown_away : ℕ)
variables (h1 : total_peaches = 250) (h2 : fresh_percentage = 60) (h3 : peaches_left = 135)

theorem peaches_thrown_away :
  thrown_away = (total_peaches * (fresh_percentage / 100)) - peaches_left :=
sorry

end peaches_thrown_away_l568_568969


namespace a_n_general_term_sum_b_n_l568_568712

noncomputable section

def a (n : ℕ) : ℕ := n.succ.recOn (2 : ℕ) (λ k a_k, 2 * a_k - k + 1)

def b (n : ℕ) : ℚ := 1 / (n * (a n - 2^(n-1) + 2))

def a_general_term (n : ℕ) : ℕ := n + 2^(n-1)

def S (n : ℕ) : ℚ := ∑ k in range (n+1), b k

def S_formula (n : ℕ) : ℚ := 3 / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))

theorem a_n_general_term (n : ℕ) : a n = a_general_term n := by
  sorry

theorem sum_b_n (n : ℕ) : S n = S_formula n := by
  sorry

end a_n_general_term_sum_b_n_l568_568712


namespace area_ratio_of_shapes_l568_568111

theorem area_ratio_of_shapes (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * π * r) (h2 : l = 3 * w) :
  (l * w) / (π * r^2) = (3 * π) / 16 :=
by sorry

end area_ratio_of_shapes_l568_568111


namespace tripod_new_height_l568_568951

-- Variables for initial and new lengths
def initial_length : ℝ := 6
def broken_length : ℝ := 4
def top_height_before_break : ℝ := 5

-- Goal height and floor value
def new_height : ℝ := 2 * Real.sqrt 5
def floor_value : ℝ := Real.floor (2 + Real.sqrt 5)

-- Lean theorem statement
theorem tripod_new_height (initial_length broken_length top_height_before_break : ℝ)
   (h_eq : new_height = 2 * Real.sqrt 5) :
   ∃ m n : ℕ, 
      new_height = m / Real.sqrt n ∧ 
      ¬ ∃ p : ℕ, p^2 ∣ n ∧ p > 1 ∧ 
      floor_value = 4 :=
begin
  -- Proof omitted
  sorry
end

end tripod_new_height_l568_568951


namespace least_addend_to_divisible_23_l568_568158

theorem least_addend_to_divisible_23 (a : ℕ) (d : ℕ) (k : ℕ) : 
  let b := 23 
  in a = 1054 ∧ b = 23 ∧ d = b - (a % b) ∧ k = a + d 
  → k % b = 0 ∧ d = 4 := 
by
  intros
  sorry

end least_addend_to_divisible_23_l568_568158


namespace helga_extra_hours_last_thursday_l568_568356

variable (A : Type)

-- Definitions for the given conditions
def articles_per_30_minutes := 5
def articles_per_hour := 2 * articles_per_30_minutes
def hours_per_day := 4
def days_per_week := 5
def normal_weekly_articles := articles_per_hour * hours_per_day * days_per_week
def extra_friday_hours := 3
def total_weekly_articles := 250
def extra_articles := total_weekly_articles - normal_weekly_articles
def extra_friday_articles := extra_friday_hours * articles_per_hour
def extra_thursday_articles := extra_articles - extra_friday_articles

-- The statement to be proven
def extra_thursday_hours := extra_thursday_articles / articles_per_hour

-- The theorem
theorem helga_extra_hours_last_thursday : extra_thursday_hours = 2 :=
begin
  sorry,
end

end helga_extra_hours_last_thursday_l568_568356


namespace sum_of_possible_a_l568_568126

theorem sum_of_possible_a:
  (∃ p q : ℤ, p + q = a ∧ p * q = 3 * a) → 
  (finset.sum (finset.filter (λ x, ∃ p q : ℤ, p + q = x ∧ p * q = 3 * x) 
    (finset.range 100)) = 30) :=
begin
  sorry
end

end sum_of_possible_a_l568_568126


namespace dodecagon_diagonals_l568_568583

theorem dodecagon_diagonals (n : ℕ) (h : n = 12) : (n * (n - 3)) / 2 = 54 :=
by
  rw [h]
  norm_num
  sorry

end dodecagon_diagonals_l568_568583


namespace melanie_missed_games_l568_568052

theorem melanie_missed_games (total_games : ℕ) (attended_games : ℕ) (h1 : total_games = 89) (h2 : attended_games = 47) : total_games - attended_games = 42 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end melanie_missed_games_l568_568052


namespace sum_of_replaced_numbers_l568_568807

theorem sum_of_replaced_numbers (a b c : ℝ) :
  let s := a + b + c in
  {a, b, c} = {a^2 + 2 * b * c, b^2 + 2 * c * a, c^2 + 2 * a * b} →
  (s = 0 ∨ s = 1) :=
by
  let s := a + b + c
  intro h
  -- Here, we would proceed to prove the theorem.
  -- The proof would involve showing that the sum of the numbers
  -- remains the same and using the properties of quadratic equations.
  sorry

end sum_of_replaced_numbers_l568_568807


namespace inequality_no_solution_l568_568831

theorem inequality_no_solution : 
  ∀ x : ℝ, -2 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 2 → false :=
by sorry

end inequality_no_solution_l568_568831


namespace gallons_left_l568_568046

theorem gallons_left (initial_gallons : ℚ) (gallons_given : ℚ) (gallons_left : ℚ) : 
  initial_gallons = 4 ∧ gallons_given = 16/3 → gallons_left = -4/3 :=
by
  sorry

end gallons_left_l568_568046


namespace min_unit_cubes_l568_568534

/-!
## Problem Description

Given a 3D figure where:
1. Each cube shares at least one face with another cube.
2. The front view depicts two columns with heights 3 and 2 units, respectively.
3. The side view depicts a depth of at least 3 units with varied heights.

Prove that the minimum number of unit cubes required to construct this figure is 11.
-/

theorem min_unit_cubes (front_view heights : List ℕ) (side_view depth : ℕ) (columns : ℕ)
  (condition1 : ∀ (i : ℕ), i < columns → front_view.nth i ≠ none)
  (condition2 : ∀ (i : ℕ), i < columns → (front_view.nth i = some heights i))
  (condition3 : ∀ (j : ℕ), j < depth → side_view.nth j ≠ none)
  (condition4 : ∀ (j : ℕ), j < depth → (side_view.nth j = some (height_at_depth j heights))) :
  (Σ i, front_view.nth i.get_or_else 0) + (Σ j, side_view.nth j.get_or_else 0) = 11 :=
by
  -- We state the conditions and goal without proving
  sorry

def height_at_depth (d : ℕ) (heights : List ℕ) := 
  if d < heights.length then heights.nth_le d (by assumption) else 0

end min_unit_cubes_l568_568534


namespace molecular_weight_N2O5_correct_l568_568238

noncomputable def atomic_weight_N : ℝ := 14.01
noncomputable def atomic_weight_O : ℝ := 16.00
def molecular_formula_N2O5 : (ℕ × ℕ) := (2, 5)

theorem molecular_weight_N2O5_correct :
  let weight := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  weight = 108.02 :=
by
  sorry

end molecular_weight_N2O5_correct_l568_568238


namespace tim_earnings_l568_568885

/-
  Tim's website got 100 visitors a day for the first 6 days and then on the last day 
  of the week it got twice as many visitors as every other day combined. If he gets 
  $0.01 per visit, prove that he made $18 that week.
-/

noncomputable def visitors_day (n : ℕ) : ℕ :=
  if n < 6 then 100 else 1200

def total_visitors (n : ℕ) : ℕ :=
  let first_6_days := List.sum (List.map visitors_day [0, 1, 2, 3, 4, 5])
  let last_day := visitors_day 6
  first_6_days + last_day

theorem tim_earnings :
    let weekly_earnings := total_visitors 7 * 0.01
    weekly_earnings = 18 :=
by
  sorry

end tim_earnings_l568_568885


namespace find_k_l568_568690

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def is_monotonous (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x < f y

theorem find_k (f : ℝ → ℝ) (h_odd : is_odd f) (h_monotonous : is_monotonous f) (h_zero : ∃! x, f (x ^ 2) + f (k - x) = 0) : k = 1 / 4 :=
begin
  sorry
end

end find_k_l568_568690


namespace original_fish_count_l568_568049

def initial_fish_count (fish_taken_out : ℕ) (current_fish : ℕ) : ℕ :=
  fish_taken_out + current_fish

theorem original_fish_count :
  initial_fish_count 16 3 = 19 :=
by
  sorry

end original_fish_count_l568_568049


namespace neg_sqrt_sq_eq_eleven_l568_568616

theorem neg_sqrt_sq_eq_eleven : (-real.sqrt 11) ^ 2 = 11 := by
  sorry

end neg_sqrt_sq_eq_eleven_l568_568616


namespace verify_cost_prices_l568_568950

noncomputable def cost_price_per_meter_for_first_consignment := 7682 / 92

def cost_price_per_meter_for_second_consignment (SP2 : ℝ) := (SP2 - 3600) / 120

def cost_price_per_meter_for_third_consignment (SP3 : ℝ) := (SP3 - 1500) / 75

theorem verify_cost_prices :
  cost_price_per_meter_for_first_consignment = 83.50 :=
by
  calc 
    cost_price_per_meter_for_first_consignment 
        = 7682 / 92 : rfl
    ... = 83.50     : by norm_num

end verify_cost_prices_l568_568950


namespace total_nails_sum_to_73_l568_568152

variables (Tickletoe Violet SillySocks : ℕ)

theorem total_nails_sum_to_73 (h1 : Violet = 2 * Tickletoe + 3)
                              (h2 : SillySocks = 3 * Tickletoe - 2)
                              (h3 : Violet + Tickletoe + SillySocks = 3 * 27 ∧ 4 * 27)          -- ratio condition
                              (h4 : Violet = 27) : 
                      Violet + Tickletoe + SillySocks = 73 := 
begin
  sorry
end

end total_nails_sum_to_73_l568_568152


namespace age_sum_l568_568907

theorem age_sum (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 10) : a + b + c = 27 := by
  sorry

end age_sum_l568_568907


namespace exists_good_filling_no_good_filling_2017_l568_568018

-- Defining the concept of a good filling for an n x n square.
def good_filling (n : ℕ) (M : finset ℕ) (square : list (list ℕ)) : Prop :=
  (∀ i : ℕ, i < n → ∃ s : finset ℕ, (s = (finset.range n).bUnion (λ j, (if i == j then {square i j} else ∅)) ∧ s = M))

-- Part (a): Prove that there exists n ≥ 3 for which a good filling exists.
theorem exists_good_filling : ∃ (n : ℕ) (square : list (list ℕ)), 3 ≤ n ∧ good_filling n (finset.range (2 * n - 1)) square :=
sorry

-- Part (b): Prove that for n = 2017 there is no good filling.
theorem no_good_filling_2017 : ∀ (square : list (list ℕ)), ¬ good_filling 2017 (finset.range (2 * 2017 - 1)) square :=
sorry

end exists_good_filling_no_good_filling_2017_l568_568018


namespace triangle_inequality_equality_condition_l568_568782

noncomputable def semiperimeter (A B C: ℝ) : ℝ := (A + B + C) / 2

theorem triangle_inequality
  (A B C M : Type*)
  [EuclideanPlane A B C M]
  (P: semiperimeter A B C)
  (AM BM CM : ℝ)
  (angleBMC angleAMC angleAMB : ℝ) :
  AM * Math.sin(angleBMC) + BM * Math.sin(angleAMC) + CM * Math.sin(angleAMB) 
  ≤ P :=
sorry

theorem equality_condition
  (A B C M : Type*)
  [EuclideanPlane A B C M]
  (P: semiperimeter A B C) 
  (M: incenter A B C) 
  (AM BM CM : ℝ)
  (angleBMC angleAMC angleAMB : ℝ) :
  AM * Math.sin(angleBMC) + BM * Math.sin(angleAMC) + CM * Math.sin(angleAMB) 
  = P :=
sorry

end triangle_inequality_equality_condition_l568_568782


namespace min_value_of_x_l568_568664

theorem min_value_of_x (x : ℝ) (h : min (min (sqrt x) (x^2)) x = 1 / 16) : 
    x = 1 / 4 :=
begin
  sorry
end

end min_value_of_x_l568_568664


namespace radius_H_sum_p_q_l568_568244

noncomputable def radius_circle_H {G H I : Type} [MetricSpace G] [MetricSpace H]
  [MetricSpace I] (rG : Float) (rH_4rI : Float) (tangent_internal : i → h → Prop)
  (tangent_external : i → h → Prop) (tangent_line : i → Line → Prop) : Prop :=
  ∃ s : Float,
  s > 0 ∧ ∃ rH : Float, rH = 4 * s ∧ (3 - 4 * s)² + s² = (3 - s)² - s² ∧
  rH = 2 * sqrt(117) - 18 ∧ 135 = 117 + 18

-- Theorem to be proved
theorem radius_H_sum_p_q : Prop :=
  ∃ rH : Float, ∃ p q : Int, 
  (rH = 2 * sqrt 117 - 18 ∧ p + q = 135)

-- Skipping the proof using sorry
proof radius_H_sum_p_q :=
  sorry

end radius_H_sum_p_q_l568_568244


namespace DK_parallel_BE_l568_568666

open Real EuclideanGeometry

variables {A B C D E F K M N : Point}
variables [incircle : Incircle A B C I]
variables [Midpoints : Midpoints D E M]
variables [Midpoints2 : Midpoints D F N]
variables [LineMN : Line (M, N)]
variables [IntersectionMNCA : Intersection (M, N) CA K]

theorem DK_parallel_BE 
  [IsBD : IsTangent I B D] [IsED : IsTangent I C E] [IsFD : IsTangent I A F]
  [IsIncircle : IncircleTriangle I A B C D E F]
  : Parallel DK BE :=
sorry

end DK_parallel_BE_l568_568666


namespace ordered_quadruples_sum_l568_568304

theorem ordered_quadruples_sum (a b c d : ℕ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < 500) :
  (a + d = b + c) → (b * c - a * d = 93) → ∃ n, n = 870 :=
by {
  intros,
  sorry
}

end ordered_quadruples_sum_l568_568304


namespace monotonically_increasing_interval_l568_568103

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

noncomputable def f (x: ℝ) : ℝ := real.logb 2 (x * |x|)

theorem monotonically_increasing_interval :
  is_monotonically_increasing f {x : ℝ | 0 < x} :=
by
  sorry

end monotonically_increasing_interval_l568_568103


namespace marble_ratio_l568_568884

theorem marble_ratio (A J C : ℕ) (h1 : 3 * (A + J + C) = 60) (h2 : A = 4) (h3 : C = 8) : A / J = 1 / 2 :=
by sorry

end marble_ratio_l568_568884


namespace dodecagon_diagonals_l568_568582

theorem dodecagon_diagonals (n : ℕ) (h : n = 12) : (n * (n - 3)) / 2 = 54 :=
by
  rw [h]
  norm_num
  sorry

end dodecagon_diagonals_l568_568582


namespace sum_simplify_1_sum_simplify_2_l568_568180

-- Problem 1: Simplify \sum_{k=0}^{n} (-1)^k C_{n}^{k} \cdot 2^{n-k}
theorem sum_simplify_1 (n : ℕ) (h : n ≠ 0) :
  ∑ k in finset.range (n+1), (-1)^k * nat.choose n k * 2^(n-k) = 1 :=
begin
  sorry
end

-- Problem 2: Simplify \sum_{k=0}^{n} (C_{n}^{k})^2
theorem sum_simplify_2 (n : ℕ) :
  ∑ k in finset.range (n+1), (nat.choose n k)^2 = nat.choose (2*n) n :=
begin
  sorry
end

end sum_simplify_1_sum_simplify_2_l568_568180


namespace distance_inequality_l568_568471

theorem distance_inequality (a : ℝ) (h : |a - 1| < 3) : -2 < a ∧ a < 4 :=
sorry

end distance_inequality_l568_568471


namespace population_net_change_l568_568872

theorem population_net_change :
  let initial_population := 100 -- assuming initial population as a normalized value for simplicity
  let final_population := initial_population * (6 / 5) * (9 / 10) * (13 / 10) * (17 / 20)
  let net_change := ((final_population - initial_population) / initial_population) * 100
  round net_change = 51 :=
by
  sorry

end population_net_change_l568_568872


namespace clock_starting_time_at_noon_l568_568849

theorem clock_starting_time_at_noon (degrees_moved : ℝ) (end_time: ℝ) (end_angle: ℝ) (rate_of_rotation: ℝ) : 
  degrees_moved = 75 ∧ end_time = 14.5 ∧ end_angle = 75 ∧ rate_of_rotation = 30 → 
  (∃ (start_time: ℝ), start_time = 12) :=
by
  intros,
  sorry

end clock_starting_time_at_noon_l568_568849


namespace combined_tax_rate_approx_l568_568441

def income_john := 56000
def tax_rate_john := 0.30
def income_ingrid := 74000
def tax_rate_ingrid := 0.40
def income_alice := 62000
def tax_rate_alice := 0.25
def income_ben := 80000
def tax_rate_ben := 0.35

def tax_john := tax_rate_john * income_john
def tax_ingrid := tax_rate_ingrid * income_ingrid
def tax_alice := tax_rate_alice * income_alice
def tax_ben := tax_rate_ben * income_ben

def total_tax := tax_john + tax_ingrid + tax_alice + tax_ben
def total_income := income_john + income_ingrid + income_alice + income_ben
def combined_tax_rate := (total_tax / total_income) * 100

theorem combined_tax_rate_approx : combined_tax_rate ≈ 33.42 := by
  sorry

end combined_tax_rate_approx_l568_568441


namespace verify_YX_l568_568785

def matrix_equality (X Y: Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  X + Y = X ⬝ Y

def given_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![25 / 4, 5 / 4], ![-10 / 4, 10 / 4]]

theorem verify_YX (X Y: Matrix (Fin 2) (Fin 2) ℚ)
  (h1: matrix_equality X Y)
  (h2: X ⬝ Y = given_matrix) :
  Y ⬝ X = given_matrix :=
sorry

end verify_YX_l568_568785


namespace complex_solutions_l568_568271

theorem complex_solutions (z : ℂ) : (z^2 = -45 - 28 * complex.I) ↔ (z = 2 - 7 * complex.I ∨ z = -2 + 7 * complex.I) := 
by 
  sorry

end complex_solutions_l568_568271


namespace length_of_fountain_built_by_20_men_in_6_days_l568_568183

noncomputable def work (workers : ℕ) (days : ℕ) : ℕ :=
  workers * days

theorem length_of_fountain_built_by_20_men_in_6_days :
  (work 35 3) / (work 20 6) * 49 = 56 :=
by
  sorry

end length_of_fountain_built_by_20_men_in_6_days_l568_568183


namespace polynomial_simplification_l568_568828

theorem polynomial_simplification (x : ℝ) : (3 * x^2 + 6 * x - 5) - (2 * x^2 + 4 * x - 8) = x^2 + 2 * x + 3 := 
by 
  sorry

end polynomial_simplification_l568_568828


namespace part_I_part_II_part_III_l568_568340

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + π / 6)

-- Given conditions
axiom A_gt_0 : 3 > 0
axiom ω_gt_0 : 2 > 0
axiom alpha_bound : -π / 2 < π / 6 ∧ π / 6 < π / 2
axiom period_pi : ∀ x, f x = f (x + π)
axiom max_value_at_pi_by_6 : f (π / 6) = 3

-- Prove that the analytical expression of f(x) is already defined as 3*sin(2x + π/6)
-- and its interval of increase
theorem part_I : 
  (∀ x, f(x) = 3 * Real.sin (2 * x + π / 6)) ∧ 
  (∀ k : ℤ, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → f x < f (x + π / 2))) :=
sorry

-- Prove the values of x0 given f(x0) = 3/2 and x0 in [0, 2pi)
theorem part_II (x0 : ℝ) (hx0 : 0 ≤ x0 ∧ x0 < 2 * π) (hf_x0 : f x0 = 3 / 2) : 
  x0 = 0 ∨ x0 = π ∨ x0 = π / 3 ∨ x0 = 4 * π / 3 :=
sorry

-- Prove the minimum value of m such that g(x) is an even function
def g (m : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (2 * (x - m) + π / 6)

theorem part_III (m : ℝ) (hm : m > 0) (heven: ∀ x, g m x = g m (-x)) :
  m = π / 3 :=
sorry

end part_I_part_II_part_III_l568_568340


namespace shift_right_three_units_l568_568414

theorem shift_right_three_units (x : ℝ) : (λ x, -2 * x) (x - 3) = -2 * x + 6 :=
by
  sorry

end shift_right_three_units_l568_568414


namespace cumulative_revenue_eq_l568_568856

-- Define the initial box office revenue and growth rate
def initial_revenue : ℝ := 3
def growth_rate (x : ℝ) : ℝ := x

-- Define the cumulative revenue equation after 3 days
def cumulative_revenue (x : ℝ) : ℝ :=
  initial_revenue + initial_revenue * (1 + growth_rate x) + initial_revenue * (1 + growth_rate x) ^ 2

-- State the theorem that proves the equation
theorem cumulative_revenue_eq (x : ℝ) :
  cumulative_revenue x = 10 :=
sorry

end cumulative_revenue_eq_l568_568856


namespace find_complex_solutions_l568_568268

-- Define the given conditions
def z : Type := ℂ
def equation (z : ℂ) := z^2 = -45 - 28 * complex.I

-- State the problem as a theorem
theorem find_complex_solutions (z : ℂ) : 
  equation z ↔ (z = sqrt 7 - 2 * sqrt 7 * complex.I ∨ z = -sqrt 7 + 2 * sqrt 7 * complex.I) := 
sorry

end find_complex_solutions_l568_568268


namespace proof_problem_l568_568845

-- Ellipse definition
def ellipse (x y : ℝ) := (x^2 / 5) + (3 * y^2 / 5) = 1

-- Line passing through point C(-1, 0) with slope k
def line (x k : ℝ) := k * (x + 1)

-- Intersection points A and B on the ellipse
def intersects (x1 y1 x2 y2 k : ℝ) (h1 : ellipse x1 y1) (h2 : ellipse x2 y2) : 
  Prop :=
  y1 = line x1 k ∧ y2 = line x2 k

-- Midpoint of A and B is (-1/2, n)
def midpoint (x1 y1 x2 y2 n : ℝ) := (x1 + x2) / 2 = -1 / 2

-- Fixed point M on the x-axis exists such that MA • MB is constant
def fixed_point (x0 λ x1 y1 x2 y2 k : ℝ) (h1 : ellipse x1 y1) (h2 : ellipse x2 y2)
  (h_mid : midpoint x1 y1 x2 y2 ((k * (-0.5) + y1) / 2)) : Prop :=
  let ma := (x1 - x0, y1)
  let mb := (x2 - x0, y2)
  (ma.1 * mb.1 + ma.2 * mb.2) = λ ∧ x0 = -7 / 3

theorem proof_problem (x1 y1 x2 y2 k n x0 λ : ℝ) (h1 : ellipse x1 y1) (h2 : ellipse x2 y2)
  (h_mid : midpoint x1 y1 x2 y2 n) :
  k = ℝ.sqrt 3 / 3 ∨ k = -ℝ.sqrt 3 / 3 ∧ fixed_point x0 λ x1 y1 x2 y2 k h1 h2 h_mid :=
by
  sorry

end proof_problem_l568_568845


namespace find_a_range_l568_568430

def polar_to_rectangular (ρ θ : ℝ) (a : ℝ) : Prop :=
  ρ = 2 * a * Real.sin θ ∧ ρ^2 = (2 * a * Real.sin θ)^2

def parametric_line (t : ℝ) : (ℝ × ℝ) :=
  let x := - ((Real.sqrt 2) / 2) * t - 1
  let y := ((Real.sqrt 2) / 2) * t
  (x, y)

def line_eq (t : ℝ) : Prop :=
  let (x, y) := parametric_line t
  x + y + 1 = 0

def distance_to_center (a : ℝ) : ℝ :=
  abs (a + 1) / Real.sqrt 2

def radius (a : ℝ) : ℝ :=
  2 * abs a

def valid_a (a : ℝ) : Prop :=
  (distance_to_center a) ≤ (radius a)

theorem find_a_range (a : ℝ) (h : ∃ t, line_eq t) :
  (valid_a a) ↔ (a ≤ (1 - 4 * Real.sqrt 2) / 7 ∨ a ≥ (1 + 4 * Real.sqrt 2) / 7) :=
sorry

end find_a_range_l568_568430


namespace werewolf_knight_is_A_l568_568514

structure Person :=
  (isKnight : Prop)
  (isLiar : Prop)
  (isWerewolf : Prop)

variables (A B C : Person)

-- A's statement: "At least one of us is a liar."
def statementA (A B C : Person) : Prop := A.isLiar ∨ B.isLiar ∨ C.isLiar

-- B's statement: "C is a knight."
def statementB (C : Person) : Prop := C.isKnight

theorem werewolf_knight_is_A (A B C : Person) 
  (hA : statementA A B C)
  (hB : statementB C)
  (hWerewolfKnight : ∃ x : Person, x.isWerewolf ∧ x.isKnight ∧ ¬ (A ≠ x ∧ B ≠ x ∧ C ≠ x))
  : A.isWerewolf ∧ A.isKnight :=
sorry

end werewolf_knight_is_A_l568_568514


namespace inequality_proof_l568_568033

theorem inequality_proof (p : ℝ) (x y z v : ℝ) (hp : p ≥ 2) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v ≥ 0) :
  (x + y) ^ p + (z + v) ^ p + (x + z) ^ p + (y + v) ^ p ≤ x ^ p + y ^ p + z ^ p + v ^ p + (x + y + z + v) ^ p := 
by sorry

end inequality_proof_l568_568033


namespace find_line_l_l568_568324

-- Define the matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]
def B : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![0, 1]]

-- Define transformations TA and TB
def TA (P : ℝ × ℝ) : ℝ × ℝ := (A ⬝ ![P.1, P.2]).toFun
def TB (P : ℝ × ℝ) : ℝ × ℝ := (B ⬝ ![P.1, P.2]).toFun

-- Define the equation of line l'
def line_l' (P' : ℝ × ℝ) : Prop := 2 * P'.1 + P'.2 - 2 = 0

-- Define the transformation sequence T_B(T_A(P))
def T_seq (P : ℝ × ℝ) : ℝ × ℝ := TB (TA P)

-- Define the equation of line l as given in the problem
def line_l (P : ℝ × ℝ) : Prop := P.1 + 5 * P.2 - 1 = 0

-- The theorem proving the required equivalence
theorem find_line_l : ∀ P : ℝ × ℝ, line_l' (T_seq P) → line_l P :=
by
  intro P H
  unfold T_seq at H
  simp [TA, TB, A, B] at H
  -- You can add more steps here if needed, or use sorry to complete the theorem
  sorry

end find_line_l_l568_568324


namespace convert_binary₁₀₁₀_to_decimal_l568_568253

def binary₁₀₁₀ : List ℕ := [1, 0, 1, 0]

def binary_to_decimal (bin : List ℕ) : ℕ :=
  bin.reverse.enum_from 0 |>.map (λ ⟨i, b⟩, b * 2^i) |> List.sum

theorem convert_binary₁₀₁₀_to_decimal : binary_to_decimal binary₁₀₁₀ = 10 := by
  sorry

end convert_binary₁₀₁₀_to_decimal_l568_568253


namespace fewer_pages_read_l568_568776

theorem fewer_pages_read (total_pages : ℕ) (pages_yesterday : ℕ) (pages_tomorrow : ℕ) (pages_today : ℕ) : 
  total_pages = 100 ∧ pages_yesterday = 35 ∧ pages_tomorrow = 35 ∧ pages_today = total_pages - pages_yesterday - pages_tomorrow → 
  pages_yesterday - pages_today = 5 :=
by 
  intros h
  cases h with ht hp
  cases hp with hy ht
  cases ht with ht hp
  rw [hy, ht, hp]
  convert rfl
  sorry -- Placeholder for calculations

end fewer_pages_read_l568_568776


namespace unit_vector_parallel_to_OA_l568_568433

variable (O : ℝ × ℝ × ℝ := (0, 0, 0))
variable (OA : ℝ × ℝ × ℝ := (-1, 2, 1))
variable (OB : ℝ × ℝ × ℝ := (-1, 2, -1))
variable (OC : ℝ × ℝ × ℝ := (2, 3, -1))

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem unit_vector_parallel_to_OA : 
  ∃ (u : ℝ × ℝ × ℝ), (u = (1 / magnitude OA) • OA ∨ u = -(1 / magnitude OA) • OA) :=
by
  sorry

end unit_vector_parallel_to_OA_l568_568433


namespace donald_oranges_l568_568634

-- Define the initial number of oranges
def initial_oranges : ℕ := 4

-- Define the number of additional oranges found
def additional_oranges : ℕ := 5

-- Define the total number of oranges as the sum of initial and additional oranges
def total_oranges : ℕ := initial_oranges + additional_oranges

-- Theorem stating that the total number of oranges is 9
theorem donald_oranges : total_oranges = 9 := by
    -- Proof not provided, so we put sorry to indicate that this is a place for the proof.
    sorry

end donald_oranges_l568_568634


namespace sum_three_digit_no_0_or_5_l568_568285

theorem sum_three_digit_no_0_or_5 :
  let valid_digits := {1, 2, 3, 4, 6, 7, 8, 9}
  let count_valid_numbers := 8 * 8 * 8
  let each_digit_frequency := count_valid_numbers / 8
  let sum_valid_digits := (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9)
  let sum_each_position := each_digit_frequency * sum_valid_digits
  let sum_hundreds := sum_each_position * 100
  let sum_tens := sum_each_position * 10
  let sum_units := sum_each_position * 1
  let total_sum := sum_hundreds + sum_tens + sum_units
  total_sum = 284160 := 
by
  let valid_digits := {1, 2, 3, 4, 6, 7, 8, 9}
  let count_valid_numbers := 8 * 8 * 8
  let each_digit_frequency := count_valid_numbers / 8
  let sum_valid_digits := (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9)
  let sum_each_position := each_digit_frequency * sum_valid_digits
  let sum_hundreds := sum_each_position * 100
  let sum_tens := sum_each_position * 10
  let sum_units := sum_each_position * 1
  let total_sum := sum_hundreds + sum_tens + sum_units
  have : total_sum = 284160 := by sorry
  exact this

end sum_three_digit_no_0_or_5_l568_568285


namespace geometric_sequence_correct_l568_568675

open nat

variable {α : Type*} [field α] [decidable_eq α]

noncomputable def a (n : ℕ) : α :=  
if n = 0 then 1 
else let q := (2 : α)^(1/2) in
$q ^ (n - 1)$ 

theorem geometric_sequence_correct:
  (a 3 = (2 : α)) ∧ (a 4 * a 6 = (16 : α)) → (a 10 - a 12) / (a 6 - a 8) = (4 : α) :=
by
  sorry

end geometric_sequence_correct_l568_568675


namespace track_meet_girls_with_short_hair_l568_568875

theorem track_meet_girls_with_short_hair (total_people : ℕ) (pct_boys : ℚ) (half_long_hair : ℚ) (third_medium_hair : ℚ) :
  total_people = 200 → pct_boys = 0.6 → half_long_hair = 1/2 → third_medium_hair = 1/3 → 
  let total_boys := (pct_boys * total_people).to_nat in
  let total_girls := total_people - total_boys in
  let long_hair_girls := (half_long_hair * total_girls).to_nat in
  let medium_hair_girls := (third_medium_hair * total_girls).to_nat in
  total_girls - long_hair_girls - medium_hair_girls = 13 :=
begin
  intros h_tot h_pct h_long h_third,
  let total_boys := (pct_boys * total_people).to_nat,
  let total_girls := total_people - total_boys,
  let long_hair_girls := (half_long_hair * total_girls).to_nat,
  let medium_hair_girls := (third_medium_hair * total_girls).to_nat,
  have h1 : total_boys = 120, by sorry,  -- This follows from the input values simple calculation
  have h2 : total_girls = 80, by sorry,  -- Again a simple calculation
  have h3 : long_hair_girls = 40, by sorry, -- Direct from multiplying half_long_hair and total_girls
  have h4 : medium_hair_girls = 27, by sorry, -- Given rounding rules,
  rw [h1, h2, h3, h4],
  norm_num,
end

end track_meet_girls_with_short_hair_l568_568875


namespace find_remainder_l568_568282

noncomputable def remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  (p /ₘ q).snd

theorem find_remainder :
  remainder (Polynomial.C 2 + Polynomial.X ^ 4) (Polynomial.X - Polynomial.C 2) ^ 2 = 32 * Polynomial.X - 46 :=
by
  sorry

end find_remainder_l568_568282


namespace red_car_speed_is_10mph_l568_568521

noncomputable def speed_of_red_car (speed_black : ℝ) (initial_distance : ℝ) (time_to_overtake : ℝ) : ℝ :=
  (speed_black * time_to_overtake - initial_distance) / time_to_overtake

theorem red_car_speed_is_10mph :
  ∀ (speed_black initial_distance time_to_overtake : ℝ),
  speed_black = 50 →
  initial_distance = 20 →
  time_to_overtake = 0.5 →
  speed_of_red_car speed_black initial_distance time_to_overtake = 10 :=
by
  intros speed_black initial_distance time_to_overtake hb hd ht
  rw [hb, hd, ht]
  norm_num
  sorry

end red_car_speed_is_10mph_l568_568521


namespace corrected_mean_l568_568854

open Real

theorem corrected_mean (n : ℕ) (mu_incorrect : ℝ)
                      (x1 y1 x2 y2 x3 y3 : ℝ)
                      (h1 : mu_incorrect = 41)
                      (h2 : n = 50)
                      (h3 : x1 = 48 ∧ y1 = 23)
                      (h4 : x2 = 36 ∧ y2 = 42)
                      (h5 : x3 = 55 ∧ y3 = 28) :
                      ((mu_incorrect * n + (x1 - y1) + (x2 - y2) + (x3 - y3)) / n = 41.92) :=
by
  sorry

end corrected_mean_l568_568854


namespace part1_and_odd_solve_inequality_l568_568344

def f (m x : ℝ) : ℝ := log m ((1 + x) / (1 - x))

theorem part1_and_odd (m : ℝ) (hpos : 0 < m) (hneq1 : m ≠ 1) (x : ℝ) (hx : x ∈ Ioo (-1 : ℝ) 1) :
  f m x = log m ((1 + x) / (1 - x)) ∧ ∀ x, f m (-x) = -f m x :=
sorry

theorem solve_inequality (m : ℝ) (hpos : 0 < m) (hneq1 : m ≠ 1) :
  (m > 1 ↔ ∀ x, f m x ≤ 0 → x ∈ set.Ioo (-1 : ℝ) 0) ∧
  (m < 1 ↔ ∀ x, f m x ≤ 0 → x ∈ set.Ico 0 1) :=
sorry

end part1_and_odd_solve_inequality_l568_568344


namespace completing_square_l568_568149

theorem completing_square (x : ℝ) (h : x^2 - 6 * x - 7 = 0) : (x - 3)^2 = 16 := 
sorry

end completing_square_l568_568149


namespace total_rainfall_hours_l568_568890

theorem total_rainfall_hours (r1 r2 : ℕ) (h1 : r1 = 30) (h2 : r2 = 15)
                             (h1_hours : 20) (total : 975) :
  ∃ T, T = h1_hours + (total - r1 * h1_hours) / r2 ∧ T = 45 :=
by {
  use 45,
  split,
  {
    sorry,  -- proving T = h1_hours + (total - r1 * h1_hours) / r2
  },
  {
    sorry   -- proving T = 45
  }
}

end total_rainfall_hours_l568_568890


namespace sum_of_roots_l568_568619

theorem sum_of_roots : 
  let f := (λ x : ℝ, (3*x + 4)*(x - 2) + (3*x + 4)*(x - 8)) in
  (∃ (r1 r2 : ℝ), f r1 = 0 ∧ f r2 = 0 ∧ r1 + r2 = 11/3) :=
sorry

end sum_of_roots_l568_568619


namespace dodecagon_diagonals_l568_568588

theorem dodecagon_diagonals :
  ∀ n : ℕ, n = 12 → (n * (n - 3)) / 2 = 54 :=
begin
  intros n hn,
  rw hn,
  norm_num,
end

end dodecagon_diagonals_l568_568588


namespace range_of_m_for_point_in_second_quadrant_l568_568757

theorem range_of_m_for_point_in_second_quadrant (m : ℝ) :
  (m - 3 < 0) ∧ (m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  -- The proof will be inserted here.
  sorry

end range_of_m_for_point_in_second_quadrant_l568_568757


namespace max_knights_of_grid_l568_568843

def is_knight (g : ℕ → ℕ → bool) (x y : ℕ) : Prop :=
if g x y = tt then True else False

def is_liar (g : ℕ → ℕ → bool) (x y : ℕ) : Prop :=
if g x y = ff then True else False

def neighbors (x y : ℕ) : list (ℕ × ℕ) :=
[(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def is_valid (x y : ℕ) : Prop :=
1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5

def count_knights_and_liars (g : ℕ → ℕ → bool) (x y : ℕ) : ℕ × ℕ :=
let
  N := filter (λ (xy : ℕ × ℕ), is_valid xy.1 xy.2) (neighbors x y),
  K := filter (λ (xy : ℕ × ℕ), g xy.1 xy.2 = tt) N
in (K.length, N.length - K.length)

def satisfies_condition (g : ℕ → ℕ → bool) (x y : ℕ) : Prop :=
let (k, l) := count_knights_and_liars g x y in k = l

def all_satisfy_condition (g : ℕ → ℕ → bool) : Prop :=
∀ x y : ℕ, is_valid x y → satisfies_condition g x y

theorem max_knights_of_grid :
  ∃ g : ℕ → ℕ → bool, all_satisfy_condition g ∧
  (∑ x in Finset.range 5, ∑ y in Finset.range 5, if g (x+1) (y+1) then 1 else 0) = 8 :=
by sorry

end max_knights_of_grid_l568_568843


namespace profit_percentage_l568_568944

/-- 
A retailer bought a machine at a wholesale price of $90 and later on sold it after a 10% discount 
of the retail price. The retailer made a profit equivalent to a certain percentage of the wholesale price. 
The retail price of the machine is $120. 
-/
theorem profit_percentage (wholesale_price retail_price : ℕ) (discount_percentage : ℝ) : 
  wholesale_price = 90 → 
  retail_price = 120 → 
  discount_percentage = 0.10 → 
  let discount := discount_percentage * retail_price in 
  let selling_price := retail_price - discount in 
  let profit := selling_price - wholesale_price in 
  ((profit / wholesale_price) * 100) = 20 :=
begin
  intros,
  sorry
end

end profit_percentage_l568_568944


namespace gain_percentage_is_five_percent_l568_568954

variables (CP SP New_SP Loss Loss_Percentage Gain Gain_Percentage : ℝ)
variables (H1 : Loss_Percentage = 10)
variables (H2 : CP = 933.33)
variables (H3 : Loss = (Loss_Percentage / 100) * CP)
variables (H4 : SP = CP - Loss)
variables (H5 : New_SP = SP + 140)
variables (H6 : Gain = New_SP - CP)
variables (H7 : Gain_Percentage = (Gain / CP) * 100)

theorem gain_percentage_is_five_percent :
  Gain_Percentage = 5 :=
by
  -- Proof goes here
  sorry

end gain_percentage_is_five_percent_l568_568954


namespace ratio_of_arithmetic_sequence_sums_l568_568239

-- Definitions of the arithmetic sequences based on the conditions
def numerator_seq (n : ℕ) : ℕ := 3 + (n - 1) * 3
def denominator_seq (m : ℕ) : ℕ := 4 + (m - 1) * 4

-- Definitions of the number of terms based on the conditions
def num_terms_num : ℕ := 32
def num_terms_den : ℕ := 16

-- Definitions of the sums based on the sequences
def sum_numerator_seq : ℕ := (num_terms_num / 2) * (3 + 96)
def sum_denominator_seq : ℕ := (num_terms_den / 2) * (4 + 64)

-- Calculate the ratio of the sums
def ratio_of_sums : ℚ := sum_numerator_seq / sum_denominator_seq

-- Proof statement
theorem ratio_of_arithmetic_sequence_sums : ratio_of_sums = 99 / 34 := by
  sorry

end ratio_of_arithmetic_sequence_sums_l568_568239


namespace minimum_value_of_log_l568_568392

noncomputable theory

open Real

-- Define the function y = a^x
def exp_func (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Define the function y = log_a x
def log_func (a : ℝ) (x : ℝ) : ℝ := log x / log a

-- Define the condition given in the problem
def condition (a : ℝ) : Prop := 1 + a^3 = 9/8

-- The main statement to be proved
theorem minimum_value_of_log (a x : ℝ) (h_condition : condition a) (h_pos : 0 < a) (h_interval : x ∈ Icc (1/4) 2) : 
  log_func a 2 = -1 :=
sorry

end minimum_value_of_log_l568_568392


namespace sum_distances_eq_radius_and_inradius_l568_568096

variables (R r d_a d_b d_c : ℝ) -- circumradius, inradius and distances
variables (a b c : ℝ) -- side lengths

-- Define the distances from the center of the circumscribed circle to the triangle sides.
variables [acute_triangle: ∀ (d_a d_b d_c R r : ℝ), 
  is_acute_angled_triangle(a, b, c) →
  is_circumradius(R, a, b, c) → 
  is_inradius(r, a, b, c) → 
  is_distance_to_side(d_a, O, BC) → 
  is_distance_to_side(d_b, O, CA) → 
  is_distance_to_side(d_c, O, AB)]

theorem sum_distances_eq_radius_and_inradius
  (h : acute_triangle a b c d_a d_b d_c R r) :
  d_a + d_b + d_c = R + r :=
sorry

end sum_distances_eq_radius_and_inradius_l568_568096


namespace shifted_function_is_correct_l568_568419

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end shifted_function_is_correct_l568_568419


namespace circle_center_radius_l568_568092

theorem circle_center_radius (x y : ℝ) :
  (x - 1)^2 + y^2 = 1 → ((1, 0), 1) :=
begin
  intro h,
  sorry
end

end circle_center_radius_l568_568092


namespace final_coordinates_l568_568504

open Matrix

noncomputable def rotate_z_90 (p : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![0, -1, 0; 1, 0, 0; 0, 0, 1] ⬝ p

noncomputable def reflect_xy (p : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![p 0, p 1, -p 2]

noncomputable def rotate_x_90 (p : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![1, 0, 0; 0, 0, -1; 0, 1, 0] ⬝ p

noncomputable def reflect_yz (p : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![-p 0, p 1, p 2]

theorem final_coordinates (initial_point : Fin 3 → ℝ) :
  initial_point = ![2, 2, 2] →
  let p1 := rotate_z_90 initial_point in
  let p2 := reflect_xy p1 in
  let p3 := rotate_x_90 p2 in
  let p4 := reflect_yz p3 in
  p4 = ![2, 2, 2] :=
by
  intro h
  -- Using matrix multiplication and transformations directly
  sorry

end final_coordinates_l568_568504


namespace problem1_problem2_l568_568389

-- Theorem 1: Given a^2 - b^2 = 1940:
theorem problem1 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1940 → 
  (a = 102 ∧ b = 92) := 
by 
  sorry

-- Theorem 2: Given a^2 - b^2 = 1920:
theorem problem2 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1920 → 
  (a = 101 ∧ b = 91) ∨ 
  (a = 58 ∧ b = 38) ∨ 
  (a = 47 ∧ b = 17) ∨ 
  (a = 44 ∧ b = 4) := 
by 
  sorry

end problem1_problem2_l568_568389


namespace type_II_patterns_l568_568527

noncomputable def h (m n : ℕ) : ℚ :=
if n % 2 = 1 then
  (1 / (2 * n)) * ∑ (d : ℕ) in Finset.filter (λ d, d ∣ n ∧ d ≠ n) (Finset.range n), Nat.totient d * ((m - 1) ^ (n / d) + (-1) ^ (n / d) * (m - 1))
else
  (1 / (2 * n)) * ∑ (d : ℕ) in Finset.filter (λ d, d ∣ n ∧ d ≠ n) (Finset.range n), Nat.totient d * ((m - 1) ^ (n / d) + (-1) ^ (n / d) * (m - 1)) + 
  (1 / 4) * m * (m - 1) ^ (n / 2)

theorem type_II_patterns (m n : ℕ) (hm : 2 ≤ m) (hn : 3 ≤ n) : 
  h(m, n) = if n % 2 = 1 then
              (1 / (2 * n)) * ∑ (d : ℕ) in Finset.filter (λ d, d ∣ n ∧ d ≠ n) (Finset.range n), Nat.totient d * ((m - 1) ^ (n / d) + (-1) ^ (n / d) * (m - 1))
            else
              (1 / (2 * n)) * ∑ (d : ℕ) in Finset.filter (λ d, d ∣ n ∧ d ≠ n) (Finset.range n), Nat.totient d * ((m - 1) ^ (n / d) + (-1) ^ (n / d) * (m - 1)) + 
              (1 / 4) * m * (m - 1) ^ (n / 2) :=
sorry

end type_II_patterns_l568_568527


namespace elias_total_spend_in_two_years_l568_568263

def price_soap (type : String) : Nat :=
  if type = "Lavender" then 4
  else if type = "Lemon" then 5
  else if type = "Sandalwood" then 6
  else 0

def discount (n : Nat) : Float :=
  if n >= 10 then 0.15
  else if n >= 7 then 0.10
  else if n >= 4 then 0.05
  else 0

def total_cost_of_soap (type : String) (n : Nat) : Float :=
  let original_cost := price_soap type * n
  if n > 7 then
    let discounted_bars := 7
    let full_price_bars := n - 7
    let discounted_cost := discounted_bars * price_soap type * (1 - discount discounted_bars)
    let full_price_cost := full_price_bars * price_soap type
    discounted_cost + full_price_cost
  else
    original_cost * (1 - discount n)

theorem elias_total_spend_in_two_years : 
  ((total_cost_of_soap "Lavender" 8) + (total_cost_of_soap "Lemon" 8) + (total_cost_of_soap "Sandalwood" 8)) = 109.50 :=
by {
  sorry
}

end elias_total_spend_in_two_years_l568_568263


namespace hours_week3_and_4_l568_568627

variable (H3 H4 : Nat)

def hours_worked_week1_and_2 : Nat := 35 + 35
def extra_hours_worked_week3_and_4 : Nat := 26
def total_hours_week3_and_4 : Nat := hours_worked_week1_and_2 + extra_hours_worked_week3_and_4

theorem hours_week3_and_4 :
  H3 + H4 = total_hours_week3_and_4 := by
sorry

end hours_week3_and_4_l568_568627


namespace quadratic_root_range_specific_m_value_l568_568677

theorem quadratic_root_range (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1^2 - 2 * (1 - m) * x1 + m^2 = 0 ∧ x2^2 - 2 * (1 - m) * x2 + m^2 = 0 ↔ m ≤ 1/2 :=
by
  sorry

theorem specific_m_value (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2 * (1 - m) * x1 + m^2 = 0)
  (h2 : x2^2 - 2 * (1 - m) * x2 + m^2 = 0) (h3 : x1^2 + 12 * m + x2^2 = 10) : 
  m = -3 :=
by
  sorry

end quadratic_root_range_specific_m_value_l568_568677


namespace num_integers_satisfying_inequality_l568_568182

theorem num_integers_satisfying_inequality:
  {x : ℤ | -10 ≤ 3 * x - 3 ∧ 3 * x - 3 ≤ 9}.to_finset.card = 7 :=
by
  sorry

end num_integers_satisfying_inequality_l568_568182


namespace circle_tangent_line_radius_l568_568710

-- Definition of the distance from a point to a line.
def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / sqrt (a ^ 2 + b ^ 2)

theorem circle_tangent_line_radius :
  ∃ r : ℝ, r > 0 ∧
    (∀ x y : ℝ, (x - 4) ^ 2 + y ^ 2 = r ^ 2) ∧ 
    (∀ x y : ℝ, sqrt 3 * x - 2 * y = 0) →
    r = 4 * sqrt 21 / 7 :=
by
  sorry

end circle_tangent_line_radius_l568_568710


namespace probability_of_yellow_on_second_draw_l568_568873

-- Definitions based on conditions
def total_balls : ℕ := 10
def yellow_balls : ℕ := 6
def white_balls : ℕ := 4

-- Probability of drawing a yellow ball on the second draw without replacement
theorem probability_of_yellow_on_second_draw
  (total_balls = 10)
  (yellow_balls = 6)
  (white_balls = 4)
  : (white_balls / ℕ.toReal total_balls) * (yellow_balls / (ℕ.toReal (total_balls - 1))) = 4 / 15 := 
  sorry

end probability_of_yellow_on_second_draw_l568_568873


namespace largest_possible_percent_error_l568_568400

theorem largest_possible_percent_error
  (C : ℝ)
  (error_percent : ℝ)
  (actual_circumference : C = 30)
  (measurement_error : error_percent = 0.15) :
  let actual_area := (30 / (2 * Real.pi)) ^ 2 * Real.pi in
  let max_possible_circumference := 30 + 30 * error_percent in
  let min_possible_circumference := 30 - 30 * error_percent in
  let min_possible_diameter := min_possible_circumference / Real.pi in
  let max_possible_diameter := max_possible_circumference / Real.pi in
  let min_possible_area := (min_possible_diameter / 2) ^ 2 * Real.pi in
  let max_possible_area := (max_possible_diameter / 2) ^ 2 * Real.pi in
  let percent_error_lower := (actual_area - min_possible_area) / actual_area * 100 in
  let percent_error_upper := (max_possible_area - actual_area) / actual_area * 100 in
  percent_error_upper = 32.25 := 
sorry

end largest_possible_percent_error_l568_568400


namespace line_properties_l568_568995

theorem line_properties : 
  ∃ (m b : ℝ), 
  (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 7) → y = m * x + b) ∧
  m + b = 3 ∧
  (∀ x : ℝ, ∀ y : ℝ, (x = 0 ∧ y = 1) → y = m * x + b) :=
sorry

end line_properties_l568_568995


namespace verify_shifted_function_l568_568407

def linear_function_shift_3_units_right (k b : ℝ) (hk : k ≠ 0) : Prop :=
  ∀ (x : ℝ), (k = -2) → (b = 6) → (λ x, -2 * (x - 3) + 6) = (λ x, k * x + b)

theorem verify_shifted_function : 
  linear_function_shift_3_units_right (-2) 6 (by norm_num) :=
sorry

end verify_shifted_function_l568_568407


namespace verify_shifted_function_l568_568404

def linear_function_shift_3_units_right (k b : ℝ) (hk : k ≠ 0) : Prop :=
  ∀ (x : ℝ), (k = -2) → (b = 6) → (λ x, -2 * (x - 3) + 6) = (λ x, k * x + b)

theorem verify_shifted_function : 
  linear_function_shift_3_units_right (-2) 6 (by norm_num) :=
sorry

end verify_shifted_function_l568_568404


namespace part_a_part_b_l568_568395

-- Define the problem conditions and questions
noncomputable def digit_arrangement_exists : Prop :=
  ∃ arrangement : (Fin 10) → (Fin 10) → Fin 10,
    (∀ i, ∀ j, ∑ k, (if arrangement i j = k then 1 else 0) = 1) ∧
    (∀ k, ∑ i, ∑ j, (if arrangement i j = k then 1 else 0) = 10) ∧
    (∀ i, ∃ d : Finset (Fin 10), d.card ≤ 4 ∧
      ∀ j, ∃ k ∈ d, arrangement i j = k) ∧
    (∀ j, ∃ d : Finset (Fin 10), d.card ≤ 4 ∧
      ∀ i, ∃ k ∈ d, arrangement i j = k)
  
theorem part_a : digit_arrangement_exists :=
sorry

theorem part_b : 
  ∀ arrangement : (Fin 10) → (Fin 10) → Fin 10, 
    (∀ k, ∑ i, ∑ j, (if arrangement i j = k then 1 else 0) = 10) →
    ¬(∀ i, ∃ d : Finset (Fin 10), d.card ≤ 3 ∧ ∀ j, ∃ k ∈ d, arrangement i j = k) →
    ∃ i, ∃ d : Finset (Fin 10), d.card ≥ 4 ∧ ( ∀ j, ∃ k ∈ d, arrangement i j = k) :=
begin
  sorry
end

end part_a_part_b_l568_568395


namespace possible_first_terms_l568_568513

noncomputable def sequence (a b : ℕ) : ℕ → ℕ
| 0 => a
| 1 => b
| n + 2 => sequence a b n + sequence a b (n + 1)

theorem possible_first_terms :
  ∃ a : ℕ, (a = 1 ∨ a = 5) ∧ 
           (∀ b : ℕ, sequence a b 2 = 7) ∧ 
           (∀ b : ℕ, (sequence a b 2013) % 4 = 1) :=
sorry

end possible_first_terms_l568_568513


namespace base_four_odd_last_digit_l568_568295

theorem base_four_odd_last_digit :
  ∃ b : ℕ, b = 4 ∧ (b^4 ≤ 625 ∧ 625 < b^5) ∧ (625 % b % 2 = 1) :=
by
  sorry

end base_four_odd_last_digit_l568_568295


namespace rectangle_y_value_l568_568942

theorem rectangle_y_value (y : ℝ) (h1 : -2 < 6) (h2 : y > 2) 
    (h3 : 8 * (y - 2) = 64) : y = 10 :=
by
  sorry

end rectangle_y_value_l568_568942


namespace milk_owed_l568_568044

theorem milk_owed (initial_milk : ℚ) (given_milk : ℚ) (h_initial : initial_milk = 4) (h_given : given_milk = 16 / 3) :
  initial_milk - given_milk = -4 / 3 :=
by {
  rw [h_initial, h_given],
  norm_num,
}

end milk_owed_l568_568044


namespace value_large_cube_l568_568207

-- Definitions based on conditions
def volume_small := 1 -- volume of one-inch cube in cubic inches
def volume_large := 64 -- volume of four-inch cube in cubic inches
def value_small : ℝ := 1000 -- value of one-inch cube of gold in dollars
def proportion (x y : ℝ) : Prop := y = 64 * x -- proportionality condition

-- Prove that the value of the four-inch cube of gold is $64000
theorem value_large_cube : proportion value_small 64000 := by
  -- Proof skipped
  sorry

end value_large_cube_l568_568207


namespace pyramid_surface_area_l568_568212

-- Definitions for the conditions
structure Rectangle where
  length : ℝ
  width : ℝ

structure Pyramid where
  base : Rectangle
  height : ℝ

-- Create instances representing the given conditions
noncomputable def givenRectangle : Rectangle := {
  length := 8,
  width := 6
}

noncomputable def givenPyramid : Pyramid := {
  base := givenRectangle,
  height := 15
}

-- Statement to prove the surface area of the pyramid
theorem pyramid_surface_area
  (rect: Rectangle)
  (length := rect.length)
  (width := rect.width)
  (height: ℝ)
  (hy1: length = 8)
  (hy2: width = 6)
  (hy3: height = 15) :
  let base_area := length * width
  let slant_height := Real.sqrt (height^2 + (length / 2)^2)
  let lateral_area := 2 * ((length * slant_height) / 2 + (width * slant_height) / 2)
  let total_surface_area := base_area + lateral_area 
  total_surface_area = 48 + 7 * Real.sqrt 241 := 
  sorry

end pyramid_surface_area_l568_568212


namespace B_investment_l568_568186

theorem B_investment (A : ℝ) (t_B : ℝ) (profit_ratio : ℝ) (B_investment_result : ℝ) : 
  A = 27000 → t_B = 4.5 → profit_ratio = 2 → B_investment_result = 36000 :=
by
  intro hA htB hpR
  sorry

end B_investment_l568_568186


namespace smallest_positive_q_with_property_l568_568631

theorem smallest_positive_q_with_property :
  ∃ q : ℕ, (
    q > 0 ∧
    ∀ m : ℕ, (1 ≤ m ∧ m ≤ 1006) →
    ∃ n : ℤ, 
      (m * q : ℤ) / 1007 < n ∧
      (m + 1) * q / 1008 > n) ∧
   q = 2015 := 
sorry

end smallest_positive_q_with_property_l568_568631


namespace probability_at_least_one_first_class_part_l568_568131

-- Define the problem constants
def total_parts : ℕ := 6
def first_class_parts : ℕ := 4
def second_class_parts : ℕ := 2
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the target probability
def target_probability : ℚ := 14 / 15

-- Statement of the problem as a Lean theorem
theorem probability_at_least_one_first_class_part :
  (1 - (choose second_class_parts 2 : ℚ) / (choose total_parts 2 : ℚ)) = target_probability :=
by
  -- the proof is omitted
  sorry

end probability_at_least_one_first_class_part_l568_568131


namespace sum_distances_equilateral_triangle_l568_568866

theorem sum_distances_equilateral_triangle (p : ℝ) (X : Type) [point_in_triangle : X] 
  (s : ℝ) : (∃ (ABC : Type) [eq_triangle : ABC], 
  (∀ (x : X), ∃ (d : ℝ), sum_of_distances_to_sides x d = s)) → s * Real.sqrt 12 = p :=
by
  sorry

end sum_distances_equilateral_triangle_l568_568866


namespace number_of_solutions_l568_568834

theorem number_of_solutions (h₁ : ∀ x, 50 * x % 100 = 0 → (x % 2 = 0)) 
                            (h₂ : ∀ x, (x % 2 = 0) → (∀ k, 1 ≤ k ∧ k ≤ 49 → (k * x % 100 ≠ 0)))
                            (h₃ : ∀ x, 1 ≤ x ∧ x ≤ 100) : 
  ∃ count, count = 20 := 
by {
  -- Here, we usually would provide a method to count all valid x values meeting the conditions,
  -- but we skip the proof as instructed.
  sorry
}

end number_of_solutions_l568_568834


namespace beam_count_represents_number_of_beams_l568_568605

def price := 6210
def transport_cost_per_beam := 3
def beam_condition (x : ℕ) : Prop := 
  transport_cost_per_beam * x * (x - 1) = price

theorem beam_count_represents_number_of_beams (x : ℕ) :
  beam_condition x → (∃ n : ℕ, x = n) := 
sorry

end beam_count_represents_number_of_beams_l568_568605


namespace mr_wang_withdrawal_l568_568468

theorem mr_wang_withdrawal (m : ℝ) (a : ℝ) (h1 : m > 0) (h2 : a > 0) : 
  let withdraw_amount := m * (1 + a)^5 in
  withdraw_amount = m * (1 + a)^5 :=
by 
  sorry

end mr_wang_withdrawal_l568_568468


namespace roots_quadratic_sum_of_squares_l568_568312

theorem roots_quadratic_sum_of_squares :
  ∀ x1 x2 : ℝ, (x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0) → x1^2 + x2^2 = 6 :=
by
  intros x1 x2 h
  -- proof goes here
  sorry

end roots_quadratic_sum_of_squares_l568_568312


namespace problem_solution_count_l568_568278

theorem problem_solution_count (θ : ℝ) (hθ : 0 < θ ∧ θ < 2 * π) :
  ∃! (θ : ℝ), (θ > 0 ∧ θ < 2 * π) ∧ (Real.sec (2 * π * Real.sin θ) = Real.csc (2 * π * Real.cos θ)) :=
sorry

end problem_solution_count_l568_568278


namespace value_of_x_squared_plus_9y_squared_l568_568386

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l568_568386


namespace vector_c_solution_l568_568715

variables {ℝ : Type} [field ℝ]

noncomputable def a : ℝ × ℝ := (1, 2)

noncomputable def b : ℝ × ℝ := (2, -3)

noncomputable def c : ℝ × ℝ := (7 / 9, 7 / 3)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem vector_c_solution :
  perpendicular c (a.1 + b.1, a.2 + b.2) ∧ parallel b (a.1 - c.1, a.2 - c.2) →
  c = (7 / 9, 7 / 3) :=
by
  -- proof
  sorry

end vector_c_solution_l568_568715


namespace sum_three_digit_no_0_or_5_l568_568286

theorem sum_three_digit_no_0_or_5 :
  let valid_digits := {1, 2, 3, 4, 6, 7, 8, 9}
  let count_valid_numbers := 8 * 8 * 8
  let each_digit_frequency := count_valid_numbers / 8
  let sum_valid_digits := (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9)
  let sum_each_position := each_digit_frequency * sum_valid_digits
  let sum_hundreds := sum_each_position * 100
  let sum_tens := sum_each_position * 10
  let sum_units := sum_each_position * 1
  let total_sum := sum_hundreds + sum_tens + sum_units
  total_sum = 284160 := 
by
  let valid_digits := {1, 2, 3, 4, 6, 7, 8, 9}
  let count_valid_numbers := 8 * 8 * 8
  let each_digit_frequency := count_valid_numbers / 8
  let sum_valid_digits := (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9)
  let sum_each_position := each_digit_frequency * sum_valid_digits
  let sum_hundreds := sum_each_position * 100
  let sum_tens := sum_each_position * 10
  let sum_units := sum_each_position * 1
  let total_sum := sum_hundreds + sum_tens + sum_units
  have : total_sum = 284160 := by sorry
  exact this

end sum_three_digit_no_0_or_5_l568_568286


namespace ring_stack_vertical_distance_l568_568592

theorem ring_stack_vertical_distance :
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  ∃ n, (top_ring_outer_diameter - bottom_ring_outer_diameter) / decrement + 1 = n ∧
       n * ring_thickness = 260 :=
by {
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  sorry
}

end ring_stack_vertical_distance_l568_568592


namespace Skylar_chickens_less_than_triple_Colten_l568_568483

def chickens_count (S Q C : ℕ) : Prop := 
  Q + S + C = 383 ∧ 
  Q = 2 * S + 25 ∧ 
  C = 37

theorem Skylar_chickens_less_than_triple_Colten (S Q C : ℕ) 
  (h : chickens_count S Q C) : (3 * C - S = 4) := 
sorry

end Skylar_chickens_less_than_triple_Colten_l568_568483


namespace sum_of_remainders_l568_568844

theorem sum_of_remainders :
  let n (a : Nat) := 1111 * a + 123
  let rem31 (x : Nat) := x % 31
  let valid_a (a : Nat) := 1 ≤ a ∧ a ≤ 6 
  let remainders := (List.range' 1 6).map (λ a => rem31 (n a))
  List.sum remainders = 99 :=
by
  let n (a : Nat) := 1111 * a + 123
  let rem31 (x : Nat) := x % 31
  let valid_a (a : Nat) := 1 ≤ a ∧ a ≤ 6 
  let remainders := (List.range' 1 6).map (λ a => rem31 (n a))
  have h : remainders = [4, 9, 14, 19, 24, 29], from sorry
  calc List.sum remainders
      = List.sum [4, 9, 14, 19, 24, 29] : by rw h
  ... = 4 + 9 + 14 + 19 + 24 + 29     : by simp
  ... = 99                            : by norm_num
  sorry

end sum_of_remainders_l568_568844


namespace max_value_f_monotonic_f_f_leq_g_l568_568309

-- Definition of f(x)
def f (x a : ℝ) : ℝ := (-x^2 + 2 * a * x) * Real.exp x

-- Definition of g(x)
def g (x : ℝ) : ℝ := (x - 1) * Real.exp (2 * x)

-- Given a >= 0, prove x = a - 1 ± √(a^2 + 1) is where f(x) attains its maximum
theorem max_value_f (a : ℝ) (ha : 0 ≤ a) : 
  ∃ x1 x2 : ℝ, x1 = a - 1 - Real.sqrt (a^2 + 1) ∧ x2 = a - 1 + Real.sqrt (a^2 + 1) ∧ 
               (∀ x : ℝ, f x a ≤ f x1 a ∨ f x a ≤ f x2 a) := 
sorry

-- Given f(x) is monotonic on [-1, 1], prove a ≥ 3/4
theorem monotonic_f (a : ℝ) (hmon : ∀ x1 x2 : ℝ, -1 ≤ x1 → x1 ≤ x2 → x2 ≤ 1 → f x1 a ≤ f x2 a) : 
  3 / 4 ≤ a := 
sorry

-- Given f(x) ≤ g(x) for x ≥ 1, prove 0 ≤ a ≤ 1/2
theorem f_leq_g (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → f x a ≤ g x) : 
  0 ≤ a ∧ a ≤ 1 / 2 := 
sorry

end max_value_f_monotonic_f_f_leq_g_l568_568309


namespace find_c_value_l568_568732

noncomputable def line_translated (c : ℤ) : ℤ × ℤ → ℤ :=
  λ x, 2 * x.1 - x.2 + c - 3

def circle (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = 5

def distance_from_origin (x : ℝ × ℝ) : ℝ :=
  Real.sqrt (x.1^2 + x.2^2)

def tangent_to_circle (c : ℤ) : Prop :=
  let L := line_translated c in
  abs (L (0, 0)) / Real.sqrt 5 = Real.sqrt 5

theorem find_c_value : ∃ c : ℤ, tangent_to_circle c ∧ (c = -2 ∨ c = 8) :=
  by
    sorry

end find_c_value_l568_568732


namespace minimum_distance_ellipse_l568_568107

noncomputable def minimum_distance (P : ℝ × ℝ) (M : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 9) + (P.2^2 / 4) = 1

theorem minimum_distance_ellipse (M : ℝ × ℝ) (hM : M = (1, 0)) :
  ∃ (P : ℝ × ℝ), on_ellipse P ∧ minimum_distance P M = 4 * real.sqrt 5 / 5 := 
by
  sorry

end minimum_distance_ellipse_l568_568107


namespace number_of_sides_l568_568635

-- Define the conditions
def interior_angle (n : ℕ) : ℝ := 156

-- The main theorem to prove the number of sides
theorem number_of_sides (n : ℕ) (h : interior_angle n = 156) : n = 15 :=
by
  sorry

end number_of_sides_l568_568635


namespace quadratic_solution_conditions_l568_568039

noncomputable def f (x : ℝ) : ℝ :=
if x = 2 then 1 else 1 / |x - 2|

theorem quadratic_solution_conditions (a b x1 x2 x3 : ℝ) 
(h_distinct_solutions : x1 < x2 ∧ x2 < x3) 
(h_solutions_eq : f^2 x1 + a * f x1 + b = 0 ∧ f^2 x2 + a * f x2 + b = 0 ∧ f^2 x3 + a * f x3 + b = 0) :
(x1^2 + x2^2 + x3^2 = 14) ∧ (1 + a + b = 0) ∧ (a^2 - 4 * b = 0) ∧ ¬(x1 + x3 = 0) :=
sorry

end quadratic_solution_conditions_l568_568039


namespace min_value_inequality_l568_568688

theorem min_value_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 :=
by
  sorry

end min_value_inequality_l568_568688


namespace shift_right_linear_function_l568_568413

theorem shift_right_linear_function (x : ℝ) : 
  (∃ k b : ℝ, k ≠ 0 ∧ (∀ x : ℝ, y = -2x → y = kx + b) → (x, y) = (x - 3, -2(x-3))) → y = -2x + 6 :=
by
  sorry

end shift_right_linear_function_l568_568413


namespace painting_area_l568_568047

theorem painting_area (wall_height wall_length bookshelf_height bookshelf_length : ℝ)
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_bookshelf_height : bookshelf_height = 3)
  (h_bookshelf_length : bookshelf_length = 5) :
  wall_height * wall_length - bookshelf_height * bookshelf_length = 135 := 
by
  sorry

end painting_area_l568_568047


namespace price_of_basic_computer_l568_568916

-- Define the problem conditions
variables (C P : ℕ)

-- Condition 1: The total price of a basic computer and printer is $2,500.
def condition1 := C + P = 2500

-- Condition 2: The enhanced computer's price is $500 more than the price of the basic computer,
--              and the price of the printer would have been 1/4 of the total price with the enhanced computer.
def condition2 := P = (1/4 : ℝ) * (C + 500 + P)

-- Theorem: The price of the basic computer is $1,750
theorem price_of_basic_computer (h1 : condition1) (h2 : condition2) : C = 1750 :=
by
  sorry

end price_of_basic_computer_l568_568916


namespace problem_statement_l568_568720

variable (x P : ℝ)

theorem problem_statement
  (h1 : x^2 - 5 * x + 6 < 0)
  (h2 : P = x^2 + 5 * x + 6) :
  (20 < P) ∧ (P < 30) :=
sorry

end problem_statement_l568_568720


namespace sum_of_valid_a_eq_53_l568_568127

theorem sum_of_valid_a_eq_53:
  ∀ (f : ℤ → ℤ), 
  (∀ x, f x = x^2 - (a : ℤ) * x + 3 * a) → 
  (∃ r s : ℤ, f r = 0 ∧ f s = 0 ∧ r ≠ s ∧ r + s = a ∧ r * s = 3 * a) →
  (let a_values := {a | ∃ r s, r + s = a ∧ r * s = 3 * a ∧ (a - 6)^2 = (a^2 - 12 * a)} in 
   ∑ a in (a_values.filter (λ a, a ∈ ℤ)), a = 53) := sorry

end sum_of_valid_a_eq_53_l568_568127


namespace equivalent_exponentiation_l568_568972

theorem equivalent_exponentiation (h : 64 = 8^2) : 8^15 / 64^3 = 8^9 :=
by
  sorry

end equivalent_exponentiation_l568_568972


namespace triangle_properties_l568_568509

-- Definitions and assumptions
variable {d : ℝ} (T : ℝ)
variable (a : ℝ) (b : ℝ) (c : ℝ)
variable (s : ℝ := (a + b + c) / 2)

-- Condition: Sides form an arithmetic progression with common difference d
def sides_form_arithmetic_prog (a b c : ℝ) (d : ℝ) : Prop :=
  b = a + d ∧ c = a + 2 * d

-- Condition: Area of the triangle is T
def area_of_triangle (a b c s T : ℝ) : Prop :=
  T = Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Specific values
def specific_values (d : ℝ) (T : ℝ) : Prop :=
  d = 1 ∧ T = 6

-- Specific sides
def specific_sides (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5

-- Specific angles
def specific_angles (a b c : ℝ) : Prop :=
  ∃ (α β γ : ℝ), α = Real.arcsin (a / c) ∧
  β = 90 - α ∧
  γ = 90 ∧
  α ≈ 36.87 ∧
  β ≈ 53.13 ∧
  γ = 90

-- Proving the main theorem
theorem triangle_properties :
  (∀ a b c s T,
    sides_form_arithmetic_prog a b c d →
    area_of_triangle a b c s T →
    specific_values d T →
    specific_sides a b c ∧ specific_angles a b c) :=
sorry -- The proof is omitted.

end triangle_properties_l568_568509


namespace probability_of_sum_leq_10_l568_568881

open Nat

-- Define the three dice roll outcomes
def dice_outcomes := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define the total number of outcomes when rolling three dice
def total_outcomes : ℕ := 6 ^ 3

-- Count the number of valid outcomes where the sum of three dice is less than or equal to 10
def count_valid_outcomes : ℕ := 75  -- This is determined through combinatorial calculations or software

-- Define the desired probability
def desired_probability := (count_valid_outcomes : ℚ) / total_outcomes

-- Prove that the desired probability equals 25/72
theorem probability_of_sum_leq_10 :
  desired_probability = 25 / 72 :=
by sorry

end probability_of_sum_leq_10_l568_568881


namespace find_m_from_permutation_l568_568305

theorem find_m_from_permutation (A : Nat → Nat → Nat) (m : Nat) (hA : A 11 m = 11 * 10 * 9 * 8 * 7 * 6 * 5) : m = 7 :=
sorry

end find_m_from_permutation_l568_568305


namespace find_smallest_a_l568_568283

-- Given definitions
def expr (a : ℝ) := (8 * real.sqrt ((3 * a) ^ 2 + 2 ^ 2) - 5 * a ^ 2 - 2) / (real.sqrt (2 + 5 * a ^ 2) + 4)

-- Lean theorem statement
theorem find_smallest_a : 
  ∃ a : ℝ, expr a = 3 ∧ ∀ b : ℝ, expr b = 3 → a ≤ b := 
begin
  sorry
end

end find_smallest_a_l568_568283


namespace car_average_speed_l568_568927

noncomputable def average_speed (speeds : List ℝ) (distances : List ℝ) (times : List ℝ) : ℝ :=
  (distances.sum + times.sum) / times.sum

theorem car_average_speed :
  let distances := [30, 35, 35, 52 / 3, 15]
  let times := [30 / 45, 35 / 55, 30 / 60, 20 / 60, 15 / 65]
  average_speed [45, 55, 70, 52, 65] distances times = 64.82 := by
  sorry

end car_average_speed_l568_568927


namespace final_value_T_l568_568967

theorem final_value_T : 
  let T := (1 + (List.range 10).map (fun x => x + 1)).sum 
  in T = 56 := by
  sorry

end final_value_T_l568_568967


namespace graph_f_shifted_up_3_is_C_l568_568497

noncomputable def f (x : ℝ) : ℝ :=
if hx₁ : x ∈ Icc (-3 : ℝ) 0 then -2 - x
else if hx₂ : x ∈ Icc (0 : ℝ) 2 then Real.sqrt (4 - (x - 2)^2) - 2
else if hx₃ : x ∈ Icc (2 : ℝ) 3 then 2 * (x - 2)
else 0

theorem graph_f_shifted_up_3_is_C : 
  ∀ x ∈ Icc (-3 : ℝ) 3, f(x) + 3 = 
    if hx₁ : x ∈ Icc (-3 : ℝ) 0 then 1 - x
    else if hx₂ : x ∈ Icc (0 : ℝ) 2 then Real.sqrt (4 - (x - 2)^2) + 1
    else if hx₃ : x ∈ Icc (2 : ℝ) 3 then 2 * (x - 2) + 3
    else 0 :=
by
  sorry

end graph_f_shifted_up_3_is_C_l568_568497


namespace union_A_B_eq_C_l568_568684

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
noncomputable def C : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}

theorem union_A_B_eq_C : A ∪ B = C := by
  sorry

end union_A_B_eq_C_l568_568684


namespace length_SD_l568_568756

-- Define the geometric settings and variables
structure Rectangle (A B C D : Type*) : Prop :=
(is_rectangle : ∀ (P : Type*), ∃ (TS : Type*), TS ⊥ BC ∧ BP = PT / 2)

variables {A B C D P Q R S T : Type*}
variables (PQ AQ PA BP PT TQ TS AB SD PQA : ℝ)

noncomputable def rectangle_conditions : Rectangle A B C D :=
{ is_rectangle := λ P, ⟨TS, by sorry⟩ }

-- Given conditions
axiom angle_APD_90 : ∠APD = 90
axiom angle_APQ_90 : ∠APQ = 90
axiom is_rectangle_ABCD : Rectangle A B C D
axiom triangle_PQA : PQ PQA AQ PA (angle_APQ_90) -- Represent triangle properties  
axiom length_PA : PA = 24
axiom length_AQ : AQ = 18
axiom length_QP : QP = 30

-- Theorem stating the final geometric proof
theorem length_SD : SD = (sqrt 1872) / 3 :=
begin
  sorry -- Provide the actual proof in this section
end

end length_SD_l568_568756


namespace shortest_player_height_correct_l568_568868

def tallest_player_height : Real := 77.75
def height_difference : Real := 9.5
def shortest_player_height : Real := 68.25

theorem shortest_player_height_correct :
  tallest_player_height - height_difference = shortest_player_height :=
by
  sorry

end shortest_player_height_correct_l568_568868


namespace digit_place_value_ratio_l568_568759

theorem digit_place_value_ratio : 
  let num := 43597.2468
  let digit5_place_value := 10    -- tens place
  let digit2_place_value := 0.1   -- tenths place
  digit5_place_value / digit2_place_value = 100 := 
by 
  sorry

end digit_place_value_ratio_l568_568759


namespace system_cos_eq_unique_solution_l568_568259

theorem system_cos_eq_unique_solution (n : ℕ) (hn : n > 0) :
  ∃ x_0, (cos x_0 = x_0) ∧ (∀ (i : ℕ) (h : i < n), ∃ (x : ℝ), cos x = x ∧ x = x_0) := 
sorry

end system_cos_eq_unique_solution_l568_568259


namespace increased_volume_l568_568090

theorem increased_volume (base_area : ℝ) (height_increase : ℝ) :
  base_area = 12 ∧ height_increase = 5 → base_area * height_increase = 60 :=
begin
  intros h,
  cases h with h_base h_height,
  rw [h_base, h_height],
  norm_num
end

end increased_volume_l568_568090


namespace midpoints_diagonal_intersection_l568_568777

variable {Point : Type}

structure Quadrilateral (Point : Type) :=
  (A B C D : Point)

def isMidpoint (P A B : Point) : Prop :=
  ∃ (M : Point), (P = M) ∧ (P = midpoint A B) -- A simplified definition of midpoint

def intersection (P Q R S : Point) : Point :=
  sorry -- Assuming a function to calculate intersection (usually this requires more setup)

theorem midpoints_diagonal_intersection
          (A B C D P Q R S O : Point)
          (quad : Quadrilateral Point := ⟨A, B, C, D⟩)
          (hP : isMidpoint P A B)
          (hQ : isMidpoint Q B C)
          (hR : isMidpoint R C D)
          (hS : isMidpoint S D A)
          (hO : O = intersection P R Q S) :
        (distance P O = distance R O) ∧ (distance Q O = distance S O) :=
  by
    sorry

end midpoints_diagonal_intersection_l568_568777


namespace certain_event_implies_at_least_one_genuine_l568_568300

theorem certain_event_implies_at_least_one_genuine :
  ∀ (products : Fin 12 → bool),
    (∃ i, products i = true) →
    ∀ (selection : Finset (Fin 12)),
      selection.card = 3 → 
      (∃ i ∈ selection, products i = true) :=
begin
  intros products h_ex selection h_card,
  sorry
end

end certain_event_implies_at_least_one_genuine_l568_568300


namespace joseph_cards_l568_568442

theorem joseph_cards (cards_per_student : ℕ) (students : ℕ) (cards_left : ℕ) 
    (H1 : cards_per_student = 23)
    (H2 : students = 15)
    (H3 : cards_left = 12) 
    : (cards_per_student * students + cards_left = 357) := 
  by
  sorry

end joseph_cards_l568_568442


namespace sin_x_cos_x_value_l568_568725

theorem sin_x_cos_x_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 :=
  sorry

end sin_x_cos_x_value_l568_568725


namespace ratio_night_to_day_l568_568928

-- Definitions based on conditions
def birds_day : ℕ := 8
def birds_total : ℕ := 24
def birds_night : ℕ := birds_total - birds_day

-- Theorem statement
theorem ratio_night_to_day : birds_night / birds_day = 2 := by
  sorry

end ratio_night_to_day_l568_568928


namespace impossibility_of_unique_path_l568_568154

-- Define the bowls and marbles
noncomputable def bowls := ["A", "B", "C", "D"]
noncomputable def marbles := 4

-- Define type for distributions
def distribution : Type := (ℕ × ℕ × ℕ × ℕ)

-- Condition: A move consists of transferring one marble from a bowl to one of the adjacent bowls
def valid_move (d1 d2 : distribution) : Prop :=
  ∃ i j : ℕ, i ≠ j ∧ d1.1 i - d2.1 i = 1 ∧ d2.1 j - d1.1 j = 1 ∧
    (bowls.nth i = bowls.nth (j - 1) ∨ bowls.nth i = bowls.nth (j + 1))

-- The main question is whether it is possible to perform a succession of moves such that
-- every distribution appears exactly once.
theorem impossibility_of_unique_path : 
  ¬ ∃ f : ℕ → distribution, 
    (∀ n, valid_move (f n) (f (n + 1))) ∧ 
    (∀ d : distribution, ∃ n : ℕ, f n = d) := 
sorry

end impossibility_of_unique_path_l568_568154


namespace product_of_r_for_exactly_one_real_solution_l568_568281

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem product_of_r_for_exactly_one_real_solution :
  let a := 6
  let c := 10
  let discriminant_is_zero := discriminant a (-3 * r) c = 0
  let r1 := sqrt (80 / 3)
  let r2 := -sqrt (80 / 3)
  r1 * r2 = -80 / 3 :=
by
  sorry

end product_of_r_for_exactly_one_real_solution_l568_568281


namespace son_age_l568_568206

-- Defining the variables
variables (S F : ℕ)

-- The conditions
def condition1 : Prop := F = S + 25
def condition2 : Prop := F + 2 = 2 * (S + 2)

-- The statement to be proved
theorem son_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 23 :=
sorry

end son_age_l568_568206


namespace twin_primes_sum_l568_568034

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_twin_prime_pair (P : ℕ) : Prop := is_prime P ∧ is_prime (P + 2)

theorem twin_primes_sum : 
  let S := ∑ P in { P | P * (P + 2) ≤ 2007 ∧ is_twin_prime_pair P }, P 
  in S = 106 :=
by
  sorry

end twin_primes_sum_l568_568034


namespace eight_letter_good_words_count_l568_568255

def is_good_word (s : List Char) : Prop :=
  ∀ i : ℕ, i < s.length - 1 →
  (s[i] = 'A' → s[i + 1] ≠ 'C') ∧
  (s[i] = 'B' → s[i + 1] ≠ 'A') ∧
  (s[i] = 'C' → s[i + 1] ≠ 'B')

def count_good_words (n : ℕ) : ℕ :=
  if h : 1 ≤ n then
    (3 : ℕ) * (2 : ℕ)^(n-1)
  else
    0

theorem eight_letter_good_words_count :
  count_good_words 8 = 384 :=
by
  sorry

end eight_letter_good_words_count_l568_568255


namespace difference_between_Annette_and_Sara_l568_568606

-- Define the weights of the individuals
variables (A C S B E : ℝ)

-- Conditions given in the problem
def condition1 := A + C = 95
def condition2 := C + S = 87
def condition3 := A + S = 97
def condition4 := C + B = 100
def condition5 := A + C + B = 155
def condition6 := A + S + B + E = 240
def condition7 := E = 1.25 * C

-- The theorem that we want to prove
theorem difference_between_Annette_and_Sara (A C S B E : ℝ)
  (h1 : condition1 A C)
  (h2 : condition2 C S)
  (h3 : condition3 A S)
  (h4 : condition4 C B)
  (h5 : condition5 A C B)
  (h6 : condition6 A S B E)
  (h7 : condition7 C E) :
  A - S = 8 :=
by {
  sorry
}

end difference_between_Annette_and_Sara_l568_568606


namespace domain_of_sqrt_expression_l568_568651

def isDomain (x : ℝ) : Prop := x ≥ -3 ∧ x < 7

theorem domain_of_sqrt_expression : 
  { x : ℝ | isDomain x } = { x | x ≥ -3 ∧ x < 7 } :=
by
  sorry

end domain_of_sqrt_expression_l568_568651


namespace polynomial_zero_unique_l568_568779

theorem polynomial_zero_unique (α : ℝ) (P : ℝ[X]) :
  (∀ x : ℝ, P.eval (2 * x + α) ≤ (x^20 + x^19) * P.eval x) ↔ P = 0 :=
sorry

end polynomial_zero_unique_l568_568779


namespace minimum_percentage_of_poor_works_l568_568218

-- Definition of the problem.
def total_works (N : ℕ) := N
def fraction_poor_works (N : ℕ) := 0.20 * N
def fraction_good_works (N : ℕ) := 0.80 * N
def misclassified_good_as_poor (N : ℕ) := 0.10 * fraction_good_works N
def misclassified_poor_as_good (N : ℕ) := 0.10 * fraction_poor_works N
def classified_as_poor_by_network (N : ℕ) := 
  fraction_poor_works N - misclassified_poor_as_good N + misclassified_good_as_poor N

-- Statement of the theorem: minimum percentage of actual poor works among re-checked works.
theorem minimum_percentage_of_poor_works (N : ℕ) :
  let fraction_rechecked_poor := (fraction_poor_works N - misclassified_poor_as_good N) / classified_as_poor_by_network N
  floor ((fraction_rechecked_poor * 100).to_float) = 69 := 
sorry

end minimum_percentage_of_poor_works_l568_568218


namespace hyperbolas_same_asymptotes_l568_568850

theorem hyperbolas_same_asymptotes :
  (∀ x y, (x^2 / 4 - y^2 / 9 = 1) → (∃ k, y = k * x)) →
  (∀ x y, (y^2 / 18 - x^2 / N = 1) → (∃ k, y = k * x)) →
  N = 8 :=
by sorry

end hyperbolas_same_asymptotes_l568_568850


namespace vectors_orthogonal_if_magnitudes_equal_l568_568397

variables (a b : E) [InnerProductSpace ℝ E] (hab : a + b = a - b)

theorem vectors_orthogonal_if_magnitudes_equal (a b : E) 
  [InnerProductSpace ℝ E] 
  (hna : a ≠ 0) 
  (hnb : b ≠ 0) 
  (h : ∥a + b∥ = ∥a - b∥) :
  ⟪a, b⟫ = 0 := 
by {
  sorry
}

end vectors_orthogonal_if_magnitudes_equal_l568_568397


namespace world_internet_conference_l568_568739

noncomputable def promote_chinese_culture (blending: Prop) (embracing: Prop): Prop := 
  blending ∧ embracing

noncomputable def innovate_world_culture (blending: Prop) (embracing: Prop): Prop := 
  blending ∧ embracing

noncomputable def enhance_international_influence (blending: Prop) (embracing: Prop): Prop := 
  blending ∧ embracing

theorem world_internet_conference 
  (blending: Prop) 
  (embracing: Prop) :
  promote_chinese_culture blending embracing ∧ 
  innovate_world_culture blending embracing ∧ 
  enhance_international_influence blending embracing :=
by
  split
  · exact blending ∧ embracing
  · exact blending ∧ embracing
  · exact blending ∧ embracing

end world_internet_conference_l568_568739


namespace percentage_of_purple_compared_to_yellow_l568_568799

-- Definition of variables used in the problem
variables (yellow purple green : ℕ)

-- Given conditions
def cond1 := yellow = 10
def cond2 := green = 0.25 * (yellow + purple)
def cond3 := yellow + purple + green = 35

-- The proof goal
theorem percentage_of_purple_compared_to_yellow 
  (yellow purple green : ℕ) 
  (h1 : cond1 yellow) 
  (h2 : cond2 yellow purple green) 
  (h3 : cond3 yellow purple green) 
  : (purple / yellow : ℚ) * 100 = 180 :=
by
  sorry

end percentage_of_purple_compared_to_yellow_l568_568799


namespace total_players_on_ground_l568_568741

def cricket_players : ℕ := 15
def hockey_players : ℕ := 12
def football_players : ℕ := 13
def softball_players : ℕ := 15

theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + softball_players = 55 := 
by
  sorry

end total_players_on_ground_l568_568741


namespace coefficient_of_x2_l568_568387

theorem coefficient_of_x2 (a : ℝ) : 
     let expr := (1 + a * X) * (1 + X) ^ 5
     in coeff expr 2 = 15 -> a = 1 := 
by
  sorry

end coefficient_of_x2_l568_568387


namespace binomial_probability_l568_568391

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability mass function
def binomial_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coeff n k) * (p^k) * ((1 - p)^(n - k))

-- Define the conditions of the problem
def n := 5
def k := 2
def p : ℚ := 1/3

-- State the theorem
theorem binomial_probability :
  binomial_pmf n k p = binomial_coeff 5 2 * (1/3)^2 * (2/3)^3 := by
  sorry

end binomial_probability_l568_568391


namespace time_for_c_l568_568174

   variable (A B C : ℚ)

   -- Conditions
   def condition1 : Prop := (A + B = 1/6)
   def condition2 : Prop := (B + C = 1/8)
   def condition3 : Prop := (C + A = 1/12)

   -- Theorem to be proved
   theorem time_for_c (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 C A) :
     1 / C = 48 :=
   sorry
   
end time_for_c_l568_568174


namespace trapezoid_area_is_64_l568_568247

def shorter_base : ℝ := 4
def longer_base : ℝ := 3 * shorter_base
def height : ℝ := 2 * shorter_base
def area_of_trapezoid (b1 b2 h : ℝ) : ℝ := (1 / 2) * (b1 + b2) * h

theorem trapezoid_area_is_64 :
  area_of_trapezoid shorter_base longer_base height = 64 :=
by
  sorry

end trapezoid_area_is_64_l568_568247


namespace factorize_expr_l568_568265

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end factorize_expr_l568_568265


namespace find_b_l568_568752

theorem find_b :
  ∀ (A B C : ℝ) (a b c : ℝ),
  sin A = 2 * sqrt 2 / 3 →
  sin B > sin C →
  a = 3 →
  (1/2) * b * c * sin A = 2 * sqrt 2 →
  b = 3 :=
by sorry

end find_b_l568_568752


namespace four_digit_numbers_divisible_by_6_count_l568_568362

-- Definitions based on the conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def is_divisible_by_2 (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_3 (n : ℕ) : Prop := n.digits.sum % 3 = 0
def is_divisible_by_6 (n : ℕ) : Prop := is_divisible_by_2 n ∧ is_divisible_by_3 n

-- The main theorem stating the problem
theorem four_digit_numbers_divisible_by_6_count : 
  (finset.Icc 1000 9999).filter is_divisible_by_6 = 1350 :=
sorry

end four_digit_numbers_divisible_by_6_count_l568_568362


namespace winning_pair_probability_l568_568198

noncomputable def probability_of_winning_pair : ℚ :=
  let total_cards := 12
  let total_ways := (total_cards.choose 2)  -- Total ways to choose 2 cards from 12
  let ways_same_label := 5                 -- 5 ways to choose same label pair
  let ways_same_color := 2 * (6.choose 2)  -- 2 colors, each 6 cards, choose 2 out of 6
  let favorable_ways := ways_same_label + ways_same_color
  favorable_ways / total_ways

theorem winning_pair_probability :
  probability_of_winning_pair = 35 / 66 :=
by
  sorry

end winning_pair_probability_l568_568198


namespace trajectory_equation_k2_find_lambda_k0_l568_568760

section part1

variables {x y k: ℝ}

def vector_A_P (P : ℝ × ℝ) : ℝ × ℝ := (P.1, P.2 - 1)
def vector_B_P (P : ℝ × ℝ) : ℝ × ℝ := (P.1, P.2 + 1)
def vector_P_C (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - 1, P.2)

def dot_prod (u v: ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def sq_magnitude (v: ℝ × ℝ) : ℝ := v.1 ^ 2 + v.2 ^ 2

theorem trajectory_equation_k2 :
  (k = 2) → 
  (∀ P, 
    dot_prod (vector_A_P P) (vector_B_P P) = k * sq_magnitude (vector_P_C P) → 
      (P.1 - 2) ^ 2 + P.2 ^ 2 = 1) :=
sorry

end part1

section part2

variables {x y λ: ℝ}

def max_ap_bp (P : ℝ × ℝ) (λ : ℝ) : ℝ := 
  (λ^2 * (P.1^2 + (P.2 - 1) ^ 2) + P.1^2 + (P.2 + 1) ^ 2)^0.5

theorem find_lambda_k0 :
  (k = 0) → 
  (∀ P, 
    dot_prod (vector_A_P P) (vector_B_P P) = 0 → 
    sq_magnitude P = 1 →
    (max_ap_bp P λ) = 4 → 
    λ = 2 ∨ λ = -2) :=
sorry

end part2

end trajectory_equation_k2_find_lambda_k0_l568_568760


namespace value_of_x2_plus_9y2_l568_568380

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l568_568380


namespace fraction_power_mult_correct_l568_568991

noncomputable def fraction_power_mult : Prop :=
  (\left(\frac{1}{3} \right)^{4}) * \left(\frac{1}{5} \right) = \left(\frac{1}{405} \right)

theorem fraction_power_mult_correct : fraction_power_mult :=
by
  -- The complete proof will be here.
  sorry

end fraction_power_mult_correct_l568_568991


namespace total_expenditure_eq_fourteen_l568_568475

variable (cost_barrette cost_comb : ℕ)
variable (kristine_barrettes kristine_combs crystal_barrettes crystal_combs : ℕ)

theorem total_expenditure_eq_fourteen 
  (h_cost_barrette : cost_barrette = 3)
  (h_cost_comb : cost_comb = 1)
  (h_kristine_barrettes : kristine_barrettes = 1)
  (h_kristine_combs : kristine_combs = 1)
  (h_crystal_barrettes : crystal_barrettes = 3)
  (h_crystal_combs : crystal_combs = 1) :
  (kristine_barrettes * cost_barrette + kristine_combs * cost_comb) +
  (crystal_barrettes * cost_barrette + crystal_combs * cost_comb) = 14 := 
by 
  sorry

end total_expenditure_eq_fourteen_l568_568475


namespace three_dice_sum_divisible_by_3_l568_568136

-- We define a function to represent the event that the sum of three dice is divisible by 3
def event_sum_divisible_by_3 (d1 d2 d3 : ℕ) : Prop :=
  ((d1 + d2 + d3) % 3) = 0

-- Define the probability that the sum of the numbers on three dice is divisible by 3
noncomputable def probability_sum_divisible_by_3 : ℚ :=
  13 / 27

-- Now we state the theorem that the probability of the event is 13/27
theorem three_dice_sum_divisible_by_3 :
  ∀ (d1 d2 d3 : fin 6) (h1 : d1.val < 6) (h2 : d2.val < 6) (h3 : d3.val < 6),
    d1.val + 1 + d2.val + 1 + d3.val + 1 ≡ 0 [MOD 3] ↔
    probability_sum_divisible_by_3 = 13 / 27 :=
begin
  sorry
end

end three_dice_sum_divisible_by_3_l568_568136


namespace sin_double_angle_half_pi_l568_568307

theorem sin_double_angle_half_pi (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1 / 3) : 
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by
  sorry

end sin_double_angle_half_pi_l568_568307


namespace modified_calendar_leap_years_l568_568745

theorem modified_calendar_leap_years : 
  ∀ (years : ℕ), years = 300 →
  let leap_years_4 := years / 4,
      non_leap_centuries := 2
  in leap_years_4 - non_leap_centuries = 73 :=
by
  intros years h_y
  dsimp only
  sorry

end modified_calendar_leap_years_l568_568745


namespace minimum_checkers_l568_568833

variables (n : ℕ) (board : fin n × fin n → Prop)

-- Defining conditions
def condition_a (board : fin n × fin n → Prop) : Prop :=
∀ (i j : fin n), ¬ board (i, j) → 
  (i > 0 ∧ board (i-1, j)) ∨ 
  (i < n-1 ∧ board (i+1, j)) ∨ 
  (j > 0 ∧ board (i, j-1)) ∨ 
  (j < n-1 ∧ board (i, j+1))

def condition_b (board : fin n × fin n → Prop) : Prop :=
∀ (i1 j1 i2 j2 : fin n), board (i1, j1) → board (i2, j2) →
  ∃ seq : list (fin n × fin n), 
    seq.head = (i1, j1) ∧
    seq.last = (i2, j2) ∧
    (∀ (p : fin n × fin n), p ∈ seq → board p) ∧
    (∀ (p1 p2 : fin n × fin n), p1 ∈ seq → p2 ∈ seq → 
      ((p1.1 = p2.1 ∧ (p1.2 = p2.2-1 ∨ p1.2 = p2.2+1)) ∨ 
       (p1.2 = p2.2 ∧ (p1.1 = p2.1-1 ∨ p1.1 = p2.1+1))))

-- The theorem statement
theorem minimum_checkers (n : ℕ) (board : fin n × fin n → Prop) 
  (h1 : condition_a board) (h2 : condition_b board) : 
  ∃ (V : set (fin n × fin n)), V.size ≥ (n^2 - 2) / 3 :=
sorry

end minimum_checkers_l568_568833


namespace molecular_weight_l568_568161

theorem molecular_weight (w8 : ℝ) (n : ℝ) (w1 : ℝ) (h1 : w8 = 2376) (h2 : n = 8) : w1 = 297 :=
by
  sorry

end molecular_weight_l568_568161


namespace valid_arrangements_count_l568_568073

def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def is_strictly_ascending (lst : List ℕ) : Prop :=
  ∀ i j, i < j → lst.get? i < lst.get? j

def is_strictly_descending (lst : List ℕ) : Prop :=
  ∀ i j, i < j → lst.get? i > lst.get? j

def count_valid_arrangements (cards : List ℕ) : ℕ :=
  (cards.permutations.filter (λ arr, ∃ i, 
    is_strictly_ascending (arr.removeNth i) ∨ 
    is_strictly_descending (arr.removeNth i))).length

theorem valid_arrangements_count :
  count_valid_arrangements cards = 72 := by
  sorry

end valid_arrangements_count_l568_568073


namespace football_games_this_year_l568_568768

theorem football_games_this_year 
  (total_games : ℕ) 
  (games_last_year : ℕ) 
  (games_this_year : ℕ) 
  (h1 : total_games = 9) 
  (h2 : games_last_year = 5) 
  (h3 : total_games = games_last_year + games_this_year) : 
  games_this_year = 4 := 
sorry

end football_games_this_year_l568_568768


namespace compute_fraction_pow_mult_l568_568980

def frac_1_3 := (1 : ℝ) / (3 : ℝ)
def frac_1_5 := (1 : ℝ) / (5 : ℝ)
def target := (1 : ℝ) / (405 : ℝ)

theorem compute_fraction_pow_mult :
  (frac_1_3^4 * frac_1_5) = target :=
by
  sorry

end compute_fraction_pow_mult_l568_568980


namespace number_of_points_on_P_shape_l568_568173

theorem number_of_points_on_P_shape (side_length : ℕ) (h : side_length = 10) :
  ∑ i in (finset.range(side_length + 1)), 1 + 
  ∑ i in (finset.range(side_length + 1)), 1 +
  ∑ i in (finset.range(side_length + 1)), 1 - 2 = 31 :=
by
  sorry

end number_of_points_on_P_shape_l568_568173


namespace range_of_x_l568_568705

def piecewise_f (x : ℝ) : ℝ := if x ≤ 0 then 1 + x^2 else 1

def satisfy_condition (x : ℝ) : Prop := piecewise_f (x - 4) > piecewise_f (2 * x - 3)

theorem range_of_x : {x : ℝ | satisfy_condition x} = (Set.Ioc (3 / 2) 4) :=
by
  sorry

end range_of_x_l568_568705


namespace corvette_trip_average_rate_l568_568913

theorem corvette_trip_average_rate (total_distance : ℕ) (first_half_distance : ℕ)
  (first_half_rate : ℕ) (second_half_time_multiplier : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  first_half_distance = total_distance / 2 →
  first_half_rate = 80 →
  second_half_time_multiplier = 3 →
  total_time = (first_half_distance / first_half_rate) + (second_half_time_multiplier * (first_half_distance / first_half_rate)) →
  (total_distance / total_time) = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end corvette_trip_average_rate_l568_568913


namespace Earl_rate_36_l568_568636

theorem Earl_rate_36 (E : ℝ) (h1 : E + (2 / 3) * E = 60) : E = 36 :=
by {
  sorry
}

end Earl_rate_36_l568_568636


namespace final_weights_are_correct_l568_568835

-- Definitions of initial weights and reduction percentages per week
def initial_weight_A : ℝ := 300
def initial_weight_B : ℝ := 450
def initial_weight_C : ℝ := 600
def initial_weight_D : ℝ := 750

def reduction_A_week1 : ℝ := 0.20 * initial_weight_A
def reduction_B_week1 : ℝ := 0.15 * initial_weight_B
def reduction_C_week1 : ℝ := 0.30 * initial_weight_C
def reduction_D_week1 : ℝ := 0.25 * initial_weight_D

def weight_A_after_week1 : ℝ := initial_weight_A - reduction_A_week1
def weight_B_after_week1 : ℝ := initial_weight_B - reduction_B_week1
def weight_C_after_week1 : ℝ := initial_weight_C - reduction_C_week1
def weight_D_after_week1 : ℝ := initial_weight_D - reduction_D_week1

def reduction_A_week2 : ℝ := 0.25 * weight_A_after_week1
def reduction_B_week2 : ℝ := 0.30 * weight_B_after_week1
def reduction_C_week2 : ℝ := 0.10 * weight_C_after_week1
def reduction_D_week2 : ℝ := 0.20 * weight_D_after_week1

def weight_A_after_week2 : ℝ := weight_A_after_week1 - reduction_A_week2
def weight_B_after_week2 : ℝ := weight_B_after_week1 - reduction_B_week2
def weight_C_after_week2 : ℝ := weight_C_after_week1 - reduction_C_week2
def weight_D_after_week2 : ℝ := weight_D_after_week1 - reduction_D_week2

def reduction_A_week3 : ℝ := 0.15 * weight_A_after_week2
def reduction_B_week3 : ℝ := 0.10 * weight_B_after_week2
def reduction_C_week3 : ℝ := 0.20 * weight_C_after_week2
def reduction_D_week3 : ℝ := 0.30 * weight_D_after_week2

def weight_A_after_week3 : ℝ := weight_A_after_week2 - reduction_A_week3
def weight_B_after_week3 : ℝ := weight_B_after_week2 - reduction_B_week3
def weight_C_after_week3 : ℝ := weight_C_after_week2 - reduction_C_week3
def weight_D_after_week3 : ℝ := weight_D_after_week2 - reduction_D_week3

def reduction_A_week4 : ℝ := 0.10 * weight_A_after_week3
def reduction_B_week4 : ℝ := 0.20 * weight_B_after_week3
def reduction_C_week4 : ℝ := 0.25 * weight_C_after_week3
def reduction_D_week4 : ℝ := 0.15 * weight_D_after_week3

def final_weight_A : ℝ := weight_A_after_week3 - reduction_A_week4
def final_weight_B : ℝ := weight_B_after_week3 - reduction_B_week4
def final_weight_C : ℝ := weight_C_after_week3 - reduction_C_week4
def final_weight_D : ℝ := weight_D_after_week3 - reduction_D_week4

theorem final_weights_are_correct :
  final_weight_A = 137.7 ∧ 
  final_weight_B = 192.78 ∧ 
  final_weight_C = 226.8 ∧ 
  final_weight_D = 267.75 :=
by
  unfold final_weight_A final_weight_B final_weight_C final_weight_D
  sorry

end final_weights_are_correct_l568_568835


namespace prob_B_serves_in_third_round_prob_A_wins_majority_in_first_three_rounds_l568_568817

/-- Player A and player B are preparing for a badminton match. -/
def player_A_serves_first : Prop := true

/-- Probability that player A wins a round when serving -/
def prob_A_wins_if_serves : ℚ := 3 / 4

/-- Probability that player A wins a round when B is serving -/
def prob_A_wins_if_B_serves : ℚ := 1 / 4

/-- Results of each round are independent -/
def rounds_independent : Prop := true

/-- Prove that the probability player B will serve in the third round is 3/8 -/
theorem prob_B_serves_in_third_round : 
  player_A_serves_first → 
  prob_A_wins_if_serves = 3 / 4 → 
  prob_A_wins_if_B_serves = 1 / 4 → 
  rounds_independent →
  (3 / 4 * (1 - 3 / 4) + (1 - 3 / 4) * 3 / 4 = 3 / 8) :=
begin
  sorry
end

/-- Prove that the probability player A wins at least as many rounds as player B in the first three rounds is 21/32 -/
theorem prob_A_wins_majority_in_first_three_rounds : 
  player_A_serves_first → 
  prob_A_wins_if_serves = 3 / 4 → 
  prob_A_wins_if_B_serves = 1 / 4 → 
  rounds_independent →
  (15 / 64 + 27 / 64 = 21 / 32) :=
begin
  sorry
end

end prob_B_serves_in_third_round_prob_A_wins_majority_in_first_three_rounds_l568_568817


namespace pinching_area_preservation_l568_568816

def is_convex (P : List ℝ → ℝ → Bool) := sorry -- placeholder for convexity definition
def area (P : List ℝ) : ℝ := sorry -- placeholder for area calculation

theorem pinching_area_preservation (n : ℕ) (h₀ : n ≥ 6)
  (h₁ : ∀ k ≥ 6, ∃ Pₖ : List ℝ, is_convex Pₖ ∧ area Pₖ > 1/2)
  (h₂ : area (List ℝ.repeat 6 (n+1)) = 1) :
  ∃ P_n : List ℝ, is_convex P_n ∧ area P_n > 1/2 :=
sorry

end pinching_area_preservation_l568_568816


namespace daily_profit_at_45_selling_price_for_1200_profit_l568_568193

-- Definitions for the conditions
def cost_price (p: ℝ) : Prop := p = 30
def initial_sales (p: ℝ) (s: ℝ) : Prop := p = 40 ∧ s = 80
def sales_decrease_rate (r: ℝ) : Prop := r = 2
def max_selling_price (p: ℝ) : Prop := p ≤ 55

-- Proof for Question 1
theorem daily_profit_at_45 (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate) :
  (price = 45) → profit = 1050 :=
by sorry

-- Proof for Question 2
theorem selling_price_for_1200_profit (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate)
  (h_max_price : ∀ p, max_selling_price p → p ≤ 55) :
  profit = 1200 → price = 50 :=
by sorry

end daily_profit_at_45_selling_price_for_1200_profit_l568_568193


namespace ammonium_iodide_molecular_weight_l568_568614

theorem ammonium_iodide_molecular_weight :
  let N := 14.01
  let H := 1.008
  let I := 126.90
  let NH4I_weight := (1 * N) + (4 * H) + (1 * I)
  NH4I_weight = 144.942 :=
by
  -- The proof will go here
  sorry

end ammonium_iodide_molecular_weight_l568_568614


namespace root_in_interval_l568_568703

def f (x : ℝ) : ℝ := log x + 2 * x - 6

theorem root_in_interval (h_mono : ∀ x y : ℝ, x > 0 → y > 0 → x < y → f x < f y) :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end root_in_interval_l568_568703


namespace prism_lateral_surface_area_l568_568841

theorem prism_lateral_surface_area (α β b : ℝ) 
  (AB AC : ℝ) 
  (h1 : AB = AC) 
  (h2 : |∠ABC| = α) 
  (h3 : |CD| = b) 
  (h4 : ∠DCA = β) :
  let P : ℝ := 2 * b * Math.cos β * (1 + Math.cos α)
  let H : ℝ := 2 * b * Math.sin β
  4 * b^2 * Math.sin (2 * β) * Math.cos^2 (α / 2) :=
sorry

end prism_lateral_surface_area_l568_568841


namespace compute_fraction_power_mul_l568_568986

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end compute_fraction_power_mul_l568_568986


namespace number_of_valid_three_digit_numbers_l568_568365

-- Define the set of digits available
def digits : Finset ℕ := {0, 1, 2, 3, 4}

-- Define the condition that a number must be three digits
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the condition that a number must have no repeated digits
def has_no_repeated_digits (n : ℕ) : Prop :=
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10)

-- Define the condition that the hundreds place cannot be 0
def hundreds_place_nonzero (n : ℕ) : Prop :=
  (n / 100) ≠ 0

-- Combining all conditions together
def valid_three_digit_numbers : Finset ℕ :=
  (Finset.Icc 100 999).filter (λ n, has_no_repeated_digits n ∧ hundreds_place_nonzero n ∧ ∀ d ∈ digits, d ∈ {n / 100, (n / 10) % 10, n % 10})

-- The theorem we need to prove
theorem number_of_valid_three_digit_numbers : Finset.card valid_three_digit_numbers = 48 := by
  sorry

end number_of_valid_three_digit_numbers_l568_568365


namespace minimum_days_to_pay_back_l568_568050

theorem minimum_days_to_pay_back (x : ℕ) : 
  (50 + 5 * x ≥ 150) → x = 20 :=
sorry

end minimum_days_to_pay_back_l568_568050


namespace train_cross_time_proof_l568_568146

noncomputable def train_cross_time_opposite (L : ℝ) (v1 v2 : ℝ) (t_same : ℝ) : ℝ :=
  let speed_same := (v1 - v2) * (5/18)
  let dist_same := speed_same * t_same
  let speed_opposite := (v1 + v2) * (5/18)
  dist_same / speed_opposite

theorem train_cross_time_proof : 
  train_cross_time_opposite 69.444 50 40 50 = 5.56 :=
by
  sorry

end train_cross_time_proof_l568_568146


namespace sum_of_interior_diagonals_is_12sqrt7_l568_568575

theorem sum_of_interior_diagonals_is_12sqrt7 
  (x y z : ℝ)
  (h1 : x + y + z = 13)
  (h2 : 2 * (x * y + y * z + z * x) = 106) :
  4 * Real.sqrt (x^2 + y^2 + z^2) = 12 * Real.sqrt 7 :=
by
  have h3 : (x + y + z) ^ 2 = 13 ^ 2 := by rw [h1]
  have h4 : 13 ^ 2 = x^2 + y^2 + z^2 + 2 * (x * y + y * z + z * x) := by rw [sqr_add_sub_mixed x y z]
  have h5 : 169 = x^2 + y^2 + z^2 + 106 := by rw [h2] at h4; exact h4
  have h6 : x^2 + y^2 + z^2 = 63 := by linarith [h5]
  exact eq.trans (congr_arg (HasMul.mul 4) (congr_arg Real.sqrt h6)) (by norm_num)

end sum_of_interior_diagonals_is_12sqrt7_l568_568575


namespace equivalent_single_discount_l568_568201

-- Define the price and discount percentages
def regular_price : ℝ := 50
def first_discount_percentage : ℝ := 0.30
def second_discount_percentage : ℝ := 0.15

-- Define the successive discount calculations
def single_discount_equivalent : ℝ := 1 - ((regular_price * (1 - first_discount_percentage) * (1 - second_discount_percentage)) / regular_price)

-- Prove the single discount equivalent is 40.5%
theorem equivalent_single_discount : single_discount_equivalent = 0.405 := 
by 
  -- Skip the proof
  sorry

end equivalent_single_discount_l568_568201


namespace bus_trip_speed_l568_568187

theorem bus_trip_speed :
  ∃ v : ℝ, v > 0 ∧ (660 / v - 1 = 660 / (v + 5)) ∧ v = 55 :=
by
  sorry

end bus_trip_speed_l568_568187


namespace rectangle_area_l568_568823

open Classical

noncomputable def point := {x : ℝ × ℝ // x.1 >= 0 ∧ x.2 >= 0}

structure Triangle :=
  (X Y Z : point)

structure Rectangle :=
  (P Q R S : point)

def height_from (t : Triangle) : ℝ :=
  8

def xz_length (t : Triangle) : ℝ :=
  15

def ps_on_xz (r : Rectangle) (t : Triangle) : Prop :=
  r.S.val.1 = r.P.val.1 ∧ r.S.val.1 = t.X.val.1 ∧ r.S.val.2 = 0 ∧ r.P.val.2 = 0

def pq_is_one_third_ps (r : Rectangle) : Prop :=
  dist r.P.1 r.Q.1 = (1/3) * dist r.P.1 r.S.1

theorem rectangle_area : ∀ (R : Rectangle) (T : Triangle),
  height_from T = 8 → xz_length T = 15 → ps_on_xz R T → pq_is_one_third_ps R →
  (dist R.P.1 R.Q.1) * (dist R.P.1 R.S.1) = 4800/169 :=
by
  intros
  sorry

end rectangle_area_l568_568823


namespace number_of_vegetables_per_plant_is_correct_l568_568075

noncomputable def vegetables_per_plant {tomato_plants_survived pepper_plants_survived eggplant_plants_survived total_vegetables : ℕ} 
  (h1 : tomato_plants_survived = 3)
  (h2 : pepper_plants_survived = 3)
  (h3 : eggplant_plants_survived = 2)
  (h4 : total_vegetables = 56) : ℕ :=
total_vegetables / (tomato_plants_survived + pepper_plants_survived + eggplant_plants_survived)

theorem number_of_vegetables_per_plant_is_correct :
  let tomato_plants_survived := 3 in
  let pepper_plants_survived := 3 in
  let eggplant_plants_survived := 2 in
  let total_vegetables := 56 in
  vegetables_per_plant rfl rfl rfl rfl = 7 :=
by
  sorry

end number_of_vegetables_per_plant_is_correct_l568_568075


namespace length_of_segment_l568_568160

theorem length_of_segment : 
  ∀ x : ℝ, abs (x - real.cbrt 27) = 4 → abs ((real.cbrt 27) + 4 - ((real.cbrt 27) - 4)) = 8 :=
by 
  intro x h
  sorry

end length_of_segment_l568_568160


namespace maximal_area_convex_quadrilateral_maximal_area_convex_ngon_l568_568546

theorem maximal_area_convex_quadrilateral 
  (angles : Fin 4 → ℝ) 
  (perimeter : ℝ) 
  (is_convex : ∀ i j k l : Fin 4, i ≠ j → j ≠ k → k ≠ l → l ≠ i → ∠(i j k) + ∠(j k l) + ∠(k l i) + ∠(l i j) = 2 * π)
  (exists_inscribed_circle : ∃ r : ℝ, ∀ i : Fin 4, dist (center_incircle i) (side i) = r) :
  ∃ (Q : Quadrilateral),
  Q.angles = angles ∧ Q.perimeter = perimeter ∧ Q.has_largest_area :=
sorry

theorem maximal_area_convex_ngon 
  (n : ℕ) 
  (angles : Fin n → ℝ) 
  (perimeter : ℝ) 
  (is_convex : ∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ∠(i j k) + ∠(j k i) = (n - 2) * π / n)
  (exists_inscribed_circle : ∃ r : ℝ, ∀ i : Fin n, dist (center_incircle i) (side i) = r) :
  ∃ (P : Polygon),
  P.angles = angles ∧ P.perimeter = perimeter ∧ P.has_largest_area :=
sorry

end maximal_area_convex_quadrilateral_maximal_area_convex_ngon_l568_568546


namespace functional_equation_solution_l568_568030

theorem functional_equation_solution {f : ℝ → ℝ} :
  (∀ x y : ℝ, f(x + f(x + y)) + f(x * y) = x + f(x + y) + y * f(x)) →
  (f = id ∨ f = (λ x, 2 - x)) := 
by
  intros h
  sorry

end functional_equation_solution_l568_568030


namespace dodecagon_diagonals_l568_568589

theorem dodecagon_diagonals :
  ∀ n : ℕ, n = 12 → (n * (n - 3)) / 2 = 54 :=
begin
  intros n hn,
  rw hn,
  norm_num,
end

end dodecagon_diagonals_l568_568589


namespace circle_fixed_point_l568_568822

-- Definitions and assumptions
variable (O : Point)
variable (ℓ ℓ₁ ℓ₂ : Ray O)
variable (acute_angle : acute (angle ℓ ℓ₂))
variable (ℓ₁_inside : angle ℓ₁ ℓ < angle ℓ₁ ℓ₂)
variable (F : Point)
variable (L : Point)

-- Points on respective rays
variable (F_on_ℓ : F ∈ ℓ)
variable (L_on_ℓ : L ∈ ℓ)

-- Define points L₁ and L₂
variable (L₁_touch_ℓ₁ : tangential_point L₁ ℓ₁ L)
variable (L₂_touch_ℓ₂ : tangential_point L₂ ℓ₂ L)

-- Define the circle passing through F, L₁, L₂
variable (circle_through_FL₁L₂ : circle ℓ₁.touch L₁ [L,F,L₂])

theorem circle_fixed_point :
  ∃ (F' : Point), (F' ≠ F ∧ ∀ (L : Point), L_lin L ℓ → circle_through_FL₁L₂ L F L₁ L₂) :=
sorry

end circle_fixed_point_l568_568822


namespace teacher_city_subject_l568_568518

theorem teacher_city_subject :
  ∀ (teacher city subject : Type) 
    (Zhang Li Wang : teacher) 
    (Beijing Shanghai Shenzhen : city) 
    (Math Chinese English : subject)
    (from : teacher → city) 
    (teaches : teacher → subject),

    -- conditions
    (from Zhang ≠ Beijing) ∧ (from Li ≠ Shanghai) ∧
    (∀ t, from t = Beijing → teaches t ≠ English) ∧
    (∀ t, from t = Shanghai → teaches t = Math) ∧
    (teaches Li ≠ Chinese) →
    
    -- conclusion
    (from Zhang = Shanghai ∧ teaches Zhang = Math) ∧
    (from Wang = Beijing ∧ teaches Wang = Chinese) ∧
    (from Li = Shenzhen ∧ teaches Li = English) :=
by
  intros teacher city subject Zhang Li Wang Beijing Shanghai Shenzhen Math Chinese English from teaches
  intros h
  sorry

end teacher_city_subject_l568_568518


namespace correct_pronouns_usage_l568_568057

/- 
  Our neighbors gave us a baby bird yesterday that hurt itself when it fell from its nest.
  
  To verify the correctness of the chosen pronouns, we need to show:
  1. For the first blank: the structure "give sb sth" requires the personal pronoun "us".
  2. For the second blank: the reflexive pronoun "itself" is appropriate for a baby bird injuring itself.
-/

theorem correct_pronouns_usage : 
  (∃ (us : String) (itself : String), us = "us" ∧ itself = "itself") :=
begin
  use ["us", "itself"],
  split;
  refl
end

end correct_pronouns_usage_l568_568057


namespace eraser_price_correct_l568_568593

noncomputable def price_of_eraser (pencil_price : ℝ) : ℝ :=
  (1 / 2) * pencil_price

theorem eraser_price_correct (
  pencil_price eraser_price : ℝ) 
  (bundle_price : ℝ)
  (sold_bundles : ℕ) 
  (store_revenue : ℝ) 
  (discount : ℝ) 
  (tax : ℝ)
  (h1 : eraser_price = price_of_eraser pencil_price)
  (h2 : bundle_price = pencil_price + 2 * eraser_price)
  (h3 : sold_bundles = 20)
  (h4 : discount = 0.30)
  (h5 : tax = 0.10)
  (h6 : store_revenue = 80) :
  eraser_price = 1.30 :=
by
  let original_bundle_price := 2 * pencil_price
  let discounted_bundle_price := original_bundle_price * (1 - discount)
  let total_price_before_tax := sold_bundles * discounted_bundle_price
  let total_price_after_tax := total_price_before_tax * (1 + tax)
  have h_total_price : total_price_after_tax = store_revenue := by
    rw [h3, h4, h5, h6]
    sorry
  have h_correct_price : 30.8 * pencil_price = 80 := by sorry
  have h_p : pencil_price = 2.5974 := by
    rw [h_correct_price]
    sorry
  have h_e : eraser_price = 1.2987 := by
    rw [h1, h_p]
    sorry
  have h_approx : eraser_price ≈ 1.30 := by
    linarith [h_e]
  exact h_approx

end eraser_price_correct_l568_568593


namespace number_of_possible_second_largest_values_l568_568204

theorem number_of_possible_second_largest_values
  (a : Fin 6 → ℕ)
  (h_sorted : ∀ i j, (i : ℕ) ≤ j → a i ≤ a j)
  (h_mean : ∑ i, a i = 66)
  (h_range : a 5 - a 0 = 24)
  (h_mode : ∃ i, ∃ j ≠ i, a i = 9 ∧ a j = 9)
  (h_median : a 2 = 9 ∧ a 3 = 9) :
  ∃ n : ℕ, n = (number_of_possible_second_largest_values_implementation a) := by
  sorry

end number_of_possible_second_largest_values_l568_568204


namespace common_chord_properties_l568_568245

-- Definition of a trapezoid with given properties
structure Trapezoid (α : Type*) :=
  (A B C D : α) -- vertices of the trapezoid
  (AB_parallel_CD : function.extends (line A B) (line C D)) -- bases are parallel
  (non_parallel_sides : ¬ function.extends (line A D) (line B C)) -- non-parallel sides
  (AC_diameter_circle : circle_with_diameter A C)
  (BD_diameter_circle : circle_with_diameter B D)

-- Prove the common chord properties
theorem common_chord_properties (α : Type*) [Euclidean_plane α] (T : Trapezoid α) :
  ∃ P, is_perpendicular_to_bases (common_chord_of_diameter_circles T.AC_diameter_circle T.BD_diameter_circle) ∧ passes_through_intersection_of_non_parallel_sides T.AC_diameter_circle T.BD_diameter_circle P :=
sorry

end common_chord_properties_l568_568245


namespace f_2019_is_zero_l568_568028

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_non_negative
  (x : ℝ) : 0 ≤ f x

axiom f_satisfies_condition
  (a b c : ℝ) : f (a^3) + f (b^3) + f (c^3) = 3 * f a * f b * f c

axiom f_one_not_one : f 1 ≠ 1

theorem f_2019_is_zero : f 2019 = 0 := 
  sorry

end f_2019_is_zero_l568_568028


namespace cosine_angle_BHD_l568_568747

theorem cosine_angle_BHD (CD DH HG DG CH HB : ℝ) (BD: ℝ) (h_CD : CD = 2) (h_DH : DH = 2) 
(h_HG : HG = Real.sqrt 3) (h_DG : DG = 1) (h_CH : CH = Real.sqrt 3) 
(h_HB: HB = 3) (h_BD: BD = Real.sqrt 13) : 
Real.cos (angle B H D) = 3 / Real.sqrt 13 :=
by {
  -- Skipping the steps to complete the proof
  sorry  
}

end cosine_angle_BHD_l568_568747


namespace solution_set_l568_568322

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f y ≤ f x

variables (f : ℝ → ℝ)
variables (h1: odd_function f)
variables (h2: monotone_decreasing f (-∞) 0)
variables (h3: f 2 = 0)

theorem solution_set :
  {x : ℝ | f (x - 1) > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end solution_set_l568_568322


namespace inequality_proof_l568_568063

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_proof_l568_568063


namespace lily_geese_count_l568_568821

-- Define variables for the conditions
variables {G : ℕ} -- the number of geese Lily bought
def lily_ducks : ℕ := 20
def rayden_ducks : ℕ := 3 * lily_ducks
def rayden_geese : ℕ := 4 * G
def lily_total : ℕ := lily_ducks + G
def rayden_total : ℕ := rayden_ducks + rayden_geese

-- State the main theorem
theorem lily_geese_count :
  rayden_total = lily_total + 70 ↔ G = 10 :=
sorry

end lily_geese_count_l568_568821


namespace calculate_binom_l568_568970

theorem calculate_binom : 2 * Nat.choose 30 3 = 8120 := 
by 
  sorry

end calculate_binom_l568_568970


namespace chocolate_bars_remaining_l568_568013

theorem chocolate_bars_remaining (total_bars sold_week1 sold_week2 : ℕ) (h_total : total_bars = 18) (h_sold1 : sold_week1 = 5) (h_sold2 : sold_week2 = 7) : total_bars - (sold_week1 + sold_week2) = 6 :=
by {
  sorry
}

end chocolate_bars_remaining_l568_568013


namespace part_1_part_2_l568_568702

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.sin (Real.pi / 4 + x / 2) ^ 2 + Real.cos (2 * x)

-- Part (1) proof statement
theorem part_1 (ω : ℝ) (hω : ω > 0) :
  (∀ x ∈ set.Icc (-Real.pi / 2) (2 * Real.pi / 3), deriv (λ x, f (ω * x)) x > 0) ↔ ω ∈ set.Ioc 0 (3 / 4) := sorry

-- Part (2) proof statement
theorem part_2 (m : ℝ) :
  (∀ x ∈ set.Icc (Real.pi / 6) (2 * Real.pi / 3), abs (f x - m) < 2) ↔ m ∈ set.Ioo 1 4 := sorry

end part_1_part_2_l568_568702


namespace incorrect_statements_count_l568_568501

theorem incorrect_statements_count :
  let s1 := "Every proposition has a converse"
  let s2 := "If the original proposition is false, then its converse is also false"
  let s3 := "Every theorem has a converse"
  let s4 := "If the original proposition is true, then its converse is also true"
  -- Assume the correctness analysis in the problem:
  s1_is_correct : true,
  s2_is_correct : false,
  s3_is_correct : false,
  s4_is_correct : false
  -- Prove:
  in
  s1_is_correct && not s2_is_correct && not s3_is_correct && not s4_is_correct -> 
  3 = (nat.add 1 (nat.add 1 1)) -- 3 is the count of false statements
:= by
  -- Using 'by' here as a placeholder to specify the proof is bypassed
  sorry

end incorrect_statements_count_l568_568501


namespace max_value_of_sides_l568_568453

theorem max_value_of_sides (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h4 : a^2 + b^2 > c^2) (h5 : a + b > c) (hS : 0.5 * c^2 = abs ((1 / 2) * a * b * sin C))
    (hab_sqrt2 : a * b = sqrt 2) : (a^2 + b^2 + c^2 ≤ 4) :=
sorry

end max_value_of_sides_l568_568453


namespace ellipse_equation_chord_length_l568_568321

-- Define the properties of the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

-- Define the focal distance and sum of distances properties
def focal_distance (a : ℝ) : ℝ := 2 * Real.sqrt 6
def sum_of_distances (a : ℝ) : ℝ := 6

-- Prove the equation of the ellipse given the conditions
theorem ellipse_equation (a b : ℝ) :
  focal_distance a = 2 * Real.sqrt 6 →
  2 * a = 6 →
  ellipse a b 0 0 → 
  b^2 = a^2 - (Real.sqrt 6)^2 →
  (x y : ℝ) : ellipse 3 (Real.sqrt 3) x y → x^2 / 9 + y^2 / 3 = 1 := by sorry

-- Define the chord length given the ellipse equation and the line
def line (x y : ℝ) : Prop := y = x + 1

-- Prove the length of the chord
theorem chord_length (x1 x2 y1 y2 : ℝ) :
  ellipse 3 (Real.sqrt 3) x1 y1 →
  ellipse 3 (Real.sqrt 3) x2 y2 →
  line x1 y1 →
  line x2 y2 →
  x1 + x2 = 3 / 2 →
  x1 * x2 = -3 / 2 →
  (|AB| = Real.sqrt 66 / 2) := by sorry

end ellipse_equation_chord_length_l568_568321


namespace find_angle_between_vectors_l568_568786

open Real InnerProductSpace

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2)) (ha : ‖a‖ ≠ 0) (hb : ‖b‖ ≠ 0)
  (h1 : ‖a‖ = 2 * ‖b‖) (h2 : ‖a + b‖ = 2 * ‖b‖) : ℝ :=
  let cos_theta := (inner a b) / (‖a‖ * ‖b‖) in
  real.arccos cos_theta

theorem find_angle_between_vectors {a b : EuclideanSpace ℝ (Fin 2)}
  (ha : ‖a‖ ≠ 0) (hb : ‖b‖ ≠ 0) 
  (h1 : ‖a‖ = 2 * ‖b‖) (h2 : ‖a + b‖ = 2 * ‖b‖) :
  180/π * angle_between_vectors a b ha hb h1 h2 ≈ 104.48 :=
by
  sorry

end find_angle_between_vectors_l568_568786


namespace number_of_children_l568_568836

-- Define the conditions
variables (m f g x : ℕ) (xy : ℕ)
variables (m f g xy x : ℕ)
axiom (age_father : f = 50)
axiom (age_grandfather : g = 70)
axiom (average_family : (m + f + g + xy) / (3 + x) = 25)
axiom (average_others : (m + g + xy) / (2 + x) = 20)

-- State the theorem to prove the number of children is 3
theorem number_of_children : x = 3 :=
by
  sorry

end number_of_children_l568_568836


namespace shift_right_three_units_l568_568415

theorem shift_right_three_units (x : ℝ) : (λ x, -2 * x) (x - 3) = -2 * x + 6 :=
by
  sorry

end shift_right_three_units_l568_568415


namespace proof_problem_l568_568311

-- Conditions
def p (x : ℝ) : Prop := abs (4 - x) ≤ 6
def q (x : ℝ) (m : ℝ) : Prop := m > 0 ∧ (x^2 - 2 * x + 1 ≤ 0)

-- Objective
theorem proof_problem (m : ℝ) : m ≥ 9 :=
begin
  -- sorry means the proof is missing; statement only.
  sorry
end

end proof_problem_l568_568311


namespace number_of_solutions_l568_568657

-- Define the equation and the constraints
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + z = 800

def positive_integer (n : ℕ) : Prop := n > 0

-- The main theorem statement
theorem number_of_solutions : ∃ s, s = 127 ∧ ∀ (x y z : ℕ), positive_integer x → positive_integer y → positive_integer z → equation x y z → s = 127 :=
by
  sorry

end number_of_solutions_l568_568657


namespace increasing_iff_a_ge_one_l568_568331

-- Declare the variable 'a' and an arbitrary 'x' in ℝ
variable (a : ℝ)
variable (x : ℝ)

-- Define the function y
def y (x : ℝ) := Real.sin x + a * x

-- Define the derivative y'
def y' (x : ℝ) := Real.cos x + a

-- Theorem stating that if y is increasing on ℝ, then a ≥ 1
theorem increasing_iff_a_ge_one (h : ∀ x, y' a x ≥ 0) : a ≥ 1 := 
  sorry

end increasing_iff_a_ge_one_l568_568331


namespace problem1_problem2_problem3_l568_568503

namespace ProofProblems

-- Problem 1
theorem problem1 (x : ℝ) (h : x = 2 - sqrt 7) : x^2 - 4 * x + 5 = 8 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : 2 * x = sqrt 5 + 1) : x^3 - 2 * x^2 = -1 :=
by sorry

-- Problem 3
theorem problem3 (a : ℝ) (h : a^2 = sqrt (a^2 + 10) + 3) : a^2 + 1 / a^2 = sqrt 53 :=
by sorry

end ProofProblems

end problem1_problem2_problem3_l568_568503


namespace incorrect_expression_l568_568443

variable (D : ℚ) (P Q : ℕ) (r s : ℕ)

-- D represents a repeating decimal.
-- P denotes the r figures of D which do not repeat themselves.
-- Q denotes the s figures of D which repeat themselves.

theorem incorrect_expression :
  10^r * (10^s - 1) * D ≠ Q * (P - 1) :=
sorry

end incorrect_expression_l568_568443


namespace cosine_sine_difference_identity_l568_568632

theorem cosine_sine_difference_identity :
  (Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
  - Real.sin (255 * Real.pi / 180) * Real.sin (165 * Real.pi / 180)) = 1 / 2 := by
  -- Proof goes here
  sorry

end cosine_sine_difference_identity_l568_568632


namespace B_investment_is_72000_l568_568225

noncomputable def A_investment : ℝ := 27000
noncomputable def C_investment : ℝ := 81000
noncomputable def C_profit : ℝ := 36000
noncomputable def total_profit : ℝ := 80000

noncomputable def B_investment : ℝ :=
  let total_investment := (C_investment * total_profit) / C_profit
  total_investment - A_investment - C_investment

theorem B_investment_is_72000 :
  B_investment = 72000 :=
by
  sorry

end B_investment_is_72000_l568_568225


namespace statement_C_not_true_l568_568905

theorem statement_C_not_true (a b c : ℝ) (h1 : a > b) (h2 : c = 0) : ac^2 ≤ bc^2 :=
by
  sorry

end statement_C_not_true_l568_568905


namespace sum_series_l568_568618

theorem sum_series (h : ∀ n : ℤ, (-1)^(-n : ℤ) = 1 / (-1)^n) : 
  (∑ n in finset.range 25, 2 * (-1)^(n - 12)) = 0 := 
by
  let hs_even : ∀ n : ℤ, even n → (-1)^n = 1 := 
    by 
      intros n heven 
      exact pow_even_neg_one n heven
  
  let hs_odd : ∀ n : ℤ, odd n → (-1)^n = -1 := 
    by 
      intros n hodd 
      exact pow_odd_neg_one n hodd

  let numerator := finset.sum (finset.range 13) (λ n, 2)
  let denominator := finset.sum (finset.range 13) (λ n, -2)
  
  calc numerator + denominator = 0 : 
    by
      rw [numerator, denominator]
      exact finset.sum_range_add_sum_range _
  sorry

end sum_series_l568_568618


namespace compute_expression_l568_568992

-- Definition of the operation "minus the reciprocal of"
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement to prove the given problem
theorem compute_expression :
  ((diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5))) = -71 / 380 := 
sorry

end compute_expression_l568_568992


namespace max_decimal_of_four_digit_binary_l568_568106

theorem max_decimal_of_four_digit_binary : 
    ∃ n : ℕ, (n = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0) ∧ n = 15 :=
by
  existsi 15
  split
  {
    calc
      1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 = 8 + 4 + 2 + 1 : by norm_num
      ... = 15 : by norm_num
  }
  {
    refl
  }

end max_decimal_of_four_digit_binary_l568_568106


namespace number_of_boys_l568_568832

def trees := 29
def leftover_trees := 2
def watered_trees := trees - leftover_trees

theorem number_of_boys :
  ∃ B : ℕ, B ≠ 0 ∧ watered_trees % B = 0 ∧ B = 3 :=
begin
  sorry
end

end number_of_boys_l568_568832


namespace Razorback_shop_profit_l568_568838

def profit_per_item := 
{ jerseys := 5, tshirts := 15, hats := 8, hoodies := 25 }

def items_sold := 
{ jerseys := 64, tshirts := 20, hats := 30, hoodies := 10 }

def discount_rate := 0.10
def vendor_fee := 50

def total_profit (items : {jerseys : Nat, tshirts : Nat, hats : Nat, hoodies : Nat})
  (price : {jerseys : Nat, tshirts : Nat, hats : Nat, hoodies : Nat}) : Nat :=
  (items.jerseys * price.jerseys) + 
  (items.tshirts * price.tshirts) + 
  (items.hats * price.hats) + 
  (items.hoodies * price.hoodies)

theorem Razorback_shop_profit :
  let total_before_discount := total_profit items_sold profit_per_item
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let final_total := total_after_discount - vendor_fee
  final_total = 949 := by
    sorry

end Razorback_shop_profit_l568_568838


namespace correct_multiplication_l568_568168

variable {a : ℕ} -- Assume 'a' to be a natural number for simplicity in this example

theorem correct_multiplication : (3 * a) * (4 * a^2) = 12 * a^3 := by
  sorry

end correct_multiplication_l568_568168


namespace find_x_l568_568355

-- Definitions used in conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Main statement of the problem to be proved
theorem find_x (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = -1) : x = -1 / 5 :=
by {
  sorry
}

end find_x_l568_568355


namespace ordered_pairs_count_l568_568359

theorem ordered_pairs_count :
  ∃ n, (∀ (A B : ℕ), (A * B = 24) → (A > 0 ∧ B > 0) → n = 8) :=
begin
  use 8,
  intros A B h h_pos,
  sorry
end

end ordered_pairs_count_l568_568359


namespace min_value_one_over_a_plus_nine_over_b_l568_568330

theorem min_value_one_over_a_plus_nine_over_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  16 ≤ (1 / a) + (9 / b) :=
sorry

end min_value_one_over_a_plus_nine_over_b_l568_568330


namespace sequencing_inequality_l568_568032

noncomputable def x_k (m n : ℕ) : ℕ → ℝ
| 0 => real.sqrt m
| k + 1 => real.sqrt (m + x_k n m k)

noncomputable def y_k (m n : ℕ) : ℕ → ℝ
| 0 => real.sqrt n
| k + 1 => real.sqrt (n + y_k n m k)

theorem sequencing_inequality (m n : ℕ) (h : m > n) : ∀ k : ℕ, x_k m n k > y_k m n k := 
sorry

end sequencing_inequality_l568_568032


namespace will_catches_16_catfish_l568_568542

theorem will_catches_16_catfish (C : ℝ) :
  (let W_eels := 10 in
   let H_trout := 3 * C in
   let H_kept_trout := H_trout / 2 in
   C + W_eels + H_kept_trout = 50) → C = 16 :=
by
  intros h
  sorry

end will_catches_16_catfish_l568_568542


namespace perimeter_equal_if_base_eq_height_base_eq_height_if_perimeter_equal_l568_568068

variables (ABC : Triangle) (h : ℝ) (inscribed_rectangles : List (Rectangle))

-- Assume the conditions 
axiom base_eq_height
  (h_base_eq_height : ABC.base = h) 
  (vertices_on_sides : ∀ R ∈ inscribed_rectangles, 
    ∃ A B C D, A ∈ ABC.AC ∧ B ∈ ABC.BC ∧ C ∈ ABC.CB ∧ D ∈ ABC.CB ∧ sides AB AC BC AD) 

-- Proof Problem 1: Given the base = height, prove the perimeters of inscribed rectangles are equal.
theorem perimeter_equal_if_base_eq_height
  (h_base_eq_height : ABC.base = h) 
  (vertices_on_sides : ∀ R ∈ inscribed_rectangles, 
    ∃ A B C D, A ∈ ABC.AB ∧ B ∈ ABC.AC ∧ C ∈ ABC.CB ∧ D ∈ ABC.BC) 
  : ∃ p, ∀ R ∈ inscribed_rectangles, Rectangle.perimeter R = p := 
sorry

-- Proof Problem 2: Given the perimeters are equal, prove the base = height.
theorem base_eq_height_if_perimeter_equal
  (p : ℝ) 
  (vertices_on_sides : ∀ R ∈ inscribed_rectangles, 
    ∃ A B C D, A ∈ ABC.AB ∧ B ∈ ABC.AC ∧ C ∈ ABC.CB ∧ D ∈ ABC.BC)
  (h_perimeter_equal : ∀ R ∈ inscribed_rectangles, Rectangle.perimeter R = p) 
  : ABC.base = h := 
sorry

end perimeter_equal_if_base_eq_height_base_eq_height_if_perimeter_equal_l568_568068


namespace tetrahedron_angle_equal_l568_568298

theorem tetrahedron_angle_equal {O A B C D : Type} [MetricSpace O] 
  (d : ℝ) (distance_OA : Metric.dist O A = d)
  (distance_OB : Metric.dist O B = d)
  (distance_OC : Metric.dist O C = d)
  (distance_OD : Metric.dist O D = d)
  (angle_equal : ∀ {X Y Z W : Type}, Metric.dist X Y = d → Metric.dist Y Z = d → Metric.dist Z W = d → Metric.dist W X = d →
    ∀ {θ : ℝ}, θ ≠ 0 → θ = ∠X Y Z → θ = ∠Y Z W → θ = ∠Z W X → θ = ∠W X Y) : 
  ∃ θ : ℝ, θ = real.arccos (-1 / 3) :=
begin
  sorry
end

end tetrahedron_angle_equal_l568_568298


namespace sum_of_integers_k_l568_568284

theorem sum_of_integers_k (k : ℕ) (h1 : nat.choose 30 6 + nat.choose 30 7 = nat.choose 31 k)
  (h2 : nat.choose 30 6 + nat.choose 30 7 = nat.choose 31 7) : 
  k = 7 ∨ k = 24 → 7 + 24 = 31 :=
by
  intros hk
  sorry -- proof here

end sum_of_integers_k_l568_568284


namespace cone_base_circumference_l568_568196

theorem cone_base_circumference (radius : ℝ) (angle : ℝ) (c_base : ℝ) :
  radius = 6 ∧ angle = 180 ∧ c_base = 6 * Real.pi →
  (c_base = (angle / 360) * (2 * Real.pi * radius)) :=
by
  intros h
  rcases h with ⟨h_radius, h_angle, h_c_base⟩
  rw [h_radius, h_angle]
  norm_num
  sorry

end cone_base_circumference_l568_568196


namespace range_of_a_l568_568713

-- Conditions for sets A and B
def SetA := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def SetB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 2}

-- Main statement to show that A ∪ B = A implies the range of a is [-2, 0]
theorem range_of_a (a : ℝ) : (SetB a ⊆ SetA) → (-2 ≤ a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l568_568713


namespace Kimberley_collected_10_pounds_l568_568016

variable (K H E total : ℝ)

theorem Kimberley_collected_10_pounds (h_total : total = 35) (h_Houston : H = 12) (h_Ela : E = 13) :
    K + H + E = total → K = 10 :=
by
  intros h_sum
  rw [h_Houston, h_Ela] at h_sum
  linarith

end Kimberley_collected_10_pounds_l568_568016


namespace pop_spent_l568_568078

def Pop (x : ℕ) := x
def Crackle (x : ℕ) := 3 * x
def Snap (x : ℕ) := 6 * x
def total_expenditure (x : ℕ) := Pop x + Crackle x + Snap x

theorem pop_spent (x : ℕ) : total_expenditure x = 150 → x = 15 := by
  intro h
  have h_exp : 10 * x = 150 := by
    rw [total_expenditure, Pop, Crackle, Snap] at h
    simp [total_expenditure, Pop, Crackle, Snap] at h
    exact h
  exact eq_of_mul_eq_mul_right (by norm_num) h_exp

# Testing the theorem
example : total_expenditure 15 = 150 := by
  simp [total_expenditure, Pop, Crackle, Snap]
  norm_num

-- Uncomment the following line to see an error if the theorem is incorrect
-- #eval pop_spent 15 (by norm_num)

end pop_spent_l568_568078


namespace probability_red_balls_by_4th_draw_l568_568512

theorem probability_red_balls_by_4th_draw :
  let total_balls := 10
  let red_prob := 2 / total_balls
  let white_prob := 1 - red_prob
  (white_prob^3) * red_prob = 0.0434 := sorry

end probability_red_balls_by_4th_draw_l568_568512


namespace probability_cd_l568_568558

theorem probability_cd (P_A P_B : ℚ) (h1 : P_A = 1/4) (h2 : P_B = 1/3) :
  (1 - P_A - P_B = 5/12) :=
by
  -- Placeholder for the proof
  sorry

end probability_cd_l568_568558


namespace quad_root_when_x_is_neg4_l568_568899

theorem quad_root_when_x_is_neg4 : ∀ x : ℝ, x = -4 → sqrt (1 - 2 * x) = 3 := by
  intro x hx
  rw hx
  rw [mul_neg, ←sub_add]
  simp [sqrt, sq]
  sorry

end quad_root_when_x_is_neg4_l568_568899


namespace sin_sum_bound_l568_568065

theorem sin_sum_bound (x : ℝ) : 
  |(Real.sin x) + (Real.sin (Real.sqrt 2 * x))| < 2 - 1 / (100 * (x^2 + 1)) :=
by sorry

end sin_sum_bound_l568_568065


namespace power_of_two_divisor_l568_568256

theorem power_of_two_divisor {n : ℕ} (h_pos : n > 0) : 
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) → ∃ r : ℕ, n = 2^r :=
by
  sorry

end power_of_two_divisor_l568_568256


namespace required_rate_correct_l568_568957

-- Define the given conditions
variable (total_investment : ℝ) (investment_1 : ℝ) (rate_1 : ℝ) (investment_2 : ℝ) (rate_2 : ℝ) (desired_income : ℝ)

-- Define the invested amount and investment rates
def investments := {total := 12000, 
                    part1 := 5000, 
                    rate1 := 3.5 / 100, 
                    part2 := 4000, 
                    rate2 := 4.5 / 100, 
                    income_goal := 600}

-- Define the income generated from the first two investments
def income_1 : ℝ := investments.part1 * investments.rate1
def income_2 : ℝ := investments.part2 * investments.rate2
def total_income : ℝ := income_1 + income_2

-- Define the remaining investment
def remaining_investment : ℝ := investments.total - investments.part1 - investments.part2

-- Define the additional income needed
def additional_income_needed : ℝ := investments.income_goal - total_income

-- Equation to find the rate of return needed for the remaining investment to achieve the total income
def required_rate_of_return : ℝ := (additional_income_needed * 100) / remaining_investment

-- Lean proof statement
theorem required_rate_correct :
  required_rate_of_return = 8.2 := by sorry

end required_rate_correct_l568_568957


namespace geometric_means_insertion_l568_568480

noncomputable def is_geometric_progression (s : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (r_pos : r > 0), ∀ n, s (n + 1) = s n * r

theorem geometric_means_insertion (s : ℕ → ℝ) (n : ℕ)
  (h : is_geometric_progression s)
  (h_pos : ∀ i, s i > 0) :
  ∃ t : ℕ → ℝ, is_geometric_progression t :=
sorry

end geometric_means_insertion_l568_568480


namespace test_completion_ways_l568_568545

theorem test_completion_ways (questions : ℕ) (choices : ℕ) 
  (h_questions : questions = 4) (h_choices : choices = 5)
  (h_unanswered : ∀ q, q < questions → (∃ ans, ans = none)) : 
  ∃ n, n = 1 :=
by
  exists 1
  sorry

end test_completion_ways_l568_568545


namespace ellipse_properties_l568_568327

noncomputable def ellipse_standard_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∃ (C : set (ℝ × ℝ)), (∀ p ∈ C, (p.1^2 / (a^2)) + (p.2^2 / (b^2)) = 2)

theorem ellipse_properties
  (a b : ℝ) (ha : a > b) (hb : b > 0)
  (O : ℝ × ℝ) (hO : O = (0, 0))
  (F1 F2 : ℝ × ℝ) (A B : ℝ × ℝ)
  (hF1 : F1 = (-√6, 0)) (hF2 : F2 = (√6, 0))
  (hA : A = (√6, 0)) (hB : B = (0, √2))
  (seq : |b / (a / (a/√6))| = 1)
  (d : ℝ) (hd : d = ℝ.sqrt 6 - 2) :
  ellipse_standard_equation 6 2 ∧ (∃ m : ℝ, ∀ T, T ∈ {p : ℝ × ℝ | p.1 = -3} → 
    (min_value (λ T : ℝ × ℝ, min_value (λ PQ : ℝ × ℝ × ℝ × ℝ, 
      |d / ((m ^ 2 + 1) * (m ^ 2 + 3) / (2 * √6 * (m ^ 2 + 1))) * 
      (λ y1 y2, y1 + y2 = (4 * m) / (m ^ 2 + 3) ∧ y1 * y2 = -2 / (m ^ 2 + 3))| = √3/3))) :=
begin
  sorry
end

end ellipse_properties_l568_568327


namespace daily_sales_profit_45_selling_price_for_1200_profit_l568_568191

-- Definitions based on given conditions

def cost_price : ℤ := 30
def base_selling_price : ℤ := 40
def base_sales_volume : ℤ := 80
def price_increase_effect : ℤ := 2
def max_selling_price : ℤ := 55

-- Part (1): Prove that for a selling price of 45 yuan, the daily sales profit is 1050 yuan.
theorem daily_sales_profit_45 :
  let selling_price := 45
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = 1050 := by sorry

-- Part (2): Prove that to achieve a daily profit of 1200 yuan, the selling price should be 50 yuan.
theorem selling_price_for_1200_profit :
  let target_profit := 1200
  ∃ (selling_price : ℤ), 
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = target_profit ∧ selling_price ≤ max_selling_price ∧ selling_price = 50 := by sorry

end daily_sales_profit_45_selling_price_for_1200_profit_l568_568191


namespace polygon_with_interior_angle_150_has_54_diagonals_l568_568931

theorem polygon_with_interior_angle_150_has_54_diagonals
  (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 150 = (((n - 2) * 180) / n)) : 
  (n = 12) → ∑ i in (finset.range (n-1)), i = 54 :=
by
  sorry

end polygon_with_interior_angle_150_has_54_diagonals_l568_568931


namespace initial_counts_l568_568952

-- Define the initial conditions and the values
variables b0 g0 : ℕ

-- Conditions at stop 1
def b1 : ℕ := b0 + g0 / 3
def g1 : ℕ := 2 * g0 / 3

-- Conditions at stop 2
def b2 : ℕ := 2 * b1 / 3 + 2 * g0 / 9
def g2 : ℕ := 7 * g0 / 9 + b0 / 3

-- The statements to prove
theorem initial_counts :
  (b2 = g0) → (g2 = b2 + 2) → (b0 = 14 ∧ g0 = 12) := by
  intros h1 h2
  sorry

end initial_counts_l568_568952


namespace cos_A_minus_C_l568_568011

theorem cos_A_minus_C (A B C : ℝ) (h1 : Real.cos A + Real.sin B = 1)
(h2 : Real.sin A + Real.cos B = sqrt 3) :
  Real.cos (A - C) = sqrt 3 / 2 := by
  sorry

end cos_A_minus_C_l568_568011


namespace sum_of_digits_of_joeys_next_multiple_age_l568_568440

noncomputable def chloe_age := 18
def liam_age := 2
def joey_age := chloe_age + 2

theorem sum_of_digits_of_joeys_next_multiple_age :
  let joey_next_multiple_age := joey_age + 18 in
  joey_next_multiple_age % (liam_age + 18) = 0 →
  (joey_next_multiple_age / 10) + (joey_next_multiple_age % 10) = 11 :=
by
  let joey_next_multiple_age := joey_age + 18
  have : joey_next_multiple_age % (liam_age + 18) = 0 := sorry
  have : (joey_next_multiple_age / 10) + (joey_next_multiple_age % 10) = 11 := sorry
  exact this

end sum_of_digits_of_joeys_next_multiple_age_l568_568440


namespace faucet_open_duration_l568_568129

-- Initial definitions based on conditions in the problem
def init_water : ℕ := 120
def flow_rate : ℕ := 4
def rem_water : ℕ := 20

-- The equivalent Lean 4 statement to prove
theorem faucet_open_duration (t : ℕ) (H1: init_water - rem_water = flow_rate * t) : t = 25 :=
sorry

end faucet_open_duration_l568_568129


namespace domain_of_expression_l568_568648

theorem domain_of_expression (x : ℝ) : 
  x + 3 ≥ 0 → 7 - x > 0 → (x ∈ Set.Icc (-3) 7) :=
by 
  intros h1 h2
  sorry

end domain_of_expression_l568_568648


namespace cardinality_intersection_A_B_l568_568037

open Set Nat

def A : Set ℕ := {a | ∃ k, a = 3 * k + 2 ∧ k ≤ 2000 ∧ 0 < k}
def B : Set ℕ := {b | ∃ k, b = 4 * k - 1 ∧ k ≤ 2000 ∧ 0 < k}

theorem cardinality_intersection_A_B : card (A ∩ B) = 500 := by
  sorry

end cardinality_intersection_A_B_l568_568037


namespace multiplication_of_powers_l568_568537

theorem multiplication_of_powers :
  2^4 * 3^2 * 5^2 * 11 = 39600 := by
  sorry

end multiplication_of_powers_l568_568537


namespace field_area_is_correct_l568_568908

noncomputable def field_area : ℝ := 2000000

theorem field_area_is_correct :
  ∀ (perimeter_jogs : ℕ) (jogging_rate : ℝ) (jogging_time : ℝ) (length_width_ratio : ℝ),
  perimeter_jogs = 10 →
  jogging_rate = 12 →
  jogging_time = 0.5 →
  length_width_ratio = 2 →
  let total_distance_km := jogging_rate * jogging_time * perimeter_jogs in
  let total_distance_m := total_distance_km * 1000 in
  let perimeter := total_distance_m / perimeter_jogs in
  let width := perimeter / (2 * (length_width_ratio + 1)) in
  let length := length_width_ratio * width in
  let area := length * width in
  area = field_area :=
by
  intros,
  simp,
  sorry

end field_area_is_correct_l568_568908


namespace symmetric_point_correct_l568_568280

variables (M : ℝ × ℝ × ℝ) (M' : ℝ × ℝ × ℝ)
variables (a b c d : ℝ) -- Plane coefficients

def is_symmetric (M M' : ℝ × ℝ × ℝ) (a b c d : ℝ) : Prop :=
  let (x1, y1, z1) := M in
  let (x2, y2, z2) := M' in
  x2 = (2 * ((a * x1 + b * y1 + c * z1 + d) / (a^2 + b^2 + c^2)) - x1) / a ∧
  y2 = (2 * ((a * x1 + b * y1 + c * z1 + d) / (a^2 + b^2 + c^2)) - y1) / b ∧
  z2 = (2 * ((a * x1 + b * y1 + c * z1 + d) / (a^2 + b^2 + c^2)) - z1) / c 

theorem symmetric_point_correct :
  is_symmetric (2, -1, 1) (1, 0, -1) 1 (-1) 2 (-2) :=
by {
  sorry
}

end symmetric_point_correct_l568_568280


namespace number_of_four_digit_integers_divisible_by_6_l568_568363

theorem number_of_four_digit_integers_divisible_by_6: 
  {x : ℕ // 1000 ≤ x ∧ x ≤ 9999 ∧ x % 6 = 0}.to_finset.card = 1350 :=
by
  sorry

end number_of_four_digit_integers_divisible_by_6_l568_568363


namespace shifted_function_is_correct_l568_568422

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end shifted_function_is_correct_l568_568422


namespace matches_C_won_l568_568663

variable (A_wins B_wins D_wins total_matches wins_C : ℕ)

theorem matches_C_won 
  (hA : A_wins = 3)
  (hB : B_wins = 1)
  (hD : D_wins = 0)
  (htot : total_matches = 6)
  (h_sum_wins: A_wins + B_wins + D_wins + wins_C = total_matches)
  : wins_C = 2 :=
by
  sorry

end matches_C_won_l568_568663


namespace domain_of_sqrt_expression_l568_568650

def isDomain (x : ℝ) : Prop := x ≥ -3 ∧ x < 7

theorem domain_of_sqrt_expression : 
  { x : ℝ | isDomain x } = { x | x ≥ -3 ∧ x < 7 } :=
by
  sorry

end domain_of_sqrt_expression_l568_568650


namespace stock_price_l568_568401

theorem stock_price
  (income : ℝ) (dividend_rate : ℝ) (investment : ℝ) (FV : ℝ) (P : ℝ)
  (h1 : income = 900)
  (h2 : dividend_rate = 0.20)
  (h3 : investment = 4590)
  (h4 : income = FV * dividend_rate)
  (h5 : P = (investment / FV) * 100) :
  P = 102 :=
begin
  sorry
end

end stock_price_l568_568401


namespace arc_length_of_curve_l568_568918

noncomputable def arc_length : ℝ :=
∫ t in (0 : ℝ)..(Real.pi / 3),
  (Real.sqrt ((t^2 * Real.cos t)^2 + (t^2 * Real.sin t)^2))

theorem arc_length_of_curve :
  arc_length = (Real.pi^3 / 81) :=
by
  sorry

end arc_length_of_curve_l568_568918


namespace find_original_cost_price_l568_568544

variable (C S : ℝ)

-- Conditions
def original_profit (C S : ℝ) : Prop := S = 1.25 * C
def new_profit_condition (C S : ℝ) : Prop := 1.04 * C = S - 12.60

-- Main Theorem
theorem find_original_cost_price (h1 : original_profit C S) (h2 : new_profit_condition C S) : C = 60 := 
sorry

end find_original_cost_price_l568_568544


namespace span_two_faces_dominos_odd_l568_568261

/-- 
Given a 9 × 9 × 9 cube where each face is covered completely by 2 × 1 dominos 
along the grid lines without overlap or gaps, 
prove that the number of dominos that span two faces is odd.
-/
theorem span_two_faces_dominos_odd :
  let n := 9 in
  let face := n * n in
  let total_faces := 6 in
  let total_area := total_faces * face in
  let b := total_faces * (face // 2 + face % 2) in
  let w := total_faces * (face // 2) in
  let discrepancy := b - w in
  (discrepancy % 2 = 0) →
  ∃ cross : ℕ, cross % 2 = 1 :=
by
  sorry

end span_two_faces_dominos_odd_l568_568261


namespace george_says_25_l568_568074

def alice_skips (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k - 1 ∧ k ∈ set.Ico 1 202

def barbara_skips (n : ℕ) : Prop := ∃ k : ℕ, (n % 5 ≠ 0) ∧ (n ≠ k) ∧ ¬ alice_skips (n) ∧ ¬ alice_skips (n - 1)

def candice_skips (n : ℕ) : Prop := ∃ k : ℕ, (n % 5 ≠ 0) ∧ (n ≠ k) ∧ ¬ barbara_skips (n) ∧ ¬ barbara_skips (n - 1)

def debbie_skips (n : ℕ) : Prop := ∃ k : ℕ, (n % 5 ≠ 0) ∧ (n ≠ k) ∧ ¬ candice_skips (n) ∧ ¬ candice_skips (n - 1)

def eliza_skips (n : ℕ) : Prop := ∃ k : ℕ, (n % 5 ≠ 0) ∧ (n ≠ k) ∧ ¬ debbie_skips (n) ∧ ¬ debbie_skips (n - 1)

def fatima_skips (n : ℕ) : Prop := ∃ k : ℕ, (n % 5 ≠ 0) ∧ (n ≠ k) ∧ ¬ eliza_skips (n) ∧ ¬ eliza_skips (n - 1)

def george's_number (n : ℕ) : Prop :=
  (n % 5 ≠ 0) ∧
  (n ≠ ∃ k : ℕ, (n ≠ k)) ∧
  (¬ alice_skips (n)) ∧
  (¬ barbara_skips (n)) ∧
  (¬ candice_skips (n)) ∧
  (¬ debbie_skips (n)) ∧
  (¬ eliza_skips (n)) ∧
  (¬ fatima_skips (n)) ∧
  (∃ k, n = k^2 + (k + 1)^2)

theorem george_says_25 : george's_number 25 := 
sorry

end george_says_25_l568_568074


namespace sum_of_valid_three_digit_numbers_l568_568287

def is_valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d ≠ 5

def is_valid_three_digit_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let units := n % 10
  100 ≤ n ∧ n < 1000 ∧ is_valid_digit hundreds ∧ is_valid_digit tens ∧ is_valid_digit units

theorem sum_of_valid_three_digit_numbers : 
  (∑ n in finset.filter is_valid_three_digit_number (finset.range 1000), n) = 284160 :=
by
  sorry

end sum_of_valid_three_digit_numbers_l568_568287


namespace juvy_chives_l568_568773

-- Definitions based on the problem conditions
def total_rows : Nat := 20
def plants_per_row : Nat := 10
def parsley_rows : Nat := 3
def rosemary_rows : Nat := 2
def chive_rows : Nat := total_rows - (parsley_rows + rosemary_rows)

-- The statement we want to prove
theorem juvy_chives : chive_rows * plants_per_row = 150 := by
  sorry

end juvy_chives_l568_568773


namespace nine_digit_valid_numbers_count_l568_568358

-- Definitions
def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3
def has_exactly_one_pair_of_consecutive_twos (digits : List ℕ) : Prop :=
  digits.countp (λ (t : ℕ × ℕ), t = (2, 2)) (List.zip digits digits.tail) = 1
def has_no_consecutive_threes (digits : List ℕ) : Prop :=
  digits.all (λ (t : ℕ × ℕ), t ≠ (3, 3)) (List.zip digits digits.tail)

def a : ℕ → ℕ
| 1 := 3
| 2 := 8
| n := 2 * (a (n - 1)) + 2 * (a (n - 2))

-- The proof problem
theorem nine_digit_valid_numbers_count : a 9 = 1232 := 
by sorry

end nine_digit_valid_numbers_count_l568_568358


namespace total_students_proof_l568_568219

variable (studentsA studentsB : ℕ) (ratioAtoB : ℕ := 3/2)
variable (percentA percentB : ℕ := 10/100)
variable (diffPercent : ℕ := 20/100)
variable (extraStudentsInA : ℕ := 190)
variable (totalStudentsB : ℕ := 650)

theorem total_students_proof :
  (studentsB = totalStudentsB) ∧ 
  ((percentA * studentsA - diffPercent * studentsB = extraStudentsInA) ∧
  (studentsA / studentsB = ratioAtoB)) →
  (studentsA + studentsB = 1625) :=
by
  sorry

end total_students_proof_l568_568219


namespace sum_of_digits_1_to_55_l568_568922

-- Define the sequence of digits formed by concatenating integers from 1 to 55.
def concatenated_sequence : List ℕ := (List.range 55).map (fun n => n + 1) |> List.join_map (fun n => n.toString.data |> List.map (fun c => c.toNat - '0'.toNat))

-- Define the function to calculate the sum of digits in the given sequence.
def sum_of_digits (l : List ℕ) : ℕ := l.foldl (· + ·) 0

-- Statement of the problem: The sum of the digits of the concatenated sequence from 1 to 55 equals 370.
theorem sum_of_digits_1_to_55 : sum_of_digits concatenated_sequence = 370 := by
  sorry

end sum_of_digits_1_to_55_l568_568922


namespace area_of_pentagon_is_10_l568_568530

noncomputable def area_geoboard_pentagon : ℝ :=
  let v1 := (1, 2)
  let v2 := (2, 8)
  let v3 := (5, 5)
  let v4 := (7, 2)
  let v5 := (3, 0)
  let verts := [v1, v2, v3, v4, v5, v1] -- Close the polygon

  (1 / 2 : ℝ) *
    |(verts[0].1 * verts[1].2) + (verts[1].1 * verts[2].2) + (verts[2].1 * verts[3].2) + 
    (verts[3].1 * verts[4].2) + (verts[4].1 * verts[5].2) - 
    (verts[0].2 * verts[1].1) - (verts[1].2 * verts[2].1) - (verts[2].2 * verts[3].1) - 
    (verts[3].2 * verts[4].1) - (verts[4].2 * verts[5].1)| -- Shoelace theorem formula

theorem area_of_pentagon_is_10 : area_geoboard_pentagon = 10 := by
  sorry

end area_of_pentagon_is_10_l568_568530


namespace volume_of_Q3_l568_568679

theorem volume_of_Q3 : 
  ∃ (Q₀ : ℝ), Q₀ = 8 ∧ 
  ∀ (n : ℕ), let Q := λ n, if n = 0 then 8 else Q(n-1) - 8 * (4^(n-1)) / (27^n)
  in Q 3 = 8 - (1 / 27) - (4 / 729) - (16 / 19683) :=
begin
  sorry
end

end volume_of_Q3_l568_568679


namespace geometric_sequence_product_l568_568867

-- Define the geometric sequence sum and the initial conditions
variables {S : ℕ → ℚ} {a : ℕ → ℚ}
variables (q : ℚ) (h1 : a 1 = -1/2)
variables (h2 : S 6 / S 3 = 7 / 8)

-- The main proof problem statement
theorem geometric_sequence_product (h_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 2 * a 4 = 1 / 64 :=
sorry

end geometric_sequence_product_l568_568867


namespace max_sum_of_factors_l568_568784

theorem max_sum_of_factors (heartsuit spadesuit : ℕ) (h : heartsuit * spadesuit = 24) :
  heartsuit + spadesuit ≤ 25 :=
sorry

end max_sum_of_factors_l568_568784


namespace sandy_initial_payment_l568_568826

variable (P : ℝ) 

theorem sandy_initial_payment
  (h1 : (1.2 : ℝ) * (P + 200) = 1200) :
  P = 800 :=
by
  -- Proof goes here
  sorry

end sandy_initial_payment_l568_568826


namespace factorize_expr_l568_568267

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end factorize_expr_l568_568267


namespace max_magnetic_field_intensity_at_triangle_l568_568135

noncomputable def magnetic_field_intensity (shape : Type) (area : ℝ) (current : ℝ) : ℝ := 
  sorry -- function to compute the magnetic field intensity given a shape, area, and current

def equilateral_triangle : Type := sorry
def square : Type := sorry
def regular_pentagon : Type := sorry
def regular_hexagon : Type := sorry
def circle : Type := sorry

theorem max_magnetic_field_intensity_at_triangle (A : ℝ) (I : ℝ) :
  let H_triangle := magnetic_field_intensity equilateral_triangle A I in
  let H_square := magnetic_field_intensity square A I in
  let H_pentagon := magnetic_field_intensity regular_pentagon A I in
  let H_hexagon := magnetic_field_intensity regular_hexagon A I in
  let H_circle := magnetic_field_intensity circle A I in
  H_triangle > H_square ∧
  H_triangle > H_pentagon ∧
  H_triangle > H_hexagon ∧
  H_triangle > H_circle :=
sorry

end max_magnetic_field_intensity_at_triangle_l568_568135


namespace ratio_of_areas_eq_3_l568_568676

section AreaRatio

variables {A B C O : Type} [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ O]

def inside_triangle (O A B C : A) : Prop := 
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 ∧ 
  O = a • A + b • B + c • C

def condition (O A B C : A) : Prop := 
  (A -ᵥ O) + 2 • (B -ᵥ O) + 3 • (C -ᵥ O) = 0 

/-- Proof that the ratio of the area of triangle ABC to the area of triangle AOC is 3 -/
theorem ratio_of_areas_eq_3 
  (hO_in_triangle : inside_triangle O A B C) 
  (h_condition : condition O A B C) : 
  (area (triangle A B C)) / (area (triangle A O C)) = 3 := 
sorry

end AreaRatio

end ratio_of_areas_eq_3_l568_568676


namespace length_of_third_wall_l568_568438

-- Define the dimensions of the first two walls
def wall1_length : ℕ := 30
def wall1_height : ℕ := 12
def wall1_area : ℕ := wall1_length * wall1_height

def wall2_length : ℕ := 30
def wall2_height : ℕ := 12
def wall2_area : ℕ := wall2_length * wall2_height

-- Total area needed
def total_area_needed : ℕ := 960

-- Calculate the area for the third wall
def two_walls_area : ℕ := wall1_area + wall2_area
def third_wall_area : ℕ := total_area_needed - two_walls_area

-- Height of the third wall
def third_wall_height : ℕ := 12

-- Calculate the length of the third wall
def third_wall_length : ℕ := third_wall_area / third_wall_height

-- Final claim: Length of the third wall is 20 feet
theorem length_of_third_wall : third_wall_length = 20 := by
  sorry

end length_of_third_wall_l568_568438


namespace negation_proposition_true_l568_568857

theorem negation_proposition_true (x : ℝ) : (¬ (|x| > 1 → x > 1)) ↔ (|x| ≤ 1 → x ≤ 1) :=
by sorry

end negation_proposition_true_l568_568857


namespace train_length_l568_568202

theorem train_length 
  (jogger_speed_kmh : ℕ) 
  (train_speed_kmh : ℕ) 
  (head_start_m : ℕ) 
  (time_to_pass_s : ℕ) 
  (h1 : jogger_speed_kmh = 9) 
  (h2 : train_speed_kmh = 45) 
  (h3 : head_start_m = 240) 
  (h4 : time_to_pass_s = 37) 
  : 
  let relative_speed_ms := (train_speed_kmh - jogger_speed_kmh) * 5 / 18 in
  let distance_traveled_m := relative_speed_ms * time_to_pass_s in
  let train_length_m := distance_traveled_m - head_start_m in
  train_length_m = 130 :=
by
  simp [h1, h2, h3, h4]
  sorry

end train_length_l568_568202


namespace daily_profit_at_45_selling_price_for_1200_profit_l568_568192

-- Definitions for the conditions
def cost_price (p: ℝ) : Prop := p = 30
def initial_sales (p: ℝ) (s: ℝ) : Prop := p = 40 ∧ s = 80
def sales_decrease_rate (r: ℝ) : Prop := r = 2
def max_selling_price (p: ℝ) : Prop := p ≤ 55

-- Proof for Question 1
theorem daily_profit_at_45 (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate) :
  (price = 45) → profit = 1050 :=
by sorry

-- Proof for Question 2
theorem selling_price_for_1200_profit (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate)
  (h_max_price : ∀ p, max_selling_price p → p ≤ 55) :
  profit = 1200 → price = 50 :=
by sorry

end daily_profit_at_45_selling_price_for_1200_profit_l568_568192


namespace ratio_of_eighth_terms_l568_568026

theorem ratio_of_eighth_terms 
  (S_n T_n : ℕ → ℝ)
  (h_ratio : ∀ n : ℕ, (n > 0) → S_n n / T_n n = (5 * n + 6) / (3 * n + 30)): 
  let a_8 := (a + 7 * d), b_8 := (b + 7 * e) in
  let ratio := ((a + 7 * d) / (b + 7 * e)) in
  ratio = 4 / 3 :=
sorry

end ratio_of_eighth_terms_l568_568026


namespace count_numeric_hex_integers_up_to_500_l568_568357

def is_numeric_hex (n : ℕ) : Prop :=
  ∀ c ∈ n.to_nat.base_repr 16, c.val < 10

theorem count_numeric_hex_integers_up_to_500 : 
  (finset.filter is_numeric_hex (finset.range 501)).card = 199 := 
sorry

end count_numeric_hex_integers_up_to_500_l568_568357


namespace amount_exceeds_l568_568813

variables {a b : ℕ}

theorem amount_exceeds : (1/2 : ℚ) * (10 * a + b) - (1/4 : ℚ) * (10 * a + b) = 21 / 4 :=
begin
  -- Given conditions
  assume (cond1 : a + b = 3),
  assume (cond2 : (10 * a + b) = 21),

  -- Sorry is used here to skip the proof steps
  sorry,
end

end amount_exceeds_l568_568813


namespace probability_both_heads_on_last_flip_l568_568524

noncomputable def fair_coin_flip : probabilityₓ ℙ :=
  probabilityₓ.ofUniform [true, false]

def both_coins_heads (events : list (bool × bool)) : bool :=
  events.all (λ event, event.1 = true)

def stops_with_heads (events : list (bool × bool)) : bool :=
  events.any (λ event, event.1 = true ∨ event.2 = true)

theorem probability_both_heads_on_last_flip :
  ∀ events : list (bool × bool), probabilityₓ (fair_coin_flip ×ₗ fair_coin_flip)
  (λ event, both_coins_heads events = true ∧ stops_with_heads events = true) = 1 / 3 :=
sorry

end probability_both_heads_on_last_flip_l568_568524


namespace paint_rate_l568_568105

theorem paint_rate (l b : ℝ) (cost : ℕ) (rate_per_sq_m : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : cost = 300) 
  (h3 : l = 13.416407864998739) 
  (area : ℝ := l * b) : 
  rate_per_sq_m = 5 :=
by
  sorry

end paint_rate_l568_568105


namespace correct_mark_l568_568572

theorem correct_mark (wrong_mark : ℕ) (num_pupils : ℕ) (average_increase : ℕ) (correct_mark : ℕ) :
  wrong_mark = 67 → num_pupils = 44 → average_increase = num_pupils / 2 → (wrong_mark - correct_mark) = average_increase → correct_mark = 45 :=
by
  intros h_wrong_mark h_num_pupils h_average_increase h_eq
  rw [h_wrong_mark, h_num_pupils] at *
  have h1 : average_increase = 44 / 2 := h_average_increase
  have h2 : average_increase = 22 := by
      norm_num at h1
  rw h2 at h_eq
  simp at h_eq
  assumption

end correct_mark_l568_568572


namespace compute_fraction_pow_mult_l568_568981

def frac_1_3 := (1 : ℝ) / (3 : ℝ)
def frac_1_5 := (1 : ℝ) / (5 : ℝ)
def target := (1 : ℝ) / (405 : ℝ)

theorem compute_fraction_pow_mult :
  (frac_1_3^4 * frac_1_5) = target :=
by
  sorry

end compute_fraction_pow_mult_l568_568981


namespace enclosed_area_of_curve_l568_568492

theorem enclosed_area_of_curve
  (count_arcs : Nat)
  (arc_length : ℝ)
  (side_length : ℝ) :
  count_arcs = 9 →
  arc_length = π →
  side_length = 3 →
  let radius := arc_length / π in
  let sector_area := (π * radius^2) / 2 in
  let total_sector_area := count_arcs * sector_area in
  let square_area := side_length^2 in
  total_sector_area + square_area = 9 + 4.5 * π :=
by
  intros count_arcs_eq arc_length_eq side_length_eq
  let radius : ℝ := arc_length / π
  let sector_area : ℝ := (π * radius^2) / 2
  let total_sector_area : ℝ := count_arcs * sector_area
  let square_area : ℝ := side_length^2
  sorry

end enclosed_area_of_curve_l568_568492


namespace intersection_A_B_l568_568683

def setA (x : ℝ) : Prop := 3 * x + 2 > 0
def setB (x : ℝ) : Prop := (x + 1) * (x - 3) > 0
def A : Set ℝ := { x | setA x }
def B : Set ℝ := { x | setB x }

theorem intersection_A_B : A ∩ B = { x | 3 < x } := by
  sorry

end intersection_A_B_l568_568683


namespace area_of_trapezoid_l568_568554

-- Definitions for conditions
def Rectangle (A B C D : Point) : Prop :=
  -- Assuming a helper definition of coordinates and properties for a rectangle
  IsRectangle A B C D

-- Area of a rectangle
def area (A B C D : Point) : ℝ :=
  20 -- Given area

-- Proportions for points E and F
def ratio_AE_ED (A D E : Point) : Prop :=
  -- Assuming some coordinates where E divides AD in the ratio 1:3
  AE : R = 1/3 * |AD|

def ratio_BF_FC (B C F : Point) : Prop :=
  -- Assuming some coordinates where F divides BC in the ratio 1:3
  BF : R = 1/3 * |BC|

-- The main statement
theorem area_of_trapezoid {A B C D E F : Point} :
  Rectangle A B C D →
  area A B C D = 20 →
  ratio_AE_ED A D E →
  ratio_BF_FC B C F →
  area_of_trapezoid_EFBA A B E F = 4.375 :=
by
  sorry

end area_of_trapezoid_l568_568554


namespace red_jelly_beans_are_coconut_flavored_l568_568133

theorem red_jelly_beans_are_coconut_flavored (total_jelly_beans : ℕ) (three_fourths_red : total_jelly_beans * 3 / 4 = 3000) (one_quarter_coconut : 3000 * 1 / 4 = 750) :
  ∃ n, n = 750 := 
begin
  use 750,
  sorry
end

end red_jelly_beans_are_coconut_flavored_l568_568133


namespace complex_solutions_l568_568270

theorem complex_solutions (z : ℂ) : (z^2 = -45 - 28 * complex.I) ↔ (z = 2 - 7 * complex.I ∨ z = -2 + 7 * complex.I) := 
by 
  sorry

end complex_solutions_l568_568270


namespace abc_inequality_l568_568549

-- Define a mathematical statement to encapsulate the problem
theorem abc_inequality (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by sorry

end abc_inequality_l568_568549


namespace tenth_term_is_115_l568_568801

-- Define the sequence transformation rules
def next_term (n : ℕ) : ℕ :=
  if n < 10 then n * 10
  else if n % 2 = 0 then n * 3
  else n + 10

-- Define the sequence
noncomputable def sequence : ℕ → ℕ
| 0     := 15
| (n+1) := next_term (sequence n)

-- The theorem to prove
theorem tenth_term_is_115 : sequence 9 = 115 :=
by sorry

end tenth_term_is_115_l568_568801


namespace combination_schemes_l568_568403

def number_of_combinations (total number_of_salespersons number_of_technicians number_to_select: ℕ) : ℕ := 
  Nat.binomial total number_to_select - Nat.binomial number_of_salespersons number_to_select - Nat.binomial number_of_technicians number_to_select

theorem combination_schemes (total number_of_salespersons number_of_technicians number_to_select : ℕ) :
  total = 9 →
  number_of_salespersons = 5 →
  number_of_technicians = 4 →
  number_to_select = 3 →
  number_of_combinations total number_of_salespersons number_of_technicians number_to_select = 70 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp [number_of_combinations]
  sorry

end combination_schemes_l568_568403


namespace andrew_grapes_purchase_l568_568231

theorem andrew_grapes_purchase (G : ℕ) (rate_grape rate_mango total_paid total_mango_cost : ℕ)
  (h1 : rate_grape = 54)
  (h2 : rate_mango = 62)
  (h3 : total_paid = 1376)
  (h4 : total_mango_cost = 10 * rate_mango)
  (h5 : total_paid = rate_grape * G + total_mango_cost) : G = 14 := by
  sorry

end andrew_grapes_purchase_l568_568231


namespace range_of_f_l568_568113

def f (x : ℝ) : ℝ := (3 * x + 1) / (x - 1)

theorem range_of_f :
  (set.range (λ x : {y // y ≠ 1}, f y)) = {y : ℝ | y ≠ 3} :=
by
  sorry

end range_of_f_l568_568113


namespace sin_neg_1740_eq_sqrt3_div_2_l568_568871

theorem sin_neg_1740_eq_sqrt3_div_2 : Real.sin (-1740 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_1740_eq_sqrt3_div_2_l568_568871


namespace solve_for_y_l568_568082

theorem solve_for_y : ∃ y : ℝ, (5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3)) ↔ y = 1000 := by
  sorry

end solve_for_y_l568_568082


namespace count_three_digit_even_mountain_numbers_l568_568529

def mountain_number (n : ℕ) : Prop := ∃ (a b c : ℕ), 
  n = 100 * a + 10 * b + c ∧ 
  b > a ∧ b > c ∧ 
  (c % 2 = 0) ∧ 
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9)

theorem count_three_digit_even_mountain_numbers : 
  set.count {n | mountain_number n ∧ 100 ≤ n ∧ n < 1000 ∧ (n % 2 = 0)} = 76 :=
sorry

end count_three_digit_even_mountain_numbers_l568_568529


namespace min_value_abs_sum_correct_l568_568323

noncomputable def min_value_abs_sum (a : list ℝ) : ℝ :=
if h : a.length % 2 = 0 then
  let m := a.length / 2 in
  - (list.sum (a.take m)) + list.sum (a.drop m)
else
  let m := a.length / 2 in
  - (list.sum (a.take m)) + list.sum (a.drop (m + 1))

theorem min_value_abs_sum_correct (a : list ℝ) (h_distinct : a.nodup) :
  ∃ x : ℝ, ∀ x', (list.sum (a.map (λ ai, abs (x' - ai)))) ≥
             (list.sum (a.map (λ ai, abs (x - ai)))) ∧
             (list.sum (a.map (λ ai, abs (x - ai)))) = min_value_abs_sum a :=
sorry

end min_value_abs_sum_correct_l568_568323


namespace shift_right_linear_function_l568_568409

theorem shift_right_linear_function (x : ℝ) : 
  (∃ k b : ℝ, k ≠ 0 ∧ (∀ x : ℝ, y = -2x → y = kx + b) → (x, y) = (x - 3, -2(x-3))) → y = -2x + 6 :=
by
  sorry

end shift_right_linear_function_l568_568409


namespace problem_l568_568085

noncomputable def f (x : ℝ) : ℝ := 5 * x - 7
noncomputable def g (x : ℝ) : ℝ := x / 5 + 3

theorem problem : ∀ x : ℝ, f (g x) - g (f x) = 6.4 :=
by
  intro x
  sorry

end problem_l568_568085


namespace new_trailers_added_l568_568137

theorem new_trailers_added (n : ℕ) :
  let original_trailers := 15
  let original_age := 12
  let years_passed := 3
  let current_total_trailers := original_trailers + n
  let current_average_age := 10
  let total_age_three_years_ago := original_trailers * original_age
  let new_trailers_age := 3
  let total_current_age := (original_trailers * (original_age + years_passed)) + (n * new_trailers_age)
  (total_current_age / current_total_trailers = current_average_age) ↔ (n = 10) :=
by
  sorry

end new_trailers_added_l568_568137


namespace sums_of_squares_divisibility_l568_568451

theorem sums_of_squares_divisibility :
  (∀ n : ℤ, (3 * n^2 + 2) % 3 ≠ 0) ∧ (∃ n : ℤ, (3 * n^2 + 2) % 11 = 0) := 
by
  sorry

end sums_of_squares_divisibility_l568_568451


namespace shift_right_linear_function_l568_568410

theorem shift_right_linear_function (x : ℝ) : 
  (∃ k b : ℝ, k ≠ 0 ∧ (∀ x : ℝ, y = -2x → y = kx + b) → (x, y) = (x - 3, -2(x-3))) → y = -2x + 6 :=
by
  sorry

end shift_right_linear_function_l568_568410


namespace student_exchanges_l568_568398

theorem student_exchanges (x : ℕ) : x * (x - 1) = 72 :=
sorry

end student_exchanges_l568_568398


namespace increasing_on_interval_l568_568603

-- Define the four functions 
def f1 (x : ℝ) : ℝ := x⁻¹
def f2 (x : ℝ) : ℝ := (1/2)^x
def f3 (x : ℝ) : ℝ := 1 / (1 - x)
def f4 (x : ℝ) : ℝ := x^2 - 4 * x

-- State the theorem
theorem increasing_on_interval : ∀ x > 1, strict_mono (f3) :=
by
  sorry

end increasing_on_interval_l568_568603


namespace part_I_part_II_i_part_II_ii_l568_568326

-- Definition and proof statement for part (I)
noncomputable def foci₁ := ( -1 : ℝ, 0 : ℝ)
noncomputable def foci₂ := (  1 : ℝ, 0 : ℝ)
noncomputable def ellipse_C : Set (ℝ × ℝ) := {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

theorem part_I : 
  (∃ a > 0, ∃ b > 0, Set (ℝ × ℝ) = {p | (p.1^2 / (a^2)) + (p.2^2 / (b^2)) = 1} 
    ∧ foci₁ = (-1, 0) ∧ foci₂ = (1, 0) ∧ a^2 - b^2 = 1) → 
    ellipse_C = {p | (p.1^2 / 4) + (p.2^2 / 3) = 1} :=
sorry

-- Definitions for part (II)(i)
noncomputable def A (m : ℝ) : Set (ℝ × ℝ) := (-2, m)
noncomputable def B (n : ℝ) : Set (ℝ × ℝ) := (2, n)
noncomputable def F₁ : (ℝ × ℝ) := (-1, 0)

theorem part_II_i (m n : ℝ) 
  (h1 : ∃ m n, (m * n = 3) ∧ (m^2 - n^2 = 8)) :
  |((A m).Prod (λ a₁ a₂, (a₁ - F₁.1, a₂ - F₁.2))).dist (0, 0)| = |((B n).Prod (λ b₁ b₂, (b₁ - F₁.1, b₂ - F₁.2))).dist (0, 0)| ∧
  (triangle_area 5) :=
sorry

-- Additional definitions for distances and minimum calculation for part (II)(ii)
noncomputable def distance_to_line (A B : (ℝ × ℝ)) := 
  (|2 (A.1 + B.1) - (B.1 - A.1)| / sqrt ((B.1 - A.1)^2 + 16)) + 
  (|2 (A.1 + B.1) + (B.1 - A.1)| / sqrt ((B.1 - A.1)^2 + 16))

theorem part_II_ii (m n : ℝ)
  (h1 : ∃ m n, m * n = 3) :
  min (distance_to_line (A m) (B n)) (4 ⋅ sqrt 1 - 4 / (m^2 + n^2 + 10)) = 2√3 :=
sorry

end part_I_part_II_i_part_II_ii_l568_568326


namespace lucky_lucy_l568_568797

theorem lucky_lucy (a b c d e : ℤ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = 6)
  (hd : d = 8)
  (he : a + b - c + d - e = a + (b - (c + (d - e)))) :
  e = 8 :=
by
  rw [ha, hb, hc, hd] at he
  exact eq_of_sub_eq_zero (by linarith)

end lucky_lucy_l568_568797


namespace day_of_month_l568_568489

/--
The 25th day of a particular month is a Monday. 
We need to prove that the 1st day of that month is a Friday.
-/
theorem day_of_month (h : (25 % 7 = 1)) : (1 % 7 = 5) :=
sorry

end day_of_month_l568_568489


namespace find_equation_of_s_l568_568633

theorem find_equation_of_s (Q Q'' : ℝ × ℝ)
  (r : ℝ → ℝ → Prop)
  (s : ℝ → ℝ → Prop)
  (Q_reflected_about_r : ∀ x y, r x y → Q = (3, -5) → Q' = (2, -3))
  (Q'_reflected_about_s : ∀ x y, s x y → Q' = (2, -3) → Q'' = (7, -2))
  (Line_r : ∀ x y, r x y ↔ 3 * x + y = 0)
  (Line_s : ∀ x y, s x y ↔ (1 + 3 * Real.sqrt(3)) * y + (-3 + Real.sqrt(3)) * x = 0) :
  Line_s (fst Q'') (snd Q'') :=
by
  sorry

end find_equation_of_s_l568_568633


namespace bowling_team_avg_weight_l568_568876

noncomputable def total_weight (weights : List ℕ) : ℕ :=
  weights.foldr (· + ·) 0

noncomputable def average_weight (weights : List ℕ) : ℚ :=
  total_weight weights / weights.length

theorem bowling_team_avg_weight :
  let original_weights := [76, 76, 76, 76, 76, 76, 76]
  let new_weights := [110, 60, 85, 65, 100]
  let combined_weights := original_weights ++ new_weights
  average_weight combined_weights = 79.33 := 
by 
  sorry

end bowling_team_avg_weight_l568_568876


namespace fraction_sum_value_l568_568452

theorem fraction_sum_value (a b c D E F : ℝ) (h_poly_roots : Polynomial.Roots (Polynomial.Cubic 1 (-36) 215 (-470)) = {a, b, c})
  (h_fraction : ∀ t : ℝ, t ∉ {a, b, c} → (1 / (t^3 - 36 * t^2 + 215 * t - 470) = D / (t - a) + E / (t - b) + F / (t - c))) :
  1 / D + 1 / E + 1 / F = 105 :=
sorry

end fraction_sum_value_l568_568452


namespace dodecagon_diagonals_l568_568586

theorem dodecagon_diagonals : 
  let n := 12 in 
  (n * (n - 3)) / 2 = 54 :=
by
  sorry

end dodecagon_diagonals_l568_568586


namespace ln_t_increasing_on_0_1_l568_568654

open Real

-- Definitions and Conditions
def t (x : ℝ) : ℝ := 3 * x - x ^ 3

-- Theorem Statement
theorem ln_t_increasing_on_0_1 :
  ∀ x, x > 0 ∧ x < sqrt 3 → (deriv (λ x, log (t x))) x > 0 →
    ∀ y, y ∈ Ioo 0 (sqrt 3) → (y > 0 ∧ y < 1 → deriv (λ y, log (t y)) y > 0) :=
begin
  sorry
end

end ln_t_increasing_on_0_1_l568_568654


namespace count_special_three_digit_numbers_l568_568719

/-- The set of prime digits for the first two positions -/
def prime_digits := {2, 3, 5, 7}

/-- The set of non-prime odd digits for the last position -/
def non_prime_odd_digits := {1, 9}

/-- Prove that the number of positive three-digit integers, where the first two digits are prime
and the last digit is a non-prime odd number is 32 -/
theorem count_special_three_digit_numbers : 
  (prime_digits.card * prime_digits.card * non_prime_odd_digits.card) = 32 :=
by sorry

end count_special_three_digit_numbers_l568_568719


namespace g_g_2_eq_78652_l568_568369

def g (x : ℝ) : ℝ := 4 * x^3 - 3 * x + 1

theorem g_g_2_eq_78652 : g (g 2) = 78652 := by
  sorry

end g_g_2_eq_78652_l568_568369


namespace sequence_solution_l568_568644

theorem sequence_solution :
  ∀ (a : ℕ → ℝ), (∀ m n : ℕ, a (m^2 + n^2) = a m ^ 2 + a n ^ 2) →
  (0 ≤ a 0 ∧ a 0 ≤ a 1 ∧ a 1 ≤ a 2 ∧ ∀ n, a n ≤ a (n + 1)) →
  (∀ n, a n = 0) ∨ (∀ n, a n = n) ∨ (∀ n, a n = 1 / 2) :=
sorry

end sequence_solution_l568_568644


namespace ratio_of_percentage_change_l568_568929

theorem ratio_of_percentage_change
  (P U U' : ℝ)
  (h_price_decrease : U' = 4 * U)
  : (300 / 75) = 4 := 
by
  sorry

end ratio_of_percentage_change_l568_568929


namespace factor_by_which_sides_are_multiplied_l568_568508

theorem factor_by_which_sides_are_multiplied (s f : ℝ) (h : s^2 = 20 * (f * s)^2) : 
  f = real.sqrt 5 / 10 :=
by
  sorry

end factor_by_which_sides_are_multiplied_l568_568508


namespace largest_angle_triangle_l568_568510

-- Definition of constants and conditions
def right_angle : ℝ := 90
def angle_sum : ℝ := 120
def angle_difference : ℝ := 20

-- Given two angles of a triangle sum to 120 degrees and one is 20 degrees greater than the other,
-- Prove the largest angle in the triangle is 70 degrees
theorem largest_angle_triangle (A B C : ℝ) (hA : A + B = angle_sum) (hB : B = A + angle_difference) (hC : A + B + C = 180) : 
  max A (max B C) = 70 := 
by 
  sorry

end largest_angle_triangle_l568_568510


namespace similar_triangles_area_ratio_l568_568886

theorem similar_triangles_area_ratio {ABC DEF : Type} 
  (h_sim : similar ABC DEF) 
  (h_ratio : ∀ (a b : ℝ), side_ratio ABC DEF a b = 1 / 2) 
  (area_ABC : real_area ABC = 3) :
  real_area DEF = 12 := by 
  sorry

end similar_triangles_area_ratio_l568_568886


namespace trig_identity_example_l568_568617

theorem trig_identity_example :
  (Real.cos (47 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - 
   Real.sin (47 * Real.pi / 180) * Real.sin (13 * Real.pi / 180)) = 
  (Real.cos (60 * Real.pi / 180)) := by
  sorry

end trig_identity_example_l568_568617


namespace gradient_magnitude_at_point_1_1_1_l568_568645

noncomputable def scalar_field (x y z : ℝ) : ℝ := x * y + y * z + z * x

def partial_derivative_x (x y z : ℝ) : ℝ := y + z
def partial_derivative_y (x y z : ℝ) : ℝ := x + z
def partial_derivative_z (x y z : ℝ) : ℝ := x + y

def gradient_at_point (x y z : ℝ) := (partial_derivative_x x y z, partial_derivative_y x y z, partial_derivative_z x y z)

theorem gradient_magnitude_at_point_1_1_1 : 
  let g := gradient_at_point 1 1 1 in
  (∥(g.1, g.2, g.2)∥ = 2 * sqrt 3) :=
by 
  let g := gradient_at_point 1 1 1 
  sorry

end gradient_magnitude_at_point_1_1_1_l568_568645


namespace certain_number_mult_three_l568_568539

theorem certain_number_mult_three :
  ∃ x : ℕ, (x + 14 = 56) → 3 * x = 126 :=
begin
  -- The proof will go here
  sorry
end

end certain_number_mult_three_l568_568539


namespace value_of_f_5_l568_568912

variable (f : ℕ → ℕ) (x y : ℕ)

theorem value_of_f_5 (h1 : f 2 = 50) (h2 : ∀ x, f x = 2 * x ^ 2 + y) : f 5 = 92 :=
by
  sorry

end value_of_f_5_l568_568912


namespace sum_first_2019_terms_l568_568115

-- Define the sequence according to the given conditions
def seq : ℕ → ℤ
| 0       := 1
| 1       := -1
| 2       := -2
| (n + 3) := seq (n + 2) - seq (n + 1)

-- Define a function to compute the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℤ :=
  (Finset.range n).sum seq

-- State the theorem that we want to prove
theorem sum_first_2019_terms : sum_seq 2019 = -2 :=
  sorry

end sum_first_2019_terms_l568_568115


namespace add_n_to_constant_l568_568701

theorem add_n_to_constant (y n : ℝ) (h_eq : y^4 - 20 * y + 1 = 22) (h_n : n = 3) : y^4 - 20 * y + 4 = 25 :=
by
  sorry

end add_n_to_constant_l568_568701


namespace meat_purchase_l568_568425

theorem meat_purchase :
  ∃ x y : ℕ, 16 * x = y + 25 ∧ 8 * x = y - 15 ∧ y / x = 11 :=
by
  sorry

end meat_purchase_l568_568425


namespace polynomial_no_positive_roots_theorem_l568_568793

noncomputable def polynomial_no_positive_roots (a : List ℕ) (k M : ℕ) : Prop :=
  (∀ x : ℝ, x > 0 → (M : ℝ) * (1 + x)^k > (List.foldr (*) 1 ((a.map (λ ai, (x + (ai : ℝ)))))))

theorem polynomial_no_positive_roots_theorem (a : List ℕ) (k M : ℕ)
  (h1 : ∑ i in a, 1 / (i : ℝ) = k)
  (h2 : List.foldr (*) 1 a = M)
  (h3 : M > 1) :
  polynomial_no_positive_roots a k M :=
by
  sorry

end polynomial_no_positive_roots_theorem_l568_568793


namespace shifted_linear_function_correct_l568_568847

def original_function (x : ℝ) : ℝ := 5 * x - 8
def shifted_function (x : ℝ) : ℝ := original_function x + 4

theorem shifted_linear_function_correct (x : ℝ) :
  shifted_function x = 5 * x - 4 :=
by
  sorry

end shifted_linear_function_correct_l568_568847


namespace average_speeds_l568_568051

-- Definitions from the conditions
def uphill_distance := 1.5 -- in km
def uphill_time := 45 / 60 -- in hours
def downhill_distance := 1.5 -- in km
def downhill_time := 5 / 60 -- in hours

-- Total distance for the round trip
def total_distance := uphill_distance + downhill_distance -- in km

-- Total time for the round trip
def total_time := uphill_time + downhill_time -- in hours

-- Proving the average speeds
theorem average_speeds :
  (uphill_distance / uphill_time = 2) ∧
  (downhill_distance / downhill_time = 18) ∧
  (total_distance / total_time = 3.6) :=
by
  sorry

end average_speeds_l568_568051


namespace proof_equation_l568_568974

theorem proof_equation :
  2^0 + (1/2)^(-2) = 5 :=
by
  have h1 : 2^0 = 1 := by 
    apply pow_zero
  have h2 : (1/2)^(-2) = 4 := by 
    rw [rat.pow_neg, pow_two, inv_inv, mul_inv_cancel]; norm_num
  rw [h1, h2]
  norm_num
  sorry

end proof_equation_l568_568974


namespace quadratic_real_roots_l568_568700

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, m * x^2 + x - 1 = 0) ↔ (m ≥ -1/4 ∧ m ≠ 0) :=
by
  sorry

end quadratic_real_roots_l568_568700


namespace reeya_third_subject_score_l568_568070

theorem reeya_third_subject_score
  (score1 score2 score4 : ℕ)
  (avg_score : ℕ)
  (num_subjects : ℕ)
  (total_score : ℕ)
  (score3 : ℕ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 85 →
  avg_score = 75 →
  num_subjects = 4 →
  total_score = avg_score * num_subjects →
  score1 + score2 + score3 + score4 = total_score →
  score3 = 83 :=
by
  intros h1 h2 h4 h5 h6 h7 h8
  sorry

end reeya_third_subject_score_l568_568070


namespace mixed_solution_concentration_correct_l568_568800

variable {a b : ℝ}

def concentration_of_mixed_solution (a b : ℝ) : ℝ :=
  (0.15 * a + 0.2 * b) / (a + b)

theorem mixed_solution_concentration_correct (a b : ℝ) :
  concentration_of_mixed_solution a b = (0.15 * a + 0.2 * b) / (a + b) :=
by sorry

end mixed_solution_concentration_correct_l568_568800


namespace car_enters_and_leaves_storm_l568_568562

def car_position (t : ℝ) : ℝ × ℝ := (4/5 * t, 0)
def storm_center_position (t : ℝ) : ℝ × ℝ := (3/5 * t, 150 - 3/5 * t)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem car_enters_and_leaves_storm (t_1 t_2 : ℝ) :
  (distance (car_position t_1) (storm_center_position t_1) = 75) ∧ 
  (distance (car_position t_2) (storm_center_position t_2) = 75) →
  (t_1 < t_2) → 
  (\exists (a b : ℝ), distance (car_position a) (storm_center_position a) < 75 ∧ 
  distance (car_position b) (storm_center_position b) > 75) →
  (1 / 2 * (t_1 + t_2) = 225) :=
sorry

end car_enters_and_leaves_storm_l568_568562


namespace day_of_month_l568_568490

/--
The 25th day of a particular month is a Monday. 
We need to prove that the 1st day of that month is a Friday.
-/
theorem day_of_month (h : (25 % 7 = 1)) : (1 % 7 = 5) :=
sorry

end day_of_month_l568_568490


namespace prove_value_of_f_l568_568567

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 1/2 then -x^2 else
if h : -1/2 ≤ x ∧ x < 0 then -(-x)^2 else
if h : x > 1/2 then f (1 - x) else
if h : x < -1/2 then -f (-x) else 0

noncomputable def check_odd (f : ℝ → ℝ) :=
∀ x : ℝ, f (-x) = - f (x)

noncomputable def check_symmetric (f : ℝ → ℝ) :=
∀ t : ℝ, f t = f (1 - t)

theorem prove_value_of_f :
  check_odd f →
  check_symmetric f →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1/2 → f x = -x^2) →
  f 3 + f (-3/2) = -1/4 :=
by
  intros h_odd h_symm h_f
  sorry

end prove_value_of_f_l568_568567


namespace sorting_children_descending_l568_568874

theorem sorting_children_descending :
  ∃ (rearrange : Π (heights : list ℕ) (start end : ℕ), list ℕ)
  (heights : list ℕ), 
  (heights.length = 100) ∧
  (∀ start end, 0 ≤ start ∧ start < end ∧ end ≤ 100 → 
                let new_heights := rearrange heights start end 
                in new_heights.length = 100 ∧ 
                   ∀ i j, i < j ∧ (i < start ∨ i ≥ end) ∧ (j < start ∨ j ≥ end) → 
                          heights[i] ≥ heights[j]) → 
  (∃ steps, steps.length = 6 ∧ 
            ∀ i, i < 6 → let (start, end) := steps[i] in
                          0 ≤ start ∧ start < end ∧ end ≤ 100 ∧ 
                          rearrange heights start end = heights) →
                  (∀ i j, i < j → heights[i] ≥ heights[j]) := sorry

end sorting_children_descending_l568_568874


namespace pole_intersection_height_l568_568889

theorem pole_intersection_height :
  ∀ (d h1 h2 : ℝ), d = 120 ∧ h1 = 30 ∧ h2 = 90 → 
  ∃ y : ℝ, y = 18 :=
by
  sorry

end pole_intersection_height_l568_568889


namespace liars_on_black_chairs_after_movement_l568_568924

theorem liars_on_black_chairs_after_movement
    (people : ℕ) (chairs : ℕ) (initially_claimed_black : ℕ) (claimed_white_after : ℕ) (truthful_or_liar : Prop) :
  people = 40 →
  chairs = 40 →
  initially_claimed_black = 40 →
  claimed_white_after = 16 →
  (∀ p, (p=40 → (p=claimed_white_after → truthful_or_liar))) →
  8 = (liars_on_black_chairs_after_movement people chairs initially_claimed_black claimed_white_after truthful_or_liar) :=
  sorry

end liars_on_black_chairs_after_movement_l568_568924


namespace circles_lines_parallel_l568_568444

open EuclideanGeometry

noncomputable def circles_intersecting (Γ1 Γ2 : Circle) (P Q : Point) : Prop :=
Γ1.1 P ∧ Γ1.1 Q ∧ Γ2.1 P ∧ Γ2.1 Q ∧ P ≠ Q ∧ Γ1 ≠ Γ2

noncomputable def line_through_p (Γ1 Γ2 : Circle) (P A A' : Point) : Prop :=
Γ1.1 P ∧ Γ2.1 P ∧ Γ1.1 A ∧ Γ1.2 A ∧ Γ2.1 A' ∧ Γ2.2 A' ∧ P ≠ A ∧ P ≠ A'

noncomputable def line_through_q (Γ1 Γ2 : Circle) (Q B B' : Point) : Prop :=
Γ1.1 Q ∧ Γ2.1 Q ∧ Γ1.1 B ∧ Γ1.2 B ∧ Γ2.1 B' ∧ Γ2.2 B' ∧ Q ≠ B ∧ Q ≠ B'

noncomputable def lines_parallel (A B A' B' : Point) : Prop :=
∃ k : ℝ, A.x + k * (B.x - A.x) = A'.x ∧ A.y + k * (B.y - A.y) = A'.y ∧ (k ≠ 0)

theorem circles_lines_parallel
  (Γ1 Γ2 : Circle) (P Q A A' B B' : Point)
  (h_intersect : circles_intersecting Γ1 Γ2 P Q)
  (h_line_p : line_through_p Γ1 Γ2 P A A')
  (h_line_q : line_through_q Γ1 Γ2 Q B B') :
  lines_parallel A B A' B' :=
by
  sorry

end circles_lines_parallel_l568_568444


namespace differential_at_zero_l568_568652

noncomputable def y (x : ℝ) : ℝ := exp (3 * x) * log (1 + x^2)

theorem differential_at_zero :
  let dy := (3 * exp (3 * 0) * log (1 + (0:ℝ)^2) + exp (3 * 0) * (2 * (0:ℝ) / (1 + (0:ℝ)^2))) * 0.1 in
  dy = 0 :=
by
  let y' := (3 * exp (3 * 0) * log (1 + (0:ℝ)^2) + exp (3 * 0) * (2 * (0:ℝ) / (1 + (0:ℝ)^2)))
  let dy := y' * 0.1
  show dy = 0
  sorry

end differential_at_zero_l568_568652


namespace john_reads_days_per_week_l568_568769

-- Define the conditions
def john_reads_books_per_day := 4
def total_books_read := 48
def total_weeks := 6

-- Theorem statement
theorem john_reads_days_per_week :
  (total_books_read / john_reads_books_per_day) / total_weeks = 2 :=
by
  sorry

end john_reads_days_per_week_l568_568769


namespace common_chord_length_l568_568681

noncomputable def length_of_common_chord (C1 C2 : ℝ) : ℝ :=
sorry

-- Given circles C1: x^2 + y^2 = 9
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Given circles C2: x^2 + y^2 - 4x + 2y - 3 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4x + 2y - 3 = 0

-- Prove the length of their common chord
theorem common_chord_length : length_of_common_chord C1 C2 = (12 * Real.sqrt 5) / 5 :=
sorry

end common_chord_length_l568_568681


namespace coloring_scheme_count_l568_568746

/-- Given the set of points in the Cartesian plane, where each point (m, n) with
    1 <= m, n <= 6 is colored either red or blue, the number of ways to color these points
    such that each unit square has exactly two red vertices is 126. -/
theorem coloring_scheme_count 
  (color : Fin 6 → Fin 6 → Bool)
  (colored_correctly : ∀ m n, (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ 
    (color m n = true ∨ color m n = false) :=
    sorry
  )
  : (∃ valid_coloring : Nat, valid_coloring = 126) :=
  sorry

end coloring_scheme_count_l568_568746


namespace ratio_sides_to_hotdogs_l568_568293

-- Declare noncomputable theory because we will use real numbers and ratios
noncomputable theory

-- Define the main theorem with the given conditions and the required ratio proof
theorem ratio_sides_to_hotdogs :
  ∀ (chicken hamburgers hotdogs sides total_food : ℝ),
    chicken = 16 →
    hamburgers = chicken / 2 →
    hotdogs = hamburgers + 2 →
    total_food = 39 →
    chicken + hamburgers + hotdogs + sides = total_food →
    sides / hotdogs = 1 / 2 :=
by
  intros chicken hamburgers hotdogs sides total_food;
  intros h_chicken h_hamburgers h_hotdogs h_total_food h_equation;
  sorry

end ratio_sides_to_hotdogs_l568_568293


namespace find_complex_z_find_real_xy_l568_568672

namespace ComplexProofs

open Complex

-- Question 1: Determine the value of z
theorem find_complex_z (z : ℂ) (h1 : ∃ (r : ℝ), z - 1 = r * I)
    (h2 : ∃ (r : ℝ), (1 - 2 * I) * z = r) : z = 1 + 2 * I := sorry

-- Question 2: Determine the values of x and y
theorem find_real_xy (x y : ℝ) (z : ℂ) (h1 : z = 1 + 2 * I) 
    (h2 : x * z + y * conj(z) = z * conj(z)) : x = 5 / 2 ∧ y = 5 / 2 := sorry

end ComplexProofs

end find_complex_z_find_real_xy_l568_568672


namespace log_b_cot_x_eq_neg_a_l568_568022

theorem log_b_cot_x_eq_neg_a (b : ℝ) (x : ℝ) (a : ℝ) (hb : b > 1) (h₁ : tan x = 3) (h₂ : log b (tan x) = a) :
  log b (cot x) = -a :=
by
  sorry

end log_b_cot_x_eq_neg_a_l568_568022


namespace sum_of_seven_unique_digits_l568_568486

open Finset

theorem sum_of_seven_unique_digits :
  ∃ (digits : Finset ℕ), 
  digits ⊆ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  digits.card = 7 ∧
  ∃ (a b e c d f g : ℕ), 
    a ∈ digits ∧
    b ∈ digits ∧
    e ∈ digits ∧
    c ∈ digits ∧
    d ∈ digits ∧
    f ∈ digits ∧
    g ∈ digits ∧
    a + b + e = 17 ∧
    a + c + d = 18 ∧
    e + f + g = 13 ∧
    (a + b + e + c + d + f + g) = 34 := 
sorry

end sum_of_seven_unique_digits_l568_568486


namespace product_of_roots_increased_by_6_l568_568789

theorem product_of_roots_increased_by_6 :
  (let a : ℤ := 1
       b : ℤ := 17
       c : ℤ := -96
   in (c / a) + 6) = -90 := by
  sorry

end product_of_roots_increased_by_6_l568_568789


namespace exists_triangle_area_leq_7_over_72_l568_568761

noncomputable def unit_cube : set (ℝ × ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.3 ∧ p.3 ≤ 1}

axiom seventy_five_points (ps : set (ℝ × ℝ × ℝ)) : ps ⊆ unit_cube ∧ ps.card = 75 ∧ ∀ p1 p2 p3 ∈ ps, ¬collinear ℝ {p1, p2, p3}

theorem exists_triangle_area_leq_7_over_72 (ps : set (ℝ × ℝ × ℝ)) (h : seventy_five_points ps) :
  ∃ (a b c : ℝ × ℝ × ℝ), a ∈ ps ∧ b ∈ ps ∧ c ∈ ps ∧ ¬collinear ℝ {a, b, c} ∧ 
  triangle_area a b c ≤ 7 / 72 :=
sorry

end exists_triangle_area_leq_7_over_72_l568_568761


namespace solve_fraction_eq_zero_l568_568735

theorem solve_fraction_eq_zero (x : ℝ) (h : (x - 3) / (2 * x + 5) = 0) (h2 : 2 * x + 5 ≠ 0) : x = 3 :=
sorry

end solve_fraction_eq_zero_l568_568735


namespace eight_exp_neg_x_l568_568728

theorem eight_exp_neg_x (x : ℝ) (h : 8^(2 * x) = 64) : 8^(-x) = 1 / 8 :=
by
  sorry

end eight_exp_neg_x_l568_568728


namespace find_m_l568_568706

open Real

noncomputable def f (x m : ℝ) : ℝ :=
  2 * (sin x ^ 4 + cos x ^ 4) + m * (sin x + cos x) ^ 4

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x m ≤ 5) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x m = 5) :=
sorry

end find_m_l568_568706


namespace volume_of_isosceles_pyramid_l568_568091

def volume_of_pyramid (a α β : ℝ) : ℝ :=
  (a^3 / 12) * Real.cot α * Real.cot (α / 2) * Real.tan β

theorem volume_of_isosceles_pyramid (a α β : ℝ) (hα : 0 < α ∧ α < π / 2) :
  volume_of_pyramid a α β = (a^3 / 12) * Real.cot α * Real.cot (α / 2) * Real.tan β :=
by
  sorry

end volume_of_isosceles_pyramid_l568_568091


namespace volume_of_prism_l568_568213

theorem volume_of_prism {x y z : ℝ} (h1 : x * y = 72) (h2 : y * z = 75) (h3 : z * x = 80) :
  x * y * z ≈ 658 :=
by
  sorry

end volume_of_prism_l568_568213


namespace weights_diff_20_l568_568132

theorem weights_diff_20 (W : Finset ℕ) (L R : Finset ℕ) :
  (W = Finset.range 1 41) ∧
  (L = W.filter (λ x, x % 2 = 0)) ∧ (R = W.filter (λ x, x % 2 = 1)) ∧
  (L.card = 10) ∧ (R.card = 10) ∧
  (L.sum = R.sum) →
  ∃ a b ∈ L, (|a - b| = 20) ∨ ∃ a b ∈ R, (|a - b| = 20) :=
by
  sorry

end weights_diff_20_l568_568132


namespace planes_perpendicular_l568_568402

-- Defining planes and line
variables (a b g : Plane) (l : Line)

-- Stating the theorem to be proved
theorem planes_perpendicular (h1 : l ⊥ a) (h2 : l ∥ b) : a ⊥ b :=
sorry

end planes_perpendicular_l568_568402


namespace blue_balls_to_remove_l568_568203

variables (N P_red P'_red : ℕ)
variable x : ℕ

-- Define initial conditions
def total_balls := 100
def red_ball_percentage := 36 / 100
def desired_red_ball_percentage := 72 / 100

-- Problem statement
theorem blue_balls_to_remove :
  N = total_balls →
  P_red = red_ball_percentage →
  P'_red = desired_red_ball_percentage →
  let remaining_balls := N - x in
  let red_balls := N * P_red in
  P'_red * remaining_balls = red_balls →
  x = 50 :=
by
  intros hN hP_red hP'_red hr_eq
  sorry

end blue_balls_to_remove_l568_568203


namespace quadratic_variation_y_l568_568727

theorem quadratic_variation_y (k : ℝ) (x y : ℝ) (h1 : y = k * x^2) (h2 : (25 : ℝ) = k * (5 : ℝ)^2) :
  y = 25 :=
by
sorry

end quadratic_variation_y_l568_568727


namespace verify_shifted_function_l568_568406

def linear_function_shift_3_units_right (k b : ℝ) (hk : k ≠ 0) : Prop :=
  ∀ (x : ℝ), (k = -2) → (b = 6) → (λ x, -2 * (x - 3) + 6) = (λ x, k * x + b)

theorem verify_shifted_function : 
  linear_function_shift_3_units_right (-2) 6 (by norm_num) :=
sorry

end verify_shifted_function_l568_568406


namespace eight_friends_permutation_count_l568_568234

theorem eight_friends_permutation_count : 
  let n := 8 in n.factorial = 40320 :=
by
  sorry

end eight_friends_permutation_count_l568_568234


namespace sin_double_angle_value_l568_568667

open Real

theorem sin_double_angle_value (x : ℝ) 
  (h1 : sin (x + π/3) * cos (x - π/6) + sin (x - π/6) * cos (x + π/3) = 5 / 13)
  (h2 : -π/3 ≤ x ∧ x ≤ π/6) :
  sin (2 * x) = (5 * sqrt 3 - 12) / 26 :=
by
  sorry

end sin_double_angle_value_l568_568667


namespace bucket_full_weight_l568_568166

variable {a b x y : ℝ}

theorem bucket_full_weight (h1 : x + 2/3 * y = a) (h2 : x + 1/2 * y = b) : 
  (x + y) = 3 * a - 2 * b := 
sorry

end bucket_full_weight_l568_568166


namespace infinite_x_for_multiple_of_144_l568_568109

def star (a b : ℤ) : ℤ := a^2 * b

theorem infinite_x_for_multiple_of_144 :
  ∃ x : ℤ, (12 ∨ x) % 144 = 0 := by
sorry

end infinite_x_for_multiple_of_144_l568_568109


namespace coins_after_tenth_hour_l568_568139

-- Given variables representing the number of coins added or removed each hour.
def coins_put_in : ℕ :=
  20 + 30 + 30 + 40 + 50 + 60 + 70

def coins_taken_out : ℕ :=
  20 + 15 + 25

-- Definition of the full proof problem
theorem coins_after_tenth_hour :
  coins_put_in - coins_taken_out = 240 :=
by
  sorry

end coins_after_tenth_hour_l568_568139


namespace no_positive_integral_solution_l568_568642

theorem no_positive_integral_solution 
  (n : ℕ) 
  (h_pos : 0 < n) : 
  (4 + 6 + 8 + ... + 2 * (n + 1)) / (2 + 4 + 6 + ... + 2 * n) ≠ 123 / 124 := 
sorry

end no_positive_integral_solution_l568_568642


namespace point_on_diagonal_l568_568811

variables {A B C D P Q M : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variables (P Q M : A) {AP CQ : ℝ}

def is_parallelogram (A B C D : α) : Prop := 
∃ (a1 a2 b1 b2 : ℝ), 
  A = (a1, b1) ∧ B = (a2, b1) ∧ C = (a2, b2) ∧ D = (a1, b2)

def on_diagonal (P Q : ℝ) (AC : ℝ) : Prop := P + Q = AC

def parallel (x y : A) : Prop := ∃ c : ℝ, x = c • y

theorem point_on_diagonal (A B C D P Q M : A) (h1 : is_parallelogram A B C D) 
  (h2 : on_diagonal P Q (A - C)) (hAP : AP = CQ) 
  (hPM_parallel_AD : parallel (P - M) (A - D)) 
  (hQM_parallel_AB : parallel (Q - M) (B - A)) : 
  on_diagonal M (B - D) := 
sorry

end point_on_diagonal_l568_568811


namespace compute_fraction_pow_mult_l568_568983

def frac_1_3 := (1 : ℝ) / (3 : ℝ)
def frac_1_5 := (1 : ℝ) / (5 : ℝ)
def target := (1 : ℝ) / (405 : ℝ)

theorem compute_fraction_pow_mult :
  (frac_1_3^4 * frac_1_5) = target :=
by
  sorry

end compute_fraction_pow_mult_l568_568983


namespace Q_value_when_n_is_2023_l568_568456

def Q (n : ℕ) : ℚ := ∏ k in (finset.range (n - 2)).image (λ i, i + 3), (1 - 1 / k)

theorem Q_value_when_n_is_2023 : Q 2023 = 2 / 2023 :=
by
  sorry

end Q_value_when_n_is_2023_l568_568456


namespace husband_bath_towels_l568_568754

-- Define the given conditions
def kylie_bath_towels : nat := 3
def daughters_bath_towels : nat := 6
def towels_per_load : nat := 4
def number_of_loads : nat := 3

-- The main theorem to be proved
theorem husband_bath_towels : ∃ H : nat, kylie_bath_towels + daughters_bath_towels + H = towels_per_load * number_of_loads :=
by
  use 3 -- Propose the correct answer
  sorry -- Proof to be filled in

end husband_bath_towels_l568_568754


namespace max_area_triangle_BQC_l568_568434

noncomputable def triangle_area_problem (AB BC CA : ℝ) (E : ℝ) (d e f : ℝ) : ℝ :=
let cosBAC := (AB^2 + CA^2 - BC^2) / (2 * AB * CA) in
let angle_BAC := Real.arccos cosBAC in
let max_area := 162 - 81 * Real.sqrt 3 in
if AB = 12 ∧ BC = 18 ∧ CA = 22 ∧ E ∈ Set.Ioo 0 BC ∧ d = 162 ∧ e = 81 ∧ f = 3 then 
  max_area
else 
  0

theorem max_area_triangle_BQC : 
  ∃ (d e f : ℕ), ∀ (AB BC CA : ℝ) (E : ℝ), (AB = 12 ∧ BC = 18 ∧ CA = 22 ∧ E ∈ Set.Ioo 0 BC) →
    triangle_area_problem AB BC CA E d e f = 162 - 81 * Real.sqrt 3 ∧ d + e + f = 246 := 
sorry

end max_area_triangle_BQC_l568_568434


namespace determine_s_l568_568031

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem determine_s (s : ℝ) (h : g (-3) s = 0) : s = -192 :=
by
  sorry

end determine_s_l568_568031


namespace range_of_values_l568_568459

variables {R : Type*} [linear_order R] [topological_space R] [order_topology R] 

-- Given conditions
def is_odd (f : R → R) : Prop := ∀ x, f (-x) = -f x
def second_derivative (f : R → R) : R → R := sorry -- assuming this is defined elsewhere

variables (f : R → R)
hypothesis h_odd : is_odd f
hypothesis h_f_neg2_zero : f (-2) = 0
hypothesis h_inequality : ∀ x, 0 < x → x * (second_derivative f x) - f x < 0

-- Proof statement (no proof, only statement with sorry)
theorem range_of_values :
  {x : R | f x > 0} = {x : R | -2 < x ∧ x < 0} ∪ {x : R | 0 < x ∧ x < 2} :=
sorry

end range_of_values_l568_568459


namespace studentsInBandOrSports_l568_568176

-- conditions definitions
def totalStudents : ℕ := 320
def studentsInBand : ℕ := 85
def studentsInSports : ℕ := 200
def studentsInBoth : ℕ := 60

-- theorem statement
theorem studentsInBandOrSports : studentsInBand + studentsInSports - studentsInBoth = 225 :=
by
  sorry

end studentsInBandOrSports_l568_568176


namespace polynomial_degree_rational_roots_l568_568937

theorem polynomial_degree_rational_roots :
  ∃ (p : polynomial ℚ), 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 500 → 
    eval (n + real.sqrt (n+1 : ℚ)) p = 0 ∧ 
    eval (n - real.sqrt (n+1 : ℚ)) p = 0) →
  degree p = 979 := 
sorry

end polynomial_degree_rational_roots_l568_568937


namespace company_buys_uniforms_l568_568930

theorem company_buys_uniforms (stores : ℕ) (uniforms_per_store : ℕ) (total_uniforms : ℕ) : 
  stores = 32 → uniforms_per_store = 4 → total_uniforms = 32 * 4 → total_uniforms = 128 :=
by
  intros h1 h2 h3
  rw [h3]
  norm_num

end company_buys_uniforms_l568_568930


namespace greatest_possible_value_MPM_l568_568222

-- Definitions according to conditions:
def is_two_digit_integer (x : ℕ) : Prop := 10 ≤ x ∧ x < 100
def is_one_digit_integer (x : ℕ) : Prop := 1 ≤ x ∧ x < 10
def different_digits (M N : ℕ) : Prop := M ≠ N

-- Problem statement:
theorem greatest_possible_value_MPM :
  ∃ M N P, is_one_digit_integer M ∧ is_one_digit_integer N ∧ is_two_digit_integer (10 * M + N) ∧ different_digits M N ∧ 
           10 * M + N * M = 100 * M + 10 * P + M ∧ 
           (∀ M' N' P', is_one_digit_integer M' ∧ is_one_digit_integer N' ∧ is_two_digit_integer (10 * M' + N') ∧ different_digits M' N' ∧ 
             10 * M' + N' * M' = 100 * M' + 10 * P' + M' → 100 * M + 10 * P + M ≥ 100 * M' + 10 * P' + M') :=
  ∃ (M N P : ℕ), M = 8 ∧ N = 9 ∧ P = 9.

end greatest_possible_value_MPM_l568_568222


namespace find_angle_B_l568_568763

noncomputable def angle_B (a b c : ℝ) (A B C : ℝ) (h : b * Real.cos A - c * Real.cos B = (c - a) * Real.cos B) (h_sum : A + B + C = Real.pi) : ℝ :=
  B

theorem find_angle_B (a b c : ℝ) (A B C : ℝ) (h : b * Real.cos A - c * Real.cos B = (c - a) * Real.cos B) (h_sum : A + B + C = Real.pi) :
  B = Real.pi / 3 :=
sorry

end find_angle_B_l568_568763


namespace solutions_exist_l568_568272

theorem solutions_exist (k : ℤ) : ∃ x y : ℤ, (x = 3 * k + 2) ∧ (y = 7 * k + 4) ∧ (7 * x - 3 * y = 2) :=
by {
  -- Proof will be filled in here
  sorry
}

end solutions_exist_l568_568272


namespace intersection_of_sets_l568_568730

def setA : Set ℝ := {x | abs (x - 1) ≤ 1}
def setB : Set ℝ := {-2, -1, 0, 1, 2}
def intersectionSet := {0, 1, 2}

theorem intersection_of_sets : setA ∩ setB = intersectionSet :=
by
  sorry

end intersection_of_sets_l568_568730


namespace batsman_new_average_l568_568185

variable (A : ℝ) -- Assume that A is the average before the 17th inning
variable (score : ℝ) -- The score in the 17th inning
variable (new_average : ℝ) -- The new average after the 17th inning

-- The conditions
axiom H1 : score = 85
axiom H2 : new_average = A + 3

-- The statement to prove
theorem batsman_new_average : 
    new_average = 37 :=
by 
  sorry

end batsman_new_average_l568_568185


namespace tim_change_l568_568138

theorem tim_change :
  ∀ (initial_amount : ℕ) (amount_paid : ℕ),
  initial_amount = 50 →
  amount_paid = 45 →
  initial_amount - amount_paid = 5 :=
by
  intros
  sorry

end tim_change_l568_568138


namespace inverse_of_log_base_3_l568_568852

def f (x : ℝ) : ℝ := Real.logBase 3 x

theorem inverse_of_log_base_3 (x : ℝ) (hx : x > 0) : ∃ g : ℝ → ℝ, ∀ y : ℝ, g (f y) = y ∧ f (g y) = y :=
by
  let g := (λ y : ℝ, 3 ^ y)
  have hg : ∀ y : ℝ, f (g y) = y := by sorry
  have h : ∀ y : ℝ, g (f y) = y := by sorry
  exact ⟨g, h, hg⟩

end inverse_of_log_base_3_l568_568852


namespace area_of_rectangular_plot_l568_568498

-- Defining the breadth
def breadth : ℕ := 26

-- Defining the length as thrice the breadth
def length : ℕ := 3 * breadth

-- Defining the area as the product of length and breadth
def area : ℕ := length * breadth

-- The theorem stating the problem to prove
theorem area_of_rectangular_plot : area = 2028 := by
  -- Initial proof step skipped
  sorry

end area_of_rectangular_plot_l568_568498


namespace resultant_number_after_trebled_l568_568938

theorem resultant_number_after_trebled (x : ℤ) (h : x = 4) : 3 * (2 * x + 9) = 51 :=
by
  rw [h]
  norm_num
  sorry

end resultant_number_after_trebled_l568_568938


namespace boundary_length_of_new_figure_l568_568216

def squareBoundaryLength : ℝ :=
  let area : ℝ := 64
  let side_length : ℝ := real.sqrt area
  let segment_length : ℝ := side_length / 4
  let num_sides : ℕ := 4
  let num_segments_per_side : ℕ := 4
  let quarter_circle_arcs_per_side : ℕ := 4
  let total_quarter_circle_arcs : ℕ := 16
  let radius : ℝ := segment_length
  let full_circle_circumference : ℝ := 2 * real.pi * radius
  let total_boundary_length : ℝ := (full_circle_circumference / 4) * total_quarter_circle_arcs
  total_boundary_length

theorem boundary_length_of_new_figure :
  squareBoundaryLength = 50.3 := by
  sorry

end boundary_length_of_new_figure_l568_568216


namespace inequality_solution_set_l568_568865

open Set

theorem inequality_solution_set (x : ℝ) : 
  ∀ x, (x - 2) * real.sqrt (x + 3) ≥ 0 ↔ (x = -3 ∨ x ≥ 2) :=
sorry

end inequality_solution_set_l568_568865


namespace dodecagon_diagonals_l568_568587

theorem dodecagon_diagonals : 
  let n := 12 in 
  (n * (n - 3)) / 2 = 54 :=
by
  sorry

end dodecagon_diagonals_l568_568587


namespace number_of_students_supporting_both_number_of_students_not_supporting_both_l568_568178

def students_total := 50
def support_A := students_total * 3 / 5
def support_B := support_A + 3

def students_supporting_both : ℕ := 21
def students_not_supporting_both : ℕ := (students_supporting_both / 3) + 1

theorem number_of_students_supporting_both  : 
    ∃ x : ℕ, 
    support_A + 3 = support_B ∧
    50 = (support_A - x) + (support_B - x) + x + ((x / 3) + 1) ∧
    x = students_supporting_both
by 
  sorry

theorem number_of_students_not_supporting_both :
    ∃ x : ℕ, 
    support_A + 3 = support_B ∧
    50 = (support_A - x) + (support_B - x) + x + ((x / 3) + 1) ∧
    x / 3 + 1 = students_not_supporting_both
by 
  sorry

end number_of_students_supporting_both_number_of_students_not_supporting_both_l568_568178


namespace fraction_power_mult_correct_l568_568990

noncomputable def fraction_power_mult : Prop :=
  (\left(\frac{1}{3} \right)^{4}) * \left(\frac{1}{5} \right) = \left(\frac{1}{405} \right)

theorem fraction_power_mult_correct : fraction_power_mult :=
by
  -- The complete proof will be here.
  sorry

end fraction_power_mult_correct_l568_568990


namespace problem_l568_568682

-- Definitions based on the given conditions
def A : ℝ × ℝ := (a, 5)
def B : ℝ × ℝ := (2, 2 - b)
def C : ℝ × ℝ := (4, 2)

-- Prove that a + b = 1, given that:
-- 1. Line AB is parallel to the x-axis
-- 2. Line AC is parallel to the y-axis
theorem problem (a b : ℝ) 
  (h1 : A.2 = B.2)          -- 5 = 2 - b
  (h2 : A.1 = C.1)          -- a = 4
  : a + b = 1 := 
by
  -- Skipping the proof
  sorry

end problem_l568_568682


namespace Sam_has_seven_watermelons_l568_568485

-- Declare the initial number of watermelons
def initial_watermelons : Nat := 4

-- Declare the additional number of watermelons Sam grew
def more_watermelons : Nat := 3

-- Prove that the total number of watermelons is 7
theorem Sam_has_seven_watermelons : initial_watermelons + more_watermelons = 7 :=
by
  sorry

end Sam_has_seven_watermelons_l568_568485


namespace zero_in_M_l568_568351

def M : Set Int := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
  by
  -- Proof is omitted
  sorry

end zero_in_M_l568_568351


namespace keystone_arch_trapezoid_angle_l568_568246

def trapezoid_larger_interior_angle := 
let n := 12 in
let total_angle_degrees := 360 in
let angle_per_section := total_angle_degrees / n in
let half_angle_per_section := angle_per_section / 2 in
let vertex_angle := 180 - half_angle_per_section in
let smaller_interior_angle := vertex_angle / 2 in
let larger_interior_angle := 180 - smaller_interior_angle in
larger_interior_angle = 97.5

theorem keystone_arch_trapezoid_angle :
  trapezoid_larger_interior_angle := by 
  sorry

end keystone_arch_trapezoid_angle_l568_568246


namespace incorrect_intersection_point_l568_568294

def linear_function (x : ℝ) : ℝ := -2 * x + 4

theorem incorrect_intersection_point : ¬(linear_function 0 = 4) :=
by {
  /- Proof can be filled here later -/
  sorry
}

end incorrect_intersection_point_l568_568294


namespace sum_of_numbers_l568_568095

theorem sum_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 130) : x + y = -7 :=
by
  sorry

end sum_of_numbers_l568_568095


namespace initial_number_of_girls_is_21_l568_568488

variables (p : ℝ) (initial_girls : ℝ) (after_leave_girls : ℝ) (new_total : ℝ)

def initially_thirty_percent_girls (p : ℝ) := initial_girls = 0.3 * p
def after_changes_group (p : ℝ) := new_total = p + 2
def after_changes_girls (p : ℝ) := after_leave_girls = 0.3 * p - 3
def twenty_five_percent_girls (new_total : ℝ) (after_leave_girls : ℝ) := (after_leave_girls / new_total) = 0.25

theorem initial_number_of_girls_is_21 (p : ℝ) :
  initially_thirty_percent_girls p →
  after_changes_group p →
  after_changes_girls p →
  twenty_five_percent_girls new_total after_leave_girls →
  initial_girls = 21 :=
by {
  intros h1 h2 h3 h4,
  sorry -- Proof will be written here.
}

end initial_number_of_girls_is_21_l568_568488


namespace polynomial_roots_l568_568658

-- Define the polynomial
def polynomial (x : ℤ) : ℤ := 
  3 * x^4 + 17 * x^3 - 23 * x^2 - 7 * x

-- Define what it means for a number to be a root of the polynomial
def is_root (x : ℝ) := polynomial x = 0

-- Specify the known roots of the polynomial
def roots : List ℝ := [0, -1/3, -4 + sqrt 23, -4 - sqrt 23]

-- The main theorem stating the roots of the polynomial
theorem polynomial_roots : ∀ x, x ∈ roots → polynomial x = 0 := by
  sorry

end polynomial_roots_l568_568658


namespace probability_point_between_lines_l568_568736

theorem probability_point_between_lines :
  let l (x : ℝ) := -2 * x + 8
  let m (x : ℝ) := -3 * x + 9
  let area_l := 1 / 2 * 4 * 8
  let area_m := 1 / 2 * 3 * 9
  let area_between := area_l - area_m
  let probability := area_between / area_l
  probability = 0.16 :=
by
  sorry

end probability_point_between_lines_l568_568736


namespace cube_section_area_l568_568316

theorem cube_section_area (a : ℝ) :
  let d := a * real.sqrt 2 in
  let area_triangle := (d^2 * real.sqrt 3) / 4 in
  (area_triangle = (a^2 * real.sqrt 3) / 2) :=
by
  let d := a * real.sqrt 2
  let area_triangle := (d^2 * real.sqrt 3) / 4
  sorry

end cube_section_area_l568_568316


namespace total_friends_met_l568_568802

def num_friends_with_pears : Nat := 9
def num_friends_with_oranges : Nat := 6

theorem total_friends_met : num_friends_with_pears + num_friends_with_oranges = 15 :=
by
  sorry

end total_friends_met_l568_568802


namespace simplify_and_substitute_substituted_value_l568_568077

theorem simplify_and_substitute (x : ℝ) (hx : x ≠ -2 ∧ x ≠ 1) :
  let expr := (1 - 3 / (x + 2)) / ((x - 1) / (x + 2) ^ 2)
  in expr = x + 2 :=
sorry

theorem substituted_value : 
  (let x := -1 in (1 - 3 / (x + 2)) / ((x - 1) / (x + 2) ^ 2)) = 1 :=
sorry

end simplify_and_substitute_substituted_value_l568_568077


namespace probability_sum_6_8_10_is_five_twelfths_l568_568932

-- Define the possible outcomes of Die A and Die B
def outcomes_A := [1, 2, 2, 4, 4, 5]
def outcomes_B := [1, 1, 3, 6, 6, 8]

-- Define a function to calculate the probability of a given sum
def prob_sum (sum : ℕ) : ℚ :=
  let total_outcomes := (outcomes_A.product outcomes_B).filter (λ (a, b), a + b = sum)
  total_outcomes.length / (outcomes_A.length * outcomes_B.length)

-- Calculate the total probability of getting a sum of 6, 8, or 10
def total_probability : ℚ := prob_sum 6 + prob_sum 8 + prob_sum 10

theorem probability_sum_6_8_10_is_five_twelfths :
  total_probability = 5 / 12 := sorry

end probability_sum_6_8_10_is_five_twelfths_l568_568932


namespace problem_l568_568862

theorem problem :
  ∀ (x y a b : ℝ), 
  |x + y| + |x - y| = 2 → 
  a > 0 → 
  b > 0 → 
  ∀ z : ℝ, 
  z = 4 * a * x + b * y → 
  (∀ (x y : ℝ), |x + y| + |x - y| = 2 → 4 * a * x + b * y ≤ 1) →
  (1 = 4 * a * 1 + b * 1) →
  (1 = 4 * a * (-1) + b * 1) →
  (1 = 4 * a * (-1) + b * (-1)) →
  (1 = 4 * a * 1 + b * (-1)) →
  ∀ a b : ℝ, a > 0 → b > 0 → (1 = 4 * a + b) →
  (a = 1 / 6 ∧ b = 1 / 3) → 
  (1 / a + 1 / b = 9) :=
by
  sorry

end problem_l568_568862


namespace exists_polyhedron_with_projections_l568_568260

theorem exists_polyhedron_with_projections :
  ∃ (P : Type) [polyhedron P], 
    (∃ (plane1 plane2 plane3 : Plane), 
      (projection P plane1) = Triangle ∧
      (projection P plane2) = Quadrilateral ∧
      (projection P plane3) = Pentagon) :=
sorry

end exists_polyhedron_with_projections_l568_568260


namespace find_slope_and_intercept_l568_568934

noncomputable def line_equation_to_slope_intercept_form 
  (x y : ℝ) : Prop :=
  (3 * (x - 2) - 4 * (y + 3) = 0) ↔ (y = (3 / 4) * x - 4.5)

theorem find_slope_and_intercept : 
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), (line_equation_to_slope_intercept_form x y) → m = 3/4 ∧ b = -4.5) :=
sorry

end find_slope_and_intercept_l568_568934


namespace inequality_proof_l568_568064

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_proof_l568_568064


namespace evaluate_rationality_l568_568264

noncomputable section

def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem evaluate_rationality :
  ¬ is_rational (Real.sqrt (4 * Real.pi^2)) ∧
  ¬ is_rational (Real.cbrt 0.64) ∧
  is_rational (Real.root 4 0.0001) ∧
  is_rational (Real.cbrt (-8) * Real.sqrt 25) :=
by
  sorry

end evaluate_rationality_l568_568264


namespace ellipse_focus_value_of_k_l568_568693

theorem ellipse_focus_value_of_k (k : ℝ) (h1 : ∃ (c : ℝ), (c = 2) ∧ (2 = sqrt ((2 / k) - 2))) : k = 1 / 3 :=
by
  sorry

end ellipse_focus_value_of_k_l568_568693


namespace value_of_x_squared_plus_9y_squared_l568_568383

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l568_568383


namespace shift_right_three_units_l568_568416

theorem shift_right_three_units (x : ℝ) : (λ x, -2 * x) (x - 3) = -2 * x + 6 :=
by
  sorry

end shift_right_three_units_l568_568416


namespace rr_sr_sum_le_one_l568_568120

noncomputable def rr_sr_le_one (r s : ℝ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : Prop :=
  r^r * s^s + r^s * s^r ≤ 1

theorem rr_sr_sum_le_one {r s : ℝ} (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : rr_sr_le_one r s h_pos_r h_pos_s h_sum :=
  sorry

end rr_sr_sum_le_one_l568_568120


namespace B_C_cooperate_l568_568188
-- Using a broader import to bring in all necessary libraries

-- Define the problem as a Lean 4 theorem
theorem B_C_cooperate (A B C: Type) [has_divide ℚ A B C]:
  (A 12) → (A 5) → (B 4) → (C 3) → ∃ x, x = 12 :=
by
  sorry

end B_C_cooperate_l568_568188


namespace trig_identity_l568_568289

theorem trig_identity : 
  sin^2 (120 * Real.pi / 180) + cos (180 * Real.pi / 180) + tan (45 * Real.pi / 180) - cos^2 (-330 * Real.pi / 180) + sin (-210 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trig_identity_l568_568289


namespace total_new_bottles_l568_568291

theorem total_new_bottles (initial_bottles : ℕ) (recycle_ratio : ℕ) (bonus_ratio : ℕ) (final_bottles : ℕ) :
  initial_bottles = 625 →
  recycle_ratio = 5 →
  bonus_ratio = 20 →
  final_bottles = 163 :=
by {
  sorry -- Proof goes here
}

end total_new_bottles_l568_568291


namespace sum_of_possible_a_l568_568125

theorem sum_of_possible_a:
  (∃ p q : ℤ, p + q = a ∧ p * q = 3 * a) → 
  (finset.sum (finset.filter (λ x, ∃ p q : ℤ, p + q = x ∧ p * q = 3 * x) 
    (finset.range 100)) = 30) :=
begin
  sorry
end

end sum_of_possible_a_l568_568125


namespace annual_rent_per_square_foot_correct_l568_568108

-- Define the dimensions of the shop
def length : ℝ := 18
def width : ℝ := 22

-- Define the monthly rent of the shop
def monthly_rent : ℝ := 2244

-- Define the area of the shop
def area : ℝ := length * width

-- Define the annual rent of the shop
def annual_rent : ℝ := monthly_rent * 12

-- Define the annual rent per square foot
def annual_rent_per_square_foot : ℝ := annual_rent / area

theorem annual_rent_per_square_foot_correct : annual_rent_per_square_foot = 68 := by
  sorry

end annual_rent_per_square_foot_correct_l568_568108


namespace granddaughter_age_is_12_l568_568612

/-
Conditions:
- Betty is 60 years old.
- Her daughter is 40 percent younger than Betty.
- Her granddaughter is one-third her mother's age.

Question:
- Prove that the granddaughter is 12 years old.
-/

def age_of_Betty := 60

def age_of_daughter (age_of_Betty : ℕ) : ℕ :=
  age_of_Betty - age_of_Betty * 40 / 100

def age_of_granddaughter (age_of_daughter : ℕ) : ℕ :=
  age_of_daughter / 3

theorem granddaughter_age_is_12 (h1 : age_of_Betty = 60) : age_of_granddaughter (age_of_daughter age_of_Betty) = 12 := by
  sorry

end granddaughter_age_is_12_l568_568612


namespace cryptarithm_solution_l568_568426

theorem cryptarithm_solution (A B : ℕ) (h_digit_A : A < 10) (h_digit_B : B < 10)
  (h_equation : 9 * (10 * A + B) = 110 * A + B) :
  A = 2 ∧ B = 5 :=
sorry

end cryptarithm_solution_l568_568426


namespace proof_fraction_l568_568879

def find_fraction (x : ℝ) : Prop :=
  (2 / 9) * x = 10 → (2 / 5) * x = 18

-- Optional, you can define x based on the condition:
noncomputable def certain_number : ℝ := 10 * (9 / 2)

theorem proof_fraction :
  find_fraction certain_number :=
by
  intro h
  sorry

end proof_fraction_l568_568879


namespace speed_of_stream_l568_568870

variable (D : ℝ) -- The distance rowed in both directions
variable (vs : ℝ) -- The speed of the stream
variable (Vb : ℝ := 78) -- The speed of the boat in still water

theorem speed_of_stream (h : (D / (Vb - vs) = 2 * (D / (Vb + vs)))) : vs = 26 := by
    sorry

end speed_of_stream_l568_568870


namespace parabola_hyperbola_intersection_l568_568334

open Real

theorem parabola_hyperbola_intersection (p : ℝ) (hp : p > 0)
  (h_hyperbola : ∀ x y, (x^2 / 4 - y^2 = 1) → (y = 2*x ∨ y = -2*x))
  (h_parabola_directrix : ∀ y, (x^2 = 2 * p * y) → (x = -p/2)) 
  (h_area_triangle : (1/2) * (p/2) * (2 * p) = 1) :
  p = sqrt 2 := sorry

end parabola_hyperbola_intersection_l568_568334


namespace product_not_divisible_by_prime_l568_568827

theorem product_not_divisible_by_prime (p a b : ℕ) (hp : Prime p) (ha : 1 ≤ a) (hpa : a < p) (hb : 1 ≤ b) (hpb : b < p) : ¬ (p ∣ (a * b)) :=
by
  sorry

end product_not_divisible_by_prime_l568_568827


namespace profit_percentage_is_20_l568_568945

def wholesale_price : ℝ := 90
def retail_price : ℝ := 120
def discount_percentage : ℝ := 10

def discount_amount : ℝ := (discount_percentage / 100) * retail_price
def selling_price : ℝ := retail_price - discount_amount
def profit : ℝ := selling_price - wholesale_price
def profit_percentage : ℝ := (profit / wholesale_price) * 100

theorem profit_percentage_is_20 : profit_percentage = 20 := by
  sorry

end profit_percentage_is_20_l568_568945


namespace second_player_wins_l568_568184

-- Defining the condition of the game
structure Game (n : ℕ) :=
  (grid_size : ℕ := n)
  (domino_size : Fin 2)
  (initial_grid : Fin n × Fin n)
  (domino_moves : List (Fin n × Fin n × Fin n × Fin n))
  (is_connected : ∀ (moves : List (Fin n × Fin n × Fin n × Fin n)), Bool)

-- The statement of the problem
theorem second_player_wins : (n : ℕ) (h : n = 100) → 
  ∀ (first_move_strategy second_move_strategy : List (Fin n × Fin n × Fin n × Fin n) → List (Fin n × Fin n × Fin n × Fin n)), 
    ∃ (second_player_wins : Bool), True := sorry

end second_player_wins_l568_568184


namespace smallest_n_subsets_have_power_of_2_or_sum_of_powers_of_2_l568_568792

def isPowerOf2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sumIsPowerOf2 (a b : ℕ) : Prop :=
  isPowerOf2 (a + b)

noncomputable def smallestNForSubset (X : Finset ℕ) (n : ℕ) : Prop :=
  ∀ A : Finset ℕ, A.card = n → (∃ a ∈ A, isPowerOf2 a) ∨ (∃ a b ∈ A, a ≠ b ∧ sumIsPowerOf2 a b)

theorem smallest_n_subsets_have_power_of_2_or_sum_of_powers_of_2 :
  smallestNForSubset (Finset.range 2002) 999 :=
sorry

end smallest_n_subsets_have_power_of_2_or_sum_of_powers_of_2_l568_568792


namespace range_of_x_l568_568341

def f (x : ℝ) : ℝ := (|x| + x) / 2 + 1

theorem range_of_x (x : ℝ) : 
  f (1 - x^2) > f (2 * x) ↔ -1 < x ∧ x < real.sqrt 2 - 1 :=
by sorry

end range_of_x_l568_568341


namespace consumption_increase_percentage_l568_568123

theorem consumption_increase_percentage (T C : ℝ) (T_pos : 0 < T) (C_pos : 0 < C) :
  (0.7 * (1 + x / 100) * T * C = 0.84 * T * C) → x = 20 :=
by sorry

end consumption_increase_percentage_l568_568123


namespace condition1_condition2_condition3_condition4_l568_568810

-- Proof for the equivalence of conditions and point descriptions

theorem condition1 (x y : ℝ) : 
  (x >= -2) ↔ ∃ y : ℝ, x = -2 ∨ x > -2 := 
by
  sorry

theorem condition2 (x y : ℝ) : 
  (-2 < x ∧ x < 2) ↔ ∃ y : ℝ, -2 < x ∧ x < 2 := 
by
  sorry

theorem condition3 (x y : ℝ) : 
  (|x| < 2) ↔ -2 < x ∧ x < 2 :=
by
  sorry

theorem condition4 (x y : ℝ) : 
  (|x| ≥ 2) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by 
  sorry

end condition1_condition2_condition3_condition4_l568_568810


namespace problem_statement_l568_568691

variable (x1 x2 x3 x4 x5 x6 x7 : ℝ)

theorem problem_statement
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 20)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 145) :
  16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 380 :=
sorry

end problem_statement_l568_568691


namespace extra_time_75_percent_speed_l568_568153

-- Defining the usual time to cover the distance
def usual_time := 72.00000000000001

-- Condition: he walks at 75% of his usual speed
def reduced_speed_factor := 0.75

-- Prove the extra time taken when walking at 75% speed
theorem extra_time_75_percent_speed (S D : ℝ) (T_extra : ℝ) :
  (D = S * usual_time) → 
  (D = reduced_speed_factor * S * (usual_time + T_extra)) → 
  T_extra = 24 := 
by
  sorry

end extra_time_75_percent_speed_l568_568153


namespace sum_of_valid_a_eq_53_l568_568128

theorem sum_of_valid_a_eq_53:
  ∀ (f : ℤ → ℤ), 
  (∀ x, f x = x^2 - (a : ℤ) * x + 3 * a) → 
  (∃ r s : ℤ, f r = 0 ∧ f s = 0 ∧ r ≠ s ∧ r + s = a ∧ r * s = 3 * a) →
  (let a_values := {a | ∃ r s, r + s = a ∧ r * s = 3 * a ∧ (a - 6)^2 = (a^2 - 12 * a)} in 
   ∑ a in (a_values.filter (λ a, a ∈ ℤ)), a = 53) := sorry

end sum_of_valid_a_eq_53_l568_568128


namespace min_edge_coloring_l568_568556

noncomputable def phones := 20
noncomputable def max_wires_per_phone := 2

-- Define the conditions: 
-- 1. There are 20 phones.
-- 2. Each wire connects two phones.
-- 3. No pair of phones is connected by more than one wire.
-- 4. No more than two wires come out of each phone.
structure Graph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  no_multiple_edges : ∀ e ∈ edges, ∀ f ∈ edges, e ≠ f → e.1 ≠ f.1 ∨ e.2 ≠ f.2
  incidence_constraint : ∀ v ∈ vertices, (edges.filter (λ e, e.1 = v ∨ e.2 = v)).card ≤ max_wires_per_phone

-- Define the specific graph for the problem
def problem_graph : Graph where
  vertices := Finset.range phones
  edges := {edge | ∃ v w, (v < w) ∧ v ∈ Finset.range phones ∧ w ∈ Finset.range phones ∧ abs (v - w) ≤ max_wires_per_phone}
  no_multiple_edges := sorry
  incidence_constraint := sorry

-- The theorem to prove: 
-- The minimum number of colors required to edge-color the graph such that no two edges incident to the same vertex share the same color is 2.
theorem min_edge_coloring (G : Graph) : ∃ k, k ≤ 2 ∧ ∀ coloring : G.edges → Fin k, 
  ∀ v ∈ G.vertices, ∀ e1 e2 ∈ G.edges, 
    v = e1.1 ∨ v = e1.2 ∨ v = e2.1 ∨ v = e2.2 → e1 ≠ e2 → coloring e1 ≠ coloring e2 :=
sorry

end min_edge_coloring_l568_568556


namespace normal_line_eq_enclosed_area_eq_l568_568199

noncomputable def f (x : ℝ) : ℝ := ∫ t in 0..x, 1 / (1 + t^2)

theorem normal_line_eq (x : ℝ) : 
  let normal_slope := -(1 / (∂ t in 1..1, 1 / (1 + t^2))) in
  let y_intercept := (∫ t in 0..1, 1 / (1 + t^2)) - normal_slope * 1 in
  normal_slope * x + y_intercept = -2 * x + 2 + π / 4 := sorry

theorem enclosed_area_eq : 
  let intersection_x := 1 + π / 8 in
  1 / 2 * (π / 8) * (π / 4) = π^2 / 64 := sorry

end normal_line_eq_enclosed_area_eq_l568_568199


namespace neg_p_sufficient_not_necessary_q_l568_568670

-- Definitions from the given conditions
def p (a : ℝ) : Prop := a ≥ 1
def q (a : ℝ) : Prop := a ≤ 2

-- The theorem stating the mathematical equivalence
theorem neg_p_sufficient_not_necessary_q (a : ℝ) : (¬ p a → q a) ∧ ¬ (q a → ¬ p a) := 
by sorry

end neg_p_sufficient_not_necessary_q_l568_568670


namespace reciprocal_of_neg_one_third_l568_568507

theorem reciprocal_of_neg_one_third : ∃ x : ℝ, (-1/3) * x = 1 ∧ x = -3 :=
by
  use -3
  split
  · norm_num
  · rfl

end reciprocal_of_neg_one_third_l568_568507


namespace model_y_completion_time_l568_568566

theorem model_y_completion_time
  (rate_model_x : ℕ → ℝ)
  (rate_model_y : ℕ → ℝ)
  (num_model_x : ℕ)
  (num_model_y : ℕ)
  (time_model_x : ℝ)
  (combined_rate : ℝ)
  (same_number : num_model_y = num_model_x)
  (task_completion_x : ∀ x, rate_model_x x = 1 / time_model_x)
  (total_model_x : num_model_x = 24)
  (task_completion_y : ∀ y, rate_model_y y = 1 / y)
  (one_minute_completion : num_model_x * rate_model_x 1 + num_model_y * rate_model_y 36 = combined_rate)
  : 36 = time_model_x * 2 :=
by
  sorry

end model_y_completion_time_l568_568566


namespace geometric_sequence_a4_l568_568429

theorem geometric_sequence_a4 
(a_n : ℕ → ℝ) (h1 : ∃ x : ℝ, x² - 34 * x + 81 = 0 ∧ a_n 2 = x) 
(h2 : ∃ y : ℝ, y² - 34 * y + 81 = 0 ∧ a_n 6 = y) 
(h3 : ∀ n, a_n (n + 2) = a_n n * a_n 2) 
: a_n 4 = 9 := 
sorry

end geometric_sequence_a4_l568_568429


namespace min_value_expression_l568_568671

theorem min_value_expression (x y : ℝ) (h : y^2 - 2*x + 4 = 0) : 
  ∃ z : ℝ, z = x^2 + y^2 + 2*x ∧ z = -8 :=
by
  sorry

end min_value_expression_l568_568671


namespace earring_price_l568_568054

theorem earring_price
  (necklace_price bracelet_price ensemble_price total_sales : ℝ)
  (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ)
  (total_weekend_sales : ℝ) :
  necklace_price = 25 →
  bracelet_price = 15 →
  ensemble_price = 45 →
  necklaces_sold = 5 →
  bracelets_sold = 10 →
  earrings_sold = 20 →
  ensembles_sold = 2 →
  total_weekend_sales = 565 →
  (20 * (total_weekend_sales - (5 * 25 + 10 * 15 + 2 * 45)) / 20) = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  norm_num
  sorry

end earring_price_l568_568054


namespace diff_of_squares_l568_568973

variable (a : ℝ)

theorem diff_of_squares (a : ℝ) : (a + 3) * (a - 3) = a^2 - 9 := by
  sorry

end diff_of_squares_l568_568973


namespace value_of_x2_plus_9y2_l568_568377

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l568_568377


namespace integral_of_x_squared_minus_one_n_l568_568910

theorem integral_of_x_squared_minus_one_n (n : ℕ) :
  ∫ x in -1..1, (x^2 - 1)^n = ( (-1)^n * 2^(2*n+1) * (nat.factorial n)^2 ) / (nat.factorial (2*n + 1)) :=
sorry

end integral_of_x_squared_minus_one_n_l568_568910


namespace num_integer_solutions_is_two_l568_568455

def Q (x : ℤ) : ℤ := x^4 + 8 * x^3 + 20 * x^2 + 16 * x + 64

theorem num_integer_solutions_is_two : 
  {x : ℤ | ∃ a : ℤ, Q x = a^2}.finite.card = 2 := by
  sorry

end num_integer_solutions_is_two_l568_568455


namespace domain_of_expression_l568_568649

theorem domain_of_expression (x : ℝ) : 
  x + 3 ≥ 0 → 7 - x > 0 → (x ∈ Set.Icc (-3) 7) :=
by 
  intros h1 h2
  sorry

end domain_of_expression_l568_568649


namespace transpositions_same_parity_l568_568445

theorem transpositions_same_parity (n : ℕ) (σ : equiv.perm (fin n)) 
  (m1 m2 : ℕ) 
  (h1 : ∃ l1 : list (equiv.perm (fin n)), l1.length = m1 ∧ l1.foldr (*) 1 = σ) 
  (h2 : ∃ l2 : list (equiv.perm (fin n)), l2.length = m2 ∧ l2.foldr (*) 1 = σ) :
  (m1 - m2) % 2 = 0 :=
sorry

end transpositions_same_parity_l568_568445


namespace find_valid_7_digit_numbers_l568_568643

def is_valid_digit (d : ℕ) : Prop := d = 3 ∨ d = 7

def is_valid_7_digit_number (n : ℕ) : Prop :=
  n / 1000000 ∈ {3, 7} ∧
  (n / 100000) % 10 ∈ {3, 7} ∧
  (n / 10000) % 10 ∈ {3, 7} ∧
  (n / 1000) % 10 ∈ {3, 7} ∧
  (n / 100) % 10 ∈ {3, 7} ∧
  (n / 10) % 10 ∈ {3, 7} ∧
  n % 10 ∈ {3, 7}

def is_multiple_of_21 (n : ℕ) : Prop :=
  n % 21 = 0

theorem find_valid_7_digit_numbers :
  { n : ℕ | is_valid_7_digit_number n ∧ is_multiple_of_21 n } =
  { 3373377, 7373373, 7733733, 3733737, 7337337, 3777333 } :=
by sorry

end find_valid_7_digit_numbers_l568_568643


namespace inequality_proof_l568_568461

theorem inequality_proof (n : ℕ) (x : Fin n → ℝ) (h₀ : n ≥ 2) 
  (h₁ : ∀ i, 0 < x i) (h₂ : (∑ i, x i) = 1) : 
  (∑ i : Fin n, 1 / (1 - x i)) * 
  (∑ i in Finset.Ico 0 n, ∑ j in Finset.Ico (i + 1) n, x i * x j) 
  ≤ n / 2 := 
sorry

end inequality_proof_l568_568461


namespace sin_x_cos_x_value_l568_568724

theorem sin_x_cos_x_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 :=
  sorry

end sin_x_cos_x_value_l568_568724


namespace find_MN_distance_l568_568714

noncomputable def distance_MN 
  (AB BC CD DA : ℝ) (phi : ℝ) (BC_proj_on_AD : ℝ) : ℝ :=
  let cos_half_phi := real.sqrt ((1 + real.cos phi) / 2) in
  (1 / (2 * cos_half_phi)) * (DA + BC - AB + CD)

theorem find_MN_distance :
  distance_MN 70 100 35 75 (real.acos 0.96) 96 = 25 * real.sqrt 2 :=
sorry

end find_MN_distance_l568_568714


namespace distributive_example_l568_568553

theorem distributive_example : (25 + 9) * 4 = 25 * 4 + 9 * 4 :=
by
  exact distrib 25 9 4 -- or you can directly expand and verify it

end distributive_example_l568_568553


namespace sum_of_coordinates_X_l568_568019

def Point := ℝ × ℝ

def X (Y Z : Point) : Point :=
  let (y1, y2) := Y
  let (z1, z2) := Z
  (2 * z1 + y1) / 3, (2 * z2 + y2) / 3

theorem sum_of_coordinates_X (Y Z : Point) (hY : Y = (2, 6)) (hZ : Z = (0, -6)) :
  let (x1, x2) := X Y Z
  x1 + x2 = -4 / 3 := by
  cases hY with | intro y1 y2 => cases hZ with | intro z1 z2 => sorry

end sum_of_coordinates_X_l568_568019


namespace triangle_area_ratio_l568_568758

noncomputable def equilateral_triangle_area_ratio :
  Type* :=
  ∀ (A B C T R N : Type)
    [isEquilateralTriangle A B C]
    [isCentroid T A B C]
    [isReflection R T (convexHull {A, B})]
    [isReflection N T (convexHull {B, C})],
    area A B C / area T R N = 3

axiom isEquilateralTriangle (A B C : Type) : Prop
axiom isCentroid (T A B C : Type) : Prop
axiom isReflection (R T : Type) (l : set Type) : Prop
axiom area (P Q R : Type) : ℝ

theorem triangle_area_ratio :
  equilateral_triangle_area_ratio := 
  by sorry

end triangle_area_ratio_l568_568758


namespace simple_interest_problem_l568_568915

theorem simple_interest_problem 
  (P R : ℝ)
  (h1 : 600 = (P * R * 10) / 100)
  (h2 : ∃ (P : ℝ), (R = 6000 / P) ∧ (600 = (P * (6000 / P) * 10) / 100))
  : 
  let I1 := (P * R * 5) / 100,
      I2 := (3 * P * R * 5) / 100
  in I1 + I2 = 1200 :=
by
  sorry

end simple_interest_problem_l568_568915


namespace bingley_bracelets_final_l568_568235

-- Definitions
def initial_bingley_bracelets : Nat := 5
def kelly_bracelets_given : Nat := 16 / 4
def bingley_bracelets_after_kelly : Nat := initial_bingley_bracelets + kelly_bracelets_given
def bingley_bracelets_given_to_sister : Nat := bingley_bracelets_after_kelly / 3
def bingley_remaining_bracelets : Nat := bingley_bracelets_after_kelly - bingley_bracelets_given_to_sister

-- Theorem
theorem bingley_bracelets_final : bingley_remaining_bracelets = 6 := by
  sorry

end bingley_bracelets_final_l568_568235


namespace dodecagon_diagonals_l568_568585

theorem dodecagon_diagonals : 
  let n := 12 in 
  (n * (n - 3)) / 2 = 54 :=
by
  sorry

end dodecagon_diagonals_l568_568585


namespace jeff_total_distance_l568_568766

-- Define the conditions as constants
def speed1 : ℝ := 80
def time1 : ℝ := 3

def speed2 : ℝ := 50
def time2 : ℝ := 2

def speed3 : ℝ := 70
def time3 : ℝ := 1

def speed4 : ℝ := 60
def time4 : ℝ := 2

def speed5 : ℝ := 45
def time5 : ℝ := 3

def speed6 : ℝ := 40
def time6 : ℝ := 2

def speed7 : ℝ := 30
def time7 : ℝ := 2.5

-- Define the equation for the total distance traveled
def total_distance : ℝ :=
  speed1 * time1 + 
  speed2 * time2 + 
  speed3 * time3 + 
  speed4 * time4 + 
  speed5 * time5 + 
  speed6 * time6 + 
  speed7 * time7

-- Prove that the total distance is equal to 820 miles
theorem jeff_total_distance : total_distance = 820 := by
  sorry

end jeff_total_distance_l568_568766


namespace derivative_at_zero_does_not_exist_l568_568968

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≠ 0 then (exp (x * sin (5 / x)) - 1) else 0

theorem derivative_at_zero_does_not_exist :
  ¬(∃ l : ℝ, has_deriv_at f l 0) :=
begin
  sorry
end

end derivative_at_zero_does_not_exist_l568_568968


namespace integer_solutions_count_l568_568859

theorem integer_solutions_count :
  {n : ℤ | 3 * |n - 1| - 2 * n > 2 * |3 * n + 1|}.toFinset.card = 5 :=
sorry

end integer_solutions_count_l568_568859


namespace intersection_points_l568_568252

-- Definitions and conditions
def is_ellipse (e : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, e x y ↔ x^2 + 2*y^2 = 2

def is_tangent_or_intersects (l : ℝ → ℝ) (e : ℝ → ℝ → Prop) : Prop :=
  ∃ z1 z2 : ℝ, (e z1 (l z1) ∨ e z2 (l z2))

def lines_intersect (l1 l2 : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, l1 x = l2 x

theorem intersection_points :
  ∀ (e : ℝ → ℝ → Prop) (l1 l2 : ℝ → ℝ),
  is_ellipse e →
  is_tangent_or_intersects l1 e →
  is_tangent_or_intersects l2 e →
  lines_intersect l1 l2 →
  ∃ n : ℕ, n = 2 ∨ n = 3 ∨ n = 4 :=
by
  intros e l1 l2 he hto1 hto2 hl
  sorry

end intersection_points_l568_568252


namespace find_unique_function_l568_568780

open Nat

noncomputable def equiv_fun (f : ℕ → ℕ) : Prop :=
  ∀ m n, (f(m) * f(m) + f(n)) ∣ (m * m + n) * (m * m + n)

theorem find_unique_function :
  ∀ f : ℕ → ℕ, equiv_fun f → (∀ m, f(m) = m) :=
by
  intros
  sorry

end find_unique_function_l568_568780


namespace conjugate_of_z_l568_568038

open Complex

noncomputable def z : ℂ := (|1 - I|) / (1 + I)

theorem conjugate_of_z : conjugate z = (√2 / 2) + (√2 / 2) * I := by 
  sorry

end conjugate_of_z_l568_568038


namespace area_union_of_triangles_l568_568008

noncomputable def triangle_ABC (A B C : Type) [metric_space A] [metric_space B] [metric_space C] (AB BC AC : ℝ) := 
  AB = 15 ∧ BC = 20 ∧ AC = 25

noncomputable def centroid (G : Type) [metric_space G] := true

noncomputable def rotated_points (A' B' C' G : Type) [metric_space A'] [metric_space B'] [metric_space C'] [metric_space G] := 
  true

theorem area_union_of_triangles {A B C A' B' C' G : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space A'] [metric_space B'] [metric_space C'] [metric_space G] 
  (hABC : triangle_ABC A B C 15 20 25) 
  (hC : centroid G) 
  (hR : rotated_points A' B' C' G) :
  area (union (triangle A B C) (triangle A' B' C')) = 150 := 
  sorry

end area_union_of_triangles_l568_568008


namespace verify_shifted_function_l568_568408

def linear_function_shift_3_units_right (k b : ℝ) (hk : k ≠ 0) : Prop :=
  ∀ (x : ℝ), (k = -2) → (b = 6) → (λ x, -2 * (x - 3) + 6) = (λ x, k * x + b)

theorem verify_shifted_function : 
  linear_function_shift_3_units_right (-2) 6 (by norm_num) :=
sorry

end verify_shifted_function_l568_568408


namespace find_numbers_l568_568511

theorem find_numbers (x y z : ℕ) (h_sum : x + y + z = 6) (h_prod_lt_sum : x * y * z < 6) : 
  {x, y, z} = {1, 1, 4} := 
by 
  sorry

end find_numbers_l568_568511


namespace volume_polyhedron_abc_115_l568_568550

noncomputable def volume_polyhedron (a b c : ℕ) := 
  ∃ (V : ℝ), V = (32 * ⟦sqrt 2⟧) / 81 ∧ gcd (32 : ℕ) 81 = 1 ∧ b = 2 ∧ a = 32 ∧ c = 81

theorem volume_polyhedron_abc_115 : 
  ∃ (a b c : ℕ), 
  volume_polyhedron a b c ∧ gcd a c = 1 ∧ b∣sqrt(b) ∧ b = 2 ∧ a = 32 ∧ c = 81 ∧ a + b + c = 115 :=
sorry

end volume_polyhedron_abc_115_l568_568550


namespace value_of_x_squared_plus_9y_squared_l568_568384

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l568_568384


namespace find_X_value_l568_568731

noncomputable def X (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

noncomputable def S : ℕ → ℕ
| 0     := 0
| (n+1) := S n + X n ^ 2

theorem find_X_value (n : ℕ) (hn : S n ≥ 1000) :
  X n = 22 :=
begin
  unfold X at *,
  induction n,
  { -- base case
    contradiction }, -- S 0 < 1000
  {
    -- inductive case
    sorry
  }
end

end find_X_value_l568_568731


namespace chives_planted_l568_568775

theorem chives_planted (total_rows : ℕ) (plants_per_row : ℕ)
  (parsley_rows : ℕ) (rosemary_rows : ℕ) :
  total_rows = 20 →
  plants_per_row = 10 →
  parsley_rows = 3 →
  rosemary_rows = 2 →
  (plants_per_row * (total_rows - (parsley_rows + rosemary_rows))) = 150 :=
by
  intro h1 h2 h3 h4
  sorry

end chives_planted_l568_568775


namespace four_digit_numbers_divisible_by_6_count_l568_568361

-- Definitions based on the conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def is_divisible_by_2 (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_3 (n : ℕ) : Prop := n.digits.sum % 3 = 0
def is_divisible_by_6 (n : ℕ) : Prop := is_divisible_by_2 n ∧ is_divisible_by_3 n

-- The main theorem stating the problem
theorem four_digit_numbers_divisible_by_6_count : 
  (finset.Icc 1000 9999).filter is_divisible_by_6 = 1350 :=
sorry

end four_digit_numbers_divisible_by_6_count_l568_568361


namespace none_of_the_methods_belong_to_simple_random_sampling_l568_568604

def method_1_does_not_belong_to_simple_random_sampling : Prop :=
  ¬ (∃ (S : set ℕ), infinite S ∧ (∀ (x y : ℕ), x ≠ y → (x ∈ S) ∧ (y ∈ S)))

def method_2_does_not_belong_to_simple_random_sampling : Prop :=
  (∀ (parts : set ℕ), (80 = parts.card) → (∃ (sample : finset ℕ), sample.card = 5 ∧ (∀ p ∈ sample, p ∈ parts)) → false)

def method_3_does_not_belong_to_simple_random_sampling : Prop :=
  (∃ (toys : finset ℕ), (toys.card = 20) ∧ ¬ (∀ (samples : finset ℕ), samples.card = 3 ∧ samples ⊆ toys))

def method_4_does_not_belong_to_simple_random_sampling : Prop :=
  ∀ (students : finset ℕ), students.card = 56 ∧ ∃ (tallest : finset ℕ), (tallest.card = 5 ∧ tallest ⊆ students 
    ∧ ∀ student ∈ tallest, tallest.min > students.min) → false

theorem none_of_the_methods_belong_to_simple_random_sampling :
  method_1_does_not_belong_to_simple_random_sampling ∧
  method_2_does_not_belong_to_simple_random_sampling ∧
  method_3_does_not_belong_to_simple_random_sampling ∧ 
  method_4_does_not_belong_to_simple_random_sampling :=
begin
  sorry
end

end none_of_the_methods_belong_to_simple_random_sampling_l568_568604


namespace largest_integer_y_l568_568533

theorem largest_integer_y (y : ℤ) : (y / (4:ℚ) + 3 / 7 < 2 / 3) → y ≤ 0 :=
by
  sorry

end largest_integer_y_l568_568533


namespace base7_divisible_by_5_l568_568296

theorem base7_divisible_by_5 :
  ∃ (d : ℕ), (0 ≤ d ∧ d < 7) ∧ (344 * d + 56) % 5 = 0 ↔ d = 1 :=
by
  sorry

end base7_divisible_by_5_l568_568296


namespace value_of_x_squared_plus_9y_squared_l568_568385

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l568_568385


namespace t50_mod_7_l568_568102

def T (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | n + 1 => 3 ^ T n

theorem t50_mod_7 : T 50 % 7 = 6 := sorry

end t50_mod_7_l568_568102


namespace jumping_contest_proof_l568_568101

theorem jumping_contest_proof :
  ∀ (g f m : ℕ), g = 14 → f = g + 37 → m = f - 16 → (m - g = 21) :=
by
  intros g f m h_g h_f h_m
  rw [h_g] at h_f ⊢
  rw [h_f, h_m]
  sorry

end jumping_contest_proof_l568_568101


namespace max_green_vertices_example_a_example_b_l568_568462

theorem max_green_vertices (n : ℕ) (h : n ≥ 3) : 
    let k : ℕ := ⌊(n + 1) / 2⌋ in 
    ∀ (k' : ℕ), (k' > k → ∀ (verts : Finset ℕ), verts.card = k' → ⊥) :=
begin
  sorry
end

theorem example_a : max_green_vertices 2019 (by linarith) := sorry
theorem example_b : max_green_vertices 2020 (by linarith) := sorry

end max_green_vertices_example_a_example_b_l568_568462


namespace cafeteria_pies_l568_568491

noncomputable def number_of_pies (total_apples handed_out apples_per_pie : ℝ) : ℕ :=
  Int.floor ((total_apples - handed_out) / apples_per_pie)

theorem cafeteria_pies :
  number_of_pies 135.5 89.75 5.25 = 8 :=
by 
  sorry

end cafeteria_pies_l568_568491


namespace find_GH_l568_568233

variables {A B C D H G M : Point}
variables {distance : Point → Point → ℝ}

/-- Conditions of the problem --/
constants (face_ABC : Plane) (face_BCD : Plane)
constants (angle_ABC_BCD : ℝ) (proj_A_on_BCD : Point) (orthocenter_BCD : Point)
constants (centroid_ABC : Point) (a_h_distance : ℝ) (ab_eq_ac : Prop)

/-- Given the problem conditions --/
axiom dihedral_angle_is_60 : angle_ABC_BCD = 60
axiom A_projection_is_H : proj_A_on_BCD = H
axiom H_is_orthocenter_BCD : orthocenter_BCD = H
axiom G_is_centroid_ABC : centroid_ABC = G
axiom AH_is_4 : distance A H = 4
axiom AB_eq_AC : A B = A C

/-- Prove that G H = 4√21/9 --/
theorem find_GH : distance G H = (4 * sqrt 21) / 9 :=
by 
  sorry

end find_GH_l568_568233


namespace watch_correction_l568_568953

noncomputable def correction_time (loss_per_day : ℕ) (start_date : ℕ) (end_date : ℕ) (spring_forward_hour : ℕ) (correction_time_hour : ℕ) : ℝ :=
  let n_days := end_date - start_date
  let total_hours_watch := n_days * 24 + correction_time_hour - spring_forward_hour
  let loss_rate_per_hour := (loss_per_day : ℝ) / 24
  let total_loss := loss_rate_per_hour * total_hours_watch
  total_loss

theorem watch_correction :
  correction_time 3 1 5 1 6 = 6.625 :=
by
  sorry

end watch_correction_l568_568953


namespace part1_solution_set_part2_min_value_l568_568343

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 2|

theorem part1_solution_set : {x : ℝ | f x ≤ 2} = Icc (-5) 1 :=
by
  sorry

theorem part2_min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  (∃ (a b c : ℝ), a + 2 * b + 3 * c = 9) :=
by
  sorry

end part1_solution_set_part2_min_value_l568_568343


namespace strictly_decreasing_on_0_1_l568_568653

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem strictly_decreasing_on_0_1 :
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) → (deriv f x < 0) :=
by
  intro x hx
  have h_deriv : deriv f x = x - 1 / x := by
    calc
      deriv (λ x, (1 / 2) * x^2) x - deriv (λ x, Real.log x) x
    ... = x - 1 / x : sorry -- the details of derivative calculation
  rw h_deriv
  -- need to show x - 1 / x < 0 for 0 < x ≤ 1
  sorry

end strictly_decreasing_on_0_1_l568_568653


namespace complex_magnitude_l568_568024
open Complex

theorem complex_magnitude (z w : ℂ)
  (h1 : |3 * z - w| = 15)
  (h2 : |z + 3 * w| = 10)
  (h3 : |z - w| = 3) :
  |z| = 6 :=
sorry

end complex_magnitude_l568_568024


namespace value_of_square_sum_l568_568371

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l568_568371


namespace find_divisor_l568_568809

/-- Given a dividend of 15698, a quotient of 89, and a remainder of 14, find the divisor. -/
theorem find_divisor :
  ∃ D : ℕ, 15698 = 89 * D + 14 ∧ D = 176 :=
by
  sorry

end find_divisor_l568_568809


namespace sin_cos_identity_l568_568723

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end sin_cos_identity_l568_568723


namespace final_selling_price_l568_568205

theorem final_selling_price (
  original_price : ℝ,
  discount_rate : ℝ,
  vat_rate : ℝ,
  initial_exchange_rate : ℝ,
  loss_rate : ℝ,
  selling_discount_rate : ℝ,
  sale_exchange_rate : ℝ,
  sales_tax_rate : ℝ
) : 
  let discounted_price := original_price * (1 - discount_rate) in
  let final_purchase_price := discounted_price * (1 + vat_rate) in
  let selling_price_before_discount := original_price * (1 - loss_rate) in
  let selling_price_after_discount := selling_price_before_discount * (1 - selling_discount_rate) in
  let final_selling_price_in_rs := selling_price_after_discount * (1 + sales_tax_rate) in
  let final_selling_price_in_usd := final_selling_price_in_rs / sale_exchange_rate in
  final_selling_price_in_usd = 13.23 :=
by
  have discount_rate_eq : discount_rate = 0.05 := rfl
  have vat_rate_eq : vat_rate = 0.15 := rfl
  have initial_exchange_rate_eq : initial_exchange_rate = 70 := rfl
  have loss_rate_eq : loss_rate = 0.25 := rfl
  have selling_discount_rate_eq : selling_discount_rate = 0.10 := rfl
  have sale_exchange_rate_eq : sale_exchange_rate = 75 := rfl
  have sales_tax_rate_eq : sales_tax_rate = 0.05 := rfl

  have discounted_price_eq : discounted_price = original_price * (1 - discount_rate) := rfl
  have final_purchase_price_eq : final_purchase_price = discounted_price * (1 + vat_rate) := rfl
  have selling_price_before_discount_eq : selling_price_before_discount = original_price * (1 - loss_rate) := rfl
  have selling_price_after_discount_eq : selling_price_after_discount = selling_price_before_discount * (1 - selling_discount_rate) := rfl
  have final_selling_price_in_rs_eq : final_selling_price_in_rs = selling_price_after_discount * (1 + sales_tax_rate) := rfl
  have final_selling_price_in_usd_eq : final_selling_price_in_usd = final_selling_price_in_rs / sale_exchange_rate := rfl

  sorry

end final_selling_price_l568_568205


namespace find_m_l568_568699

noncomputable def m_value (α : ℝ) (m : ℝ) : Prop :=
  let P := (-8 * m, -6 * sin (Real.pi / 6))
  (cos α = -4 / 5) ∧ (P = (-8 * m, -3)) →
  m = 1 / 2

theorem find_m (α : ℝ) (m : ℝ) (h : cos α = -4 / 5) (P : Prod ℝ ℝ) (hP : P = (-8 * m, -6 * sin (Real.pi / 6))) :
  m_value α m :=
by
  simp [m_value, h, hP]
  sorry

end find_m_l568_568699


namespace cos_square_minus_sin_square_15_l568_568621

theorem cos_square_minus_sin_square_15 (cos_30 : Real.cos (30 * Real.pi / 180) = (Real.sqrt 3) / 2) : 
  Real.cos (15 * Real.pi / 180) ^ 2 - Real.sin (15 * Real.pi / 180) ^ 2 = (Real.sqrt 3) / 2 := 
by 
  sorry

end cos_square_minus_sin_square_15_l568_568621


namespace binomial_sum_mod_prime_l568_568781

theorem binomial_sum_mod_prime (n : ℕ) (p : ℕ) (h : Prime p) (H : p = 2023) :
  (∑ k in Finset.range 101, Nat.choose 2020 (k + 3)) % 2023 = 578 :=
by
  sorry

end binomial_sum_mod_prime_l568_568781


namespace parabola_directrix_x_eq_neg1_eqn_l568_568118

theorem parabola_directrix_x_eq_neg1_eqn :
  (∀ y : ℝ, ∃ x : ℝ, x = -1 → y^2 = 4 * x) :=
by
  sorry

end parabola_directrix_x_eq_neg1_eqn_l568_568118


namespace verify_shifted_function_l568_568405

def linear_function_shift_3_units_right (k b : ℝ) (hk : k ≠ 0) : Prop :=
  ∀ (x : ℝ), (k = -2) → (b = 6) → (λ x, -2 * (x - 3) + 6) = (λ x, k * x + b)

theorem verify_shifted_function : 
  linear_function_shift_3_units_right (-2) 6 (by norm_num) :=
sorry

end verify_shifted_function_l568_568405


namespace range_of_x_l568_568114

theorem range_of_x (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x / (Real.sqrt (x + 2)) :=
by {
  sorry
}

end range_of_x_l568_568114


namespace distinct_real_roots_l568_568495

theorem distinct_real_roots :
  ∀ x : ℝ, (x^3 - 3*x^2 + x - 2) * (x^3 - x^2 - 4*x + 7) + 6*x^2 - 15*x + 18 = 0 ↔
  x = 1 ∨ x = -2 ∨ x = 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 1 + Real.sqrt 2 :=
by sorry

end distinct_real_roots_l568_568495


namespace crates_initially_bought_l568_568569

-- Initial conditions
variables (C : ℕ)

-- Definition of initial constraints and goal
def initial_crates (total_cost : ℕ) (lost_crates : ℕ) (profit_percent : ℕ) (sale_price : ℕ) :=
  C ≠ 0 ∧ total_cost = 160 ∧ lost_crates = 2 ∧ profit_percent = 25 ∧ sale_price = 25 ∧
  (sale_price * (C - lost_crates) = total_cost * (profit_percent + 100) / 100)

-- Theorem to prove the number of crates initially bought is 10
theorem crates_initially_bought (total_cost : ℕ := 160) (lost_crates : ℕ := 2) (profit_percent : ℕ := 25) (sale_price : ℕ := 25) :
  initial_crates C total_cost lost_crates profit_percent sale_price → C = 10 := by
  intros h
  cases h with h_nonzero h_rest
  sorry

end crates_initially_bought_l568_568569


namespace water_glass_ounces_l568_568465

theorem water_glass_ounces (glasses_per_day : ℕ) (days_per_week : ℕ)
    (bottle_ounces : ℕ) (bottle_fills_per_week : ℕ)
    (total_glasses_per_week : ℕ)
    (total_ounces_per_week : ℕ)
    (glasses_per_week_eq : glasses_per_day * days_per_week = total_glasses_per_week)
    (ounces_per_week_eq : bottle_ounces * bottle_fills_per_week = total_ounces_per_week)
    (ounce_per_glass : ℕ)
    (glasses_per_week : ℕ)
    (ounces_per_week : ℕ) :
    total_ounces_per_week / total_glasses_per_week = 5 :=
by
  sorry

end water_glass_ounces_l568_568465


namespace festival_year_l568_568999

noncomputable def population (year : ℕ) : ℕ :=
  if year < 2020 then 0
  else 500 * 2 ^ ((year - 2020) / 30)

theorem festival_year : (∃ year : ℕ, year >= 2020 ∧ population year >= 12000) →
  (∃ year : ℕ, year = 2170 ∧ population year >= 12000) :=
begin 
  sorry 
end

end festival_year_l568_568999


namespace gcd_102_238_eq_34_l568_568148

theorem gcd_102_238_eq_34 :
  Int.gcd 102 238 = 34 :=
sorry

end gcd_102_238_eq_34_l568_568148


namespace elder_age_is_30_l568_568548

-- Define the ages of the younger and elder persons
variables (y e : ℕ)

-- We have the following conditions:
-- Condition 1: The elder's age is 16 years more than the younger's age
def age_difference := e = y + 16

-- Condition 2: Six years ago, the elder's age was three times the younger's age
def six_years_ago := e - 6 = 3 * (y - 6)

-- We need to prove that the present age of the elder person is 30
theorem elder_age_is_30 (y e : ℕ) (h1 : age_difference y e) (h2 : six_years_ago y e) : e = 30 :=
sorry

end elder_age_is_30_l568_568548


namespace total_dots_on_seven_faces_is_24_l568_568888

theorem total_dots_on_seven_faces_is_24 :
  ∀ (faces : Fin 6 → ℕ),
    (∀ i, faces i ∈ {1, 2, 3, 4, 5, 6}) →
    let total_faces_sum := (∑ i, faces i) in
    let visible_faces := {4, 1, 5, 6, 2} in
    let visible_sum := visible_faces.sum in
    let remaining_faces_sum := 2 * total_faces_sum - visible_sum in
    remaining_faces_sum = 24 :=
by
  intros faces face_labels total_faces_sum visible_faces visible_sum remaining_faces_sum
  sorry

end total_dots_on_seven_faces_is_24_l568_568888


namespace sin_double_angle_cos_condition_l568_568306

theorem sin_double_angle_cos_condition (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) :
  Real.sin (2 * x) = -7 / 25 :=
sorry

end sin_double_angle_cos_condition_l568_568306


namespace star_wars_cost_l568_568837

theorem star_wars_cost 
    (LK_cost LK_earn SW_earn: ℕ) 
    (half_profit: ℕ → ℕ)
    (h1: LK_cost = 10)
    (h2: LK_earn = 200)
    (h3: SW_earn = 405)
    (h4: LK_earn - LK_cost = half_profit SW_earn)
    (h5: half_profit SW_earn * 2 = SW_earn - (LK_earn - LK_cost)) :
    ∃ SW_cost : ℕ, SW_cost = 25 := 
by
  sorry

end star_wars_cost_l568_568837


namespace find_central_angle_of_sector_l568_568333

variables (r θ : ℝ)

def sector_arc_length (r θ : ℝ) := r * θ
def sector_area (r θ : ℝ) := 0.5 * r^2 * θ

theorem find_central_angle_of_sector
  (l : ℝ)
  (A : ℝ)
  (hl : l = sector_arc_length r θ)
  (hA : A = sector_area r θ)
  (hl_val : l = 4)
  (hA_val : A = 2) :
  θ = 4 :=
sorry

end find_central_angle_of_sector_l568_568333


namespace jack_water_running_time_l568_568012

noncomputable def dripping_rate := 40 -- ml/minute
noncomputable def evaporation_rate_per_hour := 200 -- ml/hour
noncomputable def evaporation_rate := evaporation_rate_per_hour / 60 -- converting to ml/minute
noncomputable def water_dumped := 12000 -- ml
noncomputable def water_left := 7800 -- ml
noncomputable def total_water_before_dumped := water_left + water_dumped
noncomputable def net_filling_rate := dripping_rate - evaporation_rate

def time_in_minutes := total_water_before_dumped / net_filling_rate
def time_in_hours := time_in_minutes / 60 -- minutes/hour

theorem jack_water_running_time : time_in_hours = 9 := by
  -- proof steps if needed
  sorry

end jack_water_running_time_l568_568012


namespace recycled_bottles_l568_568883

-- From 729 bottles, we eventually create 364 new bottles, given that 3 old bottles can make 1 new bottle.
theorem recycled_bottles (initial_bottles : ℕ) (recycling_rate : ℕ) : initial_bottles = 729 → recycling_rate = 3 → 
∑ i in Finset.range 6, (recycling_rate^(5-i)) = 364 := by
  intros h1 h2
  rw [h1, h2]
  calc
    ∑ i in Finset.range 6, (3^(5-i))
      = (3^5 + 3^4 + 3^3 + 3^2 + 3^1 + 3^0) : by rfl
  ... = (243 + 81 + 27 + 9 + 3 + 1)        : by rfl
  ... = 364                                 : by norm_num


end recycled_bottles_l568_568883


namespace log_function_passes_through_point_l568_568555

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (x - 1) / Real.log a - 1

theorem log_function_passes_through_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -1 :=
by
  -- To complete the proof, one would argue about the properties of logarithms in specific bases.
  sorry

end log_function_passes_through_point_l568_568555


namespace count_valid_placements_is_162_l568_568581

def board : Type := fin 3 → fin 3 → option char

def is_valid_placement (b : board) : Prop :=
  (∀ r : fin 3, (∃ c : fin 3, b r c = some 'A') ∧ (∃ c : fin 3, b r c = some 'B') ∧ (∃ c : fin 3, b r c = some 'C')) ∧
  (∀ r1 r2 : fin 3, r1 ≠ r2 → ∀ c1 c2 : fin 3, c1 ≠ c2 → b r1 c1 ≠ b r2 c2)

noncomputable def count_valid_placements : ℕ :=
  finset.card {b : board | is_valid_placement b}

theorem count_valid_placements_is_162 : count_valid_placements = 162 :=
by
  sorry

end count_valid_placements_is_162_l568_568581


namespace rabbit_can_escape_l568_568210

def RabbitEscapeExists
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : Prop :=
 ∃ (rabbit_escapes : Bool), rabbit_escapes = true

theorem rabbit_can_escape
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : RabbitEscapeExists center_x center_y wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 wolf_speed rabbit_speed condition1 condition2 condition3 := 
sorry

end rabbit_can_escape_l568_568210


namespace find_minimum_value_l568_568742

variable {a : ℕ → ℝ} -- geometric sequence with all positive terms
variable {m n : ℕ} -- terms m and n
variable {q : ℝ} -- common ratio

-- the sequence is a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a n = a 0 * q ^ n

-- the standard conditions
def conditions (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) : Prop :=
geometric_sequence a q ∧ 
sqrt (a m * a n) = 8 * a 0 ∧ 
a 9 = a 8 + 2 * a 7

theorem find_minimum_value (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h : conditions a m n q) : 
  ∃ (m n : ℕ), (m + n = 8) ∧ 
  (∀ (m' n' : ℕ), m' + n' = 8 → (1/m + 4/n) ≥ (1/m' + 4/n')) ∧ 
  (1/m + 4/n = 17/15) := 
sorry

end find_minimum_value_l568_568742


namespace g_g_of_x_l568_568023

-- Definition of the function g
def g (x : ℝ) : ℝ := 1 / 2

-- Statement that we need to prove
theorem g_g_of_x : ∀ x : ℝ, g (g x) = 1 / 2 :=
by 
  intro x
  -- Proof goes here
  sorry

end g_g_of_x_l568_568023


namespace max_height_l568_568940

-- Definitions based on the conditions in a)
def h (t : ℝ) : ℝ := 180 * t - 18 * t^2

-- The theorem to prove, based on the question and correct answer in c)
theorem max_height : ∃ t : ℝ, h t = 450 :=
by
  use 5
  calc
    h 5 = 180 * 5 - 18 * 5^2 : by rfl
    ... = 900 - 18 * 25 : by rfl
    ... = 900 - 450 : by rfl
    ... = 450 : by rfl

end max_height_l568_568940


namespace smallest_num_hot_dog_packages_smallest_hot_dog_packages_l568_568543

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem smallest_num_hot_dog_packages :
  ∀ n_buns n_dogs : ℕ,
  n_buns = 8 → n_dogs = 6 →
  n_buns * 3 = n_dogs * 4 :=
begin
  sorry
end

theorem smallest_hot_dog_packages (h_buns := 8) (h_dogs := 6) :
  ∃ k, h_buns * k = h_dogs * 4 :=
begin
  use 3,
  sorry
end

end smallest_num_hot_dog_packages_smallest_hot_dog_packages_l568_568543


namespace proving_a_minus_b_l568_568228

-- Definitions based on conditions
def Alice_paid : ℕ := 130
def Bob_paid : ℕ := 160
def Charlie_paid : ℕ := 210
def total_paid : ℕ := Alice_paid + Bob_paid + Charlie_paid
def share_each : ℝ := total_paid / 3

def Alice_owes : ℝ := share_each - Alice_paid
def Bob_owes : ℝ := share_each - Bob_paid
def Charlie_owes : ℝ := Charlie_paid - share_each

def a : ℝ := Charlie_owes
def b : ℝ := Bob_owes

-- The theorem statement
theorem proving_a_minus_b : a - b = 30 :=
by
  sorry

end proving_a_minus_b_l568_568228


namespace smallest_a_l568_568021

theorem smallest_a (a b : ℝ) (h_a : a ≥ 0) (h_b : b ≥ 0) (h : ∀ x : ℤ, sin (a * x + b) = sin (37 * x)) : a = 37 :=
  sorry

end smallest_a_l568_568021


namespace john_salary_april_l568_568771

theorem john_salary_april 
  (initial_salary : ℤ)
  (raise_percentage : ℤ)
  (cut_percentage : ℤ)
  (bonus : ℤ)
  (february_salary : ℤ)
  (march_salary : ℤ)
  : initial_salary = 3000 →
    raise_percentage = 10 →
    cut_percentage = 15 →
    bonus = 500 →
    february_salary = initial_salary + (initial_salary * raise_percentage / 100) →
    march_salary = february_salary - (february_salary * cut_percentage / 100) →
    march_salary + bonus = 3305 :=
by
  intros
  sorry

end john_salary_april_l568_568771


namespace walking_distance_l568_568936

variable (x t d : ℝ)

-- Define the conditions given in the problem
def condition1 := d = x * t
def condition2 := d = (x + 1/3) * (5 * t / 6)
def condition3 := d = (x - 1/3) * (t + 3.5)

-- The main statement to prove
theorem walking_distance :
  condition1 → condition2 → condition3 → d = 35 / 96 :=
by
  intros h1 h2 h3
  sorry

end walking_distance_l568_568936


namespace a8_b8_value_l568_568805

variable {a b : ℝ}

def problem_conditions : Prop :=
  a + b = 1 ∧
  a^2 + b^2 = 3 ∧
  a^3 + b^3 = 4 ∧
  a^4 + b^4 = 7 ∧
  a^5 + b^5 = 11

theorem a8_b8_value (h : problem_conditions) : a^8 + b^8 = 47 :=
sorry

end a8_b8_value_l568_568805


namespace min_moves_to_break_chocolate_l568_568194

theorem min_moves_to_break_chocolate (n m : ℕ) (tiles : ℕ) (moves : ℕ) :
    (n = 4) → (m = 10) → (tiles = n * m) → (moves = tiles - 1) → moves = 39 :=
by
  intros hnm hn4 hm10 htm
  sorry

end min_moves_to_break_chocolate_l568_568194


namespace probability_heads_given_heads_l568_568525

-- Definitions for fair coin flips and the stopping condition
noncomputable def fair_coin_prob (event : ℕ → Prop) : ℝ :=
  sorry -- Probability function for coin events (to be defined in proofs)

-- The main statement
theorem probability_heads_given_heads :
  let p : ℝ := 1 / 3 in
  ∃ p: ℝ, p = 1 / 3 ∧ fair_coin_prob (λ n, (n = 1 ∧ (coin_flip n = (TT)) ∧ ((coin_flip (n+1) = (HH) ∨ coin_flip (n+1) = (TH))) ∧ ¬has_heads_before n)) = p :=
sorry

end probability_heads_given_heads_l568_568525


namespace compare_y1_y2_y3_l568_568791

def y1 : ℝ := 4^(0.9)
def y2 : ℝ := 8^(0.48)
def y3 : ℝ := (1 / 2)^(-1.5)

theorem compare_y1_y2_y3 : y1 > y3 ∧ y3 > y2 := by
  -- Given definitions
  let y1 := 4^(0.9)
  let y2 := 8^(0.48)
  let y3 := (1 / 2)^(-1.5)
  sorry

end compare_y1_y2_y3_l568_568791


namespace point_in_second_quadrant_range_l568_568424

theorem point_in_second_quadrant_range (m : ℝ) :
  (m - 3 < 0 ∧ m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  sorry

end point_in_second_quadrant_range_l568_568424


namespace prime_factor_of_reversed_difference_l568_568494

theorem prime_factor_of_reversed_difference (A B C : ℕ) (hA : A ≠ C) (hA_d : 1 ≤ A ∧ A ≤ 9) (hB_d : 0 ≤ B ∧ B ≤ 9) (hC_d : 1 ≤ C ∧ C ≤ 9) :
  ∃ p, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 11 := 
by
  sorry

end prime_factor_of_reversed_difference_l568_568494


namespace average_of_25_results_is_24_l568_568089

theorem average_of_25_results_is_24 
  (first12_sum : ℕ)
  (last12_sum : ℕ)
  (result13 : ℕ)
  (n1 n2 n3 : ℕ)
  (h1 : n1 = 12)
  (h2 : n2 = 12)
  (h3 : n3 = 25)
  (avg_first12 : first12_sum = 14 * n1)
  (avg_last12 : last12_sum = 17 * n2)
  (res_13 : result13 = 228) :
  (first12_sum + last12_sum + result13) / n3 = 24 :=
by
  sorry

end average_of_25_results_is_24_l568_568089


namespace exponent_comparison_of_equation_l568_568249

theorem exponent_comparison_of_equation
  (a b n p r m : ℕ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hn : 0 < n) 
  (hp : 0 < p) 
  (hr : 0 < r)
  (hm : 0 < m)
  (h_eq : ((a ^ m) * (b ^ n) / (5 ^ m * 7 ^ n * 4 ^ p) = 1 / (2 * (10 * r) ^ 31))) :
  m = 31 :=
begin
  sorry
end

end exponent_comparison_of_equation_l568_568249


namespace proof_y_coordinate_of_P_l568_568448

noncomputable def y_coordinate_of_P := 
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (-3, 2)
  let C : ℝ × ℝ := (3, 2)
  let D : ℝ × ℝ := (4, 0)
  let f (P : ℝ × ℝ) : Prop := 
    (dist P A + dist P D = 10) ∧ (dist P B + dist P C = 10)
  (∃ P : ℝ × ℝ, f P ∧ P.2 = 6 / 7)

theorem proof_y_coordinate_of_P (a b c d : ℕ) (ha : a = 6) (hb : b = 0) (hc : c = 0) (hd : d = 7) :
  y_coordinate_of_P ∧ a + b + c + d = 13 :=
by
  sorry

end proof_y_coordinate_of_P_l568_568448


namespace selling_price_of_radio_l568_568094

theorem selling_price_of_radio (CP LP : ℝ) (hCP : CP = 1500) (hLP : LP = 14.000000000000002) : 
  CP - (LP / 100 * CP) = 1290 :=
by
  -- Given definitions
  have h1 : CP - (LP / 100 * CP) = 1290 := sorry
  exact h1

end selling_price_of_radio_l568_568094


namespace probability_x_1_probability_telepathic_connection_l568_568059

-- Definitions from the problem's conditions
def player_set := {1, 2, 3, 4, 5, 6}
def x (a b : ℕ) : ℕ := abs (a - b)
def num_events := 36
def num_events_x_1 := 10
def num_events_telepathic := 16

-- Theorem statements for the mathematically equivalent proof problem
theorem probability_x_1 : (num_events_x_1 : ℚ) / num_events = 5 / 18 := sorry

theorem probability_telepathic_connection : (num_events_telepathic : ℚ) / num_events = 4 / 9 := sorry

end probability_x_1_probability_telepathic_connection_l568_568059


namespace proof_problem_l568_568241

theorem proof_problem :
  (real.pi - 2023)^0 + real.sqrt ((-2)^2) + (1 / 3)^(-2) - 4 * real.sin (real.pi / 6) = 10 :=
by
  sorry

end proof_problem_l568_568241


namespace area_ratio_BCE_ACE_l568_568007

variables {α : Type*} [ordered_field α]

structure Triangle (α : Type*) :=
  (A B C : α)

structure Point (α : Type*) :=
  (x : α)
  (y : α)

-- Given conditions
variables (A B C D E : Point α)
variable (AD DB AC BC : α)
variable [hD_on_AB : AD + DB = 25]
variable (hAD : AD = 15)
variable (hDB : DB = 10)
variable (hAC : AC = 50)
variable (hBC : BC = 45)
variable (angle_bisector_E : True) -- Simplified as True because the full geometric construction is complex

-- Proving the ratio of areas
theorem area_ratio_BCE_ACE (h_ratio : BC / AC = 9 / 10) : (area (triangle.mk B C E)) / (area (triangle.mk A C E)) = 9 / 10 := 
by
  sorry -- Proof would go here

end area_ratio_BCE_ACE_l568_568007


namespace combined_value_l568_568794

noncomputable def sum_even (a l : ℕ) : ℕ :=
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

noncomputable def sum_odd (a l : ℕ) : ℕ :=
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

theorem combined_value : 
  let i := sum_even 2 500
  let k := sum_even 8 200
  let j := sum_odd 5 133
  2 * i - k + 3 * j = 128867 :=
by
  sorry

end combined_value_l568_568794


namespace bricks_needed_l568_568717

-- Definitions
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 900
def wall_width : ℝ := 600
def wall_height : ℝ := 22.5

-- Volumes calculation
def brick_volume : ℝ := brick_length * brick_width * brick_height
def wall_volume : ℝ := wall_length * wall_width * wall_height

-- Number of bricks needed
def number_of_bricks_needed : ℝ := wall_volume / brick_volume

-- Proof statement
theorem bricks_needed :
  number_of_bricks_needed = 7200 :=
by
  sorry

end bricks_needed_l568_568717


namespace scalar_d_value_l568_568248

noncomputable def orthogonal_unit_vectors_4d (i j k w : V) : Prop :=
  (orthonormal ℝ ![i, j, k, w])

theorem scalar_d_value {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V] [fact (finite_dimensional.finrank ℝ V = 4)] 
  (i j k w : V) (h : orthogonal_unit_vectors_4d i j k w) :
  ∃ d : ℝ, (∀ (v : V), i × (v × i) + j × (v × j) + k × (v × k) + w × (v × w) = d • v) ∧ d = 3 :=
begin
  sorry
end

end scalar_d_value_l568_568248


namespace cafe_customers_at_10_30am_l568_568560

def initial_customers : ℕ := 25
def percentage_left : ℝ := 0.40
def busload_customers : ℕ := 15
def fraction_leave : ℝ := 1 / 5

theorem cafe_customers_at_10_30am :
  let customers_left := initial_customers * percentage_left;
      customers_remaining := initial_customers - customers_left;
      total_customers := customers_remaining + busload_customers;
      new_customers_left := busload_customers * fraction_leave;
      final_customers := total_customers - new_customers_left
  in final_customers = 27 :=
by
  sorry

end cafe_customers_at_10_30am_l568_568560


namespace train_crosses_bridge_time_l568_568911

theorem train_crosses_bridge_time (
  train_length : ℝ,
  bridge_length : ℝ,
  speed_kmph : ℝ)
  (h1 : train_length = 150)
  (h2 : bridge_length = 250)
  (h3 : speed_kmph = 50) :
  (train_length + bridge_length) / (speed_kmph * 1000 / 3600) ≈ 28.8 :=
sorry

end train_crosses_bridge_time_l568_568911


namespace water_level_rise_ratio_l568_568891

noncomputable def ratio_of_water_level_rise (r₁ r₂ : ℝ) (h₁ h₂ : ℝ) (s₁ s₂ : ℝ) (volume_init_eq : (1/3) * π * r₁^2 * h₁ = (1/3) * π * r₂^2 * h₂) : ℝ :=
  let Vcube := s₁^3 in
  let h₁' := h₁ + Vcube / ((1/3) * π * r₁^2) in
  let h₂' := h₂ + Vcube / ((1/3) * π * r₂^2) in
  (h₁' - h₁) / (h₂' - h₂)

theorem water_level_rise_ratio (h₁ h₂ : ℝ) (r₁ r₂ s : ℝ)
  (eq_volumes : (1/3) * π * r₁^2 * h₁ = (1/3) * π * r₂^2 * h₂) 
  (r₁_eq : r₁ = 4) (r₂_eq : r₂ = 9) (s_eq : s = 2) :
  ratio_of_water_level_rise r₁ r₂ h₁ h₂ s s eq_volumes = 81 / 16 :=
by 
  have h_ratio : h₁ / h₂ = 81 / 16 := by sorry
  have h₁' := h₁ + 8 / (π * 16 / 3); have h₂' := h₂ + 8 / (π * 81 / 3)
  have Δh₁ := h₁' - h₁; have Δh₂ := h₂' - h₂
  have Δh₁_over_Δh₂ := Δh₁ / Δh₂ = 81 / 16; sorry

end water_level_rise_ratio_l568_568891


namespace a_2009_is_65_l568_568463

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def a (i : ℕ) : ℕ :=
  let rec a_aux : ℕ → ℕ → ℕ
    | 1, n => n^2 + 1
    | k+2, n => a_aux k (sum_digits (n^2 + 1))
  in a_aux i 5

theorem a_2009_is_65 : a 2009 = 65 :=
sorry

end a_2009_is_65_l568_568463


namespace proof_conjugate_of_complex_fraction_l568_568274

noncomputable def conjugate_of_complex_fraction : Prop :=
  let z := 1 / (1 - complex.I) in
  complex.conj z = (1 / 2) - (1 / 2) * complex.I

theorem proof_conjugate_of_complex_fraction : conjugate_of_complex_fraction := by
  sorry

end proof_conjugate_of_complex_fraction_l568_568274


namespace g_ratio_l568_568098

noncomputable def g : ℝ → ℝ := sorry

theorem g_ratio :
  (∀ (c d : ℝ), c^2 * g(d) = d^2 * g(c)) → g 3 ≠ 0 → (g 6 - g 2) / g 3 = 32 / 9 := 
by
  intros h1 h2
  sorry

end g_ratio_l568_568098


namespace bug_returns_to_starting_vertex_after_8_moves_l568_568559

noncomputable def Q : ℕ → ℚ
| 0       := 1
| (n + 1) := 1 - Q n

theorem bug_returns_to_starting_vertex_after_8_moves :
  Q 8 = 1 := sorry

end bug_returns_to_starting_vertex_after_8_moves_l568_568559


namespace triangle_solution_l568_568738

variables {a b c C : ℝ}

theorem triangle_solution (h1 : sin C + cos C = 1 - sin (C / 2))
                         (h2 : a^2 + b^2 = 2 * (2 * a + sqrt 7 * b) - 11)
                         (h3 : 0 < C ∧ C < π) :
  cos C = -sqrt 7 / 4 ∧ c = 3 * sqrt 2 := 
by
  sorry

end triangle_solution_l568_568738


namespace q_at_10_l568_568447

-- Define the quadratic polynomial q
def q (x : ℝ) := -4/21 * x^2 + x + 16/21

-- Define main theorem to prove q(10) = -58/7
theorem q_at_10 : q 10 = -58/7 := by
  sorry

end q_at_10_l568_568447


namespace paco_min_cookies_l568_568476

theorem paco_min_cookies (x : ℕ) (h_initial : 25 - x ≥ 0) : 
  x + (3 + 2) ≥ 5 := by
  sorry

end paco_min_cookies_l568_568476


namespace concyclic_ARQD_ratio_MC_CL_eq_BE_CE_l568_568607

-- Definitions from conditions
def tangent_points (O : Point) (A : Point) : (Point × Point) := sorry
def midpoint (X Y : Point) : Point := sorry
def extension (X Y : Point) (ratio : ℝ) : Point := sorry
def intersection (line1 line2 : Line) : Point := sorry

axiom point_A : Point
axiom point_O : Point
axiom [tangent_ABC : tangent_points point_O point_A = (B, C)]
axiom point_D : Point := extension B C 0.5
axiom point_P : Point := midpoint point_A point_D
axiom [tangent_PQR : tangent_points point_O point_P = (Q, R)]
axiom point_E : Point := intersection (line_of_points Q R) (line_of_points B C)
axiom point_M : Point := extension C B 2
axiom point_N : Point := midpoint point_A point_M
axiom [tangent_NJK : tangent_points point_O point_N = (J, K)]
axiom point_L : Point := intersection (line_of_points J K) (line_of_points B C)

-- Proof statements
theorem concyclic_ARQD : Concyclic {point_A, R, Q, point_D} :=
by sorry

theorem ratio_MC_CL_eq_BE_CE (MC CL BE CE : ℝ) : MC / CL = BE / CE :=
by sorry

end concyclic_ARQD_ratio_MC_CL_eq_BE_CE_l568_568607


namespace simplify_expr1_simplify_expr2_l568_568242

-- Define the first problem with necessary conditions
theorem simplify_expr1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) - (b / (b - a)) = (a + b) / (a - b) :=
by
  sorry

-- Define the second problem with necessary conditions
theorem simplify_expr2 (x : ℝ) (hx1 : x ≠ -3) (hx2 : x ≠ 4) (hx3 : x ≠ -4) :
  ((x - 4) / (x + 3)) / (x - 3 - (7 / (x + 3))) = 1 / (x + 4) :=
by
  sorry

end simplify_expr1_simplify_expr2_l568_568242


namespace smallest_lucky_number_exists_l568_568058

theorem smallest_lucky_number_exists :
  ∃ (a b c d N: ℕ), 
  N = a^2 + b^2 ∧ 
  N = c^2 + d^2 ∧ 
  a - c = 7 ∧ 
  d - b = 13 ∧ 
  N = 545 := 
by {
  sorry
}

end smallest_lucky_number_exists_l568_568058


namespace part1_part2_l568_568348

theorem part1 (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 - x - m < 0) ↔ (2 < m) := by
  sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, (3 * a > 2 + a → 2 + a < x ∧ x < 3 * a) ∨ (3 * a = 2 + a → false) ∨ (3 * a < 2 + a → 3 * a < x ∧ x < 2 + a) →
  ((3 * a > 2 → 1 / 2 < a) ∧ (3 * a = 2 → a = 2 / 3) ∧ (3 * a < 2 → 0 ≤ a ∧ a < 1)) →
  2 + a = x ∨ x = 3 * a) ↔ (a ∈ Icc (2 / 3) ∞) := by
  sorry

end part1_part2_l568_568348


namespace compute_fraction_product_l568_568976

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l568_568976


namespace general_term_sequence_l568_568350

noncomputable def a (t : ℝ) (n : ℕ) : ℝ :=
if h : t ≠ 1 then (2 * (t^n - 1) / n) - 1 else 0

theorem general_term_sequence (t : ℝ) (n : ℕ) (hn : n ≠ 0) (h : t ≠ 1) :
  a t (n+1) = (2 * (t^(n+1) - 1) / (n+1)) - 1 := 
sorry

end general_term_sequence_l568_568350


namespace magnitude_of_perpendicular_l568_568716

def vector_a (x : Real) : Real × Real := (x, -1)
def vector_b : Real × Real := (1, Real.sqrt 3)

def perpendicular (a b : Real × Real) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

def magnitude (a : Real × Real) : Real :=
  Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

theorem magnitude_of_perpendicular (x : Real)
  (h : perpendicular (vector_a x) vector_b) :
  magnitude (vector_a (Real.sqrt 3)) = 2 :=
by
  sorry

end magnitude_of_perpendicular_l568_568716


namespace factorization_2109_two_digit_l568_568366

theorem factorization_2109_two_digit (a b: ℕ) : 
  2109 = a * b ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 → false :=
by
  sorry

end factorization_2109_two_digit_l568_568366


namespace most_likely_outcome_l568_568661

/-- Given a scenario where there are five children and each child is equally likely to be a boy or a girl,
    the most likely outcome is that three children are of one gender and two are of the other gender. -/
theorem most_likely_outcome : 
  ∃ p, 
  (∀ k : ℕ, (k = 0 ∨ k = 1) → (p k = 1/32)) → 
  (∀ k : ℕ, (k = 2 ∨ k = 3) → (p k = 5/16)) →
  (p 5 = 5/8) :=
sorry

end most_likely_outcome_l568_568661


namespace max_black_balls_C_is_22_l568_568134

-- Define the given parameters
noncomputable def balls_A : ℕ := 100
noncomputable def black_balls_A : ℕ := 15
noncomputable def balls_B : ℕ := 50
noncomputable def balls_C : ℕ := 80
noncomputable def probability : ℚ := 101 / 600

-- Define the maximum number of black balls in box C given the conditions
theorem max_black_balls_C_is_22 (y : ℕ) (h : (1/3 * (black_balls_A / balls_A) + 1/3 * (y / balls_B) + 1/3 * (22 / balls_C)) = probability  ) :
  ∃ (x : ℕ), x ≤ 22 := sorry

end max_black_balls_C_is_22_l568_568134


namespace fraction_power_mult_correct_l568_568988

noncomputable def fraction_power_mult : Prop :=
  (\left(\frac{1}{3} \right)^{4}) * \left(\frac{1}{5} \right) = \left(\frac{1}{405} \right)

theorem fraction_power_mult_correct : fraction_power_mult :=
by
  -- The complete proof will be here.
  sorry

end fraction_power_mult_correct_l568_568988


namespace solution_l568_568454

def f(n : ℕ) : ℕ := 
  (x : ℕ) → (y : ℕ) → (z : ℕ) → (1 ≤ x) → (1 ≤ y) → (1 ≤ z) → 4 * x + 3 * y + 2 * z = n

theorem solution : f(2009) - f(2000) = 1000 := 
by 
  sorry

end solution_l568_568454


namespace projection_of_b_onto_a_l568_568308

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := a.1 * a.1 + a.2 * a.2
  let scalar := dot_product / magnitude_squared
  (scalar * a.1, scalar * a.2)

theorem projection_of_b_onto_a :
  vector_projection (2, -1) (6, 2) = (4, -2) :=
by
  simp [vector_projection]
  sorry

end projection_of_b_onto_a_l568_568308


namespace count_valid_tables_l568_568399

def is_divisible_by (a b : Nat) : Prop := b % a = 0

def valid_table (table : List (List Nat)) : Prop :=
  table.length = 4 ∧
  (∀ row, row ∈ table → row.length = 4 ∧ (∀ col_num in [0, 1, 2, 3], is_divisible_by (table[0][col_num]) (row[col_num]))) ∧
  (∀ col, col ∈ (List.transpose table) → col.length = 4 ∧ (∀ row_num in [0, 1, 2, 3], is_divisible_by (table[row_num][0]) (col[row_num])))

theorem count_valid_tables : 
  (count_fun (λ table, valid_table table) (all_4x4_tables_possible_with_digits (Finset.range 1 10))) = 9 := sorry

end count_valid_tables_l568_568399


namespace sum_of_reciprocal_squares_lt_fraction_l568_568250

theorem sum_of_reciprocal_squares_lt_fraction (n : ℕ) :
  1 + ∑ k in Finset.range (n+1), (1 / (k+2)^2 : ℝ) < (2*n+1) / (n+1) :=
sorry

end sum_of_reciprocal_squares_lt_fraction_l568_568250


namespace probability_triangle_side_decagon_l568_568301

theorem probability_triangle_side_decagon (total_vertices : ℕ) (choose_vertices : ℕ)
  (total_triangles : ℕ) (favorable_outcomes : ℕ)
  (triangle_formula : total_vertices = 10)
  (choose_vertices_formula : choose_vertices = 3)
  (total_triangle_count_formula : total_triangles = 120)
  (favorable_outcome_count_formula : favorable_outcomes = 70)
  : (favorable_outcomes : ℚ) / total_triangles = 7 / 12 := 
by 
  sorry

end probability_triangle_side_decagon_l568_568301


namespace solve_for_y_l568_568080

theorem solve_for_y (y : ℝ) (h : 5 * (y ^ (1/3)) - 3 * (y / (y ^ (2/3))) = 10 + (y ^ (1/3))) :
  y = 1000 :=
by {
  sorry
}

end solve_for_y_l568_568080


namespace prove_diophantine_solution_l568_568496

-- Definition of non-negative integers k within the required range.
def valid_k (k : ℕ) : Prop := k ≤ 4

-- Substitution definitions for x and y in terms of k.
def x (k : ℕ) : ℕ := 13 - 3 * k
def y (k : ℕ) : ℕ := 5 * k + 2

-- The main theorem to be proved.
theorem prove_diophantine_solution :
  ∀ (k : ℕ), valid_k k → 5 * x k + 3 * y k = 71 :=
by {
  intro k,
  intro h_k,
  have : 5 * x k + 3 * y k = 5 * (13 - 3 * k) + 3 * (5 * k + 2),
  { simp [x, y] },
  calc
    5 * (13 - 3 * k) + 3 * (5 * k + 2)
       = 5 * 13 - 5 * (3 * k) + 3 * (5 * k) + 3 * 2 : by ring
   ... = 65 - 15 * k + 15 * k + 6 : by simp
   ... = 71 : by ring
}

end prove_diophantine_solution_l568_568496


namespace sqrt_div_l568_568901

theorem sqrt_div (a : Real) (b : Real) (h : (a / b = 9)) : (sqrt a) / (sqrt b) = 3 :=
by 
  sorry

end sqrt_div_l568_568901


namespace value_of_x_l568_568522

-- Define the structures and conditions of the problem
variables (r x : ℝ)

-- Define the conditions
def circles_equal_radii_and_enclosed_by_rectangle : Prop :=
  2 * r = x

def distance_between_centers : Prop :=
  dist = (2 * x) / 3

-- The main theorem to prove
theorem value_of_x 
  (h1 : circles_equal_radii_and_enclosed_by_rectangle r x)
  (h2 : distance_between_centers x) : 
  x = 6 := by
  sorry

end value_of_x_l568_568522


namespace integer_solutions_system_l568_568275

theorem integer_solutions_system :
  {x : ℤ | (4 * (1 + x) / 3 - 1 ≤ (5 + x) / 2) ∧ (x - 5 ≤ (3 * (3 * x - 2)) / 2)} = {0, 1, 2} :=
by
  sorry

end integer_solutions_system_l568_568275


namespace distance_between_points_l568_568896

noncomputable def distance (x1 y1 x2 y2 : ℝ) := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points : 
  distance (-3) (1/2) 4 (-7) = Real.sqrt 105.25 := 
by 
  sorry

end distance_between_points_l568_568896


namespace problem_1_problem_2_l568_568353

-- Define the universal set U
def U := Set.Univ

-- Define set A
def A := {x : ℝ | -1 ≤ x ∧ x < 3}

-- Define set B as a function of k
def B (k : ℝ) := {x : ℝ | x ≤ k}

-- Define the complement of B in U
def C_U (k : ℝ) := λ x, ¬ (B k x)

-- Problem 1: If k = 1, find A ∩ C_U B
theorem problem_1 : (A ∩ C_U 1) = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

-- Problem 2: If A ∩ B ≠ ∅, find the range of values for k
theorem problem_2 : (∃ x : ℝ, x ∈ A ∧ x ∈ B k) → k ≥ -1 :=
sorry

end problem_1_problem_2_l568_568353


namespace reconstruct_numbers_l568_568303

theorem reconstruct_numbers
  (x : Fin 5 → ℝ)
  (a : Fin 10 → ℝ)
  (hx_ordered : ∀ i j, i < j → x i ≤ x j)
  (ha_ordered : ∀ i j, i < j → a i ≤ a j)
  (h_a_sums : ∀ i j k l, (i ≠ j ∧ k ≠ l) → a i + a j ≠ a k + a l) :
  ∃ (x' : Fin 5 → ℝ), x' = x :=
by
  have h_sum: ∑ i, a i = 4 * ∑ i, x i := sorry
  have h_specific_sums: ∃ x1 x2 x3 x4 x5,
    a 0 = x1 + x2 ∧ a 1 = x1 + x3 ∧ a 9 = x3 + x5 ∧ a 10 = x4 + x5 := sorry
  have x3 := (∑ i, x i - a 0 - a 10) / 2 := sorry
  have x1 := (a 1 - x3) := sorry
  have x2 := (a 0 - x1) := sorry
  have x5 := (a 9 - x3) := sorry
  have x4 := (a 10 - x5) := sorry
  use [x1, x2, x3, x4, x5]
  sorry

end reconstruct_numbers_l568_568303


namespace frog_jump_sequences_l568_568449

-- Define the vertices of the regular hexagon
inductive Vertex : Type
| A | B | C | D | E | F
deriving DecidableEq, Repr

-- Define the adjacency relation for the vertices
def adjacent : Vertex → list Vertex
| Vertex.A => [Vertex.B, Vertex.F]
| Vertex.B => [Vertex.A, Vertex.C]
| Vertex.C => [Vertex.B, Vertex.D]
| Vertex.D => [Vertex.C, Vertex.E]
| Vertex.E => [Vertex.D, Vertex.F]
| Vertex.F => [Vertex.E, Vertex.A]

-- Define the movement of the frog with conditions
def frog_stops (seq: list Vertex) : Bool :=
  match seq with
  | [] => false
  | x::xs => x = Vertex.D ∨ seq.length = 5

-- Define the function to calculate the number of valid sequences
def count_sequences (start : Vertex) : Nat :=
  let rec count_sequences_aux (remaining_moves : Nat) (current_pos : Vertex) : Nat :=
    if remaining_moves = 0 then
      if current_pos = Vertex.D then 1 else 0
    else
      let next_pos := adjacent current_pos
      let valid_moves := next_pos.filter (λ v => frog_stops (v::[]))
      valid_moves.length
  count_sequences_aux 5 start

-- The main theorem
theorem frog_jump_sequences : count_sequences Vertex.A = 26 := sorry

end frog_jump_sequences_l568_568449


namespace problem_abc_l568_568062

theorem problem_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by
  sorry

end problem_abc_l568_568062


namespace max_cheeses_for_jerry_l568_568767

def is_valid_digit (d : ℕ) : Prop := d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_nine_digit_number (n : List ℕ) : Prop := 
  n.length = 9 ∧ (∀ d ∈ n, is_valid_digit d)

def two_digit_numbers (n : List ℕ) : List ℕ := 
  (List.zipWith (fun a b => 10 * a + b) n (n.tail)).take 8

def is_divisible_by_9 (x : ℕ) : Prop := x % 9 = 0

def count_divisible_by_9_pairs (n : List ℕ) : ℕ := 
  (two_digit_numbers n).countp is_divisible_by_9

theorem max_cheeses_for_jerry (n : List ℕ) :
  is_nine_digit_number n → count_divisible_by_9_pairs n ≤ 4 :=
sorry

end max_cheeses_for_jerry_l568_568767


namespace probability_closer_to_origin_l568_568941

noncomputable def rectangle : set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

def is_closer_to_origin (p : ℝ × ℝ) : Prop :=
  (p.1^2 + p.2^2) ≤ ((p.1 - 4)^2 + (p.2 - 2)^2)

theorem probability_closer_to_origin :
  (∫ p in rectangle, if is_closer_to_origin p then 1 else 0) / (∫ p in rectangle, 1) = 5 / 12 :=
sorry

end probability_closer_to_origin_l568_568941


namespace binomial_coefficient_x2_is_35_l568_568734

-- Define the problem conditions and statement
theorem binomial_coefficient_x2_is_35 :
  (∑ k in Finset.range (7 + 1), (Nat.choose 7 k) * (x^(2*(7-k)) * (1/x)^k)) = 128 →
  ∀ (T : ℕ → ℝ), T (4) = Nat.choose 7 4 * (-1)^4 * x^(14-3*4) →
  (Nat.choose 7 4 = 35) :=
begin
  intros h1 h2,
  sorry
end

end binomial_coefficient_x2_is_35_l568_568734


namespace zero_count_at_most_two_l568_568628

-- Let f be a differentiable function that is continuous on ℝ
variable {f : ℝ → ℝ}

-- f'(x) + x⁻¹ f(x) > 0 for all x ≠ 0
axiom condition_1 : ∀ (x : ℝ), x ≠ 0 → deriv f x + (1 / x) * f x > 0

-- Define the function g(x) = f(x) - x⁻¹
noncomputable def g (x : ℝ) : ℝ := f x - 1 / x

-- Theorem stating the maximum number of zeros of g(x)
theorem zero_count_at_most_two : ∃ (n ≤ 2), ∀ (a1 a2) ∈ fintype.elems (finsupp.support g), g a1 = 0 → g a2 = 0 → a1 = a2 :=
sorry

end zero_count_at_most_two_l568_568628


namespace triangle_RSP_angle_l568_568737

theorem triangle_RSP_angle (P Q R S : Type) [InnerProductSpace ℝ P] 
  [InnerProductSpace ℝ Q] [InnerProductSpace ℝ R] [InnerProductSpace ℝ S] 
  (hsq : Segment ℝ P Q S) (hRS_SQ : dist R S = dist S Q) (angle_RSQ : angle R S Q = 60) :
  angle R S P = 120 :=
sorry

end triangle_RSP_angle_l568_568737


namespace Nancy_needs_5_loads_l568_568804

/-- Definition of the given problem conditions. -/
def pieces_of_clothing (shirts sweaters socks jeans : ℕ) : ℕ :=
  shirts + sweaters + socks + jeans

def washing_machine_capacity : ℕ := 12

def loads_required (total_clothing capacity : ℕ) : ℕ :=
  (total_clothing + capacity - 1) / capacity -- integer division with rounding up

/-- Theorem statement. -/
theorem Nancy_needs_5_loads :
  loads_required (pieces_of_clothing 19 8 15 10) washing_machine_capacity = 5 :=
by
  -- Insert proof here when needed.
  sorry

end Nancy_needs_5_loads_l568_568804


namespace cube_product_l568_568535

/-- A cube is a three-dimensional shape with a specific number of vertices and faces. -/
structure Cube where
  vertices : ℕ
  faces : ℕ

theorem cube_product (C : Cube) (h1: C.vertices = 8) (h2: C.faces = 6) : 
  (C.vertices * C.faces = 48) :=
by sorry

end cube_product_l568_568535


namespace milk_owed_l568_568043

theorem milk_owed (initial_milk : ℚ) (given_milk : ℚ) (h_initial : initial_milk = 4) (h_given : given_milk = 16 / 3) :
  initial_milk - given_milk = -4 / 3 :=
by {
  rw [h_initial, h_given],
  norm_num,
}

end milk_owed_l568_568043


namespace partition_into_five_non_empty_disjoint_sets_l568_568457

section partition_problem

variables {X : Type} [Fintype X] [DecidableEq X] (n : ℕ) (K : set (X × X))
variables [Card_X : Fintype.card X = 2 * n] (hX : n ≥ 3)

-- Condition I: If (x, y) ∈ K then (y, x) ∉ K
def directed (K : set (X × X)) : Prop :=
  ∀ {x y : X}, (x, y) ∈ K → (y, x) ∉ K

-- Condition II: Every number x ∈ X belongs to at most 19 pairs of K
def bounded_pairs (K : set (X × X)) : Prop :=
  ∀ (x : X), (∃ opts : list X, opts.nodup ∧ list.length opts ≤ 19 ∧ (∀ y, (x, y) ∈ K ↔ y ∈ opts))

-- Main theorem to be proven
theorem partition_into_five_non_empty_disjoint_sets (hK1 : directed K) (hK2 : bounded_pairs K) :
  ∃ (X1 X2 X3 X4 X5 : set X), (∀ i, X1 ∪ X2 ∪ X3 ∪ X4 ∪ X5 = univ) ∧
  (∀ i, ∀ (xi yi : X), (xi ∈ X1 ∨ xi ∈ X2 ∨ xi ∈ X3 ∨ xi ∈ X4 ∨ xi ∈ X5) →
           (yi ∈ X1 ∨ yi ∈ X2 ∨ yi ∈ X3 ∨ yi ∈ X4 ∨ yi ∈ X5) →
           (xi, yi) ∈ K → (i = 1 ∧ (xi ∈ X1 ∧ yi ∈ X1)) →
           (i = 2 ∧ (xi ∈ X2 ∧ yi ∈ X2)) →
           (i = 3 ∧ (xi ∈ X3 ∧ yi ∈ X3)) →
           (i = 4 ∧ (xi ∈ X4 ∧ yi ∈ X4)) →
           (i = 5 ∧ (xi ∈ X5 ∧ yi ∈ X5)) → 
        card({(x, y) ∈ K | x ∈ Xi ∧ y ∈ Xi}) ≤ 3 * n := 
sorry

end partition_into_five_non_empty_disjoint_sets_l568_568457


namespace total_distribution_schemes_l568_568130

theorem total_distribution_schemes : 
  let computers := 6
  let total_schools := 5
  let mandatory_schools := 2
  let min_computers_per_mandatory_school := 2
  let remaining_schools := total_schools - mandatory_schools
  (distribution_schemes computers total_schools mandatory_schools min_computers_per_mandatory_school) = 15 :=
by
  sorry

noncomputable def distribution_schemes (computers total_schools mandatory_schools min_computers_per_mandatory_school : ℕ) : ℕ :=
  if (computers < mandatory_schools * min_computers_per_mandatory_school) then 0
  else
    let remaining_computers := computers - mandatory_schools * min_computers_per_mandatory_school
    let remaining_schools := total_schools - mandatory_schools
    -- Use fair non-negative integer random distribution to find the number of ways
    -- to distribute the remaining computers to the remaining schools
    let combinations := (finset.range (remaining_computers + remaining_schools)).choose (remaining_computers)
    combinations.sum

end total_distribution_schemes_l568_568130


namespace ranking_most_economical_l568_568594

theorem ranking_most_economical (c_T c_R c_J q_T q_R q_J : ℝ)
  (hR_cost : c_R = 1.25 * c_T)
  (hR_quantity : q_R = 0.75 * q_J)
  (hJ_quantity : q_J = 2.5 * q_T)
  (hJ_cost : c_J = 1.2 * c_R) :
  ((c_J / q_J) ≤ (c_R / q_R)) ∧ ((c_R / q_R) ≤ (c_T / q_T)) :=
by {
  sorry
}

end ranking_most_economical_l568_568594


namespace Charlie_wins_l568_568600

theorem Charlie_wins
  (A_wins : ℕ) (A_loses : ℕ)
  (B_wins : ℕ) (B_loses : ℕ)
  (C_loses : ℕ) :
  A_wins = 2 → A_loses = 1 →
  B_wins = 1 → B_loses = 2 →
  C_loses = 2 →
  ∃ (C_wins : ℕ), C_wins = 2 :=
by
  intros hA_wins hA_loses hB_wins hB_loses hC_loses
  have : 2 + 1 + C_wins = (8 + C_wins) / 2,
  sorry

end Charlie_wins_l568_568600


namespace problem_solution_l568_568258

-- Definitions based on conditions
def valid_sequence (b : Fin 7 → Nat) : Prop :=
  (∀ i j : Fin 7, i ≤ j → b i ≥ b j) ∧ 
  (∀ i : Fin 7, b i ≤ 1500) ∧ 
  (∀ i : Fin 7, (b i + i) % 3 = 0)

-- The main theorem
theorem problem_solution :
  (∃ b : Fin 7 → Nat, valid_sequence b) →
  @Nat.choose 506 7 % 1000 = 506 :=
sorry

end problem_solution_l568_568258


namespace sequence_formula_and_88_not_element_l568_568432

noncomputable theory

-- Define the sequence as a linear function
def a_n (n : ℕ) : ℤ := 4 * n - 2

-- Conditions given in the problem
axiom a1_equals_2 : a_n 1 = 2
axiom a17_equals_66 : a_n 17 = 66

-- Proving the general term formula and whether 88 is in the sequence
theorem sequence_formula_and_88_not_element :
  (∀ n, a_n n = 4 * n - 2) ∧ (¬ ∃ n : ℕ, a_n n = 88) :=
by sorry

end sequence_formula_and_88_not_element_l568_568432


namespace daily_sales_profit_45_selling_price_for_1200_profit_l568_568190

-- Definitions based on given conditions

def cost_price : ℤ := 30
def base_selling_price : ℤ := 40
def base_sales_volume : ℤ := 80
def price_increase_effect : ℤ := 2
def max_selling_price : ℤ := 55

-- Part (1): Prove that for a selling price of 45 yuan, the daily sales profit is 1050 yuan.
theorem daily_sales_profit_45 :
  let selling_price := 45
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = 1050 := by sorry

-- Part (2): Prove that to achieve a daily profit of 1200 yuan, the selling price should be 50 yuan.
theorem selling_price_for_1200_profit :
  let target_profit := 1200
  ∃ (selling_price : ℤ), 
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = target_profit ∧ selling_price ≤ max_selling_price ∧ selling_price = 50 := by sorry

end daily_sales_profit_45_selling_price_for_1200_profit_l568_568190


namespace rook_traversal_impossible_l568_568481

theorem rook_traversal_impossible :
  ∀ (chessboard : ℕ × ℕ) (A B : ℕ × ℕ),
  chessboard = (8, 8) →
  (A = (1,1) ∧ B = (8,8)) →
  (A.1 + A.2) % 2 = (B.1 + B.2) % 2 →
  ∃ (moves : ℕ), moves = 63 ∧ ¬ (∃ path : list (ℕ × ℕ),
    path.length = 64 ∧ 
    (∀ i, i < path.length - 1 → 
    (path.nth i).get_or_else (0, 0) ≠ (path.nth (i+1)).get_or_else (0, 0) ∧ 
    ((path.nth i).get_or_else (0, 0) = A ∧ 
     (path.nth (path.length - 1)).get_or_else (0,0) = B ∧ 
    (∀ j k, j ≠ k → (path.nth j).get_or_else (0, 0) ≠ (path.nth k).get_or_else (0, 0)))))
:= sorry

end rook_traversal_impossible_l568_568481


namespace area_of_rectangle_l568_568914

theorem area_of_rectangle (r l b : ℝ) (h1 : l = r / 6) (h2 : r = real.sqrt 1296) (h3 : b = 10) : l * b = 60 :=
by sorry

end area_of_rectangle_l568_568914


namespace count_buses_passed_l568_568613

def buses_from_Dallas_to_Houston : ℕ → ℕ → bool :=
λ t d, (t - d + 6) % 24 = 0

def buses_from_Houston_to_Dallas : ℕ → ℕ → bool :=
λ t h, (t - h + 21) % 24 = 0

def trip_duration : ℕ := 6

theorem count_buses_passed :
  ∀ (t : ℕ), (∃ (h : ℕ), h = (t + 15) % 60) →
              ∃ (n : ℕ), n = 11 :=
by
  intros t h
  simp
  sorry

end count_buses_passed_l568_568613


namespace gallons_left_l568_568045

theorem gallons_left (initial_gallons : ℚ) (gallons_given : ℚ) (gallons_left : ℚ) : 
  initial_gallons = 4 ∧ gallons_given = 16/3 → gallons_left = -4/3 :=
by
  sorry

end gallons_left_l568_568045


namespace spaghetti_cost_l568_568290

theorem spaghetti_cost (hamburger_cost french_fry_cost soda_cost spaghetti_cost split_payment friends : ℝ) 
(hamburger_count : ℕ) (french_fry_count : ℕ) (soda_count : ℕ) (friend_count : ℕ)
(h_split_payment : split_payment * friend_count = 25)
(h_hamburger_cost : hamburger_cost = 3 * hamburger_count)
(h_french_fry_cost : french_fry_cost = 1.20 * french_fry_count)
(h_soda_cost : soda_cost = 0.5 * soda_count)
(h_total_order_cost : hamburger_cost + french_fry_cost + soda_cost + spaghetti_cost = split_payment * friend_count) :
spaghetti_cost = 2.70 :=
by {
  sorry
}

end spaghetti_cost_l568_568290


namespace legs_walking_on_ground_l568_568565

def number_of_horses : ℕ := 14
def number_of_men : ℕ := number_of_horses
def legs_per_man : ℕ := 2
def legs_per_horse : ℕ := 4
def half (n : ℕ) : ℕ := n / 2

theorem legs_walking_on_ground :
  (half number_of_men) * legs_per_man + (half number_of_horses) * legs_per_horse = 42 :=
by
  sorry

end legs_walking_on_ground_l568_568565


namespace isosceles_trapezoid_diagonal_eq_l568_568964

theorem isosceles_trapezoid_diagonal_eq {a b r : ℝ} (h : ℝ) 
  (h_h : h = 2 * r) (tangent_points : a > 0 ∧ b > 0 ∧ r > 0) :
  let BD := (1 / 2) * real.sqrt(a^2 + 6 * a * b + b^2)
  in 
  true :=
sorry

end isosceles_trapezoid_diagonal_eq_l568_568964


namespace convert_101101_is_correct_l568_568254

def bin_to_dec (n : Nat) : Nat := 
  List.foldl (λ acc d, acc * 2 + d) 0 (Nat.digits 2 n)

def dec_to_base (n b : Nat) : List Nat := 
  if n = 0 then [0]
  else 
    let rec f (n : Nat) : List Nat :=
      if n = 0 then [] else (n % b) :: f (n / b)
    f n

theorem convert_101101_is_correct :
  bin_to_dec 0b101101 = 45 ∧ dec_to_base 45 7 = [3, 6] := 
by
  sorry

end convert_101101_is_correct_l568_568254


namespace coffee_ounces_per_cup_l568_568798

theorem coffee_ounces_per_cup :
  (∀ cups_per_day : ℕ, cups_per_day = 2) →
  (∀ bean_cost_per_bag : ℝ, bean_cost_per_bag = 8) →
  (∀ beans_per_bag : ℝ, beans_per_bag = 10.5) →
  (∀ milk_usage_per_week : ℝ, milk_usage_per_week = 1/2) →
  (∀ milk_cost_per_gallon : ℝ, milk_cost_per_gallon = 4) →
  (∀ coffee_expense_per_week : ℝ, coffee_expense_per_week = 18) →
  (∃ ounces_per_cup : ℝ, ounces_per_cup = 1.5) :=
by
  intro cups_per_day cups_per_day_eq
  intro bean_cost_per_bag bean_cost_per_bag_eq
  intro beans_per_bag beans_per_bag_eq
  intro milk_usage_per_week milk_usage_per_week_eq
  intro milk_cost_per_gallon milk_cost_per_gallon_eq
  intro coffee_expense_per_week coffee_expense_per_week_eq
  use 1.5
  sorry

end coffee_ounces_per_cup_l568_568798


namespace eval_expr_equals_1_l568_568829

noncomputable def eval_expr (a b : ℕ) : ℚ :=
  (a + b) / (a * b) / ((a / b) - (b / a))

theorem eval_expr_equals_1 (a b : ℕ) (h₁ : a = 3) (h₂ : b = 2) : eval_expr a b = 1 :=
by
  sorry

end eval_expr_equals_1_l568_568829


namespace rectangular_solid_length_l568_568898

theorem rectangular_solid_length (w h A l : ℝ) 
  (hw : w = 9) 
  (hh : h = 6) 
  (hA : A = 408) 
  (h_formula : A = 2 * l * w + 2 * l * h + 2 * w * h) : 
  l = 10 :=
by {
  subst hw,
  subst hh,
  subst hA,
  simp at h_formula,
  sorry
}

end rectangular_solid_length_l568_568898


namespace area_of_triangle_AED_l568_568608

-- Definitions according to the conditions
variables {A E D C B : Type}
variables [AffineSpace ℝ A C]
variables [AffineSpace ℝ A E]
variables [AffineSpace ℝ A D]
variables [AffineSpace ℝ C B]
variables [LinearOrderedField ℝ]

-- Points and their collinearity
variable (collinear_CEB : AffineSpan ℝ ({(C, B), (E, B), (C, E)}) = AffineSpan ℝ ({(C, B)}))

-- Perpendicularity of CB and AB
variable (perpendicular_CB_AB : ∃ (C : A), LineThrough C B ⊥ LineThrough A B)

-- Parallelism of AE and DC
variable (parallel_AE_DC : ∃ (D : A), LineThrough A E ∥ LineThrough D C)

-- Lengths of AB and CE
variable (length_AB : dist A B = 8)
variable (length_CE : dist C E = 5)

-- We need to prove the area of triangle AED
theorem area_of_triangle_AED : 
  ∃ (area : ℝ), area = 20 :=
by
  sorry

end area_of_triangle_AED_l568_568608


namespace number_of_positive_numbers_l568_568996

noncomputable def count_positive_numbers (s : set ℝ) : ℕ :=
  finset.card (finset.filter (λ x, 0 < x) s.to_finset)

theorem number_of_positive_numbers :
  count_positive_numbers ({5⁻², 2², 0, real.tan (real.pi / 4), -3.cbrt (-27)} : set ℝ) = 3 := by
  sorry

end number_of_positive_numbers_l568_568996


namespace find_f_2011_l568_568919

open Function

variable {R : Type} [Field R]

def functional_equation (f : R → R) : Prop :=
  ∀ a b : R, f (a * f b) = a * b

theorem find_f_2011 (f : ℝ → ℝ) (h : functional_equation f) : f 2011 = 2011 :=
sorry

end find_f_2011_l568_568919


namespace geometric_sequence_product_l568_568336

theorem geometric_sequence_product (b : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, b (n+1) = b n * r)
  (h_b9 : b 9 = (3 + 5) / 2) : b 1 * b 17 = 16 :=
by
  sorry

end geometric_sequence_product_l568_568336


namespace trip_time_difference_l568_568561

def travel_time (distance speed : ℕ) : ℕ :=
  distance / speed

theorem trip_time_difference
  (speed : ℕ)
  (speed_pos : 0 < speed)
  (distance1 : ℕ)
  (distance2 : ℕ)
  (time_difference : ℕ)
  (h1 : distance1 = 540)
  (h2 : distance2 = 600)
  (h_speed : speed = 60)
  (h_time_diff : time_difference = (travel_time distance2 speed) - (travel_time distance1 speed) * 60)
  : time_difference = 60 :=
by
  sorry

end trip_time_difference_l568_568561


namespace chives_planted_l568_568774

theorem chives_planted (total_rows : ℕ) (plants_per_row : ℕ)
  (parsley_rows : ℕ) (rosemary_rows : ℕ) :
  total_rows = 20 →
  plants_per_row = 10 →
  parsley_rows = 3 →
  rosemary_rows = 2 →
  (plants_per_row * (total_rows - (parsley_rows + rosemary_rows))) = 150 :=
by
  intro h1 h2 h3 h4
  sorry

end chives_planted_l568_568774


namespace value_of_square_sum_l568_568374

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l568_568374


namespace purely_imaginary_complex_expression_l568_568332

-- Stating the problem in Lean 4

theorem purely_imaginary_complex_expression (a : ℝ) (h: (a^2 - 1) + (a - 1) * complex.i).im = (a - 1) * complex.i ∧ (a^2 - 1) + (a - 1) * complex.i).re = 0 : 
  (complex.of_real (a^2) + complex.i) / (complex.of_real 1 + complex.of_real a * complex.i) = complex.i :=
sorry

end purely_imaginary_complex_expression_l568_568332


namespace probability_A_given_B_probability_A_or_B_l568_568143

-- Definitions of the given conditions
def PA : ℝ := 0.2
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

-- Theorem to prove the probability that city A also experiences rain when city B is rainy
theorem probability_A_given_B : PA * PB = PAB -> PA = 2 / 3 := by
  sorry

-- Theorem to prove the probability that at least one of the two cities experiences rain
theorem probability_A_or_B (PA PB PAB : ℝ) : (PA + PB - PAB) = 0.26 := by
  sorry

end probability_A_given_B_probability_A_or_B_l568_568143


namespace extra_spacy_subsets_count_l568_568629

def is_extra_spacy (S : set ℕ) : Prop :=
  ∀ n ∈ S, ∀ k ∈ S, (0 < k - n) ∧ (k - n < 4) → false

def d : ℕ → ℕ 
| 0 := 1
| 1 := 2
| 2 := 3
| 3 := 4
| 4 := 5
| (n+5) := d n + d (n+1)

theorem extra_spacy_subsets_count : d 15 = _ :=
sorry

end extra_spacy_subsets_count_l568_568629


namespace ratio_A_B_l568_568119

theorem ratio_A_B (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : 5 * C = 8 * B) : A / B = 2 / 3 := 
by sorry

end ratio_A_B_l568_568119


namespace right_triangle_area_semi_perimeter_inequality_l568_568819

theorem right_triangle_area_semi_perimeter_inequality 
  (x y : ℝ) (h : x > 0 ∧ y > 0) 
  (p : ℝ := (x + y + Real.sqrt (x^2 + y^2)) / 2)
  (S : ℝ := x * y / 2) 
  (hypotenuse : ℝ := Real.sqrt (x^2 + y^2)) 
  (right_triangle : hypotenuse ^ 2 = x ^ 2 + y ^ 2) : 
  S <= p^2 / 5.5 := 
sorry

end right_triangle_area_semi_perimeter_inequality_l568_568819


namespace molecular_weight_of_compound_l568_568162

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00
def num_C : ℕ := 4
def num_H : ℕ := 1
def num_O : ℕ := 1

theorem molecular_weight_of_compound : 
  (num_C * atomic_weight_C + num_H * atomic_weight_H + num_O * atomic_weight_O) = 65.048 := 
  by 
  -- proof skipped
  sorry

end molecular_weight_of_compound_l568_568162


namespace term_100_is_981_l568_568320

def sequence_term (n : ℕ) : ℕ :=
  if n = 100 then 981 else sorry

theorem term_100_is_981 : sequence_term 100 = 981 := by
  rfl

end term_100_is_981_l568_568320


namespace probability_smallest_divides_product_l568_568880

theorem probability_smallest_divides_product : 
  let S := {1, 2, 3, 4, 5, 6}
  let total_combinations := Nat.choose 6 3
  let successful_combinations := 10 + 1 + 2
  (successful_combinations / total_combinations : ℚ) = 13 / 20 := by
sorry

end probability_smallest_divides_product_l568_568880


namespace max_reflections_l568_568574

theorem max_reflections (A B D : Point) (n : ℕ) (angle_CDA : ℝ) (incident_angle : ℕ → ℝ)
  (h1 : angle_CDA = 12)
  (h2 : ∀ k : ℕ, k ≤ n → incident_angle k = k * angle_CDA)
  (h3 : incident_angle n = 90) :
  n = 7 := 
sorry

end max_reflections_l568_568574


namespace probability_divisible_by_3_and_5_l568_568042

theorem probability_divisible_by_3_and_5 (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) (ones_digit_five : N % 10 = 5) :
  let P := (∑ x in Finset.range 10, ∑ y in Finset.range 10, 
            if (x + y + 5) % 3 = 0 then 1 else 0) / 90 in
  P = 1 / 3 := 
  sorry

end probability_divisible_by_3_and_5_l568_568042


namespace value_of_x2_plus_9y2_l568_568381

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l568_568381


namespace find_larger_number_l568_568175

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 :=
sorry

end find_larger_number_l568_568175


namespace doubled_money_is_1_3_l568_568960

-- Define the amounts of money Alice and Bob have
def alice_money := (2 : ℚ) / 5
def bob_money := (1 : ℚ) / 4

-- Define the total money before doubling
def total_money_before_doubling := alice_money + bob_money

-- Define the total money after doubling
def total_money_after_doubling := 2 * total_money_before_doubling

-- State the proposition to prove
theorem doubled_money_is_1_3 : total_money_after_doubling = 1.3 := by
  -- The proof will be filled in here
  sorry

end doubled_money_is_1_3_l568_568960


namespace part_a_part_b_l568_568640

theorem part_a (x y : ℂ) : (3 * y + 5 * x * Complex.I = 15 - 7 * Complex.I) ↔ (x = -7/5 ∧ y = 5) := by
  sorry

theorem part_b (x y : ℝ) : (2 * x + 3 * y + (x - y) * Complex.I = 7 + 6 * Complex.I) ↔ (x = 5 ∧ y = -1) := by
  sorry

end part_a_part_b_l568_568640


namespace newspaper_subscription_probability_l568_568861

theorem newspaper_subscription_probability:
  let p_D := 0.6
  let p_Q := 0.3
  P(at_least_one := 1 - (1 - p_D) * (1 - p_Q)) :=
    P(at_least_one) = 1 - (1 - 0.6) * (1 - 0.3) :=
by
  sorry

end newspaper_subscription_probability_l568_568861


namespace shift_right_three_units_l568_568418

theorem shift_right_three_units (x : ℝ) : (λ x, -2 * x) (x - 3) = -2 * x + 6 :=
by
  sorry

end shift_right_three_units_l568_568418


namespace mean_temperature_is_88_75_l568_568858

def temperatures : List ℕ := [85, 84, 85, 88, 91, 93, 94, 90]

theorem mean_temperature_is_88_75 : (List.sum temperatures : ℚ) / temperatures.length = 88.75 := by
  sorry

end mean_temperature_is_88_75_l568_568858


namespace determineHairColors_l568_568516

structure Person where
  name : String
  hairColor : String

def Belokurov : Person := { name := "Belokurov", hairColor := "" }
def Chernov : Person := { name := "Chernov", hairColor := "" }
def Ryzhev : Person := { name := "Ryzhev", hairColor := "" }

-- Define the possible hair colors
def Blonde : String := "Blonde"
def Brunette : String := "Brunette"
def RedHaired : String := "Red-Haired"

-- Define the conditions based on the problem statement
axiom hairColorConditions :
  Belokurov.hairColor ≠ Blonde ∧
  Belokurov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ Blonde ∧
  ∀ p : Person, p.hairColor = Brunette → p.name ≠ "Belokurov"

-- Define the uniqueness condition that each person has a different hair color
axiom uniqueHairColors :
  Belokurov.hairColor ≠ Chernov.hairColor ∧
  Belokurov.hairColor ≠ Ryzhev.hairColor ∧
  Chernov.hairColor ≠ Ryzhev.hairColor

-- Define the proof problem
theorem determineHairColors :
  Belokurov.hairColor = RedHaired ∧
  Chernov.hairColor = Blonde ∧
  Ryzhev.hairColor = Brunette := by
  sorry

end determineHairColors_l568_568516


namespace total_cost_correct_l568_568473

def cost_barette : ℕ := 3
def cost_comb : ℕ := 1

def kristine_barrettes : ℕ := 1
def kristine_combs : ℕ := 1

def crystal_barrettes : ℕ := 3
def crystal_combs : ℕ := 1

def total_spent (cost_barette : ℕ) (cost_comb : ℕ) 
  (kristine_barrettes : ℕ) (kristine_combs : ℕ) 
  (crystal_barrettes : ℕ) (crystal_combs : ℕ) : ℕ :=
  (kristine_barrettes * cost_barette + kristine_combs * cost_comb) + 
  (crystal_barrettes * cost_barette + crystal_combs * cost_comb)

theorem total_cost_correct :
  total_spent cost_barette cost_comb kristine_barrettes kristine_combs crystal_barrettes crystal_combs = 14 :=
by
  sorry

end total_cost_correct_l568_568473


namespace carlson_total_land_l568_568243

def carlson_initial_land (initial_land: ℕ) : Prop :=
  initial_land = 300

def first_land_cost (cost: ℕ) : Prop :=
  cost = 8000

def first_land_rate (rate: ℕ) : Prop :=
  rate = 20

def second_land_cost (cost: ℕ) : Prop :=
  cost = 4000

def second_land_rate (rate: ℕ) : Prop :=
  rate = 25

theorem carlson_total_land (initial_land: ℕ) (first_land_cost: ℕ) (first_land_rate: ℕ) (second_land_cost: ℕ) (second_land_rate: ℕ) :
  carlson_initial_land initial_land → first_land_cost first_land_cost → first_land_rate first_land_rate → 
  second_land_cost second_land_cost → second_land_rate second_land_rate → 
  initial_land + (first_land_cost / first_land_rate) + (second_land_cost / second_land_rate) = 860 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  done

end carlson_total_land_l568_568243


namespace total_spend_on_four_games_l568_568519

noncomputable def calculate_total_spend (batman_price : ℝ) (superman_price : ℝ)
                                        (batman_discount : ℝ) (superman_discount : ℝ)
                                        (tax_rate : ℝ) (game1_price : ℝ) (game2_price : ℝ) : ℝ :=
  let batman_discounted_price := batman_price - batman_discount * batman_price
  let superman_discounted_price := superman_price - superman_discount * superman_price
  let batman_price_after_tax := batman_discounted_price + tax_rate * batman_discounted_price
  let superman_price_after_tax := superman_discounted_price + tax_rate * superman_discounted_price
  batman_price_after_tax + superman_price_after_tax + game1_price + game2_price

theorem total_spend_on_four_games :
  calculate_total_spend 13.60 5.06 0.10 0.05 0.08 7.25 12.50 = 38.16 :=
by sorry

end total_spend_on_four_games_l568_568519


namespace distinct_integer_sums_l568_568623

/-- Definition of a special fraction: a/b is special if a + b = 16 -/
def isSpecialFraction (a b : ℕ) : Prop :=
  a + b = 16

/-- Ensuring the positivity of integers a and b --/
def positiveInts (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0

/-- Definition of a set containing all special fractions -/
def specialFractions : set (ℚ) :=
  { f | ∃ a b : ℕ, f = a / b ∧ isSpecialFraction a b ∧ positiveInts a b }

/-- Definition of sum of two special fractions resulting in integer -/
def integerSum (x y : ℚ) : ℤ :=
  if (x + y) ∈ (set.range coe : set ℤ) then (x + y).to_int else 0

/-- The main theorem to prove: The number of distinct integers that can be written as 
    the sum of two special fractions is 10 -/
theorem distinct_integer_sums : 
  {n : ℤ | ∃ x y ∈ specialFractions, n = integerSum x y}.to_finset.card = 10 :=
sorry

end distinct_integer_sums_l568_568623


namespace expression_divisible_by_11_l568_568478

theorem expression_divisible_by_11 (n : ℕ) : (3 ^ (2 * n + 2) + 2 ^ (6 * n + 1)) % 11 = 0 :=
sorry

end expression_divisible_by_11_l568_568478


namespace solution_set_of_inequality_system_l568_568117

theorem solution_set_of_inequality_system (x : ℝ) :
  (frac (x-1) 2 + 2 > x) ∧ (2 * (x-2) ≤ 3 * x - 5) ↔ (1 ≤ x ∧ x < 3) :=
by sorry

end solution_set_of_inequality_system_l568_568117


namespace volleyball_team_starters_l568_568814

noncomputable def volleyball_team_count : ℕ := 14
noncomputable def triplets_count : ℕ := 3
noncomputable def starters_count : ℕ := 6

theorem volleyball_team_starters : 
  (choose (volleyball_team_count - triplets_count) starters_count) + 
  (triplets_count * choose (volleyball_team_count - triplets_count) (starters_count - 1)) = 1848 :=
by sorry

end volleyball_team_starters_l568_568814


namespace ratio_of_perimeters_l568_568591

theorem ratio_of_perimeters (s : ℝ) (h1 : s > 0) :
  let P_original := 4 * s
  let P_smallest := (3 / 2) * s
  P_smallest / P_original = (3 / 8) :=
by {
  let P_original := 4 * s,
  let P_smallest := (3 / 2) * s,
  sorry
}

end ratio_of_perimeters_l568_568591


namespace minimize_integral_l568_568297

noncomputable def f (a : ℝ) := ∫ x in a..a^2, (1 / x) * log ((x - 1) / 32)

theorem minimize_integral : 
  ∀ a : ℝ, a > 1 → f a = ∫ x in a..a^2, (1 / x) * log ((x - 1) / 32) → f 3 ≤ f a :=
begin
  sorry
end

end minimize_integral_l568_568297


namespace digitalEarth_correct_l568_568229

-- Define the possible descriptions of "Digital Earth"
inductive DigitalEarthDescription
| optionA : DigitalEarthDescription
| optionB : DigitalEarthDescription
| optionC : DigitalEarthDescription
| optionD : DigitalEarthDescription

-- Define the correct description according to the solution
def correctDescription : DigitalEarthDescription := DigitalEarthDescription.optionB

-- Define the theorem to prove the equivalence
theorem digitalEarth_correct :
  correctDescription = DigitalEarthDescription.optionB :=
sorry

end digitalEarth_correct_l568_568229


namespace solve_mt_eq_l568_568273

theorem solve_mt_eq (m n : ℤ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^2 + n) * (m + n^2) = (m - n)^3 →
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by
  sorry

end solve_mt_eq_l568_568273


namespace triangle_angles_correct_l568_568646

noncomputable def triangle_angles (a b c : ℝ) (C A B : ℝ) : Prop :=
  a = 3 ∧ b = 3 ∧ c = real.sqrt 8 - real.sqrt 3 ∧
  C = real.arccos ((7 / 18) + (2 * real.sqrt 6 / 9)) ∧
  A = (180 - C) / 2 ∧ B = (180 - C) / 2

-- Statement that the proof is about
theorem triangle_angles_correct :
  ∃ (C A B : ℝ), triangle_angles 3 3 (real.sqrt 8 - real.sqrt 3) C A B := sorry

end triangle_angles_correct_l568_568646


namespace total_ticket_cost_l568_568215

theorem total_ticket_cost (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : V = G - 276) 
  (price_vip : ℕ := 45) 
  (price_regular : ℕ := 20) : 
  (price_vip * V + price_regular * G = 6950) :=
by sorry

end total_ticket_cost_l568_568215


namespace ideal_number_l568_568345

open Real

noncomputable def f (x : ℝ) : ℝ := (9 * sin x * cos x) / ((1 + sin x) * (1 + cos x))

theorem ideal_number (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < π / 2) (h3 : f x0 ∈ ℕ) :
  x0 = 1 / 2 * arcsin (9 / 16) ∨ x0 = π / 2 - 1 / 2 * arcsin (9 / 16) :=
sorry

end ideal_number_l568_568345


namespace find_r_l568_568499

theorem find_r (a r : ℝ) (h : a > 0) (htangent : ∀ x y, x + y = r → (x - a)^2 + (y - a)^2 = r → False) :
  r = 2 * a + 1 + 2 * real.sqrt (4 * a + 1) :=
by sorry

end find_r_l568_568499


namespace distance_between_cars_after_third_checkpoint_l568_568142

theorem distance_between_cars_after_third_checkpoint
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (speed_after_first : ℝ)
  (speed_after_second : ℝ)
  (speed_after_third : ℝ)
  (distance_travelled : ℝ) :
  initial_distance = 100 →
  initial_speed = 60 →
  speed_after_first = 80 →
  speed_after_second = 100 →
  speed_after_third = 120 →
  distance_travelled = 200 :=
by
  sorry

end distance_between_cars_after_third_checkpoint_l568_568142


namespace arithmetic_mean_neg7_to_6_l568_568893

theorem arithmetic_mean_neg7_to_6 : 
  (list.range' (-7) 14).sum / 14 = -0.5 := 
by sorry

end arithmetic_mean_neg7_to_6_l568_568893


namespace volume_of_intersection_l568_568998

noncomputable section

open Real

def region1 (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1

def region2 (x y z : ℝ) : Prop :=
  abs x + abs y + abs (z - 1.5) ≤ 1

def intersection_volume_of_regions : ℝ :=
  0.1839

theorem volume_of_intersection : 
  (volume {(x, y, z) | region1 x y z ∧ region2 x y z}) = intersection_volume_of_regions :=
by
  sorry

end volume_of_intersection_l568_568998


namespace trash_cans_street_count_l568_568220

theorem trash_cans_street_count (S B : ℕ) (h1 : B = 2 * S) (h2 : S + B = 42) : S = 14 :=
by
  sorry

end trash_cans_street_count_l568_568220


namespace intersections_of_perpendiculars_form_square_l568_568570

-- Definitions
variables (Point : Type) [Geometry Point] (A B C D E F G H O : Point)
variables (line : Point → Point → set Point)

-- Conditions
def is_parallelogram (ABCD : Point × Point × Point × Point) : Prop := 
  let (A, B, C, D) := ABCD in 
  line A B ∥ line C D ∧ line B C ∥ line D A

def is_square (EFGH : Point × Point × Point × Point) : Prop :=
  let (E, F, G, H) := EFGH in
  dist E F = dist F G ∧ dist G H = dist H E ∧
  angle E F G = 90 ∧ angle F G H = 90

def perpendicular (P Q R : Point) : Prop :=
  angle P Q R = 90

def perpendiculars_from_parallelogram_to_square (A B C D E F G H : Point) : Prop :=
  let a := λ P, line P (closest_side P) -- Function to get perpendicular to closest side
  perpendicular A E (a A) ∧ perpendicular B F (a B) ∧
  perpendicular C G (a C) ∧ perpendicular D H (a D)

-- Theorem
theorem intersections_of_perpendiculars_form_square 
  (ABCD : Point × Point × Point × Point) (EFGH : Point × Point × Point × Point)
  (h_parallelogram : is_parallelogram ABCD)
  (h_square : is_square EFGH)
  (h_perpendiculars : perpendiculars_from_parallelogram_to_square A B C D E F G H) :
  ∃ P Q R S : Point, is_square (P, Q, R, S) := 
sorry

end intersections_of_perpendiculars_form_square_l568_568570


namespace find_angle_ONM_l568_568815

noncomputable def given_conditions (P Q R O : Point) (polygon : RegularPolygon) 
  (is_adjacent : adjacent_vertices polygon P Q)
  (is_adjacent2 : adjacent_vertices polygon Q R)
  (center : center polygon O) (M N : Point)
  (midpoint_M : midpoint_segment O (midpoint QR))
  (midpoint_N : midpoint_segment P Q) : Prop :=
  ∃ (angle_PQO : Angle), angle_PQO = 40 ∧
  ∃ (triangle_POQ : EquilateralTriangle), relate polygon triangle_POQ ∧
  ∃ (midpoint_M_angle : Angle), midpoint_M_angle = 30

theorem find_angle_ONM (P Q R O : Point) (polygon : RegularPolygon)
  (adj_pq : adjacent_vertices polygon P Q)
  (adj_qr : adjacent_vertices polygon Q R)
  (center_o : center polygon O) (M N : Point)
  (mid_M : midpoint_segment O (midpoint QR))
  (mid_N : midpoint_segment P Q) : 
  (angle ONM = 30) :=
  by sorry

end find_angle_ONM_l568_568815


namespace length_of_third_side_l568_568749

theorem length_of_third_side (a b : ℝ) (θ : ℝ) (h : a = 9) (h2 : b = 10) (h3 : θ = real.pi * 5 / 6) :
  ∃ c : ℝ, c = real.sqrt (a^2 + b^2 - 2 * a * b * real.cos θ) ∧ c = real.sqrt (181 + 90 * real.sqrt 3) :=
by {
  sorry
}

end length_of_third_side_l568_568749


namespace union_of_sets_l568_568687

open Set

variable (A B : Set ℝ)

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x ∧ x < 4}

theorem union_of_sets :
  A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} :=
sorry

end union_of_sets_l568_568687


namespace circle_passes_through_fixed_point_Q_l568_568060

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the point being on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define the distance from a point to the directrix x = -1
def distance_to_directrix (P : ℝ × ℝ) : ℝ :=
  P.1 + 1

-- The fixed point Q we want to prove the circle passes through
def fixed_point_Q : ℝ × ℝ :=
  (1, 0)

-- The main theorem stating the circle passes through the fixed point Q
theorem circle_passes_through_fixed_point_Q (P : ℝ × ℝ) (hP : point_on_parabola P) :
  let radius := distance_to_directrix P
  ∈ circle P.1 P.2 radius :=
  sorry

end circle_passes_through_fixed_point_Q_l568_568060


namespace parabola_proof_line_AB_proof_l568_568346

-- Define the parabola and given parameters
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

-- Define the parabola for specific p
def given_parabola (x y : ℝ) := parabola 4 x y

-- Define the midpoint condition
def midpoint (A B P : Point) := (A.x + B.x) / 2 = P.x ∧ (A.y + B.y) / 2 = P.y

-- Define the line equation
def line (k b x y : ℝ) := y = k * x + b

-- Define the specific line through P that intersects parabola at A and B
def line_AB (x y : ℝ) := line (-4) (-3) x y

-- Point data structure
structure Point where
  x : ℝ
  y : ℝ

def P : Point := Point.mk 1 (-1)

theorem parabola_proof : ∀ (x y : ℝ), parabola 4 x y → y^2 = 8 * x :=
by sorry

theorem line_AB_proof : ∀ (A B : Point), parabola 4 A.x A.y → parabola 4 B.x B.y → midpoint A B P → line_AB A.x A.y :=
by sorry

end parabola_proof_line_AB_proof_l568_568346


namespace dog_teeth_count_l568_568151

def cats_have_30_teeth : Nat := 30
def pigs_have_28_teeth : Nat := 28
def num_dogs : Nat := 5
def num_cats : Nat := 10
def num_pigs : Nat := 7
def total_teeth : Nat := 706

theorem dog_teeth_count :
  let D := 42 in
  5 * D + 10 * cats_have_30_teeth + 7 * pigs_have_28_teeth = total_teeth :=
by
  sorry

end dog_teeth_count_l568_568151


namespace find_b_value_l568_568673

noncomputable def z (b : ℝ) : ℂ := (2 + b * complex.i) / (1 - complex.i)

theorem find_b_value (b : ℝ) (h : (z b).re = -1) : b = 4 :=
  sorry

end find_b_value_l568_568673


namespace shifted_function_is_correct_l568_568421

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end shifted_function_is_correct_l568_568421


namespace binom_six_two_l568_568624

-- Define the binomial coefficient function
def binom (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- State the theorem
theorem binom_six_two : binom 6 2 = 15 := by
  sorry

end binom_six_two_l568_568624


namespace diana_total_cost_l568_568610

noncomputable def shopping_total_cost := 
  let t_shirt_price := 10
  let sweater_price := 25
  let jacket_price := 100
  let jeans_price := 40
  let shoes_price := 70 

  let t_shirt_discount := 0.20
  let sweater_discount := 0.10
  let jacket_discount := 0.15
  let jeans_discount := 0.05
  let shoes_discount := 0.25

  let clothes_tax := 0.06
  let shoes_tax := 0.09

  let t_shirt_qty := 8
  let sweater_qty := 5
  let jacket_qty := 3
  let jeans_qty := 6
  let shoes_qty := 4

  let t_shirt_total := t_shirt_qty * t_shirt_price 
  let sweater_total := sweater_qty * sweater_price 
  let jacket_total := jacket_qty * jacket_price 
  let jeans_total := jeans_qty * jeans_price 
  let shoes_total := shoes_qty * shoes_price 

  let t_shirt_discounted := t_shirt_total * (1 - t_shirt_discount)
  let sweater_discounted := sweater_total * (1 - sweater_discount)
  let jacket_discounted := jacket_total * (1 - jacket_discount)
  let jeans_discounted := jeans_total * (1 - jeans_discount)
  let shoes_discounted := shoes_total * (1 - shoes_discount)

  let t_shirt_final := t_shirt_discounted * (1 + clothes_tax)
  let sweater_final := sweater_discounted * (1 + clothes_tax)
  let jacket_final := jacket_discounted * (1 + clothes_tax)
  let jeans_final := jeans_discounted * (1 + clothes_tax)
  let shoes_final := shoes_discounted * (1 + shoes_tax)

  t_shirt_final + sweater_final + jacket_final + jeans_final + shoes_final

theorem diana_total_cost : shopping_total_cost = 927.97 :=
by sorry

end diana_total_cost_l568_568610


namespace tangent_line_eq_segment_length_45_degree_l568_568314

noncomputable def circle_center : ℝ × ℝ := (3, 4)
noncomputable def radius : ℝ := 2
noncomputable def point_A : ℝ × ℝ := (1, 0)

theorem tangent_line_eq (l : ℝ → ℝ) (tangent : ∀ x, (x - 3)^2 + (l x - 4)^2 = radius^2 → False) :
  (∀ x, l x = 4/3 * x - 1) ∨ (l = λ x, x - 1) := by sorry

theorem segment_length_45_degree (l : ℝ → ℝ) (slope_45 : ∀ x, l x = x - 1) :
  ∀ P Q, (P ≠ Q ∧ (P = (3 + √2, 4 + √2) ∨ P = (3 - √2, 4 - √2)) ∧ 
  (Q = (3 + √2, 4 + √2) ∨ Q = (3 - √2, 4 - √2)))
  → dist P Q = 2 * sqrt (radius^2 - (sqrt 2)^2) := by sorry

end tangent_line_eq_segment_length_45_degree_l568_568314


namespace wrong_observation_value_l568_568853

theorem wrong_observation_value (n : ℕ) (initial_mean corrected_mean correct_value wrong_value : ℚ) 
  (h₁ : n = 50)
  (h₂ : initial_mean = 36)
  (h₃ : corrected_mean = 36.5)
  (h₄ : correct_value = 60)
  (h₅ : n * corrected_mean = n * initial_mean - wrong_value + correct_value) :
  wrong_value = 35 := by
  have htotal₁ : n * initial_mean = 1800 := by sorry
  have htotal₂ : n * corrected_mean = 1825 := by sorry
  linarith

end wrong_observation_value_l568_568853


namespace range_of_x_l568_568342

-- Definition of the function
def f (a x : ℝ) : ℝ := log a (a^(2 * x) - 4 * a^x + 1)

-- Given the condition 0 < a < 1 and f(a, x) < 0
noncomputable def condition (a : ℝ) := 0 < a ∧ a < 1

-- Prove that the range of x for which f(a, x) < 0 is (-∞, 2 * log a 2)
theorem range_of_x (a : ℝ) (x : ℝ) (h : condition a) : f a x < 0 ↔ x < 2 * log a 2 :=
sorry

end range_of_x_l568_568342


namespace number_division_l568_568540

theorem number_division (x : ℝ) (h : 11 * x = 103.95) : x = 9.45 :=
begin
  sorry
end

end number_division_l568_568540


namespace boadecia_birth_l568_568236

noncomputable def year_when_boadicea_born : ℤ :=
  let C := -30 -- Cleopatra died in 30 B.C, hence her death year is -30 in integer representation
  let BC_Death_Year := C
  let Boadecia_Difference := 129
  let Combined_Age := 100
  let Boadecia_Birth_Year := -BC_Death_Year + Boadecia_Difference - Combined_Age
  (Boadecia_Birth_Year + 1)

theorem boadecia_birth : year_when_boadicea_born = 1 :=
by
  have BC_Death_Year := -30
  have Boadecia_Difference := 129
  have Combined_Age := 100
  let Boadecia_Birth_Year := -BC_Death_Year + Boadecia_Difference - Combined_Age
  show year_when_boadicea_born = 1, by sorry

end boadecia_birth_l568_568236


namespace find_a_of_parabola_l568_568099

theorem find_a_of_parabola 
  (a b c : Int)
  (h1 : vertex_of_parabola (a*x^2 + b*x + c) = (1, 5))
  (h2 : point_on_parabola (a*x^2 + b*x + c) (0, 2)) :
  a = -3 := 
by 
  sorry

end find_a_of_parabola_l568_568099


namespace find_f_of_2_l568_568317

def f : ℤ → ℤ
| x := if x < 0 then 2 * x - 3 else f (x - 1)

theorem find_f_of_2 : f 2 = -5 := by
  sorry

end find_f_of_2_l568_568317


namespace max_dot_product_OA_OP_l568_568458

noncomputable def max_dot_product 
  (a : ℝ) (h_pos : 0 < a) : ℝ :=
  (λ t, a^2 * (1 - t)) (0)

theorem max_dot_product_OA_OP : ∀ (a : ℝ) (h_pos : 0 < a),
  (∃ t, 0 ≤ t ∧ t ≤ 1 ∧ a^2 * (1 - t) = a^2) :=
by {
  intro a,
  intro h_pos,
  existsi 0,
  split,
  linarith,
  split,
  linarith,
  rw [mul_sub, sub_self, mul_zero, add_zero, mul_one],
  sorry
}

end max_dot_product_OA_OP_l568_568458


namespace total_pies_sold_l568_568576

-- Defining the conditions
def pies_per_day : ℕ := 8
def days_in_week : ℕ := 7

-- Proving the question
theorem total_pies_sold : pies_per_day * days_in_week = 56 :=
by
  sorry

end total_pies_sold_l568_568576


namespace angle_C_is_100_l568_568141

-- Define the initial measures in the equilateral triangle
def initial_angle (A B C : ℕ) (h_equilateral : A = B ∧ B = C ∧ C = 60) : ℕ := C

-- Definition to capture the increase in angle C
def increased_angle (C : ℕ) : ℕ := C + 40

-- Now, we need to state the theorem assuming the given conditions
theorem angle_C_is_100
  (A B C : ℕ)
  (h_equilateral : A = 60 ∧ B = 60 ∧ C = 60)
  (h_increase : C = 60 + 40)
  : C = 100 := 
sorry

end angle_C_is_100_l568_568141


namespace value_of_x2_plus_9y2_l568_568379

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l568_568379


namespace find_value_of_expression_l568_568339

theorem find_value_of_expression (x y : ℝ)
  (h1 : 5 * x + y = 19)
  (h2 : x + 3 * y = 1) :
  3 * x + 2 * y = 10 :=
sorry

end find_value_of_expression_l568_568339


namespace opposite_of_2021_l568_568110

theorem opposite_of_2021 : ∃ y : ℝ, 2021 + y = 0 ∧ y = -2021 :=
by
  sorry

end opposite_of_2021_l568_568110


namespace sin_double_angle_l568_568368

theorem sin_double_angle (θ : ℝ) (h : sin θ + cos θ = 1 / 5) : sin (2 * θ) = -24 / 25 :=
by
  sorry

end sin_double_angle_l568_568368


namespace discount_per_tshirt_l568_568839

/-
The Razorback t-shirt shop sells each t-shirt for $51. During the Arkansas and Texas Tech game, they sold 130 t-shirts at a discounted price and made $5590. We are to determine the discount per t-shirt.
-/

theorem discount_per_tshirt
    (full_price : ℕ) -- $51 per t-shirt
    (num_tshirts : ℕ) -- 130 t-shirts sold
    (total_revenue : ℕ) -- $5590 total revenue
    (expected_discount : ℕ) -- $8 expected discount per t-shirt
    (h_full_price : full_price = 51)
    (h_num_tshirts : num_tshirts = 130)
    (h_total_revenue : total_revenue = 5590)
    (h_expected_discount : expected_discount = 8):
  let total_full_price_revenue := num_tshirts * full_price in
  let total_discount := total_full_price_revenue - total_revenue in
  (total_discount / num_tshirts) = expected_discount :=
by {
  sorry
}

end discount_per_tshirt_l568_568839


namespace smallest_number_of_groups_l568_568223

theorem smallest_number_of_groups
  (participants : ℕ)
  (max_group_size : ℕ)
  (h1 : participants = 36)
  (h2 : max_group_size = 12) :
  participants / max_group_size = 3 :=
by
  sorry

end smallest_number_of_groups_l568_568223


namespace exists_function_f_l568_568066

theorem exists_function_f (f : ℕ → ℕ) : (∀ n : ℕ, f (f n) = n^2) → ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 :=
sorry

end exists_function_f_l568_568066


namespace union_of_sets_l568_568686

open Set

variable (A B : Set ℝ)

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x ∧ x < 4}

theorem union_of_sets :
  A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} :=
sorry

end union_of_sets_l568_568686


namespace sum_of_valid_three_digit_numbers_l568_568288

def is_valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d ≠ 5

def is_valid_three_digit_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let units := n % 10
  100 ≤ n ∧ n < 1000 ∧ is_valid_digit hundreds ∧ is_valid_digit tens ∧ is_valid_digit units

theorem sum_of_valid_three_digit_numbers : 
  (∑ n in finset.filter is_valid_three_digit_number (finset.range 1000), n) = 284160 :=
by
  sorry

end sum_of_valid_three_digit_numbers_l568_568288


namespace solve_inequality_l568_568487

theorem solve_inequality (x : ℝ) : -7/3 < x ∧ x < 7 → |x+2| + |x-2| < x + 7 :=
by
  intro h
  sorry

end solve_inequality_l568_568487


namespace quadratic_has_one_solution_l568_568641

theorem quadratic_has_one_solution (q : ℝ) (hq : q ≠ 0) : 
  qx^2 - 8x + 2 = 0 → q = 8 :=
by
  -- we can add the proof here or just end with sorry if focusing on statement
  sorry

end quadratic_has_one_solution_l568_568641


namespace hypotenuse_length_l568_568580

theorem hypotenuse_length (x y : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = 1350 * Real.pi) 
  (h2 : V2 = 2430 * Real.pi) 
  (h3 : (1/3) * Real.pi * y^2 * x = V1) 
  (h4 : (1/3) * Real.pi * x^2 * y = V2) 
  : Real.sqrt (x^2 + y^2) = Real.sqrt 954 :=
sorry

end hypotenuse_length_l568_568580


namespace sum_of_exponents_l568_568313

theorem sum_of_exponents (n : ℕ) (α : fin n → ℕ)
  (h : (∑ i, 2^(α i) = 1990) ∧ function.injective α) :
  (∑ i, α i) = 43 :=
begin
  sorry
end

end sum_of_exponents_l568_568313


namespace virginia_eggs_l568_568528

theorem virginia_eggs : 
    let V := 372
    let A := 15
    let J := 27
    let L := 63
in V - (A + J + L) = 267 := 
by
  sorry

end virginia_eggs_l568_568528


namespace square_of_harmonic_mean_l568_568848

theorem square_of_harmonic_mean :
  let a := 5
  let b := 10
  let c := 20
  let H := (3 / (1/a + 1/b + 1/c)) in
  H^2 = (3600 / 49) :=
by
  sorry

end square_of_harmonic_mean_l568_568848


namespace eighth_square_shaded_fraction_l568_568390

theorem eighth_square_shaded_fraction :
  ∀ (n : ℕ), (n > 0) → 
    let shaded_squares := n^2
    let total_squares := n^2
  in n = 8 → (shaded_squares / total_squares = 1) :=
by
  intro n hn
  let shaded_squares := n^2
  let total_squares := n^2
  intro hn8
  sorry

end eighth_square_shaded_fraction_l568_568390


namespace total_cost_with_discount_and_tax_l568_568536

theorem total_cost_with_discount_and_tax
  (sandwich_cost : ℝ := 2.44)
  (soda_cost : ℝ := 0.87)
  (num_sandwiches : ℕ := 2)
  (num_sodas : ℕ := 4)
  (discount : ℝ := 0.15)
  (tax_rate : ℝ := 0.09) : 
  (num_sandwiches * sandwich_cost * (1 - discount) + num_sodas * soda_cost) * (1 + tax_rate) = 8.32 :=
by
  sorry

end total_cost_with_discount_and_tax_l568_568536


namespace neg_sqrt_sq_eq_eleven_l568_568615

theorem neg_sqrt_sq_eq_eleven : (-real.sqrt 11) ^ 2 = 11 := by
  sorry

end neg_sqrt_sq_eq_eleven_l568_568615


namespace vasya_numbers_l568_568056

theorem vasya_numbers
  {n : ℕ} (a : fin n → ℝ) (ha : ∀ i, 0 < a i) :
  ∃ b : fin n → ℝ, 
    (∀ i, b i ≥ a i) ∧ 
    (∀ i j, b i / b j ∈ ℤ) ∧
    (∏ i, b i ≤ 2^((n-1)/2) * ∏ i, a i) :=
sorry

end vasya_numbers_l568_568056


namespace transformation_result_l568_568740

def f (x y : ℝ) : ℝ × ℝ := (y, x)
def g (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem transformation_result : g (f (-6) (7)).1 (f (-6) (7)).2 = (-7, 6) :=
by
  sorry

end transformation_result_l568_568740


namespace unique_triple_exists_l568_568360

theorem unique_triple_exists :
  ∃! (a b c : ℤ), 2 ≤ a ∧ 1 ≤ b ∧ 0 ≤ c ∧ (real.log b / real.log a = (c:ℝ)^3) ∧ a + b + c = 100 := 
sorry

end unique_triple_exists_l568_568360


namespace compute_fraction_power_mul_l568_568987

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end compute_fraction_power_mul_l568_568987


namespace compute_fraction_pow_mult_l568_568982

def frac_1_3 := (1 : ℝ) / (3 : ℝ)
def frac_1_5 := (1 : ℝ) / (5 : ℝ)
def target := (1 : ℝ) / (405 : ℝ)

theorem compute_fraction_pow_mult :
  (frac_1_3^4 * frac_1_5) = target :=
by
  sorry

end compute_fraction_pow_mult_l568_568982


namespace triangle_inequality_l568_568923

theorem triangle_inequality (A B C D : Type) 
(inside_triangle : D ∈ triangle A B C) :
  ∃ BC AD BD CD,
  let min_dist := min (AD, min (BD, CD)) in
  if angle A < 90 then 
    BC / min_dist ≥ 2 * real.sin (angle A)
  else
    BC / min_dist ≥ 2 :=
sorry

end triangle_inequality_l568_568923


namespace problem_1_problem_2_problem_3_l568_568041

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.log (x + 1) - (a*x / (x + 1))

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := (1 + k) ^ x - k * x - 1

theorem problem_1 {a : ℝ} : 
  (a ≤ 0 → ∀ x > -1, deriv (λ x, f x a) x > 0) ∧
  (a > 0 → ∀ x ∈ Ioo (-1) (a-1), deriv (λ x, f x a) x < 0 ∧ ∀ x ∈ Ioo (a-1) 1, deriv (λ x, f x a) x > 0) :=
sorry

theorem problem_2 {k : ℝ} (hk : k ∈ Ioi (-1)) : 
  ∀ x ∈ Icc (0 : ℝ) 1, g x k = 0 :=
sorry

theorem problem_3 (n : ℕ) (hn : 0 < n) : 
  ∑ k in finset.range n, (1 : ℝ) / (k + 2) < real.log (n+1) ∧ 
  real.log (n+1) < ∑ k in finset.range n, (1 : ℝ) / (k + 1) :=
sorry

end problem_1_problem_2_problem_3_l568_568041


namespace length_of_median_to_AC_l568_568005

-- Declare the parameters and main theorem statement
theorem length_of_median_to_AC
  (A B C : Type)
  [triangle A B C] 
  (is_right_triangle : is_right ∆ABC)
  (AC BC : ℝ)
  (BC_eq_a : BC = a)
  (perpendicular_medians : is_perpendicular (median_to AC) (median_to AB)) :
  median_length BF = sqrt(3/2) * a :=
sorry

end length_of_median_to_AC_l568_568005


namespace round_2_6575_to_2_66_l568_568824

-- Define the conditions.
def hundredth_place (x : ℝ) : ℝ := (⌊ x * 100 ⌋ % 10 : ℕ) / 10
def thousandth_place (x : ℝ) : ℝ := (⌊ x * 1000 ⌋ % 10 : ℕ) / 10

-- Define the rounding to the nearest hundredth function.
def round_nearest_hundredth (x : ℝ) : ℝ :=
  let hp := hundredth_place x
  let tp := thousandth_place x
  if tp >= 0.5 then (⌊ x * 100 ⌋ + 1) / 100
  else (⌊ x * 100 ⌋) / 100

-- Statement to be proven
theorem round_2_6575_to_2_66 : round_nearest_hundredth 2.6575 = 2.66 :=
by
  sorry

end round_2_6575_to_2_66_l568_568824


namespace probability_one_head_one_tail_l568_568167

def toss_outcomes : List (String × String) := [("head", "head"), ("head", "tail"), ("tail", "head"), ("tail", "tail")]

def favorable_outcomes (outcomes : List (String × String)) : List (String × String) :=
  outcomes.filter (fun x => (x = ("head", "tail")) ∨ (x = ("tail", "head")))

theorem probability_one_head_one_tail :
  (favorable_outcomes toss_outcomes).length / toss_outcomes.length = 1 / 2 :=
by
  -- Proof will be filled in here
  sorry

end probability_one_head_one_tail_l568_568167


namespace withheld_percentage_l568_568601

variables (hourly_wage : ℝ) (hours_worked : ℝ) (reduced_pay : ℝ)

def original_pay := hourly_wage * hours_worked

theorem withheld_percentage (h1 : hourly_wage = 50) (h2 : hours_worked = 10) (h3 : reduced_pay = 400) : 
  (original_pay hourly_wage hours_worked - reduced_pay) / original_pay hourly_wage hours_worked * 100 = 20 :=
by
  simp [original_pay, h1, h2, h3],
  sorry

end withheld_percentage_l568_568601


namespace max_value_f_min_value_range_g_l568_568709

open Real

-- Prove that the maximum value of f(x) = (ln x) / x for x > 0 is 1 / e
theorem max_value_f : ∃ x > 0, (∀ y > 0, (ln y / y ≤ ln x / x)) ∧ ln x / x = 1 / e :=
sorry

-- Prove that for a ∈ [0, 1 / e], the range of the minimum value of g(x) = x (ln x - (a x) / 2 - 1) over x ∈ (0, e] is [-e / 2, -1]
theorem min_value_range_g : ∀ a ∈ Icc 0 (1 / e), ∃ x ∈ Ioo 0 e, 
  (∀ y ∈ Ioo 0 e, x (ln x - a * x / 2 - 1) ≤ y (ln y - a * y / 2 - 1)) ∧ -e / 2 ≤ x (ln x - a * x / 2 - 1) ∧ x (ln x - a * x / 2 - 1) ≤ -1 :=
sorry

end max_value_f_min_value_range_g_l568_568709


namespace cos_C_value_triangle_perimeter_l568_568010

variables (A B C a b c : ℝ)
variables (cos_B : ℝ) (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3)
variables (dot_product_88 : a * b * (Real.cos C) = 88)

theorem cos_C_value (A B : ℝ) (a b : ℝ) (cos_B : ℝ) (cos_C : ℝ) (dot_product_88 : a * b * cos_C = 88) :
  A = 2 * B →
  cos_B = 2 / 3 →
  cos_C = 22 / 27 :=
sorry

theorem triangle_perimeter (A B C a b c : ℝ) (cos_B : ℝ)
  (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3) (dot_product_88 : a * b * (Real.cos C) = 88)
  (a_val : a = 12) (b_val : b = 9) (c_val : c = 7) :
  a + b + c = 28 :=
sorry

end cos_C_value_triangle_perimeter_l568_568010


namespace sum_of_like_terms_l568_568855

-- Define the given conditions
def m_condition : Prop := m + 2 = 4
def n_condition : Prop := 3n - 2 = 7

-- The main statement to prove
theorem sum_of_like_terms (m n : ℕ) (h_m : m_condition) (h_n : n_condition) : 
  2 * n * x^(m + 2) * y^7 + -4 * m * x^4 * y^(3 * n - 2) = -2 * x^4 * y^7 := 
  by
  sorry

end sum_of_like_terms_l568_568855


namespace find_points_l568_568812

theorem find_points :
  ∀ (x₀ : ℝ), (∃ (x₀ : ℝ), (M : ℝ×ℝ) → M = (x₀, -13/6) ∧ (∃ (k₁ k₂ : ℝ),
    k₁ + k₂ = 2 * x₀ ∧ k₁ * k₂ = -13/3 ∧
    (k₂ - k₁) / (1 + k₂ * k₁)) = sqrt 3) →
      (x₀ = 2 ∨ x₀ = -2) :=
by
  sorry

end find_points_l568_568812


namespace value_of_x2_plus_9y2_l568_568375

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l568_568375


namespace power_function_odd_l568_568697

theorem power_function_odd :
  ∃ (f : ℝ → ℝ), (∀ x, f x = x^(-1)) ∧ f (1/√3) = √3 ∧ (∀ x, f (-x) = -f x) := 
by
  use (λ x, x⁻¹)
  sorry

end power_function_odd_l568_568697


namespace each_person_share_l568_568917

theorem each_person_share (total_bill : ℝ) (tip_percentage : ℝ) (num_people : ℕ) (share : ℝ) : 
  total_bill = 211 → tip_percentage = 0.15 → num_people = 8 → share ≈ 30.33 :=
by
  intros h1 h2 h3
  sorry

end each_person_share_l568_568917


namespace sum_of_reciprocals_eq_one_l568_568121

theorem sum_of_reciprocals_eq_one {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y = (x * y) ^ 2) : (1/x) + (1/y) = 1 :=
sorry

end sum_of_reciprocals_eq_one_l568_568121


namespace correct_judgments_are_13_l568_568224

theorem correct_judgments_are_13
  (A_rounds : ℕ) (B_rounds : ℕ) (C_referee : ℕ) (total_rounds : ℕ)
  (consecutive_play : ∀ n m : ℕ, n ≠ m → (n ≤ A_rounds + B_rounds ∧ m ≤ B_rounds + C_referee)) :

  A_rounds = 10 →
  B_rounds = 17 →
  C_referee = 6 →
  total_rounds = 33 →
  (consecutive_play 0 1) →
  (∃ judgments : Set ℕ, judgments = {1, 3}) :=
by
  intros A_rounds_eq B_rounds_eq C_referee_eq total_rounds_eq consecutive_play_01
  sorry

end correct_judgments_are_13_l568_568224


namespace rudy_first_run_rate_l568_568825

def first_run_rate (R : ℝ) : Prop :=
  let time_first_run := 5 * R
  let time_second_run := 4 * 9.5
  let total_time := time_first_run + time_second_run
  total_time = 88

theorem rudy_first_run_rate : first_run_rate 10 :=
by
  unfold first_run_rate
  simp
  sorry

end rudy_first_run_rate_l568_568825


namespace competition_winner_is_C_l568_568299

-- Define the type for singers
inductive Singer
| A | B | C | D
deriving DecidableEq

-- Assume each singer makes a statement
def statement (s : Singer) : Prop :=
  match s with
  | Singer.A => Singer.B ≠ Singer.C
  | Singer.B => Singer.A ≠ Singer.C
  | Singer.C => true
  | Singer.D => Singer.B ≠ Singer.D

-- Define that two and only two statements are true
def exactly_two_statements_are_true : Prop :=
  (statement Singer.A ∧ statement Singer.C ∧ ¬statement Singer.B ∧ ¬statement Singer.D) ∨
  (statement Singer.A ∧ statement Singer.D ∧ ¬statement Singer.B ∧ ¬statement Singer.C)

-- Define the winner
def winner : Singer := Singer.C

-- The main theorem to be proved
theorem competition_winner_is_C :
  exactly_two_statements_are_true → (winner = Singer.C) :=
by
  intro h
  exact sorry

end competition_winner_is_C_l568_568299


namespace max_value_of_n_l568_568655

theorem max_value_of_n (A B : ℤ) (h1 : A * B = 48) : 
  ∃ n, (∀ n', (∃ A' B', (A' * B' = 48) ∧ (n' = 2 * B' + 3 * A')) → n' ≤ n) ∧ n = 99 :=
by
  sorry

end max_value_of_n_l568_568655


namespace find_real_number_a_l568_568354

theorem find_real_number_a (a : ℝ) (h : {1, a} ∪ {a ^ 2} = {1, a}) : a = -1 ∨ a = 0 :=
by {
  sorry -- proof to be done
}

end find_real_number_a_l568_568354


namespace percentage_cut_third_week_l568_568217

noncomputable def initial_weight : ℝ := 300
noncomputable def first_week_percentage : ℝ := 0.30
noncomputable def second_week_percentage : ℝ := 0.30
noncomputable def final_weight : ℝ := 124.95

theorem percentage_cut_third_week :
  let remaining_after_first_week := initial_weight * (1 - first_week_percentage)
  let remaining_after_second_week := remaining_after_first_week * (1 - second_week_percentage)
  let cut_weight_third_week := remaining_after_second_week - final_weight
  let percentage_cut_third_week := (cut_weight_third_week / remaining_after_second_week) * 100
  percentage_cut_third_week = 15 :=
by
  sorry

end percentage_cut_third_week_l568_568217


namespace qin_jiushao_algorithm_correct_operations_l568_568820

def qin_jiushao_algorithm_operations (f : ℝ → ℝ) (x : ℝ) : ℕ × ℕ := sorry

def f (x : ℝ) : ℝ := 4 * x^5 - x^2 + 2
def x : ℝ := 3

theorem qin_jiushao_algorithm_correct_operations :
  qin_jiushao_algorithm_operations f x = (5, 2) :=
sorry

end qin_jiushao_algorithm_correct_operations_l568_568820


namespace car_speed_is_48_l568_568926

theorem car_speed_is_48 {v : ℝ} : (3600 / v = 75) → v = 48 := 
by {
  sorry
}

end car_speed_is_48_l568_568926


namespace neither_music_nor_art_count_l568_568564

def total_students : Nat := 500
def music_students : Nat := 20
def art_students : Nat := 20
def both_students : Nat := 10
def neither_music_nor_art := total_students - (music_students + art_students - both_students)

theorem neither_music_nor_art_count :
  neither_music_nor_art = 470 :=
by
  -- Condition statements, ensuring correctness.
  have h1 : music_students = 20 := rfl
  have h2 : art_students = 20 := rfl
  have h3 : both_students = 10 := rfl
  have h4 : total_students = 500 := rfl

  -- Calculating number of students taking either music or art (or both):
  have students_taking_either := music_students + art_students - both_students
  have : students_taking_either = 30 := by simp [students_taking_either, h1, h2, h3]
  
  -- Calculating number of students taking neither music nor art:
  have students_taking_neither := total_students - students_taking_either
  have : neither_music_nor_art = students_taking_neither := by rfl
  have : students_taking_neither = 470 := by simp [students_taking_neither, h4, this]

  -- Concluding desired proof.
  exact this

end neither_music_nor_art_count_l568_568564


namespace value_of_x2_plus_9y2_l568_568382

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l568_568382


namespace value_of_square_sum_l568_568373

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l568_568373


namespace range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l568_568711

-- Define the propositions p and q
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := x^2 - 5 * x + 6 < 0

-- Question 1: When a = 1, if p ∧ q is true, determine the range of x
theorem range_of_x_when_a_is_1_and_p_and_q_are_true :
  ∀ x, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
by
  sorry

-- Question 2: If p is a necessary but not sufficient condition for q, determine the range of a
theorem range_of_a_when_p_necessary_for_q :
  ∀ a, (∀ x, q x → p x a) ∧ ¬ (∀ x, p x a → q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l568_568711


namespace part1_part2_l568_568436

-- Part (1)
theorem part1 (A B C : ℝ) (a b c : ℝ)
  (h1 : A = π / 3)
  (h2 : a = 3)
  (h3 : sin B + sin C = 2 * √3 * (sin B) * (sin C)) :
  (1 / b) + (1 / c) = 1 :=
sorry

-- Part (2)
theorem part2 (A B C D : ℝ) (a b c AD : ℝ)
  (h1 : A = π / 3)
  (h2 : a = 3)
  (h3 : b = √6)
  (h4 : BD * cos (π / 3 - B) = 0 { -- we assume the perpendicularity condition along with coordinates
  (h5 : AD = 2 * √6 + 3 * √2} :
  AD = 2 * √6 + 3 * √2 :=
sorry

end part1_part2_l568_568436


namespace number_of_three_digit_numbers_l568_568150

theorem number_of_three_digit_numbers (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5}) :
  let n := 3 in
  let choices_per_digit := S.card in
  let total_choices := choices_per_digit ^ n in
  total_choices = 125 :=
by
  sorry

end number_of_three_digit_numbers_l568_568150


namespace population_increase_rate_correct_l568_568860

variable (P0 P1 : ℕ)
variable (r : ℚ)

-- Given conditions
def initial_population := P0 = 200
def population_after_one_year := P1 = 220

-- Proof problem statement
theorem population_increase_rate_correct :
  initial_population P0 →
  population_after_one_year P1 →
  r = (P1 - P0 : ℚ) / P0 * 100 →
  r = 10 :=
by
  sorry

end population_increase_rate_correct_l568_568860


namespace magic_square_sum_l568_568428

theorem magic_square_sum (S : ℕ) (c : ℕ) (a b d e : ℕ) :
  let center := 18 in
  let sum_all := 325 in
  let sum_per_row := S in
  let sum_four_parts := 4 * sum_per_row in
  let total := 206 in
  a = 1 -> b = 25 -> d + e = 50 -> c = 10 -> 
  5 * sum_per_row = sum_all -> 
  (4 * sum_per_row) - 3 * center = total -> 
  sum_all - total = 119 :=
by 
  intro h1 h2 h3 h4 h5 h6
  have center_def : center = 18 := by rfl
  have sum_all_def : sum_all = 325 := by rfl
  have sum_per_row_def : 5 * sum_per_row = 325 := h5
  have sum_four_parts_def : sum_four_parts = 4 * sum_per_row := by rfl
  have total_def : sum_four_parts - 3 * center = total := h6
  have shaded_sum : sum_all - total = 119 := rfl
  sorry

end magic_square_sum_l568_568428


namespace actual_distance_mountains_approx_l568_568808

/-- Mathematical definitions based on given conditions --/
def map_distance_mountains_inch := 312
def map_distance_ram_inch := 42
def actual_distance_ram_km := 18.307692307692307

/-- The main theorem to prove the actual distance between the two mountains --/
theorem actual_distance_mountains_approx : 312 * (18.307692307692307 / 42) ≈ 136.0738178335298 :=
by
  sorry

end actual_distance_mountains_approx_l568_568808


namespace consecutive_even_legs_sum_l568_568851

theorem consecutive_even_legs_sum (x : ℕ) (h : x % 2 = 0) (hx : x ^ 2 + (x + 2) ^ 2 = 34 ^ 2) : x + (x + 2) = 48 := by
  sorry

end consecutive_even_legs_sum_l568_568851


namespace curve_cross_intersection_l568_568965

theorem curve_cross_intersection : 
  ∃ (t_a t_b : ℝ), t_a ≠ t_b ∧ 
  (3 * t_a^2 + 1 = 3 * t_b^2 + 1) ∧
  (t_a^3 - 6 * t_a^2 + 4 = t_b^3 - 6 * t_b^2 + 4) ∧
  (3 * t_a^2 + 1 = 109 ∧ t_a^3 - 6 * t_a^2 + 4 = -428) := by
  sorry

end curve_cross_intersection_l568_568965


namespace number_of_stanzas_is_correct_l568_568072

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Define the number of lines per stanza
def lines_per_stanza : ℕ := 10

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Calculate the number of words per stanza
def words_per_stanza : ℕ := lines_per_stanza * words_per_line

-- Define the number of stanzas
def stanzas (total_words words_per_stanza : ℕ) := total_words / words_per_stanza

-- Theorem: Prove that given the conditions, the number of stanzas is 20
theorem number_of_stanzas_is_correct : stanzas total_words words_per_stanza = 20 :=
by
  -- Insert the proof here
  sorry

end number_of_stanzas_is_correct_l568_568072


namespace area_of_triangle_PQE_l568_568116

-- Define the geometric problem
variable (P Q E : Type) -- Points P, Q, E
variable (AB BC : ℕ) -- sides of the rectangle

-- Given the conditions
axiom H1 : AB = 3
axiom H2 : BC = 2
axiom H3 : ∃ P ∈ (segment AB), ∀ PD tangent to the circle with diameter BC at E

-- Definition of tangent, segment, and similar triangles properties (as required)
-- Proof of areas and triangles properties would be in proof section

-- Statement to prove
theorem area_of_triangle_PQE (P Q E : Point) :
  area_triangle P Q E = 1/24 := sorry

end area_of_triangle_PQE_l568_568116


namespace product_is_zero_l568_568637

def product_series (a : ℤ) : ℤ :=
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * 
  (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

theorem product_is_zero : product_series 3 = 0 :=
by
  sorry

end product_is_zero_l568_568637


namespace intersection_point_of_planes_l568_568237

theorem intersection_point_of_planes :
  ∃ (x y z : ℚ), 
    3 * x - y + 4 * z = 2 ∧ 
    -3 * x + 4 * y - 3 * z = 4 ∧ 
    -x + y - z = 5 ∧ 
    x = -55 ∧ 
    y = -11 ∧ 
    z = 39 := 
by
  sorry

end intersection_point_of_planes_l568_568237


namespace total_expenditure_eq_fourteen_l568_568474

variable (cost_barrette cost_comb : ℕ)
variable (kristine_barrettes kristine_combs crystal_barrettes crystal_combs : ℕ)

theorem total_expenditure_eq_fourteen 
  (h_cost_barrette : cost_barrette = 3)
  (h_cost_comb : cost_comb = 1)
  (h_kristine_barrettes : kristine_barrettes = 1)
  (h_kristine_combs : kristine_combs = 1)
  (h_crystal_barrettes : crystal_barrettes = 3)
  (h_crystal_combs : crystal_combs = 1) :
  (kristine_barrettes * cost_barrette + kristine_combs * cost_comb) +
  (crystal_barrettes * cost_barrette + crystal_combs * cost_comb) = 14 := 
by 
  sorry

end total_expenditure_eq_fourteen_l568_568474


namespace log_equation_solutions_l568_568367

variables {b x : ℝ}

theorem log_equation_solutions (hb : b > 0) (hb_ne_one : b ≠ 1) (hx_ne_one : x ≠ 1)
  (h : (Real.log x / Real.log (b ^ 4) - Real.log b / Real.log (x ^ 4)) = 3) :
  x = b ^ 6 ∨ x = b ^ (-2) :=
sorry

end log_equation_solutions_l568_568367


namespace value_of_x2_plus_9y2_l568_568376

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l568_568376


namespace sum_of_digits_joey_age_l568_568439

def int.multiple (a b : ℕ) := ∃ k : ℕ, a = k * b

theorem sum_of_digits_joey_age (J C M n : ℕ) (h1 : J = C + 2) (h2 : M = 2) (h3 : ∃ k, C = k * M) (h4 : C = 12) (h5 : J + n = 26) : 
  (2 + 6 = 8) :=
by
  sorry

end sum_of_digits_joey_age_l568_568439


namespace sin_cos_identity_l568_568722

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end sin_cos_identity_l568_568722


namespace P_2_plus_P_neg2_l568_568783

variable {R : Type*} [Ring R]

-- Define the polynomial P(x)
structure Poly :=
  (a b c d : R)

def P (p : Poly) (x : R) : R := p.a * x^3 + p.b * x^2 + p.c * x + p.d

-- Given conditions
variables {k : R} {p : Poly}

-- The conditions stated in the problem
axiom h₀ : P p 0 = k
axiom h₁ : P p 1 = 3 * k
axiom h₂ : P p (-1) = 4 * k

-- The proof goal
theorem P_2_plus_P_neg2 : P p 2 + P p (-2) = 22 * k :=
  sorry

end P_2_plus_P_neg2_l568_568783


namespace scientific_notation_l568_568226

theorem scientific_notation (a : ℝ) (n : ℤ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 43050000 = a * 10^n) : a = 4.305 ∧ n = 7 :=
by
  sorry

end scientific_notation_l568_568226


namespace wheel_distance_l568_568956

theorem wheel_distance (d : ℝ) (rev : ℝ) (C : ℝ) (π : ℝ) (Distance : ℝ) :
  d = 10 ∧ rev = 19.108280254777068 ∧ π ≈ 3.14159 ∧ C = π * d ∧ Distance = C * rev → Distance ≈ 600 :=
by
  sorry

end wheel_distance_l568_568956


namespace tangent_intersection_locus_l568_568696

theorem tangent_intersection_locus :
  ∀ (l : ℝ → ℝ) (C : ℝ → ℝ), 
  (∀ x > 0, C x = x + 1/x) →
  (∃ k : ℝ, ∀ x, l x = k * x + 1) →
  ∃ (P : ℝ × ℝ), (P = (2, 2)) ∨ (P = (2, 5/2)) :=
by sorry

end tangent_intersection_locus_l568_568696


namespace conjugate_of_z_l568_568662

def z : ℂ := 1 + complex.I

theorem conjugate_of_z : complex.conj z = 1 - complex.I :=
  sorry

end conjugate_of_z_l568_568662


namespace anya_more_erasers_l568_568232

theorem anya_more_erasers (andrea_erasers : ℕ) (h1 : andrea_erasers = 6) (anya_multiplier : ℝ) (h2 : anya_multiplier = 4.5) : 
  let anya_erasers := (anya_multiplier * andrea_erasers) in
  anya_erasers - andrea_erasers = 21 :=
by 
  sorry

end anya_more_erasers_l568_568232


namespace problem1_problem2_l568_568665

variable (α : ℝ)

-- Condition from the problem
def condition : Prop := sin α = 2 * cos α

-- First proof problem
theorem problem1 (h : condition α) : 
  (2 * sin α - cos α) / (sin α + 2 * cos α) = 3 / 4 := 
sorry

-- Second proof problem
theorem problem2 (h : condition α) : 
  sin α ^ 2 + sin α * cos α - 2 * cos α ^ 2 = 4 / 5 := 
sorry

end problem1_problem2_l568_568665


namespace condition_for_odd_function_l568_568704

def f (x b : ℝ) := x + b * Real.cos x

theorem condition_for_odd_function (b : ℝ) : 
  (b = 0) ↔ (∀ x : ℝ, f (-x) b = -f x b) :=
by
  sorry

end condition_for_odd_function_l568_568704


namespace age_of_female_employee_when_hired_l568_568189

-- Defining the conditions
def hired_year : ℕ := 1989
def retirement_year : ℕ := 2008
def sum_age_employment : ℕ := 70

-- Given the conditions we found that years of employment (Y):
def years_of_employment : ℕ := retirement_year - hired_year -- 19

-- Defining the age when hired (A)
def age_when_hired : ℕ := sum_age_employment - years_of_employment -- 51

-- Now we need to prove
theorem age_of_female_employee_when_hired : age_when_hired = 51 :=
by
  -- Here should be the proof steps, but we use sorry for now
  sorry

end age_of_female_employee_when_hired_l568_568189


namespace price_of_each_bottle_is_3_l568_568887

/-- Each bottle of iced coffee has 6 servings. -/
def servings_per_bottle : ℕ := 6

/-- Tricia drinks half a container (bottle) a day. -/
def daily_consumption_rate : ℕ := servings_per_bottle / 2

/-- Number of days in 2 weeks. -/
def duration_days : ℕ := 14

/-- Number of servings Tricia consumes in 2 weeks. -/
def total_servings : ℕ := daily_consumption_rate * duration_days

/-- Number of bottles needed to get the total servings. -/
def bottles_needed : ℕ := total_servings / servings_per_bottle

/-- The total cost of the bottles is $21. -/
def total_cost : ℕ := 21

/-- The price per bottle is the total cost divided by the number of bottles. -/
def price_per_bottle : ℕ := total_cost / bottles_needed

/-- The price of each bottle is $3. -/
theorem price_of_each_bottle_is_3 : price_per_bottle = 3 :=
by
  -- We assume the necessary steps and mathematical verifications have been done.
  sorry

end price_of_each_bottle_is_3_l568_568887


namespace area_of_given_triangle_l568_568157

def point := (ℝ × ℝ)

def A : point := (2, 3)
def B : point := (7, 3)
def C : point := (4, 9)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs (fst A * (snd B - snd C) + fst B * (snd C - snd A) + fst C * (snd A - snd B))

theorem area_of_given_triangle :
  area_of_triangle A B C = 15 := 
sorry

end area_of_given_triangle_l568_568157


namespace radius_of_circumscribed_sphere_l568_568864

noncomputable def circumscribedSphereRadius (a : ℝ) (α := 60 * Real.pi / 180) : ℝ :=
  5 * a / (4 * Real.sqrt 3)

theorem radius_of_circumscribed_sphere (a : ℝ) :
  circumscribedSphereRadius a = 5 * a / (4 * Real.sqrt 3) := by
  sorry

end radius_of_circumscribed_sphere_l568_568864


namespace ellipse_represents_condition_l568_568338

theorem ellipse_represents_condition (m : ℝ) : 
  (-3 < m ∧ m < 1 ∨ 1 < m ∧ m < 5) ↔ 
  (5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3) := 
begin
  sorry
end

end ellipse_represents_condition_l568_568338


namespace digit_407_of_15_div_37_l568_568156

theorem digit_407_of_15_div_37 : 
    (let repeating_decimal : ℕ → ℕ := λ n, [4, 0, 5].nth ((n % 3) % 3).get_or_else 0
    in repeating_decimal 407 = 0) :=
begin
  -- Since this theorem statement includes the essential parts and definitions,
  -- placeholder for the proof has to be given.
  sorry
end

end digit_407_of_15_div_37_l568_568156


namespace circle_center_distance_l568_568004

theorem circle_center_distance :
  let C := {p: ℝ × ℝ | ∃ θ : ℝ, p = (2 * sin θ * cos θ, 2 * sin θ * sin θ)} in
  ∃ (center : ℝ × ℝ), center = (0, 1) ∧ dist (0, 1) (1, 0) = real.sqrt 2 := 
by 
  sorry

end circle_center_distance_l568_568004


namespace division_of_expressions_l568_568124

theorem division_of_expressions : 
  (2 * 3 + 4) / (2 + 3) = 2 :=
by
  sorry

end division_of_expressions_l568_568124


namespace sin_mul_cos_eq_neg_3_over_10_l568_568328

theorem sin_mul_cos_eq_neg_3_over_10 (θ : ℝ) (h1 : π / 2 < θ ∧ θ < π) (h2 : tan (θ + π / 4) = 1 / 2) : sin θ * cos θ = -3 / 10 :=
by
  sorry

end sin_mul_cos_eq_neg_3_over_10_l568_568328


namespace amounts_divided_correctly_l568_568484

noncomputable def A := 1428.57
noncomputable def B := 952.38
noncomputable def C := 1190.48
noncomputable def D := 714.29
noncomputable def E := 714.29

theorem amounts_divided_correctly:
  ∃ (a b c d e : ℝ),
    a / b = 3 / 2 ∧
    b / c = 4 / 5 ∧
    d = 0.6 * c ∧
    e = 0.6 * c ∧
    a + b + c + d + e = 5000 ∧
    a = 1428.57 ∧
    b = 952.38 ∧
    c = 1190.48 ∧
    d = 714.29 ∧
    e = 714.29 :=
by {
  use [A, B, C, D, E],
  split; {norm_num},
  split; {norm_num},
  split; {norm_num},
  split; {norm_num},
  split; {norm_num},
  }

-- sorry  -- Uncomment this line if necessary to ensure the Lean code can build successfully.

end amounts_divided_correctly_l568_568484


namespace sufficient_but_not_necessary_l568_568726

variables {a b : ℝ}

theorem sufficient_but_not_necessary (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end sufficient_but_not_necessary_l568_568726


namespace number_of_four_digit_integers_divisible_by_6_l568_568364

theorem number_of_four_digit_integers_divisible_by_6: 
  {x : ℕ // 1000 ≤ x ∧ x ≤ 9999 ∧ x % 6 = 0}.to_finset.card = 1350 :=
by
  sorry

end number_of_four_digit_integers_divisible_by_6_l568_568364


namespace solution_set_f1_geq_4_min_value_pq_l568_568708

-- Define the function f(x) for the first question
def f1 (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem for part (I)
theorem solution_set_f1_geq_4 (x : ℝ) : f1 x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 4 :=
by
  sorry

-- Define the function f(x) for the second question
def f2 (m x : ℝ) : ℝ := |x - m| + |x - 3|

-- Theorem for part (II)
theorem min_value_pq (p q m : ℝ) (h_pos_p : p > 0) (h_pos_q : q > 0)
    (h_eq : 1 / p + 1 / (2 * q) = m)
    (h_min_f : ∀ x : ℝ, f2 m x ≥ 3) :
    pq = 1 / 18 :=
by
  sorry

end solution_set_f1_geq_4_min_value_pq_l568_568708


namespace rectangle_width_percentage_change_l568_568104

theorem rectangle_width_percentage_change
  (L W : ℝ)
  (hL : 0 < L)
  (hW : 0 < W)
  (h_new_length : 1.2 * L)
  (h_new_area : 1.04 * (L * W))
  : ∃ x : ℝ, 1.2 * L * (W - (x / 100) * W) = 1.04 * (L * W) ∧ x = 40 / 3 :=
by {
  sorry
}

end rectangle_width_percentage_change_l568_568104


namespace profit_percentage_from_first_venture_l568_568596

theorem profit_percentage_from_first_venture
  (total_investment : ℝ)
  (investment_each : ℝ)
  (total_return_percentage : ℝ)
  (loss_percentage_second_venture : ℝ)
  (total_return : ℝ)
  (loss_second_venture : ℝ)
  (profit_first_venture : ℝ)
  (x : ℝ)
  (condition1 : total_investment = 25000)
  (condition2 : investment_each = 16250)
  (condition3 : total_return_percentage = 0.08)
  (condition4 : loss_percentage_second_venture = 0.05)
  (condition5 : total_return = total_return_percentage * total_investment)
  (condition6 : loss_second_venture = loss_percentage_second_venture * investment_each)
  (condition7 : profit_first_venture = total_return + loss_second_venture)
  (condition8 : x = (profit_first_venture * 100) / investment_each) :
  x ≈ 17.31 := sorry

end profit_percentage_from_first_venture_l568_568596


namespace length_of_third_side_l568_568748

theorem length_of_third_side (a b : ℝ) (θ : ℝ) (h : a = 9) (h2 : b = 10) (h3 : θ = real.pi * 5 / 6) :
  ∃ c : ℝ, c = real.sqrt (a^2 + b^2 - 2 * a * b * real.cos θ) ∧ c = real.sqrt (181 + 90 * real.sqrt 3) :=
by {
  sorry
}

end length_of_third_side_l568_568748


namespace inverse_graph_pass_point_l568_568460

variable {f : ℝ → ℝ}
variable {f_inv : ℝ → ℝ}

noncomputable def satisfies_inverse (f f_inv : ℝ → ℝ) : Prop :=
∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

theorem inverse_graph_pass_point
  (hf : satisfies_inverse f f_inv)
  (h_point : (1 : ℝ) - f 1 = 3) :
  f_inv (-2) + 3 = 4 :=
by
  sorry

end inverse_graph_pass_point_l568_568460


namespace designer_suit_size_l568_568993

theorem designer_suit_size : ∀ (waist_in_inches : ℕ) (comfort_in_inches : ℕ) 
  (inches_per_foot : ℕ) (cm_per_foot : ℝ), 
  waist_in_inches = 34 →
  comfort_in_inches = 2 →
  inches_per_foot = 12 →
  cm_per_foot = 30.48 →
  (((waist_in_inches + comfort_in_inches) / inches_per_foot : ℝ) * cm_per_foot) = 91.4 :=
by
  intros waist_in_inches comfort_in_inches inches_per_foot cm_per_foot
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_cast
  norm_num
  sorry

end designer_suit_size_l568_568993


namespace trajectory_through_centroid_l568_568179

variables (O A B C P : Type) [AddGroup O] [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup P]
variables (λ : ℝ)

-- Conditions
def condition1 (h₁ : O ≠ A) : Prop := true
def condition2 (h₂ : O ≠ B) : Prop := true
def condition3 (h₃ : O ≠ C) : Prop := true
def condition4 (h₄ : A ≠ B) : Prop := true
def condition5 (h₅ : A ≠ C) : Prop := true
def condition6 (h₆ : B ≠ C) : Prop := true
def condition7 (λ : ℝ) (h₇ : λ ∈ Set.Ici 0) : Prop := true
def condition8 : P = (A + λ * ((B - A) + (C - A))) := true

-- Statement to prove (the trajectory passes through the centroid G)
theorem trajectory_through_centroid
  (h₁ : condition1 O A)
  (h₂ : condition2 O B)
  (h₃ : condition3 O C)
  (h₄ : condition4 A B)
  (h₅ : condition5 A C)
  (h₆ : condition6 B C)
  (h₇ : condition7 λ)
  (h₈ : condition8 O A B C P λ) :
  ∃ G : O, P = (A + 2 * λ * ((B + C) / 2 - A)) :=
sorry

end trajectory_through_centroid_l568_568179


namespace trig_formula_identity_l568_568997

theorem trig_formula_identity :
  2 * real.sin (real.pi / 3.6) * (1 + real.sqrt 3 * real.tan (real.pi / 18)) = 2 := by
  sorry

end trig_formula_identity_l568_568997


namespace area_of_quadrilateral_is_16_l568_568892

-- Define the vertices of the quadrilateral
def vertex1 := (2, 1)
def vertex2 := (1, 6)
def vertex3 := (4, 5)
def vertex4 := (7, 2)

-- Define a method to compute the area using the Shoelace theorem
def shoelace_area (v1 v2 v3 v4 : (ℝ × ℝ)) : ℝ := 
  (1 / 2) * abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) - (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1))

-- The theorem to prove the area is 16
theorem area_of_quadrilateral_is_16 : shoelace_area vertex1 vertex2 vertex3 vertex4 = 16 := by
  sorry

end area_of_quadrilateral_is_16_l568_568892


namespace qin_jiushao_V1_value_l568_568147

noncomputable def calculate_V1 (x : ℕ) : ℕ :=
  let V0 := 3 in
  let V1 := V0 * x + 2 in
  V1

theorem qin_jiushao_V1_value : calculate_V1 10 = 32 := by
  sorry

end qin_jiushao_V1_value_l568_568147


namespace minimum_discount_l568_568563

theorem minimum_discount (cost_price marked_price : ℝ) (desired_margin : ℝ)
  (h_cost_price : cost_price = 800)
  (h_marked_price : marked_price = 1200)
  (h_desired_margin : desired_margin = 0.2) : 
  ∃ (minimum_discount : ℝ), minimum_discount = 0.8 :=
by
  use 0.8
  sorry

end minimum_discount_l568_568563


namespace min_sum_of_gcd_and_lcm_eq_three_times_sum_l568_568925

theorem min_sum_of_gcd_and_lcm_eq_three_times_sum (a b d : ℕ) (h1 : d = Nat.gcd a b)
  (h2 : Nat.gcd a b + Nat.lcm a b = 3 * (a + b)) :
  a + b = 12 :=
by
sorry

end min_sum_of_gcd_and_lcm_eq_three_times_sum_l568_568925


namespace find_a_l568_568869

theorem find_a (a : ℝ) :
  (∀ x : ℝ, deriv (fun x => a * x^3 - 2) x * x = 1) → a = 1 / 3 :=
by
  intro h
  have slope_at_minus_1 := h (-1)
  sorry -- here we stop as proof isn't needed

end find_a_l568_568869


namespace rowing_time_l568_568571

theorem rowing_time (rowing_speed : ℕ) (current_speed : ℕ) (distance : ℕ) 
  (h_rowing_speed : rowing_speed = 10)
  (h_current_speed : current_speed = 2)
  (h_distance : distance = 24) : 
  2 * distance / (rowing_speed + current_speed) + 2 * distance / (rowing_speed - current_speed) = 5 :=
by
  rw [h_rowing_speed, h_current_speed, h_distance]
  norm_num
  sorry

end rowing_time_l568_568571


namespace value_of_x2_plus_9y2_l568_568378

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l568_568378


namespace smallest_k_divisibility_l568_568659

theorem smallest_k_divisibility :
  ∃ k : ℕ, k = 40 ∧ (Polynomial.X ^ k - 1) % (Polynomial.X ^ 11 + Polynomial.X ^ 10 + Polynomial.X ^ 8 + Polynomial.X ^ 6 + Polynomial.X ^ 3 + Polynomial.X + 1) = 0 :=
by
  use 40
  sorry

end smallest_k_divisibility_l568_568659


namespace total_cost_correct_l568_568472

def cost_barette : ℕ := 3
def cost_comb : ℕ := 1

def kristine_barrettes : ℕ := 1
def kristine_combs : ℕ := 1

def crystal_barrettes : ℕ := 3
def crystal_combs : ℕ := 1

def total_spent (cost_barette : ℕ) (cost_comb : ℕ) 
  (kristine_barrettes : ℕ) (kristine_combs : ℕ) 
  (crystal_barrettes : ℕ) (crystal_combs : ℕ) : ℕ :=
  (kristine_barrettes * cost_barette + kristine_combs * cost_comb) + 
  (crystal_barrettes * cost_barette + crystal_combs * cost_comb)

theorem total_cost_correct :
  total_spent cost_barette cost_comb kristine_barrettes kristine_combs crystal_barrettes crystal_combs = 14 :=
by
  sorry

end total_cost_correct_l568_568472


namespace coin_flip_prob_nickel_halfdollar_heads_l568_568087

def coin_prob : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 2^3
  successful_outcomes / total_outcomes

theorem coin_flip_prob_nickel_halfdollar_heads :
  coin_prob = 1 / 4 :=
by
  sorry

end coin_flip_prob_nickel_halfdollar_heads_l568_568087


namespace BP_le_CP_l568_568067

theorem BP_le_CP
  (A B C P : Point)
  (b c AP BP CP : ℝ)
  (hP_inside : inside_triangle P A B C)
  (hb_eq_c : b = c)
  (h_angle : ∠APC ≤ ∠APB) :
  BP ≤ CP := 
sorry

end BP_le_CP_l568_568067


namespace other_candidate_valid_votes_l568_568753

theorem other_candidate_valid_votes (total_votes : ℕ) (invalid_percentage : ℕ) (first_candidate_percentage : ℕ)
  (valid_votes : ℕ) (other_candidate_votes : ℕ) :
  total_votes = 7500 →
  invalid_percentage = 20 →
  first_candidate_percentage = 55 →
  valid_votes = (0.80 * total_votes).toNat →
  other_candidate_votes = (0.45 * valid_votes).toNat →
  other_candidate_votes = 2700 :=
by
  intros h_total_votes h_invalid_percentage h_first_candidate_percentage h_valid_votes h_other_candidate_votes
  sorry

end other_candidate_valid_votes_l568_568753


namespace cylinder_volume_l568_568695

variables (a : ℝ) (π_ne_zero : π ≠ 0) (two_ne_zero : 2 ≠ 0) 

theorem cylinder_volume (h1 : ∃ (h r : ℝ), (2 * π * r = 2 * a ∧ h = a) 
                        ∨ (2 * π * r = a ∧ h = 2 * a)) :
  (∃ (V : ℝ), V = a^3 / π) ∨ (∃ (V : ℝ), V = a^3 / (2 * π)) :=
by
  sorry

end cylinder_volume_l568_568695


namespace scrap_cookie_radius_is_sqrt_21_l568_568088

noncomputable def radius_of_scrap_cookie (original_radius : ℝ) (large_cookie_radius : ℝ) (num_large_cookies : ℕ) 
(num_small_cookies : ℕ) (small_cookie_radius : ℝ) : ℝ :=
  let original_area := π * original_radius^2
  let large_cookie_area := π * large_cookie_radius^2
  let small_cookie_area := π * small_cookie_radius^2
  let total_cookies_area := (num_large_cookies * large_cookie_area) + (num_small_cookies * small_cookie_area)
  let scrap_area := original_area - total_cookies_area
  (scrap_area / π)^(1/2)

theorem scrap_cookie_radius_is_sqrt_21 :
  radius_of_scrap_cookie 5 1 3 4 0.5 = real.sqrt 21 :=
by
  sorry

end scrap_cookie_radius_is_sqrt_21_l568_568088


namespace geometric_seq_tenth_term_l568_568164

theorem geometric_seq_tenth_term :
  let a := 12
  let r := (1 / 2 : ℝ)
  (a * r^9) = (3 / 128 : ℝ) :=
by
  let a := 12
  let r := (1 / 2 : ℝ)
  show a * r^9 = 3 / 128
  sorry

end geometric_seq_tenth_term_l568_568164


namespace shifted_function_is_correct_l568_568423

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end shifted_function_is_correct_l568_568423


namespace min_cos_for_sqrt_l568_568277

theorem min_cos_for_sqrt (x : ℝ) (h : 2 * Real.cos x - 1 ≥ 0) : Real.cos x ≥ 1 / 2 := 
by
  sorry

end min_cos_for_sqrt_l568_568277


namespace echo_students_earnings_l568_568517

theorem echo_students_earnings (n : ℕ) (d : ℕ) (e : ℕ) (f : ℕ) (days_d : ℕ) (days_e : ℕ) (days_f : ℕ) (rate_multiplier : ℚ) 
  (total_payment : ℚ) (daily_wage : ℚ) :
  d = 8 → e = 6 → f = 7 → days_d = 4 → days_e = 6 → days_f = 7 → rate_multiplier = 1.5 → 
  total_payment = 1284 → daily_wage = total_payment / (n * rate_multiplier) → n = d * days_d + f * days_f +
  (e * days_e * rate_multiplier) → n ≠ 0 → 
  let echo_earnings := (e * days_e * daily_wage * rate_multiplier)
  in echo_earnings = 513.60 :=
by sorry

end echo_students_earnings_l568_568517


namespace debate_team_boys_l568_568863

/-
Given:
1. The debate team had 32 girls.
2. They were split into groups of 9.
3. There were 7 groups.
Prove:
- The number of boys on the debate team is 31.
-/
theorem debate_team_boys (girls : ℕ) (group_size : ℕ) (num_groups : ℕ) (total_students : ℕ) (boys : ℕ)
  (h1 : girls = 32)
  (h2 : group_size = 9)
  (h3 : num_groups = 7)
  (h4 : total_students = num_groups * group_size)
  (h5 : boys = total_students - girls) :
  boys = 31 :=
by {
  rw [h1, h2, h3, h4],
  rw h5,
  norm_num,
  sorry
}

end debate_team_boys_l568_568863


namespace series_sum_value_l568_568214

noncomputable def a_sequence : ℕ+ → ℕ
| 1 := 1
| (n + 1) := a_sequence n + n + 1

theorem series_sum_value :
  (∑ k in Finset.range (2017), (1 : ℚ) / (a_sequence k)) = 4032 / 2017 :=
sorry

end series_sum_value_l568_568214


namespace Sergey_teaches_History_in_Kaluga_l568_568515

structure Person :=
(name : String)
(city : String)
(subject : String)

axioms
  (Ivan Dmitry Sergey : Person)
  (Moscow SaintPetersburg Kaluga : String)
  (History Chemistry Biology : String)
  (Ivan_does_not_work_in_Moscow : Ivan.city ≠ Moscow)
  (Dmitry_does_not_work_in_SaintPetersburg : Dmitry.city ≠ SaintPetersburg)
  (Moscow_teacher_does_not_teach_History : ∀ p : Person, p.city = Moscow → p.subject ≠ History)
  (SaintPetersburg_teacher_teaches_Chemistry : ∀ p : Person, p.city = SaintPetersburg → p.subject = Chemistry)
  (Dmitry_teaches_Biology : Dmitry.subject = Biology)

theorem Sergey_teaches_History_in_Kaluga : Sergey.city = Kaluga ∧ Sergey.subject = History := by
  sorry

end Sergey_teaches_History_in_Kaluga_l568_568515


namespace ann_taxi_fare_l568_568227

theorem ann_taxi_fare :
  let d := 216 in
  let booking_fee := 15 in
  let fare_50_miles := 120 in
  let distance_50_miles := 50 in
  let distance_90_miles := 90 in
  let proportion := fare_50_miles / distance_50_miles in
  let fare_90_miles := proportion * distance_90_miles in
  d = fare_90_miles → 
  d + booking_fee = 231 :=
by
  intros d booking_fee fare_50_miles distance_50_miles distance_90_miles proportion fare_90_miles h
  have h1 : proportion = 120 / 50 := rfl
  have h2 : fare_90_miles = 216 := by rwa [← h, h1]
  have h3 : 216 + 15 = 231 := rfl
  exact h3

end ann_taxi_fare_l568_568227


namespace perimeter_of_parallelogram_in_triangle_l568_568394

theorem perimeter_of_parallelogram_in_triangle {P Q R S T U : Point} 
  (hPQ_PR : dist P Q = dist P R)
  (hPQ_PR_val : dist P Q = 17) 
  (hQR : dist Q R = 16)
  (hPQ_parallel : ∃ l : Line, Parallel l (Line.mk P R) ∧ contains l S ∧ contains l T)
  (hTU_parallel : ∃ m : Line, Parallel m (Line.mk P Q) ∧ contains m T ∧ contains m U) :
  dist P S + dist S T + dist T U + dist U P = 34 := sorry

end perimeter_of_parallelogram_in_triangle_l568_568394


namespace delta_equals_57_l568_568647

open Real

def sum_sin_range : ℤ → ℤ → Real
| a, b := ∑ i in Finset.range(b - a + 1), sin (a + i : ℝ)

def sum_cos_range : ℤ → ℤ → Real
| a, b := ∑ i in Finset.range(b - a + 1), cos (a + i : ℝ)

noncomputable def delta : Real :=
arccos ((sum_sin_range 2193 5793) ^ (sum_cos_range 2160 5760))

theorem delta_equals_57 :

  sum_sin_range 2193 5793 = sin 33 ∧
  sum_cos_range 2160 5760 = 1 →
  delta = 57 := by
  intro h
  cases h with hs hc
  unfold delta
  rw [hs, hc]
  simp
  sorry

end delta_equals_57_l568_568647


namespace shaded_area_approx_l568_568001

noncomputable def total_shaded_area (r_small r_medium r_large : ℝ) : ℝ :=
  let area_small := 3 * 6 - (1 / 2) * Real.pi * (r_small ^ 2)
  let area_medium := 6 * 12 - (1 / 2) * Real.pi * (r_medium ^ 2)
  let area_large := 9 * 18 - (1 / 2) * Real.pi * (r_large ^ 2)
  area_small + area_medium + area_large

theorem shaded_area_approx :
  total_shaded_area 3 6 9 ≈ 82.7 := sorry

end shaded_area_approx_l568_568001


namespace value_of_square_sum_l568_568372

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l568_568372


namespace arithmetic_sequence_term_l568_568329

noncomputable theory

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (a1 : ℝ) (d : ℝ) (n : ℕ)

-- Definitions based on the problem's conditions
def arithmetic_seq : Prop := ∀ n: ℕ, a(n) = a1 + (n - 1) * d

def sum_first_n_terms : Prop := ∀ n: ℕ, S(n) = n * a1 + (n * (n - 1) / 2) * d

-- Specific conditions from the problem
def specific_conditions : Prop :=
  d = 1 ∧ S(8) = 4 * S(4)

-- The theorem to prove
theorem arithmetic_sequence_term :
  arithmetic_seq a a1 1 → sum_first_n_terms a S a1 1 → specific_conditions a S →
  a 10 = 19 / 2 := 
sorry

end arithmetic_sequence_term_l568_568329


namespace find_common_difference_l568_568680

def common_difference (S_odd S_even n : ℕ) (d : ℤ) : Prop :=
  S_even - S_odd = n / 2 * d

theorem find_common_difference :
  ∃ d : ℤ, common_difference 132 112 20 d ∧ d = -2 :=
  sorry

end find_common_difference_l568_568680


namespace remainder_polynomial_division_l568_568318

def p (x : ℝ) : ℝ := sorry 

theorem remainder_polynomial_division :
  (p 1 = 5) →
  (p 3 = 7) →
  (p (-1) = 9) →
  ∃ (a b c : ℝ), 
    (∀ x, p x = ((-x^2 + 4x + 2) : ℝ) + ((x - 1) * (x + 1) * (x - 3)) * a) ∧ 
    (a = 0 ∧ b = 4 ∧ c = 2) :=
begin
  sorry
end

end remainder_polynomial_division_l568_568318


namespace positive_difference_sum_l568_568014

def sum_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def round_to_nearest_5 (n : ℕ) : ℕ :=
  (n + 2) / 5 * 5

def sum_round_nearest_5 (n : ℕ) : ℕ :=
  ∑ i in finset.range n, round_to_nearest_5 (i + 1)

theorem positive_difference_sum (n : ℕ) (hn : n = 60) :
  |sum_round_nearest_5 n - sum_n n| = 1560 :=
by
  sorry

end positive_difference_sum_l568_568014


namespace compare_shaded_areas_l568_568262

-- Definitions based on the problem conditions
def square_area (total_area : ℝ) : ℝ := total_area -- Total area for a single square

def shaded_area_square_I (total_area : ℝ) : ℝ :=
  4 * (total_area / 16) -- 4 out of 16 smaller squares are shaded

def shaded_area_square_II (total_area : ℝ) : ℝ :=
  16 * (total_area / 64) -- 16 out of 64 smaller squares are shaded

def shaded_area_square_III (total_area : ℝ) : ℝ :=
  8 * (total_area / 16) -- 8 out of 16 smaller squares are shaded

-- The theorem statement for comparing shaded areas
theorem compare_shaded_areas (total_area : ℝ) (h: total_area > 0) :
  shaded_area_square_I total_area = shaded_area_square_II total_area ∧
  shaded_area_square_I total_area ≠ shaded_area_square_III total_area :=
begin
  sorry -- Proof not required
end

end compare_shaded_areas_l568_568262


namespace green_disks_more_than_blue_l568_568396

theorem green_disks_more_than_blue 
  (total_disks : ℕ) (blue_ratio yellow_ratio green_ratio red_ratio : ℕ)
  (h1 : total_disks = 132)
  (h2 : blue_ratio = 3)
  (h3 : yellow_ratio = 7)
  (h4 : green_ratio = 8)
  (h5 : red_ratio = 4)
  : 6 * green_ratio - 6 * blue_ratio = 30 :=
by
  sorry

end green_disks_more_than_blue_l568_568396


namespace eq1_solution_eq2_solution_l568_568830

theorem eq1_solution (x : ℝ) : (x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2) ↔ (x^2 - 6 * x + 1 = 0) :=
by
  sorry

theorem eq2_solution (x : ℝ) : (x = 1 ∨ x = -5 / 2) ↔ (2 * x^2 + 3 * x - 5 = 0) :=
by
  sorry

end eq1_solution_eq2_solution_l568_568830


namespace find_mnk_l568_568435

noncomputable def triangleABC : Type :=
  { A B C : Point // dist A B = 130 ∧ dist A C = 130 ∧ dist B C = 78 }

def circleP : Type := { P : Point // radius P = 25 ∧ tangent P AC ∧ tangent P BC }

def circleQ : Type := 
  { Q : Point // ∃ r, radius Q = r ∧ r = 41 - 6 * Real.sqrt 15 ∧ 
    externally_tangent Q P ∧ tangent Q AB ∧ tangent Q BC ∧ 
    within_triangle Q A B C }

theorem find_mnk (ABC : triangleABC) (P : circleP) (Q : circleQ) : 
  ∃ (m n k : ℕ), k = 15 ∧ m = 41 ∧ n = 6 ∧ m + n * k = 131 :=
by
  sorry

end find_mnk_l568_568435


namespace average_player_footage_l568_568477

theorem average_player_footage :
  let point_guard := 130
  let shooting_guard := 145
  let small_forward := 85
  let power_forward := 60
  let center := 180
  let game_footage := 120
  let interviews := 90
  let opening_closing := 30
  let total_player_footage := point_guard + shooting_guard + small_forward + power_forward + center
  let number_of_players := 5
  let average_player_footage_in_seconds := total_player_footage / number_of_players
  let average_player_footage_in_minutes := average_player_footage_in_seconds / 60
  average_player_footage_in_minutes = 2 := by
  -- We specify the conditions
  have h1 : point_guard = 130 := rfl
  have h2 : shooting_guard = 145 := rfl
  have h3 : small_forward = 85 := rfl
  have h4 : power_forward = 60 := rfl
  have h5 : center = 180 := rfl
  have h6 : game_footage = 120 := rfl
  have h7 : interviews = 90 := rfl
  have h8 : opening_closing = 30 := rfl
  have h_total_player_footage : total_player_footage = 600 := by
    calc
      total_player_footage = 130 + 145 + 85 + 60 + 180 : by rw [h1, h2, h3, h4, h5]
                          ... = 600                   : by norm_num
  have h_average_player_footage_in_seconds : average_player_footage_in_seconds = 600 / 5 := by
    unfold average_player_footage_in_seconds number_of_players at *
    rw [h_total_player_footage]
  have h_average_player_footage_in_minutes : average_player_footage_in_minutes = (600 / 5) / 60 := by
    unfold average_player_footage_in_minutes at *
    rw [h_average_player_footage_in_seconds]
  show average_player_footage_in_minutes = 2 := by
    rw h_average_player_footage_in_minutes
    norm_num

end average_player_footage_l568_568477


namespace circle_projections_distances_l568_568315

noncomputable def r : ℕ := 5  -- r is given to be an odd number
def u : ℕ := 4  -- u = 2^2 = 4
def v : ℕ := 3  -- v = 3

def A : ℕ × ℕ := (r, 0)
def B : ℕ × ℕ := (-r, 0)
def C : ℕ × ℕ := (0, -r)
def D : ℕ × ℕ := (0, r)
def P : ℕ × ℕ := (u, v)
def M : ℕ × ℕ := (u, 0)
def N : ℕ × ℕ := (0, v)

theorem circle_projections_distances :
  abs (A.1 - M.1) = 1 ∧ abs (B.1 - M.1) = 9 ∧ abs (C.2 - N.2) = 8 ∧ abs (D.2 - N.2) = 2 :=
by
  sorry

end circle_projections_distances_l568_568315


namespace proof_a2_minus_b2_l568_568027

def a : ℝ := 3003 ^ 1502 - 3003 ^ (-1502)
def b : ℝ := 3003 ^ 1502 + 3003 ^ (-1502)

theorem proof_a2_minus_b2 : a^2 - b^2 = -4 := by
  sorry

end proof_a2_minus_b2_l568_568027


namespace rowing_speed_downstream_correct_l568_568935

/-- Given:
- The speed of the man upstream V_upstream is 20 kmph.
- The speed of the man in still water V_man is 40 kmph.
Prove:
- The speed of the man rowing downstream V_downstream is 60 kmph.
-/
def rowing_speed_downstream : Prop :=
  let V_upstream := 20
  let V_man := 40
  let V_s := V_man - V_upstream
  let V_downstream := V_man + V_s
  V_downstream = 60

theorem rowing_speed_downstream_correct : rowing_speed_downstream := by
  sorry

end rowing_speed_downstream_correct_l568_568935


namespace triangle_third_side_l568_568751

noncomputable def c := sqrt (181 + 90 * Real.sqrt 3)

theorem triangle_third_side {a b : ℝ} (A : ℝ) (ha : a = 9) (hb : b = 10) (hA : A = 150) :
  c = sqrt (9^2 + 10^2 - 2 * 9 * 10 * Real.cos (A * Real.pi / 180)) := by
  rw [Real.cos_of_real (150 * Real.pi / 180)]
  -- Expecting this cosine computation is correct per original problem solution
  sorry

end triangle_third_side_l568_568751


namespace sin_ratio_l568_568009

-- Define the problem parameters
variables {P Q R S : Type}
variables [AngleMeasure PQR Q R P S]
variables (α β θ : Real)
variables (k : ℝ) (h₁ : α = 45) (h₂ : β = 30)
variables (q : Real) (r : Real)

-- Given the conditions in the problem
noncomputable def AngleQ : ℝ := 45
noncomputable def AngleR : ℝ := 30
noncomputable def QR : ℝ := 5 * k
noncomputable def QS : ℝ := 2 * k
noncomputable def RS : ℝ := 3 * k

-- Main theorem to prove
theorem sin_ratio (h₃ : ∀ (α β), θ = 180 - α - β) : 
  (sin (3 * (sin 105)) / sin (2 * (sin 105))) = (3 / 2) :=
by
  sorry

end sin_ratio_l568_568009


namespace slower_train_passing_time_l568_568177

/--
Two goods trains, each 500 meters long, are running in opposite directions on parallel tracks. 
Their respective speeds are 45 kilometers per hour and 15 kilometers per hour. 
Prove that the time taken by the slower train to pass the driver of the faster train is 30 seconds.
-/
theorem slower_train_passing_time : 
  ∀ (distance length_speed : ℝ), 
    distance = 500 →
    ∃ (v1 v2 : ℝ), 
      v1 = 45 * (1000 / 3600) → 
      v2 = 15 * (1000 / 3600) →
      (distance / ((v1 + v2) * (3/50)) = 30) :=
by
  sorry

end slower_train_passing_time_l568_568177


namespace part_a_part_b_l568_568921

-- Define the structure and properties of the octahedron
variables {V : Type*}
variables [DecidableEq V] [Inhabited V] [Fintype V]

-- Conditional definitions for the problem
def is_congruent_quadrilateral_face (face : set (set V)) : Prop := sorry
def tetragonal_trapezohedron (o : V → set (set V)) : Prop := sorry
def edge_lengths (o : V → set (set V)) : set ℝ := sorry

-------------------
-- Part (a) Statement: Prove the set M of edge lengths has at most three distinct elements
theorem part_a (o : V → set (set V)) 
  (h_tetra : tetragonal_trapezohedron o) 
  (h_congruent : ∀ face, face ∈ (⋃ v, o v) → is_congruent_quadrilateral_face face) :
  (edge_lengths o).card ≤ 3 := 
sorry

-------------------
-- Part (b) Statement: Prove each quadrangle has two equal sides meeting at a common vertex
theorem part_b (o : V → set (set V)) 
  (h_tetra : tetragonal_trapezohedron o) 
  (h_congruent : ∀ face, face ∈ (⋃ v, o v) → is_congruent_quadrilateral_face face) :
  ∀ v ∈ V, ∃ u1 u2 u3 u4 ∈ V, u1 ≠ u2 ∧ u1 ≠ u3 ∧ u1 ≠ u4 ∧ is_congruent_quadrilateral_face {u1, u2, u3, u4} ∧
    (∃ x y, {x, y} ⊆ {u1, u2, u3, u4} ∧ x ≠ y ∧ ∀ face ∈ (⋃ v, o v), 
      ((x ∈ face ∧ y ∈ face) → dist x y = dist (u1 ∩ u2) (u3 ∩ u4))) :=
sorry

end part_a_part_b_l568_568921


namespace beetle_walks_less_percentage_l568_568963

-- Define the conditions
def distance_ant := 500 / 1000.0  -- km, since 1 km = 1000 meters
def time := 1.0  -- hour, as 60 minutes is 1 hour
def speed_beetle := 0.425  -- km/h

-- Define the proof problem
theorem beetle_walks_less_percentage (ant_speed beetle_speed : ℝ) (distance_ant : ℝ) (percentage_less : ℝ) :
  ant_speed = distance_ant / time → 
  beetle_speed = speed_beetle →
  percentage_less = ((ant_speed - beetle_speed) / ant_speed) * 100 →
  percentage_less = 15 :=
by
  -- Placeholder for proof
  sorry

end beetle_walks_less_percentage_l568_568963


namespace distance_from_center_to_plane_l568_568947

noncomputable def sphere_radius : ℝ := 8
noncomputable def triangle_a : ℝ := 17
noncomputable def triangle_b : ℝ := 17
noncomputable def triangle_c : ℝ := 26

theorem distance_from_center_to_plane (O : Point) (T : Triangle) 
  (h_sphere_radius : T.sphere_radius = sphere_radius)
  (h_triangle_sides : T.sides = (triangle_a, triangle_b, triangle_c))
  (h_tangent_sides : T.is_tangent_to_sphere O) : 
  T.distance_from_center_to_plane O = (Real.sqrt 2047) / 6 := 
sorry

end distance_from_center_to_plane_l568_568947
