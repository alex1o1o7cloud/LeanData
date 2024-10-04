import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Combinatorics.Combinations
import Mathlib.Combinatorics.Composition
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.EuclideanSpace.Basic
import Probability.CondCount
import Real

namespace formula1_correct_formula2_correct_formula3_correct_l126_126029

noncomputable def formula1 (n : ℕ) := (Real.sqrt 2 / 2) * (1 - (-1 : ℝ) ^ n)
noncomputable def formula2 (n : ℕ) := Real.sqrt (1 - (-1 : ℝ) ^ n)
noncomputable def formula3 (n : ℕ) := if (n % 2 = 1) then Real.sqrt 2 else 0

theorem formula1_correct (n : ℕ) : 
  (n % 2 = 1 → formula1 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula1 n = 0) := 
by
  sorry

theorem formula2_correct (n : ℕ) : 
  (n % 2 = 1 → formula2 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula2 n = 0) := 
by
  sorry
  
theorem formula3_correct (n : ℕ) : 
  (n % 2 = 1 → formula3 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula3 n = 0) := 
by
  sorry

end formula1_correct_formula2_correct_formula3_correct_l126_126029


namespace perpendicular_lines_planes_l126_126127

section
variables {a b : Type*} [inner_product_space ℝ a] [inner_product_space ℝ b]
variables {α β : set (vector_space ℝ a)} -- Planes α and β as sets of vectors

-- Conditions
variable (a_perp_alpha : ∀ v ∈ α, inner_product a v = 0)
variable (b_perp_beta : ∀ v ∈ β, inner_product b v = 0)

-- Problem statement
theorem perpendicular_lines_planes (a_perp_b : inner_product a b = 0) :
  (∀ v_α ∈ α, ∀ v_β ∈ β, inner_product v_α v_β = 0) ↔ a_perp_b :=
sorry
end

end perpendicular_lines_planes_l126_126127


namespace piecewise_function_evaluation_l126_126488

-- Define the piecewise function
def f (x : ℝ) : ℝ :=
  if x < 4 then 2^x else real.sqrt x

-- Define the theorem to prove f[f(2)] = 2
theorem piecewise_function_evaluation :
  f (f 2) = 2 :=
by {
  sorry -- Proof to be filled in
}

end piecewise_function_evaluation_l126_126488


namespace side_length_of_square_IJKL_l126_126815

theorem side_length_of_square_IJKL 
  (x y : ℝ) (hypotenuse : ℝ) 
  (h1 : x - y = 3) 
  (h2 : x + y = 9) 
  (h3 : hypotenuse = Real.sqrt (x^2 + y^2)) : 
  hypotenuse = 3 * Real.sqrt 5 :=
by
  sorry

end side_length_of_square_IJKL_l126_126815


namespace prism_volume_l126_126978

theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 5 :=
by
  sorry

end prism_volume_l126_126978


namespace max_min_values_l126_126450

theorem max_min_values (n : ℕ) (x : Fin n.succ → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i) 
  (h_norm : (Finset.univ : Finset (Fin n.succ)).sum (λ i, (x i)^2) = 1) :
  (n.succ * x 0 + (n : ℕ).sum (λ i, (n - i) * x i.succ.val)) *
  (x 0 + (n : ℕ).sum (λ i, (i + 1) * x i.succ.val)) ≤
  (↑n.succ * (↑n.succ + 1) / 2) ^ 2 ∧
  n.succ * x 0 ≤ ↑n.succ := sorry

end max_min_values_l126_126450


namespace arrangement_of_six_students_l126_126735

theorem arrangement_of_six_students :
  ∃ n : ℕ, n = 72 ∧
  (∀ boys girls pos,
    -- 1. Girls cannot be at the ends
    (¬ (girls.head = pos.head) ∧ ¬ (girls.last = pos.last)) ∧
    -- 2. Girls A and B are not next to girl C
    (∀ i, (girls[i] = "C" → girls[i + 1] ≠ "A" ∧ girls[i + 1] ≠ "B")) →
    (count_arrangements boys girls) = n) := sorry

def count_arrangements (boys girls : list string) : ℕ := sorry

noncomputable def pos : list ℕ := sorry

end arrangement_of_six_students_l126_126735


namespace anca_stopped_for_forty_minutes_l126_126739

/-
  Problem Statement:
  Anca and Bruce left Mathville at the same time, driving along a straight highway towards Staton.
  Bruce drove at 50 km/h. Anca drove at 60 km/h but stopped along the way to rest. 
  They both arrived at Staton at the same time.
  
  Prove that Anca stopped to rest for 40 minutes.
-/

def distance_to_staton : ℝ := 200 -- distance in km
def bruce_speed : ℝ := 50 -- speed in km/h
def anca_speed : ℝ := 60 -- speed in km/h
def anca_rest_time_in_hours : ℝ := 4 - (distance_to_staton / anca_speed)
def anca_rest_time_in_minutes : ℝ := anca_rest_time_in_hours * 60

theorem anca_stopped_for_forty_minutes :
  anca_rest_time_in_minutes = 40 :=
by
  -- Strategy: Calculate the rest time and show it equals 40 minutes
  sorry

end anca_stopped_for_forty_minutes_l126_126739


namespace arithmetic_sum_l126_126149

variable {a : ℕ → ℝ}

def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sum :
  is_arithmetic_seq a →
  a 5 + a 6 + a 7 = 15 →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  intros
  sorry

end arithmetic_sum_l126_126149


namespace distinct_values_of_exponentiation_l126_126297

theorem distinct_values_of_exponentiation : 
  let e := (3: ℕ)^(3: ℕ)^(3: ℕ)^(3: ℕ),
  let p1 := (3: ℕ) ^ ((3: ℕ) ^ ((3: ℕ) ^ (3: ℕ))),
  let p2 := (3: ℕ) ^ (((3: ℕ) ^ (3: ℕ)) ^ (3: ℕ)),
  let p3 := (((3: ℕ) ^ (3: ℕ)) ^ (3: ℕ)) ^ (3: ℕ),
  let p4 := ((3: ℕ) ^ ((3: ℕ) ^ (3: ℕ))) ^ (3: ℕ),
  let p5 := ((3: ℕ) ^ (3: ℕ)) ^ ((3: ℕ) ^ (3: ℕ)),
  let distinct_values := {p1, p2, p3, p4, p5}.to_finset.card
  in distinct_values = 3 :=
by
  sorry

end distinct_values_of_exponentiation_l126_126297


namespace best_sampling_method_l126_126704

/-- 
  Given a high school that wants to understand the psychological 
  pressure of students from three different grades, prove that 
  stratified sampling is the best method to use, assuming students
  from different grades may experience different levels of psychological
  pressure.
-/
theorem best_sampling_method
  (students_from_three_grades : Type)
  (survey_psychological_pressure : students_from_three_grades → ℝ)
  (potential_differences_by_grade : students_from_three_grades → ℝ → Prop):
  ∃ sampling_method, sampling_method = "stratified_sampling" :=
sorry

end best_sampling_method_l126_126704


namespace problem_I_problem_II_l126_126112

noncomputable def f (x : ℝ) (a : ℤ) : ℝ := 
  exp x - (1/2 : ℝ) * a * x^2

noncomputable def f' (x : ℝ) (a : ℤ) : ℝ :=
  exp x - a * x

theorem problem_I (x : ℝ) (h : x > 0) : 
  f x 2 > 1 := 
by 
  sorry 

theorem problem_II : 
  ∃ (a : ℤ), 0 < a ∧ (∀ (x : ℝ), x > 0 -> f' x a ≥ x^2 * log x) ∧ ∀ (b : ℤ), 0 < b -> b > a := 
by 
  use 2 
  sorry

end problem_I_problem_II_l126_126112


namespace find_g_plus_h_l126_126408

theorem find_g_plus_h (g h : ℚ) (d : ℚ) 
  (h_prod : (7 * d^2 - 4 * d + g) * (3 * d^2 + h * d - 9) = 21 * d^4 - 49 * d^3 - 44 * d^2 + 17 * d - 24) :
  g + h = -107 / 24 :=
sorry

end find_g_plus_h_l126_126408


namespace arrange_points_in_square_l126_126387

-- Define the square and the condition on the points
theorem arrange_points_in_square :
  ∃ (points : set (ℝ × ℝ))
    (N : ℕ),
    (N = 1965) ∧
    (∀ rect_corner_x rect_corner_y,
       0 ≤ rect_corner_x ∧ rect_corner_x ≤ 14 ∧
       0 ≤ rect_corner_y ∧ rect_corner_y ≤ 14 ∧
       (∀ x y, rect_corner_x ≤ x ∧ x < rect_corner_x + 1 ∧ rect_corner_y ≤ y ∧ y < rect_corner_y + 1 →
               (x, y) ∈ points) →
       (∃ p ∈ points, 
         rect_corner_x ≤ p.1 ∧ p.1 < rect_corner_x + 1 ∧ 
         rect_corner_y ≤ p.2 ∧ p.2 < rect_corner_y + 1)) :=
sorry

end arrange_points_in_square_l126_126387


namespace prob_dice_sum_15_l126_126306

theorem prob_dice_sum_15 (n : ℕ) :
  (∃ (f : Fin 5 → Fin 6), (∑ i, (f i).val + 1) = 15 ∧ n = 1001) ↔ 
  n = 1001 := by
  sorry

end prob_dice_sum_15_l126_126306


namespace perpendicular_lines_l126_126845

def line1_slope (a : ℝ) : ℝ := -a / 3
def line2_slope : ℝ := 3

theorem perpendicular_lines (a : ℝ) (h : line1_slope a * line2_slope = 1) : a = 1 :=
by {
  sorry
}

end perpendicular_lines_l126_126845


namespace total_blocks_fell_is_63_l126_126913

def heights : List ℕ := [9, 11, 13, 15, 17, 19]
def left_standing : List ℕ := [6, 5, 4, 3, 2, 1]

def blocks_fell_down (heights left_standing : List ℕ) : List ℕ :=
  List.map₂ (λ h s => h - s) heights left_standing

-- Sum of blocks fell down
def total_blocks_fell (heights left_standing : List ℕ) : ℕ :=
  (blocks_fell_down heights left_standing).sum

theorem total_blocks_fell_is_63 : total_blocks_fell heights left_standing = 63 := 
  by
    sorry

end total_blocks_fell_is_63_l126_126913


namespace sum_of_roots_eq_16_l126_126304

theorem sum_of_roots_eq_16 :
  let a := 1
  let b := -16
  let c := 15
  (z : ℂ) (hz : z^2 - 16 * z + 15 = 0) ∃ s1 s2, ((s1 + s2 = 16) ∧ (s1 * s2 = c) ∧ (s1 = z \/ s2 = z)) :=
by
  sorry

end sum_of_roots_eq_16_l126_126304


namespace simple_compound_interest_difference_l126_126305

def P : ℝ := 3600
def R : ℝ := 25
def T : ℝ := 2

def SI : ℝ := P * R * T / 100
def CI : ℝ := P * (1 + R / 100)^T - P

theorem simple_compound_interest_difference :
  CI - SI = 225 := 
by
  sorry

end simple_compound_interest_difference_l126_126305


namespace expedition_max_distance_l126_126195

/-- 
A, B, and C form an expedition team. Each member can carry enough water and food 
to survive in the desert for 36 days. The plan is to travel 30 kilometers into 
the desert each day. Each person can give some of their water and food to the 
others and then return alone. If member A cooperates with the other two, and it 
is required that all three can safely return, then the furthest A can go into 
the desert is 900 kilometers.
-/
theorem expedition_max_distance 
  (A B C : Type) 
  (foodA foodB foodC : A → B → C → ℕ := λ _, 36)
  (travel_distance_per_day : ℕ := 30)
  (cooperate : ∀ a b c, a + b + c < 70 → Prop) 
  (safe_return : ∀ a b c, a + b + c < 36 → Prop) : 
  ∃ (a_max_distance : ℕ), a_max_distance = 900 :=
sorry

end expedition_max_distance_l126_126195


namespace find_b_l126_126395

def h (x : ℝ) : ℝ := 4 * x - 5

theorem find_b (b : ℝ) (h_b : h b = 1) : b = 3 / 2 :=
by
  sorry

end find_b_l126_126395


namespace haley_shopping_time_l126_126506

variable (S T : ℚ)

-- Conditions as per problem statement
def setup_time : ℚ := 0.5
def snack_time : ℚ := 3 * setup_time
def watch_time : ℚ := 20 / 60
def total_time : ℚ := S + setup_time + snack_time + watch_time

-- The equation 8% of total time is the time she watched the comet
def eight_percent_eq_watch : Prop := watch_time = 0.08 * total_time

-- The main statement to prove:
theorem haley_shopping_time (eight_percent_eq_watch : eight_percent_eq_watch) : S = 1.8334 :=
by
  have T_eq : T = 25 / 6 := by
    sorry
  have total_eq : T = total_time := by
    sorry
  exact sorry

end haley_shopping_time_l126_126506


namespace derivative_at_0_l126_126151

def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem derivative_at_0 : deriv f 0 = 1 := sorry

end derivative_at_0_l126_126151


namespace sum_T_gt_five_ninths_l126_126970

noncomputable def a (n : ℕ) : ℕ := 4 * n - 2
noncomputable def b (n : ℕ) : ℕ := 2 * (1 / 4) ^ (n - 1)
noncomputable def c (n : ℕ) : ℕ := a n / b n
noncomputable def T (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), c i

theorem sum_T_gt_five_ninths (n : ℕ) (n_pos : 0 < n) : T n > 5/9 := sorry

end sum_T_gt_five_ninths_l126_126970


namespace num_possible_arrangements_l126_126176

def tea_picking : Fin 6 := 0
def cherry_picking : Fin 6 := 1
def strawberry_picking : Fin 6 := 2
def weeding : Fin 6 := 3
def tree_planting : Fin 6 := 4
def cow_milking : Fin 6 := 5

def activities := {tea_picking, cherry_picking, strawberry_picking, weeding, tree_planting, cow_milking}

theorem num_possible_arrangements : 
  let A62 := @Finset.choose 6 2 activities
  let A42 := @Finset.choose 4 2 activities
  let C61 := @Finset.choose 6 1 activities
  let C51 := @Finset.choose 5 1 activities
  let C41 := @Finset.choose 4 1 activities
  2 * A62.length * A42.length + C61.length * C51.length * C41.length * 2 + (A62.length * 2) = 630 :=
by
  sorry

end num_possible_arrangements_l126_126176


namespace probability_of_continuous_stripe_l126_126412

-- Define the basic setup for the cube and stripes
def cube_has_stripes (faces : Fin 6 → Fin 3) : Prop :=
  ∃ f : Fin 6 → Fin 3,
    (∀ i : Fin 6, faces i = f i) ∧
    continuous_stripe_encircling_cube f

-- Define a helper predicate for a continuous stripe encircling the cube
def continuous_stripe_encircling_cube (faces : Fin 6 → Fin 3) : Prop :=
  sorry -- Assume the actual condition is established here

-- Calculate the total number of combinations and favorable outcomes
def total_combinations : ℕ := 3 ^ 6
def favorable_outcomes : ℕ := 18

-- Calculate the expected probability
def expected_probability := favorable_outcomes.toRat / total_combinations.toRat

-- Prove the probability is 2 / 81
theorem probability_of_continuous_stripe :
  expected_probability = 2 / 81 :=
begin
  sorry
end

end probability_of_continuous_stripe_l126_126412


namespace knight_moves_equal_l126_126177

theorem knight_moves_equal (n : ℕ) (hn : n = 7) :
  let move_count (start finish : ℕ × ℕ) : ℕ := sorry in
  move_count (0, 0) (n-1, n-1) = move_count (0, 0) (n-1, 0) :=
by
  rw hn
  sorry

end knight_moves_equal_l126_126177


namespace exists_k_gt_2k_add_2020_l126_126083

def g (k : ℕ) : ℕ := -- Placeholder for g(k). Here, g(k) is a function taking a natural number and returning a natural number.
  sorry

theorem exists_k_gt_2k_add_2020 : ∃ k : ℕ, g(k) > 2 * k + 2020 :=
by
  sorry

end exists_k_gt_2k_add_2020_l126_126083


namespace minimal_diverse_set_size_l126_126227

theorem minimal_diverse_set_size (N : ℕ) (h : N ≥ 5) :
  ∃ M : ℕ, (∀ (flags : finset (vector (fin 2) N)), flags.card = M → 
    ∃ diverse_flags : finset (vector (fin 2) N), diverse_flags.card = N ∧ 
    ∀ i : fin N, ∃ color : fin 2, ∀ j : fin N, (diverse_flags.to_list[i][j] = color)) ∧
  M = 2^(N-2) + 1 :=
by
  sorry

end minimal_diverse_set_size_l126_126227


namespace solve_m_l126_126128

theorem solve_m (m : ℝ) : (m + 1) / 6 = m / 1 → m = 1 / 5 :=
by
  intro h
  sorry

end solve_m_l126_126128


namespace number_of_correct_statements_is_2_l126_126736

def statement1_correct : Prop :=
  ∀ (f : Flowchart), f.has_unique_start_and_end

def statement2_incorrect : Prop :=
  ¬(∀ (f : Flowchart), f.input_only_after_start_and_output_only_after_end)

def statement3_correct : Prop :=
  ∀ (f : Flowchart), f.decision_box_more_than_one_exit ∧ (∀ (b ≠ decision_box, ¬b.has_more_than_one_exit))

theorem number_of_correct_statements_is_2 :
  (∃ (s1 s3 : Prop), statement1_correct ∧ statement3_correct) ∧ (statement2_incorrect) →
  (number_of_correct_statements = 2) :=
by
  sorry

end number_of_correct_statements_is_2_l126_126736


namespace Kay_time_proof_l126_126597

-- Definitions corresponding to the conditions
def Lisa_time : ℝ := 8
def Combined_time : ℝ := 4.8

-- Definition of combined rate equation
def combined_rate_eq (Lisa_time Combined_time Kay_time : ℝ) : Prop :=
  (1 / Lisa_time) + (1 / Kay_time) = (1 / Combined_time)

-- Statement to prove
theorem Kay_time_proof : ∃ (Kay_time : ℝ), 
  combined_rate_eq Lisa_time Combined_time Kay_time ∧ Kay_time = 12 :=
by {
  use 12,
  unfold combined_rate_eq,
  simp,
  sorry
}

end Kay_time_proof_l126_126597


namespace intersects_l126_126847

-- Define the given conditions
def radius : ℝ := 5
def distance_to_line : ℝ := 3 * Real.sqrt 2

-- Define the relationship to prove
def line_intersects_circle : Prop :=
  radius > distance_to_line

-- Proof Statement
theorem intersects (r d : ℝ) (h_r : r = radius) (h_d : d = distance_to_line) : r > d :=
by {
  rw [h_r, h_d],
  exact Real.lt_of_lt_of_le (by norm_num) (by norm_num),
}

end intersects_l126_126847


namespace integral_evaluation_l126_126418

theorem integral_evaluation :
  (∫ x in 1..Real.exp 1, x + 1/x) = (Real.exp 2 + 1) / 2 :=
by
  sorry

end integral_evaluation_l126_126418


namespace perimeter_of_shaded_region_l126_126294

-- Define the given conditions
variables (A B C D E F G : Type) [is_square A B C D] [is_square B E F G]
variables (BC_side AB_side : ℝ) (CG : ℝ) (shaded_area : ℝ)

-- Given specific values
def given_conditions : Prop :=
  BC_side = CG ∨ CG = 9 ∧ shaded_area = 47

-- Define the theorem to be proven
theorem perimeter_of_shaded_region (h : given_conditions) :
  (4 * BC_side) = 32 := sorry

end perimeter_of_shaded_region_l126_126294


namespace gage_skating_time_l126_126445

theorem gage_skating_time : 
  (600 + x) / 9 = 100 -> x = 300 :=
by 
  intros h
  linarith [h, 9 * 100]

end gage_skating_time_l126_126445


namespace winning_candidate_votes_l126_126330

-- Define the conditions
variables (V : ℕ) -- Let V be the total number of votes.
variable (w : ℕ) -- Let w be the number of votes for the winning candidate.
variable (l : ℕ) -- Let l be the number of votes for the losing candidate.

-- State the conditions
axiom h1 : w = 0.70 * V
axiom h2 : l = 0.30 * V
axiom h3 : w - l = 280

-- State the question to prove
theorem winning_candidate_votes : w = 490 :=
by
  sorry

end winning_candidate_votes_l126_126330


namespace find_t_value_l126_126489

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 - 2*x - 1 else x^2 + 2*x - 1

theorem find_t_value :
  let t := -7/4 in
  ∀ x : ℝ, f(-x) = f(x) ∧
  (let y := t in y = f(x)) ∧
  (∃ A B C D, 
   A = ( -1 - real.sqrt(t + 2) ) ∧
   B = ( 1 - real.sqrt(t + 2) ) ∧
   C = ( 1 + real.sqrt(t + 2) ) ∧
   D = ( -1 + real.sqrt(t + 2) ) ∧
   B - A = C - B) :=
by intro t; intro x; sorry

end find_t_value_l126_126489


namespace connected_components_inequality_l126_126915

variable (V E : Type) [Fintype V] [Fintype E] [Inhabited E]
variable (G : SimpleGraph V) (A B : Set E)

def spanning_subgraph (G : SimpleGraph V) (edges : Set E) : SimpleGraph V :=
  G.subgraph (λ e, e ∈ edges)

variable [Fintype {G_A : SimpleGraph V // spanning_subgraph G A = G_A}]
variable [Fintype {G_B : SimpleGraph V // spanning_subgraph G B = G_B}]
variable [Fintype {G_A_union_B : SimpleGraph V // spanning_subgraph G (A ∪ B) = G_A_union_B}]
variable [Fintype {G_A_inter_B : SimpleGraph V // spanning_subgraph G (A ∩ B) = G_A_inter_B}]

def num_connected_components (G : SimpleGraph V) : Nat := 
  G.connectedComponents.card

theorem connected_components_inequality 
  (a b c d : Nat)
  (h1 : num_connected_components (spanning_subgraph G A) = a)
  (h2 : num_connected_components (spanning_subgraph G B) = b)
  (h3 : num_connected_components (spanning_subgraph G (A ∪ B)) = c)
  (h4 : num_connected_components (spanning_subgraph G (A ∩ B)) = d) :
  a + b ≤ c + d :=
sorry

end connected_components_inequality_l126_126915


namespace problem1_correct_problem2_correct_l126_126386

-- Definition for Problem 1
def problem1 (a b c d : ℚ) : ℚ :=
  (a - b + c) * d

-- Statement for Problem 1
theorem problem1_correct : problem1 (1/6) (5/7) (2/3) (-42) = -5 :=
by
  sorry

-- Definitions for Problem 2
def problem2 (a b c d : ℚ) : ℚ :=
  (-a^2 + b^2 * c - d^2 / |d|)

-- Statement for Problem 2
theorem problem2_correct : problem2 (-2) (-3) (-2/3) 4 = -14 :=
by
  sorry

end problem1_correct_problem2_correct_l126_126386


namespace find_a_exponential_l126_126830

theorem find_a_exponential (a : ℝ) (h1 : ∀ x ∈ set.Icc 0 1, 0 < a) 
  (h2 : (set.Icc 0 1).eq_on (λ x, a ^ x) (λ x, a ^ x)) : 
  (∀ x ∈ set.Icc 0 1, a ^ x) ∧ ((a > 1 ∧ max (a ^ 1) (a ^ 0) - min (a ^ 1) (a ^ 0) = 1 / 2) ∨ 
  (0 < a < 1 ∧ max (a ^ 0) (a ^ 1) - min (a ^ 0) (a ^ 1) = 1 / 2)) → 
  a = 1 / 2 ∨ a = 3 / 2 :=
by
  sorry

end find_a_exponential_l126_126830


namespace sin_alpha_plus_7pi_over_6_l126_126100

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) :
  (cos (α - π / 6) + sin α = (4 / 5) * sqrt 3) →
  sin (α + 7 * π / 6) = -4 / 5 :=
by
  sorry

end sin_alpha_plus_7pi_over_6_l126_126100


namespace cross_product_identity_l126_126865

variables (v w : Vector ℝ 3)

theorem cross_product_identity :
  (2 • v + 3 • w) ×ₗ (3 • v + 2 • w) = 
  (-5:ℝ) • (v ×ₗ w) :=
begin
  sorry
end

example (h : v ×ₗ w = ![1, -3, 2]) :
  (2 • v + 3 • w) ×ₗ (3 • v + 2 • w) = ![-5, 15, -10] :=
begin
  rw cross_product_identity v w,
  simp [h],
end

end cross_product_identity_l126_126865


namespace cot_sum_eq_l126_126431

theorem cot_sum_eq : 
  ∀ (a b c d : ℝ), 
    \cot(\cot⁻¹ a + \cot⁻¹ b + \cot⁻¹ c + \cot⁻¹ d) = \frac{8120}{1121} :=
by
  sorry

end cot_sum_eq_l126_126431


namespace tetrahedron_labeling_count_l126_126413

def is_valid_tetrahedron_labeling (labeling : Fin 4 → ℕ) : Prop :=
  let f1 := labeling 0 + labeling 1 + labeling 2
  let f2 := labeling 0 + labeling 1 + labeling 3
  let f3 := labeling 0 + labeling 2 + labeling 3
  let f4 := labeling 1 + labeling 2 + labeling 3
  labeling 0 + labeling 1 + labeling 2 + labeling 3 = 10 ∧ 
  f1 = f2 ∧ f2 = f3 ∧ f3 = f4

theorem tetrahedron_labeling_count : 
  ∃ (n : ℕ), n = 3 ∧ (∃ (labelings: Finset (Fin 4 → ℕ)), 
  ∀ labeling ∈ labelings, is_valid_tetrahedron_labeling labeling) :=
sorry

end tetrahedron_labeling_count_l126_126413


namespace set_of_points_equidistant_from_intersecting_planes_is_two_planes_l126_126429

-- Definitions of planes
variable (P1 P2 : ℝ → ℝ → ℝ → Prop)
-- Conditions of the planes being intersecting
variable (intersecting_planes : ∃ (line_of_intersection : ℝ → ℝ → Prop),
                               ∀ (x y z : ℝ), (P1 x y z ∧ P2 x y z) ↔ line_of_intersection x y)

-- We are looking for the set of points equidistant from P1 and P2
noncomputable def equidistant_points_from_planes (x y z : ℝ) : ℝ → ℝ → ℝ → Prop :=
  λ x y z, ∀ (d1 d2 : ℝ), (P1 x y z ∧ P2 x y z) → d1 = d2

-- Final goal: Set of such points forms two planes S1 and S2
theorem set_of_points_equidistant_from_intersecting_planes_is_two_planes :
  ∃ (S1 S2 : ℝ → ℝ → ℝ → Prop), (equidistant_points_from_planes P1 P2 = λ x y z, S1 x y z ∨ S2 x y z) :=
sorry

end set_of_points_equidistant_from_intersecting_planes_is_two_planes_l126_126429


namespace number_arrangement_impossible_l126_126744

theorem number_arrangement_impossible :
  ¬ ∃ (a b : Fin (3972)) (S T : Finset (Fin (3972))),
    S.card = 1986 ∧ T.card = 1986 ∧
    (∀ k : Nat, 1 ≤ k ∧ k ≤ 1986 →
      ∃ (ak bk : Fin (3972)), ak ∈ S ∧ bk ∈ T ∧ ak < bk ∧ bk.val - ak.val = k + 1) :=
sorry

end number_arrangement_impossible_l126_126744


namespace circular_field_area_l126_126975

noncomputable def area_of_circular_field (cost_per_meter total_cost : ℝ) : ℝ :=
  let C := total_cost / cost_per_meter in
  let r := C / (2 * Real.pi) in
  let A_square_meters := Real.pi * r^2 in
  A_square_meters / 10000

theorem circular_field_area 
  (cost_per_meter : ℝ)
  (total_cost : ℝ)
  (h_cost_per_meter : cost_per_meter = 4.80)
  (h_total_cost : total_cost = 6334.72526658735) :
  area_of_circular_field cost_per_meter total_cost = 13.8545 :=
by
  rw [h_cost_per_meter, h_total_cost]
  --   The intermediate steps will be filled in the proof.
  sorry

end circular_field_area_l126_126975


namespace exp_inequality_l126_126143

theorem exp_inequality
    (m n : ℝ) 
    (h1 : m > 0) 
    (h2 : n > 0) 
    (h3 : m > n) : 
    0.3^m < 0.3^n :=
sorry

end exp_inequality_l126_126143


namespace linear_function_behavior_l126_126566

theorem linear_function_behavior (x y : ℝ) (h : y = -3 * x + 6) :
  ∀ x1 x2 : ℝ, x1 < x2 → (y = -3 * x1 + 6) → (y = -3 * x2 + 6) → -3 * (x1 - x2) > 0 :=
by
  sorry

end linear_function_behavior_l126_126566


namespace shortest_altitude_l126_126656

theorem shortest_altitude (a b c : ℕ) (h1 : a = 12) (h2 : b = 16) (h3 : c = 20) (h4 : a^2 + b^2 = c^2) : ∃ x, x = 9.6 :=
by
  sorry

end shortest_altitude_l126_126656


namespace shoe_pair_probability_l126_126630

theorem shoe_pair_probability :
  let m := 7
  let n := 50
  (∀ (k : ℕ), k < 5 → 
    ¬ ∃ (pairs : Finset (Finset (Fin 10))), 
      pairs.card = k ∧ 
      ∀ (pair : Finset (Fin 10)), 
        pair ∈ pairs → 
        ∃ (adult_shoes : Finset (Fin 10)), 
          adult_shoes.card = k ∧ 
          adult_shoes ⊆ pair) → 
  m + n = 57 :=
by
  sorry

end shoe_pair_probability_l126_126630


namespace complex_exponentiation_l126_126047

-- Define the imaginary unit i where i^2 = -1.
def i : ℂ := Complex.I

-- Lean statement for proving the problem.
theorem complex_exponentiation :
  (1 + i)^6 = -8 * i :=
sorry

end complex_exponentiation_l126_126047


namespace large_rectangle_perimeter_l126_126351

-- Definitions from the conditions
def side_length_of_square (perimeter_square : ℕ) : ℕ := perimeter_square / 4
def width_of_small_rectangle (perimeter_rect : ℕ) (side_length : ℕ) : ℕ := (perimeter_rect / 2) - side_length

-- Given conditions
def perimeter_square := 24
def perimeter_rect := 16
def side_length := side_length_of_square perimeter_square
def rect_width := width_of_small_rectangle perimeter_rect side_length
def large_rectangle_height := side_length + rect_width
def large_rectangle_width := 3 * side_length

-- Perimeter calculation
def perimeter_large_rectangle (width height : ℕ) : ℕ := 2 * (width + height)

-- Proof problem statement
theorem large_rectangle_perimeter : 
  perimeter_large_rectangle large_rectangle_width large_rectangle_height = 52 :=
sorry

end large_rectangle_perimeter_l126_126351


namespace maximizing_probability_of_committee_win_l126_126160

theorem maximizing_probability_of_committee_win :
  ∃ k : ℕ, k = 6 ∧ (∀ l : ℕ, l ≤ 8 ∧ l ≠ k → (N l < N 6)) :=
begin
  sorry
end

def N (k : ℕ) : ℚ :=
  if k ≤ 8 then (factorial 8 * factorial 8 / ((factorial (8 - k)) * (factorial (8 - k)) * (factorial k)))
  else 0

# Here, the theorem statement captures:
# - the existence of k such that k = 6
# - and that for any l, where l is a natural number less than or equal to 8 and l is not k, the value N(l) should be less than N(6)


end maximizing_probability_of_committee_win_l126_126160


namespace factorization_correct_l126_126788

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l126_126788


namespace factorize_expression_l126_126787

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end factorize_expression_l126_126787


namespace num_sol_and_sum_sol_l126_126508

-- Definition of the main problem condition
def equation (x : ℝ) := (4 * x^2 - 9)^2 = 49

-- Proof problem statement
theorem num_sol_and_sum_sol :
  (∃ s : Finset ℝ, (∀ x, equation x ↔ x ∈ s) ∧ s.card = 4 ∧ s.sum id = 0) :=
sorry

end num_sol_and_sum_sol_l126_126508


namespace isosceles_triangle_area_is_sqrt_3_l126_126556

def isosceles_triangle_area (h : ℝ) (base_angle : ℝ) : ℝ :=
  let base := 2 * h in
  let sin_base_angle := Real.sin base_angle in
  let area := 0.5 * base * h * sin_base_angle in
  area

theorem isosceles_triangle_area_is_sqrt_3 (h : ℝ) (base_angle : ℝ) : 
  h = Real.sqrt 3 → base_angle = Real.pi / 3 → isosceles_triangle_area h base_angle = Real.sqrt 3 :=
by intros h_eq base_angle_eq; sorry

end isosceles_triangle_area_is_sqrt_3_l126_126556


namespace sum_inequality_l126_126237

theorem sum_inequality (n : ℕ) : (∑ k in finset.range (n + 1), (k / (n^2 + k))) < (1 / 2) + (1 / (6 * n)) :=
sorry

end sum_inequality_l126_126237


namespace product_of_four_consecutive_integers_is_perfect_square_l126_126650

theorem product_of_four_consecutive_integers_is_perfect_square :
  ∃ k : ℤ, ∃ n : ℤ, k = (n-1) * n * (n+1) * (n+2) ∧
    k = 0 ∧
    ((n = 0) ∨ (n = -1) ∨ (n = 1) ∨ (n = -2)) :=
by
  sorry

end product_of_four_consecutive_integers_is_perfect_square_l126_126650


namespace original_profit_percentage_l126_126718

-- Define the variables to represent the problem conditions
def CP := 60   -- Cost Price
def NCP := 48  -- New Cost Price
def NSP := 62.40  -- New Selling Price
def discount_percentage := 0.20
def new_profit_percentage := 0.30  -- 30%
def decrease_in_selling_price := 12.60

-- State the theorem to be proved
theorem original_profit_percentage :
  let SP := NSP + decrease_in_selling_price in
  let Profit := SP - CP in
  (Profit / CP) * 100 = 25 :=
by
  sorry

end original_profit_percentage_l126_126718


namespace part_I_part_II_l126_126826

def S (n : ℕ) : ℕ := (n^2 + n) / 2
def a (n : ℕ) : ℕ := n
def b (n : ℕ) : ℕ := 2^a n + (-1:ℤ)^n * n

theorem part_I (n : ℕ) : a n = n := sorry

theorem part_II (n : ℕ) : (∑ i in finset.range (2 * n), b i) = 2^(2 * n + 1) + n - 2 := sorry

end part_I_part_II_l126_126826


namespace M_subsetneq_N_l126_126121

def set_M : Set ℕ := { x | x^2 - 3 * x + 2 = 0 }
def set_N : Set ℕ := {0, 1, 2}

theorem M_subsetneq_N : set_M ⊂ set_N :=
by
  let M := { x : ℕ | x^2 - 3 * x + 2 = 0 }
  let N := {0, 1, 2}
  have H1 : M = {1, 2}, by sorry
  have H2 : N = {0, 1, 2}, by sorry
  rw [H1, H2]
  exact set.ssubset_insert (by simp)

end M_subsetneq_N_l126_126121


namespace quadratic_with_root_l126_126398

-- Problem statement
theorem quadratic_with_root (x : ℝ) :
  (∃ q : ℚ, x = q) → (∃ p : Polynomial ℚ, p.coeff 2 = 1 ∧ p.eval (Real.sqrt 2 - 3) = 0) :=
sorry

end quadratic_with_root_l126_126398


namespace range_of_m_l126_126152

noncomputable
def valid_range (m : ℝ) : Prop :=
  ∀ x : ℝ, mx^2 + 4 * m * x - 4 < 0

theorem range_of_m : ∀ m : ℝ, valid_range m ↔ -1 < m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l126_126152


namespace find_b6_l126_126918

variable (a_1 d : ℝ) (S_9 S_13 : ℝ)
variable (a_5 a_7 b_5 b_6 b_7: ℝ)

-- Conditions
def conditions :=
  S_9 = 9 * a_1 + (9 * 8 / 2) * d ∧
  S_13 = 13 * a_1 + (13 * 12 / 2) * d ∧
  S_9 = -36 ∧
  S_13 = -104 ∧
  b_5 = a_5 ∧
  b_7 = a_7 ∧
  b_5 = a_1 + 4 * d ∧
  b_7 = a_1 + 6 * d

-- Main statement
theorem find_b6 : conditions a_1 d S_9 S_13 a_5 a_7 b_5 b_6 b_7 →
  b_6 = 4 * Real.sqrt 2 ∨ b_6 = -4 * Real.sqrt 2 :=
sorry

end find_b6_l126_126918


namespace operator_is_division_l126_126873

theorem operator_is_division (a b : ℝ) (h₁ : a = 3 * real.sqrt 2) (h₂ : b = real.sqrt 2) (h₃ : a / b = 3) :
  ∃ square, (square = (/)) :=
by {
  use (/),
  exact h₃,
  sorry,
}

end operator_is_division_l126_126873


namespace segments_divide_ratio_3_to_1_l126_126300

-- Define points and segments
structure Point :=
  (x : ℝ) (y : ℝ)

structure Segment :=
  (A B : Point)

-- Define T-shaped figure consisting of 22 unit squares
noncomputable def T_shaped_figure : ℕ := 22

-- Define line p passing through point V
structure Line :=
  (p : Point → Point)
  (passes_through : Point)

-- Define equal areas condition
def equal_areas (white_area gray_area : ℝ) : Prop := 
  white_area = gray_area

-- Define the problem
theorem segments_divide_ratio_3_to_1
  (AB : Segment)
  (V : Point)
  (white_area gray_area : ℝ)
  (p : Line)
  (h1 : equal_areas white_area gray_area)
  (h2 : T_shaped_figure = 22)
  (h3 : p.passes_through = V) :
  ∃ (C : Point), (p.p AB.A = C) ∧ ((abs (AB.A.x - C.x)) / (abs (C.x - AB.B.x))) = 3 :=
sorry

end segments_divide_ratio_3_to_1_l126_126300


namespace square_side_length_l126_126312

theorem square_side_length (π : ℝ) (s : ℝ) :
  (∃ r : ℝ, 100 = π * r^2) ∧ (4 * s = 100) → s = 25 := by
  sorry

end square_side_length_l126_126312


namespace reflection_matrix_over_vector_l126_126799

/-- The matrix that reflects any vector over the vector ⟨4, 1⟩ is 
(⟨15/17, 8/17⟩, ⟨8/17, -15/17⟩).-/
theorem reflection_matrix_over_vector :
  ∃ (M : Matrix (fin 2) (fin 2) ℚ), 
  (∀ (v : Vector2), reflect v ⟨4, 1⟩ = M.mul_vec v) ∧
  M = ⟨⟨15 / 17, 8 / 17⟩, ⟨8 / 17, -15 / 17⟩⟩ :=
by
  sorry

end reflection_matrix_over_vector_l126_126799


namespace willy_has_more_crayons_l126_126691

theorem willy_has_more_crayons
(Willy_crayons : ℕ)
(Lucy_crayons : ℕ)
(h1 : Willy_crayons = 1400)
(h2 : Lucy_crayons = 290) :
Willy_crayons - Lucy_crayons = 1110 :=
by
  rw [h1, h2]
  exact Eq.refl 1110

end willy_has_more_crayons_l126_126691


namespace maple_taller_than_birch_l126_126577

def birch_tree_height : ℚ := 49 / 4
def maple_tree_height : ℚ := 102 / 5

theorem maple_taller_than_birch : maple_tree_height - birch_tree_height = 163 / 20 :=
by
  sorry

end maple_taller_than_birch_l126_126577


namespace isosceles_right_triangle_sums_l126_126102

theorem isosceles_right_triangle_sums (m n : ℝ)
  (h1: (1 * 2 + m * m + 2 * n) = 0)
  (h2: (1 + m^2 + 4) = (4 + m^2 + n^2)) :
  m + n = -1 :=
by {
  sorry
}

end isosceles_right_triangle_sums_l126_126102


namespace centroid_positions_correct_l126_126252

noncomputable def centroid_positions : ℕ := 
  let vertices : set (ℚ × ℚ) := {(0, 0), (15, 0), (15, 20), (0, 20)}
  in
  let points : set (ℚ × ℚ) := (set.univ : set (ℚ × ℚ)).filter (λ p, 
    (p.2 = 0 ∧ p.1 ∈ {0, 1, 2, ..., 15}) ∨ 
    (p.1 = 15 ∧ p.2 ∈ {0, 2, 4, ..., 20}) ∨ 
    (p.2 = 20 ∧ p.1 ∈ {0, 1, 2, ..., 15}) ∨ 
    (p.1 = 0 ∧ p.2 ∈ {0, 2, 4, ..., 20})
  )
  in 2596

theorem centroid_positions_correct : centroid_positions = 2596 := 
  sorry

end centroid_positions_correct_l126_126252


namespace price_of_jumbo_pumpkin_l126_126233

theorem price_of_jumbo_pumpkin (total_pumpkins : ℕ) (total_revenue : ℝ)
  (regular_pumpkins : ℕ) (price_regular : ℝ)
  (sold_jumbo_pumpkins : ℕ) (revenue_jumbo : ℝ): 
  total_pumpkins = 80 →
  total_revenue = 395.00 →
  regular_pumpkins = 65 →
  price_regular = 4.00 →
  sold_jumbo_pumpkins = total_pumpkins - regular_pumpkins →
  revenue_jumbo = total_revenue - (price_regular * regular_pumpkins) →
  revenue_jumbo / sold_jumbo_pumpkins = 9.00 :=
by
  intro h_total_pumpkins
  intro h_total_revenue
  intro h_regular_pumpkins
  intro h_price_regular
  intro h_sold_jumbo_pumpkins
  intro h_revenue_jumbo
  sorry

end price_of_jumbo_pumpkin_l126_126233


namespace compare_b_d_l126_126522

noncomputable def percentage_increase (x : ℝ) (p : ℝ) := x * (1 + p)
noncomputable def percentage_decrease (x : ℝ) (p : ℝ) := x * (1 - p)

theorem compare_b_d (a b c d : ℝ)
  (h1 : 0 < b)
  (h2 : a = percentage_increase b 0.02)
  (h3 : c = percentage_decrease a 0.01)
  (h4 : d = percentage_decrease c 0.01) :
  b > d :=
sorry

end compare_b_d_l126_126522


namespace gcd_b_n_b_n_plus_1_l126_126402

-- Definitions based on the conditions in the problem
def b_n (n : ℕ) : ℕ := 150 + n^3

theorem gcd_b_n_b_n_plus_1 (n : ℕ) : gcd (b_n n) (b_n (n + 1)) = 1 := by
  -- We acknowledge that we need to skip the proof steps
  sorry

end gcd_b_n_b_n_plus_1_l126_126402


namespace prob_complement_A_l126_126086

theorem prob_complement_A (P : Set (Set α) → ℝ) (A B : Set α)
  (hPB : P B = 0.3)
  (hPBA : P (B \ A) / P (A) = 0.9)
  (hPBNegA : P (B \ ¬A) / P (¬A) = 0.2) :
  P (¬A) = 6 / 7 := by
  sorry

end prob_complement_A_l126_126086


namespace quadratic_graph_behavior_l126_126455

-- Define the quadratic function and conditions
def quadratic (a x : ℝ) : ℝ := a * x^2 - 2 * x + 1 / 2
def a_pos (a : ℝ) : Prop := 0 < a

-- State the conclusions we need to prove
def conclusion_2 (a x : ℝ) : Prop := 
  ∀ (x y : ℝ), quadratic a x = y → (¬ (x < 0 ∧ y < 0))

def conclusion_3 (a x : ℝ) : Prop := 
  ∀ x, x < 0 → (quadratic a (x + 1) < quadratic a x)

-- Now state the main theorem
theorem quadratic_graph_behavior (a : ℝ) :
  a_pos a →
  (conclusion_2 a 0) ∧ (conclusion_3 a 0) :=
by
  sorry

end quadratic_graph_behavior_l126_126455


namespace triangle_area_l126_126571

theorem triangle_area {a c : ℝ} (h_a : a = 3 * Real.sqrt 3) (h_c : c = 2) (angle_B : ℝ) (h_B : angle_B = Real.pi / 3) : 
  (1 / 2) * a * c * Real.sin angle_B = 9 / 2 :=
by
  rw [h_a, h_c, h_B]
  sorry

end triangle_area_l126_126571


namespace find_n_from_binomial_expansion_l126_126811

theorem find_n_from_binomial_expansion (x a : ℝ) (n : ℕ)
  (h4 : (Nat.choose n 3) * x^(n - 3) * a^3 = 210)
  (h5 : (Nat.choose n 4) * x^(n - 4) * a^4 = 420)
  (h6 : (Nat.choose n 5) * x^(n - 5) * a^5 = 630) :
  n = 19 :=
sorry

end find_n_from_binomial_expansion_l126_126811


namespace expectation_hyperbolic_cosine_l126_126589

noncomputable def hyperbolic_cosine (x : ℝ) : ℝ := (Real.exp(x) + Real.exp(-x)) / 2

theorem expectation_hyperbolic_cosine (X : ℝ → ℝ) [IsRandomVariable X] (t : ℝ) :
  (∀ n : ℕ, 0 < n → E[X ^ (2 * n)] < ⊤) →
  E[hyperbolic_cosine (t * X)] = ∑ n in Finset.range(1) ∞, (t ^ (2 * n)) / Real.factorial (2 * n) * E[X ^ (2 * n)] := 
sorry

end expectation_hyperbolic_cosine_l126_126589


namespace samuel_faster_than_sarah_l126_126957

-- Definitions based on the conditions
def time_samuel : ℝ := 30
def time_sarah : ℝ := 1.3 * 60

-- The theorem to prove that Samuel finished his homework 48 minutes faster than Sarah
theorem samuel_faster_than_sarah : (time_sarah - time_samuel) = 48 := by
  sorry

end samuel_faster_than_sarah_l126_126957


namespace domain_g_l126_126400

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (-8 * x^2 + 14 * x - 3)

theorem domain_g :
  {x : ℝ | -8 * x^2 + 14 * x - 3 ≥ 0} = { x : ℝ | x ≤ 1 / 4 ∨ x ≥ 3 / 2 } :=
by
  sorry

end domain_g_l126_126400


namespace min_value_sqrt_expression_l126_126427

open Real

theorem min_value_sqrt_expression : ∃ x : ℝ, ∀ y : ℝ, 
  sqrt (y^2 + (2 - y)^2) + sqrt ((y - 1)^2 + (y + 2)^2) ≥ sqrt 17 :=
by
  sorry

end min_value_sqrt_expression_l126_126427


namespace sin_cos_identity_proof_l126_126805

noncomputable def solution : ℝ := Real.sin (Real.pi / 6) * Real.cos (Real.pi / 12) + Real.cos (Real.pi / 6) * Real.sin (Real.pi / 12)

theorem sin_cos_identity_proof : solution = Real.sqrt 2 / 2 := by
  sorry

end sin_cos_identity_proof_l126_126805


namespace angle_B_possible_values_l126_126158

theorem angle_B_possible_values
  (a b : ℝ) (A B : ℝ)
  (h_a : a = 2)
  (h_b : b = 2 * Real.sqrt 3)
  (h_A : A = Real.pi / 6) 
  (h_A_range : (0 : ℝ) < A ∧ A < Real.pi) :
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
  sorry

end angle_B_possible_values_l126_126158


namespace final_price_of_book_l126_126728

theorem final_price_of_book (original_price : ℝ) (d1_percentage : ℝ) (d2_percentage : ℝ) 
  (first_discount : ℝ) (second_discount : ℝ) (new_price1 : ℝ) (final_price : ℝ) :
  original_price = 15 ∧ d1_percentage = 0.20 ∧ d2_percentage = 0.25 ∧
  first_discount = d1_percentage * original_price ∧ new_price1 = original_price - first_discount ∧
  second_discount = d2_percentage * new_price1 ∧ 
  final_price = new_price1 - second_discount → final_price = 9 := 
by 
  sorry

end final_price_of_book_l126_126728


namespace part1_correct_part2_correct_l126_126705

-- Define the mathematical constructs and conditions
def original_purchase_price : ℝ := 30
def original_selling_price : ℝ := 40
def original_sales_volume : ℝ := 600
def selling_price_range : Set ℝ := Set.Icc 40 60
def sales_decrease_factor : ℝ := 10

-- Define the functions for profit and sales volume
def new_selling_price (x : ℝ) : ℝ := original_selling_price + x
def new_sales_volume (x : ℝ) : ℝ := original_sales_volume - sales_decrease_factor * x
def profit_per_lamp (x : ℝ) : ℝ := new_selling_price(x) - original_purchase_price
def total_profit (x : ℝ) : ℝ := profit_per_lamp(x) * new_sales_volume(x)

-- Conditions for part 1
def profit_condition (x : ℝ) : Prop := total_profit(x) = 10000
def valid_selling_price (x : ℝ) : Prop := new_selling_price(x) ∈ selling_price_range

-- Answer for part 1
def part1_answer : ℝ := 50

-- Statement for part 1
theorem part1_correct (x : ℝ) (hx : valid_selling_price x) : profit_condition x -> new_selling_price x = part1_answer := by
  sorry

-- Condition for part 2 (Maximize the profit within the given range with linear decrease in quantity)
def maximize_profit_condition (x : ℝ) : Prop := ∀ y : ℝ, valid_selling_price y -> total_profit(x) ≥ total_profit(y)

-- Answer for part 2
def part2_answer : ℝ := 60

-- Statement for part 2
theorem part2_correct (x : ℝ) (hx : valid_selling_price x) : maximize_profit_condition x -> new_selling_price x = part2_answer := by
  sorry

end part1_correct_part2_correct_l126_126705


namespace train_length_l126_126366

theorem train_length (time_crossing : ℝ) (speed_train : ℝ) (speed_man : ℝ) (rel_speed : ℝ) (length_train : ℝ) 
    (h1 : time_crossing = 39.99680025597952)
    (h2 : speed_train = 56)
    (h3 : speed_man = 2)
    (h4 : rel_speed = (speed_train - speed_man) * (1000 / 3600))
    (h5 : length_train = rel_speed * time_crossing):
 length_train = 599.9520038396928 :=
by 
  sorry

end train_length_l126_126366


namespace volume_of_cone_l126_126433

-- Define the conditions in the problem:
structure SpheresInCone (r : ℝ) where
  inside_cone : Prop
  four_on_base : Prop
  each_touches_two : Prop 
  fifth_touches_four : Prop 
  fifth_touches_sides : Prop

-- Hypothetical theorem stating the volume of the cone
theorem volume_of_cone (r : ℝ) (c : SpheresInCone r) :
  c.inside_cone →
  c.four_on_base →
  c.each_touches_two →
  c.fifth_touches_four →
  c.fifth_touches_sides →
  ∃ V, V = (1 / 3) * π * r^3 * (2 * real.sqrt 2 + 1)^3 := 
by
  intro h1 h2 h3 h4 h5
  use (1 / 3) * π * r^3 * (2 * real.sqrt 2 + 1)^3
  sorry

end volume_of_cone_l126_126433


namespace unique_solution_of_equation_l126_126133

theorem unique_solution_of_equation :
  ∃! x : ℝ, 3^(2 * x + 1) - 3^(x + 2) - 9 * 3^x + 27 = 0 :=
sorry

end unique_solution_of_equation_l126_126133


namespace exponential_value_at_3_exponential_inequality_solution_set_l126_126844

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x

theorem exponential_value_at_3 :
  f 3 = 1 / 8 := 
by
  -- placeholder for the proof
  sorry

theorem exponential_inequality_solution_set :
  { x : ℝ | f x + f (-x) < 5 / 2 } = Ioo (-1 : ℝ) 1 :=
by
  -- placeholder for the proof
  sorry

end exponential_value_at_3_exponential_inequality_solution_set_l126_126844


namespace length_of_arc_l126_126976

def radius : ℝ := 5
def area_of_sector : ℝ := 10
def expected_length_of_arc : ℝ := 4

theorem length_of_arc (r : ℝ) (A : ℝ) (l : ℝ) (h₁ : r = radius) (h₂ : A = area_of_sector) : l = expected_length_of_arc := by
  sorry

end length_of_arc_l126_126976


namespace prob_at_least_two_heads_is_eleven_sixteen_l126_126289

/-
  Definitions reflecting the conditions from the given problem.
  - We have a model of tossing a fair coin.
  - We are interested in the probability of getting at least two heads out of four tosses.
-/
def fair_coin_toss_outcomes : List (List Bool) := [
    [true, true, true, true], [true, true, true, false], [true, true, false, true], [true, false, true, true], [false, true, true, true], 
    [false, false, true, true], [false, true, false, true], [false, true, true, false], [true, false, false, true], [true, false, true, false], 
    [true, true, false, false], [true, false, false, false], [false, true, false, false], [false, false, true, false], [false, false, false, true], 
    [false, false, false, false]
]

def at_least_two_heads (outcome : List Bool) : Bool :=
  outcome.count (λ x => x = true) ≥ 2

def probability_at_least_two_heads (outcomes : List (List Bool)) : ℚ :=
  let success_events := outcomes.filter at_least_two_heads
  success_events.length / outcomes.length

/-
  The statement representing the proof problem:
  Prove that the probability of getting at least two heads
  in four tosses of a fair coin is 11/16.
-/
theorem prob_at_least_two_heads_is_eleven_sixteen : probability_at_least_two_heads fair_coin_toss_outcomes = 11 / 16 := by
  sorry

end prob_at_least_two_heads_is_eleven_sixteen_l126_126289


namespace range_f_minus_2x_l126_126761

noncomputable def f : ℤ → ℤ
| -4 := 0
| -3 := -1
| -2 := -2
| -1 := -3
| 0  := -2
| 1  := -1
| 2  := 0
| 3  := 1
| 4  := 2
| _  := 0  -- Assuming any other value not given is 0, for simplicity.

theorem range_f_minus_2x : 
  ∀ x : ℤ, -4 ≤ x ∧ x ≤ 4 → f(x) - 2 * x ∈ set.Icc (-7) 8 :=
by
  sorry

end range_f_minus_2x_l126_126761


namespace area_of_OBEC_is_125_l126_126353

noncomputable def area_of_quadrilateral_OBEC : ℝ :=
  let A := (20 / 3, 0) in
  let B := (0, 20) in
  let C := (10, 0) in
  let E := (5, 5) in
  125

theorem area_of_OBEC_is_125 :
  area_of_quadrilateral_OBEC = 125 :=
sorry

end area_of_OBEC_is_125_l126_126353


namespace fencing_required_l126_126016

theorem fencing_required {length width : ℝ} 
  (uncovered_side : length = 20)
  (field_area : length * width = 50) :
  2 * width + length = 25 :=
by
  sorry

end fencing_required_l126_126016


namespace solution_set_l126_126258

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x ∈ set.univ
axiom f_val_at_neg_one : f (-1) = 2
axiom f_derivative_strict_ineq : ∀ x : ℝ, (deriv f x) > 2

theorem solution_set {f : ℝ → ℝ} (h_domain : ∀ x : ℝ, x ∈ set.univ)
  (h_val_at_neg_one : f (-1) = 2)
  (h_derivative_strict_ineq : ∀ x : ℝ, (deriv f x) > 2) :
  {x : ℝ | f x > (2 * x + 4)} = set.Ioi (-1) :=
sorry

end solution_set_l126_126258


namespace probability_max_roll_correct_l126_126755
open Classical

noncomputable def probability_max_roll_fourth : ℚ :=
  let six_sided_max := 1 / 6
  let eight_sided_max := 3 / 4
  let ten_sided_max := 4 / 5

  let prob_A_given_B1 := (1 / 6) ^ 3
  let prob_A_given_B2 := (3 / 4) ^ 3
  let prob_A_given_B3 := (4 / 5) ^ 3

  let prob_B1 := 1 / 3
  let prob_B2 := 1 / 3
  let prob_B3 := 1 / 3

  let prob_A := prob_A_given_B1 * prob_B1 + prob_A_given_B2 * prob_B2 + prob_A_given_B3 * prob_B3

  -- Calculate probabilities with Bayes' Theorem
  let P_B1_A := (prob_A_given_B1 * prob_B1) / prob_A
  let P_B2_A := (prob_A_given_B2 * prob_B2) / prob_A
  let P_B3_A := (prob_A_given_B3 * prob_B3) / prob_A

  -- Probability of the fourth roll showing the maximum face value
  P_B1_A * six_sided_max + P_B2_A * eight_sided_max + P_B3_A * ten_sided_max

theorem probability_max_roll_correct : 
  ∃ (p q : ℕ), probability_max_roll_fourth = p / q ∧ Nat.gcd p q = 1 ∧ p + q = 4386 :=
by sorry

end probability_max_roll_correct_l126_126755


namespace total_savings_l126_126286

-- Definitions and Conditions
def thomas_monthly_savings : ℕ := 40
def joseph_saving_ratio : ℚ := 3 / 5
def saving_period_months : ℕ := 72

-- Problem Statement
theorem total_savings :
  let thomas_total := thomas_monthly_savings * saving_period_months
  let joseph_monthly_savings := thomas_monthly_savings * joseph_saving_ratio
  let joseph_total := joseph_monthly_savings * saving_period_months
  thomas_total + joseph_total = 4608 := 
by
  sorry

end total_savings_l126_126286


namespace solve_equation_find_real_part_l126_126965

theorem solve_equation_find_real_part (a b : ℝ) (z : ℂ) (h1 : z = a + b * complex.I)
  (h2 : z * (z + complex.I) * (z + 3 * complex.I) * (z - 2) = 180 * complex.I)
  (h3 : a > 0) (h4 : b > 0) : a = real.sqrt 180 :=
sorry

end solve_equation_find_real_part_l126_126965


namespace min_value_of_sum_l126_126503

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + 2 * b = 1) : 
  (∃ x, x = (3 / a + 2 / b) ∧ x = 25) :=
sorry

end min_value_of_sum_l126_126503


namespace leftover_value_correct_l126_126361

noncomputable def leftover_value (nickels_per_roll pennies_per_roll : ℕ) (sarah_nickels sarah_pennies tom_nickels tom_pennies : ℕ) : ℚ :=
  let total_nickels := sarah_nickels + tom_nickels
  let total_pennies := sarah_pennies + tom_pennies
  let leftover_nickels := total_nickels % nickels_per_roll
  let leftover_pennies := total_pennies % pennies_per_roll
  (leftover_nickels * 5 + leftover_pennies) / 100

theorem leftover_value_correct :
  leftover_value 40 50 132 245 98 203 = 1.98 := 
by
  sorry

end leftover_value_correct_l126_126361


namespace area_of_rectangle_EFGH_l126_126643

-- Definitions based on the given conditions
def smallest_square : ℝ := 1  -- One of the squares has an area of 1 square inch
def side_length_smallest_square : ℝ := real.sqrt smallest_square  -- Side length of the smallest square
def area_second_smallest_square : ℝ := side_length_smallest_square ^ 2  -- Area of the second smallest square
def side_length_third_square : ℝ := 2  -- Side length of the third square
def area_third_square : ℝ := side_length_third_square ^ 2  -- Area of the third square

-- The areas of all the squares
def total_area_of_squares : ℝ := smallest_square + area_second_smallest_square + area_third_square

-- The area of the rectangle EFGH
def area_rect_EFGH : ℝ := total_area_of_squares

-- The area of rectangle EFGH should be 6 square inches
theorem area_of_rectangle_EFGH : area_rect_EFGH = 6 := by
  sorry

end area_of_rectangle_EFGH_l126_126643


namespace max_omega_for_increasing_function_l126_126495

theorem max_omega_for_increasing_function :
  ∀ (ω : ℝ), ω > 0 → (∀ x ∈ Icc 0 (π / 4), deriv (λ x, 2 * sin (ω * x)) x > 0) → ω ≤ 2 :=
by sorry

end max_omega_for_increasing_function_l126_126495


namespace find_a1_an_l126_126460

noncomputable def arith_geo_seq (a : ℕ → ℝ) : Prop :=
  (∃ d ≠ 0, (a 2 + a 4 = 10) ∧ (a 2 ^ 2 = a 1 * a 5))

theorem find_a1_an (a : ℕ → ℝ)
  (h_arith_geo_seq : arith_geo_seq a) :
  a 1 = 1 ∧ (∀ n, a n = 2 * n - 1) :=
sorry

end find_a1_an_l126_126460


namespace domain_f_log_l126_126854

noncomputable def domain_f (u : Real) : u ∈ Set.Icc (1 : Real) 2 := sorry

theorem domain_f_log (x : Real) : (x ∈ Set.Icc (4 : Real) 16) :=
by
  have h : ∀ x, (1 : Real) ≤ 2^x ∧ 2^x ≤ 2
  { intro x
    sorry }
  have h_log : ∀ x, 2 ≤ x ∧ x ≤ 4 
  { intro x
    sorry }
  have h_domain : ∀ x, 4 ≤ x ∧ x ≤ 16
  { intro x
    sorry }
  exact sorry

end domain_f_log_l126_126854


namespace factorize_expr_l126_126783

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end factorize_expr_l126_126783


namespace find_y_coordinate_and_sum_of_a_b_c_d_l126_126204

-- Define the coordinates of the points A, B, C, and D
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the condition for the point P
def is_on_ellipse (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (major_axis_length : ℝ) : Prop :=
  (dist P F1 + dist P F2) = major_axis_length

-- Given that P lies on both ellipses
def condition (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P A D 10 ∧ is_on_ellipse P B C 10

-- Define the y-coordinate of P
def y_coordinate_of_P (P : ℝ × ℝ) : ℝ := P.snd

-- Define the form (a, b, c, d)
structure Form : Type :=
(a : ℕ)
(b : ℕ)
(c : ℕ)
(d : ℕ)

-- Define the result of the problem
def result : Form := ⟨0, 0, 7, 7⟩

-- The main statement: if P satisfies the conditions, then the y-coordinate of P is as expected
theorem find_y_coordinate_and_sum_of_a_b_c_d (P : ℝ × ℝ) (h : condition P) :
  ∃ (a b c d : ℕ), y_coordinate_of_P P = (result.a - result.b * real.sqrt result.c) / result.d ∧ 
  result.a + result.b + result.c + result.d = 20 :=
sorry

end find_y_coordinate_and_sum_of_a_b_c_d_l126_126204


namespace raul_money_left_l126_126616

theorem raul_money_left (initial_dollars : ℝ) (comic_price_euro : ℝ) (novel_price_euro : ℝ) (magazine_price_euro : ℝ)
                        (num_comics : ℕ) (num_novels : ℕ) (num_magazines : ℕ)
                        (tax_rate : ℝ) (exchange_rate : ℝ) :
  let total_cost_euro := (num_comics * comic_price_euro + num_novels * novel_price_euro + num_magazines * magazine_price_euro) in
  let total_cost_with_tax_euro := total_cost_euro * (1 + tax_rate) in
  let total_cost_dollars := total_cost_with_tax_euro * (1 / exchange_rate) in
  initial_dollars = 87 →
  comic_price_euro = 4 →
  novel_price_euro = 7 →
  magazine_price_euro = 5.50 →
  num_comics = 8 →
  num_novels = 3 →
  num_magazines = 2 →
  tax_rate = 0.05 →
  exchange_rate = 0.85 →
  initial_dollars - total_cost_dollars ≈ 7.93 := 
by
  intros
  sorry

end raul_money_left_l126_126616


namespace prob_complement_A_l126_126087

theorem prob_complement_A (P : Set (Set α) → ℝ) (A B : Set α)
  (hPB : P B = 0.3)
  (hPBA : P (B \ A) / P (A) = 0.9)
  (hPBNegA : P (B \ ¬A) / P (¬A) = 0.2) :
  P (¬A) = 6 / 7 := by
  sorry

end prob_complement_A_l126_126087


namespace combine_like_terms_l126_126044

theorem combine_like_terms (a : ℝ) : 2 * a + 3 * a = 5 * a := 
by sorry

end combine_like_terms_l126_126044


namespace percentage_equivalence_l126_126535

theorem percentage_equivalence (x : ℝ) :
  (70 / 100) * 600 = (x / 100) * 1050 → x = 40 :=
by
  sorry

end percentage_equivalence_l126_126535


namespace constant_term_in_expansion_l126_126988

open Real

def general_term (n r : ℕ) (x : ℝ) : ℝ :=
  (binomial n r) * (1 / (2 * x))^(n - r) * (-sqrt x)^r

theorem constant_term_in_expansion : general_term 9 6 x = 21 / 2 :=
  by
  sorry

end constant_term_in_expansion_l126_126988


namespace three_digit_numbers_with_6_or_8_l126_126134

theorem three_digit_numbers_with_6_or_8 : 
  let total_count := 900 in
  let without_6_or_8 := 7 * 8 * 8 in
  total_count - without_6_or_8 = 452 :=
by
  let total_count := 900
  let without_6_or_8 := 7 * 8 * 8
  show total_count - without_6_or_8 = 452
  sorry

end three_digit_numbers_with_6_or_8_l126_126134


namespace find_acute_angle_l126_126101

variables (α : ℝ)

def vector_a := (3 / 2 : ℝ, real.sin α)
def vector_b := (real.cos α, 1 / 3 : ℝ)

-- Define parallelism condition
def is_parallel : Prop :=
  vector_a.2 * vector_b.1 = vector_b.2 * vector_a.1

theorem find_acute_angle
  (h: is_parallel α) : 
  α = (real.pi / 4) :=
by
  sorry

end find_acute_angle_l126_126101


namespace ellipse_equation_correct_values_of_m_correct_l126_126461

namespace EllipseProblem

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def major_axis_length (a : ℝ) : Prop := 
  2 * a = 4

def cos_theta_minimum_value (a b : ℝ) : Prop :=
  (2 * b^2) / a^2 - 1 = 1 / 2

def condition_m (m : ℝ) (a b : ℝ) (x1 x2 y1 y2 : ℝ) : Prop :=
  ∀ k : ℝ, 
    (-3 + (4 * m^2 - 3) / (4 * k^2 + 3) = -3) → 
    (m = sqrt 3 / 2 ∨ m = -sqrt 3 / 2)

theorem ellipse_equation_correct : 
  ∀ (x y : ℝ), 
  ∀ (a b : ℝ), 
  (a > b ∧ b > 0) →
  major_axis_length a →
  cos_theta_minimum_value a b →
  ellipse_equation a b x y :=
sorry

theorem values_of_m_correct : 
  ∀ (m : ℝ), 
  ∀ (a b x1 x2 y1 y2 : ℝ), 
  (a > b ∧ b > 0) →
  ellipse_equation a b x1 y1 →
  ellipse_equation a b x2 y2 →
  condition_m m a b x1 x2 y1 y2 :=
sorry

end EllipseProblem

end ellipse_equation_correct_values_of_m_correct_l126_126461


namespace largest_five_digit_divisible_by_97_l126_126677

theorem largest_five_digit_divisible_by_97 :
  ∃ n, (99999 - n % 97) = 99930 ∧ n % 97 = 0 ∧ 10000 ≤ n ∧ n ≤ 99999 :=
by
  sorry

end largest_five_digit_divisible_by_97_l126_126677


namespace stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l126_126440

-- Definitions of fixed points and stable points
def is_fixed_point(f : ℝ → ℝ) (x : ℝ) : Prop := f x = x
def is_stable_point(f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x 

-- Problem 1: Stable points of g(x) = 2x - 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem stable_points_of_g : {x : ℝ | is_stable_point g x} = {1} :=
sorry

-- Problem 2: Prove A ⊂ B for any function f
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) : 
  {x : ℝ | is_fixed_point f x} ⊆ {x : ℝ | is_stable_point f x} :=
sorry

-- Problem 3: Range of a for f(x) = ax^2 - 1 when A = B ≠ ∅
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

theorem range_of_a (a : ℝ) (h : ∃ x, is_fixed_point (f a) x ∧ is_stable_point (f a) x):
  - (1/4 : ℝ) ≤ a ∧ a ≤ (3/4 : ℝ) :=
sorry

end stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l126_126440


namespace distance_from_point_to_line_is_one_l126_126183

def polarPoint := (ρ : ℝ) × (θ : ℝ)

def polarLine := {ρ : ℝ // ∃ θ : ℝ, ρ = 1 / real.sin (θ - real.pi / 6) }

noncomputable def distanceFromPointToLine (P : polarPoint) (L : polarLine) : ℝ := 
  let (ρ, θ) := P
  let (x, y) := (ρ * real.cos θ, ρ * real.sin θ)
  let (a, b, c) := (1.0, -real.sqrt 3, 2.0) -- coefficients for x - sqrt(3)y + 2 = 0
  |a * x + b * y + c| / real.sqrt (a^2 + b^2)

theorem distance_from_point_to_line_is_one 
  (P : polarPoint) 
  (L : polarLine) 
  (hP : P = (2, real.pi / 6))
  (hL : L = ⟨1, ⟨real.pi / 6, rfl⟩⟩) : distanceFromPointToLine P L = 1 := by
  sorry

end distance_from_point_to_line_is_one_l126_126183


namespace tom_loss_at_least_one_event_l126_126046

-- Definitions based on the problem conditions
variables (A B : Prop) -- A and B represent winning Event A and Event B respectively.
variable (trophy : Prop) -- trophy represents receiving the special trophy.

-- The conditions given
def condition (A B : Prop) (trophy : Prop) := (A ∧ B) → trophy

-- The statement we need to prove: If Tom did not receive a special trophy, then he lost at least one of the events.
theorem tom_loss_at_least_one_event (A B trophy : Prop) :
  condition A B trophy → ¬ trophy → (¬ A ∨ ¬ B) :=
by
  intros h not_trophy
  change (¬ (A ∧ B))
  apply mt h
  exact not_trophy

end tom_loss_at_least_one_event_l126_126046


namespace arrange_in_descending_order_l126_126446

noncomputable def a : ℝ := Real.log 3 / Real.log 2  -- log base 2 of 3
noncomputable def b : ℝ := Real.log 3 / Real.log (1/2)  -- log base 1/2 of 3
noncomputable def c : ℝ := 3^(-1/2)  -- 3 raised to the power of -1/2

theorem arrange_in_descending_order : a > c ∧ c > b :=
by
  sorry

end arrange_in_descending_order_l126_126446


namespace horner_method_evaluation_l126_126295

-- Define the polynomial and the evaluation point
def polynomial : ℕ → ℤ
| 3 := 7
| 2 := 3
| 1 := -5
| 0 := 11
| _ := 0

noncomputable def evaluate (p : ℕ → ℤ) (x : ℤ) : ℤ :=
let coeffs := [p 3, p 2, p 1, p 0]
in coeffs.foldr (λ a acc, a + x * acc) 0

theorem horner_method_evaluation :
  evaluate polynomial 23 = 86652 :=
by 
  sorry

end horner_method_evaluation_l126_126295


namespace determine_d_l126_126770

theorem determine_d :
  (∀ x : ℝ, x * (2 * x + 3) < d ↔ x ∈ set.Ioo (-5 / 2) 1) → d = -5 :=
by
  intro h
  sorry

end determine_d_l126_126770


namespace points_of_tangency_passes_through_common_point_l126_126822

noncomputable theory

open_locale classical

variables {R r1 r2 : ℝ} 

-- Define the larger circle
def larger_circle (O : ℝ × ℝ) (R : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = R^2}

-- Define the smaller circles
def smaller_circle1 (O1 : ℝ × ℝ) (r1 : ℝ) := {P : ℝ × ℝ | (P.1 - O1.1)^2 + (P.2 - O1.2)^2 = r1^2}
def smaller_circle2 (O2 : ℝ × ℝ) (r2 : ℝ) := {P : ℝ × ℝ | (P.1 - O2.1)^2 + (P.2 - O2.2)^2 = r2^2}

-- Define the points of tangency
variables {A B : ℝ × ℝ}

-- Condition: The sum of the radii is R
axiom sum_of_radii : r1 + r2 = R

-- Condition: The smaller circles touch the larger circle from the inside
axiom tangency1 : (A.1 - O.1)^2 + (A.2 - O.2)^2 = (R - r1)^2
axiom tangency2 : (B.1 - O.1)^2 + (B.2 - O.2)^2 = (R - r2)^2

-- Define the line segment AB
def line_segment (A B : ℝ × ℝ) := {P : ℝ × ℝ | ∃ λ ∈ (set.Icc 0 1), P = (λ • A) + ((1 - λ) • B)}

-- Statement to prove
theorem points_of_tangency_passes_through_common_point 
  (O O1 O2 : ℝ × ℝ) (P : ℝ × ℝ)
  (hc1 : P ∈ smaller_circle1 O1 r1)
  (hc2 : P ∈ smaller_circle2 O2 r2) :
  P ∈ line_segment A B := 
sorry

end points_of_tangency_passes_through_common_point_l126_126822


namespace least_sum_of_exponents_500_l126_126070

theorem least_sum_of_exponents_500 : ∃ l : list ℕ, (∀ x ∈ l, ∃ k, x = 2^k) ∧ list.distinct l ∧ list.sum l = 500 ∧ list.sum (l.map (λ x, (nat.log (x) / nat.log 2))) = 30 :=
sorry

end least_sum_of_exponents_500_l126_126070


namespace problem_statement_l126_126812

def F (p q : ℕ) : ℕ := sorry -- Define the function F appropriately

theorem problem_statement (m n : ℕ) (a b : ℕ) 
  (h1 : a = 10 + m) 
  (h2 : b = 10 * n + 5) 
  (h3 : 0 ≤ m)
  (h4 : m ≤ 9)
  (h5 : 1 ≤ n)
  (h6 : n ≤ 9)
  (h7 : 150 * F (a, 18) + F (b, 26) = 32761) : 
  m + n = 12 ∨ m + n = 11 ∨ m + n = 10 :=
sorry

end problem_statement_l126_126812


namespace cannot_finish_third_l126_126623

-- Definitions for the orders of runners
def order (a b : String) : Prop := a < b

-- The problem statement and conditions
def conditions (P Q R S T U : String) : Prop :=
  order P Q ∧ order P R ∧ order Q S ∧ order P U ∧ order U T ∧ order T Q

theorem cannot_finish_third (P Q R S T U : String) (h : conditions P Q R S T U) :
  (P = "third" → False) ∧ (S = "third" → False) :=
by
  sorry

end cannot_finish_third_l126_126623


namespace vat_percentage_is_15_l126_126872

def original_price : ℝ := 1700
def final_price : ℝ := 1955
def tax_amount := final_price - original_price

theorem vat_percentage_is_15 :
  (tax_amount / original_price) * 100 = 15 := 
sorry

end vat_percentage_is_15_l126_126872


namespace yellow_day_before_yesterday_white_dandelions_tomorrow_l126_126711

variable (yellow_yesterday white_yesterday yellow_today white_today : ℕ)
variable (bloom_dandelion : ℕ → ℕ)

-- Given conditions
axiom yellow_yesterday_eq : yellow_yesterday = 20
axiom white_yesterday_eq : white_yesterday = 14
axiom yellow_today_eq : yellow_today = 15
axiom white_today_eq : white_today = 11

-- Statement for Part (a)
theorem yellow_day_before_yesterday (yellow_yesterday_obs white_yesterday_obs yellow_today_obs white_today_obs total_yesterday_obs : ℕ) :
  yellow_yesterday_obs = 20 → white_yesterday_obs = 14 → yellow_today_obs = 15 → white_today_obs = 11 →
  total_yesterday_obs = 14 + 11 →
  total_yesterday_obs = 25 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rwa [h₁, h₂, h₃, h₄, h₅]

-- Statement for Part (b)
theorem white_dandelions_tomorrow (yellow_yesterday_obs white_yesterday_obs yellow_today_obs white_today_obs white_tomorrow : ℕ) :
  yellow_yesterday_obs = 20 → white_yesterday_obs = 14 → yellow_today_obs = 15 → white_today_obs = 11 →
  white_tomorrow = 20 - 11 →
  white_tomorrow = 9 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rwa [h₁, h₂, h₃, h₄, h₅]

-- Add the theorem proofs later
sorry

end yellow_day_before_yesterday_white_dandelions_tomorrow_l126_126711


namespace lines_MN_PQ_perpendicular_l126_126456

-- Define the geometric entities and assumptions
variables {A B C D X P Q M N : Type}
variables [IsCyclicQuadrilateral A B C D] 
          [IsIntersection X A C B D]
          [IsOrthogonalProjection P X A D]
          [IsOrthogonalProjection Q X B C]
          [IsMidpoint M A B]
          [IsMidpoint N C D]

-- State the theorem to be proven
theorem lines_MN_PQ_perpendicular 
  (h_cyclic : IsCyclicQuadrilateral A B C D)
  (h_intersection : IsIntersection X A C B D)
  (h_projection_P : IsOrthogonalProjection P X A D)
  (h_projection_Q : IsOrthogonalProjection Q X B C)
  (h_midpoint_M : IsMidpoint M A B)
  (h_midpoint_N : IsMidpoint N C D) :
  ArePerpendicular (LineSegment M N) (LineSegment P Q) := 
by 
  sorry

end lines_MN_PQ_perpendicular_l126_126456


namespace base_is_10_l126_126757

noncomputable def b_base : ℕ := 10

def magic_square : list (list ℕ) := [
  [   1,     b_base + 5,  15,  16],
  [   4,     3,           2 * b_base,  15],
  [2 * b_base + 11, b_base + 7, b_base + 2, b_base + 8],
  [  2,  b_base + 4, b_base, b_base + 3]
]

def row_sum : ℕ := 1 + (b_base + 5) + 15 + 16
def col_sum : ℕ := 4 + 3 + (2 * b_base) + 15

theorem base_is_10 (b : ℕ) (h_row_sum : row_sum = b + 32) (h_col_sum : col_sum = 2 * b + 22) : b = 10 :=
by {
  have h := congr_arg (λ x, x - 22) h_col_sum,
  linarith,
  -- Alternatively, you can break down into smaller steps
  -- rw [add_assoc, add_right_inj] at this,
  -- have : 2 * b = b + 10 := eq_of_add_eq_add_right h,
  -- linarith,
}

end base_is_10_l126_126757


namespace cars_with_both_features_l126_126604

theorem cars_with_both_features (T P_s P_w N B : ℕ)
  (hT : T = 65) 
  (hPs : P_s = 45) 
  (hPw : P_w = 25) 
  (hN : N = 12) 
  (h_equation : P_s + P_w - B + N = T) :
  B = 17 :=
by
  sorry

end cars_with_both_features_l126_126604


namespace percentage_of_y_salary_x_l126_126291

-- Define the conditions as per the problem statement
def total_salary : ℝ := 590
def salary_y : ℝ := 268.1818181818182
def salary_x : ℝ := 321.8181818181818  -- salary_x is calculated as total_salary - salary_y
def percentage (x y : ℝ) : ℝ := (x / y) * 100

-- State the theorem to be proven
theorem percentage_of_y_salary_x (S_X S_Y : ℝ) (total : ℝ) (P : ℝ) :
  S_Y = salary_y → 
  total = total_salary → 
  S_X = total - S_Y → 
  P = percentage S_X S_Y → 
  P = 120 :=
by
  sorry

end percentage_of_y_salary_x_l126_126291


namespace ball_distribution_l126_126518

theorem ball_distribution : 
  let balls := 6
  let boxes := 3
  num_ways balls boxes = 7 :=
sorry

end ball_distribution_l126_126518


namespace tan_ratio_l126_126472

theorem tan_ratio (a β : ℝ) (h : 3 * sin β = sin (2 * a + β)) :
  (tan (a + β)) / (tan a) = 2 :=
sorry

end tan_ratio_l126_126472


namespace remainder_of_x_div_9_is_8_l126_126684

variable (x y r : ℕ)
variable (r_lt_9 : r < 9)
variable (h1 : x = 9 * y + r)
variable (h2 : 2 * x = 14 * y + 1)
variable (h3 : 5 * y - x = 3)

theorem remainder_of_x_div_9_is_8 : r = 8 := by
  sorry

end remainder_of_x_div_9_is_8_l126_126684


namespace sqrt_defined_iff_le_three_l126_126269

theorem sqrt_defined_iff_le_three (x : ℝ) : (∃ y, y = sqrt (3 - x)) ↔ x ≤ 3 := by
  sorry

end sqrt_defined_iff_le_three_l126_126269


namespace unique_solution_k_l126_126771

theorem unique_solution_k (k : ℝ) : 
  (∀ x : ℝ, (3 * x + 5) * (x - 3) = -15 + k * x → (3 * x ^ 2 - (4 + k) * x = 0)) → k = -4 :=
begin
  sorry
end

end unique_solution_k_l126_126771


namespace monochromatic_equilateral_triangle_exists_l126_126780

open Classical

theorem monochromatic_equilateral_triangle_exists (plane_color : Point ℝ × Point ℝ → bool) :
  ∀ (x y z : Point ℝ × Point ℝ), equilateral_triangle x y z → (plane_color x = plane_color y ∨ plane_color y = plane_color z ∨ plane_color z = plane_color x) → False :=
by sorry

end monochromatic_equilateral_triangle_exists_l126_126780


namespace polynomial_negative_l126_126954

-- Definition of f(x)
def f (a x : ℝ) := 
  (a - x) ^ 6 - 3 * a * (a - x) ^ 5 + (5 / 2) * a ^ 2 * (a - x) ^ 4 - (1 / 2) * a ^ 4 * (a - x) ^ 2

theorem polynomial_negative (a x : ℝ) (h1 : 0 < x) (h2 : x < a) : f a x < 0 :=
  sorry

end polynomial_negative_l126_126954


namespace vasya_digits_l126_126298

theorem vasya_digits (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (hsum : 33 * (a + b + c) = 231) :
  {a, b, c} = {1, 2, 4} :=
sorry

end vasya_digits_l126_126298


namespace sum_of_squares_roots_eq_zero_l126_126051

theorem sum_of_squares_roots_eq_zero (r : Fin 2023 → ℂ) (hr : ∀ i, (Polynomial.root_set (Polynomial.X ^ 2023 + 50 * Polynomial.X ^ 2020 + 4 * Polynomial.X ^ 4 + 505) ℂ) i) : 
  (Finset.univ.sum (λ i, r i ^ 2)) = 0 :=
begin
  -- Conditions as assumptions
  have hsum : Finset.univ.sum r = 0,
  { sorry }, -- This is where the proof of sum of roots being zero goes.

  have hprod : Finset.univ.sum (λ (i j : ℕ), r i * r j) = 0,
  { sorry }, -- This is where the proof of sum of products of roots being zero goes.

  -- Proof of the required statement, following the steps in the solution
  -- Reason based on conditions hsum and hprod
  sorry -- Proof of the final statement.
end

end sum_of_squares_roots_eq_zero_l126_126051


namespace perpendicular_lines_implies_parallel_l126_126467

variables (a b c d : Type)
variables [inner_product_space ℝ a] [inner_product_space ℝ b] [inner_product_space ℝ c] [inner_product_space ℝ d]

-- Defining perpendicular relationships
def is_perpendicular (x y : Type) [inner_product_space ℝ x] [inner_product_space ℝ y] : Prop := sorry

-- Defining parallel relationships
def is_parallel (x y : Type) [inner_product_space ℝ x] [inner_product_space ℝ y] : Prop := sorry

axiom a_perpendicular_b : is_perpendicular a b
axiom b_perpendicular_c : is_perpendicular b c
axiom c_perpendicular_d : is_perpendicular c d
axiom d_perpendicular_a : is_perpendicular d a

-- Proof problem statement
theorem perpendicular_lines_implies_parallel :
  (is_parallel b d) ∨ (is_parallel a c) :=
by sorry

end perpendicular_lines_implies_parallel_l126_126467


namespace chord_tangent_conjugate_diameters_l126_126986

def circle_eq (a b : ℝ) : ℝ → ℝ → Prop :=
λ x y, x^2 + y^2 = a^2 + b^2

def ellipse_eq (a b : ℝ) : ℝ → ℝ → Prop :=
λ x y, x^2 / a^2 + y^2 / b^2 = 1

def is_tangent (P Q : ℝ × ℝ) (E : ℝ → ℝ → Prop) : Prop :=
-- definition of tangent to be formulated
sorry

def conjugate_diameters (P Q : ℝ × ℝ) (O : ℝ × ℝ) (E : ℝ → ℝ → Prop) : Prop :=
-- definition of conjugate diameters to be formulated
sorry

theorem chord_tangent_conjugate_diameters
  (a b : ℝ)
  (P Q : ℝ × ℝ)
  (h1 : circle_eq a b P.1 P.2 ∧ circle_eq a b Q.1 Q.2)
  (h2 : ∃ P Q, is_tangent (P, Q) (ellipse_eq a b))
  (O : ℝ × ℝ) (hc : O = (0, 0)) :
  conjugate_diameters P Q O (ellipse_eq a b) :=
sorry

end chord_tangent_conjugate_diameters_l126_126986


namespace find_t_l126_126858

-- Definitions based on given conditions
def l1_slope : ℝ := sqrt 3 / 3
def l2_slope (t : ℝ) : ℝ := -1 / t
def angle_60_degrees : ℝ := Real.arctan (sqrt 3)

-- The main theorem stating that t = 0 or t = sqrt(3)
theorem find_t (t : ℝ) : arctan (abs ((l1_slope - l2_slope t) / (1 + l1_slope * l2_slope t))) = angle_60_degrees ↔ t = 0 ∨ t = sqrt 3 := 
by sorry

end find_t_l126_126858


namespace solve_for_x_l126_126521

theorem solve_for_x (x : ℝ) : (9 / x^2 = x / 36) → x = real.cbrt 324 :=
by
  sorry

end solve_for_x_l126_126521


namespace isabel_ds_games_left_l126_126572

-- Define the initial number of DS games Isabel had
def initial_ds_games : ℕ := 90

-- Define the number of DS games Isabel gave to her friend
def ds_games_given : ℕ := 87

-- Define a function to calculate the remaining DS games
def remaining_ds_games (initial : ℕ) (given : ℕ) : ℕ := initial - given

-- Statement of the theorem we need to prove
theorem isabel_ds_games_left : remaining_ds_games initial_ds_games ds_games_given = 3 := by
  sorry

end isabel_ds_games_left_l126_126572


namespace convex_hull_not_triangle_l126_126831

theorem convex_hull_not_triangle (n : ℕ) (h : n > 0)
  (points : Finset (E)) (h_size : points.card = 3*n - 1)
  (h_no_three_collinear : ∀ {a b c : E}, a ∈ points → b ∈ points → c ∈ points 
    → a ≠ b → b ≠ c → a ≠ c → ¬Collinear k {a, b, c}) :
  ∃ (subset : Finset (E)), subset.card = 2*n ∧ ¬∃ (triangle : Set E), 
  ConvexHull k ↑subset = triangle ∧ Affine.independent k (↑(Finset.coe subset)) :=
begin
  sorry
end

end convex_hull_not_triangle_l126_126831


namespace percentage_transform_l126_126531

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end percentage_transform_l126_126531


namespace no_arrangement_13_arrangement_14_l126_126950

def can_be_arranged_13 (s : List ℕ) : Prop :=
  -- Define the properties for arranging numbers 1 to 13
  s.length = 13 ∧
  ∀ (i : ℕ), i < 13 → 
  abs (s.get! i - s.get! ((i + 1) % 13)) ∈ {3, 4, 5}

def can_be_arranged_14 (s : List ℕ) : Prop :=
  -- Define the properties for arranging numbers 1 to 14
  s.length = 14 ∧
  ∀ (i : ℕ), i < 14 → 
  abs (s.get! i - s.get! ((i + 1) % 14)) ∈ {3, 4, 5}

theorem no_arrangement_13 : ¬∃ (s : List ℕ), can_be_arranged_13 s :=
by
  sorry -- Prove that such an arrangement for {1, 2, ..., 13} is impossible

theorem arrangement_14 : ∃ (s : List ℕ), can_be_arranged_14 s :=
by
  sorry -- Prove that such an arrangement for {1, 2, ..., 14} is possible

end no_arrangement_13_arrangement_14_l126_126950


namespace sqrt_square_eq_self_l126_126756

theorem sqrt_square_eq_self (n : ℝ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n :=
by
  sorry

example : (Real.sqrt 930249) ^ 2 = 930249 :=
begin
  apply sqrt_square_eq_self,
  -- We need to check that 930249 is nonnegative to use the theorem.
  norm_num,
end

end sqrt_square_eq_self_l126_126756


namespace line_equation_correct_l126_126641

-- The slope of a line given by the angle θ can be defined as the tangent of θ.
def slope (θ : ℝ) : ℝ := Real.tan θ

-- Given conditions
def θ : ℝ := Real.pi + Real.pi / 4  -- 135 degrees in radians
def y_intercept : ℝ := 3
def expected_line_eq (x : ℝ) : ℝ := -x + 3  -- Expected equation y = -x + 3

-- The slope-intercept form of the line equation is y = mx + b
def line_eq (m b x : ℝ) : ℝ := m * x + b

theorem line_equation_correct (x : ℝ) :
  line_eq (slope θ) y_intercept x = expected_line_eq x := by
  sorry

end line_equation_correct_l126_126641


namespace angles_of_triangle_DEF_are_acute_l126_126156

noncomputable theory
open Classical

variables {A B C D E F : Type} [EuclideanGeometry]

-- Given any triangle ABC with an incircle touching sides BC, CA, and AB at D, E, and F respectively
def is_incircle_tangent_points (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- To prove: the angles of the resulting triangle DEF are always acute
theorem angles_of_triangle_DEF_are_acute (α β γ : ℝ)
  (h : is_incircle_tangent_points α β γ) :
  0 < (α + β) / 2 ∧ (α + β) / 2 < 90 ∧
  0 < (β + γ) / 2 ∧ (β + γ) / 2 < 90 ∧
  0 < (γ + α) / 2 ∧ (γ + α) / 2 < 90 :=
sorry

end angles_of_triangle_DEF_are_acute_l126_126156


namespace length_AB_is_sqrt_2_l126_126906

noncomputable def parametric_line (s : ℝ) : ℝ × ℝ :=
  (1 + s, 1 - s)

noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (t + 2, t^2)

def is_intersection (p : ℝ × ℝ) : Prop :=
  ∃ s t : ℝ, parametric_line s = p ∧ parametric_curve t = p

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem length_AB_is_sqrt_2 : 
  ∃ A B : ℝ × ℝ, is_intersection A ∧ is_intersection B ∧ distance A B = real.sqrt 2 :=
by
  sorry

end length_AB_is_sqrt_2_l126_126906


namespace num_ways_to_pick_marbles_l126_126139

theorem num_ways_to_pick_marbles : 
  let my_marbles := {2, 3, 4, 5, 6, 7, 8, 9}
  let sara_marbles := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
  ∃ num_matches : ℕ, num_matches = 4 ∧
  ∀ m ∈ my_marbles, ∃ s ∈ sara_marbles, 3 * m = s := sorry

end num_ways_to_pick_marbles_l126_126139


namespace groupings_of_guides_and_tourists_l126_126666

theorem groupings_of_guides_and_tourists :
  let guides := 3
  let tourists := 8
  -- The number of different groupings where each guide has at least one tourist
  ∑ (partitions : Fin.tourists -> Fin.guides), (⧸ ∀ g : Fin.guides, ∃ t : Fin.tourists, partitions t = g) = 5796 :=
sorry

end groupings_of_guides_and_tourists_l126_126666


namespace book_pages_l126_126444

theorem book_pages (days_to_finish: ℕ) (pages_per_day: ℕ) (total_pages: ℕ) : 
  (days_to_finish = 3) ∧ (pages_per_day = 83) → total_pages = 249 := 
by
  intros h
  cases h with h1 h2
  have h3 := h2 ▸ h1 ▸ rfl
  exact h3
  sorry

end book_pages_l126_126444


namespace base_6_digit_divisibility_l126_126057

theorem base_6_digit_divisibility (d : ℕ) (h1 : d < 6) : ∃ t : ℤ, (655 + 42 * d) = 13 * t :=
by sorry

end base_6_digit_divisibility_l126_126057


namespace evaluate_expression_at_x_zero_l126_126619

theorem evaluate_expression_at_x_zero (x : ℕ) (h1 : x < 3) (h2 : x ≠ 1) (h3 : x ≠ 2) : ((3 / (x - 1) - x - 1) / (x - 2) / (x^2 - 2 * x + 1)) = 2 :=
by
  -- Here we need to provide our proof, though for now it’s indicated by sorry
  sorry

end evaluate_expression_at_x_zero_l126_126619


namespace area_of_transformed_region_l126_126584

-- Define the region area condition
def region_area (T : Type) [MeasureSpace T] : Real := 9

-- Define the transformation matrix
def transform_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 2], ![5, 3]]

-- Define the transformed region area proof statement
theorem area_of_transformed_region (T T' : Type) [MeasureSpace T] [MeasureSpace T'] (h : region_area T = 9) :
  let area_T' := abs (Matrix.det transform_matrix) * region_area T in
  area_T' = 9 :=
by
  -- Proof steps and actual proof would go here
  sorry

end area_of_transformed_region_l126_126584


namespace part1_part2_l126_126867

variable (x : ℝ)
def a : ℝ × ℝ := (2 * sin (x + π/6), -2)
def b : ℝ × ℝ := (2, (sqrt 3) / 2 - 2 * cos x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Part (Ⅰ): Prove that if a is perpendicular to b, then sin (x + 4 * π / 3) = -1 / 4.
theorem part1 (h : (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0) :
  sin (x + 4 * π / 3) = -1 / 4 :=
sorry

-- Part (Ⅱ): Prove that for f(x), the range of f(x) when x ∈ [0, π] is [-6 - sqrt 3, 3 * sqrt 3].
theorem part2 : 
  x ∈ Set.Icc 0 π → f x ∈ Set.Icc (-6 - sqrt 3) (3 * sqrt 3) :=
sorry

end part1_part2_l126_126867


namespace percent_problem_l126_126534

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end percent_problem_l126_126534


namespace additional_laps_needed_l126_126936

-- Definitions of problem conditions
def total_required_distance : ℕ := 2400
def lap_length : ℕ := 150
def madison_laps : ℕ := 6
def gigi_laps : ℕ := 6

-- Target statement to prove the number of additional laps needed
theorem additional_laps_needed : (total_required_distance - (madison_laps + gigi_laps) * lap_length) / lap_length = 4 := by
  sorry

end additional_laps_needed_l126_126936


namespace overall_percentage_increase_l126_126991

theorem overall_percentage_increase
  (initial_visitors_children initial_visitors_seniors initial_visitors_13_to_59 : ℕ)
  (discount_children discount_seniors discount_13_to_59 : ℕ)
  (increase_children increase_seniors : ℕ)
  (remains_unchanged : ℕ)
  (initial_visitors : initial_visitors_children = 100 ∧ initial_visitors_seniors = 100 ∧ initial_visitors_13_to_59 = 100)
  (discounts : discount_children = 20 ∧ discount_seniors = 30 ∧ discount_13_to_59 = 10)
  (increases : increase_children = 25 ∧ increase_seniors = 15 ∧ remains_unchanged = 0):
  let total_before := initial_visitors_children + initial_visitors_seniors + initial_visitors_13_to_59,
      total_after := (initial_visitors_children + initial_visitors_children * increase_children / 100) +
                      (initial_visitors_seniors + initial_visitors_seniors * increase_seniors / 100) +
                      initial_visitors_13_to_59
  in total_before = 300 ∧ total_after = 340 → 
     let increase := total_after - total_before 
     in ((increase * 100) / total_before) = 13.33 := 
by
  intros initial_visitors discounts increases total_before total_after h,
  sorry

end overall_percentage_increase_l126_126991


namespace triangle_inequality_l126_126208

variable (a b c : ℝ) (t : ℝ)
variable (h1 : a + b > c)
variable (h2 : b + c > a)
variable (h3 : c + a > b)

noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_inequality (a b c : ℝ) (t : ℝ)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) 
  (h4 : t = area a b c) : 4 * t ≤ Real.sqrt 3 * Real.cbrt (a^2 * b^2 * c^2) :=
by
  sorry

end triangle_inequality_l126_126208


namespace arithmetic_sequence_solution_l126_126421

noncomputable def fractional_part (x : ℝ) : ℝ := x - Real.floor x

theorem arithmetic_sequence_solution (x : ℝ) (hx : x ≠ 0) (h_seq : fractional_part x + (Real.floor x + 1) + x = Real.floor x + 1 + x * 2) : x = -2 ∨ x = -1 / 2 :=
  sorry

end arithmetic_sequence_solution_l126_126421


namespace determine_a_l126_126475

/-- Given the complex number \( z \) which satisfies \( z = \frac{2 + a * complex.I}{1 + complex.I} \) where \( a \) is a real number, 
 and if the point corresponding to \( z \) lies on the line \( y = -x \), then prove that \( a \) equals 0. -/
theorem determine_a 
  (a : ℝ)
  (z : ℂ)
  (hz : z = (2 + a * complex.I) / (1 + complex.I))
  (hline : (z.re = z.im)) : a = 0 :=
sorry

end determine_a_l126_126475


namespace min_omega_value_shifted_cos_sin_l126_126145

theorem min_omega_value_shifted_cos_sin (ω : ℝ) (k : ℤ) (π_pos : π > 0) : 
  (ω > 0) ∧ (∀ x, cos (ω * x + π / 3) = sin (ω * x)) → ω = 5 / 2 := 
by {
  sorry
}

end min_omega_value_shifted_cos_sin_l126_126145


namespace coordinates_of_point_A_l126_126901

theorem coordinates_of_point_A (A B : ℝ × ℝ) (hAB : B.1 = 2 ∧ B.2 = 4) (hParallel : A.2 = B.2) (hDist : abs (A.1 - B.1) = 3) :
  A = (5, 4) ∨ A = (-1, 4) :=
by
  cases hAB with hx hy
  rw [hx, hy] at *
  cases hParallel
  rw [hParallel] at hDist
  cases abs_eq (A.1 - 2) (A.1 - 2) with ha ha
  case h1 =>
    rw [add_comm, add_right_eq_self, sub_eq_iff_eq_add] at ha
    left
    exact ⟨hx.symm ▸ ha, hParallel⟩
  case h2 =>
    rw [add_neg_cancel_right, add_eq_zero_iff_eq_neg_eq] at ha
    right
    exact ⟨hx.symm ▸ ha, hParallel⟩

end coordinates_of_point_A_l126_126901


namespace evaluate_expression_l126_126068

def floor_0_998 : ℤ := Int.floor 0.998
def ceiling_3_002 : ℤ := Int.ceil 3.002

theorem evaluate_expression : floor_0_998 + ceiling_3_002 = 4 :=
by
  -- Directly use the given and previously defined values.
  have h1 : floor_0_998 = 0 := by norm_cast
    rw [floor_0_998, Int.floor_eq_nat_floor, Nat.floor_cast]
    norm_num

  have h2 : ceiling_3_002 = 4 := by norm_cast
    rw [ceiling_3_002, Int.ceil_eq_nat_ceil, Nat.ceil_cast]
    norm_num

  -- Conclude the expression
  rw [h1, h2]
  norm_num

end evaluate_expression_l126_126068


namespace calculate_expression_evaluate_expression_l126_126385

theorem calculate_expression (a : ℕ) (h : a = 2020) :
  (a^4 - 3*a^3*(a+1) + 4*a*(a+1)^3 - (a+1)^4 + 1) / (a*(a+1)) = a^2 + 4*a + 6 :=
by sorry

theorem evaluate_expression :
  (2020^2 + 4 * 2020 + 6) = 4096046 :=
by sorry

end calculate_expression_evaluate_expression_l126_126385


namespace instantaneous_velocity_at_3s_l126_126105

noncomputable def S (t : ℝ) : ℝ := 2 * t^3 + t

theorem instantaneous_velocity_at_3s : 
  let S' := deriv S in S' 3 = 55 := 
by
  sorry

end instantaneous_velocity_at_3s_l126_126105


namespace square_factors_product_l126_126516

noncomputable def count_square_factors_of_prime_powers (n : ℕ) : ℕ :=
  (n / 2) + 1

theorem square_factors_product :
  let n := (2^12) * (3^10) * (5^9) * (7^8) in
  (count_square_factors_of_prime_powers 12) *
  (count_square_factors_of_prime_powers 10) *
  (count_square_factors_of_prime_powers 9) *
  (count_square_factors_of_prime_powers 8) = 1050 :=
by
  sorry

end square_factors_product_l126_126516


namespace percentage_transform_l126_126529

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end percentage_transform_l126_126529


namespace sum_of_valid_integers_k_l126_126801

theorem sum_of_valid_integers_k :
  let valid_k_values := {k : ℕ | k = 7 ∨ k = 19} in
  (∑ k in valid_k_values, k) = 26 :=
by
  sorry

end sum_of_valid_integers_k_l126_126801


namespace first_business_owner_donation_l126_126772

theorem first_business_owner_donation
    (cakes : ℕ)
    (slices_per_cake : ℕ)
    (price_per_slice : ℚ)
    (total_amount_raised : ℚ)
    (second_donation_per_slice : ℚ) :
    cakes = 10 →
    slices_per_cake = 8 →
    price_per_slice = 1 →
    total_amount_raised = 140 →
    second_donation_per_slice = 0.25 →
    (let total_slices := cakes * slices_per_cake,
         money_from_slices := total_slices * price_per_slice,
         total_donation := total_amount_raised - money_from_slices,
         second_total_donation := total_slices * second_donation_per_slice,
         first_total_donation := total_donation - second_total_donation,
         first_donation_per_slice := first_total_donation / total_slices
   in first_donation_per_slice) = 0.50 :=
by
  intros _ _ _ _ _
  sorry

end first_business_owner_donation_l126_126772


namespace cos_540_eq_neg_1_l126_126043

theorem cos_540_eq_neg_1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end cos_540_eq_neg_1_l126_126043


namespace total_trip_time_l126_126376

noncomputable def speed_coastal := 10 / 20  -- miles per minute
noncomputable def speed_highway := 4 * speed_coastal  -- miles per minute
noncomputable def time_highway := 50 / speed_highway  -- minutes
noncomputable def total_time := 20 + time_highway  -- minutes

theorem total_trip_time : total_time = 45 := 
by
  -- Proof omitted
  sorry

end total_trip_time_l126_126376


namespace max_non_overlapping_squares_l126_126944

theorem max_non_overlapping_squares (m n : ℕ) : 
  ∃ max_squares : ℕ, max_squares = m :=
by
  sorry

end max_non_overlapping_squares_l126_126944


namespace solve_for_constants_l126_126767

theorem solve_for_constants : ∃ a b : ℝ, 3 * a + b = 0 ∧ -a + 4 * b = 5 :=
by {
  use [-5/13, 15/13],
  split;
  ring.
}

end solve_for_constants_l126_126767


namespace determine_vertices_l126_126737

-- Definitions based on conditions
variables (A B C E K M : Point)
variables [is_vertex A] [is_vertex B] [is_vertex C]
variables [on_triangle_boundary A B C E]
variables [on_triangle_boundary A B C K]
variables [on_triangle_boundary A B C M]

-- Define the relationship between the points in Lean 4 
theorem determine_vertices (A B C E K M : Point)
  (h1 : is_vertex A ∧ is_vertex B ∧ is_vertex C)
  (h2 : on_triangle_boundary A B C E ∧ on_triangle_boundary A B C K ∧ on_triangle_boundary A B C M) :
  ∃ (A B C : Point), true :=
by 
  sorry

end determine_vertices_l126_126737


namespace average_eq_1_4_l126_126098

noncomputable def average_value (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

theorem average_eq_1_4 (x : Fin 10 → ℝ)
  (h1 : ∑ i, |x i - 1| ≤ 4)
  (h2 : ∑ i, |x i - 2| ≤ 6) :
  average_value x = 1.4 :=
sorry

end average_eq_1_4_l126_126098


namespace resultant_after_trebled_l126_126356

variable (x : ℕ)

theorem resultant_after_trebled (h : x = 7) : 3 * (2 * x + 9) = 69 := by
  sorry

end resultant_after_trebled_l126_126356


namespace find_m_l126_126883

variable (m : ℝ)

def P : ℝ × ℝ := (m, 3)
def line_eq (x y : ℝ) := 4 * x - 3 * y + 1 = 0
def distance_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  let (x, y) := P in
  abs (a * x + b * y + c) / Math.sqrt (a * a + b * b)

def region (x : ℝ) (y : ℝ) := 2 * x + y < 3

theorem find_m
  (h1 : distance_to_line P 4 (-3) 1 = 4)
  (h2 : region m 3) :
  m = -3 := sorry

end find_m_l126_126883


namespace func_above_x_axis_l126_126769

theorem func_above_x_axis (a : ℝ) :
  (∀ x : ℝ, (x^4 + 4*x^3 + a*x^2 - 4*x + 1) > 0) ↔ a > 2 :=
sorry

end func_above_x_axis_l126_126769


namespace find_missing_ratio_l126_126639

theorem find_missing_ratio
  (x y : ℕ)
  (h : ((2 / 3 : ℚ) * (x / y : ℚ) * (11 / 2 : ℚ) = 2)) :
  x = 6 ∧ y = 11 :=
sorry

end find_missing_ratio_l126_126639


namespace smallest_a_undefined_inverse_l126_126681

theorem smallest_a_undefined_inverse (a : ℕ) (ha : a = 2) :
  (∀ (a : ℕ), 0 < a → ((Nat.gcd a 40 > 1) ∧ (Nat.gcd a 90 > 1)) ↔ a = 2) :=
by
  sorry

end smallest_a_undefined_inverse_l126_126681


namespace lambda_over_mu_neg_half_l126_126868

open Real

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem lambda_over_mu_neg_half
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (nonzero_a_b : (a - 2 * b) ≠ (0, 0))
  (h : collinear (λ λ μ : ℝ, (λ * a.1 + μ * b.1, λ * a.2 + μ * b.2)) (a.1 - 2 * b.1, a.2 - 2 * b.2)) :
  ∀ λ μ : ℝ,
  (λ a b : ℝ × ℝ, b ≠ 0) (λ * a + μ * b) (a - 2 * b) → λ / μ = -1 / 2 :=
by
  sorry

end lambda_over_mu_neg_half_l126_126868


namespace num_solutions_l126_126428

theorem num_solutions (x y : ℤ) : 
  (∃ x y : ℤ, 6 * x^2 - 7 * x * y + y^2 = 10^100) ↔ (19998 = (set.count {⟨x, y⟩ : ℤ × ℤ | 6 * x^2 - 7 * x * y + y^2 = 10^100})) :=
sorry

end num_solutions_l126_126428


namespace conditional_probability_l126_126538

open Probability

def study_group : Finset (Fin 6) := {0, 1, 2, 3, 4, 5}
def halls : Finset (Finite 3) := {0, 1, 2} -- Halls A, B, C

noncomputable def event_A (visiting_order : Fin 6 → Fin 3) : Prop :=
  ∀ h : Fin 3, (study_group.filter (λ i => visiting_order i = h)).card = 2

noncomputable def event_B (visiting_order_2 : Fin 6 → Fin 3) : Prop :=
  (study_group.filter (λ i => visiting_order_2 i = 0)).card = 2

theorem conditional_probability
  (visiting_order_1 visiting_order_2 : Fin 6 → Fin 3)
  (h_A : event_A visiting_order_1) :
  P (event_B visiting_order_2) ∣ event_A visiting_order_1 = 3 / 8 :=
sorry

end conditional_probability_l126_126538


namespace total_revenue_is_correct_l126_126219

-- Define the constants and conditions
def price_of_jeans : ℕ := 11
def price_of_tees : ℕ := 8
def quantity_of_tees_sold : ℕ := 7
def quantity_of_jeans_sold : ℕ := 4

-- Define the total revenue calculation
def total_revenue : ℕ :=
  (price_of_tees * quantity_of_tees_sold) +
  (price_of_jeans * quantity_of_jeans_sold)

-- The theorem to prove
theorem total_revenue_is_correct : total_revenue = 100 := 
by
  -- Proof is omitted for now
  sorry

end total_revenue_is_correct_l126_126219


namespace problem_statement_l126_126210

theorem problem_statement (n : ℤ) (h_odd: Odd n) (h_pos: n > 0) (h_not_divisible_by_3: ¬(3 ∣ n)) : 24 ∣ (n^2 - 1) :=
sorry

end problem_statement_l126_126210


namespace ineq_geom_triangle_l126_126199

-- Definitions for geometric entities and concepts
variables {α : Type*} [ordered_field α]
variables (A B C M X : Type*)
variable {R r : α}
variable (AM AX : α)
variable (circumradius inscribe_radius : α)
variable (midpoint largest_side : Type*)
variable (tangent_intersection : Type*)

-- Hypotheses
def is_circumradius (R : α) (circumradius : α) : Prop :=
  R = circumradius

def is_inradius (r : α) (inscribe_radius : α) : Prop :=
  r = inscribe_radius

def is_midpoint (M : Type*) (largest_side : Type*) : Prop :=
  M = midpoint

def is_tangent_intersection (X : Type*) (tangent_intersection : Type*) : Prop :=
  X = tangent_intersection

-- Theorems
theorem ineq_geom_triangle (h1 : is_circumradius R circumradius) 
                           (h2 : is_inradius r inscribe_radius)
                           (h3 : is_midpoint M largest_side)
                           (h4 : is_tangent_intersection X tangent_intersection)
                           (h5 : AM = (A, M)) 
                           (h6 : AX = (A, X)) : 
                           r / R ≥ AM / AX := 
sorry

end ineq_geom_triangle_l126_126199


namespace intersection_M_N_l126_126122

def M : Set ℝ := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
def N : Set ℝ := Set.univ

theorem intersection_M_N : M ∩ N = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l126_126122


namespace square_side_length_l126_126313

theorem square_side_length (π : ℝ) (s : ℝ) :
  (∃ r : ℝ, 100 = π * r^2) ∧ (4 * s = 100) → s = 25 := by
  sorry

end square_side_length_l126_126313


namespace monotonically_increasing_interval_l126_126648

open Real

-- Given definition of the function
def f (x : ℝ) : ℝ := sqrt 3 * sin x + cos x

-- Statement: interval of monotonically increasing function
theorem monotonically_increasing_interval :
  ∀ k : ℤ, monotone_on f (set.Ioo (2 * k * π - (2 * π / 3)) (2 * k * π + (π / 3))) :=
sorry

end monotonically_increasing_interval_l126_126648


namespace geometric_progression_fifth_term_sum_l126_126125

def gp_sum_fifth_term
    (p q : ℝ)
    (hpq_sum : p + q = 3)
    (hpq_6th : p^5 + q^5 = 573) : ℝ :=
p^4 + q^4

theorem geometric_progression_fifth_term_sum :
    ∃ p q : ℝ, p + q = 3 ∧ p^5 + q^5 = 573 ∧ gp_sum_fifth_term p q (by sorry) (by sorry) = 161 :=
by
  sorry

end geometric_progression_fifth_term_sum_l126_126125


namespace sin_pi_over_2_plus_2alpha_l126_126478

theorem sin_pi_over_2_plus_2alpha (α : ℝ) (y : ℝ) 
  (h1 : (1 / 2) ^ 2 + y ^ 2 = 1) : 
  sin (π / 2 + 2 * α) = -1 / 2 := 
sorry

end sin_pi_over_2_plus_2alpha_l126_126478


namespace find_coefficients_l126_126062

theorem find_coefficients (A B C D : ℚ) :
  (∀ x : ℚ, x ≠ -1 → 
  (A / (x + 1)) + (B / (x + 1)^2) + ((C * x + D) / (x^2 + x + 1)) = 
  1 / ((x + 1)^2 * (x^2 + x + 1))) →
  A = 1 ∧ B = 1 ∧ C = -1 ∧ D = -1 :=
sorry

end find_coefficients_l126_126062


namespace area_of_R3_l126_126363

theorem area_of_R3 (r1 r2 r3 : ℝ) (h1: r1^2 = 25) 
                   (h2: r2 = (2/3) * r1) (h3: r3 = (2/3) * r2) :
                   r3^2 = 400 / 81 := 
by
  sorry

end area_of_R3_l126_126363


namespace ordered_pair_represents_5_1_l126_126539

structure OrderedPair (α : Type) :=
  (fst : α)
  (snd : α)

def represents_rows_cols (pair : OrderedPair ℝ) (rows cols : ℕ) : Prop :=
  pair.fst = rows ∧ pair.snd = cols

theorem ordered_pair_represents_5_1 :
  represents_rows_cols (OrderedPair.mk 2 3) 2 3 →
  represents_rows_cols (OrderedPair.mk 5 1) 5 1 :=
by
  intros h
  sorry

end ordered_pair_represents_5_1_l126_126539


namespace makeup_palette_cost_l126_126599

variable (lipstick_cost : ℝ := 2.5)
variable (num_lipsticks : ℕ := 4)
variable (hair_color_cost : ℝ := 4)
variable (num_boxes_hair_color : ℕ := 3)
variable (total_cost : ℝ := 67)
variable (num_palettes : ℕ := 3)

theorem makeup_palette_cost :
  (total_cost - (num_lipsticks * lipstick_cost + num_boxes_hair_color * hair_color_cost)) / num_palettes = 15 := 
by
  sorry

end makeup_palette_cost_l126_126599


namespace evaluate_f_at_5_l126_126586

def f (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 38*x^2 - 35*x - 40

theorem evaluate_f_at_5 : f 5 = 110 :=
by
  sorry

end evaluate_f_at_5_l126_126586


namespace divides_difference_l126_126435

theorem divides_difference (n : ℕ) (h_composite : ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k) : 
  6 ∣ ((n^2)^3 - n^2) := 
sorry

end divides_difference_l126_126435


namespace number_of_integers_le_100_with_D_2_and_S_3_l126_126809
-- Lean 4 statement

def D (n : ℕ) : ℕ := sorry  -- Number of pairs of different adjacent digits in binary representation of n
def S (n : ℕ) : ℕ := sorry  -- Sum of the digits in binary representation of n

theorem number_of_integers_le_100_with_D_2_and_S_3 : 
  (∑ n in Finset.range 101, if D n = 2 ∧ S n = 3 then 1 else 0) = 8 :=
by sorry

end number_of_integers_le_100_with_D_2_and_S_3_l126_126809


namespace tina_bought_more_notebooks_l126_126912

theorem tina_bought_more_notebooks :
  let j_expenditure := 1.80
  let t_expenditure := 3.00
  let price_per_notebook := 0.30 in
  ∃ n : ℕ, t_expenditure / price_per_notebook - j_expenditure / price_per_notebook = n ∧ n = 4 :=
begin
  let j_expenditure := 1.80,
  let t_expenditure := 3.00,
  let price_per_notebook := 0.30,
  use 4,
  split,
  { calc t_expenditure / price_per_notebook - j_expenditure / price_per_notebook
        = 3.00 / 0.30 - 1.80 / 0.30 : by refl
    ... = 10 - 6 : by norm_num
    ... = 4 : by norm_num },
  { refl }
end

end tina_bought_more_notebooks_l126_126912


namespace eighth_root_390625000000000_l126_126766

theorem eighth_root_390625000000000 : (390625000000000 : ℕ) = (101 : ℕ) ^ 8 :=
by {
  have h1 : 390625000000000 = ∑ k in finset.range 9, nat.choose 8 k * (100^(8-k)) := sorry,
  rw [h1],
  have h2 : ∑ k in finset.range 9, nat.choose 8 k * (100^(8-k)) = (100 + 1) ^ 8 := sorry,
  rw [h2],
  exact pow_succ_eq_mul (100 + 1) 7,
  norm_num,
}

end eighth_root_390625000000000_l126_126766


namespace length_of_k_eq_4_l126_126436

def length_of_integer (n : ℤ) : ℕ :=
  if n > 1 then (Multiset.card (prime_multiset n)) else 0

theorem length_of_k_eq_4 (k : ℤ) (h1 : k = 24) (h2 : k > 1) : length_of_integer k = 4 :=
by {
  sorry
}

end length_of_k_eq_4_l126_126436


namespace xy_range_xy_sum_range_x_2y_min_value_x_5y_not_min_value_l126_126449

theorem xy_range (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y + x * y = 3) : 0 < x * y ∧ x * y ≤ 1 := 
sorry

theorem xy_sum_range (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y + x * y = 3) : ¬ (2 ≤ x + y ∧ x + y ≤ 3) := 
sorry

theorem x_2y_min_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y + x * y = 3) : ∃ x y, x = -1 + 4 / (y + 1) ∧ y = sqrt 2 - 1 ∧ x + 2 * y = 4 * sqrt 2 - 3 := 
sorry

theorem x_5y_not_min_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y + x * y = 3) : ¬ (∃ x y, x = -1 + 4 / (y + 1) ∧ y = sqrt 2 - 1 ∧ x + 5 * y = 4 * sqrt 5 - 6) := 
sorry

end xy_range_xy_sum_range_x_2y_min_value_x_5y_not_min_value_l126_126449


namespace arithmetic_sequence_formula_and_sum_l126_126280

theorem arithmetic_sequence_formula_and_sum
  (a n : ℕ)  
  (S₅ : ℤ) 
  (a₅ : ℤ)
  (h1 : a₅ = -1)
  (h2 : S₅ = 15)
  : 
  (∀n, ∃aₙ, aₙ = -2 * n + 9) ∧ (∀n, ∃Sₙ, Sₙ = -n^2 + 8 * n ∧ ∀m, Sₙ ≤ 16) :=
by
  -- This statement asserts that there exists a general formula for the arithmetic sequence,
  -- and the sum of the first n terms follows a specific formula, with a maximum value of 16.
  sorry

end arithmetic_sequence_formula_and_sum_l126_126280


namespace smallest_integer_solution_l126_126452

theorem smallest_integer_solution (n : ℕ) (h : n ≥ 3) : ∃ x : ℕ, x = 2 := by
  have initial_set := (1 to n) -- hypothetical representation of initial set
  have operation : (ℕ × ℕ) → ℕ := λ ⟨a, b⟩, (a + b) / 2
  have process : ℕ → ℕ := sorry

  -- The proof that the result is always 2 after all operations
  sorry
  
#eval smallest_integer_solution 3 (by norm_num)

end smallest_integer_solution_l126_126452


namespace coefficient_x4_term_l126_126075

def polynomial : Polynomial ℚ := (X - 1) * (X - 2) * (X - 3) * (X - 4) * (X - 5)

theorem coefficient_x4_term:
  polynomial.coeff 4 = -15 :=
sorry

end coefficient_x4_term_l126_126075


namespace reflection_matrix_over_vector_l126_126798

/-- The matrix that reflects any vector over the vector ⟨4, 1⟩ is 
(⟨15/17, 8/17⟩, ⟨8/17, -15/17⟩).-/
theorem reflection_matrix_over_vector :
  ∃ (M : Matrix (fin 2) (fin 2) ℚ), 
  (∀ (v : Vector2), reflect v ⟨4, 1⟩ = M.mul_vec v) ∧
  M = ⟨⟨15 / 17, 8 / 17⟩, ⟨8 / 17, -15 / 17⟩⟩ :=
by
  sorry

end reflection_matrix_over_vector_l126_126798


namespace samuel_faster_than_sarah_l126_126958

-- Definitions based on the conditions
def time_samuel : ℝ := 30
def time_sarah : ℝ := 1.3 * 60

-- The theorem to prove that Samuel finished his homework 48 minutes faster than Sarah
theorem samuel_faster_than_sarah : (time_sarah - time_samuel) = 48 := by
  sorry

end samuel_faster_than_sarah_l126_126958


namespace received_for_jacket_l126_126961

theorem received_for_jacket (shorts_price shirt_price net_spent : ℝ) 
  (h1 : shorts_price = 13.99) 
  (h2 : shirt_price = 12.14) 
  (h3 : net_spent = 18.7) 
  : shorts_price + shirt_price - net_spent = 7.43 := by
  rw [h1, h2, h3]
  norm_num
  sorry

end received_for_jacket_l126_126961


namespace number_of_rooms_l126_126710

theorem number_of_rooms (total_outlets : ℕ) (outlets_per_room : ℕ) (h : total_outlets = 42) (h_room : outlets_per_room = 6) : (total_outlets / outlets_per_room) = 7 := 
by 
  rw [h, h_room]
  exact Nat.div_eq_of_eq_mul (by norm_num)

end number_of_rooms_l126_126710


namespace find_ages_l126_126035

theorem find_ages (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 := 
sorry

end find_ages_l126_126035


namespace find_cartesian_curve_l126_126558

section
  -- Define a line l passing through the point P(2, -1) with an inclination angle of 45 degrees
  def is_line (l : ℝ → ℝ × ℝ) :=
    ∃ t, (l t).1 = 2 + (Real.sqrt 2 / 2) * t ∧ (l t).2 = -1 + (Real.sqrt 2 / 2) * t

  -- Define the polar equation of the curve C
  def is_polar_curve (f : ℝ → ℝ → Prop) :=
    ∀ (ρ θ : ℝ), f ρ θ ↔ ρ * (Real.sin θ) ^ 2 = 4 * Real.cos θ

  -- Convert the polar equation to Cartesian coordinates
  def is_cartesian_curve (g : ℝ × ℝ → Prop) :=
    ∀ (x y : ℝ), g x y ↔ y^2 = 4 * x

  -- Conditions
  variable (P : ℝ × ℝ := (2, -1)) (θ : ℝ := Real.pi / 4)
  variable (line : ℝ → ℝ × ℝ)
  variable (polar_curve : ℝ → ℝ → Prop)
  variable (cartesian_curve : ℝ × ℝ → Prop)

  -- Proof problem
  theorem find_cartesian_curve :
    is_polar_curve polar_curve →
    is_cartesian_curve cartesian_curve →
    (P = (2, -1)) →
    θ = Real.pi / 4 →
    (∀ t, line t = (2 + (Real.sqrt 2 / 2) * t, -1 + (Real.sqrt 2 / 2) * t)) →
    ∃ A B : ℝ × ℝ,
      cartesian_curve A ∧ cartesian_curve B ∧
      (fst A = fst B) ∧
      |(2 - fst A | * | (2 - fst B)| = 14 :=
    sorry
end

end find_cartesian_curve_l126_126558


namespace chairs_per_row_l126_126441

theorem chairs_per_row (rows chairs_empty chairs_taken : ℕ)
  (hrows : rows = 40)
  (hempty : chairs_empty = 10)
  (htaken : chairs_taken = 790) :
  rows * 20 = chairs_empty + chairs_taken :=
by {
    rw [hrows, hempty, htaken],
    exact rfl,
}

end chairs_per_row_l126_126441


namespace sector_area_correct_l126_126150

-- Defining the chord length condition
def chord_length (θ r : ℝ) : ℝ := 2 * r * sin (θ / 2)

-- Given that the chord length is 2 when θ is 2 radians
def given_chord_length : Prop := chord_length 2 r = 2

-- Area of the sector
def sector_area (r θ : ℝ) : ℝ := 0.5 * r * θ * r

theorem sector_area_correct :
  given_chord_length →
  sector_area (1 / sin 1) 2 = 1 / (sin 1 ^ 2) :=
sorry

end sector_area_correct_l126_126150


namespace Tamika_greater_than_Carlos_probability_l126_126971

-- Definitions for the sets involved
def Tamika_set : set ℕ := {7, 10, 12}
def Carlos_set : set ℤ := {-4, 2, 6}

-- Calculate all possible sums for Tamika
def Tamika_sums : set ℕ := {a + b | a b : ℕ, a ∈ Tamika_set ∧ b ∈ Tamika_set ∧ a ≠ b}

-- Calculate all possible products for Carlos
def Carlos_products : set ℤ := {a * b | a b : ℤ, a ∈ Carlos_set ∧ b ∈ Carlos_set ∧ a ≠ b}

-- Proof statement
theorem Tamika_greater_than_Carlos_probability :
  let pairs := { (t, c) | t c : ℤ, t ∈ Tamika_sums ∧ c ∈ Carlos_products } in
  let favorable_pairs := {p ∈ pairs | p.1 > p.2} in
  (card favorable_pairs) / (card pairs) = 8/9 :=
sorry

end Tamika_greater_than_Carlos_probability_l126_126971


namespace percentage_equivalence_l126_126526

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end percentage_equivalence_l126_126526


namespace coeff_x_expansion_l126_126907

noncomputable def poly : ℕ → ℕ := 
  λ n, ((x^2 + 3*x + 2)^n).coeff 1

theorem coeff_x_expansion :
  poly 5 = 240 :=
by sorry

end coeff_x_expansion_l126_126907


namespace trees_left_after_typhoon_l126_126505

variable (initial_trees : ℕ)
variable (died_trees : ℕ)
variable (remaining_trees : ℕ)

theorem trees_left_after_typhoon :
  initial_trees = 20 →
  died_trees = 16 →
  remaining_trees = initial_trees - died_trees →
  remaining_trees = 4 :=
by
  intros h_initial h_died h_remaining
  rw [h_initial, h_died] at h_remaining
  exact h_remaining

end trees_left_after_typhoon_l126_126505


namespace sum_of_vectors_eq_zero_l126_126587

-- Definition of points and midpoints in a triangle
variables (A B C D E F : Type)
variables [AddGroup A] [AddGroup B] [AddGroup C]
variables [HasAdd A] [HasAdd B] [HasAdd C]
variables [HasSub A] [HasSub B] [HasSub C]

-- Assuming D, E, and F are the midpoints of sides BC, CA, and AB.
variables (midpoint_D : D = midpoint B C)
          (midpoint_E : E = midpoint C A)
          (midpoint_F : F = midpoint A B)

-- The theorem to prove
theorem sum_of_vectors_eq_zero 
  (DA EB FC : A)
  (hDA : DA = ⟨A, D⟩)
  (hEB : EB = ⟨B, E⟩)
  (hFC : FC = ⟨C, F⟩) : 
  DA + EB + FC = 0 :=
sorry

end sum_of_vectors_eq_zero_l126_126587


namespace percentage_increase_l126_126598

-- Definitions based on the problem conditions:
def X_rate : ℝ := 5.999999999999999
def B_time : ℝ := 100
def X_time : ℝ := B_time + 10
def B_rate : ℝ := 660 / B_time

-- Theorem to prove the percentage increase in production rate.
theorem percentage_increase (X_rate X_time B_rate : ℝ) :
  (B_rate - X_rate) / X_rate * 100 = 10 := by
  sorry

end percentage_increase_l126_126598


namespace fraction_area_above_line_l126_126717

-- Define the problem conditions
def point1 : ℝ × ℝ := (4, 1)
def point2 : ℝ × ℝ := (9, 5)
def vertex1 : ℝ × ℝ := (4, 0)
def vertex2 : ℝ × ℝ := (9, 0)
def vertex3 : ℝ × ℝ := (9, 5)
def vertex4 : ℝ × ℝ := (4, 5)

-- Define the theorem statement
theorem fraction_area_above_line :
  let area_square := 25
  let area_below_line := 2.5
  let area_above_line := area_square - area_below_line
  area_above_line / area_square = 9 / 10 :=
by
  sorry -- Proof omitted

end fraction_area_above_line_l126_126717


namespace money_distribution_l126_126146

variables (B e j d : ℝ)

-- Conditions
def total_money := 120
def josh_double_brad := j = 2 * B
def josh_three_fourths_doug := j = (3 / 4) * d
def brad_half_emily := B = (1 / 2) * e

-- Proof statement
theorem money_distribution 
  (total : B + e + j + d = total_money)
  (cond1 : josh_double_brad)
  (cond2 : josh_three_fourths_doug)
  (cond3 : brad_half_emily) :
  B ≈ 15.65 ∧ e ≈ 31.30 ∧ j ≈ 31.30 ∧ d ≈ 41.73 :=
sorry

end money_distribution_l126_126146


namespace inequality_8xyz_l126_126239

theorem inequality_8xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) > 8 * x * y * z := 
  by sorry

end inequality_8xyz_l126_126239


namespace coordinates_of_point_A_l126_126902

theorem coordinates_of_point_A (A B : ℝ × ℝ) (hAB : B.1 = 2 ∧ B.2 = 4) (hParallel : A.2 = B.2) (hDist : abs (A.1 - B.1) = 3) :
  A = (5, 4) ∨ A = (-1, 4) :=
by
  cases hAB with hx hy
  rw [hx, hy] at *
  cases hParallel
  rw [hParallel] at hDist
  cases abs_eq (A.1 - 2) (A.1 - 2) with ha ha
  case h1 =>
    rw [add_comm, add_right_eq_self, sub_eq_iff_eq_add] at ha
    left
    exact ⟨hx.symm ▸ ha, hParallel⟩
  case h2 =>
    rw [add_neg_cancel_right, add_eq_zero_iff_eq_neg_eq] at ha
    right
    exact ⟨hx.symm ▸ ha, hParallel⟩

end coordinates_of_point_A_l126_126902


namespace average_student_age_before_leaving_l126_126707

theorem average_student_age_before_leaving
  (A : ℕ)
  (student_count : ℕ := 30)
  (leaving_student_age : ℕ := 11)
  (teacher_age : ℕ := 41)
  (new_avg_age : ℕ := 11)
  (new_total_students : ℕ := 30)
  (initial_total_age : ℕ := 30 * A)
  (remaining_students : ℕ := 29)
  (total_age_after_leaving : ℕ := initial_total_age - leaving_student_age)
  (total_age_including_teacher : ℕ := total_age_after_leaving + teacher_age) :
  total_age_including_teacher / new_total_students = new_avg_age → A = 10 := 
  by
    intros h
    sorry

end average_student_age_before_leaving_l126_126707


namespace longer_leg_third_triangle_l126_126776

-- Given constants
def hypotenuse_triangle1 : ℝ := 12
def longer_leg_triangle1 : ℝ := hypotenuse_triangle1 / 2 * Real.sqrt 3
def hypotenuse_triangle2 : ℝ := longer_leg_triangle1
def longer_leg_triangle2 : ℝ := hypotenuse_triangle2 / 2 * Real.sqrt 3
def hypotenuse_triangle3 : ℝ := longer_leg_triangle2
def shorter_leg_triangle3 : ℝ := hypotenuse_triangle3 / 2
def longer_leg_triangle3 : ℝ := shorter_leg_triangle3 * Real.sqrt 3

-- Theorem statement
theorem longer_leg_third_triangle (hyp1 : hypotenuse_triangle1 = 12) :
  longer_leg_triangle3 = 4.5 * Real.sqrt 3 :=
sorry

end longer_leg_third_triangle_l126_126776


namespace cost_of_3600_pens_l126_126724

theorem cost_of_3600_pens
  (pack_size : ℕ)
  (pack_cost : ℝ)
  (n_pens : ℕ)
  (pen_cost : ℝ)
  (total_cost : ℝ)
  (h1: pack_size = 150)
  (h2: pack_cost = 45)
  (h3: n_pens = 3600)
  (h4: pen_cost = pack_cost / pack_size)
  (h5: total_cost = n_pens * pen_cost) :
  total_cost = 1080 :=
sorry

end cost_of_3600_pens_l126_126724


namespace abs_x_minus_one_sufficient_not_necessary_l126_126523

variable (x : ℝ) -- x is a real number

theorem abs_x_minus_one_sufficient_not_necessary (h : |x - 1| > 2) :
  (x^2 > 1) ∧ (∃ (y : ℝ), x^2 > 1 ∧ |y - 1| ≤ 2) := by
  sorry

end abs_x_minus_one_sufficient_not_necessary_l126_126523


namespace q_and_r_are_negations_l126_126886

variables (p q r : Prop) (m n : Prop)

/- Assume proposition p is: If m, then n -/
def prop_p : Prop := m → n

/- Define q as the inverse of p -/
def prop_q : Prop := n → m

/- Define r as the contrapositive of p -/
def prop_r : Prop := ¬n → ¬m

/- State the theorem that q and r are negation propositions of each other -/
theorem q_and_r_are_negations (h_p : prop_p p m n) (h_q : prop_q q n m) (h_r : prop_r r ¬n ¬m) : 
  (n → m) ↔ (¬n → ¬m) :=
sorry

end q_and_r_are_negations_l126_126886


namespace find_polynomial_solution_l126_126079

theorem find_polynomial_solution (p : ℝ → ℝ) (h : ∀ x, p(p(x)) = 2*x*p(x) + x^2) :
  p = λ x, (1 + Real.sqrt 2) * x ∨ p = λ x, (1 - Real.sqrt 2) * x :=
sorry

end find_polynomial_solution_l126_126079


namespace min_positive_period_increasing_intervals_max_min_values_l126_126115

noncomputable def f (x : ℝ) : ℝ := 3 * sin (x / 2 + π / 6) + 3

theorem min_positive_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  sorry

theorem increasing_intervals (k : ℤ) :
  ∀ x1 x2 : ℝ, -4 * π / 3 + 4 * k * π ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2 * π / 3 + 4 * k * π →
               f x1 ≤ f x2 :=
  sorry

theorem max_min_values :
  (∀ x : ℝ, π / 3 ≤ x ∧ x ≤ 4 * π / 3 → f x ≤ 6) ∧ (f (2 * π / 3) = 6) ∧ 
  (∀ x : ℝ, π / 3 ≤ x ∧ x ≤ 4 * π / 3 → f (4 * π / 3) ≤ f x) ∧ (f (4 * π / 3) = 9 / 2) :=
  sorry

end min_positive_period_increasing_intervals_max_min_values_l126_126115


namespace geo_seq_fifth_term_l126_126661

theorem geo_seq_fifth_term (a r : ℝ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h3 : a * r^2 = 8) (h7 : a * r^6 = 18) : a * r^4 = 12 :=
sorry

end geo_seq_fifth_term_l126_126661


namespace smallest_pos_int_for_Tn_is_integer_l126_126585

-- Define T_n as the sum of the reciprocals of all unique digits appearing in the binary representations of numbers from 1 to 2^n
def T (n : ℕ) : ℚ :=
  (by sorry : ℚ) -- This is placeholder code until the definition of T is provided.

-- The proof statement to verify the smallest integer n where T_n is an integer.
theorem smallest_pos_int_for_Tn_is_integer : ∀ n : ℕ, n = 1 → (↑ (T n)).den = 1 :=
by
  sorry -- Placeholder for the proof

end smallest_pos_int_for_Tn_is_integer_l126_126585


namespace sum_quotient_dividend_divisor_l126_126148

theorem sum_quotient_dividend_divisor (D : ℕ) (d : ℕ) (Q : ℕ) 
  (h1 : D = 54) (h2 : d = 9) (h3 : D = Q * d) : 
  (Q + D + d) = 69 :=
by
  sorry

end sum_quotient_dividend_divisor_l126_126148


namespace max_fruit_to_teacher_l126_126340

theorem max_fruit_to_teacher (A G : ℕ) : (A % 7 ≤ 6) ∧ (G % 7 ≤ 6) :=
by
  sorry

end max_fruit_to_teacher_l126_126340


namespace tan_sum_identity_l126_126184

theorem tan_sum_identity
  {α β γ : ℝ} 
  (hαβγ_acute : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ 0 < γ ∧ γ < π / 2)
  (h_sum : α + β + γ = π / 2) :
  tan α * tan β + tan α * tan γ + tan β * tan γ = 1 :=
sorry

end tan_sum_identity_l126_126184


namespace right_angled_triangle_ratio_3_4_5_l126_126689

theorem right_angled_triangle_ratio_3_4_5 : 
  ∀ (a b c : ℕ), 
  (a = 3 * d) → (b = 4 * d) → (c = 5 * d) → (a^2 + b^2 = c^2) :=
by
  intros a b c h1 h2 h3
  sorry

end right_angled_triangle_ratio_3_4_5_l126_126689


namespace sum_of_squares_inequality_l126_126457

theorem sum_of_squares_inequality 
  (n : ℕ) 
  (h : n ≥ 2) 
  (x : Fin n → ℝ) 
  (E : list (Fin n × Fin n))
  (S : ℝ) 
  (hS : S = ∑ e in E, (x e.fst) * (x e.snd)) 
  (tree : is_tree E) : 
  sqrt (n - 1) * (∑ i, (x i) ^ 2) ≥ 2 * S := 
sorry

end sum_of_squares_inequality_l126_126457


namespace profit_value_l126_126221

variable (P : ℝ) -- Total profit made by the business in that year.
variable (MaryInvestment : ℝ) -- Mary's investment
variable (MikeInvestment : ℝ) -- Mike's investment
variable (MaryExtra : ℝ) -- Extra money received by Mary

-- Conditions
axiom mary_investment : MaryInvestment = 900
axiom mike_investment : MikeInvestment = 100
axiom mary_received_more : MaryExtra = 1600
axiom profit_shared_equally : (P / 3) / 2 + (MaryInvestment / (MaryInvestment + MikeInvestment)) * (2 * P / 3) 
                           = MikeInvestment / (MaryInvestment + MikeInvestment) * (2 * P / 3) + MaryExtra

-- Statement
theorem profit_value : P = 4000 :=
by
  sorry

end profit_value_l126_126221


namespace pens_cost_l126_126721

theorem pens_cost (pens_pack_cost : ℝ) (pens_pack_quantity : ℕ) (total_pens : ℕ) (unit_price : ℝ) (total_cost : ℝ)
  (h1 : pens_pack_cost = 45) (h2 : pens_pack_quantity = 150) (h3 : total_pens = 3600) (h4 : unit_price = pens_pack_cost / pens_pack_quantity)
  (h5 : total_cost = total_pens * unit_price) : total_cost = 1080 := by
  sorry

end pens_cost_l126_126721


namespace evaluate_f_of_5_l126_126485

def f : ℤ → ℤ
| x := if x ≥ 10 then x - 2 else f (f (x + 6))

theorem evaluate_f_of_5 : f 5 = 11 := 
sorry

end evaluate_f_of_5_l126_126485


namespace find_range_m_l126_126470

/- 
  Proposition (p): The equation (x^2 / (2 * m) - y^2 / (m - 1) = 1) represents an ellipse with foci on the y-axis.
  Proposition (q): The eccentricity (e) of the hyperbola (y^2 / 5 - x^2 / m = 1) is in the interval (1, 2).
-/

def proposition_p (m : ℝ) : Prop := ∃ x y, (x^2 / (2 * m) - y^2 / (m - 1) = 1) ∧ (1 - m > 2 * m) ∧ (2 * m > 0)

def proposition_q (m : ℝ) : Prop := ∃ e, (y^2 / 5 - x^2 / m = 1) ∧ (1 < e) ∧ (e < 2)

theorem find_range_m (m : ℝ) (hp_false : ¬ proposition_p m) (hq_false : ¬ proposition_q m) (h_condition : proposition_p m ∨ proposition_q m) : (1/3 ≤ m) ∧ (m < 15) := 
by sorry

end find_range_m_l126_126470


namespace bike_owners_without_scooter_l126_126164

open Set

variables (U : Type) [Fintype U]
variable (adults : U → Prop)
variable (owns_bike : U → Prop)
variable (owns_scooter : U → Prop)

noncomputable def adults_count : ℕ := Fintype.card (Subtype adults)
noncomputable def bikes_count : ℕ := Fintype.card { x : U // owns_bike x }
noncomputable def scooters_count : ℕ := Fintype.card { x : U // owns_scooter x }
noncomputable def both_count : ℕ := Fintype.card { x : U // owns_bike x ∧ owns_scooter x }

theorem bike_owners_without_scooter (h1 : adults_count adults = 450)
    (h2 : bikes_count owns_bike = 400)
    (h3 : scooters_count owns_scooter = 120)
    (h4 : ∀ x, adults x ↔ owns_bike x ∨ owns_scooter x) :
    bikes_count owns_bike - both_count owns_bike owns_scooter = 330 := by
  sorry

end bike_owners_without_scooter_l126_126164


namespace line_intersects_circle_l126_126848

theorem line_intersects_circle (r d : ℝ) (hr : r = 5) (hd : d = 3 * Real.sqrt 2) : d < r :=
by
  rw [hr, hd]
  exact sorry

end line_intersects_circle_l126_126848


namespace packages_eq_nine_l126_126934

-- Definitions of the given conditions
def x : ℕ := 50
def y : ℕ := 5
def z : ℕ := 5

-- Statement: Prove that the number of packages Amy could make equals 9
theorem packages_eq_nine : (x - y) / z = 9 :=
by
  sorry

end packages_eq_nine_l126_126934


namespace original_price_per_lesson_l126_126575

theorem original_price_per_lesson (piano_cost lessons_cost : ℤ) (number_of_lessons discount_percent : ℚ) (total_cost : ℤ) (original_price : ℚ) :
  piano_cost = 500 ∧
  number_of_lessons = 20 ∧
  discount_percent = 0.25 ∧
  total_cost = 1100 →
  lessons_cost = total_cost - piano_cost →
  0.75 * (number_of_lessons * original_price) = lessons_cost →
  original_price = 40 :=
by
  intros h h1 h2
  sorry

end original_price_per_lesson_l126_126575


namespace slope_angle_of_line_parallel_to_x_axis_is_zero_l126_126276

theorem slope_angle_of_line_parallel_to_x_axis_is_zero (y : ℝ) (h : y = 1) : 
  ∀  line : ℝ → ℝ, (∀ x : ℝ, line x = y) → 
  let slope_angle := 0 
  in slope_angle = 0 :=
by
  intro line hin
  let slope_angle := 0 
  sorry

end slope_angle_of_line_parallel_to_x_axis_is_zero_l126_126276


namespace isosceles_triangle_angle_l126_126375

theorem isosceles_triangle_angle (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) 
  (h_iso : a = b ∨ b = c ∨ c = a) (h_A : A = 70) (h_sum : A + B + C = 180) : 
  B ∈ {55, 70, 40} :=
sorry

end isosceles_triangle_angle_l126_126375


namespace proposition_incorrect_l126_126688

theorem proposition_incorrect :
  ¬(∀ x : ℝ, x^2 + 3 * x + 1 > 0) :=
by
  sorry

end proposition_incorrect_l126_126688


namespace concyclicity_equivalence_l126_126841

-- Define the quadrilateral and its properties
variables {A B C D P E F G H : Point}
variables {O1 O2 O3 O4 : CircleCenter}
variable [ConvexQuadrilateral A B C D]

-- Conditions
variables (h1 : IntersectDiagonals A C B D P)
variables (h2 : Midpoint E A B)
variables (h3 : Midpoint F B C)
variables (h4 : Midpoint G C D)
variables (h5 : Midpoint H D A)
variables (h6 : Circumcenter O1 P H E)
variables (h7 : Circumcenter O2 P E F)
variables (h8 : Circumcenter O3 P F G)
variables (h9 : Circumcenter O4 P G H)

-- Proof statement
theorem concyclicity_equivalence :
  (Concyclic O1 O2 O3 O4) ↔ (Concyclic A B C D) :=
sorry

end concyclicity_equivalence_l126_126841


namespace pascal_probability_one_l126_126738

theorem pascal_probability_one : 
  let total_elements := (20 * 21) / 2 in
  let ones_count := 1 + 2 * 19 in
  total_elements = 210 ∧ ones_count = 39 ∧ 
  (ones_count / total_elements : ℚ) = 13 / 70 := 
by
  sorry

end pascal_probability_one_l126_126738


namespace square_side_length_l126_126311

theorem square_side_length (π : ℝ) (s : ℝ) :
  (∃ r : ℝ, 100 = π * r^2) ∧ (4 * s = 100) → s = 25 := by
  sorry

end square_side_length_l126_126311


namespace factorize_expression_l126_126785

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end factorize_expression_l126_126785


namespace contestant_in_top_8_knows_score_and_median_l126_126175

-- Definition of the problem in Lean 4
theorem contestant_in_top_8_knows_score_and_median (scores : List ℕ) (contestant_score : ℕ) :
  (List.length scores = 15) →
  (∃ med : ℕ, (median scores = med) ∧ (contestant_score > med → contestant_is_in_top_8)) :=
by
  sorry

end contestant_in_top_8_knows_score_and_median_l126_126175


namespace relationship_between_exponents_l126_126519

variable {M P T Q : Type}
variable {m p t q : M}
variable {n r s u : ℕ}
variable eq1 : ((m ^ n) ^ 2 = p ^ r) ∧ (p ^ r = t)
variable eq2 : ((m ^ u) ^ 3 = p ^ s) ∧ (p ^ s = q)

theorem relationship_between_exponents :
  3 * u * r = 2 * n * s :=
by
  sorry

end relationship_between_exponents_l126_126519


namespace find_S100_l126_126093

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a(n + 1) = a n + d

def sequence_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 1 / (a n ^ 2 - 1)

def sum_sequence (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), b i

theorem find_S100 :
  ∃ a d (b S : ℕ → ℝ),
    a 2 = 7 ∧
    a 4 + a 6 = 26 ∧
    arithmetic_sequence a d ∧
    sequence_b a b ∧
    sum_sequence b S ∧
    S 100 = 25 / 101 := by
  sorry

end find_S100_l126_126093


namespace log_base_change_l126_126069

-- Define the conditions: 8192 = 2 ^ 13 and change of base formula
def x : ℕ := 8192
def a : ℕ := 2
def n : ℕ := 13
def b : ℕ := 5

theorem log_base_change (log : ℕ → ℕ → ℝ) 
  (h1 : x = a ^ n) 
  (h2 : ∀ (x b c: ℕ), c ≠ 1 → log x b = (log x c) / (log b c) ): 
  log x b = 13 / (log 5 2) :=
by
  sorry

end log_base_change_l126_126069


namespace min_positive_announcements_l126_126378

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 90)
  (h2 : y * (y - 1) + (x - y) * (x - y - 1) = 48) 
  : y = 3 :=
sorry

end min_positive_announcements_l126_126378


namespace factorize_expr_l126_126784

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end factorize_expr_l126_126784


namespace samuel_faster_l126_126960

theorem samuel_faster (S T_h : ℝ) (hT_h : T_h = 1.3) (hS : S = 30) :
  (T_h * 60) - S = 48 :=
by
  sorry

end samuel_faster_l126_126960


namespace num_female_officers_l126_126608

/-- Given:
- 56% of the male officers and 32% of the female officers were on duty.
- 280 police officers were on duty that night.
- 40% of the total police force were female officers.
Prove: The number of female officers is 241.
-/
theorem num_female_officers (P : ℝ) (num_male_on_duty : ℝ) (num_female_on_duty : ℝ) (total_on_duty : ℝ) (percent_female_officers : ℝ) 
    (h1 : num_male_on_duty = 0.56 * 0.6 * P) 
    (h2 : num_female_on_duty = 0.32 * 0.4 * P) 
    (h3 : total_on_duty = num_male_on_duty + num_female_on_duty)
    (h4 : total_on_duty = 280)
    (h5 : percent_female_officers = 0.4) : 
  0.4 * P ≈ 241 := 
by 
  sorry

end num_female_officers_l126_126608


namespace angle_same_terminal_side_l126_126974

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, -\frac{7} {4} * Real.pi = \frac {1} {4} * Real.pi + 2 * k * Real.pi := sorry

end angle_same_terminal_side_l126_126974


namespace hired_is_B_l126_126709

-- Define the individuals
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

open Person

-- Define the statements made by each person
def statement (p : Person) (hired : Person) : Prop :=
  match p with
  | A => hired = C
  | B => hired ≠ B
  | C => hired = D
  | D => hired ≠ D

-- The main theorem is to prove B is hired given the conditions
theorem hired_is_B :
  (∃! p : Person, ∃ t : Person → Prop,
    (∀ h : Person, t h ↔ h = p) ∧
    (∃ q : Person, statement q q ∧ ∀ r : Person, r ≠ q → ¬statement r q) ∧
    t B) :=
by
  sorry

end hired_is_B_l126_126709


namespace parallel_lines_slope_l126_126095

theorem parallel_lines_slope (a : ℝ) (a_ne_zero : a ≠ 0) (par: ∀ (l1 l2 : ℝ) , l1 = l2 := by intro; exact (a/2) = (1/(a + 1))) : a = 1 := by sorry

end parallel_lines_slope_l126_126095


namespace red_flowers_needed_l126_126649

-- Define the number of white and red flowers
def white_flowers : ℕ := 555
def red_flowers : ℕ := 347

-- Define the problem statement.
theorem red_flowers_needed : red_flowers + 208 = white_flowers := by
  -- The proof goes here.
  sorry

end red_flowers_needed_l126_126649


namespace axis_of_symmetry_y1_less_than_y2_value_of_a_range_of_a_l126_126182

variables {a c y1 y2 y3: ℝ}
variables {m: ℕ}

-- Definitions for the points A, B, C on the parabola and given conditions
def parabola (x : ℝ) : ℝ :=
  -x^2 + 2 * a * x + c

def point_A := (-3, y1)
def point_B := (a / 2, y2)
def point_C (m: ℝ) := (m, y3)

-- Conditions provided in the problem
axiom A_on_parabola : parabola (-3) = y1
axiom B_on_parabola : parabola (a / 2) = y2
axiom C_on_parabola : parabola m = y3
axiom cond_1 : a > 0
axiom cond_2 : m = 4 → y1 = y3
axiom cond_3 : (1 ≤ m ∧ m ≤ 4) → y1 < y3 ∧ y3 < y2

-- Theorem statements to be proven
theorem axis_of_symmetry : ∀ (a : ℝ), axis_of_symmetry (parabola x) = a := sorry
theorem y1_less_than_y2 : y1 < y2 := sorry
theorem value_of_a : (m = 4) → y1 = y3 → a = 1/2 := sorry
theorem range_of_a : ((1 / 2 < a ∧ a < 2 / 3) ∨ (8 < a)) := sorry

end axis_of_symmetry_y1_less_than_y2_value_of_a_range_of_a_l126_126182


namespace train_passes_man_in_time_l126_126731

def length_of_train : ℝ := 605
def speed_of_train_kmh : ℝ := 60
def speed_of_man_kmh : ℝ := 6
def kmh_to_mps (kmh : ℝ) : ℝ := kmh * 1000 / 3600

def relative_speed_mps : ℝ := kmh_to_mps (speed_of_train_kmh + speed_of_man_kmh)
def time_to_pass (distance speed : ℝ) : ℝ := distance / speed

theorem train_passes_man_in_time :
  abs (time_to_pass length_of_train relative_speed_mps - 33.01) < 0.01 :=
by
  sorry

end train_passes_man_in_time_l126_126731


namespace probability_at_least_5_stayed_l126_126618

theorem probability_at_least_5_stayed (prob_unsure : ℕ → ℚ)
    (prob_unsure 0 = 1/3)
    (prob_unsure 1 = 1/3)
    (prob_unsure 2 = 1/3)
    (prob_unsure 3 = 1/3)
    (prob_sure : ℕ → ℚ)
    (prob_sure 4 = 1)
    (prob_sure 5 = 1)
    (prob_sure 6 = 1)
    (prob_sure 7 = 1) :
    ∑ (n : ℕ) in (finset.range 8).filter (λ n, n ≥ 5), prob_unsure n = 32 / 81 := by
  sorry

end probability_at_least_5_stayed_l126_126618


namespace find_f_of_2_l126_126855

theorem find_f_of_2 : (∀ x : ℝ, f (2 * x) = 4 * x - 1) → f 2 = 3 :=
by
  intro h
  specialize h 1
  rw [mul_one] at h
  exact h

end find_f_of_2_l126_126855


namespace box_volume_increase_l126_126359

-- Conditions
def volume (l w h : ℝ) : ℝ := l * w * h
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def sum_of_edges (l w h : ℝ) : ℝ := 4 * (l + w + h)

-- The main theorem we want to state
theorem box_volume_increase
  (l w h : ℝ)
  (h_volume : volume l w h = 5000)
  (h_surface_area : surface_area l w h = 1800)
  (h_sum_of_edges : sum_of_edges l w h = 210) :
  volume (l + 2) (w + 2) (h + 2) = 7018 := 
by sorry

end box_volume_increase_l126_126359


namespace find_income_day_4_l126_126341

-- Define the incomes on the five days
def income_day_1 := 400
def income_day_2 := 250
def income_day_3 := 650
-- income_day_4 is unknown, we will define it as a variable
variable income_day_4 : ℕ
def income_day_5 := 500

-- Define the average income
def average_income := 440

-- Define the total income computed from the average
def total_income := 5 * average_income

-- Define the known total income of the first, second, third, and fifth days
def known_income := income_day_1 + income_day_2 + income_day_3 + income_day_5

-- State the theorem
theorem find_income_day_4 : income_day_4 = total_income - known_income := by
  sorry

end find_income_day_4_l126_126341


namespace solve_inequality_l126_126967

theorem solve_inequality : { x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4 } = { x | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) } :=
by
  sorry

end solve_inequality_l126_126967


namespace xiao_li_hits_bullseye_14_times_l126_126296

theorem xiao_li_hits_bullseye_14_times
  (initial_rifle_bullets : ℕ := 10)
  (initial_pistol_bullets : ℕ := 14)
  (reward_per_bullseye_rifle : ℕ := 2)
  (reward_per_bullseye_pistol : ℕ := 4)
  (xiao_wang_bullseyes : ℕ := 30)
  (total_bullets : ℕ := initial_rifle_bullets + xiao_wang_bullseyes * reward_per_bullseye_rifle) :
  ∃ (xiao_li_bullseyes : ℕ), total_bullets = initial_pistol_bullets + xiao_li_bullseyes * reward_per_bullseye_pistol ∧ xiao_li_bullseyes = 14 :=
by sorry

end xiao_li_hits_bullseye_14_times_l126_126296


namespace domain_f_0_lt_a_lt_1_domain_f_a_gt_1_range_f_0_lt_a_lt_1_range_f_a_gt_1_l126_126486

-- Define the function f(x)
def f (a x : ℝ) : ℝ := Real.log (1 - a^x) / Real.log a

-- Conditions
variables (a x : ℝ)

-- Domain of the function f(x)
theorem domain_f_0_lt_a_lt_1 (ha : 0 < a ∧ a < 1) : { x : ℝ | 1 - a^x > 0 } = Set.Ioi 0 :=
by sorry

theorem domain_f_a_gt_1 (ha : a > 1) : { x : ℝ | 1 - a^x > 0 } = Set.Iio 0 :=
by sorry

-- Range of x satisfying log_a(1 - a^x) > f(1)
theorem range_f_0_lt_a_lt_1 (ha : 0 < a ∧ a < 1) : { x : ℝ | Real.log (1 - a^x) / Real.log a > Real.log (1 - a) / Real.log a } = Set.Ioo 0 1 :=
by sorry

theorem range_f_a_gt_1 (ha : a > 1) : { x : ℝ | Real.log (1 - a^x) / Real.log a > Real.log (1 - a) / Real.log a } = Set.Iio 0 :=
by sorry

end domain_f_0_lt_a_lt_1_domain_f_a_gt_1_range_f_0_lt_a_lt_1_range_f_a_gt_1_l126_126486


namespace volume_of_rectangular_prism_l126_126980

-- Define the dimensions a, b, c as non-negative real numbers
variables (a b c : ℝ)

-- Given conditions
def condition_1 := a * b = 30
def condition_2 := a * c = 50
def condition_3 := b * c = 75

-- The theorem statement
theorem volume_of_rectangular_prism :
  (a * b * c) = 335 :=
by
  -- Assume the given conditions
  assume h1 : condition_1 a b,
  assume h2 : condition_2 a c,
  assume h3 : condition_3 b c,
  -- Proof skipped
  sorry

end volume_of_rectangular_prism_l126_126980


namespace diagonals_perpendicular_triangle_BMC_right_l126_126332

variables {a b h : ℝ}

-- Proposition 1: Diagonals AC and BD are perpendicular iff h^2 = ab
theorem diagonals_perpendicular (A B C D : Type) [trapezoid ABCD] (AB_eq_a : AB = a) (CD_eq_b : CD = b) (AD_eq_h : AD = h) 
  (right_angle_A : ∠ A = 90) (right_angle_D : ∠ D = 90): 
  diagonals_perpendicular AC BD ↔ h^2 = ab := 
sorry

-- Proposition 2: Triangle BMC is right at M iff h^2 = 4ab
theorem triangle_BMC_right (A B C D M : Type) [trapezoid ABCD] (AB_eq_a : AB = a) (CD_eq_b : CD = b) (AD_eq_h : AD = h) 
  (midpoint_M : M = midpoint AD) (right_angle_A : ∠ A = 90) (right_angle_D : ∠ D = 90): 
  is_right_triangle BMC M ↔ h^2 = 4ab := 
sorry

end diagonals_perpendicular_triangle_BMC_right_l126_126332


namespace sufficient_steps_l126_126911

theorem sufficient_steps (ε : ℝ) (a : ℝ) : ε = 0.001 → a = 0.01 → 
  ∃ n : ℕ, n ≥ 4 * 10^6 ∧ ∀ l : ℕ, l ≤ n → (∣(2 * l / n) - 1∣ < a) → 1 - ε ≤ (1 - abs ((2 * l / n) - 1)) :=
by
  intro hε ha
  use 4 * 10^6
  intro l
  intro hl
  intro hln
  sorry

end sufficient_steps_l126_126911


namespace total_stones_is_odd_l126_126732

variable (d : ℕ) (total_distance : ℕ)

theorem total_stones_is_odd (h1 : d = 10) (h2 : total_distance = 4800) :
  ∃ (N : ℕ), N % 2 = 1 ∧ total_distance = ((N - 1) * 2 * d) :=
by
  -- Let's denote the number of stones as N
  -- Given dx = 10 and total distance as 4800, we want to show that N is odd and 
  -- satisfies the equation: total_distance = ((N - 1) * 2 * d)
  sorry

end total_stones_is_odd_l126_126732


namespace sunil_total_amount_l126_126628

-- Definitions based on conditions
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ := 
  P * (1 + r / n)^(n * t) - P

def total_amount (P : ℝ) (CI : ℝ) : ℝ := 
  P + CI

-- Given conditions
def principal (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  CI / ((1 + r / n)^(n * t) - 1)

def conditions_holds (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (A : ℝ) : Prop := 
  A = total_amount (principal CI r n t) CI

-- The proof statement
theorem sunil_total_amount :
  conditions_holds 326.40 0.04 1 2 4326.40 :=
sorry

end sunil_total_amount_l126_126628


namespace equilateral_triangle_distances_equal_l126_126816

variables {A B C M P Q R : Type} -- Define points as types

-- Define the conditions
variables [equilateral_triangle A B C] [inside_triangle M A B C]
variables (MP MQ MR : line) -- Perpendiculars from M to sides

-- Define lengths from M to sides and vertices
variables (AP BQ CR PB QC RA : ℝ) -- Distances

-- Define the perpendicular condition
variables (h₁ : MP ⊥ AB) (h₂ : MQ ⊥ BC) (h₃ : MR ⊥ CA)

-- State the proof problem
theorem equilateral_triangle_distances_equal :
  AP^2 + BQ^2 + CR^2 = PB^2 + QC^2 + RA^2 ∧ AP + BQ + CR = PB + QC + RA :=
by
  sorry

end equilateral_triangle_distances_equal_l126_126816


namespace triangle_area_l126_126025

theorem triangle_area (A B C : ℝ × ℝ)
  (hA : A = (3, -3))
  (hB : B = (3, 6))
  (hC : C = (8, 6)) :
  let base := (C.1 - B.1).abs,
      height := (B.2 - A.2).abs in
  (1/2) * base * height = 22.5 :=
by {
  sorry
}

end triangle_area_l126_126025


namespace friend_saves_per_week_l126_126318

theorem friend_saves_per_week (x : ℕ) : 
  160 + 7 * 25 = 210 + x * 25 → x = 5 := 
by 
  sorry

end friend_saves_per_week_l126_126318


namespace problem_l126_126447

variable {a b c : ℤ}

noncomputable def f (x : ℤ) : ℤ := a * x ^ 7 - b * x ^ 5 + c * x ^ 3 + 2

theorem problem (h : f (-5) = 17) : f 5 = -13 :=
by
  sorry

end problem_l126_126447


namespace Tanya_efficiency_l126_126956

theorem Tanya_efficiency : 
  ∃ (S : ℕ) (T : ℕ), S = 12 → (1 / T : ℝ) = 1.2 * (1 / S : ℝ) → T = 10 :=
begin
  sorry
end

end Tanya_efficiency_l126_126956


namespace relay_arrangements_l126_126244

theorem relay_arrangements 
  (students : Fin 10 → Prop)
  (A B : Fin 10 → Prop)
  (valid_arrangements : ∀ (team : Fin 4 → Fin 10), team 0 ≠ A ∧ team 3 ≠ B → Prop)
  : ∃ (total_arrangements : Nat), total_arrangements = 4008 :=
by
  sorry

end relay_arrangements_l126_126244


namespace prob_condition_sum_of_k_l126_126804

open Nat

def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then
    nat.choose n k
  else
    0

lemma pascal_identity :
  binom 25 6 + binom 25 7 = binom 26 7 :=
begin
  rw [binom, if_pos (nat.le_refl 25)],
  rw [binom, if_pos (nat.le_refl 25)],
  rw nat.choose_succ_succ,
  exact rfl,
end

theorem prob_condition (k : ℕ) :
  binom 25 6 + binom 25 7 = binom 26 k → k = 7 ∨ k = 19 :=
begin
  intro h,
  rw pascal_identity at h,
  have h_k : binom 26 k = binom 26 7 := h,
  rw nat.choose_eq_choose_symm at h_k,
  cases h_k,
  { exact or.inl h_k },
  { exact or.inr h_k }
end

theorem sum_of_k :
  ∑ k in {k ∈ Finset.range 27 | binom 25 6 + binom 25 7 = binom 26 k}, k = 26 :=
begin
  apply Finset.sum_eq_of_subset,
  { exact λ x hx, by_cases h:(binom 25 6 + binom 25 7 = binom 26 x) → x ∈ {7, 19},
      { exact h },
      { exfalso, simpa using h } },
  { exact λ x hx, or.inl hx }
end

end prob_condition_sum_of_k_l126_126804


namespace gain_percentage_of_watch_l126_126368

theorem gain_percentage_of_watch :
  let CP := 1076.923076923077
  let S1 := CP * 0.90
  let S2 := S1 + 140
  let gain_percentage := ((S2 - CP) / CP) * 100
  gain_percentage = 3 := by
  sorry

end gain_percentage_of_watch_l126_126368


namespace chocolate_bars_per_box_l126_126716

theorem chocolate_bars_per_box (total_chocolate_bars num_small_boxes : ℕ) (h1 : total_chocolate_bars = 300) (h2 : num_small_boxes = 15) : 
  total_chocolate_bars / num_small_boxes = 20 :=
by 
  sorry

end chocolate_bars_per_box_l126_126716


namespace more_oaks_than_willows_l126_126284

theorem more_oaks_than_willows (total_trees willows : ℕ) (h1 : total_trees = 83) (h2 : willows = 36) :
  (total_trees - willows) - willows = 11 :=
by
  sorry

end more_oaks_than_willows_l126_126284


namespace mary_shirt_fraction_l126_126940

theorem mary_shirt_fraction (f : ℝ) : 
  26 * (1 - f) + 36 - 36 / 3 = 37 → f = 1 / 2 :=
by
  sorry

end mary_shirt_fraction_l126_126940


namespace parallel_sides_of_quadrilateral_l126_126647

theorem parallel_sides_of_quadrilateral {A B C D M N M1 N1 : Type*} [AddGroup M] [AddGroup N] [Module ℝ M] [Module ℝ N] 
  (hM_midpoint : midpoint A C = M) (hN_midpoint : midpoint B D = N) (hMN_neq : M ≠ N)
  (hMMN1_intersections : lineThrough M N ∩ lineThrough A B = {M1}) (hNMN1_intersections : lineThrough M N ∩ lineThrough C D = {N1})
  (hMM1_eq_NN1 : dist M M1 = dist N N1) : parallel (lineThrough A D) (lineThrough B C) :=
sorry

end parallel_sides_of_quadrilateral_l126_126647


namespace geometric_sequence_sum_l126_126839

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_a1 : a 1 = 1)
  (h_sum : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end geometric_sequence_sum_l126_126839


namespace typing_orders_after_lunch_l126_126894

theorem typing_orders_after_lunch : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 10, 11} in
  ∑ k in (range 10), (choose 9 k) * (k + 2) = 3328 :=
by
  sorry

end typing_orders_after_lunch_l126_126894


namespace groupings_of_guides_and_tourists_l126_126665

theorem groupings_of_guides_and_tourists :
  let guides := 3
  let tourists := 8
  -- The number of different groupings where each guide has at least one tourist
  ∑ (partitions : Fin.tourists -> Fin.guides), (⧸ ∀ g : Fin.guides, ∃ t : Fin.tourists, partitions t = g) = 5796 :=
sorry

end groupings_of_guides_and_tourists_l126_126665


namespace algorithm_output_correct_l126_126372

-- Define the algorithm's operation as a function
noncomputable def final_output (i : ℕ) : ℕ :=
  if i < 10 then
    final_output (i + 2)
  else
    2 * i + 3

theorem algorithm_output_correct : final_output 1 = 25 :=
  sorry

end algorithm_output_correct_l126_126372


namespace palindrome_divisibility_probability_l126_126725

-- Definition of a five-digit palindrome
def isPalindrome (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ n = 10001 * a + 1010 * b + 100 * c + 10 * b + a

-- Definition of being divisible by 11
def divisibleBy11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Probability calculation for palindromes
theorem palindrome_divisibility_probability :
  ( @Probability.distribution {n // isPalindrome n} (λ n, divisibleBy11 n.val) ) = 1 / 50 :=
sorry

end palindrome_divisibility_probability_l126_126725


namespace symmetrical_circle_proof_l126_126642

open Real

-- Definition of the original circle equation
def original_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Defining the symmetrical circle equation to be proven
def symmetrical_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 5

theorem symmetrical_circle_proof :
  ∀ x y : ℝ, original_circle x y ↔ symmetrical_circle x y :=
by sorry

end symmetrical_circle_proof_l126_126642


namespace tank_capacity_l126_126322

theorem tank_capacity (C : ℝ) (h_leak : ∀ t, t = 6 -> C / 6 = C / t)
    (h_inlet : ∀ r, r = 240 -> r = 4 * 60)
    (h_net : ∀ t, t = 8 -> 240 - C / 6 = C / 8) :
    C = 5760 / 7 := 
by 
  sorry

end tank_capacity_l126_126322


namespace percentage_transform_l126_126530

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end percentage_transform_l126_126530


namespace coplanar_vectors_sum_bound_l126_126930

theorem coplanar_vectors_sum_bound :
  ∀ (n : ℕ) (v : Fin n → ℝ^2), (∀ i : Fin n, ‖v i‖ ≤ 1) → n = 1989 →
  ∃ (ε : Fin n → ℤ), (∀ i : Fin n, ε i = 1 ∨ ε i = -1) ∧  ‖(∑ i : Fin n, ε i * v i)‖ ≤ Real.sqrt 3 :=
by
  sorry

end coplanar_vectors_sum_bound_l126_126930


namespace value_of_a4_l126_126194

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := n^2 - 3 * n - 4

-- State the main proof problem.
theorem value_of_a4 : a_n 4 = 0 := by
  sorry

end value_of_a4_l126_126194


namespace number_of_solutions_l126_126209

noncomputable def g_n (n : ℕ) (x : ℝ) := (Real.sin x)^(2 * n) + (Real.cos x)^(2 * n)

theorem number_of_solutions : ∀ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi) -> 
  8 * g_n 3 x - 6 * g_n 2 x = 3 * g_n 1 x -> false :=
by sorry

end number_of_solutions_l126_126209


namespace triangle_area_l126_126174

-- Define the triangle and conditions
def right_triangle (A B C : Type) := ∃ (a b c : A), angle a = 90 ∧ (length a b = length a c) ∧ (length b c = 8 * real.sqrt 2)

-- Define the area calculation as a theorem to prove
theorem triangle_area (A B C : Type) [triangle : right_triangle A B C] : triangle_area A B C = 32 :=
sorry

end triangle_area_l126_126174


namespace area_of_rhombus_l126_126615

variable (EF GH : Type) -- Define the rhombus as a type

noncomputable def side_length (EFGH : Type) (perimeter : ℕ) := perimeter / 4

noncomputable def diagonal_half (diagonal : ℕ) := diagonal / 2

noncomputable def other_half_diagonal (side diagonal_half : ℕ) := 
  Real.sqrt (side ^ 2 - diagonal_half ^ 2)

theorem area_of_rhombus (perimeter : ℕ) (d1 : ℕ) 
  (h1 : perimeter = 80) 
  (h2 : d1 = 36) 
  (EFGH_rhombus : EFGH): 
  ∃ area : ℝ, area = 72 * Real.sqrt 19 :=
  by
    have s : ℕ := side_length EFGH perimeter
    have d1_half : ℕ := diagonal_half d1
    have s2 : ℝ := other_half_diagonal s d1_half
    have d2 := 2 * s2
    have area := (d1 * d2) / 2
    use area
    sorry

end area_of_rhombus_l126_126615


namespace integer_roots_count_l126_126637

-- Define the conditions
variables {a b c d : ℤ} -- coefficients are integers
def P (x : ℤ) : ℤ := x^4 + a*x^3 + b*x^2 + c*x + d

-- Formalize the proof problem
theorem integer_roots_count (m : ℕ) :
  (∃ x1 x2 x3 x4 : ℤ, P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0 ∧ P x4 = 0 ∧
                       multiset.card (multiset.of_list [x1, x2, x3, x4]) = m) →
  m ∈ {0, 1, 2, 4} :=
sorry

end integer_roots_count_l126_126637


namespace max_additional_payment_l126_126065

theorem max_additional_payment (t1 t2 t3 : ℝ) (a b c d e f : ℕ) (paid : ℝ) :
  t1 = 4.03 ∧ t2 = 1.01 ∧ t3 = 3.39 ∧ 
  a = 1214 ∧ b = 1270 ∧ c = 1298 ∧ d = 1337 ∧ e = 1347 ∧ f = 1402 ∧ 
  paid = 660.72 → 
  max_additional = 397.34 :=
begin
  sorry
end

end max_additional_payment_l126_126065


namespace diameter_segments_l126_126163

theorem diameter_segments (r : ℝ) (h : CH = 10) (diam_AB : AB = 2 * r)
    (perp : CD ⊥ AB) : ∃ (a b : ℝ), AB = a + b ∧ abs (a - b) = 4 * sqrt 6 := sorry

end diameter_segments_l126_126163


namespace cube_surface_area_increase_l126_126331

def increase_surface_area (side_length : ℝ) (cuts : ℕ) : ℝ :=
  2 * (cuts * (side_length ^ 2))

theorem cube_surface_area_increase :
  let side_length := 10 in
  let cuts := 3 in
  increase_surface_area side_length cuts = 600 :=
by
  sorry

end cube_surface_area_increase_l126_126331


namespace P_necessary_but_not_sufficient_for_q_l126_126409

def M : Set ℝ := {x : ℝ | (x - 1) * (x - 2) > 0}
def N : Set ℝ := {x : ℝ | x^2 + x < 0}

theorem P_necessary_but_not_sufficient_for_q :
  (∀ x, x ∈ N → x ∈ M) ∧ (∃ x, x ∈ M ∧ x ∉ N) :=
by
  sorry

end P_necessary_but_not_sufficient_for_q_l126_126409


namespace dot_product_AD_BC_l126_126984

variable (AB CD : ℝ) (A B C D : Type*) [InnerProductSpace ℝ (A → ℝ)] 
variable (a b : A → ℝ) 

axiom h1 : AB = 155
axiom h2 : CD = 13
axiom h3 : inner_product (A → ℝ) (b) a = 0

noncomputable def vectors_AD_BC (a b : A → ℝ) : ℝ :=
let AD := a + (CD / AB) • b in
let BC := b + (CD / AB) • a in
inner_product (AD) (BC)

theorem dot_product_AD_BC
  (AB CD : ℝ) (A B C D : Type*) [InnerProductSpace ℝ (A → ℝ)]
  (a b : A → ℝ)
  (h1 : AB = 155) (h2 : CD = 13) (h3 : inner_product (A → ℝ) (b) a = 0) :
  vectors_AD_BC AB CD A B C D a b = 2015 :=
sorry

end dot_product_AD_BC_l126_126984


namespace total_items_purchased_l126_126889

/-- Proof that Ike and Mike buy a total of 9 items given the constraints. -/
theorem total_items_purchased
  (total_money : ℝ)
  (sandwich_cost : ℝ)
  (drink_cost : ℝ)
  (combo_factor : ℕ)
  (money_spent_on_sandwiches : ℝ)
  (number_of_sandwiches : ℕ)
  (number_of_drinks : ℕ)
  (num_free_sandwiches : ℕ) :
  total_money = 40 →
  sandwich_cost = 5 →
  drink_cost = 1.5 →
  combo_factor = 5 →
  number_of_sandwiches = 9 →
  number_of_drinks = 0 →
  money_spent_on_sandwiches = number_of_sandwiches * sandwich_cost →
  total_money = money_spent_on_sandwiches →
  num_free_sandwiches = number_of_sandwiches / combo_factor →
  number_of_sandwiches = number_of_sandwiches + num_free_sandwiches →
  number_of_sandwiches + number_of_drinks = 9 :=
by
  intros
  sorry

end total_items_purchased_l126_126889


namespace count_four_digit_integers_with_3_or_7_digits_l126_126512

-- Definitions based on conditions
def is_four_digit_integer (n : ℕ) : Prop :=
  n >= 1000 ∧ n <= 9999

def is_made_of_3_and_7 (n : ℕ) : Prop :=
  n.digits 10.All (λ d, d = 3 ∨ d = 7)

-- Problem statement:
theorem count_four_digit_integers_with_3_or_7_digits : 
  {n : ℕ // is_four_digit_integer n ∧ is_made_of_3_and_7 n}.card = 16 :=
by 
  sorry

end count_four_digit_integers_with_3_or_7_digits_l126_126512


namespace exists_X_make_XYZ_isosceles_l126_126459

-- Define the concepts of points and angles
noncomputable def point := ℝ × ℝ
noncomputable def angle (O A B : point) : ℝ := sorry -- Definition for the angle ∠AOB

-- Define the problem setup
variables (O M N A B : point)
variable (acute_MON : angle O M N < π / 2)
variable (inside_A : sorry) -- A is inside angle MON
variable (inside_B : sorry) -- B is inside angle MON

-- Define the points Y and Z as intersections of lines XA and XB with ON
noncomputable def intersection_of_line_and_segment (X A O N : point) : point := sorry
noncomputable def Y := intersection_of_line_and_segment X A O N
noncomputable def Z := intersection_of_line_and_segment X B O N

-- Define the condition for an isosceles triangle XYZ
def is_isosceles_triangle (X Y Z : point) : Prop := distance X Y = distance X Z

-- The main statement
theorem exists_X_make_XYZ_isosceles :
  ∃ X : point, X ∈ (line_through O M) ∧ is_isosceles_triangle X Y Z :=
sorry

end exists_X_make_XYZ_isosceles_l126_126459


namespace lemonade_water_requirement_l126_126935

variables (W S L H : ℕ)

-- Definitions based on the conditions
def water_equation (W S : ℕ) := W = 5 * S
def sugar_equation (S L : ℕ) := S = 3 * L
def honey_equation (H L : ℕ) := H = L
def lemon_juice_amount (L : ℕ) := L = 2

-- Theorem statement for the proof problem
theorem lemonade_water_requirement :
  ∀ (W S L H : ℕ), 
  (water_equation W S) →
  (sugar_equation S L) →
  (honey_equation H L) →
  (lemon_juice_amount L) →
  W = 30 :=
by
  intros W S L H hW hS hH hL
  sorry

end lemonade_water_requirement_l126_126935


namespace dara_waiting_years_l126_126263

theorem dara_waiting_years :
  let minimum_age_required : ℕ := 25,
      years_in_future : ℕ := 6,
      jane_current_age : ℕ := 28,
      year_increment_period : ℕ := 5,
      age_increment : ℕ := 1,
      tom_age_difference : ℕ := 10,
      initial_minimum_age : ℕ := 24 in
  let jane_future_age := jane_current_age + years_in_future,
      dara_future_age := jane_future_age / 2,
      dara_current_age := dara_future_age - years_in_future,
      tom_current_age := jane_current_age + tom_age_difference,
      period_count_past := (minimum_age_required - initial_minimum_age) / age_increment,
      future_years_wait := minimum_age_required - dara_current_age,
      period_count_future := future_years_wait / year_increment_period,
      adjusted_minimum_age := minimum_age_required + period_count_future * age_increment,
      waiting_years := adjusted_minimum_age - dara_current_age in
  waiting_years = 16 := by
  sorry

end dara_waiting_years_l126_126263


namespace shortest_distance_between_points_on_hyperbola_l126_126800

noncomputable def shortest_distance_hyperbola (a : ℝ) : ℝ :=
  (a^(2/3) + a^(-2/3))^(3/2)

theorem shortest_distance_between_points_on_hyperbola (a b : ℝ)
  (ha : a > 0) (hb : b > 0) (h_ab : a * b = 1):
  shortest_distance_hyperbola a = (a^(2/3) + a^(-2/3))^(3/2) :=
by
  sorry

end shortest_distance_between_points_on_hyperbola_l126_126800


namespace sqrt_of_four_is_pm_two_l126_126658

theorem sqrt_of_four_is_pm_two (y : ℤ) : y * y = 4 → y = 2 ∨ y = -2 := by
  sorry

end sqrt_of_four_is_pm_two_l126_126658


namespace projection_line_l126_126655

-- Define the projection function for a 2D vector
def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot := v.1 * w.1 + v.2 * w.2
  let norm_sq := w.1 * w.1 + w.2 * w.2
  (dot / norm_sq * w.1, dot / norm_sq * w.2)

-- Define the vectors u, w and the projection result
def u (x y : ℝ) : ℝ × ℝ := (x, y)
def w : ℝ × ℝ := (3, 1)
def proj_result : ℝ × ℝ := (3 / 5, 1 / 5)

-- The theorem to prove
theorem projection_line (x y : ℝ) (h : proj (u x y) w = proj_result) : y = -3 * x + 2 :=
  sorry

end projection_line_l126_126655


namespace expression_for_f_n_f_n_is_rational_l126_126932

noncomputable def f_n (n : ℕ) (θ : ℝ) : ℝ :=
  (Real.sin θ) ^ n + (Real.cos θ) ^ n

variables {a : ℚ} (h_a : 0 < a ∧ a < 1) 

theorem expression_for_f_n (θ : ℝ) (n : ℕ) (h_n : 0 < n) (h_a_n : ∑ i in finset.range n, ((a + (Real.sqrt (2 - a^2))) / 2) ^ i + ((a - (Real.sqrt (2 - a^2))) / 2) ^ i = f_n n θ) :
  f_n n θ = (a + Real.sqrt (2 - a ^ 2)) / 2 ^ n + (a - Real.sqrt (2 - a ^ 2)) / 2 ^ n := sorry

theorem f_n_is_rational (θ : ℝ) (h_a : f_1 θ = a) (n : ℕ) (h_n : 0 < n) :
  f_n n θ ∈ ℚ := sorry

end expression_for_f_n_f_n_is_rational_l126_126932


namespace minimum_value_of_m_l126_126484

theorem minimum_value_of_m (a m : ℝ) (f : ℝ → ℝ) (h₁ : f = λ x, 2 ^ abs (x - a))
  (symmetry_cond : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (monotonic_cond : ∀ x : ℝ, m ≤ x → f x ≤ f (x + 1)) : m = 1 := sorry

end minimum_value_of_m_l126_126484


namespace polynomial_negative_l126_126953

-- Definition of f(x)
def f (a x : ℝ) := 
  (a - x) ^ 6 - 3 * a * (a - x) ^ 5 + (5 / 2) * a ^ 2 * (a - x) ^ 4 - (1 / 2) * a ^ 4 * (a - x) ^ 2

theorem polynomial_negative (a x : ℝ) (h1 : 0 < x) (h2 : x < a) : f a x < 0 :=
  sorry

end polynomial_negative_l126_126953


namespace same_quadratic_function_b_l126_126876

theorem same_quadratic_function_b (a c b : ℝ) :
    (∀ x : ℝ, a * (x - 2)^2 + c = (2 * x - 5) * (x - b)) → b = 3 / 2 :=
by
  sorry

end same_quadratic_function_b_l126_126876


namespace tangent_parabola_line_l126_126154

theorem tangent_parabola_line (a : ℝ) :
  (∃ x0 : ℝ, ax0^2 + 3 = 2 * x0 + 1) ∧ (∀ x : ℝ, a * x^2 - 2 * x + 2 = 0 → x = x0) → a = 1/2 :=
by
  intro h
  sorry

end tangent_parabola_line_l126_126154


namespace hindu_percentage_l126_126552

theorem hindu_percentage (total_boys : ℕ)
  (percentage_muslim : ℝ)
  (percentage_sikh : ℝ)
  (other_communities : ℕ) :
  (total_boys = 850) →
  (percentage_muslim = 34 / 100) →
  (percentage_sikh = 10 / 100) →
  (other_communities = 238) →
  (100 * ((total_boys - (total_boys * percentage_muslim + total_boys * percentage_sikh + other_communities)) / total_boys) = 28) :=
begin
  intros h_total h_muslim h_sikh h_other,
  sorry
end

end hindu_percentage_l126_126552


namespace possible_paths_3_seconds_number_of_paths_10_seconds_l126_126001

def ant_moves (t : Nat) : Set (List Nat) :=
  if t = 0 then {{[0]}}
  else if t = 1 then {{[0, 1]}}
  else if t = 2 then {{[0, 1, 0], [0, 1, 2]}}
  else if t = 3 then {{[0, 1, 0, 1], [0, 1, 2, 1]}}
  else sorry -- Define recursion for the general case if needed.

theorem possible_paths_3_seconds :
  ant_moves 3 = {{[0, 1, 0, 1], [0, 1, 2, 1]}} := sorry

def number_of_possible_paths (t : Nat) : Nat :=
  if t = 0 then 1
  else if t % 2 = 1 then number_of_possible_paths (t - 1)
  else 2 * number_of_possible_paths (t - 2)

theorem number_of_paths_10_seconds :
  number_of_possible_paths 10 = 32 := sorry

end possible_paths_3_seconds_number_of_paths_10_seconds_l126_126001


namespace number_of_customers_trimmed_l126_126881

-- Definitions based on the conditions
def total_sounds : ℕ := 60
def sounds_per_person : ℕ := 20

-- Statement to prove
theorem number_of_customers_trimmed :
  ∃ n : ℕ, n * sounds_per_person = total_sounds ∧ n = 3 :=
sorry

end number_of_customers_trimmed_l126_126881


namespace analytical_expression_range_of_m_l126_126836

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ∈ Icc (-3 : ℝ) (0 : ℝ) then (1 / 9 ^ x) + (a / 4 ^ x)
  else 0

theorem analytical_expression (x a : ℝ) :
  (∀ x, f (-x) a = -f x a) → 
  (∀ x, x ∈ Icc (-3 : ℝ) 0 → f x a = (1 / 9 ^ x) + (a / 4 ^ x)) →
  (a = -1) →
  (∀ x, x ∈ Icc (0 : ℝ) 3 → f x a = 4 ^ x - 9 ^ x) :=
sorry

theorem range_of_m (m : ℝ) (a : ℝ) :
  (∀ x, f (-x) a = -f x a) → 
  (∀ x, x ∈ Icc (-3 : ℝ) 0 → f x a = (1 / 9 ^ x) + (a / 4 ^ x)) →
  (a = -1) →
  (∀ x, x ∈ Icc (-1 : ℝ) (-1 / 2) → f x a ≤ (m / 3 ^ x) - (1 / 4 ^ (x - 1))) →
  (7 ≤ m) :=
sorry

end analytical_expression_range_of_m_l126_126836


namespace combined_molecular_weight_l126_126040

theorem combined_molecular_weight :
  let H_weight := 1.008
  let O_weight := 15.999
  let C_weight := 12.011
  let H2O_weight := (2 * H_weight) + O_weight
  let CO2_weight := C_weight + (2 * O_weight)
  let CH4_weight := C_weight + (4 * H_weight)
  H2O_weight + CO2_weight + CH4_weight = 78.067 :=
by
  -- The let expressions define the atomic weights and molecular weights of each compound.
  let H_weight := 1.008
  let O_weight := 15.999
  let C_weight := 12.011
  let H2O_weight := (2 * H_weight) + O_weight
  let CO2_weight := C_weight + (2 * O_weight)
  let CH4_weight := C_weight + (4 * H_weight)
  -- We then specify the combined molecular weight and show the expected result.
  trivial
  sorry -- Proof of the numerical calculation follows from the provided solution steps.

end combined_molecular_weight_l126_126040


namespace roots_sum_equality_l126_126860

theorem roots_sum_equality {a b c : ℝ} {x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ} :
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 1 = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 2 = 0 → x = y₁ ∨ x = y₂ ∨ x = y₃ ∨ x = y₄) →
  x₁ + x₂ = x₃ + x_₄ →
  y₁ + y₂ = y₃ + y₄ :=
sorry

end roots_sum_equality_l126_126860


namespace jill_net_monthly_salary_l126_126325

variable (S : ℝ)
variable (discretionary_income : ℝ) (vacation_fund savings socializing gifts_charity : ℝ)

def jill_discretionary_income (S : ℝ) : ℝ :=
  (1 / 5) * S

def jill_vacation_fund (discretionary_income : ℝ) : ℝ :=
  0.30 * discretionary_income

def jill_savings (discretionary_income : ℝ) : ℝ :=
  0.20 * discretionary_income

def jill_socializing (discretionary_income : ℝ) : ℝ :=
  0.35 * discretionary_income

def jill_gifts_charity (discretionary_income : ℝ) : ℝ :=
  discretionary_income - (jill_vacation_fund discretionary_income + jill_savings discretionary_income + jill_socializing discretionary_income)

theorem jill_net_monthly_salary (h : jill_gifts_charity (jill_discretionary_income S) = 111) :
  S = 3700 :=
by
  sorry

end jill_net_monthly_salary_l126_126325


namespace find_missing_student_number_l126_126161

theorem find_missing_student_number:
  ∃ (n : ℕ), n ∈ {1, ..., 48} ∧ 
             (let k := 12 in 
              ∀ i ∈ {0, 1, 2, 3}, 5 + i * k ∈ {5, 29, 41, n}) ∧ 
             (n ≠ 5 ∧ n ≠ 29 ∧ n ≠ 41) := 
begin
  sorry
end

end find_missing_student_number_l126_126161


namespace percent_problem_l126_126532

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end percent_problem_l126_126532


namespace solution_set_x_plus_3_f_x_plus_4_l126_126843

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom even_f_x_plus_1 : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom deriv_negative_f : ∀ x : ℝ, x > 1 → f' x < 0
axiom f_at_4_equals_zero : f 4 = 0

-- To prove
theorem solution_set_x_plus_3_f_x_plus_4 :
  {x : ℝ | (x + 3) * f (x + 4) < 0} = {x : ℝ | -6 < x ∧ x < -3} ∪ {x : ℝ | x > 0} := sorry

end solution_set_x_plus_3_f_x_plus_4_l126_126843


namespace result_has_five_digits_l126_126582

def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else ⌊log 10 n⌋ + 1

theorem result_has_five_digits (A B C : ℕ) 
  (hA : 1 ≤ A ∧ A ≤ 9) 
  (hB : 1 ≤ B ∧ B ≤ 9) 
  (hC : 1 ≤ C ∧ C ≤ 9) :
  num_digits ((9876 + (100 * A + 54) + (10 * B + 2)) - C) = 5 := 
by sorry

end result_has_five_digits_l126_126582


namespace count_positive_integers_l126_126515

theorem count_positive_integers (n : ℕ) (m : ℕ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k < 100 ∧ (∃ (n : ℕ), n = 2 * k + 1 ∧ n < 200) 
  ∧ (∃ (m : ℤ), m = k * (k + 1) ∧ m % 5 = 0)) → 
  ∃ (cnt : ℕ), cnt = 20 :=
by
  sorry

end count_positive_integers_l126_126515


namespace ab_value_l126_126293

noncomputable def a := sorry
noncomputable def b := sorry

theorem ab_value (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 :=
by
  sorry

end ab_value_l126_126293


namespace perpendicular_length_difference_l126_126896

noncomputable def circle_radius (R : ℝ) := R

structure RegularPentagon (R : ℝ) :=
(center : ℝ)
(radius : ℝ)
(vertices : list (ℝ × ℝ))
(h_radius : radius = R)
(h_pentagon : vertices.length = 5)
/- Definition assumptions for regularity and inscribed conditions -/

axiom perpendicular_intersects (R : ℝ) (pent : RegularPentagon R) :
  ∃ A' (B C D : ℝ × ℝ), (/* vertex positions and perpendicular condition goes here */)

theorem perpendicular_length_difference (R : ℝ) (pent : RegularPentagon R) (A' B C D : ℝ × ℝ) :
  (∥A' - B∥ - ∥A' - C∥ = R) :=
by
  sorry

end perpendicular_length_difference_l126_126896


namespace circular_garden_area_l126_126670

theorem circular_garden_area :
  ∀ (A B C D : Type)
    (midpoint : (A → B → D) → Prop)
    (dist : (A → B → ℝ) → Prop)
    (R : ℝ) (π : ℝ)
    (AC BC : A → ℝ),
    midpoint D →
    dist A B = 20 →
    dist D C = 15 →
    (((dist A B) / 2) ^ 2 + 15 ^ 2 = R ^ 2) →
    π * R ^ 2 = 325 * π :=
by
  intros A B C D midpoint dist R π AC BC midpointAB distAB distDC hypothesis
  sorry

end circular_garden_area_l126_126670


namespace sum_non_solutions_eq_neg21_l126_126594

theorem sum_non_solutions_eq_neg21
  (A B C : ℝ)
  (h1 : ∀ x, ∃ k : ℝ, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h2 : ∃ A B C, ∀ x, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h3 : ∃! x, (x + C) * (x + 9) = 0)
   :
  -9 + -12 = -21 := by sorry

end sum_non_solutions_eq_neg21_l126_126594


namespace moles_of_Cl2_combined_l126_126078

-- Chemical equation representation (not necessary in practical Lean code, 
-- but helpful for explanation):
-- 2 NaOH + Cl2 → NaCl + NaOCl + H2O

def balanced_chemical_reaction (NaOH Cl2 NaCl NaOCl H2O : ℕ) : Prop :=
  2 * NaOH + Cl2 = NaCl + NaOCl + H2O

-- Given conditions
constants 
  (naoh molesH2O cl2 naCl naOCl : ℕ)
  (h2o : ℕ)

axiom naoh_value : naoh = 2
axiom h2o_value : h2o = 1
axiom chemical_equation : balanced_chemical_reaction naoh cl2 naCl naOCl h2o

-- The statement to prove
theorem moles_of_Cl2_combined : cl2 = 1 := by
  -- Proof omitted
  sorry

end moles_of_Cl2_combined_l126_126078


namespace bank1_more_advantageous_l126_126606

-- Define the quarterly interest rate for Bank 1
def bank1_quarterly_rate : ℝ := 0.8

-- Define the annual interest rate for Bank 2
def bank2_annual_rate : ℝ := 9.0

-- Define the annual compounded interest rate for Bank 1
def bank1_annual_yield : ℝ :=
  (1 + bank1_quarterly_rate) ^ 4

-- Define the annual rate directly for Bank 2
def bank2_annual_yield : ℝ :=
  1 + bank2_annual_rate

-- The theorem stating that Bank 1 is more advantageous than Bank 2
theorem bank1_more_advantageous : bank1_annual_yield > bank2_annual_yield :=
  sorry

end bank1_more_advantageous_l126_126606


namespace ellipse_om_length_l126_126829

theorem ellipse_om_length :
  let a := 5
  let b := 4
  let c := sqrt (a ^ 2 - b ^ 2)
  let left_directrix := - (a ^ 2 / c)
  let F := (-c, 0)
  (∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ∧ (abs (x + left_directrix) = 10) →
  (∃ P : ℝ × ℝ, P = (x, y)) →
  let M := ((x + (-c)) / 2, (y + 0) / 2)
  sqrt (M.1 ^ 2 + M.2 ^ 2) = 2) :=
by
  intros
  have : x = 5 / 3 ∨ x = -55 / 3 := sorry -- Discarding step
  have : M = (-2 / 3, ± 4 * sqrt 2 / 3) := sorry -- Calculation of the midpoint
  rw [sqrt_add_eq] -- Magnitude calculation
  sorry

end ellipse_om_length_l126_126829


namespace area_increase_of_pentagon_l126_126726

-- Defining the initial conditions
def initial_area : ℝ := 200 * Real.sqrt 5
def new_side_length_increase : ℝ := 2
noncomputable def pentagon_area (s : ℝ) : ℝ := (5 * s^2) / 4 * (Real.sqrt 5 + 1)

-- The statement to prove
theorem area_increase_of_pentagon :
  pentagon_area (12) - pentagon_area (10) = 180 - 20 * Real.sqrt 5 :=
sorry

end area_increase_of_pentagon_l126_126726


namespace ball_in_box_l126_126138

theorem ball_in_box : 
  let num_ways_to_place_balls := 3^5 in 
  num_ways_to_place_balls = 243 :=
by
  sorry

end ball_in_box_l126_126138


namespace min_value_expression_l126_126591

theorem min_value_expression (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) : 
  (pt2 + qu2 + rv2 + sw2)2 = 400 := 
by 
  sorry

end min_value_expression_l126_126591


namespace additional_laps_needed_l126_126937

-- Definitions of problem conditions
def total_required_distance : ℕ := 2400
def lap_length : ℕ := 150
def madison_laps : ℕ := 6
def gigi_laps : ℕ := 6

-- Target statement to prove the number of additional laps needed
theorem additional_laps_needed : (total_required_distance - (madison_laps + gigi_laps) * lap_length) / lap_length = 4 := by
  sorry

end additional_laps_needed_l126_126937


namespace concentric_circle_ratio_l126_126673

theorem concentric_circle_ratio (r R : ℝ) (hRr : R > r)
  (new_circles_tangent : ∀ (C1 C2 C3 : ℝ), C1 = C2 ∧ C2 = C3 ∧ C1 < R ∧ r < C1): 
  R = 3 * r := by sorry

end concentric_circle_ratio_l126_126673


namespace cost_difference_is_360_l126_126196

def sailboat_cost_per_day : ℕ := 60
def ski_boat_cost_per_hour : ℕ := 80
def ken_days : ℕ := 2
def aldrich_hours_per_day : ℕ := 3
def aldrich_days : ℕ := 2

theorem cost_difference_is_360 :
  let ken_total_cost := sailboat_cost_per_day * ken_days
  let aldrich_total_cost_per_day := ski_boat_cost_per_hour * aldrich_hours_per_day
  let aldrich_total_cost := aldrich_total_cost_per_day * aldrich_days
  let cost_diff := aldrich_total_cost - ken_total_cost
  cost_diff = 360 :=
by
  sorry

end cost_difference_is_360_l126_126196


namespace perpendicular_parallel_imp_perpendicular_l126_126036

variables {Point : Type*} [inner_product_space ℝ Point]

structure Line (P : Type*) := 
  (contains : P → Prop)

structure Plane (P : Type*) := 
  (contains : P → Prop)

variables {m n : Line Point} {α β : Plane Point}

def is_parallel (l : Line Point) (p : Plane Point) : Prop :=
  ∀ x y : Point, l.contains x → l.contains y → p.contains x ↔ p.contains y

def is_perpendicular (l : Line Point) (p : Plane Point) : Prop :=
  ∀ x y : Point, l.contains x → l.contains y → inner_product_space.dot x y = 0

theorem perpendicular_parallel_imp_perpendicular {m : Line Point} {α β : Plane Point} :
  (is_perpendicular m α) → (is_parallel m β) → (∀ x y, α.contains x ∧ β.contains y → inner_product_space.dot x y = 0) :=
by
  intros h_perp h_par
  sorry

end perpendicular_parallel_imp_perpendicular_l126_126036


namespace employees_no_increase_l126_126231

def total_employees : ℕ := 480
def salary_increase_ratio : ℝ := 0.10
def travel_increase_ratio : ℝ := 0.20
def both_increase_ratio : ℝ := 0.05

def salary_increase_count : ℕ := (salary_increase_ratio * total_employees).to_nat
def travel_increase_count : ℕ := (travel_increase_ratio * total_employees).to_nat
def both_increase_count : ℕ := (both_increase_ratio * total_employees).to_nat

def at_least_one_increase_count : ℕ := salary_increase_count + travel_increase_count - both_increase_count

def no_increase_count : ℕ := total_employees - at_least_one_increase_count

theorem employees_no_increase :
  no_increase_count = 360 := by
  sorry

end employees_no_increase_l126_126231


namespace permutation_absolute_differences_l126_126424

def is_permutation (l1 l2 : List ℕ) : Prop :=
  l1.length = l2.length ∧ ∀ x, l1.count x = l2.count x 

def distinct_abs_diffs (a : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (a.nthLe i sorry - (i + 1)) ≠ abs (a.nthLe j sorry - (j + 1))

theorem permutation_absolute_differences (n : ℕ) :
  (∃ (a : List ℕ), a.length = n ∧ is_permutation a (List.range n) ∧ distinct_abs_diffs a) ↔
  (∃ k : ℕ, n = 4 * k ∨ n = 4 * k + 1) :=
by
  sorry

end permutation_absolute_differences_l126_126424


namespace sound_frequency_and_speed_glass_proof_l126_126713

def length_rod : ℝ := 1.10 -- Length of the glass rod, l in meters
def nodal_distance_air : ℝ := 0.12 -- Distance between nodal points in air, l' in meters
def speed_sound_air : ℝ := 340 -- Speed of sound in air, V in meters per second

-- Frequency of the sound produced
def frequency_sound_produced : ℝ := 1416.67

-- Speed of longitudinal waves in the glass
def speed_longitudinal_glass : ℝ := 3116.67

theorem sound_frequency_and_speed_glass_proof :
  (2 * nodal_distance_air = 0.24) ∧
  (frequency_sound_produced * (2 * length_rod) = speed_longitudinal_glass) :=
by
  -- Here we will include real equivalent math proof in the future
  sorry

end sound_frequency_and_speed_glass_proof_l126_126713


namespace pens_cost_l126_126722

theorem pens_cost (pens_pack_cost : ℝ) (pens_pack_quantity : ℕ) (total_pens : ℕ) (unit_price : ℝ) (total_cost : ℝ)
  (h1 : pens_pack_cost = 45) (h2 : pens_pack_quantity = 150) (h3 : total_pens = 3600) (h4 : unit_price = pens_pack_cost / pens_pack_quantity)
  (h5 : total_cost = total_pens * unit_price) : total_cost = 1080 := by
  sorry

end pens_cost_l126_126722


namespace relationship_necessary_but_not_sufficient_l126_126272

theorem relationship_necessary_but_not_sufficient (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a ^ 3 > b ^ 3 → log 3 a > log 3 b) ∧ ¬ (log 3 a > log 3 b → a ^ 3 > b ^ 3) :=
by
  sorry

end relationship_necessary_but_not_sufficient_l126_126272


namespace more_valley_than_humpy_l126_126676

def is_humpy (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 > d4 ∧ d4 > d5

def is_valley (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 > d2 ∧ d2 > d3 ∧ d3 < d4 ∧ d4 < d5

def starts_with_5 (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  d1 = 5

theorem more_valley_than_humpy :
  (∃ m, starts_with_5 m ∧ is_humpy m) → (∃ n, starts_with_5 n ∧ is_valley n) ∧ 
  (∀ x, starts_with_5 x → is_humpy x → ∃ y, starts_with_5 y ∧ is_valley y ∧ y ≠ x) :=
by sorry

end more_valley_than_humpy_l126_126676


namespace mrs_McGillicuddy_student_count_l126_126229

theorem mrs_McGillicuddy_student_count :
  let morning_registered := 25
  let morning_absent := 3
  let early_afternoon_registered := 24
  let early_afternoon_absent := 4
  let late_afternoon_registered := 30
  let late_afternoon_absent := 5
  let evening_registered := 35
  let evening_absent := 7
  let morning_present := morning_registered - morning_absent
  let early_afternoon_present := early_afternoon_registered - early_afternoon_absent
  let late_afternoon_present := late_afternoon_registered - late_afternoon_absent
  let evening_present := evening_registered - evening_absent
  let total_present := morning_present + early_afternoon_present + late_afternoon_present + evening_present
  total_present = 95 :=
by
  sorry

end mrs_McGillicuddy_student_count_l126_126229


namespace percentage_of_nuts_in_Jane_mix_l126_126968

variable (x : ℝ) -- percentage of nuts in Jane's trail mix

-- Conditions
def condition1 := (0.3 * (Sue's trail given amount) + x * (Jane's trail) combined amount) / new given total = 0.45
def condition2 := (0.3 * (Sue's trail given amount) + x * (Jane's trail) combined amount) / new given total = 0.60

theorem percentage_of_nuts_in_Jane_mix : ∀ x, (0.3 * (Sue's trail given amount) + x * (Jane's trail) combined amount eq 0.45 → x = 0.60) := by begin {
  sorry
} end

end percentage_of_nuts_in_Jane_mix_l126_126968


namespace find_f_neg2016_l126_126448

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg2016 (a b k : ℝ) (h : f a b 2016 = k) (h_ab : a * b ≠ 0) : f a b (-2016) = 2 - k :=
by
  sorry

end find_f_neg2016_l126_126448


namespace adam_first_half_correct_l126_126690

-- Define the conditions
def second_half_correct := 2
def points_per_question := 8
def final_score := 80

-- Define the number of questions Adam answered correctly in the first half
def first_half_correct :=
  (final_score - (second_half_correct * points_per_question)) / points_per_question

-- Statement to prove
theorem adam_first_half_correct : first_half_correct = 8 :=
by
  -- skipping the proof
  sorry

end adam_first_half_correct_l126_126690


namespace continuous_jensen_implies_convex_l126_126346

variable {α : Type*} [LinearOrderedField α] (f : α → α)
variable (x₁ x₂ : α) (λ : α) (hλ : 0 ≤ λ ∧ λ ≤ 1)

-- Define the Jensen convexity condition
def jensen_convex (f : α → α) : Prop :=
∀ (x₁ x₂ : α), f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

-- Define the general convexity condition
def convex (f : α → α) : Prop :=
∀ (x₁ x₂ : α) (λ : α), 0 ≤ λ ∧ λ ≤ 1 → f (λ * x₁ + (1 - λ) * x₂) ≤ λ * f x₁ + (1 - λ) * f x₂

-- Formalize the theorem statement
theorem continuous_jensen_implies_convex (hf : continuous f) (hf_jensen : jensen_convex f) :
  convex f :=
sorry

end continuous_jensen_implies_convex_l126_126346


namespace second_solution_carbonated_water_percentage_l126_126729

theorem second_solution_carbonated_water_percentage :
  ∃ (x : ℝ), 
  (0.80 * 0.40 + x * 0.60 = 0.65) → x = 0.55 :=
begin
  sorry -- proof to be implemented
end

end second_solution_carbonated_water_percentage_l126_126729


namespace cost_price_per_meter_proof_l126_126696

noncomputable def cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) : ℝ :=
  total_cost / total_length

theorem cost_price_per_meter_proof :
  ∀ (total_cost total_length : ℝ), total_cost = 407 → total_length = 9.25 →
    cost_price_per_meter total_cost total_length = 44 :=
by
  intros total_cost total_length h_cost h_length
  unfold cost_price_per_meter
  rw [h_cost, h_length]
  norm_num
  sorry

end cost_price_per_meter_proof_l126_126696


namespace calculate_triangle_area_l126_126796

-- Define the side lengths of the triangle.
def side1 : ℕ := 13
def side2 : ℕ := 13
def side3 : ℕ := 24

-- Define the area calculation.
noncomputable def triangle_area : ℕ := 60

-- Statement of the theorem we wish to prove.
theorem calculate_triangle_area :
  ∃ (a b c : ℕ) (area : ℕ), a = side1 ∧ b = side2 ∧ c = side3 ∧ area = triangle_area :=
sorry

end calculate_triangle_area_l126_126796


namespace arithmetic_sequence_solution_l126_126422

noncomputable def fractional_part (x : ℝ) : ℝ := x - Real.floor x

theorem arithmetic_sequence_solution (x : ℝ) (hx : x ≠ 0) (h_seq : fractional_part x + (Real.floor x + 1) + x = Real.floor x + 1 + x * 2) : x = -2 ∨ x = -1 / 2 :=
  sorry

end arithmetic_sequence_solution_l126_126422


namespace preimage_of_20_l126_126969

theorem preimage_of_20 (A B : set ℕ) (f : ℕ → ℕ) (h₁ : ∀ n ∈ A, f n = 2 * n^2) (h₂ : ∀ m ∈ B, ∃ n ∈ A, m = f n) :
  ∃ n ∈ A, f n = 20 ∧ n = 3 :=
by
  sorry

end preimage_of_20_l126_126969


namespace hyperbola_imaginary_axis_length_l126_126885

theorem hyperbola_imaginary_axis_length (m : ℝ)
  (h : ∃ (m : ℝ), (λ (x y : ℝ), x^2 + m * y^2 = 1) (-real.sqrt 2) 2) :
  let b := 2 in
  (2 * b) = 4 :=
by
  sorry

end hyperbola_imaginary_axis_length_l126_126885


namespace complex_number_solution_l126_126857

theorem complex_number_solution {i z : ℂ} (h : (2 : ℂ) / (1 + i) = z + i) : z = 1 + 2 * i :=
sorry

end complex_number_solution_l126_126857


namespace no_abundant_deficient_even_numbers_lt_50_l126_126000

def proper_factors (n : ℕ) : list ℕ := 
  (list.filter (λ x, x < n ∧ n % x = 0) (list.range n))

def sum_proper_factors (n : ℕ) : ℕ :=
  list.sum (proper_factors n)

def abundant_deficient (n : ℕ) : Prop :=
  sum_proper_factors n = n / 2

def even (n : ℕ) : Prop :=
  n % 2 = 0

theorem no_abundant_deficient_even_numbers_lt_50 :
  finset.card (finset.filter (λ n, abundant_deficient n ∧ even n) (finset.range 50)) = 0 := 
by
  sorry

end no_abundant_deficient_even_numbers_lt_50_l126_126000


namespace right_triangle_sides_l126_126850

theorem right_triangle_sides :
  ∃ (a b c : ℕ), a + b + c = 80 ∧ a^2 + b^2 = c^2 ∧ a = 30 ∧ b = 16 ∧ c = 34 :=
by
  use 30, 16, 34
  simp
  sorry

end right_triangle_sides_l126_126850


namespace fraction_evaluation_l126_126792

theorem fraction_evaluation :
  (1 / 2 + 1 / 3) / (3 / 7 - 1 / 5) = 175 / 48 :=
by
  sorry

end fraction_evaluation_l126_126792


namespace find_m_l126_126498

-- Define the function and conditions
def power_function (x : ℝ) (m : ℕ) : ℝ := x^(m - 2)

theorem find_m (m : ℕ) (x : ℝ) (h1 : 0 < m) (h2 : power_function 0 m = 0 → false) : m = 1 ∨ m = 2 :=
by
  sorry -- Skip the proof

end find_m_l126_126498


namespace angle_y_measure_l126_126179

theorem angle_y_measure (p q : Type) [parallel p q]
  (angle1 angle2 : ℝ) (h_angle1 : angle1 = 40)
  (h_angle2 : angle2 = 90) :
  ∃ y, y = 140 :=
by {
  use 140,
  sorry
}

end angle_y_measure_l126_126179


namespace volume_in_each_beaker_after_reaction_l126_126774

noncomputable def solution_volumes : List (String × Nat) :=
  [("A", 7), ("A", 4), ("B", 5), ("B", 4), ("A", 6), ("B", 8), ("A", 7), ("B", 3), ("A", 9), ("B", 6)]

def volume_reduction_percentage : Float := 0.20

def number_of_beakers : Nat := 5

theorem volume_in_each_beaker_after_reaction :
  let total_volume_A := solution_volumes.filter (λ p => p.1 = "A").map (λ p => p.2).sum
  let total_volume_B := solution_volumes.filter (λ p => p.1 = "B").map (λ p => p.2).sum
  let total_volume_before_reaction := total_volume_A + total_volume_B
  let reduced_volume := total_volume_before_reaction * (1 - volume_reduction_percentage)
  let volume_in_each_beaker := reduced_volume / number_of_beakers
  volume_in_each_beaker = 9.44 :=
by {
  -- This is where the proof would go.
  sorry
}

end volume_in_each_beaker_after_reaction_l126_126774


namespace max_square_le_sum_of_diffs_squared_l126_126271

variable (n : ℕ)
variable (a : Fin n → ℝ)

theorem max_square_le_sum_of_diffs_squared (h : ∑ i, a i = 0) :
    (Finset.univ.sup (λ k, (a k) ^ 2)) ≤ (n / 3) * ∑ i in Finset.range (n - 1), (a ⟨i, I_i_lt_n_pred n⟩ - a ⟨i + 1, I_succ_i_lt_n n i⟩) ^ 2 := sorry

end max_square_le_sum_of_diffs_squared_l126_126271


namespace prime_squares_mod_504_l126_126741

theorem prime_squares_mod_504 (p : ℕ) (hp : Nat.Prime p) (hgt7 : p > 7) :
  ∃ remainders : Finset (ZMod 504), remainders.card = 3 ∧
    ∀ q ∈ remainders, ∃ k : ℕ, q = (p ^ 2) % 504 := sorry

end prime_squares_mod_504_l126_126741


namespace exists_nat_n_gt_one_sqrt_expr_nat_l126_126410

theorem exists_nat_n_gt_one_sqrt_expr_nat (n : ℕ) : ∃ (n : ℕ), n > 1 ∧ ∃ (m : ℕ), n^(7 / 8) = m :=
by
  sorry

end exists_nat_n_gt_one_sqrt_expr_nat_l126_126410


namespace totalTrianglesInFigure_l126_126517

-- Definition of the problem involving a rectangle with subdivisions creating triangles
def numberOfTrianglesInRectangle : Nat :=
  let smallestTriangles := 24   -- Number of smallest triangles
  let nextSizeTriangles1 := 8   -- Triangles formed by combining smallest triangles
  let nextSizeTriangles2 := 12
  let nextSizeTriangles3 := 16
  let largestTriangles := 4
  smallestTriangles + nextSizeTriangles1 + nextSizeTriangles2 + nextSizeTriangles3 + largestTriangles

-- The Lean 4 theorem statement, stating that the total number of triangles equals 64
theorem totalTrianglesInFigure : numberOfTrianglesInRectangle = 64 := 
by
  sorry

end totalTrianglesInFigure_l126_126517


namespace find_p_l126_126884

theorem find_p (p: ℝ) (x1 x2: ℝ) (h1: p > 0) (h2: x1^2 + p * x1 + 1 = 0) (h3: x2^2 + p * x2 + 1 = 0) (h4: |x1^2 - x2^2| = p) : p = 5 :=
sorry

end find_p_l126_126884


namespace angie_age_problem_l126_126290

theorem angie_age_problem (a certain_number : ℕ) 
  (h1 : 2 * 8 + certain_number = 20) : 
  certain_number = 4 :=
by 
  sorry

end angie_age_problem_l126_126290


namespace find_n_l126_126338

theorem find_n:
  ∃ n : ℕ, (n : ℚ) / 2^n = 5 / 32 :=
sorry

end find_n_l126_126338


namespace no_way_to_write_as_sum_l126_126173

def can_be_written_as_sum (S : ℕ → ℕ) (n : ℕ) (k : ℕ) : Prop :=
  n + k - 1 + (n - 1) * (k - 1) / 2 = 528 ∧ n > 0 ∧ 2 ∣ n ∧ k > 1

theorem no_way_to_write_as_sum : 
  ∀ (S : ℕ → ℕ) (n k : ℕ), can_be_written_as_sum S n k →
    0 = 0 :=
by
  -- Problem states that there are 0 valid ways to write 528 as the sum
  -- of an increasing sequence of two or more consecutive positive integers
  sorry

end no_way_to_write_as_sum_l126_126173


namespace algebraic_expression_defined_iff_l126_126541

theorem algebraic_expression_defined_iff (x : ℝ) : (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 := by
  sorry

end algebraic_expression_defined_iff_l126_126541


namespace find_parallel_line_l126_126496

theorem find_parallel_line (m c offset : ℝ)
  (h : m = 1 / 2)
  (h' : c = 3)
  (D : offset = 5)
  : ∃ c1 c2 : ℝ, 
      (y = m * x + c1 ∧ 
       y = m * x + c2) ∧
      (c1 = c + (5 * sqrt 5 / 2)) ∧
      (c2 = c - (5 * sqrt 5 /2 )) :=
by 
  sorry

end find_parallel_line_l126_126496


namespace part1_l126_126336

theorem part1 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 3^x = 4^y ∧ 4^y = 6^z) : 
  (y / z - y / x) = Real.logb 6 4 - Real.logb 3 4 := 
sorry

end part1_l126_126336


namespace fish_per_multicolor_duck_l126_126439

theorem fish_per_multicolor_duck (num_white_ducks num_black_ducks num_multicolored_ducks fish_total : ℕ) 
    (fish_per_white_duck fish_per_black_duck : ℕ)
    (h1 : num_white_ducks = 3) (h2 : num_black_ducks = 7) (h3 : num_multicolored_ducks = 6)
    (h4 : fish_per_white_duck = 5) (h5 : fish_per_black_duck = 10) 
    (h6 : fish_total = 157) :
    (fish_total - (num_white_ducks * fish_per_white_duck + num_black_ducks * fish_per_black_duck))
    / num_multicolored_ducks = 12 := 
by 
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end fish_per_multicolor_duck_l126_126439


namespace points_cover_unit_area_rectangles_l126_126390

theorem points_cover_unit_area_rectangles :
  ∃ (points : Set (ℝ × ℝ)), 
  (∀ (x y : ℝ), ((x, y) ∈ points) → (0 ≤ x ∧ x ≤ 15 ∧ 0 ≤ y ∧ y ≤ 15)) ∧
  (points.finite ∧ points.to_finset.card = 1965) ∧
  (∀ (R : Set (ℝ × ℝ)), (R.measurable ∧ R.volume = 1 ∧ (∀ p ∈ R, (fst p ≥ 0 ∧ fst p ≤ 15 ∧ snd p ≥ 0 ∧ snd p ≤ 15)) → (∃ p ∈ points, p ∈ R))) :=
sorry

end points_cover_unit_area_rectangles_l126_126390


namespace point_a_coordinates_l126_126904

open Set

theorem point_a_coordinates (A B : ℝ × ℝ) :
  B = (2, 4) →
  (A.1 = B.1 + 3 ∨ A.1 = B.1 - 3) ∧ A.2 = B.2 →
  dist A B = 3 →
  A = (5, 4) ∨ A = (-1, 4) :=
by
  intros hB hA hDist
  sorry

end point_a_coordinates_l126_126904


namespace range_of_log_function_l126_126268

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x + 8

def is_positive (x : ℝ) : Prop := f x > 0

theorem range_of_log_function :
  (∀ x : ℝ, is_positive x → (-2 : ℝ) < x ∧ x < 4) →
  (∀ x : ℝ, y = Math.logBase 1/3 (f x) → y ≥ -2) →
  ∀ y : ℝ, (-2 : ℝ) <= y ∧ (y = -2 → x = 1) ∧ (Math.logBase 1/3 (f x) = y) :=
sorry

end range_of_log_function_l126_126268


namespace exists_white_vertices_parallelepiped_inscribed_l126_126369

noncomputable def inscribed_parallelepiped_all_white (S : Sphere) (red_fraction : ℝ) :=
  red_fraction = 0.12 → 
  ∃ P : Parallelepiped, inscribed P S ∧ (∀ v : Vertex, v ∈ vertices P → v.is_white)

def Sphere.surface_area_painted (S : Sphere) : ℝ := sorry
def Sphere.mutually_perpendicular_planes (S : Sphere) : set Plane := sorry
def Sphere.has_symmetric_points (S : Sphere) : Prop := sorry
def Parallelepiped.inscribed (P : Parallelepiped) (S : Sphere) : Prop := sorry
def Vertex.is_white (v : Vertex) : Prop := sorry

theorem exists_white_vertices_parallelepiped_inscribed (S : Sphere) :
  S.surface_area_painted = 0.12 →
  ∃ P : Parallelepiped, P.inscribed S ∧ (∀ v : Vertex, v ∈ P.vertices → v.is_white) := 
sorry

end exists_white_vertices_parallelepiped_inscribed_l126_126369


namespace compound_interest_rate_l126_126693

theorem compound_interest_rate
  (P : ℝ)
  (r : ℝ)
  (A2 A3 : ℝ)
  (h1 : A2 = P * (1 + r) ^ 2)
  (h2 : A3 = P * (1 + r) ^ 3) :
  r = 0.25 :=
by
  -- Setup the problem conditions and simplify the equations
  have h3 : A3 / A2 = (P * (1 + r) ^ 3) / (P * (1 + r) ^ 2),
  sorry.

end compound_interest_rate_l126_126693


namespace counting_positive_integers_l126_126514

theorem counting_positive_integers (n : ℕ) (m : ℕ) :
  ∃ (n_values : Finset ℕ), 
    (∀ n ∈ n_values, n < 50 ∧ (∃ m, m % 5 = 0 ∧ ∃ (α : ℕ), α * (α + 1) = m ∧ n = 2 * α + 1)) ∧
    n_values.card = 5 :=
by 
  let α_values := {4, 9, 14, 19, 24} : Finset ℕ
  let n_values := α_values.image (λ α, 2 * α + 1)
  use n_values
  split
  {
    intros n hn
    simp only [Finset.mem_image, Finset.mem_insert, Finset.mem_singleton] at hn
    rcases hn with ⟨α, hα, rfl⟩
    split
    {
      linarith [hα],
    }
    {
      use α * (α + 1)
      split
      {
        exact Nat.mod_eq_zero_of_dvd (by norm_num [Nat.dvd_mul, hα, add_comm])
      }
      {
        use α
        split
        { refl }
        { refl }
      }
    }
  }
  {
    simp only [α_values], norm_num
  }
  sorry

end counting_positive_integers_l126_126514


namespace sum_of_possible_values_l126_126266

-- Definitions of conditions
def satisfies_equation (x y : ℝ) : Prop :=
  x / (x + y) + y / (2 * (x - y)) = 1

def expression_value (x y : ℝ) : ℝ :=
  (5 * x + y) / (x - 2 * y)

-- The statement of the proof problem
theorem sum_of_possible_values : 
  ∀ x y : ℝ, satisfies_equation x y → 
  (expression_value x y = 5 ∨ expression_value x y = 16)
  → expression_value x y = 21 :=
sorry

end sum_of_possible_values_l126_126266


namespace marseille_hairs_l126_126013

theorem marseille_hairs (N : ℕ) (M : ℕ) (hN : N = 2000000) (hM : M = 300001) :
  ∃ k, k ≥ 7 ∧ ∃ b : ℕ, b ≤ M ∧ b > 0 ∧ ∀ i ≤ M, ∃ l : ℕ, l ≥ k → l ≤ (N / M + 1) :=
by
  sorry

end marseille_hairs_l126_126013


namespace triangle_similarity_l126_126337

variable (ABC : Type) [InnerProductSpace ℝ ABC]

structure Triangle (P Q R : ABC) :=
  (angle_bisector : P -> Prop)
  (on_sides : Q ∈ segment P R ∧ R ∈ segment P Q)

variables {A B C K L T : ABC} {TL BK BL CT : ℝ}

theorem triangle_similarity 
  (hCK : ∃ K, Triangle A B C .angle_bisector C K)
  (hL : L ∈ segment B C)
  (hT : T ∈ segment A C)
  (hCT_BL : dist C T = dist B L)
  (hTL_BK : dist T L = dist B K) :
  similar (Triangle.mk C T L) (Triangle.mk A B C) :=
sorry

end triangle_similarity_l126_126337


namespace point_a_coordinates_l126_126903

open Set

theorem point_a_coordinates (A B : ℝ × ℝ) :
  B = (2, 4) →
  (A.1 = B.1 + 3 ∨ A.1 = B.1 - 3) ∧ A.2 = B.2 →
  dist A B = 3 →
  A = (5, 4) ∨ A = (-1, 4) :=
by
  intros hB hA hDist
  sorry

end point_a_coordinates_l126_126903


namespace dilation_image_l126_126091

-- Define the necessary complex numbers and parameters
def c : ℂ := 1 - 3 * complex.I
def k : ℝ := 3
def z0 : ℂ := -1 + 2 * complex.I
def z : ℂ := -5 + 12 * complex.I

-- State the theorem
theorem dilation_image :
  z = c + k * (z0 - c) :=
sorry

end dilation_image_l126_126091


namespace perimeter_of_rectangular_garden_l126_126017

theorem perimeter_of_rectangular_garden (L W : ℝ) (h : L + W = 28) : 2 * (L + W) = 56 :=
by sorry

end perimeter_of_rectangular_garden_l126_126017


namespace circumscribed_radius_of_triangle_ABC_l126_126760

variable (A B C R : ℝ) (a b c : ℝ)

noncomputable def triangle_ABC (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ B = 2 * A ∧ C = 3 * A

noncomputable def side_length (A a : ℝ) : Prop :=
  a = 6

noncomputable def circumscribed_radius (A B C a R : ℝ) : Prop :=
  2 * R = a / (Real.sin (Real.pi * A / 180))

theorem circumscribed_radius_of_triangle_ABC:
  triangle_ABC A B C →
  side_length A a →
  circumscribed_radius A B C a R →
  R = 6 :=
by
  intros
  sorry

end circumscribed_radius_of_triangle_ABC_l126_126760


namespace volume_of_tetrahedron_ABCD_l126_126899

noncomputable def volume_of_tetrahedron (AB AC BC BD AD CD : ℝ) :=
  sorry

theorem volume_of_tetrahedron_ABCD :
  volume_of_tetrahedron 6 5 7 5 6 7 = 27 := 
  sorry

end volume_of_tetrahedron_ABCD_l126_126899


namespace lines_parallel_or_parallel_l126_126464

variables {a b c d : ℝ³} -- ℝ³ for representing lines in 3D space

-- Defining perpendicularity and parallelism in ℝ³
def perpend (u v : ℝ³) : Prop := u ⬝ v = 0  -- using dot product for perpendicularity
def parallel (u v : ℝ³) : Prop := ∃ k : ℝ, k ≠ 0 ∧ u = k • v  -- scalar multiple for parallelism

-- Given conditions
variable (h1 : perpend a b)
variable (h2 : perpend b c)
variable (h3 : perpend c d)
variable (h4 : perpend d a)

-- Theorem to be proved
theorem lines_parallel_or_parallel :
  parallel b d ∨ parallel a c :=
  sorry

end lines_parallel_or_parallel_l126_126464


namespace yogurt_combinations_l126_126370

theorem yogurt_combinations : (4 * Nat.choose 8 3) = 224 := by
  sorry

end yogurt_combinations_l126_126370


namespace range_of_a_l126_126862

noncomputable def sequence (a : ℝ) (n : ℕ) : ℝ := a * (n:ℝ)^2 + (n:ℝ)

theorem range_of_a (a : ℝ) :
  (sequence a 1 < sequence a 2) ∧
  (sequence a 2 < sequence a 3) ∧
  (sequence a 3 < sequence a 4) ∧
  (sequence a 4 < sequence a 5) ∧
  (sequence a 5 < sequence a 6) ∧
  (∀ n : ℕ, 10 ≤ n → sequence a n > sequence a (n + 1)) →
  -1/12 ≤ a ∧ a ≤ -1/20 :=
by
  sorry

end range_of_a_l126_126862


namespace quadrilateral_is_parallelogram_l126_126560

--  Define points and quadrilateral
variables (A B C D P M N : Type*)

-- Define the convex quadrilateral and midpoints
def convex_quadrilateral (A B C D : Type*) := sorry
def midpoint (M : Type*) (X Y : Type*) := sorry

-- Define lines intersection at point P
def lines_intersect_at (L1 L2 : Type*) (P : Type*) := sorry

-- Define the ratios given in the conditions
def ratio (X1 X2 : Type*) (r : ℚ) := sorry

-- Given conditions
axiom cond1 : convex_quadrilateral A B C D
axiom cond2 : midpoint M B C
axiom cond3 : midpoint N C D
axiom cond4 : lines_intersect_at (line A M) (line B N) P
axiom cond5 : ratio P M (1/5 : ℚ)
axiom cond6 : ratio B P (2/5 : ℚ)

-- Statement to prove
theorem quadrilateral_is_parallelogram
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint M B C)
  (h3 : midpoint N C D)
  (h4 : lines_intersect_at (line A M) (line B N) P)
  (h5 : ratio P M (1 / 5))
  (h6 : ratio B P (2 / 5)) :
  parallelogram A B C D :=
sorry

end quadrilateral_is_parallelogram_l126_126560


namespace range_a_l126_126494

theorem range_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x ≤ 2) → x^2 - 2 * a * x + 1 ≥ 0) → a ≤ 1 :=
by
  sorry

end range_a_l126_126494


namespace tom_robot_counts_l126_126411

-- Define the conditions
def michael_animal_robots : ℕ := 8
def michael_humanoid_robots : ℕ := 12
def michael_vehicle_robots : ℕ := 20

def tom_animal_robots := 2 * michael_animal_robots
def tom_humanoid_robots := 1.5 * michael_humanoid_robots
def tom_vehicle_robots := michael_vehicle_robots / 1.25

-- Define the proof problem
theorem tom_robot_counts :
  tom_animal_robots = 16 ∧
  tom_humanoid_robots = 18 ∧
  tom_vehicle_robots = 16 :=
by {
  -- Formal proof steps would go here
  sorry
}

end tom_robot_counts_l126_126411


namespace volume_of_pyramid_l126_126635

-- Definitions of the given conditions
def parallelogram := sorry   -- This defines our specific parallelogram PABCD
def volume (PABCD : parallelogram) : ℝ := sorry

-- Given side lengths and diagonals
def AB : ℝ := 10
def AD : ℝ := 9
def BD : ℝ := 11

-- Given lengths of the lateral edges
def longer_edge : ℝ := real.sqrt 10

-- The specific pyramid
noncomputable def pyramid_volume := volume parallelogram

-- The theorem we're proving
theorem volume_of_pyramid : pyramid_volume = 200 := 
by
  sorry

end volume_of_pyramid_l126_126635


namespace triangle_area_conditions_l126_126333

open Real

/--
Given a triangle on a coordinate plane such that translations by vectors with integer coordinates do not overlap:
1. Show that the area of the triangle can be greater than 1/2.
2. Prove that the largest possible area of such a triangle is 2/3.
-/
theorem triangle_area_conditions :
  (∃ (vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)), 
    let A := vertices.1,
        B := vertices.2.1,
        C := vertices.2.2 in
    (1/2 < abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) ∧
    (∀ (vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)),
      let A := vertices.1,
          B := vertices.2.1,
          C := vertices.2.2 in
      abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2 ≤ 2/3)) :=
sorry

end triangle_area_conditions_l126_126333


namespace range_of_a_l126_126900

open Real

-- Definitions of conditions
def curve (a : ℝ) : set (ℝ × ℝ) :=
  { p | let (x, y) := p in x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0 }

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem range_of_a (a : ℝ) (h : ∀ p ∈ curve a, in_fourth_quadrant p) : a < -2 :=
sorry

end range_of_a_l126_126900


namespace prod_fraction_lemma_l126_126049

lemma product_fraction (n : ℕ) (h : 1 ≤ n ∧ n ≤ 10) : Real :=
  ∏ n in Finset.range 10, (n + 2) / n

theorem prod_fraction_lemma : 
  ∏ n in Finset.range 10, (n + 2) / n = 132 :=
sorry

end prod_fraction_lemma_l126_126049


namespace average_of_first_two_numbers_l126_126253

theorem average_of_first_two_numbers 
  (numbers : Fin 6 → ℝ)
  (h_sum : (∑ i, numbers i) = 23.4)
  (h_avg_2_4 : (numbers 2 + numbers 3) / 2 = 3.85)
  (h_avg_4_6 : (numbers 4 + numbers 5) / 2 = 4.45) :
  (numbers 0 + numbers 1) / 2 = 3.4 :=
begin
  -- Proof steps to be filled in later
  sorry,
end

end average_of_first_two_numbers_l126_126253


namespace problem_l126_126141

theorem problem (α : ℝ) 
  (h : 4 * sin α - 3 * cos α = 0) : 
  1 / (cos α^2 + 2 * sin (2 * α)) = 25 / 64 :=
begin
  sorry
end

end problem_l126_126141


namespace factorize_expression_l126_126786

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end factorize_expression_l126_126786


namespace directrix_parabola_l126_126992

theorem directrix_parabola (x y : ℝ) :
  (x^2 = (1/4 : ℝ) * y) → (y = -1/16) :=
sorry

end directrix_parabola_l126_126992


namespace laps_needed_to_reach_total_distance_l126_126939

-- Define the known conditions
def total_distance : ℕ := 2400
def lap_length : ℕ := 150
def laps_run_each : ℕ := 6
def total_laps_run : ℕ := 2 * laps_run_each

-- Define the proof goal
theorem laps_needed_to_reach_total_distance :
  (total_distance - total_laps_run * lap_length) / lap_length = 4 :=
by
  sorry

end laps_needed_to_reach_total_distance_l126_126939


namespace inequality_cube_of_greater_l126_126921

variable {a b : ℝ}

theorem inequality_cube_of_greater (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : a^3 > b^3 :=
sorry

end inequality_cube_of_greater_l126_126921


namespace social_gathering_handshakes_l126_126037

/-- 
There are 8 married couples at a social gathering, i.e., 16 people in total. 
Each person shakes hands with every other person except their own spouse. 
We want to prove that the total number of handshakes is 112. 
-/
theorem social_gathering_handshakes : 
  let n := 8 in -- number of couples
  let total_people := 2 * n in -- total number of people
  let handshakes_per_person := total_people - 2 in -- shakes hands with everyone except self and spouse
  (total_people * handshakes_per_person) / 2 = 112 :=
by 
  -- Proof will be provided here
  sorry

end social_gathering_handshakes_l126_126037


namespace ellipse_eccentricity_l126_126542

theorem ellipse_eccentricity (m : ℝ) (e : ℝ) : 
  (∀ x y : ℝ, (x^2 / m) + (y^2 / 4) = 1) ∧ foci_y_axis ∧ e = 1 / 2 → m = 3 :=
by
  sorry

end ellipse_eccentricity_l126_126542


namespace exists_bijection_sequence_divisible_by_9_l126_126463

open List

-- Definition for sequence of n letters chosen from set {A, B, C, D, E, F, G, H, I, J}
variable (n : Nat)
variable (letters : List Char)
namespace DigitsBijection

-- Definition for bijection from {A, B, C, D, E, F, G, H, I, J} ↔ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
variable (bijection : Char → Nat)

-- Conditions of the problem
variable (h_letters : ∀ c ∈ letters, c ∈ ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
variable (h_bijection : Function.Bijective bijection)
variable (no_leading_zero : bijection (letters.head! ≠ 0))

-- Main theorem
theorem exists_bijection_sequence_divisible_by_9 : 
  ∃ bijection : Char → Nat, Function.Bijective bijection ∧
  ∀ c ∈ letters, c ∈ ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] ∧
  bijection (letters.head! ≠ 0) ∧ 
  ((letters.map bijection).foldr (· + ·) 0 % 9 = 0) := sorry

end DigitsBijection

end exists_bijection_sequence_divisible_by_9_l126_126463


namespace problem1_problem2_l126_126129

-- Problem 1: Monotonicity of f(x)
theorem problem1 (k : ℤ) :
    ∀ (x : ℝ),
    ∃ (a : ℝ → ℝ),
    let f := (λ x, 2 * ((1 / 2) - (cos (x / 2))^2) + (sqrt 3) * (sin x)) in
    f(x) = 2 * sin(x - π / 6) ∧ 
    -π / 3 + 2 * k * π ≤ x ∧ x ≤ 2 * π / 3 + 2 * k * π → 
    (f (x + ε) - f(x)) / ε > 0
  sorry

-- Problem 2: Range of m for g(x)
theorem problem2 :
  ∀ (m : ℝ), 
  ∃ (g : ℝ → ℝ), 
  let g := (λ x, 2 * sin(2 * x - π / 6)) in
  (m ∈ [-1, 2]) → 
  ∃ x, 0 ≤ x ∧ x ≤ π / 2 ∧ g(x) - m = 0
  sorry

end problem1_problem2_l126_126129


namespace inequality_proved_l126_126246

theorem inequality_proved (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_proved_l126_126246


namespace factorize_expr_l126_126782

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end factorize_expr_l126_126782


namespace minimum_value_of_f_l126_126931

def f (x : ℝ) : ℝ := Real.exp x - Real.exp (2 * x)

theorem minimum_value_of_f :
  ∃ x, ∀ y, f(x) ≤ f(y) ∧ f(x) = -Real.exp 2 := by
  sorry

end minimum_value_of_f_l126_126931


namespace sum_of_roots_of_polynomial_l126_126660

theorem sum_of_roots_of_polynomial :
  let p := (3 : ℚ) * (Polynomial.X ^ 4) - (6 : ℚ) * (Polynomial.X ^ 3) - (17 : ℚ) * (Polynomial.X ^ 2) + (6 : ℚ) * (Polynomial.X) - (2 : ℚ),
  Polynomial.roots_sum p = 2 := 
by
  sorry

end sum_of_roots_of_polynomial_l126_126660


namespace correct_operation_l126_126687

theorem correct_operation (a : ℝ) : (-a^3)^4 = a^12 :=
by sorry

end correct_operation_l126_126687


namespace sum_of_abcd_l126_126920

variable (a b c d : ℚ)

def condition (x : ℚ) : Prop :=
  x = a + 3 ∧
  x = b + 7 ∧
  x = c + 5 ∧
  x = d + 9 ∧
  x = a + b + c + d + 13

theorem sum_of_abcd (x : ℚ) (h : condition a b c d x) : a + b + c + d = -28 / 3 := 
by sorry

end sum_of_abcd_l126_126920


namespace tetrahedron_parallel_projection_l126_126405

theorem tetrahedron_parallel_projection (A B C D : ℝ × ℝ × ℝ) (plane_alpha : ℝ × ℝ × ℝ → Prop)
  (h_tetrahedron : is_regular_tetrahedron D A B C)
  (h_planes_parallel : are_parallel (line_through A B) plane_alpha ∧ are_parallel (line_through C D) plane_alpha)
  (h_skew: skew_lines A B C D)
  (h_perpendicular: are_perpendicular (line_through A B) (line_through C D))
  (h_equal_length: length A B = length C D) :
  ∃ (shape : Type), parallel_projection shape plane_alpha D A B C :=
sorry

end tetrahedron_parallel_projection_l126_126405


namespace compare_fractions_l126_126791

theorem compare_fractions :
  (1 / (Real.sqrt 5 - 2) < 1 / (Real.sqrt 6 - Real.sqrt 5)) :=
  have h1 : 1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := sorry,
  have h2 : 1 / (Real.sqrt 6 - Real.sqrt 5) = Real.sqrt 6 + Real.sqrt 5 := sorry,
  have h3 : 2 = Real.sqrt 4 ∧ Real.sqrt 4 < Real.sqrt 6 := sorry,
  sorry

end compare_fractions_l126_126791


namespace correct_calculation_l126_126686

theorem correct_calculation (x y : ℝ) :
  ¬ ((x + y)^2 = x^2 + y^2) ∧
  ¬ ((x^2)^3 = x^5) ∧
  (x^2 * x^3 = x^5) ∧
  ¬ (4 * x^2 - y^2 = (4 * x + y) * (4 * x - y)) :=
by
  intro h
  cases h with h1 h2h3h4
  cases h2h3h4 with h2 h3h4
  cases h3h4 with h3 h4
  sorry

end correct_calculation_l126_126686


namespace volume_of_rectangular_prism_l126_126979

-- Define the dimensions a, b, c as non-negative real numbers
variables (a b c : ℝ)

-- Given conditions
def condition_1 := a * b = 30
def condition_2 := a * c = 50
def condition_3 := b * c = 75

-- The theorem statement
theorem volume_of_rectangular_prism :
  (a * b * c) = 335 :=
by
  -- Assume the given conditions
  assume h1 : condition_1 a b,
  assume h2 : condition_2 a c,
  assume h3 : condition_3 b c,
  -- Proof skipped
  sorry

end volume_of_rectangular_prism_l126_126979


namespace smallest_positive_period_of_f_max_min_values_of_f_in_interval_l126_126483

-- Definitions
def f (x : ℝ) : ℝ := 2 * cos x * sin (x + π / 6) + (cos x) ^ 4 - (sin x) ^ 4

-- Theorem for smallest positive period
theorem smallest_positive_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = π :=
sorry

-- Theorem for maximum and minimum values within a specified interval
theorem max_min_values_of_f_in_interval :
  (∀ x ∈ Icc (-π / 12) (π / 6), f (-π / 12) ≤ f x ∧ f x ≤ f (π / 6)) ∧
  f (-π / 12) = (1 - sqrt 3) / 2 ∧
  f (π / 6) = sqrt 3 + 1 / 2 :=
sorry

end smallest_positive_period_of_f_max_min_values_of_f_in_interval_l126_126483


namespace equal_length_if_and_only_if_l126_126895

variables (A B C D E F G P Q : Point)
variables (O : Circle)
variables [AcuteAngledTriangle (Triangle A B C)]
variables (tangent_at_A : Tangent O A D)
variables (E_on_DA : Collinear E D A)
variables (F_on_minor_arc_BC : Arc F)
variables (G_on_minor_arc_AB_inter_EF : IntersectArcLine G (Arc A B) (Line E F))
variables (P_inter_FB_tangent_at_A : IntersectLineTangent P (Line F B) (Tangent O A D))
variables (Q_inter_GC_tangent_at_A : IntersectLineTangent Q (Line G C) (Tangent O A D))

theorem equal_length_if_and_only_if :
  EuclideanGeometry.dist A D = EuclideanGeometry.dist A E ↔
  EuclideanGeometry.dist A P = EuclideanGeometry.dist A Q :=
sorry

end equal_length_if_and_only_if_l126_126895


namespace candy_bag_division_l126_126443

theorem candy_bag_division (total_candy bags_candy : ℕ) (h1 : total_candy = 42) (h2 : bags_candy = 21) : 
  total_candy / bags_candy = 2 := 
by
  sorry

end candy_bag_division_l126_126443


namespace correct_statements_l126_126856

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) * cos (ω * x) - sqrt 3 * sin (ω * x) ^ 2

theorem correct_statements (ω : ℝ) [hω : Fact (ω > 0)] :
  (∀ k : ℤ, -5 * π / 12 + 2 * k * π ≠ π / 12 + 2 * k * π) ∧
  (∀ x : ℝ, f 1 x = sin (2 * x + π / 3) - sqrt 3 / 2) ∧
  (∀ x : ℝ, f 1 x = sin (2 * (x + π / 12) + π / 3) - sqrt 3 / 2 ↔ f 1 (-x) = sin (-2 * x + π / 2) - sqrt 3 / 2) ∧
  (∀ x : ℝ, f 1 (π / 3 + x) + f 1 (π / 3 - x) = -sqrt 3) :=
by {
  sorry -- proof not required.
}

end correct_statements_l126_126856


namespace value_bounds_of_expression_l126_126207

theorem value_bounds_of_expression
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (triangle_ineq1 : a + b > c)
  (triangle_ineq2 : a + c > b)
  (triangle_ineq3 : b + c > a)
  : 4 ≤ (a+b+c)^2 / (b*c) ∧ (a+b+c)^2 / (b*c) ≤ 9 := sorry

end value_bounds_of_expression_l126_126207


namespace garden_perimeter_is_56_l126_126679

-- Define the conditions
def width_garden : ℕ := 16            -- Garden's width
def length_playground : ℕ := 16       -- Playground's length
def width_playground : ℕ := 12        -- Playground's width
def area_playground : ℕ := length_playground * width_playground
def length_garden : ℕ := area_playground / width_garden

-- The proof problem statement
theorem garden_perimeter_is_56 (width_garden length_playground width_playground : ℕ) (h1 : width_garden = 16) (h2 : length_playground = 16) (h3 : width_playground = 12) :
  let area_playground := length_playground * width_playground in
  let length_garden := area_playground / width_garden in
  2 * (length_garden + width_garden) = 56 :=
by
  sorry

end garden_perimeter_is_56_l126_126679


namespace total_employees_l126_126638

-- Definitions based on the conditions:
variables (N S : ℕ)
axiom condition1 : 75 % 100 * S = 75 / 100 * S
axiom condition2 : 65 % 100 * S = 65 / 100 * S
axiom condition3 : N - S = 40
axiom condition4 : 5 % 6 * N = 5 / 6 * N

-- The statement to be proven:
theorem total_employees (N S : ℕ)
    (h1 : 75 % 100 * S = 75 / 100 * S)
    (h2 : 65 % 100 * S = 65 / 100 * S)
    (h3 : N - S = 40)
    (h4 : 5 % 6 * N = 5 / 6 * N)
    : N = 240 :=
sorry

end total_employees_l126_126638


namespace first_tap_fills_cistern_in_4_hours_l126_126706

theorem first_tap_fills_cistern_in_4_hours :
  (∃ T : ℝ, (1/T - 1/9 = 1/7.2) ∧ T = 4) :=
begin
  have h : ∃ T : ℝ, 1/T - 1/9 = 1/7.2,
  { 
    -- Placeholder for the proof that 1/T - 1/9 = 1/7.2
    sorry
  },
  cases h with T hT,
  use T,
  split,
  {
    exact hT
  },
  {
    rw ←hT,
    -- Placeholder for algebraic manipulation to show T = 4
    sorry
  }
end

end first_tap_fills_cistern_in_4_hours_l126_126706


namespace tan_alpha_l126_126142

variable (α : ℝ)
variable (H_cos : Real.cos α = 12/13)
variable (H_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)

theorem tan_alpha :
  Real.tan α = -5/12 :=
sorry

end tan_alpha_l126_126142


namespace line_intersects_circle_l126_126849

theorem line_intersects_circle (r d : ℝ) (hr : r = 5) (hd : d = 3 * Real.sqrt 2) : d < r :=
by
  rw [hr, hd]
  exact sorry

end line_intersects_circle_l126_126849


namespace sum_nth_consecutive_integers_median_25_l126_126553

theorem sum_nth_consecutive_integers_median_25 (L : List ℤ) (n : ℕ) :
  (∃ m : ℤ, L = List.range (m - (List.length L) / 2) (List.length L) ∧ L.nth ((List.length L) / 2) = some 25) →
  (L.nth n + L.nth (List.length L - 1 - n)) = 50 := by
  sorry

end sum_nth_consecutive_integers_median_25_l126_126553


namespace amount_in_cup_after_division_l126_126255

theorem amount_in_cup_after_division (removed remaining cups : ℕ) (h : remaining + removed = 40) : 
  (40 / cups = 8) :=
by
  sorry

end amount_in_cup_after_division_l126_126255


namespace total_pairs_of_shoes_equivalence_l126_126617

variable (Scott Anthony Jim Melissa Tim: ℕ)

theorem total_pairs_of_shoes_equivalence
    (h1 : Scott = 7)
    (h2 : Anthony = 3 * Scott)
    (h3 : Jim = Anthony - 2)
    (h4 : Jim = 2 * Melissa)
    (h5 : Tim = (Anthony + Melissa) / 2):

  Scott + Anthony + Jim + Melissa + Tim = 71 :=
  by
  sorry

end total_pairs_of_shoes_equivalence_l126_126617


namespace smallest_a_for_f_iter_3_l126_126595

def f (x : Int) : Int :=
  if x % 4 = 0 ∧ x % 9 = 0 then x / 36
  else if x % 9 = 0 then 4 * x
  else if x % 4 = 0 then 9 * x
  else x + 4

def f_iter (f : Int → Int) (a : Nat) (x : Int) : Int :=
  if a = 0 then x else f_iter f (a - 1) (f x)

theorem smallest_a_for_f_iter_3 (a : Nat) (h : a > 1) : 
  (∀b, b > 1 → b < a → f_iter f b 3 ≠ f 3) ∧ f_iter f a 3 = f 3 ↔ a = 9 := 
  by
  sorry

end smallest_a_for_f_iter_3_l126_126595


namespace B_finishes_alone_in_27_days_l126_126009

-- Definitions for conditions
def isHalfAsGood (A B: Type) (x: ℕ) := 
  ∀ A B : Type, A = 2 * B

def togetherFinishIn18Days (A B: Type) (x: ℕ) := 
  ∀ (x: ℕ), (1/x + 1/(2*x)) = 1/18

-- The main theorem to prove
theorem B_finishes_alone_in_27_days :
  ∀ (A B: Type) (x: ℕ), isHalfAsGood A B x ∧ togetherFinishIn18Days A B x → x = 27 :=
by 
  intros A B x h,
  cases h with h1 h2,
  sorry

end B_finishes_alone_in_27_days_l126_126009


namespace part1_part2_part3_l126_126499

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ)

axiom a_recurrence : ∀ n ≥ 2, 4 * a n = a (n - 1) - 3
axiom a_initial : a 1 = -3/4
axiom b_definition : ∀ n, b n + 2 = 3 * (Real.log (a n + 1) / Real.log (1/4))
axiom c_definition : ∀ n, c n = (a n + 1) * b n

theorem part1:
  ∀ n, a n = (1/4 : ℝ)^n - 1 :=
sorry

theorem part2 (S : ℕ → ℝ) :
  S = λ n, ∑ k in Finset.range n, c k →
  ∀ n, S n = (2/3 : ℝ) - (3 * n + 2)/3 * (1/4 : ℝ)^n :=
sorry

theorem part3 :
  ∀ n t, 0 ≤ t ∧ t ≤ 1 → (c n ≤ t * m^2 - m - 1/2) → m ≤ -3/4 :=
sorry

end part1_part2_part3_l126_126499


namespace solve_for_x_l126_126625

theorem solve_for_x (x : ℝ) (h : (16^x) * (16^x) * (16^x) = 256^3) : x = 2 := 
by
  -- proof skipped
  sorry

end solve_for_x_l126_126625


namespace bijection_gcd_l126_126916

open Set

/-- Definition of the problem in Lean 4 -/
theorem bijection_gcd (D : Set ℕ) (hD : D.Nonempty) (d : ℕ) (hd : d = finset.gcd D.to_finset id) :
  ∃ (f : ℤ → ℤ), bijective f ∧ ∀ n : ℤ, ∃ k ∈ D, |f(n + 1) - f n| = d * k := 
sorry

end bijection_gcd_l126_126916


namespace cubefunc_value_l126_126147

theorem cubefunc_value (x : ℝ) (h : 4^(3*x) + 64 = 68 * 4^x) : x^3 + 2 = 29 := by
  sorry

end cubefunc_value_l126_126147


namespace find_cubic_polynomial_l126_126763

noncomputable def f : Polynomial ℚ := Polynomial.mk [4, 3, -2, 1]

theorem find_cubic_polynomial (r : ℚ) (h : f.eval r = 0) :
  ∃ b c d, (r^3 - 2 * r^2 + 3 * r + 4 = 0) → (let s := r^2 in
  Polynomial.eval₂ (λ x : ℚ, s) g = 0) ∧ b = -4 ∧ c = -7 ∧ d = 16 := sorry

end find_cubic_polynomial_l126_126763


namespace AM_GM_inequality_example_l126_126808

open Real

theorem AM_GM_inequality_example 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ((a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2)) ≥ 9 * (a^2 * b^2 * c^2) :=
sorry

end AM_GM_inequality_example_l126_126808


namespace exp_25_pi_i_div_2_to_rectangular_l126_126397

noncomputable def exp_rectangular_form : ℂ :=
complex.exp (25 * real.pi * complex.I / 2)

theorem exp_25_pi_i_div_2_to_rectangular :
  exp_rectangular_form = complex.I :=
by
  sorry

end exp_25_pi_i_div_2_to_rectangular_l126_126397


namespace cost_of_one_pencil_l126_126891

theorem cost_of_one_pencil (students : ℕ) (more_than_half : ℕ) (pencil_cost : ℕ) (pencils_each : ℕ)
  (total_cost : ℕ) (students_condition : students = 36) 
  (more_than_half_condition : more_than_half > 18) 
  (pencil_count_condition : pencils_each > 1) 
  (cost_condition : pencil_cost > pencils_each) 
  (total_cost_condition : students * pencil_cost * pencils_each = 1881) : 
  pencil_cost = 17 :=
sorry

end cost_of_one_pencil_l126_126891


namespace relative_order_exponents_log_l126_126063

open Real

/-- The theorem states the relative order of 0.7^6, 6^0.7, and log_{0.7}6. -/
theorem relative_order_exponents_log : ∀  (a b : ℝ), (0.7 < 1) → (6 > 1) → 
(log a 6 < a ^ 6) ∧ (a ^ 6 < 6 ^ b) := 
by
  intros a b h1 h2
  sorry

end relative_order_exponents_log_l126_126063


namespace sum_powers_of_i_l126_126622

theorem sum_powers_of_i : ∑ k in (Finset.range 2010), (Complex.I ^ k) = 1 + Complex.I :=
by
  -- proof skipped
  sorry

end sum_powers_of_i_l126_126622


namespace find_incorrect_statement_l126_126006

def scores : List ℕ := [85, 95, 85, 80, 80, 85]

def mode (l : List ℕ) : ℕ :=
  (l.groupBy id).values.maxBy (List.length ∘ lean.Array.mkArray).head.getD 0

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  if h : sorted.length % 2 = 1 then
    sorted.get! (sorted.length / 2)
  else
    let mid := sorted.length / 2
    (sorted.get! (mid - 1) + sorted.get! mid) / 2

def range (l : List ℕ) : ℕ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem find_incorrect_statement : 
  (mode scores = 85) ∧ 
  (mean scores = 85) ∧ 
  (median scores ≠ 80) ∧ 
  (range scores = 15) → 
  (∃ incorrect_statement ∈ ["The mode is 85", "The mean is 85", "The median is 80", "The range is 15"], 
  incorrect_statement = "The median is 80") :=
by 
  sorry

end find_incorrect_statement_l126_126006


namespace planted_fraction_correct_l126_126419

-- Define the vertices of the triangle
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (5, 0)
def C : (ℝ × ℝ) := (0, 12)

-- Define the length of the legs
def leg1 := 5
def leg2 := 12

-- Define the shortest distance from the square to the hypotenuse
def distance_to_hypotenuse := 3

-- Define the area of the triangle
def triangle_area := (1 / 2) * (leg1 * leg2)

-- Assume the side length of the square
def s := 6 / 13

-- Define the area of the square
def square_area := s^2

-- Define the fraction of the field that is unplanted
def unplanted_fraction := square_area / triangle_area

-- Define the fraction of the field that is planted
def planted_fraction := 1 - unplanted_fraction

theorem planted_fraction_correct :
  planted_fraction = 5034 / 5070 :=
sorry

end planted_fraction_correct_l126_126419


namespace prob_condition_sum_of_k_l126_126803

open Nat

def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then
    nat.choose n k
  else
    0

lemma pascal_identity :
  binom 25 6 + binom 25 7 = binom 26 7 :=
begin
  rw [binom, if_pos (nat.le_refl 25)],
  rw [binom, if_pos (nat.le_refl 25)],
  rw nat.choose_succ_succ,
  exact rfl,
end

theorem prob_condition (k : ℕ) :
  binom 25 6 + binom 25 7 = binom 26 k → k = 7 ∨ k = 19 :=
begin
  intro h,
  rw pascal_identity at h,
  have h_k : binom 26 k = binom 26 7 := h,
  rw nat.choose_eq_choose_symm at h_k,
  cases h_k,
  { exact or.inl h_k },
  { exact or.inr h_k }
end

theorem sum_of_k :
  ∑ k in {k ∈ Finset.range 27 | binom 25 6 + binom 25 7 = binom 26 k}, k = 26 :=
begin
  apply Finset.sum_eq_of_subset,
  { exact λ x hx, by_cases h:(binom 25 6 + binom 25 7 = binom 26 x) → x ∈ {7, 19},
      { exact h },
      { exfalso, simpa using h } },
  { exact λ x hx, or.inl hx }
end

end prob_condition_sum_of_k_l126_126803


namespace area_of_triangle_AFE_l126_126497

noncomputable def par := {x y : ℝ // y^2 = 4 * x}

def focus_par : par := ⟨1, 0, by norm_num⟩

def directrix_par (p : par) : Prop :=
  ∃ y : ℝ, p.1 = -1 ∧ p.2 = y ∧ p.2 = 2 * y

def in_first_quadrant (p : par) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def angle_AFE (A F E : par) : Prop :=
  let angle_rad := Real.pi / 3 in
  let calculated_angle := Real.atan2 (E.2 - F.2) (E.1 - F.1) - Real.atan2 (A.2 - F.2) (A.1 - F.1) in
  abs (calculated_angle - angle_rad) < 1e-9

def area_AFE (A F E : par) : ℝ :=
  (1 / 2) * abs (F.1 * (A.2 - E.2) + A.1 * (E.2 - F.2) + E.1 * (F.2 - A.2))

theorem area_of_triangle_AFE :
  ∀ (A F E : par), 
  directrix_par E → 
  in_first_quadrant A → 
  angle_AFE A F E → 
  A = ⟨3, 2 * Real.sqrt 3, by norm_num⟩ → 
  F = ⟨1, 0, by norm_num⟩ → 
  area_AFE A F E = 4 * Real.sqrt 3 :=
by 
  intros A F E dir_E fst_quad_A ang_AFE hA hF
  sorry

end area_of_triangle_AFE_l126_126497


namespace x_eq_1_sufficient_not_necessary_l126_126060

theorem x_eq_1_sufficient_not_necessary (x : ℝ) : 
    (x = 1 → (x^2 - 3 * x + 2 = 0)) ∧ ¬((x^2 - 3 * x + 2 = 0) → (x = 1)) := 
by
  sorry

end x_eq_1_sufficient_not_necessary_l126_126060


namespace find_deducted_salary_l126_126602

noncomputable def dailyWage (weeklySalary : ℝ) (workingDays : ℕ) : ℝ := weeklySalary / workingDays

noncomputable def totalDeduction (dailyWage : ℝ) (absentDays : ℕ) : ℝ := dailyWage * absentDays

noncomputable def deductedSalary (weeklySalary : ℝ) (totalDeduction : ℝ) : ℝ := weeklySalary - totalDeduction

theorem find_deducted_salary
  (weeklySalary : ℝ := 791)
  (workingDays : ℕ := 5)
  (absentDays : ℕ := 4)
  (dW := dailyWage weeklySalary workingDays)
  (tD := totalDeduction dW absentDays)
  (dS := deductedSalary weeklySalary tD) :
  dS = 158.20 := 
  by
    sorry

end find_deducted_salary_l126_126602


namespace weight_of_11th_person_l126_126283

theorem weight_of_11th_person
  (n : ℕ) (avg1 avg2 : ℝ)
  (hn : n = 10)
  (havg1 : avg1 = 165)
  (havg2 : avg2 = 170)
  (W : ℝ) (X : ℝ)
  (hw : W = n * avg1)
  (havg2_eq : (W + X) / (n + 1) = avg2) :
  X = 220 :=
by
  sorry

end weight_of_11th_person_l126_126283


namespace percent_problem_l126_126533

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end percent_problem_l126_126533


namespace find_m_l126_126852

def complex_number_Z : ℂ := (4 + 2 * Complex.i) / ((1 + Complex.i)^2)

theorem find_m :
  let Z : ℂ := complex_number_Z
  let point := (1 : ℤ, -2 : ℤ)
  ∃ m : ℤ, (point.1 - 2 * point.2 + m = 0) ∧ Z = complex_number_Z → m = -5 :=
by
  let Z := complex_number_Z
  let point := (1, -2)
  use -5
  split
  show point.1 - 2 * point.2 + -5 = 0
  sorry
  show Z = complex_number_Z
  sorry

end find_m_l126_126852


namespace samuel_faster_l126_126959

theorem samuel_faster (S T_h : ℝ) (hT_h : T_h = 1.3) (hS : S = 30) :
  (T_h * 60) - S = 48 :=
by
  sorry

end samuel_faster_l126_126959


namespace students_passed_in_both_subjects_l126_126555

theorem students_passed_in_both_subjects:
  ∀ (F_H F_E F_HE : ℝ), F_H = 0.30 → F_E = 0.42 → F_HE = 0.28 → (1 - (F_H + F_E - F_HE)) = 0.56 :=
by
  intros F_H F_E F_HE h1 h2 h3
  sorry

end students_passed_in_both_subjects_l126_126555


namespace complex_line_segment_properties_l126_126052

noncomputable def midpoint (z1 z2 : ℂ) : ℂ :=
  (z1 + z2) / 2

noncomputable def distance_from_origin (z : ℂ) : ℝ :=
  complex.abs z

theorem complex_line_segment_properties :
  let z1 := -15 + 5 * complex.I
  let z2 := 5 - 15 * complex.I
  let mid := midpoint z1 z2
  (mid = -5 - 5 * complex.I) ∧ (distance_from_origin mid = 5 * real.sqrt 2) :=
by
  -- Definitions
  let z1 := -15 + 5 * complex.I
  let z2 := 5 - 15 * complex.I
  let mid := midpoint z1 z2
  -- Statements
  have midpoint_correct : mid = -5 - 5 * complex.I := sorry
  have distance_correct : distance_from_origin mid = 5 * real.sqrt 2 := sorry
  exact ⟨midpoint_correct, distance_correct⟩

end complex_line_segment_properties_l126_126052


namespace area_of_region_S_l126_126249

-- Definition of the square and region S inside the Lean code
variables (M N P Q : Type) [point_space MNPQ : MetricSpace] (MNPQ : Square M N P Q)
variable (side_length : MNPQ.side_length = 4)
variable (S : Region)
variable (closer_to_M : ∀ p ∈ S, dist p M < min (dist p N) (dist p P) (dist p Q))

-- Prove that the area of region S is 2
theorem area_of_region_S : area S = 2 :=
sorry

end area_of_region_S_l126_126249


namespace right_angle_triangle_lines_l126_126124

theorem right_angle_triangle_lines {m : ℝ} :
  (∃ a b c d e f : ℝ, a = 3 ∧ b = 2 ∧ c = 6 ∧ d = 2 ∧ e = -3 * m^2 ∧ f = 18 ∧ 
  a * d + b * e = 0) ∧
  (∃ g h i k l : ℝ, g = 2 * m ∧ h = -3 ∧ i = 12 ∧ j = 2 ∧ l = 18 ∧ 
  g * a + h * b = 0)  ∧
  (∃ p q r s t u : ℝ, p = 2 ∧ q = -3 * m^2 ∧ r = 18 ∧ s = 2 * m ∧ t = -3 ∧ u = 12 ∧ 
  p * s + q * t = 0) :=
  m = 0 ∨ m = -1 ∨ m = -4 / 9 :=
sorry

end right_angle_triangle_lines_l126_126124


namespace joan_total_socks_l126_126198

theorem joan_total_socks (n : ℕ) (h1 : n / 3 = 60) : n = 180 :=
by
  -- Proof goes here
  sorry

end joan_total_socks_l126_126198


namespace no_nat_with_odd_even_divisors_l126_126191

theorem no_nat_with_odd_even_divisors:
  ¬∃ n : ℕ, 
    (∃ even_divisor_odd_count : ℕ, 
      even_divisor_odd_count % 2 = 1 ∧ 
      even_divisor_odd_count = (d : finset (ℕ)) → d ∣ n ∧ d % 2 = 0 ∧ 
      count {d | d ∣ n ∧ d % 2 = 0} = even_divisor_odd_count) ∧ 
    (∃ odd_divisor_even_count : ℕ, 
      odd_divisor_even_count % 2 = 0 ∧ 
      odd_divisor_even_count = (d : finset (ℕ)) → d ∣ n ∧ d % 2 ≠ 0 ∧ 
      count {d | d ∣ n ∧ d % 2 ≠ 0} = odd_divisor_even_count) := 
begin
  sorry
end

end no_nat_with_odd_even_divisors_l126_126191


namespace correct_hyperbola_l126_126030

-- Define the given functions
def f1 (x : ℝ) : ℝ := -2 * x
def f2 (x : ℝ) : ℝ := -4 / x
def f3 (x : ℝ) : ℝ := -8 / x
def f4 (x : ℝ) : ℝ := x - 6

-- Define the conditions
def passes_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop := f p.1 = p.2

-- The point in question
def point : ℝ × ℝ := (2, -4)

-- The theorem statement
theorem correct_hyperbola : 
  passes_through f3 point :=
by
  unfold passes_through
  simp
  sorry

end correct_hyperbola_l126_126030


namespace lions_after_one_year_l126_126664

def initial_lions : ℕ := 100
def birth_rate : ℕ := 5
def death_rate : ℕ := 1
def months_in_year : ℕ := 12

theorem lions_after_one_year : 
  initial_lions + (birth_rate * months_in_year) - (death_rate * months_in_year) = 148 :=
by
  sorry

end lions_after_one_year_l126_126664


namespace sum_of_valid_integers_k_l126_126802

theorem sum_of_valid_integers_k :
  let valid_k_values := {k : ℕ | k = 7 ∨ k = 19} in
  (∑ k in valid_k_values, k) = 26 :=
by
  sorry

end sum_of_valid_integers_k_l126_126802


namespace sin_theta_given_point_l126_126545

theorem sin_theta_given_point (x y : ℝ) (h1 : x = - √3 / 2) (h2 : y = 1 / 2)
  (h3 : x^2 + y^2 = 1) : sin (real.arctan y x) = 1 / 2 := 
sorry

end sin_theta_given_point_l126_126545


namespace log_2_sufficient_but_not_necessary_l126_126821

theorem log_2_sufficient_but_not_necessary (x : ℝ) :
  (log 2 x < 0) → ((1 / 2)^(x - 1) > 1) :=
sorry

end log_2_sufficient_but_not_necessary_l126_126821


namespace common_sum_is_10_l126_126972

-- Define the problem conditions
def integers_from_neg5_to_10 : List Int :=
  [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_4x4_matrix (lst : List Int) : Prop :=
  lst.length = 16

def has_equal_sums (matrix : List (List Int)) : Prop :=
  ∀ i, (i < 4 → matrix.map (λ row, row.nth i).sum = 10) ∧
       (matrix.nth i).sum = 10 ∧
       (matrix.nth i).sum = List.range 4.map (λ x, matrix.nth x.i).sum)

-- Translate problem to Lean statement
theorem common_sum_is_10
  (matrix : List (List Int))
  (H₁ : integers_from_neg5_to_10 ⊂ matrix.join)
  (H₂ : is_4x4_matrix matrix.join)
  (H₃ : has_equal_sums matrix) : 
  ∃ s, s = 10 :=
by
  sorry

end common_sum_is_10_l126_126972


namespace sum_of_tangents_l126_126394

noncomputable def f (x : ℝ) : ℝ :=
  max (-7 * x - 57) (max (5 * x + 11) (2 * x - 8))

theorem sum_of_tangents :
  ∃ x₄ x₅ x₆ : ℝ, 
    (∀ x, polynomial.eval (x - x₄) (polynomial.C 1) +
          polynomial.eval (x - x₅) (polynomial.C 1) +
          polynomial.eval (x - x₆) (polynomial.C 1) = f x) →
    x₄ + x₅ + x₆ = -117 / 17 :=
sorry

end sum_of_tangents_l126_126394


namespace hannah_spent_65_l126_126131

-- Definitions based on the conditions
def sweatshirts_count : ℕ := 3
def t_shirts_count : ℕ := 2
def sweatshirt_cost : ℕ := 15
def t_shirt_cost : ℕ := 10

-- The total amount spent
def total_spent : ℕ := sweatshirts_count * sweatshirt_cost + t_shirts_count * t_shirt_cost

-- The theorem stating the problem
theorem hannah_spent_65 : total_spent = 65 :=
by
  sorry

end hannah_spent_65_l126_126131


namespace range_h_l126_126114

noncomputable def f (x : ℝ) : ℝ := (√3) * Real.sin (2 * x) + 2 * (Real.cos x)^2 - 1

def M_t (t : ℝ) : ℝ := Real.sup (Set.image f (Set.Icc t (t + (Real.pi / 4))))
def m_t (t : ℝ) : ℝ := Real.inf (Set.image f (Set.Icc t (t + (Real.pi / 4))))
def h (t : ℝ) : ℝ := M_t t - m_t t

theorem range_h (t : ℝ) (ht : t ∈ Set.Icc (Real.pi / 12) (5 * Real.pi / 12)) :
  1 ≤ h t ∧ h t ≤ 2 * √2 := 
sorry

end range_h_l126_126114


namespace find_x_l126_126123

def vector_a (x : ℝ) : ℝ × ℝ := (x, 2)
def vector_b : ℝ × ℝ := (1, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

def projection (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

theorem find_x (x : ℝ) (h : projection (vector_a x) vector_b = Real.sqrt 2) : x = 4 := 
  sorry

end find_x_l126_126123


namespace number_of_white_balls_l126_126172

-- Definition of the conditions
def total_balls : ℕ := 40
def prob_red : ℝ := 0.15
def prob_black : ℝ := 0.45
def prob_white := 1 - prob_red - prob_black

-- The statement that needs to be proved
theorem number_of_white_balls : (total_balls : ℝ) * prob_white = 16 :=
by
  sorry

end number_of_white_balls_l126_126172


namespace triangle_perimeter_l126_126547

theorem triangle_perimeter (A B C : Type) 
  (x : ℝ) 
  (a b c : ℝ) 
  (h₁ : a = x + 1) 
  (h₂ : b = x) 
  (h₃ : c = x - 1) 
  (α β γ : ℝ) 
  (angle_condition : α = 2 * γ) 
  (law_of_sines : a / Real.sin α = c / Real.sin γ)
  (law_of_cosines : Real.cos γ = ((a^2 + b^2 - c^2) / (2 * b * a))) :
  a + b + c = 15 :=
  by
  sorry

end triangle_perimeter_l126_126547


namespace wallet_cost_l126_126578

-- Define the conditions from the problem
variables (W : ℕ) -- Wallet cost
variables (L_Sneakers_Cost : ℕ := 200) -- Leonard's sneakers total cost
variables (M_Backpack_Jeans_Cost : ℕ := 200) -- Michael's backpack and jeans total cost
variables (Total_Spending : ℕ := 450) -- Total combined spending

-- The theorem we want to prove
theorem wallet_cost :
  W + L_Sneakers_Cost + M_Backpack_Jeans_Cost = Total_Spending → W = 50 :=
by
  intro H,
  have H1 : W + 400 = 450 := by rw [H, L_Sneakers_Cost, M_Backpack_Jeans_Cost],
  exact Nat.sub_eq_of_eq_add' H1.symm

end wallet_cost_l126_126578


namespace final_amount_in_account_l126_126671

noncomputable def initial_deposit : ℝ := 1000
noncomputable def first_year_interest_rate : ℝ := 0.2
noncomputable def first_year_balance : ℝ := initial_deposit * (1 + first_year_interest_rate)
noncomputable def withdrawal_amount : ℝ := first_year_balance / 2
noncomputable def after_withdrawal_balance : ℝ := first_year_balance - withdrawal_amount
noncomputable def second_year_interest_rate : ℝ := 0.15
noncomputable def final_balance : ℝ := after_withdrawal_balance * (1 + second_year_interest_rate)

theorem final_amount_in_account : final_balance = 690 := by
  sorry

end final_amount_in_account_l126_126671


namespace sqrt_sum_bounds_l126_126593

open Real

theorem sqrt_sum_bounds (a b c d e : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                                      (hb : 0 ≤ b) (hb1 : b ≤ 1)
                                      (hc : 0 ≤ c) (hc1 : c ≤ 1)
                                      (hd : 0 ≤ d) (hd1 : d ≤ 1)
                                      (he : 0 ≤ e) (he1 : e ≤ 1) :
  2 * sqrt 2 ≤ sqrt (a^2 + (1 - b)^2) + sqrt (b^2 + (1 - c)^2) + sqrt (c^2 + (1 - d)^2) + sqrt (d^2 + (1 - e)^2) + sqrt (e^2 + (1 - a)^2)
  ∧ sqrt (a^2 + (1 - b)^2) + sqrt (b^2 + (1 - c)^2) + sqrt (c^2 + (1 - d)^2) + sqrt (d^2 + (1 - e)^2) + sqrt (e^2 + (1 - a)^2) ≤ 5 :=
begin
  sorry
end

end sqrt_sum_bounds_l126_126593


namespace average_remaining_two_numbers_l126_126981

theorem average_remaining_two_numbers 
  (h1 : (40.5 : ℝ) = 10 * 4.05)
  (h2 : (11.1 : ℝ) = 3 * 3.7)
  (h3 : (11.85 : ℝ) = 3 * 3.95)
  (h4 : (8.6 : ℝ) = 2 * 4.3)
  : (4.475 : ℝ) = (40.5 - (11.1 + 11.85 + 8.6)) / 2 := 
sorry

end average_remaining_two_numbers_l126_126981


namespace speed_ratio_l126_126692

theorem speed_ratio (va vb : ℝ) (L : ℝ) (h : va = vb * k) (head_start : vb * (L - 0.05 * L) = vb * L) : 
    (va / vb) = (1 / 0.95) :=
by
  sorry

end speed_ratio_l126_126692


namespace correct_sampling_methods_l126_126391

-- Define the conditions as variables
variables (products_total : ℕ) (students_total : ℕ) (year1 : ℕ) (year2 : ℕ) (year3 : ℕ) (theater_rows : ℕ) (seats_per_row : ℕ) 

-- Define the tasks
def task1 := products_total = 30
def task2 := students_total = 2460 ∧ year1 + year2 + year3 = students_total ∧ year1 = 890 ∧ year2 = 820 ∧ year3 = 810
def task3 := theater_rows = 28 ∧ seats_per_row = 32 ∧ theater_rows * seats_per_row = 28 * 32

-- Define the correct sampling methods
def sampling_methods := 
  task1 → "Simple random sampling" ∧ 
  task2 → "Stratified sampling" ∧ 
  task3 → "Systematic sampling"

-- The proof statement: Prove the correct sampling methods given the conditions
theorem correct_sampling_methods : 
  sampling_methods :=
  sorry

end correct_sampling_methods_l126_126391


namespace friend_saves_per_week_l126_126317

theorem friend_saves_per_week (x : ℕ) : 
  160 + 7 * 25 = 210 + x * 25 → x = 5 := 
by 
  sorry

end friend_saves_per_week_l126_126317


namespace smallest_positive_solution_l126_126430

open Real

noncomputable def option_A := 5 - 2 * sqrt 8
noncomputable def option_B := 2 * sqrt 8 - 5
noncomputable def option_C := 12 - 3 * sqrt 9
noncomputable def option_D := 27 - 5 * sqrt 18
noncomputable def option_E := 5 * sqrt 18 - 27

theorem smallest_positive_solution :
  (∀ x ∈ {option_A, option_B, option_C, option_D, option_E}, x > 0 → x ≥ option_B) :=
sorry

end smallest_positive_solution_l126_126430


namespace factorization_correct_l126_126789

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l126_126789


namespace first_term_geometric_sequence_l126_126261

theorem first_term_geometric_sequence :
  ∃ (a r : ℝ), (a * r^3 = 5! ∧ a * r^6 = 7!) → a = 120 / (42)^(1/3) :=
by
  sorry

end first_term_geometric_sequence_l126_126261


namespace multiple_of_p_l126_126540

variable {p q : ℚ}
variable (m : ℚ)

theorem multiple_of_p (h1 : p / q = 3 / 11) (h2 : m * p + q = 17) : m = 2 :=
by sorry

end multiple_of_p_l126_126540


namespace fraction_value_l126_126251

theorem fraction_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 :=
by
  sorry

end fraction_value_l126_126251


namespace tan_cot_prod_identity_l126_126613

open Real -- Open the real number namespace to use real number functions directly.

theorem tan_cot_prod_identity (n : ℕ) (hn : 2 ≤ n) : 
  (∏ k in (finset.range n).map (nat.succ_above 1), 
  tan (π / 3 * (1 + 3 ^ k / (3 ^ n - 1)))) = 
  (∏ k in (finset.range n).map (nat.succ_above 1), 
  cot (π / 3 * (1 - 3 ^ k / (3 ^ n - 1)))) :=
sorry

end tan_cot_prod_identity_l126_126613


namespace inequality_proof_l126_126922

variables {a b c : ℝ}

theorem inequality_proof (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l126_126922


namespace asterisks_replacement_l126_126567

theorem asterisks_replacement :
  let digits := Finset.range 9
  let e_digits := Finset.filter (λ x, x % 2 = 0) digits -- even digits only
  let num_fixed_sum := 2 + 0 + 1 + 6 + 0
  ∃ (m : ℕ), m = 3645 ∧ (
    ∃ (x1 x2 x3 x4 x5 : ℕ),
    x1 ∈ digits ∧ x2 ∈ digits ∧ x3 ∈ digits ∧ x4 ∈ digits ∧ x5 ∈ e_digits ∧
    (num_fixed_sum + x1 + x2 + x3 + x4 + x5) % 9 = 0
  )
:= sorry

end asterisks_replacement_l126_126567


namespace max_independent_sets_l126_126033

-- Define conditions
def G : Type := Graph
def vertices_G : ℕ := 2000
def degree (v : G) : ℕ := 10

-- Define the theorem
theorem max_independent_sets (G : Type) [Graph G] (vertices : G → Prop) (degree : G → ℕ) :
  (vertices_G = 2000) ∧ (∀ v, degree v = 10) →
  ∃ max_sets, max_sets = 2047^100 :=
by
  intro h
  use 2047^100
  sorry

end max_independent_sets_l126_126033


namespace simplify_expression_l126_126927

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem simplify_expression (x : ℝ) (h : 2^x - 3 / 2^(x-1) = 5) : |x - 3| + x = 3 := by
  sorry

end simplify_expression_l126_126927


namespace log_eq_values_l126_126481

noncomputable def y (m : ℝ) : ℝ := 49

theorem log_eq_values (y m : ℝ) (h1 : log m y * log 7 m = 4) (h2 : log 7 y = 2) : y = 49 :=
by
  sorry

end log_eq_values_l126_126481


namespace eccentricity_of_hyperbola_l126_126859

variable {a : ℝ}

-- Definitions based on provided conditions
def parabola := ∀ x y : ℝ, y^2 = 4 * x
def hyperbola (a : ℝ) := a > 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 = 1
def directrix := ∀ x : ℝ, x = -1
def intersect_points (a : ℝ) := ∃ y : ℝ, let x := -1 in x^2 / a^2 - y^2 = 1
def focus_of_parabola := (1/0,0) -- Representing focus since it is a fixed property of the parabola

-- Statement of the problem: Given the conditions, the eccentricity of the hyperbola
theorem eccentricity_of_hyperbola (h_parabola : parabola) (h_hyperbola : hyperbola a) (h_directrix : directrix) (h_intersect : intersect_points a)
  (F : focus_of_parabola)
  (right_triangle : ∃ A B : (ℝ × ℝ), A ∈ hyperbola a ∧ B ∈ hyperbola a ∧ (F,A,B) form a right-angled triangle):
  ∃ e : ℝ, e = sqrt 6 :=
sorry

end eccentricity_of_hyperbola_l126_126859


namespace max_angle_between_vectors_l126_126504

variables {a b : ℝ^2}

-- Conditions
def condition1 : Prop := ∥a - b∥ = 3
def condition2 : Prop := ∥a∥ = 2 * ∥b∥

-- The theorem we aim to prove
theorem max_angle_between_vectors (h1 : condition1) (h2 : condition2) : 
  ∃ θ : ℝ, θ = real.pi / 6 ∧ (∀ φ : ℝ, φ ≤ θ) := sorry

end max_angle_between_vectors_l126_126504


namespace solve_for_x_l126_126624

theorem solve_for_x (x : ℝ) (h : (16^x) * (16^x) * (16^x) = 256^3) : x = 2 := 
by
  -- proof skipped
  sorry

end solve_for_x_l126_126624


namespace factorial_expression_value_l126_126682

theorem factorial_expression_value :
  (Nat.sqrt ((5.factorial * 4.factorial) + 2^2)) ^ 2 = 2884 :=
by
  sorry

end factorial_expression_value_l126_126682


namespace water_required_l126_126073

-- Definitions based on the conditions
def balanced_equation : Prop := ∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl + H2O = NH4OH + HCl

-- New problem with the conditions translated into Lean
theorem water_required 
  (h_eq : balanced_equation)
  (n : ℕ)
  (m : ℕ)
  (mole_NH4Cl : n = 2 * m)
  (mole_H2O : m = 2) :
  n = m :=
by
  sorry

end water_required_l126_126073


namespace complex_number_expression_l126_126926

noncomputable def x : ℂ := sorry
noncomputable def y : ℂ := sorry

def given_condition : Prop :=
  (x - y) / (2 * x + 3 * y) + (2 * x + 3 * y) / (x - y) = 2

def target_expression : ℂ :=
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4)

theorem complex_number_expression
  (h : given_condition) :
  target_expression = 34 / 15 :=
sorry

end complex_number_expression_l126_126926


namespace seashells_total_l126_126669

theorem seashells_total (tim_seashells sally_seashells : ℕ) (ht : tim_seashells = 37) (hs : sally_seashells = 13) :
  tim_seashells + sally_seashells = 50 := 
by 
  sorry

end seashells_total_l126_126669


namespace female_student_count_l126_126554

theorem female_student_count (F : ℝ) : 
  let male_students := 120 in
  let male_eng_students := 0.25 * male_students in
  let male_eng_pass := 0.20 * male_eng_students in
  let female_eng_students := 0.20 * F in
  let female_eng_pass := 0.25 * female_eng_students in
  (male_eng_pass + female_eng_pass) / (male_eng_students + female_eng_students) = 0.22 → F = 100 :=
by
  sorry

end female_student_count_l126_126554


namespace juliet_age_l126_126914

theorem juliet_age
    (M J R : ℕ)
    (h1 : J = M + 3)
    (h2 : J = R - 2)
    (h3 : M + R = 19) : J = 10 := by
  sorry

end juliet_age_l126_126914


namespace range_of_f_2x_le_1_l126_126462

-- Given conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

def cond_f_neg_2_eq_1 (f : ℝ → ℝ) : Prop :=
  f (-2) = 1

-- Main theorem
theorem range_of_f_2x_le_1 (f : ℝ → ℝ) 
  (h1 : is_odd f)
  (h2 : is_monotonically_decreasing f (Set.Iic 0))
  (h3 : cond_f_neg_2_eq_1 f) :
  Set.Icc (-1 : ℝ) 1 = { x | |f (2 * x)| ≤ 1 } :=
sorry

end range_of_f_2x_le_1_l126_126462


namespace exist_pos_integer_l126_126564

theorem exist_pos_integer 
  (a b : ℤ) (c d : ℚ) 
  (h_a : a ∈ {0, -3, 5, -100, 2008, -1}) 
  (h_b : b ∈ {0, -3, 5, -100, 2008, -1}) 
  (h_c : c ∈ {1/2, -1/3, 0.2, -3/2, -1/100}) 
  (h_d : d ∈ {1/2, -1/3, 0.2, -3/2, -1/100})
  : ∃ (x y : ℚ), (x = a + c) ∧ (y = b - d) ∧ (x + y = 1) :=
  sorry

end exist_pos_integer_l126_126564


namespace not_cube_100_zeros_l126_126335

theorem not_cube_100_zeros (k : ℤ) :
  let N := 10^202 + 5 * 10^101 + 1 in
  N ≠ k^3 := 
sorry

end not_cube_100_zeros_l126_126335


namespace number_of_pipes_used_l126_126365

-- Definitions
def T1 : ℝ := 15
def T2 : ℝ := T1 - 5
def T3 : ℝ := T2 - 4
def condition : Prop := 1 / T1 + 1 / T2 = 1 / T3

-- Proof Statement
theorem number_of_pipes_used : condition → 3 = 3 :=
by intros h; sorry

end number_of_pipes_used_l126_126365


namespace root_interval_l126_126925

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2^x - x^2

-- State the proof problem in Lean
theorem root_interval (h1 : f (-1) < 0) 
                      (h2 : f (-0.5) > 0) 
                      (h3 : f (-0.75) > 0) : 
  ∃ x ∈ Set.Ioo (-1 : ℝ) (-0.75 : ℝ), f x = 0 := 
sorry

end root_interval_l126_126925


namespace boys_girls_arrangement_l126_126943

theorem boys_girls_arrangement (boys girls : ℕ) (hb : boys = 3) (hg : girls = 2) :
  ∃ n : ℕ, n = (3! * 3!) ∧ n = 36 :=
by
  use 36
  rw [hb, hg]
  simp
  sorry

end boys_girls_arrangement_l126_126943


namespace church_rows_count_l126_126285

theorem church_rows_count {chairs_per_row people_per_chair total_people rows : ℕ} :
  chairs_per_row = 6 →
  people_per_chair = 5 →
  total_people = 600 →
  rows = total_people / (chairs_per_row * people_per_chair) →
  rows = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  have : 600 / (6 * 5) = 20, by norm_num
  rw this at h4
  exact h4

end church_rows_count_l126_126285


namespace file_organization_count_l126_126168

theorem file_organization_count {files : Finset ℕ}
  (h₀ : files = Finset.range 1 13 )
  (h₁ : ∀ i ∈ files, i ≠ 10 → (Finset.ite i files (files \ {10})) = (files \ {i}))
  (h₂ : ∀ i ∈ files \ {10}, i < 10)
  (h₃ : Finset.range 0 10.succ.sum (λ k, (Finset.card (finset.powerset (Finset.range k))).binomial 9 * (k + 2) ) = 2028 ):
  Finset.range 0 10.succ.sum (λ k, (Finset.card (finset.powerset (Finset.range k))).binomial 9 * (k + 2)) = 2028 :=
sorry

end file_organization_count_l126_126168


namespace range_of_k_l126_126824

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x < 3 → x - k < 2 * k) → 1 ≤ k :=
by
  sorry

end range_of_k_l126_126824


namespace find_prob_complement_l126_126085

variable {Ω : Type*} [MeasurableSpace Ω] (P : ProbabilityMeasure Ω)

variable (A B : Set Ω)

noncomputable def problem_conditions :=
P(B) = 0.3 ∧ P(B ∩ A) / P(A) = 0.9 ∧ P(B ∩ Aᶜ) / P(Aᶜ) = 0.2

theorem find_prob_complement (h : problem_conditions P A B) : 
  P(Aᶜ) = 6 / 7 := sorry

end find_prob_complement_l126_126085


namespace robert_has_2_more_years_l126_126082

theorem robert_has_2_more_years (R P T Rb M : ℕ) 
                                 (h1 : R = P + T + Rb + M)
                                 (h2 : R = 42)
                                 (h3 : P = 12)
                                 (h4 : T = 2 * Rb)
                                 (h5 : Rb = P - 4) : Rb - M = 2 := 
by 
-- skipped proof
  sorry

end robert_has_2_more_years_l126_126082


namespace common_ratio_of_log_geometric_sequence_l126_126211

noncomputable def harmonic_sequence (a b c : ℝ) :=
  c ≠ b ∧ b ≠ a ∧ a ≠ c ∧ 2 * b = a * c / (a + c)

noncomputable def log_geometric_sequence (a b c : ℝ) (r : ℂ) :=
  (complex.log (b / a) = complex.log (c / b) * r) ∧
  (complex.log (c / b) = complex.log (c / a) / r)

theorem common_ratio_of_log_geometric_sequence
  (a b c : ℝ) (r : ℂ)
  (h1 : harmonic_sequence a b c)
  (h2 : log_geometric_sequence a b c r) :
  r ^ 3 = 1 ∧ r ≠ 1 :=
  sorry

end common_ratio_of_log_geometric_sequence_l126_126211


namespace Carter_baked_more_cakes_l126_126754

/--
Carter usually bakes 6 cheesecakes, 5 muffins, and 8 red velvet cakes regularly for a week.
For this week he was able to bake triple the number of cheesecakes, muffins, and red velvet cakes.
Prove that Carter was able to bake 38 more cakes this week than he usually does.
-/
theorem Carter_baked_more_cakes :
    let cheesecakes_usual := 6
    let muffins_usual := 5
    let red_velvet_usual := 8
    let total_usual := cheesecakes_usual + muffins_usual + red_velvet_usual
    let cheesecakes_this_week := 3 * cheesecakes_usual
    let muffins_this_week := 3 * muffins_usual
    let red_velvet_this_week := 3 * red_velvet_usual
    let total_this_week := cheesecakes_this_week + muffins_this_week + red_velvet_this_week
    total_this_week - total_usual = 38 :=
by
    let cheesecakes_usual := 6
    let muffins_usual := 5
    let red_velvet_usual := 8
    let total_usual := cheesecakes_usual + muffins_usual + red_velvet_usual
    let cheesecakes_this_week := 3 * cheesecakes_usual
    let muffins_this_week := 3 * muffins_usual
    let red_velvet_this_week := 3 * red_velvet_usual
    let total_this_week := cheesecakes_this_week + muffins_this_week + red_velvet_this_week
    show total_this_week - total_usual = 38 from sorry

end Carter_baked_more_cakes_l126_126754


namespace midpoint_trajectory_of_circle_l126_126720

theorem midpoint_trajectory_of_circle 
  (M P : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hx : B = (3, 0))
  (hp : ∃(a b : ℝ), (P = (2 * a - 3, 2 * b)) ∧ (a^2 + b^2 = 1))
  (hm : M = ((P.1 + B.1) / 2, (P.2 + B.2) / 2)) :
  M.1^2 + M.2^2 - 3 * M.1 + 2 = 0 :=
by {
  -- Proof goes here
  sorry
}

end midpoint_trajectory_of_circle_l126_126720


namespace distinguishable_balls_in_distinguishable_boxes_l126_126135

theorem distinguishable_balls_in_distinguishable_boxes :
  ∃ (ways : ℕ), ways = (3^5) ∧ ways = 243 :=
by {
  use (3^5), -- Define the number of ways.
  split,
  exact rfl, -- Prove that (3^5) = (3^5).
  norm_num, -- Normalize the numerical expression to show that (3^5) = 243.
  sorry -- Placeholder for the proof.
}

end distinguishable_balls_in_distinguishable_boxes_l126_126135


namespace ratio_of_binomials_l126_126437

theorem ratio_of_binomials (a : ℝ) (k : ℕ) (h_k_positive : k > 0) :
  let binom_coeff (a : ℝ) (k : ℕ) : ℝ :=
    (1 : ℝ) / (Real.ofNat (Nat.factorial k)) *
    Finset.prod (Finset.range k) (λ i, a - (i : ℝ))
  in
  (binom_coeff (-3/2) 50) / (binom_coeff (3/2) 50) = 1 :=
by
  sorry

end ratio_of_binomials_l126_126437


namespace second_plan_fee_l126_126014

theorem second_plan_fee (M : ℝ) : 
  let cost_first_plan := 50 + 2000 * 0.35 in
  let cost_second_plan := M + 1500 * 0.45 in
  (cost_first_plan = cost_second_plan) → M = 75 :=
begin
  let cost_first_plan := 50 + 2000 * 0.35,
  let cost_second_plan := M + 1500 * 0.45,
  assume h: cost_first_plan = cost_second_plan,
  have h1: 50 + 2000 * 0.35 = 750 := rfl, -- verifying cost_first_plan calculation
  have h2: 750 = M + 675 := by rw [h1, h],
  have M_eq := calc
    M + 675 = 750                     : by rw [h2]
    ...       = 75 + 675 + 75 - 675   : by ring, -- simple manipulations
  exact M_eq,
end

end second_plan_fee_l126_126014


namespace a8_equals_two_or_minus_two_l126_126565

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + m) = a n * a m / a 0

theorem a8_equals_two_or_minus_two (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a)
    (h_roots : ∃ x y : ℝ, x^2 - 8 * x + 4 = 0 ∧ y^2 - 8 * y + 4 = 0 ∧ a 6 = x ∧ a 10 = y) :
  a 8 = 2 ∨ a 8 = -2 :=
by
  sorry

end a8_equals_two_or_minus_two_l126_126565


namespace distance_from_A_to_alpha_l126_126878

def vector (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def distance_point_to_plane (A B n : ℝ × ℝ × ℝ) : ℝ :=
  Real.abs (dot_product (B.1 - A.1, B.2 - A.2, B.3 - A.3) n) / magnitude n

theorem distance_from_A_to_alpha :
  let A := vector 1 0 (-1)
  let B := vector 0 (-1) 1
  let n := vector 1 2 1
  distance_point_to_plane A B n = Real.sqrt 6 / 6 := by
  sorry

end distance_from_A_to_alpha_l126_126878


namespace hyperbola_eccentricity_is_sqrt5_div_2_l126_126842

-- Definition of the ellipse with given eccentricity
def ellipse_properties (a b c : ℝ) (h_ab : a > b) (h_b0 : b > 0) 
  (h_c : c = sqrt (a^2 - b^2)) (h_e : c / a = sqrt 3 / 2) : Prop := 
  c / a = sqrt 3 / 2

-- Definition of the hyperbola's eccentricity
def hyperbola_eccentricity (a b c : ℝ) (ecc : ℝ) : Prop := 
  (c^2 = a^2 + b^2) ∧ (ecc = c / a)

-- Main theorem statement
theorem hyperbola_eccentricity_is_sqrt5_div_2 (a b c : ℝ)
  (h_ab : a > b) (h_b0 : b > 0) (h_c : c = sqrt (a^2 - b^2))
  (h_e : c / a = sqrt 3 / 2) : 
  ∃ c' : ℝ, hyperbola_eccentricity a b c' (sqrt 5 / 2) :=
by
  sorry

end hyperbola_eccentricity_is_sqrt5_div_2_l126_126842


namespace line_equation_passing_through_P_and_equal_intercepts_l126_126259

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition: line passes through point P(1, 3)
def passes_through_P (P : Point) (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq 1 3 = 0

-- Define the condition: equal intercepts on the x-axis and y-axis
def has_equal_intercepts (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ (∀ x y, line_eq x y = 0 ↔ x / a + y / a = 1)

-- Define the specific lines x + y - 4 = 0 and 3x - y = 0
def specific_line1 (x y : ℝ) : ℝ := x + y - 4
def specific_line2 (x y : ℝ) : ℝ := 3 * x - y

-- Define the point P(1, 3)
def P := Point.mk 1 3

theorem line_equation_passing_through_P_and_equal_intercepts :
  (passes_through_P P specific_line1 ∧ has_equal_intercepts specific_line1) ∨
  (passes_through_P P specific_line2 ∧ has_equal_intercepts specific_line2) :=
by
  sorry

end line_equation_passing_through_P_and_equal_intercepts_l126_126259


namespace not_all_ones_possible_l126_126451

def cell_value := ℤ
def grid := list (list cell_value)

def adjacent_cells (g : grid) (row col : ℕ) : list cell_value := sorry

def update_cell (g : grid) (row col : ℕ) : cell_value :=
  list.prod (adjacent_cells g row col)

def update_grid (g : grid) : grid :=
  g.map.with_index (λ row row_vals =>
    row_vals.map.with_index (λ col _ =>
      update_cell g row col))

def all_ones (g : grid) : Prop :=
  g.all (λ row => row.all (λ c => c = 1))

theorem not_all_ones_possible (initial_grid : grid)
  (h : initial_grid.length = 9 ∧ initial_grid.all (λ row => row.length = 9)) :
  ¬ ∃ n, all_ones (nat.iterate update_grid n initial_grid) :=
sorry

end not_all_ones_possible_l126_126451


namespace negation_of_prop_l126_126612

theorem negation_of_prop :
  (¬ ∀ (x y : ℝ), x^2 + y^2 ≥ 0) ↔ (∃ (x y : ℝ), x^2 + y^2 < 0) :=
by
  sorry

end negation_of_prop_l126_126612


namespace ellipse_standard_eq_of_common_foci_and_eccentricity_intersection_through_origin_minimum_area_of_triangle_l126_126828

-- Given conditions
def common_foci (C : Ellipse) (H : Hyperbola) := C.foci = H.foci
def eccentricity (C : Ellipse) (e : ℝ) := C.eccentricity = e

-- Corresponding proofs required (Lean Statements)
theorem ellipse_standard_eq_of_common_foci_and_eccentricity
  (C : Ellipse) (H : Hyperbola)
  (hfoci : common_foci C H)
  (hecc : eccentricity C (sqrt 6 / 3)) :
  C.equation = (λ p : ℝ × ℝ, (p.2^2) / 3 + (p.1^2) = 1) :=
sorry

theorem intersection_through_origin
  (C : Ellipse)
  (A : ℝ × ℝ)
  (M N: ℝ × ℝ)
  (h_vertex : A = ⟨0, -sqrt 3⟩)
  (h_M_on_C : C.contains M)
  (h_N_on_C : C.contains N)
  (h_MN_product_slopes : slope A M * slope A N = -3) :
  ∃ P : ℝ × ℝ, P = ⟨0, 0⟩ ∧ on_line_through P M ∧ on_line_through P N :=
sorry

theorem minimum_area_of_triangle
  (C : Ellipse)
  (M N P : ℝ × ℝ)
  (h_P_on_C : C.contains P)
  (h_MP_eq_NP : dist M P = dist N P) :
  area_of_triangle M N P ≥ 3 / 2 :=
sorry

end ellipse_standard_eq_of_common_foci_and_eccentricity_intersection_through_origin_minimum_area_of_triangle_l126_126828


namespace bob_correct_answer_l126_126382

theorem bob_correct_answer (y : ℕ) (h : (y - 7) / 5 = 47) : (y - 5) / 7 = 33 :=
by 
  -- assumption h and the statement to prove
  sorry

end bob_correct_answer_l126_126382


namespace decrease_in_sales_l126_126945

-- Conditions
variable (P Q : ℝ) -- Original price and quantity
variable (D : ℝ) -- Percentage decrease in sales, to be proven as 20%
def original_revenue := P * Q
def new_price := 1.7 * P
def new_quantity := Q * (1 - D / 100)
def new_revenue := new_price * new_quantity
def expected_new_revenue := 1.36 * original_revenue

-- The theorem to be proven
theorem decrease_in_sales : new_revenue = expected_new_revenue → D = 20 := 
by
  sorry

end decrease_in_sales_l126_126945


namespace least_perimeter_is_107_l126_126997

-- Define the lengths of the sides of the triangle
def side1 : ℕ := 47
def side2 : ℕ := 53

-- Define the conditions for the third side based on the triangle inequality
def valid_third_side (x : ℕ) : Prop :=
  x > 6 ∧ side1 + side2 > x ∧ side1 + x > side2 ∧ side2 + x > side1

-- Define the least possible perimeter
def least_possible_perimeter : ℕ :=
  side1 + side2 + 7

theorem least_perimeter_is_107 :
  (∃ (x : ℕ), valid_third_side x ∧ side1 + side2 + x = least_possible_perimeter) :=
begin
  sorry
end

end least_perimeter_is_107_l126_126997


namespace prime_ap_div_by_six_l126_126614

theorem prime_ap_div_by_six 
  (p1 p2 p3 : ℕ) 
  (h1 : p1 > 3) 
  (h2 : p2 > 3) 
  (h3 : p3 > 3) 
  (hp1 : nat.prime p1) 
  (hp2 : nat.prime p2) 
  (hp3 : nat.prime p3) 
  (h_ap : p2 = p1 + (p3 - p2)) :
  (p3 - p1) % 6 = 0 := 
sorry

end prime_ap_div_by_six_l126_126614


namespace find_daily_charge_first_agency_l126_126993

def daily_charge_first_agency (first_agency_daily : ℝ) : Prop :=
  ∀ (miles : ℝ), miles = 25 →
    (first_agency_daily + 0.14 * miles) = (18.25 + 0.22 * miles)

theorem find_daily_charge_first_agency :
  daily_charge_first_agency 20.25 :=
by {
  intros,
  sorry
}

end find_daily_charge_first_agency_l126_126993


namespace find_vector_n_l126_126501

variable (a b : ℝ)

def is_orthogonal (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def is_same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_vector_n (m n : ℝ × ℝ) (h1 : is_orthogonal m n) (h2 : is_same_magnitude m n) :
  n = (b, -a) :=
  sorry

end find_vector_n_l126_126501


namespace trig_values_of_point_l126_126454

theorem trig_values_of_point (r : ℝ) (h : r ≠ 0) :
  (let x := 3 * r,
       y := -4 * r,
       hyp := (x^2 + y^2)^(1/2);
   sin (atan2 y x) = y / hyp ∧
   cos (atan2 y x) = x / hyp ∧
   tan (atan2 y x) = y / x) ↔
  (sin (atan2 (-4 * r) (3 * r)) = -4/5 ∨ sin (atan2 (-4 * r) (3 * r)) = 4/ 5) ∧
  (cos (atan2 (-4 * r) (3 * r)) = 3/5 ∨ cos (atan2 (-4 * r) (3 * r)) = -3/5) ∧
  (tan (atan2 (-4 * r) (3 * r)) = -4/3) := by
  sorry

end trig_values_of_point_l126_126454


namespace adam_apples_l126_126373

theorem adam_apples (x : ℕ) : 
  let monday := 15 in
  let tuesday := 3 * monday in
  let wednesday := x * tuesday in
  monday + tuesday + wednesday = 240 → x = 4 :=
by
  intros monday tuesday wednesday
  sorry

end adam_apples_l126_126373


namespace a_minus_b_eq_zero_l126_126882

-- Definitions from the conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- The point (0, b)
def point_b (b : ℝ) : (ℝ × ℝ) := (0, b)

-- Slope condition at point (0, b)
def slope_of_f_at_0 (a : ℝ) : ℝ := a
def slope_of_tangent_line : ℝ := 1

-- Prove a - b = 0 given the conditions
theorem a_minus_b_eq_zero (a b : ℝ) 
    (h1 : f 0 a b = b)
    (h2 : tangent_line 0 b) 
    (h3 : slope_of_f_at_0 a = slope_of_tangent_line) : a - b = 0 :=
by
  sorry

end a_minus_b_eq_zero_l126_126882


namespace replaced_man_age_l126_126634

-- Definitions of the conditions
def average_age_old : ℝ -- Average age of the old group of 15 men
def age_replaced_1 : ℝ := 21 -- Age of the first man being replaced
def age_replaced_2 : ℝ -- Age of the second man being replaced
def average_age_new : ℝ := 37 -- Average age of the two new men
def age_difference : ℝ := 2 -- Increase in average age

-- Total age of the group increased by this amount due to the replacement of men
def total_age_increase : ℝ := 15 * age_difference

def age_new_1 : ℝ := 37 -- Age of the first new man
def age_new_2 : ℝ := 37 -- Age of the second new man
def total_age_new : ℝ := age_new_1 + age_new_2

-- Initial total age of the group
def total_age_old : ℝ := 15 * average_age_old

-- Problem statement
theorem replaced_man_age : age_replaced_2 = 23 :=
by
  -- sorry will be replaced with the actual proof steps.
  sorry

end replaced_man_age_l126_126634


namespace green_edges_count_l126_126299

theorem green_edges_count (V E : Type) [Fintype V] [Fintype E] (v_label : V → Finset E) (e_color : E → Fin 3)
  (octahedron : Fintype.card E = 12) (vertices_property : ∀ v : V, ∃! es : Finset E, es.card = 3 ∧ (∀ c : Fin 3, ∃ e ∈ es, e_color e = c)) :
  (∃ k : ℕ, 3 ≤ k ∧ k ≤ 6 ∧ (Fintype.card (Finset.filter (λ e, e_color e = 2) (Finset.univ : Finset E)) = k)) :=
sorry

end green_edges_count_l126_126299


namespace secret_sharing_problem_l126_126603

theorem secret_sharing_problem : 
  ∃ n : ℕ, (3280 = (3^(n + 1) - 1) / 2) ∧ (n = 7) :=
by
  use 7
  sorry

end secret_sharing_problem_l126_126603


namespace greatest_common_multiple_less_than_bound_l126_126301

-- Define the numbers and the bound
def num1 : ℕ := 15
def num2 : ℕ := 10
def bound : ℕ := 150

-- Define the LCM of num1 and num2
def lcm_num1_num2 : ℕ := Nat.lcm num1 num2

-- Define the greatest multiple of LCM less than bound
def greatest_multiple_less_than_bound (lcm : ℕ) (b : ℕ) : ℕ :=
  (b / lcm) * lcm

-- Main theorem
theorem greatest_common_multiple_less_than_bound :
  greatest_multiple_less_than_bound lcm_num1_num2 bound = 120 :=
by
  sorry

end greatest_common_multiple_less_than_bound_l126_126301


namespace friend_saves_per_week_l126_126319

theorem friend_saves_per_week
  (x : ℕ) 
  (you_have : ℕ := 160)
  (you_save_per_week : ℕ := 7)
  (friend_have : ℕ := 210)
  (weeks : ℕ := 25)
  (total_you_save : ℕ := you_have + you_save_per_week * weeks)
  (total_friend_save : ℕ := friend_have + x * weeks) 
  (h : total_you_save = total_friend_save) : x = 5 := 
by 
  sorry

end friend_saves_per_week_l126_126319


namespace carter_extra_cakes_l126_126750

def regular_cakes : ℕ := 6 + 5 + 8
def triple_cakes : ℕ := 3 * 6 + 3 * 5 + 3 * 8
def extra_cakes : ℕ := triple_cakes - regular_cakes

theorem carter_extra_cakes : extra_cakes = 38 :=
by
  unfold regular_cakes triple_cakes extra_cakes
  calc
    3 * 6 + 3 * 5 + 3 * 8 - (6 + 5 + 8)
      = 57 - 19 : by norm_num
    ... = 38 : by norm_num

end carter_extra_cakes_l126_126750


namespace total_area_of_plots_l126_126663

theorem total_area_of_plots (n : ℕ) (side_length : ℕ) (area_one_plot : ℕ) (total_plots : ℕ) (total_area : ℕ)
  (h1 : n = 9)
  (h2 : side_length = 6)
  (h3 : area_one_plot = side_length * side_length)
  (h4 : total_plots = n)
  (h5 : total_area = area_one_plot * total_plots) :
  total_area = 324 := 
by
  sorry

end total_area_of_plots_l126_126663


namespace solve_inequality_l126_126626

theorem solve_inequality (x : ℝ) : x * (|x| - 1) < 0 ↔ x ∈ set.Ioo (-∞) (-1) ∪ set.Ioo 0 1 := sorry

end solve_inequality_l126_126626


namespace avg_of_arithmetic_series_is_25_l126_126384

noncomputable def arithmetic_series_avg : ℝ :=
  let a₁ := 15
  let d := 1 / 4
  let aₙ := 35
  let n := (aₙ - a₁) / d + 1
  let S := n * (a₁ + aₙ) / 2
  S / n

theorem avg_of_arithmetic_series_is_25 : arithmetic_series_avg = 25 := 
by
  -- Sorry, proof omitted due to instruction.
  sorry

end avg_of_arithmetic_series_is_25_l126_126384


namespace least_bulbs_needed_l126_126241

/-- Tulip bulbs come in packs of 15, and daffodil bulbs come in packs of 16.
  Rita wants to buy the same number of tulip and daffodil bulbs. 
  The goal is to prove that the least number of bulbs she needs to buy is 240, i.e.,
  the least common multiple of 15 and 16 is 240. -/
theorem least_bulbs_needed : Nat.lcm 15 16 = 240 := 
by
  sorry

end least_bulbs_needed_l126_126241


namespace parallel_lines_l126_126126

theorem parallel_lines (m : ℝ) (l1_parallel_l2 : 
  m * x + y - 1 = 0 ∧ (4 * m - 3) * x + m * y - 1 = 0 ∧ l1_parallel_l2 (l1_slope_eq_l2_slope : -m = -((4 * m - 3) / m)) :
  m = 3 :=
begin
  sorry
end

end parallel_lines_l126_126126


namespace area_AGE_l126_126230

-- Definitions based on conditions
def A := (0 : ℝ, 0 : ℝ)
def B := (5 : ℝ, 0 : ℝ)
def C := (5 : ℝ, 5 : ℝ)
def D := (0 : ℝ, 5 : ℝ)
def E := (5 : ℝ, 2 : ℝ)
def G : ℝ × ℝ := sorry  -- G coordinates derived from circumcircle intersection, yet to be defined

-- Lean definition to calculate the area of triangle given three vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  real.abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)

-- The problem statement in Lean 4
theorem area_AGE : triangle_area A G E = 48.5 := sorry

end area_AGE_l126_126230


namespace books_sold_fraction_l126_126708

noncomputable def fraction_of_books_sold (B : ℕ) (F : ℚ) : Prop :=
  (255 / 4.25 = B - 30) ∧ (F = (B - 30) / B)

theorem books_sold_fraction:
  fraction_of_books_sold 90 (2 / 3) :=
by
  sorry

end books_sold_fraction_l126_126708


namespace amy_homework_time_l126_126032

def total_time_to_finish_homework 
  (math_problems : ℕ) 
  (spelling_problems : ℕ) 
  (math_per_hour : ℕ)
  (spelling_per_hour : ℕ) 
  (break_minutes : ℕ) 
  (one_hour_in_minutes : ℕ) : ℕ :=
  let math_hours := math_problems / math_per_hour
  let spelling_hours := spelling_problems / spelling_per_hour
  let work_hours := math_hours + spelling_hours
  let breaks_hours := ((work_hours - 1) * break_minutes) / one_hour_in_minutes
  work_hours + breaks_hours

theorem amy_homework_time 
  (math_problems : ℕ = 18)
  (spelling_problems : ℕ = 6) 
  (math_per_hour : ℕ = 3)
  (spelling_per_hour : ℕ = 2)
  (break_minutes : ℕ = 15)
  (one_hour_in_minutes : ℕ = 60) : 
  total_time_to_finish_homework math_problems spelling_problems math_per_hour spelling_per_hour break_minutes one_hour_in_minutes = 11 := 
by 
  sorry

end amy_homework_time_l126_126032


namespace sum_of_coefficients_is_neg_one_l126_126407

noncomputable def polynomial := (5:ℤ) * (2 * (x:ℤ)^8 + 5 * x^3 - 9) + 3 * (2 * x^6 - 3 * x^3 + 4)

theorem sum_of_coefficients_is_neg_one : 
  let sumCoeffs := 10 + 6 + 16 - 33 in
  sumCoeffs = -1 :=
by
  let sumCoeffs := (5 * 2 + 5 * 5 * 1^3 - 9) + 3 * (2 * 1^6 - 3 * 1^3 + 4)
  have sumCoeffs_eq : sumCoeffs = 32 - 33 := by sorry
  exact sumCoeffs_eq

end sum_of_coefficients_is_neg_one_l126_126407


namespace find_b_l126_126675

variable (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ)

def is_parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2, k * w.3)

def is_orthogonal (v w : ℝ × ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3 = 0

def vector_add (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2, v.3 + w.3)

theorem find_b :
  ∃ a b : ℝ × ℝ × ℝ,
    vector_add a b = (3, -6, 9) ∧
    is_parallel a (2, 2, 2) ∧
    is_orthogonal b (2, 2, 2) ∧
    b = (1, -8, 7) :=
by
  sorry

end find_b_l126_126675


namespace equilateral_triangle_APB_l126_126581

noncomputable def quadrilateral (A B C D : Type*) := sorry
noncomputable def equilateral_triangle (D P C : Type*) := sorry

theorem equilateral_triangle_APB 
  {A B C D P : Type*} [quadrilateral A B C D] [equilateral_triangle D P C]
  (h1 : ∃ (AD BC : ℝ), AD = BC)
  (h2 : ∃ (DAB ABC : ℝ), DAB + ABC = 120)
  : ∃ (AP BP : ℝ) (APB : ℝ), AP = BP ∧ APB = 60 :=
sorry

end equilateral_triangle_APB_l126_126581


namespace Carter_baked_more_cakes_l126_126752

/--
Carter usually bakes 6 cheesecakes, 5 muffins, and 8 red velvet cakes regularly for a week.
For this week he was able to bake triple the number of cheesecakes, muffins, and red velvet cakes.
Prove that Carter was able to bake 38 more cakes this week than he usually does.
-/
theorem Carter_baked_more_cakes :
    let cheesecakes_usual := 6
    let muffins_usual := 5
    let red_velvet_usual := 8
    let total_usual := cheesecakes_usual + muffins_usual + red_velvet_usual
    let cheesecakes_this_week := 3 * cheesecakes_usual
    let muffins_this_week := 3 * muffins_usual
    let red_velvet_this_week := 3 * red_velvet_usual
    let total_this_week := cheesecakes_this_week + muffins_this_week + red_velvet_this_week
    total_this_week - total_usual = 38 :=
by
    let cheesecakes_usual := 6
    let muffins_usual := 5
    let red_velvet_usual := 8
    let total_usual := cheesecakes_usual + muffins_usual + red_velvet_usual
    let cheesecakes_this_week := 3 * cheesecakes_usual
    let muffins_this_week := 3 * muffins_usual
    let red_velvet_this_week := 3 * red_velvet_usual
    let total_this_week := cheesecakes_this_week + muffins_this_week + red_velvet_this_week
    show total_this_week - total_usual = 38 from sorry

end Carter_baked_more_cakes_l126_126752


namespace count_four_digit_integers_with_3_or_7_digits_l126_126513

-- Definitions based on conditions
def is_four_digit_integer (n : ℕ) : Prop :=
  n >= 1000 ∧ n <= 9999

def is_made_of_3_and_7 (n : ℕ) : Prop :=
  n.digits 10.All (λ d, d = 3 ∨ d = 7)

-- Problem statement:
theorem count_four_digit_integers_with_3_or_7_digits : 
  {n : ℕ // is_four_digit_integer n ∧ is_made_of_3_and_7 n}.card = 16 :=
by 
  sorry

end count_four_digit_integers_with_3_or_7_digits_l126_126513


namespace triangle_formation_segments_l126_126020

theorem triangle_formation_segments (a b c : ℝ) (h_sum : a + b + c = 1) (h_a : a < 1/2) (h_b : b < 1/2) (h_c : c < 1/2) : 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := 
by
  sorry

end triangle_formation_segments_l126_126020


namespace ceil_neg_seven_fourths_cubed_eq_neg_five_l126_126416

noncomputable def ceil_of_neg_seven_fourths_cubed : ℤ :=
  Int.ceil ((-7 / 4 : ℚ)^3)

theorem ceil_neg_seven_fourths_cubed_eq_neg_five :
  ceil_of_neg_seven_fourths_cubed = -5 := by
  sorry

end ceil_neg_seven_fourths_cubed_eq_neg_five_l126_126416


namespace train_rate_third_hour_l126_126367

theorem train_rate_third_hour
  (x : ℝ)
  (h1 : ∀ (t : ℕ), (t ≥ 4) → rate (t) = x + 10 * (↑t - 3))
  (h2 : rate (1) = 10)
  (h3 : rate (2) = 20)
  (h4 : ∀ (t : ℕ), distance (rate t) 1 hr = rate t)
  (h5 : total_distance = ∑ t in 1..11, distance (rate t) 1)
  (h6 : total_distance = 660) : x = 33.75 :=
by {
  sorry
}

end train_rate_third_hour_l126_126367


namespace sum_first_10_terms_l126_126092

-- Define the general arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the conditions of the problem
def given_conditions (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = 2 * a 4 ∧ arithmetic_seq a d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Statement of the problem
theorem sum_first_10_terms (a : ℕ → ℤ) (d : ℤ) (S₁₀ : ℤ) :
  given_conditions a d →
  (S₁₀ = 20 ∨ S₁₀ = 110) :=
sorry

end sum_first_10_terms_l126_126092


namespace angle_A_in_trapezoid_l126_126910

theorem angle_A_in_trapezoid 
  (ABCD : Type) 
  (A B C D : ABCD)
  (AB_parallel_CD : IsParallel (Line AB) (Line CD))
  (angleA_eq_3angleD : ∃ angleD : ℝ, angleA = 3 * angleD)
  (angleC_eq_2angleD : ∃ angleD : ℝ, angleC = 2 * angleD)
  (angle_sum : angleA + angleD = 180) :
  angleA = 135 :=
by 
  sorry

end angle_A_in_trapezoid_l126_126910


namespace x_intercept_of_line_l126_126695

def point1 := (10, 3)
def point2 := (-12, -8)

theorem x_intercept_of_line :
  let m := (point2.snd - point1.snd) / (point2.fst - point1.fst)
  let line_eq (x : ℝ) := m * (x - point1.fst) + point1.snd
  ∃ x : ℝ, line_eq x = 0 ∧ x = 4 :=
by
  sorry

end x_intercept_of_line_l126_126695


namespace fraction_of_seashells_given_to_liam_l126_126132

theorem fraction_of_seashells_given_to_liam (n : ℕ) (Hannah Liam Noah : ℕ) 
  (h1 : Hannah = 4 * Liam)
  (h2 : Liam = 3 * Noah) :
  (let needed_for_liam := ((16 * n) / 3) - (3 * n) in
  let fraction_given_to_liam := needed_for_liam / (12 * n) in
  fraction_given_to_liam = 7 / 36) :=
sorry

end fraction_of_seashells_given_to_liam_l126_126132


namespace f_negative_l126_126952

-- Define the function f(x)
def f (a x : ℝ) : ℝ :=
  (a-x)^6 - 3*a*(a-x)^5 + (5/2) * a^2 * (a-x)^4 - (1/2) * a^4 * (a-x)^2

-- Define the conditions
variable (a x : ℝ)
variable (h₀ : 0 < x)
variable (h₁ : x < a)

-- The theorem to be proven
theorem f_negative (h₀ : 0 < x) (h₁ : x < a) : f a x < 0 :=
sorry

end f_negative_l126_126952


namespace frog_final_position_probability_l126_126712

noncomputable def frog_jump_probability : ℝ := 0.40

theorem frog_final_position_probability :
  ∀ (jumps : Fin 5 → EuclideanSpace ℝ 2),
    (∀ i, ∥jumps i∥ = 1) →             -- Each jump is 1 meter
    Prob (∥Finset.univ.sum jumps∥ ≤ 2) = frog_jump_probability := -- Probability that final position is ≤ 2 meters
sorry

end frog_final_position_probability_l126_126712


namespace sum_of_f_l126_126213

def f (x : ℝ) : ℝ := 2^x / (2^x + real.sqrt 2)

theorem sum_of_f : (∑ i in finset.range (2 * 2017 + 2), f (-2016 + i)) = 2017 :=
by
  sorry

end sum_of_f_l126_126213


namespace scalene_triangles_with_perimeter_le_15_count_l126_126403

-- Define a helper function to check if three sides form a scalene triangle
def is_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- Define the main theorem stating the number of scalene triangles with perimeter <= 15
theorem scalene_triangles_with_perimeter_le_15_count :
  {t : ℕ × ℕ × ℕ // let (a, b, c) := t in is_scalene_triangle a b c ∧ a + b + c ≤ 15}.to_finset.card = 6 :=
sorry

end scalene_triangles_with_perimeter_le_15_count_l126_126403


namespace number_of_blue_marbles_l126_126339

-- Definitions based on the conditions
def total_marbles : ℕ := 20
def red_marbles : ℕ := 9
def probability_red_or_white : ℚ := 0.7

-- The question to prove: the number of blue marbles (B)
theorem number_of_blue_marbles (B W : ℕ) (h1 : B + W + red_marbles = total_marbles)
  (h2: (red_marbles + W : ℚ) / total_marbles = probability_red_or_white) : 
  B = 6 := 
by
  sorry

end number_of_blue_marbles_l126_126339


namespace increased_square_side_percentage_l126_126274

theorem increased_square_side_percentage (p : ℤ) (h : (1 + p / 100)^2 = 1.96) : p = 40 :=
by
  sorry

end increased_square_side_percentage_l126_126274


namespace base7_multiplication_l126_126080

def base7_to_base10 (n : ℕ) : ℕ :=
  n.digits₇.foldl (λ acc d, acc * 7 + d) 0

def base10_to_base7 (n : ℕ) : ℕ :=
let rec toBase (n : ℕ) (acc : ℕ) : ℕ :=
  if n = 0 then acc
  else toBase (n / 7) (acc * 10 + n % 7)
toBase n 0

theorem base7_multiplication (a b : ℕ) (c : ℕ) (h₁ : a = 345) (h₂ : b = 3) (h₃ : c = 1401) :
  base10_to_base7 ((base7_to_base10 a) * (base7_to_base10 b)) = c := by sorry

end base7_multiplication_l126_126080


namespace Ak_geq_Aj_l126_126590

variable {n : ℕ} (h : n > 1)
variable {a : Fin n → ℝ} (pos : ∀ i, 0 < a i)

def A (k : ℕ) : ℝ :=
  ((Finset.univ.sum (λ i : Fin n, (a i)^k)) * (Finset.univ.sum (λ i : Fin n, (a i)^(n - k))))

theorem Ak_geq_Aj (k j : ℕ) (hkj : 1 ≤ k) (hk_lt_hj : k < j) (hj_leq_halfn : j ≤ n / 2) : 
  A k ≥ A j ∧ (A k = A j ↔ ∀ i j, a i = a j) := sorry

end Ak_geq_Aj_l126_126590


namespace replace_square_l126_126877

theorem replace_square (x : ℝ) (h : 10.0003 * x = 10000.3) : x = 1000 :=
sorry

end replace_square_l126_126877


namespace difference_of_integers_l126_126281

theorem difference_of_integers : ∃ (x y : ℕ), x + y = 20 ∧ x * y = 96 ∧ (x - y = 4 ∨ y - x = 4) :=
by
  sorry

end difference_of_integers_l126_126281


namespace sum_coefficients_l126_126929

theorem sum_coefficients (c : Fin 301 → ℤ) :
  (1 + x + x^2) ^ 150 = ∑ k in Finset.range 301, (c k) * x ^ k →
  (∑ k in Finset.range 101, c (3 * k)) = 3 ^ 149 :=
by
  sorry

end sum_coefficients_l126_126929


namespace log_bounds_l126_126432

theorem log_bounds :
  (∃ m n : ℤ, 1 < real.log10 50 ∧ real.log10 50 < 2 ∧ m + n = 3) :=
sorry

end log_bounds_l126_126432


namespace triangle_BX_eq_CY_l126_126583

theorem triangle_BX_eq_CY 
  (ABC : Triangle)
  (Γ : Circumcircle ABC)
  (N : Point)
  (HN : IsMidpointOfArcContaining N B A C Γ)
  (𝒞 : Circle)
  (H𝒞1 : PassesThrough 𝒞 A)
  (H𝒞2 : PassesThrough 𝒞 N)
  (X : Point)
  (HX : Intersects 𝒞 (LineSegment A B) X)
  (Y : Point)
  (HY : Intersects 𝒞 (LineSegment A C) Y) 
  : SegmentLength B X = SegmentLength C Y := 
sorry

end triangle_BX_eq_CY_l126_126583


namespace problem_statement_l126_126765

def bracket (a b c : ℕ) (h : c ≠ 0) : ℕ := (a + b) / c

theorem problem_statement : bracket (bracket 80 40 120 (by decide)) 
                                   (bracket 4 2 6 (by decide)) 
                                   (bracket 50 25 75 (by decide)) 
                                   (by decide) = 2 := 
begin
  sorry
end

end problem_statement_l126_126765


namespace binomial_coefficients_equality_l126_126474

theorem binomial_coefficients_equality (n k : ℕ) (h : binom n 2 = binom n 6) :
  n = 8 ∧ (k = 5 ∨ k = 6) :=
by
  sorry

end binomial_coefficients_equality_l126_126474


namespace part1_part2_l126_126201

noncomputable def f (m n : ℕ) : ℕ → ℤ
  | k := (multiset.Nat.choose m (k / 2) - multiset.Nat.choose (2 * n) k) * (if even k then 1 else 0) 

theorem part1 : f 4 3 3 + f 4 3 4 = -29 := 
  by
  sorry

theorem part2 (h : f m n 2 = 20) : 2 * n + 20 / n - 1 ≥ 35 / 3 := 
  by
  sorry

end part1_part2_l126_126201


namespace curve_is_circle_l126_126076

theorem curve_is_circle (r θ: ℝ):
  r = 3 * cos θ -> ∃ (x y: ℝ), (x = r * cos θ) ∧ (y = r * sin θ) ∧ (x^2 + y^2 = 9) :=
by
  intros h
  sorry

end curve_is_circle_l126_126076


namespace find_a_l126_126097

variables (a : ℝ) (p q : Prop)

def prop_p := (-2 < a) ∧ (a < 2)
def prop_q := a < 2

theorem find_a (hpq : (prop_p a) ∨ (prop_q a)) (hnp_q : ¬((prop_p a) ∧ (prop_q a))) : a ∈ Iic (-2) :=
by 
  sorry

end find_a_l126_126097


namespace min_value_expr_l126_126059

-- Definition of the expression given a real constant k
def expr (k : ℝ) (x y : ℝ) : ℝ := 9 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

-- The proof problem statement
theorem min_value_expr (k : ℝ) (h : k = 2 / 9) : ∃ x y : ℝ, expr k x y = 1 ∧ ∀ x y : ℝ, expr k x y ≥ 1 :=
by
  sorry

end min_value_expr_l126_126059


namespace distance_point_to_line_example_l126_126990

def distance_to_line (x1 y1 a b c : ℝ) : ℝ :=
  (abs (a * x1 + b * y1 + c)) / (Real.sqrt (a^2 + b^2))

theorem distance_point_to_line_example :
  distance_to_line 1 1 1 1 (-1) = Real.sqrt 2 / 2 :=
by
  sorry

end distance_point_to_line_example_l126_126990


namespace parallel_lines_iff_m_eq_neg2_l126_126987

theorem parallel_lines_iff_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0 → m * x + 2 * y - m + 2 = 0 ↔ m = -2) :=
sorry

end parallel_lines_iff_m_eq_neg2_l126_126987


namespace sandy_correct_sums_l126_126697

theorem sandy_correct_sums (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x - 2 * y = 50) : x = 22 :=
  by
  sorry

end sandy_correct_sums_l126_126697


namespace probability_truth_or_lies_l126_126165

def probability_truth := 0.30
def probability_lies := 0.20
def probability_both := 0.10

theorem probability_truth_or_lies :
  (probability_truth + probability_lies - probability_both) = 0.40 :=
by
  sorry

end probability_truth_or_lies_l126_126165


namespace perimeter_of_rectangle_l126_126701

-- Given conditions
variables (AE BE CF : ℕ) (m n : ℕ)
axiom AE_def : AE = 12
axiom BE_def : BE = 17
axiom CF_def : CF = 5

-- Define the problem to prove the perimeter of the rectangle as a fraction m/n, and find m+n
theorem perimeter_of_rectangle (m n : ℕ) (AE BE CF : ℕ) [fact (nat.coprime m n)] :
  AE = 12 ∧ BE = 17 ∧ CF = 5 → 
  2 * (AE + BE + CF) = m / n → 
  ∃ k : ℕ, k = m + n :=
begin
  intros h1 h2,
  sorry
end

end perimeter_of_rectangle_l126_126701


namespace angle_BDC_l126_126275

theorem angle_BDC (BAC ABC : ℝ) (h₁ : BAC = 48) (h₂ : ABC = 71) :
  ∃ BDC : ℝ, BDC = 30.5 := 
by
  -- initializing the angle calculation
  let BCA := 180 - BAC - ABC
  -- D is incenter and the angle bisector property
  let BDC := BCA / 2
  -- exact result
  use BDC
  -- providing necessary degree conditions 
  have h₃ : BCA = 61 := by linarith [h₁, h₂]
  have h₄ : BDC = 30.5 := by linarith [h₃]
  exact h₄
  sorry

end angle_BDC_l126_126275


namespace smallest_positive_alpha_l126_126453

theorem smallest_positive_alpha (α : ℝ) :
  let P := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) in
  P = (Real.sin α, Real.cos α) → 
  (α = 11 * Real.pi / 6) :=
sorry

end smallest_positive_alpha_l126_126453


namespace matt_without_calculator_5_minutes_l126_126573

-- Define the conditions
def time_with_calculator (problems : Nat) : Nat := 2 * problems
def time_without_calculator (problems : Nat) (x : Nat) : Nat := x * problems
def time_saved (problems : Nat) (x : Nat) : Nat := time_without_calculator problems x - time_with_calculator problems

-- State the problem
theorem matt_without_calculator_5_minutes (x : Nat) :
  (time_saved 20 x = 60) → x = 5 := by
  sorry

end matt_without_calculator_5_minutes_l126_126573


namespace Susan_ate_six_candies_l126_126773

def candy_consumption_weekly : Prop :=
  ∀ (candies_bought_Tue candies_bought_Wed candies_bought_Thu candies_bought_Fri : ℕ)
    (candies_left : ℕ) (total_spending : ℕ),
    candies_bought_Tue = 3 →
    candies_bought_Wed = 0 →
    candies_bought_Thu = 5 →
    candies_bought_Fri = 2 →
    candies_left = 4 →
    total_spending = 9 →
    candies_bought_Tue + candies_bought_Wed + candies_bought_Thu + candies_bought_Fri - candies_left = 6

theorem Susan_ate_six_candies : candy_consumption_weekly :=
by {
  -- The proof will be filled in later
  sorry
}

end Susan_ate_six_candies_l126_126773


namespace son_present_age_l126_126323

theorem son_present_age (S F : ℕ) (h1 : F = S + 34) (h2 : F + 2 = 2 * (S + 2)) : S = 32 :=
by
  sorry

end son_present_age_l126_126323


namespace max_possible_desks_bookcases_l126_126324

theorem max_possible_desks_bookcases (D B : ℕ) 
  (hEq : D = B) 
  (hCond : 2 * D + 1.5 * B ≤ 15) : 
  15 - (2 * D + 1.5 * B) = 1 :=
by
  sorry

end max_possible_desks_bookcases_l126_126324


namespace reduced_price_of_oil_is_40_l126_126018

variables 
  (P R : ℝ) 
  (hP : 0 < P)
  (hR : R = 0.75 * P)
  (hw : 800 / (0.75 * P) = 800 / P + 5)

theorem reduced_price_of_oil_is_40 : R = 40 :=
sorry

end reduced_price_of_oil_is_40_l126_126018


namespace certain_number_is_1008_l126_126525

theorem certain_number_is_1008 :
  ∀ (w : ℕ),
  w > 0 → w = 168 →
  (∀ (n : ℕ),
  ∃ (a b c : ℕ), 
  n = (2^a) * (3^b) * (14^c) ∧ a ≥ 5 ∧ b ≥ 3 ∧ c ≥ 2 →
  ∃ (k : ℕ), (2^5) ∣ (k * w) ∧ (3^3) ∣ (k * w) ∧ (14^2) ∣ (k * w)) →
  ∃ n : ℕ, n = 1008 :=
begin
  sorry
end

end certain_number_is_1008_l126_126525


namespace area_triangle_two_lines_l126_126550

open Real

/-- Given points B(1, 0) and C(-1, 0), the set of all A(x, y) such that
the area of triangle ABC is 2 consists of two horizontal lines y = 2 and y = -2. -/
theorem area_triangle_two_lines (x y : ℝ) :
  let B := (1 : ℝ, 0 : ℝ)
  let C := (-1 : ℝ, 0 : ℝ)
  let area := (1 / 2: ℝ) * |(x * (0 - 0) + 1 * (0 - y) + -1 * (y - 0))| in
  area = 2 → (y = 2 ∨ y = -2) :=
sorry

end area_triangle_two_lines_l126_126550


namespace obtuse_triangle_with_consecutive_sides_unique_l126_126171

theorem obtuse_triangle_with_consecutive_sides_unique :
  ∃! (x : ℕ), (x > 0) ∧ is_obtuse (x, x + 1, x + 2) := sorry

noncomputable def is_obtuse (a b c : ℕ) : Prop :=
  if c >= max a b then a^2 + b^2 < c^2
  else if b >= max a c then a^2 + c^2 < b^2
  else b^2 + c^2 < a^2

-- Properties and Lemmas go here
-- (To be filled during proof construction)

end

end obtuse_triangle_with_consecutive_sides_unique_l126_126171


namespace radar_coverage_proof_l126_126334

theorem radar_coverage_proof (n : ℕ) (r : ℝ) (w : ℝ) (d : ℝ) (A : ℝ) : 
  n = 9 ∧ r = 37 ∧ w = 24 ∧ d = 35 / Real.sin (Real.pi / 9) ∧
  A = 1680 * Real.pi / Real.tan (Real.pi / 9) → 
  ∃ OB S_ring, OB = d ∧ S_ring = A 
:= by sorry

end radar_coverage_proof_l126_126334


namespace order_of_numbers_l126_126031

def base16_to_dec (s : String) : ℕ := sorry
def base6_to_dec (s : String) : ℕ := sorry
def base4_to_dec (s : String) : ℕ := sorry
def base2_to_dec (s : String) : ℕ := sorry

theorem order_of_numbers:
  let a := base16_to_dec "3E"
  let b := base6_to_dec "210"
  let c := base4_to_dec "1000"
  let d := base2_to_dec "111011"
  a = 62 ∧ b = 78 ∧ c = 64 ∧ d = 59 →
  b > c ∧ c > a ∧ a > d :=
by
  intros
  sorry

end order_of_numbers_l126_126031


namespace four_digit_integers_with_3_or_7_only_l126_126511

theorem four_digit_integers_with_3_or_7_only : 
  ∃ n : ℕ, n = 16 ∧ ∀ x : ℕ, x ∈ {x | x >= 1000 ∧ x < 10000 ∧ x.digits 10 ⊆ {3, 7}} ↔ x ∈ set.range (λ (i : ℕ), 1000 + 1000 * (i / 8) + 100 * ((i / 4) % 2) + 10 * ((i / 2) % 2) + (i % 2)) ∧ i < 16 :=
sorry

end four_digit_integers_with_3_or_7_only_l126_126511


namespace strictly_increasing_range_l126_126492

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x + 1 else a ^ x

theorem strictly_increasing_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 < a ∧ a ≤ 2) :=
sorry

end strictly_increasing_range_l126_126492


namespace probability_exactly_half_girls_l126_126962

noncomputable def binomial_distribution (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_exactly_half_girls (n : ℕ) (p : ℝ) (h : n = 7 ∧ p = 0.5) :
  (binomial_distribution n 3 p + binomial_distribution n 4 p) = 35 / 64 := 
by
  rw [h.1, h.2]
  sorry

end probability_exactly_half_girls_l126_126962


namespace binomial_converges_to_poisson_l126_126240

noncomputable def binomial_distribution (n m : ℕ) (λ : ℝ) : ℝ :=
(nat.factorial n) / ((nat.factorial m) * (nat.factorial (n - m))) * ((λ / n) ^ m) * ((1 - (λ / n)) ^ (n - m))

noncomputable def poisson_distribution (m : ℕ) (λ : ℝ) : ℝ :=
(λ ^ m) * real.exp (-λ) / (nat.factorial m)

theorem binomial_converges_to_poisson (m : ℕ) (λ : ℝ) :
  (λ n, binomial_distribution n m λ) ⟶ (poisson_distribution m λ) := 
begin
  sorry
end

end binomial_converges_to_poisson_l126_126240


namespace driving_time_ratio_l126_126574

theorem driving_time_ratio 
  (t : ℝ)
  (h : 30 * t + 60 * (2 * t) = 75) : 
  t / (2 * t) = 1 / 2 := 
by
  sorry

end driving_time_ratio_l126_126574


namespace f_negative_l126_126951

-- Define the function f(x)
def f (a x : ℝ) : ℝ :=
  (a-x)^6 - 3*a*(a-x)^5 + (5/2) * a^2 * (a-x)^4 - (1/2) * a^4 * (a-x)^2

-- Define the conditions
variable (a x : ℝ)
variable (h₀ : 0 < x)
variable (h₁ : x < a)

-- The theorem to be proven
theorem f_negative (h₀ : 0 < x) (h₁ : x < a) : f a x < 0 :=
sorry

end f_negative_l126_126951


namespace find_number_l126_126683

theorem find_number 
  (x : ℝ)
  (h : (1 / 10) * x - (1 / 1000) * x = 700) :
  x = 700000 / 99 :=
by 
  sorry

end find_number_l126_126683


namespace machine_value_correct_l126_126354

-- The present value of the machine
def present_value : ℝ := 1200

-- The depreciation rate function based on the year
def depreciation_rate (year : ℕ) : ℝ :=
  match year with
  | 1 => 0.10
  | 2 => 0.12
  | n => if n > 2 then 0.10 + 0.02 * (n - 1) else 0

-- The repair rate
def repair_rate : ℝ := 0.03

-- Value of the machine after n years
noncomputable def machine_value_after_n_years (initial_value : ℝ) (n : ℕ) : ℝ :=
  let value_first_year := (initial_value - (depreciation_rate 1 * initial_value)) + (repair_rate * initial_value)
  let value_second_year := (value_first_year - (depreciation_rate 2 * value_first_year)) + (repair_rate * value_first_year)
  match n with
  | 1 => value_first_year
  | 2 => value_second_year
  | _ => sorry -- Further generalization would be required for n > 2

-- Theorem statement
theorem machine_value_correct (initial_value : ℝ) :
  machine_value_after_n_years initial_value 2 = 1015.56 := by
  sorry

end machine_value_correct_l126_126354


namespace guides_and_tourists_groupings_l126_126667

open Nat

/-- Three tour guides are leading eight tourists. Each tourist must choose one of the guides, 
    but with the stipulation that each guide must take at least one tourist. Prove 
    that the number of different groupings of guides and tourists is 5796. -/
theorem guides_and_tourists_groupings : 
  let total_groupings := 3 ^ 8,
      at_least_one_no_tourists := binom 3 1 * 2 ^ 8,
      at_least_two_no_tourists := binom 3 2 * 1 ^ 8,
      total_valid_groupings := total_groupings - at_least_one_no_tourists + at_least_two_no_tourists
  in total_valid_groupings = 5796 :=
by
  sorry

end guides_and_tourists_groupings_l126_126667


namespace order_of_a_b_c_l126_126820

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := (1 / 2) * (Real.log 5 / Real.log 2)

theorem order_of_a_b_c : a > c ∧ c > b :=
by
  -- proof here
  sorry

end order_of_a_b_c_l126_126820


namespace number_of_girls_who_left_l126_126028

-- Definitions for initial conditions and event information
def initial_boys : ℕ := 24
def initial_girls : ℕ := 14
def final_students : ℕ := 30

-- Main theorem statement translating the problem question
theorem number_of_girls_who_left (B G : ℕ) (h1 : B = G) 
  (h2 : initial_boys + initial_girls - B - G = final_students) :
  G = 4 := 
sorry

end number_of_girls_who_left_l126_126028


namespace investment_three_years_ago_l126_126287

noncomputable def initial_investment (final_amount : ℝ) : ℝ :=
  final_amount / (1.08 ^ 3)

theorem investment_three_years_ago :
  abs (initial_investment 439.23 - 348.68) < 0.01 :=
by
  sorry

end investment_three_years_ago_l126_126287


namespace area_of_T_l126_126596

-- Definition of a five-presentable complex number
def is_five_presentable (z : ℂ) : Prop :=
  ∃ (w : ℂ), (abs w = 5) ∧ (z = w - 1 / w)

-- Definition of the set T consisting of all five-presentable complex numbers
def T : set ℂ :=
  {z : ℂ | is_five_presentable z}

-- Theorem stating the area inside T
theorem area_of_T : measure_theory.measure_theory.inner_measure 2 ( T ) = (624 / 25) * real.pi :=
sorry

end area_of_T_l126_126596


namespace rectangle_length_width_l126_126632

-- Given conditions
variables (L W : ℕ)

-- Condition 1: The area of the rectangular field is 300 square meters
def area_condition : Prop := L * W = 300

-- Condition 2: The perimeter of the rectangular field is 70 meters
def perimeter_condition : Prop := 2 * (L + W) = 70

-- Condition 3: One side of the rectangle is 20 meters
def side_condition : Prop := L = 20

-- Conclusion
def length_width_proof : Prop :=
  L = 20 ∧ W = 15

-- The final mathematical proof problem statement
theorem rectangle_length_width (L W : ℕ) 
  (h1 : area_condition L W) 
  (h2 : perimeter_condition L W) 
  (h3 : side_condition L) : 
  length_width_proof L W :=
sorry

end rectangle_length_width_l126_126632


namespace range_of_a_l126_126646

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≤ 3) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 3) ∧ (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≥ 2) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 2) ↔ 1 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end range_of_a_l126_126646


namespace min_value_of_b_l126_126559

theorem min_value_of_b (a b : ℝ) (h : a > 0) :
  (∃ x, (y = a * log x → y = 2 * x + b)) ∧ (y = 2 * x + b ∧ y = a * log x) → b = -2 := 
begin
  sorry
end

end min_value_of_b_l126_126559


namespace smallest_angle_opposite_smallest_side_l126_126897

theorem smallest_angle_opposite_smallest_side 
  (a b c : ℝ) 
  (h_triangle : triangle_inequality_proof)
  (h_condition : 3 * a = b + c) :
  smallest_angle_proof :=
sorry

end smallest_angle_opposite_smallest_side_l126_126897


namespace find_n_coefficient_x2_rational_terms_in_expansion_l126_126482

open BigOperators

noncomputable def T (n r : ℕ) (x : ℕ) : ℚ :=
  (-1 / 2) ^ r * Nat.choose n r * x ^ ((n - 2 * r) / 3)

theorem find_n (const_term : ℚ) (h_const : T 10 5 const_term = 1) : n = 10 := 
by
  sorry

theorem coefficient_x2 (h_n : n = 10) : 
  T 10 2 x = 45 / 4 := 
by
  sorry

theorem rational_terms_in_expansion (h_n : n = 10) : 
  ∀ r, T 10 r x ∈ { 45 / 4 * x ^ 2, -63 / 8, 45 / 256 * x ^ (-2) } := 
by
  sorry

end find_n_coefficient_x2_rational_terms_in_expansion_l126_126482


namespace simplify_ratio_l126_126434

theorem simplify_ratio (n : ℕ) (h : 0 < n) :
  let a_n := ∑ k in Finset.range (n + 1), 1 / (Nat.choose n k)^2
  let b_n := ∑ k in Finset.range (n + 1), k^2 / (Nat.choose n k)^2
  a_n / b_n = 2 / n^2 :=
by
  -- The proof will be inserted here
  sorry

end simplify_ratio_l126_126434


namespace count_valid_4_digit_odd_numbers_l126_126507

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def no_consecutive_same (n : ℕ) : Prop :=
  ∀ i j, i < 4 → j < 4 → i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)

def all_odd_digits (n : ℕ) : Prop :=
  (n / 1000 % 10) ∈ {1,3,5,7,9} ∧
  (n / 100 % 10) ∈ {1,3,5,7,9} ∧
  (n / 10 % 10) ∈ {1,3,5,7,9} ∧
  (n % 10) ∈ {1,3,5,7,9}

def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem count_valid_4_digit_odd_numbers :
  ∃ n, n = 107 ∧ n = (card {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ all_odd_digits x ∧ no_consecutive_same x ∧ divisible_by_3 x}) :=
begin
  sorry
end

end count_valid_4_digit_odd_numbers_l126_126507


namespace book_configurations_l126_126010

theorem book_configurations : 
  (∃ (configurations : Finset ℕ), configurations = {1, 2, 3, 4, 5, 6, 7} ∧ configurations.card = 7) 
  ↔ 
  (∃ (n : ℕ), n = 7) :=
by 
  sorry

end book_configurations_l126_126010


namespace minimum_w_value_l126_126543

theorem minimum_w_value : 
  (∀ (w : ℝ), (0 < w) → ∃ (n : ℕ), n ≥ 50 ∧ ∀ x ∈ set.Icc 0 1, (sin (w * x)) = 1 → (↑n : ℕ) ≤ ⌊ 1 / (2 * π / w) ⌋ ) ↔ (w = 100 * π)
 := by
 sorry

end minimum_w_value_l126_126543


namespace wire_length_l126_126985

theorem wire_length (dist height_short height_tall : ℕ) (h1 : dist = 20) (h2 : height_short = 10) (h3 : height_tall = 22) : 
  sqrt (dist^2 + (height_tall - height_short)^2) = sqrt 544 := by
  sorry

end wire_length_l126_126985


namespace trapezoid_sides_trapezoid_area_impossible_l126_126996

variables {α : Type*} [Real uniforms]

-- Let the sides of the trapezoid be a, b, c, and d with a = 13
noncomputable def is_trapezoid (a b c d : ℝ) : Prop :=
  a = 13 ∧ (a + b + c + d = 28) ∧ (1 / 2) * h * (a + c) = 27 

theorem trapezoid_sides (a b c d : ℝ) (h : ℝ) :
  is_trapezoid a b c d → (a, b, c, d) = (13, 5, 5, 5) ∧ h = 3 :=
sorry

theorem trapezoid_area_impossible (a b c d : ℝ) (h : ℝ) :
  is_trapezoid a b c d → ¬ (1 / 2) * h * (a + c) = 27.001 :=
sorry

end trapezoid_sides_trapezoid_area_impossible_l126_126996


namespace number_of_solutions_l126_126871

theorem number_of_solutions :
  { x : ℝ | |x - 2| = |x - 3| + |x - 6| + 2 }.finite.toFinset.card = 2 :=
sorry

end number_of_solutions_l126_126871


namespace evaluate_expression_right_to_left_l126_126568

variable (a b c d : ℝ)

theorem evaluate_expression_right_to_left:
  (a * b + c - d) = (a * (b + c - d)) :=
by {
  -- Group operations from right to left according to the given condition
  sorry
}

end evaluate_expression_right_to_left_l126_126568


namespace value_of_a_pow_sum_l126_126875

variable {a : ℝ}
variable {m n : ℕ}

theorem value_of_a_pow_sum (h1 : a^m = 5) (h2 : a^n = 3) : a^(m + n) = 15 := by
  sorry

end value_of_a_pow_sum_l126_126875


namespace part1_part2_part3_l126_126520

noncomputable def m := 1
noncomputable def a := (3 * m - 1) ^ 2

theorem part1 : 3 * m - 1 = 7 - 5 * m := by
  sorry

theorem part2 : a = 4 := by
  calc
    a = (3 * m - 1)^2 : by rfl
    ... = (3 * 1 - 1)^2 : by rw [m]
    ... = 2^2 : by rfl
    ... = 4 : by rfl

theorem part3 : - real.cbrt (-4) = real.cbrt 4 := by
  sorry

end part1_part2_part3_l126_126520


namespace system_equations_solution_l126_126627

theorem system_equations_solution (x y z : ℝ) 
  (h1 : sqrt (2 * x^2 + 2) = y + 1)
  (h2 : sqrt (2 * y^2 + 2) = z + 1)
  (h3 : sqrt (2 * z^2 + 2) = x + 1) : 
  x = 1 ∧ y = 1 ∧ z = 1 :=
sorry

end system_equations_solution_l126_126627


namespace find_prob_complement_l126_126084

variable {Ω : Type*} [MeasurableSpace Ω] (P : ProbabilityMeasure Ω)

variable (A B : Set Ω)

noncomputable def problem_conditions :=
P(B) = 0.3 ∧ P(B ∩ A) / P(A) = 0.9 ∧ P(B ∩ Aᶜ) / P(Aᶜ) = 0.2

theorem find_prob_complement (h : problem_conditions P A B) : 
  P(Aᶜ) = 6 / 7 := sorry

end find_prob_complement_l126_126084


namespace quadratic_form_completion_l126_126652

theorem quadratic_form_completion (b c : ℤ)
  (h : ∀ x:ℂ, x^2 + 520*x + 600 = (x+b)^2 + c) :
  c / b = -258 :=
by sorry

end quadratic_form_completion_l126_126652


namespace lines_perpendicular_l126_126200

-- Definitions of the geometric entities based on the conditions
variables {A B C B' I M K L : Type*}
variables (circumcircle : ∀ T, set T → Prop) (incircle : ∀ T, set T → Prop)
variables [linear_ordered_field ℝ]
variables (triangle : Type*) [metric_space triangle] 
variables (side : triangle → ℝ → Prop)
variables (incenter : triangle → triangle) 
variables (tangent_point : triangle → triangle → triangle)

-- Conditions
hypotheses (h1 : circumcircle A C B B')
(h2 : incenter ABC = I)
(h3 : tangent_point incircle ABC AC = M)
(h4 : B' ∈ circumcircle ABC ∧ ∃ X, midpoint B X = ∅)
(h5 : K ∈ side AB)
(h6 : L ∈ side BC)
(h7 : dist K B = dist M C)
(h8 : dist L B = dist A M)

-- Proof statement
theorem lines_perpendicular : ∀ {B' I K L},
  is_perpendicular (line_through_points B I) (line_through_points K L) :=
sorry

end lines_perpendicular_l126_126200


namespace circumcenters_cyclic_l126_126764

-- Definitions for circumcenters in Euclidean geometry
noncomputable def circumcenter (A B C : Point) : Point := sorry

theorem circumcenters_cyclic (A B C P O O1 O2 : Point)
  (h1: circumcenter A B C = O)
  (h2: circumcenter A B P = O1)
  (h3: circumcenter A C P = O2)
  (h_AP_intersect_BC: P ∈ Line A (LineSegment B C)) :
  geom.circle (circle.center O) (circle.radius O) A O1 O2 :=
sorry

end circumcenters_cyclic_l126_126764


namespace range_of_m_l126_126888

theorem range_of_m (x m : ℝ) :
  ( (x + m) / (x - 3) + (3 * m) / (3 - x) = 3 ∧ x > 0 ) ↔ ( m < 9 / 2 ∧ m ≠ 3 / 2 ) :=
begin
  sorry
end

end range_of_m_l126_126888


namespace geometric_sequence_S4_over_a3_l126_126840

variable {a : ℕ → ℝ} -- geometric sequence
variable {S : ℕ → ℝ} -- sum of the first n terms

-- definition of the common ratio
def common_ratio (q : ℝ) := ∀ n : ℕ, a (n + 1) = q * a n

-- definition of the sum of the first n terms
def sum_of_terms (S : ℕ → ℝ) := ∀ n, S n = (a 0 * (1 - 2^n)) / (1 - 2)

theorem geometric_sequence_S4_over_a3 (q : ℝ := 2) (h1 : common_ratio q) (h2 : sum_of_terms S) :
  S 4 / a 3 = 15 / 4 :=
by
  sorry

end geometric_sequence_S4_over_a3_l126_126840


namespace geometric_mean_unique_solution_l126_126108

-- Define the conditions
variable (k : ℕ) -- k is a natural number
variable (hk_pos : 0 < k) -- k is a positive natural number

-- The geometric mean condition translated to Lean
def geometric_mean_condition (k : ℕ) : Prop :=
  (2 * k)^2 = (k + 9) * (6 - k)

-- The main statement to prove
theorem geometric_mean_unique_solution (k : ℕ) (hk_pos : 0 < k) (h: geometric_mean_condition k) : k = 3 :=
sorry -- proof placeholder

end geometric_mean_unique_solution_l126_126108


namespace parabola_range_for_AB_l126_126352

-- Given Condition definitions
def parabola (p : ℝ) : Prop := p > 0

def intersects_parabola (p : ℝ) (k : ℝ) : Prop :=
  let line := λ x, k * (x - p / 2)
  let parabola := λ x, (2 * p * x)
  ∃ A B : ℝ × ℝ, A ≠ B ∧ line (fst A) = snd A ∧ parabola (fst A) = snd A
                   ∧ line (fst B) = snd B ∧ parabola (fst B) = snd B ∧ 
                   dist A B = 4

-- The main proof statement
theorem parabola_range_for_AB (p : ℝ) (k : ℝ) : 
  parabola p ∧ intersects_parabola p k → 0 < p ∧ p < 2 :=
by
  sorry

end parabola_range_for_AB_l126_126352


namespace circle_properties_l126_126823

noncomputable def circle_center_on_line (A B: ℝ × ℝ) (c_line_eqn : ℝ → ℝ → Prop) :=
  ∃ C : ℝ × ℝ, 
    (C.1 + 3 * C.2 - 15 = 0) ∧ 
    (dist C A = dist C B)

theorem circle_properties 
  (A B C : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (3, 4))
  (hCenter : C.1 + 3 * C.2 - 15 = 0)
  (hDist : dist C A = dist C B)
  : let r := sqrt (((C.1 - A.1)^2 + (C.2 - A.2)^2)) in
    (C = (0, 5) ∧ r = sqrt 10 ∧ ∀ (P : ℝ × ℝ) (hP_on_circle: (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2), 
      let h_max := sqrt 5 + sqrt 10 in 
      let AB_dist := dist A B in
        (1/2 * AB_dist * h_max = 5 + 5 * sqrt 2)) :=
funit sorry

end circle_properties_l126_126823


namespace find_a_plus_b_l126_126054

theorem find_a_plus_b (a b : ℚ)
  (h1 : 3 = a + b / (2^2 + 1))
  (h2 : 2 = a + b / (1^2 + 1)) :
  a + b = 1 / 3 := 
sorry

end find_a_plus_b_l126_126054


namespace students_exceed_pets_by_70_l126_126377

theorem students_exceed_pets_by_70 :
  let n_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let hamsters_per_classroom := 5
  let total_students := students_per_classroom * n_classrooms
  let total_rabbits := rabbits_per_classroom * n_classrooms
  let total_hamsters := hamsters_per_classroom * n_classrooms
  let total_pets := total_rabbits + total_hamsters
  total_students - total_pets = 70 :=
  by
    sorry

end students_exceed_pets_by_70_l126_126377


namespace routes_through_checkpoint_l126_126509

theorem routes_through_checkpoint (A B C : ℕ × ℕ) :
  -- assumptions on positions of A, B, C
  A = (0, 15) ∧ B = (15, 0) ∧ C = (10, 10) →
  -- movements only right or down
  (∀ p, p = A ∨ p = B ∨ p = C ∨
    ((p.fst = 0 ∨ p.fst = 5 ∨ p.fst = 10 ∨ p.fst = 15) 
    ∧ (p.snd = 0 ∨ p.snd = 5 ∨ p.snd = 10 ∨ p.snd = 15))) →
  -- number of routes from A to B through C
  (number_of_routes A C * number_of_routes C B = 30) :=
sorry

end routes_through_checkpoint_l126_126509


namespace total_amount_correct_l126_126216

-- Define the prices of jeans and tees
def price_jean : ℕ := 11
def price_tee : ℕ := 8

-- Define the quantities sold
def quantity_jeans_sold : ℕ := 4
def quantity_tees_sold : ℕ := 7

-- Calculate the total amount earned
def total_amount : ℕ := (price_jean * quantity_jeans_sold) + (price_tee * quantity_tees_sold)

-- Now, we state and prove the theorem
theorem total_amount_correct : total_amount = 100 :=
by
  -- Here we assert the correctness of the calculation
  sorry

end total_amount_correct_l126_126216


namespace ratio_of_average_speed_to_still_water_speed_l126_126004

noncomputable def speed_of_current := 6
noncomputable def speed_in_still_water := 18
noncomputable def downstream_speed := speed_in_still_water + speed_of_current
noncomputable def upstream_speed := speed_in_still_water - speed_of_current
noncomputable def distance_each_way := 1
noncomputable def total_distance := 2 * distance_each_way
noncomputable def time_downstream := (distance_each_way : ℝ) / (downstream_speed : ℝ)
noncomputable def time_upstream := (distance_each_way : ℝ) / (upstream_speed : ℝ)
noncomputable def total_time := time_downstream + time_upstream
noncomputable def average_speed := (total_distance : ℝ) / (total_time : ℝ)
noncomputable def ratio_average_speed := (average_speed : ℝ) / (speed_in_still_water : ℝ)

theorem ratio_of_average_speed_to_still_water_speed :
  ratio_average_speed = (8 : ℝ) / (9 : ℝ) :=
sorry

end ratio_of_average_speed_to_still_water_speed_l126_126004


namespace max_lambda_for_arithmetic_sequence_l126_126919

theorem max_lambda_for_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) :
  (∀ (n : ℕ), S n = n * (a 1 + a n) / 2) →
  (∀ (n : ℕ), n^2 * (a n)^2 + 4 * (S n)^2 ≥ λ * n^2 * (a 1)^2) →
  λ ≤ 1/2 :=
by
  intros h1 h2
  sorry

end max_lambda_for_arithmetic_sequence_l126_126919


namespace circle_equations_l126_126825

theorem circle_equations {A : ℝ × ℝ} (hA : A = (2, -3))
  (line : ℝ → ℝ → Prop) (hline : ∀ x y, line x y ↔ x + 2 * y = 0)
  (h_symmetric : ∀ B : ℝ × ℝ, symmetric_with_respect_to_line A line B → is_on_circle B)
  (h_tangent_y_axis : ∀ center : ℝ × ℝ, tangent_to_y_axis center)
  : ∃ center₁ radius₁ center₂ radius₂ : ℝ, 
    circle_equation center₁ radius₁ = "(x - 2)^2 + (y + 1)^2 = 4" ∧ 
    circle_equation center₂ radius₂ = "(x - 26)^2 + (y + 13)^2 = 676" :=
begin
  sorry
end

end circle_equations_l126_126825


namespace function_positive_domain_l126_126442

noncomputable def f (x : ℝ) : ℝ := real.root 4 (10 + x) - real.sqrt (2 - x)

theorem function_positive_domain :
  ∀ x : ℝ, -10 ≤ x ∧ x ≤ 2 → (f x > 0 ↔ -1 < x ∧ x ≤ 2) :=
by
  intro x hx
  sorry

end function_positive_domain_l126_126442


namespace inequality1_inequality2_l126_126966

theorem inequality1 (x : ℝ) : x ≠ 2 → (x + 1)/(x - 2) ≥ 3 → 2 < x ∧ x ≤ 7/2 :=
sorry

theorem inequality2 (x a : ℝ) : 
  (x^2 - a * x - 2 * a^2 ≤ 0) → 
  (a = 0 → x = 0) ∧ 
  (a > 0 → -a ≤ x ∧ x ≤ 2 * a) ∧ 
  (a < 0 → 2 * a ≤ x ∧ x ≤ -a) :=
sorry

end inequality1_inequality2_l126_126966


namespace min_max_rectangle_perimeter_l126_126393

-- Definitions representing the ellipse and the properties of the rectangles.
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- The smallest perimeter rectangle definition
def smallest_perimeter_rectangle (a b : ℝ) : ℝ :=
  2 * (a + b)

-- The largest perimeter rectangle definition (which is a square)
def largest_perimeter_rectangle (a b : ℝ) : ℝ :=
  4 * real.sqrt (a^2 + b^2)

-- The theorem stating that these are the smallest and largest perimeters
theorem min_max_rectangle_perimeter (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ u v, ellipse a b u v ∧ smallest_perimeter_rectangle a b = 2 * b ∧ largest_perimeter_rectangle a b = 4 * real.sqrt (a^2 + b^2) :=
by
  sorry

end min_max_rectangle_perimeter_l126_126393


namespace remaining_money_after_purchases_l126_126942

-- Define the initial amount of money Mrs. Hilt had
def initial_amount : ℝ := 5.00

-- Define the prices of items
def price_per_pencil : ℝ := 0.11
def price_of_notebook : ℝ := 1.45
def price_of_colored_pencils : ℝ := 2.50

-- Define the discount rate for colored pencils
def discount_rate : ℝ := 0.10

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the exchange rate from EUR to USD
def exchange_rate : ℝ := 1.10

-- Define the expected money left in USD
def expected_money_left_in_usd : ℝ := 0.72

-- Prove that the remaining money in USD is as expected
theorem remaining_money_after_purchases :
  let total_cost_pencils := 3 * price_per_pencil in
  let discounted_colored_pencils := price_of_colored_pencils * (1 - discount_rate) in
  let total_before_tax := total_cost_pencils + price_of_notebook + discounted_colored_pencils in
  let sales_tax := (total_before_tax * sales_tax_rate).round in
  let total_incl_tax := total_before_tax + sales_tax in
  let remaining_money := initial_amount - total_incl_tax in
  let remaining_money_usd := remaining_money * exchange_rate in
  remaining_money_usd.round = expected_money_left_in_usd :=
by {
  sorry
}

end remaining_money_after_purchases_l126_126942


namespace minimum_norm_value_l126_126203

noncomputable theory

open Complex

-- Define the condition |z - 8| + |z - 7 * I| = 15
def satisfies_condition (z : ℂ) : Prop :=
  abs (z - 8) + abs (z - 7 * I) = 15

-- The main theorem: proving the minimum possible value of |z|
theorem minimum_norm_value : ∃ z : ℂ, satisfies_condition z ∧ abs z = 56 / 15 :=
  sorry

end minimum_norm_value_l126_126203


namespace find_p_plus_q_l126_126292

noncomputable def problem (X Y E F : Point) (XY EY FX : ℝ) (EX FY : ℝ) : Prop :=
  XY = 7 ∧ EY = 15 ∧ FX = 15 ∧ EX = 13 ∧ FY = 13 ∧ 
  ∃ p q : ℕ, (gcd p q = 1) ∧ (area_of_intersection X Y E F XY EY EX FX = p / q) ∧ (p + q = 179)

theorem find_p_plus_q :
  ∀ (X Y E F : Point) (XY EY FX : ℝ) (EX FY : ℝ),
  problem X Y E F XY EY FX EX FY → 
  ∃ (p q : ℕ), (gcd p q = 1) ∧ area_of_intersection X Y E F XY EY EX FX = p/q ∧ (p + q = 179) :=
by
  intro X Y E F XY EY FX EX FY
  intro h
  sorry

end find_p_plus_q_l126_126292


namespace sequence_inequality_l126_126699

theorem sequence_inequality (a : ℕ → ℝ) 
  (h₀ : a 0 = 5) 
  (h₁ : ∀ n, a (n + 1) * a n - a n ^ 2 = 1) : 
  35 < a 600 ∧ a 600 < 35.1 :=
sorry

end sequence_inequality_l126_126699


namespace number_of_lines_at_distances_l126_126561

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / sqrt (A * A + B * B)

def line_through_points (A₁ A₂ : Point) (distance: ℝ) : Prop :=
  ∃ (A B C : ℝ), distance_point_to_line A₁.x A₁.y A B C = distance ∧ distance_point_to_line A₂.x A₂.y A B C = distance

noncomputable def num_lines_through_points_with_distances (A B : Point) (d₁ d₂ : ℝ) : ℕ :=
  4

theorem number_of_lines_at_distances (A B : Point) (d₁ d₂ : ℝ) :
  A = ⟨1, 3⟩ → B = ⟨3, 1⟩ → d₁ = 2 → d₂ = 3 →
  num_lines_through_points_with_distances A B d₁ d₂ = 4 :=
by
  intros hA hB hd₁ hd₂
  rw [hA, hB, hd₁, hd₂]
  exact rfl
  -- Here, the proof would follow
  sorry

end number_of_lines_at_distances_l126_126561


namespace midpoint_locus_l126_126110

-- Definitions of the cube and its properties
variables {A B C D A' B' C' D' X Y Z : point}
variables {ABCD : square} {A'B'C'D' : square} {cube : set point}
variables (perimeter : set (point))

-- Conditions on motion and starting points
axiom CubeStructure : cube = {A, B, C, D, A', B', C', D'}
axiom ParallelLines : AA' ∥ BB' ∧ BB' ∥ CC' ∧ CC' ∥ DD'
axiom XPath : ∀ t, X t ∈ perimeter ABCD
axiom YPath : ∀ t, Y t ∈ perimeter B'C'C B
axiom SameSpeed : constant_speed X ∧ constant_speed Y
axiom StartPoints : X 0 = A ∧ Y 0 = B'

-- Locus of midpoint Z
theorem midpoint_locus : 
  locus_of_midpoint Z X Y = perimeter EFCG := 
sorry

end midpoint_locus_l126_126110


namespace scalar_d_l126_126064

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (i j v : V)
def u : V := i + j

theorem scalar_d : (i × (v × i)) + (j × (v × j)) + (u i j) × (v × (u i j)) = 2 • v :=
sorry

end scalar_d_l126_126064


namespace min_number_of_male_patients_team_B_correct_choice_l126_126342

section
  variables (x m : ℕ) (p q : ℝ) (alpha : ℝ)

  -- Condition definitions
  def num_females := 2 * x
  def males_type_I := (5 * x) / 6
  def females_type_I := (2 * x) / 3
  def males_type_II := x / 6
  def females_type_II := (4 * x) / 3

  -- Problem decompositions
  def K_sq := (3 * x * (males_type_I * females_type_II - males_type_II * females_type_I) ^ 2) / ((3 * x / 2) ^ 2 * 2 * x * x)
  def alpha_crit_val := 7.897

  def team_A_expected_cost := -2 * m * p^2 + 6 * m
  def team_B_expected_cost := 6 * (2 * m / 3) * q^3 - 9 * (2 * m / 3) * q^2 + 6 * (2 * m / 3)

  -- Proof goals
  theorem min_number_of_male_patients (hx : K_sq > alpha_crit_val) : x = 12 := sorry

  theorem team_B_correct_choice (hn : n = (2 * m) / 3) (hpq : p = q) (hq1 : 0 < q) (hq2 : q < 1) :
    team_B_expected_cost < team_A_expected_cost := sorry
end

end min_number_of_male_patients_team_B_correct_choice_l126_126342


namespace square_area_eq_36_l126_126364

theorem square_area_eq_36 :
  let triangle_side1 := 5.5
  let triangle_side2 := 7.5
  let triangle_side3 := 11
  let triangle_perimeter := triangle_side1 + triangle_side2 + triangle_side3
  let square_perimeter := triangle_perimeter
  let square_side_length := square_perimeter / 4
  let square_area := square_side_length * square_side_length
  square_area = 36 := by
  sorry

end square_area_eq_36_l126_126364


namespace sum_of_squares_formula_l126_126099

theorem sum_of_squares_formula (n : ℕ) (hn : n > 0) : 
  (Finset.range (n+1)).sum (λ i, i^2) = (1 / 6) * n * (n + 1) * (2 * n + 1) := by
  sorry

end sum_of_squares_formula_l126_126099


namespace rhombus_perimeter_l126_126326

-- Define the conditions for the rhombus
variable (d1 d2 : ℝ) (a b s : ℝ)

-- State the condition that the diagonals of a rhombus measure 24 cm and 10 cm
def diagonal_condition := (d1 = 24) ∧ (d2 = 10)

-- State the Pythagorean theorem for the lengths of half-diagonals
def pythagorean_theorem := a^2 + b^2 = s^2

-- State the relationship of diagonals bisecting each other at right angles
def bisect_condition := (a = d1 / 2) ∧ (b = d2 / 2)

-- State the definition of the perimeter for a rhombus
def perimeter (s : ℝ) : ℝ := 4 * s

-- The theorem we want to prove
theorem rhombus_perimeter : diagonal_condition d1 d2 →
                            bisect_condition d1 d2 a b →
                            pythagorean_theorem a b s →
                            perimeter s = 52 :=
by
  intros h1 h2 h3
  -- Proof would go here, but it is omitted
  sorry

end rhombus_perimeter_l126_126326


namespace members_per_team_l126_126742

-- Define the number of teams
def num_teams : ℕ := 4

-- Each member needs 2 pairs of skates
def pairs_per_member : ℕ := 2

-- Each pair of skates needs 3 sets of laces
def laces_per_pair : ℕ := 3

-- Total amount of sets of laces distributed
def total_laces : ℕ := 240

-- Prove that the number of members in each team is 10
theorem members_per_team (num_teams pairs_per_member laces_per_pair total_laces : ℕ) (H : num_teams = 4 ∧ pairs_per_member = 2 ∧ laces_per_pair = 3 ∧ total_laces = 240) : 
  let M := 10 in
  30*num_teams*pairs_per_member*laces_per_pair = total_laces :=
by sorry

end members_per_team_l126_126742


namespace find_initial_distance_between_trains_l126_126674

noncomputable def initial_distance_between_trains
  (len_train1 : ℝ) (len_train2 : ℝ)
  (speed_train1 : ℝ) (speed_train2 : ℝ)
  (time_to_meet : ℝ) : ℝ :=
  let relative_speed := speed_train1 + speed_train2 in
  let distance_covered := relative_speed * time_to_meet in
  distance_covered - (len_train1 + len_train2)

theorem find_initial_distance_between_trains :
  initial_distance_between_trains 100 200 25 20 9.99920006399488 = 149.964002879744 := 
by
  sorry

end find_initial_distance_between_trains_l126_126674


namespace exponent_problem_l126_126524

variable (x m n : ℝ)
variable (h1 : x^m = 3)
variable (h2 : x^n = 5)

theorem exponent_problem : x^(2 * m - 3 * n) = 9 / 125 :=
by 
  sorry

end exponent_problem_l126_126524


namespace find_number_l126_126005

theorem find_number (x : ℝ) (h : x / 0.07 = 700) : x = 49 :=
sorry

end find_number_l126_126005


namespace quadrilateral_area_l126_126235

/-- Points K, L, M divide the sides of a convex quadrilateral ABCD in the ratio AK:BK = CL:BL = CM:DM = 1:2.
    The radius of the circumcircle of triangle KLM is 5/2, KL = 4, and LM = 3. 
    Prove that the area of quadrilateral ABCD is 189/25 given KM < KL. -/
theorem quadrilateral_area 
  (K L M A B C D : Point)
  (h1 : divides_ratio A B K 1 2)
  (h2 : divides_ratio C D M 1 2)
  (h3 : divides_ratio B L C 1 2)
  (R : ℝ)
  (KL LM : ℝ)
  (h4 : circumradius K L M = 5 / 2)
  (h5 : KL = 4)
  (h6 : LM = 3)
  (h7 : KM < KL) :
  area ABCD = 189 / 25 := 
sorry

end quadrilateral_area_l126_126235


namespace cube_root_of_a_plus_one_l126_126278

theorem cube_root_of_a_plus_one (x a : ℕ) (h : a = x^2) : real.cbrt (a + 1) = real.cbrt (x^2 + 1) :=
by {
  rw [h],
  sorry
}

end cube_root_of_a_plus_one_l126_126278


namespace area_of_triangle_ABC_l126_126570

theorem area_of_triangle_ABC 
  (A B C D : Point)
  (h_angle_A : ∠A = 45)
  (h_altitude_CD : altitude D C A B)
  (h_CD_length : CD = 2)
  : (∃ s : ℝ, s = 2 * sqrt 2) :=
sorry

end area_of_triangle_ABC_l126_126570


namespace maximum_value_of_function_l126_126835

theorem maximum_value_of_function (x : ℝ) (h : x < 1 / 2) :
  (∀ y, y = 2 * x + 1 / (2 * x - 1) → y ≤ -1) :=
begin
  sorry
end

end maximum_value_of_function_l126_126835


namespace batsman_average_after_11th_inning_l126_126003

theorem batsman_average_after_11th_inning (x : ℝ) (h : 10 * x + 110 = 11 * (x + 5)) : 
    (10 * x + 110) / 11 = 60 := by
  sorry

end batsman_average_after_11th_inning_l126_126003


namespace arrangement_schemes_count_l126_126806

-- Given conditions
variable (students : Finset ℕ) (tasks : Finset ℕ) (taskAssignment : students → tasks)
variable (A B : ℕ) (C D E : ℕ)

-- Definitions for constraints
def valid_students : Finset ℕ := {A, B, C, D, E}
def valid_tasks : Finset ℕ := {reception, tour_guiding, translation, explanation}

-- Constraint that each task must have at least one person
def each_task_at_least_one_person (taskAssignment : students → tasks) (tasks : Finset ℕ) : Prop := 
  ∀ t ∈ tasks, ∃ s ∈ students, taskAssignment s = t

-- Constraint on A and B cannot be tour guides
def A_B_constraints (taskAssignment : students → tasks) : Prop :=
  taskAssignment A ≠ tour_guiding ∧ taskAssignment B ≠ tour_guiding

-- Constraint on C, D, E can handle all tasks
def C_D_E_constraints (taskAssignment : students → tasks) : Prop := 
  ∀ s ∈ {C, D, E}, taskAssignment s ∈ valid_tasks

-- The main theorem statement
theorem arrangement_schemes_count :
  each_task_at_least_one_person taskAssignment valid_tasks →
  A_B_constraints taskAssignment →
  C_D_E_constraints taskAssignment →
  (∃ arrangement_count : ℕ, arrangement_count = 126) := 
  by
    sorry

end arrangement_schemes_count_l126_126806


namespace pouring_out_beads_probability_comparison_l126_126758

-- Problem Statement
theorem pouring_out_beads_probability_comparison :
  let total_outcomes := 15 in
  let odd_outcomes := 8 in
  let even_outcomes := 7 in
  let p_odd := (8:ℝ) / (15:ℝ) in
  let p_even := (7:ℝ) / (15:ℝ) in
  p_odd > p_even :=
by
  sorry

end pouring_out_beads_probability_comparison_l126_126758


namespace derivative_of_y_l126_126425

variable (α x : ℝ)

def y : ℝ := (1 / (2 * sin (α / 2))) * arctan ((2 * x * sin (α / 2)) / (1 - x^2))

theorem derivative_of_y :
  deriv y x = (1 + x^2) / ((1 - x^2)^2 + 4 * x^2 * (sin (α / 2))^2) :=
by
  -- Proof goes here
  sorry

end derivative_of_y_l126_126425


namespace triplet_transformation_impossible_l126_126250

theorem triplet_transformation_impossible :
  let initial_triplet := (2 : ℝ, Real.sqrt 2, 1 / Real.sqrt 2)
  let final_triplet := (1 : ℝ, Real.sqrt 2, 1 + Real.sqrt 2)
  ∀ (a b c : ℝ), 
    (a, b, c) = initial_triplet → 
    (∀ x y z : ℝ,
      ((x = (a - b) / Real.sqrt 2 ∧ y = (a + b) / Real.sqrt 2 ∧ z = c) ∨
       (x = (a - c) / Real.sqrt 2 ∧ y = b ∧ z = (a + c) / Real.sqrt 2) ∨
       (x = a ∧ y = (b - c) / Real.sqrt 2 ∧ z = (b + c) / Real.sqrt 2)) →
      (x, y, z) ≠ final_triplet)
  sorry

end triplet_transformation_impossible_l126_126250


namespace angle_PA1Q_twice_angle_A_l126_126636

open EuclideanGeometry

/-- Given a triangle ABC with AB > AC, a circle passing through B and C intersecting the extensions of AB and AC at P and Q respectively, and AA_1 being the altitude of the triangle such that A_1P = A_1Q, prove that the angle PA_1Q is twice the angle BAC. -/
theorem angle_PA1Q_twice_angle_A (A B C P Q A1 : Point) (k : Circle) (h_triangle : Triangle A B C)
  (h_AB_gt_AC : A.dist B > A.dist C)
  (h_circle : k ∈ (Circle.through B C))
  (h_intersect_extensions_P Q : Intersects k (Line.extend_side A B B P) ∧ Intersects k (Line.extend_side A C C Q))
  (h_AA1_altitude : Altitude A A1 h_triangle)
  (h_A1P_eq_A1Q : A1.dist P = A1.dist Q) :
  ∠ P A1 Q = 2 * ∠ A :=
sorry

end angle_PA1Q_twice_angle_A_l126_126636


namespace chestnut_picking_l126_126601

theorem chestnut_picking 
  (P : ℕ)
  (h1 : 12 + P + (P + 2) = 26) :
  12 / P = 2 :=
sorry

end chestnut_picking_l126_126601


namespace coefficient_of_third_term_l126_126399

open Nat

theorem coefficient_of_third_term (x : ℝ) : 
  (let poly := (x - 1 / (2 * x)) ^ 6 in 
    nth_coeff poly 2 = 15 / 4) :=
by
  sorry

end coefficient_of_third_term_l126_126399


namespace find_smallest_n_l126_126050

theorem find_smallest_n :
  ∃ n : ℕ, n > 0 ∧ (∑ k in Finset.range (n + 1), Real.log (1 + 1 / 2 ^ (3 ^ k)) / Real.log 2) ≥ 
                 1 + (Real.log 500 - Real.log 501) / Real.log 2 :=
by
  sorry

end find_smallest_n_l126_126050


namespace ratio_of_time_charged_l126_126609

theorem ratio_of_time_charged (P K M : ℕ) (r : ℚ) 
  (h1 : P + K + M = 144) 
  (h2 : P = r * K)
  (h3 : P = 1/3 * M)
  (h4 : M = K + 80) : 
  r = 2 := 
  sorry

end ratio_of_time_charged_l126_126609


namespace length_CF_l126_126180

-- Given conditions
variables {A B C D F : Type}
variables [add_comm_group A] [module ℝ A] [inner_product_space ℝ A] [finite_dimensional ℝ A]

-- Point coordinates (Pretend these are concrete coordinates in affine space)
variables (A B C D F : A)
variables (AF : ℝ) (angleABF angleBCF angleCDF : ℝ)

-- Conditions as hypotheses
hypothesis h_rightAngleABF : (angleABF = real.pi / 4)
hypothesis h_rightAngleBCF : (angleBCF = real.pi / 4)
hypothesis h_rightAngleCDF : (angleCDF = real.pi / 4)
hypothesis h_AF_eq_32 : (AF = 32)

-- Theorem to prove
theorem length_CF : 
  ∃ (CF : ℝ), CF = 16 := 
sorry

end length_CF_l126_126180


namespace eccentricity_of_given_hyperbola_l126_126106

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : b = 2 * a) : ℝ :=
  Real.sqrt (1 + (b * b) / (a * a))

theorem eccentricity_of_given_hyperbola (a b : ℝ) 
  (h_hyperbola : b = 2 * a)
  (h_asymptote : ∃ k, k = 2 ∧ ∀ x, y = k * x → ((y * a) = (b * x))) :
  hyperbola_eccentricity a b h_hyperbola = Real.sqrt 5 :=
by
  sorry

end eccentricity_of_given_hyperbola_l126_126106


namespace find_selling_price_find_selling_price_example_l126_126651

def profit (sp cp : ℕ) : ℕ := sp - cp
def loss (cp sp : ℕ) : ℕ := cp - sp

theorem find_selling_price 
  (sp_profit : ℕ) (cp : ℕ) (profit_eq_loss : profit sp_profit cp = loss cp sp_loss) : 
  sp_loss = 42 := 
by
  sorry

-- Assign values based on given problem
def sp_profit := 86
def cp := 64
def sp_loss := 42

theorem find_selling_price_example 
    : sp_loss = 42 := 
by
  show sp_loss = 42
  -- Using the given problem information:
  have profit_ : profit sp_profit cp = 22 :=
    by
      show profit sp_profit cp = 22
      -- Simplify profit:
      simp [profit, sp_profit, cp]
  have loss_ : loss cp sp_loss = 22 :=
    by
      show loss cp sp_loss = 22
      -- Simplify loss:
      simp [loss, cp, sp_loss]
  exact congrArg (fun x => 64 - x) (congrArg (fun x => sp_loss - x) loss_)

end find_selling_price_find_selling_price_example_l126_126651


namespace min_fraction_sum_l126_126832

theorem min_fraction_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (∃ (z : ℝ), z = (1 / (x + 1)) + (4 / (y + 2)) ∧ z = 9 / 4) :=
by 
  sorry

end min_fraction_sum_l126_126832


namespace ratio_of_investments_l126_126270

theorem ratio_of_investments (P Q : ℝ)
  (h_ratio_profits : (20 * P) / (40 * Q) = 7 / 10) : P / Q = 7 / 5 := 
sorry

end ratio_of_investments_l126_126270


namespace option_A_true_l126_126089

theorem option_A_true (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : ac : a * c < 0) : a * b > a * c :=
by
  sorry

end option_A_true_l126_126089


namespace pow_mod_remainder_l126_126303

theorem pow_mod_remainder : (3 ^ 304) % 11 = 4 := by
  sorry

end pow_mod_remainder_l126_126303


namespace weavers_problem_l126_126248

theorem weavers_problem 
  (W : ℕ) 
  (H1 : 1 = W / 4) 
  (H2 : 3.5 = 49 / 14) :
  W = 4 :=
by
  sorry

end weavers_problem_l126_126248


namespace car_selling_price_l126_126007

def car_material_cost : ℕ := 100
def car_production_per_month : ℕ := 4
def motorcycle_material_cost : ℕ := 250
def motorcycles_sold_per_month : ℕ := 8
def motorcycle_selling_price : ℤ := 50
def additional_motorcycle_profit : ℤ := 50

theorem car_selling_price (x : ℤ) :
  (motorcycles_sold_per_month * motorcycle_selling_price - motorcycle_material_cost)
  = (car_production_per_month * x - car_material_cost + additional_motorcycle_profit) →
  x = 50 :=
by
  sorry

end car_selling_price_l126_126007


namespace exists_factor_between_10_and_20_l126_126140

theorem exists_factor_between_10_and_20 (n : ℕ) : ∃ k, (10 ≤ k ∧ k ≤ 20) ∧ k ∣ (2^n - 1) → k = 17 :=
by
  sorry

end exists_factor_between_10_and_20_l126_126140


namespace find_b_l126_126186

noncomputable def triangle_side_b (a : ℝ) (B : ℝ) (area : ℝ) : ℝ :=
  let C := 45 * (Real.pi / 180) -- convert degrees to radians
  have hB : B = 45 * (Real.pi / 180) := by sorry
  have h_area : (1 / 2) * a * b * (Real.sin C) = area := by sorry
  4 * Real.sqrt 2 -- using the derived solution for b

theorem find_b : triangle_side_b 1 (45 * (Real.pi / 180)) 2 = 4 * Real.sqrt 2 :=
by
  sorry

end find_b_l126_126186


namespace inequality_xyz_l126_126810

-- Problem and conditions
variables {x y z : ℝ}

-- Asserting that x, y, z are positive and their squares sum to 1
axiom (hx_pos : 0 < x)
axiom (hy_pos : 0 < y)
axiom (hz_pos : 0 < z)
axiom (h_sum_squares : x^2 + y^2 + z^2 = 1)

-- Statement to prove
theorem inequality_xyz (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x^2 + y^2 + z^2 = 1) :
  (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2)) ≥ (3 / 2 * Real.sqrt 3) :=
sorry

end inequality_xyz_l126_126810


namespace height_of_box_l126_126350

-- Definitions of given conditions
def length_box : ℕ := 9
def width_box : ℕ := 12
def num_cubes : ℕ := 108
def volume_cube : ℕ := 3
def volume_box : ℕ := num_cubes * volume_cube  -- Volume calculated from number of cubes and volume of each cube

-- The statement to prove
theorem height_of_box : 
  ∃ h : ℕ, volume_box = length_box * width_box * h ∧ h = 3 := by
  sorry

end height_of_box_l126_126350


namespace min_M_value_l126_126592

theorem min_M_value (x₁ x₂ x₃ x₄ x₅ : ℝ) (hx₁ : 0 ≤ x₁) (hx₂ : 0 ≤ x₂) (hx₃ : 0 ≤ x₃) (hx₄ : 0 ≤ x₄) (hx₅ : 0 ≤ x₅) (hsum : x₁ + x₂ + x₃ + x₄ + x₅ = 300) :
  let M := max (max (x₁ + x₂) (x₂ + x₃)) (max (x₃ + x₄) (x₄ + x₅))
  in M ≥ 100 :=
by
  sorry

end min_M_value_l126_126592


namespace quadrilateral_cyclic_l126_126288

theorem quadrilateral_cyclic (ABCD : Quadrilateral) (O : Point)
  (h_convex : convex ABCD)
  (h_circles : ∀ (A B : Point), isCircleThroughWithEqualRadius O [A, B] ABCD) :
  cyclic ABCD :=
by sorry

end quadrilateral_cyclic_l126_126288


namespace no_guaranteed_win_l126_126265

-- Given conditions
def initial_number := 7
def is_perfect_square (n : ℕ) : Prop := ∃ (m : ℕ), m * m = n

-- Define the game mechanics somehow
def allowed_next_numbers (current : ℕ) : list ℕ :=
  -- This function should generate the list of allowed numbers
  -- after appending a digit to current; assuming we have
  -- a function that does this (details omitted for brevity)
  sorry

-- Main statement
theorem no_guaranteed_win :
  ∀ (player : ℕ → list ℕ → ℕ) (turns : ℕ) (numbers : list ℕ), 
    numbers.head = initial_number ∧
    (∀ i < turns, numbers[i+1] ∈ allowed_next_numbers numbers[i]) →
    ∀ (i < turns + 1), ¬ is_perfect_square numbers[i] :=
by sorry

end no_guaranteed_win_l126_126265


namespace least_n_such_that_A_0A_n_geq_100_l126_126588

-- Definition of the conditions
def A_0 := (0, 0 : ℝ × ℝ)
def A (n : ℕ) : ℝ × ℝ
def B (n : ℕ) := (n, n^2 : ℝ × ℝ)
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Proof problem statement
theorem least_n_such_that_A_0A_n_geq_100 :
  ∃ (n : ℕ), n ≥ 10 ∧ dist A_0 (A n) ≥ 100 := sorry

end least_n_such_that_A_0A_n_geq_100_l126_126588


namespace vertical_asymptote_l126_126396

theorem vertical_asymptote (x : ℚ) : (7 * x + 4 = 0) → (x = -4 / 7) :=
by
  intro h
  sorry

end vertical_asymptote_l126_126396


namespace complement_of_A_in_U_l126_126500

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set A
def A : Set ℤ := {x | x ∈ Set.univ ∧ x^2 + x - 2 < 0}

-- State the theorem about the complement of A in U
theorem complement_of_A_in_U :
  (U \ A) = {-2, 1, 2} :=
sorry

end complement_of_A_in_U_l126_126500


namespace shift_graph_to_obtain_g_from_f_l126_126116

def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)
def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_graph_to_obtain_g_from_f :
  ∃ d : ℝ, ∀ x, g (x) = f (x + d) ∧ d = Real.pi / 6 :=
by
  sorry

end shift_graph_to_obtain_g_from_f_l126_126116


namespace range_of_b_l126_126471

theorem range_of_b (a b x : ℝ) (ha : 0 < a ∧ a ≤ 5 / 4) (hb : 0 < b) :
  (∀ x, |x - a| < b → |x - a^2| < 1 / 2) ↔ 0 < b ∧ b ≤ 3 / 16 :=
by
  sorry

end range_of_b_l126_126471


namespace least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l126_126678

theorem least_addition_for_divisibility (n : ℕ) : (1100 + n) % 53 = 0 ↔ n = 9 := by
  sorry

theorem least_subtraction_for_divisibility (n : ℕ) : (1100 - n) % 71 = 0 ↔ n = 0 := by
  sorry

theorem least_addition_for_common_divisibility (X : ℕ) : (1100 + X) % (Nat.lcm 19 43) = 0 ∧ X = 534 := by
  sorry

end least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l126_126678


namespace min_value_f_g_ge_half_l126_126119

-- Define the functions and the conditions
def f (x t : ℝ) : ℝ := exp (2 * x) - 2 * t * x
def g (x : ℝ) : ℝ := -x^2 + 2 * exp x - 3/2

-- Statement for Part 1
theorem min_value_f (x t : ℝ) (h_ge_0 : 0 ≤ x) :
  (t ≤ 1 → f x t ≥ 1) ∧ (t > 1 → f x t ≥ t - t * log t) := 
sorry

-- Statement for Part 2
theorem g_ge_half (x : ℝ) (h_ge_0 : 0 ≤ x) : g x ≥ 1/2 :=
sorry

end min_value_f_g_ge_half_l126_126119


namespace div_by_6_l126_126236

theorem div_by_6 (m : ℕ) : 6 ∣ (m^3 + 11 * m) :=
sorry

end div_by_6_l126_126236


namespace tetrahedron_volume_gt_one_sixth_cube_min_height_l126_126955

variable {A B C D : Type}

-- Definitions for vertices and conditions
variables (A B C D : Type)
variable (h_min : ℝ) -- h_min denotes the smallest height
variable (V : ℝ) -- Volume of the tetrahedron
variable (A_base : ℝ) -- Area of the base triangle

-- Definition of volume
def volume (A_base : ℝ) (h : ℝ) : ℝ := (1/6) * A_base * h

-- Condition that volume with h_min will bound the realistic volume
axiom volume_with_h_min : ∀ (A_base h : ℝ), h ≥ h_min → volume A_base h > (1/6) * h_min^3

theorem tetrahedron_volume_gt_one_sixth_cube_min_height (A_base : ℝ) (h_min : ℝ) :
  (∃ h, h ≥ h_min ∧ volume A_base h > (1/6) * h_min^3) → V > (1/6) * h_min^3 :=
by {
  intro h_existence,
  cases h_existence with h h_properties,
  cases h_properties with h_ge_min volume_gt_cube_min_height,
  exact volume_gt_cube_min_height,
}

end tetrahedron_volume_gt_one_sixth_cube_min_height_l126_126955


namespace cannot_determine_right_triangle_l126_126187

namespace TriangleRightAngle

variables {α β γ a b c : ℝ}

def angle_sum_is_C := α + β = γ
def angle_ratios := α / β = 1 / 2 ∧ β / γ = 2 / 3
def sides_square_relation := a^2 = c^2 - b^2
def side_ratios := a / b = 3 / 4 ∧ b / c = 4 / 6

theorem cannot_determine_right_triangle :
  ¬ (angle_sum_is_C ∨ angle_ratios ∨ sides_square_relation) ∧ side_ratios :=
sorry

end TriangleRightAngle

end cannot_determine_right_triangle_l126_126187


namespace y_plus_z_l126_126537

noncomputable def u1 : ℝ × ℝ × ℝ := (-3, y, 2)
noncomputable def u2 : ℝ × ℝ × ℝ := (6, -2, z)
axiom parallel_planes (y z : ℝ) : ∃ λ : ℝ, u1 = λ • u2

theorem y_plus_z (y z : ℝ) (h : ∃ λ : ℝ, u1 = λ • u2) : y + z = -3 :=
by
  -- Proof goes here
  sorry

end y_plus_z_l126_126537


namespace number_of_ways_to_form_committee_with_president_l126_126345

open Nat

def number_of_ways_to_choose_members (total_members : ℕ) (committee_size : ℕ) (president_required : Bool) : ℕ :=
  if president_required then choose (total_members - 1) (committee_size - 1) else choose total_members committee_size

theorem number_of_ways_to_form_committee_with_president :
  number_of_ways_to_choose_members 30 5 true = 23741 :=
by
  -- Given that total_members = 30, committee_size = 5, and president_required = true,
  -- we need to show that the number of ways to choose the remaining members is 23741.
  sorry

end number_of_ways_to_form_committee_with_president_l126_126345


namespace E_midpoint_of_AI_l126_126096

-- Define the points A, B, C, D, E, F, G, H, I
variables {A B C D E F G H I : EuclideanSpace ℝ (Fin 2)}

-- Definitions for equilateral triangles with positive orientation
def equilateral_positive (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R P ∧
  (⊿PQR) > 0 -- placeholder for positive orientation condition.

axiom ABD_equilateral : equilateral_positive A B D
axiom BAE_equilateral : equilateral_positive B A E
axiom CAF_equilateral : equilateral_positive C A F
axiom DFG_equilateral : equilateral_positive D F G
axiom ECH_equilateral : equilateral_positive E C H
axiom GHI_equilateral : equilateral_positive G H I

-- The main theorem statement
theorem E_midpoint_of_AI : 
  ∃ M : EuclideanSpace ℝ (Fin 2), (M = E) ∧ (dist A E = dist E I) ∧ (affine_space.between ℝ A E I) :=
sorry

end E_midpoint_of_AI_l126_126096


namespace correct_option_l126_126307

theorem correct_option (a : ℝ) : 8 * a^2 - 5 * a^2 = 3 * a^2 := by
  calc
  8 * a^2 - 5 * a^2 = (8 - 5) * a^2 : by sorry -- Using algebraic simplification
                  ... = 3 * a^2 : by sorry

end correct_option_l126_126307


namespace arithmetic_mean_of_fractions_l126_126225

theorem arithmetic_mean_of_fractions :
  let a := (9 : ℝ) / 12
  let b := (5 : ℝ) / 6
  let c := (11 : ℝ) / 12
  (a + c) / 2 = b := 
by
  sorry

end arithmetic_mean_of_fractions_l126_126225


namespace four_digit_integers_with_3_or_7_only_l126_126510

theorem four_digit_integers_with_3_or_7_only : 
  ∃ n : ℕ, n = 16 ∧ ∀ x : ℕ, x ∈ {x | x >= 1000 ∧ x < 10000 ∧ x.digits 10 ⊆ {3, 7}} ↔ x ∈ set.range (λ (i : ℕ), 1000 + 1000 * (i / 8) + 100 * ((i / 4) % 2) + 10 * ((i / 2) % 2) + (i % 2)) ∧ i < 16 :=
sorry

end four_digit_integers_with_3_or_7_only_l126_126510


namespace number_of_students_is_800_l126_126169

noncomputable def number_of_students (T : ℝ) :=
  let juniors := 0.23 * T
  let sophomores := 0.25 * T
  let seniors := 160
  let freshmen := sophomores + 56
  freshmen + sophomores + juniors + seniors = T

theorem number_of_students_is_800 : ∃ T : ℝ, number_of_students T ∧ T = 800 :=
by
  use 800
  unfold number_of_students
  have : 0.25 * 800 + 56 + 0.25 * 800 + 0.23 * 800 + 160 = 800
  { norm_num }
  rw this
  exact ⟨rfl⟩

end number_of_students_is_800_l126_126169


namespace length_of_marquita_garden_l126_126220

variable (length_marquita_garden : ℕ)

def total_area_mancino_gardens : ℕ := 3 * (16 * 5)
def total_gardens_area : ℕ := 304
def total_area_marquita_gardens : ℕ := total_gardens_area - total_area_mancino_gardens
def area_one_marquita_garden : ℕ := total_area_marquita_gardens / 2

theorem length_of_marquita_garden :
  (4 * length_marquita_garden = area_one_marquita_garden) →
  length_marquita_garden = 8 := by
  sorry

end length_of_marquita_garden_l126_126220


namespace no_natural_number_with_odd_even_divisors_l126_126189

theorem no_natural_number_with_odd_even_divisors :
  ∀ N : ℕ, ¬ (∃ odd_count even_count : ℕ, (odd_count % 2 = 0) ∧ (even_count % 2 = 1) ∧
              (odd_count = (N.divisors.filter (λ n, n % 2 = 1)).length) ∧
              (even_count = (N.divisors.filter (λ n, n % 2 = 0)).length)) :=
by
  intros N
  sorry

end no_natural_number_with_odd_even_divisors_l126_126189


namespace probability_two_diff_colors_l126_126162

-- Defining the conditions based on part (a)
constant blue_chips : ℕ := 6
constant red_chips : ℕ := 5
constant yellow_chips : ℕ := 3
constant green_chips : ℕ := 2
constant total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

-- Defining the probability statement
noncomputable def prob_diff_colors : ℚ :=
  (6 / 16) * (10 / 16) + (5 / 16) * (11 / 16) + (3 / 16) * (13 / 16) + (2 / 16) * (14 / 16)

-- The goal is to prove that this probability is equal to 91/128
theorem probability_two_diff_colors : prob_diff_colors = 91 / 128 :=
by
  sorry -- Proof to be provided

end probability_two_diff_colors_l126_126162


namespace grid_coloring_even_l126_126793

theorem grid_coloring_even (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n)
  (H : ∀ (A : ℕ × ℕ),
    let cells := {(x, y) | 1 ≤ x ∧ x ≤ m ∧ 1 ≤ y ∧ y ≤ n ∧ (x, y) = A ∨ (x, y) ≠ A ∧ ((A.1 - x) = 1 ∨ (x - A.1) = 1 ∨ (A.2 - y) = 1 ∨ (y - A.2) = 1 ∨ ((A.1 - x) = 1 ∧ (A.2 - y) = 1) ∨ ((x - A.1) = 1 ∧ (y - A.2) = 1))}
    in even (card {c : ℕ × ℕ | c ∈ cells ∧ c = A ∨ c ≠ A ∧ f(c) = f(A)})
  )
  : even (m * n) :=
sorry

end grid_coloring_even_l126_126793


namespace geometric_sequence_theorem_l126_126909

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = a n * r

def holds_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 10 = -2

theorem geometric_sequence_theorem (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_cond : holds_condition a) : a 4 * a 7 = -2 :=
by
  sorry

end geometric_sequence_theorem_l126_126909


namespace ball_in_box_l126_126137

theorem ball_in_box : 
  let num_ways_to_place_balls := 3^5 in 
  num_ways_to_place_balls = 243 :=
by
  sorry

end ball_in_box_l126_126137


namespace final_probability_l126_126371

-- Define the structure of the problem
structure GameRound :=
  (green_ball : ℕ)
  (red_ball : ℕ)
  (blue_ball : ℕ)
  (white_ball : ℕ)

structure GameState :=
  (coins : ℕ)
  (players : ℕ)

-- Define the game rules and initial conditions
noncomputable def initial_coins := 5
noncomputable def rounds := 5

-- Probability-related functions and game logic
noncomputable def favorable_outcome_count : ℕ := 6
noncomputable def total_outcomes_per_round : ℕ := 120
noncomputable def probability_per_round : ℚ := favorable_outcome_count / total_outcomes_per_round

theorem final_probability :
  probability_per_round ^ rounds = 1 / 3200000 :=
by
  sorry

end final_probability_l126_126371


namespace find_number_l126_126343

theorem find_number (num : ℝ) (x : ℝ) (h1 : x = 0.08999999999999998) (h2 : num / x = 0.1) : num = 0.008999999999999999 :=
by 
  sorry

end find_number_l126_126343


namespace solution_set_condition_l126_126155

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - 1| > a) ↔ a < 1 :=
by
  sorry

end solution_set_condition_l126_126155


namespace fraction_value_l126_126923

theorem fraction_value (a b c d : ℝ) (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - d) * (b - c) / ((a - b) * (c - d)) = -4 / 3 :=
sorry

end fraction_value_l126_126923


namespace intersects_l126_126846

-- Define the given conditions
def radius : ℝ := 5
def distance_to_line : ℝ := 3 * Real.sqrt 2

-- Define the relationship to prove
def line_intersects_circle : Prop :=
  radius > distance_to_line

-- Proof Statement
theorem intersects (r d : ℝ) (h_r : r = radius) (h_d : d = distance_to_line) : r > d :=
by {
  rw [h_r, h_d],
  exact Real.lt_of_lt_of_le (by norm_num) (by norm_num),
}

end intersects_l126_126846


namespace coefficient_x2_in_expansion_l126_126058

theorem coefficient_x2_in_expansion :
  coefficient ((X ^ 2 - C 2 * X - C 3) ^ 3) 2 = -9 :=
sorry

end coefficient_x2_in_expansion_l126_126058


namespace second_player_wins_l126_126038

-- Define the initial grid size and positions
def grid_size : ℕ := 20
def initial_positions : ℕ × ℕ := (1, 20)

-- Define the move constraints
def move_constraints (a b : ℕ) : Prop :=
  (a < b) ∧ (b - a - 1) % 3 = 0

-- Initial condition check for the winning position
theorem second_player_wins (a b : ℕ) (h : a = 1 ∧ b = 20) : (b - a - 1) % 3 = 0 :=
by
  rcases h with ⟨h1, h2⟩
  rw [h1, h2]
  norm_num
  sorry

end second_player_wins_l126_126038


namespace similar_triangles_l126_126610

open EuclideanGeometry

variables {A B C : Point} {A1 B1 C1 A2 B2 C2 A3 B3 C3 : Point}
variable {circ_A1B1C1 circ_B1C1A1 circ_C1A1B1 circ_ABC : Circle}

-- Conditions
variable (h1 : On A1 (Segment B C))
variable (h2 : On B1 (Segment C A))
variable (h3 : On C1 (Segment A B))
variable (h4 : On A2 circ_ABC ∧ On A2 circ_A1B1C1 ∧ A2 ≠ A)
variable (h5 : On B2 circ_ABC ∧ On B2 circ_B1C1A1 ∧ B2 ≠ B)
variable (h6 : On C2 circ_ABC ∧ On C2 circ_C1A1B1 ∧ C2 ≠ C)
variable (h7 : Symmetric A1 A3 (Midpoint B C))
variable (h8 : Symmetric B1 B3 (Midpoint C A))
variable (h9 : Symmetric C1 C3 (Midpoint A B))

-- Proof that triangles are similar
theorem similar_triangles :
  Similar (Triangle.mk A2 B2 C2) (Triangle.mk A3 B3 C3) :=
sorry

end similar_triangles_l126_126610


namespace find_remainder_l126_126685

-- Definitions
variable (x y : ℕ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : (x : ℝ) / y = 96.15)
variable (h4 : approximately_equal (y : ℝ) 60)

-- Target statement
theorem find_remainder : x % y = 9 :=
sorry

end find_remainder_l126_126685


namespace outlet_pipe_takes_8_hours_longer_l126_126034

/-
Given:
1. The capacity of the tank is 21600 litres.
2. The outlet pipe empties the tank in 10 hours.
3. The inlet pipe fills the tank at a rate of 16 litres per minute.
Prove:
The outlet pipe takes 8 hours longer to empty the tank when the inlet pipe is open.
-/

noncomputable def hours_longer_to_empty_tank (capacity : ℕ) (outlet_time : ℕ) (inlet_rate : ℕ) : ℕ :=
  let outlet_rate := capacity / outlet_time in
  let inlet_rate_per_hour := inlet_rate * 60 in
  let effective_rate := outlet_rate - inlet_rate_per_hour in
  let time_with_inlet := capacity / effective_rate in
  time_with_inlet - outlet_time

theorem outlet_pipe_takes_8_hours_longer (capacity : ℕ) (outlet_time : ℕ) (inlet_rate : ℕ) :
  capacity = 21600 →
  outlet_time = 10 →
  inlet_rate = 16 →
  hours_longer_to_empty_tank capacity outlet_time inlet_rate = 8 :=
by
  intros _ _ _
  sorry

end outlet_pipe_takes_8_hours_longer_l126_126034


namespace cone_cube_volume_ratio_l126_126360

noncomputable def volumeRatio (s : ℝ) : ℝ :=
  let r := s / 2
  let h := s
  let volume_cone := (1 / 3) * Real.pi * r^2 * h
  let volume_cube := s^3
  volume_cone / volume_cube

theorem cone_cube_volume_ratio (s : ℝ) (h_cube_eq_s : s > 0) :
  volumeRatio s = Real.pi / 12 :=
by
  sorry

end cone_cube_volume_ratio_l126_126360


namespace f1_negative_inversion_f2_not_negative_inversion_f3_negative_inversion_l126_126349

-- Define the "negative inversion" property
def negative_inversion (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = -f x

-- Define the three functions
def f1 (x : ℝ) : ℝ := x - (1 / x)
def f2 (x : ℝ) : ℝ := x + (1 / x)
noncomputable def f3 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x else if x = 1 then 0 else - (1 / x)

-- Prove which functions satisfy the "negative inversion" property
theorem f1_negative_inversion : negative_inversion f1 :=
by {
  sorry
}

theorem f2_not_negative_inversion : ¬negative_inversion f2 :=
by {
  -- Counter-example already provided in the English solution
  sorry
}

theorem f3_negative_inversion : negative_inversion f3 :=
by {
  sorry
}

end f1_negative_inversion_f2_not_negative_inversion_f3_negative_inversion_l126_126349


namespace sum_of_distances_to_focus_is_ten_l126_126107

theorem sum_of_distances_to_focus_is_ten (P : ℝ × ℝ) (A B F : ℝ × ℝ)
  (hP : P = (2, 1))
  (hA : A.1^2 = 12 * A.2)
  (hB : B.1^2 = 12 * B.2)
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hFocus : F = (3, 0)) :
  |A.1 - F.1| + |B.1 - F.1| = 10 :=
by
  sorry

end sum_of_distances_to_focus_is_ten_l126_126107


namespace min_value_f_l126_126994

noncomputable def f (x : ℝ) : ℝ := (5 - 4 * x + x^2) / (2 - x)

theorem min_value_f : ∃ a ∈ Iio (2 : ℝ), ∀ x ∈ Iio (2 : ℝ), f x ≥ a ∧ f x = a → a = 2 := 
sorry

end min_value_f_l126_126994


namespace volume_of_triangle_pyramid_l126_126026

-- Define the problem stating the sides of the triangle
def sides (a b c : ℝ) : Prop := 
  a = 5 ∧ b = 6 ∧ c = 7

-- Define the statement of area and volume of the pyramid
theorem volume_of_triangle_pyramid (a b c : ℝ) (V : ℝ) :
  sides a b c → V = 2 * real.sqrt 95 :=
begin
  intros h_sides,
  apply sorry,
end

end volume_of_triangle_pyramid_l126_126026


namespace total_money_remains_constant_l126_126814

def initial_amounts (a j b : ℕ) := (24 + a + j + b)
def total_money_after_redistribution (a j b : ℕ) := initial_amounts a j b

theorem total_money_remains_constant (a j b : ℕ) :
  let initial := initial_amounts a j b in
  t = 24 → (after_steps a j b 24) = (192) := sorry

end total_money_remains_constant_l126_126814


namespace cat_and_mouse_capture_two_cats_l126_126226

theorem cat_and_mouse_capture_two_cats (
    mouse_pos : ℕ × ℕ,
    cat1_pos : ℕ × ℕ := (1, 1),
    cat2_pos : ℕ × ℕ := (8, 8)
) (h_mouse_non_edge : mouse_pos.1 ≠ 1 ∧ mouse_pos.2 ≠ 1 ∧ mouse_pos.1 ≠ 8 ∧ mouse_pos.2 ≠ 8)
    (h_cat_moves : ∀ (m : ℕ × ℕ), m ∈ {(1,0), (-1,0), (0, 1), (0, -1)} → ∃ (c1 c2 : ℕ × ℕ),
    (c1 = (cat1_pos.1 + m.1, cat1_pos.2 + m.2) ∨ c2 = (cat2_pos.1 + m.1, cat2_pos.2 + m.2))) : Prop :=
begin
    sorry
end

end cat_and_mouse_capture_two_cats_l126_126226


namespace square_side_length_l126_126308

theorem square_side_length (area_circle perimeter_square : ℝ) (h1 : area_circle = 100) (h2 : perimeter_square = area_circle) :
  side_length_square perimeter_square = 25 :=
by
  let s := 25 -- The length of one side of the square is 25
  sorry

def side_length_square (perimeter_square : ℝ) : ℝ :=
  perimeter_square / 4

end square_side_length_l126_126308


namespace sum_of_real_values_of_x_satisfying_series_eq_l126_126406

noncomputable def series_expression (x : ℝ) : ℝ := 1 + (∑' n, (-x)^(n + 1))

theorem sum_of_real_values_of_x_satisfying_series_eq :
  let x₁ := (-1 + Real.sqrt 5) / 2 in
  let x₂ := (-1 - Real.sqrt 5) / 2 in
  |x : ℝ| < 1 → x = series_expression x → (x = x₁ ∨ x = x₂) ∧ x₁ = (2 + sqrt 5) / 2 :=
by
  sorry

end sum_of_real_values_of_x_satisfying_series_eq_l126_126406


namespace prism_volume_l126_126977

theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 5 :=
by
  sorry

end prism_volume_l126_126977


namespace ball_draw_probabilities_l126_126892

-- Definitions
namespace BallDraw

def balls := {w := 1, r := 2, y := 1}

-- Probability of drawing 2 balls
def total_outcomes := Combinatorics.choose 4 2

-- Probability that both balls drawn are red
def prob_both_red := 2 / total_outcomes

-- Probability that at least one ball is red
def prob_at_least_one_red := 1 - (Combinatorics.choose 2 2) / total_outcomes

end BallDraw

-- Problem statement
theorem ball_draw_probabilities :
  BallDraw.prob_both_red = 1 / 6 ∧ BallDraw.prob_at_least_one_red = 5 /6 := 
by sorry

end ball_draw_probabilities_l126_126892


namespace triangle_is_isosceles_or_right_l126_126890

theorem triangle_is_isosceles_or_right
  {A B C : Type*}
  [metric_space A] [metric_space B] [metric_space C]
  (AB AC BC : ℝ)
  (h : BC * real.cos A = AC * real.cos B) :
  (A = B) ∨ (A + B = real.pi / 2) :=
by
  sorry

end triangle_is_isosceles_or_right_l126_126890


namespace total_amount_correct_l126_126217

-- Define the prices of jeans and tees
def price_jean : ℕ := 11
def price_tee : ℕ := 8

-- Define the quantities sold
def quantity_jeans_sold : ℕ := 4
def quantity_tees_sold : ℕ := 7

-- Calculate the total amount earned
def total_amount : ℕ := (price_jean * quantity_jeans_sold) + (price_tee * quantity_tees_sold)

-- Now, we state and prove the theorem
theorem total_amount_correct : total_amount = 100 :=
by
  -- Here we assert the correctness of the calculation
  sorry

end total_amount_correct_l126_126217


namespace symmetric_point_on_x_axis_l126_126267

theorem symmetric_point_on_x_axis :
  ∀ (a b : ℝ), (∀ (x y : ℝ), (x, y) = (-3, 1) → (a, b) = (x, -y)) → (a = -3 ∧ b = -1) :=
by {
  intros a b H,
  specialize H -3 1 rfl,
  rw H,
  split;
  refl,
  sorry
}

end symmetric_point_on_x_axis_l126_126267


namespace constant_term_expansion_l126_126640

theorem constant_term_expansion :
  (∃ c : ℤ, ∀ x : ℝ, (2 * x - 1 / x) ^ 4 = c * x^0) ∧ c = 24 :=
by
  sorry

end constant_term_expansion_l126_126640


namespace glasses_displayed_is_correct_l126_126055

-- Definitions from the problem conditions
def tall_cupboard_capacity : Nat := 20
def wide_cupboard_capacity : Nat := 2 * tall_cupboard_capacity
def per_shelf_narrow_cupboard : Nat := 15 / 3
def usable_narrow_cupboard_capacity : Nat := 2 * per_shelf_narrow_cupboard

-- Theorem to prove that the total number of glasses displayed is 70
theorem glasses_displayed_is_correct :
  (tall_cupboard_capacity + wide_cupboard_capacity + usable_narrow_cupboard_capacity) = 70 :=
by
  sorry

end glasses_displayed_is_correct_l126_126055


namespace cost_of_3600_pens_l126_126723

theorem cost_of_3600_pens
  (pack_size : ℕ)
  (pack_cost : ℝ)
  (n_pens : ℕ)
  (pen_cost : ℝ)
  (total_cost : ℝ)
  (h1: pack_size = 150)
  (h2: pack_cost = 45)
  (h3: n_pens = 3600)
  (h4: pen_cost = pack_cost / pack_size)
  (h5: total_cost = n_pens * pen_cost) :
  total_cost = 1080 :=
sorry

end cost_of_3600_pens_l126_126723


namespace integral_cos2x_over_cosx_plus_sinx_l126_126778

theorem integral_cos2x_over_cosx_plus_sinx :
  ∫ x in 0..(Real.pi / 4), (Real.cos (2 * x) / (Real.cos x + Real.sin x)) = Real.sqrt 2 - 1 :=
by
  sorry

end integral_cos2x_over_cosx_plus_sinx_l126_126778


namespace find_AD_l126_126185

theorem find_AD
  (A B C D : Type)
  (BD BC CD AD : ℝ)
  (hBD : BD = 21)
  (hBC : BC = 30)
  (hCD : CD = 15)
  (hAngleBisect : true) -- Encode that D bisects the angle at C internally
  : AD = 35 := by
  sorry

end find_AD_l126_126185


namespace device_prices_within_budget_l126_126576

-- Given conditions
def x : ℝ := 12 -- Price of each type A device in thousands of dollars
def y : ℝ := 10 -- Price of each type B device in thousands of dollars
def budget : ℝ := 110 -- The budget in thousands of dollars

-- Conditions as given equations and inequalities
def condition1 : Prop := 3 * x - 2 * y = 16
def condition2 : Prop := 3 * y - 2 * x = 6
def budget_condition (a : ℕ) : Prop := 12 * a + 10 * (10 - a) ≤ budget

-- Theorem to prove
theorem device_prices_within_budget :
  condition1 ∧ condition2 ∧
  (∀ a : ℕ, a ≤ 5 → budget_condition a) :=
by sorry

end device_prices_within_budget_l126_126576


namespace six_parallelepipeds_visibility_l126_126045

structure Point (space : Type) := { coords : space }

structure Parallelepiped (space : Type) :=
  (vertices : set (Point space)) -- Set of vertices
  (opaque : Prop := True)

theorem six_parallelepipeds_visibility :
  ∃ (p : Point ℝ^3),
    ∀ (parallelepipeds : Fin 6 → Parallelepiped ℝ^3),
      (∀ i j, i ≠ j → Disjoint (parallelepipeds i).vertices (parallelepipeds j).vertices) → 
      (∀ i, p ∉ (parallelepipeds i).vertices) →
      ∀ i, ¬ (∃ vertex, vertex ∈ (parallelepipeds i).vertices ∧ 
                          visible_from p vertex) :=
begin
  sorry -- Proof goes here
end

end six_parallelepipeds_visibility_l126_126045


namespace roger_price_correct_l126_126740

-- Define the conditions given in the problem
def art_circle_radius : ℝ := 2
def art_cookies_count : ℕ := 15
def art_cookie_price_in_cents : ℕ := 50
def roger_cookies_count : ℕ := 20

-- Define the main function to compute the price per Roger's cookie to match Art's earnings
def roger_cookie_price_in_cents : ℝ :=
  art_cookies_count * art_cookie_price_in_cents / roger_cookies_count

-- Theorem stating the desired outcome
theorem roger_price_correct : roger_cookie_price_in_cents = 37.5 := by
  sorry

end roger_price_correct_l126_126740


namespace red_squares_multiple_of_three_max_red_squares_exact_red_squares_l126_126548

noncomputable theory

variables (n : ℕ)

def unicorn_grid (n : ℕ) := fin (3 * n) → fin (3 * n) → Prop

-- Proof Problem 1: The number of red squares is a multiple of 3
theorem red_squares_multiple_of_three 
  (unicorn_moves : ∀ (i j : fin (3 * n)), unicorn_grid n)
  (condition : ∀ (i j : fin (3 * n)), 
    unicorn_moves i j ∧ unicorn_moves j i → ∃ k, unicorn_moves k k) :
  ∃ m, m % 3 = 0 :=
sorry

-- Proof Problem 2: The maximum number of red squares is \(9n^2 - 3\)
theorem max_red_squares
  (unicorn_moves : ∀ (i j : fin (3 * n)), unicorn_grid n)
  (condition : ∀ (i j : fin (3 * n)), 
    unicorn_moves i j ∧ unicorn_moves j i → ∃ k, unicorn_moves k k) :
  ∃ l, l ≤ 9 * n^2 - 3 :=
sorry

-- Proof Problem 3: There exists a path with exactly \(9n^2 - 3\) red squares
theorem exact_red_squares
  (unicorn_moves : ∀ (i j : fin (3 * n)), unicorn_grid n)
  (condition : ∀ (i j : fin (3 * n)), 
    unicorn_moves i j ∧ unicorn_moves j i → ∃ k, unicorn_moves k k) :
  ∃ p, p = 9 * n^2 - 3 :=
sorry

end red_squares_multiple_of_three_max_red_squares_exact_red_squares_l126_126548


namespace number_of_people_speak_latin_l126_126167

theorem number_of_people_speak_latin (total_people F neither F_and_L : ℕ) 
  (h1 : total_people = 25) 
  (h2 : F = 15) 
  (h3 : neither = 6) 
  (h4 : F_and_L = 9) : 
  ∃ L, L = 13 := 
by
  have h5 : 19 = total_people - neither := by sorry
  have h6 : L + F - F_and_L = 19 := by sorry
  use 13
  sorry

end number_of_people_speak_latin_l126_126167


namespace ratio_of_areas_l126_126374

theorem ratio_of_areas (side_large : ℝ) (side_small : ℝ) 
  (h_large : side_large = 10) (h_small : side_small = 3) :
  let area_large := (sqrt 3 / 4) * side_large^2,
      area_small := (sqrt 3 / 4) * side_small^2,
      area_central := area_large - 3 * area_small
  in 
  (area_small / area_central) = (9 / 73) := 
by
  sorry

end ratio_of_areas_l126_126374


namespace sum_of_angles_is_360_l126_126908

-- Let's define the specific angles within our geometric figure
variables (A B C D F G : ℝ)

-- Define a condition stating that these angles form a quadrilateral inside a geometric figure, such that their sum is valid
def angles_form_quadrilateral (A B C D F G : ℝ) : Prop :=
  (A + B + C + D + F + G = 360)

-- Finally, we declare the theorem we want to prove
theorem sum_of_angles_is_360 (A B C D F G : ℝ) (h : angles_form_quadrilateral A B C D F G) : A + B + C + D + F + G = 360 :=
  h


end sum_of_angles_is_360_l126_126908


namespace betty_wallet_l126_126380

theorem betty_wallet :
  let wallet_cost := 125.75
  let initial_amount := wallet_cost / 2
  let parents_contribution := 45.25
  let grandparents_contribution := 2 * parents_contribution
  let brothers_contribution := 3/4 * grandparents_contribution
  let aunts_contribution := 1/2 * brothers_contribution
  let total_amount := initial_amount + parents_contribution + grandparents_contribution + brothers_contribution + aunts_contribution
  total_amount - wallet_cost = 174.6875 :=
by
  sorry

end betty_wallet_l126_126380


namespace edge_coloring_number_tournament_l126_126024

/-- Definition of a tournament graph -/
structure TournamentGraph (V : Type) :=
  (E : V → V → Prop)
  (total_relation : ∀ u v, u ≠ v → (E u v ∨ E v u) ∧ ¬ (E u v ∧ E v u))

/-- A proper directed edge coloring -/
def proper_directed_edge_coloring
  {V : Type} (G : TournamentGraph V) (coloring : (V → V → ℕ)) : Prop :=
  ∀ (u v w : V), (G.E u v ∧ G.E v w) → coloring u v ≠ coloring v w

/-- The edge coloring number of a tournament graph is the minimum number of colors needed to properly color the directed edges -/
def edge_coloring_number {V : Type} (G : TournamentGraph V) : ℕ :=
  Inf { k | ∃ coloring : (V → V → ℕ), (proper_directed_edge_coloring G coloring ∧ ∀ (u v : V), u ≠ v → coloring u v < k) }

/-- Main problem statement: For each positive integer n and for all tournament graphs with n vertices, the minimum edge coloring number equals ⌈ log₂ n ⌉ -/
theorem edge_coloring_number_tournament (n : ℕ) (hn : 0 < n) :
  ∀ (G : TournamentGraph (Fin n)), edge_coloring_number G = Nat.ceil (Real.log2 n) :=
sorry

end edge_coloring_number_tournament_l126_126024


namespace marbles_remainder_l126_126817

theorem marbles_remainder 
  (g r p : ℕ) 
  (hg : g % 8 = 5) 
  (hr : r % 7 = 2) 
  (hp : p % 7 = 4) : 
  (r + p + g) % 7 = 4 := 
sorry

end marbles_remainder_l126_126817


namespace task_completion_time_l126_126880

theorem task_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ t : ℝ, t = (a * b) / (a + b) := 
sorry

end task_completion_time_l126_126880


namespace area_of_enclosed_region_is_zero_l126_126768

theorem area_of_enclosed_region_is_zero :
  (∃ (x y : ℝ), x^2 + y^2 = |x| - |y|) → (0 = 0) :=
sorry

end area_of_enclosed_region_is_zero_l126_126768


namespace angle_PCQ_measure_l126_126205

open EuclideanGeometry

-- Definitions of points and triangles based on given conditions
variables {A B C P Q : Point}
variables (x : Real)

-- Isosceles triangle ABC with AB < BC and specific properties of points P and Q
axiom h1 : IsoscelesTriangle A B C
axiom h2 : A B < B C
axiom h3 : OnLine P B C ∧ P ≠ B ∧ P ≠ C ∧ Distance B P = Distance B A
axiom h4 : OnLine Q A P ∧ Q ≠ A ∧ Distance C Q = Distance C A

-- Declare the conjecture (to be proved)
theorem angle_PCQ_measure
  (h : ∠ Q P C = x) :
  ∠ P C Q = 180 - 4 * x :=
sorry  -- Proof omitted

end angle_PCQ_measure_l126_126205


namespace ratio_of_part_to_whole_l126_126607

theorem ratio_of_part_to_whole (N : ℝ) (P : ℝ) (h1 : (1/4) * (2/5) * N = 17) (h2 : 0.40 * N = 204) :
  P = (2/5) * N → P / N = 2 / 5 :=
by
  intro h3
  sorry

end ratio_of_part_to_whole_l126_126607


namespace scheduling_methods_l126_126549

def total_schedules (subjects : List String) : Nat :=
  let schedules := subjects.permutations.filter (λ sched, 
      sched[1] = "Physical Education" ∨ sched[2] = "Physical Education")                                                             -- Physical Education not in first or last period
  schedules.length

theorem scheduling_methods :
  total_schedules ["Chinese", "Mathematics", "English", "Physical Education"] = 12 :=
by
  sorry

end scheduling_methods_l126_126549


namespace array_sums_diff_iff_even_l126_126813

/-- 
  This statement expresses that an \( n \times n \) array with 
  entries in \{-1, 0, 1\} such that all \(2n\) row and column 
  sums are different exists if and only if \( n \) is even.
--/
theorem array_sums_diff_iff_even (n : ℕ) : 
  (∃ (A : matrix (fin n) (fin n) ℤ), (∀ i j, A i j ∈ ({-1, 0, 1} : set ℤ)) ∧ 
  (∀ i j, ∃ i' j', (∑ x, A i x ≠ ∑ x, A i' x) ∧ 
  (∑ y, A y j ≠ ∑ y, A y j'))) ↔ even n :=
sorry

end array_sums_diff_iff_even_l126_126813


namespace aziz_age_l126_126379

-- Definitions of the conditions
def year_moved : ℕ := 1982
def years_before_birth : ℕ := 3
def current_year : ℕ := 2021

-- Prove the main statement
theorem aziz_age : current_year - (year_moved + years_before_birth) = 36 :=
by
  sorry

end aziz_age_l126_126379


namespace find_eccentricity_find_equation_l126_126834

open Real

-- Conditions for the first question
def is_ellipse (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def are_focus (a b : ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = ( - sqrt (a^2 - b^2), 0) ∧ F2 = (sqrt (a^2 - b^2), 0)

def arithmetic_sequence (a b : ℝ) (A B : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  let dist_AF1 := abs (A.1 - F1.1)
  let dist_BF1 := abs (B.1 - F1.1)
  let dist_AB := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (dist_AF1 + dist_AB + dist_BF1 = 4 * a) ∧
  (dist_AF1 + dist_BF1 = 2 * dist_AB)

-- Proof statement for the eccentricity
theorem find_eccentricity (a b : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : is_ellipse a b)
  (h4 : are_focus a b F1 F2)
  (h5 : arithmetic_sequence a b A B F1) :
  ∃ e : ℝ, e = sqrt 2 / 2 :=
sorry

-- Conditions for the second question
def geometric_property (a b : ℝ) (A B P : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, P = (0, -1) → 
             (x^2 / a^2) + (y^2 / b^2) = 1 → 
             abs ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 
             abs ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Proof statement for the equation of the ellipse
theorem find_equation (a b : ℝ) (A B P : ℝ × ℝ)
  (h1 : a = 3 * sqrt 2) (h2 : b = 3) (h3 : P = (0, -1))
  (h4 : is_ellipse a b) (h5 : geometric_property a b A B P) :
  ∃ E : Prop, E = ((x : ℝ) * 2 / 18 + (y : ℝ) * 2 / 9 = 1) :=
sorry

end find_eccentricity_find_equation_l126_126834


namespace factorization_correct_l126_126790

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l126_126790


namespace triangle_isosceles_x_value_l126_126181

theorem triangle_isosceles_x_value (AB AC : ℝ) (BAC ABC ACB DBC : ℝ) 
  (h_isosceles : AB = AC) 
  (h_BAC : BAC = 40) 
  (h_collinear : ∀ B C D, LinearTriple B C D) 
  (h_DBC : DBC = 2 * x) :
  x = 55 :=
by
  sorry

end triangle_isosceles_x_value_l126_126181


namespace perpendicular_lines_implies_parallel_l126_126466

variables (a b c d : Type)
variables [inner_product_space ℝ a] [inner_product_space ℝ b] [inner_product_space ℝ c] [inner_product_space ℝ d]

-- Defining perpendicular relationships
def is_perpendicular (x y : Type) [inner_product_space ℝ x] [inner_product_space ℝ y] : Prop := sorry

-- Defining parallel relationships
def is_parallel (x y : Type) [inner_product_space ℝ x] [inner_product_space ℝ y] : Prop := sorry

axiom a_perpendicular_b : is_perpendicular a b
axiom b_perpendicular_c : is_perpendicular b c
axiom c_perpendicular_d : is_perpendicular c d
axiom d_perpendicular_a : is_perpendicular d a

-- Proof problem statement
theorem perpendicular_lines_implies_parallel :
  (is_parallel b d) ∨ (is_parallel a c) :=
by sorry

end perpendicular_lines_implies_parallel_l126_126466


namespace M_is_infinite_l126_126605

noncomputable theory

-- Define the set M and conditions
def M : Set ℝ := sorry 
def midpoint_condition (M : Set ℝ) : Prop := ∀ A ∈ M, ∃ B C ∈ M, A = (B + C) / 2

-- State the theorem
theorem M_is_infinite (M : Set ℝ) 
  (h_midpoint : midpoint_condition M) : 
  ¬(Finite M) := 
sorry

end M_is_infinite_l126_126605


namespace find_BE_l126_126569

-- Define the setup of the problem
variables {A B C D E : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (AB AC BC : ℝ) (CD BE : ℝ)
variables (h : ∀ α β : ℝ, ∃ k l : ℝ, α + β = α + l / k ∧ β = k / l)

-- Define the given conditions
def given_conditions (AB AC BC CD BE : ℝ) : Prop :=
  AB = 13 ∧ BC = 15 ∧ CA = 14 ∧ CD = 6 ∧
  ∃ α : ℝ, ∃ β : ℝ, α = ∠BAE ∧ α = ∠CAD

-- Define the goal
theorem find_BE (h: given_conditions AB AC BC CD BE): BE = 2535 / 463 :=
sorry

end find_BE_l126_126569


namespace determinant_of_matrix_l126_126111

theorem determinant_of_matrix :
  let a := 3
  let b := Real.sin (Real.pi / 6)
  let c := 5
  let d := Real.cos (Real.pi / 3)
  b = 1 / 2 →
  d = 1 / 2 →
  a * d - b * c = -1 :=
by
  intros ha hd
  rw [ha, hd]
  have h1 : 3 * (1 / 2) = 3 / 2 := by norm_num
  have h2 : 5 * (1 / 2) = 5 / 2 := by norm_num
  rw [h1, h2]
  norm_num

end determinant_of_matrix_l126_126111


namespace y_intercept_of_linear_function_l126_126995

theorem y_intercept_of_linear_function 
  (k : ℝ)
  (h : (∃ k: ℝ, ∀ x y: ℝ, y = k * (x - 1) ∧ (x, y) = (-1, -2))) : 
  ∃ y : ℝ, (0, y) = (0, -1) :=
by {
  -- Skipping the proof as per the instruction
  sorry
}

end y_intercept_of_linear_function_l126_126995


namespace edward_original_lawns_l126_126414

-- Definitions based on conditions
def dollars_per_lawn : ℕ := 4
def lawns_forgotten : ℕ := 9
def dollars_earned : ℕ := 32

-- The original number of lawns to mow
def original_lawns_to_mow (L : ℕ) : Prop :=
  dollars_per_lawn * (L - lawns_forgotten) = dollars_earned

-- The proof problem statement
theorem edward_original_lawns : ∃ L : ℕ, original_lawns_to_mow L ∧ L = 17 :=
by
  sorry

end edward_original_lawns_l126_126414


namespace total_pots_needed_l126_126381

theorem total_pots_needed
    (p : ℕ) (s : ℕ) (h : ℕ)
    (hp : p = 5)
    (hs : s = 3)
    (hh : h = 4) :
    p * s * h = 60 := by
  sorry

end total_pots_needed_l126_126381


namespace three_numbers_sum_at_least_54_l126_126279

theorem three_numbers_sum_at_least_54 (nums : Fin 10 → ℕ) (h_distinct : Function.Injective nums)
    (h_sum : (∑ i, nums i) > 144) : ∃ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ nums a + nums b + nums c ≥ 54 :=
by
  sorry

end three_numbers_sum_at_least_54_l126_126279


namespace exam_papers_count_l126_126021

theorem exam_papers_count (F x : ℝ) :
  (∀ n : ℕ, n = 5) →    -- condition 1: equivalence of n to proportions count
  (6 * x + 7 * x + 8 * x + 9 * x + 10 * x = 40 * x) →    -- condition 2: sum of proportions
  (40 * x = 0.60 * n * F) →   -- condition 3: student obtained 60% of total marks
  (7 * x > 0.50 * F ∧ 8 * x > 0.50 * F ∧ 9 * x > 0.50 * F ∧ 10 * x > 0.50 * F ∧ 6 * x ≤ 0.50 * F) →  -- condition 4: more than 50% in 4 papers
  ∃ n : ℕ, n = 5 :=    -- prove: number of papers is 5
sorry

end exam_papers_count_l126_126021


namespace probability_of_section_4_l126_126357

def pentagon := Fin 5

theorem probability_of_section_4 : 
  (∃ (sections : pentagon), sections = 4) → 
  (P : ℚ) := 
    P = 1 / 5 :=
by
  sorry

end probability_of_section_4_l126_126357


namespace ratio_of_a_to_b_l126_126838

-- Given conditions
variables {a b x : ℝ}
-- a and b are positive real numbers distinct from 1
variables (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1)
-- Given equation involving logarithms
variables (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2)

-- Prove that the ratio of a to b is a^(sqrt(7/5))
theorem ratio_of_a_to_b (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1) (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2) :
  b = a ^ Real.sqrt (7 / 5) :=
sorry

end ratio_of_a_to_b_l126_126838


namespace proposition_q_iff_proposition_p_iff_proposition_p_or_q_l126_126861

open Real

theorem proposition_q_iff (k : ℝ) : 
  (∀ x y : ℝ, ((x^2 / (4 - k)) + (y^2 / (k - 1)) = 1) ↔ (1 < k ∧ k < 5/2)) := 
sorry

theorem proposition_p_iff (k : ℝ) :
  (∀ x : ℝ, x^2 + k*x + 2*k + 5 ≥ 0) ↔ (-2 ≤ k ∧ k ≤ 10) :=
sorry

theorem proposition_p_or_q (k : ℝ) :
  (∃ k, proposition_p_iff k ∨ proposition_q_iff k) ∧ ¬ (∃ k, proposition_p_iff k ∧ proposition_q_iff k)
  ↔ (k ∈ set.Icc (-2 : ℝ) 1 ∪ set.Icc (5/2) 10) :=
sorry

end proposition_q_iff_proposition_p_iff_proposition_p_or_q_l126_126861


namespace a_1994_is_7_l126_126282

def f (m : ℕ) : ℕ := m % 10

def a (n : ℕ) : ℕ := f (2^(n + 1) - 1)

theorem a_1994_is_7 : a 1994 = 7 :=
by
  sorry

end a_1994_is_7_l126_126282


namespace carter_extra_cakes_l126_126751

def regular_cakes : ℕ := 6 + 5 + 8
def triple_cakes : ℕ := 3 * 6 + 3 * 5 + 3 * 8
def extra_cakes : ℕ := triple_cakes - regular_cakes

theorem carter_extra_cakes : extra_cakes = 38 :=
by
  unfold regular_cakes triple_cakes extra_cakes
  calc
    3 * 6 + 3 * 5 + 3 * 8 - (6 + 5 + 8)
      = 57 - 19 : by norm_num
    ... = 38 : by norm_num

end carter_extra_cakes_l126_126751


namespace false_propositions_l126_126260

theorem false_propositions :
  (¬ (∀ {l₁ l₂ m : Line}, l₁ ∥ l₂ → l₁ ∥ m → corresponding_angles l₁ l₂ = corresponding_angles l₁ m)) ∧
  (∀ (α : ℝ), 45 < α ∧ α < 90 → sin α > cos α) ∧
  (∀ (x m : ℝ), (3 * x - m) / (x + 2) = 2 ∧ x < 0 → m < -4) ∧
  (¬ (∀ (O : Point) (r : ℝ), ∀ {α β : ℝ}, circle O r → α = β → equal_arcs O α β)) :=
sorry

end false_propositions_l126_126260


namespace ceil_neg_seven_fourths_cubed_eq_neg_five_l126_126417

noncomputable def ceil_of_neg_seven_fourths_cubed : ℤ :=
  Int.ceil ((-7 / 4 : ℚ)^3)

theorem ceil_neg_seven_fourths_cubed_eq_neg_five :
  ceil_of_neg_seven_fourths_cubed = -5 := by
  sorry

end ceil_neg_seven_fourths_cubed_eq_neg_five_l126_126417


namespace equal_good_and_bad_times_l126_126989

def is_good_time (t : ℝ) : Prop :=
  let hour_hand_angle := (t / 12) * 360
  let minute_hand_angle := (t % 1) * 360
  let second_hand_angle := (t * 60 % 1) * 360
  hour_hand_angle < minute_hand_angle ∧ minute_hand_angle < second_hand_angle

theorem equal_good_and_bad_times :
  ∫ t in 0..24, if is_good_time t then 1 else 0 = ∫ t in 0..24, if ¬ is_good_time t then 1 else 0 := 
sorry

end equal_good_and_bad_times_l126_126989


namespace number_of_possible_values_of_a_l126_126611

theorem number_of_possible_values_of_a :
  ∃ (a b c d : ℕ), 
    a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2060 ∧ a^2 - b^2 + c^2 - d^2 = 2060 ∧ 
      {a : ℕ | ∃ (b c d : ℕ), 
        a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2060 ∧ a^2 - b^2 + c^2 - d^2 = 2060}.to_finset.card = 513 :=
by {
  sorry
}

end number_of_possible_values_of_a_l126_126611


namespace finite_or_all_but_finitely_many_l126_126206

noncomputable def P : (ℤ → ℤ) := sorry -- P is a polynomial with integer coefficients

def S (P : ℤ → ℤ) : set ℤ := {n : ℤ | n ≠ 0 ∧ ∃ k : ℤ, P n = k * n}

theorem finite_or_all_but_finitely_many (P : ℤ → ℤ) (h : ∀ n : ℤ, ∃ k : ℤ, P n = k * n):
  S P.finite_or_all_but_finitely_many :=
begin
  -- sorry
end

end finite_or_all_but_finitely_many_l126_126206


namespace part2_l126_126117

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x + (1/2:ℝ) * x^2 - m * x
noncomputable def h (x : ℝ) (c : ℝ) (b : ℝ) : ℝ := f x - c * x^2 - b * x

lemma part1 :
  ∃! x : ℝ, f (x + 1) = x :=
sorry

theorem part2 (m : ℝ) (hm : m ≥ 5/2) (c b : ℝ) (x1 x2 : ℝ) (hx1 : h x1 c b = 0) (hx2 : h x2 c b = 0) (hx1_lt : x1 < x2):
  let y := (x1 - x2) * (2 / (x1 + x2) - c * (x1 + x2) - b) in
  y = -6/5 + Real.log 4 :=
sorry

end part2_l126_126117


namespace unique_triple_gcd_square_l126_126072

theorem unique_triple_gcd_square (m n l : ℕ) (H1 : m + n = Nat.gcd m n ^ 2)
                                  (H2 : m + l = Nat.gcd m l ^ 2)
                                  (H3 : n + l = Nat.gcd n l ^ 2) : (m, n, l) = (2, 2, 2) :=
by
  sorry

end unique_triple_gcd_square_l126_126072


namespace area_ratio_l126_126344

-- Definitions based on problem conditions
def original_area (r : ℝ) : ℝ := π * r^2
def redesigned_area (r : ℝ) : ℝ := π * (3 * r)^2

theorem area_ratio (r : ℝ) (h : r > 0) : original_area r / redesigned_area r = 1 / 9 :=
by
  -- Placeholder proof
  sorry

end area_ratio_l126_126344


namespace find_a_max_integer_k_l126_126493

open Real

/-- Define the function f -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * log x

theorem find_a {a : ℝ} (h_tangent : deriv (f a) e = 3) : a = 1 :=
begin
  sorry
end

theorem max_integer_k (a : ℝ) (h_a : a = 1) (k : ℤ) (hx : ∀ x > 1, (k : ℝ) < (f a x) / (x - 1)) : k ≤ 3 :=
begin
  sorry
end

end find_a_max_integer_k_l126_126493


namespace round_trip_completion_percentage_l126_126023

-- Define the distances for each section
def sectionA_distance : Float := 10
def sectionB_distance : Float := 20
def sectionC_distance : Float := 15

-- Define the speeds for each section
def sectionA_speed : Float := 50
def sectionB_speed : Float := 40
def sectionC_speed : Float := 60

-- Define the delays for each section
def sectionA_delay : Float := 1.15
def sectionB_delay : Float := 1.10

-- Calculate the time for each section without delays
def sectionA_time : Float := sectionA_distance / sectionA_speed
def sectionB_time : Float := sectionB_distance / sectionB_speed
def sectionC_time : Float := sectionC_distance / sectionC_speed

-- Calculate the time with delays for the trip to the center
def sectionA_time_with_delay : Float := sectionA_time * sectionA_delay
def sectionB_time_with_delay : Float := sectionB_time * sectionB_delay
def sectionC_time_with_delay : Float := sectionC_time

-- Total time with delays to the center
def total_time_to_center : Float := sectionA_time_with_delay + sectionB_time_with_delay + sectionC_time_with_delay

-- Total distance to the center
def total_distance_to_center : Float := sectionA_distance + sectionB_distance + sectionC_distance

-- Total round trip distance
def total_round_trip_distance : Float := total_distance_to_center * 2

-- Distance covered on the way back
def distance_back : Float := total_distance_to_center * 0.2

-- Total distance covered considering the delays and the return trip
def total_distance_covered : Float := total_distance_to_center + distance_back

-- Effective completion percentage of the round trip
def completion_percentage : Float := (total_distance_covered / total_round_trip_distance) * 100

-- The main theorem statement
theorem round_trip_completion_percentage :
  completion_percentage = 60 := by
  sorry

end round_trip_completion_percentage_l126_126023


namespace last_digit_of_product_of_consecutive_numbers_l126_126264

theorem last_digit_of_product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h1 : k > 5)
    (h2 : n = (k + 1) * (k + 2) * (k + 3) * (k + 4))
    (h3 : n % 10 ≠ 0) : n % 10 = 4 :=
sorry -- Proof not provided as per instructions.

end last_digit_of_product_of_consecutive_numbers_l126_126264


namespace modulus_of_complex_solution_l126_126479

theorem modulus_of_complex_solution (z : ℂ) (h : (2 - complex.I) * z = 5) : complex.abs z = real.sqrt 5 := 
sorry

end modulus_of_complex_solution_l126_126479


namespace find_longer_diagonal_l126_126130

-- Define the necessary conditions
variables (d1 d2 : ℝ)
variable (A : ℝ)
axiom ratio_condition : d1 / d2 = 2 / 3
axiom area_condition : A = 12

-- Define the problem of finding the length of the longer diagonal
theorem find_longer_diagonal : ∃ (d : ℝ), d = d2 → d = 6 :=
by 
  sorry

end find_longer_diagonal_l126_126130


namespace sum_of_fractions_l126_126042

theorem sum_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (7 : ℚ) / 9
  a + b = 83 / 72 := 
by
  sorry

end sum_of_fractions_l126_126042


namespace min_value_x_plus_y_l126_126103

theorem min_value_x_plus_y (a b : ℝ) (x y : ℝ)
  (h₁ : 0 < a) (h₂ : 0 < b)
  (h₃ : 0 < x) (h₄ : 0 < y)
  (h₅ : a / x + b / y = 2) :
  x + y ≥ (a + b) / 2 + sqrt (a * b) :=
sorry

end min_value_x_plus_y_l126_126103


namespace trapezoid_angles_l126_126998

/-- A trapezoid ABCD with the longer base AB and shorter base CD. 
    AC bisects the interior angle at A. 
    The interior bisector at B meets diagonal AC at E.
    Line DE meets segment AB at F.
    Given AD = FB, BC = AF, and ∠BEC = 54°. 
    Prove that the interior angles of quadrilateral ABCD are 
    ∠A = 72°, ∠B = 36°, ∠C = 108°, and ∠D = 144°. -/
theorem trapezoid_angles
  (A B C D E F : Type)
  (a b c d e f : ℝ)
  (ab bc cd da af fb : ℝ)
  (H1 : cd < ab)
  (H2 : a = 2 * 36)
  (H3 : b = 2 * 18)
  (H4 : 2 * a + 2 * b = 108)
  (H5 : da = fb)
  (H6 : bc = af)
  (H7 : ∠A + ∠B + ∠C + ∠D = 360)
  (H8 : ∠BEC = 54) :
  ∠A = 72 ∧ ∠B = 36 ∧ ∠C = 108 ∧ ∠D = 144 := 
sorry

end trapezoid_angles_l126_126998


namespace value_of_y_l126_126144

theorem value_of_y (x y : ℝ) (hx : x = 3) (h : x^(3 * y) = 9) : y = 2 / 3 := by
  sorry

end value_of_y_l126_126144


namespace number_of_scalene_triangles_l126_126383

-- Define a triangle
structure Triangle :=
  (a : ℕ) (b : ℕ) (c : ℕ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : a + b > c)
  (h4 : a + c > b)
  (h5 : b + c > a)

-- Define a predicate for a scalene triangle with a perimeter less than 20
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b + t.c < 20

-- Main theorem statement
theorem number_of_scalene_triangles : 
  {t : Triangle // isValidTriangle t}.card > 7 := 
sorry

end number_of_scalene_triangles_l126_126383


namespace problem_solution_l126_126120

def p : Prop := ∀ x : ℝ, |x| ≥ 0
def q : Prop := ∃ x : ℝ, x = 2 ∧ x + 2 = 0

theorem problem_solution : p ∧ ¬q :=
by
  -- Here we would provide the proof to show that p ∧ ¬q is true
  sorry

end problem_solution_l126_126120


namespace simplify_expression_l126_126621

theorem simplify_expression (x y : ℝ) : (2 * x + 3 * complex.I * y) * (2 * x - 3 * complex.I * y) = 4 * x ^ 2 - 9 * y ^ 2 := 
by 
  -- proof steps will go here, but for now we just put sorry
  sorry

end simplify_expression_l126_126621


namespace spinsters_count_l126_126653

variable (S C : ℕ)

-- defining the conditions
def ratio_condition (S C : ℕ) : Prop := 9 * S = 2 * C
def difference_condition (S C : ℕ) : Prop := C = S + 63

-- theorem to prove
theorem spinsters_count 
  (h1 : ratio_condition S C) 
  (h2 : difference_condition S C) : 
  S = 18 :=
sorry

end spinsters_count_l126_126653


namespace cubic_intersection_unique_point_l126_126853

-- Define the cubic functions f and g
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d
def g (a b c d x : ℝ) : ℝ := -a * x^3 + b * x^2 - c * x + d

-- Translate conditions into Lean conditions
variables (a b c d : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Lean statement to prove the intersection point
theorem cubic_intersection_unique_point :
  ∀ x y : ℝ, (f a b c d x = y) ↔ (g a b c d x = y) → (x = 0 ∧ y = d) :=
by
  -- Mathematical steps would go here (omitted with sorry)
  sorry

end cubic_intersection_unique_point_l126_126853


namespace product_cubed_roots_l126_126041

-- Given conditions
def cbrt (x : ℝ) : ℝ := x^(1/3)
def expr : ℝ := cbrt (1 + 27) * cbrt (1 + cbrt 27) * cbrt 9

-- Main statement to prove
theorem product_cubed_roots : expr = cbrt 1008 :=
by sorry

end product_cubed_roots_l126_126041


namespace paul_baseball_cards_l126_126232

-- Define the necessary variables and statements
variable {n : ℕ}

-- State the problem and the proof target
theorem paul_baseball_cards : ∃ k, k = 3 * n + 1 := sorry

end paul_baseball_cards_l126_126232


namespace jacob_dimes_l126_126224

-- Definitions of the conditions
def mrs_hilt_total_cents : ℕ := 2 * 1 + 2 * 10 + 2 * 5
def jacob_base_cents : ℕ := 4 * 1 + 1 * 5
def difference : ℕ := 13

-- The proof problem: prove Jacob has 1 dime.
theorem jacob_dimes (d : ℕ) (h : mrs_hilt_total_cents - (jacob_base_cents + 10 * d) = difference) : d = 1 := by
  sorry

end jacob_dimes_l126_126224


namespace ellipse_foci_coordinates_l126_126476

theorem ellipse_foci_coordinates :
  (∃ m : ℝ, m > 0 ∧ m^2 = 2 * 8) →
  (∀ x y : ℝ, x^2 + y^2 / m = 1 → (x = 0 ∧ (y = sqrt 3 ∨ y = -sqrt 3))) :=
by
  intro h
  obtain ⟨m, hm_pos, hm_eq⟩ := h
  have hm : m = 4 := by sorry  -- proof that m = 4
  sorry  -- remaining proof

end ellipse_foci_coordinates_l126_126476


namespace max_objective_function_value_l126_126426

def objective_function (x1 x2 : ℝ) := 4 * x1 + 6 * x2

theorem max_objective_function_value :
  ∃ x1 x2 : ℝ, 
    (x1 >= 0) ∧ 
    (x2 >= 0) ∧ 
    (x1 + x2 <= 18) ∧ 
    (0.5 * x1 + x2 <= 12) ∧ 
    (2 * x1 <= 24) ∧ 
    (2 * x2 <= 18) ∧ 
    (∀ y1 y2 : ℝ, 
      (y1 >= 0) ∧ 
      (y2 >= 0) ∧ 
      (y1 + y2 <= 18) ∧ 
      (0.5 * y1 + y2 <= 12) ∧ 
      (2 * y1 <= 24) ∧ 
      (2 * y2 <= 18) -> 
      objective_function y1 y2 <= objective_function x1 x2) ∧
    (objective_function x1 x2 = 84) :=
by
  use 12, 6
  sorry

end max_objective_function_value_l126_126426


namespace log_conversion_l126_126874

theorem log_conversion (x : ℝ) (h : log 4 (x - 3) = 1 / 2) : log 216 x = (1 / 3) * log 6 5 := by
  sorry

end log_conversion_l126_126874


namespace area_of_cos_closed_figure_l126_126633

theorem area_of_cos_closed_figure :
  ∫ x in (Real.pi / 2)..(3 * Real.pi / 2), Real.cos x = 2 :=
by
  sorry

end area_of_cos_closed_figure_l126_126633


namespace g_domain_F_domain_F_max_l126_126487

noncomputable def f (x : ℝ) := real.logb 2 (x + 1)

noncomputable def g (x : ℝ) := 2 * real.logb 2 (x + 2)

noncomputable def F (x : ℝ) := real.logb 2 x - g x

-- Domain for g(x)
theorem g_domain : ∀ x, x > -2 → 0 < x + 2 := sorry

-- Domain for F(x)
theorem F_domain : ∀ x, x > 0 → 0 < x := sorry

-- Maximum value of F(x)
theorem F_max : ∀ x, x > 0 → F x ≤ F 2 := sorry

end g_domain_F_domain_F_max_l126_126487


namespace supercomputer_multiplications_per_half_day_l126_126022

theorem supercomputer_multiplications_per_half_day :
  let r := 80000 in
  let t := 12 in
  let seconds_per_hour := 3600 in
  r * t * seconds_per_hour = 3456000000 := by
  sorry

end supercomputer_multiplications_per_half_day_l126_126022


namespace percentage_equivalence_l126_126528

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end percentage_equivalence_l126_126528


namespace cant_obtain_target_from_1_l126_126193

/-- Define the 100-digit number 222...2225 (with 99 twos and one five) as a string representation -/
def target_num := "2"^(99) ++ "5"

/-- Define a function to check if it is possible to obtain a certain number by repeatedly multiplying by 5 or rearranging digits -/
def can_obtain (start : String) (target : String) : Prop :=
  sorry  -- This would involve complex logic to encode allowed operations, not required for this statement

/-- Prove that it is not possible to obtain the target 100-digit number from 1 using the allowed operations -/
theorem cant_obtain_target_from_1 : ¬ can_obtain "1" target_num :=
sorry

end cant_obtain_target_from_1_l126_126193


namespace megan_pages_left_l126_126223

theorem megan_pages_left (total_problems completed_problems problems_per_page : ℕ)
    (h_total : total_problems = 40)
    (h_completed : completed_problems = 26)
    (h_problems_per_page : problems_per_page = 7) :
    (total_problems - completed_problems) / problems_per_page = 2 :=
by
  sorry

end megan_pages_left_l126_126223


namespace ratio_of_surface_areas_eq_l126_126544

-- Definitions of the given conditions
def side_length_of_square (a : ℝ) : ℝ := a
def height_of_cylinder (a : ℝ) : ℝ := a
def circumference_of_base (a : ℝ) : ℝ := a
def radius_of_base (a : ℝ) : ℝ := a / (2 * π)

def base_area (a : ℝ) : ℝ := π * (radius_of_base a) ^ 2
def side_surface_area (a : ℝ) : ℝ := a ^ 2
def total_surface_area (a : ℝ) : ℝ := 2 * base_area a + side_surface_area a

-- The ratio of total surface area to side surface area
def ratio_of_surface_areas (a : ℝ) : ℝ := total_surface_area a / side_surface_area a

-- The theorem to be proven
theorem ratio_of_surface_areas_eq (a : ℝ) : ratio_of_surface_areas a = (1 + 2 * π) / (2 * π) :=
by 
    -- This is where you would provide the proof.
    -- Currently left as a placeholder.
    sorry

end ratio_of_surface_areas_eq_l126_126544


namespace lex_read_pages_l126_126215

theorem lex_read_pages (total_pages days : ℕ) (h1 : total_pages = 240) (h2 : days = 12) :
  total_pages / days = 20 :=
by sorry

end lex_read_pages_l126_126215


namespace side_of_square_is_25_l126_126315

theorem side_of_square_is_25 (area_of_circle : ℝ) (perimeter_of_square : ℝ) (h1 : area_of_circle = 100) (h2 : area_of_circle = perimeter_of_square) : perimeter_of_square / 4 = 25 :=
by {
  -- Insert the steps here if necessary.
  sorry
}

end side_of_square_is_25_l126_126315


namespace chromatic_number_of_union_l126_126438

open_locale classical

-- Define the chromatic number of a graph
noncomputable def chromatic_number (G : Type*) [graph G] : ℕ := sorry

-- Define the union of two graphs on the same vertex set
def graph_union {V : Type*} (G H : V → V → Prop) : V → V → Prop :=
λ v1 v2, G v1 v2 ∨ H v1 v2

-- Define the main theorem statement
theorem chromatic_number_of_union {V : Type*} (G H : V → V → Prop) (a b : ℕ) :
  chromatic_number G = a →
  chromatic_number H = b →
  (∃ (G_union : V → V → Prop), ∀ v, G_union v v = (G v v ∨ H v v)) →
  chromatic_number (graph_union G H) ≤ a * b :=
sorry

end chromatic_number_of_union_l126_126438


namespace find_number_l126_126012

theorem find_number :
  ∃ (x : ℤ), 
  x * (x + 6) = -8 ∧ 
  x^4 + (x + 6)^4 = 272 :=
by
  sorry

end find_number_l126_126012


namespace arithmetic_geom_problem_l126_126094

theorem arithmetic_geom_problem
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (b_n : ℕ → ℤ)
  (T_n : ℕ → ℤ)
  (d : ℤ)
  (h1 : a_n 1 = 1)
  (h2 : ∃ k, k ≠ 0 ∧ (a_n 2, a_n 5, a_n 14) = (1 + d, 1 + 4 * d, 1 + 13 * d) ∧ (1 + d) * (1 + 13 * d) = (1 + 4 * d) ^ 2)
  (h3 : ∀ n, S_n n = n * a_n 1 + (n * (n - 1) / 2) * d)
  (h4 : ∀ n, b_n n = (-1) ^ n * S_n n)
  (h5 : ∀ n, T_n n = ∑ i in range n, b_n (i + 1)) :
  (∀ n, a_n n = 2 * n - 1) ∧ (∀ n, T_n n = (-1)^n * (n * (n + 1) / 2)) := 
sorry

end arithmetic_geom_problem_l126_126094


namespace distinguishable_balls_in_distinguishable_boxes_l126_126136

theorem distinguishable_balls_in_distinguishable_boxes :
  ∃ (ways : ℕ), ways = (3^5) ∧ ways = 243 :=
by {
  use (3^5), -- Define the number of ways.
  split,
  exact rfl, -- Prove that (3^5) = (3^5).
  norm_num, -- Normalize the numerical expression to show that (3^5) = 243.
  sorry -- Placeholder for the proof.
}

end distinguishable_balls_in_distinguishable_boxes_l126_126136


namespace function_characterization_l126_126423

theorem function_characterization (f : ℕ → ℕ)
  (H1 : ∀ x y : ℕ, x + y ∣ f x + f y)
  (H2 : ∀ x : ℕ, 1395 ≤ x → x^3 ≥ 2 * f x)
  : ∃ k : ℕ, (∀ n : ℕ, f n = k * n) ∧ k ≤ 1395^2 / 2 :=
begin
  sorry
end

end function_characterization_l126_126423


namespace surface_area_increase_factor_l126_126347

theorem surface_area_increase_factor (n : ℕ) (h : n > 0) : 
  (6 * n^3) / (6 * n^2) = n :=
by {
  sorry -- Proof not required
}

end surface_area_increase_factor_l126_126347


namespace equal_magnitude_vectors_l126_126928

variables {V : Type*} [normed_add_comm_group V] [normed_space ℝ V]
variables {A B C D O : V}

-- Conditions given in the problem
def is_center_of_square (O A B C D : V) : Prop :=
  ∥O - A∥ = ∥O - B∥ ∧ ∥O - B∥ = ∥O - C∥ ∧ ∥O - C∥ = ∥O - D∥ ∧
  ∥A - B∥ = ∥B - C∥ ∧ ∥B - C∥ = ∥C - D∥ ∧ ∥C - D∥ = ∥D - A∥ ∧
  ∥A - C∥ = ∥B - D∥

-- Theorem to be proved
theorem equal_magnitude_vectors (h : is_center_of_square O A B C D) :
  ∥O - A∥ = ∥O - B∥ ∧ ∥O - B∥ = ∥O - C∥ ∧ ∥O - C∥ = ∥O - D∥ :=
by {
  sorry
}

end equal_magnitude_vectors_l126_126928


namespace angle_A_of_triangle_l126_126188

theorem angle_A_of_triangle (a c : ℝ) (C A : ℝ) 
  (h1 : a = 1) 
  (h3 : c = real.sqrt 3) 
  (h4 : C = real.pi / 3) : 
  A = real.pi / 6 :=
by
  sorry

end angle_A_of_triangle_l126_126188


namespace carter_bakes_more_cakes_l126_126747

/-- Carter usually bakes 6 cheesecakes, 5 muffins, and 8 red velvet cakes in a week.
This week he was able to bake triple the number of each type of cake.
Prove that Carter baked 38 more cakes this week than usual. -/
theorem carter_bakes_more_cakes :
  let regular_cheesecakes := 6
  let regular_muffins := 5
  let regular_red_velvet_cakes := 8
  let regular_total := regular_cheesecakes + regular_muffins + regular_red_velvet_cakes
  let tripled_cheesecakes := 3 * regular_cheesecakes
  let tripled_muffins := 3 * regular_muffins
  let tripled_red_velvet_cakes := 3 * regular_red_velvet_cakes
  let tripled_total := tripled_cheesecakes + tripled_muffins + tripled_red_velvet_cakes
  in tripled_total - regular_total = 38 :=
by
  let regular_cheesecakes := 6
  let regular_muffins := 5
  let regular_red_velvet_cakes := 8
  let regular_total := regular_cheesecakes + regular_muffins + regular_red_velvet_cakes
  let tripled_cheesecakes := 3 * regular_cheesecakes
  let tripled_muffins := 3 * regular_muffins
  let tripled_red_velvet_cakes := 3 * regular_red_velvet_cakes
  let tripled_total := tripled_cheesecakes + tripled_muffins + tripled_red_velvet_cakes
  show tripled_total - regular_total = 38 from sorry

end carter_bakes_more_cakes_l126_126747


namespace range_of_a_l126_126657

theorem range_of_a (x a : ℝ) :
  (∀ x : ℝ, x - 1 < 0 ∧ x < a + 3 → x < 1) → a ≥ -2 :=
by
  sorry

end range_of_a_l126_126657


namespace area_of_enclosed_figure_l126_126074

open Real

noncomputable def enclosed_area : ℝ :=
  2 * (∫ x in 0..(π / 3), (2 * cos x ^ 2 - 1 / 2)) + 
  2 * (∫ x in (π / 3)..(π / 2), (-2 * cos x ^ 2 + 1 / 2))

theorem area_of_enclosed_figure :
  enclosed_area = sqrt 3 + (π / 6) :=
by
  sorry

end area_of_enclosed_figure_l126_126074


namespace sin_alpha_minus_pi_over_6_l126_126819

open Real

theorem sin_alpha_minus_pi_over_6 (α : ℝ) (h : sin (α + π / 6) + 2 * sin (α / 2) ^ 2 = 1 - sqrt 2 / 2) : 
  sin (α - π / 6) = -sqrt 2 / 2 :=
sorry

end sin_alpha_minus_pi_over_6_l126_126819


namespace trigonometric_identity_l126_126088

open Real

theorem trigonometric_identity (α : ℝ) (hα : sin (2 * π - α) = 4 / 5) (hα_range : 3 * π / 2 < α ∧ α < 2 * π) : 
  (sin α + cos α) / (sin α - cos α) = 1 / 7 := 
by
  sorry

end trigonometric_identity_l126_126088


namespace minimum_shirts_to_save_money_by_using_Acme_l126_126027

-- Define the cost functions for Acme and Gamma
def Acme_cost (x : ℕ) : ℕ := 60 + 8 * x
def Gamma_cost (x : ℕ) : ℕ := 12 * x

-- State the theorem to prove that for x = 16, Acme is cheaper than Gamma
theorem minimum_shirts_to_save_money_by_using_Acme : ∀ x ≥ 16, Acme_cost x < Gamma_cost x :=
by
  intros x hx
  sorry

end minimum_shirts_to_save_money_by_using_Acme_l126_126027


namespace monotonic_intervals_min_x2_minus_x1_range_of_a_l126_126490

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x^2 + 2*x + a else real.log x

-- Definitions for the points A and B on the graph of function f
def A (x1 : ℝ) (a : ℝ) := (x1, f x1 a)
def B (x2 : ℝ) (a : ℝ) := (x2, f x2 a)

-- Ⅰ. Prove the intervals of monotonicity of the function f(x)
theorem monotonic_intervals (a : ℝ) : 
  (∀ x, x < 0 → ((-∞ < x) ∧ (x < -1) → f(x) = (x+1)^2 + a ∧ ((deriv (λ x, (x+1)^2 + a) x < 0))  ∧ ([-1 < x] ∧ (x < 0) → f(x) = (x+1)^2 + a ∧ (deriv (λ x, (x+1)^2 + a) x > 0))) ∧ 
  (∀ x, 0 < x → f(x) = real.log x ∧ (deriv real.log x > 0)))  := 
sorry

-- Ⅱ. Prove the minimum value of x2 - x1 when tangents to f(x) at points A and B are perpendicular, and x2 < 0
theorem min_x2_minus_x1 (a : ℝ) (x1 : ℝ) (x2 : ℝ) (h1 : x1 < x2) (h2 : x2 < 0) : 
  ∀ (x1 x2 : ℝ), (tangent_perpendicular (A x1 a) (B x2 a)) → (x2 - x1 ≥ 1) :=
sorry

-- Ⅲ. Prove the range of values for a when the tangents to f(x) at points A and B coincide
theorem range_of_a (a : ℝ) (x1 : ℝ) (x2 : ℝ) (h : x1 < 0 < x2) : 
  (∃ a : ℝ, tangents_coincide (A x1 a) (B x2 a) ∧ a ∈ Ioi (-1 - real.log 2)) :=
sorry

-- Definitions for tangent_perpendicular and tangents_coincide must be defined

end monotonic_intervals_min_x2_minus_x1_range_of_a_l126_126490


namespace min_value_of_f_l126_126473

theorem min_value_of_f (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  let f (x y z : ℝ) := (3 * x^2 - x) / (1 + x^2) + (3 * y^2 - y) / (1 + y^2) + (3 * z^2 - z) / (1 + z^2)
  in f x y z ≥ 0 :=
by
  sorry

end min_value_of_f_l126_126473


namespace angle_BAT_eq_angle_DAC_l126_126700

theorem angle_BAT_eq_angle_DAC 
  (ABCD : Quadrilateral)
  (T : Point)
  (h1 : angle ACB = angle ACD)
  (h2 : angle ADC - angle ATB = angle BAC)
  (h3 : angle ABC - angle ATD = angle CAD) : 
  angle BAT = angle DAC := 
sorry

end angle_BAT_eq_angle_DAC_l126_126700


namespace ellipse_equation_is_correct_area_AMN_is_correct_l126_126827

-- Definitions for conditions
def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)

def slope_condition (k : ℝ) : Prop := 
  k > 0

def point_A : Prop :=
  A = (-2, 0)

def intersect_points (x y k : ℝ) : Prop :=
  (y = k * (x + 2)) ∧ (x^2 / 4 + y^2 / 3 = 1)

def points_A_M_N (x_A y_A x_M y_M x_N y_N k : ℝ) : Prop :=
  pt_A x_A y_A ∧ pt_M x_M y_M ∧ pt_N x_N y_N ∧
  (y_M = k * (x_M + 2)) ∧ (x_M^2 / 4 + y_M^2 / 3 = 1) ∧ 
  (x_N^2 / 4 + y_N^2 / 3 = 1) ∧ (x_A = -2) ∧ (y_A = 0) ∧
  (y_N = -x_N / (4 * k))

def length_and_perpendicular (x_A y_A x_M y_M x_N y_N : ℝ) : Prop :=
  (y_M - y_A)^2 + (x_M - x_A)^2 = (y_N - y_A)^2 + (x_N - x_A)^2 ∧
  (x_N - x_A)*(x_M - x_A) + (y_N - y_A)*(y_M - y_A) = 0


def area_AMN (x_A y_A x_M y_M x_N y_N : ℝ) : ℝ :=
  (1 / 2) * abs (x_A*(y_M - y_N) + x_M*(y_N - y_A) + x_N*(y_A - y_M))

noncomputable def correct_ellipse : Prop :=
  ellipse_equation

noncomputable def correct_area_of_AMN (x_A y_A x_M y_M x_N y_N : ℝ) : Prop :=
  length_and_perpendicular x_A y_A x_M y_M x_N y_N →
  area_AMN x_A y_A x_M y_M x_N y_N = 144 / 49

-- Statements
theorem ellipse_equation_is_correct : correct_ellipse :=
sorry

theorem area_AMN_is_correct {k : ℝ} : slope_condition k → 
                                   points_A_M_N x_A 0 x_M y_M x_N y_N k → 
                                   length_and_perpendicular x_A 0 x_M y_M x_N y_N → 
                                   correct_area_of_AMN x_A 0 x_M y_M x_N y_N :=
sorry

end ellipse_equation_is_correct_area_AMN_is_correct_l126_126827


namespace number_of_correct_propositions_l126_126502

variable (α β : Plane) (m n : Line)

-- Conditions
variable (h₀: m ∉ α) (h₁: m ∉ β) (h₂: n ∉ α) (h₃: n ∉ β) 
variable (h₄: m ≠ n) (h₅: α ≠ β)

-- Statements
variable (s₁: Perpendicular m n) (s₂: Perpendicular α β) (s₃: Perpendicular n β) (s₄: Perpendicular m α)

-- Conclusion: Number of correct propositions is 2
theorem number_of_correct_propositions : 
  -- The number of ways to pick three statements to imply the remaining one
  (1 + 1) = 2 := 
by sorry

end number_of_correct_propositions_l126_126502


namespace youngest_child_age_l126_126659

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) : 
  x = 4 := by
  sorry

end youngest_child_age_l126_126659


namespace side_of_square_is_25_l126_126316

theorem side_of_square_is_25 (area_of_circle : ℝ) (perimeter_of_square : ℝ) (h1 : area_of_circle = 100) (h2 : area_of_circle = perimeter_of_square) : perimeter_of_square / 4 = 25 :=
by {
  -- Insert the steps here if necessary.
  sorry
}

end side_of_square_is_25_l126_126316


namespace problem1_problem2_problem3_l126_126727

-- Definitions for a ladder sequence
def ladder_sequence (a : ℕ → ℕ) : Prop := 
  ∀ n : ℕ, 0 < n → a (2 * n - 1) = a (2 * n) ∧ a (2 * n) < a (2 * n + 1)

-- Problem 1: Define the specific sequence and prove the value
def b : ℕ → ℕ
| 0 := 0
| 1 := 1
| (2 * n + 1) := 9 * b (2 * n - 1)
| (2 * n) := b (2 * n - 1)

theorem problem1 : b 2016 = 3 ^ 2014 := by sorry

-- Problem 2: Prove the properties about the sequence S_n
def S (c : ℕ → ℕ) (n : ℕ) : ℕ := (finset.Ico 0 n).sum c

theorem problem2 (c : ℕ → ℕ) (h : ladder_sequence c) : 
  (∃ k, c (2 * k - 1) = c (2 * k)) ∧ 
  ¬ ∃ m, c (2 * m) = c (2 * m + 1) ∧ c (2 * m + 1) = c (2 * m + 2) := by sorry

-- Problem 3: Define the specific sequence and prove the range of t
def d : ℕ → ℕ
| 0 := 0
| 1 := 1
| (2 * n + 1) := d (2 * n - 1) + 2
| (2 * n) := d (2 * n - 1)

def T (n : ℕ) : ℝ := (finset.Ico 0 n).sum (λ i, 1 / (d i * d (i + 2)))

theorem problem3 : ∃ t : ℝ, -1 ≤ t ∧ t < (1 : ℝ) / 3 ∧ ∀ n : ℕ, (t - T n) * (t + T n) < 0 := by sorry

end problem1_problem2_problem3_l126_126727


namespace modular_inverse_of_31_mod_35_is_1_l126_126077

theorem modular_inverse_of_31_mod_35_is_1 :
  ∃ a : ℕ, 0 ≤ a ∧ a < 35 ∧ 31 * a % 35 = 1 := sorry

end modular_inverse_of_31_mod_35_is_1_l126_126077


namespace chromium_first_alloy_percentage_l126_126898

-- Defining the conditions
def percentage_chromium_first_alloy : ℝ := 10 
def percentage_chromium_second_alloy : ℝ := 6
def mass_first_alloy : ℝ := 15
def mass_second_alloy : ℝ := 35
def percentage_chromium_new_alloy : ℝ := 7.2

-- Proving the percentage of chromium in the first alloy is 10%
theorem chromium_first_alloy_percentage : percentage_chromium_first_alloy = 10 :=
by
  sorry

end chromium_first_alloy_percentage_l126_126898


namespace admits_primitives_l126_126579

theorem admits_primitives (a : ℝ) (h_a : a > 0)
  (f : ℝ → ℝ) (h_f_cont : ∀ x ∈ Ioo 0 ∞, continuous_at f x)
  (h_f_darboux : ∀ y z : ℝ, y ∈ Icc 0 ∞ → z ∈ Icc 0 ∞ →
    (∃ c, c ∈ Icc (min y z) (max y z) ∧ f c ∈ Icc (f y) (f z)))
  (h_f0 : f 0 = 0)
  (h_f_condition : ∀ x : ℝ, 0 ≤ x → x * f x ≥ ∫ t in 0..x, f t) :
  ∃ (F : ℝ → ℝ), ∀ x : ℝ, 0 ≤ x → has_deriv_at F (f x) x :=
begin
  sorry,
end

end admits_primitives_l126_126579


namespace percentage_equivalence_l126_126527

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end percentage_equivalence_l126_126527


namespace friend_saves_per_week_l126_126320

theorem friend_saves_per_week
  (x : ℕ) 
  (you_have : ℕ := 160)
  (you_save_per_week : ℕ := 7)
  (friend_have : ℕ := 210)
  (weeks : ℕ := 25)
  (total_you_save : ℕ := you_have + you_save_per_week * weeks)
  (total_friend_save : ℕ := friend_have + x * weeks) 
  (h : total_you_save = total_friend_save) : x = 5 := 
by 
  sorry

end friend_saves_per_week_l126_126320


namespace monotonic_increasing_interval_log2_abs_l126_126999

noncomputable def log2_abs : ℝ → ℝ := λ x, abs (Real.log 2 x)

theorem monotonic_increasing_interval_log2_abs :
  { x : ℝ | x > 0 } ∩ { x : ℝ | log2_abs x >= log2_abs 1 } = { x : ℝ | x ≥ 1 } := by
  sorry

end monotonic_increasing_interval_log2_abs_l126_126999


namespace no_infinite_family_of_lines_l126_126228

theorem no_infinite_family_of_lines :
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, (1 : ℝ) = k n * (1 : ℝ)) ∧ (∀ n, k (n+1) = k n - 1 / k n) ∧ (∀ n, k n * k (n+1) ≥ 0) :=
begin
  sorry
end

end no_infinite_family_of_lines_l126_126228


namespace unit_vector_AB_l126_126468

theorem unit_vector_AB (A B : ℝ × ℝ) (hA : A = (1, 3)) (hB : B = (4, -1)) :
  ∃ u : ℝ × ℝ, u = (3 / 5, -4 / 5) ∧ ∃ v, v = (B.1 - A.1, B.2 - A.2) ∧ u = (1 / real.sqrt ((v.1)^2 + (v.2)^2)) • v :=
by
  -- This is the statement of the theorem.
  -- The actual proof involves calculating vector AB, its magnitude,
  -- and finally, the unit vector in the direction of AB.
  sorry

end unit_vector_AB_l126_126468


namespace pre_images_of_one_l126_126214

def f (x : ℝ) := x^3 - x + 1

theorem pre_images_of_one : {x : ℝ | f x = 1} = {-1, 0, 1} :=
by {
  sorry
}

end pre_images_of_one_l126_126214


namespace polynomial_simplification_l126_126964

theorem polynomial_simplification (x : ℝ) :
    (3 * x - 2) * (5 * x^12 - 3 * x^11 + 4 * x^9 - 2 * x^8)
    = 15 * x^13 - 19 * x^12 + 6 * x^11 + 12 * x^10 - 14 * x^9 - 4 * x^8 := by
  sorry

end polynomial_simplification_l126_126964


namespace length_of_QZ_l126_126563

-- Definitions based on the given problem conditions
variables {A B Z Y Q : Type} [MetricSpace A] [MetricSpace B] [MetricSpace Z] 
[MetricSpace Y] [MetricSpace Q]
variable (AB_parallel_YZ : ∃ (A B Z Y : Type), 
  ∀ (l₁ l₂ : ℝ), IsParallel l₁ l₂ → (segment AB ∥ segment YZ))
variable (AZ_len : Real := 54)
variable (BQ_len : Real := 18)
variable (QY_len : Real := 36)
variable (QZ_len : Real := 36)

-- The statement to be proved
theorem length_of_QZ (AB_parallel_YZ : ∀ (l₁ l₂ : ℝ), IsParallel l₁ l₂ → (segment AB ∥ segment YZ)) 
  (AZ_eq : AZ = 54)
  (BQ_eq : BQ = 18)
  (QY_eq : QY = 36) : 
  QZ = 36 :=
by 
  sorry

end length_of_QZ_l126_126563


namespace sum_geometric_series_l126_126743

theorem sum_geometric_series :
  let S := (finset.range 2013).sum (λ n, 5 ^ n)
  in S = (5 ^ 2013 - 1) / 4 :=
by
  sorry

end sum_geometric_series_l126_126743


namespace ratio_of_length_to_width_of_field_l126_126262

theorem ratio_of_length_to_width_of_field (L W : ℕ) (pond_side length_of_field : ℕ) 
  (h1 : pond_side = 8) (h2 : length_of_field = 96) 
  (h3 : L = length_of_field) 
  (h4 : 64 * 72 = L * W := by sorry)
  : L / W = 2 := by sorry

-- Use these default values for proving our specific scenario
def main : IO Unit :=
  IO.println (ratio_of_length_to_width_of_field 96 48 8 96 rfl rfl rfl sorry) -- This line is just to check Lean syntax

end ratio_of_length_to_width_of_field_l126_126262


namespace red_segments_diagonal_length_l126_126469

theorem red_segments_diagonal_length (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  let L := ((m^2 + n^2 : ℕ) : ℤ)
  L / 2 := (m^2 + n^2 : ℕ).to_real.sqrt / 2 :=
sorry

end red_segments_diagonal_length_l126_126469


namespace solution_l126_126104

-- Define M and N according to the given conditions
def M : Set ℝ := {x | x < 0 ∨ x > 2}
def N : Set ℝ := {x | x ≥ 1}

-- Define the complement of M in Real numbers
def complementM : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the union of the complement of M and N
def problem_statement : Set ℝ := complementM ∪ N

-- State the theorem
theorem solution :
  problem_statement = { x | x ≥ 0 } :=
by
  sorry

end solution_l126_126104


namespace babysitter_weekly_hour_limit_l126_126703

theorem babysitter_weekly_hour_limit
  (R : ℝ) (O : ℝ) (E : ℝ) (H : ℝ) (L : ℝ)
  (hR : R = 16)
  (hO : O = 1.75 * R)
  (hE : E = 760)
  (hH : H = 40)
  (earnings_eq : E = L * R + (H - L) * O) :
  L = 30 :=
by
  -- import conditions
  have h1 : 16 = R := by exact hR
  have h2 : O = 1.75 * R := by exact hO
  have h3 : 760 = E := by exact hE
  have h4 : 40 = H := by exact hH
  
  -- calculate overtime rate
  rw h1 at h2
  have h5 : O = 1.75 * 16 := by exact h2
  norm_num at h5
  rw ←h5 at earnings_eq
  
  -- simplify the earnings equation
  rw h3 at earnings_eq
  rw h4 at earnings_eq
  norm_num at earnings_eq
  
  admit

end babysitter_weekly_hour_limit_l126_126703


namespace ellipse_problem_l126_126933

-- Let the left and right foci of the ellipse be F1 and F2 respectively
def ellipse_foci (a b c : ℝ) (hab : a > b) (hbc : b > 0) (h_eq : a = 2 * c) : Prop :=
  let F1 := (-c, 0)
  let F2 := (c, 0)
  ∃ P : ℝ × ℝ, P = (a, b) ∧ (a > b) ∧ (b > 0) ∧ (∥P - F2∥ = ∥F1 - F2∥) 

-- Given conditions and calculate eccentricity 'e'
def eccentricity (a c : ℝ) (h_eq : a = 2 * c) : ℝ := c / a

-- The main problem statement to prove
theorem ellipse_problem (a b c : ℝ)
  (hab : a > b) (hbc : b > 0) (h_eq : a = 2 * c) 
  (h_eccentricity : eccentricity a c h_eq = 1/2) :
  ellipse_foci a b c hab hbc h_eq → ∃ k: ℝ, ellipse.eq (a:=k*sqrt 3) (b:=3*k) (c:=4*k^2) :=
sorry

end ellipse_problem_l126_126933


namespace largest_four_digit_number_with_digits_sum_25_l126_126302

def four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  (n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10) = s)

theorem largest_four_digit_number_with_digits_sum_25 :
  ∃ n, four_digit n ∧ digits_sum_to n 25 ∧ ∀ m, four_digit m → digits_sum_to m 25 → m ≤ n :=
sorry

end largest_four_digit_number_with_digits_sum_25_l126_126302


namespace total_number_of_elements_l126_126982

theorem total_number_of_elements (a b c : ℕ) : 
  (a = 2 ∧ b = 2 ∧ c = 2) ∧ 
  (3.95 = ((4.4 * 2 + 3.85 * 2 + 3.6000000000000014 * 2) / 6)) ->
  a + b + c = 6 := 
by
  sorry

end total_number_of_elements_l126_126982


namespace dan_spent_amount_l126_126056

-- Defining the prices of items
def candy_bar_price : ℝ := 7
def chocolate_price : ℝ := 6
def gum_price : ℝ := 3
def chips_price : ℝ := 4

-- Defining the discount and tax rates
def candy_bar_discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

-- Defining the steps to calculate the total price including discount and tax
def total_before_discount_and_tax := candy_bar_price + chocolate_price + gum_price + chips_price
def candy_bar_discount := candy_bar_discount_rate * candy_bar_price
def candy_bar_after_discount := candy_bar_price - candy_bar_discount
def total_after_discount := candy_bar_after_discount + chocolate_price + gum_price + chips_price
def tax := tax_rate * total_after_discount
def total_with_discount_and_tax := total_after_discount + tax

theorem dan_spent_amount : total_with_discount_and_tax = 20.27 :=
by sorry

end dan_spent_amount_l126_126056


namespace total_revenue_is_correct_l126_126218

-- Define the constants and conditions
def price_of_jeans : ℕ := 11
def price_of_tees : ℕ := 8
def quantity_of_tees_sold : ℕ := 7
def quantity_of_jeans_sold : ℕ := 4

-- Define the total revenue calculation
def total_revenue : ℕ :=
  (price_of_tees * quantity_of_tees_sold) +
  (price_of_jeans * quantity_of_jeans_sold)

-- The theorem to prove
theorem total_revenue_is_correct : total_revenue = 100 := 
by
  -- Proof is omitted for now
  sorry

end total_revenue_is_correct_l126_126218


namespace matchstick_rearrangement_l126_126222

theorem matchstick_rearrangement :
  ∃ (lhs rhs : ℝ), lhs - rhs < 0.1 :=
by
  let lhs_initial := 23 / 12
  let rhs_initial := 2
  have diff_initial: rhs_initial - lhs_initial < 0.1 := 
    by norm_num; exact sub_lt_of_abs_sub_lt _ _ _ (abs_sub_lt_iff.mp (abs_sub_lt_iff.mpr (sub_lt_iff_lt_add.mpr (by norm_num)))).left

  let lhs_after := 22 / 7
  let rhs_after := real.pi
  have diff_after: abs(lhs_after - rhs_after) < 0.1 :=
    by norm_num; exact sub_lt_of_abs_sub_lt _ _ _ (abs_sub_lt_iff.mp (abs_sub_lt_iff.mpr (sub_lt_iff_lt_add.mpr (by norm_num)))).left

  use lhs_initial, rhs_initial
  exact diff_initial
  -- Alternatively prove it for after rearrangement values
  -- use lhs_after, rhs_after
  -- exact diff_after

end matchstick_rearrangement_l126_126222


namespace constant_term_binomial_expansion_l126_126256

theorem constant_term_binomial_expansion :
  let C := Nat.choose
  let general_term (r : Nat) := C 8 r * (-1) ^ r * x^(8 - 2 * r)
  (finset.range 9).sum (λ r, (1 + 2 * x^2) * general_term r) = -42 :=
by
  sorry

end constant_term_binomial_expansion_l126_126256


namespace points_cover_unit_area_rectangles_l126_126389

theorem points_cover_unit_area_rectangles :
  ∃ (points : Set (ℝ × ℝ)), 
  (∀ (x y : ℝ), ((x, y) ∈ points) → (0 ≤ x ∧ x ≤ 15 ∧ 0 ≤ y ∧ y ≤ 15)) ∧
  (points.finite ∧ points.to_finset.card = 1965) ∧
  (∀ (R : Set (ℝ × ℝ)), (R.measurable ∧ R.volume = 1 ∧ (∀ p ∈ R, (fst p ≥ 0 ∧ fst p ≤ 15 ∧ snd p ≥ 0 ∧ snd p ≤ 15)) → (∃ p ∈ points, p ∈ R))) :=
sorry

end points_cover_unit_area_rectangles_l126_126389


namespace flowers_sold_at_sale_l126_126329

theorem flowers_sold_at_sale (n : ℕ) (prices : finset ℕ) (radio_price : ℕ) 
  (h_diff : prices.card = n) 
  (h_17th_highest : prices.sort (≥) ! order_of radio_price = 17 - 1) 
  (h_42nd_lowest : prices.sort (≤) ! order_of radio_price = 42 - 1) : 
  n = 58 := 
by 
  sorry

end flowers_sold_at_sale_l126_126329


namespace intersection_M_N_l126_126863

open Set

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | x^2 - 2*x - 3 < 0}
def intersection_sets := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = intersection_sets :=
  sorry

end intersection_M_N_l126_126863


namespace plane_equation_correct_l126_126061

noncomputable def plane_passing_through_point_and_line (A B C D : ℤ) 
(pt : ℝ × ℝ × ℝ) (dir : ℝ × ℝ × ℝ) : Prop :=
  let (px, py, pz) := pt in
  let (dx, dy, dz) := dir in
  (A * px + B * py + C * pz + D = 0) ∧
  (A * dx + B * dy + C * dz = 0)

theorem plane_equation_correct :
  plane_passing_through_point_and_line 2 0 (-5) (-17) (1, 4, -3) (4, -5, 2) :=
by
  sorry

end plane_equation_correct_l126_126061


namespace log2_7_l126_126113

def f : ℝ → ℝ 
| x => if (x < 2) then (2^x) else f (x - 1)

theorem log2_7 : f (log (7) / log 2) = 7 / 2 := by sorry

end log2_7_l126_126113


namespace minutes_after_midnight_l126_126869

theorem minutes_after_midnight :
  let start_date := (2023, 5, 5, 0, 0) -- (Year, Month, Day, Hour, Minute)
  let elapsed_minutes := 1723
  let days := elapsed_minutes / (24 * 60)
  let remaining_minutes := elapsed_minutes % (24 * 60)
  let hours := remaining_minutes / 60
  let minutes := remaining_minutes % 60
  let resulting_date := (start_date.0, start_date.1, start_date.2 + days, hours, minutes)
  resulting_date = (2023, 5, 6, 4, 43) :=
by
  sorry

end minutes_after_midnight_l126_126869


namespace smallest_single_embrasure_length_total_length_embrasures_reliable_system_exits_l126_126733

variable (s : ℝ)

/-- Two guards walk around a circular wall with length 1, one guard walking twice as fast as the other. -/
def guards_system_reliable_length (s : ℝ) : Prop :=
  ∃ l : ℝ, l ≤ s ∧ ∀ (t : ℝ), is_open ({x : ℝ | ∃ y : ℝ, 0 <= y ∧ y < l ∧ (x = (t + y) % 1) ∨ (x = (2*t + y) % 1)})

theorem smallest_single_embrasure_length (s : ℝ) : s ≥ (2 / 3) :=
  sorry

theorem total_length_embrasures (s : ℝ) : 
  (s > 1 / 2) → ∀ (embrasures : set ℝ) (h_embrasures : guards_system_reliable_length embrasures), 
  Σ' s ∈ embrasures, s > 1 / 2 :=
  sorry

theorem reliable_system_exits (s : ℝ) (h_s : s > 1 / 2) :
  ∃ (system : set ℝ), guards_system_reliable_length system ∧ Σ' s ∈ system, s < s :=
  sorry

end smallest_single_embrasure_length_total_length_embrasures_reliable_system_exits_l126_126733


namespace find_a1_q_find_Tn_l126_126477

-- Define the geometric sequence a_n and its sum S_n
noncomputable def a_n (n : ℕ) : ℝ := (1/2)^(n-1)
noncomputable def S_n : ℕ → ℝ 
| 0       := 0
| (n + 1) := S_n n + a_n n

-- Define the sequence b_n and its sum T_n
noncomputable def b_n (n : ℕ) : ℝ := n * a_n n
noncomputable def T_n : ℕ → ℝ 
| 0       := 0
| (n + 1) := T_n n + b_n n

-- Problem statement: Part 1
theorem find_a1_q (S_n : ℕ → ℝ) (h : ∀ n, S_n (n+2) = 1/4 * S_n n + 3/2) : a_n 1 = 1 ∧ q = 1/2 :=
sorry

-- Problem statement: Part 2
theorem find_Tn (n : ℕ) (h1 : ∀ n, S_n (n+2) = 1/4 * S_n n + 3/2) (h2 : ∀ n, a_n n = (1/2)^(n-1)) (h3 : ∀ n, b_n n = n * a_n n) :
  T_n n = 4 - (n+2)/(2^(n-1)) :=
sorry

end find_a1_q_find_Tn_l126_126477


namespace ratio_of_area_to_perimeter_of_square_l126_126680

theorem ratio_of_area_to_perimeter_of_square (s : ℕ) (h : s = 10) :
  (s * s : ℚ) / (4 * s) = 5 / 2 :=
by
  have h_perimeter : (4 * s : ℚ) = 40 :=
    by rw [h, Nat.cast_mul, Nat.cast_bit0, Nat.cast_bit0]; norm_num
  have h_area : (s * s : ℚ) = 100 :=
    by rw [h, Nat.cast_mul]; norm_num
  rw [h_area, h_perimeter]; norm_num

end ratio_of_area_to_perimeter_of_square_l126_126680


namespace largest_polygon_has_13_sides_l126_126166

-- Define a convex 13-gon and its properties
def convex_13_gon : Type := sorry

-- Define what it means to draw all diagonals of a convex 13-gon
def draw_all_diagonals (polygon : convex_13_gon) : Prop := sorry

-- Prove that the polygon with the largest number of sides formed by the diagonals has 13 sides
theorem largest_polygon_has_13_sides (polygon : convex_13_gon) (h : draw_all_diagonals polygon) :
  ∃ p : polygon, p.sides = 13 :=
sorry

end largest_polygon_has_13_sides_l126_126166


namespace largest_subset_50_no_4x_relation_l126_126730

theorem largest_subset_50_no_4x_relation:
  ∃ S : set ℕ, S ⊆ {x | 1 ≤ x ∧ x ≤ 50} ∧ (∀ a b, a ∈ S ∧ b ∈ S → ¬ (a = 4 * b ∨ b = 4 * a)) ∧ (card S = 40) :=
sorry

end largest_subset_50_no_4x_relation_l126_126730


namespace problem_complex_range_l126_126109

theorem problem_complex_range (m : ℝ) : 
  let z : ℂ := (1 + complex.i) * (m - 2 * complex.i) in
  (z.re > 0 ∧ z.im > 0) → m > 2 := 
by
  assume h : z.re > 0 ∧ z.im > 0
  sorry

end problem_complex_range_l126_126109


namespace digit_invariant_divisibility_l126_126949

-- Problem statement
theorem digit_invariant_divisibility (n : ℕ) :
  (∀ m : ℕ, (digits ℕ n) = digits ℕ m → (n ∣ m ↔ n = 3 ∨ n = 9)) :=
sorry

end digit_invariant_divisibility_l126_126949


namespace number_of_selected_third_year_students_l126_126362

/-
A school wants to understand the learning situation of its students and decides to use stratified sampling to select 50 students 
from 150 first-year high school students, 120 second-year students, and 180 third-year students for a questionnaire survey.
Given the total number of students and the number of students to be sampled, find the number of third-year students selected.
-/

def first_year_students : Nat := 150
def second_year_students : Nat := 120
def third_year_students : Nat := 180
def total_students_to_sample : Nat := 50
def total_population : Nat := first_year_students + second_year_students + third_year_students

def sample_proportion (stratum_size : Nat) (total_population : Nat) : Rat :=
  stratum_size / (total_population : Rat)

def number_third_year_students_selected : Nat :=
  (sample_proportion third_year_students total_population * total_students_to_sample).toNat

theorem number_of_selected_third_year_students :
  number_third_year_students_selected = 20 := by
  sorry

end number_of_selected_third_year_students_l126_126362


namespace hexagon_area_correct_l126_126777

noncomputable def hexagon_area (a : ℝ) : ℝ :=
  let triangle_area := (sqrt 3 / 4) * (1 + sqrt 3) ^ 2 in
  let squares_area := 3 in
  triangle_area - squares_area

theorem hexagon_area_correct : hexagon_area 1 = 3 + sqrt 3 :=
by
  sorry

end hexagon_area_correct_l126_126777


namespace smallest_a_plus_b_l126_126833

theorem smallest_a_plus_b : ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2^3 * 3^7 * 7^2 = a^b ∧ a + b = 380 :=
sorry

end smallest_a_plus_b_l126_126833


namespace probability_quadrant_l126_126698

theorem probability_quadrant
    (r : ℝ) (x y : ℝ)
    (h : x^2 + y^2 ≤ r^2) :
    (∃ p : ℝ, p = (1 : ℚ)/4) :=
by
  sorry

end probability_quadrant_l126_126698


namespace laps_needed_to_reach_total_distance_l126_126938

-- Define the known conditions
def total_distance : ℕ := 2400
def lap_length : ℕ := 150
def laps_run_each : ℕ := 6
def total_laps_run : ℕ := 2 * laps_run_each

-- Define the proof goal
theorem laps_needed_to_reach_total_distance :
  (total_distance - total_laps_run * lap_length) / lap_length = 4 :=
by
  sorry

end laps_needed_to_reach_total_distance_l126_126938


namespace sum_of_digits_10n_minus_1_m_l126_126837

theorem sum_of_digits_10n_minus_1_m (m n d : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (hd_digits : m.digits 10 = d) (h_d_le_n : d ≤ n) : 
  sum_of_digits ((10^n - 1) * m) = 9 * n := sorry

end sum_of_digits_10n_minus_1_m_l126_126837


namespace positional_relationship_uncertain_l126_126546

variables (l₁ l₂ l₃ l₄ : Type) [l₁.is_linear_space] [l₂.is_linear_space] [l₃.is_linear_space] [l₄.is_linear_space]
            [x₁ : l₁] [x₂ : l₂] [x₃ : l₃] [x₄ : l₄]
            (h₁ : x₁ ⊥ x₂)
            (h₂ : x₂ ⊥ x₃)
            (h₃ : x₃ ⊥ x₄)


theorem positional_relationship_uncertain 
  (h₁ : x₁ ⊥ x₂) 
  (h₂ : x₂ ⊥ x₃)
  (h₃ : x₃ ⊥ x₄) :
  ∃ (r:relation), (r l₁ l₄) = uncertain := sorry

end positional_relationship_uncertain_l126_126546


namespace carter_bakes_more_cakes_l126_126746

/-- Carter usually bakes 6 cheesecakes, 5 muffins, and 8 red velvet cakes in a week.
This week he was able to bake triple the number of each type of cake.
Prove that Carter baked 38 more cakes this week than usual. -/
theorem carter_bakes_more_cakes :
  let regular_cheesecakes := 6
  let regular_muffins := 5
  let regular_red_velvet_cakes := 8
  let regular_total := regular_cheesecakes + regular_muffins + regular_red_velvet_cakes
  let tripled_cheesecakes := 3 * regular_cheesecakes
  let tripled_muffins := 3 * regular_muffins
  let tripled_red_velvet_cakes := 3 * regular_red_velvet_cakes
  let tripled_total := tripled_cheesecakes + tripled_muffins + tripled_red_velvet_cakes
  in tripled_total - regular_total = 38 :=
by
  let regular_cheesecakes := 6
  let regular_muffins := 5
  let regular_red_velvet_cakes := 8
  let regular_total := regular_cheesecakes + regular_muffins + regular_red_velvet_cakes
  let tripled_cheesecakes := 3 * regular_cheesecakes
  let tripled_muffins := 3 * regular_muffins
  let tripled_red_velvet_cakes := 3 * regular_red_velvet_cakes
  let tripled_total := tripled_cheesecakes + tripled_muffins + tripled_red_velvet_cakes
  show tripled_total - regular_total = 38 from sorry

end carter_bakes_more_cakes_l126_126746


namespace infinite_hexagons_exist_l126_126887

theorem infinite_hexagons_exist :
  ∃ (a1 a2 a3 a4 a5 a6 : ℤ), 
  (a1 + a2 + a3 + a4 + a5 + a6 = 20) ∧
  (a1 ≤ a2) ∧ (a1 + a2 ≤ a3) ∧ (a2 + a3 ≤ a4) ∧
  (a3 + a4 ≤ a5) ∧ (a4 + a5 ≤ a6) ∧ (a1 + a2 + a3 + a4 + a5 > a6) :=
sorry

end infinite_hexagons_exist_l126_126887


namespace election_winner_votes_margin_l126_126170

theorem election_winner_votes_margin (V : ℕ) (h1 : 0.62 * V = 744) (h2 : 744 = 0.62 * V) : 744 - (0.38 * V) = 288 :=
by
  sorry

end election_winner_votes_margin_l126_126170


namespace no_nat_with_odd_even_divisors_l126_126192

theorem no_nat_with_odd_even_divisors:
  ¬∃ n : ℕ, 
    (∃ even_divisor_odd_count : ℕ, 
      even_divisor_odd_count % 2 = 1 ∧ 
      even_divisor_odd_count = (d : finset (ℕ)) → d ∣ n ∧ d % 2 = 0 ∧ 
      count {d | d ∣ n ∧ d % 2 = 0} = even_divisor_odd_count) ∧ 
    (∃ odd_divisor_even_count : ℕ, 
      odd_divisor_even_count % 2 = 0 ∧ 
      odd_divisor_even_count = (d : finset (ℕ)) → d ∣ n ∧ d % 2 ≠ 0 ∧ 
      count {d | d ∣ n ∧ d % 2 ≠ 0} = odd_divisor_even_count) := 
begin
  sorry
end

end no_nat_with_odd_even_divisors_l126_126192


namespace square_side_length_l126_126310

theorem square_side_length (area_circle perimeter_square : ℝ) (h1 : area_circle = 100) (h2 : perimeter_square = area_circle) :
  side_length_square perimeter_square = 25 :=
by
  let s := 25 -- The length of one side of the square is 25
  sorry

def side_length_square (perimeter_square : ℝ) : ℝ :=
  perimeter_square / 4

end square_side_length_l126_126310


namespace new_person_weight_l126_126254

noncomputable def weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

theorem new_person_weight 
  (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) 
  (weight_eqn : weight_increase n avg_increase = new_weight - old_weight) : 
  new_weight = 87.5 :=
by
  have n := 9
  have avg_increase := 2.5
  have old_weight := 65
  have weight_increase := 9 * 2.5
  have weight_eqn := weight_increase = 87.5 - 65
  sorry

end new_person_weight_l126_126254


namespace four_digit_number_difference_l126_126321

theorem four_digit_number_difference :
  (∃ (a b c d e f : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ -∃ g h, 
  g = 1036 ∧ h = 1037 ∧  (({a, b, c, d} = {1, 0, 3, 6} ∧ {e, f} = {5, 7} ∧ 
  (a + 10*b + 100*c + 1000*d = g) ∧ (a + 10*b + 100*c + 1000*d + 1 = h) ∧ 
  (h-g) = 1)))) := sorry

end four_digit_number_difference_l126_126321


namespace seaweed_for_livestock_l126_126745

theorem seaweed_for_livestock (h : 500 : ℕ) (f : 0.40 * h) (m : 0.20 * h) (c : 0.30 * (0.40 * h) = a) (l : 0.70 * (0.40 * h) = b) : b = 140 := 
by
  let total_seaweed := 500
  let for_fire := 0.40 * total_seaweed
  let for_medicine := 0.20 * total_seaweed
  let total_for_consumption_and_feed := 0.40 * total_seaweed
  let for_human_consumption := 0.30 * total_for_consumption_and_feed
  let for_livestock_feed := 0.70 * total_for_consumption_and_feed
  have livestock_feed_pounds : for_livestock_feed = 140, sorry
  exact livestock_feed_pounds

end seaweed_for_livestock_l126_126745


namespace Carter_baked_more_cakes_l126_126753

/--
Carter usually bakes 6 cheesecakes, 5 muffins, and 8 red velvet cakes regularly for a week.
For this week he was able to bake triple the number of cheesecakes, muffins, and red velvet cakes.
Prove that Carter was able to bake 38 more cakes this week than he usually does.
-/
theorem Carter_baked_more_cakes :
    let cheesecakes_usual := 6
    let muffins_usual := 5
    let red_velvet_usual := 8
    let total_usual := cheesecakes_usual + muffins_usual + red_velvet_usual
    let cheesecakes_this_week := 3 * cheesecakes_usual
    let muffins_this_week := 3 * muffins_usual
    let red_velvet_this_week := 3 * red_velvet_usual
    let total_this_week := cheesecakes_this_week + muffins_this_week + red_velvet_this_week
    total_this_week - total_usual = 38 :=
by
    let cheesecakes_usual := 6
    let muffins_usual := 5
    let red_velvet_usual := 8
    let total_usual := cheesecakes_usual + muffins_usual + red_velvet_usual
    let cheesecakes_this_week := 3 * cheesecakes_usual
    let muffins_this_week := 3 * muffins_usual
    let red_velvet_this_week := 3 * red_velvet_usual
    let total_this_week := cheesecakes_this_week + muffins_this_week + red_velvet_this_week
    show total_this_week - total_usual = 38 from sorry

end Carter_baked_more_cakes_l126_126753


namespace arrange_points_in_square_l126_126388

-- Define the square and the condition on the points
theorem arrange_points_in_square :
  ∃ (points : set (ℝ × ℝ))
    (N : ℕ),
    (N = 1965) ∧
    (∀ rect_corner_x rect_corner_y,
       0 ≤ rect_corner_x ∧ rect_corner_x ≤ 14 ∧
       0 ≤ rect_corner_y ∧ rect_corner_y ≤ 14 ∧
       (∀ x y, rect_corner_x ≤ x ∧ x < rect_corner_x + 1 ∧ rect_corner_y ≤ y ∧ y < rect_corner_y + 1 →
               (x, y) ∈ points) →
       (∃ p ∈ points, 
         rect_corner_x ≤ p.1 ∧ p.1 < rect_corner_x + 1 ∧ 
         rect_corner_y ≤ p.2 ∧ p.2 < rect_corner_y + 1)) :=
sorry

end arrange_points_in_square_l126_126388


namespace carter_bakes_more_cakes_l126_126748

/-- Carter usually bakes 6 cheesecakes, 5 muffins, and 8 red velvet cakes in a week.
This week he was able to bake triple the number of each type of cake.
Prove that Carter baked 38 more cakes this week than usual. -/
theorem carter_bakes_more_cakes :
  let regular_cheesecakes := 6
  let regular_muffins := 5
  let regular_red_velvet_cakes := 8
  let regular_total := regular_cheesecakes + regular_muffins + regular_red_velvet_cakes
  let tripled_cheesecakes := 3 * regular_cheesecakes
  let tripled_muffins := 3 * regular_muffins
  let tripled_red_velvet_cakes := 3 * regular_red_velvet_cakes
  let tripled_total := tripled_cheesecakes + tripled_muffins + tripled_red_velvet_cakes
  in tripled_total - regular_total = 38 :=
by
  let regular_cheesecakes := 6
  let regular_muffins := 5
  let regular_red_velvet_cakes := 8
  let regular_total := regular_cheesecakes + regular_muffins + regular_red_velvet_cakes
  let tripled_cheesecakes := 3 * regular_cheesecakes
  let tripled_muffins := 3 * regular_muffins
  let tripled_red_velvet_cakes := 3 * regular_red_velvet_cakes
  let tripled_total := tripled_cheesecakes + tripled_muffins + tripled_red_velvet_cakes
  show tripled_total - regular_total = 38 from sorry

end carter_bakes_more_cakes_l126_126748


namespace vector_CD_l126_126157

variable {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {A B C D : V}
variables (a c : V)

-- Introduce the assumptions
axiom midpoint_D : D = (A + B) / 2
axiom vector_BC : B - C = a
axiom vector_BA : -. B + A = c

theorem vector_CD (midpoint_D : D = (A + B) / 2) (vector_BC : B - C = a) (vector_BA : A - B = c) :
  D - C = -a + (1 / 2) * c :=
sorry

end vector_CD_l126_126157


namespace sequence_nth_and_sum_l126_126053

theorem sequence_nth_and_sum (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 3 ∧
  S 2 = 9 ∧
  S 3 = 22 ∧
  (∀ n, a (n + 1) = S (n + 1) - S n) ∧
  (∀ n, S n = ∑ k in finset.range (n + 1), a (k + 1)) →
  (∀ n, a n = 2 * n ^ 2 - n + 2) ∧
  (∀ n, S n = n * (2 * n + 1) * (2 * n + 3) / 3) := by
  sorry

end sequence_nth_and_sum_l126_126053


namespace probability_of_losing_game_l126_126153

theorem probability_of_losing_game (wins losses : ℕ) (ht : wins = 5) (hl : losses = 6) :
  let total := wins + losses in
  total = 11 → 
  (losses : ℚ) / total = 6 / 11 :=
by
  intros total htotal
  rw [ht, hl]
  sorry

end probability_of_losing_game_l126_126153


namespace ratio_of_roots_ratio_l126_126067

noncomputable def sum_roots_first_eq (a b c : ℝ) := b / a
noncomputable def product_roots_first_eq (a b c : ℝ) := c / a
noncomputable def sum_roots_second_eq (a b c : ℝ) := a / c
noncomputable def product_roots_second_eq (a b c : ℝ) := b / c

theorem ratio_of_roots_ratio (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (h3 : (b ^ 2 - 4 * a * c) > 0)
  (h4 : (a ^ 2 - 4 * c * b) > 0)
  (h5 : sum_roots_first_eq a b c ≥ 0)
  (h6 : product_roots_first_eq a b c = 9 * sum_roots_second_eq a b c) :
  sum_roots_first_eq a b c / product_roots_second_eq a b c = -3 :=
sorry

end ratio_of_roots_ratio_l126_126067


namespace distance_OP_l126_126247

-- Define the variables and constants
variables {a : ℝ} [Fact (0 < a)]

-- Define the points based on the given distances
def O : ℝ := 0
def A : ℝ := 2 * a
def B : ℝ := 3 * a
def C : ℝ := 5 * a
def D : ℝ := 7 * a
def E : ℝ := 8 * a

-- Define point P located at 2/3 of the way from B to C
def P : ℝ := B + (2 / 3) * (C - B)

-- Translate the conditions to Lean definitions
def AP := |A - P|
def PE := |P - E|
def BP := |B - P|
def PC := |P - C|

-- Statement to prove the distance OP
theorem distance_OP : P = 4.33 * a :=
by
  -- The proof is not required as per the instructions
  sorry

end distance_OP_l126_126247


namespace equilateral_ABC_l126_126212

theorem equilateral_ABC
  (A B C D P : Type)
  [ConvexQuadrilateral A B C D]
  (h1 : length AD = length BC)
  (h2 : angle A + angle B = 120)
  (h3 : separates_line CD A P)
  (h4 : equilateral_triangle D C P)
  :
  equilateral_triangle A B P := 
begin
  sorry
end

end equilateral_ABC_l126_126212


namespace total_area_to_be_painted_l126_126002

theorem total_area_to_be_painted (length width height partition_length partition_height : ℝ) 
(partition_along_length inside_outside both_sides : Bool)
(h1 : length = 15)
(h2 : width = 12)
(h3 : height = 6)
(h4 : partition_length = 15)
(h5 : partition_height = 6) 
(h_partition_along_length : partition_along_length = true)
(h_inside_outside : inside_outside = true)
(h_both_sides : both_sides = true) :
    let end_wall_area := 2 * 2 * width * height
    let side_wall_area := 2 * 2 * length * height
    let ceiling_area := length * width
    let partition_area := 2 * partition_length * partition_height
    (end_wall_area + side_wall_area + ceiling_area + partition_area) = 1008 :=
by
    sorry

end total_area_to_be_painted_l126_126002


namespace compute_fraction_l126_126039

theorem compute_fraction : (20 * 2 + 10) / (5 + 3 - 1) = 50 / 7 :=
by
  calc
    (20 * 2 + 10) / (5 + 3 - 1) = 50 / 7 : by sorry

end compute_fraction_l126_126039


namespace count_true_statements_l126_126629

def reciprocal (n : ℕ) : ℝ := 1 / n

def statement_i := reciprocal 4 + reciprocal 8 = reciprocal 12
def statement_ii := reciprocal 9 - reciprocal 3 = reciprocal 6
def statement_iii := reciprocal 3 * reciprocal 9 = reciprocal 27
def statement_iv := reciprocal 15 / reciprocal 3 = reciprocal 5

theorem count_true_statements : (¬ statement_i ∧ ¬ statement_ii ∧ statement_iii ∧ statement_iv) → (2 = 2) :=
by
  intros
  sorry

end count_true_statements_l126_126629


namespace polygon_interior_angle_l126_126879

theorem polygon_interior_angle (n : ℕ) (h1 : ∀ (i : ℕ), i < n → (n - 2) * 180 / n = 140): n = 9 := 
sorry

end polygon_interior_angle_l126_126879


namespace trapezoid_property_l126_126946

variables (A B C D E O P : Type)
variables [line_segment A B] [line_segment B C] [line_segment C D] [line_segment D A]
          [line_segment A E] [line_segment B D] [line_segment C A] [line_segment C E] 
          [line_segment E D] [line_segment B O] [line_segment P D] 

variables (AD BC: ℝ)

-- conditions
axiom trapezoid_ABCD : is_trapezoid A B C D
axiom point_E_on_AD : is_on_line_segment E A D
axiom AE_eq_BC : AE = BC
axiom intersects_CE_BD_at_P : intersects C E B D P
axiom intersects_CA_BD_at_O : intersects C A B D O
axiom BO_eq_PD : BO = PD

-- question
theorem trapezoid_property : AD^2 = BC^2 + AD * BC :=
sorry

end trapezoid_property_l126_126946


namespace seating_arrangements_l126_126557

-- Define the specific people
def Alice := 0
def Bob := 1
def Cindy := 2
def Dave := 3

-- Define the main theorem statement
theorem seating_arrangements : 
  let total_arrangements := Nat.factorial 10,
      consecutive_arrangements := (Nat.factorial 7) * (Nat.factorial 4),
      non_consecutive_arrangements := total_arrangements - consecutive_arrangements
  in
  non_consecutive_arrangements = 3507840 :=
by
  sorry

end seating_arrangements_l126_126557


namespace f_diff_formula_l126_126924

noncomputable def f (n : ℕ) : ℚ := ∑ i in finset.range (2 * n + 1), if n ≤ i then 1 / (i + 1 : ℚ) else 0

theorem f_diff_formula (n : ℕ) : n > 0 → 
  f (n + 1) - f n = (1 / (2 * n + 1) - 1 / (2 * n + 2) : ℚ) :=
by sorry

end f_diff_formula_l126_126924


namespace person_B_D_coins_l126_126905

theorem person_B_D_coins
  (a d : ℤ)
  (h1 : a - 3 * d = 58)
  (h2 : a - 2 * d = 58)
  (h3 : a + d = 60)
  (h4 : a + 2 * d = 60)
  (h5 : a + 3 * d = 60) :
  (a - 2 * d = 28) ∧ (a = 24) :=
by
  sorry

end person_B_D_coins_l126_126905


namespace remainder_of_11th_num_in_sequence_is_3_l126_126734

theorem remainder_of_11th_num_in_sequence_is_3
  (a : ℕ) (h : ∀ k : ℕ, k < 50 → (if k < 7 then k ≡ (a + k) % 7 else 
   if k < 14 then k - 7 ≡ (a + k) % 7 + 1 else 
   if k < 21 then k - 14 ≡ (a + k) % 7 + 2 else 
   if k < 28 then k - 21 ≡ (a + k) % 7 + 3 else 
   if k < 35 then k - 28 ≡ (a + k) % 7 + 4 else 
   if k < 42 then k - 35 ≡ (a + k) % 7 + 5 else 
   if k < 50 then k - 42 ≡ (a + k) % 7 + 6 else 0) 
   = 7) 
  (eight_div_by_seven : ∃ (S : finset ℕ), S.card = 8 ∧ ∀ x ∈ S, x % 7 = 0) :
  (a + 10) % 7 = 3 :=
by sorry

end remainder_of_11th_num_in_sequence_is_3_l126_126734


namespace abs_inequality_l126_126090

theorem abs_inequality (x y : ℝ) (h1 : |x| < 2) (h2 : |y| < 2) : |4 - x * y| > 2 * |x - y| :=
by
  sorry

end abs_inequality_l126_126090


namespace area_ABCD_twice_area_KLMN_l126_126948

noncomputable def midpoint (A B : Point) : Point := sorry

/-- Midpoints of sides AB and CD, respectively -/
def K := midpoint A B
def M := midpoint C D

/-- Points on sides BC and AD forming rectangle KLMN -/
def L : Point := sorry -- precise definition skipped for the example
def N : Point := sorry -- precise definition skipped for the example

/-- Definition of area function -/
noncomputable def area (P : Quadrilateral) : Real := sorry

/-- Given quadrilateral ABCD -/
structure Quadrilateral :=
(A B C D : Point)

/-- Rectangle defined by midpoints and other given points -/
structure Rectangle :=
(K L M N : Point)

/-- Main theorem stating the relationship between the areas of the quadrilateral and the rectangle -/
theorem area_ABCD_twice_area_KLMN (ABCD : Quadrilateral) (KLMN : Rectangle)
  (hK : K = midpoint ABCD.A ABCD.B) 
  (hM : M = midpoint ABCD.C ABCD.D) 
  (hKL : ∃ A B C D : Point, L ∈ Line ABCD.B ABCD.C ∧ N ∈ Line ABCD.A ABCD.D) 
  (h_rect : is_rectangle KLMN.K KLMN.L KLMN.M KLMN.N) :
  area ABCD = 2 * area KLMN := sorry

end area_ABCD_twice_area_KLMN_l126_126948


namespace partition_sum_equal_l126_126794

theorem partition_sum_equal (k : ℕ) (hk1 : k ≥ 92) (hk2 : k % 4 = 0 ∨ k % 4 = 3) :
  ∃ (A B : Finset ℕ), Disjoint A B ∧ A ∪ B = Finset.range (k + 1) \ 0 + 1990 ∧ A.sum id = B.sum id := 
sorry

end partition_sum_equal_l126_126794


namespace time_for_tap_b_l126_126358

variable (total_time : ℝ) (A_rate B_rate : ℝ)

-- The total time to drain the pool when both taps are open for 30 minutes.
axiom combined_draining_time : total_time = 30

-- The combined rate of flow when both taps are open.
axiom combined_rate : A_rate + B_rate = 1 / total_time

-- Draining work performed in 10 minutes by both taps.
axiom initial_draining_work : 10 * (A_rate + B_rate) = 1 / 3

-- Remaining work to be done by tap B alone.
axiom remaining_work (initial_draining_work = 1 / 3) : 2 / 3 = B_rate * 30

-- Prove that the time taken by tap B alone to drain the full pool is 45 minutes.
theorem time_for_tap_b : 1 / B_rate = 45 :=
  sorry

end time_for_tap_b_l126_126358


namespace final_amount_in_account_l126_126672

noncomputable def initial_deposit : ℝ := 1000
noncomputable def first_year_interest_rate : ℝ := 0.2
noncomputable def first_year_balance : ℝ := initial_deposit * (1 + first_year_interest_rate)
noncomputable def withdrawal_amount : ℝ := first_year_balance / 2
noncomputable def after_withdrawal_balance : ℝ := first_year_balance - withdrawal_amount
noncomputable def second_year_interest_rate : ℝ := 0.15
noncomputable def final_balance : ℝ := after_withdrawal_balance * (1 + second_year_interest_rate)

theorem final_amount_in_account : final_balance = 690 := by
  sorry

end final_amount_in_account_l126_126672


namespace find_value_of_a_l126_126159

-- Define the setting for triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : b^2 - c^2 + 2 * a = 0)
variables (h2 : Real.tan C / Real.tan B = 3)

-- Given conditions and conclusion for the proof problem
theorem find_value_of_a 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) : 
  a = 4 := 
sorry

end find_value_of_a_l126_126159


namespace boots_ratio_l126_126941

noncomputable def problem_statement : Prop :=
  let total_money : ℝ := 50
  let cost_toilet_paper : ℝ := 12
  let cost_groceries : ℝ := 2 * cost_toilet_paper
  let remaining_after_groceries : ℝ := total_money - cost_toilet_paper - cost_groceries
  let extra_money_per_person : ℝ := 35
  let total_extra_money : ℝ := 2 * extra_money_per_person
  let total_cost_boots : ℝ := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots : ℝ := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3

theorem boots_ratio (total_money : ℝ) (cost_toilet_paper : ℝ) (extra_money_per_person : ℝ) : 
  let cost_groceries := 2 * cost_toilet_paper
  let remaining_after_groceries := total_money - cost_toilet_paper - cost_groceries
  let total_extra_money := 2 * extra_money_per_person
  let total_cost_boots := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3 :=
by
  sorry

end boots_ratio_l126_126941


namespace value_of_60th_number_l126_126273

-- Define the sequence arrangement
def sequence (n : ℕ) : ℕ := 2 * n

-- Define the cumulative number of elements in rows up to nth row
def cumulative (n : ℕ) : ℕ := (n * (n + 1))

theorem value_of_60th_number : 
  ∃ k m, cumulative k < 60 ∧ 60 ≤ cumulative (k + 1) ∧ (sequence (k + 1) = 16) := 
by
  sorry

end value_of_60th_number_l126_126273


namespace combined_solid_sum_faces_edges_vertices_l126_126019

noncomputable def prism_faces : ℕ := 6
noncomputable def prism_edges : ℕ := 12
noncomputable def prism_vertices : ℕ := 8
noncomputable def new_pyramid_faces : ℕ := 4
noncomputable def new_pyramid_edges : ℕ := 4
noncomputable def new_pyramid_vertex : ℕ := 1

theorem combined_solid_sum_faces_edges_vertices :
  prism_faces - 1 + new_pyramid_faces + prism_edges + new_pyramid_edges + prism_vertices + new_pyramid_vertex = 34 :=
by
  -- proof would go here
  sorry

end combined_solid_sum_faces_edges_vertices_l126_126019


namespace square_side_length_l126_126309

theorem square_side_length (area_circle perimeter_square : ℝ) (h1 : area_circle = 100) (h2 : perimeter_square = area_circle) :
  side_length_square perimeter_square = 25 :=
by
  let s := 25 -- The length of one side of the square is 25
  sorry

def side_length_square (perimeter_square : ℝ) : ℝ :=
  perimeter_square / 4

end square_side_length_l126_126309


namespace football_defeat_points_l126_126893

theorem football_defeat_points (V D F : ℕ) (x : ℕ) :
    3 * V + D + x * F = 8 →
    27 + 6 * x = 32 →
    x = 0 :=
by
    intros h1 h2
    sorry

end football_defeat_points_l126_126893


namespace angle_RSP_l126_126562

theorem angle_RSP (PQ PR RS PS : Type) 
  (P Q R S : PQ) 
  (bisects_SQR : QP bisects ∠(SQR : angle)) 
  (bisects_QRP : PS bisects ∠(QRP : angle)) 
  (PQ_eq_PR : PQ = PR)
  (angle_RSQ_eq : (RSQ : angle) = 2 * x)
  (angle_RQP_eq : (RQP : angle) = 4 * x) :
  angle RSP = 2 * x :=
sorry

end angle_RSP_l126_126562


namespace carter_extra_cakes_l126_126749

def regular_cakes : ℕ := 6 + 5 + 8
def triple_cakes : ℕ := 3 * 6 + 3 * 5 + 3 * 8
def extra_cakes : ℕ := triple_cakes - regular_cakes

theorem carter_extra_cakes : extra_cakes = 38 :=
by
  unfold regular_cakes triple_cakes extra_cakes
  calc
    3 * 6 + 3 * 5 + 3 * 8 - (6 + 5 + 8)
      = 57 - 19 : by norm_num
    ... = 38 : by norm_num

end carter_extra_cakes_l126_126749


namespace sin_two_alpha_l126_126818

theorem sin_two_alpha (alpha : ℝ) (h : Real.cos (π / 4 - alpha) = 4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_two_alpha_l126_126818


namespace cyclic_quadrilateral_area_correct_l126_126795

noncomputable def cyclic_quadrilateral_area 
  (d : ℝ) (θ1 θ2 : ℝ) (m1 m2 : ℝ) : ℝ :=
  let h1 := m1 * real.sin θ1 in
  let h2 := m2 * real.sin θ2 in
  0.5 * d * h1 + 0.5 * d * h2

theorem cyclic_quadrilateral_area_correct :
  cyclic_quadrilateral_area 42 (real.pi / 6) (real.pi / 4) 15 20 ≈ 454.44 := sorry

end cyclic_quadrilateral_area_correct_l126_126795


namespace phraseCompletion_l126_126702

-- Define the condition for the problem
def isCorrectPhrase (phrase : String) : Prop :=
  phrase = "crying"

-- State the theorem to be proven
theorem phraseCompletion : ∃ phrase, isCorrectPhrase phrase :=
by
  use "crying"
  sorry

end phraseCompletion_l126_126702


namespace PQR_meet_same_time_l126_126947

def lcm (a b : Nat) : Nat := sorry -- Defining lcm function placeholder

def lcm_three (a b c : Nat) : Nat := lcm (lcm a b) c  -- Defining lcm of three numbers

theorem PQR_meet_same_time :
  lcm_three 252 198 315 = 13860 :=
by
  sorry

end PQR_meet_same_time_l126_126947


namespace guides_and_tourists_groupings_l126_126668

open Nat

/-- Three tour guides are leading eight tourists. Each tourist must choose one of the guides, 
    but with the stipulation that each guide must take at least one tourist. Prove 
    that the number of different groupings of guides and tourists is 5796. -/
theorem guides_and_tourists_groupings : 
  let total_groupings := 3 ^ 8,
      at_least_one_no_tourists := binom 3 1 * 2 ^ 8,
      at_least_two_no_tourists := binom 3 2 * 1 ^ 8,
      total_valid_groupings := total_groupings - at_least_one_no_tourists + at_least_two_no_tourists
  in total_valid_groupings = 5796 :=
by
  sorry

end guides_and_tourists_groupings_l126_126668


namespace three_more_than_seven_in_pages_l126_126551

theorem three_more_than_seven_in_pages : 
  ∀ (pages : List Nat), (∀ n, n ∈ pages → 1 ≤ n ∧ n ≤ 530) ∧ (List.length pages = 530) →
  ((List.count 3 (pages.bind (λ n => Nat.digits 10 n))) - (List.count 7 (pages.bind (λ n => Nat.digits 10 n)))) = 100 :=
by
  intros pages h
  sorry

end three_more_than_seven_in_pages_l126_126551


namespace abs_z_eq_2_sqrt_5_l126_126480

def i : ℂ := complex.I
def z : ℂ := (1 + i) * (1 + 3 * i)

theorem abs_z_eq_2_sqrt_5 : complex.abs z = 2 * Real.sqrt 5 :=
by sorry

end abs_z_eq_2_sqrt_5_l126_126480


namespace selling_price_correct_l126_126015

-- Conditions
def gain : ℝ := 45
def gain_percentage : ℝ := 0.3

-- Define what we need to prove: the selling price (SP) of the article
def cost_price (gain : ℝ) (percentage : ℝ) := gain / percentage
def selling_price (cost : ℝ) (gain : ℝ) := cost + gain

theorem selling_price_correct : selling_price (cost_price gain gain_percentage) gain = 195 := by
  sorry

end selling_price_correct_l126_126015


namespace simplify_fraction_l126_126620

theorem simplify_fraction (n : ℤ) : 
  (3^(n + 4) - 3 * 3^n) / (3 * 3^(n + 3)) = 26 / 27 :=
by 
  sorry

end simplify_fraction_l126_126620


namespace net_rate_of_pay_equals_39_dollars_per_hour_l126_126348

-- Definitions of the conditions
def hours_travelled : ℕ := 3
def speed_per_hour : ℕ := 60
def car_consumption_rate : ℕ := 30
def earnings_per_mile : ℕ := 75  -- expressing $0.75 as 75 cents to avoid floating-point
def gasoline_cost_per_gallon : ℕ := 300  -- expressing $3.00 as 300 cents to avoid floating-point

-- Proof statement
theorem net_rate_of_pay_equals_39_dollars_per_hour : 
  (earnings_per_mile * (speed_per_hour * hours_travelled) - gasoline_cost_per_gallon * ((speed_per_hour * hours_travelled) / car_consumption_rate)) / hours_travelled = 3900 := 
by 
  -- The statement below essentially expresses 39 dollars per hour in cents (i.e., 3900 cents per hour).
  sorry

end net_rate_of_pay_equals_39_dollars_per_hour_l126_126348


namespace tan_of_13pi_over_6_l126_126420

theorem tan_of_13pi_over_6 : Real.tan (13 * Real.pi / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_of_13pi_over_6_l126_126420


namespace salary_for_may_l126_126983

theorem salary_for_may (J F M A May : ℝ) 
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8200)
  (h3 : J = 5700) : 
  May = 6500 :=
by 
  have eq1 : J + F + M + A = 32000 := by
    linarith
  have eq2 : F + M + A + May = 32800 := by
    linarith
  have eq3 : May - J = 800 := by
    linarith [eq1, eq2]
  have eq4 : May = 6500 := by
    linarith [eq3, h3]
  exact eq4

end salary_for_may_l126_126983


namespace a_10_value_l126_126654

-- Define the sequence {a_n} recursively
def a : ℕ → ℕ 
| 1       := 2
| (n + 1) := 4 * a n - 3

-- Assert the value of a_10
theorem a_10_value : a 10 = 2^18 + 1 := 
by {
  sorry
}

end a_10_value_l126_126654


namespace log_sum_of_squares_l126_126048

theorem log_sum_of_squares :
  let log_5 := Real.log 5 / Real.log 10
  let log_2 := Real.log 2 / Real.log 10 in
  log_5^2 + log_2 * log_5 + log_2 = 1 :=
by
  sorry

end log_sum_of_squares_l126_126048


namespace total_seashells_l126_126197

def joans_seashells : Nat := 6
def jessicas_seashells : Nat := 8

theorem total_seashells : joans_seashells + jessicas_seashells = 14 :=
by
  sorry

end total_seashells_l126_126197


namespace find_a_for_geom_sequence_l126_126963

variables (ρ θ a t x y : ℝ) (h1 : a > 0)
def polar_eq : Prop := ρ * sin θ^2 = 2 * a * cos θ

variables (x y : ℝ) (hₓ : x = -2 + (sqrt 2 / 2) * t) (hy : y = -4 + (sqrt 2 / 2) * t)
def parametric_eq : Prop :=
  x = -2 + (sqrt 2 / 2) * t ∧ y = -4 + (sqrt 2 / 2) * t

def rectangular_eq : Prop := y^2 = 2 * a * x
def standard_line_eq : Prop := x - y - 2 = 0

def geometric_sequence (M N P : ℝ) : Prop :=
  abs M = abs t ∧ abs N = abs t ∧ abs (M - N) = abs t ∧
  (abs (M - N)) ^ 2 = abs (M * N)

theorem find_a_for_geom_sequence
(h1 : a > 0) (h2 : polar_eq ρ θ a)
(h3 : parametric_eq x y)
(h4 : rectangular_eq y x a)
(h5 : standard_line_eq x y)
: a = 1 := by sorry

end find_a_for_geom_sequence_l126_126963


namespace range_of_a_l126_126491

def f (m : ℝ) (x : ℝ) : ℝ := 3 * m * x - 1 / x - (3 + m) * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ (m : ℝ) (x1 x2 : ℝ),
      (4 < m) → (m < 5) →
      (1 ≤ x1) → (x1 ≤ 3) →
      (1 ≤ x2) → (x2 ≤ 3) →
      (a - Real.log 3) * m - 3 * Real.log 3 > |f m x1 - f m x2|) →
  a ≥ 37 / 6 :=
by
  sorry

end range_of_a_l126_126491


namespace problem_statement_l126_126401

def is_monotonically_increasing (f : ℝ → ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f x ≤ f y

theorem problem_statement :
  is_monotonically_increasing (λ x : ℝ, x^3) ∧ 
  ¬ is_monotonically_increasing (λ x : ℝ, x^2) ∧ 
  ¬ is_monotonically_increasing (λ x : ℝ, Real.log x) ∧ 
  ¬ is_monotonically_increasing (λ x : ℝ, 0^x) :=
sorry

end problem_statement_l126_126401


namespace calculate_M_minus_m_l126_126066

def total_students : ℕ := 2001
def students_studying_spanish (S : ℕ) : Prop := 1601 ≤ S ∧ S ≤ 1700
def students_studying_french (F : ℕ) : Prop := 601 ≤ F ∧ F ≤ 800
def studying_both_languages_lower_bound (S F m : ℕ) : Prop := S + F - m = total_students
def studying_both_languages_upper_bound (S F M : ℕ) : Prop := S + F - M = total_students

theorem calculate_M_minus_m :
  ∀ (S F m M : ℕ),
    students_studying_spanish S →
    students_studying_french F →
    studying_both_languages_lower_bound S F m →
    studying_both_languages_upper_bound S F M →
    S = 1601 ∨ S = 1700 →
    F = 601 ∨ F = 800 →
    M - m = 298 :=
by
  intros S F m M hs hf hl hb Hs Hf
  sorry

end calculate_M_minus_m_l126_126066


namespace number_of_incorrect_propositions_is_1_l126_126864

-- Definitions for lines and planes with given properties
variables (α β : Plane) (m n : Line)

-- Conditions (the given propositions)
def proposition1 : Prop := (m ∥ n) ∧ (m ⊥ α) → (n ⊥ α)
def proposition2 : Prop := (m ⊥ α) ∧ (m ⊥ β) → (α ∥ β)
def proposition3 : Prop := (m ⊥ α) ∧ (m ∥ n) ∧ (n ⊂ β) → (α ⊥ β)
def proposition4 : Prop := (m ∥ α) ∧ (α ∩ β = n) → (m ∥ n)

-- The main theorem to check the number of incorrect propositions.
theorem number_of_incorrect_propositions_is_1 :
  (¬ proposition1 α β m n) ∧ (¬ proposition2 α β m n) ∧ (¬ proposition3 α β m n) ∧ (¬ proposition4 α β m n) → 
  (∃ k, k = 1 ∧ k = (nat.eq_zero_of_le_zero $ ∑ i, if i = 1 then 1 else 0)) :=
by sorry

end number_of_incorrect_propositions_is_1_l126_126864


namespace HCl_yield_l126_126404

noncomputable def total_moles_HCl (moles_C2H6 moles_Cl2 yield1 yield2 : ℝ) : ℝ :=
  let theoretical_yield1 := if moles_C2H6 ≤ moles_Cl2 then moles_C2H6 else moles_Cl2
  let actual_yield1 := theoretical_yield1 * yield1
  let theoretical_yield2 := actual_yield1
  let actual_yield2 := theoretical_yield2 * yield2
  actual_yield1 + actual_yield2

theorem HCl_yield (moles_C2H6 moles_Cl2 : ℝ) (yield1 yield2 : ℝ) :
  moles_C2H6 = 3 → moles_Cl2 = 3 → yield1 = 0.85 → yield2 = 0.70 →
  total_moles_HCl moles_C2H6 moles_Cl2 yield1 yield2 = 4.335 :=
by
  intros h1 h2 h3 h4
  simp [total_moles_HCl, h1, h2, h3, h4]
  sorry

end HCl_yield_l126_126404


namespace largest_spherical_radius_on_torus_l126_126759

theorem largest_spherical_radius_on_torus :
  ∃ (r : ℝ), 
    r = 4 ∧
    (∃ (inner_radius outer_radius : ℝ), 
      inner_radius = 3 ∧ 
      outer_radius = 5 ∧
      (let O := (0, 0, r) in
      let P := (4, 0, 1) in
      let torus_center_circle_radius := 1 in
      let horizontal_distance := 4 in
      let vertical_distance := r - 1 in
      let hypotenuse := r + 1 in
      hypotenuse^2 = horizontal_distance^2 + vertical_distance^2)) := sorry

end largest_spherical_radius_on_torus_l126_126759


namespace sara_savings_plan_is_correct_l126_126243

variable (S : ℕ)

theorem sara_savings_plan_is_correct (h1 : ∃ (S : ℕ), 4100 + 820 * S = 12300) : S = 10 := by
  have h2 : 4100 + 820 * S = 12300 := exists.elim h1 (λ S h, h)
  calc
    4100 + 820 * S = 12300   : h2
                ... = 4100 + 820 * 10 : by norm_num
  sorry

end sara_savings_plan_is_correct_l126_126243


namespace floor_eq_solution_l126_126071

noncomputable def floor_eq_intervals (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1⌋ = ⌊x + 3⌋

theorem floor_eq_solution :
  ∀ x : ℝ, (x ∈ set.Ico (5 / 3) 2 ∨ x ∈ set.Ico 2 3) ↔ floor_eq_intervals x :=
by
  intro x
  sorry

end floor_eq_solution_l126_126071


namespace probability_twelfth_roll_is_last_l126_126536

noncomputable def probability_twelfth_roll_last : ℝ :=
  (7/8 : ℝ) ^ 10 * (1/8 : ℝ)

theorem probability_twelfth_roll_is_last : 
  probability_twelfth_roll_last ≈ 0.020 :=
begin
  sorry
end

end probability_twelfth_roll_is_last_l126_126536


namespace curve_C_general_curve_C_polar_distance_between_points_l126_126851

noncomputable section

def parametric_eq (ϕ : ℝ) : ℝ × ℝ := (2 * cos ϕ, 2 + 2 * sin ϕ)

def general_eq (x y : ℝ) : Prop := x^2 + (y-2)^2 = 4

def polar_eq (ρ θ : ℝ) : Prop := ρ = 4 * sin θ

def point_A := (sqrt 3, 3)
def point_B := (-sqrt 3, 1)
def distance_AB := 4

theorem curve_C_general :
  (∀ ϕ : ℝ, general_eq (fst (parametric_eq ϕ)) (snd (parametric_eq ϕ))) :=
by
  sorry

theorem curve_C_polar :
  (∀ θ : ℝ, polar_eq (sqrt (4^2 + (4 * sin θ - 2)^2)) θ) :=
by
  sorry 

theorem distance_between_points :
  dist point_A point_B = distance_AB :=
by
  sorry

end curve_C_general_curve_C_polar_distance_between_points_l126_126851


namespace smallest_n_l126_126081

theorem smallest_n (n : ℤ) : 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 5673 [MOD 16] → n = 9 :=
by
  sorry

end smallest_n_l126_126081


namespace depression_comparative_phrase_l126_126245

def correct_comparative_phrase (phrase : String) : Prop :=
  phrase = "twice as…as"

theorem depression_comparative_phrase :
  correct_comparative_phrase "twice as…as" :=
by
  sorry

end depression_comparative_phrase_l126_126245


namespace alice_wins_chomp_l126_126781

def symmetrical_strategy (n : ℕ) : Prop :=
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), 
  (∀ turn : ℕ × ℕ, 
    strategy turn = 
      if turn = (1,1) then (1,1)
      else if turn.fst = 2 ∧ turn.snd = 2 then (2,2)
      else if turn.fst = 1 then (turn.snd, 1)
      else (1, turn.fst)) 

theorem alice_wins_chomp (n : ℕ) (h : 1 ≤ n) : 
  symmetrical_strategy n := 
sorry

end alice_wins_chomp_l126_126781


namespace triangle_side_lengths_l126_126458

open Real

theorem triangle_side_lengths (a b c : ℕ) (R : ℝ)
    (h1 : a * a + 4 * d * d = 2500)
    (h2 : b * b + 4 * e * e = 2500)
    (h3 : R = 12.5)
    (h4 : (2:ℝ) * d ≤ a)
    (h5 : (2:ℝ) * e ≤ b)
    (h6 : a > b)
    (h7 : a ≠ b)
    (h8 : 2 * R = 25) :
    (a, b, c) = (15, 7, 20) := by
  sorry

end triangle_side_lengths_l126_126458


namespace marcella_pairs_l126_126600

theorem marcella_pairs (pairs_initial : ℕ) (shoes_lost : ℕ) (h1 : pairs_initial = 50) (h2 : shoes_lost = 15) :
  ∃ pairs_left : ℕ, pairs_left = 35 := 
by
  existsi 35
  sorry

end marcella_pairs_l126_126600


namespace find_radius_of_circle_l126_126645

noncomputable def radius_of_circle (len1 len2 distance : ℤ) : ℤ :=
  if len1 = 40 ∧ len2 = 48 ∧ distance = 22 then 25 else 0

theorem find_radius_of_circle (len1 len2 distance : ℤ) 
  (h1 : len1 = 40) (h2 : len2 = 48) (h3 : distance = 22) : radius_of_circle len1 len2 distance = 25 := 
by 
  rw [radius_of_circle, if_pos (and.intro h1 (and.intro h2 h3))]
  exact rfl

end find_radius_of_circle_l126_126645


namespace arithmetic_progression_sum_l126_126807

-- Problem statement: calculate S_1 + S_2 + ... + S_10
theorem arithmetic_progression_sum :
  let S_p (p : ℕ) := 50 * (p + ((p + (50 - 1) * (3 * p)))) / 2 in
  (∑ p in finset.range 10, 3725 * p.succ) = 204875 :=
by
  -- Proof details are omitted
  sorry

end arithmetic_progression_sum_l126_126807


namespace age_of_father_l126_126631

theorem age_of_father (F C : ℕ) 
  (h1 : F = C)
  (h2 : C + 5 * 15 = 2 * (F + 15)) : 
  F = 45 := 
by 
sorry

end age_of_father_l126_126631


namespace no_natural_number_with_odd_even_divisors_l126_126190

theorem no_natural_number_with_odd_even_divisors :
  ∀ N : ℕ, ¬ (∃ odd_count even_count : ℕ, (odd_count % 2 = 0) ∧ (even_count % 2 = 1) ∧
              (odd_count = (N.divisors.filter (λ n, n % 2 = 1)).length) ∧
              (even_count = (N.divisors.filter (λ n, n % 2 = 0)).length)) :=
by
  intros N
  sorry

end no_natural_number_with_odd_even_divisors_l126_126190


namespace numberOfTermsArithmeticSequence_l126_126870

theorem numberOfTermsArithmeticSequence (a1 d l : ℕ) (h1 : a1 = 3) (h2 : d = 4) (h3 : l = 2012) :
  ∃ n : ℕ, 3 + (n - 1) * 4 ≤ 2012 ∧ (n : ℕ) = 502 :=
by {
  sorry
}

end numberOfTermsArithmeticSequence_l126_126870


namespace number_of_incorrect_statements_l126_126644

def statement1 : Prop :=
  ∀ (P : ℝ) (freq : ℝ), ∃ n : ℕ, freq = P ∨ (n → freq → P) -- Formalize the idea of frequency getting closer to probability

def statement2 : Prop :=
  ∀ (A B : Prop), (A ≠ B → A ∨ B) ∧ (A ∧ B → A ≠ B)

def statement3 : Prop :=
  ∀ (X : ℝ), X ∈ set.Icc 0 3 → ℙ(X ≤ 1) = 1 / 3 

def statement4 : Prop :=
  ∀ (students : list ℝ), list.length students = 4 → 
    (∀ (A : ℝ), A ∈ students → ℙ(A is selected among 2 out of 4) = 1 / 2)

theorem number_of_incorrect_statements (s1 s2 s3 s4 : Prop) : (¬s1 ∨ ¬s2 ∨ ¬s3 ∨ ¬s4) = 2 :=
by { sorry }

end number_of_incorrect_statements_l126_126644


namespace mr_bhaskar_tour_duration_l126_126327

theorem mr_bhaskar_tour_duration :
  ∃ d : Nat, 
    (d > 0) ∧ 
    (∃ original_daily_expense new_daily_expense : ℕ,
      original_daily_expense = 360 / d ∧
      new_daily_expense = original_daily_expense - 3 ∧
      360 = new_daily_expense * (d + 4)) ∧
      d = 20 :=
by
  use 20
  -- Here would come the proof steps to verify the conditions and reach the conclusion.
  sorry

end mr_bhaskar_tour_duration_l126_126327


namespace least_possible_integer_discussed_l126_126714
open Nat

theorem least_possible_integer_discussed (N : ℕ) (H : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → k ≠ 8 ∧ k ≠ 9 → k ∣ N) : N = 2329089562800 :=
sorry

end least_possible_integer_discussed_l126_126714


namespace isosceles_triangle_inequality_l126_126973

theorem isosceles_triangle_inequality {A B C : Type*} [DecidableEq A]
  (AB BC : ℝ) (AC : ℝ) (B_angle : ℝ) (h_triangle: is_isosceles_triangle AB BC B_angle) :
  3 * AC > AB ∧ 2 * AC < AB :=
by
  sorry

end isosceles_triangle_inequality_l126_126973


namespace intersection_eq_l126_126662

open Set

noncomputable def A : Set ℝ := {x : ℝ | x^2 > 4}
noncomputable def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

theorem intersection_eq : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end intersection_eq_l126_126662


namespace dot_product_of_vectors_l126_126866

def vec_a : ℝ × ℝ × ℝ := (1, 2, -2)
def vec_b : ℝ × ℝ × ℝ := (1, 0, 2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem dot_product_of_vectors :
  dot_product ((vec_a.1 - vec_b.1, vec_a.2 - vec_b.2, vec_a.3 - vec_b.3))
              ((vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2, vec_a.3 + 2 * vec_b.3)) = -4 :=
by sorry

end dot_product_of_vectors_l126_126866


namespace projection_of_u_l126_126917

noncomputable def n := ⟨ 1, -1, 2 ⟩
noncomputable def v := ⟨ 5, 3, 5 ⟩
noncomputable def v_proj := ⟨ 3, 5, 1 ⟩
noncomputable def u := ⟨ 4, 0, 7 ⟩

theorem projection_of_u (p : ℝ × ℝ × ℝ) :
  let t := ((u.1 - p.1) * n.1 + (u.2 - p.2) * n.2 + (u.3 - p.3) * n.3) / (n.1 * n.1 + n.2 * n.2 + n.3 * n.3)
  in (u.1 - t * n.1, u.2 - t * n.2, u.3 - t * n.3) = (1, 3, 1) :=
begin
  let t := ((u.1 - 1) * 1 + (u.2 - 3) * -1 + (u.3 - 1) * 2) / (1 * 1 + (-1) * -1 + 2 * 2),
  have : (u.1 - t * 1, u.2 - t * -1, u.3 - t * 2) = (1, 3, 1),
  { sorry },
  exact this,
end

end projection_of_u_l126_126917


namespace impossible_all_diamonds_one_dwarf_l126_126415

theorem impossible_all_diamonds_one_dwarf:
  (∀ (distribution: Fin 8 → Fin 4 → List (Fin 8 × ℕ)), -- Condition on distribution function
    (∃ d (D: Fin 8 → ℕ), -- Initial number of diamonds
    (∀ i, D i = 3) → -- Each dwarf initially has 3 diamonds
    (∀ i t, let (left, right) := distribution i t in
    left.2 + right.2 = D i ∧ -- Split correctly
    left.1 = if i = 0 then 7 else i - 1 ∧ -- Left neighbor
    right.1 = if i = 7 then 0 else i + 1)) → -- Right neighbor
  (∀ d', (|{i | d' i = 24}| = 1) → False)) := -- Impossibility of one dwarf having all diamonds
sorry

end impossible_all_diamonds_one_dwarf_l126_126415


namespace gcd_1021_2729_l126_126797

theorem gcd_1021_2729 : Int.gcd 1021 2729 = 1 :=
by
  sorry

end gcd_1021_2729_l126_126797


namespace evaluate_expression_l126_126694

noncomputable def greatest_integer (x : ℝ) : ℤ :=
  int.floor x

theorem evaluate_expression :
  greatest_integer 6.5 * greatest_integer (2 / 3) + greatest_integer 2 * 7.2 + greatest_integer 8.4 - 6.0 = 16.4 :=
by
  sorry

end evaluate_expression_l126_126694


namespace sum_possible_x_values_l126_126762

theorem sum_possible_x_values : 
  let nums := [24, 27, 55, 64] in 
  let mean_is_prime (x : ℕ) := Nat.prime (170 + x) / 5 in
  let median_is_multiple_of_3 (x : ℕ) := ((List.insert x nums).nth 2) % 3 = 0 in
  ∑ x in { x | mean_is_prime x ∧ median_is_multiple_of_3 x }, x = 60 := 
sorry

end sum_possible_x_values_l126_126762


namespace length_of_goods_train_is_280_meters_l126_126719

-- Define the speeds of the trains
def speed_mans_train_kmh : ℕ := 50
def speed_goods_train_kmh : ℕ := 62

-- Define the time taken for the goods train to pass the man in seconds
def passing_time_seconds : ℕ := 9

-- Convert km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℝ := (speed_kmh * 1000) / 3600

-- Compute the relative speed in m/s
def relative_speed_ms : ℝ := kmh_to_ms (speed_mans_train_kmh + speed_goods_train_kmh)

-- Define the length of the goods train calculation
def length_goods_train : ℝ := relative_speed_ms * passing_time_seconds

-- The proof statement
theorem length_of_goods_train_is_280_meters : length_goods_train ≈ 280 := by
  sorry

end length_of_goods_train_is_280_meters_l126_126719


namespace evaluate_polynomial_l126_126779

-- Define the polynomial function
def polynomial (x : ℝ) : ℝ := x^3 + 3 * x^2 - 9 * x - 5

-- Define the condition: x is the positive root of the quadratic equation
def is_positive_root_of_quadratic (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 9 = 0

-- The main theorem stating the polynomial evaluates to 22 given the condition
theorem evaluate_polynomial {x : ℝ} (h : is_positive_root_of_quadratic x) : polynomial x = 22 := 
by 
  sorry

end evaluate_polynomial_l126_126779


namespace lines_parallel_or_parallel_l126_126465

variables {a b c d : ℝ³} -- ℝ³ for representing lines in 3D space

-- Defining perpendicularity and parallelism in ℝ³
def perpend (u v : ℝ³) : Prop := u ⬝ v = 0  -- using dot product for perpendicularity
def parallel (u v : ℝ³) : Prop := ∃ k : ℝ, k ≠ 0 ∧ u = k • v  -- scalar multiple for parallelism

-- Given conditions
variable (h1 : perpend a b)
variable (h2 : perpend b c)
variable (h3 : perpend c d)
variable (h4 : perpend d a)

-- Theorem to be proved
theorem lines_parallel_or_parallel :
  parallel b d ∨ parallel a c :=
  sorry

end lines_parallel_or_parallel_l126_126465


namespace chili_pepper_cost_l126_126008

theorem chili_pepper_cost :
  ∃ x : ℝ, 
    (3 * 2.50 + 4 * 1.50 + 5 * x = 18) ∧ 
    x = 0.90 :=
by
  use 0.90
  sorry

end chili_pepper_cost_l126_126008


namespace two_workers_two_hours_holes_l126_126277

theorem two_workers_two_hours_holes
    (workers1: ℝ) (holes1: ℝ) (hours1: ℝ)
    (workers2: ℝ) (hours2: ℝ)
    (h1: workers1 = 1.5)
    (h2: holes1 = 1.5)
    (h3: hours1 = 1.5)
    (h4: workers2 = 2)
    (h5: hours2 = 2)
    : (workers2 * (holes1 / (workers1 * hours1)) * hours2 = 8 / 3) := 
by {
   -- To be filled with proof, currently a placeholder.
  sorry
}

end two_workers_two_hours_holes_l126_126277


namespace eiffel_tower_height_l126_126242

-- Define the constants for heights and difference
def BurjKhalifa : ℝ := 830
def height_difference : ℝ := 506

-- The goal: Prove that the height of the Eiffel Tower is 324 m.
theorem eiffel_tower_height : BurjKhalifa - height_difference = 324 := 
by 
sorry

end eiffel_tower_height_l126_126242


namespace rectangle_area_ratio_l126_126328

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 4) (h2 : b / d = 3 / 4) :
  (a * b) / (c * d) = 9 / 16 := 
  sorry

end rectangle_area_ratio_l126_126328


namespace solution_set_of_inequality_l126_126118

open Real Set

noncomputable def f (x : ℝ) : ℝ := exp (-x) - exp x - 5 * x

theorem solution_set_of_inequality :
  { x : ℝ | f (x ^ 2) + f (-x - 6) < 0 } = Iio (-2) ∪ Ioi 3 :=
by
  sorry

end solution_set_of_inequality_l126_126118


namespace trajectory_equation_parabola_parameter_l126_126715

noncomputable def projectile_trajectory (c g : ℝ) : ℝ → ℝ :=
  λ x, (g * x^2) / (2 * c^2)

theorem trajectory_equation (c g : ℝ) (h : g > 0): 
  ∀ x y : ℝ, y = projectile_trajectory c g x ↔ x^2 = (2 * c^2 / g) * y :=
by
  intro x y
  unfold projectile_trajectory
  split
  • intro hy
    rw [hy]
    field_simp [ne_of_gt h]
    ring
  • intro hx
    rw [hx]
    field_simp [ne_of_gt h]
    ring

theorem parabola_parameter (c g : ℝ) (h : g > 0) : 
  ∃ p : ℝ, p = c^2 / g :=
by
  use c^2 / g
  rfl

end trajectory_equation_parabola_parameter_l126_126715


namespace minimum_distance_AB_l126_126234

noncomputable def min_distance_between_A_and_B : ℝ :=
  let f (x1 x2 : ℝ) : ℝ := (x2 - x1)^2 + (x2^2 - (12/5)*x1 + 5)^2
  in Real.sqrt (f (some x1) (some x2))

theorem minimum_distance_AB :
  ∃ (x1 x2 : ℝ), x2^2 = (12/5) * x1 - 5 ∧ x2 = x1^2 ∧ min_distance_between_A_and_B = 89 / 65 :=
sorry

end minimum_distance_AB_l126_126234


namespace side_of_square_is_25_l126_126314

theorem side_of_square_is_25 (area_of_circle : ℝ) (perimeter_of_square : ℝ) (h1 : area_of_circle = 100) (h2 : area_of_circle = perimeter_of_square) : perimeter_of_square / 4 = 25 :=
by {
  -- Insert the steps here if necessary.
  sorry
}

end side_of_square_is_25_l126_126314


namespace not_all_perfect_squares_l126_126238

theorem not_all_perfect_squares (k : ℕ) : ∃ i ∈ {2*k - 1, 5*k - 1, 13*k - 1}, ¬∃ n : ℕ, i = n^2 := sorry

end not_all_perfect_squares_l126_126238


namespace perpendicular_AO_BE_l126_126178

variable {A B C D E O : Type}
-- Conditions
axiom convex_pentagon (ABCDE : ConvexPoly)
axiom angle_B_right : ∠B = 90
axiom angle_E_right : ∠E = 90
axiom equal_angles : ∠BAC = ∠EAD
axiom intersect_at_O : line (BD) ∩ line (CE) = O

-- Conclusion
theorem perpendicular_AO_BE : line (AO) ⊥ line (BE) :=
sorry

end perpendicular_AO_BE_l126_126178


namespace comb_factorial_mult_result_l126_126392

theorem comb_factorial_mult_result :
  (Nat.binomial 12 4) * (Nat.factorial 4) = 11880 := by
  sorry

end comb_factorial_mult_result_l126_126392


namespace center_of_circle_l126_126257

theorem center_of_circle (x y : ℝ) : 
  (x - 1) ^ 2 + (y + 1) ^ 2 = 4 ↔ (x^2 + y^2 - 2*x + 2*y - 2 = 0) :=
sorry

end center_of_circle_l126_126257


namespace goods_train_length_l126_126011

theorem goods_train_length (v1 v2 : ℕ) (t : ℕ) (h1 : v1 = 15) (h2 : v2 = 97) (h3 : t = 9) : 
  let relative_speed_mps := (v2 + v1) * 1000 / 3600,
      length := relative_speed_mps * t
  in length = 279.99 := by
  sorry

end goods_train_length_l126_126011


namespace upstream_speed_l126_126355

-- Speed of the man in still water
def V_m : ℕ := 32

-- Speed of the man rowing downstream
def V_down : ℕ := 42

-- Speed of the stream
def V_s : ℕ := V_down - V_m

-- Speed of the man rowing upstream
def V_up : ℕ := V_m - V_s

theorem upstream_speed (V_m : ℕ) (V_down : ℕ) (V_s : ℕ) (V_up : ℕ) : 
  V_m = 32 → 
  V_down = 42 → 
  V_s = V_down - V_m → 
  V_up = V_m - V_s → 
  V_up = 22 := 
by intros; 
   repeat {sorry}

end upstream_speed_l126_126355


namespace parabola_line_2014_l126_126775

noncomputable def fib : ℕ → ℝ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem parabola_line_2014 :
  let k_2014 := fib 2015 / fib 2014 in
  ∀ x : ℝ, 
  y = (1/2) * ((1 + Real.sqrt 5)^2015 - (1 - Real.sqrt 5)^2015) / 
        ((1 + Real.sqrt 5)^2014 - (1 - Real.sqrt 5)^2014) * (x - 1/4) :=
by
  sorry

end parabola_line_2014_l126_126775


namespace solve_prime_equation_l126_126580

theorem solve_prime_equation (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) :
  x^3 + y^3 - 3 * x * y = p - 1 ↔
  (x = 1 ∧ y = 0 ∧ p = 2) ∨
  (x = 0 ∧ y = 1 ∧ p = 2) ∨
  (x = 2 ∧ y = 2 ∧ p = 5) := 
sorry

end solve_prime_equation_l126_126580


namespace roots_polynomial_l126_126202

theorem roots_polynomial :
  let p q : ℝ 
  in (x^2 - 5*x + 6 = 0)
  (p^4 + p^3 * q^2 + p^2 * q^3 + q^4 = 241) := by
  sorry

end roots_polynomial_l126_126202
